"""LangGraph skeleton with mock model runner and MLflow logging.

This keeps the model execution stubbed; swap `run_model` with real loader later.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

import langgraph
# from langchain_openai import ChatOpenAI  # leaving for later switch-back
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from mlflow import MlflowClient
import mlflow
from mlflow.exceptions import MlflowException

from .state import DEFAULT_STATE, PipelineState

PROMPT_DIR = Path(__file__).parent / "prompts"


def _normalize_prompt(raw: Any) -> str:
    """Return a plain string prompt from various prompt objects."""
    if isinstance(raw, str):
        return raw
    template = getattr(raw, "template", None)
    if template:
        return template
    definition = getattr(raw, "definition", None)
    if definition:
        if hasattr(definition, "template") and definition.template:
            return definition.template
        if isinstance(definition, dict):
            for key in ("template", "text", "prompt"):
                if key in definition:
                    return definition[key]
    if isinstance(raw, dict):
        for key in ("template", "text", "prompt"):
            if key in raw:
                return raw[key]
    return str(raw)


def load_prompt(name: str) -> str:
    """Load prompt text from local file or MLflow prompt registry."""
    path = PROMPT_DIR / name
    if path.exists():
        return path.read_text().strip()
    try:
        prompt_version = mlflow.genai.load_prompt(name)
        return _normalize_prompt(prompt_version)
    except Exception:
        return _normalize_prompt(name)

@mlflow.trace
def configure_mlflow() -> None:
    """Ensure MLflow has a usable tracking URI and experiment.

    Default to sqlite backend to avoid impending file store deprecation.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "ad-astrAI")
    try:
        mlflow.set_experiment(experiment)
    except MlflowException:
        # If remote server denies, fall back to local sqlite store.
        if tracking_uri != "sqlite:///mlflow.db":
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment(experiment)

@mlflow.trace
def log_mlflow(event: str, state: PipelineState, run_id: str) -> None:
    mlflow_client = MlflowClient()
    mlflow_client.log_text(run_id, json.dumps({"event": event, "state": state}, default=str), f"events/{event}.json")

@mlflow.trace
def orchestrate(state: PipelineState) -> PipelineState:
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Prefer the latest stable model ID returned by model_catalog.ipynb.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = load_prompt("orchestrator_prompt.txt")
    resp = llm.invoke([
        ("system", prompt),
        ("human", json.dumps({"file": state.get("input_path", "")}, default=str)),
    ])
    try:
        parsed = json.loads(resp.content)
    except Exception as exc:  # defensive fallback
        state.setdefault("errors", []).append(f"orchestrator_parse_error: {exc}")
        file = state.get("input_path", "")
        if file.endswith((".fits", ".csv")):
            parsed = {"route": "spectral", "model": "mock-spectra-v1", "note": "fallback route: spectral"}
        elif file.endswith((".png", ".jpg", ".jpeg")):
            parsed = {"route": "image", "model": "mock-image-v1", "note": "fallback route: image"}
        else:
            parsed = {"route": "error", "model": "mock-unknown", "note": "unsupported file type"}

    state["route"] = parsed.get("route", "error")
    state["model_name"] = parsed.get("model", "mock-spectra-v1")
    state.setdefault("metadata", {})["orchestrator_note"] = parsed.get("note", "")
    return state

@mlflow.trace
def preprocess(state: PipelineState) -> PipelineState:
    # Placeholder normalization.
    state.setdefault("metadata", {})["preprocess"] = "done"
    return state

@mlflow.trace
def run_model(state: PipelineState) -> PipelineState:
    # Mocked model output. Replace with real model call later.
    state["predictions"] = [
        {"element": "H", "probability": 0.72, "rationale": "Strong Balmer lines"},
        {"element": "He", "probability": 0.48, "rationale": "Helium lines present"},
    ]
    return state

@mlflow.trace
def report(state: PipelineState) -> PipelineState:
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = load_prompt("reporter_prompt.txt")
    summary = llm.invoke([
        ("system", prompt),
        ("human", json.dumps({
            "modality": state.get("modality"),
            "model": state.get("model_name"),
            "predictions": state.get("predictions", []),
            "mlflow_run_uri": state.get("log_refs", {}).get("mlflow", ""),
        }, default=str)),
    ])
    state.setdefault("metadata", {})["report"] = summary.content
    return state

@mlflow.trace
def build_graph() -> Callable[[PipelineState], PipelineState]:
    graph = StateGraph(PipelineState)

    graph.add_node("orchestrate", orchestrate)
    graph.add_node("preprocess", preprocess)
    graph.add_node("run_model", run_model)
    graph.add_node("report", report)

    graph.set_entry_point("orchestrate")
    graph.add_edge("orchestrate", "preprocess")
    graph.add_edge("preprocess", "run_model")
    graph.add_edge("run_model", "report")
    graph.add_edge("report", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    def runner(initial: PipelineState) -> PipelineState:
        configure_mlflow()
        nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name="agent-run", nested=nested) as run:
            initial = {**DEFAULT_STATE, **initial}
            initial.setdefault("log_refs", {})["mlflow"] = run.info.run_id
            log_mlflow("start", initial, run.info.run_id)
            config = {"configurable": {"thread_id": initial.get("input_path", "run")}}
            final_state = compiled.invoke(initial, config=config)
            log_mlflow("end", final_state, run.info.run_id)
            return final_state

    return runner


RUN_GRAPH = build_graph()


if __name__ == "__main__":  # manual smoke test
    result = RUN_GRAPH({"input_path": "example.fits", "modality": "spectral"})
    print(json.dumps(result, indent=2))
