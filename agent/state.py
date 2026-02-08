"""Shared state definition for LangGraph pipeline.

This stays small on purpose; downstream nodes can append to the dict.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


Modality = Literal["spectral", "image"]


class Prediction(TypedDict, total=False):
    element: str
    probability: float
    rationale: str


class PipelineState(TypedDict, total=False):
    modality: Modality
    input_path: str
    metadata: Dict[str, Any]
    model_name: str
    predictions: List[Prediction]
    log_refs: Dict[str, str]
    errors: List[str]
    prompt_version: str


DEFAULT_STATE: PipelineState = {
    "metadata": {},
    "predictions": [],
    "log_refs": {},
    "errors": [],
    "prompt_version": "v1",
}

