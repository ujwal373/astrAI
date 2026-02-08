"""Simple Streamlit UI for the multi-agent pipeline.

Uploads a data file, routes it through the LangGraph pipeline, and shows
the summary and predictions. Detailed logs remain in MLflow.
"""

import os
from pathlib import Path
import tempfile

import streamlit as st
import mlflow
from dotenv import load_dotenv, find_dotenv

from agent import run_pipeline
from agent.graph import configure_mlflow


load_dotenv(find_dotenv(usecwd=True))


st.set_page_config(page_title="ad-astrAI", layout="centered")
st.title("ad-astrAI: Element Detector")

@mlflow.trace
def infer_modality(filename: str) -> str:
    name = filename.lower()
    if name.endswith((".fits", ".csv")):
        return "spectral"
    if name.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    return "spectral"


with st.sidebar:
    st.header("Run Settings")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    st.write(f"MLflow URI: `{mlflow_uri}`")
    key_status = {
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
    }
    st.write("API keys loaded:")
    st.write({k: "✔" if v else "✖" for k, v in key_status.items()})


uploaded = st.file_uploader(
    "Upload spectral (.fits/.csv) or image (.png/.jpg) data",
    type=["fits", "csv", "png", "jpg", "jpeg"],
)

if uploaded:
    modality = infer_modality(uploaded.name)
    st.caption(f"Detected modality: {modality}")
    if st.button("Run analysis"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Running agents..."):
                configure_mlflow()
                with mlflow.start_run(run_name="ui-session") as ui_run:
                    mlflow.log_param("source", "streamlit")
                    mlflow.log_param("uploaded_filename", uploaded.name)
                    mlflow.log_param("modality", modality)
                    mlflow.log_param("upload_size_bytes", len(uploaded.getbuffer()))
                    result = run_pipeline(tmp_path, modality=modality)
                    # Link child run ID from pipeline
                    child_run = result.get("log_refs", {}).get("mlflow")
                    if child_run:
                        mlflow.log_param("child_run_id", child_run)
                    mlflow.log_text(str(result), "ui/result_snapshot.json")
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(f"Run failed: {exc}")
        else:
            st.success("Done")

            report = result.get("metadata", {}).get("report", "")
            if report:
                st.subheader("Summary")
                st.write(report)

            preds = result.get("predictions", [])
            if preds:
                st.subheader("Predictions")
                st.dataframe(preds)

            mlflow_run = result.get("log_refs", {}).get("mlflow")
            if mlflow_run:
                st.info(f"MLflow run ID (pipeline): {mlflow_run}")

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
else:
    st.info("Upload a file to start.")
