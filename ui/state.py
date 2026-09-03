"""세션 상태와 세션별 작업 경로 관리."""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from ui.constants import NO_TARGET, OUTPUT_ROOT, SAMPLE_DATASETS, UPLOAD_DIR


def get_session_id() -> str:
    """Stable per-browser-session id. Concurrent visitors must not share working paths."""
    session_id = st.session_state.get("session_id")
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]
        st.session_state["session_id"] = session_id
    return session_id


def session_upload_dir() -> Path:
    return UPLOAD_DIR / get_session_id()


def session_output_dir() -> Path:
    return OUTPUT_ROOT / get_session_id()


def save_uploaded_file(uploaded_file) -> Path:
    # Two visitors uploading the same file name would otherwise overwrite each other,
    # and the first one's analysis would silently run on the second one's data.
    upload_dir = session_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def clear_analysis_state() -> None:
    for key in ("analysis_result", "analysis_df", "analysis_mode", "analysis_file_key"):
        st.session_state.pop(key, None)


def current_file_key(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def get_sample_dataset(key: str | None) -> dict | None:
    if key is None:
        return None
    return next((sample for sample in SAMPLE_DATASETS if sample["key"] == key), None)


def select_sample_dataset(sample: dict) -> None:
    st.session_state["sample_dataset_key"] = sample["key"]
    # Pre-fill the target so the visitor can press 분석 실행 without further setup.
    st.session_state["target_option"] = sample["target"] or NO_TARGET
    st.session_state.pop("selected_features", None)
    clear_analysis_state()


def remove_feature_from_selection(feature: str) -> None:
    st.session_state["pending_remove_feature"] = feature


def _valid_feature_options(all_columns: list[str], target_value: str) -> list[str]:
    if target_value == NO_TARGET:
        return all_columns
    return [column for column in all_columns if column != target_value]


def _sanitize_selected_features(valid_options: list[str]) -> list[str]:
    selected = st.session_state.get("selected_features")
    if selected is None:
        return valid_options.copy()
    sanitized = [column for column in selected if column in valid_options]
    return sanitized or valid_options.copy()
