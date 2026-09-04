"""StatAutoLab 웹 UI 진입점.

화면 구성 요소는 ui 패키지에, 분석 로직은 app 패키지에 있습니다.
이 파일은 두 계층을 이어 붙이는 흐름만 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.analysis_runner import build_run_context, execute_analysis
from app.config import DEFAULT_SETTINGS, resolve_settings
from app.io import SUPPORTED_EXTENSIONS
from ui.analysis import get_available_models, infer_problem_type
from ui.constants import (
    APP_MODE_LABELS,
    EVAL_METHOD_LABELS,
    MODEL_LABELS,
    NO_MODEL,
    NO_TARGET,
    REPORT_FORMAT_LABELS,
    TASK_TYPE_LABELS,
)
from ui.data_view import (
    render_feature_selection_feedback,
    render_sample_picker,
    render_uploaded_data_preview,
)
from ui.result_view import render_result_panel, render_saved_result
from ui.state import (
    _sanitize_selected_features,
    _valid_feature_options,
    clear_analysis_state,
    current_file_key,
    get_sample_dataset,
    get_session_id,
    save_uploaded_file,
    session_output_dir,
)
from ui.analysis import load_preview_dataset


def main() -> None:
    st.set_page_config(page_title="StatAutoLab", layout="wide")
    st.title("StatAutoLab")
    st.caption("파일을 올리면 바로 EDA를 보여주고, 이후 분석 또는 예측 결과를 확인할 수 있습니다.")

    # Establish the session id up front so every path below writes to isolated folders.
    get_session_id()

    st.markdown("## 1. 데이터 선택")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 선택하세요.",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
    )

    active_sample = get_sample_dataset(st.session_state.get("sample_dataset_key"))
    if uploaded_file is not None and active_sample is not None:
        # An explicit upload always wins over a previously chosen sample.
        st.session_state.pop("sample_dataset_key", None)
        active_sample = None

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            st.error("지원하지 않는 파일 형식입니다. CSV, XLSX, XLS 파일만 업로드할 수 있습니다.")
            return

        file_key = current_file_key(uploaded_file)
        source_label = f"업로드 완료: {uploaded_file.name}"
        try:
            saved_input_path = save_uploaded_file(uploaded_file)
        except Exception as exc:
            clear_analysis_state()
            st.error(f"파일을 저장하는 중 문제가 발생했습니다. 상세 오류: {exc}")
            return
    elif active_sample is not None:
        file_key = f"sample:{active_sample['key']}"
        source_label = f"예제 데이터 사용 중: {active_sample['label']}"
        saved_input_path = active_sample["path"]
    else:
        clear_analysis_state()
        st.info("CSV/XLSX 파일을 업로드하면 데이터 미리보기와 분석 옵션이 표시됩니다.")
        render_sample_picker()
        return

    if st.session_state.get("analysis_file_key") not in (None, file_key):
        clear_analysis_state()

    try:
        preview_df = load_preview_dataset(str(saved_input_path))
    except Exception as exc:
        clear_analysis_state()
        st.error(f"파일을 읽는 중 문제가 발생했습니다. 파일 형식과 내용을 확인해 주세요. 상세 오류: {exc}")
        return

    st.session_state["analysis_file_key"] = file_key
    st.success(source_label)
    if active_sample is not None and st.button("다른 데이터 선택", key="clear-sample"):
        st.session_state.pop("sample_dataset_key", None)
        clear_analysis_state()
        st.rerun()

    render_uploaded_data_preview(preview_df, str(saved_input_path))

    all_columns = preview_df.columns.astype(str).tolist()
    target_options = [NO_TARGET] + all_columns

    if st.session_state.get("target_option") not in target_options:
        st.session_state["target_option"] = NO_TARGET

    pending_remove_feature = st.session_state.pop("pending_remove_feature", None)
    if pending_remove_feature is not None:
        selected = st.session_state.get("selected_features", [])
        if pending_remove_feature in selected:
            st.session_state["selected_features"] = [column for column in selected if column != pending_remove_feature]

    st.markdown("## 2. 사용 모드")
    app_mode = st.radio(
        "무슨 목적에 더 가깝나요?",
        options=["analysis", "prediction"],
        horizontal=True,
        format_func=lambda value: APP_MODE_LABELS[value],
    )
    if app_mode == "analysis":
        st.info("분석 모드는 회귀식, 계수, p-value 같은 해석 결과 중심으로 보여줍니다.")
    else:
        st.info("예측 모드는 예측 모델과 성능 결과 중심으로 보여줍니다.")

    st.markdown("## 3. 분석 옵션")
    left, right = st.columns(2)
    with left:
        target_value = st.selectbox(
            "종속변수 / 타깃 컬럼 (Target Column)",
            target_options,
            key="target_option",
        )
        valid_feature_options = _valid_feature_options(all_columns, target_value)
        st.session_state["selected_features"] = _sanitize_selected_features(valid_feature_options)
        st.multiselect(
            "독립변수 / 모델 입력 컬럼",
            options=valid_feature_options,
            key="selected_features",
            help="타깃 컬럼은 독립변수 목록에서 자동으로 제외됩니다.",
        )
        selected_features = st.session_state["selected_features"] or []
        if selected_features:
            render_feature_selection_feedback(preview_df, selected_features)
    with right:
        task_type = st.selectbox(
            "문제 유형",
            options=list(TASK_TYPE_LABELS.keys()),
            index=0,
            format_func=lambda value: TASK_TYPE_LABELS[value],
        )
        resolved_problem_type = infer_problem_type(preview_df, target_value, task_type)
        available_models = get_available_models(resolved_problem_type, app_mode)
        if available_models:
            model_options = available_models if app_mode == "analysis" else [NO_MODEL] + available_models
            selected_model = st.selectbox(
                "사용할 모델",
                options=model_options,
                format_func=lambda value: "자동 선택" if value == NO_MODEL else MODEL_LABELS.get(value, value),
            )
        else:
            selected_model = NO_MODEL
            st.caption("타깃 컬럼을 선택하면 사용할 수 있는 모델이 표시됩니다.")
        report_format = st.selectbox(
            "리포트 형식",
            options=list(REPORT_FORMAT_LABELS.keys()),
            index=0,
            format_func=lambda value: REPORT_FORMAT_LABELS[value],
        )
        if app_mode == "analysis":
            st.caption("분석 모드는 해석 중심 결과를 우선 보여주며, 성능 비교는 화면에서 최소화합니다.")

    with st.expander("고급 옵션 (평가 방식·시드)"):
        st.caption("기본값 그대로 두어도 됩니다. CLI의 평가 옵션과 동일한 설정입니다.")
        adv_left, adv_right = st.columns(2)
        with adv_left:
            eval_method = st.selectbox(
                "평가 방식",
                options=list(EVAL_METHOD_LABELS.keys()),
                index=0,
                format_func=lambda value: EVAL_METHOD_LABELS[value],
                key="eval_method",
                help="홀드아웃은 한 번 나눠 빠르게, 교차검증은 여러 번 나눠 더 안정적으로 성능을 추정합니다.",
            )
            random_state = st.number_input(
                "랜덤 시드",
                min_value=0,
                max_value=9999,
                value=int(DEFAULT_SETTINGS["random_state"]),
                step=1,
                key="random_state",
                help="같은 시드를 쓰면 같은 결과가 재현됩니다.",
            )
        with adv_right:
            if eval_method == "cv":
                cv_folds = st.slider(
                    "교차검증 fold 수",
                    min_value=2,
                    max_value=10,
                    value=int(DEFAULT_SETTINGS["cv_folds"]),
                    key="cv_folds",
                )
                test_size = float(DEFAULT_SETTINGS["test_size"])
                st.caption("데이터를 fold 수만큼 나눠 번갈아 검증하고 평균과 표준편차를 함께 보고합니다.")
            else:
                test_size = st.slider(
                    "검증 데이터 비율",
                    min_value=0.1,
                    max_value=0.5,
                    value=float(DEFAULT_SETTINGS["test_size"]),
                    step=0.05,
                    key="test_size",
                )
                cv_folds = int(DEFAULT_SETTINGS["cv_folds"])
                st.caption(f"전체의 {test_size:.0%}를 검증용으로 떼어 두고 나머지로 학습합니다.")

            tune = st.checkbox(
                "하이퍼파라미터 탐색",
                value=bool(DEFAULT_SETTINGS["tune"]),
                key="tune",
                help="모델별로 후보 파라미터를 격자 탐색합니다. 성능이 조금 오르지만 실행 시간이 늘어납니다.",
            )
            if tune:
                st.caption("탐색 때문에 실행이 느려집니다. 교차검증과 함께 쓰면 특히 오래 걸립니다.")

    just_rendered_result = False

    if st.button("분석 실행", key="run_analysis", width="stretch"):
        feature_columns = st.session_state["selected_features"] or None
        cli_values = {
            "input_path": str(saved_input_path),
            "target": None if target_value == NO_TARGET else target_value,
            "feature_columns": feature_columns,
            "selected_model": None if selected_model == NO_MODEL else selected_model,
            "output_dir": str(session_output_dir()),
            "report_format": report_format,
            "task_type": task_type,
            "random_state": int(random_state),
            "test_size": float(test_size),
            "eval_method": eval_method,
            "cv_folds": int(cv_folds),
            "tune": bool(tune),
        }

        try:
            settings = resolve_settings(cli_values, {})
            context = build_run_context(settings)
            with st.spinner("분석을 실행하는 중입니다. 데이터 크기에 따라 시간이 걸릴 수 있습니다."):
                result = execute_analysis(context)
        except Exception as exc:
            clear_analysis_state()
            st.error(
                "분석 실행에 실패했습니다. 입력 파일, 타깃 컬럼, 독립변수 목록, 설정값을 다시 확인해 주세요. "
                f"상세 오류: {exc}"
            )
        else:
            st.session_state["analysis_result"] = result
            st.session_state["analysis_df"] = preview_df
            st.session_state["analysis_mode"] = app_mode
            st.session_state["analysis_file_key"] = file_key
            render_result_panel(result, preview_df, app_mode)
            just_rendered_result = True

    if not just_rendered_result:
        render_saved_result()


if __name__ == "__main__":
    main()
