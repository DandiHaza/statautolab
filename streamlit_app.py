from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.analysis_runner import build_run_context, execute_analysis
from app.config import DEFAULT_SETTINGS, resolve_settings
from app.io import SUPPORTED_EXTENSIONS, load_dataset


UPLOAD_DIR = Path(".streamlit_uploads")
NO_TARGET = "선택 안 함"

TASK_TYPE_LABELS = {
    "auto": "자동 판별",
    "regression": "회귀",
    "classification": "분류",
}

REPORT_FORMAT_LABELS = {
    "md": "Markdown",
    "html": "HTML",
}

EVAL_METHOD_LABELS = {
    "holdout": "홀드아웃",
    "cv": "교차검증",
}


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def collect_download_files(output_dir: Path) -> list[Path]:
    return [path for path in sorted(output_dir.rglob("*")) if path.is_file()]


def strip_markdown_images(markdown_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def localize_profile_dtypes(df):
    return df.rename(
        columns={
            "column": "컬럼명",
            "dtype": "데이터 타입",
            "non_null_count": "비결측 개수",
            "unique_count": "고유값 개수",
        }
    )


def localize_missing_summary(df):
    return df.rename(
        columns={
            "column": "컬럼명",
            "missing_count": "결측치 개수",
            "missing_ratio_pct": "결측치 비율(%)",
        }
    )


def localize_outlier_summary(df):
    return df.rename(
        columns={
            "column": "컬럼명",
            "outlier_count": "이상치 개수",
            "outlier_ratio_pct": "이상치 비율(%)",
            "lower_bound": "하한",
            "upper_bound": "상한",
        }
    )


def localize_model_metrics(df):
    return df.rename(
        columns={
            "model": "모델",
            "problem_type": "문제 유형",
            "evaluated_folds": "평가 fold 수",
            "rmse": "RMSE",
            "rmse_std": "RMSE 표준편차",
            "mae": "MAE",
            "mae_std": "MAE 표준편차",
            "r2": "R2",
            "r2_std": "R2 표준편차",
            "accuracy": "정확도",
            "accuracy_std": "정확도 표준편차",
            "f1": "F1",
            "f1_std": "F1 표준편차",
            "roc_auc": "ROC-AUC",
            "roc_auc_std": "ROC-AUC 표준편차",
        }
    )


def render_chart_gallery(charts_dir: Path) -> None:
    image_paths = sorted(charts_dir.glob("*.png"))
    if not image_paths:
        return

    st.markdown("### 생성된 차트")
    grouped_images = {
        "히스토그램": [path for path in image_paths if path.stem.startswith("histogram_")],
        "박스플롯": [path for path in image_paths if path.stem.startswith("boxplot_")],
        "상관행렬": [path for path in image_paths if path.stem == "correlation_matrix"],
    }

    for section_title, section_images in grouped_images.items():
        if not section_images:
            continue
        st.markdown(f"#### {section_title}")
        columns = st.columns(2)
        for index, image_path in enumerate(section_images):
            with columns[index % 2]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _valid_feature_options(all_columns: list[str], target_value: str) -> list[str]:
    if target_value == NO_TARGET:
        return all_columns
    return [column for column in all_columns if column != target_value]


def _sync_selected_features(all_columns: list[str]) -> None:
    target_value = st.session_state.get("target_option", NO_TARGET)
    valid_options = _valid_feature_options(all_columns, target_value)
    selected = st.session_state.get("selected_features", [])
    selected = [column for column in selected if column in valid_options]

    if not selected:
        selected = valid_options.copy()

    st.session_state["selected_features"] = selected


def _on_target_change(all_columns: list[str]) -> None:
    _sync_selected_features(all_columns)


def render_result_summary(result) -> None:
    profile = result.profile

    st.subheader("결과 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("행 개수", f"{profile.row_count:,}")
    col2.metric("열 개수", f"{profile.column_count:,}")
    col3.metric("경고 수", str(len(result.warnings)))

    if result.model_result is not None:
        selected_features = result.model_result.preprocessing_summary.selected_feature_columns
        st.markdown("### 모델 입력 변수")
        st.write(", ".join(selected_features) if selected_features else "없음")

    st.markdown("### 데이터 개요")
    st.dataframe(localize_profile_dtypes(profile.dtypes), use_container_width=True)

    st.markdown("### 결측치 요약")
    missing_df = profile.missing[profile.missing["missing_count"] > 0].head(10)
    if missing_df.empty:
        st.info("결측치가 있는 컬럼이 없습니다.")
    else:
        st.dataframe(localize_missing_summary(missing_df), use_container_width=True)

    st.markdown("### 이상치 요약")
    outlier_df = profile.outliers[profile.outliers["outlier_count"] > 0].head(10)
    if outlier_df.empty:
        st.info("IQR 기준으로 뚜렷한 이상치가 많은 컬럼은 없습니다.")
    else:
        st.dataframe(localize_outlier_summary(outlier_df), use_container_width=True)

    if result.model_result is not None:
        st.markdown("### 모델 비교")
        st.dataframe(localize_model_metrics(result.model_result.metrics), use_container_width=True)
        st.success(f"가장 좋은 모델: {result.model_result.best_model_name}")

    if result.warnings:
        st.markdown("### 주의사항 및 경고")
        for record in result.warnings:
            st.warning(record.message)

    render_chart_gallery(result.charts_dir)

    st.markdown("### 리포트 미리보기")
    report_text = result.report_path.read_text(encoding="utf-8-sig")
    if result.report_path.suffix.lower() == ".html":
        st.info("HTML 리포트는 로컬 이미지 경로 때문에 미리보기 대신 파일 다운로드로 확인하는 편이 더 정확합니다.")
        st.code(report_text[:4000], language="html")
    else:
        st.markdown(strip_markdown_images(report_text))

    st.markdown("### 결과 파일 다운로드")
    for file_path in collect_download_files(result.context.output_dir):
        mime = "application/octet-stream"
        if file_path.suffix.lower() in {".md", ".txt", ".json", ".csv", ".html"}:
            mime = "text/plain"
        with file_path.open("rb") as file_handle:
            st.download_button(
                label=f"{file_path.relative_to(result.context.output_dir).as_posix()} 다운로드",
                data=file_handle.read(),
                file_name=file_path.name,
                mime=mime,
                key=f"download-{file_path.as_posix()}",
            )


def main() -> None:
    st.set_page_config(page_title="StatAutoLab", layout="wide")
    st.title("StatAutoLab")
    st.caption("CSV/XLSX 업로드 후 EDA와 baseline 모델 비교를 웹에서 실행합니다.")

    st.markdown("## 1. 데이터 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 선택하세요.",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
    )

    if uploaded_file is None:
        st.info("분석할 CSV/XLSX 파일을 업로드하면 미리보기와 실행 옵션이 표시됩니다.")
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        st.error("지원하지 않는 파일 형식입니다. CSV, XLSX, XLS 파일만 업로드할 수 있습니다.")
        return

    try:
        saved_input_path = save_uploaded_file(uploaded_file)
        preview_df = load_dataset(saved_input_path)
    except Exception as exc:
        st.error(f"파일을 읽는 중 문제가 발생했습니다. 파일 형식과 내용을 확인해 주세요. 상세 오류: {exc}")
        return

    st.success(f"업로드 완료: {uploaded_file.name}")
    st.dataframe(preview_df.head(20), use_container_width=True)

    all_columns = preview_df.columns.astype(str).tolist()
    target_options = [NO_TARGET] + all_columns

    if "target_option" not in st.session_state or st.session_state["target_option"] not in target_options:
        st.session_state["target_option"] = NO_TARGET
    if "selected_features" not in st.session_state:
        st.session_state["selected_features"] = all_columns.copy()

    _sync_selected_features(all_columns)

    st.markdown("## 2. 분석 옵션")
    left, right = st.columns(2)
    with left:
        st.selectbox(
            "종속변수(타깃 컬럼)",
            target_options,
            key="target_option",
            on_change=_on_target_change,
            args=(all_columns,),
        )
        valid_feature_options = _valid_feature_options(all_columns, st.session_state["target_option"])
        st.multiselect(
            "독립변수(모델 입력 컬럼)",
            options=valid_feature_options,
            key="selected_features",
            help="종속변수로 선택한 컬럼은 자동으로 제외됩니다.",
        )
        task_type = st.selectbox(
            "문제 유형",
            options=list(TASK_TYPE_LABELS.keys()),
            index=0,
            format_func=lambda value: TASK_TYPE_LABELS[value],
        )
        report_format = st.selectbox(
            "리포트 형식",
            options=list(REPORT_FORMAT_LABELS.keys()),
            index=0,
            format_func=lambda value: REPORT_FORMAT_LABELS[value],
        )
    with right:
        eval_method = st.selectbox(
            "평가 방식",
            options=list(EVAL_METHOD_LABELS.keys()),
            index=0,
            format_func=lambda value: EVAL_METHOD_LABELS[value],
        )
        cv_folds = st.number_input("교차검증 fold 수", min_value=2, max_value=10, value=int(DEFAULT_SETTINGS["cv_folds"]), step=1)
        output_dir = st.text_input("출력 폴더", value=str(DEFAULT_SETTINGS["output_dir"]))

    submit = st.button("분석 실행", use_container_width=True)
    if not submit:
        return

    target_value = st.session_state["target_option"]
    feature_columns = st.session_state["selected_features"] or None

    cli_values = {
        "input_path": str(saved_input_path),
        "target": None if target_value == NO_TARGET else target_value,
        "feature_columns": feature_columns,
        "output_dir": output_dir.strip() or str(DEFAULT_SETTINGS["output_dir"]),
        "report_format": report_format,
        "task_type": task_type,
        "random_state": DEFAULT_SETTINGS["random_state"],
        "test_size": DEFAULT_SETTINGS["test_size"],
        "eval_method": eval_method,
        "cv_folds": int(cv_folds),
    }

    try:
        settings = resolve_settings(cli_values, {})
        context = build_run_context(settings)
        with st.spinner("분석을 실행하는 중입니다. 데이터 크기에 따라 시간이 걸릴 수 있습니다."):
            result = execute_analysis(context)
    except Exception as exc:
        st.error(
            "분석 실행에 실패했습니다. 입력 파일, 종속변수, 독립변수 목록, 평가 옵션을 다시 확인해 주세요. "
            f"상세 오류: {exc}"
        )
        return

    st.success(f"분석이 완료되었습니다. 결과 폴더: {result.context.output_dir}")
    render_result_summary(result)


if __name__ == "__main__":
    main()
