from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app.analysis_runner import AnalysisRunResult, build_run_context, execute_analysis
from app.config import DEFAULT_SETTINGS, resolve_settings
from app.io import SUPPORTED_EXTENSIONS, load_dataset
from app.profiling import profile_dataset
from app.regression_insights import build_regression_dashboard_data


UPLOAD_DIR = Path(".streamlit_uploads")
NO_TARGET = "선택 안 함"
NO_MODEL = "자동 선택"

APP_MODE_LABELS = {
    "analysis": "분석 모드",
    "prediction": "예측 모드",
}

TASK_TYPE_LABELS = {
    "auto": "자동 판별",
    "regression": "회귀",
    "classification": "분류",
}

REPORT_FORMAT_LABELS = {
    "md": "Markdown",
    "html": "HTML",
}

MODEL_LABELS = {
    "LinearRegression": "선형회귀 (LinearRegression)",
    "RandomForestRegressor": "랜덤포레스트 회귀 (RandomForestRegressor)",
    "LogisticRegression": "로지스틱회귀 (LogisticRegression)",
    "RandomForestClassifier": "랜덤포레스트 분류 (RandomForestClassifier)",
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


def clear_analysis_state() -> None:
    for key in ("analysis_result", "analysis_df", "analysis_mode", "analysis_file_key"):
        st.session_state.pop(key, None)


def current_file_key(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def localize_profile_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "column": "컬럼명 (Column)",
            "dtype": "데이터 타입 (Dtype)",
            "non_null_count": "비결측 개수 (Non-null Count)",
            "unique_count": "고유값 개수 (Unique Count)",
        }
    )


def localize_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "column": "컬럼명 (Column)",
            "missing_count": "결측치 개수 (Missing Count)",
            "missing_ratio_pct": "결측치 비율 (Missing Ratio %)",
        }
    )


def localize_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "column": "컬럼명 (Column)",
            "outlier_count": "이상치 개수 (Outlier Count)",
            "outlier_ratio_pct": "이상치 비율 (Outlier Ratio %)",
            "lower_bound": "하한 (Lower Bound)",
            "upper_bound": "상한 (Upper Bound)",
        }
    )


def localize_combined_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.rename(
        columns={
            "feature": "변수 (Feature)",
            "coefficient": "계수",
            "std_error": "Std Err",
            "t_value": "t",
            "p_value": "p-value",
            "ci_lower": "0.025",
            "ci_upper": "0.975",
        }
    ).copy()
    if "section" in localized.columns:
        localized = localized.drop(columns=["section"])
    if "변수 (Feature)" in localized.columns:
        localized["변수 (Feature)"] = localized["변수 (Feature)"].replace({"const": "절편 (Intercept)"})
    return localized


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


def infer_problem_type(df: pd.DataFrame, target_value: str, task_type: str) -> str | None:
    if target_value == NO_TARGET:
        return None
    if task_type != "auto":
        return task_type
    return "regression" if pd.api.types.is_numeric_dtype(df[target_value]) else "classification"


def get_available_models(problem_type: str | None, app_mode: str) -> list[str]:
    if problem_type == "regression":
        if app_mode == "analysis":
            return ["LinearRegression"]
        return ["LinearRegression", "RandomForestRegressor"]
    if problem_type == "classification":
        if app_mode == "analysis":
            return ["LogisticRegression"]
        return ["LogisticRegression", "RandomForestClassifier"]
    return []


def build_ols_display_text(dashboard_data) -> str | None:
    if dashboard_data.ols_summary_text:
        return f"=== 최종 회귀 분석 결과 ===\n\n{dashboard_data.ols_summary_text}"
    return None


def render_uploaded_data_preview(df: pd.DataFrame) -> None:
    profile = profile_dataset(df)

    st.markdown("## 데이터 파일 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("## 데이터 개요")
    st.caption(f"행: {profile.row_count}개 / 열: {profile.column_count}개")
    st.dataframe(localize_profile_dtypes(profile.dtypes), use_container_width=True)

    st.markdown("## 결측치 요약")
    missing_df = profile.missing[profile.missing["missing_count"] > 0].head(10)
    if missing_df.empty:
        st.info("결측치가 있는 컬럼이 없습니다.")
    else:
        st.dataframe(localize_missing_summary(missing_df), use_container_width=True)

    st.markdown("## 이상치 요약")
    outlier_df = profile.outliers[profile.outliers["outlier_count"] > 0].head(10)
    if outlier_df.empty:
        st.info("IQR 기준으로 탐지된 주요 이상치 컬럼이 없습니다.")
    else:
        st.dataframe(localize_outlier_summary(outlier_df), use_container_width=True)

    render_inline_charts(df, profile.correlation)


def render_inline_charts(df: pd.DataFrame, correlation_df: pd.DataFrame) -> None:
    st.markdown("## 차트")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.info("수치형 컬럼이 없어 차트를 만들지 않았습니다.")
        return

    st.markdown("### 히스토그램")
    hist_cols = st.columns(2)
    for index, column in enumerate(numeric_columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        ax.set_title(column)
        with hist_cols[index % 2]:
            st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.markdown("### 박스플롯")
    box_cols = st.columns(2)
    for index, column in enumerate(numeric_columns):
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[column].dropna(), ax=ax)
        ax.set_title(column)
        with box_cols[index % 2]:
            st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    if not correlation_df.empty:
        st.markdown("### 상관행렬")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(correlation_df, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def render_result_extras(result: AnalysisRunResult) -> None:
    if st.toggle("리포트 미리보기", value=False, key="toggle_report_preview"):
        report_text = result.report_path.read_text(encoding="utf-8-sig")
        if result.report_path.suffix.lower() == ".html":
            st.info("HTML 리포트는 다운로드 후 브라우저에서 보는 편이 더 정확합니다.")
            st.code(report_text[:4000], language="html")
        else:
            st.markdown(strip_markdown_images(report_text))

    if st.toggle("결과 파일 다운로드", value=False, key="toggle_downloads"):
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


def render_regression_dashboard(result: AnalysisRunResult, source_df: pd.DataFrame, app_mode: str) -> None:
    if result.model_result is None or result.model_result.problem_type != "regression":
        return

    if app_mode != "analysis":
        st.markdown("## 예측 결과")
        metrics_row = result.model_result.metrics.iloc[0].to_dict()
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("RMSE", f"{float(metrics_row.get('rmse', 0.0)):.4f}")
        metric_col2.metric("MAE", f"{float(metrics_row.get('mae', 0.0)):.4f}")
        metric_col3.metric("R2", f"{float(metrics_row.get('r2', 0.0)):.4f}")
        st.info("예측 모드에서는 성능 중심으로 결과를 보여줍니다.")
        return

    if result.model_result.best_model_name != "LinearRegression":
        st.markdown("## 회귀 분석 대시보드")
        st.info("전형적인 회귀분석 해석은 선형회귀 (LinearRegression) 선택 시 제공합니다.")
        return

    dashboard_data = build_regression_dashboard_data(source_df, result.model_result)
    if dashboard_data is None:
        return

    st.markdown("## 회귀 분석 대시보드")

    ols_text = build_ols_display_text(dashboard_data)
    if ols_text:
        st.code(ols_text, language="text")

    st.markdown("### 쉬운 해석")
    overview = dashboard_data.ols_overview
    prob_f = overview.get("prob_f_statistic") if overview else None
    f_stat = overview.get("f_statistic") if overview else None
    adj_r2 = overview.get("adj_r_squared") if overview else None

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("R2", f"{overview['r_squared']:.4f}" if overview and overview.get("r_squared") is not None else "N/A")
    metric_col2.metric("Adj. R2", f"{adj_r2:.4f}" if adj_r2 is not None else "N/A")
    metric_col3.metric("F-statistic", f"{f_stat:.4f}" if f_stat is not None else "N/A")
    metric_col4.metric("p-value", f"{prob_f:.6f}" if prob_f is not None else "N/A")

    st.info(
        "R2는 설명력, Adj. R2는 변수 수를 고려한 설명력입니다. "
        "p-value가 작을수록 모델 관계가 우연일 가능성이 낮다고 해석할 수 있습니다."
    )

    if dashboard_data.intercept is not None:
        st.markdown("### 절편 (Intercept)")
        st.code(f"{dashboard_data.intercept:.6f}")

    if dashboard_data.regression_equation:
        st.markdown("### 회귀식 (Regression Equation)")
        st.code(dashboard_data.regression_equation)

    if dashboard_data.combined_summary_table is not None and not dashboard_data.combined_summary_table.empty:
        st.markdown("### 회귀계수표")
        st.dataframe(localize_combined_summary_table(dashboard_data.combined_summary_table), use_container_width=True)

    st.markdown("### 해석 가이드")
    st.write("- 계수가 양수면 해당 변수가 커질수록 종속변수도 커지는 경향이 있습니다.")
    st.write("- 계수가 음수면 해당 변수가 커질수록 종속변수는 작아지는 경향이 있습니다.")
    st.write("- p-value가 작은 변수일수록 통계적으로 더 뚜렷한 관계일 가능성이 있습니다.")
    st.write("- 신뢰구간이 0을 지나지 않으면 효과 방향이 비교적 안정적일 가능성이 있습니다.")

    if dashboard_data.ols_diagnostics:
        st.markdown("### 진단 지표")
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        diag_col1.metric("Durbin-Watson", f"{dashboard_data.ols_diagnostics['durbin_watson']:.4f}")
        diag_col2.metric("Condition Number", f"{dashboard_data.ols_diagnostics['condition_number']:.4f}")
        diag_col3.metric("Residual Skew", f"{dashboard_data.ols_diagnostics['residual_skew']:.4f}")
        st.caption(f"Residual Kurtosis: {dashboard_data.ols_diagnostics['residual_kurtosis']:.4f}")


def render_classification_or_prediction_result(result: AnalysisRunResult, app_mode: str) -> None:
    if result.model_result is None or result.model_result.problem_type == "regression":
        return

    st.markdown("## 모델 결과")
    metrics_row = result.model_result.metrics.iloc[0].to_dict()
    metric_columns = st.columns(3)
    metric_columns[0].metric("Accuracy", f"{float(metrics_row.get('accuracy', 0.0)):.4f}")
    metric_columns[1].metric("F1", f"{float(metrics_row.get('f1', 0.0)):.4f}")
    roc_auc = metrics_row.get("roc_auc")
    metric_columns[2].metric("ROC-AUC", "N/A" if pd.isna(roc_auc) else f"{float(roc_auc):.4f}")
    if app_mode == "analysis":
        st.info("분류 분석은 회귀처럼 OLS 해석표가 없어 성능 지표 중심으로 보여줍니다.")


def render_saved_result() -> None:
    result = st.session_state.get("analysis_result")
    preview_df = st.session_state.get("analysis_df")
    app_mode = st.session_state.get("analysis_mode")
    if result is None or preview_df is None or app_mode is None:
        return

    st.markdown("---")
    st.success(f"분석이 완료되었습니다. 결과 폴더: {result.context.output_dir}")

    if result.model_result is not None:
        st.markdown("## 선택한 모델")
        st.success(result.model_result.best_model_name)

    render_regression_dashboard(result, preview_df, app_mode)
    render_classification_or_prediction_result(result, app_mode)

    if result.warnings:
        st.markdown("## 주의사항 및 경고")
        for record in result.warnings:
            st.warning(record.message)

    render_result_extras(result)


def main() -> None:
    st.set_page_config(page_title="StatAutoLab", layout="wide")
    st.title("StatAutoLab")
    st.caption("파일을 올리면 바로 EDA를 보여주고, 이후 분석 또는 예측 결과를 확인할 수 있습니다.")

    st.markdown("## 1. 데이터 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 선택하세요.",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
    )

    if uploaded_file is None:
        clear_analysis_state()
        st.info("분석할 CSV/XLSX 파일을 업로드하면 데이터 미리보기와 분석 옵션이 표시됩니다.")
        return

    file_key = current_file_key(uploaded_file)
    if st.session_state.get("analysis_file_key") not in (None, file_key):
        clear_analysis_state()

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        st.error("지원하지 않는 파일 형식입니다. CSV, XLSX, XLS 파일만 업로드할 수 있습니다.")
        return

    try:
        saved_input_path = save_uploaded_file(uploaded_file)
        preview_df = load_dataset(saved_input_path)
    except Exception as exc:
        clear_analysis_state()
        st.error(f"파일을 읽는 중 문제가 발생했습니다. 파일 형식과 내용을 확인해 주세요. 상세 오류: {exc}")
        return

    st.session_state["analysis_file_key"] = file_key
    st.success(f"업로드 완료: {uploaded_file.name}")
    render_uploaded_data_preview(preview_df)

    all_columns = preview_df.columns.astype(str).tolist()
    target_options = [NO_TARGET] + all_columns

    if st.session_state.get("target_option") not in target_options:
        st.session_state["target_option"] = NO_TARGET

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
    with right:
        output_dir = st.text_input(
            "출력 폴더",
            value=str(DEFAULT_SETTINGS["output_dir"]),
        )
        if app_mode == "prediction":
            st.caption("예측 모드는 내부적으로 기본 holdout 평가를 사용합니다.")
        else:
            st.caption("분석 모드는 해석 중심 결과를 우선 보여주며, 성능 비교는 화면에서 최소화합니다.")

    if st.button("분석 실행", use_container_width=True):
        feature_columns = st.session_state["selected_features"] or None
        cli_values = {
            "input_path": str(saved_input_path),
            "target": None if target_value == NO_TARGET else target_value,
            "feature_columns": feature_columns,
            "selected_model": None if selected_model == NO_MODEL else selected_model,
            "output_dir": output_dir.strip() or str(DEFAULT_SETTINGS["output_dir"]),
            "report_format": report_format,
            "task_type": task_type,
            "random_state": DEFAULT_SETTINGS["random_state"],
            "test_size": DEFAULT_SETTINGS["test_size"],
            "eval_method": DEFAULT_SETTINGS["eval_method"],
            "cv_folds": DEFAULT_SETTINGS["cv_folds"],
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

    render_saved_result()


if __name__ == "__main__":
    main()
