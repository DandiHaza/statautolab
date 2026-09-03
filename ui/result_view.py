"""분석 결과 화면."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app.analysis_runner import AnalysisRunResult
from app.regression_insights import build_regression_dashboard_data
from ui.analysis import collect_download_files
from ui.formatting import (
    build_ols_display_text,
    localize_combined_summary_table,
    localize_corr_pairs,
    localize_vif_table,
    strip_markdown_images,
)


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


def render_residual_plots(dashboard_data) -> None:
    residual_df = dashboard_data.predictions_preview.copy()
    if residual_df.empty:
        return

    st.markdown("### 잔차 플롯")
    left, right = st.columns(2)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=residual_df, x="predicted", y="residual", ax=ax1)
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax1.set_title("Predicted vs Residual")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residual")
    with left:
        st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=residual_df, x="actual", y="predicted", ax=ax2)
    min_value = min(residual_df["actual"].min(), residual_df["predicted"].min())
    max_value = max(residual_df["actual"].max(), residual_df["predicted"].max())
    ax2.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=1)
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    with right:
        st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    st.markdown("#### 잔차 플롯 읽는 법")
    st.write("- `Predicted vs Residual`에서 점들이 0선을 기준으로 위아래에 고르게 퍼지면 비교적 양호합니다.")
    st.write("- 점들이 한쪽으로 치우치거나 곡선처럼 보이면, 선형관계가 충분하지 않을 가능성이 있습니다.")
    st.write("- 예측값이 커질수록 퍼짐이 넓어지면 오차 크기가 일정하지 않을 가능성이 있습니다.")
    st.write("- `Actual vs Predicted`에서 점들이 빨간 기준선에 가까울수록 예측이 실제값에 더 가깝습니다.")
    st.caption("잔차 플롯은 모델이 데이터를 무리 없이 설명하는지 빠르게 점검하는 그림입니다. 완벽한 직선보다, 뚜렷한 패턴이 없는지가 더 중요합니다.")


def render_multicollinearity_section(dashboard_data) -> None:
    st.markdown("### 다중공선성 점검")

    if dashboard_data.high_correlation_pairs.empty and dashboard_data.vif_table.empty:
        st.info("수치형 독립변수가 적어 다중공선성 점검 결과를 만들지 않았습니다.")
        return

    if not dashboard_data.high_correlation_pairs.empty:
        st.write("상관계수 절댓값이 0.7 이상인 변수쌍입니다.")
        st.dataframe(localize_corr_pairs(dashboard_data.high_correlation_pairs.head(10)), width="stretch")
    else:
        st.info("상관계수 절댓값이 0.7 이상인 주요 변수쌍은 없습니다.")

    if not dashboard_data.vif_table.empty:
        max_vif = float(dashboard_data.vif_table["vif"].max())
        if max_vif >= 10:
            st.error("현재 모델에는 VIF가 10 이상인 변수가 있습니다. 해석이 크게 흔들릴 수 있습니다.")
        elif max_vif >= 5:
            st.warning("현재 모델에는 VIF가 5 이상인 변수가 있습니다. 계수 해석 시 주의가 필요합니다.")
        else:
            st.success("현재 모델의 VIF는 크게 높지 않습니다.")
        st.dataframe(localize_vif_table(dashboard_data.vif_table.head(10)), width="stretch")
        st.caption("보통 VIF 5 이상이면 주의, 10 이상이면 강한 다중공선성 가능성을 의심합니다.")


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
        st.dataframe(localize_combined_summary_table(dashboard_data.combined_summary_table), width="stretch")

    render_residual_plots(dashboard_data)
    render_multicollinearity_section(dashboard_data)

    if dashboard_data.ols_diagnostics:
        st.markdown("### 진단 지표")
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        diag_col1.metric("Durbin-Watson", f"{dashboard_data.ols_diagnostics['durbin_watson']:.4f}")
        diag_col2.metric("Condition Number", f"{dashboard_data.ols_diagnostics['condition_number']:.4f}")
        diag_col3.metric("Residual Skew", f"{dashboard_data.ols_diagnostics['residual_skew']:.4f}")
        st.caption(
            "Durbin-Watson은 잔차 자기상관, Condition Number는 수치적 불안정성, "
            f"Residual Kurtosis는 잔차 뾰족함을 봅니다. 현재 Residual Kurtosis: {dashboard_data.ols_diagnostics['residual_kurtosis']:.4f}"
        )


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


def render_result_panel(result: AnalysisRunResult, preview_df: pd.DataFrame, app_mode: str) -> None:
    st.markdown("---")
    st.success("분석이 완료되었습니다. 아래에서 결과를 확인하고 파일로 내려받을 수 있습니다.")

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


def render_saved_result() -> None:
    result = st.session_state.get("analysis_result")
    preview_df = st.session_state.get("analysis_df")
    app_mode = st.session_state.get("analysis_mode")
    if result is None or preview_df is None or app_mode is None:
        return
    render_result_panel(result, preview_df, app_mode)
