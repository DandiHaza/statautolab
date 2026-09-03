"""데이터 선택과 EDA 화면."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ui.analysis import analyze_feature_selection, build_profile, build_recommended_removals
from ui.constants import SAMPLE_DATASETS
from ui.formatting import (
    localize_corr_pairs,
    localize_missing_summary,
    localize_outlier_summary,
    localize_profile_dtypes,
    localize_vif_table,
)
from ui.state import remove_feature_from_selection, select_sample_dataset


def render_sample_picker() -> None:
    st.markdown("### 파일이 없다면 예제 데이터로 바로 체험해 보세요")
    st.caption("하나를 고르면 타깃 컬럼까지 설정되어 곧바로 `분석 실행`을 누를 수 있습니다.")

    for row_start in range(0, len(SAMPLE_DATASETS), 2):
        columns = st.columns(2)
        for column, sample in zip(columns, SAMPLE_DATASETS[row_start : row_start + 2]):
            with column:
                st.markdown(f"**{sample['label']}**")
                st.caption(sample["detail"])
                if not sample["path"].exists():
                    st.caption("예제 파일을 찾을 수 없습니다.")
                elif st.button("이 데이터로 시작", key=f"sample-{sample['key']}", width="stretch"):
                    select_sample_dataset(sample)
                    st.rerun()


def render_uploaded_data_preview(df: pd.DataFrame, file_path: str) -> None:
    profile = build_profile(file_path)

    st.markdown("## 데이터 파일 미리보기")
    st.dataframe(df.head(20), width="stretch")

    st.markdown("## 데이터 개요")
    st.caption(f"행: {profile.row_count}개 / 열: {profile.column_count}개")
    st.dataframe(localize_profile_dtypes(profile.dtypes), width="stretch")

    st.markdown("## 결측치 요약")
    missing_df = profile.missing[profile.missing["missing_count"] > 0].head(10)
    if missing_df.empty:
        st.info("결측치가 있는 컬럼이 없습니다.")
    else:
        st.dataframe(localize_missing_summary(missing_df), width="stretch")

    st.markdown("## 이상치 요약")
    outlier_df = profile.outliers[profile.outliers["outlier_count"] > 0].head(10)
    if outlier_df.empty:
        st.info("IQR 기준으로 탐지된 주요 이상치 컬럼이 없습니다.")
    else:
        st.dataframe(localize_outlier_summary(outlier_df), width="stretch")

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

    if not correlation_df.empty:
        st.markdown("### 상관행렬")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(correlation_df, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def render_feature_selection_feedback(df: pd.DataFrame, selected_features: list[str]) -> None:
    if len(selected_features) < 2:
        return

    analysis = analyze_feature_selection(df, tuple(selected_features))
    high_corr_pairs = analysis["high_corr_pairs"]
    vif_table = analysis["vif_table"]
    max_vif = analysis["max_vif"]

    st.markdown("### 변수 선택 안내")

    if not high_corr_pairs.empty:
        top_pair = high_corr_pairs.iloc[0]
        st.warning(
            f"상관이 높은 변수쌍이 있습니다: {top_pair['feature_1']} / {top_pair['feature_2']} "
            f"(상관계수 {top_pair['correlation']:.2f}). 해석 목적이면 둘 중 하나만 쓰는 편이 더 안정적일 수 있습니다."
        )
        st.dataframe(localize_corr_pairs(high_corr_pairs.head(5)), width="stretch")
    else:
        st.info("선택한 수치형 변수 중 상관계수 절댓값 0.7 이상인 주요 변수쌍은 없습니다.")

    if not vif_table.empty:
        if max_vif is not None and max_vif >= 10:
            st.error("VIF가 10 이상인 변수가 있어 다중공선성 가능성이 큽니다.")
        elif max_vif is not None and max_vif >= 5:
            st.warning("VIF가 5 이상인 변수가 있어 해석이 불안정할 수 있습니다.")
        else:
            st.success("현재 선택 기준에서는 VIF가 크게 높지 않습니다.")
        st.dataframe(localize_vif_table(vif_table.head(5)), width="stretch")
        st.caption("보통 VIF 5 이상이면 주의, 10 이상이면 강한 다중공선성 가능성을 의심합니다.")

    recommendations = build_recommended_removals(high_corr_pairs, vif_table)
    if recommendations:
        st.markdown("#### 제거 추천")
        st.caption("아래 추천은 자동 강제가 아니라 해석 안정성을 높이기 위한 제안입니다.")
        for index, recommendation in enumerate(recommendations):
            feature = recommendation["feature"]
            reason = recommendation["reason"]
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"- `{feature}` 제거 추천: {reason}")
            with col2:
                if st.button("제거", key=f"remove-feature-{index}-{feature}", width="stretch"):
                    remove_feature_from_selection(feature)
                    st.rerun()
