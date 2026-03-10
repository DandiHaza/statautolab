from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from app.preprocessing import PreprocessingSummary
from app.profiling import ProfileResult
from app.train import ModelResult


def _table_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_데이터가 없습니다._"
    return df.to_markdown(index=index)


def _build_dtype_overview(profile: ProfileResult) -> pd.DataFrame:
    dtype_counts = profile.dtypes.copy()
    dtype_counts["type_group"] = dtype_counts["dtype"].astype(str)
    return (
        dtype_counts.groupby("type_group", as_index=False)
        .size()
        .rename(columns={"size": "column_count", "type_group": "dtype"})
        .sort_values(["column_count", "dtype"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _build_top_missing(profile: ProfileResult, top_n: int = 5) -> pd.DataFrame:
    missing = profile.missing.copy()
    missing = missing[missing["missing_count"] > 0]
    return missing.head(top_n).reset_index(drop=True)


def _build_top_outliers(profile: ProfileResult, top_n: int = 5) -> pd.DataFrame:
    if profile.outliers.empty:
        return pd.DataFrame(columns=["column", "outlier_count", "outlier_ratio_pct"])

    outliers = profile.outliers.copy()
    outliers = outliers[outliers["outlier_count"] > 0]
    if outliers.empty:
        return outliers
    return outliers[["column", "outlier_count", "outlier_ratio_pct"]].head(top_n).reset_index(drop=True)


def _build_numeric_focus(profile: ProfileResult, top_n: int = 10) -> pd.DataFrame:
    if profile.numeric_summary.empty:
        return pd.DataFrame(columns=["column", "mean", "std", "min", "max"])

    columns = ["column", "mean", "std", "min", "max"]
    numeric_focus = profile.numeric_summary[columns].copy()
    return numeric_focus.head(top_n).reset_index(drop=True)


def _build_top_correlations(profile: ProfileResult, top_n: int = 5) -> pd.DataFrame:
    if profile.correlation.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])

    pairs: list[dict[str, object]] = []
    columns = profile.correlation.columns.tolist()
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(profile.correlation.loc[left, right])
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )

    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])

    return (
        pd.DataFrame(pairs)
        .sort_values(["abs_correlation", "feature_a", "feature_b"], ascending=[False, True, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def _build_model_focus(model_result: ModelResult | None) -> pd.DataFrame:
    if model_result is None or model_result.metrics.empty:
        return pd.DataFrame()

    metrics = model_result.metrics.copy()
    if model_result.problem_type == "regression":
        preferred_columns = ["model", "rmse", "mae", "r2"]
    else:
        preferred_columns = ["model", "accuracy", "f1", "roc_auc"]
    existing_columns = [column for column in preferred_columns if column in metrics.columns]
    return metrics[existing_columns]


def _get_target_correlation_insight(model_result: ModelResult | None, profile: ProfileResult) -> str | None:
    if model_result is None or model_result.problem_type != "regression":
        return None
    if profile.correlation.empty or model_result.target not in profile.correlation.columns:
        return None

    target_corr = (
        profile.correlation[model_result.target]
        .drop(labels=[model_result.target], errors="ignore")
        .dropna()
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    if target_corr.empty:
        return None

    feature = str(target_corr.index[0])
    value = float(target_corr.iloc[0])
    return f"`{model_result.target}`와 가장 관련성이 높아 보이는 변수는 `{feature}`이며, 상관계수는 {value:.3f}입니다."


def _build_auto_insights(
    profile: ProfileResult,
    top_missing: pd.DataFrame,
    top_outliers: pd.DataFrame,
    top_correlations: pd.DataFrame,
    model_result: ModelResult | None,
) -> list[str]:
    insights: list[str] = []

    if not top_missing.empty:
        row = top_missing.iloc[0]
        insights.append(
            f"결측치가 가장 많은 컬럼은 `{row['column']}`이며 결측 비율은 약 {float(row['missing_ratio_pct']):.1f}%입니다. 추가 검토가 필요할 가능성이 있습니다."
        )
    else:
        insights.append("결측치가 보고되지 않아 기본 품질은 비교적 안정적으로 보입니다.")

    if not top_outliers.empty:
        row = top_outliers.iloc[0]
        insights.append(
            f"IQR 기준 이상치가 가장 많은 컬럼은 `{row['column']}`이며 비율은 약 {float(row['outlier_ratio_pct']):.1f}%입니다. 값 분포를 추가 확인할 필요가 있습니다."
        )
    else:
        insights.append("IQR 기준으로 뚜렷한 이상치가 보고되지 않았습니다.")

    if not top_correlations.empty:
        row = top_correlations.iloc[0]
        strength = float(row["abs_correlation"])
        if strength >= 0.7:
            descriptor = "강한 선형 관계 가능성"
        elif strength >= 0.4:
            descriptor = "중간 수준의 선형 관계 가능성"
        else:
            descriptor = "약한 선형 관계"
        insights.append(
            f"`{row['feature_a']}`와 `{row['feature_b']}`는 상관계수 {float(row['correlation']):.3f}로 관찰되며, {descriptor}이 있습니다."
        )
    else:
        insights.append("수치형 변수 수가 적어 상관분석에서 뚜렷한 변수쌍을 찾기 어려웠습니다.")

    target_insight = _get_target_correlation_insight(model_result, profile)
    if target_insight is not None:
        insights.append(target_insight)

    if model_result is not None and not model_result.metrics.empty:
        best_row = model_result.metrics.loc[model_result.metrics["model"] == model_result.best_model_name].iloc[0]
        if model_result.problem_type == "regression":
            insights.append(
                f"현재 baseline 기준으로 가장 성능이 좋은 모델은 `{model_result.best_model_name}`이며, RMSE는 {float(best_row['rmse']):.3f}, MAE는 {float(best_row['mae']):.3f}입니다."
            )
        else:
            roc_auc_text = ""
            if "roc_auc" in best_row.index and pd.notna(best_row["roc_auc"]):
                roc_auc_text = f", ROC-AUC는 {float(best_row['roc_auc']):.3f}"
            insights.append(
                f"현재 baseline 기준으로 가장 성능이 좋은 모델은 `{model_result.best_model_name}`이며, Accuracy는 {float(best_row['accuracy']):.3f}, F1은 {float(best_row['f1']):.3f}{roc_auc_text}입니다."
            )
    else:
        insights.append("모델 자동화가 실행되지 않아 모델 성능에 대한 해석은 포함되지 않았습니다.")

    return insights


def build_markdown_report(
    source_name: str,
    profile: ProfileResult,
    histogram_paths: list[Path],
    boxplot_paths: list[Path],
    correlation_path: Path | None,
    preprocessing_summary: PreprocessingSummary | None = None,
    model_result: ModelResult | None = None,
) -> str:
    dtype_overview = _build_dtype_overview(profile)
    top_missing = _build_top_missing(profile)
    numeric_focus = _build_numeric_focus(profile)
    top_outliers = _build_top_outliers(profile)
    top_correlations = _build_top_correlations(profile)
    model_focus = _build_model_focus(model_result)
    auto_insights = _build_auto_insights(profile, top_missing, top_outliers, top_correlations, model_result)

    lines: list[str] = []
    lines.append(f"# 데이터 분석 리포트: {source_name}")
    lines.append("")
    lines.append("이 리포트는 데이터의 구조, 결측치, 수치형 변수 특성, 상관관계, 그리고 가능한 경우 baseline 모델 성능을 중심으로 요약합니다.")
    lines.append("")

    lines.append("## 핵심 인사이트")
    lines.append("")
    for insight in auto_insights:
        lines.append(f"- {insight}")
    lines.append("")

    lines.append("## 1. 데이터 개요")
    lines.append("")
    lines.append(
        f"데이터는 총 **{profile.row_count}개 행**, **{profile.column_count}개 열**로 구성되어 있습니다. "
        "아래 표는 컬럼 타입 분포를 요약한 것입니다."
    )
    lines.append("")
    lines.append(_table_to_markdown(dtype_overview))
    lines.append("")

    lines.append("## 2. 결측치 요약")
    lines.append("")
    if top_missing.empty:
        lines.append("결측치가 있는 컬럼이 확인되지 않았습니다.")
        lines.append("")
    else:
        lines.append("결측치가 많은 상위 컬럼은 아래와 같습니다. 비율이 높은 컬럼은 수집 과정이나 입력 규칙을 추가 점검하는 것이 좋습니다.")
        lines.append("")
        lines.append(_table_to_markdown(top_missing))
        lines.append("")

    lines.append("## 3. 수치형 변수 요약")
    lines.append("")
    if numeric_focus.empty:
        lines.append("수치형 컬럼이 없어 평균, 표준편차, 최소값, 최대값 요약을 생략합니다.")
        lines.append("")
    else:
        lines.append("수치형 컬럼의 기본 분포를 빠르게 확인할 수 있도록 핵심 통계량만 정리했습니다.")
        lines.append("")
    lines.append(_table_to_markdown(numeric_focus))
    lines.append("")

    lines.append("## 4. 이상치 요약")
    lines.append("")
    if top_outliers.empty:
        lines.append("IQR 기준으로 보고할 이상치가 많은 수치형 컬럼은 확인되지 않았습니다.")
        lines.append("")
    else:
        lines.append("이상치 개수와 비율이 높은 상위 수치형 컬럼은 아래와 같습니다. 현재 단계에서는 탐지만 수행하며 자동 삭제는 하지 않았습니다.")
        lines.append("")
        lines.append(_table_to_markdown(top_outliers))
        lines.append("")

    lines.append("## 5. 상관분석 요약")
    lines.append("")
    if top_correlations.empty:
        lines.append("상관분석을 수행하기에 충분한 수치형 컬럼 쌍이 없었습니다.")
        lines.append("")
    else:
        lines.append("절대 상관계수가 높은 변수쌍 상위 항목입니다. 이는 선형 관계의 신호일 수 있지만 인과를 의미하지는 않습니다.")
        lines.append("")
        lines.append(_table_to_markdown(top_correlations.drop(columns=["abs_correlation"])))
        lines.append("")
        if correlation_path is not None:
            lines.append(f"![correlation_matrix]({correlation_path.as_posix()})")
            lines.append("")

    lines.append("## 6. 전처리 요약")
    lines.append("")
    if preprocessing_summary is None:
        lines.append("`--target`이 지정되지 않아 모델 학습용 전처리 파이프라인은 실행되지 않았습니다.")
        lines.append("")
    else:
        lines.append("모델 학습 전 전처리는 아래 기준으로 적용되었습니다.")
        lines.append("")
        lines.append(f"- 타깃 컬럼 제외: `{preprocessing_summary.target_column}`")
        lines.append(
            f"- 수치형 컬럼: {', '.join(preprocessing_summary.numeric_columns) if preprocessing_summary.numeric_columns else '없음'}"
        )
        lines.append(
            f"- 범주형 컬럼: {', '.join(preprocessing_summary.categorical_columns) if preprocessing_summary.categorical_columns else '없음'}"
        )
        lines.append(
            f"- 날짜형 컬럼: {', '.join(preprocessing_summary.datetime_columns) if preprocessing_summary.datetime_columns else '없음'}"
        )
        lines.append("- 수치형 결측치: 평균값 대체")
        lines.append("- 범주형 결측치: 최빈값 대체")
        lines.append("- 범주형 인코딩: OneHotEncoder")
        if preprocessing_summary.datetime_columns:
            lines.append("- 날짜형 컬럼은 자동 feature engineering 없이 제외되었습니다.")
        lines.append("")

    lines.append("## 7. 모델 결과 요약")
    lines.append("")
    if model_result is None:
        lines.append("모델 자동화는 실행되지 않았습니다. baseline 학습을 보려면 `--target`을 지정하세요.")
        lines.append("")
    else:
        problem_type_label = "회귀" if model_result.problem_type == "regression" else "분류"
        lines.append(
            f"타깃 컬럼은 `{model_result.target}`이며 문제 유형은 **{problem_type_label}**로 판별되었습니다. "
            f"학습 데이터는 {model_result.train_rows}행, 검증 데이터는 {model_result.validation_rows}행입니다."
        )
        lines.append("")
        lines.append(f"현재 baseline 비교에서 가장 좋은 모델은 **{model_result.best_model_name}** 입니다.")
        lines.append("")
        lines.append(_table_to_markdown(model_focus))
        lines.append("")

    lines.append("## 부록: 데이터 미리보기")
    lines.append("")
    lines.append(_table_to_markdown(profile.preview))
    lines.append("")

    if histogram_paths:
        lines.append("## 부록: 히스토그램")
        lines.append("")
        for path in histogram_paths:
            lines.append(f"![{path.stem}]({path.as_posix()})")
        lines.append("")

    if boxplot_paths:
        lines.append("## 부록: 박스플롯")
        lines.append("")
        for path in boxplot_paths:
            lines.append(f"![{path.stem}]({path.as_posix()})")
        lines.append("")

    return "\n".join(lines)


def save_markdown_report(content: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig")
    return path


def _render_markdown_table(lines: list[str]) -> str:
    if len(lines) < 2:
        return ""

    header_cells = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    body_lines = lines[2:] if len(lines) >= 2 else []

    parts = ["<table>", "<thead>", "<tr>"]
    for cell in header_cells:
        parts.append(f"<th>{html.escape(cell)}</th>")
    parts.extend(["</tr>", "</thead>", "<tbody>"])

    for line in body_lines:
        row_cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        parts.append("<tr>")
        for cell in row_cells:
            parts.append(f"<td>{html.escape(cell)}</td>")
        parts.append("</tr>")

    parts.extend(["</tbody>", "</table>"])
    return "\n".join(parts)


def render_html_report(markdown_content: str, title: str) -> str:
    lines = markdown_content.splitlines()
    html_lines = [
        "<!DOCTYPE html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #1f2937; }",
        "h1, h2, h3 { color: #111827; }",
        "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
        "th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; }",
        "th { background: #f3f4f6; }",
        "code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }",
        "img { max-width: 100%; height: auto; margin: 12px 0; border: 1px solid #e5e7eb; }",
        "ul { padding-left: 20px; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    in_list = False
    in_table = False
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_lines
        if in_table and table_lines:
            html_lines.append(_render_markdown_table(table_lines))
        in_table = False
        table_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            in_table = True
            table_lines.append(stripped)
            continue

        flush_table()

        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            alt_text = stripped[2 : stripped.index("]")]
            src = stripped[stripped.index("(") + 1 : -1]
            html_lines.append(f'<img src="{html.escape(src)}" alt="{html.escape(alt_text)}">')
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{html.escape(stripped)}</p>")

    flush_table()
    if in_list:
        html_lines.append("</ul>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def save_html_report(markdown_content: str, output_path: str | Path, title: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(markdown_content, title), encoding="utf-8-sig")
    return path
