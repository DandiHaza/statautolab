from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from app.preprocessing import PreprocessingSummary
from app.profiling import ProfileResult

HIGH_MISSING_RATIO_THRESHOLD_PCT = 30.0
SEVERE_CLASS_IMBALANCE_THRESHOLD_PCT = 80.0


@dataclass
class WarningRecord:
    code: str
    level: str
    message: str
    details: dict[str, object]


def collect_data_warnings(
    profile: ProfileResult,
    preprocessing_summary: PreprocessingSummary | None = None,
    target_series: pd.Series | None = None,
    problem_type: str | None = None,
) -> list[WarningRecord]:
    warnings: list[WarningRecord] = []

    if preprocessing_summary is not None and preprocessing_summary.datetime_columns:
        warnings.append(
            WarningRecord(
                code="datetime_columns_excluded",
                level="warning",
                message="날짜형 컬럼이 감지되어 자동 feature engineering 없이 학습 대상에서 제외되었습니다.",
                details={"columns": preprocessing_summary.datetime_columns},
            )
        )

    high_missing = profile.missing[profile.missing["missing_ratio_pct"] >= HIGH_MISSING_RATIO_THRESHOLD_PCT].copy()
    if not high_missing.empty:
        warnings.append(
            WarningRecord(
                code="high_missing_ratio_columns",
                level="warning",
                message=f"결측치 비율이 {HIGH_MISSING_RATIO_THRESHOLD_PCT:.0f}% 이상인 컬럼이 있어 추가 검토가 필요합니다.",
                details={
                    "threshold_pct": HIGH_MISSING_RATIO_THRESHOLD_PCT,
                    "columns": high_missing[["column", "missing_ratio_pct"]].to_dict(orient="records"),
                },
            )
        )

    numeric_count = len(profile.numeric_summary)
    if numeric_count < 2 or profile.correlation.empty:
        warnings.append(
            WarningRecord(
                code="correlation_matrix_not_generated",
                level="warning",
                message="수치형 컬럼이 충분하지 않아 상관행렬을 생성하지 못했습니다.",
                details={"numeric_column_count": numeric_count},
            )
        )

    if target_series is not None and problem_type == "classification":
        distribution = target_series.value_counts(normalize=True, dropna=True).sort_values(ascending=False)
        if not distribution.empty:
            majority_ratio_pct = float(distribution.iloc[0] * 100)
            if majority_ratio_pct >= SEVERE_CLASS_IMBALANCE_THRESHOLD_PCT:
                warnings.append(
                    WarningRecord(
                        code="severe_target_class_imbalance",
                        level="warning",
                        message="타깃 클래스 분포가 크게 불균형할 가능성이 있어 성능 해석에 주의가 필요합니다.",
                        details={
                            "threshold_pct": SEVERE_CLASS_IMBALANCE_THRESHOLD_PCT,
                            "majority_class": str(distribution.index[0]),
                            "majority_ratio_pct": majority_ratio_pct,
                            "class_distribution_pct": {
                                str(label): float(ratio * 100) for label, ratio in distribution.items()
                            },
                        },
                    )
                )

    return warnings


def dedupe_warnings(warnings: list[WarningRecord]) -> list[WarningRecord]:
    seen: set[tuple[str, str]] = set()
    deduped: list[WarningRecord] = []
    for record in warnings:
        key = (record.code, str(record.details))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def save_warnings_json(warnings: list[WarningRecord], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "warnings.json"
    json_path.write_text(
        json.dumps([asdict(record) for record in warnings], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    return json_path


def build_warnings_summary_markdown(warnings: list[WarningRecord]) -> str:
    lines = ["# 주의사항 및 경고", ""]
    if not warnings:
        lines.append("- 기록된 주요 경고가 없습니다.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- 총 경고 수: {len(warnings)}")
    lines.append("")
    for record in warnings:
        lines.append(f"## {record.code}")
        lines.append("")
        lines.append(f"- 수준: {record.level}")
        lines.append(f"- 메시지: {record.message}")
        if record.details:
            lines.append(f"- 상세 정보: `{json.dumps(record.details, ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines)


def save_warnings_summary(warnings: list[WarningRecord], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    markdown_path = output_path / "warnings_summary.md"
    markdown_path.write_text(build_warnings_summary_markdown(warnings), encoding="utf-8-sig")
    json_path = save_warnings_json(warnings, output_path)
    return markdown_path, json_path
