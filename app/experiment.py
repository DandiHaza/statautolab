from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from app.profiling import ProfileResult
from app.train import ModelResult


def _to_serializable(value: object) -> object:
    if is_dataclass(value):
        return {key: _to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    return value


def save_config_snapshot(config: dict[str, object], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "config_snapshot.json"
    config_path.write_text(
        json.dumps(_to_serializable(config), ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    return config_path


def save_data_summary(profile: ProfileResult, input_file: str | Path, output_dir: str | Path) -> Path:
    summary = {
        "input_file": str(input_file),
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "dtype_summary": profile.dtypes.to_dict(orient="records"),
        "missing_top": profile.missing.head(10).to_dict(orient="records"),
        "outlier_top": profile.outliers.head(10).to_dict(orient="records"),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "data_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return summary_path


def _get_main_metric(model_result: ModelResult | None) -> str:
    if model_result is None or model_result.metrics.empty:
        return ""

    best_row = model_result.metrics.loc[model_result.metrics["model"] == model_result.best_model_name].iloc[0]
    if model_result.problem_type == "regression":
        return f"rmse={float(best_row['rmse']):.4f}"

    if "roc_auc" in best_row.index and pd.notna(best_row["roc_auc"]):
        return f"accuracy={float(best_row['accuracy']):.4f}, roc_auc={float(best_row['roc_auc']):.4f}"
    return f"accuracy={float(best_row['accuracy']):.4f}"


def append_experiment_log(
    base_output_dir: str | Path,
    run_id: str,
    timestamp: str,
    input_file: str | Path,
    target: str | None,
    task_type: str,
    model_result: ModelResult | None,
    output_path: str | Path,
) -> Path:
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    log_path = base_path / "experiments_log.csv"

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "input_file": str(input_file),
        "target": target or "",
        "task_type": model_result.problem_type if model_result is not None else task_type,
        "best_model": model_result.best_model_name if model_result is not None else "",
        "main_metric": _get_main_metric(model_result),
        "output_path": str(output_path),
    }

    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([row])

    log_df.to_csv(log_path, index=False)
    return log_path
