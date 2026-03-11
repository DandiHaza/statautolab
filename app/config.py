from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_SETTINGS: dict[str, object] = {
    "input_path": None,
    "target": None,
    "feature_columns": None,
    "output_dir": "outputs",
    "report_format": "md",
    "task_type": "auto",
    "random_state": 42,
    "test_size": 0.2,
    "eval_method": "holdout",
    "cv_folds": 5,
}

ALLOWED_REPORT_FORMATS = {"md", "html"}
ALLOWED_TASK_TYPES = {"auto", "regression", "classification"}
ALLOWED_EVAL_METHODS = {"holdout", "cv"}
ALLOWED_KEYS = set(DEFAULT_SETTINGS.keys())


def load_config_file(config_path: str | Path | None) -> dict[str, object]:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path.resolve()}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"설정 파일을 읽는 중 오류가 발생했습니다: {path.resolve()}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("설정 파일 최상위 구조는 key-value 형태여야 합니다.")

    unknown_keys = set(loaded.keys()) - ALLOWED_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ValueError(f"설정 파일에 지원하지 않는 키가 있습니다: {unknown}")

    return loaded


def resolve_settings(cli_values: dict[str, object], config_values: dict[str, object]) -> dict[str, object]:
    resolved = dict(DEFAULT_SETTINGS)
    resolved.update(config_values)
    for key, value in cli_values.items():
        if value is not None:
            resolved[key] = value
    validate_settings(resolved)
    return resolved


def validate_settings(settings: dict[str, object]) -> None:
    if not settings.get("input_path"):
        raise ValueError("입력 파일이 필요합니다. CLI의 `--input` 또는 config의 `input_path`를 지정해 주세요.")

    report_format = str(settings["report_format"])
    if report_format not in ALLOWED_REPORT_FORMATS:
        raise ValueError(f"`report_format`은 {sorted(ALLOWED_REPORT_FORMATS)} 중 하나여야 합니다.")

    task_type = str(settings["task_type"])
    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(f"`task_type`은 {sorted(ALLOWED_TASK_TYPES)} 중 하나여야 합니다.")

    eval_method = str(settings["eval_method"])
    if eval_method not in ALLOWED_EVAL_METHODS:
        raise ValueError(f"`eval_method`는 {sorted(ALLOWED_EVAL_METHODS)} 중 하나여야 합니다.")

    try:
        test_size = float(settings["test_size"])
    except (TypeError, ValueError) as exc:
        raise ValueError("`test_size`는 0과 1 사이의 실수여야 합니다.") from exc
    if not 0 < test_size < 1:
        raise ValueError("`test_size`는 0과 1 사이 값이어야 합니다.")

    try:
        int(settings["random_state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("`random_state`는 정수여야 합니다.") from exc

    try:
        cv_folds = int(settings["cv_folds"])
    except (TypeError, ValueError) as exc:
        raise ValueError("`cv_folds`는 2 이상의 정수여야 합니다.") from exc
    if cv_folds < 2:
        raise ValueError("`cv_folds`는 2 이상이어야 합니다.")

    if task_type != "auto" and not settings.get("target"):
        raise ValueError("`task_type`을 regression/classification으로 지정하면 `target`이 필요합니다.")

    feature_columns = settings.get("feature_columns")
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or not all(isinstance(column, str) for column in feature_columns):
            raise ValueError("`feature_columns`는 컬럼명 문자열 리스트여야 합니다.")
