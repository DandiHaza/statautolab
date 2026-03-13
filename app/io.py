from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
CSV_FALLBACK_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = []
    for column in df.columns:
        cleaned = str(column).replace("\ufeff", "").replace("ï»¿", "").strip()
        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
            cleaned = cleaned[1:-1]
        renamed.append(cleaned)
    result = df.copy()
    result.columns = renamed
    return result


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        sample_hint = (
            "샘플 파일 `data/sample.csv`가 있으면 `python run_analysis.py --input data/sample.csv`로 바로 실행할 수 있습니다. "
            if Path("data/sample.csv").exists()
            else "샘플 파일 `data/sample.csv`가 현재 없으므로 분석할 CSV/XLSX 파일 경로를 직접 지정해야 합니다. "
        )
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {path.resolve()}. "
            f"{sample_hint}"
            "예시: `python run_analysis.py --input your_data.csv`"
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: '{suffix}'. 지원 형식: {supported}. "
            "CSV 또는 Excel 파일을 사용하세요."
        )

    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in CSV_FALLBACK_ENCODINGS:
            try:
                return _normalize_column_names(pd.read_csv(path, encoding=encoding))
            except Exception as exc:
                last_error = exc
        raise ValueError(
            f"CSV 파일을 읽는 중 오류가 발생했습니다: {path.resolve()}. "
            "UTF-8, CP1252, Latin-1 인코딩으로 시도했지만 읽지 못했습니다. 파일 인코딩이나 구분자를 확인하세요."
        ) from last_error

    try:
        return _normalize_column_names(pd.read_excel(path))
    except Exception as exc:
        raise ValueError(
            f"Excel 파일을 읽는 중 오류가 발생했습니다: {path.resolve()}. "
            "파일이 열려 있거나 손상되지 않았는지 확인하세요."
        ) from exc
