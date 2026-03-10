from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


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
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise ValueError(
                f"CSV 파일을 읽는 중 오류가 발생했습니다: {path.resolve()}. "
                "파일 인코딩이나 구분자를 확인하세요."
            ) from exc

    try:
        return pd.read_excel(path)
    except Exception as exc:
        raise ValueError(
            f"Excel 파일을 읽는 중 오류가 발생했습니다: {path.resolve()}. "
            "파일이 열려 있거나 손상되지 않았는지 확인하세요."
        ) from exc
