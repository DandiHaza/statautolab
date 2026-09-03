"""화면 표시에 쓰는 상수와 예제 데이터 카탈로그."""

from __future__ import annotations

from pathlib import Path

from app.config import DEFAULT_SETTINGS


UPLOAD_DIR = Path(".streamlit_uploads")


OUTPUT_ROOT = Path(str(DEFAULT_SETTINGS["output_dir"]))


NO_TARGET = "선택 안 함"


NO_MODEL = "자동 선택"


SAMPLE_DATASETS = [
    {
        "key": "wine",
        "path": Path("data/real/winequality-red.csv"),
        "label": "와인 품질 (실데이터)",
        "detail": "1,599행 · 12열 — 화학 성분으로 품질 점수를 예측하는 회귀 문제입니다.",
        "target": "quality",
    },
    {
        "key": "regression",
        "path": Path("data/examples/regression_sample.csv"),
        "label": "고객 지출 점수",
        "detail": "소형 회귀 예제 — 식별자 컬럼(customer_id)이 자동 제외되는 동작을 볼 수 있습니다.",
        "target": "spending_score",
    },
    {
        "key": "classification",
        "path": Path("data/examples/classification_sample.csv"),
        "label": "구매 여부 분류",
        "detail": "소형 분류 예제 — 결측치 처리와 범주형 인코딩이 함께 적용됩니다.",
        "target": "buy",
    },
    {
        "key": "datetime",
        "path": Path("data/examples/datetime_sample.csv"),
        "label": "날짜 컬럼 경고",
        "detail": "날짜형 컬럼이 자동 감지되어 학습에서 제외되고 경고로 남는 과정을 보여줍니다.",
        "target": "target",
    },
    {
        "key": "eda",
        "path": Path("data/examples/eda_sample.csv"),
        "label": "EDA만 살펴보기",
        "detail": "타깃 없이 데이터 개요·결측치·이상치·상관관계만 확인합니다.",
        "target": None,
    },
]


APP_MODE_LABELS = {
    "analysis": "분석 모드",
    "prediction": "예측 모드",
}


TASK_TYPE_LABELS = {
    "auto": "자동 판별",
    "regression": "회귀",
    "classification": "분류",
}


EVAL_METHOD_LABELS = {
    "holdout": "홀드아웃 (한 번 나눠서 검증)",
    "cv": "교차검증 (K-Fold)",
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
