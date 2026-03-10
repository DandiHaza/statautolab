# StatAutoLab MVP

StatAutoLab MVP는 CSV 및 Excel 파일을 대상으로 탐색적 데이터 분석과 baseline 모델 비교를 수행하는 최소 기능 도구입니다.

## 주요 기능

- `.csv`, `.xlsx`, `.xls` 데이터셋 로딩
- 데이터 미리보기 및 컬럼 요약 생성
- 결측치 분석
- 수치형 및 범주형 기술통계 생성
- 수치형 컬럼 히스토그램 저장
- 수치형 컬럼 박스플롯 저장
- 수치형 컬럼이 충분할 경우 상관행렬 히트맵 저장
- IQR 기반 이상치 탐지 및 요약 저장
- 모델 학습 전 자동 전처리 파이프라인 적용
- 전체 분석 결과를 Markdown 리포트로 저장
- `--target` 기준 회귀/분류 문제 자동 판별
- baseline 모델 학습 및 검증 성능 비교
- 실행 결과를 `outputs/YYYYMMDD/HHMMSS/` 구조로 저장
- 실행별 설정/요약/성능을 누적 실험 로그로 기록

## 프로젝트 구조

```text
StatAutoLab/
|-- app/
|   |-- __init__.py
|   |-- config.py
|   |-- evaluate.py
|   |-- experiment.py
|   |-- io.py
|   |-- model_selection.py
|   |-- preprocessing.py
|   |-- profiling.py
|   |-- report.py
|   |-- train.py
|   `-- visualization.py
|-- configs/
|   `-- default.yaml
|-- data/
|   `-- sample.csv
|-- tests/
|-- run_analysis.py
|-- requirements.txt
`-- README.md
```

## 설치

```bash
.\.venv\Scripts\pip3.exe install -r requirements.txt
```

## 사용 방법

```bash
python run_analysis.py --help
python run_analysis.py --config configs/default.yaml
python run_analysis.py --input data/sample.csv
python run_analysis.py --input data/sample.csv --target buy
python run_analysis.py --input my_data.xlsx --output-dir outputs
python run_analysis.py --input my_data.csv --target target_column --task-type classification
python run_analysis.py --input my_data.csv --target target_column --report-format html
python run_analysis.py my_data.csv
```

실행 예시:

```bash
python run_analysis.py --config configs/default.yaml
python run_analysis.py --input data/sample.csv
python run_analysis.py --input data/sample.csv --target buy --task-type auto
python run_analysis.py --input my_data.csv --target churn --task-type classification --report-format html --output-dir outputs
```

config 기반 실행:

```bash
python run_analysis.py --config configs/default.yaml
python run_analysis.py --config configs/default.yaml --report-format html
python run_analysis.py --config configs/default.yaml --target another_target
```

기본 출력 경로 예시:

```text
outputs/
├── experiments_log.csv
└── 20260310/
    └── 113000/
        ├── charts/
        ├── config_snapshot.json
        ├── data_summary.json
        ├── outlier_summary.csv
        ├── preprocessing_summary.md
        ├── report.md
        ├── model_comparison.csv
        └── model_summary.md
```

샘플 데이터가 없는 경우:

```bash
python run_analysis.py --input your_data.csv
```

## 출력 결과

- `outputs/YYYYMMDD/HHMMSS/report.md`
- `outputs/YYYYMMDD/HHMMSS/report.html`
- `outputs/YYYYMMDD/HHMMSS/config_snapshot.json`
- `outputs/YYYYMMDD/HHMMSS/data_summary.json`
- `outputs/YYYYMMDD/HHMMSS/charts/histogram_<column>.png`
- `outputs/YYYYMMDD/HHMMSS/charts/boxplot_<column>.png`
- `outputs/YYYYMMDD/HHMMSS/charts/correlation_matrix.png`
- `outputs/YYYYMMDD/HHMMSS/outlier_summary.csv`
- `outputs/YYYYMMDD/HHMMSS/preprocessing_summary.md`
- `outputs/YYYYMMDD/HHMMSS/model_comparison.csv`
- `outputs/YYYYMMDD/HHMMSS/model_summary.md`
- `outputs/experiments_log.csv`

## 참고

- 숫자형 타깃은 회귀, 비숫자형 타깃은 분류로 처리합니다.
- `--task-type`으로 회귀/분류를 강제 지정할 수 있습니다.
- CLI 인자와 config 파일이 동시에 주어지면 CLI 인자가 우선합니다.
- 상관분석과 히스토그램은 수치형 컬럼에 대해서만 생성됩니다.
- 수치형 결측치는 평균값, 범주형 결측치는 최빈값으로 대체합니다.
- 날짜형 컬럼은 경고만 출력하고 자동 feature engineering 없이 학습에서 제외합니다.

## 테스트

```bash
python -m pytest tests
```
