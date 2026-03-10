# StatAutoLab

StatAutoLab은 CSV/XLSX 데이터를 빠르게 탐색하고, 기본 EDA와 baseline 모델 비교까지 한 번에 실행하는 CLI 프로젝트입니다.

## 빠른 시작

5분 안에 실행하려면 아래 순서대로 진행하면 됩니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_analysis.py --input data/examples/eda_sample.csv
```

첫 실행이 끝나면 `outputs/YYYYMMDD/HHMMSS/` 아래에 리포트와 차트가 생성됩니다.

## 예제 데이터

바로 실행해볼 수 있도록 `data/examples` 폴더를 제공합니다.

- `data/examples/eda_sample.csv`: 타깃 없이 EDA만 확인하는 예제
- `data/examples/classification_sample.csv`: 분류 예제. 타깃 컬럼은 `buy`
- `data/examples/regression_sample.csv`: 회귀 예제. 타깃 컬럼은 `spending_score`

## 실행 예시

### 1. 비타깃 EDA 예시

```powershell
python run_analysis.py --input data/examples/eda_sample.csv
```

생성 결과:
- `report.md` 또는 `report.html`
- 히스토그램, 박스플롯, 상관행렬 이미지
- `outlier_summary.csv`
- `warnings_summary.md`, `warnings.json`

### 2. 분류 예시

```powershell
python run_analysis.py --input data/examples/classification_sample.csv --target buy
```

추가 생성 결과:
- `preprocessing_summary.md`
- `model_comparison.csv`
- `model_summary.md`
- `best_model.joblib`
- `model_metadata.json`

### 3. 회귀 예시

```powershell
python run_analysis.py --input data/examples/regression_sample.csv --target spending_score
```

평가 방식을 바꾸고 싶으면 아래처럼 실행할 수 있습니다.

```powershell
python run_analysis.py --input data/examples/regression_sample.csv --target spending_score --eval-method cv --cv-folds 3
```

## 주요 기능

- CSV/XLSX/XLS 로딩
- 데이터 개요, 결측치, 기술통계, 상관분석, 이상치 탐지
- 히스토그램, 박스플롯, 상관행렬 시각화
- 전처리 자동화
  - 수치형: 평균값 대체
  - 범주형: 최빈값 대체 + OneHotEncoder
  - 날짜형: 감지 후 경고만 남기고 제외
- 문제 유형 자동 판별
  - 숫자형 타깃: 회귀
  - 범주형 타깃: 분류
- baseline 모델 비교
  - 회귀: `LinearRegression`, `RandomForestRegressor`
  - 분류: `LogisticRegression`, `RandomForestClassifier`
- 평가 방식 선택
  - `holdout`
  - `cv`
- 리포트 생성
  - `report.md`
  - `report.html`
- 경고 및 실패 로그 생성
- best model artifact 저장
- 실행 이력 누적 기록

## CLI 옵션

대표 옵션은 아래와 같습니다.

- `--input`: 입력 데이터 파일 경로
- `--config`: YAML 설정 파일 경로
- `--target`: 모델링에 사용할 타깃 컬럼
- `--output-dir`: 결과 저장 루트 폴더
- `--report-format {md,html}`: 리포트 형식
- `--task-type {auto,regression,classification}`: 문제 유형 강제 지정
- `--eval-method {holdout,cv}`: 평가 방식
- `--cv-folds`: CV fold 수
- `--random-state`: 랜덤 시드
- `--test-size`: holdout 검증 비율

도움말:

```powershell
python run_analysis.py --help
```

## outputs 설명

기본 출력 구조는 아래와 같습니다.

```text
outputs/
|-- experiments_log.csv
`-- YYYYMMDD/
    `-- HHMMSS/
        |-- charts/
        |-- config_snapshot.json
        |-- data_summary.json
        |-- outlier_summary.csv
        |-- warnings_summary.md
        |-- warnings.json
        |-- report.md 또는 report.html
        |-- preprocessing_summary.md
        |-- model_comparison.csv
        |-- model_summary.md
        |-- best_model.joblib
        `-- model_metadata.json
```

각 파일 의미:

- `config_snapshot.json`: 실행 당시 설정값 스냅샷
- `data_summary.json`: 데이터 구조와 결측치/이상치 요약
- `outlier_summary.csv`: 수치형 컬럼별 IQR 이상치 개수와 비율
- `warnings_summary.md`, `warnings.json`: 경고와 실패 원인 기록
- `report.md` / `report.html`: 사람이 읽는 최종 분석 리포트
- `preprocessing_summary.md`: 모델링 전처리 요약
- `model_comparison.csv`: baseline 모델별 성능 비교
- `model_summary.md`: 최고 모델, 평가 방식, 전처리 요약
- `best_model.joblib`: 저장된 최고 성능 모델 파이프라인
- `model_metadata.json`: 저장 모델의 메타정보
- `experiments_log.csv`: 실행 이력 누적 로그

## Config 실행

`configs/default.yaml`을 사용하면 긴 CLI 인자 없이 실행할 수 있습니다.

```powershell
python run_analysis.py --config configs/default.yaml
python run_analysis.py --config configs/default.yaml --report-format html
```

CLI 인자와 config를 동시에 주면 CLI 값이 우선합니다.

## 테스트

```powershell
python -m pytest tests
```
