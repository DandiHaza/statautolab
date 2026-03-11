# StatAutoLab

StatAutoLab은 CSV/XLSX 데이터를 업로드하거나 CLI로 입력해 EDA, 이상치 탐지, baseline 모델 비교, 리포트 생성을 수행하는 분석 도구입니다.

## 빠른 시작

### 1. 가상환경 및 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 웹 UI 실행

```powershell
.\.venv\Scripts\python.exe -m streamlit.web.cli run streamlit_app.py
```

브라우저에서 파일을 업로드한 뒤 종속변수와 독립변수를 선택하고 `분석 실행` 버튼을 누르면 됩니다.

### 3. CLI 실행

```powershell
python run_analysis.py --input data/examples/eda_sample.csv
```

## 새로 추가된 변수 선택 기능

- `--target`: 종속변수(타깃 컬럼)
- `--features`: 독립변수 컬럼 목록
- 예시: `--features age,income,city,visits`

지정하지 않으면 기본 규칙으로 입력 변수를 선택합니다.

- 타깃 컬럼은 자동 제외
- `id`, `customer_id`, `user_id`, `*_id` 형태 식별자 컬럼은 자동 제외
- 날짜형 컬럼은 경고 후 제외

## 웹 UI 기능

- CSV/XLSX/XLS 파일 업로드
- 업로드 후 데이터 미리보기
- 종속변수 선택
- 독립변수 멀티셀렉트
- 문제 유형 선택: `auto`, `regression`, `classification`
- 리포트 형식 선택: `md`, `html`
- 평가 방식 선택: `holdout`, `cv`
- 교차검증 fold 수 입력
- 결과 요약 확인
- 생성된 리포트와 결과 파일 다운로드

## 예제 데이터

`data/examples` 폴더에 바로 실행 가능한 샘플 데이터가 있습니다.

- `data/examples/eda_sample.csv`
- `data/examples/classification_sample.csv`
- `data/examples/regression_sample.csv`

## CLI 실행 예시

### EDA만 실행

```powershell
python run_analysis.py --input data/examples/eda_sample.csv
```

### 분류 예시

```powershell
python run_analysis.py --input data/examples/classification_sample.csv --target buy
```

### 회귀 예시

```powershell
python run_analysis.py --input data/examples/regression_sample.csv --target spending_score
```

### 독립변수 직접 선택

```powershell
python run_analysis.py --input data/examples/regression_sample.csv --target spending_score --features age,income,city,visits
```

### 교차검증 예시

```powershell
python run_analysis.py --input data/examples/regression_sample.csv --target spending_score --features age,income,city,visits --eval-method cv --cv-folds 3
```

### 설정 파일 실행

```powershell
python run_analysis.py --config configs/default.yaml
```

## 주요 옵션

- `--input`: 입력 파일 경로
- `--config`: YAML 설정 파일 경로
- `--target`: 종속변수 컬럼
- `--features`: 독립변수 컬럼 목록
- `--output-dir`: 결과 저장 루트 폴더
- `--report-format {md,html}`: 리포트 형식
- `--task-type {auto,regression,classification}`: 문제 유형 강제 지정
- `--eval-method {holdout,cv}`: 평가 방식
- `--cv-folds`: CV fold 수
- `--random-state`: 랜덤 시드
- `--test-size`: holdout 검증 비율

전체 도움말:

```powershell
python run_analysis.py --help
```

## 출력 결과

기본 출력 구조는 다음과 같습니다.

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

주요 파일 설명:

- `report.md` / `report.html`: 최종 분석 리포트
- `outlier_summary.csv`: IQR 기반 이상치 요약
- `warnings_summary.md`, `warnings.json`: 주의사항 및 경고 기록
- `model_comparison.csv`: baseline 모델 비교 결과
- `model_summary.md`: 최고 모델과 평가 방식 요약
- `best_model.joblib`: 저장된 최고 성능 모델
- `model_metadata.json`: 모델 메타정보와 실제 사용한 독립변수 목록
- `experiments_log.csv`: 실행 이력 누적 로그

## 테스트

```powershell
python -m pytest tests -q
```
