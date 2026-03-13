# StatAutoLab

StatAutoLab은 CSV/XLSX 데이터를 업로드하거나 CLI로 입력해 EDA, 회귀/분류 분석, 리포트 생성까지 수행할 수 있는 초보자 친화형 데이터 분석 도구입니다.

## 빠른 시작

### 1. 가상환경 생성 및 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Streamlit 웹 UI 실행

```powershell
.\.venv\Scripts\python.exe -m streamlit.web.cli run streamlit_app.py
```

웹 화면에서 CSV/XLSX 파일을 업로드한 뒤 타깃 컬럼, 독립변수, 문제 유형, 모델을 선택하고 `분석 실행`을 누르면 됩니다.

### 3. CLI 실행

```powershell
python run_analysis.py --input data/examples/eda_sample.csv
```

## 현재 웹 화면 흐름

1. CSV/XLSX/XLS 파일 업로드
2. 아래 내용을 바로 확인
   - 데이터 미리보기
   - 데이터 개요
   - 결측치 요약
   - 이상치 요약
   - 히스토그램
   - 상관행렬
3. 아래 옵션 선택
   - 분석 모드 / 예측 모드
   - 타깃 컬럼
   - 독립변수 컬럼
   - 문제 유형
   - 사용할 모델
   - 리포트 형식
4. 분석 실행
5. 결과 확인
   - 회귀 분석 대시보드 또는 예측 결과
   - 다중공선성 점검
   - 경고 메시지
   - 리포트 미리보기 / 결과 파일 다운로드

## 주요 기능

- CSV/XLSX/XLS 로딩
- 자동 EDA
  - 데이터 개요
  - 결측치 요약
  - 이상치 요약
  - 히스토그램
  - 상관행렬
- 분석 모드 / 예측 모드 분리
- 회귀 분석 대시보드
  - OLS 요약
  - 회귀식
  - 회귀계수표
  - 잔차 플롯
  - 다중공선성 점검
- 기본 예측 모델
  - 회귀: `LinearRegression`, `RandomForestRegressor`
  - 분류: `LogisticRegression`, `RandomForestClassifier`
- Markdown / HTML 리포트 생성
- 실험 로그 저장

## 변수 선택 규칙

독립변수를 따로 지정하지 않으면 기본적으로 다음 규칙을 적용합니다.

- 타깃 컬럼은 자동 제외
- `id`, `customer_id`, `user_id`, `*_id` 형태 식별자 컬럼은 자동 제외
- 날짜형 컬럼은 감지 후 경고와 함께 제외

웹 UI에서는 추가로 다음을 보여줍니다.

- 상관이 높은 변수쌍 안내
- VIF 계산 결과
- 제거 추천 변수와 제거 버튼

## 예제 데이터

프로젝트에 포함된 예제 파일:

- `data/examples/eda_sample.csv`
- `data/examples/classification_sample.csv`
- `data/examples/regression_sample.csv`
- `data/real/winequality-red.csv`

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

### config 파일 실행

```powershell
python run_analysis.py --config configs/default.yaml
```

## 주요 CLI 옵션

- `--input`: 입력 파일 경로
- `--config`: YAML 설정 파일 경로
- `--target`: 타깃 컬럼
- `--features`: 쉼표로 구분한 독립변수 컬럼 목록
- `--output-dir`: 결과 저장 루트 폴더
- `--report-format {md,html}`: 리포트 형식
- `--task-type {auto,regression,classification}`: 문제 유형
- `--eval-method {holdout,cv}`: 평가 방식
- `--cv-folds`: 교차검증 fold 수
- `--random-state`: 랜덤 시드
- `--test-size`: holdout 검증 비율

전체 도움말:

```powershell
python run_analysis.py --help
```

## 출력 결과 구조

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

주요 결과 파일:

- `report.md` / `report.html`: 최종 분석 리포트
- `outlier_summary.csv`: IQR 기반 이상치 요약
- `warnings_summary.md`, `warnings.json`: 경고 로그
- `model_comparison.csv`: baseline 모델 비교 결과
- `model_summary.md`: 선택된 모델 요약
- `best_model.joblib`: 저장된 모델 아티팩트
- `model_metadata.json`: 모델 메타데이터
- `experiments_log.csv`: 실행 이력 로그

## 업데이트 기록

- 상세 기록은 [UPDATE_LOG.md](c:\StatAutoLab\UPDATE_LOG.md) 참고

## 테스트

```powershell
python -m pytest tests -q
```
