# 모델 성능 요약

- 타깃 컬럼: quality
- 문제 유형: 회귀
- 평가 방식: cv
- CV fold 수: 5
- 학습 데이터 수: fold별 분할
- 검증 데이터 수: fold별 분할
- 최고 성능 모델: RandomForestRegressor
- 하이퍼파라미터 탐색: 미수행 (기본 파라미터)
- best model 저장: 완료
- 모델 artifact 경로: best_model.joblib
- 모델 metadata 경로: model_metadata.json

## 모델 비교

| model                 | problem_type   |   evaluated_folds |      mae |   mae_std |       r2 |    r2_std |     rmse |   rmse_std |
|:----------------------|:---------------|------------------:|---------:|----------:|---------:|----------:|---------:|-----------:|
| LinearRegression      | regression     |                 5 | 0.506955 | 0.0368316 | 0.342415 | 0.0641863 | 0.653576 |  0.0400071 |
| RandomForestRegressor | regression     |                 5 | 0.413615 | 0.0204198 | 0.493668 | 0.0274801 | 0.573773 |  0.0213231 |

## 전처리 요약

- 선택된 독립변수: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- 수치형 컬럼: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- 범주형 컬럼: 없음
- 날짜형 컬럼: 없음
- 식별자 자동 제외: 없음
