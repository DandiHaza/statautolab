# 샘플 실행 결과

`data/real/winequality-red.csv`(UCI Wine Quality, 1,599행 × 12열)를 대상으로
아래 명령을 실행한 실제 산출물입니다. 설치 없이 결과물만 훑어보고 싶을 때 참고하세요.

```powershell
python run_analysis.py --input data/real/winequality-red.csv --target quality --eval-method cv --cv-folds 5
```

- 먼저 볼 파일: **[report.md](report.md)** — EDA부터 모델 비교까지 한 번에 정리된 최종 리포트
- `best_model.joblib`(약 8MB)은 저장소 용량 문제로 제외했습니다. 위 명령을 직접 실행하면 생성됩니다.
- `config_snapshot.json`과 `model_metadata.json`의 경로 값은 기본 출력 경로(`outputs/`) 기준으로 정규화했습니다.
