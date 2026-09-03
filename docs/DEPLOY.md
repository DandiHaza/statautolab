# 배포 메모 — Streamlit Community Cloud

**배포 주소: <https://statautolab.streamlit.app/>**

`main` 브랜치에 푸시하면 자동으로 재배포됩니다. 별도 배포 명령은 없습니다.

## 배포 구성

| 항목 | 값 |
|:---|:---|
| 플랫폼 | Streamlit Community Cloud |
| 저장소 / 브랜치 | `DandiHaza/statautolab` / `main` |
| 진입점 | `streamlit_app.py` (저장소 루트) |
| pip 패키지 | `requirements.txt` |
| apt 패키지 | `packages.txt` — 차트 한글 표시를 위한 `fonts-nanum` |
| headless 렌더링 | `app/visualization.py`의 `matplotlib.use("Agg")` |

## 공개 설정 확인

앱이 비공개 상태면 링크를 연 사람이 Streamlit 로그인 화면으로 넘어가 내용을 볼 수 없습니다.
포트폴리오 링크로 공유한다면 공개로 두어야 합니다.

1. [share.streamlit.io](https://share.streamlit.io)에서 앱을 엽니다.
2. 우측 상단 **Settings** → **Sharing**으로 들어갑니다.
3. 공개 범위를 **Public**(링크가 있는 누구나 조회 가능)으로 지정합니다.

브라우저 시크릿 창으로 주소를 열어 로그인 요구 없이 화면이 뜨면 정상입니다.
터미널로 확인하려면 아래가 `200`이어야 하고, `303`이면 아직 비공개입니다.

```powershell
curl -s -o NUL -w "%{http_code}" https://statautolab.streamlit.app/
```

## 재배포 후 점검 목록

- [ ] CSV 업로드 → 데이터 미리보기까지 진행되는지
- [ ] **한글 컬럼명이 들어간 CSV**로 차트 축 라벨이 깨지지 않는지
      (`packages.txt`의 나눔폰트가 잡혔는지 확인하는 단계입니다)
- [ ] 타깃을 지정해 분석을 실행하고 결과 파일이 다운로드되는지
- [ ] 회귀 대시보드의 OLS 요약이 표시되는지 (statsmodels 설치 확인)

## 플랫폼 제약

**파일시스템이 휘발성입니다.** 업로드 파일(`.streamlit_uploads/`)과 분석 결과(`outputs/`)는
컨테이너 재시작 시 사라집니다. 사용자가 결과를 남기려면 화면의 다운로드 버튼을 써야 합니다.
데모 용도로는 문제가 없지만, 영구 보관하려면 외부 스토리지 연동이 필요합니다.

**비활동 시 잠자기 모드로 전환됩니다.** 일정 기간 접속이 없으면 앱이 중지되고,
다음 접속자가 깨우는 동안 수십 초가 걸립니다. 삭제된 것이 아니며 자동으로 복구됩니다.

**`requirements.txt`에 `pytest`가 포함되어 있습니다.** 실행에는 불필요한 테스트 의존성이라
배포 이미지가 조금 커집니다. 신경 쓰인다면 테스트 의존성을 별도 파일로 분리하면 되지만,
동작에는 영향이 없습니다.
