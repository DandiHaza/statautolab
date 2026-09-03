# 배포 가이드 — Streamlit Community Cloud

이 저장소는 Streamlit Community Cloud에 그대로 올라가도록 준비되어 있습니다.
무료이고, 공개 GitHub 저장소면 별도 설정 없이 배포됩니다.

## 준비 상태

| 요건 | 상태 |
|:---|:---|
| 공개 GitHub 저장소 | `DandiHaza/statautolab` |
| 진입점 `streamlit_app.py` (저장소 루트) | 있음 — Community Cloud 기본값과 이름이 같아 따로 지정할 필요가 없습니다 |
| `requirements.txt` (pip 패키지) | 있음 |
| `packages.txt` (apt 패키지) | 있음 — 차트 한글 표시를 위한 `fonts-nanum` |
| headless 렌더링 | `app/visualization.py`에서 `matplotlib.use("Agg")` 설정 |
| 절대경로 의존 | 없음 (전부 상대경로) |

## 배포 절차

1. [share.streamlit.io](https://share.streamlit.io) 접속 후 **GitHub 계정으로 로그인**합니다.
   저장소 읽기 권한을 요청하며, 이 단계는 본인 계정으로 직접 진행해야 합니다.
2. **Create app** → **Deploy a public app from GitHub**를 선택합니다.
3. 아래 값을 입력합니다.
   - Repository: `DandiHaza/statautolab`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. **Advanced settings**에서 Python 버전을 **3.12**로 맞춥니다.
   (로컬 개발 환경과 동일한 버전입니다.)
5. **Deploy**를 누르면 몇 분 안에 `https://<앱이름>.streamlit.app` 형태의 주소가 발급됩니다.

## 배포 후 확인할 것

- [ ] CSV 업로드 → 데이터 미리보기까지 진행되는지
- [ ] **한글 컬럼명이 들어간 CSV**를 올려 차트 축 라벨이 □□□로 깨지지 않는지
      (`packages.txt`의 나눔폰트가 제대로 잡혔는지 확인하는 단계입니다)
- [ ] 타깃을 지정해 분석을 실행하고 결과 파일이 다운로드되는지
- [ ] 회귀 대시보드의 OLS 요약이 표시되는지 (statsmodels 설치 확인)

## 알아둘 제약

**파일시스템이 휘발성입니다.** 업로드 파일(`.streamlit_uploads/`)과 분석 결과(`outputs/`)는
컨테이너 재시작 시 사라집니다. 사용자가 결과를 남기려면 화면의 다운로드 버튼을 써야 합니다.
데모 용도로는 문제가 없지만, 결과를 영구 보관하려면 외부 스토리지 연동이 필요합니다.

**비활동 시 잠자기 모드로 전환됩니다.** 일정 기간 접속이 없으면 앱이 중지되고,
다음 접속자가 깨우는 동안 수십 초가 걸립니다. 포트폴리오 링크로 공유할 때 참고하세요.

**`requirements.txt`에 `pytest`가 포함되어 있습니다.** 실행에는 불필요한 테스트 의존성이라
배포 이미지가 조금 커집니다. 신경 쓰인다면 테스트 의존성을 별도 파일로 분리하면 되지만,
동작에는 영향이 없습니다.

## 배포한 뒤

발급받은 URL을 README 상단에 추가하면 됩니다. 배지 형태로 넣으려면:

```markdown
[![Live Demo](https://img.shields.io/badge/Live%20Demo-statautolab-FF4B4B?logo=streamlit&logoColor=white)](https://<발급받은주소>.streamlit.app)
```
