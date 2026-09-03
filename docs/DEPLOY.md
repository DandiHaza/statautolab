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

## 공개 상태 확인

현재 앱은 **공개(Public)** 상태이며, 로그인 없이 누구나 접속할 수 있습니다.

가장 확실한 확인 방법은 브라우저 시크릿 창으로 주소를 여는 것입니다.
로그인 요구 없이 앱 화면이 뜨면 정상입니다.

**터미널로 확인할 때 주의할 점이 있습니다.** Streamlit은 접속자에게 익명 세션 쿠키를
발급하려고 `/-/auth/app`으로 리다이렉트합니다. 이는 비공개 앱의 로그인 벽이 아니라
모든 방문자에게 일어나는 정상 동작이므로, 쿠키를 저장하지 않으면 공개 앱인데도
`303`만 보고 비공개로 오판하게 됩니다. 반드시 쿠키 저장소(`-c`/`-b`)를 붙여야 합니다.

```powershell
curl -s -L -c cookies.txt -b cookies.txt -o NUL -w "%{http_code}" https://statautolab.streamlit.app/
```

리다이렉트를 세 번 거쳐 최종적으로 `200`이 나오면 공개 상태입니다.
공개 범위를 바꾸려면 [share.streamlit.io](https://share.streamlit.io)에서 앱을 열고
**Settings** → **Sharing**에서 조정합니다.

## 개발 컨테이너 (GitHub Codespaces)

`.devcontainer/devcontainer.json`이 있어 Codespaces에서 바로 실행할 수 있습니다.
컨테이너가 `packages.txt`와 `requirements.txt`를 설치한 뒤 8501 포트로 앱을 띄웁니다.

다만 컨테이너 이미지가 **Python 3.11**이라 이 프로젝트의 기준 버전(3.12)과 다릅니다.
현재 코드는 3.11에서도 동작하지만 로컬·배포 환경과 버전을 맞추려면
`devcontainer.json`의 `image`를 `mcr.microsoft.com/devcontainers/python:1-3.12-bookworm`으로
바꾸면 됩니다.

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
