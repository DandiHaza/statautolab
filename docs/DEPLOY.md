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

컨테이너 이미지는 **Python 3.12**로, 로컬 개발 환경 및 Streamlit Cloud 설정과 같은 버전입니다.

## 재배포 후 점검 목록

- [ ] CSV 업로드 → 데이터 미리보기까지 진행되는지
- [ ] **한글 컬럼명이 들어간 CSV**로 차트 축 라벨이 깨지지 않는지
      (`packages.txt`의 나눔폰트가 잡혔는지 확인하는 단계입니다)
- [ ] 타깃을 지정해 분석을 실행하고 결과 파일이 다운로드되는지
- [ ] 회귀 대시보드의 OLS 요약이 표시되는지 (statsmodels 설치 확인)

## 트러블슈팅

### 새 설정 키를 추가한 뒤 `KeyError` 가 뜬다

```
KeyError: ...
File "/mount/src/statautolab/streamlit_app.py", line NNN, in main
    value=bool(DEFAULT_SETTINGS["새키"]),
```

`app/config.py` 의 `DEFAULT_SETTINGS` 에 키를 추가하고 푸시했을 때 나타납니다.
Streamlit Cloud 가 새 커밋을 받은 뒤 메인 스크립트(`streamlit_app.py`)만 다시 읽어
실행하고, 이미 `sys.modules` 에 올라간 `app.config` 는 옛 버전 그대로 두기 때문입니다.
새 코드가 옛 딕셔너리를 조회해서 나는 오류입니다.

**진단** — GitHub 에서 두 파일이 같은 커밋인지 확인합니다. 저장소가 정상인데도 오류가
나면 배포 프로세스의 모듈 캐시 문제입니다.

```powershell
curl -s https://raw.githubusercontent.com/DandiHaza/statautolab/main/app/config.py | Select-String "새키"
```

**조치** — 앱을 재시작하면 파이썬 프로세스가 새로 뜨면서 모든 모듈이 다시 import 됩니다.
화면 우측 하단 `Manage app` → `⋮` → `Reboot app`.

방어 코드로는 막을 수 없습니다. 크래시 지점을 `.get()` 으로 감싸도 분석을 실행하는
순간 다른 모듈에서 같은 이유로 실패합니다.

## 플랫폼 제약

**파일시스템이 휘발성입니다.** 업로드 파일(`.streamlit_uploads/`)과 분석 결과(`outputs/`)는
컨테이너 재시작 시 사라집니다. 사용자가 결과를 남기려면 화면의 다운로드 버튼을 써야 합니다.
데모 용도로는 문제가 없지만, 영구 보관하려면 외부 스토리지 연동이 필요합니다.

**비활동 시 잠자기 모드로 전환됩니다.** 일정 기간 접속이 없으면 앱이 중지되고,
다음 접속자가 깨우는 동안 수십 초가 걸립니다. 삭제된 것이 아니며 자동으로 복구됩니다.

**의존성은 두 파일로 나뉘어 있습니다.** 배포 환경은 `requirements.txt`(실행용)만
설치합니다. `pytest` 같은 개발용은 `requirements-dev.txt` 에 있어 배포 이미지에 포함되지
않습니다.
