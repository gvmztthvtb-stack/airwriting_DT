# AirWriting 설치 및 실행 안내서

이 문서는 특정 Windows 계정이나 바탕화면 경로에 의존하지 않습니다. 저장소를 원하는 폴더에 내려받은 뒤 아래 순서대로 실행하세요.

## 1. 필요한 항목

- Windows 10 또는 11 64-bit
- Python 3.12 64-bit 권장
- ESP32 AirWriting 장갑과 데이터 통신이 가능한 USB 케이블
- GitHub Releases에서 받은 Unity Windows 실행 ZIP

## 2. 저장소 준비

Git을 사용하는 경우:

```powershell
git clone https://github.com/gvmztthvtb-stack/airwriting_DT.git
cd airwriting_DT\bridge
```

GitHub에서 Source code ZIP을 받은 경우 압축을 풀고 그 안의 `bridge` 폴더로 이동합니다.

다음 세 항목이 함께 있는지 확인하세요.

```text
bridge/
├─ unity_ai_bridge_purebilstm_bestperformance.py
├─ airwriting_imu/
└─ weights/
   ├─ pure_bilstm.pt
   └─ meta.pkl
```

Bridge 파일이나 모델 파일 하나만 따로 이동하면 실행되지 않습니다.

## 3. Python 환경 설치

PowerShell에서 `bridge` 폴더로 이동한 상태에서 실행합니다.

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

설치 확인:

```powershell
.\.venv\Scripts\python.exe -c "import numpy, torch, sklearn, serial, scipy; print('AI PACKAGE OK')"
```

마지막에 `AI PACKAGE OK`가 나오면 준비가 끝난 것입니다.

## 4. 실행 순서

1. ESP32 장갑의 전원과 USB 연결을 확인합니다.
2. 장치 연결 후 3~5초 기다립니다.
3. Arduino Serial Monitor, 다른 Bridge, 같은 COM 포트를 사용하는 프로그램을 모두 종료합니다.
4. PowerShell에서 `bridge` 폴더로 이동합니다.
5. Bridge를 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\unity_ai_bridge_purebilstm_bestperformance.py
```

6. 캘리브레이션 안내가 나오면 실제 필기 자세로 장갑을 들고 4~6초 동안 최대한 움직이지 않습니다.
7. 다음 상태를 확인합니다.

```text
모델 로드 완료: pure_bilstm
ESP32 Serial 연결 성공
자이로 보정 완료
상태: TRACKING
```

8. GitHub Releases에서 받은 Unity ZIP을 완전히 압축 해제합니다.
9. 압축을 푼 폴더의 `AirWriting_Unity.exe`를 실행합니다.
10. Unity에서 연결 상태와 포인터 움직임을 확인합니다.

Bridge를 먼저 실행하고 Unity를 나중에 실행하는 순서를 권장합니다.

## 5. 기본 동작 설정

- AI 센서 입력: S2 hand mapped
- AI Y축: 반전
- 문자 확정: 필기 종료 후 0.8초
- 공백 확정: 2.0초
- Unity 전송 주소: `127.0.0.1:12346`
- Serial 속도: 921600 baud

인터넷 연결은 설치와 다운로드에만 필요합니다. 실행 중 Bridge와 Unity는 같은 컴퓨터의 localhost로 통신합니다.

## 6. 조작 키

- `R`: 현재 방향을 화면 중앙으로 재설정
- `K`: 자이로·중력·기준 자세 전체 캘리브레이션 재시작
- `D`: 기본 로그와 상세 로그 전환
- `Q`: Bridge 안전 종료

포인터 중심만 어긋난 경우에는 `R`, 포인터가 계속 흐르거나 끝으로 이동하는 경우에는 `K`를 사용합니다.

## 7. 선택 실행 옵션

기본 실행은 `--ai-s2 --ai-flip-y`와 같습니다.

```powershell
# 기본값을 명시해서 실행
.\.venv\Scripts\python.exe .\unity_ai_bridge_purebilstm_bestperformance.py --ai-s2 --ai-flip-y

# 다른 장갑에서 AI Y축 방향 비교
.\.venv\Scripts\python.exe .\unity_ai_bridge_purebilstm_bestperformance.py --ai-s2 --ai-normal-y

# S3 손가락 IMU 진단
.\.venv\Scripts\python.exe .\unity_ai_bridge_purebilstm_bestperformance.py --ai-s3 --ai-flip-y
```

최종 시연 장갑에서는 옵션 없이 실행하는 기본 설정을 권장합니다.

## 8. 종료

1. Unity를 종료합니다.
2. Bridge 터미널에서 `Q`를 누릅니다.
3. 종료 메시지를 확인한 뒤 장갑을 분리합니다.

문제가 발생하면 [문제 해결 안내서](TROUBLESHOOTING_KO.md)를 참고하세요.
