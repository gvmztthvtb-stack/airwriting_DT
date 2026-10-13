# AirWriting Digital Twin

IMU 센서가 부착된 장갑의 움직임을 Unity 디지털 트윈으로 표시하고, Pure BiLSTM 모델로 영문 필기를 인식하는 졸업작품입니다.

## 최종 시연 구성

```text
ESP32 AirWriting Glove
        │ USB Serial
        ▼
Python AI Bridge
  - 센서 패킷 수신
  - 자세·포인터 계산
  - Pure BiLSTM 문자 인식
        │ UDP 127.0.0.1:12346
        ▼
Unity Digital Twin
  - 손/포인터 시각화
  - 필기 궤적과 인식 결과 표시
```

최종 시연은 다음 파일 조합을 사용합니다.

- Bridge: `bridge/unity_ai_bridge_purebilstm_bestperformance.py`
- Model: `bridge/weights/pure_bilstm.pt`
- Metadata: `bridge/weights/meta.pkl`
- Firmware: `firmware/airwriting_esp32_v3/airwriting_esp32_v3.ino` (v3.4)
- Unity Windows build: GitHub Releases에서 배포

`pure_bilstm.pt`와 `meta.pkl`은 같은 학습 결과에서 생성된 한 쌍이므로 반드시 함께 사용해야 합니다.

## 빠른 실행

### 1. Python 준비

Windows와 Python 3.12 64-bit를 권장합니다.

```powershell
cd bridge
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 장갑과 Bridge 실행

1. ESP32 장갑을 USB로 연결합니다.
2. Arduino Serial Monitor와 다른 Bridge를 모두 종료합니다.
3. 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\unity_ai_bridge_purebilstm_bestperformance.py
```

초기 보정 안내가 나오면 실제 필기 자세로 장갑을 들고 4~6초간 움직이지 않습니다. 터미널에 `TRACKING`이 표시되면 다음 단계로 진행합니다.

### 3. Unity 실행

Releases에서 Windows 데모 ZIP을 내려받아 압축을 푼 뒤 `AirWriting_Unity.exe`를 실행합니다. EXE만 따로 옮기면 실행되지 않으므로 압축 해제된 폴더 전체를 유지해야 합니다.

### 4. 주요 키

키 입력은 현재 선택된 창에 전달됩니다.

Unity 실행 화면을 클릭한 뒤 사용하는 키:

- `F2`: 설정 패널 열기/닫기
- `Esc`: 열려 있는 설정 패널 닫기
- `F3`: 발표용 인식 결과 화면 표시/숨기기
- `Backspace`: 발표 화면에서 마지막 인식 문자 삭제
- `Delete`: 발표 화면의 인식 결과 전체 삭제
- `C`: Unity에 그려진 필기 궤적 전체 지우기
- `R`: Unity 손·포인터 기준 위치 재설정

Bridge 터미널을 클릭한 뒤 사용하는 키:

- `R`: Bridge 포인터 중앙 재설정
- `K`: 전체 캘리브레이션 재시작
- `D`: 상세 로그 전환
- `Q`: Bridge 안전 종료

Windows 디스플레이를 `확장`으로 설정하고 모니터가 2대 이상 연결되어 있으면 발표 화면이 두 번째 모니터에 자동으로 나타납니다. 모니터가 1대일 때는 `F3`로 같은 화면을 미리 볼 수 있습니다.

## 최종 설정

- AI 입력: S2 mapped
- AI 궤적: Y축 반전
- 글자 확정 대기: 0.8초
- 띄어쓰기 대기: 2.0초
- Unity UDP: `127.0.0.1:12346`
- 선호 COM 포트: `COM7`, `COM3` 후 자동 탐색

## 모델 성능

선정된 체크포인트는 저장된 균형 평가 세트 2,600개 중 2,578개를 맞혀 99.15%를 기록했습니다. 이는 오프라인 평가 결과이며, 실제 장갑에서는 센서 장착 위치, 보정 상태, 필기 습관과 신호 품질에 따라 정확도가 달라질 수 있습니다.

자세한 모델 정보와 해시는 [`bridge/weights/MODEL_CARD.md`](bridge/weights/MODEL_CARD.md)를 참고하세요.

## 저장소 구성

```text
bridge/                         최종 Python Bridge와 런타임 모듈, 모델
firmware/                       ESP32 장갑 펌웨어
docs/                           시연 및 문제 해결 문서
releases/                       Unity 실행본 배포 안내
Unity_Integration/              이전 Unity 통합 실험 자료
*.cs, *.unity, legacy bridge    기존 개발 기록(추후 archive 정리 대상)
```

현재 루트에 남아 있는 Unity 장면, C# 스크립트와 예전 Bridge는 기존 저장소의 개발 기록입니다. 최종 시연 시에는 반드시 위의 `bridge/` 경로를 사용하세요.

## 포함하지 않은 항목

- 원본 학습 데이터와 캡처 데이터
- `.npy` 평가 캐시와 학습 중간 산출물
- 이전 모델, CTC/Hybrid 모델과 스냅샷
- Python 가상환경과 `__pycache__`
- Unity `BurstDebugInformation_DoNotShip`
- 로컬 로그와 개인 환경 설정

대용량 Unity 실행본은 Git 기록에 넣지 않고 GitHub Release 자산으로 제공합니다.

## 문서

- [설치 및 실행 안내서](docs/INSTALL_AND_RUN_KO.md)
- [문제 해결 안내서](docs/TROUBLESHOOTING_KO.md)

## 공개 범위

이 저장소에는 최종 시연에 필요한 코드, 펌웨어와 추론 모델만 포함합니다. 학습 데이터는 팀 권리와 데이터 라이선스를 별도로 확인하기 전까지 공개하지 않습니다.

별도의 `LICENSE`가 추가되기 전까지 저장소 공개는 소스 사용·재배포 권한을 자동으로 부여하지 않습니다.
