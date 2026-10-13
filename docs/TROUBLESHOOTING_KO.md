# AirWriting 문제 해결 안내서

한 번에 여러 설정을 바꾸지 말고 아래 항목을 위에서부터 하나씩 확인하세요.

## Python 또는 `py` 명령을 찾을 수 없음

- Python 3.12 64-bit 설치 여부를 확인합니다.
- 설치 프로그램에서 Python Launcher 또는 PATH 등록 항목을 선택합니다.
- 설치 후 PowerShell을 완전히 닫았다가 다시 엽니다.

```powershell
py --version
```

## Python 패키지가 없다고 표시됨

반드시 저장소의 `bridge` 폴더에서 설치합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`torch` 설치에는 시간이 걸릴 수 있습니다.

## Bridge 실행 직후 파일을 찾을 수 없다고 표시됨

Bridge와 다음 폴더가 같은 `bridge` 디렉터리 안에 있어야 합니다.

```text
unity_ai_bridge_purebilstm_bestperformance.py
airwriting_imu/
weights/pure_bilstm.pt
weights/meta.pkl
```

모델과 `meta.pkl`은 반드시 같은 학습 결과 한 쌍을 사용해야 합니다.

## scikit-learn 버전 경고

이 저장소의 모델은 `scikit-learn 1.8.0`을 권장합니다. 다음 명령으로 저장소 요구 버전을 다시 설치하세요.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 사용할 수 있는 COM 포트가 없음

1. 장갑 전원이 켜졌는지 확인합니다.
2. 충전 전용이 아닌 데이터 USB 케이블을 사용합니다.
3. Windows 장치 관리자에서 `포트(COM 및 LPT)`를 확인합니다.
4. USB를 다시 연결하고 3~5초 기다린 뒤 Bridge를 재실행합니다.
5. 다른 USB 포트 또는 케이블로 시험합니다.

Bridge는 `COM7`, `COM3`을 먼저 찾은 뒤 사용 가능한 다른 COM 포트를 자동 탐색합니다.

## COM 포트 접근 거부

Arduino Serial Monitor, 다른 Bridge 또는 센서 프로그램이 같은 포트를 사용 중입니다. COM 포트를 사용하는 프로그램을 모두 종료하고 Bridge 하나만 실행하세요.

## S3 센서가 0이거나 끊김

장갑 전원을 끄고 S3 센서 케이블과 커넥터를 다시 연결합니다. 펌웨어에서 S3 데이터 자체가 0으로 들어오면 소프트웨어만으로 복구할 수 없습니다.

## 캘리브레이션 뒤 포인터가 화면 끝으로 움직임

1. `K`를 누릅니다.
2. 실제 필기 자세로 장갑을 듭니다.
3. 4~6초 동안 손과 손가락을 고정합니다.
4. `TRACKING`이 나온 뒤 움직입니다.

중심만 어긋났다면 전체 보정 대신 `R`을 누릅니다.

## Unity에서 연결되지 않음

- Bridge 터미널에 `Unity 100 Hz` 또는 UDP 시작 메시지가 표시되는지 확인합니다.
- Bridge를 먼저 실행하고 Unity를 다시 시작합니다.
- 다른 Bridge가 동시에 실행 중인지 확인합니다.
- UDP 포트 `12346`을 다른 프로그램이 사용 중인지 확인합니다.
- Windows 방화벽 알림이 나오면 개인 네트워크 통신을 허용합니다.

Bridge와 Unity가 같은 컴퓨터에서 실행될 때는 인터넷 연결이 필요하지 않습니다.

## 포인터는 움직이지만 선이 그려지지 않음

- FSR을 눌렀을 때 Bridge의 writing 또는 Pen Down 상태가 바뀌는지 확인합니다.
- FSR 커넥터와 장갑 펌웨어를 확인합니다.
- 다른 Bridge가 동시에 실행 중이지 않은지 확인합니다.

## Python에는 인식 결과가 있지만 Unity에는 표시되지 않음

1. Unity와 모든 Bridge를 종료합니다.
2. 이 저장소의 최종 Bridge 하나만 실행합니다.
3. `TRACKING`을 확인합니다.
4. Unity를 실행합니다.
5. 한 글자를 쓰고 약 1초 기다립니다.

`airwriting_imu/core/streaming.py`를 다른 폴더의 버전으로 교체하지 마세요.

## 글자가 두 번 인식됨

- Bridge가 두 개 실행 중이지 않은지 확인합니다.
- 글자 한 개가 끝난 뒤 손을 떼고 결과가 확정될 때까지 기다립니다.
- 획을 쓰는 도중 FSR 신호가 반복해서 끊기는지 확인합니다.

## 특정 글자의 실시간 인식률이 낮음

오프라인 평가 정확도와 실제 장갑 인식률은 다를 수 있습니다.

- 비슷한 크기와 일정한 획순으로 씁니다.
- 글자를 지나치게 작거나 크게 쓰지 않습니다.
- `F/E`, `D/P`처럼 비슷한 글자는 구별되는 획을 분명하게 씁니다.
- 기본 `S2 mapped + Y 반전` 설정을 먼저 사용합니다.

## Unity 실행 파일이 열리지 않음

- ZIP을 완전히 압축 해제했는지 확인합니다.
- `AirWriting_Unity.exe`만 따로 복사하지 않습니다.
- 다음 항목이 같은 폴더에 있어야 합니다.

```text
AirWriting_Unity.exe
AirWriting_Unity_Data/
MonoBleedingEdge/
D3D12/
UnityPlayer.dll
```

Windows SmartScreen이 표시되면 파일 출처와 저장소를 확인한 후 실행 여부를 결정합니다.

## 문제 보고 시 함께 기록할 정보

- Windows와 Python 버전
- 사용 중인 COM 포트
- Bridge 터미널의 마지막 오류 메시지
- `Sensors OK`, checksum error, drop 값
- Unity 연결 상태
- 문제가 발생한 순서와 사용한 키

비밀번호, 개인 토큰 또는 개인 경로가 포함된 화면은 공개 이슈에 올리지 마세요.
