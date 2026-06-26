# AirWriting Digital Twin

IMU 센서가 부착된 Air Writing 장갑의 방향 변화를 Unity 손 모델과 필기 궤적으로 실시간 시각화하는 디지털 트윈 프로젝트입니다.

> 이 프로젝트는 IMU의 가속도와 각속도를 이용해 센서의 자세와 방향 벡터를 추정하고, 이를 가상 2차원 필기 평면의 `pointer_x`, `pointer_y` 좌표로 변환합니다.  
> 실제 손의 절대 XYZ 위치나 평행이동 거리를 직접 추적하는 방식은 아닙니다.

---

## 1. Project Overview

장갑에서 수집한 IMU 및 Writing 입력을 Python Bridge에서 처리한 뒤 UDP로 Unity에 전송합니다. Unity에서는 수신한 포인터 좌표로 손 모델을 이동시키고, 검지 끝의 `PenTip` 오브젝트를 기준으로 TrailRenderer 필기 궤적을 생성합니다.

### 주요 기능

- ESP32 Serial 센서 데이터 수신
- 68 / 70 / 92 / 94 byte 패킷 파싱
- 초기 자이로 바이어스 보정
- Madgwick 기반 자세 Quaternion 추정
- Forward Vector 기반 좌우·상하 방향각 계산
- 방향각을 `-1 ~ +1` 범위의 2D 포인터로 변환
- One Euro Filter 기반 포인터 스무딩
- 버튼 입력 기반 Writing ON/OFF
- 향후 압력센서 입력 대응을 위한 `pressure_raw` 필드
- UDP 기반 Python–Unity 실시간 통신
- Unity 손 모델 이동 및 Trail 생성
- 연결·Tracking·Writing·센서 상태 UI
- Virtual Writing Area 및 중앙 기준점 표시
- AI 인식 결과 표시를 위한 UI 영역

---

## 2. System Architecture

### A. Unity 독립 실행 모드

Unity 기능을 단독으로 테스트하거나 통합 백엔드에 문제가 있을 때 사용하는 안정적인 백업 방식입니다.

```text
ESP32 Glove
    ↓ USB Serial
unity_bridge_final_v1.py
    ↓ UDP 127.0.0.1:12346
Unity Digital Twin
```

Python Bridge가 다음 과정을 모두 직접 수행합니다.

```text
Serial Receive
→ Packet Parsing
→ Sensor Validation
→ Gyro Bias Calibration
→ Madgwick Orientation Estimation
→ Direction Vector Calculation
→ Relative 2D Pointer Mapping
→ UDP JSON Transmission
```

### B. S2 Hand IMU 테스트 모드

S2 손등 IMU를 주 포인터 센서로 사용하는 테스트 방식입니다.

```text
ESP32 Glove
    ↓ USB Serial
unity_bridge_s2_hand_test_v1.py
    ↓ UDP 127.0.0.1:12346
Unity Digital Twin
```

S2와 S3의 안정성, 드리프트, 필기 반복성 및 AI 인식률을 비교할 때 사용할 수 있습니다.

### C. Web + AI + Unity 통합 모드 · Experimental

웹 백엔드와 Unity를 한 컴퓨터에서 동시에 실행하기 위한 통합 구조입니다.

```text
ESP32 Glove
    ↓ USB Serial
main.py
    ├─ WebSocket → Web Digital Twin
    ├─ AI Character Recognition
    └─ WebSocket → Unity Relay
                       ↓ UDP 12346
                 Unity Digital Twin
```

```text
unity_bridge_from_main_ws_v2.py
```

는 COM 포트를 직접 열지 않고 `main.py`가 계산한 `ray_hit`, `is_writing`, 자세 정보 및 AI 결과를 Unity용 UDP JSON으로 변환합니다.

> 통합 Relay는 현재 검증 중인 프로토타입입니다.  
> 웹과 Unity의 공통 Recenter, 감도, 포인터 범위 및 장시간 드리프트를 실제 장갑으로 추가 확인해야 합니다.

---

## 3. Important: Serial Port Conflict

다음 두 프로그램을 동시에 실행하면 안 됩니다.

```text
main.py
unity_bridge_team_review_v1_1.py
```

두 프로그램 모두 같은 ESP32 COM 포트를 직접 열기 때문에 Windows에서 다음 오류가 발생할 수 있습니다.

```text
PermissionError: Access is denied
SerialException
COM port connection failed
```

### 실행 모드별 규칙

| 실행 목적 | 실행할 Python |
|---|---|
| Unity 단독 시연 | `unity_bridge_team_review_v1_1.py` |
| S2 센서 테스트 | `unity_bridge_s2_hand_test_v1.py` |
| 웹·AI·Unity 통합 | `main.py` + `unity_bridge_from_main_ws_v2.py` |

통합 모드에서는 기존 단독 Unity Bridge를 실행하지 않습니다.

---

## 4. Sensor Configuration

| Sensor | Position | Role |
|---|---|---|
| S1 | Forearm | 전완 자세 및 보조 센서 |
| S2 | Back of Hand | 손등 방향 및 Unity 포인터 테스트 |
| S3 | Finger | 손가락 방향 및 Air Writing 포인터 |

S3 손가락 IMU의 SCL 배선은 수리되었으며 현재 정상 작동합니다.  
S2와 S3 모두 Unity 포인터로 사용할 수 있으며, 최종 주 포인터 센서는 반복 필기와 AI 인식률 비교 후 결정할 예정입니다.

---

## 5. How the Pointer Works

이 프로젝트의 포인터는 실제 손 위치가 아니라 **손가락 또는 손등이 가리키는 방향**을 나타냅니다.

```text
Accelerometer + Gyroscope
        ↓
Madgwick Filter
        ↓
Quaternion
        ↓
Forward Vector
        ↓
Horizontal / Vertical Angle
        ↓
Relative Angle from Reference
        ↓
pointer_x / pointer_y
        ↓
Virtual Writing Area
```

### 사용하는 적분

- 자이로 각속도 → 자세 Quaternion 계산: 사용
- 가속도 → 이동 속도 계산: 사용하지 않음
- 이동 속도 → 실제 위치 계산: 사용하지 않음

따라서 장갑의 방향을 유지한 채 손 전체를 좌우로 평행이동하면 포인터 변화가 작을 수 있습니다.

---

## 6. Requirements

### Hardware

- ESP32 기반 Air Writing 장갑
- S1 / S2 / S3 IMU 센서
- Writing 입력 버튼 또는 압력센서
- Windows PC
- Unity 프로젝트

### Python Packages

```bash
py -m pip install pyserial numpy
```

통합 Relay를 사용할 경우:

```bash
py -m pip install websockets
```

---

## 7. Standalone Unity Mode

### 1) ESP32 연결

Windows 장치 관리자에서 현재 COM 포트를 확인합니다.

COM 포트 번호는 연결 환경에 따라 달라질 수 있습니다.

### 2) Bridge 실행

```bash
py unity_bridge_team_review_v1_1.py
```

정상 실행 예시:

```text
[연결 성공] ESP32 Serial: COMx / 921600
[UDP 시작] 127.0.0.1:12346
[준비] 장갑을 움직이지 말고 가만히 두세요.
[상태] TRACKING
```

초기 보정 중에는 장갑을 약 4~6초 동안 움직이지 않습니다.

### 3) Unity 실행

Unity 프로젝트를 열고 Play를 누릅니다.

### 4) Recenter

Python 콘솔에 포커스를 둔 뒤:

```text
R
```

을 누릅니다.

현재 방향이 Unity 포인터의 중앙 기준으로 설정됩니다.

### 5) 종료

```text
Q
```

또는 `Ctrl + C`로 Bridge를 종료합니다.

---

## 8. S2 Hand IMU Test Mode

```bash
py unity_bridge_s2_hand_test_v1.py
```

정상 실행 시 콘솔에서 다음을 확인합니다.

```text
pointer_source=S2_HAND
```

S2 테스트 모드는 다음 작업에 사용할 수 있습니다.

- Unity UI 구성
- Trail 디자인
- 포인터 범위 조정
- 손 모델 이동 테스트
- S2·S3 안정성 비교
- 하드웨어 비상 테스트

---

## 9. Integrated Web + Unity Mode

### 1) 통합 백엔드 실행

```bash
py main.py
```

### 2) Relay 실행

```bash
py unity_bridge_from_main_ws_v2.py
```

정상 연결 예시:

```text
[연결 성공] main.py WebSocket
[Unity 중앙 설정] ray_hit=(...)
[Unity Relay] pointer=(...) writing=...
```

### 3) Web UI 실행

백엔드 설정에 따라 웹 화면을 실행합니다.

예:

```text
http://localhost:8080
```

### 4) Unity 실행

Unity는 UDP `12346`을 수신합니다.

### 5) Relay Recenter

Relay 콘솔에서:

```text
R
```

을 눌러 Unity 기준점을 재설정합니다.

> 현재 웹 Recenter와 Relay Recenter는 자동으로 완전히 동기화되지 않을 수 있습니다.  
> 통합 테스트 시 같은 장갑 자세에서 웹과 Relay의 기준점을 맞추는 것을 권장합니다.

---

## 10. Unity Inspector Recommended Settings

`UDPReceiver` 권장 설정:

```text
Port                                12346
Auto Calibrate On Start             ON
Use Additional Unity Pointer Center OFF
Apply Pointer Position              ON
Apply Hand Rotation                 OFF
Pointer Sensitivity                 (1, 1)
Pointer Dead Zone                   0
Position Smooth                     25
```

이동 범위 예시:

```text
Max Hand Move Local = (0.70, 0.50, 0)
```

축 방향이 반대일 경우 `Pointer Axis Sign`만 수정합니다.

```text
좌우 반대   → (-1, 1)
상하 반대   → (1, -1)
둘 다 반대 → (-1, -1)
```

---

## 11. Unity UI

현재 Unity 시연 화면은 다음 요소로 구성됩니다.

### System Status

- `CONNECTION`
- `TRACKING`
- `WRITING`
- `SENSOR`
- `PACKET`
- `POINTER`
- `INPUT RAW`

### Virtual Writing Area

- 손 모델
- PenTip
- Trail
- 중앙 십자선
- Center Reference
- Live Tracking 상태
- Calibration Overlay

### Additional UI

- `CLEAR CANVAS`
- `RECOGNITION RESULT`
- 작품 제목 및 설명

---

## 12. Controls

| Key / UI | Function |
|---|---|
| `R` | Python 또는 Relay 기준점 재설정 |
| `Q` | Python Bridge 종료 |
| Writing Button | Trail 생성 시작·종료 |
| `CLEAR CANVAS` | Unity Trail 삭제 |
| Unity Play | Unity Digital Twin 실행 |

---

## 13. Troubleshooting

### ESP32 연결 실패

```text
PermissionError: Access is denied
SerialException
```

확인 사항:

- 다른 Python Bridge가 실행 중인지
- `main.py`가 같은 COM 포트를 사용 중인지
- Arduino Serial Monitor가 열려 있는지
- 설정된 COM 포트가 현재 장치와 일치하는지

모든 Serial 프로그램을 종료한 후 하나의 프로그램만 실행합니다.

---

### 자이로 보정이 0%에서 진행되지 않음

확인 사항:

- 패킷 크기가 정상인지
- 선택한 센서의 가속도·자이로 값이 0인지
- 장갑이 너무 크게 움직이고 있지 않은지
- SCL / SDA 배선이 정상인지

---

### 손 모델이 이동 범위 끝에 붙음

가능한 원인:

- Yaw 드리프트 누적
- `pointer_x` 또는 `pointer_y`가 `-1` 또는 `+1`에 포화됨
- 포인터 감도가 너무 큼
- 기준점이 중앙에서 벗어남

대응:

1. Python 콘솔에서 `R` 실행
2. 장갑을 정면에 두고 기준점 재설정
3. `Pointer Sensitivity` 확인
4. Python의 포인터 각도 범위 확인
5. 통합 모드에서는 웹과 Relay 기준점 확인

---

### Play할 때마다 이동 범위가 달라짐

Unity Inspector에서 확인:

```text
Use Additional Unity Pointer Center = OFF
```

Python이 이미 상대 포인터를 생성하므로 Unity에서 추가 중앙 보정을 적용하면 안 됩니다.

---

### 실제 손 이동과 Unity 방향이 반대임

Unity의:

```text
Pointer Axis Sign
```

값을 수정합니다.

---

### 버튼을 눌러도 Trail이 생성되지 않음

확인 사항:

- UDP 상태의 `button` 또는 `pressure_raw`
- `PenTip` 연결
- `TrailObject` 프리팹 연결
- `AirWritingTrailDrawer`의 Writing 입력
- Unity Console 오류

---

### 센서가 연결됐는데 OFFLINE으로 표시됨

확인 사항:

- Python Bridge가 UDP `12346`으로 전송 중인지
- Unity `UDPReceiver` Port가 `12346`인지
- 방화벽이 로컬 UDP를 차단하는지
- 같은 포트를 사용하는 다른 프로그램이 있는지

---

## 14. Current Status

### Completed

- Unity 디지털 트윈 기본 환경
- 클릭 기반 사전 프로토타입
- Python 가상 UDP 데이터 검증
- 실제 ESP32 Serial 연동
- 92 byte 패킷 수신 및 파싱
- 버튼 기반 Writing
- 방향 기반 2D 포인터
- Unity 손 모델 이동
- TrailRenderer 필기
- 반복 Play 이동 범위 안정화
- S2 손등 IMU 테스트 모드
- S3 SCL 배선 수리 및 정상 작동
- System Status UI
- Virtual Writing Area
- Calibration Overlay
- Live Tracking 표시
- Recognition Result UI
- Clear Canvas UI

### In Progress

- S2·S3 포인터 성능 비교
- Trail 디자인 최종 조정
- AI 결과 Unity 표시
- WebSocket–UDP Relay 통합 테스트
- 웹·Unity 공통 Recenter
- 압력센서 입력 적용
- Unity Windows Build
- 장시간 안정성 테스트

---

## 15. Known Limitations

- 실제 손의 절대 XYZ 위치를 추적하지 않음
- 기울기 없는 순수 평행이동을 직접 인식하지 못함
- 자력계 기준이 없거나 불안정하면 Yaw 드리프트가 누적될 수 있음
- 포인터가 `-1 ~ +1` 범위 끝에 도달하면 Unity 손 모델이 경계에 머무를 수 있음
- 통합 Relay의 Recenter와 감도는 추가 검증이 필요함
- 실제 압력센서 임계값 및 히스테리시스는 아직 적용 중임
- AI confidence가 백엔드에서 제공되지 않는 경우 Unity에 임의로 표시하지 않음

---

## 16. Roadmap

1. S2·S3 필기 안정성 및 AI 인식률 비교
2. 최종 주 포인터 센서 선정
3. Trail 색상·두께·삭제 기능 마무리
4. `main.py`와 Relay 실제 동시 실행
5. 웹과 Unity의 방향·감도·범위 통일
6. 공통 Recenter 구조 구현
7. AI Recognition Result 실제 연동
8. 압력센서 입력 적용
9. Unity Windows Build 제작
10. 장시간 통합 안정성 테스트
11. 최종 실행 매뉴얼 및 시연 영상 제작

---

## 17. Recommended Repository Structure

```text
airwriting_DT/
├─ README.md
├─ python/
│  ├─ unity_bridge_final_v1.py
│  ├─ unity_bridge_s2_hand_test_v1.py
│  └─ unity_bridge_from_main_ws_v2.py
│
├─ UnityProject/
│  ├─ Assets/
│  │  ├─ Scripts/
│  │  │  ├─ UDPReceiver.cs
│  │  │  ├─ AirWritingTrailDrawer.cs
│  │  │  └─ DemoStatusUI.cs
│  │  ├─ Scenes/
│  │  ├─ Prefabs/
│  │  └─ Materials/
│  ├─ Packages/
│  └─ ProjectSettings/
│
└─ docs/
   ├─ architecture/
   ├─ screenshots/
   └─ troubleshooting/
```

`Library`, `Temp`, `Logs`, `Obj` 등의 Unity 자동 생성 폴더는 Git에 포함하지 않는 것을 권장합니다.

---

## 18. Demo Checklist

- [ ] ESP32 및 센서 배선 확인
- [ ] 현재 COM 포트 확인
- [ ] 장갑을 고정하고 초기 보정
- [ ] `pointer=(0.00, 0.00)` 기준 확인
- [ ] Unity UDP Port `12346` 확인
- [ ] `Use Additional Unity Pointer Center` OFF
- [ ] 손 모델 좌우·상하 방향 확인
- [ ] Writing ON/OFF 확인
- [ ] Clear Canvas 확인
- [ ] AI 결과 표시 확인
- [ ] 웹과 Unity 동시 실행 확인
- [ ] 비상용 독립 Bridge 준비

---

## Project Status

This repository is under active development for a university capstone project.
