# Air Writing Web–Unity Integration

웹 디지털 트윈과 Unity 디지털 트윈을 한 컴퓨터에서 동시에 실행하기 위한 통합 모듈입니다.

## 핵심 구조

```text
ESP32
  ↓ Serial (main.py만 사용)
main.py
  ├─ WebSocket → Web Digital Twin
  └─ WebSocket → Unity Relay
                    ↓ UDP 12346
                  Unity
```

Relay는 COM 포트를 열지 않습니다. `main.py`가 보내는 `ray_hit`을 Unity용 `pointer_x`, `pointer_y`로 변환합니다.

## Yaw 드리프트

기존 단일형 Unity Bridge가 Yaw 드리프트를 “사용”한 것은 아닙니다. Yaw 드리프트는 자이로 오차가 누적되어 발생한 현상입니다. 기존 Bridge는 초기 자이로 바이어스 보정, One Euro Filter, 수동 `R` 기준점 재설정을 사용했지만 자력계가 기본 OFF라 장시간 드리프트가 남을 수 있습니다. Relay는 별도의 Yaw 보정을 하지 않으며 `main.py`의 `ray_hit`을 따라갑니다.

## 폴더 구성

```text
airwriting_unity_integration/
├─ README.md
├─ TROUBLESHOOTING.md
├─ requirements.txt
├─ relay/
│  └─ unity_bridge_from_main_ws_v3.py
└─ unity_scripts/
   ├─ UDPReceiver_AI_Integrated_v4.cs
   ├─ DemoStatusUI_AI_v3.cs
   └─ AirWritingTrailDrawer_v2.cs
```

## 설치

```bash
py -m pip install -r requirements.txt
```

## 실행 순서

```text
1. Arduino Serial Monitor와 단독 Unity Bridge 종료
2. py main.py
3. 웹 디지털 트윈 정상 동작 확인
4. py relay/unity_bridge_from_main_ws_v3.py
5. Unity 실행
```

## Unity 권장 설정

```text
Port                                12346
Apply Pointer Position              체크
Use Additional Unity Pointer Center 해제
Pointer Sensitivity                 (1, 1)
Pointer Dead Zone                   0
Position Smooth                     25
```

## 기준점 재설정

```text
1. 장갑을 정면 자세로 유지
2. 웹의 기준점 재설정
3. 같은 자세를 유지
4. Relay 콘솔에서 R
5. Unity 중앙 확인
```

`main.py`를 수정하지 않는 구조이므로 웹 기준점과 Relay 기준점은 각각 맞춥니다.

## 범위가 너무 좁거나 끝에 붙을 때

Relay 상단의 값을 조절합니다.

```python
WEB_SENSITIVITY_X = 0.30
WEB_SENSITIVITY_Y = 0.30
```

너무 빨리 끝에 붙으면 `0.25`, `0.22`로 낮추고, 너무 적게 움직이면 `0.35`로 높입니다. 한 번에 하나의 값만 변경합니다.

## 통합 실패 시 백업 모드

```text
Unity 단독 (S3): unity_bridge_final_v1.py + Unity
S2 비상: unity_bridge_s2_hand_test_v1.py + Unity
```

## GitHub 폴더 추가

GitHub는 빈 폴더만 저장하지 않습니다. 웹에서 `Add file → Create new file`을 선택하고 파일 이름에 `unity_integration/README.md`처럼 `/`를 포함해 입력하면 폴더와 파일이 함께 만들어집니다. 로컬에서는 폴더를 만든 뒤 파일을 넣고 `git add`, `git commit`, `git push` 하면 됩니다.
