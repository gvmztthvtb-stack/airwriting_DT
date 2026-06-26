"""
Air Writing 통합 시연용 Unity Relay
===================================

버전: tp-main-ws-unity-relay-v3

역할
----
팀 공통 main.py는 수정하지 않습니다.
main.py가 WebSocket으로 보내는 ray_hit, is_writing, orientations,
streaming_text/prediction을 받아 Unity UDP JSON으로 변환합니다.

중요
----
- 이 파일은 ESP32 COM 포트를 열지 않습니다.
- 따라서 main.py와 동시에 실행할 수 있습니다.
- 통합 시연 중에는 기존 단독 Bridge와 Serial Monitor를 종료하세요.
- Relay는 Yaw 드리프트를 새로 보정하지 않습니다.
- main.py의 ray_hit이 밀리면 웹과 Unity가 같은 방향으로 밀립니다.
- Unity는 여전히 pointer_x / pointer_y(-1~+1)를 사용합니다.

실행
----
1) py main.py
2) 웹 디지털 트윈 정상 동작 확인
3) py unity_bridge_from_main_ws_v3.py
4) Unity 실행

단축키
------
R : 현재 ray_hit을 Unity 중앙 기준으로 재설정
Q : Relay 종료

문제 발생 시 대처
-----------------
1) [연결 성공]이 안 뜸
   - main.py 실행 여부 확인
   - MAIN_WS_URL과 WebSocket 포트 확인

2) 웹은 움직이는데 Unity가 안 움직임
   - Relay pointer 로그가 변하는지 확인
   - Unity UDP Port=12346
   - Apply Pointer Position 체크
   - Use Additional Unity Pointer Center 해제

3) 손 모델이 너무 빨리 끝에 붙음
   - WEB_SENSITIVITY_X/Y를 낮춤: 0.30 → 0.25 → 0.22
   - 장갑을 정면에 두고 Relay에서 R

4) 움직임이 너무 작음
   - WEB_SENSITIVITY_X/Y를 높임: 0.30 → 0.35

5) 좌우/상하가 반대
   - Unity Pointer Axis Sign에서 해당 축만 -1
   - Relay와 Unity 양쪽에서 동시에 부호를 바꾸지 않음

6) Trail이 안 그려짐
   - Relay 로그 writing=1 확인
   - writing=0이면 main.py is_writing 확인
   - writing=1이면 Unity Trail/PenTip 연결 확인

7) AI 결과가 WAITING에서 안 바뀜
   - Relay [AI 인식] 로그 확인
   - AI 통합 UDPReceiver/DemoStatusUI 적용 확인

8) 통합 모드가 실패함
   - main.py와 Relay 종료
   - unity_bridge_team_review_v1_1.py + Unity 단독 시연
   - 또는 unity_bridge_s2_hand_test_v1.py + Unity 비상 시연
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import time
from typing import Any, Optional

try:
    import websockets
except ImportError:
    print("[오류] websockets 패키지가 필요합니다.")
    print("설치: py -m pip install websockets")
    raise SystemExit(1)

try:
    import msvcrt
except ImportError:
    msvcrt = None


# ============================================================================
# 설정
# ============================================================================

MAIN_WS_URL = "ws://127.0.0.1:12347"

UNITY_IP = "127.0.0.1"
UNITY_PORT = 12346

# 웹 프론트의 main.js와 동일한 감도:
# const SENSITIVITY = 0.30
WEB_SENSITIVITY_X = 0.30
WEB_SENSITIVITY_Y = 0.30

# 웹 프론트와 동일하게 X축을 반전합니다.
OUTPUT_SIGN_X = -1.0
OUTPUT_SIGN_Y = 1.0

# Unity로 보내기 전에 최종 -1~+1 제한
POINTER_LIMIT = 1.0

# 현재 장갑이 92B Legacy이면 92, 94B 최신 펌웨어이면 94로 변경
DISPLAY_PACKET_SIZE = 92

RECONNECT_DELAY_SEC = 2.0
LOG_INTERVAL_SEC = 1.0


# ============================================================================
# 유틸리티
# ============================================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def quaternion_wxyz_to_xyzw(values: Any) -> tuple[float, float, float, float]:
    """
    main.py orientations는 [w, x, y, z].
    Unity JSON은 qx, qy, qz, qw로 전달합니다.
    """
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return 0.0, 0.0, 0.0, 1.0

    w = safe_float(values[0], 1.0)
    x = safe_float(values[1])
    y = safe_float(values[2])
    z = safe_float(values[3])
    return x, y, z, w


def check_key() -> Optional[str]:
    if msvcrt is None or not msvcrt.kbhit():
        return None
    try:
        return msvcrt.getwch().lower()
    except Exception:
        return None


# ============================================================================
# Relay
# ============================================================================

class UnityRelay:
    def __init__(self) -> None:
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.center_x: Optional[float] = None
        self.center_y: Optional[float] = None
        self.pending_recenter = False
        self.frame_count = 0

        self.pointer_x = 0.0
        self.pointer_y = 0.0
        self.is_writing = 0
        self.last_ts = 0

        self.hand_accel = (0.0, 0.0, 0.0)
        self.hand_q = (0.0, 0.0, 0.0, 1.0)
        self.finger_q = (0.0, 0.0, 0.0, 1.0)

        self.recognized_sentence = ""
        self.latest_char = ""
        self.confidence = -1.0

        self.last_log_time = 0.0
        self.raw_ray_x = 0.0
        self.raw_ray_y = 0.0
        self.saturation_start_x: Optional[float] = None
        self.saturation_start_y: Optional[float] = None
        self.saturation_warned_x = False
        self.saturation_warned_y = False
        self.running = True

    def close(self) -> None:
        self.udp.close()

    def request_recenter(self) -> None:
        self.pending_recenter = True
        print("[요청] 다음 ray_hit을 Unity 중앙 기준으로 저장합니다.")

    def update_center(self, ray_x: float, ray_y: float) -> None:
        self.center_x = ray_x
        self.center_y = ray_y
        self.pending_recenter = False
        print(
            "[Unity 중앙 설정]",
            f"ray_hit=({self.center_x:+.3f}, {self.center_y:+.3f})",
        )

    def handle_frame(self, data: dict[str, Any]) -> None:
        ray_hit = data.get("ray_hit")
        if not isinstance(ray_hit, (list, tuple)) or len(ray_hit) < 2:
            return

        ray_x = safe_float(ray_hit[0])
        ray_y = safe_float(ray_hit[1])
        self.raw_ray_x = ray_x
        self.raw_ray_y = ray_y

        self.frame_count += 1

        # 시작 시 한 번, 또는 사용자가 R을 눌렀을 때만 중앙을 갱신합니다.
        # 실행 중 자동으로 중심이 다시 바뀌지 않도록 합니다.
        if (
            self.center_x is None
            or self.center_y is None
            or self.pending_recenter
        ):
            self.update_center(ray_x, ray_y)

        relative_x = ray_x - float(self.center_x)
        relative_y = ray_y - float(self.center_y)

        self.pointer_x = clamp(
            OUTPUT_SIGN_X * relative_x * WEB_SENSITIVITY_X,
            -POINTER_LIMIT,
            POINTER_LIMIT,
        )
        self.pointer_y = clamp(
            OUTPUT_SIGN_Y * relative_y * WEB_SENSITIVITY_Y,
            -POINTER_LIMIT,
            POINTER_LIMIT,
        )

        self._check_saturation()

        self.is_writing = 1 if bool(data.get("is_writing", False)) else 0
        self.last_ts = safe_int(data.get("ts"), int(time.time() * 1000))

        raw_sensors = data.get("raw_sensors", {})
        if isinstance(raw_sensors, dict):
            s2 = raw_sensors.get("s2", {})
            if isinstance(s2, dict):
                self.hand_accel = (
                    safe_float(s2.get("ax")),
                    safe_float(s2.get("ay")),
                    safe_float(s2.get("az")),
                )

        orientations = data.get("orientations", {})
        if isinstance(orientations, dict):
            self.hand_q = quaternion_wxyz_to_xyzw(
                orientations.get("hand")
            )
            self.finger_q = quaternion_wxyz_to_xyzw(
                orientations.get("finger")
            )

        self.send_unity()

    def _check_saturation(self) -> None:
        """포인터가 범위 끝에 오래 붙어 있으면 해결 방법을 출력합니다."""
        now = time.time()
        self.saturation_start_x, self.saturation_warned_x = self._check_axis(
            "X", self.pointer_x, self.saturation_start_x,
            self.saturation_warned_x, now
        )
        self.saturation_start_y, self.saturation_warned_y = self._check_axis(
            "Y", self.pointer_y, self.saturation_start_y,
            self.saturation_warned_y, now
        )

    @staticmethod
    def _check_axis(
        axis: str,
        value: float,
        started: Optional[float],
        warned: bool,
        now: float,
    ) -> tuple[Optional[float], bool]:
        if abs(value) < 0.95:
            return None, False
        if started is None:
            return now, False
        if not warned and now - started >= 1.0:
            print(
                f"[범위 경고] {axis} pointer={value:+.2f} | "
                "R로 중앙을 다시 잡거나 감도를 낮추세요."
            )
            return started, True
        return started, warned

    def handle_ai_message(self, data: dict[str, Any]) -> None:
        message_type = str(data.get("type", ""))

        if message_type == "streaming_text":
            sentence = str(data.get("sentence", ""))
            latest_char = str(data.get("latest_char", ""))

            self.recognized_sentence = sentence

            if latest_char == "<ERASE>":
                self.latest_char = sentence[-1:] if sentence else ""
            elif latest_char.strip():
                self.latest_char = latest_char

            # 현재 main.py streaming_text에는 confidence가 없습니다.
            self.confidence = safe_float(data.get("confidence"), -1.0)
            self.send_unity()
            print(
                "[AI 인식]",
                f"latest='{self.latest_char}'",
                f"sentence='{self.recognized_sentence}'",
            )

        elif message_type == "prediction":
            label = str(data.get("label", ""))
            if label:
                self.latest_char = label
                self.recognized_sentence = label
                self.confidence = safe_float(data.get("confidence"), -1.0)
                self.send_unity()
                print("[AI Prediction]", label)

    def send_unity(self) -> None:
        hand_qx, hand_qy, hand_qz, hand_qw = self.hand_q
        finger_qx, finger_qy, finger_qz, finger_qw = self.finger_q

        payload = {
            "type": "glove",
            "ts": int(self.last_ts),
            "calibrated": 1,

            "pointer_x": float(self.pointer_x),
            "pointer_y": float(self.pointer_y),
            "pointer_angle_x_deg": 0.0,
            "pointer_angle_y_deg": 0.0,

            "button": int(self.is_writing),
            "pressure_raw": int(self.is_writing),
            "writing_input_mode": "MAIN_WS",

            "packet_size": int(DISPLAY_PACKET_SIZE),
            "pointer_source": "MAIN_WS_RAY_HIT",
            "tracking_state": "TRACKING",
            "transport": "WS_TO_UDP_RELAY",

            "hand_ax": float(self.hand_accel[0]),
            "hand_ay": float(self.hand_accel[1]),
            "hand_az": float(self.hand_accel[2]),

            "hand_qx": float(hand_qx),
            "hand_qy": float(hand_qy),
            "hand_qz": float(hand_qz),
            "hand_qw": float(hand_qw),

            "finger_qx": float(finger_qx),
            "finger_qy": float(finger_qy),
            "finger_qz": float(finger_qz),
            "finger_qw": float(finger_qw),

            "recognized_text": self.recognized_sentence,
            "recognized_char": self.latest_char,
            "recognition_confidence": float(self.confidence),
        }

        self.udp.sendto(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            (UNITY_IP, UNITY_PORT),
        )

        now = time.time()
        if now - self.last_log_time >= LOG_INTERVAL_SEC:
            center_x = self.center_x if self.center_x is not None else 0.0
            center_y = self.center_y if self.center_y is not None else 0.0
            print(
                "[Unity Relay]",
                f"ray=({self.raw_ray_x:+.3f},{self.raw_ray_y:+.3f})",
                f"center=({center_x:+.3f},{center_y:+.3f})",
                f"pointer=({self.pointer_x:+.2f},{self.pointer_y:+.2f})",
                f"writing={self.is_writing}",
                f"result='{self.latest_char or '-'}'",
            )
            self.last_log_time = now


async def keyboard_loop(relay: UnityRelay) -> None:
    while relay.running:
        key = check_key()

        if key == "r":
            relay.request_recenter()
        elif key == "q":
            print("[종료] Q 입력")
            relay.running = False
            return

        await asyncio.sleep(0.03)


async def websocket_loop(relay: UnityRelay) -> None:
    while relay.running:
        try:
            print(f"[연결 시도] {MAIN_WS_URL}")

            async with websockets.connect(
                MAIN_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=2,
                max_size=2**20,
            ) as websocket:
                print("[연결 성공] main.py WebSocket")
                relay.frame_count = 0
                relay.center_x = None
                relay.center_y = None

                while relay.running:
                    try:
                        raw_message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=1.0,
                        )
                    except asyncio.TimeoutError:
                        # 연결은 유지하면서 키 입력과 종료 상태를 계속 확인합니다.
                        continue

                    try:
                        data = json.loads(raw_message)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(data, dict):
                        continue

                    if "ray_hit" in data:
                        relay.handle_frame(data)
                    elif data.get("type") in {
                        "streaming_text",
                        "prediction",
                    }:
                        relay.handle_ai_message(data)
        except (
            OSError,
            ConnectionError,
            websockets.ConnectionClosed,
        ) as error:
            if relay.running:
                relay.is_writing = 0
                relay.send_unity()
                print(f"[WebSocket 끊김] {error}")
                print(f"{RECONNECT_DELAY_SEC:.0f}초 후 재연결합니다...")
                await asyncio.sleep(RECONNECT_DELAY_SEC)
        except Exception as error:
            if relay.running:
                relay.is_writing = 0
                relay.send_unity()
                print(f"[Relay 오류] {type(error).__name__}: {error}")
                await asyncio.sleep(RECONNECT_DELAY_SEC)


async def async_main() -> None:
    relay = UnityRelay()

    print("=" * 68)
    print("Air Writing main.py → Unity 통합 Relay")
    print(f"WebSocket: {MAIN_WS_URL}")
    print(f"Unity UDP: {UNITY_IP}:{UNITY_PORT}")
    print("R=Unity 중앙 재설정 / Q=종료")
    print("=" * 68)

    try:
        await asyncio.gather(
            websocket_loop(relay),
            keyboard_loop(relay),
        )
    finally:
        relay.running = False
        relay.close()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[종료] 사용자 중단")
