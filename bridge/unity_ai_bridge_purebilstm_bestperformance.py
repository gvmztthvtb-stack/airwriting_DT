"""
Air Writing → Unity AI Bridge
====================================

파일 배치:
    이 파일을 사용 프로젝트의 main.py, airwriting_imu/, weights/와
    같은 폴더에 넣습니다.

실행 방법:
    python unity_ai_bridge.py
        별도 옵션 없이 실행해도 아래 권장 설정이 기본 적용됩니다.
        AI 센서 = S2(손등) mapped
        AI 궤적 Y = 반전
        즉, 현재 기본 실행은 --ai-s2 --ai-flip-y와 같습니다.

    python unity_ai_bridge.py --ai-flip-y
        S2(손등) mapped 센서 + AI 궤적 Y 반전입니다.
        옵션을 생략한 기본 실행과 같은 설정입니다.

    python unity_ai_bridge.py --ai-normal-y
        일반 Y 방향으로 AI 궤적을 입력합니다.

    python unity_ai_bridge.py --ai-flip-y --ai-s2
        S2(손등) mapped 센서 + Y 반전. 기본값을 명시해서 실행하는
        표현이며, 옵션 없는 실행과 동작이 같습니다.

    python unity_ai_bridge.py --ai-normal-y --ai-s2
        S2(손등) mapped 센서 + 일반 Y 방향입니다.

    python unity_ai_bridge.py --ai-s2 --ai-flip-y --capture-label A
        실제 장갑으로 쓴 A의 AI 입력을 진단용 JSON으로 저장합니다.
        결과는 live_captures_pure/ 폴더에 저장됩니다.

진단용 센서 변경:
    --ai-s2 : AI에 S2(손등) mapped 센서를 사용합니다. 기본값입니다.
    --ai-s3 : AI에 S3(손가락) mapped 센서를 사용합니다.

주의:
    main.py, 다른 Bridge, Serial Monitor와 동시에 실행하지 마세요.
    한 번에 하나의 프로그램만 장갑 COM 포트를 열 수 있습니다.
    Unity 손·포인터·Trail은 어떤 AI 옵션을 사용해도 기존 S3를 사용합니다.

캘리브레이션 자세와 포인터 급이동:
    시작 보정과 K 전체 재보정 중에는 장갑을 단순히 멈추는 것뿐 아니라,
    실제 필기할 때의 편한 정면 자세로 유지해야 합니다.

    움직이는 순간의 센서값은 보정 샘플에서 제외되므로 짧은 흔들림만으로
    보정이 바로 망가지지는 않습니다. 하지만 마지막에 장갑을 옆으로 크게
    기울인 채 가만히 있으면 그 기울어진 자세가 중력·포인터 기준으로 저장되어
    특정 방향에서 포인터가 갑자기 크게 움직일 수 있습니다.

    포인터가 갑자기 화면 끝으로 이동하거나 작은 동작에 과하게 반응하면:
        1. 실제 필기할 때처럼 장갑을 정면의 편한 자세로 둡니다.
        2. K를 눌러 전체 캘리브레이션을 다시 시작합니다.
        3. READY가 표시될 때까지 같은 자세로 가만히 유지합니다.
        4. 이후 중심만 다시 잡고 싶을 때는 R을 사용합니다.

    R은 현재 방향을 화면 중앙으로만 다시 지정하며 자이로·중력 보정은
    다시 하지 않습니다. 이상 움직임이 계속되면 R보다 K를 사용합니다.

AI 인식을 위해 같은 프로젝트의 airwriting_imu/와 weights/가 필요합니다.
해당 main.py가 정상 동작하는 Python 환경을 그대로 사용하세요.

설치:
    py -m pip install pyserial

실행:
    python unity_ai_bridge.py --ai-s2 --ai-flip-y

모델 교체:
    동일한 pure_bilstm 구조를 재학습한 경우 weights/의
    pure_bilstm.pt와 meta.pkl을 같은 학습 결과 한 쌍으로 교체합니다.
    입력 feature 순서·개수나 모델 구조가 바뀌지 않았다면
    이 Bridge는 수정할 필요가 없습니다.

처리 흐름
---------
ESP32 Serial
    → 68 / 70 / 92 / 94-byte Packet Framing
    → S1(전완), S2(손등), S3(손가락) 데이터 분리
    → 초기 정지 상태 자이로 바이어스 보정
    → S3 Madgwick 자세 추정
    → 선택적 Magnetometer Yaw 미세 보정 (기본 OFF)
    → 손가락 Forward Vector의 좌우/상하 각도 계산
    → 시작 자세 대비 상대 좌표
    → One Euro Filter
    → Unity UDP JSON 전송

중요한 한계
-----------
이 코드는 IMU의 '방향 변화'를 2D 포인터로 변환합니다.
외부 카메라나 위치 센서가 없으므로, 장갑 각도를 유지한 완전한 평행이동을
실제 cm 단위 위치로 추적할 수는 없습니다. 대신 Air Writing에 필요한
손가락 방향 기반 궤적을 넓고 부드럽게 표시하도록 설계했습니다.
Magnetometer를 사용하지 않는 경우 장시간 사용 시 Yaw 드리프트가
누적될 수 있으므로, 현재 방향 기준점 재설정(R)을 함께 제공합니다.

──────────────────────────────────────────────────────────────────────────────
현재 버튼 → 추후 압력센서 변경 방법
──────────────────────────────────────────────────────────────────────────────

현재 펌웨어:
    패킷의 writing_raw 1바이트가 0 또는 1
    WRITING_INPUT_MODE = "BUTTON"

압력센서로 바꾸는 가장 쉬운 방법(권장):
    ESP32 펌웨어에서 ADC 압력값을 임계값으로 판정한 뒤,
    기존 button 자리에 0/1만 그대로 전송합니다.

    예:
        int pressure = analogRead(PRESSURE_PIN);
        packet.button = pressure >= PRESSURE_THRESHOLD ? 1 : 0;

    이 방식은 Python/Unity 코드를 바꿀 필요가 없습니다.
    WRITING_INPUT_MODE = "BUTTON"을 그대로 사용합니다.

압력 원시값을 0~255 한 바이트로 보내는 방법:
    1. ESP32에서 ADC 0~4095를 0~255로 변환해 기존 button 자리에 전송
       packet.button = map(pressure, 0, 4095, 0, 255);

    2. 아래 설정을 변경
       WRITING_INPUT_MODE = "PRESSURE_U8"
       PRESSURE_ON_THRESHOLD = 실제 테스트값
       PRESSURE_OFF_THRESHOLD = ON보다 낮은 값

    ON/OFF 임계값을 다르게 둔 이유는 압력값 경계에서 Trail이 빠르게
    켜졌다 꺼지는 현상을 막기 위한 히스테리시스입니다.

압력 원시값을 uint16(0~4095)로 직접 보내는 경우:
    패킷 길이와 struct 포맷이 바뀌므로 PACKET FORMAT 영역에서
    새로운 struct.Struct 포맷과 parse 함수를 추가해야 합니다.
    실제 펌웨어의 패킷 구조가 확정된 뒤 수정해야 합니다.

──────────────────────────────────────────────────────────────────────────────
Unity 권장 초기 설정
──────────────────────────────────────────────────────────────────────────────

UDPReceiver:
    Port                    = 12346
    Apply Hand Rotation     = 우선 해제
    Apply Pointer Position  = 체크
    Pointer Sensitivity     = (1, 1)
    Pointer Dead Zone       = 0
    Max Hand Move Local     = (0.55, 0.40, 0)
    Position Smooth         = 25~35

Pointer Axis Sign:
    방향이 반대인 축만 -1로 바꾸세요.
    처음에는 (1, 1)로 시작합니다.

Trail:
    Trail Time              = 20~30
    Min Vertex Distance     = 0.003~0.005
    Delete Delay            = 8~15초

단축키
------
R : 현재 손가락 방향만 Unity 화면 중앙으로 빠르게 재설정
K : 자이로·중력·포인터 전체 캘리브레이션을 처음부터 재시작
Q : Bridge 종료(터미널 창에 포커스가 있을 때만 동작)
D : 기본/상세 로그 전환(터미널 창에 포커스가 있을 때만 동작)
"""

from __future__ import annotations

import json
import math
import socket
import statistics
import struct
import sys
import time
import ctypes
import warnings
from importlib import metadata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[오류] pyserial 패키지가 필요합니다.")
    print("설치 명령: py -m pip install pyserial")
    raise SystemExit(1)

try:
    import msvcrt
except ImportError:
    msvcrt = None


# ============================================================================
# 사용자 설정
# ============================================================================

BRIDGE_VERSION = "tp-unity-ai-bridge-v2.1-console-ui"

SERIAL_PORT = "AUTO"
PREFERRED_PORTS = ("COM7", "COM3")
BAUD_RATE = 921600

UDP_IP = "127.0.0.1"
UDP_PORT = 12346

SAMPLE_RATE = 85.0

# 새 펌웨어는 체크섬을 사용합니다.
# AUTO는 체크섬이 맞으면 검증하고, 기존 92B 펌웨어처럼 맞지 않는 경우에도
# Header/Footer가 정상이라면 수신하되 콘솔에 경고합니다.
CHECKSUM_MODE = "AUTO"  # "STRICT", "AUTO", "OFF"

# Unity에서 현재 사용 중인 좌표 방향에 맞춘 센서 축 변환입니다.
# 웹 백엔드가 raw-axis 모드를 사용한다면 두 출력의 부호가 다를 수 있으므로
# 통합 시 실제 장갑으로 축 방향을 확인해야 합니다.
AXIS_REMAP = True

# 보정
GYRO_CALIBRATION_SAMPLES = 380
REFERENCE_SETTLE_SAMPLES = 200
CALIBRATION_MAX_GYRO_NORM = 0.35
CALIBRATION_MIN_ACCEL_NORM = 0.20
CALIBRATION_MAX_ACCEL_NORM = 40.0
# 이 시간 동안 계속 흔들리면 잘못 모인 샘플을 버리고 자동 재시작합니다.
# 센서 필터 수치는 바꾸지 않고, 보정 샘플 수집만 제어합니다.
CALIBRATION_UNSTABLE_RESTART_SEC = 3.0
# 아주 짧은 센서 잡음에는 경고를 띄우지 않고, 실제 움직임이 지속될 때만
# 안내합니다. 보정 판정 기준이나 추적 필터 값은 변경하지 않습니다.
CALIBRATION_MOTION_NOTICE_DELAY_SEC = 0.4

# 화면 끝에 도달하는 상대 방향각.
# 너무 작으면 빨리 끝에 걸리고, 너무 크면 이동이 작게 느껴집니다.
POINTER_X_RANGE_DEG = 70.0
POINTER_Y_RANGE_DEG = 50.0

# 작은 움직임 확대. 1보다 작을수록 작은 움직임이 더 잘 보입니다.
POINTER_RESPONSE_GAMMA_X = 1.12
POINTER_RESPONSE_GAMMA_Y = 1.10

POINTER_GAIN_X = 0.90
POINTER_GAIN_Y = 0.92

# Python에서 작은 떨림 제거. Unity Dead Zone은 0으로 두는 것을 권장합니다.
POINTER_DEAD_ZONE = 0.007

# 출력 부호. 최종 방향은 Unity Pointer Axis Sign에서 맞추는 것을 권장합니다.
OUTPUT_SIGN_X = 1.0
OUTPUT_SIGN_Y = 1.0

# 최신 파이프라인의 One Euro Filter 기본값
ONE_EURO_MIN_CUTOFF = 0.7
ONE_EURO_BETA = 1.0
ONE_EURO_D_CUTOFF = 1.0

# Madgwick
MADGWICK_BETA = 0.05

# Magnetometer가 정상일 때만 Yaw를 아주 약하게 보정
USE_MAGNETOMETER = False
MAG_REFERENCE_NORM_MIN = 10.0
MAG_REFERENCE_NORM_MAX = 120.0
MAG_REFERENCE_MAX_ABS_DIP_DEG = 70.0
MAG_NORM_TOLERANCE = 0.30
MAG_DIP_TOLERANCE_DEG = 15.0
MAG_RATE_LIMIT = 50.0
MAG_CORRECTION_GAIN = 0.005
MAG_MAX_CORRECTION_RAD_S = 0.002

# 손가락 IMU가 일시적으로 비정상이면 손등으로 바꾸지 않고 마지막 좌표 유지
SHORT_SENSOR_HOLD_SEC = 0.20
LONG_SENSOR_LOST_SEC = 0.80
S3_AUTO_RECONNECT_SEC = 2.0

# Writing 입력
WRITING_INPUT_MODE = "BUTTON"  # "BUTTON" 또는 "PRESSURE_U8"
BUTTON_DEBOUNCE_FRAMES = 2
FSR_HOST_LONGPRESS_SEC = 3.0

# PRESSURE_U8 모드에서 사용
PRESSURE_ON_THRESHOLD = 90
PRESSURE_OFF_THRESHOLD = 65
PRESSURE_DEBOUNCE_FRAMES = 2

LOG_INTERVAL_SEC = 1.0
SUMMARY_LOG_INTERVAL_SEC = 2.0
RECOMMENDED_SKLEARN_VERSION = "1.8.0"

# 기존 웹 인식 파이프라인과 동일한 AI 설정입니다.
# 이 Bridge를 프로젝트 루트(main.py, airwriting_imu, weights가 있는 폴더)에
# 넣고 실행해야 해당 폴더의 pure_bilstm.pt와 meta.pkl을 불러옵니다.
AI_CHAR_TIMEOUT_SEC = 0.8
AI_SPACE_TIMEOUT_SEC = 2.0
AI_TRAJECTORY_SCALE = 4.0
AI_ONE_EURO_MIN_CUTOFF = 1.5
if "--ai-s2" in sys.argv and "--ai-s3" in sys.argv:
    print("[오류] --ai-s2와 --ai-s3를 동시에 사용할 수 없습니다.")
    raise SystemExit(2)
AI_SENSOR_SOURCE = (
    "S3_MAPPED"
    if "--ai-s3" in sys.argv
    else "S2_MAPPED"
)

# 장갑/좌표계별 AI 궤적 Y 방향 선택:
#   기본 또는 --ai-flip-y : Y 반전 (현재 사용자 장갑)
#   --ai-normal-y          : Y 정상 (다른 장갑 비교용)
if "--ai-flip-y" in sys.argv and "--ai-normal-y" in sys.argv:
    print("[오류] --ai-flip-y와 --ai-normal-y를 동시에 사용할 수 없습니다.")
    raise SystemExit(2)
AI_FLIP_Y = "--ai-normal-y" not in sys.argv


def command_line_value(flag: str) -> Optional[str]:
    """명령행에서 `--flag value` 형식의 값을 읽습니다."""
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None

    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        print(f"[오류] {flag} 뒤에 값을 입력해야 합니다.")
        raise SystemExit(2)
    return sys.argv[index + 1]


CAPTURE_LABEL = command_line_value("--capture-label")
if CAPTURE_LABEL is not None:
    CAPTURE_LABEL = CAPTURE_LABEL.strip().upper()
    if len(CAPTURE_LABEL) != 1 or CAPTURE_LABEL not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        print("[오류] --capture-label에는 A~Z 한 글자만 입력하세요.")
        raise SystemExit(2)
CAPTURE_ENABLED = CAPTURE_LABEL is not None


# ============================================================================
# 작은 벡터 / Quaternion 유틸리티
# ============================================================================

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]  # [w, x, y, z]


# ============================================================================
# AI 결과 상태
# ============================================================================

latest_recognized_text = ""
latest_recognized_char = ""
latest_recognition_confidence = 0.0


class DiagnosticCapture:
    """실시간 추론에 들어가는 프레임을 글자 단위 JSON으로 보관합니다."""

    def __init__(
        self,
        output_dir: Path,
        label: str,
        timeout_sec: float,
        *,
        model_path: Path,
        meta_path: Path,
    ) -> None:
        self.output_dir = output_dir
        self.label = label
        self.timeout_sec = timeout_sec
        self.model_path = model_path
        self.meta_path = meta_path
        self.strokes: list[list[dict[str, float]]] = []
        self.last_writing = False
        self.last_write_time: Optional[float] = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, frame: dict[str, float], is_writing: bool) -> None:
        now = time.time()

        if is_writing:
            if not self.last_writing:
                self.strokes.append([])
            self.strokes[-1].append(dict(frame))
            self.last_write_time = now
        elif (
            self.strokes
            and self.last_write_time is not None
            and now - self.last_write_time >= self.timeout_sec
        ):
            self.save()
            self.strokes = []
            self.last_write_time = None

        self.last_writing = is_writing

    def save(self) -> None:
        sample_count = sum(len(stroke) for stroke in self.strokes)
        if sample_count == 0:
            return

        now = datetime.now()
        filename = (
            f"{self.label}_{now:%Y%m%d_%H%M%S}_"
            f"{now.microsecond // 1000:03d}.json"
        )
        path = self.output_dir / filename
        payload = {
            "label": self.label,
            "source": "unity_ai_bridge_live_diagnostic",
            "captured_at": now.isoformat(timespec="milliseconds"),
            "bridge_version": BRIDGE_VERSION,
            "model_type": "pure_bilstm",
            "model_path": str(self.model_path),
            "meta_path": str(self.meta_path),
            "ai_sensor_source": AI_SENSOR_SOURCE,
            "ai_flip_y": AI_FLIP_Y,
            "strokes": self.strokes,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[진단 저장]",
            f"label={self.label}",
            f"strokes={len(self.strokes)}",
            f"samples={sample_count}",
            f"file={path}",
        )

    def reset(self) -> None:
        """중앙 재설정 전에 작성 중이던 진단 획을 폐기합니다."""
        self.strokes = []
        self.last_writing = False
        self.last_write_time = None


def restore_raw_axis(vector: Vector3) -> Vector3:
    """Unity 포인터용 축 변환을 되돌려 기존 main.py의 raw 축과 맞춥니다."""
    if AXIS_REMAP:
        return -vector[0], -vector[1], vector[2]
    return vector


def handle_ai_text_updated(
    sentence: str,
    char: str,
    confidence: float,
) -> None:
    """StreamingInference 결과를 Unity UI용 상태로 보관합니다."""
    global latest_recognized_text
    global latest_recognized_char
    global latest_recognition_confidence

    latest_recognized_text = sentence
    latest_recognition_confidence = float(confidence)

    if not sentence and not char:
        latest_recognized_char = ""
        latest_recognition_confidence = 0.0
    elif char == "<ERASE>":
        stripped = sentence.rstrip()
        latest_recognized_char = stripped[-1] if stripped else ""
    elif char and not char.isspace():
        latest_recognized_char = char[-1]

    # 공백 추가나 내부 초기화 때 같은 결과 카드를 반복 출력하지 않습니다.
    if char and not char.isspace() and char != "<ERASE>":
        confidence_percent = latest_recognition_confidence * 100.0
        if confidence_percent >= 90.0:
            confidence_level = "● HIGH"
        elif confidence_percent >= 70.0:
            confidence_level = "● MEDIUM"
        else:
            confidence_level = "▲ LOW · 다시 작성 권장"

        print()
        print("┌─ AI RECOGNITION " + "─" * 36 + "┐")
        print(f"│  Result       {latest_recognized_char:<35}│")
        print(
            f"│  Confidence   {confidence_percent:5.1f}%  "
            f"{confidence_level:<21}│"
        )
        print("└" + "─" * 54 + "┘")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def is_finite_vector(vector: Vector3) -> bool:
    return all(math.isfinite(value) for value in vector)


def vector_norm(vector: Vector3) -> float:
    x, y, z = vector
    return math.sqrt(x * x + y * y + z * z)


def vector_subtract(a: Vector3, b: Vector3) -> Vector3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def median_vector(samples: list[Vector3]) -> Vector3:
    if not samples:
        return 0.0, 0.0, 0.0

    return (
        float(statistics.median(sample[0] for sample in samples)),
        float(statistics.median(sample[1] for sample in samples)),
        float(statistics.median(sample[2] for sample in samples)),
    )


def mean_angle(angles: list[float]) -> float:
    if not angles:
        return 0.0

    sin_sum = sum(math.sin(angle) for angle in angles)
    cos_sum = sum(math.cos(angle) for angle in angles)
    return math.atan2(sin_sum, cos_sum)


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_normalize(q: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in q))
    if not math.isfinite(norm) or norm < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def quaternion_from_accel(accel: Vector3) -> Quaternion:
    """현재 중력 방향을 world +Z로 맞추는 초기 sensor→world 자세."""
    norm = vector_norm(accel)
    if not math.isfinite(norm) or norm < 1e-8:
        return 1.0, 0.0, 0.0, 0.0

    source = (
        accel[0] / norm,
        accel[1] / norm,
        accel[2] / norm,
    )
    target: Vector3 = (0.0, 0.0, 1.0)
    dot = clamp(
        source[0] * target[0]
        + source[1] * target[1]
        + source[2] * target[2],
        -1.0,
        1.0,
    )

    if dot > 1.0 - 1e-8:
        return 1.0, 0.0, 0.0, 0.0
    if dot < -1.0 + 1e-8:
        return 0.0, 1.0, 0.0, 0.0

    cross = (
        source[1] * target[2] - source[2] * target[1],
        source[2] * target[0] - source[0] * target[2],
        source[0] * target[1] - source[1] * target[0],
    )
    return quaternion_normalize(
        (1.0 + dot, cross[0], cross[1], cross[2])
    )


def rotate_vector(q: Quaternion, vector: Vector3) -> Vector3:
    """Quaternion [w,x,y,z]의 active rotation."""
    w, x, y, z = q
    vx, vy, vz = vector

    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,

        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,

        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def quaternion_to_unity_xyzw(q: Quaternion) -> tuple[float, float, float, float]:
    return q[1], q[2], q[3], q[0]


# ============================================================================
# 패킷
# ============================================================================

HEADER = 0xAA
FOOTER = 0x55


@dataclass(slots=True)
class SensorFrame:
    timestamp_ms: int
    wrist_accel: Vector3
    wrist_gyro: Vector3
    finger_accel: Vector3
    finger_gyro: Vector3
    finger_mag: Vector3
    writing_raw: int
    packet_size: int
    hand_accel: Optional[Vector3] = None
    hand_gyro: Optional[Vector3] = None
    seq: int = -1


class PacketParser:
    """
    지원 형식:
      68B Dual Legacy
      70B Dual + seq
      92B Tri-Node Legacy
      94B Tri-Node + seq
    """

    FORMAT_70 = struct.Struct("<BHI6f9fBBB")
    FORMAT_94 = struct.Struct("<BHI6f6f9fBBB")
    FORMAT_68 = struct.Struct("<BI6f9fBBB")
    FORMAT_92 = struct.Struct("<BI6f6f9fBBB")

    def __init__(self) -> None:
        self.total_packets = 0
        self.valid_packets = 0
        self.checksum_errors = 0
        self.format_errors = 0
        self.dropped_packets = 0
        self.last_seq = -1
        self.auto_checksum_warning_printed = False

    @staticmethod
    def _checksum_ok(data: bytes) -> bool:
        checksum = 0
        for value in data[1:-2]:
            checksum ^= value
        return checksum == data[-2]

    def _accept_checksum(self, data: bytes) -> bool:
        if CHECKSUM_MODE == "OFF":
            return True

        valid = self._checksum_ok(data)
        if valid:
            return True

        self.checksum_errors += 1

        if CHECKSUM_MODE == "STRICT":
            return False

        if not self.auto_checksum_warning_printed:
            print(
                "[체크섬 경고] 패킷 Header/Footer는 정상이지만 체크섬이 맞지 않습니다."
            )
            print(
                "[AUTO 모드] 기존 펌웨어 호환을 위해 계속 수신합니다. "
                "최신 펌웨어 적용 후 STRICT 사용을 권장합니다."
            )
            self.auto_checksum_warning_printed = True

        return True

    @staticmethod
    def _remap(vector: Vector3) -> Vector3:
        if AXIS_REMAP:
            return -vector[0], -vector[1], vector[2]
        return vector

    def _track_seq(self, seq: int) -> None:
        if self.last_seq >= 0:
            expected = (self.last_seq + 1) & 0xFFFF
            if seq != expected:
                dropped = (seq - expected) & 0xFFFF
                if dropped < 1000:
                    self.dropped_packets += dropped
        self.last_seq = seq

    def parse(self, data: bytes) -> Optional[SensorFrame]:
        self.total_packets += 1

        try:
            if len(data) == self.FORMAT_94.size:
                frame = self._parse_94(data)
            elif len(data) == self.FORMAT_92.size:
                frame = self._parse_92(data)
            elif len(data) == self.FORMAT_70.size:
                frame = self._parse_70(data)
            elif len(data) == self.FORMAT_68.size:
                frame = self._parse_68(data)
            else:
                self.format_errors += 1
                return None
        except (struct.error, ValueError, IndexError):
            self.format_errors += 1
            return None

        if frame is not None:
            self.valid_packets += 1
        return frame

    def _base_valid(self, data: bytes, values: tuple) -> bool:
        if values[0] != HEADER or values[-1] != FOOTER:
            self.format_errors += 1
            return False
        return self._accept_checksum(data)

    def _parse_94(self, data: bytes) -> Optional[SensorFrame]:
        values = self.FORMAT_94.unpack(data)
        if not self._base_valid(data, values):
            return None

        self._track_seq(int(values[1]))

        return SensorFrame(
            timestamp_ms=int(values[2]),
            wrist_accel=self._remap(tuple(map(float, values[3:6]))),
            wrist_gyro=self._remap(tuple(map(float, values[6:9]))),
            hand_accel=self._remap(tuple(map(float, values[9:12]))),
            hand_gyro=self._remap(tuple(map(float, values[12:15]))),
            finger_accel=self._remap(tuple(map(float, values[15:18]))),
            finger_gyro=self._remap(tuple(map(float, values[18:21]))),
            finger_mag=tuple(map(float, values[21:24])),
            writing_raw=int(values[24]),
            packet_size=94,
            seq=int(values[1]),
        )

    def _parse_92(self, data: bytes) -> Optional[SensorFrame]:
        values = self.FORMAT_92.unpack(data)
        if not self._base_valid(data, values):
            return None

        return SensorFrame(
            timestamp_ms=int(values[1]),
            wrist_accel=self._remap(tuple(map(float, values[2:5]))),
            wrist_gyro=self._remap(tuple(map(float, values[5:8]))),
            hand_accel=self._remap(tuple(map(float, values[8:11]))),
            hand_gyro=self._remap(tuple(map(float, values[11:14]))),
            finger_accel=self._remap(tuple(map(float, values[14:17]))),
            finger_gyro=self._remap(tuple(map(float, values[17:20]))),
            finger_mag=tuple(map(float, values[20:23])),
            writing_raw=int(values[23]),
            packet_size=92,
        )

    def _parse_70(self, data: bytes) -> Optional[SensorFrame]:
        values = self.FORMAT_70.unpack(data)
        if not self._base_valid(data, values):
            return None

        self._track_seq(int(values[1]))

        return SensorFrame(
            timestamp_ms=int(values[2]),
            wrist_accel=self._remap(tuple(map(float, values[3:6]))),
            wrist_gyro=self._remap(tuple(map(float, values[6:9]))),
            finger_accel=self._remap(tuple(map(float, values[9:12]))),
            finger_gyro=self._remap(tuple(map(float, values[12:15]))),
            finger_mag=tuple(map(float, values[15:18])),
            writing_raw=int(values[18]),
            packet_size=70,
            seq=int(values[1]),
        )

    def _parse_68(self, data: bytes) -> Optional[SensorFrame]:
        values = self.FORMAT_68.unpack(data)
        if not self._base_valid(data, values):
            return None

        return SensorFrame(
            timestamp_ms=int(values[1]),
            wrist_accel=self._remap(tuple(map(float, values[2:5]))),
            wrist_gyro=self._remap(tuple(map(float, values[5:8]))),
            finger_accel=self._remap(tuple(map(float, values[8:11]))),
            finger_gyro=self._remap(tuple(map(float, values[11:14]))),
            finger_mag=tuple(map(float, values[14:17])),
            writing_raw=int(values[17]),
            packet_size=68,
        )


def extract_packets(buffer: bytearray) -> list[bytes]:
    packets: list[bytes] = []
    sizes = (94, 92, 70, 68)

    while len(buffer) >= min(sizes):
        header_index = buffer.find(bytes([HEADER]))

        if header_index < 0:
            buffer.clear()
            break

        if header_index > 0:
            del buffer[:header_index]

        matched = False

        for size in sizes:
            if len(buffer) < size:
                continue
            if buffer[size - 1] != FOOTER:
                continue

            packets.append(bytes(buffer[:size]))
            del buffer[:size]
            matched = True
            break

        if matched:
            continue

        if len(buffer) >= max(sizes):
            del buffer[0]
        else:
            break

    return packets


# ============================================================================
# Madgwick
# ============================================================================

class MadgwickFilter:
    def __init__(self, beta: float, sample_rate: float) -> None:
        self.beta = beta
        self.default_dt = 1.0 / sample_rate
        self.q: Quaternion = (1.0, 0.0, 0.0, 0.0)

    def reset(self, initial_q: Optional[Quaternion] = None) -> None:
        self.q = (
            initial_q
            if initial_q is not None
            else (1.0, 0.0, 0.0, 0.0)
        )

    def update_imu(
        self,
        accel: Vector3,
        gyro: Vector3,
        dt: Optional[float] = None,
    ) -> Quaternion:
        step_dt = dt if dt is not None else self.default_dt

        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel

        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if accel_norm < 1e-12:
            return self._integrate_gyro(gyro, step_dt)

        ax /= accel_norm
        ay /= accel_norm
        az /= accel_norm

        two_q0 = 2.0 * q0
        two_q1 = 2.0 * q1
        two_q2 = 2.0 * q2
        two_q3 = 2.0 * q3
        four_q0 = 4.0 * q0
        four_q1 = 4.0 * q1
        four_q2 = 4.0 * q2
        eight_q1 = 8.0 * q1
        eight_q2 = 8.0 * q2

        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        s0 = (
            four_q0 * q2q2
            + two_q2 * ax
            + four_q0 * q1q1
            - two_q1 * ay
        )
        s1 = (
            four_q1 * q3q3
            - two_q3 * ax
            + 4.0 * q0q0 * q1
            - two_q0 * ay
            - four_q1
            + eight_q1 * q1q1
            + eight_q1 * q2q2
            + four_q1 * az
        )
        s2 = (
            4.0 * q0q0 * q2
            + two_q0 * ax
            + four_q2 * q3q3
            - two_q3 * ay
            - four_q2
            + eight_q2 * q1q1
            + eight_q2 * q2q2
            + four_q2 * az
        )
        s3 = (
            4.0 * q1q1 * q3
            - two_q1 * ax
            + 4.0 * q2q2 * q3
            - two_q2 * ay
        )

        step_norm = math.sqrt(
            s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3
        )
        if step_norm > 1e-12:
            s0 /= step_norm
            s1 /= step_norm
            s2 /= step_norm
            s3 /= step_norm

        q_dot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz) - self.beta * s0
        q_dot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy) - self.beta * s1
        q_dot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx) - self.beta * s2
        q_dot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx) - self.beta * s3

        self.q = quaternion_normalize(
            (
                q0 + q_dot0 * step_dt,
                q1 + q_dot1 * step_dt,
                q2 + q_dot2 * step_dt,
                q3 + q_dot3 * step_dt,
            )
        )
        return self.q

    def _integrate_gyro(
        self,
        gyro: Vector3,
        dt: float,
    ) -> Quaternion:
        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro

        self.q = quaternion_normalize(
            (
                q0 + 0.5 * (-q1 * gx - q2 * gy - q3 * gz) * dt,
                q1 + 0.5 * (q0 * gx + q2 * gz - q3 * gy) * dt,
                q2 + 0.5 * (q0 * gy - q1 * gz + q3 * gx) * dt,
                q3 + 0.5 * (q0 * gz + q1 * gy - q2 * gx) * dt,
            )
        )
        return self.q


# ============================================================================
# Adaptive Magnetometer Yaw 보정
# ============================================================================

class AdaptiveMagYaw:
    def __init__(self) -> None:
        self.ref_norm: Optional[float] = None
        self.ref_dip: Optional[float] = None
        self.ref_heading: Optional[float] = None

        self.prev_mag: Optional[Vector3] = None
        self.trust_ema = 0.0

    def calibrate(
        self,
        mag_samples: list[Vector3],
        current_q: Quaternion,
    ) -> None:
        if not USE_MAGNETOMETER:
            self.ref_norm = None
            self.ref_dip = None
            self.ref_heading = None
            print("[MagFusion] 비활성화 — 자력계 Yaw 보정을 사용하지 않습니다.")
            return

        valid_samples = [
            sample
            for sample in mag_samples
            if is_finite_vector(sample) and vector_norm(sample) > 1e-6
        ]

        if len(valid_samples) < 6:
            self.ref_norm = None
            self.ref_dip = None
            self.ref_heading = None
            print(
                "[MagFusion] 유효한 자기장 샘플이 부족합니다. "
                "Yaw는 자이로 바이어스 보정만 사용합니다."
            )
            return

        norms = [vector_norm(sample) for sample in valid_samples]
        dips = [
            math.degrees(
                math.atan2(
                    sample[2],
                    math.hypot(sample[0], sample[1]) + 1e-8,
                )
            )
            for sample in valid_samples
        ]

        candidate_norm = float(statistics.median(norms))
        candidate_dip = float(statistics.median(dips))

        plausible_reference = (
            MAG_REFERENCE_NORM_MIN
            <= candidate_norm
            <= MAG_REFERENCE_NORM_MAX
            and abs(candidate_dip)
            <= MAG_REFERENCE_MAX_ABS_DIP_DEG
        )

        if not plausible_reference:
            self.ref_norm = None
            self.ref_dip = None
            self.ref_heading = None
            print(
                "[MagFusion] 기준 자기장 거부",
                f"norm={candidate_norm:.2f}",
                f"dip={candidate_dip:.1f}°",
            )
            return

        self.ref_norm = candidate_norm
        self.ref_dip = candidate_dip
        self.ref_heading = self._heading(valid_samples[-1], current_q)
        self.prev_mag = valid_samples[-1]

        print(
            "[MagFusion] 활성화",
            f"norm={self.ref_norm:.2f}",
            f"dip={self.ref_dip:.1f}°",
        )

    @staticmethod
    def _heading(
        mag: Vector3,
        q: Quaternion,
    ) -> float:
        world_mag = rotate_vector(q, mag)
        return math.atan2(world_mag[1], world_mag[0])

    def correct_gyro(
        self,
        mag: Vector3,
        gyro: Vector3,
        current_q: Quaternion,
        dt: float,
    ) -> Vector3:
        if (
            not USE_MAGNETOMETER
            or self.ref_norm is None
            or self.ref_dip is None
            or self.ref_heading is None
            or not is_finite_vector(mag)
        ):
            return gyro

        mag_norm = vector_norm(mag)
        if mag_norm < 1e-6:
            return gyro

        norm_ratio = mag_norm / max(self.ref_norm, 1e-6)
        norm_score = max(
            0.0,
            1.0
            - abs(norm_ratio - 1.0) / MAG_NORM_TOLERANCE,
        )

        current_dip = math.degrees(
            math.atan2(
                mag[2],
                math.hypot(mag[0], mag[1]) + 1e-8,
            )
        )
        dip_score = max(
            0.0,
            1.0
            - abs(current_dip - self.ref_dip)
            / MAG_DIP_TOLERANCE_DEG,
        )

        rate_score = 1.0
        if self.prev_mag is not None and dt > 1e-6:
            delta = vector_subtract(mag, self.prev_mag)
            rate = vector_norm(delta) / dt
            rate_score = max(0.0, 1.0 - rate / MAG_RATE_LIMIT)

        self.prev_mag = mag

        trust = min(norm_score, dip_score, rate_score)
        alpha = 0.5 if trust < self.trust_ema else 0.05
        self.trust_ema = alpha * trust + (1.0 - alpha) * self.trust_ema

        if self.trust_ema <= 0.5:
            return gyro

        heading = self._heading(mag, current_q)
        yaw_error = wrap_angle(heading - self.ref_heading)

        correction = -yaw_error * MAG_CORRECTION_GAIN * self.trust_ema
        correction = clamp(
            correction,
            -MAG_MAX_CORRECTION_RAD_S,
            MAG_MAX_CORRECTION_RAD_S,
        )

        return gyro[0], gyro[1], gyro[2] + correction


# ============================================================================
# Ray Projection
# ============================================================================

class RayProjection:
    def __init__(self) -> None:
        # 최신 RayProjection과 동일한 검지 센서 장착 보정
        tilt = math.radians(15.0)
        self.forward_local: Vector3 = (
            math.sin(tilt),
            -math.cos(tilt),
            0.0,
        )

    def project(
        self,
        q: Quaternion,
    ) -> tuple[float, float]:
        forward = rotate_vector(q, self.forward_local)

        horizontal = math.atan2(
            forward[0],
            -forward[1],
        )

        vertical = math.atan2(
            forward[2],
            math.hypot(forward[0], forward[1]),
        )

        return horizontal, vertical


# ============================================================================
# One Euro Filter
# ============================================================================

class LowPassFilter:
    def __init__(self) -> None:
        self.filtered: Optional[float] = None
        self.raw: Optional[float] = None
        self.alpha = 1.0

    def set_alpha(self, alpha: float) -> None:
        self.alpha = clamp(alpha, 0.0, 1.0)

    def filter(self, value: float) -> float:
        if self.filtered is None:
            self.filtered = value
        else:
            self.filtered = (
                self.alpha * value
                + (1.0 - self.alpha) * self.filtered
            )

        self.raw = value
        return self.filtered

    def reset(self, value: Optional[float] = None) -> None:
        self.filtered = value
        self.raw = value


class OneEuroFilter:
    def __init__(
        self,
        frequency: float,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
    ) -> None:
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff

        self.value_filter = LowPassFilter()
        self.derivative_filter = LowPassFilter()
        self.last_value: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        dt = 1.0 / max(self.frequency, 1e-6)
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: float, dt: float) -> float:
        if dt > 1e-6:
            self.frequency = 1.0 / dt

        derivative = 0.0
        if self.last_value is not None:
            derivative = (
                value - self.last_value
            ) * self.frequency

        self.last_value = value

        self.derivative_filter.set_alpha(
            self._alpha(self.derivative_cutoff)
        )
        filtered_derivative = self.derivative_filter.filter(derivative)

        cutoff = (
            self.min_cutoff
            + self.beta * abs(filtered_derivative)
        )

        self.value_filter.set_alpha(self._alpha(cutoff))
        return self.value_filter.filter(value)

    def reset(self, value: Optional[float] = None) -> None:
        self.value_filter.reset(value)
        self.derivative_filter.reset(0.0)
        self.last_value = value


# ============================================================================
# 좌표 응답
# ============================================================================

def shape_pointer(
    normalized: float,
    dead_zone: float,
    gamma: float,
    gain: float,
) -> float:
    """
    데드존 경계에서 갑자기 점프하지 않는 연속 응답.
    tanh soft limiter로 화면 끝에서 급격히 가로막히는 느낌을 줄입니다.
    """
    magnitude = abs(normalized)

    if magnitude <= dead_zone:
        return 0.0

    continuous = (
        magnitude - dead_zone
    ) / max(1.0 - dead_zone, 1e-6)

    curved = continuous ** gamma
    soft = math.tanh(1.15 * curved) / math.tanh(1.15)

    return math.copysign(
        clamp(soft * gain, 0.0, 1.0),
        normalized,
    )


# ============================================================================
# Writing 입력
# ============================================================================

class WritingInput:
    def __init__(self) -> None:
        self.state = 0
        self.candidate = 0
        self.candidate_count = 0

    def _debounce(
        self,
        desired: int,
        frames: int,
    ) -> int:
        desired = 1 if desired else 0

        if desired == self.state:
            self.candidate = desired
            self.candidate_count = 0
            return self.state

        if desired != self.candidate:
            self.candidate = desired
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= frames:
            self.state = desired
            self.candidate_count = 0

        return self.state

    def update(self, raw_value: int) -> int:
        if WRITING_INPUT_MODE == "PRESSURE_U8":
            if self.state:
                desired = raw_value >= PRESSURE_OFF_THRESHOLD
            else:
                desired = raw_value >= PRESSURE_ON_THRESHOLD

            return self._debounce(
                desired,
                PRESSURE_DEBOUNCE_FRAMES,
            )

        desired = raw_value > 0
        return self._debounce(
            int(desired),
            BUTTON_DEBOUNCE_FRAMES,
        )


# ============================================================================
# Serial
# ============================================================================

def choose_serial_port() -> str:
    if SERIAL_PORT != "AUTO":
        return SERIAL_PORT

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise serial.SerialException(
            "사용 가능한 COM 포트가 없습니다."
        )

    esp32_vids = {0x10C4, 0x1A86, 0x0403}
    vid_matches = [
        port.device
        for port in ports
        if port.vid in esp32_vids
    ]
    if vid_matches:
        return vid_matches[0]

    names = [port.device for port in ports]
    for preferred in PREFERRED_PORTS:
        if preferred in names:
            return preferred

    non_com1 = [
        port.device
        for port in ports
        if port.device.upper() != "COM1"
    ]

    return non_com1[0] if non_com1 else ports[0].device


def open_serial() -> serial.Serial:
    while True:
        try:
            port = choose_serial_port()
            connection = serial.Serial(
                port,
                BAUD_RATE,
                timeout=1,
                inter_byte_timeout=0.001,
            )
            print(
                f"[연결 성공] ESP32 Serial: "
                f"{port} / {BAUD_RATE}"
            )
            return connection
        except serial.SerialException as error:
            print(f"[연결 실패] {error}")
            print("3초 후 다시 시도합니다...")
            time.sleep(3)


def check_console_key() -> Optional[str]:
    if msvcrt is None or not msvcrt.kbhit():
        return None

    try:
        return msvcrt.getwch().lower()
    except Exception:
        return None


_global_r_was_down = False
_global_k_was_down = False


def global_recenter_key_pressed() -> bool:
    """Unity 창에 포커스가 있어도 Windows의 R 눌림 순간을 감지합니다."""
    global _global_r_was_down

    if sys.platform != "win32":
        return False

    try:
        is_down = bool(
            ctypes.windll.user32.GetAsyncKeyState(ord("R")) & 0x8000
        )
    except Exception:
        return False

    pressed = is_down and not _global_r_was_down
    _global_r_was_down = is_down
    return pressed


def global_calibration_key_pressed() -> bool:
    """Unity 창에 포커스가 있어도 Windows의 K 눌림 순간을 감지합니다."""
    global _global_k_was_down

    if sys.platform != "win32":
        return False

    try:
        is_down = bool(
            ctypes.windll.user32.GetAsyncKeyState(ord("K")) & 0x8000
        )
    except Exception:
        return False

    pressed = is_down and not _global_k_was_down
    _global_k_was_down = is_down
    return pressed


def progress_bar(progress: float, width: int = 16) -> str:
    """터미널 캘리브레이션 진행률을 간결한 바 형태로 만듭니다."""
    ratio = clamp(progress, 0.0, 1.0)
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


# ============================================================================
# 센서 상태
# ============================================================================

def finger_sensor_valid(frame: SensorFrame) -> tuple[bool, str]:
    accel = frame.finger_accel
    gyro = frame.finger_gyro

    if not is_finite_vector(accel) or not is_finite_vector(gyro):
        return False, "NAN_OR_INF"

    accel_norm = vector_norm(accel)
    if accel_norm < CALIBRATION_MIN_ACCEL_NORM:
        return False, "ACCEL_ZERO_OR_TOO_SMALL"

    if accel_norm > CALIBRATION_MAX_ACCEL_NORM:
        return False, "ACCEL_TOO_LARGE"

    if max(abs(value) for value in (*accel, *gyro)) < 1e-8:
        return False, "ALL_ZERO"

    return True, "OK"


# ============================================================================
# Unity 전송
# ============================================================================

def send_unity(
    udp_socket: socket.socket,
    frame: SensorFrame,
    *,
    calibrated: bool,
    pointer_x: float,
    pointer_y: float,
    horizontal_angle: float,
    vertical_angle: float,
    writing_state: int,
    tracking_state: str,
    hand_q: Quaternion,
    finger_q: Quaternion,
    recenter_id: int,
) -> None:
    hand_accel = (
        frame.hand_accel
        if frame.hand_accel is not None
        else frame.wrist_accel
    )

    hand_q_xyzw = quaternion_to_unity_xyzw(hand_q)
    finger_q_xyzw = quaternion_to_unity_xyzw(finger_q)

    message = {
        "type": "glove",
        "ts": int(frame.timestamp_ms),
        "calibrated": 1 if calibrated else 0,

        "pointer_x": float(pointer_x),
        "pointer_y": float(pointer_y),
        "pointer_angle_x_deg": math.degrees(horizontal_angle),
        "pointer_angle_y_deg": math.degrees(vertical_angle),

        "button": int(writing_state),
        "pressure_raw": int(frame.writing_raw),
        "writing_input_mode": WRITING_INPUT_MODE,

        "packet_size": int(frame.packet_size),
        "pointer_source": "S3_FINGER_IMU",
        "tracking_state": tracking_state,
        "axis_remap": 1 if AXIS_REMAP else 0,
        "mag_yaw_enabled": 1 if USE_MAGNETOMETER else 0,

        "hand_ax": float(hand_accel[0]),
        "hand_ay": float(hand_accel[1]),
        "hand_az": float(hand_accel[2]),

        "hand_qx": float(hand_q_xyzw[0]),
        "hand_qy": float(hand_q_xyzw[1]),
        "hand_qz": float(hand_q_xyzw[2]),
        "hand_qw": float(hand_q_xyzw[3]),

        "finger_qx": float(finger_q_xyzw[0]),
        "finger_qy": float(finger_q_xyzw[1]),
        "finger_qz": float(finger_q_xyzw[2]),
        "finger_qw": float(finger_q_xyzw[3]),

        "recognized_text": latest_recognized_text,
        "recognized_char": latest_recognized_char,
        "recognition_confidence": latest_recognition_confidence,
        "recenter_id": int(recenter_id),
    }

    udp_socket.sendto(
        json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
        (UDP_IP, UDP_PORT),
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # 이 파일이 놓인 폴더의 airwriting_imu/weights를 우선 사용합니다.
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    model_path = project_root / "weights" / "pure_bilstm.pt"
    meta_path = project_root / "weights" / "meta.pkl"

    print()
    print("━" * 58)
    print("  AIR WRITING · UNITY AI BRIDGE")
    print(f"  {BRIDGE_VERSION}")
    print("━" * 58)

    try:
        sklearn_version = metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        sklearn_version = "설치되지 않음"

    # sklearn이 출력하는 긴 영문 경고 대신 아래 SYSTEM CHECK에서
    # 현재 버전과 모델 권장 버전을 알아보기 쉽게 안내합니다.
    warnings.filterwarnings(
        "ignore",
        message="Trying to unpickle estimator StandardScaler.*",
    )

    try:
        from airwriting_imu.core.ai_model import AirWritingAI
        from airwriting_imu.core.streaming import StreamingInference

        ai_engine = AirWritingAI()
        if not ai_engine.load_model():
            raise RuntimeError("AI 모델 로드 결과가 False입니다.")
        if ai_engine.model_type != "pure_bilstm":
            raise RuntimeError(
                "pure_bilstm이 아닌 모델이 선택되었습니다: "
                f"{ai_engine.model_type}"
            )

        ai_streamer = StreamingInference(
            ai_engine,
            char_timeout=AI_CHAR_TIMEOUT_SEC,
            space_timeout=AI_SPACE_TIMEOUT_SEC,
        )
        ai_streamer.on_text_updated = handle_ai_text_updated
        print(
            "[AI 준비 완료]",
            f"device={ai_engine.device}",
            f"model={ai_engine.model_type}",
            f"char_timeout={AI_CHAR_TIMEOUT_SEC:.1f}s",
            f"space_timeout={AI_SPACE_TIMEOUT_SEC:.1f}s",
        )
        for label, path in (
            ("pure_bilstm.pt", model_path),
            ("meta.pkl", meta_path),
        ):
            if path.exists():
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                print(
                    "[모델 파일]",
                    f"name={label}",
                    f"path={path}",
                    f"modified={modified:%Y-%m-%d %H:%M:%S}",
                    f"size={path.stat().st_size:,} bytes",
                )
            else:
                print("[모델 파일 경고]", f"찾을 수 없음: {path}")

        print()
        print("  SYSTEM CHECK")
        print(f"  ✓ AI Model       {ai_engine.model_type} · {ai_engine.device}")
        print(f"  ✓ Model Weights  {model_path.name}")
        if sklearn_version == RECOMMENDED_SKLEARN_VERSION:
            print(f"  ✓ Model Runtime  scikit-learn {sklearn_version}")
        else:
            print(
                "  ▲ Model Runtime  "
                f"scikit-learn {sklearn_version} "
                f"(권장 {RECOMMENDED_SKLEARN_VERSION})"
            )
            print("    모델 생성 환경과 맞추면 가장 안전합니다.")
    except Exception as error:
        print("[AI 초기화 실패]", repr(error))
        print(
            "[확인] 이 파일을 main.py, airwriting_imu, "
            "weights 폴더와 같은 위치에 넣으세요."
        )
        raise SystemExit(1)

    diagnostic_capture: Optional[DiagnosticCapture] = None
    if CAPTURE_ENABLED and CAPTURE_LABEL is not None:
        diagnostic_capture = DiagnosticCapture(
            project_root / "live_captures_pure",
            CAPTURE_LABEL,
            AI_CHAR_TIMEOUT_SEC,
            model_path=model_path,
            meta_path=meta_path,
        )
        print(
            "[진단 저장 준비]",
            f"label={CAPTURE_LABEL}",
            f"folder={diagnostic_capture.output_dir}",
        )

    parser = PacketParser()
    writing_input = WritingInput()

    finger_filter = MadgwickFilter(
        beta=MADGWICK_BETA,
        sample_rate=SAMPLE_RATE,
    )
    hand_filter = MadgwickFilter(
        beta=MADGWICK_BETA,
        sample_rate=SAMPLE_RATE,
    )

    mag_yaw = AdaptiveMagYaw()
    ray_projector = RayProjection()

    pointer_filter_x = OneEuroFilter(
        SAMPLE_RATE,
        ONE_EURO_MIN_CUTOFF,
        ONE_EURO_BETA,
        ONE_EURO_D_CUTOFF,
    )
    pointer_filter_y = OneEuroFilter(
        SAMPLE_RATE,
        ONE_EURO_MIN_CUTOFF,
        ONE_EURO_BETA,
        ONE_EURO_D_CUTOFF,
    )
    ai_filter_x = OneEuroFilter(
        SAMPLE_RATE,
        AI_ONE_EURO_MIN_CUTOFF,
        ONE_EURO_BETA,
        ONE_EURO_D_CUTOFF,
    )
    ai_filter_y = OneEuroFilter(
        SAMPLE_RATE,
        AI_ONE_EURO_MIN_CUTOFF,
        ONE_EURO_BETA,
        ONE_EURO_D_CUTOFF,
    )

    serial_connection = open_serial()
    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    print(
        f"  ✓ Glove          {serial_connection.port} · "
        f"{BAUD_RATE} baud"
    )
    print(f"  ✓ Unity Link     {UDP_IP}:{UDP_PORT}")
    receive_buffer = bytearray()

    last_timestamp_ms: Optional[int] = None
    last_valid_sensor_time: Optional[float] = None
    invalid_sensor_since: Optional[float] = None
    sensor_lost_printed = False

    finger_gyro_samples: list[Vector3] = []
    hand_gyro_samples: list[Vector3] = []
    finger_accel_samples: list[Vector3] = []
    hand_accel_samples: list[Vector3] = []
    mag_samples: list[Vector3] = []

    finger_gyro_bias: Vector3 = (0.0, 0.0, 0.0)
    hand_gyro_bias: Vector3 = (0.0, 0.0, 0.0)

    reference_horizontal_samples: list[float] = []
    reference_vertical_samples: list[float] = []
    reference_horizontal: Optional[float] = None
    reference_vertical: Optional[float] = None

    current_horizontal = 0.0
    current_vertical = 0.0
    last_relative_horizontal = 0.0
    last_relative_vertical = 0.0
    pointer_x = 0.0
    pointer_y = 0.0

    finger_q: Quaternion = (1.0, 0.0, 0.0, 0.0)
    hand_q: Quaternion = (1.0, 0.0, 0.0, 0.0)

    first_packet_size: Optional[int] = None
    tracking_state = "CALIBRATING"
    last_log_time = time.time()
    detailed_log_enabled = False
    last_calibration_print = 0.0
    calibration_unstable_since: Optional[float] = None
    calibration_motion_notice_shown = False
    sent_count = 0
    fsr_long_press_latched = False
    fsr_press_started_at: Optional[float] = None
    recenter_id = 0

    def reset_reference() -> None:
        nonlocal reference_horizontal
        nonlocal reference_vertical
        nonlocal reference_horizontal_samples
        nonlocal reference_vertical_samples
        nonlocal pointer_x
        nonlocal pointer_y
        nonlocal recenter_id

        reference_horizontal = current_horizontal
        reference_vertical = current_vertical
        reference_horizontal_samples = []
        reference_vertical_samples = []
        pointer_filter_x.reset(0.0)
        pointer_filter_y.reset(0.0)
        ai_filter_x.reset(0.0)
        ai_filter_y.reset(0.0)
        pointer_x = 0.0
        pointer_y = 0.0
        ai_streamer.reset()
        if diagnostic_capture is not None:
            diagnostic_capture.reset()
        recenter_id += 1

        print(
            "[✓ 중앙 재설정] 현재 손가락 방향을 "
            "Unity 화면 중앙 (0,0)으로 저장했습니다.",
            f"recenter_id={recenter_id}",
        )

    def start_full_calibration(reason: str) -> None:
        """필터 설정값은 유지한 채 센서 상태와 기준 자세만 초기화합니다."""
        nonlocal tracking_state
        nonlocal finger_gyro_bias
        nonlocal hand_gyro_bias
        nonlocal reference_horizontal
        nonlocal reference_vertical
        nonlocal reference_horizontal_samples
        nonlocal reference_vertical_samples
        nonlocal pointer_x
        nonlocal pointer_y
        nonlocal finger_q
        nonlocal hand_q
        nonlocal writing_input
        nonlocal recenter_id
        nonlocal calibration_unstable_since
        nonlocal calibration_motion_notice_shown
        nonlocal last_calibration_print
        nonlocal fsr_press_started_at
        nonlocal fsr_long_press_latched

        finger_gyro_samples.clear()
        hand_gyro_samples.clear()
        finger_accel_samples.clear()
        hand_accel_samples.clear()
        mag_samples.clear()
        finger_gyro_bias = (0.0, 0.0, 0.0)
        hand_gyro_bias = (0.0, 0.0, 0.0)

        reference_horizontal = None
        reference_vertical = None
        reference_horizontal_samples = []
        reference_vertical_samples = []

        # 기존 필터의 beta/cutoff 값은 변경하지 않고 내부 상태만 비웁니다.
        finger_filter.reset()
        hand_filter.reset()
        pointer_filter_x.reset(0.0)
        pointer_filter_y.reset(0.0)
        ai_filter_x.reset(0.0)
        ai_filter_y.reset(0.0)
        finger_q = (1.0, 0.0, 0.0, 0.0)
        hand_q = (1.0, 0.0, 0.0, 0.0)
        pointer_x = 0.0
        pointer_y = 0.0
        writing_input = WritingInput()
        ai_streamer.reset()
        if diagnostic_capture is not None:
            diagnostic_capture.reset()

        tracking_state = "CALIBRATING"
        calibration_unstable_since = None
        calibration_motion_notice_shown = False
        last_calibration_print = 0.0
        fsr_press_started_at = None
        fsr_long_press_latched = False
        recenter_id += 1

        print()
        print("[↻ 전체 캘리브레이션 재시작]")
        print(f"  사유: {reason}")
        print("  장갑을 정면의 편한 자세로 두고 4~6초간 가만히 유지하세요.")

    print(f"[버전] {BRIDGE_VERSION}")
    print(f"[UDP 시작] {UDP_IP}:{UDP_PORT}")
    print(
        "[설정]",
        f"axis_remap={int(AXIS_REMAP)}",
        f"mag_yaw={int(USE_MAGNETOMETER)}",
        f"ai_sensor={AI_SENSOR_SOURCE}",
        f"ai_flip_y={int(AI_FLIP_Y)}",
        f"pointer_range=({POINTER_X_RANGE_DEG:.1f}°,"
        f"{POINTER_Y_RANGE_DEG:.1f}°)",
    )
    print(
        "[중요] 다른 unity_bridge.py, main.py, Serial Monitor를 "
        "모두 종료하고 이 파일 하나만 실행하세요."
    )
    print(
        "[준비] 자이로 보정과 기준점 설정이 끝날 때까지 "
        "장갑을 정면의 편한 자세로 4~6초간 가만히 두세요."
    )
    print(
        "[단축키] R=빠른 중앙 재설정 · "
        "K=전체 캘리브레이션 재시작 · D=상세 로그 · Q=종료"
    )
    print("          R/K는 Unity Game 화면에서도, D/Q는 터미널에서 동작합니다.")

    try:
        while True:
            key = check_console_key()
            global_r_pressed = global_recenter_key_pressed()
            global_k_pressed = global_calibration_key_pressed()

            if key == "q":
                print("\n[종료] Q 키 입력")
                break

            if key == "d":
                detailed_log_enabled = not detailed_log_enabled
                mode_name = "상세 로그" if detailed_log_enabled else "간단 로그"
                print(f"\n[화면 모드] {mode_name}로 전환했습니다.")

            if key == "r" or global_r_pressed:
                if tracking_state == "TRACKING":
                    reset_reference()
                else:
                    print("[안내] TRACKING 상태가 된 뒤 R을 눌러주세요.")

            if key == "k" or global_k_pressed:
                start_full_calibration("K 키 입력")

            try:
                data = serial_connection.read(
                    max(1, serial_connection.in_waiting)
                )
            except serial.SerialException as error:
                print(f"[Serial 오류] {error}")
                try:
                    serial_connection.close()
                except Exception:
                    pass

                receive_buffer.clear()
                time.sleep(2)
                serial_connection = open_serial()
                last_timestamp_ms = None
                continue

            if not data:
                continue

            receive_buffer.extend(data)

            for packet in extract_packets(receive_buffer):
                frame = parser.parse(packet)
                if frame is None:
                    continue

                if first_packet_size is None:
                    first_packet_size = frame.packet_size
                    print(f"[패킷 확인] packet={first_packet_size}")

                    if first_packet_size == 94:
                        print(
                            "[정상] 최신 Tri-Node + sequence 패킷입니다."
                        )
                    elif first_packet_size == 92:
                        print(
                            "[주의] 92B Legacy 패킷입니다. "
                            "새 ICM-20948 gyro 수정 펌웨어가 실제 장갑에 "
                            "아직 업로드되지 않았을 수 있습니다."
                        )

                if last_timestamp_ms is None:
                    dt = 1.0 / SAMPLE_RATE
                else:
                    dt = (
                        frame.timestamp_ms - last_timestamp_ms
                    ) / 1000.0

                    if dt <= 0.0 or dt > 0.1:
                        dt = 1.0 / SAMPLE_RATE

                last_timestamp_ms = frame.timestamp_ms

                valid_sensor, invalid_reason = finger_sensor_valid(frame)
                now = time.time()
                # 펌웨어 v3.4의 raw=2를 우선 사용합니다. 펌웨어가 raw=1만
                # 보내는 장갑도 있으므로 Python에서 3초 연속 눌림을 보조 감지합니다.
                if frame.writing_raw == 1:
                    if fsr_press_started_at is None:
                        fsr_press_started_at = time.monotonic()
                elif frame.writing_raw == 0:
                    fsr_press_started_at = None
                    fsr_long_press_latched = False

                host_long_press = (
                    frame.writing_raw == 1
                    and fsr_press_started_at is not None
                    and time.monotonic() - fsr_press_started_at
                    >= FSR_HOST_LONGPRESS_SEC
                )
                is_fsr_long_press = (
                    frame.writing_raw == 2 or host_long_press
                )
                effective_writing_raw = (
                    0 if is_fsr_long_press else frame.writing_raw
                )

                # 최신 펌웨어의 button==2는 필기가 아니라 장압 리센터
                # 이벤트입니다. 누르고 있는 동안 한 번만 기준점을 갱신합니다.
                if is_fsr_long_press:
                    if (
                        not fsr_long_press_latched
                        and tracking_state == "TRACKING"
                    ):
                        reset_reference()
                        source = (
                            "firmware raw=2"
                            if frame.writing_raw == 2
                            else "Python 3초 보조 감지"
                        )
                        print(
                            "[FSR 장압] 포인터 중앙을 다시 설정했습니다.",
                            f"source={source}",
                        )
                    fsr_long_press_latched = True

                if not valid_sensor:
                    if invalid_sensor_since is None:
                        invalid_sensor_since = now

                    elapsed = (
                        float("inf")
                        if last_valid_sensor_time is None
                        else now - last_valid_sensor_time
                    )

                    invalid_duration = now - invalid_sensor_since
                    if invalid_duration >= S3_AUTO_RECONNECT_SEC:
                        print(
                            "[S3 자동 복구]",
                            f"{invalid_duration:.1f}초 동안 {invalid_reason} 상태라 "
                            "ESP32 Serial을 다시 연결합니다.",
                        )
                        try:
                            serial_connection.close()
                        except Exception:
                            pass

                        receive_buffer.clear()
                        time.sleep(0.5)
                        serial_connection = open_serial()

                        # 재연결 후 이전 자세·바이어스·획 상태를 재사용하면
                        # 큰 점프와 잘못된 AI 입력이 생기므로 처음부터 다시 보정합니다.
                        last_timestamp_ms = None
                        last_valid_sensor_time = None
                        invalid_sensor_since = None
                        sensor_lost_printed = False
                        first_packet_size = None
                        tracking_state = "CALIBRATING"
                        calibration_unstable_since = None
                        calibration_motion_notice_shown = False

                        finger_gyro_samples.clear()
                        hand_gyro_samples.clear()
                        finger_accel_samples.clear()
                        hand_accel_samples.clear()
                        mag_samples.clear()
                        finger_gyro_bias = (0.0, 0.0, 0.0)
                        hand_gyro_bias = (0.0, 0.0, 0.0)

                        reference_horizontal = None
                        reference_vertical = None
                        reference_horizontal_samples.clear()
                        reference_vertical_samples.clear()
                        finger_filter.reset()
                        hand_filter.reset()
                        pointer_filter_x.reset(0.0)
                        pointer_filter_y.reset(0.0)
                        ai_filter_x.reset(0.0)
                        ai_filter_y.reset(0.0)
                        pointer_x = 0.0
                        pointer_y = 0.0
                        writing_input = WritingInput()
                        ai_streamer.reset()

                        print(
                            "[S3 자동 복구] 재연결 완료. 장갑을 4~6초간 "
                            "가만히 두고 다시 보정하세요."
                        )
                        break

                    if elapsed > LONG_SENSOR_LOST_SEC:
                        tracking_state = "S3_SENSOR_LOST"
                        writing_state = 0

                        if not sensor_lost_printed:
                            print(
                                "[S3 손가락 IMU 끊김]",
                                f"reason={invalid_reason}",
                                f"|accel|={vector_norm(frame.finger_accel):.3f}",
                                f"|gyro|={vector_norm(frame.finger_gyro):.3f}",
                            )
                            print(
                                "[안전 동작] 손등 IMU로 전환하지 않고 "
                                "Trail을 종료합니다."
                            )
                            sensor_lost_printed = True
                    else:
                        tracking_state = "SHORT_SENSOR_HOLD"
                        writing_state = writing_input.update(
                            effective_writing_raw
                        )

                    hold_calibrated = (
                        tracking_state == "SHORT_SENSOR_HOLD"
                        and reference_horizontal is not None
                        and reference_vertical is not None
                    )

                    send_unity(
                        udp_socket,
                        frame,
                        calibrated=hold_calibrated,
                        pointer_x=pointer_x,
                        pointer_y=pointer_y,
                        horizontal_angle=last_relative_horizontal,
                        vertical_angle=last_relative_vertical,
                        writing_state=writing_state,
                        tracking_state=tracking_state,
                        hand_q=hand_q,
                        finger_q=finger_q,
                        recenter_id=recenter_id,
                    )
                    continue

                invalid_sensor_since = None
                last_valid_sensor_time = now
                if sensor_lost_printed:
                    print("[S3 손가락 IMU 복구]")
                    sensor_lost_printed = False

                writing_state = writing_input.update(
                    effective_writing_raw
                )

                hand_accel = (
                    frame.hand_accel
                    if frame.hand_accel is not None
                    else frame.wrist_accel
                )
                hand_gyro = (
                    frame.hand_gyro
                    if frame.hand_gyro is not None
                    else frame.wrist_gyro
                )

                # ------------------------------------------------------------
                # 1. 자이로 바이어스 보정
                # ------------------------------------------------------------
                if len(finger_gyro_samples) < GYRO_CALIBRATION_SAMPLES:
                    tracking_state = "CALIBRATING"

                    finger_gyro_norm = vector_norm(frame.finger_gyro)
                    hand_gyro_norm = vector_norm(hand_gyro)
                    calibration_is_stable = (
                        finger_gyro_norm <= CALIBRATION_MAX_GYRO_NORM
                        and hand_gyro_norm <= CALIBRATION_MAX_GYRO_NORM * 2.0
                    )

                    if calibration_is_stable:
                        if calibration_motion_notice_shown:
                            print(
                                "[✓ 자세 안정] 움직임이 멈췄습니다. "
                                "캘리브레이션을 이어서 진행합니다."
                            )
                        calibration_unstable_since = None
                        calibration_motion_notice_shown = False
                        finger_gyro_samples.append(frame.finger_gyro)
                        hand_gyro_samples.append(hand_gyro)
                        finger_accel_samples.append(frame.finger_accel)
                        hand_accel_samples.append(hand_accel)

                        if vector_norm(frame.finger_mag) > 1e-6:
                            mag_samples.append(frame.finger_mag)
                    else:
                        if calibration_unstable_since is None:
                            calibration_unstable_since = now

                        unstable_duration = now - calibration_unstable_since
                        if (
                            not calibration_motion_notice_shown
                            and unstable_duration
                            >= CALIBRATION_MOTION_NOTICE_DELAY_SEC
                        ):
                            calibration_motion_notice_shown = True
                            print(
                                "[! 움직임 감지] 캘리브레이션을 잠시 멈춥니다. "
                                "장갑을 편한 자세로 가만히 두세요."
                            )

                        if unstable_duration >= CALIBRATION_UNSTABLE_RESTART_SEC:
                            finger_gyro_samples.clear()
                            hand_gyro_samples.clear()
                            finger_accel_samples.clear()
                            hand_accel_samples.clear()
                            mag_samples.clear()
                            calibration_unstable_since = now
                            print(
                                "[↻ 자동 재시작] 움직임이 계속되어 기존 보정 샘플을 "
                                "지우고 처음부터 다시 측정합니다."
                            )

                    if now - last_calibration_print >= 0.5:
                        progress_ratio = (
                            len(finger_gyro_samples)
                            / GYRO_CALIBRATION_SAMPLES
                        )
                        progress = progress_ratio * 100.0
                        calibration_status = (
                            "진행 중" if calibration_is_stable else "일시 정지"
                        )
                        print(
                            f"[CALIBRATION] {progress_bar(progress_ratio)} "
                            f"{progress:5.1f}% · {calibration_status} "
                            f"(S3|g|={finger_gyro_norm:.3f})"
                        )
                        last_calibration_print = now

                    if (
                        len(finger_gyro_samples)
                        >= GYRO_CALIBRATION_SAMPLES
                    ):
                        finger_gyro_bias = median_vector(
                            finger_gyro_samples
                        )
                        hand_gyro_bias = median_vector(
                            hand_gyro_samples
                        )
                        finger_initial_accel = median_vector(
                            finger_accel_samples
                        )
                        hand_initial_accel = median_vector(
                            hand_accel_samples
                        )
                        finger_q = quaternion_from_accel(
                            finger_initial_accel
                        )
                        hand_q = quaternion_from_accel(
                            hand_initial_accel
                        )
                        finger_filter.reset(finger_q)
                        hand_filter.reset(hand_q)

                        print(
                            "[✓ 자이로 보정 완료]",
                            f"finger_bias={tuple(round(v, 5) for v in finger_gyro_bias)}",
                            f"hand_bias={tuple(round(v, 5) for v in hand_gyro_bias)}",
                        )
                        print(
                            "[✓ 중력 기준 자세 설정]",
                            f"S3_accel={tuple(round(v, 3) for v in finger_initial_accel)}",
                            f"S3_q={tuple(round(v, 5) for v in finger_q)}",
                        )

                    send_unity(
                        udp_socket,
                        frame,
                        calibrated=False,
                        pointer_x=0.0,
                        pointer_y=0.0,
                        horizontal_angle=0.0,
                        vertical_angle=0.0,
                        writing_state=0,
                        tracking_state=tracking_state,
                        hand_q=hand_q,
                        finger_q=finger_q,
                        recenter_id=recenter_id,
                    )
                    continue

                # ------------------------------------------------------------
                # 2. 자세 추정
                # ------------------------------------------------------------
                corrected_finger_gyro = vector_subtract(
                    frame.finger_gyro,
                    finger_gyro_bias,
                )
                corrected_hand_gyro = vector_subtract(
                    hand_gyro,
                    hand_gyro_bias,
                )

                corrected_finger_gyro = mag_yaw.correct_gyro(
                    frame.finger_mag,
                    corrected_finger_gyro,
                    finger_q,
                    dt,
                )

                finger_q = finger_filter.update_imu(
                    frame.finger_accel,
                    corrected_finger_gyro,
                    dt,
                )
                hand_q = hand_filter.update_imu(
                    hand_accel,
                    corrected_hand_gyro,
                    dt,
                )

                current_horizontal, current_vertical = (
                    ray_projector.project(finger_q)
                )

                # ------------------------------------------------------------
                # 3. 기준 자세 설정
                # ------------------------------------------------------------
                if (
                    reference_horizontal is None
                    or reference_vertical is None
                ):
                    tracking_state = "SETTING_REFERENCE"

                    reference_horizontal_samples.append(
                        current_horizontal
                    )
                    reference_vertical_samples.append(
                        current_vertical
                    )

                    if (
                        len(reference_horizontal_samples)
                        >= REFERENCE_SETTLE_SAMPLES
                    ):
                        reference_horizontal = mean_angle(
                            reference_horizontal_samples
                        )
                        reference_vertical = float(
                            statistics.mean(
                                reference_vertical_samples
                            )
                        )

                        mag_yaw.calibrate(
                            mag_samples,
                            finger_q,
                        )

                        pointer_filter_x.reset(0.0)
                        pointer_filter_y.reset(0.0)

                        print(
                            "[✓ 포인터 기준점 설정 완료]",
                            f"horizontal={math.degrees(reference_horizontal):+.1f}°",
                            f"vertical={math.degrees(reference_vertical):+.1f}°",
                        )
                        print("━" * 58)
                        print("  ✓ READY · Unity에서 필기를 시작할 수 있습니다.")
                        print(
                            "  R 중앙 재설정 · K 전체 재보정 · "
                            "D 상세 로그 · Q 종료"
                        )
                        print("━" * 58)

                    send_unity(
                        udp_socket,
                        frame,
                        calibrated=False,
                        pointer_x=0.0,
                        pointer_y=0.0,
                        horizontal_angle=current_horizontal,
                        vertical_angle=current_vertical,
                        writing_state=0,
                        tracking_state=tracking_state,
                        hand_q=hand_q,
                        finger_q=finger_q,
                        recenter_id=recenter_id,
                    )
                    continue

                # ------------------------------------------------------------
                # 4. 상대 좌표와 필터
                # ------------------------------------------------------------
                tracking_state = "TRACKING"

                relative_horizontal = wrap_angle(
                    current_horizontal - reference_horizontal
                )
                relative_vertical = (
                    current_vertical - reference_vertical
                )
                last_relative_horizontal = relative_horizontal
                last_relative_vertical = relative_vertical

                raw_x = (
                    relative_horizontal
                    / math.radians(POINTER_X_RANGE_DEG)
                )
                raw_y = (
                    relative_vertical
                    / math.radians(POINTER_Y_RANGE_DEG)
                )

                shaped_x = shape_pointer(
                    raw_x,
                    POINTER_DEAD_ZONE,
                    POINTER_RESPONSE_GAMMA_X,
                    POINTER_GAIN_X,
                )
                shaped_y = shape_pointer(
                    raw_y,
                    POINTER_DEAD_ZONE,
                    POINTER_RESPONSE_GAMMA_Y,
                    POINTER_GAIN_Y,
                )

                pointer_x = clamp(
                    OUTPUT_SIGN_X
                    * pointer_filter_x.filter(shaped_x, dt),
                    -1.0,
                    1.0,
                )
                pointer_y = clamp(
                    OUTPUT_SIGN_Y
                    * pointer_filter_y.filter(shaped_y, dt),
                    -1.0,
                    1.0,
                )

                # ------------------------------------------------------------
                # 5. 기존 웹과 동일한 pure_bilstm 입력 생성
                # ------------------------------------------------------------
                # Unity 화면 좌표(pointer_x/y)는 절대 AI 입력으로 재사용하지
                # 않습니다. 기존 main.py처럼 무왜곡 상대 물리각을 별도
                # One Euro Filter(1.5Hz)로 처리하고 ×4 스케일을 적용합니다.
                ai_x = (
                    ai_filter_x.filter(relative_horizontal, dt)
                    * AI_TRAJECTORY_SCALE
                )
                ai_y = (
                    ai_filter_y.filter(relative_vertical, dt)
                    * AI_TRAJECTORY_SCALE
                )
                if AI_FLIP_Y:
                    ai_y = -ai_y

                # 실제 저장 캡처 A/B 비교 결과에 맞춰 AI에만
                # S2(손등) mapped 축을 사용합니다. Unity 포인터/손 모델은
                # 기존처럼 S3(손가락)를 계속 사용하므로 움직임에는 영향이 없습니다.
                if (
                    AI_SENSOR_SOURCE == "S2_MAPPED"
                    and frame.hand_accel is not None
                    and frame.hand_gyro is not None
                ):
                    ai_accel = frame.hand_accel
                    ai_gyro = frame.hand_gyro
                else:
                    # --ai-s3 진단 모드 또는 S2가 없는 구형 패킷 fallback.
                    ai_accel = frame.finger_accel
                    ai_gyro = frame.finger_gyro
                ai_frame = {
                    "x": float(ai_x),
                    "y": float(ai_y),
                    "ax": float(ai_accel[0]),
                    "ay": float(ai_accel[1]),
                    "az": float(ai_accel[2]),
                    "gx": float(ai_gyro[0]),
                    "gy": float(ai_gyro[1]),
                    "gz": float(ai_gyro[2]),
                }
                if diagnostic_capture is not None:
                    diagnostic_capture.process(
                        ai_frame,
                        bool(writing_state),
                    )
                ai_streamer.process_frame(ai_frame, bool(writing_state))

                send_unity(
                    udp_socket,
                    frame,
                    calibrated=True,
                    pointer_x=pointer_x,
                    pointer_y=pointer_y,
                    horizontal_angle=relative_horizontal,
                    vertical_angle=relative_vertical,
                    writing_state=writing_state,
                    tracking_state=tracking_state,
                    hand_q=hand_q,
                    finger_q=finger_q,
                    recenter_id=recenter_id,
                )

                sent_count += 1

                log_interval = (
                    LOG_INTERVAL_SEC
                    if detailed_log_enabled
                    else SUMMARY_LOG_INTERVAL_SEC
                )
                if now - last_log_time >= log_interval:
                    if detailed_log_enabled:
                        print(
                            "[Unity 전송]",
                            f"packet={frame.packet_size}",
                            f"pointer=({pointer_x:+.2f},{pointer_y:+.2f})",
                            f"angle=({math.degrees(relative_horizontal):+.1f}°,"
                            f"{math.degrees(relative_vertical):+.1f}°)",
                            f"writing={writing_state}",
                            f"raw={frame.writing_raw}",
                            f"tx={sent_count}/s",
                            f"cksum_err={parser.checksum_errors}",
                            f"drop={parser.dropped_packets}",
                        )
                    else:
                        pen_state = "DOWN" if writing_state else "UP"
                        sensor_state = (
                            "OK"
                            if parser.checksum_errors == 0
                            and parser.dropped_packets == 0
                            else "CHECK"
                        )
                        elapsed_log_time = max(now - last_log_time, 1e-6)
                        print(
                            f"[● LIVE] Unity {sent_count / elapsed_log_time:.0f} Hz · "
                            f"Pointer ({pointer_x:+.2f}, {pointer_y:+.2f}) · "
                            f"Pen {pen_state} · Sensors {sensor_state}"
                        )
                    sent_count = 0
                    last_log_time = now

    except KeyboardInterrupt:
        print("\n[종료] 사용자 요청으로 Bridge를 종료합니다.")

    finally:
        try:
            serial_connection.close()
        except Exception:
            pass
        udp_socket.close()


if __name__ == "__main__":
    main()
