"""설정 로더 — Hybrid-AirScribe Dual-Node"""
import yaml, logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger(__name__)


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class IMUSensorConfig:
    role: str
    sensor_type: str
    bus: int
    address: int
    sample_rate_hz: int = 100
    has_magnetometer: bool = False
    i2c_pins: dict = field(default_factory=dict)
    accel_range: str = "8g"
    gyro_range: str = "1000dps"
    registers: dict = field(default_factory=dict)
    unit_conversion: dict = field(default_factory=dict)
    magnetometer: dict = field(default_factory=dict)
    bias_accel: List[float] = field(default_factory=lambda: [0, 0, 0])
    bias_gyro: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class DualNodeConfig:
    reference: str = ""       # 손목 센서 ID (Global Motion Reference)
    primary: str = ""         # 손가락 센서 ID (필기 궤적)
    motion_disentangle: bool = True


@dataclass
class SystemConfig:
    imu_sensors: Dict[str, IMUSensorConfig] = field(default_factory=dict)
    dual_node: DualNodeConfig = field(default_factory=DualNodeConfig)
    axis_remap: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)
    packet: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)
    preprocessing: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)


class ConfigLoader:
    _REQ = ("imu.yaml", "system.yaml")

    def __init__(self, config_dir=None):
        if config_dir is None:
            root = Path(__file__).parent.parent.parent
            for c in [root / "config", root / "config" / "products"]:
                if (c / "imu.yaml").exists():
                    config_dir = c
                    break
            if config_dir is None:
                raise FileNotFoundError("config/ not found")
        self.dir = Path(config_dir)
        miss = [f for f in self._REQ if not (self.dir / f).exists()]
        if miss:
            raise FileNotFoundError(f"Missing: {miss}")

    def _yaml(self, name):
        with open(self.dir / name, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        if d is None:
            raise ValueError(f"Empty: {name}")
        return d

    def _load_system_config(self):
        base = self._yaml("system.yaml")
        local_path = self.dir / "system.local.yaml"
        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                local = yaml.safe_load(f) or {}
            base = _deep_merge_dict(base, local)
        return base

    def load_all(self) -> SystemConfig:
        imu_raw = self._yaml("imu.yaml")
        sys_raw = self._load_system_config()

        # ── IMU sensors ──
        sensors = {}
        for name, c in imu_raw.get("sensors", {}).items():
            addr = c.get("address", 0x68)
            if isinstance(addr, str):
                addr = int(addr, 16)
            sensors[name] = IMUSensorConfig(
                role=c["role"],
                sensor_type=c["type"],
                bus=c["bus"],
                address=addr,
                sample_rate_hz=c.get("sample_rate_hz", 100),
                has_magnetometer=c.get("has_magnetometer", False),
                i2c_pins=c.get("i2c_pins", {}),
                accel_range=c.get("accel_range", "8g"),
                gyro_range=c.get("gyro_range", "1000dps"),
                registers=c.get("registers", {}),
                unit_conversion=c.get("unit_conversion", {}),
                magnetometer=c.get("magnetometer", {}),
                bias_accel=c.get("bias", {}).get("accel", [0, 0, 0]),
                bias_gyro=c.get("bias", {}).get("gyro", [0, 0, 0]),
            )

        # I2C conflict check (같은 버스에서 같은 주소 금지)
        seen = {}
        for n, s in sensors.items():
            k = (s.bus, s.address)
            if k in seen:
                raise ValueError(f"I2C conflict: {n} vs {seen[k]}")
            seen[k] = n

        # ── Dual-Node config ──
        dn_raw = imu_raw.get("dual_node", {})
        dual_node = DualNodeConfig(
            reference=dn_raw.get("reference", ""),
            primary=dn_raw.get("primary", ""),
            motion_disentangle=dn_raw.get("motion_disentangle", True),
        )

        # Validate dual-node references
        if dual_node.reference and dual_node.reference not in sensors:
            raise ValueError(
                f"dual_node.reference '{dual_node.reference}' "
                f"not found in sensors: {list(sensors.keys())}"
            )
        if dual_node.primary and dual_node.primary not in sensors:
            raise ValueError(
                f"dual_node.primary '{dual_node.primary}' "
                f"not found in sensors: {list(sensors.keys())}"
            )

        # ── Port conflict check ──
        ports = list(sys_raw.get("network", {}).get("ports", {}).values())
        if len(ports) != len(set(ports)):
            raise ValueError(f"Duplicate ports: {ports}")

        cfg = SystemConfig(
            imu_sensors=sensors,
            dual_node=dual_node,
            axis_remap=imu_raw.get("axis_remap", {}),
            network=sys_raw.get("network", {}),
            packet=sys_raw.get("packet", {}),
            fusion=sys_raw.get("fusion", {}),
            preprocessing=sys_raw.get("preprocessing", {}),
            calibration=sys_raw.get("calibration", {}),
        )
        # config dir 참조 보존 (bias 저장용)
        cfg._config_dir = str(self.dir)
        log.info(
            f"✅ Config: {len(sensors)} sensors, "
            f"dual_node={dual_node.reference}→{dual_node.primary} "
            f"(Hybrid-AirScribe)"
        )
        return cfg
