"""
패킷 파서 — AirWritingPacketV3/V4
==================================
ESP32에서 UDP로 수신한 바이너리 패킷을 파싱합니다.

패킷 포맷 V4 (Dual-Node + Seq, 70 Bytes):
  [0xAA] [2B seq] [4B timestamp_ms] [24B S1/WRIST] [36B S3/FINGER] [1B btn] [1B cksum] [0x55]

패킷 포맷 V3 (Dual-Node, 68 Bytes — 이전 펌웨어 호환):
  [0xAA] [4B timestamp_ms] [24B S1/WRIST] [36B S3/FINGER] [1B btn] [1B cksum] [0x55]

패킷 포맷 Legacy (3-IMU, 92 Bytes):
  [0xAA] [4B ts] [24B S1] [24B S2/HAND] [36B S3] [1B btn] [1B cksum] [0x55]
  → S2는 무시하고 S1/S3만 사용
"""
from __future__ import annotations

import struct
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

HEADER = 0xAA
FOOTER = 0x55


@dataclass(slots=True)
class SensorFrame:
    """파싱된 하나의 센서 프레임"""
    timestamp_ms: int
    wrist_accel: np.ndarray   # (3,) m/s²   — S1 (전완근)
    wrist_gyro: np.ndarray    # (3,) rad/s   — S1
    finger_accel: np.ndarray  # (3,) m/s²   — S3 (손가락)
    finger_gyro: np.ndarray   # (3,) rad/s   — S3
    finger_mag: np.ndarray    # (3,) µT      — S3 magnetometer
    button: int               # 0=up, 1=down
    packet_size: int          # 68, 70, 92, or 94
    hand_accel: Optional[np.ndarray] = None # (3,) m/s² — S2 (손등)
    hand_gyro: Optional[np.ndarray] = None  # (3,) rad/s — S2
    seq: int = -1             # 시퀀스 번호 (-1 = 없음/레거시)

    @property
    def pen_down(self) -> bool:
        return self.button == 1


class PacketParser:
    """
    AirWritingPacketV3 파서.
    68B (Dual-Node) 와 92B (Legacy 3-IMU) 패킷 모두 자동 감지.
    체크섬 검증 + 패킷 유실 감지 + 통계 수집.
    """

    # struct 포맷 (little-endian)
    _FMT_V4 = struct.Struct("<BHI6f9fBBB")       # 70 bytes (V4: seq 추가)
    _FMT_TRI_NODE = struct.Struct("<BHI6f6f9fBBB") # 94 bytes (Tri-Node V4: s1, s2, s3, seq)
    _FMT_DUAL = struct.Struct("<BI6f9fBBB")       # 68 bytes (V3)
    _FMT_LEGACY = struct.Struct("<BI6f6f9fBBB")    # 92 bytes (레거시)

    def __init__(self, axis_remap: bool = True):
        """
        Args:
            axis_remap: True이면 축 반전 적용 (x=-x, y=-y, z=z)
        """
        self.axis_remap = axis_remap

        # 통계
        self.total_packets = 0
        self.valid_packets = 0
        self.checksum_errors = 0
        self.format_errors = 0
        self.last_timestamp_ms = -1
        self.gap_count = 0
        self.max_gap_ms = 0
        # 시퀀스 기반 유실 감지
        self.last_seq = -1
        self.dropped_packets = 0

    def parse(self, data: bytes) -> Optional[SensorFrame]:
        """
        바이너리 데이터를 SensorFrame으로 파싱.
        실패 시 None 반환.
        """
        self.total_packets += 1

        # 크기별 자동 감지
        if len(data) == self._FMT_TRI_NODE.size: # 94B — Tri-Node V4
            return self._parse_tri_node(data)
        elif len(data) == self._FMT_V4.size:      # 70B — V4 (seq 포함)
            return self._parse_v4(data)
        elif len(data) == self._FMT_DUAL.size:   # 68B — V3
            return self._parse_dual(data)
        elif len(data) == self._FMT_LEGACY.size: # 92B — Legacy
            return self._parse_legacy(data)
        else:
            self.format_errors += 1
            return None

    def _verify_checksum(self, data: bytes) -> bool:
        """XOR 체크섬 검증 (header 제외, checksum+footer 제외)"""
        cksum_calc = 0
        for i in range(1, len(data) - 2):
            cksum_calc ^= data[i]
        return cksum_calc == data[-2]

    def _remap(self, ax, ay, az):
        """축 반전: x=-x, y=-y, z=z"""
        if self.axis_remap:
            return -ax, -ay, az
        return ax, ay, az

    def _track_timing(self, ts_ms: int):
        """타임스탬프 갭 추적"""
        if self.last_timestamp_ms >= 0:
            gap = ts_ms - self.last_timestamp_ms
            if gap < 0:
                gap += 0xFFFFFFFF
            if gap > 15:
                self.gap_count += 1
                if gap > self.max_gap_ms:
                    self.max_gap_ms = gap
        self.last_timestamp_ms = ts_ms

    def _track_seq(self, seq: int):
        """시퀀스 번호 기반 유실 감지"""
        if self.last_seq >= 0:
            expected = (self.last_seq + 1) & 0xFFFF  # uint16 wrap
            if seq != expected:
                dropped = (seq - expected) & 0xFFFF
                if dropped < 1000:  # 합리적 범위
                    self.dropped_packets += dropped
        self.last_seq = seq

    def _parse_tri_node(self, data: bytes) -> Optional[SensorFrame]:
        """94B Tri-Node V4 패킷 파싱"""
        if not self._verify_checksum(data):
            self.checksum_errors += 1
            return None

        v = self._FMT_TRI_NODE.unpack(data)
        if v[0] != HEADER or v[-1] != FOOTER:
            self.format_errors += 1
            return None

        seq = v[1]
        ts = v[2]
        self._track_timing(ts)
        self._track_seq(seq)

        # S1 (WRIST/전완근): indices 3-8
        wa = self._remap(v[3], v[4], v[5])
        wg = self._remap(v[6], v[7], v[8])
        
        # S2 (HAND/손등): indices 9-14
        ha = self._remap(v[9], v[10], v[11])
        hg = self._remap(v[12], v[13], v[14])

        # S3 (FINGER/손가락): indices 15-23
        fa = self._remap(v[15], v[16], v[17])
        fg = self._remap(v[18], v[19], v[20])
        fm = (v[21], v[22], v[23])

        self.valid_packets += 1
        return SensorFrame(
            timestamp_ms=ts,
            wrist_accel=np.array(wa, dtype=np.float32),
            wrist_gyro=np.array(wg, dtype=np.float32),
            hand_accel=np.array(ha, dtype=np.float32),
            hand_gyro=np.array(hg, dtype=np.float32),
            finger_accel=np.array(fa, dtype=np.float32),
            finger_gyro=np.array(fg, dtype=np.float32),
            finger_mag=np.array(fm, dtype=np.float32),
            button=v[24],
            packet_size=94,
            seq=seq,
        )

    def _parse_v4(self, data: bytes) -> Optional[SensorFrame]:
        """70B V4 패킷 파싱 (seq 포함)"""
        if not self._verify_checksum(data):
            self.checksum_errors += 1
            return None

        v = self._FMT_V4.unpack(data)
        if v[0] != HEADER or v[-1] != FOOTER:
            self.format_errors += 1
            return None

        seq = v[1]
        ts = v[2]
        self._track_timing(ts)
        self._track_seq(seq)

        # S1 (WRIST): indices 3-8
        wa = self._remap(v[3], v[4], v[5])
        wg = self._remap(v[6], v[7], v[8])

        # S3 (FINGER): indices 9-17
        fa = self._remap(v[9], v[10], v[11])
        fg = self._remap(v[12], v[13], v[14])
        fm = (v[15], v[16], v[17])

        self.valid_packets += 1
        return SensorFrame(
            timestamp_ms=ts,
            wrist_accel=np.array(wa, dtype=np.float32),
            wrist_gyro=np.array(wg, dtype=np.float32),
            finger_accel=np.array(fa, dtype=np.float32),
            finger_gyro=np.array(fg, dtype=np.float32),
            finger_mag=np.array(fm, dtype=np.float32),
            button=v[18],
            packet_size=70,
            seq=seq,
        )

    def _parse_dual(self, data: bytes) -> Optional[SensorFrame]:
        """68B Dual-Node 패킷 파싱"""
        if not self._verify_checksum(data):
            self.checksum_errors += 1
            return None

        v = self._FMT_DUAL.unpack(data)
        if v[0] != HEADER or v[-1] != FOOTER:
            self.format_errors += 1
            return None

        ts = v[1]
        self._track_timing(ts)

        # S1 (WRIST): indices 2-7
        wa = self._remap(v[2], v[3], v[4])
        wg = self._remap(v[5], v[6], v[7])

        # S3 (FINGER): indices 8-16
        fa = self._remap(v[8], v[9], v[10])
        fg = self._remap(v[11], v[12], v[13])
        fm = (v[14], v[15], v[16])  # mag는 축 반전 별도

        self.valid_packets += 1
        return SensorFrame(
            timestamp_ms=ts,
            wrist_accel=np.array(wa, dtype=np.float32),
            wrist_gyro=np.array(wg, dtype=np.float32),
            finger_accel=np.array(fa, dtype=np.float32),
            finger_gyro=np.array(fg, dtype=np.float32),
            finger_mag=np.array(fm, dtype=np.float32),
            button=v[17],
            packet_size=68,
        )

    def _parse_legacy(self, data: bytes) -> Optional[SensorFrame]:
        """92B Legacy 3-IMU 패킷 파싱 (S2/HAND 무시)"""
        if not self._verify_checksum(data):
            self.checksum_errors += 1
            return None

        v = self._FMT_LEGACY.unpack(data)
        if v[0] != HEADER or v[-1] != FOOTER:
            self.format_errors += 1
            return None

        ts = v[1]
        self._track_timing(ts)

        # S1 (WRIST): indices 2-7
        wa = self._remap(v[2], v[3], v[4])
        wg = self._remap(v[5], v[6], v[7])

        # S2 (HAND): indices 8-13
        ha = self._remap(v[8], v[9], v[10])
        hg = self._remap(v[11], v[12], v[13])

        # S3 (FINGER): indices 14-22
        fa = self._remap(v[14], v[15], v[16])
        fg = self._remap(v[17], v[18], v[19])
        fm = (v[20], v[21], v[22])

        self.valid_packets += 1
        return SensorFrame(
            timestamp_ms=ts,
            wrist_accel=np.array(wa, dtype=np.float32),
            wrist_gyro=np.array(wg, dtype=np.float32),
            hand_accel=np.array(ha, dtype=np.float32),
            hand_gyro=np.array(hg, dtype=np.float32),
            finger_accel=np.array(fa, dtype=np.float32),
            finger_gyro=np.array(fg, dtype=np.float32),
            finger_mag=np.array(fm, dtype=np.float32),
            button=v[23],
            packet_size=92,
        )

    def get_stats(self) -> dict:
        """파서 통계 반환"""
        loss_rate = 0.0
        if self.total_packets > 0:
            loss_rate = 1.0 - (self.valid_packets / self.total_packets)
        return {
            "total": self.total_packets,
            "valid": self.valid_packets,
            "checksum_errors": self.checksum_errors,
            "format_errors": self.format_errors,
            "loss_rate": round(loss_rate * 100, 2),
            "gap_count": self.gap_count,
            "max_gap_ms": self.max_gap_ms,
            "dropped_packets": self.dropped_packets,
        }

    def reset_stats(self):
        """통계 초기화"""
        self.total_packets = 0
        self.valid_packets = 0
        self.checksum_errors = 0
        self.format_errors = 0
        self.last_timestamp_ms = -1
        self.gap_count = 0
        self.max_gap_ms = 0
        self.last_seq = -1
        self.dropped_packets = 0
