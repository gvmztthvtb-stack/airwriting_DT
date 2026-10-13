"""
시간 동기화 — ESP32 ↔ Python 클럭 오프셋 추정
===============================================
ESP32의 millis() 타임스탬프와 Python time.time()의
오프셋을 추정하여 네트워크 지연(latency) 측정.

방식: 수신된 패킷의 (ESP32 timestamp, Python 수신시간) 쌍으로
     이동평균 오프셋 계산.
"""
from __future__ import annotations

import time
import logging
from collections import deque

log = logging.getLogger(__name__)


class TimeSync:
    """
    ESP32 ↔ Python 시간 오프셋 추정기.

    ESP32 타임스탬프는 millis() (부팅 후 경과 ms).
    Python 시간은 time.time() (epoch seconds).

    오프셋 = python_time_ms - esp32_timestamp_ms
    → 안정화된 오프셋으로 end-to-end 지연시간 추정 가능.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._offsets = deque(maxlen=window_size)
        self._latencies = deque(maxlen=window_size)
        self.offset_ms = 0.0        # 추정된 클럭 오프셋 (ms)
        self.latency_ms = 0.0       # 추정된 편도 지연 (ms)
        self.is_synced = False       # 최소 10 샘플 수집 후 True
        self._sync_count = 0

    def update(self, esp32_timestamp_ms: int):
        """
        패킷 수신 시 호출. ESP32 타임스탬프와 현재 시간으로 오프셋 계산.

        Args:
            esp32_timestamp_ms: ESP32의 millis() 값
        """
        now_ms = time.time() * 1000.0
        offset = now_ms - esp32_timestamp_ms

        self._offsets.append(offset)
        self._sync_count += 1

        # 중앙값 기반 오프셋 (아웃라이어 내성, 짝수 길이도 정확한 중앙값 사용)
        if len(self._offsets) >= 5:
            sorted_offsets = sorted(self._offsets)
            n = len(sorted_offsets)
            mid = n // 2
            self.offset_ms = sorted_offsets[mid] if n % 2 != 0 else \
                             (sorted_offsets[mid - 1] + sorted_offsets[mid]) / 2.0

        if not self.is_synced and self._sync_count >= 10:
            self.is_synced = True
            log.info(f"⏱️  시간 동기화 완료: offset={self.offset_ms:.0f}ms")

    def estimate_latency(self, esp32_timestamp_ms: int) -> float:
        """
        패킷의 네트워크 지연 추정 (ms).

        Args:
            esp32_timestamp_ms: 패킷의 ESP32 타임스탬프

        Returns:
            추정 지연시간 (ms). 동기화 전이면 -1.
        """
        if not self.is_synced:
            return -1.0

        now_ms = time.time() * 1000.0
        expected_now = esp32_timestamp_ms + self.offset_ms
        latency = now_ms - expected_now

        self._latencies.append(max(0.0, latency))

        if self._latencies:
            self.latency_ms = sum(self._latencies) / len(self._latencies)

        return max(0.0, latency)

    def esp32_to_python_time(self, esp32_timestamp_ms: int) -> float:
        """ESP32 타임스탬프 → Python epoch time (seconds)"""
        return (esp32_timestamp_ms + self.offset_ms) / 1000.0

    def get_stats(self) -> dict:
        return {
            "synced": self.is_synced,
            "offset_ms": round(self.offset_ms, 1),
            "latency_ms": round(self.latency_ms, 1),
            "samples": self._sync_count,
        }

    def reset(self):
        """ESP32 리셋 시 동기화 상태 초기화"""
        self._offsets.clear()
        self._latencies.clear()
        self.offset_ms = 0.0
        self.latency_ms = 0.0
        self.is_synced = False
        self._sync_count = 0
