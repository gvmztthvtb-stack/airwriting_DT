"""
OLED 상태 전송 — Python → ESP32 OLED 디스플레이
================================================
인식 결과, 시스템 상태 등을 ESP32의 OLED에 표시.

프로토콜: UDP로 텍스트 메시지 전송
  - "OLED|{state}|{mode}|{headline}|{accuracy}"
  - "{letter},{accuracy}" (간단 형식)
"""
from __future__ import annotations

import socket
import logging

log = logging.getLogger(__name__)


class OLEDSender:
    """ESP32 OLED로 상태/인식 결과를 UDP 전송"""

    def __init__(self, oled_port: int = 5555):
        self.oled_port = oled_port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._esp32_addr = None

    def set_esp32_addr(self, ip: str, port: int = None):
        """ESP32 주소 설정. port는 일반적으로 localPort (5555)."""
        self._esp32_addr = (ip, port or self.oled_port)
        log.info(f"📺 OLED 대상: {self._esp32_addr}")

    def send_status(self, state: str, mode: str, headline: str, accuracy: float = 0.0):
        """
        OLED 전체 상태 업데이트.

        Args:
            state: "READY", "CAL", "WRITE", "OK", "FAIL" 등
            mode: "DAILY", "RUNNING" 등
            headline: 큰 글씨로 표시할 텍스트
            accuracy: 인식 정확도 (0~100)
        """
        if not self._esp32_addr:
            log.debug("OLED: ESP32 주소 미설정, 전송 스킵")
            return
        accuracy = max(0.0, min(100.0, accuracy))
        msg = f"OLED|{state}|{mode}|{headline}|{accuracy:.1f}"
        try:
            self._sock.sendto(msg.encode(), self._esp32_addr)
        except OSError as e:
            log.debug(f"OLED 전송 실패: {e}")

    def send_recognition(self, letter: str, accuracy: float):
        """
        간단한 인식 결과 전송 (글자 + 정확도).

        Args:
            letter: 인식된 글자 ("A", "B" 등)
            accuracy: 신뢰도 (0~100)
        """
        if not self._esp32_addr:
            return
        msg = f"{letter},{accuracy:.1f}"
        try:
            self._sock.sendto(msg.encode(), self._esp32_addr)
        except OSError as e:
            log.debug(f"OLED 전송 실패: {e}")

    def send_calibration_start(self):
        """캘리브레이션 시작 알림"""
        self.send_status("CAL", "SYSTEM", "HOLD", 0.0)

    def send_calibration_done(self, quality: str = "GOOD"):
        """캘리브레이션 완료 알림"""
        self.send_status("READY", "SYSTEM", quality, 100.0)

    def close(self):
        if hasattr(self, '_sock') and self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def __del__(self):
        self.close()
