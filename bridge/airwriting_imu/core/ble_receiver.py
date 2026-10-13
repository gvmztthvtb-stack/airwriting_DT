"""
BLE UART 수신기 — Nordic NUS 프로토콜로 ESP32에서 94B 패킷 수신.

의존성: pip install bleak
지원: Windows 10+, Linux, macOS

핵심 설계:
  - asyncio 네이티브 (main.py의 이벤트 루프와 직접 통합)
  - 자동 스캔 + 재연결 (연결 끊김 시 3초 후 재시도)
  - 수신 패킷 → asyncio.Queue로 전달 (기존 serial_reader_thread와 동일 인터페이스)
  - 연결 안정성: 연결 후 안정화 대기, 타임아웃 설정, 끊김 시 graceful 정리
"""
import asyncio
import logging
import time

log = logging.getLogger("airscribe")

NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # ESP32→PC (Notify)
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # PC→ESP32 (Write)

DEVICE_NAME = "AirScribe"
SCAN_TIMEOUT = 10.0
RECONNECT_DELAY = 3.0
VALID_PACKET_SIZES = [94, 92, 70, 68]


class BLEReceiver:
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop
        self.client = None
        self.connected = False
        self._packet_buffer = bytearray()
        self._packet_count = 0
        self._last_rx_time = 0.0

    async def scan_and_connect(self):
        """AirScribe 디바이스 스캔 → 연결 → 끊기면 재시도 (무한 루프)"""
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            log.error("❌ bleak 패키지가 필요합니다: pip install bleak")
            return

        while True:
            try:
                # ── 스캔 ──
                log.info(f"🔵 BLE 스캔 중... ('{DEVICE_NAME}' 검색, {SCAN_TIMEOUT}초)")
                device = await BleakScanner.find_device_by_name(
                    DEVICE_NAME, timeout=SCAN_TIMEOUT
                )
                if device is None:
                    log.warning("🔵 AirScribe를 찾지 못함 — 재스캔...")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue

                log.info(f"🔵 AirScribe 발견: {device.name} [{device.address}]")

                # ── 연결 (최대 3회 재시도) ──
                self._packet_buffer.clear()
                self._packet_count = 0
                
                connected = False
                for attempt in range(1, 4):
                    try:
                        log.info(f"🔵 BLE 연결 시도 {attempt}/3...")
                        self.client = BleakClient(
                            device.address,  # address 직접 사용 (Windows 호환성↑)
                            timeout=10.0,
                            disconnected_callback=self._on_disconnect,
                        )
                        await self.client.connect()
                        connected = True
                        break
                    except TimeoutError:
                        log.warning(f"🔵 연결 타임아웃 (시도 {attempt}/3)")
                        await asyncio.sleep(1.0)
                    except Exception as ce:
                        log.warning(f"🔵 연결 실패 (시도 {attempt}/3): {type(ce).__name__}: {ce}")
                        await asyncio.sleep(1.0)
                
                if not connected:
                    log.warning("🔵 3회 연결 실패 — Windows 블루투스 설정에서 AirScribe 페어링 제거 후 재시도하세요")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue

                # 연결 안정화 대기
                await asyncio.sleep(0.5)

                # NUS TX notify 구독
                await self.client.start_notify(NUS_TX_CHAR_UUID, self._on_notify)
                self.connected = True
                self._last_rx_time = time.monotonic()
                log.info("🔵 BLE 연결 성공! NUS notify 수신 시작")

                # ── 연결 유지 루프 ──
                while self.connected:
                    await asyncio.sleep(2.0)

                    # 수신 통계 출력
                    if self._packet_count > 0:
                        elapsed = time.monotonic() - self._last_rx_time
                        if elapsed > 10.0:
                            log.warning("🔵 BLE 데이터 수신 없음 (10초+) — 연결 확인 필요")

                # 연결 끊긴 후 정리
                try:
                    if self.client and self.client.is_connected:
                        await self.client.stop_notify(NUS_TX_CHAR_UUID)
                        await self.client.disconnect()
                except Exception:
                    pass

            except Exception as e:
                log.warning(f"🔵 BLE 에러: {type(e).__name__}: {e!r}")
                self.connected = False

            # 재연결 대기
            log.info(f"🔵 {RECONNECT_DELAY}초 후 재연결 시도...")
            await asyncio.sleep(RECONNECT_DELAY)

    def _on_notify(self, sender, data: bytearray):
        """BLE Notify 콜백: 20B 청크 → 94B 패킷 조립 → Queue 전달"""
        self._packet_buffer.extend(data)
        self._last_rx_time = time.monotonic()

        while len(self._packet_buffer) >= 68:
            idx = self._packet_buffer.find(0xAA)
            if idx < 0:
                self._packet_buffer.clear()
                break
            if idx > 0:
                self._packet_buffer = self._packet_buffer[idx:]

            packet_found = False
            for p_size in VALID_PACKET_SIZES:
                if (len(self._packet_buffer) >= p_size
                        and self._packet_buffer[p_size - 1] == 0x55):
                    packet = bytes(self._packet_buffer[:p_size])
                    try:
                        self.queue.put_nowait(packet)
                        self._packet_count += 1
                    except asyncio.QueueFull:
                        pass  # 최신 데이터 보존
                    self._packet_buffer = self._packet_buffer[p_size:]
                    packet_found = True
                    break
            if not packet_found:
                if len(self._packet_buffer) > 100:
                    self._packet_buffer = self._packet_buffer[1:]
                else:
                    break

    def _on_disconnect(self, client):
        self.connected = False
        log.warning(f"🔵 BLE 연결 끊김 (수신 패킷: {self._packet_count}개) — 재연결 예정")

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def packet_count(self) -> int:
        return self._packet_count
