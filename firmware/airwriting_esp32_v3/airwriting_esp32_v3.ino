// AirWriting_ESP32_v3.4 — USB Serial + BLE NUS + FSR
// - S1 (WRIST):  MPU6050 on I2C bus 0 (0x68)
// - S2 (HAND):   MPU6050 on I2C bus 0 (0x69)
// - S3 (FINGER): ICM20948 + AK09916 on I2C bus 1 (0x68)
// - FSR 장압 3s → button=2 (영점 리셋)
// - BLE NUS (AirScribe 디바이스명)
// - 94B 패킷 (Tri-Node V4, main.py 호환)
// - OLED/진동모터 제거 (미사용 + GPIO 충돌 방지)

#include <nvs_flash.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── 핀 설정 ──
const int FSR_PIN = 15;  // FSR 압력센서 (기존 버튼 핀에 FSR 연결)

// ── FSR ──
#define FSR_THRESHOLD    2048
#define FSR_LONGPRESS_MS 3000

// ── I2C ──
const int I2C0_SDA = 4,  I2C0_SCL = 5;
const int I2C1_SDA = 6,  I2C1_SCL = 7;
TwoWire *I2C_MPU;
TwoWire *I2C_ICM;

// ── 센서 주소 ──
const int ADDR_S1_MPU = 0x68;
const int ADDR_S2_MPU = 0x69;
const int ADDR_S3_ICM = 0x68;
const int ADDR_MAG    = 0x0C;

// ── BLE NUS ──
#define NUS_SERVICE_UUID  "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX_CHAR_UUID  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX_CHAR_UUID  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define BLE_DEVICE_NAME   "AirScribe"

BLEServer*         bleServer          = nullptr;
BLECharacteristic* bleTxChar          = nullptr;
bool               bleClientConnected = false;

// ── 패킷 구조체 (94 bytes, Tri-Node V4) ──
#pragma pack(push, 1)
struct SensorData6 { float ax, ay, az, gx, gy, gz; };
struct SensorData9 { float ax, ay, az, gx, gy, gz, mx, my, mz; };
struct AirWritingPacketV4 {
  uint8_t  header;
  uint16_t seq;
  uint32_t timestamp;
  SensorData6 s1;
  SensorData6 s2;
  SensorData9 s3;
  uint8_t  button;
  uint8_t  checksum;
  uint8_t  footer;
};
#pragma pack(pop)

AirWritingPacketV4 packet;
uint16_t packetSeq = 0;

// ── 링 버퍼 ──
const int RING_BUF_SIZE = 16;
AirWritingPacketV4 ringBuffer[RING_BUF_SIZE];
volatile int rbHead = 0, rbTail = 0;
bool rbEmpty() { return rbHead == rbTail; }
void rbPush(const AirWritingPacketV4 &item) {
  ringBuffer[rbHead] = item;
  rbHead = (rbHead + 1) % RING_BUF_SIZE;
  if (rbHead == rbTail) rbTail = (rbTail + 1) % RING_BUF_SIZE;
}
AirWritingPacketV4* rbPeek() { if (rbEmpty()) return nullptr; return &ringBuffer[rbTail]; }
void rbPop() { if (!rbEmpty()) rbTail = (rbTail + 1) % RING_BUF_SIZE; }

uint8_t stableBtnState      = HIGH;
uint8_t lastUnstableBtnState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// ── FSR ──
unsigned long fsrPressStartMs = 0;
bool fsrOverThreshold = false;
bool fsrLongTriggered = false;

// ── 샘플링 ──
unsigned long lastSampleMicros = 0;
const unsigned long SAMPLE_INTERVAL_US = 10000;

// ─────────────────────────────────────
// BLE 콜백
class AirScribeBLECallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* srv, esp_ble_gatts_cb_param_t *param) override {
    bleClientConnected = true;
    Serial.println("[BLE] 연결됨");
    srv->updateConnParams(param->connect.remote_bda, 6, 12, 0, 200);
  }
  void onDisconnect(BLEServer* srv) override {
    bleClientConnected = false;
    Serial.println("[BLE] 끊김 - 재광고 시작");
    BLEDevice::startAdvertising();
  }
};

void setupBLE() {
  BLEDevice::init(BLE_DEVICE_NAME);
  BLEDevice::setMTU(185);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new AirScribeBLECallbacks());
  BLEService* service = bleServer->createService(NUS_SERVICE_UUID);
  bleTxChar = service->createCharacteristic(NUS_TX_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  bleTxChar->addDescriptor(new BLE2902());
  service->createCharacteristic(NUS_RX_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  service->start();
  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);
  adv->setScanResponse(true);
  adv->setMinPreferred(0x06);
  BLEDevice::startAdvertising();
  Serial.println("[BLE] 광고 시작 - AirScribe (MTU=185)");
}

void sendPacketBLE(const AirWritingPacketV4& pkt) {
  if (!bleClientConnected || !bleTxChar) return;
  uint16_t mtu = BLEDevice::getMTU() - 3;
  if (mtu < 20) mtu = 20;
  const uint8_t* raw = (const uint8_t*)&pkt;
  int total = sizeof(AirWritingPacketV4);
  for (int offset = 0; offset < total; offset += mtu) {
    int chunk = min((int)mtu, total - offset);
    bleTxChar->setValue((uint8_t*)(raw + offset), chunk);
    bleTxChar->notify();
  }
}

// ── 센서 ──
void setupMPU6050(uint8_t addr) {
  I2C_MPU->beginTransmission(addr); I2C_MPU->write(0x6B); I2C_MPU->write(0x01); I2C_MPU->endTransmission(true);
  I2C_MPU->beginTransmission(addr); I2C_MPU->write(0x1C); I2C_MPU->write(0x10); I2C_MPU->endTransmission(true);
  I2C_MPU->beginTransmission(addr); I2C_MPU->write(0x1B); I2C_MPU->write(0x10); I2C_MPU->endTransmission(true);
}

void setICMBank(uint8_t bank) {
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x7F); I2C_ICM->write(bank << 4);
  I2C_ICM->endTransmission(true);
}

void setupICM20948() {
  setICMBank(0);
  I2C_ICM->beginTransmission(ADDR_S3_ICM); I2C_ICM->write(0x06); I2C_ICM->write(0x01); I2C_ICM->endTransmission(true); delay(10);
  I2C_ICM->beginTransmission(ADDR_S3_ICM); I2C_ICM->write(0x03); I2C_ICM->write(0x00); I2C_ICM->endTransmission(true); delay(10);
  I2C_ICM->beginTransmission(ADDR_S3_ICM); I2C_ICM->write(0x0F); I2C_ICM->write(0x02); I2C_ICM->endTransmission(true); delay(10);
  I2C_ICM->beginTransmission(ADDR_MAG);    I2C_ICM->write(0x31); I2C_ICM->write(0x08); I2C_ICM->endTransmission(true); delay(10);
  setICMBank(2);
  I2C_ICM->beginTransmission(ADDR_S3_ICM); I2C_ICM->write(0x14); I2C_ICM->write(0x04); I2C_ICM->endTransmission(true);
  I2C_ICM->beginTransmission(ADDR_S3_ICM); I2C_ICM->write(0x01); I2C_ICM->write(0x04); I2C_ICM->endTransmission(true);
  setICMBank(0);
}

void readMPU6050(uint8_t addr, SensorData6 &data) {
  I2C_MPU->beginTransmission(addr); I2C_MPU->write(0x3B); I2C_MPU->endTransmission(false);
  I2C_MPU->requestFrom((int)addr, 14, (int)true);
  if (I2C_MPU->available() == 14) {
    int16_t ax=(I2C_MPU->read()<<8|I2C_MPU->read()), ay=(I2C_MPU->read()<<8|I2C_MPU->read()), az=(I2C_MPU->read()<<8|I2C_MPU->read());
    I2C_MPU->read(); I2C_MPU->read();
    int16_t gx=(I2C_MPU->read()<<8|I2C_MPU->read()), gy=(I2C_MPU->read()<<8|I2C_MPU->read()), gz=(I2C_MPU->read()<<8|I2C_MPU->read());
    data.ax=ax*(9.81f/4096.0f); data.ay=ay*(9.81f/4096.0f); data.az=az*(9.81f/4096.0f);
    data.gx=gx*(3.14159f/180.0f/32.8f); data.gy=gy*(3.14159f/180.0f/32.8f); data.gz=gz*(3.14159f/180.0f/32.8f);
  } else { data = {}; }
}

void readICM20948(SensorData9 &data) {
  I2C_ICM->beginTransmission(ADDR_S3_ICM); I2C_ICM->write(0x2D); I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_S3_ICM, 14, (int)true);
  if (I2C_ICM->available() == 14) {
    int16_t ax=(I2C_ICM->read()<<8|I2C_ICM->read()), ay=(I2C_ICM->read()<<8|I2C_ICM->read()), az=(I2C_ICM->read()<<8|I2C_ICM->read());
    int16_t gx=(I2C_ICM->read()<<8|I2C_ICM->read()), gy=(I2C_ICM->read()<<8|I2C_ICM->read()), gz=(I2C_ICM->read()<<8|I2C_ICM->read());
    data.ax=ax*(9.81f/4096.0f); data.ay=ay*(9.81f/4096.0f); data.az=az*(9.81f/4096.0f);
    data.gx=gx*(3.14159f/180.0f/32.8f); data.gy=gy*(3.14159f/180.0f/32.8f); data.gz=gz*(3.14159f/180.0f/32.8f);
  } else { data.ax=data.ay=data.az=data.gx=data.gy=data.gz=0; }
  I2C_ICM->beginTransmission(ADDR_MAG); I2C_ICM->write(0x11); I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_MAG, 7, (int)true);
  if (I2C_ICM->available() == 7) {
    int16_t mx=(I2C_ICM->read()|(I2C_ICM->read()<<8)), my=(I2C_ICM->read()|(I2C_ICM->read()<<8)), mz=(I2C_ICM->read()|(I2C_ICM->read()<<8));
    I2C_ICM->read();
    data.mx=mx*0.15f; data.my=my*0.15f; data.mz=mz*0.15f;
  } else { data.mx=data.my=data.mz=0; }
}

// ── FSR + 버튼 상태 ──
// button=0: pen up (안 누름)
// button=1: pen down (FSR 누름 또는 물리버튼)
// button=2: 영점 리셋 (FSR 3초 장압)
uint8_t readButtonState() {
  unsigned long now = millis();
  int fsrValue = analogRead(FSR_PIN);
  // 풀다운 배선: 안 누름=낮은값, 누름=높은값
  if (fsrValue >= FSR_THRESHOLD) {
    if (!fsrOverThreshold) {
      fsrOverThreshold = true; fsrPressStartMs = now; fsrLongTriggered = false;
    } else if (!fsrLongTriggered && (now - fsrPressStartMs >= FSR_LONGPRESS_MS)) {
      fsrLongTriggered = true;
      Serial.println("[FSR] 장압 3s -> button=2 (영점 리셋)");
      return 2;
    }
    if (fsrLongTriggered) return 2;
    return 1;  // FSR 누름 → pen down
  } else {
    fsrOverThreshold = false; fsrLongTriggered = false; fsrPressStartMs = 0;
  }
  return 0;  // FSR 안 누름 → pen up
}

// ─────────────────────────────────────
void setup() {
  Serial.begin(921600);
  delay(500);
  Serial.println("[v3.4] AirScribe USB+BLE starting...");

  esp_err_t nvsRet = nvs_flash_init();
  if (nvsRet == ESP_ERR_NVS_NO_FREE_PAGES || nvsRet == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    Serial.println("[NVS] 손상 감지 -> 초기화 중...");
    nvs_flash_erase();
    nvs_flash_init();
    Serial.println("[NVS] 초기화 완료");
  } else {
    Serial.println("[NVS] OK");
  }

  I2C_MPU = new TwoWire(0);
  I2C_ICM = new TwoWire(1);
  I2C_MPU->begin(I2C0_SDA, I2C0_SCL, 400000);
  I2C_ICM->begin(I2C1_SDA, I2C1_SCL, 400000);

  pinMode(FSR_PIN, INPUT);
  analogReadResolution(12);

  setupMPU6050(ADDR_S1_MPU);
  setupMPU6050(ADDR_S2_MPU);
  setupICM20948();

  setupBLE();

  memset(&packet, 0, sizeof(packet));
  packet.header = 0xAA;
  packet.footer = 0x55;

  Serial.println("[v3.4] Ready! USB+BLE 100Hz streaming...");
}

void loop() {
  unsigned long nowMicros = micros();

  if (nowMicros - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;
    if (nowMicros - lastSampleMicros > SAMPLE_INTERVAL_US * 3)
      lastSampleMicros = nowMicros;

    packet.seq = packetSeq++;
    packet.timestamp = millis();

    readMPU6050(ADDR_S1_MPU, packet.s1);
    readMPU6050(ADDR_S2_MPU, packet.s2);
    readICM20948(packet.s3);

    // FSR만 사용 (물리버튼 없음)

    packet.button = readButtonState();

    // 체크섬
    uint8_t *ptr = (uint8_t *)&packet;
    uint8_t cksum = 0;
    for (int i = 1; i < (int)sizeof(AirWritingPacketV4) - 2; i++) cksum ^= ptr[i];
    packet.checksum = cksum;

    rbPush(packet);
  }

  // ── 전송 (BLE 우선, USB Serial은 연결 시에만) ──
  while (!rbEmpty()) {
    AirWritingPacketV4 *pkt = rbPeek();
    sendPacketBLE(*pkt);
    if (Serial) {
      Serial.write((uint8_t *)pkt, sizeof(AirWritingPacketV4));
    }
    rbPop();
  }
}
