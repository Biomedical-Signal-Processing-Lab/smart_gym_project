#include <ArduinoBLE.h>
#include <Arduino_LSM6DS3.h>
#include <math.h>

// ===== BLE UUID (Nordic UART Service) =====
const char* DEVICE_NAME  = "NANO33_UART";
const char* SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
const char* TX_UUID      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";
const char* RX_UUID      = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";

BLEService uartService(SERVICE_UUID);
BLECharacteristic txChar(TX_UUID, BLENotify | BLERead, 128);
BLECharacteristic rxChar(RX_UUID, BLEWrite | BLEWriteWithoutResponse, 128);

// ===== 하드웨어 =====
const int ECG_PIN = A0;

// ===== 설정값 =====
const unsigned long TARGET_PERIOD_MS = 10;    // 200Hz
const unsigned long GYRO_BIAS_CAL_MS = 2000;  // 바이어스 캘리브
const float TAU_CF = 0.5f;                    // 보완필터 τ
const bool USE_GY_AS_PITCH = true;            // Gy축을 pitch속도로 사용
const bool USE_VEL_LPF = true;
const float VEL_LPF_TAU = 0.05f;
const float ACC_EMA_ALPHA = 0.2f;
const float V_THRESH = 10.0f;
const float V_STOP   = 5.0f;
const float MIN_REP_SEC = 0.6f;

// ===== 내부 상태 =====
float gx_bias=0, gy_bias=0, gz_bias=0;
float ax_lp=0, ay_lp=0, az_lp=0;
float pitch_deg=0, pitch_offset=0;
float vel_lp=0.0f;
float pitch_vel=0.0f;
int state = 0, rep_id = 0;
bool have_prev_bottom = false;
unsigned long prev_bottom_ms = 0;
unsigned long last_us = 0;

// ===== 자이로 바이어스 보정 =====
void calibrateGyroBias(unsigned long ms) {
  Serial.println("[Calib] 자이로 바이어스 캘리브 중...");
  unsigned long t0 = millis();
  long n=0; double sx=0, sy=0, sz=0;
  while (millis() - t0 < ms) {
    float gx, gy, gz;
    if (IMU.readGyroscope(gx, gy, gz)) {
      sx += gx; sy += gy; sz += gz; n++;
    }
  }
  if (n > 0) {
    gx_bias = sx/n; gy_bias = sy/n; gz_bias = sz/n;
  }
  Serial.println("[Calib] 완료!");
}

// ===== setup =====
void setup() {
  Serial.begin(115200);
  while(!Serial);

  if (!IMU.begin()) {
    Serial.println("IMU 초기화 실패!");
    while(1);
  }
  calibrateGyroBias(GYRO_BIAS_CAL_MS);

  // BLE init
  if (!BLE.begin()) {
    Serial.println("BLE 초기화 실패!");
    while(1);
  }

  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setAdvertisedService(uartService);
  uartService.addCharacteristic(txChar);
  uartService.addCharacteristic(rxChar);
  BLE.addService(uartService);
  BLE.advertise();
  Serial.println("[BLE] Advertising 시작!");

  // 초기 pitch 오프셋
  float ax, ay, az;
  if (IMU.readAcceleration(ax, ay, az)) {
    ax_lp = ax; ay_lp = ay; az_lp = az;
    float accPitch = atan2(ax, sqrt(ay*ay + az*az)) * 180.0/PI;
    pitch_deg = accPitch;
    pitch_offset = pitch_deg;
  }

  last_us = micros();
}

// ===== loop =====
void loop() {
  BLEDevice central = BLE.central();
  if (!central) return;

  Serial.print("[BLE] 연결됨: ");
  Serial.println(central.address());

  while (central.connected()) {
    unsigned long now_ms = millis();
    static unsigned long last_ms = 0;
    if (now_ms - last_ms < TARGET_PERIOD_MS) continue;
    last_ms = now_ms;

    unsigned long now_us = micros();
    float dt = (now_us - last_us) / 1e6f;
    if (dt <= 0) dt = TARGET_PERIOD_MS / 1000.0f;
    last_us = now_us;

    float ax, ay, az, gx, gy, gz;
    if (!IMU.readAcceleration(ax, ay, az)) continue;
    if (!IMU.readGyroscope(gx, gy, gz)) continue;

    // --- 바이어스 보정 ---
    gx -= gx_bias; gy -= gy_bias; gz -= gz_bias;

    // --- 가속도 EMA ---
    ax_lp = ACC_EMA_ALPHA*ax + (1-ACC_EMA_ALPHA)*ax_lp;
    ay_lp = ACC_EMA_ALPHA*ay + (1-ACC_EMA_ALPHA)*ay_lp;
    az_lp = ACC_EMA_ALPHA*az + (1-ACC_EMA_ALPHA)*az_lp;

    // --- 보완필터 ---
    float accPitch = atan2(ax_lp, sqrt(ay_lp*ay_lp + az_lp*az_lp)) * 180.0/PI;
    float g_pitch = USE_GY_AS_PITCH ? gy : gx;
    float alpha = TAU_CF / (TAU_CF + dt);
    pitch_deg = alpha*(pitch_deg + g_pitch*dt) + (1-alpha)*accPitch;

    // --- pitch offset 적용 ---
    float pitch0 = pitch_deg - pitch_offset;

    // --- 속도 LPF ---
    if (USE_VEL_LPF) {
      float a_v = dt / (VEL_LPF_TAU + dt);
      vel_lp += a_v * (g_pitch - vel_lp);
      pitch_vel = vel_lp;
    } else {
      pitch_vel = g_pitch;
    }

    // --- 상태 판정 ---
    int s = state;
    if (pitch_vel >= +V_THRESH) s = -1;
    else if (pitch_vel <= -V_THRESH) s = +1;
    else if (fabs(pitch_vel) < V_STOP) s = 0;
    if (s != state) state = s;

    // --- rep 판정 ---
    static bool seenDesc=false, seenAsc=false;
    if (state == -1) { seenDesc = true; seenAsc = false; }
    if (state == +1 && seenDesc) { seenAsc = true; }
    if (state == 0 && seenDesc && seenAsc) {
      float dt_sec = have_prev_bottom ? (now_ms - prev_bottom_ms)/1000.0f : 1.0f;
      if (dt_sec >= MIN_REP_SEC) {
        rep_id++;
        prev_bottom_ms = now_ms;
        have_prev_bottom = true;
      }
      seenDesc = false; seenAsc = false;
    }

    // --- ECG raw ---
    int ecg_raw = analogRead(ECG_PIN);

    // --- 6필드 전송 ---
    char buf[100];
    snprintf(buf, sizeof(buf), "%lu,%.3f,%.3f,%d,%d,%d\n",
             now_ms, pitch0, pitch_vel, state, rep_id, ecg_raw);
    txChar.writeValue((uint8_t*)buf, strlen(buf));
    Serial.print(buf);
  }

  Serial.println("[BLE] 연결 종료됨");
}
