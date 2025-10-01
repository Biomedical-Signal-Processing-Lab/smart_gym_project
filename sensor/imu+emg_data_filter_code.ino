#include <Arduino_LSM6DS3.h>                    // ⟵ Nano 33 IoT 내장 IMU(LSM6DS3) 라이브러리 포함
#include <math.h>                               // ⟵ 수학 함수(atan2, sqrt 등) 사용

// ====== 출력 포맷 (고정) ======
// timestamp_ms,pitch_deg,pitch_vel_dps,state,rep_id,ecg_raw  // ⟵ 시리얼 CSV 컬럼 설명

// ====== 하드웨어 ======
const int ECG_PIN = A0;                         // ⟵ ECG/EMG 대용의 아날로그 입력 핀(ADC)

// ====== 설정값 ======
const unsigned long TARGET_PERIOD_MS = 10;      // ⟵ 메인 루프 출력 주기 목표(10ms ≈ 100Hz)
const unsigned long GYRO_BIAS_CAL_MS = 2000;    // ⟵ 자이로 바이어스 캘리브레이션(정지) 시간(ms)

// 보완필터 (gy 중심, acc 보정)
const float TAU_CF = 0.5f;                      // ⟵ 보완필터 시정수 τ (클수록 자이로 가중 ↑)

// 속도 임계 (히스테리시스 포함)
const float V_THRESH = 10.0f;                   // ⟵ 상승/하강 판정 기준 각속도(°/s)
const float V_HYST   = 10.0f;                   // ⟵ 히스테리시스 폭(코드에선 정의만; 실제 미사용)
// 정지 정의 (선택): 아주 느린 구간은 0으로 보고 싶다면 사용
const float V_STOP   = 5.0f;                    // ⟵ |속도| < 5 dps 이면 정지로 스냅

// rep 최소 주기
const float MIN_REP_SEC = 0.6f;                 // ⟵ 너무 빠른 잡카운트 방지(반복 최소 간격 s)

// 각도 방향이 반대면 true
bool INVERT_PITCH = false;                      // ⟵ 장착 반전 시 각도/속도 부호 뒤집기

// pitch 회전축: 허벅지 장착 가정
const bool USE_GY_AS_PITCH = true;              // ⟵ true면 gy를 pitch 속도로 사용(아니면 gx 사용)

// 속도 LPF 옵션 (원시 자이로 속도 그대로 쓸지 여부)
const bool USE_VEL_LPF = true;                 // ⟵ true면 속도 1차 IIR 저역통과 활성화
const float VEL_LPF_TAU = 0.05f;                // ⟵ 속도 LPF 시정수 50ms

// 가속도 LPF (EMA)
const float ACC_EMA_ALPHA = 0.2f;               // ⟵ 가속도 지수이동평균 계수(0.1~0.3)

// ====== 내부 상태 ======
float gx_bias=0, gy_bias=0, gz_bias=0;          // ⟵ 자이로 바이어스(오프셋) 저장
float pitch_deg = 0.0f;                          // ⟵ 필터로 추정한 pitch(도)
float pitch_offset = 0.0f;                       // ⟵ 선자세 기준 0도로 만들기 위한 오프셋

// 가속도 LPF
float ax_lp=0, ay_lp=0, az_lp=0;                 // ⟵ EMA 적용된 가속도 값 저장

// 속도 LPF 상태
float vel_lp = 0.0f;                             // ⟵ 속도 IIR 필터 내부 상태

// 상태/rep
int state = 0;                                   // ⟵ -1: 하강, 0: 정지, +1: 상승 (의도 주석)
unsigned long stateStartMs = 0;                  // ⟵ 현재 상태 머문 시작 시각(ms)
int rep_id = 0;                                  // ⟵ 누적 반복 횟수
bool seenDesc = false, seenAsc = false;          // ⟵ (미사용) 전역 플래그 자리

// 타이밍
unsigned long last_us = 0;                       // ⟵ 이전 루프의 micros() 저장

// 바닥(최저점) 기반 rep 주기용 (옵션)
unsigned long prev_bottom_ms = 0;                // ⟵ 마지막 바닥(정지) 시각
bool have_prev_bottom = false;                   // ⟵ 첫 바닥 시각 보유 여부

// ====== 유틸 ======
void calibrateGyroBias(unsigned long ms=2000) {  // ⟵ 자이로 바이어스 평균내는 함수(정지 가정)
  unsigned long t0 = millis();                   // ⟵ 시작 시각
  long n=0; double sx=0, sy=0, sz=0;            // ⟵ 샘플 수와 합계
  while (millis() - t0 < ms) {                   // ⟵ ms 동안 반복
    float gx, gy, gz;                            // ⟵ 임시 자이로 변수
    if (IMU.readGyroscope(gx, gy, gz)) {         // ⟵ 자이로 값 읽기 성공 시
      sx += gx; sy += gy; sz += gz; n++;         // ⟵ 합계/카운트 누적
    }
  }
  if (n > 0) { gx_bias = sx/n; gy_bias = sy/n; gz_bias = sz/n; } // ⟵ 평균 → 바이어스 저장
}

void setup() {
  Serial.begin(115200);                          // ⟵ 시리얼 115200bps 시작
  while(!Serial) {}                              // ⟵ USB 시리얼 준비까지 대기(레오나르도 계열 패턴)

  if (!IMU.begin()) {                            // ⟵ IMU 초기화
    Serial.println("IMU init failed!");          // ⟵ 실패 메시지
    while(1);                                    // ⟵ 멈춤
  }

  // 1) 자이로 바이어스 캘리브레이션 (정지)
  calibrateGyroBias(GYRO_BIAS_CAL_MS);           // ⟵ 정해둔 시간 동안 평균 내서 바이어스 산출

  // 2) 초기 각도 오프셋(서있는 자세 = 0°)
  float ax, ay, az;                              // ⟵ 가속도 임시 변수
  if (IMU.readAcceleration(ax, ay, az)) {        // ⟵ 가속도 읽기
    ax_lp = ax; ay_lp = ay; az_lp = az;          // ⟵ EMA 초기값을 현재 측정으로 설정
    // 가속도만으로 추정한 pitch (허벅지 장착 기준: x축 전후 기울기 가정)
    float accPitch = atan2(ax, sqrt(ay*ay + az*az)) * 180.0/PI; // ⟵ 중력기반 pitch(도)
    pitch_deg = accPitch;                        // ⟵ 초기 pitch를 acc 기준으로 설정
    pitch_offset = pitch_deg;                    // ⟵ 현재 자세를 0도로 만들기 위한 오프셋 저장
  }

  last_us = micros();                            // ⟵ 루프 dt 계산 기준 시각 초기화
  stateStartMs = millis();                       // ⟵ 상태 시작 시각 초기화

  // CSV 헤더 (고정)
  Serial.println("timestamp_ms,pitch_deg,pitch_vel_dps,state,rep_id,ecg_raw"); // ⟵ 컬럼명 출력
}

void loop() {
  static unsigned long last_ms = 0;              // ⟵ 마지막 주기 체크 시각(정적, 1회만 생성)
  unsigned long now_ms = millis();               // ⟵ 현재 ms
  if (now_ms - last_ms < TARGET_PERIOD_MS) return; // ⟵ 주기(10ms) 이전이면 아무것도 하지 않음

  unsigned long now_us = micros();               // ⟵ 현재 us
  float dt = (now_us - last_us) / 1e6f;          // ⟵ 지난 루프와의 경과시간(s)
  if (dt <= 0) dt = TARGET_PERIOD_MS/1000.0f;    // ⟵ us wrap 등 비정상 방지 보정
  last_us = now_us;                               // ⟵ 이번 us 저장
  last_ms = now_ms;                               // ⟵ 이번 ms 저장

  // --- 센서 읽기 ---
  float ax, ay, az, gx, gy, gz;                  // ⟵ 가속도/자이로 임시 변수
  if (!IMU.readAcceleration(ax, ay, az)) return; // ⟵ 가속도 읽기 실패 시 이번 턴 패스
  if (!IMU.readGyroscope(gx, gy, gz)) return;    // ⟵ 자이로 읽기 실패 시 이번 턴 패스

  // 바이어스 보정
  gx -= gx_bias; gy -= gy_bias; gz -= gz_bias;   // ⟵ 미리 구해둔 바이어스로 오프셋 제거

  // 가속도 LPF (EMA)
  ax_lp = ACC_EMA_ALPHA*ax + (1-ACC_EMA_ALPHA)*ax_lp; // ⟵ EMA: 새가중 α, 과거 1-α
  ay_lp = ACC_EMA_ALPHA*ay + (1-ACC_EMA_ALPHA)*ay_lp; // ⟵ y축 EMA
  az_lp = ACC_EMA_ALPHA*az + (1-ACC_EMA_ALPHA)*az_lp; // ⟵ z축 EMA

  // 보완필터 (자이로 적분 + 가속도 보정)
  float accPitch = atan2(ax_lp, sqrt(ay_lp*ay_lp + az_lp*az_lp)) * 180.0/PI; // ⟵ LPF한 가속도로 pitch
  float alpha = TAU_CF / (TAU_CF + dt);           // ⟵ 보완필터 계수 α = τ/(τ+dt)

  // pitch 회전축 선택 (gy 또는 gx)
  float g_pitch = USE_GY_AS_PITCH ? gy : gx;      // ⟵ 선택한 자이로 축을 pitch 속도로 사용

  // 보완필터 갱신
  pitch_deg = alpha*(pitch_deg + g_pitch*dt) + (1.0f - alpha)*accPitch; // ⟵ gyro 적분 + acc 보정

  // 0도 오프셋 및 방향 반전
  float pitch0 = pitch_deg - pitch_offset;        // ⟵ 초기 자세를 0도로 만들기
  if (INVERT_PITCH) pitch0 = -pitch0;             // ⟵ 방향 반전 옵션

  // 각속도 (원시 자이로 → 선택적으로 LPF)
  float pitch_vel = g_pitch;                      // ⟵ 선택 축의 각속도(dps)를 속도로 사용
  if (INVERT_PITCH) pitch_vel = -pitch_vel;       // ⟵ 속도 부호도 반전

  if (USE_VEL_LPF) {                              // ⟵ 속도 LPF 사용하면
    float a_v = dt / (VEL_LPF_TAU + dt);          // ⟵ 1차 IIR 계수
    vel_lp += a_v * (pitch_vel - vel_lp);         // ⟵ IIR 업데이트
    pitch_vel = vel_lp;                            // ⟵ 필터 결과를 속도로 사용
  }

  // ---- 상태판정 (히스테리시스 + 정지대) ----
  // 기본 규칙(주석): vel <= -V_THRESH → DESC(-1), vel >= +V_THRESH → ASC(+1)
  // 실제 코드: 다음 두 줄이 주석과 반대로 매핑됨(주의)
  int s = state;                                   // ⟵ 임시 상태 s에 현재 상태 복사
  if (pitch_vel >= +V_THRESH)       s = -1;        // ⟵ (코드상) 속도 크면 하강으로 설정
  else if (pitch_vel <= -V_THRESH)  s = +1;        // ⟵ (코드상) 속도 음수 크면 상승으로 설정
  else if (fabs(pitch_vel) < V_STOP) s = 0;        // ⟵ 속도가 아주 작으면 정지대

  // (히스테리시스: 임계 바로 주변에서는 이전 상태 유지 효과) // ⟵ 주석만 있고 실제 V_HYST 미적용
  if (s != state) {                                // ⟵ 상태 변하면
    state = s;                                     // ⟵ 새 상태 반영
    stateStartMs = now_ms;                         // ⟵ 상태 시작 시각 갱신
  }

  // ---- rep 카운트 (하강→상승 후 정지에서 +1) ----
  static bool seenDescFlag=false, seenAscFlag=false; // ⟵ 구간 플래그(정적 지역)
  if (state == -1) { seenDescFlag = true; seenAscFlag = false; } // ⟵ 하강 중 감지, 상승 플래그 초기화
  if (state == +1 && seenDescFlag) { seenAscFlag = true; }       // ⟵ 하강을 거친 뒤 상승 감지

  if (state == 0 && seenDescFlag && seenAscFlag) {  // ⟵ 정지에 도달했고 하강→상승을 봤다면
    // 최소 주기 검사(너무 빠른 잡카운트 방지)
    if (!have_prev_bottom) {                        // ⟵ 첫 바닥이면
      prev_bottom_ms = now_ms; have_prev_bottom = true; // ⟵ 기준만 설정
    } else {
      float dt_sec = (now_ms - prev_bottom_ms)/1000.0f; // ⟵ 지난 바닥 이후 경과(s)
      if (dt_sec >= MIN_REP_SEC) {                   // ⟵ 충분히 시간이 지났다면
        rep_id++;                                    // ⟵ 반복 횟수 +1
        prev_bottom_ms = now_ms;                     // ⟵ 바닥 시각 갱신
      }
    }
    seenDescFlag = false; seenAscFlag = false;       // ⟵ 플래그 리셋(다음 rep 준비)
  }

  // ECG/EMG raw 읽기
  int ecg_raw = analogRead(ECG_PIN);                 // ⟵ A0에서 원시 ADC 값(0~1023/4095) 읽기

  // ---- 출력 (요청한 기존 포맷 유지) ----
  if (!isfinite(pitch0) || !isfinite(pitch_vel)) return; // ⟵ NaN/inf 방지
  Serial.print(now_ms);           Serial.print(",");     // ⟵ 타임스탬프(ms)
  Serial.print(pitch0, 3);        Serial.print(",");     // ⟵ pitch(도), 소수 3자리
  Serial.print(pitch_vel, 3);     Serial.print(",");     // ⟵ pitch 속도(dps), 소수 3자리
  Serial.print(state);            Serial.print(",");     // ⟵ 상태(-1/0/+1)
  Serial.print(rep_id);           Serial.print(",");     // ⟵ 누적 rep
  Serial.println(ecg_raw);                                  // ⟵ ADC 원시값(개행)
}
