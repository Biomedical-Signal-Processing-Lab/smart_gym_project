# **AI Smart Gym**

## **프로젝트 개요 (Overview)**

### 포즈 랜드마크 기반 운동 분류 + 실시간 분석 앱, 그리고 EMG·IMU 센서 융합 파워리프팅 스쿼트 분석까지 지원하는 파이썬 중심 프로젝트입니다. 애플리케이션은 Raspberry Pi에서 구동되며, 카메라·센서 스트림을 받아 운동을 인식하고, 자세·가동범위(ROM)·템포 등 핵심 지표를 화면과 로그로 제공합니다.

---

## **프로젝트 목표**

- 운동 동작의 **정확한 분류 및 실시간 분석**
- IMU, EMG 등 센서를 통한 **정량적 운동 데이터 수집**
- AI 기반 운동 수행 **평가 알고리즘 및 피드백 제공**
- Raspberry Pi / Hailo-8 **경량화 및 실시간 처리**

---

## **구현 범위 (Minimum Viable Product)**

- 목표: RPi5+Hailo-8에서 스쿼트 실시간 카운트/점수/요약 제공
- 입력: 측면 1카메라(단일 인원)
- 처리: YOLOv8-Pose → 각도(무릎/힙) → 카운트 FSM → 간단 점수
- 출력: 오버레이(UI), 세션 리포트(SQLite),
- 성능: FPS≥15, 지연<250ms, 카운트 정확도≥90%
- 비범위: 다중인원/EMG/클라우드/3D
- 위험대응: 런타임 중간에 프레임 드랍 현상
- 데모: 3회 수행→실시간 표시→요약 확인

## 구현( Done )

- [x]  **카메라 입력 파이프라인**(OpenCV 경로, 해상도 고정)
- [x]  **YOLOv8-Pose(ONNX) 추론** 및 COCO17 키포인트 추출
- [x]  **각도 계산 모듈**(무릎/힙) + 기본 안정화(클램프/스무딩)
- [x]  **7개 운동 카운트 로직 설계**
- [x]  **점수 매핑**(Good/OK/Retry) & **화면 오버레이(Pyside6)**
- [x]  **HailoCamAdapter API** (`frame()/people()/meta`)
- [x]  **세션 요약 저장**(SQLite 최소 컬럼)
- [x]  **데모 스크립트**(3회 수행 → 요약 확인)
- [x]  **EMG/IMU 센서 데이터 수집 구현 (Arduino → pi5 Bluetooth 전송)**
- [x]  **센서 데이터로 근육 피로도, 불균형 판단 모델 아키텍처 구현 및 프로토타입 모델 생성**

- [ ]  **품질관리 기준을 충족한 데이터셋 수집**
    **사유**: 프로젝트 사양을 만족하는 EMG 센서의 국내 공급망 제약과 **추석 연휴 물류 지연**으로 **마감 5일 전 수령**.
    
- [ ]  ** yocto **
       **사유**: 깃에서 모든 코드를 당겨오지 못해서 오류가 날때마다 필요한 부분 패치 하여서(우회하여서) 사용했으나
                 빌드 하는중에 패치한 코드안에서 디렉토리를 찾지 못하는 문제가 생김 
		

## **주요 기능**

### **운동 분류 모델 (AI)**

- TCN **동작 분류 모델**
- 영상 기반 포즈 추정과 센서 데이터 결합
- ONNX, hef 기반 **경량 추론 및 시각화**

### **운동 분석 알고리즘**

- 속도, 가속도, 파워 등 **운동 성능 지표 계산**
- 센서 데이터 기반 **운동 품질 평가 로직 설계**

### **센서 하드웨어 데이터 수집**

- IMU, EMG, Force Sensor 등 실시간 데이터 수집
- I2C / SPI / UART 통신 기반 데이터 취득
- 노이즈 필터링 및 캘리브레이션 적용

### **통합 어플리케이션**

- AI, 알고리즘, 센서 모듈 통합
- PySide6 기반 UI 시각화 및 리포트 출력
- BLE / Wi-Fi 통신을 통한 데이터 연동

---

## **시스템 구성도**

<p align="center"> <img src="assets/system_architecture.png" alt="System Architecture" width="85%"> </p>

---

## **Team: 자세어때**

### **팀 구성**

| **이름** | **역할** | **주요 담당** |  |
| --- | --- | --- | --- |
| 서민솔 | 팀장 | 프로젝트 총괄, 운동 분류 모델 설계 |  |
| 이동현 | 부팀장 | 통합 어플리케이션 시스템 개발 |  |
| 유종민 | 센서 | 센서 신호처리 AI 개발, 3D 모델링 |  |
| 윤찬민 | AI 개발 | 운동 분류 모델 개발 |  |
| 임정민 | 운동 분석 | 운동 분석 알고리즘 개발 |  |

---

## **기술 스택**

| **분야** | **기술 스택** |
| --- | --- |
| AI / ML | PyTorch, ONNX, TCN |
| 임베디드 | Raspberry Pi, Hailo-8 |
| 센서 | IMU, EMG, Load Cell, Force Sensor |
| 프론트엔드 / 앱 | PySide6, BLE 통신, Python |
| 기타 | YAML Config, CSV/JSON Logging |

<p align="center"> <img src="https://skillicons.dev/icons?i=python,pytorch,raspberrypi,arduino,opencv,linux" height="45" /> <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/PySide_logo.png" height="45" alt="PySide6" /> <img src="https://hailo.ai/wp-content/uploads/2023/04/hailo-logo.svg" height="45" alt="Hailo" /> <img src="https://onnxruntime.ai/images/onnxruntime-logo.svg" height="45" alt="ONNX Runtime" /> <img width="1217" height="657" alt="image" src="https://github.com/user-attachments/assets/18c66f81-7b4d-4bad-86aa-9a49e9cd3659" /></p><img width="874" height="515" alt="image" src="https://github.com/user-attachments/assets/0876492a-0212-499d-b1fa-2304882e3de1" />


---

## **기대 효과**

- 운동 수행 정확도 향상 및 **부상 예방**
- 개인 맞춤형 피드백을 통한 **훈련 효율 극대화**
- AI + 센서 융합을 통한 **스마트 피트니스 솔루션 실현**

---

## **시연 예시**

<p align="center"> <img src="assets/demo.gif" alt="Demo" width="70%"> </p>

---

## **Clone code**

```
git clone https://github.com/Biomedical-Signal-Processing-Lab/smart_gym_project.git
```

## **Prerequisites**
```
Hardware: Raspberry Pi 5, Hailo-8, 1.8 mm wide-angle camera
OS: Raspberry Pi OS Bookworm (64-bit)
Hailo Packages: hailo-all  # drivers + HailoRT + GStreamer + tools
Python: 3.12 (recommended; use venv)
```
## **Steps to build**
```
# 0) 기본
```
sudo apt update
sudo apt install -y git curl wget build-essential pkg-config
python -m .venv .sgym_venv
source .sgym_venv/bin/activate
cd smart_gym_project/app
pip install -r requirements.txt
```
# 1) Hailo (공식 APT 레포 추가 후 설치)  ← 벤더 문서 절차로 레포 먼저 추가
```
sudo apt install -y hailo-all
```
# 2) GStreamer 런타임 + 플러그인 묶음
```
	sudo apt install -y \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  gstreamer1.0-gl gstreamer1.0-alsa
```
# 3) GI(PyGObject) 바인딩 (파이썬에서 GStreamer 쓰는 경우)
```
	sudo apt install -y \
  python3-gi python3-gi-cairo gobject-introspection \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 libgirepository1.0-dev
```
# 4) 카메라 유틸
```
sudo apt install -y v4l-utils libcamera-apps
```
## **Step to run**
python main.py

# **OutPut**


## 운동 분류 시스템
운동 분류 시스템은 **포즈 랜드마크 추출 → 시퀀스 분류 → (상세 분석) → (후속 처리)** 순서로 동작


---

## **1 영상 입력·전처리 (Raspberry Pi 5 + Hailo-8)**

- 프레임 리사이즈
- 색공간 변환

## **2 포즈 키포인트 추출**

- **YOLOv8s-Pose @ Hailo-8**
- 매 프레임 키포인트 추출
- 키포인트 정규화: 스케일·중심

## **3 시퀀스 버퍼링**

- 윈도우: **60 프레임**
- 실시간 **stride = 1**
- 슬라이딩 업데이트

## **4 운동 분류 (ONNX TCN)**

- 입력: 정규화 키포인트 시퀀스(60 프레임)
- 분류 클래스: **idle / shorder_press / 덤벨로우 / 점핑잭 / 스쿼트 / 푸쉬업 / 레그레이즈 / 버피 / 사이드 레터럴 레이즈**
- (옵션) 히스테리시스·스무딩

### 후속 처리(운동 채점)
- 각도 기반 공통 채점: 각 운동 자세 채점에 필요한 관절만 사용
- 가중치·진행률: 관절별 가중치치, 목표 각도 범위 대비 progress(0~1) 계산
- 점수 산출: 한 동작의 인식 가동 범위 내에서 최고혹은 최저 각으로 산출




## 운동 분석 시스템
운동 분석 시스템은 **센서 데이터 수짐 → 전처리 → 전송 → 신호 분석 → 추론
- arduino 33 iot 에서 근전도 센서 2개와 imu센서 데이터 수집
## 2 전처리
- 
- 
- 
## 3 데이터 전송
- 블루투스로 전송
## 4. 신호 분석

## 5. 추론



## **프로젝트 결과**

## **기대 효과**

- 운동 수행 정확도 향상 및 부상 예방
- 개인 맞춤형 피드백을 통한 훈련 효율 극대화
- AI + 센서 융합을 통한 스마트 피트니스 솔루션 실현
- 실시간 채점 및 시각화로 재미 향상

---

## **시행착오 및 해결방안**

| **No.** | **시행착오** | **해결 방안** | **결과/교훈** |
| --- | --- | --- | --- |
| 1 | 프로토타입(**MediaPipe**) ↔ 배포(**Hailo-8, COCO-17**) **포즈 스키마 불일치**로 입력 붕괴 | **COCO-17 기준 전 파이프라인 정렬**(키포인트 매핑/좌표·정규화 통일) + 데이터/레이블 **재생성·재학습** | 배포 타깃 스키마 **데이터 계약** 고정 · 스키마 변경 시 **어댑터·회귀 테스트·버전 태깅** 필수 |
| 1 | 오프라인 지표 우수 ↔ 실사용에서 **숄더프레스·덤벨로우 혼동**(유사 패턴, 시간 특성 미활용/지름길) | 정규화 키포인트에 **속도·가속도(1차/2차 차분)** 채널 추가 → **TCN 입력 다채널화**(윈도우 60, stride 1) | 지름길 억제 · 유사 클래스 **분리도↑** · 경계 흔들림 **완화**(히스테리시스/스무딩) |
| 3 | 기본 카메라 **FOV 협소**로 공간 제약·스케일 변동 | **광각 카메라 전환**, **왜곡 보정 없이** 광각 전용 데이터 **재수집·재학습** | 배포 **광학 스펙을 데이터 계약**으로 고정 · 보정 미적용 시 **훈련=추론 조건 일치** 유지 |

## **개발 후기**

## **Appendix**
