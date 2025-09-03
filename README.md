# **안녕하세요 여러분**
## “eye tracker raw output(좌표/각도)” → “object selection event” 로 넘어가는 중간 계층 규칙
- 단순히 눈동자 좌표 찍는 수준에서 끝나는 게 아니라, 어떤 조건에서 “사용자가 객체를 보았다”로 인정할지를 정의하는 게 진짜 시스템 완성도를 가름함.

### 🔹 아키텍처: Gaze → Fixation → Object Selection

1. Raw Gaze Estimation

- 모델(Mediapipe Iris, RT-GENE 등)로부터 매 프레임마다 눈동자 중심 좌표 혹은 yaw/pitch 벡터 획득

 - 좌표계: 화면 pixel, 혹은 camera extrinsic 좌표계

2. Gaze Event Processing

- Fixation detection: 시선이 일정 범위 내에서 유지되면 “fixation”이라고 판단

 - Saccade detection: 빠르게 이동하면 단순 eye movement (객체 선택 아님)

 - Smooth pursuit: 천천히 따라가는 움직임 (보통은 객체 추적 맥락에서만 인정)

3. Object Selection Logic

- 화면에 있는 bounding box(객체 후보)들과 시선 좌표를 매칭

 - fixated point가 특정 bbox 안에 일정 시간 머물렀을 때 → 선택

 - 다중 후보 시에는 가장 오래 fixated된 객체, 혹은 중심점과 가장 가까운 bbox 선택

### 🔹 핵심 인식 조건들 (객체 선택 기준)
1. 속도 기반  

- Saccade (도약):

눈동자 속도 > ~30–40°/s (디스플레이 상 대략 200–400 px/s)  

→ “빠른 이동”으로 간주, 선택 무효

- Fixation (응시):

속도 < 5°/s  

→ “한 지점 응시”로 인정

2. 위치 기반  

- Dispersion threshold (좌표 퍼짐도):

예: 시선 좌표 표준편차 < 1.0° (약 30 px)  

→ 일정 영역 안에서만 눈동자 분포 허용

- Spatial tolerance:

bbox 중심에서 반경 r 이내 (화면 좌표에서 50–100 px, 시야각 2–3° 정도)  

3. 시간 기반

- Minimum fixation duration:

최소 100–150 ms 이상 머물러야 “fixation”

보통 3–5 frame 이상 (30fps 기준)

- Selection dwell time:

객체 선택 인식은 300–500 ms (사용자 지향성 보장)

너무 짧으면 오탐(깜빡임), 너무 길면 반응 느려짐

### 🔹 시스템 전체 아키텍처 개요
[카메라 입력]   
   ↓  
[Eye tracker] (gaze 좌표 추정, yaw/pitch 벡터)  
   ↓  
[Gaze event processor]   
   ├─ Fixation detection (속도, 위치 분산, 시간)  
   ├─ Saccade filtering  
   ↓  
[Object candidate matcher]  
   ├─ Gaze point ↔ Object bbox overlap 검사  
   ├─ Temporal smoothing (최근 0.5–1s 시선 기록 활용)  
   ↓  
[Selection logic]  
   ├─ Dwell time 충족? → 선택 이벤트 발생  
   ├─ 캐싱된 객체 좌표 활용 (latency 감소)  
   ↓  
[AR action executor]  
   └─ 확대/검색/저장 등 명령 수행  

### 🔹 완성도 좌우하는 양각(두 가지 축) 조건

민감도(Sensitivity)

짧은 fixation에도 반응 (빠른 인터랙션 가능)

하지만 오탐↑ (깜빡임·우연 응시도 선택됨)

특이도(Specificity)

긴 dwell time + 좁은 분산 허용 (정확도↑)

하지만 반응 느려짐 (UX↓)

→ 밸런스가 관건: 프로토타입에선 dwell time 300–500ms, dispersion threshold 1–2° 정도가 가장 무난.

✅ 요약:

**“눈동자 좌표 잡기”**는 그냥 raw 신호일 뿐이고,

“fixation/saccade 판별 + dwell time 기반 선택” 규칙을 설계해야만 “객체 선택 시스템”으로 완성됨.

최적의 지표: 속도(°/s), 분산(px), 시간(ms) 세 축으로 임계값을 두고 튜닝.