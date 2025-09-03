# openSmile
## 현재 완료된 것
1. 영상에서 오디오만 추출(openSmile, ffmpeg 사용) → 성공
2. 확률 계산 → 이거 어떻게 구현할지 **의견 구함**. 현재는 다음 “설명”에 있는 내용으로 구현. 하지만 “설명” 상태로 하게 되면 거의 다 급발진으로 판단하는거 같음…(자료를 너무 다양한걸 써서 그런건지 다른 이유인지는 몰라도…) → 이렇게 되면 차 타다가 신나서 노래 부르는 순간에도 급발진으로 잡힐 각…

## 활용된 자료
### 1️. **음성 톤과 높낮이 관련 (Pitch / F0)**

- **F0semitoneFrom27.5Hz_sma3nz_amean** : 평균 피치
- **F0semitoneFrom27.5Hz_sma3nz_stddevNorm** : 피치 변화 폭
- **F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope / meanFallingSlope** : 상승/하강 기울기
- 급발진 상황에서 사람이 **당황하거나 비명을 지르면 톤이 급격히 높아지고 변화가 큼** → F0 관련 feature 중요

---

### 2️. **목소리 흔들림과 잡음 (Jitter / Shimmer / HNR)**

- **jitterLocal_sma3nz_amean** : 목소리 흔들림 정도
- **shimmerLocaldB_sma3nz_amean** : 음량 변동의 미세한 변화
- **HNRdBACF_sma3nz_amean** : 하모닉 대비 잡음 비율
- 급발진 상황처럼 긴장/공포 상태에서는 **목소리가 떨리거나 불안정** → Jitter, Shimmer, HNR 증가

---

### 3️. **음량과 소리 강도 (Loudness / Energy)**

- **loudness_sma3_amean / stddevNorm / percentile50.0 등** : 평균 음량, 표준편차, 중앙값
- **loudnessPeaksPerSec** : 초당 음량 급상승 횟수
- **VoicedSegmentsPerSec** : 초당 발성 구간 수
- 큰 소리, 반복적인 경고음, 비명 → Loudness 관련 feature 상승

---

### 4️. **스펙트럼 / 음색 특징 (Spectral Features)**

- **alphaRatioV / alphaRatioUV, hammarbergIndexV / UV** : 고주파 대비 저주파 비율, 음색
- **slopeV0-500 / slopeV500-1500** : 특정 대역에서 에너지 기울기
- 긴장/공포 목소리는 **고주파 성분이 늘어나고 음색이 날카로워지는 경향** → Spectral feature 변화

---

### 5️. **세그먼트 관련 (Segment / Voice Activity)**

- **MeanVoicedSegmentLengthSec / StddevVoicedSegmentLengthSec** : 발성 구간 길이
- **MeanUnvoicedSegmentLength / StddevUnvoicedSegmentLength** : 무성 구간 길이
- 급발진 상황에서는 **짧은 비명, 짧은 호흡 반복** → Segment feature 변화

## 구현
1. ffmpeg로 영상(mp4 등) → 오디오(WAV, 16kHz/mono) 추출

2. openSMILE (GeMAPSv01b/eGeMAPSv02 등) 기능을 사용해 기능(Functional) CSV 출력

- (가능하면) LLD CSV도 함께 출력하여 시간축 기반 파생 지표 계산

3. 아래 지표를 산출

- Pitch/F0: 평균, 표준편차 정규화, 상승/하강 기울기

- Jitter/Shimmer/HNR

- Loudness: 평균/표준편차정규화/중앙값 + 초당 피크(loudnessPeaksPerSec)

- Segments: 발성/무성 세그먼트 길이 통계, 초당 발성 구간 수(VoicedSegmentsPerSec)

- Spectral: alphaRatio(V/UV), hammarbergIndex(V/UV), slopeV0-500, slopeV500-1500 등

4. 간단한 규칙기반 지표 플래그와 요약 리포트 생성 → 이걸 통해서 확률 계산

## 실험(영상 다운은 노션에 있음)

sample1 : 급발진 상황 o / sample2: x / sample3: o / sample4: x

## 결과

- sample1
    
    === 급발진 확률 분석 ===
    확률: 100.0%
    활성 플래그: ['F0 변화 폭이 큼', 'F0 상승/하강 기울기 큼', 'F0 상승/하강 기울기 큼', 'Jitter 증가(떨림)', 'Shimmer 증가(음량 미세 변동)', 'HNR 낮음(잡음↑)', '음량 피크 빈도 증가', '발성 구간 평균 길이가 매우 짧음']
    
- sample2
    
    === 급발진 확률 분석 ===
    확률: 100.0%
    활성 플래그: ['F0 변화 폭이 큼', 'F0 상승/하강 기울기 큼', 'F0 상승/하강 기울기 큼', 'Jitter 증가(떨림)', 'Shimmer 증가(음량 미세 변동)', 'HNR 낮음(잡음↑)', '음량 피크 빈도 증가', '초당 발성 구간 수 많음', '발성 구간 평균 길이가 매우 짧음']
    
- sample3
    
    === 급발진 확률 분석 ===
    확률: 100.0%
    활성 플래그: ['F0 변화 폭이 큼', 'F0 상승/하강 기울기 큼', 'F0 상승/하강 기울기 큼', 'Jitter 증가(떨림)', 'Shimmer 증가(음량 미세 변동)', 'HNR 낮음(잡음↑)', '음량 피크 빈도 증가', '발성 구간 평균 길이가 매우 짧음']
    
- sample4
    
    === 급발진 확률 분석 ===
    확률: 100.0%
    활성 플래그: ['F0 변화 폭이 큼', 'F0 상승/하강 기울기 큼', 'F0 상승/하강 기울기 큼', 'Jitter 증가(떨림)', 'Shimmer 증가(음량 미세 변동)', 'HNR 낮음(잡음↑)', '음량 피크 빈도 증가', '발성 구간 평균 길이가 매우 짧음']
