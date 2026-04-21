# RVC AI 음성 변환 프로젝트 구축 계획

이 프로젝트는 Retrieval-based Voice Conversion (RVC) 프레임워크를 사용하여 가수 **규현**의 목소리 학습 모델을 구축하는 것을 목표로 합니다. 사용자가 제시한 전략을 바탕으로 고품질의 결과물을 얻을 수 있도록 구성합니다.

## 전략 리뷰
제시해주신 전략은 매우 전문적이고 효과적입니다:
- **20분 데이터**: 고품질 모델을 위한 최적의 분량입니다.
- **슬라이싱 (Slicing)**: 학습 효율과 메모리 관리를 위해 필수적입니다.
- **단계별 학습 (50 -> 200 Epochs)**: 데이터의 적합성을 먼저 확인하여 시간을 절약하는 스마트한 접근입니다.
- **Harvest F0**: 보컬의 음정 안정성을 위해 가장 추천되는 방식입니다.

## 사용자 확인 사항

> [!IMPORTANT]
> 이 설정은 학습을 위해 **NVIDIA GPU**가 필요합니다. 사용 가능한 GPU가 있는지 확인해 주세요.
> 환경 설정이 완료되면 `dataset/singer_name/raw` 폴더에 singer_name의 원본 오디오 데이터를 넣어주셔야 합니다.

## 제안하는 변경 사항

### 1. 저장소 설정
- [RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 저장소를 워크스페이스에 클론합니다.
- 파이썬 가상 환경을 생성하고 필요한 의존성 라이브러리를 설치합니다.

### 2. 데이터 전처리 도구
- [NEW] `scripts/preprocess_audio.py`: 오디오 파일을 자동으로 `.wav` (40k 샘플 레이트)로 변환하고 5~10초 단위로 자르는 스크립트를 제작합니다.

### 3. 프로젝트 구조
```text
ai-voice-conversion/
├── RVC-WebUI/          # 클론된 RVC 저장소
├── dataset/
│   └── singer_name/
│       ├── raw/        # 20분 분량의 singer_name 음성 원본 (사용자 준비)
│       └── processed/  # 슬라이싱 및 변환된 학습용 데이터
├── weights/            # 최종 모델 파일 (.pth, .index)
└── scripts/            # 커스텀 유틸리티 스크립트
```

### 4. 설정 가이드
- 요청하신 파라미터들을 `training_config.md`에 문서화하여 언제든 참조할 수 있게 합니다.

## 검증 계획

### 자동화 테스트
- `preprocess_audio.py`를 샘플 파일에 실행하여 정상적으로 슬라이싱되는지 확인합니다.
- RVC WebUI를 실행하여 환경 설정이 올바른지 확인합니다.

### 수동 검증
- 5 Epoch 정도 짧게 학습을 돌려 GPU 사용량과 로그 기록이 정상인지 확인합니다.
- 전략에 따라 5초 구간 추론 테스트를 진행합니다.
