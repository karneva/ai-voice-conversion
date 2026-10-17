# 실험 파이프라인 스크립트

[implementation_plan.md](../implementation_plan.md)의 실험을 재현하기 위한 스크립트 모음이다.
아래 경로·파일 형식·열 이름은 스크립트들끼리 주고받는 **공통 규약**이다. 이 규약을 바꿀 때는 이 문서도 같이 고친다.

## 실행 환경

| 스크립트 | Python |
| --- | --- |
| `scripts/data/*.py`, `scripts/run_experiment.py`, `scripts/evaluate.py`, `scripts/report.py` | `evaluation\env\python.exe` |
| Applio CLI (`run_experiment.py`가 subprocess로 호출) | `Applio\env\python.exe` (작업 폴더는 `Applio\`) |
| 보컬 분리 | `scripts\separate_vocals.bat` (`separator\env`) |

- 모든 스크립트는 레포 루트에서 실행한다.
- 경로 인자는 레포 루트 기준 상대 경로를 받는다.
- CSV와 JSON에 기록하는 경로도 레포 루트 기준 상대 경로이고, 구분자는 `/`를 쓴다.
- 오디오 읽기는 `soundfile`이나 `librosa`로 한다. `torchaudio.load`는 torchaudio 2.9부터 TorchCodec이 필요해서 쓰지 않는다.

## 데이터 레이아웃

`<track>`은 `private` 또는 `public`이다.

```text
dataset/<track>/
├── songs/                 # 원본 곡 (직접 준비)
├── separated/             # separate_vocals.bat 출력: *_(vocals)_*.wav, *_(other)_*.wav
├── vocals/                # 검수를 마친 보컬. manifest의 입력
├── manifest.csv           # build_manifest.py → split_dataset.py가 split 열을 채움
├── qc_report.csv          # quality_check.py
├── qc_segments.csv        # quality_check.py: 직접 들어봐야 할 의심 구간
├── train/                 # Applio --dataset-path. split=train 파일만 평탄화해서 복사
├── heldout/clips/*.wav    # held-out 곡에서 자른 5~10초 유성음 클립
├── heldout/clips.csv
├── source_eval/clips/*.wav  # 다른 가수들의 평가용 source 클립 (프로젝트 내내 고정)
└── source_eval/clips.csv
```

### manifest.csv

파일 하나가 한 행이다.

| 열 | 설명 |
| --- | --- |
| `file_id` | 고유 ID. 파일명으로 쓸 수 있는 문자만 쓴다 |
| `path` | 레포 루트 기준 상대 경로 |
| `song_id` | 곡 단위 분할의 기준. 같은 곡이 여러 파일로 나뉘어 있으면 값이 같다 |
| `singer` | 가수 이름 |
| `duration_s` | 길이(초) |
| `sr` | 샘플레이트 |
| `channels` | 채널 수 |
| `sha256` | 파일 해시 |
| `split` | `train`, `heldout`, 또는 빈 값(아직 분할 안 함) |

### heldout/clips.csv, source_eval/clips.csv

| 열 | 설명 |
| --- | --- |
| `clip_id` | 고유 ID. 출력 파일명으로도 쓴다 |
| `path` | 클립 경로 |
| `singer` | 가수 이름 |
| `song_id` | 원래 곡 |
| `start_s`, `end_s` | 원본 파일에서의 시작·끝 위치(초) |
| `source_path` | 원본 파일 경로 |
| `transpose` | 변환할 때 적용할 반음 수(정수). held-out은 항상 0 |
| `lyrics` | 가사. 비어 있어도 된다. 원본 옆에 GTSinger 형식의 `<stem>.json`(단어별 시각)이 있으면 `split_dataset.py`와 `make_source_set.py`가 클립 구간 안의 단어로 채운다 |

- `clip_id`는 클립 경계가 바뀌어도 그대로일 수 있다. 그래서 변환 결과와 평가 결과는 ID가 아니라 클립 오디오의 SHA-256과 transpose로 짝을 확인한다(아래 `_sources.json`).

## 실험 Run

실험 하나는 `configs/<run_id>.json` 설정 파일 하나로 정의한다. 기준은 `configs/baseline.json`이다.

```text
runs/<run_id>/
├── run.json        # 설정, 시작·종료 시각, 레포/Applio 버전, GPU 정보, 단계별 소요 시간, 초/에폭, 최대 VRAM
├── commands.log    # 실제로 실행한 명령 전문
├── logs/           # 단계별 stdout/stderr
├── applio/         # Applio logs/<run_id>/의 config.json, model_info.json 사본
├── infer/<set>/<epoch>e/<combo>/<clip_id>.wav   # (git 제외)
├── infer/<set>/<epoch>e/<combo>/_sources.json   # clip_id → 변환에 쓴 source 클립의 sha256, transpose
└── metrics/<set>/<epoch>e/<combo>.csv           # 클립별 지표
```

- `run_experiment.py infer`는 출력 파일이 있어도 `_sources.json`의 sha256·transpose가 지금 `clips.csv`의 클립과 다르면(또는 기록이 없으면) 다시 변환한다.
- `evaluate.py run`은 `_sources.json`과 다른 출력이 하나라도 있으면 채점을 거부한다. 기록이 없는 출력은 경고만 하고 채점한다.
- 클립별 지표 CSV에는 `output_sha256`, `reference_sha256`, `ref_chars`(정규화한 CER 기준 글자 수)도 들어간다. `--skip-existing`은 두 해시와 transpose가 지금 파일과 같을 때만 그 CSV를 재사용한다.

- `<set>`은 `source_eval`(교차 변환) 또는 `heldout`(재구성)이다.
- `<combo>`는 추론 설정을 나타내는 태그로, 형식은 `f0-<method>_ir<index_rate:.2f>_p<protect:.2f>_ve<volume_envelope:.2f>`이다. 예: `f0-rmvpe_ir0.30_p0.33_ve1.00`
- 체크포인트는 Applio의 `logs/<run_id>/<run_id>_<epoch>e_<step>s.pth`를 쓴다.

## results/runs.csv

(run, set, epoch, 추론 설정) 조합마다 한 행이다. 다음 7개 열이 같은 행은 덮어쓴다(upsert): `run_id, set, epoch, f0_method, index_rate, protect, volume_envelope`

| 열 | 설명 |
| --- | --- |
| `run_id`, `track`, `set`, `epoch`, `step` | 식별자 |
| `f0_method`, `index_rate`, `protect`, `volume_envelope` | 추론 설정 |
| `n_clips` | 평가한 클립 수 |
| `n_pitch_clips` | 피치 지표(`f0_rmse_cents`, `gpe50`, `octave_err`, `fpc`)가 계산된 클립 수. 기준과 출력이 함께 유성인 프레임이 2개 미만인 클립(예: 무음 출력)은 피치 평균에서 빠지므로, `n_clips`보다 작으면 피치 지표를 다른 행과 비교하면 안 된다. `report.py`는 피치 지표로 best 에폭을 고를 때 이런 행을 건너뛴다 |
| `sim_target` | 목표 가수 held-out centroid와의 ECAPA cosine 평균 |
| `sim_target_norm` | (sim_target − 하한) / (상한 − 하한). 하한과 상한은 calibration 파일에서 읽는다 |
| `sim_source` | source 클립 자체와의 cosine. 원래 목소리가 남아 있는지 보는 값 |
| `f0_rmse_cents` | 기준 F0 대비 오차. 기준은 source F0에 2^(transpose/12)를 곱한 값, heldout은 원본 F0 |
| `gpe50` | 50 cents 넘게 벗어난 유성 프레임 비율 |
| `octave_err` | 600 cents 넘게 벗어난 유성 프레임 비율 |
| `vuv_err` | 유성/무성 판정이 다른 프레임 비율 |
| `fpc` | F0 Pearson 상관 |
| `cer`, `cer_source`, `dcer` | 변환 결과와 source의 Whisper CER, 그리고 둘의 차이(cer − cer_source). 클립 평균이 아니라 전체 편집 수 ÷ 전체 기준 글자 수(corpus CER)다. `report.py`의 CER 신뢰구간도 같은 값을 클립 bootstrap으로 계산한다 |
| `cer_ref` | CER 기준. `lyrics`(clips.csv 가사), `asr`(가사가 없어 source의 Whisper 전사를 기준으로 씀. 이때 `cer_source`는 0이고 `dcer`는 source 전사와의 일치도일 뿐이다), `mixed` |
| `utmos`, `utmos_source`, `singmos`, `singmos_source` | 자연스러움 예측 점수. 선택 지표라 비어 있어도 된다 |
| `mcd` | heldout 세트에서만 계산한다 |
| `eval_version` | evaluate.py 버전 문자열 |

Calibration 결과는 `results/calibration_<track>.json`에 저장한다. 담는 값은 다음과 같다.

- target centroid를 만든 클립 목록
- `clip_sha256`: held-out과 source_eval 클립별 SHA-256. `evaluate.py run`과 `source-only`는 지금 두 세트의 클립 ID나 오디오가 이 값과 다르면 다시 calibrate하라며 멈춘다
- 상한: held-out 클립을 하나씩 빼고 만든 centroid와 그 클립의 cosine 평균(leave-one-out)
- 하한: source 클립들과 centroid의 cosine 평균

## 보컬 검수: curate_vocals.py → vocals/

`scripts/data/curate_vocals.py`는 `separated/`의 `*_(vocals)_*` 파일만 골라 `vocals/`로 옮긴다. 파일 이름은 이름을 다시 조립하지 않고 `_(vocals)_` 표시로만 찾는다. 이 표시 앞부분이 `source_stem`(원곡 파일명)이다.

```text
dataset/<track>/
├── vocals_names.csv                       # source_stem,song_id. 매번 다시 쓴다. 고친 뒤 --names로 넘긴다
├── vocals_exclude.csv                     # (직접 작성) 잘라낼 구간
└── vocals/
    ├── _curation.json                     # 입력 경로와 sha256, 적용한 구간, 파트별 샘플 경계, 시각
    └── <song_id>/<song_id>__pNN.wav       # 24-bit PCM, 원본 샘플레이트, 기본 mono
```

- 기본 `song_id`는 원곡 파일명에서 장식(`【...】`, `[MV]`, `@TVPP`, `__ 규티비`, `클린버전`, `무자막`, `무대 오디오 Only`, `Cover by. ...`, 이모지)과 가수 이름을 지우고 곡 제목만 남긴 slug다. 가수 이름은 `--artist`로 주거나, `--separated` 폴더 이름과 3곡 이상에서 반복되는 제목 조각에서 추정한다(`--no-infer-artist`로 끈다). `song_id`는 `build_manifest.py`가 다시 slugify해도 바뀌지 않는 값(60자 이하)으로 맞춘다.
- `--names`를 주지 않고 실행했는데 기존 `vocals_names.csv`에 기본값과 다른 이름이 있으면 `vocals_names.csv.bak`으로 옮겨 두고 경고한다.
- 여러 `source_stem`에 같은 `song_id`를 주면 한 곡의 파트로 합친다(같은 곡의 다른 녹음이 train과 held-out에 갈라지지 않게).
- 구간을 잘라내면 곡이 파트로 나뉜다. 이어 붙이거나 crossfade하지 않는다. `--min-part-s`(기본 2초)보다 짧은 파트는 버린다.
- 다시 실행하면 이 스크립트가 관리하는 곡(이번 실행과 지난 `_curation.json`의 곡)의 `<song_id>__pNN.wav` 중 이번에 만들지 않은 파일만 지운다. 다른 파일은 건드리지 않는다. 원본·구간이 같은 파트는 다시 쓰지 않는다.
- Windows 경로 길이 제한(260자)을 넘길 파트가 있으면 아무것도 쓰기 전에 멈춘다.

### vocals_exclude.csv (잘라낼 구간)

| 열 | 설명 |
| --- | --- |
| `song_id` 또는 `source_stem` | 대상. 한 행에 하나만 채운다. `song_id`가 여러 stem에 걸리면 `source_stem`을 쓴다. 첫 칸이 `#`이면 주석 |
| `start_s`, `end_s` | 분리 전 원곡과 같은 시간축의 초. `83.5`, `1:23.5`, `1:02:03` 모두 된다. 비우면 처음/끝 |
| `reason` | 메모(예: `duet`, `chorus`, `reverb tail`). `_curation.json`에 남는다 |

```powershell
evaluation\env\python.exe scripts/data/curate_vocals.py --separated dataset/private/separated/kyuhyun --output dataset/private/vocals --exclude dataset/private/vocals_exclude.csv --names dataset/private/vocals_names.csv
evaluation\env\python.exe scripts/data/build_manifest.py --input dataset/private/vocals --track private --singer kyuhyun --song-level 0
evaluation\env\python.exe scripts/data/split_dataset.py --manifest dataset/private/manifest.csv --train-only "*킬링보이스*"
```

`--song-level 0`이면 `vocals/` 바로 아래 폴더 이름이 `song_id`가 되어 한 곡의 파트들이 같은 split에 들어간다.

## split_dataset.py --train-only

`--train-only PATTERN`(반복 가능)은 `song_id`에 fnmatch로 맞는 곡을 항상 train에 두고 held-out 후보에서 뺀다. 20분짜리 라이브 메들리처럼 한 곡이 너무 길어서 held-out에 뽑히면 held-out이 데이터의 대부분을 가져가는 곡에 쓴다.

- held-out 목표 길이는 여전히 **전체 곡 길이**(train-only 곡 포함)의 `--heldout-fraction`이다. 이 값의 뜻이 옵션 유무와 상관없이 같고, 끝에 출력하는 train/heldout 비율도 전체 기준이며, 긴 곡 하나를 train에 고정했다고 평가용 held-out 분량이 줄지 않게 하려는 것이다. 대신 목표가 남은 후보 곡 길이의 50%를 넘으면 경고한다.
- 후보 곡이 `--min-heldout-songs`보다 많아야 한다(후보 중 적어도 한 곡은 train에 남는다).
- 기존 split을 유지하는 모드에서 train-only 곡이 이미 held-out이면 멈춘다. `--resplit`으로 다시 뽑는다. 아무 곡에도 맞지 않는 패턴은 경고한다.

## 다른 목소리 구간 찾기: speaker_outliers.py

`scripts/data/speaker_outliers.py`는 보컬 파일에서 목표 가수가 아닐 가능성이 큰 구간(듀엣 상대, 게스트, 코러스, MC 멘트)을 찾아 사람이 그 구간만 들어 보게 한다. `curate_vocals.py` 전에 `separated/`에 돌리고, 확인한 구간을 `vocals_exclude.csv`에 옮겨 적는다(시간축이 같다. `source_stem`은 파일명에서 `_(vocals)_` 앞부분).

- 3초 창, 1초 hop, 16 kHz mono. pyworld 유성 프레임이 `--min-voiced`(기본 0.5) 미만인 창(무음, 반주 잔향)은 건너뛴다. 말소리는 유성 비율이 0.35~0.5인 창이 많아서 멘트가 많은 파일은 `--min-voiced 0.35`가 더 잘 잡는다.
- 각 창의 ECAPA 임베딩(`eval_lib/speaker.py`, 모델은 `results/cache/models/`)과 centroid의 cosine을 구한다. centroid는 `--reference`(순수한 목표 가수 보컬)가 있으면 그 창들의 평균, 없으면 모든 입력 창의 평균에서 cosine 하위 25%를 빼고 다시 구하기를 3번 반복한 값이다.
- 점수는 이웃한 유성 창 `--smooth`개(기본 3)의 cosine 평균이다. 임계값은 모든 입력 창 점수의 `median − k × MAD`(MAD는 1.4826을 곱한 값, `--k` 기본 3.5)이고, `--floor`를 주면 그 값보다 낮은 창도 표시한다. 연속 2개(`--min-consecutive`) 이상 낮은 창만 구간이 되고, 무성 창 1개(`--bridge`)는 연속을 끊지 않는다. 창 하나는 가운데 1초를 대표하므로 구간은 `[첫 창 중심 − 0.5초, 마지막 창 중심 + 0.5초]`이고, 2초(`--merge-gap-s`) 이하로 떨어진 구간은 합친다.
- 같은 사람의 말소리(MC 멘트)는 노래와 cosine이 0.35~0.45 정도라 경계에 걸려 일부만 잡힐 수 있다.
- 라이브 무대는 녹음 환경 때문에 파일 전체의 cosine이 홈 커버보다 0.05~0.15 낮다. 그래서 임계값은 파일별이 아니라 전체 창 기준이다(파일별로 중앙값을 빼면 듀엣처럼 파일 절반이 다른 목소리인 경우를 놓친다). 대신 출력 끝의 "Whole-file check"는 창 30개 이상인 파일 중 중앙값이 파일 중앙값들의 `median − 2 × MAD`보다, 그리고 0.1 넘게 낮은 파일을 따로 알려 준다.
- 듣는 목록을 넉넉하게 받고 싶으면 `--k 3.0 --min-voiced 0.35`를 쓴다(합성 테스트에서 오탐이 약 2배).

```powershell
evaluation\env\python.exe scripts/data/speaker_outliers.py --input dataset/private/separated/kyuhyun --output dataset/private/speaker_outliers.csv --k 3.0 --min-voiced 0.35 --plot
```

| 파일 | 열 |
| --- | --- |
| `--output` (예: `dataset/<track>/speaker_outliers.csv`) | `path, start_s, end_s, min_cos, mean_cos, n_windows`. 한 행이 한 구간이고 `mean_cos` 오름차순(가장 의심스러운 것부터). `min_cos`, `mean_cos`는 구간 안 창들의 (smoothing 전) cosine |
| `--summary` (기본 `<output stem>_summary.csv`) | `path, duration_min, voiced_min, n_windows, median_cos, p10_cos, n_segments, flagged_s, flagged_pct`. 파일당 한 행, `flagged_s` 내림차순. `voiced_min`은 점수를 매긴 창 수 × hop, `flagged_pct`는 그 대비 비율 |
| `--windows` (선택) | `path, start_s, end_s, voiced_frac, cos, score, flagged`. 창마다 한 행. 건너뛴 창은 `cos`가 비어 있다 |
| `--plot` | `<output stem>_plots/<slug>_<hash>.png`: 파일별 cosine·점수 그래프와 표시 구간 |

## 감상용 커버 한 번에 만들기: make_cover.py

`scripts/make_cover.py`는 곡 파일 하나에서 완성된 커버까지 한 번에 만든다. 결과물은 개인 감상용이고 `dataset/` 아래에 두므로 git에 들어가지 않는다.

1. **보컬 분리:** `dataset/*/separated/` 아래(하위 폴더 포함)에서 같은 곡의 `*_(vocals)_*.wav`와 짝인 `*_(other)_*.wav`를 찾아 재사용한다. 파일명은 Windows나 audio-separator가 바꾸므로(`__`→`_`, 끝의 `.` 삭제) 글자·숫자만 남긴 이름으로 비교하고, 길이가 원곡과 0.5초 넘게 다르면 버린다. 없으면 `separate_vocals.py`로 `dataset/<track>/separated/_cache/`에 분리한다. ffmpeg가 PATH에 없으면 WinGet 설치 위치에서 찾는다.
2. **transpose 추정** (`--reference`): 목표 가수가 같은 곡을 부른 음원(곡 파일 또는 분리된 보컬)과 원곡 보컬을 비교한다. 유성 프레임 F0 중앙값의 비(torchcrepe, `--pitch-tracker harvest`면 pyworld)로 반음 수를, 활성 프레임 평균 CQT chroma의 원형 상호상관으로 조 차이(mod 12)를 구한다. 음이름은 chroma로, 옥타브는 F0 비로 정한다. F0 pitch-class 히스토그램 상관도 교차 확인용으로 출력한다. `--estimate-only`면 여기서 멈춘다.
3. **변환:** Applio `core.py infer`에 모든 값을 명시해서 넘긴다. embedder는 weights 옆 `model_info.json`에서 읽고, `--embedder`가 그와 다르면 멈춘다. 경로에 `trained`가 있으면 멈춘다. 입력 보컬은 작업 폴더에 ASCII 이름으로 복사해서 넘긴다. 곡 전체를 한 번에 변환하면 4분 곡에서 VRAM이 GPU 전체 사용량 기준 최대 약 5.9 GB까지 올라갔다. 메모리가 부족하면 `--split-audio`(Applio가 -60 dB 무음에서 자른 조각을 따로 변환, 최대 약 4.5 GB)를 쓴다.
4. **반주 피치 시프트:** `--instrumental-shift auto`는 transpose를 [-6, 5]로 감은 값이다(옥타브 이동은 반주를 옮기지 않는다). ffmpeg `rubberband` 필터(tempo=1, pitchq=quality, channels=together)를 쓰고, 없으면 `librosa.effects.pitch_shift`를 쓴다.
5. **믹스:** `mix_cover.py` 함수를 import해서 쓴다. 변환 보컬을 원곡 보컬 stem의 LUFS에 맞춘 뒤 `--vocal-gain-db`를 더하고, 길이를 반주(= 원곡 길이)에 맞추고, -14 LUFS로 맞춘 뒤 -1 dBFS 샘플 피크로 제한한다. 출력 길이는 시프트 전 반주 길이(= 원곡)로 자르거나 채운다(rubberband는 끝을 약 80 ms 잘라 먹는다). MP3/OGG는 디코딩하면 피크가 약 0.1 dB 넘치므로 limiter를 0.3 dB 더 낮게 건다(`settings.limiter_ceiling_db`).

감상용 기본값: `--f0-method rmvpe --index-rate 0.75 --protect 0.33 --volume-envelope 1.0`. index-rate는 실험 Baseline(0.3)보다 높여 음색을 목표 가수 쪽으로 당긴다(발음이 뭉개지면 낮춘다). protect 0.5는 보호 꺼짐이다.

```powershell
evaluation\env\python.exe scripts/make_cover.py --song "dataset/private/cover_input/<곡>.m4a" --run E1-base --epoch 200 --reference "dataset/private/reference/<목표 가수 버전>.m4a"
```

- 기본 출력: `dataset/private/covers/<song_id>__<run>_<E>e_t<+N>.mp3` (`song_id`는 `slugify(곡 파일 stem)`). `--keep-intermediates`면 `<출력 stem>_work/`(Applio 입력·출력, `applio_infer.log`, 시프트한 반주)를 남긴다. 실패하면 작업 폴더는 항상 남는다.
- 옆에 같은 이름의 `.json`을 쓴다: `settings`(모델 태그, embedder, 추론 값, transpose와 출처, 반주 시프트와 방법, 믹스 값), `inputs`(곡·보컬·반주·reference 보컬·pth·index의 경로와 sha256), `separation`(`reused`/`separated`), `transpose_estimate`(두 트랙의 F0 중앙값·p10·p90·유성 초, `f0_median_semitones`, `chroma_offset`, `chroma_scores`[k=0..11], `chroma_margin`, `f0_pitch_class_offset`, `transpose`), `applio_command`, `mix`(`vocal_level_match_db`, `vocal_lag_ms`, `instrumental_lag_ms`: onset envelope 상호상관으로 잰 원본 대비 지연, 20 ms 넘으면 경고), `result`(출력을 다시 디코딩해서 잰 길이·LUFS·피크와 원곡 길이 차), `timings`.

## 보컬 분리 임시 파일과 학습 금지 폴더

- `separate_vocals.py`는 m4a·mp3 같은 압축 파일을 출력 폴더 안의 숨김 임시 폴더(`<output>/.decode_*`)에 WAV로 풀고, 끝나면 지운다. 개인 음원이 레포 밖(`%TEMP%`)으로 복사되지 않게 하려는 것이다. 숨김 폴더라 `build_manifest.py`·`speaker_outliers.py`는 이 폴더를 읽지 않는다.
- `curate_vocals.py`는 `--separated` 경로에 `reference` 또는 `cover_input` 폴더가 들어 있으면 멈춘다. 정답 커버와 변환할 원곡은 학습 데이터가 되면 안 된다.
