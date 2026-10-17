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
