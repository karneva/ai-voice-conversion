# 실험 설정 (configs/)

실험 하나는 이 폴더의 JSON 파일 하나로 정의한다. 실행은 `scripts/run_experiment.py`가 하고, 기록은 `runs/<run_id>/`에 남는다. 경로와 파일 형식 규약은 [scripts/README.md](../scripts/README.md)를 따른다.

| 파일 | 내용 |
| --- | --- |
| `baseline.json` | Baseline-0 ([implementation_plan.md §6](../implementation_plan.md#6-baseline-0), [training_config.md](../training_config.md)). Run ID는 `E1-base`. 500 에폭까지 학습하고 10 에폭마다 저장된 체크포인트를 전부 평가한다 |
| `E0-smoke-public.json` | Public 트랙 스모크 테스트. 5 에폭 학습하고 매 에폭 저장한다. 나머지는 Baseline과 같다 |

## 실행

레포 루트에서 평가 환경 Python으로 실행한다.

```bat
evaluation\env\python.exe scripts\run_experiment.py configs\E0-smoke-public.json all --dry-run
evaluation\env\python.exe scripts\run_experiment.py configs\E0-smoke-public.json prepare train
evaluation\env\python.exe scripts\run_experiment.py configs\baseline.json infer --checkpoints 100,200 --sets heldout
```

- 단계: `prepare`(preprocess + extract), `train`(학습 + index), `infer`(평가 세트 일괄 변환), `all`(세 단계 전부).
- `--dry-run`: 실행할 명령만 출력한다. 파일은 하나도 만들지 않는다.
- `--checkpoints`: `all`, `last`, 또는 `10,20`처럼 에폭 목록. 설정의 `infer.checkpoints`를 덮어쓴다.
- `--sets`: `source_eval,heldout` 중 일부. 설정의 `infer.sets`를 덮어쓴다.
- `--force`: 아래 "같은 Run ID 재사용 규칙"에 걸려도 그대로 진행한다. 기록에는 `forced: true`가 남는다.

## 새 실험 설정 만들기

**Baseline에서 한 가지만 바꾼다.** 파일을 통째로 복사하지 말고 `base`로 상속받은 뒤 바꾸는 키만 적는다. 상속은 딕셔너리 단위로 깊게 합쳐지고(deep merge), 리스트(`infer.grid`, `infer.sets`)는 통째로 바뀐다.

1. `configs/<run_id>.json`을 새로 만든다. Run ID 형식은 `E<번호>-<변경점>`이다.
2. `"base": "baseline.json"`과 새 `run_id`를 적는다. `base` 경로는 이 설정 파일이 있는 폴더 기준이다.
3. 바꾸는 키 하나만 적는다. `description`에 무엇을 왜 바꿨는지 한 줄로 쓴다.

예: E7 임베더 비교 (재학습 필요)

```json
{
  "base": "baseline.json",
  "run_id": "E7-korean-hubert",
  "description": "E7: embedder contentvec -> korean-hubert-base",
  "extract": {"embedder_model": "korean-hubert-base"}
}
```

예: E2 index-rate × protect (재학습 없음). `model_run_id`로 이미 학습한 모델을 가리키고, 추론 설정만 바꾼다. 이 경우 `prepare`와 `train`은 거부되고 `infer`만 돈다.

```json
{
  "base": "baseline.json",
  "run_id": "E2-ir-protect",
  "model_run_id": "E1-base",
  "description": "E2: index_rate x protect grid on the best E1 checkpoint",
  "infer": {
    "checkpoints": [200],
    "grid": [
      {"f0_method": "rmvpe", "index_rate": 0.0, "protect": 0.2, "volume_envelope": 1.0},
      {"f0_method": "rmvpe", "index_rate": 0.0, "protect": 0.33, "volume_envelope": 1.0}
    ]
  }
}
```

해석된(상속이 적용된) 최종 설정은 `runs/<run_id>/run.json`의 `config`에 저장된다.

## 설정 키

| 키 | 설명 |
| --- | --- |
| `run_id` | 실험 ID. `runs/<run_id>/`와 Applio `--model-name`에 쓴다. `trained`라는 글자는 넣을 수 없다(아래 구현 메모) |
| `model_run_id` | (선택) Applio 모델 이름. 기본값은 `run_id`. 다른 run의 모델로 추론만 할 때 쓴다. `trained`를 넣을 수 없다 |
| `track` | `private` 또는 `public`. 평가 클립을 `dataset/<track>/<set>/clips.csv`에서 읽는다 |
| `dataset_dir` | 학습 데이터 폴더(레포 루트 기준). split=train 보컬만 평탄화해서 넣는다. 하위 폴더는 Applio가 화자 ID로 해석하므로 두지 않는다 |
| `eval_dir` | (선택, 테스트용) 평가 클립 폴더. 기본값은 `dataset/<track>` |
| `description` | 무엇을 바꿨는지 |
| `preprocess` | `sample_rate`(32000/40000/48000. extract와 train도 이 값을 쓴다), `cpu_cores`, `cut_preprocess`, `process_effects`, `noise_reduction`, `noise_reduction_strength`, `chunk_len`, `overlap_len`, `normalization_mode` |
| `extract` | `f0_method`(crepe/rmvpe/fcpe), `embedder_model`, `embedder_model_custom`, `gpu`, `cpu_cores`, `include_mutes` |
| `train` | `vocoder`, `pretrained`(`"default"`, `"none"`, 또는 `{"g": 경로, "d": 경로}`), `batch_size`, `save_every_epoch`, `total_epoch`, `gpu`, `index_algorithm`, `checkpointing`, `cache_data_in_gpu`. `total_epoch`는 `save_every_epoch`의 배수로 둔다. 아니면 마지막 에폭에 G_/D_가 없어서 나중에 학습을 늘릴 때 더 이른 에폭부터 이어진다(경고를 낸다) |
| `infer.grid` | 추론 설정 목록. 항목마다 `f0_method`, `index_rate`, `protect`, `volume_envelope`. hybrid f0는 쓸 수 없다 |
| `infer.checkpoints` | `"all"`, `"last"`, 또는 에폭 리스트 |
| `infer.sets` | `source_eval`, `heldout` 중 일부 |

러너가 항상 고정하는 값(설정으로 바꿀 수 없음):

- 에폭별 가중치 저장은 켜고, `save_only_latest`와 `cleanup`은 끈다.
- 추론할 때 autotune, clean-audio, split-audio, formant, post-process는 끈다. 출력은 WAV, sid는 0이다.
- 추론의 `--embedder-model`은 `extract.embedder_model`을 그대로 넘긴다. 이 값이 모델의 `Applio/logs/<model>/model_info.json`에 기록된 embedder와 다르면 `infer`를 거부한다. `model_run_id`로 다른 모델을 쓸 때는 그 모델의 `extract` 설정을 맞춘다.
- held-out 세트는 `clips.csv`의 transpose 값과 상관없이 항상 pitch 0으로 변환한다.

## 같은 Run ID 재사용 규칙

Applio는 extract에서 이미 있는 특징 파일을 건너뛰고, `config.json`과 `.index`를 덮어쓰지 않는다. 같은 Run ID로 설정을 바꿔 다시 돌리면 예전 설정과 새 설정이 섞인다. 그래서 러너는 다음 경우에 실행을 거부한다.

- `runs/<run_id>/run.json`에 기록된 설정과 `track`, `dataset_dir`, `preprocess`, `extract`, `train`(`total_epoch` 제외) 중 하나라도 다를 때. 이때는 새 Run ID를 쓴다.
- `prepare`를 다시 돌리려는데 다음 중 하나에 해당할 때:
  - `Applio/logs/<run_id>`에 이미 학습된 체크포인트가 있다.
  - 그 폴더의 embedder나 샘플레이트가 설정과 다르다.
  - 데이터 폴더 내용(SHA-256)이 지난 prepare 때와 다르다. 데이터 지문은 Applio를 돌리기 전에 기록하므로 실패한 prepare 뒤에도 비교된다.
  - 지난 prepare 기록에 데이터 지문이 없다(이전 버전 러너에서 실패한 경우).
  - 러너 기록 없이 폴더만 있다.
- `train`을 이어서 돌리는데, 마지막 G_/D_ 체크포인트보다 뒤 에폭의 가중치 파일이 있을 때. Applio는 그 G_/D_에서 다시 학습하고 이미 있는 가중치 파일은 덮어쓰지 않으므로, 그 파일은 버려진 학습 경로의 것으로 남는다. 옮기거나 지운 뒤 다시 돌린다.

거부되지 않는 경우는 다음과 같다.

- 설정이 같은 상태로 `prepare`를 다시 돌리는 경우. 실패한 prepare를 이어서 돌릴 때가 여기에 해당한다.
- 지난 prepare가 성공했고 설정, 데이터 지문, embedder, 샘플레이트가 모두 같으면 `prepare`는 건너뛴다(학습된 체크포인트가 있어도). 그래서 중단된 학습도 `all`로 이어서 돌릴 수 있다. 다시 돌리려면 `--force`.
- `total_epoch`를 늘려서 `train`을 다시 돌리는 경우. Applio가 G_/D_ 체크포인트에서 이어서 학습한다.
- `infer` 설정을 바꾸는 경우.

`infer`는 이미 만들어진 출력 파일을 건너뛴다. 단, 같은 폴더의 `_sources.json`에 기록된 source 클립 SHA-256과 transpose가 지금 `clips.csv`와 같을 때만이다. 클립을 다시 잘랐거나 transpose가 바뀐 출력은 다시 변환한다. `--dry-run`도 같은 기준으로 건너뛸 개수와 다시 변환할 개수를 보여 준다. 전부 다시 만들려면 해당 `runs/<run_id>/infer/...` 폴더를 지운다.

`train`을 다시 돌렸는데 마지막 가중치가 이미 있으면 학습은 건너뛰고(`training_skipped: true`) index만 확인한다. 이때 지난 학습에서 잰 `sec_per_epoch`, `steps_per_epoch`, `pretrained`, `vram_before_mib`는 그대로 남는다.

## 기록되는 것 (`runs/<run_id>/`)

- `run.json`:
  - `config`: 해석된 최종 설정
  - `environment`: 레포 커밋과 dirty 여부, Applio 버전과 precision, GPU 이름과 드라이버, 두 환경의 Python 버전, Applio torch 버전
  - `steps.<단계>`: 시작·종료 시각, 소요 시간, 상태, 산출물 검증 결과
  - `invocations`: 실행할 때마다 한 줄씩 쌓이는 이력
- `steps.train`:
  - `epochs`: 에폭별 step, Applio가 찍은 소요 시간, 실제 경과 시간
  - `sec_per_epoch`: 첫 에폭을 뺀 실제 경과 시간의 중앙값
  - `steps_per_epoch`
  - `peak_vram_mib`: `nvidia-smi`로 5초마다 잰 GPU 전체 사용량의 최댓값. 다른 프로세스가 쓴 양도 포함된다. 학습 시작 직전 값은 `vram_before_mib`에 있다.
  - `pretrained`: pretrain 경로와 SHA-256
  - `checkpoints`
  - `disk_mib`
- `commands.log`: 실제로 실행한 명령 전문과, 같은 동작을 하는 `core.py` 명령
- `logs/<단계>.log`: 각 단계의 stdout/stderr. tqdm 진행 표시줄은 마지막 상태만 남긴다.
- `applio/`: `Applio/logs/<model>/config.json`과 `model_info.json`의 사본
- `infer/<set>/<epoch>e/<combo>/<clip_id>.wav`: 변환 결과 (git 제외)

## 구현 메모

- **Applio 스크립트를 직접 호출한다.** `core.py`는 preprocess, extract, train, index를 `CREATE_NO_WINDOW` 서브프로세스로 띄운다. 그런데 `core.py`의 출력을 파이프나 파일로 받으면 그 하위 프로세스의 출력이 사라진다(Windows에서 확인함). 그러면 학습 로그를 저장하거나 초/에폭을 파싱할 수 없다.
  - 그래서 러너는 `core.py`가 만드는 것과 같은 위치 인자로 `rvc/train/...` 스크립트를 직접 실행한다. 작업 폴더는 `Applio\`이고, 환경 변수도 `core.py`와 같게 맞춘다.
  - 이렇게 하면 실패했을 때 종료 코드도 그대로 받는다. `core.py train`은 실패해도 0을 반환할 수 있다.
  - 추론은 `core.py batch-infer`가 같은 프로세스 안에서 처리하므로 그대로 호출한다.
- **실패는 세 가지로 판정한다.** 종료 코드, 출력의 실패 문구(`Traceback`, `out of memory` 등), 예상 산출물이 있는지 여부다.
  - prepare: 슬라이스, f0, 특징 파일 개수가 서로 같은지, `config.json`과 `filelist.txt`가 있는지 확인한다.
  - train: 마지막 에폭의 가중치 파일과 `.index` 파일이 있는지 확인한다.
  - infer: 모든 클립의 출력 파일이 있는지 확인한다.
- **추론은 transpose별로 묶어서 돌린다.** batch-infer는 호출 한 번에 `--pitch`를 하나만 받는다. 그래서 체크포인트 × 추론 설정 × transpose 조합마다 `runs/<run_id>/infer/_tmp/`에 입력 폴더를 만들고 한 번씩 호출한다. 결과 파일은 규약 경로로 옮긴다.
  - 호출할 때마다 모델을 새로 읽느라 20초 안팎이 더 걸린다.
- **Run ID와 경로에 `trained`를 쓰지 않는다.** Applio `rvc/infer/infer.py`는 index 경로 전체에 `.replace("trained", "added")`를 적용한다. 그러면 없는 경로가 되어 index 없이 조용히 변환된다. 러너는 이런 `run_id`/`model_run_id`와 index 경로를 거부한다.
- **`APPLIO_LOGS_DIR`를 `Applio\logs`로 고정한다.** Applio 사용자 설정의 `logs_dir`가 바뀌어 있어도 산출물 위치가 달라지지 않는다.
