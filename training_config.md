# Baseline-0 실행 레퍼런스 (Applio 3.6.6 CLI)

실험 설계와 이유는 [implementation_plan.md](implementation_plan.md)에 있다. 이 문서는 **그대로 복사해서 실행할 명령**만 모은다.

Applio의 CLI 기본값과 UI 기본값이 다르기 때문에, 아래 명령은 모든 값을 명시한다. 실험할 때는 `<RUN>`을 Run ID로 바꾸고, Baseline-0에서 값 하나만 바꾼다.

## 공통

- 모든 명령은 Applio 폴더에서 Applio 전용 Python으로 실행한다.

  ```bat
  cd C:\DEV\ai-voice-conversion\Applio
  ```

- 학습 산출물은 `Applio\logs\<RUN>\`에 생긴다.
- 학습 데이터 폴더(`--dataset-path`)에는 분리·검수가 끝난 **train 분할의 보컬 wav만** 넣는다. held-out은 절대 넣지 않는다.

## 0. 보컬 분리 (레포 루트에서)

```bat
scripts\separate_vocals.bat dataset\private\songs dataset\private\separated
```

## 1. Preprocess

```bat
env\python.exe core.py preprocess --model-name <RUN> --dataset-path "C:\DEV\ai-voice-conversion\dataset\private\train" ^
  --sample-rate 40000 --cpu-cores 8 --cut-preprocess Automatic --normalization-mode post
```

- `--process-effects`(48Hz 고역 통과 필터)와 `--noise-reduction`은 Baseline에서 쓰지 않는다.
  - `--process-effects`는 도움말에 "Disable filters"라고 적혀 있지만, 실제로는 넘기면 필터가 **켜진다**.
- `--cpu-cores`를 생략하면 1코어로 돈다.

## 2. Extract

```bat
env\python.exe core.py extract --model-name <RUN> --sample-rate 40000 --f0-method rmvpe ^
  --embedder-model contentvec --gpu 0 --cpu-cores 8 --include-mutes 2
```

- `--gpu`를 생략하면 CPU(`-`)로 돈다.
- 이미 있는 특징 파일은 건너뛴다. f0 방식, embedder, 샘플레이트를 바꾸는 실험은 **새 Run ID**로 돌린다.

## 3. Train (끝나면 index가 자동으로 생성됨)

```bat
env\python.exe core.py train --model-name <RUN> --vocoder HiFi-GAN --sample-rate 40000 ^
  --total-epoch 500 --save-every-epoch 10 --batch-size 8 --gpu 0 --index-algorithm Auto
```

- `--save-every-weights`는 **넘기지 않는다**. 넘기면 에폭별 가중치 저장이 꺼진다.
- `--save-only-latest`도 넘기지 않는다. 체크포인트를 남겨야 E1 곡선을 그릴 수 있다. 대신 디스크 사용량은 확인한다.
- VRAM이 부족(OOM)하면 `--checkpointing`을 넘기거나 `--batch-size 4`로 낮춘다. 이 경우 Baseline 값 자체를 바꾸고 기록한다.
- 같은 Run ID로 다시 실행하면 이어서 학습한다. 처음부터 다시 하려면 새 Run ID를 쓴다.
- 이 명령은 실패해도 종료 코드가 0일 수 있다. 출력 텍스트를 확인한다.
- **스모크 테스트:** `--total-epoch 5`로 먼저 돌려서 초/에폭, 최대 VRAM(`nvidia-smi`), 디스크 사용량을 `runs/<RUN>/`에 기록한다.

## 4. 추론 (평가용 일괄 변환)

```bat
env\python.exe core.py batch-infer --input-folder <source_dir> --output-folder <out_dir> ^
  --pth-path logs\<RUN>\<RUN>_<epoch>e_<step>s.pth --index-path logs\<RUN>\<RUN>.index ^
  --f0-method rmvpe --pitch 0 --index-rate 0.3 --protect 0.33 --volume-envelope 1.0 ^
  --embedder-model contentvec --export-format WAV
```

- `--embedder-model`은 학습 때와 같은 값을 직접 지정한다. `.pth`에 기록된 값을 읽지 않는다.
- `--f0-method hybrid[...]`는 구현되어 있지 않아서 실행하면 오류가 난다.
- `--f0-autotune`, `--clean-audio`, `--post-process`는 평가할 때 쓰지 않는다. 오류를 가려서 지표를 왜곡하기 때문이다.
- 출력 파일 이름은 `<원본이름>_output.wav`이다. 출력 폴더에 이미 있는 파일은 건너뛰므로, 체크포인트마다 출력 폴더를 분리한다.

## 5. 모니터링

```bat
env\python.exe core.py tensorboard
```

- 곡선은 `loss_avg_50/g/total`, `loss_avg_50/g/mel`, `loss_avg_50/d/adv`를 본다.
- epoch 단위 태그는 마지막 배치 값이라서 노이즈가 크다.
- loss가 낮다고 더 좋은 모델은 아니다. 체크포인트 선택은 held-out 지표로 한다(E1).

## 고정 하이퍼파라미터 (참고)

`logs/<RUN>/config.json`에 복사되는 값이다. 바꾸려면 이 사본을 수정하고, 수정 내용을 Run 기록에 남긴다.

| 항목 | 값 |
| --- | --- |
| optimizer | AdamW, lr 1e-4, betas [0.8, 0.99], eps 1e-9 |
| lr schedule | ExponentialLR, 에폭마다 ×0.999875 |
| loss 가중치 | c_mel 45, c_kl 1.0 |
| seed | 1234 |
| segment_size | 12800 (40k) |
| precision | fp16. CLI 플래그가 아니라 `%APPDATA%\Applio\config.json`에서 정해진다 |

## 권리와 공개 범위

공개 범위는 [implementation_plan.md §1](implementation_plan.md#1-트랙-분리-개인용과-공개용)을 따른다. Private 트랙의 가중치(`.pth`, `.index`)와 변환 음원은 커밋하거나 업로드하지 않는다.
