# vn-streaming-asr-bench

Vietnamese Streaming ASR Benchmark: offline + streaming simulation, reproducible evaluation.

## Goals

- Standardize datasets into a single JSONL manifest schema
- Consistent text normalization for fair WER
- Inference-only benchmark (offline + pseudo-streaming simulation)
- Metrics: WER + bootstrap CI, RTF, latency proxy
- Reproducible runs: pinned deps, seed control, config snapshot, git hash
- Reporting: export `results.csv` + figures from one command

## Quickstart

```bash
# 1) Setup
poetry install --with dev -E gpu
pre-commit install
make test

# 2) Prepare data (example: VIVOS test)
# Put wavs into data/vivos/wav/
# Create transcript file data/vivos/transcript_test.tsv with:
# vivos_test_0001<TAB>xin chào

poetry run python scripts/create_manifest.py   --wav_dir data/vivos/wav   --transcript data/vivos/transcript_test.tsv   --out data/manifests/vivos_test.jsonl

# 3) Validate
poetry run python -c "from vnasrbench.data.validate import validate_manifest; print(validate_manifest('data/manifests/vivos_test.jsonl'))"

# 4) Run benchmark
make run
# or
poetry run vnasrbench --config configs/default.yaml

# 5) Aggregate results
make aggregate
```

## Avoid Large Files In Git (HF Cache)

Do not place Hugging Face cache inside the repository. Keep cache under `$HOME`:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
```

Models and datasets are downloaded directly from Hugging Face by code (`from_pretrained` / `snapshot_download`) at runtime, so only configs/code should be committed.

## Output

- `runs/exp01/metrics.json`
- `runs/exp01/hypotheses.jsonl` (normalized refs/hyps)
- `runs/exp01/config.yaml` (raw config file)
- `runs/exp01/config_snapshot.yaml` (parsed config snapshot)
- `runs/exp01/git_hash.txt`
- `runs/exp01/run_info.json` (timestamp + git hash)
- `results.csv` (from `scripts/aggregate.py`)

## Datasets

Recommended layout (kept out of git):

- `data/<dataset>/raw/` (original downloaded/extracted data for audit)
- `data/<dataset>/wav16k/` (mono 16kHz WAV PCM for benchmarking)
- `data/<dataset>/transcripts/` (TSV: `utt_id<TAB>text`)
- `data/manifests/` (JSONL manifests consumed by the benchmark)
- `data/<dataset>/subsets/` (fixed `utt_id` lists for reproducible subsets)

Create a subset manifest from an `utt_id` list:

```bash
poetry run python scripts/subset_manifest.py \
  --manifest_in data/manifests/<dataset>_<split>_full.jsonl \
  --id_list data/<dataset>/subsets/<split>.txt \
  --out_manifest data/manifests/<dataset>_<split>.jsonl \
  --strict
```

Report manifest composition (hours + duration stats):

```bash
poetry run python scripts/report_manifest.py --manifest data/manifests/<dataset>_<split>.jsonl
```

## Hugging Face Streaming (No Full Download)

Minimal deps (for HF Audio):

```bash
pip install -U "datasets[audio]" soundfile tqdm pyyaml
```

Example HF streaming config (no local audio files):

```yaml
dataset:
  source: "hf"
  hf_id: "doof-ferb/vlsp2020_vinai_100h"
  hf_split: "train"
  text_key: "transcription"
  id_key: "id"           # optional; use if dataset provides an id column
  id_list_path: ""       # optional: path to fixed utt_id list
  streaming: true
  limit: 0               # optional: set to N for quick smoke
  sample_rate: 16000
```

Run:

```bash
poetry run vnasrbench --config configs/default.yaml
```

If you want a metadata-only manifest from HF (for composition reports):

```bash
poetry run python scripts/hf_to_manifest.py \
  --hf_id doof-ferb/vlsp2020_vinai_100h \
  --split train \
  --text_key transcription \
  --dataset_tag vlsp2020_train \
  --out data/manifests/vlsp2020_train_streaming.jsonl
```

Note: HF streaming runs do not require local `audio_path`; use `dataset.source: "hf"` in config.

## Cloud Matrix Runner

Configs for multi-dataset / multi-model runs live under `configs/cloud/` and a ready-to-run matrix is in `configs/cloud/matrix.yaml`.

Install deps (GPU + matrix models):

```bash
poetry install --with dev -E gpu -E faster -E chunkformer
```

Note:
- `chunkformer` requires Python `>=3.11`. If your server uses Python 3.10, create a Python 3.11 Poetry env before running the full matrix.

Dry-run to see planned runs:

```bash
poetry run python scripts/run_cloud_matrix.py --matrix configs/cloud/matrix.yaml --dry-run
```

Run the full matrix:

```bash
poetry run python scripts/run_cloud_matrix.py --matrix configs/cloud/matrix.yaml
```

Filter by dataset/model keyword:

```bash
poetry run python scripts/run_cloud_matrix.py --matrix configs/cloud/matrix.yaml --filter speech_massive_vie
```

Notes:
- VIVOS on HF uses a dataset script. This repo pins `datasets<4.0` to keep it working.
- The VIVOS dataset card indicates the transcript field is `sentence`.
- For HF datasets with custom loading scripts (e.g. VIVOS), set `dataset.trust_remote_code: true` in the dataset config.

## Reproducibility checklist

- Pin dependencies in `pyproject.toml`
- Commit `poetry.lock` for fully reproducible environment resolution (recommended).
- Use `configs/*.yaml` for all run parameters
- Seed control in `vnasrbench.utils.seed`
- Store run artifacts under `runs/<exp_id>`

## Evaluation Protocol (Reviewer-proof)

- HF datasets/models can be pinned with `hf_revision` in config. The resolved commit SHA is logged to `runs/<exp_id>/run_info.json`.
- If a dataset has no official test split, use a fixed held-out list via `id_list_path` and report the list hash (logged in `run_info.json`).
- Streaming results here are **simulation** (chunked decoding + LCS stitching). If a model exposes a native streaming API, report it as a separate track.
- Report strict WER as the main metric and CER (tokenization-robust) for completeness (now logged in `metrics.json`).

## Notes

Current v0.1 implements offline inference and pseudo-streaming simulation via LCS stitching. Metrics include offline WER/CI, streaming WER/CI, latency proxy, RTF, and stability metrics.

## Repo Guide

Detailed structure and file-by-file explanation:

- `docs/REPO_GUIDE.md`
- `docs/VM_H100_FRONTIER_RUNBOOK.md` (SSH VM -> setup -> run frontier 2h -> merge results)

## Docker (GPU)

Build:

```bash
docker build -t vnasrbench:0.1 -f docker/Dockerfile .
```

Run with GPU (outputs written to your local `runs/` via the mounted volume):

```bash
docker run --gpus all --rm -it -v "$PWD":/workspace vnasrbench:0.1 --config configs/default.yaml
```

## Docker (Step-by-step, dành cho người mới)

1) Cài Docker Desktop và kiểm tra:
```bash
docker --version
```

2) (Nếu dùng GPU NVIDIA trên Linux/cloud) cài NVIDIA Container Toolkit và kiểm tra:
```bash
nvidia-smi
```

3) Build image:
```bash
docker build -t vnasrbench:0.1 -f docker/Dockerfile .
```

4) Chuẩn bị config:
- Nếu chạy CPU: sửa `configs/default.yaml` thành `device: "cpu"` và `dtype: "float32"`.
- Nếu chạy GPU: giữ `device: "cuda"` và `dtype: "float16"`.

5) Chạy benchmark:
- GPU (Linux/cloud):
```bash
docker run --gpus all --rm -it -v "$PWD":/workspace vnasrbench:0.1 --config configs/default.yaml
```
- CPU (mọi máy):
```bash
docker run --rm -it -v "$PWD":/workspace vnasrbench:0.1 --config configs/default.yaml
```

6) Xem kết quả:
- `runs/exp01/metrics.json`
- `runs/exp01/hypotheses.jsonl`

Ghi chú:
- Lần chạy đầu sẽ tải model và dataset về cache của container. Nếu muốn tái dùng cache giữa các lần chạy, mount thêm một volume cache (ví dụ `-v "$PWD/.cache":/root/.cache"`).
- Trên macOS, Docker không hỗ trợ CUDA. Nếu dùng macOS, hãy chạy CPU hoặc dùng môi trường conda/poetry trực tiếp trên máy.

## Smoke Test

If you already have a manifest JSONL, validate it quickly:

```bash
poetry run python -c "from vnasrbench.data.validate import validate_manifest; print(validate_manifest('path/to/manifest.jsonl'))"
```
