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

## Output

- `runs/exp01/metrics.json`
- `runs/exp01/hypotheses.jsonl` (normalized refs/hyps)
- `runs/exp01/config.yaml` (snapshot of the run config)
- `runs/exp01/run_info.json` (timestamp + git hash)
- `results.csv` (from `scripts/aggregate.py`)

## Reproducibility checklist

- Pin dependencies in `pyproject.toml`
- Use `configs/*.yaml` for all run parameters
- Seed control in `vnasrbench.utils.seed`
- Store run artifacts under `runs/<exp_id>`

## Notes

Current v0.1 implements offline inference and streaming latency simulation only. Pseudo-streaming stitching and detailed latency metrics are planned in phase 2.
