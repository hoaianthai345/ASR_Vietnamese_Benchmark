# Hướng dẫn Repo

Tài liệu này giải thích cấu trúc repo hiện tại, cách pipeline hoạt động và vai trò của từng file.

## Tổng quan

Repo này benchmark ASR tiếng Việt với:
- Mô phỏng offline và streaming (chia chunk + ghép LCS).
- Chỉ số: WER + bootstrap CI, RTF, độ trễ proxy, chỉ số ổn định.
- Hai chế độ nạp dữ liệu:
  - Manifest JSONL cục bộ (`audio_path` trỏ tới WAV local).
  - Hugging Face streaming (không tải toàn bộ).

## Quickstart (Local Smoke)

```bash
poetry install --with dev -E gpu
poetry run vnasrbench --config configs/default.yaml
```

## Output Artifactsc

Mỗi lần chạy ghi vào `runs/<exp_id>/`:
- `config.yaml`: file config gốc được dùng.
- `config_snapshot.yaml`: snapshot đã parse (config đã resolve).
- `git_hash.txt`: `git rev-parse HEAD` hoặc `unknown`.
- `run_info.json`: timestamp + git hash + đường dẫn config.
- `metrics.json`: chỉ số offline + streaming, WER, CI, RTF, độ trễ proxy.
- `hypotheses.jsonl`: ref/hyp + bản normalize (+ partials streaming nếu bật).

## Bố cục Repo

### Root
- `pyproject.toml`: phụ thuộc được ghim + CLI entrypoint.
- `Makefile`: lệnh chuẩn (`lint`, `test`, `run`, `aggregate`).
- `ruff.toml`, `mypy.ini`, `pytest.ini`: cấu hình tooling.
- `.pre-commit-config.yaml`: hooks ruff + formatting.
- `.gitignore`, `.dockerignore`: bỏ qua data/runs/cache.
- `LICENSE`: MIT license.

### `docker/`
- `docker/Dockerfile`: runtime GPU (CUDA 12.4.1), cài Poetry, deps, và chạy `vnasrbench`.
- `docker/README.md`: lệnh build/run đầy đủ với GPU + mount cache.

### `configs/`
- `configs/default.yaml`: config baseline.
- `configs/datasets/`: template config theo dataset.
  - `vivos.yaml`: ví dụ config cho dataset VIVOS.
- `configs/models/`: config mô hình.
  - `whisper_small.yaml`: ví dụ config HF Whisper.

### `scripts/`
Tiện ích dữ liệu và orchestration:
- `create_manifest.py`: tạo manifest local từ thư mục wav + file transcript.
- `subset_manifest.py`: tạo manifest con từ danh sách `utt_id`.
- `report_manifest.py`: báo cáo số giờ + thống kê duration cho manifest.
- `hf_to_manifest.py`: tạo manifest metadata-only từ HF streaming.
- `aggregate.py`: tổng hợp `runs/*/metrics.json` vào `results.csv`.
- `run_benchmark.py`: wrapper mỏng gọi CLI.

### `src/vnasrbench/`
Thư viện lõi.

#### Entry point
- `cli.py`: tải YAML config, snapshot run artifacts, build model, chạy benchmark.

#### Data
- `data/manifest.py`: schema JSONL (`utt_id`, `audio_path`, `text`, `duration_sec`).
- `data/validate.py`: kiểm tra thiếu audio, text rỗng, và sample rate.
- `data/hf.py`: iterator HF streaming (`datasets.load_dataset(..., streaming=True)`).

#### Models
- `models/base.py`: `ASRModel` protocol + `DecodeResult`.
- `models/hf_whisper.py`: adapter HF Transformers Whisper (`AutoProcessor` + `AutoModelForSpeechSeq2Seq`).

#### Streaming
- `streaming/chunker.py`: logic chunk/overlap/lookahead.
- `streaming/stitch.py`: ghép dựa trên LCS + chỉ số ổn định.
- `streaming/protocol.py`: helper cấu hình streaming.

#### Evaluation
- `eval/wer.py`: WER với normalization + tương thích jiwer.
- `eval/runner.py`: đánh giá offline + streaming, metrics, ghi hypotheses.

#### Utilities
- `utils/textnorm.py`: chuẩn hóa tiếng Việt (strict WER).
- `utils/bootstrap.py`: paired bootstrap CI.
- `utils/seed.py`: kiểm soát seed determinism.
- `utils/io.py`: helpers JSON/text.
- `utils/logging.py`: thiết lập Rich logger.

### `tests/`
- `test_textnorm.py`: normalization.
- `test_wer.py`: độ đúng WER.
- `test_stitch.py`: ghép LCS + chỉ số ổn định.

## Ghi chú Pipeline chính

1) **Đánh giá offline**: giải mã toàn câu + WER.
2) **Mô phỏng streaming** (nếu bật):
   - Chia audio: `chunk_ms` + `overlap_ms` + `lookahead_ms`.
   - Decode từng chunk.
   - Ghép transcript bằng LCS (WER chỉ trên bản final).
   - Tính chỉ số ổn định từ partials (phụ lục).
3) **Chỉ số**:
   - `wer` và `wer_ci_*` (bootstrap).
   - `avg_rtf`: thời gian tính / thời gian audio.
   - `latency_proxy_ms`: chunk + lookahead.
   - `delta_wer`: streaming - offline.

## Quy trình khuyến nghị

1) Chạy smoke local (manifest nhỏ).
2) Validate dữ liệu bằng `validate.py`.
3) Chạy benchmark đầy đủ trên GPU/cloud.
4) Tổng hợp kết quả → `results.csv`.
5) Vẽ biểu đồ từ kết quả đã tổng hợp.

---

Nếu bạn cần walkthrough theo từng file hoặc muốn mở rộng guide với ví dụ cho từng dataset, hãy cho tôi biết phần nào cần chi tiết thêm.
