# Fixed Split Guide

This project supports fixed held-out evaluation splits via `dataset.id_list_path`.

Why:
- Some HF datasets only expose `train`, or lack official public test splits.
- A fixed, immutable ID list ensures reproducible and reviewer-proof evaluation.

## How to Create a Fixed Held-out Split

1) Decide the split size and selection policy (e.g., random but fixed seed, or a published official subset).
2) Write one `utt_id` per line:

```
utt_id_0001
utt_id_0002
...
```

3) Save to `data/splits/<dataset>_fixed_test_ids.txt` (example).
4) Compute and record SHA256:

```bash
shasum -a 256 data/splits/<dataset>_fixed_test_ids.txt
```

5) Add to dataset config:

```yaml
dataset:
  source: "hf"
  hf_id: "..."
  hf_split: "train"
  id_list_path: "data/splits/<dataset>_fixed_test_ids.txt"
```

The SHA256 is logged in `runs/<exp_id>/run_info.json` as `dataset_id_list_sha256`.
