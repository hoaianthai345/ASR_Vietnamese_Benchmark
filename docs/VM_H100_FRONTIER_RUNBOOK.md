# VM H100 Frontier Runbook (SSH -> Setup -> Run)

Runbook nay huong dan end-to-end de chay bo Latency-Accuracy frontier (ban 2h):
- Chay moi 2 profile: `lat1200`, `lat2400`
- Giu diem `lat4000` tu `results/15_00.csv` cu

Tạo SSH key trên máy bạn (macOS/Linux):

ssh-keygen -t ed25519 -C "your_email"

Lấy public key để dán vào ô SSH Public Key *:

cat ~/.ssh/id_ed25519.pub

## 1) Thong tin VM can co

- Public IP (vi du: `124.x.x.x`)
- User mac dinh: `ubuntu`
- SSH key private tren may local (vi du: `~/.ssh/id_ed25519`)
- Port: `22`

## 2) Cau hinh SSH tren may local

Them vao `~/.ssh/config`:

```sshconfig
Host asr-h100
  HostName <PUBLIC_IP>
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 120
```

Test ket noi:

```bash
ssh asr-h100
```

## 3) Mo bang VS Code Remote SSH

1. `Cmd/Ctrl + Shift + P`
2. Chon `Remote-SSH: Connect to Host...`
3. Chon `asr-h100`
4. Open folder: `/home/ubuntu/vn-streaming-asr-bench`

## 4) Setup server (lan dau)

### 4.1 Cai package he thong

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg tmux build-essential
```

### 4.2 Dam bao Python 3.11 (can cho chunkformer extra)

Kiem tra:

```bash
python3.11 --version
```

Neu chua co:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv
```

### 4.3 Clone repo dung vi tri

Khong clone vao `/tmp`. Dung `/home/ubuntu`:

```bash
cd /home/ubuntu
git clone <YOUR_REPO_URL> vn-streaming-asr-bench
cd vn-streaming-asr-bench
```

Neu da clone roi:

```bash
cd /home/ubuntu/vn-streaming-asr-bench
git pull
```

### 4.4 Cai Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry --version
```

### 4.5 Cai dependencies cua project

```bash
cd /home/ubuntu/vn-streaming-asr-bench
poetry env use python3.11
poetry install --with dev -E gpu -E chunkformer
```

Them bien moi truong cho HF datasets co remote code:

```bash
echo 'export HF_DATASETS_TRUST_REMOTE_CODE=1' >> ~/.bashrc
source ~/.bashrc
```

## 5) Chay Latency-Accuracy Frontier (ban 2h)

Config da dung:
- `configs/cloud/h100/matrix_frontier_300_2h.yaml`
- Chi chay `lat1200`, `lat2400`

### 5.1 Dry-run de check plan

```bash
poetry run python scripts/run_cloud_matrix.py \
  --matrix configs/cloud/h100/matrix_frontier_300_2h.yaml \
  --dry-run
```

### 5.2 Chay that (co resume + log)

```bash
mkdir -p logs
poetry run python scripts/run_cloud_matrix.py \
  --matrix configs/cloud/h100/matrix_frontier_300_2h.yaml \
  --resume --continue-on-error \
  2>&1 | tee logs/frontier_2h_$(date +%F_%H%M).log
```

## 6) Chay an toan bang tmux (khuyen nghi)

```bash
tmux new -s frontier
```

Sau do chay lenh o muc 5.2.

Detach:

```bash
Ctrl+b d
```

Attach lai:

```bash
tmux attach -t frontier
```

## 7) Theo doi tien do

Log:

```bash
tail -f logs/frontier_2h_*.log
```

Kiem tra nhanh run thanh cong/that bai:

```bash
grep -E "Running:|Done:|FAILED:" logs/frontier_2h_*.log | tail -n 100
```

GPU:

```bash
watch -n 5 nvidia-smi
```

## 8) Tong hop ket qua moi

```bash
mkdir -p results
poetry run python scripts/aggregate.py \
  --runs_dir runs \
  --out_csv results/frontier_new_raw.csv
```

## 9) Ghep voi diem lat4000 tu file cu

Muc tieu:
- Lay `lat1200`, `lat2400` tu run moi
- Lay `lat4000` tu `results/15_00.csv` cu

```bash
python3 - <<'PY'
import pandas as pd

old = pd.read_csv("results/15_00.csv")
new = pd.read_csv("results/frontier_new_raw.csv")

# chi lay run moi co profile lat1200/lat2400
new = new[new["run_id"].str.contains("__lat1200__|__lat2400__", regex=True, na=False)]

# lay diem 4000ms cu (2 model streaming tren bo 300)
old4000 = old[
    old["run_id"].str.contains(
        r"_300__chunkformer_ctc_large_vie__stream|_300__wav2vec2_base_vn_250h__stream",
        regex=True,
        na=False,
    )
]

merged = pd.concat([new, old4000], ignore_index=True)
merged.to_csv("results/frontier_merged.csv", index=False)
print("Wrote results/frontier_merged.csv:", len(merged), "rows")
PY
```

## 10) Copy ket qua ve may local

Tu may local:

```bash
scp asr-h100:/home/ubuntu/vn-streaming-asr-bench/results/frontier_merged.csv ./results/
scp asr-h100:/home/ubuntu/vn-streaming-asr-bench/logs/frontier_2h_*.log ./logs/
```

## 11) Tat VM de tranh ton chi phi

Sau khi copy xong ket qua:
- Stop VM tren dashboard cloud.

---

## Troubleshooting nhanh

- `poetry: command not found`  
  Kiem tra `PATH` da add `$HOME/.local/bin` chua, roi `source ~/.bashrc`.

- `Could not find python executable python3.11`  
  Cai Python 3.11 theo muc 4.2 roi chay lai `poetry env use python3.11`.

- `Missing dependency: chunkformer`  
  Chay lai: `poetry install --with dev -E gpu -E chunkformer`.

- Prompt `trust_remote_code` khi load dataset  
  Export: `HF_DATASETS_TRUST_REMOTE_CODE=1`.

- Warning ffmpeg/avconv  
  Cai: `sudo apt-get install -y ffmpeg`.

- Ket noi SSH hay bi rot  
  Dung `tmux` de job tiep tuc du mat ket noi.
