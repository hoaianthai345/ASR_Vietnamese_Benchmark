# Docker (GPU)

Build:

```bash
docker build -t vnasrbench:0.1 -f docker/Dockerfile .
```

Run with GPU (outputs written to your local `runs/` via the mounted volume):

```bash
docker run --gpus all --rm -it -v "$PWD":/workspace vnasrbench:0.1 --config configs/default.yaml
```
