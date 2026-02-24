FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /workspace

# Copy dependency files first for caching (poetry.lock if it exists in context)
COPY p* /workspace/

RUN poetry config virtualenvs.create false \
    && poetry install --with dev -E gpu --no-root

COPY . /workspace/

RUN poetry install --with dev -E gpu --only-root

RUN useradd -m -u 1000 app && chown -R app:app /workspace
USER app

ENTRYPOINT ["poetry", "run", "vnasrbench"]
