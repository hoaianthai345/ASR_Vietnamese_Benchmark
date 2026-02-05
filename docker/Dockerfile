FROM python:3.10-slim

ENV POETRY_VERSION=1.8.4
WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts

RUN poetry config virtualenvs.create false     && poetry install --with dev

CMD ["python", "-m", "vnasrbench.cli", "--config", "configs/default.yaml"]
