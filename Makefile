.PHONY: setup lint type test format run aggregate

setup:
	poetry install --with dev -E gpu

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

type:
	poetry run mypy src

test:
	poetry run pytest -q

run:
	poetry run vnasrbench --config configs/default.yaml

aggregate:
	poetry run python scripts/aggregate.py --runs_dir runs --out_csv results.csv --out_fig results.png
