from __future__ import annotations

import argparse

from vnasrbench.cli import run_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    metrics = run_from_config(args.config)
    print("Done. Metrics:", metrics)


if __name__ == "__main__":
    main()
