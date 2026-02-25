from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_LABELS: Dict[str, str] = {
    "vivos_300": "VIVOS",
    "vlsp2020_300": "VLSP2020",
    "viet_youtube_asr_v2_300": "Viet YouTube ASR v2",
    "speech_massive_vie_300": "Speech-MASSIVE_vie",
}

MODEL_LABELS: Dict[str, str] = {
    "chunkformer_ctc_large_vie": "ChunkFormer-CTC-large",
    "wav2vec2_base_vn_250h": "wav2vec2-base-vi",
}

MODEL_COLORS: Dict[str, str] = {
    "chunkformer_ctc_large_vie": "#1b9e77",
    "wav2vec2_base_vn_250h": "#d95f02",
}


def _parse_run_id(run_id: str) -> Tuple[str, str, str]:
    parts = run_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected run_id format: {run_id}")
    dataset = parts[0]
    model = parts[1]
    # New frontier runs: dataset__model__latXXXX__stream
    if len(parts) >= 4 and parts[2].startswith("lat"):
        profile = parts[2]
    else:
        # Legacy streaming runs from 15_00.csv
        profile = "lat4000"
    return dataset, model, profile


def _extract_frontier_rows(df_frontier_new: pd.DataFrame, df_baseline: pd.DataFrame) -> pd.DataFrame:
    # Keep only new frontier rows (lat1200/lat2400)
    new_mask = df_frontier_new["run_id"].str.contains(r"__lat1200__|__lat2400__", regex=True, na=False)
    new_df = df_frontier_new.loc[new_mask].copy()

    # Keep only legacy 4000ms streaming rows for the 2 streaming models on *_300 datasets
    old_mask = (
        df_baseline["run_id"].str.contains(r"_300__", regex=True, na=False)
        & df_baseline["run_id"].str.contains(
            r"__chunkformer_ctc_large_vie__stream|__wav2vec2_base_vn_250h__stream", regex=True, na=False
        )
    )
    old_df = df_baseline.loc[old_mask].copy()

    merged = pd.concat([new_df, old_df], ignore_index=True, sort=False)

    parsed = merged["run_id"].map(_parse_run_id)
    merged["dataset_key"] = [x[0] for x in parsed]
    merged["model_key"] = [x[1] for x in parsed]
    merged["profile"] = [x[2] for x in parsed]
    merged["latency_ms"] = merged["streaming_chunk_ms"].fillna(merged["streaming_latency_proxy_ms"]).astype(float)

    merged = merged[
        merged["dataset_key"].isin(DATASET_LABELS.keys()) & merged["model_key"].isin(MODEL_LABELS.keys())
    ].copy()

    merged["dataset"] = merged["dataset_key"].map(DATASET_LABELS)
    merged["model"] = merged["model_key"].map(MODEL_LABELS)
    merged["streaming_wer_pct"] = merged["streaming_wer"].astype(float) * 100.0
    merged["streaming_wer_ci_low_pct"] = merged["streaming_wer_ci_low"].astype(float) * 100.0
    merged["streaming_wer_ci_high_pct"] = merged["streaming_wer_ci_high"].astype(float) * 100.0
    merged["delta_wer_pct"] = merged["delta_wer"].astype(float) * 100.0
    merged["stability_change_rate_pct"] = merged["stability_change_rate"].astype(float) * 100.0

    # Keep exactly one row per (dataset, model, profile)
    merged = merged.sort_values("run_id").drop_duplicates(["dataset_key", "model_key", "profile"], keep="last")
    merged = merged.sort_values(["dataset", "model", "latency_ms"]).reset_index(drop=True)
    return merged


def _plot_frontier_wer(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    datasets = list(DATASET_LABELS.values())

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        for mkey, mlabel in MODEL_LABELS.items():
            ms = sub[sub["model_key"] == mkey].sort_values("latency_ms")
            if ms.empty:
                continue
            y = ms["streaming_wer_pct"].to_numpy()
            y_low = ms["streaming_wer_ci_low_pct"].to_numpy()
            y_high = ms["streaming_wer_ci_high_pct"].to_numpy()
            yerr = np.vstack([y - y_low, y_high - y])
            ax.errorbar(
                ms["latency_ms"].to_numpy(),
                y,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=3,
                color=MODEL_COLORS[mkey],
                label=mlabel,
            )
        ax.set_title(ds)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Latency Proxy (ms)")
        ax.set_ylabel("Streaming WER (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Latency-Accuracy Frontier by Dataset", y=0.995, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.07, right=0.98, hspace=0.28, wspace=0.13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_frontier_delta(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    datasets = list(DATASET_LABELS.values())

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        for mkey, mlabel in MODEL_LABELS.items():
            ms = sub[sub["model_key"] == mkey].sort_values("latency_ms")
            if ms.empty:
                continue
            ax.plot(
                ms["latency_ms"].to_numpy(),
                ms["delta_wer_pct"].to_numpy(),
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[mkey],
                label=mlabel,
            )
        ax.set_title(ds)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Latency Proxy (ms)")
        ax.set_ylabel("Delta WER (pp)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Streaming Degradation (Delta WER) vs Latency", y=0.995, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.07, right=0.98, hspace=0.28, wspace=0.13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_stability_change(df: pd.DataFrame, out_path: Path) -> None:
    # Two panels: one per model; rows=dataset, cols=latency profile
    profiles = ["lat1200", "lat2400", "lat4000"]
    datasets = list(DATASET_LABELS.values())
    fig = plt.figure(figsize=(11.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
    cax = fig.add_subplot(gs[0, 2])
    ax_right.tick_params(axis="y", labelleft=False)
    axes = [ax_left, ax_right]

    for ax, mkey in zip(axes, MODEL_LABELS.keys()):
        mat = np.full((len(datasets), len(profiles)), np.nan, dtype=float)
        sub = df[df["model_key"] == mkey]
        for i, ds in enumerate(datasets):
            for j, p in enumerate(profiles):
                x = sub[(sub["dataset"] == ds) & (sub["profile"] == p)]
                if not x.empty:
                    mat[i, j] = float(x.iloc[0]["stability_change_rate_pct"])
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_title(MODEL_LABELS[mkey])
        ax.set_xticks(np.arange(len(profiles)))
        ax.set_xticklabels([p.replace("lat", "") for p in profiles])
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_xlabel("Latency (ms)")
        for i in range(len(datasets)):
            for j in range(len(profiles)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}"
                color = "black" if v < 55 else "white"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Stability Change Rate (%)")
    fig.suptitle("Streaming Stability: Hypothesis Change Rate", y=0.99, fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.80, bottom=0.16, wspace=0.18)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_robustness_4000(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df["profile"] == "lat4000"].copy()
    rows: List[Dict[str, float | str]] = []
    for mkey, mlabel in MODEL_LABELS.items():
        sub = ref[ref["model_key"] == mkey]
        if sub.empty:
            continue
        clean = sub[sub["dataset_key"] == "vivos_300"]
        if clean.empty:
            continue
        clean_wer = float(clean.iloc[0]["streaming_wer_pct"])
        for _, r in sub.iterrows():
            wer = float(r["streaming_wer_pct"])
            abs_delta = wer - clean_wer
            rel_delta = 0.0 if clean_wer == 0 else (abs_delta / clean_wer) * 100.0
            rows.append(
                {
                    "model_key": mkey,
                    "model": mlabel,
                    "dataset_key": str(r["dataset_key"]),
                    "dataset": str(r["dataset"]),
                    "wer_4000_pct": wer,
                    "abs_delta_pp": abs_delta,
                    "rel_delta_pct": rel_delta,
                }
            )
    out = pd.DataFrame(rows).sort_values(["model", "dataset"])
    return out


def _plot_robustness_abs_rel(rob: pd.DataFrame, out_path: Path) -> None:
    # Exclude clean rows for bar comparison
    xdf = rob[rob["dataset_key"] != "vivos_300"].copy()
    datasets = [DATASET_LABELS[k] for k in ["vlsp2020_300", "viet_youtube_asr_v2_300", "speech_massive_vie_300"]]
    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title, ylabel in [
        (axes[0], "abs_delta_pp", "Absolute Shift vs Clean (4000ms)", "Absolute Delta WER (pp)"),
        (axes[1], "rel_delta_pct", "Relative Shift vs Clean (4000ms)", "Relative Delta WER (%)"),
    ]:
        for i, mkey in enumerate(MODEL_LABELS.keys()):
            vals = []
            ms = xdf[xdf["model_key"] == mkey]
            for d in ["vlsp2020_300", "viet_youtube_asr_v2_300", "speech_massive_vie_300"]:
                row = ms[ms["dataset_key"] == d]
                vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, color=MODEL_COLORS[mkey], label=MODEL_LABELS[mkey])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Robustness Across Domains (Streaming @ 4000ms)", y=0.99, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.07, right=0.98, wspace=0.15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff_scatter(df: pd.DataFrame, out_path: Path) -> None:
    # A clean replacement for crowded "pareto" annotations: no per-point text labels.
    # Uses all latency profiles so the latency axis is meaningful.
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    marker_by_dataset = {
        "VIVOS": "o",
        "VLSP2020": "s",
        "Viet YouTube ASR v2": "^",
        "Speech-MASSIVE_vie": "D",
    }
    sub = df.copy()
    throughput = 1.0 / sub["streaming_rtf"].clip(lower=1e-9)
    size = 60 + 220 * (throughput - throughput.min()) / max(1e-9, (throughput.max() - throughput.min()))

    for mkey, mlabel in MODEL_LABELS.items():
        ms = sub[sub["model_key"] == mkey]
        for ds in DATASET_LABELS.values():
            dms = ms[ms["dataset"] == ds]
            if dms.empty:
                continue
            ax.scatter(
                dms["latency_ms"],
                dms["streaming_wer_pct"],
                s=size[dms.index],
                c=MODEL_COLORS[mkey],
                marker=marker_by_dataset[ds],
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

    ax.set_title("Streaming WER-Latency Trade-off (size ~ throughput)")
    ax.set_xlabel("Latency Proxy (ms)")
    ax.set_ylabel("Streaming WER (%)")
    ax.grid(True, alpha=0.3)

    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m], markeredgecolor="black", markersize=8, label=l)
        for m, l in MODEL_LABELS.items()
    ]
    dataset_handles = [
        plt.Line2D([0], [0], marker=mk, color="black", linestyle="None", markersize=7, label=ds)
        for ds, mk in marker_by_dataset.items()
    ]
    leg1 = ax.legend(handles=model_handles, title="Model", loc="upper left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles, title="Dataset", loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier_csv", default="results/frontier_new_raw.csv")
    ap.add_argument("--baseline_csv", default="results/15_00.csv")
    ap.add_argument("--out_merged_csv", default="results/frontier_merged.csv")
    ap.add_argument("--out_robust_csv", default="results/robustness_4000_streaming.csv")
    ap.add_argument("--fig_dir", default="report/figures")
    args = ap.parse_args()

    frontier_csv = Path(args.frontier_csv)
    baseline_csv = Path(args.baseline_csv)
    out_merged_csv = Path(args.out_merged_csv)
    out_robust_csv = Path(args.out_robust_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    out_robust_csv.parent.mkdir(parents=True, exist_ok=True)

    df_frontier_new = pd.read_csv(frontier_csv)
    df_baseline = pd.read_csv(baseline_csv)

    merged = _extract_frontier_rows(df_frontier_new, df_baseline)
    merged.to_csv(out_merged_csv, index=False)
    print(f"Wrote: {out_merged_csv} ({len(merged)} rows)")

    _plot_frontier_wer(merged, fig_dir / "fig_frontier_wer_latency.png")
    _plot_frontier_delta(merged, fig_dir / "fig_frontier_delta_latency.png")
    _plot_stability_change(merged, fig_dir / "fig_stability_change_rate_heatmap.png")
    _plot_tradeoff_scatter(merged, fig_dir / "fig_pareto_streaming.png")

    robust = _build_robustness_4000(merged)
    robust.to_csv(out_robust_csv, index=False)
    print(f"Wrote: {out_robust_csv} ({len(robust)} rows)")
    _plot_robustness_abs_rel(robust, fig_dir / "fig_robustness_abs_rel_4000.png")

    print(f"Wrote: {fig_dir / 'fig_frontier_wer_latency.png'}")
    print(f"Wrote: {fig_dir / 'fig_frontier_delta_latency.png'}")
    print(f"Wrote: {fig_dir / 'fig_stability_change_rate_heatmap.png'}")
    print(f"Wrote: {fig_dir / 'fig_pareto_streaming.png'}")
    print(f"Wrote: {fig_dir / 'fig_robustness_abs_rel_4000.png'}")


if __name__ == "__main__":
    main()
