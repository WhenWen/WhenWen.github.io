#!/usr/bin/env python3
"""
Static width-scaling visualization using Matplotlib.

This script mirrors plot_width_scaling_interactive.py but outputs a static
figure using Matplotlib instead of an interactive Plotly HTML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

CSV_PATH = Path(__file__).with_name("wandb_hyperball_runs.csv")
OUTPUT_PNG = Path(__file__).with_name("width_scaling_static.png")
OUTPUT_PDF = Path(__file__).with_name("width_scaling_static.pdf")

LR_WINDOW: Tuple[float, float] = (1.25e-3, 3.4e-2)
LR_TICKVALS: List[float] = [2.5e-3, 5e-3, 1e-2, 2e-2]
LR_TICKTEXT: List[str] = ["2.5e-3", "5e-3", "1e-2", "2e-2"]
MAX_HIDDEN_DIM = 1024
OPTIMIZER_ORDER = ["adam", "adamh", "muon", "muonh"]
OPTIMIZER_POSITIONS = {
    "adam": (0, 0),
    "adamh": (0, 1),
    "muon": (1, 0),
    "muonh": (1, 1),
}


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Normalize heterogeneous boolean encodings."""
    lowered = series.map(lambda v: str(v).strip().lower() if pd.notna(v) else "")
    return lowered.map({"true": True, "false": False})


def _coerce_warmup(series: pd.Series) -> pd.Series:
    """Handle warmup encodings (ints, floats, strings, NaNs)."""
    as_numeric = pd.to_numeric(series, errors="coerce")
    if as_numeric.notna().all():
        return as_numeric
    return series.fillna("unknown").astype(str)


def load_width_slice(csv_path: Path) -> pd.DataFrame:
    """Replicate the filtering logic from plot_width_scaling.py."""
    df = pd.read_csv(csv_path)

    numeric_cols = ["hidden_dim", "learning_rate", "min_lr_ratio", "num_layers", "loss", "warmup"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "input_norm" in df.columns:
        df["input_norm"] = _coerce_bool(df["input_norm"])
    else:
        df["input_norm"] = False

    if "warmup" in df.columns:
        df["warmup"] = _coerce_warmup(df["warmup"])
    else:
        df["warmup"] = 0

    filtered = df[
        (df["num_layers"] == 4.0)
        & (df["min_lr_ratio"] == 0.0)
        & (df["loss"] < 5.0)
        & (df["input_norm"] == False)
        & (df["hidden_dim"] <= MAX_HIDDEN_DIM)
    ].copy()

    filtered = filtered.dropna(subset=["learning_rate", "loss", "hidden_dim", "optimizer_name"])
    filtered["optimizer_name"] = filtered["optimizer_name"].astype(str)
    filtered["hidden_dim"] = filtered["hidden_dim"].astype(int)

    if "run_id" in filtered.columns:
        filtered["run_id"] = filtered["run_id"].astype(str)
    else:
        filtered["run_id"] = ""
    filtered["run_name"] = filtered.get("run_name", filtered["run_id"]).fillna(filtered["run_id"])

    return filtered


def _pick_palette(keys: List[int]) -> Dict[int, str]:
    """Generate a color palette for the given hidden_dim keys."""
    cmap = plt.cm.tab20
    mapping: Dict[int, str] = {}
    for idx, key in enumerate(sorted(keys)):
        mapping[key] = cmap(idx % 20)
    return mapping


def _warmup_styles(values: Sequence) -> Dict:
    """Generate line styles for different warmup values."""
    styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
    mapping = {}
    for idx, val in enumerate(sorted(values, key=lambda x: str(x))):
        mapping[val] = styles[idx % len(styles)]
    return mapping


def build_width_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the matplotlib figure with 2x2 subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    fig.suptitle("Width scaling @ num_layers=4", fontsize=16, fontweight="bold", y=0.98)

    all_dims: set = set()

    for optimizer_name in OPTIMIZER_ORDER:
        row_idx, col_idx = OPTIMIZER_POSITIONS[optimizer_name]
        ax = axes[row_idx, col_idx]
        subset = df[df["optimizer_name"] == optimizer_name].copy()

        if optimizer_name == "muonh":
            subset = subset[subset["run_name"].str.contains("linear-input-norm")].copy()
        if optimizer_name == "adamh":
            subset = subset[subset["run_name"].str.contains("mup")].copy()
            subset = subset[subset["warmup"] == "0.1"].copy()

        dims = sorted(subset["hidden_dim"].unique())
        all_dims.update(dims)
        palette = _pick_palette(dims)
        warmup_styles = _warmup_styles(subset["warmup"].unique())

        best_points: List[Dict[str, float]] = []

        for dim in dims:
            dim_df = subset[subset["hidden_dim"] == dim]
            warmups = sorted(dim_df["warmup"].unique(), key=lambda x: (str(type(x)), x))

            for warmup in warmups:
                trace_df = dim_df[dim_df["warmup"] == warmup].sort_values("learning_rate")
                trace_df = trace_df[
                    (trace_df["learning_rate"] > LR_WINDOW[0]) & (trace_df["learning_rate"] < LR_WINDOW[1])
                ]
                if trace_df.empty:
                    continue

                color = palette[dim]
                linestyle = warmup_styles[warmup]

                ax.plot(
                    trace_df["learning_rate"],
                    trace_df["loss"],
                    marker="o",
                    markersize=5,
                    linewidth=1.5,
                    color=color,
                    linestyle=linestyle,
                    label=f"{dim} hidden",
                )

            if dim_df.empty:
                continue

            best_idx = dim_df["loss"].idxmin()
            best_row = dim_df.loc[best_idx]
            best_points.append(
                {
                    "lr": float(best_row["learning_rate"]),
                    "loss": float(best_row["loss"]),
                    "hidden_dim": int(best_row["hidden_dim"]),
                }
            )

            ax.plot(
                best_row["learning_rate"],
                best_row["loss"],
                marker="*",
                markersize=14,
                color=palette[dim],
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=10,
            )

        if best_points:
            best_sorted = sorted(best_points, key=lambda item: item["loss"])
            ax.plot(
                [p["lr"] for p in best_sorted],
                [p["loss"] for p in best_sorted],
                linestyle="--",
                linewidth=1.5,
                color="#111827",
                zorder=5,
            )

        ax.set_xscale("log")
        ax.set_xlim(LR_WINDOW)
        ax.set_xticks(LR_TICKVALS)
        ax.set_xticklabels(LR_TICKTEXT)
        ax.set_xlabel("learning rate", fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        title_map = {"adam": "AdamW", "adamh": "AdamH", "muon": "Muon", "muonh": "MuonH"}
        title = title_map.get(optimizer_name, optimizer_name.upper())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_facecolor("white")

        if col_idx == 0:
            ax.set_ylabel("loss", fontsize=14)

    global_palette = _pick_palette(list(all_dims))
    legend_elements = [
        Line2D([0], [0], color=global_palette[d], marker="o", markersize=6, linewidth=1.5, label=f"{d} hidden")
        for d in sorted(all_dims)
    ]
    legend_elements.append(
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.5, label="min-loss envelope")
    )
    legend_elements.append(
        Line2D([0], [0], color="gray", marker="*", markersize=12, linestyle="None", label="min loss")
    )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(legend_elements),
        fontsize=12,
        framealpha=0.95,
        edgecolor="#94a3b8",
    )

    plt.tight_layout(rect=[0, 0, 1.0, 0.90])
    return fig


def save_figure(fig: plt.Figure, output_png: Path, output_pdf: Path) -> None:
    """Save the figure to PNG and PDF formats."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")


def main() -> None:
    df = load_width_slice(CSV_PATH)
    if df.empty:
        raise SystemExit("No runs matched the width-scaling slice; nothing to plot.")

    fig = build_width_figure(df)
    save_figure(fig, OUTPUT_PNG, OUTPUT_PDF)

    print(f"Static width plot saved to:\n  - {OUTPUT_PNG}\n  - {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
