#!/usr/bin/env python3
"""
Static depth-scaling visualization using Matplotlib.

This script mirrors plot_depth_scaling_interactive.py but outputs a static
figure using Matplotlib instead of an interactive Plotly HTML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = Path(__file__).with_name("wandb_hyperball_runs.csv")
OUTPUT_PNG = Path(__file__).with_name("depth_scaling_static.png")
OUTPUT_PDF = Path(__file__).with_name("depth_scaling_static.pdf")

LR_WINDOW: Tuple[float, float] = (1.25e-3, 3.4e-2)
LR_TICKVALS: List[float] = [2.5e-3, 5e-3, 1e-2, 2e-2]
LR_TICKTEXT: List[str] = ["2.5e-3", "5e-3", "1e-2", "2e-2"]
OPTIMIZER_ORDER = ["adam", "adamh", "muon", "muonh"]
OPTIMIZER_POSITIONS = {
    "adam": (0, 0),
    "adamh": (0, 1),
    "muon": (1, 0),
    "muonh": (1, 1),
}


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Normalize heterogeneous boolean encodings (True/False strings, etc.)."""
    lowered = series.map(lambda v: str(v).strip().lower() if pd.notna(v) else "")
    return lowered.map({"true": True, "false": False})


def load_depth_slice(csv_path: Path) -> pd.DataFrame:
    """Replicate the filtering logic from plot_depth_scaling2."""
    df = pd.read_csv(csv_path)

    numeric_cols = ["hidden_dim", "learning_rate", "min_lr_ratio", "num_layers", "loss"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "input_norm" in df.columns:
        df["input_norm"] = _coerce_bool(df["input_norm"])
    else:
        df["input_norm"] = False

    filtered = df[
        (df["hidden_dim"] == 128.0)
        & (df["min_lr_ratio"] == 0.0)
        & (df["loss"] < 4.6)
        & (df["input_norm"] == False)
    ].copy()

    filtered = filtered.dropna(subset=["learning_rate", "loss", "num_layers", "optimizer_name"])
    filtered["optimizer_name"] = filtered["optimizer_name"].astype(str)
    filtered["num_layers"] = filtered["num_layers"].astype(int)
    filtered["run_id"] = filtered["run_id"].astype(str)
    filtered["run_name"] = filtered["run_name"].fillna(filtered["run_id"])

    return filtered


def _pick_palette(keys: List[int]) -> Dict[int, str]:
    """Generate a color palette for the given depth keys."""
    cmap = plt.cm.tab20
    mapping: Dict[int, str] = {}
    for idx, key in enumerate(sorted(keys)):
        mapping[key] = cmap(idx % 20)
    return mapping


def _format_depth(depth: int) -> str:
    return f"{depth} layers"


def build_depth_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the matplotlib figure with 2x2 subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    fig.suptitle("Depth scaling @ hidden_dim=128", fontsize=16, fontweight="bold", y=0.98)

    palette_cache: Dict[str, Dict[int, str]] = {}
    all_depths: set = set()

    for optimizer_name in OPTIMIZER_ORDER:
        row_idx, col_idx = OPTIMIZER_POSITIONS[optimizer_name]
        ax = axes[row_idx, col_idx]
        subset = df[df["optimizer_name"] == optimizer_name].copy()

        if optimizer_name == "muonh":
            subset = subset[subset["run_name"].str.contains("linear-input-norm")].copy()
        if optimizer_name == "adamh":
            subset = subset[subset["run_name"].str.contains("mup")].copy()
            subset = subset[subset["run_name"].str.contains("nl")].copy()

        depths = sorted(subset["num_layers"].unique())
        all_depths.update(depths)
        palette_cache[optimizer_name] = _pick_palette(depths)

        best_points: List[Dict[str, float]] = []

        for depth in depths:
            depth_df = subset[subset["num_layers"] == depth].copy()
            depth_df = depth_df.sort_values("learning_rate")
            depth_df = depth_df[
                (depth_df["learning_rate"] > LR_WINDOW[0]) & (depth_df["learning_rate"] < LR_WINDOW[1])
            ]
            if depth_df.empty:
                continue

            color = palette_cache[optimizer_name][depth]

            ax.plot(
                depth_df["learning_rate"],
                depth_df["loss"],
                marker="o",
                markersize=5,
                linewidth=1.5,
                color=color,
                label=_format_depth(depth),
            )

            best_idx = depth_df["loss"].idxmin()
            best_row = depth_df.loc[best_idx]
            best_points.append(
                {
                    "lr": float(best_row["learning_rate"]),
                    "loss": float(best_row["loss"]),
                    "depth": int(best_row["num_layers"]),
                }
            )

            ax.plot(
                best_row["learning_rate"],
                best_row["loss"],
                marker="*",
                markersize=14,
                color=color,
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

    global_palette = _pick_palette(list(all_depths))
    legend_elements = [
        Line2D([0], [0], color=global_palette[d], marker="o", markersize=6, linewidth=1.5, label=_format_depth(d))
        for d in sorted(all_depths)
    ]
    legend_elements.append(
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.5, label="min-loss envelope")
    )
    legend_elements.append(
        Line2D([0], [0], color="gray", marker="*", markersize=12, linestyle="None", label="min loss")
    )

    ncols = (len(legend_elements) + 1) // 2
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=ncols,
        fontsize=12,
        framealpha=0.95,
        edgecolor="#94a3b8",
    )

    plt.tight_layout(rect=[0, 0, 1.0, 0.87])
    return fig


def save_figure(fig: plt.Figure, output_png: Path, output_pdf: Path) -> None:
    """Save the figure to PNG and PDF formats."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")


def main() -> None:
    df = load_depth_slice(CSV_PATH)
    if df.empty:
        raise SystemExit("No runs matched the depth-scaling slice; nothing to plot.")

    fig = build_depth_figure(df)
    save_figure(fig, OUTPUT_PNG, OUTPUT_PDF)

    print(f"Static depth plot saved to:\n  - {OUTPUT_PNG}\n  - {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
