#!/usr/bin/env python3
"""
Interactive depth-scaling visualization.

This script bridges the static Matplotlib workflow from plot_depth_scaling2.py
and the Plotly-based approach from analyze.py. It reads the CSV produced by
plot_wb.py / plot_depth_scaling2.py, filters to the hidden-dim/min-lr slice of
interest, and emits an interactive HTML figure that highlights how depth
changes the optimal learning-rate window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = Path(__file__).with_name("wandb_hyperball_runs.csv")
OUTPUT_HTML = PROJECT_ROOT / "public" / "experiments" / "depth_scaling_interactive.html"
WANDB_PROJECT_RUNS = "https://wandb.ai/marin-community/Hyperball/runs"

LR_WINDOW: Tuple[float, float] = (1.25e-3, 3.4e-2)
LR_TICKVALS: List[float] = [2.5e-3, 5e-3, 1e-2, 2e-2]
LR_TICKTEXT: List[str] = ["2.5e-3", "5e-3", "1e-2", "2e-2"]
OPTIMIZER_ORDER = ["adam", "adamh", "muon", "muonh"]
OPTIMIZER_POSITIONS = {
    "adam": (1, 1),
    "adamh": (1, 2),
    "muon": (2, 1),
    "muonh": (2, 2),
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
    filtered["run_url"] = filtered["run_id"].map(
        lambda rid: f"{WANDB_PROJECT_RUNS}/{rid}" if rid and rid.lower() != "nan" else ""
    )

    return filtered


def _pick_palette(keys: Iterable[int]) -> Dict[int, str]:
    palette = qualitative.Dark24 + qualitative.Set3 + qualitative.Bold
    mapping: Dict[int, str] = {}
    for idx, key in enumerate(sorted(keys)):
        mapping[key] = palette[idx % len(palette)]
    return mapping


def _format_depth(depth: int) -> str:
    return f"{depth} layers"


def build_depth_figure(df: pd.DataFrame) -> go.Figure:
    subplot_titles = [opt.upper() for opt in OPTIMIZER_ORDER]
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.28,
        subplot_titles=subplot_titles,
    )

    palette_cache: Dict[str, Dict[int, str]] = {}
    unique_opts = list(df["optimizer_name"].astype(str).unique())
    available_order = [opt for opt in OPTIMIZER_ORDER if opt in unique_opts]
    legend_anchor = available_order[0] if available_order else OPTIMIZER_ORDER[0]
    envelope_anchor = available_order[-1] if available_order else OPTIMIZER_ORDER[-1]

    for optimizer_name in OPTIMIZER_ORDER:
        row_idx, col_idx = OPTIMIZER_POSITIONS[optimizer_name]
        subset = df[df["optimizer_name"] == optimizer_name].copy()
        if optimizer_name == "muonh":
            subset = subset[subset["run_name"].str.contains("linear-input-norm")].copy()
        if optimizer_name == "adamh":
            subset = subset[subset["run_name"].str.contains("mup")].copy()
            subset = subset[subset["run_name"].str.contains("nl")].copy()

        depths = sorted(subset["num_layers"].unique())
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
            label = f"{_format_depth(depth)}"
            fig.add_trace(
                go.Scatter(
                    x=depth_df["learning_rate"],
                    y=depth_df["loss"],
                    mode="lines+markers",
                    marker=dict(size=7, color=color),
                    line=dict(width=2, color=color),
                    name=label,
                    legendgroup=f"{optimizer_name}-{depth}",
                    showlegend=(optimizer_name == legend_anchor),
                    text=depth_df["run_name"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"optimizer: {optimizer_name}<br>"
                        "depth: %{customdata[0]:.0f} layers<br>"
                        "learning rate: %{x:.4f}<br>"
                        "loss: %{y:.3f}<extra></extra>"
                    ),
                    customdata=depth_df[["num_layers", "run_url"]],
                ),
                row=row_idx,
                col=col_idx,
            )

            best_idx = depth_df["loss"].idxmin()
            best_row = depth_df.loc[best_idx]
            best_points.append(
                {
                    "lr": float(best_row["learning_rate"]),
                    "loss": float(best_row["loss"]),
                    "depth": int(best_row["num_layers"]),
                    "run_name": str(best_row["run_name"]),
                    "run_url": str(best_row["run_url"]),
                }
            )

            fig.add_trace(
                go.Scatter(
                    x=[best_row["learning_rate"]],
                    y=[best_row["loss"]],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=16,
                        line=dict(color="black", width=1.2),
                        color=color,
                    ),
                    name="min loss",
                    legendgroup=f"{optimizer_name}-best",
                    showlegend=False,
                    text=[best_row["run_name"]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"optimizer: {optimizer_name}<br>"
                        "depth: %{customdata[0]:.0f} layers<br>"
                        "learning rate: %{x:.4f}<br>"
                        "loss (min): %{y:.3f}<extra></extra>"
                    ),
                    customdata=[[best_row["num_layers"], best_row["run_url"]]],
                ),
                row=row_idx,
                col=col_idx,
            )

        if best_points:
            best_sorted = sorted(best_points, key=lambda item: item["loss"])
            fig.add_trace(
                go.Scatter(
                    x=[p["lr"] for p in best_sorted],
                    y=[p["loss"] for p in best_sorted],
                    mode="lines",
                    line=dict(color="#111827", width=1.5, dash="dash"),
                    name="min-loss envelope",
                    legendgroup=f"{optimizer_name}-envelope",
                    showlegend=(optimizer_name == envelope_anchor),
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=col_idx,
            )

        fig.update_xaxes(
            row=row_idx,
            col=col_idx,
            type="log",
            title_text="learning rate",
            ticks="outside",
            tickvals=LR_TICKVALS,
            ticktext=LR_TICKTEXT,
            range=[np.log10(LR_WINDOW[0]), np.log10(LR_WINDOW[1])],
        )

    fig.update_yaxes(title_text="loss", row=1, col=1)

    fig.update_layout(
        title=(
            "Depth scaling @ hidden_dim=128"
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=13),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.45)",
            borderwidth=1,
        ),
        margin=dict(t=120, l=70, r=190, b=80),
    )

    return fig


def save_html(fig: go.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    html = output_path.read_text()
    styled_html = html.replace(
        "</head>",
        """
<style>
    body {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background: #f9fafb;
        margin: 0;
        padding: 24px;
    }
    .plotly-graph-div {
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
    }
</style>
</head>
""",
        1,
    )
    styled_html = styled_html.replace(
        "</body>",
        """
<script>
(function attachWandbLinks() {
    function connect(graph) {
        if (!graph || typeof graph.on !== "function") {
            return;
        }
        graph.on("plotly_click", function(ev) {
            if (!ev || !ev.points || !ev.points.length) {
                return;
            }
            var custom = ev.points[0].customdata;
            if (!custom || custom.length < 2) {
                return;
            }
            var url = custom[1];
            if (typeof url === "string" && url.length > 0) {
                window.open(url, "_blank", "noopener");
            }
        });
    }
    function init() {
        document.querySelectorAll(".plotly-graph-div").forEach(connect);
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
</script>
</body>
""",
        1,
    )
    output_path.write_text(styled_html)


def main() -> None:
    df = load_depth_slice(CSV_PATH)
    if df.empty:
        raise SystemExit("No runs matched the depth-scaling slice; nothing to plot.")

    fig = build_depth_figure(df)
    save_html(fig, OUTPUT_HTML)

    print(f"✓ Interactive depth plot saved to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()

