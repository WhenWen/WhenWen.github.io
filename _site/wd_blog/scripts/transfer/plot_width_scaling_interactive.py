#!/usr/bin/env python3
"""
Interactive width-scaling visualization.

Mirrors the Matplotlib workflow from plot_width_scaling.py but renders an
interactive Plotly figure similar to plot_depth_scaling_interactive.py. The
focus slice keeps depth fixed (num_layers=4) while sweeping the hidden_dim /
warmup grid to illustrate how width alters the optimal learning-rate window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = Path(__file__).with_name("wandb_hyperball_runs.csv")
OUTPUT_HTML = PROJECT_ROOT / "public" / "experiments" / "width_scaling_interactive.html"
WANDB_PROJECT_RUNS = "https://wandb.ai/marin-community/Hyperball/runs"

LR_WINDOW = (1.25e-3, 3.4e-2)
LR_TICKVALS: List[float] = [2.5e-3, 5e-3, 1e-2, 2e-2]
LR_TICKTEXT: List[str] = ["2.5e-3", "5e-3", "1e-2", "2e-2"]
MAX_HIDDEN_DIM = 1024
OPTIMIZER_ORDER = ["adam", "adamh", "muon", "muonh"]
OPTIMIZER_POSITIONS = {
    "adam": (1, 1),
    "adamh": (1, 2),
    "muon": (2, 1),
    "muonh": (2, 2),
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
    # fall back to string labels so we can still differentiate traces
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


def _warmup_styles(values: Sequence) -> Dict:
    styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    mapping = {}
    for idx, val in enumerate(sorted(values, key=lambda x: str(x))):
        mapping[val] = styles[idx % len(styles)]
    return mapping


def build_width_figure(df: pd.DataFrame) -> go.Figure:
    subplot_titles = [opt.upper() for opt in OPTIMIZER_ORDER]
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.28,
        subplot_titles=subplot_titles,
    )

    unique_opts = list(df["optimizer_name"].astype(str).unique())
    available_order = [opt for opt in OPTIMIZER_ORDER if opt in unique_opts]
    legend_anchor = available_order[0] if available_order else OPTIMIZER_ORDER[0]
    envelope_anchor = available_order[-1] if available_order else OPTIMIZER_ORDER[-1]

    for optimizer_name in OPTIMIZER_ORDER:
        row_idx, col_idx = OPTIMIZER_POSITIONS[optimizer_name]
        subset = df[df["optimizer_name"] == optimizer_name].copy()
        if optimizer_name == "muonh":
            # subset = subset[subset["run_name"].str.contains("warmup-0")].copy()
            subset = subset[subset["run_name"].str.contains("linear-input-norm")].copy()
        if optimizer_name == "adamh":
            subset = subset[subset["run_name"].str.contains("mup")].copy()
            subset = subset[subset["warmup"] == "0.1"].copy()
            # subset = subset[subset["run_name"].str.contains("nl")].copy()
            # subset = subset[subset["run_name"].str.contains("warmup")].copy()
        dims = sorted(subset["hidden_dim"].unique())
        palette = _pick_palette(dims)
        warmup_styles = _warmup_styles(subset["warmup"].unique())

        best_points: List[dict] = []
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
                dash = warmup_styles[warmup]
                label = f"{dim} hidden"
                fig.add_trace(
                    go.Scatter(
                        x=trace_df["learning_rate"],
                        y=trace_df["loss"],
                        mode="lines+markers",
                        marker=dict(size=7, color=color),
                        line=dict(width=2, color=color, dash=dash),
                        name=label,
                        legendgroup=f"{optimizer_name}-{dim}-{warmup}",
                        showlegend=(optimizer_name == legend_anchor),
                        text=trace_df["run_name"],
                        customdata=trace_df[["hidden_dim", "warmup", "run_url"]],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            f"optimizer: {optimizer_name}<br>"
                            "hidden_dim: %{customdata[0]:.0f}<br>"
                            "warmup: %{customdata[1]}<br>"
                            "learning rate: %{x:.4f}<br>"
                            "loss: %{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
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
                    "run_name": str(best_row["run_name"]),
                    "run_url": str(best_row["run_url"]),
                }
            )

            fig.add_trace(
                go.Scatter(
                    x=[best_row["learning_rate"]],
                    y=[best_row["loss"]],
                    mode="markers",
                    marker=dict(symbol="star", size=16, line=dict(color="black", width=1.2), color=palette[dim]),
                    name="min loss",
                    legendgroup=f"{optimizer_name}-{dim}-best",
                    showlegend=False,
                    text=[best_row["run_name"]],
                    customdata=[[best_row["hidden_dim"], best_row["run_url"]]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"optimizer: {optimizer_name}<br>"
                        "hidden_dim: %{customdata[0]:.0f}<br>"
                        "learning rate: %{x:.4f}<br>"
                        "loss (min): %{y:.3f}<extra></extra>"
                    ),
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
        title="Width scaling @ num_layers=4",
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
        margin=dict(t=120, l=70, r=210, b=80),
    )

    return fig


def save_html(fig: go.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
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
            if (!custom || custom.length < 3) {
                return;
            }
            var url = custom[custom.length - 1];
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
    df = load_width_slice(CSV_PATH)
    if df.empty:
        raise SystemExit("No runs matched the width-scaling slice; nothing to plot.")

    fig = build_width_figure(df)
    save_html(fig, OUTPUT_HTML)
    print(f"✓ Interactive width plot saved to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()


