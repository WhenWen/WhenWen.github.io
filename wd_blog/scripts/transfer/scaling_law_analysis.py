#!/usr/bin/env python3
"""
Fit scaling laws for each optimizer and compute speedups.
Scaling law form: loss = exp(intercept) * compute^slope + asymptote
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

try:
    import wandb
except ImportError as exc:
    raise SystemExit(
        "This script depends on wandb. Install it with `pip install wandb`."
    ) from exc


MODEL_ORDER = ["130m", "300m", "520m", "1.2b"]
EXPECTED_PARAMS = {
    "130m": 134_217_728,
    "300m": 301_989_888,
    "520m": 536_870_912,
    "1.2b": 1_207_959_552,
}

RUN_GROUPS: Dict[str, Sequence[str]] = {
    "MuonH": [
        "https://wandb.ai/marin-community/marin/runs/qwen3_130m_muonh_4096_lr_0.02_adam_lr_0.008-354a89",
        "https://wandb.ai/marin-community/marin/runs/qwen3_300m_muonh_4096_lr_0.01-f5ae20",
        "https://wandb.ai/marin-community/marin/runs/qwen3_520m_muonh_4096_lr_0.01-cb97ec",
        "https://wandb.ai/marin-community/marin/runs/qwen3_1_2b_muonh_4096_low_lr-f47219",
    ],
    "AdamH": [
        "https://wandb.ai/marin-community/marin/runs/llama_130m_adamh_lr0.02_adam_lr0.008_warmup1000_qk-add079",
        "https://wandb.ai/marin-community/marin/runs/llama_300m_adamh_lr0.02_adam_lr0.008_warmup1000_qk-dbc23e",
        "https://wandb.ai/marin-community/marin/runs/llama_520m_adamh_lr0.02_adam_lr0.004_warmup1000_qk-86d687",
        "https://wandb.ai/marin-community/marin/runs/llama_1_2b_adamh_lr0.02_adam_lr0.0015_warmup1000_qk-28a210",
    ],
    "Muon": [
        "https://wandb.ai/marin-community/marin/runs/qwen3_130m_muon_4096-04770b",
        "https://wandb.ai/marin-community/marin/runs/qwen3_300m_muon_4096-ee4f99",
        "https://wandb.ai/marin-community/marin/runs/qwen3_520m_muon_4096-361875",
        "https://wandb.ai/marin-community/marin/runs/qwen3_1_2b_muon_4096-d117d2",
    ],
    "AdamW": [
        "https://wandb.ai/marin-community/marin/runs/llama_130m_adamc_cos_4096-55946b",
        "https://wandb.ai/marin-community/marin/runs/llama_300m_adamc_cos_4096-a94d5d",
        "https://wandb.ai/marin-community/marin/runs/llama_520m_adamc_cos_4096-a3c775",
        "https://wandb.ai/marin-community/marin/runs/llama_1_2b_adamc_cos_4096-6b3a21",
    ],
}

METRIC_KEYS = [
    "eval/paloma/c4_en/loss",
    "eval/c4_en/loss",
    "metrics/eval/paloma/c4_en/loss",
]


@dataclass
class RunRecord:
    optimizer: str
    size_label: str
    params: int
    run_url: str
    loss: Optional[float]


def _coerce_summary_to_dict(summary_obj: object) -> Dict[str, object]:
    if isinstance(summary_obj, dict):
        return summary_obj
    for attr in ("_json_dict", "_items"):
        maybe_dict = getattr(summary_obj, attr, None)
        if isinstance(maybe_dict, dict):
            return maybe_dict
    try:
        return dict(summary_obj)
    except Exception:
        return {}


def extract_run_path(url: str) -> str:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 4 or path_parts[2] != "runs":
        raise ValueError(f"Unexpected W&B run URL format: {url}")
    entity, project, _, run_slug = path_parts[:4]
    return f"{entity}/{project}/{run_slug}"


def get_metric_from_summary(run: "wandb.apis.public.Run", key: str) -> Optional[float]:
    summary_dict = _coerce_summary_to_dict(getattr(run, "summary", {}))
    if key in summary_dict and isinstance(summary_dict[key], (int, float)):
        return float(summary_dict[key])
    if "/" in key:
        nested = summary_dict
        for part in key.split("/"):
            if not isinstance(nested, dict) or part not in nested:
                break
            nested = nested[part]
        else:
            if isinstance(nested, (int, float)):
                return float(nested)
    return None


def get_metric_from_history(run: "wandb.apis.public.Run", key: str) -> Optional[float]:
    try:
        history_df = run.history(keys=[key], pandas=True)
    except Exception:
        return None
    if key not in history_df.columns:
        return None
    series = history_df[key].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def fetch_final_loss(run: "wandb.apis.public.Run", metric_keys) -> Optional[float]:
    for metric_key in metric_keys:
        value = get_metric_from_summary(run, metric_key)
        if value is not None:
            return value
        value = get_metric_from_history(run, metric_key)
        if value is not None:
            return value
    return None


def build_run_records(api: Optional["wandb.Api"]) -> List[RunRecord]:
    records: List[RunRecord] = []
    for optimizer, run_urls in RUN_GROUPS.items():
        for idx, size_label in enumerate(MODEL_ORDER):
            run_url = run_urls[idx] if idx < len(run_urls) else ""
            if not run_url:
                continue
            params = EXPECTED_PARAMS[size_label]
            run_path = extract_run_path(run_url)
            loss = None
            if api is not None:
                try:
                    run = api.run(run_path)
                    loss = fetch_final_loss(run, METRIC_KEYS)
                    if loss is None:
                        print(f"Warning: could not find C4/en loss for {run_path}")
                except Exception as exc:
                    print(f"Warning: failed to load {run_path}: {exc}")
            records.append(
                RunRecord(
                    optimizer=optimizer,
                    size_label=size_label,
                    params=params,
                    run_url=run_url,
                    loss=loss,
                )
            )
    return records


def scaling_law(compute: np.ndarray, intercept: float, slope: float, asymptote: float) -> np.ndarray:
    """Scaling law: loss = exp(intercept) * compute^slope + asymptote"""
    return np.exp(intercept) * np.power(compute, slope) + asymptote


def fit_scaling_law(params: np.ndarray, losses: np.ndarray) -> Tuple[float, float, float]:
    """Fit the scaling law to data points."""
    # Initial guesses
    p0 = [5.0, -0.1, 2.5]  # intercept, slope, asymptote

    # Bounds to keep parameters reasonable
    bounds = (
        [-np.inf, -1.0, 0.0],  # lower bounds
        [np.inf, 0.0, np.inf]   # upper bounds (slope should be negative)
    )

    try:
        popt, _ = curve_fit(
            scaling_law,
            params,
            losses,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return tuple(popt)
    except Exception as e:
        print(f"Fitting failed: {e}")
        return p0[0], p0[1], p0[2]


def compute_speedup(
    params_fit_better: Tuple[float, float, float],
    params_fit_worse: Tuple[float, float, float],
    target_loss: float
) -> float:
    """
    Compute how much more compute the worse optimizer needs to reach the same loss
    as the better optimizer.

    Returns the ratio: compute_worse / compute_better
    """
    intercept_b, slope_b, asymptote_b = params_fit_better
    intercept_w, slope_w, asymptote_w = params_fit_worse

    # For better optimizer: loss = exp(intercept_b) * compute_b^slope_b + asymptote_b
    # Solving for compute_b: compute_b = ((loss - asymptote_b) / exp(intercept_b))^(1/slope_b)

    if target_loss <= asymptote_b or target_loss <= asymptote_w:
        return np.nan  # Can't reach this loss

    compute_better = np.power((target_loss - asymptote_b) / np.exp(intercept_b), 1.0 / slope_b)
    compute_worse = np.power((target_loss - asymptote_w) / np.exp(intercept_w), 1.0 / slope_w)

    return compute_worse / compute_better


def create_plot(
    records: List[RunRecord],
    fits: Dict[str, Tuple[float, float, float]],
    output_file: Path
) -> None:
    fig = go.Figure()

    palette = {
        "MuonH": "#1d4ed8",
        "Muon": "#60a5fa",
        "AdamH": "#f87171",
        "AdamW": "#fda4af",
    }

    # Group records by optimizer
    by_optimizer: Dict[str, List[RunRecord]] = {}
    for record in records:
        if record.loss is None:
            continue
        by_optimizer.setdefault(record.optimizer, []).append(record)

    # Plot data points and fitted curves
    param_range = np.logspace(
        np.log10(min(EXPECTED_PARAMS.values()) * 0.5),
        np.log10(max(EXPECTED_PARAMS.values()) * 2),
        100
    )

    for optimizer in ["MuonH", "Muon", "AdamH", "AdamW"]:
        if optimizer not in by_optimizer:
            continue

        optimizer_records = sorted(by_optimizer[optimizer], key=lambda r: r.params)
        color = palette.get(optimizer, "#6366f1")

        # Plot data points
        fig.add_trace(
            go.Scatter(
                x=[r.params for r in optimizer_records],
                y=[r.loss for r in optimizer_records],
                mode="markers",
                name=f"{optimizer} (data)",
                marker=dict(
                    size=12,
                    line=dict(width=1.5, color="#ffffff"),
                    color=color,
                ),
                customdata=[[r.run_url] for r in optimizer_records],
                hovertemplate=(
                    f"<b>{optimizer}</b><br>"
                    "Params: %{x:,}<br>"
                    "Loss: %{y:.4f}<extra></extra>"
                ),
                legendgroup=optimizer,
            )
        )

        # Plot fitted curve
        if optimizer in fits:
            intercept, slope, asymptote = fits[optimizer]
            fitted_loss = scaling_law(param_range, intercept, slope, asymptote)

            fig.add_trace(
                go.Scatter(
                    x=param_range,
                    y=fitted_loss,
                    mode="lines",
                    name=f"{optimizer} (fit)",
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=(
                        f"<b>{optimizer} fit</b><br>"
                        "Params: %{x:,.0f}<br>"
                        "Loss: %{y:.4f}<extra></extra>"
                    ),
                    legendgroup=optimizer,
                    showlegend=False,
                )
            )

    tickvals = [EXPECTED_PARAMS[label] for label in MODEL_ORDER]
    ticktext = MODEL_ORDER

    fig.update_layout(
        title=dict(
            text="Scaling Laws by Optimizer",
            x=0.5,
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis=dict(
            title="Model size (params)",
            tickvals=tickvals,
            ticktext=ticktext,
            type="log",
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(
            title="Final C4/en loss",
            type="log",
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="closest",
        margin=dict(l=60, r=40, t=80, b=60),
        plot_bgcolor="rgba(249, 250, 251, 0.4)",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
    )

    div_id = "scaling-law-plot"
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        div_id=div_id,
    )

    style_block = """
    <style>
        body {
            margin: 0;
            padding: 24px;
            background: #ffffff;
            font-family: 'Inter', system-ui, sans-serif;
        }
        .plotly-graph-div {
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
        }
    </style>
    """

    click_handler = f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var plot = document.getElementById('{div_id}');
            if (plot && plot.on) {{
                plot.on('plotly_click', function(eventData) {{
                    if (eventData && eventData.points && eventData.points.length) {{
                        var url = eventData.points[0].customdata ? eventData.points[0].customdata[0] : null;
                        if (url) {{
                            window.open(url, '_blank');
                        }}
                    }}
                }});
            }}
        }});
    </script>
    """

    html = html.replace("</head>", f"{style_block}</head>")
    html = html.replace("</body>", f"{click_handler}</body>")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    print(f"\n✓ Interactive plot written to {output_file}")


def main() -> None:
    try:
        api = wandb.Api()
    except Exception as exc:
        print(f"Warning: could not initialize W&B API ({exc}). Exiting.")
        return

    print("Fetching run data from W&B...")
    records = build_run_records(api)
    available = [r for r in records if r.loss is not None]

    if not available:
        raise SystemExit("No loss values were retrieved; aborting.")

    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    for record in sorted(available, key=lambda r: (r.optimizer, r.params)):
        print(f"{record.optimizer:6s} | {record.size_label:4s} | {record.params:>12,} | loss: {record.loss:.4f}")

    # Group by optimizer and fit scaling laws
    by_optimizer: Dict[str, List[RunRecord]] = {}
    for record in available:
        by_optimizer.setdefault(record.optimizer, []).append(record)

    fits: Dict[str, Tuple[float, float, float]] = {}

    print("\n" + "=" * 70)
    print("Scaling Law Fits: loss = exp(intercept) * params^slope + asymptote")
    print("=" * 70)

    for optimizer, optimizer_records in by_optimizer.items():
        params = np.array([r.params for r in optimizer_records])
        losses = np.array([r.loss for r in optimizer_records])

        intercept, slope, asymptote = fit_scaling_law(params, losses)
        fits[optimizer] = (intercept, slope, asymptote)

        print(f"\n{optimizer}:")
        print(f"  intercept = {intercept:.4f}")
        print(f"  slope     = {slope:.4f}")
        print(f"  asymptote = {asymptote:.4f}")
        print(f"  Formula: loss = exp({intercept:.4f}) * params^({slope:.4f}) + {asymptote:.4f}")

        # Show fit quality
        fitted = scaling_law(params, intercept, slope, asymptote)
        mse = np.mean((losses - fitted) ** 2)
        print(f"  MSE = {mse:.6f}")

    # Compute speedups
    print("\n" + "=" * 70)
    print("Speedup Analysis")
    print("=" * 70)

    # Get representative loss values for comparison
    loss_values = [r.loss for r in available]
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    test_losses = np.linspace(min_loss * 1.05, max_loss * 0.95, 5)

    # MuonH vs Muon
    if "MuonH" in fits and "Muon" in fits:
        print("\n--- MuonH vs Muon ---")
        print("How much more compute does Muon need to reach the same loss as MuonH?")
        for target_loss in test_losses:
            speedup = compute_speedup(fits["MuonH"], fits["Muon"], target_loss)
            if not np.isnan(speedup):
                print(f"  At loss={target_loss:.4f}: Muon needs {speedup:.2f}x more compute")

        # Average speedup at 1.2B loss level
        loss_1_2b_muonh = [r.loss for r in by_optimizer.get("MuonH", []) if r.size_label == "1.2b"]
        if loss_1_2b_muonh:
            target = loss_1_2b_muonh[0]
            speedup = compute_speedup(fits["MuonH"], fits["Muon"], target)
            print(f"\n  At MuonH 1.2B loss ({target:.4f}): Muon needs {speedup:.2f}x more compute")

    # AdamH vs AdamW
    if "AdamH" in fits and "AdamW" in fits:
        print("\n--- AdamH vs AdamW ---")
        print("How much more compute does AdamW need to reach the same loss as AdamH?")
        for target_loss in test_losses:
            speedup = compute_speedup(fits["AdamH"], fits["AdamW"], target_loss)
            if not np.isnan(speedup):
                print(f"  At loss={target_loss:.4f}: AdamW needs {speedup:.2f}x more compute")

        # Average speedup at 1.2B loss level
        loss_1_2b_adamh = [r.loss for r in by_optimizer.get("AdamH", []) if r.size_label == "1.2b"]
        if loss_1_2b_adamh:
            target = loss_1_2b_adamh[0]
            speedup = compute_speedup(fits["AdamH"], fits["AdamW"], target)
            print(f"\n  At AdamH 1.2B loss ({target:.4f}): AdamW needs {speedup:.2f}x more compute")

    # Create the plot
    output_path = Path("wd_blog/public/experiments/scaling_law_analysis.html")
    create_plot(available, fits, output_path)


if __name__ == "__main__":
    main()
