#!/usr/bin/env python3
"""
Fetch final C4/en losses for the runs listed in plot.py and create an
interactive Plotly chart with clickable points that open the W&B run.

Requires a valid W&B API key (wandb login) to pull the metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go

try:
    import wandb
except ImportError as exc:  # pragma: no cover - wandb should be available in prod
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
    "AdamC": [
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
    run_path: str
    run_name: str
    loss: Optional[float]


def _coerce_summary_to_dict(summary_obj: object) -> Dict[str, object]:
    if isinstance(summary_obj, dict):
        return summary_obj
    for attr in ("_json_dict", "_items"):
        maybe_dict = getattr(summary_obj, attr, None)
        if isinstance(maybe_dict, dict):
            return maybe_dict
    try:
        return dict(summary_obj)  # type: ignore[arg-type]
    except Exception:
        return {}


def extract_run_path(url: str) -> str:
    """
    Convert a run URL like https://wandb.ai/entity/project/runs/run-id to
    the API path entity/project/run-id.
    """
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


def fetch_final_loss(
    run: "wandb.apis.public.Run", metric_keys: Iterable[str]
) -> Optional[float]:
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
            run_name = f"{optimizer} {size_label}"
            loss = None
            if api is not None:
                try:
                    run = api.run(run_path)
                    run_name = run.name or run_path.split("/")[-1]
                    loss = fetch_final_loss(run, METRIC_KEYS)
                    if loss is None:
                        print(f"Warning: could not find C4/en loss for {run_path}")
                except Exception as exc:
                    print(f"Warning: failed to load {run_path}: {exc}")
            else:
                # Synthetic fallback so the plot still renders for debugging
                loss = max(0.5, 1.8 - 0.2 * idx)
            records.append(
                RunRecord(
                    optimizer=optimizer,
                    size_label=size_label,
                    params=params,
                    run_url=run_url,
                    run_path=run_path,
                    run_name=run_name,
                    loss=loss,
                )
            )
    return records


def create_plot(records: Sequence[RunRecord], output_file: Path) -> None:
    fig = go.Figure()

    # Order points by parameter count per optimizer
    by_optimizer: Dict[str, List[RunRecord]] = {}
    for record in records:
        if record.loss is None:
            continue
        by_optimizer.setdefault(record.optimizer, []).append(record)

    palette = {
        "MuonH": "#1d4ed8",
        "Muon": "#60a5fa",
        "AdamH": "#f87171",
        "AdamW": "#fda4af",
    }

    for optimizer, optimizer_records in by_optimizer.items():
        optimizer_records.sort(key=lambda r: r.params)
        fig.add_trace(
            go.Scatter(
                x=[r.params for r in optimizer_records],
                y=[r.loss for r in optimizer_records],
                mode="lines+markers",
                name=optimizer,
                marker=dict(
                    size=12,
                    line=dict(width=1.5, color="#ffffff"),
                    color=palette.get(optimizer, "#6366f1"),
                ),
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    "Optimizer: %{customdata[3]}<br>"
                    "Size: %{customdata[1]}<br>"
                    "Params: %{x:,}<br>"
                    "c4/en loss: %{y:.3f}<extra></extra>"
                ),
                customdata=[
                    [r.run_url, r.size_label, r.run_name, r.optimizer]
                    for r in optimizer_records
                ],
            )
        )

    tickvals = [EXPECTED_PARAMS[label] for label in MODEL_ORDER]
    ticktext = MODEL_ORDER

    fig.update_layout(
        title=dict(
            text="Final Eval Loss by Optimizer & Model Size",
            x=0.5,
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis=dict(
            title="Model size (params)",
            tickvals=tickvals,
            ticktext=ticktext,
            type="log",
            showgrid=False,
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

    div_id = "c4-loss-plot"
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "c4_loss_scaling",
                "scale": 2,
            },
        },
        div_id=div_id,
    )

    style_block = """
    <style>
        body {
            margin: 0;
            padding: 24px;
            background: #ffffff;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
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
            if (!plot || !plot.on) {{
                return;
            }}
            plot.on('plotly_click', function(eventData) {{
                if (!eventData || !eventData.points || !eventData.points.length) {{
                    return;
                }}
                var url = eventData.points[0].customdata[0];
                if (url) {{
                    window.open(url, '_blank');
                }}
            }});
        }});
    </script>
    """

    if "</head>" in html:
        html = html.replace("</head>", f"{style_block}</head>")
    else:
        html = style_block + html

    html = html.replace("</body>", f"{click_handler}</body>")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    print(f"✓ Interactive plot written to {output_file}")


def main() -> None:
    try:
        api = wandb.Api()
    except Exception as exc:
        print(f"Warning: could not initialize W&B API ({exc}). Using dummy data.")
        api = None

    records = build_run_records(api)
    available = [r for r in records if r.loss is not None]
    if not available:
        raise SystemExit("No loss values were retrieved; aborting.")

    for record in available:
        print(
            f"{record.optimizer:6s} | {record.size_label:4s} | "
            f"{record.loss:.4f} | {record.run_name}"
        )

    output_path = Path("public/experiments/c4_en_loss_interactive.html")
    create_plot(available, output_path)


if __name__ == "__main__":
    main()

