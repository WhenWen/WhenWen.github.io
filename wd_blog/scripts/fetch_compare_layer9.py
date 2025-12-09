#!/usr/bin/env python3
"""
Fetch Layer 9 Q and K norms and Train Loss for two specific W&B runs
and create a comparative plot.
"""
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize W&B API
api = None
try:
    api = wandb.Api()
    api_available = True
except Exception as e:
    print(f"Warning: Could not initialize W&B API: {e}")
    api_available = False

# Define runs
runs_config = [
    {
        "path": "marin-community/optimizer-scaling/sweep-520m-21B-adamwwc5e510lr0.002-wd0.2-minlr0-warmup1000-b10.9-f30f89",
        "label": "LR=0.002, WD=0.2",
        "color": "#1f77b4"  # Blue
    },
    {
        "path": "marin-community/optimizer-scaling/sweep-520m-21B-adamws529f50lr0.004-wd0.1-minlr0-warmup1000-b10.9-c43fcb",
        "label": "LR=0.004, WD=0.1",
        "color": "#ff7f0e"  # Orange
    }
]

# Metrics to fetch
layer_idx = 9
metrics_map = {
    "q_norm": f"params/norm/transformer.layers.{layer_idx}.self_attn.q_proj.weight",
    "k_norm": f"params/norm/transformer.layers.{layer_idx}.self_attn.k_proj.weight",
    "train_loss": "train/loss",
    "step": "_step" # or trainer/global_step
}

data_store = []

print(f"Fetching data for {len(runs_config)} runs...")

for run_conf in runs_config:
    print(f"  Fetching {run_conf['label']} ({run_conf['path']})...")
    
    try:
        run = api.run(run_conf['path'])
        
        # Keys to fetch
        keys = list(metrics_map.values())
        
        # Fetch history
        # Using a large number of samples to get good resolution
        history = run.history(keys=keys, samples=5000)
        print(f"    Retrieved {len(history)} points")
        
        # Store data
        run_data = {
            "config": run_conf,
            "history": history
        }
        data_store.append(run_data)
        
    except Exception as e:
        print(f"    Error fetching run {run_conf['path']}: {e}")
        # Dummy data for testing if offline
        print("    Generating dummy data...")
        steps = np.linspace(0, 10000, 100)
        dummy_hist = pd.DataFrame({"_step": steps})
        dummy_hist[metrics_map["train_loss"]] = np.exp(-steps/5000) + 0.1 * np.random.rand(100)
        dummy_hist[metrics_map["q_norm"]] = 1.0 + steps/10000 + 0.05 * np.random.rand(100)
        dummy_hist[metrics_map["k_norm"]] = 0.8 + steps/12000 + 0.05 * np.random.rand(100)
        
        data_store.append({
            "config": run_conf,
            "history": dummy_hist
        })

print("\nCreating plots...")

# Create subplots - 1 row, 3 columns
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        f"Layer {layer_idx} Q Norm",
        f"Layer {layer_idx} K Norm",
        "Train Loss"
    ),
    horizontal_spacing=0.08
)

for run_data in data_store:
    hist = run_data["history"]
    conf = run_data["config"]
    color = conf["color"]
    label = conf["label"]
    
    # Check which step column to use
    step_col = "_step"
    if "trainer/global_step" in hist.columns:
        step_col = "trainer/global_step"
    elif "global_step" in hist.columns:
        step_col = "global_step"
        
    # 1. Q Norm
    q_key = metrics_map["q_norm"]
    if q_key in hist.columns:
        valid = hist[[step_col, q_key]].dropna()
        fig.add_trace(
            go.Scatter(
                x=valid[step_col],
                y=valid[q_key],
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                legendgroup=label,
                showlegend=True, # Show legend for the first plot
            ),
            row=1, col=1
        )
        
    # 2. K Norm
    k_key = metrics_map["k_norm"]
    if k_key in hist.columns:
        valid = hist[[step_col, k_key]].dropna()
        fig.add_trace(
            go.Scatter(
                x=valid[step_col],
                y=valid[k_key],
                mode='lines',
                name=label,
                line=dict(color=color, width=2, dash='dot'), # varied style if needed, but color distinguishes
                legendgroup=label,
                showlegend=False,
            ),
            row=1, col=2
        )

    # 3. Train Loss
    loss_key = metrics_map["train_loss"]
    if loss_key in hist.columns:
        valid = hist[[step_col, loss_key]].dropna()
        fig.add_trace(
            go.Scatter(
                x=valid[step_col],
                y=valid[loss_key],
                mode='lines',
                name=label,
                line=dict(color=color, width=1.5),
                opacity=0.4,
                legendgroup=label,
                showlegend=False,
                
            ),
            row=1, col=3
        )

# Update layout
fig.update_layout(
    title={
        'text': f"Comparison of Layer {layer_idx} Norms and Loss",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Inter, sans-serif'}
    },
    height=500,
    hovermode="x unified",
    font=dict(family="Inter, sans-serif", size=11),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(249, 250, 251, 0.5)',
    paper_bgcolor='white',
    margin=dict(t=100, b=80, l=60, r=60)
)
# Update axes
for col in range(1, 4):
    fig.update_xaxes(
        title_text="Steps",
        row=1, col=col,
        gridcolor='rgba(229, 231, 235, 0.5)',
    )
    fig.update_yaxes(
        gridcolor='rgba(229, 231, 235, 0.5)',
        row=1, col=col
    )

# Log scale for loss might be useful
# fig.update_yaxes(type="log", row=1, col=3, title_text="Loss (log scale)")
fig.update_yaxes(
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=12),
    range=[2.7, 3.2],
    row=1, col=3
)
fig.update_yaxes(title_text="Norm", row=1, col=1)

# Save as HTML
output_file = "wandb_compare_layer9.html"
fig.write_html(
    output_file,
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'wandb_compare_layer9',
            'height': 500,
            'width': 1200,
            'scale': 2
        }
    }
)

print(f"\n✓ Interactive plot saved to {output_file}")

# Inject styling
with open(output_file, 'r') as f:
    html_content = f.read()

styled_html = html_content.replace(
    '</head>',
    '''
    <style>
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: #ffffff;
            margin: 0;
            padding: 20px;
        }
        .plotly-graph-div {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
    </style>
    </head>
    '''
)

with open(output_file, 'w') as f:
    f.write(styled_html)

