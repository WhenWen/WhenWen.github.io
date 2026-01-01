#!/usr/bin/env python3
"""
Fetch data from W&B runs and create interactive plots
8B Model: MuonH Feistel vs PT to Cooldown comparison
"""
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize W&B API
api = wandb.Api()

# Define run paths for 8B models
run_paths = {
    "muonh_feistel": "marin-community/marin/ferry_muonh_qwen3_8b_feistel-515e16",
    "pt_to_cooldown": "marin-community/marin/ferry_qwen3_8b_pt_to_cooldown-1fc2cb"
}

# First, let's explore what metrics are available
print("Fetching run info and exploring available metrics...")

# Get available metrics from first run
for run_name, run_path in run_paths.items():
    print(f"\n--- {run_name} ---")
    run = api.run(run_path)
    
    # Get all logged keys
    history = run.history(samples=1)
    if len(history) > 0:
        all_keys = [k for k in history.columns if not k.startswith('_')]
        print(f"Available metrics ({len(all_keys)}):")
        
        # Filter for interesting metrics
        norm_keys = [k for k in all_keys if 'norm' in k.lower()]
        loss_keys = [k for k in all_keys if 'loss' in k.lower()]
        eval_keys = [k for k in all_keys if 'eval' in k.lower()]
        
        print(f"\n  Norm metrics: {norm_keys[:10]}...")
        print(f"\n  Loss metrics: {loss_keys[:10]}...")
        print(f"\n  Eval metrics: {eval_keys[:10]}...")
    
    # Print config info
    print(f"\n  Config keys: {list(run.config.keys())[:10]}")
    break  # Just check first run

# Try common metrics that might exist
# We'll use similar metrics to the 1.2B script but adapt layer numbers
metrics = [
    "params/norm/transformer.layers.9.self_attn.v_proj.weight",
    "eval/paloma/c4_en/loss"
]

print("\n\nFetching data from W&B runs...")

# Fetch data and calculate total tokens
data = {}
total_tokens = None
for run_name, run_path in run_paths.items():
    print(f"Fetching {run_name}...")
    run = api.run(run_path)
    
    # First get all available keys to find the right ones
    sample_history = run.history(samples=1)
    available_keys = [k for k in sample_history.columns if not k.startswith('_')]
    
    # Find appropriate metrics
    actual_metrics = []
    for m in metrics:
        if m in available_keys:
            actual_metrics.append(m)
        else:
            # Try to find a similar metric
            if 'norm' in m.lower():
                matching = [k for k in available_keys if 'norm' in k.lower() and 'v_proj' in k.lower()]
                if matching:
                    actual_metrics.append(matching[0])
                    print(f"  Using {matching[0]} instead of {m}")
            elif 'loss' in m.lower() or 'eval' in m.lower():
                matching = [k for k in available_keys if 'c4' in k.lower() and 'loss' in k.lower()]
                if not matching:
                    matching = [k for k in available_keys if 'eval' in k.lower() and 'loss' in k.lower()]
                if matching:
                    actual_metrics.append(matching[0])
                    print(f"  Using {matching[0]} instead of {m}")
    
    if actual_metrics:
        history = run.history(keys=actual_metrics + ["_step"])
        data[run_name] = {"df": history, "metrics": actual_metrics}
        print(f"  Retrieved {len(history)} steps with metrics: {actual_metrics}")
    else:
        print(f"  Warning: No matching metrics found!")
        # Just get all history
        history = run.history()
        data[run_name] = {"df": history, "metrics": []}
        print(f"  Retrieved {len(history)} steps (all data)")
    
    # Calculate total tokens from config
    if total_tokens is None:
        try:
            config = run.config
            num_train_steps = config.get('trainer', {}).get('num_train_steps', 0)
            train_batch_size = config.get('trainer', {}).get('train_batch_size', 0)
            seq_len = config.get('model', {}).get('seq_len', 0)
            
            if num_train_steps and train_batch_size and seq_len:
                total_tokens = num_train_steps * train_batch_size * seq_len
                total_tokens_b = int(total_tokens / 1e9)
                print(f"  Config: num_train_steps={num_train_steps}, train_batch_size={train_batch_size}, seq_len={seq_len}")
                print(f"  Total tokens: {total_tokens_b}B")
        except Exception as e:
            print(f"  Could not calculate tokens: {e}")

# Get the actual metrics used (from first run)
first_run_data = list(data.values())[0]
actual_metrics = first_run_data["metrics"]

if not actual_metrics:
    print("\nNo metrics found! Let's see what's available...")
    for run_name, run_data in data.items():
        df = run_data["df"]
        print(f"\n{run_name} columns:")
        for col in df.columns:
            if not col.startswith('_'):
                print(f"  {col}")
    exit(1)

print(f"\nUsing metrics: {actual_metrics}")

print("\nAnalyzing data ranges...")

# Analyze data ranges
for metric in actual_metrics:
    all_values = []
    for run_name, run_data in data.items():
        df = run_data["df"]
        if metric in df.columns:
            valid_data = df[metric].dropna()
            all_values.extend(valid_data.values)
    
    if all_values:
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        range_val = max_val - min_val
        print(f"{metric}:")
        print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}, Range: {range_val:.6f}")

print("\nCreating plots...")

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Weight Norm",
        "Evaluation Loss"
    ),
    horizontal_spacing=0.1,
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

colors = {
    "muonh_feistel": "#3b82f6",  # blue
    "pt_to_cooldown": "#f59e0b"  # amber/orange
}

display_names = {
    "muonh_feistel": "MuonH (Feistel)",
    "pt_to_cooldown": "AdamW"
}

# Add traces for each metric
for idx, metric in enumerate(actual_metrics, start=1):
    for run_name, run_data in data.items():
        df = run_data["df"]
        if metric in df.columns:
            # Filter out NaN values
            valid_data = df[["_step", metric]].dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=valid_data["_step"],
                    y=valid_data[metric],
                    mode='lines',
                    name=display_names[run_name],
                    line=dict(color=colors[run_name], width=2.5),
                    legendgroup=run_name,
                    showlegend=(idx == 1),
                    hovertemplate='<b>%{fullData.name}</b><br>Step: %{x}<br>Value: %{y:.6f}<extra></extra>'
                ),
                row=1, col=idx
            )

# Set y-axis properties
fig.update_yaxes(
    title_text="Norm", 
    row=1, col=1, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13)
)

fig.update_yaxes(
    title_text="Loss", 
    row=1, col=2, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13),
    range=[2.45, 3]
)

# Update x-axes
for col in range(1, 3):
    fig.update_xaxes(
        row=1, col=col, 
        gridcolor='rgba(229, 231, 235, 0.5)',
        tickfont=dict(size=13)
    )

# Update layout
title_text = "8B Model"
if total_tokens is not None:
    title_text += f", {total_tokens_b}B Tokens"

fig.update_layout(
    title={
        'text': title_text,
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
        y=1.05,
        xanchor="center",
        x=0.5,
        font=dict(size=15)
    ),
    plot_bgcolor='rgba(249, 250, 251, 0.5)',
    paper_bgcolor='white',
    margin=dict(t=100, b=80, l=60, r=60)
)

# Add centered x-axis label
fig.add_annotation(
    text="Training Steps",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.13,
    xanchor='center',
    yanchor='top',
    showarrow=False,
    font=dict(size=16, family='Inter, sans-serif')
)

# Save as HTML
output_file = "wandb_metrics_8b.html"
fig.write_html(
    output_file,
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'wandb_metrics_8b',
            'height': 500,
            'width': 1400,
            'scale': 2
        }
    }
)

print(f"\n✓ Interactive plot saved to {output_file}")

# Add custom CSS styling
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

print(f"✓ Styling applied to {output_file}")
print(f"✓ Open it in your browser to view the interactive chart")

