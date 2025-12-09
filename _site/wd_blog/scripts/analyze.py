#!/usr/bin/env python3
"""
Fetch data from W&B runs and create interactive plots
First plot: Weight Decay comparison
"""
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize W&B API
api = wandb.Api()

# Define run paths
run_paths = {
    "No Weight Decay (wd=0)": "marin-community/optimizer-scaling/rerun_baseline-muon-130m-lr4.00e-3-cosine-fda4ea",
    "With Weight Decay (wd=0.1)": "marin-community/optimizer-scaling/rerun_baseline-muon-130m-lr4.00e-3-cosine-wd0.1-370ba9"
}

# Metrics to retrieve
fetch_metrics = [
    "params/norm/transformer.layers.9.self_attn.v_proj.weight",
    "eval/paloma/c4_en/loss",
    "optim/learning_rate"
]

print("Fetching data from W&B runs...")

# Fetch data and calculate total tokens
data = {}
total_tokens = None
for run_name, run_path in run_paths.items():
    print(f"Fetching {run_name}...")
    run = api.run(run_path)
    history = run.history(keys=fetch_metrics + ["_step"])
    
    # Calculate approximated effective learning rate
    # Formula: learning_rate * sqrt(512) / params/norm/transformer.layers.9.self_attn.v_proj.weight
    weight_norm_key = "params/norm/transformer.layers.9.self_attn.v_proj.weight"
    lr_key = "optim/learning_rate"
    eff_lr_key = "approx_effective_lr"
    
    if weight_norm_key in history.columns and lr_key in history.columns:
        history[eff_lr_key] = history[lr_key] * np.sqrt(512) / history[weight_norm_key]
    
    # Calculate theoretical effective learning rate
    # Formula: sqrt(learning_rate * 2 * weight_decay)
    theo_eff_lr_key = "theoretical_effective_lr"
    theo_norm_key = "theoretical_norm"
    wd = 0.0
    if "wd=0.1" in run_name:
        wd = 0.1
    
    if lr_key in history.columns:
        alpha = 1 - history[lr_key] * wd
        beta = 0.98
        history[theo_eff_lr_key] = np.sqrt((1 - alpha**2) * (1 - alpha * beta) / (1 + alpha * beta))
        eta = history[lr_key]
        if wd != 0:
            alpha = 1.0 - eta * wd                       # 1 - ηλ
            num_inside = eta * (1.0 + beta * alpha)    # η [1 + β(1-ηλ)]
            den_inside = wd * (2.0 - eta * wd) * (1.0 - beta * alpha)
            print(num_inside, den_inside)
            ratio_inside = num_inside / den_inside     # no division by η here
            W_norm = np.sqrt(512) * np.sqrt(ratio_inside)
            history[theo_norm_key] = W_norm

    data[run_name] = history
    print(f"  Retrieved {len(history)} steps")
    
    # Calculate total tokens from config (only need to do once)
    if total_tokens is None:
        try:
            num_train_steps = run.config.get('trainer', {}).get('num_train_steps', 0)
            train_batch_size = run.config.get('trainer', {}).get('train_batch_size', 0)
            seq_len = run.config.get('model', {}).get('seq_len', 0)
            
            if num_train_steps and train_batch_size and seq_len:
                total_tokens = num_train_steps * train_batch_size * seq_len
                total_tokens_b = int(total_tokens / 1e9)  # Convert to billions
                print(f"  Config: num_train_steps={num_train_steps}, train_batch_size={train_batch_size}, seq_len={seq_len}")
                print(f"  Total tokens: {total_tokens_b}B")
        except Exception as e:
            print(f"  Could not calculate tokens: {e}")

print("\nAnalyzing data ranges...")

# Metrics to plot
plot_metrics = [
    "params/norm/transformer.layers.9.self_attn.v_proj.weight",
    "optim/learning_rate",
    "eval/paloma/c4_en/loss",
    "approx_effective_lr",
    "theoretical_effective_lr",
    "theoretical_norm"
]

# Analyze data ranges for better y-axis limits
for idx, metric in enumerate(plot_metrics):
    all_values = []
    for run_name, df in data.items():
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

# Create subplots - 2 rows, 2 columns
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Weight Norm",
        "Learning Rate",
        "Evaluation Loss",
        r"$\eta \|u_t\|_F / \|W_t\|_F$"
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.12,
    specs=[
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}]
    ]
)

colors = {
    "No Weight Decay (wd=0)": "#3b82f6",  # blue
    "With Weight Decay (wd=0.1)": "#f59e0b"  # amber/orange
}

# Plot configuration mapping
# metric -> (row, col)
plot_positions = {
    "params/norm/transformer.layers.9.self_attn.v_proj.weight": (1, 1),
    "optim/learning_rate": (1, 2),
    "eval/paloma/c4_en/loss": (2, 1),
    "approx_effective_lr": (2, 2),
    "theoretical_effective_lr": (2, 2),
    "theoretical_norm": (1, 1)
}

# Add traces for each metric
for metric in plot_metrics:
    row, col = plot_positions[metric]
    for run_name, df in data.items():
        if metric not in df.columns:
            continue
            
        # Filter out NaN values
        valid_data = df[["_step", metric]].dropna()
        
        # Customize style for theoretical metric
        line_style = dict(color=colors[run_name], width=2.5)
        hover_name = run_name
        
        if metric == "theoretical_effective_lr":
            line_style['dash'] = 'dash'
            hover_name = f"{run_name} (Theory)"
            if run_name == "No Weight Decay (wd=0)":
                continue
        elif metric == "theoretical_norm":
            line_style['dash'] = 'dash'
            hover_name = f"{run_name} (Theory)"
            if run_name == "No Weight Decay (wd=0)":
                continue
        elif metric == "approx_effective_lr":
            hover_name = f"{run_name} (Measured)"
        
        showlegend = (row == 1 and col == 1)
        if metric in {"theoretical_effective_lr", "theoretical_norm"}:
            showlegend = False

        fig.add_trace(
            go.Scatter(
                x=valid_data["_step"],
                y=valid_data[metric],
                mode='lines',
                name=run_name,
                line=line_style,
                legendgroup=run_name,
                showlegend=showlegend,  # Only show legend for primary traces
                hovertemplate=f'<b>{hover_name}</b><br>Step: %{{x}}<br>Value: %{{y:.6f}}<extra></extra>'
            ),
            row=row, col=col
        )

# Set appropriate y-axis ranges based on data

# Parameter norm (1,1)
fig.update_yaxes(
    title_text="Norm", 
    row=1, col=1, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13)
)

# Learning Rate (1,2)
fig.update_yaxes(
    title_text="LR", 
    row=1, col=2, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13)
)

# Loss (2,1) - fixed range
fig.update_yaxes(
    title_text="Loss", 
    row=2, col=1, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    range=[3.18, 4.0],
    tickfont=dict(size=13)
)

# Approx Effective LR (2,2)
fig.update_yaxes(
    title_text="Eff. Step Size", 
    row=2, col=2, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13)
)

# Update x-axes - larger tick font
for row in range(1, 3):
    for col in range(1, 3):
        fig.update_xaxes(
            row=row, col=col, 
            gridcolor='rgba(229, 231, 235, 0.5)',
            tickfont=dict(size=13)
        )

# Update layout - with model size and data tokens title
title_text = "130M Model"
if total_tokens is not None:
    title_text += f", {total_tokens_b}B Tokens"

fig.update_layout(
    title={
        'text': title_text,
        'y': 0.93,  # Position title high
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Inter, sans-serif'}
    },
    height=800,  # Increased height for 2x2 grid
    hovermode="x unified",
    font=dict(family="Inter, sans-serif", size=11),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.08,  # Slightly higher to clear subplots, but gap created by margin
        xanchor="center",
        x=0.5,
        font=dict(size=15)
    ),
    plot_bgcolor='rgba(249, 250, 251, 0.5)',
    paper_bgcolor='white',
    margin=dict(t=160, b=80, l=60, r=60)  # Increased top margin for more gap
)

# Add single centered x-axis label
fig.add_annotation(
    text="Training Steps",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.08,  # Adjusted for 2x2 grid
    xanchor='center',
    yanchor='top',
    showarrow=False,
    font=dict(size=16, family='Inter, sans-serif')
)

# Save as HTML
output_file = "wandb_metrics_plot_analyze.html"
fig.write_html(
    output_file,
    include_mathjax='cdn',
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'wandb_metrics',
            'height': 800,
            'width': 1000,
            'scale': 2
        }
    }
)

print(f"\n✓ Interactive plot saved to {output_file}")
print(f"  Open it in your browser to view the interactive chart")

# Also create a standalone version with some styling
with open(output_file, 'r') as f:
    html_content = f.read()

# Inject some custom CSS
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

print(f"\n✓ Styling applied to {output_file}")
print(f"✓ Title added: '130M Model' with token count")
print(f"✓ Loss y-axis range set to [3.18, 4.0]")
print(f"✓ New metrics added: Approx. Effective LR and Theoretical Effective LR")
print(f"✓ Layout updated to 2x2 grid with improved header spacing")
