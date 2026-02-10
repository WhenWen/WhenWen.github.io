#!/usr/bin/env python3
"""
Fetch data from W&B runs and create interactive plots
Second plot: MuonH vs Muon comparison
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
    "no_hybrid_norm": "marin-community/Hyperball/qwen3_1_2b_muonh_4096_lr_0.01_no_hybrid_norm-2128ed",
    "remove_attn_bias": "marin-community/Hyperball/qwen3_1_2b_muon_4096_remove_attn_bias-4ab02c"
}

# Metrics to retrieve
metrics = [
    "params/norm/transformer.layers.9.self_attn.v_proj.weight",
    "eval/paloma/c4_en/loss"
]

print("Fetching data from W&B runs...")

# Fetch data and calculate total tokens
data = {}
total_tokens = None
for run_name, run_path in run_paths.items():
    print(f"Fetching {run_name}...")
    run = api.run(run_path)
    history = run.history(keys=metrics + ["_step"])
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

# Analyze data ranges for better y-axis limits
for idx, metric in enumerate(metrics):
    all_values = []
    for run_name, df in data.items():
        valid_data = df[metric].dropna()
        all_values.extend(valid_data.values)
    
    if all_values:
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        range_val = max_val - min_val
        print(f"{metric}:")
        print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}, Range: {range_val:.6f}")

print("\nCreating plots...")

# Create subplots - horizontal layout (1 row, 2 columns)
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Weight Norm (Layer 9 V Proj)",
        "Evaluation Loss"
    ),
    horizontal_spacing=0.1,
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

colors = {
    "no_hybrid_norm": "#3b82f6",  # blue
    "remove_attn_bias": "#f59e0b"  # amber/orange
}

display_names = {
    "no_hybrid_norm": "MuonH",
    "remove_attn_bias": "Muon"
}

# Add traces for each metric
for idx, metric in enumerate(metrics, start=1):
    for run_name, df in data.items():
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
                showlegend=(idx == 1),  # Only show legend for first subplot
                hovertemplate='<b>%{fullData.name}</b><br>Step: %{x}<br>Value: %{y:.6f}<extra></extra>'
            ),
            row=1, col=idx
        )

# Set appropriate y-axis ranges based on data
# Weight norm - larger tick font
fig.update_yaxes(
    title_text="Norm", 
    row=1, col=1, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13)
)

# Loss - fixed range for better visualization
fig.update_yaxes(
    title_text="Loss", 
    row=1, col=2, 
    gridcolor='rgba(229, 231, 235, 0.5)',
    tickfont=dict(size=13),
    range=[3, 4]
)

# Update x-axes - larger tick font
for col in range(1, 3):
    fig.update_xaxes(
        row=1, col=col, 
        gridcolor='rgba(229, 231, 235, 0.5)',
        tickfont=dict(size=13)
    )

# Update layout - with model size and data tokens title
title_text = "1.2B Model"
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

# Add single centered x-axis label by appending to existing annotations
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
output_file = "wd_blog/public/experiments/wandb_metrics_plot_new.html"
fig.write_html(
    output_file,
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'wandb_metrics_new',
            'height': 500,
            'width': 1400,
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
print(f"✓ Title added: '1.2B Model' with token count")
print(f"✓ Legend updated: 'MuonH' and 'Muon'")
print(f"✓ Loss y-axis range set to [3, 4]")
print(f"✓ Y-axis and X-axis tick labels enlarged (size 13)")
print(f"✓ Legend text enlarged (size 15)")

