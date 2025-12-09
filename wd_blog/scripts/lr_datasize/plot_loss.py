from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Read the data
df = pd.read_csv('./wandb_hyperball_runs.csv')

# Convert num_tokens to billions for better readability
df['data_size_B'] = df['num_tokens'] / 1e9

# Get unique optimizers and data sizes
optimizers = sorted(df['optimizer_name'].unique())
data_sizes = sorted(df['data_size_B'].unique())

WANDB_PROJECT_URL = "https://wandb.ai/marin-community/LR_datasize/runs/{run_id}"

def build_run_url(run_id: str) -> str:
    if isinstance(run_id, str) and run_id:
        return WANDB_PROJECT_URL.format(run_id=run_id)
    return ""

# Pre-compute the best run (min loss) per optimizer/data size combo
best_run_lookup = {}
for (optimizer, data_size), group in df.groupby(['optimizer_name', 'data_size_B']):
    group = group.dropna(subset=['loss'])
    if group.empty:
        continue
    best_idx = group['loss'].idxmin()
    best_row = group.loc[best_idx]
    best_run_lookup[(optimizer, data_size)] = {
        'loss': best_row['loss'],
        'run_id': best_row['run_id'],
        'run_name': best_row['run_name'],
        'run_url': build_run_url(best_row['run_id']),
    }

print(f"Optimizers: {optimizers}")
print(f"Data sizes (B): {data_sizes}")
print(f"Total runs: {len(df)}")

# ==============================================================================
# Plot 1: How different optimizer's loss scales with data
# ==============================================================================
print("\nGenerating Plot 1: Loss vs Data Size for different optimizers...")

fig1, ax1 = plt.subplots(figsize=(10, 6))

for optimizer in optimizers:
    # Get the best (minimum) loss for each data size
    best_losses = []
    for data_size in data_sizes:
        record = best_run_lookup.get((optimizer, data_size))
        if record:
            best_losses.append(record['loss'])
        else:
            best_losses.append(np.nan)
    
    ax1.plot(data_sizes, best_losses, marker='o', linewidth=2, markersize=8, label=optimizer)

ax1.set_xlabel('Data Size (Billions of Tokens)', fontsize=12)
ax1.set_ylabel('Best Loss', fontsize=12)
ax1.set_title('Best Loss vs Data Size for Different Optimizers', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Add minor gridlines
ax1.grid(True, which='minor', alpha=0.2)

plt.tight_layout()
plt.savefig('./plot1_loss_vs_datasize.png', dpi=300, bbox_inches='tight')
print("Saved: plot1_loss_vs_datasize.png")
plt.close()

# ==============================================================================
# Plot 1b: Scaling Law Fit - Loss vs Data Size with Power Law + Constant
# ==============================================================================
print("\nGenerating Plot 1b: Scaling Law Fit (Power Law + Constant)...")

def scaling_law(N, L_inf, A, alpha):
    """
    Neural network scaling law: Loss(N) = L_inf + A * N^(-alpha)
    
    Parameters:
    - N: data size (in billions of tokens)
    - L_inf: irreducible loss (loss with infinite data)
    - A: coefficient
    - alpha: power law exponent
    """
    return L_inf + A * N**(-alpha)

fig1b, ax1b = plt.subplots(figsize=(12, 7))

# Store fitted parameters for each optimizer
scaling_law_params = {}
colors_opt = plt.cm.tab10(np.arange(len(optimizers)))

for opt_idx, optimizer in enumerate(optimizers):
    # Get the best (minimum) loss for each data size
    best_losses = []
    valid_data_sizes = []
    run_ids = []
    run_names = []
    run_urls = []
    for data_size in data_sizes:
        record = best_run_lookup.get((optimizer, data_size))
        if record:
            best_losses.append(record['loss'])
            valid_data_sizes.append(data_size)
            run_ids.append(record['run_id'])
            run_names.append(record['run_name'])
            run_urls.append(record['run_url'])
    
    if len(valid_data_sizes) >= 3:  # Need at least 3 points to fit
        try:
            # Initial guess for parameters
            # L_inf should be less than min loss, A positive, alpha between 0 and 1
            min_loss = min(best_losses)
            max_loss = max(best_losses)
            initial_L_inf = min_loss * 0.9  # Slightly below min loss
            initial_A = (max_loss - min_loss) * (max(valid_data_sizes) ** 0.3)
            initial_alpha = 0.3
            
            # Fit the scaling law with bounds
            popt, pcov = curve_fit(
                scaling_law, 
                valid_data_sizes, 
                best_losses,
                p0=[initial_L_inf, initial_A, initial_alpha],
                bounds=([0, 0, 0], [min_loss, np.inf, 2.0]),  # Reasonable bounds
                maxfev=10000
            )
            
            L_inf, A, alpha = popt
            perr = np.sqrt(np.diag(pcov))  # Standard errors
            
            # Calculate R-squared
            residuals = np.array(best_losses) - scaling_law(np.array(valid_data_sizes), L_inf, A, alpha)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((np.array(best_losses) - np.mean(best_losses))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Store parameters
            scaling_law_params[optimizer] = {
                'L_inf': L_inf,
                'A': A,
                'alpha': alpha,
                'L_inf_err': perr[0],
                'A_err': perr[1],
                'alpha_err': perr[2],
                'r_squared': r_squared,
                'data_sizes': valid_data_sizes,
                'losses': best_losses,
                'run_ids': run_ids,
                'run_names': run_names,
                'run_urls': run_urls,
            }
            
            # Plot actual data points
            ax1b.scatter(valid_data_sizes, best_losses, s=150, color=colors_opt[opt_idx], 
                        marker='o', label=f'{optimizer} (data)', zorder=3, alpha=0.8,
                        edgecolors='black', linewidth=1.5)
            
            # Create smooth curve for the fit
            N_range = np.logspace(np.log10(min(valid_data_sizes)), 
                                 np.log10(max(valid_data_sizes) * 3), 200)
            fitted_losses = scaling_law(N_range, L_inf, A, alpha)
            
            # Plot fitted curve
            ax1b.plot(N_range, fitted_losses, linewidth=3, color=colors_opt[opt_idx],
                     label=f'{optimizer} fit: L∞={L_inf:.4f}', linestyle='-', alpha=0.7)
            
            # Add horizontal line for L_inf (asymptote)
            ax1b.axhline(y=L_inf, color=colors_opt[opt_idx], linestyle='--', 
                        linewidth=1.5, alpha=0.4)
            
            # Annotate L_inf value
            ax1b.text(max(data_sizes) * 1.1, L_inf, f'L∞={L_inf:.4f}', 
                     color=colors_opt[opt_idx], fontsize=10, fontweight='bold',
                     verticalalignment='center')
            
            print(f"\n{optimizer}:")
            print(f"  L_inf (loss with infinite data) = {L_inf:.6f} ± {perr[0]:.6f}")
            print(f"  A (coefficient) = {A:.6f} ± {perr[1]:.6f}")
            print(f"  alpha (power law exponent) = {alpha:.4f} ± {perr[2]:.4f}")
            print(f"  R² = {r_squared:.4f}")
            print(f"  Scaling law: Loss(N) = {L_inf:.6f} + {A:.6f} * N^(-{alpha:.4f})")
            
            # Predict losses at larger data sizes
            prediction_sizes = [100, 500, 1000]  # Billions of tokens
            print(f"  Predictions:")
            for pred_size in prediction_sizes:
                pred_loss = scaling_law(pred_size, L_inf, A, alpha)
                print(f"    N = {pred_size:>4}B tokens: Loss = {pred_loss:.6f}")
            
        except Exception as e:
            print(f"  Warning: Could not fit scaling law for {optimizer}: {e}")
    else:
        # Still plot the points even if we can't fit
        ax1b.scatter(valid_data_sizes, best_losses, s=150, color=colors_opt[opt_idx], 
                    marker='o', label=f'{optimizer} (data)', zorder=3, alpha=0.8,
                    edgecolors='black', linewidth=1.5)
        print(f"  Warning: Not enough data points to fit scaling law for {optimizer}")

ax1b.set_xlabel('Data Size (Billions of Tokens)', fontsize=14, fontweight='bold')
ax1b.set_ylabel('Best Loss', fontsize=14, fontweight='bold')
ax1b.set_title('Scaling Law: Loss vs Data Size\nLoss(N) = L∞ + A·N^(-α)', 
              fontsize=16, fontweight='bold')
ax1b.legend(fontsize=10, loc='best', framealpha=0.95, ncol=1)
ax1b.grid(True, alpha=0.3)
ax1b.set_xscale('log')
ax1b.grid(True, which='minor', alpha=0.2)

# Set y-axis to start slightly below the minimum L_inf
if scaling_law_params:
    min_L_inf = min(p['L_inf'] for p in scaling_law_params.values())
    max_loss_all = max(max(p['losses']) for p in scaling_law_params.values())
    ax1b.set_ylim([min_L_inf * 0.98, max_loss_all * 1.02])

plt.tight_layout()
plt.savefig('./plot1b_scaling_law_fit.png', dpi=300, bbox_inches='tight')
print("\nSaved: plot1b_scaling_law_fit.png")
plt.close()

# ==============================================================================
# Plot 1b Interactive (Plotly)
# ==============================================================================
interactive_output_path = Path("public/experiments/plot1b_scaling_law_fit_interactive.html")

if go is None:
    print("Plotly is not installed; skipping interactive Plot 1b export.")
elif not scaling_law_params:
    print("No scaling law fits available; skipping interactive Plot 1b export.")
else:
    print("Generating interactive Plot 1b (Plotly)...")
    plotly_palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#f97316",
        "#0ea5e9",
        "#9333ea",
        "#0891b2",
        "#f59e0b",
    ]

    fig = go.Figure()
    plotted_any = False

    for opt_idx, optimizer in enumerate(optimizers):
        params = scaling_law_params.get(optimizer)
        if not params or not params['data_sizes']:
            continue

        plotted_any = True
        color = plotly_palette[opt_idx % len(plotly_palette)]
        data_sizes_plot = params['data_sizes']
        losses_plot = params['losses']
        L_inf = params['L_inf']
        alpha_val = params['alpha']
        run_urls_plot = params.get('run_urls', [""] * len(data_sizes_plot))
        run_names_plot = params.get('run_names', [""] * len(data_sizes_plot))
        if len(run_urls_plot) < len(data_sizes_plot):
            run_urls_plot = run_urls_plot + [""] * (len(data_sizes_plot) - len(run_urls_plot))
        if len(run_names_plot) < len(data_sizes_plot):
            run_names_plot = run_names_plot + [""] * (len(data_sizes_plot) - len(run_names_plot))

        fig.add_trace(
            go.Scatter(
                x=data_sizes_plot,
                y=losses_plot,
                mode="markers",
                name=f"{optimizer} data",
                legendgroup=optimizer,
                marker=dict(size=12, color=color, line=dict(color="#111827", width=1.5)),
                customdata=[
                    [optimizer, L_inf, alpha_val, run_url, run_name]
                    for run_url, run_name in zip(run_urls_plot, run_names_plot)
                ],
                hovertemplate=(
                    "<b>%{customdata[4]}</b><br>"
                    "Optimizer: %{customdata[0]}<br>"
                    "Data size: %{x:.2f}B tokens<br>"
                    "Best loss: %{y:.4f}<br>"
                    "L∞: %{customdata[1]:.4f}<br>"
                    "α: %{customdata[2]:.3f}<br>"
                    "Click to open run<extra></extra>"
                ),
            )
        )

        min_x = min(data_sizes_plot)
        max_x = max(data_sizes_plot) * 3
        if max_x <= min_x:
            max_x = min_x * 1.5
        N_range = np.logspace(np.log10(min_x), np.log10(max_x), 200)
        fitted_curve = scaling_law(N_range, L_inf, params['A'], alpha_val)

        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=fitted_curve,
                mode="lines",
                name=f"{optimizer} fit",
                legendgroup=optimizer,
                line=dict(color=color, width=3),
                hovertemplate=(
                    f"<b>{optimizer} fit</b><br>"
                    "Data size: %{x:.2f}B tokens<br>"
                    "Predicted loss: %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_shape(
            type="line",
            x0=min_x,
            x1=max_x,
            y0=L_inf,
            y1=L_inf,
            line=dict(color=color, dash="dash"),
            opacity=0.6,
        )

        fig.add_annotation(
            x=max_x,
            y=L_inf,
            text=f"{optimizer} L∞={L_inf:.4f}",
            showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left",
            bgcolor="rgba(255,255,255,0.7)",
        )

    if plotted_any:
        major_tick_vals = [1, 10, 100]
        fig.update_layout(
            title=dict(
                text="Scaling Law: Loss vs Data Size for Muon and MuonH",
                x=0.5,
                font=dict(size=20, family="Inter, sans-serif"),
                y=0.94,
            ),
            xaxis=dict(
                title="Data Size (Billions of Tokens)",
                type="log",
                showgrid=True,
                gridcolor="#e5e7eb",
                tickmode="array",
                tickvals=major_tick_vals,
                ticktext=[str(v) for v in major_tick_vals],
            ),
            yaxis=dict(
                title="Best Loss",
                showgrid=True,
                gridcolor="#e5e7eb",
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            hovermode="closest",
            margin=dict(l=60, r=40, t=80, b=110),
            plot_bgcolor="rgba(249, 250, 251, 0.6)",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", size=13),
        )

        div_id = "plot1b-scaling-law"
        html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=True,
            div_id=div_id,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "plot1b_scaling_law_fit",
                    "scale": 2,
                },
            },
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
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
            }
        </style>
        """

        if "</head>" in html:
            html = html.replace("</head>", f"{style_block}</head>")
        else:
            html = style_block + html

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
                    var point = eventData.points[0];
                    if (!point.customdata || point.customdata.length < 4) {{
                        return;
                    }}
                    var url = point.customdata[3];
                    if (url) {{
                        window.open(url, '_blank');
                    }}
                }});
            }});
        </script>
        """

        if "</body>" in html:
            html = html.replace("</body>", f"{click_handler}</body>")
        else:
            html = html + click_handler

        interactive_output_path.parent.mkdir(parents=True, exist_ok=True)
        interactive_output_path.write_text(html, encoding="utf-8")
        print(f"Saved interactive Plot 1b: {interactive_output_path}")
    else:
        print("No valid scaling law fits to render interactively.")

# ==============================================================================
# Plot 2: Loss vs Learning Rate for each optimizer (different lines per data size)
# with Quadratic Fits
# ==============================================================================
print("\nGenerating Plot 2: Loss vs Learning Rate for each optimizer...")

# Define quadratic function in log space
def quadratic_log(log_lr, a, b, c):
    """Quadratic function: loss = a*(log(lr))^2 + b*log(lr) + c"""
    return a * log_lr**2 + b * log_lr + c

# Store fitted optimal learning rates for later analysis
fitted_optimal_lrs = {opt: {'data_sizes': [], 'optimal_lrs': [], 'r_squared': []} 
                      for opt in optimizers}

# Create a colormap for different data sizes
colors = plt.cm.viridis(np.linspace(0, 0.95, len(data_sizes)))

fig2, axes = plt.subplots(1, len(optimizers), figsize=(10 * len(optimizers), 6))

# Handle case where we might have only one optimizer
if len(optimizers) == 1:
    axes = [axes]

for j, optimizer in enumerate(optimizers):
    ax = axes[j]
    
    for i, data_size in enumerate(data_sizes):
        # Filter data for this specific combination
        mask = (df['optimizer_name'] == optimizer) & (df['data_size_B'] == data_size)
        subset = df[mask].sort_values('learning_rate')
        
        if len(subset) > 0:
            # Plot the actual data points
            ax.plot(subset['learning_rate'], subset['loss'], 
                   marker='o', linewidth=2.5, markersize=8, 
                   color=colors[i], label=f'{data_size:.0f}B tokens', alpha=0.8)
            
            # Mark the best (minimum) loss with a star
            best_idx = subset['loss'].idxmin()
            best_lr = subset.loc[best_idx, 'learning_rate']
            best_loss = subset.loc[best_idx, 'loss']
            ax.scatter([best_lr], [best_loss], s=300, color=colors[i], marker='*', 
                      zorder=5, edgecolors='darkred', linewidth=2)
            
            # Fit quadratic in log space (if we have enough points)
            if len(subset) >= 3:
                try:
                    log_lrs = np.log(subset['learning_rate'].values)
                    losses = subset['loss'].values
                    
                    # Fit quadratic
                    popt, _ = curve_fit(quadratic_log, log_lrs, losses)
                    a, b, c = popt
                    
                    # Calculate R-squared
                    residuals = losses - quadratic_log(log_lrs, a, b, c)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((losses - np.mean(losses))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Find optimal learning rate from the fitted quadratic
                    # Vertex of parabola: -b/(2a)
                    if a > 0:  # Only if parabola opens upward
                        optimal_log_lr = -b / (2 * a)
                        optimal_lr = np.exp(optimal_log_lr)
                        
                        # Store for later plotting
                        fitted_optimal_lrs[optimizer]['data_sizes'].append(data_size)
                        fitted_optimal_lrs[optimizer]['optimal_lrs'].append(optimal_lr)
                        fitted_optimal_lrs[optimizer]['r_squared'].append(r_squared)
                        
                        # Create smooth curve for plotting
                        log_lr_range = np.linspace(log_lrs.min(), log_lrs.max(), 100)
                        lr_range = np.exp(log_lr_range)
                        fitted_losses = quadratic_log(log_lr_range, a, b, c)
                        
                        # Plot the fitted curve (dashed line)
                        ax.plot(lr_range, fitted_losses, '--', linewidth=2, 
                               color=colors[i], alpha=0.5)
                        
                        # Mark the fitted optimal LR with a diamond
                        fitted_optimal_loss = quadratic_log(optimal_log_lr, a, b, c)
                        ax.scatter([optimal_lr], [fitted_optimal_loss], s=200, 
                                 color=colors[i], marker='D', 
                                 zorder=4, edgecolors='black', linewidth=1.5, alpha=0.7)
                        
                except Exception as e:
                    print(f"  Warning: Could not fit quadratic for {optimizer}, {data_size}B: {e}")
    
    ax.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{optimizer.upper()}\n(★=best, ◆=fitted opt)', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('./plot2_loss_vs_lr_by_optimizer.png', dpi=300, bbox_inches='tight')
print("Saved: plot2_loss_vs_lr_by_optimizer.png")
plt.close()

# ==============================================================================
# Plot 2b: Fitted Optimal Learning Rate Scaling vs Data Size
# ==============================================================================
print("\nGenerating Plot 2b: Fitted Optimal Learning Rate Scaling vs Data Size...")

fig2b, ax2b = plt.subplots(figsize=(10, 6))

for optimizer in optimizers:
    if len(fitted_optimal_lrs[optimizer]['data_sizes']) > 0:
        data_sizes_opt = fitted_optimal_lrs[optimizer]['data_sizes']
        optimal_lrs = fitted_optimal_lrs[optimizer]['optimal_lrs']
        r_squared = fitted_optimal_lrs[optimizer]['r_squared']
        
        # Plot the fitted optimal learning rates
        ax2b.plot(data_sizes_opt, optimal_lrs, marker='D', linewidth=2.5, 
                 markersize=10, label=f'{optimizer} (fitted)', alpha=0.8)
        
        # Try to fit a power law: optimal_lr = k * data_size^alpha
        if len(data_sizes_opt) >= 2:
            try:
                # Fit in log-log space
                log_data_sizes = np.log(data_sizes_opt)
                log_optimal_lrs = np.log(optimal_lrs)
                coeffs = np.polyfit(log_data_sizes, log_optimal_lrs, 1)
                alpha = coeffs[0]
                log_k = coeffs[1]
                k = np.exp(log_k)
                
                # Create smooth curve
                data_size_range = np.logspace(np.log10(min(data_sizes_opt)), 
                                             np.log10(max(data_sizes_opt)), 100)
                fitted_curve = k * data_size_range**alpha
                
                # Plot the power law fit
                ax2b.plot(data_size_range, fitted_curve, '--', linewidth=2, 
                         alpha=0.5, label=f'{optimizer}: LR ∝ N^{{{alpha:.3f}}}')
                
                print(f"  {optimizer}: optimal_lr = {k:.6f} * data_size^{alpha:.4f}")
                
            except Exception as e:
                print(f"  Warning: Could not fit power law for {optimizer}: {e}")

# Also plot the empirical best LRs for comparison
empirical_power_law_params = {}
for optimizer in optimizers:
    best_lrs = []
    valid_data_sizes = []
    for data_size in data_sizes:
        mask = (df['optimizer_name'] == optimizer) & (df['data_size_B'] == data_size)
        if mask.any():
            subset = df[mask]
            best_idx = subset['loss'].idxmin()
            best_lr = subset.loc[best_idx, 'learning_rate']
            best_lrs.append(best_lr)
            valid_data_sizes.append(data_size)
    
    if len(valid_data_sizes) > 0:
        ax2b.plot(valid_data_sizes, best_lrs, marker='*', linewidth=2, 
                 markersize=12, label=f'{optimizer} (empirical best)', 
                 linestyle=':', alpha=0.6)
        
        # Fit a power law on empirical best LRs: optimal_lr = k * data_size^alpha
        if len(valid_data_sizes) >= 2:
            try:
                # Fit in log-log space
                log_data_sizes_emp = np.log(valid_data_sizes)
                log_best_lrs = np.log(best_lrs)
                coeffs_emp = np.polyfit(log_data_sizes_emp, log_best_lrs, 1)
                alpha_emp = coeffs_emp[0]
                log_k_emp = coeffs_emp[1]
                k_emp = np.exp(log_k_emp)
                
                # Calculate R-squared for the empirical power law fit
                fitted_log_lrs_emp = np.polyval(coeffs_emp, log_data_sizes_emp)
                ss_res_emp = np.sum((log_best_lrs - fitted_log_lrs_emp)**2)
                ss_tot_emp = np.sum((log_best_lrs - np.mean(log_best_lrs))**2)
                r2_emp = 1 - (ss_res_emp / ss_tot_emp) if ss_tot_emp > 0 else 0
                
                # Store parameters
                empirical_power_law_params[optimizer] = {
                    'k': k_emp, 
                    'alpha': alpha_emp, 
                    'r_squared': r2_emp,
                    'data_sizes': valid_data_sizes,
                    'best_lrs': best_lrs
                }
                
                # Create smooth curve
                data_size_range_emp = np.logspace(np.log10(min(valid_data_sizes)), 
                                             np.log10(max(valid_data_sizes) * 3), 100)
                fitted_curve_emp = k_emp * data_size_range_emp**alpha_emp
                
                # Plot the power law fit for empirical best LRs
                ax2b.plot(data_size_range_emp, fitted_curve_emp, '-.', linewidth=2.5, 
                         alpha=0.7, label=f'{optimizer} empirical fit: LR ∝ N^{{{alpha_emp:.3f}}} (R²={r2_emp:.3f})')
                
                print(f"  {optimizer} (empirical): optimal_lr = {k_emp:.6f} * data_size^{alpha_emp:.4f} (R² = {r2_emp:.4f})")
                
            except Exception as e:
                print(f"  Warning: Could not fit power law for empirical {optimizer}: {e}")

ax2b.set_xlabel('Data Size (Billions of Tokens)', fontsize=12)
ax2b.set_ylabel('Optimal Learning Rate', fontsize=12)
ax2b.set_title('Fitted Optimal Learning Rate Scaling vs Data Size\n(Quadratic Fit on Loss vs LR Curves)', 
              fontsize=14, fontweight='bold')
ax2b.legend(fontsize=10, loc='best', framealpha=0.9)
ax2b.grid(True, alpha=0.3)
ax2b.set_xscale('log')
ax2b.set_yscale('log')
ax2b.grid(True, which='minor', alpha=0.2)

plt.tight_layout()
plt.savefig('./plot2b_fitted_optimal_lr_scaling.png', dpi=300, bbox_inches='tight')
print("Saved: plot2b_fitted_optimal_lr_scaling.png")
plt.close()

# ==============================================================================
# Plot 3: Best Learning Rate vs Data Size for different optimizers
# ==============================================================================
print("\nGenerating Plot 3: Best Learning Rate vs Data Size for different optimizers...")

fig3, ax3 = plt.subplots(figsize=(10, 6))

for optimizer in optimizers:
    best_lrs = []
    for data_size in data_sizes:
        mask = (df['optimizer_name'] == optimizer) & (df['data_size_B'] == data_size)
        if mask.any():
            subset = df[mask]
            best_idx = subset['loss'].idxmin()
            best_lr = subset.loc[best_idx, 'learning_rate']
            best_lrs.append(best_lr)
        else:
            best_lrs.append(np.nan)
    
    ax3.plot(data_sizes, best_lrs, marker='o', linewidth=2, markersize=8, label=optimizer)

ax3.set_xlabel('Data Size (Billions of Tokens)', fontsize=12)
ax3.set_ylabel('Best Learning Rate', fontsize=12)
ax3.set_title('Best Learning Rate vs Data Size for Different Optimizers', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')
ax3.set_yscale('log')

# Add minor gridlines
ax3.grid(True, which='minor', alpha=0.2)

plt.tight_layout()
plt.savefig('./plot3_best_lr_vs_datasize.png', dpi=300, bbox_inches='tight')
print("Saved: plot3_best_lr_vs_datasize.png")
plt.close()

# ==============================================================================
# Summary Statistics
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for optimizer in optimizers:
    print(f"\n{optimizer.upper()}:")
    print("-" * 40)
    for i, data_size in enumerate(data_sizes):
        mask = (df['optimizer_name'] == optimizer) & (df['data_size_B'] == data_size)
        if mask.any():
            subset = df[mask]
            best_idx = subset['loss'].idxmin()
            best_lr = subset.loc[best_idx, 'learning_rate']
            best_loss = subset.loc[best_idx, 'loss']
            
            # Find fitted optimal LR if available
            fitted_info = ""
            if data_size in fitted_optimal_lrs[optimizer]['data_sizes']:
                idx = fitted_optimal_lrs[optimizer]['data_sizes'].index(data_size)
                fitted_lr = fitted_optimal_lrs[optimizer]['optimal_lrs'][idx]
                r2 = fitted_optimal_lrs[optimizer]['r_squared'][idx]
                fitted_info = f", Fitted LR = {fitted_lr:.6f} (R²={r2:.3f})"
            
            print(f"  {data_size:>5.0f}B tokens: Best LR = {best_lr:.6f}, Best Loss = {best_loss:.4f}{fitted_info}")

print("\n" + "="*80)
print("QUADRATIC FIT SCALING ANALYSIS")
print("="*80)
print("Power law fits: optimal_lr = k * data_size^alpha")
print("(Based on fitted optimal LRs from quadratic fits)")
print("-" * 80)

# Store power law parameters for prediction
lr_power_law_params = {}

for optimizer in optimizers:
    if len(fitted_optimal_lrs[optimizer]['data_sizes']) >= 2:
        data_sizes_opt = fitted_optimal_lrs[optimizer]['data_sizes']
        optimal_lrs = fitted_optimal_lrs[optimizer]['optimal_lrs']
        
        # Fit power law
        log_data_sizes = np.log(data_sizes_opt)
        log_optimal_lrs = np.log(optimal_lrs)
        coeffs = np.polyfit(log_data_sizes, log_optimal_lrs, 1)
        alpha = coeffs[0]
        log_k = coeffs[1]
        k = np.exp(log_k)
        
        # Calculate R-squared for the power law fit
        fitted_log_lrs = np.polyval(coeffs, log_data_sizes)
        ss_res = np.sum((log_optimal_lrs - fitted_log_lrs)**2)
        ss_tot = np.sum((log_optimal_lrs - np.mean(log_optimal_lrs))**2)
        r2_power = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Store parameters
        lr_power_law_params[optimizer] = {'k': k, 'alpha': alpha, 'r_squared': r2_power}
        
        print(f"{optimizer:>15}: optimal_lr = {k:.6f} * N^{alpha:.4f}  (R² = {r2_power:.4f})")

print("\n" + "="*80)
print("EMPIRICAL BEST LEARNING RATE SCALING ANALYSIS")
print("="*80)
print("Power law fits: optimal_lr = k * data_size^alpha")
print("(Based on empirically best LRs)")
print("-" * 80)

for optimizer in optimizers:
    if optimizer in empirical_power_law_params:
        params = empirical_power_law_params[optimizer]
        k_emp = params['k']
        alpha_emp = params['alpha']
        r2_emp = params['r_squared']
        print(f"{optimizer:>15}: optimal_lr = {k_emp:.6f} * N^{alpha_emp:.4f}  (R² = {r2_emp:.4f})")

# Predicted optimal learning rates at small data sizes
print("\n" + "="*80)
print("PREDICTED PEAK LEARNING RATE AT SMALL DATA SIZES (Quadratic Fit)")
print("="*80)
print("Using power law: optimal_lr = k * N^alpha")
print("(Based on fitted optimal LRs from quadratic fits)")
print("-" * 80)

# Predict at data_size = 1B tokens and at the minimum data size
prediction_data_sizes = [1.0, min(data_sizes)]  # 1B tokens and minimum in dataset
if prediction_data_sizes[0] == prediction_data_sizes[1]:
    prediction_data_sizes = [prediction_data_sizes[0]]

for optimizer in optimizers:
    if optimizer in lr_power_law_params:
        params = lr_power_law_params[optimizer]
        k = params['k']
        alpha = params['alpha']
        
        print(f"\n{optimizer.upper()}:")
        print(f"  Power law: optimal_lr = {k:.6f} * N^{alpha:.4f}")
        
        for pred_size in prediction_data_sizes:
            predicted_lr = k * (pred_size ** alpha)
            print(f"  Predicted optimal LR at N = {pred_size:.2f}B tokens: {predicted_lr:.6f}")
            
        # Also show at data_size = 0.5B, 0.1B, 0.01B, 100B, 1000B for reference
        for extra_size in [0.5, 0.1, 0.01, 100, 1000, 10000, 40000]:
            predicted_lr = k * (extra_size ** alpha)
            print(f"  Predicted optimal LR at N = {extra_size:.2f}B tokens: {predicted_lr:.6f} (extrapolation)")
    else:
        print(f"  {optimizer}: Not enough data to fit power law")

# Predicted optimal learning rates at small data sizes (Empirical)
print("\n" + "="*80)
print("PREDICTED PEAK LEARNING RATE AT SMALL DATA SIZES (Empirical)")
print("="*80)
print("Using power law: optimal_lr = k * N^alpha")
print("(Based on empirically best LRs)")
print("-" * 80)

for optimizer in optimizers:
    if optimizer in empirical_power_law_params:
        params = empirical_power_law_params[optimizer]
        k = params['k']
        alpha = params['alpha']
        
        print(f"\n{optimizer.upper()}:")
        print(f"  Power law: optimal_lr = {k:.6f} * N^{alpha:.4f}")
        
        for pred_size in prediction_data_sizes:
            predicted_lr = k * (pred_size ** alpha)
            print(f"  Predicted optimal LR at N = {pred_size:.2f}B tokens: {predicted_lr:.6f}")
            
        # Also show at data_size = 0.5B, 0.1B, 0.01B, 100B, 1000B for reference
        for extra_size in [0.5, 0.1, 0.01, 100, 1000, 10000, 40000]:
            predicted_lr = k * (extra_size ** alpha)
            print(f"  Predicted optimal LR at N = {extra_size:.2f}B tokens: {predicted_lr:.6f} (extrapolation)")
    else:
        print(f"  {optimizer}: Not enough data to fit power law")

print("\n" + "="*80)
print("SCALING LAW ANALYSIS (Power Law + Constant)")
print("="*80)
print("Model: Loss(N) = L_inf + A * N^(-alpha)")
print("where N is data size in billions of tokens")
print("-" * 80)

if 'scaling_law_params' in locals() and scaling_law_params:
    for optimizer in optimizers:
        if optimizer in scaling_law_params:
            params = scaling_law_params[optimizer]
            print(f"\n{optimizer.upper()}:")
            print(f"  L_inf (loss with infinite data): {params['L_inf']:.6f} ± {params['L_inf_err']:.6f}")
            print(f"  A (coefficient):                 {params['A']:.6f} ± {params['A_err']:.6f}")
            print(f"  alpha (power law exponent):      {params['alpha']:.4f} ± {params['alpha_err']:.4f}")
            print(f"  R² (goodness of fit):            {params['r_squared']:.4f}")
            print(f"  Formula: Loss(N) = {params['L_inf']:.6f} + {params['A']:.6f} * N^(-{params['alpha']:.4f})")
            
            # Extrapolation predictions
            print(f"\n  Extrapolated Predictions:")
            prediction_sizes = [50, 100, 500, 1000, 10000]
            for pred_size in prediction_sizes:
                pred_loss = scaling_law(pred_size, params['L_inf'], params['A'], params['alpha'])
                improvement = ((max(params['losses']) - pred_loss) / max(params['losses'])) * 100
                print(f"    N = {pred_size:>5}B tokens: Loss = {pred_loss:.6f} (↓{improvement:.1f}% from max)")
    
    # Compare optimizers
    print("\n" + "-" * 80)
    print("OPTIMIZER COMPARISON (at infinite data):")
    print("-" * 80)
    sorted_opts = sorted(scaling_law_params.items(), key=lambda x: x[1]['L_inf'])
    best_L_inf = sorted_opts[0][1]['L_inf']
    
    for rank, (optimizer, params) in enumerate(sorted_opts, 1):
        diff_from_best = params['L_inf'] - best_L_inf
        pct_diff = (diff_from_best / best_L_inf) * 100 if best_L_inf > 0 else 0
        print(f"  {rank}. {optimizer:>15}: L_inf = {params['L_inf']:.6f} "
              f"(+{diff_from_best:.6f} / +{pct_diff:.2f}% vs best)")
else:
    print("  No scaling law fits were computed.")

print("\n" + "="*80)
print("All plots generated successfully!")
print("="*80)

