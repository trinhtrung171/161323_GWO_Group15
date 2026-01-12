#!/usr/bin/env python3
"""
Script to regenerate convergence plot with better clarity.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'

# Read the detailed metrics
df = pd.read_csv('results/detailed_metrics.csv')

# Group by algorithm and iteration
algorithms = df['Algorithm'].unique()
colors = {'TSILS': '#9467bd', 'GWO': '#1f77b4', 'WDGWO': '#d62728', 'CGWO': '#2ca02c'}
linestyles = {'TSILS': '-', 'GWO': '-', 'WDGWO': '-', 'CGWO': '--'}
linewidths = {'TSILS': 3, 'GWO': 2.5, 'WDGWO': 2.5, 'CGWO': 2.5}

fig, ax = plt.subplots(figsize=(12, 7))

for alg in algorithms:
    alg_data = df[df['Algorithm'] == alg]
    
    # Group by iteration and calculate mean and std
    grouped = alg_data.groupby('Iteration')['Best_Fitness'].agg(['mean', 'std']).reset_index()
    
    iterations = grouped['Iteration'].values
    mean_fitness = grouped['mean'].values
    std_fitness = grouped['std'].values
    
    # Plot mean line
    ax.plot(iterations, mean_fitness, 
            label=alg, 
            color=colors[alg], 
            linestyle=linestyles[alg],
            linewidth=linewidths[alg],
            alpha=0.9)
    
    # Plot std shadow (except TSILS which has no variance)
    if alg != 'TSILS':
        ax.fill_between(iterations, 
                        mean_fitness - std_fitness, 
                        mean_fitness + std_fitness,
                        color=colors[alg], 
                        alpha=0.15)

# Styling
ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax.set_ylabel('Best Fitness', fontsize=14, fontweight='bold')
ax.set_title('Convergence Comparison: JCAS Multibeam Optimization', 
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend with better positioning
legend = ax.legend(loc='upper right', 
                   fontsize=12, 
                   framealpha=0.95,
                   edgecolor='black',
                   shadow=True,
                   title='Algorithms',
                   title_fontsize=13)
legend.get_title().set_fontweight('bold')

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Target')

# Add annotations for key points
# TSILS failure
tsils_y = df[df['Algorithm'] == 'TSILS']['Best_Fitness'].iloc[0]
ax.annotate(f'TSILS: {tsils_y:.2f}\n(Failed)', 
            xy=(150, tsils_y), 
            xytext=(150, tsils_y + 3),
            fontsize=11,
            color=colors['TSILS'],
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors['TSILS'], alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=colors['TSILS'], lw=1.5))

# GWO success
gwo_final = df[(df['Algorithm'] == 'GWO') & (df['Iteration'] == 199)]['Best_Fitness'].mean()
ax.annotate(f'GWO: {gwo_final:.2f}\n(Best)', 
            xy=(199, gwo_final), 
            xytext=(160, gwo_final - 2),
            fontsize=11,
            color=colors['GWO'],
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors['GWO'], alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=colors['GWO'], lw=1.5))

# Set better y-axis limits
ax.set_ylim(bottom=-9, top=12)
ax.set_xlim(left=0, right=200)

# Add improvement percentage text box
improvement_text = (
    "Improvements vs TSILS:\n"
    "• GWO:    +164.33%\n"
    "• WDGWO: +160.03%\n"
    "• CGWO:  +146.58%"
)
ax.text(0.02, 0.02, improvement_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig('results/convergence_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Đã tạo lại đồ thị convergence với độ rõ nét cao hơn")
plt.close()

# Also create a zoomed version focusing on GWO variants
fig, ax = plt.subplots(figsize=(12, 7))

for alg in ['GWO', 'WDGWO', 'CGWO']:  # Exclude TSILS
    alg_data = df[df['Algorithm'] == alg]
    grouped = alg_data.groupby('Iteration')['Best_Fitness'].agg(['mean', 'std']).reset_index()
    
    iterations = grouped['Iteration'].values
    mean_fitness = grouped['mean'].values
    std_fitness = grouped['std'].values
    
    ax.plot(iterations, mean_fitness, 
            label=alg, 
            color=colors[alg], 
            linestyle=linestyles[alg],
            linewidth=linewidths[alg],
            alpha=0.9)
    
    ax.fill_between(iterations, 
                    mean_fitness - std_fitness, 
                    mean_fitness + std_fitness,
                    color=colors[alg], 
                    alpha=0.2)

ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax.set_ylabel('Best Fitness', fontsize=14, fontweight='bold')
ax.set_title('Convergence Comparison: GWO Variants (Zoomed)', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', shadow=True)
ax.set_ylim(bottom=-8.5, top=-4)

plt.tight_layout()
plt.savefig('results/convergence_comparison_zoomed.png', dpi=300, bbox_inches='tight')
print("✓ Đã tạo thêm đồ thị zoomed cho GWO variants")
plt.close()

print("\n✅ Hoàn tất tạo lại các đồ thị!")
