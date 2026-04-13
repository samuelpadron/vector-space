import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

with open('temporal_analysis/ncc_misalignment.json') as f:
    results = json.load(f)

# Filter to yaw rate 0-40 degrees/s
filtered = [r for r in results if 0 <= r['yaw_rate'] <= 40]
print(f"Samples in 0-40 deg/s range: {len(filtered)} / {len(results)}")

speeds    = np.array([r['speed']    for r in filtered])
yaw_rates = np.array([r['yaw_rate'] for r in filtered])
nccs      = np.array([r['ncc']      for r in filtered])

# Pearson correlation

r_speed,    p_speed    = stats.pearsonr(speeds,    nccs)
r_yaw,      p_yaw      = stats.pearsonr(yaw_rates, nccs)

print(f"\nSpeed    vs NCC:    r={r_speed:.4f}  p={p_speed:.4f}")
print(f"Yaw rate vs NCC:    r={r_yaw:.4f}  p={p_yaw:.4f}")

# Binned means

def bin_means(x, y, n_bins):
    bins   = np.linspace(x.min(), x.max(), n_bins + 1)
    centres, means, sems = [], [], []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() < 3:
            continue
        centres.append((bins[i] + bins[i + 1]) / 2)
        means.append(y[mask].mean())
        sems.append(y[mask].std() / np.sqrt(mask.sum()))
    return np.array(centres), np.array(means), np.array(sems)

speed_c,   speed_m,   speed_e   = bin_means(speeds,    nccs, n_bins=8)
yaw_c,     yaw_m,     yaw_e     = bin_means(yaw_rates, nccs, n_bins=8)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Camera-LiDAR BEV NCC vs Ego-Motion (yaw rate 0-40°/s)', fontsize=13)

configs = [
    (axes[0], speeds,    speed_c, speed_m, speed_e,
     'Speed (m/s)',     'steelblue',  r_speed, p_speed),
    (axes[1], yaw_rates, yaw_c,   yaw_m,   yaw_e,
     'Yaw Rate (°/s)', 'darkorange', r_yaw,   p_yaw),
]

for ax, x, centres, means, sems, xlabel, color, r_val, p_val in configs:
    # Scatter
    ax.scatter(x, nccs, alpha=0.3, s=15, color=color, zorder=1)

    # Binned means with error bars
    ax.errorbar(centres, means, yerr=sems, fmt='o-', color='black',
                linewidth=2, markersize=6, capsize=4, zorder=3,
                label='Bin mean ± SE')

    # Trend line
    z = np.polyfit(x, nccs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=1.5,
            label=f'slope={z[0]:.4f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('NCC (camera BEV vs LiDAR BEV)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f'r={r_val:.3f}  p={p_val:.3f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('temporal_analysis/ncc_filtered_binned.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to temporal_analysis/ncc_filtered_binned.png")
plt.show()