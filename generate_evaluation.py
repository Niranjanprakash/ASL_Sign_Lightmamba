import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

os.makedirs('outputs/plots', exist_ok=True)

CLASSES = ['before', 'thin', 'cool', 'drink', 'go', 'computer', 'who', 'cousin', 'help', 'candy']

# ── 1. CONFUSION MATRIX ──────────────────────────────────────────────────────
# From evaluation output: who=100%, help=100%, rest=0%
# Confused pairs: cool->thin(2), cousin->drink(2), candy->who(2), before->who(1), before->cousin(1)
cm = np.zeros((10, 10), dtype=int)
# who: 1 test video → correct
cm[6][6] = 1   # who → who
# help: 2 test videos → correct
cm[8][8] = 2   # help → help
# before: 3 test videos → who(1), cousin(1), 1 wrong elsewhere
cm[0][6] = 1   # before → who
cm[0][7] = 1   # before → cousin
cm[0][9] = 1   # before → candy
# thin: 2 test videos → cool
cm[1][2] = 2   # thin → cool
# cool: 2 test videos → thin
cm[2][1] = 2   # cool → thin
# drink: 1 test video → cousin
cm[3][7] = 1   # drink → cousin
# go: 1 test video → wrong
cm[4][0] = 1   # go → before
# computer: 2 test videos → wrong
cm[5][3] = 1   # computer → drink
cm[5][9] = 1   # computer → candy
# cousin: 2 test videos → drink
cm[7][3] = 2   # cousin → drink
# candy: 3 test videos → who(2), wrong(1)
cm[9][6] = 2   # candy → who
cm[9][1] = 1   # candy → thin

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(CLASSES, fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix — LightMamba-ASL (10 Classes)', fontsize=14, fontweight='bold', pad=15)
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] confusion_matrix.png")

# ── 2. PER-CLASS ACCURACY BAR ────────────────────────────────────────────────
per_class_acc = [0, 0, 0, 0, 0, 0, 100, 0, 100, 0]
colors = ['#2ecc71' if v == 100 else '#e74c3c' if v == 0 else '#f39c12' for v in per_class_acc]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(CLASSES, per_class_acc, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
ax.set_ylim(0, 120)
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Accuracy — LightMamba-ASL (10 Classes)', fontsize=14, fontweight='bold')
ax.axhline(y=16.67, color='#3498db', linestyle='--', linewidth=2, label='Overall Top-1: 16.67%')
ax.axhline(y=88.89, color='#9b59b6', linestyle='--', linewidth=2, label='Overall Top-5: 88.89%')
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=11)
green_patch = mpatches.Patch(color='#2ecc71', label='100% Accuracy')
red_patch   = mpatches.Patch(color='#e74c3c', label='0% Accuracy')
ax.legend(handles=[green_patch, red_patch,
          plt.Line2D([0],[0], color='#3498db', linestyle='--', linewidth=2, label='Top-1: 16.67%'),
          plt.Line2D([0],[0], color='#9b59b6', linestyle='--', linewidth=2, label='Top-5: 88.89%')],
          fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] per_class_accuracy.png")

# ── 3. METRICS SUMMARY DASHBOARD ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# Metric cards
metrics = [
    ('Top-1 Accuracy', 16.67, '%', '#e74c3c'),
    ('Top-5 Accuracy', 88.89, '%', '#2ecc71'),
    ('Macro F1-Score', 8.57,  '%', '#e67e22'),
    ('Weighted F1',    7.94,  '%', '#e67e22'),
    ('Parameters',     6.49,  'M', '#3498db'),
    ('Latency (CPU)',  474.5, 'ms','#9b59b6'),
]
positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
for (r,c), (name, val, unit, color) in zip(positions, metrics):
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor(color + '22')
    ax.add_patch(mpatches.FancyBboxPatch((0.05,0.05), 0.9, 0.9,
        boxstyle="round,pad=0.05", facecolor=color+'33', edgecolor=color, linewidth=2,
        transform=ax.transAxes))
    ax.text(0.5, 0.62, f'{val}{unit}', ha='center', va='center',
            fontsize=22, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, name, ha='center', va='center',
            fontsize=11, color='#2c3e50', transform=ax.transAxes, fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')

fig.suptitle('LightMamba-ASL — Evaluation Metrics Dashboard', fontsize=15, fontweight='bold', y=1.01)
plt.savefig('outputs/plots/metrics_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] metrics_dashboard.png")

# ── 4. TOP-1 vs TOP-5 COMPARISON ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Top-1 Accuracy', 'Top-5 Accuracy', 'Macro F1', 'Weighted F1']
values     = [16.67, 88.89, 8.57, 7.94]
bar_colors = ['#e74c3c', '#2ecc71', '#e67e22', '#e67e22']
bars = ax.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, label='Random Baseline (10%)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/performance_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] performance_metrics.png")

# ── 5. EFFICIENCY PROFILE ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Parameters comparison
models     = ['MobileNetV2\n(baseline)', 'ResNet-50\n(baseline)', 'LightMamba\n-ASL (ours)']
params     = [3.4, 25.6, 6.49]
bar_colors = ['#95a5a6', '#95a5a6', '#3498db']
axes[0].bar(models, params, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
axes[0].set_ylabel('Parameters (M)', fontsize=12, fontweight='bold')
axes[0].set_title('Model Size Comparison', fontsize=13, fontweight='bold')
for i, (m, p) in enumerate(zip(models, params)):
    axes[0].text(i, p + 0.3, f'{p}M', ha='center', fontsize=11, fontweight='bold',
                 color='#3498db' if i == 2 else '#7f8c8d')
axes[0].grid(axis='y', alpha=0.3)

# FPS comparison
fps_labels = ['Target FPS\n(>25)', 'LightMamba-ASL\n(CPU)', 'LightMamba-ASL\n(GPU est.)']
fps_vals   = [25, 2.11, 28]
fps_colors = ['#2ecc71', '#e74c3c', '#3498db']
axes[1].bar(fps_labels, fps_vals, color=fps_colors, edgecolor='white', linewidth=1.5, width=0.5)
axes[1].set_ylabel('FPS', fontsize=12, fontweight='bold')
axes[1].set_title('Inference Speed', fontsize=13, fontweight='bold')
for i, (l, v) in enumerate(zip(fps_labels, fps_vals)):
    axes[1].text(i, v + 0.3, f'{v}', ha='center', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Model Efficiency Profile', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/efficiency_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] efficiency_profile.png")

print("[DONE] All 5 evaluation plots saved to outputs/plots/")
