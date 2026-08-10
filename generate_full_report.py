import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

CLASSES = ['before', 'thin', 'cool', 'drink', 'go', 'computer', 'who', 'cousin', 'help', 'candy']
per_class_acc = [0, 0, 0, 0, 0, 0, 100, 0, 100, 0]

# ── FIGURE 1: COMPLETE EVALUATION DASHBOARD ──────────────────────────────────
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#f8f9fa')
gs = GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle('LightMamba-ASL — Complete Evaluation Report\n10 Classes | WLASL Dataset | CPU Inference',
             fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')

# ── Plot 1: Confusion Matrix (top-left, spans 2 cols) ────────────────────────
ax1 = fig.add_subplot(gs[0:2, 0:2])
cm = np.zeros((10, 10), dtype=int)
cm[6][6]=1; cm[8][8]=2
cm[0][6]=1; cm[0][7]=1; cm[0][9]=1
cm[1][2]=2; cm[2][1]=2
cm[3][7]=1; cm[4][0]=1
cm[5][3]=1; cm[5][9]=1
cm[7][3]=2
cm[9][6]=2; cm[9][1]=1
im = ax1.imshow(cm, interpolation='nearest', cmap='YlOrRd')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xticks(range(10)); ax1.set_yticks(range(10))
ax1.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=10)
ax1.set_yticklabels(CLASSES, fontsize=10)
ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        if cm[i,j] > 0:
            ax1.text(j, i, str(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j] > thresh else 'black', fontsize=11, fontweight='bold')

# ── Plot 2: Pie Chart — Correct vs Wrong ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
correct = 3; wrong = 15
sizes   = [correct, wrong]
labels  = [f'Correct\n({correct})', f'Wrong\n({wrong})']
colors  = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)
wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
    explode=explode, autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 10}, pctdistance=0.75)
for at in autotexts:
    at.set_fontweight('bold')
ax2.set_title('Prediction\nOutcome', fontsize=12, fontweight='bold')

# ── Plot 3: Pie Chart — Class Distribution ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
test_counts = [3, 2, 2, 1, 1, 2, 1, 2, 2, 3]
pie_colors  = plt.cm.Set3(np.linspace(0, 1, 10))
wedges, texts, autotexts = ax3.pie(test_counts, labels=CLASSES, colors=pie_colors,
    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8}, pctdistance=0.8)
for at in autotexts:
    at.set_fontsize(7)
ax3.set_title('Test Set\nClass Distribution', fontsize=12, fontweight='bold')

# ── Plot 4: Per-Class Accuracy Bar ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0:2])
bar_colors = ['#2ecc71' if v==100 else '#e74c3c' for v in per_class_acc]
bars = ax4.bar(CLASSES, per_class_acc, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
ax4.set_ylim(0, 130)
ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
ax4.axhline(y=16.67, color='#3498db', linestyle='--', linewidth=2, label='Top-1 Avg: 16.67%')
ax4.axhline(y=88.89, color='#9b59b6', linestyle='--', linewidth=2, label='Top-5 Avg: 88.89%')
ax4.axhline(y=10,    color='gray',    linestyle=':',  linewidth=1.5, label='Random: 10%')
for bar, val in zip(bars, per_class_acc):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=10)
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(axis='y', alpha=0.3)
ax4.set_facecolor('#fafafa')

# ── Plot 5: Pie Chart — Classes Correct vs Zero ──────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
correct_classes = 2; zero_classes = 8
sizes2  = [correct_classes, zero_classes]
labels2 = [f'100% Acc\n({correct_classes} classes)', f'0% Acc\n({zero_classes} classes)']
colors2 = ['#2ecc71', '#e74c3c']
wedges2, texts2, autotexts2 = ax5.pie(sizes2, labels=labels2, colors=colors2,
    autopct='%1.1f%%', startangle=90, explode=(0.05,0),
    textprops={'fontsize': 10}, pctdistance=0.75)
for at in autotexts2:
    at.set_fontweight('bold')
ax5.set_title('Class-Level\nPerformance', fontsize=12, fontweight='bold')

# ── Plot 6: Metrics Bar Chart ─────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 0])
metrics      = ['Top-1\nAcc', 'Top-5\nAcc', 'Macro\nF1', 'Weighted\nF1']
metric_vals  = [16.67, 88.89, 8.57, 7.94]
metric_colors= ['#e74c3c', '#2ecc71', '#e67e22', '#e67e22']
bars6 = ax6.bar(metrics, metric_vals, color=metric_colors, edgecolor='white', linewidth=1.5, width=0.5)
ax6.set_ylim(0, 110)
ax6.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold')
for bar, val in zip(bars6, metric_vals):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
             f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax6.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, label='Random (10%)')
ax6.legend(fontsize=9)
ax6.grid(axis='y', alpha=0.3)
ax6.set_facecolor('#fafafa')

# ── Plot 7: Efficiency Bar Chart ──────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
eff_labels = ['Params\n(M)', 'Checkpoint\n(MB)', 'Latency\n(x10 ms)', 'FPS\n(CPU)']
eff_vals   = [6.49, 55.17, 47.45, 2.11]
eff_colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
bars7 = ax7.bar(eff_labels, eff_vals, color=eff_colors, edgecolor='white', linewidth=1.5, width=0.5)
ax7.set_title('Efficiency Profile', fontsize=12, fontweight='bold')
for bar, val in zip(bars7, eff_vals):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)
ax7.set_facecolor('#fafafa')

# ── Plot 8: Top Confused Pairs ────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 2])
pairs  = ['cool→thin', 'cousin→drink', 'candy→who', 'before→who', 'before→cousin']
counts = [2, 2, 2, 1, 1]
colors8= ['#e74c3c','#e67e22','#f1c40f','#95a5a6','#95a5a6']
bars8 = ax8.barh(pairs, counts, color=colors8, edgecolor='white', linewidth=1.5)
ax8.set_xlabel('Confusion Count', fontsize=10, fontweight='bold')
ax8.set_title('Top Confused Pairs', fontsize=12, fontweight='bold')
for bar, val in zip(bars8, counts):
    ax8.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
             str(val), va='center', fontsize=10, fontweight='bold')
ax8.set_xlim(0, 3)
ax8.grid(axis='x', alpha=0.3)
ax8.set_facecolor('#fafafa')

plt.savefig('outputs/plots/complete_evaluation_report.png', dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
plt.close()
print("Saved: complete_evaluation_report.png")
