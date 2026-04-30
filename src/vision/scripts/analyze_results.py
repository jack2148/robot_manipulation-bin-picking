import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RUNS = {
    "cross":    "runs/segment/train-3",
    "cylinder": "runs/segment/train-4",
    "square":   "runs/segment/train-5",
}
COLORS = {"cross": "#4C72B0", "cylinder": "#55A868", "square": "#C44E52"}

def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v.strip() for k, v in r.items()} for r in rows]

data = {name: load_csv(f"{base}/results.csv") for name, base in RUNS.items()}

def get(rows, key):
    return [float(r.get(key, 0) or 0) for r in rows]

# ── 1. 요약 표 ──────────────────────────────────────────────────────────────
best = {}
for name, rows in data.items():
    b = max(rows, key=lambda r: float(r.get("metrics/mAP50(M)", 0) or 0))
    best[name] = {
        "Best Epoch":     b["epoch"],
        "mAP50":          f"{float(b['metrics/mAP50(M)']):.4f}",
        "mAP50-95":       f"{float(b['metrics/mAP50-95(M)']):.4f}",
        "Precision":      f"{float(b['metrics/precision(M)']):.4f}",
        "Recall":         f"{float(b['metrics/recall(M)']):.4f}",
    }

fig_table, ax_t = plt.subplots(figsize=(10, 2.5))
ax_t.axis("off")
col_labels = ["Class", "Best Epoch", "mAP50", "mAP50-95", "Precision", "Recall"]
table_data = [
    [name] + list(best[name].values())
    for name in ["cross", "cylinder", "square"]
]
table = ax_t.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.2)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

row_colors = ["#d6eaf8", "#d5f5e3", "#fadbd8"]
for i, color in enumerate(row_colors, start=1):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)

fig_table.suptitle("YOLOv8-seg Training Summary", fontsize=14, fontweight="bold", y=0.98)
fig_table.tight_layout()
fig_table.savefig("training_summary_table.png", dpi=150, bbox_inches="tight")
print("저장: training_summary_table.png")

# ── 2. 학습 곡선 그래프 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("YOLOv8-seg Training Curves", fontsize=15, fontweight="bold")

metrics = [
    ("metrics/mAP50(M)",    "mAP50 (Mask)",    axes[0, 0]),
    ("metrics/mAP50-95(M)", "mAP50-95 (Mask)", axes[0, 1]),
    ("val/seg_loss",        "Val Seg Loss",    axes[1, 0]),
    ("val/box_loss",        "Val Box Loss",    axes[1, 1]),
]

for key, title, ax in metrics:
    for name, rows in data.items():
        epochs = [int(r["epoch"]) for r in rows]
        values = get(rows, key)
        ax.plot(epochs, values, label=name, color=COLORS[name], linewidth=2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if "loss" not in title.lower():
        ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("저장: training_curves.png")

# ── 3. 최종 지표 막대 그래프 ────────────────────────────────────────────────
fig_bar, ax_b = plt.subplots(figsize=(10, 5))
metric_keys = ["mAP50", "mAP50-95", "Precision", "Recall"]
classes = ["cross", "cylinder", "square"]
x = np.arange(len(metric_keys))
width = 0.25

for i, name in enumerate(classes):
    values = [float(best[name][k]) for k in metric_keys]
    bars = ax_b.bar(x + i * width, values, width, label=name, color=COLORS[name], alpha=0.85)
    for bar, val in zip(bars, values):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax_b.set_xticks(x + width)
ax_b.set_xticklabels(metric_keys, fontsize=12)
ax_b.set_ylim(0, 1.15)
ax_b.set_ylabel("Score")
ax_b.set_title("Best Metrics per Class", fontsize=13, fontweight="bold")
ax_b.legend()
ax_b.grid(True, axis="y", alpha=0.3)

fig_bar.tight_layout()
fig_bar.savefig("training_best_metrics.png", dpi=150, bbox_inches="tight")
print("저장: training_best_metrics.png")

plt.show()
