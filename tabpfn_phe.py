import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.ticker import ScalarFormatter
import numpy as np

# Set data directory path
data_dir = ""

# MAPE function (percentage form)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

# Store test results
test_metrics = {
    'R2': {},
    'RMSE': {},
    'MAE': {},
    'MAPE': {},
}

# English label mapping
DATASET_NAME_MAP = {
    'phe1': 'Final Normalized Flux',
    'phe2': 'Irreversible Resistance',
}

# Global style settings
sns.set_style("whitegrid")
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False

COLOR_PALETTE = sns.color_palette("tab10", 10)
MARKER_STYLES = ['o', 's', 'D', '^', 'v', 'p', '*', 'h']
LINE_STYLES = ['-', '--', '-.', ':']
FIG_SIZE = (12, 7)
FONT_SIZES = {
    'title': 16,
    'label': 16,
    'ticks': 14,
    'legend': 14,
    'annotation': 12
}

# Process CSV files
for filename in os.listdir(data_dir):
    if filename.startswith("TabPFN_results_") and filename.endswith(".xlsx"):
        match = re.match(r"TabPFN_results_(phe\d+)_(\d+(?:\.\d+)?)\.xlsx", filename)
        if not match:
            continue

        dataset = match.group(1)
        time = float(match.group(2))
        filepath = os.path.join(data_dir, filename)

        df = pd.read_excel(filepath, skiprows=1, header=None)
        test_data = df.iloc[:, [2, 3]].dropna()

        if not test_data.empty:
            y_true = test_data.iloc[:, 0]
            y_pred = test_data.iloc[:, 1]

            test_metrics['R2'].setdefault(dataset, []).append((time, r2_score(y_true, y_pred)))
            test_metrics['RMSE'].setdefault(dataset, []).append((time, mean_squared_error(y_true, y_pred, squared=False)))
            test_metrics['MAE'].setdefault(dataset, []).append((time, mean_absolute_error(y_true, y_pred)))
            test_metrics['MAPE'].setdefault(dataset, []).append((time, mean_absolute_percentage_error(y_true, y_pred)))

            print(f"🔎 Test set size: {y_true.shape}, Prediction set size: {y_pred.shape}")

# Plotting functions
def plot_single_axis_on_ax(ax, results_dict, ylabel, fixed_ylim=None, highlight_max=True):
    color_cycle = iter(COLOR_PALETTE)
    all_times, all_vals = [], []

    for dataset_idx, dataset in enumerate(sorted(results_dict.keys())):
        data_sorted = sorted(results_dict[dataset], key=lambda x: x[0])
        times = [x[0] for x in data_sorted]
        values = [x[1] for x in data_sorted]

        all_times.extend(times)
        all_vals.extend(values)

        color = next(color_cycle)
        marker = MARKER_STYLES[dataset_idx % len(MARKER_STYLES)]
        line_style = LINE_STYLES[dataset_idx % len(LINE_STYLES)]
        label_name = DATASET_NAME_MAP.get(dataset, dataset)

        ax.plot(times, values,
                color=color, linestyle=line_style,
                marker=marker, markersize=8,
                markerfacecolor='white', markeredgewidth=1.5,
                linewidth=2.5, label=label_name)

        best_val = max(values) if highlight_max else min(values)
        idx = values.index(best_val)
        best_time = times[idx]
        offset_y = 0.01 * (max(all_vals) - min(all_vals)) if highlight_max else -0.01 * (max(all_vals) - min(all_vals))
        text_y = best_val + offset_y

        ax.annotate(f'{best_val:.4f}',
                    xy=(best_time, best_val),
                    xytext=(best_time, text_y),
                    textcoords='data',
                    fontsize=14,  # 标注字体大小，可以调大
                    ha='center',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
                    bbox=dict(boxstyle="round,pad=0.2", fc="w", ec=color, lw=1.0),
                    clip_on=True)

    ax.set_xscale('log')
    unique_times = sorted(set(all_times))
    step = max(1, len(unique_times) // 5)
    xticks = unique_times[::step] + ([unique_times[-1]] if unique_times[-1] not in unique_times[::step] else [])
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())

    # x/y轴标签字体加粗且字体变大
    ax.set_xlabel('Optimization Time (s)', fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')

    # 刻度字体加粗且字体变大，刻度线加粗
    ax.tick_params(labelsize=16, width=2, length=6, labelcolor='black')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # 设置y轴范围，带一定边距或固定范围
    if fixed_ylim:
        ax.set_ylim(fixed_ylim)
    else:
        margin = 0.05 * (max(all_vals) - min(all_vals))
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    # 黑色边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.grid(True, linestyle='--', alpha=0.6)

    ax.legend(fontsize=16, ncol=len(results_dict), frameon=False, loc='upper right', bbox_to_anchor=(1, 1))


def plot_twin_axis_on_ax(ax_left, results_dict, ylabel, left_ylim, right_ylim, left_yticks=None, right_yticks=None, highlight_max=False):
    ax_right = ax_left.twinx()
    color_cycle = iter(COLOR_PALETTE)
    all_times = []

    for dataset_idx, dataset in enumerate(sorted(results_dict.keys())):
        data_sorted = sorted(results_dict[dataset], key=lambda x: x[0])
        times = [x[0] for x in data_sorted]
        values = [x[1] for x in data_sorted]
        all_times.extend(times)

        color = next(color_cycle)
        marker = MARKER_STYLES[dataset_idx % len(MARKER_STYLES)]
        line_style = LINE_STYLES[dataset_idx % len(LINE_STYLES)]
        label_name = DATASET_NAME_MAP.get(dataset, dataset)

        ax = ax_left if dataset == 'phe1' else ax_right
        ylim = left_ylim if dataset == 'phe1' else right_ylim

        ax.plot(times, values,
                color=color,
                linestyle=line_style,
                marker=marker,
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5,
                linewidth=2.5,
                label=label_name)

        best_val = max(values) if highlight_max else min(values)
        idx = values.index(best_val)
        best_time = times[idx]
        offset = 0.05 * (ylim[1] - ylim[0])
        text_y = best_val + offset if highlight_max else best_val - offset
        text_y = np.clip(text_y, ylim[0] + 0.01 * (ylim[1] - ylim[0]), ylim[1] - 0.01 * (ylim[1] - ylim[0]))

        annotation_text = f'{best_val:.4f}' if dataset == 'phe1' else f'{best_val:.2e}'

        ax.annotate(annotation_text,
                    xy=(best_time, best_val),
                    xytext=(best_time, text_y),
                    textcoords='data',
                    fontsize=14,  # 可以调大
                    ha='center',
                    arrowprops=dict(arrowstyle="->", color='black'),
                    bbox=dict(boxstyle="round,pad=0.3", fc="w", ec=color, lw=1))

    ax_left.set_xscale('log')
    unique_times = sorted(set(all_times))
    step = max(1, len(unique_times) // 5)
    xticks = unique_times[::step] + ([unique_times[-1]] if unique_times[-1] not in unique_times[::step] else [])
    ax_left.set_xticks(xticks)
    ax_left.xaxis.set_major_formatter(ScalarFormatter())
    ax_left.xaxis.set_minor_locator(plt.NullLocator())

    # 字体加粗且尺寸调大
    ax_left.set_xlabel('Optimization Time (s)', fontsize=18, fontweight='bold')
    ax_left.set_ylabel(ylabel, fontsize=18, fontweight='bold')

    ax_left.set_ylim(left_ylim)
    if left_yticks:
        ax_left.set_yticks(left_yticks)

    ax_right.set_ylim(right_ylim)
    if right_yticks:
        ax_right.set_yticks(right_yticks)
    ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_right.yaxis.offsetText.set_fontsize(16)

    # 刻度字体加粗且尺寸调大，刻度线加粗
    ax_left.tick_params(labelsize=16, width=2, length=6, labelcolor='black')
    ax_right.tick_params(labelsize=16, width=2, length=6, labelcolor='black')

    for label in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax_right.get_yticklabels():
        label.set_fontweight('bold')

    # 黑色边框加粗
    for spine in ax_left.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    for spine in ax_right.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax_left.grid(True, linestyle='--', alpha=0.6)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    combined = dict(zip(labels_left + labels_right, handles_left + handles_right))

    ax_left.legend(combined.values(), combined.keys(), fontsize=16, ncol=len(combined), frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0))

# Combine subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

plot_single_axis_on_ax(axs[0, 0], test_metrics['R2'], 'R²', fixed_ylim=(0.75, 1.0), highlight_max=True)
plot_single_axis_on_ax(axs[0, 1], test_metrics['MAPE'], 'MAPE (%)', fixed_ylim=(5, 35), highlight_max=False)
plot_twin_axis_on_ax(axs[1, 0], test_metrics['MAE'], 'MAE', left_ylim=(0, 0.02), right_ylim=(6e11, 12e11),
                     left_yticks=[0.00, 0.005, 0.01, 0.015, 0.02],
                     right_yticks=[6e11, 8e11, 10e11, 12e11],
                     highlight_max=False)
plot_twin_axis_on_ax(axs[1, 1], test_metrics['RMSE'], 'RMSE', left_ylim=(0.01, 0.03), right_ylim=(0.75e12, 1.75e12),
                     left_yticks=[0.01, 0.015, 0.020, 0.025, 0.030],
                     right_yticks=[0.75e12, 1.0e12, 1.25e12, 1.5e12, 1.75e12],
                     highlight_max=False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(data_dir, 'combined_performance_plot.png'), dpi=1200)
plt.show()
