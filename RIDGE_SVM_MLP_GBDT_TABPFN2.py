import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import calculate_metrics

sns.set_style("whitegrid")
sns.set_palette("deep")

file_paths = [
  
]
model_names = ["Ridge", "SVM", "MLP", "GBDT", "TabPFN"]

colors = ['#2A5CAA', '#D62728', '#2E9F2E', '#9467BD', '#FF7F0E']
test_colors = ['#88A4D3', '#FF9896', '#9ED69E', '#C5B0D5', '#FFBB78']

results_summary = pd.DataFrame(columns=["Model", "Set", "R²", "RMSE", "MAE", "MAPE"])

plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

for i, (file_path, model_name) in enumerate(zip(file_paths, model_names)):
    df = pd.read_excel(file_path)
    df_train = df.iloc[:, :2].dropna()
    df_test = df.iloc[:, 2:4].dropna()

    y_train_true = df_train.iloc[:, 0].values
    y_train_pred = df_train.iloc[:, 1].values
    y_test_true = df_test.iloc[:, 0].values
    y_test_pred = df_test.iloc[:, 1].values

    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)

    results_summary = pd.concat([
        results_summary,
        pd.DataFrame({
            "Model": [model_name, model_name],
            "Set": ["Train", "Test"],
            "R²": [train_metrics[0], test_metrics[0]],
            "RMSE": [train_metrics[1], test_metrics[1]],
            "MAE": [train_metrics[2], test_metrics[2]],
            "MAPE": [train_metrics[3], test_metrics[3]]
        })
    ], ignore_index=True)

    scale_factor = 1e12
    y_train_true_scaled = y_train_true / scale_factor
    y_train_pred_scaled = y_train_pred / scale_factor
    y_test_true_scaled = y_test_true / scale_factor
    y_test_pred_scaled = y_test_pred / scale_factor

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        y_train_true_scaled, y_train_pred_scaled,
        s=40, alpha=0.7,
        color=colors[i],
        edgecolor='w', linewidth=0.3,
        label='Train Set'
    )
    ax.scatter(
        y_test_true_scaled, y_test_pred_scaled,
        s=40, alpha=0.7,
        color=test_colors[i],
        edgecolor='w', linewidth=0.3,
        label='Test Set'
    )

    def plot_trendline(x, y, color, label):
        coeff = np.polyfit(x, y, 1)
        poly = np.poly1d(coeff)
        x_range = np.linspace(min(x), max(x), 100)
        ax.plot(
            x_range, poly(x_range),
            color=color, linestyle='--',
            linewidth=1.5, alpha=0.8,
            label=label
        )
        return coeff

    train_coeff = plot_trendline(
        y_train_true_scaled, y_train_pred_scaled,
        color=colors[i],
        label=f'Train Trendline (slope={np.polyfit(y_train_true_scaled, y_train_pred_scaled, 1)[0]:.2f})'
    )
    test_coeff = plot_trendline(
        y_test_true_scaled, y_test_pred_scaled,
        color=test_colors[i],
        label=f'Test Trendline (slope={np.polyfit(y_test_true_scaled, y_test_pred_scaled, 1)[0]:.2f})'
    )

    min_val = min(np.min(y_train_true_scaled), np.min(y_test_true_scaled))
    max_val = max(np.max(y_train_true_scaled), np.max(y_test_true_scaled))
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', lw=1, alpha=0.5, label='Ideal Fit')

    text_content = (
        f'Train Set Errors:\n'
        f'R² = {train_metrics[0]:.4f}\n'
        f'RMSE = {train_metrics[1]:.2e}\n'
        f'MAE = {train_metrics[2]:.2e}\n'
        f'MAPE = {train_metrics[3]:.2f}%\n\n'
        f'Test Set Errors:\n'
        f'R² = {test_metrics[0]:.4f}\n'
        f'RMSE = {test_metrics[1]:.2e}\n'
        f'MAE = {test_metrics[2]:.2e}\n'
        f'MAPE = {test_metrics[3]:.2f}%'
    )
    ax.text(
        0.75, 0.05, text_content,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            edgecolor='black',
            boxstyle='round,pad=0.5',
            linewidth=1.5
        ),
        verticalalignment='bottom',
        horizontalalignment='left'
    )

    ax.set_title(model_name, fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual Value (10¹² m⁻¹)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Value (10¹² m⁻¹)', fontsize=14, fontweight='bold')

    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.05, 0.95),
        frameon=True,
        ncol=1,
        fontsize=12,
        title_fontsize='12',
        shadow=True
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('black')

    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.tight_layout()

    output_path = f""
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

summary_path = ""
results_summary.to_csv(summary_path, index=False)

print("\n✅ All enhanced plots and error summaries saved successfully.")
