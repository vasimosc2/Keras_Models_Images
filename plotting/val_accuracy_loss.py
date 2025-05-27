import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
csvPath = os.path.join("results", "val_accuracy_comparison.csv")
savePath = os.path.join("plotting", "val_accuracy_loss_percentages.png")

# Load CSV
df = pd.read_csv(csvPath)

# Total models
total_models = len(df)

# Calculate thresholds
pct_under_1 = (df["Projected_Accuracy_Loss"] < 0.05).sum() / total_models * 100
pct_under_2 = (df["Projected_Accuracy_Loss"] < 0.10).sum() / total_models * 100
pct_under_3 = (df["Projected_Accuracy_Loss"] < 0.20).sum() / total_models * 100
pct_under_4 = (df["Projected_Accuracy_Loss"] < 0.30).sum() / total_models * 100

# Labels and data
thresholds = ["≤ 5% Relevant Loss", "≤ 10% Relevant Loss", "≤ 20% Relevant Loss", "≤ 30% Relevant Loss"]
percentages = [pct_under_1, pct_under_2, pct_under_3, pct_under_4]
colors = ['#91d1c2', '#7fbfff', '#aed581', '#c3aed6']

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(thresholds, percentages, color=colors)
plt.ylabel("Percentage of Models (%)")
plt.title("Projected Validation Accuracy Loss Thresholds")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add labels above bars
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)

# Optional: rotate x labels
plt.xticks(rotation=10)
plt.tight_layout()

# Save plot
plt.savefig(savePath, dpi=300)
print(f"✅ Plot saved as '{savePath}'")
