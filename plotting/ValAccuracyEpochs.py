import pandas as pd
import matplotlib.pyplot as plt
import os

# === Customize these with your actual file paths ===
csv_files = [
    "../Manual_Run/10.csv",
    "../Manual_Run/20.csv",
    "../Manual_Run/30.csv",
    "../Manual_Run/50.csv",
]
labels = [
    "TakuNet model trained for 10 epochs",
    "TakuNet model trained for 20 epochs",
    "TakuNet model trained for 30 epochs",
    "TakuNet model trained for 50 epochs"
]
all_val_acc = []

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    all_val_acc.extend(df['val_accuracy'].tolist())

global_min = min(all_val_acc)
global_max = max(all_val_acc)

# Step 2: Plot with global normalization
plt.figure(figsize=(10, 6))

for df, label in zip(dfs, labels):
    val_acc = df['val_accuracy']
    normalized_val_acc = (val_acc - global_min) / (global_max - global_min)
    plt.plot(df.index, normalized_val_acc, label=label)

plt.xlabel("Epochs")
plt.ylabel("Normalized Validation Accuracy (Global)")
plt.title("Validation Accuracy Comparison (Global Normalization)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = "val_accuracy_comparison_global.png"
plt.savefig(plot_path)
print(f"✅ Plot saved as {plot_path}")
