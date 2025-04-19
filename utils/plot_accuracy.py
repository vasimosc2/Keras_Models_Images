manual_run = 'Manual_Run'
nas = "Nas"
name = "Random_1"

import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_curves_normalized(history_csv_path):
    df = pd.read_csv(history_csv_path)

    # Normalize to max value = 1
    max_acc = max(df["accuracy"].max(), df["val_accuracy"].max())
    df["accuracy_norm"] = df["accuracy"] / max_acc
    df["val_accuracy_norm"] = df["val_accuracy"] / max_acc

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["accuracy_norm"], label="Train Accuracy (normalized)")
    plt.plot(df["val_accuracy_norm"], label="Val Accuracy (normalized)")
    plt.title("Normalized Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Accuracy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)  # Enforce 0–1 scale
    plt.show()

# Example usage
# plot_accuracy_curves_normalized("Manual_Run/results/TakuNet_Random_1_history.csv")
plot_accuracy_curves_normalized("test/history_test.csv")