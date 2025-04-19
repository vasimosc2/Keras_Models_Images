import pandas as pd
import matplotlib.pyplot as plt

manual_run = 'Manual_Run'
nas = "Nas"
name = "Random_1"

def plot_training_curves(history_csv_path):
    df = pd.read_csv(history_csv_path)
    plt.figure(figsize=(10,6))
    plt.plot(df["loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.plot(df["accuracy"], label="Train Accuracy")
    plt.plot(df["val_accuracy"], label="Val Accuracy")
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_training_curves(f"{manual_run}/results/TakuNet_{name}_history.csv")
