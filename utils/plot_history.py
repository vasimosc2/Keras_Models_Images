manual_run = 'Manual_Run'
nas = "Nas"
name = "Random_1"

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(history_csv_path):
    df = pd.read_csv(history_csv_path)
    plt.figure(figsize=(10,6))
    plt.plot(df["loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss_curves("test/history_test.csv")