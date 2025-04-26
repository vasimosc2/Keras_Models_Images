import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

def save_flash_model():
    print(f"\n🚀 Training FLASH model...")

    CSV_PATH = "flash.csv"
    ESTIMATE_COL = "Estimated Flash Memory (KB)"
    ACTUAL_COL = "TFlite Estimation size(KB)"

    df = pd.read_csv(CSV_PATH)

    X = df[ESTIMATE_COL].values.reshape(-1, 1)
    y = df[ACTUAL_COL].values

    model = LinearRegression()
    model.fit(X, y)

    model_save_path = "flash_regression_model.pkl"
    plot_save_path = "flash_regression_plot.png"

    joblib.dump(model, model_save_path)
    print(f"✅ FLASH model saved to: {model_save_path}")

    print(f"📈 FLASH Regression formula: FLASH ≈ {model.coef_[0]:.4f} × Estimate + {model.intercept_:.2f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Actual TFLite Sizes", color='blue')
    plt.plot(X, model.predict(X), color='red', label="Regression Line")
    plt.xlabel("Estimated Flash Memory (KB)")
    plt.ylabel("True TFLite Size (KB)")
    plt.title("Flash Estimation vs Actual TFLite Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"📷 FLASH Plot saved to: {plot_save_path}")

def save_ram_model():
    print(f"\n🚀 Training RAM model...")

    CSV_PATH = "ram_inference_flash.csv"
    ESTIMATE_COL = "StartingRam(KB)"
    ACCURATE_COL = "AccurateRam(KB)"
    ACTUAL_COL = "MeasuredRam(KB)"

    df = pd.read_csv(CSV_PATH)

    X_accurate = df[ACCURATE_COL].values.reshape(-1, 1)
    y = df[ACTUAL_COL].values

    model = LinearRegression()
    model.fit(X_accurate, y)

    model_save_path = "ram_regression_model.pkl"
    plot_save_path = "ram_regression_plot.png"

    joblib.dump(model, model_save_path)
    print(f"✅ RAM model saved to: {model_save_path}")

    print(f"📈 RAM Regression formula: RAM ≈ {model.coef_[0]:.4f} × Estimated + {model.intercept_:.2f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(df[ESTIMATE_COL], y, label="Estimated RAM vs Measured", color='orange', marker='x')
    plt.scatter(df[ACCURATE_COL], y, label="Accurate RAM vs Measured", color='green', marker='o')
    plt.plot(df[ESTIMATE_COL], model.predict(X_accurate), color='red', label="Regression Line (Estimated)")
    plt.xlabel("RAM Memory (KB)")
    plt.ylabel("Measured RAM (KB)")
    plt.title("Estimated vs Accurate vs Measured RAM Memory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"📷 RAM Plot saved to: {plot_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flash or RAM regression model.")
    parser.add_argument("--memory_type", default="ram", choices=["flash", "ram"], help="Memory type to process")
    args = parser.parse_args()

    if args.memory_type.lower() == "flash":
        save_flash_model()
    else:
        save_ram_model()

    print("\n🎯 Finished training and saving models!")
