import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
import numpy as np

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

    # Prepare data
    X_estimated = df[ESTIMATE_COL].values.reshape(-1, 1)
    X_accurate = df[ACCURATE_COL].values.reshape(-1, 1)
    y = df[ACTUAL_COL].values

    # Train model on Accurate RAM
    model = LinearRegression()
    model.fit(X_accurate, y)

    model_save_path = "ram_regression_model.pkl"
    joblib.dump(model, model_save_path)
    print(f"✅ RAM model saved to: {model_save_path}")

    print(f"📈 RAM Regression formula: RAM ≈ {model.coef_[0]:.4f} × Accurate + {model.intercept_:.2f}")

    # === 1st Diagram: Accurate RAM vs Measured RAM ===
    plt.figure(figsize=(8, 5))
    plt.scatter(df[ACCURATE_COL], y, color='blue', label="Accurate RAM vs Measured")

    # Perfect y=x line
    #line_range = np.linspace(min(df[ACCURATE_COL].min(), y.min()), max(df[ACCURATE_COL].max(), y.max()), 100)
    #plt.plot(line_range, line_range, color='grey', linestyle='--', label="Perfect Line (y=x)")

    # Linear Regression Line (Accurate RAM)
    plt.plot(df[ACCURATE_COL], model.predict(X_accurate), color='blue', linestyle='-', label="Regression Line (Accurate)")

    plt.xlabel("Accurate RAM (KB)")
    plt.ylabel("Measured RAM (KB)")
    plt.title("Accurate RAM vs Measured RAM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ram_accurate_vs_measured.png")
    print(f"📷 Accurate RAM Plot saved to: ram_accurate_vs_measured.png")

    # === 2nd Diagram: Estimated RAM vs Measured RAM ===
    plt.figure(figsize=(8, 5))
    plt.scatter(df[ESTIMATE_COL], y, color='red', label="Estimated RAM vs Measured")

    # Perfect y=x line
    line_range = np.linspace(min(df[ESTIMATE_COL].min(), y.min()), max(df[ESTIMATE_COL].max(), y.max()), 100)
    plt.plot(line_range, line_range, color='grey', linestyle='--', label="Perfect Line (y=x)")

    plt.xlabel("Estimated RAM (KB)")
    plt.ylabel("Measured RAM (KB)")
    plt.title("Estimated RAM vs Measured RAM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ram_estimated_vs_measured.png")
    print(f"📷 Estimated RAM Plot saved to: ram_estimated_vs_measured.png")


def save_inference_time_model():
    print(f"\n🚀 Training Inference Time model...")

    CSV_PATH = "ram_inference_flash.csv"  # Same file
    ACCURATE_COL = "MeasuredRam(KB)"
    INFERENCE_TIME_COL = "InferenceTime(S)"

    df = pd.read_csv(CSV_PATH)

    X = df[ACCURATE_COL].values.reshape(-1, 1)
    y = df[INFERENCE_TIME_COL].values

    model = LinearRegression()
    model.fit(X, y)

    model_save_path = "inference_time_regression_model.pkl"
    plot_save_path = "inference_time_regression_plot.png"

    joblib.dump(model, model_save_path)
    print(f"✅ Inference Time model saved to: {model_save_path}")

    print(f"📈 Inference Time Regression formula: Time ≈ {model.coef_[0]:.6f} × Accurate RAM + {model.intercept_:.6f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='purple', label="Actual Inference Times")
    plt.plot(X, model.predict(X), color='orange', label="Regression Line")
    plt.xlabel("Accurate RAM (KB)")
    plt.ylabel("Inference Time (s)")
    plt.title("Accurate RAM vs Inference Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"📷 Inference Time Plot saved to: {plot_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flash or RAM regression model.")
    parser.add_argument("--memory_type", default="ram", choices=["flash", "ram", "time"], help="Memory type to process")
    args = parser.parse_args()

    if args.memory_type.lower() == "flash":
        save_flash_model()
    elif args.memory_type.lower() == "ram":
        save_ram_model()
    else:
        save_inference_time_model()

    print("\n🎯 Finished training and saving models!")
