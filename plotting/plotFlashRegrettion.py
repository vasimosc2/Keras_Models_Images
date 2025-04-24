import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_PATH = "../Manual_Run/results/Old/Training_Results.csv"
MODEL_PATH = "../utils/flash_regression_model.pkl"
ESTIMATE_COL = "Estimated Flash Memory (KB)"
TFLITE_COL = "TFlite Estimation size(KB)"
SAVE_PATH = "images/flash_regression_plot.png"

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
X = df[ESTIMATE_COL].values.reshape(-1, 1)
y = df[TFLITE_COL].values

# === LOAD MODEL ===
model = joblib.load(MODEL_PATH)

# === PLOT ===
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual TFLite Sizes", color='blue')
plt.plot(X, model.predict(X), color='red', label="Regression Line")

plt.xlabel("Estimated Flash Memory (KB)")
plt.ylabel("True TFLite Size (KB)")
plt.title("Flash Estimation vs Actual TFLite Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f"✅ Plot saved to {SAVE_PATH}")
