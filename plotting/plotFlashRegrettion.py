import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_PATH = "../Manual_Run/results/Old/Training_Results.csv"
MODEL_PATH = "../utils/flash_regression_model_poly.pkl"
ESTIMATE_COL = "Estimated Flash Memory (KB)"
TFLITE_COL = "TFlite Estimation size(KB)"
SAVE_PATH = "images/flash_regression_plot.png"

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
X = df[ESTIMATE_COL].values.reshape(-1, 1)
y = df[TFLITE_COL].values

# === LOAD MODEL (unpack) ===
model_small, model_large, threshold = joblib.load(MODEL_PATH)

# === Predict using piecewise logic ===
predictions = [
    model_small.predict([[x[0]]])[0] if x[0] <= threshold else model_large.predict([[x[0]]])[0]
    for x in X
]

# === SORT FOR PLOTTING ===
sorted_data = sorted(zip(X.flatten(), predictions))
sorted_X, sorted_preds = zip(*sorted_data)

# === PLOT ===
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual TFLite Sizes", color='blue')
plt.plot(sorted_X, sorted_preds, color='red', label="Piecewise Regression")

plt.xlabel("Estimated Flash Memory (KB)")
plt.ylabel("True TFLite Size (KB)")
plt.title("Flash Estimation: Piecewise Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f"✅ Plot saved to {SAVE_PATH}")

