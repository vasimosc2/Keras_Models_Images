import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_PATH = "../Manual_Run/results/Old/Training_Results.csv"
ESTIMATE_COL = "Estimated Flash Memory (KB)"
TFLITE_COL = "TFlite Estimation size(KB)"
MODEL_SAVE_PATH = "flash_regression_model.pkl"
PLOT = False

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)

# === PREPARE DATA ===
X = df[ESTIMATE_COL].values.reshape(-1, 1)
y = df[TFLITE_COL].values

# === TRAIN MODEL ===
model = LinearRegression()
model.fit(X, y)

# === SAVE MODEL ===
joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ Regression model saved to: {MODEL_SAVE_PATH}")

# === SHOW EQUATION ===
print(f"📈 Regression formula: TFLite ≈ {model.coef_[0]:.4f} × Estimate + {model.intercept_:.2f}")

# === OPTIONAL: PLOT ===
if PLOT:
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Actual TFLite Sizes", color='blue')
    plt.plot(X, model.predict(X), color='red', label="Regression Line")
    plt.xlabel("Estimated Flash Memory (KB)")
    plt.ylabel("True TFLite Size (KB)")
    plt.title("Flash Estimation vs Actual TFLite Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
