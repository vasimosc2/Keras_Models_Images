import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# === CONFIG ===
CSV_PATH = "../Manual_Run/results/Old/Training_Results.csv"
ESTIMATE_COL = "Estimated Flash Memory (KB)"
TFLITE_COL = "TFlite Estimation size(KB)"
MODEL_SAVE_PATH = "flash_regression_model_smooth.pkl"
PLOT = True

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)
X = df[ESTIMATE_COL].values.reshape(-1, 1)
y = df[TFLITE_COL].values

# === POLYNOMIAL REGRESSION FOR ALL DATA ===
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# === SAVE MODEL ===
joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ Smooth polynomial model saved to: {MODEL_SAVE_PATH}")

# === OPTIONAL: PLOT ===
if PLOT:
    # Sort for smooth plotting
    sorted_data = sorted(zip(X.flatten(), model.predict(X)))
    sorted_X, sorted_preds = zip(*sorted_data)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Actual TFLite Sizes", color='blue')
    plt.plot(sorted_X, sorted_preds, color='red', label="Polynomial Fit (Degree 2)")

    plt.xlabel("Estimated Flash Memory (KB)")
    plt.ylabel("True TFLite Size (KB)")
    plt.title("Flash Estimation: Smooth Polynomial Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../plotting/images/flash_regression_poly_fit.png")
    print("📊 Plot saved to: images/flash_regression_poly_fit.png")

