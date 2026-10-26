"""
evaluate.py
-----------
Scores the saved model on the held-out test set (MAE, RMSE, R-squared),
and saves diagnostic charts: actual-vs-predicted, residual distribution,
and feature importance (coefficients for linear models, or built-in
importances for tree-based models).

Usage:
    python evaluate.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


def get_feature_names(pipeline):
    pre = pipeline.named_steps["preprocess"]
    num_names = config.NUMERIC_FEATURES
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES))
    return list(num_names) + cat_names


def get_importances(pipeline):
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        return np.abs(model.coef_)
    elif hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None


def main():
    pipeline = joblib.load(config.BEST_MODEL_PATH)
    test_df = pd.read_csv(config.OUTPUT_DIR + "/test_set.csv")

    X_test = test_df.drop(columns=[config.TARGET_COL])
    y_test = test_df[config.TARGET_COL]
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2_score": round(float(r2), 4),
        "mape_percent": round(mape, 2),
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

    # --- Actual vs Predicted ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.35, s=18, color="#2F6FED")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "--", color="gray", label="Perfect prediction")
    plt.xlabel("Actual Salary ($)")
    plt.ylabel("Predicted Salary ($)")
    plt.title("Actual vs. Predicted Salary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR + "/actual_vs_predicted.png", dpi=120)
    plt.close()

    # --- Residuals ---
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4.5))
    plt.hist(residuals, bins=40, color="#2F6FED", alpha=0.85)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Residual (Actual - Predicted), $")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR + "/residuals.png", dpi=120)
    plt.close()

    # --- Feature importance ---
    importances = get_importances(pipeline)
    if importances is not None:
        names = get_feature_names(pipeline)
        imp_df = pd.DataFrame({"feature": names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        imp_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)

        plt.figure(figsize=(8, 6))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2F6FED")
        plt.xlabel("Importance (|coefficient| or model importance)")
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR + "/feature_importance.png", dpi=120)
        plt.close()

    print("Saved charts to", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
