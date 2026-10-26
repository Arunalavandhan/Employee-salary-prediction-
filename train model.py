"""
train_model.py
---------------
Cross-validates three regression models on R-squared, picks the best,
refits it on the full training split, and saves the fitted pipeline
(preprocessing + model bundled together) plus a training summary.

Usage:
    python train_model.py
"""
import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import config
from preprocessing import build_preprocessor, clean_dataframe

CANDIDATES = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_STATE),
    "gradient_boosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
}


def main():
    df = pd.read_csv(config.DATA_PATH)
    df = clean_dataframe(df)

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    cv_results = {}
    best_name, best_score = None, -float("inf")

    for name, model in CANDIDATES.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        start = time.time()
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        elapsed = time.time() - start

        mean_r2 = float(scores.mean())
        cv_results[name] = {"mean_r2": round(mean_r2, 4), "fit_time_seconds": round(elapsed, 2)}
        print(f"{name}: mean CV R2 = {mean_r2:.4f} ({elapsed:.2f}s)")

        if mean_r2 > best_score:
            best_score = mean_r2
            best_name = name

    print(f"\nBest model: {best_name} (CV R2 = {best_score:.4f})")

    best_pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", CANDIDATES[best_name])])
    best_pipeline.fit(X_train, y_train)

    joblib.dump(best_pipeline, config.BEST_MODEL_PATH)
    X_test.assign(**{config.TARGET_COL: y_test}).to_csv(
        config.OUTPUT_DIR + "/test_set.csv", index=False
    )

    summary = {"best_model": best_name, "cv_results": cv_results}
    with open(config.TRAINING_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved model to {config.BEST_MODEL_PATH}")
    print(f"Saved training summary to {config.TRAINING_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
