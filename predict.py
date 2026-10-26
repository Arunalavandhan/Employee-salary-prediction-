"""
predict.py
----------
Batch-scores a CSV of employee records using the trained pipeline.

Usage:
    python predict.py --input new_employees.csv --output predictions.csv
"""
import argparse

import joblib
import pandas as pd

import config


def predict_from_dataframe(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    if pipeline is None:
        pipeline = joblib.load(config.BEST_MODEL_PATH)
    predictions = pipeline.predict(df)
    result = df.copy()
    result["PredictedSalary"] = predictions.round(2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = predict_from_dataframe(df)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")
