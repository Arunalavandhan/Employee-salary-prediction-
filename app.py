"""
app.py
------
Flask web application for the employee salary prediction system.

Pages:
    GET  /          -> dashboard (model performance, comparison, feature importance)
    GET  /predict     -> form to enter an employee profile
    POST /predict     -> scores the form and shows the predicted salary

JSON API:
    GET  /health       -> {"status": "ok"}
    POST /api/predict  -> single employee record (JSON) -> predicted salary

Run:
    python app.py
Then open:
    http://localhost:5003
"""
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import config
import predict as predict_mod

app = Flask(__name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(config.BEST_MODEL_PATH)
    return _pipeline


def load_json(path):
    with open(path) as f:
        return json.load(f)


@app.route("/", methods=["GET"])
def dashboard():
    metrics = load_json(config.METRICS_PATH)
    training_summary = load_json(config.TRAINING_SUMMARY_PATH)
    return render_template(
        "dashboard.html",
        active="dashboard",
        metrics=metrics,
        best_model=training_summary["best_model"],
        cv_results=training_summary["cv_results"],
    )


EMPTY_FORM = {
    "Age": "30", "Gender": "Female", "EducationLevel": "Bachelor's",
    "YearsExperience": "5", "YearsAtCompany": "2", "JobTitle": "Engineer",
    "Department": "Engineering", "Location": "Remote",
}


def build_reasons(record: dict) -> list:
    reasons = []
    reasons.append(f"{record['JobTitle']} role")
    try:
        exp = float(record.get("YearsExperience", 0))
        if exp >= 15:
            reasons.append("Senior-level experience (15+ years)")
        elif exp <= 2:
            reasons.append("Early-career experience")
    except (TypeError, ValueError):
        pass
    if record.get("EducationLevel") in ("Master's", "PhD"):
        reasons.append(f"Advanced degree ({record['EducationLevel']})")
    if record.get("Location") in ("San Francisco", "New York", "Seattle"):
        reasons.append(f"High cost-of-living market ({record['Location']})")
    reasons.append(f"{record.get('Department')} department")
    return reasons


@app.route("/predict", methods=["GET", "POST"])
def predict_form():
    if request.method == "GET":
        return render_template("predict_form.html", active="predict", form=EMPTY_FORM, error=None)

    form = request.form.to_dict()
    record = {
        "Age": form.get("Age"),
        "Gender": form.get("Gender"),
        "EducationLevel": form.get("EducationLevel"),
        "JobTitle": form.get("JobTitle"),
        "Department": form.get("Department"),
        "Location": form.get("Location"),
        "YearsExperience": form.get("YearsExperience"),
        "YearsAtCompany": form.get("YearsAtCompany"),
    }

    try:
        df = pd.DataFrame([record])
        for col in ["Age", "YearsExperience", "YearsAtCompany"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[["Age", "YearsExperience", "YearsAtCompany"]].isnull().any().any():
            return render_template(
                "predict_form.html", active="predict", form=form,
                error="Age, years of experience, and years at company must all be numbers.",
            )
        result_df = predict_mod.predict_from_dataframe(df, get_pipeline())
        predicted_salary = float(result_df["PredictedSalary"].iloc[0])
    except Exception as e:
        return render_template(
            "predict_form.html", active="predict", form=form,
            error="Something went wrong while predicting. Please check your inputs and try again.",
        )

    metrics = load_json(config.METRICS_PATH)
    reasons = build_reasons(record)
    return render_template(
        "predict_result.html", active="predict",
        result={"predicted_salary": predicted_salary},
        mae=metrics["mae"], mape=metrics["mape_percent"], reasons=reasons,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    record = request.get_json(force=True)
    df = pd.DataFrame([record])
    try:
        result = predict_mod.predict_from_dataframe(df, get_pipeline())
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result.to_dict(orient="records")[0])


if __name__ == "__main__":
    get_pipeline()
    app.run(host="0.0.0.0", port=5003, debug=False)
