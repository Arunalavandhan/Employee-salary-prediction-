"""
generate_data.py
-----------------
Generates a synthetic employee salary dataset. Salary is built from a
realistic formula (experience, education, job title, department,
location, tenure) plus noise, so the trained model has real signal to
find. Gender is included as a column but deliberately given NO effect
on salary, so a well-trained model should assign it near-zero
importance -- a useful, honest talking point for the project.

Usage:
    python generate_data.py --n 3000 --out ../data/employee_salary_data.csv
"""
import argparse

import numpy as np
import pandas as pd

import config

EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
EDUCATION_BUMP = {"High School": 0, "Bachelor's": 8000, "Master's": 18000, "PhD": 30000}

JOB_TITLES = ["Analyst", "Engineer", "Senior Engineer", "Manager", "Director", "VP"]
JOB_BUMP = {"Analyst": 0, "Engineer": 12000, "Senior Engineer": 25000,
            "Manager": 35000, "Director": 60000, "VP": 90000}
# roughly how common each title is, junior-heavy like a real org
JOB_WEIGHTS = [0.28, 0.27, 0.20, 0.13, 0.08, 0.04]

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Marketing"]
DEPT_MULTIPLIER = {"Engineering": 1.06, "Sales": 1.00, "HR": 0.90, "Finance": 1.03, "Marketing": 0.95}

LOCATIONS = ["San Francisco", "New York", "Seattle", "Chicago", "Austin", "Remote"]
LOCATION_BUMP = {"San Francisco": 25000, "New York": 20000, "Seattle": 15000,
                 "Chicago": 8000, "Austin": 5000, "Remote": 0}

GENDERS = ["Female", "Male", "Non-binary"]


def generate(n: int, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 65, size=n)
    years_experience = np.clip(age - rng.integers(18, 24, size=n), 0, None)
    years_experience = np.minimum(years_experience, age - 18)
    years_at_company = np.array([rng.integers(0, max(1, min(exp, 20)) + 1) for exp in years_experience])

    gender = rng.choice(GENDERS, size=n, p=[0.48, 0.48, 0.04])
    education = rng.choice(EDUCATION_LEVELS, size=n, p=[0.15, 0.5, 0.28, 0.07])
    job_title = rng.choice(JOB_TITLES, size=n, p=JOB_WEIGHTS)
    department = rng.choice(DEPARTMENTS, size=n)
    location = rng.choice(LOCATIONS, size=n, p=[0.15, 0.18, 0.12, 0.15, 0.15, 0.25])

    base = 32000.0
    salary = (
        base
        + years_experience * 2200
        + years_at_company * 450
        + np.vectorize(EDUCATION_BUMP.get)(education)
        + np.vectorize(JOB_BUMP.get)(job_title)
        + np.vectorize(LOCATION_BUMP.get)(location)
    )
    salary = salary * np.vectorize(DEPT_MULTIPLIER.get)(department)
    salary = salary + rng.normal(0, 5500, size=n)  # noise
    salary = np.clip(salary, 28000, None).round(2)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "EducationLevel": education,
        "JobTitle": job_title,
        "Department": department,
        "Location": location,
        "YearsExperience": years_experience,
        "YearsAtCompany": years_at_company,
        "Salary": salary,
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out", type=str, default=config.DATA_PATH)
    args = parser.parse_args()

    df = generate(args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
