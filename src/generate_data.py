"""Generate a realistic synthetic student dataset.

The dataset mimics the structure of public education datasets (demographics,
study behaviour, engagement, prior performance) so the full pipeline can be
run end-to-end without exposing any real student records.

Usage:
    python -m src.generate_data --rows 1500 --out data/students.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def generate(n_rows: int = 1500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Female", "Male"], n_rows)
    age = rng.integers(17, 24, n_rows)
    parental_education = rng.choice(
        ["High School", "Some College", "Bachelor", "Master", "Doctorate"],
        n_rows, p=[0.30, 0.25, 0.27, 0.13, 0.05],
    )
    internet_access = rng.choice(["Yes", "No"], n_rows, p=[0.85, 0.15])
    part_time_job = rng.choice(["Yes", "No"], n_rows, p=[0.35, 0.65])
    extracurricular = rng.choice(["Yes", "No"], n_rows, p=[0.45, 0.55])
    tutoring = rng.choice(["Yes", "No"], n_rows, p=[0.25, 0.75])

    study_hours = np.clip(rng.normal(12, 5, n_rows), 0, 35).round(1)
    attendance = np.clip(rng.normal(85, 10, n_rows), 40, 100).round(1)
    sleep_hours = np.clip(rng.normal(7, 1.1, n_rows), 4, 10).round(1)
    previous_gpa = np.clip(rng.normal(3.0, 0.5, n_rows), 1.0, 4.0).round(2)
    past_failures = rng.choice([0, 1, 2, 3], n_rows, p=[0.72, 0.17, 0.08, 0.03])
    lms_logins = np.clip(rng.poisson(25, n_rows) + (study_hours * 0.8).astype(int), 0, None)
    assignment_completion = np.clip(
        rng.normal(80, 12, n_rows) + (attendance - 85) * 0.4, 20, 100
    ).round(1)

    edu_bonus = pd.Series(parental_education).map(
        {"High School": 0, "Some College": 1, "Bachelor": 2, "Master": 3, "Doctorate": 3.5}
    ).to_numpy()

    midterm = np.clip(
        25
        + previous_gpa * 9
        + study_hours * 0.6
        + (attendance - 85) * 0.25
        - past_failures * 3
        + rng.normal(0, 6, n_rows),
        0, 100,
    ).round(1)

    final_score = (
        12
        + 0.45 * midterm
        + 0.55 * study_hours
        + 0.22 * (attendance - 60)
        + 0.12 * (assignment_completion - 50)
        + 4.0 * previous_gpa
        - 3.5 * past_failures
        + 1.2 * edu_bonus
        + np.where(internet_access == "Yes", 2.0, -2.0)
        + np.where(tutoring == "Yes", 2.5, 0.0)
        + np.where(part_time_job == "Yes", -1.8, 0.0)
        - 1.5 * np.abs(sleep_hours - 7.5)
        + 0.05 * lms_logins
        + rng.normal(0, 5, n_rows)
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame({
        "student_id": [f"S{100000 + i}" for i in range(n_rows)],
        "gender": gender,
        "age": age,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "part_time_job": part_time_job,
        "extracurricular": extracurricular,
        "tutoring": tutoring,
        "study_hours_weekly": study_hours,
        "attendance_pct": attendance,
        "sleep_hours": sleep_hours,
        "previous_gpa": previous_gpa,
        "past_failures": past_failures,
        "lms_logins_monthly": lms_logins,
        "assignment_completion_pct": assignment_completion,
        "midterm_score": midterm,
        "final_score": final_score,
    })
    df["at_risk"] = (df["final_score"] < 60).astype(int)
    df["grade"] = pd.cut(
        df["final_score"], bins=[-1, 59.99, 69.99, 79.99, 89.99, 100],
        labels=["F", "D", "C", "B", "A"],
    ).astype(str)

    # Inject a small amount of missing data so the cleaning step is meaningful.
    for col in ["sleep_hours", "lms_logins_monthly", "parental_education"]:
        mask = rng.random(n_rows) < 0.02
        df.loc[mask, col] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="data/students.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out} (at-risk rate {df['at_risk'].mean():.1%})")


if __name__ == "__main__":
    main()
