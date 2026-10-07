"""Score new students and produce an advisor-ready early-warning list.

Usage:
    python -m src.predict --data data/students.csv --out reports/early_warning_list.csv
"""
import argparse

import joblib
import pandas as pd

from src.features import BINARY, NUMERIC, ORDINAL, add_engineered_features


def recommend(row: pd.Series) -> str:
    """Rule-based interventions that turn a risk score into a concrete next step."""
    actions = []
    if row["attendance_pct"] < 75:
        actions.append("attendance outreach")
    if row["assignment_completion_pct"] < 70:
        actions.append("coursework check-in")
    if row["study_hours_weekly"] < 8:
        actions.append("study-skills workshop")
    if row["tutoring"] == "No" and row["midterm_score"] < 55:
        actions.append("refer to tutoring")
    if row["past_failures"] >= 2:
        actions.append("academic advisor meeting")
    return "; ".join(actions) or "monitor"


def score(df: pd.DataFrame, clf, reg) -> pd.DataFrame:
    X = add_engineered_features(df)[NUMERIC + BINARY + ORDINAL]
    out = df[["student_id"]].copy()
    out["risk_probability"] = clf.predict_proba(X)[:, 1].round(3)
    out["predicted_final_score"] = reg.predict(X).round(1)
    out["risk_band"] = pd.cut(out["risk_probability"], [-0.01, 0.3, 0.6, 1.0],
                              labels=["Low", "Medium", "High"]).astype(str)
    out["recommended_action"] = df.apply(recommend, axis=1)
    return out.sort_values("risk_probability", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/students.csv")
    parser.add_argument("--out", default="reports/early_warning_list.csv")
    args = parser.parse_args()

    clf = joblib.load("models/at_risk_classifier.joblib")
    reg = joblib.load("models/final_score_regressor.joblib")
    result = score(pd.read_csv(args.data), clf, reg)
    result.to_csv(args.out, index=False)
    print(result["risk_band"].value_counts().to_string())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
