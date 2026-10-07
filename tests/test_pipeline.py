import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import TARGET_CLASS, build_preprocessor, split_xy
from src.generate_data import generate
from src.predict import recommend


def test_generated_data_schema_and_ranges():
    df = generate(300, seed=1)
    assert len(df) == 300
    assert df["student_id"].is_unique
    assert df["final_score"].between(0, 100).all()
    assert set(df["at_risk"].unique()) <= {0, 1}
    assert (df["at_risk"] == (df["final_score"] < 60)).all()


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate(50, seed=7), generate(50, seed=7))


def test_preprocessor_handles_missing_values():
    df = generate(300, seed=2)
    X, y = split_xy(df, TARGET_CLASS)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (300, 2)


def test_model_beats_baseline():
    df = generate(800, seed=3)
    X, y = split_xy(df, TARGET_CLASS)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X[:600], y[:600])
    acc = (pipe.predict(X[600:]) == y[600:]).mean()
    baseline = max(y[600:].mean(), 1 - y[600:].mean())
    assert acc > baseline


def test_recommendations():
    row = pd.Series({"attendance_pct": 60, "assignment_completion_pct": 90, "study_hours_weekly": 15,
                     "tutoring": "Yes", "midterm_score": 70, "past_failures": 0})
    assert recommend(row) == "attendance outreach"
    row.update({"attendance_pct": 95})
    assert recommend(row) == "monitor"
