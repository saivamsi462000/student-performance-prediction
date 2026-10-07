"""Feature definitions and the shared preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = [
    "age", "study_hours_weekly", "attendance_pct", "sleep_hours", "previous_gpa",
    "past_failures", "lms_logins_monthly", "assignment_completion_pct", "midterm_score",
    "engagement_index",
]
BINARY = ["gender", "internet_access", "part_time_job", "extracurricular", "tutoring"]
ORDINAL = ["parental_education"]
EDU_ORDER = [["High School", "Some College", "Bachelor", "Master", "Doctorate"]]

TARGET_CLASS = "at_risk"
TARGET_REG = "final_score"
DROP = ["student_id", "final_score", "at_risk", "grade"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features: a single engagement signal combining attendance, LMS use and coursework."""
    out = df.copy()
    logins = out["lms_logins_monthly"].fillna(out["lms_logins_monthly"].median())
    out["engagement_index"] = (
        0.4 * out["attendance_pct"] / 100
        + 0.3 * out["assignment_completion_pct"] / 100
        + 0.3 * (logins / logins.max())
    ).round(4)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=EDU_ORDER)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("bin", binary, BINARY),
        ("ord", ordinal, ORDINAL),
    ])


def split_xy(df: pd.DataFrame, target: str):
    df = add_engineered_features(df)
    return df[NUMERIC + BINARY + ORDINAL], df[target]
