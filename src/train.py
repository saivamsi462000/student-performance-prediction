"""Train, compare and persist the at-risk classifier and the final-score regressor.

Usage:
    python -m src.train --data data/students.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.features import TARGET_CLASS, TARGET_REG, build_preprocessor, split_xy

SEED = 42
FIG_DIR = Path("reports/figures")
MODEL_DIR = Path("models")


def classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }


def train_classifier(df: pd.DataFrame) -> dict:
    X, y = split_xy(df, TARGET_CLASS)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in classification_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "cv_roc_auc_mean": round(cv_auc.mean(), 4),
            "cv_roc_auc_std": round(cv_auc.std(), 4),
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_precision": round(precision_score(y_te, pred), 4),
            "test_recall": round(recall_score(y_te, pred), 4),
            "test_f1": round(f1_score(y_te, pred), 4),
            "test_roc_auc": round(roc_auc_score(y_te, proba), 4),
        }
        fitted[name] = pipe
        print(f"[classify] {name:20s} AUC={results[name]['test_roc_auc']:.3f} "
              f"Recall={results[name]['test_recall']:.3f}")

    best = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    best_pipe = fitted[best]

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_te, y_te, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curves - at-risk classification")
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        best_pipe, X_te, y_te, display_labels=["On track", "At risk"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {best}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    imp = permutation_importance(best_pipe, X_te, y_te, n_repeats=10,
                                 random_state=SEED, scoring="roc_auc")
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(X.columns)[order], imp.importances_mean[order], color="#2b6cb0")
    ax.set_xlabel("Mean drop in ROC-AUC when shuffled")
    ax.set_title(f"Permutation feature importance - {best}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    joblib.dump(best_pipe, MODEL_DIR / "at_risk_classifier.joblib")
    top = sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])[:8]
    return {"models": results, "best_model": best,
            "top_features": [{"feature": f, "importance": round(v, 4)} for f, v in top]}


def train_regressor(df: pd.DataFrame) -> dict:
    X, y = split_xy(df, TARGET_REG)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    results, fitted = {}, {}
    for name, model in regression_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "test_r2": round(r2_score(y_te, pred), 4),
            "test_mae": round(mean_absolute_error(y_te, pred), 3),
            "test_rmse": round(root_mean_squared_error(y_te, pred), 3),
        }
        fitted[name] = pipe
        print(f"[regress]  {name:20s} R2={results[name]['test_r2']:.3f} MAE={results[name]['test_mae']:.2f}")

    best = max(results, key=lambda k: results[k]["cv_r2_mean"])
    pred = fitted[best].predict(X_te)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_te, pred, alpha=0.45, s=14, color="#2f855a")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set_xlabel("Actual final score"); ax.set_ylabel("Predicted final score")
    ax.set_title(f"Predicted vs actual - {best}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150); plt.close(fig)

    joblib.dump(fitted[best], MODEL_DIR / "final_score_regressor.joblib")
    return {"models": results, "best_model": best}


def eda_figures(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["final_score"], bins=30, color="#4a5568", edgecolor="white")
    ax.axvline(60, color="#c53030", ls="--", label="At-risk threshold (60)")
    ax.set_xlabel("Final score"); ax.set_ylabel("Students"); ax.legend()
    ax.set_title("Distribution of final scores")
    fig.tight_layout(); fig.savefig(FIG_DIR / "score_distribution.png", dpi=150); plt.close(fig)

    num = df.select_dtypes("number").drop(columns=["at_risk"])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix (numeric features)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "correlation_matrix.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/students.csv")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(args.data)
    eda_figures(df)
    metrics = {
        "dataset": {"rows": len(df), "at_risk_rate": round(df["at_risk"].mean(), 4)},
        "classification": train_classifier(df),
        "regression": train_regressor(df),
    }
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved reports/metrics.json, figures and models.")


if __name__ == "__main__":
    main()
