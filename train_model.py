"""
train_model.py
---------------
Trains the XGBoost churn classifier, evaluates it (AUC-ROC, precision,
recall, F1, confusion matrix), and persists the model + metrics.
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from config import RANDOM_SEED, MODEL_PATH, METRICS_PATH, REPORTS_DIR
from feature_pipeline import load_model_matrix

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=1.5,
    reg_alpha=0.5,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def train():
    X, y, ids = load_model_matrix(fit_schema=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    # further carve a validation split from train for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )

    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=30)

    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - t0

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_time_sec": round(train_time, 2),
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    model.save_model(MODEL_PATH)

    # save ROC points for plotting/reporting without retraining
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(REPORTS_DIR / "roc_curve.csv", index=False)

    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Precision:{precision:.4f}  Recall:{recall:.4f}  F1:{f1:.4f}")
    print(f"Model saved -> {MODEL_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")

    return model, X_test, y_test, metrics


if __name__ == "__main__":
    train()
