"""
Online Shopper Purchase Intent Prediction
Neuro Five Solutions Internship - Task 11 (Capstone Project)

PROBLEM STATEMENT
------------------
Can we predict, from a visitor's on-site browsing behavior during a
session, whether that session will end in a purchase? This is a real
conversion-rate-optimization (CRO) problem: an e-commerce store (Shopify
or otherwise) could use a live version of this to flag low-intent sessions
in real time and trigger an intervention (exit-intent discount, live chat
prompt, retargeting ad) before the visitor leaves empty-handed.

DATASET
-------
UCI "Online Shoppers Purchasing Intention Dataset" (Sakar et al.) -
12,330 real e-commerce sessions collected over one year, each belonging to
a different user (to avoid bias toward any one campcampaign or day).
Target: Revenue (True = made a purchase).

WORKFLOW
--------
  1. EDA - what session behaviors correlate with purchasing
  2. Clean + feature engineer
  3. Train 3 models (Logistic Regression, Random Forest, XGBoost), each a
     full Pipeline (preprocessing + classifier) trained with class
     weighting to handle the ~85/15 imbalance
  4. Evaluate and pick the best model
  5. Save the winning pipeline with joblib for deployment
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
import joblib

RANDOM_STATE = 42


def load_data(path="data/online_shoppers_intention.csv"):
    return pd.read_csv(path)


def quick_eda(df):
    print("=" * 60)
    print("EDA")
    print("=" * 60)
    rate = df["Revenue"].mean()
    print(f"Overall purchase rate: {rate:.1%}  (class balance: {1-rate:.1%} no purchase / {rate:.1%} purchase)")
    print("\nPurchase rate by visitor type:")
    print(df.groupby("VisitorType")["Revenue"].mean().round(3))
    print("\nPurchase rate by month (top 5):")
    print(df.groupby("Month")["Revenue"].mean().sort_values(ascending=False).head(5).round(3))
    print("\nPageValues - purchasers vs. non-purchasers:")
    print(df.groupby("Revenue")["PageValues"].mean().round(2))
    print("\nCorrelation of numeric features with Revenue:")
    num_cols = ["Administrative", "Administrative_Duration", "Informational",
                "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    print(df[num_cols].corrwith(df["Revenue"].astype(int)).sort_values(ascending=False).round(3))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    df.groupby("Revenue")["PageValues"].mean().plot(
        kind="bar", ax=axes[0], color=["#2563eb", "#059669"]
    )
    axes[0].set_title("Avg PageValues: Purchase vs. No Purchase")
    axes[0].set_xticklabels(["No purchase", "Purchase"], rotation=0)

    df.groupby("VisitorType")["Revenue"].mean().sort_values(ascending=False).plot(
        kind="bar", ax=axes[1], color="#dc2626"
    )
    axes[1].set_title("Purchase Rate by Visitor Type")
    axes[1].tick_params(axis="x", rotation=20)

    df.groupby("Month")["Revenue"].mean().reindex(
        ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ).plot(kind="bar", ax=axes[2], color="#7c3aed")
    axes[2].set_title("Purchase Rate by Month")

    plt.tight_layout()
    plt.savefig("eda_shopper_patterns.png", dpi=150)
    print("\nSaved EDA chart to eda_shopper_patterns.png")


def engineer_features(df):
    """
    3 new engineered features:
      1. TotalPages = Administrative + Informational + ProductRelated
         (how many pages total the visitor viewed, across all page types)
      2. TotalDuration = sum of all 3 duration columns
         (total time spent on the site this session)
      3. AvgTimePerPage = TotalDuration / TotalPages
         (engagement depth per page - a visitor lingering on each page vs.
         quickly skimming many pages may signal different intent)
    """
    df = df.copy()
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (
        df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    )
    df["AvgTimePerPage"] = np.where(df["TotalPages"] > 0, df["TotalDuration"] / df["TotalPages"], 0)
    return df


NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "TotalPages", "TotalDuration", "AvgTimePerPage",
]
CATEGORICAL_FEATURES = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend"]


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision (Purchase)": precision_score(y_test, y_pred),
        "recall (Purchase)": recall_score(y_test, y_pred),
        "f1 (Purchase)": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n--- {label} ---")
    print(classification_report(y_test, y_pred, target_names=["No purchase", "Purchase"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return metrics


def main():
    df = load_data()
    print(f"Loaded {len(df)} sessions, {df.shape[1]} columns\n")

    quick_eda(df)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Revenue"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} ({y_train.sum()} purchases)")
    print(f"Test size:  {len(X_test)} ({y_test.sum()} purchases)")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE,
            )),
        ]),
    }

    results = {}
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test, name.upper())
        fitted[name] = pipe

    print(f"\n{'='*70}\nMODEL COMPARISON\n{'='*70}")
    header = f"{'Metric':<24}{'Logistic Regression':<22}{'Random Forest':<16}{'XGBoost':<10}"
    print(header)
    print("-" * len(header))
    for metric in results["Logistic Regression"]:
        vals = [results[m][metric] for m in models]
        print(f"{metric:<24}{vals[0]:<22.4f}{vals[1]:<16.4f}{vals[2]:<10.4f}")

    comparison_rows = [
        {"model": m, "metric": metric, "score": round(results[m][metric], 4)}
        for m in models for metric in results[m]
    ]
    pd.DataFrame(comparison_rows).to_csv("model_comparison.csv", index=False)
    print("\nSaved model_comparison.csv")

    # Pick best model by F1 on the Purchase class (business-relevant: balances
    # catching real purchase-intent sessions against not crying wolf too often)
    best_name = max(results, key=lambda m: results[m]["f1 (Purchase)"])
    best_pipe = fitted[best_name]
    print(f"\nBest model by F1 (Purchase): {best_name}  (F1={results[best_name]['f1 (Purchase)']:.4f})")

    joblib.dump(best_pipe, "shopper_pipeline.joblib")
    print(f"Saved best pipeline to shopper_pipeline.joblib")

    # Feature importance for the winning model, if tree-based
    classifier = best_pipe.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        feature_names = (
            NUMERIC_FEATURES
            + list(best_pipe.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
        )
        importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
        print(f"\nTop 10 features ({best_name}):")
        print(importances.head(10).round(4))

        plt.figure(figsize=(8, 6))
        importances.head(12).sort_values().plot(kind="barh", color="#059669")
        plt.title(f"Top Features Driving Purchase Prediction ({best_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        print("Saved feature_importance.png")

    return results, best_name


if __name__ == "__main__":
    main()
