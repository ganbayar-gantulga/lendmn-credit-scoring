"""
LendMN Credit Scoring — ML Training Pipeline
Trains and evaluates Logistic Regression, Random Forest, and XGBoost.
Saves the best model + preprocessor for inference.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

COLORS = {"LendMN Red": "#E03131", "bg": "#FAFAFA", "text": "#1A1A2E"}


# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────
def load_and_preprocess(path: Path):
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df):,} rows | Default rate: {df['default'].mean():.1%}")

    # Encode categorical
    le = LabelEncoder()
    df["employment_type_enc"] = le.fit_transform(df["employment_type"])

    feature_cols = [
        "age", "employment_type_enc", "monthly_income",
        "employment_months", "num_existing_loans", "credit_history_months",
        "previous_defaults", "num_dependents", "loan_amount",
        "debt_to_income", "loan_to_income",
    ]
    X = df[feature_cols].copy()
    y = df["default"].copy()

    return X, y, le, feature_cols


# ─────────────────────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
    }


# ─────────────────────────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob  = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)
        auc     = roc_auc_score(y_test, y_prob)
        cv_auc  = cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="roc_auc", n_jobs=-1).mean()
        report  = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model":     model,
            "auc":       auc,
            "cv_auc":    cv_auc,
            "precision": report["1"]["precision"],
            "recall":    report["1"]["recall"],
            "f1":        report["1"]["f1-score"],
            "y_prob":    y_prob,
            "y_pred":    y_pred,
        }
        print(f"  {name:25s} AUC={auc:.4f}  CV-AUC={cv_auc:.4f}  "
              f"P={report['1']['precision']:.3f}  R={report['1']['recall']:.3f}")

    return results


# ─────────────────────────────────────────────────────────────
# 4. CREDIT SCORE MAPPING  (300 – 850 like FICO)
# ─────────────────────────────────────────────────────────────
def prob_to_score(prob: float) -> int:
    """Map default probability [0,1] → credit score [300,850]."""
    score = 850 - (prob ** 0.5) * 550
    return int(np.clip(score, 300, 850))


def score_band(score: int) -> str:
    if score >= 750: return "Маш сайн"       # Excellent
    if score >= 700: return "Сайн"            # Good
    if score >= 650: return "Дундаж"          # Fair
    if score >= 580: return "Дор"             # Poor
    return "Маш дор"                           # Very Poor


# ─────────────────────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────────────────────
def plot_results(results, X_test, y_test, feature_cols):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FAFAFA")
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    palette = ["#E03131", "#2B6CB0", "#2D9D60"]

    # — ROC Curves —
    ax = axes[0, 0]
    ax.set_facecolor("#F5F5F5")
    for (name, res), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0,1],[0,1],"--", color="#AAAAAA")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); ax.grid(alpha=0.3)

    # — AUC Comparison Bar —
    ax = axes[0, 1]
    ax.set_facecolor("#F5F5F5")
    names = list(results.keys())
    aucs  = [results[n]["auc"] for n in names]
    bars  = ax.barh(names, aucs, color=palette, height=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(auc - 0.015, bar.get_y() + bar.get_height()/2,
                f"{auc:.4f}", va="center", ha="right", color="white", fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("AUC Comparison", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # — Confusion Matrix (best model) —
    best_name = max(results, key=lambda n: results[n]["auc"])
    best_res  = results[best_name]
    ax = axes[1, 0]
    ax.set_facecolor("#F5F5F5")
    cm   = confusion_matrix(y_test, best_res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
    disp.plot(ax=ax, colorbar=False, cmap="Reds")
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")

    # — Feature Importance (XGBoost if available) —
    ax = axes[1, 1]
    ax.set_facecolor("#F5F5F5")
    if "XGBoost" in results:
        model = results["XGBoost"]["model"]
        imp   = model.feature_importances_
        sorted_idx = np.argsort(imp)
        ax.barh(
            [feature_cols[i].replace("_", " ").title() for i in sorted_idx],
            imp[sorted_idx], color="#E03131", height=0.6,
        )
        ax.set_title("Feature Importance (XGBoost)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("LendMN Credit Scoring — Model Evaluation",
                 fontsize=16, fontweight="bold", color="#1A1A2E", y=1.02)
    plt.tight_layout()
    out = ROOT / "models" / "evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    print(f"✓ Plot saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 6. SCORE DISTRIBUTION PLOT
# ─────────────────────────────────────────────────────────────
def plot_score_dist(best_model, X_test, y_test):
    probs  = best_model.predict_proba(X_test)[:, 1]
    scores = [prob_to_score(p) for p in probs]

    df_s = pd.DataFrame({"score": scores, "default": y_test.values})

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    ax.hist(df_s[df_s.default==0]["score"], bins=30, alpha=0.7,
            color="#2B6CB0", label="No Default", edgecolor="white")
    ax.hist(df_s[df_s.default==1]["score"], bins=30, alpha=0.7,
            color="#E03131", label="Default",    edgecolor="white")

    for x, label in [(580,"Poor"),(650,"Fair"),(700,"Good"),(750,"Excellent")]:
        ax.axvline(x, color="#555", linestyle="--", lw=1, alpha=0.6)
        ax.text(x+2, ax.get_ylim()[1]*0.9, label, fontsize=8, color="#555")

    ax.set_xlabel("Credit Score (300–850)")
    ax.set_ylabel("Count")
    ax.set_title("Credit Score Distribution by Outcome", fontweight="bold")
    ax.legend()
    out = ROOT / "models" / "score_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor="#FAFAFA")
    print(f"✓ Score dist saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n══ LendMN Credit Scoring Pipeline ══\n")

    # 1. Data
    X, y, le, feature_cols = load_and_preprocess(DATA / "credit_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    # 2. Train & Evaluate
    print("── Model Training & Evaluation ──")
    models  = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # 3. Best Model
    best_name  = max(results, key=lambda n: results[n]["auc"])
    best_model = results[best_name]["model"]
    print(f"\n✓ Best model: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

    # 4. Plots
    print("\n── Generating Plots ──")
    plot_results(results, X_test, y_test, feature_cols)
    plot_score_dist(best_model, X_test, y_test)

    # 5. Save artifacts
    joblib.dump(best_model,   MODELS / "best_model.joblib")
    joblib.dump(le,           MODELS / "label_encoder.joblib")
    joblib.dump(feature_cols, MODELS / "feature_cols.joblib")

    # Save metrics summary
    metrics = {
        name: {k: float(v) for k, v in res.items()
               if k not in ("model", "y_prob", "y_pred")}
        for name, res in results.items()
    }
    metrics["best_model"] = best_name
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Saved model artifacts → {MODELS}/")
    print("\n══ Pipeline Complete ══\n")

    # 6. Demo inference
    print("── Sample Predictions ──")
    sample = pd.DataFrame([{
        "age": 30, "employment_type_enc": le.transform(["private"])[0],
        "monthly_income": 1_500_000, "employment_months": 24,
        "num_existing_loans": 1, "credit_history_months": 36,
        "previous_defaults": 0, "num_dependents": 1,
        "loan_amount": 3_000_000, "debt_to_income": 0.10, "loan_to_income": 0.17,
    }])
    prob  = best_model.predict_proba(sample)[0][1]
    score = prob_to_score(prob)
    print(f"  Sample applicant → Default prob: {prob:.2%} | "
          f"Credit score: {score} ({score_band(score)})")


if __name__ == "__main__":
    main()
