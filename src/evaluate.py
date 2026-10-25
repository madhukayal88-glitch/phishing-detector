"""Metrics, plots, and reports."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay,
)
from src.config import PLOTS_DIR, REPORTS_DIR


def evaluate_model(model, X_test, y_test, tag="test"):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)

    n_classes = len(model.classes_)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if n_classes > 1:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        metrics["roc_auc"] = float("nan")

    print(f"\n[evaluate] {tag.upper()} METRICS")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit","Phish"], yticklabels=["Legit","Phish"])
    plt.title(f"Confusion Matrix ({tag})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{tag}.png", dpi=120)
    plt.close()

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve ({tag}) - AUC={metrics['roc_auc']:.3f}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"roc_curve_{tag}.png", dpi=120)
        plt.close()

    with open(REPORTS_DIR / f"metrics_{tag}.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n" + classification_report(y_test, y_pred, zero_division=0))
    return metrics

