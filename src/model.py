"""
Training and evaluation pipeline for an XGBoost binary classifier.

Pipeline steps:
    1. Load train/test datasets (parquet)
    2. Handle class imbalance via scale_pos_weight
    3. Use early stopping to find the optimal number of boosting rounds
    4. Tune hyperparameters with RandomizedSearchCV
    5. Evaluate the final model on the held-out test set
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_PATH = "../data/processed/train_dataset.parquet"
TEST_PATH = "../data/processed/test_dataset.parquet"

ID_COLS = ["venture_id", "cohort_year"]  
TARGET_COL = "target"
CLASS_LABELS = ["Dropped", "Mentored"]

RANDOM_STATE = 42
N_SPLITS_CV = 5
N_ITER_SEARCH = 60

PARAM_GRID = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "gamma": [0, 0.05, 0.1, 0.3],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
}


def load_data(train_path: str, test_path: str):
    """Load train/test parquet files and split into features/target."""
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    X_train = train.drop(columns=ID_COLS + [TARGET_COL])
    y_train = train[TARGET_COL]

    X_test = test.drop(columns=ID_COLS + [TARGET_COL])
    y_test = test[TARGET_COL]

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Positive rate (train): {y_train.mean():.2%}")
    print(f"Positive rate (test):  {y_test.mean():.2%}")

    return X_train, y_train, X_test, y_test


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute the class imbalance ratio used by XGBoost's scale_pos_weight."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    print(f"\nNegatives: {neg}  |  Positives: {pos}")
    print(f"scale_pos_weight: {spw:.2f}")

    return spw


def find_best_n_estimators(X_train, y_train, scale_pos_weight: float) -> int:
    """Use early stopping on a validation split to pick n_estimators."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    baseline_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        enable_categorical=True,
    )

    baseline_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    best_n_estimators = baseline_model.best_iteration + 1  # 0-indexed
    print(f"\n-> Best n_estimators: {best_n_estimators}")
    print(f"-> Validation AUC:    {baseline_model.best_score:.4f}")

    return best_n_estimators


def tune_hyperparameters(X_train, y_train, n_estimators: int, scale_pos_weight: float):
    """Run RandomizedSearchCV over the XGBoost hyperparameter grid."""
    tuning_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        enable_categorical=True,
    )

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=tuning_model,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        verbose=0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print("\n-> Best hyperparameters:")
    for param, value in search.best_params_.items():
        print(f"   {param:20s}: {value}")
    print(f"\n-> Mean CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_model(model, X_test, y_test, output_dir: str = "."):
    """Evaluate the final model and save ROC / confusion matrix / feature importance plots."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    print("\n=== FINAL RESULTS ===")
    print(f"AUC-ROC: {auc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[0])
    axes[0].set_title("ROC Curve - Test set")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=CLASS_LABELS,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Feature importance (top 20) - useful to see which feature blocks matter most
    xgb.plot_importance(model, max_num_features=20, importance_type="gain")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    return auc, y_pred_proba


def plot_threshold_analysis(y_test, y_pred_proba, output_dir: str = ".", n_thresholds: int = 200):
    """Plot TPR (recall on positive class) and TNR (specificity) across decision thresholds. Also prints a summary table around the TPR/TNR crossing point.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    tnr_list = []

    for t in thresholds:
        y_tmp = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_tmp).ravel()
        tpr_list.append(tp / (tp + fn))  # recall on "Mentored"
        tnr_list.append(tn / (tn + fp))  # recall on "Dropped" (specificity)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, tpr_list, label="TPR - Mentored (recall)", color="#1D9E75")
    plt.plot(thresholds, tnr_list, label="TNR - Dropped (specificity)", color="#E24B4A")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default threshold (0.5)")

    # Highlight the point where TPR and TNR cross
    diff = np.array(tpr_list) - np.array(tnr_list)
    crossing_idx = np.argmin(np.abs(diff))
    crossing_threshold = thresholds[crossing_idx]
    plt.axvline(
        crossing_threshold, color="orange", linestyle="--",
        label=f"Crossing point ({crossing_threshold:.2f})",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("TPR and TNR as a function of the decision threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_plot.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Summary table around the crossing point (focus on the 0.4-0.8 range)
    print(f"\n{'Threshold':>10} {'TPR':>8} {'TNR':>8}")
    print("-" * 30)
    for i, t in enumerate(thresholds):
        if 0.4 <= t <= 0.8 and i % 10 == 0:
            print(f"{t:>10.2f} {tpr_list[i]:>8.3f} {tnr_list[i]:>8.3f}")

    return crossing_threshold


def save_model(model, path: str = "xgboost_final.ubj"):
    """Save the trained model to disk in XGBoost's binary UBJ format."""
    model.save_model(path)
    print(f"Model saved -> {path}")


def main():
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    best_n_estimators = find_best_n_estimators(X_train, y_train, scale_pos_weight)

    best_model = tune_hyperparameters(
        X_train, y_train, best_n_estimators, scale_pos_weight
    )

    _, y_pred_proba = evaluate_model(best_model, X_test, y_test)

    plot_threshold_analysis(y_test, y_pred_proba)

    save_model(best_model)


if __name__ == "__main__":
    main()