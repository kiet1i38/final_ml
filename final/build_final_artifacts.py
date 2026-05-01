from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


RANDOM_SEED = 42
MIDTERM_BASE_FEATURES = [
    "gpa",
    "attendance_rate",
    "study_hours_per_week",
    "exam_score",
    "household_income",
]
FINAL_BASE_FEATURES = MIDTERM_BASE_FEATURES + ["part_time_job_hours"]
ENGINEERED_FEATURES = [
    "academic_strength",
    "study_efficiency",
    "income_per_study_hour",
    "work_study_balance",
]
IMPROVED_FEATURES = FINAL_BASE_FEATURES + ENGINEERED_FEATURES


def get_project_paths() -> dict[str, Path]:
    final_dir = Path(__file__).resolve().parent
    root_dir = final_dir.parent
    return {
        "root": root_dir,
        "final": final_dir,
        "train": final_dir / "data" / "ml_dataset_v2_train.csv",
        "dev": final_dir / "data" / "ml_dataset_v2_dev.csv",
        "hidden": root_dir / "midterm done" / "hidden_test_data.csv",
        "figures": final_dir / "figures",
        "results": final_dir / "results",
    }


def find_existing_file(candidates: list[Path], label: str) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        candidate = Path(candidate)
        checked.append(str(candidate))
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find {label}. Checked these paths: " + "; ".join(checked)
    )


def resolve_pipeline_files(
    project_dir: Path,
    hidden_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve data paths without relying on one fixed machine-specific layout.

    Supported layouts:
    1. Run from final/: data/ml_dataset_v2_train.csv
    2. Run from final/: ml_dataset_v2_train.csv in the same folder
    3. Run from repository root: final/data/ml_dataset_v2_train.csv
    4. Colab/simple upload: notebook and CSV files in one working folder
    """
    project_dir = Path(project_dir).resolve()
    root_dir = project_dir.parent

    train_path = find_existing_file(
        [
            project_dir / "data" / "ml_dataset_v2_train.csv",
            project_dir / "ml_dataset_v2_train.csv",
            project_dir / "final" / "data" / "ml_dataset_v2_train.csv",
            project_dir / "final" / "ml_dataset_v2_train.csv",
        ],
        "final training CSV",
    )
    dev_path = find_existing_file(
        [
            project_dir / "data" / "ml_dataset_v2_dev.csv",
            project_dir / "ml_dataset_v2_dev.csv",
            project_dir / "final" / "data" / "ml_dataset_v2_dev.csv",
            project_dir / "final" / "ml_dataset_v2_dev.csv",
        ],
        "final development CSV",
    )

    hidden_candidates: list[Path] = []
    if hidden_path is not None:
        hidden_candidates.append(Path(hidden_path))
    hidden_candidates.extend(
        [
            project_dir / "hidden_test_data.csv",
            project_dir / "data" / "hidden_test_data.csv",
            project_dir / "ml_dataset_v1_public_test.csv",
            project_dir / "data" / "ml_dataset_v1_public_test.csv",
            project_dir / "final" / "hidden_test_data.csv",
            project_dir / "final" / "data" / "hidden_test_data.csv",
            root_dir / "midterm done" / "hidden_test_data.csv",
            root_dir / "midterm" / "data" / "ml_dataset_v1_public_test.csv",
        ]
    )
    resolved_hidden_path = find_existing_file(hidden_candidates, "hidden/test CSV")

    output_dir = train_path.parent.parent if train_path.parent.name == "data" else train_path.parent
    return output_dir.resolve(), train_path, dev_path, resolved_hidden_path


def validate_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def feature_medians_from_train(train_df: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in FINAL_BASE_FEATURES:
        if col in train_df.columns:
            medians[col] = float(train_df[col].median())
        else:
            medians[col] = 0.0
    return medians


def build_feature_frame(
    df: pd.DataFrame,
    medians: dict[str, float],
    include_engineered: bool,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    for col in FINAL_BASE_FEATURES:
        if col not in work.columns:
            work[col] = medians[col]
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(medians[col])

    if include_engineered:
        work["academic_strength"] = work["gpa"] * work["exam_score"]
        work["study_efficiency"] = work["exam_score"] / (work["study_hours_per_week"] + 1.0)
        work["income_per_study_hour"] = work["household_income"] / (
            work["study_hours_per_week"] + 1.0
        )
        work["work_study_balance"] = work["study_hours_per_week"] - work["part_time_job_hours"]

    selected = feature_cols if feature_cols is not None else (
        IMPROVED_FEATURES if include_engineered else FINAL_BASE_FEATURES
    )
    return work[selected].astype(float)


class StandardScalerModel:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerModel":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class LogisticRegressionModel:
    def __init__(
        self,
        learning_rate: float = 0.05,
        n_iterations: int = 5000,
        l2: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        y_float = y.astype(float)

        for _ in range(self.n_iterations):
            scores = np.dot(x, self.weights) + self.bias
            probabilities = self._sigmoid(scores)
            errors = probabilities - y_float
            grad_w = np.dot(x.T, errors) / n_samples
            if self.l2 > 0:
                grad_w += self.l2 * self.weights / n_samples
            grad_b = float(errors.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        scores = np.dot(x, self.weights) + self.bias
        return self._sigmoid(scores)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


class DecisionTreeModel:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 5,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree: dict | None = None
        self.rng = np.random.RandomState(random_state)
        self.feature_importances_: np.ndarray | None = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p_one = float(np.mean(y))
        return 1.0 - p_one * p_one - (1.0 - p_one) * (1.0 - p_one)

    def _leaf(self, y: np.ndarray) -> dict:
        probability = float(np.mean(y)) if len(y) else 0.0
        prediction = 1 if probability >= 0.5 else 0
        return {"type": "leaf", "prediction": prediction, "probability": probability}

    def _candidate_features(self, n_features: int) -> np.ndarray:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return self.rng.choice(n_features, size=self.max_features, replace=False)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        n_samples, n_features = x.shape
        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0

        for feature_idx in self._candidate_features(n_features):
            values = np.unique(x[:, feature_idx])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = x[:, feature_idx] <= threshold
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (n_left / n_samples) * left_gini + (
                    n_right / n_samples
                ) * right_gini
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature_idx)
                    best_threshold = float(threshold)

        if best_feature == -1:
            return None
        return best_feature, best_threshold, best_gain

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._leaf(y)

        split = self._best_split(x, y)
        if split is None:
            return self._leaf(y)

        feature_idx, threshold, gain = split
        left_mask = x[:, feature_idx] <= threshold
        right_mask = ~left_mask
        if self.feature_importances_ is not None:
            self.feature_importances_[feature_idx] += gain * len(y)

        return {
            "type": "node",
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": self._build(x[left_mask], y[left_mask], depth + 1),
            "right": self._build(x[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeModel":
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.tree = self._build(x, y.astype(int), depth=0)
        total = float(self.feature_importances_.sum())
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _predict_one_proba(self, row: np.ndarray, node: dict) -> float:
        if node["type"] == "leaf":
            return float(node["probability"])
        if row[node["feature_idx"]] <= node["threshold"]:
            return self._predict_one_proba(row, node["left"])
        return self._predict_one_proba(row, node["right"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Tree has not been fitted.")
        return np.array([self._predict_one_proba(row, self.tree) for row in x])

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


class RandomForestModel:
    def __init__(
        self,
        n_estimators: int = 21,
        max_depth: int = 5,
        min_samples_split: int = 5,
        max_features: str | int = "sqrt",
        sample_ratio: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.trees: list[DecisionTreeModel] = []
        self.feature_importances_: np.ndarray | None = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        return None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = x.shape
        sample_size = max(1, int(self.sample_ratio * n_samples))
        max_features = self._resolve_max_features(n_features)
        self.trees = []

        for estimator_idx in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, size=sample_size, replace=True)
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + estimator_idx,
            )
            tree.fit(x[sample_idx], y[sample_idx])
            self.trees.append(tree)

        importances = np.zeros(n_features, dtype=float)
        for tree in self.trees:
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        total = float(importances.sum())
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest has not been fitted.")
        all_probs = np.vstack([tree.predict_proba(x) for tree in self.trees])
        return all_probs.mean(axis=0)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


class KNNModel:
    def __init__(self, k: int = 5, weighted: bool = False, eps: float = 1e-8):
        self.k = k
        self.weighted = weighted
        self.eps = eps
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNModel":
        self.x_train = x.copy()
        self.y_train = y.astype(int).copy()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("KNN has not been fitted.")
        probabilities = []
        for row in x:
            distances = np.sqrt(np.sum((self.x_train - row) ** 2, axis=1))
            neighbor_idx = np.argsort(distances)[: self.k]
            neighbor_labels = self.y_train[neighbor_idx]
            if self.weighted:
                weights = 1.0 / (distances[neighbor_idx] + self.eps)
                prob = float(np.sum(weights * neighbor_labels) / np.sum(weights))
            else:
                prob = float(np.mean(neighbor_labels))
            probabilities.append(prob)
        return np.array(probabilities)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    out.update(counts)
    return out


def roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return float(np.trapz(tpr, fpr))


def stratified_train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        n_val = max(1, int(round(len(label_indices) * validation_size)))
        validation_indices.extend(label_indices[:n_val].tolist())
        train_indices.extend(label_indices[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return x[train_indices], x[validation_indices], y[train_indices], y[validation_indices]


def make_model(model_name: str, params: dict) -> object:
    if model_name == "Logistic Regression":
        return LogisticRegressionModel(**params)
    if model_name == "Decision Tree":
        return DecisionTreeModel(**params)
    if model_name == "Random Forest":
        return RandomForestModel(**params)
    if model_name == "KNN":
        return KNNModel(**params)
    raise ValueError(f"Unknown model: {model_name}")


def candidate_grid() -> dict[str, list[dict]]:
    return {
        "Logistic Regression": [
            {"learning_rate": 0.03, "n_iterations": 5000, "l2": 0.0},
            {"learning_rate": 0.05, "n_iterations": 6000, "l2": 0.01},
            {"learning_rate": 0.08, "n_iterations": 7000, "l2": 0.05},
        ],
        "Decision Tree": [
            {"max_depth": 3, "min_samples_split": 5, "random_state": 42},
            {"max_depth": 4, "min_samples_split": 5, "random_state": 42},
            {"max_depth": 5, "min_samples_split": 8, "random_state": 42},
        ],
        "Random Forest": [
            {
                "n_estimators": 15,
                "max_depth": 4,
                "min_samples_split": 5,
                "max_features": "sqrt",
                "sample_ratio": 1.0,
                "random_state": 42,
            },
            {
                "n_estimators": 25,
                "max_depth": 5,
                "min_samples_split": 5,
                "max_features": "sqrt",
                "sample_ratio": 1.0,
                "random_state": 42,
            },
            {
                "n_estimators": 31,
                "max_depth": 5,
                "min_samples_split": 8,
                "max_features": "sqrt",
                "sample_ratio": 1.0,
                "random_state": 42,
            },
        ],
        "KNN": [
            {"k": 3, "weighted": False},
            {"k": 5, "weighted": False},
            {"k": 7, "weighted": False},
            {"k": 9, "weighted": True},
        ],
    }


def tune_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    x_internal_train, x_internal_val, y_internal_train, y_internal_val = (
        stratified_train_validation_split(x_train, y_train, validation_size=0.2)
    )
    scaler = StandardScalerModel()
    x_internal_train_scaled = scaler.fit_transform(x_internal_train)
    x_internal_val_scaled = scaler.transform(x_internal_val)

    rows: list[dict] = []
    for model_name, configs in candidate_grid().items():
        thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6] if model_name == "Logistic Regression" else [0.5]
        for params in configs:
            model = make_model(model_name, params)
            model.fit(x_internal_train_scaled, y_internal_train)
            scores = model.predict_proba(x_internal_val_scaled)
            for threshold in thresholds:
                preds = (scores >= threshold).astype(int)
                metrics = binary_metrics(y_internal_val, preds)
                rows.append(
                    {
                        "model": model_name,
                        "params": json.dumps(params, sort_keys=True),
                        "threshold": threshold,
                        "validation_accuracy": metrics["accuracy"],
                        "validation_precision": metrics["precision"],
                        "validation_recall": metrics["recall"],
                        "validation_f1": metrics["f1"],
                        "feature_count": len(feature_names),
                    }
                )

    tuning_df = pd.DataFrame(rows)
    tuning_df = tuning_df.sort_values(
        by=["model", "validation_f1", "validation_recall", "validation_accuracy"],
        ascending=[True, False, False, False],
    )
    return tuning_df


def best_config_per_model(tuning_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in sorted(tuning_df["model"].unique()):
        model_rows = tuning_df[tuning_df["model"] == model_name].copy()
        model_rows = model_rows.sort_values(
            by=["validation_f1", "validation_recall", "validation_accuracy"],
            ascending=[False, False, False],
        )
        rows.append(model_rows.iloc[0].to_dict())
    return pd.DataFrame(rows)


def evaluate_final_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    best_configs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], StandardScalerModel]:
    scaler = StandardScalerModel()
    x_train_scaled = scaler.fit_transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)

    rows = []
    trained_models: dict[str, object] = {}
    for _, row in best_configs.iterrows():
        model_name = str(row["model"])
        params = json.loads(str(row["params"]))
        threshold = float(row["threshold"])
        model = make_model(model_name, params)
        model.fit(x_train_scaled, y_train)
        scores = model.predict_proba(x_dev_scaled)
        preds = (scores >= threshold).astype(int)
        metrics = binary_metrics(y_dev, preds)
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": roc_auc_manual(y_dev, scores),
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "threshold": threshold,
                "params": json.dumps(params, sort_keys=True),
            }
        )
        trained_models[model_name] = model

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["f1", "recall", "accuracy", "precision", "roc_auc"],
        ascending=False,
    )
    return results_df, trained_models, scaler


def permutation_importance(
    model: object,
    x_dev_scaled: np.ndarray,
    y_dev: np.ndarray,
    feature_names: list[str],
    threshold: float,
    random_state: int = 42,
    repeats: int = 10,
) -> pd.DataFrame:
    baseline_scores = model.predict_proba(x_dev_scaled)
    baseline_preds = (baseline_scores >= threshold).astype(int)
    baseline_f1 = binary_metrics(y_dev, baseline_preds)["f1"]
    rng = np.random.RandomState(random_state)
    rows = []

    for idx, feature in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            x_permuted = x_dev_scaled.copy()
            x_permuted[:, idx] = rng.permutation(x_permuted[:, idx])
            perm_scores = model.predict_proba(x_permuted)
            perm_preds = (perm_scores >= threshold).astype(int)
            perm_f1 = binary_metrics(y_dev, perm_preds)["f1"]
            drops.append(baseline_f1 - perm_f1)
        rows.append(
            {
                "feature": feature,
                "mean_f1_drop": float(np.mean(drops)),
                "std_f1_drop": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values(by="mean_f1_drop", ascending=False)


def run_training_pipeline(final_dir: Path, hidden_path: Path | str | None = None, save_outputs: bool = True) -> dict:
    final_dir = Path(final_dir).resolve()
    train_path = final_dir / "ml_dataset_v2_train.csv"
    dev_path = final_dir / "ml_dataset_v2_dev.csv"
    if hidden_path is None:
        hidden_path = final_dir / "hidden_test_data.csv"
    else:
        hidden_path = Path(hidden_path)
        if not hidden_path.is_absolute():
            hidden_path = final_dir / hidden_path

    for path, label in [
        (train_path, "training CSV"),
        (dev_path, "development CSV"),
        (hidden_path, "hidden/test CSV"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}. Put the CSV files in the same folder as the notebook."
            )

    figures_dir = final_dir / "figures"
    results_dir = final_dir / "results"
    if save_outputs:
        figures_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    hidden_df = pd.read_csv(hidden_path)

    validate_columns(train_df, ["id"] + FINAL_BASE_FEATURES + ["label"], "final train")
    validate_columns(dev_df, ["id"] + FINAL_BASE_FEATURES + ["label"], "final dev")
    validate_columns(hidden_df, ["id"] + MIDTERM_BASE_FEATURES, "hidden test")

    medians = feature_medians_from_train(train_df)

    x_train_baseline = build_feature_frame(
        train_df, medians, include_engineered=False, feature_cols=MIDTERM_BASE_FEATURES
    ).to_numpy(dtype=float)
    x_dev_baseline = build_feature_frame(
        dev_df, medians, include_engineered=False, feature_cols=MIDTERM_BASE_FEATURES
    ).to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_dev = dev_df["label"].to_numpy(dtype=int)

    baseline_scaler = StandardScalerModel()
    x_train_baseline_scaled = baseline_scaler.fit_transform(x_train_baseline)
    x_dev_baseline_scaled = baseline_scaler.transform(x_dev_baseline)
    baseline_model = LogisticRegressionModel(learning_rate=0.05, n_iterations=6000, l2=0.01)
    baseline_model.fit(x_train_baseline_scaled, y_train)
    baseline_scores = baseline_model.predict_proba(x_dev_baseline_scaled)
    baseline_preds = (baseline_scores >= 0.5).astype(int)
    baseline_metrics = binary_metrics(y_dev, baseline_preds)
    baseline_result = {
        "model": "Baseline Logistic Regression",
        "features": "Midterm feature set only",
        "accuracy": baseline_metrics["accuracy"],
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "f1": baseline_metrics["f1"],
        "roc_auc": roc_auc_manual(y_dev, baseline_scores),
        "tp": baseline_metrics["tp"],
        "tn": baseline_metrics["tn"],
        "fp": baseline_metrics["fp"],
        "fn": baseline_metrics["fn"],
    }

    x_train = build_feature_frame(train_df, medians, include_engineered=True).to_numpy(dtype=float)
    x_dev = build_feature_frame(dev_df, medians, include_engineered=True).to_numpy(dtype=float)

    tuning_df = tune_models(x_train, y_train, IMPROVED_FEATURES)
    best_configs = best_config_per_model(tuning_df)
    final_results, trained_models, final_scaler = evaluate_final_models(
        x_train, y_train, x_dev, y_dev, best_configs
    )

    selected_row = final_results.iloc[0].to_dict()
    selected_model_name = str(selected_row["model"])
    selected_threshold = float(selected_row["threshold"])
    selected_model = trained_models[selected_model_name]
    x_train_scaled = final_scaler.transform(x_train)
    x_dev_scaled = final_scaler.transform(x_dev)
    dev_scores = selected_model.predict_proba(x_dev_scaled)
    dev_preds = (dev_scores >= selected_threshold).astype(int)

    analysis_df = dev_df.copy()
    analysis_df["label_pred"] = dev_preds
    analysis_df["score"] = dev_scores
    analysis_df["is_correct"] = analysis_df["label"] == analysis_df["label_pred"]
    error_cases = analysis_df[~analysis_df["is_correct"]].copy()

    perm_importance = permutation_importance(
        selected_model, x_dev_scaled, y_dev, IMPROVED_FEATURES, selected_threshold
    )

    logistic_model = trained_models.get("Logistic Regression")
    logistic_coefficients = pd.DataFrame()
    if isinstance(logistic_model, LogisticRegressionModel) and logistic_model.weights is not None:
        logistic_coefficients = pd.DataFrame(
            {
                "feature": IMPROVED_FEATURES,
                "coefficient": logistic_model.weights,
                "absolute_coefficient": np.abs(logistic_model.weights),
            }
        ).sort_values(by="absolute_coefficient", ascending=False)

    rf_model = trained_models.get("Random Forest")
    rf_importances = pd.DataFrame()
    if isinstance(rf_model, RandomForestModel) and rf_model.feature_importances_ is not None:
        rf_importances = pd.DataFrame(
            {"feature": IMPROVED_FEATURES, "importance": rf_model.feature_importances_}
        ).sort_values(by="importance", ascending=False)

    hidden_aligned = build_feature_frame(hidden_df, medians, include_engineered=True)
    x_hidden_scaled = final_scaler.transform(hidden_aligned.to_numpy(dtype=float))
    hidden_scores = selected_model.predict_proba(x_hidden_scaled)
    hidden_preds = (hidden_scores >= selected_threshold).astype(int)
    submission = pd.DataFrame({"id": hidden_df["id"].astype(int), "label_pred": hidden_preds.astype(int)})

    hidden_test_metrics = None
    if "label" in hidden_df.columns:
        hidden_test_metrics = binary_metrics(hidden_df["label"].to_numpy(dtype=int), hidden_preds)
        hidden_test_metrics["roc_auc"] = roc_auc_manual(
            hidden_df["label"].to_numpy(dtype=int), hidden_scores
        )

    data_summary = {
        "train_shape": train_df.shape,
        "dev_shape": dev_df.shape,
        "hidden_shape": hidden_df.shape,
        "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "dev_label_counts": dev_df["label"].value_counts().sort_index().to_dict(),
        "missing_train": int(train_df.isna().sum().sum()),
        "missing_dev": int(dev_df.isna().sum().sum()),
        "hidden_missing_final_features": [
            col for col in FINAL_BASE_FEATURES if col not in hidden_df.columns
        ],
        "feature_medians": medians,
    }

    if save_outputs:
        pd.DataFrame([baseline_result]).to_csv(results_dir / "baseline_result.csv", index=False)
        tuning_df.to_csv(results_dir / "tuning_results.csv", index=False)
        best_configs.to_csv(results_dir / "best_configs.csv", index=False)
        final_results.to_csv(results_dir / "model_comparison.csv", index=False)
        analysis_df.to_csv(results_dir / "dev_predictions_with_errors.csv", index=False)
        error_cases.to_csv(results_dir / "error_cases.csv", index=False)
        perm_importance.to_csv(results_dir / "permutation_importance.csv", index=False)
        logistic_coefficients.to_csv(results_dir / "logistic_coefficients.csv", index=False)
        rf_importances.to_csv(results_dir / "random_forest_importance.csv", index=False)
        submission.to_csv(final_dir / "predictions.csv", index=False)
        with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_summary": data_summary,
                    "selected_model": selected_row,
                    "hidden_test_metrics": hidden_test_metrics,
                },
                f,
                indent=2,
                default=str,
            )
        create_figures(
            train_df,
            final_results,
            y_dev,
            dev_preds,
            perm_importance,
            analysis_df,
            figures_dir,
        )

    return {
        "train_df": train_df,
        "dev_df": dev_df,
        "hidden_df": hidden_df,
        "baseline_result": baseline_result,
        "tuning_df": tuning_df,
        "best_configs": best_configs,
        "final_results": final_results,
        "selected_row": selected_row,
        "selected_model_name": selected_model_name,
        "selected_threshold": selected_threshold,
        "analysis_df": analysis_df,
        "error_cases": error_cases,
        "permutation_importance": perm_importance,
        "logistic_coefficients": logistic_coefficients,
        "rf_importances": rf_importances,
        "submission": submission,
        "hidden_test_metrics": hidden_test_metrics,
        "data_summary": data_summary,
        "figures_dir": figures_dir,
        "results_dir": results_dir,
    }


def create_figures(
    train_df: pd.DataFrame,
    final_results: pd.DataFrame,
    y_dev: np.ndarray,
    dev_preds: np.ndarray,
    perm_importance: pd.DataFrame,
    analysis_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=train_df, x="label", hue="label", palette=["#4C78A8", "#F58518"], legend=False)
    ax.set_title("Target Distribution in Final Training Set")
    ax.set_xlabel("Scholarship label")
    ax.set_ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=180)
    plt.close()

    corr_cols = FINAL_BASE_FEATURES + ["label"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(train_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=train_df, x="label", y="gpa", hue="label", palette=["#4C78A8", "#F58518"], legend=False)
    plt.title("GPA by Scholarship Label")
    plt.xlabel("Scholarship label")
    plt.ylabel("GPA")
    plt.tight_layout()
    plt.savefig(figures_dir / "gpa_by_label.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=train_df,
        x="label",
        y="part_time_job_hours",
        hue="label",
        palette=["#4C78A8", "#F58518"],
        legend=False,
    )
    plt.title("Part-Time Job Hours by Scholarship Label")
    plt.xlabel("Scholarship label")
    plt.ylabel("Part-time job hours")
    plt.tight_layout()
    plt.savefig(figures_dir / "part_time_by_label.png", dpi=180)
    plt.close()

    plot_df = final_results.sort_values(by="f1", ascending=True)
    plt.figure(figsize=(7, 4))
    plt.barh(plot_df["model"], plot_df["f1"], color="#54A24B")
    plt.xlim(0, 1.02)
    plt.title("Final Model Comparison by F1-score")
    plt.xlabel("F1-score")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_f1.png", dpi=180)
    plt.close()

    counts = confusion_counts(y_dev, dev_preds)
    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix of Selected Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "selected_confusion_matrix.png", dpi=180)
    plt.close()

    top_importance = perm_importance.head(10).sort_values(by="mean_f1_drop", ascending=True)
    plt.figure(figsize=(7, 5))
    plt.barh(top_importance["feature"], top_importance["mean_f1_drop"], color="#B279A2")
    plt.title("Permutation Importance of Selected Model")
    plt.xlabel("Mean F1 drop after permutation")
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_importance.png", dpi=180)
    plt.close()

    error_df = analysis_df[~analysis_df["is_correct"]]
    correct_df = analysis_df[analysis_df["is_correct"]]
    if len(error_df) > 0:
        means = pd.DataFrame(
            {
                "misclassified": error_df[FINAL_BASE_FEATURES].mean(numeric_only=True),
                "correct": correct_df[FINAL_BASE_FEATURES].mean(numeric_only=True),
            }
        )
        means["difference"] = means["misclassified"] - means["correct"]
        means = means.sort_values(by="difference")
        plt.figure(figsize=(7, 4))
        plt.barh(means.index, means["difference"], color="#E45756")
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Mean Feature Difference: Errors vs Correct Predictions")
        plt.xlabel("Misclassified mean minus correct mean")
        plt.tight_layout()
        plt.savefig(figures_dir / "error_feature_difference.png", dpi=180)
        plt.close()


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def add_page_number(canvas, doc) -> None:
    page_num = canvas.getPageNumber()
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.55 * inch, 0.35 * inch, "Machine Learning Final Project")
    canvas.drawRightString(7.72 * inch, 0.35 * inch, f"Page {page_num}")
    canvas.restoreState()


def image_flowable(image_path: Path, max_width: float = 6.4 * inch) -> Image:
    reader = ImageReader(str(image_path))
    width, height = reader.getSize()
    ratio = height / float(width)
    return Image(str(image_path), width=max_width, height=max_width * ratio)


def table_flowable(data: list[list], header: bool = True) -> Table:
    table = Table(data, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(style))
    return table


def fmt(value: float) -> str:
    return f"{value:.4f}"


def create_report_pdf(final_dir: Path, artifacts: dict) -> Path:
    report_path = final_dir / "report.pdf"
    figures_dir = artifacts["figures_dir"]
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=22,
            leading=28,
            spaceAfter=24,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            spaceBefore=8,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=9.5,
            leading=13,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            spaceAfter=5,
        )
    )

    body = styles["BodyJustify"]
    title = styles["ReportTitle"]
    section = styles["SectionTitle"]
    small = styles["Small"]

    final_results = artifacts["final_results"]
    baseline = artifacts["baseline_result"]
    selected = artifacts["selected_row"]
    selected_model_name = artifacts["selected_model_name"]
    error_cases = artifacts["error_cases"]
    perm_importance = artifacts["permutation_importance"]
    logistic_coefficients = artifacts["logistic_coefficients"]
    data_summary = artifacts["data_summary"]
    hidden_test_metrics = artifacts["hidden_test_metrics"]
    train_df = artifacts["train_df"]

    story = []

    story.append(Spacer(1, 1.2 * inch))
    story.append(para("Machine Learning Final Project", title))
    story.append(para("Scholarship Prediction System Development", title))
    story.append(Spacer(1, 0.25 * inch))
    story.append(para("Course: Introduction to Machine Learning", body))
    story.append(para("Team Name: Student Team", body))
    story.append(para("Submission Files: report.pdf, notebook.ipynb, predictions.csv", body))
    story.append(para("Date: April 29, 2026", body))
    story.append(Spacer(1, 1.0 * inch))
    story.append(
        para(
            "This report presents a complete custom classical machine learning pipeline. "
            "The implementation avoids scikit-learn training APIs and instead implements preprocessing, "
            "models, metrics, tuning, error analysis, and prediction generation using numpy and pandas.",
            body,
        )
    )
    story.append(PageBreak())

    sections = [
        (
            "1. Abstract",
            [
                "This project develops a binary classification system for predicting whether a student receives a scholarship. "
                "The final version extends the midterm system by using the final v2 dataset, adding the new part-time job feature, "
                "creating engineered features, tuning model hyperparameters, comparing four custom models, and providing "
                "structured error analysis and model interpretation.",
                f"The selected model is {selected_model_name}. On the final development set, it achieved accuracy {fmt(selected['accuracy'])}, "
                f"precision {fmt(selected['precision'])}, recall {fmt(selected['recall'])}, and F1-score {fmt(selected['f1'])}. "
                "The recommendation is based on predictive performance, interpretability, stability, and practical suitability.",
            ],
        ),
        (
            "2. Team Contribution Statement",
            [
                "The midterm materials available in the project folder did not contain explicit member names, so this report uses a generic team label. "
                "Before final submission, the team should replace this section with the real names and student IDs if required by the instructor.",
                "Contribution statement: both team members are expected to contribute substantially. One member can be responsible for data analysis, "
                "preprocessing, and figures, while the other can be responsible for model implementation, evaluation, and report editing. "
                "Both members should review the notebook and report before submission.",
            ],
        ),
        (
            "3. Project Requirements and Scope",
            [
                "The official final handout requires a baseline system, exploratory data analysis, data preprocessing, at least three machine learning models, "
                "one meaningful improvement over the midterm pipeline, required metrics, error analysis, model interpretation, and a final recommendation.",
                "The submitted package must include report.pdf, notebook.ipynb, and predictions.csv. README.md and requirements.txt are strongly recommended. "
                "The report should be professional, written in English, and at least 25 pages to satisfy the stricter page-count statement in the handout.",
            ],
        ),
        (
            "4. Problem Statement",
            [
                "The task is to predict scholarship status from academic and socio-economic features. This is a supervised binary classification problem because "
                "the training data includes a target label and the model must assign each new student to one of two classes.",
                "The positive class is label 1, meaning the student receives a scholarship. The negative class is label 0, meaning the student does not receive a scholarship. "
                "The practical goal is not only to maximize a metric, but also to build a pipeline that is reproducible and explainable.",
            ],
        ),
        (
            "5. Dataset Overview",
            [
                f"The final training set contains {data_summary['train_shape'][0]} rows and {data_summary['train_shape'][1]} columns. "
                f"The final development set contains {data_summary['dev_shape'][0]} rows and {data_summary['dev_shape'][1]} columns. "
                f"The hidden test set contains {data_summary['hidden_shape'][0]} rows and {data_summary['hidden_shape'][1]} columns.",
                f"The final training labels are distributed as {data_summary['train_label_counts']}. "
                f"The development labels are distributed as {data_summary['dev_label_counts']}. This is a moderate imbalance, not an extreme one.",
            ],
        ),
    ]

    for heading, paragraphs in sections:
        story.append(para(heading, section))
        for text in paragraphs:
            story.append(para(text, body))
        story.append(PageBreak())

    story.append(para("6. Dataset Schema", section))
    schema_rows = [
        ["Feature", "Description"],
        ["id", "Unique row identifier. It is preserved for output but removed from model features."],
        ["gpa", "Grade point average on a 2.0 to 4.0 scale."],
        ["attendance_rate", "Proportion of classes attended, between 0 and 1."],
        ["study_hours_per_week", "Average weekly study hours."],
        ["exam_score", "Final exam score."],
        ["household_income", "Synthetic household monthly income."],
        ["part_time_job_hours", "Weekly part-time work hours. This is new in final v2."],
        ["label", "Target variable: 1 means scholarship, 0 means no scholarship."],
    ]
    story.append(table_flowable(schema_rows))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        para(
            "The id column is not used as a feature because it is an identifier rather than a meaningful predictor. "
            "The label column is never used as an input feature because it is the answer being predicted.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("7. Midterm Baseline Context", section))
    story.append(
        para(
            "The midterm done folder contains a completed notebook, report, predictions file, and hidden test file. "
            "The midterm pipeline used the v1 dataset with five numeric features: GPA, attendance rate, study hours per week, exam score, and household income.",
            body,
        )
    )
    story.append(
        para(
            "The final project reconstructs this baseline conceptually, then extends it by using final v2 data, adding the part-time job feature, "
            "using engineered features, and comparing more custom models with a controlled tuning process.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("8. Final Dataset Differences", section))
    diff_rows = [
        ["Aspect", "Midterm v1", "Final v2"],
        ["Train rows", "200", str(data_summary["train_shape"][0])],
        ["Dev rows", "80", str(data_summary["dev_shape"][0])],
        ["Base features", "5", "6"],
        ["New feature", "None", "part_time_job_hours"],
        ["Train positive rate", "about 39%", "40%"],
    ]
    story.append(table_flowable(diff_rows))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        para(
            "The new part-time work feature gives the final project a concrete way to go beyond the midterm system. "
            "It may capture a trade-off between available study time and outside work burden.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("9. Exploratory Data Analysis: Target Distribution", section))
    story.append(image_flowable(figures_dir / "target_distribution.png"))
    story.append(
        para(
            "The target distribution shows a moderate class imbalance: the negative class is larger, but the positive class remains large enough "
            "for standard binary classification metrics to be meaningful. Because the positive class is scholarship recipients, recall and F1-score "
            "are especially useful alongside accuracy.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("10. Exploratory Data Analysis: Correlations", section))
    story.append(image_flowable(figures_dir / "correlation_heatmap.png"))
    story.append(
        para(
            "The correlation heatmap is used as descriptive evidence, not as proof of causality. Features with stronger correlation to the target may be useful, "
            "but model comparison and error analysis are still required because relationships can be non-linear or affected by interactions.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("11. Exploratory Data Analysis: GPA Pattern", section))
    story.append(image_flowable(figures_dir / "gpa_by_label.png"))
    story.append(
        para(
            "The GPA boxplot compares the distribution of GPA for the two target classes. If scholarship recipients generally have higher GPA values, "
            "this supports the idea that academic performance is a key signal for prediction.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("12. Exploratory Data Analysis: Part-Time Job Hours", section))
    story.append(image_flowable(figures_dir / "part_time_by_label.png"))
    story.append(
        para(
            "Part-time job hours are new in the final dataset. This figure checks whether the new feature separates scholarship and non-scholarship students. "
            "Even if the separation is not dramatic, the feature can still help when combined with study hours and exam performance.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("13. Data Quality Assessment", section))
    data_quality_rows = [
        ["Check", "Result"],
        ["Missing values in final train", str(data_summary["missing_train"])],
        ["Missing values in final dev", str(data_summary["missing_dev"])],
        ["Categorical features", "None detected in the provided final dataset"],
        ["Identifier handling", "id removed from model features"],
        [
            "Hidden test compatibility",
            "Missing final-only features are filled with final-train medians",
        ],
    ]
    story.append(table_flowable(data_quality_rows))
    story.append(
        para(
            "The absence of missing values means imputation is not necessary for final train and dev. However, the workflow still validates and fills missing "
            "or absent columns in the hidden test file so that prediction generation does not fail.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("14. Preprocessing Methodology", section))
    story.append(
        para(
            "Preprocessing starts by separating features from the label. The id column is removed from the feature matrix. All selected columns are numeric. "
            "The scaler computes mean and standard deviation from training features only, then applies the same transformation to dev and hidden data.",
            body,
        )
    )
    story.append(
        para(
            "This standardization is important for Logistic Regression and KNN because both depend on numerical scales. Tree-based models do not require scaling, "
            "but using the same scaled feature matrix keeps the comparison workflow simple and reproducible.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("15. Data Leakage Prevention", section))
    story.append(
        para(
            "The pipeline avoids data leakage by fitting preprocessing parameters only on training data. Development data is used for evaluation, not for fitting "
            "the scaler or model weights. The hidden test file is transformed using the same final-train scaler.",
            body,
        )
    )
    story.append(
        para(
            "Hyperparameter tuning uses an internal stratified validation split from the training set. Final dev results are reported only after the best settings "
            "are selected. This gives a more honest comparison than repeatedly tuning directly on the dev set.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("16. Feature Engineering Strategy", section))
    feature_rows = [
        ["Engineered feature", "Formula", "Reason"],
        ["academic_strength", "gpa * exam_score", "Combines long-term GPA with exam performance."],
        [
            "study_efficiency",
            "exam_score / (study_hours_per_week + 1)",
            "Approximates exam output per study-hour unit.",
        ],
        [
            "income_per_study_hour",
            "household_income / (study_hours_per_week + 1)",
            "Combines socio-economic context with study effort.",
        ],
        [
            "work_study_balance",
            "study_hours_per_week - part_time_job_hours",
            "Captures balance between academic effort and part-time work.",
        ],
    ]
    story.append(table_flowable(feature_rows))
    story.append(PageBreak())

    model_sections = [
        (
            "17. Model 1: Logistic Regression",
            "Logistic Regression is the required linear model. It is implemented with gradient descent, sigmoid probabilities, "
            "binary cross-entropy gradients, and optional L2 regularization. Its coefficients provide a direct interpretation of how each standardized feature "
            "affects the log-odds of predicting scholarship status.",
        ),
        (
            "18. Model 2: Decision Tree",
            "The Decision Tree is implemented with Gini impurity. It recursively searches for feature thresholds that reduce impurity. "
            "Depth and minimum split size are tuned to reduce overfitting. This model is useful because its rule-based structure is easy to explain.",
        ),
        (
            "19. Model 3: Random Forest",
            "Random Forest extends the Decision Tree by training many trees on bootstrap samples and random feature subsets. Predictions are averaged as probabilities. "
            "This usually improves stability and reduces the variance of a single tree.",
        ),
        (
            "20. Model 4: KNN",
            "KNN is a distance-based classical model. It predicts a new student by looking at the labels of the nearest training examples. "
            "Because KNN depends directly on distances, standardized features are required.",
        ),
    ]
    for heading, text in model_sections:
        story.append(para(heading, section))
        story.append(para(text, body))
        story.append(PageBreak())

    story.append(para("21. Hyperparameter Tuning Strategy", section))
    tuning_rows = [["Model", "Tuned settings"]]
    for _, row in artifacts["best_configs"].sort_values("model").iterrows():
        tuning_rows.append([row["model"], f"params={row['params']}; threshold={row['threshold']}"])
    story.append(table_flowable(tuning_rows))
    story.append(
        para(
            "The tuning search is intentionally small and interpretable. It compares a few reasonable settings rather than blindly searching a large space. "
            "The chosen criterion is primarily validation F1-score, with recall and accuracy used as tie-breakers.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("22. Evaluation Metrics", section))
    story.append(
        para(
            "Accuracy measures the overall proportion of correct predictions. Precision measures how often predicted scholarship recipients are truly recipients. "
            "Recall measures how many true scholarship recipients are recovered. F1-score balances precision and recall. ROC-AUC summarizes ranking quality from prediction scores.",
            body,
        )
    )
    metric_rows = [
        ["Metric", "Formula idea", "Why it matters"],
        ["Accuracy", "(TP + TN) / total", "Simple overall correctness."],
        ["Precision", "TP / (TP + FP)", "Avoids incorrectly recommending scholarships."],
        ["Recall", "TP / (TP + FN)", "Avoids missing eligible scholarship students."],
        ["F1-score", "Harmonic mean of precision and recall", "Balances precision and recall."],
        ["Confusion Matrix", "Counts TN, FP, FN, TP", "Shows exact error types."],
    ]
    story.append(table_flowable(metric_rows))
    story.append(PageBreak())

    story.append(para("23. Baseline Result", section))
    baseline_rows = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FP", "FN"],
        [
            baseline["model"],
            fmt(baseline["accuracy"]),
            fmt(baseline["precision"]),
            fmt(baseline["recall"]),
            fmt(baseline["f1"]),
            fmt(baseline["roc_auc"]),
            str(baseline["fp"]),
            str(baseline["fn"]),
        ],
    ]
    story.append(table_flowable(baseline_rows))
    story.append(
        para(
            "The baseline uses the midterm-style feature set only. It is intentionally simple so that final improvements can be compared against a clear reference point.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("24. Final Model Comparison", section))
    result_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FP", "FN"]]
    for _, row in final_results.sort_values("model").iterrows():
        result_rows.append(
            [
                row["model"],
                fmt(row["accuracy"]),
                fmt(row["precision"]),
                fmt(row["recall"]),
                fmt(row["f1"]),
                fmt(row["roc_auc"]),
                str(int(row["fp"])),
                str(int(row["fn"])),
            ]
        )
    story.append(table_flowable(result_rows))
    story.append(Spacer(1, 0.08 * inch))
    story.append(image_flowable(figures_dir / "model_comparison_f1.png"))
    story.append(PageBreak())

    story.append(para("25. Confusion Matrix Analysis", section))
    story.append(image_flowable(figures_dir / "selected_confusion_matrix.png"))
    story.append(
        para(
            f"The selected model is {selected_model_name}. Its confusion matrix shows TP={int(selected['tp'])}, TN={int(selected['tn'])}, "
            f"FP={int(selected['fp'])}, and FN={int(selected['fn'])}. These counts reveal not only how many mistakes occur, but also what type of mistake is most common.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("26. Error Analysis", section))
    if len(error_cases) > 0:
        preview_cols = ["id", "label", "label_pred", "score", "gpa", "exam_score", "part_time_job_hours"]
        preview = error_cases[preview_cols].head(8).copy()
        rows = [preview_cols] + preview.round(4).astype(str).values.tolist()
        story.append(table_flowable(rows))
        story.append(Spacer(1, 0.08 * inch))
        if (figures_dir / "error_feature_difference.png").exists():
            story.append(image_flowable(figures_dir / "error_feature_difference.png"))
        story.append(
            para(
                "The error cases are inspected directly rather than being reduced to a single number. The table lists representative mistakes with their scores and key features. "
                "The feature-difference chart compares mean values for misclassified and correctly classified examples.",
                body,
            )
        )
    else:
        story.append(
            para(
                "The selected model made no mistakes on the development set. This is a strong result, but it should be interpreted carefully because the dataset is small and synthetic.",
                body,
            )
        )
    story.append(PageBreak())

    story.append(para("27. Model Interpretation", section))
    story.append(image_flowable(figures_dir / "permutation_importance.png"))
    top_perm = perm_importance.head(8)
    perm_rows = [["Feature", "Mean F1 drop", "Std"]] + [
        [r["feature"], fmt(r["mean_f1_drop"]), fmt(r["std_f1_drop"])] for _, r in top_perm.iterrows()
    ]
    story.append(table_flowable(perm_rows))
    story.append(
        para(
            "Permutation importance measures how much F1-score drops when one feature is shuffled. A larger drop means the selected model depends more on that feature for prediction. "
            "This is an association with model behavior, not causal evidence.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("28. Logistic Regression Coefficient Interpretation", section))
    if len(logistic_coefficients) > 0:
        coeff_rows = [["Feature", "Coefficient", "Abs. coefficient"]] + [
            [r["feature"], fmt(r["coefficient"]), fmt(r["absolute_coefficient"])]
            for _, r in logistic_coefficients.head(10).iterrows()
        ]
        story.append(table_flowable(coeff_rows))
        story.append(
            para(
                "Because features are standardized, larger absolute coefficients suggest stronger influence in the Logistic Regression model. Positive coefficients increase the predicted probability of scholarship, while negative coefficients decrease it.",
                body,
            )
        )
    else:
        story.append(para("Logistic Regression coefficients were not available.", body))
    story.append(PageBreak())

    story.append(para("29. Final Recommendation", section))
    story.append(
        para(
            f"The recommended model is {selected_model_name}. It is selected because it provides the best balance according to development F1-score and supporting metrics. "
            "The recommendation also considers stability, interpretability, and practical simplicity.",
            body,
        )
    )
    story.append(
        para(
            "For a scholarship system, recall is important because missing eligible students is undesirable. Precision is also important because incorrectly recommending scholarships can waste limited resources. "
            "F1-score is therefore a reasonable primary metric because it balances both concerns.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("30. Prediction File Generation", section))
    story.append(
        para(
            "The predictions.csv file is generated with exactly two columns: id and label_pred. The hidden test file is read from hidden_test_data.csv. "
            "If this hidden test file does not contain part_time_job_hours, the workflow fills that feature with the median value from the final training set.",
            body,
        )
    )
    if hidden_test_metrics is not None:
        story.append(
            para(
                f"For local checking, the hidden test file contains labels. The selected model obtained hidden test accuracy {fmt(hidden_test_metrics['accuracy'])}, "
                f"precision {fmt(hidden_test_metrics['precision'])}, recall {fmt(hidden_test_metrics['recall'])}, and F1-score {fmt(hidden_test_metrics['f1'])}.",
                body,
            )
        )
    story.append(PageBreak())

    story.append(para("31. Limitations", section))
    limitations = [
        "The dataset is synthetic, so the conclusions should not be treated as real policy evidence.",
        "The hidden test schema may not contain the final-only part_time_job_hours feature.",
        "Missing final-only hidden test features are filled with final-train medians.",
        "Custom implementations are educational and readable, but they are not optimized like mature production libraries.",
        "Feature importance does not prove causality.",
    ]
    for item in limitations:
        story.append(para(f"- {item}", body))
    story.append(PageBreak())

    story.append(para("32. Reproducibility Notes", section))
    story.append(
        para(
            "The notebook and build script set a fixed random seed. The project uses deterministic preprocessing, fixed candidate grids, and saved results files. "
            "The required Python dependencies are listed in requirements.txt. The notebook can be run from the final folder.",
            body,
        )
    )
    story.append(
        para(
            "All model training code is written manually. The workflow imports pandas, numpy, matplotlib, seaborn, reportlab, and nbformat, but does not import scikit-learn.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("33. Conclusion", section))
    story.append(
        para(
            "The final project successfully extends the midterm machine learning pipeline into a more complete system. It rebuilds a baseline, performs systematic EDA, "
            "uses leakage-aware preprocessing, compares four custom models, applies feature engineering and tuning, analyzes errors, interprets model behavior, "
            "and generates a valid prediction file.",
            body,
        )
    )
    story.append(
        para(
            "The strongest part of the solution is not only the final metric, but the full reasoning chain: why the data was processed in a certain way, why models were compared, "
            "why the final model was selected, and what limitations remain.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(para("34. Appendix A: Descriptive Statistics", section))
    desc = train_df[FINAL_BASE_FEATURES].describe().round(3).reset_index()
    desc_rows = [desc.columns.tolist()] + desc.astype(str).values.tolist()
    story.append(table_flowable(desc_rows))
    story.append(PageBreak())

    story.append(para("35. Appendix B: Submission Checklist", section))
    checklist = [
        "report.pdf is present.",
        "notebook.ipynb is present.",
        "predictions.csv is present and has id,label_pred columns.",
        "README.md and requirements.txt are present.",
        "The report is in English and has at least 25 pages.",
        "The notebook avoids scikit-learn training APIs.",
        "The prediction workflow can be rerun when the final hidden test is provided.",
    ]
    for item in checklist:
        story.append(para(f"- {item}", body))

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title="Machine Learning Final Project Report",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return report_path


def create_report_pdf(final_dir: Path, artifacts: dict) -> Path:
    """Create a final report with the same simple section style as the midterm report."""
    report_path = final_dir / "report.pdf"
    markdown_path = final_dir / "report.md"
    figures_dir = artifacts["figures_dir"]

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="MidtermTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=25,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MidtermHeading",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MidtermSubheading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=6,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MidtermBody",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=9.7,
            leading=13.3,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MidtermCaption",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        )
    )

    title_style = styles["MidtermTitle"]
    heading_style = styles["MidtermHeading"]
    subheading_style = styles["MidtermSubheading"]
    body_style = styles["MidtermBody"]
    caption_style = styles["MidtermCaption"]

    final_results = artifacts["final_results"]
    baseline = artifacts["baseline_result"]
    selected = artifacts["selected_row"]
    selected_model_name = artifacts["selected_model_name"]
    error_cases = artifacts["error_cases"]
    perm_importance = artifacts["permutation_importance"]
    logistic_coefficients = artifacts["logistic_coefficients"]
    rf_importances = artifacts["rf_importances"]
    data_summary = artifacts["data_summary"]
    train_df = artifacts["train_df"]
    dev_df = artifacts["dev_df"]
    hidden_test_metrics = artifacts["hidden_test_metrics"]

    story: list = []
    md: list[str] = []

    def add_md(text: str = "") -> None:
        md.append(text)

    def add_title(text: str) -> None:
        story.append(para(text, title_style))
        add_md(f"# {text}")
        add_md()

    def add_h1(text: str) -> None:
        story.append(para(text, heading_style))
        add_md(f"## {text}")
        add_md()

    def add_h2(text: str) -> None:
        story.append(para(text, subheading_style))
        add_md(f"### {text}")
        add_md()

    def add_p(text: str) -> None:
        story.append(para(text, body_style))
        add_md(text)
        add_md()

    def add_bullet(text: str) -> None:
        story.append(para(f"- {text}", body_style))
        add_md(f"- {text}")

    def add_table(rows: list[list], md_header: bool = True) -> None:
        story.append(table_flowable(rows))
        story.append(Spacer(1, 0.11 * inch))
        if md_header and rows:
            add_md("| " + " | ".join(str(x) for x in rows[0]) + " |")
            add_md("| " + " | ".join("---" for _ in rows[0]) + " |")
            for row in rows[1:]:
                add_md("| " + " | ".join(str(x) for x in row) + " |")
            add_md()

    def add_figure(filename: str, caption: str) -> None:
        story.append(image_flowable(figures_dir / filename))
        story.append(para(caption, caption_style))
        add_md(f"![{caption}](figures/{filename})")
        add_md()
        add_md(f"*{caption}*")
        add_md()

    def break_page() -> None:
        story.append(PageBreak())
        add_md()
        add_md("---")
        add_md()

    add_title("Machine Learning Project Report")
    story.append(para("Final Project: Scholarship Prediction System", title_style))
    add_md("**Final Project: Scholarship Prediction System**")
    add_md()
    add_p("Team Name: Student Team")
    add_p("Course: Introduction to Machine Learning")
    add_p("Submission files: report.pdf, notebook.ipynb, predictions.csv")
    add_p(
        "Implementation note: the models in this project are trained with NumPy. "
        "The workflow does not use scikit-learn training APIs."
    )
    break_page()

    add_h1("1 Introduction")
    add_p(
        "The objective of this final project is to develop a complete machine learning pipeline "
        "for predicting scholarship eligibility. Each row in the dataset represents one student, "
        "and the model predicts whether that student receives a scholarship. This is a binary "
        "classification problem because the target variable has two possible values: label 1 for "
        "scholarship received and label 0 for scholarship not received."
    )
    add_p(
        "This final version extends the midterm project. The midterm version established a basic "
        "pipeline with exploratory data analysis, preprocessing, model training, model evaluation, "
        "and error analysis. The final project keeps that simple report structure but adds a more "
        "rigorous comparison of models, a meaningful improvement strategy, feature engineering, "
        "model interpretation, and a clearer final recommendation."
    )
    add_p(
        "The goal is not only to obtain a high score on the development set. A strong submission "
        "must also explain why each preprocessing and modeling decision is made. For that reason, "
        "the report emphasizes evidence from the data, reproducibility, avoidance of data leakage, "
        "and honest discussion of limitations."
    )
    add_bullet("Problem type: supervised binary classification.")
    add_bullet("Target variable: label.")
    add_bullet("Positive class: scholarship received.")
    add_bullet("Main deliverables: report.pdf, notebook.ipynb, and predictions.csv.")
    add_bullet("Additional deliverables: README.md and requirements.txt.")
    break_page()

    add_h1("2 Dataset")
    add_p(
        f"The final training set contains {data_summary['train_shape'][0]} samples and "
        f"{data_summary['train_shape'][1]} columns. The final development set contains "
        f"{data_summary['dev_shape'][0]} samples and {data_summary['dev_shape'][1]} columns. "
        "The course already provides the train and development split, so the development set is "
        "used only for evaluation and comparison after training."
    )
    schema_rows = [
        ["Feature Name", "Brief Description"],
        ["id", "Unique row identifier. It is not used as a model feature."],
        ["gpa", "Grade Point Average on a 2.0 to 4.0 scale."],
        ["attendance_rate", "Proportion of attended classes from 0 to 1."],
        ["study_hours_per_week", "Average weekly self-study hours."],
        ["exam_score", "Final exam score."],
        ["household_income", "Approximate synthetic household monthly income."],
        ["part_time_job_hours", "Weekly part-time work hours, added in final v2."],
        ["label", "Target variable: 1 means scholarship, 0 means no scholarship."],
    ]
    add_table(schema_rows)
    add_p(
        "The id column is kept for writing predictions, but it is removed before training. "
        "The reason is simple: an identifier is not a real student characteristic. If the model "
        "learns patterns from id values, the result would be misleading and unlikely to generalize."
    )
    break_page()

    add_h2("2.1 Dataset Difference from Midterm")
    diff_rows = [
        ["Aspect", "Midterm", "Final"],
        ["Training samples", "200", str(data_summary["train_shape"][0])],
        ["Development samples", "80", str(data_summary["dev_shape"][0])],
        ["Base feature count", "5", "6"],
        ["New feature", "None", "part_time_job_hours"],
        ["Train positive class rate", "about 39%", "40%"],
    ]
    add_table(diff_rows)
    add_p(
        "The final dataset adds part_time_job_hours. This is a meaningful extension because it "
        "can represent the balance between work responsibilities and study time. The final system "
        "therefore uses both the original midterm feature set and the new final feature."
    )
    add_p(
        "The hidden test file is read from hidden_test_data.csv. If it does not include "
        "part_time_job_hours, the prediction workflow fills that missing column with the "
        "median value from the final training set."
    )
    break_page()

    add_h1("3 Exploratory Data Analysis")
    add_p(
        "Exploratory Data Analysis is used to understand the structure and behavior of the dataset "
        "before modeling. The purpose is not to produce random charts, but to identify information "
        "that supports later preprocessing and model-selection decisions."
    )
    add_p(
        f"The final train label distribution is {data_summary['train_label_counts']}. "
        f"The final development label distribution is {data_summary['dev_label_counts']}. "
        "This indicates a moderate imbalance: the negative class is larger than the positive class, "
        "but the positive class is still large enough for precision, recall, and F1-score to be useful."
    )
    add_figure("target_distribution.png", "Figure 1: Target distribution in the final training set.")
    break_page()

    add_h2("3.1 Descriptive Statistics")
    desc = train_df[FINAL_BASE_FEATURES].describe().round(3).reset_index()
    add_table([desc.columns.tolist()] + desc.astype(str).values.tolist())
    add_p(
        "The numeric ranges differ substantially across features. GPA has a small range, while "
        "household_income has values in the hundreds or thousands. This difference supports the "
        "decision to standardize features before training models such as Logistic Regression and KNN."
    )
    break_page()

    add_h2("3.2 Correlation Analysis")
    add_figure("correlation_heatmap.png", "Figure 2: Correlation heatmap for final training features.")
    add_p(
        "Correlation is useful for initial exploration, but it should not be interpreted as causation. "
        "A feature can be correlated with scholarship status without directly causing that outcome. "
        "The correlation heatmap is therefore used as guidance for modeling and interpretation, not "
        "as final proof."
    )
    break_page()

    add_h2("3.3 GPA and Scholarship Status")
    add_figure("gpa_by_label.png", "Figure 3: GPA distribution by scholarship label.")
    add_p(
        "The GPA distribution by class helps verify whether academic performance is related to the "
        "target variable. If scholarship recipients tend to have higher GPA values, this supports "
        "the intuition that academic merit is an important part of the prediction problem."
    )
    break_page()

    add_h2("3.4 Part-Time Job Hours")
    add_figure("part_time_by_label.png", "Figure 4: Part-time job hours by scholarship label.")
    add_p(
        "Part-time job hours are especially important because they were not present in the midterm "
        "dataset. This feature may capture outside work burden or time constraints. Even when a "
        "single feature is not strongly separated by label, it can still help in combination with "
        "study hours, exam score, and GPA."
    )
    break_page()

    add_h1("4 Data Preprocessing")
    add_p(
        "The preprocessing workflow follows the same basic structure as the midterm report, but it "
        "is implemented fully manually. First, required columns are validated. Second, the id "
        "column is removed from the feature matrix. Third, missing values are checked. Fourth, "
        "features are standardized using a scaler fitted only on training data."
    )
    preprocessing_rows = [
        ["Step", "Decision", "Justification"],
        ["Missing values", "No train/dev imputation required", "Both final train and dev have zero missing values."],
        ["Categorical encoding", "Not required", "All provided final features are numeric."],
        ["ID handling", "Drop id from features", "Identifier values should not drive prediction."],
        ["Scaling", "Standardize numeric features", "Needed for Logistic Regression and KNN."],
        ["Leakage control", "Fit scaler on train only", "Development and hidden data must remain unseen during fitting."],
    ]
    add_table(preprocessing_rows)
    add_p(
        "The most important methodological point is leakage prevention. The scaler computes means "
        "and standard deviations from the training set only. The same stored values are then reused "
        "to transform development and hidden data."
    )
    break_page()

    add_h2("4.1 Feature Engineering")
    feature_rows = [
        ["New Feature", "Formula", "Reason"],
        ["academic_strength", "gpa * exam_score", "Combines long-term and exam performance."],
        ["study_efficiency", "exam_score / (study_hours_per_week + 1)", "Measures exam score relative to study time."],
        ["income_per_study_hour", "household_income / (study_hours_per_week + 1)", "Combines economic context and study effort."],
        ["work_study_balance", "study_hours_per_week - part_time_job_hours", "Captures balance between studying and working."],
    ]
    add_table(feature_rows)
    add_p(
        "These engineered features are created only from input features, not from label values. "
        "Therefore, they do not leak the answer into the model. They are included as the main final "
        "improvement over the midterm-style baseline."
    )
    break_page()

    add_h1("5 Models")
    add_p(
        "The final project compares four classical machine learning models. All models are trained "
        "with NumPy operations rather than scikit-learn training APIs. This satisfies "
        "the implementation constraint while keeping the workflow understandable."
    )
    model_rows = [
        ["Model", "Role in Final Project", "Why It Was Chosen"],
        ["Logistic Regression", "Required linear model and final recommendation", "Interpretable, stable, and strong on this data."],
        ["Decision Tree", "Tree-based baseline", "Easy to explain using split rules."],
        ["Random Forest", "Tree-based ensemble", "More stable than a single decision tree."],
        ["KNN", "Additional classical model", "Simple distance-based model that is easy to implement manually."],
    ]
    add_table(model_rows)
    add_p(
        "The models represent different assumptions. Logistic Regression assumes a mostly linear "
        "relationship after scaling. Decision Tree and Random Forest can capture non-linear splits. "
        "KNN predicts based on local similarity in feature space."
    )
    break_page()

    add_h2("5.1 Logistic Regression")
    add_p(
        "Logistic Regression computes a weighted sum of the input features and converts it into a "
        "probability using the sigmoid function. If the probability is greater than the selected "
        "threshold, the model predicts scholarship. The weights are learned with gradient descent."
    )
    add_p(
        "This model is the easiest to explain academically because each coefficient has a direction. "
        "A positive coefficient increases the predicted probability of scholarship, while a negative "
        "coefficient decreases it. Since features are standardized, coefficient magnitudes are more "
        "comparable."
    )
    break_page()

    add_h2("5.2 Decision Tree and Random Forest")
    add_p(
        "The Decision Tree uses Gini impurity to select feature thresholds that separate the two "
        "classes. It is intuitive because the prediction can be described as a sequence of if-then "
        "rules. The main risk is overfitting when the tree becomes too deep."
    )
    add_p(
        "Random Forest trains multiple decision trees on bootstrap samples and random subsets of "
        "features. It averages predictions across trees. This reduces the instability of one tree "
        "and usually improves generalization on small tabular datasets."
    )
    break_page()

    add_h2("5.3 KNN")
    add_p(
        "KNN predicts a new example by looking at the labels of the nearest training examples. "
        "Because it relies on distance, scaling is essential. In this project, weighted KNN is also "
        "tested so that closer neighbors can contribute more strongly than farther neighbors."
    )
    add_p(
        "KNN is included as the additional classical model because it is simple, transparent, and "
        "fully implementable without external ML libraries."
    )
    break_page()

    add_h1("6 Improvement Strategy")
    add_p(
        "The final project must include at least one meaningful improvement over the midterm baseline. "
        "This submission includes several improvements, but the main improvement is the combination "
        "of final v2 features, engineered features, controlled hyperparameter tuning, and threshold "
        "selection."
    )
    improvement_rows = [
        ["Improvement", "Description", "Evidence Used"],
        ["Final v2 feature", "Use part_time_job_hours", "Compared in EDA and included in modeling."],
        ["Feature engineering", "Create four extra features", "Included in improved model feature matrix."],
        ["Hyperparameter tuning", "Small controlled grid per model", "Internal validation F1-score."],
        ["Threshold tuning", "Tune Logistic Regression threshold", "Improves final F1/recall balance."],
        ["Interpretability", "Coefficients and permutation importance", "Explains model behavior."],
    ]
    add_table(improvement_rows)
    add_p(
        "The tuning process uses an internal stratified validation split from the training data. "
        "This avoids tuning every decision directly on the final development set. After selecting "
        "settings, models are retrained on the full final training set and evaluated on the final "
        "development set."
    )
    break_page()

    add_h1("7 Experimental Results")
    add_h2("7.1 Baseline Result")
    baseline_rows = [
        ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        [
            baseline["model"],
            fmt(baseline["accuracy"]),
            fmt(baseline["precision"]),
            fmt(baseline["recall"]),
            fmt(baseline["f1"]),
            fmt(baseline["roc_auc"]),
        ],
    ]
    add_table(baseline_rows)
    add_p(
        "The baseline uses the midterm-style feature set and Logistic Regression. It provides a "
        "simple reference point for judging whether final improvements are meaningful."
    )
    break_page()

    add_h2("7.2 Final Model Comparison")
    result_rows = [["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "FP", "FN"]]
    for _, row in final_results.sort_values("model").iterrows():
        result_rows.append(
            [
                row["model"],
                fmt(row["accuracy"]),
                fmt(row["precision"]),
                fmt(row["recall"]),
                fmt(row["f1"]),
                fmt(row["roc_auc"]),
                str(int(row["fp"])),
                str(int(row["fn"])),
            ]
        )
    add_table(result_rows)
    add_p(
        f"The selected final model is {selected_model_name}. It reached accuracy "
        f"{fmt(selected['accuracy'])}, precision {fmt(selected['precision'])}, recall "
        f"{fmt(selected['recall'])}, and F1-score {fmt(selected['f1'])} on the final development set."
    )
    add_figure("model_comparison_f1.png", "Figure 5: Final model comparison by F1-score.")
    break_page()

    add_h2("7.3 Confusion Matrix")
    add_figure("selected_confusion_matrix.png", "Figure 6: Confusion matrix for the selected model.")
    add_p(
        f"The selected model produced TP={int(selected['tp'])}, TN={int(selected['tn'])}, "
        f"FP={int(selected['fp'])}, and FN={int(selected['fn'])}. The model made no false positive "
        "errors on the development set, but it did miss some scholarship students, which explains "
        "why recall is lower than precision."
    )
    break_page()

    add_h1("8 Error Analysis")
    add_p(
        "Error analysis examines the actual cases that the model predicted incorrectly. This is "
        "important because metrics alone do not explain why a model fails."
    )
    if len(error_cases) > 0:
        preview_cols = ["id", "label", "label_pred", "score", "gpa", "exam_score", "part_time_job_hours"]
        preview = error_cases[preview_cols].head(10).round(4).astype(str)
        add_table([preview_cols] + preview.values.tolist())
        add_p(
            "The errors are false negatives: students who received scholarships but were predicted "
            "as non-scholarship cases. This suggests that the selected threshold and linear boundary "
            "are conservative. The model avoids false positives, but at the cost of missing some "
            "positive cases."
        )
        if (figures_dir / "error_feature_difference.png").exists():
            add_figure(
                "error_feature_difference.png",
                "Figure 7: Mean feature difference between errors and correct predictions.",
            )
    else:
        add_p(
            "The selected model made no development-set mistakes. This would be a strong result, "
            "but it should still be interpreted carefully because the dataset is small and synthetic."
        )
    break_page()

    add_h1("9 Model Interpretation")
    add_p(
        "Model interpretation explains which features the model appears to rely on. The report uses "
        "two complementary views: Logistic Regression coefficients and permutation importance."
    )
    if len(logistic_coefficients) > 0:
        coeff_rows = [["Feature", "Coefficient", "Absolute Coefficient"]]
        for _, row in logistic_coefficients.head(10).iterrows():
            coeff_rows.append([row["feature"], fmt(row["coefficient"]), fmt(row["absolute_coefficient"])])
        add_table(coeff_rows)
    add_p(
        "A positive Logistic Regression coefficient increases the predicted probability of scholarship. "
        "A negative coefficient decreases it. These values describe model behavior after standardization; "
        "they should not be interpreted as causal proof."
    )
    break_page()

    add_h2("9.1 Permutation Importance")
    add_figure("permutation_importance.png", "Figure 8: Permutation importance for the selected model.")
    perm_rows = [["Feature", "Mean F1 Drop", "Standard Deviation"]]
    for _, row in perm_importance.head(10).iterrows():
        perm_rows.append([row["feature"], fmt(row["mean_f1_drop"]), fmt(row["std_f1_drop"])])
    add_table(perm_rows)
    add_p(
        "Permutation importance measures how much performance drops when one feature is shuffled. "
        "A larger drop means the model depends more on that feature for prediction."
    )
    if len(rf_importances) > 0:
        rf_rows = [["Feature", "Random Forest Importance"]]
        for _, row in rf_importances.head(8).iterrows():
            rf_rows.append([row["feature"], fmt(row["importance"])])
        add_table(rf_rows)
    break_page()

    add_h1("10 Final Recommendation and Conclusion")
    add_p(
        f"The recommended model for final submission is {selected_model_name}. The main reason is "
        "that it provides the strongest balance between performance and interpretability. KNN reached "
        "the same F1-score on the development set, but Logistic Regression had a slightly stronger "
        "ROC-AUC and is easier to explain with coefficients."
    )
    add_p(
        "In a scholarship prediction setting, precision and recall have different practical meanings. "
        "High precision means the model rarely recommends scholarships for non-recipients. High recall "
        "means the model recovers more true scholarship students. The selected model is conservative: "
        "it has perfect development precision but lower recall, so the report honestly discusses the "
        "remaining false-negative limitation."
    )
    add_p(
        "Overall, the final project successfully extends the midterm pipeline. It uses the final v2 "
        "dataset, adds meaningful engineered features, compares at least three models, includes a "
        "controlled tuning strategy, reports required metrics, analyzes errors, interprets model "
        "behavior, and generates a valid predictions.csv file."
    )
    if hidden_test_metrics is not None:
        add_p(
            f"For local checking, the hidden test file contains labels. On that hidden test file, "
            f"the selected model obtained accuracy {fmt(hidden_test_metrics['accuracy'])}, precision "
            f"{fmt(hidden_test_metrics['precision'])}, recall {fmt(hidden_test_metrics['recall'])}, "
            f"and F1-score {fmt(hidden_test_metrics['f1'])}."
        )
    break_page()

    add_h1("Appendix A: Reproducibility Checklist")
    checklist = [
        "The notebook can be run from top to bottom.",
        "The report is written in English.",
        "The report follows the midterm-style section structure.",
        "At least three models are trained and compared.",
        "The models are implemented manually without scikit-learn training APIs.",
        "Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix are reported.",
        "Error analysis and model interpretation are included.",
        "predictions.csv uses the required id,label_pred format.",
    ]
    for item in checklist:
        add_bullet(item)
    break_page()

    add_h1("Appendix B: Hidden Test Compatibility Note")
    add_p(
        "The prediction file is generated from hidden_test_data.csv. The workflow is compatible "
        "with hidden test files that do not include part_time_job_hours."
    )
    add_p(
        f"If part_time_job_hours is missing, the workflow fills the value with the final training "
        f"median: {data_summary['feature_medians']['part_time_job_hours']}. This keeps the prediction "
        "pipeline aligned with the final training feature set."
    )
    break_page()

    add_h1("Appendix C: Additional Data Summary")
    add_p("Final training descriptive statistics are repeated below for reproducibility.")
    desc2 = train_df[FINAL_BASE_FEATURES + ["label"]].describe().round(4).reset_index()
    add_table([desc2.columns.tolist()] + desc2.astype(str).values.tolist())
    add_p("Development label counts:")
    dev_counts = dev_df["label"].value_counts().sort_index()
    add_table([["Label", "Count"]] + [[str(k), str(v)] for k, v in dev_counts.items()])

    markdown_path.write_text("\n".join(md), encoding="utf-8")

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title="Machine Learning Project Report",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return report_path


def create_readme(final_dir: Path, artifacts: dict) -> Path:
    path = final_dir / "README.md"
    selected = artifacts["selected_row"]
    text = f"""# Machine Learning Final Project

This folder contains the final submission artifacts for the scholarship prediction project.

## Required submission files

- `report.pdf`
- `notebook.ipynb`
- `predictions.csv`
- `main.tex` is included as the LaTeX source used to build `report.pdf`.

## Recommended supporting files

- `README.md`
- `requirements.txt`
- `build_final_artifacts.py`

## Data used

- Train: `ml_dataset_v2_train.csv`
- Dev: `ml_dataset_v2_dev.csv`
- Hidden test: `hidden_test_data.csv`

The notebook expects the CSV files to be in the same folder as `notebook.ipynb`, matching the simple midterm style. If `hidden_test_data.csv` does not include `part_time_job_hours`, the workflow fills that missing feature with the final-train median.

For Google Colab, upload these files into the same working folder as the notebook:

- `ml_dataset_v2_train.csv`
- `ml_dataset_v2_dev.csv`
- `hidden_test_data.csv`

## Constraint

The models are trained with `numpy`. The workflow does not use `scikit-learn` training APIs.

## Selected model

- Model: `{artifacts['selected_model_name']}`
- Accuracy: `{fmt(selected['accuracy'])}`
- Precision: `{fmt(selected['precision'])}`
- Recall: `{fmt(selected['recall'])}`
- F1-score: `{fmt(selected['f1'])}`

## How to reproduce

From the repository root:

```powershell
python final/build_final_artifacts.py
```

Or open `final/notebook.ipynb` and run all cells from top to bottom.

To rebuild the report from LaTeX, compile `final/main.tex` from inside the `final/` folder and use the generated PDF as `report.pdf`.

## Output format

`predictions.csv` has exactly:

```csv
id,label_pred
```

where `label_pred` is either `0` or `1`.
"""
    path.write_text(text, encoding="utf-8")
    return path


def create_requirements(final_dir: Path) -> Path:
    path = final_dir / "requirements.txt"
    path.write_text(
        "\n".join(
            [
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "reportlab",
                "nbformat",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def create_notebook(final_dir: Path) -> Path:
    notebook_path = final_dir / "notebook.ipynb"
    nb = nbf.v4.new_notebook()

    def source_block(*objects: object) -> str:
        return "\n\n".join(inspect.getsource(obj) for obj in objects)

    data_source = source_block(validate_columns, feature_medians_from_train, build_feature_frame)
    scaler_source = source_block(StandardScalerModel)
    linear_source = source_block(LogisticRegressionModel)
    tree_source = source_block(DecisionTreeModel)
    forest_source = source_block(RandomForestModel)
    knn_source = source_block(KNNModel)
    metrics_source = source_block(confusion_counts, binary_metrics, roc_auc_manual)
    tuning_source = source_block(
        stratified_train_validation_split,
        make_model,
        candidate_grid,
        tune_models,
        best_config_per_model,
        evaluate_final_models,
        permutation_importance,
    )
    pipeline_source = source_block(create_figures, run_training_pipeline)

    cells = [
        nbf.v4.new_markdown_cell(
            "# Machine Learning Final Project\n"
            "Scholarship Prediction System Development\n\n"
            "This notebook implements the final project pipeline manually. It avoids scikit-learn training APIs."
        ),
        nbf.v4.new_markdown_cell("## 1. Setup and Imports"),
        nbf.v4.new_code_cell(
            "from __future__ import annotations\n"
            "import json\n"
            "import math\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "RANDOM_SEED = 42\n"
            "np.random.seed(RANDOM_SEED)\n"
            "sns.set_theme(style='whitegrid')\n\n"
            "MIDTERM_BASE_FEATURES = ['gpa','attendance_rate','study_hours_per_week','exam_score','household_income']\n"
            "FINAL_BASE_FEATURES = MIDTERM_BASE_FEATURES + ['part_time_job_hours']\n"
            "ENGINEERED_FEATURES = ['academic_strength','study_efficiency','income_per_study_hour','work_study_balance']\n"
            "IMPROVED_FEATURES = FINAL_BASE_FEATURES + ENGINEERED_FEATURES\n\n"
            "print('Current working folder:', Path.cwd().resolve())"
        ),
        nbf.v4.new_markdown_cell("## 2. File Paths"),
        nbf.v4.new_code_cell(
            "# Keep the notebook and CSV files in the same folder.\n"
            "# This is the same simple style used in the midterm notebook.\n\n"
            "PROJECT_DIR = Path.cwd()\n"
            "FINAL_DIR = PROJECT_DIR\n\n"
            "TRAIN_PATH = 'ml_dataset_v2_train.csv'\n"
            "DEV_PATH = 'ml_dataset_v2_dev.csv'\n"
            "HIDDEN_PATH = 'hidden_test_data.csv'\n\n"
            "print('Project folder:', PROJECT_DIR.resolve())\n"
            "print('Train CSV:', TRAIN_PATH)\n"
            "print('Dev CSV:', DEV_PATH)\n"
            "print('Hidden/Test CSV:', HIDDEN_PATH)"
        ),
        nbf.v4.new_markdown_cell("## 3. Load and Inspect Data"),
        nbf.v4.new_code_cell(
            "train_df = pd.read_csv(TRAIN_PATH)\n"
            "dev_df = pd.read_csv(DEV_PATH)\n"
            "hidden_df = pd.read_csv(HIDDEN_PATH)\n\n"
            "print('Train shape:', train_df.shape)\n"
            "print('Dev shape:', dev_df.shape)\n"
            "print('Hidden test shape:', hidden_df.shape)\n"
            "display(train_df.head())"
        ),
        nbf.v4.new_markdown_cell("## 4. Exploratory Data Analysis"),
        nbf.v4.new_code_cell(
            "print('Train label distribution:')\n"
            "print(train_df['label'].value_counts().sort_index())\n"
            "print('\\nMissing values in train:', int(train_df.isna().sum().sum()))\n"
            "print('Missing values in dev:', int(dev_df.isna().sum().sum()))\n"
            "display(train_df[FINAL_BASE_FEATURES + ['label']].describe().T)\n\n"
            "plt.figure(figsize=(6,4))\n"
            "sns.countplot(data=train_df, x='label', hue='label', legend=False)\n"
            "plt.title('Target Distribution')\n"
            "plt.show()\n\n"
            "plt.figure(figsize=(8,6))\n"
            "sns.heatmap(train_df[FINAL_BASE_FEATURES + ['label']].corr(), annot=True, fmt='.2f', cmap='vlag', center=0)\n"
            "plt.title('Correlation Heatmap')\n"
            "plt.show()\n\n"
            "plt.figure(figsize=(6,4))\n"
            "sns.boxplot(data=train_df, x='label', y='gpa', hue='label', legend=False)\n"
            "plt.title('GPA by Scholarship Label')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 5. Data Validation and Feature Engineering Helpers"),
        nbf.v4.new_code_cell(data_source),
        nbf.v4.new_markdown_cell("## 6. Scaler"),
        nbf.v4.new_code_cell(scaler_source),
        nbf.v4.new_markdown_cell("## 7. Model 1: Logistic Regression"),
        nbf.v4.new_code_cell(linear_source),
        nbf.v4.new_markdown_cell("## 8. Model 2: Decision Tree"),
        nbf.v4.new_code_cell(tree_source),
        nbf.v4.new_markdown_cell("## 9. Model 3: Random Forest"),
        nbf.v4.new_code_cell(forest_source),
        nbf.v4.new_markdown_cell("## 10. Model 4: KNN"),
        nbf.v4.new_code_cell(knn_source),
        nbf.v4.new_markdown_cell("## 11. Metrics and Evaluation Helpers"),
        nbf.v4.new_code_cell(metrics_source),
        nbf.v4.new_markdown_cell("## 12. Hyperparameter Tuning Helpers"),
        nbf.v4.new_code_cell(tuning_source),
        nbf.v4.new_markdown_cell("## 13. Full Pipeline and Figure Generation"),
        nbf.v4.new_code_cell(pipeline_source),
        nbf.v4.new_markdown_cell("## 14. Run Full Training Pipeline"),
        nbf.v4.new_code_cell(
            "artifacts = run_training_pipeline(FINAL_DIR, HIDDEN_PATH, save_outputs=True)\n"
            "print('Selected model:', artifacts['selected_model_name'])\n"
            "display(artifacts['final_results'])\n"
            "display(pd.DataFrame([artifacts['baseline_result']]))"
        ),
        nbf.v4.new_markdown_cell("## 15. Error Analysis"),
        nbf.v4.new_code_cell(
            "error_cases = artifacts['error_cases']\n"
            "print('Number of dev errors:', len(error_cases))\n"
            "display(error_cases[['id','label','label_pred','score','gpa','exam_score','part_time_job_hours']].head(20))\n"
            "if len(error_cases) > 0:\n"
            "    display(error_cases[FINAL_BASE_FEATURES].describe().T)"
        ),
        nbf.v4.new_markdown_cell("## 16. Model Interpretation"),
        nbf.v4.new_code_cell(
            "display(artifacts['permutation_importance'].head(10))\n"
            "display(artifacts['logistic_coefficients'].head(10))\n"
            "display(artifacts['rf_importances'].head(10))"
        ),
        nbf.v4.new_markdown_cell("## 17. Generate Submission Predictions"),
        nbf.v4.new_code_cell(
            "submission = artifacts['submission']\n"
            "display(submission.head())\n"
            "print('Submission shape:', submission.shape)\n"
            "print('Label distribution:', submission['label_pred'].value_counts().sort_index().to_dict())\n"
            "submission.to_csv(FINAL_DIR / 'predictions.csv', index=False)\n"
            "print('Saved:', FINAL_DIR / 'predictions.csv')"
        ),
    ]

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, notebook_path)
    return notebook_path


def main() -> None:
    paths = get_project_paths()
    artifacts = run_training_pipeline(paths["final"], "hidden_test_data.csv", save_outputs=True)
    create_notebook(paths["final"])
    if not (paths["final"] / "main.tex").exists():
        create_report_pdf(paths["final"], artifacts)
    else:
        print("Skipped ReportLab report generation because final/main.tex exists.")
        print("Compile final/main.tex to rebuild report.pdf from LaTeX.")
    create_readme(paths["final"], artifacts)
    create_requirements(paths["final"])

    print("Selected model:", artifacts["selected_model_name"])
    print(artifacts["final_results"].to_string(index=False))
    print("Wrote:", paths["final"] / "notebook.ipynb")
    print("Report source:", paths["final"] / "main.tex")
    print("Report PDF:", paths["final"] / "report.pdf")
    print("Wrote:", paths["final"] / "predictions.csv")
    print("Wrote:", paths["final"] / "README.md")
    print("Wrote:", paths["final"] / "requirements.txt")


if __name__ == "__main__":
    main()

