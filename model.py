"""
model.py
--------
ModelTrainer — wraps model selection, training, hyperparameter tuning,
cross-validation, and evaluation reporting.

Supports:
  Classification : RandomForestClassifier, GradientBoostingClassifier,
                   LogisticRegression
  Regression     : RandomForestRegressor, GradientBoostingRegressor,
                   LinearRegression
"""

import logging
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any, Dict, Optional, Tuple

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Model catalogue
# ------------------------------------------------------------------ #

CLASSIFIERS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

REGRESSORS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "linear_regression": LinearRegression,
}

DEFAULT_PARAMS: Dict[str, Dict] = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    },
    "linear_regression": {},
}


class ModelTrainer:
    """
    Train, tune, and evaluate a sklearn model.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    algorithm : str
        Model key — see CLASSIFIERS / REGRESSORS above.
    test_size : float
        Fraction of data held out for evaluation.
    cv_folds : int
        Number of cross-validation folds.
    tune_hyperparameters : bool
        Whether to run GridSearchCV.
    random_state : int
        Seed for reproducibility.
    report_dir : str
        Directory to save plots and reports.
    """

    def __init__(
        self,
        task: str = "classification",
        algorithm: str = "random_forest",
        test_size: float = 0.2,
        cv_folds: int = 5,
        tune_hyperparameters: bool = True,
        random_state: int = 42,
        report_dir: str = "reports",
    ):
        self.task = task
        self.algorithm = algorithm
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.tune_hyperparameters = tune_hyperparameters
        self.random_state = random_state
        self.report_dir = report_dir

        self.model = None
        self.best_params_: Dict = {}
        self.metrics_: Dict = {}
        self._X_test = None
        self._y_test = None
        self._y_pred = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def train(self, X: pd.DataFrame, y: pd.Series) -> "ModelTrainer":
        """Split, optionally tune, train, and evaluate the model."""
        logger.info(f"Training [{self.algorithm}] for task={self.task} ...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task == "classification" else None,
        )

        self._X_test, self._y_test = X_test, y_test

        base_model = self._get_base_model()

        if self.tune_hyperparameters:
            logger.info("Running GridSearchCV ...")
            param_grid = DEFAULT_PARAMS.get(self.algorithm, {})
            if param_grid:
                t0 = time.time()
                gs = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    n_jobs=-1,
                    scoring=self._scoring_metric(),
                    verbose=0,
                )
                gs.fit(X_train, y_train)
                self.model = gs.best_estimator_
                self.best_params_ = gs.best_params_
                logger.info(
                    f"Best params: {self.best_params_} "
                    f"(CV score: {gs.best_score_:.4f}, {round(time.time()-t0, 1)}s)"
                )
            else:
                # Algorithm has no tunable params (e.g. linear regression)
                self.model = base_model
                self.model.fit(X_train, y_train)
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)

        self._y_pred = self.model.predict(X_test)
        self.metrics_ = self._compute_metrics(y_test, self._y_pred)
        self._log_metrics()

        return self

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Run k-fold CV on the full dataset and return mean ± std scores."""
        if self.model is None:
            raise RuntimeError("Call train() before cross_validate().")
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=self.cv_folds,
            scoring=self._scoring_metric(),
            n_jobs=-1,
        )
        result = {
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "cv_scores": cv_scores.tolist(),
        }
        logger.info(f"CV ({self.cv_folds}-fold): {result['cv_mean']} ± {result['cv_std']}")
        return result

    def feature_importance(self, feature_names: list) -> Optional[pd.DataFrame]:
        """Return feature importances if available."""
        if not hasattr(self.model, "feature_importances_"):
            return None
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    #  Report generation
    # ------------------------------------------------------------------ #

    def generate_report(self) -> str:
        """Generate an HTML report with metrics, confusion matrix, and feature importances."""
        os.makedirs(self.report_dir, exist_ok=True)
        plots = []

        if self.task == "classification":
            plots.append(self._plot_confusion_matrix())

        plots.append(self._plot_feature_importances())
        plots = [p for p in plots if p]  # remove None

        html = self._build_html_report(plots)
        report_path = os.path.join(self.report_dir, "report.html")
        with open(report_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved to '{report_path}'")
        return report_path

    def _plot_confusion_matrix(self) -> Optional[str]:
        if self._y_test is None or self._y_pred is None:
            return None
        cm = confusion_matrix(self._y_test, self._y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        path = os.path.join(self.report_dir, "confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return path

    def _plot_feature_importances(self) -> Optional[str]:
        if not hasattr(self.model, "feature_importances_"):
            return None
        imp = self.model.feature_importances_
        names = [f"f{i}" for i in range(len(imp))]
        top_n = 20
        sorted_idx = np.argsort(imp)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(top_n), imp[sorted_idx], color="steelblue")
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([names[i] for i in sorted_idx], rotation=45, ha="right")
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.set_ylabel("Importance")
        path = os.path.join(self.report_dir, "feature_importances.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return path

    def _build_html_report(self, plot_paths: list) -> str:
        metrics_rows = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in self.metrics_.items()
        )
        params_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in self.best_params_.items()
        )
        img_tags = "".join(
            f'<img src="{os.path.basename(p)}" style="max-width:600px;margin:10px">'
            for p in plot_paths
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Noisy Data Handler — Model Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 500px; margin-bottom: 30px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
    th {{ background-color: #2c3e50; color: white; }}
    tr:nth-child(even) {{ background-color: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>🧹 Noisy Data Handler — Model Evaluation Report</h1>
  <p><em>Task: {self.task} | Algorithm: {self.algorithm}</em></p>

  <h2>📊 Evaluation Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {metrics_rows}
  </table>

  <h2>⚙️ Best Hyperparameters</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    {params_rows if params_rows else "<tr><td colspan='2'>No tuning performed</td></tr>"}
  </table>

  <h2>📈 Plots</h2>
  {img_tags if img_tags else "<p>No plots generated.</p>"}
</body>
</html>"""

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _get_base_model(self):
        catalogue = CLASSIFIERS if self.task == "classification" else REGRESSORS
        if self.algorithm not in catalogue:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}' for task '{self.task}'. "
                f"Choose from: {list(catalogue.keys())}"
            )
        klass = catalogue[self.algorithm]
        if self.algorithm in ("random_forest", "gradient_boosting"):
            return klass(random_state=self.random_state)
        elif self.algorithm == "logistic_regression":
            return klass(random_state=self.random_state, max_iter=1000)
        return klass()

    def _scoring_metric(self) -> str:
        return "f1_weighted" if self.task == "classification" else "r2"

    def _compute_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        if self.task == "classification":
            metrics = {
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            }
            try:
                if len(np.unique(y_true)) == 2:
                    proba = self.model.predict_proba(self._X_test)[:, 1]
                    metrics["roc_auc"] = round(roc_auc_score(y_true, proba), 4)
            except Exception:
                pass
        else:
            mse = mean_squared_error(y_true, y_pred)
            metrics = {
                "mae": round(mean_absolute_error(y_true, y_pred), 4),
                "mse": round(mse, 4),
                "rmse": round(np.sqrt(mse), 4),
                "r2": round(r2_score(y_true, y_pred), 4),
            }
        return metrics

    def _log_metrics(self) -> None:
        logger.info("--- Evaluation Results ---")
        for k, v in self.metrics_.items():
            logger.info(f"  {k:<20}: {v}")
        logger.info("--------------------------")

    def save_metrics(self, path: str = "reports/metrics.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics_, f, indent=2)
        logger.info(f"Metrics saved to '{path}'")
