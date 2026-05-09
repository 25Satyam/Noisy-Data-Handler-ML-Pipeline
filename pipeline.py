"""
pipeline.py
-----------
NoisyDataPipeline — orchestrates all preprocessing stages end-to-end
and exposes a clean fit / transform / fit_transform interface.

Stage order:
  1. BasicPreprocessor   — dedup, name standardise, drop bad columns
  2. AdvancedImputer     — fill missing values
  3. OutlierDetector     — detect and treat outliers
  4. FeatureEngineer     — encode + scale features
"""

import logging
import os
import time
import numpy as np
import pandas as pd
import yaml
from typing import Optional, Tuple

from src.preprocessor import BasicPreprocessor
from src.imputer import AdvancedImputer
from src.outlier_detector import OutlierDetector
from src.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at '{config_path}', using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


class NoisyDataPipeline:
    """
    Full noisy-data preprocessing pipeline.

    Parameters
    ----------
    target_col : str
        Name of the target column (excluded from all transformations).
    config_path : str
        Path to YAML config file.
    **kwargs
        Override any config key directly.
    """

    def __init__(
        self,
        target_col: str = "target",
        config_path: str = "config/config.yaml",
        **kwargs,
    ):
        self.target_col = target_col
        self.config = _load_config(config_path)
        self.config.update(kwargs)   # keyword overrides take precedence

        # Build pipeline stages from config
        self._build_stages()

        # Tracking
        self._is_fitted: bool = False
        self._pipeline_log: list = []

    # ------------------------------------------------------------------ #
    #  Stage construction
    # ------------------------------------------------------------------ #

    def _build_stages(self) -> None:
        data_cfg = self.config.get("data", {})
        imp_cfg = self.config.get("imputation", {})
        out_cfg = self.config.get("outlier_detection", {})
        fe_cfg = self.config.get("feature_engineering", {})

        self.preprocessor = BasicPreprocessor(
            missing_threshold=data_cfg.get("missing_threshold", 0.5),
            low_variance_threshold=data_cfg.get("low_variance_threshold", 0.01),
            duplicate_action=data_cfg.get("duplicate_action", "drop"),
            target_col=self.target_col,
        )

        self.imputer = AdvancedImputer(
            numeric_strategy=imp_cfg.get("numeric_strategy", "knn"),
            categorical_strategy=imp_cfg.get("categorical_strategy", "most_frequent"),
            knn_neighbors=imp_cfg.get("knn_neighbors", 5),
            iterative_max_iter=imp_cfg.get("iterative_max_iter", 10),
            fill_value=imp_cfg.get("fill_value", "MISSING"),
        )

        self.outlier_detector = OutlierDetector(
            method=out_cfg.get("method", "isolation_forest"),
            treatment=out_cfg.get("treatment", "clip"),
            iqr_multiplier=out_cfg.get("iqr_multiplier", 1.5),
            zscore_threshold=out_cfg.get("zscore_threshold", 3.0),
            contamination=out_cfg.get("contamination", 0.05),
            lof_n_neighbors=out_cfg.get("lof_n_neighbors", 20),
            target_col=self.target_col,
        )

        self.feature_engineer = FeatureEngineer(
            scaling=fe_cfg.get("scaling", "robust"),
            encode_categoricals=fe_cfg.get("encode_categoricals", True),
            encoding_strategy=fe_cfg.get("encoding_strategy", "onehot"),
            onehot_max_categories=fe_cfg.get("onehot_max_categories", 15),
            apply_log_transform=fe_cfg.get("apply_log_transform", True),
            skewness_threshold=fe_cfg.get("skewness_threshold", 1.0),
            polynomial_features=fe_cfg.get("polynomial_features", False),
            polynomial_degree=fe_cfg.get("polynomial_degree", 2),
            target_col=self.target_col,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "NoisyDataPipeline":
        """Fit all stages on df (including target column for reference)."""
        logger.info("=" * 55)
        logger.info("  NoisyDataPipeline — FIT")
        logger.info("=" * 55)

        self._log_stage("fit", "start", df)

        # Stage 1: Basic preprocessing
        t0 = time.time()
        df_clean = self.preprocessor.fit_transform(df)
        self._log_stage("BasicPreprocessor", "fit+transform", df_clean, t0)

        # Extract target before feature engineering
        y = df_clean[self.target_col] if self.target_col in df_clean.columns else None
        X = df_clean.drop(columns=[self.target_col], errors="ignore")

        # Update column lists in downstream transformers
        num_cols, cat_cols = self.preprocessor.get_feature_types()
        self.imputer.numeric_cols = num_cols
        self.imputer.categorical_cols = cat_cols

        # Stage 2: Imputation
        t0 = time.time()
        X_imp = self.imputer.fit_transform(X)
        self._log_stage("AdvancedImputer", "fit+transform", X_imp, t0)

        # Stage 3: Outlier detection
        t0 = time.time()
        self.outlier_detector.numeric_cols = X_imp.select_dtypes(include=[np.number]).columns.tolist()
        X_out = self.outlier_detector.fit_transform(X_imp)
        self._log_stage("OutlierDetector", "fit+transform", X_out, t0)

        # Re-align y after row removals
        if y is not None and len(X_out) != len(y):
            logger.warning("Row count changed during outlier removal; re-aligning target.")
            y = y.iloc[:len(X_out)].reset_index(drop=True)

        # Stage 4: Feature engineering
        t0 = time.time()
        self.feature_engineer.numeric_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_engineer.categorical_cols = X_out.select_dtypes(exclude=[np.number]).columns.tolist()
        self.feature_engineer.fit(X_out)
        self._log_stage("FeatureEngineer", "fit", X_out, t0)

        self._is_fitted = True
        logger.info("Pipeline fitting complete.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform df through all fitted stages.

        Returns
        -------
        X : pd.DataFrame   — transformed feature matrix
        y : pd.Series|None — target column (None if absent)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df_clean = self.preprocessor.transform(df)

        y = df_clean[self.target_col].reset_index(drop=True) if self.target_col in df_clean.columns else None
        X = df_clean.drop(columns=[self.target_col], errors="ignore")

        X_imp = self.imputer.transform(X)
        X_out = self.outlier_detector.transform(X_imp)

        # Re-align y
        if y is not None and len(X_out) != len(y):
            y = y.iloc[:len(X_out)].reset_index(drop=True)

        X_fe = self.feature_engineer.transform(X_out)

        return X_fe, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _log_stage(self, stage: str, action: str, df: pd.DataFrame, t0: float = None) -> None:
        entry = {
            "stage": stage,
            "action": action,
            "rows": len(df),
            "cols": df.shape[1] if hasattr(df, "shape") else "N/A",
        }
        if t0 is not None:
            entry["elapsed_s"] = round(time.time() - t0, 3)
        self._pipeline_log.append(entry)
        logger.info(
            f"[{stage}] {action} — {entry['rows']} rows × {entry['cols']} cols"
            + (f" ({entry['elapsed_s']}s)" if t0 else "")
        )

    def pipeline_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising each pipeline stage."""
        return pd.DataFrame(self._pipeline_log)

    def save_processed(self, df: pd.DataFrame, path: str = "data/processed/cleaned.csv") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Processed data saved to '{path}'")
