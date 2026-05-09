"""
outlier_detector.py
-------------------
OutlierDetector — detects and treats outliers in numeric columns.

Detection methods:
  iqr              — Interquartile Range (Tukey fences)
  zscore           — Standard score threshold
  isolation_forest — Tree-based anomaly scoring
  lof              — Local Outlier Factor (density-based)

Treatment strategies:
  clip   — clip values to the fence boundaries
  remove — drop outlier rows
  flag   — add boolean column, keep rows
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detect and treat outliers in numeric features.

    Parameters
    ----------
    method : str
        Detection algorithm: 'iqr' | 'zscore' | 'isolation_forest' | 'lof' | 'none'
    treatment : str
        What to do with detected outliers: 'clip' | 'remove' | 'flag'
    iqr_multiplier : float
        IQR fence multiplier (default 1.5 = Tukey's standard).
    zscore_threshold : float
        Absolute z-score above which a point is an outlier (default 3.0).
    contamination : float
        Expected fraction of outliers, used by IsolationForest and LOF.
    lof_n_neighbors : int
        Number of neighbours for LOF.
    numeric_cols : list or None
        Columns to check. Auto-detects numeric columns if None.
    target_col : str or None
        Target column to skip entirely.
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        treatment: str = "clip",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: float = 0.05,
        lof_n_neighbors: int = 20,
        numeric_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ):
        self.method = method
        self.treatment = treatment
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.lof_n_neighbors = lof_n_neighbors
        self.numeric_cols = numeric_cols
        self.target_col = target_col

        # Learned boundaries (for iqr / zscore)
        self._bounds: Dict[str, Dict] = {}
        # Fitted model (for isolation_forest / lof)
        self._model = None
        self._num_cols_fit: List[str] = []

    # ------------------------------------------------------------------ #
    #  Fit
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, y=None) -> "OutlierDetector":
        if self.method == "none":
            return self

        # Select columns
        if self.numeric_cols is not None:
            num_cols = [c for c in self.numeric_cols if c in df.columns]
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self.target_col and self.target_col in num_cols:
            num_cols.remove(self.target_col)

        self._num_cols_fit = num_cols

        if self.method == "iqr":
            self._fit_iqr(df)
        elif self.method == "zscore":
            self._fit_zscore(df)
        elif self.method == "isolation_forest":
            self._fit_isolation_forest(df)
        elif self.method == "lof":
            self._fit_lof(df)
        else:
            raise ValueError(f"Unknown method: '{self.method}'")

        return self

    def _fit_iqr(self, df: pd.DataFrame) -> None:
        for col in self._num_cols_fit:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self._bounds[col] = {
                "lower": q1 - self.iqr_multiplier * iqr,
                "upper": q3 + self.iqr_multiplier * iqr,
            }
        logger.info(f"IQR bounds computed for {len(self._num_cols_fit)} columns.")

    def _fit_zscore(self, df: pd.DataFrame) -> None:
        for col in self._num_cols_fit:
            self._bounds[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
            }
        logger.info(f"Z-score stats computed for {len(self._num_cols_fit)} columns.")

    def _fit_isolation_forest(self, df: pd.DataFrame) -> None:
        X = df[self._num_cols_fit].copy()
        # Fill remaining NaN with median for fitting
        X = X.fillna(X.median())
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X)
        logger.info("IsolationForest fitted.")

    def _fit_lof(self, df: pd.DataFrame) -> None:
        X = df[self._num_cols_fit].copy()
        X = X.fillna(X.median())
        # LOF with novelty=False for fit+predict in one go
        self._model = LocalOutlierFactor(
            n_neighbors=self.lof_n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        # LOF must be fit+predict at once in non-novelty mode;
        # store predictions at fit time.
        self._lof_labels = self._model.fit_predict(X)
        logger.info("LocalOutlierFactor fitted.")

    # ------------------------------------------------------------------ #
    #  Transform
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.method == "none":
            return df

        df = df.copy()
        outlier_mask = self._get_outlier_mask(df)

        n_outliers = outlier_mask.sum()
        logger.info(
            f"[{self.method}] {n_outliers} outlier rows detected "
            f"({n_outliers / len(df) * 100:.1f}% of data)."
        )

        if self.treatment == "clip":
            df = self._clip_outliers(df, outlier_mask)
        elif self.treatment == "remove":
            df = df[~outlier_mask].reset_index(drop=True)
            logger.info(f"Removed {n_outliers} outlier rows.")
        elif self.treatment == "flag":
            df["is_outlier"] = outlier_mask
        else:
            raise ValueError(f"Unknown treatment: '{self.treatment}'")

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    #  Outlier mask helpers
    # ------------------------------------------------------------------ #

    def _get_outlier_mask(self, df: pd.DataFrame) -> pd.Series:
        if self.method == "iqr":
            return self._mask_iqr(df)
        elif self.method == "zscore":
            return self._mask_zscore(df)
        elif self.method == "isolation_forest":
            return self._mask_model(df)
        elif self.method == "lof":
            return self._mask_lof(df)
        return pd.Series(False, index=df.index)

    def _mask_iqr(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for col in self._num_cols_fit:
            if col not in df.columns or col not in self._bounds:
                continue
            lo, hi = self._bounds[col]["lower"], self._bounds[col]["upper"]
            mask |= (df[col] < lo) | (df[col] > hi)
        return mask

    def _mask_zscore(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for col in self._num_cols_fit:
            if col not in df.columns or col not in self._bounds:
                continue
            mu = self._bounds[col]["mean"]
            sigma = self._bounds[col]["std"]
            if sigma == 0:
                continue
            z = (df[col] - mu).abs() / sigma
            mask |= z > self.zscore_threshold
        return mask

    def _mask_model(self, df: pd.DataFrame) -> pd.Series:
        X = df[[c for c in self._num_cols_fit if c in df.columns]].copy()
        X = X.fillna(X.median())
        preds = self._model.predict(X)          # 1 = inlier, -1 = outlier
        return pd.Series(preds == -1, index=df.index)

    def _mask_lof(self, df: pd.DataFrame) -> pd.Series:
        # In non-novelty LOF mode, reuse stored labels from fit
        labels = pd.Series(self._lof_labels, dtype=int)
        # Align to current df length (may differ if df was subset)
        labels = labels.reindex(df.index, fill_value=1)
        return labels == -1

    # ------------------------------------------------------------------ #
    #  Clipping
    # ------------------------------------------------------------------ #

    def _clip_outliers(self, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Clip numeric columns to IQR/zscore boundaries where available."""
        for col in self._num_cols_fit:
            if col not in df.columns:
                continue
            if self.method == "iqr" and col in self._bounds:
                lo, hi = self._bounds[col]["lower"], self._bounds[col]["upper"]
                df[col] = df[col].clip(lower=lo, upper=hi)
            elif self.method == "zscore" and col in self._bounds:
                mu = self._bounds[col]["mean"]
                sigma = self._bounds[col]["std"]
                lo = mu - self.zscore_threshold * sigma
                hi = mu + self.zscore_threshold * sigma
                df[col] = df[col].clip(lower=lo, upper=hi)
            else:
                # For model-based, clip to column percentiles
                lo = df[col].quantile(0.01)
                hi = df[col].quantile(0.99)
                df.loc[mask, col] = df.loc[mask, col].clip(lower=lo, upper=hi)
        return df

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-column outlier count using IQR (independent of chosen method)."""
        rows = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            rows.append({
                "column": col,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "lower_fence": round(lo, 4),
                "upper_fence": round(hi, 4),
                "outlier_count": n_out,
                "outlier_pct": round(n_out / len(df) * 100, 2),
            })
        return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)
