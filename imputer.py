"""
imputer.py
----------
AdvancedImputer — handles missing value imputation for both numeric
and categorical columns using multiple strategies.

Strategies:
  Numeric   : mean | median | knn | iterative (MICE)
  Categorical: most_frequent | constant
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class AdvancedImputer(BaseEstimator, TransformerMixin):
    """
    Two-stage imputer: numeric columns first, then categorical.

    Parameters
    ----------
    numeric_strategy : str
        One of 'mean', 'median', 'knn', 'iterative'.
    categorical_strategy : str
        One of 'most_frequent', 'constant'.
    knn_neighbors : int
        Number of neighbours for KNN imputation. Default 5.
    iterative_max_iter : int
        Max iterations for IterativeImputer. Default 10.
    fill_value : str
        Constant fill for categorical when strategy='constant'. Default 'MISSING'.
    numeric_cols : list or None
        Explicit list of numeric columns. Auto-detected if None.
    categorical_cols : list or None
        Explicit list of categorical columns. Auto-detected if None.
    """

    def __init__(
        self,
        numeric_strategy: str = "knn",
        categorical_strategy: str = "most_frequent",
        knn_neighbors: int = 5,
        iterative_max_iter: int = 10,
        fill_value: str = "MISSING",
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.knn_neighbors = knn_neighbors
        self.iterative_max_iter = iterative_max_iter
        self.fill_value = fill_value
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        self._num_imputer = None
        self._cat_imputer = None
        self._num_cols_fit: List[str] = []
        self._cat_cols_fit: List[str] = []

    # ------------------------------------------------------------------ #
    #  Build imputers
    # ------------------------------------------------------------------ #

    def _build_numeric_imputer(self):
        s = self.numeric_strategy
        if s == "mean":
            return SimpleImputer(strategy="mean")
        elif s == "median":
            return SimpleImputer(strategy="median")
        elif s == "knn":
            return KNNImputer(n_neighbors=self.knn_neighbors)
        elif s == "iterative":
            return IterativeImputer(max_iter=self.iterative_max_iter, random_state=42)
        else:
            raise ValueError(f"Unknown numeric_strategy: '{s}'")

    def _build_categorical_imputer(self):
        s = self.categorical_strategy
        if s == "most_frequent":
            return SimpleImputer(strategy="most_frequent")
        elif s == "constant":
            return SimpleImputer(strategy="constant", fill_value=self.fill_value)
        else:
            raise ValueError(f"Unknown categorical_strategy: '{s}'")

    # ------------------------------------------------------------------ #
    #  Fit / Transform
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, y=None) -> "AdvancedImputer":
        # Auto-detect columns
        self._num_cols_fit = (
            self.numeric_cols
            if self.numeric_cols is not None
            else df.select_dtypes(include=[np.number]).columns.tolist()
        )
        self._cat_cols_fit = (
            self.categorical_cols
            if self.categorical_cols is not None
            else df.select_dtypes(exclude=[np.number]).columns.tolist()
        )

        # Filter to actually-present columns
        self._num_cols_fit = [c for c in self._num_cols_fit if c in df.columns]
        self._cat_cols_fit = [c for c in self._cat_cols_fit if c in df.columns]

        if self._num_cols_fit:
            self._num_imputer = self._build_numeric_imputer()
            self._num_imputer.fit(df[self._num_cols_fit])
            missing = df[self._num_cols_fit].isna().sum().sum()
            logger.info(
                f"Numeric imputer ({self.numeric_strategy}) fitted on "
                f"{len(self._num_cols_fit)} cols — {missing} missing values."
            )

        if self._cat_cols_fit:
            self._cat_imputer = self._build_categorical_imputer()
            self._cat_imputer.fit(df[self._cat_cols_fit])
            missing = df[self._cat_cols_fit].isna().sum().sum()
            logger.info(
                f"Categorical imputer ({self.categorical_strategy}) fitted on "
                f"{len(self._cat_cols_fit)} cols — {missing} missing values."
            )

        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df = df.copy()

        if self._num_imputer and self._num_cols_fit:
            present = [c for c in self._num_cols_fit if c in df.columns]
            imputed = self._num_imputer.transform(df[present])
            df[present] = imputed

        if self._cat_imputer and self._cat_cols_fit:
            present = [c for c in self._cat_cols_fit if c in df.columns]
            imputed = self._cat_imputer.transform(df[present])
            df[present] = imputed

        remaining_missing = df.isna().sum().sum()
        if remaining_missing:
            logger.warning(f"{remaining_missing} missing values remain after imputation.")
        else:
            logger.info("All missing values successfully imputed.")

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-column missing value stats."""
        report = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "dtype": df.dtypes,
        })
        return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
