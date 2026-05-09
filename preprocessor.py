"""
preprocessor.py
---------------
BasicPreprocessor — handles the first-pass cleaning of a raw DataFrame:
  - Standardise column names
  - Remove duplicate rows
  - Drop high-missing-value columns
  - Separate numeric vs categorical columns
  - Remove near-zero variance features
  - Fix mixed-type columns
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class BasicPreprocessor:
    """
    First-pass data cleaning transformer.

    Parameters
    ----------
    missing_threshold : float
        Drop columns where the fraction of NaNs exceeds this value. Default 0.5.
    low_variance_threshold : float
        Drop numeric columns whose variance is below this value. Default 0.01.
    duplicate_action : str
        'drop' to remove duplicate rows, 'flag' to add a boolean column. Default 'drop'.
    target_col : str or None
        Target column name — excluded from all dropping logic.
    """

    def __init__(
        self,
        missing_threshold: float = 0.5,
        low_variance_threshold: float = 0.01,
        duplicate_action: str = "drop",
        target_col: Optional[str] = None,
    ):
        self.missing_threshold = missing_threshold
        self.low_variance_threshold = low_variance_threshold
        self.duplicate_action = duplicate_action
        self.target_col = target_col

        # Tracked during fit
        self.columns_to_drop_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.original_columns_: List[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "BasicPreprocessor":
        """Learn which columns to drop and categorise all features."""
        self.original_columns_ = df.columns.tolist()
        self.columns_to_drop_ = []

        feature_cols = [c for c in df.columns if c != self.target_col]

        # 1. High-missing columns
        null_fractions = df[feature_cols].isna().mean()
        high_null = null_fractions[null_fractions > self.missing_threshold].index.tolist()
        self.columns_to_drop_.extend(high_null)
        if high_null:
            logger.info(f"High-missing columns flagged for drop: {high_null}")

        # 2. Near-zero variance numeric columns
        remaining = [c for c in feature_cols if c not in self.columns_to_drop_]
        num_cols = df[remaining].select_dtypes(include=[np.number]).columns.tolist()
        low_var = [
            c for c in num_cols
            if df[c].var(ddof=0) < self.low_variance_threshold
        ]
        self.columns_to_drop_.extend(low_var)
        if low_var:
            logger.info(f"Near-zero variance columns flagged for drop: {low_var}")

        # 3. Catalogue surviving columns
        keep = [c for c in feature_cols if c not in self.columns_to_drop_]
        self.numeric_cols_ = df[keep].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = df[keep].select_dtypes(exclude=[np.number]).columns.tolist()

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules to df."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()

        # Standardise column names
        df = self._standardise_names(df)

        # Handle duplicates
        n_before = len(df)
        if self.duplicate_action == "drop":
            df = df.drop_duplicates()
            dropped = n_before - len(df)
            if dropped:
                logger.info(f"Removed {dropped} duplicate rows.")
        elif self.duplicate_action == "flag":
            df["is_duplicate"] = df.duplicated()

        # Fix mixed-type columns
        df = self._fix_mixed_types(df)

        # Drop unwanted columns (update names after standardising)
        cols_to_drop = [self._std_name(c) for c in self.columns_to_drop_]
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        if cols_to_drop:
            logger.info(f"Dropped columns: {cols_to_drop}")

        return df.reset_index(drop=True)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _std_name(name: str) -> str:
        return (
            name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
        )

    def _standardise_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [self._std_name(c) for c in df.columns]
        return df

    @staticmethod
    def _fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        For object columns that are actually numeric, coerce to float.
        Strings like 'UNKNOWN', 'N/A', 'nan' become NaN.
        """
        na_strings = {"nan", "none", "null", "n/a", "na", "unknown", "missing", ""}
        for col in df.select_dtypes(include="object").columns:
            # Try numeric coercion
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0.5 * df[col].notna().sum():
                df[col] = coerced
            else:
                # Clean up na-like strings
                df[col] = df[col].apply(
                    lambda x: np.nan
                    if isinstance(x, str) and x.strip().lower() in na_strings
                    else x
                )
        return df

    def get_feature_types(self) -> Tuple[List[str], List[str]]:
        """Return (numeric_cols, categorical_cols) identified during fit."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        return self.numeric_cols_, self.categorical_cols_

    def report(self) -> dict:
        return {
            "columns_dropped": self.columns_to_drop_,
            "numeric_columns": self.numeric_cols_,
            "categorical_columns": self.categorical_cols_,
        }
