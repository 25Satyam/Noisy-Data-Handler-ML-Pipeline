"""
feature_engineer.py
-------------------
FeatureEngineer — transforms cleaned features into model-ready arrays:
  - Skewness correction via log / sqrt transforms
  - Numeric scaling (Standard, MinMax, Robust)
  - Categorical encoding (One-Hot, Label, Ordinal)
  - Optional polynomial / interaction feature generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer.

    Parameters
    ----------
    scaling : str
        'standard' | 'minmax' | 'robust' | 'none'
    encode_categoricals : bool
        Whether to encode object/category columns.
    encoding_strategy : str
        'onehot' | 'label' | 'ordinal'
    onehot_max_categories : int
        Switch to label encoding when cardinality exceeds this. Default 15.
    apply_log_transform : bool
        Apply log1p to numeric columns with high skewness.
    skewness_threshold : float
        Absolute skewness above which log transform is applied.
    polynomial_features : bool
        Add polynomial feature interactions.
    polynomial_degree : int
        Degree for PolynomialFeatures (used only if polynomial_features=True).
    numeric_cols : list or None
        Explicit numeric column list (auto-detected if None).
    categorical_cols : list or None
        Explicit categorical column list (auto-detected if None).
    target_col : str or None
        Column to exclude from all transforms.
    """

    def __init__(
        self,
        scaling: str = "robust",
        encode_categoricals: bool = True,
        encoding_strategy: str = "onehot",
        onehot_max_categories: int = 15,
        apply_log_transform: bool = True,
        skewness_threshold: float = 1.0,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ):
        self.scaling = scaling
        self.encode_categoricals = encode_categoricals
        self.encoding_strategy = encoding_strategy
        self.onehot_max_categories = onehot_max_categories
        self.apply_log_transform = apply_log_transform
        self.skewness_threshold = skewness_threshold
        self.polynomial_features = polynomial_features
        self.polynomial_degree = polynomial_degree
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col

        # Learned state
        self._scaler = None
        self._log_cols: List[str] = []
        self._num_cols_fit: List[str] = []
        self._cat_cols_fit: List[str] = []
        # Per-column encoders for label encoding
        self._label_encoders: Dict[str, LabelEncoder] = {}
        # One-hot encoder (fits all cat cols jointly)
        self._ohe: Optional[OneHotEncoder] = None
        self._ohe_cols: List[str] = []           # cols sent to OHE
        self._label_enc_cols: List[str] = []     # cols handled by LabelEncoder
        self._ohe_feature_names: List[str] = []

    # ------------------------------------------------------------------ #
    #  Fit
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, y=None) -> "FeatureEngineer":
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

        # Remove target from feature lists
        for lst in (self._num_cols_fit, self._cat_cols_fit):
            if self.target_col and self.target_col in lst:
                lst.remove(self.target_col)

        # Filter to columns actually present
        self._num_cols_fit = [c for c in self._num_cols_fit if c in df.columns]
        self._cat_cols_fit = [c for c in self._cat_cols_fit if c in df.columns]

        # Log transform: identify skewed columns
        if self.apply_log_transform and self._num_cols_fit:
            self._log_cols = [
                c for c in self._num_cols_fit
                if abs(df[c].skew()) > self.skewness_threshold and (df[c] >= 0).all()
            ]
            if self._log_cols:
                logger.info(f"Log-transform applied to: {self._log_cols}")

        # Scaler
        if self.scaling != "none" and self._num_cols_fit:
            self._scaler = self._build_scaler()
            # Fit on (possibly log-transformed) data
            tmp = df[self._num_cols_fit].copy()
            for col in self._log_cols:
                tmp[col] = np.log1p(tmp[col].clip(lower=0))
            self._scaler.fit(tmp)

        # Encoders
        if self.encode_categoricals and self._cat_cols_fit:
            self._fit_encoders(df)

        return self

    def _build_scaler(self):
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if self.scaling not in scalers:
            raise ValueError(f"Unknown scaling: '{self.scaling}'")
        return scalers[self.scaling]

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        if self.encoding_strategy == "onehot":
            # Split by cardinality
            self._ohe_cols = [
                c for c in self._cat_cols_fit
                if df[c].nunique() <= self.onehot_max_categories
            ]
            self._label_enc_cols = [
                c for c in self._cat_cols_fit
                if c not in self._ohe_cols
            ]
            if self._ohe_cols:
                self._ohe = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                )
                self._ohe.fit(df[self._ohe_cols].astype(str))
                self._ohe_feature_names = list(
                    self._ohe.get_feature_names_out(self._ohe_cols)
                )
            for col in self._label_enc_cols:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self._label_encoders[col] = le

        elif self.encoding_strategy in ("label", "ordinal"):
            self._label_enc_cols = self._cat_cols_fit[:]
            for col in self._label_enc_cols:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self._label_encoders[col] = le

        logger.info(
            f"Encoders fitted — OHE cols: {len(self._ohe_cols)}, "
            f"Label-encoded cols: {len(self._label_enc_cols)}"
        )

    # ------------------------------------------------------------------ #
    #  Transform
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df = df.copy()

        # 1. Log transform
        for col in self._log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        # 2. Scale numerics
        if self._scaler is not None:
            present = [c for c in self._num_cols_fit if c in df.columns]
            df[present] = self._scaler.transform(df[present])

        # 3. Encode categoricals
        if self.encode_categoricals:
            df = self._apply_encoders(df)

        # 4. Optional polynomial features (numeric only)
        if self.polynomial_features:
            df = self._apply_polynomial(df)

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    #  Encoder application
    # ------------------------------------------------------------------ #

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        # One-hot encoding
        if self._ohe is not None and self._ohe_cols:
            present = [c for c in self._ohe_cols if c in df.columns]
            ohe_array = self._ohe.transform(df[present].astype(str))
            ohe_df = pd.DataFrame(
                ohe_array,
                columns=self._ohe_feature_names,
                index=df.index,
            )
            df = df.drop(columns=present)
            df = pd.concat([df, ohe_df], axis=1)

        # Label encoding for high-cardinality / label-strategy cols
        for col in self._label_enc_cols:
            if col not in df.columns:
                continue
            le = self._label_encoders[col]
            # Map unseen labels to -1
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

        return df

    def _apply_polynomial(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col and self.target_col in num_cols:
            num_cols.remove(self.target_col)
        if not num_cols:
            return df
        pf = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=False,
            interaction_only=False,
        )
        poly_array = pf.fit_transform(df[num_cols])
        poly_df = pd.DataFrame(
            poly_array,
            columns=pf.get_feature_names_out(num_cols),
            index=df.index,
        )
        non_num = [c for c in df.columns if c not in num_cols]
        return pd.concat([df[non_num], poly_df], axis=1)

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def skewness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return skewness stats for all numeric columns."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        rows = [{"column": c, "skewness": round(df[c].skew(), 4)} for c in num_cols]
        return pd.DataFrame(rows).sort_values("skewness", key=abs, ascending=False)
