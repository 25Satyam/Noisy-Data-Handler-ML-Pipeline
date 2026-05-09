"""
data_loader.py
--------------
Handles loading data from multiple file formats (CSV, JSON, Excel, Parquet)
and provides a built-in noisy demo dataset generator for testing.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads datasets from various file formats.
    Also provides a synthetic noisy dataset generator for demo/testing.
    """

    SUPPORTED_FORMATS = {
        ".csv": "_load_csv",
        ".json": "_load_json",
        ".xlsx": "_load_excel",
        ".xls": "_load_excel",
        ".parquet": "_load_parquet",
    }

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        """Load data from self.filepath; infers format from extension."""
        if self.filepath is None:
            raise ValueError("No filepath provided. Use load_demo() instead.")

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        ext = os.path.splitext(self.filepath)[-1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {list(self.SUPPORTED_FORMATS)}"
            )

        loader_method = getattr(self, self.SUPPORTED_FORMATS[ext])
        self.df = loader_method(self.filepath)
        logger.info(f"Loaded {self.df.shape[0]} rows × {self.df.shape[1]} cols from '{self.filepath}'")
        return self.df

    def load_demo(self, task: str = "classification", n_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
        """
        Generate a synthetic noisy dataset for demo/testing.

        Returns
        -------
        df : pd.DataFrame  — the noisy dataset
        target_col : str   — name of the target column
        """
        logger.info(f"Generating synthetic noisy dataset (task={task}, n={n_samples}) ...")
        np.random.seed(42)

        if task == "classification":
            df, target = self._make_noisy_classification(n_samples)
        else:
            df, target = self._make_noisy_regression(n_samples)

        self.df = df
        logger.info(f"Demo dataset shape: {df.shape}")
        return df, target

    # ------------------------------------------------------------------ #
    #  Private loaders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def _load_json(path: str) -> pd.DataFrame:
        return pd.read_json(path)

    @staticmethod
    def _load_excel(path: str) -> pd.DataFrame:
        return pd.read_excel(path)

    @staticmethod
    def _load_parquet(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    # ------------------------------------------------------------------ #
    #  Demo dataset generators
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_noisy_classification(n: int) -> Tuple[pd.DataFrame, str]:
        """Breast-cancer-style dataset with injected noise."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_clusters_per_class=2,
            flip_y=0.03,
            random_state=42,
        )

        cols = [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry",
            "fractal_dim", "age", "bmi", "blood_pressure",
            "cholesterol", "glucose",
        ]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y

        # --- Inject realistic noise ---

        # 1. Missing values (random ~12%)
        for col in ["texture", "area", "bmi", "cholesterol", "glucose"]:
            mask = np.random.rand(n) < 0.12
            df.loc[mask, col] = np.nan

        # 2. Outliers via extreme values
        outlier_idx = np.random.choice(n, size=int(n * 0.04), replace=False)
        df.loc[outlier_idx, "radius"] = df["radius"].mean() + 15 * df["radius"].std()
        df.loc[outlier_idx[:10], "perimeter"] = -999

        # 3. Categorical column with noise
        categories = ["low", "medium", "high", "very_high", None, "UNKNOWN", "low"]
        df["risk_level"] = np.random.choice(categories, size=n)

        # 4. Duplicate rows (3%)
        dup_rows = df.sample(frac=0.03, random_state=1)
        df = pd.concat([df, dup_rows], ignore_index=True)

        # 5. Constant (near-zero variance) column
        df["constant_col"] = 1.0

        # 6. Highly skewed column
        df["skewed_feature"] = np.random.exponential(scale=2.0, size=len(df))

        return df.reset_index(drop=True), "target"

    @staticmethod
    def _make_noisy_regression(n: int) -> Tuple[pd.DataFrame, str]:
        """Housing-price-style dataset with injected noise."""
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=n,
            n_features=12,
            n_informative=8,
            noise=15,
            random_state=42,
        )

        cols = [
            "lot_size", "house_age", "num_rooms", "num_bathrooms",
            "garage_size", "distance_school", "distance_hospital",
            "crime_rate", "air_quality", "green_space",
            "public_transport", "employment_rate",
        ]
        df = pd.DataFrame(X, columns=cols)
        df["price"] = y

        # Missing values
        for col in ["garage_size", "air_quality", "green_space"]:
            mask = np.random.rand(n) < 0.1
            df.loc[mask, col] = np.nan

        # Outliers
        outlier_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[outlier_idx, "lot_size"] = df["lot_size"].max() * 10

        # Categorical
        df["neighborhood"] = np.random.choice(
            ["urban", "suburban", "rural", None, "UNKNOWN"], size=n
        )

        # Duplicates
        dup_rows = df.sample(frac=0.02, random_state=2)
        df = pd.concat([df, dup_rows], ignore_index=True)

        df["constant_col"] = 0.0
        df["skewed_feature"] = np.random.exponential(scale=3.0, size=len(df))

        return df.reset_index(drop=True), "price"

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def summary(self) -> pd.DataFrame:
        """Return a diagnostic summary of the loaded DataFrame."""
        if self.df is None:
            raise RuntimeError("No data loaded yet.")
        df = self.df
        summary = pd.DataFrame({
            "dtype": df.dtypes,
            "non_null": df.notna().sum(),
            "null_count": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
            "sample": df.iloc[0],
        })
        return summary

    def save(self, path: str) -> None:
        """Save the current DataFrame to CSV."""
        if self.df is None:
            raise RuntimeError("No data to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        self.df.to_csv(path, index=False)
        logger.info(f"Saved to {path}")
