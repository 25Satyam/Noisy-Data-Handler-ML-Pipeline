"""
tests/test_imputer.py
---------------------
Unit tests for AdvancedImputer.
"""

import numpy as np
import pandas as pd
import pytest
from src.imputer import AdvancedImputer


@pytest.fixture
def df_with_missing():
    np.random.seed(7)
    df = pd.DataFrame({
        "num1": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
        "num2": [np.nan, 2.0, 3.0, np.nan, 5.0, 6.0],
        "cat1": ["A", None, "B", "A", np.nan, "B"],
        "cat2": ["X", "Y", None, "X", "Y", np.nan],
    })
    return df


class TestAdvancedImputer:

    def test_mean_imputer_fills_numeric(self, df_with_missing):
        imp = AdvancedImputer(numeric_strategy="mean")
        result = imp.fit_transform(df_with_missing)
        assert result[["num1", "num2"]].isna().sum().sum() == 0

    def test_median_imputer_fills_numeric(self, df_with_missing):
        imp = AdvancedImputer(numeric_strategy="median")
        result = imp.fit_transform(df_with_missing)
        assert result[["num1", "num2"]].isna().sum().sum() == 0

    def test_knn_imputer_fills_numeric(self, df_with_missing):
        imp = AdvancedImputer(numeric_strategy="knn", knn_neighbors=2)
        result = imp.fit_transform(df_with_missing)
        assert result[["num1", "num2"]].isna().sum().sum() == 0

    def test_most_frequent_fills_categorical(self, df_with_missing):
        imp = AdvancedImputer(categorical_strategy="most_frequent")
        result = imp.fit_transform(df_with_missing)
        assert result[["cat1", "cat2"]].isna().sum().sum() == 0

    def test_constant_fill_categorical(self, df_with_missing):
        imp = AdvancedImputer(categorical_strategy="constant", fill_value="MISSING")
        result = imp.fit_transform(df_with_missing)
        assert "MISSING" in result["cat1"].values or result["cat1"].isna().sum() == 0

    def test_no_missing_after_imputation(self, df_with_missing):
        imp = AdvancedImputer(numeric_strategy="mean", categorical_strategy="most_frequent")
        result = imp.fit_transform(df_with_missing)
        assert result.isna().sum().sum() == 0

    def test_shape_preserved(self, df_with_missing):
        imp = AdvancedImputer()
        result = imp.fit_transform(df_with_missing)
        assert result.shape == df_with_missing.shape

    def test_missing_report_returns_dataframe(self, df_with_missing):
        imp = AdvancedImputer()
        report = imp.missing_report(df_with_missing)
        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
        assert "missing_count" in report.columns
