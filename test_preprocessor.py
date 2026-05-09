"""
tests/test_preprocessor.py
---------------------------
Unit tests for BasicPreprocessor.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessor import BasicPreprocessor


@pytest.fixture
def sample_df():
    """Return a small noisy DataFrame for testing."""
    np.random.seed(0)
    df = pd.DataFrame({
        "Age": [25, 30, np.nan, 40, 25, 30],          # has NaN
        "Salary": [50000, 60000, 70000, np.nan, 50000, 60000],
        "Department": ["HR", "IT", "Finance", "IT", "HR", "IT"],
        "Score": [1, 1, 1, 1, 1, 1],                  # constant column
        "Notes": [np.nan] * 6,                         # all missing
        "target": [0, 1, 0, 1, 0, 1],
    })
    return df


class TestBasicPreprocessor:

    def test_fit_identifies_high_missing_column(self, sample_df):
        pp = BasicPreprocessor(missing_threshold=0.5, target_col="target")
        pp.fit(sample_df)
        assert "notes" in [c.lower() for c in pp.columns_to_drop_]

    def test_fit_identifies_low_variance_column(self, sample_df):
        pp = BasicPreprocessor(low_variance_threshold=0.01, target_col="target")
        pp.fit(sample_df)
        assert "score" in [c.lower() for c in pp.columns_to_drop_]

    def test_transform_removes_columns(self, sample_df):
        pp = BasicPreprocessor(target_col="target")
        result = pp.fit_transform(sample_df)
        assert "notes" not in result.columns
        assert "score" not in result.columns

    def test_transform_removes_duplicates(self, sample_df):
        pp = BasicPreprocessor(duplicate_action="drop", target_col="target")
        result = pp.fit_transform(sample_df)
        # Rows 0 & 4 and rows 1 & 5 are duplicates
        assert len(result) < len(sample_df)

    def test_flag_duplicates_action(self, sample_df):
        pp = BasicPreprocessor(duplicate_action="flag", target_col="target")
        result = pp.fit_transform(sample_df)
        assert "is_duplicate" in result.columns

    def test_target_column_preserved(self, sample_df):
        pp = BasicPreprocessor(target_col="target")
        result = pp.fit_transform(sample_df)
        assert "target" in result.columns

    def test_column_names_standardised(self):
        df = pd.DataFrame({"First Name": [1, 2], "Last-Name": [3, 4], "target": [0, 1]})
        pp = BasicPreprocessor(target_col="target")
        result = pp.fit_transform(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_report_returns_dict(self, sample_df):
        pp = BasicPreprocessor(target_col="target")
        pp.fit(sample_df)
        report = pp.report()
        assert "columns_dropped" in report
        assert "numeric_columns" in report
        assert "categorical_columns" in report

    def test_transform_before_fit_raises(self, sample_df):
        pp = BasicPreprocessor()
        with pytest.raises(RuntimeError):
            pp.transform(sample_df)
