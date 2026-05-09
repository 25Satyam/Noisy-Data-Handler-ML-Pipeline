"""
tests/test_pipeline.py
-----------------------
Integration tests for the full NoisyDataPipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_loader import DataLoader
from src.pipeline import NoisyDataPipeline


@pytest.fixture
def demo_classification_data():
    loader = DataLoader()
    df, target = loader.load_demo(task="classification", n_samples=300)
    return df, target


@pytest.fixture
def demo_regression_data():
    loader = DataLoader()
    df, target = loader.load_demo(task="regression", n_samples=300)
    return df, target


class TestNoisyDataPipelineClassification:

    def test_fit_transform_returns_dataframe(self, demo_classification_data):
        df, target = demo_classification_data
        pipeline = NoisyDataPipeline(target_col=target, config_path="config/config.yaml")
        X, y = pipeline.fit_transform(df)
        assert isinstance(X, pd.DataFrame)
        assert y is not None

    def test_no_missing_values_after_pipeline(self, demo_classification_data):
        df, target = demo_classification_data
        pipeline = NoisyDataPipeline(target_col=target, config_path="config/config.yaml")
        X, y = pipeline.fit_transform(df)
        assert X.isna().sum().sum() == 0

    def test_X_and_y_same_length(self, demo_classification_data):
        df, target = demo_classification_data
        pipeline = NoisyDataPipeline(target_col=target, config_path="config/config.yaml")
        X, y = pipeline.fit_transform(df)
        assert len(X) == len(y)

    def test_target_not_in_X(self, demo_classification_data):
        df, target = demo_classification_data
        pipeline = NoisyDataPipeline(target_col=target, config_path="config/config.yaml")
        X, _ = pipeline.fit_transform(df)
        assert target not in X.columns

    def test_transform_before_fit_raises(self, demo_classification_data):
        df, target = demo_classification_data
        pipeline = NoisyDataPipeline(target_col=target)
        with pytest.raises(RuntimeError):
            pipeline.transform(df)

    def test_pipeline_summary_returns_dataframe(self, demo_classification_data):
        df, target = demo_classification_data
        pipeline = NoisyDataPipeline(target_col=target, config_path="config/config.yaml")
        pipeline.fit(df)
        summary = pipeline.pipeline_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0


class TestNoisyDataPipelineRegression:

    def test_fit_transform_regression(self, demo_regression_data):
        df, target = demo_regression_data
        pipeline = NoisyDataPipeline(target_col=target, config_path="config/config.yaml")
        X, y = pipeline.fit_transform(df)
        assert isinstance(X, pd.DataFrame)
        assert y is not None
        assert X.isna().sum().sum() == 0


class TestDataLoader:

    def test_load_demo_classification(self):
        loader = DataLoader()
        df, target = loader.load_demo(task="classification", n_samples=200)
        assert isinstance(df, pd.DataFrame)
        assert target in df.columns
        assert len(df) > 0

    def test_load_demo_regression(self):
        loader = DataLoader()
        df, target = loader.load_demo(task="regression", n_samples=200)
        assert target in df.columns

    def test_summary_returns_dataframe(self):
        loader = DataLoader()
        loader.load_demo()
        summary = loader.summary()
        assert isinstance(summary, pd.DataFrame)

    def test_load_nonexistent_file_raises(self):
        loader = DataLoader(filepath="nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()
