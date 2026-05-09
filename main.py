"""
main.py
-------
Command-line entry point for the Noisy Data Handler ML Pipeline.

Usage examples:
  python main.py --demo --task classification
  python main.py --input data/raw/mydata.csv --target label --task classification
  python main.py --input data/raw/prices.csv --target price --task regression --no-tune
"""

import argparse
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Noisy Data Handler — ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo --task classification
  python main.py --input data/raw/dataset.csv --target species --task classification
  python main.py --input data/raw/housing.csv --target price --task regression
        """,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Use built-in synthetic noisy dataset")
    source.add_argument("--input", type=str, help="Path to input CSV/JSON/Excel/Parquet file")

    parser.add_argument("--target", type=str, default="target", help="Target column name (default: 'target')")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="ML task type (default: classification)",
    )
    parser.add_argument("--algorithm", type=str, default="random_forest",
                        help="Model algorithm (default: random_forest)")
    parser.add_argument("--output", type=str, default="data/processed/cleaned.csv",
                        help="Path to save cleaned dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (faster)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples for demo dataset (default: 1000)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  Imports (here so --help is fast)
    # ------------------------------------------------------------------ #
    from src.data_loader import DataLoader
    from src.pipeline import NoisyDataPipeline
    from src.model import ModelTrainer

    banner = """
╔══════════════════════════════════════════════════════╗
║        🧹  Noisy Data Handler — ML Pipeline          ║
║             Python · Scikit-Learn · v1.0.0           ║
╚══════════════════════════════════════════════════════╝
"""
    print(banner)

    # ------------------------------------------------------------------ #
    #  1. Load data
    # ------------------------------------------------------------------ #
    loader = DataLoader(filepath=args.input if args.input else None)

    if args.demo or args.input is None:
        logger.info("Using built-in synthetic noisy demo dataset ...")
        df, target_col = loader.load_demo(task=args.task, n_samples=args.n_samples)
    else:
        df = loader.load()
        target_col = args.target

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target column: '{target_col}'")
    logger.info(f"Missing values before cleaning: {df.isna().sum().sum()}")

    # ------------------------------------------------------------------ #
    #  2. Run pipeline
    # ------------------------------------------------------------------ #
    pipeline = NoisyDataPipeline(target_col=target_col, config_path=args.config)

    logger.info("\n--- Running preprocessing pipeline ---")
    X, y = pipeline.fit_transform(df)

    logger.info(f"\nPost-pipeline shape: {X.shape}")
    logger.info(f"Missing values after pipeline: {X.isna().sum().sum()}")

    # Save cleaned data
    cleaned_df = X.copy()
    cleaned_df[target_col] = y.values
    pipeline.save_processed(cleaned_df, path=args.output)

    # Print pipeline summary
    print("\n📋 Pipeline Stage Summary:")
    print(pipeline.pipeline_summary().to_string(index=False))

    # ------------------------------------------------------------------ #
    #  3. Train model
    # ------------------------------------------------------------------ #
    logger.info("\n--- Training model ---")
    trainer = ModelTrainer(
        task=args.task,
        algorithm=args.algorithm,
        tune_hyperparameters=not args.no_tune,
        report_dir="reports",
    )
    trainer.train(X, y)

    # Cross-validation
    cv_results = trainer.cross_validate(X, y)
    print(f"\n📊 Cross-Validation ({trainer.cv_folds}-fold):")
    print(f"  Mean score : {cv_results['cv_mean']}")
    print(f"  Std dev    : {cv_results['cv_std']}")

    # ------------------------------------------------------------------ #
    #  4. Report
    # ------------------------------------------------------------------ #
    print("\n📈 Final Evaluation Metrics:")
    for k, v in trainer.metrics_.items():
        print(f"  {k:<22}: {v}")

    trainer.save_metrics("reports/metrics.json")

    if not args.no_report:
        report_path = trainer.generate_report()
        print(f"\n✅ HTML report saved to: {report_path}")

    print("\n✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()
