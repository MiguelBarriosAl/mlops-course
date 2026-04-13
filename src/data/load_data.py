"""
This module loads the raw dataset and introduces controlled noise to simulate
real-world data issues for the data engineering pipeline.
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_raw_data() -> pd.DataFrame:
    """Load the California Housing dataset and return it as a DataFrame."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    return df


def introduce_noise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Introduce controlled noise to simulate real-world data problems.

    Args:
        df: Input DataFrame.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with noise introduced.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Add missing values (~10%) to MedInc and AveRooms
    for col in ["MedInc", "AveRooms"]:
        missing_mask = rng.random(len(df)) < 0.10
        df.loc[missing_mask, col] = np.nan

    # Add outliers (~2%) to Population by multiplying by 10
    outlier_mask = rng.random(len(df)) < 0.02
    df.loc[outlier_mask, "Population"] = df.loc[outlier_mask, "Population"] * 10

    # Convert HouseAge to string to simulate incorrect data type
    df["HouseAge"] = df["HouseAge"].astype(str)

    # Add categorical feature based on Latitude
    df["LocationCluster"] = pd.cut(
        df["Latitude"],
        bins=3,
        labels=["Low", "Mid", "High"],
    )

    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a CSV file, creating directories as needed.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the data ingestion pipeline."""
    output_path = "data/raw/housing_noisy.csv"

    print("Loading dataset...")
    df = load_raw_data()

    print("Introducing noise...")
    df_noisy = introduce_noise(df)

    print("Saving dataset...")
    save_dataset(df_noisy, output_path)

    print(f"Dataset shape: {df_noisy.shape}")
    print(f"Output path: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
