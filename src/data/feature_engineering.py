"""
This module applies feature engineering to the cleaned dataset.
It creates new features, applies transformations, and saves the result
ready for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "train_features_v1.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering steps and return the enriched DataFrame.

    New features created:
    - rooms_per_household: average rooms divided by average occupancy
    - bedrooms_per_room: average bedrooms divided by average rooms
    - log_population: log(Population + 1) to reduce skewness
    """
    df = df.copy()

    # Ratio of rooms to household occupants
    # Use np.where to avoid division by zero (returns 0 when denominator is 0)
    df["rooms_per_household"] = np.where(
        df["AveOccup"] != 0,
        df["AveRooms"] / df["AveOccup"],
        0,
    )

    # Ratio of bedrooms to total rooms
    df["bedrooms_per_room"] = np.where(
        df["AveRooms"] != 0,
        df["AveBedrms"] / df["AveRooms"],
        0,
    )

    # Log transformation of Population to reduce the effect of large values
    df["log_population"] = np.log1p(df["Population"])

    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index.

    Raises FileExistsError if the output file already exists.
    """
    if path.exists():
        raise FileExistsError(f"Output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full feature engineering pipeline."""
    print("Loading processed data...")
    df = load_data(INPUT_PATH)
    print(f"Rows before feature engineering: {len(df)}")

    print("\nApplying feature engineering...")
    df = feature_engineering(df)

    print("\nSaving dataset...")
    save_data(df, OUTPUT_PATH)

    print(f"Final shape: {df.shape}")
    print(f"\nOutput path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
