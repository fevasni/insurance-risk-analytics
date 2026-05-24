import logging
import os
import sys
import pandas as pd

# Set up logging to track errors cleanly in the console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Loads an insurance CSV dataset and applies strict production sanity checks."""
    # Define columns your analysis or model absolutely depends on
    expected_columns = {"age", "charges", "bmi"}

    try:
        logging.info(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)

        # 1. Check if the file has data
        if df.empty:
            raise pd.errors.EmptyDataError("The dataset file is completely empty.")

        # 2. Schema Validation: Check for missing columns
        missing_cols = expected_columns - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Schema corruption! Missing required columns: {missing_cols}"
            )

        # 3. Data Integrity Validation: Check for impossible values
        if (df["age"] <= 0).any():
            logging.warning("Anomaly Warning: Found rows where age is <= 0.")

        if (df["charges"] < 0).any():
            logging.warning(
                "Anomaly Warning: Found rows where insurance charges are negative."
            )

        logging.info("Dataset successfully validated.")
        return df

    except FileNotFoundError:
        logging.error(f"Critical Failure: The file '{file_path}' does not exist.")
        sys.exit(1)  # Aborts the pipeline immediately so bad data doesn't propagate
    except pd.errors.EmptyDataError as ede:
        logging.error(f"Data Integrity Failure: {ede}")
        sys.exit(1)
    except KeyError as ke:
        logging.error(f"Schema Integrity Failure: {ke}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unexpected pipeline disruption: {e}")
        sys.exit(1)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning transformations safely."""
    try:
        logging.info("Starting data cleaning operations...")

        # Example transformation: Remove duplicates safely
        initial_rows = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {initial_rows - len(df)} duplicate rows.")

        # Handle missing values safely without crashing
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

        return df
    except Exception as e:
        logging.error(f"Error during cleaning phase: {e}")
        sys.exit(1)


if __name__ == "__main__":
    input_path = "data/insurance_data.csv"
    output_dir = "data"
    output_path = os.path.join(output_dir, "cleaned_insurance_data.csv")

    # Run the hardened pipeline
    raw_df = load_and_validate_data(input_path)
    cleaned_df = clean_data(raw_df)

    # Save output dataset
    cleaned_df.to_csv(output_path, index=False)
    logging.info(f"Pipeline finished. Cleaned data saved to {output_path}")