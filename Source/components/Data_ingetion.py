import pandas as pd
import os 
import re
#from logger import logging
#from Source.exception import CustomException

class DataIngetion():
    pass
import os
import pandas as pd

def load_data(file_path='Data/data.csv'):
    """
    Loads data from a CSV file.

    Args
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        print(f"[INFO] Loading data from: {file_path}")
        df = pd.read_csv(file_path)

        print(f"[INFO] Data shape: {df.shape}")
        print(f"[INFO] Columns: {df.columns.tolist()}")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"[WARNING] Data contains {missing} missing values.")
        else:
            print("[INFO] No missing values found.")

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        raise

if __name__ == "__main__":
    df = load_data()
    print(df.head())  # Preview the first 5 rows




    