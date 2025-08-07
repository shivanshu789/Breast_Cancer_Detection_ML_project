import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Define file paths
PROCESSED_DATA_PATH = "Data/data.csv"
TRANSFORMED_DATA_PATH = "processed/transformed_data.csv"

def load_data(path):
    """Load data from the given path"""
    return pd.read_csv(path)

def encode_categorical(df):
    """Encode categorical columns like 'diagnosis'"""
    if 'diagnosis' in df.columns:
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M=1, B=0
    return df

def scale_features(df):
    """Scale numerical features"""
    scaler = StandardScaler()

    # Drop unnecessary columns and handle missing values
    features_to_scale = df.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors='ignore')

    # Remove any remaining columns with all NaN values
    features_to_scale = features_to_scale.dropna(axis=1, how='all')

    scaled = scaler.fit_transform(features_to_scale)

    scaled_df = pd.DataFrame(scaled, columns=features_to_scale.columns)

    # Add back 'id' and 'diagnosis' columns if they exist
    if 'id' in df.columns:
        scaled_df.insert(0, 'id', df['id'].values)
    if 'diagnosis' in df.columns:
        scaled_df.insert(1, 'diagnosis', df['diagnosis'].values)

    return scaled_df

def save_transformed_data(df, path):
    """Save transformed data to the given path"""
    df.to_csv(path, index=False)
    print(f" Transformed data saved to: {path}")

def main():
    print(" Starting data transformation...")
    df = load_data(PROCESSED_DATA_PATH)

    df = encode_categorical(df)
    df = scale_features(df)

    save_transformed_data(df, TRANSFORMED_DATA_PATH)
    print(" Data transformation complete.")

if __name__ == "__main__":
    main()