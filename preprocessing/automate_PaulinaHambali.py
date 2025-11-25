import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def preprocess_data(df):
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Dropped duplicates: {initial_shape[0] - df.shape[0]} rows removed.")

    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()

    existing_num_cols = [col for col in numerical_cols if col in df.columns]
    if existing_num_cols:
        df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])

    return df

def save_data(df, output_path):
    """Saves the processed dataframe to a CSV file."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing for Heart Failure Prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset (CSV)")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed dataset (CSV)")

    args = parser.parse_args()

    df = load_data(args.input)
    if df is not None:
        processed_df = preprocess_data(df)
        save_data(processed_df, args.output)

if __name__ == "__main__":
    main()
