import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load diabetes dataset"""
    return pd.read_csv(file_path)

def handle_zeros(df, columns):
    """Replace zero values with median"""
    df_clean = df.copy()
    for col in columns:
        # Replace zeros with NaN then with median
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    return df_clean

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    # Define columns that shouldn't have zero values
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Handle zero values
    df_clean = handle_zeros(df, zero_columns)
    
    # Separate features and target
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Example usage
    df = load_data('../data/raw/diabetes.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print("Data preprocessing completed successfully!")