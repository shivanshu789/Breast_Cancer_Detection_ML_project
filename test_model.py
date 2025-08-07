#!/usr/bin/env python3
"""
Test script to run the prediction model and identify any errors
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def test_data_loading():
    """Test data loading functionality"""
    print("=" * 50)
    print("Testing Data Loading...")
    try:
        # Load data
        df = pd.read_csv("Data/data.csv")
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        print(f"✓ Columns: {len(df.columns)}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠ Warning: {missing_values} missing values found")
        else:
            print("✓ No missing values")
            
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def test_data_preprocessing(df):
    """Test data preprocessing"""
    print("=" * 50)
    print("Testing Data Preprocessing...")
    try:
        # Drop unnecessary columns
        df_clean = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
        print("✓ Dropped unnecessary columns")
        
        # Encode diagnosis column
        if 'diagnosis' in df_clean.columns:
            df_clean['diagnosis'] = df_clean['diagnosis'].map({'M': 1, 'B': 0})
            print("✓ Encoded diagnosis column")
        
        # Split features and target
        X = df_clean.drop('diagnosis', axis=1)
        y = df_clean['diagnosis']
        print(f"✓ Split data - Features: {X.shape}, Target: {y.shape}")
        
        return X, y
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return None, None

def test_model_training(X, y):
    """Test model training"""
    print("=" * 50)
    print("Testing Model Training...")
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("✓ Data split into train/test sets")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✓ Features scaled")
        
        # Train AdaBoost model
        base_estimator = DecisionTreeClassifier(max_depth=1)
        ada_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=1.0, random_state=42)
        ada_model.fit(X_train_scaled, y_train)
        print("✓ AdaBoost model trained")
        
        # Train XGBoost model
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        print("✓ XGBoost model trained")
        
        return ada_model, xgb_model, scaler, X_test_scaled, y_test
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return None, None, None, None, None

def test_model_evaluation(ada_model, xgb_model, X_test_scaled, y_test):
    """Test model evaluation"""
    print("=" * 50)
    print("Testing Model Evaluation...")
    try:
        # AdaBoost predictions
        ada_pred = ada_model.predict(X_test_scaled)
        ada_accuracy = accuracy_score(y_test, ada_pred)
        print(f"✓ AdaBoost Accuracy: {ada_accuracy:.4f}")
        
        # XGBoost predictions
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"✓ XGBoost Accuracy: {xgb_accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error in model evaluation: {e}")
        return False

def test_prediction_on_sample(ada_model, xgb_model, scaler, X):
    """Test prediction on sample data"""
    print("=" * 50)
    print("Testing Sample Predictions...")
    try:
        # Create sample data (first 3 rows)
        sample_data = X.head(3)
        sample_scaled = scaler.transform(sample_data)
        
        # AdaBoost predictions
        ada_pred = ada_model.predict(sample_scaled)
        ada_prob = ada_model.predict_proba(sample_scaled)
        print(f"✓ AdaBoost predictions: {ada_pred}")
        
        # XGBoost predictions
        xgb_pred = xgb_model.predict(sample_scaled)
        xgb_prob = xgb_model.predict_proba(sample_scaled)
        print(f"✓ XGBoost predictions: {xgb_pred}")
        
        return True
    except Exception as e:
        print(f"✗ Error in sample predictions: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Starting Prediction Model Tests...")
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        return
    
    # Test preprocessing
    X, y = test_data_preprocessing(df)
    if X is None or y is None:
        return
    
    # Test model training
    ada_model, xgb_model, scaler, X_test_scaled, y_test = test_model_training(X, y)
    if ada_model is None:
        return
    
    # Test model evaluation
    if not test_model_evaluation(ada_model, xgb_model, X_test_scaled, y_test):
        return
    
    # Test sample predictions
    if not test_prediction_on_sample(ada_model, xgb_model, scaler, X):
        return
    
    print("=" * 50)
    print("🎉 All tests passed successfully!")
    print("The prediction model is working correctly.")

if __name__ == "__main__":
    main()
