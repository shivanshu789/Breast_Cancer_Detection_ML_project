# train_model.py

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Step 1: Load the transformed data
data_path = "processed/transformed_data.csv"
data = pd.read_csv(data_path)

# Step 2: Prepare features (X) and target (y)
X = data.drop(columns=["id", "diagnosis"])  # Drop ID and target
y = data["diagnosis"]  # Target column

# Step 3: Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Initialize AdaBoost classifier
model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)

# Step 5: Fit the model
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# Step 8: Save the model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "adaboost_model.pkl"))
print(f"\n Model saved to: {model_dir}/adaboost_model.pkl")