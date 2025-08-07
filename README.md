# Breast Cancer Detection Using AdaBoost

This project implements a complete machine learning pipeline to detect breast cancer based on diagnostic measurements. The model used is **AdaBoost**, which is known for its performance in classification tasks. The objective is to differentiate between malignant and benign tumors using structured data.

---

## Project Objective

The aim of this project is to build a predictive model that assists healthcare professionals in identifying whether a tumor is **benign (non-cancerous)** or **malignant (cancerous)** based on diagnostic features extracted from digitized images.

---

## Dataset Description

- **Source**: Breast Cancer Wisconsin Diagnostic Dataset  
- **Features**: 30 numerical features computed from digitized images of a breast mass (e.g., radius, texture, perimeter, area, smoothness)
- **Target Variable**:
  - `0`: Benign
  - `1`: Malignant

---

## Machine Learning Pipeline

### 1. Data Ingestion
- Loads the dataset from a CSV file (`data.csv`)
- Splits the data into training and testing sets
- Saves processed data to the `artifacts/` directory

### 2. Data Preprocessing
- Handles missing or inconsistent values
- Standardizes feature values using `StandardScaler`
- Transformed data is stored for modeling

### 3. Exploratory Data Analysis (EDA)
- Summarizes the dataset
- Checks for feature distributions and class balance
- Correlation analysis is performed to identify important variables

### 4. Model Training
- **Algorithm**: AdaBoostClassifier (Adaptive Boosting)
- Trained using the transformed training dataset
- Trained model is saved as `models/adaboost_model.pkl`

### 5. Model Evaluation
- Evaluates the trained model on the test dataset
- Outputs include:
  - Accuracy
  - Confusion matrix
  - Classification report
  - ROC-AUC score
  - Feature importance rankings

---

## Model Used

- **AdaBoostClassifier**
- Base Estimator: DecisionTreeClassifier (default settings)
- Chosen due to its ability to combine multiple weak learners into a strong learner, increasing both accuracy and robustness.

---

## Evaluation Metrics

| Metric              | Score    |
|---------------------|----------|
| Accuracy            | 97.37%   |
| Precision (Class 1) | 100%     |
| Recall (Class 1)    | 93%      |
| ROC-AUC Score       | 0.9854   |

The high accuracy and ROC-AUC score indicate that the model performs well in distinguishing between malignant and benign tumors.

---

## Top Feature Importances

| Feature                | Importance |
|------------------------|------------|
| concave points_worst   | 0.1044     |
| texture_mean           | 0.0935     |
| compactness_se         | 0.0891     |
| perimeter_worst        | 0.0857     |
| area_se                | 0.0774     |

These features contribute most significantly to the model's prediction capability.

---

## Project Structure

prediction_model/
├── data/
│ └── data.csv
├── artifacts/
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
├── models/
│ └── adaboost_model.pkl
├── Source/
│ └── components/
│ ├── Data_Ingestion.py
│ ├── Data_Transformation.py
│ ├── Model_Training.py
│ └── Model_Evaluation.py
├── README.md



---

## Running the Project

To execute the pipeline, run the following scripts in order:

```bash
python Source/components/Data_Ingestion.py
python Source/components/Data_Transformation.py
python Source/components/Model_Training.py
python Source/components/Model_Evaluation.py


Requirements
Python 3.x

pandas

scikit-learn

matplotlib

joblib

Install required packages using:

bash
Copy
Edit

pip install -r requirements.txt


Contribution to Breast Cancer Detection
This machine learning solution supports early diagnosis by automating the classification of tumors based on diagnostic measurements. It enhances decision-making for medical professionals by reducing manual effort and improving accuracy. With the high performance of the AdaBoost model, it offers a reliable tool in the fight against breast cancer.


---

Let me know if you want:
- A **PDF version**
- A **GitHub-ready version** with license and contribution guidelines
- Additional visuals or charts added to the README
