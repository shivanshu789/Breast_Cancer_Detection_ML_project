# Source/components/Model_Evaluation.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

def load_data():
    # Load transformed test data (assumes it's already prepared)
    data_path = "processed/transformed_data.csv"
    data = pd.read_csv(data_path)
    X = data.drop(columns=["id", "diagnosis"])  # Drop ID and target
    y = data["diagnosis"]  # Target column
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test

def evaluate_model():
    # Load saved model
    model = joblib.load("models/adaboost_model.pkl")
    
    # Load test data
    X_test, y_test = load_data()

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability for ROC

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {acc:.4f}")

    # Classification report
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\n Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"\n ROC-AUC Score: {auc_score:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Feature importances (AdaBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_test.columns

        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        print("\n Top Feature Importances:")
        print(fi_df.head())

        # Plot
        fi_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title("Top 10 Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    evaluate_model()
