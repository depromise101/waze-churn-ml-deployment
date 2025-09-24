#Evaluate.py file
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. Load Data and Re-create Test Set ---
# It is crucial to load the original dataset and perform the exact same splits
# as in train.py to ensure the evaluation is on the correct, unseen data.

try:
    df = pd.read_csv('waze.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'waze.csv' not found. Please ensure it is in the same directory.")
    exit()

# Re-apply the same feature engineering as in train.py
df['device2'] = np.where(df['device'] == 'Android', 0, 1)
df['label2'] = np.where(df['label'] == 'churned', 1, 0)

# Re-define features and target
X = df.drop(columns=['label', 'label2', 'device'])
y = df['label2']

# Re-create the same test split as in train.py
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

print("Test data separated.")

# --- 2. Load the Saved Models ---
# Both models trained in train.py are loaded for evaluation.

try:
    rf_cv_model = joblib.load("rf_cv_model.pkl")
    xgb_cv_model = joblib.load("xgb_cv_model.pkl")
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Error: 'rf_cv_model.pkl' or 'xgb_cv_model.pkl' not found.")
    exit()

# --- 3. Evaluate Each Model on the Test Set ---
# This function calculates and prints key metrics for each model.

def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluates a given model on the test set and prints metrics.

    Args:
        model: The trained model to evaluate.
        model_name (str): The name of the model (e.g., "Random Forest").
        X_test (pd.DataFrame): The feature data for testing.
        y_test (pd.Series): The true target values for testing.

    Returns:
        dict: A dictionary of the calculated metrics.
    """
    print(f"\n--- Evaluating {model_name} on the test set ---")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Print the metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

    return metrics

# Evaluate both models and store their metrics
rf_final_metrics = evaluate_model(rf_cv_model, "Random Forest", X_test, y_test)
xgb_final_metrics = evaluate_model(xgb_cv_model, "XGBoost", X_test, y_test)

# --- 4. Save Final Metrics ---
# Save the final evaluation results to a JSON file for easy access.

final_metrics = {
    "random_forest_final": rf_final_metrics,
    "xgboost_final": xgb_final_metrics
}

with open("final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=4)

print("\n Final evaluation complete. Metrics saved to 'final_metrics.json'.")
