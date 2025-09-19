# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
import json
import os

# 1. Load the trained model and test data
print("Loading model and test data...")
model = joblib.load('model.pkl')

# Define your test dataset path using a relative path
file_path = 'waze.csv'
df = pd.read_csv(file_path)

# Features and target
X = df.drop(columns=['label', 'label2', 'device'])
y = df['label2']

# Split into train/test (same as train.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# 2. Make predictions
print("Evaluating model performance...")
y_pred = model.predict(X_test)

# 3. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"Precision: {precision}")

# 4. Save metrics
metrics = {
    'accuracy': accuracy,
    'f1_score': f1,
    'precision': precision
}
with open('evaluation_results.json', 'w') as f:
    json.dump(metrics, f)

print("Evaluation results saved as 'evaluation_results.json'.")
