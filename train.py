import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance

# Load dataset
try:
    df = pd.read_csv('waze.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'waze.csv' not found. Please ensure it is in the same directory.")
    exit()

# Feature engineering
df['device2'] = np.where(df['device'] == 'Android', 0, 1)
df['label2'] = np.where(df['label'] == 'churned', 1, 0)

# Features and target
X = df.drop(columns=['label', 'label2', 'device'])
y = df['label2']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# =========================
# Random Forest Classifier
# =========================
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'max_depth': [None],
    'max_features': [1.0],
    'max_samples': [1.0],
    'min_samples_leaf': [2],
    'min_samples_split': [2],
    'n_estimators': [300],
}
rf_cv = GridSearchCV(rf, rf_params, scoring='recall', cv=4)
rf_cv.fit(X_tr, y_tr) # Corrected to use the training data

# =========================
# XGBoost Classifier
# =========================
xgb = XGBClassifier(objective='binary:logistic', random_state=42, eval_metric="logloss")
xgb_params = {
    'max_depth': [6, 12],
    'min_child_weight': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [300]
}
xgb_cv = GridSearchCV(xgb, xgb_params, scoring='recall', cv=4)
xgb_cv.fit(X_tr, y_tr) # Corrected to use the training data

# Save feature importance plot
plot_importance(xgb_cv.best_estimator_)
plt.savefig("xgb_feature_importance.png")
plt.close()

# Save models
joblib.dump(rf_cv, "rf_cv_model.pkl")
joblib.dump(xgb_cv, "xgb_cv_model.pkl")

# Print
print("Training complete. Models and feature importance plot saved.")
