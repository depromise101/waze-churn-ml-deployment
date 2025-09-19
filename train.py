# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load your dataset from the repository folder
file_path = 'featured-waze.csv'
df = pd.read_csv(file_path)

# Add this line to print the column headers
print("DataFrame columns:", df.columns)

# 2. Prepare the data (e.g., split into features and target)
X = df.drop(columns=['label', 'label2', 'device'])
y = df['label2']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# 4. Train the ML model
print("Training the model...")
# Define your model with hyperparameters directly
model = RandomForestClassifier(random_state=42, n_estimators=300, min_samples_leaf=2, max_features=1.0)

# 5. Fit the model to the training data
model.fit(X_train, y_train)

# 6. Save the trained model to a file
print("Saving the trained model...")
joblib.dump(model, 'model.pkl')

print("Model training complete. Model saved as 'model.pkl'.")
