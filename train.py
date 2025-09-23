import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier  # or your model

# Load dataset
df = pd.read_csv("waze.csv")

# Drop only columns that exist
drop_cols = [col for col in ['label', 'label2', 'device'] if col in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")

# Define target column (pick whichever is available)
target = 'label2' if 'label2' in df.columns else 'label'

y = df[target]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")
print(" Model saved as model.pkl")
