import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the MAGIC dataset
df = pd.read_csv('magicdataset.csv')

# Step 2: Data Preprocessing
# Rename columns for easier understanding
df.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

df['class'] = df['class'].map({'g': 1, 'h': 0})  # Convert class labels to 1 (gamma) and 0 (hadron)

# Splitting features and target variable
X = df.drop(columns=['class'])
y = df['class']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Model Selection & Training
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Step 6: Save the Model
joblib.dump(model, 'mace_ml_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model saved successfully!")
