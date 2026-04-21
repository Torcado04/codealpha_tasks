import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Load dataset (robust path)


base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "data", "diabetes.csv")

data = pd.read_csv(file_path)

# Basic checks

print(data.head())
print(data.isnull().sum())
print(data.columns)

# Fix missing values (IMPORTANT)
# In this dataset, 0 = missing

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols] = data[cols].replace(0, np.nan)

# Fill missing with median
data.fillna(data.median(), inplace=True)

# Features and target

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model training

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Custom prediction
# (MUST match all 8 features)

print("\nFeature order:", list(X.columns))

sample = pd.DataFrame([[
    2,      # Pregnancies
    120,    # Glucose
    70,     # BloodPressure
    20,     # SkinThickness
    85,     # Insulin
    25.0,   # BMI
    0.5,    # DiabetesPedigreeFunction
    30      # Age
]], columns=X.columns)

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("\nPrediction (0=No Diabetes, 1=Diabetes):", prediction[0])

# Info

print("\nDataset size:", data.shape)
print("Test size:", X_test.shape)