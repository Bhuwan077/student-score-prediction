import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Features and target
X = df.drop("math score", axis=1)
y = df["math score"]

# One-hot encode categorical variables
categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.index = X.index

# Combine encoded categorical with numeric features
numeric_cols = ["reading score", "writing score"]
X_final = pd.concat([X[numeric_cols], X_encoded], axis=1)

# FIX: Convert all column names to strings
X_final.columns = X_final.columns.astype(str)

# Train model
model = LinearRegression()
model.fit(X_final, y)

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Model and encoder saved successfully!")
