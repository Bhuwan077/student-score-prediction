import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
print("Data Loaded Successfully\n")

# Basic info
print(df.head())
print("\nColumns:", df.columns)

# Data cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Select features & target
X = df_encoded.drop("math score", axis=1)
y = df_encoded["math score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print("MAE:", mae)
print("R² Score:", r2)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Math Scores")
plt.ylabel("Predicted Math Scores")
plt.title("Actual vs Predicted Student Scores")
plt.show()
