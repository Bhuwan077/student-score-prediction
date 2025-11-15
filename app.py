# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Score Prediction", layout="centered")

st.title("🎓 Student Score Prediction App")
st.write("Predict a student's math score using study scores & demographics. (Demo app)")

DATA_FILE = "StudentsPerformance.csv"
MODEL_FILE = "model.pkl"
ENCODER_FILE = "encoder.pkl"
MODEL_COLS_FILE = "model_columns.pkl"

# -------------------------
# 1) Load dataset
# -------------------------
if not os.path.exists(DATA_FILE):
    st.error(f"Dataset file '{DATA_FILE}' not found in project folder.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Basic sanity
if "math score" not in df.columns:
    st.error("Dataset does not contain 'math score' column. Check the CSV.")
    st.stop()

# -------------------------
# 2) Load model artifacts or train & save them if missing
# -------------------------
def train_and_save():
    st.info("Training model (only runs if model files are missing)...")
    X = df.drop("math score", axis=1)
    y = df["math score"]

    categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    numeric_cols = ["reading score", "writing score"]

    # Fit encoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    # create readable column names for encoded features
    encoded_col_names = []
    for i, cat_col in enumerate(categorical_cols):
        cats = encoder.categories_[i][1:]  # dropped first
        for cat in cats:
            # replace spaces/slashes to keep column names safe
            safe_cat = str(cat).replace(" ", "_").replace("/", "_")
            encoded_col_names.append(f"{cat_col}_{safe_cat}")
    X_encoded.columns = encoded_col_names
    X_encoded.index = X.index

    X_final = pd.concat([X[numeric_cols], X_encoded], axis=1)
    # ensure column names are strings
    X_final.columns = X_final.columns.astype(str)

    # Train
    model = LinearRegression()
    model.fit(X_final, y)

    # Save artifacts
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(encoder, f)
    with open(MODEL_COLS_FILE, "wb") as f:
        pickle.dump(X_final.columns.tolist(), f)

    st.success("Training complete — model, encoder and column list saved.")
    return model, encoder, X_final.columns.tolist()

# try load
model = encoder = None
model_columns = None
missing = []
for fname in (MODEL_FILE, ENCODER_FILE, MODEL_COLS_FILE):
    if not os.path.exists(fname):
        missing.append(fname)

if missing:
    model, encoder, model_columns = train_and_save()
else:
    # load them
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        encoder = pickle.load(f)
    with open(MODEL_COLS_FILE, "rb") as f:
        model_columns = pickle.load(f)

# -------------------------
# 3) Show graphs (dataset insights)
# -------------------------
st.subheader("Dataset overview & visualizations")

# Math score distribution
fig, ax = plt.subplots()
sns.histplot(df["math score"], kde=True, ax=ax)
ax.set_xlabel("Math score")
st.pyplot(fig)

# Prepare X_final from full df for actual vs predicted
categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
numeric_cols = ["reading score", "writing score"]

# transform categorical using saved encoder
try:
    X_enc_full = pd.DataFrame(encoder.transform(df[categorical_cols]))
except Exception as e:
    st.warning("Encoder transform failed for full dataset — will retrain encoder.")
    # fallback: retrain quickly
    model, encoder, model_columns = train_and_save()
    X_enc_full = pd.DataFrame(encoder.transform(df[categorical_cols]))

# build column names for full encoded
encoded_names = []
for i, cat_col in enumerate(categorical_cols):
    cats = encoder.categories_[i][1:]
    for cat in cats:
        safe_cat = str(cat).replace(" ", "_").replace("/", "_")
        encoded_names.append(f"{cat_col}_{safe_cat}")

X_enc_full.columns = encoded_names
X_full = pd.concat([df[numeric_cols].reset_index(drop=True), X_enc_full.reset_index(drop=True)], axis=1)

# reindex to model_columns to ensure order
X_full = X_full.reindex(columns=model_columns, fill_value=0)

# predict full dataset
y_true = df["math score"].values
y_pred_full = model.predict(X_full)

st.subheader("Actual vs Predicted (on full dataset)")
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.scatter(y_true, y_pred_full, alpha=0.6)
ax2.plot([0,100],[0,100], color="red", linewidth=1, linestyle="--")  # ideal line
ax2.set_xlabel("Actual Math Score")
ax2.set_ylabel("Predicted Math Score")
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
st.pyplot(fig2)

# show simple metrics
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_true, y_pred_full)
r2 = r2_score(y_true, y_pred_full)
st.markdown(f"**Model performance on full dataset** — MAE: `{mae:.2f}` , R²: `{r2:.3f}`")

# -------------------------
# 4) Prediction UI (user inputs)
# -------------------------
st.subheader("Make a prediction")

gender = st.selectbox("Gender", df["gender"].unique())
race = st.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
parent_edu = st.selectbox("Parental Level of Education", df["parental level of education"].unique())
lunch = st.selectbox("Lunch Type", df["lunch"].unique())
test_prep = st.selectbox("Test Preparation Course", df["test preparation course"].unique())
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

# Build input dict: numeric first
input_dict = {
    "reading score": [reading_score],
    "writing score": [writing_score]
}

# build a small DataFrame for categorical transform
cat_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "parental level of education": [parent_edu],
    "lunch": [lunch],
    "test preparation course": [test_prep]
})

# transform categorical with encoder
try:
    enc_values = encoder.transform(cat_df[categorical_cols])
except Exception as e:
    st.error("Encoding the inputs failed. Try retraining or check encoder. Error: " + str(e))
    st.stop()

# enc_values is array with columns equal to encoded_names
enc_df = pd.DataFrame(enc_values, columns=encoded_names)

# combine and align to model columns
input_full = pd.concat([pd.DataFrame(input_dict), enc_df], axis=1)
input_full = input_full.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Math Score"):
    pred = model.predict(input_full)[0]
    st.success(f"Predicted Math Score: {pred:.2f}")

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.markdown("**Note:** The model is a simple Linear Regression trained on the small Kaggle StudentsPerformance dataset. This demo is for learning and demo purposes.")
