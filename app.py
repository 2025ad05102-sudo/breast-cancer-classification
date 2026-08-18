

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification Models")

st.markdown("""
Upload a test dataset and evaluate the trained models.
""")

# -----------------------------
# Load Models
# -----------------------------
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl")
}

# -----------------------------
# Model Selection
# -----------------------------
selected_model_name = st.selectbox(
    "Select a Model",
    list(models.keys())
)

selected_model = models[selected_model_name]

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    if "diagnosis" not in df.columns:
        st.error(
            "The uploaded dataset must contain a 'diagnosis' column."
        )
        st.stop()

    # --------------------------------------
    # Encode Diagnosis
    # --------------------------------------
    df["diagnosis"] = df["diagnosis"].replace({
        "M": 1,
        "B": 0
    })

    y_test = df["diagnosis"]

    X_test = df.drop("diagnosis", axis=1)

    # Remove id column if present
    if "id" in X_test.columns:
        X_test = X_test.drop("id", axis=1)

    # Remove unnamed column if present
    if "Unnamed: 32" in X_test.columns:
        X_test = X_test.drop("Unnamed: 32", axis=1)

    # --------------------------------------
    # Predictions
    # --------------------------------------
    y_pred = selected_model.predict(X_test)

    if hasattr(selected_model, "predict_proba"):
        y_prob = selected_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    # --------------------------------------
    # Metrics
    # --------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ],
        "Value": [
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc
        ]
    })

    st.subheader("Evaluation Metrics")

    st.dataframe(metrics_df)

    # --------------------------------------
    # Confusion Matrix
    # --------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # --------------------------------------
    # Classification Report
    # --------------------------------------
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # --------------------------------------
    # Predictions
    # --------------------------------------
    st.subheader("Predictions")

    prediction_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    prediction_df["Actual"] = prediction_df["Actual"].replace({
        1: "M",
        0: "B"
    })

    prediction_df["Predicted"] = prediction_df["Predicted"].replace({
        1: "M",
        0: "B"
    })

    st.dataframe(prediction_df.head(20))

    # --------------------------------------
    # Comparison of All Models
    # --------------------------------------
    st.subheader("Comparison of All Models")

    results = []

    for model_name, model in models.items():

        pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred

        results.append([
            model_name,
            accuracy_score(y_test, pred),
            roc_auc_score(y_test, prob),
            precision_score(y_test, pred),
            recall_score(y_test, pred),
            f1_score(y_test, pred),
            matthews_corrcoef(y_test, pred)
        ])

    comparison_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]
    )

    st.dataframe(comparison_df)
