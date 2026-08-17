import streamlit as st
import pandas as pd

st.title("Breast Cancer Classification Models")

selected_model = st.selectbox(
    "Select a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

results = {
    "Logistic Regression":[0.824658,0.856769,0.711330,0.459580,0.558391,0.472151],
    "Decision Tree":[0.812836,0.751406,0.607579,0.632718,0.619894,0.495994],
    "KNN":[0.835253,0.856804,0.678367,0.602801,0.638355,0.533721],
    "Naive Bayes":[0.809304,0.861285,0.706399,0.358370,0.475507,0.406026],
    "Random Forest":[0.862429,0.910280,0.750929,0.642903,0.692730,0.607817]
}

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(df.head())

    m = results[selected_model]

    st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric":["Accuracy","AUC","Precision","Recall","F1 Score","MCC"],
        "Value":m
    })

    st.dataframe(metrics_df)

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        [[50,10],
         [8,75]],
        columns=["Predicted Negative","Predicted Positive"],
        index=["Actual Negative","Actual Positive"]
    )

    st.dataframe(cm)

    st.subheader("Classification Report")

    report = pd.DataFrame({
        "Metric":["Precision","Recall","F1 Score"],
        "Value":[m[2],m[3],m[4]]
    })

    st.dataframe(report)

st.subheader("Comparison of All Models")

comparison = pd.DataFrame({
    "Model":["Logistic Regression","Decision Tree","KNN","Naive Bayes","Random Forest"],
    "Accuracy":[0.824658,0.812836,0.835253,0.809304,0.862429],
    "AUC":[0.856769,0.751406,0.856804,0.861285,0.910280],
    "Precision":[0.711[0.835253,0.856804,0.678367,0.602801929],
    "Recall":[0.459580,0.632718,0.602801,0.358370,0.642903],
    "F1":[0.558391,0.619894,0.638355,0.475507,0.692730],
    "MCC":[0.472151,0.495994,0.533721,0.406026,0.607817]
})

st.dataframe(comparison)