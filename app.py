
import streamlit as st
import pandas as pd
import numpy as np

# Machine Learning Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics Imports
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Set page configurations
st.set_page_config(page_title="BC LAB: Model Evaluation App", layout="wide")

st.title("🔬 Breast Cancer Wisconsin Diagnostic Evaluation Dashboard")
st.write("Fulfill your BITS Virtual Lab Assignment steps by uploading test data and choosing a model.")

# ----------------------------------------------------
# 1. TRAIN THE MODELS AUTOMATICALLY ON GROUND TRUTH
# ----------------------------------------------------
@st.cache_resource
def train_and_cache_models():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0: Malignant, 1: Benign
    
    # Stratified Train/Test split to mimic baseline environment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Dictionary to hold models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes Classifier": GaussianNB(),
        "Ensemble Model - Random Forest": RandomForestClassifier(random_state=42)
    }
    
    # Train all models
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
    return models, scaler, data.feature_names

# Load models and pipelines
models_dict, data_scaler, feature_names = train_and_cache_models()


# ----------------------------------------------------
# 2. STREAMLIT UI: INPUT & CONFIGURATIONS
# ----------------------------------------------------
st.sidebar.header("📁 User Action Controls")

# Feature a: Dataset upload option (CSV) [Test Data Only]
uploaded_file = st.sidebar.file_uploader("Upload your test dataset (CSV Format)", type=["csv"])

# Feature b: Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "Choose ML Model for Evaluation", 
    list(models_dict.keys())
)

# Provide sample download helper so grading schema can be tested quickly
st.sidebar.markdown("---")
st.sidebar.write("💡 **No Test File?**")
if st.sidebar.button("Generate Sample Test CSV data"):
    raw_data = load_breast_cancer()
    df_sample = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    # Ensure mapping matches Kaggle format ('M' / 'B') or standard target column
    df_sample['diagnosis'] = ['M' if t == 0 else 'B' for t in raw_data.target]
    
    # Sample out 100 entries as a baseline test file
    sample_csv = df_sample.sample(100, random_state=12).to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Sample Test Data",
        data=sample_csv,
        file_name="breast_cancer_test_data.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# 3. PROCESSING PIPELINE & INFERENCE
# ----------------------------------------------------
if uploaded_file is not None:
    try:
        # Load user file
        user_df = pd.read_csv(uploaded_file)
        st.success("Test dataset loaded successfully!")
        
        # Data Cleaning: Handle Kaggle's explicit column variations
        # Drop metadata columns if they exist in the uploaded file
        cols_to_drop = ['id', 'Unnamed: 32']
        user_df = user_df.drop(columns=[c for c in cols_to_drop if c in user_df.columns])
        
        # Locate Target column variations ('diagnosis' or 'target')
        target_col = None
        for col in ['diagnosis', 'target', 'Class', 'class']:
            if col in user_df.columns:
                target_col = col
                break
                
        if target_col is None:
            st.error("Error: Could not identify a target classification column (e.g., 'diagnosis') in the CSV.")
            st.stop()
            
        # Clean target variables to numeric indicators (0: Malignant, 1: Benign)
        # Kaggle uses 'M' and 'B'
        y_user = user_df[target_col].copy()
        if y_user.dtype == 'O': 
            y_user = y_user.map({'M': 0, 'B': 1, 'malignant': 0, 'benign': 1})
            
        # Extract features and ensure order matches training features
        X_user = user_df.drop(columns=[target_col])
        
        # Match scikit-learn standard feature mapping vs Kaggle header text structures
        # Mapping Kaggle column layouts to standard dataset fields
        rename_dict = {}
        for col in X_user.columns:
            cleaned_col = col.replace('_', ' ').strip().lower()
            for feat in feature_names:
                if cleaned_col == feat.lower().strip() or cleaned_col == feat.replace(' ', '').lower():
                    rename_dict[col] = feat
        X_user = X_user.rename(columns=rename_dict)
        
        # Handle structural missing column safety checks
        missing_cols = [c for c in feature_names if c not in X_user.columns]
        if missing_cols:
            st.warning(f"Note: Adjusting column naming syntax compatibility indices.")
            # Fallback alignment using exact column sequence if named maps fail
            if len(X_user.columns) == len(feature_names):
                X_user.columns = feature_names
            else:
                st.error(f"Dataset column footprint mismatch! Expected columns matching features. Missing: {missing_cols}")
                st.stop()
                
        # Reorder features explicitly
        X_user = X_user[feature_names]
        
        # Scale test data safely using baseline fitted scalars
        X_user_scaled = data_scaler.transform(X_user)
        
        # Select active model and generate inference arrays
        model = models_dict[selected_model_name]
        y_pred = model.predict(X_user_scaled)
        
        # Check if model supports probability vectors for precise AUC rendering
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_user_scaled)[:, 1]
        else:
            y_proba = y_pred

        # ----------------------------------------------------
        # 4. COMPUTE & DISPLAY EVALUATION METRICS
        # ----------------------------------------------------
        st.subheader(f"📈 Evaluation Metrics: {selected_model_name}")
        
        # Metric Calculations
        acc = accuracy_score(y_user, y_pred)
        precision = precision_score(y_user, y_pred, zero_division=0)
        recall = recall_score(y_user, y_pred, zero_division=0)
        f1 = f1_score(y_user, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_user, y_pred)
        
        try:
            auc = roc_auc_score(y_user, y_proba)
        except ValueError:
            auc = 0.0 # Fallback calculation alternative if single-class variant encountered
            
        # Metric Display Layout Grid
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric(label="Accuracy", value=f"{acc:.4f}")
        col2.metric(label="AUC Score", value=f"{auc:.4f}")
        col3.metric(label="Precision", value=f"{precision:.4f}")
        col4.metric(label="Recall", value=f"{recall:.4f}")
        col5.metric(label="F1 Score", value=f"{f1:.4f}")
        col6.metric(label="MCC Score", value=f"{mcc:.4f}")

        # ----------------------------------------------------
        # 5. CONFUSION MATRIX & CLASSIFICATION REPORT
        # ----------------------------------------------------
        st.markdown("---")
        layout_col1, layout_col2 = st.columns(2)
        
        with layout_col1:
            st.subheader("📋 Classification Report")
            rep_dict = classification_report(y_user, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
            st.dataframe(pd.DataFrame(rep_dict).transpose().style.format("{:.4f}"))
            
        with layout_col2:
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_user, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual Malignant", "Actual Benign"], columns=["Predicted Malignant", "Predicted Benign"])
            st.dataframe(cm_df)
            
            # Interactive visualization context check
            st.info("💡 Row labels describe True Classes; Column layouts display Predicted values.")
            
    except Exception as e:
        st.error(f"An unexpected data processing mismatch occurred: {e}")
        st.info("Tip: Use the sidebar button to generate an ideally-formatted sample test file.")

else:
    st.info("👈 Please upload a processed test data snippet (CSV) in the left sidebar to generate performance evaluations.")
