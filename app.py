import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Evaluation Metrics Imports
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Set page configurations
st.set_page_config(page_title="BC LAB: Model Evaluation App", layout="wide")

st.title("🔬 Breast Cancer Wisconsin Diagnostic Evaluation Dashboard")
st.write("Fulfill your BITS Virtual Lab Assignment steps by evaluating pre-trained model pickle files.")

# ----------------------------------------------------
# 1. RESOLVE DIRECTORY PATHS AND LOAD MODEL PICKLES
# ----------------------------------------------------
@st.cache_resource
def load_github_pickle_models():
    # Detect if code runs locally or deep within Streamlit Cloud repo root
    # Adjust paths automatically if files live inside the 'model' subdirectory
    possible_paths = [
        "model", 
        os.path.join(os.path.dirname(__file__), "model"),
        "."
    ]
    
    target_dir = None
    for path in possible_paths:
        if os.path.isdir(path) and any(f.endswith('.pkl') for f in os.listdir(path)):
            target_dir = path
            break
            
    if target_dir is None:
        st.error("🚨 System Error: Could not locate the '.pkl' files directory in your repository layout.")
        st.info("Ensure your pickle files are placed in a folder named 'model' or directly alongside app.py.")
        st.stop()

    # Dictionary mapping display selection names to specific pickle filenames
    model_mapping = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree Classifier": "decision_tree.pkl",
        "K-Nearest Neighbor Classifier": "knn.pkl",
        "Naive Bayes Classifier - Gaussian or Multinomial": "gaussian_naive_bayes.pkl",
        "Ensemble Model - Random Forest": "random_forest.pkl"
    }
    
    loaded_models = {}
    for display_name, file_name in model_mapping.items():
        file_path = os.path.join(target_dir, file_name)
        
        # Soft fallback if names differ slightly (e.g. random_forest.pkl vs ensemble.pkl)
        if not os.path.exists(file_path):
            # Attempt to locate by fuzzy text matches inside directory files list
            all_files = os.listdir(target_dir)
            matched = [f for f in all_files if file_name.split('.')[0] in f or f.endswith('.pkl')]
            if matched:
                # Use alternative fallback pick found in target pathing arrays
                file_path = os.path.join(target_dir, matched[0])
            else:
                st.warning(f"⚠️ Missing Expected Model File: {file_name}")
                continue
                
        try:
            with open(file_path, 'rb') as f:
                loaded_models[display_name] = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to unpickle model file {file_name}: {e}")
            
    return loaded_models

# Run safe resource cache extraction
models_dict = load_github_pickle_models()

# Hardcoded feature feature array tracking canonical layout requirements for the Kaggle dataset
# Standard sequence utilized during model production
EXPECTED_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# ----------------------------------------------------
# 2. STREAMLIT UI: INPUT & CONFIGURATIONS
# ----------------------------------------------------
st.sidebar.header("📁 User Action Controls")

# Feature a: Dataset upload option (CSV) [Test Data Only]
uploaded_file = st.sidebar.file_uploader("Upload your test dataset (CSV Format)", type=["csv"])

# Feature b: Model selection dropdown
if models_dict:
    selected_model_name = st.sidebar.selectbox(
        "Choose ML Model for Evaluation", 
        list(models_dict.keys())
    )
else:
    st.error("No valid model files loaded.")
    st.stop()

# ----------------------------------------------------
# 3. PROCESSING PIPELINE & INFERENCE VIA PICKLES
# ----------------------------------------------------
if uploaded_file is not None:
    try:
        # Load user file
        user_df = pd.read_csv(uploaded_file)
        st.success("Test dataset loaded successfully!")
        
        # Clean metadata features out instantly if they appear inside Kaggle sheets
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
            
        # Standardize target values to binary classes (0: Malignant, 1: Benign)
        y_user = user_df[target_col].copy()
        if y_user.dtype == 'O': 
            y_user = y_user.map({'M': 0, 'B': 1, 'malignant': 0, 'benign': 1})
            
        # Isolate baseline classification inputs
        X_user = user_df.drop(columns=[target_col])
        
        # Enforce name formatting compatibility matches with training features schema
        # Normalizes spaces/underscores (e.g., 'radius_mean' vs 'radius mean')
        rename_dict = {}
        for col in X_user.columns:
            cleaned_col = str(col).replace('_', ' ').strip().lower()
            for feat in EXPECTED_FEATURES:
                clean_feat = str(feat).replace('_', ' ').strip().lower()
                if cleaned_col == clean_feat:
                    rename_dict[col] = feat
        X_user = X_user.rename(columns=rename_dict)
        
        # Safety Check: If names mismatch structural profiles, force override sequencing
        missing_cols = [c for c in EXPECTED_FEATURES if c not in X_user.columns]
        if missing_cols:
            if len(X_user.columns) == len(EXPECTED_FEATURES):
                X_user.columns = EXPECTED_FEATURES
            else:
                st.error(f"Column counts mismatch training space specifications. Missing: {missing_cols}")
                st.stop()
                
        # Lock column sequencing order
        X_user = X_user[EXPECTED_FEATURES]
        
        # Extract operational active model out from pickle store
        model = models_dict[selected_model_name]
        
        # Generate Predictions directly using the pickled model
        y_pred = model.predict(X_user)
        
        # Determine performance confidence scores if supported by object class
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_user)[:, 1]
        else:
            y_proba = y_pred

        # ----------------------------------------------------
        # 4. COMPUTE & DISPLAY EVALUATION METRICS
        # ----------------------------------------------------
        st.subheader(f"📈 Evaluation Metrics: {selected_model_name}")
        
        acc = accuracy_score(y_user, y_pred)
        precision = precision_score(y_user, y_pred, zero_division=0)
        recall = recall_score(y_user, y_pred, zero_division=0)
        f1 = f1_score(y_user, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_user, y_pred)
        
        try:
            auc = roc_auc_score(y_user, y_proba)
        except ValueError:
            auc = 0.0
            
        # Display Metric Matrix Configuration layout
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
            
            st.info("💡 Row labels describe True Classes; Column layouts display Predicted values.")
            
    except Exception as e:
        st.error(f"An error occurred during evaluation profiling: {e}")

else:
    st.info("👈 Please upload a processed verification dataset (CSV snippet) in the left sidebar area to evaluate loaded models.")

