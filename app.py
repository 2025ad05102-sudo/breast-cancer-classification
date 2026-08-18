

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page configuration layout
st.set_page_config(page_title="BC Wisconsin Diagnostics", layout="wide")
st.title("🩺 Breast Cancer Wisconsin Diagnostic Platform")
st.write("---")

# 1. Dataset Upload Option (Sidebar Widget)
st.sidebar.header("📁 Upload Segment")
uploaded_file = st.sidebar.file_uploader("Upload your test_data.csv file", type="csv")

if uploaded_file is not None:
    # Read the dataset csv
    raw_df = pd.read_csv(uploaded_file)
    
    st.success("✅ Dataset uploaded successfully!")
    st.write("### Test Dataset Preview (Top 5 rows)", raw_df.head(5))
    
    # Standardize data cleanup pipeline 
    df = raw_df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
        
    # Check for target presence or valid labeling configurations
    if 'diagnosis' in df.columns:
        # Standardize target column mapping: 'M' or 1 -> 1, 'B' or 0 -> 0
        if df['diagnosis'].dtype == object:
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            
        X = df.drop(columns=['diagnosis'])
        y = df['diagnosis']
        
        # Consistent evaluation splitting structure
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scaling feature distributions uniformly
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. Model Selection Dropdown
        st.sidebar.write("---")
        st.sidebar.header("🤖 Model Selection")
        model_choice = st.sidebar.selectbox(
            "Choose Classification Model to Evaluate",
            ["Logistic Regression", "Decision Tree", "kNN Classifier", "Naive Bayes", "Random Forest"]
        )
        
        # Model dictionary tracking
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "kNN Classifier": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        
        # Fit models interactively and score
        clf = models[model_choice]
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred
        
        # 3. Display Evaluation Metrics Dashboard
        st.write(f"## 📊 Evaluation Metrics Performance for: **{model_choice}**")
        
        m_accuracy = accuracy_score(y_test, y_pred)
        m_auc = roc_auc_score(y_test, y_proba)
        m_precision = precision_score(y_test, y_pred)
        m_recall = recall_score(y_test, y_pred)
        m_f1 = f1_score(y_test, y_pred)
        m_mcc = matthews_corrcoef(y_test, y_pred)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric(label="Accuracy", value=f"{m_accuracy:.3f}")
        col2.metric(label="AUC Score", value=f"{m_auc:.3f}")
        col3.metric(label="Precision", value=f"{m_precision:.3f}")
        col4.metric(label="Recall", value=f"{m_recall:.3f}")
        col5.metric(label="F1 Score", value=f"{m_f1:.3f}")
        col6.metric(label="MCC Score", value=f"{m_mcc:.3f}")
        
        # 4. Confusion Matrix and Classification Report Area
        st.write("---")
        st.write("### 🔍 Model Diagnostic Outputs")
        
        block1, block2 = st.columns([1, 1])
        
        with block1:
            st.write("#### Confusion Matrix Graphic")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign (0)', 'Malignant (1)'], 
                yticklabels=['Benign (0)', 'Malignant (1)'], 
                ax=ax, cbar=False
            )
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            
        with block2:
            st.write("#### Detailed Classification Report Text")
            st.code(classification_report(y_test, y_pred), language='text')
            
    else:
        st.error("❌ Target Error: The uploaded CSV data is missing the required 'diagnosis' classification target row.")
else:
    # Landing state guide indicator
    st.info("💡 Getting Started: Please drag and drop or upload your 'test_data.csv' file via the left sidebar panel to compute execution diagnostics.")
