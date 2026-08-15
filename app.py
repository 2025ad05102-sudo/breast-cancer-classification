import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# Load test data
df = pd.read_csv("test_data.csv")

X_test = df.drop("diagnosis", axis=1)
y_test = df["diagnosis"]

models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

for name, path in models.items():

    model = joblib.load(path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("\n", "="*60)
    print(name)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("AUC      :", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("MCC      :", matthews_corrcoef(y_test, y_pred))