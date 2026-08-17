Breast Cancer Wisconsin (Diagnostic) Classification Project
a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for diagnosing breast cancer tumors as either Malignant (M) or Benign (B) based on characteristics computed from digitized images of breast mass cell nuclei.

The performance of different classification algorithms is evaluated using standard metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC) to identify the most effective model for breast cancer diagnosis.

b. Dataset Description
Dataset Name

Breast Cancer Wisconsin (Diagnostic) Dataset

Dataset Source

Kaggle

Dataset Link

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Dataset Characteristics
Classification Type: Binary Classification
Number of Instances: 569
Number of Features: 30
Target Variable: Diagnosis
M = Malignant
B = Benign
Feature Information

The dataset contains numerical features describing cell nuclei characteristics extracted from breast mass images, including:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave Points
Symmetry
Fractal Dimension

For each characteristic, mean, standard error, and worst values are provided.

Objective

Predict whether a tumor is malignant or benign using machine learning classification techniques.

c. GitHub Repository Link

Repository Link:https://github.com/2025ad05102-sudo/breast-cancer-classification


d. Models Used and Evaluation Metrics

The following machine learning algorithms were implemented:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest Classifier (Ensemble Model)


Comparison Table


| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.824658 | 0.856769 | 0.711330  | 0.459580 | 0.558391 | 0.472151 |
| Decision Tree            | 0.812836 | 0.751406 | 0.607579  | 0.632718 | 0.619894 | 0.495994 |
| KNN                      | 0.835253 | 0.856804 | 0.678367  | 0.602801 | 0.638355 | 0.533721 |
| Naive Bayes              | 0.809304 | 0.861285 | 0.706399  | 0.358370 | 0.475507 | 0.406026 |
| Random Forest (Ensemble) | 0.862429 | 0.910280 | 0.750929  | 0.642903 | 0.692730 | 0.607817 |


Model Performance Observations

| ML Model Name            | Observation about Model Performance                                                                                                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression achieved an accuracy of 82.47% with a strong AUC of 85.68%. It produced good precision but comparatively lower recall, indicating that some positive cases were missed.                                  |
| Decision Tree            | Decision Tree provided balanced precision and recall values and performed reasonably well. However, its AUC score was the lowest among all models, suggesting weaker class discrimination capability.                        |
| KNN                      | KNN achieved better accuracy and MCC than Logistic Regression and Decision Tree. Its balanced precision, recall, and F1 score indicate stable performance on the dataset.                                                    |
| Naive Bayes              | Naive Bayes produced a good AUC score but had the lowest recall and F1 score. The independence assumption of the model may have limited its predictive performance.                                                          |
| Random Forest (Ensemble) | Random Forest achieved the highest Accuracy (86.24%), AUC (91.03%), Precision (75.09%), F1 Score (69.27%), and MCC (60.78%). It showed the best overall balance between correctly identifying classes and minimizing errors. |



Reason:

Highest Accuracy
High AUC Score
Strong Precision and Recall
Best F1 Score
Highest MCC Score
Better generalization compared to individual classifiers

Therefore, the Random Forest Classifier is selected as the most suitable model for the Breast Cancer Wisconsin (Diagnostic) dataset.

Conclusion

Five machine learning classification algorithms were implemented and evaluated on the Breast Cancer Wisconsin dataset using Accuracy, AUC, Precision, Recall, F1 Score, and MCC Score. Among the evaluated models, the Random Forest Classifier demonstrated the best overall predictive performance and is recommended for breast cancer diagnosis classification tasks.