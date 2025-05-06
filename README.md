⸻

Polynomial, Logistic, and Ridge Regression on Wine Quality Dataset

This project explores three types of supervised learning techniques—Polynomial Regression, Logistic Regression, and Ridge Regression—using the Wine Quality (white) dataset. The dataset is sourced from the UCI Machine Learning Repository and contains physicochemical attributes of white wine samples.

⸻

Dataset
	•	File: winequality-white.csv
	•	Source: UCI Machine Learning Repository
	•	Format: CSV (semicolon ; delimited)
	•	Target Variables:
	•	alcohol (for regression tasks)
	•	quality (for classification tasks)

⸻

Question 1: Polynomial Regression

Goal:

Predict the alcohol content of wine using polynomial features of other numerical attributes.

Steps:
	1.	Preprocessing: Dropped alcohol and quality from the feature set.
	2.	Train/Test Split: 70% training, 30% testing.
	3.	Modeling: Fit a Linear Regression model with polynomial features ranging from degree 1 to 10.
	4.	Evaluation: Mean Squared Error (MSE) was computed for both training and test datasets.

Results:
	•	Training MSE: 0.198 (for degree 10)
	•	Test MSE: ~1,728,325 (for degree 10), indicating significant overfitting.
	•	Insight: Training MSE remains low, while test MSE increases sharply after degree 6, visualized through MSE plots.

⸻

Question 2: Logistic Regression

Binary Classification

Goal:

Classify wine as high-quality (quality ≥ 6) or low-quality (quality < 6).

Steps:
	1.	Target Transformation: Quality was binarized.
	2.	Train/Test Split: 70% training, 30% testing.
	3.	Modeling: Trained a LogisticRegression model (liblinear solver).
	4.	Evaluation: Confusion matrix, accuracy, precision, recall, and F1-score.

Results:
	•	Accuracy: ~73.9%
	•	Precision (Class 1): 75%
	•	Recall (Class 1): 90%
	•	Conclusion: The model performs well on identifying good quality wines (class 1), with some misclassification of lower-quality wines.

⸻

Multi-class Classification

Goal:

Classify wine into its exact quality level (3–9) using logistic regression.

Steps:
	1.	Used the original multiclass target (quality) without binarization.
	2.	Used LogisticRegression with multi_class='ovr'.

Results:
	•	Accuracy: ~53%
	•	Observation: High misclassification for rare quality levels (3, 4, 7–9). Best recall seen in class 6 (83%).

⸻

Question 3: Ridge Regression

Goal:

Use Ridge Regression to predict alcohol and observe how coefficients and performance vary with different regularization strengths.

Steps:
	1.	Preprocessing: Removed quality and alcohol from features.
	2.	Train/Test Split: 70% training, 30% testing.
	3.	Modeling: Ridge Regression with varying alpha from 1 to 10 (step 0.2).
	4.	Coefficient Visualization: Plotted how each feature’s coefficient changes with alpha.
	5.	Model Evaluation: Calculated and plotted MSE for test set across different alphas.

Results:
	•	Coefficient Analysis: Demonstrated regularization effect—shrinking coefficients as alpha increases.
	•	MSE Trends: Visual representation shows how regularization helps reduce overfitting at optimal alpha.

⸻

Visualizations
	•	Polynomial Regression: Training vs Test MSE across degrees.
	•	Ridge Regression:
	•	Coefficient paths ($\hat{w}^{ridge}$ vs alpha)
	•	MSE vs alpha

⸻

Requirements
	•	pandas
	•	numpy
	•	scikit-learn
	•	matplotlib

Install via pip:

pip install pandas numpy scikit-learn matplotlib



⸻

Summary
	•	Polynomial Regression can easily overfit with high-degree polynomials.
	•	Logistic Regression works reasonably well for binary classification but struggles with multiclass scenarios involving imbalanced classes.
	•	Ridge Regression effectively controls overfitting through regularization.

⸻

Would you like this as a downloadable README.md file?
