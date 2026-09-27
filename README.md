# Fraud Detection — Classification Model Evaluation

A practical Machine Learning project for detecting fraudulent transactions and evaluating a classification model using **Accuracy, Precision, Recall, F1 Score, and Confusion Matrix**.

## Project Overview

Fraud detection is a binary classification problem where the model predicts whether a transaction is:

* `0` → Normal transaction
* `1` → Fraudulent transaction

This project demonstrates a complete Machine Learning workflow, from dataset preparation to model evaluation.

## Machine Learning Workflow

```text
Dataset Creation
       ↓
Data Exploration
       ↓
Train / Test Split
       ↓
Feature Scaling
       ↓
Logistic Regression
       ↓
Predictions
       ↓
Confusion Matrix
       ↓
Model Evaluation
       ↓
Accuracy / Precision / Recall / F1 Score
```

## Dataset

The dataset is generated using `sklearn.datasets.make_classification()`.

The dataset contains:

* 5,000 transactions
* 8 numerical features
* Approximately 95% normal transactions
* Approximately 5% fraudulent transactions

The dataset is intentionally **imbalanced** to demonstrate why Accuracy alone may not be sufficient for fraud detection.

### Features

The project uses the following example transaction-related features:

```text
transaction_amount
transaction_frequency
account_age
avg_transaction
location_risk
device_risk
hour_risk
merchant_risk
```

> Note: The dataset is synthetically generated for educational and demonstration purposes. It does not represent real banking transaction data.

## Model

The project uses **Logistic Regression** as the classification algorithm.

```python
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)
```

The model is trained on the training set and then used to predict fraudulent transactions in the test set.

## Evaluation Metrics

### Accuracy

Accuracy measures the percentage of all predictions that are correct.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Precision

Precision measures how many transactions predicted as fraud were actually fraudulent.

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

Recall measures how many of the actual fraudulent transactions were detected by the model.

$$
Recall = \frac{TP}{TP + FN}
$$

### F1 Score

F1 Score combines Precision and Recall into a single metric.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

## Confusion Matrix

The confusion matrix shows the four possible prediction outcomes:

```text
                    Actual
                 Normal   Fraud
              ┌────────┬────────┐
Predicted     │        │        │
Normal        │   TN   │   FN   │
              ├────────┼────────┤
Fraud         │   FP   │   TP   │
              └────────┴────────┘
```

Where:

* **True Positive (TP):** Fraud correctly detected
* **True Negative (TN):** Normal transaction correctly identified
* **False Positive (FP):** Normal transaction incorrectly classified as fraud
* **False Negative (FN):** Fraudulent transaction missed by the model

## Why Recall Matters in Fraud Detection

Because fraud cases are much less common than normal transactions, a model can achieve high Accuracy while still missing many fraudulent transactions.

For example, a model could predict almost every transaction as normal and still achieve a high Accuracy on an imbalanced dataset.

Recall helps answer an important question:

> How many of the actual fraudulent transactions did the model detect?

This makes Recall an important metric to examine alongside Precision and F1 Score.

## Project Structure

```text
fraud-detection-ml/
│
├── fraud_detection.py
├── confusion_matrix.png
├── model_metrics.png
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/fraud-detection-ml.git
```

Move into the project directory:

```bash
cd fraud-detection-ml
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the Python script:

```bash
python fraud_detection.py
```

The program will:

1. Generate the dataset
2. Display the class distribution
3. Split the dataset into training and testing sets
4. Scale the features
5. Train a Logistic Regression model
6. Generate predictions
7. Calculate Accuracy, Precision, Recall, and F1 Score
8. Display the Confusion Matrix
9. Save the generated visualizations

## Output

The project generates:

```text
confusion_matrix.png
model_metrics.png
```

### Confusion Matrix

The confusion matrix provides a visual representation of the model's classification results.

### Model Metrics

The metrics chart compares:

* Accuracy
* Precision
* Recall
* F1 Score

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Learning Objectives

This project was created to practice:

* Binary Classification
* Supervised Machine Learning
* Train/Test Split
* Feature Scaling
* Logistic Regression
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* Imbalanced Classification

## Future Improvements

Possible improvements for this project include:

* Use a real Credit Card Fraud dataset
* Try other classification algorithms
* Handle class imbalance using techniques such as class weighting or resampling
* Tune the classification threshold
* Add ROC-AUC and Precision-Recall curves
* Compare multiple models
* Perform hyperparameter tuning

## Author

**Mohamed Shaban**

Computer Science Student | Aspiring AI Engineer

GitHub: https://github.com/mohamed-shaban-eng

LinkedIn: https://linkedin.com/in/eng-mohamed-shaban/
