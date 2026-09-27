# ============================================
# Fraud Detection - Classification Evaluation
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# --------------------------------------------
# 1. Create a realistic imbalanced dataset
# --------------------------------------------

X, y = make_classification(
    n_samples=5000,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.95, 0.05],  # 95% normal, 5% fraud
    random_state=42,
)

# Convert to DataFrame
feature_names = [
    "transaction_amount",
    "transaction_frequency",
    "account_age",
    "avg_transaction",
    "location_risk",
    "device_risk",
    "hour_risk",
    "merchant_risk",
]

df = pd.DataFrame(X, columns=feature_names)
df["fraud"] = y

print("=" * 60)
print("FRAUD DETECTION PROJECT")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["fraud"].value_counts())

print("\nClass distribution (%):")
print(df["fraud"].value_counts(normalize=True) * 100)


# --------------------------------------------
# 2. Split features and target
# --------------------------------------------

X = df.drop("fraud", axis=1)
y = df["fraud"]


# --------------------------------------------
# 3. Train / Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------
# 4. Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------
# 5. Build the Machine Learning Model
# --------------------------------------------

model = LogisticRegression(random_state=42, max_iter=1000)

model.fit(X_train_scaled, y_train)


# --------------------------------------------
# 6. Make Predictions
# --------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------
# 7. Calculate Evaluation Metrics
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# --------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------
# 9. Detailed Classification Report
# --------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))


# --------------------------------------------
# 10. Visualize Confusion Matrix
# --------------------------------------------

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])

disp.plot()

plt.title("Fraud Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)

plt.show()


# --------------------------------------------
# 11. Visualize Model Metrics
# --------------------------------------------

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

plt.figure(figsize=(9, 5))

plt.bar(metrics.keys(), metrics.values())

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Classification Model Evaluation")

for i, value in enumerate(metrics.values()):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center")

plt.tight_layout()

plt.savefig("model_metrics.png", dpi=150)

plt.show()


# --------------------------------------------
# 12. Interpretation
# --------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(f"""
Accuracy : {accuracy:.2%}
Precision: {precision:.2%}
Recall   : {recall:.2%}
F1 Score : {f1:.2%}

Recall answers:
"Out of all actual fraud transactions,
how many did the model detect?"

Precision answers:
"Out of all transactions predicted as fraud,
how many were actually fraud?"

Accuracy answers:
"Out of all transactions,
how many predictions were correct?"

F1 Score combines Precision and Recall
into one metric.
""")
