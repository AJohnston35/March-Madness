import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load tournament data
df = pd.read_csv('cleaned_dataset.csv')

X = df.drop(columns=['Unnamed: 0', 'game_id', 'game_date', 'home_team', 'home_color','away_team','away_color','winner'])
print(X.columns)
y = df['winner'].astype(bool).astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weight_0, weight_1 = class_weights
print(weight_0, weight_1)

# Calculate scale_pos_weight for LightGBM
scale_pos_weight = weight_0 / weight_1

# Initialize the model and train it
model = LGBMClassifier(
    objective='binary',
    metric='f1',
    boosting_type='gbdt',
    verbose=-1,
    learning_rate=0.05,
    max_depth=7,
    min_child_samples=20,
    n_estimators=300,
    num_leaves=63
)
model.fit(X_train, y_train)

# Evaluate the model with best hyperparameters
y_pred_proba = model.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt
import seaborn as sns

# Convert predicted probabilities to binary labels (using an optimal threshold, e.g., 0.5)
optimal_threshold = 0.75

y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Loss (0)", "Win (1)"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# Test with different thresholds
thresholds = np.arange(0.25, 0.76, 0.05)
for threshold in thresholds:
    y_pred = (y_pred_proba > threshold).astype(int)

    # Display performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}\n")

# Plot ROC and Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'PR curve (Avg Precision = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Feature importance
importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.to_csv('feature_importance.txt')

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features')
plt.show()

joblib.dump(model, 'lgbm_model.joblib')