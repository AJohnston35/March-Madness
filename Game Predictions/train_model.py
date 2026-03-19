# %% [markdown]
# # Train Model Workflow
# 
# This notebook mirrors `train_model.py`, organized into logical steps.

# %% [markdown]
# ## 1. Imports

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
import joblib
from data_processing import load_and_prepare_dataset

# %% [markdown]
# ## 2. Load And Prepare Dataset

# %%
# Load dataset
df = load_and_prepare_dataset(
    'dataset.csv',
    drop_cols=[
        'fast_break_pct_avg_diff',
        'fast_break_pct_rolling_5_diff',
        'points_off_turnover_pct_avg_diff',
        'points_off_turnover_pct_rolling_5_diff'
    ]
)

# %%

df.loc[df['team_home_away'] == 1, 'margin_estimate'] = df['margin_estimate'] + 7.00
df.loc[df['team_home_away'] == 0, 'margin_estimate'] = df['margin_estimate'] - 7.00
df.loc[df['team_home_away'] == 1, 'point_differential_avg_diff'] = df['point_differential_avg_diff'] + 6.90
df.loc[df['team_home_away'] == 0, 'point_differential_avg_diff'] = df['point_differential_avg_diff'] - 6.90

# %% [markdown]
# ## 3. Four-Way Split And Shared Config

# %%
base_train_end = '2023-10-31'
meta_train_end = '2024-10-31'
validation_end = '2025-10-31'


def games_played_weight(games_played, k=10):
    games_played = pd.to_numeric(games_played, errors='coerce').fillna(0)
    return np.minimum(1.0, games_played / float(k))


df['target'] = df['team_winner']
df.drop(columns=['team_winner'], inplace=True)

base_train_df = df[df['game_date'] <= base_train_end].copy()
meta_train_df = df[(df['game_date'] > base_train_end) & (df['game_date'] <= meta_train_end)].copy()
validation_df = df[(df['game_date'] > meta_train_end) & (df['game_date'] <= validation_end)].copy()
test_df = df[df['game_date'] > validation_end].copy()

test_game_dates = pd.to_datetime(test_df['game_date']).copy()

base_train_sample_weights = games_played_weight(base_train_df['games_played'])
meta_train_sample_weights = games_played_weight(meta_train_df['games_played'])
validation_sample_weights = games_played_weight(validation_df['games_played'])

base_train_df.drop(columns=['game_date','game_id'], inplace=True)
meta_train_df.drop(columns=['game_date','game_id'], inplace=True)
validation_df.drop(columns=['game_date','game_id'], inplace=True)
test_df.drop(columns=['game_date','game_id'], inplace=True)

# %%
print(f"Base train rows: {len(base_train_df)}")
print(f"Meta train rows: {len(meta_train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Test rows: {len(test_df)}")

# %% [markdown]
# ## 4. Train Winner Prediction Model

# %%
# Winner prediction model (classification)

X_base_train_winner = base_train_df.drop(columns=['target','spread'])
y_base_train_winner = base_train_df['target'].astype(int)

X_meta_train_winner = meta_train_df.drop(columns=['target','spread'])
y_meta_train_winner = meta_train_df['target'].astype(int)

X_validation_winner = validation_df.drop(columns=['target','spread'])
y_validation_winner = validation_df['target'].astype(int)

X_test_winner = test_df.drop(columns=['target','spread'])
y_test_winner = test_df['target'].astype(int)

# Improved model parameters for better generalization and handling nonlinearity
winner_params = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.04,           # Slower learning rate for more robust learning
    n_estimators=2000,             # More trees to compensate for lower learning rate
    max_depth=8,                   # Slightly deeper trees to fit more complex patterns
    num_leaves=2**8 - 1,                # More leaves to capture nonlinearity
    min_child_samples=50,          # Reduce overfitting, but not too high to retain learning small data regions
    feature_fraction=0.85,         # Try more features per tree
    bagging_fraction=0.85,         # Slightly more bagging for generalization
    bagging_freq=1,
    lambda_l1=1.0,                 # Slightly stronger L1 to encourage sparsity
    lambda_l2=6.0,                 # Slightly stronger L2 for regularization
    min_gain_to_split=0.005,       # Require a bit more information gain to split
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    metric="auc",
    early_stopping_rounds=200
)

model_winner = LGBMClassifier(**winner_params)

print("\nTraining winner prediction model (improved hyperparameters)...")
model_winner.fit(
    X_base_train_winner,
    y_base_train_winner,
    eval_set=[(X_base_train_winner, y_base_train_winner), (X_meta_train_winner, y_meta_train_winner)],
    eval_sample_weight=[base_train_sample_weights, meta_train_sample_weights],
    eval_names=['Base Train', 'Meta Train'],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True)]
)

# %% [markdown]
# ## 5. Save Models

# %%
'''joblib.dump(model_winner, 'models/lgbm_winner_model.joblib')'''

# %% [markdown]
# ## 6. Define Evaluation Helper

# %%
def evaluate_classification_thresholds(y_pred_proba, y_test, model_name):
    from sklearn.metrics import matthews_corrcoef

    thresholds = np.linspace(0.1, 0.8, 10)  # Test thresholds from 0.1 to 0.8
    metrics = []
    
    for thresh in thresholds:
        # Get predictions based on threshold
        y_pred_thresh = (y_pred_proba > thresh).astype(int)

        # Calculate standard metrics
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        accuracy = accuracy_score(y_test, y_pred_thresh)
        mcc = matthews_corrcoef(y_test, y_pred_thresh)

        metrics.append([thresh, precision, recall, f1, accuracy, mcc])

    # Convert to DataFrame
    metrics_df = pd.DataFrame(
        metrics,
        columns=[
            "Threshold", "Precision", "Recall", "F1 Score", "Accuracy", "Matthews_CC",
        ]
    )

    # Plot metrics
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x="Threshold", y="Precision", label="Precision", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Recall", label="Recall", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="F1 Score", label="F1 Score", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Accuracy", label="Accuracy", marker="o")
    sns.lineplot(
        data=metrics_df,
        x="Threshold",
        y="Matthews_CC",
        label="Matthews Corr. Coefficient",
        marker="o",
        linestyle="--",
        color="darkgreen"
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Optimization - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Find and return the best threshold based on Matthews Correlation Coefficient (MCC)
    best_threshold = metrics_df.loc[metrics_df["Matthews_CC"].idxmax(), "Threshold"]

    return best_threshold

def print_classification_metrics(y_true, y_pred_bin, y_pred_proba, model_name):
    """
    Print standard classification metrics for binary classification.
    """
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin)
    rec = recall_score(y_true, y_pred_bin)
    f1s = f1_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred_bin)
    print(f"\n===== {model_name} Metrics =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1s:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("="*36)


def print_regression_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n===== {model_name} Regression Metrics =====")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R^2:   {r2:.4f}")
    print("="*36)


def plot_monthly_roc_auc(y_true, y_pred_proba, game_dates, model_name):
    monthly_df = pd.DataFrame({
        'game_date': pd.to_datetime(game_dates),
        'y_true': np.asarray(y_true),
        'y_pred_proba': np.asarray(y_pred_proba),
    }).dropna(subset=['game_date'])

    monthly_df['month'] = monthly_df['game_date'].dt.to_period('M').dt.to_timestamp()

    monthly_rows = []
    skipped_months = []
    for month, group in monthly_df.groupby('month'):
        if group['y_true'].nunique() < 2:
            skipped_months.append(month.strftime('%Y-%m'))
            continue
        monthly_rows.append({
            'Month': month,
            'ROC AUC': roc_auc_score(group['y_true'], group['y_pred_proba']),
            'Games': len(group),
        })

    if not monthly_rows:
        print(f"No monthly ROC AUC plot generated for {model_name}; each month needs both classes present.")
        return pd.DataFrame()

    monthly_auc_df = pd.DataFrame(monthly_rows).sort_values('Month')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_auc_df, x='Month', y='ROC AUC', marker='o', linewidth=2)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Month")
    plt.ylabel("ROC AUC")
    plt.title(f"Monthly ROC AUC - {model_name}")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if skipped_months:
        print(f"Skipped months for {model_name} ROC AUC (single class only): {', '.join(skipped_months)}")

    return monthly_auc_df

# %% [markdown]
# ## 7. Evaluate Models

# %%
winner_meta_train_proba = model_winner.predict_proba(X_meta_train_winner)[:, 1]
winner_validation_proba = model_winner.predict_proba(X_validation_winner)[:, 1]
winner_test_proba = model_winner.predict_proba(X_test_winner)[:, 1]

print("\n== Winner Model Results ==")
winner_pred_bin = (winner_test_proba > 0.5).astype(int)
print_classification_metrics(y_test_winner, winner_pred_bin, winner_test_proba, "LGBM Winner Model")
plot_monthly_roc_auc(y_test_winner, winner_test_proba, test_game_dates, "LGBM Winner Model")

# %% [markdown]
# ## 8. Define Feature Importance Helper

# %%
def plot_feature_importance(model, X_train, model_name):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    feat_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    top_20_features = feat_importance_df.sort_values(by="Importance", ascending=False).head(30)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_20_features, palette="viridis")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title(f"Top 20 Features - {model_name}")
    plt.show()

# %% [markdown]
# ## 9. Plot Feature Importances

# %%
plot_feature_importance(model_winner, X_base_train_winner, "Winner Prediction Model")

# %%
# Spread prediction model (regression)

# Drop the other target column to avoid leakage
X_base_train_spread = base_train_df.drop(columns=['target','spread'])
y_base_train_spread = base_train_df['spread'].astype(float)

X_meta_train_spread = meta_train_df.drop(columns=['target','spread'])
y_meta_train_spread = meta_train_df['spread'].astype(float)

X_validation_spread = validation_df.drop(columns=['target','spread'])
y_validation_spread = validation_df['spread'].astype(float)

X_test_spread = test_df.drop(columns=['target','spread'])
y_test_spread = test_df['spread'].astype(float)

# Improved model parameters for better generalization and handling nonlinearity (regression)
spread_params = dict(
    objective="regression",
    metric="mae",
    boosting_type="gbdt",
    learning_rate=0.04,           # Slower learning rate for more robust learning
    n_estimators=2000,             # More trees to compensate for lower learning rate
    max_depth=7,                   # Slightly deeper trees to fit more complex patterns
    num_leaves=96,                # More leaves to capture nonlinearity
    min_child_samples=50,          # Reduce overfitting, but not too high to retain learning small data regions
    feature_fraction=0.85,         # Try more features per tree
    bagging_fraction=0.85,         # Slightly more bagging for generalization
    bagging_freq=1,
    lambda_l1=1.0,                 # Slightly stronger L1 to encourage sparsity
    lambda_l2=6.0,                 # Slightly stronger L2 for regularization
    min_gain_to_split=0.005,       # Require a bit more information gain to split
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    early_stopping_rounds=200
    # Remove early_stopping_rounds, eval_set, eval_names, eval_metric, callbacks from constructor
)

model_spread = LGBMRegressor(**spread_params)

print("\nTraining spread prediction model (improved hyperparameters)...")
model_spread.fit(
    X_base_train_spread, y_base_train_spread,
    eval_set=[(X_base_train_spread, y_base_train_spread), (X_meta_train_spread, y_meta_train_spread)],
    eval_sample_weight=[base_train_sample_weights, meta_train_sample_weights],
    eval_names=['Base Train', 'Meta Train'],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True)]
)

# %%

def evaluate_regression_thresholds(y_pred_raw, y_test, model_name):
    """
    Evaluate thresholds for the regression task, mapping predicted spread > threshold to win (1), else loss (0).
    Additionally, always compute the hard classification: pred > 0 => winner.
    Prints ROC AUC using the raw spread as a score for the winner class (y_test should be binary).
    """
    from sklearn.metrics import matthews_corrcoef, roc_auc_score

    thresholds = np.linspace(-10, 10, 21)  # Try thresholds from -10 to 10
    metrics = []

    # Compute ROC AUC for regression output as probabilities (using spread as score for winner class = 1)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_raw)
        print(f"ROC AUC (raw regression scores): {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC AUC computation failed: {str(e)}")

    for thresh in thresholds:
        # Mask spread prediction: classify as win (1) if pred > thresh, else loss (0)
        y_pred_thresh = (y_pred_raw > thresh).astype(int)

        # Calculate metrics for this threshold
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        accuracy = accuracy_score(y_test, y_pred_thresh)
        mcc = matthews_corrcoef(y_test, y_pred_thresh)

        metrics.append([thresh, precision, recall, f1, accuracy, mcc])

    # Also evaluate the standard "pred > 0" as a classification prediction
    y_pred_binary = (y_pred_raw > 0).astype(int)
    binary_metrics = {
        "Precision": precision_score(y_test, y_pred_binary),
        "Recall": recall_score(y_test, y_pred_binary),
        "F1 Score": f1_score(y_test, y_pred_binary),
        "Accuracy": accuracy_score(y_test, y_pred_binary),
        "Matthews_CC": matthews_corrcoef(y_test, y_pred_binary)
    }
    print("\nMetrics for hard mask (prediction > 0 ==> Winner):")
    print(f"Precision: {binary_metrics['Precision']:.4f}")
    print(f"Recall:    {binary_metrics['Recall']:.4f}")
    print(f"F1 Score:  {binary_metrics['F1 Score']:.4f}")
    print(f"Accuracy:  {binary_metrics['Accuracy']:.4f}")
    print(f"MCC:       {binary_metrics['Matthews_CC']:.4f}\n")

    # Convert to DataFrame
    metrics_df = pd.DataFrame(
        metrics,
        columns=[
            "Threshold", "Precision", "Recall", "F1 Score", "Accuracy", "Matthews_CC",
        ]
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x="Threshold", y="Precision", label="Precision", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Recall", label="Recall", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="F1 Score", label="F1 Score", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Accuracy", label="Accuracy", marker="o")
    sns.lineplot(
        data=metrics_df,
        x="Threshold",
        y="Matthews_CC",
        label="Matthews Corr. Coefficient",
        marker="o",
        linestyle="--",
        color="darkgreen"
    )

    plt.xlabel("Spread Threshold (win if pred > thresh)")
    plt.ylabel("Score")
    plt.title(f"Threshold Optimization for Regression-as-Classification - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Best metrics over thresholds (MCC):")
    print(metrics_df.loc[metrics_df['Matthews_CC'].idxmax()])

    # Find and return the best threshold based on Matthews Correlation Coefficient (MCC)
    best_threshold = metrics_df.loc[metrics_df["Matthews_CC"].idxmax(), "Threshold"]

    return best_threshold

def print_regression_classification_metrics(y_true, y_pred_raw, model_name, threshold=0.0):
    """
    Print accuracy metrics for regression models treated as binary classifiers (pred > threshold).
    """
    y_pred_class = (y_pred_raw > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class)
    rec = recall_score(y_true, y_pred_class)
    f1s = f1_score(y_true, y_pred_class)
    auc = roc_auc_score(y_true, y_pred_raw)
    cm = confusion_matrix(y_true, y_pred_class)
    print(f"\n===== {model_name} | Threshold={threshold:.2f} =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1s:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("="*36)

# %%
spread_meta_train_pred = model_spread.predict(X_meta_train_spread)
spread_validation_pred = model_spread.predict(X_validation_spread)
spread_test_pred = model_spread.predict(X_test_spread)

print("\n== Spread Model Results ==")
print_regression_metrics(y_test_spread, spread_test_pred, "LGBM Spread Model")
print_regression_classification_metrics(y_test_winner, spread_test_pred, "LGBM Spread Model")

# %% [markdown]
# ## 10. Train Meta Model From Winner + Spread Predictions

# %%
meta_train = pd.DataFrame(
    {
        'winner_model_proba': winner_meta_train_proba,
        'spread_model_pred': spread_meta_train_pred,
    }
)
meta_validation = pd.DataFrame(
    {
        'winner_model_proba': winner_validation_proba,
        'spread_model_pred': spread_validation_pred,
    }
)
meta_test = pd.DataFrame(
    {
        'winner_model_proba': winner_test_proba,
        'spread_model_pred': spread_test_pred,
    }
)

meta_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42),
)

meta_model.fit(
    meta_train,
    y_meta_train_winner
)

meta_validation_proba = meta_model.predict_proba(meta_validation)[:, 1]
meta_test_proba = meta_model.predict_proba(meta_test)[:, 1]
meta_test_pred_bin = (meta_test_proba > 0.5).astype(int)
meta_validation_pred_bin = (meta_validation_proba > 0.5).astype(int)

print("\n== Meta Model Results ==")
print_classification_metrics(
    y_validation_winner,
    meta_validation_pred_bin,
    meta_validation_proba,
    "Winner/Spread Meta Model Validation",
)
print_classification_metrics(y_test_winner, meta_test_pred_bin, meta_test_proba, "Winner/Spread Meta Model")
plot_monthly_roc_auc(y_test_winner, meta_test_proba, test_game_dates, "Winner/Spread Meta Model")

meta_coefficients = pd.Series(
    meta_model.named_steps['logisticregression'].coef_[0],
    index=meta_train.columns,
).sort_values(ascending=False)
print("\nMeta model coefficients:")
print(meta_coefficients)

# %% [markdown]
# ## 11. Plot Feature Importances (Spread)

# %%
plot_feature_importance(model_spread, X_base_train_spread, "Spread Prediction Model")

# %%
# Save trained models to disk
import os

os.makedirs('models', exist_ok=True)

joblib.dump(model_winner, 'models/lgbm_winner_model.joblib')
joblib.dump(model_spread, 'models/lgbm_spread_model.joblib')
joblib.dump(meta_model, 'models/meta_model.joblib')

print("Models saved to the 'models' directory:")
print("- models/lgbm_winner_model.joblib")
print("- models/lgbm_spread_model.joblib")
print("- models/meta_model.joblib")