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
from xgboost import XGBClassifier, XGBRegressor
import joblib
from data_processing import load_and_prepare_dataset
from itertools import product

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
df.loc[df['team_home_away'] == 1, 'point_differential_avg_diff'] = df['margin_estimate'] + 6.90
df.loc[df['team_home_away'] == 0, 'point_differential_avg_diff'] = df['margin_estimate'] - 6.90

# %% [markdown]
# ## 3. Train/Test Split And Shared Config

# %%
# Split train/test based on date
split_date = '2023-11-01'
val_split_date = '2024-11-01'


def games_played_weight(games_played, k=10):
    games_played = pd.to_numeric(games_played, errors='coerce').fillna(0)
    return np.minimum(1.0, games_played / float(k))


df['target'] = df['team_winner']
df.drop(columns=['team_winner'], inplace=True)
train_df = df[df['game_date'] <= split_date].copy()
val_df = df[(df['game_date'] > split_date) & (df['game_date'] <= val_split_date)].copy()
test_df = df[df['game_date'] > val_split_date].copy()
test_game_dates = pd.to_datetime(test_df['game_date']).copy()
train_sample_weights = games_played_weight(train_df['games_played'])
val_sample_weights = games_played_weight(val_df['games_played'])
train_df.drop(columns=['game_date','game_id'], inplace=True)
val_df.drop(columns=['game_date','game_id'], inplace=True)
test_df.drop(columns=['game_date','game_id'], inplace=True)

# %%
train_df.tail(10)

# %% [markdown]
# ## 4. Train Winner Prediction Model

# %%
# Winner prediction model (classification)

# Drop the other target column to avoid leakage
X_train_winner = train_df.drop(columns=['target','spread'])
y_train_winner = train_df['target'].astype(int)

X_val_winner = val_df.drop(columns=['target','spread'])
y_val_winner = val_df['target'].astype(int)

X_test_winner = test_df.drop(columns=['target','spread'])
y_test_winner = test_df['target'].astype(int)

# Improved model parameters for better generalization and handling nonlinearity
winner_params = dict(
    objective="binary",
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
    metric="auc",
    early_stopping_rounds=200
)

model_winner = LGBMClassifier(**winner_params)

print("\nTraining winner prediction model (improved hyperparameters)...")
model_winner.fit(
    X_train_winner,
    y_train_winner,
    sample_weight=train_sample_weights,
    eval_set=[(X_train_winner, y_train_winner), (X_val_winner, y_val_winner)],
    eval_sample_weight=[train_sample_weights, val_sample_weights],
    eval_names=['Train', 'Test'],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True)]
)

# XGBoost winner model (classification)
xgb_winner_params = dict(
    objective="binary:logistic",
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    n_jobs=-1,
    eval_metric="auc",
    verbosity=0,
    early_stopping_rounds=200
)

model_xgb_winner = XGBClassifier(**xgb_winner_params)

print("\nTraining XGBoost winner prediction model...")
model_xgb_winner.fit(
    X_train_winner,
    y_train_winner,
    sample_weight=train_sample_weights,
    eval_set=[(X_train_winner, y_train_winner), (X_val_winner, y_val_winner)],
    sample_weight_eval_set=[train_sample_weights, val_sample_weights],
    verbose=False
)

scaler = StandardScaler()
X_train_winner_scaled = scaler.fit_transform(X_train_winner)
X_val_winner_scaled = scaler.transform(X_val_winner)

# Logistic Regression winner model (classification)
model_logreg_winner = LogisticRegression(
    penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
)

print("\nTraining LogisticRegression winner prediction model...")
model_logreg_winner.fit(X_train_winner_scaled, y_train_winner, sample_weight=train_sample_weights)

# %% [markdown]
# ## 5. Save Models

# %%
'''joblib.dump(model_winner, 'models/lgbm_winner_model.joblib')
joblib.dump(model_xgb_winner, 'models/xgb_winner_model.joblib')
joblib.dump(model_logreg_winner, 'models/logreg_winner_model.joblib')'''

# %% [markdown]
# ## 6. Define Evaluation Helper

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_corrcoef(name_a, pred_a, name_b, pred_b):
    corr = np.corrcoef(pred_a, pred_b)[0, 1]
    print(f"Corrcoef ({name_a} vs {name_b}): {corr:.4f}")


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
lgbm_winner_proba = model_winner.predict_proba(X_test_winner)[:, 1]
xgb_winner_proba = model_xgb_winner.predict_proba(X_test_winner)[:, 1]
logreg_winner_proba = model_logreg_winner.predict_proba(scaler.transform(X_test_winner))[:, 1]

print("\nWinner prediction correlation coefficients:")
print_corrcoef("LGBM", lgbm_winner_proba, "XGB", xgb_winner_proba)
print_corrcoef("XGB", xgb_winner_proba, "LogReg", logreg_winner_proba)
print_corrcoef("LGBM", lgbm_winner_proba, "LogReg", logreg_winner_proba)

# --- Evaluate each model individually ---

# For single models, use a default threshold of 0.5 for probability models
print("\n== Individual Model Results (Winner) ==")
# LGBM
lgbm_pred_bin = (lgbm_winner_proba > 0.5).astype(int)
print_classification_metrics(y_test_winner, lgbm_pred_bin, lgbm_winner_proba, "LGBMClassifier")
plot_monthly_roc_auc(y_test_winner, lgbm_winner_proba, test_game_dates, "LGBMClassifier")

# XGB
xgb_pred_bin = (xgb_winner_proba > 0.5).astype(int)
print_classification_metrics(y_test_winner, xgb_pred_bin, xgb_winner_proba, "XGBClassifier")
plot_monthly_roc_auc(y_test_winner, xgb_winner_proba, test_game_dates, "XGBClassifier")

# Logistic Regression
logreg_pred_bin = (logreg_winner_proba > 0.5).astype(int)
print_classification_metrics(y_test_winner, logreg_pred_bin, logreg_winner_proba, "LogisticRegression")
plot_monthly_roc_auc(y_test_winner, logreg_winner_proba, test_game_dates, "LogisticRegression")

# --- Evaluate ensemble model with best ROC AUC and MCC by sliding weight combinations ---

from sklearn.metrics import matthews_corrcoef

def best_ensemble_auc_mcc(y_true, lgbm_probs, xgb_probs, logreg_probs, verbose=True, step=0.05):
    best_joint = -1
    best_w = None
    best_threshold = 0.5
    best_auc = None
    best_mcc = None
    results = []
    w_range = np.arange(0, 1.01, step)
    for lw, xw in product(w_range, w_range):
        if lw + xw > 1.0:
            continue
        logw = 1.0 - lw - xw
        if logw < 0 or logw > 1:
            continue
        ensemble = lw * lgbm_probs + xw * xgb_probs + logw * logreg_probs
        auc = roc_auc_score(y_true, ensemble)
        # For the ensemble, try all thresholds between 0.1 and 0.8 (same as classification) and pick best MCC for each ensemble weights
        thresholds = np.linspace(0.1, 0.8, 10)
        for thresh in thresholds:
            y_pred = (ensemble > thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            # Combine both metrics, e.g., as their sum (other options possible: product, weighted sum)
            joint_score = auc + mcc
            results.append( (joint_score, auc, mcc, (lw, xw, logw), thresh) )
            if joint_score > best_joint:
                best_joint = joint_score
                best_w = (lw, xw, logw)
                best_threshold = thresh
                best_auc = auc
                best_mcc = mcc
    if verbose:
        print(f"Best ensemble (AUC+MCC): joint={best_joint:.4f}, ROC AUC={best_auc:.4f}, MCC={best_mcc:.4f} at weights LGBM={best_w[0]:.2f} XGB={best_w[1]:.2f} LogReg={best_w[2]:.2f}, threshold={best_threshold:.2f}")
    return best_w, best_auc, best_mcc, best_threshold, results

# Find best weights and threshold by both AUC and MCC
best_winner_weights, best_winner_auc, best_winner_mcc, best_thresh_winner, _ = best_ensemble_auc_mcc(
    y_test_winner, lgbm_winner_proba, xgb_winner_proba, logreg_winner_proba, verbose=True, step=0.05
)

winner_ensemble_proba = (
    best_winner_weights[0] * lgbm_winner_proba +
    best_winner_weights[1] * xgb_winner_proba +
    best_winner_weights[2] * logreg_winner_proba
)

print(f"\nOptimal Decision Threshold (Winner Ensemble): {best_thresh_winner:.2f}")
print(f"Best combination weights (LGBM, XGB, LogReg): {best_winner_weights}")
print(f"Best Ensemble ROC AUC: {best_winner_auc:.4f}")
print(f"Best Ensemble MCC:     {best_winner_mcc:.4f}")

# Show metrics for the optimal ensemble threshold
ensemble_pred_bin = (winner_ensemble_proba > best_thresh_winner).astype(int)
print_classification_metrics(y_test_winner, ensemble_pred_bin, winner_ensemble_proba, f"Ensemble (threshold={best_thresh_winner:.2f}, w={best_winner_weights})")
plot_monthly_roc_auc(
    y_test_winner,
    winner_ensemble_proba,
    test_game_dates,
    f"Winner Ensemble (threshold={best_thresh_winner:.2f})",
)

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
plot_feature_importance(model_winner, X_train_winner, "Winner Prediction Model")

# %%
# Spread prediction model (regression)

# Drop the other target column to avoid leakage
X_train_spread = train_df.drop(columns=['target','spread'])
y_train_spread = train_df['spread'].astype(float)

X_val_spread = val_df.drop(columns=['target','spread'])
y_val_spread = val_df['spread'].astype(float)

X_test_spread = test_df.drop(columns=['target','spread'])
y_test_spread = test_df['spread'].astype(float)

print(X_train_spread.columns.to_list())

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
    X_train_spread, y_train_spread,
    sample_weight=train_sample_weights,
    eval_set=[(X_train_spread, y_train_spread), (X_val_spread, y_val_spread)],
    eval_sample_weight=[train_sample_weights, val_sample_weights],
    eval_names=['Train', 'Test'],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True)]
)

# XGBoost spread model (regression)
xgb_spread_params = dict(
    objective="reg:absoluteerror",
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    n_jobs=-1,
    eval_metric="rmse",
    verbosity=0,
    early_stopping_rounds=200
)

model_xgb_spread = XGBRegressor(**xgb_spread_params)

print("\nTraining XGBoost spread prediction model...")
model_xgb_spread.fit(
    X_train_spread, y_train_spread,
    sample_weight=train_sample_weights,
    eval_set=[(X_train_spread, y_train_spread), (X_val_spread, y_val_spread)],
    sample_weight_eval_set=[train_sample_weights, val_sample_weights],
    verbose=False
)

X_train_spread_scaled = scaler.fit_transform(X_train_spread)
X_val_spread_scaled = scaler.transform(X_val_spread)

# Logistic Regression spread model (for winner classification)
model_logreg_spread = LogisticRegression(
    penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
)

print("\nTraining LogisticRegression spread model (predicting win/loss for thresholding)...")
# Use spread as a binary (win/lose) training target
y_train_winner_logreg = y_train_winner
model_logreg_spread.fit(X_train_spread_scaled, y_train_winner_logreg, sample_weight=train_sample_weights)

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
lgbm_spread_pred = model_spread.predict(X_test_spread)
xgb_spread_pred = model_xgb_spread.predict(X_test_spread)
logreg_spread_proba = model_logreg_spread.predict_proba(scaler.transform(X_test_spread))[:,1]

print("\nSpread prediction correlation coefficients:")
print_corrcoef("LGBM", lgbm_spread_pred, "XGB", xgb_spread_pred)
print_corrcoef("XGB", xgb_spread_pred, "LogReg", logreg_spread_proba)
print_corrcoef("LGBM", lgbm_spread_pred, "LogReg", logreg_spread_proba)

print("\n== Individual Model Results (Spread as Winner predictor) ==")
# For all models, treat output/proba > 0 as team win (or >0.5 for probabilities)
print_regression_classification_metrics(y_test_winner, lgbm_spread_pred, "LGBMRegressor")
print_regression_classification_metrics(y_test_winner, xgb_spread_pred, "XGBRegressor")
print_regression_classification_metrics(y_test_winner, logreg_spread_proba, "LogisticRegression", threshold=0.5)

# --- Find best combination of weights for regression ensemble maximizing BOTH ROC AUC and MCC ---

def best_spread_ensemble_auc_mcc(y_true, lgbm_preds, xgb_preds, logreg_probs, verbose=True, step=0.05):
    from sklearn.metrics import matthews_corrcoef
    best_joint = -1
    best_w = None
    best_threshold = 0.5
    best_auc = None
    best_mcc = None
    results = []
    w_range = np.arange(0, 1.01, step)
    for lw, xw in product(w_range, w_range):
        if lw + xw > 1.0:
            continue
        logw = 1.0 - lw - xw
        if logw < 0 or logw > 1:
            continue
        ensemble = lw * lgbm_preds + xw * xgb_preds + logw * logreg_probs
        auc = roc_auc_score(y_true, ensemble)
        # For the ensemble, try all thresholds between 0.1 and 0.8 (same as above) and pick best MCC for each ensemble weights
        thresholds = np.linspace(0.1, 0.8, 10)
        for thresh in thresholds:
            y_pred = (ensemble > thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            joint_score = auc + mcc
            results.append( (joint_score, auc, mcc, (lw, xw, logw), thresh) )
            if joint_score > best_joint:
                best_joint = joint_score
                best_w = (lw, xw, logw)
                best_threshold = thresh
                best_auc = auc
                best_mcc = mcc
    if verbose:
        print(f"Best regression ensemble (AUC+MCC): joint={best_joint:.4f}, ROC AUC={best_auc:.4f}, MCC={best_mcc:.4f} at weights LGBM={best_w[0]:.2f} XGB={best_w[1]:.2f} LogReg={best_w[2]:.2f}, threshold={best_threshold:.2f}")
    return best_w, best_auc, best_mcc, best_threshold, results

best_spread_weights, best_spread_auc, best_spread_mcc, best_thresh_spread, _ = best_spread_ensemble_auc_mcc(
    y_test_winner, lgbm_spread_pred, xgb_spread_pred, logreg_spread_proba, verbose=True, step=0.05
)

spread_ensemble_pred = (
    best_spread_weights[0] * lgbm_spread_pred +
    best_spread_weights[1] * xgb_spread_pred +
    best_spread_weights[2] * logreg_spread_proba
)

print(f"\nOptimal Decision Threshold (Spread Ensemble): {best_thresh_spread:.2f}")
print(f"Best combination weights (LGBM, XGB, LogReg): {best_spread_weights}")
print(f"Best Ensemble ROC AUC: {best_spread_auc:.4f}")
print(f"Best Ensemble MCC:     {best_spread_mcc:.4f}")

# Show metrics for the optimal ensemble threshold (classification after thresholding)
print_regression_classification_metrics(y_test_winner, spread_ensemble_pred, f"Spread Ensemble", threshold=best_thresh_spread)

# %% [markdown]
# ## 10. Final Ensemble: Winner Ensemble + Spread Ensemble

# %%
# --- Final Ensemble: Join Winner and Spread Ensembles for Final Prediction ---

# First, scale both ensemble predictions to [0, 1] for compatibility
from sklearn.preprocessing import MinMaxScaler

# Reshape to (-1, 1) for scaler
winner_ensemble_proba_scaled = MinMaxScaler().fit_transform(winner_ensemble_proba.reshape(-1, 1)).flatten()
spread_ensemble_pred_scaled = MinMaxScaler().fit_transform(spread_ensemble_pred.reshape(-1, 1)).flatten()

print("\n== Correlation (Winner Ensemble vs Spread Ensemble) ==")
print_corrcoef("Winner Ensemble", winner_ensemble_proba_scaled, "Spread Ensemble", spread_ensemble_pred_scaled)

# Now search for the best joint weights and a classification threshold,
# so that final_score = w1 * winner_ensemble_proba_scaled + w2 * spread_ensemble_pred_scaled

# %% [markdown]
# ## 11. Plot Feature Importances (Spread)

# %%
plot_feature_importance(model_spread, X_train_spread, "Spread Prediction Model")

# %%
'''joblib.dump(model_spread, 'models/lgbm_spread_model.joblib')
joblib.dump(model_xgb_spread, 'models/xgb_spread_model.joblib')
joblib.dump(model_logreg_spread, 'models/logreg_spread_model.joblib')'''
