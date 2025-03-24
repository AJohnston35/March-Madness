import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier
import joblib

# Load dataset
df = pd.read_csv('Game Predictions/cleaned_dataset.csv')
df = df.sort_values(by='game_date')  # Sort by date

player_cols = ["double_digit_scorers", 'top_points_avg', 'top_points_position', 'top_player_last_5_avg',
               'top_pra_avg', 'one_player_reliance', 'reliance_ranking', 'double_digit_scorers_ranking']

# Drop unnecessary features
drop_cols = ['adjem_diff', 'net_rating_diff', 'net_rating_rank_diff', 'rankadjem_diff', 'raw_offensive_efficiency_diff',
             'raw_defensive_efficiency_diff', 'adjusted_offensive_efficiency_diff', 'adjusted_defensive_efficiency_diff',
             'adjoe_diff', 'adjde_diff', 'adjusted_offensive_efficiency_rank_diff', 'raw_offensive_efficiency_rank_diff',
             'adjusted_defensive_efficiency_rank_diff', 'raw_defensive_efficiency_rank_diff', 'oe_diff', 'de_diff',
             'rankadjoe_diff', 'rankadjde_diff', 'rankde_diff', 'rankoe_diff', 'conference_win_loss_percentage_diff',
             'topct_diff', 'efgpct_diff', 'ranktopct_diff', 'fg2pct_diff', 'fg3pct_diff', 'orpct_diff',
             'blockpct_diff', 'stlrate_diff', 'ftpct_diff']
df.drop(columns=drop_cols, inplace=True)

# Drop rank, pct, and unnecessary rate columns
df.drop(columns=[col for col in df.columns if 'rank' in col.lower() and col not in player_cols], inplace=True)
df.drop(columns=[col for col in df.columns if 'pct' in col.lower()], inplace=True)
df.drop(columns=[col for col in df.columns if 'rate' in col.lower() and not any(x in col.lower() for x in ['turnover_rate', 'offensive_rebound_rate', 'free_throw_rate'])], inplace=True)

# Split train/test based on date
split_date = '2023-11-01'
train_df = df[df['game_date'] <= split_date].copy()
test_df = df[df['game_date'] > split_date].copy()

excluded_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'game_id', 'game_date', 'home_team', 'home_color', 'away_team', 'away_color', 'target', 'seed_diff']

# **Winner Prediction Model**
tournament_games = train_df[train_df['season_type'] == 3].copy()
train_df_weighted = pd.concat([train_df] + [tournament_games.copy() for _ in range(9)], ignore_index=True)  # 5x weight

X_train = train_df_weighted.drop(columns=excluded_columns)
y_train = train_df_weighted['target'].astype(int)
print(X_train.columns.to_list())
X_test = test_df.drop(columns=excluded_columns)
y_test = test_df['target'].astype(int)

model_winner = LGBMClassifier(objective='binary', metric='logloss', boosting_type='gbdt', verbose=-1,
                              learning_rate=0.03, min_child_samples=20, max_depth=6, num_leaves=255, min_data_in_leaf=5,
                              n_estimators=500)

print("\nTraining winner prediction model...")
model_winner.fit(X_train, y_train)

# **Upset Prediction Model**
tournament_df = train_df[train_df['season_type'] == 3].copy()
upset_games = tournament_df[(tournament_df['seed_diff'] < -4) & (tournament_df['target'] == 0)].copy()
upset_df = pd.concat([tournament_df] + [upset_games.copy() for _ in range(9)], ignore_index=True)  # 5x weight

X_train_upset = upset_df.drop(columns=excluded_columns)
y_train_upset = upset_df['target'].astype(int)

model_upset = LGBMClassifier(objective='binary', metric='logloss', boosting_type='gbdt', verbose=-1,
                             learning_rate=0.05, min_child_samples=20, max_depth=4, num_leaves=127, min_data_in_leaf=5,
                             n_estimators=500, scale_pos_weight=0.4)

print("\nTraining upset prediction model...")
model_upset.fit(X_train_upset, y_train_upset)

# Save models
joblib.dump(model_winner, 'Game Predictions/models/lgbm_winner_model.joblib')
joblib.dump(model_upset, 'Game Predictions/models/lgbm_upset_model.joblib')

# **Evaluation Function**
def evaluate_thresholds(model, X_test, y_test, model_name):
    thresholds = np.linspace(0.1, 0.8, 10)  # Test thresholds from 0.1 to 0.8
    metrics = []
    
    # Create mask for actual 0 values (away team wins)
    actual_zero_mask = (y_test == 0)
    
    for thresh in thresholds:
        # Get predictions based on threshold
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        
        # Calculate standard metrics
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        accuracy = accuracy_score(y_test, y_pred_thresh)
        
        # Calculate accuracy for cases where the actual value is 0
        # This shows how well the model identifies games where away team wins
        if np.sum(actual_zero_mask) > 0:
            actual_zero_accuracy = accuracy_score(
                y_test[actual_zero_mask], 
                y_pred_thresh[actual_zero_mask]
            )
        else:
            actual_zero_accuracy = 0
        
        metrics.append([thresh, precision, recall, f1, accuracy, actual_zero_accuracy])
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(
        metrics, 
        columns=["Threshold", "Precision", "Recall", "F1 Score", "Accuracy", "Accuracy_Actual_0"]
    )
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=metrics_df, x="Threshold", y="Precision", label="Precision", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Recall", label="Recall", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="F1 Score", label="F1 Score", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Accuracy", label="Accuracy", marker="o")
    sns.lineplot(data=metrics_df, x="Threshold", y="Accuracy_Actual_0", 
                 label="Accuracy When Away Team Wins", marker="o", linestyle="--", color="purple")
    
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Optimization - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print count of actual away team wins
    print(f"Number of actual away team wins in test set: {np.sum(actual_zero_mask)}")
    
    # Find and return the best threshold based on F1 Score
    best_threshold = metrics_df.loc[metrics_df["F1 Score"].idxmax(), "Threshold"]
    return best_threshold

# **Plot Threshold Optimization for Both Models**
best_thresh_winner = evaluate_thresholds(model_winner, X_test, y_test, "Winner Prediction Model")
best_thresh_upset = evaluate_thresholds(model_upset, X_test, y_test, "Upset Prediction Model")

print(f"\nOptimal Decision Threshold (Winner Model): {best_thresh_winner:.2f}")
print(f"Optimal Decision Threshold (Upset Model): {best_thresh_upset:.2f}")

# **Feature Importance Plot**
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

# **Plot Feature Importance for Both Models**
plot_feature_importance(model_winner, X_train, "Winner Prediction Model")
plot_feature_importance(model_upset, X_train_upset, "Upset Prediction Model")
