import pandas as pd

df = pd.read_csv('final_merged_data.csv')

# Convert all column names to lowercase
df.columns = df.columns.str.lower()

round_mapping = {
    'First Round': 0,
    'Second Round': 1,
    'Sweet Sixteen': 2,
    'Elite Eight': 3,
    'Final Four': 4,
    'Championship': 5
}

df['round'] = df['round'].map(round_mapping)

df = df.drop(columns=['region'])

# Ensure no duplicate columns before processing
df = df.loc[:, ~df.columns.duplicated()]

for column in df.columns:
    if column.endswith('_team2'):
        base_column = column[:-6]  # Remove the '_team2' suffix
        if base_column in df.columns:  # Check if base_column exists
            diff_column = f"{base_column}_diff"
            df[diff_column] = df[base_column] - df[column]
            df = df.drop(columns=[base_column, column])

for column in df.columns:
    if column.endswith('_rating_team2'):
        base_column = column[:-12]  # Remove the '_rating_team2' suffix
        if base_column in df.columns:  # Check if base_column exists
            diff_column = f"{base_column}_rating_diff"
            df[diff_column] = df[base_column] - df[column]
            df = df.drop(columns=[base_column, column])

columns_to_diff = [
    'school_team2', 'rk_rating_team2', 'school_rating_team2', 
    'conf_rating_team2', 'w_rating_team2', 'l_rating_team2', 
    'pts_rating_team2', 'opp_rating_team2', 'mov_rating_team2', 
    'sos_rating_team2', 'osrs_rating_team2', 'dsrs_rating_team2', 
    'srs_rating_team2', 'ortg_rating_team2', 'drtg_rating_team2', 
    'nrtg_rating_team2'
]

for column in columns_to_diff:
    if column in df.columns and df[column].dtype in ['int64', 'float64']:  # Check if the column is numeric
        base_column_name = column.replace('_rating_team2', '')  # Remove the '_rating_team2' suffix to get the base column name
        diff_column = f"{base_column_name}_diff"
        if column.replace('_rating_team2', '') in df.columns:  # Check if the corresponding team1 column exists
            df[diff_column] = df[column] - df[column.replace('_rating_team2', '')]  # Calculate the difference
            df = df.drop(columns=[column, column.replace('_rating_team2', '')])  # Drop both original columns after differencing

for column in df.columns:
    if '%' in column:
        df[column] = df[column] * 100


# Create a new column for the target variable which is 1 if score1 > score2 and 0 if not
df['target'] = (df['score1'] > df['score2']).astype(int)

subset_df = df[['team1', 'seed1', 'team2', 'seed2', 'winner', 'year', 'round']]

df = df.drop(columns=['seed1','seed2','team1','team2','score1', 'score2', 'winner', 'school_team2', 'conf', 'rk_rating_team2','school_rating_team2','conf_rating_team2','w_rating_team2','l_rating_team2','sos_rating_team2','srs_rating_team2','ortg_rating_team2', 'rk_diff','g_diff','rk_diff'])

print(df['target'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier

# Split the dataset into features and target variable
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Store indices to match with subset_df
train_indices = X_train.index
test_indices = X_test.index

# Reset the subset_df index
subset_df.reset_index(drop=True, inplace=True)

# Create the subset data for train and test data based on indices
train_subset = subset_df.loc[train_indices].reset_index(drop=True)
test_subset = subset_df.loc[test_indices].reset_index(drop=True)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weight_0, weight_1 = class_weights

# Calculate scale_pos_weight for LightGBM
scale_pos_weight = weight_0 / weight_1

# Initialize the model and train it
model = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    boosting_type='gbdt',
    verbose=-1,
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    min_child_samples=20,
    n_estimators=100,
    num_leaves=31
)
model.fit(X_train, y_train)

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities
y_pred = (y_pred_proba > 0.5).astype(int)  # Convert to binary predictions

# Visualizing matchups and predictions:
# Merge the predictions with the corresponding matchups
test_predictions_df = test_subset.copy()  # Create a copy of the test_subset DataFrame
test_predictions_df['predicted_winner'] = y_pred  # Add predictions to the DataFrame

# Map predicted winner (1 or 0) to the corresponding team
test_predictions_df['predicted_team'] = test_predictions_df.apply(
    lambda row: row['team1'] if row['predicted_winner'] == 1 else row['team2'], axis=1)

# Now, you can visualize this dataframe with the predictions
print(test_predictions_df[['team1', 'seed1', 'team2', 'seed2', 'predicted_team', 'predicted_winner']])

# Create a subset where the actual winner and predicted team do not match
mismatches_df = test_predictions_df[test_predictions_df['winner'] != test_predictions_df['predicted_team']]

# Save the mismatches to a CSV file
#mismatches_df.to_csv('alt_mismatches_predictions.csv', index=False)

# You can save it for later use or visualization
#test_predictions_df.to_csv('alt_predictions_with_matchups.csv', index=False)

# Additional performance metrics (accuracy, precision, recall, etc.)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Get feature importances
importances = model.feature_importances_
feature_names = X_train.columns  # Assuming X_train is a DataFrame

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))  # Top 10 features
plt.title('Top 10 Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
