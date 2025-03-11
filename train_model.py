import random
import joblib
import pandas as pd
from helper import get_data, preprocess_data

tournaments = pd.read_csv('tournament_history/all_tournaments.csv')

all_data = []

for year in range(2023, 1985, -1):
    all_data.append(get_data(year))

all_data = pd.concat(all_data)

processed_data = []

for year in range(2023, 1985, -1):
    print(year)

    tournaments_year = tournaments[tournaments['year'] == year]

    # Filter all_data for the current year to avoid extra matches
    yearly_data = all_data[all_data['year'] == year]

    # Merge tournament data with all_data for both teams
    team1_df = tournaments_year.merge(
        yearly_data, left_on=['team1'], right_on=['school']
    )
    team2_df = tournaments_year.merge(
        yearly_data, left_on=['team2'], right_on=['school']
    )

    team1_df = team1_df.drop(columns=['region','team1','team2','seed1','seed2', 'score1', 'score2'])
    team2_df = team2_df.drop(columns=['region','round','team1','team2','winner', 'seed1', 'seed2', 'score1', 'score2'])

    team1_df['year'] = year
    team2_df['year'] = year

    team1_df = team1_df.drop(columns=['year_x','year_y'])
    team2_df = team2_df.drop(columns=['year_x','year_y'])

    # Process all matchups at once
    processed = preprocess_data(team1_df, team2_df)

    # Ensure processed data has the same number of rows as tournaments_year
    processed = processed.iloc[:len(tournaments_year)].copy()

    # Add relevant columns from tournaments_year
    processed['team1'] = tournaments_year['team1'].values
    processed['team2'] = tournaments_year['team2'].values
    processed['seed1'] = tournaments_year['seed1'].values
    processed['seed2'] = tournaments_year['seed2'].values
    processed['score1'] = tournaments_year['score1'].values
    processed['score2'] = tournaments_year['score2'].values
    processed['winner'] = tournaments_year['winner'].values

    processed_data.append(processed)

# Combine all processed data into a single DataFrame
processed_data = pd.concat(processed_data, ignore_index=True)


# Create a new column for the target variable which is 1 if score1 > score2 and 0 if not
processed_data['target'] = (processed_data['score1'] > processed_data['score2']).astype(int)

subset_df = processed_data[['team1', 'seed1', 'team2', 'seed2', 'winner', 'round']]

processed_data = processed_data.drop(columns=['seed1','seed2','team1','team2','score1','score2','winner', 'g_diff', 'w_diff'])

processed_data.to_csv('full_processed_data.csv', index=False)

print(processed_data['target'])

from lightgbm import LGBMClassifier

# Split the dataset into features and target variable
X = processed_data.drop(columns=['target'])
y = processed_data['target']

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
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
model.fit(X, y)

# Save the model
joblib.dump(model, 'lgbm_model.joblib')