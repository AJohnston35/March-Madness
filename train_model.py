import joblib
import pandas as pd
from helper import get_data, preprocess_data

tournaments = pd.read_csv('tournament_history/all_tournaments.csv')

all_data = []

for year in range(2024, 1985, -1):
    print(year)
    all_data.append(get_data(year))

all_data = pd.concat(all_data)

processed_data = []

for year in range(2024, 1985, -1):
    tournaments_year = tournaments[tournaments['year'] == year]
    for index, row in tournaments_year.iterrows():
        print(row['team1'], row['team2'])
        team1_data = all_data[(all_data['school'] == row['team1']) & (all_data['year'] == row['year'])]
        team2_data = all_data[(all_data['school'] == row['team2']) & (all_data['year'] == row['year'])]
        processed_data_entry = preprocess_data(team1_data, team2_data, row['round'])
        processed_data_entry['team1'] = row['team1']
        processed_data_entry['team2'] = row['team2']
        processed_data_entry['seed1'] = row['seed1']
        processed_data_entry['seed2'] = row['seed2']
        processed_data_entry['score1'] = row['score1']
        processed_data_entry['score2'] = row['score2']
        processed_data_entry['winner'] = row['winner']
        processed_data.append(processed_data_entry)

processed_data = pd.concat(processed_data)

processed_data.to_csv('full_processed_data.csv', index=False)

# Create a new column for the target variable which is 1 if score1 > score2 and 0 if not
processed_data['target'] = (processed_data['score1'] > processed_data['score2']).astype(int)

subset_df = processed_data[['team1', 'seed1', 'team2', 'seed2', 'winner', 'year', 'round']]

processed_data = processed_data.drop(columns=['seed1','seed2','team1','team2','score1','score2','winner'])

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