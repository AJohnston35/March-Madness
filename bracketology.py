import joblib

# Load the trained LightGBM model
model = joblib.load('lgbm_model.joblib')

from helper import get_data, preprocess_data

team1_data = get_data(2024)
team2_data = get_data(2024)

team1_data = team1_data[team1_data['school'] == 'Tennessee']
team2_data = team2_data[team2_data['school'] == 'Vanderbilt']

processed_data = preprocess_data(team1_data, team2_data, 'Championship')

processed_data.to_csv('processed_data.csv', index=False)

print(processed_data)

prediction = model.predict(processed_data)

print(prediction)
