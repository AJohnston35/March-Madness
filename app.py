from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from helper import get_data, preprocess_data  # Import your data functions

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

# Load trained model
model = joblib.load('lgbm_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    team1 = data['team1']
    team2 = data['team2']
    
    # Get data for both teams
    team1_data = get_data(2024)
    team2_data = get_data(2024)

    team1_data = team1_data[team1_data['school'] == team1]
    team2_data = team2_data[team2_data['school'] == team2]

    # Preprocess data for model
    processed_data = preprocess_data(team1_data, team2_data, 'Championship')

    # Predict probabilities
    prediction_proba = model.predict_proba(processed_data)[0]
    
    response = {
        "team1": team1,
        "team2": team2,
        "team1_win_prob": round(prediction_proba[1] * 100, 2),
        "team2_win_prob": round(prediction_proba[0] * 100, 2)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
