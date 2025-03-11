import random
import joblib
import pandas as pd
from helper import get_data, preprocess_data

# Load the trained LightGBM model
model = joblib.load('lgbm_model.joblib')

year = 2024  # Change for different years

# Load actual tournament matchups
df = pd.read_csv(f'tournament_history/old/tournament_games_{year}.csv')

# Add a unique game_id to each game in the tournament
df['game_id'] = range(1, len(df) + 1)

# Define ESPN scoring system
scoring_system = {
    "First Round": 10,
    "Second Round": 20,
    "Sweet Sixteen": 40,
    "Elite Eight": 80,
    "Final Four": 160,
    "Championship": 320
}

# Initialize scores
score = 0
possible_score = 0

# Dictionary to track predicted winners by game_id
predicted_winners = {}

# Tournament rounds
rounds = ["First Round", "Second Round", "Sweet Sixteen", "Elite Eight"]
national_rounds = ["Final Four", "Championship"]

# Get unique regions
regions = df['region'].unique()

print("\nPredicting Tournament Outcomes\n" + "="*30)

# Process each region
for region in regions:
    # Skip National region - we'll process it separately
    if region == "National":
        continue
        
    print(f"\nProcessing {region} Region")
    
    # Process each round for this region
    for current_round in rounds:
        print(f"\n  {current_round}")
        
        # Get games for this region and round
        region_round_games = df[(df['region'] == region) & (df['round'] == current_round)]
        
        if region_round_games.empty:
            print(f"    No games found for {region} {current_round}")
            continue
        
        # Process each game in this round
        for _, game in region_round_games.iterrows():
            game_id = game['game_id']
            team1 = game['team1']
            team2 = game['team2']
            actual_winner = game['winner']
            
            # For first round, use actual teams
            if current_round == "First Round":
                team1_for_prediction = team1
                team2_for_prediction = team2
            else:
                # For later rounds, we need to find the predicted winners from previous round
                # Find games from previous round where winners advanced to this game
                prev_round = rounds[rounds.index(current_round) - 1]
                prev_round_games = df[(df['region'] == region) & (df['round'] == prev_round)]
                
                # Find the games that led to team1 and team2
                team1_prev_game = None
                team2_prev_game = None
                
                # This logic needs to match the tournament bracket structure
                # In a real tournament, this would be based on seeding and bracket position
                # For simplicity, we're using a sequential approach within each region
                for _, prev_game in prev_round_games.iterrows():
                    if prev_game['winner'] == team1:
                        team1_prev_game = prev_game['game_id']
                    if prev_game['winner'] == team2:
                        team2_prev_game = prev_game['game_id']
                
                # Use our predicted winners from previous games if available
                if team1_prev_game in predicted_winners:
                    team1_for_prediction = predicted_winners[team1_prev_game]
                else:
                    team1_for_prediction = team1
                    
                if team2_prev_game in predicted_winners:
                    team2_for_prediction = predicted_winners[team2_prev_game]
                else:
                    team2_for_prediction = team2
            
            print(f"    Game {game_id}: {team1_for_prediction} vs {team2_for_prediction}")
            if current_round != "First Round":
                print(f"    (Actual matchup: {team1} vs {team2})")
            
            # Get team data for prediction
            team_data = get_data(year)
            team1_data = team_data[team_data['school'] == team1_for_prediction].copy()
            team2_data = team_data[team_data['school'] == team2_for_prediction].copy()
            
            # Skip if data is missing
            if team1_data.empty or team2_data.empty:
                print(f"      Skipping game {game_id}: Missing data for {team1_for_prediction} or {team2_for_prediction}")
                continue
            
            # Prepare data for prediction
            team1_data['round'] = current_round
            team2_data['round'] = current_round
            
            team1_data = team1_data.drop(columns=['g', 'w'], errors='ignore')
            team2_data = team2_data.drop(columns=['g', 'w'], errors='ignore')
            
            # Predict winner
            try:
                # Process data in both directions to reduce home/away bias
                processed_data_1 = preprocess_data(team1_data, team2_data)
                processed_data_2 = preprocess_data(team2_data, team1_data)
                
                processed_data_1 = processed_data_1.drop(columns=['round_team2'], errors='ignore')
                processed_data_2 = processed_data_2.drop(columns=['round_team2'], errors='ignore')
                
                # Get predictions
                proba_1 = model.predict_proba(processed_data_1)[0]
                proba_2 = model.predict_proba(processed_data_2)[0]
                
                # Average the probabilities (team1 winning vs team2 losing)
                team1_avg_proba = (proba_1[1] + proba_2[0]) / 2
                team2_avg_proba = (proba_1[0] + proba_2[1]) / 2

                if abs(team1_avg_proba - team2_avg_proba) < 0.15:
                    # Get the predicted winner with a random value based on the predicted probabilities
                    random_value = random.uniform(0, 1)
                    if random_value < team1_avg_proba:
                        predicted_winner = team1_for_prediction
                    else:
                        predicted_winner = team2_for_prediction
                else: 
                    # Determine predicted winner
                    predicted_winner = team1_for_prediction if team1_avg_proba > team2_avg_proba else team2_for_prediction

                probability = 0
                if predicted_winner == team1_for_prediction:
                    probability = team1_avg_proba * 100
                else:
                    probability = team2_avg_proba * 100

                # Store the predicted winner by game_id
                predicted_winners[game_id] = predicted_winner
                
                # Check if prediction matches actual winner
                prediction_correct = predicted_winner == actual_winner
                
                # Update score
                round_points = scoring_system[current_round]
                if prediction_correct:
                    score += round_points
                    result = "CORRECT"
                else:
                    result = "WRONG"
                    
                possible_score += round_points
                
                print(f"      Predicted winner: {predicted_winner} ({probability:.2f}%), Actual winner: {actual_winner} - {result} (+{round_points if prediction_correct else 0} pts)")
            
            except Exception as e:
                print(f"      Error predicting game {game_id}: {e}")

# Process National Region (Final Four and Championship)
print("\nProcessing National Region")

# Get Final Four teams (region winners from Elite Eight)
elite_eight_winners = {}
for region in regions:
    if region != "National":
        # Get the Elite Eight game for this region
        elite_eight_game = df[(df['region'] == region) & (df['round'] == "Elite Eight")]
        if not elite_eight_game.empty:
            game_id = elite_eight_game.iloc[0]['game_id']
            if game_id in predicted_winners:
                # Use our predicted winner
                elite_eight_winners[region] = predicted_winners[game_id]
            else:
                # Use actual winner if no prediction
                elite_eight_winners[region] = elite_eight_game.iloc[0]['winner']

print("\n  Final Four Teams (predicted):")
for region, team in elite_eight_winners.items():
    print(f"    {region}: {team}")

# Create predicted Final Four matchups based on bracket structure
# In NCAA tournament, the regions are matched: East vs West, Midwest vs South
predicted_final_four = []
region_pairs = [("East", "West"), ("Midwest", "South")]

for regions_pair in region_pairs:
    if regions_pair[0] in elite_eight_winners and regions_pair[1] in elite_eight_winners:
        predicted_final_four.append({
            "team1": elite_eight_winners[regions_pair[0]],
            "team2": elite_eight_winners[regions_pair[1]]
        })

# Process National rounds
for current_round in national_rounds:
    print(f"\n  {current_round}")
    
    # Get games for National region and this round
    national_games = df[(df['region'] == "National") & (df['round'] == current_round)]
    
    if national_games.empty:
        print(f"    No games found for National {current_round}")
        continue
    
    # Process each game in this round
    for _, game in national_games.iterrows():
        game_id = game['game_id']
        team1 = game['team1']
        team2 = game['team2']
        actual_winner = game['winner']
        
        if current_round == "Final Four":
            # For Final Four, use our predicted matchups
            ff_idx = list(national_games.index).index(game.name)
            
            if ff_idx < len(predicted_final_four):
                team1_for_prediction = predicted_final_four[ff_idx]["team1"]
                team2_for_prediction = predicted_final_four[ff_idx]["team2"]
            else:
                # Fallback if something went wrong with our predictions
                team1_for_prediction = team1
                team2_for_prediction = team2
        else:  # Championship
            # For Championship, use our predicted winners from Final Four
            ff_games = df[(df['region'] == "National") & (df['round'] == "Final Four")]
            
            # Get the game_ids for the Final Four games
            ff_game_ids = list(ff_games['game_id'])
            
            # Use our predicted winners for the championship
            if len(ff_game_ids) >= 2 and ff_game_ids[0] in predicted_winners and ff_game_ids[1] in predicted_winners:
                team1_for_prediction = predicted_winners[ff_game_ids[0]]
                team2_for_prediction = predicted_winners[ff_game_ids[1]]
            else:
                # Fallback to actual teams if predictions aren't available
                team1_for_prediction = team1
                team2_for_prediction = team2
        
        print(f"    Game {game_id}: {team1_for_prediction} vs {team2_for_prediction}")
        if current_round != "First Round":
            print(f"    (Actual matchup: {team1} vs {team2})")
        
        # Get team data for prediction
        team_data = get_data(year)
        team1_data = team_data[team_data['school'] == team1_for_prediction].copy()
        team2_data = team_data[team_data['school'] == team2_for_prediction].copy()
        
        # Skip if data is missing
        if team1_data.empty or team2_data.empty:
            print(f"      Skipping game {game_id}: Missing data for {team1_for_prediction} or {team2_for_prediction}")
            continue
        
        # Prepare data for prediction
        team1_data['round'] = current_round
        team2_data['round'] = current_round
        
        team1_data = team1_data.drop(columns=['g', 'w'], errors='ignore')
        team2_data = team2_data.drop(columns=['g', 'w'], errors='ignore')
        
        # Predict winner
        try:
            # Process data in both directions to reduce home/away bias
            processed_data_1 = preprocess_data(team1_data, team2_data)
            processed_data_2 = preprocess_data(team2_data, team1_data)
            
            processed_data_1 = processed_data_1.drop(columns=['round_team2'], errors='ignore')
            processed_data_2 = processed_data_2.drop(columns=['round_team2'], errors='ignore')
            
            # Get predictions
            proba_1 = model.predict_proba(processed_data_1)[0]
            proba_2 = model.predict_proba(processed_data_2)[0]
            
            # Average the probabilities (team1 winning vs team2 losing)
            team1_avg_proba = (proba_1[1] + proba_2[0]) / 2
            team2_avg_proba = (proba_1[0] + proba_2[1]) / 2
            
            # Determine predicted winner
            predicted_winner = team1_for_prediction if team1_avg_proba > team2_avg_proba else team2_for_prediction
            
            # Store the predicted winner by game_id
            predicted_winners[game_id] = predicted_winner
            
            # Check if prediction matches actual winner
            prediction_correct = predicted_winner == actual_winner
            
            # Update score
            round_points = scoring_system[current_round]
            if prediction_correct:
                score += round_points
                result = "CORRECT"
            else:
                result = "WRONG"
                
            possible_score += round_points
            
            print(f"      Predicted winner: {predicted_winner}, Actual winner: {actual_winner} - {result} (+{round_points if prediction_correct else 0} pts)")
        
        except Exception as e:
            print(f"      Error predicting game {game_id}: {e}")

# Print final results
print("\n" + "="*50)
print(f"Final Score: {score}/{possible_score}")

# Find the championship game
championship_game = df[(df['region'] == "National") & (df['round'] == "Championship")]
if not championship_game.empty:
    champion_game_id = championship_game.iloc[0]['game_id']
    if champion_game_id in predicted_winners:
        print(f"Predicted Champion: {predicted_winners[champion_game_id]}")
    else:
        print("No prediction for championship game")
    print(f"Actual Champion: {championship_game.iloc[0]['winner']}")
print("="*50)