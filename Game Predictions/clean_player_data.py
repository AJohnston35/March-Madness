import pandas as pd
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

dataset = []

for year in range(2003, 2026):  # Update range to process all years
    try:
        print(f"Processing {year}...")
        box_scores = pd.read_csv(f"Data/box_scores/player_games_{year}.csv")
        
        # Sort data by game date
        box_scores = box_scores.sort_values(by=['game_date'])
        
        # Fill missing values
        box_scores.fillna(0, inplace=True)
        
        # Drop unnecessary columns
        cols_to_drop = [
            'season', 'game_date_time', 'team_name', 'team_location', 'team_short_display_name',
            'athlete_short_name', 'athlete_position_name', 'team_display_name', 'team_uid', 'team_slug',
            'team_logo', 'team_abbreviation', 'team_color', 'team_alternate_color', 'team_winner',
            'opponent_team_name', 'opponent_team_display_name', 'opponent_team_abbreviation',
            'opponent_team_logo', 'opponent_team_color', 'opponent_team_alternate_color',
            'offensive_rebounds', 'defensive_rebounds', 'athlete_jersey'
        ]
        
        # Only drop columns that exist
        box_scores = box_scores.drop(columns=[col for col in cols_to_drop if col in box_scores.columns])
        
        # Map position to numerical values
        if 'athlete_position_abbreviation' in box_scores.columns:
            box_scores['position'] = box_scores['athlete_position_abbreviation'].map(
                {'G': 1, 'G-F': 2, 'F': 3, 'F-C': 4, 'C': 5}
            )
            box_scores['position'] = box_scores['position'].fillna(0).astype(int)
            box_scores.drop(columns=['athlete_position_abbreviation'], inplace=True)
        else:
            # If position data is missing, set a default
            box_scores['position'] = 0
        
        # Select relevant columns
        key_columns = [
            'athlete_id', 'game_id', 'game_date', 'season_type', 'position', 'team_id', 
            'opponent_team_id', 'points', 'rebounds', 'assists', 'field_goals_attempted',
            'free_throws_attempted', 'turnovers', 'minutes'
        ]
        
        # Make sure all required columns exist
        existing_columns = set(box_scores.columns)
        required_columns = ['athlete_id', 'game_id', 'game_date', 'team_id', 'points']
        
        if not all(col in existing_columns for col in required_columns):
            missing = [col for col in required_columns if col not in existing_columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        selected_cols = [col for col in key_columns if col in existing_columns]
        box_scores = box_scores[selected_cols]
        
        # Ensure numeric columns are numeric
        numeric_cols = ['points', 'rebounds', 'assists', 'field_goals_attempted', 
                        'free_throws_attempted', 'turnovers', 'minutes']
        for col in numeric_cols:
            if col in box_scores.columns:
                box_scores[col] = pd.to_numeric(box_scores[col], errors='coerce').fillna(0)
        
        # Add combined stat for points + rebounds + assists
        if all(col in box_scores.columns for col in ['points', 'rebounds', 'assists']):
            box_scores['pra'] = box_scores['points'] + box_scores['rebounds'] + box_scores['assists']
        else:
            # If any component is missing, just use points
            box_scores['pra'] = box_scores['points']
        
        # Calculate rolling averages for each player
        N = [5]  # Rolling window sizes
        
        # Compute player-level rolling averages
        numeric_columns = [col for col in ['points', 'rebounds', 'assists', 'pra', 'field_goals_attempted', 
                                          'free_throws_attempted', 'turnovers', 'minutes'] 
                           if col in box_scores.columns]
        
        for stat in numeric_columns:
            # Per-game rolling stats
            for period in N:
                box_scores[f'{stat}_per_game_last_{period}'] = (
                    box_scores.groupby('athlete_id')[stat].transform(
                        lambda x: x.shift(1).rolling(period, min_periods=1).mean()
                    )
                )
            
            # Expanding mean (average of all previous games)
            box_scores[f'{stat}_per_game'] = box_scores.groupby('athlete_id')[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )
        
        # Function to safely find top player info
        def get_top_player_info(group, stat_col):
            col_name = f'{stat_col}_per_game'
            if group.empty or col_name not in group.columns:
                return pd.Series({
                    f'top_{stat_col}_avg': 0,
                    f'top_{stat_col}_position': 0
                })
            
            # Handle NaN values safely
            valid_data = group[~group[col_name].isna()]
            if valid_data.empty:
                return pd.Series({
                    f'top_{stat_col}_avg': 0,
                    f'top_{stat_col}_position': 0
                })
            
            # Find player with highest average
            try:
                top_idx = valid_data[col_name].idxmax()
                return pd.Series({
                    f'top_{stat_col}_avg': valid_data.loc[top_idx, col_name],
                    f'top_{stat_col}_position': valid_data.loc[top_idx, 'position']
                })
            except:
                # Fallback if idxmax fails
                return pd.Series({
                    f'top_{stat_col}_avg': 0,
                    f'top_{stat_col}_position': 0
                })
        
        # Process each game date to avoid data leakage
        game_dates = sorted(box_scores['game_date'].unique())
        team_level_metrics = []
        
        for date in game_dates:
            # Data available up to current date (inclusive)
            data_up_to_date = box_scores[box_scores['game_date'] <= date].copy()
            
            # Games on the current date
            current_games = data_up_to_date[data_up_to_date['game_date'] == date]
            unique_game_teams = current_games[['game_id', 'team_id']].drop_duplicates()
            
            for _, row in unique_game_teams.iterrows():
                game_id = row['game_id']
                team_id = row['team_id']
                
                # Historical data for this team (excluding current game)
                team_history = data_up_to_date[
                    (data_up_to_date['team_id'] == team_id) & 
                    (data_up_to_date['game_date'] < date)
                ].copy()
                
                if team_history.empty:
                    # If this is the first game, use default values
                    team_metrics = {
                        'game_id': game_id,
                        'team_id': team_id,
                        'game_date': date,
                        'double_digit_scorers': 0,
                        'top_points_avg': 0,
                        'top_points_position': 0,
                        'top_player_last_5_avg': 0,
                        'top_pra_avg': 0,
                        'one_player_reliance': 0,
                    }
                else:
                    # Get most recent stats for each player
                    try:
                        latest_player_stats = team_history.groupby('athlete_id').last().reset_index()
                        
                        # Count double-digit scorers - safely handle missing columns
                        if 'points_per_game' in latest_player_stats.columns:
                            double_digit_scorers = sum(latest_player_stats['points_per_game'] >= 10)
                        else:
                            double_digit_scorers = 0
                        
                        # Get top scorer info
                        top_player_info = get_top_player_info(latest_player_stats, 'points')
                        
                        # Get top player's last 5 game average
                        top_player_last_5_avg = 0
                        if 'points_per_game_last_5' in latest_player_stats.columns:
                            valid_data = latest_player_stats[~latest_player_stats['points_per_game_last_5'].isna()]
                            if not valid_data.empty:
                                try:
                                    top_last_5_idx = valid_data['points_per_game_last_5'].idxmax()
                                    top_player_last_5_avg = valid_data.loc[top_last_5_idx, 'points_per_game_last_5']
                                except:
                                    top_player_last_5_avg = 0
                        
                        # Get top P+R+A player
                        top_pra_avg = 0
                        if 'pra_per_game' in latest_player_stats.columns:
                            valid_data = latest_player_stats[~latest_player_stats['pra_per_game'].isna()]
                            if not valid_data.empty:
                                try:
                                    top_pra_idx = valid_data['pra_per_game'].idxmax()
                                    top_pra_avg = valid_data.loc[top_pra_idx, 'pra_per_game']
                                except:
                                    top_pra_avg = 0
                        
                        # Calculate reliance on one player (top scorer's points as % of team avg points)
                        one_player_reliance = 0
                        if 'points_per_game' in latest_player_stats.columns:
                            team_total_points_avg = latest_player_stats['points_per_game'].sum()
                            if team_total_points_avg > 0:
                                one_player_reliance = (top_player_info['top_points_avg'] / team_total_points_avg) * 100
                        
                        team_metrics = {
                            'game_id': game_id,
                            'team_id': team_id,
                            'game_date': date,
                            'double_digit_scorers': double_digit_scorers,
                            'top_points_avg': top_player_info['top_points_avg'],
                            'top_points_position': top_player_info['top_points_position'],
                            'top_player_last_5_avg': top_player_last_5_avg,
                            'top_pra_avg': top_pra_avg,
                            'one_player_reliance': one_player_reliance,
                        }
                    except Exception as e:
                        print(f"Error processing team {team_id} on {date}: {e}")
                        team_metrics = {
                            'game_id': game_id,
                            'team_id': team_id,
                            'game_date': date,
                            'double_digit_scorers': 0,
                            'top_points_avg': 0,
                            'top_points_position': 0,
                            'top_player_last_5_avg': 0,
                            'top_pra_avg': 0,
                            'one_player_reliance': 0,
                        }
                
                team_level_metrics.append(team_metrics)
        
        # Convert to DataFrame 
        if team_level_metrics:
            team_metrics_df = pd.DataFrame(team_level_metrics)
            
            # Calculate rankings at each game date (within this season/year)
            for date in team_metrics_df['game_date'].unique():
                date_data = team_metrics_df[team_metrics_df['game_date'] == date].copy()
                
                # Only calculate rankings if we have data
                if not date_data.empty:
                    # Calculate reliance ranking (higher reliance = higher rank number)
                    team_metrics_df.loc[date_data.index, 'reliance_ranking'] = date_data['one_player_reliance'].rank(
                        ascending=False, method='dense', na_option='bottom'
                    )
                    
                    # Double digit scorers ranking
                    team_metrics_df.loc[date_data.index, 'double_digit_scorers_ranking'] = date_data['double_digit_scorers'].rank(
                        ascending=True, method='dense', na_option='bottom'
                    )
            
            dataset.append(team_metrics_df)
            print(f"Completed {year}")
        else:
            print(f"No team metrics data for {year}")
    
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")

# Combine datasets across years
if dataset:
    team_data = pd.concat(dataset, axis=0)
    
    # Save team-level dataset
    team_data.to_csv("Game Predictions/team_metrics.csv", index=False)
    print("Processing complete! Team metrics saved to 'Game Predictions/team_metrics.csv'")
else:
    print("No data was processed successfully.")