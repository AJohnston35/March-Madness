import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

dataset = []

kenpom = pd.read_csv('Data/kenpom/raw_data.csv')

player_data = pd.read_csv('Game Predictions/team_metrics.csv')
player_data.drop(columns=['game_date'], inplace=True)

# Convert all column names to lowercase and replace spaces with underscores
kenpom.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('.', '_') for col in kenpom.columns]

def calculate_additional_metrics_fixed(games, year):
    # Make a copy to avoid modifying the input DataFrame
    games_copy = games.copy()
    
    # 1. Strength of Schedule (SOS) - Average opponent win percentage up to current game
    # Instead of grouping by team, we'll calculate a running average for each game
    games_copy['strength_of_schedule'] = games_copy.groupby('team_location')['opponent_win_loss_percentage'].transform(
        lambda x: x.expanding().mean()
    )
    
    # Calculate rolling SOS (last 10 games)
    games_copy['rolling_sos_10'] = games_copy.groupby('team_location')['opponent_win_loss_percentage'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    
    # 2. Away record calculation - maintained as time series
    # First, identify away games
    games_copy['is_away_game'] = (games_copy['team_home_away'] == 'away').astype(int)
    
    # Calculate cumulative away games played (shift to avoid current game)
    games_copy['away_games_played'] = games_copy.groupby('team_location')['is_away_game'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    
    # Calculate cumulative away wins and losses for each game date
    # We use the shifted team_winner to avoid including the current game
    games_copy['away_win'] = ((games_copy['team_home_away'] == 'away') & 
                            (games_copy['team_winner_shifted'] == True)).astype(int)
    games_copy['away_loss'] = ((games_copy['team_home_away'] == 'away') & 
                             (games_copy['team_winner_shifted'] == False)).astype(int)
    
    # Calculate running totals for each team
    games_copy['away_wins'] = games_copy.groupby('team_location')['away_win'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    games_copy['away_losses'] = games_copy.groupby('team_location')['away_loss'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    
    # Calculate away win percentage for each game
    games_copy['away_win_percentage'] = games_copy.apply(
        lambda row: (100 * row['away_wins'] / (row['away_wins'] + row['away_losses'])) 
                   if (row['away_wins'] + row['away_losses']) > 0 else 50,
        axis=1
    )
    
    # 3. Non-conference record - maintained as time series
    # Create a flag for non-conference games
    games_copy['prev_conference'] = games_copy.groupby('team_location')['short_conference_name'].shift(1)
    games_copy['is_non_conf_game'] = (games_copy['prev_conference'] != games_copy['short_conference_name']).astype(int)
    
    # Calculate non-conference wins and losses with shifted results
    games_copy['non_conf_win'] = ((games_copy['is_non_conf_game'] == 1) & 
                                (games_copy['team_winner_shifted'] == True)).astype(int)
    games_copy['non_conf_loss'] = ((games_copy['is_non_conf_game'] == 1) & 
                                 (games_copy['team_winner_shifted'] == False)).astype(int)
    
    # Calculate running totals for each team
    games_copy['non_conf_wins'] = games_copy.groupby('team_location')['non_conf_win'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    games_copy['non_conf_losses'] = games_copy.groupby('team_location')['non_conf_loss'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    
    # Calculate non-conference win percentage
    games_copy['non_conf_win_percentage'] = games_copy.apply(
        lambda row: (100 * row['non_conf_wins'] / (row['non_conf_wins'] + row['non_conf_losses']))
                   if (row['non_conf_wins'] + row['non_conf_losses']) > 0 else 50,
        axis=1
    )
    
    # Clean up intermediate columns
    games_copy = games_copy.drop(columns=[
        'is_away_game', 'away_win', 'away_loss', 
        'prev_conference', 'is_non_conf_game', 'non_conf_win', 'non_conf_loss'
    ])
    
    print(f"Added time-series SOS, away record, and non-conference record for {year}")
    
    return games_copy

for year in range(2003, 2026):
    games = pd.read_csv(f"Data/game_results/games_{year}.csv")

    games['year'] = year

    # Drop unnecessary columns
    games = games.drop(columns=['season', 'game_date_time', 'team_uid', 'team_slug', 'team_name',
                                'team_abbreviation', 'team_display_name', 'team_short_display_name',
                                'team_alternate_color', 'team_logo', 'opponent_team_id', 'opponent_team_uid',
                                'opponent_team_slug', 'opponent_team_name', 'opponent_team_abbreviation',
                                'opponent_team_display_name', 'opponent_team_short_display_name',
                                'opponent_team_alternate_color', 'opponent_team_logo',
                                'opponent_team_color'])
    
    games = games.rename(columns={"team_score": "points", "opponent_team_score": "opponent_points"})
    
    # Sort by game date
    games = games.sort_values(by=['game_date'])

    games = games.merge(
        kenpom[['team_location', 'short_conference_name', 'year']],  
        on=['team_location', 'year'],       
        how='left'              
    )

    games = games.drop(columns=['year'])

    games['possessions'] = games['field_goals_attempted'] - games['offensive_rebounds'] + games['turnovers'] + (games['free_throws_attempted']/2)
    games['offensive_efficiency'] = (games['points'] / games['possessions']) * 100
    games['defensive_efficiency'] = (games['opponent_points'] / games['possessions']) * 100
    games['net_efficiency'] = games['offensive_efficiency'] - games['defensive_efficiency']
    games['efg_percent'] = (((games['field_goals_made'] + 0.5) * games['three_point_field_goals_made']) / games['field_goals_attempted']) * 100
    games['turnover_rate'] = (games['turnovers'] / games['possessions']) * 100
    games['offensive_rebound_rate'] = games['offensive_rebounds'] / (games['field_goals_attempted'] - games['field_goals_made']) * 100
    games['free_throw_rate'] = (games['free_throws_attempted'] / games['field_goals_attempted']) * 100
    games['single_digit_win'] = np.where(games['points'] - games['opponent_points'] < 10, 1, 0)
    games['single_digit_loss'] = np.where(games['opponent_points'] - games['points'] < 10, 1, 0)
    games['blowout_win'] = np.where(games['points'] - games['opponent_points'] > 19, 1, 0)
    games['blowout_loss'] = np.where(games['opponent_points'] - games['points'] > 19, 1, 0)
    games['clutch_win'] = np.where(games['points'] - games['opponent_points'] <= 3, 1, 0)
    games['clutch_loss'] = np.where(games['opponent_points'] - games['points'] <= 3, 1, 0)
    games['competitive_game'] = np.where(abs(games['points'] - games['opponent_points']) < 6, 1, 0)
    games['margin_of_victory'] = games['points'] - games['opponent_points']
    games['blown_lead'] = games['largest_lead'] - games['margin_of_victory']

    games['single_digit_win'] = games.groupby('team_location')['single_digit_win'].shift(1)
    games['single_digit_loss'] = games.groupby('team_location')['single_digit_loss'].shift(1)
    games['blowout_win'] = games.groupby('team_location')['blowout_win'].shift(1)
    games['blowout_loss'] = games.groupby('team_location')['blowout_loss'].shift(1)
    games['clutch_win'] = games.groupby('team_location')['clutch_win'].shift(1)
    games['clutch_loss'] = games.groupby('team_location')['clutch_loss'].shift(1)
    games['competitive_game'] = games.groupby('team_location')['competitive_game'].shift(1)
    games['margin_of_victory'] = games.groupby('team_location')['margin_of_victory'].shift(1)
    games['blown_lead'] = games.groupby('team_location')['blown_lead'].shift(1)

    # Identify numeric columns for cumulative averages (excluding certain non-statistical columns)
    numeric_columns = [col for col in games.columns if games[col].dtype in ['int64', 'float64']
                       and col not in ['team_id', 'game_id', 'season_type', 'team_winner']]

    # Calculate cumulative averages using shifted values to prevent leakage
    for stat in numeric_columns:
        games[f'{stat}_per_game'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).expanding().mean())
        games[f'{stat}_stdev'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).expanding().std())
    
    N = [3,5,10]  # Number of past games to consider

    for stat in numeric_columns:
        for n in N:
            # Rolling mean of past N games
            games[f'{stat}_rolling_mean_{n}'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).rolling(n).mean())

            # Rolling standard deviation of past N games (consistency measure)
            games[f'{stat}_rolling_stdev_{n}'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).rolling(n).std())
    
    games = games.merge(
        player_data,
        left_on=['game_id', 'team_id'],
        right_on=['game_id', 'team_id'],
        how='left'
    )

    games.drop(columns=['team_id'], inplace=True)

    # Shift 'team_winner' before cumulative counts to avoid including the current game
    games['team_winner_shifted'] = games.groupby('team_location')['team_winner'].shift(1)

    # Calculate cumulative wins and losses
    games['wins'] = games.groupby('team_location')['team_winner_shifted'].transform(lambda x: (x == True).cumsum())
    games['losses'] = games.groupby('team_location')['team_winner_shifted'].transform(lambda x: (x == False).cumsum())

    # Conference-level win/loss calculations with shifting
    games['conference_wins'] = games.groupby('short_conference_name')['team_winner_shifted'].transform(lambda x: (x == True).cumsum())
    games['conference_losses'] = games.groupby('short_conference_name')['team_winner_shifted'].transform(lambda x: (x == False).cumsum())
    games['conference_win_loss_percentage'] = games['conference_wins'] / (games['conference_wins'] + games['conference_losses'])

    games.drop(columns=['conference_wins', 'conference_losses'], inplace=True)

    # Create conference rankings based on win percentage
    conference_rankings = (
        games.groupby(['game_date', 'short_conference_name'])
        .agg({'conference_win_loss_percentage': 'last'})  
        .reset_index()
    )

    conference_rankings['conference_ranking'] = conference_rankings.groupby('game_date')['conference_win_loss_percentage'].rank(
        ascending=False, method='dense'
    )

    # Merge rankings back into games data
    games = games.merge(conference_rankings[['game_date', 'short_conference_name', 'conference_ranking']], 
                        on=['game_date', 'short_conference_name'], 
                        how='left')

    # Win/loss percentage (using shifted values)
    games['win_loss_percentage'] = games.groupby('team_location')['team_winner_shifted'].transform(lambda x: x.expanding().mean()) * 100

    # Merge opponent win/loss percentage (using game_id)
    opponent_stats = games[['game_id', 'team_location', 'win_loss_percentage']].copy()
    opponent_stats.rename(columns={'team_location': 'opponent_team_location',
                                   'win_loss_percentage': 'opponent_win_loss_percentage'}, inplace=True)

    games = games.merge(opponent_stats, on=['game_id', 'opponent_team_location'], how='left')

    games = calculate_additional_metrics_fixed(games, year)
    games = games.drop(columns=['short_conference_name'])

    # Calculate past 10-game and 15-game win-loss percentages for opponents (shifted)
    games['opponent_recent_win_loss_5'] = games.groupby('opponent_team_location')['opponent_win_loss_percentage']\
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()) * 100

    # Calculate past 10-game and 15-game win-loss percentages for opponents (shifted)
    games['opponent_recent_win_loss_10'] = games.groupby('opponent_team_location')['opponent_win_loss_percentage']\
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()) * 100

    games['opponent_recent_win_loss_15'] = games.groupby('opponent_team_location')['opponent_win_loss_percentage']\
        .transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()) * 100

    # Fill missing values (assumption: 0.5 for neutral)
    games[['opponent_win_loss_percentage', 'opponent_recent_win_loss_5', 'opponent_recent_win_loss_10', 'opponent_recent_win_loss_15']] = \
        games[['opponent_win_loss_percentage', 'opponent_recent_win_loss_5','opponent_recent_win_loss_10', 'opponent_recent_win_loss_15']].fillna(50)

    games.drop(columns=['opponent_team_location'], inplace=True)

    # Calculate momentum (past 10 and 15-game moving averages of wins)
    games['momentum_5'] = games.groupby('team_location')['team_winner_shifted']\
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean()) * 100

    # Calculate momentum (past 10 and 15-game moving averages of wins)
    games['momentum_10'] = games.groupby('team_location')['team_winner_shifted']\
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean()) * 100

    games['momentum_15'] = games.groupby('team_location')['team_winner_shifted']\
        .transform(lambda x: x.rolling(window=15, min_periods=1).mean()) * 100
    
    games['weighted_momentum_5'] = (games['opponent_recent_win_loss_5'] * games['momentum_5']) / games['conference_ranking']
    games['weighted_momentum_10'] = (games['opponent_recent_win_loss_10'] * games['momentum_10']) / games['conference_ranking']
    games['weighted_momentum_15'] = (games['opponent_recent_win_loss_15'] * games['momentum_15']) / games['conference_ranking']

    games.drop(columns=['team_winner_shifted'], inplace=True)

    # Drop original numeric columns (since their per-game versions are used)
    games = games.drop(columns=numeric_columns)

    games.to_csv(f'Game Predictions/assets/games/games_{year}.csv')

    # Split into home and away
    home = games[games['team_home_away'] == "home"]

    away = games[games['team_home_away'] == "away"]

    home['year'] = year
    away['year'] = year

    home = home.merge(kenpom, on=['team_location', 'year'], how='left')
    home = home.drop(columns=['year'])

    for column in home.columns:
        if home[column].dtype not in ['int64','float64']:
            try:
                home[column] = home[column].astype(int)
            except:
                print(f"Could not convert column {column} to integer")


    away = away.merge(kenpom, on=['team_location', 'year'], how='left')
    away = away.drop(columns=['year', 'season_type'])

    for column in away.columns:
        if away[column].dtype not in ['int64','float64']:
            try:
                away[column] = away[column].astype(int)
            except:
                print(f"Could not convert column {column} to integer")

    merged_data = pd.merge(home, away, on='game_id', suffixes=('_home', '_away'))

    merged_data['seed_home'] = merged_data.apply(lambda x: 0 if x['season_type'] == 2 else x['seed_home'], axis=1)
    merged_data['seed_away'] = merged_data.apply(lambda x: 0 if x['season_type'] == 2 else x['seed_away'], axis=1)

    merged_data = merged_data.drop(columns=['team_winner_away', 'team_home_away_home', 'team_home_away_away',
                                            'game_date_away', 'short_conference_name_home','short_conference_name_away'])

    merged_data.rename(columns={
        "team_location_home": "home_team",
        "team_color_home": "home_color",
        "team_location_away": "away_team",
        "team_color_away": "away_color",
        "season_type_home": "season_type",
        "game_date_home": "game_date",
        "team_winner_home": "target"
    }, inplace=True)

    # Calculate stat differences
    for column in merged_data.columns:
        if column.endswith('_home'):
            base_column = column[:-5]
            if f'{base_column}_away' in merged_data.columns:
                merged_data[base_column + '_diff'] = merged_data[column] - merged_data[f'{base_column}_away']
                merged_data.drop([column, f'{base_column}_away'], axis=1, inplace=True)

    dataset.append(merged_data)

    print(f"{year} processed.")

full_data = pd.concat(dataset, axis=0)
seed_records = pd.read_csv('data/seed_records_fixed.csv')
full_data = full_data.merge(
    seed_records,
    on=['seed_diff'],
    how='left'
)
full_data.to_csv("Game Predictions/cleaned_dataset.csv")