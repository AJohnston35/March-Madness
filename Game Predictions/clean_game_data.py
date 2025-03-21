import pandas as pd

dataset = []

sos = pd.read_csv('Game Predictions/assets/all_ratings.csv')
kenpom = pd.read_csv('Data/kenpom/raw_data.csv')

# Convert all column names to lowercase and replace spaces with underscores
kenpom.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('.', '_') for col in kenpom.columns]

for year in range(2003, 2026):
    games = pd.read_csv(f"Data/game_results/games_{year}.csv")

    games['year'] = year

    # Drop unnecessary columns
    games = games.drop(columns=['season', 'game_date_time', 'team_id', 'team_uid', 'team_slug', 'team_name',
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

    # Identify numeric columns for cumulative averages (excluding certain non-statistical columns)
    numeric_columns = [col for col in games.columns if games[col].dtype in ['int64', 'float64']
                       and col not in ['game_id', 'season_type', 'team_winner']]

    # Calculate cumulative averages using shifted values to prevent leakage
    for stat in numeric_columns:
        games[f'{stat}_per_game'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).expanding().mean())
        games[f'{stat}_stdev'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).expanding().std())
    
    N = 5  # Number of past games to consider

    for stat in numeric_columns:
        # Rolling mean of past N games
        games[f'{stat}_rolling_mean_{N}'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).rolling(N).mean())

        # Rolling standard deviation of past N games (consistency measure)
        games[f'{stat}_rolling_stdev_{N}'] = games.groupby('team_location')[stat].transform(lambda x: x.shift(1).rolling(N).std())

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
    
    games = games.drop(columns=['short_conference_name'])

    # Win/loss percentage (using shifted values)
    games['win_loss_percentage'] = games.groupby('team_location')['team_winner_shifted'].transform(lambda x: x.expanding().mean()) * 100

    # Merge opponent win/loss percentage (using game_id)
    opponent_stats = games[['game_id', 'team_location', 'win_loss_percentage']].copy()
    opponent_stats.rename(columns={'team_location': 'opponent_team_location',
                                   'win_loss_percentage': 'opponent_win_loss_percentage'}, inplace=True)

    games = games.merge(opponent_stats, on=['game_id', 'opponent_team_location'], how='left')

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

    # Merge SOS ratings for home teams
    home = home.merge(sos[['School', 'SOS', 'year']], left_on=['team_location', 'year'], right_on=['School', 'year'], how='left')
    home['momentum_strength_5'] = home['SOS'] * home['weighted_momentum_5']
    home['momentum_strength_10'] = home['SOS'] * home['weighted_momentum_10']
    home['momentum_strength_15'] = home['SOS'] * home['weighted_momentum_15']
    home['win_strength'] = home['SOS'] * home['win_loss_percentage']
    home = home.drop(columns=['School'])

    home = home.merge(kenpom, on=['team_location', 'year'], how='left')
    home = home.drop(columns=['year'])

    for column in home.columns:
        if home[column].dtype not in ['int64','float64']:
            try:
                home[column] = home[column].astype(int)
            except:
                print(f"Could not convert column {column} to integer")

    # Merge opponent data for away teams
    away = away.merge(sos[['School', 'SOS', 'year']], left_on=['team_location', 'year'], right_on=['School', 'year'], how='left')
    away = away.drop(columns=['School'])
    away['momentum_strength_5'] = away['SOS'] * away['weighted_momentum_5']
    away['momentum_strength_10'] = away['SOS'] * away['weighted_momentum_10']
    away['momentum_strength_15'] = away['SOS'] * away['weighted_momentum_15']
    away['win_strength'] = away['SOS'] * away['win_loss_percentage']

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