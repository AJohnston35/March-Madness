import pandas as pd

dataset = []

sos = pd.read_csv('assets/all_ratings.csv')

for year in range(2003, 2026):
    games = pd.read_csv(f"data/game_results/games_{year}.csv")
    # Drop unnecessary columns
    games = games.drop(columns=['season', 'game_date_time', 'team_id', 'team_uid', 'team_slug', 'team_name',
                                'team_abbreviation', 'team_display_name', 'team_short_display_name',
                                'team_alternate_color', 'team_logo', 'opponent_team_id', 'opponent_team_uid',
                                'opponent_team_slug', 'opponent_team_name', 'opponent_team_abbreviation',
                                'opponent_team_display_name', 'opponent_team_short_display_name',
                                'opponent_team_alternate_color', 'opponent_team_logo',
                                'opponent_team_location', 'opponent_team_color'])
    
    games = games.rename(columns={
        "team_score": "points",
        "opponent_team_score": "opponent_points"
    })
    
    # Sort by game date
    games = games.sort_values(by=['game_date'])
    
    # Identify numeric columns for cumulative averages
    numeric_columns = [col for col in games.columns if games[col].dtype in ['int64', 'float64']
                       and col not in ['game_id', 'season_type']]
    
    # Calculate cumulative averages for each team
    for stat in numeric_columns:
        games[f'{stat}_per_game'] = games.groupby('team_location')[stat].transform(lambda x: x.expanding().mean())
    
    # Calculate cumulative wins and losses
    games['wins'] = games.groupby('team_location')['team_winner'].transform(lambda x: (x == True).cumsum())
    games['losses'] = games.groupby('team_location')['team_winner'].transform(lambda x: (x == False).cumsum())
    
    # Win/loss percentage
    games['win_loss_percentage'] = games['wins'] / (games['wins'] + games['losses'])

    # Create a 'last_10_games' column (1 for win, 0 for loss)
    games['game_result'] = games['team_winner'].astype(int)

    # Calculate momentum (10-game moving average of win percentage)
    games['momentum'] = games.groupby('team_location')['game_result'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

    games.drop(columns=['game_result'], inplace=True)
    
    # Drop original numeric columns
    games = games.drop(columns=numeric_columns)

    games.to_csv(f'assets/games/games_{year}.csv')
    
    # Split into home and away only for merging
    home = games[games['team_home_away'] == "home"]
    away = games[games['team_home_away'] == "away"]

    home['year'] = year
    away['year'] = year
    
    # First merge
    home = home.merge(
        sos[['School', 'SOS', 'year']],  
        left_on=['team_location', 'year'],  # Need to match the same number of columns
        right_on=['School', 'year'],       
        how='left'              
    )

    # Drop the duplicate School column
    home = home.drop(columns=['School','year'])

    # Second merge
    away = away.merge(
        sos[['School', 'SOS', 'year']],
        left_on=['team_location', 'year'],  # Match opponent location with School
        right_on=['School', 'year'],
        how='left',
        suffixes=('_home', '_away')
    )

    # Drop the duplicate School column again
    away = away.drop(columns=['School','year'])
    
    # Merge data on game_id
    merged_data = pd.merge(home, away, on='game_id', suffixes=('_home', '_away'))
    
    # Drop redundant columns
    merged_data = merged_data.drop(columns=['team_winner_away', 'team_home_away_home', 'team_home_away_away',
                                            'game_date_away', 'season_type_away'])
    
    # Rename columns
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
    
    merged_data['winner'] = merged_data['target']
    merged_data.drop(columns=['target'], inplace=True)
    
    dataset.append(merged_data)
    print(f"{year} data processed.")

full_data = pd.concat(dataset, axis=0)
full_data.to_csv("cleaned_dataset.csv")
