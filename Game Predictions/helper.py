import pandas as pd
import os
from pathlib import Path
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


sos = pd.read_csv('Game Predictions/assets/all_ratings.csv')
kenpom = pd.read_csv('Data/kenpom/raw_data.csv')
seeds = pd.read_csv('Data/seed_records_new.csv')
seed_records = pd.read_csv('Data/seed_records_fixed.csv')
# Convert all column names to lowercase and replace spaces with underscores
kenpom.columns = [col.lower().replace(' ', '_') for col in kenpom.columns]

# If you need to replace other characters like hyphens or periods as well:
kenpom.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('.', '_') 
                  for col in kenpom.columns]

def get_win_probability(seed, opponent_seed, seed_df):
    """
    Retrieve win probability for a given matchup based on seeds.
    
    Parameters:
    - seed: The seed of the team
    - opponent_seed: The seed of the opponent team
    - seed_df: DataFrame containing seed matchup statistics
    
    Returns:
    - Win probability as a float between 0 and 1
    """
    print(seed, opponent_seed)
    # Find the row that matches the seed matchup
    matchup = seed_df[(seed_df['Seed'] == opponent_seed) & (seed_df['Opponent_Seed'] == seed)]
    
    # If the matchup exists, return the win percentage
    if not matchup.empty:
        return matchup['Win_Percentage'].values[0]
    
    # If the reverse matchup exists, return 1 minus the win percentage
    reverse_matchup = seed_df[(seed_df['Seed'] == opponent_seed) & (seed_df['Opponent_Seed'] == seed)]
    if not reverse_matchup.empty:
        return 1 - reverse_matchup['Win_Percentage'].values[0]
    
    # If no matchup found, return 0.5 (even odds) or another default
    return 0.5

def get_data(team, year):
    df = pd.read_csv(f'Game Predictions/assets/games/games_{year}.csv')

    df = df.sort_values(by=['game_date'], ascending=False)

    df = df[df['team_location'] == team]

    df = df.iloc[[0]]

    return df

def merge_data(home, away, home_year, away_year):
    home['year'] = home_year
    away['year'] = away_year

    # Merge SOS ratings for home teams
    home = home.merge(sos[['School', 'SOS', 'year']], left_on=['team_location', 'year'], right_on=['School', 'year'], how='left')
    home['momentum_strength_5'] = home['SOS'] * home['weighted_momentum_5']
    home['momentum_strength_10'] = home['SOS'] * home['weighted_momentum_10']
    home['momentum_strength_15'] = home['SOS'] * home['weighted_momentum_15']
    home['win_strength'] = home['SOS'] * home['win_loss_percentage']
    home = home.drop(columns=['School'])

    home = home.merge(kenpom, on=['team_location', 'year'], how='left')

    for column in home.columns:
        if home[column].dtype not in ['int64','float64']:
            try:
                home[column] = home[column].astype(int)
            except:
                pass

    # Convert non-numeric columns
    for column in home.columns:
        if home[column].dtype not in ['int64', 'float64']:
            try:
                home[column] = home[column].astype(int)
            except:
                pass

    # Merge opponent data for away teams
    away = away.merge(sos[['School', 'SOS', 'year']], left_on=['team_location', 'year'], right_on=['School', 'year'], how='left')
    away = away.drop(columns=['School'])
    away['momentum_strength_5'] = away['SOS'] * away['weighted_momentum_5']
    away['momentum_strength_10'] = away['SOS'] * away['weighted_momentum_10']
    away['momentum_strength_15'] = away['SOS'] * away['weighted_momentum_15']
    away['win_strength'] = away['SOS'] * away['win_loss_percentage']

    away = away.merge(kenpom, on=['team_location', 'year'], how='left')
    away = away.drop(columns=['season_type'])

    for column in away.columns:
        if away[column].dtype not in ['int64','float64']:
            try:
                away[column] = away[column].astype(int)
            except:
                pass

    merged_data = pd.merge(home, away, on='year', suffixes=('_home', '_away'))

    merged_data['seed_home'] = merged_data.apply(lambda x: 0 if x['season_type'] == 2 else x['seed_home'], axis=1)
    merged_data['seed_away'] = merged_data.apply(lambda x: 0 if x['season_type'] == 2 else x['seed_away'], axis=1)
    
    seed1 = merged_data['seed_home'].iloc[0]

    implied_probability = get_win_probability(merged_data['seed_home'].iloc[0], merged_data['seed_away'].iloc[0], seeds)

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

    merged_data = merged_data.merge(
        seed_records,
        on=['seed_diff'],
        how='left'
    )
    
    return merged_data, implied_probability, seed1
