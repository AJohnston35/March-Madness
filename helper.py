import pandas as pd
import os
from pathlib import Path
import numpy as np

sos = pd.read_csv('assets/all_ratings.csv')

def fix_team_names(df):
    try: 
        #df['school'] = df['school'].str.replace('Connecticut', 'UConn', regex=False)
        #df['school'] = df['school'].str.replace('North Carolina', 'UNC', regex=False)
        #df['school'] = df['school'].str.replace("Saint Mary's (CA)", "Saint Mary's", regex=False)
        #df['school'] = df['school'].str.replace('Pittsburgh', 'Pitt', regex=False)
        #df['school'] = df['school'].str.replace('Southern California', 'USC', regex=False)
        #df['school'] = df['school'].str.replace("Saint Joseph's", "St. Joseph's", regex=False)
        #df['school'] = df['school'].str.replace('Southern Methodist', 'SMU', regex=False)
        #df['school'] = df['school'].str.replace('Virginia Commonwealth', 'VCU', regex=False)
        #df['school'] = df['school'].str.replace("Saint Peter's", "St. Peter's", regex=False)
        #df['school'] = df['school'].str.replace('Louisiana State', 'LSU', regex=False)
        #df['school'] = df['school'].str.replace('Mississippi', 'Ole Miss', regex=False)
        #df['school'] = df['school'].str.replace('Pennsylvania', 'Penn', regex=False)
        df['school'] = df['school'].str.replace('North Carolina State', 'NC State', regex=False)
        #df['school'] = df['school'].str.replace('UC Irvine', 'UC-Irvine', regex=False)
        #df['school'] = df['school'].str.replace('UC Santa Barbara', 'UCSB', regex=False)
        #df['school'] = df['school'].str.replace('Maryland-Baltimore County', 'UMBC', regex=False)
        #df['school'] = df['school'].str.replace('UC Davis', 'UC-Davis', regex=False)
        #df['school'] = df['school'].str.replace('UC Riverside', 'UC-Riverside', regex=False)
        df['school'] = df['school'].str.replace('East Tennessee State', 'ETSU', regex=False)
        #df['school'] = df['school'].str.replace('Massachusetts', 'UMass', regex=False)
        df['school'] = df['school'].str.replace('North Carolina State', 'NC State', regex=False)
        #df['school'] = df['school'].str.replace('Nevada-Las Vegas', 'UNLV', regex=False)
        df['school'] = df['school'].str.replace('Southern Mississippi', 'Southern Miss', regex=False)
        #df['school'] = df['school'].str.replace('Ole Miss State', 'Mississippi State', regex=False)
        #df['school'] = df['school'].str.replace('Brigham Young', 'BYU', regex=False)
        #df['school'] = df['school'].str.replace('UNC Central', 'North Carolina Central', regex=False)
        #df['school'] = df['school'].str.replace('UNC A&T', 'North Carolina A&T', regex=False)
        #df['school'] = df['school'].str.replace('Southern Ole Miss', 'Southern Miss', regex=False)
        #df['school'] = df['school'].str.replace('Long Island University', 'LIU', regex=False)
        #df['school'] = df['school'].str.replace('Central UConn', 'Central Connecticut', regex=False)
    except: 
        print('Error changing school name')
    
    return df

def get_data(year):
    ratings_file = Path("data/team_ratings/old") / f"ratings_{year}.csv"
    # Initialize empty DataFrames
    basic_df = pd.DataFrame()  
    adv_df = pd.DataFrame()  
    
    # Read in the data
    ratings_df = pd.read_csv(ratings_file)
    # Convert columns to lower
    ratings_df.columns = ratings_df.columns.str.lower()
    ratings_df['year'] = year
    
    ratings_df = fix_team_names(ratings_df)

    # Drop unnamed columns
    unnamed_cols = [col for col in ratings_df.columns if col.startswith('unnamed')]
    ratings_df = ratings_df.drop(columns=unnamed_cols)
    
    if year > 1992:
        basic_file = Path("data/school_stats/old") / f"basic_{year}.csv"
        adv_file = Path("data/school_stats/old") / f"adv_{year}.csv"
        
        def load_and_process_data(file_path, year):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            df['school'] = df['school'].astype(str)
            df['school'] = df['school'].str.replace("NCAA", "").str.strip()
            df = df.loc[:, ~df.columns.str.startswith('unnamed')]
            df['year'] = year
            return df
        try:
            basic_df = load_and_process_data(basic_file, year)
            adv_df = load_and_process_data(adv_file, year)
        except:
            print('Error loading data')

        basic_df = fix_team_names(basic_df)
        adv_df = fix_team_names(adv_df)

    # Only load _opp files if year is 2010 or later
    basic_opp_df = pd.DataFrame()  # Empty DataFrame if not loading
    adv_opp_df = pd.DataFrame()  # Empty DataFrame if not loading
    if year >= 2010:
        basic_opp_file = Path("data/school_stats/old") / f"basic_opp_{year}.csv"
        adv_opp_file = Path("data/school_stats/old") / f"adv_opp_{year}.csv"
        basic_opp_df = load_and_process_data(basic_opp_file, year)
        adv_opp_df = load_and_process_data(adv_opp_file, year)
        
        basic_opp_df = fix_team_names(basic_opp_df)
        adv_opp_df = fix_team_names(adv_opp_df)

    if not basic_df.empty:
        merged_df = ratings_df.merge(basic_df, on=['school'], how='left')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
        merged_df = merged_df.rename(columns=lambda x: x.rstrip('_x'))
    else:
        merged_df = ratings_df
    
    if not basic_opp_df.empty:
        merged_df = merged_df.merge(basic_opp_df, on=['school'], how='left')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
        merged_df = merged_df.rename(columns=lambda x: x.rstrip('_x'))
    
    if not adv_df.empty:
        merged_df = merged_df.merge(adv_df, on=['school'], how='left')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
        merged_df = merged_df.rename(columns=lambda x: x.rstrip('_x'))
    
    if not adv_opp_df.empty:
        merged_df = merged_df.merge(adv_opp_df, on=['school'], how='left')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
        merged_df = merged_df.rename(columns=lambda x: x.rstrip('_x'))
    
    return merged_df

def preprocess_data(team1_df, team2_df, round):
    team1_df = team1_df.drop(columns=['g','w'], errors='ignore')
    team2_df = team2_df.drop(columns=['g','w'], errors='ignore')
    # Give team2 columns, besides year, a _team2 suffix
    team2_df = team2_df.rename(columns=lambda x: x + '_team2' if x != 'year' else x)
   
    # Merge the dataframes, and give team2 df a _team2 suffix
    merged_df = team1_df.merge(team2_df, on=['year'], how='left')
    
    # Drop year_y and remove _x from the other year column
    merged_df = merged_df.drop(columns=['year'])
    
    # Map the round
    round_mapping = {
        'First Round': 0,
        'Second Round': 1,
        'Sweet Sixteen': 2,
        'Elite Eight': 3,
        'Final Four': 4,
        'Championship': 5
    }
    mapped_round = round_mapping.get(round, -1)  # Default to -1 if round is not found
    merged_df['round'] = mapped_round
    
    # Fix columns that need type conversion
    wrong_columns = ['pace','ftr','3par','ts%','trb%','ast%','stl%','blk%','efg%','tov%','orb%','ft/fga']
    for column in wrong_columns:
        column_plus = column + '_team2'
        merged_df[column] = merged_df[column].astype(float)
        merged_df[column_plus] = merged_df[column_plus].astype(float)
   
    # Convert percentage columns
    for column in merged_df.columns:
        if '%' in column:
            merged_df[column] = merged_df[column] * 100
   
    # Collect all columns we need to process for creating features
    feature_cols = []
    for column in merged_df.columns:
        if column.endswith('_team2'):
            base_column = column[:-6]
            if base_column in merged_df.columns and base_column not in ['seed1', 'seed2', 'sos', 'srs']:
                if merged_df[column].dtype in ['int64', 'float64']:
                    feature_cols.append((base_column, column))
    
    # Create all new columns in one go to avoid fragmentation
    new_features = {}
    
    # Important stats that strongly correlate with winning
    key_stats = ['ortg', 'drtg', 'nrtg']
    
    for base_column, column in feature_cols:
        # Original directional difference - ALWAYS include this
        new_features[f"{base_column}_diff"] = merged_df[column] - merged_df[base_column]
        
        # Only add position-agnostic features for specific important stats
        # This helps maintain model confidence while reducing bias
        if any(stat in base_column.lower() for stat in key_stats):
            new_features[f"{base_column}_abs_diff"] = abs(merged_df[column] - merged_df[base_column])
            
            # For truly key stats, add even more position-agnostic features
            if base_column in ['ortg', 'drtg', 'w', 'l']:
                new_features[f"{base_column}_min"] = merged_df[[column, base_column]].min(axis=1)
                new_features[f"{base_column}_max"] = merged_df[[column, base_column]].max(axis=1)
    
    # Add all new features at once
    new_df = pd.DataFrame(new_features, index=merged_df.index)
    merged_df = pd.concat([merged_df, new_df], axis=1)
    
    # Drop the original columns that were used to create differences
    cols_to_drop = []
    for base_column, column in feature_cols:
        cols_to_drop.extend([base_column, column])
    
    merged_df = merged_df.drop(columns=cols_to_drop)
    
    # Drop other columns that aren't needed
    try:
        merged_df = merged_df.drop(columns=['school','conf','school_team2','conf_team2','rk_diff','g_diff'])
    except:
        try:
            merged_df = merged_df.drop(columns=['school','conf','school_team2','conf_team2','rk_diff'])
        except:
            pass
    
    try:
        merged_df = merged_df.drop(columns=['ap rank_diff'])
    except:
        pass
    
    # Add a feature to enhance the model's confidence in lopsided matchups
    # This helps with the Duke vs bad team scenario
    if 'w_diff' in merged_df.columns and 'l_diff' in merged_df.columns:
        # Win percentage difference (emphasizes team strength differences)
        team1_wp = merged_df['w_diff'] / (merged_df['w_diff'] + merged_df['l_diff'])
        team2_wp = -merged_df['l_diff'] / (-merged_df['w_diff'] + -merged_df['l_diff'])
        merged_df['win_pct_diff'] = team1_wp - team2_wp
        
        # Squared win percentage difference to emphasize large gaps
        merged_df['win_pct_diff_squared'] = merged_df['win_pct_diff'] ** 2 * np.sign(merged_df['win_pct_diff'])
    
    # Add strength indicator for large differences in key metrics
    if 'ortg_diff' in merged_df.columns and 'drtg_diff' in merged_df.columns:
        # Net rating with emphasis on large differences
        merged_df['extreme_rating_diff'] = merged_df['ortg_diff'] - merged_df['drtg_diff']
        # Apply a non-linear transformation to emphasize large differences
        merged_df['extreme_rating_indicator'] = np.tanh(merged_df['extreme_rating_diff'] / 10) * 10
    
    return merged_df

def get_data(team, year):
    df = pd.read_csv(f'assets/games/games_{year}.csv')

    df = df.sort_values(by=['game_date'], ascending=False)

    df = df[df['team_location'] == team]

    df = df.iloc[[0]]

    return df

def merge_data(home, away, home_year, away_year):

    home['year'] = home_year
    away['year'] = away_year
    
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

    merged_data = pd.merge(home, away, on='index', suffixes=('_home', '_away'))

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
    
    return merged_data
