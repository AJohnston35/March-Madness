import pandas as pd
import os
from pathlib import Path

def fix_team_names(df):
    df['school'] = df['school'].str.replace('Connecticut', 'UConn', regex=False)
    df['school'] = df['school'].str.replace('North Carolina', 'UNC', regex=False)
    df['school'] = df['school'].str.replace("Saint Mary's (CA)", "Saint Mary's", regex=False)
    df['school'] = df['school'].str.replace('Pittsburgh', 'Pitt', regex=False)
    df['school'] = df['school'].str.replace('Southern California', 'USC', regex=False)
    df['school'] = df['school'].str.replace("Saint Joseph's", "St. Joseph's", regex=False)
    df['school'] = df['school'].str.replace('Southern Methodist', 'SMU', regex=False)
    df['school'] = df['school'].str.replace('Virginia Commonwealth', 'VCU', regex=False)
    df['school'] = df['school'].str.replace("Saint Peter's", "St. Peter's", regex=False)
    df['school'] = df['school'].str.replace('Louisiana State', 'LSU', regex=False)
    df['school'] = df['school'].str.replace('Mississippi', 'Ole Miss', regex=False)
    df['school'] = df['school'].str.replace('Pennsylvania', 'Penn', regex=False)
    df['school'] = df['school'].str.replace('North Carolina State', 'NC State', regex=False)
    df['school'] = df['school'].str.replace('UC Irvine', 'UC-Irvine', regex=False)
    df['school'] = df['school'].str.replace('UC Santa Barbara', 'UCSB', regex=False)
    df['school'] = df['school'].str.replace('Maryland-Baltimore County', 'UMBC', regex=False)
    df['school'] = df['school'].str.replace('UC Davis', 'UC-Davis', regex=False)
    df['school'] = df['school'].str.replace('UC Riverside', 'UC-Riverside', regex=False)
    df['school'] = df['school'].str.replace('East Tennessee State', 'ETSU', regex=False)
    df['school'] = df['school'].str.replace('Massachusetts', 'UMass', regex=False)
    df['school'] = df['school'].str.replace('North Carolina State', 'NC State', regex=False)
    df['school'] = df['school'].str.replace('Nevada-Las Vegas', 'UNLV', regex=False)
    df['school'] = df['school'].str.replace('Southern Mississippi', 'Southern Miss', regex=False)
    df['school'] = df['school'].str.replace('Ole Miss State', 'Mississippi State', regex=False)
    df['school'] = df['school'].str.replace('Brigham Young', 'BYU', regex=False)
    df['school'] = df['school'].str.replace('UNC Central', 'North Carolina Central', regex=False)
    df['school'] = df['school'].str.replace('UNC A&T', 'North Carolina A&T', regex=False)
    df['school'] = df['school'].str.replace('Southern Ole Miss', 'Southern Miss', regex=False)
    df['school'] = df['school'].str.replace('Long Island University', 'LIU', regex=False)
    df['school'] = df['school'].str.replace('Central UConn', 'Central Connecticut', regex=False)
    
    return df

def get_data(year):
    ratings_file = Path("team_ratings/old") / f"ratings_{year}.csv"
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
        basic_file = Path("school_stats/old") / f"basic_{year}.csv"
        adv_file = Path("school_stats/old") / f"adv_{year}.csv"
        
        def load_and_process_data(file_path, year):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            df['school'] = df['school'].str.replace("NCAA", "").str.strip()
            df = df.loc[:, ~df.columns.str.startswith('unnamed')]
            df['year'] = year
            return df

        basic_df = load_and_process_data(basic_file, year)
        adv_df = load_and_process_data(adv_file, year)

        basic_df = fix_team_names(basic_df)
        adv_df = fix_team_names(adv_df)

    # Only load _opp files if year is 2010 or later
    basic_opp_df = pd.DataFrame()  # Empty DataFrame if not loading
    adv_opp_df = pd.DataFrame()  # Empty DataFrame if not loading
    if year >= 2010:
        basic_opp_file = Path("school_stats/old") / f"basic_opp_{year}.csv"
        adv_opp_file = Path("school_stats/old") / f"adv_opp_{year}.csv"
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
    # Give team2 columns, besides year, a _team2 suffix
    team2_df = team2_df.rename(columns=lambda x: x + '_team2' if x != 'year' else x)
    
    # Merge the dataframes, and give team2 df a _team2 suffix
    merged_df = team1_df.merge(team2_df, on=['year'], how='left')
    
    merged_df['round'] = round
    
    round_mapping = {
        'First Round': 0,
        'Second Round': 1,
        'Sweet Sixteen': 2,
        'Elite Eight': 3,
        'Final Four': 4,
        'Championship': 5
    }

    merged_df['round'] = merged_df['round'].map(round_mapping)
    
    for column in merged_df.columns:
        if '%' in column:
            merged_df[column] = merged_df[column] * 100
        if merged_df[column].dtype in ['int64', 'float64']:
            if column.endswith('_team2'):
                base_column = column[:-6]
                if base_column in merged_df.columns:
                    diff_column = f"{base_column}_diff"
                    merged_df[diff_column] = merged_df[column] - merged_df[base_column]
                    merged_df = merged_df.drop(columns=[column, base_column])
                    
    merged_df = merged_df.drop(columns=['school','conf','school_team2','conf_team2','rk_diff'])
                
    return merged_df