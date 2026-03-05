import pandas as pd
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def get_data(team, year):
    df = pd.read_csv(f'Game Predictions/assets/games/games_{year}.csv')

    df = df.sort_values(by=['game_date'], ascending=False)

    df = df[df['team_location'] == team]

    df = df.iloc[[0]]

    return df

def merge_data(home, away, home_year, away_year):
    home = home.copy()
    away = away.copy()

    # Keep season_type from home side and avoid duplicate season_type columns.
    away = away.drop(columns=['season_type'], errors='ignore')

    # Build one matchup row even when years differ (cross join on a constant key).
    home['_merge_key'] = 1
    away['_merge_key'] = 1
    merged_data = pd.merge(home, away, on='_merge_key', suffixes=('_home', '_away')).drop(columns=['_merge_key'])

    seed1 = float(merged_data['seed_home'].iloc[0]) if 'seed_home' in merged_data.columns else 0.0
    implied_probability = 0.5

    merged_data = merged_data.drop(
        columns=[
            'team_winner_away',
            'team_home_away_home',
            'team_home_away_away',
            'game_date_away',
            'short_conference_name_home',
            'short_conference_name_away',
        ],
        errors='ignore'
    )

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
            away_column = f'{base_column}_away'
            if away_column in merged_data.columns:
                if (
                    pd.api.types.is_numeric_dtype(merged_data[column])
                    and pd.api.types.is_numeric_dtype(merged_data[away_column])
                ):
                    home_vals = merged_data[column]
                    away_vals = merged_data[away_column]

                    # NumPy does not support bool subtraction directly; match training behavior by casting to ints.
                    if pd.api.types.is_bool_dtype(home_vals) or pd.api.types.is_bool_dtype(away_vals):
                        home_vals = home_vals.astype(np.int8)
                        away_vals = away_vals.astype(np.int8)

                    merged_data[base_column + '_diff'] = home_vals - away_vals
                    merged_data.drop([column, away_column], axis=1, inplace=True)
    
    return merged_data, implied_probability, seed1
