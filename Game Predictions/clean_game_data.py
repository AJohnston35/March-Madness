import os
import warnings

warnings.filterwarnings("ignore", message="CUDA path could not be detected.*", category=UserWarning)

USE_GPU = os.getenv("USE_GPU", "1") == "1"
GPU_STATUS = "CPU"
cp = None
if USE_GPU:
    try:
        import cupy as _cp

        device_count = _cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            cp = _cp
            GPU_STATUS = f"GPU (CuPy, devices={device_count})"
        else:
            GPU_STATUS = "CPU (no CUDA device found)"
    except Exception as gpu_error:
        GPU_STATUS = f"CPU (GPU unavailable: {gpu_error})"

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
print(f"Execution backend: {GPU_STATUS}")


def is_gpu_enabled():
    return cp is not None


def _as_backend_array(series, dtype=np.float64):
    values = series.to_numpy(dtype=dtype, copy=False)
    if is_gpu_enabled():
        return cp.asarray(values)
    return values


def _to_numpy(values):
    if is_gpu_enabled():
        return cp.asnumpy(values)
    return values


def _safe_ratio(numerator, denominator, scale=1.0, default=0.0):
    if is_gpu_enabled():
        return cp.where(denominator != 0, (numerator / denominator) * scale, default)
    return np.where(denominator != 0, (numerator / denominator) * scale, default)

dataset = []

player_data = pd.read_csv('Game Predictions/team_metrics.csv')
player_data.drop(columns=['game_date'], inplace=True)

# Load conference mapping (replaces KenPom short_conference_name dependency)
conference_mapping = pd.read_csv(
    'Data/kenpom/REF _ NCAAM Conference and ESPN Team Name Mapping.csv',
    usecols=['Conference', 'Mapped ESPN Team Name']
).dropna()
conference_mapping['team_location_key'] = conference_mapping['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()
conference_mapping = conference_mapping.drop_duplicates(subset=['team_location_key'])
conference_mapping = conference_mapping.rename(columns={'Conference': 'short_conference_name'})
conference_mapping = conference_mapping[['team_location_key', 'short_conference_name']]

# Also create mapping for opponent teams
opponent_conference_mapping = conference_mapping.copy()
opponent_conference_mapping = opponent_conference_mapping.rename(
    columns={
        'team_location_key': 'opponent_team_location_key',
        'short_conference_name': 'opponent_short_conference_name'
    }
)

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
    away_wins = _as_backend_array(games_copy['away_wins'], dtype=np.float32)
    away_losses = _as_backend_array(games_copy['away_losses'], dtype=np.float32)
    games_copy['away_win_percentage'] = _to_numpy(
        _safe_ratio(away_wins, (away_wins + away_losses), scale=100.0, default=50.0)
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
    non_conf_wins = _as_backend_array(games_copy['non_conf_wins'], dtype=np.float32)
    non_conf_losses = _as_backend_array(games_copy['non_conf_losses'], dtype=np.float32)
    games_copy['non_conf_win_percentage'] = _to_numpy(
        _safe_ratio(non_conf_wins, (non_conf_wins + non_conf_losses), scale=100.0, default=50.0)
    )
    
    # Clean up intermediate columns
    games_copy = games_copy.drop(columns=[
        'is_away_game', 'away_win', 'away_loss', 
        'prev_conference', 'is_non_conf_game', 'non_conf_win', 'non_conf_loss'
    ])
    
    print(f"Added time-series SOS, away record, and non-conference record for {year}")
    
    return games_copy

def fill_nas(games):
    games['largest_lead'] = games['largest_lead'].fillna(games['points'] - games['opponent_points'])
    games['total_turnovers'] = games['total_turnovers'].fillna(games['turnovers'])
    games['total_rebounds'] = games['total_rebounds'].fillna(games['offensive_rebounds'] + games['defensive_rebounds'])
    # For the rest, fill with 0
    games = games.fillna(0)
    return games


def new_cols_to_calculate(games):
    # Peaking/slumping signal: how recent form compares to longer-term form.
    recent_net = _as_backend_array(games['recent_net_rating'])
    net = _as_backend_array(games['net_rating'])
    games['peaking_slumping'] = _to_numpy(recent_net - net)

    # 3pt variance profile: volume times volatility of made 3s.
    three_pa = _as_backend_array(games['three_point_field_goals_attempted_per_game'])
    three_pm_std = _as_backend_array(games['three_point_field_goals_made_stdev'])
    games['three_pt_variant'] = _to_numpy(three_pa * three_pm_std)

    # Major conferences used for high-major vs mid-major flags.
    major_conferences = {'sec', 'acc', 'big10', 'big12', 'b10', 'b12'}
    conf_norm = (
        games['short_conference_name']
        .astype(str)
        .str.lower()
        .str.replace(r'[^a-z0-9]', '', regex=True)
    )
    is_high_major = conf_norm.isin(major_conferences)
    games['is_mid_major'] = (~is_high_major).astype(int)
    games['is_elite_mid_major'] = (
        (games['is_mid_major'] == 1) & (games['win_loss_percentage'] > 0.800)
    ).astype(int)
    games['is_weak_high_major'] = (
        (games['is_mid_major'] == 0) & (games['win_loss_percentage'] < 0.650)
    ).astype(int)

    return games

for year in range(2003, 2027):
    games = pd.read_csv(f"Data/game_results/games_{year}.csv")

    # Replace Hawaii with Hawai'i, St. Francis (PA) with Saint Francis, San JosÃ© St with San Jose State
    games['team_location'] = games['team_location'].replace({'Hawaii': 'Hawai\'i', 'St. Francis (PA)': 'Saint Francis', 'San JosÃ© St': 'San Jose State'})
    games['opponent_team_location'] = games['opponent_team_location'].replace({'Hawaii': 'Hawai\'i', 'St. Francis (PA)': 'Saint Francis', 'San JosÃ© St': 'San Jose State'})

    # Drop unnecessary columns
    games = games.drop(columns=['season', 'game_date_time', 'team_uid', 'team_slug', 'team_name',
                                'team_abbreviation', 'team_display_name', 'team_short_display_name',
                                'team_alternate_color', 'team_logo', 'opponent_team_id',
                                'opponent_team_name', 'opponent_team_abbreviation',
                                'opponent_team_display_name',
                                'opponent_team_alternate_color', 'opponent_team_logo',
                                'opponent_team_color'])
    
    games = games.rename(columns={"team_score": "points", "opponent_team_score": "opponent_points"})
    
    # Sort by game date
    games = games.sort_values(by=['game_date'])

    # Map conference name from reference file (instead of KenPom short_conference_name)
    games['team_location_key'] = games['team_location'].astype(str).str.strip().str.lower()
    games = games.merge(conference_mapping, on='team_location_key', how='left')
    games = games.drop(columns=['team_location_key'])
    games['short_conference_name'] = games['short_conference_name'].fillna(games['team_location'])
    games['opponent_team_location_key'] = games['opponent_team_location'].astype(str).str.strip().str.lower()
    games = games.merge(opponent_conference_mapping, on='opponent_team_location_key', how='left')
    games = games.drop(columns=['opponent_team_location_key'])
    games['opponent_short_conference_name'] = games['opponent_short_conference_name'].fillna(games['opponent_team_location'])

    points = _as_backend_array(games['points'])
    opponent_points = _as_backend_array(games['opponent_points'])
    field_goals_attempted = _as_backend_array(games['field_goals_attempted'])
    offensive_rebounds = _as_backend_array(games['offensive_rebounds'])
    turnovers = _as_backend_array(games['turnovers'])
    free_throws_attempted = _as_backend_array(games['free_throws_attempted'])
    field_goals_made = _as_backend_array(games['field_goals_made'])
    three_point_field_goals_made = _as_backend_array(games['three_point_field_goals_made'])
    largest_lead = _as_backend_array(games['largest_lead'])

    possessions = field_goals_attempted - offensive_rebounds + turnovers + (free_throws_attempted / 2.0)
    margin_of_victory = points - opponent_points

    games['possessions'] = _to_numpy(possessions).astype(np.float64)
    games['offensive_efficiency'] = _to_numpy(_safe_ratio(points, possessions, scale=100, default=0.0)).astype(np.float64)
    games['defensive_efficiency'] = _to_numpy(_safe_ratio(opponent_points, possessions, scale=100, default=0.0)).astype(np.float64)
    games['net_efficiency'] = _to_numpy(
        _as_backend_array(games['offensive_efficiency']) - _as_backend_array(games['defensive_efficiency'])
    ).astype(np.float64)
    games['efg_percent'] = _to_numpy(
        _safe_ratio(
            ((field_goals_made + 0.5) * three_point_field_goals_made),
            field_goals_attempted,
            scale=100,
            default=0.0,
        )
    ).astype(np.float64)
    games['turnover_rate'] = _to_numpy(_safe_ratio(turnovers, possessions, scale=100, default=0.0)).astype(np.float64)
    games['offensive_rebound_rate'] = _to_numpy(
        _safe_ratio(offensive_rebounds, (field_goals_attempted - field_goals_made), scale=100, default=0.0)
    ).astype(np.float64)
    games['free_throw_rate'] = _to_numpy(
        _safe_ratio(free_throws_attempted, field_goals_attempted, scale=100, default=0.0)
    ).astype(np.float64)
    games['single_digit_win'] = _to_numpy((margin_of_victory < 10).astype(np.int64)).astype(np.int64)
    games['single_digit_loss'] = _to_numpy(((opponent_points - points) < 10).astype(np.int64)).astype(np.int64)
    games['blowout_win'] = _to_numpy((margin_of_victory > 19).astype(np.int64)).astype(np.int64)
    games['blowout_loss'] = _to_numpy(((opponent_points - points) > 19).astype(np.int64)).astype(np.int64)
    games['clutch_win'] = _to_numpy((margin_of_victory <= 3).astype(np.int64)).astype(np.int64)
    games['clutch_loss'] = _to_numpy(((opponent_points - points) <= 3).astype(np.int64)).astype(np.int64)
    if is_gpu_enabled():
        games['competitive_game'] = _to_numpy((cp.abs(margin_of_victory) < 6).astype(np.int64)).astype(np.int64)
    else:
        games['competitive_game'] = _to_numpy((np.abs(margin_of_victory) < 6).astype(np.int64)).astype(np.int64)
    games['margin_of_victory'] = _to_numpy(margin_of_victory).astype(np.float64)
    games['blown_lead'] = _to_numpy(largest_lead - margin_of_victory).astype(np.float64)

    games['single_digit_win'] = games.groupby('team_location')['single_digit_win'].shift(1)
    games['single_digit_loss'] = games.groupby('team_location')['single_digit_loss'].shift(1)
    games['blowout_win'] = games.groupby('team_location')['blowout_win'].shift(1)
    games['blowout_loss'] = games.groupby('team_location')['blowout_loss'].shift(1)
    games['clutch_win'] = games.groupby('team_location')['clutch_win'].shift(1)
    games['clutch_loss'] = games.groupby('team_location')['clutch_loss'].shift(1)
    games['competitive_game'] = games.groupby('team_location')['competitive_game'].shift(1)
    games['margin_of_victory'] = games.groupby('team_location')['margin_of_victory'].shift(1)
    games['blown_lead'] = games.groupby('team_location')['blown_lead'].shift(1)

    games = fill_nas(games)

    # Net rating features (leakage-safe: based only on prior games)
    games['net_rating'] = games.groupby('team_location')['net_efficiency'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    games['recent_net_rating'] = games.groupby('team_location')['net_efficiency'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    games['ranking'] = games.groupby('game_date')['net_rating'].rank(ascending=False, method='dense')
    games['recent_ranking'] = games.groupby('game_date')['recent_net_rating'].rank(ascending=False, method='dense')

    # Identify numeric columns for cumulative averages (excluding certain non-statistical columns)
    numeric_columns = [col for col in games.columns if games[col].dtype in ['int64', 'float64']
                       and col not in ['team_id', 'game_id', 'season_type', 'team_winner',
                                       'net_rating', 'recent_net_rating', 'ranking', 'recent_ranking']]

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

    # Conference-level *non-conference* win/loss calculations with shifting
    # Determine which games are non-conference (team's conference != opponent's conference)
    games['is_non_conference_game'] = games['short_conference_name'] != games['opponent_short_conference_name']

    # Shift win value to look at past games only
    # Ensure that 'team_winner_shifted' is boolean before applying ~ (bitwise NOT)
    team_winner_shifted_bool = games['team_winner_shifted'].fillna(False).astype(bool)
    games['team_winner_nonconf_shifted'] = team_winner_shifted_bool & games['is_non_conference_game']
    games['team_loss_nonconf_shifted'] = (~team_winner_shifted_bool) & games['is_non_conference_game']

    # Calculate cumulative non-conference wins and losses for each conference over time (using game's short_conference_name)
    games['non_conference_wins'] = games.groupby('short_conference_name')['team_winner_nonconf_shifted'].transform(lambda x: x.cumsum())
    games['non_conference_losses'] = games.groupby('short_conference_name')['team_loss_nonconf_shifted'].transform(lambda x: x.cumsum())
    non_conf_wins_arr = _as_backend_array(games['non_conference_wins'])
    non_conf_losses_arr = _as_backend_array(games['non_conference_losses'])
    games['non_conference_win_loss_percentage'] = _to_numpy(
        _safe_ratio(
            non_conf_wins_arr,
            (non_conf_wins_arr + non_conf_losses_arr),
            scale=1.0,
            default=0.0,
        )
    )

    games.drop(columns=[
        'non_conference_wins', 
        'non_conference_losses', 
        'team_winner_nonconf_shifted', 
        'team_loss_nonconf_shifted'
    ], inplace=True)

    # Create conference rankings based on non-conference win percentage
    nc_conference_rankings = (
        games.groupby(['game_date', 'short_conference_name'])
        .agg({'non_conference_win_loss_percentage': 'last'})
        .reset_index()
    )

    nc_conference_rankings['conference_ranking'] = nc_conference_rankings.groupby('game_date')['non_conference_win_loss_percentage'].rank(
        ascending=False, method='dense'
    )

    # Merge rankings back into games data
    games = games.merge(nc_conference_rankings[['game_date', 'short_conference_name', 'conference_ranking']], 
                        on=['game_date', 'short_conference_name'], 
                        how='left')

    # Win/loss percentage (using shifted values)
    games['win_loss_percentage'] = games.groupby('team_location')['team_winner_shifted'].transform(lambda x: x.expanding().mean())

    games = new_cols_to_calculate(games)

    # Merge opponent win/loss percentage (using game_id)
    opponent_stats = games[['game_id', 'team_location', 'win_loss_percentage']].copy()
    opponent_stats.rename(columns={'team_location': 'opponent_team_location',
                                   'win_loss_percentage': 'opponent_win_loss_percentage'}, inplace=True)

    games = games.merge(opponent_stats, on=['game_id', 'opponent_team_location'], how='left')

    games = calculate_additional_metrics_fixed(games, year)
    games = games.drop(columns=['short_conference_name'])

    # Calculate past 10-game and 15-game win-loss percentages for opponents (shifted)
    games['opponent_recent_win_loss_5'] = games.groupby('opponent_team_location')['opponent_win_loss_percentage']\
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    # Calculate past 10-game and 15-game win-loss percentages for opponents (shifted)
    games['opponent_recent_win_loss_10'] = games.groupby('opponent_team_location')['opponent_win_loss_percentage']\
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())

    games['opponent_recent_win_loss_15'] = games.groupby('opponent_team_location')['opponent_win_loss_percentage']\
        .transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())

    # Fill missing values (assumption: 0.5 for neutral)
    games[['opponent_win_loss_percentage', 'opponent_recent_win_loss_5', 'opponent_recent_win_loss_10', 'opponent_recent_win_loss_15']] = \
        games[['opponent_win_loss_percentage', 'opponent_recent_win_loss_5','opponent_recent_win_loss_10', 'opponent_recent_win_loss_15']].fillna(50)

    games.drop(columns=['opponent_team_location'], inplace=True)

    # Calculate momentum (past 10 and 15-game moving averages of wins)
    games['momentum_5'] = games.groupby('team_location')['team_winner_shifted']\
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Calculate momentum (past 10 and 15-game moving averages of wins)
    games['momentum_10'] = games.groupby('team_location')['team_winner_shifted']\
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())

    games['momentum_15'] = games.groupby('team_location')['team_winner_shifted']\
        .transform(lambda x: x.rolling(window=15, min_periods=1).mean())
    
    opp_5 = _as_backend_array(games['opponent_recent_win_loss_5'])
    opp_10 = _as_backend_array(games['opponent_recent_win_loss_10'])
    opp_15 = _as_backend_array(games['opponent_recent_win_loss_15'])
    mom_5 = _as_backend_array(games['momentum_5'])
    mom_10 = _as_backend_array(games['momentum_10'])
    mom_15 = _as_backend_array(games['momentum_15'])
    conf_rank = _as_backend_array(games['conference_ranking'])

    games['weighted_momentum_5'] = _to_numpy(_safe_ratio((opp_5 * mom_5), conf_rank, scale=1.0, default=0.0))
    games['weighted_momentum_10'] = _to_numpy(_safe_ratio((opp_10 * mom_10), conf_rank, scale=1.0, default=0.0))
    games['weighted_momentum_15'] = _to_numpy(_safe_ratio((opp_15 * mom_15), conf_rank, scale=1.0, default=0.0))

    games.drop(columns=['team_winner_shifted'], inplace=True)

    # Drop original numeric columns (since their per-game versions are used)
    games = games.drop(columns=numeric_columns)

    games.to_csv(f'Game Predictions/assets/games/games_{year}.csv')

    # Split into home and away
    home = games[games['team_home_away'] == "home"]

    away = games[games['team_home_away'] == "away"]

    home['year'] = year
    away['year'] = year

    #home = home.merge(kenpom, on=['team_location', 'year'], how='left')
    home = home.drop(columns=['year'])

    for column in home.columns:
        if home[column].dtype not in ['int64','float64']:
            try:
                home[column] = home[column].astype(int)
            except:
                print(f"Could not convert column {column} to integer")


    #away = away.merge(kenpom, on=['team_location', 'year'], how='left')
    away = away.drop(columns=['year', 'season_type'])

    for column in away.columns:
        if away[column].dtype not in ['int64','float64']:
            try:
                away[column] = away[column].astype(int)
            except:
                print(f"Could not convert column {column} to integer")

    merged_data = pd.merge(home, away, on='game_id', suffixes=('_home', '_away'))

    # No seed adjustment section; all references to seeds have been removed.

    merged_data = merged_data.drop(
        columns=[
            'team_winner_away',
            'team_home_away_home',
            'team_home_away_away',
            'game_date_away',
            'short_conference_name_home',
            'short_conference_name_away'
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
                if pd.api.types.is_numeric_dtype(merged_data[column]) and pd.api.types.is_numeric_dtype(merged_data[away_column]):
                    home_vals = _as_backend_array(merged_data[column], dtype=np.float64)
                    away_vals = _as_backend_array(merged_data[away_column], dtype=np.float64)
                    merged_data[base_column + '_diff'] = _to_numpy(home_vals - away_vals)
                    merged_data.drop([column, away_column], axis=1, inplace=True)

    dataset.append(merged_data)

    print(f"{year} processed.")

full_data = pd.concat(dataset, axis=0)
# The following lines are commented out entirely, seed references removed.
# seed_records = pd.read_csv('data/seed_records_fixed.csv')
# full_data = full_data.merge(
#     seed_records,
#     on=['seed_diff'],
#     how='left'
# )
full_data.to_csv("Game Predictions/cleaned_dataset.csv")
