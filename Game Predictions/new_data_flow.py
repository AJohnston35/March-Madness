# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

GAME_RESULTS_DIR = Path('Data/game_results')
CONFERENCE_MAP_PATH = Path('Data/kenpom/REF _ NCAAM Conference and ESPN Team Name Mapping.csv')
REQUIRED_BASE_COLUMNS = [
    'team_score',
    'opponent_team_score',
    'field_goals_made',
    'field_goals_attempted',
    'three_point_field_goals_made',
    'three_point_field_goals_attempted',
    'free_throws_made',
    'free_throws_attempted',
    'offensive_rebounds',
    'defensive_rebounds',
    'assists',
    'steals',
    'blocks',
    'team_turnovers',
    'turnovers',
    'fast_break_points',
    'turnover_points',
    'largest_lead',
    'field_goal_pct',
    'three_point_field_goal_pct',
    'free_throw_pct',
    'flagrant_fouls',
    'fouls',
    'lead_changes',
    'lead_percentage',
    'points_in_paint',
    'technical_fouls',
    'total_rebounds',
    'total_technical_fouls',
    'total_turnovers'
]

def get_expected_score(rating, opp_rating):
    exp = (opp_rating - rating) / 400
    return 1 / (1 + 10**exp)

def get_new_elos(home_rating, away_rating, margin):
    k = 25

    # score of 0.5 for a tie
    home_score = 0.5
    if margin > 0:
        # score of 1 for a win
        home_score = 1
    elif margin < 0:
        #score of 0 for a loss
        home_score = 0

    # get expected home score
    expected_home_score = get_expected_score(home_rating, away_rating)
    # multiply difference of actual and expected score by k value and adjust home rating
    new_home_score = home_rating + k * (home_score - expected_home_score)

    # repeat these steps for the away team
    # away score is inverse of home score
    away_score = 1 - home_score
    expected_away_score = get_expected_score(away_rating, home_rating)
    new_away_score = away_rating + k * (away_score - expected_away_score)

    # return a tuple
    return (round(new_home_score), round(new_away_score))


def normalize_elo(rating, base=1500, factor=0.3):
    if rating < base:
        return rating + (base - rating) * factor
    if rating > base:
        return rating - (rating - base) * factor
    return rating


def add_elo_ratings(df_merged, elo_ratings, elo_last_season):
    df_merged['elo'] = np.nan
    ordered = df_merged.sort_values(['season', 'game_date', 'game_id'])

    for _, game_rows in ordered.groupby('game_id', sort=False):
        game_rows = game_rows.drop_duplicates(subset='team_id')
        if len(game_rows) != 2:
            continue

        idx1, idx2 = game_rows.index[0], game_rows.index[1]
        season = game_rows['season'].iloc[0]
        team1 = df_merged.at[idx1, 'team_id']
        team2 = df_merged.at[idx2, 'team_id']

        def get_team_rating(team_id):
            if team_id not in elo_ratings:
                rating = 1500
            else:
                rating = elo_ratings[team_id]
                last_season = elo_last_season.get(team_id)
                if last_season is not None and last_season != season:
                    rating = normalize_elo(rating)
            elo_ratings[team_id] = rating
            elo_last_season[team_id] = season
            return rating

        rating1 = get_team_rating(team1)
        rating2 = get_team_rating(team2)

        df_merged.at[idx1, 'elo'] = rating1
        df_merged.at[idx2, 'elo'] = rating2

        home_idx = idx1
        if 'team_home_away' in df_merged.columns:
            if df_merged.at[idx1, 'team_home_away'] == 1:
                home_idx = idx1
            elif df_merged.at[idx2, 'team_home_away'] == 1:
                home_idx = idx2

        away_idx = idx2 if home_idx == idx1 else idx1

        home_team = df_merged.at[home_idx, 'team_id']
        away_team = df_merged.at[away_idx, 'team_id']
        home_rating = df_merged.at[home_idx, 'elo']
        away_rating = df_merged.at[away_idx, 'elo']
        margin = df_merged.at[home_idx, 'team_score'] - df_merged.at[away_idx, 'team_score']

        new_home_rating, new_away_rating = get_new_elos(home_rating, away_rating, margin)
        elo_ratings[home_team] = new_home_rating
        elo_ratings[away_team] = new_away_rating

    return df_merged


def process_games_file(
    games_path: Path,
    conference_mapping: pd.DataFrame,
    elo_ratings=None,
    elo_last_season=None
) -> pd.DataFrame:
    if elo_ratings is None:
        elo_ratings = {}
    if elo_last_season is None:
        elo_last_season = {}

    df = pd.read_csv(games_path)

    for col in REQUIRED_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df['team_location'] = df['team_location'].replace(
        {'Hawaii': "Hawai'i", 'St. Francis (PA)': 'Saint Francis', 'San JosÃ© St': 'San Jose State'}
    )
    df['team_location_key'] = df['team_location'].astype(str).str.strip().str.lower()
    df = df.merge(conference_mapping, on='team_location_key', how='left')
    df.drop(columns='team_location_key', inplace=True)

    df = df.dropna(subset=['short_conference_name'])

    df = df.drop(columns=[col for col in df.columns if col.startswith('opponent_') and col != 'opponent_team_score'])

    drop_cols = [
        'game_date_time', 'team_uid', 'team_location', 'team_slug', 'team_name', 'team_abbreviation',
        'team_display_name', 'team_short_display_name', 'team_color', 'team_alternate_color', 'team_logo'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df_merged = df.merge(df, on=['game_id', 'season', 'season_type', 'game_date'], suffixes=(None, '_opponent'))
    df_merged = df_merged[df_merged['team_id'] != df_merged['team_id_opponent']]

    df_merged['poss'] = (
        df_merged['field_goals_attempted'] - df_merged['offensive_rebounds'] + df_merged['team_turnovers']
        + (0.475 * df_merged['free_throws_attempted'])
    )
    df_merged['poss_opponent'] = (
        df_merged['field_goals_attempted_opponent'] - df_merged['offensive_rebounds_opponent']
        + df_merged['team_turnovers_opponent'] + (0.475 * df_merged['free_throws_attempted_opponent'])
    )

    df_merged['off_eff'] = (df_merged['team_score'] / df_merged['poss']) * 100
    df_merged['def_eff'] = (df_merged['team_score_opponent'] / df_merged['poss_opponent']) * 100
    df_merged['net_eff'] = df_merged['off_eff'] - df_merged['def_eff']

    # For each season and date, calculate league-wide average off/def eff for all prior days
    df_merged = df_merged.sort_values(['season', 'game_date', 'game_id'])
    df_merged = df_merged.sort_values(['season', 'game_date'])

    df_merged['league_avg_off_eff'] = (
        df_merged.groupby('season')['off_eff']
        .transform(lambda x: x.shift(2).expanding().mean())
    )

    df_merged['league_avg_def_eff'] = (
        df_merged.groupby('season')['def_eff']
        .transform(lambda x: x.shift(2).expanding().mean())
    )

    df_merged['efg'] = (
        df_merged['field_goals_made'] + (0.5 * df_merged['three_point_field_goals_made'])
    ) / df_merged['field_goals_attempted']
    df_merged['efg_allowed'] = (
        df_merged['field_goals_made_opponent'] + (0.5 * df_merged['three_point_field_goals_made_opponent'])
    ) / df_merged['field_goals_attempted_opponent']
    df_merged['tov'] = df_merged['team_turnovers'] / df_merged['poss']
    df_merged['stl_rate'] = df_merged['steals'] / df_merged['poss_opponent']
    df_merged['orb'] = df_merged['offensive_rebounds'] / (
        df_merged['offensive_rebounds'] + df_merged['defensive_rebounds_opponent']
    )
    df_merged['drb'] = df_merged['defensive_rebounds'] / (
        df_merged['defensive_rebounds'] + df_merged['offensive_rebounds_opponent']
    )
    df_merged['ftr'] = df_merged['free_throws_attempted'] / df_merged['field_goals_attempted']

    df_merged['ppp'] = df_merged['team_score'] / df_merged['poss']
    df_merged['two_pm'] = df_merged['field_goals_made'] - df_merged['three_point_field_goals_made']
    df_merged['two_pa'] = df_merged['field_goals_attempted'] - df_merged['three_point_field_goals_attempted']
    df_merged['two_pct'] = df_merged['two_pm'] / df_merged['two_pa']
    df_merged['three_pct'] = df_merged['three_point_field_goals_made'] / df_merged['three_point_field_goals_attempted']
    df_merged['three_pct_opponent'] = df_merged['three_point_field_goals_made_opponent'] / df_merged['three_point_field_goals_attempted_opponent']
    df_merged['three_attempt_rate'] = df_merged['three_point_field_goals_attempted'] / df_merged['field_goals_attempted']
    df_merged['allowed_three_attempt_rate'] = df_merged['three_point_field_goals_attempted_opponent'] / df_merged['field_goals_attempted_opponent']
    df_merged['three_variance'] = df_merged.groupby('team_id')['three_pct'].transform(lambda x: x.shift(1).rolling(10).std())
    df_merged['score_variance'] = df_merged.groupby('team_id')['team_score'].transform(lambda x: x.shift(1).rolling(10).std())
    df_merged['def_score_variance'] = df_merged.groupby('team_id')['opponent_team_score'].transform(lambda x: x.shift(1).rolling(10).std())
    df_merged['off_eff_variance'] = df_merged.groupby('team_id')['off_eff'].transform(lambda x: x.shift(1).rolling(10).std())
    df_merged['pace_variance'] = df_merged.groupby('team_id')['poss'].transform(lambda x: x.shift(1).rolling(10).std())
    
    df_merged['two_pm_opponent'] = (
        df_merged['field_goals_made_opponent'] - df_merged['three_point_field_goals_made_opponent']
    )
    df_merged['two_pa_opponent'] = (
        df_merged['field_goals_attempted_opponent'] - df_merged['three_point_field_goals_attempted_opponent']
    )
    df_merged['two_pct_opponent'] = df_merged['two_pm_opponent'] / df_merged['two_pa_opponent']

    df_merged['point_differential'] = df_merged['team_score'] - df_merged['team_score_opponent']

    df_merged['assist_rate'] = df_merged['assists'] / df_merged['poss']
    df_merged['assist_to_fg'] = df_merged['assists'] / df_merged['field_goals_made']
    df_merged['block_rate'] = df_merged['blocks'] / df_merged['poss_opponent']
    df_merged['lead_vs_outcome'] = df_merged['largest_lead'] - df_merged['point_differential']
    df_merged['fast_break_pct'] = df_merged['fast_break_points'] / df_merged['team_score']
    df_merged['points_off_turnover_pct'] = df_merged['turnover_points'] / df_merged['team_score']
    df_merged['foul_rate'] = df_merged['free_throws_attempted_opponent'] / df_merged['field_goals_attempted_opponent']

    maybe_keep = [
        'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
        'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
        'offensive_rebounds', 'defensive_rebounds', 'turnovers', 'field_goal_pct',
        'three_point_field_goal_pct', 'free_throw_pct', 'assists', 'two_pm', 'two_pa',
        'two_pm_opponent', 'two_pa_opponent'
    ]
    drop_cols_two = [
        'blocks', 'fast_break_points', 'flagrant_fouls', 'fouls', 'largest_lead', 'lead_changes',
        'lead_percentage', 'points_in_paint', 'steals', 'team_turnovers', 'technical_fouls',
        'total_rebounds', 'total_technical_fouls', 'total_turnovers', 'turnover_points'
    ]
    drop_opponent_cols = [
        'team_id_opponent', 'team_home_away_opponent', 'team_score_opponent', 'team_winner_opponent',
        'assists_opponent', 'blocks_opponent', 'defensive_rebounds_opponent', 'fast_break_points_opponent',
        'field_goal_pct_opponent', 'field_goals_made_opponent', 'field_goals_attempted_opponent',
        'flagrant_fouls_opponent', 'fouls_opponent', 'free_throw_pct_opponent', 'free_throws_made_opponent',
        'free_throws_attempted_opponent', 'largest_lead_opponent', 'lead_changes_opponent',
        'lead_percentage_opponent', 'offensive_rebounds_opponent', 'points_in_paint_opponent',
        'steals_opponent', 'team_turnovers_opponent', 'technical_fouls_opponent',
        'three_point_field_goal_pct_opponent', 'three_point_field_goals_made_opponent',
        'three_point_field_goals_attempted_opponent', 'total_rebounds_opponent',
        'total_technical_fouls_opponent', 'total_turnovers_opponent', 'turnover_points_opponent',
        'turnovers_opponent', 'opponent_team_score_opponent'
    ]

    df_merged.drop(columns=maybe_keep, inplace=True, errors='ignore')
    df_merged.drop(columns=drop_cols_two, inplace=True, errors='ignore')
    df_merged.drop(columns=drop_opponent_cols, inplace=True, errors='ignore')

    get_avg_cols = [
        'team_score', 'opponent_team_score', 'poss', 'poss_opponent', 'off_eff', 'def_eff', 'net_eff',
        'efg', 'efg_allowed', 'tov', 'stl_rate', 'orb', 'drb', 'ftr', 'ppp', 'two_pct',
        'two_pct_opponent', 'point_differential', 'assist_rate', 'assist_to_fg', 'block_rate',
        'lead_vs_outcome', 'fast_break_pct', 'points_off_turnover_pct', 'foul_rate',
        'three_pct', 'three_pct_opponent', 'three_attempt_rate', 'allowed_three_attempt_rate'
    ]

    df_merged = df_merged.sort_values(by='game_date', ascending=True)

    def encode_team_home_away(row):
        if row['season_type'] in [1, 3]:
            return 2
        return 1 if str(row['team_home_away']).strip().lower() == 'home' else 0

    df_merged['team_home_away'] = df_merged.apply(encode_team_home_away, axis=1)
    df_merged['team_winner'] = df_merged['team_winner'].apply(lambda x: 1 if x is True or x == 1 else 0)
    df_merged = add_elo_ratings(df_merged, elo_ratings, elo_last_season)

    df_merged['home_off_eff'] = df_merged.groupby('team_id').apply(
        lambda group: group.loc[group['team_home_away'] == 1, 'off_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    df_merged['home_def_eff'] = df_merged.groupby('team_id').apply(
        lambda group: group.loc[group['team_home_away'] == 1, 'def_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    df_merged['away_off_eff'] = df_merged.groupby('team_id').apply(
        lambda group: group.loc[group['team_home_away'] == 0, 'off_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    df_merged['away_def_eff'] = df_merged.groupby('team_id').apply(
        lambda group: group.loc[group['team_home_away'] == 0, 'def_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    df_merged['points_last10'] = df_merged.groupby('team_id')['team_score'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).sum()
    )
    df_merged['opp_points_last10'] = df_merged.groupby('team_id')['opponent_team_score'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).sum()
    )
    df_merged['poss_last10'] = df_merged.groupby('team_id')['poss'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).sum()
    )
    df_merged['poss_opp_last10'] = df_merged.groupby('team_id')['poss_opponent'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).sum()
    )

    df_merged['last_10_efficiency'] = (
        (df_merged['points_last10'] / df_merged['poss_last10'] * 100)
        - (df_merged['opp_points_last10'] / df_merged['poss_opp_last10'] * 100)
    )

    df_merged.drop(['points_last10', 'opp_points_last10', 'poss_last10', 'poss_opp_last10'], axis=1, inplace=True)

    for col in get_avg_cols:
        df_merged[f'{col}_avg'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).expanding().mean())
        df_merged[f'{col}_rolling_5'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).rolling(5).mean())

    df_merged['is_early_season'] = df_merged.isna().any(axis=1).astype(int)

    get_avg_cols = [col for col in get_avg_cols if col not in ['team_score', 'opponent_team_score']]
    df_merged.drop(columns=get_avg_cols, inplace=True, errors='ignore')

    df_merged['conference_strength'] = df_merged.groupby('short_conference_name')['net_eff_avg'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df_merged['team_winner_shifted'] = df_merged.groupby('team_id')['team_winner'].shift(1)

    df_merged['wins'] = df_merged.groupby('team_id')['team_winner_shifted'].transform(lambda x: (x == True).cumsum())
    df_merged['losses'] = df_merged.groupby('team_id')['team_winner_shifted'].transform(lambda x: (x == False).cumsum())

    df_merged['non_conf_win'] = (df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (
        df_merged['short_conference_name'] != df_merged['short_conference_name_opponent']
    )
    df_merged['non_conf_loss'] = ~(df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (
        df_merged['short_conference_name'] != df_merged['short_conference_name_opponent']
    )

    df_merged['non_conf_wins'] = df_merged.groupby('short_conference_name')['non_conf_win'].transform(lambda x: x.cumsum())
    df_merged['non_conf_losses'] = df_merged.groupby('short_conference_name')['non_conf_loss'].transform(lambda x: x.cumsum())

    df_merged['win_loss_pct'] = df_merged['wins'] / (df_merged['wins'] + df_merged['losses'])
    df_merged['non_conf_win_loss_pct'] = df_merged['non_conf_wins'] / (
        df_merged['non_conf_wins'] + df_merged['non_conf_losses']
    )

    df_merged.drop(
        columns=['wins', 'losses', 'non_conf_win', 'non_conf_loss', 'non_conf_wins', 'non_conf_losses', 'team_winner_shifted'],
        inplace=True
    )

    df_merged['conference_nonconf_win_pct'] = df_merged.groupby('short_conference_name')['non_conf_win_loss_pct'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df_merged['points_for'] = df_merged.groupby('team_id')['team_score'].transform(lambda x: x.shift(1).cumsum())
    df_merged['points_against'] = df_merged.groupby('team_id')['opponent_team_score'].transform(lambda x: x.shift(1).cumsum())
    k = 11.5
    df_merged['pythagorean_win_pct'] = (df_merged['points_for'] ** k) / (
        (df_merged['points_for'] ** k) + (df_merged['points_against'] ** k)
    )
    df_merged['luck'] = df_merged['win_loss_pct'] - df_merged['pythagorean_win_pct']

    df_merged['spread'] = df_merged['team_score'] - df_merged['opponent_team_score']

    df_merged.drop(
        columns=['team_score', 'opponent_team_score', 'points_for', 'points_against', 'pythagorean_win_pct'],
        inplace=True,
        errors='ignore'
    )

    df_final = df_merged.merge(df_merged, on=['game_id', 'season', 'season_type', 'game_date'], suffixes=('_a', '_b'))

    df_final = df_final[df_final['team_id_a'] != df_final['team_id_b']]

    df_final.drop(columns = ['spread_b'], inplace=True)
    # Rename spread_a to spread
    df_final.rename(columns={'spread_a': 'spread'}, inplace=True)

    df_final = df_final.drop(columns=['league_avg_off_eff_b', 'league_avg_def_eff_b'])
    df_final = df_final.rename(columns={'league_avg_off_eff_a': 'league_avg_off_eff', 'league_avg_def_eff_a': 'league_avg_def_eff'})

    df_final['sos'] = df_final.groupby('team_id_a')['net_eff_avg_b'].transform(lambda x: x.shift(1).expanding().mean())
    df_final['sos_opp'] = df_final.groupby('team_id_b')['net_eff_avg_a'].transform(lambda x: x.shift(1).expanding().mean())

    # Create a DataFrame with all combinations of dates and teams, and fill forward missing net_eff_avg_a/b values, rank both team and opponent

    teams = df_final['team_id_a'].unique()
    all_dates = df_final['game_date'].unique()
    all_dates.sort()
    # Build a DataFrame of all team-date combinations
    grid = pd.MultiIndex.from_product([teams, all_dates], names=['team_id', 'game_date']).to_frame(index=False)

    # Prepare DataFrames for each team's net_eff_avg_a for every game played
    net_eff_a = df_final[['team_id_a', 'game_date', 'net_eff_avg_a']].drop_duplicates(subset=['team_id_a', 'game_date'])
    net_eff_a = net_eff_a.rename(columns={'team_id_a': 'team_id'})

    # Prepare DataFrames for each team's net_eff_avg_b for every game played (opponent versions)
    net_eff_b = df_final[['team_id_b', 'game_date', 'net_eff_avg_b']].drop_duplicates(subset=['team_id_b', 'game_date'])
    net_eff_b = net_eff_b.rename(columns={'team_id_b': 'team_id', 'net_eff_avg_b': 'net_eff_avg'})

    # For team (A) -- to get rank for team_id_a on each game_date
    grid_a = grid.copy()
    grid_a = grid_a.merge(net_eff_a, on=['team_id', 'game_date'], how='left')
    grid_a.sort_values(['team_id', 'game_date'], inplace=True)
    grid_a['net_eff_avg_a'] = grid_a.groupby('team_id')['net_eff_avg_a'].shift(1).ffill()
    grid_a['rank'] = grid_a.groupby('game_date')['net_eff_avg_a'].rank(ascending=False, method='min')

    # For opponent (B) -- to get rank for team_id_b on each game_date
    grid_b = grid.copy()
    grid_b = grid_b.merge(net_eff_b, on=['team_id', 'game_date'], how='left')
    grid_b.sort_values(['team_id', 'game_date'], inplace=True)
    grid_b['net_eff_avg'] = grid_b.groupby('team_id')['net_eff_avg'].shift(1).ffill()
    grid_b['rank_opponent'] = grid_b.groupby('game_date')['net_eff_avg'].rank(ascending=False, method='min')

    # Merge the rank (A) back into df_final for each row/record corresponding to (team_id_a, game_date)
    df_final = df_final.merge(grid_a[['team_id', 'game_date', 'rank']],
                              left_on=['team_id_a', 'game_date'],
                              right_on=['team_id', 'game_date'],
                              how='left')
    df_final.drop(columns=['team_id'], inplace=True)

    # Merge the opponent rank (B) into df_final for (team_id_b, game_date)
    df_final = df_final.merge(grid_b[['team_id', 'game_date', 'rank_opponent']],
                              left_on=['team_id_b', 'game_date'],
                              right_on=['team_id', 'game_date'],
                              how='left')
    df_final.drop(columns=['team_id'], inplace=True)

    df_final.rename(columns={'rank': 'rank_a', 'rank_opponent': 'rank_b'}, inplace=True)

    df_final['rank_a'] = df_final['rank_a'].ffill()
    df_final['rank_b'] = df_final['rank_b'].ffill()

    df_final['adj_factor_def_a'] = df_final['league_avg_def_eff'] / df_final['def_eff_avg_b']
    df_final['adj_factor_off_a'] = df_final['league_avg_off_eff'] / df_final['off_eff_avg_b']
    df_final['adj_factor_def_b'] = df_final['league_avg_def_eff'] / df_final['def_eff_avg_a']
    df_final['adj_factor_off_b'] = df_final['league_avg_off_eff'] / df_final['off_eff_avg_a']

    df_final['adj_off_eff_a'] = df_final['off_eff_avg_a'] * df_final['adj_factor_def_a']
    df_final['adj_def_eff_a'] = df_final['def_eff_avg_a'] * df_final['adj_factor_off_a']
    df_final['adj_net_eff_a'] = df_final['adj_off_eff_a'] - df_final['adj_def_eff_a']
    df_final['adj_off_eff_b'] = df_final['off_eff_avg_b'] * df_final['adj_factor_def_b']
    df_final['adj_def_eff_b'] = df_final['def_eff_avg_b'] * df_final['adj_factor_off_b']
    df_final['adj_net_eff_b'] = df_final['adj_off_eff_b'] - df_final['adj_def_eff_b']

    df_final['threes_advantage'] = (df_final['three_attempt_rate_avg_a'] * df_final['three_pct_avg_a']) - (df_final['allowed_three_attempt_rate_avg_b'] * df_final['three_pct_opponent_avg_b'])
    df_final['threes_disadvantage'] = (df_final['allowed_three_attempt_rate_avg_a'] * df_final['three_pct_opponent_avg_a']) - (df_final['three_attempt_rate_avg_b'] * df_final['three_pct_avg_b'])
    # Repeat for two-pointers (where two_attempt_rate is 1 minus three_attempt_rate)
    df_final['two_pointers_advantage'] = ((1 - df_final['three_attempt_rate_avg_a']) * df_final['two_pct_avg_a']) - ((1 - df_final['allowed_three_attempt_rate_avg_b']) * df_final['two_pct_opponent_avg_b'])
    df_final['two_pointers_disadvantage'] = ((1 - df_final['allowed_three_attempt_rate_avg_a']) * df_final['two_pct_opponent_avg_a']) - ((1 - df_final['three_attempt_rate_avg_b']) * df_final['two_pct_avg_b'])
    # Free throws advantage
    df_final['free_throws_advantage'] = df_final['ftr_avg_a'] - df_final['foul_rate_avg_b']
    df_final['free_throws_disadvantage'] = df_final['foul_rate_avg_a'] - df_final['ftr_avg_b']

    df_final['adj_sos'] = df_final.groupby('team_id_a')['adj_net_eff_b'].transform(lambda x: x.shift(1).expanding().mean())
    df_final['adj_sos_opp'] = df_final.groupby('team_id_b')['adj_net_eff_a'].transform(lambda x: x.shift(1).expanding().mean())

    df_final['power_rating_a'] = df_final['adj_net_eff_a'] + df_final['adj_sos']
    df_final['power_rating_b'] = df_final['adj_net_eff_b'] + df_final['adj_sos_opp']

    df_final.drop(
        columns=[
            'adj_factor_def_a', 'adj_factor_off_a', 'adj_factor_def_b', 'adj_factor_off_b',
            'league_avg_off_eff', 'league_avg_def_eff'
        ],
        inplace=True
    )

    df_final['off_vs_def'] = df_final['off_eff_avg_a'] - df_final['def_eff_avg_b']
    df_final['def_vs_off'] = df_final['off_eff_avg_b'] - df_final['def_eff_avg_a']

    df_final['tov_vs_stl'] = df_final['tov_avg_a'] - df_final['stl_rate_avg_b']
    df_final['stl_vs_tov'] = df_final['tov_avg_b'] - df_final['stl_rate_avg_a']

    df_final['orb_vs_drb'] = df_final['orb_avg_a'] - df_final['drb_avg_b']
    df_final['drb_vs_orb'] = df_final['orb_avg_b'] - df_final['drb_avg_a']

    df_final['pace_diff'] = df_final['poss_avg_a'] - df_final['poss_avg_b']
    df_final['exp_poss'] = (df_final['poss_avg_a'] + df_final['poss_avg_b']) / 2

    df_final['efg_vs_efg_allowed'] = df_final['efg_avg_a'] - df_final['efg_allowed_avg_b']
    df_final['efg_allowed_vs_efg'] = df_final['efg_avg_b'] - df_final['efg_allowed_avg_a']

    df_final['margin_estimate'] = ((df_final['net_eff_avg_a'] - df_final['net_eff_avg_b']) * df_final['exp_poss']) / 100

    df_final['home_off_away_def'] = df_final['home_off_eff_a'] - df_final['away_def_eff_b']
    df_final['home_def_away_off'] = df_final['home_def_eff_a'] - df_final['away_off_eff_b']
    df_final['away_off_home_def'] = df_final['away_off_eff_a'] - df_final['home_def_eff_b']
    df_final['away_def_home_off'] = df_final['away_def_eff_a'] - df_final['home_off_eff_b']

    drop_cols_final = [
        'team_winner_b', 'team_home_away_b', 'is_early_season_b',
        'short_conference_name_a', 'short_conference_name_opponent_a',
        'short_conference_name_b', 'short_conference_name_opponent_b',
        'home_off_eff_a', 'home_def_eff_a', 'away_off_eff_a', 'away_def_eff_a',
        'home_off_eff_b', 'home_def_eff_b', 'away_off_eff_b', 'away_def_eff_b'
    ]
    df_final.drop(columns=drop_cols_final, inplace=True, errors='ignore')

    df_final = df_final.rename(columns={
        'is_early_season_a': 'is_early_season',
        'team_home_away_a': 'team_home_away',
        'team_winner_a': 'team_winner'
    })

    # Compute the quad score for the *current* game, but assign as quad_score_raw (not used downstream)
    if 'rank_b' in df_final.columns:
        rank_b = pd.to_numeric(df_final['rank_b'], errors='coerce')
        location = df_final['team_home_away']

        quad_1 = (
            ((location == 1) & rank_b.between(1, 30))
            | ((location == 2) & rank_b.between(1, 50))
            | ((location == 0) & rank_b.between(1, 75))
        )
        quad_2 = (
            ((location == 1) & rank_b.between(31, 75))
            | ((location == 2) & rank_b.between(51, 100))
            | ((location == 0) & rank_b.between(76, 135))
        )
        quad_3 = (
            ((location == 1) & rank_b.between(76, 160))
            | ((location == 2) & rank_b.between(101, 200))
            | ((location == 0) & rank_b.between(136, 240))
        )
        quad_4 = (
            ((location == 1) & rank_b.between(161, 353))
            | ((location == 2) & rank_b.between(201, 353))
            | ((location == 0) & rank_b.between(241, 353))
        )

        quad_win_score = np.select([quad_1, quad_2, quad_3, quad_4], [4, 3, 2, 1], default=0)
        quad_loss_score = np.select([quad_1, quad_2, quad_3, quad_4], [-1, -2, -3, -4], default=0)
        # This is the actual quad score for this game
        df_final['quad_score_raw'] = np.where(df_final['team_winner'] == 1, quad_win_score, quad_loss_score)

        # Now: the cumulative quad score *prior* to the game (no leakage)
        # Get cumulative sum by team, shifting by 1 so it's prior to the current game
        df_final['quad_score'] = (
            df_final.groupby('team_id_a')['quad_score_raw']
            .transform(lambda x: x.shift(1).fillna(0).cumsum())
        )
        df_final.drop(['quad_score_raw'], axis=1, inplace=True)

        df_final['quad_score'] = df_final['quad_score'].ffill()

    else:
        df_final['quad_score'] = 0
    
    df_final.drop(columns=['team_id_a', 'team_id_b'], inplace=True)

    a_cols = [col for col in df_final.columns if col.endswith('_a')]
    for col_a in a_cols:
        col_b = col_a[:-2] + '_b'
        if col_b in df_final.columns:
            col_diff = col_a[:-2] + '_diff'
            df_final[col_diff] = df_final[col_a] - df_final[col_b]
            df_final.drop([col_a, col_b], axis=1, inplace=True)

    df_final = df_final.fillna(-100)
    return df_final


def get_year_from_games_filename(path: Path) -> str:
    match = re.search(r'games_(\d{4})\.csv$', path.name)
    return match.group(1) if match else 'unknown'


def main() -> None:

    conference_mapping = pd.read_csv(
        CONFERENCE_MAP_PATH,
        usecols=['Conference', 'Mapped ESPN Team Name']
    ).dropna()
    conference_mapping['Mapped ESPN Team Name'] = conference_mapping['Mapped ESPN Team Name'].replace(
        {'Hawaii': "Hawai'i", 'St. Francis (PA)': 'Saint Francis', 'San Jose State': 'San José State'}
    )
    conference_mapping['team_location_key'] = conference_mapping['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()
    conference_mapping = conference_mapping.drop_duplicates(subset=['team_location_key'])
    conference_mapping = conference_mapping.rename(columns={'Conference': 'short_conference_name'})
    conference_mapping = conference_mapping[['team_location_key', 'short_conference_name']]

    games_files = sorted(
        GAME_RESULTS_DIR.glob('games_*.csv'),
        key=lambda p: get_year_from_games_filename(p)
    )

    elo_ratings = {}
    elo_last_season = {}
    all_years = []
    for games_file in games_files:
        year = get_year_from_games_filename(games_file)
        yearly_df = process_games_file(games_file, conference_mapping, elo_ratings, elo_last_season)
        yearly_output = Path(f'Data/cleaned_data/games_{year}_final.csv')
        yearly_df.to_csv(yearly_output, index=False)
        all_years.append(yearly_df)
        print(f'Processed {games_file.name} -> {yearly_output.name} ({len(yearly_df)} rows)')

    if all_years:
        dataset = pd.concat(all_years, ignore_index=True)
        dataset.to_csv('Game Predictions/dataset.csv', index=False)
        print(f'Wrote dataset.csv ({len(dataset)} rows)')
    else:
        print('No games_*.csv files found to process.')


if __name__ == '__main__':
    main()
