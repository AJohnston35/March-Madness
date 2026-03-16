from __future__ import annotations

import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
GAME_RESULTS_DIR = DATA_DIR / "game_results"
CONFERENCE_MAP_PATH = DATA_DIR / "kenpom" / "REF _ NCAAM Conference and ESPN Team Name Mapping.csv"

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

AVG_BASE_COLS = [
    'team_score',
    'opponent_team_score',
    'poss',
    'poss_opponent',
    'off_eff',
    'def_eff',
    'net_eff',
    'efg',
    'efg_allowed',
    'tov',
    'stl_rate',
    'orb',
    'drb',
    'ftr',
    'foul_rate',
    'ppp',
    'two_pct',
    'two_pct_opponent',
    'point_differential',
    'assist_rate',
    'assist_to_fg',
    'block_rate',
    'three_pct',
    'three_pct_opponent',
    'three_attempt_rate',
    'allowed_three_attempt_rate',
]

def load_and_prepare_dataset(dataset_path: str | Path, drop_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = df.sort_values(by='game_date')
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    return df

def align_features_for_model(df: pd.DataFrame, feature_names) -> pd.DataFrame:
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype(int)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    return data.reindex(columns=list(feature_names), fill_value=0)

def _safe_divide(num, den):
    if isinstance(den, (int, float, np.floating)):
        if den == 0 or pd.isna(den):
            return np.nan
        return num / den
    den = den.replace(0, np.nan)
    return num / den

def load_conference_mapping(conference_map_path: Path | str | None = None) -> pd.DataFrame:
    mapping_path = Path(conference_map_path) if conference_map_path else CONFERENCE_MAP_PATH
    conference_mapping = pd.read_csv(
        mapping_path,
        usecols=['Conference', 'Mapped ESPN Team Name']
    ).dropna()
    conference_mapping['Mapped ESPN Team Name'] = conference_mapping['Mapped ESPN Team Name'].replace(
        {'Hawaii': "Hawai'i", 'St. Francis (PA)': 'Saint Francis', 'San Jose State': 'San JosÃ© State'}
    )
    conference_mapping['team_location_key'] = (
        conference_mapping['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()
    )
    conference_mapping = conference_mapping.drop_duplicates(subset=['team_location_key'])
    conference_mapping = conference_mapping.rename(columns={'Conference': 'short_conference_name'})
    return conference_mapping[['team_location_key', 'short_conference_name']]

def get_expected_score(rating, opp_rating):
    exp = (opp_rating - rating) / 400
    return 1 / (1 + 10**exp)

def get_new_elos(home_rating, away_rating, margin):
    k = 25

    home_score = 0.5
    if margin > 0:
        home_score = 1
    elif margin < 0:
        home_score = 0

    expected_home_score = get_expected_score(home_rating, away_rating)
    new_home_score = home_rating + k * (home_score - expected_home_score)

    away_score = 1 - home_score
    expected_away_score = get_expected_score(away_rating, home_rating)
    new_away_score = away_rating + k * (away_score - expected_away_score)

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

def _concat_all_seasons_games():
    # Find all games files in GAME_RESULTS_DIR
    import glob
    files = sorted(GAME_RESULTS_DIR.glob("games_*.csv"))
    if not files:
        raise FileNotFoundError("No game files found for any season in {}".format(GAME_RESULTS_DIR))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        season_year = int(str(f.name).split("_")[1].split(".")[0])
        df["season"] = season_year  # ensure correct season for later groupbys
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    return df_all


def _ewm_team(x, alpha=0.15):
    """EWM with alpha=0.15, shift(1) so current game not included, min_periods=1 to avoid early nulls."""
    return x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()


def _ewm_team_std(x, alpha=0.15):
    return x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).std()


def process_all_games(
    games_df: pd.DataFrame,
    conference_mapping: pd.DataFrame,
    elo_ratings=None,
    elo_last_season=None,
) -> pd.DataFrame:
    """
    Process all games (one or many seasons) in one pass.
    - Performance averages (*_avg, *_rolling_5, variance, last_10_efficiency): EWM across all time (groupby team_id), alpha=0.15, so no early-season nulls.
    - Season-reset stats (wins, losses, sos, luck, conference_strength, etc.): computed only within current season (groupby season + team/conference).
    """
    if elo_ratings is None:
        elo_ratings = {}
    if elo_last_season is None:
        elo_last_season = {}
    if 'season' not in games_df.columns:
        raise ValueError("games_df must have a 'season' column")

    df = games_df.copy()
    for col in REQUIRED_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df['team_location'] = df['team_location'].replace(
        {'Hawaii': "Hawai'i", 'St. Francis (PA)': 'Saint Francis', 'San JosÃƒÂ© St': 'San Jose State'}
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

    df_merged = df_merged.sort_values(['season', 'game_date', 'game_id'])
    df_merged = df_merged.sort_values(['season', 'game_date'])

    df_merged['league_avg_off_eff'] = (
        df_merged.groupby('season')['off_eff']
        .transform(lambda x: x.shift(2).expanding(min_periods=3).mean())
    )
    df_merged['league_avg_def_eff'] = (
        df_merged.groupby('season')['def_eff']
        .transform(lambda x: x.shift(2).expanding(min_periods=3).mean())
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

    # EWM across all seasons (groupby team_id only) — avoids early-season nulls, recent games weighted most (alpha=0.15)
    ALPHA = 0.15

    df_merged['three_variance'] = df_merged.groupby('team_id')['three_pct'].transform(_ewm_team_std)
    df_merged['score_variance'] = df_merged.groupby('team_id')['team_score'].transform(_ewm_team_std)
    df_merged['def_score_variance'] = df_merged.groupby('team_id')['opponent_team_score'].transform(_ewm_team_std)
    df_merged['off_eff_variance'] = df_merged.groupby('team_id')['off_eff'].transform(_ewm_team_std)
    df_merged['pace_variance'] = df_merged.groupby('team_id')['poss'].transform(_ewm_team_std)

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
        'two_pct_opponent', 'point_differential', 'assist_rate', 'assist_to_fg', 'block_rate','foul_rate',
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

    # EWM across all seasons for "last 10" style efficiency (no season reset)
    df_merged['points_last10'] = df_merged.groupby('team_id')['team_score'].transform(_ewm_team)
    df_merged['opp_points_last10'] = df_merged.groupby('team_id')['opponent_team_score'].transform(_ewm_team)
    df_merged['poss_last10'] = df_merged.groupby('team_id')['poss'].transform(_ewm_team)
    df_merged['poss_opp_last10'] = df_merged.groupby('team_id')['poss_opponent'].transform(_ewm_team)

    df_merged['last_10_efficiency'] = (
        (df_merged['points_last10'] / df_merged['poss_last10'].replace(0, np.nan) * 100)
        - (df_merged['opp_points_last10'] / df_merged['poss_opp_last10'].replace(0, np.nan) * 100)
    )
    df_merged.drop(['points_last10', 'opp_points_last10', 'poss_last10', 'poss_opp_last10'], axis=1, inplace=True)

    for col in get_avg_cols:
        # EWM across all seasons (alpha=0.15, min_periods=1) for both _avg and _rolling_5
        df_merged[f'{col}_avg'] = df_merged.groupby('team_id')[col].transform(_ewm_team)
        df_merged[f'{col}_rolling_5'] = df_merged.groupby('team_id')[col].transform(_ewm_team)

    df_merged['is_early_season'] = df_merged.isna().any(axis=1).astype(int)

    get_avg_cols = [col for col in get_avg_cols if col not in ['team_score', 'opponent_team_score']]
    df_merged.drop(columns=get_avg_cols, inplace=True, errors='ignore')

    df_merged['conference_strength'] = df_merged.groupby(['season', 'short_conference_name'])['net_eff_avg'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )

    df_merged['team_winner_shifted'] = df_merged.groupby(['season', 'team_id'])['team_winner'].shift(1)

    df_merged['wins'] = df_merged.groupby(['season', 'team_id'])['team_winner_shifted'].transform(lambda x: (x == True).cumsum()).fillna(0)
    df_merged['losses'] = df_merged.groupby(['season', 'team_id'])['team_winner_shifted'].transform(lambda x: (x == False).cumsum()).fillna(0)

    df_merged['non_conf_win'] = (df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (
        df_merged['short_conference_name'] != df_merged['short_conference_name_opponent']
    )
    df_merged['non_conf_loss'] = ~(df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (
        df_merged['short_conference_name'] != df_merged['short_conference_name_opponent']
    )

    df_merged['non_conf_wins'] = df_merged.groupby(['season', 'short_conference_name'])['non_conf_win'].transform(lambda x: x.cumsum()).fillna(0)
    df_merged['non_conf_losses'] = df_merged.groupby(['season', 'short_conference_name'])['non_conf_loss'].transform(lambda x: x.cumsum()).fillna(0)

    df_merged['win_loss_pct'] = df_merged['wins'] / (df_merged['wins'] + df_merged['losses'])
    df_merged['non_conf_win_loss_pct'] = df_merged['non_conf_wins'] / (
        df_merged['non_conf_wins'] + df_merged['non_conf_losses']
    )

    df_merged.drop(
        columns=['wins', 'losses', 'non_conf_win', 'non_conf_loss', 'non_conf_wins', 'non_conf_losses', 'team_winner_shifted'],
        inplace=True
    )

    df_merged['conference_nonconf_win_pct'] = df_merged.groupby(['season', 'short_conference_name'])['non_conf_win_loss_pct'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )
    df_merged['points_for'] = df_merged.groupby(['season', 'team_id'])['team_score'].transform(lambda x: x.shift(1).cumsum())
    df_merged['points_against'] = df_merged.groupby(['season', 'team_id'])['opponent_team_score'].transform(lambda x: x.shift(1).cumsum())
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

    # Use full df_merged (all seasons) for the merge
    df_final = df_merged.merge(df_merged, on=['game_id', 'season', 'season_type', 'game_date'], suffixes=('_a', '_b'))

    df_final = df_final[df_final['team_id_a'] != df_final['team_id_b']]

    df_final.drop(columns=['spread_b'], inplace=True)
    df_final.rename(columns={'spread_a': 'spread'}, inplace=True)

    df_final = df_final.drop(columns=['league_avg_off_eff_b', 'league_avg_def_eff_b'])
    df_final = df_final.rename(columns={'league_avg_off_eff_a': 'league_avg_off_eff', 'league_avg_def_eff_a': 'league_avg_def_eff'})

    # Strength of schedule: season-only (not cumulative across seasons)
    df_final['sos'] = df_final.groupby(['season', 'team_id_a'])['net_eff_avg_b'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=1, adjust=False).mean())
    df_final['sos_opp'] = df_final.groupby(['season', 'team_id_b'])['net_eff_avg_a'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=1, adjust=False).mean())

    # Rank within (game_date, season) so we don't mix seasons
    net_eff_a = df_final[['team_id_a', 'game_date', 'season', 'net_eff_avg_a']].drop_duplicates(subset=['team_id_a', 'game_date', 'season'])
    net_eff_a = net_eff_a.rename(columns={'team_id_a': 'team_id'})
    net_eff_b = df_final[['team_id_b', 'game_date', 'season', 'net_eff_avg_b']].drop_duplicates(subset=['team_id_b', 'game_date', 'season'])
    net_eff_b = net_eff_b.rename(columns={'team_id_b': 'team_id', 'net_eff_avg_b': 'net_eff_avg'})

    grid_a = net_eff_a.sort_values(['team_id', 'season', 'game_date']).copy()
    grid_a['net_eff_avg_a'] = grid_a.groupby(['team_id', 'season'])['net_eff_avg_a'].shift(1).ffill()
    grid_a['rank'] = grid_a.groupby(['game_date', 'season'])['net_eff_avg_a'].rank(ascending=False, method='min')

    grid_b = net_eff_b.sort_values(['team_id', 'season', 'game_date']).copy()
    grid_b['net_eff_avg'] = grid_b.groupby(['team_id', 'season'])['net_eff_avg'].shift(1).ffill()
    grid_b['rank_opponent'] = grid_b.groupby(['game_date', 'season'])['net_eff_avg'].rank(ascending=False, method='min')

    df_final = df_final.merge(grid_a[['team_id', 'game_date', 'season', 'rank']],
                              left_on=['team_id_a', 'game_date', 'season'],
                              right_on=['team_id', 'game_date', 'season'],
                              how='left')
    df_final.drop(columns=['team_id'], inplace=True)

    df_final = df_final.merge(grid_b[['team_id', 'game_date', 'season', 'rank_opponent']],
                              left_on=['team_id_b', 'game_date', 'season'],
                              right_on=['team_id', 'game_date', 'season'],
                              how='left')
    df_final.drop(columns=['team_id'], inplace=True)

    df_final.rename(columns={'rank': 'rank_a', 'rank_opponent': 'rank_b'}, inplace=True)

    df_final['rank_a'] = df_final.groupby('season')['rank_a'].ffill()
    df_final['rank_b'] = df_final.groupby('season')['rank_b'].ffill()

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
    df_final['two_pointers_advantage'] = ((1 - df_final['three_attempt_rate_avg_a']) * df_final['two_pct_avg_a']) - ((1 - df_final['allowed_three_attempt_rate_avg_b']) * df_final['two_pct_opponent_avg_b'])
    df_final['two_pointers_disadvantage'] = ((1 - df_final['allowed_three_attempt_rate_avg_a']) * df_final['two_pct_opponent_avg_a']) - ((1 - df_final['three_attempt_rate_avg_b']) * df_final['two_pct_avg_b'])
    df_final['free_throws_advantage'] = df_final['ftr_avg_a'] - df_final['foul_rate_avg_b']
    df_final['free_throws_disadvantage'] = df_final['foul_rate_avg_a'] - df_final['ftr_avg_b']

    # Adj SOS: season-only
    df_final['adj_sos'] = df_final.groupby(['season', 'team_id_a'])['adj_net_eff_b'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=1, adjust=False).mean())
    df_final['adj_sos_opp'] = df_final.groupby(['season', 'team_id_b'])['adj_net_eff_a'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=1, adjust=False).mean())

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
        df_final['quad_score_raw'] = np.where(df_final['team_winner'] == 1, quad_win_score, quad_loss_score)

        # Quad score: season-only (resets each season)
        df_final['quad_score'] = (
            df_final.groupby(['season', 'team_id_a'])['quad_score_raw']
            .transform(lambda x: x.shift(1).fillna(0).cumsum())
        )
        df_final.drop(['quad_score_raw'], axis=1, inplace=True)

        df_final['quad_score'] = df_final.groupby('season')['quad_score'].ffill()

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

    # Print the 5 columns with the most nulls before dropping rows with nulls
    null_counts = df_final.isnull().sum()
    most_nulls = null_counts.sort_values(ascending=False).head(5)
    print("Top 5 columns with most nulls before dropping:")
    print(most_nulls)
    df_final = df_final.dropna()
    return df_final


def process_games_file(
    games_path: Path,
    conference_mapping: pd.DataFrame,
    elo_ratings=None,
    elo_last_season=None,
    prior_season_carryover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load a single season file and process it. For multi-season processing use process_all_games with concatenated data."""
    df = pd.read_csv(games_path)
    season_match = re.search(r'games_(\d{4})\.csv$', str(games_path), re.IGNORECASE)
    this_season = int(season_match.group(1)) if season_match else (df['season'].iloc[0] if 'season' in df.columns else None)
    if this_season is None:
        raise ValueError(f'Could not determine season from path or data: {games_path}')
    df['season'] = this_season
    return process_all_games(df, conference_mapping, elo_ratings or {}, elo_last_season or {})

def build_team_feature_rows(season: int, conference_mapping: pd.DataFrame) -> pd.DataFrame:
    # CHANGE: Concat all seasons up to and including requested season
    df_all = _concat_all_seasons_games()
    df = df_all[df_all["season"] <= season].copy()
    if df.empty:
        return df

    for col in REQUIRED_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df['team_location'] = df['team_location'].replace(
        {'Hawaii': "Hawai'i", 'St. Francis (PA)': 'Saint Francis', 'San JosÃƒÆ’Ã‚Â© St': 'San Jose State'}
    )
    df['team_location_key'] = df['team_location'].astype(str).str.strip().str.lower()
    df = df.merge(conference_mapping, on='team_location_key', how='left')
    df.drop(columns='team_location_key', inplace=True)
    df = df.dropna(subset=['short_conference_name'])

    df = df.drop(columns=[c for c in df.columns if c.startswith('opponent_') and c != 'opponent_team_score'])
    df['team_name'] = df['team_location']

    drop_cols = [
        'game_date_time', 'team_uid', 'team_location', 'team_slug', 'team_abbreviation',
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

    df_merged['off_eff'] = _safe_divide(df_merged['team_score'], df_merged['poss']) * 100
    df_merged['def_eff'] = _safe_divide(df_merged['team_score_opponent'], df_merged['poss_opponent']) * 100
    df_merged['net_eff'] = df_merged['off_eff'] - df_merged['def_eff']

    df_merged = df_merged.sort_values(['season', 'game_date', 'game_id'])
    league_daily = (
        df_merged.groupby(['season', 'game_date'], as_index=False)
        .agg(
            league_off_eff=('off_eff', 'mean'),
            league_def_eff=('def_eff', 'mean')
        )
    )
    league_daily[['league_avg_off_eff', 'league_avg_def_eff']] = (
        league_daily.groupby('season')[['league_off_eff', 'league_def_eff']]
        .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )

    df_merged['efg'] = _safe_divide(
        (df_merged['field_goals_made'] + (0.5 * df_merged['three_point_field_goals_made'])),
        df_merged['field_goals_attempted'],
    )
    df_merged['efg_allowed'] = _safe_divide(
        (df_merged['field_goals_made_opponent'] + (0.5 * df_merged['three_point_field_goals_made_opponent'])),
        df_merged['field_goals_attempted_opponent'],
    )
    df_merged['tov'] = _safe_divide(df_merged['team_turnovers'], df_merged['poss'])
    df_merged['stl_rate'] = _safe_divide(df_merged['steals'], df_merged['poss_opponent'])
    df_merged['orb'] = _safe_divide(
        df_merged['offensive_rebounds'],
        (df_merged['offensive_rebounds'] + df_merged['defensive_rebounds_opponent'])
    )
    df_merged['drb'] = _safe_divide(
        df_merged['defensive_rebounds'],
        (df_merged['defensive_rebounds'] + df_merged['offensive_rebounds_opponent'])
    )
    df_merged['ftr'] = _safe_divide(df_merged['free_throws_attempted'], df_merged['field_goals_attempted'])
    df_merged['ppp'] = _safe_divide(df_merged['team_score'], df_merged['poss'])

    df_merged['two_pm'] = df_merged['field_goals_made'] - df_merged['three_point_field_goals_made']
    df_merged['two_pa'] = df_merged['field_goals_attempted'] - df_merged['three_point_field_goals_attempted']
    df_merged['two_pct'] = _safe_divide(df_merged['two_pm'], df_merged['two_pa'])

    df_merged['three_pct'] = _safe_divide(
        df_merged['three_point_field_goals_made'],
        df_merged['three_point_field_goals_attempted']
    )
    df_merged['three_pct_opponent'] = _safe_divide(
        df_merged['three_point_field_goals_made_opponent'],
        df_merged['three_point_field_goals_attempted_opponent']
    )
    df_merged['three_attempt_rate'] = _safe_divide(
        df_merged['three_point_field_goals_attempted'],
        df_merged['field_goals_attempted']
    )
    df_merged['allowed_three_attempt_rate'] = _safe_divide(
        df_merged['three_point_field_goals_attempted_opponent'],
        df_merged['field_goals_attempted_opponent']
    )

    ALPHA = 0.15
    df_merged['three_variance'] = df_merged.groupby('team_id')['three_pct'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).std()
    )
    df_merged['score_variance'] = df_merged.groupby('team_id')['team_score'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).std()
    )
    df_merged['def_score_variance'] = df_merged.groupby('team_id')['opponent_team_score'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).std()
    )
    df_merged['off_eff_variance'] = df_merged.groupby('team_id')['off_eff'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).std()
    )
    df_merged['pace_variance'] = df_merged.groupby('team_id')['poss'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).std()
    )

    df_merged['two_pm_opponent'] = (
        df_merged['field_goals_made_opponent'] - df_merged['three_point_field_goals_made_opponent']
    )
    df_merged['two_pa_opponent'] = (
        df_merged['field_goals_attempted_opponent'] - df_merged['three_point_field_goals_attempted_opponent']
    )
    df_merged['two_pct_opponent'] = _safe_divide(df_merged['two_pm_opponent'], df_merged['two_pa_opponent'])

    df_merged['point_differential'] = df_merged['team_score'] - df_merged['team_score_opponent']
    df_merged['assist_rate'] = _safe_divide(df_merged['assists'], df_merged['poss'])
    df_merged['assist_to_fg'] = _safe_divide(df_merged['assists'], df_merged['field_goals_made'])
    df_merged['block_rate'] = _safe_divide(df_merged['blocks'], df_merged['poss_opponent'])
    df_merged['foul_rate'] = _safe_divide(
        df_merged['free_throws_attempted_opponent'],
        df_merged['field_goals_attempted_opponent']
    )

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

    df_merged = df_merged.sort_values(by=['game_date', 'game_id'], ascending=True)

    def encode_team_home_away(row):
        if row['season_type'] in [1, 3]:
            return 2
        return 1 if str(row['team_home_away']).strip().lower() == 'home' else 0

    df_merged['team_home_away'] = df_merged.apply(encode_team_home_away, axis=1)
    df_merged['team_winner'] = df_merged['team_winner'].apply(lambda x: 1 if x is True or x == 1 else 0)

    # EWA of points/possessions last 10
    df_merged['points_last10'] = df_merged.groupby('team_id')['team_score'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean())
    df_merged['opp_points_last10'] = df_merged.groupby('team_id')['opponent_team_score'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean())
    df_merged['poss_last10'] = df_merged.groupby('team_id')['poss'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean())
    df_merged['poss_opp_last10'] = df_merged.groupby('team_id')['poss_opponent'].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean())

    df_merged['last_10_efficiency'] = (
        _safe_divide(df_merged['points_last10'], df_merged['poss_last10']) * 100
        - _safe_divide(df_merged['opp_points_last10'], df_merged['poss_opp_last10']) * 100
    )
    df_merged.drop(columns=['points_last10', 'opp_points_last10', 'poss_last10', 'poss_opp_last10'], inplace=True)

    for col in AVG_BASE_COLS:
        df_merged[f'{col}_avg'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean())
        df_merged[f'{col}_rolling_5'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean() if len(x.dropna()) < 5 else x.shift(1).rolling(5, min_periods=3).apply(lambda y: pd.Series(y).ewm(alpha=ALPHA, adjust=False).mean().iloc[-1]))

    df_merged['is_early_season'] = df_merged.isna().any(axis=1).astype(int)
    drop_avg_bases = [c for c in AVG_BASE_COLS if c not in ['team_score', 'opponent_team_score']]
    df_merged.drop(columns=drop_avg_bases, inplace=True, errors='ignore')

    df_merged = df_merged.merge(
        league_daily[['season', 'game_date', 'league_avg_off_eff', 'league_avg_def_eff']],
        on=['season', 'game_date'],
        how='left'
    )

    df_merged['conference_strength'] = df_merged.groupby('short_conference_name')['net_eff_avg'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
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

    df_merged['win_loss_pct'] = _safe_divide(df_merged['wins'], (df_merged['wins'] + df_merged['losses']))
    df_merged['non_conf_win_loss_pct'] = _safe_divide(
        df_merged['non_conf_wins'], (df_merged['non_conf_wins'] + df_merged['non_conf_losses'])
    )
    df_merged.drop(
        columns=['wins', 'losses', 'non_conf_win', 'non_conf_loss', 'non_conf_wins', 'non_conf_losses', 'team_winner_shifted'],
        inplace=True
    )

    df_merged['conference_nonconf_win_pct'] = df_merged.groupby('short_conference_name')['non_conf_win_loss_pct'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )

    df_merged['points_for'] = df_merged.groupby('team_id')['team_score'].transform(lambda x: x.shift(1).cumsum())
    df_merged['points_against'] = df_merged.groupby('team_id')['opponent_team_score'].transform(lambda x: x.shift(1).cumsum())
    k = 13.91
    df_merged['pythagorean_win_pct'] = _safe_divide(
        (df_merged['points_for'] ** k), ((df_merged['points_for'] ** k) + (df_merged['points_against'] ** k))
    )
    df_merged['luck'] = df_merged['win_loss_pct'] - df_merged['pythagorean_win_pct']
    df_merged.drop(
        columns=['team_score', 'opponent_team_score', 'points_for', 'points_against', 'pythagorean_win_pct'],
        inplace=True,
        errors='ignore'
    )

    # Only keep rows for this specific season at output
    dfout = df_merged[df_merged["season"] == season].copy()

    matchup_base = dfout.merge(
        dfout,
        on=['game_id', 'season', 'season_type', 'game_date'],
        suffixes=('_a', '_b')
    )
    matchup_base = matchup_base[matchup_base['team_id_a'] != matchup_base['team_id_b']]
    matchup_base = matchup_base.merge(
        league_daily[league_daily['season'] == season][['season', 'game_date', 'league_avg_off_eff', 'league_avg_def_eff']],
        on=['season', 'game_date'],
        how='left'
    )
    matchup_base = matchup_base.sort_values(by=['game_date', 'game_id'], ascending=True)

    matchup_base['sos'] = matchup_base.groupby('team_id_a')['net_eff_avg_b'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )
    matchup_base['sos_opp'] = matchup_base.groupby('team_id_b')['net_eff_avg_a'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )

    matchup_base['adj_factor_def_a'] = matchup_base['league_avg_def_eff'] / matchup_base['def_eff_avg_b']
    matchup_base['adj_factor_off_a'] = matchup_base['league_avg_off_eff'] / matchup_base['off_eff_avg_b']
    matchup_base['adj_factor_def_b'] = matchup_base['league_avg_def_eff'] / matchup_base['def_eff_avg_a']
    matchup_base['adj_factor_off_b'] = matchup_base['league_avg_off_eff'] / matchup_base['off_eff_avg_a']

    matchup_base['adj_off_eff_a'] = matchup_base['off_eff_avg_a'] * matchup_base['adj_factor_def_a']
    matchup_base['adj_def_eff_a'] = matchup_base['def_eff_avg_a'] * matchup_base['adj_factor_off_a']
    matchup_base['adj_net_eff_a'] = matchup_base['adj_off_eff_a'] - matchup_base['adj_def_eff_a']
    matchup_base['adj_off_eff_b'] = matchup_base['off_eff_avg_b'] * matchup_base['adj_factor_def_b']
    matchup_base['adj_def_eff_b'] = matchup_base['def_eff_avg_b'] * matchup_base['adj_factor_off_b']
    matchup_base['adj_net_eff_b'] = matchup_base['adj_off_eff_b'] - matchup_base['adj_def_eff_b']

    matchup_base['adj_sos'] = matchup_base.groupby('team_id_a')['adj_net_eff_b'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )
    matchup_base['adj_sos_opp'] = matchup_base.groupby('team_id_b')['adj_net_eff_a'].transform(
        lambda x: x.shift(1).ewm(alpha=ALPHA, min_periods=3, adjust=False).mean()
    )

    matchup_base['power_rating_a'] = matchup_base['adj_net_eff_a'] + matchup_base['adj_sos']
    matchup_base['power_rating_b'] = matchup_base['adj_net_eff_b'] + matchup_base['adj_sos_opp']

    sos_map = matchup_base[
        ['game_id', 'team_id_a', 'sos', 'adj_sos', 'adj_off_eff_a', 'adj_def_eff_a', 'adj_net_eff_a', 'power_rating_a']
    ].rename(
        columns={
            'team_id_a': 'team_id',
            'adj_off_eff_a': 'adj_off_eff',
            'adj_def_eff_a': 'adj_def_eff',
            'adj_net_eff_a': 'adj_net_eff',
            'power_rating_a': 'power_rating'
        }
    )
    dfout = dfout.merge(sos_map, on=['game_id', 'team_id'], how='left')

    return dfout

def build_matchup_feature_row(
    left_snapshot: pd.Series,
    right_snapshot: pd.Series,
    season: int,
    season_type: int,
    team_home_away: int,
) -> pd.DataFrame:
    def val(row: pd.Series, column: str, default: float = np.nan) -> float:
        if column not in row or pd.isna(row[column]):
            return default
        return float(row[column])

    def val_or_nan(row: pd.Series, column: str) -> float:
        if column not in row or pd.isna(row[column]):
            return np.nan
        return float(row[column])

    def safe_scalar_divide(num: float, den: float) -> float:
        if den == 0 or pd.isna(den):
            return np.nan
        return num / den

    exp_poss = (val(left_snapshot, 'poss_avg') + val(right_snapshot, 'poss_avg')) / 2

    left_date = pd.to_datetime(left_snapshot.get('game_date', None), errors='coerce')
    right_date = pd.to_datetime(right_snapshot.get('game_date', None), errors='coerce')
    if pd.isna(right_date) or (not pd.isna(left_date) and left_date >= right_date):
        league_avg_off_eff = val_or_nan(left_snapshot, 'league_avg_off_eff')
        league_avg_def_eff = val_or_nan(left_snapshot, 'league_avg_def_eff')
    else:
        league_avg_off_eff = val_or_nan(right_snapshot, 'league_avg_off_eff')
        league_avg_def_eff = val_or_nan(right_snapshot, 'league_avg_def_eff')

    off_eff_avg_a = val_or_nan(left_snapshot, 'off_eff_avg')
    def_eff_avg_a = val_or_nan(left_snapshot, 'def_eff_avg')
    off_eff_avg_b = val_or_nan(right_snapshot, 'off_eff_avg')
    def_eff_avg_b = val_or_nan(right_snapshot, 'def_eff_avg')

    adj_factor_def_a = safe_scalar_divide(league_avg_def_eff, def_eff_avg_b)
    adj_factor_off_a = safe_scalar_divide(league_avg_off_eff, off_eff_avg_b)
    adj_factor_def_b = safe_scalar_divide(league_avg_def_eff, def_eff_avg_a)
    adj_factor_off_b = safe_scalar_divide(league_avg_off_eff, off_eff_avg_a)

    adj_off_eff_a = off_eff_avg_a * adj_factor_def_a
    adj_def_eff_a = def_eff_avg_a * adj_factor_off_a
    adj_net_eff_a = adj_off_eff_a - adj_def_eff_a
    adj_off_eff_b = off_eff_avg_b * adj_factor_def_b
    adj_def_eff_b = def_eff_avg_b * adj_factor_off_b
    adj_net_eff_b = adj_off_eff_b - adj_def_eff_b

    adj_sos_a = val_or_nan(left_snapshot, 'adj_sos')
    adj_sos_b = val_or_nan(right_snapshot, 'adj_sos')
    power_rating_a = adj_net_eff_a + adj_sos_a
    power_rating_b = adj_net_eff_b + adj_sos_b

    three_attempt_rate_avg_a = val(left_snapshot, 'three_attempt_rate_avg')
    three_attempt_rate_avg_b = val(right_snapshot, 'three_attempt_rate_avg')
    allowed_three_attempt_rate_avg_a = val(left_snapshot, 'allowed_three_attempt_rate_avg')
    allowed_three_attempt_rate_avg_b = val(right_snapshot, 'allowed_three_attempt_rate_avg')
    three_pct_avg_a = val(left_snapshot, 'three_pct_avg')
    three_pct_avg_b = val(right_snapshot, 'three_pct_avg')
    three_pct_opponent_avg_a = val(left_snapshot, 'three_pct_opponent_avg')
    three_pct_opponent_avg_b = val(right_snapshot, 'three_pct_opponent_avg')
    two_pct_avg_a = val(left_snapshot, 'two_pct_avg')
    two_pct_avg_b = val(right_snapshot, 'two_pct_avg')
    two_pct_opponent_avg_a = val(left_snapshot, 'two_pct_opponent_avg')
    two_pct_opponent_avg_b = val(right_snapshot, 'two_pct_opponent_avg')
    ftr_avg_a = val(left_snapshot, 'ftr_avg')
    ftr_avg_b = val(right_snapshot, 'ftr_avg')
    foul_rate_avg_a = val(left_snapshot, 'foul_rate_avg')
    foul_rate_avg_b = val(right_snapshot, 'foul_rate_avg')

    threes_advantage = (three_attempt_rate_avg_a * three_pct_avg_a) - (
        allowed_three_attempt_rate_avg_b * three_pct_opponent_avg_b
    )
    threes_disadvantage = (allowed_three_attempt_rate_avg_a * three_pct_opponent_avg_a) - (
        three_attempt_rate_avg_b * three_pct_avg_b
    )
    two_pointers_advantage = ((1 - three_attempt_rate_avg_a) * two_pct_avg_a) - (
        (1 - allowed_three_attempt_rate_avg_b) * two_pct_opponent_avg_b
    )
    two_pointers_disadvantage = ((1 - allowed_three_attempt_rate_avg_a) * two_pct_opponent_avg_a) - (
        (1 - three_attempt_rate_avg_b) * two_pct_avg_b
    )
    free_throws_advantage = ftr_avg_a - foul_rate_avg_b
    free_throws_disadvantage = foul_rate_avg_a - ftr_avg_b

    feature_row = {
        'season': int(season),
        'season_type': int(season_type),
        'team_home_away': int(team_home_away),
        'is_early_season': int(val(left_snapshot, 'is_early_season', 1.0)),
        'sos': val(left_snapshot, 'sos'),
        'sos_opp': val(right_snapshot, 'sos'),
        'threes_advantage': threes_advantage,
        'threes_disadvantage': threes_disadvantage,
        'two_pointers_advantage': two_pointers_advantage,
        'two_pointers_disadvantage': two_pointers_disadvantage,
        'free_throws_advantage': free_throws_advantage,
        'free_throws_disadvantage': free_throws_disadvantage,
        'adj_sos': val(left_snapshot, 'adj_sos'),
        'adj_sos_opp': val(right_snapshot, 'adj_sos'),
        'off_vs_def': val(left_snapshot, 'off_eff_avg') - val(right_snapshot, 'def_eff_avg'),
        'def_vs_off': val(right_snapshot, 'off_eff_avg') - val(left_snapshot, 'def_eff_avg'),
        'tov_vs_stl': val(left_snapshot, 'tov_avg') - val(right_snapshot, 'stl_rate_avg'),
        'stl_vs_tov': val(right_snapshot, 'tov_avg') - val(left_snapshot, 'stl_rate_avg'),
        'orb_vs_drb': val(left_snapshot, 'orb_avg') - val(right_snapshot, 'drb_avg'),
        'drb_vs_orb': val(right_snapshot, 'orb_avg') - val(left_snapshot, 'drb_avg'),
        'pace_diff': val(left_snapshot, 'poss_avg') - val(right_snapshot, 'poss_avg'),
        'exp_poss': exp_poss,
        'efg_vs_efg_allowed': val(left_snapshot, 'efg_avg') - val(right_snapshot, 'efg_allowed_avg'),
        'efg_allowed_vs_efg': val(right_snapshot, 'efg_avg') - val(left_snapshot, 'efg_allowed_avg'),
        'margin_estimate': ((val(left_snapshot, 'net_eff_avg') - val(right_snapshot, 'net_eff_avg')) * exp_poss) / 100,
        'home_off_away_def': val(left_snapshot, 'home_off_eff') - val(right_snapshot, 'away_def_eff'),
        'home_def_away_off': val(left_snapshot, 'home_def_eff') - val(right_snapshot, 'away_off_eff'),
        'away_off_home_def': val(left_snapshot, 'away_off_eff') - val(right_snapshot, 'home_def_eff'),
        'away_def_home_off': val(left_snapshot, 'away_def_eff') - val(right_snapshot, 'home_off_eff'),
        'three_variance_diff': val(left_snapshot, 'three_variance') - val(right_snapshot, 'three_variance'),
        'score_variance_diff': val(left_snapshot, 'score_variance') - val(right_snapshot, 'score_variance'),
        'def_score_variance_diff': (
            val(left_snapshot, 'def_score_variance') - val(right_snapshot, 'def_score_variance')
        ),
        'off_eff_variance_diff': val(left_snapshot, 'off_eff_variance') - val(right_snapshot, 'off_eff_variance'),
        'pace_variance_diff': val(left_snapshot, 'pace_variance') - val(right_snapshot, 'pace_variance'),
        'last_10_efficiency_diff': val(left_snapshot, 'last_10_efficiency') - val(right_snapshot, 'last_10_efficiency'),
    }

    for base_col in AVG_BASE_COLS:
        feature_row[f'{base_col}_avg_diff'] = val(left_snapshot, f'{base_col}_avg') - val(right_snapshot, f'{base_col}_avg')
        feature_row[f'{base_col}_rolling_5_diff'] = (
            val(left_snapshot, f'{base_col}_rolling_5') - val(right_snapshot, f'{base_col}_rolling_5')
        )

    feature_row['conference_strength_diff'] = (
        val(left_snapshot, 'conference_strength') - val(right_snapshot, 'conference_strength')
    )
    feature_row['win_loss_pct_diff'] = val(left_snapshot, 'win_loss_pct') - val(right_snapshot, 'win_loss_pct')
    feature_row['non_conf_win_loss_pct_diff'] = (
        val(left_snapshot, 'non_conf_win_loss_pct') - val(right_snapshot, 'non_conf_win_loss_pct')
    )
    feature_row['conference_nonconf_win_pct_diff'] = (
        val(left_snapshot, 'conference_nonconf_win_pct') - val(right_snapshot, 'conference_nonconf_win_pct')
    )
    feature_row['luck_diff'] = val(left_snapshot, 'luck') - val(right_snapshot, 'luck')
    feature_row['adj_off_eff_diff'] = adj_off_eff_a - adj_off_eff_b
    feature_row['adj_def_eff_diff'] = adj_def_eff_a - adj_def_eff_b
    feature_row['adj_net_eff_diff'] = adj_net_eff_a - adj_net_eff_b
    feature_row['power_rating_diff'] = power_rating_a - power_rating_b

    return pd.DataFrame([feature_row]).replace([np.inf, -np.inf], np.nan).fillna(-100)
