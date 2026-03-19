from __future__ import annotations

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
    'total_turnovers',
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

RESIDUAL_STATS = [
    'poss',
    'poss_opponent',
    'off_eff',
    'def_eff',
    'efg',
    'ppp',
    'drb',
    'orb',
]

CLOSE_GAME_STATS = [
    'poss',
    'poss_opponent',
    'off_eff',
    'def_eff',
    'efg',
    'ppp',
    'drb',
    'orb',
]

TEAM_LOCATION_REPLACEMENTS = {
    'Hawaii': "Hawai'i",
    'St. Francis (PA)': 'Saint Francis',
    'San JosÃƒÆ’Ã‚Â© St': 'San Jose State',
    'San JosÃƒÂ© St': 'San Jose State',
}

CONFERENCE_MAPPING_REPLACEMENTS = {
    'Hawaii': "Hawai'i",
    'St. Francis (PA)': 'Saint Francis',
    'San Jose State': 'San JosÃƒÂ© State',
}

TEAM_METADATA_DROP_COLS = [
    'game_date_time',
    'team_uid',
    'team_location',
    'team_slug',
    'team_name',
    'team_abbreviation',
    'team_display_name',
    'team_short_display_name',
    'team_color',
    'team_alternate_color',
    'team_logo',
]

RAW_STAT_DROP_COLS = [
    'field_goals_made',
    'field_goals_attempted',
    'three_point_field_goals_made',
    'three_point_field_goals_attempted',
    'free_throws_made',
    'free_throws_attempted',
    'offensive_rebounds',
    'defensive_rebounds',
    'turnovers',
    'field_goal_pct',
    'three_point_field_goal_pct',
    'free_throw_pct',
    'assists',
    'two_pm',
    'two_pa',
    'two_pm_opponent',
    'two_pa_opponent',
]

BOX_SCORE_DROP_COLS = [
    'blocks',
    'fast_break_points',
    'flagrant_fouls',
    'fouls',
    'largest_lead',
    'lead_changes',
    'lead_percentage',
    'points_in_paint',
    'steals',
    'team_turnovers',
    'technical_fouls',
    'total_rebounds',
    'total_technical_fouls',
    'total_turnovers',
    'turnover_points',
]

OPPONENT_DROP_COLS = [
    'team_id_opponent',
    'team_home_away_opponent',
    'team_score_opponent',
    'team_winner_opponent',
    'assists_opponent',
    'blocks_opponent',
    'defensive_rebounds_opponent',
    'fast_break_points_opponent',
    'field_goal_pct_opponent',
    'field_goals_made_opponent',
    'field_goals_attempted_opponent',
    'flagrant_fouls_opponent',
    'fouls_opponent',
    'free_throw_pct_opponent',
    'free_throws_made_opponent',
    'free_throws_attempted_opponent',
    'largest_lead_opponent',
    'lead_changes_opponent',
    'lead_percentage_opponent',
    'offensive_rebounds_opponent',
    'points_in_paint_opponent',
    'steals_opponent',
    'team_turnovers_opponent',
    'technical_fouls_opponent',
    'three_point_field_goal_pct_opponent',
    'three_point_field_goals_made_opponent',
    'three_point_field_goals_attempted_opponent',
    'total_rebounds_opponent',
    'total_technical_fouls_opponent',
    'total_turnovers_opponent',
    'turnover_points_opponent',
    'turnovers_opponent',
    'opponent_team_score_opponent',
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


def load_conference_mapping(conference_map_path: Path | str | None = None) -> pd.DataFrame:
    mapping_path = Path(conference_map_path) if conference_map_path else CONFERENCE_MAP_PATH
    conference_mapping = pd.read_csv(
        mapping_path,
        usecols=['Conference', 'Mapped ESPN Team Name'],
    ).dropna()
    conference_mapping['Mapped ESPN Team Name'] = conference_mapping['Mapped ESPN Team Name'].replace(
        CONFERENCE_MAPPING_REPLACEMENTS
    )
    conference_mapping['team_location_key'] = (
        conference_mapping['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()
    )
    conference_mapping = conference_mapping.drop_duplicates(subset=['team_location_key'])
    conference_mapping = conference_mapping.rename(columns={'Conference': 'short_conference_name'})
    return conference_mapping[['team_location_key', 'short_conference_name']]


def _historical_mean_by_group(series: pd.Series, group_keys) -> pd.Series:
    return series.groupby(group_keys).transform(lambda x: x.shift(1).expanding(min_periods=1).mean())


def _historical_conditional_mean(series: pd.Series, condition: pd.Series, group_keys) -> pd.Series:
    valid = condition.astype(bool) & series.notna()
    running_total = series.where(valid, 0.0).groupby(group_keys).transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    running_count = valid.astype(float).groupby(group_keys).transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    return running_total / running_count.replace(0, np.nan)


def add_elo_ratings(df_merged, elo_ratings, elo_last_season):
    """
    Store a leakage-safe pre-game rating on each row.

    The current game never contributes to the row's `elo` value. Ratings are
    updated only after both teams' pre-game values are written.

    If a team does not yet have a rating for the current season, initialize it
    from that team's final Elo from the previous season when available.
    """
    base_rating = 1500.0
    k_factor = 20.0

    df_merged = df_merged.sort_values(['season', 'game_date', 'game_id', 'team_id']).copy()
    df_merged['elo'] = base_rating

    def expected_score(player_rating, opponent_rating):
        return 1.0 / (1.0 + 10.0 ** ((opponent_rating - player_rating) / 400.0))

    def get_initial_rating(team_id, season):
        season_key = (season, team_id)
        if season_key in elo_ratings:
            return float(elo_ratings[season_key])

        prior_season = elo_last_season.get(team_id)
        if prior_season is not None:
            prior_key = (prior_season, team_id)
            if prior_key in elo_ratings:
                prev_elo = float(elo_ratings[prior_key])
                # Adjust 30% of the difference to 1500 (mean), shifting toward the mean
                adjusted_elo = 1500 + 0.7 * (prev_elo - 1500)
                elo_ratings[season_key] = adjusted_elo
                return adjusted_elo

        elo_ratings[season_key] = base_rating
        return base_rating

    for _, game_rows in df_merged.groupby(['season', 'game_id'], sort=False):
        game_rows = game_rows.drop_duplicates(subset='team_id')
        if len(game_rows) != 2:
            continue

        idx1, idx2 = game_rows.index[0], game_rows.index[1]
        season = int(game_rows['season'].iloc[0])
        team1 = df_merged.at[idx1, 'team_id']
        team2 = df_merged.at[idx2, 'team_id']

        key1 = (season, team1)
        key2 = (season, team2)

        rating1 = get_initial_rating(team1, season)
        rating2 = get_initial_rating(team2, season)

        df_merged.at[idx1, 'elo'] = rating1
        df_merged.at[idx2, 'elo'] = rating2

        score1 = df_merged.at[idx1, 'team_score']
        score2 = df_merged.at[idx2, 'team_score']

        if score1 > score2:
            actual1, actual2 = 1.0, 0.0
        elif score1 < score2:
            actual1, actual2 = 0.0, 1.0
        else:
            actual1 = actual2 = 0.5

        expected1 = expected_score(rating1, rating2)
        expected2 = expected_score(rating2, rating1)

        elo_ratings[key1] = float(rating1 + k_factor * (actual1 - expected1))
        elo_ratings[key2] = float(rating2 + k_factor * (actual2 - expected2))
        elo_last_season[team1] = season
        elo_last_season[team2] = season

    return df_merged


def _concat_all_seasons_games():
    files = sorted(GAME_RESULTS_DIR.glob("games_*.csv"))
    if not files:
        raise FileNotFoundError(f"No game files found for any season in {GAME_RESULTS_DIR}")

    dfs = []
    for file_path in files:
        df = pd.read_csv(file_path)
        season_year = int(str(file_path.name).split("_")[1].split(".")[0])
        df['season'] = season_year
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, sort=False)


def _prepare_team_game_rows(
    games_df: pd.DataFrame,
    conference_mapping: pd.DataFrame,
    elo_ratings=None,
    elo_last_season=None,
    *,
    keep_team_name: bool = False,
) -> pd.DataFrame:
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

    df['team_location'] = df['team_location'].replace(TEAM_LOCATION_REPLACEMENTS)

    df['team_location_key'] = df['team_location'].astype(str).str.strip().str.lower()
    df = df.merge(conference_mapping, on='team_location_key', how='left')
    df.drop(columns='team_location_key', inplace=True)

    df = df.dropna(subset=['short_conference_name'])
    df = df.drop(columns=[col for col in df.columns if col.startswith('opponent_') and col != 'opponent_team_score'])

    drop_cols = TEAM_METADATA_DROP_COLS.copy()
    if keep_team_name:
        drop_cols.remove('team_name')
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df_merged = df.merge(df, on=['game_id', 'season', 'season_type', 'game_date'], suffixes=(None, '_opponent'))
    df_merged = df_merged[df_merged['team_id'] != df_merged['team_id_opponent']].copy()
    df_merged.sort_values(['season', 'game_date', 'game_id', 'team_id'], inplace=True)

    df_merged['poss'] = (
        df_merged['field_goals_attempted']
        - df_merged['offensive_rebounds']
        + df_merged['team_turnovers']
        + (0.475 * df_merged['free_throws_attempted'])
    )
    df_merged['poss_opponent'] = (
        df_merged['field_goals_attempted_opponent']
        - df_merged['offensive_rebounds_opponent']
        + df_merged['team_turnovers_opponent']
        + (0.475 * df_merged['free_throws_attempted_opponent'])
    )

    df_merged['off_eff'] = (df_merged['team_score'] / df_merged['poss']) * 100
    df_merged['def_eff'] = (df_merged['team_score_opponent'] / df_merged['poss_opponent']) * 100
    df_merged['net_eff'] = df_merged['off_eff'] - df_merged['def_eff']

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
    df_merged['three_pct'] = (
        df_merged['three_point_field_goals_made'] / df_merged['three_point_field_goals_attempted']
    )
    df_merged['three_pct_opponent'] = (
        df_merged['three_point_field_goals_made_opponent'] / df_merged['three_point_field_goals_attempted_opponent']
    )
    df_merged['three_attempt_rate'] = (
        df_merged['three_point_field_goals_attempted'] / df_merged['field_goals_attempted']
    )
    df_merged['allowed_three_attempt_rate'] = (
        df_merged['three_point_field_goals_attempted_opponent'] / df_merged['field_goals_attempted_opponent']
    )

    df_merged['three_variance'] = df_merged.groupby(['team_id', 'season'])['three_pct'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=2).std()
    )
    df_merged['score_variance'] = df_merged.groupby(['team_id', 'season'])['team_score'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=2).std()
    )
    df_merged['def_score_variance'] = df_merged.groupby(['team_id', 'season'])['opponent_team_score'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=2).std()
    )
    df_merged['off_eff_variance'] = df_merged.groupby(['team_id', 'season'])['off_eff'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=2).std()
    )
    df_merged['pace_variance'] = df_merged.groupby(['team_id', 'season'])['poss'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=2).std()
    )

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
    df_merged['foul_rate'] = (
        df_merged['free_throws_attempted_opponent'] / df_merged['field_goals_attempted_opponent']
    )

    df_merged.drop(columns=RAW_STAT_DROP_COLS, inplace=True, errors='ignore')
    df_merged.drop(columns=BOX_SCORE_DROP_COLS, inplace=True, errors='ignore')
    df_merged.drop(columns=OPPONENT_DROP_COLS, inplace=True, errors='ignore')

    def encode_team_home_away(row):
        if row['season_type'] in [1, 3]:
            return 2
        return 1 if str(row['team_home_away']).strip().lower() == 'home' else 0
        
    alpha = 0.15

    df_merged['team_home_away'] = df_merged.apply(encode_team_home_away, axis=1)
    df_merged['team_winner'] = df_merged['team_winner'].apply(lambda x: 1 if x is True or x == 1 else 0)
    df_merged = add_elo_ratings(df_merged, elo_ratings, elo_last_season)

    df_merged['points_last10'] = df_merged.groupby(['team_id', 'season'])['team_score'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
    )
    df_merged['opp_points_last10'] = df_merged.groupby(['team_id', 'season'])['opponent_team_score'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
    )
    df_merged['poss_last10'] = df_merged.groupby(['team_id', 'season'])['poss'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
    )
    df_merged['poss_opp_last10'] = df_merged.groupby(['team_id', 'season'])['poss_opponent'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
    )
    df_merged['last_10_efficiency'] = (
        (df_merged['points_last10'] / df_merged['poss_last10'].replace(0, np.nan) * 100)
        - (df_merged['opp_points_last10'] / df_merged['poss_opp_last10'].replace(0, np.nan) * 100)
    )
    df_merged.drop(['points_last10', 'opp_points_last10', 'poss_last10', 'poss_opp_last10'], axis=1, inplace=True)
    df_merged['games_played'] = df_merged.groupby(['team_id', 'season']).cumcount()

    for col in AVG_BASE_COLS:
        df_merged[f'{col}_avg'] = df_merged.groupby(['team_id', 'season'])[col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        df_merged[f'{col}_ewm'] = df_merged.groupby(['team_id', 'season'])[col].transform(
            lambda x: x.shift(1).ewm(alpha=alpha*2, min_periods=1).mean()
        )

    df_merged['close_game'] = df_merged['point_differential'].abs() <= 5
    group_keys = [df_merged['team_id'], df_merged['season']]

    for stat in CLOSE_GAME_STATS:
        df_merged[f'{stat}_close_game_avg'] = _historical_conditional_mean(
            df_merged[stat],
            df_merged['close_game'],
            group_keys,
        )

    expectation_inputs = ['poss_avg', 'poss_opponent_avg', 'off_eff_avg', 'def_eff_avg', 'efg_allowed_avg', 'orb_avg', 'drb_avg']
    opponent_expected = df_merged[
        ['game_id', 'season', 'season_type', 'game_date', 'team_id'] + expectation_inputs
    ].rename(
        columns={
            'team_id': 'opponent_team_id',
            **{col: f'opponent_{col}' for col in expectation_inputs},
        }
    )
    residual_rows = df_merged.reset_index().merge(
        opponent_expected,
        on=['game_id', 'season', 'season_type', 'game_date'],
        how='left',
    )
    residual_rows = residual_rows[residual_rows['team_id'] != residual_rows['opponent_team_id']].copy()
    residual_rows = residual_rows.drop_duplicates(subset='index').set_index('index')

    expected_stat_map = {
        'poss': residual_rows['opponent_poss_opponent_avg'],
        'poss_opponent': residual_rows['opponent_poss_avg'],
        'off_eff': residual_rows['opponent_def_eff_avg'],
        'def_eff': residual_rows['opponent_off_eff_avg'],
        'efg': residual_rows['opponent_efg_allowed_avg'],
        'ppp': residual_rows['opponent_def_eff_avg'] / 100.0,
        'drb': 1.0 - residual_rows['opponent_orb_avg'],
        'orb': 1.0 - residual_rows['opponent_drb_avg'],
    }

    for stat in RESIDUAL_STATS:
        df_merged[f'{stat}_residual'] = df_merged[stat] - expected_stat_map[stat]
        df_merged[f'{stat}_residual_avg'] = _historical_mean_by_group(
            df_merged[f'{stat}_residual'],
            group_keys,
        )

    drop_avg_bases = [col for col in AVG_BASE_COLS if col not in ['team_score', 'opponent_team_score']]
    df_merged.drop(columns=drop_avg_bases, inplace=True, errors='ignore')
    df_merged.drop(
        columns=['close_game'] + [f'{stat}_residual' for stat in RESIDUAL_STATS],
        inplace=True,
        errors='ignore',
    )

    df_merged['conference_strength'] = df_merged.groupby(['season', 'short_conference_name'])['net_eff_avg'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df_merged['conference_strength'].fillna(0, inplace=True)

    df_merged['team_winner_shifted'] = df_merged.groupby(['season', 'team_id'])['team_winner'].shift(1)
    df_merged['wins'] = df_merged.groupby(['season', 'team_id'])['team_winner_shifted'].transform(
        lambda x: (x == True).cumsum()
    ).fillna(0)
    df_merged['losses'] = df_merged.groupby(['season', 'team_id'])['team_winner_shifted'].transform(
        lambda x: (x == False).cumsum()
    ).fillna(0)

    df_merged['non_conf_win'] = (df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (
        df_merged['short_conference_name'] != df_merged['short_conference_name_opponent']
    )
    df_merged['non_conf_loss'] = ~(df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (
        df_merged['short_conference_name'] != df_merged['short_conference_name_opponent']
    )

    df_merged['non_conf_wins'] = df_merged.groupby(['season', 'short_conference_name'])['non_conf_win'].transform(
        lambda x: x.cumsum()
    ).fillna(0)
    df_merged['non_conf_losses'] = df_merged.groupby(['season', 'short_conference_name'])['non_conf_loss'].transform(
        lambda x: x.cumsum()
    ).fillna(0)

    df_merged['win_loss_pct'] = df_merged['wins'] / (df_merged['wins'] + df_merged['losses'])
    df_merged['non_conf_win_loss_pct'] = df_merged['non_conf_wins'] / (
        df_merged['non_conf_wins'] + df_merged['non_conf_losses']
    )
    df_merged['win_loss_pct'].fillna(0, inplace=True)
    df_merged['non_conf_win_loss_pct'].fillna(0, inplace=True)

    df_merged.drop(
        columns=[
            'wins',
            'losses',
            'non_conf_win',
            'non_conf_loss',
            'non_conf_wins',
            'non_conf_losses',
            'team_winner_shifted',
        ],
        inplace=True,
    )

    df_merged['conference_nonconf_win_pct'] = df_merged.groupby(
        ['season', 'short_conference_name']
    )['non_conf_win_loss_pct'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    df_merged['conference_nonconf_win_pct'].fillna(0, inplace=True)

    df_merged['points_for'] = df_merged.groupby(['season', 'team_id'])['team_score'].transform(
        lambda x: x.shift(1).cumsum()
    )
    df_merged['points_against'] = df_merged.groupby(['season', 'team_id'])['opponent_team_score'].transform(
        lambda x: x.shift(1).cumsum()
    )
    df_merged['points_for'].fillna(0, inplace=True)
    df_merged['points_against'].fillna(0, inplace=True)

    k = 11.5
    df_merged['pythagorean_win_pct'] = (df_merged['points_for'] ** k) / (
        (df_merged['points_for'] ** k) + (df_merged['points_against'] ** k)
    )
    df_merged['luck'] = df_merged['win_loss_pct'] - df_merged['pythagorean_win_pct']
    df_merged['luck'].fillna(0, inplace=True)

    df_merged['spread'] = df_merged['team_score'] - df_merged['opponent_team_score']
    df_merged = df_merged.fillna(0)
    df_merged.drop(
        columns=['team_score', 'opponent_team_score', 'points_for', 'points_against', 'pythagorean_win_pct'],
        inplace=True,
        errors='ignore',
    )
    season = df_merged['season'].iloc[0]
    df_merged.to_csv(f'Data/cached_data/df_{season}.csv', index=False)
    return df_merged


def _build_pair_rows(team_rows: pd.DataFrame) -> pd.DataFrame:
    pair_rows = team_rows.merge(
        team_rows,
        on=['game_id', 'season', 'season_type', 'game_date'],
        suffixes=('_a', '_b'),
    )
    pair_rows = pair_rows[pair_rows['team_id_a'] != pair_rows['team_id_b']].copy()
    pair_rows.sort_values(by=['season', 'game_date', 'game_id', 'team_id_a', 'team_id_b'], inplace=True)

    pair_rows.drop(columns=['spread_b'], inplace=True, errors='ignore')
    pair_rows.rename(columns={'spread_a': 'spread'}, inplace=True)

    pair_rows['sos'] = pair_rows.groupby(['season', 'team_id_a'])['net_eff_avg_b'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    pair_rows['sos_opp'] = pair_rows.groupby(['season', 'team_id_b'])['net_eff_avg_a'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    pair_rows['sos'].fillna(0, inplace=True)
    pair_rows['sos_opp'].fillna(0, inplace=True)

    net_eff_a = pair_rows[['team_id_a', 'game_date', 'season', 'net_eff_avg_a']].drop_duplicates(
        subset=['team_id_a', 'game_date', 'season']
    )
    net_eff_a = net_eff_a.rename(columns={'team_id_a': 'team_id'})
    net_eff_b = pair_rows[['team_id_b', 'game_date', 'season', 'net_eff_avg_b']].drop_duplicates(
        subset=['team_id_b', 'game_date', 'season']
    )
    net_eff_b = net_eff_b.rename(columns={'team_id_b': 'team_id', 'net_eff_avg_b': 'net_eff_avg'})

    grid_a = net_eff_a.sort_values(['team_id', 'season', 'game_date']).copy()
    grid_a['net_eff_avg_a'] = grid_a.groupby(['team_id', 'season'])['net_eff_avg_a'].shift(1).ffill()
    grid_a['rank'] = grid_a.groupby(['game_date', 'season'])['net_eff_avg_a'].rank(ascending=False, method='min')

    grid_b = net_eff_b.sort_values(['team_id', 'season', 'game_date']).copy()
    grid_b['net_eff_avg'] = grid_b.groupby(['team_id', 'season'])['net_eff_avg'].shift(1).ffill()
    grid_b['rank_opponent'] = grid_b.groupby(['game_date', 'season'])['net_eff_avg'].rank(
        ascending=False,
        method='min',
    )

    pair_rows = pair_rows.merge(
        grid_a[['team_id', 'game_date', 'season', 'rank']],
        left_on=['team_id_a', 'game_date', 'season'],
        right_on=['team_id', 'game_date', 'season'],
        how='left',
    )
    pair_rows.drop(columns=['team_id'], inplace=True)

    pair_rows = pair_rows.merge(
        grid_b[['team_id', 'game_date', 'season', 'rank_opponent']],
        left_on=['team_id_b', 'game_date', 'season'],
        right_on=['team_id', 'game_date', 'season'],
        how='left',
    )
    pair_rows.drop(columns=['team_id'], inplace=True)
    pair_rows.rename(columns={'rank': 'rank_a', 'rank_opponent': 'rank_b'}, inplace=True)
    pair_rows['rank_a'] = pair_rows.groupby(['season', 'team_id_a'])['rank_a'].ffill()
    pair_rows['rank_b'] = pair_rows.groupby(['season', 'team_id_b'])['rank_b'].ffill()

    pair_rows['rank_a'] = pair_rows['rank_a'].fillna(0)
    pair_rows['rank_b'] = pair_rows['rank_b'].fillna(0)

    pair_rows['threes_advantage'] = (
        pair_rows['three_attempt_rate_avg_a'] * pair_rows['three_pct_avg_a']
    ) - (
        pair_rows['allowed_three_attempt_rate_avg_b'] * pair_rows['three_pct_opponent_avg_b']
    )
    pair_rows['threes_disadvantage'] = (
        pair_rows['allowed_three_attempt_rate_avg_a'] * pair_rows['three_pct_opponent_avg_a']
    ) - (
        pair_rows['three_attempt_rate_avg_b'] * pair_rows['three_pct_avg_b']
    )
    pair_rows['two_pointers_advantage'] = (
        (1 - pair_rows['three_attempt_rate_avg_a']) * pair_rows['two_pct_avg_a']
    ) - (
        (1 - pair_rows['allowed_three_attempt_rate_avg_b']) * pair_rows['two_pct_opponent_avg_b']
    )
    pair_rows['two_pointers_disadvantage'] = (
        (1 - pair_rows['allowed_three_attempt_rate_avg_a']) * pair_rows['two_pct_opponent_avg_a']
    ) - (
        (1 - pair_rows['three_attempt_rate_avg_b']) * pair_rows['two_pct_avg_b']
    )
    pair_rows['free_throws_advantage'] = pair_rows['ftr_avg_a'] - pair_rows['foul_rate_avg_b']
    pair_rows['free_throws_disadvantage'] = pair_rows['foul_rate_avg_a'] - pair_rows['ftr_avg_b']

    pair_rows['power_rating_a'] = pair_rows['net_eff_avg_a'] + pair_rows['sos']
    pair_rows['power_rating_b'] = pair_rows['net_eff_avg_b'] + pair_rows['sos_opp']

    pair_rows['off_vs_def'] = pair_rows['off_eff_avg_a'] - pair_rows['def_eff_avg_b']
    pair_rows['def_vs_off'] = pair_rows['off_eff_avg_b'] - pair_rows['def_eff_avg_a']
    pair_rows['tov_vs_stl'] = pair_rows['tov_avg_a'] - pair_rows['stl_rate_avg_b']
    pair_rows['stl_vs_tov'] = pair_rows['tov_avg_b'] - pair_rows['stl_rate_avg_a']
    pair_rows['orb_vs_drb'] = pair_rows['orb_avg_a'] - pair_rows['drb_avg_b']
    pair_rows['drb_vs_orb'] = pair_rows['orb_avg_b'] - pair_rows['drb_avg_a']
    pair_rows['pace_diff'] = pair_rows['poss_avg_a'] - pair_rows['poss_avg_b']
    pair_rows['exp_poss'] = (pair_rows['poss_avg_a'] + pair_rows['poss_avg_b']) / 2
    pair_rows['efg_vs_efg_allowed'] = pair_rows['efg_avg_a'] - pair_rows['efg_allowed_avg_b']
    pair_rows['efg_allowed_vs_efg'] = pair_rows['efg_avg_b'] - pair_rows['efg_allowed_avg_a']
    pair_rows['margin_estimate'] = (
        (pair_rows['net_eff_avg_a'] - pair_rows['net_eff_avg_b']) * pair_rows['exp_poss']
    ) / 100

    rank_b = pd.to_numeric(pair_rows['rank_b'], errors='coerce')
    location = pair_rows['team_home_away_a']
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
    pair_rows['quad_score_raw'] = np.where(pair_rows['team_winner_a'] == 1, quad_win_score, quad_loss_score)
    pair_rows['quad_score'] = (
        pair_rows.groupby(['season', 'team_id_a'])['quad_score_raw']
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    pair_rows.drop(columns=['quad_score_raw'], inplace=True)
    pair_rows['quad_score'] = pair_rows.groupby(['season', 'team_id_a'])['quad_score'].ffill().fillna(0)

    pair_rows['games_played'] = np.minimum(
        pair_rows['games_played_a'],
        pair_rows['games_played_b'],
    )

    return pair_rows


def _flatten_pair_rows(pair_rows: pd.DataFrame, *, drop_missing: bool = True) -> pd.DataFrame:
    df_final = pair_rows.copy()
    df_final.drop(
        columns=[
            'team_winner_b',
            'team_home_away_b',
            'short_conference_name_a',
            'short_conference_name_opponent_a',
            'short_conference_name_b',
            'short_conference_name_opponent_b',
            'team_name_a',
            'team_name_b',
            'games_played_a',
            'games_played_b',
        ],
        inplace=True,
        errors='ignore',
    )

    df_final.rename(
        columns={
            'team_home_away_a': 'team_home_away',
            'team_winner_a': 'team_winner',
        },
        inplace=True,
    )
    df_final.drop(columns=['team_id_a', 'team_id_b'], inplace=True, errors='ignore')

    for col_a in [col for col in df_final.columns if col.endswith('_a')]:
        col_b = col_a[:-2] + '_b'
        if col_b in df_final.columns:
            df_final[col_a[:-2] + '_diff'] = df_final[col_a] - df_final[col_b]
            df_final.drop([col_a, col_b], axis=1, inplace=True)

    if drop_missing:
        null_counts = df_final.isnull().sum()
        most_nulls = null_counts.sort_values(ascending=False).head(5)
        df_final = df_final.dropna()

    return df_final


def _attach_matchup_context(team_rows: pd.DataFrame) -> pd.DataFrame:
    pair_rows = _build_pair_rows(team_rows)
    context = pair_rows[
        ['game_id', 'season', 'season_type', 'game_date', 'team_id_a', 'sos', 'adj_sos', 'rank_a', 'quad_score']
    ].rename(
        columns={
            'team_id_a': 'team_id',
            'rank_a': 'rank',
        }
    )
    return team_rows.merge(
        context,
        on=['game_id', 'season', 'season_type', 'game_date', 'team_id'],
        how='left',
    )


def process_all_games(
    games_df: pd.DataFrame,
    conference_mapping: pd.DataFrame,
    elo_ratings=None,
    elo_last_season=None,
) -> pd.DataFrame:
    print("Calculating game-level features...")
    team_rows = _prepare_team_game_rows(
        games_df,
        conference_mapping,
        elo_ratings=elo_ratings,
        elo_last_season=elo_last_season,
    )
    print("Getting matchup features...")
    pair_rows = _build_pair_rows(team_rows)
    return _flatten_pair_rows(pair_rows, drop_missing=True)


def build_team_feature_rows(season: int, conference_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Return the per-team pre-game rows that feed `process_all_games`, plus the
    team-side matchup context needed to mirror final matchup features later.
    """
    df_all = _concat_all_seasons_games()
    df = df_all[df_all['season'] <= season].copy()
    if df.empty:
        return df

    team_rows = _prepare_team_game_rows(df, conference_mapping, keep_team_name=True)
    team_rows = _attach_matchup_context(team_rows)
    team_rows.sort_values(by=['game_date', 'game_id', 'team_id'], inplace=True)
    return team_rows[team_rows['season'] == season].copy()


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

    def safe_scalar_divide(num: float, den: float) -> float:
        if den == 0 or pd.isna(den):
            return np.nan
        return num / den

    exp_poss = (val(left_snapshot, 'poss_avg') + val(right_snapshot, 'poss_avg')) / 2

    left_date = pd.to_datetime(left_snapshot.get('game_date', None), errors='coerce')
    right_date = pd.to_datetime(right_snapshot.get('game_date', None), errors='coerce')

    off_eff_avg_a = val(left_snapshot, 'off_eff_avg')
    def_eff_avg_a = val(left_snapshot, 'def_eff_avg')
    off_eff_avg_b = val(right_snapshot, 'off_eff_avg')
    def_eff_avg_b = val(right_snapshot, 'def_eff_avg')

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

    feature_row = {
        'season': int(season),
        'season_type': int(season_type),
        'team_home_away': int(team_home_away),
        'sos': val(left_snapshot, 'sos', 0.0),
        'sos_opp': val(right_snapshot, 'sos', 0.0),
        'threes_advantage': (three_attempt_rate_avg_a * three_pct_avg_a)
        - (allowed_three_attempt_rate_avg_b * three_pct_opponent_avg_b),
        'threes_disadvantage': (allowed_three_attempt_rate_avg_a * three_pct_opponent_avg_a)
        - (three_attempt_rate_avg_b * three_pct_avg_b),
        'two_pointers_advantage': ((1 - three_attempt_rate_avg_a) * two_pct_avg_a)
        - ((1 - allowed_three_attempt_rate_avg_b) * two_pct_opponent_avg_b),
        'two_pointers_disadvantage': ((1 - allowed_three_attempt_rate_avg_a) * two_pct_opponent_avg_a)
        - ((1 - three_attempt_rate_avg_b) * two_pct_avg_b),
        'free_throws_advantage': ftr_avg_a - foul_rate_avg_b,
        'free_throws_disadvantage': foul_rate_avg_a - ftr_avg_b,
        'adj_sos': val(left_snapshot, 'adj_sos', 0.0),
        'adj_sos_opp': val(right_snapshot, 'adj_sos', 0.0),
        'off_vs_def': off_eff_avg_a - def_eff_avg_b,
        'def_vs_off': off_eff_avg_b - def_eff_avg_a,
        'tov_vs_stl': val(left_snapshot, 'tov_avg') - val(right_snapshot, 'stl_rate_avg'),
        'stl_vs_tov': val(right_snapshot, 'tov_avg') - val(left_snapshot, 'stl_rate_avg'),
        'orb_vs_drb': val(left_snapshot, 'orb_avg') - val(right_snapshot, 'drb_avg'),
        'drb_vs_orb': val(right_snapshot, 'orb_avg') - val(left_snapshot, 'drb_avg'),
        'pace_diff': val(left_snapshot, 'poss_avg') - val(right_snapshot, 'poss_avg'),
        'exp_poss': exp_poss,
        'efg_vs_efg_allowed': val(left_snapshot, 'efg_avg') - val(right_snapshot, 'efg_allowed_avg'),
        'efg_allowed_vs_efg': val(right_snapshot, 'efg_avg') - val(left_snapshot, 'efg_allowed_avg'),
        'margin_estimate': ((val(left_snapshot, 'net_eff_avg') - val(right_snapshot, 'net_eff_avg')) * exp_poss) / 100,
        'quad_score': val(left_snapshot, 'quad_score', 0.0),
        'three_variance_diff': val(left_snapshot, 'three_variance') - val(right_snapshot, 'three_variance'),
        'score_variance_diff': val(left_snapshot, 'score_variance') - val(right_snapshot, 'score_variance'),
        'def_score_variance_diff': val(left_snapshot, 'def_score_variance') - val(right_snapshot, 'def_score_variance'),
        'off_eff_variance_diff': val(left_snapshot, 'off_eff_variance') - val(right_snapshot, 'off_eff_variance'),
        'pace_variance_diff': val(left_snapshot, 'pace_variance') - val(right_snapshot, 'pace_variance'),
        'elo_diff': val(left_snapshot, 'elo') - val(right_snapshot, 'elo'),
        'last_10_efficiency_diff': val(left_snapshot, 'last_10_efficiency') - val(right_snapshot, 'last_10_efficiency'),
    }

    for base_col in AVG_BASE_COLS:
        feature_row[f'{base_col}_avg_diff'] = val(left_snapshot, f'{base_col}_avg') - val(
            right_snapshot,
            f'{base_col}_avg',
        )
        feature_row[f'{base_col}_rolling_5_diff'] = val(left_snapshot, f'{base_col}_rolling_5') - val(
            right_snapshot,
            f'{base_col}_rolling_5',
        )

    for stat in CLOSE_GAME_STATS:
        feature_row[f'{stat}_close_game_avg_diff'] = val(left_snapshot, f'{stat}_close_game_avg') - val(
            right_snapshot,
            f'{stat}_close_game_avg',
        )

    for stat in RESIDUAL_STATS:
        feature_row[f'{stat}_residual_avg_diff'] = val(left_snapshot, f'{stat}_residual_avg') - val(
            right_snapshot,
            f'{stat}_residual_avg',
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
    feature_row['rank_diff'] = val(left_snapshot, 'rank') - val(right_snapshot, 'rank')
    feature_row['power_rating_diff'] = (
        val(left_snapshot, 'net_eff_avg') + val(left_snapshot, 'sos')
        - val(right_snapshot, 'net_eff_avg') - val(right_snapshot, 'sos')
    )

    return pd.DataFrame([feature_row]).replace([np.inf, -np.inf], np.nan).fillna(-100)
