import random
from dataclasses import dataclass
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# -----------------------------
# TEAM
# -----------------------------

@dataclass
class Team:
    name: str
    seed: int
    season: int | None = None


# -----------------------------
# GAME NODE
# -----------------------------

class Game:

    def __init__(self, team_a=None, team_b=None,
                 winner_from_a=None, winner_from_b=None):

        self.team_a = team_a
        self.team_b = team_b

        self.winner_from_a = winner_from_a
        self.winner_from_b = winner_from_b

        self.winner = None


# -----------------------------
# RESOLVE PARTICIPANTS
# -----------------------------

def resolve(team, source_game):

    if team is not None:
        return team

    if source_game is not None:
        return source_game.winner

    return None


# -----------------------------
# PLAY GAME
# -----------------------------

def play_game(game, prob_lookup=None):

    team_a = resolve(game.team_a, game.winner_from_a)
    team_b = resolve(game.team_b, game.winner_from_b)

    if team_a is None:
        game.winner = team_b
        return

    if team_b is None:
        game.winner = team_a
        return

    p = 0.5

    if prob_lookup:
        p = prob_lookup(team_a, team_b)
    else:
        p = _predict_win_prob(team_a, team_b)

    game.winner = team_a if random.random() < p else team_b


# -----------------------------
# MATCHUP PREDICTION (MODEL)
# -----------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GAME_PRED_DIR = Path(__file__).resolve().parent
_DATA_DIR = _PROJECT_ROOT / "Data"
_MODEL_PATH = _GAME_PRED_DIR / "models" / "lgbm_winner_model.joblib"

_MODEL = None
_MODEL_FEATURES = None
_CONFERENCE_MAPPING = None
_TEAM_FEATURE_CACHE = {}
_TEAM_SNAPSHOT_CACHE = {}

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
    'ppp',
    'two_pct',
    'two_pct_opponent',
    'point_differential',
    'assist_rate',
    'assist_to_fg',
    'block_rate',
    'lead_vs_outcome',
    'fast_break_pct',
    'points_off_turnover_pct',
]


def align_features_for_model(df: pd.DataFrame, feature_names) -> pd.DataFrame:
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype(int)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    return data.reindex(columns=list(feature_names), fill_value=0)


def _safe_divide(num, den):
    den = den.replace(0, np.nan)
    return num / den


def load_conference_mapping() -> pd.DataFrame:
    mapping_path = _DATA_DIR / 'kenpom' / 'REF _ NCAAM Conference and ESPN Team Name Mapping.csv'
    conference_mapping = pd.read_csv(
        mapping_path,
        usecols=['Conference', 'Mapped ESPN Team Name']
    ).dropna()
    conference_mapping['Mapped ESPN Team Name'] = conference_mapping['Mapped ESPN Team Name'].replace(
        {'Hawaii': "Hawai'i", 'St. Francis (PA)': 'Saint Francis', 'San Jose State': 'San Jose State'}
    )
    conference_mapping['team_location_key'] = (
        conference_mapping['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()
    )
    conference_mapping = conference_mapping.drop_duplicates(subset=['team_location_key'])
    conference_mapping = conference_mapping.rename(columns={'Conference': 'short_conference_name'})
    return conference_mapping[['team_location_key', 'short_conference_name']]


def build_team_feature_rows(season: int, conference_mapping: pd.DataFrame) -> pd.DataFrame:
    games_path = _DATA_DIR / 'game_results' / f'games_{season}.csv'
    if not games_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(games_path)
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
    df_merged['lead_vs_outcome'] = df_merged['largest_lead'] - df_merged['point_differential']
    df_merged['fast_break_pct'] = _safe_divide(df_merged['fast_break_points'], df_merged['team_score'])
    df_merged['points_off_turnover_pct'] = _safe_divide(df_merged['turnover_points'], df_merged['team_score'])

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
        'team_home_away_opponent', 'team_score_opponent', 'team_winner_opponent',
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

    df_merged['home_off_eff'] = df_merged.groupby('team_id').apply(
        lambda g: g.loc[g['team_home_away'] == 1, 'off_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    df_merged['home_def_eff'] = df_merged.groupby('team_id').apply(
        lambda g: g.loc[g['team_home_away'] == 1, 'def_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    df_merged['away_off_eff'] = df_merged.groupby('team_id').apply(
        lambda g: g.loc[g['team_home_away'] == 0, 'off_eff'].shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    df_merged['away_def_eff'] = df_merged.groupby('team_id').apply(
        lambda g: g.loc[g['team_home_away'] == 0, 'def_eff'].shift(1).expanding().mean()
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
        _safe_divide(df_merged['points_last10'], df_merged['poss_last10']) * 100
        - _safe_divide(df_merged['opp_points_last10'], df_merged['poss_opp_last10']) * 100
    )
    df_merged.drop(columns=['points_last10', 'opp_points_last10', 'poss_last10', 'poss_opp_last10'], inplace=True)

    for col in AVG_BASE_COLS:
        df_merged[f'{col}_avg'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).expanding().mean())
        df_merged[f'{col}_rolling_5'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).rolling(5).mean())

    df_merged['is_early_season'] = df_merged.isna().any(axis=1).astype(int)
    drop_avg_bases = [c for c in AVG_BASE_COLS if c not in ['team_score', 'opponent_team_score']]
    df_merged.drop(columns=drop_avg_bases, inplace=True, errors='ignore')

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

    df_merged['win_loss_pct'] = _safe_divide(df_merged['wins'], (df_merged['wins'] + df_merged['losses']))
    df_merged['non_conf_win_loss_pct'] = _safe_divide(
        df_merged['non_conf_wins'], (df_merged['non_conf_wins'] + df_merged['non_conf_losses'])
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

    sos_base = df_merged[['game_id', 'season', 'season_type', 'game_date', 'team_id', 'net_eff_avg']].copy()
    sos_pairs = sos_base.merge(
        sos_base, on=['game_id', 'season', 'season_type', 'game_date'], suffixes=('_a', '_b')
    )
    sos_pairs = sos_pairs[sos_pairs['team_id_a'] != sos_pairs['team_id_b']]
    sos_pairs = sos_pairs.sort_values(by=['game_date', 'game_id'], ascending=True)
    sos_pairs['sos'] = sos_pairs.groupby('team_id_a')['net_eff_avg_b'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    sos_map = sos_pairs[['game_id', 'team_id_a', 'sos']].rename(columns={'team_id_a': 'team_id'})
    df_merged = df_merged.merge(sos_map, on=['game_id', 'team_id'], how='left')

    return df_merged


def build_matchup_feature_row(
    left_snapshot: pd.Series,
    right_snapshot: pd.Series,
    season: int,
    season_type: int,
    team_home_away: int,
) -> pd.DataFrame:
    def val(row: pd.Series, column: str, default: float = -100.0) -> float:
        if column not in row or pd.isna(row[column]):
            return default
        return float(row[column])

    exp_poss = (val(left_snapshot, 'poss_avg') + val(right_snapshot, 'poss_avg')) / 2

    feature_row = {
        'season': int(season),
        'season_type': int(season_type),
        'team_home_away': int(team_home_away),
        'is_early_season': int(val(left_snapshot, 'is_early_season', 1.0)),
        'sos': val(left_snapshot, 'sos'),
        'sos_opp': val(right_snapshot, 'sos'),
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

    return pd.DataFrame([feature_row]).replace([np.inf, -np.inf], np.nan).fillna(-100)


def _load_model():
    global _MODEL, _MODEL_FEATURES
    if _MODEL is None:
        _MODEL = joblib.load(_MODEL_PATH)
        _MODEL_FEATURES = list(_MODEL.feature_name_)
    return _MODEL, _MODEL_FEATURES


def _get_conference_mapping():
    global _CONFERENCE_MAPPING
    if _CONFERENCE_MAPPING is None:
        _CONFERENCE_MAPPING = load_conference_mapping()
    return _CONFERENCE_MAPPING


def _get_team_feature_data_for_season(season: int) -> pd.DataFrame:
    if season not in _TEAM_FEATURE_CACHE:
        _TEAM_FEATURE_CACHE[season] = build_team_feature_rows(season, _get_conference_mapping())
    return _TEAM_FEATURE_CACHE[season]


def _get_latest_team_snapshot(team_name: str, season: int) -> pd.Series | None:
    cache_key = (season, team_name)
    if cache_key in _TEAM_SNAPSHOT_CACHE:
        return _TEAM_SNAPSHOT_CACHE[cache_key]

    season_features = _get_team_feature_data_for_season(season)
    if season_features.empty:
        _TEAM_SNAPSHOT_CACHE[cache_key] = None
        return None

    team_rows = season_features[season_features['team_name'] == team_name].copy()
    if team_rows.empty:
        _TEAM_SNAPSHOT_CACHE[cache_key] = None
        return None

    team_rows = team_rows.sort_values(by=['game_date', 'game_id'], ascending=True)
    snapshot = team_rows.iloc[-1].copy().replace([np.inf, -np.inf], np.nan).fillna(-100)
    _TEAM_SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


def _latest_season_from_data(default: int = 2026) -> int:
    data_dir = _DATA_DIR / "game_results"
    if not data_dir.exists():
        return default
    seasons = []
    for path in data_dir.glob("games_*.csv"):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            seasons.append(int(parts[-1]))
    return max(seasons) if seasons else default


def _resolve_season(team_a: Team, team_b: Team) -> int:
    if team_a.season is not None:
        return int(team_a.season)
    if team_b.season is not None:
        return int(team_b.season)
    return _latest_season_from_data()


def _predict_win_prob(team_a: Team, team_b: Team) -> float:
    # Modified logic for neutral court - average prediction and reversed prediction
    model, feature_names = _load_model()
    season = _resolve_season(team_a, team_b)
    left_snapshot = _get_latest_team_snapshot(team_a.name, season)
    right_snapshot = _get_latest_team_snapshot(team_b.name, season)

    if left_snapshot is None or right_snapshot is None:
        return 0.5

    season_type = 3
    team_home_away = 2

    # Normal direction
    processed_data = build_matchup_feature_row(
        left_snapshot,
        right_snapshot,
        season=season,
        season_type=season_type,
        team_home_away=team_home_away,
    )
    ordered_data = align_features_for_model(processed_data, feature_names)
    proba = model.predict_proba(ordered_data)[0]
    proba_a = float(proba[1])

    # Reverse direction (swap home/away roles)
    processed_data_inv = build_matchup_feature_row(
        right_snapshot,
        left_snapshot,
        season=season,
        season_type=season_type,
        team_home_away=team_home_away,
    )
    ordered_data_inv = align_features_for_model(processed_data_inv, feature_names)
    proba_inv = model.predict_proba(ordered_data_inv)[0]
    proba_b = float(proba_inv[1])

    # Because in the reverse, we get probability of team_b beating team_a, so we want probability of team_a: 1 - proba_b
    p_avg = (proba_a + (1.0 - proba_b)) / 2.0
    return p_avg

# -----------------------------
# BUILD BRACKET
# -----------------------------

def build_sec_bracket(seeds):
    """
    seeds = list of 16 Team objects sorted by seed
    """

    # Round 1
    g1 = Game(seeds[8], seeds[15])   # 9 vs 16
    g2 = Game(seeds[11], seeds[12])  # 12 vs 13
    g3 = Game(seeds[9], seeds[14])   # 10 vs 15
    g4 = Game(seeds[10], seeds[13])  # 11 vs 14

    # Round 2 (5-8 enter)
    g5 = Game(seeds[7], None, None, g1)  # 8 vs winner g1
    g6 = Game(seeds[4], None, None, g2)  # 5 vs winner g2
    g7 = Game(seeds[6], None, None, g3)  # 7 vs winner g3
    g8 = Game(seeds[5], None, None, g4)  # 6 vs winner g4

    # Quarterfinals (1-4 enter)
    g9  = Game(seeds[0], None, None, g5)
    g10 = Game(seeds[3], None, None, g6)
    g11 = Game(seeds[1], None, None, g7)
    g12 = Game(seeds[2], None, None, g8)

    # Semifinals
    g13 = Game(None, None, g9, g10)
    g14 = Game(None, None, g11, g12)

    # Championship
    g15 = Game(None, None, g13, g14)

    return [
        g1,g2,g3,g4,
        g5,g6,g7,g8,
        g9,g10,g11,g12,
        g13,g14,
        g15
    ]

# -----------------------------
# RUN TOURNAMENT
# -----------------------------

def run_bracket(games, prob_lookup=None):

    for game in games:
        play_game(game, prob_lookup)

    return games[-1].winner


# -----------------------------
# MONTE CARLO SIMULATION
# -----------------------------

def simulate_tournament(teams, prob_lookup=None, sims=1000):

    if len(teams) != 16:
        raise ValueError("simulate_tournament expects exactly 16 teams.")

    n = len(teams)
    prob_matrix = np.full((n, n), 0.5, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if prob_lookup:
                prob_matrix[i, j] = float(prob_lookup(teams[i], teams[j]))
            else:
                prob_matrix[i, j] = float(_predict_win_prob(teams[i], teams[j]))

    rng = np.random.default_rng()

    def play_match(a_idx, b_idx):
        p = prob_matrix[a_idx, b_idx]
        r = rng.random(size=a_idx.shape[0])
        return np.where(r < p, a_idx, b_idx)

    g1 = play_match(np.full(sims, 8), np.full(sims, 15))
    g2 = play_match(np.full(sims, 11), np.full(sims, 12))
    g3 = play_match(np.full(sims, 9), np.full(sims, 14))
    g4 = play_match(np.full(sims, 10), np.full(sims, 13))

    g5 = play_match(np.full(sims, 7), np.full(sims, 8))
    g6 = play_match(np.full(sims, 4), np.full(sims, 11))
    g7 = play_match(np.full(sims, 6), np.full(sims, 14))
    g8 = play_match(np.full(sims, 5), g4)

    g9 = play_match(np.full(sims, 0), g5)
    g10 = play_match(np.full(sims, 3), g6)
    g11 = play_match(np.full(sims, 1), g7)
    g12 = play_match(np.full(sims, 2), g8)

    g13 = play_match(g9, g10)
    g14 = play_match(g11, g12)
    winners = play_match(g13, g14)

    for i, w in enumerate(winners, start=1):
        print(f"Simulation {i} of {sims}: {teams[w].name}")

    counts = np.bincount(winners, minlength=n).astype(float)
    results = {teams[i].name: counts[i] / sims for i in range(n) if counts[i] > 0}

    return results


# -----------------------------
# EXAMPLE USAGE
# -----------------------------

if __name__ == "__main__":

    teams = [
        Team("Florida",1),
        Team("Alabama",2),
        Team("Arkansas",3),
        Team("Vanderbilt",4),
        Team("Tennessee",5),
        Team("Texas A&M",6),
        Team("Georgia",7),
        Team("Missouri",8),
        Team("Kentucky",9),
        Team("Texas",10),
        Team("Oklahoma",11),
        Team("Auburn",12),
        Team("Mississippi State",13),
        Team("South Carolina",14),
        Team("Ole Miss",15),
        Team("LSU",16),
    ]

    results = simulate_tournament(teams, None, sims=100000)

    for team, prob in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{team}: {prob*100:.1f}%")