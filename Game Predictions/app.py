import sys
import os
import csv
import random
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QPushButton, QFrame, QGridLayout, QSizePolicy,
                            QMessageBox)
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPainter, QFont, QIcon, QPen
from PyQt5.QtCore import Qt, QSize, QRect, QTimer, QUrl, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtNetwork import QNetworkRequest, QNetworkAccessManager, QNetworkReply
import joblib
import pandas as pd
import numpy as np
import data_processing as dp
import model_ensemble as ensemble
info = pd.read_csv('Data/game_results/games_2026.csv')
info['team_location_key'] = info['team_location'].astype(str).str.strip().str.lower()


def _team_key(team_name: str) -> str:
    return str(team_name).strip().lower()

def get_team_color(team):
    team_data = info[info['team_location_key'] == _team_key(team)]
    if team_data.empty:
        return DEFAULT_TEAM_COLOR
    hex_color = str(team_data['team_color'].iloc[0])
    if is_dark_color(hex_color):
        hex_color = str(team_data['team_alternate_color'].iloc[0])
    return "#" + hex_color, "#FFFFFF"

def is_dark_color(hex_color, threshold=30):
    """
    Check if a hex color is black or very dark
    
    Parameters:
    -----------
    hex_color : str
        Hex color code (with or without # prefix)
    threshold : int
        RGB value below which a color is considered dark (0-255)
        
    Returns:
    --------
    bool
        True if the color is dark, False otherwise
    """
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        # Invalid hex color
        return False
    
    # Check if all RGB values are below threshold
    return all(value < threshold for value in (r, g, b))

def get_logo_url(team_name):
    logo_url = info[info['team_location_key'] == _team_key(team_name)]
    if logo_url.empty:
        return ""
    url = logo_url['team_logo'].iloc[0]        
    return url

# Default team colors for teams not in the dictionary
DEFAULT_TEAM_COLOR = ('#333333', '#FFFFFF')  # Dark gray and white

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
    'lead_vs_outcome',
    'fast_break_pct',
    'points_off_turnover_pct',
    'three_pct',
    'three_pct_opponent',
    'three_attempt_rate',
    'allowed_three_attempt_rate',
]


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


def load_conference_mapping() -> pd.DataFrame:
    conference_mapping = pd.read_csv(
        'Data/kenpom/REF _ NCAAM Conference and ESPN Team Name Mapping.csv',
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


def build_team_feature_rows(season: int, conference_mapping: pd.DataFrame) -> pd.DataFrame:
    games_path = Path(f'Data/game_results/games_{season}.csv')
    if not games_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(games_path)
    if df.empty:
        return df

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

    # League-wide averages up to the prior day (per season)
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
        .transform(lambda s: s.shift(1).expanding().mean())
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
    df_merged['three_variance'] = df_merged.groupby('team_id')['three_pct'].transform(
        lambda x: x.shift(1).rolling(10).std()
    )
    df_merged['score_variance'] = df_merged.groupby('team_id')['team_score'].transform(
        lambda x: x.shift(1).rolling(10).std()
    )
    df_merged['def_score_variance'] = df_merged.groupby('team_id')['opponent_team_score'].transform(
        lambda x: x.shift(1).rolling(10).std()
    )
    df_merged['off_eff_variance'] = df_merged.groupby('team_id')['off_eff'].transform(
        lambda x: x.shift(1).rolling(10).std()
    )
    df_merged['pace_variance'] = df_merged.groupby('team_id')['poss'].transform(
        lambda x: x.shift(1).rolling(10).std()
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
    df_merged['lead_vs_outcome'] = df_merged['largest_lead'] - df_merged['point_differential']
    df_merged['fast_break_pct'] = _safe_divide(df_merged['fast_break_points'], df_merged['team_score'])
    df_merged['points_off_turnover_pct'] = _safe_divide(df_merged['turnover_points'], df_merged['team_score'])
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

    df_merged = df_merged.merge(
        league_daily[['season', 'game_date', 'league_avg_off_eff', 'league_avg_def_eff']],
        on=['season', 'game_date'],
        how='left'
    )

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

    matchup_base = df_merged.merge(
        df_merged,
        on=['game_id', 'season', 'season_type', 'game_date'],
        suffixes=('_a', '_b')
    )
    matchup_base = matchup_base[matchup_base['team_id_a'] != matchup_base['team_id_b']]
    matchup_base = matchup_base.merge(
        league_daily[['season', 'game_date', 'league_avg_off_eff', 'league_avg_def_eff']],
        on=['season', 'game_date'],
        how='left'
    )
    matchup_base = matchup_base.sort_values(by=['game_date', 'game_id'], ascending=True)

    matchup_base['sos'] = matchup_base.groupby('team_id_a')['net_eff_avg_b'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    matchup_base['sos_opp'] = matchup_base.groupby('team_id_b')['net_eff_avg_a'].transform(
        lambda x: x.shift(1).expanding().mean()
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
        lambda x: x.shift(1).expanding().mean()
    )
    matchup_base['adj_sos_opp'] = matchup_base.groupby('team_id_b')['adj_net_eff_a'].transform(
        lambda x: x.shift(1).expanding().mean()
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
    df_merged = df_merged.merge(sos_map, on=['game_id', 'team_id'], how='left')

    return df_merged


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


class TeamStats:
    def __init__(self, team='', year='', conference='', wins=0, losses=0, points=0.0, 
                opp_points=0.0, margin_of_victory=0.0, strength_of_schedule=0.0, 
                offensive_srs=0.0, defensive_srs=0.0, simple_rating_system=0.0,
                offensive_rating=0.0, defensive_rating=0.0, net_rating=0.0):
        self.team = team
        self.year = year
        self.conference = conference
        self.wins = wins
        self.losses = losses
        self.points = points
        self.opp_points = opp_points
        self.margin_of_victory = margin_of_victory
        self.strength_of_schedule = strength_of_schedule
        self.offensive_srs = offensive_srs
        self.defensive_srs = defensive_srs
        self.simple_rating_system = simple_rating_system
        self.offensive_rating = offensive_rating
        self.defensive_rating = defensive_rating
        self.net_rating = net_rating

class TeamSideWidget(QWidget):
    def __init__(self, is_left, parent=None, show_year_selector=True):
        super().__init__(parent)
        self.is_left = is_left
        self.parent_app = parent
        self.show_year_selector = show_year_selector
        self.team_colors = DEFAULT_TEAM_COLOR
        
        # Set side colors
        self.side_colors = (QColor("#2E2C2B"), QColor("#000000")) if is_left else (QColor("#6A6260"), QColor("#86807F"))
        self.win_gradient = False
        self.win_probability = 0.0
        
        # Border animation properties
        self.border_opacity = 0.0
        self.border_animation_timer = QTimer(self)
        self.border_animation_timer.timeout.connect(self.animate_border)
        self.border_animation_direction = 1  # 1 for increasing, -1 for decreasing
        
        # Main layout with more spacing for centered appearance
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 30, 20, 30)  # Add more margin around elements
        self.layout.setSpacing(20)  # Increase spacing between elements
        self.setLayout(self.layout)
        
        # Rest of your initialization code...
        # Top section layout
        top_layout = QHBoxLayout()
        
        # Conference logo - moved to outer corner and MUCH larger
        self.conf_logo_label = QLabel()
        self.conf_logo_label.setAlignment(Qt.AlignCenter)
        self.conf_logo_label.setFixedSize(120, 120)  # Much larger conference logo
        self.conf_logo_label.setStyleSheet("background-color: transparent; color: white; font-size: 14px;")
        
        # Year selector
        year_layout = QHBoxLayout()
        year_label = QLabel("Season:")
        year_label.setStyleSheet("background-color: transparent; color: white; font-weight: bold;")
        self.year_label = year_label
        self.year_combo = QComboBox()
        self.year_combo.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: #333333;
                border: 1px solid white;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: #333333;
            }
        """)
        year_layout.addWidget(year_label)
        year_layout.addWidget(self.year_combo)

        if not self.show_year_selector:
            self.year_label.hide()
            self.year_combo.hide()
        
        # Position based on side (left or right)
        if self.is_left:
            # Left side: keep season selector on the left edge.
            top_layout.addLayout(year_layout)
            top_layout.addStretch(1)
            top_layout.addWidget(self.conf_logo_label)
        else:
            top_layout.addStretch(1)
            top_layout.addWidget(self.conf_logo_label)
            
        self.layout.addLayout(top_layout)
        
        # Add stretch to push logo to center vertically
        self.layout.addStretch(1)
        
        # Team logo container - make it larger
        logo_container = QHBoxLayout()
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setMinimumSize(400, 400)  # Increased size for larger logos
        self.logo_label.setStyleSheet("background-color: transparent;")
        
        logo_container.addStretch(1)
        logo_container.addWidget(self.logo_label)
        logo_container.addStretch(1)
        
        self.layout.addLayout(logo_container)
        
        # Win probability label with enhanced style for better visibility
        self.probability_label = QLabel("")
        self.probability_label.setAlignment(Qt.AlignCenter)
        self.probability_label.setStyleSheet("""
            color: white; 
            font-size: 18px; 
            font-weight: bold; 
            background-color: transparent;
            padding: 10px;
        """)
        
        # Add stretch to push logo to center vertically
        self.layout.addStretch(1)
        
        # Team selector
        self.team_combo = QComboBox()
        self.team_combo.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: #333333;
                border: 1px solid white;
                padding: 5px;
                font-size: 16px;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: #333333;
            }
        """)
        self.layout.addWidget(self.team_combo)
        
        # Connect signals
        if self.show_year_selector:
            self.year_combo.currentTextChanged.connect(self.year_changed)
        self.team_combo.currentTextChanged.connect(self.team_changed)

    def animate_border(self):
        # Update border opacity based on direction
        self.border_opacity += 0.05 * self.border_animation_direction
        
        # Reverse direction at limits
        if self.border_opacity >= 1.0:
            self.border_opacity = 1.0
            self.border_animation_direction = -1
        elif self.border_opacity <= 0.3:
            self.border_opacity = 0.3
            self.border_animation_direction = 1
            
        # Update the widget to repaint with new opacity
        self.update()
    
    def set_win_gradient(self, is_winner, probability):
        self.win_gradient = is_winner
        self.win_probability = probability
        
        # Set probability text with enhanced style for winner
        if is_winner:
            self.probability_label.setText(f"Win Probability: {probability:.1f}%")
            self.probability_label.setStyleSheet("""
                color: gold; 
                font-size: 24px; 
                font-weight: bold; 
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 12px;
                padding: 10px;
            """)
            
            # Start border animation for winner
            if not self.border_animation_timer.isActive():
                self.border_opacity = 0.3  # Starting opacity
                self.border_animation_timer.start(50)  # 50ms interval for smooth animation
        else:
            self.probability_label.setText(f"Win Probability: {probability:.1f}%")
            self.probability_label.setStyleSheet("""
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                background-color: transparent;
                padding: 10px;
            """)
            
            # Stop border animation for non-winner
            if self.border_animation_timer.isActive():
                self.border_animation_timer.stop()
        
        # Update the background
        self.update()
    
    def paintEvent(self, event):
        # Create painter
        painter = QPainter(self)
        
        # Create gradient background
        gradient = QLinearGradient()
        
        # Get team primary and secondary colors
        primary_color = QColor(self.team_colors[0])
        secondary_color = QColor(self.team_colors[1])
        
        # Adjust opacity to make colors less intense
        primary_color.setAlpha(180)
        secondary_color.setAlpha(150)
        
        # Set gradient direction based on side
        if self.is_left:
            gradient.setStart(self.width(), self.height() / 2)
            gradient.setFinalStop(0, self.height() / 2)
        else:
            gradient.setStart(0, self.height() / 2)
            gradient.setFinalStop(self.width(), self.height() / 2)
        
        # Set gradient colors
        black_overlay = QColor("#000000")
        black_overlay.setAlpha(180)
        
        if self.is_left:
            gradient.setColorAt(0, primary_color)
            gradient.setColorAt(1, black_overlay)
        else:
            gradient.setColorAt(0, black_overlay)
            gradient.setColorAt(1, primary_color)
        
        # Fill background with gradient
        painter.fillRect(self.rect(), gradient)
        
        # Draw animated border for winner
        if self.win_gradient:
            # Create a golden border color with dynamic opacity
            border_color = QColor(255, 215, 0, int(255 * self.border_opacity))  # Gold color
            
            # Set pen for drawing border
            pen = QPen(border_color)
            pen.setWidth(6)  # Thicker border
            painter.setPen(pen)
            
            # Draw border around the widget with rounded corners
            painter.drawRoundedRect(3, 3, self.width() - 6, self.height() - 6, 15, 15)
            
            # Add a subtle glow effect
            glow_pen = QPen(QColor(255, 215, 0, int(50 * self.border_opacity)))
            glow_pen.setWidth(10)
            painter.setPen(glow_pen)
            painter.drawRoundedRect(5, 5, self.width() - 10, self.height() - 10, 15, 15)

    def update_conference_logo(self, conference):
        if not conference:
            self.conf_logo_label.clear()
            return
            
        # Format conference name for file path
        conf_name = conference.lower().replace(' ', '_')
        logo_path = os.path.join("Game Predictions", "assets", "logos", f"{conf_name}.png")
        
        # Create a circular background for the logo
        self.conf_logo_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.4); /* semi-transparent white */
            border: 2px solid rgba(255, 255, 255, 0.1); /* more visible white border */
            border-radius: 60px; /* half of width/height for perfect circle */
            padding: 10px;
        """)
        
        # Try to load the conference logo
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.conf_logo_label.setPixmap(pixmap)
        else:
            # Display text if logo not found
            self.conf_logo_label.setText(conference)
            self.conf_logo_label.setStyleSheet("""
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                background-color: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 60px;
                padding: 10px;
            """)
    
    def year_changed(self, year):
        if not year or not self.parent_app or not self.is_left:
            return

        self.parent_app.selected_season = year
        self.parent_app.left_year = year
        self.parent_app.right_year = year
        self.parent_app.update_team_dropdowns_for_selected_season()
        self.parent_app.update_team_stats()
            
    def team_changed(self, team):
        if not team:
            return
            
        if self.is_left:
            self.parent_app.left_team = team
        else:
            self.parent_app.right_team = team
            
        # Update team colors
        self.update_team_colors(team)
        
        # Update team logo
        self.update_logo(team)
        
        # Update stats and UI
        self.parent_app.update_team_stats()
        
        # Reset win gradient when team changes
        self.win_gradient = False
        self.win_probability = 0.0
        self.probability_label.setText("")
        
        # Stop border animation if running
        if self.border_animation_timer.isActive():
            self.border_animation_timer.stop()
            
        self.update()
    
    def update_team_colors(self, team):
        self.team_colors = get_team_color(team)
        # Update the UI with new colors
        self.update()
    
    def update_logo(self, team):
        if not team:
            # If no team selected, show placeholder
            self.logo_label.setText("")
            self.logo_label.setStyleSheet("color: white; font-size: 180px; font-weight: bold; background-color: transparent;")
            return
        
        try:
            # Get normalized team name
            logo_url = get_logo_url(team)
            if not logo_url:
                raise ValueError("Missing logo URL")
            
            # Create network request
            self.network_manager = QNetworkAccessManager()
            request = QNetworkRequest(QUrl(logo_url))
            
            # Connect signal to handle the finished download
            self.network_manager.finished.connect(self.handle_logo_response)
            
            # Start the request
            self.reply = self.network_manager.get(request)
            
            # Store the team name to use in the callback
            self.current_team = team
            
            # Show a loading indicator while waiting
            self.logo_label.setText("Loading...")
            
        except Exception as e:
            print(f"Error loading logo for {team}: {str(e)}")
            # Use first letter as placeholder
            self.logo_label.setText(team[0] if team else "")
            self.logo_label.setStyleSheet(f"color: {self.team_colors[0] if hasattr(self, 'team_colors') and self.team_colors else 'white'}; font-size: 180px; font-weight: bold; background-color: transparent;")
            
    @pyqtSlot(QNetworkReply)
    def handle_logo_response(self, reply):
        pixmap = QPixmap()
        
        if reply.error() == QNetworkReply.NoError:
            # Read image data and load into pixmap
            image_data = reply.readAll()
            pixmap.loadFromData(image_data)
            
            if not pixmap.isNull():
                # Scale to larger size for better visibility (400x400)
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pixmap)
                self.logo_label.setText("")
            else:
                # Fallback if image data couldn't be loaded
                self.logo_label.setText(self.current_team[0] if self.current_team else "")
                self.logo_label.setStyleSheet(f"color: {self.team_colors[0] if hasattr(self, 'team_colors') and self.team_colors else 'white'}; font-size: 180px; font-weight: bold; background-color: transparent;")
        else:
            # Handle error
            print(f"Error downloading logo for {self.current_team}: {reply.errorString()}")
            # Use first letter as placeholder with larger font
            self.logo_label.setText(self.current_team[0] if self.current_team else "")
            self.logo_label.setStyleSheet(f"color: {self.team_colors[0] if hasattr(self, 'team_colors') and self.team_colors else 'white'}; font-size: 180px; font-weight: bold; background-color: transparent;")
        
        # Clean up
        reply.deleteLater()

class StatsRow(QWidget):
    def __init__(self, label, left_value, right_value, left_is_better, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Left value
        self.left_label = QLabel(left_value)
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet(f"""
            color: white;
            font-weight: {'bold' if left_is_better else 'normal'};
            background-color: {f'rgba(0, 128, 0, 0.3)' if left_is_better else 'transparent'};
            padding: 4px;
        """)
        
        # Center label
        center_label = QLabel(label)
        center_label.setAlignment(Qt.AlignCenter)
        center_label.setFixedWidth(140)
        center_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        
        # Right value
        self.right_label = QLabel(right_value)
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet(f"""
            color: white;
            font-weight: {'bold' if not left_is_better else 'normal'};
            background-color: {f'rgba(0, 128, 0, 0.3)' if not left_is_better else 'transparent'};
            padding: 4px;
        """)
        
        layout.addWidget(self.left_label, 1)
        layout.addWidget(center_label)
        layout.addWidget(self.right_label, 1)

class NCAATeamMatchupApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Prediction models (ensemble)
        self.model_bundle = ensemble.load_models()
        # App settings
        self.setWindowTitle("NCAA Team Matchup")
        self.setMinimumSize(1024, 768)
        
        # Initialize data
        self.left_team = ""
        self.right_team = ""
        self.selected_season = "2026"
        self.left_year = self.selected_season
        self.right_year = self.selected_season
        self.selected_round = "Regular Season"
        self.conference_mapping = dp.load_conference_mapping()
        self.team_feature_cache = {}
        
        self.left_team_stats = TeamStats()
        self.right_team_stats = TeamStats()
        
        self.all_teams = []
        self.teams_by_year = {}
        self.all_team_stats = {}
        
        self.available_years = [
            "2026", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", 
            "2015", "2014", "2013", "2012", "2011", "2010", "2009", "2008", "2007", "2006", 
            "2005", "2004", "2003"
        ]
        
        self.tournament_rounds = [
            "Regular Season", "NCAA Tournament"
        ]
        
        # Set up UI
        self.init_ui()
        
        # Load the data
        self.load_teams_from_csv()
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create header
        header = QFrame()
        header.setStyleSheet("background-color: black;")
        header_layout = QHBoxLayout()
        header.setLayout(header_layout)
        header.setFixedHeight(120)
        
        # Logo and title
        logo_label = QLabel()
        logo_pixmap = QPixmap("Game Predictions/assets/logos/March_Madness_logo.png")
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(300, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        
        title_label = QLabel("Matchup Prediction")
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
        
        # Tournament round selector
        round_layout = QHBoxLayout()
        round_label = QLabel("Season Type:")
        round_label.setStyleSheet("color: white; font-size: 16px;")
        
        self.round_combo = QComboBox()
        self.round_combo.addItems(self.tournament_rounds)
        self.round_combo.setCurrentText(self.selected_round)
        self.round_combo.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: black;
                border-bottom: 2px solid orange;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: black;
            }
        """)
        
        # NEW - Connect round changed signal
        self.round_combo.currentTextChanged.connect(self.round_changed)
        
        round_layout.addWidget(round_label)
        round_layout.addWidget(self.round_combo)
        
        # Add items to header
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addLayout(round_layout)
        
        # Team comparison area
        teams_layout = QHBoxLayout()
        
        # Left team side
        self.left_team_widget = TeamSideWidget(is_left=True, parent=self, show_year_selector=True)
        
        # Right team side
        self.right_team_widget = TeamSideWidget(is_left=False, parent=self, show_year_selector=False)
        
        teams_layout.addWidget(self.left_team_widget)
        teams_layout.addWidget(self.right_team_widget)
        
        # Stats comparison section - improve to match screenshot
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: black; border: none;")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding inside
        stats_frame.setLayout(stats_layout)
        
        stats_title = QLabel("TEAM STATS COMPARISON")
        stats_title.setAlignment(Qt.AlignCenter)
        stats_title.setStyleSheet("color: white; font-weight: bold; font-size: 16px; padding: 6px;")
        stats_layout.addWidget(stats_title)
        
        # Stats rows will be added dynamically after data loads
        self.stats_grid = QVBoxLayout()
        self.stats_grid.setSpacing(8)  # Add more space between rows
        stats_layout.addLayout(self.stats_grid)
        
        # Predict button - improve styling to match the screenshot
        predict_button = QPushButton("PREDICT WINNER")
        predict_button.setFixedSize(270, 50)
        predict_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: black;
                border: 2px solid #444;
                border-radius: 25px;
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 0;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #222;
            }
            QPushButton:pressed {
                background-color: #111;
            }
        """)
        predict_button_layout = QHBoxLayout()
        predict_button_layout.setContentsMargins(0, 10, 0, 10)  # Reduce vertical margins
        predict_button_layout.addStretch()
        predict_button.clicked.connect(self.predict_winner)
        predict_button_layout.addWidget(predict_button)
        predict_button_layout.addStretch()
        
        # Add everything to main layout - remove the bottom spacing
        main_layout.addWidget(header)
        main_layout.addLayout(teams_layout, 1)
        main_layout.addWidget(stats_frame)
        main_layout.addLayout(predict_button_layout)
        
        # Remove any spacing from the bottom of the window
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("QMainWindow {background-color: black; margin: 0; padding: 0;}")
        main_widget.setStyleSheet("QWidget {background-color: black; margin: 0; padding: 0;}")
    
    def round_changed(self, round_text):
        self.selected_round = round_text
        # Reset the prediction highlights
        self.left_team_widget.set_win_gradient(False, 0.0)
        self.right_team_widget.set_win_gradient(False, 0.0)
    
    def load_teams_from_csv(self):
        try:
            # Open and read the CSV file
            with open('Game Predictions/assets/all_ratings.csv', 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                # Skip header
                header = next(reader)
                
                # Initialize data structures
                unique_teams = set()
                teams_by_year = {}
                all_team_stats = {}
                
                # Process each row
                for row in reader:
                    if len(row) < 15:
                        print(f"Skipping row: insufficient columns: {len(row)}")
                        continue
                    
                    team_name = row[1]
                    year = row[15]
                    
                    if not team_name or not year:
                        print("Skipping row: empty team or year")
                        continue
                    
                    unique_teams.add(team_name)
                    
                    # Store teams by year
                    if year not in teams_by_year:
                        teams_by_year[year] = set()
                        all_team_stats[year] = {}
                    
                    teams_by_year[year].add(team_name)
                    
                    # Create TeamStats object
                    all_team_stats[year][team_name] = TeamStats(
                        team=team_name,
                        year=year,
                        conference=row[2].strip(),
                        wins=self.parse_int_safe(row[3]),
                        losses=self.parse_int_safe(row[4]),
                        points=self.parse_float_safe(row[5]),
                        opp_points=self.parse_float_safe(row[6]),
                        margin_of_victory=self.parse_float_safe(row[7]),
                        strength_of_schedule=self.parse_float_safe(row[8]),
                        offensive_srs=self.parse_float_safe(row[9]),
                        defensive_srs=self.parse_float_safe(row[10]),
                        simple_rating_system=self.parse_float_safe(row[11]),
                        offensive_rating=self.parse_float_safe(row[12]),
                        defensive_rating=self.parse_float_safe(row[13]),
                        net_rating=self.parse_float_safe(row[12]) - self.parse_float_safe(row[13])
                    )
                # Convert sets to sorted lists
                sorted_teams_by_year = {}
                for year, teams in teams_by_year.items():
                    sorted_teams_by_year[year] = sorted(list(teams))
                
                # Store data
                self.all_teams = sorted(list(unique_teams))
                self.teams_by_year = sorted_teams_by_year
                self.all_team_stats = all_team_stats
                
                # Update UI with data
                self.update_ui_with_data()
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    def parse_int_safe(self, value):
        if not value:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def parse_float_safe(self, value):
        if not value:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_teams_for_year(self, year):
        return self.teams_by_year.get(year, self.all_teams)

    def get_team_feature_data_for_season(self, season: int) -> pd.DataFrame:
        if season not in self.team_feature_cache:
            self.team_feature_cache[season] = dp.build_team_feature_rows(season, self.conference_mapping)
        return self.team_feature_cache[season]

    def get_latest_team_snapshot(self, team_name: str, season: int) -> pd.Series | None:
        season_features = self.get_team_feature_data_for_season(season)
        if season_features.empty:
            return None

        team_rows = season_features[season_features['team_name'] == team_name].copy()
        if team_rows.empty:
            return None

        team_rows = team_rows.sort_values(by=['game_date', 'game_id'], ascending=True)
        snapshot = team_rows.iloc[-1].copy()
        return snapshot.replace([np.inf, -np.inf], np.nan).fillna(-100)

    def update_team_dropdowns_for_selected_season(self):
        teams = self.get_teams_for_year(self.selected_season)
        left_current = self.left_team_widget.team_combo.currentText()
        right_current = self.right_team_widget.team_combo.currentText()

        self.left_team_widget.team_combo.blockSignals(True)
        self.right_team_widget.team_combo.blockSignals(True)

        self.left_team_widget.team_combo.clear()
        self.right_team_widget.team_combo.clear()
        self.left_team_widget.team_combo.addItems(teams)
        self.right_team_widget.team_combo.addItems(teams)

        if teams:
            self.left_team = left_current if left_current in teams else teams[0]
            self.right_team = right_current if right_current in teams else teams[0]

            self.left_team_widget.team_combo.setCurrentText(self.left_team)
            self.right_team_widget.team_combo.setCurrentText(self.right_team)
            self.left_team_widget.update_logo(self.left_team)
            self.right_team_widget.update_logo(self.right_team)
        else:
            self.left_team = ""
            self.right_team = ""

        self.left_team_widget.team_combo.blockSignals(False)
        self.right_team_widget.team_combo.blockSignals(False)

    def update_ui_with_data(self):
        # Update season dropdown (left side only, shared by both teams)
        self.left_team_widget.year_combo.blockSignals(True)
        self.left_team_widget.year_combo.addItems(self.available_years)
        self.left_team_widget.year_combo.setCurrentText(self.selected_season)
        self.left_team_widget.year_combo.blockSignals(False)

        self.left_year = self.selected_season
        self.right_year = self.selected_season
        self.update_team_dropdowns_for_selected_season()
        self.update_team_stats()

    def update_team_stats(self):
        self.left_year = self.selected_season
        self.right_year = self.selected_season

        if self.selected_season not in self.all_team_stats:
            print(f"WARNING: Year {self.selected_season} not found in data")

        # Update left team stats
        if (
            self.left_team
            and self.selected_season in self.all_team_stats
            and self.left_team in self.all_team_stats[self.selected_season]
        ):
            self.left_team_stats = self.all_team_stats[self.selected_season][self.left_team]
            self.left_team_widget.update_conference_logo(self.left_team_stats.conference)
        else:
            print(f"No stats found for {self.left_team} in {self.selected_season} - using empty stats")
            self.left_team_stats = TeamStats()
            self.left_team_widget.conf_logo_label.clear()

        # Update right team stats
        if (
            self.right_team
            and self.selected_season in self.all_team_stats
            and self.right_team in self.all_team_stats[self.selected_season]
        ):
            self.right_team_stats = self.all_team_stats[self.selected_season][self.right_team]
            self.right_team_widget.update_conference_logo(self.right_team_stats.conference)
        else:
            print(f"No stats found for {self.right_team} in {self.selected_season} - using empty stats")
            self.right_team_stats = TeamStats()
            self.right_team_widget.conf_logo_label.clear()

        self.update_stats_comparison()
        self.left_team_widget.set_win_gradient(False, 0.0)
        self.right_team_widget.set_win_gradient(False, 0.0)
    
    def update_stats_comparison(self):
        # Clear existing stats rows
        for i in reversed(range(self.stats_grid.count())):
            widget = self.stats_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Helper function to calculate win percentage
        def safe_win_percentage(stats):
            total_games = stats.wins + stats.losses
            if total_games == 0:
                return 0.0
            return stats.wins / total_games
        
        # Add new stat rows
        # Record
        left_is_better = safe_win_percentage(self.left_team_stats) > safe_win_percentage(self.right_team_stats)
        record_row = StatsRow(
            "Record", 
            f"{self.left_team_stats.wins}-{self.left_team_stats.losses}", 
            f"{self.right_team_stats.wins}-{self.right_team_stats.losses}",
            left_is_better
        )
        self.stats_grid.addWidget(record_row)
        
        # Points Per Game
        ppg_row = StatsRow(
            "Points Per Game",
            f"{self.left_team_stats.points:.1f}",
            f"{self.right_team_stats.points:.1f}",
            self.left_team_stats.points > self.right_team_stats.points
        )
        self.stats_grid.addWidget(ppg_row)
        
        # Opponent PPG
        opp_ppg_row = StatsRow(
            "Opponent PPG",
            f"{self.left_team_stats.opp_points:.1f}",
            f"{self.right_team_stats.opp_points:.1f}",
            self.left_team_stats.opp_points < self.right_team_stats.opp_points
        )
        self.stats_grid.addWidget(opp_ppg_row)
        
        # Margin of Victory
        mov_row = StatsRow(
            "Margin of Victory",
            f"{self.left_team_stats.margin_of_victory:.1f}",
            f"{self.right_team_stats.margin_of_victory:.1f}",
            self.left_team_stats.margin_of_victory > self.right_team_stats.margin_of_victory
        )
        self.stats_grid.addWidget(mov_row)
        
        # Strength of Schedule
        sos_row = StatsRow(
            "Strength of Schedule",
            f"{self.left_team_stats.strength_of_schedule:.2f}",
            f"{self.right_team_stats.strength_of_schedule:.2f}",
            self.left_team_stats.strength_of_schedule > self.right_team_stats.strength_of_schedule
        )
        self.stats_grid.addWidget(sos_row)
        
        # Simple Rating System
        srs_row = StatsRow(
            "Rating (SRS)",
            f"{self.left_team_stats.simple_rating_system:.2f}",
            f"{self.right_team_stats.simple_rating_system:.2f}",
            self.left_team_stats.simple_rating_system > self.right_team_stats.simple_rating_system
        )
        self.stats_grid.addWidget(srs_row)
        
        # Net Rating
        net_row = StatsRow(
            "Net Rating",
            f"{self.left_team_stats.net_rating:.2f}",
            f"{self.right_team_stats.net_rating:.2f}",
            self.left_team_stats.net_rating > self.right_team_stats.net_rating
        )
        self.stats_grid.addWidget(net_row)
    
    def predict_winner(self):
        # Validate that both teams have data
        if not self.left_team or not self.right_team:
            QMessageBox.warning(self, "Missing Data", "Please select both teams before predicting.")
            return

        print(f"Predicting: {self.left_team} ({self.selected_season}) vs {self.right_team} ({self.selected_season}) in {self.selected_round}")

        try:
            season_int = int(self.selected_season)
            season_type = 3 if self.selected_round == 'NCAA Tournament' else 2

            left_snapshot = self.get_latest_team_snapshot(self.left_team, season_int)
            right_snapshot = self.get_latest_team_snapshot(self.right_team, season_int)

            if left_snapshot is None or right_snapshot is None:
                QMessageBox.warning(
                    self,
                    "Missing Data",
                    f"Could not find enough historical data for {self.left_team} or {self.right_team} in {self.selected_season}.",
                )
                return

            left_team_home_away = 2 if season_type == 3 else 1

            processed_data = dp.build_matchup_feature_row(
                left_snapshot,
                right_snapshot,
                season=season_int,
                season_type=season_type,
                team_home_away=left_team_home_away,
            )

            winner_prob, spread_pred = ensemble.predict_ensemble(processed_data, self.model_bundle)
            team1_wins = winner_prob > 0.50
            team1_win_prob = round(winner_prob * 100, 2)
            team2_win_prob = round((1 - winner_prob) * 100, 2)
            spread_range = ensemble.format_spread_range(spread_pred)

            # Update UI to highlight winner and show probabilities
            self.left_team_widget.set_win_gradient(team1_wins, team1_win_prob)
            self.right_team_widget.set_win_gradient(not team1_wins, team2_win_prob)

            # Determine winner and loser for display
            winner = self.left_team if team1_wins else self.right_team
            loser = self.right_team if team1_wins else self.left_team
            winner_prob = team1_win_prob if team1_wins else team2_win_prob

            prediction_text = f"Predicted Winner: {winner} ({winner_prob:.1f}% chance)\n\n"
            prediction_text += f"Projected Spread: {spread_range}\n\n"
            prediction_text += f"{self.left_team}: {team1_win_prob:.1f}% chance\n"
            prediction_text += f"{self.right_team}: {team2_win_prob:.1f}% chance\n\n"

            QMessageBox.information(self, "Prediction Result", prediction_text)

        except Exception as e:
            print(f"Error predicting winner: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred during prediction: {str(e)}")
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NCAATeamMatchupApp()
    window.show()
    sys.exit(app.exec_())
