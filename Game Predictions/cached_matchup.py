from __future__ import annotations

from pathlib import Path

import pandas as pd

import data_processing as dp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
CACHED_DATA_DIR = DATA_DIR / "cached_data"
GAME_RESULTS_DIR = DATA_DIR / "game_results"

_SEASON_DF_CACHE: dict[int, pd.DataFrame] = {}
_TEAM_ID_MAP_CACHE: dict[int, dict[str, int]] = {}


def _team_key(value: object) -> str:
    return str(value).strip().lower()


def _load_cached_season_rows(season: int) -> pd.DataFrame:
    if season not in _SEASON_DF_CACHE:
        cache_path = CACHED_DATA_DIR / f"df_{int(season)}.csv"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cached season rows: {cache_path}")
        df = pd.read_csv(cache_path)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        _SEASON_DF_CACHE[season] = df.sort_values(['season', 'game_date', 'game_id', 'team_id']).copy()
    return _SEASON_DF_CACHE[season]


def _load_team_id_map(season: int) -> dict[str, int]:
    if season not in _TEAM_ID_MAP_CACHE:
        games_path = GAME_RESULTS_DIR / f"games_{int(season)}.csv"
        if not games_path.exists():
            raise FileNotFoundError(f"Missing games file for team mapping: {games_path}")

        mapping_df = pd.read_csv(games_path, usecols=['team_location', 'team_id']).dropna()
        mapping_df['team_location'] = mapping_df['team_location'].replace(dp.TEAM_LOCATION_REPLACEMENTS)
        mapping_df['team_location_key'] = mapping_df['team_location'].map(_team_key)
        mapping_df = mapping_df.drop_duplicates(subset=['team_location_key'], keep='last')
        _TEAM_ID_MAP_CACHE[season] = dict(zip(mapping_df['team_location_key'], mapping_df['team_id']))
    return _TEAM_ID_MAP_CACHE[season]


def _latest_team_row(season_rows: pd.DataFrame, team_id: int) -> pd.Series:
    team_rows = season_rows[season_rows['team_id'] == team_id].copy()
    if team_rows.empty:
        raise ValueError(f"No cached rows found for team_id={team_id}")
    team_rows = team_rows.sort_values(['game_date', 'game_id'])
    return team_rows.iloc[-1].copy()


def _build_synthetic_rows(
    season_rows: pd.DataFrame,
    team_a_id: int,
    team_b_id: int,
    season: int,
    season_type: int,
    team_a_home_away: int,
) -> tuple[pd.DataFrame, int]:
    latest_a = _latest_team_row(season_rows, team_a_id)
    latest_b = _latest_team_row(season_rows, team_b_id)

    synthetic_game_id = int(pd.to_numeric(season_rows['game_id'], errors='coerce').max()) + 1
    last_game_date = pd.to_datetime(season_rows['game_date'], errors='coerce').max()
    synthetic_game_date = (
        (last_game_date + pd.Timedelta(days=1)) if pd.notna(last_game_date) else pd.Timestamp(f"{season}-03-19")
    )

    team_b_home_away = 2 if int(team_a_home_away) == 2 else (0 if int(team_a_home_away) == 1 else 1)

    row_a = latest_a.copy()
    row_b = latest_b.copy()

    for row, team_id, home_away, opp_conference in [
        (row_a, team_a_id, team_a_home_away, latest_b.get('short_conference_name')),
        (row_b, team_b_id, team_b_home_away, latest_a.get('short_conference_name')),
    ]:
        row['game_id'] = synthetic_game_id
        row['season'] = int(season)
        row['season_type'] = int(season_type)
        row['game_date'] = synthetic_game_date
        row['team_id'] = int(team_id)
        row['team_home_away'] = int(home_away)
        if 'short_conference_name_opponent' in row.index:
            row['short_conference_name_opponent'] = opp_conference

    synthetic_rows = pd.DataFrame([row_a, row_b])

    return synthetic_rows, synthetic_game_id


def build_prediction_feature_row(
    team_a_location: str,
    team_b_location: str,
    season: int,
    season_type: int,
    team_a_home_away: int,
) -> pd.DataFrame:
    season_rows = _load_cached_season_rows(season)
    team_id_map = _load_team_id_map(season)

    team_a_key = _team_key(team_a_location)
    team_b_key = _team_key(team_b_location)

    if team_a_key not in team_id_map:
        raise KeyError(f"Could not map team_location '{team_a_location}' to a team_id for {season}.")
    if team_b_key not in team_id_map:
        raise KeyError(f"Could not map team_location '{team_b_location}' to a team_id for {season}.")

    team_a_id = int(team_id_map[team_a_key])
    team_b_id = int(team_id_map[team_b_key])
    
    synthetic_rows, synthetic_game_id = _build_synthetic_rows(
        season_rows,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        season=season,
        season_type=season_type,
        team_a_home_away=team_a_home_away,
    )

    augmented_rows = pd.concat([season_rows, synthetic_rows], ignore_index=True, sort=False)
    augmented_rows = augmented_rows.sort_values(['season', 'game_date', 'game_id', 'team_id']).reset_index(drop=True)

    pair_rows = dp._build_pair_rows(augmented_rows)

    synthetic_pairs = pair_rows[
        (pair_rows['game_id'] == synthetic_game_id) & (pair_rows['team_id_a'] == team_a_id)
    ].copy()
   
    if synthetic_pairs.empty:
        raise ValueError("Synthetic matchup row could not be constructed.")

    feature_row = dp._flatten_pair_rows(synthetic_pairs, drop_missing=False)
    return feature_row.replace([float('inf'), float('-inf')], pd.NA).fillna(0)
