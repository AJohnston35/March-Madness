from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import data_processing as dp

CACHE_DIR = dp.PROJECT_ROOT / "Data" / "team_feature_cache"


def get_cache_path(season: int) -> Path:
    return CACHE_DIR / f"team_features_{int(season)}.csv"


def _team_key(value: object) -> str:
    return str(value).strip().lower()


def _legacy_cache_path(season: int) -> Path:
    return CACHE_DIR / f"team_features_{int(season)}.pkl"


def _canonicalize_cache_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if 'team_location' not in data.columns and 'team_name' in data.columns:
        data['team_location'] = data['team_name']
    if 'team_location_opponent' not in data.columns and 'team_name_opponent' in data.columns:
        data['team_location_opponent'] = data['team_name_opponent']
    if 'team_location' in data.columns:
        data['team_location_key'] = data['team_location'].map(_team_key)
    data.drop(columns=['team_name', 'team_name_opponent', 'team_name_key'], inplace=True, errors='ignore')
    return data


def build_team_stats_cache(season: int, *, overwrite: bool = False) -> Path:
    cache_path = get_cache_path(season)
    if cache_path.exists() and not overwrite:
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conference_mapping = dp.load_conference_mapping()
    team_rows = _canonicalize_cache_df(dp.build_team_feature_rows(int(season), conference_mapping))
    team_rows.to_csv(cache_path, index=False)
    return cache_path


def load_team_stats_cache(season: int, *, build_if_missing: bool = False) -> pd.DataFrame:
    cache_path = get_cache_path(season)
    if not cache_path.exists():
        legacy_path = _legacy_cache_path(season)
        if legacy_path.exists():
            #df = _canonicalize_cache_df(pd.read_pickle(legacy_path))
            df.to_csv(cache_path, index=False)
            return df
        if not build_if_missing:
            raise FileNotFoundError(
                f"Missing cached team stats for {season}. "
                f"Run: python \"Game Predictions/team_stats_cache.py\" --season {season}"
            )
        build_team_stats_cache(season, overwrite=False)
    raw_df = pd.read_csv(cache_path)
    df = _canonicalize_cache_df(raw_df)
    if list(df.columns) != list(raw_df.columns):
        df.to_csv(cache_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute cached team stats for matchup predictions.")
    parser.add_argument("--season", type=int, required=True, help="Season year to precompute, e.g. 2026")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the cache even if it already exists")
    args = parser.parse_args()

    cache_path = build_team_stats_cache(args.season, overwrite=args.overwrite)
    df = pd.read_csv(cache_path)
    print(f"Saved {len(df)} team feature rows to {cache_path}")


if __name__ == "__main__":
    main()
