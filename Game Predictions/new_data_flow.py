from pathlib import Path
import re

import pandas as pd

from data_processing import process_all_games, load_conference_mapping, GAME_RESULTS_DIR, DATA_DIR, PROJECT_ROOT


def get_year_from_games_filename(path: Path) -> str:
    match = re.search(r'games_(\d{4})\.csv$', path.name)
    return match.group(1) if match else 'unknown'


def main() -> None:
    conference_mapping = load_conference_mapping()

    games_files = sorted(
        GAME_RESULTS_DIR.glob('games_*.csv'),
        key=lambda p: get_year_from_games_filename(p)
    )
    if not games_files:
        print('No games_*.csv files found to process.')
        return

    elo_ratings = {}
    elo_last_season = {}

    # Load and concatenate all seasons at once
    dfs = []
    for games_file in games_files:
        year = get_year_from_games_filename(games_file)
        print(f'Processing {games_file}...')
        df = pd.read_csv(games_file)
        df['season'] = int(year)
        df = process_all_games(df, conference_mapping, elo_ratings, elo_last_season)
        dfs.append(df)
    all_games = pd.concat(dfs, ignore_index=True)

    all_games.to_csv(PROJECT_ROOT / 'Game Predictions' / 'dataset.csv', index=False)
    print(f'Wrote dataset.csv ({len(all_games)} rows)')


if __name__ == '__main__':
    main()
