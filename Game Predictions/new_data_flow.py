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

    # Load and concatenate all seasons at once
    dfs = []
    for games_file in games_files:
        year = get_year_from_games_filename(games_file)
        df = pd.read_csv(games_file)
        df['season'] = int(year)
        dfs.append(df)
    all_games = pd.concat(dfs, ignore_index=True)

    # Process all seasons in one pass (EWM across time for averages; season-only for wins/sos/luck)
    elo_ratings = {}
    elo_last_season = {}
    dataset = process_all_games(all_games, conference_mapping, elo_ratings, elo_last_season)

    # Write per-season cleaned files and full dataset
    cleaned_dir = DATA_DIR / 'cleaned_data'
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    for season in sorted(dataset['season'].unique()):
        season_df = dataset[dataset['season'] == season]
        yearly_output = cleaned_dir / f'games_{season}_final.csv'
        season_df.to_csv(yearly_output, index=False)
        print(f'Wrote {yearly_output.name} ({len(season_df)} rows)')

    dataset.to_csv(PROJECT_ROOT / 'Game Predictions' / 'dataset.csv', index=False)
    print(f'Wrote dataset.csv ({len(dataset)} rows)')


if __name__ == '__main__':
    main()
