from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

import cached_matchup
import model_ensemble as ensemble

warnings.filterwarnings("ignore")


@dataclass
class Team:
    name: str
    seed: int
    season: int | None = None
    region: str | None = None


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GAME_PRED_DIR = Path(__file__).resolve().parent
_DATA_DIR = _PROJECT_ROOT / "Data"
_MODEL_DIR = _GAME_PRED_DIR / "models"
_MODEL_BUNDLE = None


def _get_model_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = ensemble.load_models(_MODEL_DIR)
    return _MODEL_BUNDLE


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
    model_bundle = _get_model_bundle()
    season = _resolve_season(team_a, team_b)

    processed_data = cached_matchup.build_prediction_feature_row(
        team_a.name,
        team_b.name,
        season=season,
        season_type=3,
        team_a_home_away=2,
    )
    win_prob, _ = ensemble.predict_meta_ensemble(processed_data, model_bundle)
    return win_prob


def sanity_check_team_mappings(teams, season: int | None = None):
    if not teams:
        return []

    resolved_season = int(season) if season is not None else int(teams[0].season or _latest_season_from_data())
    team_id_map = cached_matchup._load_team_id_map(resolved_season)

    missing = []
    for team in teams:
        if cached_matchup._team_key(team.name) not in team_id_map:
            missing.append(team.name)

    if missing:
        print(f"Unmatched team names for {resolved_season}:")
        for name in missing:
            print(f" - {name}")
    else:
        print(f"All {len(teams)} team names matched for {resolved_season}.")

    return missing


def _ordered_64_team_field(teams):
    if len(teams) != 64:
        raise ValueError("simulate_tournament expects exactly 64 teams.")

    region_order = ["East", "South", "West", "Midwest"]
    if all(getattr(team, "region", None) for team in teams):
        region_map = {region: [] for region in region_order}
        for team in teams:
            if team.region not in region_map:
                raise ValueError(f"Unexpected region '{team.region}'. Expected one of {region_order}.")
            region_map[team.region].append(team)

        ordered = []
        for region in region_order:
            region_teams = sorted(region_map[region], key=lambda team: team.seed)
            if len(region_teams) != 16:
                raise ValueError(f"Region {region} must contain exactly 16 teams.")
            ordered.extend(region_teams)
        return ordered

    return teams


def _build_prob_lookup(teams, prob_lookup=None):
    cache: dict[tuple[int, int], float] = {}

    def get_prob(i: int, j: int) -> float:
        if i == j:
            return 0.5

        key = (i, j)
        if key in cache:
            return cache[key]

        if prob_lookup:
            p = float(prob_lookup(teams[i], teams[j]))
        else:
            print(f"Predicting win probability for {teams[i].name} vs {teams[j].name}")
            p = float(_predict_win_prob(teams[i], teams[j]))

        cache[(i, j)] = p
        cache[(j, i)] = 1.0 - p
        return p

    return get_prob


def _play_region(get_prob, base_idx, sims, rng):
    def play_match(a_idx, b_idx):
        p = np.fromiter((get_prob(int(a), int(b)) for a, b in zip(a_idx, b_idx)), dtype=float, count=len(a_idx))
        r = rng.random(size=a_idx.shape[0])
        return np.where(r < p, a_idx, b_idx)

    seeds = list(range(base_idx, base_idx + 16))

    r32_1 = play_match(np.full(sims, seeds[0]), np.full(sims, seeds[15]))
    r32_2 = play_match(np.full(sims, seeds[7]), np.full(sims, seeds[8]))
    r32_3 = play_match(np.full(sims, seeds[4]), np.full(sims, seeds[11]))
    r32_4 = play_match(np.full(sims, seeds[3]), np.full(sims, seeds[12]))
    r32_5 = play_match(np.full(sims, seeds[5]), np.full(sims, seeds[10]))
    r32_6 = play_match(np.full(sims, seeds[2]), np.full(sims, seeds[13]))
    r32_7 = play_match(np.full(sims, seeds[6]), np.full(sims, seeds[9]))
    r32_8 = play_match(np.full(sims, seeds[1]), np.full(sims, seeds[14]))

    round_of_32 = np.stack([r32_1, r32_2, r32_3, r32_4, r32_5, r32_6, r32_7, r32_8], axis=1)

    s16_1 = play_match(r32_1, r32_2)
    s16_2 = play_match(r32_3, r32_4)
    s16_3 = play_match(r32_5, r32_6)
    s16_4 = play_match(r32_7, r32_8)

    sweet_16 = np.stack([s16_1, s16_2, s16_3, s16_4], axis=1)

    e8_1 = play_match(s16_1, s16_2)
    e8_2 = play_match(s16_3, s16_4)

    elite_8 = np.stack([e8_1, e8_2], axis=1)
    final_4_team = play_match(e8_1, e8_2)

    return {
        'round_of_32': round_of_32,
        'sweet_16': sweet_16,
        'elite_8': elite_8,
        'final_4_team': final_4_team,
    }


def simulate_tournament(teams, prob_lookup=None, sims=1000):
    teams = _ordered_64_team_field(teams)
    n = len(teams)
    get_prob = _build_prob_lookup(teams, prob_lookup=prob_lookup)
    rng = np.random.default_rng()

    def play_match(a_idx, b_idx):
        p = np.fromiter((get_prob(int(a), int(b)) for a, b in zip(a_idx, b_idx)), dtype=float, count=len(a_idx))
        r = rng.random(size=a_idx.shape[0])
        return np.where(r < p, a_idx, b_idx)

    east = _play_region(get_prob, 0, sims, rng)
    south = _play_region(get_prob, 16, sims, rng)
    west = _play_region(get_prob, 32, sims, rng)
    midwest = _play_region(get_prob, 48, sims, rng)

    round_of_32_counts = np.bincount(
        np.concatenate([
            east['round_of_32'].ravel(),
            south['round_of_32'].ravel(),
            west['round_of_32'].ravel(),
            midwest['round_of_32'].ravel(),
        ]),
        minlength=n,
    ).astype(float)
    sweet_16_counts = np.bincount(
        np.concatenate([
            east['sweet_16'].ravel(),
            south['sweet_16'].ravel(),
            west['sweet_16'].ravel(),
            midwest['sweet_16'].ravel(),
        ]),
        minlength=n,
    ).astype(float)
    elite_8_counts = np.bincount(
        np.concatenate([
            east['elite_8'].ravel(),
            south['elite_8'].ravel(),
            west['elite_8'].ravel(),
            midwest['elite_8'].ravel(),
        ]),
        minlength=n,
    ).astype(float)

    final_four_teams = np.stack(
        [
            east['final_4_team'],
            south['final_4_team'],
            west['final_4_team'],
            midwest['final_4_team'],
        ],
        axis=1,
    )
    final_4_counts = np.bincount(final_four_teams.ravel(), minlength=n).astype(float)

    semifinal_1_winner = play_match(east['final_4_team'], south['final_4_team'])
    semifinal_2_winner = play_match(west['final_4_team'], midwest['final_4_team'])
    national_championship_teams = np.stack([semifinal_1_winner, semifinal_2_winner], axis=1)
    national_championship_counts = np.bincount(national_championship_teams.ravel(), minlength=n).astype(float)

    champions = play_match(semifinal_1_winner, semifinal_2_winner)
    champion_counts = np.bincount(champions, minlength=n).astype(float)

    results = {}
    for i, team in enumerate(teams):
        results[team.name] = {
            'round_of_32': round_of_32_counts[i] / sims,
            'sweet_16': sweet_16_counts[i] / sims,
            'elite_8': elite_8_counts[i] / sims,
            'final_4': final_4_counts[i] / sims,
            'national_championship': national_championship_counts[i] / sims,
            'champion': champion_counts[i] / sims,
        }

    return results


if __name__ == "__main__":

    tournament_year = 2026
    south_teams = [
        Team("Florida", 1, tournament_year, "South"),
        Team("Houston", 2, tournament_year, "South"),
        Team("Illinois", 3, tournament_year, "South"),
        Team("Nebraska", 4, tournament_year, "South"),
        Team("Vanderbilt", 5, tournament_year, "South"),
        Team("North Carolina", 6, tournament_year, "South"),
        Team("Saint Mary's", 7, tournament_year, "South"),
        Team("Clemson", 8, tournament_year, "South"),
        Team("Iowa", 9, tournament_year, "South"),
        Team("Texas A&M", 10, tournament_year, "South"),
        Team("VCU", 11, tournament_year, "South"),
        Team("McNeese", 12, tournament_year, "South"),
        Team("Troy", 13, tournament_year, "South"),
        Team("Pennsylvania", 14, tournament_year, "South"),
        Team("Idaho", 15, tournament_year, "South"),
        Team("Prairie View A&M", 16, tournament_year, "South"),
    ]
    east_teams = [
        Team("Duke", 1, tournament_year, "East"),
        Team("UConn", 2, tournament_year, "East"),
        Team("Michigan State", 3, tournament_year, "East"),
        Team("Kansas", 4, tournament_year, "East"),
        Team("St. John's", 5, tournament_year, "East"),
        Team("Louisville", 6, tournament_year, "East"),
        Team("UCLA", 7, tournament_year, "East"),
        Team("Ohio State", 8, tournament_year, "East"),
        Team("TCU", 9, tournament_year, "East"),
        Team("UCF", 10, tournament_year, "East"),
        Team("South Florida", 11, tournament_year, "East"),
        Team("Northern Iowa", 12, tournament_year, "East"),
        Team("California Baptist", 13, tournament_year, "East"),
        Team("North Dakota State", 14, tournament_year, "East"),
        Team("Furman", 15, tournament_year, "East"),
        Team("Siena", 16, tournament_year, "East"),
    ]
    west_teams = [
        Team("Arizona", 1, tournament_year, "West"),
        Team("Purdue", 2, tournament_year, "West"),
        Team("Gonzaga", 3, tournament_year, "West"),
        Team("Arkansas", 4, tournament_year, "West"),
        Team("Wisconsin", 5, tournament_year, "West"),
        Team("BYU", 6, tournament_year, "West"),
        Team("Miami", 7, tournament_year, "West"),
        Team("Villanova", 8, tournament_year, "West"),
        Team("Utah State", 9, tournament_year, "West"),
        Team("Missouri", 10, tournament_year, "West"),
        Team("Texas", 11, tournament_year, "West"),
        Team("High Point", 12, tournament_year, "West"),
        Team("Hawai'i", 13, tournament_year, "West"),
        Team("Kennesaw State", 14, tournament_year, "West"),
        Team("Queens University", 15, tournament_year, "West"),
        Team("Long Island University", 16, tournament_year, "West"),
    ]
    midwest_teams = [
        Team("Michigan", 1, tournament_year, "Midwest"),
        Team("Iowa State", 2, tournament_year, "Midwest"),
        Team("Virginia", 3, tournament_year, "Midwest"),
        Team("Alabama", 4, tournament_year, "Midwest"),
        Team("Texas Tech", 5, tournament_year, "Midwest"),
        Team("Tennessee", 6, tournament_year, "Midwest"),
        Team("Kentucky", 7, tournament_year, "Midwest"),
        Team("Georgia", 8, tournament_year, "Midwest"),
        Team("Saint Louis", 9, tournament_year, "Midwest"),
        Team("Colorado State", 10, tournament_year, "Midwest"),
        Team("Miami (OH)", 11, tournament_year, "Midwest"),
        Team("Akron", 12, tournament_year, "Midwest"),
        Team("Hofstra", 13, tournament_year, "Midwest"),
        Team("Wright State", 14, tournament_year, "Midwest"),
        Team("Tennessee State", 15, tournament_year, "Midwest"),
        Team("Howard", 16, tournament_year, "Midwest"),
    ]
    teams = south_teams + east_teams + west_teams + midwest_teams

    missing = sanity_check_team_mappings(teams, tournament_year)
    if missing:
        print(f"Unmatched team names for {tournament_year}:")
        for name in missing:
            print(f" - {name}")
        exit(1)

    results = simulate_tournament(teams, None, sims=100000)
    import csv

    with open("tournament_probabilities.csv", "w", newline='') as csvfile:
        fieldnames = [
            "Team",
            "Round of 32",
            "Sweet 16",
            "Elite 8",
            "Final 4",
            "National Championship",
            "Champion"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for team, probs in sorted(results.items(), key=lambda x: -x[1]['champion']):
            writer.writerow({
                "Team": str(team),
                "Round of 32": probs['round_of_32'] * 100,
                "Sweet 16": probs['sweet_16'] * 100,
                "Elite 8": probs['elite_8'] * 100,
                "Final 4": probs['final_4'] * 100,
                "National Championship": probs['national_championship'] * 100,
                "Champion": probs['champion'] * 100
            })
    print('Tournament probabilities saved to tournament_probabilities.csv')
