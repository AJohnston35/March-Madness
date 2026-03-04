import os
import warnings

warnings.filterwarnings("ignore", message="CUDA path could not be detected.*", category=UserWarning)

USE_GPU = os.getenv("USE_GPU", "1") == "1"
GPU_STATUS = "CPU"
cp = None
if USE_GPU:
    try:
        import cupy as _cp

        device_count = _cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            cp = _cp
            GPU_STATUS = f"GPU (CuPy, devices={device_count})"
        else:
            GPU_STATUS = "CPU (no CUDA device found)"
    except Exception as gpu_error:
        GPU_STATUS = f"CPU (GPU unavailable: {gpu_error})"

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Execution backend: {GPU_STATUS}")


def is_gpu_enabled():
    return cp is not None


def add_player_rolling_features(box_scores, numeric_columns, rolling_window=5):
    box_scores = box_scores.sort_values(by=["athlete_id", "game_date", "game_id"]).reset_index(drop=True)

    for stat in numeric_columns:
        shifted = box_scores.groupby("athlete_id")[stat].shift(1)
        box_scores[f"{stat}_per_game_last_{rolling_window}"] = (
            shifted.groupby(box_scores["athlete_id"])
            .rolling(rolling_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        box_scores[f"{stat}_per_game"] = (
            shifted.groupby(box_scores["athlete_id"])
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

    return box_scores


def build_team_metrics(box_scores):
    group_keys = ["game_id", "team_id", "game_date"]

    box_scores["is_double_digit_scorer"] = (box_scores["points_per_game"] >= 10).astype(int)

    team_metrics = box_scores.groupby(group_keys, as_index=False).agg(
        double_digit_scorers=("is_double_digit_scorer", "sum"),
        top_points_avg=("points_per_game", "max"),
        top_player_last_5_avg=("points_per_game_last_5", "max"),
        top_pra_avg=("pra_per_game", "max"),
        team_total_points_avg=("points_per_game", "sum"),
    )

    box_scores["_points_for_top"] = box_scores["points_per_game"].fillna(-np.inf)
    top_idx = box_scores.groupby(group_keys)["_points_for_top"].idxmax()
    top_positions = box_scores.loc[top_idx, group_keys + ["position"]].rename(
        columns={"position": "top_points_position"}
    )

    team_metrics = team_metrics.merge(top_positions, on=group_keys, how="left")

    team_metrics["top_points_avg"] = team_metrics["top_points_avg"].fillna(0)
    team_metrics["top_player_last_5_avg"] = team_metrics["top_player_last_5_avg"].fillna(0)
    team_metrics["top_pra_avg"] = team_metrics["top_pra_avg"].fillna(0)
    team_metrics["top_points_position"] = team_metrics["top_points_position"].fillna(0)
    team_metrics.loc[team_metrics["top_points_avg"] <= 0, "top_points_position"] = 0

    if is_gpu_enabled():
        top_points_avg = cp.asarray(team_metrics["top_points_avg"].to_numpy(dtype=np.float32, copy=False))
        team_total_points_avg = cp.asarray(
            team_metrics["team_total_points_avg"].to_numpy(dtype=np.float32, copy=False)
        )
        reliance = cp.where(
            team_total_points_avg > 0,
            (top_points_avg / team_total_points_avg) * 100,
            0,
        )
        team_metrics["one_player_reliance"] = cp.asnumpy(reliance)
    else:
        team_metrics["one_player_reliance"] = np.where(
            team_metrics["team_total_points_avg"] > 0,
            (team_metrics["top_points_avg"] / team_metrics["team_total_points_avg"]) * 100,
            0,
        )

    team_metrics.drop(columns=["team_total_points_avg"], inplace=True)

    team_metrics["reliance_ranking"] = team_metrics.groupby("game_date")["one_player_reliance"].rank(
        ascending=False, method="dense", na_option="bottom"
    )
    team_metrics["double_digit_scorers_ranking"] = team_metrics.groupby("game_date")[
        "double_digit_scorers"
    ].rank(ascending=True, method="dense", na_option="bottom")

    return team_metrics


dataset = []

for year in range(2003, 2027):
    try:
        print(f"Processing {year}...")
        box_scores = pd.read_csv(f"Data/box_scores/player_games_{year}.csv")
        box_scores = box_scores.sort_values(by=["game_date"]).fillna(0)

        cols_to_drop = [
            "season",
            "game_date_time",
            "team_name",
            "team_location",
            "team_short_display_name",
            "athlete_short_name",
            "athlete_position_name",
            "team_display_name",
            "team_uid",
            "team_slug",
            "team_logo",
            "team_abbreviation",
            "team_color",
            "team_alternate_color",
            "team_winner",
            "opponent_team_name",
            "opponent_team_display_name",
            "opponent_team_abbreviation",
            "opponent_team_logo",
            "opponent_team_color",
            "opponent_team_alternate_color",
            "offensive_rebounds",
            "defensive_rebounds",
            "athlete_jersey",
        ]
        box_scores = box_scores.drop(columns=[col for col in cols_to_drop if col in box_scores.columns])

        if "athlete_position_abbreviation" in box_scores.columns:
            box_scores["position"] = box_scores["athlete_position_abbreviation"].map(
                {"G": 1, "G-F": 2, "F": 3, "F-C": 4, "C": 5}
            )
            box_scores["position"] = box_scores["position"].fillna(0).astype(int)
            box_scores.drop(columns=["athlete_position_abbreviation"], inplace=True)
        else:
            box_scores["position"] = 0

        key_columns = [
            "athlete_id",
            "game_id",
            "game_date",
            "season_type",
            "position",
            "team_id",
            "opponent_team_id",
            "points",
            "rebounds",
            "assists",
            "field_goals_attempted",
            "free_throws_attempted",
            "turnovers",
            "minutes",
        ]

        existing_columns = set(box_scores.columns)
        required_columns = ["athlete_id", "game_id", "game_date", "team_id", "points"]
        if not all(col in existing_columns for col in required_columns):
            missing = [col for col in required_columns if col not in existing_columns]
            raise ValueError(f"Missing required columns: {missing}")

        selected_cols = [col for col in key_columns if col in existing_columns]
        box_scores = box_scores[selected_cols]

        numeric_cols = [
            "points",
            "rebounds",
            "assists",
            "field_goals_attempted",
            "free_throws_attempted",
            "turnovers",
            "minutes",
        ]
        for col in numeric_cols:
            if col in box_scores.columns:
                box_scores[col] = pd.to_numeric(box_scores[col], errors="coerce").fillna(0)

        if all(col in box_scores.columns for col in ["points", "rebounds", "assists"]):
            if is_gpu_enabled():
                points = cp.asarray(box_scores["points"].to_numpy(dtype=np.float32, copy=False))
                rebounds = cp.asarray(box_scores["rebounds"].to_numpy(dtype=np.float32, copy=False))
                assists = cp.asarray(box_scores["assists"].to_numpy(dtype=np.float32, copy=False))
                box_scores["pra"] = cp.asnumpy(points + rebounds + assists)
            else:
                box_scores["pra"] = box_scores["points"] + box_scores["rebounds"] + box_scores["assists"]
        else:
            box_scores["pra"] = box_scores["points"]

        stats_for_features = [
            col
            for col in [
                "points",
                "rebounds",
                "assists",
                "pra",
                "field_goals_attempted",
                "free_throws_attempted",
                "turnovers",
                "minutes",
            ]
            if col in box_scores.columns
        ]

        box_scores = add_player_rolling_features(box_scores, stats_for_features, rolling_window=5)
        team_metrics_df = build_team_metrics(box_scores)

        dataset.append(team_metrics_df)
        print(f"Completed {year}")
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")

if dataset:
    team_data = pd.concat(dataset, axis=0)
    team_data.to_csv("Game Predictions/team_metrics.csv", index=False)
    print("Processing complete! Team metrics saved to 'Game Predictions/team_metrics.csv'")
else:
    print("No data was processed successfully.")
