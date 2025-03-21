import pandas as pd

dataset = []

for year in range(2025, 2026):
    box_scores = pd.read_csv(f"Data/box_scores/player_games_{year}.csv")

    # Sort data by game date
    box_scores = box_scores.sort_values(by=['game_date'])

    # Fill missing values
    box_scores.fillna(0, inplace=True)

    # Drop unnecessary columns
    box_scores = box_scores.drop(columns=[
        'season', 'game_date_time', 'team_name', 'team_location', 'team_short_display_name', 
        'athlete_short_name', 'athlete_position_name', 'team_display_name', 'team_uid', 'team_slug',
        'team_logo', 'team_abbreviation', 'team_color', 'team_alternate_color', 'team_winner', 
        'opponent_team_name', 'opponent_team_display_name', 'opponent_team_abbreviation', 
        'opponent_team_logo', 'opponent_team_color', 'opponent_team_alternate_color', 
        'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'athlete_jersey'
    ])

    # Convert categorical features
    box_scores['starter'] = box_scores['starter'].astype(int)
    box_scores['ejected'] = box_scores['ejected'].astype(int)
    box_scores['did_not_play'] = box_scores['did_not_play'].astype(int)
    box_scores['active'] = box_scores['active'].astype(int)
    box_scores['athlete_position_abbreviation'] = box_scores['athlete_position_abbreviation'].map(
        {'G': 1, 'G-F': 2, 'F': 3, 'F-C': 4, 'C': 5}
    )
    box_scores['home_away'] = box_scores['home_away'].map({'home': 1, 'away': 0})

    # Compute consecutive missed games
    def count_consecutive_misses(series):
        return series.eq(0).astype(int).groupby(series.ne(0).cumsum()).cumsum()

    box_scores['consecutive_missed_games'] = box_scores.groupby('athlete_id')['active'].transform(count_consecutive_misses)

    # Define rolling windows
    N = [5, 10, 15]

    # Compute rolling averages and per-36-minute stats
    numeric_columns = [
        'minutes', 'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
        'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
        'steals', 'blocks', 'turnovers', 'fouls', 'points'
    ]

    for stat in numeric_columns:
        # Per-game rolling stats
        for period in N:
            box_scores[f'{stat}_rolling_mean_{period}'] = (
                box_scores.groupby('athlete_id')[stat].transform(lambda x: x.shift(1).rolling(period, min_periods=1).mean())
            )

        # Per-36-minute stats (standardizes player performance)
        box_scores[f'{stat}_per_36'] = (box_scores[stat] / box_scores['minutes']) * 36

    # Compute usage rate: (FGA + 0.44*FTA + TO) / Team Possessions
    box_scores['usage_rate'] = (
        (box_scores['field_goals_attempted'] + 0.44 * box_scores['free_throws_attempted'] + box_scores['turnovers']) /
        (box_scores.groupby('game_id')['field_goals_attempted'].transform('sum') + 
         0.44 * box_scores.groupby('game_id')['free_throws_attempted'].transform('sum') +
         box_scores.groupby('game_id')['turnovers'].transform('sum'))
    ) * 100

    # Efficiency metrics
    box_scores['true_shooting_percentage'] = (
        box_scores['points'] / (2 * (box_scores['field_goals_attempted'] + 0.44 * box_scores['free_throws_attempted']))
    ) * 100

    # Shot composition (3PT attempt rate)
    box_scores['shot_composition'] = (box_scores['three_point_field_goals_attempted'] / box_scores['field_goals_attempted']) * 100

    # Create team-level defensive stats
    game_stats = box_scores.groupby(['game_id', 'team_id', 'opponent_team_id', 'game_date']).agg({
        'points': 'sum',
        'field_goals_made': 'sum',
        'field_goals_attempted': 'sum',
        'three_point_field_goals_made': 'sum',
        'three_point_field_goals_attempted': 'sum',
        'free_throws_made': 'sum',
        'free_throws_attempted': 'sum',
        'steals': 'sum',
        'blocks': 'sum',
        'turnovers': 'sum',
        'active': lambda x: (x == 0).sum()  # Count of inactive players
    }).reset_index()

    # Create opponent defense stats
    team_defense_stats = game_stats.rename(columns={
        'team_id': 'opponent_team_id_temp',
        'opponent_team_id': 'team_id',
        'points': 'points_allowed',
        'field_goals_made': 'fg_made_allowed',
        'field_goals_attempted': 'fg_attempted_allowed',
        'three_point_field_goals_made': 'three_pt_made_allowed',
        'three_point_field_goals_attempted': 'three_pt_attempted_allowed',
        'free_throws_made': 'ft_made_allowed',
        'free_throws_attempted': 'ft_attempted_allowed',
        'steals': 'opponent_steals',
        'blocks': 'opponent_blocks',
        'turnovers': 'opponent_turnovers',
        'active': 'injured_players'
    })
    team_defense_stats.rename(columns={'opponent_team_id_temp': 'opponent_team_id'}, inplace=True)

    # Merge defensive stats with player dataset
    box_scores = box_scores.merge(
        team_defense_stats[['team_id', 'game_id'] + 
                        [f'{stat}_rolling_mean_{period}' for stat in 
                        ['points_allowed', 'fg_made_allowed', 'fg_attempted_allowed', 'three_pt_made_allowed',
                            'three_pt_attempted_allowed', 'ft_made_allowed', 'ft_attempted_allowed', 
                            'opponent_steals', 'opponent_blocks', 'opponent_turnovers', 'injured_players'] 
                        for period in N]],
        left_on=['opponent_team_id', 'game_id'],
        right_on=['team_id', 'game_id'],
        how='left',
        suffixes=('', '_opponent')
    )

    # Drop unneeded columns
    box_scores.drop(columns=['minutes'], inplace=True)  # Already captured in per-36 stats

    # Rename target variable
    box_scores.rename(columns={'points': 'target'}, inplace=True)

    dataset.append(box_scores)
    print(f"Processed {year}")

# Combine datasets across years
full_data = pd.concat(dataset, axis=0)

# Save cleaned dataset
full_data.to_csv("Player Predictions/cleaned_dataset.csv", index=False)
