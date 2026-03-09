# %%
#Import file
import pandas as pd

df = pd.read_csv('../Data/game_results/games_2026.csv')

# %%
# Load conference mapping (replaces KenPom short_conference_name dependency)
conference_mapping = pd.read_csv(
    '../Data/kenpom/REF _ NCAAM Conference and ESPN Team Name Mapping.csv',
    usecols=['Conference', 'Mapped ESPN Team Name']
).dropna()
conference_mapping['Mapped ESPN Team Name'] = conference_mapping['Mapped ESPN Team Name'].replace({'Hawaii': 'Hawai\'i', 'St. Francis (PA)': 'Saint Francis', 'San Jose State': 'San José State'})
conference_mapping['team_location_key'] = conference_mapping['Mapped ESPN Team Name'].astype(str).str.strip().str.lower()
conference_mapping = conference_mapping.drop_duplicates(subset=['team_location_key'])
conference_mapping = conference_mapping.rename(columns={'Conference': 'short_conference_name'})
conference_mapping = conference_mapping[['team_location_key', 'short_conference_name']]

df['team_location'] = df['team_location'].replace({'Hawaii': 'Hawai\'i', 'St. Francis (PA)': 'Saint Francis', 'San JosÃ© St': 'San Jose State'})
df['team_location_key'] = df['team_location'].astype(str).str.strip().str.lower()
df = df.merge(conference_mapping, on='team_location_key', how='left')
df.drop(columns='team_location_key',inplace=True)

# %%
df = df.dropna(subset=['short_conference_name'])

# %%
df = df.drop(columns=[col for col in df.columns if col.startswith('opponent_') and col != 'opponent_team_score'])

# %%
drop_cols = ['game_date_time','team_uid','team_location','team_slug','team_name','team_abbreviation','team_display_name','team_short_display_name','team_color','team_alternate_color','team_logo']
df.drop(columns=drop_cols,inplace=True)

# %%
df_merged = df.merge(df, on=["game_id","season","season_type","game_date"], suffixes=(None,"_opponent"))
df_merged = df_merged[df_merged["team_id"] != df_merged["team_id_opponent"]]


# %%
# Missing stats
df_merged['poss'] = df_merged['field_goals_attempted'] - df_merged['offensive_rebounds'] + df_merged['team_turnovers'] + (0.475 * df_merged['free_throws_attempted'])
df_merged['poss_opponent'] = df_merged['field_goals_attempted_opponent'] - df_merged['offensive_rebounds_opponent'] + df_merged['team_turnovers_opponent'] + (0.475 * df_merged['free_throws_attempted_opponent'])

# Core Predictors
df_merged['off_eff'] = (df_merged['team_score'] / df_merged['poss']) * 100
df_merged['def_eff'] = (df_merged['team_score_opponent'] / df_merged['poss_opponent']) * 100
df_merged['net_eff'] = df_merged['off_eff'] - df_merged['def_eff']

# %%
# Four Factors
df_merged['efg'] = (df_merged['field_goals_made'] + (0.5 * df_merged['three_point_field_goals_made']))/df_merged['field_goals_attempted']
df_merged['efg_allowed'] = (df_merged['field_goals_made_opponent'] + (0.5 * df_merged['three_point_field_goals_made_opponent']))/df_merged['field_goals_attempted_opponent']
df_merged['tov'] = df_merged['team_turnovers'] / df_merged['poss']
df_merged['stl_rate'] = df_merged['steals'] / df_merged['poss_opponent'] 
df_merged['orb'] = df_merged['offensive_rebounds'] / (df_merged['offensive_rebounds'] + df_merged['defensive_rebounds_opponent'])
df_merged['drb'] = df_merged['defensive_rebounds'] / (df_merged['defensive_rebounds'] + df_merged['offensive_rebounds_opponent'])
df_merged['ftr'] = df_merged['free_throws_attempted'] / df_merged['field_goals_attempted']

# %%
df_merged['ppp'] = df_merged['team_score'] / df_merged['poss']
df_merged['two_pm'] = df_merged['field_goals_made'] - df_merged['three_point_field_goals_made']
df_merged['two_pa'] = df_merged['field_goals_attempted'] - df_merged['three_point_field_goals_attempted']
df_merged['two_pct'] = df_merged['two_pm'] / df_merged['two_pa']

df_merged['two_pm_opponent'] = df_merged['field_goals_made_opponent'] - df_merged['three_point_field_goals_made_opponent']
df_merged['two_pa_opponent'] = df_merged['field_goals_attempted_opponent'] - df_merged['three_point_field_goals_attempted_opponent']
df_merged['two_pct_opponent'] = df_merged['two_pm_opponent'] / df_merged['two_pa_opponent']

df_merged['point_differential'] = df_merged['team_score'] - df_merged['team_score_opponent']

# %%
df_merged['assist_rate'] = df_merged['assists'] / df_merged['poss']
df_merged['assist_to_fg'] = df_merged['assists'] / df_merged['field_goals_made']
df_merged['block_rate'] = df_merged['blocks'] / df_merged['poss_opponent']
df_merged['lead_vs_outcome'] = df_merged['largest_lead'] - df_merged['point_differential']
df_merged['fast_break_pct'] = df_merged['fast_break_points'] / df_merged['team_score']
df_merged['points_off_turnover_pct'] = df_merged['turnover_points'] / df_merged['team_score']

maybe_keep = [
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
    'two_pm','two_pa','two_pm_opponent','two_pa_opponent'
]
# Keep wanted columns
drop_cols_two = ['blocks','fast_break_points','flagrant_fouls','fouls',
                'largest_lead','lead_changes','lead_percentage','points_in_paint',
                'steals','team_turnovers','technical_fouls','total_rebounds','total_technical_fouls',
                'total_turnovers','turnover_points']
drop_opponent_cols = [
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
    'opponent_team_score_opponent'
]

# %%
df_merged.drop(columns=maybe_keep, inplace=True)
df_merged.drop(columns=drop_cols_two, inplace=True)
df_merged.drop(columns=drop_opponent_cols, inplace=True)

# %%
get_avg_cols = [
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
    'points_off_turnover_pct'
]

df_merged = df_merged.sort_values(by='game_date', ascending=True)

# Encode 'team_home_away': 1 if home, 0 if away, 2 if season_type in [1, 3]
def encode_team_home_away(row):
    if row['season_type'] in [1, 3]:
        return 2
    return 1 if str(row['team_home_away']).strip().lower() == 'home' else 0

df_merged['team_home_away'] = df_merged.apply(encode_team_home_away, axis=1)

# Encode 'team_winner': 1 if True, 0 otherwise
df_merged['team_winner'] = df_merged['team_winner'].apply(lambda x: 1 if x is True or x == 1 else 0)

df_merged['home_off_eff'] = df_merged.groupby('team_id').apply(
    lambda group: group.loc[group['team_home_away'] == 1, 'off_eff']
    .shift(1).expanding().mean()
).reset_index(level=0, drop=True)
df_merged['home_def_eff'] = df_merged.groupby('team_id').apply(
    lambda group: group.loc[group['team_home_away'] == 1, 'def_eff']
    .shift(1).expanding().mean()
).reset_index(level=0, drop=True)
df_merged['away_off_eff'] = df_merged.groupby('team_id').apply(
    lambda group: group.loc[group['team_home_away'] == 0, 'off_eff']
    .shift(1).expanding().mean()
).reset_index(level=0, drop=True)
df_merged['away_def_eff'] = df_merged.groupby('team_id').apply(
    lambda group: group.loc[group['team_home_away'] == 0, 'def_eff']
    .shift(1).expanding().mean()
).reset_index(level=0, drop=True)

# Calculate last_10_efficiency = (points_last10 / poss_last10 * 100) - (opp_points_last10 / poss_last10 * 100)
df_merged['points_last10'] = df_merged.groupby('team_id')['team_score'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
df_merged['opp_points_last10'] = df_merged.groupby('team_id')['opponent_team_score'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
df_merged['poss_last10'] = df_merged.groupby('team_id')['poss'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
df_merged['poss_opp_last10'] = df_merged.groupby('team_id')['poss_opponent'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())

df_merged['last_10_efficiency'] = (
    (df_merged['points_last10'] / df_merged['poss_last10'] * 100) -
    (df_merged['opp_points_last10'] / df_merged['poss_opp_last10'] * 100)
)

df_merged.drop(['points_last10', 'opp_points_last10', 'poss_last10', 'poss_opp_last10'], axis=1, inplace=True)



for col in get_avg_cols:
    df_merged[f'{col}_avg'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).expanding().mean())
    df_merged[f'{col}_rolling_5'] = df_merged.groupby('team_id')[col].transform(lambda x: x.shift(1).rolling(5).mean())

df_merged['is_early_season'] = df_merged.isna().any(axis=1).astype(int)

get_avg_cols = [col for col in get_avg_cols if col not in ['team_score', 'opponent_team_score']]

df_merged.drop(columns=get_avg_cols, inplace=True)

# %%
df_merged['conference_strength'] = df_merged.groupby('short_conference_name')['net_eff_avg'].transform(lambda x: x.shift(1).expanding().mean())

# %%
df_merged['team_winner_shifted'] = df_merged.groupby('team_id')['team_winner'].shift(1)

# %%
# Calculate cumulative wins and losses
df_merged['wins'] = df_merged.groupby('team_id')['team_winner_shifted'].transform(lambda x: (x == True).cumsum())
df_merged['losses'] = df_merged.groupby('team_id')['team_winner_shifted'].transform(lambda x: (x == False).cumsum())

df_merged['non_conf_win'] = (df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (df_merged['short_conference_name'] != df_merged['short_conference_name_opponent'])
df_merged['non_conf_loss'] = ~(df_merged['team_winner_shifted'].fillna(False).astype(bool)) & (df_merged['short_conference_name'] != df_merged['short_conference_name_opponent'])

df_merged['non_conf_wins'] = df_merged.groupby('short_conference_name')['non_conf_win'].transform(lambda x: x.cumsum())
df_merged['non_conf_losses'] = df_merged.groupby('short_conference_name')['non_conf_loss'].transform(lambda x: x.cumsum())

df_merged['win_loss_pct'] = df_merged['wins'] / (df_merged['wins'] + df_merged['losses'])
df_merged['non_conf_win_loss_pct'] = df_merged['non_conf_wins'] / (df_merged['non_conf_wins'] + df_merged['non_conf_losses'])

df_merged.drop(columns=['wins','losses','non_conf_win','non_conf_loss','non_conf_wins','non_conf_losses','team_winner_shifted'], inplace=True)

# %%
df_merged['conference_nonconf_win_pct'] = df_merged.groupby('short_conference_name')['non_conf_win_loss_pct'].transform(lambda x: x.shift(1).expanding().mean())
df_merged['points_for'] = df_merged.groupby('team_id')['team_score'].transform(lambda x: x.cumsum())
df_merged['points_against'] = df_merged.groupby('team_id')['opponent_team_score'].transform(lambda x: x.cumsum())
k = 13.91
df_merged['pythagorean_win_pct'] = (df_merged['points_for']**k) / ((df_merged['points_for']**k)+(df_merged['points_against']**k))#= (points_for^k) / (points_for^k + points_against^k)
df_merged['luck'] = df_merged['win_loss_pct'] - df_merged['pythagorean_win_pct']
df_merged.drop(columns=['team_score','opponent_team_score','points_for','points_against','pythagorean_win_pct'], inplace=True)

# %%
df_final = df_merged.merge(df_merged, on=["game_id","season","season_type","game_date"], suffixes=("_a","_b"))
df_final = df_final[df_final["team_id_a"] != df_final["team_id_b"]]

# %%
df_final['sos'] = df_final.groupby('team_id_a')['net_eff_avg_b'].transform(lambda x: x.shift(1).expanding().mean())
df_final['sos_opp'] = df_final.groupby('team_id_b')['net_eff_avg_a'].transform(lambda x: x.shift(1).expanding().mean())
df_final['off_vs_def'] = df_final['off_eff_avg_a'] - df_final['def_eff_avg_b']
df_final['def_vs_off'] = df_final['off_eff_avg_b'] - df_final['def_eff_avg_a']

# %%
df_final['tov_vs_stl'] = df_final['tov_avg_a'] - df_final['stl_rate_avg_b']
df_final['stl_vs_tov'] = df_final['tov_avg_b'] - df_final['stl_rate_avg_a']

df_final['orb_vs_drb'] = df_final['orb_avg_a'] - df_final['drb_avg_b']
df_final['drb_vs_orb'] = df_final['orb_avg_b'] - df_final['drb_avg_a']

# %%
df_final['pace_diff'] = df_final['poss_avg_a'] - df_final['poss_avg_b']
df_final['exp_poss'] = (df_final['poss_avg_a'] + df_final['poss_avg_b']) / 2

# %%
df_final['efg_vs_efg_allowed'] = df_final['efg_avg_a'] - df_final['efg_allowed_avg_b']
df_final['efg_allowed_vs_efg'] = df_final['efg_avg_b'] - df_final['efg_allowed_avg_a']

# %%
df_final['margin_estimate'] = ((df_final['net_eff_avg_a'] - df_final['net_eff_avg_b']) * df_final['exp_poss']) / 100

# %%
df_final['home_off_away_def'] = df_final['home_off_eff_a'] - df_final['away_def_eff_b']
df_final['home_def_away_off'] = df_final['home_def_eff_a'] - df_final['away_off_eff_b']
df_final['away_off_home_def'] = df_final['away_off_eff_a'] - df_final['home_def_eff_b']
df_final['away_def_home_off'] = df_final['away_def_eff_a'] - df_final['home_off_eff_b']

# %%
drop_cols_final = ['team_id_a','team_id_b','team_winner_b','team_home_away_b','is_early_season_b',
                    'short_conference_name_a','short_conference_name_opponent_a'
                    ,'short_conference_name_b','short_conference_name_opponent_b',
                    'home_off_eff_a','home_def_eff_a','away_off_eff_a','away_def_eff_a',
                    'home_off_eff_b','home_def_eff_b','away_off_eff_b','away_def_eff_b']

df_final.drop(columns=drop_cols_final,inplace=True)

# %%
df_final = df_final.rename(columns={
    'is_early_season_a': 'is_early_season',
    'team_home_away_a': 'team_home_away',
    'team_winner_a': 'team_winner'
})

# %%
# Find all columns ending with '_a'
a_cols = [col for col in df_final.columns if col.endswith('_a')]

for col_a in a_cols:
    # derive the counterpart '_b' column name
    col_b = col_a[:-2] + '_b'
    if col_b in df_final.columns:
        # create the diff column
        col_diff = col_a[:-2] + '_diff'
        df_final[col_diff] = df_final[col_a] - df_final[col_b]
        # drop both original columns
        df_final.drop([col_a, col_b], axis=1, inplace=True)

df_final = df_final.fillna(-100)


# Add team_metrics
# Add loop by year