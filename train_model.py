import pandas as pd

df = pd.read_csv('final_merged_data.csv')

round_mapping = {
    'First Round': 0,
    'Second Round': 1,
    'Sweet Sixteen': 2,
    'Elite Eight': 3,
    'Final Four': 4,
    'Championship': 5
}

df['round'] = df['round'].map(round_mapping)

df = df.drop(columns=['region'])

for column in df.columns:
    try:
        if column.endswith('_team2'):
            base_column = column[:-6]  # Remove the '_team2' suffix
            diff_column = f"{base_column}_diff"
            df[diff_column] = df[base_column] - df[column]
            df = df.drop(columns=[base_column, column])
    except KeyError:
        continue

for column in df.columns:
    try:
        if column.endswith('_rating_team2'):
            base_column = column[:-12]  # Remove the '_rating_team2' suffix
            diff_column = f"{base_column}_rating_diff"
            df[diff_column] = df[base_column] - df[column]
            df = df.drop(columns=[base_column, column])
    except KeyError:
        continue

columns_to_diff = [
    'Rk', 'Conf', 'W', 'L', 'Pts', 'Opp', 'MOV', 'SOS', 
    'OSRS', 'DSRS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 
    'school_team2', 'Rk_rating_team2', 'School_rating_team2', 
    'Conf_rating_team2', 'W_rating_team2', 'L_rating_team2', 
    'Pts_rating_team2', 'Opp_rating_team2', 'MOV_rating_team2', 
    'SOS_rating_team2', 'OSRS_rating_team2', 'DSRS_rating_team2', 
    'SRS_rating_team2', 'ORtg_rating_team2', 'DRtg_rating_team2', 
    'NRtg_rating_team2'
]

for column in columns_to_diff:
    try:
        if df[column].dtype in ['int64', 'float64']:  # Check if the column is numeric
            diff_column = f"{column}_diff"
            df[diff_column] = df[column] - df[column.replace('team2', '')]  # Calculate the difference
            df = df.drop(columns=[column, column.replace('team2', '')])  # Drop both original columns after differencing
    except KeyError:
        continue

print(df.head())

# Create a new column for the target variable which is 1 if score1 > score 2 and 0 if not
df['target'] = (df['score1'] > df['score2']).astype(int)

df = df.drop(columns=['score1', 'score2', 'winner', 'school_team2', 'Conf', 'Rk_rating_team2','School_rating_team2','Conf_rating_team2','W_rating_team2','L_rating_team2','Pts_rating_team2','Opp_rating_team2','MOV_rating_team2','SOS_rating_team2','OSRS_rating_team2','DSRS_rating_team2','SRS_rating_team2','ORtg_rating_team2','DRtg_rating_team2','NRtg_rating_team2', 'rk_diff','g_diff','Rk_diff'])


print(df['target'])

df.to_csv('processed_data.csv', index=False)
