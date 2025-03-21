import pandas as pd

df = pd.read_csv('data/seed_records.csv')
# Convert to long format
df_melted = df.melt(id_vars=['Seed'], var_name='Opponent_Seed', value_name='Win_Percentage')
print(df_melted)
df_melted.to_csv('data/seed_records_new.csv')
# Convert columns to numeric
df_melted['Opponent_Seed'] = df_melted['Opponent_Seed'].astype(int)

# Compute seed difference
df_melted['Seed_Diff'] = df_melted['Opponent_Seed'] - df_melted['Seed']

# Group by seed difference and calculate average win percentage
seed_diff_avg_win_pct = df_melted.groupby('Seed_Diff')['Win_Percentage'].mean().reset_index()

# Display the result
print(seed_diff_avg_win_pct)

seed_diff_avg_win_pct.columns = [col.lower() for col in seed_diff_avg_win_pct.columns]

seed_diff_avg_win_pct.to_csv('data/seed_records_fixed.csv')