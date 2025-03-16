import pandas as pd

# Load the datasets
ratings_df = pd.read_csv('assets/all_ratings.csv')
games_df = pd.read_csv('cleaned_dataset.csv')

# Get unique team names from both datasets
ratings_teams = set(ratings_df['School'].unique())
games_teams = set(games_df['home_team'].unique())

# Find teams that are only in one dataset
only_in_ratings = sorted(ratings_teams - games_teams)
only_in_games = sorted(games_teams - ratings_teams)

# Determine the maximum length for padding
max_length = max(len(only_in_ratings), len(only_in_games))

# Pad the shorter list with empty strings
only_in_ratings_padded = only_in_ratings + [''] * (max_length - len(only_in_ratings))
only_in_games_padded = only_in_games + [''] * (max_length - len(only_in_games))

# Create a DataFrame for the CSV
unique_teams_df = pd.DataFrame({
    'Teams_Only_In_Ratings': only_in_ratings_padded,
    'Teams_Only_In_Games': only_in_games_padded
})

# Save to CSV
unique_teams_df.to_csv('unique_teams_by_dataset.csv', index=False)

print(f"CSV file created with {len(only_in_ratings)} teams only in ratings and {len(only_in_games)} teams only in games dataset.")
print("File saved as 'unique_teams_by_dataset.csv'")