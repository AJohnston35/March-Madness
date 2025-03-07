from bs4 import BeautifulSoup
import pandas as pd

def parse_ncaa_tournament_bracket(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    games_data = []
    
    # Extract brackets by region
    regions = ['east', 'midwest', 'south', 'west', 'national'] 
    
    for region in regions:
        region_div = soup.find('div', {'id': region})
        if not region_div:
            continue
            
        region_name = region.capitalize()
        
        # Find all rounds
        bracket_div = region_div.find('div', {'id': 'bracket'})
        if bracket_div:
            rounds = bracket_div.find_all('div', {'class': 'round'})
            
            for round_idx, round_div in enumerate(rounds):
                round_number = round_idx + 1
                
                # Encode round names based on region
                if region_name in ['East', 'West', 'Midwest', 'South']:
                    round_name = {
                        1: 'First Round',
                        2: 'Second Round', 
                        3: 'Sweet Sixteen',
                        4: 'Elite Eight'
                    }.get(round_number)
                else:  # National region
                    round_name = {
                        1: 'Final Four',
                        2: 'Championship'
                    }.get(round_number)
                
                # Find all games in this round
                games = round_div.find_all('div', recursive=False)
                
                for game_div in games:
                    # Find teams in this game
                    team_divs = game_div.find_all('div', recursive=False)
                    teams = []
                    
                    for team_div in team_divs:
                        # Skip if this is not a team div
                        if team_div.name != 'div' or 'game' in team_div.get('class', []):
                            continue
                            
                        # Extract team info
                        seed_span = team_div.find('span')
                        seed = seed_span.text.strip() if seed_span else None
                        
                        team_links = team_div.find_all('a')
                        if not team_links:
                            continue
                            
                        team_name = ' '.join(team_links[0].text.strip().replace('\n', ' ').split()) if team_links else "Unknown"
                        score = team_links[1].text.strip() if len(team_links) > 1 else None
                        
                        teams.append({
                            'seed': seed,
                            'team': team_name,
                            'score': score
                        })
                    
                    if len(teams) == 2:
                        games_data.append({
                            'region': region_name,
                            'round': round_name,
                            'team1': teams[0]['team'],
                            'seed1': teams[0]['seed'],
                            'score1': teams[0]['score'],
                            'team2': teams[1]['team'], 
                            'seed2': teams[1]['seed'],
                            'score2': teams[1]['score']
                        })
    
    return pd.DataFrame(games_data)

def main(html_file, year):
    # Read the HTML file
    with open(html_file, 'r') as file:
        tournament_page_html = file.read()

    # Parse the tournament bracket into a dataframe
    tournament_df = parse_ncaa_tournament_bracket(tournament_page_html)

    # Add winner column based on scores
    tournament_df['winner'] = tournament_df.apply(lambda row: row['team1'] if int(row['score1']) > int(row['score2']) else row['team2'], axis=1)

    # Save the dataframe to CSV
    tournament_df.to_csv(f'tournament_games_{year}.csv', index=False)

    # Display the dataframe
    print("\nTournament Games:")
    print(tournament_df)
    print("\nData saved to tournament_games.csv")
    print("\nWinning teams added to dataframe")

if __name__ == "__main__":
    main('tournament_page.html')