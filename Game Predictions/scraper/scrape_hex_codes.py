'''
Scrape NCAA team hex code colors from https://teamcolorcodes.com/
'''
import re
import requests
from bs4 import BeautifulSoup


url = "https://teamcolorcodes.com/abilene-christian-wildcats-colors/"
response = requests.get(url)

team_links = []
colors = {}

if response.status_code == 200:
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    divs = soup.find_all('div', class_='colorblock')
    links = soup.find_all('a')

    for link in links:
        team_links.append(link.get('href'))
    
    for p in paragraphs:
        team = p.get_text()
        team = team.replace(' ', '_')
        team = team.lower()

    for div in divs:
        print(div.get_text())
    print("HTML response received successfully.")
else:
    print(f"Failed to retrieve HTML. Status code: {response.status_code}")

def parse_colors(text):
    # Regex pattern to capture color name(s) and corresponding hex value
    pattern = re.compile(r"([A-Z][a-zA-Z ]*)\s*(?:PANTONE:|Hex Color:|Hex COLOR:|Hex:)\s*#([0-9a-fA-F]{6})")
    
    matches = pattern.findall(text)
    
    # Store results in a list of tuples
    colors = [(f"#{match[1]}") for match in matches]
    
    return colors

switch = 0
for link in team_links:
    if link == 'https://teamcolorcodes.com/abilene-christian-wildcats-colors/':
        switch = 1
    elif link == 'https://teamcolorcodes.com/disclaimer/':
        switch = 0
    if switch == 1:
        response = requests.get(link)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find_all('div', class_='colorblock')
        for div in divs:
            team_colors = parse_colors(div.get_text())
            team_name = link.split('/')[-2].replace('-', ' ').title()  # Extract team name from link
            if colors.get(team_name) is None:  # Check if team name is not already in colors
                colors[team_name] = team_colors[:2]  # Store the first two colors associated with the link


with open('team_colors.txt', 'w') as file:
    for team_name, color_list in colors.items():
        file.write(f"{team_name}: {', '.join(color_list)}\n")
