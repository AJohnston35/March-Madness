from selenium import webdriver
from selenium_stealth import stealth
import time
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import re
from parse import parse_ncaa_tournament_bracket

def scrape_tournament_data(url):
    # Set up Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Start WebDriver
    driver = webdriver.Chrome(options=options)
    
    # Apply Selenium Stealth
    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True)
    
    # Open URL
    print(f"Opening URL: {url}")
    driver.get(url)
    time.sleep(5)
    
    # Get page source
    page_source = driver.page_source
    
    # Close the driver
    driver.quit()
    
    return page_source

# Loop through years 1985-2024
for year in range(1985, 2025):
    url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
    print(f"\nProcessing year {year}...")
    
    try:
        # Get HTML content
        html_content = scrape_tournament_data(url)
        
        # Save raw HTML
        with open('tournament_page.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        # Parse using imported function
        tournament_df = parse_ncaa_tournament_bracket(html_content)
        
        # Add winner column based on scores
        tournament_df['winner'] = tournament_df.apply(
            lambda row: row['team1'] if int(row['score1']) > int(row['score2']) else row['team2'], 
            axis=1
        )
        
        # Save to CSV
        tournament_df.to_csv(f'tournament_games_{year}.csv', index=False)
        print(f"Successfully saved data for {year}")
        
        # Wait between requests
        time.sleep(3)
        
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        continue