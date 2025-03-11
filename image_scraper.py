import requests
from bs4 import BeautifulSoup
import os

# URL to scrape
url = "https://loodibee.com/ncaa/meac/"

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all img tags with png images
png_images = soup.find_all('img', src=lambda x: x.endswith('.png'))

# Download and save each png image
for img in png_images:
    img_url = img['src']
    img_name = os.path.join('images', os.path.basename(img_url))
    
    # Download the image
    img_data = requests.get(img_url).content
    
    img_name = img_name.replace('300x300', '')
    img_name = img_name.replace('logo', '')
    img_name = img_name.replace('-', '_')
    img_name = img_name.replace('__', '').lower()
    img_name = img_name.rstrip()
    img_name = img_name.replace(' ', '_')
    # Save the image
    with open(img_name, 'wb') as handler:
        handler.write(img_data)

print("PNG images have been scraped and saved to the 'images' folder.")

