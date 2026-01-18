import requests
from bs4 import BeautifulSoup
import pandas as pd

#I quite strickly followed the tutorial made by Oona here, because I 
# rather like to read tutorials than watch on youtube. But this is just
# to get us going.

#this is our url, it's yle news
url = "https://yle.fi/news"

#getting the contents of the page with requests
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# getting the titles
soup = soup.find('h2').find_parent()
news_titles = soup.find_all('h3')
#print(news_titles)
#for title in news_titles:
#    print(title.text.strip()) 

# Issue 8: Extract Category Information
# Print Headers
print(f"\n{'[CATEGORY]':<20} | {'HEADLINE'}")
print("-" * 70)
# Loop through each title and extract category
for title in news_titles:
    container = title.find_parent()
    category_tag = container.find('a', attrs={'data-testid': 'minimal-headline-tag'})
    if category_tag:
        category_text = category_tag.get_text().strip()
    else:
        category_text = "N/A"

    headline_text = title.get_text().strip()

    print(f"{category_text:<20} | {headline_text}")

print("-" * 70)
print("Done: Categories extracted.")