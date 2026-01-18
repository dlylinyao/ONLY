import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

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
#print(f"\n{'[CATEGORY]':<20} | {'HEADLINE'}")
#print("-" * 70)
# List to store data for CSV
all_news_data = []

# Loop through each title and extract category
for title in news_titles:
    headline_text = title.get_text().strip()
    container = title.find_parent()
    category_tag = container.find('a', attrs={'data-testid': 'minimal-headline-tag'})
    if category_tag:
        category_text = category_tag.get_text().strip()
    else:
        category_text = "N/A"

    

    #print(f"{category_text:<20} | {headline_text}")

#print("-" * 70)
#print("Done: Categories extracted.")

# Issue 9: Time Extraction
    time_tag = title.find_previous('time')
    if time_tag:
        timestamp_text = time_tag.get_text().strip()
    else:
        timestamp_text = "N/A"

# Issue 10: Data Pakaging and Export to CSV
    all_news_data.append({
        'Time': timestamp_text,
        'Category': category_text,
        'Headline': headline_text
    })

# Create DataFrame
df = pd.DataFrame(all_news_data)

df['Headline'] = df['Headline'].str.replace('\n', ' ').str.strip()

print("\nExtracted News:")
print(df.head())

# Generate Filename with date
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"week1ylenews_{current_date}.csv"

# Save to CSV
df.to_csv(filename, index=False, encoding='utf-8-sig')

print(f"Done! Saved to {filename}")