import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# We use two URLs to bypass the 'limit' of a single page
urls = [
    "https://yle.fi/t/18-220402/en", # Business Topic
    "https://yle.fi/t/18-208149/en"    # culture Section
]

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
all_news = []

for link in urls:
    print(f"Extracting from: {link}")
    res = requests.get(link, headers=headers)
    soup = BeautifulSoup(res.content, 'html.parser')
    
    # Find all headlines
    titles = soup.find_all('h3')
    for t in titles:
        headline = t.get_text().strip()
        all_news.append({'Headline': headline})

# Create DataFrame and remove duplicates
df = pd.DataFrame(all_news).drop_duplicates()

# Save to your Downloads folder
save_path = os.path.expanduser("~/Downloads/yle_master_list.csv")
df.to_csv(save_path, index=False)

print(f"\n--- MISSION ACCOMPLISHED ---")
print(f"Total Unique Articles: {len(df)}")
print(f"File saved to: {save_path}")