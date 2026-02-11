import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import os
from datetime import datetime

def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: print()

def get_english_categories(headers):
    print("Step 1: Discovering English News Categories...")
    base_news_url = "https://yle.fi/news"
    try:
        res = requests.get(base_news_url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        links = soup.select('a[href*="/t/"][href$="/en"]')
        categories = []
        seen_urls = set()
        for link in links:
            name = link.get_text().strip()
            href = link['href']
            full_url = href if href.startswith("http") else f"https://yle.fi{href}"
            if full_url not in seen_urls and name:
                categories.append((name, full_url))
                seen_urls.add(full_url)
        return categories
    except Exception as e:
        print(f"Error: {e}")
        return []

def scrape_yle_news():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    sections = get_english_categories(headers)
    
    all_news_data = []
    # Track URLs globally to prevent duplicates across categories
    globally_seen_urls = set()

    print(f"\n--- STARTING FULL SCRAPE: {len(sections)} Categories ---")

    for section_name, link in sections:
        print(f"\nProcessing Category: {section_name}")
        try:
            res = requests.get(link, headers=headers, timeout=15)
            soup = BeautifulSoup(res.content, 'html.parser')
            # Look for all article links
            links = soup.select('a[href*="/a/"]')
            
            if not links:
                print(" (No articles found in this section)")
                continue

            print_progress_bar(0, len(links), prefix='Progress:', suffix='Complete', length=40)

            for i, link_tag in enumerate(links):
                href = link_tag['href']
                article_url = href if href.startswith("http") else f"https://yle.fi{href}"
                
                # Skip if already scraped in this run
                if article_url in globally_seen_urls:
                    print_progress_bar(i + 1, len(links), prefix='Progress:', suffix='Skipping Duplicate', length=40)
                    continue

                try:
                    article_res = requests.get(article_url, headers=headers, timeout=10)
                    if article_res.status_code == 200:
                        art_soup = BeautifulSoup(article_res.content, 'html.parser')
                        
                        # Data Extraction
                        headline = art_soup.find('h1').text.strip() if art_soup.find('h1') else "N/A"
                        time_tag = art_soup.find('time')
                        timestamp = time_tag.get_text().strip() if time_tag else "N/A"
                        
                        main_content = art_soup.select_one('section.yle__article__content, article')
                        if main_content:
                            paras = main_content.find_all('p')
                            full_text = " ".join([p.text.strip() for p in paras if len(p.text.strip()) > 20])
                            
                            if len(full_text) > 100:
                                all_news_data.append({
                                    'Time': timestamp,
                                    'Category': section_name,
                                    'Headline': headline,
                                    'Full_Text': full_text,
                                    'URL': article_url
                                })
                                globally_seen_urls.add(article_url)

                    time.sleep(0.2) # Faster but safe delay
                    print_progress_bar(i + 1, len(links), prefix='Progress:', suffix='Collected', length=40)
                except:
                    continue
        except Exception as e:
            print(f"Error in {section_name}: {e}")

    # Final Save
    if all_news_data:
        df = pd.DataFrame(all_news_data)
        output_folder = os.path.expanduser("~/Downloads/YLE_Full_Dataset")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"yle_full_scrape_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\n--- SUCCESS: COLLECTED {len(df)} UNIQUE ARTICLES ---")
        print(f"File saved: {path}")

if __name__ == "__main__":
    scrape_yle_news()