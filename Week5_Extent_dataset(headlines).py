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
    globally_seen_urls = set()

    print(f"\n--- STARTING FULL SCRAPE: {len(sections)} Categories ---")

    for section_name, link in sections:
        print(f"\nProcessing Category: {section_name}")
        try:
            res = requests.get(link, headers=headers, timeout=15)
            soup = BeautifulSoup(res.content, 'html.parser')
            # Look for any link with /a/ which indicates an article
            links = [l for l in soup.find_all('a', href=True) if '/a/' in l['href']]
            
            if not links:
                continue

            print_progress_bar(0, len(links), prefix='   Progress:', suffix='Complete', length=30)

            for i, link_tag in enumerate(links):
                href = link_tag['href']
                article_url = href if href.startswith("http") else f"https://yle.fi{href}"
                
                # Strip fragments (like #comments) to avoid duplicates of the same article
                article_url = article_url.split('#')[0]
                
                if article_url in globally_seen_urls:
                    print_progress_bar(i + 1, len(links), prefix='   Progress:', suffix='Skip', length=30)
                    continue

                try:
                    article_res = requests.get(article_url, headers=headers, timeout=10)
                    if article_res.status_code == 200:
                        art_soup = BeautifulSoup(article_res.content, 'html.parser')
                        
                        # --- ROBUST HEADLINE EXTRACTION ---
                        headline = "N/A"
                        
                        # 1. Try Meta OG Title (Most reliable for specific headlines)
                        meta_t = art_soup.find("meta", property="og:title")
                        # 2. Try Article-specific H1 (Ignore the site-wide "News" H1)
                        h1_tags = art_soup.find_all('h1')
                        
                        # Logic: Use meta title if it exists, otherwise look for an H1 that isn't just "News"
                        if meta_t and meta_t.get("content"):
                            headline = meta_t["content"].split('|')[0].strip()
                        
                        # Fallback/Validation: If headline is still "News" or "N/A", try finding a better H1
                        if headline.lower() in ["news", "yle news", "n/a"]:
                            for h in h1_tags:
                                text = h.get_text().strip()
                                if text.lower() not in ["news", "yle news"]:
                                    headline = text
                                    break

                        # --- TIMESTAMP ---
                        time_tag = art_soup.find('time')
                        timestamp = time_tag.get_text().strip() if time_tag else "N/A"
                        
                        # --- CONTENT ---
                        content_body = art_soup.select_one('section.yle__article__content, article, main')
                        if content_body:
                            paras = content_body.find_all('p')
                            full_text = " ".join([p.text.strip() for p in paras if len(p.text.strip()) > 20])
                            
                            # Final Check: Headline must be useful for topic modeling
                            if headline.lower() not in ["news", "n/a"] and len(full_text) > 100:
                                all_news_data.append({
                                    'Time': timestamp,
                                    'Category': section_name,
                                    'Headline': headline,
                                    'Full_Text': full_text,
                                    'URL': article_url
                                })
                                globally_seen_urls.add(article_url)

                    time.sleep(0.2)
                    print_progress_bar(i + 1, len(links), prefix='   Progress:', suffix='Done', length=30)
                except:
                    continue
        except Exception as e:
            print(f"Error in {section_name}: {e}")

    # --- SAVE ---
    if all_news_data:
        df = pd.DataFrame(all_news_data)
        output_folder = os.path.expanduser("~/Downloads/YLE_Corrected_Data")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"yle_fixed_headlines_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\n--- SUCCESS: {len(df)} ARTICLES COLLECTED ---")
        print(f"File saved with correct headlines: {path}")
    else:
        print("\nNo data found.")

if __name__ == "__main__":
    scrape_yle_news()