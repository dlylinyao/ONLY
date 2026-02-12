import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import os
from datetime import datetime
from transformers import pipeline
#sentiment analysis pipeline
sentiment_analysis = pipeline(task="sentiment-analysis",
                              model="nlptown/bert-base-multilingual-uncased-sentiment")

def sentiment_analyse(article):
    return sentiment_analysis(article)

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

            for i, link_tag in enumerate(links):
                href = link_tag['href']
                article_url = href if href.startswith("http") else f"https://yle.fi{href}"

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
                            #implement sentiment analysis
                            sentiment = sentiment_analyse(full_text)
                            passes_treshold = None #if the sentiment is too negative, it will get a 0 and won't pass
                            if sentiment[0]['label'] == '1 star': #and sentiment[0]['score'] >= 0.50: (changed the filter)
                                passes_treshold = 0
                            else:
                                passes_treshold = 1
                            
                            if len(full_text) > 100:
                                all_news_data.append({
                                    'Time': timestamp,
                                    'Category': section_name,
                                    'Headline': headline,
                                    'Full_Text': full_text,
                                    'URL': article_url,
                                    'Sentiment_analysis': sentiment,
                                    'Passes_treshold': passes_treshold
                                })
                                globally_seen_urls.add(article_url)

                    time.sleep(0.2) # Faster but safe delay
                except:
                    continue
        except Exception as e:
            print(f"Error in {section_name}: {e}")

    # Final Save
    if all_news_data:
        df = pd.DataFrame(all_news_data).drop_duplicates(subset=['URL'])
        
        output_folder = "data"
        
        os.makedirs(output_folder, exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"yle_business_culture_{current_date}.csv"
        full_path = os.path.join(output_folder, filename)
       
        df.to_csv(full_path, index=False, encoding='utf-8-sig')

        #filter news based on passing the threshold
        filtered_news_data = [article for article in all_news_data if article['Passes_treshold'] == 1]

        df_filtered = pd.DataFrame(filtered_news_data).drop_duplicates(subset=['URL'])
        
        output_folder = "data"
        
        os.makedirs(output_folder, exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"filtered_yle_business_culture_{current_date}.csv"
        full_path = os.path.join(output_folder, filename)
       
        df_filtered.to_csv(full_path, index=False, encoding='utf-8-sig')

        #store the filtered out news too to make evaluation easier
        filtered_out_news_data = [article for article in all_news_data if article['Passes_treshold'] == 0]

        df_filtered_out = pd.DataFrame(filtered_out_news_data).drop_duplicates(subset=['URL'])
        
        output_folder = "data"
        
        os.makedirs(output_folder, exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"filtered_out_yle_business_culture_{current_date}.csv"
        full_path = os.path.join(output_folder, filename)
       
        df_filtered_out.to_csv(full_path, index=False, encoding='utf-8-sig')
    else:
        print("\nNo data found.")

if __name__ == "__main__":
    scrape_yle_news()