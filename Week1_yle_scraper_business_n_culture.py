# I merged week1ylenews.py and week2ylenews (business and culture).py into this single file.
# Now the scraper collects news from both Business and Culture sections of YLE News and 
# extracts headlines, categories, timestamps, and full article text.

import requests
from bs4 import BeautifulSoup 
import pandas as pd
import time
import sys
import os
from datetime import datetime
import torch
from transformers import pipeline
#sentiment analysis pipeline
sentiment_analysis = pipeline(task="sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")



def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print()

def scrape_yle_news():
    sections = [
        ("Business", "https://yle.fi/t/18-220402/en"), 
        ("Culture",  "https://yle.fi/t/18-208149/en")
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    all_news_data = []
    
    print("\n--- STARTING SCRAPER ---")

    for section_name, link in sections:
        print(f"\nAccessing Section: {section_name} ({link})")
        
        try:
            res = requests.get(link, headers=headers)
            if res.status_code != 200:
                print(f"Failed to access. Status: {res.status_code}")
                continue
                
            soup = BeautifulSoup(res.content, 'html.parser')
            titles = soup.find_all('h3')
            total_articles = len(titles)
            
            print(f"Found {total_articles} articles. Extracting...")
            print_progress_bar(0, total_articles, prefix='Progress:', suffix='Complete', length=40)

            for i, t in enumerate(titles):
                try:
                    headline = t.get_text().strip()
                    
                    # Logic: Traverse up 3 levels to find the main card container
                    card_container = t.find_parent()
                    for _ in range(3):
                        if card_container and card_container.parent:
                            card_container = card_container.parent
                        else:
                            break
                    
                    category = "N/A"
                    timestamp = "N/A"
                    
                    if card_container:
                        # 1. Category Extraction (Prioritize Footer tags for Generic Cards)
                        footer_tag = card_container.find(attrs={'data-testid': 'tagSection-footer'})
                        minimal_tag = card_container.find(attrs={'data-testid': 'minimal-headline-tag'})
                        normal_tag = card_container.find(attrs={'data-testid': 'tag-section-tag'})

                        if footer_tag:
                            category = footer_tag.get_text().strip()
                        elif minimal_tag:
                            category = minimal_tag.get_text().strip()
                        elif normal_tag:
                            tag_text = normal_tag.get_text().strip()
                            if tag_text != "News": 
                                category = tag_text
                        
                        # 2. Time Extraction
                        time_tag = card_container.find('time')
                        if time_tag:
                            timestamp = time_tag.get_text().strip()
                    
                    
                    if category == "N/A":
                        category = section_name

                    # Link and Full Text Extraction
                    url_tag = t.find('a') if t.find('a') else t.find_parent('a')
                    article_url = "N/A"
                    full_text = "N/A"

                    if url_tag and url_tag.has_attr('href'):
                        href = url_tag['href']
                        article_url = href if href.startswith("http") else f"https://yle.fi{href}"
                        
                        # Deep Dive into article
                        try:
                            article_res = requests.get(article_url, headers=headers)
                            if article_res.status_code == 200:
                                article_soup = BeautifulSoup(article_res.content, 'html.parser')
                                main = article_soup.find('section', class_='yle__article__content') or article_soup.find('article')
                                if main:
                                    paras = main.find_all('p')
                                    full_text = " ".join([p.text.strip() for p in paras if p.text.strip()])
                                    #implement sentiment analysis
                                    sentiment = sentiment_analyse(full_text)
                                    passes_treshold = None #if the sentiment is too negative, it will get a 0 and won't pass
                                    if sentiment[0]['label'] == 'NEGATIVE' and sentiment[0]['score'] >= 0.99:
                                        passes_treshold = 0
                                    else:
                                        passes_treshold = 1
                                if timestamp == "N/A":
                                    d_time = article_soup.find('time')
                                    if d_time: timestamp = d_time.text.strip()
                            time.sleep(1)
                        except:
                            pass
                        all_news_data.append({
                            'Time': timestamp,
                            'Category': category,
                            'Headline': headline,
                            'Full_Text': full_text,
                            'URL': article_url,
                            'Sentiment_analysis': sentiment,
                            'Passes_treshold': passes_treshold
                        })
                    
                    print_progress_bar(i + 1, total_articles, prefix='Progress:', suffix='Complete', length=40)

                except Exception:
                    continue

        except Exception as e:
            print(f"Error: {e}")

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

        print(f"\n--- MISSION ACCOMPLISHED ---")
        print(f"Total Unique Articles: {len(df)}")
        print(f"File saved to: {full_path}")  
        
        
        print("\nData Preview:")
        print(df[['Category', 'Headline']].head(5))
    else:
        print("\nNo data found.")

def sentiment_analyse(article):
    return sentiment_analysis(article)



if __name__ == "__main__":
    scrape_yle_news()