import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time # 引入时间库，用来“休息”，防止爬太快被封
import random # 随机休息时间

# --- 配置部分 ---

# 策略：通过爬取多个不同的分类页面来凑够 100 篇，避开“点击 Show More”的难题
target_urls = [
    "https://yle.fi/news",                        # 首页
    "https://yle.fi/news/topic/18-218587",        # Politics (政治)
    "https://yle.fi/news/topic/18-218588",        # Economy (经济)
    "https://yle.fi/news/topic/18-218591",        # Society (社会)
    "https://yle.fi/news/topic/18-218596"         # Culture (文化)
]

# 用来伪装成浏览器，防止被反爬
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- 辅助函数：进入文章详情页抓取正文 ---
def extract_article_content(article_url):
    try:
        # 再次发起请求，进入文章详情页
        response = requests.get(article_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # YLE 的正文通常在 article 标签里，或者由许多 <p> 组成
        # 这是一个通用的抓取策略：抓取所有段落文本
        # 注意：这里可能需要根据实际页面结构微调 class 名，但抓所有 p 标签通常能凑合用
        text_content = []
        
        # 尝试找到文章主体容器 (YLE 的 class 经常变，直接找 article 标签比较稳)
        article_body = soup.find('article')
        
        if article_body:
            paragraphs = article_body.find_all('p')
        else:
            # 如果找不到 article 标签，就找所有的 p 标签（可能会抓到杂乱信息，但也比没有好）
            paragraphs = soup.find_all('p')

        for p in paragraphs:
            text_content.append(p.get_text().strip())
        
        # 把所有段落拼成一篇文章，用空格连接
        full_text = " ".join(text_content)
        
        # 简单的清理：如果正文太短（比如少于50字），可能抓错了
        if len(full_text) < 50:
            return "Content extraction failed or content too short."
            
        return full_text

    except Exception as e:
        print(f"Error extracting content from {article_url}: {e}")
        return "Error loading content"

# --- 主程序 ---

all_news_data = []
seen_urls = set() # 用来去重，防止不同页面有重复的新闻

print(f"Starting scraping job. Target: ~{len(target_urls) * 20} articles...")

for url in target_urls:
    print(f"\nScanning page: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 获取所有包含标题的容器
        # 根据你们之前的逻辑，YLE 的标题通常在 h3 里
        # 我们这里尝试更通用的找链接的方法
        
        # 找到所有文章链接（YLE 新闻链接通常在 <a> 标签里，且 href 包含 '/a/'）
        # 这里我们结合你们之前的 h3 逻辑
        headlines = soup.find_all('h3')

        for h3 in headlines:
            # 1. 提取标题
            headline_text = h3.get_text().strip()
            
            # 2. 提取链接 (难点2的核心)
            # 链接通常在 h3 的父级或者 h3 内部的 a 标签里
            # 先找 h3 里面的 a
            link_tag = h3.find('a')
            if not link_tag:
                # 如果 h3 里面没有，就找 h3 的父级是不是 a
                link_tag = h3.find_parent('a')
            
            if not link_tag:
                # 如果还找不到，就算了
                continue
                
            href = link_tag.get('href')
            
            # 确保链接是完整的 (YLE sometimes uses relative links like /a/74-2001234)
            if href.startswith('/'):
                full_link = "https://yle.fi" + href
            else:
                full_link = href
            
            # 去重：如果你已经在其他类别抓过这条新闻，就跳过
            if full_link in seen_urls:
                continue
            seen_urls.add(full_link)

            # 3. 提取类别 (你们之前的逻辑)
            container = h3.find_parent()
            # 这种相对查找比较脆弱，如果报错可以用 try-except 包裹
            category_text = "General"
            try:
                cat_tag = container.find_parent().find('a', attrs={'data-testid': 'minimal-headline-tag'})
                if cat_tag:
                    category_text = cat_tag.get_text().strip()
            except:
                pass

            # 4. 提取时间 (你们之前的逻辑)
            timestamp_text = "Unknown"
            try:
                time_tag = h3.find_previous('time')
                if time_tag:
                    timestamp_text = time_tag.get_text().strip()
            except:
                pass
            
            print(f"-> Found: {headline_text[:30]}... | Fetching content...")

            # 5. 【关键步骤】跳转抓取正文 (Summary/Full Text)
            # 调用我们在上面写的函数
            full_content = extract_article_content(full_link)
            
            # 数据打包
            all_news_data.append({
                'Time': timestamp_text,
                'Category': category_text,
                'Headline': headline_text,
                'Content': full_content, # 这里现在存的是长文本了！
                'Link': full_link
            })

            # 休息一下，做一个有礼貌的爬虫
            time.sleep(random.uniform(0.5, 1.5))

    except Exception as e:
        print(f"Error scraping page {url}: {e}")

# --- 保存 ---
print(f"\nTotal articles scraped: {len(all_news_data)}")

df = pd.DataFrame(all_news_data)

# 简单清洗
df['Headline'] = df['Headline'].str.replace('\n', ' ').str.strip()
df['Content'] = df['Content'].str.replace('\n', ' ').str.strip()

print(df.head())

current_date = datetime.now().strftime("%Y-%m-%d")
# 注意：这次我们存两个版本，一个 CSV (给 Excel 看)，一个 TXT (给搜索引擎作业用)
filename_csv = f"week2_yle_full_{current_date}.csv"
df.to_csv(filename_csv, index=False, encoding='utf-8-sig')

print(f"Saved to {filename_csv}")

# 额外任务：Requirement 5 提到需要 "text file... one article in one string"
# 我们可以顺便生成一个 txt 文件
filename_txt = f"week2_yle_articles_{current_date}.txt"
with open(filename_txt, "w", encoding="utf-8") as f:
    for index, row in df.iterrows():
        # 写入 XML 风格的标签，方便作业 5 读取
        f.write("<article>\n")
        f.write(f"{row['Headline']}\n\n")
        f.write(f"{row['Content']}\n")
        f.write("</article>\n")
        
print(f"Saved to {filename_txt} (for Search Engine Task 5)")