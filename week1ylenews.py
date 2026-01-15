import requests
from bs4 import BeautifulSoup
import pandas as pd

#I quite strickly followed the tutorial made by Ona here, because I 
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
for title in news_titles:
    print(title.text.strip())