# ONLY in Finland – A satirical dictionary based on the latest news

## Introduction

Are you living in Finland but can't quite get a grasp of what the Finns are talking about? 

This app is for you. ONLY in Finland is an application that creates satirical dictionary definitions for your input using RAG based on latest news from [Yle](https://yle.fi/news). 

In addition, the app searches for all of the relevant news and you can choose from three different search engines: Boolean search, TF-IDF and semantic search. 

Our interactive plot shows you the topics of our current news database and has direct linsk to the source articles. 

You can find the user instructions and a thorough explanation of what our app does below. 

## 📖 User Instructions


## 🛠️ Prerequisites

Before running this application, ensure your system has the following installed:
1. **Python 3.8+**
2. **Ollama:** This app requires a local instance of [Ollama](https://ollama.com/) to run the `llama3` model for satirical definitions.
3. **Internet Connection:** Required for scraping Yle News and downloading HuggingFace models on the first run.

---

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/dlylinyao/ONLY.git](https://github.com/dlylinyao/ONLY.git)
cd ONLY
```

**2. Install Python Dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
**3. Set up the Local LLM:**
Ensure Ollama is installed and running on your machine. Pull the Llama 3 model:
```bash
ollama run llama3
```
(Keep Ollama running in the background so the Flask app can communicate with it.)

## ⚙️ Running the Application
To start the app, run the main Flask script:
```bash
python only_app.py
```

## 🎉 Server is Live!
When you run the app, it will perform a self-check. If it is the first run of the day, it will scrape Yle News, analyze sentiment, and build the search indexes. This "Cold Start" process will take several minutes.

You will know the setup is complete and the server is live when your terminal looks like this:

```text
[INIT] All search engines are ready.
[*] Initializing RAG Large Language Model (Ollama)...
 * Serving Flask app 'only_app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```
Next Step: Open your web browser and navigate to the local host address provided by Flask:
👉 http://127.0.0.1:5000

## 🎮 How to Use
1. Explore the Sky: Click on the floating blobs to expand them. Read the  topic definitions and click links to read the full Yle articles.
2. Redefine the News: * Type a term into the central search bar (e.g., "Taxation", "Trains").
   -Select your preferred search mode (TF-IDF, Boolean, Semantic).
   -Click REDEFINE.
3. Enjoy: Read the witty, AI-generated satirical definition of your term based on current Finnish news context, followed by the actual source articles.

## Troubleshooting

### Common issues and their suggested solutions:

| Issue                           | Suggested solution |
|----------------------------------|--------------------|
| Issue with Python dependencies  | We recommend using a virtual environment based on Python version 12 or lower and installing all packages from `requirements.txt`. |
| Issues with topic modeling or generating definitions | Make sure you have [Ollama](https://ollama.com) installed and running (see installation instructions).|
| Definition generation is slow   | Definitions take some time when using CPU on a laptop. If you have access to a server, try using a GPU for faster generation. |
| I don't get any search results for my input | Switch to semantic search or browse through our topic modeling on the front page for input word inspiration! | 
| None of the above, app just won't run | Make sure you have internet connection. |

## Features and responsibilities

### Scraping Yle html with Beautiful Soup

The base of our app is the scraper that scrapes 27 categories of Yle news. The scraper logic itself was created by Niki Tai and handling the output was managed by Yuxin Su. Our app runs the scraper automatically when openened if the oldest dataset is older than 24 hours.

### Sentiment analysis and filtering (Oona)

#### Sentiment analysis

To ensure our app doesn't produce any offensive or disturbing content we implemented sentiment analysis to filter out too negative news. We use Hugging Face and the bert-base-multilingual-uncased-sentiment -model from NLP-Town. As the news articles in Yle website tend to be longer than the token limit of the model, we use batching of the data to give most accurate labels to the articles. As a default, the model gives the labels as stars from 1-5 and out scraper changes that to integers and calculates the mean value of all the batches of text per article. 

We set the sentiment analysis threshold score to 1 and our scraper assigns every news article either the score 0 or 1 based on whether it passes the sentiment threshold we have set. Our scraper stores both unfiltered and filtered datasets and also the news that are filtered out to make manual evaluation of the sentiment analysis easier. 

#### Filtering based on timestamps

We want to base our app's output to most latest news, so our app filters out news that are older than a month. We use datetime and timedelta to process the timestamps of the news. To make the scraping more efficient our scraper looks at the timestamp first and processes only the articles that have a timestamp that is recent enough.

### Search engine (All)

Our app has a search engine that shows the user the relevant news for any input. The user can choose from three options: Boolean search, TF-IDF and semantic search. Boolean search andf TF-IDF both support wildcards and the whole search engine supports 3-grams, so the user can search with a longer phrase than just one word. The boolean search supports all boolean operators. 

### RAG (Yuxin)

Our RAG system is powered by a local Llama 3 (via Ollama), ensuring zero API costs and full generation control. The code includes a basic routing logic to handle different user inputs:
- **Normal Search**: If relevant news is found, it generates a short, 50-word dark definition based on the facts.
- **Easter Eggs**: To handle bad inputs, our code (_get_gibberish_type) uses funny fallbacks - Hallucinate Mode, Acronym Mode and Pet Mode.

### Flask (Linyao)
Our Flask backend acts as the brain of the application, seamlessly connecting the interactive web UI with our complex Python data pipelines and AI models. The backend is designed to handle system automation, API routing, and user experience optimizations:

#### Automated Data Pipeline:
To ensure our users always see the latest news, the app performs a daily self-check upon startup. If today's news data is missing, Flask automatically triggers the scraper and the topic clustering scripts in the background to generate fresh content.

#### Full-Stack Integration:
When a user interacts with the app, Flask acts as the traffic controller. It receives the user's search query, routes it to the search engine to find the top articles, and feeds those articles into the RAG model. It then packages the AI's definition and the search results into a JSON response, updating the UI dynamically without reloading the page.

#### Search & UI Optimization:
Cleaned the initial CSV data, merged metadata with headlines, and verified boolean search compatibility. Automated the daily news scraping process upon system startup, fixed file paths, and extended the scraping timeframe to 30 days to improve clustering data.

### User Interface (Niki)
To disrupt the doom-scrolling habit, our UI prioritizes joy and humor through a modern Glassmorphism aesthetic. By utilizing semi-transparent backgrounds and high-quality blurs, we created a light and ethereal atmosphere. This is paired with a vibrant pastel palette and a custom JavaScript Floating Emoji System that dynamically spawns animated icons across the screen. These design choices foster a whimsical environment, showing to users that staying informed can be a lighthearted and stress-free experience.
### Plotting (Yuxin)

We built a pipeline to group similar news stories together.

- **Clustering**: We turn the scraped Yle news into vectors using a MiniLM embedding model. Then, we use UMAP and BERTopic to group the articles into different topics.

- **LLM Naming**: Standard topic names (like "economy") are boring. Instead, our code (get_satirical_title_from_llama) feeds the cluster data back into Llama 3 to create a funny, satirical title for each group.

- **UI Output**: The grouped data is saved as a JSON file, which directly powers the interactive chart on our app's homepage.

## 🎙️ Summary

ONLY in Finland is a fun web app that turns Finnish news into a satirical dictionary.

The app automatically scrapes the latest articles from Yle News. To keep things lighthearted, it uses an LLM model to filter out news that is too negative or old.

Users can search for any word using three different search engines (Boolean, TF-IDF, or Semantic). The app then finds the best news articles and sends them to a local LLM model (Llama 3). This LLM acts like a cynical editor, writing a funny, 50-word definition based only on the real news. If you type random letters, the app will even give you a joke instead of an error!

Finally, the homepage shows a cool, interactive map of today's news topics, all grouped and named by the LLM. Hope it brings you a little joy amidst the dull news!




