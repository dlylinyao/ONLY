# ONLY in Finland

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
pip install flask pandas beautifulsoup4 requests transformers torch sentence-transformers umap-learn bertopic scikit-learn nltk ollama
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





