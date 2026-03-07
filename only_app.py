from flask import Flask, render_template, request, jsonify, send_file
import os
import glob
from datetime import datetime
import threading

# Import all our custom modules
import scraper as scraper
import search_engine as se
from plotting import run_clustering_pipeline
from RAG import SatiricalDictionaryOllama

# For tracking if the RAG job is running
define_running = False
lock = threading.Lock()

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def cleanup_old_files(today_str):
    """Remove old CSV data files that do not match today's date."""
    print("[*] Checking for old data files to remove...")
    all_csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    for file_path in all_csvs:
        # If the file doesn't have today's date in its name, delete it
        if today_str not in file_path:
            try:
                os.remove(file_path)
                print(f"[*] Deleted old file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[ERROR] Could not delete {file_path}: {e}")

# Startup self-check: Data update and chart generation
def initialize_data_pipeline():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Get today's date string
    today_str = datetime.now().strftime("%Y-%m-%d")
    expected_csv = os.path.join(DATA_DIR, f"yle_news_{today_str}.csv")
    topic_json_path = os.path.join(DATA_DIR, "topic_modeling_data.json")

    # 2. Clean up any CSVs from previous days
    cleanup_old_files(today_str)

    needs_clustering = False

    # 3. Check if today's data file exists
    if not os.path.exists(expected_csv):
        print(
            f"[*] Today's data ({today_str}) is missing or expired. Starting Scraper..."
        )
        scraper.scrape_yle_news()

        needs_clustering = True
    else:
        print(f"[*] Today's data already exists. Skipping scraping.")

    # 4. Check if the JSON file for D3 clustering visualization needs updating
    if needs_clustering or not os.path.exists(topic_json_path):
        print(
            "[*] Running Topic Modeling and calling Llama3 to generate satirical titles..."
        )
        run_clustering_pipeline()
    else:
        print("[*] Topic Modeling visualization data is up to date.")

# Run the self-check logic
initialize_data_pipeline()

# Initialize AI and Search Engine

print("[*] Initializing Search Engine...")
se.init_all_engines()

print("[*] Initializing RAG Large Language Model (Ollama)...")
rag_system = SatiricalDictionaryOllama(model_name="llama3")

# Flask Routes

@app.route("/")
def index():
    return render_template("only_UI.html")

@app.route("/api/topic_data")
def get_topic_data():
    """Provide clustering data to D3.js on the frontend"""
    topic_json_path = os.path.join(DATA_DIR, "topic_modeling_data.json")
    if os.path.exists(topic_json_path):
        return send_file(topic_json_path, mimetype="application/json")
    return jsonify([])

@app.route("/search", methods=["POST"])
def search():
    """Handle search requests from the frontend"""
    data = request.get_json()
    query = data.get("query", "")
    mode = data.get("mode", "tfidf")

    results = se.search(query, mode=mode, top_k=5)
    return jsonify(results)

@app.route("/define", methods=["POST"])
def define():
    global define_running

    # This is for when the user tries to search for something else when ollama is still running
    with lock:
        if define_running:
            return jsonify({"definition": "ONLY is still coming up with a definition for your previous input. Please wait for it to appear here and then you can search again."})
        define_running = True

    try:
        """Handle RAG satirical dictionary definition requests from the frontend"""
        data = request.get_json()
        query = data.get("query", "")
        mode = data.get("mode", "tfidf")

        # 1. Retrieve top 3 news articles as RAG Context
        results = se.search(query, mode=mode, top_k=3, for_rag=True)
        context_list = [res["content"] for res in results] if results else []

        # 2. Feed into the RAG system
        rag_system.ingest_context_list(context_list)

        # 3. Generate definition
        definition = rag_system.generate_definition(query)
        return jsonify({"definition": definition})
    
    except Exception as e:
        print(f"[ERROR] Ollama generation failed: {e}")
        definition = f"Oops! Connection failed, Llama3 is on strike. (Error: {e})"
    
    # This defines that the ollama is done and another search will be successful
    finally:
        with lock:
            define_running = False

if __name__ == "__main__":
    app.run(debug=True)