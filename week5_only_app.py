from flask import Flask, render_template, request, jsonify
import search_engine as se 
import scraper as scraper
import os 
import glob
import time 

app = Flask(__name__)

folder_path = "data"
file_pattern = "filtered_yle_business_culture_*.csv"
search_path = os.path.join(folder_path, file_pattern)
files = glob.glob(search_path)

if not files:
    print("[Scraper] No data found. Starting initial scrape...")
    scraper.scrape_yle_news()
else:
    latest_file = max(files, key=os.path.getctime)
    # If the file loaded over 24h
    # if time.time() - os.path.getctime(latest_file) >= 24 * 60 * 60:
    #     print("[Scraper] Data is old. Updating...")
    #     scraper.scrape_yle_news()
    pass


se.init_all_engines()

@app.route("/")
def index():
    return render_template("Week5_UI_initial_draft.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
  
    mode = data.get("mode", "tfidf") 

 
    results = se.search(query, mode=mode, top_k=5)
    return jsonify(results)

if __name__ == "__main__":
   
    app.run(debug=True)