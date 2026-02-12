from flask import Flask, render_template, request, jsonify
import search_engine_tfidf as se
import scraper as scraper
import os 
import glob
import time 

app = Flask(__name__)


folder_path = "data"
file_pattern = "filtered_yle_business_culture_*.csv"
search_path = os.path.join(folder_path, file_pattern)
files = glob.glob(search_path)
latest_file = max(files, key=os.path.getctime)
if time.time() - os.path.getctime(latest_file) >= 24 * 60 * 60:
    scraper.scrape_yle_news()

se.init_tfidf_engine()

@app.route("/")
def index():
    return render_template("Week4_UI_initial_draft.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    results = se.tfidf_search(query, top_k=5)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
