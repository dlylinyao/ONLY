from flask import Flask, render_template, request, jsonify
import search_engine_tfidf as se

app = Flask(__name__)

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
