import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import glob


#THIS FILE IS FOR EVALUATING OUR PIPELINE AS WE CHOSE TO NOT USE OTHER SEARCH EGINES THAN SEMANTIC

documents = []

bert_model = None
bert_embeddings = None



def get_news_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "data")

    file_pattern = "yle_news_*.csv"
    search_path = os.path.join(folder_path, file_pattern)
    files = glob.glob(search_path)

    files = [f for f in files if "filtered_out" not in f]

    if not files:
        return []

    latest_file = max(files, key=os.path.getctime)
    try:
        df = pd.read_csv(latest_file)

        for col in ["Time", "Category", "Headline", "Full_Text"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("")

        df["content"] = (
            df["Time"]
            + " | "
            + df["Category"]
            + " | "
            + df["Headline"]
            + "\n\n"
            + df["Full_Text"]
        )
        return df["content"].tolist()
    except Exception:
        return []


def generate_snippet(text, query, window_size=80):
    parts = text.split("\n\n", 1)
    headline = parts[0]
    body = parts[1] if len(parts) > 1 else text

    if not query:
        return f"[{headline}]\n\n{body[:window_size * 2]}..."
    
    match = re.search(re.escape(query), body, re.IGNORECASE)
    
    if match:
        start_idx = match.start()
        end_idx = match.end()
        
        snippet_start = max(0, start_idx - window_size)
        snippet_end = min(len(body), end_idx + window_size)
        
        snippet = body[snippet_start:snippet_end]
        original_word = body[start_idx:end_idx]
        snippet = snippet.replace(original_word, f"<b>{original_word}</b>")
        
        prefix = "..." if snippet_start > 0 else ""
        suffix = "..." if snippet_end < len(body) else ""
        return f"[{headline}]\n\n{prefix}{snippet}{suffix}"
    
    return f"[{headline}]\n\n{body[:window_size * 2]}..."


def init_engine():
    global documents, bert_model, bert_embeddings

    documents = get_news_data()
    if not documents:
        return
    try:
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        bert_embeddings = bert_model.encode(documents)
    except Exception:
        pass

def search_semantic(query, top_k=5, threshold=0.15):
    if bert_model is None:
        return []

    query_embedding = bert_model.encode([query])
    cosine_similarities = np.dot(query_embedding, bert_embeddings.T)[0]
    ranked_indices = np.argsort(cosine_similarities)[::-1]

    if len(ranked_indices) > 0:
        top_score = float(cosine_similarities[ranked_indices[0]])
        print(f"[DEBUG] Highest similarity scorefor '{query}': {top_score:.4f}")

    results = []
    for idx in ranked_indices[:top_k]:
        if cosine_similarities[idx] > threshold:
            results.append(
                {
                    "doc_id": int(idx),
                    "score": float(cosine_similarities[idx]),
                    "content": documents[idx],
                }
            )
    return results

def search(query, top_k=5, for_rag=False, threshold=0.15):
    results = search_semantic(query, top_k, threshold)
    
    for res in results:
        if not for_rag:
            res["content"] = generate_snippet(res["content"], query)
            
    return results