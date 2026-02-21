import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import glob
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")
original_vocabulary = set()

documents = []

cv = None
sparse_td_matrix = None
t2i = {}

tfidf_vectorizer = None
tfidf_matrix = None

bert_model = None
bert_embeddings = None

d = {
    "and": "&",
    "AND": "&",
    "or": "|",
    "OR": "|",
    "not": "1 -",
    "NOT": "1 -",
    "(": "(",
    ")": ")",
}


def stemmed_tokenizer(text):
    tokens = re.findall(r"(?u)\b\w+\b", text.lower())
    if tokens:
        original_vocabulary.update(tokens)
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed


def get_news_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "data")

    file_pattern = "yle_news_*.csv"
    search_path = os.path.join(folder_path, file_pattern)
    files = glob.glob(search_path)

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


def enable_wildcards(term, vocabulary):
    pattern = re.escape(term)
    pattern = pattern.replace(r"\*", r"\w*")
    matches = [v for v in vocabulary if re.fullmatch(pattern, v)]
    return matches


def rewrite_token(t):
    if t in d:
        return d[t]

    if "*" in t:
        raw_matches = enable_wildcards(t.lower(), original_vocabulary)
        stemmed_matches = {stemmer.stem(m) for m in raw_matches}
        valid_stems = [s for s in stemmed_matches if s in t2i]

        if not valid_stems:
            return f"np.zeros((1, {len(documents)}), dtype=int)"

        parts = [f'sparse_td_matrix[t2i["{s}"]].todense()' for s in valid_stems]
        return " | ".join(parts)

    term = stemmer.stem(t.lower())
    if term not in t2i:
        return f"np.zeros((1, {len(documents)}), dtype=int)"
    return f'sparse_td_matrix[t2i["{term}"]].todense()'


def rewrite_query(query):
    return " ".join(rewrite_token(t) for t in query.split())


def generate_snippet(text, query, window_size=80):
    parts = text.split("\n\n", 1)
    headline = parts[0]
    body = parts[1] if len(parts) > 1 else text

    if not query:
        return f"[{headline}]\n\n{body[:window_size * 2]}..."

    clean_query = re.sub(r'\b(AND|OR|NOT|and|or|not)\b', '', query).strip()
    first_term = clean_query.split()[0] if clean_query else query
    
    match = re.search(re.escape(first_term), body, re.IGNORECASE)
    
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


def init_all_engines():
    global documents, cv, sparse_td_matrix, t2i, tfidf_vectorizer, tfidf_matrix, bert_model, bert_embeddings

    documents = get_news_data()
    if not documents:
        return

    cv = CountVectorizer(lowercase=True, binary=True, tokenizer=stemmed_tokenizer)
    sparse_matrix = cv.fit_transform(documents)
    t2i = cv.vocabulary_
    sparse_td_matrix = sparse_matrix.T.tocsr()

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
        tokenizer=stemmed_tokenizer,
        ngram_range=(1, 3),
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    try:
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        bert_embeddings = bert_model.encode(documents)
    except Exception:
        pass


def search_boolean(query, top_k=5):
    try:
        hits_matrix = eval(rewrite_query(query))
        hits_list = list(np.array(hits_matrix).flatten().nonzero()[0])

        results = []
        for idx in hits_list[:top_k]:
            results.append(
                {"doc_id": int(idx), "score": 1.0, "content": documents[idx]}
            )
        return results
    except Exception:
        return []


def search_tfidf(query, top_k=5):
    if not query:
        return []
        
    expanded_query_terms = []
    for word in query.split():
        if "*" in word:
            matches = enable_wildcards(word.lower(), original_vocabulary)
            if matches:
                expanded_query_terms.extend(matches)
        else:
            expanded_query_terms.append(word)

    expanded_query_str = (
        " ".join(expanded_query_terms) if expanded_query_terms else query
    )

    query_vec = tfidf_vectorizer.transform([expanded_query_str])
    cosine_scores = np.dot(query_vec, tfidf_matrix.T).toarray()[0]
    ranked_indices = np.argsort(cosine_scores)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        if cosine_scores[idx] > 0:
            results.append(
                {
                    "doc_id": int(idx),
                    "score": float(cosine_scores[idx]),
                    "content": documents[idx],
                }
            )
    return results


def search_semantic(query, top_k=5):
    if bert_model is None:
        return []

    query_embedding = bert_model.encode([query])
    cosine_similarities = np.dot(query_embedding, bert_embeddings.T)[0]
    ranked_indices = np.argsort(cosine_similarities)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        if cosine_similarities[idx] > 0.1:
            results.append(
                {
                    "doc_id": int(idx),
                    "score": float(cosine_similarities[idx]),
                    "content": documents[idx],
                }
            )
    return results


def search(query, mode="tfidf", top_k=5):
    if mode == "boolean":
        results = search_boolean(query, top_k)
    elif mode == "semantic":
        results = search_semantic(query, top_k)
    else:
        results = search_tfidf(query, top_k)
        
    for res in results:
        res["content"] = generate_snippet(res["content"], query)
        
    return results