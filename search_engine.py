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

# 1. Global Variables 

documents = []

# Boolean Search Variables
cv = None
sparse_td_matrix = None
t2i = {}

# TF-IDF Search Variables
tfidf_vectorizer = None
tfidf_matrix = None

# Semantic Search Variables
bert_model = None
bert_embeddings = None

# Boolean Operators Map
d = {"and": "&", "AND": "&", "or": "|", "OR": "|", "not": "1 -", "NOT": "1 -", "(": "(", ")": ")"}


# 2. Helper Functions 

def stemmed_tokenizer(text):

    tokens = re.findall(r"(?u)\b\w+\b", text.lower())
    if tokens:
        original_vocabulary.update(tokens)
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

def get_week1_documents():
    
    folder_path = "data"
    file_pattern = "filtered_yle_business_culture_*.csv"
    search_path = os.path.join(folder_path, file_pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"[ERROR] No files matching '{file_pattern}' found in '{folder_path}'.")
        return []
    
    latest_file = max(files, key=os.path.getctime)
    try:
        print(f"[DATA] Loading data from {latest_file}...")
        df = pd.read_csv(latest_file)
        
        for col in ["Time", "Category", "Headline", "Full_Text"]:
            if col not in df.columns: df[col] = "" 
            df[col] = df[col].fillna("")
        
        
        df["content"] = "[" + df["Time"] + "] " + df["Category"] + ": " + df["Headline"] + "\n\n" + df["Full_Text"]
        return df["content"].tolist()
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading data: {e}")
        return []

def enable_wildcards(term, vocabulary):
 
    pattern = re.escape(term)
    pattern = pattern.replace(r'\*', r'\w*') 
    matches = [v for v in vocabulary if re.fullmatch(pattern, v)]
    return matches

def rewrite_token(t):
    
    if t in d: return d[t]
    
    # Handle wildcards
    if "*" in t: 
        raw_matches = enable_wildcards(t.lower(), original_vocabulary)
        stemmed_matches = {stemmer.stem(m) for m in raw_matches}
        valid_stems = [s for s in stemmed_matches if s in t2i]
        
        if not valid_stems: 
           return f'np.zeros((1, {len(documents)}), dtype=int)'
        
        parts = [f'sparse_td_matrix[t2i["{s}"]].todense()' for s in valid_stems]
        return " | ".join(parts)
        
    term = stemmer.stem(t.lower())
    if term not in t2i: return f'np.zeros((1, {len(documents)}), dtype=int)'
    return f'sparse_td_matrix[t2i["{term}"]].todense()'

def rewrite_query(query):
    return " ".join(rewrite_token(t) for t in query.split())


# 3. Initialization Logic 


def init_all_engines():
    global documents, cv, sparse_td_matrix, t2i, tfidf_vectorizer, tfidf_matrix, bert_model, bert_embeddings
    
    print("[INIT] Starting engine initialization...")
    documents = get_week1_documents()
    if not documents:
        print("[ERROR] Initialization failed: No documents loaded.")
        return

    print(f"[INIT] Loaded {len(documents)} documents.")

    # 1. Initialize Boolean
    print("[INIT] Building Boolean Index...", end=" ")
    cv = CountVectorizer(lowercase=True, binary=True, tokenizer=stemmed_tokenizer)
    sparse_matrix = cv.fit_transform(documents)
    t2i = cv.vocabulary_
    sparse_td_matrix = sparse_matrix.T.tocsr()
    print("Done.")

    # 2. Initialize TF-IDF
    print("[INIT] Building TF-IDF Index...", end=" ")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, use_idf=True, norm="l2",
        tokenizer=stemmed_tokenizer, ngram_range=(1, 3)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    print("Done.")

    # 3. Initialize Semantic
    print("[INIT] Loading Semantic Model (This may take a moment)...", end=" ")
    try:
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        bert_embeddings = bert_model.encode(documents)
        print("Done.")
    except Exception as e:
        print(f"\n[WARNING] Semantic model failed to load: {e}")
    
    print("[INIT] All search engines are ready.")


# 4. Search Functions 


def search_boolean(query, top_k=5):
    try:
        hits_matrix = eval(rewrite_query(query))
        hits_list = list(np.array(hits_matrix).flatten().nonzero()[0])
        
        results = []
        for idx in hits_list[:top_k]: # Limit to top_k
            results.append({
                "doc_id": int(idx),
                "score": 1.0, 
                "content": documents[idx]
            })
        return results
    except Exception as e:
        print(f"[ERROR] Boolean Search Error: {e}")
        return []

def search_tfidf(query, top_k=5):
    if not query: return []
    # Expansion for wildcards in TF-IDF
    expanded_query_terms = []
    for word in query.split():
        if "*" in word:
            matches = enable_wildcards(word.lower(), original_vocabulary)
            if matches: expanded_query_terms.extend(matches)
        else:
            expanded_query_terms.append(word)
    
    expanded_query_str = " ".join(expanded_query_terms) if expanded_query_terms else query

    query_vec = tfidf_vectorizer.transform([expanded_query_str])
    cosine_scores = np.dot(query_vec, tfidf_matrix.T).toarray()[0]
    ranked_indices = np.argsort(cosine_scores)[::-1]
    
    results = []
    for idx in ranked_indices[:top_k]:
        if cosine_scores[idx] > 0:
            results.append({
                "doc_id": int(idx),
                "score": float(cosine_scores[idx]),
                "content": documents[idx]
            })
    return results

def search_semantic(query, top_k=5):
    if bert_model is None:
        return [{"doc_id": -1, "score": 0, "content": "Semantic search is not available."}]

    query_embedding = bert_model.encode([query])
    cosine_similarities = np.dot(query_embedding, bert_embeddings.T)[0]
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    
    results = []
    for idx in ranked_indices[:top_k]:
        # Filter low relevance
        if cosine_similarities[idx] > 0.1: 
             results.append({
                "doc_id": int(idx),
                "score": float(cosine_similarities[idx]),
                "content": documents[idx]
            })
    return results

def search(query, mode="tfidf", top_k=5):
    
    print(f"[SEARCH] Query: '{query}' | Mode: '{mode}'")
    if mode == "boolean":
        return search_boolean(query, top_k)
    elif mode == "semantic":
        return search_semantic(query, top_k)
    else:
        # default TF-IDF
        return search_tfidf(query, top_k)