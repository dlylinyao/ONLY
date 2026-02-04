import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import glob
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

# [New] Global variable: Store original vocabulary (Essential for fixing wildcard functionality)
original_vocabulary = set()

def stemmed_tokenizer(text):
    
    tokens = re.findall(r"(?u)\b\w+\b", text.lower())
    
    # [New Logic] Save original words before stemming
    if tokens:
        original_vocabulary.update(tokens)
    
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

# Data Loading

def get_week1_documents():
    folder_path = "data"
    file_pattern = "yle_business_culture_*.csv"
    search_path = os.path.join(folder_path, file_pattern)
    files = glob.glob(search_path)
    if not files:
        print(f"Error: No files matching '{file_pattern}' found in '{folder_path}'.")
        return []
    latest_file = max(files, key=os.path.getctime)
    try:
        print(f"Loading data from {latest_file}...")
        df = pd.read_csv(latest_file)
        for col in ["Time", "Category", "Headline", "Full_Text"]:
            if col not in df.columns: df[col] = "" 
            df[col] = df[col].fillna("")
        df["content"] = "[" + df["Time"] + "] " + df["Category"] + ": " + df["Headline"] + "\n\n" + df["Full_Text"]
        return df["content"].tolist()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_smart_snippet(text, query, window=100):
    # Remove Boolean operators to find keywords for the snippet
    clean_query = re.sub(r'\b(AND|OR|NOT)\b', '', query, flags=re.IGNORECASE)
    
    # [Logic Update] Support searching for terms with wildcards
    words = re.findall(r'[\w\*]+', clean_query) 
    if not words: return text[:200] + "..."
    
    # Find the first match (supports regex search)
    first_match_index = -1
    for word in words:
        if "*" in word:
            pattern_str = re.escape(word).replace(r'\*', r'\w*')
            match = re.search(pattern_str, text, re.IGNORECASE)
        else:
            match = re.search(re.escape(word), text, re.IGNORECASE)
        
        if match:
            first_match_index = match.start()
            break
            
    if first_match_index == -1: return text[:200] + "..."
    
    start = max(0, first_match_index - window)
    end = min(len(text), first_match_index + window)
    return "..." + text[start:end] + "..."

def highlight_text(text, query):
    # [Logic Update] Highlight function now supports wildcards and stem matching
    clean_query = re.sub(r'\b(AND|OR|NOT)\b', '', query, flags=re.IGNORECASE)
    query_words = re.findall(r'[\w\*]+', clean_query)
    
    text_words = set(re.findall(r'\w+', text.lower()))
    words_to_highlight = set()

    for q_word in query_words:
        q_word_lower = q_word.lower()
        if "*" in q_word_lower:
            matches = enable_wildcards(q_word_lower, text_words)
            words_to_highlight.update(matches)
        else:
            q_stem = stemmer.stem(q_word_lower)
            for t_word in text_words:
                if stemmer.stem(t_word) == q_stem:
                    words_to_highlight.add(t_word)

    sorted_words = sorted(list(words_to_highlight), key=len, reverse=True)
    for word in sorted_words:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        text = pattern.sub(f"\033[1;36m\\g<0>\033[0m", text)
    return text

# Search Logic

documents = []
t2i = {}
sparse_td_matrix = None
tfidf_vectorizer = None
tfidf_matrix = None
bert_model = None
bert_embeddings = None

d = {"and": "&", "AND": "&", "or": "|", "OR": "|", "not": "1 -", "NOT": "1 -", "(": "(", ")": ")"}

# wildcards regex function (works quite nice with *ord and wor* but badly with w*d)
def enable_wildcards(term, vocabulary):
    pattern = re.escape(term)
    # [Logic Update] Actually fixed the w*d issue by changing \w+ to \w*
    pattern = pattern.replace(r'\*', r'\w*') 
    matches = [v for v in vocabulary if re.fullmatch(pattern, v)]
    return matches

def rewrite_token(t):
    if t in d: return d[t]
    
    # [Logic Update] Modified logic here to use original_vocabulary
    if "*" in t: # handle wildcard cases
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

def search_boolean(query):
    try:
        hits_matrix = eval(rewrite_query(query))
        hits_list = list(np.array(hits_matrix).flatten().nonzero()[0])
        return hits_list, None 
    except Exception as e:
        print(f"Boolean Search Error: {e}")
        return [], None

def search_tfidf(query):
    # [Logic Update] Added query expansion to solve the issue where gover* could not find government
    expanded_query_terms = []
    for word in query.split():
        if "*" in word:
            matches = enable_wildcards(word.lower(), original_vocabulary)
            if matches: expanded_query_terms.extend(matches)
        else:
            expanded_query_terms.append(word)
    
    expanded_query_str = " ".join(expanded_query_terms) if expanded_query_terms else query

    query_vec = tfidf_vectorizer.transform([expanded_query_str])
    cosine_similarities = np.dot(query_vec, tfidf_matrix.T).toarray()[0]
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    matches = [(idx, cosine_similarities[idx]) for idx in ranked_indices if cosine_similarities[idx] > 0]
    return matches

def search_semantic(query):
    query_embedding = bert_model.encode([query])
    cosine_similarities = np.dot(query_embedding, bert_embeddings.T)[0]
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    
    # Threshold to filter out low-relevance results
    threshold = 0.1
    matches = [(idx, cosine_similarities[idx]) for idx in ranked_indices if cosine_similarities[idx] > threshold]
    return matches

# Main Program
if __name__ == "__main__":
    documents = get_week1_documents()

    if documents:
        print(f"\n[Init] Loaded {len(documents)} documents.")
        
        # Boolean Index
        # print("Building Boolean Index...", end=" ")
        print("Building Boolean Index (with Stemming and wildcards)...", end=" ")
        # cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
        cv = CountVectorizer(lowercase=True, binary=True, tokenizer=stemmed_tokenizer)
        sparse_matrix = cv.fit_transform(documents)
        t2i = cv.vocabulary_
        sparse_td_matrix = sparse_matrix.T.tocsr()
        print("Done.")

        # TF-IDF Index
        # print("Building TF-IDF Index...", end=" ")
        print("Building TF-IDF Index (Stemming + 1-2 ngrams)...", end=" ")
        # tfidf_vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
        tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            sublinear_tf=True, 
            use_idf=True, 
            norm="l2",
            tokenizer=stemmed_tokenizer, # Stemming
            ngram_range=(1, 3)           # 3 Phrases 
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        print("Done.")

        # Semantic Model
        print("Loading AI Model (Semantic)...", end=" ")
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        bert_embeddings = bert_model.encode(documents)
        print("Done.")

        print("\n" + "="*50)
        # print("SEARCH ENGINE READY")
        print("SEARCH ENGINE READY") 
        print("Modes: [1] Boolean  [2] TF-IDF  [3] Semantic AI")
        print("Type '#1', '#2', or '#3' to switch modes. Default is TF-IDF.")
        print("Type '#' to quit.")
        print("="*50)

        current_mode = "2"
        mode_map = {"1": "Boolean", "2": "TF-IDF", "3": "Semantic"}

        while True:
            display_name = mode_map.get(current_mode)
            user_input = input(f"\n[{display_name} Search] > ").strip()

            if user_input == "#": break
            
            # Switch mode logic
            if user_input in ["#1", "#2", "#3"]:
                new_mode = user_input[1]
                current_mode = new_mode
                new_name = mode_map.get(current_mode)
                print(f"Switched to {new_name} mode.")
                continue
            
            if user_input == "": continue

            results = []
            
            if current_mode == "1":
                indices, _ = search_boolean(user_input)
                results = [(i, 1.0) for i in indices]
            elif current_mode == "2":
                results = search_tfidf(user_input)
            elif current_mode == "3":
                results = search_semantic(user_input)

            print(f"Found {len(results)} matches.")

            for idx, score in results[:5]:
                doc_content = documents[idx]
                snippet = get_smart_snippet(doc_content, user_input)
                highlighted = highlight_text(snippet, user_input)
                
                # Show score only for non-Boolean modes
                score_str = f"(Score: {score:.4f})" if current_mode != "1" else ""
                print(f"[-] {score_str} {highlighted}")

    else:
        print("No documents loaded.")