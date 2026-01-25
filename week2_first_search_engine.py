import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import re         
import os
import glob

# Issue 11-Implement the data we scraped on week1
# to be usable for the boolean search tutorial -Linyao

# Added globals to support the new rewrite_token logic
t2i = {}
documents = []

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

        # Updated: Combine Data now includes 'Full_Text' for full-text search
        df["content"] = "[" + df["Time"] + "] " + df["Category"] + ": " + df["Headline"] + "\n\n" + df["Full_Text"]

        return df["content"].tolist()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# Issue 12, performing a query with a parser. This is mostly copied from the tutorial once again.

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements

def rewrite_token(t):
    if t in d:
        return d[t]
    
    term = t.lower()
    
    if term not in t2i:
        return f'np.zeros((1, {len(documents)}), dtype=int)'
    
    return f'sparse_td_matrix[t2i["{term}"]].todense()'

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

# Added: Helper function for Smart Snippets
def get_smart_snippet(text, query, window=100):
    words = re.findall(r'\w+', query)
    keywords = [w for w in words if w.lower() not in ['and', 'or', 'not']]
    if not keywords: return text[:200] + "..."
    
    match = re.search(re.escape(keywords[0]), text, re.IGNORECASE)
    if not match: return text[:200] + "..."
    
    start = max(0, match.start() - window)
    end = min(len(text), match.start() + window)
    return "..." + text[start:end] + "..."

# Added: Helper function for Highlighting
def highlight_text(text, query):
    keywords = [w for w in re.findall(r'\w+', query) if w.lower() not in ['and', 'or', 'not']]
    for word in keywords:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"\033[1;36m\g<0>\033[0m", text)
    return text

# Testing Block
if __name__ == "__main__":

    # Test the data loader
    documents = get_week1_documents()

    if documents:
        print(f"\n[Success] Loaded {len(documents)} documents.")
        print("\nVerifying Compatibility with Search Engine")

        try:
            cv = CountVectorizer(
                lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b"
            )

            sparse_matrix = cv.fit_transform(documents)
            t2i = cv.vocabulary_
            sparse_td_matrix = sparse_matrix.T.tocsr() 

            print("Vectorization successful!")
            print(f"Vocabulary size: {len(t2i)} unique words.")
            
            print("The data is ready. Enter query (or '#' to exit):")
            
            while True:
                user_input = input("\nQuery > ").strip()
                
                # Update: Changed exit condition to '#' only
                if user_input == "#":
                    print("Goodbye!")
                    break
                
                # Skip empty input to avoid errors
                if user_input == "":
                    continue
                
                try:
                    hits_matrix = eval(rewrite_query(user_input))
                    hits_list = list(np.array(hits_matrix).flatten().nonzero()[0])
                    print(f"Found {len(hits_list)} matches.")

                    for idx in hits_list[:5]:
                        snippet = get_smart_snippet(documents[idx], user_input)
                        print(f"[-] {highlight_text(snippet, user_input)}")

                except Exception as e:
                    print(f"Error: {e}")

        except Exception as e:
            print(f"Vectorization failed: {e}")
    else:
        print("No documents were loaded.")