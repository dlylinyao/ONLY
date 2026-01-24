import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import re         

# Issue 11-Implement the data we scraped on week1
# to be usable for the boolean search tutorial -Linyao

# Added globals to support the new rewrite_token logic
t2i = {}
documents = []

def get_week1_documents():

    file_path = "week1ylenews_2026-01-24.csv"

    try:
        print(f"Loading data from {file_path}...")

        df = pd.read_csv(file_path)

        for col in ["Time", "Category", "Headline", "text"]:
            if col not in df.columns: df[col] = "" 
            df[col] = df[col].fillna("")

        # Updated: Combine Data now includes 'text' for full-text search
        df["content"] = "[" + df["Time"] + "] " + df["Category"] + ": " + df["Headline"] + "\n\n" + df["text"]

        return df["content"].tolist()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# Issue 12, performing a query with a parser. This is mostly copied from the tutorial once again.

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements

def rewrite_token(t):
    # Updated: Logic to handle unknown words (returns zero vector instead of crashing)
    if t not in d and t not in t2i:
        return f'np.zeros((1, {len(documents)}), dtype=int)'
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))

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
            
            print("The data is ready. Enter query (or 'quit' to exit):")
            
            while True:
                user_input = input("\nQuery > ").strip()
                if user_input == "" or user_input.lower() == 'quit': break
                
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