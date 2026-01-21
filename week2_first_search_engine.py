import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Issue 11-Implement the data we scraped on week1
# to be usable for the boolean search tutorial -Linyao


def get_week1_documents():

    # Define the file path
    file_path = "week1ylenews_2026-01-18.csv"

    try:
        print(f"Loading data from {file_path}...")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Data Cleaning
        df["Time"] = df["Time"].fillna("")
        df["Category"] = df["Category"].fillna("")
        df["Headline"] = df["Headline"].fillna("")

        # Combine Data
        df["content"] = "[" + df["Time"] + "] " + df["Category"] + ": " + df["Headline"]

        # Convert to List
        documents = df["content"].tolist()

        return documents

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

#Issue 12, performing a query with a parser. This is mostly copied from the tutorial once again.

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements

def rewrite_token(t):
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()

# Testing Block
# It validates that the data is ready for the search engine.
if __name__ == "__main__":

    # Test the data loader
    documents = get_week1_documents()

    if documents:
        print(f"\n[Success] Loaded {len(documents)} documents.")
        print(f"[Preview] First document: {documents[0]}")

        print("\nVerifying Compatibility with Search Engine")

        try:
            # Test Vectorization

            cv = CountVectorizer(
                lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b"
            )

            sparse_matrix = cv.fit_transform(documents)
            t2i = cv.vocabulary_
            sparse_td_matrix = sparse_matrix.T.tocsr() # to csr for bigger scale and conversion to term-document

            print("Vectorization successful!")
            print(f"Vocabulary size: {len(t2i)} unique words.")
            print("The data is ready for the Boolean Search implementation.")

        except Exception as e:
            print(f"Vectorization failed: {e}")
    else:
        print("No documents were loaded.")

    # test queries 

    test_query("foreign AND NOT education")
    test_query("NOT education OR foreign")
    test_query("( NOT foreign OR education ) AND policy") # AND, OR, NOT can be written either in ALLCAPS
    test_query("( not foreign or education ) and policy") # ... or all small letters
    test_query("not foreign and not policy")   