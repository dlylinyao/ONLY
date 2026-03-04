import os
import ssl
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import ollama
import glob

# Environment Preparation (SSL & NLTK)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


def get_satirical_title_from_llama(topic_id, keywords, sample_docs):
    """
    This function crafts a prompt for Llama3 to generate a witty, satirical title for a given topic cluster.
    """
    # System Prompt
    system_prompt = (
        "You are the witty author of a modern Satirical News Dictionary. "
        "Your task is to read the provided news snippets and create a dictionary entry. "
        "Step 1: Extract the core entity, concept, or subject from the news (1 to 3 words max). "
        "Step 2: Write a witty, slightly cynical, but insightful definition for it, based STRICTLY on the events described in these specific news snippets. "
        "CRITICAL FORMATTING RULES: "
        "1. Output EXACTLY in this format: [Your Word]: [Your Definition]. "
        "2. DO NOT output the literal words 'Term:', 'Word:', or 'Definition:'. "
        "3. Example 1 - Boeing: A company that occasionally forgets how to defy gravity. "
        "4. Example 2 - Food Inflation: A convenient excuse for eating more cheese and eggs. "
        "5. Do not include quotes, markdown formatting, or any conversational filler. ONLY output the final string."
    )

    context_text = "\n".join([f"- {doc[:200]}..." for doc in sample_docs])
    user_prompt = (
        f"Cluster Keywords: {', '.join(keywords)}\n\n"
        f"News Snippets (Context):\n{context_text}\n\n"
        "Now, generate the dictionary entry."
    )

    try:
        import ollama

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        result = response["message"]["content"].strip(" \"'\n")
        return result

    except Exception as e:
        print(f"LLM Error for topic {topic_id}: {e}")
        return f"Topic {topic_id}: A cluster too complex to define."


def run_clustering_pipeline():
    # --- 2. data ---

    # folder_path = "data"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "data")

    file_pattern = "yle_news_*.csv"
    search_path = os.path.join(folder_path, file_pattern)
    files = glob.glob(search_path)

    files = [f for f in files if "filtered_out" not in f]

    if not files:
        print(f"[ERROR] CSV not found in {folder_path}")
        return

    csv_path = max(files, key=os.path.getctime)
    print(f"[INFO] Reading latest data from: {csv_path}")

    df = pd.read_csv(csv_path)

    col_text, col_title, col_url = "Full_Text", "Headline", "URL"
    df = df.dropna(subset=[col_text])

    docs = df[col_text].astype(str).tolist()
    titles = df[col_title].tolist()
    urls = df[col_url].tolist()

    print("[INFO] Embedding and Clustering...")
    embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embeddings = embed_model.encode(docs, show_progress_bar=True)

    vectorizer_model = CountVectorizer(stop_words=stopwords.words("english"))

    umap_model = UMAP(
        n_neighbors=5, n_components=2, min_dist=0.01, metric="cosine", random_state=42
    )
    topic_model = BERTopic(
        language="english",
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=5,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)

    print("[INFO] Calling Llama3 for satirical naming...")
    topic_info = topic_model.get_topic_info()
    topic_mapping = {-1: "Miscellaneous Nonsense"}

    # ADDED: Set to collect relevant seed words for evaluation
    seed_words = set()

    for t_id in topic_info["Topic"]:
        if t_id != -1:
            keywords = [w for w, s in topic_model.get_topic(t_id)[:5]]
            samples = topic_model.get_representative_docs(t_id)[:2]
            topic_mapping[t_id] = get_satirical_title_from_llama(
                t_id, keywords, samples
            )
            print(f"   Topic {t_id} -> {topic_mapping[t_id]}")

            # ADDED: Extract keywords for the seed list
            for word, score in topic_model.get_topic(t_id):
                if len(word) > 2 and len(seed_words) < 60:
                    seed_words.add(word)

    print("[INFO] Formatting for UI...")

    results = pd.DataFrame({"topic_id": topics, "headline": titles, "url": urls})
    results["topic_name"] = results["topic_id"].map(topic_mapping)

    df_clean = results[results["topic_id"] != -1]

    sky_data = []
    for topic_name, group in df_clean.groupby("topic_name"):
        sky_data.append(
            {
                "topic": topic_name,
                "articles": group[["headline", "url"]].to_dict("records"),
                "count": len(group),
            }
        )

    # output_json = os.path.join("data", "topic_modeling_data.json")
    output_json = os.path.join(folder_path, "topic_modeling_data.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sky_data, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Topic Modeling Data JSON created at {output_json}")

    # ADDED: Save the 50 relevant seed words to a CSV file
    seed_words_list = list(seed_words)[:60]
    if seed_words_list:
        seed_df = pd.DataFrame({"Relevant_Words": seed_words_list})
        seed_csv_path = os.path.join(folder_path, "relevant_seed_words.csv")
        seed_df.to_csv(seed_csv_path, index=False, encoding="utf-8")
        print(f"[SUCCESS] Relevant seed words saved to {seed_csv_path}")


if __name__ == "__main__":
    run_clustering_pipeline()