import os
import ssl
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import nltk
import ollama

# --- 1. Environment Preparation ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def get_satirical_title_from_llama(topic_id, keywords, sample_docs):
    """Use Llama3 to generate a satirical title based on topic keywords and sample documents."""
    system_prompt = (
        "You are the cynical editor of a 'Satirical Dictionary'. "
        "Give this news topic a SHORT, SATIRICAL title (max 5 words). "
        "ONLY output the title. No quotes."
    )
    user_prompt = f"Keywords: {', '.join(keywords)}\n\nExamples:\n" + "\n".join([f"- {d[:150]}" for d in sample_docs])
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ])
        return response['message']['content'].strip().strip('"')
    except:
        return f"Topic {topic_id}"

def run_clustering_test():
    # --- 2. data ---
    file_name = "filtered_yle_business_culture_2026-02-12.csv"
    csv_path = os.path.join("data", file_name)
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Cannot find file: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    col_text, col_title, col_url, col_category = "Full_Text", "Headline", "URL", "Category"
    df = df.dropna(subset=[col_text])
    
    docs = df[col_text].astype(str).tolist()
    titles = df[col_title].tolist()
    urls = df[col_url].tolist()
    categories = df[col_category].tolist() if col_category in df.columns else ["N/A"] * len(docs)
    
    # --- 3. Clustering ---
    print("[INFO] Encoding texts...")
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embed_model.encode(docs, show_progress_bar=True)

    vectorizer_model = CountVectorizer(stop_words=stopwords.words('english'))
    umap_model = UMAP(n_neighbors=5, n_components=2, min_dist=0.01, metric='cosine', random_state=42)
    topic_model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer_model, min_topic_size=5)

    topics, _ = topic_model.fit_transform(docs, embeddings)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    # --- 4. Llama3 naming ---
    print("[INFO] Llama3 is naming topics...")
    topic_info = topic_model.get_topic_info()
    topic_mapping = {-1: "Unclassified"}
    for t_id in topic_info['Topic']:
        if t_id != -1:
            keywords = [w for w, s in topic_model.get_topic(t_id)[:5]]
            samples = topic_model.get_representative_docs(t_id)[:2]
            topic_mapping[t_id] = get_satirical_title_from_llama(t_id, keywords, samples)

    # --- 5. DataFrame ---
    plot_df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "topic_id": topics,
        "headline": titles,
        "url": urls,
        "category": categories  
    })
    plot_df['topic_name'] = plot_df['topic_id'].map(topic_mapping)

# --- 6. visualization and interaction ---
    df_visible = plot_df[plot_df['topic_id'] != -1]
    
    fig = px.scatter(
        df_visible, 
        x="x", 
        y="y", 
        color="topic_name",
        custom_data=["headline", "url"], 
        title="ONLY Dictionary - Click any point to read on Yle",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br><br>URL: %{customdata[1]}<extra></extra>"
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"))

    # --- 7. Add JavaScript ---
    html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
    
    script_js = """
    <script>
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        plotDiv.on('plotly_click', function(data){
            var url = data.points[0].customdata[1]; 
            if(url && url !== "nan" && url !== "") {
                window.open(url, '_blank');
            }
        });
    </script>
    """
    html_with_click = html_content.replace('</body>', script_js + '</body>')

    output_html = os.path.join("data", "cluster_plot.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_with_click)
        
    print(f"[SUCCESS] Plot saved to {output_html}")

if __name__ == "__main__":
    run_clustering_test()