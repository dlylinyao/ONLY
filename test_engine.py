import search_engine_tfidf as se

se.init_tfidf_engine()
results = se.tfidf_search("economy", top_k=3)

for r in results:
    print(r["score"])
    print(r["content"][:200])
    print("-" * 40)
