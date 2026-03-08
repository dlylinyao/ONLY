import semantic_search as se
from RAG import SatiricalDictionaryOllama
from ollama_withoutRAG import SatiricalDictionaryOllamaNoRAG
import json
import pandas as pd
import os

# THIS FILE IS FOR MAKING THE AUTOMATED PIPELINE THAT RUNS THE SEMANTIC SEARCH AND THEN FOR ALL
# INPUTS WE HAVE IN WORDS.TXT RUNS THE RAG TOO

print("get data, engine and rag")
se.init_engine()
rag_system = SatiricalDictionaryOllama(model_name="llama3")
without_rag_system = SatiricalDictionaryOllamaNoRAG(model_name="llama3")

# Load CSV File
csv_filename = "data/Definitions_Generation - Sheet1.csv"
output_filename = "data/Definitions_Generation_Results_V2.csv"

print(f"Loading words from {csv_filename}...")
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(
        f"Error: {csv_filename} not found. Please make sure it's in the same directory."
    )
    exit()
# Ensure the 'Word' column exists
if "Word" not in df.columns:
    print("Error: The CSV file must contain a 'Word' column.")
    exit()

if "Definition (A-Rag)" not in df.columns:
    df["Definition (A-Rag)"] = ""
if "Definition (B-non-Rag)" not in df.columns:
    df["Definition (B-non-Rag)"] = ""
if "Highest similarity score" not in df.columns:
    df["Highest similarity score"] = ""
if "Chunks Loaded" not in df.columns:
    df["Chunks Loaded"] = ""


print("Starting generation process...")

# Iterate through words and generate definitions (combined loop)
for index, row in df.iterrows():
    word = str(row["Word"]).strip()

    if pd.isna(word) or not word or word.lower() == "nan":
        continue

    print(f"Processing ({index+1}/{len(df)}): {word}")

    # RAG Generation
    try:
        for_rag=True
        raw_results = se.search(word, top_k=5, for_rag=True, threshold=-1.0)
        highest_score = None
        if raw_results:
            highest_score = max([res.get("score", res.get("similarity", 0)) for res in raw_results])
            
        results = [res for res in raw_results if res.get("score", res.get("similarity", 0)) > 0.25]
        
        df.at[index, "Highest similarity score"] = highest_score
        df.at[index, "Chunks Loaded"] = len(results) if results else 0
        

        context_list = [res["content"] for res in results] if results else []
        rag_system.ingest_context_list(context_list)
        rag_def = rag_system.generate_definition(word)
        df.at[index, "Definition (A-Rag)"] = rag_def
    except Exception as e:
        print(f"Error generating RAG definition for {word}: {e}")
        df.at[index, "Definition (A-Rag)"] = f"Error: {e}"

    # RAG Generation
    try:
        no_rag_def = without_rag_system.generate_definition(word)
        df.at[index, "Definition (B-non-Rag)"] = no_rag_def
    except Exception as e:
        print(f"Error generating non-RAG definition for {word}: {e}")
        df.at[index, "Definition (B-non-Rag)"] = f"Error: {e}"
        
all_columns = df.columns.tolist()


preferred_order = [
    "Word", 
    "Highest similarity score", 
    "Chunks Loaded", 
    "Definition (A-Rag)", 
    "Definition (B-non-Rag)"
]


remaining_columns = [col for col in all_columns if col not in preferred_order]
final_columns_order = preferred_order + remaining_columns


df = df[final_columns_order]

# Save results back to CSV
print(f"Saving results to {output_filename}...")
df.to_csv(output_filename, index=False, encoding="utf-8")
print("Done! Both RAG and non-RAG definitions are ready and saved.")
