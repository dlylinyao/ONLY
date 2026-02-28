
import semantic_search as se
from RAG_trial import SatiricalDictionaryOllama
import json

#THIS FILE IS FOR MAKING THE AUTOMATED PIPELINE THAT RUNS THE SEMANTIC SEARCH AND THEN FOR ALL 
#INPUTS WE HAVE IN WORDS.TXT RUNS THE RAG TOO

print("get data, engine and rag")
se.init_engine()
rag_system = SatiricalDictionaryOllama(model_name="llama3")

definitions = {}

print("open the file and perform query for each word and feed them to rag")
with open("words.txt", "r", encoding="utf-8") as inputfile:
    for line in inputfile:
        word = line.strip()
        if not word:
            continue
        results = se.search(word, top_k=3)
        context_list = [res["content"] for res in results] if results else []
        rag_system.ingest_context_list(context_list)
        definition = rag_system.generate_definition(word)
        definitions[word] = definition


with open("definitions.json", "w", encoding="utf-8") as outputfile:
    json.dump(definitions, outputfile, ensure_ascii=False, indent=4)
