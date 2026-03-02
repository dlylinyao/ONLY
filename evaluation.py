
import semantic_search as se
from RAG_trial import SatiricalDictionaryOllama
from ollama_withoutRAG import SatiricalDictionaryOllamaNoRAG
import json

#THIS FILE IS FOR MAKING THE AUTOMATED PIPELINE THAT RUNS THE SEMANTIC SEARCH AND THEN FOR ALL 
#INPUTS WE HAVE IN WORDS.TXT RUNS THE RAG TOO

print("get data, engine and rag")
se.init_engine()
rag_system = SatiricalDictionaryOllama(model_name="llama3")
without_rag_system = SatiricalDictionaryOllamaNoRAG(model_name="llama3")

RAG_definitions = {}

without_RAG_definitions = {}

print("open the file and perform query for each word and feed them to rag and without rag")
with open("words.txt", "r", encoding="utf-8") as inputfile:
    for line in inputfile:
        word = line.strip()
        if not word:
            continue
        results = se.search(word, top_k=3)
        context_list = [res["content"] for res in results] if results else []
        rag_system.ingest_context_list(context_list)
        definition = rag_system.generate_definition(word)
        RAG_definitions[word] = definition


with open("RAG_definitions.json", "w", encoding="utf-8") as outputfile:
    json.dump(RAG_definitions, outputfile, ensure_ascii=False, indent=4)

print("Rag definitions are ready")

with open("words.txt", "r", encoding="utf-8") as inputfile:
    for line in inputfile:
        word = line.strip()
        if not word:
            continue
        definition = without_rag_system.generate_definition(word)
        without_RAG_definitions[word] = definition


with open("without_RAG_definitions.json", "w", encoding="utf-8") as outputfile:
    json.dump(without_RAG_definitions, outputfile, ensure_ascii=False, indent=4)


print("No rag definitions are ready")
