import os
import ollama
from sentence_transformers import SentenceTransformer, util

class SatiricalDictionaryOllama:
    def __init__(self, model_name="llama3"):
        """
        Initializes the RAG pipeline using Ollama for generation 
        and SentenceTransformer for retrieval.
        """
        print(f"[INFO] Loading embedding model for retrieval...")
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        self.llm_model_name = model_name
        self.context_embeddings = None
        self.context_chunks = []
        
        # System prompt for the 'Defamiliarization' style
        self.system_prompt = (
    "You are the editor of a 'Satirical Dictionary'. Define the term based SOLELY on the provided news context.\n"
    "CRITICAL RULE: You must use the SPECIFIC IRONY found in the text, not generic stereotypes.\n"
    "Example: If the text says 'working people need food', do NOT joke about laziness. Joke about how wages are useless.\n\n"
    "Style Guidelines:\n"
    "1. Cynical and Dark.\n"
    "2. Highlight the absurdity of the specific situation described in the text.\n"
    "3. Keep it under 50 words."
)

    def ingest_text_file(self, file_path):
        """
        Reads a .txt file and creates embeddings.
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            print(f"[HINT] Please check if 'data' folder exists and contains 'test.txt'")
            return False
            
        print(f"[INFO] Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Simple chunking logic (splitting by newlines)
        self.context_chunks = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not self.context_chunks:
            print("[WARNING] The file is empty or contains only whitespace.")
            return False

        print(f"[INFO] Encoding {len(self.context_chunks)} text chunks...")
        self.context_embeddings = self.embed_model.encode(self.context_chunks, convert_to_tensor=True)
        print("[INFO] Ingestion complete.")
        return True

    def retrieve(self, query, top_k=3):
        """
        Finds relevant text chunks using vector similarity.
        """
        if self.context_embeddings is None:
            return []

        query_embedding = self.embed_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.context_embeddings, top_k=top_k)
        
        relevant_texts = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            relevant_texts.append(self.context_chunks[idx])
            
        return relevant_texts

    def generate_definition(self, term):
        """
        Calls Ollama API to generate the definition using the FULL text context.
        """
        # 1. Join all chunks to form the full article text
        if not self.context_chunks:
            return "Error: No text file loaded. Please load a file first."
            
        context_str = "\n".join(self.context_chunks)
        

        # 2. Call Ollama
        response = ollama.chat(model=self.llm_model_name, messages=[
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': f"Term: {term}\n\nFull News Article:\n{context_str}\n\nDefinition:",
            },
        ])
        
        return response['message']['content']
# Execution / Demo

if __name__ == "__main__":
    # 1. Initialize System
    rag_system = SatiricalDictionaryOllama(model_name="llama3")
    
    # 2. Load Data from specific path
    target_file = os.path.join("data", "test.txt")
    
    success = rag_system.ingest_text_file(target_file)
    
    if success:
        # 3. Interactive Loop
        print("\n--- Satirical Dictionary Demo (Type 'exit' to quit) ---")
        while True:
            user_input = input("\nEnter a term: ")
            if user_input.lower() == 'exit':
                break
            
            try:
                definition = rag_system.generate_definition(user_input)
                print(f"> Definition: {definition}")
            except Exception as e:
                print(f"[ERROR] Communication with Ollama failed: {e}")
    else:
        print("[SYSTEM] Exiting due to file load error.")