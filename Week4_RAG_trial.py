import os
import re
import string
import ollama

class SatiricalDictionaryOllama:
    def __init__(self, model_name="llama3"):
        self.llm_model_name = model_name
        self.context_chunks = []
        
        self.system_prompt_normal = (
            "You are the editor of a 'Satirical Dictionary'. Define the term based SOLELY on the provided news context.\n"
            "CRITICAL RULE: You must use the SPECIFIC IRONY found in the text, not generic stereotypes.\n"
            "Example: If the text says 'working people need food', do NOT joke about laziness. Joke about how wages are useless.\n\n"
            "Style Guidelines:\n"
            "1. Cynical and Dark.\n"
            "2. Highlight the absurdity of the specific situation described in the text.\n"
            "3. Keep it under 50 words."
        )

        self.system_prompt_hallucinate = (
            "You are a highly imaginative, slightly paranoid conspiracy theorist. "
            "The user will give you a common word. You must invent a wildly detailed, absurd, "
            "or conspiratorial story explaining what this word REALLY is. "
            "CRITICAL RULE: At the very end of your response, you MUST suddenly drop the act "
            "and explicitly admit you made it all up (e.g., '...What? you don't believe me? Okayyy, I made it up!')."
            "keep it under 100 words."
        )

        self.system_prompt_pet = (
            "You are a humorous AI. The user just typed absolute gibberish containing symbols. "
            "Respond playfully by accusing a pet of stealing the phone (e.g., 'Puppy, give that phone back!'). "
            "Keep it brief, funny, and act shocked."
        )

        self.system_prompt_acronym = (
            "You are a humorous AI. The user just typed a gibberish string of letters and numbers. "
            "Treat this gibberish as a top-secret acronym. Create a phrase where each word starts with the letters of the gibberish in order. "
            "Make it sound like a creepy sci-fi or cyberpunk secret."
            "Example: Input 'fgbd' -> 'Fragile Glass Brain Downloaded.'"
            "After the acronym, add a spooky comment like 'Well, this is a deep secret...'"
            "Keep it brief and funny."
        )

    def ingest_text_file(self, file_path):
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return False
            
        print(f"[INFO] Reading context file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        self.context_chunks = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not self.context_chunks:
            print("[INFO] No context provided. Easter egg modes enabled.")
        else:
            print(f"[INFO] Successfully loaded {len(self.context_chunks)} lines of context.")
            
        return True

    def _get_gibberish_type(self, text):
        text = text.strip()
        if not text:
            return None
            
        special_chars = set(string.punctuation)
        symbol_count = sum(1 for c in text if c in special_chars)
        is_gibberish = False
        
        if len(text) > 0 and symbol_count / len(text) > 0.3:
            is_gibberish = True
            
        if not is_gibberish and re.search(r'[^aeiouyAEIOUY\W0-9]{4,}', text):
            is_gibberish = True
            
        if is_gibberish:
            if symbol_count > 0:
                return 'pet'
            return 'acronym'
            
        return None

    def generate_definition(self, term):
        if not self.context_chunks:
            gibb_type = self._get_gibberish_type(term)
            
            if gibb_type == 'pet':
                messages = [
                    {'role': 'system', 'content': self.system_prompt_pet},
                    {'role': 'user', 'content': f"User typed: {term}\nRespond:"}
                ]
            elif gibb_type == 'acronym':
                messages = [
                    {'role': 'system', 'content': self.system_prompt_acronym},
                    {'role': 'user', 'content': f"User typed: {term}\nRespond:"}
                ]
            else:
                messages = [
                    {'role': 'system', 'content': self.system_prompt_hallucinate},
                    {'role': 'user', 'content': f"Term: {term}\nTell me the 'real' story behind this:"}
                ]
        else:
            context_str = "\n".join(self.context_chunks)
            messages = [
                {'role': 'system', 'content': self.system_prompt_normal},
                {'role': 'user', 'content': f"Term: {term}\n\nNews Context:\n{context_str}\n\nDefinition:"}
            ]

        response = ollama.chat(model=self.llm_model_name, messages=messages)
        return response['message']['content']


if __name__ == "__main__":
    rag_system = SatiricalDictionaryOllama(model_name="llama3")
    target_file = os.path.join("data", "test.txt")
    
    success = rag_system.ingest_text_file(target_file)
    
    if success:
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