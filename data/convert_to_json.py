import pandas as pd
import json
import os


current_dir = os.path.dirname(os.path.abspath(__file__))


csv_file = os.path.join(current_dir, "Definitions_Generation_Results_V2.csv")
json_file = os.path.join(current_dir, "Definitions_Generation_Results_ID_V2.json")

print(f"Reading the path: {csv_file}")


df = pd.read_csv(csv_file)


formatted_data = []


for index, row in df.iterrows():
    
    base_id = int(row['ID']) if 'ID' in df.columns else index + 1
    word = str(row['Word'])
    
    
    entry = {
        "word_id": f"word_{base_id}",
        "word": word,
        "definitions": [
            {
                "definition_id": f"def_rag_{base_id}",
                "type": "RAG",
                "text": str(row['Definition (A-Rag)'])
            },
            {
                "definition_id": f"def_non_rag_{base_id}",
                "type": "non-RAG",
                "text": str(row['Definition (B-non-Rag)'])
            }
        ]
    }
    formatted_data.append(entry)


with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"Succeed: {json_file}")