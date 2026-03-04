import json
import pandas as pd
import random
import os  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, 'data', 'Definitions_Generation_Results_ID.json')
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Randomly sample 50 words from the 100 available words
random.seed(42)
sampled_data = random.sample(data, 50)


rows = []
for item in sampled_data:
    word = item['word']
    for dfn in item['definitions']:
        rows.append({
            'Word': word,
            'Definition': dfn['text'],
            'Model_Type': dfn['type'],         # 'RAG' or 'non-RAG'
            'Definition_ID': dfn['definition_id'] 
        })

df = pd.DataFrame(rows)

# Global Shuffle: Completely randomize the row order
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare the Annotation Sheet (Hide the 'Model_Type' from annotators)
annotation_sheet = df_shuffled[['Word', 'Definition']].copy()
annotation_sheet['Funniness (1-5)'] = ""
annotation_sheet['Political (1-5)'] = ""

# Prepare the Master Key (Keep all info for internal evaluation)
master_key = df_shuffled[['Word', 'Definition', 'Model_Type', 'Definition_ID']].copy()

output_file = 'Annotation_Task_Global_Shuffle.xlsx'
with pd.ExcelWriter(output_file) as writer:
    annotation_sheet.to_excel(writer, sheet_name='Annotation Task', index=False)
    master_key.to_excel(writer, sheet_name='Master Key', index=False)

print(f"[SUCCESS] Saved to {output_file}.")
print(f"Total rows for annotation: {len(annotation_sheet)} (50 words * 2 definitions).")