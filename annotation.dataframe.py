import pandas as pd
import json

master = pd.read_excel("data/Annotation_Only-3.xlsx", sheet_name="Master_Key")

master = master.rename(columns={
    "Definition_ID": "item_id",
    "Model_Type": "rag"
})

annotators = ["A1","A2","A3","A4","A5","A6"]

group_map = {
    "A1": "finnish", "A2": "finnish", "A3": "finnish",
    "A4": "international", "A5": "international", "A6": "international"
}

funny_long = master.melt(
    id_vars=["item_id","rag","Score_Funniness","Score_Political"],
    value_vars=[f"Funniness_{i}" for i in range(1,7)],
    var_name="annotator_score",
    value_name="funny"
)
funny_long["annotator_id"] = [annotators[int(x.split("_")[1])-1] for x in funny_long["annotator_score"]]

political_long = master.melt(
    id_vars=["item_id","rag","Score_Funniness","Score_Political"],
    value_vars=[f"Political_{i}" for i in range(1,7)],
    var_name="annotator_score",
    value_name="political"
)
political_long["annotator_id"] = [annotators[int(x.split("_")[1])-1] for x in political_long["annotator_score"]]

human_df = pd.merge(
    funny_long[["item_id","rag","annotator_id","funny"]],
    political_long[["item_id","annotator_id","political"]],
    on=["item_id","annotator_id"],
)
human_df["annotator_group"] = human_df["annotator_id"].map(group_map)
human_df["rag"] = human_df["rag"].map(lambda x: 1 if str(x).upper() == "RAG" else 0)

with open("data/Definitions_Generation_Results_ID_merged.json") as file:
    data = json.load(file)

rows = []
for word in data:
    for d in word["definitions"]:
        item_id = d["definition_id"]
        for model, result in d["llm_judge"].items():
            rows.append({
                "item_id": item_id,
                "annotator_id": model,
                "annotator_group": "-",  # optional
                "rag": 1 if d["type"] == "RAG" else 0,
                "funny": result["funny"],
                "political": result["political"]
            })

llmasajudge_df = pd.DataFrame(rows)

final_df = pd.concat([human_df, llmasajudge_df], ignore_index=True)

print(final_df.head(1000))

final_df.to_csv("data/annotations_and_llmasajudge.csv", index=False)



