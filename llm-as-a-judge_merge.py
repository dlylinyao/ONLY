import os
import json
import argparse
from glob import glob
from copy import deepcopy

ORIGINAL_FILE = "data/Definitions_Generation_Results_ID.json"
DEFAULT_INPUT_DIR = "data/judged_models"
DEFAULT_OUTPUT_FILE = "data/Definitions_Generation_Results_ID_merged.json"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_definition_index(definitions):
    """
    Maps definition_id -> sample object
    """
    index = {}
    for word_entry in definitions:
        for sample in word_entry["definitions"]:
            definition_id = sample.get("definition_id")
            if definition_id is None:
                raise ValueError("Every definition must have a definition_id.")
            index[definition_id] = sample
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help="Directory containing judged_*.json")
    parser.add_argument("--original-file", type=str, default=ORIGINAL_FILE, help="Original unjudged dataset")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="Merged output file")
    args = parser.parse_args()

    merged = load_json(args.original_file)
    merged_index = build_definition_index(merged)

    judged_files = sorted(glob(os.path.join(args.input_dir, "judged_*.json")))

    if not judged_files:
        print(f"No judged files found in {args.input_dir}")
        return

    print(f"Found {len(judged_files)} judged files.")

    for path in judged_files:
        print(f"Merging: {path}")
        data = load_json(path)
        data_index = build_definition_index(data)

        for definition_id, sample in data_index.items():
            if definition_id not in merged_index:
                continue

            source_judges = sample.get("llm_judge", {})
            if not source_judges:
                continue

            target_sample = merged_index[definition_id]
            if "llm_judge" not in target_sample:
                target_sample["llm_judge"] = {}

            for model_name, judgment in source_judges.items():
                target_sample["llm_judge"][model_name] = deepcopy(judgment)

    save_json(merged, args.output_file)
    print(f"Saved merged file to: {args.output_file}")


if __name__ == "__main__":
    main()