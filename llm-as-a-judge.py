import os

# Set cache dirs BEFORE importing transformers/datasets
os.environ["HF_HOME"] = "/scratch/project_465001864/degibert/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")

import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ORIGINAL_FILE = "data/Definitions_Generation_Results_ID.json"
OUTPUT_DIR = "data/judged_models"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Best-effort determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_output_file(model_name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"judged_{safe_model_name(model_name)}.json")


def load_definitions_for_model(model_name: str):
    """
    Resume from the model-specific output file if it exists.
    Otherwise start from the original dataset.
    """
    output_file = get_output_file(model_name)
    if os.path.exists(output_file):
        print(f"Resuming from existing file: {output_file}")
        return load_json(output_file), output_file

    print(f"Starting from original file: {ORIGINAL_FILE}")
    return load_json(ORIGINAL_FILE), output_file


def extract_json(text: str):
    """
    Try to parse model output as JSON.
    If the model adds extra text, extract the first {...} block.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def normalize_scores(parsed):
    """
    Validate and normalize funny/political scores.
    Returns a dict if valid, else None.
    """
    if not isinstance(parsed, dict):
        return None

    if "funny" not in parsed or "political" not in parsed:
        return None

    try:
        funny = int(parsed["funny"])
        political = int(parsed["political"])
    except (ValueError, TypeError):
        return None

    if not (1 <= funny <= 5 and 1 <= political <= 5):
        return None

    return {"funny": funny, "political": political}


def tokenizer_supports_system_role(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return template is not None and "system" in template


def build_messages(word: str, text: str, model):
    instruction_prompt = """You are a strict scoring function.

Task:
Score a satirical definition on two dimensions:
- funny
- political

Use only the text provided by the user.
Do not use external knowledge.
Do not explain your answer.
Do not add any text before or after the JSON.

Scales:

funny:
1 = not funny
2 = slightly funny
3 = funny
4 = very funny
5 = extremely funny

political:
1 = not political
2 = slightly political
3 = generally political
4 = clearly political and topical
5 = strongly political and specifically relevant to Finnish political culture

Output rules:
- Output exactly one JSON object
- Use exactly these two keys: "funny", "political"
- Both values must be integers from 1 to 5
- Do not use markdown
- Do not use code fences
- Do not output anything except the JSON object

Valid output example:
{"funny": 3, "political": 4}
"""

    user_prompt = f"Word: {word}\nDefinition: {text}"

    if "gemma" in model:
        return [
            {
                "role": "user",
                "content": f"{instruction_prompt}\n\n{user_prompt}"
            }
        ]
    else:
        return [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": user_prompt},
        ]




def flatten_pending_samples(definitions, llm_name: str):
    """
    Collect all definition samples not yet judged by this model.
    Returns a list of dicts holding references back into the original structure.
    """
    pending = []

    for word_entry_idx, word_entry in enumerate(definitions):
        word = word_entry["word"]

        for def_idx, sample in enumerate(word_entry["definitions"]):
            if "llm_judge" not in sample:
                sample["llm_judge"] = {}

            if llm_name in sample["llm_judge"]:
                continue

            pending.append(
                {
                    "word_entry_idx": word_entry_idx,
                    "def_idx": def_idx,
                    "word": word,
                    "text": sample["text"],
                    "definition_id": sample.get("definition_id"),
                }
            )

    return pending


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def generate_batch(model, tokenizer, batch_messages, max_new_tokens: int):
    """
    Generate one output per chat message list in batch_messages.
    """
    chat_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    tokenized = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    results = []
    input_lengths = attention_mask.sum(dim=1).tolist()

    for i in range(outputs.shape[0]):
        generated_tokens = outputs[i][int(input_lengths[i]):]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        results.append(text)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="HF model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    model_name = args.model_name
    llm_name = model_name.split("/")[-1]

    definitions, output_file = load_definitions_for_model(model_name)

    with open(".hf_token", "r", encoding="utf-8") as f:
        hf_token = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        token=hf_token,
    )
    model.eval()

    pending = flatten_pending_samples(definitions, llm_name)

    print(f"Model: {model_name}")
    print(f"Output file: {output_file}")
    print(f"Pending definitions for {llm_name}: {len(pending)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")

    if len(pending) == 0:
        print("Nothing to do.")
        return

    parse_failures = 0
    progress = tqdm(list(batched(pending, args.batch_size)), desc="Batches")

    for batch in progress:
        batch_messages = [build_messages(item["word"], item["text"], model_name) for item in batch]

        raw_results = generate_batch(
            model=model,
            tokenizer=tokenizer,
            batch_messages=batch_messages,
            max_new_tokens=args.max_new_tokens,
        )

        for item, raw_result in zip(batch, raw_results):
            parsed = extract_json(raw_result)
            normalized = normalize_scores(parsed)

            sample = definitions[item["word_entry_idx"]]["definitions"][item["def_idx"]]

            sample["llm_judge"][llm_name] = {
                "raw_response": raw_result,
                "funny": normalized["funny"] if normalized else None,
                "political": normalized["political"] if normalized else None,
                "seed": args.seed,
            }

            if normalized is None:
                parse_failures += 1

        # Always save every batch so partial progress is not lost
        save_json(definitions, output_file)
        progress.set_postfix(parse_failures=parse_failures)

    print("Done.")
    print(f"Saved: {output_file}")
    print(f"Parse failures: {parse_failures}")


if __name__ == "__main__":
    main()