#!/bin/bash -l
sbatch llm-as-a-judge.sh Qwen/Qwen2.5-7B-Instruct
sbatch llm-as-a-judge.sh meta-llama/Llama-3.1-8B-Instruct
sbatch llm-as-a-judge.sh mistralai/Mistral-7B-Instruct-v0.3
sbatch llm-as-a-judge.sh google/gemma-2-9b-it
sbatch llm-as-a-judge.sh CohereLabs/aya-expanse-8b
sbatch llm-as-a-judge.sh utter-project/EuroLLM-9B-Instruct