#!/bin/bash -l
sbatch llm-as-a-judge_puhti.sh Qwen/Qwen2.5-7B-Instruct
sbatch llm-as-a-judge_puhti.sh meta-llama/Llama-3.1-8B-Instruct
sbatch llm-as-a-judge_puhti.sh mistralai/Mistral-7B-Instruct-v0.3
sbatch llm-as-a-judge_puhti.sh google/gemma-2-9b-it
sbatch llm-as-a-judge_puhti.sh CohereLabs/aya-expanse-8b
sbatch llm-as-a-judge_puhti.sh utter-project/EuroLLM-9B-Instruct