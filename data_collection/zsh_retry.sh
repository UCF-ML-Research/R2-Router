#!/bin/bash

source .venv/bin/activate

GPU_COUNTS=4

CSV_PATHS=(
  ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/8000_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/10_judge.csv
  ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/20_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/30_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/40_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/50_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/80_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/100_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/150_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/200_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/300_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/500_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/800_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/1200_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/2000_judge.csv
  # ./results/Qwen/Qwen3-30B-A3B-Instruct-2507/4000_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/0_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/10_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/20_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/30_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/40_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/50_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/80_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/100_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/150_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/200_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/300_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/500_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/800_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/1200_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/2000_judge.csv
  # ./results/deepseek-ai/DeepSeek-V3.1/4000_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/0_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/10_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/20_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/30_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/40_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/50_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/80_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/100_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/150_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/200_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/300_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/500_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/800_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/1200_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/2000_judge.csv
  # ./results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/4000_judge.csv
)

python retry.py ${CSV_PATHS[@]} --gpu_counts $GPU_COUNTS > retry.log 2>&1

bash zsh_combine.sh

bash zsh_dataset_1.sh