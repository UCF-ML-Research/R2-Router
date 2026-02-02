#!/bin/bash

source .venv/bin/activate

model="microsoft/phi-4"

# GPU配置
tensor_parallel_size=2  # 使用的GPU数量

buckets=(
  0
  # 10
  # 20
  # 30
  # 40
  # 50
  # 80
  # 100
  # 150
  # 200
  # 300
  # 500
  # 800
  # 1200
  # 2000
  4000
)

mkdir -p ./results/${model}

echo "Starting dataset processing for model: $model"
echo "Token limits: ${buckets[@]}"
echo "Output directory: ./results/${model}"
echo "Using $tensor_parallel_size GPUs for tensor parallelism"

log_file="./results/${model}/processing_$(date +%Y%m%d_%H%M%S).log"

python -u dataset.py \
    --model_name $model \
    --batch_size 256 \
    --token_limits "${buckets[@]}" \
    --output_dir ./results/${model} \
    --tensor_parallel_size $tensor_parallel_size 2>&1 | tee "$log_file"

echo "Processing completed. Log saved to: $log_file"