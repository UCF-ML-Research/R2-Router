module load conda
conda activate pytorch

# rm -rf ~/.cache/torch/inductor
# rm -rf /home/ji757406.ucf/.cache/vllm/torch_compile_cache

CUDA_VISIBLE_DEVICES=0,3 python extract_embeddings.py \
  --csv_files data/Qwen3-0.6B.csv data/Qwen3-Next-80B-A3B-Instruct.csv data/Llama-3.2-3B-Instruct.csv\
  --output_path data/prompt_embeddings.pkl \
  --encoder_name Qwen/Qwen3-0.6B \
  --tensor_parallel_size 2