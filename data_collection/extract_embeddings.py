#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from vllm import LLM


def check_consistency(file_list):
    """检查多个文件的 prompts_id, key, original_prompt 是否逐行一致"""
    base_df = None
    for i, fp in enumerate(file_list):
        df = pd.read_csv(fp, usecols=["prompts_id", "key", "original_prompt"])
        if i == 0:
            base_df = df
        else:
            if not df.equals(base_df):
                raise ValueError(f"❌ 文件 {fp} 的 (prompts_id, key, original_prompt) 顺序或内容不一致！")
    print(f"✅ 所有 {len(file_list)} 个文件的 (prompts_id, key, original_prompt) 完全一致")
    return base_df


def extract_all_embeddings(csv_path, output_path, encoder_name, max_length, batch_size, tensor_parallel_size):
    # 读取 CSV
    df = pd.read_csv(csv_path)
    if "original_prompt" not in df.columns:
        raise ValueError("CSV must contain column 'original_prompt'")
    prompts = df["original_prompt"].tolist()
    num_samples = len(prompts)

    # 初始化 vLLM 模型（支持多卡）
    print(f"Loading model {encoder_name} with TP={tensor_parallel_size}, max_length={max_length}")
    llm = LLM(
        model=encoder_name,
        task="embed",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_length,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_log_stats=True,
    )

    all_embeddings = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Extracting embeddings"):
        batch_prompts = prompts[i: i + batch_size]
        outputs = llm.embed(batch_prompts, use_tqdm=False)
        for out in outputs:
            all_embeddings.append(out.outputs.embedding)

    embeddings = np.array(all_embeddings)

    data = {
        "embeddings": embeddings,
        "prompts": prompts,
        "query_ids": (
            df["query_id"].tolist() if "query_id" in df.columns else list(range(num_samples))
        ),
        "metadata": {
            "encoder_name": encoder_name,
            "max_length": max_length,
            "embedding_dim": int(embeddings.shape[1]),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "tensor_parallel_size": tensor_parallel_size,
            "framework": "vLLM",
            "original_csv": csv_path,
        },
    }

    # ✅ 随机 sanity check
    print("\nSanity check (recompute & compare):")
    for idx in random.sample(range(num_samples), min(3, num_samples)):
        prompt = prompts[idx]
        (output,) = llm.embed(prompt)
        emb_new = np.array(output.outputs.embedding)
        emb_saved = embeddings[idx]
        diff = np.linalg.norm(emb_saved - emb_new)
        print(f"Index {idx}: prompt={prompt!r}")
        print(f"Saved[:5]     = {emb_saved[:5]}")
        print(f"Recomputed[:5]= {emb_new[:5]}")
        print(f"Diff (L2 norm)= {diff:.6e}\n")

    # 保存
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved {num_samples} embeddings to {output_path}, shape={embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings with vLLM (multi-GPU)")
    parser.add_argument("--csv_files", nargs="+", required=True, help="多个 CSV 文件路径")
    parser.add_argument("--output_path", type=str, default="./prompt_embeddings.pkl")
    parser.add_argument("--encoder_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    args = parser.parse_args()

    # Step 1: 一致性检查
    check_consistency(args.csv_files)

    # Step 2: 用第一个文件提取 embeddings
    extract_all_embeddings(
        csv_path=args.csv_files[0],
        output_path=args.output_path,
        encoder_name=args.encoder_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
