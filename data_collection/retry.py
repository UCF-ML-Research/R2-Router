#!/usr/bin/env python3
import pandas as pd
import argparse, os, time, gc, torch
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import ultimate_json_score_extractor
from prompt_template import build_prompt

cache_dir = "/orange/qi855292.ucf/ah872032.ucf/cache/huggingface/"
os.makedirs(cache_dir, exist_ok=True)
MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"

def load_model(gpu_counts=2):
    return LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        tensor_parallel_size=gpu_counts,
        download_dir=cache_dir,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
    )

def retry_failed_items(df, llm, batch_size=1, max_new_tokens=200, max_retries=500):
    # 让 correctness_score 是数值列（非数值统一转 NaN，便于判断）
    if "correctness_score" in df.columns:
        df["correctness_score"] = pd.to_numeric(df["correctness_score"], errors="coerce")

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_new_tokens)

    for attempt in range(1, max_retries + 1):
        failed_df = df[df["correctness_score"].isna()]
        if failed_df.empty:
            print(f"🎉 所有失败项已修复（共尝试 {attempt-1} 轮）")
            return df

        print("❌ 以下是未成功的 judge_raw 示例:")
        for idx, row in failed_df.head(5).iterrows():  # 只示例打印前 5 条，避免刷屏
            print(f"judge_raw: {row.get('judge_raw')}")
            print(f"correctness_score: {row.get('correctness_score')}")
            print("-" * 50)

        print(f"\n=== 第 {attempt} 轮重试: {len(failed_df)} 个失败项 ===")
        for idx, row in tqdm(failed_df.iterrows(), total=len(failed_df)):
            prompt = build_prompt(row["golden_answer"], row["response"])
            try:
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                text = outputs[0].outputs[0].text.strip()
            except Exception as e:
                text = f"[ERROR: {str(e)}]"
                time.sleep(1)

            df.at[idx, "judge_raw"] = text
            df.at[idx, "correctness_score"] = ultimate_json_score_extractor(text, "correctness_score")

        still_failed = df[df["correctness_score"].isna()]
        print(f"本轮结束: 仍有 {len(still_failed)} 个未成功")
        if still_failed.empty:
            print("✅ 全部修复完成")
            return df

    # ---- 到这里说明多轮后仍有失败 ----
    still_failed = df[df["correctness_score"].isna()]
    if not still_failed.empty:
        print(f"⚠️ 已达到最大重试次数，仍有 {len(still_failed)} 条未识别成功，correctness_score 将强制填 0。")

        # 填 0
        df.loc[still_failed.index, "correctness_score"] = 0.0

    return df


def main():
    parser = argparse.ArgumentParser(description="批量重试多个 judge.csv 文件（一次加载模型）")
    parser.add_argument("paths", nargs="+", help="输入 judge.csv 文件路径（可多个）")
    parser.add_argument("--gpu_counts", type=int, default=2, help="GPU 数量")
    parser.add_argument("--batch_size", type=int, default=2, help="批处理大小（默认逐个处理）")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    args = parser.parse_args()

    print(f"正在初始化模型 (GPU数量: {args.gpu_counts})...")
    llm = load_model(gpu_counts=args.gpu_counts)

    processed = 0
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"❌ 文件不存在: {path_str}")
            continue

        try:
            print(f"\n=== 处理文件: {path_str} ===")
            df = pd.read_csv(path)

            # 如果没有 correctness_score 列，先解析一次
            if "correctness_score" not in df.columns:
                df["correctness_score"] = df["judge_raw"].apply(
                    lambda x: ultimate_json_score_extractor(x, "correctness_score")
                )

            df = retry_failed_items(
                df, llm,
                batch_size=args.batch_size,
                max_retries=args.max_retries
            )

            df.to_csv(path_str, index=False)
            print(f"✅ 已保存: {path_str}")
            processed += 1

        except Exception as e:
            print(f"❌ 处理 {path_str} 出错: {str(e)}，跳过")
            continue

        # 内存清理
        del df
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass

    # 卸载一次模型
    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass

    print(f"\n📊 全部完成，成功处理 {processed}/{len(args.paths)} 个文件。")


if __name__ == "__main__":
    main()
