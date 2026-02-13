import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import time
import os
import json
import re
import gc, torch
import argparse
from pathlib import Path
from prompt_template import build_prompt

# -----------------
# 配置
# -----------------
cache_dir = "/orange/qi855292.ucf/ah872032.ucf/cache/huggingface/"
os.makedirs(cache_dir, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# -----------------
# 加载 vLLM
# -----------------
def load_model(gpu_counts=2):
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        tensor_parallel_size=gpu_counts,
        download_dir=cache_dir,
        max_model_len=32768,
        # disable_log_stats=True,
        gpu_memory_utilization=0.85, 
    )
    return llm

# -----------------
# 批量推理（vLLM）
# -----------------
def batch_inference(prompts, llm, batch_size=16, max_new_tokens=200):
    results = []
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i+batch_size]
        try:
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for output in outputs:
                text = output.outputs[0].text.strip()
                print(text)
                results.append(text)
        except Exception as e:
            print(f"\n❌ 处理 batch {i//batch_size + 1} 时发生错误: {str(e)}")
            for _ in range(len(batch_prompts)):
                results.append(f"[ERROR: {str(e)}]")
            time.sleep(1)
            continue

    return results


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用vLLM对CSV文件中的答案进行评判（支持多文件，一次加载模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python judge.py data1.csv data2.csv --gpu_counts 2 --batch_size 8
    python judge.py ./results/meta-llama/Llama-3.2-3B-Instruct/20.csv ./results/meta-llama/Llama-3.2-3B-Instruct/40.csv --gpu_counts 4 --batch_size 16
        """
    )
    
    parser.add_argument(
        "paths",
        nargs='+',
        help="输入CSV文件路径（可多个）"
    )
    
    parser.add_argument(
        "--gpu_counts",
        type=int,
        default=2,
        help="GPU数量，用于tensor_parallel_size (默认: 2)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批处理大小 (默认: 8)"
    )
    
    args = parser.parse_args()
    
    print(f"正在初始化模型 (GPU数量: {args.gpu_counts})...")
    llm = load_model(gpu_counts=args.gpu_counts)

    processed = 0
    for path_str in args.paths:
        if not os.path.exists(path_str):
            print(f"❌ 错误: 输入文件不存在: {path_str}")
            continue

        try:
            print(f"\n正在加载文件: {path_str}")
            df = pd.read_csv(path_str)
            print(f"成功加载 {len(df)} 行数据")

            print("正在构建prompts...")
            prompts = [
                build_prompt(row["golden_answer"], row["response"])
                for _, row in df.iterrows()
            ]

            print(f"开始批量推理 (批大小: {args.batch_size})...")
            results = batch_inference(prompts, llm, batch_size=args.batch_size, max_new_tokens=200)

            df["judge_raw"] = results

            # 生成输出文件路径: path_judge.csv
            input_path = Path(path_str)
            output_path = input_path.parent / f"{input_path.stem}_judge.csv"

            df.to_csv(output_path, index=False)
            print(f"✅ 评判完成！输出文件: {output_path}")
            processed += 1
        except Exception as e:
            print(f"❌ 处理文件 {path_str} 时发生错误: {str(e)}，跳过继续...")
            continue

        # 主动释放中间对象，降低峰值内存
        del df, prompts, results
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass

    # 清理资源（只卸载一次模型）
    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    print(f"\n📊 全部完成。成功处理 {processed}/{len(args.paths)} 个文件。")
    return 0


if __name__ == "__main__":
    exit(main())
