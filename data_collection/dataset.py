import argparse
import pandas as pd
import csv
from datasets import load_dataset
from tqdm import tqdm
import os
import random
import time
from vllm import LLM, SamplingParams
import gc, torch
import sys
from prompt_template import generate_prompts_with_token_limits
from transformers import AutoTokenizer, AutoConfig

# from mistral_common.protocol.instruct.request import ChatCompletionRequest
# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from huggingface_hub import hf_hub_download
# from transformers import Mistral3ForConditionalGeneration

import pandas as pd
import csv
import traceback

def debug_csv_write(df, output_file, debug_dir="./debug_logs"):
    """
    朴素 df.to_csv() 写出，如果失败则自动定位出错行，
    并将完整的行内容保存到文本文件中供人工检查。
    """
    import os
    os.makedirs(debug_dir, exist_ok=True)

    try:
        # 尝试一次性写出
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Successfully saved {output_file}")
    except Exception as e:
        print(f"❌ Error while saving {output_file}: {e}")
        print("🔍 Locating problematic row...")

        bad_row_path = os.path.join(debug_dir, os.path.basename(output_file) + ".bad_row.txt")

        # 一行一行写出以定位出错行
        for i in range(len(df)):
            try:
                df.iloc[[i]].to_csv(
                    output_file + ".debug.csv",
                    mode='a', header=(i == 0),
                    index=False,
                    encoding='utf-8'
                )
            except Exception as row_e:
                with open(bad_row_path, "w", encoding="utf-8") as f:
                    f.write(f"🚨 Error at row {i}\n")
                    f.write(f"❗ Exception: {row_e}\n\n")
                    f.write("=========== ROW CONTENT ===========\n")
                    # 不让 pandas 省略任何字符
                    pd.set_option("display.max_colwidth", None)
                    pd.set_option("display.max_columns", None)
                    pd.set_option("display.width", None)
                    # 写出所有列及值
                    for col, val in df.iloc[i].items():
                        f.write(f"{col}: {repr(val)}\n")
                    f.write("===================================\n")
                print(f"🚨 Problematic row saved to: {bad_row_path}")
                traceback.print_exc()
                break

        print("⚠️ Debugging finished.")


# -----------------
# 配置
# -----------------
cache_dir = "/blue/sgao1/ji757406.ucf/hf_cache/"
os.makedirs(cache_dir, exist_ok=True)

# 全局变量
llm = None
model_name = None


def load_model(tensor_parallel_size=4):
    """延迟加载 vLLM 模型，自动适配模型最大上下文长度，且上限为 32768"""
    global llm, model_name
    if llm is None:
        print(f"🚀 Loading model: {model_name}")
        print(f"🧩 Using {tensor_parallel_size} GPUs for tensor parallelism")

        # ---- 读取模型配置 ----
        cfg = AutoConfig.from_pretrained(model_name)
        derived_len = getattr(cfg, "max_position_embeddings", None)
        if derived_len is None:
            derived_len = 4096

        # ---- 自动选择合适长度 ----
        if derived_len >= 32768:
            max_len = 32768
        else:
            max_len = derived_len

        print(f"🧠 Model reported max context = {derived_len}")
        print(f"✅ Using effective max_model_len = {max_len}")

        # ---- 加载 vLLM 模型 ----
        llm = LLM(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            download_dir=cache_dir,
            max_model_len=max_len,
            disable_log_stats=True,
            trust_remote_code=True,
        )
        print("✨ Model loaded successfully\n")
    return llm


def load_data():
    sprout = load_dataset("CARROT-LLM-Routing/SPROUT", cache_dir=cache_dir)
    data = {
        'key': sprout['train']['key'],
        'Q_train': sprout['train']['prompt'],
        'A_train': sprout['train']['golden_answer']
    }
    print(f"Dataset loaded successfully. Train samples: {len(data['Q_train'])}")
    return data


def batch_inference(
    prompts,
    batch_size=4,
    max_tokens_true=150,
    memory_clean_interval=100,
    tensor_parallel_size=4,
):
    global model_name

    llm = load_model(tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not hasattr(llm.llm_engine, "model_config") or not hasattr(llm.llm_engine.model_config, "max_model_len"):
        raise RuntimeError("❌ vLLM 模型配置缺少 max_model_len，请检查模型加载。")

    max_model_len = llm.llm_engine.model_config.max_model_len
    print(f"🧠 Detected model context window: {max_model_len}")
    print(f"⚙️  Ensuring each prompt + output ≤ {max_model_len} tokens")

    safe_prompts = []
    truncated_count = 0
    max_prompt_len = max_model_len if max_tokens_true is None else max_model_len - max_tokens_true

    for p in prompts:
        tokens = tokenizer.encode(p)
        if len(tokens) > max_prompt_len:
            truncated_count += 1
            tokens = tokens[-max_prompt_len:]
        safe_prompts.append(tokenizer.decode(tokens, skip_special_tokens=True))

    if truncated_count > 0:
        print(f"⚠️  {truncated_count} prompts were truncated to fit within {max_model_len} tokens.")

    results = []
    token_counts = []
    skipped_batches = 0

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens_true if max_tokens_true is not None else None,
    )

    for i in tqdm(range(0, len(safe_prompts), batch_size), desc="Processing batches"):
        batch_prompts = safe_prompts[i:i + batch_size]
        try:
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for output in outputs:
                text = output.outputs[0].text.strip()
                results.append(text)
                token_counts.append(len(output.outputs[0].token_ids))
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ 显存不足，跳过 batch {i // batch_size + 1}")
                for _ in range(len(batch_prompts)):
                    results.append("[SKIPPED: GPU memory insufficient]")
                    token_counts.append(0)
                skipped_batches += 1
                time.sleep(1)
                continue
            else:
                print(f"\n❌ 处理 batch {i // batch_size + 1} 时发生未知错误: {str(e)}")
                for _ in range(len(batch_prompts)):
                    results.append(f"[ERROR: {str(e)}]")
                    token_counts.append(0)
                continue

        if (i // batch_size + 1) % memory_clean_interval == 0:
            gc.collect()
            print(f"  💾 Memory cleaned at batch {i // batch_size + 1}")

    if truncated_count > 0:
        print(f"\n✅ 自动截断保护启用：{truncated_count} 条 prompt 被安全截断。")
    if skipped_batches > 0:
        print(f"\n📊 处理完成！跳过了 {skipped_batches} 个 batch（显存不足）")

    return results, token_counts


def process_single_token_limit(original_prompts, golden_answers, key, max_tokens_prompt, max_tokens_true, batch_size, output_file, tensor_parallel_size=4):
    print(f"Processing with max_tokens_prompt={max_tokens_prompt}, max_tokens_true={max_tokens_true}")
    limited_prompts = generate_prompts_with_token_limits(original_prompts, max_tokens_prompt)
    print("Starting batch inference...")

    generated_answers, actual_token_counts = batch_inference(
        limited_prompts,
        batch_size=batch_size,
        max_tokens_true=max_tokens_true,
        tensor_parallel_size=tensor_parallel_size
    )

    results_data = []
    for i, (key_val, original, templated, golden, generated, token_count) in enumerate(
        zip(key, original_prompts, limited_prompts, golden_answers, generated_answers, actual_token_counts)
    ):
        results_data.append({
            'prompts_id': i + 1,
            'key': key_val,
            'original_prompt': original,
            'templated_prompt': templated,
            'golden_answer': golden,
            'response': generated,
            'actual_token_count': token_count
        })

    df = pd.DataFrame(results_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = df.replace({r'[\x00-\x1F\x7F]': ''}, regex=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    # debug_csv_write(df, output_file)
    print(f"✅ Results saved to {output_file}")

    del results_data, df
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='CARROT Dataset Processing with Custom Token Limits (vLLM)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file', type=str, default='results.csv')
    parser.add_argument('--token_limits', type=str, nargs='+', help='List of token limits to process (e.g., 10 20 30)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for multiple files')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Number of GPUs for tensor parallelism')

    args = parser.parse_args()

    global model_name
    model_name = args.model_name
    print(f"Using model: {model_name}")

    data = load_data()
    key = data['key']
    original_prompts = data['Q_train']
    golden_answers = data['A_train']

    print(f"Processing {len(original_prompts)} samples...")
    llm = load_model(args.tensor_parallel_size)
    model_max_len = llm.llm_engine.model_config.max_model_len
    print(f"🧠 Model context length = {model_max_len}")

    print(f"Processing multiple token limits: {args.token_limits}")
    for i, token_limit in enumerate(args.token_limits):
        max_tokens_prompt = int(token_limit)

        # 🚫 跳过过大 token 限制
        if max_tokens_prompt > model_max_len:
            print(f"⛔ Skipping token_limit={max_tokens_prompt}: exceeds model max context ({model_max_len})")
            continue

        # 智能逻辑：小模型自由生成
        if model_max_len <= 8010:
            if max_tokens_prompt == 0:
                max_tokens_true = 2500
                print(f"⚙️ Small-context model ({model_max_len}). Allowing free generation.")
            else:
                max_tokens_true = max_tokens_prompt + 10
        else:
            max_tokens_true = 8010 if max_tokens_prompt == 0 else max_tokens_prompt + 10

        output_file = os.path.join(args.output_dir, f"{max_tokens_prompt}.csv")

        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(args.token_limits)}: "
              f"max_tokens_prompt={max_tokens_prompt}, max_tokens_true={max_tokens_true}")
        print(f"{'='*60}")

        if os.path.exists(output_file):
            print(f"⚠️  File {output_file} already exists. Skipping...")
            continue

        try:
            process_single_token_limit(
                original_prompts, golden_answers, key,
                max_tokens_prompt, max_tokens_true,
                args.batch_size, output_file, args.tensor_parallel_size
            )
            print(f"✅ Successfully completed {max_tokens_prompt}.csv")
        except Exception as e:
            print(f"❌ Error processing {max_tokens_prompt}.csv: {str(e)}")
            print("Continuing with next token limit...")
            continue


if __name__ == "__main__":
    main()
    sys.exit(0)
