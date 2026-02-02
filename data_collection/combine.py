#!/usr/bin/env python3
import argparse, os, re
import pandas as pd
from pathlib import Path
from functools import reduce
import numpy as np

def extract_token(file_path: str) -> str:
    """从文件名中提取 token 数，比如 20_judge.csv -> 20"""
    base = os.path.basename(file_path)
    m = re.match(r'^(\d+)_judge(?:_retry)?\.csv$', base)
    return m.group(1) if m else Path(base).stem

def main():
    ap = argparse.ArgumentParser(description="合并一个文件夹下的 judge.csv 并生成宽表 + 最大值列")
    ap.add_argument("--input_dir", required=True, help="输入文件夹路径")
    ap.add_argument("--output_name", required=True, help="输出文件名")
    args = ap.parse_args()

    # === Step 0: 找到所有符合规则的文件 ===
    folder = Path(args.input_dir)
    file_names = sorted([str(p) for p in folder.glob("*_judge*.csv")
                         if re.match(r'^\d+_judge(?:_retry)?\.csv$', p.name)],
                        key=lambda x: int(extract_token(x)))

    if not file_names:
        raise FileNotFoundError(f"❌ 没有找到匹配的 *_judge.csv 文件，在目录 {args.input_dir}")

    print("🔍 找到以下文件，将进行合并：")
    for fn in file_names:
        print("   -", fn)

    dfs = []
    base_keys = None

    # === Step 1: 读取并检查一致性 ===
    for i, fp in enumerate(file_names):
        token = extract_token(fp)
        df = pd.read_csv(fp, usecols=["prompts_id", "key", "original_prompt",
                                      "actual_token_count", "correctness_score"])
        df = df.rename(columns={
            "actual_token_count": f"{token}_count",
            "correctness_score": f"{token}_score"
        })

        # 打印每个表的长度
        print(f"📊 文件 {os.path.basename(fp)} (token={token}): {len(df)} 行")

        # 检查三列键是否一致
        current_keys = df[["prompts_id", "key", "original_prompt"]]
        if i == 0:
            base_keys = current_keys.copy()
        else:
            if not current_keys.equals(base_keys):
                raise ValueError(f"❌ 文件 {fp} 的 (prompts_id, key, original_prompt) 顺序或内容不一致！")

        dfs.append(df)

    # === Step 2: 基于主键合并 ===
    wide = reduce(lambda left, right: pd.merge(
        left, right, on=["prompts_id", "key", "original_prompt"], how="inner"), dfs)

    # === Step 3: 保持顺序 ===
    wide = wide.set_index(["prompts_id", "key", "original_prompt"])
    wide = wide.loc[pd.MultiIndex.from_frame(base_keys)].reset_index()

    # === Step 4: 计算最大值列 ===
    tokens = [extract_token(fp) for fp in file_names]
    tokens = sorted(tokens, key=lambda x: int(x))  # 确保从小到大排序

    score_matrix = wide[[f"{tok}_score" for tok in tokens]].values
    count_matrix = wide[[f"{tok}_count" for tok in tokens]].values

    for j, t in enumerate(tokens):
        # 取 <= t 的子矩阵
        scores_sub = score_matrix[:, : j+1]
        counts_sub = count_matrix[:, : j+1]

        # 每行最大值及其索引
        idx = np.nanargmax(scores_sub, axis=1)
        wide[f"{t}_score_most"] = scores_sub[np.arange(len(idx)), idx]
        wide[f"{t}_count_most"] = counts_sub[np.arange(len(idx)), idx]

    wide.rename(columns={
        "8000_score": "unlimited_score",
        "8000_count": "unlimited_count"
    }, inplace=True)

    # === 保存结果 ===
    output_file = folder / f"{args.output_name}"
    wide.to_csv(output_file, index=False)
    print(f"\n✅ 已保存到 {output_file}")
    print(f"✅ 所有文件 (prompts_id, key, original_prompt) 完全一致，顺序与第一个文件保持一致")

if __name__ == "__main__":
    main()
