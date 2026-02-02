import pandas as pd

df1 = pd.read_csv("results/meta-llama/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct.csv")
df2 = pd.read_csv("results/Qwen/Qwen3-0.6B/Qwen3-0.6B.csv")

if df1[["key", "original_prompt"]].equals(df2[["key", "original_prompt"]]):
    print("✅ 两个 DataFrame 在顺序和内容上完全一致")
else:
    print("❌ 两个 DataFrame 顺序/内容不一致")