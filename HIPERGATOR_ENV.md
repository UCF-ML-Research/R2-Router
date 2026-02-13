# HiPerGator Environment Guide

## Critical Rules

- **NEVER compute on login node** — all computation via `sbatch`
- Login node OK for: `ls`, `cat`, `squeue`, `sbatch`, `git`, `uv add`, quick `python -c` (< 5s)
- Login node NOT OK for: training, inference, embedding extraction, data processing

## Account & Resources

- **User**: ah872032.ucf
- **Account/QOS**: qi855292.ucf
- **Group limits**: 48 CPUs, 8 GPUs, 375GB mem (shared across all group members)

## Partitions

| Partition | Hardware | Use Case |
|-----------|----------|----------|
| `hpg-b200` | NVIDIA B200 GPU | vLLM inference (235B/80B/30B) |
| `hpg-turin` | NVIDIA L4 GPU | Lightweight GPU (routing, embeddings) |
| `hpg-default` | CPU only | sklearn training, data processing |

## Storage

| Tier | Path | Quota | Use For |
|------|------|-------|---------|
| Home | `/home/ah872032.ucf/jiaqi/router/` | 40GB | Code, configs |
| Orange | `/orange/qi855292.ucf/ah872032.ucf/` | 5TB | Data, caches, checkpoints |
| Scratch | `/scratch/local/$SLURM_JOB_ID/` | Node-local | Temporary job files |

### Layout

```
~/jiaqi/router/                    # Code repository
├── data/ → /orange/.../data/      # Symlink to Orange
├── .venv/                         # uv virtual environment
├── scripts/                       # SLURM + RouterArena scripts
├── checkpoints/                   # Model checkpoints
├── logs/                          # SLURM job logs
└── main/                          # R2-Router core code

/orange/qi855292.ucf/ah872032.ucf/
├── data/                          # 9 LLM CSVs + prompt_embeddings.pkl (1024-dim)
├── router/                        # Raw training data (9 CSVs from data collection)
└── cache/
    ├── huggingface/               # HF model cache
    ├── vllm/                      # vLLM compiled kernels
    ├── flashinfer/                # FlashInfer JIT cache
    └── torch/                     # PyTorch extensions
```

## Python Environment (uv)

```bash
# 安装/同步依赖
uv sync

# 添加新包
uv add package_name

# sbatch 中使用
export PATH="/home/ah872032.ucf/jiaqi/router/.venv/bin:$PATH"
.venv/bin/python scripts/my_script.py
```

不需要 `module load python` 或 `conda activate`。

## Environment Variables (sbatch)

```bash
export PYTHONUNBUFFERED=1    # 实时日志输出
export HF_HOME=/orange/qi855292.ucf/ah872032.ucf/cache/huggingface
export HF_TOKEN=hf_xxx       # HuggingFace token (仅 gated models)
export VLLM_CACHE_ROOT=/orange/qi855292.ucf/ah872032.ucf/cache/vllm
export TORCH_EXTENSIONS_DIR=$SLURM_TMPDIR/torch_extensions
export CUDA_HOME=/apps/compilers/cuda/12.8.1
```

或直接 `source scripts/env.sh`。

## Known Pitfalls

1. **CUDA version**: 用 `cuda/12.8.1`，不要 `module load cuda` (默认版本可能过新)
2. **Stale JIT caches**: GPU 任务前清 `rm -rf ${CACHE_ROOT}/flashinfer/* ${CACHE_ROOT}/vllm/torch_compile_cache`
3. **Node c1000a-s15**: GPU 有问题，加 `#SBATCH --exclude=c1000a-s15`
4. **Home quota**: 所有 cache 重定向到 Orange，不要存 Home
5. **Buffered output**: 必须 `PYTHONUNBUFFERED=1`，否则日志延迟
6. **Time limits**: 超时无警告，中间结果定期保存
7. **vLLM API changes**: v0.7 用 `task="embed"`，v0.15+ 用 `runner="pooling"`
