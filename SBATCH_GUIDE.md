# SBATCH Guide (HiPerGator)

本项目在 HiPerGator 上通过 sbatch 提交 GPU 任务。

## 提交和监控

```bash
sbatch scripts/my_job.sbatch   # 提交
squeue -u $USER                # 查看队列 (PD=等待, R=运行, CG=收尾)
tail -f logs/job_JOBID.log     # 实时日志
scancel JOBID                  # 取消
seff JOBID                     # 资源使用 (完成后)
sacct -j JOBID --format=JobID,State,ExitCode  # 退出码
```

## 分区和 GPU

| Partition | Hardware | GPU Flag | 用途 |
|-----------|----------|----------|------|
| `hpg-b200` | NVIDIA B200 | `--gres=gpu:N` | vLLM 推理, 大模型 |
| `hpg-turin` | NVIDIA L4 | `--gres=gpu:l4:1` | 轻量 GPU (路由, embedding) |
| `hpg-default` | CPU only | (无) | sklearn 训练, 数据处理 |

## 资源限制

- **Group (qi855292.ucf)**: 48 CPUs, 8 GPUs, 375GB mem 总量
- **单 job 内存上限**: 32gb（不要超过，group 总共只有 48GB per-CPU 配额）
- 当前 inference 任务: 235B(4GPU) + 80B(2GPU) + 30B(1GPU) = 7 GPU

## 常用 SBATCH 参数

```bash
#SBATCH --job-name=myjob
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.log
```

## 资源参考

| 任务 | GPU | CPU | Mem | Time |
|------|-----|-----|-----|------|
| 路由 (route_routerarena.py) | 1× L4 | 8 | 32G | 10min |
| Embedding 提取 (30K queries) | 2× B200 | 16 | 64G | 2h |
| vLLM 推理 235B (TP=4) | 4× B200 | 8 | 32G | 5h |
| vLLM 推理 80B (TP=2) | 2× B200 | 8 | 32G | 5h |
| vLLM 推理 30B (TP=1) | 1× B200 | 8 | 32G | 5h |
| R2-Router 训练 (sklearn) | 0 | 8 | 32G | 30min |

## 环境配置

```bash
# sbatch 脚本中:
export PYTHONUNBUFFERED=1
export HF_HOME=/orange/qi855292.ucf/ah872032.ucf/cache/huggingface
export VLLM_CACHE_ROOT=/orange/qi855292.ucf/ah872032.ucf/cache/vllm
export PATH="/home/ah872032.ucf/jiaqi/router/.venv/bin:$PATH"

# GPU 任务额外:
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export TORCH_EXTENSIONS_DIR=$SLURM_TMPDIR/torch_extensions
```

项目用 **uv** 管理依赖 (不用 conda)：
```bash
uv sync              # 安装依赖
uv add package_name  # 添加新包
.venv/bin/python     # 直接用 venv
```

## 常见问题

- **PENDING (Priority)**: 正常排队，等高优先级任务完成
- **PENDING (Resources)**: 资源不够，等释放
- **QOSMaxGRESPerUser**: GPU 达到 group 上限 (8 GPU)
- **Node c1000a-s15**: GPU 有问题，用 `--exclude=c1000a-s15`
- **日志不更新**: 确保 `export PYTHONUNBUFFERED=1`
- **CUDA 版本**: 用 `cuda/12.8.1`，不要 `module load cuda` (会加载过新版本)
- **JIT 缓存过期**: GPU 任务前清理 `flashinfer/` 和 `vllm/torch_compile_cache`
