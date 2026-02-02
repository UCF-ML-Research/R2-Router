# R2-Router - 公网部署指南

本指南说明如何使用本地 GPU 部署 R2-Router 并获得公网访问。

## 🎯 部署方案

**使用 Gradio Share（推荐）**

- ✅ 使用本地 GPU（完全免费）
- ✅ 自动生成公网 URL
- ✅ 无需额外配置
- ✅ HTTPS 加密
- ⚠️ URL 有效期：**72 小时**
- ⚠️ 每次重启 URL 会变化

## 🚀 快速开始

### 1. 启动服务

```bash
cd /home/jiaq/Research/Code/router/demo
./start.sh
```

### 2. 获取公网 URL

启动后约 10 秒，终端会显示：

```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://abc123xyz.gradio.live

This share link expires in 72 hours.
```

### 3. 分享链接

- **公网 URL**: `https://abc123xyz.gradio.live` - 分享给任何人
- **本地 URL**: `http://localhost:7860` - 本地访问

### 4. 停止服务

按 `Ctrl+C` 或在新终端运行：

```bash
cd /home/jiaq/Research/Code/router/demo
./stop.sh
```

## 📁 文件说明

```
demo/
├── start.sh              # 启动脚本（推荐）
├── stop.sh               # 停止脚本
├── app.py                # 主程序
├── start_public.sh       # Cloudflare 启动脚本（网络受限时不可用）
├── stop_public.sh        # Cloudflare 停止脚本
├── README_DEPLOY.md      # 本文件
└── DEPLOYMENT.md         # Cloudflare 详细文档
```

## 🔧 常用命令

### 检查服务状态

```bash
# 查看进程
ps aux | grep "python.*app.py"

# 查看端口占用
lsof -i :7860

# 查看 GPU 使用
nvidia-smi
```

### 重启服务

```bash
./stop.sh && ./start.sh
```

### 后台运行（使用 screen）

```bash
# 创建 screen 会话
screen -S core-router

# 在 screen 中启动
./start.sh

# 断开 screen (Ctrl+A, D)
# 服务继续运行

# 重新连接
screen -r core-router
```

### 后台运行（使用 nohup）

```bash
# 后台启动
nohup ./start.sh > output.log 2>&1 &

# 查看日志
tail -f output.log

# 停止服务
./stop.sh
```

## 📊 URL 管理

### 72 小时过期提醒

Gradio Share 的公网链接会在 72 小时后失效。如需长期使用：

**选项 1**: 每 72 小时重启一次
```bash
./stop.sh && ./start.sh  # 获取新 URL
```

**选项 2**: 创建定时任务自动重启（crontab）
```bash
# 编辑 crontab
crontab -e

# 添加：每 3 天重启一次（00:00）
0 0 */3 * * cd /home/jiaq/Research/Code/router/demo && ./stop.sh && ./start.sh > restart.log 2>&1
```

**选项 3**: 使用付费服务（Hugging Face Spaces）
- 固定域名
- 无需重启
- 每月 $0.60/GPU-hour

## 🔒 安全建议

### 1. 添加密码保护

编辑 `app.py` 第 1504 行：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    show_error=True,
    auth=("your_username", "your_password")  # 添加这行
)
```

### 2. 限制 API 使用

在 OpenRouter 控制面板设置：
- 设置每日/每月预算限制
- 启用使用通知
- 定期检查账单

### 3. 监控访问日志

```bash
# Gradio 会显示访问日志
# 查看最近的请求
tail -f output.log | grep "POST\|GET"
```

## 🐛 常见问题

### 问题 1: Port 7860 already in use

```bash
# 停止占用端口的进程
./stop.sh

# 或手动杀掉
lsof -ti:7860 | xargs kill -9
```

### 问题 2: Public URL not generated

**可能原因**:
- 网络连接问题
- 防火墙阻止

**解决方案**:
```bash
# 1. 检查网络
ping 8.8.8.8

# 2. 检查 Gradio 版本
pip show gradio

# 3. 升级 Gradio
pip install --upgrade gradio

# 4. 重启服务
./stop.sh && ./start.sh
```

### 问题 3: App crashes on startup

```bash
# 查看错误信息
python app.py

# 检查 GPU
nvidia-smi

# 检查虚拟环境
source ../.venv/bin/activate
pip list | grep gradio
```

### 问题 4: Slow response

**原因**: vLLM embedding 模型加载需要时间

**正常现象**:
- 首次启动需要 1-2 分钟加载模型
- 加载完成后会显示 "✅ Initialization complete!"

**优化**:
```bash
# 使用更快的 embedding 模型
# 编辑 demo/config.py 第 76 行
EMBEDDING_MODEL = "sentence-transformers"  # CPU 模式，更快
```

## 📈 性能优化

### GPU 内存优化

如果 GPU 内存不足，编辑 `config.py`:

```python
# 减少 tensor parallel size
VLLM_TENSOR_PARALLEL_SIZE = 1  # 从 2 改为 1

# 减少最大序列长度
VLLM_MAX_MODEL_LEN = 512  # 从 1024 改为 512
```

### 加速启动

```bash
# 预热：第一次启动后保持运行
# 后续访问会很快
```

## 🌐 替代方案

如果 Gradio Share 不满足需求：

### Hugging Face Spaces（推荐长期使用）

**优点**:
- ✅ 固定域名
- ✅ 无需重启
- ✅ 自动 SSL
- ✅ 全球 CDN

**缺点**:
- ⚠️ 需要付费（GPU: ~$0.60/hour）
- ⚠️ 使用云端 GPU（不是本地）

**部署**:
```bash
pip install huggingface_hub
huggingface-cli login
cd demo
gradio deploy
```

### ngrok（固定域名）

**免费版**:
- 随机域名
- 每分钟 40 连接限制

**付费版** ($8/月):
- 固定域名
- 无连接限制

```bash
# 安装 ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/

# 配置
ngrok config add-authtoken <your-token>

# 启动
# 终端 1
python app.py  # (修改 share=False)

# 终端 2
ngrok http 7860
```

## 💡 使用技巧

1. **保存 URL**: 每次启动后保存新的公网 URL
2. **定期重启**: 在 URL 过期前重启获取新链接
3. **监控成本**: 定期检查 OpenRouter API 使用量
4. **备份配置**: 保存 `config.py` 和 checkpoints

## 📞 支持

遇到问题？

1. 查看终端输出的错误信息
2. 检查 GPU 状态: `nvidia-smi`
3. 验证网络连接: `ping 8.8.8.8`
4. 查看 Gradio 文档: https://gradio.app/docs/

---

**祝使用愉快！🎉**
