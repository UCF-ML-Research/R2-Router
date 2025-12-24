# CoRE Router - 永久域名配置指南

## 🎯 目标
配置一个永久固定的自定义域名，例如：`router.yourdomain.com`

**总费用**: ~$10-15/年（仅域名费用）
**总时间**: ~30分钟
**使用本地GPU**: ✅ 是

---

## 📝 步骤 1：购买域名（5-10分钟）

### 推荐域名注册商

**A. Namecheap（推荐 - 价格便宜）**
- 网址: https://www.namecheap.com
- 价格: $8-15/年（.com域名）
- 优点: 便宜，界面友好，支持支付宝

**B. GoDaddy**
- 网址: https://www.godaddy.com
- 价格: $12-20/年
- 优点: 知名度高，中文支持

**C. Cloudflare Registrar**
- 网址: https://www.cloudflare.com/products/registrar/
- 价格: 成本价（$8-10/年）
- 优点: 无加价，直接在Cloudflare
- 缺点: 需要先添加域名到Cloudflare

### 购买步骤（以Namecheap为例）

1. **访问 Namecheap**: https://www.namecheap.com

2. **搜索域名**: 输入你想要的名字
   - 例如：`myrouter`（会显示 `myrouter.com`）
   - 建议选择短小易记的

3. **选择后缀**（推荐程度）:
   - `.com` - 最流行，$8-12/年 ⭐⭐⭐⭐⭐
   - `.net` - 次流行，$10-15/年 ⭐⭐⭐⭐
   - `.xyz` - 便宜，$1-3/年 ⭐⭐⭐
   - `.me` - 个性化，$10-20/年 ⭐⭐⭐

4. **加入购物车并结账**
   - ⚠️ **取消勾选**附加服务（WhoisGuard, SSL等都不需要）
   - Cloudflare会免费提供这些

5. **完成支付**（支持支付宝/信用卡）

6. **记下你的域名**: _________________ （填在这里）

---

## 📝 步骤 2：添加域名到 Cloudflare（5分钟）

### 2.1 登录 Cloudflare

1. 访问: https://dash.cloudflare.com
2. 使用你的账号登录（刚才已经登录了）

### 2.2 添加网站

1. 点击 **"添加站点"** 或 **"Add a Site"**
2. 输入你购买的域名（例如：`myrouter.com`）
3. 点击 **"添加站点"**

### 2.3 选择计划

1. 选择 **"Free"**（免费计划）
2. 点击 **"继续"**

### 2.4 查看 DNS 记录

1. Cloudflare会自动扫描现有DNS记录
2. 直接点击 **"继续"**（默认即可）

### 2.5 更改域名服务器（重要！）

Cloudflare会显示两个名称服务器，例如：
```
april.ns.cloudflare.com
ben.ns.cloudflare.com
```

**在 Namecheap 中更改**:

1. 登录 Namecheap: https://ap.www.namecheap.com/domains/list/
2. 找到你的域名，点击 **"Manage"**
3. 找到 **"Nameservers"** 部分
4. 选择 **"Custom DNS"**
5. 输入 Cloudflare 提供的两个名称服务器:
   ```
   april.ns.cloudflare.com
   ben.ns.cloudflare.com
   ```
6. 点击 **"✓ Save"**

⏰ **等待生效**: 5分钟 - 24小时（通常10分钟内完成）

### 2.6 验证配置

返回 Cloudflare，点击 **"完成，检查名称服务器"**

---

## 📝 步骤 3：创建 Cloudflare Tunnel（5分钟）

现在在**服务器终端**执行以下命令：

### 3.1 登录 Cloudflare（如果还没登录）

```bash
export PATH="$HOME/.local/bin:$PATH"
cloudflared tunnel login
```

会打开浏览器，选择你刚才添加的域名，点击授权。

### 3.2 创建隧道

```bash
cloudflared tunnel create core-router
```

**输出示例**:
```
Tunnel credentials written to /home/jiaq/.cloudflared/abc-123-xyz.json
Created tunnel core-router with id abc-123-xyz
```

📝 **记下隧道ID**: _________________ （例如：abc-123-xyz）

### 3.3 配置 DNS 路由

将隧道连接到你的域名（**替换为你的域名**）:

```bash
cloudflared tunnel route dns core-router router.YOURDOMAIN.com
```

例如，如果你的域名是 `myrouter.com`，运行：
```bash
cloudflared tunnel route dns core-router router.myrouter.com
```

**输出**:
```
Created DNS route for core-router over router.YOURDOMAIN.com
```

### 3.4 编辑配置文件

```bash
nano ~/.cloudflared/config.yml
```

**替换内容为**（将 `<TUNNEL-ID>` 和 `<YOURDOMAIN>` 替换成你的实际值）:

```yaml
url: http://localhost:7860
tunnel: <TUNNEL-ID>
credentials-file: /home/jiaq/.cloudflared/<TUNNEL-ID>.json
```

例如：
```yaml
url: http://localhost:7860
tunnel: abc-123-xyz
credentials-file: /home/jiaq/.cloudflared/abc-123-xyz.json
```

保存并退出（Ctrl+X, Y, Enter）

---

## 📝 步骤 4：启动服务（1分钟）

```bash
cd /home/jiaq/Research/Code/router/demo
./start_cloudflare.sh
```

等待约30秒，你会看到：

```
========================================
✅ Deployment successful!
========================================

🌐 Your permanent URL:
   https://router.yourdomain.com

📊 Local URL:
   http://localhost:7860
```

---

## 🎉 完成！

现在你的 CoRE Router 已经部署在：

**永久公网地址**: `https://router.yourdomain.com`

- ✅ **永久固定**：重启不会变
- ✅ **自定义域名**：专业的域名
- ✅ **免费SSL**：Cloudflare自动提供HTTPS
- ✅ **使用本地GPU**：完全免费
- ✅ **全球访问**：CDN加速

---

## 🔧 管理命令

### 启动服务
```bash
cd /home/jiaq/Research/Code/router/demo
./start_cloudflare.sh
```

### 停止服务
```bash
./stop.sh
```

### 检查隧道状态
```bash
cloudflared tunnel info core-router
```

### 查看隧道列表
```bash
cloudflared tunnel list
```

### 查看日志
```bash
# App 日志
tail -f logs/app.log

# Tunnel 日志
tail -f logs/tunnel.log
```

---

## 🔄 开机自启（可选）

如果你希望服务器重启后自动启动：

### 创建 systemd 服务

```bash
sudo nano /etc/systemd/system/core-router.service
```

**添加内容**:

```ini
[Unit]
Description=CoRE Router with Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=jiaq
WorkingDirectory=/home/jiaq/Research/Code/router/demo
Environment="PATH=/home/jiaq/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/jiaq/Research/Code/router/demo/start_cloudflare.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**启用服务**:

```bash
sudo systemctl daemon-reload
sudo systemctl enable core-router
sudo systemctl start core-router
```

**管理服务**:

```bash
# 查看状态
sudo systemctl status core-router

# 重启
sudo systemctl restart core-router

# 停止
sudo systemctl stop core-router

# 查看日志
sudo journalctl -u core-router -f
```

---

## 🐛 故障排查

### 问题 1: DNS 未生效

**症状**: 访问域名显示"无法访问此网站"

**解决**:
```bash
# 检查 DNS 是否生效
nslookup router.yourdomain.com

# 如果显示 NXDOMAIN，等待DNS传播（最多24小时）
```

### 问题 2: 隧道连接失败

**症状**: 日志显示 "failed to connect"

**解决**:
```bash
# 检查配置文件
cat ~/.cloudflared/config.yml

# 验证隧道ID是否正确
cloudflared tunnel list

# 重启隧道
./stop.sh && ./start_cloudflare.sh
```

### 问题 3: 502 Bad Gateway

**症状**: 访问域名显示 502 错误

**解决**:
```bash
# 检查应用是否在运行
curl http://localhost:7860

# 查看应用日志
tail -n 50 logs/app.log

# 重启应用
./stop.sh && ./start_cloudflare.sh
```

---

## 💰 费用总结

**一次性费用**: $0
**年度费用**: $10-15（仅域名续费）
**月度费用**: $0

**对比其他方案**:
- Hugging Face Spaces (GPU): ~$432/年
- ngrok 付费版: ~$96/年
- **你的方案**: ~$12/年 ⭐

---

## 📞 需要帮助？

如果遇到问题：

1. 查看日志：`tail -f logs/app.log` 和 `tail -f logs/tunnel.log`
2. 检查 GPU：`nvidia-smi`
3. 验证网络：`ping 1.1.1.1`
4. 检查隧道：`cloudflared tunnel info core-router`

---

**祝配置顺利！🎉**
