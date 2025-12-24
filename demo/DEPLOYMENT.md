# CoRE Router - Public Deployment Guide

This guide explains how to deploy CoRE Router with public access using your local GPU.

## 🎯 Architecture

- **Backend**: Runs locally on your machine (uses your GPU)
- **Public Access**: Cloudflare Tunnel (free, permanent URL)
- **No Cloud Costs**: Everything runs on your hardware

## 🚀 Quick Start

### 1. Start the Public Deployment

```bash
cd /home/jiaq/Research/Code/router/demo
./start_public.sh
```

**What happens:**
1. Starts the Gradio app on `localhost:7860`
2. Creates a Cloudflare tunnel
3. Displays a public URL like `https://abc123.trycloudflare.com`
4. Shows real-time logs

**Output example:**
```
========================================
✅ Deployment successful!
========================================

🌐 Public URL:
   https://abc-def-123.trycloudflare.com

📊 Local URL:
   http://localhost:7860

📝 Logs:
   App:    tail -f /path/to/logs/app.log
   Tunnel: tail -f /path/to/logs/tunnel.log

🛑 To stop:
   kill 12345 67890
========================================
```

### 2. Access Your App

- **Public URL**: Share with anyone (no VPN needed)
- **Local URL**: For your own testing

### 3. Stop the Services

```bash
./stop_public.sh
```

## 📁 File Structure

```
demo/
├── start_public.sh      # Start deployment
├── stop_public.sh       # Stop deployment
├── app.py              # Main Gradio app
├── logs/               # Runtime logs
│   ├── app.log         # Gradio app logs
│   ├── tunnel.log      # Cloudflare tunnel logs
│   ├── app.pid         # App process ID
│   └── tunnel.pid      # Tunnel process ID
└── DEPLOYMENT.md       # This file
```

## 🔧 Advanced Usage

### Check if Services are Running

```bash
# Check app status
ps aux | grep "python.*app.py"

# Check tunnel status
ps aux | grep cloudflared

# Check port 7860
lsof -i :7860
```

### View Logs

```bash
# Real-time app logs
tail -f logs/app.log

# Real-time tunnel logs
tail -f logs/tunnel.log

# Last 50 lines of app logs
tail -n 50 logs/app.log
```

### Restart Services

```bash
./stop_public.sh && ./start_public.sh
```

### Change Port

Edit `app.py` line 1506 and `start_public.sh` line 75:
```python
# app.py
server_port=YOUR_PORT,

# start_public.sh
cloudflared tunnel --url http://localhost:YOUR_PORT
```

## 🌐 About Cloudflare Tunnel

### Free Features
- ✅ Unlimited bandwidth
- ✅ Automatic HTTPS
- ✅ DDoS protection
- ✅ No registration required (for quick tunnels)
- ✅ Public URL works globally

### Limitations (Free Quick Tunnel)
- ⚠️ Random URL each time (e.g., `abc-123.trycloudflare.com`)
- ⚠️ URL changes when you restart
- ⚠️ Connection may occasionally reset

### Upgrade to Named Tunnel (Optional)

For a **permanent custom URL**, create a free Cloudflare account:

```bash
# 1. Login to Cloudflare (opens browser)
cloudflared tunnel login

# 2. Create named tunnel
cloudflared tunnel create core-router

# 3. Configure DNS
cloudflared tunnel route dns core-router core-router.yourdomain.com

# 4. Start tunnel
cloudflared tunnel run core-router
```

Benefits:
- ✅ Fixed URL (never changes)
- ✅ Custom domain support
- ✅ More reliable

## 🔒 Security Considerations

### Current Setup (Quick Tunnel)
- Public URL is random and hard to guess
- HTTPS encryption enabled by default
- No authentication required

### Recommendations

1. **Add Authentication** (edit `app.py`):
```python
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True,
    auth=("username", "password")  # Add this line
)
```

2. **Monitor Access** (check logs):
```bash
grep "POST\|GET" logs/app.log | tail -n 20
```

3. **Limit OpenRouter API Usage**:
   - Set budget limits in OpenRouter dashboard
   - Monitor API usage regularly

## 🐛 Troubleshooting

### Issue: Port 7860 already in use

```bash
# Find and kill process
lsof -ti:7860 | xargs kill -9

# Or use stop script
./stop_public.sh
```

### Issue: Public URL not generated

```bash
# Check tunnel logs
cat logs/tunnel.log

# Restart cloudflared
pkill cloudflared
cloudflared tunnel --url http://localhost:7860
```

### Issue: App crashes

```bash
# Check app logs
tail -n 50 logs/app.log

# Check if GPU is available
nvidia-smi

# Restart app
./stop_public.sh && ./start_public.sh
```

### Issue: Slow response times

- Check GPU usage: `nvidia-smi`
- Check network: `ping 1.1.1.1`
- View app logs: `tail -f logs/app.log`

## 📊 Monitoring

### GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### System Resources

```bash
# CPU and memory
htop

# Disk space
df -h

# Network connections
netstat -tuln | grep 7860
```

## 🎓 Tips

1. **Keep Terminal Open**: Services run in background, but logs stream in foreground
2. **Use `screen` or `tmux`**: Run in detached session for persistent deployment
3. **Monitor Costs**: OpenRouter API usage is the only cost
4. **Backup Logs**: Rotate logs periodically to save disk space

## 📞 Support

- **App Issues**: Check `logs/app.log`
- **Tunnel Issues**: Check `logs/tunnel.log`
- **Cloudflare Docs**: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- **Gradio Docs**: https://gradio.app/docs/

## 🔄 Auto-Start on Boot (Optional)

To run services automatically when your computer starts, see the systemd service setup in the main README.
