#!/bin/bash
# CoRE Router - Cloudflare Named Tunnel Deployment
# Provides permanent, custom domain

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CoRE Router - Cloudflare Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Add cloudflared to PATH
export PATH="$HOME/.local/bin:$PATH"

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo -e "${RED}❌ cloudflared not found${NC}"
    echo "Please run setup first (see README_DEPLOY.md)"
    exit 1
fi

# Check if tunnel is configured
if [ ! -f "$HOME/.cloudflared/config.yml" ]; then
    echo -e "${RED}❌ Cloudflare tunnel not configured${NC}"
    echo ""
    echo "Please complete these steps first:"
    echo ""
    echo "1. Login to Cloudflare:"
    echo "   cloudflared tunnel login"
    echo ""
    echo "2. Create tunnel:"
    echo "   cloudflared tunnel create core-router"
    echo ""
    echo "3. Route DNS (option A - free subdomain):"
    echo "   cloudflared tunnel route dns core-router your-name.trycloudflare.com"
    echo ""
    echo "   OR (option B - your own domain):"
    echo "   cloudflared tunnel route dns core-router router.yourdomain.com"
    echo ""
    echo "4. Edit config file:"
    echo "   nano ~/.cloudflared/config.yml"
    echo "   (Replace <TUNNEL-ID> with your tunnel ID)"
    echo ""
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo "Please activate your virtual environment first"
    exit 1
fi

# Activate virtual environment
source ../.venv/bin/activate

# Create log directory
mkdir -p logs

# Stop any existing processes
echo -e "${YELLOW}Checking for existing processes...${NC}"
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "cloudflared.*tunnel.*run" 2>/dev/null || true
lsof -ti:7860 | xargs kill -9 2>/dev/null || true
sleep 2

# Start the Gradio app in background
echo -e "${GREEN}🚀 Starting Gradio app on localhost:7860...${NC}"
nohup python app.py > logs/app.log 2>&1 &
APP_PID=$!
echo "   App PID: $APP_PID"
echo "$APP_PID" > logs/app.pid

# Wait for app to start
echo -e "${YELLOW}⏳ Waiting for app to start...${NC}"
sleep 5

# Check if app is running
if ! curl -s http://127.0.0.1:7860 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  App may not be ready yet, checking logs...${NC}"
    tail -n 20 logs/app.log
    echo ""
fi

# Start Cloudflare tunnel
echo -e "${GREEN}🌐 Starting Cloudflare tunnel...${NC}"
nohup cloudflared tunnel run > logs/tunnel.log 2>&1 &
TUNNEL_PID=$!
echo "   Tunnel PID: $TUNNEL_PID"
echo "$TUNNEL_PID" > logs/tunnel.pid

# Wait for tunnel to start
echo -e "${YELLOW}⏳ Waiting for tunnel to connect...${NC}"
sleep 5

# Check tunnel status
TUNNEL_NAME=$(grep "tunnel:" ~/.cloudflared/config.yml | awk '{print $2}')
if [ -z "$TUNNEL_NAME" ]; then
    echo -e "${YELLOW}⚠️  Could not detect tunnel name from config${NC}"
else
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}🌐 Your permanent URL:${NC}"
    echo -e "${GREEN}   Check your Cloudflare dashboard or DNS settings${NC}"
    echo -e "   (The domain you configured in step 3)"
    echo ""
    echo -e "${BLUE}📊 Local URL:${NC}"
    echo -e "   http://localhost:7860"
    echo ""
    echo -e "${BLUE}📝 Logs:${NC}"
    echo -e "   App:    tail -f $SCRIPT_DIR/logs/app.log"
    echo -e "   Tunnel: tail -f $SCRIPT_DIR/logs/tunnel.log"
    echo ""
    echo -e "${BLUE}🔍 Check tunnel status:${NC}"
    echo -e "   cloudflared tunnel info $TUNNEL_NAME"
    echo ""
    echo -e "${BLUE}🛑 To stop:${NC}"
    echo -e "   ./stop.sh"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo ""
fi

# Show tunnel log
echo -e "${YELLOW}📺 Recent tunnel logs:${NC}"
echo ""
sleep 2
tail -n 30 logs/tunnel.log || echo "No tunnel logs yet"

echo ""
echo -e "${BLUE}Press Ctrl+C to stop viewing logs (services will continue)${NC}"
echo ""

# Follow app logs
tail -f logs/app.log
