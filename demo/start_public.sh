#!/bin/bash
# CoRE Router Public Deployment Startup Script
# This script starts both the Gradio app and Cloudflare tunnel

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CoRE Router Public Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Add cloudflared to PATH
export PATH="$HOME/.local/bin:$PATH"

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo -e "${YELLOW}⚠️  cloudflared not found in PATH${NC}"
    echo "Installing cloudflared..."
    mkdir -p ~/.local/bin
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O ~/.local/bin/cloudflared
    chmod +x ~/.local/bin/cloudflared
    echo -e "${GREEN}✅ cloudflared installed${NC}"
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

# Kill any existing processes on port 7860
echo -e "${YELLOW}Checking for existing processes on port 7860...${NC}"
pkill -f "python.*app.py" 2>/dev/null || true
lsof -ti:7860 | xargs kill -9 2>/dev/null || true
sleep 2

# Start the Gradio app in background
echo -e "${GREEN}🚀 Starting Gradio app on localhost:7860...${NC}"
nohup python app.py > logs/app.log 2>&1 &
APP_PID=$!
echo "   App PID: $APP_PID"

# Wait for app to start
echo -e "${YELLOW}⏳ Waiting for app to start...${NC}"
sleep 5

# Check if app is running
if ! curl -s http://127.0.0.1:7860 > /dev/null; then
    echo -e "${YELLOW}⚠️  App may not be ready yet, checking logs...${NC}"
    tail -n 20 logs/app.log
    echo ""
fi

# Start cloudflared tunnel
echo -e "${GREEN}🌐 Starting Cloudflare tunnel...${NC}"
echo -e "${YELLOW}⏳ Generating public URL (this may take 10-20 seconds)...${NC}"
echo ""

# Start cloudflared and capture output
nohup cloudflared tunnel --url http://localhost:7860 > logs/tunnel.log 2>&1 &
TUNNEL_PID=$!
echo "   Tunnel PID: $TUNNEL_PID"

# Wait for tunnel URL to appear in logs
echo -e "${YELLOW}⏳ Waiting for tunnel URL...${NC}"
for i in {1..30}; do
    if grep -q "https://" logs/tunnel.log 2>/dev/null; then
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Extract and display the public URL
PUBLIC_URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" logs/tunnel.log | head -n 1)

if [ -n "$PUBLIC_URL" ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}🌐 Public URL:${NC}"
    echo -e "${GREEN}   $PUBLIC_URL${NC}"
    echo ""
    echo -e "${BLUE}📊 Local URL:${NC}"
    echo -e "   http://localhost:7860"
    echo ""
    echo -e "${BLUE}📝 Logs:${NC}"
    echo -e "   App:    tail -f $SCRIPT_DIR/logs/app.log"
    echo -e "   Tunnel: tail -f $SCRIPT_DIR/logs/tunnel.log"
    echo ""
    echo -e "${BLUE}🛑 To stop:${NC}"
    echo -e "   kill $APP_PID $TUNNEL_PID"
    echo -e "   or run: pkill -f 'python.*app.py' && pkill -f cloudflared"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Save PIDs to file for easy cleanup
    echo "$APP_PID" > logs/app.pid
    echo "$TUNNEL_PID" > logs/tunnel.pid

    # Display real-time logs
    echo -e "${YELLOW}📺 Showing real-time logs (Ctrl+C to stop viewing, services will continue)...${NC}"
    echo ""
    sleep 2
    tail -f logs/app.log
else
    echo -e "${YELLOW}⚠️  Could not detect public URL. Check logs:${NC}"
    echo -e "   cat logs/tunnel.log"
    echo ""
    echo -e "${YELLOW}Showing last 20 lines of tunnel log:${NC}"
    tail -n 20 logs/tunnel.log
fi
