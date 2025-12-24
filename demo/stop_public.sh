#!/bin/bash
# Stop CoRE Router Public Deployment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping CoRE Router services...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Stop processes by PID if available
if [ -f logs/app.pid ]; then
    APP_PID=$(cat logs/app.pid)
    if ps -p $APP_PID > /dev/null 2>&1; then
        kill $APP_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Stopped app (PID: $APP_PID)${NC}"
    fi
    rm logs/app.pid
fi

if [ -f logs/tunnel.pid ]; then
    TUNNEL_PID=$(cat logs/tunnel.pid)
    if ps -p $TUNNEL_PID > /dev/null 2>&1; then
        kill $TUNNEL_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Stopped tunnel (PID: $TUNNEL_PID)${NC}"
    fi
    rm logs/tunnel.pid
fi

# Fallback: kill by name
pkill -f "python.*app.py" 2>/dev/null && echo -e "${GREEN}✅ Killed app process${NC}" || true
pkill -f cloudflared 2>/dev/null && echo -e "${GREEN}✅ Killed cloudflared process${NC}" || true

# Kill processes on port 7860
lsof -ti:7860 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Freed port 7860${NC}" || true

sleep 1
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All services stopped${NC}"
echo -e "${GREEN}========================================${NC}"
