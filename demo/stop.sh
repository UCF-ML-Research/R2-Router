#!/bin/bash
# Stop R2-Router

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping R2-Router...${NC}"

# Stop Python processes
pkill -f "python.*app.py" 2>/dev/null && echo -e "${GREEN}✅ App stopped${NC}" || echo "No app running"

# Free port 7860
lsof -ti:7860 | xargs kill -9 2>/dev/null && echo -e "${GREEN}✅ Port 7860 freed${NC}" || true

sleep 1
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ R2-Router stopped${NC}"
echo -e "${GREEN}========================================${NC}"
