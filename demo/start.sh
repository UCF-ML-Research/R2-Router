#!/bin/bash
# R2-Router - Simple Startup Script with Gradio Share
# Generates a public URL valid for 72 hours

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  R2-Router - Starting...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found at ../.venv${NC}"
    echo "Please activate your virtual environment first"
    exit 1
fi

# Activate virtual environment
source ../.venv/bin/activate

# Stop any existing processes
echo -e "${YELLOW}Checking for existing processes on port 7860...${NC}"
pkill -f "python.*app.py" 2>/dev/null || true
lsof -ti:7860 | xargs kill -9 2>/dev/null || true
sleep 2

echo -e "${GREEN}🚀 Starting R2-Router...${NC}"
echo -e "${BLUE}📍 Local: http://localhost:7860${NC}"
echo -e "${YELLOW}⏳ Generating public URL (takes ~10 seconds)...${NC}"
echo ""

# Start app (will display public URL when ready)
python app.py
