#!/bin/bash
# Quick start script for CoRE Router Demo

echo "=================================="
echo "CoRE Router Demo - Quick Start"
echo "=================================="
echo ""

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  WARNING: OPENROUTER_API_KEY not set!"
    echo ""
    echo "To use real LLMs, set your API key:"
    echo "  export OPENROUTER_API_KEY='your-key-here'"
    echo ""
    echo "Or run in MOCK MODE (no API calls, fake responses)"
    read -p "Run in mock mode? (y/n): " use_mock

    if [ "$use_mock" = "y" ] || [ "$use_mock" = "Y" ]; then
        echo ""
        echo "✓ Starting in MOCK MODE..."
        echo ""
        # Temporarily enable mock mode
        python3 -c "
import config
config.ENABLE_MOCK_MODE = True
" 2>/dev/null || true
    else
        echo ""
        echo "Please set OPENROUTER_API_KEY and try again."
        exit 1
    fi
fi

# Check if running in demo directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found"
    echo "Please run this script from the demo/ directory"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Gradio not installed"
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        echo "Please run: pip install -r requirements.txt"
        exit 1
    fi
else
    echo "✓ Dependencies OK"
fi

echo ""
echo "=================================="
echo "Starting CoRE Router Demo..."
echo "=================================="
echo ""

# Launch the demo
python3 app.py

# If demo exits
echo ""
echo "=================================="
echo "Demo stopped."
echo "=================================="
