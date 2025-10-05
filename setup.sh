#!/bin/bash

echo "============================================"
echo "   SkyGuard - Setup Script"
echo "============================================"
echo

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "[2/4] Setting up environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from template"
    echo "Please edit .env file with your API keys before running the app"
else
    echo ".env file already exists"
fi

echo
echo "[3/4] Testing configuration..."
python scripts/test_paths.py

echo
echo "[4/4] Setup complete!"
echo
echo "============================================"
echo "   Next Steps:"
echo "============================================"
echo "1. Edit .env file with your API keys"
echo "2. Run: streamlit run app/main_app.py"
echo "3. Open browser to: http://localhost:8501"
echo "============================================"
echo
