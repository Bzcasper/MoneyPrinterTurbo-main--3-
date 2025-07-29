#!/bin/bash
# Quick project activation script

cd /workspaces/bobby/Downloads/moneyp

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ MoneyPrinterTurbo environment activated"
echo "Python: $(python --version)"
echo "Location: $(which python)"

# Optional: Set useful aliases
alias mpt-start="uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"
alias mpt-test="python -m pytest"
alias mpt-lint="black . && flake8 ."

echo "💡 Available commands:"
echo "  mpt-start  - Start the development server"
echo "  mpt-test   - Run tests"
echo "  mpt-lint   - Format and lint code"
