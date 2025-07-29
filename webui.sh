#!/bin/bash
# If you could not download the model from the official site, you can use the mirror site.
# Just remove the comment of the following line .
# 如果你无法从官方网站下载模型，你可以使用镜像网站。
# 只需要移除下面一行的注释即可。

# export HF_ENDPOINT=https://hf-mirror.com

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Using virtual environment"
else
    echo "⚠️ Virtual environment not found, using system Python"
fi

export PYTHONPATH="$PWD:$PYTHONPATH"
echo "🌐 Starting MoneyPrinterTurbo WebUI..."
echo "Access at: http://localhost:8501"

# Use the virtual environment's streamlit if available
if [ -f "venv/bin/streamlit" ]; then
    ./venv/bin/streamlit run ./webui/Main.py --browser.serverAddress="0.0.0.0" --server.enableCORS=True --browser.gatherUsageStats=False
else
    streamlit run ./webui/Main.py --browser.serverAddress="0.0.0.0" --server.enableCORS=True --browser.gatherUsageStats=False
fi