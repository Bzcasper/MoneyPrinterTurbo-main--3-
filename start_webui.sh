#!/bin/bash
echo "Starting WebUI with Streamlit..."
streamlit run /MoneyPrinterTurbo/webui/Main.py --browser.serverAddress=0.0.0.0 --server.enableCORS=True --browser.gatherUsageStats=False
exec "$@"
