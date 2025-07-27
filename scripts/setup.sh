#!/bin/bash

# Setup script for MoneyPrinterTurbo project

# Create virtual environment using python3
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "Setup complete! To activate the virtual environment in the future, run:"
echo "source venv/bin/activate"
