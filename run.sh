#!/bin/bash
# Simbo - Quick run script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one from .env.example"
    echo "cp .env.example .env"
    echo "Then add your OpenAI API key"
fi

# Run the Streamlit app
echo "Starting Simbo..."
cd src && streamlit run simbo/app.py
