#!/bin/bash
# Setup script for face recognition and generation project

echo "Setting up face recognition and generation project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install pytorch_fid for FID calculation
echo "Installing pytorch_fid for FID calculation..."
pip install pytorch-fid

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p outputs/evaluation

echo "Setup complete! You can now use the project."
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the entire pipeline, use:"
echo "  python run_pipeline.py"
echo ""
echo "For more options, run:"
echo "  python run_pipeline.py --help" 