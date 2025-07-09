#!/bin/bash

# install.sh - Install braingraph project dependencies using uv

set -e  # Exit on any error

echo "🚀 Setting up braingraph environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment named 'braingraph'
echo "📦 Creating virtual environment 'braingraph'..."
uv venv braingraph

# Activate the virtual environment
echo "🔧 Activating virtual environment..."
source braingraph/bin/activate

# Install Python packages
echo "📚 Installing Python packages..."
uv pip install \
    numpy \
    pandas \
    networkx \
    matplotlib \
    seaborn \
    statsmodels \
    bctpy \
    python-louvain \
    torch \
    scikit-learn

echo "✅ Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "   source braingraph/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "   deactivate"
