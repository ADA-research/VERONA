#!/bin/bash

# Script to setup a virtual environment for ada-verona
# Usage: ./setup_venv.sh <venv_folder_path> [ada_verona_version] [auto_verify_version]

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <venv_folder_path> [ada_verona_version] [auto_verify_version]"
    echo "Example: $0 /path/to/venv/folder"
    echo "Example: $0 /path/to/venv/folder 0.1.7"
    echo "Example: $0 /path/to/venv/folder 0.1.7 0.1.0"
    exit 1
fi

VENV_FOLDER="$1"
ADA_VERONA_VERSION="${2:-}"  # Optional ada-verona version
AUTO_VERIFY_VERSION="${3:-}"  # Optional auto-verify version

# Validate paths
if [ ! -d "$VENV_FOLDER" ]; then
    echo "Error: Virtual environment folder '$VENV_FOLDER' does not exist"
    exit 1
fi

# Function to find next available environment number
find_next_env_number() {
    local base_folder="$1"
    local env_num=1
    
    while [ -d "$base_folder/ada_verona_env_$env_num" ]; do
        ((env_num++))
    done
    
    echo $env_num
}

# Find the next available environment number
ENV_NUM=$(find_next_env_number "$VENV_FOLDER")
VENV_NAME="ada_verona_env_$ENV_NUM"
VENV_PATH="$VENV_FOLDER/$VENV_NAME"

echo "Setting up virtual environment: $VENV_NAME"
echo "Virtual environment path: $VENV_PATH"
if [ -n "$ADA_VERONA_VERSION" ]; then
    echo "ada-verona version: $ADA_VERONA_VERSION"
else
    echo "ada-verona version: latest"
fi
if [ -n "$AUTO_VERIFY_VERSION" ]; then
    echo "auto-verify version: $AUTO_VERIFY_VERSION"
else
    echo "auto-verify version: latest"
fi

# Load required modules BEFORE creating virtual environment
echo "Loading required modules..."
module load GCC/11.3.0
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load Python/3.10.4
module save verona-modules

# Verify Python version
echo "Verifying Python version..."
python --version
if ! python --version 2>&1 | grep -q "Python 3.10"; then
    echo "Error: Python 3.10 is required but $(python --version) is active"
    echo "Please ensure Python/3.10.4 module is loaded correctly"
    exit 1
fi

# Create virtual environment with the correct Python version
echo "Creating virtual environment..."
python -m venv "$VENV_PATH"

# Verify the virtual environment was created with correct Python version
echo "Verifying virtual environment Python version..."
"$VENV_PATH/bin/python" --version
if ! "$VENV_PATH/bin/python" --version 2>&1 | grep -q "Python 3.10"; then
    echo "Error: Virtual environment was created with wrong Python version:"
    "$VENV_PATH/bin/python" --version
    echo "This indicates the Python/3.10.4 module was not loaded correctly before venv creation"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
uv pip install --upgrade pip setuptools wheel build

# Install ada-verona
echo "Installing ada-verona..."
if [ -n "$ADA_VERONA_VERSION" ]; then
    echo "Installing ada-verona version $ADA_VERONA_VERSION from PyPI..."
    uv pip install "ada-verona==$ADA_VERONA_VERSION"
else
    echo "Installing latest ada-verona from PyPI..."
    uv pip install ada-verona
fi

# Install ada-auto-verify from PyPI
echo "Installing ada-auto-verify..."
if [ -n "$AUTO_VERIFY_VERSION" ]; then
    echo "Installing auto-verify version $AUTO_VERIFY_VERSION from PyPI..."
    uv pip install "auto-verify==$AUTO_VERIFY_VERSION"
else
    echo "Installing latest auto-verify from PyPI..."
    uv pip install auto-verify
fi

echo "Virtual environment setup complete!"
echo "Virtual environment: $VENV_PATH"
echo "To activate: source $VENV_PATH/bin/activate" 