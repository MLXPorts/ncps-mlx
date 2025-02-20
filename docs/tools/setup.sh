#!/bin/bash
# Setup script for NCPS documentation tools

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up NCPS documentation tools..."

# Create Python virtual environment
echo "Creating virtual environment..."
python -m venv "$DOCS_DIR/.venv"
source "$DOCS_DIR/.venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install -r "$DOCS_DIR/../.readthedocs-requirements.txt"

# Install pandoc if not present
if ! command -v pandoc &> /dev/null; then
    echo "Installing pandoc..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install pandoc
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y pandoc
    else
        echo "Please install pandoc manually: https://pandoc.org/installing.html"
    fi
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x "$SCRIPT_DIR/doctools"
chmod +x "$SCRIPT_DIR"/*.py

# Create necessary directories
echo "Creating directories..."
mkdir -p "$DOCS_DIR/_build"
mkdir -p "$DOCS_DIR/_static"
mkdir -p "$DOCS_DIR/architecture/abstractions"
mkdir -p "$DOCS_DIR/architecture/implementation"
mkdir -p "$DOCS_DIR/architecture/knowledge"
mkdir -p "$DOCS_DIR/architecture/design"
mkdir -p "$DOCS_DIR/architecture/research"

# Create activation script
echo "Creating activation script..."
cat > "$DOCS_DIR/activate_docs" << 'EOF'
#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
export PATH="$PATH:$(dirname "${BASH_SOURCE[0]}")/tools"
echo "Documentation environment activated. Use 'doctools' to manage documentation."
EOF
chmod +x "$DOCS_DIR/activate_docs"

echo "Setup complete!"
echo
echo "To get started:"
echo "1. Run: source docs/activate_docs"
echo "2. Use 'doctools' commands to manage documentation"
echo "3. Run 'doctools help' for available commands"