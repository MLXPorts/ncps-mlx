#!/bin/bash
# Setup script for NCPS documentation tools

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"

# Log file for installation
LOG_FILE="$DOCS_DIR/logs/setup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to check command existence
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log "${RED}Error: $1 is not installed${NC}"
        log "Please install $1 first:"
        case "$1" in
            brew)
                log "Visit: https://brew.sh"
                ;;
            apt-get)
                log "Your system doesn't appear to be using apt package manager."
                log "Please install packages manually or use your system's package manager."
                ;;
            python)
                log "Visit: https://www.python.org/downloads/"
                ;;
            node)
                log "Visit: https://nodejs.org/"
                ;;
            *)
                log "Please install $1 using your system's package manager"
                ;;
        esac
        exit 1
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    log "${GREEN}Checking system dependencies...${NC}"
    
    # Check Python version
    if ! python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ is required'" 2>/dev/null; then
        log "${RED}Error: Python 3.8 or higher is required${NC}"
        exit 1
    fi

    # Install pandoc if not present
    if ! command -v pandoc &> /dev/null; then
        log "${YELLOW}Installing pandoc...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # Check for Homebrew
            check_command brew
            brew install pandoc
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Check for apt-get
            check_command apt-get
            sudo apt-get update
            sudo apt-get install -y pandoc
        else
            log "${RED}Please install pandoc manually: https://pandoc.org/installing.html${NC}"
            exit 1
        fi
    fi

    # Install Node.js if not present (needed for JS/TS formatting)
    if ! command -v node &> /dev/null; then
        log "${YELLOW}Installing Node.js...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install node
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
        else
            log "${RED}Please install Node.js manually: https://nodejs.org/${NC}"
            exit 1
        fi
    fi
}

# Function to create and activate virtual environment
setup_virtual_environment() {
    log "${GREEN}Setting up Python virtual environment...${NC}"
    
    # Remove existing venv if present
    if [ -d "$DOCS_DIR/.venv" ]; then
        log "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$DOCS_DIR/.venv"
    fi
    
    # Create new venv
    python -m venv "$DOCS_DIR/.venv"
    source "$DOCS_DIR/.venv/bin/activate"
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip
}

# Function to install Python dependencies
install_python_dependencies() {
    log "${GREEN}Installing Python dependencies...${NC}"
    
    # Install requirements
    pip install -r "$SCRIPT_DIR/requirements.txt"
    if [ -f "$DOCS_DIR/../.readthedocs-requirements.txt" ]; then
        pip install -r "$DOCS_DIR/../.readthedocs-requirements.txt"
    fi
    
    # Verify installations
    log "Verifying installations..."
    python -c "import sphinx" || { log "${RED}Error: Sphinx installation failed${NC}"; exit 1; }
    python -c "import black" || { log "${RED}Error: Black installation failed${NC}"; exit 1; }
}

# Function to setup documentation structure
setup_documentation_structure() {
    log "${GREEN}Setting up documentation structure...${NC}"
    
    # Create necessary directories
    directories=(
        "_build"
        "_static"
        "_static/img"
        "_templates"
        "architecture/abstractions"
        "architecture/implementation"
        "architecture/knowledge"
        "architecture/design"
        "architecture/research"
        "api"
        "guides"
        "examples"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$DOCS_DIR/$dir"
    done
}

# Function to setup documentation tools
setup_documentation_tools() {
    log "${GREEN}Setting up documentation tools...${NC}"
    
    # Make scripts executable
    chmod +x "$SCRIPT_DIR/doctools"
    chmod +x "$SCRIPT_DIR"/*.py
    
    # Create activation script
    log "Creating activation script..."
    cat > "$DOCS_DIR/activate_docs" << 'EOF'
#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
export PATH="$PATH:$(dirname "${BASH_SOURCE[0]}")/tools"
export PYTHONPATH="$(dirname "${BASH_SOURCE[0]}")/..:$PYTHONPATH"
echo -e "\033[0;32mDocumentation environment activated\033[0m"
echo "Use 'doctools help' to see available commands"
EOF
    
    chmod +x "$DOCS_DIR/activate_docs"
}

# Main setup process
main() {
    log "${GREEN}Starting NCPS documentation tools setup...${NC}"
    
    # Run setup steps
    install_system_dependencies
    setup_virtual_environment
    install_python_dependencies
    setup_documentation_structure
    setup_documentation_tools
    
    log "\n${GREEN}Setup completed successfully!${NC}"
    log "\nTo get started:"
    log "1. Run: ${YELLOW}source docs/activate_docs${NC}"
    log "2. Use ${YELLOW}doctools${NC} commands to manage documentation"
    log "3. Run ${YELLOW}doctools help${NC} for available commands"
    log "\nSetup log saved to: $LOG_FILE"
}

# Run main setup
main "$@"