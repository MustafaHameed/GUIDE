#!/bin/bash
# GUIDE Paper Reproduction Script
# One-shot script to reproduce all paper results with exact versioning

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts/paper-$TIMESTAMP"

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print script header
print_header() {
    echo "=========================================="
    echo "GUIDE Paper Reproduction Script"
    echo "=========================================="
    echo "Timestamp: $TIMESTAMP"
    echo "Project Root: $PROJECT_ROOT"
    echo "Artifacts Dir: $ARTIFACTS_DIR"
    echo "=========================================="
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command_exists python; then
        log_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
    
    # Check make
    if ! command_exists make; then
        log_error "Make not found. Please install make"
        exit 1
    fi
    
    # Check required data files
    if [ ! -f "$PROJECT_ROOT/student-mat.csv" ]; then
        log_warning "student-mat.csv not found in project root"
        log_info "Please ensure the UCI Student Performance dataset is available"
    fi
    
    log_success "Prerequisites check completed"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables for reproducibility
    export PYTHONHASHSEED=0
    export PYTHONDONTWRITEBYTECODE=1
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -r requirements-paper.txt
    
    # Create artifact directory
    mkdir -p "$ARTIFACTS_DIR"
    
    log_success "Environment setup completed"
}

# Function to run the complete pipeline
run_pipeline() {
    log_info "Starting complete reproduction pipeline..."
    
    cd "$PROJECT_ROOT"
    
    # Export environment variables
    export PYTHONHASHSEED=0
    
    # Run make targets in order
    local targets=(
        "setup"
        "data"
        "eda"
        "train"
        "early-risk"
        "nested-cv"
        "transfer"
        "fairness"
        "explain"
        "paper-assets"
    )
    
    for target in "${targets[@]}"; do
        log_info "Running make target: $target"
        if make "$target"; then
            log_success "Completed: $target"
        else
            log_error "Failed: $target"
            exit 1
        fi
        echo ""
    done
    
    log_success "Pipeline completed successfully"
}

# Function to copy artifacts to versioned directory
copy_artifacts() {
    log_info "Copying artifacts to versioned directory..."
    
    cd "$PROJECT_ROOT"
    
    # Copy all outputs
    for dir in figures tables reports models; do
        if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
            log_info "Copying $dir..."
            cp -r "$dir" "$ARTIFACTS_DIR/"
        fi
    done
    
    # Copy configuration and metadata
    cp requirements-paper.txt "$ARTIFACTS_DIR/"
    cp Makefile "$ARTIFACTS_DIR/"
    
    # Create reproduction metadata
    cat > "$ARTIFACTS_DIR/reproduction_metadata.txt" << EOF
GUIDE Paper Reproduction Metadata
================================
Reproduction Timestamp: $TIMESTAMP
Python Version: $(python --version 2>&1)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "Not available")
Git Branch: $(git branch --show-current 2>/dev/null || echo "Not available")
Working Directory: $PROJECT_ROOT
Environment Variables:
  PYTHONHASHSEED=$PYTHONHASHSEED
  PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE

Package Versions:
$(pip freeze)

System Information:
$(uname -a)
EOF
    
    log_success "Artifacts copied to $ARTIFACTS_DIR"
}

# Function to generate summary report
generate_summary() {
    log_info "Generating reproduction summary..."
    
    local summary_file="$ARTIFACTS_DIR/reproduction_summary.md"
    
    cat > "$summary_file" << EOF
# GUIDE Paper Reproduction Summary

**Reproduction Date:** $(date)  
**Timestamp:** $TIMESTAMP  
**Python Version:** $(python --version 2>&1)  

## Generated Artifacts

### Figures
$(find "$ARTIFACTS_DIR/figures" -name "*.png" -o -name "*.pdf" 2>/dev/null | wc -l) figures generated

Key publication figures:
$(ls "$ARTIFACTS_DIR/figures"/ 2>/dev/null | grep -E "(eda_|roc_|fairness_|shap_)" | head -10 || echo "No key figures found")

### Tables
$(find "$ARTIFACTS_DIR/tables" -name "*.csv" 2>/dev/null | wc -l) tables generated

Key publication tables:
$(ls "$ARTIFACTS_DIR/tables"/ 2>/dev/null | grep -E "(classification_|fairness_|metrics_)" | head -10 || echo "No key tables found")

### Reports
$(find "$ARTIFACTS_DIR/reports" -name "*.md" -o -name "*.html" 2>/dev/null | wc -l) reports generated

## Reproduction Instructions

To reproduce these results:

1. Clone the repository
2. Install exact dependencies: \`pip install -r requirements-paper.txt\`
3. Run: \`./scripts/reproduce_paper.sh\`

Or use the Makefile:
\`\`\`bash
export PYTHONHASHSEED=0
make all
\`\`\`

## Verification

All results are generated with:
- Fixed random seed (PYTHONHASHSEED=0)
- Exact package versions (requirements-paper.txt)
- Deterministic algorithms where possible

For questions, see the project README or documentation.
EOF
    
    log_success "Summary report generated: $summary_file"
}

# Function to display completion message
print_completion() {
    echo ""
    echo "=========================================="
    echo "Paper Reproduction Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "📁 Artifacts Location: $ARTIFACTS_DIR"
    echo ""
    echo "📊 Key Outputs:"
    echo "   • Figures: $ARTIFACTS_DIR/figures/"
    echo "   • Tables: $ARTIFACTS_DIR/tables/"
    echo "   • Reports: $ARTIFACTS_DIR/reports/"
    echo "   • Models: $ARTIFACTS_DIR/models/"
    echo ""
    echo "📝 Summary: $ARTIFACTS_DIR/reproduction_summary.md"
    echo "🔧 Metadata: $ARTIFACTS_DIR/reproduction_metadata.txt"
    echo ""
    echo "🎯 All publication-ready artifacts are now available!"
    echo "=========================================="
}

# Main execution
main() {
    print_header
    check_prerequisites
    setup_environment
    run_pipeline
    copy_artifacts
    generate_summary
    print_completion
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --help|-h)
            echo "GUIDE Paper Reproduction Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-deps   Skip dependency installation"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "This script reproduces all paper results with exact versioning."
            echo "Results are saved to artifacts/paper-TIMESTAMP/"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"