#!/usr/bin/env bash
# Script to run tests with various configurations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running Multi-Modal Neural Network Tests${NC}"
echo "==========================================="

# Parse command line arguments
RUN_ALL=false
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_COVERAGE=false
RUN_SLOW=false
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --unit)
            RUN_UNIT=true
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --slow)
            RUN_SLOW=true
            shift
            ;;
        --markers)
            MARKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all           Run all tests"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --coverage      Run tests with coverage report"
            echo "  --slow          Include slow tests"
            echo "  --markers       Specify custom pytest markers"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --unit                    # Run unit tests only"
            echo "  $0 --integration             # Run integration tests only"
            echo "  $0 --all --coverage          # Run all tests with coverage"
            echo "  $0 --markers 'not slow'      # Run tests excluding slow ones"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m '$MARKERS'"
elif [ "$RUN_UNIT" = true ]; then
    echo -e "${YELLOW}Running unit tests only${NC}"
    PYTEST_CMD="$PYTEST_CMD -m 'not integration'"
elif [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${YELLOW}Running integration tests only${NC}"
    PYTEST_CMD="$PYTEST_CMD -m integration"
fi

# Add slow test handling
if [ "$RUN_SLOW" = false ] && [ "$RUN_ALL" = false ]; then
    echo -e "${YELLOW}Skipping slow tests (use --slow to include)${NC}"
    if [ -n "$MARKERS" ]; then
        PYTEST_CMD="$PYTEST_CMD and not slow"
    else
        PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
    fi
fi

# Add coverage if requested
if [ "$RUN_COVERAGE" = true ]; then
    echo -e "${YELLOW}Running with coverage${NC}"
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

# Run tests
echo -e "${GREEN}Executing: $PYTEST_CMD${NC}"
echo ""

eval $PYTEST_CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    
    if [ "$RUN_COVERAGE" = true ]; then
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
