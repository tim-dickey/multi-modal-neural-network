# PowerShell script to run tests with various configurations

param(
    [switch]$All,
    [switch]$Unit,
    [switch]$Integration,
    [switch]$Coverage,
    [switch]$Slow,
    [string]$Markers,
    [switch]$Help
)

# Color output functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Info { Write-ColorOutput Cyan $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }

# Show help
if ($Help) {
    Write-Info "Multi-Modal Neural Network Test Runner"
    Write-Output ""
    Write-Output "Usage: .\run_tests.ps1 [OPTIONS]"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -All           Run all tests"
    Write-Output "  -Unit          Run only unit tests"
    Write-Output "  -Integration   Run only integration tests"
    Write-Output "  -Coverage      Run tests with coverage report"
    Write-Output "  -Slow          Include slow tests"
    Write-Output "  -Markers       Specify custom pytest markers"
    Write-Output "  -Help          Show this help message"
    Write-Output ""
    Write-Output "Examples:"
    Write-Output "  .\run_tests.ps1 -Unit                    # Run unit tests only"
    Write-Output "  .\run_tests.ps1 -Integration             # Run integration tests only"
    Write-Output "  .\run_tests.ps1 -All -Coverage           # Run all tests with coverage"
    Write-Output "  .\run_tests.ps1 -Markers 'not slow'      # Run tests excluding slow ones"
    exit 0
}

Write-Success "Running Multi-Modal Neural Network Tests"
Write-Output "==========================================="

# Build pytest command
$PytestCmd = "pytest"

# Add markers
if ($Markers) {
    $PytestCmd += " -m '$Markers'"
}
elseif ($Unit) {
    Write-Warning "Running unit tests only"
    $PytestCmd += " -m 'not integration'"
}
elseif ($Integration) {
    Write-Warning "Running integration tests only"
    $PytestCmd += " -m integration"
}

# Add slow test handling
if (-not $Slow -and -not $All) {
    Write-Warning "Skipping slow tests (use -Slow to include)"
    if ($Markers) {
        $PytestCmd += " and not slow"
    }
    else {
        if ($PytestCmd -match "-m") {
            $PytestCmd += " and not slow"
        }
        else {
            $PytestCmd += " -m 'not slow'"
        }
    }
}

# Add coverage if requested
if ($Coverage) {
    Write-Warning "Running with coverage"
    $PytestCmd += " --cov=src --cov-report=html --cov-report=term-missing"
}

# Run tests
Write-Info "Executing: $PytestCmd"
Write-Output ""

Invoke-Expression $PytestCmd

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Output ""
    Write-Success "✓ All tests passed!"
    
    if ($Coverage) {
        Write-Success "Coverage report generated in htmlcov\index.html"
    }
}
else {
    Write-Output ""
    Write-Error "✗ Some tests failed"
    exit 1
}
