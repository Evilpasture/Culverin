param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("Build", "Test", "Stress", "Leak", "Perf", "Full")]
    [string]$Task = "Full"
)

$ProjectRoot = $PSScriptRoot
$env:PYTHONPATH = "$ProjectRoot\src"
$env:UV_LINK_MODE = "copy"

# 1. Sync deps
Write-Host "--- Syncing Dependencies ---" -ForegroundColor Cyan
uv pip install numpy psutil scikit-build-core ninja --quiet

# 2. Clang DLL setup
$ClangPath = Get-Command clang -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if ($ClangPath) {
    $ClangBinDir = Split-Path $ClangPath
    $env:CLANG_BIN_PATH = $ClangBinDir
    $env:Path = "$ClangBinDir;$env:Path"
}

try {
    if ($Task -eq "Build" -or $Task -eq "Full") {
        Write-Host "`n>>> Task: Cleaning and Building Extension..." -ForegroundColor Magenta
        
        # 1. CLEAN
        Get-ChildItem "$ProjectRoot\src\culverin\*" -Include *.pyd, *.so, *.pdb | Remove-Item -Force -ErrorAction SilentlyContinue
        
        # 2. BUILD
        uv run python tools/build_project.py
        
        # 3. VERIFY & GENERATE STUBS
        if (-not (Get-ChildItem "$ProjectRoot\src\culverin\_culverin_c.*" -Include *.pyd, *.so)) {
            Write-Host "ERROR: Build finished but no binary was deployed to src/culverin!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ">>> Generating Type Stubs (.pyi)..." -ForegroundColor Yellow
        uv run tools/gen_stubs.py
    }

    if ($Task -eq "Test" -or $Task -eq "Full") {
        Write-Host "`n>>> Task: Running Unit Tests..." -ForegroundColor Magenta
        uv run python tests/test_core.py
    }

    if ($Task -eq "Perf" -or $Task -eq "Full") {
        Write-Host "`n>>> Task: Running Performance Regression Suite..." -ForegroundColor Magenta
        uv run python tests/test_perf.py
    }

    if ($Task -eq "Leak" -or $Task -eq "Full") {
        Write-Host "`n>>> Task: Running Memory Leak Check..." -ForegroundColor Magenta
        uv run python tests/benchmark.py --leak
    }

    if ($Task -eq "Stress") {
        Write-Host "`n>>> Task: Running Multi-threaded Stress Test..." -ForegroundColor Magenta
        uv run python tests/benchmark.py --stress
    }

    Write-Host "`nLocal CI Run Completed Successfully!" -ForegroundColor Green
}
catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}