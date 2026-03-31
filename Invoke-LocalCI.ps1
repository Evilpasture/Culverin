param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("Build", "Test", "Stress", "Leak", "Perf", "Full")]
    [string]$Task = "Full"
)

$ProjectRoot = $PSScriptRoot

# REMOVED: $env:PYTHONPATH = "$ProjectRoot\src"
# We want Python to load from the installed wheel in site-packages, not the source dir!

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
        
        # 3. VERIFY
        if (-not (Get-ChildItem "$ProjectRoot\src\culverin\_culverin_c.*" -Include *.pyd, *.so)) {
            Write-Host "ERROR: Build finished but no binary was deployed to src/culverin!" -ForegroundColor Red
            exit 1
        }

        Write-Host ">>> Installing pre-compiled wheel from dist/..." -ForegroundColor Cyan
    
        # Find the most recent wheel
        $Wheel = Get-ChildItem "$ProjectRoot\dist\culverin-*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($Wheel) {
            Write-Host "Found: $($Wheel.Name)" -ForegroundColor Gray
            
            # This completely deletes the existing site-packages/culverin folder!
            uv pip install "$($Wheel.FullName)" --force-reinstall --no-deps --quiet
            
            # Find where uv actually installed the package
            $DestDir = Get-ChildItem -Path "$ProjectRoot\.venv" -Directory -Filter "culverin" -Recurse | 
                       Where-Object { $_.FullName -match "site-packages" } | 
                       Select-Object -ExpandProperty FullName -First 1

            Write-Host ">>> Target Site-Packages: $DestDir" -ForegroundColor Cyan

            # ====================================================================
            # POST-INSTALL INJECTION: Restore the files wiped by --force-reinstall
            # ====================================================================

            # 1. Copy ASan DLLs & PDB
            $LLVMRoot = "C:\Program Files\LLVM"
            $AsanDll = Get-ChildItem -Path "C:\Program Files\LLVM\lib\clang" -Filter "clang_rt.asan_dynamic*.dll" -Recurse | Select-Object -First 1

            if ($AsanDll -and (Test-Path $DestDir)) {
                Write-Host "  -> Injecting ASan Runtime DLL..." -ForegroundColor Gray
                Copy-Item $AsanDll.FullName -Destination $DestDir -Force
                
                $AsanPdb = Get-ChildItem -Path "$LLVMRoot\lib\clang" -Filter "clang_rt.asan_dynamic-x86_64.pdb" -Recurse | Select-Object -First 1
                if ($AsanPdb) { Copy-Item $AsanPdb.FullName -Destination $DestDir -Force }
            }

            # 2. Copy the Extension's PDB (For C++ Debugging)
            $ExtPdb = Get-ChildItem -Path "$ProjectRoot" -Filter "_culverin_c*.pdb" -Recurse | Select-Object -First 1
            if ($ExtPdb -and (Test-Path $DestDir)) {
                Write-Host "  -> Injecting Extension PDB ($($ExtPdb.Name))..." -ForegroundColor Gray
                Copy-Item $ExtPdb.FullName -Destination $DestDir -Force
            }

            # 3. Copy py.typed (For Mypy/Pyright)
            $PyTyped = "$ProjectRoot\src\culverin\py.typed"
            if (Test-Path $PyTyped) {
                Write-Host "  -> Injecting py.typed..." -ForegroundColor Gray
                Copy-Item $PyTyped -Destination $DestDir -Force
            }

            # 4. Generate & Copy Type Stubs (.pyi)
            Write-Host ">>> Generating Type Stubs (.pyi)..." -ForegroundColor Yellow
            uv run tools/gen_stubs.py
            
            Get-ChildItem -Path "$ProjectRoot\src\culverin" -Filter "*.pyi" | ForEach-Object {
                Write-Host "  -> Injecting Stub: $($_.Name)" -ForegroundColor Gray
                Copy-Item $_.FullName -Destination $DestDir -Force
            }

        } else {
            Write-Host "ERROR: No wheel found in dist/ to install!" -ForegroundColor Red
            exit 1
        }
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