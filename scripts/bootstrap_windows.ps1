#Requires -Version 5.1
<#
  Create (or resume creation of) an independent TRIHAERA .venv by copying
  installed site-packages from Levis. Never changes Levis or downloads wheels.
  Invoke: powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
#>
[CmdletBinding()]
param([switch]$RunTests)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$MlRoot = Split-Path $ProjectRoot -Parent
$LevisPython = Join-Path $MlRoot 'levis-mk1\.venv\Scripts\python.exe'
$TargetEnv = Join-Path $ProjectRoot '.venv'
$TargetPython = Join-Path $TargetEnv 'Scripts\python.exe'
$Probe = Join-Path $PSScriptRoot 'env_probe.py'

function Get-PythonPath {
    param([string]$PythonExe, [string]$Action)
    # Running a Python file avoids Windows PowerShell 5.1 stripping nested
    # quotes from arguments passed to python -c.
    $lines = @(& $PythonExe $Probe $Action)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1 -or [string]::IsNullOrWhiteSpace($lines[0])) {
        throw "Unable to read $Action from $PythonExe"
    }
    return ([string]$lines[0]).Trim()
}

if (-not (Test-Path $LevisPython -PathType Leaf)) {
    throw "Cannot find Levis interpreter: $LevisPython"
}
$BasePython = Get-PythonPath -PythonExe $LevisPython -Action 'base-executable'
if (-not (Test-Path $BasePython -PathType Leaf)) {
    throw "Could not locate base Python from Levis: $BasePython"
}

if (Test-Path $TargetEnv) {
    # The first bootstrap version created .venv before its quoting failure.
    # Permit a safe resume, but never replace an unrelated existing env.
    if (-not (Test-Path $TargetPython -PathType Leaf) -or
        -not (Test-Path (Join-Path $TargetEnv 'pyvenv.cfg') -PathType Leaf)) {
        throw "Incomplete or unrelated TRIHAERA .venv at $TargetEnv; inspect manually before changing it."
    }
    $ActualPrefix = Get-PythonPath -PythonExe $TargetPython -Action 'prefix'
    $ExpectedPrefix = (Resolve-Path $TargetEnv).Path
    if (-not [string]::Equals($ActualPrefix, $ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use environment with unexpected prefix: $ActualPrefix"
    }
    Write-Host 'Found existing TRIHAERA .venv; resuming package copy safely.'
}
else {
    Write-Host 'Creating standalone TRIHAERA environment...'
    & $BasePython -m venv $TargetEnv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
}

$SourcePackages = Get-PythonPath -PythonExe $LevisPython -Action 'site-packages'
$TargetPackages = Get-PythonPath -PythonExe $TargetPython -Action 'site-packages'
if (-not (Test-Path $SourcePackages -PathType Container) -or
    -not (Test-Path $TargetPackages -PathType Container)) {
    throw 'Could not find source or target site-packages.'
}
if ([string]::Equals($SourcePackages, $TargetPackages, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source and target site-packages are identical; refusing to copy.'
}

Write-Host "Copying packages from $SourcePackages to $TargetPackages (no downloads)..."
& robocopy $SourcePackages $TargetPackages /E /XJ /R:1 /W:1 /NP /NFL /NDL
$RoboCopyCode = $LASTEXITCODE
$global:LASTEXITCODE = 0  # robocopy exit codes 0-7 mean success / skipped / differences
if ($RoboCopyCode -ge 8) { throw "Package copy failed (robocopy exit code $RoboCopyCode)" }

Write-Host 'Verifying standalone interpreter and PyTorch...'
& $TargetPython $Probe verify
if ($LASTEXITCODE -ne 0) { throw 'Standalone environment validation failed. You may rerun this bootstrap.' }

Write-Host 'TRIHAERA environment ready; Levis was not modified.'
if ($RunTests) {
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    Push-Location $ProjectRoot
    try {
        & $TargetPython -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'pytest failed.' }
        & $TargetPython scripts/smoke.py --device cuda
        if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke test failed.' }
    }
    finally { Pop-Location }
}
