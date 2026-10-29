#Requires -Version 5.1
<#
  Create an independent TRIHAERA virtual environment without re-downloading
  Levis's large installed packages. Does not modify Levis.
#>
[CmdletBinding()]
param([switch]$RunTests)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$MlRoot = Split-Path $ProjectRoot -Parent
$LevisPython = Join-Path $MlRoot 'levis-mk1\.venv\Scripts\python.exe'
$TargetEnv = Join-Path $ProjectRoot '.venv'
$TargetPython = Join-Path $TargetEnv 'Scripts\python.exe'

if (-not (Test-Path $LevisPython)) {
    throw "Cannot find Levis interpreter: $LevisPython"
}
if (Test-Path $TargetEnv) {
    throw "TRIHAERA .venv already exists: $TargetEnv. No files changed."
}

$BasePython = (& $LevisPython -c 'import sys; print(sys._base_executable)').Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $BasePython)) {
    throw 'Could not locate the base Python used by Levis.'
}

Write-Host 'Creating standalone TRIHAERA environment...'
& $BasePython -m venv $TargetEnv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }

$SourcePackages = (& $LevisPython -c 'import sysconfig; print(sysconfig.get_path("purelib"))').Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $SourcePackages)) {
    throw 'Cannot locate Levis site-packages.'
}
$TargetPackages = (& $TargetPython -c 'import sysconfig; print(sysconfig.get_path("purelib"))').Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TargetPackages)) {
    throw 'Cannot locate TRIHAERA site-packages.'
}

Write-Host "Copying installed packages (no downloads): $SourcePackages"
& robocopy $SourcePackages $TargetPackages /E /XJ /R:1 /W:1 /NP /NFL /NDL
$RoboCopyCode = $LASTEXITCODE
$global:LASTEXITCODE = 0
if ($RoboCopyCode -ge 8) { throw "robocopy failed with code $RoboCopyCode" }

$Verify = @'
import pathlib, sys, sysconfig, torch
prefix = pathlib.Path(sys.prefix).resolve()
exe = pathlib.Path(sys.executable).resolve()
torch_path = pathlib.Path(torch.__file__).resolve()
print("Python:", exe)
print("PyTorch:", torch.__version__)
print("Torch location:", torch_path)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
assert str(exe).lower().startswith(str(prefix).lower())
assert str(torch_path).lower().startswith(str(prefix).lower()), "PyTorch resolves outside TRIHAERA!"
for file in pathlib.Path(sysconfig.get_path("purelib")).glob("*.pth"):
    text = file.read_text(errors="replace")
    if "levis-mk1" in text.lower():
        print("WARNING: Levis absolute reference in", file.name, "- inspect before using editable packages")
'@
& $TargetPython -c $Verify
if ($LASTEXITCODE -ne 0) { throw 'TRIHAERA standalone environment validation failed.' }

Write-Host 'Independent environment ready. No Levis packages were changed.'
if ($RunTests) {
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    Push-Location $ProjectRoot
    try {
        & $TargetPython -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'pytest failed' }
        & $TargetPython scripts/smoke.py --device cuda
        if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke failed' }
    }
    finally { Pop-Location }
}
