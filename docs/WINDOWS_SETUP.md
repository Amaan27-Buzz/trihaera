# Windows: independent TRIHAERA .venv, reusing Levis package files

Goal: create **TRIHAERA's own environment**, not share Levis's interpreter, not download a second multi-gigabyte PyTorch wheel.

Do not simply drag/copy the entire Windows .venv: pyvenv.cfg, Scripts launchers, editable imports and absolute .pth links may retain the Levis paths. Instead create a fresh venv using the same base Python, and then copy its installed package directory. This is a pragmatic local cloning method—not a guarantee that every CLI entry point or editable install is portable.

## Clone

```powershell
$Root = 'D:\Programming\03_Projects\personal-projects\05_ML'
Set-Location $Root
git clone https://github.com/Amaan27-Buzz/trihaera.git
Set-Location (Join-Path $Root 'trihaera')
```

## Copy installed packages into a new isolated environment

Run in PowerShell (not inside an already activated Levis environment):

```powershell
$LevisPython = Join-Path $Root 'levis-mk1\.venv\Scripts\python.exe'
$Target = Join-Path $Root 'trihaera\.venv'
if (-not (Test-Path $LevisPython)) { throw 'Levis Python not found' }
if (Test-Path $Target) { throw 'TRIHAERA .venv already exists; inspect before modifying' }

$BasePython = (& $LevisPython -c 'import sys; print(sys._base_executable)').Trim()
& $BasePython -m venv $Target
if ($LASTEXITCODE -ne 0) { throw 'Could not create TRIHAERA virtual environment' }

$Src = (& $LevisPython -c 'import sysconfig; print(sysconfig.get_path("purelib"))').Trim()
$TriPython = Join-Path $Target 'Scripts\python.exe'
$Dest = (& $TriPython -c 'import sysconfig; print(sysconfig.get_path("purelib"))').Trim()

robocopy $Src $Dest /E /XJ /R:1 /W:1 /NP
if ($LASTEXITCODE -ge 8) { throw 'Package-copy failed' }
$global:LASTEXITCODE = 0

& $TriPython -c 'import sys,torch,pathlib; p=pathlib.Path(sys.executable).resolve(); t=pathlib.Path(torch.__file__).resolve(); print("Python:",p); print("Torch:",torch.__version__,t); print("CUDA:",torch.cuda.is_available()); assert "trihaera" in str(p).lower() and "trihaera" in str(t).lower(), "Environment incorrectly points outside TRIHAERA"'
```

This copies packages including native Torch libraries, not Levis's project code. The new environment has its own Python executable and site-packages. **Do not install/upgrade dependencies in Levis.**

## Inspect external editable paths

```powershell
& $TriPython -c 'import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_path("purelib")); print("\n".join(f"{f}: {f.read_text(errors="replace")[:400]}" for f in p.glob("*.pth")))'
```

If a copied .pth or editable package refers to an absolute path under levis-mk1, remove or reinstall only that reference in TRIHAERA. Other entry-point executables in Scripts are not recreated automatically; prefer python -m module commands. Conda-only DLL dependencies and some unusual editable packages may need targeted repair.

## Run tests

```powershell
Set-Location (Join-Path $Root 'trihaera')
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& $TriPython -m pytest -q
& $TriPython scripts/smoke.py --device cuda
& $TriPython scripts/train_sanity.py --device cuda --steps 15
```

If CUDA fails, verify the original Levis CUDA environment and local NVIDIA drivers, then use --device cpu for tests. If pytest was not copied, install that small package into TRIHAERA only.

Reusing already downloaded wheel caches via pip --no-index is another valid option if package copying encounters binary path issues. Do not assume a full 650M pretrain fits in 8 GB.
