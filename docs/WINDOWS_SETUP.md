# Windows setup: independent TRIHAERA .venv, no PyTorch download

**Source:** Levis Python packages at `D:\Programming\03_Projects\personal-projects\05_ML\levis-mk1\.venv`.  
**Destination:** TRIHAERA's independent `.venv`.

The bootstrap **does not modify Levis or install packages over the internet**. It creates a fresh virtual environment using Levis's base Python installation and copies Levis's existing installed `site-packages` into the new environment with `robocopy`. Some editable install paths and executable wrappers are not portable; use `python -m` for commands and inspect any printed warnings.

## Clone the correct repository

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML'
git clone https://github.com/Amaan27-Buzz/trihaera.git
cd trihaera
```

If the repository is **already cloned**, do not clone again. Use:

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
```

## Create or resume environment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

A prior bootstrap incorrectly used embedded Python `-c` string quoting, which Windows PowerShell 5.1 changed. This produced `NameError: name 'purelib' is not defined` **after creating the target .venv**. The fixed bootstrap invokes a separate Python introspection script instead of embedding quoted Python code, and safely **resumes an existing target environment** after confirming it belongs to TRIHAERA. **Do not delete the partially created .venv** for that failure; rerunning the corrected script is sufficient.

Successful output should include Python and Torch locations inside `trihaera\.venv`, CUDA availability and (if enabled) the detected GPU.

## Verify model code

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke.py --device cuda
python scripts/train_sanity.py --device cuda --steps 15
```

If activation is restricted by PowerShell policy, execute `.\.venv\Scripts\python.exe` directly in place of `python`.

**Note:** The package copy may be sizeable but does not redownload wheels. Pip entry points, DLLs from external environments, editable package links, and some .pth files can require attention. The bootstrap warns on links back to Levis. If verification fails, report the full output; don't reinstall PyTorch or remove Levis's environment without diagnosing the issue.

The prototype smoke tests establish code correctness, not language-model capability or a three-state performance advantage.
