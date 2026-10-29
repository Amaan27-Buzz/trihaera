"""PowerShell-safe environment introspection (no inline Python -c quoting).

This runs under the Python interpreter being inspected. It deliberately uses
only the standard library except when verifying the copied PyTorch installation.
"""

import argparse
import pathlib
import sys
import sysconfig


def verify() -> None:
    import torch

    prefix = pathlib.Path(sys.prefix).resolve()
    executable = pathlib.Path(sys.executable).resolve()
    torch_location = pathlib.Path(torch.__file__).resolve()
    try:
        executable.relative_to(prefix)
        torch_location.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(
            "TRIHAERA's executable and installed PyTorch must be inside its own .venv"
        ) from exc
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Target Python is not running inside a virtual environment")

    print(f"Python: {executable}")
    print(f"Environment: {prefix}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torch location: {torch_location}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    packages = pathlib.Path(sysconfig.get_path("purelib"))
    for path_file in sorted(packages.glob("*.pth")):
        text = path_file.read_text(errors="replace")
        if "levis-mk1" in text.casefold():
            print(
                f"WARNING: {path_file.name} references Levis. "
                "Inspect/remove or reinstall that editable dependency only in TRIHAERA."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["base-executable", "site-packages", "prefix", "verify"])
    args = parser.parse_args()
    if args.action == "base-executable":
        print(pathlib.Path(sys._base_executable).resolve())
    elif args.action == "site-packages":
        print(pathlib.Path(sysconfig.get_path("purelib")).resolve())
    elif args.action == "prefix":
        print(pathlib.Path(sys.prefix).resolve())
    else:
        verify()


if __name__ == "__main__":
    main()
