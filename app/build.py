#!/usr/bin/env python3
"""
build.py – one-command build & test for the entire MoneyPrinterTurbo Enhanced project.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts" if os.name == "nt" else "bin") / "python"
NODE_MODULES = ROOT / "node_modules"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd, *, cwd=ROOT, check=True):
    """Run a shell command and stream stdout/stderr."""
    print(f"\n>>> {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)

def exists(path):
    return path.exists()

# --------------------------------------------------------------------------- #
# 1. Python virtualenv + deps
# --------------------------------------------------------------------------- #
def setup_python():
    if not exists(VENV):
        run(f"{sys.executable} -m venv {VENV}")
    run(f"{PYTHON} -m pip install --upgrade pip")
    if exists(ROOT / "requirements.txt"):
        run(f"{PYTHON} -m pip install -r requirements.txt")

# --------------------------------------------------------------------------- #
# 2. Node + JS build
# --------------------------------------------------------------------------- #
def setup_node():
    if not exists(NODE_MODULES):
        run("npm ci")          # reproducible install
    run("npm run build")       # or "vite build", "webpack", etc.

# --------------------------------------------------------------------------- #
# 3. Python lint / tests
# --------------------------------------------------------------------------- #
def lint_and_test():
    if exists(ROOT / "pyproject.toml") or exists(ROOT / "setup.cfg"):
        run(f"{PYTHON} -m flake8")
    if exists(ROOT / "tests"):
        run(f"{PYTHON} -m pytest tests/")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    try:
        setup_python()
        setup_node()
        lint_and_test()
        print("\n✅ All green!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed at step: {e.cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
