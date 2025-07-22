#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[testing]"

echo "Virtual environment created and dependencies installed."
echo "You can now run tests with 'tox' or 'pytest -v'."
