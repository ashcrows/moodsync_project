#!/usr/bin/env bash
# MoodSync demo — macOS / Linux.
set -e
cd "$(dirname "$0")/.."

OS_ARG="${1:-auto}"   # pass 'mac', 'linux', 'windows', or leave blank for auto

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing light requirements"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "==> Running MoodSync demo (target OS: ${OS_ARG})"
python -m moodsync.cli demo --os "${OS_ARG}"

echo ""
echo "Done. Try the UI:   python -m moodsync.cli serve-app --os ${OS_ARG}"
echo "Or the API:         python -m moodsync.cli serve-api --os ${OS_ARG}"
