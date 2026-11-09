#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required. Install it from https://www.python.org/downloads/macos/ and run this launcher again."
    read -r -p "Press Return to close. "
    exit 1
fi
exec python3 uhem_chat_launcher.py "$@"
