#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
sara_env/bin/python -m venv --system-site-packages demo/.venv
# Reuse the research environment read-only, but put new dependencies only in demo.
project_packages="$(sara_env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$project_packages" > demo/.venv/lib/python3.11/site-packages/project_environment.pth
demo/.venv/bin/python -m pip install -r demo/requirements.txt
printf '\nReady: demo/.venv/bin/python -m streamlit run demo/app.py\n'
