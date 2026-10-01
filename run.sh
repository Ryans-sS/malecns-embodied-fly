#!/usr/bin/env sh
cd "$(dirname "$0")"
export FLASK_APP=codex.main
export FLASK_SECRET_KEY=dev_dummy_key
export APP_ENVIRONMENT=DEV
export BUILD_GIT_SHA=local
export BUILD_TIMESTAMP="$(date)"
exec ./.venv/Scripts/python.exe -m flask run --host 127.0.0.1 --port 5000 "$@"
