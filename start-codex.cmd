@echo off
cd /d "%~dp0"
set FLASK_APP=codex.main
set FLASK_SECRET_KEY=dev_dummy_key
set APP_ENVIRONMENT=DEV
set BUILD_GIT_SHA=local
set BUILD_TIMESTAMP=local
start "" http://127.0.0.1:5000
".venv\Scripts\python.exe" -m flask run --host 127.0.0.1 --port 5000
