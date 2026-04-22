@echo off
REM Launch the watermarking demo on http://127.0.0.1:8000
pushd "%~dp0backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
popd
