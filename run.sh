#!/bin/bash
# Iniciar la API TermEx
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 5042 --timeout-keep-alive 30
