source venv/bin/activate && uvicorn main:app --host $(hostname -I) --port 5050 --workers 4