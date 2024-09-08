#!/bin/sh

python train.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# python main.py
tail -f /dev/null