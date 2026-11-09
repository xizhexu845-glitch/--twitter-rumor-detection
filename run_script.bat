@echo off
cd /d "%~dp0"
python twitter_time_hawkes_rumor.py --dataset_dir twitter15 --output_json results/metrics.json
pause
