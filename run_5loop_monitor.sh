#!/bin/bash
# Wrapper script to run 5-loop monitor with proper environment

# Set all environment variables
export PER_LOOP_TIMEOUT=300
export OVERALL_TIMEOUT=1800
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002
export IBKR_CLIENT_ID=9002
export ALLOW_ORDERS=0
export DRY_RUN=1
export PYTHONHASHSEED=0

# Create timestamp
TS=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$HOME/logs/monitors/pipeline_5loops_${TS}.log"

echo "Starting 5-loop monitor at $(date)"
echo "Log file: $LOG_FILE"
echo "PER_LOOP_TIMEOUT: $PER_LOOP_TIMEOUT seconds"
echo "OVERALL_TIMEOUT: $OVERALL_TIMEOUT seconds"

# Run the monitor with timeout and logging
timeout $OVERALL_TIMEOUT /home/ubuntu/ml_env/bin/python -u tools/pipeline_monitor_5loops.py 2>&1 | tee "$LOG_FILE"

echo "Monitor completed at $(date)"