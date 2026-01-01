#!/bin/bash
# Start model server for Vast.ai Serverless
# PyWorker is started separately by Vast.ai via PYWORKER_REPO
# Note: .has_benchmark is created by worker.py, not here

MODEL_LOG="${MODEL_LOG:-/var/log/model/server.log}"

# Create log directory if not exists
mkdir -p "$(dirname "$MODEL_LOG")"

# Rotate old log if exists (for debugging)
if [ -f "$MODEL_LOG" ]; then
    mv "$MODEL_LOG" "$MODEL_LOG.old"
fi

# Start model server and log to file (PyWorker monitors this log)
# Use > to clear log on restart (not >>), stdbuf for line-buffered output
stdbuf -oL python model_server.py 2>&1 | stdbuf -oL tee "$MODEL_LOG"
