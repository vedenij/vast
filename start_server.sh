#!/bin/bash
# Start model server for Vast.ai Serverless
# PyWorker is started separately by Vast.ai via PYWORKER_REPO

# Create log directory if not exists
mkdir -p /var/log/model

# Start model server and log to file (PyWorker monitors this log)
# Use stdbuf for line-buffered output so PyWorker sees logs immediately
stdbuf -oL python model_server.py 2>&1 | stdbuf -oL tee /var/log/model/server.log
