#!/bin/bash
# Start model server with logging for Vast.ai PyWorker

# Create log directory if not exists
mkdir -p /var/log/model

# Start model server and tee output to log file
# PyWorker monitors this log for readiness detection
python model_server.py 2>&1 | tee /var/log/model/server.log
