#!/bin/bash
# Start model server and PyWorker for Vast.ai Serverless

# Create log directory if not exists
mkdir -p /var/log/model

# Start model server in background, log to file
python model_server.py 2>&1 | tee /var/log/model/server.log &
MODEL_PID=$!

# Wait for model server to start
sleep 5

# Start PyWorker (communicates with Vast.ai platform)
python worker.py &
WORKER_PID=$!

echo "Started model_server.py (PID: $MODEL_PID) and worker.py (PID: $WORKER_PID)"

# Wait for either process to exit
wait -n $MODEL_PID $WORKER_PID

# If one exits, kill the other
kill $MODEL_PID $WORKER_PID 2>/dev/null
wait
