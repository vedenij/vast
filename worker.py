"""
Vast.ai PyWorker Configuration for LLaMA GPU Benchmark.

This file configures the Vast.ai PyWorker to proxy requests to the model server.
The PyWorker handles:
- Request routing to model server
- Health checking via log monitoring
- Benchmarking for autoscaling

The model server (model_server.py) runs on port 18000 and handles GPU inference.
"""

import os
import random
import string

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Model server configuration
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE = "/var/log/model/server.log"


def generate_random_hash(length: int = 64) -> str:
    """Generate a random hex hash for benchmarking."""
    return ''.join(random.choices('0123456789abcdef', k=length))


def generate_random_public_key(length: int = 64) -> str:
    """Generate a random public key for benchmarking."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def benchmark_generator() -> dict:
    """
    Generate benchmark payload for /generate endpoint.
    Used to measure throughput for autoscaling decisions.
    """
    return {
        "block_hash": generate_random_hash(),
        "block_height": random.randint(1, 1000000),
        "public_key": generate_random_public_key(),
        "r_target": 1.5,
        "batch_size": 1024,
        "start_nonce": 0,
        "params": {
            "dim": 2048,
            "n_layers": 16,
            "n_heads": 16,
            "n_kv_heads": 8,
            "vocab_size": 128256,
            "seq_len": 1024,
            "rope_theta": 500000.0,
        }
    }


def workload_calculator(payload: dict) -> float:
    """
    Calculate workload cost for a request.
    Higher values indicate more expensive requests.
    Used for autoscaling decisions.
    """
    batch_size = payload.get("batch_size", 1024)
    # Workload is proportional to batch size
    return float(batch_size)


# Worker configuration
worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,

    # Request handlers
    handlers=[
        # Main generation endpoint with benchmarking
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,  # PoW generation uses all GPUs
            max_queue_time=60.0,            # Reject if waiting > 60s
            workload_calculator=workload_calculator,
            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                runs=2,        # Few runs since each takes time
                concurrency=1,  # One request at a time
            ),
        ),

        # Warmup endpoint (no benchmarking)
        HandlerConfig(
            route="/warmup",
            allow_parallel_requests=False,
            max_queue_time=120.0,  # Warmup can wait longer
            workload_calculator=lambda p: 100.0,  # Fixed cost
        ),

        # Pooled mode endpoint (no benchmarking)
        HandlerConfig(
            route="/pooled",
            allow_parallel_requests=False,
            max_queue_time=60.0,
            workload_calculator=lambda p: 100.0,  # Fixed cost
        ),

        # Health check (lightweight)
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,  # Health checks don't block
            max_queue_time=5.0,
            workload_calculator=lambda p: 1.0,  # Minimal cost
        ),
    ],

    # Log-based health detection
    # PyWorker monitors the model server log file for these patterns
    # NOTE: Patterns are PREFIX-based - must match START of log line!
    log_action_config=LogActionConfig(
        # Model server is ready when we see this pattern
        # Must match exactly what model_server.py prints (no timestamp prefix)
        on_load=[
            "Application startup complete.",
        ],

        # Error patterns that indicate failure
        on_error=[
            "Traceback (most recent call last):",
            "Error:",
            "CUDA error",
            "RuntimeError:",
            "OutOfMemoryError",
        ],

        # Informational patterns (optional, for logging)
        on_info=[
            "INFO - Starting",
            "INFO - Server ready",
        ],
    ),
)


if __name__ == "__main__":
    # Start the PyWorker
    # This monitors the model server log and proxies requests
    Worker(worker_config).run()
