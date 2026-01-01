"""
Vast.ai PyWorker Configuration for LLaMA GPU Benchmark.

This file configures the Vast.ai PyWorker to proxy requests to the model server.
The PyWorker handles:
- Request routing to model server
- Health checking via log monitoring
- Benchmarking for autoscaling

The model server (model_server.py) runs on port 18000 and handles GPU inference.
"""

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Model server configuration
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE = "/var/log/model/server.log"


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
        # Health check endpoint (with minimal benchmark config - required by PyWorker)
        # Actual benchmark is skipped via .has_benchmark file
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            max_queue_time=5.0,
            workload_calculator=lambda p: 1.0,
            benchmark_config=BenchmarkConfig(
                generator=lambda: {},
                runs=1,
                concurrency=1,
            ),
        ),

        # Main generation endpoint
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,  # PoW generation uses all GPUs
            max_queue_time=60.0,            # Reject if waiting > 60s
            workload_calculator=workload_calculator,
        ),

        # Warmup endpoint (waits for params from orchestrator)
        HandlerConfig(
            route="/warmup",
            allow_parallel_requests=False,
            max_queue_time=120.0,  # Warmup can wait longer
            workload_calculator=lambda p: 100.0,  # Fixed cost
        ),

        # Pooled mode endpoint (managed by orchestrator)
        HandlerConfig(
            route="/pooled",
            allow_parallel_requests=False,
            max_queue_time=60.0,
            workload_calculator=lambda p: 100.0,  # Fixed cost
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
    # Set fixed perf value to skip benchmark (min_load = number of workers)
    import os
    perf_value = os.environ.get("WORKER_PERF", "1")
    with open("/workspace/.has_benchmark", "w") as f:
        f.write(perf_value)

    # Start the PyWorker
    Worker(worker_config).run()
