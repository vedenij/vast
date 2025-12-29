"""
Vast.ai Model Server - HTTP server for PoW generation.

This is the model server component that runs alongside Vast.ai PyWorker.
PyWorker handles request routing; this server does the actual computation.

Endpoints:
- POST /generate - Standard PoW generation (streaming JSON)
- POST /warmup - Warmup mode with callback polling
- POST /pooled - Orchestrator-managed pooled mode
- GET /health - Health check
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pow.compute.gpu_group import create_gpu_groups, NotEnoughGPUResources
from pow.compute.autobs_v2 import get_batch_size_for_gpu_group
from pow.compute.worker import ParallelWorkerManager, PooledWorkerManager
from pow.compute.model_init import ModelWrapper
from pow.compute.orchestrator_client import OrchestratorClient
from pow.compute.gpu_arch import (
    get_gpu_architecture,
    get_architecture_config,
    GPUArchitecture,
    should_use_fallback_mode,
)
from pow.models.utils import Params, get_params_with_fp8

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_JOB_DURATION = 7 * 60  # 7 minutes
WARMUP_POLL_INTERVAL = 5  # seconds
WARMUP_MAX_DURATION = 10 * 60  # 10 minutes
POOLED_POLL_INTERVAL = 0.5  # 500ms
POOLED_MAX_DURATION = 10 * 60  # 10 minutes
POOLED_MODEL_LOAD_TIMEOUT = 300  # 5 minutes
SERVER_PORT = 18000

# Global state
_cuda_broken = False
_executor = ThreadPoolExecutor(max_workers=4)


def fetch_warmup_params(callback_url: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch generation params from MLNode callback URL."""
    import requests
    try:
        url = f"{callback_url}/warmup/params"
        response = requests.get(url, params={"job_id": job_id}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ready":
            return data.get("params")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch warmup params: {e}")
        return None


def send_batch_to_callback(callback_url: str, batch: Dict[str, Any], node_id: int) -> bool:
    """Send a generated batch to the callback URL."""
    import requests
    try:
        url = f"{callback_url}/generated"
        payload = {
            "public_key": batch.get("public_key", ""),
            "block_hash": batch.get("block_hash", ""),
            "block_height": batch.get("block_height", 0),
            "nonces": batch.get("nonces", []),
            "dist": batch.get("dist", []),
            "node_id": node_id,
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Failed to send batch to callback: {e}")
        return False


def notify_generation_complete(callback_url: str, job_id: str, stats: Dict[str, Any]) -> bool:
    """Notify MLNode that generation is complete."""
    import requests
    try:
        url = f"{callback_url}/warmup/complete"
        payload = {
            "job_id": job_id,
            "total_batches": stats.get("total_batches", 0),
            "total_computed": stats.get("total_computed", 0),
            "total_valid": stats.get("total_valid", 0),
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Notified MLNode of generation complete: {stats}")
        return True
    except Exception as e:
        logger.warning(f"Failed to notify generation complete: {e}")
        return False


def generation_sync(
    input_data: Dict[str, Any],
    send_to_callback: bool = False,
    callback_url: str = "",
    node_id: int = 0,
    warmup_job_id: str = "",
    warmup_callback_url: str = "",
):
    """
    Synchronous PoW generation - yields results as they are computed.
    This runs in a thread pool executor.
    """
    global _cuda_broken

    if _cuda_broken:
        logger.error("CUDA already marked as broken")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    aggregated_batch_count = 0
    total_computed = 0
    total_valid = 0

    try:
        # Get parameters
        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        r_target = input_data["r_target"]
        client_batch_size = input_data["batch_size"]
        start_nonce = input_data["start_nonce"]
        params_dict = input_data["params"]

        params = Params(**params_dict)

        # Auto-detect GPUs
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        # Detect GPU architecture
        recommended_dtype = torch.float16
        try:
            gpu_caps = get_gpu_architecture(0)
            arch_config = get_architecture_config(0)
            recommended_dtype = gpu_caps.recommended_dtype
            logger.info(
                f"GPU Architecture: {gpu_caps.device_name} "
                f"({gpu_caps.architecture.value}, SM{gpu_caps.compute_capability[0]}{gpu_caps.compute_capability[1]})"
            )
            logger.info(
                f"  Memory: {gpu_caps.total_memory_gb:.1f}GB, "
                f"FP8: {gpu_caps.supports_fp8}, BF16: {gpu_caps.supports_bfloat16}, "
                f"dtype: {recommended_dtype}"
            )

            use_fp8 = arch_config.get("use_fp8", False)
            if use_fp8 and gpu_caps.architecture == GPUArchitecture.BLACKWELL:
                logger.info("Blackwell GPU detected - enabling FP8 optimizations")
                params = get_params_with_fp8(params, enable_fp8=True)

            if should_use_fallback_mode():
                logger.warning("Fallback mode enabled - using conservative settings")

        except Exception as e:
            logger.warning(f"Could not detect GPU architecture: {e}, using defaults")

        # Create GPU groups
        gpu_groups = create_gpu_groups(params=params)
        n_workers = len(gpu_groups)

        logger.info(f"Created {n_workers} GPU groups for parallel processing:")
        for i, group in enumerate(gpu_groups):
            logger.info(f"  Worker {i}: {group} (VRAM: {group.get_total_vram_gb():.1f}GB)")

        # Calculate batch size
        batch_sizes = [get_batch_size_for_gpu_group(group, params) for group in gpu_groups]
        batch_size_per_worker = min(batch_sizes)
        total_batch_size = batch_size_per_worker * n_workers

        logger.info(f"START: block={block_height}, workers={n_workers}, "
                   f"batch_per_worker={batch_size_per_worker}, total_batch={total_batch_size}, "
                   f"start={start_nonce}")

        gpu_group_devices = [group.get_device_strings() for group in gpu_groups]

        # Build base model
        logger.info(f"Building base model on GPU 0 with dtype={recommended_dtype}...")
        base_model_start = time.time()
        base_model_data = ModelWrapper.build_base_model(
            hash_=block_hash,
            params=params,
            max_seq_len=params.seq_len,
            dtype=recommended_dtype,
        )
        logger.info(f"Base model built in {time.time() - base_model_start:.1f}s")

        # Create worker manager
        manager = ParallelWorkerManager(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            r_target=r_target,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            start_nonce=start_nonce,
            max_duration=MAX_JOB_DURATION,
            base_model_data=base_model_data,
        )

        manager.start()

        if not manager.wait_for_ready(timeout=180):
            logger.error("Workers failed to initialize within timeout")
            yield {"error": "Worker initialization timeout", "error_type": "TimeoutError"}
            manager.stop()
            return

        logger.info("All workers ready, starting streaming")

        start_time = time.time()
        last_result_time = start_time

        # Streaming results
        while True:
            elapsed = time.time() - start_time

            if elapsed > MAX_JOB_DURATION:
                logger.info(f"TIMEOUT: {elapsed:.0f}s exceeded {MAX_JOB_DURATION}s limit")
                break

            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            results = manager.get_results(timeout=0.5)

            if not results:
                if time.time() - last_result_time > 60:
                    logger.warning("No results for 60s, workers may be stuck")
                continue

            last_result_time = time.time()

            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                aggregated_batch_count += 1
                total_computed += result.get("batch_computed", 0)
                total_valid += result.get("batch_valid", 0)

                result["aggregated_batch_number"] = aggregated_batch_count
                result["aggregated_total_computed"] = total_computed
                result["aggregated_total_valid"] = total_valid
                result["n_workers"] = n_workers
                result["batch_number"] = aggregated_batch_count

                if send_to_callback and callback_url:
                    send_batch_to_callback(callback_url, result, node_id)

                logger.info(f"Batch #{aggregated_batch_count} from worker {result['worker_id']}: "
                           f"{result.get('batch_valid', 0)} valid, elapsed={int(elapsed)}s")

                yield result

        logger.info(f"STOPPED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        manager.stop()

        if warmup_job_id and warmup_callback_url:
            notify_generation_complete(warmup_callback_url, warmup_job_id, {
                "total_batches": aggregated_batch_count,
                "total_computed": total_computed,
                "total_valid": total_valid,
            })

        logger.info("Freeing GPU 0 memory...")
        del base_model_data
        torch.cuda.empty_cache()

    except NotEnoughGPUResources as e:
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        yield {"error": str(e), "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        yield {"error": str(e), "error_type": type(e).__name__}


def pooled_sync(orchestrator_url: str):
    """
    Synchronous pooled mode - managed by external orchestrator.
    This runs in a thread pool executor.
    """
    global _cuda_broken

    if _cuda_broken:
        logger.error("CUDA already marked as broken")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    client = OrchestratorClient(orchestrator_url)
    worker_id = client.worker_id

    logger.info(f"POOLED MODE: worker_id={worker_id}, orchestrator={orchestrator_url}")

    try:
        import torch

        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        if gpu_count == 0:
            yield {"error": "No GPUs detected", "error_type": "NotEnoughGPUResources", "fatal": True}
            return

        yield {"status": "registering", "worker_id": worker_id, "gpu_count": gpu_count}

        reg_response = client.register(gpu_count)
        if reg_response.get("status") == "error":
            yield {"error": "Failed to register with orchestrator", "error_type": "ConnectionError"}
            return

        yield {"status": "registered", "worker_id": worker_id}

        logger.info("Waiting for block_hash from orchestrator...")
        yield {"status": "waiting_block_hash"}

        wait_start = time.time()

        while not client.current_config.block_hash:
            if time.time() - wait_start > POOLED_MAX_DURATION:
                logger.warning("Timeout waiting for block_hash")
                yield {"status": "timeout", "message": "No block_hash received"}
                client.notify_shutdown({"reason": "timeout_waiting_block_hash"})
                return

            config = client.poll_config()
            if config and config.get("type") == "shutdown":
                logger.info("Received shutdown before block_hash")
                yield {"status": "shutdown"}
                return

            time.sleep(1)

        block_hash = client.current_config.block_hash
        params_dict = client.current_config.params
        logger.info(f"Received block_hash: {block_hash[:16]}...")

        yield {"status": "received_block_hash", "block_hash": block_hash[:16]}
        yield {"status": "loading_model"}

        params = Params(**params_dict)
        gpu_groups = create_gpu_groups(params=params)
        n_workers = len(gpu_groups)

        logger.info(f"Created {n_workers} GPU groups")
        for i, group in enumerate(gpu_groups):
            logger.info(f"  Worker {i}: {group} (VRAM: {group.get_total_vram_gb():.1f}GB)")

        batch_sizes = [get_batch_size_for_gpu_group(group, params) for group in gpu_groups]
        batch_size_per_worker = min(batch_sizes)
        logger.info(f"Batch size per worker: {batch_size_per_worker}")

        logger.info("Building base model on GPU 0...")
        model_start = time.time()
        base_model_data = ModelWrapper.build_base_model(
            hash_=block_hash,
            params=params,
            max_seq_len=params.seq_len,
        )
        logger.info(f"Base model built in {time.time() - model_start:.1f}s")

        yield {"status": "model_loaded", "load_time": int(time.time() - model_start)}

        ready_response = client.notify_model_loaded()

        if ready_response.get("status") == "error":
            logger.error("Failed to notify ready")
            yield {"error": "Failed to notify ready", "error_type": "ConnectionError"}
            return

        public_key = client.current_config.public_key
        range_start = client.current_config.nonce_range_start
        range_end = client.current_config.nonce_range_end
        r_target = client.current_config.r_target
        block_height = client.current_config.block_height

        logger.info(f"Ready to compute: public_key={public_key[:16] if public_key else 'None'}..., range={range_start}-{range_end}")

        yield {"status": "ready", "public_key": public_key[:16] if public_key else None}

        gpu_group_devices = [group.get_device_strings() for group in gpu_groups]

        manager = PooledWorkerManager(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            r_target=r_target,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            range_start=range_start,
            range_end=range_end,
            max_duration=POOLED_MAX_DURATION,
            base_model_data=base_model_data,
        )

        manager.start()

        if not manager.wait_for_ready(timeout=POOLED_MODEL_LOAD_TIMEOUT):
            logger.error("Pooled workers failed to initialize")
            yield {"error": "Worker initialization timeout", "error_type": "TimeoutError"}
            manager.stop()
            return

        if public_key:
            manager.set_public_key(public_key)

        logger.info("All pooled workers ready, starting compute loop")
        yield {"status": "computing"}

        session_start = time.time()
        total_batches_sent = 0
        last_poll_time = time.time()

        while True:
            elapsed = time.time() - session_start

            if elapsed > POOLED_MAX_DURATION:
                logger.info(f"Session timeout after {elapsed:.0f}s")
                break

            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            if time.time() - last_poll_time >= POOLED_POLL_INTERVAL:
                last_poll_time = time.time()
                command = client.poll_config()

                if command:
                    cmd_type = command.get("type")

                    if cmd_type == "switch_job":
                        pending_results = manager.get_all_pending_results()
                        for result in pending_results:
                            client.send_result(result)
                            total_batches_sent += 1

                        new_public_key = command.get("public_key")
                        if new_public_key:
                            manager.switch_public_key(new_public_key)
                            logger.info(f"Switched to public_key: {new_public_key[:16]}...")
                            yield {"status": "switched_public_key", "public_key": new_public_key[:16]}

                    elif cmd_type == "shutdown":
                        logger.info("Received shutdown command")
                        break

                    elif cmd_type == "compute" and command.get("public_key"):
                        new_public_key = command.get("public_key")
                        manager.set_public_key(new_public_key)
                        logger.info(f"Set public_key: {new_public_key[:16]}...")

            results = manager.get_results(timeout=0.1)

            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                client.send_result(result)
                total_batches_sent += 1
                yield result

            time.sleep(0.01)

        logger.info(f"Session ended: {total_batches_sent} batches sent, pending={client.get_pending_count()}")

        pending_results = manager.get_all_pending_results()
        for result in pending_results:
            client.send_result(result)
            total_batches_sent += 1

        manager.stop()

        client.notify_shutdown({
            "total_batches_sent": total_batches_sent,
            "session_duration": int(time.time() - session_start),
        })

        yield {
            "status": "shutdown",
            "total_batches_sent": total_batches_sent,
            "session_duration": int(time.time() - session_start),
        }

        del base_model_data
        torch.cuda.empty_cache()

    except NotEnoughGPUResources as e:
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        yield {"error": str(e), "error_type": "NotEnoughGPUResources", "fatal": True}
        client.notify_shutdown({"error": str(e)})
        os._exit(1)

    except Exception as e:
        logger.error(f"POOLED ERROR: {str(e)}", exc_info=True)
        yield {"error": str(e), "error_type": type(e).__name__}
        try:
            client.notify_shutdown({"error": str(e)})
        except:
            pass

    finally:
        try:
            client.close()
        except:
            pass


# ============================================================================
# HTTP Handlers
# ============================================================================

async def health_handler(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "healthy", "cuda_broken": _cuda_broken})


async def generate_handler(request: web.Request) -> web.StreamResponse:
    """
    POST /generate - Standard PoW generation with streaming response.

    Request body:
    {
        "block_hash": str,
        "block_height": int,
        "public_key": str,
        "r_target": float,
        "batch_size": int,
        "start_nonce": int,
        "params": dict
    }
    """
    try:
        input_data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    # Validate required fields
    required = ["block_hash", "block_height", "public_key", "r_target", "batch_size", "start_nonce", "params"]
    missing = [f for f in required if f not in input_data]
    if missing:
        return web.json_response({"error": f"Missing required fields: {missing}"}, status=400)

    logger.info(f"Generate request: block_height={input_data['block_height']}, public_key={input_data['public_key'][:16]}...")

    # Streaming response
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'application/x-ndjson'}
    )
    await response.prepare(request)

    loop = asyncio.get_event_loop()

    def run_generation():
        return list(generation_sync(input_data))

    try:
        # Run in thread pool to avoid blocking event loop
        for result in await loop.run_in_executor(_executor, run_generation):
            await response.write(json.dumps(result).encode() + b'\n')
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        await response.write(json.dumps({"error": str(e)}).encode() + b'\n')

    await response.write_eof()
    return response


async def warmup_handler(request: web.Request) -> web.StreamResponse:
    """
    POST /warmup - Warmup mode with callback polling.

    Request body:
    {
        "callback_url": str,
        "job_id": str (optional)
    }
    """
    try:
        input_data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    callback_url = input_data.get("callback_url", "")
    job_id = input_data.get("job_id", "unknown")

    if not callback_url:
        return web.json_response({"error": "callback_url is required"}, status=400)

    logger.info(f"WARMUP MODE: job_id={job_id}, callback_url={callback_url}")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'application/x-ndjson'}
    )
    await response.prepare(request)

    await response.write(json.dumps({"status": "warmup_ready", "job_id": job_id}).encode() + b'\n')

    # Poll for params
    warmup_start = time.time()
    poll_count = 0

    while True:
        elapsed = time.time() - warmup_start

        if elapsed > WARMUP_MAX_DURATION:
            logger.info(f"WARMUP TIMEOUT: {elapsed:.0f}s exceeded {WARMUP_MAX_DURATION}s limit")
            await response.write(json.dumps({"status": "warmup_timeout", "elapsed": int(elapsed)}).encode() + b'\n')
            break

        poll_count += 1
        if poll_count % 12 == 1:
            logger.info(f"Warmup poll #{poll_count}: waiting for params... ({int(elapsed)}s)")

        params = await asyncio.get_event_loop().run_in_executor(
            _executor, fetch_warmup_params, callback_url, job_id
        )

        if params:
            logger.info(f"Warmup got params after {elapsed:.0f}s, starting generation")
            await response.write(json.dumps({"status": "warmup_params_received", "elapsed": int(elapsed)}).encode() + b'\n')

            # Run generation
            loop = asyncio.get_event_loop()

            def run_gen():
                return list(generation_sync(
                    params,
                    send_to_callback=False,
                    warmup_job_id=job_id,
                    warmup_callback_url=callback_url,
                ))

            for result in await loop.run_in_executor(_executor, run_gen):
                await response.write(json.dumps(result).encode() + b'\n')

            break

        await response.write(json.dumps({
            "status": "warmup_waiting",
            "poll_count": poll_count,
            "elapsed": int(elapsed),
        }).encode() + b'\n')

        await asyncio.sleep(WARMUP_POLL_INTERVAL)

    await response.write_eof()
    return response


async def pooled_handler(request: web.Request) -> web.StreamResponse:
    """
    POST /pooled - Orchestrator-managed pooled mode.

    Request body:
    {
        "orchestrator_url": str
    }
    """
    try:
        input_data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    orchestrator_url = input_data.get("orchestrator_url", "")

    if not orchestrator_url:
        return web.json_response({"error": "orchestrator_url is required"}, status=400)

    logger.info(f"POOLED MODE: orchestrator_url={orchestrator_url}")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'application/x-ndjson'}
    )
    await response.prepare(request)

    loop = asyncio.get_event_loop()

    def run_pooled():
        return list(pooled_sync(orchestrator_url))

    try:
        for result in await loop.run_in_executor(_executor, run_pooled):
            await response.write(json.dumps(result).encode() + b'\n')
    except Exception as e:
        logger.error(f"Pooled error: {e}", exc_info=True)
        await response.write(json.dumps({"error": str(e)}).encode() + b'\n')

    await response.write_eof()
    return response


# ============================================================================
# Application Setup
# ============================================================================

def create_app() -> web.Application:
    """Create aiohttp application."""
    app = web.Application()

    app.router.add_get('/health', health_handler)
    app.router.add_post('/generate', generate_handler)
    app.router.add_post('/warmup', warmup_handler)
    app.router.add_post('/pooled', pooled_handler)

    return app


def main():
    """Main entry point."""
    logger.info(f"Starting Vast.ai Model Server on port {SERVER_PORT}")
    logger.info("Server ready")  # PyWorker looks for this in logs

    app = create_app()
    web.run_app(app, host='0.0.0.0', port=SERVER_PORT)


if __name__ == '__main__':
    main()
