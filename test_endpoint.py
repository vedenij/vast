#!/usr/bin/env python3
"""
Test script for Vast.ai Serverless endpoint.

Usage:
    python test_endpoint.py
"""

import requests
import json

# Configuration
VAST_API_KEY = "f497aeae82e86426c7c20215b7c9be77d2fbe0aaba69045a3d0d281771ee979d"
ENDPOINT_ID = "87aa0um1"
ROUTE_URL = "https://run.vast.ai/route/"

# Test payload
PAYLOAD = {
    "block_hash": "EFA29D99D6767DCCEC77B03A476CD7FF94345578D10B233EF249D852F750ED3A",
    "block_height": 1350372,
    "public_key": "03b3a75b8dcebdb10a86f0787d911d93441c827f553a07e1af70e97aa21e60ea43",
    "batch_size": 100,
    "start_nonce": 0,
    "r_target": 1.4013564660458173,
    "params": {
        "dim": 1792,
        "n_layers": 64,
        "n_heads": 64,
        "n_kv_heads": 64,
        "vocab_size": 8196,
        "ffn_dim_multiplier": 10.0,
        "multiple_of": 8192,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "use_scaled_rope": False,
        "seq_len": 256
    }
}


def get_worker_url():
    """Step 1: Get worker URL from Vast.ai route endpoint."""
    print(f"[1] Getting worker URL from {ROUTE_URL}...")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VAST_API_KEY}"
    }

    payload = {
        "endpoint": ENDPOINT_ID,  # Can be name or ID
        "cost": 100
    }

    print(f"    Request: {payload}")

    response = requests.post(
        ROUTE_URL,
        headers=headers,
        json=payload,
        timeout=30
    )

    print(f"    Response status: {response.status_code}")
    print(f"    Response body: {response.text}")

    if response.status_code != 200:
        return None

    data = response.json()
    print(f"    Worker URL: {data.get('url')}")
    print(f"    Request #: {data.get('reqnum')}")
    return data.get("url")


def test_health(worker_url):
    """Test health endpoint."""
    print(f"\n[2] Testing health at {worker_url}/health...")

    try:
        response = requests.get(f"{worker_url}/health", timeout=10)
        print(f"    Status: {response.status_code}")
        print(f"    Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_generate(worker_url):
    """Test generate endpoint with streaming."""
    print(f"\n[3] Testing generate at {worker_url}/generate...")
    print(f"    Payload: block_height={PAYLOAD['block_height']}, batch_size={PAYLOAD['batch_size']}")

    try:
        response = requests.post(
            f"{worker_url}/generate",
            json=PAYLOAD,
            stream=True,
            timeout=300  # 5 min timeout
        )

        print(f"    Status: {response.status_code}")
        print("\n[4] Receiving batches:\n")

        batch_count = 0
        total_valid = 0

        for line in response.iter_lines():
            if line:
                try:
                    batch = json.loads(line.decode())
                    batch_count += 1

                    if "error" in batch:
                        print(f"    ERROR: {batch['error']}")
                        break

                    nonces = batch.get("nonces", [])
                    valid_count = len(nonces)
                    total_valid += valid_count

                    print(f"    Batch #{batch.get('batch_number', batch_count)}: "
                          f"{valid_count} valid nonces, "
                          f"computed={batch.get('batch_computed', 'N/A')}, "
                          f"elapsed={batch.get('elapsed_seconds', 'N/A')}s")

                    if nonces:
                        print(f"      Nonces: {nonces[:5]}{'...' if len(nonces) > 5 else ''}")

                except json.JSONDecodeError as e:
                    print(f"    Failed to parse: {line[:100]}")

        print(f"\n[5] Summary: {batch_count} batches, {total_valid} total valid nonces")

    except requests.exceptions.Timeout:
        print("    Timeout waiting for response")
    except Exception as e:
        print(f"    Error: {e}")


def main():
    print("=" * 60)
    print("Vast.ai Serverless Endpoint Test")
    print("=" * 60)
    print(f"Endpoint ID: {ENDPOINT_ID}")
    print()

    # Step 1: Get worker URL
    worker_url = get_worker_url()
    if not worker_url:
        print("\nFailed to get worker URL. Is the endpoint running?")
        return

    # Step 2: Health check
    if not test_health(worker_url):
        print("\nHealth check failed. Worker may not be ready.")
        # Continue anyway to see error

    # Step 3: Test generate
    test_generate(worker_url)

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
