# LLaMA GPU Benchmark

GPU-accelerated LLaMA inference benchmark for Vast.ai Serverless platform.

## Architecture

This project uses Vast.ai's PyWorker architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Vast.ai Platform                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐          ┌─────────────────────────────┐  │
│  │  PyWorker   │  ──────► │     Model Server            │  │
│  │ (worker.py) │  proxy   │   (model_server.py)         │  │
│  │             │          │                             │  │
│  │ - Routing   │          │ - /generate (inference)     │  │
│  │ - Health    │          │ - /warmup (preload)         │  │
│  │ - Benchmark │          │ - /pooled (batch mode)      │  │
│  └─────────────┘          │ - /health                   │  │
│                           └─────────────────────────────┘  │
│                                      │                      │
│                           ┌──────────▼──────────┐          │
│                           │   GPU Workers       │          │
│                           │  (src/pow/...)      │          │
│                           └─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `worker.py` | Vast.ai PyWorker configuration |
| `model_server.py` | aiohttp HTTP server for GPU inference |
| `start_server.sh` | Model server startup script |
| `Dockerfile` | Container image definition |
| `requirements.txt` | Python dependencies |
| `src/` | Core inference code |

## Deployment

### 1. Build Docker Image

The image is automatically built via GitHub Actions on push to main:

```bash
# Image will be available at:
ghcr.io/<username>/llama-benchmark:latest
```

### 2. Configure Vast.ai Endpoint

1. Go to Vast.ai Serverless Dashboard
2. Create new endpoint
3. Set Docker image: `ghcr.io/<username>/llama-benchmark:latest`
4. Configure environment variable: `PYWORKER_REPO=https://github.com/<username>/llama-benchmark`
5. Set GPU requirements (min 1 GPU, 24GB+ VRAM recommended)

### 3. Scaling Configuration

| Setting | Recommended | Description |
|---------|-------------|-------------|
| Min Workers | 1 | Pre-loaded instances |
| Max Workers | 16 | Upper limit |
| Cold Multiplier | 3 | Scale by load prediction |
| Target Utilization | 0.9 | Resource usage target |

## API Endpoints

### POST /generate

Run LLaMA inference benchmark with streaming response.

```json
{
  "block_hash": "seed123...",
  "block_height": 12345,
  "public_key": "client_id...",
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
    "rope_theta": 500000.0
  }
}
```

Response (streaming NDJSON):

```json
{"nonces": [42, 156], "dist": [1.23, 1.31], "batch_number": 1, ...}
{"nonces": [789], "dist": [1.18], "batch_number": 2, ...}
```

### POST /warmup

Preload model and wait for benchmark params via callback.

```json
{
  "callback_url": "http://controller:9090",
  "job_id": "job123"
}
```

### POST /pooled

Batch processing mode for distributed benchmarking.

```json
{
  "orchestrator_url": "https://controller.example.com"
}
```

### GET /health

Health check endpoint.

```json
{"status": "healthy", "cuda_broken": false}
```

## GPU Support

- **Blackwell (B100, B200)**: Full support with FP8 optimizations
- **Hopper (H100, H200)**: Full support
- **Ampere (A100)**: Full support

Requires CUDA 12.8+ for Blackwell GPUs.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | PyTorch memory allocator config |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start model server
python model_server.py

# In another terminal, test health
curl http://localhost:18000/health
```

## License

MIT
