# Demucs API (FastAPI) for Google Cloud

This repo contains a minimal FastAPI service wrapping Demucs for music source separation. It supports CPU and GPU deployments and returns a ZIP of separated stems.


## Endpoints
입력 : 오디오 파일, 악기 이름, 시작 시간, 종료 시간
출력 : 유사한 음악 리스트 top_k = 5/ 각 항목은 다음과 같은 딕셔너리 형태
            "status": "success",
            "results": 
                        {
                            "id": result_id : int
                            "distance": distance : float,
                            "song_name": song_name : str,
                            "instrument": instrument : str,
                            "start_sec": start_sec: float,
                            "end_sec": end_sec: float
                        }

## Local run (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Docker (CPU)
```powershell
# Build
docker build -t demucs-api:cpu -f Dockerfile .
# Run
docker run --rm -p 8080:8080 demucs-api:cpu
```

## Docker (GPU)
Requires an NVIDIA GPU host with the NVIDIA Container Toolkit installed.
- Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

```powershell
# Build
docker build -t demucs-api:gpu -f Dockerfile.gpu .
# Run with GPU access
docker run --rm -p 8080:8080 --gpus all demucs-api:gpu
```

## Google Cloud deployment options

### Option A: Compute Engine (recommended for GPU)
1. Create a VM
   - For CPU: use an `e2-standard-4` or larger.
   - For GPU: choose a GPU machine (e.g., `n1-standard-8` + `T4` / `L4` / `A100`).
   - OS: Ubuntu 22.04.
2. Install Docker
   - https://docs.docker.com/engine/install/ubuntu/
3. For GPU VMs, install NVIDIA drivers and NVIDIA Container Toolkit
   - Drivers: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
   - Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
4. Copy project files to VM (e.g., `gcloud compute scp` or Git clone).
5. Build & run
   - CPU: `docker build -t demucs-api:cpu -f Dockerfile . && docker run -d -p 8080:8080 demucs-api:cpu`
   - GPU: `docker build -t demucs-api:gpu -f Dockerfile.gpu . && docker run -d -p 8080:8080 --gpus all demucs-api:gpu`
6. Open firewall rule for TCP:8080 or use a load balancer.

### Option B: Cloud Run (CPU)
- Build container: `gcloud builds submit --tag gcr.io/PROJECT_ID/demucs-api:cpu`
- Deploy: `gcloud run deploy demucs-api --image gcr.io/PROJECT_ID/demucs-api:cpu --platform managed --region REGION --allow-unauthenticated --cpu 2 --memory 4Gi`
- Note: GPU support on Cloud Run may be limited; prefer Compute Engine / GKE for GPU.

### Option C: GKE (GPU/CPU)
- Create a GKE cluster with GPU node pools.
- Install NVIDIA drivers for GKE (use GPU node images).
- Deploy as a Kubernetes Deployment + Service, with `resources.limits.nvidia.com/gpu: 1` for GPU pods.

## API usage example
```bash
curl -X POST \
  -F "file=@your_song.mp3" \
  -F "model=htdemucs" \
  -F "device=auto" \
  -F "mp3=true" \
  http://localhost:8080/separate --output out.zip
```

## Notes
- First run downloads models. Cache will persist in the container layer or volume if configured.
- For large files, consider increasing request body size or using GCS for input/output storage.
- For production, run Uvicorn with multiple workers (e.g., `--workers 2`) and set CPU/memory accordingly.
