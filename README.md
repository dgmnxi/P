# Demucs API (FastAPI) for Google Cloud

This repo contains a minimal FastAPI service wrapping Demucs for music source separation. It supports CPU and GPU deployments 


## Endpoints(/recommend)
입력 : youtube_url(str),<br>
      instrument(str),<br>
      start_sec(float),<br>
      end_sec(float)<br>
      top_k (int)[Optional : Default : 5] <br>
      


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


## Notes
- First run downloads models. Cache will persist in the container layer or volume if configured.
- For large files, consider increasing request body size or using GCS for input/output storage.
- For production, run Uvicorn with multiple workers (e.g., `--workers 2`) and set CPU/memory accordingly.
