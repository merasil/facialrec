# FacialRec (GPU-accelerated) – Docker Compose Setup

This repository runs a GPU-accelerated face recognition service with Docker.  
It’s tested on Linux with NVIDIA GPUs and Docker Engine + Docker Compose.

---

## ✨ Features

- GPU acceleration via NVIDIA Container Toolkit
- Docker Compose service: `facialrec`
- RTSP over TCP for OpenCV (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`)
- Persistent volumes for config, DB, and DeepFace weights

---

## ☑️ Requirements

- **NVIDIA GPU + Driver** installed on the host
- **Docker Engine** (v25+ recommended; v27/28+ works great)
- **Docker Compose** (V2)
- **NVIDIA Container Toolkit** (`nvidia-container-toolkit`)

> Why CDI (Container Device Interface)?  
> CDI avoids the legacy *prestart hook* that can crash with errors like  
> `nvidia-container-cli: ldcache error ... ldconfig ...` and is the recommended path going forward.

---

## 🧰 Install & Configure NVIDIA Container Toolkit (CDI mode)

> The steps below switch the toolkit to **CDI** and are **reboot-safe**.  
> You only need to regenerate the CDI spec after **driver/MIG changes**.

### Arch Linux (recommended quick path)
```bash
# 1) Install toolkit (driver/utils should already be present)
sudo pacman -S --needed nvidia-container-toolkit

# 2) Enable CDI in Docker and restart Docker
sudo nvidia-ctk runtime configure --runtime=docker --cdi.enabled
sudo systemctl restart docker

# 3) Force the NVIDIA runtime into CDI mode
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi

# 4) Generate the CDI spec (persistent)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 5) Smoke test (should print your GPUs)
docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 nvidia-smi -L
