# FacialRec - Docker Compose Setup

This repository runs a face recognition service with Docker. CPU operation is
the default; NVIDIA GPU acceleration is optional.

---

## ✨ Features

- CPU operation without NVIDIA host dependencies
- Optional GPU acceleration via NVIDIA Container Toolkit (CDI)
- Docker Compose service: `facialrec`
- RTSP over TCP for OpenCV (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`)
- Persistent volumes for config, DB, and DeepFace weights

---

## ☑️ Requirements

- **Docker Engine**
- **Docker Compose** (V2)

GPU acceleration additionally requires an NVIDIA GPU and driver, a recent
Docker Engine with CDI support, and `nvidia-container-toolkit`.

---

## Run on CPU

The base Compose file does not request a GPU:

```bash
docker compose up -d --build
```

At container startup, `test.py --base` runs before `main.py`. The service only
starts when the config, database images, configured RTSP streams, motion source,
detector, recognizer, and distance metric pass the preflight.

The CUDA-capable TensorFlow and PyTorch packages in the image fall back to CPU
execution when no GPU is exposed. For better CPU performance, start with a
lighter detector such as `opencv` or `ssd`:

```ini
[face_recognition]
detector_model = opencv
```

RetinaFace and YOLO can also run on the CPU, but may be significantly slower.

## Run with an NVIDIA GPU

The GPU override requests all GPUs through CDI and runs `--base --gpu` before
starting the service:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up -d --build
```

If no GPU is available in this mode, startup fails instead of silently falling
back to the CPU. Use the same file combination for `down`, `run`, and `logs`.

## Runtime tests

All runtime checks are available through `test.py`. Run them in the active
container with Compose:

```bash
docker compose exec facialrec python3 test.py --base
docker compose exec facialrec python3 test.py --gpu
docker compose exec facialrec python3 test.py --motion
docker compose exec facialrec python3 test.py --base --gpu --motion
```

If the normal service cannot stay up because its startup preflight fails, run
the test in a one-off container instead:

```bash
docker compose run --rm facialrec python3 test.py --base
```

The equivalent plain Docker form is:

```bash
docker exec <container-name> python3 test.py --gpu --motion
```

Available checks:

- `--base`: config schema and ranges, database images, main and optional
  low-resolution RTSP streams, configured internal or external motion source,
  face detector, recognition model, distance metric, and one sample inference
  on a database image
- `--gpu`: `nvidia-smi`, TensorFlow GPU discovery and operation, and PyTorch
  CUDA discovery and operation
- `--motion`: exercises internal motion processing or polls the external motion
  endpoint for ten seconds
- `--all`: runs all checks
- `--start`: starts `main.py` after all selected checks pass

Useful timeout options are `--timeout`, `--gpu-timeout`, and
`--motion-duration`. Running `python3 test.py` without a check option defaults
to `--base`.

---

## 🧰 Install & Configure NVIDIA Container Toolkit (CDI mode)

> NVIDIA Container Toolkit 1.18 and newer maintain the CDI specification with
> `nvidia-cdi-refresh`.

### Arch Linux (quick path)
```bash
# 1) Install toolkit (driver/utils should already be present)
sudo pacman -S --needed nvidia-container-toolkit

# 2) Configure Docker and enable automatic CDI refresh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now nvidia-cdi-refresh.path
sudo systemctl restart nvidia-cdi-refresh.service
sudo systemctl restart docker

# 3) Verify CDI and the container GPU
nvidia-ctk --debug cdi list
docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 nvidia-smi -L
```

### After replacing a GPU

CDI specifications contain device information from the installed GPU. Refresh
them before restarting this service. If an older setup created
`/etc/cdi/nvidia.yaml`, disable it so it cannot conflict with the automatically
managed `/var/run/cdi/nvidia.yaml`. Also remove an old
`ExecStartPre=...nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` line
from `systemctl edit docker.service` if it was added using an earlier version of
this guide.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
[ ! -f /etc/cdi/nvidia.yaml ] || sudo mv /etc/cdi/nvidia.yaml /etc/cdi/nvidia.yaml.disabled
sudo systemctl restart nvidia-cdi-refresh.service
sudo systemctl restart docker

nvidia-ctk --debug cdi list
docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --pull --no-cache
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm facialrec python3 test.py --gpu
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=200 facialrec
```

The diagnostic streams each framework step directly and stops a framework
check after 300 seconds. Override that limit when investigating a slow first
PTX compilation:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm \
  -e GPU_DIAGNOSTICS_TIMEOUT=900 \
  facialrec python3 test.py --gpu
```

The RTX 5060 Ti has CUDA compute capability 12.0 (`sm_120`). The container pins
the PyTorch CUDA 12.8 build so YOLO kernels support this Blackwell GPU. Do not
add the current `facenet-pytorch` package to this image: it constrains PyTorch
to 2.3 or older, which predates RTX 50-series support. The `fastmtcnn` detector
is therefore not available in this Blackwell-compatible image.

## 🧷 Boot-safe autostart (fix race conditions)

On some systems Docker may start **before** NVIDIA persistence and CDI are fully ready, which can prevent GPU containers from starting automatically after a reboot. The following makes autostart reliable **without** creating a custom unit for this project.

### 1) Enable NVIDIA Persistence Daemon
```bash
sudo systemctl enable --now nvidia-persistenced.service
```

### 2) Add a Docker service drop-in (delay Docker until NVIDIA is ready)
```bash
sudo systemctl edit docker.service
```
Insert the snippet below, save, and exit (this creates a drop-in under
/etc/systemd/system/docker.service.d/override.conf):
```ini
[Unit]
# Start Docker only after NVIDIA persistence and udev have settled
After=network-online.target nss-lookup.target docker.socket firewalld.service containerd.service time-set.target nvidia-persistenced.service nvidia-cdi-refresh.service systemd-udev-settle.service
Wants=network-online.target containerd.service nvidia-persistenced.service nvidia-cdi-refresh.service

# Optional but more robust
[Service]
# Wait for device nodes; nvidia-cdi-refresh manages the CDI specification
ExecStartPre=/usr/bin/bash -c 'for i in {1..20}; do [ -e /dev/nvidia0 ] && [ -e /dev/nvidia-uvm ] && break; sleep 1; done; [ -e /dev/nvidia0 ] && [ -e /dev/nvidia-uvm ]'
```
Reload and restart Docker:
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```
Result: When the host reboots, Docker starts only after NVIDIA is ready; containers with restart: unless-stopped then come up automatically.

On first start, DeepFace may download model weights into
`/root/.deepface/weights`, which is mounted from `./weights`.

To use a YOLO detector, set its exact DeepFace backend name in
`config/config.ini`, for example:

```ini
[face_recognition]
detector_model = yolov8m
```

Rebuild the image after updating from a version without YOLO support so the
`ultralytics` dependency is installed. PyTorch-backed YOLO detectors load their
runtime before DeepFace can initialize TensorFlow. TensorFlow detectors such as
RetinaFace keep the opposite order: TensorFlow configuration, detector, then
recognizer.

## 🔎 Verifications & Tips

- **Autostart**: Ensure the container exists and has a restart policy:
  ```bash
  docker compose ps
  docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' $(docker compose ps -q facialrec)
  ```
- **Regenerate CDI spec after driver updates, MIG changes, or replacing a GPU:**
  ```bash
  sudo systemctl restart nvidia-cdi-refresh.service
  nvidia-ctk --debug cdi list
  ```
- **Select specific GPUs:**
  Change `device_ids` in `docker-compose.gpu.yml` to
  `"nvidia.com/gpu=0"`, `"nvidia.com/gpu=1"`, and so on.

**Rootless Docker (heads-up)**
If you run Docker rootless, ensure the daemon can read CDI specs.
Either keep specs in default locations (```/etc/cdi```, ```/var/run/cdi```) supported by your version, or set ```"cdi-spec-dirs"``` in your rootless daemon config.

## 🧯 Troubleshooting

- **Error**: `CDI device injection failed: failed to stat "/dev/nvidia-modeset": no such file or directory`  
  **Fix**: Enable persistence, use the Docker drop-in above, and ensure udev creates `/dev/nvidia*`.

- **Legacy hook crash**: `nvidia-container-cli: ldcache error ... ldconfig ...`  
  **Fix**: Ensure CDI is enabled, regenerate the CDI spec, and use
  `docker-compose.gpu.yml`.

- **PyTorch error**: `sm_120 is not compatible` or `no kernel image is available`
  **Fix**: Rebuild without cache so the CUDA 12.8 PyTorch wheels are installed:
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --pull --no-cache`.

- **TensorFlow reports no GPU**: Run
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm facialrec python3 test.py --gpu`.
  If
  `nvidia-smi` fails there, repair CDI/container-toolkit access first. If only
  TensorFlow or PyTorch fails, rebuild the image without cache and inspect the
  framework versions printed by the diagnostic.

## 📂 Project volumes

- `./config` → `/app/config`  
- `./db` → `/app/db`  
- `./weights` → `/root/.deepface/weights` (DeepFace model cache)
