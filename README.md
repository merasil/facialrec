# FacialRec (GPU-accelerated) – Docker Compose Setup

This repository runs a GPU-accelerated face recognition service with Docker.
It’s tested on Linux with NVIDIA GPUs and Docker Engine + Docker Compose.

---

## ✨ Features

- GPU acceleration via NVIDIA Container Toolkit (CDI)
- Docker Compose service: `facialrec`
- RTSP over TCP for OpenCV (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`)
- Persistent volumes for config, DB, and DeepFace weights
- Live mode plus sample creation, model benchmarks, and VRAM measurements

---

## ☑️ Requirements

- **NVIDIA GPU + Driver** installed on the host
- **Docker Engine** (v25+ recommended)
- **Docker Compose** (V2)
- **NVIDIA Container Toolkit** (`nvidia-container-toolkit`)

> **Why CDI (Container Device Interface)?**  
> CDI avoids the legacy *prestart hook* that can crash with errors like  
> `nvidia-container-cli: ldcache error ... ldconfig ...` and is the recommended path going forward.

---

## 🧰 Install & Configure NVIDIA Container Toolkit (CDI mode)

> The steps below switch the toolkit to **CDI** and are **reboot-safe**.  
> You only need to regenerate the CDI spec after **driver/MIG changes**.

### Arch Linux (quick path)
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
```

## 🧪 Quickstart (Compose)

### Option A — **CDI devices (recommended)**
This avoids the legacy hook entirely.

```yaml
services:
  facialrec:
    build: .
    environment:
      - OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp
    volumes:
      - ./config:/app/config
      - ./db:/app/db
      - ./output:/app/output
      - ./weights:/root/.deepface/weights
    # Direct CDI assignment also works with `docker compose run`
    devices:
      - "nvidia.com/gpu=all"   # or "nvidia.com/gpu=0", "nvidia.com/gpu=1", ...
    restart: unless-stopped
```
Note: CDI devices in Compose require a recent Docker Engine with CDI enabled.
If your Compose implementation rejects the CDI device name, use Option B.
The direct service-level `devices` entry is inherited by one-off
`docker compose run` containers. A `deploy.resources` reservation alone may be
ignored for these containers by some Compose versions.

### Option B — **Compose with NVIDIA driver (works with CDI or legacy)**
```yaml
services:
  facialrec:
    build: .
    environment:
      - OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp
    volumes:
      - ./config:/app/config
      - ./db:/app/db
      - ./output:/app/output
      - ./weights:/root/.deepface/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all            # or set a number; or use device_ids: ['0','3']
              capabilities: [gpu]
    restart: unless-stopped
```
Caveat: Using the --gpus path may still engage the legacy hook on some setups.
If you ever see legacy-hook errors, prefer Option A (CDI devices).

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
After=network-online.target nss-lookup.target docker.socket firewalld.service containerd.service time-set.target nvidia-persistenced.service systemd-udev-settle.service
Wants=network-online.target containerd.service nvidia-persistenced.service

# Optional but more robust
[Service]
# Ensure CDI spec exists (idempotent) and wait for device nodes
ExecStartPre=/usr/bin/nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
ExecStartPre=/usr/bin/bash -c 'for i in {1..20}; do [ -e /dev/nvidia0 ] && [ -e /dev/nvidia-uvm ] && break; sleep 1; done; [ -e /dev/nvidia0 ] && [ -e /dev/nvidia-uvm ]'
```
Reload and restart Docker:
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```
Result: When the host reboots, Docker starts only after NVIDIA is ready; containers with restart: unless-stopped then come up automatically.

## ▶️ Run

```bash
docker compose up -d --build
```
On first start, DeepFace may download model weights into /root/.deepface/weights (mounted from ./weights).

## Command modes

Without a subcommand, `main.py` starts the normal live recognition mode:

```bash
python3 main.py
python3 main.py --detector retinaface --recognizer Facenet512
```

CLI options override values from `config/config.ini`. A different file can be
selected with `--config`.

### Create test samples

Record a short video and save every fifth frame as a JPEG:

```bash
python3 main.py create-test-samples --duration 10 --frame-step 5
```

Keep only selected frames containing a detected face and retain the source MP4:

```bash
python3 main.py create-test-samples \
  --face-only \
  --keep-video \
  --detector retinaface
```

Files are written below `output/<unix-timestamp>/`. JPEG names are sequential,
starting with `000001.jpg`.

### Benchmark models

Benchmark every detector, recognizer, and metric combination against an image
folder or video:

```bash
python3 main.py benchmark \
  --input test-data \
  --name Alice \
  --db db \
  --detectors retinaface mtcnn \
  --recognizers Facenet512 ArcFace \
  --metric cosine euclidean euclidean_l2 \
  --output output/benchmark.txt
```

Only images directly inside an input directory are processed. Videos are
processed frame by frame. Each combination is loaded and run once before timing
starts. The table reports processed inputs, detections, correct and incorrect
recognitions, unknown faces, errors, total time, and average time per input.
Recognition percentage is calculated from detected inputs.
Timing excludes model loading, warm-up, and video decoding.
Every combination runs in a separate process so previously loaded TensorFlow
and PyTorch models cannot distort later timing results. A native crash or OOM
kill only fails that combination, and the benchmark still writes the result
table. Workers run with Python's faulthandler enabled. Their structured phase
and result messages use stdout, while native diagnostics and a `BENCH_DIAG`
line with Python, platform, CUDA environment, and package versions are written
directly to stderr.
PyTorch-backed detectors such as YOLO and FastMTCNN import their complete
runtime before DeepFace can initialize TensorFlow. They then load the detector,
configure TensorFlow, and load the recognizer. TensorFlow detectors such as
RetinaFace configure TensorFlow before loading the detector and recognizer.
If a worker is terminated, the status includes the signal and last phase.
`SIGKILL` also includes a possible RAM/VRAM OOM hint.
The `Status` column reports model setup failures such as missing optional
packages or invalid detector names.

Some DeepFace detectors require optional dependencies:

- `fastmtcnn`: `facenet-pytorch`
- `dlib`: `dlib`
- `mediapipe`: `mediapipe`
- `yolov8n`, `yolov8m`, `yolov8l` and newer YOLO variants: `ultralytics`

Use the exact YOLO backend name. `yolov8` alone is not a valid backend in
current DeepFace releases.

### Measure VRAM

Measure detector and recognizer combinations on one representative input frame:

```bash
python3 main.py vram \
  --input test-data \
  --db db \
  --detectors retinaface mtcnn \
  --recognizers Facenet512 ArcFace \
  --runs 3 \
  --gpu 0 \
  --output output/vram.txt
```

Each combination runs in a separate process so TensorFlow and CUDA memory is
released between measurements. TensorFlow columns show allocator memory.
NVML columns show the complete process allocation, including CUDA context and
reserved memory. Values are reported in MiB.
Model loading, datastore creation, and one full inference warm-up happen before
peak measurement starts. This prevents missing DeepFace cache files from
inflating one recognizer's result.
PyTorch-backed detectors are initialized before TensorFlow configures its GPU
and before the TensorFlow recognizer is built. Live, benchmark, and VRAM modes
use this same order. If a worker exits natively, the status column reports its
exit code and the active phase, for example `detector loading` or
`recognizer loading`.

Stop the live service before GPU benchmarks. `docker compose run` starts an
additional container:

```bash
docker compose stop facialrec
docker compose run --rm facialrec python3 main.py benchmark ...
docker compose start facialrec
```

## 🔎 Verifications & Tips

- **Autostart**: Ensure the container exists and has a restart policy:
  ```bash
  docker compose ps
  docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' $(docker compose ps -q facialrec)
  ```
- **Regenerate CDI spec after driver updates/MIG changes:**
  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```
- **Select specific GPUs:**
  - CDI: set ```device_ids``` to ```"nvidia.com/gpu=0"```, ```"nvidia.com/gpu=1"```, …
  - NVIDIA driver path: use ```device_ids: ['0','3']``` or ```count: 1```.

**Rootless Docker (heads-up)**
If you run Docker rootless, ensure the daemon can read CDI specs.
Either keep specs in default locations (```/etc/cdi```, ```/var/run/cdi```) supported by your version, or set ```"cdi-spec-dirs"``` in your rootless daemon config.

## 🧯 Troubleshooting

- **Error**: `CDI device injection failed: failed to stat "/dev/nvidia-modeset": no such file or directory`  
  **Fix**: Enable persistence, use the Docker drop-in above, and ensure udev creates `/dev/nvidia*`.

- **Legacy hook crash**: `nvidia-container-cli: ldcache error ... ldconfig ...`  
  **Fix**: Ensure CDI is enabled + generate the CDI spec, then use CDI devices in Compose (Option A).

## 📂 Project volumes

- `./config` → `/app/config`  
- `./db` → `/app/db`  
- `./output` → `/app/output`
- `./weights` → `/root/.deepface/weights` (DeepFace model cache)
