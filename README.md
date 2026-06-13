# FacialRec

FacialRec is a Docker-based face recognition service for RTSP camera streams.
It supports CPU operation by default and optional NVIDIA GPU acceleration.

## Features

- Face detection and recognition through DeepFace
- Configurable detector, recognition model, and distance metric
- RTSP streams over TCP
- Internal motion detection or an external motion endpoint
- CPU operation without NVIDIA host dependencies
- Optional NVIDIA GPU acceleration through Docker CDI
- Startup checks for configuration, streams, models, and motion detection
- Persistent configuration, face database, and model weights

## Requirements

For CPU operation:

- Docker Engine
- Docker Compose V2
- Access to the configured RTSP stream

GPU operation additionally requires:

- A supported NVIDIA GPU and driver
- NVIDIA Container Toolkit
- Docker with CDI device support

## Configuration

Create the runtime configuration from the example:

```bash
cp config/config-example.ini config/config.ini
```

Important settings are:

```ini
[basic]
stream_url = rtsp://user:password@camera/main
stream_url_lowres = rtsp://user:password@camera/sub
push_url = http://server/door
motion_url = http://server/motion

[face_recognition]
detector_model = opencv
recognition_model = Facenet512
metric = euclidean
```

When `motion.use_internal = True`, `motion_url` is not used. Internal motion
detection uses the low-resolution stream when configured, otherwise the main
stream.

### Face database

Each identity needs its own directory and an image with the same name:

```text
db/
├── alice/
│   └── alice.jpg
└── bob/
    └── bob.png
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`.

## Run on CPU

The base Compose file does not request a GPU:

```bash
docker compose up -d --build
```

The image contains CUDA-capable TensorFlow and PyTorch packages, but both
frameworks can fall back to CPU execution when no GPU is exposed.

CPU performance depends heavily on the selected models. Detectors such as
`opencv` or `ssd` are generally better starting points for CPU systems than
larger RetinaFace or YOLO variants.

## Run with an NVIDIA GPU

GPU support is provided by `docker-compose.gpu.yml`. This file is an override,
not a standalone Compose configuration. It is merged with
`docker-compose.yml`:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up -d --build
```

The base file continues to provide settings such as `build`, volumes, and
`restart: unless-stopped`. The GPU override only adds:

- The NVIDIA CDI device reservation
- CUDA cache settings
- `FACIALREC_REQUIRE_GPU=1`
- GPU checks before the application starts

Use the same file combination for other Compose commands:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

If the GPU is not available or cannot be used by TensorFlow or PyTorch, the GPU
variant fails its startup checks instead of silently using the CPU.

### NVIDIA host setup

Install a current NVIDIA driver and NVIDIA Container Toolkit using the
instructions for the host operating system. The Compose override uses CDI
device names.

Typical verification commands are:

```bash
nvidia-smi
nvidia-ctk --debug cdi list
docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi
```

Depending on the installation, the CDI specification may need to be refreshed
after driver updates, GPU changes, or MIG configuration changes. With a
systemd-based NVIDIA Container Toolkit installation this is commonly done with:

```bash
sudo systemctl restart nvidia-cdi-refresh.service
nvidia-ctk --debug cdi list
```

To select a specific GPU, change `device_ids` in
`docker-compose.gpu.yml`, for example:

```yaml
device_ids:
  - "nvidia.com/gpu=0"
```

## Startup checks

The normal CPU container command is:

```text
python3 test.py --base --start
```

The GPU override changes it to:

```text
python3 test.py --base --gpu --start
```

`main.py` starts only when all selected checks pass.

The base startup check validates:

- Configuration sections, values, URLs, and value ranges
- Face database structure and image readability
- Main and optional low-resolution RTSP streams
- Internal motion processing or the external motion endpoint
- Detector and recognition model loading
- Distance metric support
- A sample face representation using a database image

The configured `push_url` is validated syntactically but is not called during
tests, because doing so could trigger the connected action.

## Manual tests

Run tests in an active container:

```bash
docker compose exec facialrec python3 test.py --base
docker compose exec facialrec python3 test.py --motion
docker compose exec facialrec python3 test.py --gpu
docker compose exec facialrec python3 test.py --all
```

Tests can be combined:

```bash
docker compose exec facialrec python3 test.py --base --motion
```

If the service cannot stay up because its startup check fails, use a one-off
container:

```bash
docker compose run --rm facialrec python3 test.py --base
```

For the GPU configuration:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm facialrec python3 test.py --gpu
```

Available options:

- `--base`: configuration, database, streams, motion source, and models
- `--gpu`: NVIDIA, TensorFlow GPU, and PyTorch CUDA checks
- `--motion`: observe the configured motion detector
- `--all`: run all checks
- `--start`: start `main.py` after successful checks
- `--timeout`: RTSP and command timeout
- `--gpu-timeout`: timeout for each GPU framework check
- `--motion-duration`: motion observation duration
- `--config-path`: use a different configuration file

Running `python3 test.py` without a test option defaults to `--base`.

## Detector notes

Set the exact DeepFace detector backend name in `config/config.ini`, for
example:

```ini
[face_recognition]
detector_model = yolov8m
```

YOLO detectors use PyTorch. Other detectors such as RetinaFace and MTCNN use
TensorFlow-backed components. Available performance and memory requirements
vary by detector and hardware.

`fastmtcnn` is not bundled because its current dependency constraints conflict
with the PyTorch version used by this image. Use another supported detector
such as `opencv`, `ssd`, `mtcnn`, `retinaface`, or a YOLO variant.

## Persistent data

The Compose configuration mounts:

- `./config` to `/app/config`
- `./db` to `/app/db`
- `./weights` to `/root/.deepface/weights`
- `./cuda-cache` to `/var/cache/nvidia/ComputeCache` in GPU mode

Model weights may be downloaded on the first start and are retained in the
`weights` directory.

## Troubleshooting

### Startup check fails

Run the base test in a one-off container:

```bash
docker compose run --rm facialrec python3 test.py --base
```

The process returns a non-zero exit code and reports each failed check.

### GPU is not visible

Run:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm facialrec python3 test.py --gpu
```

If `nvidia-smi` fails inside the container, check the host driver, NVIDIA
Container Toolkit, and CDI configuration. If only TensorFlow or PyTorch fails,
rebuild the image and inspect the reported framework and CUDA information:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  build --pull --no-cache
```

### CDI device errors

Refresh the CDI specification and verify that
`nvidia.com/gpu=all` or the configured device ID appears:

```bash
sudo systemctl restart nvidia-cdi-refresh.service
nvidia-ctk --debug cdi list
```

The exact refresh command depends on the host operating system and NVIDIA
Container Toolkit installation.

### GPU container fails after a host reboot

On some systemd-based hosts, Docker may start before the NVIDIA device nodes,
persistence service, or CDI specification are ready. Containers using
`restart: unless-stopped` can then fail during automatic startup even though a
manual restart works later.

First enable the NVIDIA persistence and CDI refresh services when they are
available on the host:

```bash
sudo systemctl enable --now nvidia-persistenced.service
sudo systemctl enable --now nvidia-cdi-refresh.path
sudo systemctl restart nvidia-cdi-refresh.service
```

If the race condition remains, add an optional Docker service override:

```bash
sudo systemctl edit docker.service
```

```ini
[Unit]
After=nvidia-persistenced.service nvidia-cdi-refresh.service systemd-udev-settle.service
Wants=nvidia-persistenced.service nvidia-cdi-refresh.service

[Service]
ExecStartPre=/usr/bin/bash -c 'for i in {1..20}; do [ -e /dev/nvidia0 ] && [ -e /dev/nvidia-uvm ] && exit 0; sleep 1; done; exit 1'
```

Reload systemd and restart Docker:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

This configuration delays Docker until the NVIDIA services are active and the
required device nodes exist. Service names and device paths can differ between
distributions, driver packages, rootless Docker installations, and systems
using MIG. Inspect the available units and devices before applying the
override.

### RTSP check fails

Verify the configured URL, credentials, network access, and camera stream. The
container uses TCP for RTSP through:

```text
OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp
```
