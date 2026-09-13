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

- Docker Engine with BuildKit
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

### Motion frame size

Use `resize_factor` in `[motion]` to reduce the work done by internal motion
detection:

```ini
[motion]
use_internal = True
resize_factor = 1
```

The factor must be an integer greater than or equal to 1. It divides the actual
stream width and height equally, preserving the aspect ratio except for rounding
down to whole pixels. Factor 1 skips resizing. For a 640x360 substream, start with
factor 1; factor 2 produces 320x180 and processes one quarter as many pixels.
Both resulting dimensions must be at least 32 pixels, which the startup motion
check validates against the stream. An excessive factor fails that check with
the source and resulting dimensions in the error message.
The existing `test.py` preflight validates the factor with the other numeric
settings before starting the service. Run `python3 test.py --base --start` when
starting outside Docker as well; `main.py` reads the settings directly.

Blur and dilation are scaled with the factor to keep their extent in the original
scene approximately consistent. Minimum motion area remains a percentage of the
processed image. Small or distant movements can still be lost when reducing the
resolution; check sensitivity in daylight and at night after changing the factor.
The service logs the source and processing dimensions on the first frame and
after a resolution change. A resolution change resets the background reference
and any active motion cooldown before detection resumes on subsequent frames.

Resizing happens after receiving and decoding the motion stream. It saves work
in motion analysis, not network bandwidth or video decoding. Face recognition
continues to use the main stream at its original resolution.

If your existing configuration contains `stream_resize` in `[basic]`, remove it
and add `resize_factor` in `[motion]`. Startup rejects the old option with a
migration message, even if the new option is also present. Old `False` corresponds
to factor 1. Old `True` forced every stream to 320x240, so there is no universal
replacement factor: for 640x480, factor 2 retains that size; for 640x360, factor 2
now correctly produces 320x180. With a high-resolution motion stream, choose the
factor according to its actual size. Omitting the new option defaults to factor 1.

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

## Image variants

The project builds three local images from the same Dockerfile:

| Variant | Local image | TensorFlow base | Additional frameworks |
| --- | --- | --- | --- |
| CPU | `facialrec:cpu` | `tensorflow/tensorflow:2.21.0` | None |
| GPU | `facialrec:gpu` | `tensorflow/tensorflow:2.21.0-gpu` | None |
| GPU extended | `facialrec:gpu-extended` | `tensorflow/tensorflow:2.21.0-gpu` | PyTorch CUDA 12.8, Torchvision, Ultralytics/YOLO |

CPU and GPU support RetinaFace + Facenet512 and other backends provided by their
installed dependencies. YOLO requires GPU extended. Model names are still chosen
in `config/config.ini`; changing the model configuration does not install extra
packages.

The `standard` build target uses the CPU or GPU base selected by the
`TENSORFLOW_IMAGE` build argument. The `gpu-extended` target adds the PyTorch and
YOLO dependencies. BuildKit skips the extended stage when building `standard`.
The legacy builder may execute unused stages and should not be used.
Compose selects the base and target together and assigns distinct image tags.
Images are built locally (`pull_policy: never`); use `up --build` for the first
start and after changing variant or code.

DeepFace's required dependencies remain installed even if the selected model
does not use all of them. Model weights, reference images, configuration, and
CUDA cache files are excluded from the build context and provided through mounts.
Application code is copied after dependency installation, so changing Python
files does not invalidate the expensive package installation layers.

## Run on CPU

The base Compose file does not request a GPU:

```bash
docker compose up -d --build
```

This uses the TensorFlow CPU base without installing PyTorch, Torchvision, or
Ultralytics. The container does not request an NVIDIA device.

CPU performance depends heavily on the selected models. Detectors such as
`opencv` or `ssd` are generally better starting points for CPU systems than
RetinaFace.

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

The base file continues to provide the build context, data mounts, and
`restart: unless-stopped`. The GPU override selects the standard GPU image and adds:

- The NVIDIA CDI device reservation
- CUDA cache settings
- `FACIALREC_REQUIRE_GPU=1`
- `FACIALREC_TF_GPU_MEMORY_LIMIT_MB=2048`
- GPU checks before the application starts

Use the same file combination for other Compose commands:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

If the GPU is not available or cannot be used by TensorFlow, the GPU variant
fails its startup checks instead of silently using the CPU. PyTorch is not
required in this variant.

By default, the GPU override caps TensorFlow-backed detectors such as
`retinaface` and `mtcnn` at about 2 GB of GPU memory. This leaves room for
another process such as Ollama. Increase `FACIALREC_TF_GPU_MEMORY_LIMIT_MB` if
TensorFlow reports out-of-memory errors for your selected detector or
recognition model. Set it to `0` or remove the variable to use TensorFlow's
memory growth mode without a hard cap.

### GPU extended with YOLO

Add the extended override after the base and GPU files:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.gpu-extended.yml \
  up -d --build
```

This selects `facialrec:gpu-extended` and adds PyTorch, Torchvision, and
Ultralytics. The GPU override still supplies the NVIDIA device access, memory
settings, and startup command. Use all three files for subsequent build, run,
logs, and down commands for this variant.

The extended image sets `FACIALREC_REQUIRE_TORCH=1`, so its GPU preflight checks
both TensorFlow and PyTorch. A missing or unusable PyTorch installation fails the
extended preflight. This flag is set in the image; it does not install packages.

For example, configure `detector_model = yolov8m` and keep
`recognition_model = Facenet512` to detect faces with PyTorch/YOLO and recognize
them with TensorFlow. With RetinaFace + Facenet512, the standard GPU variant is
sufficient.

### Build and verify on your server

Transfer the updated project files, including `.dockerignore` and the new
`docker-compose.gpu-extended.yml`, to the server. Keep your existing `config`,
`db`, and `weights` directories. Run the following from the project directory
in Bash and enable BuildKit:

```bash
export DOCKER_BUILDKIT=1
```

Then choose exactly one Compose file selection. For CPU:

```bash
export COMPOSE_FILE=docker-compose.yml
```

For GPU with RetinaFace + Facenet512:

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
```

For GPU extended with YOLO:

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.gpu-extended.yml
```

Validate the selected Compose configuration and build its image:

```bash
docker compose config --quiet
docker compose --progress plain build facialrec
```

Proceed only after the build succeeds. Run the existing preflight as a one-off
container. For CPU:

```bash
docker compose run --rm facialrec python3 test.py --base
```

For either GPU variant:

```bash
docker compose run --rm facialrec python3 test.py --base --gpu
```

Use `detector_model = retinaface` and `recognition_model = Facenet512` to test
the standard variants. To also verify YOLO in the extended image, set
`detector_model = yolov8m` in your configuration before running its preflight.
The preflight does not call the door-opening endpoint. It may download missing
model weights into the mounted `weights` directory.

After a successful preflight, start the selected service and inspect its logs:

```bash
docker compose up -d --no-build
docker compose logs --tail=100 -f facialrec
```

Compare the sizes of the variants you have built with `docker image ls facialrec`.
Use the same `COMPOSE_FILE` selection for subsequent commands; environment
variables need to be set again in a new shell. Image tags coexist, but these
Compose selections manage the same service and replace it when switching variants.

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

`--gpu` checks NVIDIA access and TensorFlow. It also checks PyTorch when that
package is installed or `FACIALREC_REQUIRE_TORCH=1`. Standard images report the
PyTorch test as skipped. CPU startup does not request GPU tests.

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
- `--gpu`: NVIDIA and TensorFlow GPU checks; PyTorch when installed or required
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

YOLO detectors require the GPU extended image and use PyTorch. Other detectors
such as RetinaFace and MTCNN use TensorFlow-backed components. Available
performance and memory requirements vary by detector and hardware.

`fastmtcnn` is not bundled because its current dependency constraints conflict
with the PyTorch version used by the extended image. Use another supported
detector such as `opencv`, `ssd`, `mtcnn`, `retinaface`, or a YOLO variant.

## Persistent data

The Compose configuration mounts:

- `./config` to `/app/config`
- `./db` to `/app/db`
- `./weights` to `/root/.deepface/weights`
- `./cuda-cache` to `/var/cache/nvidia/ComputeCache` in GPU mode

Model weights may be downloaded on the first start and are retained in the
`weights` directory. Existing weights and reference images remain on the host
when changing image variants. They are no longer copied into the image. If you
use `docker run` directly, provide the configuration, database, and weights
mounts as well.

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
