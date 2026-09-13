ARG TENSORFLOW_IMAGE=tensorflow/tensorflow:2.21.0

# Keep application changes independent of the expensive dependency layers.
FROM scratch AS application
COPY main.py test.py /app/
COPY include/ /app/include/
COPY lib/ /app/lib/

FROM ${TENSORFLOW_IMAGE} AS runtime

ENV FACIALREC_REQUIRE_TORCH=0

RUN apt-get update && \
    apt-get purge -y python3-blinker \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/ld.so.conf.d/z-cuda-stubs.conf \
 && rm -rf /usr/local/cuda/lib64/stubs \
 && ldconfig

RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY req.txt    /app

RUN pip3 install --no-cache-dir -r req.txt \
 && pip3 install --no-cache-dir "tf-keras==2.21.0"

RUN pip3 check

RUN mkdir -p /app/config /app/db /root/.deepface/weights

# Run the base preflight before entering the long-running service.
CMD ["python3", "test.py", "--base", "--start"]

# Keep standard before gpu-extended: the classic builder stops at its target.
FROM runtime AS standard
COPY --from=application /app/ /app/

# Only this target installs YOLO and the PyTorch CUDA stack.
# GPU Compose files select the TensorFlow GPU base through TENSORFLOW_IMAGE.
FROM runtime AS gpu-extended

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

RUN pip3 install --no-cache-dir \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

RUN pip3 install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        ultralytics \
 && pip3 check

ENV FACIALREC_REQUIRE_TORCH=1
COPY --from=application /app/ /app/
