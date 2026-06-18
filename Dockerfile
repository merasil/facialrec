FROM tensorflow/tensorflow:2.21.0-gpu

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

RUN apt-get update && \
    apt-get purge -y python3-blinker \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        python3-cairo \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/ld.so.conf.d/z-cuda-stubs.conf \
 && rm -rf /usr/local/cuda/lib64/stubs \
 && ldconfig

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY req.txt    /app

RUN pip3 install --no-cache-dir \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
 && pip3 install --no-cache-dir -r req.txt \
 && pip3 install --no-cache-dir "tf-keras==2.21.0"

RUN pip3 check

COPY main.py    /app
COPY test.py    /app
COPY include/   /app/include
COPY lib/       /app/lib
COPY db/        /app/db
COPY weights/   /root/.deepface/weights

# Run the base preflight before entering the long-running service.
CMD ["python3", "test.py", "--base", "--start"]
