FROM tensorflow/tensorflow:2.21.0-gpu

RUN apt-get update && \
    apt-get purge -y python3-blinker \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/ld.so.conf.d/z-cuda-stubs.conf \
 && rm -rf /usr/local/cuda/lib64/stubs \
 && ldconfig

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY req.txt req-constraints.txt /app/
COPY main.py    /app
COPY app/       /app/app
COPY include/   /app/include
COPY lib/       /app/lib
COPY db/        /app/db
COPY weights/   /root/.deepface/weights

RUN pip3 install --no-cache-dir \
      --constraint req-constraints.txt \
      --upgrade \
      "tensorflow[and-cuda]==2.21.0" tf-keras==2.21.0
RUN pip3 install --no-cache-dir \
      --constraint req-constraints.txt \
      --requirement req.txt
# facenet-pytorch 2.6.0 pins an obsolete Torch stack. Its runtime works with
# the Torch/Torchvision versions selected in req-constraints.txt.
RUN pip3 install --no-cache-dir --no-deps facenet-pytorch==2.6.0
RUN python3 -c 'import tensorflow, torch, torchvision; print("TensorFlow", tensorflow.__version__, "Torch", torch.__version__, "Torchvision", torchvision.__version__)'

#CMD ["bash"]   #DEBUG
CMD ["python3", "main.py"]
