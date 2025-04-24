FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get purge -y python3-blinker \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/ld.so.conf.d/z-cuda-stubs.conf \
 && rm -rf /usr/local/cuda/lib64/stubs \
 && ldconfig

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY req.txt    /app
COPY main.py    /app
COPY include/   /app/include
COPY lib/       /app/lib
COPY db/        /app/db
COPY weights/   /root/.deepface/weights

RUN pip3 install --no-cache-dir -r req.txt
RUN pip3 install --upgrade "tensorflow[and-cuda]"
RUN pip3 install tf-keras

#CMD ["bash"]   #DEBUG
CMD ["python3", "main.py"]