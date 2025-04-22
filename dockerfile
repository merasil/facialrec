FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY req.txt /app
COPY ./main.py /app
COPY ./include /app
COPY ./lib /app
COPY ./db /app

#RUN pip3 install --break-system-packages --no-cache-dir -r req.txt
RUN pip3 install --no-cache-dir -r req.txt

RUN sed -i '/stubs/d' /etc/ld.so.conf.d/cuda*.conf \
 && ldconfig

CMD ["python3", "main.py"]
