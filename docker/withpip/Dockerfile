FROM tsekai/a2bs_base:latest
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY requirements.txt /work/requirements.txt

RUN pip install  --no-cache-dir -r requirements.txt