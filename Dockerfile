FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV TRANSFORMERS_VERBOSITY=info

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY handler.py .

# ==========================================
# CHECK PYTHON AND AI DEPENDENCIES
# ==========================================

RUN python3 -m py_compile handler.py

RUN python3 -c "import torch; print('TORCH OK:', torch.__version__)"

RUN python3 -c "from transformers import CLIPModel, CLIPProcessor; print('CLIP IMPORT OK')"

RUN python3 -c "from rembg import new_session, remove; print('REMBG IMPORT OK')"

# ==========================================
# START RUNPOD
# ==========================================

CMD ["python3", "-u", "handler.py"]
