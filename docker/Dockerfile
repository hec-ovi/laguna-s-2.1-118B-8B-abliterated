# Laguna abliteration runtime for AMD Strix Halo (gfx1151).
# All python/torch lives inside this image; nothing is installed on the host.
# Default target is ROCm via AMD's TheRock native gfx1151 nightly wheels (self-contained,
# no host ROCm install). Build the CPU variant by overriding TORCH_INDEX / TORCH_PRE.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# venv inside the image (isolated; the host keeps no python env)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U pip wheel

# torch: default ROCm/TheRock gfx1151 nightly; CPU build overrides both args.
#   ROCm (default): TORCH_INDEX=https://rocm.nightlies.amd.com/v2/gfx1151/  TORCH_PRE=--pre
#   CPU:            TORCH_INDEX=https://download.pytorch.org/whl/cpu        TORCH_PRE=
ARG TORCH_INDEX="https://rocm.nightlies.amd.com/v2/gfx1151/"
ARG TORCH_PRE="--pre"
RUN pip install ${TORCH_PRE} torch torchvision torchaudio --index-url "${TORCH_INDEX}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# TheRock native gfx1151 wheels need no HSA_OVERRIDE_GFX_VERSION.
ENV PYTHONPATH=/work/src
ENV HF_HOME=/work/.hf
WORKDIR /work
# default: print the torch/rocm status so `docker run <img>` is a quick smoke test
CMD ["python", "-c", "import torch; print('torch', torch.__version__, '| hip', getattr(torch.version,'hip',None), '| gpu', torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"]
