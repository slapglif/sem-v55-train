FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY sem/ sem/
COPY configs/ configs/
COPY tokenizer/ tokenizer/

RUN uv pip install --system --no-cache \
    -e ".[training]" \
    huggingface_hub[hf_xet] \
    scipy>=1.11.0

ENV HF_HUB_REPO_ID=icarus112/sem-v55-lean-crystal

CMD ["python", "-m", "sem.train", "--config", "configs/cloud_a10g.yaml", "--device", "cuda"]
