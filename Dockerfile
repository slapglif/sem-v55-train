FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./
COPY sem/ sem/
COPY configs/ configs/
COPY tokenizer/ tokenizer/

RUN uv pip install --system --no-cache \
    torch>=2.2.0 \
    datasets>=2.16.0 \
    tokenizers>=0.15.0 \
    transformers>=4.36.0 \
    tqdm>=4.66.0 \
    scipy>=1.11.0 \
    einops>=0.7.0 \
    pyyaml>=6.0 \
    numpy>=1.24.0 \
    huggingface_hub[hf_xet]

ENV HF_HUB_REPO_ID=icarus112/sem-v55-lean-crystal
ENV PYTHONPATH=/app

CMD ["python", "-m", "sem.train", "--config", "configs/cloud_a10g.yaml", "--device", "cuda"]
