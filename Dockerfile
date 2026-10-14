# ── Stage 1: Build Static Web App ──────────────────────────────────────────
FROM node:22-alpine AS web-builder

WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# ── Stage 2: Runtime Backend & App ──────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# System dependencies for soundfile and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install CPU PyTorch and dependencies
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.lock ./
# Filter out torch/torchaudio from requirements.lock if present so CPU wheels are preserved
RUN pip install --no-cache-dir -r requirements.lock || pip install --no-cache-dir fastapi uvicorn soundfile scipy librosa transformers pydantic

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY data/library/ ./data/library/
COPY data/demo/ ./data/demo/
COPY --from=web-builder /app/web/out/ ./web/out/

# Install speechlens package
RUN pip install --no-cache-dir -e .

# Cache wav2vec2 model weights at build time
RUN python -m speechlens warmup || true

# Set permissions for non-root user
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
