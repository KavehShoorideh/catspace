# Serving image for the catspace engine. Build context is the REPO ROOT:
#   docker build -f catspace/deployment/docker/Dockerfile .
# .dockerignore keeps the research tree, datasets, tablebases and mlruns out of the context.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends stockfish git curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scipy python-chess qdrant-client

# The package only. Deployment consumes an end-to-end config from catspace/approaches/;
# it never needs experiments/, which is why that tree is no longer copied.
COPY pyproject.toml README.md ./
COPY catspace/ catspace/

ENV PYTHONUNBUFFERED=1
EXPOSE 8777
HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost:8777/health || exit 1
CMD ["python", "-m", "catspace.deployment.server", "--host", "0.0.0.0", "--port", "8777"]
