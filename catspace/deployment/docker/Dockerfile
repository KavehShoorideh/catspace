FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends stockfish git curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scipy python-chess qdrant-client
COPY catspace/ catspace/
COPY experiments/ experiments/
ENV PYTHONUNBUFFERED=1
EXPOSE 8777
HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost:8777/health || exit 1
CMD ["python", "experiments/viz/assistant_server.py", "--host", "0.0.0.0", "--port", "8777"]
