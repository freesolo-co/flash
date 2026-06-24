# Flash control plane (operator-side).
#
#   docker build -t flash-control-plane .
#   docker run -p 8080:8080 \
#     -e RUNPOD_API_KEY=... -e HF_TOKEN=... \
#     -v flash-state:/root/.flash flash-control-plane
#
# All persistent state (key DB, run records, results) lives under ~/.flash (fixed paths,
# = /root/.flash for the default root user) — mount a volume there. Run exactly ONE
# container instance per state volume (state is local files + SQLite; no horizontal scaling).

FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ".[server]"

VOLUME /root/.flash
EXPOSE 8080

CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
