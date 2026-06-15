# AutoSLM control plane (operator-side).
#
#   docker build -t autoslm-server .
#   docker run -p 8080:8080 \
#     -e RUNPOD_API_KEY=... -e HUGGINGFACE_TOKEN=... -e HF_REPO=org/autoslm-runs \
#     -v autoslm-state:/root/.autoslm autoslm-server
#
# All persistent state (key DB, run records, results) lives under ~/.autoslm (fixed paths,
# = /root/.autoslm for the default root user) — mount a volume there. Run exactly ONE
# container instance per state volume (state is local files + SQLite; no horizontal scaling).

FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[server]"

VOLUME /root/.autoslm
EXPOSE 8080

CMD ["slm", "server", "--host", "0.0.0.0", "--port", "8080"]
