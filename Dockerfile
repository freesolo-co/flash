# AutoSLM control plane (operator-side). See docs/self-hosting.md.
#
#   docker build -t autoslm-server .
#   docker run -p 8080:8080 \
#     -e RUNPOD_API_KEY=... -e HUGGINGFACE_TOKEN=... -e HF_REPO=org/autoslm-runs \
#     -v autoslm-state:/state autoslm-server
#
# All persistent state (key DB, run records, results) lives under /state — mount a
# volume there. Run exactly ONE container instance per state volume (state is local
# files + SQLite; horizontal scaling is not supported).

FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[server]"

ENV AUTOSLM_DB_PATH=/state/server.db \
    AUTOSLM_RUNS_DIR=/state/runs \
    RESULTS_DIR=/state/results

VOLUME /state
EXPOSE 8080

CMD ["slm", "server", "--host", "0.0.0.0", "--port", "8080"]
