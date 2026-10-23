# Build the regular worker from this checkout first, then supply its image here.
# The opt-in pilot needs CuPy for Ray's NCCL checkpoint transport. Keep this
# dependency out of the default worker until a GPU pilot validates the path.
ARG WORKER_IMAGE
FROM ${WORKER_IMAGE}
# Keep the NCCL host-allocation path consistent across pure actor and receiver
# processes. The Modal A10G pilot failed initial publication without this.
ENV NCCL_CUMEM_HOST_ENABLE=0
RUN uv pip install --python /opt/verl-venv/bin/python "cupy-cuda12x==13.6.0" \
    && /opt/verl-venv/bin/python -c \
       "from verl.checkpoint_engine.base import CheckpointEngineRegistry; CheckpointEngineRegistry.get('nccl')"
