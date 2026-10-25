"""Run the unchanged private HTTP endpoint for one precision arm."""

from flash.serving.deepinfer.app.http import build_app
from flash.serving.deepinfer.runtime.bridge import Runtime

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(Runtime()), host="127.0.0.1", port=8000)
