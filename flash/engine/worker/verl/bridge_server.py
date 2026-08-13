"""The HTTP server the reward and teacher bridges are built on.

Both bridges are pure i/o relays between the verl child and the flash worker, and both must be
unable to keep the worker alive: a hung environment reward or teacher callback must not stop the
worker from publishing its terminal result and releasing a paid GPU.

Split out of `flash.engine.worker.backend_common` to keep that module under the file-size limit, and
re-exported from there so existing importers and test patches keep resolving.
"""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _thread_module
from http.server import ThreadingHTTPServer

# worker threads each bridge serves requests from. the bridges are pure i/o relays (parse json,
# hand the payload to a callback, write json back), so this bounds concurrency, not throughput:
# requests beyond the pool wait in the listen backlog instead of each spawning an os thread.
_BRIDGE_WORKER_THREADS = 16


class _DaemonBridgeThreadPool(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose workers cannot hold the interpreter open.

    A hung environment reward or teacher callback must not stop the worker from publishing its
    terminal result and releasing a paid GPU -- and it is also the property the old daemon
    ``ThreadingHTTPServer`` had before requests moved onto a pool. Only the thread creation differs
    from the stdlib; queueing, idle reuse, and shutdown are inherited unchanged.
    """

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread = threading.Thread(
                name=f"{self._thread_name_prefix or self}_{num_threads}",
                target=_thread_module._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            thread.start()
            self._threads.add(thread)
            # deliberately NOT added to _thread_module._threads_queues: that mapping is exactly
            # what the interpreter-exit hook walks to join workers.


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """an http server that serves requests from a fixed thread pool instead of one thread each.

    unbounded ``ThreadingHTTPServer`` can exhaust the thread table and cause
    ``RemoteDisconnected`` (VERL-139). the fixed pool prevents that, while the larger listen backlog
    must absorb rollout bursts instead of resetting queued connections.
    """

    # daemon threads so a stuck handler can never keep the worker process alive at shutdown.
    # `daemon_threads` is a ThreadingMixIn flag, read only by the `process_request` this class
    # overrides, so it does nothing here on its own -- _DaemonBridgeThreadPool above is what
    # actually delivers the property it names.
    daemon_threads = True

    # measured: at the socketserver default of 5, 13 of 64 simultaneous callers were reset by peer.
    request_queue_size = 128

    def __init__(self, *args, worker_threads: int = _BRIDGE_WORKER_THREADS, **kwargs):
        self._bridge_pool = _DaemonBridgeThreadPool(
            max_workers=worker_threads, thread_name_prefix="flash-bridge"
        )
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address) -> None:
        """hand the request to the pool. socketserver's default starts an unbounded thread here."""
        self._bridge_pool.submit(self._run_request, request, client_address)

    def _run_request(self, request, client_address) -> None:
        # mirrors ThreadingMixIn.process_request_thread: the pool owns the thread, so shutdown of
        # the connection (not the server) is all this needs to do.
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        # do not wait: a handler blocked on a hung peer must not stop the worker from exiting.
        self._bridge_pool.shutdown(wait=False, cancel_futures=True)
