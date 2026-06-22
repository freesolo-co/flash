"""Pool registry + balancing policy (pure, no IO — unit-testable on CPU).

``PoolState`` is the router's source of truth: which GPU workers (``Backend``) exist, which base
model each serves, which LoRA adapters are currently loaded where, and the live in-flight count
per backend. The balancing decisions (:meth:`PoolState.pick_backend`, :meth:`place_adapter`) are
the "nginx upstream" policy — they pick *where* a request goes and *where* an adapter is loaded —
and they are pure functions of the registry so they can be reasoned about and tested without a GPU
or a network. All the actual HTTP (loading a LoRA, forwarding a generation) lives in
:mod:`flash.pool.gateway`; the router calls the state under a lock, then performs IO.

Balancing policy (least-outstanding-requests, LoRA-aware):
  * a request for adapter ``A`` prefers a healthy backend that ALREADY has ``A`` loaded, choosing
    the one with the fewest in-flight requests (warm + least-loaded);
  * otherwise it picks a healthy backend serving ``A``'s base model with a free LoRA slot (fewest
    adapters, then fewest in-flight) and signals that ``A`` must be loaded there first;
  * if every base-model backend is at its LoRA cap, it picks the least-loaded one and signals an
    eviction (vLLM LRU-swaps; we evict the least-recently-served adapter explicitly to keep the
    registry honest);
  * if no healthy backend serves that base model at all, it raises :class:`NoCapacityError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class NoCapacityError(RuntimeError):
    """No healthy backend can serve the requested base model / adapter."""


@dataclass
class Backend:
    """One pre-rented GPU inference worker — a vLLM OpenAI server (the nginx 'upstream').

    A backend serves exactly one *base model* but many LoRA adapters concurrently (vLLM
    ``--enable-lora --max-loras``), which is what lets many small runs share one GPU.
    """

    id: str
    url: str  # e.g. "http://10.0.0.5:8000" — the vLLM OpenAI-compatible server
    base_model: str  # HF id of the base weights loaded on this GPU
    gpu_label: str = ""  # human label, e.g. "vast-1781/gpu0" (provider/instance/device)
    max_loras: int = 8  # concurrent LoRA adapters this GPU can hold (vLLM --max-loras)
    max_concurrency: int = 256  # soft in-flight cap before the backend is considered saturated
    cost_per_hour: float = 0.0  # for utilization/cost reporting only

    # ---- live state (mutated by the router) ----
    healthy: bool = True
    inflight: int = 0
    adapters: set[str] = field(default_factory=set)  # adapter names currently loaded here
    # monotone counter used as an LRU clock: when an adapter is served/loaded we stamp it.
    adapter_clock: dict[str, int] = field(default_factory=dict)
    total_requests: int = 0
    total_failures: int = 0
    last_health_ok: float = 0.0
    last_health_err: str = ""

    @property
    def free_lora_slots(self) -> int:
        return max(0, self.max_loras - len(self.adapters))

    @property
    def saturated(self) -> bool:
        return self.inflight >= self.max_concurrency

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "base_model": self.base_model,
            "gpu_label": self.gpu_label,
            "healthy": self.healthy,
            "inflight": self.inflight,
            "adapters": sorted(self.adapters),
            "max_loras": self.max_loras,
            "free_lora_slots": self.free_lora_slots,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "cost_per_hour": self.cost_per_hour,
            "last_health_err": self.last_health_err,
        }


@dataclass
class Adapter:
    """A LoRA adapter = one training run's policy. ``name`` is what the trainer passes as the
    OpenAI ``model`` field; ``uri`` is where the current weights live (a node-local path on a
    shared FS, or a remote store the backends can pull). ``version`` bumps on every weight sync so
    the router knows a hot-swap (unload+reload) is due on each placed backend."""

    name: str
    base_model: str
    uri: str  # LoRA dir: local path (shared FS) or remote uri (hf://, s3://, https://)
    version: int = 0  # bumped by sync_adapter(); placements lag until reloaded
    replicas: int = 1  # desired number of backends to keep this adapter warm on
    placements: set[str] = field(default_factory=set)  # backend ids hosting the CURRENT version
    # backend_id -> version currently loaded there (so the router only reloads what's stale)
    loaded_version: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "uri": self.uri,
            "version": self.version,
            "replicas": self.replicas,
            "placements": sorted(self.placements),
            "stale": sorted(self.stale_placements()),
        }

    def stale_placements(self) -> set[str]:
        """Backends whose loaded version is behind the adapter's current version."""
        return {b for b in self.placements if self.loaded_version.get(b, -1) != self.version}


@dataclass
class PlacementDecision:
    """The balancer's answer: which backend to use, and whether the adapter must be loaded /
    evicted there first (IO the router performs before forwarding the request)."""

    backend: Backend
    needs_load: bool = False  # the adapter is not on this backend yet — load it
    evict: str | None = None  # an adapter name to unload first (backend at its LoRA cap)


class PoolState:
    """Registry of backends + adapters with the (pure) balancing policy."""

    def __init__(self) -> None:
        self.backends: dict[str, Backend] = {}
        self.adapters: dict[str, Adapter] = {}
        self._clock = 0  # monotone LRU clock

    # ---- backend registry ----
    def add_backend(self, backend: Backend) -> Backend:
        self.backends[backend.id] = backend
        return backend

    def remove_backend(self, backend_id: str) -> Backend | None:
        be = self.backends.pop(backend_id, None)
        if be is not None:
            for ad in self.adapters.values():
                ad.placements.discard(backend_id)
                ad.loaded_version.pop(backend_id, None)
        return be

    def set_health(self, backend_id: str, healthy: bool, err: str = "") -> None:
        be = self.backends.get(backend_id)
        if be is None:
            return
        be.healthy = healthy
        if healthy:
            be.last_health_err = ""
        elif err:
            be.last_health_err = err[:300]

    # ---- adapter registry ----
    def register_adapter(self, adapter: Adapter) -> Adapter:
        existing = self.adapters.get(adapter.name)
        if existing is not None:
            # Re-register = update the uri / desired replicas; keep live placements. If the weight
            # path changed, the existing placements now hold STALE files, so bump the version: that
            # marks every placement stale (``stale_placements``), and the router hot-swaps the new
            # weights on the next serve (or an explicit sync/place) instead of treating warm
            # backends as already serving the new policy.
            base_changed = existing.base_model != adapter.base_model
            weights_changed = existing.uri != adapter.uri
            existing.base_model = adapter.base_model
            existing.uri = adapter.uri
            existing.replicas = adapter.replicas
            if base_changed:
                # The base model changed under the same adapter name: every existing placement is on
                # a backend serving the OLD base, so it cannot serve this adapter anymore. Drop those
                # placements (and detach the adapter from those backends) — otherwise the warm path
                # in ``pick_backend`` would route requests to a wrong-base backend. The router will
                # lazily (re)place the adapter on a correct-base backend on the next serve.
                for bid in list(existing.placements):
                    be = self.backends.get(bid)
                    if be is not None:
                        be.adapters.discard(adapter.name)
                        be.adapter_clock.pop(adapter.name, None)
                existing.placements.clear()
                existing.loaded_version.clear()
            if weights_changed:
                existing.version += 1
            return existing
        self.adapters[adapter.name] = adapter
        return adapter

    def bump_adapter_version(self, name: str, uri: str | None = None) -> Adapter:
        """Record that ``name`` has new weights (a GRPO weight-sync). Placed backends become
        stale until the router reloads them."""
        ad = self.adapters[name]
        ad.version += 1
        if uri:
            ad.uri = uri
        return ad

    def drop_adapter(self, name: str) -> Adapter | None:
        ad = self.adapters.pop(name, None)
        if ad is not None:
            for bid in list(ad.placements):
                be = self.backends.get(bid)
                if be is not None:
                    be.adapters.discard(name)
                    be.adapter_clock.pop(name, None)
        return ad

    # ---- accounting (called by the router around each forwarded request) ----
    def mark_loaded(self, backend_id: str, adapter_name: str) -> None:
        be = self.backends[backend_id]
        be.adapters.add(adapter_name)
        self._clock += 1
        be.adapter_clock[adapter_name] = self._clock
        ad = self.adapters.get(adapter_name)
        if ad is not None:
            ad.placements.add(backend_id)
            ad.loaded_version[backend_id] = ad.version

    def mark_unloaded(self, backend_id: str, adapter_name: str) -> None:
        be = self.backends.get(backend_id)
        if be is not None:
            be.adapters.discard(adapter_name)
            be.adapter_clock.pop(adapter_name, None)
        ad = self.adapters.get(adapter_name)
        if ad is not None:
            ad.placements.discard(backend_id)
            ad.loaded_version.pop(backend_id, None)

    def acquire(self, backend_id: str, adapter_name: str | None = None) -> None:
        be = self.backends[backend_id]
        be.inflight += 1
        be.total_requests += 1
        if adapter_name and adapter_name in be.adapters:
            self._clock += 1
            be.adapter_clock[adapter_name] = self._clock  # serving touches the LRU clock

    def release(self, backend_id: str, *, failed: bool = False) -> None:
        be = self.backends.get(backend_id)
        if be is None:
            return
        be.inflight = max(0, be.inflight - 1)
        if failed:
            be.total_failures += 1

    # ---- balancing policy (the nginx upstream) ----
    def healthy_for_base(self, base_model: str, *, exclude: set[str] | None = None) -> list[Backend]:
        exclude = exclude or set()
        return [
            b
            for b in self.backends.values()
            if b.healthy and b.base_model == base_model and b.id not in exclude
        ]

    def pick_for_base(self, base_model: str, *, exclude: set[str] | None = None) -> Backend:
        """Least-outstanding-requests pick among healthy backends serving ``base_model``
        (used for raw base-model generation, no adapter)."""
        cands = self.healthy_for_base(base_model, exclude=exclude)
        if not cands:
            raise NoCapacityError(f"no healthy backend serving base model {base_model!r}")
        # Prefer backends below their in-flight cap so we don't pile more work onto an already
        # saturated GPU while a healthy one sits idle. Only when EVERY candidate is saturated do we
        # fall back to the least-loaded saturated one (better to queue than to drop the request).
        unsaturated = [b for b in cands if not b.saturated]
        pool = unsaturated or cands
        return min(pool, key=lambda b: (b.inflight, b.total_requests))

    def pick_backend(self, adapter_name: str, *, exclude: set[str] | None = None) -> PlacementDecision:
        """Pick a backend to serve ``adapter_name`` (loading/evicting if needed). ``exclude`` is
        the set of backend ids already tried this request (for retry/failover)."""
        ad = self.adapters.get(adapter_name)
        if ad is None:
            raise NoCapacityError(f"unknown adapter {adapter_name!r}")
        exclude = exclude or set()

        # 1) warm backends that already host the adapter — least in-flight, but avoid driving an
        # already-saturated backend further into overload: prefer warm backends below their cap and
        # only fall back to a saturated warm one when every warm backend is saturated.
        warm = [
            b
            for b in self.backends.values()
            if b.healthy and b.id not in exclude and adapter_name in b.adapters
        ]
        if warm:
            unsaturated = [b for b in warm if not b.saturated]
            pool = unsaturated or warm
            return PlacementDecision(min(pool, key=lambda b: (b.inflight, b.total_requests)))

        # 2) base-model backends with a free LoRA slot — fewest adapters, then least in-flight.
        cands = self.healthy_for_base(ad.base_model, exclude=exclude)
        if not cands:
            raise NoCapacityError(
                f"no healthy backend serving base model {ad.base_model!r} for adapter {adapter_name!r}"
            )
        free = [b for b in cands if b.free_lora_slots > 0]
        if free:
            be = min(free, key=lambda b: (len(b.adapters), b.inflight))
            return PlacementDecision(be, needs_load=True)

        # 3) all at their LoRA cap — least-loaded backend, evict its LRU adapter.
        be = min(cands, key=lambda b: b.inflight)
        evict = self._lru_adapter(be, keep=adapter_name)
        return PlacementDecision(be, needs_load=True, evict=evict)

    def _lru_adapter(self, be: Backend, *, keep: str) -> str | None:
        loaded = [a for a in be.adapters if a != keep]
        if not loaded:
            return None
        return min(loaded, key=lambda a: be.adapter_clock.get(a, 0))

    def plan_placements(self, adapter_name: str) -> list[PlacementDecision]:
        """Decide where to eagerly place an adapter to reach its desired ``replicas`` count.
        Returns one load decision per backend that should newly host the adapter."""
        ad = self.adapters[adapter_name]
        want = max(1, ad.replicas)
        have = {b for b in ad.placements if b in self.backends and self.backends[b].healthy}
        decisions: list[PlacementDecision] = []
        exclude = set(have)
        while len(have) + len(decisions) < want:
            cands = [
                b
                for b in self.healthy_for_base(ad.base_model)
                if b.id not in exclude and adapter_name not in b.adapters
            ]
            if not cands:
                break
            free = [b for b in cands if b.free_lora_slots > 0]
            pool = free or cands
            be = min(pool, key=lambda b: (len(b.adapters), b.inflight))
            evict = None if be.free_lora_slots > 0 else self._lru_adapter(be, keep=adapter_name)
            decisions.append(PlacementDecision(be, needs_load=True, evict=evict))
            exclude.add(be.id)
        return decisions

    # ---- reporting ----
    def snapshot(self) -> dict:
        backends = [b.snapshot() for b in self.backends.values()]
        healthy = [b for b in self.backends.values() if b.healthy]
        return {
            "backends": backends,
            "adapters": [a.snapshot() for a in self.adapters.values()],
            "summary": {
                "backends": len(self.backends),
                "healthy_backends": len(healthy),
                "adapters": len(self.adapters),
                "inflight": sum(b.inflight for b in self.backends.values()),
                "capacity": sum(b.max_concurrency for b in healthy),
                "base_models": sorted({b.base_model for b in self.backends.values()}),
                "cost_per_hour": round(sum(b.cost_per_hour for b in healthy), 4),
            },
        }
