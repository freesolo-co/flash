"""The hosted B200 tiers depend on a GDN prefill kernel that downgrades SILENTLY.

vllm 0.23.0 resolves FlashInfer GDN prefill unconditionally on SM90, but on SM10.x (Blackwell) it
additionally requires an intact ``nvidia-cutlass-dsl-libs-cu13`` install. When that term is false it
emits one ``warning_once`` and returns Triton -- no raise. Every hosted base is a Qwen3 GDN-hybrid,
so an unrepaired B200 boots green, serves correct output, reports ok:True, and bills the Blackwell
rate while running slower than the H200 it replaced.

These tests pin the reporting contract that makes that state detectable: the report must carry the
resolver's OWN answer, and it must say None rather than guess when it cannot reach the resolver.
"""

from flash.serving.src.engine.support import gdn_prefill_backend_report


def test_report_carries_the_three_fields_the_canary_asserts():
    report = gdn_prefill_backend_report()
    assert set(report) == {"resolved", "libs_cu13_intact", "compute_capability"}


def test_unreachable_resolver_reports_none_rather_than_a_backend():
    """vLLM is not installed in the offline test env, which is the same shape as the resolver having
    moved. An unknown backend must never read as a proven one: None is the honest answer, and the
    canary treats it as a FAILED assertion rather than a pass."""
    report = gdn_prefill_backend_report()
    assert report["resolved"] is None
    assert report["libs_cu13_intact"] is None


def test_report_never_raises_without_cuda_or_vllm():
    """_health must stay callable on a degraded container, so this probe swallows its own failures.
    A probe that raised would take down the only signal that reports the degraded tier."""
    assert isinstance(gdn_prefill_backend_report(), dict)
