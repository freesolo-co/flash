"""privacy-safe packaged serving startup instrumentation."""

from __future__ import annotations

from types import SimpleNamespace

from flash.serve.app import progress


def test_filesystem_usage_emits_exact_mib_counters_without_paths(monkeypatch, capsys) -> None:
    cache_root = "/private/customer/cache-sentinel"
    calls: list[object] = []
    usages = {
        "/": SimpleNamespace(
            total=10 * progress._MIB + 17,
            used=4 * progress._MIB + 23,
            free=6 * progress._MIB - 6,
        ),
        cache_root: SimpleNamespace(
            total=20 * progress._MIB + 1,
            used=7 * progress._MIB + 2,
            free=13 * progress._MIB - 1,
        ),
    }

    def disk_usage(target):
        calls.append(target)
        return usages[target]

    monkeypatch.setattr(progress.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(progress, "_STARTED_AT", 10.0)
    monkeypatch.setattr(progress.time, "perf_counter", lambda: 11.25)

    progress.emit_filesystem_usage("cache-prepared", cache_root)

    assert calls == ["/", cache_root]
    output = capsys.readouterr().out
    assert output == (
        "flash-serving boot elapsed=1.250s phase=filesystem-usage "
        'stage="cache-prepared" root_total_mib="10" root_used_mib="4" '
        'root_free_mib="5" cache_total_mib="20" cache_used_mib="7" '
        'cache_free_mib="12"\n'
    )
    assert cache_root not in output


def test_filesystem_usage_reports_only_allowlisted_unavailable_status(monkeypatch, capsys) -> None:
    cache_root = "/private/customer/cache-sentinel"
    calls: list[object] = []

    def unavailable(target):
        calls.append(target)
        raise OSError(f"secret failure for {target}")

    monkeypatch.setattr(progress.shutil, "disk_usage", unavailable)
    monkeypatch.setattr(progress, "_STARTED_AT", 20.0)
    monkeypatch.setattr(progress.time, "perf_counter", lambda: 20.5)

    progress.emit_filesystem_usage("serving-ready", cache_root)

    assert calls == ["/", cache_root]
    output = capsys.readouterr().out
    assert output == (
        "flash-serving boot elapsed=0.500s phase=filesystem-usage "
        'stage="serving-ready" root_status="unavailable" cache_status="unavailable"\n'
    )
    assert cache_root not in output
    assert "secret failure" not in output
