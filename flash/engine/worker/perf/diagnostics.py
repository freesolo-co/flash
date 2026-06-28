"""GPU memory sampling + live telemetry for the fine-tuning worker's run logs / status."""

from __future__ import annotations

import csv
import time


def _reset_peak_gpu() -> None:
    """Reset the CUDA peak-memory counter."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_gpu_gb() -> float:
    """Peak torch-allocated CUDA memory (binary GiB) since last reset; 0.0 if no CUDA.

    bnb paged optimizer state is NOT counted here — use _GpuPeakSampler for the true device peak."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**3), 3)
    except Exception:
        pass
    return 0.0


class _GpuPeakSampler:
    """Background sampler of true device memory via cuda.mem_get_info (includes bnb managed pages)."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_used = 0
        self._stop = False
        self._thread = None

    def _run(self):
        import torch

        while not self._stop:
            try:
                free, total = torch.cuda.mem_get_info()
                self.peak_used = max(self.peak_used, total - free)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        try:
            import threading

            import torch

            if not torch.cuda.is_available():
                return self
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            pass
        return self

    def stop_gb(self) -> float:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        return round(self.peak_used / (1024**3), 3)


def _float_or_none(value) -> float | None:
    try:
        text = str(value).strip()
        if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> int | None:
    num = _float_or_none(value)
    return int(num) if num is not None else None


def _round_gb_from_mib(value) -> float | None:
    num = _float_or_none(value)
    if num is None:
        return None
    return round(num / 1024.0, 3)


def _clean_diag(diag: dict) -> dict:
    return {k: v for k, v in diag.items() if v is not None and v != ""}


def _query_nvidia_gpu() -> dict:
    import subprocess

    fields = [
        "index",
        "uuid",
        "driver_version",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.used",
        "memory.free",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=8.0,  # nvidia-smi diag timeout (fixed; flash is fully managed)
    )
    raw = (out.stdout or out.stderr).strip()
    if out.returncode != 0:
        return {"nvidia_smi_err": raw[:300]}
    rows = list(csv.reader(raw.splitlines()))
    if not rows:
        return {}
    first = [cell.strip() for cell in rows[0]]
    row = dict(zip(fields, first, strict=False))
    diag = {
        "index": _int_or_none(row.get("index")),
        "uuid": row.get("uuid"),
        "driver_version": row.get("driver_version"),
        "device_name": row.get("name"),
        "gpu_util_pct": _int_or_none(row.get("utilization.gpu")),
        "mem_util_pct": _int_or_none(row.get("utilization.memory")),
        "memory_total_gb": _round_gb_from_mib(row.get("memory.total")),
        "memory_used_gb": _round_gb_from_mib(row.get("memory.used")),
        "memory_free_gb": _round_gb_from_mib(row.get("memory.free")),
        "temperature_c": _int_or_none(row.get("temperature.gpu")),
        "power_w": _float_or_none(row.get("power.draw")),
        "power_limit_w": _float_or_none(row.get("power.limit")),
        "pstate": row.get("pstate"),
        "sm_clock_mhz": _int_or_none(row.get("clocks.sm")),
        "mem_clock_mhz": _int_or_none(row.get("clocks.mem")),
        "pcie_gen": _int_or_none(row.get("pcie.link.gen.current")),
        "pcie_width": _int_or_none(row.get("pcie.link.width.current")),
    }
    clean = _clean_diag(diag)
    clean["nvidia_smi"] = raw[:300]
    return clean


def _query_nvidia_processes() -> list[dict]:
    import subprocess

    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=8.0,  # nvidia-smi diag timeout (fixed; flash is fully managed)
    )
    if out.returncode != 0 or not out.stdout.strip():
        return []
    rows = []
    for row in csv.reader(out.stdout.splitlines()):
        if len(row) < 3:
            continue
        rows.append(
            _clean_diag(
                {
                    "pid": _int_or_none(row[0]),
                    "process_name": row[1].strip(),
                    "used_memory_gb": _round_gb_from_mib(row[2]),
                }
            )
        )
    return sorted(rows, key=lambda r: float(r.get("used_memory_gb") or 0.0), reverse=True)[:8]


def gpu_diagnostics(include_torch: bool = True) -> dict:
    """Collect live CUDA/GPU telemetry for run logs and status."""
    diag = {}
    if include_torch:
        try:
            import torch

            diag["torch"] = torch.__version__
            diag["torch_cuda"] = torch.version.cuda
            diag["cuda_available"] = torch.cuda.is_available()
            try:
                diag["device_count"] = torch.cuda.device_count()
                if torch.cuda.is_available():
                    diag["device_name"] = torch.cuda.get_device_name(0)
                    free, total = torch.cuda.mem_get_info()
                    diag["torch_memory_free_gb"] = round(free / (1024**3), 3)
                    diag["torch_memory_total_gb"] = round(total / (1024**3), 3)
                    diag["torch_memory_allocated_gb"] = round(
                        torch.cuda.memory_allocated() / (1024**3), 3
                    )
                    diag["torch_memory_reserved_gb"] = round(
                        torch.cuda.memory_reserved() / (1024**3), 3
                    )
            except Exception as e:
                diag["device_query_err"] = str(e)[:160]
        except Exception as e:
            diag["torch_import_err"] = str(e)[:160]
    try:
        diag.update(_query_nvidia_gpu())
        processes = _query_nvidia_processes()
        if processes:
            diag["processes"] = processes
    except Exception as e:
        diag["nvidia_smi_err"] = str(e)[:160]
    return _clean_diag(diag)
