import json
import sys
import time

import pytest

from flash.benchmarks import command_hooks
from flash.benchmarks.evidence import Recorder, load_events
from flash.benchmarks.managed import Context


def test_command_bridge_retains_each_evaluation_and_imports_events(tmp_path, case, request_data):
    script = tmp_path / "evaluate.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "context = json.loads(Path(sys.argv[1]).read_text())\n"
        "result = {'value': 0.75, 'samples': 20, 'events': ["
        "{'kind': 'request', 'start_s': 0.0, 'end_s': 1.0, 'customer': 'alice', "
        "'adapter': 'run/final', 'success': True}]}\n"
        "Path(context['result_path']).write_text(json.dumps(result))\n"
    )
    case.configuration = {"evaluate_command": [sys.executable, str(script)]}
    recorder = Recorder(tmp_path / "events.jsonl", time.monotonic())
    context = Context(request_data, tmp_path, recorder, case)
    targets = [(object(), "run/final", "alice")]
    assert command_hooks.evaluate(context, targets) == (0.75, 20)
    assert command_hooks.evaluate(context, targets) == (0.75, 20)
    assert len(load_events(tmp_path / "events.jsonl")) == 2
    assert (tmp_path / "evaluate-0/result.json").exists()
    assert (tmp_path / "evaluate-1/result.json").exists()
    saved = json.loads((tmp_path / "evaluate-0/context.json").read_text())
    assert saved["targets"] == [{"adapter": "run/final", "customer": "alice"}]


def test_missing_agent_evaluator_rejected_before_prepare(tmp_path, case, request_data):
    case.workload = "long_multi_turn"
    case.configuration = {
        f"{name}_command": ["python", "operator.py"] for name in ("prepare", "collect", "cleanup")
    }
    context = Context(
        request_data, tmp_path, Recorder(tmp_path / "events.jsonl", time.monotonic()), case
    )
    with pytest.raises(ValueError, match="evaluate_command"):
        command_hooks.validate(context)


def test_collector_cannot_seal_its_own_completion(tmp_path, case, request_data):
    script = tmp_path / "collect.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "ctx = json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(ctx['result_path']).write_text(json.dumps({'events': ["
        "{'kind': 'complete', 'at_s': 1.0, 'outcome': 'succeeded'}]}))\n"
    )
    case.configuration = {"collect_command": [sys.executable, str(script)]}
    context = Context(
        request_data, tmp_path, Recorder(tmp_path / "events.jsonl", time.monotonic()), case
    )
    with pytest.raises(ValueError, match="only the managed driver"):
        command_hooks.collect(context)
