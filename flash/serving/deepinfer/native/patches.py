"""Reviewed compatibility changes for one pinned DeepInfer source revision."""

import hashlib
from pathlib import Path

PATCHES = {
    "deepinfer/model_executor/models/v4_modules/mlp.py": (
        "2541bcc83b178a9e2f18b387b301d60111d6d603e18e2d31df38196b03d290dd",
        [],  # Keep the released native FP4 expert path unchanged.
    ),
    "deepinfer/model_executor/utils.py": (
        "171451e25c45116994751900758ae103609b1e3818d47d8146f7da0d18867c83",
        [],  # No FP8 conversion metadata is needed.
    ),
    "csrc/engine/pybind_scheduler.cpp": (
        "a81d71dbfe5a0d7ad5c0c79c104abd65ada2ab57b8073f339e0c13a5cb74a438",
        [
            (
                '        .def("add_sequence", &Scheduler::add_sequence, py::arg("seq"))',
                '        .def("detach_waiting_for_prefix_cache", [](Scheduler &self, const std::string &request_id) {\n            for (auto it = self.waiting.begin(); it != self.waiting.end(); ++it) {\n                if ((*it)->request.request_id == request_id) {\n                    auto seq = *it;\n                    self.waiting.erase(it);\n                    return seq;\n                }\n            }\n            throw std::runtime_error("Prefix source is not in the waiting queue");\n        }, py::arg("request_id"))\n        .def("add_sequence", &Scheduler::add_sequence, py::arg("seq"))',
            )
        ],
    ),
}


def apply_runtime_patches(root: Path) -> None:
    for name, (expected, changes) in PATCHES.items():
        path = root / name
        source = path.read_text()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(f"Unrecognized pinned DeepInfer source: {name}")
        for before, after in changes:
            if source.count(before) != 1:
                raise RuntimeError(f"Ambiguous DeepInfer patch: {name}")
            source = source.replace(before, after)
        path.write_text(source)
