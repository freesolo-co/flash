"""Reviewed compatibility changes for one pinned DeepInfer source revision."""

import hashlib
from pathlib import Path

PATCHES = {
    "deepinfer/worker/base.py": (
        "40e6ca29aebd9d475eaad4b86326783f76dba2b73feaa51d86a200b7704a05ed",
        [
            (
                'torch._C._accelerator_setAllocatorSettings("expandable_segments:True")',
                # GPU Engram storage shares CUDA allocations between local ranks.
                # Expandable segments require pidfd_getfd, which Modal disallows.
                'torch._C._accelerator_setAllocatorSettings("expandable_segments:False")',
            ),
            (
                "torch.cuda.set_device(self.device)",
                (
                    "torch.cuda.set_device(self.device)\n"
                    "        from flash.serving.deepinfer.native.engram_transport import install\n"
                    "        install()"
                ),
            ),
        ],
    ),
    "csrc/engine/pybind_sequence.cpp": (
        "872775774022bb27fd4b1fade37ec9f6fe9ec4c3051844206e3ec36c8103a9ce",
        [
            (
                '    bind_sequence_property(sequence, "state_id",',
                (
                    '    bind_sequence_readonly(sequence, "block_table", [](const Sequence &seq) -> const auto & { return seq.cache.block_table; });\n'
                    '    bind_sequence_property(sequence, "state_id",'
                ),
            )
        ],
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
