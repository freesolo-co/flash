"""Reviewed compatibility changes for one pinned DeepInfer source revision."""

import hashlib
from pathlib import Path

PATCHES = {
    "deepinfer/model_executor/models/v4_modules/mlp.py": (
        "2541bcc83b178a9e2f18b387b301d60111d6d603e18e2d31df38196b03d290dd",
        [
            (
                "def _convert_mega_moe_weight(src_l1: nn.Parameter, src_l2: nn.Parameter) -> tuple[nn.Parameter, nn.Parameter]:\n    has_sf = src_l1.dtype != torch.bfloat16\n    if has_sf:\n        (l1, l1_sf), (l2, l2_sf) = deep_gemm.transform_weights_for_mega_moe((src_l1, src_l1.sf), (src_l2, src_l2.sf))\n    else:\n        l1, l2 = deep_gemm.transform_weights_for_mega_moe(src_l1, src_l2)\n    l1 = nn.Parameter(l1.clone(), requires_grad=False)\n    l2 = nn.Parameter(l2.clone(), requires_grad=False)\n    if has_sf:\n        l1.sf = l1_sf.clone()\n        l2.sf = l2_sf.clone()\n    for x in [src_l1, src_l2] + ([src_l1.sf, src_l2.sf] if has_sf else []):\n        x.untyped_storage().resize_(0)\n    return l1, l2\n",
                "def _convert_mega_moe_weight(src_l1: nn.Parameter, src_l2: nn.Parameter) -> tuple[nn.Parameter, nn.Parameter]:\n    from flash.serving.deepinfer.native.fp8 import convert_mega_moe_weight\n    return convert_mega_moe_weight(src_l1, src_l2)\n",
            ),
            (
                '        self.use_fp4_weight = cfg.get("mxfp4", False)\n',
                '        self.use_fp4_weight = cfg.get("mxfp4", False)\n        from flash.serving.deepinfer.native.fp8 import widen_expert\n        if self.use_fp4_weight:\n            for index, linear in enumerate((moe_module.linear1, moe_module.linear2)):\n                linear.weight = widen_expert(linear.weight, interleave=index == 0)\n            self.use_fp4_weight = False\n        assert moe_module.linear1.weight.dtype == torch.float8_e4m3fn\n        assert moe_module.linear2.weight.dtype == torch.float8_e4m3fn\n',
            ),
            (
                "        if self.use_fp4_weight:\n            # HF loading fills packed FP4 directly; dummy materialization converts it beforehand.",
                "        if self.use_fp4_weight or moe_module.linear1.weight.ndim == 3:\n            # HF loading fills packed FP4 directly; dummy materialization converts it beforehand.",
            ),
        ],
    ),
    "deepinfer/model_executor/utils.py": (
        "171451e25c45116994751900758ae103609b1e3818d47d8146f7da0d18867c83",
        [
            (
                '    if hasattr(param, "sf"):\n        p.sf = param.sf\n',
                '    if hasattr(param, "sf"):\n        p.sf = param.sf\n    if getattr(param, "_flash_interleaved", False):\n        p._flash_interleaved = True\n',
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
