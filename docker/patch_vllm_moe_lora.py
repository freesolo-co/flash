#!/usr/bin/env python3
"""apply the exact vllm 0.23.0 moe lora activation backport."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_VLLM_VERSION: Final = "0.23.0"


class RepairError(RuntimeError):
    """the installed vllm tree is not an approved repair state."""


@dataclass(frozen=True)
class Target:
    relative_path: str
    pre_hash: str
    post_hash: str


TARGETS: Final = (
    Target(
        "vllm/model_executor/layers/fused_moe/experts/lora_context.py",
        "050992817a1fe2a3e7604a94f34d35cc03b6b074dc54abc83e4a9e0ba9d1dbf8",
        "7a8911cfae2d7d59f38399fd402c41b341b78919e8772fae70329941c2007b20",
    ),
    Target(
        "vllm/model_executor/layers/fused_moe/experts/triton_moe.py",
        "b197ddb0606380873250d284fd56acd0c4ffad4fe4c9ffa3bb0ed4b8bf49f271",
        "c166d7eed1b1715fc717f4f040fd4c8623ed281ce5451d02efb6145fd9156b84",
    ),
    Target(
        "vllm/model_executor/layers/fused_moe/modular_kernel.py",
        "b1e73b77322363d686c524d01e482c7eabb50f46fa8d9796eebcb7976acb8aa1",
        "8a551510150c7be4dab6510f7404bf3f6b1c08e32796aa9461e55b3777245521",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RepairError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new)


def _replace_exactly(source: str, old: str, new: str, count: int, label: str) -> str:
    actual = source.count(old)
    if actual != count:
        raise RepairError(f"{label}: expected {count} anchors, found {actual}")
    return source.replace(old, new)


def _patch_lora_context(source: str) -> str:
    anchor = "    local_token_lora_mapping: torch.Tensor | None = None\n"
    replacement = (
        anchor
        + """
    # unquantized hidden states are stashed by the modular kernel with the
    # router-input weighting expected by the expert path. apply_w13_lora uses
    # them only when no all-to-all dispatch changes the activation layout.
    original_hidden_states: torch.Tensor | None = None
"""
    )
    return _replace_once(source, anchor, replacement, "lora context")


def _patch_triton_moe(source: str) -> str:
    class_start = source.find("class TritonExperts(")
    class_end = source.find("\n\nclass TritonWNA16Experts", class_start)
    if class_start < 0 or class_end < 0:
        raise RepairError("triton moe: exact TritonExperts class bounds not found")
    prefix, body, suffix = source[:class_start], source[class_start:class_end], source[class_end:]

    activation_anchor = """    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
"""
    activation_replacement = """    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # defer quantization only when lora uses dp/ep all-to-all dispatch.
        return (
            self._lora_context is not None
            and self.quant_dtype is not None
            and self.moe_config.moe_parallel_config.use_all2all_kernels
        )

    @staticmethod
    def _supports_current_device() -> bool:
"""
    body = _replace_once(
        body,
        activation_anchor,
        activation_replacement,
        "triton moe defer property",
    )

    quant_anchor = """        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
"""
    quant_replacement = """        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        # the all-to-all path deferred activation quantization to this kernel.
        # retain gathered unquantized activations for lora and quantize a copy
        # for the base moe gemm.
        lora_unquantized_hidden_states: torch.Tensor | None = None
        if self.expects_unquantized_inputs:
            assert a1q_scale is None
            lora_unquantized_hidden_states = hidden_states
            hidden_states, a1q_scale = moe_kernel_quantize_input(
                hidden_states,
                self.a1_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
                quantization_emulation=self.quantization_emulation,
            )

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
"""
    body = _replace_once(body, quant_anchor, quant_replacement, "triton moe quantization split")

    selection_anchor = """        token_lora_mapping = None
        lora_context = self._lora_context

        def _base_w13_fn():
"""
    selection_replacement = """        token_lora_mapping = None
        lora_context = self._lora_context
        if lora_unquantized_hidden_states is not None:
            lora_x = lora_unquantized_hidden_states
        elif (
            lora_context is not None
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
            and lora_context.original_hidden_states is not None
            and lora_context.original_hidden_states.shape[0] == hidden_states.shape[0]
        ):
            lora_x = lora_context.original_hidden_states
        else:
            lora_x = hidden_states

        def _base_w13_fn():
"""
    body = _replace_once(body, selection_anchor, selection_replacement, "triton moe lora input")
    body = _replace_exactly(
        body,
        "                    x=hidden_states,\n",
        "                    x=lora_x,\n",
        2,
        "triton moe w13 calls",
    )
    return prefix + body + suffix


def _patch_modular_kernel(source: str) -> str:
    prepare_anchor = """        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )

        fused_out = self._fused_experts(
"""
    prepare_replacement = """        # preserve the exact local unquantized activation before _prepare can
        # replace routing metadata with gathered all-to-all tensors. the local
        # stash is used only on non-all-to-all paths; gathered paths use the
        # activation returned by _prepare.
        lora_ctx = getattr(self.fused_experts, "_lora_context", None)
        lora_hidden_states = hidden_states
        if lora_ctx is not None and apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            lora_hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )

        if lora_ctx is not None:
            lora_ctx.original_hidden_states = lora_hidden_states

        fused_out = self._fused_experts(
"""
    source = _replace_once(
        source,
        prepare_anchor,
        prepare_replacement,
        "modular kernel context stash",
    )
    fused_anchor = """        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_tokens_meta=expert_tokens_meta,
            output_alias=output,
        )
"""
    fused_replacement = """        try:
            fused_out = self._fused_experts(
                in_dtype=hidden_states.dtype,
                a1q=a1q,
                a1q_scale=a1q_scale,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_tokens_meta=expert_tokens_meta,
                output_alias=output,
            )
        finally:
            if lora_ctx is not None:
                lora_ctx.original_hidden_states = None
"""
    return _replace_once(
        source,
        fused_anchor,
        fused_replacement,
        "modular kernel protected context clear",
    )


def _transform_sources(sources: dict[str, bytes]) -> dict[str, bytes]:
    patchers = {
        TARGETS[0].relative_path: _patch_lora_context,
        TARGETS[1].relative_path: _patch_triton_moe,
        TARGETS[2].relative_path: _patch_modular_kernel,
    }
    transformed: dict[str, bytes] = {}
    for target in TARGETS:
        try:
            source = sources[target.relative_path].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepairError(f"{target.relative_path}: source is not utf-8") from exc
        transformed[target.relative_path] = patchers[target.relative_path](source).encode()
    return transformed


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise RepairError(f"semantic verification: expected one {name} class")
    return matches[0]


def _method_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RepairError(f"semantic verification: expected one {class_node.name}.{name}")
    return matches[0]


def _keyword_name(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Name):
            return keyword.value.id
    return None


def _verify_semantics(sources: dict[str, bytes]) -> None:
    trees: dict[str, ast.Module] = {}
    for relative_path, data in sources.items():
        try:
            source = data.decode("utf-8")
            compile(source, relative_path, "exec", dont_inherit=True)
            trees[relative_path] = ast.parse(source, filename=relative_path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise RepairError(f"{relative_path}: patched source does not compile") from exc

    context_tree = trees[TARGETS[0].relative_path]
    context = _class_node(context_tree, "MoELoRAContext")
    context_fields = [
        node
        for node in context.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "original_hidden_states"
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    if len(context_fields) != 1:
        raise RepairError("semantic verification: original_hidden_states field is not exact")

    triton_tree = trees[TARGETS[1].relative_path]
    triton = _class_node(triton_tree, "TritonExperts")
    defer = _method_node(triton, "expects_unquantized_inputs")
    if not any(
        isinstance(node, ast.Attribute) and node.attr == "use_all2all_kernels"
        for node in ast.walk(defer)
    ):
        raise RepairError("semantic verification: all-to-all defer condition is missing")
    if not any(
        isinstance(node, ast.Attribute) and node.attr == "_lora_context" for node in ast.walk(defer)
    ):
        raise RepairError("semantic verification: lora defer condition is missing")

    apply = _method_node(triton, "apply")
    quant_calls = [
        node
        for node in ast.walk(apply)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "moe_kernel_quantize_input"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "hidden_states"
    ]
    if len(quant_calls) != 1:
        raise RepairError("semantic verification: exact base moe requantization is missing")

    w13_calls = [
        node
        for node in ast.walk(apply)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_w13_lora"
    ]
    if len(w13_calls) != 2 or any(_keyword_name(call, "x") != "lora_x" for call in w13_calls):
        raise RepairError("semantic verification: both w13 calls must use lora_x")

    w2_calls = [
        node
        for node in ast.walk(apply)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_w2_lora"
    ]
    if len(w2_calls) != 2 or any(
        _keyword_name(call, "x") != "intermediate_cache2" for call in w2_calls
    ):
        raise RepairError("semantic verification: w2 behavior changed")

    lora_x_assignments = [
        node
        for node in ast.walk(apply)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "lora_x" for target in node.targets)
    ]
    if len(lora_x_assignments) != 3 or any(
        isinstance(node.value, ast.Call) for node in lora_x_assignments
    ):
        raise RepairError("semantic verification: lora_x must not use a raw cast")
    lora_x_sources = {
        ast.unparse(node.value)
        for node in lora_x_assignments
        if not isinstance(node.value, ast.Call)
    }
    if lora_x_sources != {
        "lora_unquantized_hidden_states",
        "lora_context.original_hidden_states",
        "hidden_states",
    }:
        raise RepairError("semantic verification: lora_x unquantized sources are incomplete")
    local_stash_guards = [
        node
        for node in ast.walk(apply)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "lora_x" for target in child.targets
            )
            and ast.unparse(child.value) == "lora_context.original_hidden_states"
            for child in node.body
        )
    ]
    if len(local_stash_guards) != 1 or not any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "use_all2all_kernels"
        for node in ast.walk(local_stash_guards[0].test)
    ):
        raise RepairError(
            "semantic verification: local activation stash must be disabled for all-to-all"
        )

    modular_tree = trees[TARGETS[2].relative_path]
    modular = _class_node(modular_tree, "FusedMoEKernelModularImpl")
    modular_apply = _method_node(modular, "apply")
    assignments = [
        node
        for node in ast.walk(modular_apply)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "original_hidden_states"
            for target in node.targets
        )
    ]
    assigned_names = [
        node.value.id if isinstance(node.value, ast.Name) else None for node in assignments
    ]
    assigned_constants = [
        node.value.value if isinstance(node.value, ast.Constant) else object()
        for node in assignments
    ]
    if assigned_names.count("lora_hidden_states") != 1 or assigned_constants.count(None) != 1:
        raise RepairError("semantic verification: modular activation stash or clear is missing")

    local_activation_assignments = [
        node
        for node in ast.walk(modular_apply)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "lora_hidden_states"
            for target in node.targets
        )
    ]
    local_activation_sources = {ast.unparse(node.value) for node in local_activation_assignments}
    if local_activation_sources != {
        "hidden_states",
        "hidden_states * topk_weights.to(hidden_states.dtype)",
    }:
        raise RepairError(
            "semantic verification: router-weighted unquantized activation stash is missing"
        )
    router_weight_branches = [
        node
        for node in ast.walk(modular_apply)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Name) and child.id == "apply_router_weight_on_input"
            for child in ast.walk(node.test)
        )
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "lora_hidden_states"
                for target in child.targets
            )
            and ast.unparse(child.value) == "hidden_states * topk_weights.to(hidden_states.dtype)"
            for child in node.body
        )
    ]
    if len(router_weight_branches) != 1:
        raise RepairError(
            "semantic verification: router-weighted unquantized activation stash is missing"
        )

    prepare_assignments = [
        node
        for node in modular_apply.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_prepare"
    ]
    stash_assignments = [
        node
        for node in modular_apply.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "lora_hidden_states"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "hidden_states"
    ]
    if len(prepare_assignments) != 1 or len(stash_assignments) != 1:
        raise RepairError(
            "semantic verification: exact prepare and local stash ordering is missing"
        )
    if modular_apply.body.index(stash_assignments[0]) >= modular_apply.body.index(
        prepare_assignments[0]
    ):
        raise RepairError("semantic verification: local activation stash must precede prepare")

    protected_calls = []
    for node in ast.walk(modular_apply):
        if not isinstance(node, ast.Try) or len(node.body) != 1:
            continue
        statement = node.body[0]
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "_fused_experts":
            continue
        clears = [
            child
            for final_statement in node.finalbody
            for child in ast.walk(final_statement)
            if isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "original_hidden_states"
                for target in child.targets
            )
            and isinstance(child.value, ast.Constant)
            and child.value.value is None
        ]
        if len(clears) == 1:
            protected_calls.append((node, clears[0]))
    if len(protected_calls) != 1:
        raise RepairError(
            "semantic verification: activation clear must be in the fused-experts finally"
        )


def _paths(distribution: importlib.metadata.Distribution) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for target in TARGETS:
        path = Path(distribution.locate_file(target.relative_path)).resolve(strict=True)
        if not path.is_file():
            raise RepairError(f"{target.relative_path}: installed target is not a regular file")
        paths[target.relative_path] = path
    return paths


def _read(paths: dict[str, Path]) -> dict[str, bytes]:
    return {relative_path: path.read_bytes() for relative_path, path in paths.items()}


def _classify(sources: dict[str, bytes]) -> str:
    observed = {target.relative_path: _sha256(sources[target.relative_path]) for target in TARGETS}
    if all(observed[target.relative_path] == target.pre_hash for target in TARGETS):
        return "pristine"
    if all(observed[target.relative_path] == target.post_hash for target in TARGETS):
        return "patched"
    known = {
        target.relative_path: observed[target.relative_path] in {target.pre_hash, target.post_hash}
        for target in TARGETS
    }
    state = "mixed" if all(known.values()) else "unknown"
    details = ", ".join(f"{path}={digest}" for path, digest in observed.items())
    raise RepairError(f"vllm source state is {state}; refusing writes: {details}")


def _require_post_hashes(sources: dict[str, bytes]) -> None:
    for target in TARGETS:
        actual = _sha256(sources[target.relative_path])
        if actual != target.post_hash:
            raise RepairError(
                f"{target.relative_path}: transformed hash {actual} does not match {target.post_hash}"
            )


def _stage_and_replace(paths: dict[str, Path], sources: dict[str, bytes]) -> None:
    staged: dict[str, Path] = {}
    try:
        for relative_path, destination in paths.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            staged[relative_path] = temporary
            try:
                os.fchmod(descriptor, stat.S_IMODE(destination.stat().st_mode))
            except BaseException:
                os.close(descriptor)
                raise
            with os.fdopen(descriptor, "wb") as output:
                output.write(sources[relative_path])
                output.flush()
                os.fsync(output.fileno())

        for relative_path, destination in paths.items():
            os.replace(staged[relative_path], destination)
            staged.pop(relative_path)

        for directory in sorted({path.parent for path in paths.values()}):
            descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def repair(*, verify_only: bool = False) -> str:
    try:
        distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RepairError("vllm is not installed") from exc
    if distribution.version != _VLLM_VERSION:
        raise RepairError(
            f"expected vllm {_VLLM_VERSION}, found {distribution.version}; refusing writes"
        )

    paths = _paths(distribution)
    sources = _read(paths)
    state = _classify(sources)
    if state == "patched":
        _verify_semantics(sources)
        return "verified exact vllm 0.23.0 pr42120 moe lora backport"
    if verify_only:
        raise RepairError("vllm 0.23.0 is pristine; required moe lora backport is absent")

    transformed = _transform_sources(sources)
    _require_post_hashes(transformed)
    _verify_semantics(transformed)
    _stage_and_replace(paths, transformed)

    installed = _read(paths)
    if _classify(installed) != "patched":
        raise RepairError("post-write vllm source verification failed")
    _verify_semantics(installed)
    return "applied and verified exact vllm 0.23.0 pr42120 moe lora backport"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="require the exact patched state without changing installed files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(repair(verify_only=args.verify))
    except (OSError, RepairError) as exc:
        print(f"vllm moe lora repair failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
