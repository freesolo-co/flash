"""Flash's side of the verl integration: capability probing, checkpoints, child io, diagnostics,
parallelism.

This package shares a name with the `verl` distribution the training child imports, and that is
deliberate -- `flash.engine.worker.verl.capabilities` reads as what it is. It cannot shadow the real
package: a nested package is only importable as bare `verl` if its PARENT directory is on
`sys.path`, and nothing puts `flash/engine/worker/` there. The two paths that do get prepended for a
child are `shim_dir` (`<workdir>/shim`, a scratch directory the trainers write) and `code_dir`
(`/runcode/<prefix>`, the flash package root) -- neither is this directory. The child also runs in
its own interpreter (`FLASH_VERL_PYTHON`, `/opt/verl-venv`) where flash is not installed at all.

Keep it that way: putting `flash/engine/worker/` on `sys.path` would make `import verl` in this
process resolve here instead of to the trainer, which no test would catch because the offline suite
never imports the real verl.
"""
