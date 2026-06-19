"""Test package marker so ``from tests._helpers...`` resolves regardless of test-time cwd
(some tests monkeypatch.chdir, which broke namespace-package resolution of ``tests``)."""
