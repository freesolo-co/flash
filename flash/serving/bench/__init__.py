"""Hosted-serving B200 capacity benchmark: isolated, base-only, non-production.

Never imported by the production serving app or router. The Modal app that runs it lives in
``scripts/bench_b200_capacity.py`` and deploys under its own name, so this package cannot reach the
production deployment.
"""
