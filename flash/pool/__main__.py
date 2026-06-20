"""``flash-pool`` operator CLI: run the router, inspect it, plan a fleet, or run a reward worker."""

from __future__ import annotations

import argparse
import importlib
import json
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    from flash.pool.server import serve

    print(f"[flash-pool] router on {args.host}:{args.port}", flush=True)
    serve(host=args.host, port=args.port)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    import httpx

    r = httpx.get(f"{args.url.rstrip('/')}/pool/status", timeout=15.0)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from flash.pool.config import PoolPlan
    from flash.pool.provision import plan_summary

    plan = PoolPlan.from_toml(args.config)
    print(json.dumps(plan_summary(plan), indent=2))
    return 0


def _cmd_reward(args: argparse.Namespace) -> int:
    import uvicorn

    from flash.pool.rewards import create_reward_app

    mod_name, _, attr = args.scorer.partition(":")
    if not attr:
        print("scorer must be 'module.path:callable'", file=sys.stderr)
        return 2
    scorer = getattr(importlib.import_module(mod_name), attr)
    app = create_reward_app(scorer, reward_id=args.reward_id)
    print(f"[flash-pool] reward worker [{args.reward_id}] on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flash-pool", description="Flash shared rollout pool")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the rollout-pool router")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8077)
    s.set_defaults(func=_cmd_serve)

    st = sub.add_parser("status", help="print router pool status")
    st.add_argument("--url", default="http://127.0.0.1:8077")
    st.set_defaults(func=_cmd_status)

    pl = sub.add_parser("plan", help="dry-run a pool plan TOML (capacity + adapter slots)")
    pl.add_argument("config", help="path to a pool plan TOML ([[pool]] entries)")
    pl.set_defaults(func=_cmd_plan)

    rw = sub.add_parser("reward", help="run a reward worker wrapping a python scorer")
    rw.add_argument("scorer", help="'module.path:callable' — scorer(prompts, completions, info)->scores")
    rw.add_argument("--reward-id", default="default")
    rw.add_argument("--host", default="0.0.0.0")
    rw.add_argument("--port", type=int, default=8078)
    rw.set_defaults(func=_cmd_reward)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
