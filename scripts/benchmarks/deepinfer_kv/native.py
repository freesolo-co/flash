"""Pinned DeepInfer experimental adapter. All operations run at a quiescent barrier.

Snapshots retain the full raw state slot, including SWA and compression state.
They are portable only between identical model/runtime/layout identities.
"""

import hashlib
import time

import numpy as np
import torch
from deepinfer._C.engine import RequestOutput
from deepinfer.sampling_params import SamplingParams


class Native:
    def __init__(self, llm, tokenizer, engine, rank):
        self.llm, self.tokenizer, self.engine, self.rank = llm, tokenizer, engine, rank
        self.sources = {}
        self.snapshots = {}
        self.cache = engine.driver_worker.cache_engine
        self.serial = 0
        import json
        from pathlib import Path

        proof = json.loads(Path(f"/results/loaded-rank-{rank}.json").read_text())
        identity = {
            "native_sha256": proof["native_sha256"],
            "block_shape": list(self.cache.gpu_blocks.shape[1:]),
            "state_shape": list(self.cache.gpu_states.shape[1:]),
            "layout": self.cache.kvcache_layout,
            "spec": 3,
            "ep": 4,
            "precision": "expanded-expert-fp8/native-main-fp4",
            "batch_invariant": True,
        }
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def sequence(self, tokens, output=128, enqueue=True):
        self.serial += 1
        params = SamplingParams(
            temperature=0, top_p=1, max_tokens=output, seq_seed=1234, eos_id=self.engine.eos_id
        )
        seq = self.engine.add_request(
            user_id="kv-ab",
            request_id=f"r{self.rank}-{self.serial}",
            sampling_params=params,
            prompt_token_ids=np.asarray(tokens, dtype=np.int32),
            multi_modal_data=None,
            enqueue=enqueue,
        )
        assert seq is not None
        return seq

    def drain(self, records=None):
        records = records if records is not None else {}
        started = time.perf_counter()
        while True:
            unfinished = self.engine.has_unfinished_requests()
            raw = self.engine.step() if unfinished else self.engine.flush_inflight_reqs()
            now = time.perf_counter()
            for pointer in raw:
                out = RequestOutput.clone(pointer)
                if out.request_id not in records:
                    continue
                row = records[out.request_id]
                row["tokens"].append(int(out.token_id))
                row["first_at"] = row["first_at"] or now
                if out.finished:
                    row["finished_at"] = now
                    row["finish_reason"] = str(out.finish_reason)
            if not unfinished:
                break
            assert time.perf_counter() - started < 300, "Native phase deadline"
        torch.cuda.synchronize()
        return records

    def warm(self, key, tokens, cut):
        source = self.sequence(tokens)
        source.chunk_checkpoints = [cut]
        assert self.engine.scheduler.try_allocate_seq(source)
        detached = False
        started = time.perf_counter()
        while True:
            assert source.num_cached_tokens <= cut
            if source.num_cached_tokens == cut and not detached:
                saved = self.engine.scheduler.detach_waiting_for_prefix_cache(source.request_id)
                assert saved.request_id == source.request_id
                detached = True
            if not self.engine.has_unfinished_requests():
                break
            self.engine.step()
            self.engine.flush_inflight_reqs()
            assert time.perf_counter() - started < 300
        assert detached
        self.sources[key] = source
        torch.cuda.synchronize()
        return {"warm_seconds": time.perf_counter() - started, "cached_tokens": cut}

    def drop(self, key):
        source = self.sources.pop(key, None)
        if source is not None:
            self.engine.scheduler.free_seq(source)

    def clear(self):
        for key in list(self.sources):
            self.drop(key)
        self.snapshots.clear()

    def snapshot(self, key):
        source = self.sources[key]
        blocks = torch.tensor(source.block_table, dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        kv = self.cache.gpu_blocks.index_select(0, blocks).cpu()
        state = self.cache.gpu_states[source.state_id].cpu()
        torch.cuda.synchronize()
        result = {
            "identity": self.identity,
            "kv": kv,
            "state": state,
            "tokens": source.data.get_token_ids(),
            "cached_tokens": source.num_cached_tokens,
            "swa_valid_start": source.swa_valid_start,
            "runtime": "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9",
            "model": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        }
        self.snapshots[key] = result
        return {
            "export_seconds": time.perf_counter() - started,
            "bytes": kv.numel() + state.numel(),
            "blocks": len(blocks),
            "state_bytes": state.numel(),
        }

    def restore(self, key, snapshot=None):
        snapshot = snapshot if snapshot is not None else self.snapshots[key]
        assert snapshot["runtime"] == "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
        assert snapshot["model"] == "1bae0e1088993b9b82de08f59de5be5a34e42170"
        assert snapshot["identity"] == self.identity, "Incompatible runtime/cache layout"
        self.drop(key)
        torch.cuda.synchronize()
        started = time.perf_counter()
        source = self.sequence(snapshot["tokens"], enqueue=False)
        assert self.engine.scheduler.try_allocate_seq(source)
        assert len(source.block_table) == snapshot["kv"].size(0)
        assert self.cache.gpu_states[source.state_id].shape == snapshot["state"].shape
        blocks = torch.tensor(source.block_table, dtype=torch.long, device="cuda")
        self.cache.gpu_blocks.index_copy_(0, blocks, snapshot["kv"].to("cuda"))
        self.cache.gpu_states[source.state_id].copy_(snapshot["state"])
        source.num_cached_tokens = snapshot["cached_tokens"]
        source.swa_valid_start = snapshot["swa_valid_start"]
        self.sources[key] = source
        torch.cuda.synchronize()
        return {"restore_seconds": time.perf_counter() - started}

    def batch(self, requests):
        records = {}
        phase_start = time.perf_counter()
        for request in requests:
            started = time.perf_counter()
            tokens = request["tokens"]
            key = request.get("key")
            source = self.sources.get(key)
            cached = 0
            if source is not None:
                cached = source.num_cached_tokens
                assert tokens[:cached] == source.data.get_token_ids()[:cached]
                meta = self.sequence(tokens, request.get("output", 128), enqueue=False)
                seq = self.engine.scheduler.fork_seq(
                    source,
                    None,
                    "_ab-" + meta.request_id,
                    meta.sampling_params,
                    source.so_matcher,
                    source.data.get_last_token_id(),
                    -1,
                    next(self.engine.seq_counter),
                    time.monotonic(),
                )
                seq.request_id = meta.request_id
                seq.data, seq.compressed_data = meta.data, meta.compressed_data
            else:
                seq = self.sequence(tokens, request.get("output", 128))
            records[seq.request_id] = {
                "case": request["case"],
                "tokens": [],
                "first_at": None,
                "admitted_at": started,
                "phase_start": phase_start,
                "input_tokens": len(tokens),
                "cached_tokens": cached,
                "expected": request.get("expected"),
            }
        self.drain(records)
        rows = []
        for row in records.values():
            assert row.get("finish_reason") in {"length", "stop"}, row
            assert row["tokens"]
            assert min(row["tokens"]) >= 0
            row["ttft_seconds"] = row.pop("first_at") - row["phase_start"]
            row["latency_seconds"] = row.pop("finished_at") - row["phase_start"]
            row["text"] = self.tokenizer.decode(row["tokens"], skip_special_tokens=True)
            row["token_sha256"] = hashlib.sha256(
                np.asarray(row["tokens"], dtype=np.int32).tobytes()
            ).hexdigest()
            rows.append(row)
        return {
            "requests": rows,
            "elapsed_seconds": time.perf_counter() - phase_start,
            "free_blocks": self.engine.scheduler.get_num_free_gpu_blocks(),
        }
