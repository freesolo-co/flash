"""actor worker initialization and physical-device admission before NCCL startup."""

import os

from verl.workers.engine_workers import ActorRolloutRefWorker

from flash.engine.worker.train.core.child.startup import initialize_process


def claim_device(rank, path, uuid):
    import fcntl

    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"{rank} {uuid}\n")
        handle.flush()
        handle.seek(0)
        claims = {}
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                claims.setdefault(fields[1], set()).add(int(fields[0]))
        collided = {device: sorted(ranks) for device, ranks in claims.items() if len(ranks) > 1}
        if collided:
            raise RuntimeError(f"ranks were bound to the SAME physical gpu: {collided}")


class FlashActorRolloutWorker(ActorRolloutRefWorker):
    def __init__(self, config, *args, **kwargs):
        initialize_process()
        super().__init__(config, *args, **kwargs)
        options = config.model.runtime_options
        if int(options.get("dp_cards", 1)) < 2:
            return
        import torch

        if not torch.cuda.is_available():
            return
        ordinal = torch.cuda.current_device()
        uuid = str(getattr(torch.cuda.get_device_properties(ordinal), "uuid", "") or "")
        print(
            f"[rl-verl] rank-device binding rank={self.rank} cuda_ordinal={ordinal} uuid={uuid}",
            flush=True,
        )
        claims_path = os.environ.get("FLASH_RANK_DEVICE_CLAIMS")
        if uuid and claims_path:
            claim_device(self.rank, claims_path, uuid)
