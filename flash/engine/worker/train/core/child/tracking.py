"""tracking belongs to the trainer that creates and closes the run."""

import contextlib
import json

from verl.utils.tracking import Tracking


class OwnedTracking(Tracking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run = getattr(self.logger.get("wandb"), "run", None)
        if self.run is not None:
            with contextlib.suppress(Exception):
                print(
                    "FLASH_WANDB_LINK "
                    + json.dumps({"wandb_url": self.run.url, "wandb_id": self.run.id}),
                    flush=True,
                )

    def finish(self, exit_code):
        # the upstream destructor must not finish a later process-global wandb run.
        self.logger.pop("wandb", None)
        if self.run is not None:
            with contextlib.suppress(Exception):
                self.run.finish(exit_code=exit_code)
            self.run = None
        super().finish(exit_code=exit_code)


def publish_tracking_link(tracking):
    run = getattr(tracking.logger.get("wandb"), "run", None)
    if run is not None:
        print(
            "FLASH_WANDB_LINK " + json.dumps({"wandb_url": run.url, "wandb_id": run.id}), flush=True
        )
