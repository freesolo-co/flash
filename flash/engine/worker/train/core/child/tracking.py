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


class TrackingOwner:
    def tracking_class(self, **kwargs):
        self._owned_tracking = OwnedTracking(**kwargs)
        return self._owned_tracking

    def fit(self):
        exit_code = 1
        try:
            result = super().fit()
            exit_code = 0
            return result
        finally:
            tracking = getattr(self, "_owned_tracking", None)
            if tracking is not None:
                tracking.finish(exit_code)
