"""Local training algorithms controlling a managed Flash run.

The client and data contract import without torch, Modal, or server dependencies.
"""

from flash.custom.async_client import AsyncTrainingClient
from flash.custom.client import TrainingClient
from flash.custom.types import Datum

__all__ = ["AsyncTrainingClient", "Datum", "TrainingClient"]
