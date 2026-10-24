"""Identifiable backend failures shared with the hosted router's stream error mapper."""


class DeepInferError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
