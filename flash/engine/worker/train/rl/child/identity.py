"""exact occurrence identities travel with each worker batch."""


def _trajectory_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"flash rollout trajectory {name} must be an integer")
    return value


def _trajectory_bool(value, name):
    if not isinstance(value, bool):
        raise RuntimeError(f"flash rollout trajectory {name} must be a boolean")
    return value


def _identity_from_trajectory(trajectory):
    return {
        "optimizer_step": _trajectory_int(trajectory.get("step"), "step"),
        "sample_index": _trajectory_int(trajectory.get("sample_index"), "sample_index"),
        "rollout_ordinal": _trajectory_int(trajectory.get("rollout_n"), "rollout_n"),
        "validate": _trajectory_bool(trajectory.get("validate"), "validate"),
    }
