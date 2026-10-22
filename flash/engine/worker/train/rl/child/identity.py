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


def _trajectory_info_from_identities(step, index, validate, identities):
    if len(identities) != len(index):
        raise RuntimeError("flash rollout identity sidecar length does not match worker batch")
    trajectories = []
    for position, (identity, sample_index) in enumerate(zip(identities, index, strict=True)):
        if identity["optimizer_step"] != _trajectory_int(step, "step"):
            raise RuntimeError("flash rollout identity sidecar step does not match worker batch")
        if identity["sample_index"] != _trajectory_int(sample_index, "sample_index"):
            raise RuntimeError(
                f"flash rollout identity sidecar index mismatch at position {position}"
            )
        if identity["validate"] is not _trajectory_bool(validate, "validate"):
            raise RuntimeError("flash rollout identity sidecar validate flag does not match batch")
        trajectories.append(
            {
                "step": identity["optimizer_step"],
                "sample_index": identity["sample_index"],
                "rollout_n": identity["rollout_ordinal"],
                "validate": identity["validate"],
            }
        )
    return trajectories
