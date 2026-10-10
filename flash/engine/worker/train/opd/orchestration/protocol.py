"""Shared parent-child constants for OPD execution."""

PERMANENT_TEACHER_EXIT = 86
TRANSIENT_TEACHER_EXIT = 87
TEXT_TEACHER_FLUSH_WAIT_S = 0.1
TEXT_TEACHER_SHUTDOWN_WAIT_S = 5.0

# opd supervises the teacher's distribution, not a task reward, so every rollout scores zero. the
# score is unreachable either way: use_task_rewards=false makes verl zero the whole policy loss
# (distillation/losses.py:211), so nothing a scorer returns can enter the gradient. this exists
# only to keep the reward loop out of its builtin data_source registry -- see the call site.
OPD_ZERO_REWARD_SOURCE = '''"""flash opd reward shim (generated). opd carries no task reward."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 0.0
'''


def spec_gpu_type(spec) -> str:
    """Return the selected GPU class, or an empty string when absent."""
    return str(getattr(getattr(spec, "gpu", None), "type", "") or "")
