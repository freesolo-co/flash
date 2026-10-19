"""Packaged OPD reward callback: distillation carries no task reward."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 0.0
