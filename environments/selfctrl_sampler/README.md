# selfctrl-sampler

Self-CTRL biased-sampler task (arXiv:2606.18327) as a Flash `verifiers` environment.

The model imitates a biased sampler and must be self-consistent: its stated
distribution (`<predict>`) should match the empirical frequencies of its own draws
(`<samples>`). GRPO reward = `1 - TV(predicted, empirical)`; the paper's R² is a
weight-0 eval metric. The same env serves SFT (gold `answer`) and GRPO (rubric).

Publish: `slm env push environments/selfctrl_sampler/selfctrl_sampler.py`
