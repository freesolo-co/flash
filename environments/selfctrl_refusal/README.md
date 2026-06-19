# selfctrl-refusal

Self-CTRL constitutional-AI refusal task (arXiv:2606.18327) as a Flash `verifiers`
environment.

The model predicts whether it will refuse a request (`<predict>`) and then actually
responds (`<respond>`). GRPO reward = consistency (prediction matches behavior) +
constitutional term (refuse harmful, comply benign). Refusal-prediction accuracy and
HarmBench failure rate are weight-0 eval metrics. The same env serves SFT and GRPO.

Publish: `slm env push environments/selfctrl_refusal/selfctrl_refusal.py`
