# P5/P6 等价性审查

## 数学等价

- `H-011`：在二值固定 confusion channel 下，`P(r_clean=1|r_obs,z)` 代入 score-function gradient 可重写为 forward correction 的仿射权重；没有独立机制。
- `H-012`：若 contextual correction 对 reward 为仿射变换，则“先 correction 后 group center”与 H-001 在 GRPO advantage 中相同。
- `H-019`：hierarchical shrinkage 只改变 H-001 的 nuisance estimator，不改变 clean-gradient estimand。

## 语义等价

- `H-003` 等同 NoisyVerifierRL 的 negative appeal。
- `H-009` 等同 reward ensemble/hybrid verifier disagreement weighting。
- `H-015` 等同 adversarial verifier training/verification engineering。
- `H-006` 是 constrained/pessimistic reward optimization 的 confusion-matrix 实例。

## 行为等价或非独立分支

- `H-002`、`H-016`、`H-020` 只改变 H-001 的 feature 或更新时序；不能写出在 matched estimator 下不同的收敛目标。
- `H-010` 与 confidence filtering 行为相同：减少有效样本，但不校正错误方向。
- `H-017` 是 H-014 在自适应采样下保持总体 estimand 所需的 propensity control，应作为 H-014 消融而不是分支。

## 保留候选的退化边界

| 候选 | 退化条件 | 退化到 |
|---|---|---|
| H-001 | 所有 strata 共享同一 confusion matrix | NoisyVerifierRL global correction |
| H-004 | clean-reward regression 恒定为零 | channel-only correction |
| H-004 | residual correction 为零 | outcome regression policy gradient |
| H-005 | nuisance projection 系数为零 | 原始 verifier reward |
| H-008 | verifier 对所有合法变换完全 invariant | 原始 reward，control variate 为零 |
| H-014 | score vectors 等范数且 reward uncertainty 相同 | uniform audit sampling |
| H-018 | audit 数量趋于无穷且区间收缩 | point-estimate contextual correction |

这些退化条件均被写入 E0/E1 的否定测试，防止公式不同但行为相同。
