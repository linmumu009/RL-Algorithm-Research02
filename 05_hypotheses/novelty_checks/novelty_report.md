# P5/P6 新颖性审查

## 审查范围

- 本地语料：206 篇唯一 arXiv 论文。
- 针对候选额外补入 6 篇近邻工作：iterative/OOD reward modeling、efficient reward ensemble、optimal reward-data design、Active DPO、gradient-impact active preference learning、Doubly Robust Alignment。
- 检索对象：目标问题、改变的数学量、公式结构、同义机制、行为预测和最低成本实验。

## 逐候选结论

| ID | 最近工作或已有机制 | 真正差异 | 结论 |
|---|---|---|---|
| H-001 | NoisyVerifierRL 的全局 FP/FN correction | 条件通道 `T(z)`；预测由跨 stratum 异质性驱动 | 保留 |
| H-002 | H-001 | 只增加 policy/time 特征 | 淘汰：等价特例 |
| H-003 | NoisyVerifierRL appeal | 无 | 淘汰：已有 |
| H-004 | Doubly Robust Alignment；NoisyVerifierRL | clean audit 的 outcome regression + verifier-channel residual，目标是 clean policy gradient | 保留，需证明双重稳健恒等式 |
| H-005 | RCfD、uncertainty penalty、GamingVerifiers | 针对可操纵 nuisance 方向做正交化，并预测在 nuisance 真有因果作用时反向伤害 | 保留 |
| H-006 | constrained RLHF、uncertainty penalty、DRO | 只把 uncertainty set 换成 confusion matrix | 淘汰：通用 robustification |
| H-007 | H-001 | latent mixture 参数化 | 淘汰：过度参数化重命名 |
| H-008 | GamingVerifiers 的 IPT、TRACE | 使用配对 reward difference 作为 control variate，不把扰动 verifier 当 ground truth | 保留，边界差异明确 |
| H-009 | efficient reward ensemble、UP-RLHF、VerIF | 无 | 淘汰：已有 |
| H-010 | uncertainty filtering、filtered optimization | 仅 mask 高风险样本 | 淘汰：无纠错机制 |
| H-011 | NoisyVerifierRL forward correction | 固定 confusion 下只是 posterior 写法 | 淘汰：数学等价 |
| H-012 | H-001 + GRPO centering | 运算次序在仿射 correction 下不构成独立机制 | 淘汰：等价 |
| H-013 | group baseline/control variate | 只改 baseline，不能去除 action-correlated reward bias | 淘汰：机制错误 |
| H-014 | Optimal Design、Active DPO、SHARP/W-SHARP | 选择的是 rollout correctness audit，目标是 clean-gradient covariance，不是 preference-pair learning | 保留，近邻明确 |
| H-015 | verifier engineering、adversarial training | 无 | 淘汰：已有且偏离 policy-gradient 问题 |
| H-016 | two-timescale stochastic approximation | 只改更新次数 | 淘汰：调度变化 |
| H-017 | active sampling propensity correction | 是 H-014 的必要审计修正，不是独立研究分支 | 淘汰：并入 H-014 控制项 |
| H-018 | UP-RLHF、efficient ensemble LCB、constrained RLHF | 使用 held-out audits 校准覆盖率，而非 ensemble 标准差启发式 | 保留，需严格限定 exchangeability |
| H-019 | H-001 | hierarchical shrinkage 参数化 | 淘汰：等价 |
| H-020 | H-001 | change-point reset 只改变状态保留时序 | 淘汰：无新 estimand |

## 结论

20 个初始候选中保留 6 个。保留项分别改变 reward channel、gradient estimator、nuisance constraint、counterfactual estimator、audit data acquisition 和 uncertainty constraint，来自 6 个机制家族。未以“组件组合”作为保留理由。
