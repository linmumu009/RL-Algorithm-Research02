# P7 第三轮定向再生成等价性审查

## H-033 强等价基线

| 对比 | 潜在等价 | E0 必须保留的区分 |
|---|---|---|
| Reusable Holdout / Thresholdout | 都以 DP 稳定性减少 adaptive leakage | 同 clipping、同噪声标度、同 privacy accountant；H-033 必须在向量方向或可用轮数上给出独立收益 |
| Generic Holdout | 都限制 holdout 信息释放 | 比较仅发布 pass/fail 的安全 gate；若向量 release 只增加泄露而无 clean-gradient收益则淘汰 |
| private reward model | 都把 DP 用于 reward-derived signal | 比较一次训练后无限 postprocessing 与逐轮花费预算；若一次性 private model 更优则淘汰 H-033 |
| H-001 | 都输出 verifier correction | privacy-off 或单次查询时必须退化为同一 clipped audit correction，不得把噪声本身计作收益 |
| H-014 | 都使用有限 clean audit | 固定完全相同的 audit 样本；H-033 不得把新增审计或更优 acquisition 计作自身贡献 |
| H-029 | 都可能停止 release | H-029 处理时间有效性，H-033 处理 adaptive information leakage；若最终只有 stop gate 不输出有效 correction，则不独立 |

## 其他候选

- H-034 在 J=TPR-FPR 上的 barrier 与 channel invertibility 条件相同，附加 mode gate 不改变 H-001 估计对象。
- H-035 与已发表 causal reward regularization 同对象。
- H-036 与 trusted-direction projection 逐式一致，并与 H-005 同属梯度投影。
- H-037 与 ARA 的两阶段 adversarial auditing 和 reward gate 同流程。
- H-038 与 R2M 的 policy-feedback adapter 同输入与目标。
