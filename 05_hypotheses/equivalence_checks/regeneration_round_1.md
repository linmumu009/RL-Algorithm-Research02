# P7 定向再生成等价性审查

## 排除结果

- H-022 与 OCRM/importance weighting 共享同一分布运输矩，不是 H-001 之后的新识别机制。
- H-023 与 arXiv:2602.18037 的有限差分 gradient regularization 同式。
- H-024 只把 diagnostic 转成 sample gate，行为退化为已淘汰 H-010。
- H-025 若只随机选择 verifier，则是 matched-cost hybrid verification；若把随机 view 用作 proxy，则属于 H-021 的数据生成步骤。
- H-026 是 arXiv:1705.08417 的 quantilising agent 在 RLVR 上的重命名。

## H-021 与现有活跃分支的差异

| 对比 | 不等价理由 | 退化条件 |
|---|---|---|
| H-001 | H-001 假设混淆通道可由 observed strata 条件化；H-021 通过两个 proxy 的条件矩识别 latent exploit | latent exploit 被直接观测时，bridge 应退化到 observed-latent correction |
| H-005 | H-005 对声明的 nuisance 做线性/特征投影；H-021 解条件逆问题，允许 proxy 与 latent exploit 非线性关联 | bridge 为线性且 W 就是 declared nuisance 时可能接近 residualization，必须用非线性/离散反例区分 |
| H-014 | H-014 改变哪些样本被 audit；H-021 改变 audit 后识别哪个 clean reward moment | 审计为均匀全量时 H-014 退化，H-021 的 bridge estimand 不变 |
| H-008 | H-008 在声明对称结构下等于 pair average；H-021 需要同时满足 conditional moment 和 proxy rank，不等于固定配对差 | bridge 系数恰为对称 1/2 时会退化为 pair average，E0 必须包含非对称 cell |

H-021 只有在非对称 proxy cell 中仍优于 direct pair average，才可继续维持独立机制判定。
