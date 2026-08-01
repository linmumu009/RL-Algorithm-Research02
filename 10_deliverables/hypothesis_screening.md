# Q-001 假设生成与筛选结果

## 结论

**G5：PASS**

- 初始候选：20。
- 直接淘汰：14。
- 保留：6。
- 保留机制家族：6，满足至少 3 类的要求。
- 所有保留候选总分 ≥ 85、可证伪性 ≥ 14/15、与已有方法差异 ≥ 11/15。
- 所有保留候选均有明确失败阈值和预算，单候选预计预算 3–4 单位，低于 20 单位上限。

## 保留分支

| 排名 | ID | 候选 | 改变对象 | 分数 | 最小否定实验 |
|---:|---|---|---|---:|---|
| 1 | H-001 | Contextual Confusion Correction | 条件 reward channel | 95 | 等边际、异质 strata 的 clean-gradient cosine |
| 2 | H-008 | Counterfactual Pair Control Variate | 配对 counterfactual reward difference | 89 | 注入 verifier 变换敏感性的固定轨迹实验 |
| 3 | H-014 | Clean-Gradient Leverage Auditing | audit acquisition distribution | 88 | 固定 score vectors、有限 reveal budget |
| 4 | H-004 | Doubly Robust Audit Gradient | clean policy-gradient estimator | 87 | 两 nuisance 模型正确/错误四单元测试 |
| 5 | H-018 | Conformal Lower Clean-Reward Bound | coverage-calibrated reward constraint | 86 | 分层 binary coverage 与 shift stress |
| 6 | H-005 | Nuisance-Orthogonal Reward | nuisance gradient subspace | 85 | causal/spurious feature 正交对照 |

## 淘汰分布

- 数学/行为等价：H-002、H-007、H-011、H-012、H-017、H-019。
- 已有方法覆盖：H-003、H-006、H-009、H-015。
- 没有独立机制或理论不一致：H-010、H-013、H-016、H-020。

## 下一阶段

只允许这 6 个分支进入 P7 的 E0/E1。先执行数学一致性和合成数据否定实验；在 E0/E1 通过前不进行模型训练。
