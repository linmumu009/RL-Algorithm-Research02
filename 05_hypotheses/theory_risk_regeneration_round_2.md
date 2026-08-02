# P7 第二轮定向再生成理论风险审查

## H-027 主要风险

1. **identified set 覆盖错误**：audit 区间若因选择偏差或漂移排除真实通道，任何 maximin 保证都失效。
2. **过度 abstention**：高维、宽区间时凸包容易包含零，方法可能安全但不学习。
3. **标量 DRO 等价**：在低维或共线情形，min-norm gradient-set direction 可能等于某个 scalar pessimistic reward 的梯度。
4. **有限审计不稳定**：interval vertices 经通道反演后会放大，几何方向可能被单一极端 vertex 主导。
5. **计算扩展**：真实模型梯度维度极高，E0 的二维凸几何不能证明可扩展性。
6. **H-014 依赖**：区间质量依赖审计数据；H-027 只可主张 update geometry，不可把 audit acquisition 计为自身新颖性。

## E0 必须先验证

- 精确枚举二维可行通道 vertex，报告 convex hull、原点包含、min-norm point 和最坏 alignment margin。
- 同时比较 H-001 midpoint、H-018 scalar lower bound、KL-DRO scalar pessimism、norm-matched shrinkage 与 oracle clean gradient。
- 包含 point-identification、zero-in-set、wide interval、misspecified interval 和二维非共线控制。
- 只有在真实通道被覆盖时控制 false-positive direction，interval 错误时必须显式暴露失败。
- 若相对最佳 scalar baseline 没有至少三个非共线 cell 的方向增益，或主要行为等同 abstention/shrinkage，淘汰。

风险级别：高。保留理由是数学对象明确、失败阈值严格且 E0 仅需 1 单位；不代表已有训练价值。
