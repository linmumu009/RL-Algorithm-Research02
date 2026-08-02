# Q-001 第三轮替代假设筛选

## 结果先行

H-027 失败后生成 H-033–H-038 六个候选，并新增 12 篇直接近邻，语料达到 231 篇。五个候选被直接论文或已淘汰机制覆盖；仅暂留 H-033 `Privacy-Stable Reusable Audit Gradient`，使活跃组合恢复为 H-001、H-005、H-014、H-033 四条。

| 候选 | 机制家族 | 总分 | 可证伪 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| H-033 | adaptive audit release | 87 | 15 | 11 | 条件性保留并预注册 1 单位 E0 |
| H-034 | mode dynamics gate | 84 | 14 | 5 | Rate-or-Fate 已给出 J phase boundary；等价 H-001 + gate |
| H-035 | causal reward representation | 89 | 14 | 2 | 两篇 causal reward 工作直接覆盖 |
| H-036 | clean direction projection | 86 | 14 | 2 | Directional Alignment 直接覆盖，且邻近 H-005 |
| H-037 | adversarial auditing | 88 | 13 | 1 | ARA 直接覆盖 Hacker/Auditor gate |
| H-038 | dynamic reward modeling | 85 | 13 | 2 | R2M 与 iterated RLHF 直接覆盖 |

## 为什么仅暂留 H-033

H-033 改变的是 clean audit 的发布接口，而不是再构造一个 channel point estimate。固定 audit 集在多个 policy round 中被反复查询时，后续查询依赖先前 release；即使每次样本均值看似无偏，策略也可能选择性放大 audit 特有噪声。H-033 对每个 audit-gradient contribution 做固定 L2 clipping，发布 Gaussian-noised correction，组合 zCDP privacy loss并在预算耗尽时停止。

这一机制明显借用 adaptive data analysis，因此新颖性只在 RLVR 特定的连续梯度 release、方向错误和可用轮数上成立。E0 必须与 exact Reusable Holdout、Generic Holdout 和 once-trained private reward model 正面对比；若没有独立收益，直接淘汰。

## 边界

- H-033 只获得 1 单位解析/合成 E0 权限，不获得 E1 或语言模型训练权限。
- 不允许把 privacy epsilon/rho 本身当作 reward-hacking 成功证据。
- 不允许删掉低 rho、高维、population drift、clip violation 或非自适应负控。
- 下一项唯一允许任务：实现并冻结 H-033 的 adaptive audit-query E0，提交不可变代码、绑定预注册后一次性执行。
