# Core 15 Replacement Recommendations

## 结论

对原 Excel 中的 15 篇执行四项替换，结论为 `PASS_WITH_REPLACEMENTS`。替换后的集合记录于 `02_literature/core15/core15_manifest.csv`。

| 移出核心层 | 移入核心层 | 原因 |
|---|---|---|
| Self-Rewarding Language Models | Stabilizing Reinforcement Learning with LLMs | 前者主要是自奖励数据循环；后者直接覆盖 sequence/token surrogate、policy staleness 与 MoE 稳定条件。 |
| KTO | On the Hidden Objective Biases of Group-based RL | KTO 属于点式偏好对齐；新论文补充 shared-prefix bias、AdamW 与 clipping 失效分析。 |
| ORPO | GRPO Policy Gradient is a U-Statistic | ORPO 属于离线偏好目标；新论文提供 GRPO 组大小、方差和有限样本理论。 |
| SimPO | Impossibility of Unbiased and Length-Invariant Policy Optimization | SimPO 属于无参考偏好优化；新论文给出 outcome reward 下长度公平与梯度无偏不可兼得的边界。 |

## 保留在扩展层

被移出的四篇仍保留在 `01_corpus/raw/extended/` 和全量 `inventory.csv` 中，后续在偏好优化、奖励建模或对齐边界问题出现时动态检索，不视为低质量或排除论文。

## 替换后的结构

- 基础与强基线：GRPO、RLOO。
- 代表性策略优化：DAPO、Dr. GRPO、VAPO、GSPO、CISPO、SAPO。
- 奖励、信用和多轮机制：PRIME、SCoRe、RPT。
- 诊断、理论和负结果：Stabilizing RL、Hidden Objective Biases、GRPO U-Statistic、Length-Impossibility。
