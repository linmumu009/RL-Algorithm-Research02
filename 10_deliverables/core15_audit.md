# Core 15 Audit

## 审计结论

**PASS_WITH_REPLACEMENTS**

原核心清单已补齐全部 PDF，但其构成混合了在线推理 RL、直接偏好优化和自奖励范式，且对 GRPO 的结构偏差、长度边界、staleness 和统计性质覆盖不足。新增 4 篇诊断/理论论文后，可以通过替换形成更适合本项目的机制基底。

## 批准候选集合

| # | 核心论文 | 主要机制/证据角色 |
|---:|---|---|
| 1 | DeepSeekMath / GRPO | group-relative 无 critic 起点 |
| 2 | RLOO | REINFORCE leave-one-out 强基线 |
| 3 | DAPO | 解耦裁剪、动态采样、token loss、超长塑形 |
| 4 | Dr. GRPO | 标准差和长度归一化偏差诊断 |
| 5 | VAPO | critic、value pretraining 与长轨迹 GAE |
| 6 | GSPO | sequence-level ratio 与 clipping |
| 7 | CISPO | importance-weight clipping 与长上下文效率 |
| 8 | SAPO | 连续软门与 token-adaptive off-policy 控制 |
| 9 | PRIME | 隐式过程奖励与稠密信用分配 |
| 10 | SCoRe | on-policy 多轮自纠错与行为坍缩诊断 |
| 11 | RPT | 可验证奖励扩展到预训练阶段 |
| 12 | Stabilizing RL with LLMs | token surrogate 条件、staleness 与 Routing Replay |
| 13 | Hidden Objective Biases | shared-prefix、AdamW、momentum 与 surrogate 偏差 |
| 14 | GRPO U-Statistic | 组梯度统计性质与组大小缩放律 |
| 15 | Length-Impossibility | 无偏性与长度不变不可兼得的理论边界 |

## 覆盖摘要

- 设计维度：18/18。
- 失败模式：16。
- 互不等价核心机制：不少于 12。
- 加权覆盖分：89/100。
- 诊断、负结果和理论论文：5 篇。
- 语料、论文卡、PDF 和 MinerU Markdown 均可通过 `paper_id` 追溯。

## 替换

将 Self-Rewarding、KTO、ORPO、SimPO 从核心层下放扩展层，替换为 Stabilizing RL、Hidden Objective Biases、GRPO U-Statistic、Length-Impossibility。详细理由见 `02_literature/selection_audit/replacement_recommendations.md`。

## 风险

- 当前本地语料为 180 篇，尚低于长期目标 200–300；现有来源索引的全部唯一 arXiv ID 已补齐，新增论文应通过后续定向检索而不是随机凑数。
- 172 个算法条目复用了 4 个综述或汇总 URL，不能把算法条目数当作独立论文数。
- 核心集合仍偏向 GRPO 谱系，后续 P3/P4 必须从完整语料动态检索反证与替代机制。

## 下一阶段许可

在用户批准本核心集合之前：

- 不进入 P3 正式知识地图构建；
- 不提出新算法；
- 不实现训练代码；
- 只允许修正文献映射和补充审计证据。
