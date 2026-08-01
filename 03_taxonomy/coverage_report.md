# Core 15 Coverage Report

## 定量结果

| 维度 | 得分 | 证据 |
|---|---:|---|
| 设计维度覆盖 | 94/100 | 18/18 个矩阵维度至少有一篇达到核心分析级别，18/18 有直接机制或证据。 |
| 失败模式覆盖 | 90/100 | 已登记 16 类失败模式，覆盖长度、稀疏信用、裁剪、staleness、MoE、reward hacking、行为坍缩等。 |
| 证据类型覆盖 | 86/100 | 包含大规模系统实验、受控消融、理论推导、统计分析、诊断研究和负结果。 |
| 方法谱系与时间跨度 | 82/100 | 覆盖 2024–2026，并从 REINFORCE/PPO/GRPO 延伸到序列、软门、过程奖励和理论边界。 |
| 负结果和限制 | 88/100 | 至少 4 篇以诊断、结构偏差或不可能性为核心，所有论文卡均登记限制。 |
| **加权总分** | **89/100** | `30% + 25% + 20% + 15% + 10%` 加权后四舍五入。 |

## 验收条件

- 加权覆盖分 ≥ 80：通过（89）。
- 设计维度 ≥ 14：通过（18）。
- 失败模式 ≥ 10：通过（16）。
- 明确失败分析或限制论文 ≥ 3：通过（Dr. GRPO、Stabilizing RL、Hidden Biases、Length-Impossibility）。
- 互不等价核心机制 ≥ 10：通过（group baseline、LOO、动态采样、长度去偏、critic/GAE、sequence ratio、importance-weight clipping、soft gate、implicit process reward、multi-turn progress reward、RPT、staleness correction 等）。
- 任一关键维度完全空缺：无。

## 仍然偏薄的区域

- verifier 噪声和 reward hacking 的独立干预实验仍少，PRIME 主要提供过程奖励视角。
- 昂贵长轨迹的安全 off-policy 复用仍缺少专门核心论文。
- 单请求延迟与吞吐的严格区分不足，多数论文报告训练效率而非端到端延迟。
- 跨模型规模、跨任务族的独立复现不足。
- 核心集合仍以 GRPO 谱系为主；其合理性来自当前问题域，但后续问题选择时必须动态检索扩展语料防止谱系偏见。

## 结论

`PASS_WITH_REPLACEMENTS`。替换方案解决了原集合偏好优化论文占位、诊断和负结果不足的问题。进入 P3 前仍需用户批准替换后的核心 15。
