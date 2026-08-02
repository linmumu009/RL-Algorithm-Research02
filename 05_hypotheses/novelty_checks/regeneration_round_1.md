# P7 定向再生成新颖性审查

## 检索范围

- 本地语料更新至 212 篇。
- 定向新增 6 篇：corrupted reward channel、negative controls、proximal RL、OCRM、gradient regularization、rubric reward hacking。
- 复查目标包括公式对象、识别假设、退化行为、强基线和候选用途，而不只比较标题。

## 逐候选结论

| 候选 | 最近机制 | 判定 |
|---|---|---|
| H-021 Negative-Control Verifier Bridge | negative-control kernel bridge（2012.10315）；POMDP off-policy bridge（2110.15332）；corrupted reward cross-checking（1705.08417） | 暂保留。现有工作没有直接给出“随机 verifier proxy + sparse clean audit → RLVR clean score-function gradient”的估计式；但应用新颖性风险为中高，必须先过识别 E0 |
| H-022 Policy-Transported Audit Moment | OCRM（2507.15507）和通用 covariate-shift weighting | 淘汰：已有数学对象 |
| H-023 Gradient-Flat Verifier Trust Region | Gradient Regularization（2602.18037） | 淘汰：直接已有 |
| H-024 Self-Internalization Gradient Gate | rubric self-internalization diagnostic（2605.12474）+ H-010 filtering | 淘汰：诊断不构成纠错机制 |
| H-025 Hidden Verifier-View Randomization | hybrid verifier、corrupted-reward randomisation；同时是 H-021 的 proxy 采集步骤 | 淘汰：非独立分支 |
| H-026 Quantile-Randomized Reward Update | CRMDP quantilisation（1705.08417） | 淘汰：直接已有 |

## H-021 的有限新颖性主张

只主张以下待验证差异：目标是带 latent exploit 的 clean policy-gradient moment；随机 verifier view 与独立 diagnostic 分别承担 negative-control proxy 角色；稀疏 clean audit 锚定 bridge；输出是可用于 score-function 的 clean-reward bridge。不得把通用 proximal causal inference 本身称为新发明。
