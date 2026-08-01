# P5/P6 理论风险审查

| 候选 | 核心理论风险 | 最危险边界 | 必须先证明/测试 | 风险级别 |
|---|---|---|---|---|
| H-001 | 条件矩阵近奇异导致 correction 方差爆炸 | `FP + FN -> 1` 或稀有 stratum | 可辨识性、正则化偏差、condition number | 中高 |
| H-004 | 所谓 doubly robust 性质可能因 score-function 与 adaptive audit selection 失效 | 两个 nuisance model 同时错；audit 非随机 | 四单元正确/错误模型恒等式与 propensity control | 高 |
| H-005 | nuisance residualization 可能删除真实任务信号 | 长度/格式本身影响正确性 | 合成 causal/spurious 对照和 signal-protection constraint | 高 |
| H-008 | “语义保持”变换可能并不保持模型条件分布或 verifier ground truth | 变换改变可解性、tokenization、长度 | transformation validity 与 paired covariance 符号 | 中高 |
| H-014 | leverage sampling 会集中于极端梯度并放大 selection variance | 极小 acquisition propensity | logged propensity、ESS 下界、matched-audit MSE | 中高 |
| H-018 | conformal coverage 在 non-exchangeable policy drift 下失效 | policy 适应导致 audit/calibration 分布改变 | 分层 coverage、rolling recalibration、shift stress | 高 |

## 共通风险

- 任一候选如果只在增加 audit 数量后改善，不能归因于机制。
- 任一候选如果只降低 gradient norm 而不改善 clean-gradient cosine，视为保守缩步而非纠错。
- 独立正确率必须由不参与训练的 oracle/holdout verifier 给出。
- E0/E1 必须同时报告 bias、variance、MSE、cosine、effective sample size 和 false-positive update rate。
- 同一候选最多两轮局部修复；本轮所有 `repair_count=0`。
