# P7 替代分支筛选报告

## 结论

围绕 E0 后的三分支缺口生成 6 个替代候选，保留 H-021 `Negative-Control Verifier Bridge`，淘汰 5 个直接已有、诊断型或非独立候选。活跃组合恢复到 4 条：H-001、H-005、H-014、H-021。

## 候选表

| ID | 机制家族 | 总分 | 可证伪 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| H-021 | proxy bridge identification | 87 | 15 | 12 | 保留并预注册 E0 |
| H-022 | distribution transport | 82 | 14 | 6 | OCRM/importance weighting 已覆盖 |
| H-023 | gradient regularization | 84 | 14 | 4 | 2602.18037 直接覆盖 |
| H-024 | diagnostic gating | 76 | 13 | 6 | detector + H-010 filtering，无 clean-gradient 机制 |
| H-025 | verifier randomization | 83 | 15 | 8 | hybrid verifier 或 H-021 proxy 采集步骤 |
| H-026 | reward quantilisation | 74 | 12 | 4 | 1705.08417 直接覆盖 |

## 为什么保留 H-021

H-021 针对的是 observed strata 与 declared nuisance 都无法表示的 latent exploit 类型。它用 randomized verifier-only proxy、独立 exploit diagnostic 和 sparse clean audit 建立 conditional bridge moment。该数学对象不同于通道反演、nuisance projection、audit acquisition 和 pair averaging；同时具有非常强的 exclusion/completeness 风险，因此只获得 1 单位 E0 权限，不获得语言模型训练权限。

## 阶段判定

- 定向再生成：`PASS_RETAIN_ONE`。
- 活跃分支数：4，恢复施工方案下限。
- 下一项唯一允许任务：实现并冻结 H-021 的离散 bridge E0，绑定代码提交后一次性运行。
