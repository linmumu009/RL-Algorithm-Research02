# Q-001 第二轮替代假设筛选

## 结果先行

围绕 H-021 失败后的三分支缺口生成 H-027–H-032 六个候选。新增 7 篇直接近邻、语料达到 219 篇后，仅保留 H-027 `Audit-Identified Gradient Set Direction`，其余五个因直接已有、ensemble 复活或诊断/调度无独立机制而淘汰。活跃组合恢复为 H-001、H-005、H-014、H-027 四条。

| 候选 | 机制家族 | 总分 | 可证伪 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| H-027 | set-valued gradient robustness | 90 | 15 | 13 | 保留并预注册 1 单位 E0 |
| H-028 | distributional reward pessimism | 86 | 14 | 4 | 2606.09073/UARM/PET 直接覆盖 |
| H-029 | confidence-sequence gating | 83 | 15 | 5 | 2210.10768 直接覆盖；gate 非独立 |
| H-030 | verifier ensemble | 78 | 13 | 4 | H-009/ensemble 已有且共享错误不消失 |
| H-031 | verifier architecture | 88 | 15 | 2 | 2607.05904 直接覆盖 commit-first reward |
| H-032 | diagnostic schedule | 75 | 13 | 3 | CHERRL detector + H-020 reset，无 clean-gradient estimand |

## 为什么保留 H-027

H-027 不再尝试点识别一个“正确 reward”。它保留所有 audit-compatible verifier channel，将每个 channel 映射为 score-weighted clean-gradient 向量，得到 identified set；只有当该集合与零可分时，才沿对集合内所有梯度最有利的 maximin 方向更新。其直接失败方式包括集合含零导致过度 abstention、退化为 scalar DRO/shrinkage，以及 audit interval 漏盖真实 channel。

## 边界

- H-027 只获得 1 单位二维解析/合成 E0 权限，不获得 E1 或语言模型训练权限。
- 必须同时比较 point correction、scalar lower bound、KL-DRO scalar pessimism、norm-matched shrinkage 和 oracle。
- 若 H-027 主要通过 abstention 获得安全性，或与最佳 scalar baseline 行为等价，立即淘汰。
- 下一项唯一允许任务：实现并冻结 H-027 的 gradient-set E0，提交不可变代码、绑定预注册后一次性执行。
