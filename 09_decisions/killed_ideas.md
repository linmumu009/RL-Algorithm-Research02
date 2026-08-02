# Killed Ideas

## E0 淘汰 — 2026-08-02

- H-004：`REJECTED_FAILED_PREDICTION`。增强式代数上等于 channel-only correction，无法在“channel 错、regression 对”时稳健。
- H-008：`REJECTED_EQUIVALENT`。主预测成立，但行为不优于 direct pair average，不构成独立机制。
- H-018：`REJECTED_FAILED_PREDICTION`。有限样本下界在可交换条件有效，但无法覆盖所选问题中的分布漂移。

这些结果不得通过重命名、缩窄论点或只删除压力条件重新进入候选池。

## 定向再生成淘汰 — 2026-08-02

- H-022：OCRM/标准 covariate-shift importance weighting 已覆盖。
- H-023：arXiv:2602.18037 已直接提出 RLHF/RLVR gradient regularization。
- H-024：self-internalization 只提供诊断，sample gate 复活 H-010 filtering。
- H-025：随机 verifier view 是 H-021 proxy 采集步骤或 hybrid verification。
- H-026：arXiv:1705.08417 已提出 corrupted-reward quantilisation。

## H-021 正式 E0 淘汰 — 2026-08-02

- H-021：`REJECTED_FAILED_PREDICTION_AND_DOMINATED`。强代理 bias 与 oracle 对照通过，但 6 个 gain cell 全部失败；direct proxy regression 在 30 次强条件比较中赢 28 次。

不得删除 direct proxy regression、缩小 gain 阈值或把“低偏差”改名为成功后复活。

## 第二轮定向再生成淘汰 — 2026-08-02

- H-028：distributional entropic/KL-DRO reward 已由 arXiv:2606.09073 直接给出，UARM/PET 也覆盖悲观 reward。
- H-029：arXiv:2210.10768 已覆盖适应性策略与任意停止 confidence sequence；release gate 非独立。
- H-030：复活 H-009 ensemble，且跨 judge 共同 plausibility 错误已被实证证明。
- H-031：arXiv:2607.05904 已直接使用 commit-first blind solver 作为训练 reward。
- H-032：CHERRL onset detector 加 rollback 只是 H-020 schedule reset。

这些候选不得作为 H-027 的组件计入新颖性，也不得在 H-027 失败后换名补足分支数。

## H-027 正式 E0 淘汰 — 2026-08-02

- H-027：valid-coverage 安全性、point-limit 和 abstention 控制通过，但 45 个非对称非共线 cell 中 0 个达到 0.05 增益，最大仅 0.011294；H-001 midpoint 在 400 次比较中赢 332 次。
- 不允许删除 H-001、降低增益阈值或把仅有安全性改写为算法成功；任何后续 set-valued 候选必须改变信息结构或给出不同于本 convex-hull maximin direction 的可证伪对象。
