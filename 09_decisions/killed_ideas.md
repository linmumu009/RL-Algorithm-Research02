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

## 第三轮定向再生成淘汰 — 2026-08-02

- H-034：mode-wise Youden phase boundary 已由 arXiv:2601.04411 直接推导；barrier/gate 是 H-001 channel correction 的阈值版本。
- H-035：counterfactual-invariant causal reward head 已由 arXiv:2501.09620 和 2601.21350 直接覆盖。
- H-036：trusted update subspace 已由 arXiv:2605.25189 直接作为 reward-hacking 缓解提出，也邻近 H-005 projection。
- H-037：Hacker/Auditor 两阶段训练与 reward gate 已由 arXiv:2602.01750 的 ARA 直接覆盖。
- H-038：实时 policy hidden-state reward adapter 已由 arXiv:2601.22664 直接覆盖，iterated RM 也已有系统研究。

以上候选不得作为 H-033 的组件计入新颖性。H-033 若不能超过 exact Reusable Holdout、Generic Holdout 或 once-trained private reward model，也必须按等价/无增量价值淘汰。

## H-033 正式 E0 淘汰 — 2026-08-02

- H-033：`REJECTED_FAILED_PREDICTION_AND_DOMINATED`。144 个 adaptive cell 中 118 个 bias 超标、127 个 cosine 不足、82 个方向错误率超标，且 qualifying gain cell 为 0。
- 不允许删除低 rho、高维或 drift cell，不允许重分配 privacy budget 后换名复活；DP accountant 正确不等于 reward robustness 成功。

## 第四轮定向再生成淘汰 — 2026-08-02

- H-040：correlated-proxy worst-case optimization 已由 arXiv:2604.12086 直接提出。
- H-041：reward uncertainty distribution 与 action-set objective 已由 arXiv:2606.03962 直接提出。
- H-042：non-hackable confidence reward family 已由 arXiv:2607.04332 直接刻画和评估。
- H-043：current-RF optimization 与 causal tamper-resistance 原则已由 arXiv:1908.04734 直接提出。
- H-044：restricted policy class 上 unhackability 的必要充分条件已由 arXiv:2209.13085 给出；条件检查不是新的学习机制。

以上候选不得作为 H-039 的组件重新计算新颖性。H-039 若与 SignCert-PO/H-027 行为等价，或收益只来自更低接受率，也必须淘汰。
