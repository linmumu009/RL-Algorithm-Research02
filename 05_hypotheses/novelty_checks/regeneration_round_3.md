# P7 第三轮定向再生成新颖性审查

## 审查范围

H-033–H-038 与当前 231 篇语料、H-001–H-032 谱系及两轮正式失败结果逐项比较。判断标准仍为数学对象、信息结构和可观察行为，而不是名称或组件组合。

## 结果

| 候选 | 最近工作/分支 | 结论 |
|---|---|---|
| H-033 | Reusable Holdout、Generic Holdout、private RLHF、H-001/H-014/H-029 | 条件性保留：DP/holdout 理论已有，但把隐私预算用于反复 audit-gradient release 的自适应有效性而非个人隐私，当前语料无直接 RLVR 实现 |
| H-034 | 2601.04411、H-001 | 淘汰：mode-wise Youden phase boundary 已直接给出，gate/inverse-J 是通道修正附阈值 |
| H-035 | 2501.09620、2601.21350 | 淘汰：counterfactual invariance 与 factored causal reward head 已直接实现 |
| H-036 | 2605.25189、H-005 | 淘汰：trusted clean-direction projection 已直接提出 |
| H-037 | 2602.01750、H-010/H-024 | 淘汰：Hacker/Auditor 与 reward gate 已直接实现 |
| H-038 | 2601.22664、2505.18126 | 淘汰：policy-feedback reward adapter 和 iterated RM 已直接覆盖 |

## H-033 新颖性边界

H-033 不声称发明 differential privacy、Reusable Holdout 或 Gaussian mechanism。其候选贡献仅限于：把同一 clean-audit 集上的 score-weighted verifier-error correction 看作由策略自适应选择的连续向量查询；用 clipping、Gaussian release、composition 和 privacy filter 构成 verifier-correction interface，并检验它是否比 exact Reusable Holdout、Generic Holdout 和一次性 private reward model 更适合保留可用更新。

若 E0 表明 H-033 与 Reusable Holdout 的 noisy statistic release 数学同式且没有 RLVR 特有的方向/可用轮数收益，按 `REJECTED_NOT_NOVEL_OR_EQUIVALENT` 淘汰。
