# P7 第二轮定向再生成新颖性审查

## 检索范围

- 本地语料更新至 219 篇，新增 7 篇全部具有官方元数据、PDF 和 MinerU Markdown。
- 重点排查：部分识别/robust policy learning、distributional pessimism、anytime-valid inference、reward ensemble、commit-first judge 和 hacking-onset detection。
- 候选必须改变数学对象，并对 H-001、H-005、H-014 以及已淘汰 H-018/H-021 设置强基线。

## 逐候选结论

| 候选 | 最近工作 | 判定 |
|---|---|---|
| H-027 Audit-Identified Gradient Set Direction | 2507.20550 的 marginal-sensitivity worst-case welfare；2606.09073 的标量 KL-DRO reward；H-001 的点通道估计 | 暂保留。只主张“由 audit channel intervals 诱导 score-weighted clean-gradient identified set，并选择对集合内全部梯度具有最大最坏内积的方向”；一般 partial identification 与 convex projection 不是新发明 |
| H-028 Distributional Entropic Clean Reward | 2606.09073、2505.20556、2606.19818 | 淘汰：同一标量悲观/不确定性 reward 已直接覆盖 |
| H-029 Anytime-Valid Audit Release | 2210.10768 | 淘汰：适应性策略、漂移、任意停止下 confidence sequence 已直接覆盖；release 只是 gate |
| H-030 Cross-Judge Consensus Gradient | 2312.09244、2607.05904、H-009 | 淘汰：ensemble 已有且共同 plausibility basin 不因多数投票消失 |
| H-031 Commit-First De-Anchored Reward | 2607.05904 | 淘汰：commit-first/blind solving 已直接作为训练 reward 介入 |
| H-032 Hacking-Onset Sentinel Reset | 2606.04923、H-020 | 淘汰：onset detector 已有，reset 是 schedule-only |

## H-027 的有限新颖性主张

H-027 不声称发明 partial identification、DRO、convex hull 或 min-norm projection。唯一待验证差异是：把 audit 对 verifier FP/FN 的不确定区间映射成一组可行 clean policy-gradient 向量，直接对“方向是否在所有可行真值下仍为上升方向”做 maximin 决策。现有近邻优化的是标量 welfare/reward 下界，没有直接给出该 score-weighted gradient-set update。
