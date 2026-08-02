# H-027 E0 审计识别梯度集合实验报告

## 结果先行

H-027 按冻结提交 `b2778d683b22d8f7a24f60e3d3443abb2671aed2` 唯一一次执行，预注册总结果为 `FAIL`。方法在真实通道被审计区间覆盖时没有选择有害方向，也正确完成 point-limit 和 zero-in-set abstention；但相对最佳非 oracle 强基线没有足够方向增益。因此淘汰 H-027，不做结果后局部修补。

原始 JSON 规范化为 LF 换行后的 SHA-256 为 `8bbf6df31fbb17a40e3bdd723dd9edc70343da074ffe78b6f194ede1ef0190ac`，包含 400 个 valid-coverage 运行行、30 个控制行和 80 个完整汇总 cell。

## 预注册判定

| 判定项 | 结果 | 是否通过 |
|---|---:|---:|
| valid-coverage cell 的 false-positive direction rate ≤ 0.05 | 最大 0.000000 | 是 |
| 64 个 strong-identified cell 的 clean-gradient cosine ≥ 0.90 | 最小 0.995153 | 是 |
| 至少 3 个非对称非共线 cell 的 cosine gain ≥ 0.05 | 0/45 | 否 |
| zero-in-set abstention ≥ 0.90 | 1.000000 | 是 |
| point-limit direction difference ≤ 0.01 | 0.000000 | 是 |

唯一失败项是候选的独立增量价值主张。45 个非对称非共线 cell 的平均 gain 为 0.000659，最大值仅 0.011294，远低于 0.05；全体 80 个 cell 的 gain 范围为 -0.027710 至 0.011294。

## 强基线与等价性

H-001 midpoint channel correction 在 400 个 valid-coverage 运行中成为最佳非 oracle 基线 332 次；H-018 scalar lower bound 为 61 次，KL-DRO scalar pessimism 为 4 次，norm-matched shrinkage 为 3 次。H-027 的最坏情况分离方向通常保持很高 oracle 对齐，但同一审计区间的 midpoint correction 已经得到近乎相同方向，使 set-valued geometry 没有达到预声明的独立收益。

这不是安全性失败，而是行为增量失败。不能删除 H-001 或把 `0.05` 阈值事后缩至 `0.01`，也不能把“无有害方向”重新包装为新算法成功。

## 压力控制

- point-identified control 的方向差为 0，符合退化要求。
- zero-in-set 与 wide-interval 控制均 100% abstain，确认几何实现能够识别不可分集合。
- misspecified interval 在 100% 运行中选择与真实 clean gradient 相反的方向，明确暴露覆盖错误时保证失效。
- symmetric-zero-bias 与 equal-compute/audit 控制均保持 cosine 1.0，未通过不公平预算制造增益。

## 决策

H-027 记为 `REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED`，不使用局部修复额度。本轮消耗 1 单位，累计预算为 61/100；活跃分支回到 H-001、H-005、H-014 三条，低于施工方案的并行下限 4。G6 仍未通过，不允许进入 E1 或语言模型训练。
