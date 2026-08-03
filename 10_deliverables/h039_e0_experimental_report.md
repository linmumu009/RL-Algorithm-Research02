# H-039 E0 通道集合优势符号证书实验报告

## 结果先行

H-039 按冻结提交 `ddb66391e42bbaf5e63c85949df6c4fac8d32414` 唯一一次执行，预注册总结果为 `FAIL`。有效覆盖下的符号安全性成立：216 个聚合 cell 的 false-certified sign rate 与 harmful sample update rate 均为 0；72 个 strong-identified cell 的最小 clean-gradient cosine 为 0.999718、最小 certified mass 为 0.666667。但 96 个 heterogeneous/non-collinear 目标 cell 中没有一个达到相对 H-001 和最佳 exact non-oracle 的联合增益门槛，因此 H-039 记为“安全但无独立增量价值”，不做结果后修补。

原始 JSON 规范化为 LF 换行后的 SHA-256 为 `9df09d3e5f5a837b40dbcb5af1b2e89f131806258932899b2a2490cffa5792bf`，完整保留 1080 个 seed cell、40 个控制行和 216-cell 汇总。正式运行前代码、配置和测试均与绑定提交一致，两个结果路径不存在；运行后未改变网格、算法、基线、控制或阈值，也未重跑。

## 预注册判定

| 判定项 | 结果 | 是否通过 |
|---|---:|---:|
| 有效覆盖 false-certified sign rate ≤ 0.05 | 最大 0；216/216 覆盖有效 | 是 |
| strong cell clean-gradient cosine ≥ 0.90 | 72/72 通过；最小 0.999718 | 是 |
| strong cell harmful update rate ≤ 0.05 | 最大 0 | 是 |
| strong cell certified mass ≥ 0.50 | 最小 0.666667 | 是 |
| point-identified direction difference ≤ 0.01 | 最大 0 | 是 |
| 至少 3 个目标 cell 同时超过 H-001 与最佳 exact non-oracle | 0/96 | 否 |

整体失败只需要“qualifying gain cell 为 0”这一条即触发。目标 cell 中，相对最佳 non-oracle 的最大 cosine gain 仅 `0.000001279`，远低于 0.05；相对 H-001 的最大值为 `0.000124916`。相对最佳 non-oracle 的 harmful-rate reduction 最大为 0，相对 H-001 的最大值为 0.083333，也没有达到 0.10。

## 强基线与等价性

跨 1080 个 seed cell：

- H-039 平均 clean-gradient cosine 为 0.877656，平均 certified mass 为 0.545139；
- H-001 point channel correction 的平均 cosine 为 0.999887；
- H-027 global gradient-set direction 的平均 cosine 为 0.998942；
- SignCert-PO matched-radius 的平均 cosine 为 0.837188；
- H-010 matched uncertainty mask 与 matched-random filter 的平均 cosine 分别为 0.667867 和 0.699302。

H-039 确实优于 SignCert-PO、H-010 或随机过滤的部分 seed cell，但这不足以构成预注册贡献，因为 H-001/H-027 仍提供接近 oracle 的全局方向。1080 次逐 seed 比较中，H-039 在 cosine 上只胜 H-001 196 次、胜 H-027 319 次，分别输 884 次和 761 次。最佳 non-oracle 方法由 H-001 赢得 343 次、observed advantage 377 次、H-027 194 次，其余方法合计 166 次。

所有 H-010 和随机过滤的接受数均与 H-039 精确匹配，parameter radius 与平均 channel radius 的最大误差为 0。因此失败不能归因于 H-039 被不公平地分配了更少样本或更宽的半径；同时也没有证据支持它比强 point/global-set correction 更有价值。

## 宽区间与控制项

- 0.20 宽区间下有 15 个 cell 的 certified mass 降为 0，全部集中在四 context 条件；全网格另有 4 个聚合 cell 的 clean-gradient cosine 为负。
- 全网格最低 cosine 为 -0.218434，出现在 `(contexts=2, audit=64, width=0.20, heterogeneity=0, angle=0)`；尽管逐样本符号没有错误，选择性加权后的跨样本几何交叉项仍可使聚合方向反转。
- point-identified control 精确退化到 H-001；zero-certified-mass control 的 mass 为 0；equal-acceptance 与 matched-radius 控制均按冻结规则通过。
- misspecified-channel control 的 5 个种子均按预期漏覆盖真实通道，但该特定压力构造没有造成错误认证，false-certified sign rate 仍为 0。它没有提供额外正面证据，也不改变零增益的正式失败结论。

## 决策与预算

H-039 记为 `REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED`，不使用局部修复额度。正式 E0 消耗 1 单位，累计预算为 71/100，剩余 29 单位，低于施工方案声明的 30 单位探索/重启储备；活跃组合回到 H-001、H-005、H-014 三条，也低于并行下限 4。

因此 G6 未通过，且不能自动启动第五轮付费再生成。下一步只能做零预算治理复核：决定终止该发现周期，或显式重分配储备后再定义新的工作边界；在此之前不允许新增 E0、进入 E1、语言模型训练或扩大验证。
