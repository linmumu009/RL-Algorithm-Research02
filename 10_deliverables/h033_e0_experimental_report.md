# H-033 E0 可复用审计梯度实验报告

## 结果先行

H-033 按冻结提交 `90dbd2ea0db51d1a37adabb5678baab8e13bf24d` 唯一一次执行，预注册总结果为 `FAIL`。zCDP 会计、L2 clipping、隐私过滤器和 privacy-off 极限均通过实现控制，但私有向量 release 在低 rho、长查询和高维条件下产生严重效用损失，也没有超过 naive audit reuse、Reusable Holdout 或 H-001 single-query correction。因此淘汰 H-033，不做结果后局部修补。

原始 JSON 规范化为 LF 换行后的 SHA-256 为 `f07f69762e7e2b51d288feb1a76aebe9e7adcf4c8bf88fd626f82d70bef5f695`，包含 5760 个 method-seed 运行行、35 个控制行和 1152 个完整汇总 cell；其中 H-033 占 144 个预注册 adaptive cell。

## 预注册判定

| 判定项 | 结果 | 是否通过 |
|---|---:|---:|
| H-033 cell mean population-gradient bias ≤ 0.05 | 118/144 超标；最大 3.091019 | 否 |
| false-positive direction rate ≤ 0.05 | 82/144 超标；最大 0.448800 | 否 |
| clean-gradient cosine ≥ 0.90 | 127/144 未达标；最小 0.019939 | 否 |
| usable rounds ≥ 2× disjoint split | 24/144 未达标；最小比率 1.0 | 否 |
| 至少 3 个 adaptive cell 同时超过 naive 与最佳 exact non-oracle | 0/144 | 否 |
| nonadaptive 结果不计入正增益主张 | 已排除 | 是 |

最坏 cell 为审计量 128、查询 250、rho 0.1、维度 32：bias 为 3.091019、cosine 为 0.019939、方向错误率为 0.4488。该结果与冻结 Gaussian calibration 一致：全网格 sigma 从 0.005043 到 0.552427，固定总隐私预算在小样本、长序列和高维向量上需要较大噪声。

## 强基线和增量价值

H-033 跨全部 method-seed 行的平均 bias 为 0.307131、平均 cosine 为 0.471991、平均方向错误率为 0.138694。相比之下：

- H-001 single-query correction 的平均 bias 为 0.012746、cosine 为 0.988674、方向错误率为 0；
- naive exact audit reuse 的平均 bias 为 0.014500、cosine 为 0.985077、方向错误率为 0；
- Reusable Holdout/Thresholdout 的平均 bias 为 0.022236、cosine 为 0.960968、方向错误率约 0.000253。

H-033 相对 naive 的最大 cell cosine gain 仅 0.000495，相对最佳 exact holdout/private-reward baseline 的最大值仅 0.003427，均远低于预注册的 0.05；false-positive-rate reduction 最大值均为 0。720 个 H-033 seed-cell 比较中，最佳 exact non-oracle 是 Thresholdout 673 次、once-trained private reward model 47 次。不存在可支持独立算法价值的 adaptive cell。

## 实现与压力控制

- 每个 H-033 trace 均消耗完整声明 rho，最大浮点会计误差仅 `4.884981e-15`，没有实质超预算。
- privacy-off 与 naive reuse 的 bias、cosine 和 candidate sequence 逐项完全一致，证明差异来自预注册的隐私噪声，而不是额外隐藏算法。
- contribution-clip-violation 控制的最大裁剪后范数为 `1.0000000000000002`，符合数值容差。
- nonadaptive 控制平均 bias 0.106394、cosine 0.571974，不能把适应性选择当成唯一损失来源。
- population-drift 控制平均 bias 0.124585、cosine 0.441785、方向错误率 0.072，进一步触发预注册的漂移风险边界。

## 决策

H-033 记为 `REJECTED_FAILED_PREDICTION_AND_DOMINATED`，不使用局部修复额度。本轮消耗 1 单位，累计预算为 66/100；活跃分支回到 H-001、H-005、H-014 三条，低于施工方案的并行下限 4。G6 仍未通过，不允许进入 E1、语言模型训练或扩大验证。下一步只能从尚未覆盖、且不等价于 H-001/H-005/H-014/H-021/H-027/H-033 的机制族开展第四轮定向再生成。
