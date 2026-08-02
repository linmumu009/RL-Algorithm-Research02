# H-021 E0 负控桥接实验报告

## 结果先行

H-021 按冻结提交 `8b167359c3c114412af397ab69d30875d3fa1bdf` 一次性执行，预注册结论为 `FAIL`。桥接估计在强代理区间保持低偏差，也正确暴露了弱代理的秩不稳定，但没有提供超过强基线的 clean-gradient 方向收益。因此淘汰 H-021，不做结果后局部修补。

原始结果 SHA-256：`9aeaef5e30e286392fe8082e2ecb8a73af35190ef11b4023da0281956bdc5677`。

## 预注册判定

| 判定项 | 结果 | 是否通过 |
|---|---:|---:|
| 每个强代理 cell 的 gradient bias < 0.05 | 0.005995–0.026519 | 是 |
| 至少 3 个 latent-exploit cell 的 cosine gain ≥ 0.05 且 CI 下界 > 0 | 0/6 | 否 |
| 与 revealed-latent oracle 的 cosine 差 ≤ 0.02 | 最大 0.014793 | 是 |
| 弱代理条件数相对强代理恶化 | 中位数 678.12 vs 3.20 | 是 |

唯一失败项也是核心机制主张：所有 6 个强代理 latent-exploit cell 的平均增益都为负。跨 30 个种子运行，桥接 cosine 平均为 0.994602，最佳非 oracle 基线为 0.999191，平均差为 -0.004589。

## 强基线归因

在 30 个强代理 latent-exploit 运行中，direct proxy regression 赢 28 次，H-005 nuisance projection 赢 2 次。全量 90 个有效运行中，direct proxy regression 赢 59 次，direct pair average 赢 24 次，H-005 赢 5 次，H-001 赢 2 次。这说明 sparse audit 已经允许直接条件回归吸收代理信息；额外求解逆 bridge moment 没有产生独立决策价值，反而引入有限样本方差。

## 压力控制

- 弱代理行为正确：代理相关性降至 0.55/0.52/0.50 后，条件数中位数从强区间的 3.20 升至 678.12，部分 cell 的 bias 超过 0.05。
- revealed-latent oracle 对照通过：强区间与 oracle 的 cosine 差均未超过 0.02，说明桥接方程本身并非完全错误。
- invalid-exclusion 使匹配 cell 的 bias 从 0.007693 增至 0.011224，但没有形成足以挽救独立机制主张的新判据。

## 决策

H-021 记为 `REJECTED_FAILED_PREDICTION_AND_DOMINATED`，不使用两次局部修复额度。原因不是 bridge 不能拟合，而是它没有超过使用相同审计信息的直接回归强基线。累计预算增至 56/100；活跃分支回到 H-001、H-005、H-014 三条，低于施工方案的并行下限 4。G6 仍未通过，不允许进入 E1 或语言模型训练。
