# Q-001 第四轮替代假设筛选

## 结果先行

H-033 失败后生成 H-039–H-044 六个候选，并从官方 arXiv 增补 6 篇直接近邻，经 MinerU `vlm` 全量解析后语料达到 237 篇。五个候选被直接论文或既有刻画覆盖；仅条件性保留 H-039 `Channel-Set Advantage Sign Certificate`，活跃组合恢复为 H-001、H-005、H-014、H-039 四条。

| 候选 | 机制家族 | 总分 | 可证伪 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| H-039 | audit-calibrated sign identification | 89 | 15 | 10 | 条件性保留并预注册 1 单位 E0 |
| H-040 | correlated proxy DRO | 88 | 14 | 2 | 2604.12086 已直接优化 correlated proxy family |
| H-041 | reward-uncertainty action set | 87 | 14 | 2 | 2606.03962 已直接提出 reward-distribution action portfolio |
| H-042 | confidence reward design | 85 | 14 | 1 | 2607.04332 已直接研究 non-hackable confidence reward |
| H-043 | causal reward tampering | 86 | 13 | 2 | 1908.04734 已直接给出 current-RF/causal tamper-resistance 原则 |
| H-044 | reward-order geometry | 82 | 13 | 1 | 2209.13085 已刻画 restricted-policy unhackability，候选无新更新机制 |

## 为什么只暂留 H-039

H-039 改变的是识别粒度：H-001 给出一个通道点估计，H-027 对整个 clean-gradient identified set 选一个全局方向，而 H-039 先把 contextual FP/FN 通道集合映射为每个 completion 的 clean-advantage interval。只有整个区间严格同号时才更新，并以最坏绝对 margin 加权，目标是避免组内正负错误互相抵消后掩盖 harmful sample updates。

这一对象与 SignCert-PO 很接近，因此 E0 强制匹配参数扰动半径与通道区间、接受率、audit 和计算预算。只有在异质通道、mixed-sign 与非共线条件中同时超过 H-001 和最佳 exact non-oracle 的收益，才支持独立机制；行为等价或仅靠 abstention 即淘汰。

## 预算与边界

- 第四轮筛选消耗 4 单位，累计 70/100，恰好保留 30 单位探索/重启储备。
- H-039 只获得 1 单位解析/合成 E0 的预注册资格，本轮尚未执行正式实验。
- 不允许进入 E1、语言模型训练或扩大验证；不允许删掉 interval misspecification、zero-mass、matched-acceptance 或 matched-radius 控制。
- 下一项唯一允许任务：实现并冻结 H-039 E0，先提交不可变代码并绑定预注册，再只执行一次正式网格。
