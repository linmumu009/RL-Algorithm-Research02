# Q-001 E0 低成本决定性实验报告

## 结果先行

六个预注册实验已一次性完成。H-001、H-005、H-014 通过并保留；H-004 和 H-018 触发原失败阈值；H-008 虽通过主预测，但被强基线证明行为等价，因此淘汰。G6 尚未通过，任何候选都没有获得语言模型训练或扩大验证权限。

原始结果规范化换行后的 SHA-256：`6bcc39b76033f5639a5b37311f5c21e92154622e6495cada399655b40752b688`。

| 分支 | 预注册结果 | 关键证据 | 最终动作 |
|---|---|---|---|
| H-001 | PASS | 4/4 异质 cell 的 cosine gain ≥0.05 且 CI 下界 >0；同通道 gain=0 | E0_VALIDATED |
| H-004 | FAIL | regression-only-correct bias=0.191；增强式精确退化为 channel-only | REJECTED_FAILED_PREDICTION |
| H-005 | PASS | 非因果 cell 的伪梯度质量减少 99.68%，clean cosine 未下降 | E0_VALIDATED |
| H-008 | PASS + 等价失败 | 解释 63.78% exploit variance，但相对 direct-pair-average 优势 -0.00016 | REJECTED_EQUIVALENT |
| H-014 | PASS | 四个预算均比最佳基线降低 MSE 10.75%，最小 ESS 比 58.59% | E0_VALIDATED |
| H-018 | FAIL | exchangeable 通过；shift undercoverage 14.32%–99.21% | REJECTED_FAILED_PREDICTION |

## 可解释性

H-001 的收益随条件噪声异质性单调扩大，并在条件通道相同时精确退化，支持“全局通道平均产生方向偏差”的因果解释。H-005 说明显式 nuisance 投影能移除可操纵方向，但 causal stress 只保留约 73.36% 信号，E1 必须加入真实 audit 估计误差与 signal-protection 对照。H-014 的收益仅略高于 10% 门槛，下一阶段必须用实际抽样轨迹验证，而不能把解析方差优势直接外推到训练收益。

## 失败的研究价值

H-004 给出一个代数负结果：在当前表达式中不存在双重稳健性。H-018 给出覆盖负结果：更多旧分布 audit 会让漂移后的错误置信更强。H-008 证明“有用”不等于“新颖”，其行为由直接配对平均完全解释。

## 边界与下一步

- 本轮预算：6 单位；累计 51/100。
- 未进行语言模型训练。
- G6 仍为进行中；三个保留分支只通过 E0。
- 活跃分支少于并行下限，下一项先生成并完整筛选至少一个非等价替代分支，再为合格组合预注册 E1。
