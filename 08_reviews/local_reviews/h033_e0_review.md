# H-033 E0 本地审查

## 审查结论

执行记录符合预注册：运行前代码和配置相对绑定提交 `90dbd2ea0db51d1a37adabb5678baab8e13bf24d` 零差异，两个正式结果路径均不存在；随后只启动一个正式进程并完整保存 5760 个 method-seed 行、35 个控制行和 1152 个汇总 cell。没有调网格、改阈值、删除低 rho/高维/漂移控制或选择性重跑。正式结论为 `FAIL`。

## 隐私会计与实现审查

- Gaussian release 使用冻结的向量均值 L2 sensitivity 与 `sigma = sensitivity / sqrt(2 rho_per_query)`；总 rho 最大浮点误差为 `4.884981e-15`。
- privacy-off limit 与 naive exact audit reuse 的 bias、cosine 和自适应 candidate sequence 完全一致。
- clip-violation 控制的最大裁剪后 contribution norm 为 `1.0000000000000002`，属于浮点容差。
- 因此本次失败不是 privacy filter 未执行、隐式少跑查询或额外正则化造成的实现伪差异。

## 反例、效用和等价性审查

- 118/144 个 H-033 cell 的 bias 超过 0.05，127/144 的 cosine 低于 0.90，82/144 的 false-positive direction rate 超过 0.05。
- 最坏的 `(audit=128, queries=250, rho=0.1, d=32)` cell 同时达到 bias 3.091019、cosine 0.019939 和方向错误率 0.4488。
- 24/144 个 cell 没有达到相对 disjoint split 的 2× usable-round 要求，最小比率为 1.0。
- qualifying adaptive gain cell 为 0；相对 naive 和最佳 exact non-oracle 的最大 cosine gain 分别仅 0.000495 和 0.003427。
- H-001 single-query 与 Thresholdout 在准确性上显著优于 H-033，说明可复用审计接口上的 DP 合规不自动转化为 clean-gradient 鲁棒性或算法增量。

## 反事实判断

若尚未投入本轮时间，仍会执行这项 1 单位 E0，因为它同时排除了“隐私会计错误”和“弱基线制造成功”两种解释。不会删除低 rho/高维 cell、把 privacy-off 的好结果归给 H-033、降低 0.05 gain 门槛，或在看到失败后重新分配每轮 rho。

## 项目级判断

H-033 不做局部修复并移入淘汰目录。活跃组合再次降至 3 条；下一步必须从新的未覆盖机制族开展第四轮定向再生成，在恢复至少 4 条非等价分支前不进入 E1 或语言模型训练。
