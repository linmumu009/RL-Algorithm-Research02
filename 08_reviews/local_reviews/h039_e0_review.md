# H-039 E0 本地审查

## 审查结论

执行记录符合预注册：正式运行前 `main` 与远程一致，H-039 的 code/config/test Git blob 与绑定提交 `ddb66391e42bbaf5e63c85949df6c4fac8d32414` 完全一致，两个正式结果路径均不存在。随后仅启动一个正式进程，保存 1080 个 seed cell、40 个控制行和 216-cell 汇总。没有调整角点枚举、共享 group centering、接受规则、半径匹配、网格、基线、控制或阈值，也没有选择性重跑。

## 安全性审查

- 216/216 个正式 cell 均覆盖真实通道，false-certified sign rate 最大为 0。
- harmful sample update rate 在全部正式 cell 中也为 0。
- 72 个 strong cell 的最小 cosine 为 0.999718、最小 certified mass 为 0.666667，均通过门槛。
- point-limit difference 为 0；H-010 与随机过滤的接受数完全匹配；parameter/channel radius 最大差为 0。

这些结果支持“H-039 是保守且实现正确的逐 completion 符号证书”，但安全性本身不构成超过 H-001/H-027 的算法贡献。

## 增量与反例审查

- 96 个异质、非共线目标 cell 中 qualifying gain cell 为 0。
- 相对最佳 exact non-oracle 的最大 cosine gain 仅 `0.000001279`，harmful-rate reduction 最大为 0。
- 相对 H-001 的最大 cosine gain为 `0.000124916`，最大 harmful-rate reduction 为 0.083333，仍低于预注册门槛。
- H-001 与 H-027 的全体 seed-cell 平均 cosine 分别为 0.999887 和 0.998942，明显高于 H-039 的 0.877656。
- 0.20 宽区间产生 15 个 zero-mass cell 和 4 个负 cosine cell；最坏聚合 cosine 为 -0.218434，说明逐样本符号正确不保证选择性加权后的全局梯度正确。

misspecified interval 控制确实漏掉真实通道，但该冻结压力构造没有造成错误认证。这意味着它没有验证“漏覆盖必然破坏证书”的辅助预测；该控制不足不需要也不允许结果后加压重跑，因为正式零增益结论已经独立成立。

## 反事实判断

若尚未投入本轮时间，仍会执行这项 1 单位 E0：它清楚区分了“逐样本符号安全”和“超过 point/global-set correction 的算法价值”。不会删除 H-001/H-027、降低 0.05/0.10 增益阈值、只报告 strong cell、扩大 misspecification、改变 margin 权重或在看到宽区间退化后重跑。

## 项目级判断

H-039 不做局部修复并移入淘汰目录。活跃组合剩 3 条，预算累计 71/100，剩余 29 单位低于 30 单位探索/重启储备。G6 仍未通过；下一步必须先做零预算治理复核，不得自动启动第五轮候选生成、E1 或语言模型训练。
