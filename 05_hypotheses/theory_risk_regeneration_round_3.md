# P7 第三轮定向再生成理论风险审查

## H-033 主要风险

1. **直接先验等价**：Gaussian audit query release 可能只是 Reusable Holdout/DP adaptive analysis 的领域改名。
2. **噪声毁掉方向**：低 privacy budget 下二维以上的 Gaussian noise 可能使 clean-gradient cosine 低于 naive 或 sample splitting。
3. **预算快速耗尽**：连续向量 release 比 binary Generic Holdout 泄露更多，可能在少数轮后停止，无法支持训练。
4. **敏感度假设错误**：score contribution 若未严格裁剪，Gaussian calibration 和 generalization 主张均失效。
5. **分布外推不足**：DP 稳定性控制对固定总体的 adaptive generalization，但不自动解决真实 policy occupancy 改变后的 population drift。
6. **隐私与安全混淆**：个人记录隐私不是 reward-hacking 鲁棒性；E0 必须报告独立 population clean truth，而不能用 epsilon 本身作为成功指标。
7. **维度扩展**：E0 的低维向量查询不能证明语言模型参数空间中噪声可承受。

## E0 必须先验证

- 固定一个隐藏 clean-audit 表和独立 population truth，让查询由前轮 release 自适应选择。
- 同时比较 naive exact reuse、fresh-audit oracle、disjoint sample splitting、Reusable Holdout/Thresholdout、Generic Holdout、一次训练 private reward model和 privacy-off limit。
- 使用同一 contribution clipping、同一 audit 样本、同一查询序列预算与相同 score norm。
- 报告 privacy accountant、population bias、false-positive direction、cosine、usable rounds 和 budget exhaustion。
- 非自适应查询中不得宣称收益；permuted audit、zero-gradient 和 population drift 必须作为负控。

风险级别：高。保留只代表存在 1 单位可证伪路径，不代表新颖性已经成立。
