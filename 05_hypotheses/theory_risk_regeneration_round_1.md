# P7 定向再生成理论风险审查

## H-021 主要风险

1. **排除限制错误**：随机 verifier view 可能改变实际可判定性，而不只是暴露 exploit。
2. **negative-control outcome 无效**：diagnostic W 可能被 clean correctness 直接影响，破坏 proxy 图。
3. **completeness/rank 不足**：proxy 对 latent exploit 不够相关时，bridge 不可识别或病态。
4. **稀疏 audit 过拟合**：条件逆问题会放大有限样本误差。
5. **伪新颖性**：线性对称特例可能重新退化为 H-008 pair average，或退化为 H-005 residualization。

## E0 必须先验证

- 在已知离散生成模型中验证 exact bridge 恒等式。
- 报告强 proxy 与弱 proxy 的 condition number、bias、MSE、cosine 和 false-positive update rate。
- 包含 revealed-latent oracle、invalid-exclusion 和非对称 proxy cell。
- bridge 在 rank collapse 时必须显式失败；若仍显示稳定改善，说明实现或指标不可信。

风险级别：高。保留理由不是理论已经成立，而是失败条件明确、E0 成本仅 1 单位、且机制与现存三分支不同。
