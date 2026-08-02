# P7 第四轮定向再生成理论风险审查

## H-039 主要风险

1. **覆盖错配**：clean-audit channel interval 若漏掉真实 FP/FN，符号证书可变成高置信错误；misspecified interval 必须显式暴露失败。
2. **组中心依赖**：completion advantage 依赖同组 reward centering；逐样本通道角点未必能独立取到，若忽略共享基线会得到过宽或错误区间。
3. **选择偏差**：只更新可认证样本会改变 policy occupancy；短期 harmful rate 下降不保证长期总体 clean return 上升。
4. **空证书退化**：宽区间、弱审计或接近零优势时 certified mass 可能趋近零，安全但无学习价值。
5. **等价 SignCert-PO**：channel confidence set 经过局部线性化后可能只是 reward-parameter perturbation ball 的重参数化。
6. **等价 H-027**：若逐样本 interval 的加权和与全局 gradient convex set 的 maximin 解一致，则没有独立机制。
7. **接受率混淆**：H-039 可能仅因丢弃更多样本而降低 harmful update；必须进行 matched-acceptance 比较。
8. **有限维外推**：二维/低维合成 E0 只能检验识别逻辑，不能证明语言模型训练稳定性。

## E0 必须先验证

- 用独立 clean audit 冻结 contextual FP/FN confidence sets，并在有效覆盖与漏覆盖两种条件分别评估。
- 在组内同时放置正、负和近零 clean advantage，正确处理共享 group baseline，不允许把 completion 独立角点当作任意可同时实现。
- 比较 H-001、H-027、H-010、SignCert-PO、scalar pessimism、matched-acceptance filter 与 oracle。
- 报告 false-certified sign、harmful sample update、clean-gradient cosine、certified mass 和 point-limit difference。
- 保留 zero-certified-mass、symmetric channel、mixed-sign cancellation、interval misspecification 和 matched-radius 控制。

风险级别：高。预注册只说明存在低成本可证伪路径，不构成新颖性或有效性的确认。
