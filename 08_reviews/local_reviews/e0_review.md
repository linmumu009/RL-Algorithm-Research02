# Q-001 E0 本地审查

## 结论

`Q001-E0-v1` 按提交 `34fea81eb28bdba546580ba91e68d1cca5065805` 的冻结实现一次性运行。六个预注册均有结果；未训练语言模型、未更换指标、未修改网格、未选择性重跑种子。

- 保留进入后续低成本验证：H-001、H-005、H-014。
- 淘汰失败预测：H-004、H-018。
- 淘汰行为等价：H-008。

## 反例审查

- H-004：所谓 outcome-regression augmentation 在代数上消去 regression，精确等于 channel-only correction。channel 错而 regression 对时 bias 为 0.191，高于 0.05 失败线。
- H-008：主预测成立不能证明机制独立；相对 direct pair average 的 cosine 优势为 -0.00016，应按等价候选淘汰。
- H-018：可交换条件下覆盖与误更新控制均成立，但下移漂移的 undercoverage 为 0.143–0.992，且 audit 越大越确信错误分布，不允许把论点事后缩成“仅 exchangeable”。

## 项目级判断

三条保留分支仍值得从零选择，因为它们分别改变 reward channel、nuisance direction 和 audit acquisition，且改善均在预注册控制中消失或保留了正确退化行为。当前活跃分支降至 3，低于施工方案的并行下限；在 E1 前需要补充并筛选至少一个机制不同的新分支。
