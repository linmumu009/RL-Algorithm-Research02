# P7 第二轮定向再生成等价性审查

## 排除结果

- H-028 与 distributional reward 的 entropic/KL-DRO effective reward 同式。
- H-029 的 confidence sequence 已有，转成 update release 不改变估计对象，且可作为 H-014 的辅助停止规则。
- H-030 复活 H-009 reward ensemble；在相关错误下多数/几何聚合都没有 clean-truth identification。
- H-031 是 arXiv:2607.05904 的 commit-first training channel 重命名。
- H-032 是 CHERRL/RHDA 检测后附加 H-020 reset schedule。

## H-027 与已有分支/失败候选的差异

| 对比 | 不等价理由 | 必须验证的退化条件 |
|---|---|---|
| H-001 | H-001 选择一个点估计通道并输出单一 corrected gradient；H-027 保留全部可行通道诱导的 gradient set | interval width 为 0 时必须退化为 H-001 点方向 |
| H-005 | H-005 从单一观测梯度投影 declared nuisance；H-027 不声明 nuisance 子空间，而对一组 clean-gradient 向量做统一上升方向优化 | identified set 位于单一直线时可能看似 shrinkage，E0 必须比较 norm-matched shrinkage |
| H-014 | H-014 决定审计哪些样本；H-027 决定给定审计区间后是否存在可证安全方向 | audit 全量时 H-027 应退化为点梯度，H-014 的采集差异消失 |
| H-018 | H-018 对每个标量 reward 构造静态下界；H-027 对 score-weighted vector identified set 求分离方向 | 一维同号 score 时两者可能相同，E0 必须包含二维非共线反例 |
| H-021 | H-021 通过 proxy bridge 点识别 reward；H-027 不要求 proxy completeness，只在区间集合中做 partial identification | channel interval 错误排除真值时保证必须显式失败 |
| scalar DRO | scalar DRO 先选 worst-case reward 再求梯度；H-027 直接优化对所有可行 gradient 的最坏内积 | 若 E0 中行为与 scalar DRO 相同则按等价淘汰 |
