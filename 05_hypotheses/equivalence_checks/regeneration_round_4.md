# P7 第四轮定向再生成等价性审查

## H-039 强等价矩阵

| 对比 | 潜在等价 | E0 必须保留的区分 |
|---|---|---|
| H-001 point channel correction | 通道点识别时都恢复 clean advantage | point-identified limit 的方向差必须不超过 0.01；区间非零时 H-039 必须在 mixed-sign cell 给出独立安全或方向收益 |
| H-027 global gradient-set direction | 都在 audit-compatible set 上做稳健选择 | H-027 先聚合再选全局 maximin direction；H-039 先逐 completion 认证符号再聚合。若输出方向、abstention 和收益在全部网格等同，则 H-039 淘汰 |
| H-010 uncertainty mask | 都可能丢弃不确定 completion | H-010 仅按不确定度/风险排序过滤；必须加入等接受率随机过滤和 uncertainty mask，排除“少更新自然少犯错”的解释 |
| SignCert-PO | 都认证 advantage sign 后更新 | 匹配 parameter-radius 与 channel-radius、相同 completion 与接受预算；只有 audit-channel identification 在异质/错配 cell 的额外收益可计入差异 |
| scalar lower-bound pessimism | 都使用最坏情形 margin | 标量悲观只缩 reward；H-039 必须由正负两侧符号区间产生不同逐样本行为，而非统一缩放 |
| oracle clean advantage | 都可输出正确符号 | oracle 只作为上界，不计入非 oracle 增量比较 |

## 预声明判定

- 有效覆盖下 false-certified sign rate 必须不超过 0.05。
- strong-identified cell 中 clean-gradient cosine 至少 0.90，harmful sample update rate 至多 0.05，certified mass 至少 0.50。
- 至少 3 个 heterogeneous、non-collinear 目标 cell，相对 H-001 和最佳 exact non-oracle 同时达到 `cosine gain >= 0.05` 或 `harmful-rate reduction >= 0.10`。
- 等接受率过滤后增益消失、与 H-027/SignCert-PO 行为等价、或只在 interval misspecification 下成立，均直接判失败。

H-039 的保留是有条件的：E0 必须证明“逐 completion 的 audit-channel sign identification”产生独立可观察差异，不能依靠名称、新坐标或降低样本利用率。
