# P7 第四轮定向再生成新颖性审查

## 审查范围

H-039–H-044 与当前 237 篇唯一语料、H-001–H-038 谱系及 H-021/H-027/H-033 的正式失败结果逐项比较。判断依据仍是数学对象、信息结构和可观察行为，而不是名称或组件堆叠。

## 结果

| 候选 | 最近工作/分支 | 结论 |
|---|---|---|
| H-039 | 2604.02986、H-001、H-010、H-027、H-034 | 条件性保留：现有 SignCert-PO 在 reward-model 参数扰动球内认证优势符号；H-039 改为由 clean audit 识别 verifier FP/FN 通道集合，再逐 completion 认证 clean-advantage 符号，信息结构不同但仍须 E0 排除行为等价 |
| H-040 | 2604.12086 | 淘汰：论文已直接对 r-correlated proxy reward family 做 worst-case return 优化 |
| H-041 | 2606.03962 | 淘汰：论文已直接以 reward distribution 和非线性 action-set objective 诱导行为组合 |
| H-042 | 2607.04332 | 淘汰：论文已直接定义并评估 non-hackable confidence reward family |
| H-043 | 1908.04734 | 淘汰：current-RF optimization 与 causal influence diagram 的 tamper-resistance 原则已直接提出 |
| H-044 | 2209.13085 | 淘汰：restricted policy class 上 unhackability 的必要充分条件已被刻画；验证条件不是新更新机制 |

## H-039 的最小新颖性边界

H-039 不声称发明 advantage sign certification、置信集合、通道反演或 selective update。候选贡献只限于：将独立 clean audit 给出的 contextual verifier-channel confidence set 映射为每个 completion 的 clean-advantage interval；只有当整个区间严格位于零的一侧时才更新，并用最坏绝对 margin 加权。

它与 SignCert-PO 的差异必须体现为可观测行为：在 reward-model 参数半径与 channel interval 经过匹配后，H-039 仍需在异质通道、组内正负优势混合且 score/gradient 非共线的 cell 中降低 harmful sample update 或改善 clean-gradient cosine。若只是在另一坐标系中构造同一符号 gate，或收益完全来自更低接受率，则淘汰。

## 结论

仅 H-039 获得 1 单位解析/合成 E0 的预注册资格。H-040–H-044 不进入实验，不得作为 H-039 的组件重新计算新颖性。
