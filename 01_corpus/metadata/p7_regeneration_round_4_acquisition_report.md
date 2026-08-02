# P7 第四轮定向补证采集报告

## 结论

从官方 arXiv 新增 6 篇直接近邻，语料由 231 篇增至 237 篇。全部 PDF 可读，官方标题、作者、年份和摘要已写入清单；未发现 arXiv ID 重复、文件哈希重复或元数据缺失。

## MinerU 解析

- 模式：托管 Precise Parsing API，`model_version=vlm`，公式与表格解析保持开启。
- 批次：`7a4f8c1c-81fb-4789-8576-b3a6c1e8450c`。
- 状态：6/6 `done`，结果压缩包和临时解压目录已由技能脚本清理，仅保留标题命名 Markdown。
- 完整性：6 份 Markdown 均以 H1 开始、包含完整前置摘要文本和 References；`1908.04734` 的摘要为标题后无显式 Abstract heading 的正文，其余 5 篇含显式 Abstract。

## 筛查作用

本批次用于区分 audit-calibrated advantage-sign identification 与已有 SignCert-PO，同时直接封闭 correlated-proxy DRO、reward-uncertainty action set、confidence reward design、causal reward tampering 和 policy-class unhackability 等换名方向。
