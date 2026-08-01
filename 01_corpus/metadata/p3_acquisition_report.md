# P3 定向语料补充报告

## 结果

- 新增论文：20 篇。
- 新增后总量：200 篇唯一 arXiv 论文。
- 定向分布：验证器噪声/奖励投机 11 篇，轨迹复用 2 篇，异步训练与吞吐/延迟 7 篇。
- PDF：20/20 下载成功并可读。
- MinerU Markdown：20/20 解析成功。
- 官方 arXiv 元数据：200/200 完整。
- 去重：无重复 arXiv ID，无相同 SHA-256 文件。

## 选择原则

本轮没有按关键词随机扩容，而是只补 G2 明确指出的三个薄弱区。系统论文只用于成本、吞吐、延迟和 staleness 证据，不计作算法新颖性；奖励投机论文区分“检测”“干预”“迁移风险”，避免把现象报告误当作训练修复机制。

## 可追溯入口

- 选择清单：`02_literature/extended/p3_targeted_supplement.csv`
- 直接证据：`02_literature/claim_cards/p3_evidence_claims.csv`
- 全量清单：`01_corpus/inventory.csv`
- MinerU 全文：`01_corpus/text_cache/mineru/extended/`
