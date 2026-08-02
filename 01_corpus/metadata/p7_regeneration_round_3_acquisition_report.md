# P7 第三轮定向补证报告

## 结果

- 从官方 arXiv 新增 12 篇直接近邻，语料由 219 篇增至 231 篇。
- 12 份 PDF 均可读，官方标题、作者、摘要和时间元数据完整，无 arXiv ID 重复或文件哈希重复。
- 全部 PDF 经 MinerU 托管 Precise Parsing API 的 `vlm` 模式解析为标题命名 Markdown；批次 ID 为 `48f60d61-5676-4129-9d36-3d58c4b8b742`。
- 两篇 PDF 的摘要未带显式 `Abstract` 标题，但标题后的摘要正文完整；其余解析稿均显式包含 abstract 与 references。

## 补证目的

本轮不再扩展 H-021 的 proxy bridge 或 H-027 的 convex-set direction，而检查六类不同信息结构：adaptive holdout reuse、privacy-stable release、mode-wise verifier dynamics、causal reward representation、trusted update subspace、adversarial auditor 以及 policy-feedback reward adaptation。

关键结论如下：

1. counterfactual invariance 与 causal/noncausal representation 已由 2501.09620 和 2601.21350 直接提出。
2. trusted clean-direction projection 已由 2605.25189 直接提出，与 H-005 的 nuisance projection 也高度邻近。
3. hacker-auditor gate 已由 2602.01750 直接实现；实时 policy-hidden-state reward adapter 已由 2601.22664 提出。
4. mode-wise Youden phase transition 已由 2601.04411 形式化；把它变成 mode gate 或 inverse-channel scaling 会回到 H-001。
5. differential privacy 在 RLHF 中已有隐私保护用途，但现有 2310.16960/2603.22563 主要保护训练记录并研究隐私—效用，不直接检验“固定 clean-audit 集被自适应策略反复查询后过拟合”的 verifier-correction 问题。
6. H-033 因此仅作为严格受限的跨领域机制迁移暂留；Reusable Holdout、Generic Holdout 和 private reward model 必须作为强基线，而不能只和 naive reuse 比较。
