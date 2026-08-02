# P7 第二轮定向再生成补证报告

## 结果

- 新增 7 篇官方 arXiv PDF，语料由 212 篇增至 219 篇。
- `build_inventory.py --fetch-metadata` 结果：219 个唯一 arXiv ID、219 条官方元数据、0 缺失、0 不可读、0 重复。
- 7 篇 PDF 均由 MinerU hosted precise parsing API（`vlm`）解析，批次 ID：`85b35e98-be4b-409b-b783-136a95d76b90`。
- 7 份 Markdown 均具有 H1 标题、Abstract 和 References，文件大小为 38–195 KB。

## 补证目的

| arXiv | 作用 |
|---|---|
| 2210.10768 | 排除 anytime-valid audit release：已提供适应性策略、漂移和任意停止下的 off-policy confidence sequence |
| 2312.09244 | 排除简单 cross-judge ensemble：共同错误使 ensemble 只能缓解、不能消除 hacking |
| 2507.20550 | H-027 最近一般先验：部分识别、marginal sensitivity set 和 worst-case policy improvement |
| 2606.04923 | 排除 hacking-onset reset：CHERRL 已直接构造 onset detector/testbed |
| 2606.09073 | 排除分布式标量悲观 reward：已统一 Bayesian/KL-DRO effective reward |
| 2606.19818 | 排除 conformal uncertainty reweighting：UARM 已直接用于 GRPO advantage |
| 2607.05904 | 排除 commit-first verifier：blind/commit-first judging 已直接防止 shared plausibility basin |

## MinerU 执行说明

API 的 7 个任务全部达到 `done`，结果均已下载保存。脚本在最后打印汇总时因一篇标题含非 GBK 字符触发 `UnicodeEncodeError`；这发生在文件保存之后。随后逐文件检查确认 7 份 Markdown 完整，因此没有重提任务或重复解析。
