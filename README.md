# RL-Algorithm-Research02

强化学习算法研究项目，聚焦大模型后训练、对齐与相关算法资料的整理和研究。

## 当前状态

- 当前阶段：`P2_CORE15_AUDIT`
- 当前验收门：`G2`
- 状态：核心 15 审计结论为 `PASS_WITH_REPLACEMENTS`，等待人工批准后才能进入 P3。
- 本地语料：180 篇唯一 arXiv 论文，全部可读、无重复、官方元数据完整。
- 核心基底：15 篇，均已关联 PDF、MinerU Markdown、Paper Card 和机制矩阵。

## 关键入口

- 项目章程：[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md)
- 当前状态：[`research_state.yaml`](research_state.yaml)
- 施工方案：[`LLM_RL_Algorithm_Discovery_Construction_Plan.md`](LLM_RL_Algorithm_Discovery_Construction_Plan.md)
- 语料清单：[`01_corpus/inventory.csv`](01_corpus/inventory.csv)
- 核心审计：[`10_deliverables/core15_audit.md`](10_deliverables/core15_audit.md)
- 替换建议：[`02_literature/selection_audit/replacement_recommendations.md`](02_literature/selection_audit/replacement_recommendations.md)
- 机制矩阵：[`03_taxonomy/mechanism_matrix.csv`](03_taxonomy/mechanism_matrix.csv)

## 目录

项目按施工方案分为治理、语料、文献、分类体系、问题、假设、实验、结果、评审、决策和交付物目录。当前只完成 P0–P2 允许的治理、语料清点和核心覆盖审计；尚未创建算法候选或训练实现。

## 更新约定

每次项目更新均同步维护本 README 的版本说明，并在完成验证后提交、推送至远程仓库的 `main` 分支。

## 版本记录

### v0.3.0 — 2026-08-01

- 按施工方案重构完整项目目录，并建立项目章程、治理规则、状态文件、决策日志和数据结构规范。
- 将原 `reference/` 内容迁移到 `01_corpus/` 的原始论文、元数据和 MinerU 文本缓存目录。
- 纳入新增 11 篇核心候选及其 MinerU Markdown；补齐原核心清单缺少的 7 篇论文。
- 从 arXiv 补充并用 MinerU 解析来源索引剩余的 3 篇缺失论文。
- 建立 180 篇全量 `inventory.csv` 和 SHA-256 去重/完整性报告；全部论文可读且无重复。
- 建立替换后的核心 15 清单、15 份 Paper Card、18 维机制矩阵、16 类失败模式和覆盖审计。
- 核心审计结论为 `PASS_WITH_REPLACEMENTS`，加权覆盖分 89/100；在 G2 人工批准前停止进入算法与训练阶段。

### v0.2.0 — 2026-08-01

- 新增《大模型强化学习算法发现系统：施工方案与验收规范》v1.0。
- 明确研究流程、阶段门禁、实验预算、审计要求与第一阶段交付清单。

### v0.1.0 — 2026-08-01

- 初始化项目仓库。
- 纳入大模型后训练与对齐算法参考资料。
- 建立 README 版本记录与持续推送约定。
