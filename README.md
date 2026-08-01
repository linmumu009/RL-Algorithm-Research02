# RL-Algorithm-Research02

强化学习算法研究项目，聚焦大模型后训练、对齐与相关算法资料的整理和研究。

## 当前状态

- 当前阶段：`P4_PROBLEM_SELECTION`
- 当前验收门：`G4`
- 状态：P3 知识地图已通过 `G3`，等待人工确认真实研究问题后才能生成算法假设。
- 本地语料：200 篇唯一 arXiv 论文，全部可读、无重复、官方元数据完整。
- 核心基底：15 篇，均已关联 PDF、MinerU Markdown、Paper Card 和机制矩阵。
- P3 结构化成果：15 张 Mechanism Card、20 条定向 Claim Card、13 组机制冲突和 10 个低成本解释区分实验。
- G4 首选问题：`Q-001 — 适应性、实例依赖的 verifier 噪声`。

## 关键入口

- 项目章程：[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md)
- 当前状态：[`research_state.yaml`](research_state.yaml)
- 施工方案：[`LLM_RL_Algorithm_Discovery_Construction_Plan.md`](LLM_RL_Algorithm_Discovery_Construction_Plan.md)
- 语料清单：[`01_corpus/inventory.csv`](01_corpus/inventory.csv)
- 核心审计：[`10_deliverables/core15_audit.md`](10_deliverables/core15_audit.md)
- 替换建议：[`02_literature/selection_audit/replacement_recommendations.md`](02_literature/selection_audit/replacement_recommendations.md)
- 机制矩阵：[`03_taxonomy/mechanism_matrix.csv`](03_taxonomy/mechanism_matrix.csv)
- P3 知识地图：[`10_deliverables/knowledge_map.md`](10_deliverables/knowledge_map.md)
- 方法谱系：[`03_taxonomy/method_lineage.md`](03_taxonomy/method_lineage.md)
- 候选研究问题：[`04_problems/candidate_problem_statements.md`](04_problems/candidate_problem_statements.md)
- 低成本区分实验：[`04_problems/baseline_diagnostics/p3_low_cost_discriminators.md`](04_problems/baseline_diagnostics/p3_low_cost_discriminators.md)

## 目录

项目按施工方案分为治理、语料、文献、分类体系、问题、假设、实验、结果、评审、决策和交付物目录。当前已完成 P0–P3，并形成 P4 候选问题；尚未生成正式算法候选或训练实现。

## 更新约定

每次项目更新均同步维护本 README 的版本说明，并在完成验证后提交、推送至远程仓库的 `main` 分支。

## 版本记录

### v0.4.0 — 2026-08-01

- 用户批准替换后的 Core 15，冻结 `core15_v2` 并通过 G2 进入 P3。
- 针对 verifier 噪声/奖励投机、长轨迹复用、异步训练及延迟—吞吐三个薄弱区，定向补充 20 篇论文。
- 20 篇新增 PDF 全部由 MinerU 解析为 Markdown；语料达到 200 篇唯一 arXiv 论文，全部可读、无重复且官方元数据完整。
- 建立 15 张 Mechanism Card、20 条新增 Claim Card、方法谱系、问题—机制图和 13 组机制冲突。
- 建立 10 个无需立即大规模训练的低成本解释区分实验，并列出九类真实空白与伪空白。
- 完成 `knowledge_map.md` 并通过 G3；知识地图能够回答施工方案规定的全部五项验收问题。
- 形成 5 个 P4 候选研究问题，首选 `Q-001 — 适应性、实例依赖的 verifier 噪声`；按施工方案暂停在 G4 人工必审点。
- 新增 P3 自动验证器；P0–P2 与 P3 验证均无错误。

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
