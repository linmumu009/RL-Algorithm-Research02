# RL-Algorithm-Research02

强化学习算法研究项目，聚焦大模型后训练、对齐与相关算法资料的整理和研究。

## 当前状态

- 当前阶段：`P7_LOW_COST_DECISIVE_EXPERIMENTS`
- 当前验收门：`G6`
- 状态：H-027 二维 E0 已绑定不可变代码提交、尚未正式运行；活跃组合为 4 条。
- 本地语料：219 篇唯一 arXiv 论文，全部可读、无重复、官方元数据完整。
- 核心基底：15 篇，均已关联 PDF、MinerU Markdown、Paper Card 和机制矩阵。
- P3 结构化成果：15 张 Mechanism Card、20 条定向 Claim Card、13 组机制冲突和 10 个低成本解释区分实验。
- 正式研究问题：`Q-001 — 适应性、实例依赖的 verifier 噪声`。
- 活跃分支：H-001、H-005、H-014（均为 `E0_VALIDATED`）与 H-027（`BOUND_NOT_RUN`）。
- 当前边界：只允许从绑定提交一次性执行 H-027 的 1 单位二维解析/合成 E0；G6 前不进行 E1、语言模型训练或扩大验证。

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
- 正式问题：[`04_problems/selected_problem.md`](04_problems/selected_problem.md)
- 假设筛选报告：[`10_deliverables/hypothesis_screening.md`](10_deliverables/hypothesis_screening.md)
- 候选谱系：[`05_hypotheses/lineage_graph.json`](05_hypotheses/lineage_graph.json)
- E0 预注册：[`06_experiments/preregistrations/`](06_experiments/preregistrations/)
- E0 冻结配置：[`06_experiments/configs/e0_suite.yaml`](06_experiments/configs/e0_suite.yaml)
- E0 实现：[`06_experiments/code/e0_suite.py`](06_experiments/code/e0_suite.py)
- E0 数学一致性测试：[`06_experiments/unit_tests/test_e0_suite.py`](06_experiments/unit_tests/test_e0_suite.py)
- E0 原始结果：[`07_results/raw/e0_suite_results.json`](07_results/raw/e0_suite_results.json)
- E0 结果卡：[`07_results/result_cards/`](07_results/result_cards/)
- E0 实验报告：[`10_deliverables/e0_experimental_report.md`](10_deliverables/e0_experimental_report.md)
- E0 本地审查：[`08_reviews/local_reviews/e0_review.md`](08_reviews/local_reviews/e0_review.md)
- 替代分支筛选：[`10_deliverables/replacement_hypothesis_screening.md`](10_deliverables/replacement_hypothesis_screening.md)
- H-021 预注册：[`06_experiments/preregistrations/E0-H021.yaml`](06_experiments/preregistrations/E0-H021.yaml)
- H-021 冻结配置：[`06_experiments/configs/e0_h021.yaml`](06_experiments/configs/e0_h021.yaml)
- H-021 冻结实现：[`06_experiments/code/h021_bridge_e0.py`](06_experiments/code/h021_bridge_e0.py)
- H-021 数学测试：[`06_experiments/unit_tests/test_h021_bridge_e0.py`](06_experiments/unit_tests/test_h021_bridge_e0.py)
- H-021 原始结果：[`07_results/raw/e0_h021_results.json`](07_results/raw/e0_h021_results.json)
- H-021 结果卡：[`07_results/result_cards/R-E0-H021.yaml`](07_results/result_cards/R-E0-H021.yaml)
- H-021 实验报告：[`10_deliverables/h021_e0_experimental_report.md`](10_deliverables/h021_e0_experimental_report.md)
- H-021 本地审查：[`08_reviews/local_reviews/h021_e0_review.md`](08_reviews/local_reviews/h021_e0_review.md)
- P7 补证清单：[`02_literature/extended/p7_regeneration_supplement.csv`](02_literature/extended/p7_regeneration_supplement.csv)
- 第二轮替代筛选：[`10_deliverables/replacement_hypothesis_screening_round_2.md`](10_deliverables/replacement_hypothesis_screening_round_2.md)
- 第二轮补证清单：[`02_literature/extended/p7_regeneration_round_2_supplement.csv`](02_literature/extended/p7_regeneration_round_2_supplement.csv)
- H-027 预注册：[`06_experiments/preregistrations/E0-H027.yaml`](06_experiments/preregistrations/E0-H027.yaml)
- H-027 冻结配置：[`06_experiments/configs/e0_h027.yaml`](06_experiments/configs/e0_h027.yaml)
- H-027 冻结实现：[`06_experiments/code/h027_gradient_set_e0.py`](06_experiments/code/h027_gradient_set_e0.py)
- H-027 数学测试：[`06_experiments/unit_tests/test_h027_gradient_set_e0.py`](06_experiments/unit_tests/test_h027_gradient_set_e0.py)

## 目录

项目按施工方案分为治理、语料、文献、分类体系、问题、假设、实验、结果、评审、决策和交付物目录。当前已完成 P0–P6、首轮 P7/E0 和定向组合修复；尚未进行语言模型训练。

## 更新约定

每次项目更新均同步维护本 README 的版本说明，并在完成验证后提交、推送至远程仓库的 `main` 分支。

## 版本记录

### v0.9.2 — 2026-08-02

- 将 H-027 预注册绑定到已推送的不可变冻结提交 `b2778d683b22d8f7a24f60e3d3443abb2671aed2`。
- 绑定时 `e0_h027_results.json` 与 `e0_h027_summary.csv` 均不存在，确保实现、网格、基线、控制和判定阈值不能根据正式结果回改。
- 保持 5 个固定种子、400 个 valid-coverage 行、30 个显式控制行和 80-cell 汇总；不允许删除 misspecified-interval 或 wide-interval 压力条件。
- 下一步唯一允许动作是从该绑定实现一次性执行完整 H-027 E0，并按原预注册门槛作出保留或淘汰决定；本轮预算仍为 0。

### v0.9.1 — 2026-08-02

- 冻结 H-027 二维 audit-identified gradient-set E0：由 FP/FN 审计区间精确枚举四个可行 clean-gradient vertex，并以凸包到原点的最小范数点构造 maximin 分离方向。
- 固定 5 个种子、5 档区间半宽、4 档通道非对称和 4 档 score-gradient 夹角，共 400 个 valid-coverage 运行行与 80 个汇总 cell。
- 冻结 H-001 midpoint correction、H-018 scalar lower bound、KL-DRO scalar pessimism、norm-matched shrinkage 和 oracle 比较，并保证所有非 oracle 方法使用相同审计集合。
- 显式保留 point-identification、symmetric-zero-bias、zero-in-set、misspecified-interval、wide-interval 和 equal-compute 六类控制，不允许结果后删除 abstention 或覆盖失效反例。
- 新增 6 项凸几何与配置一致性测试；连同既有测试共 16 项全部通过，正式 H-027 结果文件仍不存在，本轮预算消耗为 0。
- 下一步先提交并推送不可变实现，将精确提交哈希绑定到预注册，再仅执行一次完整网格。

### v0.9.0 — 2026-08-02

- 在 H-021 失败后完成第二轮定向再生成，生成并筛查 H-027–H-032 六个候选；未把 H-021、H-018、H-009 或 H-020 换名复活。
- 从官方 arXiv 补充 7 篇最新直接近邻并由 MinerU 全部解析，覆盖 partial identification、distributional reward uncertainty、anytime-valid inference、ensemble shared errors、commit-first judging 和 hacking-onset detection；语料达到 219 篇。
- 淘汰 H-028：distributional entropic/KL-DRO reward 已直接覆盖；淘汰 H-029：anytime-valid confidence sequence 已有且 release gate 非独立；淘汰 H-030：ensemble 已有且共同错误不消失。
- 淘汰 H-031：commit-first blind solver 已直接作为训练 reward 提出；淘汰 H-032：CHERRL detector 加 rollback 只是 schedule control。
- 保留 H-027 `Audit-Identified Gradient Set Direction`：将 audit-compatible verifier channel 映射为 clean-gradient identified set，只在集合与零可分时选择对全部可行梯度均有正最坏内积的方向。
- 完成 H-027 的 1 单位 E0 预注册，锁定 point correction、scalar lower bound、KL-DRO scalar pessimism、norm-matched shrinkage 和 oracle 强基线，以及 point-limit、zero-in-set、wide interval 和 interval misspecification 控制。
- 补记 H-021 E0 的 1 单位并登记本轮 4 单位筛选成本，累计预算 60/100；活跃组合恢复为 H-001、H-005、H-014、H-027，G6 仍未通过。

### v0.8.0 — 2026-08-02

- 按绑定提交 `8b167359c3c114412af397ab69d30875d3fa1bdf` 唯一一次执行 H-021 E0，完整保留 90 个有效运行行、5 个 invalid-exclusion 行和 18-cell 汇总；未重跑、未调网格。
- H-021 在全部强代理 cell 中 bias 为 0.005995–0.026519，低于 0.05；与 revealed-latent oracle 的最大 cosine 差为 0.014793，控制项通过。
- 弱代理条件数中位数由强区间的 3.20 恶化至 678.12，确认 completeness/rank 风险，但这不构成主效应成功。
- 核心增益预测失败：6 个强代理 latent-exploit cell 中 0 个达到 0.05，平均 cosine gain 为 -0.004589；direct proxy regression 在 30 次强条件比较中赢 28 次。
- H-021 因“可识别但无增量价值”记为 `REJECTED_FAILED_PREDICTION_AND_DOMINATED`，不做结果后局部修补。
- 本轮消耗 1 单位，累计 56/100；活跃分支回到 H-001、H-005、H-014 三条，G6 未通过，下一步只能开展第二轮机制不同的定向再生成。

### v0.7.2 — 2026-08-02

- 将 H-021 预注册绑定到已推送的不可变冻结提交 `8b167359c3c114412af397ab69d30875d3fa1bdf`。
- 绑定时 H-021 正式原始结果与汇总表均不存在，避免依据结果回改实现、网格或阈值。
- 保持全部 5 个种子、18 个有效网格 cell 和 invalid-exclusion 压力控制；弱代理与负结果不得删除。
- 下一步唯一允许动作是从绑定实现一次性执行完整 H-021 E0，并按原预注册阈值审查；本轮预算仍为 0。

### v0.7.1 — 2026-08-02

- 冻结 H-021 离散负控桥接 E0：在 `(context, observed reward)` 分层内，用 sparse clean audit 求解 `E[Y-h(W,R,X)|Z,R,X]=0` 的 2×2 bridge moment。
- 固定 5 个随机种子、6 档代理相关性、3 档 latent exploit 强度和 6000/60000 的审计比例；0.70 及以上为强代理，0.55/0.52/0.50 为弱秩压力。
- 同时冻结 H-001 observed-stratum correction、H-005 nuisance projection、direct proxy regression、direct pair average 与 revealed-latent oracle，避免弱基线制造虚假增益。
- 加入 no-latent-exploit、invalid exclusion 与 rank collapse 控制；奇异 moment 仅按预声明 ridge 规则处理，不允许结果后调参。
- H-021 新增 4 项数学测试，连同原 E0 测试共 10 项全部通过；正式 H-021 网格尚未运行，本轮预算消耗为 0。
- 下一步先提交并推送这份不可变实现，再把精确提交哈希写入预注册，随后只执行一次完整网格。

### v0.7.0 — 2026-08-02

- 针对 E0 后活跃分支不足，生成并筛查 H-021–H-026 六个替代候选；未复活 H-004、H-008 或 H-018 的原论点。
- 从 arXiv 补充 6 篇直接近邻并用 MinerU 全部解析，覆盖 corrupted reward、negative controls、proximal RL、OCRM、gradient regularization 和 rubric reward hacking；语料达到 212 篇。
- 淘汰 H-022：OCRM/importance transport 已覆盖；淘汰 H-023：2026 年 gradient regularization 直接覆盖；淘汰 H-024：诊断加 gate 无 clean-gradient 机制；淘汰 H-025：属于 hybrid verification 或 H-021 proxy 步骤；淘汰 H-026：quantilisation 已有。
- 保留 H-021 `Negative-Control Verifier Bridge`：用 randomized verifier proxy、exploit-sensitive diagnostic 与 sparse clean audit 识别 latent-exploit clean-gradient bridge。
- 完成 H-021 的 1 单位 E0 预注册，锁定 strong/weak proxy、invalid exclusion、revealed latent 和非对称强基线控制。
- 活跃组合恢复为 H-001、H-005、H-014、H-021；本轮消耗 4 单位，累计 55/100，G6 仍未通过。

### v0.6.0 — 2026-08-02

- 按已推送的冻结实现提交 `34fea81` 和统一预注册，一次性完成 `Q001-E0-v1`；五个种子全部保留，未调网格、未换指标、未训练语言模型。
- H-001 通过：4/4 个异质噪声 cell 的 clean-gradient cosine 增益达到阈值且置信区间下界大于 0，同通道控制增益为 0。
- H-005 通过：非因果场景的伪梯度质量平均减少 99.68%，clean-gradient cosine 未下降；同时记录 causal stress 的信号保留风险。
- H-014 通过：四个审计预算均比最佳强基线降低 MSE 10.75%，最小 ESS 比为 58.59%，等范数控制精确退化为随机抽样。
- H-004 淘汰：增强式代数上精确等于 channel-only correction；在仅 regression 正确时 bias=0.191，超过 0.05 失败阈值。
- H-008 淘汰：主预测虽通过，但相对 direct pair average 的 cosine 优势为 -0.00016，判定为行为等价而非新机制。
- H-018 淘汰：可交换数据下覆盖成立，但下移漂移 undercoverage 为 14.32%–99.21%，触发预注册失败阈值。
- 本轮消耗 6 单位，累计 51/100；G6 仍未通过。活跃组合剩 3 条，下一步必须先补足至少一个非等价分支，再进入 E1。

### v0.5.2 — 2026-08-02

- 将 6 份 E0 预注册全部绑定到已推送的不可变实现提交 `34fea81eb28bdba546580ba91e68d1cca5065805`。
- 绑定时正式结果仍未生成，确保实验指标、网格、随机种子、控制组和失败阈值无法根据结果事后改动。
- 下一项唯一允许任务为按该提交与 `Q001-E0-v1` 冻结配置一次性执行，并完整保留正、负及等价性结果。

### v0.5.1 — 2026-08-02

- 冻结 `Q001-E0-v1`：6 个分支共享五个固定随机种子，并分别锁定异质性、模型正确性、因果/伪相关、语义不变性、审计预算和覆盖率压力网格。
- 实现统一的解析/合成实验程序，只计算 clean-gradient bias、variance、MSE、cosine、ESS、false-positive update 等预注册指标，不进行语言模型训练。
- 新增 6 项数学一致性测试，验证二值噪声反演、奇异通道拒绝、等杠杆退化、区间收缩和分支完整性。
- 在正式运行前显式保留两个危险反例：H-004 的“增强项”可能代数退化为 channel-only correction；H-018 的有限样本覆盖可能在分布漂移下失效。
- 正式 E0 结果尚未生成；先提交不可变实现快照，再将提交哈希写入预注册后一次性执行。

### v0.5.0 — 2026-08-01

- 用户确认 `Q-001 — 适应性、实例依赖的 verifier 噪声`，通过 G4 并冻结正式问题定义。
- 为新颖性审查补充 6 篇最接近论文及 MinerU Markdown，语料达到 206 篇，完整性与去重检查全部通过。
- 生成 20 个满足因果链模板的初始候选，每个均登记改变的数学量、行为预测、最低成本否定实验、失败阈值和预算。
- 完成语义近似、数学等价、行为等价和理论风险审查，淘汰 14 个已有、等价、调度型或机制不成立的候选。
- 保留 6 个候选：H-001、H-004、H-005、H-008、H-014、H-018，来自 6 个不同机制家族，评分 85–95。
- 建立完整 `lineage_graph.json`，记录父候选、修改、淘汰原因和重命名判定，防止失败想法循环出现。
- 为 6 个保留分支分别完成 E0 预注册；在 E0 审查前不允许进行语言模型训练。
- P5/P6 验收结论为 `G5 PASS`；新增自动验证器检查候选数量、状态、机制多样性、谱系和预注册一致性。

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
