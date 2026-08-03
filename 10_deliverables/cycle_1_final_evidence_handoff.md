# RL Algorithm Discovery 第一周期最终证据交接

## 交接结论

第一算法发现周期已经完整归档，状态为 `P7_CLOSED_GLOBAL_FALLBACK / G6_NOT_PASSED`。本周期没有候选获得 E1、E2、语言模型训练或扩大验证权限；H-001、H-005、H-014 保留 E0 正面证据并暂停，其他 41 条谱系节点均有淘汰或拒绝记录。累计预算 71/100，剩余 29 单位作为未探索分支/全局重启储备保留。

本交接包是当前仓库的权威入口。它不宣称 Q-001 已解决，也不把 E0 合成结果外推为真实大模型训练收益。接手者应先阅读本文件、全局回退治理复核、研究状态和决策日志，再决定是否提出一个新的 G0/G4 重启章程。

## 一页状态

| 项目 | 最终值 |
|---|---:|
| 唯一 arXiv 论文 | 237 |
| 核心机制论文 | 15 |
| 假设谱系节点 | 44 |
| 正式 E0 分支 | 10 |
| E0 验证后暂停 | 3 |
| 拒绝/淘汰节点 | 41 |
| E1/E2/模型训练 | 0 |
| 已用预算 | 71/100 |
| 剩余重启储备 | 29 |
| 最终验收门 | G6_NOT_PASSED |
| 最终决策 | D-0027 / COMPLETE_CYCLE_1_EVIDENCE_HANDOFF |

## 可复用的正面证据

### H-001 — Contextual Confusion Correction

- 四个异质通道 cell 的 clean-gradient cosine gain 均达到预注册门槛，且随异质性由 0.0609 增至 0.7151。
- 同通道控制 gain 为 0，支持“只在 contextual channel 不同的时候需要条件修正”。
- 可复用范围：解析/合成 verifier channel correction。
- 不可外推：真实审计估计误差、policy distribution shift、端到端训练稳定性。

### H-005 — Nuisance-Orthogonal Reward

- 非因果 nuisance 条件下伪梯度质量减少 99.68%。
- causal stress 只保留 73.36% 信号，明确暴露误删真实任务信号的风险。
- 可复用范围：nuisance projection 的合成机制与保护性控制。
- 不可外推：真实 nuisance 定义、因果充分性和语言模型表征空间。

### H-014 — Clean-Gradient Leverage Auditing

- 在四档预算中相对最佳基线的 MSE 改善为 10.75%，最小 ESS ratio 为 0.5859。
- equal-norm control 差异为 0，符合 leverage 机制退化预测。
- 可复用范围：固定 score vectors 下的有限总体 audit acquisition。
- 不可外推：真实 rollout acquisition、非平稳 policy 和训练收益。

三条分支当前状态均为 `E0_VALIDATED_PAUSED_GLOBAL_FALLBACK`。暂停不是淘汰；但任何后续实验都必须进入新的治理周期。

## 十条正式 E0 证据总表

| 分支 | 正式结论 | 决策 | 最关键证据 |
|---|---|---|---|
| H-001 | PASS | E0 验证后暂停 | 异质通道 gain 0.0609–0.7151，同通道退化为 0 |
| H-004 | FAIL | 失败预测淘汰 | doubly robust augmentation 代数上等于 channel-only；bias 0.191 |
| H-005 | PASS | E0 验证后暂停 | 伪梯度质量减少 99.68%，causal stress 暴露风险 |
| H-008 | 主预测 PASS / 等价 FAIL | 等价淘汰 | 相对 direct pair average 优势 -0.00016 |
| H-014 | PASS | E0 验证后暂停 | MSE 相对最佳基线改善 10.75% |
| H-018 | FAIL | 失败预测淘汰 | shift undercoverage 0.1432–0.9921 |
| H-021 | FAIL | 无增量且被支配 | 0 个 gain cell；direct proxy regression 赢 28/30 |
| H-027 | FAIL | 无增量且被支配 | 安全性通过；最大 gain 0.011294，目标 gain cell 为 0 |
| H-033 | FAIL | 失败预测且被支配 | 118/144 bias 超标，增量 cell 为 0 |
| H-039 | FAIL | 无增量且被支配 | 错误符号率为 0，但 96 个目标 cell 中 gain cell 为 0 |

## 正式结果与完整性哈希

| 结果集合 | 覆盖分支 | LF 规范 SHA-256 |
|---|---|---|
| `e0_suite_results.json` | H-001/H-004/H-005/H-008/H-014/H-018 | `6bcc39b76033f5639a5b37311f5c21e92154622e6495cada399655b40752b688` |
| `e0_h021_results.json` | H-021 | `cba8540ab7570e874e99b9adb50ddc6b85a47ae78e5b0d217d6724014105896b` |
| `e0_h027_results.json` | H-027 | `8bbf6df31fbb17a40e3bdd723dd9edc70343da074ffe78b6f194ede1ef0190ac` |
| `e0_h033_results.json` | H-033 | `f07f69762e7e2b51d288feb1a76aebe9e7adcf4c8bf88fd626f82d70bef5f695` |
| `e0_h039_results.json` | H-039 | `9df09d3e5f5a837b40dbcb5af1b2e89f131806258932899b2a2490cffa5792bf` |

更完整的关键文件列表、文件大小和 LF 规范哈希见 `10_deliverables/cycle_1_evidence_manifest.csv`。所有正式结果均有专用 validator；禁止通过再次执行正式命令来“验证”结果。

## 推荐阅读顺序

1. `README.md`：当前状态与版本轨迹。
2. `PROJECT_CHARTER.md` 和 `research_state.yaml`：权限、预算和关闭边界。
3. `10_deliverables/p7_global_fallback_governance_review.md`：为什么关闭而不挪用储备。
4. `04_problems/selected_problem.md`：Q-001 的原始问题定义。
5. `10_deliverables/knowledge_map.md`：机制—失败模式地图。
6. `05_hypotheses/lineage_graph.json` 与 `09_decisions/killed_ideas.md`：44 条谱系和禁止换名复活边界。
7. 各 E0 报告、结果卡和本地审查：正负结果详情。
8. `09_decisions/decision_log.md` 与 `09_decisions/budget_ledger.csv`：完整治理和预算轨迹。

## 验证方法

允许执行的是只读 validator 与单元测试：

```powershell
$validators = Get-ChildItem tools/corpus_indexer -Filter 'validate*.py' | Sort-Object Name
foreach ($validator in $validators) { python $validator.FullName }
python -m pytest 06_experiments/unit_tests -q
```

不得重新运行 `e0_suite.py`、`h021_bridge_e0.py`、`h027_gradient_set_e0.py`、`h033_private_audit_e0.py` 或 `h039_sign_certificate_e0.py` 的正式入口。验证器使用固定哈希、结构、阈值和结果卡检查已有证据，不需要重新采样。

## 已知边界

- 语料虽达到 237 篇，但核心结构化机制卡仍以 15 篇为主；扩展论文主要用于新颖性否定和直接近邻审查。
- 全部正式实验均为解析、合成或玩具级 E0；没有真实模型训练证据。
- H-001/H-005/H-014 的正面结果不构成论文级算法有效性结论。
- 四次替代分支失败说明当前搜索局部已饱和，不证明其他问题或全新信息结构不可能成功。
- 剩余 29 单位受到全局重启用途约束，不能自动用于当前 Q-001 的继续修补。

## 重启清单

重启前必须全部满足：

- 新的或明确修订的 G0/G4 章程；
- 独立新增预算，或人工批准并记录储备重分配；
- 至少 4 条非等价并行分支；
- 对当前 44 条谱系和 Killed Ideas 做新颖性反查；
- 新的失败阈值、预算和预注册；
- 明确说明新信息结构为什么不是 H-001/H-005/H-014/H-021/H-027/H-033/H-039 的组合、阈值或换名。

在此之前，仓库只接受维护、勘误和证据归档更新。
