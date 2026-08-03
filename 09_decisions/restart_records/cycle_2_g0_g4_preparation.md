# 第二发现周期 G0/G4 准备记录

- 记录日期：2026-08-03
- 记录状态：`PREPARED_NOT_APPROVED`
- 对应决定：D-0029
- 研究执行授权：`false`
- 第二周期选定问题：`null`
- 第二周期预算：`not_requested / not_approved`
- 第一周期剩余储备：29，继续锁定

## 准备内容

- 从现有 237 篇语料、Core 15 覆盖缺口、P3 知识地图和第一周期 P9 终评中整理五个候选问题方向；
- 重新按施工方案 G4 审计可复现性，不沿用第一周期未经用户真实日志确认的高分；
- 明确所有候选当前可复现性为 0/20、正式 G4 未通过；
- 起草第二周期使命、非目标、预算原则、停止规则和人工决策选项；
- 明确排除 Q-001 第五轮修补、暂停分支组合和 41 条拒绝谱系的换名复活；
- 定义最小真实失败事实包与问题通过后才允许开展的定向文献刷新。

## 当前判定

`G0_G4_DECISION_PACKET_READY / HUMAN_REVIEW_REQUIRED`

该状态只表示决策材料齐全，不表示第二周期已启动。用户未批准 G0、未提供或确认真实失败证据、未批准预算之前：

- 不生成 P5 算法候选；
- 不进行 P6 新颖性结论；
- 不预注册或运行 E0/E1/E2；
- 不下载无目标扩展文献；
- 不挪用第一周期 29 单位储备。

## 入口

- 决策包：`10_deliverables/cycle_2_g0_g4_decision_packet.md`
- 候选问题：`04_problems/cycle_2_candidate_problem_statements.md`
- 第一周期终评：`10_deliverables/final_research_assessment.md`
- 第一周期禁区：`09_decisions/killed_ideas.md`

## 下一项允许任务

接收用户的 G0 决定和真实失败事实包，进行只读/零成本证据核查并重新计算 G4 正式评分。若没有真实失败证据，则保持 `DEFER_G4_NO_OBSERVED_PROBLEM`。
