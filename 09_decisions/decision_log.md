# Decision Log

## D-0001 — 2026-08-01

- 阶段：P0/P1
- 动作：`RESTRUCTURE_AND_AUDIT`
- 决定：按施工方案重构目录；将新增诊断与理论论文纳入核心候选；不进入算法与训练阶段。
- 依据：原核心集合偏重算法与偏好优化，诊断、负结果和理论边界覆盖不足。
- 下一项允许任务：完成语料清点与核心 15 覆盖审计。

## D-0002 — 2026-08-01

- 阶段：P2 / G2
- 动作：`PAUSE_AT_G2`
- 决定：核心集合以 `PASS_WITH_REPLACEMENTS` 通过机器审计，但在人工批准前停止。
- 依据：18/18 设计维度、16 个失败模式、89/100 加权覆盖分；仍存在 verifier 噪声、长轨迹复用和延迟证据薄弱区。
- 下一项允许任务：修正文献证据，或在人工批准后进入 P3。

## D-0003 — 2026-08-01

- 阶段：G2 → P3
- 动作：`APPROVE_G2_ENTER_P3`
- 决定：用户明确批准开始下一阶段；冻结 `core15_v2` 为机制基底，进入机制—问题—证据知识地图建设。
- 依据：Core 15 覆盖分 89/100，18/18 设计维度和 16 个失败模式均达到 G2 验收要求。
- 边界：P3 只建设知识地图和补证据，不生成正式算法、不编写训练代码；通过 G3 后再进入真实问题选择。
- 下一项允许任务：完成三张核心图、机制卡、证据冲突表和可低成本辨别的开放问题。

## D-0004 — 2026-08-01

- 阶段：P3 / G3 → P4 / G4
- 动作：`PASS_G3_ENTER_P4`
- 决定：P3 知识地图通过 G3；进入问题选择，暂停在 G4 人工必审点。
- 依据：已形成方法谱系、问题—机制图、13 组机制冲突、15 张机制卡、20 条定向证据和 10 个低成本解释区分实验；能够回答 G3 的全部五问。
- 推荐问题：`Q-001 — 适应性、实例依赖的 verifier 噪声`，但在人工确认前不登记为 selected problem。
- 边界：不生成正式算法，不编写训练代码。

## D-0005 — 2026-08-01

- 阶段：P4 / G4 → P5/P6
- 动作：`APPROVE_G4_SELECT_Q001`
- 决定：用户按推荐确认 `Q-001 — 适应性、实例依赖的 verifier 噪声` 为正式研究问题。
- 依据：该问题得分 96/100，具有直接因果证据、明确现有缺口、低成本可证伪性和较高风险价值。
- 边界：允许生成和筛选至少 20 个假设；G5 前不实现训练代码，P7 预注册前不执行实验。
- 下一项允许任务：完成多分支生成、新颖性/等价性/理论初筛，并只保留 4–6 个候选。

## D-0006 — 2026-08-01

- 阶段：P5/P6 / G5 → P7 / G6
- 动作：`PASS_G5_RETAIN_SIX`
- 决定：20 个候选中保留 6 个进入低成本实验，淘汰 14 个。
- 保留：H-001、H-004、H-005、H-008、H-014、H-018。
- 依据：6 个候选来自 6 个机制家族，总分 85–95，可证伪性 14–15，与已有方法差异 11–13；均有明确失败阈值和 E0 预注册。
- 淘汰：等价/重命名 6 个，已有方法覆盖 4 个，无独立机制或理论不一致 4 个。
- 边界：只允许执行预注册 E0 数学/合成实验；E0 评审前不进行语言模型训练。

## D-0007 — 2026-08-02

- 阶段：P7 / G6
- 动作：`FREEZE_E0_IMPLEMENTATION`
- 决定：冻结 6 个分支共享的 E0 数学/合成实验实现、配置、随机种子、控制组和输出格式；正式结果尚未生成。
- 依据：6 项数学一致性单元测试全部通过；实现明确保留 H-004 的代数退化反例和 H-018 的分布漂移压力条件。
- 预算影响：0（尚未执行预注册实验）。
- 反事实判断：若尚未投入时间，仍会选择这套实现，因为它对 6 个分支使用同一审计框架并能直接触发原失败阈值。
- 边界：先提交不可变代码快照，再把提交哈希写入预注册；之后只允许按冻结配置执行一次，不允许看结果后调网格。
- 下一项允许任务：登记代码提交哈希并执行 `Q001-E0-v1`。

## D-0008 — 2026-08-02

- 阶段：P7 / G6
- 动作：`BIND_E0_CODE_SNAPSHOT`
- 决定：6 份 E0 预注册统一绑定不可变实现提交 `34fea81eb28bdba546580ba91e68d1cca5065805`。
- 依据：该提交已推送至 `origin/main`，包含冻结配置、实现和 6 项通过的数学一致性测试，且未生成正式实验结果。
- 预算影响：0（仍未执行实验）。
- 反事实判断：若尚未投入时间，仍会先做代码绑定，因为它把“预注册内容”和“实际执行代码”连接成可审计证据链。
- 边界：不得修改提交 `34fea81` 的实验网格、种子、指标或阈值；运行失败只能原样记录，不能静默重跑。
- 下一项允许任务：一次性执行 `Q001-E0-v1`，保存所有正负结果。

## D-0009 — 2026-08-02

- 阶段：P7 / G6
- 动作：`E0_REVIEW_RETAIN_THREE`
- 决定：保留 H-001、H-005、H-014 为 `E0_VALIDATED`；H-004、H-018 按失败预测淘汰；H-008 按行为等价淘汰。
- 依据：六份结果卡、原始 JSON、汇总表和 `10_deliverables/e0_experimental_report.md`。
- 预算影响：6 单位，累计 51/100；每个分支仅使用预注册的 1 单位。
- 反事实判断：若尚未投入时间，仍会选择三条保留路线；不会再选择 H-004 的退化估计式、H-008 的等价形式或 H-018 的漂移覆盖主张。
- G6：`NOT_YET_PASS`。三条分支仅通过 E0，尚无 E1/E2 证据，也未获得扩大实验权限。
- 组合约束：活跃分支降至 3，低于并行下限 4；触发定向再生成，不得用已淘汰候选改名补数。
- 下一项允许任务：生成并按 P5/P6 全流程筛选至少一个机制不同的替代分支，恢复四分支组合后再预注册 E1。

## D-0010 — 2026-08-02

- 阶段：P7 / G6（定向组合修复）
- 动作：`PASS_TARGETED_REGENERATION_RETAIN_H021`
- 决定：6 个替代候选中仅保留 H-021 `Negative-Control Verifier Bridge`，其余 5 个按直接已有、无机制或非独立分支淘汰；活跃组合恢复为 4 条。
- 依据：语料扩充至 212 篇；`replacement_hypothesis_screening.md`、再生成新颖性/等价性/理论风险审查以及 H-021 E0 预注册。
- 预算影响：4 单位，累计 55/100；仍保留 45 单位，满足至少 30% 探索/重启保留要求。
- 反事实判断：若尚未投入时间，仍只选择 H-021，因为其 latent-exploit bridge estimand 与三条存活分支不同，且可用 1 单位实验直接证伪；不会选择已有 OCRM、gradient regularization 或 quantilisation 的换名。
- 边界：H-021 只获得解析/合成 E0 权限；不得开始 E1 或语言模型训练。
- 下一项允许任务：实现并冻结 H-021 离散 negative-control bridge E0，提交代码快照、绑定预注册后一次性执行。

## D-0011 — 2026-08-02

- 阶段：P7 / G6
- 动作：`FREEZE_H021_E0_IMPLEMENTATION`
- 决定：冻结 H-021 的离散负控桥接实现、配置、五个随机种子、强弱代理网格、排除限制失效压力、revealed-latent oracle 和四个非 oracle 强基线；正式结果尚未生成。
- 依据：新增 4 项 H-021 数学一致性测试，与原 6 项测试合计 10 项全部通过；0.50 代理相关性明确触发秩崩塌并由预声明 ridge 规则保持数值可记录。
- 预算影响：0（尚未执行预注册网格）。
- 反事实判断：若尚未投入时间，仍会选择该实现，因为它直接检验 bridge identification、与 direct proxy regression 的增益及 completeness/rank 失败，而不是只展示合成正例。
- 边界：必须先提交并推送不可变实现，再绑定该提交哈希；不得在看到正式结果后更改强代理定义、网格、种子、指标或阈值。
- 下一项允许任务：提交并推送冻结实现，将精确提交哈希写入 `E0-H021.yaml`，然后一次性执行完整网格。

## D-0012 — 2026-08-02

- 阶段：P7 / G6
- 动作：`BIND_H021_E0_CODE_SNAPSHOT`
- 决定：H-021 预注册绑定到已推送的不可变实现提交 `8b167359c3c114412af397ab69d30875d3fa1bdf`。
- 依据：该提交包含冻结实现、配置、控制组、输出格式及 10 项通过的数学一致性测试；绑定时 `07_results/` 中不存在 H-021 正式结果。
- 预算影响：0（仍未执行实验）。
- 反事实判断：若尚未投入时间，仍会先绑定提交，因为它使预注册、执行代码与结果之间形成不可回写的证据链。
- 边界：完整网格只执行一次；不得删除弱代理、invalid-exclusion 或负结果，不得改动成功/失败阈值后重跑。
- 下一项允许任务：从绑定提交一次性执行 `E0-H021-NEGATIVE-CONTROL-BRIDGE`，保存全部原始行并按原阈值审查。

## D-0013 — 2026-08-02

- 阶段：P7 / G6
- 动作：`REJECT_H021_NO_INCREMENTAL_GAIN`
- 决定：H-021 记为 `REJECTED_FAILED_PREDICTION_AND_DOMINATED`，不做局部修复；活跃分支回到 H-001、H-005、H-014 三条。
- 依据：强代理 bias 和 oracle 对照通过，弱代理条件数按预测恶化；但 6 个强代理 latent-exploit cell 均未产生正增益，平均 cosine gain 为 -0.004589，direct proxy regression 在 30 次强条件比较中赢 28 次。
- 预算影响：1 单位，累计 56/100；完整保留 95 个运行行和 18-cell 汇总。
- 反事实判断：若尚未投入时间，仍会执行这项 E0 以低成本排除伪创新；不会删掉 direct proxy regression、缩小 gain 阈值或把低偏差重新包装成成功。
- G6：`NOT_YET_PASS`。活跃组合再次低于并行下限 4，未获得 E1 或语言模型训练权限。
- 下一项允许任务：从 H-001/H-005/H-014 未覆盖的机制族做第二轮定向再生成；不得把 H-021 改名复活。

## D-0014 — 2026-08-02

- 阶段：P7 / G6（第二轮定向组合修复）
- 动作：`PASS_REGENERATION_ROUND_2_RETAIN_H027`
- 决定：H-027–H-032 六个候选中，仅保留 H-027 `Audit-Identified Gradient Set Direction`；其余五个按直接已有、ensemble 复活或诊断/调度无独立机制淘汰，活跃组合恢复为 4 条。
- 依据：新增 7 篇官方 arXiv/MinerU 直接近邻，语料达到 219 篇；第二轮新颖性、等价性、理论风险审查和 H-027 E0 预注册。
- 预算影响：4 单位，累计 60/100；其中同步补记 H-021 E0 的 1 单位后，预算台账与状态一致。
- 反事实判断：若尚未投入时间，仍只保留 H-027，因为它直接把 clean-gradient 变成 set-valued estimand，并允许在 1 单位 E0 中用 scalar DRO、point correction 和 abstention 反例否定；不会保留已直接发表的 uncertainty、confidence-sequence、ensemble、commit-first 或 onset-reset 方案。
- 边界：H-027 只获得二维解析/合成 E0 权限；不得进入 E1 或语言模型训练。
- 下一项允许任务：实现并冻结 H-027 gradient-set E0，提交代码快照、绑定预注册后一次性执行。

## D-0015 — 2026-08-02

- 阶段：P7 / G6
- 动作：`FREEZE_H027_E0_IMPLEMENTATION`
- 决定：冻结 H-027 的二维 audit-identified gradient-set 实现、五个随机种子、80 个参数 cell、六类控制、四个非 oracle 基线和成功/失败阈值；正式结果尚未生成。
- 依据：精确枚举两类通道区间的四个 gradient vertex，计算凸包到原点的最小范数点；新增 6 项几何一致性测试，与既有测试合计 16 项全部通过。
- 预算影响：0（尚未执行预注册实验）。
- 反事实判断：若尚未投入本轮时间，仍会保留 point-limit、zero-in-set、wide interval、misspecified interval 和 scalar-equivalence 控制，因为它们分别约束退化、安全、过度 abstention、覆盖失效和伪创新风险。
- 边界：必须先提交并推送不可变实现，再把精确提交哈希写入预注册；不得在看到正式结果后更改网格、强识别定义、基线、控制或阈值。
- 下一项允许任务：提交并推送冻结实现，绑定其精确提交哈希，然后一次性执行 `E0-H027-AUDIT-IDENTIFIED-GRADIENT-SET`。

## D-0016 — 2026-08-02

- 阶段：P7 / G6
- 动作：`BIND_H027_E0_CODE_SNAPSHOT`
- 决定：H-027 预注册绑定到已推送的不可变实现提交 `b2778d683b22d8f7a24f60e3d3443abb2671aed2`。
- 依据：该提交包含冻结实现、配置、400 个 valid-coverage 运行行、六类控制、输出格式及 16 项通过的数学一致性测试；绑定时 H-027 正式结果与汇总表均不存在。
- 预算影响：0（仍未执行实验）。
- 反事实判断：若尚未投入时间，仍会先做精确提交绑定，因为它禁止根据正式结果调整区间映射、强识别定义、标量基线或 abstention 规则。
- 边界：完整网格只执行一次；不得删除 misspecified、wide-interval 或负结果，不得在运行后改阈值或静默重跑。
- 下一项允许任务：从绑定实现一次性执行 `E0-H027-AUDIT-IDENTIFIED-GRADIENT-SET`，保存全部原始行并按原预注册阈值审查。

## D-0017 — 2026-08-02

- 阶段：P7 / G6
- 动作：`REJECT_H027_NO_INCREMENTAL_GAIN`
- 决定：H-027 记为 `REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED`，不做局部修复；活跃分支回到 H-001、H-005、H-014 三条。
- 依据：400 个 valid-coverage 行的 false-positive direction rate 最大为 0，64 个 strong cell 的最小 cosine 为 0.995153；但 45 个目标 cell 中 0 个达到 0.05 gain，最大仅 0.011294，H-001 midpoint 赢 332/400 次。
- 预算影响：1 单位，累计 61/100；完整保留 400 个正式行、30 个控制行和 80-cell 汇总。
- 反事实判断：若尚未投入时间，仍会执行这项 E0，因为它证明 set-valued 安全性不自动构成超过 point correction 的算法收益；不会删基线、降低阈值或事后增加非线性集合。
- G6：`NOT_YET_PASS`。活跃组合再次低于并行下限 4，未获得 E1 或语言模型训练权限。
- 下一项允许任务：从当前语料的未覆盖机制族开展第三轮定向再生成；不得把 H-021 或 H-027 改名复活。

## D-0018 — 2026-08-02

- 阶段：P7 / G6（第三轮定向组合修复）
- 动作：`PASS_REGENERATION_ROUND_3_RETAIN_H033`
- 决定：H-033–H-038 六个候选中，仅条件性保留 H-033 `Privacy-Stable Reusable Audit Gradient`；其余五个因直接已有或等价于 H-001/H-005 的旧对象淘汰，活跃组合恢复为 4 条。
- 依据：新增 12 篇官方 arXiv/MinerU 直接近邻，语料达到 231 篇；第三轮新颖性、等价性、理论风险审查和 H-033 E0 预注册。
- 预算影响：4 单位，累计 65/100；仍保留 35 单位，满足至少 30% 探索/重启保留要求。
- 反事实判断：若尚未投入时间，仍只暂留 H-033，因为它把 adaptive leakage from reusable audits 设为独立失败对象，并强制与 Reusable/Generic Holdout 和 private reward model 正面对比；不会保留已有 causal reward、trusted projection、ARA 或 R2M 的换名。
- 边界：H-033 只获得解析/合成 E0 权限；不得把 DP 隐私保证当作 reward robustness 结果，也不得开始 E1 或语言模型训练。
- 下一项允许任务：实现并冻结 H-033 adaptive audit-query E0，提交代码快照、绑定预注册后一次性执行。

## D-0019 — 2026-08-02

- 阶段：P7 / G6
- 动作：`FREEZE_H033_E0_IMPLEMENTATION`
- 决定：冻结 H-033 的 binary clean-audit adaptive-query 实现、5 个随机种子、144 个参数 cell、7 个对照方法、7 类控制、zCDP accountant、隐私过滤器和结果聚合规则；正式结果尚未生成。
- 依据：每条向量查询先做 contribution L2 clipping，再按 `sigma = sensitivity / sqrt(2 rho_per_query)` 校准 Gaussian release；新增 9 项一致性测试，与既有 16 项测试合计 25 项全部通过。
- 反事实判断：若尚未投入本轮时间，仍会保留 Reusable Holdout、Generic Holdout、once-trained private reward model、H-001 single query、privacy-off、nonadaptive、clip-violation 与 population-drift 对照，因为缺少任一项都可能把普通 DP/holdout 搬用误判为新算法收益。
- 预算影响：0（尚未执行预注册网格，累计仍为 65/100）。
- 边界：必须先提交并推送不可变实现，再把精确提交哈希写入预注册；不得在看到正式结果后更改数据生成、网格、基线、控制、聚合或阈值。
- 下一项允许任务：提交并推送冻结实现，绑定其精确提交哈希，然后唯一一次执行 `E0-H033-PRIVACY-STABLE-REUSABLE-AUDIT-GRADIENT`。

## D-0020 — 2026-08-02

- 阶段：P7 / G6
- 动作：`BIND_H033_E0_CODE_SNAPSHOT`
- 决定：H-033 预注册绑定到已推送的不可变冻结提交 `90dbd2ea0db51d1a37adabb5678baab8e13bf24d`。
- 依据：该提交包含冻结实现、144 个参数 cell、7 个基线、7 类控制、输出格式及 25 项通过的测试；绑定时 H-033 正式原始结果与汇总表均不存在。
- 预算影响：0（仍未执行实验，累计保持 65/100）。
- 反事实判断：若尚未投入本轮时间，仍会先绑定冻结提交，因为这禁止根据正式结果调整 DP calibration、adaptive query generator、Holdout 对照、聚合或成功阈值。
- 边界：完整网格只能运行一次；必须保留低 rho、高维、population drift、nonadaptive 和 privacy-off 结果，不得静默重跑或结果后调参。
- 下一项允许任务：从绑定提交唯一一次执行 `E0-H033-PRIVACY-STABLE-REUSABLE-AUDIT-GRADIENT`，保存全部原始行并按原预注册门槛审查。

## D-0021 — 2026-08-02

- 阶段：P7 / G6
- 动作：`REJECT_H033_PRIVACY_UTILITY_AND_INCREMENTAL_GAIN`
- 决定：H-033 记为 `REJECTED_FAILED_PREDICTION_AND_DOMINATED`，不做局部修复；活跃分支回到 H-001、H-005、H-014 三条。
- 依据：唯一正式运行完整保留 5760 个 method-seed 行、35 个控制行和 1152 个汇总 cell；118/144 个 H-033 cell bias 超标、127/144 cosine 不足、82/144 方向错误率超标、24/144 round ratio 不足，qualifying gain cell 为 0。
- 增量判断：相对 naive 和最佳 exact non-oracle 的最大 cosine gain 分别只有 0.000495 和 0.003427；H-001 与 Thresholdout 显著更稳，无法支持新算法价值。
- 实现判断：privacy-off 与 naive 完全一致，rho 会计最大浮点误差 `4.884981e-15`，clipping 控制通过；失败来自预注册的隐私—效用与增量主张，而非执行错误。
- 预算影响：1 单位，累计 66/100；完整保留负结果，不重跑、不调参。
- 反事实判断：若尚未投入本轮时间，仍会执行该 E0 以区分“DP 接口正确”和“clean-gradient 算法有效”；不会删除低 rho/高维/漂移 cell、结果后重分配 rho 或降低 gain 阈值。
- G6：`NOT_YET_PASS`。活跃组合再次低于并行下限 4，未获得 E1 或语言模型训练权限。
- 下一项允许任务：从尚未覆盖且不等价于 H-001/H-005/H-014/H-021/H-027/H-033 的机制族开展第四轮定向再生成。

## D-0022 — 2026-08-02

- 阶段：P7 / G6（第四轮定向组合修复）
- 动作：`PASS_REGENERATION_ROUND_4_RETAIN_H039`
- 决定：H-039–H-044 六个候选中，仅条件性保留 H-039 `Channel-Set Advantage Sign Certificate`；其余五个因直接已有或缺少新更新机制淘汰，活跃组合恢复为 H-001、H-005、H-014、H-039 四条。
- 依据：新增 6 篇官方 arXiv 直接近邻并经 MinerU `vlm` 全量解析，唯一语料达到 237 篇；完成第四轮新颖性、等价性、理论风险审查和 H-039 E0 预注册。
- 增量判断：H-039 将 audit-compatible contextual verifier channel 映射为逐 completion clean-advantage interval，在聚合前认证符号；E0 强制与 H-001、H-027、H-010、SignCert-PO、scalar pessimism 和 matched-acceptance filtering 比较。
- 预算影响：4 单位，累计 70/100；恰好保留 30 单位探索/重启储备，H-039 正式 E0 的 1 单位尚未消耗。
- 反事实判断：若尚未投入时间，仍只暂留 H-039，因为它改变识别粒度且有明确等价性反例；不会保留已直接发表的 correlated-proxy DRO、reward uncertainty portfolio、confidence reward 或 causal tampering 方案。
- 边界：H-039 只获得解析/合成 E0 权限；不得开始 E1 或语言模型训练，不得以降低接受率替代算法增益。
- 下一项允许任务：实现并冻结 H-039 E0，提交不可变代码快照、绑定精确提交后只执行一次正式网格。

## D-0023 — 2026-08-02

- 阶段：P7 / G6
- 动作：`FREEZE_H039_E0_IMPLEMENTATION`
- 决定：冻结 H-039 的解析/合成实现、5 个固定种子、216 个聚合 cell/1080 个 seed cell、8 个强对照、8 类控制、共享组中心规则、判定阈值和输出结构；正式结果尚未生成。
- 识别实现：每个 context 的 FP/FN 置信矩形联合枚举四角，所有 completion 在同一角点配置下共同完成 group centering；不得混合不可同时实现的逐样本角点。只有 clean-advantage interval 全正或全负时才以最坏绝对 margin 更新。
- 等价性实现：H-010 uncertainty mask 与 matched-random filter 严格匹配 H-039 接受数；SignCert-PO 匹配平均 parameter/channel radius；H-027 使用同一组 channel-corner gradient set；所有方法共享 audit、completion 和 score。
- 安全实现：命令行必须同时提供显式正式执行开关与冻结 token，且结果文件已存在时拒绝覆盖或重跑；import 和测试均不写正式结果。
- 验证：新增 9 项通道反演、联合角点、共享中心、最坏 margin、point-limit、matched acceptance/radius、zero-mass 和装配测试；与既有 25 项合计 34 项通过。
- 预算影响：0（正式 E0 尚未运行，累计仍为 70/100）。
- 反事实判断：若尚未投入本轮时间，仍会保留逐 context 联合枚举、共享 group baseline、SignCert-PO/H-027 强对照和等接受率控制，因为缺少任一项都可能制造虚假符号证书收益。
- 边界：必须先提交并推送不可变实现，再将精确提交哈希绑定到预注册；不得根据正式结果调整数据、角点映射、强识别定义、基线、控制或阈值。
- 下一项允许任务：提交并推送冻结实现，绑定其精确提交哈希，然后唯一一次执行 `E0-H039-CHANNEL-SET-ADVANTAGE-SIGN-CERTIFICATE`。

## D-0024 — 2026-08-02

- 阶段：P7 / G6
- 动作：`BIND_H039_E0_CODE_SNAPSHOT`
- 决定：H-039 预注册绑定到已推送的不可变冻结提交 `ddb66391e42bbaf5e63c85949df6c4fac8d32414`。
- 依据：该提交包含 joint-context channel-corner 枚举、共享 group centering、216 个聚合 cell/1080 个 seed cell、8 个基线、8 类控制、输出结构、单次执行保护及 34 项通过的测试；绑定时正式原始结果与汇总表均不存在。
- 完整性判断：当前 code、config 和 test 的 Git blob 与冻结提交逐一一致，绑定提交只改变预注册、治理记录、校验器和 README，不回改实现或阈值。
- 预算影响：0（正式实验尚未执行，累计保持 70/100）。
- 反事实判断：若尚未投入本轮时间，仍会先独立完成提交绑定，因为这能阻止依据正式结果修改角点枚举、共享中心、SignCert-PO radius matching、matched acceptance 或成功阈值。
- 边界：完整网格只能运行一次；必须保留所有 1080 个 seed cell、40 个控制行、216-cell 汇总及负结果，不得静默重跑或结果后调参。
- 下一项允许任务：从绑定实现唯一一次执行 `E0-H039-CHANNEL-SET-ADVANTAGE-SIGN-CERTIFICATE`，保存全部原始行并按原预注册门槛审查。

## D-0025 — 2026-08-03

- 阶段：P7 / G6
- 动作：`REJECT_H039_NO_INCREMENTAL_GAIN`
- 决定：H-039 记为 `REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED`，不做局部修复；活跃分支回到 H-001、H-005、H-014 三条。
- 依据：唯一正式运行完整保留 1080 个 seed cell、40 个控制行和 216-cell 汇总；有效覆盖 false-certified sign 与 harmful update 均为 0，72 个 strong cell 全部通过，但 96 个目标 cell 中 qualifying gain 为 0。
- 增量判断：相对最佳 non-oracle 的最大 cosine gain 仅 `0.000001279`、harmful-rate reduction 最大为 0；H-001/H-027 平均 cosine 为 0.999887/0.998942，高于 H-039 的 0.877656。
- 压力判断：0.20 宽区间产生 15 个 zero-mass cell 与 4 个负 cosine cell，最坏 cosine 为 -0.218434；逐样本符号安全不保证选择性聚合后的全局梯度安全。
- 预算影响：1 单位，累计 71/100，剩余 29 单位，低于声明的 30 单位探索/重启储备。
- 反事实判断：若尚未投入时间，仍会执行该 E0 以区分“逐样本符号安全”和“超过 point/global-set correction 的算法价值”；不会删除 H-001/H-027、降低增益阈值、扩大 misspecification 或结果后重跑。
- G6：`NOT_YET_PASS`。活跃组合低于 4，且储备约束已触发；不得自动开展第五轮付费再生成。
- 下一项允许任务：进行零预算治理复核，在终止本发现周期与显式重分配储备之间作出决定；此前不得新增 E0、进入 E1 或训练语言模型。

## D-0026 — 2026-08-03

- 阶段：P7 / G6 → 全局回退关闭
- 动作：`CLOSE_DISCOVERY_CYCLE_GLOBAL_FALLBACK`
- 决定：关闭当前 Q-001 发现周期，不重分配剩余 29 单位储备；H-001/H-005/H-014 保留 `E0_VALIDATED` 证据并转为 `E0_VALIDATED_PAUSED_GLOBAL_FALLBACK`，不是淘汰。
- 强制依据：施工方案要求始终保留至少 30% 给未探索分支或全局重启，并规定“剩余预算低于 30% 且没有候选通过 E2”必须触发全局回退。当前使用 71/100，无 E1/E2 候选。
- 证据依据：四轮替代 E0 H-021/H-027/H-033/H-039 均在实现/安全子目标部分成立后，被强 non-oracle 基线证明零增量；继续第五轮相邻变体的预期信息增益低。
- 备选判断：拒绝自动挪用储备、拒绝用三条 E0 分支越过 G6、拒绝结果后修复 H-039；选择保留 29 单位作为真正的全局重启资源。
- 沉没成本判断：若从零开始只看到当前 237 篇语料、44 条谱系、10 次正式 E0 和四次替代失败，不会继续为 Q-001 生成第五轮相邻稳健变体。
- 预算影响：0，累计保持 71/100。
- G6：`NOT_PASSED`。当前周期没有候选获得 E1、E2、语言模型训练或扩大验证权限。
- 重启条件：新的/修订的 G0/G4 章程、独立预算或人工明确重分配、至少 4 条非等价分支、重新新颖性审查和新预注册。
- 下一项允许任务：仅准备零预算最终证据与交接包；在重启条件满足前不得新增候选或实验。

## D-0027 — 2026-08-03

- 阶段：P7 全局回退关闭 / 最终交接
- 动作：`COMPLETE_CYCLE_1_EVIDENCE_HANDOFF`
- 决定：完成第一发现周期最终证据交接，仓库进入 `discovery_cycle_1_archived_handoff_ready`；不恢复任何分支、不新增候选、不运行实验。
- 交付：人类可读总报告 `cycle_1_final_evidence_handoff.md`、关键文件 LF 规范哈希清单 `cycle_1_evidence_manifest.csv`、自动交接验证器及 README/章程入口。
- 证据范围：237 篇唯一语料、15 篇核心机制、44 条假设谱系、10 条正式 E0、3 条 E0 验证后暂停分支、41 条拒绝/淘汰节点和完整 71/100 预算轨迹。
- 完整性：五组正式结果继续由固定 SHA-256、结构和专用 validator 保护；交接验证不重新执行正式实验。
- 边界：所有正面结果均停留在 E0，不得外推为真实语言模型训练或论文级算法有效性。
- 预算影响：0，累计保持 71/100，剩余 29 单位继续作为全局重启储备。
- 下一项允许任务：仅仓库维护或证据勘误；研究重启必须满足 D-0026 规定的新 G0/G4、独立预算、四分支、新颖性审查和新预注册。

## D-0028 — 2026-08-03

- 阶段：P9 终局研究价值评估（P7/G6 提前关闭路径）
- 动作：`COMPLETE_P9_TERMINAL_RESEARCH_ASSESSMENT`
- 决定：按施工方案 P9 五类框架完成第一周期终评；确认新算法候选为 0，H-001/H-005/H-014 仅作为 E0 正面研究线索暂停，不授予 P8、G7 或任何新实验权限。
- 分类：保留 contextual channel heterogeneity、nuisance causal stress、audit leverage 和符号安全/聚合增益分离等 E0 诊断；登记 coverage、distribution shift、proxy rank、privacy utility、abstention 和聚合几何边界；完整保留 7 条正式 E0 负结果及 34 条筛除候选。
- 继续判断：不继续当前 Q-001，不开展第五轮相邻机制生成；仅允许准备零成本第二周期 G0/G4 决策材料。
- 重启边界：用户必须确认真实可复现问题和资源边界，取得独立预算授权，生成至少 4 条非等价分支，重新完成新颖性/等价性审查和预注册后，方可开展任何 E0/E1 或语言模型训练。
- 预算影响：0，累计保持 71/100，剩余 29 单位继续作为全局重启储备。
- 交付：`10_deliverables/final_research_assessment.md`；该文件与最终证据交接和完整性清单共同构成第一周期终局交付物。
- 下一项允许任务：准备零成本第二周期 G0/G4 决策包；不得自动选择问题、挪用储备或运行研究实验。
