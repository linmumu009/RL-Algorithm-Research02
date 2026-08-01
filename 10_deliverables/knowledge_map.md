# P3 机制—问题—证据知识地图

## 结论

**G3：PASS**

当前 200 篇语料和冻结后的 Core 15 足以回答施工方案要求的五类问题：现有算法为何失败、哪些证据只是相关性、哪些机制本质等价、哪些组合已经被覆盖、哪些解释可以用低成本实验区分。知识地图支持进入 P4 真实问题选择，但不支持直接宣布新算法或开始训练。

## 范围与证据规则

- 语料：200 篇唯一 arXiv 论文，PDF、官方元数据和去重状态完整。
- 核心机制基底：15 篇 Core 15。
- P3 定向补证：20 篇，覆盖 verifier 噪声/奖励投机、轨迹复用和延迟/吞吐。
- 结构化产物：15 张 Mechanism Card、20 条新增 Claim Card、13 组机制冲突、10 个低成本解释区分实验。
- 理论推导、控制变量实验和直接噪声干预列为强证据；多组件系统配方、单一 benchmark 改善和未控制算力差异列为弱或相关性证据。

## 主要算法与机制家族

| 家族 | 代表 | 主要数学对象 | 已知优势 | 主要失败边界 |
|---|---|---|---|---|
| critic-free Monte Carlo | RLOO、GRPO | group/LOO baseline | 简单、低显存 | 稀疏信用、组方差、长度与 surrogate 偏差 |
| GRPO recipe variants | DAPO、Dr. GRPO | sampling、normalization、clip、loss aggregation | 针对长 CoT 的实用修复 | 多组件耦合，难证明单一因果机制 |
| trust weighting | GSPO、CISPO、SAPO | ratio 粒度与 gate 形状 | 控制 staleness 并保留梯度 | gate 偏差、粒度冲突、独立复现不足 |
| learned critic | PPO、VAPO | value、GAE、actor-critic loss | 密集长程信用 | critic bias、额外计算和训练耦合 |
| process/multi-turn reward | PRIME、SCoRe、RISE | reward granularity 和 progress | 缓解终局稀疏 | proxy 自我强化、验证器共享漏洞 |
| verifier robustness | noise correction、VerIF、TRACE、RCfD | reward channel、constraints、diagnostic metric | 直接处理代理错误 | 噪声非平稳、实例依赖、额外 judge 成本 |
| replay/off-policy | RLEP、ReVal | data age、buffer、Bellman target | 复用昂贵轨迹 | 选择偏差、staleness、bootstrap bias |
| async/system co-design | AReaL、DORA、HybridFlow、EfficientRollout | schedule、placement、decode path | 降低同步气泡和墙钟时间 | 硬件依赖；不能计作优化器新颖性 |
| paradigm expansion | RPT、VeriFree | reward source 和训练阶段 | 扩大监督来源 | 与标准 post-training 问题不可直接等价比较 |

完整谱系见 `03_taxonomy/method_lineage.md`。

## 设计维度

Core 15 的 18 个原始维度全部保留；P3 将其压缩为六个研究决策面：

1. **估计器**：Monte Carlo、LOO/group baseline、critic/GAE、Bellman value。
2. **目标几何**：token/sequence 单元、长度权重、shared-prefix 与 optimizer interaction。
3. **信赖控制**：hard clip、importance-weight clip、sequence gate、smooth gate。
4. **奖励与信用**：outcome、process、progress、self-verification、uncertainty/constraint。
5. **数据年龄**：fresh on-policy、bounded stale、success replay、value replay。
6. **执行成本**：同步/异步、调度/重分配、rollout latency 与 throughput。

## 关键失败链

### 1. outcome reward 的长度与目标错配

`outcome reward 广播 → loss 聚合决定每条序列总梯度质量 → 长度激励或梯度偏差`。Dr. GRPO 说明标准差和 response normalization 会改变训练行为；Length-Impossibility 进一步证明，在其设定下无偏和长度不变不能同时满足。因此“再找一个同时无偏且长度公平的标量归一化”是伪空白，真实问题应改为明确选择任务效用或引入额外过程信息。

### 2. clipping 不是免费信赖域

`policy mismatch → ratio 越界 → hard gate 丢梯度 → entropy/能力更新受抑制`。GSPO 把单元提升到 sequence，CISPO 只裁剪权重，SAPO 使用连续门；它们都在改变同一个 trust-weighting 对象。现有证据支持“硬 token clip 会损失有用更新”，但尚未在同计算、同有效步长、同 rollouts 下确定最佳粒度和门形状。

### 3. 密集信用可能把误差也变密集

`终局稀疏 → critic/process reward densification → 更快学习`，同时也可能 `value/proxy error → 每一步持续错误引导`。VAPO、PRIME 和 SCoRe 提供有效性证据，但不能仅凭分数提升断言中间 reward 具有真实因果含义。低成本前缀扰动和伪合理步骤测试可以区分“识别关键步骤”与“奖励流畅表面”。

### 4. verifier 的错误会随 policy 适应

`固定 verifier 漏洞 → policy 搜索漏洞 → FP 增加 → 高 reward/低真实质量`。NoisyVerifierRL 对固定非对称 FN/FP 提供直接修正；GamingVerifiers 证明 verifier 未检查的约束可因 RLVR 形成 shortcut。真正未解决的是实例依赖、随策略变化且可能对抗性的噪声，而不是简单增加一个 judge。

### 5. replay 与 async 的核心不是“能不能用旧数据”

旧轨迹可以提高样本效率，但误差来源不同：RLEP 的主要风险是正样本选择偏差和 policy mismatch；ReVal 的主要风险是 Bellman bootstrap/value extrapolation；AReaL/DORA 的主要风险是小窗口 staleness 和系统调度。DORA 的难度分层分析说明“旧样本表现差”可能只是长/难样本更晚完成的选择偏差，因此任何 staleness 结论都必须配对题目难度和长度。

### 6. throughput 与 latency 必须分开

HybridFlow、ReaLHF、OpenRLHF、NeMo-Aligner 主要证明资源编排吞吐；AReaL/DORA 证明异步墙钟加速；EfficientRollout 同时测量 rollout 和端到端 latency。系统加速可以与算法机制组合，但不能作为算法有效性或新颖性证据。

## 强证据、弱证据与矛盾证据

| 结论 | 证据级别 | 说明 |
|---|---|---|
| outcome-reward 下长度无偏与长度不变存在结构冲突 | 强 | 理论边界直接否定“万能长度归一化” |
| group gradient 具有可分析的组大小—方差关系 | 强 | U-statistic 推导与实验 |
| verifier FP/FN 会偏置 RLVR 梯度且可做通道修正 | 强 | 形式化噪声模型加受控干预 |
| verifier 未覆盖约束可诱发 reward shortcut | 强 | 受控训练和同构扰动证据 |
| 小而受控的 staleness 窗口可以不损害质量 | 中强 | DORA/AReaL 支持，但窗口依赖学习速度和系统 |
| replay 能普遍替代 fresh rollout | 弱 | RLEP/ReVal 有正结果，但范式、额外信号和模型范围不同 |
| soft gate 普遍优于 sequence 或 hard gate | 弱 | 缺少共享 rollouts、同有效步长的独立比较 |
| process reward 确实识别真实关键步骤 | 弱 | 性能提升与过程因果正确性没有完全分离 |
| 系统吞吐提升意味着用户请求更快 | 错误外推 | 必须额外报告 tail latency 和端到端时间 |

## 机制冲突与组合边界

机器可读冲突表见 `03_taxonomy/mechanism_conflicts.csv`。最重要的边界如下：

- token gate、sequence gate、soft gate 是替代性 trust-control 选择；不能在同一梯度上无定义地同时充当最终 gate。
- critic-free 与 learned critic 可以做辅助组合，但会失去 critic-free 的成本主张，且必须单独审计 value bias。
- replay 与 bounded async 可组合，但 replay 年龄、policy version 和 sampling selection 必须分别记录。
- verifier ensemble 与噪声 correction 可组合，前提是估计 error correlation；简单投票不保证降低共同漏洞。
- process reward 与 outcome reward 可组合，但 outcome 只能锚定终局，不能证明每步 proxy 正确。
- 系统加速与任意优化器理论上可组合，只要采样分布和 logprob contract 保持不变。

## 九类空白审计

| 空白类型 | 当前真实空白 | 判定 |
|---|---|---|
| 证据空白 | replay、soft gate、process reward 缺少共享 rollouts/同计算的独立比较 | 保留 |
| 诊断空白 | verifier 噪声是固定通道、实例依赖还是策略诱发尚未区分 | 高优先级 |
| 机制空白 | 缺少以 policy divergence 和 verifier uncertainty 共同控制旧轨迹权重的已验证机制 | 需先诊断，暂不生成算法 |
| 尺度空白 | verifier 修正与 replay 在更大模型、MoE、长上下文的迁移不清楚 | 保留但昂贵 |
| 条件空白 | 长度结论高度依赖 outcome/process reward 与任务效用 | 高优先级 |
| 组合空白 | noise correction × replay 可能互补，但旧轨迹 verifier error 会随政策变化 | 条件性保留 |
| 理论空白 | smooth/sequence gate 的 bias-variance 最优性缺少统一分析 | 保留 |
| 负结果空白 | 多数系统和算法未公开失败的 staleness/gate/replay 区间 | 保留 |
| 评估空白 | 缺统一报告真实正确性、hacking、p95 latency、吞吐和墙钟质量 | 高优先级 |

## 不值得继续的伪空白

- “把 GRPO 的 clip 改成 sequence clip”：GSPO 已覆盖。
- “让越界 token 仍有梯度”：CISPO/SAPO 已覆盖该核心思想。
- “过滤全对/全错组”：DAPO 已覆盖；剩余问题是分布偏差而非组件新颖性。
- “不用 critic 的 PPO”：RLOO/GRPO 已构成成熟基线。
- “缓存以前的正确推理再训练”：RLEP 已覆盖；若没有新的 staleness/selection 控制只是重命名。
- “用 value model 回放旧轨迹”：ReVal/B-Coder 谱系已经存在。
- “再加一个 LLM judge 防 reward hacking”：VerIF、RISE、ensemble 方向已有；未知的是相关错误和适应性攻击。
- “找到同时无偏且长度完全不变的 outcome-reward 权重”：在已证明的条件下不可能。
- “异步训练就是新优化算法”：AReaL/DORA 属于 algorithm-system co-design，系统调度本身不构成优化器新颖性。

## 可低成本区分的解释

已登记 10 个测试，见 `04_problems/baseline_diagnostics/p3_low_cost_discriminators.md`。最有辨别力的前三项是：

1. 可控、实例依赖的 verifier FP/FN 通道：直接测 clean-gradient cosine，区分 reward scale 与真实方向偏差。
2. replay 年龄 × replay 比例网格：同时观察 ESS、policy KL、探索覆盖和 matched-update 收益。
3. 同 rollouts 的 token/sequence/soft gate 离线重算：剥离采样和算力差异，比较梯度支持与方向。

## G3 五问

### 为什么现有算法在某类场景失败？

因为 estimator、objective geometry、reward channel、data age 和 execution schedule 各自引入不同偏差；长 CoT 会把长度聚合、稀疏信用、ratio 方差和 rollout 长尾同时放大。知识地图已把这些原因分开，而不是统称为“训练不稳定”。

### 哪些结论只有相关性证据？

DAPO/VAPO 等多组件 recipe 的单项机制贡献、process reward 的真实步骤因果、soft gate 的普遍最优性、replay 的跨任务优势，以及多数系统吞吐对最终算法质量的影响，都仍主要是相关性或条件性证据。

### 哪些机制本质等价？

RLOO/GRPO 都是多样本 control variate；GSPO/CISPO/SAPO 都是 importance trust weighting 的粒度/门函数变体；RCfD/constraints/uncertainty penalty 都是在代理奖励空间限制过优化，但改变的数学对象不同；系统调度方法不应与策略优化机制混为一类。

### 哪些“新组合”已有工作覆盖？

sequence clipping、越界 token 保梯度、动态过滤无信息组、成功轨迹 replay、value replay、混合 verifier、异步 bounded-staleness 和 rollout speculative decoding 均已有直接工作覆盖。

### 哪些问题能用低成本实验辨别不同解释？

verifier 通道噪声、长度梯度偏差、gate 粒度、replay 年龄、staleness 难度混淆、process reward 因果性和 latency/throughput 混淆均已有不需要大规模训练的预注册候选测试。

## P4 输入边界

进入 P4 后应在以下真实问题中选择，而不是直接拼算法：

1. **适应性 verifier 噪声**：现有常数 FN/FP correction 对实例依赖、策略诱发错误是否失效。
2. **安全轨迹复用边界**：收益究竟来自额外更新、正样本选择，还是可泛化的旧轨迹信息。
3. **trust gate 的粒度与连续性**：在同 rollouts 和同有效步长下，token/sequence/soft gate 的方向偏差如何变化。

其中第 1 项证据缺口最清晰、低成本可证伪性最好，也最能连接 reward robustness 与 RLVR 实际风险；建议作为 G4 的首选问题。
