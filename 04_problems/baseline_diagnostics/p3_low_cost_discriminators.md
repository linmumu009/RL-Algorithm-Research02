# P3 低成本解释区分实验

这些不是正式算法实验，也不授权编写训练代码；它们是 P4 选择问题时用于判断“哪个解释值得研究”的最低成本测试。

| Test | 要区分的解释 | 最小设置 | 主要观测量 | 否定条件 |
|---|---|---|---|---|
| T01 可控 verifier 通道 | 性能下降来自 FN、FP 还是一般 reward scale | 固定小策略/离线 logits，注入可控非对称标签翻转 | clean-gradient cosine、bias、variance | correction 在已知噪声率下仍不能恢复方向 |
| T02 实例依赖 verifier 噪声 | 常数噪声率模型是否足够 | 按题型、长度和答案格式分层翻转 | 分层 FN/FP、校准误差、组间梯度差 | 分层模型不优于常数通道 |
| T03 长度对照 | 改善来自长度激励还是推理质量 | 同答案正确性下 padding、等价缩写和截断 | 每序列梯度质量、答案长度、真实正确率 | 控制长度后方法差异仍完整保留 |
| T04 gate 粒度 | token、sequence、soft gate 谁保留有效梯度 | 固定 rollouts/logprobs，离线重算三种更新 | 梯度支持率、cosine、有效步长 | 差异仅由学习率/总梯度范数解释 |
| T05 replay 年龄 | 复用收益来自更多更新还是高质量选择 | 同一 buffer，年龄×比例二维网格 | ESS、policy KL、覆盖度、离线目标改进 | matched-update 后 replay 无收益或探索显著下降 |
| T06 replay 范式 | policy correction 与 value bootstrap 的误差谁占主导 | 同一轨迹和奖励，对比 importance PG 与 Bellman target | bias proxy、variance、value calibration | 任一方法仅在其专用附加信号下占优 |
| T07 staleness 配对 | stale 样本劣化还是题目更难 | 按题目、长度、初始通过率配对旧/新轨迹 | matched reward、gradient cosine、KL | 匹配后 staleness 效应消失 |
| T08 真过程信用 | process reward 是否识别关键步骤 | 正确链条的局部置换、无关前缀和伪合理步骤 | step reward sensitivity、最终正确率 | proxy 更偏好流畅伪步骤 |
| T09 hacking 检测 | 高 verifier 分来自真解还是捷径 | 等价重写、同构扰动、reasoning truncation | independent correctness、TRACE AUC | 检测器对未知漏洞接近随机 |
| T10 成本协议 | throughput 增益是否转化为用户可见加速 | 固定硬件/模型/长度分布 | p50/p95 latency、tokens/s、end-to-end wall clock、质量 | 仅 tokens/s 改善而尾延迟/墙钟不变 |
