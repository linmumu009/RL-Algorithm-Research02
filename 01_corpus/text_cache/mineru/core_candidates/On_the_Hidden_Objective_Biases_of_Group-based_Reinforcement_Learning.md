# On the Hidden Objective Biases of Group-based Reinforcement Learning

Aleksandar Fontana<sup>1,2</sup>\*, Marco Simoni<sup>2,3</sup>\*, Giulio Rossolini<sup>1</sup>, Andrea Saracino<sup>1</sup>, Paolo Mori<sup>2</sup>

<sup>1</sup> Department of Excellence in Robotics and AI, TeCIP, Scuola Superiore Sant’Anna, Pisa

<sup>2</sup> Institute of Informatics and Telematics, National Research Council of Italy, Pisa

<sup>3</sup> National Doctorate on Artificial Intelligence, Sapienza Università di Roma

## Abstract

Group-based reinforcement learning methods, like Group Relative Policy Optimization (GRPO), are widely used nowadays to posttrain large language models. Despite their empirical success, they exhibit structural mismatches between reward optimization and the underlying training objective. In this paper, we present a theoretical analysis of GRPO style methods by studying them within a unified surrogate formulation. This perspective reveals recurring properties that affect all the methods under analysis: (i) non-uniform group weighting induces systematic gradient biases on shared prefix tokens; (ii) interactions with the AdamW optimizer make training dynamics largely insensitive to reward scaling; and (iii) optimizer momentum can push policy updates beyond the intended clipping region under repeated optimization steps. We believe that these findings highlight fundamental limitations of current approaches and provide principled guidance for the design of future formulations.

## 1 Introduction

Recent advances in Large Language Model (LLM) post-training have shown that reinforcement learning methods based on group-level feedback can effectively improve reasoning performance while avoiding the cost of explicit value-function estimation, as used in previous works (Ouyang et al., 2022; Yao et al., 2023). Among these approaches, Group Relative Policy Optimization (GRPO) and related methods have gained widespread adoption due to their simplicity and scalability, and are now commonly used in post-training pipelines for reasoning-oriented models (Shao et al., 2024; Liu et al., 2025a; Zheng et al., 2025; Yu et al., 2025).

Despite their empirical success, GRPO style methods rely on a surrogate objective whose optimization dynamics remain only partially understood. Several recent works have reported unexpected behaviors during training, including lengthrelated biases (Liu et al., 2025b), sensitivity to formatting tokens (Simoni et al., 2025), reward hacking in multi-objective settings (Ichihara et al., 2025), and instability across different optimization regimes However, these findings represent fragmented empirical observations, and a unified formal framework that systematically connects and further extends them to the surrogate objective’s implicit inductive biases is lacking.

This work offers a unified critical analysis of group-based optimization methods. We propose a general formulation of GRPO style methods, showing ten recent approaches as special cases. This view reveals shared issues, showing that the surrogate objective is often dominated by weighting schemes, regularization, and importance sampling, rather than by pure reward maximization. Building on this formulation, we identify three recurring properties of GRPO style training dynamics: (i) we analyze token-level gradients to demonstrate that non-uniform weighting induces systematic biases on shared prefix tokens; (ii) we study the interaction with AdamW (Loshchilov and Hutter, 2017), demonstrating that the training process remains invariant to global reward scaling across various scenarios; (iii) we show that optimizer momentum can drive policy updates beyond the intended clipping boundaries during multi-step optimization. Beyond empirical performance, our analysis offers theoretical insights exposing a divergence between the surrogate objective and the true training goal. By characterizing these dynamics, our findings provide the community with a reference for the design and interpretation of LLM post-training strategies.

## 2 Related Work

Recent work has started to study the problems arising during GRPO style post-training. Several stud-

$$
\mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \mathbb {E} _ {q, \{o _ {i} \}} \left[ \sum_ {i = 1} ^ {G} \left(\sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t} \min \left(s _ {i, t} (\theta) A _ {i}, \operatorname{clip} \left(s _ {i, t} (\theta), 1 - \varepsilon_ {\text {low}}, 1 + \varepsilon_ {\text {up}}\right) A _ {i}\right)\right) - \beta R (\theta) \right] \tag {1}
$$

ies report optimization issues, like systematic biases toward output length (Liu et al., 2025b). Other analyses propose simple stabilization techniques, including masking strategies, to improve robustness across different training regimes (Mroueh et al., 2025). In multi-objective settings, GRPO has is vulnerable to reward hacking, motivating the use of normalization-based mitigations (Ichihara et al., 2025). Additional work focuses on issues that emerge at the token level: formatting tokens often dominate optimization (Simoni et al., 2025), and simple cues like sequence length can drive learning (Xin et al., 2025). Clipping mechanisms used in PPO and GRPO have also been shown to introduce systematic entropy biases (Park et al., 2025). Complementary to analyses of clipping and instability, SFPO introduces a reposition-before-update scheme to control off-policy drift induced by repeated inner updates (Wang et al., 2025). Based on these observations, our work provides a unified analysis of why the surrogate loss can be misleading, how shared prefixes bias token-level gradients, and how optimizer dynamics interact with clipping under repeated updates.

## 3 Unified Formulation

In the following, we introduce a generalized surrogate objective that serves as a unified framework for a broad class of recent group-based policy optimization methods, including GRPO (R1 (Shao et al., 2024) and v3.2 (Liu et al., 2025a)), GSPO<sup>1</sup> (Zheng et al., 2025), GTPO (Simoni et al., 2025), DAPO (Yu et al., 2025), CPPO (Lin et al., 2025), Dr. GRPO (Liu et al., 2025b), GPG (Chu et al., 2025), CISPO (Chen et al., 2025), and GCPO (Wu and Liu, 2025). For a group of $G$ outputs $\{ o _ { i } \} _ { i = 1 } ^ { G }$ generated from the same prompt $q ,$ the advantage $A _ { i }$ for the i-th output is calculated by standardizing the reward $r _ { i }$ against the group’s distribution:

$$
A _ {i} = r _ {i} - \left(\frac {1}{G} \sum_ {j = 1} ^ {G} r _ {j}\right)\tag{2}
$$

This advantage term drives the GRPO style objective (Eq. 1). $A _ { i }$ usually determines the direction of the token-level policy updates weighting coefficients $\alpha _ { i , t }$ . Optimization typically involves $\mu$ gradient updates on a fixed group of samples, which progressively induces off-policy drift. To mitigate this, it is employed a token-level importance ratio

$$
s _ {i, t} (\theta) \propto \frac {\pi_ {\theta} (y _ {i , t} \mid x , y _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i , t} \mid x , y _ {i , <   t})}\tag{3}
$$

clipped to $[ 1 - \varepsilon _ { \mathrm { l o w } } , 1 + \varepsilon _ { \mathrm { u p } } ]$ following PPO (Schulman et al., 2017). Finally, a regularization term $R ( \theta )$ , generally the KL divergence from a reference policy weighted by $\beta ,$ is added for training stability. $\mathbf { A } \mathbf { s }$ detailed in Table 1, each method represents a distinct configuration of $\operatorname { E q }$ . 1 regarding the three core components: the weighting coefficients $\alpha _ { i , t } .$ the importance ratio $s _ { i , t } ( \theta )$ , and the regularization term $R ( \theta )$ . Eq. 1 acts strictly as an optimization mechanism, not as a performance metric. Since advantages are group-centered $( \sum _ { i } A _ { i } = 0 )$ , the loss value does not exclusively reflect reward improvement. Instead, the loss magnitude is dominated by nuisance factors, like importance sampling fluctuations $( s _ { i , t } \neq 1 )$ during multi-step updates. Consequently, the surrogate loss offers no monotonic or reliable signal for policy improvement and should not be used to monitor training progress (Achiam, 2018) (see Appendix A for formal analysis).

## 4 Biases in Token-level Gradients

In this section, we analyze how GRPO style objectives affect tokens that are shared across multiple answers. We focus on the initial portion of the generated sequences, where answers are most likely to share identical prefixes and where, due to leftto-right autoregressive generation, updates applied to early tokens have a global effect on the entire sequence. Consider the first k tokens that are identical across a subset of answers. For these positions, the policy probability $\pi _ { \boldsymbol { \theta } } { \left( y _ { t } \mid x , y _ { < t } \right) }$ is the same for all answers in the group. As a result, the gradient contributions derived from Equation 1 for these shared tokens differ only through their weighting terms and associated advantages. We formalize the exact form of this aggregate gradient contribution for shared prefixes in the following proposition:

Table 1: Instantiation of the unified objective in Eq. 1 for representative GRPO style methods. Weights α, importance ratios $s _ { i , t } ( \theta )$ , and regularization terms $R ( \theta )$ are reported for each algorithm. The definitions are: $\begin{array} { r } { \alpha _ { i } ^ { S } : = \frac { 1 } { { G } \cdot \lvert o _ { i } \rvert \cdot \sigma ( r ) } } \end{array}$ $\begin{array} { r } { I : = \frac { \pi _ { \theta } } { \pi _ { \theta _ { o l d } } } } \end{array}$ , and $\begin{array} { r } { \mathcal { D } _ { K L } : = \frac { \pi _ { r e f } } { \pi _ { \theta } } - \log \frac { \pi _ { r e f } } { \pi _ { \theta } } - 1 } \end{array}$ . Unless otherwise specified, dependence on $\left( y _ { i , t } \mid x , y _ { i , < t } \right)$ is implicit.

<table><tr><td>Algorithm</td><td> $\alpha_{i,t}$ </td><td> $s_{i,t}(\theta)$ </td><td> $R(\theta)$ </td><td>Algorithm</td><td> $\alpha_{i,t}$ </td><td> $s_{i,t}(\theta)$ </td><td> $R(\theta)$ </td></tr><tr><td>GRPO R1</td><td> $\alpha_{i}^{S}$ </td><td> $I$ </td><td> $\mathcal{D}_{KL}$ </td><td>CPPO</td><td> $\alpha_{i}^{S}1_{\{|A_{i}|>\gamma\}}$ </td><td> $I$ </td><td> $\mathcal{D}_{KL}$ </td></tr><tr><td>GRPO v3.2</td><td> $\frac{M_{i,t}}{G|o_{i}|}$ </td><td> $I$ </td><td> $I \cdot \mathcal{D}_{KL}$ </td><td>Dr GRPO</td><td> $\frac{1}{G}$ </td><td> $I$ </td><td> $\mathcal{D}_{KL}$ </td></tr><tr><td>GSPO</td><td> $\alpha_{i}^{S}$ </td><td> $sg\left[\frac{\frac{\pi_{\theta}(y_{i}|x_{i})}{\pi_{\theta_{old}}(y_{i}|x_{i})}}{\pi_{\theta}}\right]\pi_{\theta}$ </td><td> $\times$ </td><td>GPG</td><td> $\frac{\hat{\alpha}}{F_{norm}\sum|o_{i}|}$ </td><td> $\log(\pi_{\theta})$ </td><td> $\times$ </td></tr><tr><td>GTPO</td><td> $\frac{\delta_{i}\lambda_{i,t}}{G|o_{i}|}$ </td><td> $I$ </td><td> $\frac{1}{G}\sum_{i}\frac{\delta_{i}\langle H\rangle_{i}}{|o_{i}|}\sum_{t}I\lambda_{i,t}$ </td><td>CISPO</td><td> $\frac{M_{i,t}}{\sigma(r)\sum|o_{i}|}$ </td><td> $sg[I]\log\pi_{\theta}$ </td><td> $\times$ </td></tr><tr><td>DAPO</td><td> $\frac{1}{\sigma(r)\sum|o_{i}|}$ </td><td> $I$ </td><td> $\times$ </td><td>GCPO</td><td> $\frac{1}{\sigma(r)G}$ </td><td> $\frac{\pi_{\theta}(y_{i}|x_{i})}{\pi_{\theta_{old}}(y_{i}|x_{i})}$ </td><td> $\times$ </td></tr></table>

Proposition 1. Consider a policy π optimized with $E q .$ 1 via centered advantages $( E q . \ 2 )$ . For any subset of answers ${ \tilde { G } } \subseteq G$ sharing a common prefix $y _ { i , 1 : | k | }$ , the gradient with respect to this prefix is modulated by the aggregate term $\begin{array} { r } { \mathcal { W } _ { a g g } = \sum _ { i \in \tilde { G } } \omega _ { i } A _ { i } } \end{array}$ , where $\omega _ { i } = \alpha _ { i } * s _ { i } ( \theta )$

This observation reveals a source of structural bias in token-level gradients. This phenomenon is particularly pronounced when tokens are shared across all sequences. While Eq. 2 implies that the gradient contributions of such tokens would cancel out under uniform weighting, the actual gradient they receive depends on the aggregated term ${ \mathcal W } _ { \mathrm { a g g } } .$ Consequently, the choice of weighting scheme directly determines how much influence each completion exerts on the shared prefix, independently of the semantic content of the later tokens. For example, when $\omega _ { i }$ is inversely proportional to the output length, $\begin{array} { r } { \omega _ { i } \propto \frac { 1 } { | o _ { i } | } } \end{array}$ , answers with shorter lengths and positive advantages contribute disproportionately to the gradient of the initial tokens. As a consequence, the model is implicitly encouraged to favor shorter outputs, even when length is not aligned with task quality (Liu et al., 2025b). From an optimization perspective, the induced bias on shared prefix tokens constitutes a distinct training signal. Depending on the application, this signal may be exploited, for instance, to control verbosity, or it may need to be mitigated to avoid unintended stylistic or structural preferences (Simoni et al., 2025).

## 5 Effects of AdamW Optimizer

We now turn our attention to the AdamW optimizer (Loshchilov and Hutter, 2017), the standard choice for GRPO training setups (Simoni et al., 2025; Shao et al., 2024; Yu et al., 2025; Liu et al., 2025b). Analyzing AdamW is particularly relevant in this setting, as the interplay between multiple gradient steps per group and policy clipping significantly alters optimization dynamics. The AdamW update rule is formally defined as follows:

$$
\theta_ {t} = \theta_ {t - 1} + \xi \frac {\hat {m} _ {t}}{\sqrt {\hat {v} _ {t}} + \epsilon} + \xi \lambda \theta_ {t - 1}\tag{4}
$$

$$
m _ {t} = \frac {\beta_ {1} m _ {t - 1}}{1 - \beta_ {1} ^ {t - 1}} + \frac {(1 - \beta_ {1}) g _ {t}}{1 - \beta_ {1} ^ {t}}\tag{5}
$$

$$
v _ {t} = \frac {\beta_ {2} v _ {t - 1}}{1 - \beta_ {2} ^ {t - 1}} + \frac {(1 - \beta_ {2}) (g _ {t}) ^ {2}}{1 - \beta_ {2} ^ {t}}\tag{6}
$$

where $g _ { t } = \nabla _ { \theta } \mathcal { T } _ { \mathrm { G R P O - L } } ( \theta )$ denotes the gradient of the GRPO style objective (the full derivation is reported in Appendix Eq. 15). Unlike standard gradient descent, the update depends not only on the current gradient, but also on exponentially smoothed estimates of its first- and second-order moments.

Reward Scaling. Despite the extensive literature emphasizing the criticality of reward scaling for stabilizing reinforcement learning algorithms (van Hasselt et al., 2016; Engstrom et al., 2020), the adaptive nature of AdamW warrants a re-examination of this premise in the context of

GRPO style algorithm. We investigate the effect of scaling the reward signal by a factor $\phi \in \mathbb { R } ^ { + }$ such that $r _ { i } ^ { * } = \phi r _ { i }$ . Whether applied to control signal magnitude or induced by normalization, this scaling theoretically alters the optimization landscape. We establish the following property regarding AdamW’s response to such transformations when regularization is omitted $( \beta = 0 )$ , a configuration empirically shown to enhance performance in domains like mathematics (Liu et al., 2025a).

Proposition 2. Assume $\beta = 0$ in $E q .$ . 1 and define a scaled reward $r _ { i } ^ { * } = \phi r _ { i }$ , with $\phi \in { }$ $\mathbb { R } ^ { + }$ . In the limit where the numerical stability term $\begin{array} { r } { \frac { \epsilon } { \phi \sqrt { \hat { v } _ { t } } } \to 0 , } \end{array}$ , the Adam update in Eq. 4 is invariant to the scaling factor $\phi .$

This result, formally derived in the Appendix C, shows that without regularization, uniformly scaling the reward does not alter the optimization trajectory under AdamW. Intuitively, the adaptive normalization induced by $\hat { v } _ { t }$ compensates for changes in gradient magnitude, effectively canceling out the effect of reward scaling and preserving the update direction. However, this invariance no longer holds once a regularization term is introduced (i.e., $\beta \neq 0 )$ . In this case, scaling the reward modifies the relative strength between the reward-driven gradient and the regularization penalty, making the optimization dynamics explicitly dependent on the reward scale. As a consequence, the choice of reward normalization becomes a meaningful design decision in GRPO style training. Even when $\beta = 0$ the invariance described in Proposition 2 relies on the numerical stability constant ϵ being negligi ble compared to $\phi \sqrt { \hat { v } _ { t } }$ Although ϵ is typically set to a small value $( 1 0 ^ { - 8 }$ in PyTorch implemen-$\tan ^ { 2 } )$ , some reinforcement learning implementations adopt larger values such as $1 0 ^ { - 5 }$ (Huang et al., 2022). In these cases, ϵ may become comparable to small gradient magnitudes, reintroducing sensitivity to reward scaling. Despite its potential impact on convergence, the value of ϵ is often omitted from reported hyperparameters.

Adam Overshoot. We next analyze the interplay between AdamW and the clipping mechanism in GRPO style objectives. This interaction is critical when performing multiple optimization steps on the same batch, where clipping is intended to enforce a trust region. We consider a scenario where the parameter vector reaches the clipping boundary at iteration T. We demonstrate that even if the advantage-based gradients vanish at this boundary, the optimizer’s internal dynamics do not cease, driving updates beyond the intended constraints.

Proposition 3. Let $\theta _ { T }$ denote a parameter state at iteration T that lies on the boundary of the clipped region. Even if the instantaneous gradient ofthe advantage term becomes zero for all $t > T$ , the Adam update $\Delta \theta _ { T + k }$ continues to move the parametersfurther into the clipped region.

The underlying reason is Adam’s momentum mechanism. Once the parameters enter the clipped region, the gradient contribution of the advantage term is suppressed by the clipping operation. However, the first moment estimate retains information from previous gradients and continues to produce non-zero updates. As a result, the optimizer keeps moving in the same direction even in the absence of a corrective gradient signal. For GRPO style algorithms, this behavior induces a form of unidirectional drift. If the policy enters an untrusted region during these updates, self-correction becomes impossible. As a result, the model progressively deviates from the trust region until new data is generated in the subsequent iteration. GRPO style algorithms converge even when clipping is inactive $( \mu = 1 )$ (Shao et al., 2024; Simoni et al., 2025; Chu et al., 2025).This implies the mechanism may be unnecessary, and its complete omission is a promising direction for future work. The derivation of Proposition 3 is in Appendix D.

## 6 Conclusion

In this work, we established a unified formulation for Group Relative Policy Optimization and its variants, revealing disconnects between heuristics and theory. Our analysis identified distinct properties: first, that specific weighting schemes introduce structural gradient biases into shared prefixes; second, the interaction between AdamW momemntum and GRPO style objective, in absence of regularization term, makes the objective insensitive to the global reward scaling; and third, that the interaction between AdamW momentum and the objective clipping mechanisms causes parameters to overshoot trust regions, undermining the stability of multi-step updates. These findings suggest that the empirical scalability of GRPO style methods is achieved at the expense of optimization transparency, necessitating a re-evaluation of current post-training strategies to ensure rigorous alignment between surrogate objectives and desired policy outcomes.

## Limitations

Our theoretical analysis relies on the assumption of standard autoregressive generation and may not fully generalize to non-standard attention mechanisms or bidirectional architectures. Additionally, while we identified the momentum-induced drift in AdamW, we did not propose a closed-form correction for the optimizer itself, leaving the development of momentum-aware clipping strategies for future work. Finally, our empirical validation of the "overshoot" phenomenon (Proposition 3) focuses on the standard GRPO style implementation and may vary under aggressive regularization regimes or alternative optimizer choices such as RMSProp or SGD.

## References

Joshua Achiam. 2018. Spinning Up in Deep Reinforcement Learning.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, Chunhui Du, Congchao Guo, Da Chen, Deming Ding, and 80 others. 2025. Minimax-m1: Scaling test-time compute efficiently with lightning attention. CoRR, abs/2506.13585.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. 2025. GPG: A simple and strong reinforcement learning baseline for model reasoning. CoRR, abs/2504.02546.

Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. 2020. Implementation matters in deep policy gradients: A case study on PPO and TRPO. CoRR, abs/2005.12729.

Shengyi Huang, Rousslan Fernand Julien Dossa, An tonin Raffin, Anssi Kanervisto, and Weixun Wang. 2022. The 37 implementation details of proximal policy optimization. In ICLR Blog Track.

Yuki Ichihara, Yuu Jinnai, Tetsuro Morimura, Mitsuki Sakamoto, Ryota Mitsuhashi, and Eiji Uchibe. 2025. Mo-grpo: Mitigating reward hacking of group relative policy optimization on multi-objective problems. arXiv preprint arXiv:2509.22047.

Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. 2025. Cppo: Accelerating the training of group relative policy optimization-based reasoning models. arXiv preprint arXiv:2503.22342.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, and 1 others. 2025a. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025b. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Youssef Mroueh, Nicolas Dupuis, Brian Belgodere, Apoorva Nitsure, Mattia Rigotti, Kristjan Greenewald, Jiri Navratil, Jerret Ross, and Jesus Rios. 2025. Revisiting group relative policy optimization: Insights into on-policy and off-policy training. arXiv preprint arXiv:2505.22257.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9. 2022

Jaesung R Park, Junsu Kim, Gyeongman Kim, Jinyoung Jo, Sean Choi, Jaewoong Cho, and Ernest K Ryu. 2025. Clip-low increases entropy and clip-high decreases entropy in reinforcement learning of large language models. arXiv preprint arXiv:2509.26114.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Marco Simoni, Aleksandar Fontana, Giulio Rossolini, and Andrea Saracino. 2025. Gtpo: Trajectory-based policy optimization in large language models. arXiv preprint arXiv:2508.03772.

Hado van Hasselt, Arthur Guez, Matteo Hessel, Volodymyr Mnih, and David Silver. 2016. Learning values across many orders of magnitude. In Advances in Neural Information Processing Systems 29:

Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 4287–4295.

Ziyan Wang, Zheng Wang, Jie Fu, Xingwei Qu, Qi Cheng, Shengpu Tang, Minjia Zhang, and Xiaoming Huo. 2025. Slow-fast policy optimization: Reposition-before-update for llm reasoning. arXiv preprint arXiv:2510.04072.

Hao Wu and Wei Liu. 2025. GCPO: when contrast fails, go gold. CoRR, abs/2510.07790.

Rihui Xin, Han Liu, Zecheng Wang, Yupeng Zhang, Dianbo Sui, Xiaolin Hu, and Bingning Wang. 2025. Surrogate signals from format and length: Reinforcement learning for solving mathematical problems without ground truth answers. arXiv preprint arXiv:2505.19439.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems 36: Annual Confer ence on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, and 1 others. 2025. Group sequence policy optimization. arXiv preprint arXiv:2507.18071.

## A Inadequacy of the Surrogate Loss as a Performance Proxy

This section provides a detailed analysis of why the GRPO style surrogate objective suffers from limitations in representing a reliable performance proxy (an intermediate signal intended to estimate the underlying objective). While the objecitve is well-defined as an optimization signal, its numerical value does not admit a consistent or monotonic relationship with reward improvement, even under idealized conditions. We formalize this limitation in Proposition 4 and explicitly characterize the mechanisms that decouple the surrogate loss from true policy quality.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Proposition 4. Consider the surrogate objective $\mathcal{J}_{\text{GRPO-L}}(\theta)$ defined in Eq. 1. Assume that importance weights are computed with respect to a fixed reference policy $\pi_{old}$ sampled at the initial iteration, i.e., $s_{i,t} \propto \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,&lt;t})}{\pi_{old}(o_{i,t} \mid q, o_{i,&lt;t})}$. Under group-standardized advantages $\sum_{i=1}^{G} \mathcal{A}_i = 0$, the value of $\mathcal{J}_{\text{GRPO-L}}(\theta)$ is an inconsistent proxy for policy performance.
</div>

General form of the objective. Ignoring the clipping operation for analytical clarity, the GRPO style surrogate objective can be written as:

$$
\begin{array}{c} \mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \mathbb {E} _ {q, \{o _ {i} \}} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \left(\mathcal {A} _ {i} \sum_ {t = 1} ^ {| o _ {i} |} \omega_ {i, t} \rho_ {i, t} (\theta)\right) \right. \\ \left. - \beta R (\pi_ {\theta}) \right] \end{array}\tag{7}
$$

where: $\begin{array} { r } { A _ { i } \ = \ r _ { i } - \ \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } } \end{array}$ is the groupcentered advantage; $\left| o _ { i } \right|$ is the length of the i-th completion; $\begin{array} { r } { \rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( o _ { i , t } | q , o _ { i , < t } \right) } { \pi _ { \mathrm { o l d } } \left( o _ { i , t } | q , o _ { i , < t } \right) } } \end{array}$ is the tokenlevel importance sampling ratio; $\omega _ { i , t }$ aggregates algorithm-specific weighting choices $\left( \mathbf { e . g . , \ } \alpha _ { i , t } . \right.$ length normalization, masking strategies); $\beta R ( \pi _ { \theta } )$ denotes the regularization term.

The central question addressed in this section is whether the scalar value of ${ \mathcal { I } } _ { \mathrm { G R P O - L } } ( \theta )$ can be interpreted as a meaningful indicator of training progress or policy quality. To answer this question, we analyze two scenarios: (A) the first optimization step, where the current policy coincides with the sampling policy, and (B) later iterations, where the two policies diverge.

## A.1 Scenario A: First optimization step $( \rho _ { i , t } ( \theta ) = 1 )$

At the first update, the policy has not yet changed, so $\pi _ { \theta } = \pi _ { \mathrm { o l d } }$ and therefore $\rho _ { i , t } ( \theta ) = 1$ for all i, t. In this case, all importance sampling effects vanish.

We can absorb the remaining per-token design choices into a single effective weight $\tilde { \omega } _ { i , t }$ . The objective simplifies to:

$$
\mathcal {J} _ {\mathrm{align}} (\theta) = \mathbb {E} \left[ \sum_ {i = 1} ^ {G} \mathcal {A} _ {i} \Omega_ {i} - \beta R (\pi_ {\theta}) \right]\tag{8}
$$

where

$$
\Omega_ {i} = \sum_ {t = 1} ^ {| o _ {i} |} \tilde {\omega} _ {i, t}
$$

is the cumulative weight assigned to trajectory i.

This formulation makes explicit that the surrogate objective depends only on the interaction between advantages $\mathbf { \mathcal { A } } _ { i }$ and cumulative weights $\Omega _ { i }$ We now examine three representative weighting regimes.

## Case 1: Length-normalized weights

Many GRPO style methods normalize updates by sequence length, using weights of the form $\tilde { \omega } _ { i , t } =$ $\frac { C } { | o _ { i } | }$ . In this case,

$$
\Omega_ {i} = \sum_ {t = 1} ^ {| o _ {i} |} \frac {C}{| o _ {i} |} = C,
$$

which is constant across all trajectories. Substitut ing into Eq. 8 yields:

$$
\begin{array}{c} \mathcal {J} _ {\mathrm{align}} (\theta) = \mathbb {E} \Bigg [ C \underbrace {\sum_ {i = 1} ^ {G} \mathcal {A} _ {i}} _ {= 0} - \beta R (\pi_ {\theta}) \Bigg ] \\ = - \beta   \mathbb {E} [ R (\pi_ {\theta}) ]. \end{array}\tag{9}
$$

Thus, the entire reward-driven component of the objective cancels out. The surrogate loss is fully dominated by the regularization term and contains no information about relative reward improvement. In this regime, the loss value is fundamentally uninformative as a measure of policy performance.

## Case 2: Constant token-wise weights

If weights are constant per token, $\begin{array} { r } { \tilde { \omega } _ { i , t } = { \cal C } \left( \mathrm { e . g . } \right. } \end{array}$ Dr. GRPO), then the cumulative weight scales linearly with output length:

$$
\Omega_ {i} = C \left| o _ {i} \right|.
$$

The objective becomes:

$$
\mathcal {J} _ {\mathrm{align}} (\theta) = \mathbb {E} \left[ C \sum_ {i = 1} ^ {G} \mathcal {A} _ {i} | o _ {i} | - \beta R (\pi_ {\theta}) \right]\tag{10}
$$

In this case, the loss no longer cancels, but its sign and magnitude reflect whether positively advantaged completions tend to be longer or shorter than negatively advantaged ones. The objective therefore acts as a proxy for sequence length statistics, not for reward maximization or task correctness.

## Case 3: General parametric weighting

More complex methods (e.g., GTPO, CPPO) define $\tilde { \omega } _ { i , t }$ as a non-trivial function of i and t. Here, the reward-weighted sum does not vanish, but instead satisfies:

$$
\mathcal {J} _ {\text { align }} (\theta) \propto \operatorname{Cov} (\mathcal {A}, \Omega)\tag{11}
$$

Although the loss is non-zero, its value is entirely determined by the interaction between the advantage distribution and the chosen weighting scheme. Unless the weights are explicitly designed to encode task-relevant structure, the loss magnitude is an artifact of hyperparameterization, not a measure of learning progress.

Conclusion of Scenario A. Across all weighting regimes, the surrogate loss fails to maintain a consistent or monotonic relationship with true policy quality. Its numerical value is therefore an unreliable indicator of performance, even in the absence of importance sampling effects.

## A.2 Scenario B: Multiple optimization steps (ρ<sub>i,t</sub>(θ) ̸= 1)

After the first update, $\pi _ { \theta }$ diverges from $\pi _ { \mathrm { o l d } }$ and importance sampling ratios $\rho _ { i , t } ( \theta ) \neq 1$ appear. While this breaks the exact cancellations observed in Scenario A, it does not restore interpretability.

The loss value now depends on two independent sources of variability: the structural biases induced by the weighting scheme $\tilde { \omega } _ { i , t }$ and stochastic fluctuations of the importance ratios $\rho _ { i , t } ( \boldsymbol { \theta } )$

As a result, changes in the surrogate loss primarily reflect off-policy drift and optimizer dynamics, rather than genuine reward improvement. A decreasing loss does not imply better policies, nor does a stable loss indicate convergence.

## B Derivation of the Gradient for J<sub>GRPO-L</sub>(θ)

This appendix derives the gradient of the GRPO style surrogate objective and makes explicit the token-level structure that later induces sharedprefix biases. For clarity, we derive the gradient in the region where the unclipped term is active; when the clipped branch is active, the gradient through the advantage term is zero (up to boundary measurezero cases).

## B.1 Gradient of the GRPO style objective

Recall the GRPO style objective in Eq. 1.

$$
\begin{array}{c} \mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \mathbb {E} _ {q, \{o _ {i} \}} \Bigg [ \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t} \min \Big (s _ {i, t} (\theta)   A _ {i}, \\ \operatorname{clip} (s _ {i, t} (\theta), 1 - \epsilon_ {\text {low}}, 1 + \epsilon_ {\text {up}})   A _ {i} \Big) - \beta R (\theta) \Bigg ] \end{array}
$$

We define the token-level importance ratio as

$$
s _ {i, t} (\theta) := k _ {i, t} \cdot \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t})\tag{12}
$$

Here, $k _ { i , t }$ is a term that depends on i and t, but is independent of θ. We now apply the gradient and move $\nabla _ { \theta }$ inside expectation and sums. By linearity,

$$
\begin{array}{l} \nabla_ {\theta} \mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \mathbb {E} _ {q, \{o _ {i} \}} \left[ \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t} \nabla_ {\theta} \min (\cdot) - \beta \nabla_ {\theta} R (\theta) \right] \end{array} \tag {13}
$$

The gradient depends on the active branch. When the unclipped term is active,

$$
\nabla_ {\theta} \min (\cdot) = \nabla_ {\theta} (s _ {i, t} (\theta) A _ {i}) = A _ {i} \nabla_ {\theta} s _ {i, t} (\theta)
$$

while when the clipped term is active, its value is constant w.r.t. θ in the interior of the clipped region, hence the advantage gradient is zero (ignoring boundary non-differentiability).

Since $\pi _ { \theta _ { \mathrm { o l d } } }$ does not depend on current $\theta ,$

$$
\begin{array}{r} \nabla_ {\theta} s _ {i, t} (\theta) = \nabla_ {\theta} (k _ {i, t} \cdot \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t})) \\ = k _ {i, t} \cdot \nabla_ {\theta} \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}) \end{array}
$$

Using the log-derivative trick, $\begin{array} { r l } { \nabla _ { \theta } \pi _ { \theta } } & { { } = } \end{array}$ $\pi _ { \boldsymbol { \theta } } \nabla _ { \boldsymbol { \theta } }$ log $\pi _ { \theta } .$ , we get

$$
\begin{array}{r l} & {\nabla_ {\theta} s _ {i, t} (\theta) = [ k _ {i, t} \cdot \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}) ] \cdot} \\ & {\qquad \cdot \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t})} \\ & {\qquad = s _ {i, t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}).} \end{array}\tag{14}
$$

Substituting Eq. 14 into Eq. 13 yields:

$$
\begin{array}{c} \nabla_ {\theta} \mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \mathbb {E} _ {q, \{o _ {i} \}} \Bigg [ \sum_ {i = 1} ^ {G} A _ {i} \sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t}   s _ {i, t} (\theta) \\ \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}) - \beta \nabla_ {\theta} R (\theta) \Bigg ] \end{array}\tag{15}
$$

## B.2 First token issues

We now isolate the gradient contribution on tokens that belong to a prefix shared by multiple completions in the same group. Let $| k |$ be the length of a prefix shared by a subset of $\tilde { G } \leq G$ completions.

Prefix/deviation decomposition. Splitting the inner sum over time gives:

$$
\begin{array}{l} \nabla_ {\theta} \mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \\ \mathbb {E} _ {q, \left\{o _ {i} \right\}} \left[ \sum_ {i = 1} ^ {G} A _ {i} \left(\sum_ {t = 1} ^ {| k |} \alpha_ {i, t} s _ {i, t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid x, y _ {i, <   t}\right) \right. \right. \\ + \sum_ {t = | k | + 1} ^ {| o _ {i} |} \left. \alpha_ {i, t} s _ {i, t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid x, y _ {i, <   t}\right)\right) \\ - \beta \nabla_ {\theta} R (\theta) \Bigg ] \end{array} \tag {16}
$$

For all $t \leq | k |$ and all completions i in the subset that shares the prefix, both $y _ { i , t }$ and its context $y _ { i , < t }$ are identical. Hence,

$$
\begin{array}{c} \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}) = \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}), \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \forall   i \in \{1, \ldots , \tilde {G} \},   t \leq | k | \end{array}\tag{17}
$$

Define the aggregated coefficient

$$
\omega_ {i, t} := \alpha_ {i, t} s _ {i, t} (\theta)\tag{18}
$$

Then the gradient restricted to the shared prefix (denoted $\nabla _ { \boldsymbol { \theta } } \tilde { \mathcal { I } } _ { \mathrm { G R P O - L } } ( \boldsymbol { \theta } ) )$ becomes:

$$
\nabla_ {\theta} \tilde {\mathcal {J}} _ {\mathrm{GRPO-L}} (\theta) = \sum_ {t = 1} ^ {| k |} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \sum_ {i = 1} ^ {\tilde {G}} A _ {i} \omega_ {i, t}
$$

In the following, when it does not change the qualitative argument, we suppress the explicit dependence on t and write $\omega _ { i }$ for simplicity.

## Case 1: Constant token-wise weights

Assume uniform weights over completions: $\omega _ { i } =$ C. Then:

$$
\nabla_ {\theta} \tilde {\mathcal {J}} _ {\mathrm{GRPO-L}} (\theta) = C \sum_ {t = 1} ^ {| k |} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \sum_ {i = 1} ^ {\tilde {G}} A _ {i}
$$

Since $A _ { i } = R _ { i } - { \bar { R } }$ is group-centered, the behavior depends on which completions share the prefix: (i) if the prefix occurs only in $A _ { i } > 0$ completions, it is reinforced; (ii) in mixed regimes, the net update is the algebraic sum; (iii) if the prefix is ubiquitous across all G completions, $\textstyle \sum _ { i = 1 } ^ { G } A _ { i } = 0$ and the update cancels.

## Case 2: Non-uniform weighting over i

If weights depend on the completion index, $\omega _ { i } \neq $ const, then:

$$
\nabla_ {\theta} \tilde {\mathcal {J}} _ {\mathrm{GRPO-L}} (\theta) = \sum_ {t = 1} ^ {| k |} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \sum_ {i = 1} ^ {\tilde {G}} \omega_ {i} A _ {i}
$$

In this regime, cancellations generally do not hold: shared-prefix tokens can receive a net update dominated by the completions with larger $\omega _ { i }$ which can induce systematic biases unrelated to semantic quality (e.g., length preferences when $\omega _ { i }$ depends on $| o _ { i } | )$

## C Reward magnitude and Adam

This section analyzes how scaling the reward signal affects GRPO style training when optimization is performed with Adam/AdamW. We first show that group-centered advantages scale linearly with the reward. We then propagate this scaling through (i) the GRPO style gradient, (ii) Adam’s first and second moments, and (iii) the final parameter update. The key takeaway is that, when the regularization term is absent (or negligible), Adam is approximately invariant to global reward scaling.

## C.1 Scaling properties of the advantage term

We start by characterizing how the GRPO style advantage behaves under a linear transformation of the reward.

Proposition 5 (Advantage scaling). Let the group-centered advantage be $A _ { i } ~ = ~ R _ { i } -$ $\textstyle { \frac { 1 } { G } } \sum _ { j = 1 } ^ { G } R _ { j }$ . If rewards are scaled by a constant $\mathbf { \bar { \phi } } \phi \in \mathbb { R } , R _ { i } ^ { * } = \phi R _ { i }$ , then the transformed advantage satisfies

$$
A _ {i} ^ {*} = \phi A _ {i}\tag{19}
$$

Proof. First compute the transformed group baseline:

$$
\bar {R} ^ {*} = \frac {1}{G} \sum_ {j = 1} ^ {G} R _ {j} ^ {*} = \frac {1}{G} \sum_ {j = 1} ^ {G} \phi R _ {j} = \phi \left(\frac {1}{G} \sum_ {j = 1} ^ {G} R _ {j}\right) = \phi \bar {R}
$$

Then the transformed advantage is

$$
A _ {i} ^ {*} = R _ {i} ^ {*} - \bar {R} ^ {*} = \phi R _ {i} - \phi \bar {R} = \phi (R _ {i} - \bar {R}) = \phi A _ {i} \tag {20}
$$

□

## C.2 Gradient decomposition and scaling properties

We decompose the GRPO style gradient into an advantage-driven term and a regularization term. Using Eq. 15, define:

$$
g _ {A} (\theta) := \sum_ {i = 1} ^ {G} A _ {i} \sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t} s _ {i, t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t})\tag{21}
$$

$$
g _ {R} (\theta) := \beta \nabla_ {\theta} R (\theta)\tag{22}
$$

so that the total gradient is $g ( \theta ) ~ = ~ g _ { A } ( \theta ) ~ -$ $g _ { R } ( \theta )$

By Proposition 5, scaling the rewards by $\phi$ implies $A _ { i } ^ { * } = \phi A _ { i }$ . Therefore the advantage-driven component scales linearly:

$$
\begin{array}{l} g _ {A} ^ {*} (\theta) = \sum_ {i = 1} ^ {G} A _ {i} ^ {*} \sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t} s _ {i, t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}) \\ = \phi \sum_ {i = 1} ^ {G} A _ {i} \sum_ {t = 1} ^ {| o _ {i} |} \alpha_ {i, t} s _ {i, t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t}) = \\ = \phi g _ {A} (\theta) \end{array} \tag {23}
$$

Conversely, $g _ { R } ( \theta )$ is unaffected by reward scaling because it depends only on the regularizer and $\beta .$

## C.3 Adam moments under gradient scaling

We now study how Adam’s moments scale when the gradient is multiplied by $\phi .$ Let $g _ { t }$ be the gradient at optimization step $t ,$ and assume

$$
g _ {t} ^ {*} = \phi g _ {t} \quad \text {   for   all   } t \geq 1
$$

Adam maintains exponential moving averages:

$$
\begin{array}{r} m _ {t} = \beta_ {1} m _ {t - 1} + (1 - \beta_ {1}) g _ {t} \\ v _ {t} = \beta_ {2} v _ {t - 1} + (1 - \beta_ {2}) g _ {t} ^ {2} \end{array}
$$

with bias-corrected versions

$$
\hat {m} _ {t} = \frac {m _ {t}}{1 - \beta_ {1} ^ {t}}, \qquad \hat {v} _ {t} = \frac {v _ {t}}{1 - \beta_ {2} ^ {t}}.
$$

Proposition 6 (Moment scaling). $I f g _ { t } ^ { * } = \phi g _ { t }$ for all t, then for all $t \geq 1 .$

$$
m _ {t} ^ {*} = \phi m _ {t}, \qquad v _ {t} ^ {*} = \phi^ {2} v _ {t},\tag{24}
$$

$$
a n d e q u i v a l e n t l y \hat {m} _ {t} ^ {*} = \phi \hat {m} _ {t} a n d \hat {v} _ {t} ^ {*} = \phi^ {2} \hat {v} _ {t}.
$$

Proof. We prove by induction.

Base case $( t = 1 )$ . With $m _ { 0 } = v _ { 0 } = 0$

$$
\begin{array}{c} {m _ {1} ^ {*} = (1 - \beta_ {1}) g _ {1} ^ {*} = (1 - \beta_ {1}) \phi g _ {1} = \phi m _ {1}} \\ {v _ {1} ^ {*} = (1 - \beta_ {2}) (g _ {1} ^ {*}) ^ {2} = (1 - \beta_ {2}) \phi^ {2} g _ {1} ^ {2} = \phi^ {2} v _ {1}} \end{array}
$$

Inductive step. Assume $m _ { t - 1 } ^ { * } ~ = ~ \phi m _ { t - 1 }$ and $v _ { t - 1 } ^ { * } = \phi ^ { 2 } v _ { t - 1 }$ . Then

$$
\begin{array}{r l} & {m _ {t} ^ {*} = \beta_ {1} m _ {t - 1} ^ {*} + (1 - \beta_ {1}) g _ {t} ^ {*}} \\ & {\quad = \beta_ {1} (\phi m _ {t - 1}) + (1 - \beta_ {1}) (\phi g _ {t}) =} \\ & {\quad = \phi \big (\beta_ {1} m _ {t - 1} + (1 - \beta_ {1}) g _ {t} \big) = \phi m _ {t}} \end{array}\tag{25}
$$

$$
\begin{array}{r l} & v _ {t} ^ {*} = \beta_ {2} v _ {t - 1} ^ {*} + (1 - \beta_ {2}) (g _ {t} ^ {*}) ^ {2} \\ & \quad = \beta_ {2} (\phi^ {2} v _ {t - 1}) + (1 - \beta_ {2}) \phi^ {2} g _ {t} ^ {2} = \\ & \quad = \phi^ {2} \big (\beta_ {2} v _ {t - 1} + (1 - \beta_ {2}) g _ {t} ^ {2} \big) = \phi^ {2} v _ {t} \end{array}\tag{26}
$$

Bias correction divides by $( 1 - \beta _ { 1 } ^ { t } )$ and $( 1 - \beta _ { 2 } ^ { t } )$ hence it preserves the same scaling. □

## C.4 Adam update invariance under reward scaling

We now analyze when Adam becomes invariant to global reward scaling. Assume the regularization term is absent or negligible, i.e., $g _ { R } ( \theta ) \approx 0$ . Then $g _ { t }$ is driven only by the advantage term and scales as $g _ { t } ^ { * } = \phi g _ { t }$

AdamW updates parameters as

$$
\Delta \theta_ {t} = - \xi \frac {\hat {m} _ {t}}{\sqrt {\hat {v} _ {t}} + \epsilon} - \xi \lambda \theta_ {t - 1}
$$

where $\xi$ is the learning rate, ϵ the numerical stabilizer, and λ the weight decay coefficient.

Using Proposition 6, we have $\hat { m } _ { t } ^ { * } = \phi \hat { m } _ { t }$ and $\hat { v } _ { t } ^ { * } = \phi ^ { 2 } \hat { v } _ { t }$ , hence

$$
\begin{array}{r} \Delta \theta_ {t} ^ {*} = - \xi \frac {\hat {m} _ {t} ^ {*}}{\sqrt {\hat {v} _ {t} ^ {*}} + \epsilon} - \xi \lambda \theta_ {t - 1} \\ = - \xi \frac {\phi \hat {m} _ {t}}{\sqrt {\phi^ {2} \hat {v} _ {t}} + \epsilon} - \xi \lambda \theta_ {t - 1} = \\ = - \xi \frac {\phi \hat {m} _ {t}}{\phi \sqrt {\hat {v} _ {t}} + \epsilon} - \xi \lambda \theta_ {t - 1} \end{array}
$$

Factor $\phi$ out of the denominator (assuming $\phi >$ 0):

$$
\Delta \theta_ {t} ^ {*} = - \xi \frac {\hat {m} _ {t}}{\sqrt {\hat {v} _ {t}} \left(1 + \frac {\epsilon}{\phi \sqrt {\hat {v} _ {t}}}\right)} - \xi \lambda \theta_ {t - 1}
$$

Therefore, in the regime where $\epsilon \ll \phi \sqrt { \hat { v } _ { t } } .$ , we obtain the approximate invariance:

$$
\lim _ {\frac {\epsilon}{\phi \sqrt {\hat {v} _ {t}}} \to 0} \Delta \theta_ {t} ^ {*} = - \xi \frac {\hat {m} _ {t}}{\sqrt {\hat {v} _ {t}}} - \xi \lambda \theta_ {t - 1} = \Delta \theta_ {t}.\tag{27}
$$

This shows that when the optimization signal is purely reward-driven, Adam’s adaptive normalization cancels global reward scaling. However, if a regularization term is present $( \beta \neq 0 )$ , then the total gradient becomes $g _ { t } = g _ { A , t } - g _ { R , t }$ and scaling the rewards changes the relative strength between the two components, breaking invariance.

## D Adam overly moves your model

This section analyzes the interaction between GRPO style clipping and Adam’s momentum. The key point is that clipping can zero out the instantaneous advantage gradient once the policy ratio exits the trust region, but Adam’s first-moment accumulator can continue to move parameters in the same direction, causing overshoot into the clipped region.

## D.1 Gradient discontinuity induced by clipping

Let $\mathcal { R } _ { \mathrm { c l i p } }$ denote the subset of parameter space where the importance ratio exceeds the clip bounds in the direction favored by $A _ { i } ( \mathrm { e . g . } , s _ { i , t } > 1 + \epsilon _ { \mathrm { u p } }$ with $A _ { i } > 0$ , or $s _ { i , t } < 1 - \epsilon _ { \mathrm { l o w } }$ with $A _ { i } < 0 )$ . Inside this region, the advantage term is clipped and its gradient is zero.

Equivalently, the gradient takes the piecewise form:

$$
\nabla_ {\theta} \mathcal {J} _ {\mathrm{GRPO-L}} (\theta) = \left\{ \begin{array}{l l} \nabla_ {\theta} \mathcal {J} _ {\mathrm{ADV}} (\theta) - \beta \nabla_ {\theta} R (\theta), & \theta \notin \mathcal {R} _ {\mathrm{clip}}, \\ - \beta \nabla_ {\theta} R (\theta), & \theta \in \mathcal {R} _ {\mathrm{clip}}. \end{array} \right.\tag{28}
$$

Intuitively, if $\beta$ is small, entering $\mathcal { R } _ { \mathrm { c l i p } }$ should dramatically reduce the gradient magnitude and stop motion in that direction. The next subsection shows why Adam can violate this intuition.

## D.2 Proposition: momentum overshoot

Proposition 7 (Momentum overshoot). Let $\theta _ { T }$ be a parameter iterate lying on the boundary of the clipped region. Assume thatfor $t > T$ the advantage gradient becomes zero due to clipping, i.e., $g _ { A , t } = 0 .$ Then, even if the instantaneous advantage gradient remains zero for subsequent inner-loop steps, Adam can continue to update parameters in the same direction, pushing the iterate deeper into $\mathcal { R } _ { c l i p }$

Proof. For steps $t \ < \ T$ , assume the advantage gradient points consistently toward the upper clip boundary, i.e., $g _ { A , t }$ has a persistent sign that increases $s _ { i , t } ( \theta )$ . Adam accumulates these gradients in the first moment:

$$
m _ {t} = \beta_ {1} m _ {t - 1} + (1 - \beta_ {1}) g _ {t}
$$

At $t = T$ , the iterate enters $\mathcal { R } _ { \mathrm { c l i p } }$ and the advantage gradient is suppressed: $g _ { A , T } ~ = ~ 0$ (and similarly for all $t > T )$ . Neglecting regularization for exposition, the new first-moment update becomes:

$$
\begin{array}{c} m _ {T} = \beta_ {1} m _ {T - 1} + (1 - \beta_ {1}) \underbrace {g _ {T}} _ {= 0} = \beta_ {1} m _ {T - 1} \\ m _ {T + k} = \beta_ {1} ^ {k + 1} m _ {T - 1}, \qquad k \geq 0. \end{array}
$$

Thus, even though the instantaneous gradient is zero, $m _ { T + k }$ remains non-zero for many steps when $\beta _ { 1 }$ is close to one $( \mathbf { e . g . } , \beta _ { 1 } = 0 . 9 )$ . Since the Adam update depends on $\hat { m } _ { t }$ , the parameter update remains non-zero:

$$
\Delta \theta_ {T + k} = - \xi \frac {\hat {m} _ {T + k}}{\sqrt {\hat {v} _ {T + k}} + \epsilon} - \xi \lambda \theta_ {T + k - 1}\tag{29}
$$

Therefore, the iterate continues to move in the direction encoded by the pre-clipping momentum, pushing the ratio further beyond the clip boundary. Clipping acts as a “hard stop” for the instantaneous gradient, but Adam’s momentum makes it a “soft brake” for the parameter trajectory. □

Practical implication. When multiple optimization steps are applied on the same sampled group (inner loop), the overshoot effect becomes more pronounced: the policy can drift further into the clipped region before new samples are generated, weakening the intended trust-region interpretation of clipping.

Quantifying overshoot (Adam canonical form). We now quantify how large the Adam step can remain after entering the clipped region, even when the instantaneous advantage gradient becomes zero.

Assume that at step $T$ the iterate enters $\mathcal { R } _ { \mathrm { c l i p } }$ , so that the advantage gradient is suppressed for all subsequent inner-loop steps, $\mathrm { i . e . , } g _ { A , t } = 0$ for $t \geq T$ For clarity, we first ignore weight decay and regularization and focus on the Adam preconditioned direction $\hat { m } _ { t } / ( \sqrt { \hat { v } _ { t } } + \epsilon )$

Under $g _ { T } = 0$ , Adam moment recurrences reduce to pure exponential decay:

$$
\begin{array}{l} m _ {T} = \beta_ {1} m _ {T - 1} + (1 - \beta_ {1}) \underbrace {g _ {T}} _ {0} = \beta_ {1} m _ {T - 1} \\ v _ {T} = \beta_ {2} v _ {T - 1} + (1 - \beta_ {2}) \underbrace {g _ {T} ^ {2}} _ {0} = \beta_ {2} v _ {T - 1} \end{array}
$$

Using bias correction,

$$
\hat {m} _ {T} = \frac {m _ {T}}{1 - \beta_ {1} ^ {T}} = \frac {\beta_ {1} m _ {T - 1}}{1 - \beta_ {1} ^ {T}}, \quad \hat {v} _ {T} = \frac {v _ {T}}{1 - \beta_ {2} ^ {T}} = \frac {\beta_ {2} v _ {T - 1}}{1 - \beta_ {2} ^ {T}}\tag{30}
$$

Similarly,

$$
\hat {m} _ {T - 1} = \frac {m _ {T - 1}}{1 - \beta_ {1} ^ {T - 1}}, \quad \hat {v} _ {T - 1} = \frac {v _ {T - 1}}{1 - \beta_ {2} ^ {T - 1}}
$$

A C -like coefficient. Define the ratio between the (magnitude of the) preconditioned update immediately after clipping and the one immediately before clipping:

$$
C _ {T} := \frac {\left\| \frac {\hat {m} _ {T}}{\sqrt {\hat {v} _ {T}} + \epsilon} \right\|}{\left\| \frac {\hat {m} _ {T - 1}}{\sqrt {\hat {v} _ {T - 1}} + \epsilon} \right\|}.\tag{31}
$$

In the common regime where $\epsilon \ll \sqrt { \hat { v } _ { T - 1 } }$ and $\epsilon \ll \sqrt { \hat { v } _ { T } }$ , we can approximate

$$
C _ {T} \approx \frac {\left\| \frac {\hat {m} _ {T}}{\sqrt {\hat {v} _ {T}}} \right\|}{\left\| \frac {\hat {m} _ {T - 1}}{\sqrt {\hat {v} _ {T - 1}}} \right\|} = \frac {\| \hat {m} _ {T} \|}{\| \hat {m} _ {T - 1} \|} \cdot \frac {\sqrt {\hat {v} _ {T - 1}}}{\sqrt {\hat {v} _ {T}}}
$$

Substituting Eq. 30 yields

$$
C _ {T} \approx \left[ \beta_ {1} \frac {1 - \beta_ {1} ^ {T - 1}}{1 - \beta_ {1} ^ {T}} \right] \cdot \left[ \sqrt {\frac {1}{\beta_ {2}} \frac {1 - \beta_ {2} ^ {T}}{1 - \beta_ {2} ^ {T - 1}}} \right].\tag{32}
$$

This coefficient captures how much “inertia” remains exactly at the first step after the advantage gradient is clipped out. In the limit $T \to \infty$ , biascorrection saturates and we obtain:

$$
\lim _ {T \to \infty} C _ {T} = \frac {\beta_ {1}}{\sqrt {\beta_ {2}}}.\tag{33}
$$

For typical values $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ ,

$$
\frac {\beta_ {1}}{\sqrt {\beta_ {2}}} = \frac {0 . 9}{\sqrt {0 . 9 5}} \approx 0. 9 2 3
$$

meaning that the first post-clipping step can still be on the order of $\sim 9 2 \%$ of the previous preconditioned step once training is past the early biascorrection transient.

Overshoot across k inner-loop steps. The same reasoning extends to subsequent clipped steps. For $k \geq 0$ , when $g _ { T + k } = 0$ we have

$$
m _ {T + k} = \beta_ {1} ^ {k + 1} m _ {T - 1}, \quad v _ {T + k} = \beta_ {2} ^ {k + 1} v _ {T - 1}.\tag{34}
$$

Define an extension of Eq. 31:

$$
C _ {T, k} := \frac {\left\| \frac {\hat {m} _ {T + k}}{\sqrt {\hat {v} _ {T + k}} + \epsilon} \right\|}{\left\| \frac {\hat {m} _ {T - 1}}{\sqrt {\hat {v} _ {T - 1}} + \epsilon} \right\|}.\tag{35}
$$

Again for ϵ negligible, we obtain the closed form

$$
C _ {T, k} \approx \left[ \beta_ {1} ^ {k + 1} \frac {1 - \beta_ {1} ^ {T - 1}}{1 - \beta_ {1} ^ {T + k}} \right] \cdot \left[ \sqrt {\frac {1}{\beta_ {2} ^ {k + 1}} \frac {1 - \beta_ {2} ^ {T + k}}{1 - \beta_ {2} ^ {T - 1}}} \right].\tag{36}
$$

For large $T$ (where bias correction is stable), Eq. 36 simplifies to an exponential decay:

$$
C _ {T, k} \approx \left(\frac {\beta_ {1}}{\sqrt {\beta_ {2}}}\right) ^ {k + 1}.\tag{37}
$$

With $( \beta _ { 1 } , \beta _ { 2 } ) ~ = ~ ( 0 . 9 , 0 . 9 5 )$ this gives $C _ { T , 4 }$ ≈ $0 . 9 2 3 ^ { 5 } \approx 0 . 6 6 ,$ , i.e., even after five clipped innerloop steps the update magnitude can still be around $\sim 6 6 \%$ of the pre-clipping step, which explains why the policy can drift substantially deeper into the clipped region before new samples are generated.

Effect of ϵ. When ϵ is not negligible (e.g., for very small $\hat { v } _ { t } )$ , the ratios in Eq. 32–36 are further modulated by

$$
\frac {\sqrt {\hat {v} _ {T - 1}} + \epsilon}{\sqrt {\hat {v} _ {T + k}} + \epsilon},\tag{38}
$$

which can either dampen or amplify the residual step depending on the scale of $\hat { v } _ { t }$ relative to ϵ.
