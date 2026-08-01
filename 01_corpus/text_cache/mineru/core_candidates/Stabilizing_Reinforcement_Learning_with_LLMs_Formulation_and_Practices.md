# Stabilizing Reinforcement Learning with LLMs: Formulation and Practices

Chujie Zheng\* Kai Dang Bowen Yu\* Mingze Li Huiqiang Jiang Junrong Lin Yuqiong Liu Hao Lin Chencan Wu Feng Hu An Yang Jingren Zhou Junyang Lin Qwen Team, Alibaba Inc.

## Abstract

This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training-inference discrepancy and policy staleness are minimized. This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models. Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability. When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness. Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization. We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.

## 1 Introduction

Reinforcement learning (RL) has become a key technical paradigm for enhancing large language models' (LLMs) ability to tackle complex problem-solving tasks (OpenAI, 2024; Guo et al., 2025; Yang et al., 2025), while a stable training process $^{1}$ is crucial for successfully scaling RL. Due to the contextual nature of language, RL with LLMs usually employs sequence-level rewards, i.e., a scalar score assigned based on the complete model response. However, mainstream RL algorithms, such as REINFORCE and GRPO, typically employ token-level optimization objectives. This mismatch between the reward (assigned at the sequence level) and the optimization unit (typically at the token level) raises concerns about the soundness and training stability of such approaches, while some studies have proposed directly adopting sequence-level optimization objectives (Zheng et al., 2025; Liu et al., 2025a). In particular, token-level optimization objectives also pose unique challenges for RL training with Mixture-of-Experts (MoE) models. For instance, the dynamic expert routing mechanism can invalidate the token-level importance sampling ratios in MoE models (Zheng et al., 2025). However, it remains unclear whether optimizing sequence-level rewards using token-level objectives is justified, and if so, to what extent (or under what conditions) such an approach is valid.

In this paper, we propose a novel formulation for RL with LLMs. The key insight is that, to optimize the expected sequence-level reward, we can employ a surrogate token-level objective as its first-order approximation. Specifically, this approximation is likely to hold only when both (1) the numerical discrepancy between the training and inference engines (i.e., the training–inference discrepancy) and (2) the discrepancy between the rollout policy that samples responses and the target policy to be optimized (i.e., policy staleness) are minimized. This insight provides a principled explanation of how several techniques for stabilizing RL training work. For example, (1) the importance sampling weight is an inherent component of the surrogate token-level objective under the first-order approximation; (2) the clipping mechanism can restrain policy staleness by preventing aggressive policy updates; (3) for MoE models, the Routing Replay approach (Zheng et al., 2025; Ma et al., 2025), which fixes the routed experts during policy optimization, can reduce both the training–inference discrepancy and policy staleness.

To empirically validate our insight and investigate practical recipes for stable RL training, we conduct extensive experiments with a 30B MoE model, amounting to hundreds of thousands of GPU hours. Our main conclusions include: (1) For on-policy training $^{2}$ , the basic policy gradient algorithm with importance sampling correction yields the highest training stability; (2) When off-policy updates are introduced to accelerate convergence, i.e., a large batch of responses is split into mini-batches for multiple gradient updates, combining clipping and Routing Replay becomes necessary to mitigate instability caused by policy staleness; (3) Once training is stabilized, models with different cold-start initializations consistently achieve comparable final performance. This motivates future work to focus more on RL itself rather than overly on the specifics of cold-start initialization, as differences arising from the latter are expected to vanish given prolonged RL training.

In summary, this paper makes contributions along two axes:

\- Theoretically, we propose a novel formulation for reinforcement learning with LLMs, revealing the conditions under which optimizing sequence-level rewards via token-level objectives is justified. Specifically, the validity of the underlying first-order approximation hinges on jointly minimizing the training–inference discrepancy and policy staleness.

\- Empirically, through extensive experiments with MoE models spanning hundreds of thousands of GPU hours, we demonstrate that several techniques that preserve the validity of the first-order approximation consistently exhibit practical efficacy in stabilizing RL training, particularly the Routing Replay approach tailored for MoE models. We hope that the developed recipes for stable RL training will facilitate future research.

## 2 Formulation for Reinforcement Learning with LLMs

## 2.1 Notation

We define an autoregressive LLM parameterized by $\theta$ as a policy $\pi_{\theta}$ . We use x to denote an input prompt and D as the prompt set. Under the policy $\pi_{\theta}$ , the likelihood of a response y to a prompt x is denoted as $\pi_{\theta}(y|x)=\prod_{t=1}^{|y|}\pi_{\theta}(y_{t}|x,y_{<t})$ where $|y|$ is the number of tokens in y. Given the contextual nature of language, we focus on the sequence-level reward setting, where a whole response y is assigned a single scalar reward $R(x,y)$ . We do not consider the value-based setting (e.g., PPO, Schulman et al. 2017), where policy optimization is steered by a value model that assigns scalar scores to each token in a response y. This is because we found it inherently difficult (if not impossible) to devise general and scalable approaches to obtaining reliable value models.

## 2.2 Expected Sequence-level Reward is Hard to Directly Optimize

Our formulation starts from the true sequence-level reward that we aim to maximize:

$$
\mathcal {J} ^ {\mathrm{seq}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\theta} (\cdot | x)} [ R (x, y) ],
$$

where $\pi_{\theta}$ is the target policy to be optimized. Since the responses are typically not sampled in the training engine (e.g., Megatron and FSDP) but instead in the inference engine (e.g., SGLang and vLLM), we adopt the importance sampling (IS) trick to do a simple transformation:

$$
\mathcal {J} ^ {\mathrm{seq}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\theta} (\cdot | x)} \left[ R (x, y) \right] = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \underbrace {\frac {\pi_ {\theta} (y | x)}{\mu_ {\theta_ {\mathrm{old}}} (y | x)}} _ {\text {sequence - level IS weight}} R (x, y) \right],\tag{1}
$$

where $\mu_{\theta_{old}}$ denotes the rollout policy that samples responses. Note that we use the notation $\mu$ to distinguish the policy in the inference engine from the policy (notated as $\pi$ ) in the training engine, as there typically exists a numerical discrepancy between training and inference engines (Yao et al., 2025). The sequence-level objective in Equation (1) has the following gradient:

$$
\begin{array}{c} \nabla_ {\theta} \mathcal {J} ^ {\mathrm{seq}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {\pi_ {\theta} (y | x)}{\mu_ {\theta_ {\mathrm{old}}} (y | x)} R (x, y) \nabla_ {\theta} \log \pi_ {\theta} (y | x) \right] \\ = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {\pi_ {\theta} (y | x)}{\mu_ {\theta_ {\mathrm{old}}} (y | x)} R (x, y) \sum_ {t = 1} ^ {| y |} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right]. \end{array}\tag{2}
$$

However, this gradient is usually intractable to utilize due to the large numerical range and high variance of sequence likelihood (i.e., $\pi_{\theta}(y|x)$ and $\mu_{\theta_{\mathrm{old}}}(y|x)$ ), making it difficult to directly optimize the sequence-level objective in Equation (1).

## 2.3 Token-level Objective as a First-order Approximation to Sequence-level Objective

The critical step in our formulation is to consider the following surrogate token-level objective:

$$
\mathcal {J} ^ {\mathrm{token}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| y |} \underbrace {\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})}} _ {\text {token - level IS weight}} R (x, y) \right],\tag{3}
$$

with the following gradient:

$$
\nabla_ {\theta} \mathcal {J} ^ {\mathrm{token}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| y |} \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})} R (x, y) \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right].\tag{4}
$$

which is actually the basic policy gradient algorithm (i.e., REINFORCE) equipped with a token-level IS weight. The core insight here is that we can view the token-level optimization objective in Equation (3) as a first-order approximation to the sequence-level objective in Equation (1) that we truly aim to optimize. To be specific, suppose $\pi_{\theta}$ and $\mu_{\theta_{old}}$ are slightly different, let $\frac{\pi_{\theta}(y_{t}|x,y_{<t})}{\mu_{\theta_{old}}(y_{t}|x,y_{<t})}=1+\delta_{t}$ where $\delta_{t}$ is a small quantity. We have the following approximation:

$$
\frac {\pi_ {\theta} (y | x)}{\mu_ {\theta_ {\mathrm{old}}} (y | x)} = \prod_ {t = 1} ^ {| y |} (1 + \delta_ {t}) \approx 1 + \sum_ {t = 1} ^ {| y |} \delta_ {t} + O (\delta^ {2}) \approx 1 + \sum_ {t = 1} ^ {| y |} \delta_ {t},
$$

where the rightmost derivation neglects second-order and higher-order small terms like $\delta_{i}\delta_{j}$ . So we have:

$$
\begin{array}{r l} & {\nabla_ {\theta} \mathcal {J} ^ {\mathrm{seq}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ R (x, y) \nabla_ {\theta} \left(\frac {\pi_ {\theta} (y | x)}{\mu_ {\theta_ {\mathrm{old}}} (y | x)}\right) \right]} \\ & {\qquad \approx \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ R (x, y) \nabla_ {\theta} \left(1 + \sum_ {t = 1} ^ {| y |} \delta_ {t}\right) \right]} \\ & {\qquad = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ R (x, y) \nabla_ {\theta} \left(\sum_ {t = 1} ^ {| y |} \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})}\right) \right]} \\ & {\qquad = \nabla_ {\theta} \mathcal {J} ^ {\mathrm{token}} (\theta).} \end{array}
$$

This is why we say that Equation (3) is a first-order approximation to Equation (1). Therefore, when $\pi_{\theta}$ is close to $\mu_{\theta_{old}}$ , we can improve the sequence-level objective in Equation (1) by updating the model parameters $\theta$ with the gradient in Equation (4).

## 2.4 Conditions for First-order Approximation to Hold

For the first-order approximation to hold, we require that the target policy $\pi_{\theta}$ and the rollout policy $\mu_{\theta_{old}}$ are close, which, however, is less intuitive. To be clear, given x and for each token $y_{t}$ , we rewrite its IS weight as:

$$
\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\text { old }}} (y _ {t} | x , y _ {<   t})} = \underbrace {\frac {\pi_ {\theta_ {\text { old }}} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\text { old }}} (y _ {t} | x , y _ {<   t})}} _ {\text { training - inference discrepancy }} \times \underbrace {\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\pi_ {\theta_ {\text { old }}} (y _ {t} | x , y _ {<   t})}} _ {\text { policy   staleness }},\tag{5}
$$

where $\pi_{\theta_{old}}$ denotes the rollout policy computed by the training engine, differing from the one $\mu_{\theta_{old}}$ in the inference engine. Therefore, from the decomposition in Equation (5), the gap between $\pi_{\theta}$ and $\mu_{\theta_{old}}$ comes from two aspects: the training–inference discrepancy and policy staleness.

\- Regarding the training–inference discrepancy—i.e., the numerical differences between training and inference engines—the causes are usually complex and heavily tied to the underlying infrastructure. For example, training and inference engines typically employ different computational kernels for peak performance, which would yield inconsistent outputs given the same model input. Even within a single engine, particularly the inference side, batch-invariant kernels (He and Lab, 2025) are often disabled for maximizing throughput, so the same model input can still receive variant outputs. In the case of MoE models, the training–inference discrepancy is further amplified by inconsistent expert routing, which we will discuss detailedly in § 3.

\- Regarding policy staleness—i.e., the discrepancy between the rollout policy that samples responses and the target policy to be optimized—it usually arises from the trade-offs made to improve training efficiency and computational utilization. Since the rollout stage in RL is typically bounded in time by the generation length, to accelerate convergence through increased computational resources, we often split a large batch of sampled responses into mini-batches for multiple gradient updates. Consequently, mini-batches consumed later may exhibit greater policy staleness. In asynchronous RL frameworks, a single response can be generated sequentially by multiple model versions, which also introduces policy staleness.

Therefore, to ensure the validity of the first-order approximation that underlies the surrogate token-level objective in Equation (3), we should, in principle, narrow the gap between $\pi_{\theta}$ and $\mu_{\theta_{old}}$ from two directions: reducing the numerical discrepancy between training and inference engines, and controlling policy staleness within a moderate range.

## 3 Challenge for Mixture of Experts, and Routing Replay

## 3.1 Expert Routing Hinders First-order Approximation to Hold

When it comes to Mixture-of-Experts (MoE) models (Guo et al., 2025; Yang et al., 2025), the conditions for the first-order approximation to hold become less straightforward. Specifically, during the forward pass of generating each token, MoE models dynamically select and activate only a small subset of expert parameters via the expert routing mechanism. Incorporating expert routing into Equation (5), we can write the token-level IS weight for an MoE model as:

$$
\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})} = \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t} , e _ {t} ^ {\pi})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})} = \underbrace {\frac {\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\pi})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})}} _ {\text {training - inference discrepancy}} \times \underbrace {\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t} , e _ {t} ^ {\pi})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\pi})}} _ {\text {policy staleness}},\tag{6}
$$

where $e^{\pi}$ and $e^{\mu}$ denote the routed experts in the training and inference engines, respectively, and the subscript “old” corresponds to the rollout policy.

At this point, the challenge of reinforcement learning with MoE models becomes clear: expert routing is entangled with the training–inference discrepancy and policy staleness, increasing the likelihood that the first-order approximation underlying the surrogate token-level optimization objective in Equation (3) breaks down. More specifically, the training–inference discrepancy can cause inconsistent routed experts in the training and inference engines (i.e., $e_{old,t}^{\pi}$ versus $e_{old,t}^{\mu}$ ) given the same model parameters and input. This divergence in expert routing, in turn, further amplifies the discrepancy in final outputs. Furthermore, policy staleness manifests not only in changes in the model parameters (i.e., $\theta$ versus $\theta_{old}$ ) but also in shifts of routed experts (i.e., $e_{t}^{\pi}$ versus $e_{old,t}^{\pi}$ ), which can heavily alter the resulting policy defined by activated parameters.

## 3.2 Routing Replay Restores First-order Approximation, Yet May Introduce Bias

Identifying that expert routing undermines the validity of the first-order approximation in MoE models, we can eliminate this impact through the Routing Replay (Zheng et al., 2025) approach. The core idea of Routing Replay is to stabilize RL training of MoE models by fixing the routed experts during policy optimization, thereby enabling the model to be optimized like a dense one. Upon Equation (6), we formalize the following two concrete implementations of Routing Replay, namely Vanilla Routing Replay and Rollout Routing Replay:

\- Vanilla Routing Replay (R2) (Zheng et al., 2025) focuses on mitigating the impact of expert routing on policy staleness by replaying, during gradient updates, the routed experts determined by the rollout policy in the training engine (i.e., $e_{\mathrm{old},t}^{\pi}$ ):

$$
\frac {\pi_ {\theta} ^ {\mathrm{R2}} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})} = \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\pi})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})} = \frac {\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\pi})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})} \times \underbrace {\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\pi})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\pi})}} _ {\text {policy staleness\downarrow}}.
$$

\- Rollout Routing Replay (R3) (Ma et al., 2025) aims to reduce the impact of expert routing on the training-inference discrepancy by uniformly replaying, within the training engine, the routed experts determined by the rollout policy in the inference engine (i.e., $e_{\mathrm{old},t}^{\mu}$ ), which also simultaneously mitigates the impact of expert routing on policy staleness:

$$
\frac {\pi_ {\theta} ^ {\mathrm{R3}} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})} = \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})} = \underbrace {\frac {\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})}} _ {\text {training - inference discrepancy } \downarrow} \times \underbrace {\frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t} , e _ {\mathrm{old} , t} ^ {\mu})}} _ {\text {policy staleness} \downarrow}.
$$

Therefore, Routing Replay intuitively restores the validity of the first-order approximation in MoE models by reducing the training–inference discrepancy (in R3) and alleviating policy staleness (in R2 and R3). However, we point out that it also implicitly biases the target policy, as suggested by the notations $\pi_{\theta}^{R2}$ and $\pi_{\theta}^{R3}$ . Specifically, the original target policy we aim to optimize in Equation (3) is $\pi_{\theta}$ , where the likelihood of each token $y_{t}$ should be governed by the naturally-routed experts $e_{t}^{\pi}$ . However, Routing Replay constrains the routed experts to be $e_{old,t}^{\pi}$ or $e_{old,t}^{\mu}$ , leading to another target policy $\pi_{\theta}^{R2}$ or $\pi_{\theta}^{R3}$ that deviates from the original $\pi_{\theta}$ defined by $e_{t}^{\pi}$ . In particular, when we split a large batch into mini-batches for multiple gradient updates, R2 and R3 can possess different degrees of bias, as shown in Table 1. The key difference is that R2 does not alter the original target policy in the first mini-batch, which we conjecture may lead to R2 and R3 exhibiting different performance, especially when the ratio between batch size and mini-batch size (i.e., the degree of off-policiness) is varied.

Table 1: Comparison between R2 and R3 in how they alter the original target policy $\pi_{\theta}$ .

<table><tr><td></td><td>First mini-batch</td><td>Non-first mini-batch</td></tr><tr><td>R2(replaying  $e^{\pi}_{old,t}$ )</td><td> $e^{\pi}_{old,t}=e^{\pi}_{t}$ ,target policy is not altered</td><td> $e^{\pi}_{old,t}\neq e^{\pi}_{t}$ ,target policy is altered</td></tr><tr><td>R3(replaying  $e^{\mu}_{old,t}$ )</td><td> $e^{\mu}_{old,t}\neq e^{\pi}_{t}$ ,target policy is altered</td><td> $e^{\mu}_{old,t}\neq e^{\pi}_{t}$ ,target policy is altered</td></tr></table>

Nevertheless, it is difficult to definitively assess whether the advantages or disadvantages of Routing Replay outweigh each other. Altering routed experts, while introducing bias into the optimization objective, also makes the first-order approximation—on which the altered token-level objective using $\pi_{\theta}^{R2}$ or $\pi_{\theta}^{R3}$ as the target policy relies—more likely to hold. We need further experiments to validate the practical utility of Routing Replay.

## 4 Empirical Analyses

## 4.1 MiniRL: A Minimalist Baseline Algorithm

In our experiments, we employ two minimal modifications to the REINFORCE optimization objective in Equation (3) as a minimalist baseline algorithm. First, we apply group-normalization (Shao et al., 2024) to the raw rewards as the advantage estimate for each response $y$ : $\widehat{A}(x,y) = R(x,y) - \mathbb{E}_{y' \sim \mu_{\theta_{\mathrm{old}}}(\cdot|x)}[R(x,y')]$ , which also lowers the variance of the raw rewards. Second, we adopt the clipping mechanism in PPO (Schulman et al., 2017) that prevents aggressive policy updates by stopping gradients for certain tokens, which can hopefully restrain policy staleness. We follow the decoupled PPO approach (Hilton et al., 2022) and use $\pi_{\theta_{\mathrm{old}}}$ as the proximal policy to decide whether to clip the token $y_t$ based on the ratio of $\pi_\theta(y_t|x,y_{<t})$ and $\pi_{\theta_{\mathrm{old}}}(y_t|x,y_{<t})^3$ . The obtained minimalist baseline algorithm, which we call MiniRL, is

$^{3}$ While there are alternative clipping strategies, such as clipping a whole response based on the ratio of sequence likelihood (Zheng et al., 2025), we found that the current clipping strategy has worked decently. Therefore, we leave

as follows:

$$
\begin{array}{l} \mathcal {J} _ {\text {MiniRL}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\text {old}}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| y |} M _ {t}   \text {sg} \left[ \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\text {old}}} (y _ {t} | x , y _ {<   t})} \right]   \widehat {A} (x, y)   \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right], \\ M _ {t} = \left\{ \begin{array}{l l} 0 & \text {if} \widehat {A} (x, y) > 0 \text {and} r _ {t} > 1 + \varepsilon_ {\text {high}}, \\ 0 & \text {if} \widehat {A} (x, y) <   0 \text {and} r _ {t} <   1 - \varepsilon_ {\text {low}}, \\ 1 & \text {otherwise}, \end{array} \right. \quad r _ {t} = \frac {\pi_ {\theta} (y _ {t} | x , y _ {<   t})}{\pi_ {\theta_ {\text {old}}} (y _ {t} | x , y _ {<   t})}, \end{array}\tag{7}
$$

where sg denotes the operation of stopping gradient. It is noteworthy that MiniRL is adopted as the baseline algorithm to maintain consistency—as closely as possible—(in gradient) with the surrogate token-level objective in Equation 3, which has been justified by our formulation in § 2. In Appendix A, we will provide a comparison of MiniRL against other algorithms such as GRPO (Shao et al., 2024) and CISPO (Chen et al., 2025). All our experiments will be implemented based on MiniRL.

## 4.2 Experimental Setup

We conduct experiments on the mathematical reasoning task, where the model response is compared with the ground truth answer and then assigned a binary reward (i.e., $R(x,y) \in \{0,1\}$ ). We curate 4,096 math problems with verified answers as the prompt set for RL training. We report the average accuracy over 32 sampled responses on the HMMT25, AIME25, and AIME24 benchmarks, each consisting of 30 competition-level math problems (90 in total).

We experiment with a cold-start model fine-tuned from Qwen3-30B-A3B-Base. We adopt the setting of FP8 inference and BF16 training, providing a stress test for algorithmic correctness where the inference precision is lower than the training and the training–inference discrepancy is large. Besides the training reward, we also report the dynamics of two metrics: (1) the token-level entropy of the target policy, approximated by:

$$
\mathbb {H} \left[ \pi_ {\theta} \right] \approx \mathbb {E} _ {x \sim \mathcal {D}, y _ {<   t} \sim \mu_ {\theta_ {\mathrm{old}} (\cdot | x)}} \left[ - \sum_ {w \in \mathcal {V}} \pi_ {\theta} (w | x, y _ {<   t}) \log \pi_ {\theta} (w | x, y _ {<   t}) \right],
$$

where V denotes the vocabulary, and (2) the KL divergence between the rollout policies in the inference and training engines, calculated as:

$$
\mathbb {D} _ {\mathrm{KL}} \left[ \mu_ {\theta_ {\mathrm{old}}} \| \pi_ {\theta_ {\mathrm{old}}} \right] = \mathbb {E} _ {x \sim \mathcal {D}, y _ {t} \sim \mu_ {\theta_ {\mathrm{old}} (\cdot | x, y _ {<   t})}} \left[ \log \frac {\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})} \right].
$$

We report the latter metric because recent work (Yao et al., 2025; Liu et al., 2025a) has revealed that the instability or collapse in RL training is often accompanied by a sharp increase in the training-inference discrepancy.

To conduct controlled experiments, we employ the standard synchronous RL framework. In each global step, we first sample a batch of B prompts and sample G responses for each prompt using the rollout policy in the inference engine. Then, we split the responses into N mini-batches and apply N gradient updates in the training engine. The finally updated policy in this global step is used as the new rollout policy in the next global step. Across all experimental runs, we use the same mini-batch size of 1,024 responses (B = 64 and G = 16) for each gradient update.

For other hyperparameters, we set the maximum generation length to 32,768, and set $\varepsilon_{high}$ to 0.27 and $\varepsilon_{low}$ to 0.2 in MiniRL. We additionally apply the Truncated Importance Sampling (TIS) trick (Yao et al., 2025) to the token-level IS weight in MiniRL, with the truncation threshold set to 5. Our experiments total hundreds of thousands of GPU hours, and the consumed compute can be estimated as $5 \sim 6$ GPU hours per gradient step.

## 4.3 Results of On-policy Training

We first verify, under on-policy training where the global batch size equals the mini-batch size, whether the validity of the first-order approximation underlying the token-level optimization objective is correlated with training stability. Under this on-policy setting where $\theta = \theta_{old}$ , MiniRL degenerates to the following basic policy gradient algorithm:

$$
\mathcal {J} _ {\mathrm{MiniRL}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| y |} \frac {\pi_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\mathrm{old}}} (y _ {t} | x , y _ {<   t})} \widehat {A} (x, y) \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right],
$$

the study of clipping or masking strategies for future work. Similarly, exploring better advantage estimates $\widehat{A}(x,y)$ may also be helpful, but falls outside the scope of this work.

Training-inference KL Divergence

so the IS weight here serves only as a correction for the training–inference discrepancy. We notice that existing RL algorithms, such as GRPO and CISPO, often employ length normalization in their optimization objectives, and their original objectives do not consider IS correction for the training–inference discrepancy. We thus include the following two ablated variants of MiniRL in our experiments:

$$
\mathcal {J} _ {\text {MiniRL}} ^ {\text {w\_length - norm}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\text {old}}} (\cdot | x)} \left[ \frac {1}{| y |} \sum_ {t = 1} ^ {| y |} \frac {\pi_ {\theta_ {\text {old}}} (y _ {t} | x , y _ {<   t})}{\mu_ {\theta_ {\text {old}}} (y _ {t} | x , y _ {<   t})}   \widehat {A} (x, y)   \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right],
$$

which additionally employs length normalization, and

$$
\mathcal {J} _ {\mathrm{MiniRL}} ^ {\mathrm {wo\_train - infer - is}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| y |} \widehat {A} (x, y) \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right],
$$

which omits the IS correction for the training–inference discrepancy. Note that the two variants have no longer satisfied the aforementioned first-order approximation, as their gradients are neither equal to nor linearly correlated with the gradient of the true sequence-level objective in Equation (1) (ignoring the reward normalization). We also equip MiniRL and the two variants with R3 (R2 is inapplicable here, see Table 1) for comparison.

Benchmark Score  
Training Reward



Entropy  
Figure 1: Results of on-policy training with gbs (global batch size) = mbs (mini-batch size) = 1,024.

From Figure 1, we draw the following observations and conclusions:

\- MiniRL, i.e., the basic policy gradient algorithm with IS correction, achieves the best performance and training stability.

\- Adding length normalization leads to suboptimal performance $^{4}$ , although training remains stable. This is as expected, since length normalization invalidates the first-order approximation to the true expected sequence-level reward, resulting in a biased token-level optimization objective.

\- Removing the training–inference IS correction causes rapid training collapse and a sharp drop in entropy. This confirms that the IS weight is an inherent component of the first-order approximation, and omitting it immediately invalidates the token-level optimization objective.

\- Applying R3 in on-policy training does not yield performance gains, despite effectively reducing the training–inference discrepancy (as reflected by the training–inference KL divergence). Moreover, combining R3 with length normalization even degrades the benchmark score further, and applying R3 without the training-inference IS correction still fails rapidly $^{5}$ . This empirically confirms our speculation in § 3.2—that Routing Replay can alter the original target policy and introduce bias into the optimization objective.

These results demonstrate that, in designing token-level optimization objectives, only those that preserve the validity of the first-order approximation to the expected sequence-level reward lead to improved training stability and performance. This also validates the soundness of our proposed formulation.

## 4.4 Results of Off-policy Training

The inference time in RL is typically bounded by the generation length and cannot be accelerated by increasing computational resources. To leverage increased compute for faster convergence, a common practice is to introduce off-policy updates. Within a synchronous RL framework, this means that a large batch of responses is split into N mini-batches for multiple gradient updates. To investigate the recipes for stable RL training under off-policy settings, we experiment with three levels of off-policiness: with the mini-batch size fixed at 1,024 responses, the global batch size is varied to 2,048, 4,096, and 8,192, corresponding to N = 2, 4, and 8, respectively. With MiniRL as the baseline, we compare the following methods: MiniRL (no clipping), MiniRL + R2 (no clipping), MiniRL + R2, and MiniRL + R3.


Training-inference KL Divergence  


Figure 2: Results of off-policy training with gbs = 2 × mbs = 2,048.

From Figures 2 to 4, we draw the following observations and conclusions:

\- Once off-policy updates are introduced, both Routing Replay and clipping become essential for stable training. As shown in Figures 2 and 3, omitting either Routing Replay or clipping causes training to collapse prematurely, thereby degrading peak performance. This indicates that Routing

Entropy  

Entropy


Figure 3: Results of off-policy training with gbs = 4 × mbs = 4,096.


Training-inference KL Divergence  


Figure 4: Results of off-policy training with gbs = 8 × mbs = 8,192.

Replay alleviates the impact of expert routing, and the clipping mechanism also effectively prevents aggressive policy updates, thereby both restraining policy staleness.

\- When off-policiness is small (gbs = 2 × mbs), R2 outperforms R3, while when off-policiness is large (gbs = 4 × mbs and gbs = 8 × mbs), R3 surpasses R2. Notably, under high off-policiness, R2 fails to sustain stable training, and its peak performance achieved before training collapse is also slightly lower than that of R3. Combining our analysis in § 3.2—particularly that R2 leaves the target policy of the first mini-batch unchanged while R3 alters it—and the on-policy experimental results in § 4.3, we hypothesize that when off-policiness is small, the detrimental impact of R3's alteration to the target policy outweighs its benefit in preserving the validity of the first-order approximation, while under larger off-policiness, the opposite holds true.

In summary, we find that Routing Replay and clipping are necessary for stable off-policy training. When off-policiness is small, R2 is sufficient and more effective at stabilizing RL training for MoE models, whereas R3 becomes necessary under larger off-policiness.

## 4.5 Results of Varying Cold-start Initializations

Recall the motivation for stabilizing RL training: given a base model, once we can reach its performance limit through sufficiently long RL training, we can reliably enhance the model's capabilities by investing computational resources into RL. To this end, we investigate whether models initialized with different cold-start data can achieve similar performance when trained using a stable RL recipe. We compare three versions of cold-start data distilled from three frontier models: Qwen3-Max-Thinking-Preview, DeepSeek-R1-0528, and gpt-oss-120b (high mode). We report results based on an early-experimental small Qwen3Next MoE model, trained with a global batch size of 4,096, a mini-batch size of 2,048 (B = 128, G = 16, N = 2), and a generation length of 65,536 tokens. We employ MiniRL + R2 as the training recipe.


Figure 5: Results of varying cold-start initializations.

In Figure 5, we show that the three cold-start initializations consistently achieve comparable final performance, which encourages us to focus more on RL itself rather than overly on the specifics of cold-start initialization. Furthermore, comparing Figures 1 to 4, we find that both on-policy and off-policy training—once stabilized—also consistently achieve similar peak performance. These results further suggest that stable training plays a decisive role in successfully scaling RL.

## 5 Conclusion

We propose a new formulation for reinforcement learning with LLMs, viewing the token-level optimization objective as a first-order approximation to the true expected sequence-level reward. Through extensive experiments, we demonstrate that techniques that preserve the validity of this first-order approximation—such as importance sampling correction, clipping, and Routing Replay for MoE models—all effectively stabilize RL training. We further investigate recipes for stable RL training across varying degrees of off-policiness and show that, once training is stabilized, the same base model consistently converges to similar performance with prolonged RL. We hope that the insights and empirical results shared in this paper will inspire and facilitate future research.

## References

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Horace He and Thinking Machines Lab. Defeating nondeterminism in llm inference. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20250910. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/.

Jacob Hilton, Karl Cobbe, and John Schulman. Batch size-invariance for policy optimization. In Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=lXuZaxEaI7.

Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystifying RL collapse from the training-inference mismatch, September 2025a. URL https://richardli.xyz/rl-collapse.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, and Fuli Luo. Stabilizing moe reinforcement learning by aligning training and inference routers. arXiv preprint arXiv:2510.11370, 2025.

OpenAI. Learning to reason with LLMs, 2024. URL https://openai.com/index/learning-to-reason-with-llms/.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, August 2025. URL https://fengyao.notion.site/off-policy-rl.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

## A Comparison of MiniRL against GRPO and CISPO

We compare the optimization objective of MiniRL against those of GRPO (Shao et al., 2024) and CISPO (Chen et al., 2025). With the notations in this paper, GRPO employs the following objective:

$$
\begin{array}{l} \mathcal {J} _ {\mathrm{GRPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \\ \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \min \left(r _ {i, t} (\theta) \widehat {A} _ {i, t}, \operatorname{clip} (r _ {i, t} (\theta), 1 - \varepsilon_ {\mathrm{low}}, 1 + \varepsilon_ {\mathrm{high}}) \widehat {A} _ {i, t}\right) \right], \end{array}
$$

and CISPO is as follows:

$$
\begin{array}{l} \mathcal {J} _ {\mathrm{CISPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \mu_ {\theta_ {\mathrm{old}}} (\cdot | x)} \\ \left[ \frac {1}{\sum_ {i = 1} ^ {G} | y _ {i} |} \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| y _ {i} |} \mathrm{sg} \left[ \mathrm{clip} (r _ {i, t} (\theta), 1 - \varepsilon_ {\mathrm{low}}, 1 + \varepsilon_ {\mathrm{high}}) \right] \widehat {A} _ {i, t} \log \pi_ {\theta} (y _ {t} | x, y _ {<   t}) \right], \end{array}
$$

where in both objectives:

$$
r _ {i, t} (\theta) = \frac {\pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i , t} | x , y _ {i , <   t})}, \qquad \widehat {A} _ {i, t} = \frac {R (x , y _ {i}) - \mathrm{mean} \left(\{R (x , y _ {i}) \} _ {i = 1} ^ {G}\right)}{\mathrm{std} \left(\{R (x , y _ {i}) \} _ {i = 1} ^ {G}\right)}.
$$

Their key differences from MiniRL include the following: (1) Their original objectives do not consider the training–inference discrepancy; (2) They both employ length normalization, which we show in § 4.3 invalidates the first-order approximation to the true expected sequence-level reward and can lead to a biased token-level optimization objective and suboptimal performance; (3) CISPO does not clip the gradient of certain tokens, which we show in § 4.4 can result in unstable training.

## B Detailed Benchmark Results



Figure 6: Detailed benchmark results of on-policy training with gbs = mbs = 1,024.



Figure 7: Detailed benchmark results of off-policy training with gbs = 2 × mbs = 2,048.



Figure 8: Detailed benchmark results of off-policy training with gbs = 4 × mbs = 4,096.



Figure 9: Detailed benchmark results of off-policy training with gbs = 8 × mbs = 8,192.




Figure 10: Detailed benchmark results of varying cold-start initializations.
