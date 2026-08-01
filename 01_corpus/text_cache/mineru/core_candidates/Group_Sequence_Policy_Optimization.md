# Group Sequence Policy Optimization

Chujie Zheng<sup>∗</sup> Shixuan Liu Mingze Li Xiong-Hui Chen Bowen Yu<sup>∗</sup> Chang Gao Kai Dang Yuqiong Liu Rui Men An Yang Jingren Zhou Junyang Lin Qwen Team, Alibaba Inc.

## Abstract

This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixtureof-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models.

## 1 Introduction

Reinforcement learning (RL) has emerged as a pivotal paradigm for scaling language models (OpenAI, 2024; DeepSeek-AI, 2025; Qwen, 2025b;a). Through large-scale RL, language models develop the capability to tackle sophisticated problems, such as competition-level mathematics and programming, by undertaking deeper and longer reasoning processes.

To successfully scale RL with greater computational investment, the foremost prerequisite is maintaining stable and robust training dynamics. However, current state-of-the-art RL algorithms, exemplified by GRPO (Shao et al., 2024), exhibit severe stability issues when training gigantic language models, often resulting in catastrophic and irreversible model collapse (Qwen, 2025a; MiniMax, 2025). This instability hinders efforts to push the capability boundaries of language models through continued RL training.

In this paper, we identify that the instability of GRPO stems from the fundamental misapplication and invalidation of importance sampling weights in its algorithmic design. This introduces high-variance training noise that progressively accumulates with increased response length and is further amplified by the clipping mechanism, ultimately precipitating model collapse.

To address these core limitations, we propose Group Sequence Policy Optimization (GSPO), a new RL algorithm for training large language models. The key innovation of GSPO lies in its theoretically grounded definition of importance ratio based on sequence likelihood (Zheng et al., 2023), aligning with the basic principle of importance sampling. Additionally, GSPO computes the normalized rewards as the advantages of multiple responses to a query, ensuring the alignment between sequence-level rewarding and optimization.

Our empirical evaluation demonstrates the significant superiority of GSPO over GRPO in training stability, efficiency, and performance. Critically, GSPO has inherently resolved the stability challenges in the RL training of large Mixture-of-Experts (MoE) models, eliminating the need for complex stabilization strategies, and shows the potential for simplifying RL infrastructure. These merits of GSPO ultimately contributed to the exceptional performance improvements in the latest Qwen3 models. We envision GSPO as a robust and scalable algorithmic foundation that will enable the continued advancement of large-scale RL training with language models.

## 2 Preliminaries

Notation In this paper, an autoregressive language model parameterized by θ is defined as a policy π . We use x to denote a query and D as the query set. Given a response y to a query x, its likelihood under the policy $\pi _ { \theta }$ is denoted as $\begin{array} { r } { \pi _ { \theta } ( y | x ) = \prod _ { t = 1 } ^ { | y | } \pi _ { \theta } ( y _ { t } | x , y _ { < t } ) } \end{array}$ where $| y |$ denotes the number of tokens in y. A query-response pair $( x , y )$ can be scored by a verifier $r ,$ resulting in a reward $r ( x , y ) \in [ 0 , 1 ]$

Proximal Policy Optimization (PPO) Using samples generated from the old policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } . }$ , PPO (Schulman et al., 2017) constrains the policy update within a proximal region of the old policy through the clipping mechanism. Specifically, PPO employs the following objective for policy optimization (we omit the KL regularization term hereinafter for brevity, as it is not the focus of this paper):

$$
\mathcal {J} _ {\mathrm{PPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{| y |} \sum_ {t = 1} ^ {| y |} \min \left(w _ {t} (\theta) \widehat {A} _ {t}, \operatorname{clip} (w _ {t} (\theta), 1 - \varepsilon , 1 + \varepsilon) \widehat {A} _ {t}\right) \right],\tag{1}
$$

where the importance ratio of the token y<sub>t</sub> is defined as $\begin{array} { r } { w _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( y _ { t } | \boldsymbol { x } , \boldsymbol { y } _ { < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { t } | \boldsymbol { x } , \boldsymbol { y } _ { < t } \right) } } \end{array}$ , the advantage $\widehat { A } _ { t }$ of y<sub>t</sub> is estimated by another value model, and ε is the clipping range of importance ratios.

The core challenge of PPO in practice lies in its heavy reliance on the value model. Specifically, the value model usually has a similar size to the policy model, introducing a considerable memory and computational burden. Furthermore, the algorithmic effectiveness hinges on the reliability of its value estimate. While acquiring a reliable value model is inherently challenging, ensuring its scalability to longer responses and more complex tasks presents an even greater challenge.

Group Relative Policy Optimization (GRPO) GRPO (Shao et al., 2024) bypasses the need for the value model by computing the relative advantage of each response within a group of responses to the same query. Specifically, GRPO optimizes the following objective:

$$
\mathcal {J} _ {\mathrm{GRPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \min \left(w _ {i, t} (\theta) \widehat {A} _ {i, t}, \operatorname{clip} (w _ {i, t} (\theta), 1 - \varepsilon , 1 + \varepsilon) \widehat {A} _ {i, t}\right) \right],\tag{2}
$$

where G is the number of generated responses to each query $x \ ( \mathrm { i . e . }$ ., the group size), and the importance ratio $w _ { i , t } ( \boldsymbol { \theta } )$ and advantage $\widehat { A } _ { i , t }$ of token $y _ { i , t }$ are:

$$
w _ {i, t} (\theta) = \frac {\pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i , t} | x , y _ {i , <   t})}, \quad \widehat {A} _ {i, t} = \widehat {A} _ {i} = \frac {r (x , y _ {i}) - \operatorname{mean} \left(\{r (x , y _ {i}) \} _ {i = 1} ^ {G}\right)}{\operatorname{std} \left(\{r (x , y _ {i}) \} _ {i = 1} ^ {G}\right)},\tag{3}
$$

respectively, where all the tokens in $y _ { i }$ share the same advantage as $\widehat { A } _ { i }$

## 3 Motivation

The growth in model size, sparsity (e.g., in Mixture-of-Experts models), and response length necessitates a large rollout batch size to maximize hardware utilization during RL. To improve sample efficiency, it is standard practice to partition a large batch of rollout data into multiple mini-batches for gradient updates. This procedure inevitably introduces an off-policy learning setting, where responses y are sampled from an old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ rather than the current policy $\pi _ { \theta }$ being optimized. This also explains the necessity of the clipping mechanism in PPO and GRPO, which prevents overly “off-policy” samples from being involved in gradient estimation.

While mechanisms like clipping aim to manage this off-policy discrepancy, we identify a more fundamental issue in GRPO: its objective is ill-posed. This problem becomes particularly acute when training large models on long-response tasks, leading to catastrophic model collapse. The ill-posed nature of the GRPO objective stems from a misapplication of importance sampling weights. The principle of importance sampling is to estimate the expectation of a function f under a target distribution $\pi _ { \mathrm { t a r } }$ by re-weighting samples drawn from a behavior distribution $\pi _ { \mathrm { b e h } }$

$$
\mathbb {E} _ {z \sim \pi_ {\mathrm{tar}}} [ f (z) ] = \mathbb {E} _ {z \sim \pi_ {\mathrm{beh}}} \left[ \frac {\pi_ {\mathrm{tar}} (z)}{\pi_ {\mathrm{beh}} (z)} f (z) \right].\tag{4}
$$

Crucially, this relies on averaging over multiple samples $( N \gg 1 )$ from the behavior distribution $\pi _ { \mathrm { b e h } }$ for the importance weight $\frac { \pi _ { \mathrm { t a r } } ( z ) } { \pi _ { \mathrm { b e h } } ( z ) }$ to effectively correct for the distributional mismatch.

In contrast, GRPO applies the importance weight $\frac { \pi _ { \boldsymbol { \theta } } ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } ) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } ) }$ at each token position t. Since this weight is based on a single sample $y _ { i , t }$ from each next-token distribution $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x , y _ { i , < t } )$ , it fails to perform the intended distribution-correction role. Instead, it introduces high-variance noise into the training gradients, which accumulates over long sequences and is exacerbated by the clipping mechanism. We have empirically observed that this can lead to model collapse that is often irreversible. Once the collapse occurs, resuming training is unavailing, even when reverting to a previous checkpoint and meticulously tuning hyperparameters (e.g., the clipping ranges), extending generation length, or switching the RL queries.

The above observation suggests a fundamental issue in GRPO’s design. The failure of the token-level importance weight points to a core principle: the unit of optimization objective should match the unit of reward. Since the reward is granted to the entire sequence, applying off-policy correction at the token level appears problematic. This motivates us to forego the token-level objective and explore utilizing importance weights and performing optimization directly at the sequence level.

## 4 Algorithm

## 4.1 GSPO: Group Sequence Policy Optimization

While the token-level importance weight $\frac { \pi _ { \boldsymbol { \theta } } ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } ) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } ) }$ is problematic in GRPO, we observe that in the context of language generation, the sequence-level importance weight ${ \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y | x ) } }$ has a clear theoretical meaning: it reflects how far the response y sampled from $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x )$ deviates from $\pi _ { \theta } ( \cdot | x )$ , which naturally aligns with the sequence-level reward and can also serve as a meaningful indicator of the clipping mechanism.

Based on this straightforward observation, we propose the Group Sequence Policy Optimization (GSPO) algorithm. GSPO employs the following sequence-level optimization objective:

$$
\mathcal {J} _ {\mathrm{GSPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \min \left(s _ {i} (\theta) \widehat {A} _ {i}, \operatorname{clip} (s _ {i} (\theta), 1 - \varepsilon , 1 + \varepsilon) \widehat {A} _ {i}\right) \right],\tag{5}
$$

where we adopt the group-based advantage estimation:

$$
\widehat {A} _ {i} = \frac {r (x , y _ {i}) - \mathrm{mean} \left(\{r (x , y _ {i}) \} _ {i = 1} ^ {G}\right)}{\mathrm{std} \left(\{r (x , y _ {i}) \} _ {i = 1} ^ {G}\right)},\tag{6}
$$

and define the importance ratio $s _ { i } ( \theta )$ based on sequence likelihood (Zheng et al., 2023):

$$
s _ {i} (\theta) = \left(\frac {\pi_ {\theta} (y _ {i} | x)}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i} | x)}\right) ^ {\frac {1}{| y _ {i} |}} = \exp \left(\frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \log \frac {\pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i , t} | x , y _ {i , <   t})}\right).\tag{7}
$$

Therefore, GSPO applies clipping to entire responses instead of individual tokens to exclude the overly “off-policy” samples from gradient estimation, which matches both the sequence-level rewarding and optimization. Note that we adopt length normalization in $s _ { i } ( \theta )$ to reduce the variance and to control $s _ { i } \mathbf { \hat { ( } } \theta \mathbf { ) }$ within a unified numerical range. Otherwise, the likelihood changes of a few tokens can result in dramatic fluctuations of the sequence-level importance ratio, and the importance ratios of responses with different lengths will require varying clipping ranges. We also note that the clipping ranges in GSPO and in previous algorithms $( \mathrm { e . g . , G R P O } )$ typically differ in order of magnitude due to the distinct definitions of importance ratios.

## 4.2 Gradient Analysis

We can derive the gradient of the GSPO objective as follows (clipping is omitted for brevity):

$$
\nabla_ {\theta} \mathcal {J} _ {\mathrm{GSPO}} (\theta) = \nabla_ {\theta} \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} s _ {i} (\theta) \widehat {A} _ {i} \right]\tag{8}
$$

$$
= \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} s _ {i} (\theta) \widehat {A} _ {i} \cdot \nabla_ {\theta} \log s _ {i} (\theta) \right]\tag{9}
$$

$$
= \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \left(\frac {\pi_ {\theta} (y _ {i} | x)}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i} | x)}\right) ^ {\frac {1}{| y _ {i} |}} \widehat {A} _ {i} \cdot \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} | x, y _ {i, <   t}) \right].\tag{10}
$$

For comparison, the gradient of the GRPO objective is as follows (note that $\widehat { A } _ { i , t } = \widehat { A } _ { i } )$

$$
\begin{array}{r l} & {\nabla_ {\theta} \mathcal {J} _ {\mathrm{GRPO}} (\theta) = \nabla_ {\theta} \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} w _ {i, t} (\theta) \widehat {A} _ {i, t} \right]} \\ & {\qquad = \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \widehat {A} _ {i} \cdot \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \frac {\pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i , t} | x , y _ {i , <   t})} \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} | x, y _ {i, <   t}) \right].} \end{array}\tag{11}
$$

(12)

Therefore, the fundamental distinction between GSPO and GRPO lies in how they weight the gradients of the log likelihoods oftokens. In GRPO, the tokens are weighted according to their respective “importance weight” $\frac { \pi _ { \boldsymbol { \theta } } ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } ) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } ) }$ . However, these unequal weights, which can vary among $( 0 , 1 + \varepsilon ]$ (for $\widehat { A } _ { i } > 0 )$ or $[ 1 - \varepsilon , + \infty )$ (for $\widehat { A } _ { i } < 0 )$ , are not negligible, and their impact can accumulate and lead to unpredictable consequences as training progresses. In contrast, GSPO weights all the tokens in a response equally, eliminating this instability factor of GRPO.

## 4.3 GSPO-token: A Token-level Objective Variant

In scenarios like multi-turn RL, we may desire a finer-grained advantage adjustment than the sequence level. To this end, we introduce a token-level objective variant of GSPO, namely GSPO-token, to allow token-wise advantage customization:

$$
\mathcal {J} _ {\mathrm{GSPO-token}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \min \left(s _ {i, t} (\theta) \widehat {A} _ {i, t}, \operatorname{clip} (s _ {i, t} (\theta), 1 - \varepsilon , 1 + \varepsilon) \widehat {A} _ {i, t}\right) \right],\tag{13}
$$

where

$$
s _ {i, t} (\theta) = \mathrm{sg} [ s _ {i} (\theta) ] \cdot \frac {\pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})}{\mathrm{sg} [ \pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t}) ]},\tag{14}
$$

and $\mathrm { s g } [ \cdot ]$ denotes only taking the numerical value but stopping the gradient, corresponding to the detach operation in PyTorch. The gradient of GSPO-token can be derived as:

$$
\nabla_ {\theta} \mathcal {J} _ {\mathrm{GSPO-token}} (\theta) = \nabla_ {\theta} \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} s _ {i, t} (\theta) \widehat {A} _ {i, t} \right]\tag{15}
$$

$$
= \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} s _ {i} (\theta) \cdot \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \widehat {A} _ {i, t} \frac {\nabla_ {\theta} \pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})}{\pi_ {\theta} (y _ {i , t} | x , y _ {i , <   t})} \right]\tag{16}
$$

$$
= \mathbb {E} _ {x \sim \mathcal {D}, \{y _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \left(\frac {\pi_ {\theta} (y _ {i} | x)}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i} | x)}\right) ^ {\frac {1}{| y _ {i} |}} \cdot \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \widehat {A} _ {i, t} \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} | x, y _ {i, <   t}) \right].\tag{17}
$$

Note that the term $\frac { \pi _ { \theta } ( y _ { i , t } | x , y _ { i , < t } ) } { \mathrm { s g } \big [ \pi _ { \theta } ( y _ { i , t } | x , y _ { i , < t } ) \big ] }$ has a numerical value of 1, so $s _ { i , t } ( \theta )$ is numerically equal to $s _ { i } ( \theta )$ Comparing Equation (5) and (13), and Equation (10) and (17), GSPO-token and GSPO are numerically identical in the optimization objective, clipping condition, and theoretical gradient when we set the advantages of all the tokens in the response $y _ { i }$ to the same value (i.e., $\widehat { A } _ { i , t } \stackrel { \triangledown } { = } \widehat { A } _ { i } )$ , while GSPO-token enjoys the higher flexibility of adjusting the advantages per token.

## 5 Experiments and Discussion

## 5.1 Empirical Results

We experiment with a cold-start model fine-tuned from Qwen3-30B-A3B-Base, and report the training reward curves as well as the model performance curves on the AIME’24 (average Pass@1 over 32 samplings), LiveCodeBench (202410-202502, average Pass@1 over 8 samplings), and CodeForces (Elo Rating) benchmarks. During the RL training, each batch of rollout data is partitioned into four minibatches for gradient updates. In GSPO, we set the left and right clipping ranges in Equation (5) to 3e-4 and 4e-4, respectively. We compare against GRPO as the baseline and set the left and right clipping ranges in Equation (2) to 0.2 and 0.27, respectively, which we have carefully tuned to ensure a fair comparison.

Note that GRPO necessitates the Routing Replay training strategy for the normal convergence of MoE RL, which we will additionally discuss in § 5.3, while GSPO has obviated the need for this strategy.

Figure 1 shows that the training with GSPO proceeds stably throughout. We observe that GSPO can deliver continuous performance improvement through increasing the training compute, regularly updating the query set, and extending the generation length. Moreover, GSPO also demonstrates superior training efficiency over GRPO, achieving better training accuracy and benchmark performance under the same training compute and consumed queries. Finally, we have successfully applied GSPO to the RL training of the latest Qwen3 models, strongly proving the efficacy of GSPO in unleashing the power of RL scaling for large language models.




Figure 1: Training curves of a cold-start model fine-tuned from Qwen3-30B-A3B-Base. GSPO possesses remarkably higher training efficiency than GRPO.

## 5.2 Curious Observation on Clipping Fractions

A key distinction of GSPO compared to GRPO is its practice of clipping entire responses rather than individual tokens. Particularly, as shown in Figure 2, we observe a difference of two orders of magnitude in the fractions of clipped tokens between GSPO and GRPO (while adjusting the clipping ranges does not alter the disparity in magnitude). However, despite clipping significantly more tokens and consequently using fewer for training (or gradient estimation), GSPO still achieves higher training efficiency than GRPO. This counter-intuitive finding — that clipping a much larger fraction of tokens leads to superior training efficiency — further indicates that GRPO’s token-level gradient estimates are inherently noisy and inefficient for sample exploitation. In contrast, GSPO’s sequence-level approach provides a more reliable and effective learning signal.

Figure 2: Average fractions of clipped tokens over the RL training of GSPO and GRPO.

## 5.3 Benefit of GSPO for MoE Training

Background Compared to the RL training of dense models, the sparse activation nature of MoE models introduces unique stability challenges. In particular, we found that when adopting the GRPO algorithm, the expert-activation volatility of MoE models can prevent RL training from converging properly. To be specific, after one or more gradient updates, the experts activated for the same response can change significantly. For example, with the 48-layer Qwen3-30B-A3B-Base model, after each RL gradient update and for the same rollout sample, there are roughly 10% of the experts activated under the new policy $\pi _ { \theta }$ that are different from those under the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . This phenomenon, which becomes more prominent in deeper MoE models, makes the token-level importance ratios $\begin{array} { r } { w _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid x , y _ { i , < t } ) } } \end{array}$ fluctuate drastically and further invalidates them, as discussed in § 3 and 4.2, consequently hindering the normal convergence of RL training.

Our Previous Approach To tackle this challenge, we previously employed the Routing Replay training strategy. Specifically, we cache the activated experts in $\pi _ { \theta _ { \mathrm { o l d } } }$ and “replay” these routing modes in $\pi _ { \theta }$ when computing the importance ratios $\begin{array} { r } { w _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , t } | x , y _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } | x , y _ { i , < t } ) } } \end{array}$ . In this way, for each token $y _ { i , t } , \pi _ { \theta } ( y _ { i , t } \vert x , y _ { i , < t } )$ and $\pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } | x , y _ { i , < t } )$ share the same activated network, so that we can restore the stability of the tokenlevel importance ratios and ensure optimization of the consistent activated network across gradient updates. Figure 3 demonstrates that Routing Replay serves as an essential technique in the normal convergence of the GRPO training of MoE models.

Training Compute  
Figure 3: The Routing Replay strategy plays a critical role in the normal convergence of the GRPO training of MoE models.

Benefit of GSPO Although Routing Replay enables the GRPO training of MoE models to converge properly, its practice of reusing routing modes incurs additional memory and communication overhead and can also limit the actual capacity of the MoE model. In contrast, as shown in Figure 1, GSPO eliminates the dependency on Routing Replay and is fully capable of computing the importance ratios $s _ { i } ( \theta )$ conventionally, converging normally, and optimizing stably. The key insight is that GSPO focuses only on the sequence likelihood $\mathsf { \bar { ( i . e . , } } \pi _ { \theta } ( \dot { y _ { i } } | x ) )$ ) and is not sensitive to the individual token likelihood (i.e., $\pi _ { \boldsymbol { \theta } } \big ( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } \big ) \big )$ . Since the MoE model always maintains its language modeling capability, the sequence likelihood will not fluctuate drastically. In summary, GSPO fundamentally resolves the expert-activation volatility issue in MoE models, obviating the need for complex workarounds like Routing Replay. This not only simplifies and stabilizes the training process but also allows the model to leverage its full capacity without artificial constraints.

## 5.4 Benefit of GSPO for RL Infrastructure

Given the precision discrepancies between training engines (e.g., Megatron) and inference engines $( \mathrm { e . g . } ,$ SGLang and vLLM), in practice, we typically use the training engine to recompute the likelihoods of sampled responses under the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . However, GSPO uses only sequence-level, rather than token-level, likelihoods for optimization, and intuitively, the former is much more tolerant of precision discrepancies. Hence, GSPO makes it possible to directly use the likelihoods returned by the inference engine for optimization, thereby avoiding the need for recomputation with the training engine. This can be especially beneficial in scenarios like partial rollout and multi-turn RL and in the training-inference disaggregated frameworks.

## 6 Conclusion

We propose Group Sequence Policy Optimization (GSPO), a new reinforcement learning algorithm for training large language models. Following the basic principle of importance sampling, GSPO defines importance ratios based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. GSPO demonstrates notably superior training stability, efficiency, and performance compared to GRPO and exhibits particular efficacy for the large-scale RL training of MoE models, laying the foundation for the exceptional improvements in the latest Qwen3 models. With GSPO as a scalable algorithmic cornerstone, we will continue to scale RL and look forward to the resulting fundamental advances in intelligence.

## References

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

MiniMax. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

OpenAI. Learning to reason with LLMs, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.

Team Qwen. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Team Qwen. Qwq-32b: Embracing the power of reinforcement learning, March 2025b. URL https: //qwenlm.github.io/blog/qwq-32b/.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Chujie Zheng, Pei Ke, Zheng Zhang, and Minlie Huang. Click: Controllable text generation with sequence likelihood contrastive learning. In Findings of the Associationfor Computational Linguistics: ACL 2023, 2023. URL https://aclanthology.org/2023.findings-acl.65/.
