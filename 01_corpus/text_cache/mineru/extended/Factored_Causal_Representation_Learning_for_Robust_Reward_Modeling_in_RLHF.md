# Factored Causal Representation Learning for Robust Reward Modeling in RLHF

Yupei Yang <sup>1</sup> <sup>2</sup> <sup>†</sup> Lin Yang <sup>2</sup> Wanxi Deng <sup>2</sup> Lin Qu <sup>2</sup> Fan Feng <sup>3</sup> <sup>4</sup> Biwei Huang <sup>3</sup> Shikui Tu <sup>1</sup> <sup>\*</sup> Lei Xu <sup>1</sup>

## Abstract

A reliable reward model is essential for aligning large language models (LLMs) with human preferences through reinforcement learning from human feedback (RLHF). However, standard reward models are susceptible to spurious features that are not causally related to human labels. This can lead to reward hacking, where high predicted reward does not translate into better behavior. In this work, we address this problem from a causal perspective by proposing a factored representation learning framework that decomposes the model’s contextual embedding into (1) causal factors that are sufficient for reward prediction and (2) non– causal factors that capture reward-irrelevant attributes such as length or sycophantic bias. The reward head is then constrained to depend only on the causal component. In addition, we introduce an adversarial head trained to predict reward from the non-causal factors, while applying gradient reversal to discourage them from encoding rewardrelevant information. Experiments on both mathematical and dialogue tasks demonstrate that our method learns more robust reward models and consistently improves downstream RLHF performance over state-of-the-art baselines. Analyses on length and sycophantic bias further validate the effectiveness of our method in mitigating reward hacking behaviors.

## 1. Introduction

In recent years, RLHF has emerged as a powerful approach for aligning LLMs with human preferences (Ouyang et al., 2022; Bai et al., 2022). A core component of RLHF is the reward model, which serves as a proxy for human judgment during policy optimization. However, standard reward model training is prone to learning spurious correlations: the model may assign higher scores to preference-irrelevant patterns, such as response length or sycophantic phrasing, rather than the true causal drivers of human preference (Liu et al., 2024; Miao et al., 2024; Wang et al., 2025). The LLM then exploits these shortcuts during RLHF, achieving higher predicted rewards while drifting from the intended objective—a phenomenon known as reward hacking (Amodei et al., 2016; Gao et al., 2023b; Skalse et al., 2022).

To tackle this issue, extensive efforts have been made to mitigate the influence of known spurious factors in reward modeling. For example, ODIN (Chen et al., 2024) decomposes the reward head into separate quality and length components to reduce reward hacking driven by response length. Park et al. (2024) develops a length-regularization strategy to prevent length exploitation during DPO (Rafailov et al., 2023). Wang et al. (2025) further proposes a more general maximum mean discrepancy (MMD)-based regularization that can constrain spurious behaviors beyond length, such as sycophantic bias (Sharma et al., 2023) or concept bias (Zhou et al., 2024). Despite their effectiveness, these approaches require explicitly specifying the spurious variables to be controlled, whereas it is challenging to anticipate all possible exploitation patterns in practice.

Another line of work seeks to improve reward models by filtering out irrelevant information directly through representation learning. Nath et al. (2024) employs a contrastive objective to learn goal-conditioned representations within the reward model, which helps distinguish between preferred and dispreferred responses. InfoRM (Miao et al., 2024; 2025a) adopts an information-theoretic perspective, introducing a variational information bottleneck objective to encourage the latent representations to retain only information relevant to human preference. However, to the best of our knowledge, none of the existing methods explicitly disentangle reward-irrelevant factors from the latent state for reward modeling. Nevertheless, causal representation learning (Schölkopf et al., 2021; Huang et al., 2022; Kong et al., 2023; Yang et al., 2024b) has demonstrated strong promise in traditional reinforcement learning for learning minimal sufficient state representations that capture only the causally relevant aspects of the environment.

Motivated by these insights, we propose CausalRM, a novel framework that explores the potential of causal representation learning to mitigate spurious correlations in reward modeling. Specifically, CausalRM decomposes the model’s contextual embedding into two disentangled components: (1) causalfactors that are sufficient for reward prediction, and (2) non-causal factors that capture reward-irrelevant attributes such as response length or stylistic bias. Building on this factorized representation, we constrain the reward model to predict rewards using only the causal factors, and optimize it with a standard pairwise preference loss augmented with mutual-information-based constraints. To further discourage the non-causal factors from carrying reward-predictive signals, we introduce an adversarial head trained to predict reward from these factors. Following Ganin & Lempitsky (2015), we then optimize it with a gradient reversal layer (GRL), so that the adversarial head learns to predict reward while the encoder is pushed to remove reward-relevant information from the non-causal component.

To summarize, our main contributions are three-fold:

• We investigate the potential of causal representation learning for mitigating reward hacking in RLHF, and propose CausalRM, a framework that explicitly decomposes the latent representation into causal factors and reward-irrelevant non-causal factors.

• To characterize both causal and non-causal representations, we design a novel VAE-based architecture with (1) a reward prediction head trained with a pairwise preference loss augmented by mutual information constraints, and (2) an adversarial head trained via gradient reversal to suppress reward-relevant signals in the non– causal component, which jointly enforces sufficiency, minimality, and causal invariance in the learned representations.

• Extensive experiments on mathematical and dialogue tasks show that CausalRM improves both reward model accuracy and downstream RLHF performance. Furthermore, we demonstrate strong mitigation of reward hacking behaviors through reduced sensitivity to length and sycophantic bias, providing empirical validation of causal invariance in learned representations.

## 2. Preliminaries

Reward modeling in RLHF. In RLHF, reward modeling aims to learn a scalar-valued function that approximates human preferences. Given a prompt x and a response y, a reward model outputs a scalar reward $r _ { \theta } ( x , y )$ indicating the degree of preference for y under x.

A widely used formulation is the Bradley–Terry model (Bradley $\&$ Terry, 1952), which defines the probability that a preferred response $y ^ { w }$ is favored over a rejected response $y ^ { l }$ as:

$$
p (y ^ {w} \succ y ^ {l} \mid x) = \frac {\exp (r _ {\theta} (x , y ^ {w}))}{\exp (r _ {\theta} (x , y ^ {w})) + \exp (r _ {\theta} (x , y ^ {l}))}.\tag{1}
$$

Given a human preference dataset $\boldsymbol { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ^ { w } , y _ { i } ^ { l } ) \} _ { i = 1 } ^ { N } ,$ the reward model is typically trained by minimizing the pairwise negative log-likelihood:

$$
\mathcal {L} _ {\mathrm{RM}} (\theta) = - \mathbb {E} _ {(x, y ^ {w}, y ^ {l}) \sim \mathcal {D}} \left[ \log \sigma \big (r _ {\theta} (x, y ^ {w}) - r _ {\theta} (x, y ^ {l}) \big) \right],
$$

where $\sigma ( \cdot )$ denotes the sigmoid function.

(2)

Standard parameterization and reward hacking. In practice, the reward model is often initialized from a supervised fine-tuned (SFT) language model by reusing its backbone as a feature extractor and attaching a lightweight reward head, which is typically a single linear layer. Concretely, given a prompt–response pair $( x , y )$ , the SFT backbone produces a contextual representation

$$
h = f _ {\phi} (x, y),\tag{3}
$$

and the reward head maps it to a scalar reward

$$
r _ {\theta} (x, y) = g _ {\psi} (h), \qquad \theta = (\phi , \psi),\tag{4}
$$

where $f _ { \phi }$ denotes the pretrained backbone and $g _ { \psi }$ denotes the reward head.

While effective, this parameterization can be vulnerable to reward hacking: the reward model may inherit and amplify preference-irrelevant signals already encoded in the SFT representation, causing the learned reward to correlate with spurious patterns rather than the causal drivers of human judgment (Miao et al., 2024; Wang et al., 2025).

## 3. Methodology

In this section, we first analyze why standard reward models tend to learn spurious correlations from a causal perspective. Building on these insights, we construct a causal reward model, termed CausalRM, that filters out reward-irrelevant information via factored representation learning. Finally, we present the complete estimation procedure for CausalRM.

## 3.1. A Causal View of Reward Hacking

The causal structure underlying standard reward modeling can be represented by Figure 1, where (x, y) are the input prompt and response pair, $z ^ { c }$ denotes causal factors that contain the essential information to predict the reward $r ,$ and $z ^ { n c }$ represents spurious factors that do not causally influence the true reward, such as response length or stylistic bias.

Figure 1. Causal graph for standard reward modeling. The prompt– response pair $( x , y )$ encode both causal $( z ^ { c } )$ and non-causal $( z ^ { \hat { n } c } )$ factors, which in turn affect the predicted reward r. While the path $z ^ { c } $ r is desired, the spurious path $z ^ { n c } $ r leads to reward hacking.

As illustrated, the presence of a direct edge $z ^ { n c } \to r$ allows spurious features to directly affect the learned reward, thereby leading to reward hacking. For example, suppose $z ^ { n c }$ captures response length on mathematical tasks, then changing the length alone may substantially alter the predicted reward, even when the underlying solution quality remains unchanged. Instead, a robust reward model should satisfy causal invariance with respect to non-causal factors (Bühlmann, 2020; Veitch et al., 2021):

$$
r _ {\theta} (x, y) \perp z ^ {n c}.\tag{5}
$$

In other words, the reward value should be insensitive to non-causal attributes ofthe prompt-response pair.

## 3.2. CausalRM: Factored Causal Representation Learning for Reward Models

Motivated by the causal analysis in Section 3.1, we propose CausalRM, a reward modeling framework that aims to block the spurious path $z ^ { n c } \to r \ b y$ (1) structurally restricting reward prediction to depend only on a causal representation, and (2) actively removing reward-predictive information from the remaining representation. Concretely, CausalRM augments a standard RM with a latent-variable bottleneck and two auxiliary heads, jointly promoting sufficiency for reward prediction, invariance to non-causal variation, and non-degenerate representations.

Factored latent representation. Given a prompt– response pair $( x , y )$ , we first compute a contextual embedding $h = f _ { \phi } ( x , y )$ using the SFT backbone. CausalRM then maps h to two latent variables via a variational encoder:

$$
q _ {\alpha} (z ^ {c} \mid h), \qquad q _ {\alpha} (z ^ {n c} \mid h),\tag{6}
$$

where $z ^ { c }$ is encouraged to retain information that is sufficient for predicting human preference, while $z ^ { n c }$ is encouraged to capture reward-irrelevant attributes. We parameterize both posteriors as diagonal-covariance Gaussians whose means and log-variances are produced by separate linear projections applied to $h ,$ with standard normal priors $p ( z ^ { c } ) = p ( z ^ { n c } ) = \mathcal { N } ( 0 , I )$

This factorization provides a convenient interface for imposing causal invariance: downstream prediction modules can be forced to condition only on $z ^ { c }$ , while $z ^ { n c }$ serves as a dedicated channel for non-causal variation.

Causal reward head. We predict reward solely from the causal component:

$$
\hat {r} = r _ {\theta} (x, y) = g _ {\psi} (z ^ {c}), \quad \theta = (\phi , \alpha , \psi),\tag{7}
$$

where $g _ { \psi }$ is a linear reward head. By construction, the reward head has no access to $z ^ { n c }$ , which implements a structural bias towards the invariance principle in Eq. (5).

Reconstruction head. A structural restriction alone may lead to degenerate solutions, such as posterior collapse (Bowman et al., 2016; Alemi et al., 2016). To encourage $( z ^ { c } , z ^ { n c } )$ to retain the information in h while allowing it to be redistributed across the two factors, we add a reconstruction decoder

$$
\hat {h} = d _ {\eta} ([ z ^ {c}; z ^ {n c} ]),\tag{8}
$$

which reconstructs the backbone embedding from the concatenated latents.

Adversarial head with gradient reversal. Finally, to explicitly remove reward-predictive signals from the noncausal component, we introduce an adversarial head $a _ { \omega }$ that predicts reward from $z ^ { n c }$

$$
\hat {r} ^ {\mathrm{adv}} = a _ {\omega} (z ^ {n c}).\tag{9}
$$

The adversary is optimized to be predictive, while the encoder is optimized via a gradient reversal layer to make $z ^ { n c }$ uninformative about the reward. This adversarial objective complements the structural restriction above by penalizing any reward-relevant information that leaks into $z ^ { n c }$ , thereby encouraging the desired invariance. The overall architecture of CausalRM is illustrated in Figure 2.

## 3.3. Model Estimation

CausalRM is trained by jointly optimizing the backbone, the factorized encoder, and the three heads introduced in Section 3.2. Specifically, we learn parameters $\theta \ =$ $( \phi , \alpha , \psi , \eta , \omega )$ by solving:

$$
\min _ {\phi , \alpha , \psi , \eta} \max _ {\omega} \underbrace {\mathcal {L} _ {\text {pref}} + \lambda_ {\mathrm{KL}} ^ {c} \mathcal {L} _ {\mathrm{KL}} ^ {c}} _ {\text {minimal sufficiency}} + \underbrace {\lambda_ {\text {rec}} \mathcal {L} _ {\text {rec}}} _ {\text {non - degeneracy}} + \underbrace {\lambda_ {\mathrm{KL}} ^ {n c} \mathcal {L} _ {\mathrm{KL}} ^ {n c} - \lambda_ {\text {adv}} \mathcal {L} _ {\text {adv}}} _ {\text {invariance}},\tag{10}
$$

where $\mathcal { L } _ { \mathrm { { p r e f } } }$ encourages $z ^ { c }$ to be predictive of human preference, $\mathcal { L } _ { \mathrm { K I } } ^ { c }$ enforces an information bottleneck on $z ^ { c }$ to discourage redundant information, $\mathcal { L } _ { \mathrm { r e c } }$ prevents degenerate factorizations by reconstructing the backbone embedding, $\mathcal { L } _ { \mathrm { K L } } ^ { n c }$ regularizes the non-causal latent by matching $q _ { \alpha } ( z ^ { n c } \mid h )$ to the prior, and $\mathcal { L } _ { \mathrm { a d v } }$ measures how well the non-causal factor $z ^ { n c }$ can predict human preferences. This minimax objective reflects distinct goals for different components:

Figure 2. Overview of CausalRM. The backbone embedding h is factorized into causal latents $z ^ { c }$ and non-causal latents $z ^ { n c }$ via a variational encoder. Reward prediction is restricted to depend only on $z ^ { c }$ , while an adversarial head trained through a gradient reversal layer (GRL) discourages $z ^ { n c }$ from encoding reward-predictive information. A reconstruction decoder prevents degenerate factorization by reconstructing h from $[ z ^ { c } ; z ^ { n c } ]$

• The adversary parameters ω are minimized over $\mathcal { L } _ { \mathrm { a d v } }$ encouraging it to accurately predict preferences from the non-causal representation $z ^ { n c }$

• The encoder parameters $( \phi , \alpha )$ are maximized over $\mathcal { L } _ { \mathrm { a d v } } .$ , i.e., they are updated to increase the adversarial loss, thereby removing reward-predictive signals from $z ^ { n c }$ and enforcing causal invariance.

The coefficients $\lambda _ { \mathrm { K L } } ^ { c } , \lambda _ { \mathrm { K L } } ^ { n c } , \lambda _ { \mathrm { a d v } } , \lambda _ { \mathrm { r e c } }$ control the trade-offs between these competing objectives.

Sufficiency and minimality for the causal factor $z ^ { c } .$ . To make $z ^ { c }$ sufficient for reward prediction, we maximize the mutual information $I ( z ^ { c } ; r )$ between the causal latent and the reward signal. To further encourage a minimal causal representation, we introduce an information bottleneck that penalizes reward-irrelevant information retained from the backbone embedding, measured by $I ( h ; z ^ { c } )$ . We thus seek to maximize the following objective:

$$
\max \quad I (z ^ {c}; r) - \lambda_ {\mathrm{KL}} ^ {c} I (h; z ^ {c}).\tag{11}
$$

Given a preference triplet $( x , y ^ { w } , y ^ { l } )$ , we optimize a variational lower bound of Eq. (11). Let $h ^ { w } = f _ { \phi } ( x , y ^ { w } )$ and $h ^ { l } = f _ { \phi } ( x , y ^ { l } )$ be the backbone embeddings, and sample $z ^ { w , c } \sim q _ { \alpha } ( z ^ { c } \mid h ^ { w } )$ and $z ^ { l , c } \sim q _ { \alpha } ( z ^ { c } \mid h ^ { l } )$ . Then,

$$
\begin{array}{l} I (z ^ {c}; r) - \lambda_ {\mathrm{KL}} ^ {c} I (h; z ^ {c}) \geq \mathbb {E} _ {(x, y ^ {w}, y ^ {l}) \sim \mathcal {D}} \Big [ \underbrace {\log \sigma \big (r _ {\theta} (x , y ^ {w}) - r _ {\theta} (x , y ^ {l}) \big)} _ {\triangleq - \mathcal {L} _ {\mathrm{pref}}} \\ - \lambda_ {\mathrm{KL}} ^ {c} \underbrace {\Big (\mathrm{KL} (q _ {\alpha} (z ^ {w , c} \mid h ^ {w}) \| p (z ^ {c})) + \mathrm{KL} \big (q _ {\alpha} (z ^ {l , c} \mid h ^ {l}) \| p (z ^ {c}) \big) \Big)} _ {\triangleq \mathcal {L} _ {\mathrm{KL}} ^ {c}} \Big ] \end{array}\tag{12}
$$

Here $\mathcal { L } _ { \mathrm { p r e f } }$ is the standard pairwise preference loss, and $\mathcal { L } _ { \mathrm { K I } } ^ { c }$ upper bounds the information bottleneck term via a $\mathrm { K L }$ regularizer. We use $z ^ { w , c }$ and $z ^ { l , c }$ to denote the causal latents inferred from the preferred and dispreferred responses, respectively. A detailed derivation is provided in Appendix D.

Invariance via adversarial prediction on $z ^ { n c }$ . While the objective above encourages $z ^ { c }$ to be minimally sufficient, it does not by itself prevent reward-predictive information from being encoded in the non-causal factor $z ^ { n c }$ . To this end, we introduce an adversarial head $a _ { \omega }$ that attempts to predict human preferences from $z ^ { n c }$ . Specifically, we decompose the adversarial objective into (1) an adversarial preference loss that trains $a _ { \omega }$ to predict preferences from $z ^ { n c }$ , and $( 2 )$ a standard KL regularizer on $z ^ { n c }$ that prevents unconstrained growth of the non-causal latent:

$$
\mathcal {L} _ {\mathrm{adv}} = - \mathbb {E} _ {(x, y ^ {w}, y ^ {l}) \sim \mathcal {D}} \left[ \log \sigma \big (a _ {\omega} (z ^ {w, n c}) - a _ {\omega} (z ^ {l, n c}) \big) \right], \tag {12}
$$

$$
\mathcal {L} _ {\mathrm{KL}} ^ {n c} = \mathbb {E} _ {(x, y ^ {w}, y ^ {l}) \sim \mathcal {D}} \left[ \mathrm{KL} (q _ {\alpha} (z ^ {w, n c} | h ^ {w}) \| p (z ^ {n c})) + \mathrm{KL} (q _ {\alpha} (z ^ {l, n c} | h ^ {l}) \| p (z ^ {n c})) \right],\tag{13}
$$

where $z ^ { w , n c } \sim q _ { \alpha } ( z ^ { n c } \mid h ^ { w } )$ and $z ^ { l , n c } \sim q _ { \alpha } ( z ^ { n c } \mid h ^ { l } )$ are sampled from the non-causal posteriors for the preferred and dispreferred responses, respectively. The adversary parameters ω are optimized to minimize ${ \mathcal { L } } _ { \mathrm { a d v } } ,$ , yielding a strong predictor from $z ^ { n c }$ . Meanwhile, we place a gradient reversal layer between $z ^ { n c }$ and $a _ { \omega } .$ , so that gradients from $\mathcal { L } _ { \mathrm { a d v } }$ are negated before reaching the encoder. Consequently, the encoder is encouraged to maximize the $\mathcal { L } _ { \mathrm { a d v } }$ , making $z ^ { n c }$ uninformative for preference prediction and thereby reducing reward leakage into the non-causal component.

Non-degeneracy via reconstruction. To ensure that the factorized latent representation $( z ^ { c } , z ^ { n c } )$ collectively preserves the information in the backbone embedding $h ,$ we include a reconstruction term that minimizes the distance between h and its reconstruction:

$$
\mathcal {L} _ {\mathrm{rec}} = \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \left\| h - d _ {\eta} ([ z ^ {c}; z ^ {n c} ]) \right\| _ {2} ^ {2} \right],\tag{15}
$$

where $d _ { \eta }$ is the reconstruction decoder and $[ z ^ { c } ; z ^ { n c } ]$ denotes concatenation. This objective encourages the latent representations to preserve the information in h while allowing reward-irrelevant variation to be captured in z<sup>nc</sup>, facilitating the intended factorization.

These objectives are unified into a single training objective that jointly learns a minimal sufficient causal factor $z ^ { c }$ for preference prediction, enforces invariance by suppressing reward-predictive signals in the non-causal factor z<sup>nc</sup> through adversarial training with gradient reversal, and prevents degenerate representations via reconstruction. The complete training procedure is summarized in Algorithm 1.

## 4. Experiments

We evaluate the effectiveness of CausalRM in mitigating reward hacking across two representative tasks: mathematical reasoning and open-ended dialogue. Specifically, our evaluation focuses on three key research questions:

• RQ1 (Reward modeling accuracy): Can CausalRM better approximate human preferences compared to existing reward models?

• RQ2 (Downstream RLHF alignment): Do improvements in reward modeling translate into stronger downstream RLHF performance?

• RQ3 (Causal invariance): Does CausalRM better satisfy the causal invariance principle, particularly with respect to known spurious attributes such as response length and sycophantic bias?

We further report additional experiments in Appendix E and Appendix F, including ablation studies as well as supplementary analyses on robustness, scalability, and evaluation reliability.

## 4.1. Setup

Datasets. For mathematical reasoning, we use OpenMathInstruct-1 (Toshniwal et al., 2024), which contains 1.8M problem–solution pairs sourced from GSM8K and MATH. Following Nath et al. (2024), we adopt their constructed preference dataset to train both the reward model and the downstream policy. We evaluate reward modeling and RLHF performance on both in-distribution (ID) and out-of-distribution (OOD) benchmarks. The ID evaluation uses the test split from the same preference distribution, while the OOD evaluation includes algebra222 (He-Yueya et al., 2023), GSM-Hard (Gao et al., 2023a), ASDiv (Miao et al., 2020), MAWPS (Koncel-Kedziorski et al., 2016), and SVAMP (Patel et al., 2021).

For open-ended dialogue, we train the reward model and perform RLHF on Anthropic-RLHF-HH (Bai et al., 2022), which provides human preference annotations on helpfulness and harmlessness for assistant responses. We use the dataset’s test split as the ID evaluation set, and evaluate OOD generalization on MT-Bench (Zheng et al., 2023), PKU-SafeRLHF (Ji et al., 2023), SHP (Askell et al., 2021), and TruthfulQA (Lin et al., 2022).

Models. For mathematical reasoning, we adopt Qwen2.5- Math-7B (Yang et al., 2024a), a strong decoder-only LLM that has been tuned on GSM8K and MATH, as the base model for both reward modeling and RLHF. For openended dialogue, we first perform supervised fine-tuning on ShareGPT (Chiang et al., 2023) using Qwen2.5-7B (Qwen et al., 2025), and then use the resulting SFT backbone as the base model. Across all experiments, we employ Proximal Policy Optimization (PPO; (Schulman et al., 2017)) for RLHF. All SFT, reward model, and PPO training are implemented using the OpenRLHF (Hu et al., 2024) framework.

Baselines. We compare CausalRM against the following state-of-the-art reward modeling approaches:

• Standard RM: the conventional reward model trained with the Bradley–Terry pairwise loss, using a linear reward head on top of the backbone embedding.

• GoalRM (Nath et al., 2024): improves reward modeling by learning goal-conditioned representations via a contrastive, Q-function-based objective, which helps distinguish preferred and dispreferred responses for RLHF alignment.

• InfoRM (Miao et al., 2025a): introduces a variational information bottleneck to filter out information irrelevant for reward prediction. Our work extends InfoRM by explicitly factorizing the latent space into causal and non-causal components, and jointly optimizing them under a unified framework that enforces both sufficiency and invariance.

For downstream alignment, we additionally report the performance of the SFT model as a reference to evaluate whether RLHF with learned reward models leads to meaningful improvement. We further compare against artifactspecific baselines in Appendix F.1.

Evaluation Metrics. We evaluate reward models using pairwise accuracy, the fraction of preference pairs where the model correctly assigns a higher score to the preferred response, across both tasks. For downstream RLHF, evaluation differs by domain. On mathematical reasoning, we report the model’s final answer accuracy against ground-truth solutions. For open-ended dialogue, we follow common practices (Chen et al., 2023) and use Qwen3-Max as an

Table 1. Reward model performance measured by pairwise accuracy (%, higher is better) on Mathematical Reasoning and Open-Ended Dialogue. We report mean±std over 3 runs. ID/OOD denote in-distribution/out-of-distribution evaluation, and Avg. denotes the average across the corresponding benchmarks. Bold indicates the best mean result and underlined indicates the second best.

<table><tr><td rowspan="3">Method</td><td colspan="9">Mathematical Reasoning</td><td colspan="8">Open-Ended Dialogue</td></tr><tr><td colspan="3">ID</td><td colspan="6">OOD</td><td colspan="3">ID</td><td colspan="5">OOD</td></tr><tr><td>GSM8K</td><td>MATH</td><td>Avg.</td><td>Algebra222</td><td>GSM-Hard</td><td>ASDiv</td><td>MAWPS</td><td>SVAMP</td><td>Avg.</td><td>Helpful</td><td>Harmless</td><td>Avg.</td><td>MT-Bench</td><td>PKU-SafeRLHF</td><td>SHP</td><td>TruthfulQA</td><td>Avg.</td></tr><tr><td>Standard RM</td><td>75.5(±0.5)</td><td>60.2(±0.6)</td><td>67.9(±0.3)</td><td>83.6(±0.4)</td><td>60.8(±0.7)</td><td>90.5(±0.3)</td><td>91.5(±0.3)</td><td>88.6(±0.4)</td><td>83.0(±0.3)</td><td>67.4(±0.4)</td><td>73.8(±0.5)</td><td>70.6(±0.3)</td><td>68.2(±0.6)</td><td>57.8(±0.7)</td><td>54.2(±0.8)</td><td>58.5(±0.6)</td><td>59.7(±0.5)</td></tr><tr><td>GoalRM</td><td>80.3(±0.4)</td><td>56.2(±0.7)</td><td>68.3(±0.4)</td><td>81.6(±0.5)</td><td>64.4(±0.5)</td><td>89.5(±0.4)</td><td>89.1(±0.4)</td><td>86.4(±0.5)</td><td>82.2(±0.3)</td><td>67.5(±0.5)</td><td>74.7(±0.4)</td><td>71.1(±0.3)</td><td>66.2(±0.7)</td><td>59.3(±0.6)</td><td>53.8(±0.7)</td><td>63.6(±0.6)</td><td>60.7(±0.4)</td></tr><tr><td>InfoRM</td><td>74.1(±0.6)</td><td>58.0(±0.5)</td><td>66.1(±0.4)</td><td>82.7(±0.5)</td><td>63.5(±0.6)</td><td>88.9(±0.5)</td><td>89.9(±0.4)</td><td>87.6(±0.5)</td><td>82.5(±0.4)</td><td>67.9(±0.5)</td><td>73.7(±0.5)</td><td>70.8(±0.3)</td><td>66.7(±0.6)</td><td>60.1(±0.5)</td><td>50.4(±0.9)</td><td>61.9(±0.7)</td><td>59.8(±0.4)</td></tr><tr><td>CausalRM (Ours)</td><td>81.7(±0.3)</td><td>58.4(±0.4)</td><td>70.1(±0.2)</td><td>89.9(±0.2)</td><td>66.2(±0.4)</td><td>89.9(±0.3)</td><td>92.2(±0.2)</td><td>89.6(±0.3)</td><td>85.6(±0.2)</td><td>69.4(±0.3)</td><td>75.2(±0.3)</td><td>72.3(±0.2)</td><td>68.3(±0.4)</td><td>60.9(±0.4)</td><td>53.9(±0.5)</td><td>66.2(±0.4)</td><td>62.3(±0.3)</td></tr></table>

Table 2. Downstream RLHF performance on Mathematical Reasoning measured by final-answer accuracy (%, higher is better). We report mean±std over 3 runs.

<table><tr><td rowspan="2">Method</td><td colspan="3">ID</td><td colspan="6">OOD</td></tr><tr><td>GSM8K</td><td>MATH</td><td>Avg.</td><td>Algebra222</td><td>GSM-Hard</td><td>ASDiv</td><td>MAWPS</td><td>SVAMP</td><td>Avg.</td></tr><tr><td>SFT</td><td> $80.4_{\pm 0.8}$ </td><td> $53.3_{\pm 1.0}$ </td><td> $66.9_{\pm 0.7}$ </td><td> $80.2_{\pm 0.9}$ </td><td> $54.5_{\pm 1.1}$ </td><td> $82.4_{\pm 0.7}$ </td><td> $93.6_{\pm 0.5}$ </td><td> $88.6_{\pm 0.6}$ </td><td> $80.0_{\pm 0.6}$ </td></tr><tr><td>Standard RM</td><td> $85.5_{\pm 0.7}$ </td><td> $50.1_{\pm 1.2}$ </td><td> $67.8_{\pm 0.8}$ </td><td> $89.6_{\pm 0.6}$ </td><td> $50.3_{\pm 1.0}$ </td><td> $88.0_{\pm 0.8}$ </td><td> $95.3_{\pm 0.4}$ </td><td> $92.1_{\pm 0.5}$ </td><td> $83.1_{\pm 0.6}$ </td></tr><tr><td>GoalRM</td><td> $89.4_{\pm 0.6}$ </td><td> $55.6_{\pm 0.9}$ </td><td> $72.5_{\pm 0.6}$ </td><td> $95.1_{\pm 0.4}$ </td><td> $70.1_{\pm 0.8}$ </td><td> $88.4_{\pm 0.7}$ </td><td> $95.9_{\pm 0.4}$ </td><td> $92.9_{\pm 0.4}$ </td><td> $88.5_{\pm 0.5}$ </td></tr><tr><td>InfoRM</td><td> $71.0_{\pm 1.4}$ </td><td> $24.9_{\pm 1.6}$ </td><td> $48.0_{\pm 1.2}$ </td><td> $60.4_{\pm 1.5}$ </td><td> $37.5_{\pm 1.3}$ </td><td> $46.3_{\pm 1.4}$ </td><td> $51.4_{\pm 1.2}$ </td><td> $53.9_{\pm 1.1}$ </td><td> $49.9_{\pm 1.0}$ </td></tr><tr><td>CausalRM (Ours)</td><td> $91.8_{\pm 0.5}$ </td><td> $56.1_{\pm 0.8}$ </td><td> $74.0_{\pm 0.5}$ </td><td> $97.3_{\pm 0.3}$ </td><td> $71.0_{\pm 0.7}$ </td><td> $89.1_{\pm 0.6}$ </td><td> $96.5_{\pm 0.3}$ </td><td> $93.9_{\pm 0.4}$ </td><td> $89.6_{\pm 0.4}$ </td></tr></table>

Improved reward prediction with CausalRM translates into stronger RLHF performance (RQ2). We report downstream RLHF results on mathematical reasoning in external judge to perform pairwise comparisons between responses generated by policies trained with CausalRM and those trained with each baseline, reporting the win rate as the evaluation metric. To reduce the cost of LLM-based evaluation, we randomly sample 1,000 instances from the corresponding test set for the pairwise judging.

Notably, this advantage becomes more pronounced under OOD shifts, where reward hacking and spurious correlations are more likely to emerge. On mathematical reasoning OOD benchmarks, CausalRM attains 85.6% average pairwise accuracy, outperforming the second-best method by 2.6%. Similarly, on dialogue OOD evaluation, CausalRM yields a 1.6% improvement over GoalRM. These results suggest that by explicitly disentangling reward-relevant and rewardirrelevant factors during reward model training, CausalRM generalizes better to unseen datasets and is less prone to exploiting spurious features.

CausalRM consistently outperforms baselines in predicting human preferences (RQ1). As shown in Table 1, CausalRM achieves strong reward modeling performance across both mathematical reasoning and open-ended dialogue. On the ID splits, CausalRM reaches an average pairwise accuracy of 70.1% on math and 72.3% on dialogue, improving over the best baseline by 1.8% and 1.2%.

## 4.2. Main Results

Table 2 and on open-ended dialogue in Table 3. As shown, RLHF with CausalRM yields consistent gains over baseline reward models, improving final-answer accuracy by 1.5% on ID benchmarks and by 1.1% on OOD benchmarks.

To examine whether these gains stem from mitigating reward hacking, we further analyze the discrepancy between the reward values optimized during RLHF and the corresponding ground-truth (gold) scores. Figure 3 presents the evolution of reward predictions and ground-truth performance on the ID test set throughout RLHF training, where dashed lines denote the normalized rewards predicted by different reward models, and solid lines indicate the corresponding gold rewards. As training proceeds, policies optimized with baseline reward models exhibit a noticeable degradation in gold reward to varying degrees. This issue is particularly severe for InfoRM, where the gold reward diverges sharply from the predicted reward. Such a widening gap is a typical signature of reward hacking and helps explain InfoRM’s unexpectedly poor RLHF performance. We provide qualitative examples illustrating this reward hacking phenomenon in Appendix K. In contrast, CausalRM maintains a consistent trend between predicted and gold rewards throughout training, highlighting its robustness to spurious features.

On open-ended dialogue, CausalRM similarly demonstrates strong alignment with human preferences. On the ID split, it achieves an average win rate of 54.8% against Standard RM, 45.5% against InfoRM, and 42.3% against GoalRM. This advantage persists under OOD evaluation with an average win rate of 51.3%, 38.7%, and 31.6%, respectively.

Normalized training step Normalized training step
Table 3. Downstream RLHF performance on open-ended dialogue evaluated by Qwen3-Max pairwise comparison win rate (%). Each entry reports mean±std over 3 runs of Win/Tie/Lose for the CausalRM-trained policy against an opponent policy trained with a baseline reward model.

<table><tr><td rowspan="3">Model</td><td rowspan="3">Opponent</td><td colspan="8">ID</td><td colspan="15">OOD</td></tr><tr><td colspan="3">Anthropic-Helpful</td><td colspan="3">Anthropic-Harmless</td><td colspan="2">Avg.</td><td colspan="3">MT-Bench</td><td colspan="3">PKU-SafeRLHF</td><td colspan="3">SHP</td><td colspan="3">TruthfulQA</td><td colspan="3">Avg.</td></tr><tr><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td></tr><tr><td rowspan="4">CausalRM (Ours)</td><td>SFT</td><td>76.2(±1.0)</td><td>19.4(±0.8)</td><td>4.4(±0.4)</td><td>68.1(±1.2)</td><td>23.8(±1.0)</td><td>8.1(±0.6)</td><td>72.2(±0.9)</td><td>21.6(±0.8)</td><td>6.2(±0.4)</td><td>44.3(±1.5)</td><td>41.0(±1.3)</td><td>14.7(±0.9)</td><td>67.8(±1.1)</td><td>21.5(±0.9)</td><td>10.7(±0.7)</td><td>77.5(±1.0)</td><td>11.8(±0.8)</td><td>10.7(±0.6)</td><td>52.2(±1.4)</td><td>41.2(±1.1)</td><td>6.6(±0.5)</td><td>60.5(±1.1)</td><td>28.9(±0.9)</td></tr><tr><td>Standard RM</td><td>53.7(±1.4)</td><td>35.3(±1.1)</td><td>11.0(±0.8)</td><td>55.9(±1.3)</td><td>30.9(±1.0)</td><td>13.2(±0.9)</td><td>54.8(±1.1)</td><td>33.1(±0.9)</td><td>12.1(±0.7)</td><td>42.1(±1.6)</td><td>44.0(±1.4)</td><td>13.9(±0.8)</td><td>57.2(±1.2)</td><td>29.7(±1.0)</td><td>13.1(±0.8)</td><td>55.0(±1.5)</td><td>26.3(±1.1)</td><td>18.7(±1.0)</td><td>50.9(±1.3)</td><td>30.4(±1.0)</td><td>18.7(±0.9)</td><td>51.3(±1.2)</td><td>32.6(±0.9)</td></tr><tr><td>GoalRM</td><td>50.0(±1.5)</td><td>39.0(±1.2)</td><td>11.0(±0.8)</td><td>34.6(±1.7)</td><td>44.1(±1.5)</td><td>21.3(±1.1)</td><td>42.3(±1.4)</td><td>41.6(±1.2)</td><td>16.1(±0.9)</td><td>35.0(±1.8)</td><td>51.0(±1.6)</td><td>14.0(±0.9)</td><td>31.6(±1.6)</td><td>47.8(±1.4)</td><td>20.6(±1.1)</td><td>27.9(±1.5)</td><td>50.8(±1.6)</td><td>21.3(±1.0)</td><td>31.7(±1.6)</td><td>52.9(±1.5)</td><td>15.4(±0.9)</td><td>31.6(±1.4)</td><td>50.6(±1.3)</td></tr><tr><td>InfoRM</td><td>52.4(±1.3)</td><td>38.0(±1.2)</td><td>9.6(±0.7)</td><td>38.5(±1.6)</td><td>37.2(±1.4)</td><td>24.3(±1.2)</td><td>45.5(±1.3)</td><td>37.6(±1.1)</td><td>16.9(±0.9)</td><td>34.3(±1.7)</td><td>48.8(±1.5)</td><td>16.9(±1.0)</td><td>42.7(±1.5)</td><td>33.1(±1.2)</td><td>24.2(±1.1)</td><td>33.8(±1.6)</td><td>45.6(±1.4)</td><td>20.6(±1.0)</td><td>44.1(±1.4)</td><td>39.7(±1.3)</td><td>16.2(±0.9)</td><td>38.7(±1.3)</td><td>41.8(±1.1)</td></tr></table>

Figure 3. Reward hacking diagnosis on mathematical reasoning. The dashed curve shows the average normalized reward predicted by each reward model on the ID test set, and the solid curve is the average gold score measured by final-answer accuracy.

Following Rafailov et al. (2024) and Miao et al. (2025b), we further assess reward hacking mitigation by tracking Qwen3-Max win-rate dynamics of RLHF policies against the SFT reference throughout training. Figure 4 summarizes the results. Overall, CausalRM exhibits more stable training dynamics and sustained preference improvement, consistent with effective reward hacking mitigation, which in turn leads to the stronger RLHF outcomes reported in Table 3.

## 4.3. Causal Invariance Analyses

In this subsection, we further investigate whether CausalRM better satisfies the causal invariance principle in Eq. (5), using response length and sycophantic bias as representative spurious attributes. Our main finding is that by explicitly disentangling causal and non-causal representations, CausalRM substantially reduces the sensitivity of predicted rewards to spurious attributes compared to strong baselines (RQ3).

Length bias. Response length is a well-known spurious feature in mathematical reasoning that can induce reward hacking (Singhal et al., 2023; Zhou et al., 2025). We therefore examine how sensitive different reward models are to answer length. Figure 5 shows the average predicted reward as a function of response length on the chosen responses from the ID test sets. CausalRM remains nearly invariant across length bins, with a standard deviation of only 0.03. In contrast, baseline reward models exhibit substantial fluctuations, particularly showing a pronounced negative preference for longer responses. This length sensitivity is a plausible contributor to reward hacking during RLHF.

Figure 4. Average win rate against the SFT model on the ID test sets of open-ended dialogue benchmarks during RLHF.

Figure 5. Sensitivity of predicted reward to response length on mathematical reasoning. Length is normalized to [0, 1] and rewards are averaged within length quantile buckets. σ<sub>len</sub> denotes the standard deviation of bucket-wise mean rewards.

Sycophantic bias. In open-ended dialogue, sycophantic bias refers to a model’s tendency to produce responses that agree with the user rather than providing reliable or truthful answers. Following Liu et al. (2024) and Wang et al. (2025), we first train a hacked SFT model by prepending the prefix “Sure, here is the response: ” to assistant messages with probability $\scriptstyle { p = 0 . 8 } .$ Starting from this SFT model (which exhibits a preference for the sycophantic phrasing), we then construct a hacked version of the Anthropic-HH preference training set by prepending the same prefix to the chosen response with $p _ { \mathrm { c h o s e n } } = 0 . 8$ and to the rejected response with $p _ { \mathrm { r e j e c t e d } } = 0 . 2$ , and train reward models on this perturbed dataset. For evaluation, we perturb the test split by prepending the prefix to both chosen and rejected responses with $p = 0 . 3$ , and report pairwise accuracy of reward models on each benchmark. Table 4 summarizes the results and also reports the performance change relative to the corresponding reward model trained on the unperturbed dataset. Overall, CausalRM is more robust to the sycophantic-phrasing artifact: it achieves the best ID and OOD accuracies on the hacked tests with only a minor average drop of −1.7 points and −1.1 points, respectively. In contrast, baselines degrade much more (e.g., Standard RM drops by −11.4 on ID and −5.8 on OOD), indicating that CausalRM is less likely to exploit the spurious prefix.

Table 4. Robustness to sycophantic-phrasing artifacts measured by pairwise accuracy on hacked test sets. Each cell reports accuracy, with the relative change compared to the corresponding model trained on the unperturbed dataset shown in parentheses.

<table><tr><td rowspan="2">Method</td><td colspan="3">ID</td><td colspan="5">OOD</td></tr><tr><td>Helpful</td><td>Harmless</td><td>Avg.</td><td>MT-Bench</td><td>PKU-SafeRLHF</td><td>SHP</td><td>TruthfulQA</td><td>Avg.</td></tr><tr><td>Standard RM</td><td>56.2(-11.2)</td><td>62.1(-11.7)</td><td>59.2(-11.4)</td><td>56.2(-12.0)</td><td>54.6(-3.2)</td><td>46.5(-7.7)</td><td>58.3(-0.2)</td><td>53.9(-5.8)</td></tr><tr><td>GoalRM</td><td>60.0(-7.5)</td><td>64.3(-10.4)</td><td>62.2(-8.9)</td><td>59.6(-6.6)</td><td>55.3(-4.0)</td><td>50.7(-3.1)</td><td>61.9(-1.7)</td><td>56.9(-3.8)</td></tr><tr><td>InfoRM</td><td>63.7(-4.2)</td><td>69.8(-3.9)</td><td>66.8(-4.0)</td><td>61.4(-5.3)</td><td>57.4(-2.7)</td><td>50.0(-0.4)</td><td>60.2(-1.7)</td><td>57.3(-2.5)</td></tr><tr><td>CausalRM (Ours)</td><td>67.4(-2.0)</td><td>73.8(-1.4)</td><td>70.6(-1.7)</td><td>65.7(-2.6)</td><td>62.0(+1.1)</td><td>50.3(-3.6)</td><td>66.8(+0.6)</td><td>61.2(-1.1)</td></tr></table>

## 5. Related Work

## 5.1. Reward hacking in RLHF

Reward hacking (Amodei et al., 2016; Skalse et al., 2022; Gao et al., 2023b) remains a central challenge for aligning LLMs with human preferences via RLHF. A primary cause of this phenomenon is that reward models often exploit spurious correlations in the training data to gain maximum benefit without truly capturing the underlying intent of human judgments (Eisenstein et al., 2023). This behavior, also known as goal misgeneralization (Di Langosco et al., 2022) or shortcut learning (Geirhos et al., 2020) in traditional reinforcement learning (RL), typically arises when the reward model erroneously associates high rewards with non-causal attributes such as response length (Dubois et al., 2024), formatting cues (Chen et al., 2024), sycophantic agreement (Perez et al., 2023), or superficial conceptual patterns (Zhou et al., 2023). A growing body of work proposes to mitigate reward hacking through techniques including but are not limited to data augmentation (Liu et al., 2024), reward ensembles (Coste et al., 2023), reward shaping (Fu et al., 2025), and representation learning (Nath et al., 2024; Miao et al., 2024). Our work investigates the feasibility of addressing reward hacking from the perspective of causal representation learning, aiming to explicitly separate reward-relevant factors from spurious ones during reward modeling.

## 5.2. Causal representation learning.

In traditional RL, causal representation learning aims to extract high-level causal variables from low-level observations that are both minimal and sufficient for policy learning (Schölkopf et al., 2021). For example, ASR (Huang et al., 2022) and IFactor (Liu et al., 2023) learn more accurate world models by disentangling the most predictive features in environment dynamics. AdaRL (Huang et al., 2021) and CSR (Yang et al., 2024b) further extend causal factorization to domain adaptation, leveraging causal variables to capture environment changes. Moreover, works like Zheng & Makar (2022) and Steinmann et al. (2024) improve robustness to distribution shifts by explicitly identifying and removing spurious correlations that act as shortcuts.

Motivated by these insights, several recent approaches have applied causal principles to mitigate reward hacking in RLHF. RMM (Liu et al., 2024) proposes a data augmentation strategy grounded in causal invariance to eliminate context-free artifacts. Ovinnikov et al. (2024) and Wang et al. (2025) formalize this invariance as an explicit regularization term during reward model training. CRA (Song et al., 2025) employs backdoor adjustments to deconfound spurious associations, while DEPTH (Yang et al., 2025) adopts a fixed, template-based factorization to filter out irrelevant information, yielding reward models tailored to relation extraction. CausalRM differs from prior approaches by learning factorized latent representations without taskspecific modifications, providing a general mechanism to disentangle reward-relevant signals from spurious attributes.

## 6. Conclusion

In this paper, we propose CausalRM, a novel reward modeling framework that addresses reward hacking in RLHF through factored representation learning motivated by the causal invariance principle. By decomposing the backbone embedding into causal and non-causal latent factors, CausalRM enforces that reward prediction depends solely on the minimal sufficient causal component, while actively suppressing reward-relevant signals in the non-causal part via an adversarial head with gradient reversal. Extensive experiments on mathematical reasoning and open-ended dialogue demonstrate that CausalRM improves both reward model accuracy and downstream RLHF performance, while significantly mitigating sensitivity to spurious attributes such as response length and sycophantic bias. Future work includes extending CausalRM to process reward modeling scenarios and exploring its application in multi-turn dialogues with dynamic confounders.

## Acknowledgements

This work was supported by Alibaba Group through Alibaba Innovative Research Program, the Science and Technology Commission of Shanghai Municipality (Grant No. 24510714300), and the Shanghai Municipal Science and Technology Major Project, China (Grant No. 2021SHZDZX0102).

## Impact Statement

This paper presents a method for improving reward model robustness in RLHF by reducing reliance on spurious correlations in training data (e.g., response length or stylistic cues). More robust reward modeling may help make RLHF training more stable and improve generalization across evaluation settings. This work is a technical contribution to reward modeling and does not introduce new application domains or new data collection involving human subjects. As with RLHF methods generally, outcomes in deployed systems will depend on the quality and representativeness of the preference data and on the surrounding safety measures. We do not anticipate additional societal risks beyond those already associated with training and deploying large language models and RLHF-based alignment systems.

## References

Alemi, A. A., Fischer, I., Dillon, J. V., and Murphy, K. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016.

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Mané, D. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.

Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Bowman, S., Vilnis, L., Vinyals, O., Dai, A., Jozefowicz, R., and Bengio, S. Generating sentences from a continuous space. In Proceedings of the 20th SIGNLL conference on computational natural language learning, pp. 10–21, 2016.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Bühlmann, P. Invariance, causality and robustness. Statistical Science, 35(3):404–426, 2020.

Chen, L., Zhu, C., Soselia, D., Chen, J., Zhou, T., Goldstein, T., Huang, H., Shoeybi, M., and Catanzaro, B. Odin: Disentangled reward mitigates hacking in rlhf. arXiv preprint arXiv:2402.07319, 2024.

Chen, Y., Wang, R., Jiang, H., Shi, S., and Xu, R. Exploring the use of large language models for reference-free text quality evaluation: An empirical study. arXiv preprint arXiv:2304.00723, 2023.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.

Coste, T., Anwar, U., Kirk, R., and Krueger, D. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

Di Langosco, L. L., Koch, J., Sharkey, L. D., Pfau, J., and Krueger, D. Goal misgeneralization in deep reinforcement learning. In International Conference on Machine Learning, pp. 12004–12019. PMLR, 2022.

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Fu, J., Zhao, X., Yao, C., Wang, H., Han, Q., and Xiao, Y. Reward shaping to mitigate reward hacking in rlhf. arXiv preprint arXiv:2502.18770, 2025.

Ganin, Y. and Lempitsky, V. Unsupervised domain adaptation by backpropagation. In International conference on machine learning, pp. 1180–1189. PMLR, 2015.

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. Pal: Program-aided language models. In International Conference on Machine Learning, pp. 10764–10799. PMLR, 2023a.

Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023b.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

He-Yueya, J., Poesia, G., Wang, R. E., and Goodman, N. D. Solving math word problems by combining language models with symbolic solvers. arXiv preprint arXiv:2304.09102, 2023.

Hu, J., Wu, X., Zhu, Z., Xianyu, Wang, W., Zhang, D., and Cao, Y. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

Huang, B., Feng, F., Lu, C., Magliacane, S., and Zhang, K. Adarl: What, where, and how to adapt in transfer reinforcement learning. arXiv preprint arXiv:2107.02729, 2021.

Huang, B., Lu, C., Leqi, L., Hernández-Lobato, J. M., Glymour, C., Schölkopf, B., and Zhang, K. Action-sufficient state representation learning for control with structural constraints. In International Conference on Machine Learning, pp. 9260–9279. PMLR, 2022.

Ji, J., Liu, M., Dai, J., Pan, X., Zhang, C., Bian, C., Chen, B., Sun, R., Wang, Y., and Yang, Y. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. Advances in Neural Information Processing Systems, 36:24678–24704, 2023.

Koncel-Kedziorski, R., Roy, S., Amini, A., Kushman, N., and Hajishirzi, H. Mawps: A math word problem repository. In Proceedings of the 2016 conference of the north american chapter ofthe associationfor computational linguistics: human language technologies, pp. 1152–1157, 2016.

Kong, L., Xie, S., Yao, W., Zheng, Y., Chen, G., Stojanov, P., Akinwande, V., and Zhang, K. Partial identifiability for domain adaptation. arXiv preprint arXiv:2306.06510, 2023.

Lambert, N., Pyatkin, V., Morrison, J., Miranda, L. J. V., Lin, B. Y., Chandu, K., Dziri, N., Kumar, S., Zick, T., Choi, Y., et al. Rewardbench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 1755– 1797, 2025.

Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pp. 3214–3252, 2022.

Liu, T., Xiong, W., Ren, J., Chen, L., Wu, J., Joshi, R., Gao, Y., Shen, J., Qin, Z., Yu, T., et al. Rrm: Robust reward model training mitigates reward hacking. arXiv preprint arXiv:2409.13156, 2024.

Liu, Y., Huang, B., Zhu, Z., Tian, H., Gong, M., Yu, Y., and Zhang, K. Learning world models with identifiable factorization. Advances in Neural Information Processing Systems, 36:31831–31864, 2023.

Miao, S.-Y., Liang, C.-C., and Su, K.-Y. A diverse corpus for evaluating and developing english math word problem solvers. In Proceedings ofthe 58th annual meeting ofthe Association for Computational Linguistics, pp. 975–984, 2020.

Miao, Y., Zhang, S., Ding, L., Bao, R., Zhang, L., and Tao, D. Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling. Advances in Neural Information Processing Systems, 37:134387–134429, 2024.

Miao, Y., Ding, L., Zhang, S., Bao, R., Zhang, L., and Tao, D. Information-theoretic reward modeling for stable rlhf: Detecting and mitigating reward hacking. arXiv preprint arXiv:2510.13694, 2025a.

Miao, Y., Zhang, S., Ding, L., Zhang, Y., Zhang, L., and Tao, D. The energy loss phenomenon in rlhf: A new perspective on mitigating reward hacking. arXiv preprint arXiv:2501.19358, 2025b.

Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, Z., Paul, W., Jordan, M. I., et al. Ray: A distributed framework for emerging {AI} applications. In 13th USENIX symposium on operating systems design and implementation (OSDI 18), pp. 561– 577, 2018.

Nath, V., Slack, D., Da, J., Ma, Y., Zhang, H., Whitehead, S., and Hendryx, S. Learning goal-conditioned representations for language reward models. Advances in Neural Information Processing Systems, 37:117070–117108, 2024.

Ng, I., Blöbaum, P., Bhandari, S., Zhang, K., and Kasiviswanathan, S. Debiasing reward models by representation learning with guarantees. arXiv preprint arXiv:2510.23751, 2025.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Ovinnikov, I., Bykovets, E., and Buhmann, J. M. Learning causally invariant reward functions from diverse demonstrations. arXiv preprint arXiv:2409.08012, 2024.

Park, R., Rafailov, R., Ermon, S., and Finn, C. Disentangling length from quality in direct preference optimization. arXiv preprint arXiv:2403.19159, 2024.

Patel, A., Bhattamishra, S., and Goyal, N. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191, 2021.

Perez, E., Ringer, S., Lukosiute, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., et al. Discovering language model behaviors with modelwritten evaluations. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13387–13434, 2023.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.

Rafailov, R., Chittepu, Y., Park, R., Sikchi, H. S., Hejna, J., Knox, B., Finn, C., and Niekum, S. Scaling laws for reward model overoptimization in direct alignment algorithms. Advances in Neural Information Processing Systems, 37:126207–126242, 2024.

Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. Proceedings ofthe IEEE, 109(5): 612–634, 2021.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., et al. Towards understanding sycophancy in language models. arXiv preprint arXiv:2310.13548, 2023.

Singhal, P., Goyal, T., Xu, J., and Durrett, G. A long way to go: Investigating length correlations in rlhf. arXiv preprint arXiv:2310.03716, 2023.

Skalse, J., Howe, N., Krasheninnikov, D., and Krueger, D. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

Song, R., Song, Z., Guo, H., and Qiang, W. Causal reward adjustment: Mitigating reward hacking in external reasoning via backdoor correction. arXiv preprint arXiv:2508.04216, 2025.

Steinmann, D., Divo, F., Kraus, M., Wüst, A., Struppek, L., Friedrich, F., and Kersting, K. Navigating shortcuts, spurious correlations, and confounders: From origins via detection to mitigation. arXiv preprint arXiv:2412.05152, 2024.

Toshniwal, S., Moshkov, I., Narenthiran, S., Gitman, D., Jia, F., and Gitman, I. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. Advances in Neural Information Processing Systems, 37:34737–34774, 2024.

Veitch, V., D’Amour, A., Yadlowsky, S., and Eisenstein, J. Counterfactual invariance to spurious correlations: Why and how to pass stress tests. arXiv preprint arXiv:2106.00545, 2021.

Wang, C., Zhao, Z., Jiang, Y., Chen, Z., Zhu, C., Chen, Y., Liu, J., Zhang, L., Fan, X., Ma, H., et al. Beyond reward hacking: Causal rewards for large language model alignment. arXiv preprint arXiv:2501.09620, 2025.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement. arXiv preprint arXiv:2409.12122, 2024a.

Yang, Y., Huang, B., Feng, F., Wang, X., Tu, S., and Xu, L. Towards generalizable reinforcement learning via causality-guided self-adaptive representations. arXiv preprint arXiv:2407.20651, 2024b.

Yang, Y., Feng, F., Yang, L., Deng, W., Qu, L., Huang, B., Tu, S., and Xu, L. Depth: Hallucination-free relation extraction via dependency-aware sentence simplification and two-tiered hierarchical refinement. arXiv preprint arXiv:2508.14391, 2025.

Zheng, J. and Makar, M. Causally motivated multi-shortcut identification and removal. Advances in Neural Information Processing Systems, 35:12800–12812, 2022.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36: 46595–46623, 2023.

Zhou, Y., Xu, P., Liu, X., An, B., Ai, W., and Huang, F. Explore spurious correlations at the concept level in language models for text classification. arXiv preprint arXiv:2311.08648, 2023.

Zhou, Y., Xu, P., Liu, X., An, B., Ai, W., and Huang, F. Explore spurious correlations at the concept level in language models for text classification. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 478–492, 2024.

Zhou, Y., Liu, H., Chen, Z., Tian, Y., and Chen, B. Gsminfinite: How do your llms behave over infinitely increasing context length and reasoning complexity? arXiv preprint arXiv:2502.05252, 2025.

## A. Algorithm

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Training CausalRM
Require: Preference dataset $\mathcal{D} = \{(x, y^w, y^l)\}$, hyperparameters $\lambda_{\mathrm{KL}}^c, \lambda_{\mathrm{adv}}, \lambda_{\mathrm{KL}}^{nc}, \lambda_{\mathrm{rec}}$
Ensure: Trained parameters $\theta = (\phi, \alpha, \psi, \eta, \omega)$
1: for each batch from $\mathcal{D}$ do
2: Encode $(x, y^w)$ and $(x, y^l)$ to get $h^w, h^l$ via the LLM backbone $f_\phi$
3: Sample latent pairs $(z^{w,c}, z^{w,nc})$ and $(z^{l,c}, z^{l,nc})$ using encoder $q_\alpha(\cdot | h)$
4: Compute reward predictions $\hat{r} = g_\psi(z^c)$ and adversarial predictions $\hat{r}^{\mathrm{adv}} = a_\omega(z^{nc})$
5: Reconstruct embeddings $\hat{h} = d_\eta([z^c; z^{nc}])$
6: Compute total loss $\mathcal{L}_{\mathrm{total}}$ as in Eq. (10)
7: Apply gradient reversal to $\mathcal{L}_{\mathrm{adv}}$ during backpropagation
8: Update all parameters $\theta$ jointly by optimizing $\mathcal{L}_{\mathrm{total}}$
9: end for
</div>

## B. Implementation Details

Model Architectures. CausalRM is implemented as a lightweight latent-variable module on top of a pretrained LLM backbone. Given the backbone embedding $\textit { h } \in \mathbb { R } ^ { H }$ , we map it to two diagonal-Gaussian posteriors: $q _ { \alpha } ( z ^ { c } \mid h ) =$ $\mathcal { N } ( \mu _ { c } ( h ) , \mathrm { d i a g } ( \sigma _ { c } ^ { 2 } ( h ) ) )$ and $q _ { \alpha } ( z ^ { n c } \mid h ) = \mathcal N ( \mu _ { n c } ( h )$ , diag $( \sigma _ { n c } ^ { 2 } ( h ) )$ ). We then sample $z ^ { c }$ and $z ^ { n c }$ via the reparameterization trick. Reward prediction depends only on $z ^ { c }$ through a linear reward head, while the non-causal latent $z ^ { n c }$ is fed into an adversarial head with a gradient reversal layer (GRL). A reconstruction decoder maps the concatenated latent $[ z ^ { c } ; z ^ { n c } ]$ back to ${ \hat { h } } .$ At inference time, we use the mean $\mu _ { c }$ (instead of a stochastic sample) for stable reward prediction.

The GRL is implemented as a custom autograd function: in the forward pass it is the identity map, while in the backward pass it multiplies the gradient by $- \lambda _ { \mathrm { g r l } }$ . Table 5 summarizes the architectural choices.

Table 5. CausalRM architecture details used in our experiments. H is the backbone hidden size.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Component Specification
Causal posterior  $\mu_{c}, \log \sigma_{c}^{2}: linear H \rightarrow d_{c}$
Non-causal posterior  $\mu_{nc}, \log \sigma_{nc}^{2}: linear H \rightarrow d_{nc}$
Latent dims  $d_{c} = 128, d_{nc} = 512$
Sampling Reparameterization  $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
Reward head Linear  $d_{c} \rightarrow 1$  (no bias), input:  $z_{c}$  (train) /  $\mu_{c}$  (eval)
Adversary head Linear  $d_{nc} \rightarrow 1$  (no bias), input: GRL( $z_{nc}$ )
Reconstructor Linear ( $d_{c} + d_{nc}$ ) → H
</div>

Backbone Tuning Strategy. The backbone embedding h corresponds to the final-layer hidden state of the last non-padding token. During reward model training, the pretrained LLM backbone is fully fine-tuned jointly with all CausalRM heads. We do not use parameter freezing or LoRA adapters.<sup>1</sup>

OOD Preference Data Construction. All evaluation datasets are publicly available, and no additional label construction is performed. For mathematical reasoning, candidate responses are sourced from the GoalRM dataset (Nath et al., 2024), where preference pairs are formed by pairing correct (chosen) and incorrect (rejected) solutions generated via CodeLlama 7B greedy decoding.<sup>2</sup> For open-ended dialogue, candidates are drawn from the RewardBench (Lambert et al., 2025) preference test sets.<sup>3</sup> For TruthfulQA, we use the publicly available preference-formatted version, which pairs each question with verified correct and incorrect reference answers.<sup>4</sup> Labels strictly follow ground-truth verification or original dataset annotations, with no LLM judge involved in the OOD reward model evaluation.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$^{1}$ Code is available at https://github.com/CMACH508/CausalRM
 $^{2}$ https://github.com/vaskar-nath/goal-conditioned-rm/blob/master/examples/data/preference_ranking_dataset.zip
 $^{3}$ https://huggingface.co/datasets/allenai/preference-test-sets
 $^{4}$ https://huggingface.co/datasets/domenicrosati/TruthfulQA
</div>

LLM Judge Configuration. For downstream dialogue evaluation, we use Qwen3-Max as an external pairwise judge with deterministic decoding: temperature 0.0, top\_p 1.0, and max\_new\_tokens 512. The exact evaluation prompt is provided in Appendix C.

Training cost. Both reward model training and PPO optimization are conducted on a single compute node equipped with 8 NVIDIA H20 GPUs (96GB VRAM each), 192 CPU cores, and 1.8 TB RAM. We employ Ray-based distributed training (Moritz et al., 2018) during RLHF. Table 6 summarizes the approximate training time in our experiments.

Table 6. Approximate training time of CausalRM (in hours) for reward model (RM) and PPO stages.

<table><tr><td>Task</td><td>RM Training</td><td>PPO Training</td></tr><tr><td>Mathematical reasoning</td><td>6.5</td><td>26.1</td></tr><tr><td>Open-ended dialogue</td><td>4.0</td><td>22.3</td></tr></table>

Hyperparameters. Our implementation is based on the OpenRLHF library (v0.8.5). Unless otherwise stated, we use the same training recipe for both mathematical reasoning and dialogue experiments. Reward model and PPO hyperparameters are summarized in Tables 7 and 8, respectively.

Table 7. Reward model training hyperparameters.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Max sequence length</td><td>1024</td></tr><tr><td>Train batch size / micro-batch</td><td>256 / 1</td></tr><tr><td>Learning rate</td><td> $9 \times 10^{-6}$ </td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Latent dims ( $d_c$ ,  $d_{nc}$ )</td><td>(128, 512)</td></tr><tr><td> $\lambda_{\text{pred}}$  (pref. loss)</td><td>1.0</td></tr><tr><td> $\lambda_{\text{rec}}$  (reconstruction)</td><td>0.001</td></tr><tr><td> $\lambda_{\text{adv}}$  (adversarial)</td><td>0.05</td></tr><tr><td> $\lambda_{\text{KL}}^c$  (causal KL)</td><td>0.001</td></tr><tr><td> $\lambda_{\text{KL}}^{nc}$  (non-causal KL)</td><td>0.001</td></tr><tr><td>GRL strength  $\lambda_{\text{grl}}$ </td><td>1.0</td></tr></table>

Table 8. PPO hyperparameters.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Prompt max length</td><td>1024</td></tr><tr><td>Generation max length</td><td>1024</td></tr><tr><td>Train batch size / micro-batch</td><td>64 / 8</td></tr><tr><td>Rollout batch size / micro-rollout</td><td>512 / 16</td></tr><tr><td>Actor learning rate</td><td> $5 \times 10^{-7}$ </td></tr><tr><td>Critic learning rate</td><td> $9 \times 10^{-6}$ </td></tr><tr><td>Reward normalization</td><td>enabled</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Colocation</td><td>critic+reward, actor+ref</td></tr><tr><td>vLLM #engines / tensor parallel</td><td>2 / 2</td></tr><tr><td>vLLM GPU memory utilization</td><td>0.95</td></tr></table>

Baseline Implementations. For all baselines, we strictly follow the official configurations and training recipes. InfoRM uses an information bottleneck dimension of 128 and a KL coefficient of 0.1, as recommended in its Appendix E.2. GoalRM adopts the default hyperparameters reported in its Table 4. We note that InfoRM exhibits a severe performance drop during mathematical RLHF. Our diagnostic analysis reveals a formatting failure mode: as PPO progresses, the InfoRM-trained policy frequently generates truncated or incomplete solutions that omit the required \boxed{} final-answer format. Consequently, a large fraction of outputs lack parsable answers, which plausibly explains the rapid accuracy collapse. This behavior is supported by Figure 3 and Figure 5, and we include qualitative examples in Figure 9.

## C. Prompt for Qwen3-Max Evaluation

We use Qwen3-Max as an external judge to perform pairwise comparisons between model responses in our dialogue experiments. Given an instruction and two candidate outputs, Qwen3-Max is asked to rank the two models by the quality of their responses, reflecting the preference that the majority of humans would give. The exact prompt used for evaluation is shown below.

## Qwen3-Max Evaluation Prompt

```txt
I want you to create a leaderboard of different large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by human. All inputs and outputs should be Python objects.
Here is the prompt:
{{“instruction”: instruction}}
Here are the outputs of the models:
[
    {{ "model": model1, "answer": output1}},    {{ "model": model2, "answer": output2}}]
]
Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{“model”: "<model-name>", “rank”: <model-rank>}},    {{“model”: "<model-name>", “rank”: <model-rank>}}
]
Your response must be a valid Python object and should contain nothing else because we will directly use it in Python. Please provide the ranking that the majority of humans would give.
```

## D. Derivation of the Minimal Sufficiency Objective

In this section, we derive a variational lower bound for Eq. (11), following Alemi et al. (2016) and Miao et al. (2024).

Recall that our minimal sufficiency objective for the causal latent $z ^ { c }$ is

$$
\max \quad I (z ^ {c}; r) - \lambda_ {\mathrm{KL}} ^ {c} I (h; z ^ {c}),\tag{16}
$$

where $h = f _ { \phi } ( x , y )$ is the backbone embedding and r denotes the preference signal induced by human rankings.

Step 1: A variational lower bound for $I ( z ^ { c } ; r )$ . By definition,

$$
I (z ^ {c}; r) = \mathbb {E} _ {p (z ^ {c}, r)} \biggl [ \log \frac {p (r \mid z ^ {c})}{p (r)} \biggr ] = \mathbb {E} _ {p (z ^ {c}, r)} [ \log p (r \mid z ^ {c}) ] + H (r),\tag{17}
$$

where $H ( r )$ does not depend on model parameters. Introducing an auxiliary variational distribution $q ( r \mid z ^ { c } )$ to approximate $p ( r \mid z ^ { c } )$ and applying Gibbs’ inequality yields

$$
I (z ^ {c}; r) \geq \mathbb {E} _ {p (z ^ {c}, r)} [ \log q (r \mid z ^ {c}) ] + H (r).\tag{18}
$$

Dropping the constant $H ( r )$ , we maximize $\mathbb { E } _ { p ( z ^ { c } , r ) } [ \log q ( r \mid z ^ { c } ) ]$

In our reward modeling setup, preferences are given as pairwise comparisons. For each triplet $( x , y ^ { w } , y ^ { l } ) \sim \mathcal { D }$ , let $h ^ { w } = f _ { \phi } ( x , y ^ { w } )$ and $h ^ { l } = f _ { \phi } ( x , y ^ { l } )$ , and sample

$$
z ^ {w, c} \sim q _ {\alpha} (z ^ {c} \mid h ^ {w}), \qquad z ^ {l, c} \sim q _ {\alpha} (z ^ {c} \mid h ^ {l}).\tag{19}
$$

We instantiate the variational likelihood with a Bradley–Terry model:

$$
q (r \mid z ^ {c}) = q (y ^ {w} \succ y ^ {l} \mid z ^ {w, c}, z ^ {l, c}) = \sigma \big (g _ {\psi} (z ^ {w, c}) - g _ {\psi} (z ^ {l, c}) \big),\tag{20}
$$

where $\sigma ( \cdot )$ is the sigmoid function. Taking the log-likelihood gives

$$
\log q (y ^ {w} \succ y ^ {l} \mid z ^ {w, c}, z ^ {l, c}) = \log \sigma \big (g _ {\psi} (z ^ {w, c}) - g _ {\psi} (z ^ {l, c}) \big).\tag{21}
$$

Thus, maximizing the lower bound in Eq. (18) corresponds to minimizing the standard preference loss $\mathcal { L } _ { \mathrm { p r e f } }$ in the main paper.

Step 2: A variational upper bound for $I ( h ; z ^ { c } )$ . Let the joint distribution be defined by the data marginal $p ( h )$ and the encoder $q _ { \alpha } ( z ^ { c } \mid h ) , \mathrm { i . e . , } p _ { \mathrm { t r u e } } ( h , z ^ { c } ) \triangleq p ( h ) q _ { \alpha } ( z ^ { c } \mid h )$ . Then the mutual information can be written as

$$
I (h; z ^ {c}) = \mathbb {E} _ {p (h)} \left[ \mathrm{KL} \bigl (q _ {\alpha} (z ^ {c} \mid h) \| p _ {\text { true }} (z ^ {c}) \bigr) \right],\tag{22}
$$

where $\begin{array} { r } { p _ { \mathrm { t r u e } } ( z ^ { c } ) = \int q _ { \alpha } ( z ^ { c } \mid h ) p ( h ) } \end{array}$ dh is the aggregated posterior, which is generally intractable. Following the variational information bottleneck (Alemi et al., 2016), we upper-bound $I ( h ; z ^ { c } )$ by replacing $p _ { \mathrm { t r u e } } ( z ^ { c } )$ with a tractable variational prior $p ( z ^ { c } ) = \mathcal { N } ( 0 , I )$ . Since $\mathrm { K L } ( p _ { \mathrm { t r u e } } ( z ^ { c } ) \| p ( z ^ { c } ) ) \geq 0$ , we have $\mathbb { E } _ { p _ { \mathrm { t r u e } } ( z ^ { c } ) } [ \log p _ { \mathrm { t r u e } } ( z ^ { c } ) ] \geq \mathbb { E } _ { p _ { \mathrm { t r u e } } ( z ^ { c } ) } [ \log p ( z ^ { c } ) ]$ , which implies

$$
\begin{array}{r l} & I (h; z ^ {c}) = \mathbb {E} _ {p (h)} \mathbb {E} _ {q _ {\alpha} (z ^ {c} | h)} [ \log q _ {\alpha} (z ^ {c} \mid h) ] - \mathbb {E} _ {p _ {\mathrm{true}} (z ^ {c})} [ \log p _ {\mathrm{true}} (z ^ {c}) ] \\ & \quad \leq \mathbb {E} _ {p (h)} \mathbb {E} _ {q _ {\alpha} (z ^ {c} | h)} [ \log q _ {\alpha} (z ^ {c} \mid h) ] - \mathbb {E} _ {p (h)} \mathbb {E} _ {q _ {\alpha} (z ^ {c} | h)} [ \log p (z ^ {c}) ] \\ & \quad = \mathbb {E} _ {p (h)} \big [ \mathrm{KL} \big (q _ {\alpha} (z ^ {c} \mid h) \| p (z ^ {c}) \big) \big ]. \end{array}\tag{23}
$$

For a preference pair $( x , y ^ { w } , y ^ { l } )$ , we use both embeddings and obtain the empirical estimate

$$
\mathrm{KL} (q _ {\alpha} (z ^ {c} \mid h ^ {w}) \parallel p (z ^ {c})) + \mathrm{KL} \big (q _ {\alpha} (z ^ {c} \mid h ^ {l}) \parallel p (z ^ {c}) \big)  ,\tag{24}
$$

which corresponds to $\mathcal { L } _ { \mathrm { K I } } ^ { c }$ in Eq. (12).

Step 3: Putting the bounds together. Combining the variational lower bound for $I ( z ^ { c } ; r ) ( \mathrm { E q }$ . 18) and the variationa upper bound for $I ( h ; z ^ { c } ) ( \mathrm { E q . } 2 3 )$ ), and using the Bradley–Terry likelihood in Eq. (21), we obtain the following variational lower bound for Eq. (16):

$$
\begin{array}{r l} I (z ^ {c}; r) - \lambda_ {\mathrm{KL}} ^ {c} I (h; z ^ {c}) \geq & \mathbb {E} _ {(x, y ^ {w}, y ^ {l}) \sim \mathcal {D}} \Big [ \log \sigma \big (g _ {\psi} (z ^ {w, c}) - g _ {\psi} (z ^ {l, c}) \big) \\ & - \lambda_ {\mathrm{KL}} ^ {c} \Big (\mathrm{KL} (q _ {\alpha} (z ^ {c} \mid h ^ {w}) \parallel p (z ^ {c})) + \mathrm{KL} \big (q _ {\alpha} (z ^ {c} \mid h ^ {l}) \parallel p (z ^ {c}) \big) \Big) \Big ]. \end{array}\tag{25}
$$

Identifying

$$
\mathcal {L} _ {\mathrm{pref}} = - \log \sigma \big (r _ {\theta} (x, y ^ {w}) - r _ {\theta} (x, y ^ {l}) \big), \qquad \mathcal {L} _ {\mathrm{KL}} ^ {c} = \mathrm{KL} (q _ {\alpha} (z ^ {c} \mid h ^ {w}) \parallel p (z ^ {c})) + \mathrm{KL} \big (q _ {\alpha} (z ^ {c} \mid h ^ {l}) \parallel p (z ^ {c}) \big)  ,\tag{26}
$$

and using $r _ { \theta } ( x , y ) = g _ { \psi } ( z ^ { c } )$ ) with $z ^ { c } \sim q _ { \alpha } ( z ^ { c } \mid f _ { \phi } ( x , y ) )$ ) yields the training objective in Eq. (12) in the main paper. This completes the derivation.

Table 9. Ablation on mathematical reasoning reward modeling. We report pairwise accuracy (%, higher is better) on ID and OOD benchmarks. Numbers in parentheses are deltas relative to full CausalRM.

<table><tr><td rowspan="2">Variant</td><td colspan="3">ID</td><td colspan="6">OOD</td></tr><tr><td>GSM8K</td><td>MATH</td><td>Avg.</td><td>Algebra222</td><td>GSM-Hard</td><td>ASDiv</td><td>MAWPS</td><td>SVAMP</td><td>Avg.</td></tr><tr><td>CausalRM</td><td>81.7</td><td>58.4</td><td>70.1 (+0.0)</td><td>89.9</td><td>66.2</td><td>89.9</td><td>92.2</td><td>89.6</td><td>85.6 (+0.0)</td></tr><tr><td>w/o factorization</td><td>74.1</td><td>58.0</td><td>66.1 (-4.0)</td><td>82.7</td><td>63.5</td><td>88.9</td><td>89.9</td><td>87.6</td><td>82.5 (-3.1)</td></tr><tr><td>w/o reconstruction</td><td>78.8</td><td>58.1</td><td>68.5 (-1.6)</td><td>87.7</td><td>64.5</td><td>87.3</td><td>88.7</td><td>87.4</td><td>83.1 (-2.5)</td></tr><tr><td>w/o adversarial / GRL</td><td>76.4</td><td>57.6</td><td>67.0 (-3.1)</td><td>84.4</td><td>61.9</td><td>87.1</td><td>88.5</td><td>85.6</td><td>81.5 (-4.1)</td></tr><tr><td>w/o KL on  $z^c$ </td><td>78.0</td><td>56.5</td><td>67.3 (-2.8)</td><td>84.4</td><td>60.3</td><td>86.2</td><td>87.9</td><td>86.9</td><td>81.1 (-4.5)</td></tr><tr><td>w/o KL on  $z^{nc}$ </td><td>72.7</td><td>55.3</td><td>64.0 (-6.1)</td><td>83.3</td><td>63.0</td><td>86.4</td><td>88.7</td><td>86.8</td><td>81.6 (-4.0)</td></tr><tr><td>w/o KL on both</td><td>71.3</td><td>49.4</td><td>60.4 (-9.7)</td><td>77.9</td><td>55.4</td><td>78.5</td><td>81.7</td><td>79.4</td><td>74.6 (-11.0)</td></tr></table>

Table 10. Ablation on robustness to sycophantic-phrasing artifacts (pairwise accuracy on hacked test sets; %, higher is better). Numbers in parentheses are deltas relative to full CausalRM.

<table><tr><td rowspan="2">Variant</td><td colspan="3">ID</td><td colspan="5">OOD</td></tr><tr><td>Helpful</td><td>Harmless</td><td>Avg.</td><td>MT-Bench</td><td>PKU-SafeRLHF</td><td>SHP</td><td>TruthfulQA</td><td>Avg.</td></tr><tr><td>CausalRM</td><td>67.4</td><td>73.8</td><td>70.6 (+0.0)</td><td>65.7</td><td>62.0</td><td>50.3</td><td>66.8</td><td>61.2 (+0.0)</td></tr><tr><td>w/o factorization</td><td>63.7</td><td>69.8</td><td>66.8 (-3.8)</td><td>61.4</td><td>57.4</td><td>50.0</td><td>60.2</td><td>57.3 (-3.9)</td></tr><tr><td>w/o reconstruction</td><td>66.8</td><td>73.5</td><td>70.2 (-0.4)</td><td>63.1</td><td>59.2</td><td>48.0</td><td>64.2</td><td>58.6 (-2.6)</td></tr><tr><td>w/o adversarial / GRL</td><td>62.9</td><td>71.6</td><td>67.3 (-3.3)</td><td>61.9</td><td>59.6</td><td>46.7</td><td>60.7</td><td>57.2 (-4.0)</td></tr><tr><td>w/o KL on  $z^c$ </td><td>67.0</td><td>73.2</td><td>70.1 (-0.5)</td><td>62.3</td><td>58.2</td><td>46.1</td><td>61.9</td><td>57.1 (-4.1)</td></tr><tr><td>w/o KL on  $z^{nc}$ </td><td>62.7</td><td>65.0</td><td>63.9 (-6.7)</td><td>63.7</td><td>60.0</td><td>48.5</td><td>63.9</td><td>59.0 (-2.2)</td></tr><tr><td>w/o KL on both</td><td>46.4</td><td>45.5</td><td>46.0 (-24.6)</td><td>43.3</td><td>44.4</td><td>43.1</td><td>61.1</td><td>48.0 (-13.2)</td></tr></table>

## E. Ablation Studies

This section ablates the key design choices in CausalRM to better understand which components are essential for robust reward modeling and, in particular, for mitigating spurious correlations. Since our primary goal is to improve reward model robustness, we focus on reward model evaluation: (1) pairwise accuracy on ID/OOD benchmarks for mathematical reasoning, (2) length-sensitivity analysis on math, and (3) robustness to sycophantic artifacts on open-ended dialogue via the hacked evaluation described in Section 4.3.

Settings. Starting from the full CausalRM objective in Eq. (10), we consider the following variants:

• w/o factorization: remove the latent factorization and train a single latent z with the preference loss and the KL bottleneck on the reward-predictive latent (i.e., only $\mathcal { L } _ { \mathrm { p r e f } } + \lambda _ { \mathrm { K L } } ^ { c } \mathcal { L } _ { \mathrm { K L } } ^ { c } )$ . This corresponds to the InfoRM-style information bottleneck baseline.

• w/o reconstruction: set $\lambda _ { \mathrm { { r e c } } } = 0$ (remove $\mathcal { L } _ { \mathrm { r e c } } )$ while keeping factorization and adversarial training.

• w/o adversarial/GRL: set $\lambda _ { \mathrm { a d v } } = 0$ (remove $\mathcal { L } _ { \mathrm { a d v } }$ and the GRL) while keeping factorization and reconstruction.

• w/o KL on z<sup>c</sup>: set $\lambda _ { \mathrm { K L } } ^ { c } = 0 .$

• w/o KL on $z ^ { n c } \colon$ set $\lambda _ { \mathrm { K L } } ^ { n c } = 0 .$

• w/o KL on both latents: set $\lambda _ { \mathrm { K L } } ^ { c } = \lambda _ { \mathrm { K L } } ^ { n c } = 0 .$

All other training details follow the main experiments to ensure a controlled comparison.

Results and analysis. Tables 9 and 10 summarize the ablation results. Overall, we observe that the full CausalRM consistently performs best, and different components contribute in complementary ways.

Factorization and the structural restriction are important. Compared to the InfoRM-equivalent variant (w/o factoriza tion), the full CausalRM improves math pairwise accuracy by +4.0 points on ID (70.1 vs. 66.1) and +3.1 on OOD (85.6 vs. 82.5), and substantially reduces length sensitivity (std. across length bins drops from 0.14 to 0.03; Figure 6). On the hacked sycophancy evaluation for dialogue, factorization also yields consistent gains (ID +3.8, OOD +3.9). These results suggest that explicitly separating reward-relevant and reward-irrelevant channels provides a stronger inductive bias than a single bottlenecked latent: even when capacity is constrained, a single latent can still entangle spurious cues with reward-relevant features, whereas the factorized design makes it easier to route spurious variation away from the reward head.

Figure 6. Length sensitivity under ablations on mathematical reasoning. Length is normalized to [0, 1] and rewards are averaged within length quantile buckets on chosen responses from the ID test set.

Adversarial training (GRL) is crucial for invariance and OOD robustness. Removing the adversarial head/GRL causes a noticeable degradation on mathematical reasoning, especially under distribution shift (ID: -3.1, OOD: -4.1), and increases length sensitivity (std. rises from 0.03 to 0.13; Figure 6). On dialogue, the same ablation reduces accuracy on the hacked evaluation by -3.3 (ID) and -4.0 (OOD). This pattern supports that the adversarial head helps suppress reward-predictive leakage into $z ^ { n c }$ , which becomes particularly important when spurious artifacts correlate with preference labels or when test distributions shift.

Reconstruction improves robustness but is not the main driver of accuracy. The reconstruction term has a relatively small effect on ID accuracy (math: -1.6; dialogue hacked: -0.4), but provides consistent gains on OOD benchmarks (math: -2.5; dialogue hacked: -2.6 when removed). This indicates that reconstruction acts as a stabilizer that helps preserve information from the backbone embedding under the factorized bottleneck, complementing the structural and adversarial mechanisms.

Both KL terms matter and are complementary. Removing the KL $\mathrm { o n } z ^ { c } \left( \lambda _ { \mathrm { K L } } ^ { c } { = } 0 \right)$ hurts generalization on math (OOD: -4.5) and reduces robustness on hacked dialogue OOD (-4.1), consistent with the information bottleneck on $z ^ { c }$ discouraging the reward head from exploiting redundant or spurious signals. Removing the KL on $z ^ { n c } \left( \lambda _ { \mathrm { K L } } ^ { n c } { = } 0 \right)$ also degrades robustness (math OOD: -4.0; dialogue hacked ID: -6.7), suggesting that regularizing the non-causal channel is important for stable learning and robustness, as an unconstrained $z ^ { n c }$ can become an easy pathway for dataset-specific signals. Finally, removing both KL terms yields the largest drop (math OOD: -11.0; dialogue hacked ID: -24.6), showing that explicit factorization and adversarial training still benefit from information-theoretic regularization to achieve a robust decomposition in practice.

In summary, the ablations support our design choices: factorization with a reward head conditioned only on $z ^ { c }$ provides a strong inductive bias, GRL-based adversarial training is critical for enforcing invariance under spurious shifts, reconstruction offers additional robustness, and KL regularization on both latents is necessary to avoid brittle solutions.

Figure 7. Sensitivity of predicted reward to response length on mathematical reasoning. Length is normalized to [0, 1] and rewards are averaged within length quantile buckets. $\sigma _ { \mathrm { l e n } }$ denotes the standard deviation of bucket-wise mean rewards.

## F. Additional Results

## F.1. Comparisons with Artifact-Specific Baselines

To contextualize CausalRM against methods that explicitly target known spurious attributes, we compare it with ODIN (Chen et al., 2024) and MMD-based regularization (Wang et al., 2025). Unlike our general latent factorization approach, these methods require explicit specification of the spurious variable (e.g., response-length bins or sycophantic prefixes) during training. We therefore evaluate them under the targeted bias settings described in Section 4.3, where artifact attributes are controllably perturbed.

Length bias on mathematical reasoning. We compare against ODIN and MMD when response length is treated as a known spurious factor. Table 11 reports reward model accuracy on the mathematical reasoning benchmarks. CausalRM achieves the best performance on both ID and OOD evaluations, outperforming Standard RM as well as the two artifactspecific baselines.

Table 11. Reward model accuracy (%) on mathematical reasoning with length bias mitigation.

<table><tr><td>Method</td><td>ID Avg.</td><td>OOD Avg.</td></tr><tr><td>Standard RM</td><td>67.9</td><td>83.0</td></tr><tr><td>ODIN (length-specific)</td><td>68.8</td><td>83.2</td></tr><tr><td>MMD (length-specific)</td><td>69.3</td><td>83.6</td></tr><tr><td>CausalRM (Ours)</td><td>70.1</td><td>85.6</td></tr></table>

To further quantify sensitivity to response length, Figure 7 plots the normalized reward as a function of normalized answer length. CausalRM exhibits the flattest response curve, with the lowest bucket-wise reward variance $\sigma _ { \mathrm { l e n } } = 0 . 0 3 .$ , compared with 0.12 for ODIN and 0.09 for MMD. This indicates that CausalRM is substantially less sensitive to response length, even relative to methods explicitly designed for known length bias.

Sycophantic artifacts on open-ended dialogue. Table 12 presents the robustness evaluation on dialogue tasks with sycophantic-phrasing artifacts. CausalRM again achieves the highest average pairwise accuracy across both ID (70.6%) and OOD (61.2%) splits, outperforming ODIN (66.7% / 57.5%) and MMD (67.1% / 57.9%).

These results demonstrate that while artifact-specific regularization can be competitive when the target bias is known a priori, CausalRM consistently achieves stronger robustness and OOD generalization. Crucially, our method operates without requiring task-specific artifact annotations or explicit supervision of spurious factors. This highlights the practical advantage of CausalRM, particularly in real-world settings where the exploited spurious variables are typically unknown.

## F.2. Hyperparameter Sensitivity

To assess the training stability of CausalRM and its sensitivity to key loss coefficients, we conduct a systematic sensitivity analysis of the hyperparameters in Eq. (10). We vary each coefficient across a wide range while keeping the others fixed at their default values, and report the average pairwise accuracy on the seven mathematical reasoning benchmarks.

Table 12. Robustness to sycophantic-phrasing artifacts measured by pairwise accuracy (%) on hacked test sets. Higher is better.

<table><tr><td rowspan="2">Method</td><td colspan="3">ID</td><td colspan="5">OOD</td></tr><tr><td>Helpful</td><td>Harmless</td><td>Avg.</td><td>MT-Bench</td><td>PKU-SafeRLHF</td><td>SHP</td><td>TruthfulQA</td><td>Avg.</td></tr><tr><td>ODIN</td><td>61.5</td><td>71.9</td><td>66.7</td><td>62.6</td><td>59.5</td><td>47.2</td><td>60.8</td><td>57.5</td></tr><tr><td>MMD</td><td>63.9</td><td>70.3</td><td>67.1</td><td>65.2</td><td>55.2</td><td>49.6</td><td>61.6</td><td>57.9</td></tr><tr><td>CausalRM (Ours)</td><td>67.4</td><td>73.8</td><td>70.6</td><td>65.7</td><td>62.0</td><td>50.3</td><td>66.8</td><td>61.2</td></tr></table>

Table 13. Scalability of CausalRM on mathematical reasoning across different backbone sizes. We report pairwise accuracy (%, higher is better) and approximate reward model training time.

<table><tr><td>Model</td><td>GSM8K</td><td>MATH</td><td>Algebra222</td><td>GSM-Hard</td><td>ASDiv</td><td>MAWPS</td><td>SVAMP</td><td>RM Train Time</td></tr><tr><td>Qwen2.5-Math-1.5B</td><td>70.0</td><td>51.8</td><td>80.1</td><td>59.5</td><td>88.2</td><td>86.2</td><td>85.4</td><td>0.8h</td></tr><tr><td>Qwen2.5-Math-7B</td><td>81.7</td><td>58.4</td><td>89.9</td><td>66.2</td><td>89.9</td><td>92.2</td><td>89.6</td><td>6.5h</td></tr><tr><td>Qwen2.5-Math-72B</td><td>80.6</td><td>63.5</td><td>91.6</td><td>69.8</td><td>92.6</td><td>94.3</td><td>89.4</td><td>26.5h</td></tr></table>

Figure 8 summarizes the results. The model exhibits robust performance across a reasonable range of settings for all five coefficients. Specifically, the adversarial weight $\lambda _ { \mathrm { a d v } }$ and the gradient reversal coefficient $\lambda _ { \mathrm { g r l } }$ maintain stable behavior within [0.01, 0.1] and [0.25, 2.0], respectively, with peak accuracy around the chosen defaults. Similarly, the KL regularization weights $\lambda _ { \mathrm { K L } } ^ { c }$ and $\lambda _ { \mathrm { K L } } ^ { n c }$ , as well as the reconstruction weight $\lambda _ { \mathrm { { r e c } } }$ , sustain high performance within $[ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ]$ . Accuracy degrades only when these coefficients are pushed to extreme values $( \mathrm { e . g . } , \lambda _ { \mathrm { a d v } } = 0 . 2 \ : \mathrm { o r } \ : \lambda _ { \mathrm { K L } } ^ { c } = 0 . 0 1 )$ , where the invariance or bottleneck objectives begin to overly suppress the primary preference signal.

Overall, these results indicate that CausalRM exhibits stable behavior across a reasonably broad range of settings, which supports its practical robustness.

## F.3. Scalability Across Model Sizes

CausalRM is designed as a lightweight extension of a standard reward model: it only introduces a factorized latent bottleneck, two small auxiliary heads, and a gradient reversal layer on top of the backbone representation, without modifying the underlying LLM architecture. As a result, the additional computational overhead is modest. In our main mathematical reasoning experiments, for example, training a Standard RM takes 6.2 hours, while CausalRM requires 6.5 hours under the same hardware setup.

To further examine scalability, we evaluate CausalRM across three backbone sizes on mathematical reasoning: Qwen2.5- Math-1.5B, Qwen2.5-Math-7B, and Qwen2.5-Math-72B. Table 13 reports pairwise accuracy on both ID and OOD bench marks together with reward model training time.

Overall, CausalRM scales well with model size. Moving from 1.5B to 7B yields substantial gains across nearly all benchmarks, indicating that the proposed factorization framework remains effective when paired with stronger backbones. Scaling further to 72B continues to improve performance on more challenging benchmarks such as MATH, Algebra222, GSM-Hard, ASDiv, and MAWPS, while maintaining competitive results on the remaining datasets. These results suggest that CausalRM does not depend on a narrow model scale and can be applied across a broad range of backbone capacities.

At the same time, the computational cost grows in a predictable manner with the backbone size and remains dominated by the underlying LLM rather than the additional CausalRM modules. This supports the practical scalability of our method: the robustness benefits of factorized reward modeling can be obtained without introducing substantial architectural complexity beyond standard reward model training.

## F.4. Multi-Judge Validation for Dialogue Evaluation

We provide additional pairwise evaluations using two alternative judge models, DeepSeek-V3.2 and Kimi-k2.5, under the same protocol and prompt template as described in Appendix C.

Tables 14 and 15 summarize the results. Across both judges, CausalRM remains consistently preferred over all baselines on both ID and OOD benchmarks. Under DeepSeek-V3.2, CausalRM achieves average OOD win rates of 58.3%, 51.7%,


λ<sub>grl</sub>

λ<sup>c</sup><sub>KL</sub>

λ<sup>nc</sup><sub>KL</sub>

λ<sub>rec</sub>
Figure 8. Sensitivity analysis of key hyperparameters on mathematical reasoning. The y-axis shows the average pairwise accuracy (%) across 7 datasets. CausalRM maintains stable performance across a broad range of loss coefficients.

Table 14. Downstream RLHF performance on open-ended dialogue evaluated by DeepSeek-V3.2 pairwise comparison win rate (%). Each entry reports Win/Tie/Lose of the CausalRM-trained policy against an opponent policy trained with a baseline reward model.

<table><tr><td rowspan="3">Model</td><td rowspan="3">Opponent</td><td colspan="9">ID</td><td colspan="14">OOD</td><td></td></tr><tr><td colspan="3">Anthropic-Helpful</td><td colspan="3">Anthropic-Harmless</td><td colspan="3">Avg.</td><td colspan="3">MT-Bench</td><td colspan="3">PKU-SafeRLHF</td><td colspan="3">SHP</td><td colspan="3">TruthfulQA</td><td colspan="2">Avg.</td><td></td></tr><tr><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td></tr><tr><td rowspan="4">CausalRM (Ours)</td><td>SFT</td><td>69.9</td><td>20.6</td><td>9.5</td><td>58.1</td><td>33.1</td><td>8.8</td><td>64.0</td><td>26.9</td><td>9.1</td><td>39.4</td><td>47.1</td><td>13.5</td><td>67.0</td><td>22.3</td><td>10.7</td><td>77.5</td><td>10.3</td><td>12.2</td><td>49.3</td><td>31.6</td><td>19.1</td><td>58.3</td><td>27.8</td><td>13.9</td></tr><tr><td>Standard RM</td><td>51.5</td><td>28.7</td><td>19.8</td><td>59.3</td><td>26.7</td><td>14.0</td><td>55.4</td><td>27.7</td><td>16.9</td><td>43.5</td><td>47.1</td><td>9.4</td><td>56.5</td><td>25.3</td><td>18.2</td><td>58.7</td><td>21.6</td><td>19.7</td><td>47.9</td><td>30.2</td><td>21.9</td><td>51.7</td><td>31.1</td><td>17.2</td></tr><tr><td>GoalRM</td><td>43.4</td><td>44.1</td><td>12.5</td><td>34.6</td><td>43.4</td><td>22.0</td><td>39.0</td><td>43.8</td><td>17.2</td><td>35.8</td><td>42.7</td><td>21.5</td><td>35.3</td><td>41.2</td><td>23.5</td><td>38.7</td><td>36.0</td><td>25.3</td><td>33.1</td><td>41.9</td><td>25.0</td><td>35.7</td><td>40.5</td><td>23.8</td></tr><tr><td>InfoRM</td><td>64.0</td><td>22.8</td><td>13.2</td><td>51.1</td><td>24.5</td><td>24.4</td><td>57.6</td><td>23.7</td><td>18.7</td><td>39.7</td><td>44.9</td><td>15.4</td><td>37.5</td><td>39.0</td><td>23.5</td><td>50.7</td><td>26.5</td><td>22.8</td><td>51.5</td><td>30.9</td><td>17.6</td><td>44.9</td><td>35.3</td><td>19.8</td></tr></table>

35.7%, and 44.9% against SFT, Standard RM, GoalRM, and InfoRM, respectively. Under Kimi-k2.5, the corresponding OOD win rates are 56.7%, 53.5%, 34.8%, and 36.3%. The same overall trend also holds on the ID benchmarks. This multi-judge validation further strengthens the statistical reliability and practical robustness of our downstream RLHF conclusions in open-ended dialogue settings.

## F.5. Paired Bootstrap Analysis for Dialogue Evaluation

To quantify the statistical reliability of the dialogue results in Table 3, we additionally perform paired bootstrap analysis for the pairwise win-rate evaluation.

Setup. For each opponent policy, we collect paired LLM-judge outcomes on the same test instances across the six dialogue benchmarks. We then repeatedly resample instances with replacement and compute the preference margin

$$
\Delta = \frac {\# \mathrm{Win} - \# \mathrm{Lose}}{N},
$$

Table 15. Downstream RLHF performance on open-ended dialogue evaluated by Kimi-k2.5 pairwise comparison win rate (%). Each entry reports Win/Tie/Lose of the CausalRM-trained policy against an opponent policy trained with a baseline reward model.

<table><tr><td rowspan="3">Model</td><td rowspan="3">Opponent</td><td colspan="9">ID</td><td colspan="14">OOD</td></tr><tr><td colspan="3">Anthropic-Helpful</td><td colspan="3">Anthropic-Harmless</td><td colspan="3">Avg.</td><td colspan="3">MT-Bench</td><td colspan="3">PKU-SafeRLHF</td><td colspan="3">SHP</td><td colspan="3">TruthfulQA</td><td colspan="2">Avg.</td></tr><tr><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td><td>Lose</td><td>Win</td><td>Tie</td></tr><tr><td rowspan="4">CausalRM (Ours)</td><td>SFT</td><td>64.7</td><td>25.7</td><td>9.6</td><td>62.5</td><td>25.7</td><td>11.8</td><td>63.6</td><td>25.7</td><td>10.7</td><td>38.7</td><td>44.4</td><td>16.9</td><td>68.5</td><td>21.4</td><td>10.1</td><td>74.1</td><td>14.4</td><td>11.5</td><td>45.6</td><td>38.2</td><td>16.2</td><td>56.7</td><td>29.6</td></tr><tr><td>Standard RM</td><td>52.9</td><td>19.9</td><td>27.2</td><td>57.5</td><td>20.9</td><td>21.6</td><td>55.2</td><td>20.4</td><td>24.4</td><td>44.7</td><td>40.0</td><td>15.3</td><td>58.4</td><td>28.7</td><td>12.9</td><td>58.7</td><td>20.6</td><td>20.7</td><td>52.1</td><td>27.2</td><td>20.7</td><td>53.5</td><td>29.1</td></tr><tr><td>GoalRM</td><td>50.0</td><td>41.2</td><td>8.8</td><td>41.5</td><td>34.8</td><td>23.7</td><td>45.8</td><td>38.0</td><td>16.2</td><td>30.2</td><td>49.3</td><td>20.5</td><td>33.1</td><td>44.9</td><td>22.0</td><td>40.4</td><td>27.9</td><td>31.7</td><td>35.3</td><td>36.8</td><td>27.9</td><td>34.8</td><td>39.7</td></tr><tr><td>InfoRM</td><td>50.7</td><td>36.1</td><td>13.2</td><td>39.0</td><td>30.9</td><td>30.1</td><td>44.9</td><td>33.5</td><td>21.6</td><td>26.5</td><td>47.1</td><td>26.4</td><td>37.2</td><td>35.8</td><td>27.0</td><td>37.5</td><td>26.5</td><td>36.0</td><td>43.8</td><td>30.2</td><td>26.0</td><td>36.3</td><td>34.9</td></tr></table>

Table 16. Paired bootstrap analysis for the dialogue results in Table 3. We report the average win rate of CausalRM against each opponent across the six dialogue benchmarks, together with the 95% confidence interval (CI) of the preference margin $\Delta = ( \dot { \# } \mathrm { W i n } - \# \mathrm { L } \dot { \mathrm { o s e } } ) / N$

<table><tr><td>Opponent</td><td>Avg. win rate (%)</td><td>95% CI of Δ (%)</td></tr><tr><td>SFT</td><td>61.6</td><td>[50.7, 59.7]</td></tr><tr><td>Standard RM</td><td>42.9</td><td>[17.2, 27.7]</td></tr><tr><td>GoalRM</td><td>33.2</td><td>[4.2, 14.6]</td></tr><tr><td>InfoRM</td><td>34.1</td><td>[6.0, 15.9]</td></tr></table>

from the perspective of the CausalRM-trained policy, where ties contribute 0. We run 10,000 bootstrap resamples and report the 95% percentile confidence interval (CI) of ∆.

Results. Table 16 summarizes the average win rate of CausalRM against each opponent together with the corresponding 95% bootstrap CI of the preference margin. Across all comparisons, the confidence intervals remain strictly above 0, indicating that the CausalRM-trained policy is consistently preferred over the corresponding baseline under the pairwise evaluation protocol.

## G. Limitations and Discussion

CausalRM is built upon a foundational assumption that reward-relevant factors can be separated from spurious ones. This factorization assumption is standard in causal RL and has been successfully operationalized in diverse real-world domains (Huang et al., 2022; Liu et al., 2023; 2024; Ovinnikov et al., 2024; Wang et al., 2025; Song et al., 2025; Yang et al., 2025). While recovering true causal mechanisms from purely observational preference data remains a known open challenge in general (Schölkopf et al., 2021; Kong et al., 2023; Ng et al., 2025), we show in Section H that the reward-relevant latent factor is theoretically identifiable up to an invertible transformation under suitable variability, invariance, and sufficiency conditions. This provides a theoretical grounding for our latent factorization approach.

Beyond the theoretical result, we provide targeted empirical evidence to validate the causal behavior of the learned representations. As shown in Appendix I, controlled interventions on the true reward-determining factor (the final boxed answer in mathematical tasks) yield significantly larger and more consistent reward shifts for CausalRM compared to all baselines. This intervention-based sensitivity complements the invariance analyses in Section 4.3, where CausalRM demonstrates markedly reduced sensitivity to known spurious attributes such as response length and sycophantic phrasing. These results reveal the intended behavioral pattern: CausalRM is more sensitive to reward-relevant interventions while remaining less sensitive to spurious correlates, providing empirical support for the intended causal decomposition.

From a practical perspective, CausalRM integrates seamlessly into standard RLHF pipelines without requiring explicit artifact annotations or task-specific supervision. Our experiments across mathematical reasoning and open-ended dialogue consistently demonstrate improved reward modeling accuracy, stronger downstream RLHF performance, and enhanced OOD generalization. We view this work as a practical step toward causally robust reward modeling, demonstrating that invariance-motivated factorization can yield empirically reliable reward models. An important direction for future work is to bridge empirically effective invariance objectives with even stronger identifiability guarantees, potentially through leveraging explicit surrogates for spurious variables, exploiting multi-environment controlled perturbations, or integrating structural causal models with latent variable generation.

## H. Identifiability Analysis

In this section, we present an identifiability result for the reward-relevant factor $z ^ { c }$ under our factorization framework. Specifically, Theorem H.2 shows that $z ^ { c }$ is identifiable up to an invertible transformation when the learned representation satisfies suitable invariance, variability, and sufficiency conditions. Detailed proofs are presented in Appendix J. To further validate the causal behavior of the learned factors in our experiments, we additionally provide intervention-based empirical evidence in Appendix I. Below we first introduce the definition of identifiability used in this paper and then present the theoretical result.

Definition H.1 (Identifiability up to invertible transformation (Liu et al., 2023)). Let $\tilde { z } ^ { c }$ be the learned representation of $z ^ { c } .$ We say that $z ^ { c }$ is identifiable from $\tilde { z } ^ { c }$ up to an invertible transformation if there exists an invertible measurable map ψ such that $z ^ { c } = \psi ( \tilde { z } ^ { c } )$ ) almost everywhere.

Theorem H.2 (Identifiability of $z ^ { c } )$ . Assume the prompt–response representation o is generated by an invertible mixing $o = g ( z ^ { c } , z ^ { n c } )$ ), where $z ^ { c } \in \mathcal { Z } ^ { c }$ denotes reward-relevant factors and $z ^ { n c } \in { \mathcal { Z } } ^ { n c }$ denotes spurious factors. Suppose environments $e \in { \mathcal { E } }$ affect the data distribution only through the spurious mechanism, while the reward is generated as $r = m ( z ^ { c } , \varepsilon _ { r } )$ , satisfying $r ~ \perp ~ z ^ { n c } ~ | ~ z ^ { c }$ and $p _ { e } ( r \mid z ^ { c } ) = p ( r \mid z ^ { c } )$ for all $e \in { \mathcal { E } }$ . Let the learned representation be $\tilde { z } ^ { c } = \phi ( o )$ . Ifthefollowing conditions hold:

• (A1) Sufficient Environment Variability (Kong et al., 2023). For any measurable set $A \subseteq \mathcal Z ^ { c } \times \mathcal Z ^ { n c }$ with positive probability under at least one environment, ifA cannot be written as $B \times \mathcal { Z } ^ { n c }$ for any measurable $B \subseteq { \mathcal { Z } } ^ { c }$ , then there exist $e _ { 1 } , e _ { 2 } \in \mathcal { E }$ such that $P _ { e _ { 1 } } ( A ) \neq P _ { e _ { 2 } } ( A )$

• (A2) Representation Invariance (Schölkopf et al., 2021). The learned representation is environment-invariant: $\tilde { z } ^ { c } \perp e .$

• (A3) Minimal Sufficiency (Huang et al., 2022). $z ^ { c }$ is a minimal sufficient statistic for predicting r, and $\tilde { z } ^ { c }$ is also sufficient for predicting r.

Then $z ^ { c }$ is identifiablefrom $\tilde { z } ^ { c }$ up to an invertible transformation.

Table 17. Intervention-based evidence on mathematical reasoning. Results are averaged across 7 datasets: GSM8K, MATH, Algebra222, GSM-Hard, ASDiv, MAWPS, and SVAMP. For each dataset, we randomly sample 100 preference pairs for the correct → wrong intervention and 100 preference pairs for the wrong → correct intervention. ∆r denotes the average reward change after intervention. For correct → wrong, a more negative ∆r and a higher proportion of reward decreases are better; for wrong → correct, a more positive ∆r and a higher proportion of reward increases are better.

<table><tr><td rowspan="2">Method</td><td colspan="2">Correct → Wrong</td><td colspan="2">Wrong → Correct</td></tr><tr><td>Δr</td><td>Reward ↓ Ratio (%)</td><td>Δr</td><td>Reward ↑ Ratio (%)</td></tr><tr><td>Standard RM</td><td>-0.48</td><td>65.0</td><td>+0.51</td><td>63.7</td></tr><tr><td>GoalRM</td><td>-0.62</td><td>70.1</td><td>+0.63</td><td>70.9</td></tr><tr><td>InfoRM</td><td>-0.25</td><td>29.1</td><td>+0.26</td><td>27.6</td></tr><tr><td>CausalRM (Ours)</td><td>-0.76</td><td>86.1</td><td>+0.77</td><td>86.6</td></tr></table>

## I. Intervention-Based Empirical Evidence

We provide additional intervention-based evidence to complement the identifiability analysis in Section H. Our goal is to examine whether the learned reward-relevant representation is sensitive to interventions on a true reward-determining factor, rather than merely exploiting superficial correlates.

Motivation. For mathematical reasoning tasks, the final boxed answer is the most direct reward-determining variable (see Figure 9): changing a correct final answer to an incorrect one should decrease the reward, while changing an incorrect final answer to the correct one should increase the reward. We therefore use controlled answer-edit interventions to test whether the reward models respond in the expected direction.

Intervention protocol. We consider two intervention types:

• (i) Correct → Wrong: replace the final boxed answer in a correct response with an incorrect one, while keeping the rest of the response unchanged;

• (ii) Wrong → Correct: replace the final boxed answer in an incorrect response with the correct one, again keeping the remainder of the response unchanged.

If a reward model captures this reward-determining factor, its predicted reward should decrease under the first intervention and increase under the second.

Evaluation setup. We perform this analysis on the mathematical reasoning benchmarks used in the main paper, including GSM8K, MATH, Algebra222, GSM-Hard, ASDiv, MAWPS, and SVAMP. For each dataset, we randomly sample 100 preference pairs for the correct → wrong intervention and 100 preference pairs for the wrong → correct intervention. We report:

• the average reward change ∆r after intervention; and

• the proportion of examples whose reward changes in the expected direction.

For correct → wrong, a more negative ∆r and a higher reward-decrease ratio are better. For wrong → correct, a more positive ∆r and a higher reward-increase ratio are better.

Results. Table 17 shows that CausalRM consistently exhibits the expected directional response under both interventions. In the correct → wrong setting, CausalRM yields the largest reward drop $( \Delta r = - 0 . 7 6 )$ , compared with Standard RM (−0.48), GoalRM (−0.62), and InfoRM (−0.25). In the wrong → correct setting, CausalRM similarly achieves the largest reward increase $( \Delta r = + 0 . 7 7 )$ , again exceeding all baselines.

A similar pattern is observed in the directional consistency ratios. CausalRM decreases reward in 86.1% of the correct → wrong cases and increases reward in 86.6% of the wrong → correct cases, substantially higher than Standard RM (65.0% / 63.7%), GoalRM (70.1% / 70.9%), and InfoRM (29.1% / 27.6%). These results suggest that CausalRM is more responsive to interventions on the true reward-determining factor, rather than relying primarily on superficial correlates.

Complementary perspective. This intervention analysis should be interpreted together with the invariance analyses in Section 4.3. There, we showed that CausalRM is markedly less sensitive to non-causal attributes such as response length and sycophantic phrasing. Taken together, these findings support the intended pattern behind our approach: CausalRM is more sensitive to reward-relevant changes while being more invariant to spurious ones.

## J. Proof of Theorem H.2

In this section, we present the proof of Theorem H.2. We first restate the setting and assumptions, then show that the learned representation $\tilde { z } ^ { c }$ cannot depend on the spurious factor $z ^ { n c }$ . Finally, by combining this result with the minimal sufficiency assumption, we conclude that $z ^ { c }$ is identifiable from $\tilde { z } ^ { c }$ up to an invertible transformation.

Notation. Let the observed prompt–response representation be denoted by

$$
o = g (z ^ {c}, z ^ {n c}),\tag{27}
$$

where $z ^ { c } \in \mathcal { Z } ^ { c }$ denotes the reward-relevant factor and $z ^ { n c } \in { \mathcal { Z } } ^ { n c }$ denotes spurious factors. The learned representation is

$$
\tilde {z} ^ {c} = \phi (o).\tag{28}
$$

Since $o = g ( z ^ { c } , z ^ { n c } )$ , we define the latent-space form of the encoder by

$$
\bar {\phi} := \phi \circ g, \qquad \tilde {z} ^ {c} = \bar {\phi} (z ^ {c}, z ^ {n c}).\tag{29}
$$

We assume that environments $e \in { \mathcal { E } }$ affect the data distribution only through the spurious mechanism, while the reward is generated as

$$
r = m (z ^ {c}, \varepsilon_ {r}),\tag{30}
$$

so that

$$
r \perp z ^ {n c} \mid z ^ {c}, \qquad p _ {e} (r \mid z ^ {c}) = p (r \mid z ^ {c}), \quad \forall e \in \mathcal {E}.\tag{31}
$$

For convenience, we restate the assumptions used in the proof:

• (A1) Sufficient environment variability. For any measurable set $A \subseteq \mathcal Z ^ { c } \times \mathcal Z ^ { n c }$ with positive probability under at least one environment, if A cannot be written as $B \times \mathcal { Z } ^ { n c }$ for any measurable $B \subseteq { \mathcal { Z } } ^ { c }$ , then there exist $e _ { 1 } , e _ { 2 } \in \mathcal { E }$ such that

$$
P _ {e _ {1}} (A) \neq P _ {e _ {2}} (A).\tag{32}
$$

• (A2) Representation invariance. The learned representation is environment-invariant:

$$
\tilde {z} ^ {c} \perp e.\tag{33}
$$

• (A3) Minimal sufficiency. The latent factor $z ^ { c }$ is a minimal sufficient statistic for predicting $r ,$ and $\tilde { z } ^ { c }$ is also sufficient for predicting r.

We first record a direct consequence of Assumption (A2).

Lemma J.1. $H \tilde { z } ^ { c } \perp e ,$ thenfor every measurable set $A _ { \tilde { z } }$ in the range of $\cdot \tilde { z } ^ { c }$ ,

$$
P _ {e _ {1}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big) = P _ {e _ {2}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big), \quad \forall e _ {1}, e _ {2} \in \mathcal {E}.\tag{34}
$$

Proof. Since $\tilde { z } ^ { c } = \bar { \phi } ( z ^ { c } , z ^ { n c } )$ , for any measurable set $A _ { \tilde { z } }$ we have

$$
\{\tilde {z} ^ {c} \in A _ {\tilde {z}} \} = \{(z ^ {c}, z ^ {n c}) \in \bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \}.\tag{35}
$$

Assumption (A2) implies that the distribution of $\tilde { z } ^ { c }$ is identical across environments. Therefore,

$$
P _ {e _ {1}} (\tilde {z} ^ {c} \in A _ {\tilde {z}}) = P _ {e _ {2}} (\tilde {z} ^ {c} \in A _ {\tilde {z}}), \quad \forall e _ {1}, e _ {2} \in \mathcal {E},\tag{36}
$$

which is equivalent to

$$
P _ {e _ {1}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big) = P _ {e _ {2}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big).\tag{37}
$$

This proves the claim.

□

We next show that the learned representation cannot depend on the spurious factor $z ^ { n c }$

Lemma J.2. Under Assumptions (A1) and (A2), $\bar { \phi } ( z ^ { c } , z ^ { n c } )$ cannot depend on $z ^ { n c }$ . Equivalently, there exists a measurable map ψ such tha

$$
\tilde {z} ^ {c} = \psi (z ^ {c}).\tag{38}
$$

Proof. We prove the result by contradiction. Suppose $\bar { \phi }$ depends on $z ^ { n c }$ . Then there exists a measurable set $A _ { \tilde { z } }$ in the range of $\tilde { z } ^ { c }$ such that its preimage

$$
\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) = \{(z ^ {c}, z ^ {n c}): \bar {\phi} (z ^ {c}, z ^ {n c}) \in A _ {\tilde {z}} \}\tag{39}
$$

cannot be written as $B \times \mathcal { Z } ^ { n c }$ for any measurable $B \subseteq { \mathcal { Z } } ^ { c }$ . Intuitively, this means that membership in the preimage depends nontrivially on the spurious factor $z ^ { n c }$

By Assumption (A1), there exist two environments $e _ { 1 } , e _ { 2 } \in \mathcal { E }$ such that

$$
P _ {e _ {1}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big) \neq P _ {e _ {2}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big).\tag{40}
$$

However, by Lemma J.1, Assumption (A2) implies that for every measurable set $A _ { \tilde { z } }$ ,

$$
P _ {e _ {1}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big) = P _ {e _ {2}} \big (\bar {\phi} ^ {- 1} (A _ {\tilde {z}}) \big),\tag{41}
$$

which is a contradiction. Therefore, $\bar { \phi }$ cannot depend on $z ^ { n c }$

Hence there exists a measurable map $\psi : \mathcal { Z } ^ { c } \to \tilde { \mathcal { Z } } ^ { c }$ such that

$$
\tilde {z} ^ {c} = \psi (z ^ {c}).\tag{42}
$$

We are now ready to prove the main theorem.

Proof of Theorem H.2. By Lemma J.2, there exists a measurable map ψ such that

$$
\tilde {z} ^ {c} = \psi (z ^ {c}).\tag{43}
$$

It remains to show that $z ^ { c }$ can be recovered from $\tilde { z } ^ { c }$ up to an invertible transformation.

Under Assumption $( \mathbf { A } 3 ) , z ^ { c }$ is a minimal sufficient statistic for predicting $r ,$ and $\tilde { z } ^ { c }$ is also sufficient for predicting $^ { r } \cdot$ Suppose, for contradiction, that $\psi$ is not injective on a set of positive measure. Then there exist distinct values $z _ { 1 } ^ { c } \neq z _ { 2 } ^ { c }$ such that

$$
\psi (z _ {1} ^ {c}) = \psi (z _ {2} ^ {c}).\tag{44}
$$

Hence, $\tilde { z } ^ { c }$ is a strict coarsening of $z ^ { c }$ , in the sense that it maps distinct values of $z ^ { c }$ to the same representation.

Because $\tilde { z } ^ { c }$ is sufficient for predicting r, this strict coarsening still preserves all reward-relevant information. This contradicts the minimal sufficiency of $z ^ { c }$ , since a minimal sufficient statistic cannot be further compressed through a non-injective measurable map while remaining sufficient. Therefore, ψ must be injective almost everywhere.

Consequently, ψ is invertible almost everywhere on its image, and there exists an invertible measurable map from $\tilde { z } ^ { c }$ back to $z ^ { c } .$ Equivalently, $z ^ { c }$ is identifiable from $\tilde { z } ^ { c }$ up to an invertible transformation. □

## K. Reward Hacking Example

To better illustrate the effectiveness of CausalRM in mitigating reward hacking, we provide concrete examples from both mathematical reasoning and dialogue tasks, where CausalRM outperforms baselines by avoiding spurious patterns such as format hacking (see Figure 9), off-topic continuation (see Figures 9, 10 and 11), redundant repetition (see Figure 11), and misleading or incomplete reasoning (see Figure 12).

Figure 9. Reward hacking behaviors on an ID MATH prompt. Standard RM outputs an incorrect boxed answer (-22), InfoRM exhibits format hacking by outputting code without a final boxed answer, and GoalRM answers correctly but continues with an unrelated prompt (off-topic continuation). In contrast, CausalRM follows the instruction and produces the correct boxed answer (-10).

Factored Causal Representation Learning for Robust Reward Modeling in RLHF
Figure 10. Reward hacking behaviors on a GSM-Hard prompt. Standard RM computes the correct numerical result but outputs an incorrect boxed answer due to arithmetic error. InfoRM correctly calculates the balance but hacks the format by overriding the true answer with 0, falsely claiming no overpayment. GoalRM produces the right magnitude but misses the negative sign and appends an unrelated continuity proof (off-topic continuation). In contrast, CausalRM faithfully follows the instruction, correctly computes the negative remaining balance, and outputs the exact answer (-70145984) as required.

Figure 11. Reward hacking behaviors on an Anthropic-Helpful prompt. Standard RM and GoalRM exhibit verbosity hacking by generating excessively long, repetitive ingredient lists (e.g., duplicating the same vegetables or repeatedly listing “bay leaf”), which inflates superficial “helpfulness” without adding useful content. InfoRM produces a reasonable recipe but drifts off-topic by continuing into an unrelated dialogue about fruits and vegetables. In contrast, CausalRM provides a concise, coherent recipe that stays on-topic and avoids redundant repetition.

Figure 12. Reward hacking behaviors on a SHP prompt. Standard RM exhibits misleading explanations by providing a factually incorrect rationale (claiming that “Tupperware is not a dish, so it does not get wet”). GoalRM and InfoRM avoid the explicit error but give shallow, incomplete explanations that do not account for how plastic and container geometry affect drying. In contrast, CausalRM produces a coherent, physically plausible explanation for why plastic containers often remain wet after a dishwasher cycle.
