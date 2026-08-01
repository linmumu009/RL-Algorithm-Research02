# Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers

Xin-Qiang Cai<sup>1</sup> Wei Wang<sup>2,1</sup> Feng Liu<sup>2</sup> Tongliang liu<sup>3</sup> Gang Niu<sup>1</sup> Masashi Sugiyama<sup>1,2</sup>

<sup>1</sup>RIKEN AIP, Tokyo, Japan <sup>2</sup>The University of Tokyo, Tokyo, Japan <sup>3</sup>The University of Melbourne, Melbourne, Australia <sup>4</sup>The University of Sydney, Sydney, Australia

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) replaces costly human labeling with automated verifiers. To reduce verifier hacking, many RLVR systems binarize rewards to {0, 1}, but imperfect verifiers inevitably introduce false negatives (FN, rejecting correct answers) and false positives (FP, accepting incorrect ones). We formalize verifier unreliability as a stochastic reward channel with asymmetric noise rates $\rho _ { 0 }$ and $\rho _ { 1 }$ —the FP rate and the FN rate, respectively. From this abstraction we derive two lightweight corrections: (i) a backward correction that yields an unbiased surrogate reward and thus an unbiased policy-gradient estimator in expectation, and (ii) a forward correction that reweights score-function terms so the expected update aligns with the clean gradient direction and requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization pipeline, both corrections improve RLVR reasoning under synthetic and real verifier noise, with the forward variant being more stable under heavier noise. We further use a lightweight appeals mechanism to estimate FN rates online and show that the correction can complement stronger verifier-side baselines.

## 1. Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) ofers a scalable paradigm for improving the reasoning abilities of Large Language Models (LLMs) by replacing expensive human annotation with automated feedback [36]. In this problem, a policy is trained using rewards from a verifier that automatically checks the correctness of a model’s output [27]. The eficacy of this

Figure 1: Verifier-noise flow in RLVR. An AI agent produces candidate solutions that are scored by automated verifiers. While verifiers would yield false negatives $\left( \begin{array} { l l l } { \frac { 1 2 } { 3 6 } } & { \mathrm { v s . } } & { \frac { 1 } { 3 } } \end{array} \right.$ , reaching 38% rates [38]) and false positives (mislead by “Let’s solve it step by step...”, reaching 35% − 68% rates [41]), confusing the agent; applying our backward/forward corrections restores correct signals.  
approach, however, heavily depends on the verifier’s reliability. Emerging evidence reveals that verifiers are systematically fallible in two critical ways: they can accept incorrect solutions (false positives; FPs) or reject correct ones (false negatives; FNs) [38, 41].

FPs have been widely documented as a vulnerability of LLM-based verifiers. Recent studies showed that LLM judges can be swayed by superficial cues—e.g., popular specialized verifiers, such as GPT-4o, give 35 % – 66.8% FP rate when the answer starts with Let’s solve this problem step by step [41, 29, 28, 1]. Conversely, FNs are common with rule-based verifiers. These checkers, while highly precise, are often brittle; they may reject valid solutions that are formatted diferently, expressed in an algebraically equivalent form, or embedded in explanatory text [10]. A recent analysis of a math-RL dataset found that over 38% of responses flagged as incorrect by a rule-based system were in fact correct, a gap that a lightweight LLM verifier could partially close [38]. Both FPs and FNs materially degrade RLVR training: FNs deprive the agent of informative gradients and slow convergence, while FPs reward hackable patterns and inflate returns during policy optimization [38, 9].

Motivated by these failures, we address these challenges by explicitly treating verifier errors as noise in the reward signal. We model the verifier as a stochastic reward channel that corrupts the true, latent reward with verifier-conditional probabilities—the FP rate and the FN rate. Building on this formulation, we derive two estimators to counteract this noise. Our first method, noisy Policy Gradient with Backward Correction (PGBC), inverts the noise process to compute an unbiased estimator of the true reward, which can be used as a drop-in replacement in any advantage-estimation in RL. Our second method, noisy Policy Gradient with Forward Correction (PGFC), directly reweights the terms of the policy gradient to ensure its expected direction aligns with the clean gradient. This approach requires only an estimate of the FN rate to achieve directional correctness, making it suitable for the often harder-to-estimate FP rate [38]. The pipeline is illustrated in Figure 1.

We integrate these corrections into a group relative policy optimization (GRPO) [27]-based RLVR pipeline and demonstrate their efectiveness on competitive math reasoning benchmarks [27]. Both methods consistently outperform uncorrected training and achieve performance nearly equivalent to that of noise-free scenarios, with the forward correction ofering the fastest and most stable convergence. We also propose a practical online estimation scheme for the FN rate: a querying mechanism where a low-cost LLM verifier like TinyV [38] re-evaluates outputs rejected by a primary rule-based checker. This hybrid approach provides a reliable estimate of the FN rate with minimal computational overhead, recovering near-oracle performance. Furthermore, we stress-test our approach under complex, non-i.i.d. noise distributions that violate our FP and FN modeling, but both PGBC and PGFC corrections remain robust and efective even in these more challenging conditions. Our contributions are: (i) a formal verifier-channel model for RLVR that captures the prevalent FP and FN errors; (ii) backward and forward corrections derived from RL principles to calibrate policy learning; and (iii) a practical implementation for online noise estimation that makes these corrections viable in real-world systems.

## 2. Related Works

Reasoning with LLMs.. A large body of work improves LLM reasoning through prompting and search at inference time. Chain-of-Thought (CoT) prompting elicits step-by-step rationales and boosts arithmetic and commonsense reasoning [35], while self-consistency aggregates diverse reasoning paths to further improve robustness [34]. Decomposition-based prompting, such as Leastto-Most, solves complex problems via ordered sub-problems [42]; search-based schemes like Treeof-Thoughts explicitly explore and evaluate alternative reasoning branches [39]. Orthogonally, training-time verifier signals can be used to re-rank candidate solutions, as in Cobbe et al. [3]. Our work is complementary: rather than proposing a new inference-time strategy or re-ranking scheme, we address how to perform policy optimization when the reward itself—supplied by a verifier—is noisy.

RLVR and verifier reliability.. Recent math-RL pipelines combine on-policy RL (e.g., GRPO) with programmatic or LLM-based verifiers to yield verifiable rewards [21]. However, LLM-as-ajudge is prone to systematic biases, including position bias and prompt-based attacks, producing FPs (accepting incorrect solutions) [5, 32, 29, 41, 28]. Conversely, exact-match or brittle parsers in rule-based checkers can miss algebraic equivalences and text-formatted answers, leading to widespread FNs [17, 9]. A recent study showed that over 38% of model responses labeled incorrect by a rule-based pipeline were in fact correct, and introduced a lightweight verifier (TinyV) that recovers many such cases with minimal overhead [38]. Prior eforts therefore improved the verifier (e.g., better judges or appeals) or evaluation suites; in contrast, we treat verifier errors as verifierconditional noise and directly correct the policy gradient so that learning remains aligned with the clean objective even when the verifier is imperfect.

Learning with noisy labels.. While our method is derived from RL, it shares intuition with the literature on learning with label noise. Wang et al. [33] modeled reward corruption via a classconditional confusion matrix and derived an unbiased surrogate reward for Q-learning algorithms to solve control tasks. In supervised learning, a central thread is to make empirical-risk minimization robust either by correcting the loss given a noise-transition model or by avoiding or down-weighting suspected noisy examples [30, 16]. The former includes the unbiased-risk estimators of Natarajan et al. [22] and the now-standard loss corrections of Patrini et al. [24]. A second family avoids the noise explicitly by relying on the “small-loss first” memorization dynamics of deep nets: mentor methods (MentorNet) learn a weighting network that feeds cleaner samples to the student, and Co-teaching [6] trains two peers that exchange their selected small-loss examples; semi-supervised hybrids such as DivideMix [15] split data into clean/noisy partitions via mixture modeling and co-train with consistency regularization [12]. Unlike them, we target the policy gradient estimator in RLVR: we instantiate both backward and forward corrections inside the policy gradient estimator to stabilize and align RL under verifier noise, instead of a direct supervised-loss substitution.

## 3. Problem Setup and Algorithms

In this section, we introduce the problem formulations and our proposed algorithms.

## 3.1. Problem Setup

We consider the standard RLVR setting where a stochastic policy $\pi _ { \boldsymbol { \theta } } ( y | \boldsymbol { x } )$ generates a response y for a given prompt x, where θ denotes learnable parameters. The goal is to maximize an objective based on the true, or clean, reward $R ^ { * } ( x , y ) \in \{ 0 , 1 \}$ , which indicates whether the response y is genuinely correct. This objective is typically regularized by a KL-divergence term to a reference policy $\pi _ { \mathrm { r e f } }$ to maintain stability:

$$
\max _ {\theta} J _ {\beta} (\theta) = \mathbb {E} _ {x} \mathbb {E} _ {y \sim \pi_ {\theta} (\cdot | x)} \Big [ R ^ {*} (x, y) - \beta \operatorname{KL} (\pi_ {\theta} (\cdot | x) \| \pi_ {\mathrm{ref}} (\cdot | x)) \Big ],\tag{1}
$$

in which KL denotes the Kullback-Leibler (KL)-divergence. The policy gradient with respect to the unregularized reward is given by the REINFORCE estimator [31], $\begin{array} { r } { \nabla _ { \theta } J ( \theta ) = \mathbb E [ R ^ { * } ( x , y ) \sum _ { t } G _ { t } ] } \end{array}$ where $G _ { t } = \nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } )$ is the score function for the token at step $t ,$ and $y _ { < t }$ denotes the sequence of tokens generated prior to step t. In the following, we will denote $R ( x , y )$ with R for brevity. In practice, the clean reward $R ^ { * }$ is unavailable. Instead, the agent receives a noisy reward $\tilde { R } ( x , y ) \in \{ 0 , 1 \}$ from an automated verifier. We model this verifier as a stochastic reward channel that flips the latent clean reward to an observed noisy one. This channel is characterized by asymmetric, verifier-conditional noise rates.

Definition 3.1 (Verifier Reward Channel). The observed verifier reward $\tilde { R }$ is generated from the latent clean reward $R ^ { * }$ according to the following conditional probabilities:

$$
\mathbb {P} \big (\tilde {R} = 1 \mid R ^ {*} = 0 \big) = \rho_ {0}, \mathbb {P} \big (\tilde {R} = 0 \mid R ^ {*} = 1 \big) = \rho_ {1},\tag{2}
$$

where $\rho _ { 0 } , \rho _ { 1 } \in [ 0 , 1 )$ and $\rho _ { 0 } + \rho _ { 1 } < 1$

The condition $\rho _ { 0 } + \rho _ { 1 } < 1$ implies that the verifier channel is informative and invertible. Equivalently, the $2 \times 2$ binary transition matrix has determinant $1 - \rho _ { 0 } - \rho _ { 1 } > 0$

Assumptions and scope of the verifier channel.. Definition 3.1 is a deliberately lightweight abstraction of verifier unreliability. It assumes (i) instance-independent class-conditional noise rates $( \rho _ { 0 } , \rho _ { 1 } )$ that do not vary with $( x , y )$ , and (ii) conditional independence: given the latent correctness $R ^ { * } ( x , y )$ , the verifier’s stochastic flip $\tilde { R }$ is independent of the policy’s sampling randomness (hence independent of the score-function terms used by policy gradients). These assumptions are standard in class-conditional noise models and allow closed-form reward/gradient corrections. In real RLVR deployments, verifier errors can be content-dependent (e.g., spurious phrases can increase FPs), which violates (i) and can partially violate (ii). We therefore view $( \rho _ { 0 } , \rho _ { 1 } )$ as efective noise rates, and empirically test robustness to mis-specification in Section 4.4; we further estimate $\rho _ { 1 }$ online via a low-cost appeals mechanism in Section 4.3. When rates are instance-dependent, the same formulas are treated as correcting an efective marginal channel rather than eliminating all bias; Appendix B gives the resulting covariance residual.

Based on this noise model, we establish a linear relationship between the expected noisy reward and the clean reward, which is presented in Proposition 3.2.

Proposition 3.2 (Connection between Corrupted Rewards and True Rewards). Under the Verifier Reward Channel model, the expectation of the noisy reward $\tilde { R }$ conditioned on the clean reward $R ^ { * }$ is an afine transformation of $R ^ { * }$ :

$$
\mathbb {E} [ \tilde {R} \mid R ^ {*} ] = (1 - \rho_ {0} - \rho_ {1}) R ^ {*} + \rho_ {0}.
$$

The proof is provided in Appendix D.1. The central challenge of RLVR is that naively optimizing with the noisy reward $\tilde { R }$ leads to a biased policy gradient, causing the policy to learn from mistakes of the verifier. Our goal is to develop policy gradient estimators that use only the observable noisy reward $\tilde { R }$ but the expectation is either identical or parallel to the true policy gradient $\nabla _ { \boldsymbol { \theta } } J ( \boldsymbol { \theta } )$ We integrate these estimators within REINFORCE-style policy gradient algorithms widely used for reasoning tasks that compute normalized advantages over a group of sampled trajectories [27].

## 3.2. Noisy Policy Gradient with Backward Correction (PGBC)

Our first proposed approach aims to construct an unbiased estimator of the true reward $R ^ { * }$ by “inverting” the noise process. Since the expected noisy reward $\mathbb { E } [ \tilde { R } \mid R ^ { * } ]$ is an afine transformation of the clean reward $R ^ { * }$ in Proposition 3.2, we can solve for $R ^ { * }$ to derive a corrected reward estimator, ${ \widehat { R } } ,$ that is unbiased in expectation.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Noisy Policy Gradient with Backward Correction (PGBC)

Input: Initial policy  $\theta_{0}$ ; learning rate  $\eta$ ; batch size M; estimates of noise rates,  $(\hat{\rho}_{0}, \hat{\rho}_{1})$ .

loop

Roll out M trajectories  $\{(x_{i}, y_{i})\}_{i=1}^{M}$ , obtaining observed rewards  $\{\tilde{R}_{i}\}_{i=1}^{M}$ .

For each trajectory i, compute the unbiased reward estimate:

 $\widehat{R}_{i} \leftarrow \frac{\tilde{R}_{i} - \hat{\rho}_{0}}{1 - \hat{\rho}_{0} - \hat{\rho}_{1}}.$ 

Compute the policy gradient using the corrected rewards  $\{\widehat{R}_{i}\}_{i=1}^{M}$ .

Update parameters:  $\theta \leftarrow \theta + \eta \Delta\theta$ .

end loop
</div>

Theorem 3.3 (Oracle-Rate Unbiased Reward Estimator for the Binary Verifier Channel). Given the verifier channel with known noise rates $\rho _ { 0 }$ and $\rho _ { 1 }$ with $\rho _ { 0 } + \rho _ { 1 } < 1$ , the estimator

$$
\widehat {R} = \frac {\tilde {R} - \rho_ {0}}{1 - \rho_ {0} - \rho_ {1}}\tag{3}
$$

is an unbiased estimator of the true reward $R ^ { * } ( x , y ) , i . e . , \mathbb { E } [ \widehat { R } ] = R ^ { * }$

The proof is provided in Appendix D.2. The theorem is an oracle-rate population statement: unbiasedness holds when the channel rates used in the correction equal the true efective rates and before finite-sample GRPO normalization, clipping, or KL terms are applied. Theorem 3.3 shows that, by replacing the noisy reward $\tilde { R }$ with $\widehat { R }$ in any standard policy gradient formulation, we obtain an unbiased estimate of the true policy gradient. This allows $\widehat { R }$ to serve as a dropin replacement for the reward signal in complex RL algorithms, seamlessly integrating with advantage estimation and other machinery. The details of the algorithm are summarized in Algorithm 1. After rolling out a set of trajectories and obtaining their rewards, we use the corrected rewards to compute the policy gradients for model update.

## 3.3. Noisy Policy Gradient with Forward Correction (PGFC)

While PGBC provides an elegant unbiased estimator, it faces two practical challenges. First, the denominator $\left( 1 - \rho _ { 0 } - \rho _ { 1 } \right)$ can be small if the total noise rate is high, leading to high variance in the reward estimate $\widehat { R }$ and potentially unstable training. Second, it requires accurate estimation of both the false positive rate $\rho _ { 0 }$ and the false negative rate $\rho _ { 1 }$ , which are usually dificult to obtain in practice. The PGFC approach is introduced to mitigate these issues.

Instead of correcting the reward itself, this method directly modifies the policy gradient estimator to ensure its expectation is correctly aligned. The core idea is to reweight the score function $G _ { t }$ under the $^ { t - \mathrm { t h } }$ token based on the observed noisy reward ${ \tilde { R } } .$ We define an update term $h _ { t } = w _ { \tilde { R } } G _ { t }$ and choose the forward weights $w _ { 0 }$ (for $\tilde { R } = 0 )$ and $w _ { 1 }$ (for $\tilde { R } = 1 )$ such that the expected update $\mathbb { E } [ h _ { t } ]$ is parallel to the clean gradient $\nabla _ { \boldsymbol { \theta } } J ( \boldsymbol { \theta } )$ . The key insight is to choose the forward weights so that the conditional expectations given the true reward $R ^ { * }$ have a specific structure.

Proposition 3.4 (Conditional Expectation of Forward Weights). Let the forward weights be defined as $w _ { 0 } = \rho _ { 1 } - 1$ and $w _ { 1 } = \rho _ { 1 }$ . Under the Verifier Reward Channel model, the conditional expectations of the corresponding weight $w _ { \tilde { R } }$ given the true reward $R ^ { * }$ are:

1. $\mathbb { E } [ w _ { \tilde { R } } \mid R ^ { * } = 1 ] \ = \ 0 .$

$$
2. \mathbb {E} [ w _ {\tilde {R}} \mid R ^ {*} = 0 ] = - (1 - \rho_ {0} - \rho_ {1}).
$$

The proof can be found in Appendix D.3. Proposition 3.4 reveals that, in expectation, the forward weights are chosen so that their conditional expectation vanishes when $R ^ { * } = 1$ , i.e., truly positive cases do not require correction. Consequently, the explicit contribution to the correction term arises from samples with $R ^ { * } = 0$ . This structure allows us to recover the correct gradient direction, as shown in the following theorem.

Theorem 3.5 (REINFORCE-Style Directional Alignment of the Forward Correction). Let the gradient-update term be $\begin{array} { r } { \Delta \theta = \frac { 1 } { M } \sum w _ { \tilde { R } } G _ { t } } \end{array}$ , where $w _ { 0 } = \rho _ { 1 } - 1$ and $w _ { 1 } = \rho _ { 1 }$ with M trajectories. Under the Verifier Reward Channel model, the expected update is parallel to the clean policy gradient:

$$
\mathbb {E} [ \Delta \theta ] = c \nabla_ {\theta} J (\theta),\tag{4}
$$

in which $c = \left( 1 - \rho _ { 0 } - \rho _ { 1 } \right)$

The proof is given in Appendix D.4. Since $1 - \rho _ { 0 } - \rho _ { 1 } > 0$ , Theorem 3.5 guarantees that the expected update of the policy is in the same direction as the true gradient; meanwhile, the positive scaling factor $( 1 - \rho _ { 0 } - \rho _ { 1 } )$ can be absorbed into the learning rate. By avoiding the inverse operation of PGBC, PGFC circumvents the variance-inflation problem.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Noisy Policy Gradient with Forward Correction (PGFC)

Input: Initial policy  $\theta_{0}$ ; learning rate  $\eta$ ; batch size M; an estimate of the false negative rate,  $\hat{\rho}_{1}$ .

loop

Roll out M trajectories and observe rewards  $\{\tilde{R}_{i}\}_{i=1}^{M}$ .

For each trajectory i, define the weight based on the observed reward  $\tilde{R}_{i}$ :

 $w_{\tilde{R}_{i}} \leftarrow \begin{cases} \hat{\rho}_{1} - 1, &amp; \text{if } \tilde{R}_{i} = 0, \\ \hat{\rho}_{1}, &amp; \text{if } \tilde{R}_{i} = 1. \end{cases}$ 

For each trajectory i, compute the score-function term  $G_{i}$  and the weighted term  $h_{i} \leftarrow w_{\tilde{R}_{i}} G_{i}$ .

Form the policy gradient estimate:  $\hat{g} \leftarrow \frac{1}{M} \sum_{i=1}^{M} h_{i}$ .

Update parameters:  $\theta \leftarrow \theta + \eta \hat{g}$ .

end loop
</div>

In practice, we estimate $\hat { \rho } _ { 1 }$ during training by appealing a small, uniformly random subset of rulebased negatives to a lightweight LLM verifier (e.g., TinyV [38]) and then smoothing the empirical flip rate. Concretely, at step $t ,$ let $\mathcal { N } _ { \mathrm { R } } ^ { ( t ) }$ be items labeled negative by the rule-based verifier, $\mathcal { P } _ { \mathrm { R } } ^ { ( t ) }$ the rule-based positives, and sample a fraction $q \in ( 0 , 1 ]$ of $\mathcal { N } _ { \mathrm { R } } ^ { ( t ) }$ for appeal; denote by $\mathcal { P } _ { \mathrm { L } } ^ { ( t ) }$ those appealed items that the LLM flips to positive. Using a Horvitz–Thompson correction [13], we estimate the number of FNs as $| \mathcal { P } _ { \mathrm { L } } ^ { ( t ) } | / q$ , treat $| \mathcal { P } _ { \mathrm { R } } ^ { ( t ) } |$ as true positives in the rule-checker regime, and set $\begin{array} { r } { \hat { \rho } _ { 1 } ^ { ( t ) } = \frac { | \mathcal { P } _ { \mathrm { L } } ^ { ( t ) } | / q + \alpha } { | \mathcal { P } _ { \mathrm { L } } ^ { ( t ) } | / q + | \mathcal { P } _ { \mathrm { R } } ^ { ( t ) } | + \alpha + \beta } } \end{array}$ with small Beta-prior pseudocounts α, $\beta = 1 0 ^ { - 5 }$ . We then apply exponential moving average (EMA) smoothing over a sliding window. If both $\hat { \rho } _ { 0 }$ and $\hat { \rho } _ { 1 }$ are available, one can use PGBC; if only $\hat { \rho } _ { 1 }$ is available, PGFC is the default. We released our code in https://github.com/caixq1996/noisy-RLVR. More details of appealing mechanism can be found in Appendix E.4.

Bridging REINFORCE-Style Corrections to GRPO Updates. Our analysis is stated for the rewardgradient component of a score-function estimator, while GRPO implements this component through scalar trajectory coeficients broadcast as token-level advantages inside a clipped surrogate. We therefore insert each correction at the coeficient level that best matches its estimator:

(a) Qwen2.5-Math-1.5B

(b) DeepSeek-R1-Distill-Qwen-1.5B

(c) Llama-3.2-3B-Instruct

(d) Qwen2.5-Math-7B  
Figure 2: Synthetic-Noise Results (pass@1) with 16 samples and 5 random seeds on the four backbones. Base: baseline without RL; Oracle: Training with clean rewards; Noise: Training with noisy verifier rewards; Noise BC: Training with noise under backward correction; Noise FC: Training with noise under forward correction.  
PGBC is implemented as a corrected-utility coeficient based on $\widehat { r } _ { i } = ( \tilde { r } _ { i } - \hat { \rho } _ { 0 } ) / ( 1 - \hat { \rho } _ { 0 } - \hat { \rho } _ { 1 } )$ whereas PGFC is implemented as a score-level coeficient $w _ { \tilde { r } _ { i } } \in \{ \hat { \rho } _ { 1 } - 1 , \hat { \rho } _ { 1 } \}$ , not as a reward proxy. In both cases, the corrected coeficient is passed directly to the GRPO surrogate, with the KL term unchanged. This implementation avoids a subtle binary-reward degeneracy: any pre-normalization proxy afine in $\tilde { r } _ { i }$ can be removed by group centering and standardization, making the corrected update identical to uncorrected noisy-reward GRPO. Appendix A states the centered-update guarantee and Appendix B gives the implementation recipe.

## 4. Experiments

We evaluate our approach under both synthetic and real-world verifier noise. We first spell out the experimental protocol, and then present main and ablation results. Additional experimental results, including training dynamics, pass@8 performance under real-world noise, appeals statistics, complementarity with stronger verifiers, and non-math evaluations are provided in Appendix C.

(a) Qwen2.5-Math-1.5B

(b) DeepSeek-R1-Distill-Qwen-1.5B

(c) Llama-3.2-3B-Instruct

(d) Qwen2.5-Math-7B  
Figure 3: Synthetic-Noise Results (pass@8) with 16 samples and 5 random seeds on the four backbones. Base: baseline without RL; Oracle: Training with clean rewards; Noise: Training with noisy verifier rewards; Noise BC: Training with noise under backward correction; Noise FC: Training with noise under forward correction.

## 4.1. Experimental Setup

We train on three small backbones, Qwen2.5-Math-1.5B, DeepSeek-R1-Distill-Qwen-1.5B, and Llama-3.2-3B-Instruct, and probe scale with Qwen2.5-Math-7B. Rewards come from either (i) a rule-based checker that extracts the final \boxed{\cdot} answer and tests numeric/rational equivalence or (ii) a lightweight LLM verifier (TinyV 1.5B) used for appeals/estimation; prior work motivates explicit noise modeling due to systematic FP/FN behavior in these verifiers. Unless stated otherwise, we follow the DeepScaleR corpus and a GRPO-style on-policy recipe with BoN sampling, implementing our backward and forward corrections as drop-in hooks at the advantage-construction stage within VERL. Evaluation uses six verifiable math suites—AIME-2024, AIME-2025, AMC-2023, MATH500, Minerva MATH, and OlympiadBench—reportin Pass@1 and Pass@8 with 16 samples and 5 random seeds in average. Compute is 8×A100 (40GB) GPUs servers; unless noted, KL schedules, sampling temperatures, and other rollout settings are kept identical across compared conditions. More details can be found in Appendix E.

## 4.2. Synthetic Noise

To disentangle optimization efects from verifier unreliability, we inject verifier-conditional noise into the binary reward stream during training. Concretely, when the latent clean reward is $R ^ { * } \in \{ 0 , 1 \}$ , the observed reward $\tilde { R }$ is drawn from a reward channel with $\operatorname* { P r } ( \tilde { R } { = } 1 | R ^ { * } { = } 0 ) = \rho _ { 0 }$ and $\operatorname* { P r } ( \tilde { R } { = } 0 | R ^ { * } { = } 1 ) = \rho _ { 1 } ;$ unless stated otherwise we use $\rho _ { 0 } { = } 0 . 1$ and $\rho _ { 1 } = 0 . 2$ . We train GRPO on DeepScaleR with identical sampling and KL settings across conditions, and evaluate Pass@1 (16 samples) and Pass@8 on AIME-2024/2025, AMC-2023, MATH500, Minerva MATH, and OlympiadBench. We compare five variants: Base (no RL), Oracle (clean rewards), Noise (uncorrected), and our two corrections—PGBC (backward correction using $\begin{array} { r } { \widehat { R } = \frac { \tilde { R } - \rho _ { 0 } } { 1 - \rho _ { 0 } - \rho _ { 1 } } ) } \end{array}$ and PGFC (forward correction that rescales the gradient using $\rho _ { 1 }$ only).

As illustrated in Figure 2 and Figure 3, across models and benchmarks, the injected noise degrades uncorrected GRPO substantially, while both corrections recover most of the gap to the oracle. On DeepSeek-R1-Distill-Qwen-1.5B, uncorrected training under noise underperforms the oracle across all tasks (e.g., a noticeably lower Average), whereas PGBC and PGFC nearly match oracle performance; PGFC is consistently as good as or slightly better than PGBC, echoing its variance advantages from avoiding division by $\left( 1 - \rho _ { 0 } - \rho _ { 1 } \right)$ . From all scales of models, we observe the same pattern: noise hurts, PGBC/PGFC close the gap compared with Oracle. Additionally, due to a finite-sample / seed-level efect, potentially combined with mild optimization regularization efects, even the Oracle condition varies across seeds, and we observe PGFC and PGBC occasionally outperform Oracle. On the other hand, in RL, a small amount of injected noise can occasionally improve final policy quality by encouraging more efective exploration and by acting as an implicit regularizer [25, 4], so a corrected method slightly outperforming the oracle in one setting is not that surprising. But we do not interpret this as evidence that the corrected method is systematically better than oracle rewards.

## 4.3. Real-World Noise

As discussed above, automated verifiers exhibit both false positives (LLM judges over-crediting incorrect solutions) and false negatives (rule-based checkers rejecting correct ones) [5, 38, 9, 17]. In math RL pipelines, the latter is particularly prevalent: brittle exact-match or limited equivalence rules lead to many valid answers being scored as incorrect, depriving the agent of learning signal. Motivated by this, in this subsection we specifically investigate whether denoising false negatives—i.e., estimating $\hat { \rho } _ { 1 }$ as in Section 3.3—can measurably improve policy optimization. Concretely, we keep a fast rule-based checker as the primary reward source and, whenever it returns $\tilde { R } { = } 0$ , trigger an appeals pass with a lightweight LLM verifier (TinyV) [38]. From disagreements on negatives we maintain an online estimate $\hat { \rho } _ { 1 }$ (EMA over a sliding window to track policy drift).

Results in Table 1 show a consistent pattern across backbones and benchmarks. Using a LLM directly as the reward source (LLM-as-Judge) underperforms the rule-based pipeline, corroborating prior observations that LLM judges are bias-prone and gameable [5, 29]. Employing appeals to recheck rule-based negatives (Appeals) reduces FNs and yields a modest but reliable lift, yet the benefit remains indirect because the primary gradient is still driven by noisy binary rewards. In contrast, Appeals+PGFC uses the same appeals stream to estimate $\hat { \rho } _ { 1 }$ and directly applies the FN correction to the policy gradient via weights $w _ { \tilde { R } } \in \{ \hat { \rho } _ { 1 } - 1 , \hat { \rho } _ { 1 } \}$ , delivering the strongest and most stable improvements.

## 4.4. Robustness to Noise Mis-Specification

There now exist practical procedures to estimate verifier FP/FN rates via rule-based equivalence checkers (e.g., Math-Verify), lightweight appeals/judges such as TinyV, and meta-evaluation suites that quantify verifier reliability [10, 38, 17]. In real deployments, however, these estimates can be imperfect. We therefore test how backward and forward corrections behave under misspecified noise rates. We follow the synthetic-noise setting from Section 4.2 with Qwen2.5-math 1.5B: rewards are corrupted at data-time with $\rho _ { 0 } { = } 0 . 1 , ~ \rho _ { 1 } { = } 0 . 2$ . During training, we intentionally feed each algorithm mis-specified rates from a grid $\hat { \rho } _ { 0 } \in [ 0 . 0 , 0 . 2 ]$ $\hat { \rho } _ { 1 } \in [ 0 . 1 , 0 . 3 ]$ , and report the Average Pass@1 (16 samples) across our six math benchmarks. Because the forward method only requires the false negative rate, we vary $\hat { \rho } _ { 1 }$ for forward correction while keeping $\hat { \rho } _ { 0 }$ unused, as shown in Fig. 4.

Backward correction remains strong when the total noise is underestimated, but performance degrades as we overestimate the rates (i.e., as $1 - \hat { \rho } _ { 0 } - \hat { \rho } _ { 1 }$ shrinks). Intuitively, the unbiased de-biasing, $\begin{array} { r } { { \widehat R } = \frac { { \widetilde R } - { \widehat \rho } _ { 0 } } { 1 - { \widehat \rho } _ { 0 } - { \widehat \rho } _ { 1 } } } \end{array}$ , involves an inverse factor that amplifies reward variance and accentuates estimation error when the efective denominator becomes small. In contrast, forward

Table 1: Mean and standard deviation (pass@1) with 16 samples and 5 random seeds under realworld noise. Rule: rule-based rewards; LLM-as-Judge: direct LLM-judge rewards; Appeals: rulebased reward plus LLM appeals on negative samples without gradient correction; Appeals+PGFC: appeals-based FN-rate estimation plus forward correction.

<table><tr><td>Dataset</td><td>AIME2024</td><td>AIME2025</td><td>AMC2023</td><td>MATH500</td><td>Minerva MATH</td><td>Olympiad Bench</td><td>Average</td></tr><tr><td colspan="8">Qwen2.5-Math-1.5B</td></tr><tr><td>Base</td><td>6.0 ± 1.9</td><td>4.0 ± 0.6</td><td>34.2 ± 0.2</td><td>47.5 ± 0.3</td><td>5.1 ± 0.4</td><td>25.1 ± 0.4</td><td>20.3 ± 0.6</td></tr><tr><td>Rule</td><td>15.0 ± 0.4</td><td>5.6 ± 0.6</td><td>50.3 ± 0.6</td><td>69.4 ± 0.4</td><td>17.8 ± 0.6</td><td>31.6 ± 0.0</td><td>31.6 ± 0.4</td></tr><tr><td>LLM-as-Judge</td><td>10.9 ± 1.3</td><td>4.7 ± 1.0</td><td>42.1 ± 1.8</td><td>63.0 ± 0.7</td><td>15.9 ± 0.7</td><td>25.3 ± 0.5</td><td>27.0 ± 1.0</td></tr><tr><td>Appeals</td><td>11.9 ± 0.6</td><td>5.8 ± 1.2</td><td>47.8 ± 1.2</td><td>68.3 ± 0.1</td><td>16.7 ± 0.6</td><td>29.8 ± 0.1</td><td>30.1 ± 0.6</td></tr><tr><td>Appeals+PGFC (Ours)</td><td>20.3 ± 0.0</td><td>10.7 ± 1.7</td><td>53.3 ± 1.4</td><td>68.6 ± 0.8</td><td>16.5 ± 0.4</td><td>32.9 ± 0.2</td><td>33.7 ± 0.8</td></tr><tr><td colspan="8">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Base</td><td>9.0 ± 0.6</td><td>9.4 ± 0.6</td><td>41.4 ± 1.4</td><td>61.1 ± 0.1</td><td>10.5 ± 0.6</td><td>22.9 ± 0.4</td><td>25.7 ± 0.6</td></tr><tr><td>Rule</td><td>19.0 ± 1.0</td><td>17.5 ± 0.8</td><td>65.6 ± 2.5</td><td>77.6 ± 0.4</td><td>19.9 ± 0.0</td><td>38.5 ± 0.3</td><td>39.7 ± 0.8</td></tr><tr><td>LLM-as-Judge</td><td>11.9 ± 1.0</td><td>12.7 ± 1.7</td><td>52.3 ± 1.6</td><td>69.8 ± 0.8</td><td>14.2 ± 1.0</td><td>31.9 ± 0.5</td><td>32.1 ± 1.1</td></tr><tr><td>Appeals</td><td>21.7 ± 0.4</td><td>17.3 ± 0.6</td><td>66.2 ± 0.3</td><td>77.4 ± 0.0</td><td>20.0 ± 0.6</td><td>37.9 ± 0.1</td><td>40.1 ± 0.3</td></tr><tr><td>Appeals+PGFC (Ours)</td><td>23.2 ± 1.2</td><td>22.5 ± 0.8</td><td>70.7 ± 0.6</td><td>78.2 ± 0.0</td><td>19.4 ± 0.4</td><td>41.0 ± 0.4</td><td>42.5 ± 0.6</td></tr><tr><td colspan="8">Llama-3.2-3B-Instruct</td></tr><tr><td>Base</td><td>5.7 ± 1.2</td><td>0.6 ± 0.4</td><td>17.2 ± 1.5</td><td>34.8 ± 0.7</td><td>4.8 ± 0.0</td><td>12.7 ± 0.2</td><td>12.6 ± 0.7</td></tr><tr><td>Rule</td><td>6.0 ± 1.0</td><td>1.0 ± 0.4</td><td>24.3 ± 1.7</td><td>40.4 ± 0.5</td><td>9.2 ± 0.0</td><td>15.0 ± 0.5</td><td>16.0 ± 0.7</td></tr><tr><td>LLM-as-Judge</td><td>1.4 ± 1.1</td><td>0.1 ± 0.3</td><td>12.7 ± 2.1</td><td>26.4 ± 0.2</td><td>3.7 ± 0.0</td><td>7.7 ± 0.4</td><td>8.7 ± 0.7</td></tr><tr><td>Appeals</td><td>3.8 ± 1.4</td><td>0.6 ± 0.7</td><td>16.9 ± 1.8</td><td>40.8 ± 0.3</td><td>8.8 ± 0.0</td><td>13.7 ± 0.3</td><td>14.1 ± 0.7</td></tr><tr><td>Appeals+PGFC (Ours)</td><td>9.4 ± 1.9</td><td>0.4 ± 0.6</td><td>21.2 ± 1.3</td><td>46.4 ± 0.3</td><td>9.5 ± 0.2</td><td>16.3 ± 0.3</td><td>17.2 ± 0.8</td></tr><tr><td colspan="8">Qwen2.5-Math-7B</td></tr><tr><td>Base</td><td>12.7 ± 0.9</td><td>5.8 ± 0.7</td><td>44.4 ± 2.2</td><td>52.0 ± 0.4</td><td>9.8 ± 0.7</td><td>26.4 ± 0.4</td><td>25.2 ± 0.9</td></tr><tr><td>Rule</td><td>29.2 ± 2.5</td><td>13.5 ± 1.2</td><td>62.8 ± 1.9</td><td>78.9 ± 0.5</td><td>24.2 ± 0.7</td><td>37.8 ± 0.5</td><td>41.1 ± 1.2</td></tr><tr><td>LLM-as-Judge</td><td>16.8 ± 2.1</td><td>6.8 ± 1.4</td><td>50.2 ± 1.8</td><td>62.5 ± 1.0</td><td>10.1 ± 0.4</td><td>31.0 ± 0.9</td><td>29.6 ± 1.3</td></tr><tr><td>Appeals</td><td>27.4 ± 1.9</td><td>11.8 ± 1.9</td><td>63.7 ± 1.6</td><td>74.9 ± 0.7</td><td>20.6 ± 0.8</td><td>37.6 ± 0.3</td><td>39.3 ± 1.2</td></tr><tr><td>Appeals+PGFC (Ours)</td><td>31.0 ± 1.5</td><td>14.6 ± 1.2</td><td>65.7 ± 1.4</td><td>81.6 ± 0.4</td><td>26.2 ± 0.5</td><td>39.3 ± 0.5</td><td>43.1 ± 0.9</td></tr></table>

correction exhibits a notably flat response curve around the true $\rho _ { 1 } ;$ perturbing $\hat { \rho } _ { 1 }$ changes only the per-sample weights and avoids any inverse operation, yielding lower-variance policy gradient estimates.

We also stress-test the constant-channel assumption using format-dependent, non-iid verifier noise with the same Qwen2.5-Math-1.5B / DeepScaleR setup. Both the FN and FP rates depend on a prompt-level tag and a response-level format tag (e.g., decimal or missing boxed answer vs. canonical format). Specifically, we use a shared FP rate of $\rho _ { 0 } = 0 . 2$ but a format-dependent FN rate: $\rho _ { 1 } = 0 . 0 5$ for low-risk formats and $\rho _ { 1 } = 0 . 4 5$ for high-risk formats, with the actual

(a) BC: $\hat { \rho } _ { 0 } { = } 0 . 1$ , ˆρ<sub>1</sub> ∈ [0.1, 0.3]

(b) BC: $\hat { \rho } _ { 1 } = 0 . 2$ , ˆρ<sub>0</sub> ∈ [0.0, 0.2]

(c) FC: $\hat { \rho } _ { 0 } { = } 0 . 1 .$ , ˆρ<sub>1</sub> ∈ [0.1, 0.3]  
Figure 4: Robustness results. (a) Backward correction (BC) with $\hat { \rho } _ { 0 }$ fixed and sweeping ˆρ<sub>1</sub>; (b) Backward correction (BC) with $\hat { \rho } _ { 1 }$ fixed and sweeping $\hat { \rho } _ { 0 } ;$ (c) Forward correction (FC) with $\hat { \rho } _ { 0 }$ fixed and sweeping $\hat { \rho } _ { 1 }$

Table 2: Format-dependent non-iid verifier-noise stress test on Qwen2.5-Math-1.5B. Each cell reports pass@1 / pass@8.

<table><tr><td>Method</td><td>MATH500</td><td>AIME2024</td><td>AIME2025</td><td>AMC2023</td><td>Minerva Math</td><td>OlympiadBench</td><td>Average</td></tr><tr><td>Rule</td><td>47.5 / 52.6</td><td>7.1 / 14.2</td><td>4.5 / 8.3</td><td>33.8 / 42.8</td><td>4.9 / 6.2</td><td>24.6 / 30.8</td><td>20.4 / 25.8</td></tr><tr><td>PGBC</td><td>66.7 / 68.2</td><td>10.9 / 15.5</td><td>7.0 / 11.8</td><td>44.9 / 55.6</td><td>16.8 / 17.7</td><td>30.4 / 32.6</td><td>29.5 / 33.6</td></tr><tr><td>PGFC</td><td>68.6 / 69.7</td><td>12.9 / 19.2</td><td>7.5 / 12.2</td><td>49.1 / 61.5</td><td>18.7 / 19.6</td><td>30.5 / 32.7</td><td>31.2 / 35.8</td></tr></table>

flip decision made by hashing (data source, ground truth, solution str). This explicitly violates the instance-independent channel model, so the result should be read as an efectivechannel robustness test rather than as a proof that the model is literally correct. Table 2 shows that both corrections still improve over uncorrected training under this non-iid noise setting, with PGFC again strongest performance.

## 5. Conclusion

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a practical and popular paradigm for training reasoning-focused large language models (LLMs). Yet binarized rewards inevitably sufer from false positives (FP) and false negatives (FN). Prior eforts largely mitigated these errors from the verifier side—e.g., debiasing LLM judges or strengthening equivalence checking—to reduce the downstream impact on learning [17, 9]. In contrast, we modeled verifier unreliability as a verifier-conditional noise channel with the FP and FN rates, and introduced two theory-driven corrections that act directly at the policy-learning interface. The backward correction de-biases the observed reward to yield an unbiased gradient estimator; it requires estimates of the FP and FN rates but is agnostic to the choice of policy-optimization algorithm and can thus be used beyond policy gradient methods. The forward correction rescales score-function terms so that the expected update is aligned with the clean gradient; notably, it needs only the FN rate estimate, which is often the dominant source of real-world noise due to rule-based FNs [9, 38].

While we instantiated appeals with a lightweight LLM verifier for eficiency, the same backward/forward correction mechanisms should also gain from stronger verifiers [38]. Meanwhile, although modeled with the constant noise rates, the proposed methods already show clear advantages, even under a more challenging non-iid verifier noise setting. This means that the constant noise rates approximate the real-world noise rates quite well. On the other hand, real-world noise can depend on content, formatting, prompt style, and verifier family, motivating future work on instance-dependent reliability models and adversarially robust verifiers.

## References

[1] Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. Humans or llms as the judge? a study on judgement bias. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8301–8327, 2024.

[2] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. 2018.

[3] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[4] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Matteo Hessel, Ian Osband, Alex Graves, Volodymyr Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and Shane Legg. Noisy networks for exploration. In International Conference on Learning Representations, 2018.

[5] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen

Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

[6] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. Advances in neural information processing systems, 31, 2018.

[7] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 3828– 3850. Association for Computational Linguistics, 2024.

[8] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

[9] Yuzhen Huang, Weihao Zeng, Xingshan Zeng, Qi Zhu, and Junxian He. Pitfalls of rule- and model-based verifiers–a case study on mathematical reasoning. arXiv preprint arXiv:2505.22203, 2025.

[10] Hugging Face. Math-verify: A robust mathematical expression evaluation system designed for assessing large language model outputs in mathematical tasks. GitHub repository, 2025. URL https://github.com/huggingface/Math-Verify.

[11] HuggingFaceH4. Aime 2024 (dataset card). Hugging Face, 2024. URL https:// huggingface.co/datasets/HuggingFaceH4/aime\_2024.

[12] Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International conference on machine learning, pp. 2304–2313. PMLR, 2018.

[13] Vishesh Karwa and Edoardo M Airoldi. On the admissibility of horvitz-thompson estimator for estimating causal efects under network interference. arXiv preprint arXiv:2312.01234, 2023.

[14] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

[15] Junnan Li, Richard Socher, and Steven C. H. Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.

[16] Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-toend label-noise learning without anchor points. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 6403–6413. PMLR, 2021.

[17] Xuzhao Li, Xuchen Li, Shiyu Hu, Yongzhen Guo, and Wentao Zhang. Verifybench: A systematic benchmark for evaluating reasoning verifiers across domains. arXiv preprint arXiv:2507.09884, 2025.

[18] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

[19] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jefrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/ DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca30 2025. Notion Blog.

[20] math-ai. Amc 2023 (dataset card). Hugging Face, 2025. URL https://huggingface.co/ datasets/math-ai/amc23.

[21] Youssef Mroueh. Reinforcement learning with verifiable rewards: Grpo’s efective loss, dynamics, and success amplification. arXiv preprint arXiv:2503.06639, 2025.

[22] Nagarajan Natarajan, Inderjit S. Dhillon, Pradeep Ravikumar, and Ambuj Tewari. Learning with noisy labels. In Christopher J. C. Burges, L´eon Bottou, Zoubin Ghahramani, and Kilian Q. Weinberger (eds.), Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pp. 1196–1204, 2013.

[23] OpenCompass. Aime 2025 (dataset card). Hugging Face, 2025. URL https://huggingface. co/datasets/opencompass/AIME2025.

[24] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 2233–2241. IEEE Computer Society, 2017.

[25] Matthias Plappert, Rein Houthooft, Prafulla Dhariwal, Szymon Sidor, Richard Y. Chen, Xi Chen, Tamim Asfour, Pieter Abbeel, and Marcin Andrychowicz. Parameter space noise for exploration. In International Conference on Learning Representations, 2018.

[26] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024.

[27] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. 2024. URL https://arxiv.org/abs/ 2402.03300.

[28] Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, and Neil Zhenqiang Gong. Optimization-based prompt injection attack to llm-as-a-judge. In Bo Luo, Xiaojing Liao, Jun Xu, Engin Kirda, and David Lie (eds.), Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, CCS 2024, Salt Lake City, UT, USA, October 14-18, 2024, pp. 660–674. ACM, 2024.

[29] Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. Judging the judges: A systematic study of position bias in llm-as-a-judge. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 292–314, 2025.

[30] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE transactions on neural networks and learning systems, 34(11):8135–8153, 2022.

[31] Richard S. Sutton, David A. McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Sara A. Solla, Todd K. Leen, and Klaus-Robert M¨uller (eds.), Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999], pp. 1057–1063. The MIT Press, 1999.

[32] Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. Judging the judges: Evaluating alignment and vulnerabilities in llms-as-judges. arXiv preprint arXiv:2406.12624, 2024.

[33] Jingkang Wang, Yang Liu, and Bo Li. Reinforcement learning with perturbed rewards. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pp. 6202–6209. AAAI Press, 2020.

[34] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference

on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

[36] Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, Jiang Bian, and Mao Yang. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

[37] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.

[38] Zhangchen Xu, Yuetai Li, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. Tinyv: Reducing false negatives in verification improves rl for llm reasoning. arXiv preprint arXiv:2505.14625, 2025.

[39] Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Tom Grifiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

[40] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 4791–4800, 2019.

[41] Yulai Zhao, Haolin Liu, Dian Yu, Sunyuan Kung, Meijia Chen, Haitao Mi, and Dong Yu. One token to fool llm-as-a-judge. arXiv preprint arXiv:2507.08794, 2025.

[42] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

## A. From REINFORCE-Style Corrections to GRPO Updates

Setup (single prompt and group).. Fix a prompt x and sample a group of K i.i.d. responses $y _ { 1 } , \dots , y _ { K } \sim \pi _ { \theta } ( \cdot \ | \ x )$ . Let $\begin{array} { r } { G _ { i } \triangleq \sum _ { t = 1 } ^ { | y _ { i } | } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i , t } \ | \ x , y _ { i , < t } ) } \end{array}$ denote the trajectory scorefunction. Let $u _ { i }$ be any scalar proxy reward computed from the trajectory. For PGBC, $u _ { i } =$ $\widehat { r _ { i } }$ . For the population PGFC theorem, the score-function term is weighted as $w _ { \tilde { r } _ { i } } G _ { i } ;$ for the default GRPO implementation, the pre-centering scalar proxy is instead $u _ { i } = \tilde { r } _ { i } w _ { \tilde { r } _ { i } }$ before group centering. Define the group mean $\begin{array} { r } { \bar { \boldsymbol { u } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \boldsymbol { u } _ { j } } \end{array}$ . Consider the centered score-function update

$$
\Delta_ {\mathrm{center}} (\theta) \triangleq \frac {1}{K} \sum_ {i = 1} ^ {K} (u _ {i} - \bar {u}) G _ {i}.\tag{5}
$$

Proposition A.1 (Group centering preserves expected direction). Assume $\{ ( u _ { i } , G _ { i } ) \} _ { i = 1 } ^ { K }$ are i.i.d. under $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ and $\mathbb { E } [ G _ { i } \mid x ] = 0$ (the standard score-function identity $\it { 1 3 7 , ~ 3 1 7 ) }$ . Then the centered update in (5) satisfies

$$
\mathbb {E} [ \Delta_ {c e n t e r} (\theta) \mid x ] = \left(1 - \frac {1}{K}\right) \mathbb {E} [ u G \mid x ].\tag{6}
$$

In particular, $i f \operatorname { \mathbb { E } } [ u G \mid x ]$ is parallel to the clean gradient direction for the prompt-conditioned objective, then so is $\mathbb { E } [ \Delta _ { c e n t e r } ( \theta ) \mid x ]$

Proof. Expand $\begin{array} { r } { \Delta _ { \mathrm { c e n t e r } } = \frac { 1 } { K } \sum _ { i } u _ { i } G _ { i } - \bar { u } \cdot \frac { 1 } { K } \sum _ { i } G _ { i } } \end{array}$ . Taking expectation conditioned on x yields

$$
\mathbb {E} [ \bar {u} \cdot \frac {1}{K} \sum_ {i} G _ {i} \mid x ] = \frac {1}{K ^ {2}} \sum_ {i, j} \mathbb {E} [ u _ {j} G _ {i} \mid x ].
$$

The $i = j$ terms contribute $\frac { K } { K ^ { 2 } } \mathbb { E } [ u G \ | \ x ]$ . For $i \neq j$ , independence gives $\mathbb { E } [ u _ { j } G _ { i } \ | \ x ] = \mathbb { E } [ u \ |$ $x ] \mathbb { E } [ G \mid x ] = 0$ . Thus $\begin{array} { r } { \mathbb { E } [ \bar { u } \cdot \frac { 1 } { K } \sum _ { i } G _ { i } \mid x ] = \frac { 1 } { K } \mathbb { E } [ u G \mid x ] } \end{array}$ , implying (6). □

Corollary A.2 (Directional correctness of PGFC under centered GRPO-style updates). Under the Verifier Reward Channel (Definition 3.1) and the conditional-independence assumption stated above, let $\boldsymbol { u } = \boldsymbol { w } _ { \tilde { R } }$ with $w _ { 0 } = \rho _ { 1 } - 1$ and $w _ { 1 } = \rho _ { 1 }$ as in Proposition $\ 3 . 4 \cdot$ Then for each prompt $x ,$

$$
\mathbb {E} [ \Delta_ {c e n t e r} (\theta) \mid x ] = \Big (1 - \frac {1}{K} \Big) (1 - \rho_ {0} - \rho_ {1}) \nabla_ {\theta} J (\theta \mid x),
$$

so the expected centered update remains aligned with the clean policy-gradient direction.

Binary-proxy equivalence caveat.. The corollary explains how centering interacts with a generic proxy whose score-function expectation is aligned with the clean gradient. It does not by itself prove that every PGFC proxy creates a distinct GRPO update. For binary rewards, $\begin{array} { r l } { w _ { \tilde { r } } } & { { } = } \end{array}$ $\tilde { r } + \rho _ { 1 } - 1$ difers from $\tilde { r }$ by an additive constant, so group centering removes the diference. Likewise, the implemented pre-centering proxy $u = \tilde { r } w _ { \tilde { r } } = \rho _ { 1 } \tilde { r }$ is a positive rescaling of ${ \tilde { r } } ;$ if within-group standardization is enabled, this rescaling also cancels. A PGFC configuration must therefore be audited on actual batches: the token-level advantages for Noise and PGFC should be compared after centering/normalization, and a zero-diference result means the run is not testing a distinct gradient correction. Distinct behavior can arise from non-binary verifier scores, disabled standardization, post-center weighting, process rewards, or online appeal streams that change the observed reward distribution.

Standardization, clipping, and KL (what is guaranteed, and what is approximate).. GRPO further divides centered proxy rewards by the within-group standard deviation to form normalized advantages, and then applies a PPO-style clipping on token-level ratios and an explicit KL penalty to $\pi _ { \mathrm { r e f } }$ [27]. The exact direction-preservation statement in Proposition A.1 covers the centering step, which is the main nontrivial departure from vanilla REINFORCE. The additional standardization is a multiplication by a random positive scalar; this primarily acts as an adaptive step-size (and cannot flip the sign of individual sample advantages), though a strict equality in expectation would require additional concentration assumptions. Clipping and the KL penalty add further stabilizers: in the small-update regime (encouraged by the KL term and typical learning rates), clipping is rarely active and the update is close to the unclipped score-function form; when clipping activates, it intentionally limits step size and may deviate from the exact REINFORCE direction. Importantly, our corrections target the reward-gradient component; the KL gradient is unchanged, so any global scaling factor $( \mathrm { e . g . , 1 - } \rho _ { 0 } - \rho _ { 1 } )$ can be absorbed into the efective learning rate and $/ \mathrm { o r }$ interpreted as modifying the reward–KL trade-of. The unified PGBC and PGFC with GRPO algorithm can be found in Algorithm 3.

## B. Implementation Details

This appendix describes the implementation of PGBC and PGFC inside a GRPO-style RLVR pipeline. The key design choice is to insert each correction at the level of the scalar coeficient multiplying the score-function term. PGBC is a corrected-utility method: it replaces the observed verifier reward by an estimate of the clean reward. PGFC is a score-corrected method: it replaces the usual reward-derived advantage by a signed channel-calibrated multiplier. In both cases, the corrected coeficient is passed directly to the PPO-style GRPO surrogate and is not converted into a binary reward proxy that is subsequently group-standardized.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 3 Unified reward-channel correction inside GRPO advantage construction

Input: group rewards  $\{\tilde{r}_{i}\}_{i=1}^{K}$ ; correction mode  $m \in \{none, PGBC, PGFC\}$ ; estimates  $(\hat{\rho}_{0}, \hat{\rho}_{1})$ ; baseline type.

for  $i = 1, \ldots, K$  do

if m = PGBC then

 $u_{i} \leftarrow (\tilde{r}_{i} - \hat{\rho}_{0}) / (1 - \hat{\rho}_{0} - \hat{\rho}_{1})$ 

else if m = PGFC then

 $w_{i} \leftarrow \hat{\rho}_{1}$  if  $\tilde{r}_{i} = 1$ , else  $\hat{\rho}_{1} - 1$ $u_{i} \leftarrow \tilde{r}_{i} w_{i}$  {implemented pre-center proxy; audit for binary equivalence}

else

 $u_{i} \leftarrow \tilde{r}_{i}$ 

end if

end for

Compute group or leave-one-out baseline  $b_{i}$  from  $\{u_{j}\}_{j=1}^{K}$ .

 $a_{i} \leftarrow u_{i} - b_{i}$ ; optionally divide by  $\text{std}(\{u_{j}\}) + \epsilon$ .

Broadcast  $a_{i}$  to response tokens and optimize the usual clipped GRPO objective with unchanged KL.
</div>

## B.1. Notation and GRPO Surrogate

For a prompt x, GRPO samples a group of K responses

$$
y _ {1}, \dots , y _ {K} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | x).
$$

Let $\tilde { r } _ { i } \in \{ 0 , 1 \}$ be the observed verifier reward for response $y _ { i }$ . For token t in response $i ,$ define the policy ratio

$$
r _ {i, t} (\theta) = \frac {\pi_ {\theta} (y _ {i , t} \mid x , y _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (y _ {i , t} \mid x , y _ {i , <   t})}.
$$

Given a scalar trajectory coeficient $a _ { i }$ , we broadcast it to all response tokens:

$$
A _ {i, t} \equiv a _ {i}, \qquad t = 1, \ldots , | y _ {i} |.
$$

The clipped GRPO reward surrogate is then

$$
\mathcal {L} _ {\mathrm{GRPO}} ^ {\mathrm{rew}} (\theta) = \frac {1}{K} \sum_ {i = 1} ^ {K} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \min (r _ {i, t} (\theta) A _ {i, t}, \operatorname{clip} (r _ {i, t} (\theta), 1 - \varepsilon , 1 + \varepsilon) A _ {i, t}).
$$

The full objective keeps the usual KL penalty:

$$
\mathcal {L} _ {\mathrm{GRPO}} (\theta) = \mathcal {L} _ {\mathrm{GRPO}} ^ {\mathrm{rew}} (\theta) - \beta \operatorname{KL} (\pi_ {\theta} (\cdot | x) \| \pi_ {\mathrm{ref}} (\cdot | x)).
$$

At $\theta \ : = \ : \theta _ { \mathrm { o l d } }$ , when clipping is inactive, the reward-gradient component reduces to the score function update

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm{GRPO}} ^ {\mathrm{rew}} (\theta) = \frac {1}{K} \sum_ {i = 1} ^ {K} a _ {i} \left(\frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid x, y _ {i, <   t})\right),
$$

up to the conventional length normalization used by the implementation. Therefore, the scalar coeficient $a _ { i }$ is the correct place to insert PGBC or PGFC.

## B.2. PGBC: Corrected-Utility GRPO

PGBC estimates the clean binary reward by inverting the verifier channel. Given estimates $\hat { \rho } _ { 0 } , \hat { \rho } _ { 1 }$ 2 define

$$
\hat {c} = 1 - \hat {\rho} _ {0} - \hat {\rho} _ {1}.
$$

We require $\hat { c } > 0$ and in code clamp it below by a small constant $\epsilon _ { c } > 0$ for numerical safety. The corrected utility is

$$
\widehat {r _ {i}} = \frac {\tilde {r} _ {i} - \hat {\rho_ {0}}}{\hat {c}}.
$$

This quantity is used as the trajectory coeficient, optionally with a variance-reduction baseline:

$$
a _ {i} ^ {\mathrm{BC}} = \frac {\widehat {r} _ {i} - b _ {i} ^ {\mathrm{BC}}}{s _ {\mathrm{BC}}}.
$$

Here $s _ { \mathrm { B C } } > 0$ is a fixed or slowly moving positive scale, such as an EMA scale, and should not be the current group’s standard deviation. The default baseline is leave-one-out:

$$
b _ {i} ^ {\mathrm{BC}} = \frac {1}{K - 1} \sum_ {j \neq i} \widehat {r} _ {j} \quad (K > 1),
$$

with $b _ { i } ^ { \mathrm { B C } } = 0$ when $K = 1$ . The leave-one-out baseline is independent of $G _ { i }$ conditional on x, so before clipping and KL it preserves the expected score-function update:

$$
\mathbb {E} \left[ \frac {1}{K} \sum_ {i = 1} ^ {K} (\widehat {r} _ {i} - b _ {i} ^ {\mathrm{BC}}) G _ {i} \Bigg | x \right] = \mathbb {E} [ \widehat {r} G \mid x ].
$$

When the channel rates are correct, $\mathbb { E } [ \widehat { r } G \ | \ x ]$ equals the clean reward-gradient component. Using the group mean including the i-th sample is also possible, but it introduces the familiar multiplicative factor $1 - 1 / K$

The corrected coeficient $a _ { i } ^ { \mathrm { B C } }$ is then broadcast to all tokens:

$$
A _ {i, t} ^ {\mathrm{BC}} \equiv a _ {i} ^ {\mathrm{BC}}.
$$

No current-group z-score standardization is applied to $\{ \widehat { r } _ { i } \} _ { i = 1 } ^ { K }$ , because such standardization would remove the afine calibration induced by PGBC in the binary-reward case.

## B.3. PGFC: Score-Corrected GRPO

PGFC is implemented as a score-level correction rather than as a reward proxy. Given an estimate $\hat { \rho } _ { 1 }$ , define the signed score multiplier

$$
w _ {\tilde {r} _ {i}} = \tilde {r} _ {i} - (1 - \hat {\rho} _ {1}) = \left\{ \begin{array}{l l} \hat {\rho} _ {1} - 1, & \tilde {r} _ {i} = 0, \\ \hat {\rho} _ {1}, & \tilde {r} _ {i} = 1. \end{array} \right.
$$

The trajectory coeficient is

$$
a _ {i} ^ {\mathrm{FC}} = \frac {w _ {\tilde {r} _ {i}} - b _ {i} ^ {\mathrm{FC}}}{s _ {\mathrm{FC}}}.
$$

Our default is

$$
b _ {i} ^ {\mathrm{FC}} = 0.
$$

The reason is that subtracting the current group mean of $w _ { \tilde { r } _ { i } }$ would reduce PGFC to an ordinary relative noisy-reward update, because $w _ { \tilde { r } _ { i } }$ is an afine function of $\tilde { r } _ { i }$ . A fixed or historical baseline independent of the current sampled actions may be used, but current-group centering is not the default.

The scale $s _ { \mathrm { F C } } > 0$ controls the reward–KL trade-of. For pure score-function direction, any fixed positive scale preserves the direction. When both noise rates are available, one may set

$$
s _ {\mathrm{FC}} = 1 - \hat {\rho} _ {0} - \hat {\rho} _ {1}
$$

to approximately match the clean reward-gradient magnitude. In the FN-dominant rule-checker regime where $\rho _ { 0 } \approx 0$ , a practical choice is

$$
s _ {\mathrm{FC}} = 1 - \hat {\rho} _ {1}.
$$

If no calibrated scale is desired, we set $s _ { \mathrm { F C } } = 1$ and tune $\beta$ or the learning rate as usual.

The token-level advantage is

$$
A _ {i, t} ^ {\mathrm{FC}} \equiv a _ {i} ^ {\mathrm{FC}}.
$$

Before clipping and KL, if $\hat { \rho } _ { 1 } = \rho _ { 1 }$ , the population score-function component satisfies

$$
\mathbb {E} [ a _ {i} ^ {\mathrm{FC}} G _ {i} \mid x ] = \frac {1 - \rho_ {0} - \rho_ {1}}{s _ {\mathrm{FC}}} \nabla_ {\theta} J (\theta \mid x).
$$

Thus PGFC preserves the clean reward-gradient direction and, with a suitable positive scale, can also calibrate the reward-gradient magnitude. The KL gradient is not corrected; therefore s<sub>FC</sub> and $\beta$ jointly determine the efective reward–KL trade-of.

Remark on the role of $\hat { \rho } _ { 1 }$ .. The theoretical conditional-cancellation identity for PGFC uses the true $\rho _ { 1 }$ . In an unclipped population score-function estimator, adding any action-independent constant to $\tilde { r } _ { i }$ has zero expected gradient by the score-function identity. However, in finite-batch GRPO with clipping, KL regularization, length normalization, and nonstationary rollouts, the fixed channel baseline $1 - \hat { \rho } _ { 1 }$ afects the realized signed advantages and the clipped surrogate. We therefore treat $\hat { \rho } _ { 1 }$ as a channel-calibrated baseline parameter and audit the resulting token-level advantages.

## $B . 4$ . Why We Do Not Use Direct-Group z-Scoring for Binary Channel Corrections

Let $Z _ { i } ( u )$ denote the standard group-normalized advantage

$$
Z _ {i} (u) = \frac {u _ {i} - \bar {u}}{\operatorname{std} (\{u _ {j} \} _ {j = 1} ^ {K}) + \epsilon}, \quad \bar {u} = \frac {1}{K} \sum_ {j = 1} ^ {K} u _ {j}.
$$

If $u _ { i } = a \tilde { r } _ { i } + b$ with $a > 0$ , then

$$
u _ {i} - \bar {u} = a (\tilde {r} _ {i} - \bar {\tilde {r}}), \quad \operatorname{std} (\{u _ {j} \}) = a \operatorname{std} (\{\tilde {r} _ {j} \}),
$$

and therefore

$$
Z _ {i} (u) = \frac {\tilde {r} _ {i} - \bar {\tilde {r}}}{\mathrm{std} (\{\tilde {r} _ {j} \}) + \epsilon / a}.
$$

When ϵ is small and the group is non-degenerate, this is numerically identical to the uncorrected noisy-reward advantage.

This degeneracy applies to the most obvious binary implementations of both corrections:

$$
\widehat {r} _ {i} = \frac {\tilde {r} _ {i} - \hat {\rho} _ {0}}{1 - \hat {\rho} _ {0} - \hat {\rho} _ {1}} \mathrm{isaffinein} \tilde {r} _ {i},
$$

and

$$
w _ {\tilde {r} _ {i}} = \tilde {r} _ {i} + \hat {\rho} _ {1} - 1, \qquad \tilde {r} _ {i} w _ {\tilde {r} _ {i}} = \hat {\rho} _ {1} \tilde {r} _ {i} \quad \text { for } \tilde {r} _ {i} \in \{0, 1 \}.
$$

Thus, feeding any of these quantities into the usual current-group z-score advantage construction can make the corrected method exactly equivalent to the uncorrected noisy baseline. For this reason, our corrected implementations use the scalar coeficients $a _ { i } ^ { \mathrm { B C } }$ and $a _ { i } ^ { \mathrm { F C } }$ directly, with only fixed or running positive scales.

## B.5. Noise Control

To isolate the efect of the channel correction from the efect of changing the advantage plumbing, we use an implementation-matched noisy control. For this control, the token-level coeficient is

$$
a _ {i} ^ {\mathrm{Noise}} = \frac {\tilde {r} _ {i} - b _ {i} ^ {\mathrm{Noise}}}{s _ {\mathrm{Noise}}},
$$

where $b _ { i } ^ { \mathrm { N o i s e } }$ and $s _ { \mathrm { N o i s e } }$ follow the same convention as the corrected method being compared. For example, when comparing PGBC with leave-one-out baselines, the noisy control also uses a leaveone-out baseline. When comparing PGFC, the noisy control uses the same direct coeficient path, rather than the direct group-standardized GRPO path. This avoids confounding the correction with the removal of group standardization.

## B.6. Unified Outcome-Level Algorithm

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 4 Channel-Corrected GRPO Advantage Construction

Input: observed rewards $\{\tilde{r}_i\}_{i=1}^K$; mode $m \in \{Noise, PGBC, PGFC\}$; rate estimates $(\hat{\rho}_0, \hat{\rho}_1)$;

positive scales $s_{Noise}, s_{BC}, s_{FC}$.

for $i = 1, \ldots, K$ do

    if $m = PGBC$ then

    $\hat{c} \leftarrow \max(1 - \hat{\rho}_0 - \hat{\rho}_1, \epsilon_c)$ $\hat{r}_i \leftarrow (\tilde{r}_i - \hat{\rho}_0) / \hat{c}$

    else if $m = PGFC$ then

    $w_i \leftarrow \tilde{r}_i - (1 - \hat{\rho}_1)$

    else

    $u_i \leftarrow \tilde{r}_i$

    end if

end for

if $m = PGBC$ then

    for $i = 1, \ldots, K$ do

    $b_i^{BC} \leftarrow \frac{1}{K-1} \sum_{j \neq i} \hat{r}_j$ if $K &gt; 1$, else 0

    $a_i \leftarrow (\hat{r}_i - b_i^{BC}) / s_{BC}$

    end for

else if $m = PGFC$ then

    for $i = 1, \ldots, K$ do

    $a_i \leftarrow w_i / s_{FC}$ {default: no current-group centering}

    end for

else

    for $i = 1, \ldots, K$ do

    Choose $b_i^{Noise}$ and $s_{Noise}$ to match the comparison path.

    $a_i \leftarrow (\tilde{r}_i - b_i^{Noise}) / s_{Noise}$

    end for

end if

Broadcast $A_{i,t} \leftarrow a_i$ for all response tokens $t$.

Optimize the usual clipped GRPO surrogate with unchanged KL penalty.
</div>

## B.7. Process-Supervised Extension

For process supervision, suppose response $y _ { i }$ has step-level verifier rewards

$$
\tilde {r} _ {i} ^ {(1)}, \ldots , \tilde {r} _ {i} ^ {(J _ {i})}
$$

attached to step boundaries

$$
\tau_ {i} ^ {(1)} <   \dots <   \tau_ {i} ^ {(J _ {i})}.
$$

PGBC is applied to each step reward:

$$
\widehat {r} _ {i} ^ {(j)} = \frac {\tilde {r} _ {i} ^ {(j)} - \hat {\rho} _ {0}}{1 - \hat {\rho} _ {0} - \hat {\rho} _ {1}}.
$$

The token coeficient is the return-to-go of corrected utilities:

$$
A _ {i, t} ^ {\mathrm{BC}} = \frac {1}{s _ {\mathrm{BC}}} \sum_ {j: \tau_ {i} ^ {(j)} \geq t} \left(\widehat {r} _ {i} ^ {(j)} - b _ {i, j} ^ {\mathrm{BC}}\right),
$$

where $b _ { i , j } ^ { \mathrm { B C } }$ is an optional leave-one-out or historical baseline for the corresponding step.

PGFC is applied as a step-level signed score multiplier:

$$
w _ {i} ^ {(j)} = \tilde {r} _ {i} ^ {(j)} - (1 - \hat {\rho} _ {1}),
$$

and the token coeficient is

$$
A _ {i, t} ^ {\mathrm{FC}} = \frac {1}{s _ {\mathrm{FC}}} \sum_ {j: \tau_ {i} ^ {(j)} \geq t} w _ {i} ^ {(j)}.
$$

As in the outcome case, we avoid current-batch z-scoring of these binary afine quantities. If a positive normalization is needed for numerical stability, we use a fixed or EMA scale rather than the current batch’s standard deviation.

## B.8. When Conditional Independence Is Violated (Bias Decomposition).

Our directional-alignment results rely on conditional independence to factorize $\mathbb { E } [ w _ { \tilde { R } } G \mid R ^ { * } ] =$ $\mathbb { E } [ w _ { \tilde { R } } \ | \ R ^ { * } ] \mathbb { E } [ G \ | \ R ^ { * } ]$ . If verifier errors depend on trajectory features beyond correctness (e.g., style-based FP hacks), the update admits the decomposition

$$
\mathbb {E} [ w _ {\tilde {R}} G \mid R ^ {*} ] = \mathbb {E} [ w _ {\tilde {R}} \mid R ^ {*} ] \mathbb {E} [ G \mid R ^ {*} ] + \mathrm{Cov} (w _ {\tilde {R}}, G \mid R ^ {*}),
$$

so the residual bias is governed by the conditional covariance term. This highlights a practical diagnostic: if verifier decisions are strongly predictable from superficial cues that also afect generation likelihood, $\operatorname { C o v } ( w _ { \tilde { R } } , G \mid R ^ { * } )$ may be non-negligible and one should combine channelbased correction with verifier hardening or instance-dependent reliability modeling.

## B.9. Online FN-Rate Estimation via Appeals

In the FN-dominant rule-checker regime, PGFC requires an estimate of the false-negative rate $\rho _ { 1 }$ . At training step t, let

$$
\mathcal {N} _ {\mathrm{R}} ^ {(t)} = \{i: \tilde {r} _ {i} = 0 \}, \qquad \mathcal {P} _ {\mathrm{R}} ^ {(t)} = \{i: \tilde {r} _ {i} = 1 \}
$$

be the rule-based negatives and positives. We appeal each rule-negative item independently with probability $q \in \mathsf { \Gamma } ( 0 , 1 ]$ using a lightweight auxiliary verifier. Let $S ^ { ( t ) } \subseteq \mathcal { N } _ { \mathrm { { R } } } ^ { ( t ) }$ be the appealed subset, and let

$$
\mathcal {P} _ {\mathrm{L}} ^ {(t)} \subseteq S ^ {(t)}
$$

be the appealed items flipped to positive by the auxiliary verifier. The Horvitz–Thompson estimate of the number of false negatives is

$$
\widehat {\mathrm{FN}} ^ {(t)} = \frac {| \mathcal {P} _ {\mathrm{L}} ^ {(t)} |}{q}.
$$

Assuming rule-based positives have negligible FP rate, we estimate

$$
\hat {\rho} _ {1} ^ {(t)} = \frac {\widehat {\mathrm{FN}} ^ {(t)} + \alpha}{\widehat {\mathrm{FN}} ^ {(t)} + | \mathcal {P} _ {\mathrm{R}} ^ {(t)} | + \alpha + \beta},
$$

where $\alpha , \beta > 0$ are small pseudocounts. To track policy drift, we use EMA smoothing:

$$
\bar {\rho} _ {1} ^ {(t)} = \lambda \bar {\rho} _ {1} ^ {(t - 1)} + (1 - \lambda) \hat {\rho} _ {1} ^ {(t)}.
$$

The smoothed value $\bar { \rho } _ { 1 } ^ { ( t ) }$ is used as $\hat { \rho } _ { 1 }$ in PGFC. The expected number of auxiliary verifier calls per update is

$$
q | \mathcal {N} _ {\mathrm{R}} ^ {(t)} |.
$$

For PGBC, both $\rho _ { 0 }$ and $\rho _ { 1 }$ are needed. When $\rho _ { 0 }$ is unavailable or unreliable, we prefer PGFC in the FN-dominant setting. When both rates can be estimated from a calibration set or from a dual-verifier protocol, PGBC can be used with the corrected utility above.

## B.10. Advantage-Level Audit

For every corrected run, we audit the final token-level advantages actually passed to the GRPO surrogate. On the same rollout batch, compute

$$
A ^ {\text { Noise }}, \qquad A ^ {\text { Corr }},
$$

where $A ^ { \mathrm { C o r r } }$ is either $A ^ { \mathrm { B C } }$ or $A ^ { \mathrm { F C } }$ . We report

$$
d _ {\mathrm{mean}} = \frac {1}{N} \sum_ {i, t} \left| A _ {i, t} ^ {\mathrm{Corr}} - A _ {i, t} ^ {\mathrm{Noise}} \right|,
$$

$$
d _ {\mathrm{max}} = \max _ {i, t} \left| A _ {i, t} ^ {\mathrm{Corr}} - A _ {i, t} ^ {\mathrm{Noise}} \right|,
$$

and the cosine similarity

$$
\cos (A ^ {\mathrm{Corr}}, A ^ {\mathrm{Noise}}) = \frac {\langle A ^ {\mathrm{Corr}} , A ^ {\mathrm{Noise}} \rangle}{\| A ^ {\mathrm{Corr}} \| _ {2} \| A ^ {\mathrm{Noise}} \| _ {2}}.
$$

A near-zero $d _ { \mathrm { m e a n } }$ and $d _ { \mathrm { m a x } }$ , together with cosine similarity 1, means the correction is not changing the optimization signal on that batch. As a sanity check, the intentionally degenerate implementation that feeds $u _ { i } = a \tilde { r } _ { i } + b$ into current-group z-score normalization should produce zero diference from the uncorrected noisy baseline up to numerical precision. The reported corrected runs should have nonzero advantage diferences under this audit.

## C. Additional Experimental Results

## C.1. Performance under Pass@8 Metric

As shown in Table 3, we evaluate the mean and standard deviation of pass@8 performance across 16 samples and 5 random seeds under real-world noise. The qualitative conclusions remain highly consistent with the pass@1 results for both Llama-3.2-3B-Instruct and Qwen2.5-Math-7B. Specif ically, relying directly on an LLM as the primary reward source (LLM-as-Judge) systematically underperforms the baseline rule-based pipeline, reinforcing the notion that LLM judges are prone to biases and gaming behaviors. While introducing a simple appeal mechanism to re-check rulebased negatives provides a small but consistent improvement, the primary gradient updates are still influenced by noisy signals. In contrast, our Appeals+PGFC method leverages the appeal stream to dynamically estimate the false-negative rate $\hat { \rho } _ { 1 }$ online, directly applying the forward correction to the policy gradient. This mechanism produces the highest and most robust performance improvements across nearly all evaluated datasets.

## C.2. Complementarity with Stronger Verifiers

We additionally investigate whether the proposed correction mechanisms are synergistic with more capable verifiers. While our main experiments instantiated the appeal process with a lightweight LLM verifier (TinyV) to minimize computational overhead, Table 5 demonstrates that backward/forward corrections yield consistent improvements even when complementing stronger verifier baselines. Specifically, we compare against a “Stronger checker” that extracts the final \boxed{} answer and applies numeric/rational equivalence checking, using the Math-Verify library as a robust baseline. Concretely, we parse the gold answer using a gold-appropriate extraction configuration (ExprExtractionConfig for plain expressions or LatexExtractionConfig for LaTeX), and parse model predictions using LatexExtractionConfig together with ExprExtractionCo Verification is then performed by Math-Verify’s parse-and-verify pipeline with reward-verificationoriented normalization. Under this setup, PGFC uses the standard rule-based reward as the primary signal and calls the auxiliary Math-Verify checker only on appealed rule-negative samples. With the synthetic noise rates fixed at $\rho _ { 0 } = 0 . 1$ and $\rho _ { 1 } = 0 . 2$ as in the main paper, our Appeals+PGFC method achieves notable gains over the strong baseline (e.g., on MATH500 and AMC2023). These results suggest that the theoretical corrections act orthogonally to verifier engineering: regardless of how accurate the underlying verifier is, explicitly modeling and correcting for the residual false-negative rate provides a reliable performance lift.

## C.3. Training Dynamics under Synthetic Noise

To better understand how the proposed corrections influence the learning process, we analyze the training dynamics under synthetic i.i.d. noise. As shown in Figure 5, we monitor the group reward standard deviation, normalized advantage standard deviation, and gradient norm across training steps. The purpose of this analysis is to verify that the backward and forward corrections actively modify the advantage construction during training, rather than merely shifting the final endpoint scores. The results demonstrate that both PGBC and PGFC measurably alter these intermediate training statistics relative to the uncorrected noisy baseline, providing evidence that our corrections dynamically adjust the policy gradient throughout the optimization process.

Table 3: Mean and standard deviation (pass@8) with 16 samples and 5 random seeds under real-world noise. Method names follow Table 1 in the main paper.

<table><tr><td>Dataset</td><td>AIME2024</td><td>AIME2025</td><td>AMC2023</td><td>MATH500</td><td>Minerva MATH</td><td>Olympiad Bench</td><td>Average</td></tr><tr><td colspan="8">Qwen2.5-Math-1.5B</td></tr><tr><td>Base</td><td> $32.4 \pm 0.5$ </td><td> $16.7 \pm 0.7$ </td><td> $79.3 \pm 1.4$ </td><td> $62.8 \pm 1.3$ </td><td> $15.5 \pm 1.1$ </td><td> $31.3 \pm 0.8$ </td><td> $39.7 \pm 1.0$ </td></tr><tr><td>Rule</td><td> $34.3 \pm 1.2$ </td><td> $19.9 \pm 1.1$ </td><td> $80.7 \pm 1.4$ </td><td> $66.9 \pm 0.5$ </td><td> $17.6 \pm 0.6$ </td><td> $32.8 \pm 1.2$ </td><td> $42.0 \pm 1.0$ </td></tr><tr><td>LLM-as-Judge</td><td> $29.6 \pm 1.1$ </td><td> $15.4 \pm 0.4$ </td><td> $80.0 \pm 0.6$ </td><td> $63.4 \pm 1.2$ </td><td> $16.2 \pm 0.8$ </td><td> $28.6 \pm 0.3$ </td><td> $38.9 \pm 0.7$ </td></tr><tr><td>Appeals</td><td> $30.5 \pm 0.2$ </td><td> $20.5 \pm 0.7$ </td><td> $80.7 \pm 0.9$ </td><td> $68.9 \pm 1.3$ </td><td> $17.6 \pm 1.0$ </td><td> $30.5 \pm 0.5$ </td><td> $41.4 \pm 0.8$ </td></tr><tr><td>Appeals+PGFC (Ours)</td><td> $31.0 \pm 0.2$ </td><td> $20.0 \pm 0.6$ </td><td> $82.2 \pm 0.4$ </td><td> $69.8 \pm 1.1$ </td><td> $18.2 \pm 0.5$ </td><td> $33.3 \pm 0.3$ </td><td> $42.4 \pm 0.5$ </td></tr><tr><td colspan="8">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Base</td><td> $28.7 \pm 1.0$ </td><td> $23.3 \pm 0.8$ </td><td> $77.5 \pm 0.2$ </td><td> $67.6 \pm 1.3$ </td><td> $14.0 \pm 0.3$ </td><td> $29.8 \pm 0.4$ </td><td> $40.1 \pm 0.7$ </td></tr><tr><td>Rule</td><td> $32.9 \pm 0.3$ </td><td> $25.7 \pm 1.2$ </td><td> $79.3 \pm 0.7$ </td><td> $71.2 \pm 1.3$ </td><td> $15.5 \pm 0.3$ </td><td> $34.7 \pm 1.4$ </td><td> $43.2 \pm 0.9$ </td></tr><tr><td>LLM-as-Judge</td><td> $29.2 \pm 1.3$ </td><td> $28.3 \pm 0.6$ </td><td> $81.2 \pm 1.2$ </td><td> $68.8 \pm 0.3$ </td><td> $14.7 \pm 1.0$ </td><td> $32.0 \pm 1.4$ </td><td> $42.4 \pm 1.0$ </td></tr><tr><td>Appeals</td><td> $33.8 \pm 1.5$ </td><td> $28.7 \pm 1.1$ </td><td> $80.0 \pm 1.5$ </td><td> $70.8 \pm 1.2$ </td><td> $15.8 \pm 1.3$ </td><td> $34.5 \pm 1.4$ </td><td> $43.9 \pm 1.3$ </td></tr><tr><td>Appeals+PGFC (Ours)</td><td> $36.2 \pm 0.5$ </td><td> $27.5 \pm 1.1$ </td><td> $81.2 \pm 0.4$ </td><td> $71.4 \pm 0.2$ </td><td> $15.8 \pm 1.0$ </td><td> $34.4 \pm 0.4$ </td><td> $44.4 \pm 0.6$ </td></tr><tr><td colspan="8">Llama-3.2-3B-Instruct</td></tr><tr><td>Base</td><td> $18.8 \pm 1.1$ </td><td> $3.3 \pm 0.2$ </td><td> $54.7 \pm 1.4$ </td><td> $40.8 \pm 0.8$ </td><td> $4.8 \pm 1.0$ </td><td> $16.4 \pm 0.5$ </td><td> $23.1 \pm 0.8$ </td></tr><tr><td>Rule</td><td> $22.4 \pm 1.0$ </td><td> $6.7 \pm 1.4$ </td><td> $50.0 \pm 1.3$ </td><td> $44.8 \pm 0.6$ </td><td> $9.2 \pm 0.7$ </td><td> $18.7 \pm 1.5$ </td><td> $25.3 \pm 1.1$ </td></tr><tr><td>LLM-as-Judge</td><td> $10.0 \pm 0.9$ </td><td> $0.8 \pm 0.9$ </td><td> $44.4 \pm 1.4$ </td><td> $27.2 \pm 0.9$ </td><td> $3.7 \pm 1.2$ </td><td> $9.5 \pm 0.2$ </td><td> $15.9 \pm 0.9$ </td></tr><tr><td>Appeals</td><td> $17.5 \pm 0.2$ </td><td> $4.2 \pm 1.0$ </td><td> $53.8 \pm 0.2$ </td><td> $42.3 \pm 1.5$ </td><td> $8.8 \pm 1.3$ </td><td> $14.9 \pm 1.2$ </td><td> $23.6 \pm 0.9$ </td></tr><tr><td>Appeals+PGFC (Ours)</td><td> $20.8 \pm 0.9$ </td><td> $3.3 \pm 1.5$ </td><td> $58.1 \pm 1.3$ </td><td> $47.6 \pm 0.6$ </td><td> $9.6 \pm 1.4$ </td><td> $18.5 \pm 0.6$ </td><td> $26.3 \pm 1.0$ </td></tr><tr><td colspan="8">Qwen2.5-Math-7B</td></tr><tr><td>Base</td><td> $36.2 \pm 1.2$ </td><td> $23.3 \pm 1.0$ </td><td> $81.9 \pm 1.3$ </td><td> $66.6 \pm 0.5$ </td><td> $11.8 \pm 0.6$ </td><td> $33.9 \pm 0.7$ </td><td> $42.3 \pm 0.9$ </td></tr><tr><td>Rule</td><td> $50.0 \pm 0.1$ </td><td> $28.7 \pm 0.8$ </td><td> $83.4 \pm 0.6$ </td><td> $82.8 \pm 0.5$ </td><td> $29.0 \pm 1.1$ </td><td> $45.8 \pm 1.3$ </td><td> $53.3 \pm 0.7$ </td></tr><tr><td>LLM-as-Judge</td><td> $41.7 \pm 0.3$ </td><td> $22.1 \pm 0.3$ </td><td> $81.9 \pm 1.5$ </td><td> $69.0 \pm 0.2$ </td><td> $13.2 \pm 1.2$ </td><td> $39.4 \pm 1.1$ </td><td> $44.6 \pm 0.8$ </td></tr><tr><td>Appeals</td><td> $47.1 \pm 1.0$ </td><td> $30.4 \pm 0.3$ </td><td> $84.4 \pm 1.2$ </td><td> $80.8 \pm 0.3$ </td><td> $23.5 \pm 1.3$ </td><td> $45.6 \pm 0.5$ </td><td> $52.0 \pm 0.8$ </td></tr><tr><td>Appeals+PGFC (Ours)</td><td> $54.6 \pm 1.4$ </td><td> $30.4 \pm 1.2$ </td><td> $82.8 \pm 0.2$ </td><td> $83.2 \pm 1.3$ </td><td> $29.0 \pm 0.6$ </td><td> $47.6 \pm 0.9$ </td><td> $54.6 \pm 0.9$ </td></tr></table>

Figure 5: Training dynamics under synthetic iid noise. PGBC and PGFC change the group reward standard deviation, normalized advantage standard deviation, and gradient norm relative to uncorrected noisy training, illustrating that the correction is active inside advantage construction rather than only changing endpoint scores.

## C.4. Appeals Statistics and Computational Overhead

To quantify the computational cost and the empirical behavior of the appeal mechanism under real-world verifier noise, we track the appeal and flip statistics during training on Qwen2.5-Math-1.5B. As detailed in Table 4, we measure the appeal rate (the fraction of rollout responses that trigger an appeal) and the flip rate (the fraction of appealed responses that are successfully overturned by the LLM judge). Notably, our Appeals+PGFC method leverages the exact same appeal stream used by the standard Appeals baseline. Consequently, Appeals+PGFC achieves significant gains in both pass@1 and pass@8 without requiring additional verifier queries or incurring extra computational overhead. The total number of auxiliary TinyV calls remains small relative to the total number of training rollouts, demonstrating that our online false-negative rate estimation and forward correction are highly computationally eficient.

## C.5. General Capability Retention

We also check whether the corrections preserve non-math general benchmark performance relative to noisy training. Table 6 reports that PGBC and PGFC recover most of the retention loss caused by uncorrected noisy rewards and remain close to Base/Oracle on these auxiliary evaluations.

Table 4: Appeals statistics for Qwen2.5-Math-1.5B real-world verifier noise. Appeals+PGFC uses the same appeal stream both for recovery and for online FN-rate estimation.

<table><tr><td>Method</td><td>Avg pass@1</td><td>Avg pass@8</td><td>Appeal rate</td><td>Flip rate</td><td>TinyV calls</td></tr><tr><td>Rule</td><td>31.6±0.7</td><td>42.0±0.8</td><td>0.000</td><td>0.0000</td><td>0</td></tr><tr><td>Appeals</td><td>30.1±0.6</td><td>41.4±0.7</td><td>0.081</td><td>0.2288</td><td>7,963</td></tr><tr><td>Appeals+PGFC</td><td>33.7±0.6</td><td>42.4±0.8</td><td>0.093</td><td>0.2745</td><td>8,255</td></tr></table>

Table 5: Complementarity with a stronger verifier-side baseline on Qwen2.5-Math-1.5B. The correction still improves over a stronger checker, indicating that PGFC is not merely replacing verifier engineering.

<table><tr><td>Method</td><td>MATH500</td><td>AIME2024</td><td>AIME2025</td><td>AMC2023</td><td>Minerva Math</td><td>OlympiadBench</td><td>Average</td></tr><tr><td>Stronger checker</td><td>68.1 / 69.2</td><td>12.3 / 18.4</td><td>7.2 / 12.3</td><td>47.9 / 59.7</td><td>17.8 / 18.6</td><td>30.9 / 33.1</td><td>30.7 / 35.2</td></tr><tr><td>Appeals+PGFC</td><td>72.0 / 75.1</td><td>14.2 / 21.6</td><td>8.8 / 15.8</td><td>51.8 / 65.8</td><td>20.9 / 23.8</td><td>33.3 / 37.2</td><td>33.5 / 39.8</td></tr></table>

## D. Proofs and Derivations

## D.1. Proof of Proposition 3.2

Proof. We compute the expectation of the noisy reward $\tilde { R }$ conditioned on the clean reward $R ^ { * }$ which is a binary variable. By the definition of expectation:

$$
\begin{array}{c} \mathbb {E} [ \tilde {R} ] = 1 \cdot \mathbb {P} (\tilde {R} = 1 \mid R ^ {*}) + 0 \cdot \mathbb {P} (\tilde {R} = 0 \mid R ^ {*}) \\ = \mathbb {P} (\tilde {R} = 1 \mid R ^ {*}). \end{array}
$$

We can expand this using the law of total probability, conditioning on the value of $R ^ { * } \in \{ 0 , 1 \}$ :

$$
\mathbb {E} [ \tilde {R} \mid R ^ {*} ] = R ^ {*} \cdot \mathbb {P} (\tilde {R} = 1 \mid R ^ {*} = 1) + (1 - R ^ {*}) \cdot \mathbb {P} (\tilde {R} = 1 \mid R ^ {*} = 0).
$$

From Definition 3.1, we have $\mathbb { P } ( \tilde { R } = 1 \mid R ^ { * } = 0 ) = \rho _ { 0 }$ and $\mathbb { P } ( \tilde { R } = 0 \mid R ^ { * } = 1 ) = \rho _ { 1 }$ , which implies $\mathbb P ( \tilde { R } = 1 \mid R ^ { * } = 1 ) = 1 - \rho _ { 1 }$ . Substituting these values:

$$
\begin{array}{r l} & {\mathbb {E} [ \tilde {R} \mid R ^ {*} ] = R ^ {*} (1 - \rho_ {1}) + (1 - R ^ {*}) \rho_ {0}} \\ & {\qquad = R ^ {*} - \rho_ {1} R ^ {*} + \rho_ {0} - \rho_ {0} R ^ {*}} \\ & {\qquad = (1 - \rho_ {0} - \rho_ {1}) R ^ {*} + \rho_ {0}.} \end{array}
$$

This completes the proof.

Table 6: General capability retention after RLVR training on Llama-3.2-3B-Instruct. Metrics are accuracies.

<table><tr><td>Method</td><td>MMLU</td><td>ARC-Challenge</td><td>HellaSwag</td><td>GPQA</td></tr><tr><td>Base</td><td>60.7</td><td>73.9</td><td>55.0</td><td>28.4</td></tr><tr><td>Oracle</td><td>61.1</td><td>73.9</td><td>55.9</td><td>29.1</td></tr><tr><td>Noise</td><td>50.0</td><td>43.9</td><td>33.9</td><td>19.4</td></tr><tr><td>PGBC</td><td>60.8</td><td>73.8</td><td>55.6</td><td>28.8</td></tr><tr><td>PGFC</td><td>61.3</td><td>74.2</td><td>56.1</td><td>29.2</td></tr></table>

## D.2. Proof of Theorem 3.3

Proof. Recall the verifier reward channel (Definition 3.1): for the latent clean reward $R ^ { * } \in \{ 0 , 1 \}$ and the observed noisy reward $\tilde { R } \in \{ 0 , 1 \}$ ,

$$
\operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*} = 0) = \rho_ {0}, \quad \operatorname * {P r} (\tilde {R} = 0 \mid R ^ {*} = 1) = \rho_ {1},
$$

hence $\operatorname* { P r } ( \tilde { R } = 1 \mid R ^ { * } = 1 ) = 1 - \rho _ { 1 }$ and $\operatorname* { P r } ( { \tilde { R } } = 0 \mid R ^ { * } = 0 ) = 1 - \rho _ { 0 }$ . Since $1 - \rho _ { 0 } - \rho _ { 1 } > 0$ , the estimator

$$
\widehat {R} \triangleq \frac {\tilde {R} - \rho_ {0}}{1 - \rho_ {0} - \rho_ {1}}
$$

is well-defined.

We prove unbiasedness by computing the conditional expectation given $R ^ { * }$ . First note that because $\tilde { R }$ is binary,

$$
\mathbb {E} [ \tilde {R} \mid R ^ {*} ] = 1 \cdot \operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*}) + 0 \cdot \operatorname * {P r} (\tilde {R} = 0 \mid R ^ {*}) = \operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*}).\tag{7}
$$

By the channel definition, for the two possible values of $R ^ { * }$ we have

$$
\mathbb {E} [ \tilde {R} \mid R ^ {*} = 1 ] = 1 - \rho_ {1}, \qquad \mathbb {E} [ \tilde {R} \mid R ^ {*} = 0 ] = \rho_ {0}.
$$

These two cases can be written compactly as the afine form (Proposition 3.2)

$$
\mathbb {E} [ \tilde {R} \mid R ^ {*} ] = (1 - \rho_ {0} - \rho_ {1}) R ^ {*} + \rho_ {0}.\tag{8}
$$

Now take the conditional expectation of $\widehat { R }$ :

$$
\begin{array}{l} \mathbb {E} [ \widehat {R} \mid R ^ {*} ] = \mathbb {E} \left[ \frac {\tilde {R} - \rho_ {0}}{1 - \rho_ {0} - \rho_ {1}} \Bigg | R ^ {*} \right] \\ = \frac {\mathbb {E} [ \tilde {R} \mid R ^ {*} ] - \rho_ {0}}{1 - \rho_ {0} - \rho_ {1}} \\ = \frac {(1 - \rho_ {0} - \rho_ {1}) R ^ {*} + \rho_ {0} - \rho_ {0}}{1 - \rho_ {0} - \rho_ {1}} \quad (\text {by (8)}) \\ = R ^ {*}. \end{array}\tag{9}
$$

Equation (9) shows that $\widehat { R }$ is an unbiased estimator of the latent clean reward in the conditional sense.

Finally, taking total expectation (law of iterated expectation) yields

$$
\mathbb {E} [ \widehat {R} ] = \mathbb {E} \big [ \mathbb {E} [ \widehat {R} | R ^ {*} ] \big ] = \mathbb {E} [ R ^ {*} ],
$$

so $\widehat { R }$ is unbiased for the clean reward both conditionally and unconditionally.

## D.3. Proof of Proposition 3.4

Proof. The proposition states two claims about the conditional expectation of the forward weights. The weights are defined as:

$$
w _ {\tilde {R}} = \left\{ \begin{array}{l l} w _ {0} = \rho_ {1} - 1 & \text {if} \tilde {R} = 0, \\ w _ {1} = \rho_ {1} & \text {if} \tilde {R} = 1. \end{array} \right.
$$

The noise model provides the conditional probabilities:

$$
\operatorname * {P r} (\tilde {R} = 0 \mid R ^ {*} = 1) = \rho_ {1}, \qquad \operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*} = 1) = 1 - \rho_ {1}
$$

$$
\operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*} = 0) = \rho_ {0}, \qquad \operatorname * {P r} (\tilde {R} = 0 \mid R ^ {*} = 0) = 1 - \rho_ {0}
$$

Part 1: Positive-reward identity. We compute the expectation of $w _ { \tilde { R } }$ conditioned on the true reward being positive $( R ^ { * } = 1 )$ ):

$$
\begin{array}{l} \mathbb {E} [ w _ {\tilde {R}} \mid R ^ {*} = 1 ] = \sum_ {k \in \{0, 1 \}} w _ {k} \cdot \operatorname * {P r} (\tilde {R} = k \mid R ^ {*} = 1) \\ \qquad = w _ {0} \cdot \operatorname * {P r} (\tilde {R} = 0 \mid R ^ {*} = 1) + w _ {1} \cdot \operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*} = 1) \\ \qquad = (\rho_ {1} - 1) \cdot (\rho_ {1}) + (\rho_ {1}) \cdot (1 - \rho_ {1}) \\ \qquad = (\rho_ {1} ^ {2} - \rho_ {1}) + (\rho_ {1} - \rho_ {1} ^ {2}) \\ \qquad = 0. \end{array}
$$

Part 2: Negative-reward identity. Next, we compute the expectation of $w _ { \tilde { R } }$ conditioned on the true reward being negative $( R ^ { * } = 0 )$ :

$$
\begin{array}{l} \mathbb {E} [ w _ {\tilde {R}} \mid R ^ {*} = 0 ] = \sum_ {k \in \{0, 1 \}} w _ {k} \cdot \operatorname * {P r} (\tilde {R} = k \mid R ^ {*} = 0) \\ \qquad = w _ {0} \cdot \operatorname * {P r} (\tilde {R} = 0 \mid R ^ {*} = 0) + w _ {1} \cdot \operatorname * {P r} (\tilde {R} = 1 \mid R ^ {*} = 0) \\ \qquad = (\rho_ {1} - 1) \cdot (1 - \rho_ {0}) + (\rho_ {1}) \cdot (\rho_ {0}) \\ \qquad = (\rho_ {1} - \rho_ {0} \rho_ {1} - 1 + \rho_ {0}) + \rho_ {0} \rho_ {1} \\ \qquad = \rho_ {1} + \rho_ {0} - 1 \\ \qquad = - (1 - \rho_ {0} - \rho_ {1}). \end{array}
$$

This proves both claims of the proposition.

## $D . 4 .$ Proof of Theorem 3.5

Proof. We want to show that $\mathbb { E } [ \Delta \theta ] = ( 1 - \rho _ { 0 } - \rho _ { 1 } ) \nabla _ { \theta } J ( \theta )$ , where $\begin{array} { r } { \Delta \theta = \frac { 1 } { M } \sum _ { t = 1 } ^ { M } h _ { t } } \end{array}$ and $h _ { t } =$ $w _ { \tilde { R } } G _ { t }$ . By linearity of expectation and assuming i.i.d. samples, it sufices to show this for a single sample’s contribution, $\mathbb { E } [ h _ { t } ]$

We use the law of total expectation, conditioning on the latent true reward $R ^ { * } \in \{ 0 , 1 \}$ :

$$
\begin{array}{r l} & {\mathbb {E} [ h _ {t} ] = \mathbb {E} [ w _ {\tilde {R}} G _ {t} ] = \mathbb {E} \left[ \mathbb {E} [ w _ {\tilde {R}} G _ {t} \mid R ^ {*} ] \right]} \\ & {\qquad = \operatorname * {P r} (R ^ {*} = 1)   \mathbb {E} [ w _ {\tilde {R}} G _ {t} \mid R ^ {*} = 1 ] + \operatorname * {P r} (R ^ {*} = 0)   \mathbb {E} [ w _ {\tilde {R}} G _ {t} \mid R ^ {*} = 0 ].} \end{array}
$$

The noise process generating $\tilde { R }$ is independent of the policy’s action generation process (which produces $G _ { t } )$ , conditional on the true reward $R ^ { * }$ . Thus, we can separate the expectations:

$$
\mathbb {E} [ w _ {\tilde {R}} G _ {t} \mid R ^ {*} ] = \mathbb {E} [ w _ {\tilde {R}} \mid R ^ {*} ] \cdot \mathbb {E} [ G _ {t} \mid R ^ {*} ].
$$

Using the results from Proposition 3.4:

$\mathbb { E } [ w _ { \tilde { R } } \mid R ^ { * } = 1 ] = 0 .$

$\mathbb { E } [ w _ { \tilde { R } } \mid R ^ { * } = 0 ] = - ( 1 - \rho _ { 0 } - \rho _ { 1 } ) .$

Substituting these back into the main expectation formula:

$$
\begin{array}{l} \mathbb {E} [ w _ {\bar {R}} G _ {t} ] = \operatorname * {P r} (R ^ {*} = 1) \cdot (0) \cdot \mathbb {E} [ G _ {t} \mid R ^ {*} = 1 ] \\ \qquad + \operatorname * {P r} (R ^ {*} = 0) \cdot (- (1 - \rho_ {0} - \rho_ {1})) \cdot \mathbb {E} [ G _ {t} \mid R ^ {*} = 0 ] \\ \qquad = - (1 - \rho_ {0} - \rho_ {1}) \cdot \operatorname * {P r} (R ^ {*} = 0) \mathbb {E} [ G _ {t} \mid R ^ {*} = 0 ] \\ \qquad = - (1 - \rho_ {0} - \rho_ {1}) \cdot \mathbb {E} [ \mathbf {1} _ {\{R ^ {*} = 0 \}} G _ {t} ], \end{array}
$$

where $\mathbf { 1 } _ { \{ \cdot \} }$ is the indicator function. From two fundamental properties of the score function:

1. The unconditional expectation is zero: $\mathbb { E } [ G _ { t } ] = 0 \ [ 3 7 , 3 1 ]$

2. The clean policy gradient is $\nabla _ { \theta } J ( \theta ) = \mathbb { E } [ R ^ { * } G _ { t } ]$

From property 1, we have $\mathbb { E } [ G _ { t } ] = \mathbb { E } [ ( \mathbf { 1 } _ { \{ R ^ { * } = 1 \} } + \mathbf { 1 } _ { \{ R ^ { * } = 0 \} } ) G _ { t } ] = \mathbb { E } [ R ^ { * } G _ { t } ] + \mathbb { E } [ \mathbf { 1 } _ { \{ R ^ { * } = 0 \} } G _ { t } ] = 0$ . This implies that $\mathbb { E } [ \mathbf { 1 } _ { \{ R ^ { * } = 0 \} } G _ { t } ] = - \mathbb { E } [ R ^ { * } G _ { t } ] = - \nabla _ { \theta } J ( \theta )$

Finally, we substitute this back into our expression for the expected update direction:

$$
\begin{array}{l} \mathbb {E} [ h _ {t} ] = \mathbb {E} [ w _ {\tilde {R}} G _ {t} ] \\ \qquad = - (1 - \rho_ {0} - \rho_ {1}) \cdot \mathbb {E} [ \mathbf {1} _ {\{R ^ {*} = 0 \}} G _ {t} ] \\ \qquad = - (1 - \rho_ {0} - \rho_ {1}) \cdot (- \nabla_ {\theta} J (\theta)) \\ \qquad = (1 - \rho_ {0} - \rho_ {1}) \nabla_ {\theta} J (\theta). \end{array}
$$

Therefore, the expectation of the full update is $\begin{array} { r } { \mathbb { E } [ \Delta \theta ] = \frac { 1 } { M } \sum \mathbb { E } [ h _ { t } ] = ( 1 - \rho _ { 0 } - \rho _ { 1 } ) \nabla _ { \theta } J ( \theta ) } \end{array}$ . This completes the proof. □

## E. Prompt Templates and Training/Evaluation Details

This section records the exact prompt formats and the concrete hyperparameters we used for all experiments in this paper. We mirror the level of detail used in recent RLVR appendices and report settings suficient for full reproducibility from our released code.

## E.1. Prompt Templates

Training (generation) prompt.. For each math problem x (a plain-text question), the user message is built by concatenating the raw question with a short instruction that elicits chain-of-thought and enforces a verifiable answer format.

```txt
<user>
{QUESTION}
Let's think step by step and enclose the reasoning process within <think> and </think > tags.
The final result in the answer MUST BE within \boxed{}.
</user>
```

During data preprocessing, we write chat-style JSON with a single user turn as shown above and attach the rule-based ground-truth answer for reward checking.

Evaluation (validation/test) prompt.. We use the same prompt template as training for validation and test-time generation so that the rule-based verifier can parse the boxed answer consistently.

Verifier $I / O .$ . The rule-based checker operates on the model’s final string and extracts the last \boxed{...} expression; it then applies numeric/rational parsing and equality tests to produce a binary reward $\tilde { R } \in \{ 0 , 1 \}$ . When the LLM verifier is enabled, it receives the pair (problem, model solution) and returns a binary correctness decision used only to estimate the false negative rate $\rho _ { 1 }$ over a sliding window. The LLM verifier does not replace the rule-based reward.

## E.2. Data Preprocessing

We load the preview split of the math-reasoning corpus and map each example to a chat-style record as above, keeping the reference (ground-truth) answer for programmatic checking.

## E.3. Training Configuration

Unless otherwise stated, all runs use GRPO (outcome supervision) with the following constants.

<table><tr><td colspan="2">Training (GRPO)</td></tr><tr><td>Train batch size</td><td>128</td></tr><tr><td>Rollouts per question (group size)</td><td>8</td></tr><tr><td>Max prompt length (tokens)</td><td>512</td></tr><tr><td>Max response length (tokens)</td><td>3072</td></tr><tr><td>Sampling temperature (rollouts)</td><td>1.0</td></tr><tr><td>Advantage estimator</td><td>Group-normalized (GRPO)</td></tr><tr><td>KL regularization</td><td>Enabled</td></tr><tr><td>KL coefficient β</td><td>0.001</td></tr><tr><td>Entropy coefficient</td><td>0.0</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>3e-6</td></tr><tr><td>Total epochs</td><td>1</td></tr></table>

Table 7: Core training settings.

Model/backbone.. We load the base model from local cache (HuggingFace layout), enable FSDP2 for actor/ref, and use shared-memory weights with remove-padding for eficient vLLM rollouts. The KL is computed w.r.t. a frozen reference initialized from the same base.

## E.4. Online FN-Rate Estimation Hyperparameters and Ablations

This subsection reports the concrete hyperparameters for our online false-negative (FN) rate estimation $\hat { \rho } _ { 1 }$ used in the real-world-noise experiments (Section 4.3), and provides a systematic ablation over the key knobs that control the cost–accuracy trade-of of the appeals mechanism.

Appeals protocol and estimator.. At training step t, let $\mathcal { N } _ { \mathrm { R } } ^ { ( t ) }$ and $\mathcal { P } _ { \mathrm { R } } ^ { ( t ) }$ be the sets labeled negative/positive by the primary rule-based verifier. We appeal each item in $\mathcal { N } _ { \mathrm { R } } ^ { ( t ) }$ independently with probability $q \in ( 0 , 1 ]$ and re-evaluate it with a lightweight LLM verifier (TinyV). Let $\mathcal { P } _ { \mathrm { L } } ^ { ( t ) }$ be the appealed negatives that TinyV flips to positive. We use the Horvitz–Thompson estimator

$$
\widehat {\mathrm{FN}} ^ {(t)} = \frac {| \mathcal {P} _ {\mathrm{L}} ^ {(t)} |}{q}\tag{10}
$$

and estimate the FN rate with a Beta prior:

$$
\hat {\rho} _ {1} ^ {(t)} = \frac {\widehat {\mathrm{FN}} ^ {(t)} + \alpha}{\widehat {\mathrm{FN}} ^ {(t)} + | \mathcal {P} _ {\mathrm{R}} ^ {(t)} | + \alpha + \beta}.\tag{11}
$$

We then apply exponential moving average (EMA) smoothing

$$
\bar {\rho} _ {1} ^ {(t)} = \lambda \bar {\rho} _ {1} ^ {(t - 1)} + (1 - \lambda) \hat {\rho} _ {1} ^ {(t)},\tag{12}
$$

and use $\bar { \rho } _ { 1 } ^ { ( t ) }$ as the plug-in estimate for PGFC weights $w _ { \tilde { r } } \in \{ \bar { \rho } _ { 1 } ^ { ( t ) } - 1 , \ \bar { \rho } _ { 1 } ^ { ( t ) } \}$

Default hyperparameters.. Unless otherwise stated, real-world-noise runs use: (i) appeal probability q = 0.1; (ii) Beta prior pseudocounts $\alpha = \beta = 1 0 ^ { - 5 }$ in Eq. (11); (iii) EMA decay $\lambda = 0 . 9$ in Eq. (12). We compute $\hat { \rho } _ { 1 } ^ { ( t ) }$ and update $\bar { \rho } _ { 1 } ^ { ( t ) }$ once per policy-update step. The expected number of TinyV calls per update is $q \cdot | \mathcal { N } _ { \mathrm { R } } ^ { ( t ) } |$ (reported below as a cost proxy).

## E.5. Datasets

Training..

• DeepScaleR [19]: the math-reasoning corpus used for RLVR rollouts and policy updates.

Evaluation.. We evaluate with the standard oficial/problem releases for each benchmark and apply the same rule-based programmatic checking protocol throughout.

• MATH500 [18]: a verifiable subset of math reasoning problems for reliable evaluation.

• AIME 2024 [11]: the 2024 American Invitational Mathematics Examination set.

```txt
False Positive (FP).
Problem. Compute: 1 - 2 + 3 - 4 + ··· + 99 - 100.
Ground truth. -50
Model response (excerpt).
Compute: $1+2-3-4+5+6-7-8+\dots+97+98-99-100$.
\boxed{} \boxed{} \boxed{} \dots
```

• AIME 2025 [23]: the 2025 AIME set.

• AMC 2023 [20]: problems from the 2023 American Mathematics Competitions.

• Minerva Math [14]: the Minerva math evaluation suite.

• OlympiadBench [7]: olympiad-level mathematics benchmark.

• MMLU [8]: a broad multi-task benchmark covering 57 academic and professional subjects for evaluating world knowledge and problem solving.

• ARC-C [2]: the Challenge split of the AI2 Reasoning Challenge, consisting of grade-school science questions that are dificult for retrieval and word co-occurrence baselines.

• HellaSwag [40]: a commonsense natural language inference benchmark for physically grounded sentence-completion reasoning.

• GPQA [26]: a graduate-level, Google-proof multiple-choice QA benchmark written by domain experts in biology, physics, and chemistry.

## F. False Positive (FP) and False Negative (FN) Examples

We also provide typical failure modes observed during math RL with automated verifiers on Qwen2.5-Math-7B under MATH500 dataset. For rule-based verifier: 52 FNs and 0 FP out of 500 items (marginal rates 10.4% and 0%); for LLM verifier: 81 FNs and 168 FPs out of 500 items (marginal rates 16.2% and 33.6%). The FP case shows an LLM judge over-crediting a malformed solution, while the FN case shows a rule-based checker rejecting a numerically correct answer due to formatting. The raw JSON context for these examples is provided below.

```txt
Verifier signals. Rule-based checker: ✗ (incorrect); LLM judge: √ (incorrectly accepts) ⇒ FP.
```

## False Negative (FN).

Problem. Each of seven islands independently has treasure with probability <sup>1</sup>. What is the

probability that exactly four islands have treasure?

Correct answer. $\left\lfloor { \frac { 4 4 8 } { 1 5 6 2 5 } } \right\rceil$ (equals 0.028672).

Model answer. 0.028672

Verifier signals. Rule-based exact-match: ✗ (expects a fraction); numeric equivalence check:

We provide FP and FN examples using Qwen2.5-Math-7B base model on MATH500.

```json
{
    "idx": 16,
    "question": "Compute: $1-2+3-4+5-\dots +99-100$.",
    "model_prediction": "{}",
    "model_response": "Compute: $1+2-3-4+5+6-7-8+\dots +97+98-99-100$. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\. \boxed {}\.
    {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},
```

```txt
"llm_verifier_score": 1.0,
"llm_verifier_raw_value": "1.0",
"math_verify": false,
"math_verify_score": 0.0
},
```

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
"idx": 92,
"question": "A pirate searches seven islands for buried treasure. If each island has a $\frac{1}{5}$ chance of having treasure, what is the probability that exactly 4 of the islands have treasure?",
"answer": "\frac{448}{15625}",
"model_prediction": "0.028672",
"model_response": "To solve this problem, we can use the binomial probability formula. The probability of having exactly $k$ successes in $n$ independent Bernoulli trials (each with success probability $p$) is given by:\n\n\[ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} \\]\n\nHere, $n = 7$ (the number of islands), $k = 4$ (the number of islands with treasure), and $p = \frac{1}{5}$ (the probability that a single island has treasure).\n\nThe binomial coefficient $\binom{n}{k}$ is the number of ways to choose $k$ successes out of $n$ trials, and it is calculated as:\n\n[\ \binom{n}{k} = \frac{n!}{k!(n-k)!} \\]\nSo, the probability that exactly 4 of the 7 islands have treasure is:\n\n\[ P(X = 4) = \binom{7}{4} \left(\frac{1}{5}\right)^4 \left(\frac{4}{5}\right)^{3} \\]\n\nLet's calculate this using Python.\n''python\nimport math\n# Number of islands\n = 7\n# Number of islands with treasure\nk = 4\n# Probability of a single island having treasure\np = 1/5\n\n# Binomial coefficient\nbinom_coeff = math.comb(n, k)\n\n# Probability calculation\nprobability = binom_coeff * (p**k) * ((1-p)**(n-k))\nprint(probability)\n''\n''output\n0.02867200000000001\n''\nThe probability that exactly 4 of the 7 islands have treasure is $\boxed{0.028672}$".",
"ground_truth": "\frac{448}{15625}",
"rule_based": false,
"llm_verifier": false,
"llm_verifier_score": 0.0,
"llm_verifier_raw_value": "0.0",
</div>

"math\_verify": true,

"math\_verify\_score": 1.0
