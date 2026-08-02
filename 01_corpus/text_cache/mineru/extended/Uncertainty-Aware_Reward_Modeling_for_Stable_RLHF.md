# Uncertainty-Aware Reward Modeling for Stable RLHF

Licheng Pan <sup>1</sup> <sup>2</sup> Haocheng Yang <sup>3</sup> Haoxuan Li <sup>4</sup> Yichen Sun <sup>1</sup> Yunsheng Lu <sup>1</sup> Shijian Wang <sup>2</sup> Lei Shen <sup>2</sup> Yuan Lu <sup>2</sup> Zhixuan Chu <sup>1</sup> Hao Wang <sup>1</sup> <sup>2</sup>

## Abstract

Reinforcement learning from human feedback (RLHF) aligns large language models by training reward models on preference data and optimizing policies to maximize predicted rewards. However, this pipeline faces two fundamental challenges: ❶ reward models cannot signal when their predictions are unreliable, since they usually act as deterministic point estimators; and ❷ modern group-based policy optimization can amplify unreliable reward signals, as exemplified by GRPO’s uniform treatment of rewards during advantage computation. As policies explore increasingly diverse responses, these two limitations create a critical vulnerability: unreliable reward estimates may be granted disproportionate influence, triggering severe reward hacking. We propose Uncertainty-Aware Reward Modeling (UARM), which equips reward models with calibrated uncertainty via quantile-based conformal prediction and reweights GRPO advantages through heteroscedastic variance decomposition. Experiments across HelpSteer, Ultra-Feedback, and PKU-SafeRLHF demonstrate that UARM significantly improves reward model calibration, reduces reward hacking, and enhances downstream alignment quality compared to standard GRPO and uncertainty-agnostic baselines.

## 1. Introduction

Reinforcement learning from human feedback (Christiano et al., 2017; Ouyang et al., 2022) has emerged as the dominant paradigm for aligning large language models with human values and preferences. In this framework, a reward model is first trained on pairwise preference data—typically modeled via the Bradley-Terry comparison model (Bradley & Terry, 1952)—to proxy human judgment, and the policy is then optimized to maximize the predicted reward (Dong et al., 2024). Recent state-of-the-art systems, from GPT-4 (Achiam et al., 2023) to DeepSeek-R1 (Guo et al., 2025) and Gemini (Comanici et al., 2025), rely heavily on this pipeline to produce helpful, harmless, and honest responses. Yet a fundamental question remains unresolved: when reward models are uncertain about their predictions, should policies trust them unconditionally?

This question reveals two central challenges that become especially consequential in modern group-based policy optimization methods such as Group Relative Policy Optimization (GRPO) (Guo et al., 2025; Zheng et al., 2025). ❶ Reward models cannot signal when their predictions are unreliable. Current reward models are deterministic point estimators: they output a single scalar score for each prompt-response pair, with no indication of whether that score reflects a confident judgment or an unreliable guess (Lambert et al., 2025). As policies evolve during training, they inevitably generate responses that lie outside the reward model’s training distribution—responses the model evaluates with high uncertainty. Without any signal of this uncertainty, the policy treats all reward estimates as equally trustworthy, even when some are fundamentally unreliable. This blind trust creates a vulnerability: the policy may aggressively optimize toward responses the reward model scores highly but uncertainly, leading to misalignment and reward hacking (Amodei et al., 2016; Gao et al., 2023; Skalse et al., 2022).

❷ Group-based advantage standardization can amplify the least trustworthy samples. In GRPO, advantages are standardized uniformly within each rollout group. By treating all reward signals as equally reliable, this standardization procedure can amplify exactly the samples the reward model is least certain about: a confusing response that receives a spuriously extreme reward skews the group mean and variance, and after standardization is assigned a disproportionately large advantage. Meanwhile, genuinely highquality responses are pushed below the distorted mean and under-rewarded. This uniform treatment thus triggers severe reward hacking, steering policies toward unreliable signals and away from truly aligned behavior (Fu et al., 2025; Miao et al., 2024; Liu et al., 2024).

Prior work has explored uncertainty quantification for neural networks, but their application to stable RLHF remains limited. Ensemble-based methods (Lakshminarayanan et al., 2017; Gal & Ghahramani, 2016) provide uncertainty estimates by training multiple models or performing stochastic forward passes, but incur prohibitive computational overhead for large language models deployed in online RLHF (Coste et al., 2023; Eisenstein et al., 2023). Conformal prediction (Lei et al., 2018b; Romano et al., 2019; Le et al., 2018a; Shafer & Vovk, 2008) offers distribution-free coverage guarantees for uncertainty intervals, but classical variants focus on marginal coverage and are not readily adapted to the conditional, sample-specific reliability signals needed for reweighting advantages in reinforcement learning (Tibshirani et al., 2019; Gibbs & Candes, 2021). Heteroscedastic modeling techniques (Immer et al., 2023; Amini et al., 2020) can capture varying uncertainty across inputs, yet have not been integrated into RLHF’s advantage computation. Consequently, no prior work systematically addresses the gap between calibrated uncertainty quantification in reward models and its direct integration into policy optimization to prevent reward hacking.

We propose Uncertainty-Aware Reward Modeling (UARM), a unified framework that equips reward models with calibrated uncertainty and leverages it to stabilize GRPO. Our approach operates in two phases. In the offline phase, we train the reward model as a quantile regression estimator (Steinwart & Christmann, 2011) that outputs multiple conditional quantiles of the reward distribution, from which we derive both a point estimate (the median) and a prediction interval whose width captures per-sample uncertainty. We calibrate these intervals on a held-out set via a conformal prediction procedure (Romano et al., 2019), achieving conditional coverage guarantees (Theorem 3.2) that ensure the interval width faithfully reflects the model’s confidence. In the online phase, we reinterpret the interval width as observation noise under a heteroscedastic model and decompose the observed reward variance into signal and noise components. This variance decomposition yields a sample-specific reliability weight for each rollout, which we use to construct a heteroscedastic advantage that provably down-weights high-uncertainty samples without requiring costly ensemble evaluations. The entire pipeline integrates seamlessly into GRPO with negligible computational overhead.

Our contributions are summarized as follows:

• We develop a quantile-based conformal reward model that provides calibrated per-sample uncertainty estimates with theoretical coverage guarantees.

• We introduce a heteroscedastic advantage reweighting scheme that uses uncertainty to suppress unreliable samples in GRPO’s standardization.

• We conduct comprehensive experiments across three preference datasets, demonstrating that UARM improves reward model calibration, reduces reward hacking, and enhances downstream alignment quality against strong baselines.

## 2. Preliminaries

## 2.1. Reinforcement Learning from Human Feedback

The standard RLHF pipeline typically consists of two sequential stages (Ouyang et al., 2022): reward modeling and policy optimization. First, a reward model (RM) $r _ { \theta } : \mathcal { X } $ R parameterized by θ is trained on an offline dataset comprising human preferences. With point-wise or pair-wise optimization (Wang et al., 2026), it learns to map a prompt-response pair $x = ( p , o ) \in \mathcal { X }$ (where $p$ is the prompt and o is the generated response) to a scalar reward $r \in \mathbb { R }$ . Subsequently, the policy model $\pi _ { \phi }$ (i.e., the LLM) is optimized via reinforcement learning to maximize the expected rewards assigned by the learned RM.

Recently, value-function-free algorithms, particularly Group Relative Policy Optimization (GRPO) (Guo et al., 2025), have emerged as the mainstream paradigm for policy optimization due to their efficiency. For a given prompt $q ,$ GRPO samples a group of $\mathrm { N _ { r o l } }$ responses $\{ o _ { i } \} _ { i = 1 } ^ { \mathbf { \hat { N } _ { r o l } } }$ from the old policy $\pi _ { \phi _ { \mathrm { o l d } } } .$ The RM evaluates these promptresponse pairs $\{ x _ { i } \} _ { i = 1 } ^ { \mathrm { N _ { r o l } } }$ to obtain raw terminal rewards $r _ { i } = r _ { \theta } ( x _ { i } )$ . To stabilize training without the computational burden of maintaining a critic model, GRPO computes the advantage $A _ { i } = ( r _ { i } - \mu ) / \sigma$ by standardizing the rewards within the sampled group, where $\begin{array} { r } { \mu = \frac { 1 } { \mathrm { N } _ { \mathrm { r o l } } } \sum _ { i = 1 } ^ { \mathrm { N } _ { \mathrm { r o l } } } r _ { i } } \end{array}$ and $\begin{array} { r } { \sigma ^ { 2 } = \frac { 1 } { \mathrm { N } _ { \mathrm { r o l } } } \sum _ { i = 1 } ^ { \mathrm { N } _ { \mathrm { r o l } } } ( r _ { i } - \mu ) ^ { 2 } } \end{array}$ are the intra-group mean and variance, respectively. Building upon the advantage $A _ { i }$ , the policy model $\pi _ { \phi }$ is optimized as follows:

$$
\mathcal {L} _ {\mathrm{GRPO}} (\phi) = \mathbb {E} \left[ \frac {1}{\mathrm{N} _ {\mathrm{rol}}} \sum_ {i = 1} ^ {\mathrm{N} _ {\mathrm{rol}}} \left(\mathcal {L} _ {i} ^ {\mathrm{CLIP}} (\phi) - \beta \mathcal {L} _ {i} ^ {\mathrm{KL}} (\phi)\right) \right],\tag{1}
$$

$$
\mathcal {L} _ {i} ^ {\mathrm{CLIP}} (\phi) = \min \left(\rho_ {i} (\phi) A _ {i}, \operatorname{clip} \left(\rho_ {i} (\phi), 1 - \epsilon , 1 + \epsilon\right) A _ {i}\right),\tag{2}
$$

$$
\mathcal {L} _ {i} ^ {\mathrm{KL}} (\phi) = \gamma_ {i} (\phi) - \log \gamma_ {i} (\phi) - 1,\tag{3}
$$

where $\beta$ controls the KL penalty strength, $\epsilon$ is the clipping parameter, $\begin{array} { r } { \rho _ { i } ( \phi ) = \frac { \pi _ { \phi } ( o _ { i } | q ) } { \pi _ { \phi _ { \mathrm { o l d } } } ( o _ { i } | q ) } } \end{array}$ is the probability ratio between the current policy and the old policy, and $\begin{array} { r } { \gamma _ { i } ( \phi ) = \frac { \pi _ { \mathrm { r e f } } ( o _ { i } | q ) } { \pi _ { \phi } ( o _ { i } | q ) } } \end{array}$ is the probability ratio of the reference policy to the current policy. The advantage $A _ { i }$ acts as a multiplicative weight, directly scaling the policy update.

While GRPO effectively stabilizes optimization under ideal conditions, its intra-group standardization treats every sample in a group uniformly, implicitly assuming that all reward signals $r _ { i }$ are equally reliable. This homogeneous treatment is problematic because the RM is not equally confident about every rollout: as the policy $\pi _ { \phi }$ continuously evolves (Miao et al., 2024), it produces increasingly diverse responses, many of which the RM finds confusing and scores unreliably. Crucially, the standardization in GRPO is oblivious to this unreliability. When the RM assigns a confusing rollout a spuriously extreme reward, this single outlier inflates the group statistics $\mu$ and σ and, after standardization, is granted a disproportionately large advantage. The policy is thus pushed to imitate exactly those samples the RM is least certain about, while genuinely high-quality responses are squeezed toward (or below) the mean and under-rewarded. This amplification of unreliable signals, induced by the uniform advantage weighting, lies at the heart of reward hacking (Fu et al., 2025) and unstable training.

## 2.2. Uncertainty Quantification

Uncertainty quantification (UQ) aims to equip a deterministic predictor with a measure of how reliable its outputs are (Azizi et al., 2026). Instead of returning a single scalar $r _ { \theta } ( x )$ , a UQ method augments the RM with a prediction interval $\mathcal { T } ( x ) = [ r _ { \mathrm { l o } } ( x ) , r _ { \mathrm { h i } } ( x ) ]$ that is expected to contain the unobserved ground-truth reward R with high probability. Formally, given a target miscoverage rate $\alpha \in ( 0 , 1 )$ , the desired marginal coverage property requires

$$
\mathbb {P} \big [ R \in \mathcal {I} (X) \big ] \geq 1 - \alpha .\tag{4}
$$

However, it only constrains the coverage on average over X, which may mask variation across single input. A more desirable guarantee is conditional coverage,

$$
\mathbb {P} \big [ R \in \mathcal {I} (X) \mid X = x \big ] \geq 1 - \alpha , \quad \forall x,\tag{5}
$$

which a sound procedure approximates in practice and attains asymptotically under appropriate conditions. Conditional coverage is essential in our setting: only by reflecting the conditional reward distribution $\mathbb { P } _ { R | X }$ can the interval width faithfully capture the reliability of each individual prompt-response pair, rather than an average over the rollout group. Among intervals meeting these coverage targets, narrower ones are preferred, as they yield a sharper and more discriminative uncertainty measure.

The half-width of the interval, $\Delta ( x ) = \textstyle \frac { 1 } { 2 } ( r _ { \mathrm { h i } } ( x ) - r _ { \mathrm { l o } } ( x ) )$ thus serves as a natural, instance-wise measure of predictive uncertainty: a wide interval signals that the RM is confused and evaluates the sample unreliably, whereas a narrow one reflects a confident and trustworthy prediction. This is precisely the reliability signal that $\mathrm { G R P O ^ { \circ } s }$ uniform standardization lacks, as it allows us to distinguish rollouts the RM scores confidently from those it merely guesses at, preventing unreliable samples from dominating the advantage. To enforce (4) without distributional assumptions on the data, we reserve a held-out calibration set $\mathcal { D } _ { \mathrm { c a l } }$ , drawn from the same train distribution as ${ \mathcal { D } } _ { \mathrm { t r } }$ , for interval calibration.

## 3. Methodology

## 3.1. Motivation

The reliability of reward signals is the cornerstone of stable RLHF. In the standard pipeline, a reward model is trained on a static preference dataset to proxy human values, and the policy is then optimized to maximize its scores. However, GRPO aggregates rewards within each rollout group through intra-group standardization, which weights every sample uniformly and presumes that all reward estimates are equally trustworthy. In practice this assumption breaks down: as the policy explores, it generates responses the RM is confused about and scores unreliably. Because standardization is blind to this unreliability, a confusing sample that happens to receive an extreme score is granted an outsized advantage, steering the policy toward unreliable signals and triggering severe reward hacking (Amodei et al., 2016; Fu et al., 2025; Miao et al., 2024) that misguides the optimization process.

Eliminating this homogeneous treatment of rewards in GRPO introduces two fundamental challenges. ❶ Reward models cannot signal when their predictions are unreliable. They output a single scalar score for any given prompt-response pair, providing no indication of whether a particular rollout is evaluated reliably or merely guessed at. Consequently, there is no way to tell trustworthy reward estimates apart from confusing, unreliable ones. ❷ GRPO standardization amplifies exactly the samples that are least trustworthy. By standardizing rewards within a generated group, GRPO treats all signals as equally reliable. A confusing rollout that receives a spuriously extreme reward thus skews the group’s mean and variance and, after standardization, is assigned a disproportionately large advantage, while well-evaluated, high-quality responses are pushed below the mean and under-rewarded.

Case study. To provide concrete evidence for the above challenges, we present a representative case study in Figure 1. Consider a policy generating a group of four responses to a prompt asking for a “brief and practical tip”. The first response accurately follows the instruction, providing concise and useful advice. In contrast, the fourth response is an atypical, hard-to-judge sample that violates the “brief” constraint by exploiting verbosity, bold formatting, and repetitive buzzwords. For Challenge ❶, the deterministic RM cannot express that it is confused by this unusual response and instead emits a single, spuriously high score (20.0) that overshadows the genuinely helpful response (8.0), with no accompanying signal of its low reliability. For Challenge ❷, during GRPO standardization this single outlier inflates the group mean to 9.0; the highquality response is consequently pushed below the mean and penalized with a negative advantage (−0.15), while the unreliable response receives a massive positive advantage (+1.66). The uniform standardization thus amplifies precisely the sample the RM is least certain about, injecting misleading updates that penalize aligned behavior while reinforcing reward hacking.

Some might note prior works on uncertainty quantification; however, their practical utility for stable RLHF remains underexplored. For instance, ensemble-based uncertainty methods incur prohibitive computational overhead for LLMs (Coste et al., 2023; Eisenstein et al., 2023). While distribution-free interval estimators offer rigorous coverage guarantees, classical variants are difficult to deploy efficiently within the online RLHF loop. Therefore, developing an effective uncertainty quantification framework for reward modeling and adapting it to reweight GRPO advantages remains an open and critical challenge.

## 3.2. Reward Model Uncertainty Estimation

To resolve Challenge ❶, instead of treating the RM as a deterministic point estimator, we model the reward through uncertainty quantification: the RM directly estimates the conditional reward distribution via a set of quantiles, from which both a point reward and an adaptive prediction interval are derived for every prompt-response pair.

Quantile Estimation. Rather than producing a single deterministic score, the RM is parameterized to output K+1 equiprobable conditional quantiles of the distribution $\mathbb { P } _ { R | X } ,$

$$
\hat {q} _ {0} (x) \leq \hat {q} _ {1} (x) \leq \dots \leq \hat {q} _ {\mathrm{K}} (x),\tag{6}
$$

where ${ \hat { q } } _ { k } ( x )$ estimates the conditional quantile at level $\tau _ { k } =$ $k / \mathrm { K }$ . All quantile outputs are trained jointly on ${ \mathcal { D } } _ { \mathrm { t r } }$ by minimizing the pinball loss (Steinwart & Christmann, 2011)

$$
\begin{array}{c} \mathcal {L} _ {\text {pinball}} (\theta) = \frac {1}{| \mathcal {D} _ {\text {tr}} |} \sum_ {i = 1} ^ {| \mathcal {D} _ {\text {tr}} |} \sum_ {k = 0} ^ {K} \rho_ {\tau_ {k}} \big (r _ {i} - \hat {q} _ {k} (x _ {i}) \big), \\ \rho_ {\tau} (u) = \tau \cdot \max (0, u) + (1 - \tau) \cdot \max (0, - u), \end{array}\tag{7}
$$

where the check function $\rho _ { \tau } ( \cdot )$ penalizes under- and overestimation asymmetrically according to the target level $\tau ,$ so that its minimizer recovers the conditional τ-quantile.

We take the median quantile as the point reward consumed by GRPO, i.e., $r _ { \theta } ( x ) \triangleq { \hat { q } } _ { \mathrm { K / 2 } } ( x ) ^ { 1 }$ <sup>1</sup>, so that the scalar score and its surrounding uncertainty are jointly produced by a single quantile-based model. The consecutive quantiles partition the reward axis into K interquantile intervals

$$
\mathcal {I} _ {k} (x) = \left(\hat {q} _ {k - 1} (x), \hat {q} _ {k} (x) \right], \quad k = 1, \dots , K,\tag{8}
$$

each carrying approximately probability mass $1 / \mathrm { K } .$ . Narrow intervals indicate confident regions, while wide intervals signal uncertain regions, which naturally captures the skewness and heteroscedasticity of reward distributions.

Conformity Score. For any integer m, let ${ \mathcal { I } } _ { m } ( x )$ denote the shortest union of m consecutive interquantile intervals

$$
\begin{array}{r l} & {\mathcal {J} _ {m} (x) = \left(\hat {q} _ {k _ {m}} (x), \hat {q} _ {k _ {m} + m} (x) \right],} \\ & {\quad k _ {m} = \underset {0 \leq k \leq K - m} {\arg \min} \left(\hat {q} _ {k + m} (x) - \hat {q} _ {k} (x)\right),} \end{array}\tag{9}
$$

where $k _ { m }$ is the lower-endpoint index of the narrowest minterval block. For each calibration sample $( x _ { i } , r _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } }$ we define the conformity score as the minimum number of interquantile intervals needed to cover the observed reward,

$$
s (x _ {i}, r _ {i}) = \min \left\{m \in \{1, \dots , K \}: r _ {i} \in \mathcal {J} _ {m} (x _ {i}) \right\}.\tag{10}
$$

Intuitively, samples that fall into long intervals receive larger scores, while those in short intervals receive smaller ones.

Calibration and Prediction. Following the standard thresholding principle, we set mˆ to the n-th smallest $s ( x _ { i } , r _ { i } )$ , where $n = \left\lceil ( 1 - \alpha ) ( 1 + | \mathcal { D } _ { \mathrm { c a l } } | ) \right\rceil$ . So that at least a $( 1 - \alpha )$ fraction of calibration responses are covered. For a new rollout sample $\boldsymbol { x } ^ { \mathrm { n e w } }$ , the prediction interval is

$$
\mathcal {I} (x ^ {\text { new }}) = \mathcal {J} _ {\hat {m}} (x ^ {\text { new }}) = \left(\hat {q} _ {k _ {\hat {m}}} (x ^ {\text { new }}), \hat {q} _ {k _ {\hat {m}} + \hat {m}} (x ^ {\text { new }}) \right].\tag{11}
$$

Crucially, the width of this interval expands automatically for responses whose rewards fall in sparsely supported, lowdensity regions. These are precisely the samples the RM evaluates unreliably, and the resulting width provides the uncertainty signal needed to reweight the GRPO advantage.

Theoretical Guarantees. Our construction attains conditional coverage, which is what makes the interval width a faithful per-sample reliability signal for the reweighting in Section 3.3. We use the following standard assumptions in the coverage analysis: the calibration samples and test point are exchangeable; for conditional coverage, calibration/test samples are i.i.d., the learned quantiles consistently approximate the true conditional reward distribution, and the conditional reward distribution is unimodal so that the merged intervals are nested and become wider as m increases.

Theorem 3.1 (Marginal Coverage). If the calibration set $\mathcal { D } _ { \mathrm { c a l } }$ and a rollout point $( X , R )$ are exchangeable, then the prediction interval $\mathcal { T } = \mathcal { T } _ { \hat { m } }$ satisfies

$$
\mathbb {P} [ R \in \mathcal {J} _ {\hat {m}} (X) ] \geq 1 - \alpha .\tag{12}
$$

Proof. Let $n = | \mathcal { D } _ { \mathrm { c a l } } |$ and denote the calibration conformity scores by $s _ { i } = s ( X _ { i } , R _ { i } )$ . By exchangeability, the augmented collection $\{ s _ { 1 } , \ldots , s _ { n } , s ( X , R ) \}$ is exchangeable, so the rank of the test score among these $n + 1$ scores is uniform up to tie-breaking. Our calibration rule chooses mˆ as the $\lceil ( 1 - \alpha ) ( n + 1 ) \rceil$ -th smallest calibration score. Since the intervals ${ \mathcal { I } } _ { m } ( x )$ are nested in m, the coverage event is equivalent to the score event,

Figure 1. Case study of how GRPO’s uniform standardization amplifies unreliable rewards. The deterministic RM emits a spuriously high score for an atypical, hard-to-judge response; standardization inflates its advantage while unfairly penalizing the aligned response.

Figure 2. Framework of our proposed UARM. The offline phase equips the reward model with calibrated uncertainty estimation, and the online phase reweights the GRPO advantage by the estimated interval width to suppress unreliable samples.

$$
R \in \mathcal {J} _ {\hat {m}} (X) \quad \Longleftrightarrow \quad s (X, R) \leq \hat {m}.\tag{13}
$$

Therefore, the test point is covered whenever its rank is no larger than $\lceil ( 1 - \alpha ) ( n + 1 ) \rceil$ . Consequently,

$$
\begin{array}{r l} & {\mathbb {P} [ R \in \mathcal {J} _ {\hat {m}} (X) ] = \mathbb {P} [ s (X, R) \leq \hat {m} ]} \\ & {\qquad \geq \frac {\lceil (1 - \alpha) (n + 1) \rceil}{n + 1} \geq 1 - \alpha ,} \end{array}\tag{14}
$$

which proves the finite-sample marginal coverage guarantee.

Theorem 3.2 (Conditional Coverage). Assume that calibration and test samples are i.i.d.; the learned quantiles consistently estimate the conditional reward distribution, $i . e . ,$ for some $\rho _ { n } \to 0 , F ( \hat { q } _ { k } ( X ) \mid X )$ is within o(1) of $k / \mathrm { K }$ uniformly over quantile levels with high probability; and the conditional reward distribution is unimodal so that the merged intervals $\mathcal { I } _ { m }$ are nested. Then, as $| \mathcal { D } _ { \mathrm { c a l } } | \to \infty ,$ , there exist $\gamma , \zeta \to 0$ such that the prediction interval $\mathcal { T } = \mathcal { I } _ { \hat { m } }$ achieves asymptotic conditional coverage,

$$
\mathbb {P} \left[ \mathbb {P} [ R \in \mathcal {J} _ {\hat {m}} (X) \mid X ] \geq 1 - \alpha - \gamma \right] \geq 1 - \zeta .\tag{15}
$$

Proof. Let $n = | \mathcal { D } _ { \mathrm { c a l } } |$ . The proof has three steps. First, define the empirical conditional CDF induced by the learned quantiles as ${ \hat { F } } ( { \hat { q } } _ { k } ( X ) \mid X ) = k / \mathrm { K }$ . By quantile consistency, $\hat { F }$ uniformly approximates the true conditional CDF $F$ over the quantile grid with high probability. More concretely, there exists a bad set $A _ { n }$ such that

$$
\sup _ {k} | \hat {F} (\hat {q} _ {k} (X) \mid X) - F (\hat {q} _ {k} (X) \mid X) | \leq O (\rho_ {n} ^ {1 / 3})\tag{16}
$$

for all $X \notin A _ { n }$ , while $\mathbb { P } [ X \in A _ { n } ] \le O ( \rho _ { n } ^ { 1 / 3 } )$ . Hence, on the good set $A _ { n } ^ { c }$ , any merged interval spanning m adjacent interquantile bins captures conditional mass close to $m / \mathrm { K } \colon$

$$
\begin{array}{c} \mathbb {P} [ R \in \mathcal {J} _ {m} (X) \mid X ] \\ = F (\hat {q} _ {k _ {m} + m} (X) \mid X) - F (\hat {q} _ {k _ {m}} (X) \mid X) \\ \geq \frac {m}{K} - O (\rho_ {n} ^ {1 / 3}). \end{array}\tag{17}
$$

Second, this pointwise mass control transfers to the calibration scores. Because $s ( X _ { i } , R _ { i } ) \leq m$ iff $R _ { i } \in \mathcal { I } _ { m } ( X _ { i } )$ ， Hoeffding concentration implies that the empirical fraction of calibration samples with scores at most m concentrates around its expectation, up to $O ( { \sqrt { \log n / n } } )$ . Taking $m ^ { \star } = \lceil ( 1 - \alpha ) \mathrm { K } \rceil$ , enough calibration scores fall below $m ^ { \star } + O ( \rho _ { n } ^ { 1 / 3 } + \sqrt { \log n / n } )$ with probability tending to one. Since mˆ is the empirical $( 1 - \alpha )$ quantile of the calibration scores, we obtain

$$
\hat {m} = m ^ {\star} + O \left(\rho_ {n} ^ {1 / 3} + \sqrt {\frac {\log n}{n}}\right)\tag{18}
$$

with high probability. A symmetric lower-tail argument, together with unimodality/nestedness of the intervals, prevents mˆ from being asymptotically smaller than the oracle count.

Finally, substituting this concentration of mˆ into the conditional mass bound for $\mathcal { I } _ { \hat { m } } ( X )$ gives, outside a set whose probability vanishes,

$$
\mathbb {P} [ R \in \mathcal {J} _ {\hat {m}} (X) \mid X ] \geq 1 - \alpha - O \left(\rho_ {n} ^ {1 / 3} + \sqrt {\frac {\log n}{n}}\right).\tag{19}
$$

Thus the theorem holds by setting $\gamma = O ( \rho _ { n } ^ { 1 / 3 } + \sqrt { \log n / n } )$ and $\zeta = O ( \rho _ { n } ^ { 1 / 3 } ) + o ( 1 )$ , both of which vanish as $n $ ∞. □

## 3.3. Uncertainty-Aware Advantage Reweighting

To resolve Challenge ❷, we replace GRPO’s uniform intragroup standardization with a heteroscedastic advantage reweighting that systematically down-weights unreliable samples by treating the conformal interval width as observation noise and decomposing the observed reward variance into signal and noise components.

Observation Noise Model. We interpret the prediction interval width $\varphi ( x _ { i } ) \triangleq | { \mathcal { T } } ( x _ { i } ) | = { \hat { q } } _ { k _ { \hat { m } } + { \hat { m } } } ( x _ { i } ) - { \hat { q } } _ { k _ { \hat { m } } } ( x _ { i } )$ as capturing per-sample measurement uncertainty in the reward estimate. Under a local Gaussianity assumption, we convert this width into an observation noise variance,

$$
\sigma_ {\mathrm{noise}, i} ^ {2} = \left(\frac {\varphi (x _ {i})}{z _ {1 - \frac {\alpha}{2}}}\right) ^ {2}, \quad \bar {\sigma} _ {\mathrm{noise}} ^ {2} = \frac {1}{N _ {\mathrm{rol}}} \sum_ {j = 1} ^ {N _ {\mathrm{rol}}} \sigma_ {\mathrm{noise}, j} ^ {2},\tag{20}
$$

where $z _ { 1 - \frac { \alpha } { 2 } }$ is the standard normal quantile corresponding to the coverage level. Samples with wide intervals yield large $\sigma _ { \mathrm { n o i s e } , i } ^ { 2 }$ , indicating heteroscedastic observation uncertainty that varies across the rollout group.

Signal-Noise Decomposition. The naive group variance $\begin{array} { r } { \sigma ^ { 2 } \ = \ \frac { 1 } { \mathrm { N } _ { \mathrm { r o l } } } \sum _ { j } ( r _ { j } \ - \ \mu ) ^ { 2 } } \end{array}$ conflates true signal variation with measurement error. Under an additive noise model $r _ { i } = r _ { \mathrm { t r u e } , i } + \varepsilon _ { i }$ where $\varepsilon _ { i }$ has variance $\sigma _ { \mathrm { n o i s e } , i } ^ { 2 } ,$ the observed variance decomposes as $\sigma ^ { 2 } \approx \mathrm { V a r } [ r _ { \mathrm { t r u e } } ] + \mathbb { E } [ \sigma _ { \mathrm { n o i s e } } ^ { 2 } ]$ We recover the signal variance by subtracting the average observation noise,

$$
\sigma_ {\mathrm{signal}} ^ {2} = \max \left(0, \sigma^ {2} - \bar {\sigma} _ {\mathrm{noise}} ^ {2}\right) + \zeta ,\tag{21}
$$

where $\zeta > 0$ ensures numerical stability. This decomposition isolates the variance attributable to genuine reward differences from that due to unreliable measurement.

Heteroscedastic Advantage. We define the uncertaintyaware advantage as

$$
\tilde {A} _ {i} = \frac {\sigma_ {\mathrm{signal}} ^ {2}}{\sigma_ {\mathrm{signal}} ^ {2} + \sigma_ {\mathrm{noise} , i} ^ {2}} \cdot \frac {r _ {i} - \mu}{\sigma_ {\mathrm{signal}}},\tag{22}
$$

where $\begin{array} { r } { \mu \ = \ \frac { 1 } { \mathrm { N } _ { \mathrm { r o l } } } \sum _ { j } r _ { j } } \end{array}$ is the unweighted group mean. The prefactor $\sigma _ { \mathrm { s i g n a l } } ^ { - 2 } / ( \sigma _ { \mathrm { s i g n a l } } ^ { 2 } + \sigma _ { \mathrm { n o i s e } , i } ^ { 2 } )$ acts as a samplespecific reliability weight: for high-uncertainty samples with large $\sigma _ { \mathrm { n o i s e } , i } ^ { 2 } ,$ this ratio approaches zero, effectively suppressing their influence; for confident samples with small $\sigma _ { \mathrm { n o i s e } , i } ^ { 2 } .$ , the weight approaches one, preserving the full advantage magnitude. This heteroscedastic formulation provably down-weights the samples the RM evaluates least reliably, without requiring costly ensemble forward passes.

Connection to GRPO. When observation noise is uniform across the group $( \sigma _ { \mathrm { n o i s e } , i } ^ { 2 } \equiv \sigma _ { \mathrm { n o i s e } } ^ { 2 } )$ , the reliability weight becomes constant and Eq. (22) reduces to standard GRPO standardization. The computational overhead is negligible, as all quantities follow directly from the conformal intervals computed in Section 3.2. Returning to the case study in Figure 1, the atypical response with spuriously high reward now exhibits a wide interval and large $\sigma _ { \mathrm { n o i s e } , 4 } ^ { 2 } ,$ receiving a reliability weight near zero; its advantage is suppressed to near-zero magnitude, preventing it from dominating the policy update and steering training away from reward hacking.

## 3.4. The Workflow of UARM

We present the workflow of UARM in Algorithm 1, which couples an offline uncertainty-calibration with an online uncertainty-aware optimization, detailed as follows.

First, in the offline phase, we train and calibrate the reward model. We parameterize the RM as a multi-output quantile estimator and train it on ${ \mathcal { D } } _ { \mathrm { t r } }$ by minimizing the pinball loss in Eq. (7), reading off the point reward as the median quantile $r _ { \theta } = \hat { q } _ { \mathrm { K / 2 } }$ (step 1). For each calibration sample, we compute its conformity score as the minimum number of interquantile intervals needed to cover the observed reward (step 2), and select the threshold mˆ as the n-th smallest score with $n = \lceil ( 1 - \alpha ) ( 1 + | \mathcal { D } _ { \mathrm { c a l } } | ) \rceil$ (step 3). This phase equips the RM with a calibrated interval $\mathcal { I } _ { \hat { m } }$ satisfying the coverage guarantees, and is performed only once.

Second, in the online phase, we optimize the policy with uncertainty-aware GRPO. At each iteration, we sample a rollout group from the old policy and score it with the reward head $r _ { \theta }$ (step 4), then form the prediction intervals and compute their widths $\varphi ( x _ { i } )$ as well as the corresponding observation noise variances $\sigma _ { \mathrm { n o i s e } , i } ^ { 2 }$ via Eq. (20) (step 5). We decompose the observed group variance into signal and noise components via Eq. (21) (step 6), and construct the heteroscedastic advantages ${ \tilde { A } } _ { i }$ via Eq. (22) (step 7); finally, the policy is updated by the GRPO objective in Eq. (1) with A<sup>˜</sup> (step 8). This phase reuses the calibrated reward model at negligible overhead, since the intervals follow directly from quantile evaluations without any binning or density-ratio estimation.

## 4. Experiments

In this section, we empirically validate the efficacy of UARM on three preference datasets. Specifically, we evaluate whether UARM can produce reliable uncertainty estimates and improve uncertainty-ranked reward prediction quality compared with competitive uncertainty quantification baselines.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 The workflow of UARM.

Input: offline preference set  $D_{tr}$ , calibration set  $D_{cal}$ , miscoverage rate  $\alpha$ , learning rate  $\eta$

Parameter: quantile model  $\{\hat{q}_{k}\}_{k=0}^{K}$  (point reward  $r_{\theta} = \hat{q}_{K/2}$ ), policy  $\pi_{\phi}$

Offline UQ Calibration

1: train  $\{\hat{q}_{k}\}_{k=0}^{K}$  on  $D_{tr}$  by minimizing the pinball loss in Eq. (7)

2:  $s(x_{i}, r_{i}) \leftarrow \min\{m : r_{i} \in \mathcal{J}_{m}(x_{i})\}$ ,  $\forall(x_{i}, r_{i}) \in \mathcal{D}_{\mathrm{cal}}$

3:  $\hat{m} \leftarrow \text{the } n\text{-th smallest } s(x_{i}, r_{i})$ ,  $n = \lceil (1 - \alpha)(1 + |\mathcal{D}_{\mathrm{cal}}|) \rceil$

Online Uncertainty-Aware GRPO

4: for each GRPO iteration do

5:  $\{o_{i}\}_{i=1}^{N_{\mathrm{rol}}} \sim \pi_{\phi_{\mathrm{old}}}(\cdot | p)$ ;  $r_{i} \leftarrow r_{\theta}(x_{i})$ ,  $\forall i$

6:  $\varphi(x_{i}) \leftarrow |\mathcal{J}_{\hat{m}}(x_{i})|$ ;  $\sigma_{\mathrm{noise},i}^{2} \leftarrow (\varphi(x_{i}) / z_{1-\alpha/2})^{2}$ ,  $\forall i$

7:  $\mu, \sigma^{2} \leftarrow \text{unweighted group mean and variance}; \sigma_{\mathrm{signal}}^{2} \leftarrow Eq. (21)$

8:  $\tilde{A}_{i} \leftarrow \frac{\sigma_{\mathrm{signal}}^{2}}{\sigma_{\mathrm{signal}}^{2} + \sigma_{\mathrm{noise},i}^{2}} \cdot \frac{r_{i}-\mu}{\sigma_{\mathrm{signal}}}$ ,  $\forall i$

9:  $\phi \leftarrow \phi - \eta \cdot \nabla L_{GRPO}(\phi)$  with advantages  $\tilde{A}_{i}$

10: end for
</div>

## 4.1. Experimental Setup

Datasets. We conduct empirical evaluations on Help-Steer (Wang et al., 2024), UltraFeedback (Cui et al., 2023), and PKU-SafeRLHF (Ji et al., 2025), using Helpfulness, Overall Score, and Severity Level as preference proxies, respectively. For each dataset, we hold out 20% of the training split as the calibration set, while keeping the original test set exclusively for evaluation. Detailed dataset statistics and configurations are provided in the Appendix.

Baselines. We benchmark UARM against a comprehensive suite of uncertainty quantification methods, including: (1) Model-based Uncertainty Estimation methods, such as MC-Dropout (Gal & Ghahramani, 2016), Deep Ensembles (Lakshminarayanan et al., 2017), DER (Amini et al., 2020), Packed Ensemble (Laurent et al., 2023), VBLL (Harrison et al., 2024), and TorchNaut (Kelen et al., 2025); and (2) Distribution-free Interval Estimation methods: SCP (Lei et al., 2018b), CQR (Romano et al., 2019), WCP (Tibshiran et al., 2019), ACI (Gibbs & Candes, 2021), PRCP (Yan et al., 2024), SCCP (van der Laan & Alaa, 2024), Clear (Azizi et al., 2026), and CPCP (Chen & Li, 2026).

Evaluation Metrics. We employ three uncertainty-ranked regression metrics, namely $\dot { \mathrm { R } ^ { 2 } } @ 5 0 .$ MSE@50, and MAE@50, to evaluate point prediction quality on samples with lower estimated uncertainty. Specifically, each method first estimates uncertainty on the test set and ranks test samples in ascending order of uncertainty; the top 50% least uncertain samples are then selected for evaluation. For the naive baseline without uncertainty estimates, we randomly select the corresponding percentage of test samples and repeat this process five times for reporting. Let $ { s } _ { 5 0 }$ denote this selected subset, with ground-truth rewards $y _ { i } .$ , point predictions $\hat { y } _ { i }$ , and mean target value $\bar { y } _ { 5 0 } = | S _ { 5 0 } | ^ { - 1 } \sum _ { i \in S _ { 5 0 } } y _ { i }$ The metrics are defined as

$$
\mathbf {R} ^ {2} @ 5 0 = 1 - \frac {\sum_ {i \in \mathcal {S} _ {5 0}} (y _ {i} - \hat {y} _ {i}) ^ {2}}{\sum_ {i \in \mathcal {S} _ {5 0}} (y _ {i} - \bar {y} _ {5 0}) ^ {2}},
$$

$$
\mathrm{MSE@50} = \frac {1}{| \mathcal {S} _ {5 0} |} \sum_ {i \in \mathcal {S} _ {5 0}} (y _ {i} - \hat {y} _ {i}) ^ {2},\tag{23}
$$

$$
\mathrm{MAE@50} = \frac {1}{| \mathcal {S} _ {5 0} |} \sum_ {i \in \mathcal {S} _ {5 0}} | y _ {i} - \hat {y} _ {i} |.
$$

Implementation Details. We implement the quantile reward model using an LLM backbone followed by a lightweight multi-layer perceptron head. To ensure a fair comparison, we initialize the backbone from FsfairX-LLaMA3-RM-v0.1<sup>2</sup>, and fix the MLP head to hidden dimensions of 256, 64, 1. We optimize the models using Adam (Kingma & Ba, 2015) for up to 600 epochs, employing early stopping with a patience of 30 epochs to ensure convergence. Key hyperparameters are tuned on a validation set, with update rate $\eta \in [ 1 \times 1 0 ^ { - 5 } , 1 \times 1 0 ^ { - 3 } ]$ and batch size $B \in [ 6 4 , 2 0 4 8 ]$ . Further details are provided in the Appendix.

## 4.2. Results & Analysis

Table 1 presents the comparative results of uncertainty quantification on three preference datasets. We have the following observations: ❶ Naive confidence selection is insufficient. Without uncertainty estimates, the Naive baseline can only evaluate randomly selected samples and thus consistently lags behind uncertainty-aware methods. This confirms that reliable confidence estimation is crucial for identifying samples on which the reward model can make accurate point predictions. ❷ Existing uncertainty quantification methods improve reward reliability to varying degrees. Model-based approaches such as MC-Dropout, Deep Ensembles, and MCNF, as well as distribution-free interval estimation methods such as CQR and Clear, generally outperform the Naive baseline by selecting lower-uncertainty samples. Nevertheless, their performance remains inconsistent across datasets and metrics, suggesting that either model-intrinsic uncertainty or generic conformal intervals alone may be insufficient for reward modeling. ❸ UARM consistently achieves the best uncertainty-ranked prediction performance. Across all three datasets, UARM obtains the highest $\mathbb { R } ^ { 2 } @ 5 0$ and the lowest MSE@50 and MAE@50. Compared with the strongest baselines, UARM improves $\scriptstyle \mathrm { R } ^ { 2 } @ 5 0$ from 0.527 to 0.543 on HelpSteer, from 0.770 to 0.794 on UltraFeedback, and from 0.955 to 0.985 on PKU-SafeRLHF. The gains are especially pronounced on PKU-SafeRLHF, where UARM reduces MSE@50 from

Uncertainty-Aware Reward Modeling for Stable RLHF
Table 1. Comparative analysis of UARM versus baseline models with fixed miscoverage rate α = 0.1.

<table><tr><td>Dataset</td><td colspan="3">HelpSteer</td><td colspan="3">UltraFeedback</td><td colspan="3">PKU-SafeRLHF</td></tr><tr><td>Method</td><td> $R^2@50$ </td><td>MSE@50</td><td>MAE@50</td><td> $R^2@50$ </td><td>MSE@50</td><td>MAE@50</td><td> $R^2@50$ </td><td>MSE@50</td><td>MAE@50</td></tr><tr><td colspan="10">Model-based Uncertainty Estimation Methods</td></tr><tr><td>Naive</td><td>0.357</td><td>0.611</td><td>0.595</td><td>0.563</td><td>1.481</td><td>0.832</td><td>0.850</td><td>0.173</td><td>0.206</td></tr><tr><td>MC-Dropout (Gal &amp; Ghahramani, 2016)</td><td>0.369</td><td>0.437</td><td>0.506</td><td>0.607</td><td>0.949</td><td>0.717</td><td>0.863</td><td>0.073</td><td>0.135</td></tr><tr><td>Deep Ensemble (Lakshminarayanan et al., 2017)</td><td>0.395</td><td>0.433</td><td>0.509</td><td>0.632</td><td>0.473</td><td>0.507</td><td>0.881</td><td>0.110</td><td>0.167</td></tr><tr><td>DER (Amini et al., 2020)</td><td>0.420</td><td>0.557</td><td>0.581</td><td>0.663</td><td>0.403</td><td>0.470</td><td>0.881</td><td>0.098</td><td>0.091</td></tr><tr><td>Packed Ensemble (Laurent et al., 2023)</td><td>0.462</td><td>0.476</td><td>0.537</td><td>0.710</td><td>0.491</td><td>0.514</td><td>0.905</td><td>0.103</td><td>0.147</td></tr><tr><td>TorchNaut (Kelen et al., 2025)</td><td>0.527</td><td>0.499</td><td>0.519</td><td>0.746</td><td>0.463</td><td>0.499</td><td>0.933</td><td>0.050</td><td>0.052</td></tr><tr><td>MCNF (Sosa Marco et al., 2026)</td><td>0.527</td><td>0.428</td><td>0.507</td><td>0.769</td><td>0.503</td><td>0.513</td><td>0.955</td><td>0.042</td><td>0.059</td></tr><tr><td colspan="10">Distribution-free Interval Estimation Methods</td></tr><tr><td>SCP (Lei et al., 2018b)</td><td>0.378</td><td>0.544</td><td>0.569</td><td>0.609</td><td>1.345</td><td>0.815</td><td>0.866</td><td>0.159</td><td>0.200</td></tr><tr><td>CQR (Romano et al., 2019)</td><td>0.409</td><td>0.432</td><td>0.458</td><td>0.623</td><td>0.406</td><td>0.510</td><td>0.881</td><td>0.147</td><td>0.305</td></tr><tr><td>WCP (Tibshirani et al., 2019)</td><td>0.438</td><td>0.545</td><td>0.570</td><td>0.646</td><td>1.150</td><td>0.793</td><td>0.883</td><td>0.141</td><td>0.192</td></tr><tr><td>ACI (Gibbs &amp; Candes, 2021)</td><td>0.476</td><td>0.469</td><td>0.544</td><td>0.678</td><td>0.523</td><td>0.534</td><td>0.905</td><td>0.113</td><td>0.256</td></tr><tr><td>SCCP (van der Laan &amp; Alaa, 2024)</td><td>0.512</td><td>0.491</td><td>0.553</td><td>0.750</td><td>0.547</td><td>0.530</td><td>0.925</td><td>0.064</td><td>0.090</td></tr><tr><td>Clear (Azizi et al., 2026)</td><td>0.521</td><td>0.396</td><td>0.478</td><td>0.770</td><td>0.432</td><td>0.513</td><td>0.940</td><td>0.060</td><td>0.096</td></tr><tr><td>UARM (Ours)</td><td>0.543</td><td>0.387</td><td>0.423</td><td>0.794</td><td>0.383</td><td>0.461</td><td>0.985</td><td>0.013</td><td>0.016</td></tr></table>

Note: “@50” reports the metric on the 50% most confident samples (lowest uncertainty).

0.042 to 0.013 and MAE@50 from 0.052 to 0.016, demonstrating that its calibrated uncertainty estimates more effectively identify reliable reward predictions.

## 5. Conclusion

In this paper, we present UARM, an uncertainty-aware reward modeling framework for more reliable RLHF. UARM addresses two key challenges in reward-based policy optimization: reward models often cannot indicate when their predictions are unreliable, and uniform advantage computation can amplify such unreliable reward signals. To this end, UARM equips reward models with calibrated persample uncertainty estimates through quantile-based conformal prediction and incorporates these estimates into a heteroscedastic advantage reweighting scheme. Experiments on three preference datasets show that UARM consistently improves uncertainty-ranked reward prediction performance over both model-based uncertainty estimation methods and distribution-free interval estimation baselines, demonstrating its effectiveness in identifying more reliable reward predictions.

Limitations & Future Work. This work focuses primarily on the main offline reward modeling experiments, while broader evaluations of downstream online RLHF performance, sensitivity to hyperparameters, and generalization across larger backbones remain important directions for future work. In addition, UARM relies on a held-out calibration set drawn from the training distribution, and its empirical reliability may be affected by severe distribution shift during policy optimization. Future work will extend UARM to adaptive online calibration, study its integration with broader policy optimization algorithms beyond GRPO, and provide more comprehensive theoretical analysis of its impact on RLHF convergence and reward hacking mitigation.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Amini, A., Schwarting, W., Soleimany, A., and Rus, D. Deep evidential regression. Proc. Adv. Neural Inf. Process. Syst., 33:14927–14937, 2020.

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Mane, D. Concrete problems in ai safety.´ arXiv preprint arXiv:1606.06565, 2016.

Azizi, I., Bodik, J., Heiss, J., and Yu, B. CLEAR: Calibrated learning for epistemic and aleatoric risk. In Proc. Int. Conf. Learn. Represent., 2026.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Chen, Q. and Li, B. Colorful pinball: Density-weighted quantile regression for conditional guarantee of conformal prediction, 2026.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Coste, T., Anwar, U., Kirk, R., and Krueger, D. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., et al. Ultrafeedback: Boosting language models with scaled ai feedback. arXiv preprint arXiv:2310.01377, 2023.

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Fu, J., Zhao, X., Yao, C., Wang, H., Han, Q., and Xiao, Y. Reward shaping to mitigate reward hacking in rlhf. arXiv preprint arXiv:2502.18770, 2025.

Gal, Y. and Ghahramani, Z. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Proc. Int. Conf. Mach. Learn., pp. 1050–1059. PMLR, 2016.

Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Gibbs, I. and Candes, E. Adaptive conformal inference under distribution shift. Proc. Adv. Neural Inf. Process. Syst., 34:1660–1672, 2021.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al. Deepseekr1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

Harrison, J., Willes, J., and Snoek, J. Variational bayesian last layers. In Proc. Int. Conf. Learn. Represent., 2024.

Immer, A., Palumbo, E., Marx, A., and Vogt, J. Effective bayesian heteroscedastic regression with deep neural networks. Proc. Adv. Neural Inf. Process. Syst., 36:53996– 54019, 2023.

Ji, J., Hong, D., Zhang, B., Chen, B., Dai, J., Zheng, B., Qiu, T. A., Zhou, J., Wang, K., Li, B., et al. Pku-saferlhf: Towards multi-level safety alignment for llms with human preference. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 31983–32016, 2025.

Kelen, D. M., Jung, A., Kersch, P., and Benczur, A. A.<sup>´</sup> Distribution-free data uncertainty for neural network regression. In Proc. Int. Conf. Learn. Represent., 2025.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Proc. Int. Conf. Learn. Represent., pp. 1–9, 2015.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. Proc. Adv. Neural Inf. Process. Syst., 30, 2017.

Lambert, N., Pyatkin, V., Morrison, J., Miranda, L. J. V., Lin, B. Y., Chandu, K., Dziri, N., Kumar, S., Zick, T., Choi, Y., et al. Rewardbench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 1755– 1797, 2025.

Laurent, O., Lafage, A., Tartaglione, E., Daniel, G., marc Martinez, J., Bursuc, A., and Franchi, G. Packed ensembles for efficient uncertainty estimation. In Proc. Int. Conf. Learn. Represent., 2023.

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. Distribution-free predictive inference for regression. Journal ofthe American Statistical Association, 113(523):1094–1111, 2018a.

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. Distribution-free predictive inference for regression. Journal ofthe American Statistical Association, 113(523):1094–1111, 2018b.

Liu, T., Xiong, W., Ren, J., Chen, L., Wu, J., Joshi, R., Gao, Y., Shen, J., Qin, Z., Yu, T., et al. Rrm: Robust reward model training mitigates reward hacking. arXiv preprint arXiv:2409.13156, 2024.

Miao, Y., Zhang, S., Ding, L., Bao, R., Zhang, L., and Tao, D. Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling. Advances in Neural Information Processing Systems, 37:134387–134429, 2024.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Romano, Y., Patterson, E., and Candes, E. Conformalized quantile regression. Proc. Adv. Neural Inf. Process. Syst., 32, 2019.

Shafer, G. and Vovk, V. A tutorial on conformal prediction. Journal ofmachine learning research, 9(3), 2008.

Skalse, J., Howe, N., Krasheninnikov, D., and Krueger, D. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

Sosa Marco, A., Kirwan, J. D., Toumpa, A., and Gerasimou, S. Uncertainty quantification for deep regression using contextualised normalizing flows. Proc. Adv. Neural Inf. Process. Syst., 38:50711–50736, 2026.

Steinwart, I. and Christmann, A. Estimating conditional quantiles with the help of the pinball loss. Bernoulli, 17, 02 2011. doi: 10.3150/10-BEJ267.

Tibshirani, R. J., Foygel Barber, R., Candes, E., and Ramdas, A. Conformal prediction under covariate shift. Proc. Adv. Neural Inf. Process. Syst., 32, 2019.

van der Laan, L. and Alaa, A. M. Self-calibrating conformal prediction. Proc. Adv. Neural Inf. Process. Syst., 37: 107138–107170, 2024.

Wang, H., Pan, L., Chen, Z., Zheng, C., Chu, Z., Li, X., Lu, Y., Liu, X., Li, H., and Lin, Z. Causalrm: Causaltheoretic reward modeling for rlhf from observational user feedbacks. arXiv preprint arXiv:2603.18736, 2026.

Wang, Z., Dong, Y., Zeng, J., Adams, V., Sreedhar, M. N., Egert, D., Delalleau, O., Scowcroft, J., Kant, N., Swope, A., et al. Helpsteer: Multi-attribute helpfulness dataset for steerlm. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3371–3384, 2024.

Yan, G., Romano, Y., and Weng, T.-W. Provably robust conformal prediction with improved efficiency. In Proc. Int. Conf. Learn. Represent., 2024.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
