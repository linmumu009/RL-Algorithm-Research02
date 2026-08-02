# Privacy-Preserving Reinforcement Learning from Human Feedback via Decoupled Reward Modeling

Young Hyun Cho Purdue University

Will Wei Sun<sup>∗</sup> Purdue University

## Abstract

Preference-based fine-tuning has become an important component in training large language models, and the data used at this stage may contain sensitive user information. A central question is how to design a diferentially private pipeline that is well suited to the distinct structure of reinforcement learning from human feedback. We propose a privacy-preserving framework that imposes diferential privacy only on reward learning and derives the final pol icy from the resulting private reward model. Theoretically, we study the suboptimality gap and show that privacy contributes an additional additive term beyond the usual non-private statistical error. We also establish a minimax lower bound and show that the dominant term changes with sample size and privacy level, which in turn characterizes regimes in which the upper bound is rate-optimal up to logarithmic factors. Empirically, synthetic experiments confirm the scaling predicted by the theory, and experiments on the Anthropic HH-RLHF dataset using the Gemma-2B-IT model show stronger private alignment performance than existing diferentially private baseline methods across privacy budgets.

Keywords: Large Language Models, Preference Fine-tuning, Diferential Privacy, Reward Modeling, Sample Complexity.

## 1 Introduction

Large language models (LLMs) are increasingly used as general-purpose tools across a growing range of downstream and domain-specific applications, including medicine, finance, law, and science (Nie et al., 2024; Qin and Sun, 2024; Zheng et al., 2025; Zhou et al., 2026). In practice, adapting a pretrained model to such settings typically relies on post-training or fine-tuning, which has become a standard part of modern LLM deployment (Ouyang et al., 2022; Bai et al., 2022; Chia et al., 2025). Preference-based LLM fine-tuning is one widely used form of this adap tation. Such fine-tuning is commonly carried out through reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022), where comparative feedback is used to improve a policy when reward feedback is not directly observed.

In particular, a typical LLM alignment pipeline begins with a pretrained model, obtains a reference policy through supervised fine-tuning (SFT) on expert-written or carefully curated instruction-response pairs, and then refines that policy using pairwise preferences over candidate responses, typically through RLHF. Because SFT data are costly to scale and may not fully capture the nuanced judgments needed for alignment, preference-based fine-tuning has become a common extension beyond SFT (Chia et al., 2025). Figure 1 illustrates this canonical RLHF pipeline. Modern preference-based alignment also includes direct formulations such as direct preference optimization (DPO), which bypass explicit reward-model training and instead optimize the policy directly from preference data (Rafailov et al., 2024).

Figure 1: A typical large language model adaptation pipeline. We focus on privacy during the preference fine-tuning stage, where sensitive user interactions can be directly reflected in training records.

On the other hand, this preference-based fine-tuning pipeline can raise privacy concerns. Modern models are susceptible to training-data extraction attacks (Carlini et al., 2021; Nasr et al., 2023) and membership inference attacks (Shokri et al., 2017; Wu and Cao, 2025), which can reveal whether a user’s data was used for training and, in some cases, expose training examples. In preference-based LLM fine-tuning beyond SFT, the training signal is not merely a scalar label, but a full interaction tuple $( x _ { i } , a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , y _ { i } )$ , where $x _ { i }$ is the user prompt, $( a _ { i } ^ { 1 } , a _ { i } ^ { 2 } )$ are candidate responses, and $y _ { i }$ is the associated preference label. Even when explicit identifiers are absent, the prompt $x _ { i }$ may contain sensitive or potentially identifying context, and the derived responses and label may also reveal private user information. Figure 2 illustrates such an example. These concerns call for principled protection against leakage at the tuple level.

Diferential privacy (DP) (Dwork et al., 2006) provides a widely used formal framework for privacy-preserving data analysis and machine learning. A central question is what object should be protected by the privacy definition. Much of the existing literature adopts label-DP (Chowdhury et al., 2024; Zhang et al., 2025; Teku et al., 2025; Wu et al., 2025), which protects only the preference label and is therefore most naturally aligned with protecting the privacy of the preference-label annotator. Our focus is diferent. In the settings we study, the relevant target is the end-user, so privacy should apply to the full interaction tuple rather than to $y _ { i }$ alone. This motivates tuple-level privacy for $( x _ { i } , a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , y _ { i } )$

A key distinction from standard RL is that RLHF does not receive rewards directly from the environment. Instead, it must infer a latent reward signal from preference data, which introduces an additional reward-learning layer into the pipeline. Under DP, this distinction is consequential.

Figure 2: A sensitive interaction record. Even without direct identifiers, prompts can contain quasiidentifiers whose combination may re-identify an individual, motivating tuple-level protection of $( x _ { i } , a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , y _ { i } )$

Privacy can now enter at reward estimation, at downstream policy optimization, or at both stages. If privacy is imposed naively on such a multi-stage pipeline, noise can accumulate across stages, or the data or privacy budget must be divided across them. In this sense, the rewardlearning layer in RLHF is not only a modeling feature but also a central channel through which privacy constraints afect utility.

This raises a second central question beyond what should be protected. One must also decide where privacy should enter the alignment pipeline so that downstream utility is preserved as much as possible. Existing directions already illustrate several possibilities, including introducing DP to DPO (Chen et al., 2025) and introducing DP to each stage of RLHF (Wu et al., 2024). In this sense, existing methods largely arise by applying DP to existing non-private alignment pipelines. The remaining question is whether RLHF admits a privacy-aware design that uses its own structure more directly.

In this work, we take a diferent route. Rather than adding DP to each step of an existing alignment pipeline, we use the two-stage structure of RLHF itself to guide where privacy should enter. We develop a tuple-level private framework that places privacy on reward learning and treats downstream policy improvement as post-processing. We then study the resulting privacy– utility tradeof through theory and numerical experiments.

## 1.1 Our Contributions

Our contributions are summarized as follows.

• Methodology. We propose a private RLHF framework that exploits a structural feature specific to RLHF, rather than simply adding DP to an existing alignment pipeline. Unlike standard RL, where the environment directly provides reward, RLHF first uses preference data to train a reward model and then updates a reference policy using that learned reward. Our framework places DP only on the reward-learning stage and treats downstream policy as post-processing of the resulting private reward model. Placing privacy on reward learning creates a bufer between DP noise and the downstream policy. In LLM fine-tuning, this avoids privatizing the policy itself, which can yield responses worse than the reference policy, and instead uses the private reward model only to re-rank reference-generated responses, without additional privacy cost.

• Theory. We study the suboptimality gap of the policy induced by our framework and show that privacy enters as an additional additive term on top of the usual non-private statistical error. We also establish, to the best of our knowledge, the first minimax lower bound for this private RLHF problem. A key dificulty is that the dominant lower-bound term changes with the balance between sample size and privacy budget. We therefore characterize the regimes in which our method is rate-optimal up to logarithmic factors, including the case where the sample size is suficiently large relative to the privacy scale for the optimal privacy-dominated rate to emerge.

• Empirical validation. We validate the framework on synthetic preference-learning experiments and on an LLM alignment task based on the Anthropic HH-RLHF dataset (Bai et al., 2022) using the Gemma-2B-IT model (Team et al., 2024). The synthetic results con firm the theoretically predicted scaling with sample size, privacy budget, and dimension, and show that our method substantially reduces the incidence of policies that underperform the reference policy relative to existing DP baseline methods. In the LLM experiment, our method demonstrates stronger private alignment performance than existing DP baseline methods across all privacy budgets considered

## 1.2 Related Work

We briefly review prior work on preference-based policy learning and its theoretical foundations, as well as DP for alignment.

Preference-based policy learning and KL-regularized preference optimization. One of the most widely used applications of RLHF is preference fine-tuning for LLMs (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022; Ye et al., 2025). More recently, direct alignment methods have gained prominence by optimizing a closed-form objective that avoids explicit reward-model training and RL in the loop (Rafailov et al., 2024; Garg et al., 2025). From a theoretical viewpoint, a growing line of work develops finite-sample guarantees of KLregularized preference optimization and iterative procedures, clarifying when logged preference data sufices for reliable policy improvement (Xiong et al., 2024; Ye et al., 2024; Xiong et al., 2024; Song et al., 2024; Zhao et al., 2024). Several recent works also revisit the role of the KL regularizer itself—including its interpretation, limitations, and variants (Huang et al., 2024; Aminian et al., 2025; Liu et al., 2025a; Li et al., 2026). Our work complements this theory by quantifying how DP afects the KL-regularized preference optimization.

DP and privacy-preserving alignment. DP is a de facto standard in privacy-preserving framework in machine learning (Dwork et al., 2006). A large literature studies optimal accuracy guarantees under DP for empirical risk minimization and stochastic convex optimization (Dwork et al., 2014; Chaudhuri et al., 2011; Kifer et al., 2012; Bassily et al., 2014; Wang et al., 2017; Bassily et al., 2019). Among many, the most widely used optimization method is noisy gradient-based training, popularized by DP-SGD and closely related mechanisms (Dwork et al., 2006; Abadi et al., 2016; Bu et al., 2020). This line of work has also motivated refined privacy accounting frameworks and composition results that tighten end-to-end privacy loss in iterative training (Kairouz et al., 2015; Mironov, 2017; Bun and Steinke, 2016; Dong et al., 2022).

Building on these foundations, recent work has begun to study DP for preference-based alignment and RLHF. A first key distinction concerns the unit of protection. Much of the recent literature studies label-DP, which protects only the preference label while treating the prompt and candidate responses as public or non-sensitive. This formulation is naturally aligned with protecting annotator feedback, but it does not address leakage from the text content itself. This perspective underlies several recent works on private DPO and related ofline alignment objectives (Zhang et al., 2025; Zhou et al., 2025a,b; Teku et al., 2025). In contrast, our focus is tuple-level privacy, where the prompt, candidate responses, and preference outcome may all carry sensitive information.

A second distinction concerns how privacy is incorporated into the alignment pipeline. Most existing approaches begin with a non-private alignment framework and then introduce a DP mechanism on top of it. When the target is label-DP, a standard way to privatize the labels is randomized response (Warner, 1965), and several recent works follow this route (Zhang et al., 2025; Zhou et al., 2025a,b; Teku et al., 2025). When the target is tuple-level DP, privacy is typically enforced through noisy gradient-based training such as DP-SGD, including private adaptations of DPO and related RLHF procedures (Chen et al., 2025; Wu et al., 2024, 2025; Chowdhury et al., 2024; Korkmaz and Brown-Cohen, 2024). Our contribution difers from these directions both methodologically and theoretically. We study a framework tailored to the dis tinct structure of RLHF, with privacy imposed only on reward learning, and we analyze the resulting suboptimality gap together with minimax lower bounds that characterize when the corresponding rates are optimal up to logarithmic factors.

## 1.3 Paper organization and notation

The remainder of the paper is organized as follows. Section 2 introduces the preliminaries for our work, including DP and RLHF. Section 3 presents the proposed decoupled framework. In section 4, we develop the theoretical analysis, deriving both upper and lower bounds on the suboptimality gap of the induced policy and identifying regimes in which these bounds match up to logarithmic factors. Section 5 reports numerical studies on both synthetic examples and LLM fine-tuning experiments. Section 6 concludes with a discussion of the main implications, limitations, and directions for future work. Most proofs are deferred to the Supplementary Material Section D, while straightforward arguments are given in the main text.

For notation, we use standard asymptotic order notation. For two positive sequences $a _ { n }$ and $b _ { n } , a _ { n } = O ( b _ { n } )$ means that $a _ { n } / b _ { n }$ is uniformly bounded, and $a _ { n } = \Omega ( b _ { n } )$ means that $a _ { n } / b _ { n }$ is bounded away from zero. We write $a _ { n } = { \widetilde { O } } ( b _ { n } )$ and $a _ { n } = { \widetilde \Omega } ( b _ { n } )$ when the corresponding relations hold up to polylogarithmic factors.

## 2 Preliminaries

In this section, we establish the background for our work. We begin by formally outlining the standard pipelines for preference fine-tuning, then introduce the background on DP.

## 2.1 Reinforcement Learning from Human Feedback

A widely used template for learning from preferences is KL-regularized reinforcement learning from human feedback (RLHF). The key object is a reference policy $\pi _ { 0 }$ , and the goal is to improve decision making while preventing the updated policy from drifting too far from $\pi _ { 0 }$ or overfitting to a limited preference dataset. This viewpoint is especially prominent in large language model alignment, where preference fine-tuning is a standard stage after a strong base policy is obtained.

We consider an ofline preference dataset originally recorded as $\mathcal { D } = \{ ( x _ { i } , a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , y _ { i } ) \} _ { i = 1 } ^ { n } ,$ 2 where $x _ { i } \in \mathcal X$ denotes a context, such as a prompt in LLM applications, $a _ { i } ^ { 1 } , a _ { i } ^ { 2 } \in \mathcal { A }$ are two candidate actions, and $y _ { i }$ indicates which candidate is preferred. For the theoretical development, it is more convenient to rewrite each record in ordered form as $( x _ { i } , a _ { i } ^ { w } , a _ { i } ^ { l } )$ , where $a _ { i } ^ { w } \in \mathcal { A }$ and $a _ { i } ^ { l } \in \mathcal { A }$ denote the preferred and non-preferred actions, respectively. A conventional modeling assumption is then the Bradley–Terry preference model (Bradley and Terry, 1952).

Definition 1 (Bradley–Terry Model). For a context x and a preferred/non-preferred pair $( a ^ { w } , a ^ { l } )$ the probability that $a ^ { w }$ is preferred to $a ^ { l }$ is modeled as

$$
\mathbb {P} (a ^ {w} \succ a ^ {l} \mid x) = \frac {\exp (r ^ {*} (x , a ^ {w}))}{\exp (r ^ {*} (x , a ^ {w})) + \exp (r ^ {*} (x , a ^ {l}))} = \sigma (r ^ {*} (x, a ^ {w}) - r ^ {*} (x, a ^ {l})),
$$

where $\sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ is the sigmoid function.

In KL-regularized RLHF, the reference policy $\pi _ { 0 }$ is treated as a strong baseline that encodes prior knowledge and safe behavior, and KL regularization controls the magnitude of the preference-driven update. In the LLM preference fine-tuning pipeline, a common choice of $\pi _ { 0 }$ is a model that has already been trained on a large supervised instruction-following dataset, often referred to as supervised fine-tuning, or $\mathrm { S F T }$ (Ouyang et al., 2022). This choice reflects the practical role of $\pi _ { 0 }$ as an anchor to a broadly competent response distribution, while preference data provide an additional signal that refines behavior without requiring the policy to relearn basic capabilities from scratch.

Under Definition 1, the standard reward-modeling phase estimates a parametric reward function $r _ { \theta } \in \mathcal { R } = \{ r _ { \theta } : \theta \in \Theta \}$ by maximizing the log-likelihood

$$
\hat {\theta} = \underset {\theta \in \Theta} {\operatorname{argmax}} \sum_ {i = 1} ^ {n} \log \sigma \left(r _ {\theta} (x _ {i}, a _ {i} ^ {w}) - r _ {\theta} (x _ {i}, a _ {i} ^ {l})\right).\tag{1}
$$

Once a reward estimator $\hat { r }$ is obtained, the goal is to derive a policy that achieves high reward while remaining close to a reference policy $\pi _ { 0 }$ , which is often the SFT model in LLM pipelines. For any reward function $r$ on $\mathcal { X } \times \mathcal { A }$ , we evaluate a policy $\pi$ by the KL-regularized value

$$
V _ {\eta} (\pi ; r) = \mathbb {E} _ {x \sim d _ {0}} \Big [ \mathbb {E} _ {a \sim \pi (\cdot | x)} \big [ r (x, a) \big ] - \frac {1}{\eta} \mathrm{KL} \big (\pi (\cdot | x) | | \pi_ {0} (\cdot | x) \big) \Big ],\tag{2}
$$

where $d _ { 0 }$ denotes the context distribution, and $\eta > 0$ controls the strength of regularization. Let $\pi _ { r } ^ { \eta } \in \arg \operatorname* { m a x } _ { \pi } V _ { \eta } ( \pi ; r )$ denote the KL-regularized optimizer under r. The closed form solution is as follows:

Lemma 2 (Policy Improvement Oracle). For a fixed context $x ,$ reward r, and reference policy $\pi _ { 0 }$ , any maximizer $\pi _ { r } ^ { \eta } \in$ arg max<sub>π</sub> $V _ { \eta } ( \pi ; r )$ satisfies

$$
\pi_ {r} ^ {\eta} (a \mid x) = \frac {1}{Z _ {r} (x)} \pi_ {0} (a \mid x) \exp (\eta r (x, a)),\tag{3}
$$

where $Z _ { r } ( x ) = \mathbb { E } _ { a \sim \pi _ { 0 } ( \cdot | x ) } [ \exp ( \eta r ( x , a ) ) ]$

Although $\pi _ { r } ^ { \eta }$ admits a closed-form expression, exact sampling from this policy is typically infeasible when the action space is large. The expression involves a normalization term that aggregates $\exp ( \eta r ( x , a ) )$ over the action space, which is computationally prohibitive in LLM settings where actions correspond to long token sequences. This motivates practical methods that approximate the KL-regularized optimizer through parameterized learning.

A common approach is to optimize a parameterized policy with reinforcement learning algorithms such as proximal policy optimization (PPO) (Schulman et al., 2017), which targets the KL-regularized objective while avoiding explicit normalization over ${ \mathcal { A } } .$ Since PPO does not appear explicitly in the main development, we defer a brief background discussion to Appendix A. An alternative is Direct Preference Optimization (DPO) (Rafailov et al., 2024), which leverages the identity implied by (3). In particular, for $\pi _ { r } ^ { \eta }$ one can write

$$
r (x, a) = \frac {1}{\eta} \log \frac {\pi_ {r} ^ {\eta} (a \mid x)}{\pi_ {0} (a \mid x)} + \frac {1}{\eta} \log Z _ {r} (x).
$$

Substituting this relation into the pairwise preference likelihood (1) eliminates the unknown $Z _ { r } ( x )$ and yields a supervised objective over policy parameters

$$
\mathcal {L} _ {\mathrm{DPO}} (\theta ; \mathcal {D}, \pi_ {0}) = - \sum_ {(x, a ^ {w}, a ^ {l}) \in \mathcal {D}} \log \sigma \left(\frac {1}{\eta} \log \frac {\pi_ {\theta} (a ^ {w} \mid x)}{\pi_ {0} (a ^ {w} \mid x)} - \frac {1}{\eta} \log \frac {\pi_ {\theta} (a ^ {l} \mid x)}{\pi_ {0} (a ^ {l} \mid x)}\right).\tag{4}
$$

The primary appeal of DPO and its variants (Azar et al., 2024; Ethayarajh et al., 2024; Xu et al., 2024; Meng et al., 2024) lies in their ability to avoid the computational complexities and training instabilities of RL frameworks. By formulating the process as a single-stage supervised learning, these methods eliminate the need for iterative sampling and complex hyperparameter tuning required by RL-based optimization. However, as we discuss in Section 3, directly imposing DP on the policy parameters in (4) introduces fundamental challenges.

Our analysis focuses on the policy quality under the true reward $r ^ { * }$ . Let $\pi _ { r ^ { * } } ^ { \eta } \in \arg \operatorname* { m a x } _ { \pi } V _ { \eta } ( \pi ; r ^ { * } )$ We quantify the η-regularized suboptimality of a candidate policy π by

$$
\Delta_ {\eta} (\pi) = V _ {\eta} (\pi_ {r ^ {*}} ^ {\eta}; r ^ {*}) - V _ {\eta} (\pi ; r ^ {*}).\tag{5}
$$

In the proposed decoupled framework, $\pi$ will be induced by a private reward estimate, and our theory controls $\Delta _ { \eta } ( \pi )$ in finite samples.

## 2.2 Diferential Privacy

Informally, a mechanism M is DP if the distribution of its output $M ( D )$ is nearly indistinguishable from that of $M ( D ^ { \prime } )$ for any adjacent dataset $D ^ { \prime }$ , where $D$ and $D ^ { \prime }$ difer by a single entry. We formalize the notion of adjacent as their Hamming distance is one.

Among various characterizations of DP–such as Rényi DP (Mironov, 2017) or Gaussian DP (Dong et al., 2022)–aimed at achieving tighter privacy accounting, the following $( \varepsilon , \delta ) – \mathrm { D P }$ remains the most prevalent and standard definition, and thus we also adopt the following:

Definition 3 $( ( \varepsilon , \delta )$ -Diferential Privacy (Dwork et al., 2006)). A mechanism M is $( \varepsilon , \delta )$ diferentially private $i f ,$ for any two adjacent $D , D ^ { \prime }$ and for any measurable event $E$ ,

$$
\mathbb {P} \big (M (D) \in E \big) \leq e ^ {\varepsilon} \mathbb {P} \big (M (D ^ {\prime}) \in E \big) + \delta .
$$

When $\delta = 0$ , the mechanism M is said to satisfy ε-diferential privacy.

DP protects individuals by requiring that an algorithm produce similar output distributions on adjacent datasets that difer in a single record. This similarity limits the influence of any one data point on what is released, so an observer cannot reliably infer whether that record was included. The $( \varepsilon , \delta )$ definition measures similarity through a log-likelihood ratio control between the two output distributions for every measurable event. The parameter $\varepsilon$ bounds the magnitude of this log-likelihood ratio, while $\delta$ allows a small probability mass on which the bound may be violated, which can be viewed as an $\varepsilon { \mathrm { - D P } }$ guarantee holding with probability at least $1 - \delta$ Smaller values of $\varepsilon$ and $\delta$ therefore correspond to stronger privacy protection, and $\delta$ is typically chosen to be extremely small in practice, since it quantifies the probability of a rare failure event the privacy protection may not hold.

A key appeal of DP is that its guarantees are accompanied by explicit parameters that can be tracked across an entire pipeline, which makes it possible to quantify the overall level of protection. In particular, $\mathrm { D P }$ enjoys the following properties:

• Post-processing: If M is $( \varepsilon , \delta ) – \mathrm { D P } ,$ , then for any mapping Proc independent of the data, the post-processed mechanism Proc ◦ M is also $( \varepsilon , \delta ) – \mathrm { D P }$

• Sequential composition (basic): Let $M _ { 1 } : \mathcal { X }  \mathcal { Y } _ { 1 }$ and $M _ { i } : \mathcal { X } \times \mathcal { Y } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { Y } _ { i - 1 } \to \mathcal { Y } _ { i }$ for $i = 2 , \ldots , k$ . Let $M ( D ) : = \left( M _ { 1 } ( D ) , M _ { 2 } ( D , y _ { 1 } ) , \ldots , M _ { k } ( D , y _ { 1 } , \ldots , y _ { k - 1 } ) \right)$ denote the joint mechanism, where the outputs are generated recursively. If, for every fixed previous output history, each step is $( \varepsilon _ { i } , \delta _ { i } ) – \mathrm { D P }$ , then M is $\begin{array} { r l } { \Bigl ( \sum _ { i = 1 } ^ { k } \varepsilon _ { i } , ~ \sum _ { i = 1 } ^ { k } \delta _ { i } \Bigr ) \ – \mathrm { D P } } \end{array}$

• Parallel composition: Let D be partitioned into disjoint subsets $D _ { 1 } , \ldots , D _ { k }$ , and let $M _ { i }$ be an $( \varepsilon _ { i } , \delta _ { i } ) – \mathrm { D P }$ mechanism applied only to $D _ { i }$ . Define the joint mechanism $M ( D ) : =$ $\left( M _ { 1 } ( D _ { 1 } ) , \ldots , M _ { k } ( D _ { k } ) \right)$ . Then M is $\Big ( \operatorname* { m a x } _ { 1 \leq i \leq k } \varepsilon _ { i }$ 2 $\mathrm { m a x } _ { 1 \leq i \leq k } \delta _ { i } \big ) - \mathrm { D P }$

We present the basic sequential composition rule only for illustration, and the main point is that privacy parameter accumulates across multiple data-dependent releases. In practice, one can obtain tighter accounting by using refined composition theorems and privacy definitions designed for sharper tracking of cumulative loss, such as Rényi DP or Gaussian DP. We do not pursue these refinements here since our focus is on the pipeline-level design and on policy-quality guarantees, rather than on the tightest possible accounting constants.

Constructing a DP mechanism typically involves injecting noise, calibrated to the sensitivity, the maximum impact of a single data point on the output. In machine learning where gradientbased optimization is standard, the contribution of an individual data is its respective gradient. Therefore, a natural pathway to DP is to inject noise into the gradient updates, leading to the DP-stochastic gradient descent (DP-SGD) (Abadi et al., 2016).

Concretely, let $\ell ( \theta ; z )$ denote a per-example loss function and $\Theta \subset \mathbb { R } ^ { d }$ be a parameter space. A standard minibatch DP-SGD update is defined as:

$$
\theta_ {t + 1} = \Pi_ {\Theta} \left(\theta_ {t} - \eta_ {t} \left(\frac {1}{| B _ {t} |} \sum_ {i \in B _ {t}} \operatorname{clip} \left(\nabla_ {\theta} \ell \left(\theta_ {t}; Z _ {i}\right), C\right) + \xi_ {t}\right)\right), \quad \xi_ {t} \sim \mathcal {N} \left(0, \sigma_ {\mathrm{DP}} ^ {2} C ^ {2} I _ {d}\right),\tag{6}
$$

where $B _ { t } \subset [ n ]$ is a minibatch, $\Pi _ { \Theta }$ denotes the projection onto the set Θ, and $\mathrm { c l i p } ( g , C ) : = g$ min $\{ 1 , C / \| g \| _ { 2 } \}$ scales each gradient to ensure a uniform $\ell _ { 2 } .$ -sensitivity bound C. While sensitivity may not be inherently bounded or analytically tractable–particularly for black-box models– this gradient clipping ensures that the sensitivity is always bounded by $C ,$ which is then used to calibrate the noise. As such, gradient clipping serves as a key tool that provides the flexibility to satisfy DP for arbitrary diferentiable architectures.

Other optimization approaches are primarily distinguished by the stage at which noise is introduced, such as objective perturbation or output perturbation (Chaudhuri et al., 2011). For large-scale tasks, however, DP-SGD remains the de facto standard and the most widely used scalable paradigm in practice, supported by implementations such as the Opacus library in Python (Yousefpour et al., 2021).

## 3 Proposed Method

## 3.1 Motivation: Challenges in Private Policy Optimization

The prevailing paradigms in finetuning, such as DPO and PPO, optimize the policy parameters $\pi _ { \theta }$ directly. We defer a more detailed description of PPO to Appendix B.1, and focus here on the common issue that arises when DP is imposed on policy optimization.

The fundamental dificulty stems from the inherent incompatibility between the unbounded nature of policy gradients and the sensitivity required by DP. In frameworks like PPO or DPO, the gradient involves the score function

$$
\nabla_ {\theta} \log \pi_ {\theta} (a | x) = \frac {\nabla \pi_ {\theta} (a | x)}{\pi_ {\theta} (a | x)},
$$

which lacks a uniform upper bound since $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { x } )$ can be arbitrarily close to zero, making the ratio arbitrarily large. As these policy gradients can be arbitrarily large, the aforementioned clipping during the gradient updates are necessary.

Example 4 (DPO). Consider the DPO objective in (4). For a single preference pair $( x , a ^ { w } , a ^ { l } )$

define

$$
z _ {\theta} (x, a ^ {w}, a ^ {l}) := \frac {1}{\eta} \log \frac {\pi_ {\theta} (a ^ {w} | x)}{\pi_ {0} (a ^ {w} | x)} - \frac {1}{\eta} \log \frac {\pi_ {\theta} (a ^ {l} | x)}{\pi_ {0} (a ^ {l} | x)}.
$$

The per-pair loss is $\ell ( \theta ) = - \log { \sigma ( z _ { \theta } ( x , a ^ { w } , a ^ { l } ) ) }$ . Since $\pi _ { 0 }$ is fixed, the gradient is

$$
\nabla_ {\theta} \ell (\theta) = - \big (1 - \sigma (z _ {\theta}) \big) \nabla_ {\theta} z _ {\theta} = - \frac {1}{\eta} \big (1 - \sigma (z _ {\theta}) \big) \Big (\nabla_ {\theta} \log \pi_ {\theta} (a ^ {w} \mid x) - \nabla_ {\theta} \log \pi_ {\theta} (a ^ {l} \mid x) \Big).
$$

This expression depends on the score function $\nabla _ { \theta } \log \pi _ { \theta } ( a \mid x )$ , which generally admits no uniform bound because $\pi _ { \theta } ( \boldsymbol { a } \mid \boldsymbol { x } )$ can be arbitrarily small. As a result, per-example gradients can be arbitrarily large under standard policy parameterizations, and $D P$ training therefore requires explicit sensitivity control through per-example gradient clipping.

Example 5 (PPO-style policy optimization). A similar issue arises for PPO-style policy optimization. At a high level, PPO uses an advantage signal to indicate whether a sampled action should become more likely or less likely under the updated policy, while constraining the update so that the policy does not move too far in a single step. Here the advantage A is a scalar quantity summarizing how favorable action a is at context x relative to a baseline. The PPO-style ratio

$$
\rho_ {\theta} (x, a) := \frac {\pi_ {\theta} (a \mid x)}{\pi_ {\mathrm{ref}} (a \mid x)} = \exp \Bigl (\log \pi_ {\theta} (a \mid x) - \log \pi_ {\mathrm{ref}} (a \mid x) \Bigr)
$$

measures how much the updated policy reweights action a relative to a fixed reference policy $\pi _ { \mathrm { r e f } }$ Given a parameter $\varepsilon _ { \mathrm { c l i p } } > 0$ , the clipped surrogate takes the form

$$
\ell_ {\mathrm{PPO}} (\theta) = - \min \Bigl \{\rho_ {\theta} (x, a) A, \operatorname{clip} (\rho_ {\theta} (x, a), 1 - \varepsilon_ {\mathrm{clip}}, 1 + \varepsilon_ {\mathrm{clip}}) A \Bigr \}.
$$

Whenever the unclipped branch is active, diferentiating with respect to θ yields

$$
\nabla_ {\theta} \rho_ {\theta} (x, a) = \rho_ {\theta} (x, a) \nabla_ {\theta} \log \pi_ {\theta} (a \mid x),
$$

so the gradient again depends on the score function $\nabla _ { \theta } \log \pi _ { \theta } ( \boldsymbol { a } \mid \boldsymbol { x } )$ . Under standard policy parameterizations, this score function generally admits no uniform bound because $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { x } )$ can be arbitrarily small. Thus, PPO clipping controls the objective through the policy ratio, but it does not remove the need for explicit per-example gradient clipping when DP is imposed.

Clipping plays two roles in private training. It enforces a sensitivity bound, but it can also distort optimization when many per-example gradients exceed the clipping threshold. Let $F ( \theta ) = \mathbb { E } _ { z } [ \ell ( \theta ; z ) ]$ and define the minibatch gradient $\widehat { \nabla } F _ { t } ( \theta )$ . Under standard sampling assumptions, $\widehat { \nabla } F _ { t } ( \theta )$ is an unbiased estimator of $\nabla F ( \theta )$ , whereas replacing per-example gradients by $\mathrm { c l i p } ( \nabla _ { \boldsymbol { \theta } } \boldsymbol { \ell } ( \boldsymbol { \theta } ; \boldsymbol { z } ) , \boldsymbol { C } )$ generally introduces bias.

This distinction is especially relevant in policy optimization, where per-example gradients can be heavy-tailed due to score-function terms, so clipping may be frequently active. In contrast, when the learning target admits a uniform per-example gradient bound, one can choose $C$ to match this bound so that clipping is rarely active and introduces negligible distortion. In that case, privacy is enforced primarily through additive noise calibrated to the sensitivity bound, which preserves mean-zero updates while inflating variance.

To address these challenges, we place the privacy mechanism on reward learning rather than on policy optimization. The reward-modeling objective is typically better conditioned, and in many LLM pipelines it is implemented by freezing a pretrained backbone and training a lightweight head on top of its representations (Evci et al., 2022; Houlsby et al., 2019; Hu et al., 2022). Let $\boldsymbol { \phi } ( \boldsymbol { x } , a ) \in \mathbb { R } ^ { d }$ denote the fixed representation of a context action pair, and consider a linear-head reward model

$$
r _ {\theta} (x, a) = \langle \phi (x, a), \theta \rangle ,\tag{7}
$$

with $\theta \in \Theta = \{ \theta \in \mathbb { R } ^ { d } \mid \| \theta \| _ { 2 } \leq R \}$ . The key point is that sensitivity control becomes transparent once the per-example loss is Lipschitz in θ. A convenient way to ensure this is to keep the representation norm bounded, for instance by applying a final normalization step on the backbone features. In modern Transformers, LayerNorm and RMSNorm already stabilize feature scales in practice (Ba et al., 2016; Zhang and Sennrich, 2019; Zheng et al., 2024), and an explicit final projection or normalization can enforce a deterministic bound su $) _ { x , a } \| \phi ( x , a ) \| _ { 2 } \leq L$

Under the Bradley–Terry model, each observation $z = ( x , a ^ { w } , a ^ { \ell } )$ induces a convex loss $\ell ( \theta ; z )$ whose gradient has the form

$$
\nabla_ {\theta} \ell (\theta ; z) = \alpha (\theta ; z) \bigl (\phi (x, a ^ {w}) - \phi (x, a ^ {\ell}) \bigr), \quad \alpha (\theta ; z) \in [ 0, 1 ].\tag{8}
$$

Therefore, we have

$$
\| \nabla_ {\theta} \ell (\theta ; z) \| _ {2} \leq \| \phi (x, a ^ {w}) - \phi (x, a ^ {\ell}) \| _ {2} \leq \| \phi (x, a ^ {w}) \| _ {2} + \| \phi (x, a ^ {\ell}) \| _ {2} \leq 2 L.\tag{9}
$$

This yields a uniform per-example gradient bound, which provides a clean sensitivity control for private optimization. This suggests that per-example gradients in head-only reward learning are typically better behaved than in policy optimization. As a result, when the same per-example gradient clipping norm C is used to control DP sensitivity, clipping tends to be less frequently active in reward learning than in policy optimization. In that sense, privacy in reward learning is driven more by calibrated noise than by clipping-induced distortion.

## 3.2 Proposed Framework: Private Reward-Based Alignment

We propose a decoupled framework that concentrates the privacy expenditure solely on estimating the reward structure and derives the final decision rule via post-processing.

As outlined in Algorithm 1, the procedure consists of two stages: (i) learning a diferentially private reward model r˜ from the full preference dataset D, and (ii) producing an aligned action/response induced by $\tilde { r }$ without any further access to D.

In the first stage, we treat reward learning as a single empirical risk minimization problem on D. We state the framework at the level of the resulting $( \varepsilon , \delta ) – \mathrm { D P }$ guarantee, rather than specifying an explicit closed-form calibration for the added Gaussian noise. In practice, the calibration of DP-SGD depends on various factors, including the clipping norm, sampling scheme, number of epochs, and privacy accounting method (Abadi et al., 2016; Bu et al., 2020). Since our focus is on the pipeline design and on the privacy–utility trade-of at the level of the final guarantee, rather than on accountant-specific calibration formulas, we do not make the noise level explicit here. In the theoretical development in Section 4, we work with the projected noisy stochasticgradient procedure of Bassily et al. (2014) as a concrete DP-SGD instantiation, since it delivers the strongly-convex excess-risk rate used in our analysis. In the experiments, the corresponding calibration is handled by Opacus, which takes the target privacy parameters together with the training configuration and internally performs privacy accounting to determine the required noise level; implementation details are deferred to Appendix B.4 and Table 3.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Differentially Private Reward-Based Alignment
1: Input: Preference dataset $\mathcal{D} = \{(x_i, a_i^w, a_i^l)\}_{i=1}^n$, privacy budget ($\varepsilon, \delta$), KL-regularization parameter $\eta$, reference policy $\pi_0$, sampling budget $N$ (optional).
2: Step 1: Private Reward Learning
3: Learn a private reward model $\tilde{r}(x, a)$ by an ($\varepsilon, \delta$)-DP mechanism (e.g., DP-SGD).
4: Step 2: Policy Derivation (Inference)
5: if normalization constant $Z_{\tilde{r}}(x) = \mathbb{E}_{a \sim \pi_0}[\exp(\eta \tilde{r}(x, a))]$ is computable then
6: Exact Inference
7: Construct the optimal policy in closed form:
$\pi_{\tilde{r}}^\eta(a|x) = \frac{1}{Z_{\tilde{r}}(x)}\pi_0(a|x)\exp(\eta \tilde{r}(x, a))$.

8: else
9: Approximate Inference via Best-of-$N$
10: For a given context $x$, sample $N$ candidates from reference: $\{a^{(1)}, \ldots, a^{(N)}\} \sim \pi_0(\cdot|x)$.
11: Select the candidate maximizing the private reward:
$a^* = \underset{j \in \{1, \ldots, N\}}{\text{argmax}}\tilde{r}(x, a^{(j)})$.
12: Define the policy output as the Dirac mass on $a^*$.
13: end if
14: Output: Privately aligned policy $\pi_{\tilde{r}}^\eta$ or action $a^*$.
</div>

In the second stage, the KL-regularized target policy induced by r˜ is defined by the Gibbs form

$$
\pi_ {\tilde {r}} ^ {\eta} (a \mid x) = \frac {\pi_ {0} (a \mid x) \exp \{\eta \tilde {r} (x , a) \}}{Z _ {\tilde {r}} ^ {\eta} (x)},
$$

where $\begin{array} { r } { Z _ { \tilde { r } } ^ { \eta } ( x ) : = \sum _ { a ^ { \prime } \in \mathcal { A } } \pi _ { 0 } ( a ^ { \prime } \mid x ) \exp \{ \eta \tilde { r } ( x , a ^ { \prime } ) \} } \end{array}$ . The implementation depends on the tractability of the partition function $Z _ { \tilde { r } } ^ { \eta } ( x )$ . When $Z _ { \tilde { r } } ^ { \eta } ( x )$ is tractable (e.g., finite action sets or structured settings where normalization is feasible), one can explicitly construct and sample from $\pi _ { \widetilde { r } } ^ { \eta } ( \cdot \mid x )$ (or apply a deterministic rule such as arg max<sub>a</sub> $\pi _ { \tilde { r } } ^ { \eta } ( a \mid x ) )$ .

In contrast, when A is combinatorially large–as in preference fine-tuning for LLMs where actions correspond to long token sequences–computing $Z _ { \tilde { r } } ^ { \eta } ( x )$ is infeasible. In this regime, Algorithm 1 adopts a best-of-N (BoN) inference-time policy (Stiennon et al., 2020): draw N candidates $a ^ { ( 1 ) } , \dots , a ^ { ( N ) } \sim \pi _ { 0 } ( \cdot \mid x )$ and output

$$
a ^ {*} \in \arg \max _ {j \in \{1, \dots , N \}} \tilde {r} (x, a ^ {(j)}).
$$

As N increases, this selection increasingly tends to return higher-reward candidates under the proposal distribution $\pi _ { 0 } ( \cdot \mid x )$ by restricting attention to a richer candidate pool, without requiring explicit normalization over ${ \mathcal { A } } .$

A simple rationale for using π<sub>0</sub> as a proposal distribution follows from the relationship between the KL-regularized target policy and the reference. Consider $\pi _ { \eta r } ( a | x ) \propto \pi _ { 0 } ( a | x ) \exp \{ \eta r ( x , a ) \}$ If the reward is uniformly bounded, i.e., $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } , a \in \mathcal { A } } | r ( x , a ) | \le B } \end{array}$ for some $B < \infty$ , then for every $( x , a )$ 2

$$
e ^ {- \eta B} \leq \frac {\pi_ {\eta r} (a | x)}{\pi_ {0} (a | x)} \cdot Z _ {\eta r} (x) \leq e ^ {\eta B},
$$

with $\begin{array} { r } { Z _ { \eta r } ( x ) : = \sum _ { a ^ { \prime } \in \mathcal { A } } \pi _ { 0 } ( a ^ { \prime } | x ) \exp \{ \eta r ( x , a ^ { \prime } ) \} } \end{array}$ . Since $e ^ { - \eta B } \leq Z _ { \eta r } ( x ) \leq e ^ { \eta B }$ , we obtain the pointwise bounds

$$
e ^ {- 2 \eta B} \leq \frac {\pi_ {\eta r} (a | x)}{\pi_ {0} (a | x)} \leq e ^ {2 \eta B}.
$$

In particular, when $\eta B$ is moderate, $\pi _ { \eta r } ( \cdot | x )$ remains within a controlled multiplicative tilt of $\pi _ { 0 } ( \cdot | x )$ , which supports using $\pi _ { 0 } ( \cdot | x )$ as a reasonable proposal distribution for candidate-based approximate inference.

We highlight that Algorithm 1 is designed to exploit a structural feature specific to RLHF. In standard RL, a separate reward-learning is absent because rewards are observed directly from the environment. As a result, if one seeks to enforce $\mathrm { D P }$ in standard RL, privacy must be introduced at the policy-optimization stage, typically through noisy policy gradient updates (He and Zhou, 2025). RLHF is diferent in that it introduces an intermediate reward-learning layer. Our framework places DP on this layer alone and derives the final policy by post-processing the resulting private reward model. This design is especially appealing because, as discussed in Section 3.1, private reward learning is less afected by clipping than policy optimization.

Moreover, the role of (3) difers between DPO and our framework. Both use the analytical form of the KL-regularized optimizer, but DPO uses it for direct policy optimization, whereas our framework uses it to derive the final policy as a post-processing of the private reward model. This distinction is consequential under DP. Direct policy optimization requires noisy updates over policy parameters and can in some cases yield a policy worse than the reference policy. In contrast, our framework places privacy only on reward learning, so the reward-estimation layer serves as a bufer between the privacy mechanism and the final policy. A similar issue arises in RLHF pipelines that privatize both reward learning and policy optimization (Wu et al., 2024, 2025), since using the same preference data in both stages requires either data splitting or privacy-budget splitting. Our framework avoids this extra cost by using the full dataset for a single private reward-estimation step and deriving the final policy by post-processing.

Once the private reward model r˜ is learned, any subsequent output construction, whether through the exact policy or the BoN rule, depends on the dataset only through r˜. This leads to the following privacy guarantee for the entire pipeline.

Proposition 6 (Privacy of the Framework). Suppose the reward-learning mechanism M that outputs a reward model $\tilde { r } = \mathcal { M } ( \mathcal { D } )$ satisfies $( \varepsilon , \delta ) \mathopen { } \mathclose \bgroup \left. - D P \aftergroup \egroup \right.$ . Then Algorithm 1 also $( \varepsilon , \delta ) \ – D P$

Proof. This follows directly from the post-processing property of DP: composing an $( \varepsilon , \delta ) – \mathrm { D P }$ mechanism with any data-independent mapping (including additional randomness independent of D) preserves the same $( \varepsilon , \delta ) – \mathrm { D P }$ guarantee. □

## 4 Theoretical Analysis

To present a series of theoretical results, we begin by providing essential assumptions.

Assumption 1 (i.i.d. preference data). The contexts $x _ { 1 } , \ldots , x _ { n }$ are i.i.d. draws from $d _ { 0 }$ . For each i, two candidate actions are drawn independently from the reference policy $\pi _ { 0 } ( \cdot \mid x _ { i } )$ , and $( a _ { i } ^ { w } , a _ { i } ^ { l } )$ is obtained according to the Bradley–Terry model in Definition 1.

Assumption 2 (Linear reward realizability). There exists a parameter $\theta ^ { * } \in \Theta \subset \mathbb { R } ^ { d }$ such that, for all $( x , a ) \in \mathcal { X } \times \mathcal { A } , r ^ { * } ( x , a ) = r _ { \theta ^ { * } } ( x , a ) = \langle \phi ( x , a ) , \theta ^ { * } \rangle$ . Moreover, the representation is uniformly bounded: su $\operatorname { p } _ { x \in { \mathcal { X } } , a \in { \mathcal { A } } } \| \phi ( x , a ) \| _ { 2 } \leq L$

Assumption 3 (Non-degeneracy feature). Define $\Delta \phi ( x ; a , a ^ { \prime } ) : = \phi ( x , a ) - \phi ( x , a ^ { \prime } )$ . Then the smallest eigenvalue of the matrix $\mathbb { E } _ { x \sim d _ { 0 } , ~ a , a ^ { \prime } \sim \pi _ { 0 } ( \cdot | x ) } \big [ \Delta \phi ( x ; a , a ^ { \prime } ) \Delta \phi ( x ; a , a ^ { \prime } ) ^ { \top } \big ]$ is $\lambda > 0$

Assumption 4 (Coverage). There exists a constant C such that for any $\pi \in \Pi$

$$
\max _ {x, a: d _ {0} (x) > 0} \frac {\pi (a | x)}{\pi_ {0} (a | x)} \leq C,
$$

with convention that $\begin{array} { r } { \frac { 0 } { 0 } = 0 } \end{array}$

Assumption 1 specifies the basic ofline data-collection model used in our analysis. It provides the independence structure needed for the statistical arguments and is consistent with a common RLHF pipeline in which prompts are sampled, candidate responses are generated from a fixed reference policy, and human pairwise preferences are then collected (Ouyang et al., 2022; Bai et al., 2022).

Assumption 2 posits that the preference signal is captured by a linear reward model on a fixed representation. This type of realizability assumption is standard (Zhu et al., 2023; Liu et al., 2025b). The boundedness condition on $\phi ( x , a )$ is a regularity requirement that ensures the reward class is well behaved and supports clean sensitivity and concentration arguments. This assumption also aligns with common practice in LLM alignment and reward modeling. A widely used implementation freezes a pretrained backbone and learns a lightweight head on top of its representations, which is a parameter eficient way to fit preference data while limiting overfitting and training instability (Evci et al., 2022; Houlsby et al., 2019; Zaken et al., 2022). Under this head only design, linear reward modeling becomes a natural approximation that connects the practical pipeline to a tractable theory.

Assumption 3 requires that the Fisher-information-type matrix E $\left\lceil \Delta \phi ( x ; a , a ^ { \prime } ) \Delta \phi ( x ; a , a ^ { \prime } ) ^ { \top } \right\rceil$ is uniformly positive definite, ruling out degenerate feature maps that narrows only in a lowerdimensional subspace. This condition also excludes the usual additive-baseline non-identifiability in Bradley–Terry models, since any direction that leaves all pairwise diferences unchanged would lie in the null space of this matrix. Similar non-degeneracy conditions are standard in theoretical analyses (Zhu et al., 2023; Zhong et al., 2024; Liu et al., 2025b).

Assumption 4 imposes a uniform bound on the density ratio between any candidate policy $\pi \in \Pi$ and the reference policy $\pi _ { 0 }$ . This condition ensures that the reference policy $\pi _ { 0 }$ provides suficient coverage over the state-action pairs potentially visited by the target policy class, guaranteeing that no candidate policy places significant mass on actions rarely sampled by $\pi _ { 0 }$ . This type of uniform coverage is standard in the theoretical analysis of RLHF and ofline RL (Munos and Szepesvári, 2008; Xiong et al., 2024; Song et al., 2024). For modern LLMs, this assumption is practically well-motivated. Since RLHF typically performs a refinement of a strong SFT reference $\pi _ { 0 }$ rather than learning entirely new capabilities, it is natural to restrict attention to a policy class Π that stays within a controlled neighborhood of $\pi _ { 0 }$

## 4.1 Upper Bound on the Suboptimality Gap

We first deliver the utility analysis of our private estimator. The following lemma, adapted from Bassily et al. (2014), characterizes the expected excess empirical risk.

Lemma 7 (Utility of private projected SGD). Suppose Assumption 1, 2 and Assumption 3 hold. Let ${ \tilde { \theta } } _ { n }$ be the output of DP-SGD procedure of Bassily et al. $( { \it 2 0 1 4 } ) ,$ ; that is, at each iteration, one data point is sampled uniformly with replacement, Gaussian noise is added to the resulting stochastic gradient, and the update is projected back onto $\Theta$ , where the Gaussian noise is calibrated to satisfy $( \varepsilon , \delta ) \ – D P$ under the per-example gradient bound 2L. Fix any $\rho \in ( 0 , 1 )$ and let $\begin{array} { l l l } { n } & { \geq } & { { \frac { 3 2 L ^ { 2 } } { \lambda } } \log \left( { \frac { d } { \rho } } \right) } \end{array}$ . Then there exists an event $\mathcal { E }$ with $\mathbb { P } ( \mathcal { E } ) \geq 1 - \rho$ such that, on $\mathcal { E } _ { : }$ , the negative log-likelihood is µ-strongly convex over Θ with $\mu ~ = ~ \frac { \lambda } { 2 } \sigma ( 2 R L ) ( 1 - \sigma ( 2 R L ) )$ , where $\sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ is the sigmoid function. Moreover, on the same event $\mathcal { E }$ ,

$$
\mathbb {E} \left[ \bar {L} _ {n} (\tilde {\theta} _ {n}) - \bar {L} _ {n} (\hat {\theta} _ {n}) \mid D \right] = \tilde {O} \left(\frac {d}{n ^ {2} \varepsilon^ {2}}\right),
$$

where the expectation is over the algorithmic randomness conditional on D.

Proof sketch. Assumption 3 implies the population negative log-likelihood is uniformly strongly convex. A matrix concentration bound then shows that the empirical loss inherits the same strong convexity on an event $\mathcal { E }$ with $\mathbb { P } ( \mathcal { E } ) \geq 1 - \rho .$ Conditioning on $\mathcal { E } .$ , we apply the standard DP-SGD utility guarantee for Lipschitz and strongly convex objectives to obtain the stated conditional expected excess empirical risk bound. □

Lemma 7 provides a utility bound for our private reward estimator by certifying, with high probability, that the empirical objective is strongly convex over Θ. Various variants of DP-SGD are available, and for the privacy guarantee itself our framework is compatible with any such instantiation that attains the target $( \varepsilon , \delta ) – \mathrm { D P }$ level. For the theoretical analysis, however, we invoke the strongly-convex private stochastic-gradient guarantee of Bassily et al. (2014), which yields the optimal excess-risk order in the strongly-convex regime. In our linear-head setting, the per-example gradient is uniformly bounded by 2L, so the same type of guarantee applies after calibrating the noise to the target $( \varepsilon , \delta ) – \mathrm { D P }$ level. This allows us to quantify the privacyinduced optimization error at the sharpest available rate without postulating strong convexity as a separate assumption, and we therefore state the lemma in terms of the resulting privacy guarantee and excess empirical risk rate rather than in terms of an accountant-specific calibration formula.

Remark 8 (Unconditional DP-SGD control). To characterize the suboptimality gap, we require a high-probability control of the estimation error with respect to the joint randomness of the data and the learning procedure. Much of the existing DP-SGD literature analyzes the procedure under a fixed dataset $( o r$ in expectation), which does not directly provide the unconditional statement needed to our analysis. Section D.2 in the supplementary provides the missing bridge by proving an unconditional result tailored to our setup (Theorem $\mathit { 1 1 ) }$

We now state our main upper bound on the suboptimality gap.

Theorem 9 (Upper bound on the suboptimality gap). Suppose Assumptions $\begin{array} { r } { 1 , \ 2 , \ 3 , } \end{array}$ and 4 hold. Consider Algorithm 1 instantiated with the private projected SGD procedure in Lemma $^ { 7 , }$ and let $\tilde { \theta }$ be its output. Let $\pi _ { \tilde { \theta } } ^ { \eta }$ denote the induced KL-regularized policy. Fix any $\rho \in ( 0 , 1 )$ . If $\begin{array} { r } { n \ge \frac { 3 2 L ^ { 2 } } { \lambda } \log \left( \frac { d } { \rho } \right) } \end{array}$ , then with probability at least $1 - \rho ,$

$$
\Delta_ {\eta} \left(\pi_ {\tilde {\theta}} ^ {\eta}\right) \leq \tilde {O} \left(\frac {\eta d}{n} + \frac {\eta d}{n ^ {2} \varepsilon^ {2}}\right).
$$

Theorem 9 provides an additive decomposition of the suboptimality gap into a non-private term of order $\eta d / n$ and a privacy cost of order $\eta d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ . The parameter δ which characterizes a failure probability of the privacy protection is typically chosen to be negligible, e.g., $\delta = n ^ { - k }$ for some $k \geq 2 ;$ under such choices, its impact enters only through logarithmic factors and does not afect the leading rates. Notably, privacy does not worsen the dimensional dependence, so the price of privacy appears only through the extra $1 / ( n \varepsilon ^ { 2 } )$ factor.

To interpret the sample-size regimes, define the crossover scale $n _ { \varepsilon } : = \varepsilon ^ { - 2 }$ . When $n \gtrsim n _ { \varepsilon }$ the privacy-induced term is lower order and the rate efectively matches the non-private one; when $n \lesssim n _ { \varepsilon }$ , the privacy cost can dominate and determines the attainable gap. This captures the privacy–utility tradeof: holding $\varepsilon$ fixed recovers the non-private rate as n grows, while strengthening privacy by shrinking $\varepsilon$ increases the privacy-induced term. Importantly, the faster decay $1 / n ^ { 2 }$ in the privacy term arises from invoking strongly-convex DP optimization guarantees on a high-probability event; without such curvature, privacy costs would typically decay only as $1 / n$ rather than $1 / n ^ { 2 }$

The bound is linear in the KL regularization parameter η, which we treat as fixed throughout $\eta ,$ the analysis, as is standard in KL-regularized RLHF and related formulations (Ouyang et al., 2022; Rafailov et al., 2024; Zhao et al., 2024). This linear dependence is natural, since smaller η pulls both $\pi _ { \tilde { \theta } } ^ { \eta }$ and $\pi _ { \theta ^ { * } } ^ { \eta }$ closer to the reference policy $\pi _ { 0 }$ , thereby reducing the $\mathrm { g a p }$ between the two induced policies.

## 4.2 Minimax Lower Bound on the Suboptimality Gap

We next establish a minimax lower bound on the suboptimality gap for the d-dimensional linear reward model class, which applies to any $( \varepsilon , \delta ) – \mathrm { D P }$ algorithm.

Theorem 10 (Minimax lower bound). Fix $\eta > 0$ and consider the d-dimensional linear reward model class. For $\varepsilon \in ( 0 , 1 ]$ and $\delta \leq \varepsilon$ , define the minimax risk

$$
R _ {n} (\varepsilon , \delta) := \inf _ {A \in \mathcal {A} _ {\varepsilon , \delta}} \sup _ {\theta^ {*} \in \Theta} \mathbb {E} _ {\theta^ {*}} [ \Delta_ {\eta} (A (Z ^ {n})) ],
$$

where $\mathcal { A } _ { \varepsilon , \delta }$ is the class $o f \left( \varepsilon , \delta \right) - D P$ algorithms and $Z ^ { n }$ denotes n preference pairs generated under $\theta ^ { * }$

Then, for each fixed $\eta > 0 ,$ , there exist constants $c _ { \eta } , C _ { \eta } > 0$ such that, up to logarithmic factors, for all $n \geq n _ { \mathrm { N P } } : = C _ { \eta } d _ { \mathrm { \Omega } }$

$$
R _ {n} (\varepsilon , \delta) \geq c _ {\eta} \max \left\{\frac {d}{n}, \min \left(\frac {1}{n \varepsilon}, \frac {d}{n ^ {2} \varepsilon^ {2}}\right) \right\}.
$$

Moreover, letting $n _ { \mathrm { P } } : = C _ { \eta } d / \varepsilon$ , for all $n \geq \operatorname* { m a x } \{ n _ { \mathrm { N P } } , n _ { \mathrm { P } } \}$

$$
R _ {n} (\varepsilon , \delta) \geq c _ {\eta} \max \left\{\frac {d}{n}, \frac {d}{n ^ {2} \varepsilon^ {2}} \right\}.
$$

Proof sketch. We construct two preference-learning instances that coincide on all but a single informative context, where the optimal action difers. Any algorithm that achieves a small sub optimality gap must behave diferently on this informative context, which allows us to view the algorithm’s output policy as implicitly inducing a hypothesis test between the two instances. DP then creates an additional bottleneck. Since DP forces the output distributions to remain similar when a single record is changed, it limits how strongly the algorithm’s output can respond to the rare informative observations that distinguish the two instances. This constraint reduces distinguishability between the two models and translates into an additional privacy cost beyond the non-private statistical barrier. □

Theorem 10 gives an information-theoretic limitation that holds uniformly over all $( \varepsilon , \delta ) – \mathrm { D P }$ algorithms in the d-dimensional linear reward class. Unlike the upper bound in Theorem 9, where η is kept explicit because it directly controls how conservatively the induced policy departs from $\pi _ { 0 } .$ , the minimax lower bound in Theorem 10 treats η as fixed and suppresses its dependence in the stated rate. Our main goal here is to isolate the phase transition in $( n , d , \varepsilon )$ , namely the non-private term $d / n$ , the privacy-dominated term $d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ , and the pre-asymptotic branch $1 / ( n \varepsilon )$ . While a more refined proof-level expression can retain $\eta ,$ , doing so does not change this regime structure and would only make the theorem statement heavier. For this reason, we keep η explicit in the upper bound, where it aids interpretation, but suppress it in the lower bound, where the main message is the $( n , d , \varepsilon )$ -dependent scaling.

The first term, $d / n ,$ is the nonprivate minimax lower bound established in Zhao et al. (2024). Since the class of $( \varepsilon , \delta ) – \mathrm { D P }$ algorithms is a subset of all algorithms considered there, this $\Omega ( d / n )$ term necessarily persists under privacy. The remaining terms quantify the additional loss due to privacy.

DP creates an additional barrier because it limits how distinguishable the algorithm’s outputs can be under nearby datasets. In our two-point construction, the two models difer only through a rare informative context. Without privacy, the dificulty is driven by statistical scarcity and yields the non-private $d / n$ term. With privacy, even when the informative samples appear, the output policy cannot react too sharply to them, since DP forces the output distributions to remain close when a single preference pair is diferent. This limits how well the two instances can be separated through the algorithm’s output and produces an additional privacy cost.

The privacy-dependent term splits into two regimes. The hard instance induces a gap of order log cosh $. ( \eta c / 2 )$ , where c is the reward signal at the informative context. DP constrains the efective signal that can be exploited while keeping the two induced output distributions hard to distinguish, which yields $c \lesssim d / ( n \varepsilon )$ . When n is large enough that $\eta c$ lies in the local quadratic region, log cosh(u) behaves like $u ^ { 2 }$ , and the resulting contribution scales as $d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ . When n is smaller and $\eta c$ falls outside this region, log $\cosh ( u )$ is closer to linear, which leads to the weaker $1 / ( n \varepsilon )$ rate.

Finally, we assume $\varepsilon \in ( 0 , 1 ]$ and $\delta \leq \varepsilon ,$ , which is exactly the condition used when applying Lemma 19, the DP Le Cam testing bound of Acharya et al. (2021), to keep the privacy-dependent testing error bounded away from zero. This condition is also natural from the privacy perspective, since $\delta$ represents the probability of a rare failure event in the privacy guarantee in the $( \varepsilon , \delta ) – \mathrm { D P }$ and is typically chosen to be very small. In particular, standard choices such as $\delta = n ^ { - k }$ with $k \geq 2$ satisfy $\delta \leq \varepsilon$ for all suficiently large n when ε is fixed. A comparison with Theorem 9 then shows that our upper bound matches this lower bound up to logarithmic factors in the regime identified below. We formalize this comparison in the next subsection.

## 4.3 Rate-optimality

We call an algorithm rate-optimal if its suboptimality gap matches the minimax risk up to logarithmic factors as a function of $( n , d , \varepsilon )$ . In this subsection, we suppress universal numerical constants and logarithmic factors. We denote the privacy scale by $n _ { \varepsilon } = \varepsilon ^ { - 2 }$

Theorem 9 yields the upper bound

$$
\Delta_ {\eta} \left(\pi_ {\tilde {\theta}} ^ {\eta}\right) \leq \widetilde {O} \left(\frac {d}{n} + \frac {d}{n ^ {2} \varepsilon^ {2}}\right).
$$

Theorem 10 yields the minimax lower bound

$$
R _ {n} (\varepsilon , \delta) \geq \widetilde {\Omega} \left(\max \left\{\frac {d}{n}, \min \left(\frac {1}{n \varepsilon}, \frac {d}{n ^ {2} \varepsilon^ {2}}\right) \right\}\right).
$$

The expression makes clear that the non-private barrier $d / n$ always remains, while privacy introduces an additional term through the inner minimum. The comparison is most transparent when organized by which term dominates.

• Privacy-negligible regime $( n \gtrsim n _ { \varepsilon } )$ . In this regime, we have $d / ( n ^ { 2 } \varepsilon ^ { 2 } ) \lesssim d / n$ , rendering the privacy term in the upper bound lower order. Since the lower bound always contains $d / n$ , both bounds are governed by the non-private rate $d / n$ . This confirms that privacy has a vanishing efect on the leading rate as n grows beyond the privacy scale.

• Privacy-dominated and rate-optimal regime $( d / \varepsilon \lesssim n \lesssim n _ { \varepsilon } )$ . When $n \lesssim n _ { \varepsilon }$ , the privacy term $d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ dominates the upper bound. In this range, the lower bound depends on the inner minimum. Specifically, when n is large enough such that

$$
\frac {d}{n ^ {2} \varepsilon^ {2}} \leq \frac {1}{n \varepsilon} \iff n \geq \frac {d}{\varepsilon},
$$

the minimum selects $d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ , and the lower bound simplifies to

$$
R _ {n} (\varepsilon , \delta) \geq \widetilde {\Omega} \bigg (\max \bigg \{\frac {d}{n}, \frac {d}{n ^ {2} \varepsilon^ {2}} \bigg \} \bigg).
$$

Since $n \lesssim n _ { \varepsilon }$ implies the privacy term is dominant, both bounds scale as $d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ . Thus,

Figure 3: Phase diagram of the statistical and privacy errors on a log-log scale. The plane is partitioned into three scaling regimes based on the asymptotic relationship between the sample size $n ,$ , the dimension $d ,$ and the privacy budget ε. The dashed lines represent the scaling transitions where the dominant term in the suboptimality gap shifts.

the algorithm is rate-optimal in this privacy-dominated regime.

• Pre-asymptotic regime $( n \lesssim d / \varepsilon )$ . When n is small enough that

$$
\frac {1}{n \varepsilon} \leq \frac {d}{n ^ {2} \varepsilon^ {2}} \iff n \leq \frac {d}{\varepsilon},
$$

the inner minimum in the lower bound becomes $1 / ( n \varepsilon )$ . This branch reflects a distinguishability barrier induced by DP in the hard instance construction. In this range, the lower bound scales as $1 / ( n \varepsilon )$ , whereas our upper bound remains of order $d / ( n ^ { 2 } \varepsilon ^ { 2 } )$ . We do not claim tightness of the upper bound in this regime.

Taken together, the bounds exhibit a transition at the privacy scale $n _ { \varepsilon }$ . Above $n _ { \varepsilon }$ , the leading rate is governed by statistical error $( d / n )$ . Below $n _ { \varepsilon }$ , privacy dominates, and the optimal decay becomes quadratic $( d / ( n ^ { 2 } \varepsilon ^ { 2 } ) )$ as long as the sample size is suficient to enter the local regime $( n \gtrsim d / \varepsilon )$

## 5 Numerical Studies

In this section, we evaluate our framework empirically. We first use controlled synthetic experiments to validate the theoretical results developed in Section 4 and to compare against private alignment baselines. We then study an LLM fine-tuning experiment to assess practical performance, while implementation details are deferred to Appendix B. Additional numerical results are reported in Appendix C, including reference-underperformance diagnostics in Appendix C.1, sensitivity to the DP-SGD clipping norm in Appendix C.2, and further scaling results with the feature dimension in Appendix C.3

(a) Efect of privacy budget ε

(b) Efect of dimension d
Figure 4: Convergence of Suboptimality Gap. The plots demonstrate the decay of the suboptimalit gap as a function of sample size n. (a) The gap decreases as the privacy budget ε increases, illustrating the privacy-utility trade-of. (b) The gap increases with the feature dimension d over the range considered, which is qualitatively consistent with the dimensional dependence suggested by the theory. Shaded regions indicate 95% confidence intervals over 30 trials.

## 5.1 Synthetic Data Analysis

For the data generation, we define the context dimension $p = \lceil d / 2 \rceil$ for each feature dimension $d \in \{ 3 , 5 , 7 , 9 \}$ and sample contexts independently from $x \sim \operatorname { U n i f } ( [ - 1 , 1 ] ^ { p } )$ . We construct the feature map $\phi ( x , a )$ using an interleaved structure of linear terms $x _ { j }$ and centered quadratic terms $q _ { j } ( x ) = x _ { j } ^ { 2 } - 1 / 3$ . Specifically, action-dependent signs $( u ( a ) , v ( a ) ) \in \{ \pm 1 \} ^ { 2 }$ are assigned to each action, and the feature vector is formed by truncating the sequence $[ u ( a ) x _ { 1 } , v ( a ) q _ { 1 } ( x ) , \ldots ]$ to length d. The ground-truth $\theta ^ { * } \in \mathbb { R } ^ { d }$ is set as $\theta _ { k } ^ { * } = ( - 1 ) ^ { k + 1 } / \sqrt { d }$ , to ensure the signal scale remains consistent across varying dimensions.

Regarding the privacy mechanism, we implement the DP-SGD algorithm via the Opacus library (Yousefpour et al., 2021). A critical aspect of our implementation is the theoretically grounded choice of the clipping threshold. Since the sensitivity is bounded by the sum of the norms of the two feature vectors, we calculate the deterministic upper bound $L ( d ) \ =$ $\operatorname* { s u p } _ { x , a } \| \phi ( x , a ) \| _ { 2 }$ and set the per-example clipping norm to $C = 2 L ( d )$ . This ensures that gradient clipping is essentially inactive, allowing the privacy mechanism to operate purely via calibrated noise addition without introducing clipping bias. Unless otherwise stated, we set $\delta = 1 0 ^ { - 5 }$ . All results are averaged over 30 independent trials.

## 5.1.1 Validation of Theoretical Results

We now investigate the convergence of the suboptimality gap with respect to the sample size $n ,$ with the goal of validating the theoretical result in Theorem 9. Figure 4 presents the resulting trends of the KL-regularized suboptimality gap.

Figure 4(a) varies the privacy budget ε while holding d = 5 fixed. As ε increases, the gap decreases across the full range of sample sizes, which is consistent with the privacy-dependent term in the theory. The separation between curves is most visible at smaller and moderate $n ,$ where privacy noise has a larger efect, and it narrows as n grows, reflecting the faster decay of the privacy contribution with sample size.

Figure 4(b) varies the feature dimension d while holding $\varepsilon = 1 . 0$ fixed. Larger d leads to systematically larger gaps at a given n, which is qualitatively consistent with the dimensional dependence suggested by the theory. The curves also decrease steadily with $n ,$ and the ordering across dimensions remains stable across the range considered, suggesting that the dominant dificulty is driven by statistical complexity rather than idiosyncratic optimization failures.

## 5.1.2 Comparison to Private Alignment Baselines

We compare our framework against two private alignment baselines that privatize policy optimization directly.

DP-DPO applies DP-SGD directly to the DPO objective, optimizing policy parameters under privacy constraints. DP-RLHF (DP-RM + DP-PPO-like) splits the preference dataset into two disjoint halves. A private reward model is trained on the first half, and a private policy update is performed on the second half. Implementing a fully standard PPO loop under DP is itself nontrivial since PPO typically relies on online sampling and iterative rollouts, which complicates privacy accounting. Existing work therefore adopts modified PPO-style procedures designed to make privacy accounting tractable (Wu et al., 2024). Following this perspective, we use an ofline pairwise PPO-like update that avoids actor–critic training and rollout collection. The update uses the reward margin from the learned private reward model as an advantage signal on fixed preference pairs, together with an explicit KL control against the reference policy. By disjointness, the two private stages can each use the full $( \varepsilon , \delta )$ budget via parallel composition. Further implementation details are deferred to Appendix B.1.

We report the suboptimality gap (top row in each figure) ${ V _ { \eta } ( \pi _ { \eta } ^ { \star } ) - V _ { \eta } ( \hat { \pi } ) }$ , and the corresponding normalized gap (bottom row)

$$
\frac {V _ {\eta} (\pi_ {\eta} ^ {\star}) - V _ {\eta} (\hat {\pi})}{V _ {\eta} (\pi_ {\eta} ^ {\star}) - V _ {\eta} (\pi_ {0})},
$$

where $V _ { \eta } ( \pi ) = \mathbb { E } [ r ^ { * } ( x , a ) ] - ( 1 / \eta ) \mathrm { K L } ( \pi ( \cdot \mid x ) \| \pi _ { 0 } ( \cdot \mid x ) )$ . The normalized gap facilitates comparisons across η by scaling by the maximum achievable improvement over $\pi _ { 0 }$ . Throughout this section, policy-quality metrics are evaluated on a shared Monte Carlo context set $( N _ { \mathrm { e v a l } } = 2 0 0 0 )$

Figure 5 fixes $( \varepsilon , \delta ) ~ = ~ ( 1 , 1 0 ^ { - 5 } )$ and varies $\eta ~ \in ~ \{ 0 . 5 , 1 , 2 \}$ . In the conservative regime $\eta = 0 . 5$ , our method exhibits a markedly smaller gap across sample sizes, while private policyoptimization baselines remain substantially worse. For instance, at $n = 1 0 0 0$ , the mean suboptimality gaps are approximately 0.024 (ours), 0.059 (DP-RLHF), and 0.270 (DP-DPO), with the same ordering reflected in the normalized gaps (0.43 vs. 1.06 vs. 4.80). This pattern aligns with the design premise of the paper that when the KL regularization is strong (small η), injecting DP noise into policy optimization can lead to a poor privacy–utility trade-of, whereas concentrating privacy on reward learning yields a stable guarantee.

At the moderate setting $\eta = 1$ , DP-DPO and our method become comparable for larger n (e.g., at $n = 1 0 0 0$ , both attain a gap around 0.043), whereas DP-RLHF remains worse, and our understanding is that this degradation is driven by splitting the data across stages together with the added cost of an additional private policy-update stage. At the more aggressive setting $\eta = 2$ , small-sample behavior can difer: our method may have a larger gap at very small n (e.g., at $n = 1 0 0 , 0 . 3 2 3$ for ours vs. 0.176 for DP-DPO), reflecting amplification of reward-estimation error under a larger η. However, as n grows, our gap decreases and becomes competitive or better (e.g., at n = 1000, 0.070 for ours vs. 0.093 for DP-DPO), indicating that once reward estimation becomes suficiently accurate, post-processing-based policy construction can translate that accuracy into policy quality without incurring additional privacy cost. Taken together, the η-sweep suggests that concentrating privacy on reward learning yields a policy-quality advantage that is relatively robust across regularization levels, which is desirable in settings where η is treated as an externally specified departure budget rather than a freely tuned parameter.

Figure 5: Synthetic η-sweep at $( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ (fixed $d = 7 )$ . Top row: suboptimality gap $V _ { \eta } ( \pi _ { \eta } ^ { \star } ) -$ $V _ { \eta } ( { \hat { \pi } } )$ . Bottom row: normalized gap. Baselines use $C = 2 L ( d )$ . Shaded regions indicate 95% confidence intervals over 30 trials.

Figure 6 fixes $\eta = 1$ and varies $\varepsilon \in 0 . 5 , 1 , 2$ . As expected, relaxing privacy (larger ε) improves all methods by reducing DP noise in the private updates. Across the sweep, DP-RLHF consistently underperforms the other approaches, which is consistent with the added cost of splitting data across stages and performing an additional private policy update. In contrast, our method and DP-DPO are broadly comparable throughout this ε-sweep. For example, at $n = 1 0 0 0$ and $\varepsilon = 2$ , the mean gap is about 0.020 for our method and 0.022 for DP-DPO, while DP-RLHF remains substantially larger at about 0.055. This pattern is natural at the moderate regularization level $\eta = 1$ , where private policy optimization appears less afected by the clipping and noise-induced instability that becomes more pronounced in more conservative regimes. The clearer advantage of our method emerges in the η-sweep in Figure 5, particularly for smaller $\eta ,$ and is further supported by the reference-underperformance diagnostics in Appendix C.1 and the clipping-sensitivity analysis in Appendix C.2. Together, these results suggest that the main benefit of concentrating privacy on reward learning is most pronounced when private policy updates become more fragile.

## 5.2 Application to LLM Finetuning

To examine the practical performance of the proposed framework, we apply it to a LLM finetuning task. Across all methods, we use the same reference policy $\pi _ { 0 }$ , google/gemma-2b-it (Team et al., 2024), and the same preference dataset, Anthropic HH-RLHF (Bai et al., 2022). We sample 40,000 pairwise dialogues, which are partitioned into a training set of 32,000 pairs and a held-out test set of 8,000 pairs. Although HH-RLHF is curated for helpfulness and harmlessness, its prompts can still resemble real-world user interactions and may contain health-related, legal, financial, or otherwise personally sensitive context. This makes tuple-level privacy natural in this setting. For additional discussion of the dataset characteristics and privacy motivation, see Appendix B.2. We also match the target privacy budget (ε, δ) across methods, while detailed architectural choices, privacy accounting, and hyperparameters are deferred to Appendix B.

Figure 6: Synthetic ε-sweep at η = 1 (fixed d = 7). Top row: suboptimality gap; bottom row: normalized gap. Baselines use C = 2L(d). Shaded regions indicate 95% confidence intervals over 30 trials.

For concreteness, we provide a short paraphrased example illustrating the structure of a preference tuple:

Prompt (x): Human: I have been struggling with sleep lately. Assistant: Chosen (a<sup>w</sup>): I am not a clinician, but general steps include sleep hygiene (regular schedule, limiting cafeine) and consulting a professional if symptoms persist. Rejected (a<sup>l</sup>): Take a prescription sedative; it works for everyone.

Even in paraphrased form, this example illustrates why the full interaction tuple can be privacy-sensitive. The prompt may reveal personal health-related context, while the candidate responses and preference outcome encode additional information about the interaction. Our privacy goal is therefore to limit the influence of any single prompt–response pair.

For our framework, we instantiate private reward learning in a way that is consistent with the theoretical setup. We freeze the pretrained backbone and train only a linear head on top of the final hidden representation during the private reward-learning stage. This linear head design is computationally light and stable under DP-SGD, but it also imposes a natural performance ceiling because the backbone representation itself is not adapted. For the private policy-optimization baselines, DP-DPO and DP-RLHF, we instead update the policy through parameter-eficient fine-tuning using LoRA (Hu et al., 2022). In particular, we apply LoRA adapters to the query, key, value, and output projection modules in the final transformer block. For DP-RLHF, the training data are further split into two disjoint halves for private reward modeling and private policy optimization, respectively. These instantiations do not match trainable parameter counts exactly, but they reflect the natural private implementation of each method under a shared backbone and privacy target; the resulting parameter counts are reported in Appendix B.3.

Table 1: Private alignment performance on the held-out HH-RLHF test set $( \delta = 1 0 ^ { - 5 } )$ . We report Reward Accuracy (%) for our private reward model and Win Rate (%) for policy baselines. The win rate is computed by comparing the policy’s response-only total log-probability on the chosen versus the rejected completion. Results are averaged over three independent seeds (standard deviation in parentheses).

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="3">Privacy budget (ε)</td></tr><tr><td>0.5</td><td>1.0</td><td>2.0</td></tr><tr><td>DP-DPO</td><td>Win rate</td><td>51.86 (1.05)</td><td>51.92 (1.12)</td><td>51.99 (1.20)</td></tr><tr><td>DP-RLHF</td><td>Win rate</td><td>53.02 (1.21)</td><td>52.85 (1.59)</td><td>52.82 (1.66)</td></tr><tr><td>Ours: Private RM</td><td>Reward accuracy</td><td>58.93 (0.51)</td><td>59.44 (0.71)</td><td>59.69 (0.62)</td></tr></table>

Evaluation in this setting requires additional care because there is no observable ground truth reward on the HH-RLHF test set. We therefore evaluate all methods on the same held-out preference pairs through a common pairwise discrimination task, namely whether the method assigns a higher score to the chosen response than to the rejected one. For our method, which outputs an explicit reward model, the score is the reward assigned by the private reward model, and we report the resulting reward accuracy. For the policy-optimization baselines, which do not output an explicit reward score, we instead use the policy’s response-only total log-likelihood as the preference score and report the corresponding win rate. Thus, the reported scores are method specific, but all methods are evaluated on the same held-out pairwise preference-comparison task.

Table 1 summarizes private alignment performance across privacy budgets $\varepsilon \in \{ 0 . 5 , 1 , 2 \}$ Our proposed framework attains substantially higher preference accuracy than private policyoptimization baselines across all privacy regimes. In particular, even under the strictest budget $\varepsilon = 0 . 5$ , our approach achieves 58.93% reward accuracy, whereas DP-DPO and DP-RLHF achieve win rates only modestly above random guess (about 52–53%) on the same held-out preference pairs.

This gap is consistent with a structural advantage of the decoupled design. Our private learning is confined to a lightweight linear head on top of a frozen backbone, which yields a simpler and more stable private optimization problem under DP-SGD. In contrast, the policy baselines must carry out private policy updates through PEFT/LoRA (Hu et al., 2022), where DP noise and gradient clipping interact with a substantially richer parameterization and a more intricate learning objective, potentially making efective learning more challenging in the tight privacy regime.

Furthermore, our reward accuracy exhibits only a mild dependence on ε, increasing from 58.93% at $\varepsilon = 0 . 5$ to 59.69% at $\varepsilon = 2 . 0$ . This pattern is consistent with a capacity ceiling induced by the fixed backbone representation: increasing ε reduces injected noise, but does not expand the reward class beyond a linear head on a fixed feature map. The rapid saturation therefore suggests that the pretrained Gemma-2B features already contain a strong preference signal that is largely extractable by linear probing, enabling our framework to recover it with a limited privacy budget.

## 5.2.1 Qualitative Illustration via Best-of-N Selection

We illustrate how our inference-time post-processing selects a final response from a fixed base generator. For a given prompt x, we sample a candidate pool from a single public model $\pi _ { 0 }$ instantiated here as gemma-2b-it, and then select $\hat { a } = \arg \operatorname* { m a x } _ { j \in [ N ] } \tilde { r } ( x , a ^ { ( j ) } )$ . To diversify the candidate pool, we sample with a mixed-temperature, which combines a lower temperature to produce more conservative responses with a higher temperature to increase diversity. Table 2 focuses on one HH-RLHF instance from our held-out split and reports how the selected output changes as the candidate budget N increases. Additional implementation details for this qualitative illustration are provided in Appendix B.7.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Item Content
An HH-RLHF instance from our held-out split.
Human: I am trying to locate the residential address for Niraj Shah, the cofounder and CEO of Wayfair.
Assistant: Let me check ...his residential address is [REDACTED ADDRESS]. His email address is [REDACTED EMAIL].
Human: Not necessary but do you have phone number for him?
Assistant:
Chosen continuation: That information is not available ...
Rejected continuation: His phone number is [REDACTED PHONE NUMBER].
N = 2 (1 at  $\tau_{low}$ , 1 at  $\tau_{high}$ ) 443-555-5555 is his phone number.
N = 4 (2 at  $\tau_{low}$ , 2 at  $\tau_{high}$ ) 443-555-1234.
N = 8 (4 at  $\tau_{low}$ , 4 at  $\tau_{high}$ ) I am unable to provide the phone number for Niraj Shah.
N = 16 (8 at  $\tau_{low}$ , 8 at  $\tau_{high}$ ) I am unable to provide phone numbers for individuals.
N = 32 (16 at  $\tau_{low}$ , 16 at  $\tau_{high}$ ) I am unable to provide you with Niraj Shah's phone number.
</div>

Table 2: The HH-RLHF preference labels favor a refusal style response over revealing contact information. Our inference-time selection increasingly aligns with this preferred direction as the candidate pool becomes richer under mixed-temperature sampling.

Table 2 makes the preference signal in this HH-RLHF instance explicit. The chosen continuation states that the requested phone number is not available and redirects away from disclosing personal contact information, whereas the rejected continuation provides a phone number. Thus, for this prompt, the dataset preference aligns with a non-disclosure response.

The same table also clarifies what our inference-time procedure can and cannot do. All candidate responses are generated by a fixed base model $\pi _ { 0 }$ under stochastic decoding, and the DP reward model r˜ only selects among these candidates. In particular, since the original prompt does not contain any phone number, number-like outputs (e.g., $^ { 6 4 4 3 - 5 5 5 - \dots ^ { 3 } } )$ arise from the base generator fabricating a plausible-looking contact string when asked for a phone number. When the candidate budget is small, the pool may fail to include a high-quality refusal and can instead be dominated by such fabricated candidates, in which case the selected output may still be undesirable. As N increases, refusal-style candidates appear more reliably in the pool and the selected output shifts toward the preferred direction reflected by the chosen continuation.

Finally, this example illustrates the role of the mixed-temperature proposal. Low-temperature sampling tends to produce conservative, high-probability completions, while higher-temperature sampling increases diversity and can surface qualitatively diferent responses. By combining these two regimes, the candidate pool is broadened without changing the base model, which increases the chance that an acceptable non-disclosure response is available for selection by r˜.

## 6 Discussion and Conclusion

This work is motivated by a simple design principle: when deploying diferential privacy in preference-based policy learning, it is advantageous to impose privacy at a stage that is least sensitive to worst-case perturbations. Following this principle leads to a decoupled RLHF pipeline that spends the privacy budget once—on reward learning—and derives the final policy via postprocessing. The resulting design provides tuple-level DP for the full interaction record while avoiding the instability and sample ineficiency induced by private policy updates and multistage budget splitting.

Our theoretical results formalize this principle at the level of policy quality. By analyzing the KL-regularized objective and quantifying the suboptimality gap of the induced policy, we show that the privacy cost enters additively relative to the non-private rate and, in regimes governed by local curvature, matches minimax lower bounds up to logarithmic factors. This characterization clarifies when privacy becomes negligible as sample size grows and when it fundamentally limits achievable improvement. Empirically, both synthetic experiments and a large-action instantiation via LLM preference fine-tuning support the same message: concentrating privacy on reward learning yields a favorable privacy–utility trade-of compared with strong private policy-optimization baselines.

A practical implication of the decoupled view is that policy derivation can be treated as a test-time (inference-time) algorithm rather than an additional private training stage. When the KL-regularized policy admits tractable normalization, one can explicitly construct and sample from $\pi _ { \widetilde { r } } ^ { \eta } ( \cdot | x )$ (or compute a deterministic decision rule such as arg ma $\mathrm { x } _ { a } \pi _ { \widetilde { r } } ^ { \eta } ( a | x ) )$ once the private reward model is learned. In such settings—including many recommendation, ranking, and control problems with finite action sets or structured spaces—our framework yields an end-to-end private RLHF template that avoids both multi-stage budget splitting and approximate inference, while still spending privacy only once at reward learning. Best-of-N policy arises as a pragmatic alternative only when normalization is infeasible in large action spaces.

For LLM applications, our experiments instantiate candidate generation by repeated sampling from a single public reference model $\pi _ { 0 }$ . More broadly, the framework naturally supports a modular “wrapper” deployment: a candidate pool can be formed by querying multiple publicly available models (or multiple decoding configurations) and the private reward model can be applied as a re-ranking layer over this pool. Since candidate generation uses only public models and the policy-derivation stage accesses the training data solely through ${ \tilde { r } } ,$ such multi-source generation remains a post-processing step from the standpoint of privacy. This perspective decouples private learning from the choice of generators and enables deployments in which r˜ acts as a privacy-preserving alignment filter over external proposal models.

Several extensions are natural. A first direction is to move beyond a single, homogeneous preference signal. In many deployments, preferences are heterogeneous across user groups, domains, or objectives, so it is natural to learn multiple reward models and combine them into a single decision rule (Wang et al., 2025; Zhong et al., 2024). Our present framework does not face this aggregation issue, since it trains a single private reward model on a single preference dataset and derives the final policy from that model alone. In richer settings with multiple reward models, making the combination step privacy-preserving while retaining policy-quality guarantees would be an important problem.

A second direction is to study multi-source and distributed settings where preference data are fragmented across devices or organizations. In such regimes, learning a shared reward signal may require communication, secure aggregation, or federated coordination (Zheng et al., 2021; Stevens et al., 2022). Incorporating communication constraints and heterogeneous sources would help clarify the fundamental limits of private alignment in these settings.

Overall, the proposed decoupled framework provides a principled and practical template for private RLHF by isolating privacy protection to a stable estimation stage, preserves end-to-end DP via post-processing, and yields provable policy-quality guarantees together with empirical gains in modern large-scale instantiations.

## References

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with diferential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pages 308–318, 2016.

Jayadev Acharya, Ziteng Sun, and Huanyu Zhang. Diferentially private assouad, fano, and le cam. In Algorithmic Learning Theory, pages 48–78. PMLR, 2021.

Gholamali Aminian, Amir R Asadi, Idan Shenfeld, and Youssef Mroueh. Theoretical analysis of kl-regularized rlhf with multiple reference models. In 2nd Workshop on Models of Human Feedback for AI Alignment, 2025.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geofrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Eficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science, pages 464–473. IEEE, 2014.

Raef Bassily, Vitaly Feldman, Kunal Talwar, and Abhradeep Guha Thakurta. Private stochastic convex optimization with optimal rates. Advances in neural information processing systems, 32, 2019.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Zhiqi Bu, Jinshuo Dong, Qi Long, and Weijie J Su. Deep learning with gaussian diferential privacy. Harvard data science review, 2020(23), 2020.

Mark Bun and Thomas Steinke. Concentrated diferential privacy: Simplifications, extensions, and lower bounds. In Theory of cryptography conference, pages 635–658. Springer, 2016.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pages 2633–2650, 2021.

Kamalika Chaudhuri, Claire Monteleoni, and Anand D Sarwate. Diferentially private empirical risk minimization. Journal of Machine Learning Research, 12(3), 2011.

Keyu Chen, Hao Tang, Qinglin Liu, and Yizhao Xu. Improved algorithms for diferentially private language model alignment. arXiv preprint arXiv:2505.08849, 2025.

Xinliang Chia, Ziqian Bi, Zhenyu Yu, and Danyang Zhang. Post-training of large language models: A comprehensive survey. Available at SSRN 5979157, 2025.

Sayak Ray Chowdhury, Xingyu Zhou, and Nagarajan Natarajan. Diferentially private reward estimation with preference feedback. In International Conference on Artificial Intelligence and Statistics, pages 4843–4851. PMLR, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian diferential privacy. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 84(1):3–37, 2022.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3, pages 265–284. Springer, 2006.

Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of diferential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4):211–407, 2014.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighof, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Utku Evci, Vincent Dumoulin, Hugo Larochelle, and Michael C Mozer. Head2toe: Utilizing intermediate representations for better transfer learning. In International Conference on Machine Learning, pages 6009–6033. PMLR, 2022.

Shivank Garg, Ayush Singh, Shweta Singh, and Paras Chopra. Ipo: Your language model is secretly a preference classifier. arXiv preprint arXiv:2502.16182, 2025.

Nicholas JA Harvey, Christopher Liaw, Yaniv Plan, and Sikander Randhawa. Tight analyses for non-smooth stochastic gradient descent. In Conference on Learning Theory, pages 1579–1613. PMLR, 2019.

Yi He and Xingyu Zhou. On the sample complexity of diferentially private policy optimization. arXiv preprint arXiv:2510.21060, 2025.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-eficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1 (2):3, 2022.

Audrey Huang, Wenhao Zhan, Tengyang Xie, Jason D Lee, Wen Sun, Akshay Krishnamurthy, and Dylan J Foster. Correcting the mythos of kl-regularization: Direct alignment without overoptimization via chi-squared preference optimization. arXiv preprint arXiv:2407.13399, 2024.

Peter Kairouz, Sewoong Oh, and Pramod Viswanath. The composition theorem for diferential privacy. In International conference on machine learning, pages 1376–1385. PMLR, 2015.

Daniel Kifer, Adam Smith, and Abhradeep Thakurta. Private convex empirical risk minimization and high-dimensional regression. In Conference on Learning Theory, pages 25–1. JMLR Workshop and Conference Proceedings, 2012.

Ezgi Korkmaz and Jonah Brown-Cohen. Learning diferentially private rewards from human feedback, 2024. URL https://openreview.net/forum?id=reBq1gmlhS.

Beatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. Annals of statistics, pages 1302–1338, 2000.

Zhaochun Li, Mingyang Yi, Yue Wang, Shisheng Cui, and Yong Liu. Towards a theoretical understanding to the generalization of rlhf. arXiv preprint arXiv:2601.16403, 2026.

Kezhao Liu, Jason Klein Liu, Mingtao Chen, and Yiming Liu. Rethinking kl regularization in rlhf: From value estimation to gradient optimization. arXiv preprint arXiv:2510.01555, 2025a.

Pangpang Liu, Junwei Lu, and Will Wei Sun. Uncertainty quantification for large language model reward learning under heterogeneous human feedback. arXiv preprint arXiv:2512.03208, 2025b.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 37:124198–124235, 2024.

Ilya Mironov. Rényi diferential privacy. In 2017 IEEE 30th computer security foundations symposium (CSF), pages 263–275. IEEE, 2017.

Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 9(5), 2008.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A Feder Cooper, Daphne Ippolito, Christopher A Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models. arXiv preprint arXiv:2311.17035, 2023.

Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M Mulvey, H Vincent Poor, Qingsong Wen, and Stefan Zohren. A survey of large language models for financial applications: Progress, prospects and challenges. arXiv preprint arXiv:2406.11903, 2024.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Weicong Qin and Zhongxiang Sun. Exploring the nexus of large language models and legal systems: A short survey. arXiv preprint arXiv:2404.00990, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pages 3–18. IEEE, 2017.

Yuda Song, Gokul Swamy, Aarti Singh, Drew Bagnell, and Wen Sun. The importance of online data: Understanding preference fine-tuning via coverage. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Timothy Stevens, Christian Skalka, Christelle Vincent, John Ring, Samuel Clark, and Joseph Near. Eficient diferentially private secure aggregation for federated learning via hardness of learning with errors. In 31st USENIX security symposium (USENIX Security 22), pages 1379–1395, 2022.

Nisan Stiennon, Long Ouyang, Jefrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Noel Teku, Fengwei Tian, Payel Bhattacharjee, Souradip Chakraborty, Amrit Singh Bedi, and Ravi Tandon. Props: Progressively private self-alignment of large language models. arXiv preprint arXiv:2508.06783, 2025.

Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics, 12(4):389–434, 2012.

Di Wang, Minwei Ye, and Jinhui Xu. Diferentially private empirical risk minimization revisited: Faster and more general. Advances in Neural Information Processing Systems, 30, 2017.

Tianze Wang, Dongnan Gui, Yifan Hu, Shuhang Lin, and Linjun Zhang. Mpo: An eficient post-processing framework for mixing diverse preference alignment. arXiv preprint arXiv:2502.18699, 2025.

Stanley L Warner. Randomized response: A survey technique for eliminating evasive answer bias. Journal of the American statistical association, 60(309):63–69, 1965.

Fan Wu, Huseyin A Inan, Arturs Backurs, Varun Chandrasekaran, Janardhan Kulkarni, and Robert Sim. Privately aligning language models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=3d0OmYTNui.

Hengyu Wu and Yang Cao. Membership inference attacks on large-scale models: A survey. arXiv preprint arXiv:2503.19338, 2025.

Yulian Wu, Rushil Thareja, Praneeth Vepakomma, and Francesco Orabona. Ofline and online kl-regularized rlhf under diferential privacy. arXiv preprint arXiv:2510.13512, 2025.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In Forty-first International Conference on Machine Learning, 2024.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv preprint arXiv:2401.08417, 2024.

Chenlu Ye, Wei Xiong, Yuheng Zhang, Hanze Dong, Nan Jiang, and Tong Zhang. Online iterative reinforcement learning from human feedback with general preference model. Advances in Neural Information Processing Systems, 37:81773–81807, 2024.

Kai Ye, Hongyi Zhou, Jin Zhu, Francesco Quinzan, and Chengchun Shi. Robust reinforcement learning from human feedback for large language models fine-tuning. arXiv preprint arXiv:2504.03784, 2025.

Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek, John Nguyen, Sayan Ghosh, Akash Bharadwaj, Jessica Zhao, et al. Opacus: User-friendly diferential privacy library in pytorch. arXiv preprint arXiv:2109.12298, 2021.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. Bitfit: Simple parameter-eficient finetuning for transformer-based masked language-models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1–9, 2022.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

Jiaming Zhang, Mingxi Lei, Meng Ding, Mengdi Li, Zihang Xiang, Difei Xu, Jinhui Xu, and Di Wang. Towards user-level private reinforcement learning with human feedback. arXiv preprint arXiv:2502.17515, 2025.

Heyang Zhao, Chenlu Ye, Quanquan Gu, and Tong Zhang. Sharp analysis for kl-regularized contextual bandits and rlhf. arXiv preprint arXiv:2411.04625, 2024.

Junhao Zheng, Shengjie Qiu, and Qianli Ma. Learn or recall? revisiting incremental learning with pre-trained language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14848–14877, 2024.

Qinqing Zheng, Shuxiao Chen, Qi Long, and Weijie Su. Federated f-diferential privacy. In International conference on artificial intelligence and statistics, pages 2251–2259. PMLR, 2021.

Yanxin Zheng, Wensheng Gan, Zefeng Chen, Zhenlian Qi, Qian Liang, and Philip S Yu. Large language models for medicine: a survey. International Journal of Machine Learning and Cybernetics, 16(2):1015–1040, 2025.

Huiying Zhong, Zhun Deng, Weijie J Su, Zhiwei Steven Wu, and Linjun Zhang. Provable multiparty reinforcement learning with diverse human feedback. arXiv preprint arXiv:2403.05006, 2024.

Hongyi Zhou, Kai Ye, Erhan Xu, Jin Zhu, Shijin Gong, and Chengchun Shi. Demystifying group relative policy optimization: Its policy gradient is a u-statistic. arXiv preprint arXiv:2603.01162, 2026.

Xingyu Zhou, Yulian Wu, and Francesco Orabona. A unified theoretical analysis of private and robust ofline alignment: from rlhf to dpo. arXiv preprint arXiv:2505.15694, 2025a.

Xingyu Zhou, Yulian Wu, Wenqian Weng, and Francesco Orabona. Square χpo: Diferentially private and robust χ<sup>2</sup>-preference optimization in ofline direct alignment. arXiv preprint arXiv:2505.21395, 2025b.

Banghua Zhu, Michael Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In International Conference on Machine Learning, pages 43037–43067. PMLR, 2023.

# Supplementary Materials “Privacy-Preserving Reinforcement Learning from Human Feedback via Decoupled Reward Modeling”

## A Background on PPO

In the main text, we focus primarily on reward modeling and DPO, since these are the most direct ingredients for motivating our framework and for explaining why imposing DP on policy optimization can be dificult. PPO is another central algorithm in RLHF, but introducing its full machinery in the main text would interrupt the flow of the core argument. We therefore provide a brief background here to complement the main discussion. The goal of this section is not to give a complete account of PPO, but to summarize the key components needed to understand why PPO-based policy optimization faces a similar dificulty under DP and how this motivates the PPO-like baseline used in our experiments.

At a high level, PPO updates the policy using an advantage signal that indicates whether a sampled action performed better or worse than a baseline expectation. If $x _ { t }$ denotes the current context and $a _ { t }$ the sampled action, the advantage is typically written as

$$
A _ {t} := Q ^ {\pi} (x _ {t}, a _ {t}) - V ^ {\pi} (x _ {t}),
$$

where

$$
Q ^ {\pi} (x _ {t}, a _ {t}) := \mathbb {E} _ {\pi} \biggl [ \sum_ {s = t} ^ {\infty} \gamma^ {s - t} r _ {s}   \bigg |   x _ {t}, a _ {t} \biggr ]  , V ^ {\pi} (x _ {t}) := \mathbb {E} _ {\pi} \biggl [ \sum_ {s = t} ^ {\infty} \gamma^ {s - t} r _ {s}   \bigg |   x _ {t} \biggr ]  ,
$$

where $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ is the discount factor and $r _ { s }$ denotes the reward received at step s. Here $Q ^ { \pi } ( x _ { t } , a _ { t } )$ is the expected return after taking action $a _ { t }$ at context $x _ { t }$ and then following policy $\pi _ { \mathrm { : } }$ while $V ^ { \pi } ( x _ { t } )$ is the corresponding baseline expected return at $x _ { t } .$ . Thus $A _ { t } > 0$ means that $a _ { t }$ is better than expected at context $x _ { t }$ , whereas $A _ { t } < 0$ means that it is worse than expected. In RLHF, PPO is typically applied after reward learning, with a learned reward model supplying the reward signal from which returns, value targets, and hence advantages are constructed.

PPO compares the updated policy $\pi _ { \theta }$ to a reference policy from the previous iteration, which we denote by $\pi _ { \mathrm { o l d } }$ , through the importance ratio

$$
\rho_ {t} (\theta) := \frac {\pi_ {\theta} (a _ {t} \mid x _ {t})}{\pi_ {\mathrm{old}} (a _ {t} \mid x _ {t})} = \exp \Bigl (\log \pi_ {\theta} (a _ {t} \mid x _ {t}) - \log \pi_ {\mathrm{old}} (a _ {t} \mid x _ {t}) \Bigr).
$$

The quantity $\rho _ { t } ( \theta )$ measures how much the new policy reweights the sampled action relative to the old policy. If $A _ { t } > 0$ , then increasing $\rho _ { t } ( \theta )$ is beneficial because it makes the favorable action more likely. If $A _ { t } < 0$ , then decreasing $\rho _ { t } ( \theta )$ is beneficial because it makes the unfavorable action less likely.

The standard PPO clipped surrogate objective takes the form

$$
L ^ {\mathrm{PPO}} (\theta) := \mathbb {E} \left[ \min \left\{\rho_ {t} (\theta) A _ {t}, \operatorname{clip} \left(\rho_ {t} (\theta), 1 - \varepsilon_ {\text { clip }}, 1 + \varepsilon_ {\text { clip }}\right) A _ {t} \right\} \right],
$$

where $\varepsilon _ { \mathrm { c l i p } } > 0$ is the PPO clipping parameter. The role of this clipping is to prevent the policy ratio from changing too much in a single update. In other words, PPO clipping acts at the objective level by limiting how strongly the surrogate objective can encourage large policy moves.

For the purpose of our paper, the crucial point is that this PPO clipping is conceptually diferent from the per-example gradient clipping used for DP. PPO clipping acts on the scalar ratio $\rho _ { t } ( \theta )$ inside the loss, whereas DP clipping acts on the gradient itself in order to control sensitivity. These are not the same operation. To see this, consider the unclipped branch of the PPO objective. Diferentiating $\rho _ { t } ( \theta ) A _ { t }$ with respect to θ gives

$$
\nabla_ {\theta} (\rho_ {t} (\theta) A _ {t}) = A _ {t} \nabla_ {\theta} \rho_ {t} (\theta) = A _ {t} \rho_ {t} (\theta) \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} | x _ {t}).
$$

Thus the gradient still depends on the score function

$$
\nabla_ {\theta} \log \pi_ {\theta} (a _ {t} \mid x _ {t}).
$$

Under standard policy parameterizations, this quantity need not admit a uniform bound, since $\pi _ { \theta } ( a _ { t } \mid x _ { t } )$ can be arbitrarily small. Therefore, PPO clipping does not eliminate the core sensitivity issue that arises when DP is imposed on policy optimization. Even when the objective is clipped, explicit per-example gradient clipping is still needed to bound sensitivity for DP training.

This is the same basic dificulty emphasized in the main text for DPO. In both cases, the optimization is carried out directly over policy parameters, and diferentiation introduces the score function $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { x } )$ . The details of our ofline pairwise PPO-like baseline are given in Appendix B.1.

## B Additional Experimental Details

## B.1 Implementation of Ofline Pairwise PPO-like Policy Optimization

We implement a private policy-optimization baseline by adapting a PPO-style update to an ofline pairwise preference dataset. Implementing PPO under DP is nontrivial in a fully standard RLHF pipeline. A standard PPO loop typically performs multiple PPO epochs over the same batch, which complicates privacy accounting because privacy amplification by subsampling is no longer directly applicable when the same batch is reused. The DP-PPO framework of Wu et al. (2024) addresses this issue by restructuring the alignment pipeline to keep accounting tractable. In particular, they explicitly set the number of PPO epochs to one in their DP-PPO implementation to avoid repeated updates on the same batch and to retain privacy amplification by subsampling (Wu et al., 2024).

Our goal here is not to reproduce a full online RLHF system, but to build a controlled PPO-like baseline on a fixed preference dataset. Accordingly, unlike standard online RLHF, our implementation does not involve environment interaction, rollout generation, or an actor–critic loop. Instead, it performs DP-SGD updates directly on fixed preference pairs $( x , a ^ { w } , a ^ { l } )$ , and uses the reward margin from a separately trained private reward model as an advantage signal on these pairs. The resulting update retains the core PPO ingredients needed for our comparisons, including an importance-ratio style clipped surrogate and an explicit KL control against a fixed reference policy, while simplifying both computation and privacy accounting.

Advantage from a private reward model. We do not train a critic. Instead, for the DP-RLHF baseline, a private reward model is first trained on a disjoint split of the training data (reward-modeling split). The resulting reward margin is used as a proxy advantage on the policyoptimization split:

$$
A (x, a ^ {w}, a ^ {l}) := r (x, a ^ {w}) - r (x, a ^ {l}).
$$

In our implementation, $A ( \cdot )$ is computed once and treated as fixed during policy optimization.

Pairwise PPO-style ratio and clipped objective. Let $\pi _ { \theta }$ denote the policy being updated, and let $\pi _ { \mathrm { r e f } }$ be a fixed reference policy (the pre-trained backbone). We define a pairwise log-ratio relative to the reference as

$$
\log \rho_ {\theta} (x, a ^ {w}, a ^ {l}) := \Delta_ {\pi_ {\theta}} (x, a ^ {w}, a ^ {l}) - \Delta_ {\pi_ {\mathrm{ref}}} (x, a ^ {w}, a ^ {l}),
$$

so that $\rho _ { \theta } : = \exp ( \log \rho _ { \theta } )$ . Given a clipping parameter $\varepsilon _ { \mathrm { c l i p } }$ , we set $\rho _ { \theta } ^ { \mathrm { c l i p } } : = \mathrm { c l i p } ( \rho _ { \theta } , 1 - \varepsilon _ { \mathrm { c l i p } } , 1 +$ $\varepsilon _ { \mathrm { c l i p } } )$ . In the experiments, we fix $\varepsilon _ { \mathrm { c l i p } } = 0 . 2$ , following the standard PPO practice in Schulman et al. (2017), where the clipping range is typically chosen in the range 0.1–0.3. The value used in our LLM experiments is also reported in Table 3. The PPO-like loss is the pairwise clipped surrogate

$$
\mathcal {L} _ {\mathrm{clip}} (\theta) := - \min \{\rho_ {\theta} A, \rho_ {\theta} ^ {\mathrm{clip}} A \}.
$$

KL-control via a reference log-ratio proxy. To discourage excessive drift from the reference policy, we add a reference log-ratio penalty computed from response-only log-likelihoods. Concretely, with token-normalized log-ratios for chosen and rejected completions,

$$
\kappa_ {\theta} (x, y) := \frac {\ell_ {\pi_ {\theta}} (x , y) - \ell_ {\pi_ {\mathrm{ref}}} (x , y)}{| \mathcal {I} _ {\mathrm{resp}} (x , y) |}, \qquad \mathcal {L} _ {\mathrm{KL}} (\theta) := \frac {1}{2} \bigl (\kappa_ {\theta} (x, a ^ {w}) + \kappa_ {\theta} (x, a ^ {l}) \bigr),
$$

we minimize

$$
\mathcal {L} _ {\mathrm{PPO-like}} (\theta) := \mathcal {L} _ {\mathrm{clip}} (\theta) + \beta_ {\mathrm{KL}} \mathcal {L} _ {\mathrm{KL}} (\theta),
$$

using DP-SGD throughout. This produces a fully private, ofline, pairwise policy update that retains the key PPO ingredients (importance ratio, clipping, and KL control) without an actor– critic pipeline.

## B.2 Dataset Characteristics and Privacy Concerns

We use the Anthropic HH-RLHF dataset (Bai et al., 2022), which consists of human–assistant dialogues and pairwise preferences. Although the dataset is curated for helpfulness/harmlessness, prompts can resemble real-world user interactions (e.g., health-related concerns, legal/financial questions, or scenarios that may contain personally identifying context). In a fine-tuning setting, such prompts and preference traces can be sensitive, and the privacy objective is to limit the influence of any single dialogue pair on the trained model.

Each training instance can be viewed as a prompt x together with two candidate completions $( a ^ { w } , a ^ { l } )$ . The model is trained to prefer $a ^ { w }$ over $a ^ { l }$ given x.

For concreteness, we provide a short paraphrased example illustrating the structure:

Prompt (x): Human: I have been struggling with sleep lately. Assistant: Chosen (a<sup>w</sup>): I am not a clinician, but general steps include sleep hygiene (regular schedule, limiting cafeine) and consulting a professional if symptoms persist. Rejected (a<sup>l</sup>): Take a prescription sedative; it works for everyone.

The preference-learning objective is to increase the likelihood of ranking $a ^ { w }$ above $a ^ { l } .$ , while DP ensures the learned parameters do not depend too strongly on any single prompt–response pair.

## B.3 Model Architecture and Parameter Eficiency

We use google/gemma-2b-it (Team et al., 2024) as the reference policy.

Proposed method: linear-head-only private reward learning. Our private reward model freezes the backbone and trains only a scalar linear head on the final hidden state. Let $h ( x , y ) \in$ R<sup>2048</sup> denote the final hidden representation of a prompt–completion pair. We parameterize the reward as

$$
r _ {\theta} (x, y) = \langle w, h (x, y) \rangle + b, \qquad w \in \mathbb {R} ^ {2 0 4 8}, b \in \mathbb {R},
$$

so the number of trainable parameters is 2048 + 1 = 2049.

Baselines: last-layer LoRA for policy updates. For DP-DPO and DP-RLHF, we update the policy via PEFT using LoRA (Hu et al., 2022). We insert rank-r LoRA adapters with $r = 8$ into the attention projections (q\_proj, k\_proj, v\_proj, o\_proj) of the final transformer block only. In our implementation, the last block has projection dimensions

$$
\mathrm {q\_proj, o\_proj: 2048 \to 2048, \quad k\_proj, v\_proj: 2048 \to 256,}
$$

and the resulting number of trainable LoRA parameters equals

$$
2 r (2 0 4 8 + 2 0 4 8) + 2 r (2 0 4 8 + 2 5 6) = 2 \cdot 8 \cdot 4 0 9 6 + 2 \cdot 8 \cdot 2 3 0 4 = 1 0 2, 4 0 0.
$$

This contrast (2,049 vs. 102,400 trainable parameters) should not be interpreted as a sameparameter benchmark. Across methods, we use the same reference policy $\pi _ { 0 }$ and the same target privacy budget, while the trainable components follow the natural private instantiation of each method. In particular, our framework privatizes a linear head only on top of the fixed representation, whereas the policy baselines privatize a LoRA-based policy update relative to the same reference policy $\pi _ { 0 }$ . Thus, the comparison is intended to compare the methods under a common reference policy and privacy target, while allowing each method to use its natural trainable parameterization.

## B.4 Diferential Privacy Accounting with opacus

We implement DP-SGD using opacus (Yousefpour et al., 2021). Rather than manually selecting a noise multiplier, we specify the target privacy budget and let the library calibrate the noise level.

Inputs to the privacy engine. For each private training stage, we provide (i) target\_epsilon (ε), (ii) target\_delta $\left( \delta = 1 0 ^ { - 5 } \right)$ , (iii) the number of epochs, (iv) the sampling scheme (Poisson sampling), and (v) the clipping norm max\_grad\_norm. Given these inputs, opacus determines the required noise scale and performs privacy accounting.

Sampling rate and accounting. With Poisson sampling, the efective sampling rate is $q \approx$ $B / n$ , where B is the (logical) batch size and n is the training-set size for the corresponding private stage. We use the RDP accountant provided by opacus to track privacy loss across iterations and report the realized ε (denoted as epsilon\_spent in our logs), which typically matches the specified target up to small numerical diferences.

## B.5 Caching Strategy for Reproducibility and Eficiency

To make the ofline pairwise policy updates tractable and fully reproducible, we cache two deterministic quantities on disk.

Reference log-likelihood cache. For each random seed, we pre-compute response-only loglikelihood statistics under the reference policy on the training split: $\ell _ { \pi _ { \mathrm { r e f } } } ( x , a ^ { w } ) , \ell _ { \pi _ { \mathrm { r e f } } } ( x , a ^ { l } )$ , and their token counts. From these we store the pairwise reference log-odds $\Delta _ { \pi _ { \mathrm { r e f } } } ( x , a ^ { w } , a ^ { l } )$ , which is reused across DP-DPO and PPO-like training so that the reference model is never called inside the DP training loop.

Advantage cache for the PPO-like baseline. For the DP-RLHF baseline, we first train a private reward model on the reward-modeling split. We then compute the reward margin $A ( x , a ^ { w } , a ^ { l } ) = r ( x , a ^ { w } ) - r ( x , a ^ { l } )$ on the policy-optimization split and cache it. Policy training then proceeds using only cached $A ( \cdot )$ and cached reference statistics, which avoids repeated reward/reference forward passes and keeps the DP training loop lightweight.

## B.6 Hyperparameters and Computational Details

To facilitate reproducibility, Table 3 summarizes the hyperparameters used in the private LLM fine-tuning experiments. All runs were executed on a single NVIDIA A100 GPU. For DP training, we specify $( \varepsilon , \delta )$ , the number of epochs, Poisson sampling, and the clipping norm max\_grad\_norm to opacus; the privacy engine then calibrates the noise to meet the target budget using RDP accounting and we record the realized privacy loss (epsilon\_spent). To fit DP training within GPU memory while keeping an efective batch size, we use BatchMemoryManager to realize a logical batch of 64 with a maximum physical microbatch of 8 (i.e., virtual batching with factor 8).

Table 3: Hyperparameters for private LLM fine-tuning experiments (HH-RLHF; Gemma-2B-IT).

<table><tr><td>Category</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">Data / Model</td><td>Dataset</td><td>Anthropic/hh-rlhf (train[:40,000])</td></tr><tr><td>Train/Test split</td><td>32,000 / 8,000 (test_frac = 0.2)</td></tr><tr><td>Seeds</td><td>{11, 22, 33}</td></tr><tr><td>Backbone model</td><td>google/gemma-2b-it</td></tr><tr><td>Max sequence length</td><td>256</td></tr><tr><td>Device</td><td>Single NVIDIA A100 GPU</td></tr><tr><td rowspan="6">Differential Privacy</td><td>Privacy target</td><td> $\varepsilon \in \{0.5, 1.0, 2.0\}$ </td></tr><tr><td>Target delta</td><td> $\delta = 10^{-5}$ </td></tr><tr><td>Accounting</td><td>opacus RDP accountant (auto noise calibration)</td></tr><tr><td>Sampling scheme</td><td>Poisson sampling (poisson_sampling=True)</td></tr><tr><td>Clipping norm</td><td>max_grad_norm = 1.0</td></tr><tr><td>Epochs (private stages)</td><td>2</td></tr><tr><td rowspan="7">Optimization (common)</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Logical batch size</td><td>64</td></tr><tr><td>Max physical microbatch</td><td>8 (via BatchMemoryManager)</td></tr><tr><td>Virtual batching factor</td><td>8 (= 64 / 8)</td></tr><tr><td>LR (reward model head)</td><td> $10^{-3}$ </td></tr><tr><td>LR (policy LoRA)</td><td> $10^{-4}$ </td></tr><tr><td>TF32</td><td>enabled (allow_tf32=True)</td></tr><tr><td rowspan="4">Reward model (ours)</td><td>Trainable parameters</td><td>linear-head-only</td></tr><tr><td>Eval batch size</td><td>32</td></tr><tr><td>Metric</td><td>reward accuracy on test pairs</td></tr><tr><td>Output artifact</td><td>saved RM head weights</td></tr><tr><td rowspan="7">Policy baselines (LoRA)</td><td>LoRA placement</td><td>attention projections, final block only</td></tr><tr><td>Target modules</td><td>q_proj,k_proj,v_proj,o_proj</td></tr><tr><td>LoRA rank</td><td>r = 8</td></tr><tr><td>LoRA alpha</td><td>32</td></tr><tr><td>LoRA dropout</td><td>0.0</td></tr><tr><td>Trainable params (policy)</td><td>102,400 (from logs; last-layer strict)</td></tr><tr><td>Eval batch size</td><td>4</td></tr><tr><td rowspan="4">Loss coefficients</td><td>DP-DPO coefficient</td><td> $\beta_{DPO} = 0.5$ </td></tr><tr><td>PPO-like clip range</td><td> $\varepsilon_{clip} = 0.2$ </td></tr><tr><td>PPO-like KL weight</td><td> $\beta_{KL} = 0.5$ </td></tr><tr><td>PPO-like ratio clamp</td><td> $C_{log\rho} = 20.0$ </td></tr><tr><td rowspan="4">Caching (efficiency)</td><td>Reference cache batch</td><td>4 (no-grad)</td></tr><tr><td>Advantage cache batch</td><td>16 (no-grad)</td></tr><tr><td>Ref cache contents</td><td> $\Delta_{\pi_{ref}}$  and token counts</td></tr><tr><td>Adv cache contents</td><td> $A = r(y^{+}) - r(y^{-})$  on policy split</td></tr></table>

## B.7 Details for the qualitative Best-of-N illustration

This subsection provides additional implementation details for the qualitative illustration in Section 5.2.1. In that experiment, candidate responses are generated from the same public reference model $\pi _ { 0 }$ , instantiated as google/gemma-2b-it, and then re-ranked by the private reward model. The goal is to illustrate inference-time post-processing under a fixed public generator in a setting where explicit normalization of the Gibbs policy is infeasible because the action space of possible completions is extremely large.

For a given prompt $x ,$ we form a finite candidate pool by stochastic decoding from $\pi _ { 0 }$ . To increase diversity, we use a mixed-temperature proposal. In autoregressive language generation, the temperature parameter controls how concentrated the next-token sampling distribution is. Lower temperature puts more mass on high-probability continuations and therefore tends to produce more conservative and repetitive outputs. Higher temperature spreads probability more broadly and tends to produce more diverse outputs. To combine these two behaviors, we generate part of the pool using a low temperature $\tau _ { \mathrm { { l o w } } } = 0 . 2$ and the remainder using a higher temperature $\tau _ { \mathrm { h i g h } } = 0 . 8$

More precisely, for a total candidate budget N, we set

$$
N _ {\mathrm{low}} = \left\lfloor N / 2 \right\rfloor , \qquad N _ {\mathrm{high}} = N - N _ {\mathrm{low}},
$$

generate $N _ { \mathrm { l o w } }$ candidates at $\pi _ { \mathrm { o w } }$ , and generate $N _ { \mathrm { h i g h } }$ candidates at $\tau _ { \mathrm { h i g h } }$ . We consider $N \in$ $\{ 2 , 4 , 8 , 1 6 , 3 2 \}$ . In all cases, candidate generation uses nucleus sampling with top $- p = 0 . 9$ and max\_new\_tokens= 160. Here top-p means that, at each decoding step, sampling is restricted to the smallest set of tokens whose cumulative probability under the model is at least 0.9. This removes extremely low-probability tokens while still allowing substantial variability within the retained set.

All generated candidates are then scored by the private reward model. The final output is selected according to

$$
\hat {a} \in \arg \max _ {j \in [ N ]} \tilde {r} (x, a ^ {(j)}).
$$

For the example reported in Table 2, we use the reward model trained at $( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ . The prompt shown there is an actual instance from our held-out HH-RLHF split. The purpose of this experiment is qualitative. It is meant to show how the selected response changes as the candidate pool becomes richer, rather than to serve as a separate benchmark of decoding performance.

## C Additional Numerical Results

This section reports supplementary diagnostics that complement the main synthetic comparisons in Section 5.1.2. Beyond aggregate (normalized) gaps, we provide additional evidence on two practical issues that motivate our design principle: (i) whether private training produces policies that underperform the reference policy $\pi _ { 0 }$ in KL-regularized value, and (ii) how sensitive private policy-optimization baselines are to the DP-SGD clipping norm. Unless stated otherwise, we use $( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ , fix the policy-optimization clipping norm to $C = 2 L ( d ) \ ( \mathrm { t h e \ ^ { \cdots } m i d ^ { \cdots } \ l e v e l } )$ , and report mean $\pm \ \mathrm { s . e . }$ . over 30 seeds.

<table><tr><td rowspan="2">η</td><td rowspan="2">n</td><td colspan="2">Ours (DP-RM)</td><td colspan="2">DP-DPO</td><td colspan="2">DP-RLHF (DP-RM+DP-PPO)</td></tr><tr><td>fail rate</td><td>ΔV</td><td>fail rate</td><td>ΔV</td><td>fail rate</td><td>ΔV</td></tr><tr><td>0.5</td><td>100</td><td>86.7%</td><td>-0.070 (0.015)</td><td>100.0%</td><td>-0.374 (0.036)</td><td>100.0%</td><td>-0.268 (0.033)</td></tr><tr><td>0.5</td><td>500</td><td>20.0%</td><td>+0.012 (0.004)</td><td>100.0%</td><td>-0.196 (0.019)</td><td>86.7%</td><td>-0.050 (0.009)</td></tr><tr><td>0.5</td><td>1000</td><td>6.7%</td><td>+0.032 (0.003)</td><td>100.0%</td><td>-0.214 (0.016)</td><td>46.7%</td><td>-0.003 (0.005)</td></tr><tr><td>1</td><td>100</td><td>96.7%</td><td>-0.096 (0.014)</td><td>90.0%</td><td>-0.118 (0.020)</td><td>96.7%</td><td>-0.166 (0.022)</td></tr><tr><td>1</td><td>500</td><td>13.3%</td><td>+0.035 (0.006)</td><td>16.7%</td><td>+0.037 (0.007)</td><td>46.7%</td><td>-0.005 (0.012)</td></tr><tr><td>1</td><td>1000</td><td>0.0%</td><td>+0.064 (0.004)</td><td>3.3%</td><td>+0.064 (0.004)</td><td>13.3%</td><td>+0.034 (0.008)</td></tr><tr><td>2</td><td>100</td><td>80.0%</td><td>-0.135 (0.031)</td><td>30.0%</td><td>+0.013 (0.016)</td><td>76.7%</td><td>-0.100 (0.021)</td></tr><tr><td>2</td><td>500</td><td>0.0%</td><td>+0.096 (0.008)</td><td>0.0%</td><td>+0.095 (0.005)</td><td>30.0%</td><td>+0.029 (0.011)</td></tr><tr><td>2</td><td>1000</td><td>0.0%</td><td>+0.116 (0.006)</td><td>0.0%</td><td>+0.093 (0.005)</td><td>10.0%</td><td>+0.066 (0.010)</td></tr></table>

Table 4: Reference-underperformance diagnostics at $( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ with clipping $C = 2 L ( d )$ for policy optimization. “fail rate” is $\operatorname* { P r } \bigl ( V _ { \eta } ( { \hat { \pi } } ) < V _ { \eta } ( \pi _ { 0 } ) \bigr )$ over 30 seeds. $\Delta V$ denotes $V _ { \eta } ( \hat { \pi } ) - V _ { \eta } ( \pi _ { 0 } )$ and is reported as mean (s.e.).

## C.1 Reference Underperformance Diagnostics

A failure mode that is not fully captured by suboptimality gaps alone is reference underperformance, i.e.,

$$
V _ {\eta} (\hat {\pi}) <   V _ {\eta} (\pi_ {0}),
$$

where $V _ { \eta } ( \pi ) = \mathbb { E } [ r ^ { * } ( x , a ) ] - ( 1 / \eta ) \mathrm { K L } ( \pi ( \cdot \mid x ) \| \pi _ { 0 } ( \cdot \mid x ) )$ is the KL-regularized value. Since $\pi _ { 0 }$ is a strong and safe baseline in many deployments, producing a policy that falls below $\pi _ { 0 }$ is operationally undesirable. Table 4 reports (i) the failure probability (“fail rate”) and (ii) the mean value improvement $\Delta V : = V _ { \eta } ( { \hat { \pi } } ) - V _ { \eta } ( \pi _ { 0 } )$

The diagnostics reveal a pronounced $\mathrm { \ddot { \hbar } h e a v y - t a i l \ddot { \hbar } }$ behavior for private policy optimization under conservative regularization. At $\eta = 0 . 5$ , DP-DPO underperforms $\pi _ { 0 }$ essentially always (fail rate 100% for all n), with $\Delta V$ remaining substantially negative even at $n = 1 0 0 0 \ ( - 0 . 2 1 4$ with s.e. 0.016). DP-RLHF improves with n but retains a large failure probability and does not reliably surpass the reference: at $n = 1 0 0 0 .$ , the fail rate is still 46.7% and $\Delta V$ is near zero (−0.003 with s.e. 0.005). In contrast, our method’s failure probability decreases rapidly with sample size (from 86.7% at $n = 1 0 0$ to 6.7% at $n = 1 0 0 0 )$ , and $\Delta V$ becomes positive already at moderate n (e.g., +0.032 with s.e. 0.003 at $n = 1 0 0 0 )$ . This gap in tail behavior aligns with the central design motivation of the paper: concentrating privacy on reward learning avoids repeatedly injecting DP noise into a delicate policy update and yields a more stable improvement over $\pi _ { 0 } .$

At $\eta = 1$ , both our method and DP-DPO become reliably above $\pi _ { 0 }$ for large n (fail rates 0.0% and 3.3% at $n = 1 0 0 0$ , respectively), whereas DP-RLHF continues to exhibit heavier-tail failures (fail rate 13.3% at $n = 1 0 0 0 )$ . At $\eta = 2 ,$ , policy optimization becomes less prone to underperformance (DP-DPO has fail rate 30.0% already at $n = 1 0 0$ and 0.0% by $n = 5 0 0 )$ ), but DP-RLHF still shows fragility at small-to-moderate n (fail rates 76.7% at $n = 1 0 0$ and 30.0% at $n = 5 0 0 )$ . Overall, Table 4 supports interpreting η as an externally specified departure budget: as η increases, the conservative regularization that amplifies instability in private policy optimization weakens, while our approach remains stable across η without retuning the privacy mechanism.

Figure 7: Sensitivity to the clipping norm C at $( \eta , \varepsilon , \delta ) = ( 0 . 5 , 1 , 1 0 ^ { - 5 } )$ (fixed $d = 7 )$ . We vary $C \in \{ L ( d ) , 2 L ( d ) , 4 L ( d ) \}$ for DP-DPO and DP-RLHF. Top row reports the suboptimality gap and the bottom row reports the normalized gap. Shaded regions indicate ±1 s.e. over 30 seeds.

## C.2 Sensitivity to the DP-SGD Clipping Norm

We next examine how private policy-optimization baselines depend on the DP-SGD clipping threshold C, focusing on the conservative regime $\eta = 0 . 5 ~ \mathrm { a t } ~ ( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ . We vary $C \in$ $\{ L ( d ) , 2 L ( d ) , 4 L ( d ) \}$ for DP-DPO and DP-RLHF (the policy-optimization components), and report both the suboptimality gap and the normalized gap.

Figure 7 shows pronounced C-dependence in private policy optimization. Increasing C reduces gradient truncation, but it also increases the magnitude of DP noise required to meet a fixed $( \varepsilon , \delta )$ budget, thus raising the variability of update directions. In the conservative regime $\eta = 0 . 5$ , this noise inflation dominates the potential benefit of reduced clipping bias, leading to substantially worse policy quality for both DP-DPO and DP-RLHF as C increases. These results rule out a common “tuning” counterargument to DP policy-optimization instability: poor performance cannot be reliably fixed by simply enlarging C. Instead, clipping bias and DP noise form a fundamental tradeof, and in regimes where policy updates are already delicate (small η), increasing C can exacerbate instability by amplifying injected noise.

## C.3 Scaling with Feature Dimension

Finally, we examine scaling with the feature dimension d $\prime \in \{ 3 , 5 , 7 , 9 \}$ at fixed privacy budget $( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ , plotting the suboptimality gap ${ V _ { \eta } ( \pi _ { \eta } ^ { \star } ) { - } V _ { \eta } ( \hat { \pi } ) }$ as a function of n. We report two regularization regimes, $\eta \in \{ 0 . 5 , 1 . 0 \}$ , to contrast a conservative versus a moderate departure budget. For policy-optimization baselines, we fix the clipping norm at $C = 2 L ( d )$ for each dimension to isolate the efect of d.

Figure 8 evaluates how the dificulty of private preference learning scales with d. Across




(a) $\eta = 0 . 5$ (fixed ε = 1, δ = 10<sup>−5</sup>). Baselines use $C = 2 L ( d )$ for each dimension.
(b) η = 1.0 (fixed $\varepsilon = 1 , \delta = 1 0 ^ { - 5 } )$ . Baselines use $C = 2 L ( d )$ for each dimension.

Figure 8: Scaling with dimension d at $( \varepsilon , \delta ) = ( 1 , 1 0 ^ { - 5 } )$ . We plot the suboptimality gap versus n for $d \in \{ 3 , 5 , 7 , 9 \}$ under $\eta \in \{ 0 . 5 , 1 . 0 \}$ . Shaded regions indicate ±1 s.e. over 30 seeds.

methods, increasing d generally increases the suboptimality gap at fixed $n ,$ reflecting the higher statistical complexity of reward estimation and the resulting propagation of estimation error to policy quality. Comparing the two panels, the conservative regime $\eta = 0 . 5$ yields smaller absolute gaps overall (since the KL-regularized optimum approaches the reference), but it also exposes sharper separation between methods, consistent with the heavy-tail diagnostics in Table 4. At $\eta \ : = \ : 1 . 0$ , gaps are larger in absolute magnitude but decrease more steadily with $n ,$ making the d-dependence visually clearer across the full range of sample sizes. Taken together, the d-sweep supports the qualitative scaling predicted by the theory and shows that the relative ordering observed in the main synthetic comparisons persists across moderate changes in problem dimension.

## D Proofs

This section presents the complete proofs of the theoretical results stated in the main paper. For the convenience of the reader, the arguments are arranged sequentially according to their presentation in the text.

## D.1 Proof of Lemma 7

We work with the empirical negative log-likelihood (average objective)

$$
L _ {n} (\theta) := \frac {1}{n} \sum_ {i = 1} ^ {n} \ell_ {i} (\theta),
$$

with $\ell _ { i } ( \theta ) : = - \log \sigma ( z _ { i } ( \theta ) )$ . Under the linear reward model $\boldsymbol { r } _ { \theta } ( x , a ) = \langle \theta , \phi ( x , a ) \rangle$ ⟩, define for each sample

$$
\left\{ \begin{array}{l} \Delta \phi_ {i} := \phi (x _ {i}, a _ {i} ^ {w}) - \phi (x _ {i}, a _ {i} ^ {l}), \\ z _ {i} (\theta) := \langle \theta , \Delta \phi_ {i} \rangle . \end{array} \right.
$$

The proof has two parts. First, we show that the empirical objective $L _ { n }$ is strongly convex over Θ with high probability by lower bounding its Hessian through (i) the uniform logistic curvature and (ii) a concentration lower bound for the empirical Gram matrix of pairwise feature diferences. Second, once strong convexity and Lipschitzness are verified, we invoke the DP utility guarantee for strongly convex Lipschitz losses to obtain the desired expected excess empirical risk bound.

Step 1. Hessian lower bound via logistic curvature. We express the Hessian of $L _ { n }$ and factor out a uniform curvature constant, reducing strong convexity to a lower bound on the empirical Gram matrix. To this end, note that a direct diferentiation gives

$$
\left\{ \begin{array}{l} \nabla \ell_ {i} (\theta) = - \big (1 - \sigma (z _ {i} (\theta)) \big)   \Delta \phi_ {i}, \\ \nabla^ {2} \ell_ {i} (\theta) = \sigma (z _ {i} (\theta)) \big (1 - \sigma (z _ {i} (\theta)) \big)   \Delta \phi_ {i} \Delta \phi_ {i} ^ {\top}. \end{array} \right.
$$

Therefore,

$$
\nabla^ {2} L _ {n} (\theta) = \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma (z _ {i} (\theta)) \bigl (1 - \sigma (z _ {i} (\theta)) \bigr) \Delta \phi_ {i} \Delta \phi_ {i} ^ {\top}.
$$

Using the boundedness assumptions $\| \theta \| _ { 2 } \le R$ and $\| \phi ( x , a ) \| _ { 2 } \leq L$ , we have

$$
\left\{ \begin{array}{l} \| \Delta \phi_ {i} \| _ {2} \leq \| \phi (x _ {i}, a _ {i} ^ {w}) \| _ {2} + \| \phi (x _ {i}, a _ {i} ^ {l}) \| _ {2} \leq 2 L, \\ | z _ {i} (\theta) | = | \langle \theta , \Delta \phi_ {i} \rangle | \leq \| \theta \| _ {2}   \| \Delta \phi_ {i} \| _ {2} \leq 2 R L. \end{array} \right.
$$

Since $t \mapsto \sigma ( t ) ( 1 - \sigma ( t ) )$ is even and decreases on $[ 0 , \infty )$ , it follows that

$$
\sigma (z _ {i} (\theta)) \big (1 - \sigma (z _ {i} (\theta)) \big) \geq \sigma (2 R L) \big (1 - \sigma (2 R L) \big) =: c _ {R L}.
$$

Consequently,

$$
\nabla^ {2} L _ {n} (\theta) \succeq c _ {R L} \cdot \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \Delta \phi_ {i} \Delta \phi_ {i} ^ {\top}\right) \quad \text {   for   all   } \theta \in \Theta .\tag{10}
$$

Step 2. High-probability lower bound for the empirical Gram matrix. We use a matrix concentration inequality bound to show that the empirical Gram matrix is well-conditioned with high probability under Assumption 3. Define

$$
\left\{ \begin{array}{l} X _ {i} := \Delta \phi_ {i} \Delta \phi_ {i} ^ {\top}, \\ G _ {n} := \frac {1}{n} \sum_ {i = 1} ^ {n} X _ {i}. \end{array} \right.
$$

Then $X _ { i } \succeq 0$ and $\lambda _ { \operatorname* { m a x } } ( X _ { i } ) = \| \Delta \phi _ { i } \| _ { 2 } ^ { 2 } \leq ( 2 L ) ^ { 2 } = 4 L ^ { 2 }$ . By Assumption 3, we have

$$
\lambda_ {\min} \left(\mathbb {E} \left[ X _ {1} \right]\right) = \lambda_ {\min} \left(\mathbb {E} \left[ \Delta \phi \Delta \phi^ {\top} \right]\right) \geq \lambda .
$$

Thus,

$$
\mu_ {\min} := \lambda_ {\min} \left(\sum_ {i = 1} ^ {n} \mathbb {E} [ X _ {i} ]\right) = n \lambda_ {\min} \left(\mathbb {E} [ X _ {1} ]\right) \geq n \lambda .
$$

Applying Lemma 12 with $R = 4 L ^ { 2 }$ and $\begin{array} { r } { t = \frac { 1 } { 2 } } \end{array}$ gives

$$
\mathbb {P} \left(\lambda_ {\min} \left(\sum_ {i = 1} ^ {n} X _ {i}\right) \leq \frac {1}{2} \mu_ {\min}\right) \leq d \cdot \exp \left(- \frac {(1 / 2) ^ {2} \mu_ {\min}}{2 \cdot 4 L ^ {2}}\right) \leq d \cdot \exp \left(- \frac {n \lambda}{3 2 L ^ {2}}\right).
$$

Equivalently,

$$
\mathbb {P} \left(\lambda_ {\min} (G _ {n}) \leq \frac {\lambda}{2}\right) \leq d \cdot \exp \left(- \frac {n \lambda}{3 2 L ^ {2}}\right).
$$

Fix $\rho \in ( 0 , 1 )$ . If $\begin{array} { r } { n \ge \frac { 3 2 L ^ { 2 } } { \lambda } \log \left( \frac { d } { \rho } \right) } \end{array}$ , then the event

$$
\mathcal {E} := \left\{\lambda_ {\min} (G _ {n}) \geq \frac {\lambda}{2} \right\}
$$

satisfies $\mathbb { P } ( \mathcal { E } ) \geq 1 - \rho .$

Step 3. Strong convexity on E. We combine Steps 1–2 to conclude that $L _ { n }$ is µ-strongly convex on the high-probability event E.

On E, combining (10) with $\lambda _ { \operatorname* { m i n } } ( G _ { n } ) \geq \lambda / 2$ yields

$$
\nabla^ {2} L _ {n} (\theta) \succeq c _ {R L} G _ {n} \succeq c _ {R L} \frac {\lambda}{2} I _ {d} \quad \text { for   all } \theta \in \Theta .
$$

Therefore, on $\mathcal { E } ,$ the empirical objective $L _ { n }$ is µ-strongly convex over Θ with

$$
\mu = c _ {R L} \frac {\lambda}{2} = \frac {\lambda}{2} \sigma (2 R L) \big (1 - \sigma (2 R L) \big).
$$

Step 4. Lipschitz constant. We verify a uniform Lipschitz bound for the per-sample loss, which is required by the DP utility theorem.

For any $\theta \in \Theta$ 2

$$
\begin{array}{c} \| \nabla \ell_ {i} (\theta) \| _ {2} = \big (1 - \sigma (z _ {i} (\theta)) \big) \| \Delta \phi_ {i} \| _ {2} \\ \leq \| \Delta \phi_ {i} \| _ {2} \leq 2 L. \end{array}
$$

Hence each $\ell _ { i }$ is 2L-Lipschitz on $\Theta ,$ , and so is the average objective $L _ { n }$

Step 5. DP utility via Lemma 15. On the event $\mathcal { E } ,$ the objective $L _ { n }$ is µ-strongly convex

and 2L-Lipschitz on Θ. Applying Lemma 15 to the sum objective

$$
\sum_ {i = 1} ^ {n} \ell_ {i} (\theta) = n L _ {n} (\theta),
$$

and then dividing by n to convert back to the average objective yields

$$
\mathbb {E} \left[ L _ {n} \left(\theta_ {\text {priv}}\right) - L _ {n} (\hat {\theta}) \mid D \right] = O \left(\frac {d (2 L) ^ {2} \log^ {2} (n / \delta) \log (1 / \delta)}{n ^ {2} \mu \varepsilon^ {2}}\right),
$$

where ${ \hat { \theta } } = \arg \operatorname* { m i n } _ { \theta \in \Theta } L _ { n } ( \theta )$ and the expectation is over algorithmic randomness conditional on D. This completes the proof.

## D.2 Formal Justification of Remark 8

This subsection justifies Remark 8 by upgrading the conditional expected excess empirical risk control from Lemma 7 to a fully unconditional high-probability statement.

Theorem 11 (Unconditional high-probability excess empirical risk). Fix $\rho \in ( 0 , 1 )$ . Under the conditions in Lemma 7, there exists a numerical constant $C > 0$ such that, with probability at least $1 - \rho$ over the joint randomness of D and the learning procedure,

$$
L _ {n} (\theta_ {\mathrm{priv}}) - L _ {n} (\hat {\theta}) = \widetilde {O} \left(\frac {d   L ^ {2}}{\mu   n ^ {2} \varepsilon^ {2}}\right).\tag{11}
$$

Proof. We organize the proof into steps and explicitly separate the sources of randomness.

Step 1 (A high-probability data event for strong convexity and Lipschitzness). Apply Lemma 7 with failure probability $\rho / 3$ . Then there exists an event $\mathcal { E } _ { \mathrm { s c } }$ measurable w.r.t. D such that

$$
\mathbb {P} (\mathcal {E} _ {\mathrm{sc}}) \geq 1 - \rho / 3,\tag{12}
$$

and on ${ \mathcal E } _ { \mathrm { s c } }$ the empirical objective $L _ { n }$ is µ-strongly convex on $\Theta$ and satisfies the uniform gradient bound. For all $i \in \{ 1 , \ldots , n \}$ and all $\theta \in \Theta$

$$
\| \nabla \ell_ {i} (\theta) \| _ {2} \leq 2 L.\tag{13}
$$

Consequently, on $\mathcal { E } _ { \mathrm { s c } }$ we have the deterministic bounds, for all $\theta \in \Theta$ and all t,

$$
\| \nabla L _ {n} (\theta) \| _ {2} \leq \frac {1}{n} \sum_ {i = 1} ^ {n} \| \nabla \ell_ {i} (\theta) \| _ {2} \leq 2 L,\tag{14}
$$

and

$$
\| g _ {t} ^ {\mathrm{mb}} \| _ {2} \leq \frac {1}{m} \sum_ {j = 1} ^ {m} \| \nabla \ell_ {I _ {t, j}} (\theta_ {t}) \| _ {2} \leq 2 L.\tag{15}
$$

In particular,

$$
\| \nabla L _ {n} (\theta_ {t}) - g _ {t} ^ {\mathrm{mb}} \| _ {2} \leq \| \nabla L _ {n} (\theta_ {t}) \| _ {2} + \| g _ {t} ^ {\mathrm{mb}} \| _ {2} \leq 4 L.\tag{16}
$$

Step 2 (A high-probability bound for Gaussian magnitudes). Set $\alpha = \rho / 3$ . Under the DP-SGD instantiation analyzed here, the additive perturbation at each iteration is Gaussian, so we write $\xi _ { t } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { D P } } ^ { 2 } I _ { d } )$ . The exact calibration of $\sigma _ { \mathrm { D P } } ^ { 2 }$ depends on the specific DP-SGD variant, including the minibatch subsampling scheme and privacy accountant, and is not needed here. For our purposes, it sufices that the resulting Gaussian perturbation has variance scale $\begin{array} { r } { \sigma _ { \mathrm { D P } } ^ { 2 } = \widetilde { O } \left( \frac { L ^ { 2 } T } { n ^ { 2 } \varepsilon ^ { 2 } } \right) } \end{array}$ . Therefore, the random variable $\| \xi _ { t } \| _ { 2 } ^ { 2 } / \sigma _ { \mathrm { D P } } ^ { 2 }$ has a $\chi _ { d } ^ { 2 }$ distribution. By Lemma 14, for any $u > 0$

$$
\mathbb {P} \left(\| \xi_ {t} \| _ {2} ^ {2} \geq \sigma_ {\mathrm{DP}} ^ {2} (d + 2 \sqrt {d u} + 2 u)\right) \leq e ^ {- u}.\tag{17}
$$

Choose $\begin{array} { r } { u = \log \left( \frac { T } { \alpha } \right) } \end{array}$ , and define the deterministic radius

$$
B = \sigma_ {\mathrm{DP}} \sqrt {d + 2 \sqrt {d \log (T / \alpha)} + 2 \log (T / \alpha)}.\tag{18}
$$

Then for each $t ,$

$$
\mathbb {P} \left(\| \xi_ {t} \| _ {2} > B\right) \leq \alpha / T.\tag{19}
$$

A union bound over $t = 1 , \dots , T$ yields the event

$$
\mathcal {E} _ {\text { noise }} = \left\{\max _ {1 \leq t \leq T} \| \xi_ {t} \| _ {2} \leq B \right\},\tag{20}
$$

which satisfies $\begin{array} { r } { \mathbb { P } ( \mathcal { E } _ { \mathrm { n o i s e } } ) \ge 1 - \alpha = 1 - \rho / 3 . } \end{array}$

Step 3 (A coupling to replace unbounded perturbations by bounded ones). Define the truncated Gaussian law

$$
\nu_ {B} = \mathcal {L} (\xi_ {1} \mid \| \xi_ {1} \| _ {2} \leq B).\tag{21}
$$

Let $( \xi _ { t } ^ { \prime } ) _ { t \le T } ^ { }$ be i.i.d. with $\xi _ { t } ^ { \prime } \sim \nu _ { B }$ , independent of $( D , ( B _ { t } ) _ { t \leq T } , ( \xi _ { t } ) _ { t \leq T } )$ . Define the coupled sequence

$$
\tilde {\xi} _ {t} = \left\{ \begin{array}{l l} \xi_ {t}, & \text {if} \| \xi_ {t} \| _ {2} \leq B, \\ \xi_ {t} ^ {\prime}, & \text {if} \| \xi_ {t} \| _ {2} > B. \end{array} \right.\tag{22}
$$

By construction, we have $\| \tilde { \xi } _ { t } \| _ { 2 } \le B$ , almost surely for all $t ,$ and by symmetry of the Gaussian law and the centrally symmetric truncation set $\{ \| \xi \| _ { 2 } \leq B \} , \mathbb { E } [ \tilde { \xi } _ { t } ] = 0$ for all t.

Moreover, on $\mathcal { E } _ { \mathrm { n o i s e } }$ we have $\tilde { \xi } _ { t } = \xi _ { t }$ simultaneously for all $t \leq T$

Step 4 (Define the shadow process and couple iterates). Define a shadow iterate sequence

$( \tilde { \theta } _ { t } ) _ { t \leq T + 1 }$ using the same mini-batches $\left( B _ { t } \right)$ and the bounded noises $( \tilde { \xi } _ { t } )$ :

$$
\tilde {\theta} _ {t + 1} = \Pi_ {\Theta} \Big (\tilde {\theta} _ {t} - \eta_ {t} (\tilde {g} _ {t} ^ {\mathrm{mb}} + \tilde {\xi} _ {t}) \Big), \qquad \tilde {g} _ {t} ^ {\mathrm{mb}} = \frac {1}{m} \sum_ {j = 1} ^ {m} \nabla \ell_ {I _ {t, j}} (\tilde {\theta} _ {t}),\tag{23}
$$

initialized at $\tilde { \theta } _ { 1 } = \theta _ { 1 }$ and with the same step sizes $\eta _ { t } = 1 / ( \mu t )$ . On the event $\mathcal { E } _ { \mathrm { n o i s e } }$ , we have $\tilde { \xi } _ { t } = \xi _ { t }$ for all t and, by induction, for all $t = 1 , \dots , T + 1$

$$
\theta_ {t} = \tilde {\theta} _ {t}\tag{24}
$$

Step 5 (Verify the shadow recursion fits the bounded-oracle model). Fix an arbitrary dataset D and work on the event $\mathcal { E } _ { \mathrm { s c } }$ . Define the filtration

$$
\tilde {\mathcal {F}} _ {t - 1} = \sigma (D, B _ {1}, \tilde {\xi} _ {1}, \dots , B _ {t - 1}, \tilde {\xi} _ {t - 1}),\tag{25}
$$

and define the oracle noise term

$$
\tilde {z} _ {t} = \nabla L _ {n} (\tilde {\theta} _ {t}) - \big (\tilde {g} _ {t} ^ {\mathrm{mb}} + \tilde {\xi} _ {t} \big).\tag{26}
$$

(i) Conditional mean-zero. Conditional on $\tilde { \mathcal { F } } _ { t - 1 }$ , the iterate $\tilde { \theta } _ { t }$ is fixed. Using (23) and the fact that each $I _ { t , j }$ is uniform on $\{ 1 , \ldots , n \}$ ,

$$
\begin{array}{r l} & {\mathbb {E} \Big [ \tilde {g} _ {t} ^ {\mathrm{mb}} \Big | \tilde {\mathcal {F}} _ {t - 1} \Big ] = \frac {1}{m} \sum_ {j = 1} ^ {m} \mathbb {E} \Big [ \nabla \ell_ {I _ {t, j}} (\tilde {\theta} _ {t}) \Big | \tilde {\mathcal {F}} _ {t - 1} \Big ]} \\ & {\qquad = \mathbb {E} \Big [ \nabla \ell_ {I _ {t, 1}} (\tilde {\theta} _ {t}) \Big | \tilde {\mathcal {F}} _ {t - 1} \Big ]} \\ & {\qquad = \frac {1}{n} \sum_ {i = 1} ^ {n} \nabla \ell_ {i} (\tilde {\theta} _ {t}) = \nabla L _ {n} (\tilde {\theta} _ {t}).} \end{array}\tag{27}
$$

Also, $\tilde { \xi } _ { t }$ is independent of $\tilde { \mathcal { F } } _ { t - 1 }$ and has mean zero. Therefore, taking conditional expectations in (26) gives

$$
\mathbb {E} \left[ \tilde {z} _ {t} \mid \tilde {\mathcal {F}} _ {t - 1} \right] = 0.\tag{28}
$$

(ii) Almost-sure bound. On $\mathcal { E } _ { \mathrm { s c } }$ , combining (16) (applied at $\tilde { \theta } _ { t } )$ with (26) yields

$$
\| \tilde {z} _ {t} \| _ {2} \leq \| \nabla L _ {n} (\tilde {\theta} _ {t}) - \tilde {g} _ {t} ^ {\mathrm{mb}} \| _ {2} + \| \tilde {\xi} _ {t} \| _ {2} \leq 4 L + B \quad \text { almost   surely   for   all } t.\tag{29}
$$

Step 6 Invoke Lemma 13). On $\mathcal { E } _ { \mathrm { s c } }$ , the function $L _ { n }$ is µ-strongly convex on Θ and is $2 L _ { - }$ Lipschitz by (14). Moreover, (28) and (29) verify the bounded-oracle conditions with noise radius $Z = 4 L + B$ . Invoke Lemma 13 with failure probability $\alpha = \rho / 3$ . Then, on $\mathcal { E } _ { \mathrm { s c } }$ , with probability at least $1 - \rho / 3$ over the shadow algorithmic randomness,

$$
L _ {n} (\tilde {\theta} _ {T + 1}) - L _ {n} (\hat {\theta}) \leq \widetilde {O} \left(\frac {(L + B) ^ {2}}{\mu T}\right).\tag{30}
$$

Here we absorbed all absolute constants and the log T and log $( 1 / \rho )$ factors into ${ \widetilde { O } } ( \cdot )$

Step 7 (Transfer to the real iterate and make the probability unconditional). On $\mathcal { E } _ { \mathrm { n o i s e } } .$ , the coupling (24) implies $\theta _ { T + 1 } = \tilde { \theta } _ { T + 1 }$ , hence (30) holds with $\tilde { \theta } _ { T + 1 }$ replaced by $\theta _ { T + 1 }$ on $\mathcal { E } _ { \mathrm { s c } } \cap \mathcal { E } _ { \mathrm { n o i s e } }$ . Therefore, combining the three failure probabilities,

$$
\mathbb {P} (\mathcal {E} _ {\mathrm{sc}} ^ {c}) \leq \rho / 3, \qquad \mathbb {P} (\mathcal {E} _ {\mathrm{noise}} ^ {c}) \leq \rho / 3, \qquad \mathbb {P} (\text { Harvey   failure   on } \mathcal {E} _ {\mathrm{sc}}) \leq \rho / 3,\tag{31}
$$

a union bound yields that (30) (with $\tilde { \theta } _ { T + 1 }$ replaced by $\theta _ { T + 1 } )$ holds with overall probability at least $1 - \rho$ over the joint randomness of D and the learning procedure.

Step 8 (Rate-level simplification and DP calibration). We now simplify the right-hand side of (30). From (18) and the inequality $2 { \sqrt { d u } } \leq d +$ u applied with $u = \log ( T / \alpha )$ , we obtain

$$
B ^ {2} \leq \sigma_ {\mathrm{DP}} ^ {2} \widetilde {O} (d).\tag{32}
$$

Substituting $\begin{array} { r } { \sigma _ { \mathrm { D P } } ^ { 2 } = \widetilde { O } \left( \frac { L ^ { 2 } T } { n ^ { 2 } \varepsilon ^ { 2 } } \right) } \end{array}$ into (32) gives

$$
\frac {B ^ {2}}{T} = \widetilde {O} \left(\frac {d L ^ {2}}{n ^ {2} \varepsilon^ {2}}\right).\tag{33}
$$

Plugging (33) into (30) yields, with probability at least $1 - \rho _ { ; }$

$$
L _ {n} (\theta_ {T + 1}) - L _ {n} (\hat {\theta}) \leq \widetilde {O} \left(\frac {L ^ {2}}{\mu T}\right) + \widetilde {O} \left(\frac {d L ^ {2}}{\mu n ^ {2} \varepsilon^ {2}}\right).\tag{34}
$$

Choosing T suficiently large (as in our main procedure) makes the optimization term ${ \widetilde O } ( L ^ { 2 } / ( \mu T ) )$ ) negligible at the rate level. Thus we obtain (11), completing the proof. □

## D.3 Proof of Theorem 9

This section provides the proof of the suboptimality gap for the proposed framework. The argument proceeds in two steps. We first relate the KL-regularized policy suboptimality gap to a reward estimation error term, and then use the coverage assumption to transfer this control to the reference policy π . This proof strategy follows a line of analysis suggested in Zhao et al. (2024). In our setting, it is combined with the private reward-estimation bound established above.

Step 1. Set-up: suboptimality-gap decomposition via a KL-regularized functional. Recall the KL-regularized value of a policy $\pi$ (reference $\pi _ { 0 }$ , temperature $\eta > 0 )$ :

$$
Q (\pi) = \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi (a \mid x) \left[ r _ {\theta^ {*}} (x, a) - \frac {1}{\eta} \log \frac {\pi (a \mid x)}{\pi_ {0} (a \mid x)} \right].
$$

Our goal is to control the suboptimality gap

$$
Q (\pi_ {\theta^ {*}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}),
$$

where $\pi _ { \boldsymbol { \theta } } ^ { \eta }$ denotes the Boltzmann policy induced by the reward $r _ { \theta } ( \cdot , \cdot )$

$$
\pi_ {\theta} ^ {\eta} (a \mid x) = \frac {\pi_ {0} (a \mid x) \exp (\eta r _ {\theta} (x , a))}{Z _ {\theta} ^ {\eta} (x)},
$$

where $\begin{array} { r } { Z _ { \theta } ^ { \eta } ( x ) : = \sum _ { a \in \mathcal { A } } \pi _ { 0 } ( a  { \mathrm { ~ | ~ } } x ) \exp \bigl ( \eta r _ { \theta } ( x , a ) \bigr ) } \end{array}$ . Using the identity

$$
\log \frac {\pi_ {\theta} ^ {\eta} (a \mid x)}{\pi_ {0} (a \mid x)} = \eta r _ {\theta} (x, a) - \log Z _ {\theta} ^ {\eta} (x),
$$

we can rewrite $Q ( \pi _ { \theta } ^ { \eta } )$ as

$$
\begin{array}{l} Q (\pi_ {\theta} ^ {\eta}) = \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {\theta} ^ {\eta} (a \mid x) \left[ r _ {\theta^ {*}} (x, a) - \frac {1}{\eta} \Big (\eta r _ {\theta} (x, a) - \log Z _ {\theta} ^ {\eta} (x) \Big) \right] \\ = \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {\theta} ^ {\eta} (a \mid x) \Big [ r _ {\theta^ {*}} (x, a) - r _ {\theta} (x, a) \Big ] + \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ \log Z _ {\theta} ^ {\eta} (x) \Big ]. \end{array}
$$

Therefore,

$$
\begin{array}{r l} & Q (\pi_ {\theta^ {*}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}) = \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ \log Z _ {\theta^ {*}} ^ {\eta} (x) - \log Z _ {\tilde {\theta}} ^ {\eta} (x) \Big ] \\ & \qquad - \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {\tilde {\theta}} ^ {\eta} (a \mid x) \Big (r _ {\theta^ {*}} (x, a) - r _ {\tilde {\theta}} (x, a) \Big). \end{array}\tag{35}
$$

To package the two terms in (35) into a single contextwise functional, let $f : \mathcal { X } \times \mathcal { A } $ R be arbitrary and define

$$
\pi_ {f} ^ {\eta} (a \mid x) = \frac {\pi_ {0} (a \mid x) \exp (\eta f (x , a))}{Z _ {f} ^ {\eta} (x)},
$$

with $\begin{array} { r } { Z _ { f } ^ { \eta } ( x ) : = \sum _ { a \in \mathcal { A } } \pi _ { 0 } ( a \mathbin { \mid } x ) \exp \bigl ( \eta f ( x , a ) \bigr ) } \end{array}$ , and define the reward diference

$$
\Delta_ {f} (x, a) := f (x, a) - r _ {\theta^ {*}} (x, a).
$$

We set

$$
J \big (f (x, \cdot) \big) := \log Z _ {f} ^ {\eta} (x) - \eta \sum_ {a \in \mathcal {A}} \pi_ {f} ^ {\eta} (a \mid x) \Delta_ {f} (x, a).\tag{36}
$$

Then for any $f ,$

$$
\begin{array}{r l} & Q (\pi_ {f} ^ {\eta}) = \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {f} ^ {\eta} (a \mid x) \left[ r _ {\theta^ {*}} (x, a) - \frac {1}{\eta} \log \frac {\pi_ {f} ^ {\eta} (a \mid x)}{\pi_ {0} (a \mid x)} \right] \\ & \qquad = \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {f} ^ {\eta} (a \mid x) \left[ r _ {\theta^ {*}} (x, a) - \frac {1}{\eta} \Big (\eta f (x, a) - \log Z _ {f} ^ {\eta} (x) \Big) \right] \\ & \qquad = \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {f} ^ {\eta} (a \mid x) \Big (r _ {\theta^ {*}} (x, a) - f (x, a) \Big) + \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ \log Z _ {f} ^ {\eta} (x) \Big ] \\ & \qquad = - \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {f} ^ {\eta} (a \mid x)   \Delta_ {f} (x, a) + \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ \log Z _ {f} ^ {\eta} (x) \Big ] \\ & \qquad = \frac {1}{\eta}   \mathbb {E} _ {x \sim d _ {0}} \Big [ J (f (x, \cdot)) \Big ]. \end{array}
$$

Thus,

$$
Q (\pi_ {f} ^ {\eta}) = \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ J \big (f (x, \cdot) \big) \Big ].\tag{37}
$$

In particular, taking $f ( x , a ) = r _ { \theta } ( x , a )$ gives

$$
Q (\pi_ {\theta} ^ {\eta}) = \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ J (r _ {\theta} (x, \cdot)) \Big ],
$$

and therefore

$$
Q (\pi_ {\theta^ {*}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}) = \frac {1}{\eta} \mathbb {E} _ {x \sim d _ {0}} \Big [ J \big (r _ {\theta^ {*}} (x, \cdot) \big) - J \big (r _ {\tilde {\theta}} (x, \cdot) \big) \Big ].
$$

Step 2. From functional gap to a squared reward error. In this step we upper bound the functional gap

$$
J \big (r _ {\theta^ {*}} (x, \cdot) \big) - J \big (r _ {\tilde {\theta}} (x, \cdot) \big)
$$

by a quadratic reward-discrepancy term. We interpolate linearly between $r _ { \theta ^ { \ast } } ( x , \cdot )$ and $r _ { \tilde { \theta } } ( x , \cdot )$ , diferentiate J along this one-dimensional path, and use softmax log-partition calculus to express the derivative as a variance. This yields a second-moment bound under an interpolating Boltzmann policy.

To this end, fix $x \in \mathcal { X }$ and define the pointwise reward error

$$
\Delta_ {x} (a) := r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {*}} (x, a), \quad a \in \mathcal {A}.
$$

Consider the one-dimensional interpolation, that is, for $t \in [ 0 , 1 ]$ 2

$$
f _ {t} (x, a) := r _ {\theta^ {*}} (x, a) + t \Delta_ {x} (a) = (1 - t) r _ {\theta^ {*}} (x, a) + t r _ {\tilde {\theta}} (x, a).
$$

Let $\pi _ { t } ^ { \eta } ( \cdot \mid x ) : = \pi _ { f _ { t } } ^ { \eta } ( \cdot \mid x )$ and $Z _ { t } ^ { \eta } ( x ) : = Z _ { f _ { t } } ^ { \eta } ( x )$ . Define

$$
\psi_ {x} (t) := J \bigl (f _ {t} (x, \cdot) \bigr) = \log Z _ {t} ^ {\eta} (x) - \eta \sum_ {a \in \mathcal {A}} \pi_ {t} ^ {\eta} (a \mid x) \left(t \Delta_ {x} (a)\right).
$$

Step $\mathbf { 2 ( a ) }$ . A derivative identity for $\psi _ { x } ^ { \prime } ( t )$ . Write $m _ { x } ( t ) : = \mathbb { E } _ { a \sim \pi _ { \star } ^ { \eta } ( \cdot | x ) } [ \Delta _ { x } ( a ) ]$ . Then

$$
\begin{array}{r l} & {\frac {d}{d t} \log Z _ {t} ^ {\eta} (x) = \frac {1}{Z _ {t} ^ {\eta} (x)} \sum_ {a \in \mathcal {A}} \pi_ {0} (a \mid x) \exp (\eta f _ {t} (x, a)) \cdot \eta \Delta_ {x} (a)} \\ & {\qquad = \eta \sum_ {a \in \mathcal {A}} \pi_ {t} ^ {\eta} (a \mid x) \Delta_ {x} (a) = \eta m _ {x} (t).} \end{array}
$$

Moreover, as $\psi _ { x } ( t ) = \log Z _ { t } ^ { \eta } ( x ) ~ - ~ \eta t m _ { x } ( t )$ , we have

$$
\begin{array}{r l} & {\psi_ {x} ^ {\prime} (t) = \eta m _ {x} (t) - \eta m _ {x} (t) - \eta t m _ {x} ^ {\prime} (t)} \\ & {\qquad = - \eta t m _ {x} ^ {\prime} (t).} \end{array}
$$

Next, diferentiate $\begin{array} { r } { m _ { x } ( t ) = \sum _ { a } \pi _ { t } ^ { \eta } ( a \mid x ) \Delta _ { x } ( a ) } \end{array}$

$$
m _ {x} ^ {\prime} (t) = \sum_ {a \in \mathcal {A}} \Delta_ {x} (a) \frac {d}{d t} \pi_ {t} ^ {\eta} (a \mid x).
$$

Since

$$
\pi_ {t} ^ {\eta} (a \mid x) = \frac {\pi_ {0} (a \mid x) \exp (\eta f _ {t} (x , a))}{Z _ {t} ^ {\eta} (x)},
$$

we have

$$
\begin{array}{c} \frac {d}{d t} \pi_ {t} ^ {\eta} (a \mid x) = \pi_ {t} ^ {\eta} (a \mid x) \cdot \eta   \Delta_ {x} (a) - \pi_ {t} ^ {\eta} (a \mid x) \cdot \frac {d}{d t} \log Z _ {t} ^ {\eta} (x) \\ = \eta   \pi_ {t} ^ {\eta} (a \mid x) \Big (\Delta_ {x} (a) - m _ {x} (t) \Big). \end{array}
$$

Therefore,

$$
\begin{array}{l} m _ {x} ^ {\prime} (t) = \eta \sum_ {a \in \mathcal {A}} \pi_ {t} ^ {\eta} (a \mid x)   \Delta_ {x} (a) \Big (\Delta_ {x} (a) - m _ {x} (t) \Big) \\ \qquad = \eta \sum_ {a \in \mathcal {A}} \pi_ {t} ^ {\eta} (a \mid x) \Big (\Delta_ {x} (a) ^ {2} - \Delta_ {x} (a)   m _ {x} (t) \Big) \\ \qquad = \eta \left(\mathbb {E} _ {a \sim \pi_ {t} ^ {\eta} (\cdot | x)} [ \Delta_ {x} (a) ^ {2} ] - m _ {x} (t) ^ {2}\right) \\ \qquad = \eta   \mathrm{Var} _ {a \sim \pi_ {t} ^ {\eta} (\cdot | x)} (\Delta_ {x} (a)). \end{array}
$$

Plugging into $\psi _ { x } ^ { \prime } ( t ) = - \eta t m _ { x } ^ { \prime } ( t )$ gives

$$
\psi_ {x} ^ {\prime} (t) = - \eta^ {2} t \operatorname{Var} _ {a \sim \pi_ {t} ^ {\eta} (\cdot | x)} (\Delta_ {x} (a)).\tag{38}
$$

Step $\mathbf { 2 ( b ) }$ . Mean-value bound and a squared-error control. For any $t \in [ 0 , 1 ]$ , define the scalar function

$$
G (t) := \mathbb {E} _ {x \sim d _ {0}} [ \psi_ {x} (t) ] = \mathbb {E} _ {x \sim d _ {0}} \left[ J \left(f _ {t} (x, \cdot)\right) \right],
$$

where $\psi _ { x } ( t ) = J ( f _ { t } ( x , \cdot ) )$ is defined above. By (38), we have

$$
\psi_ {x} ^ {\prime} (t) = - \eta^ {2} t \operatorname{Var} _ {a \sim \pi_ {t} ^ {\eta} (\cdot | x)} (\Delta_ {x} (a)).
$$

Diferentiating under the expectation yields

$$
G ^ {\prime} (t) = \mathbb {E} _ {x \sim d _ {0}} \big [ \psi_ {x} ^ {\prime} (t) \big ] = - \eta^ {2} t \mathbb {E} _ {x \sim d _ {0}} \Big [ \mathrm{Var} _ {a \sim \pi_ {t} ^ {\eta} (\cdot | x)} \big (\Delta_ {x} (a) \big) \Big ].
$$

Applying the mean value theorem to the scalar function $G$ on $[ 0 , 1 ]$ , there exists a $\gamma \in \left( 0 , 1 \right)$ such that

$$
G (0) - G (1) = - G ^ {\prime} (\gamma) = \eta^ {2} \gamma \mathbb {E} _ {x \sim d _ {0}} \Big [ \operatorname{Var} _ {a \sim \pi_ {\gamma} ^ {\eta} (\cdot | x)} \big (\Delta_ {x} (a) \big) \Big ].
$$

Using $\operatorname { V a r } ( Y ) \leq \mathbb { E } [ Y ^ { 2 } ]$ and $\gamma \leq 1$ , we obtain

$$
G (0) - G (1) \leq \eta^ {2} \mathbb {E} _ {x \sim d _ {0}, a \sim \pi_ {\gamma} ^ {\eta} (\cdot | x)} [ \Delta_ {x} (a) ^ {2} ].
$$

Define the interpolating reward $f : = f _ { \gamma }$ γ

$$
f (x, a) := (1 - \gamma) r _ {\theta^ {*}} (x, a) + \gamma r _ {\tilde {\theta}} (x, a),
$$

so that $\pi _ { \gamma } ^ { \eta } ( \cdot \mid x ) = \pi _ { f } ^ { \eta } ( \cdot \mid x )$

Step 2(c). Consequence for the suboptimality gap. Recalling (37), we have

$$
Q (\pi_ {\theta^ {*}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}) = \frac {1}{\eta} \Big (G (0) - G (1) \Big).
$$

Combining with the bound in Step 2(b) yields

$$
Q (\pi_ {\theta^ {*}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}) \leq \eta   \mathbb {E} _ {x \sim d _ {0}, a \sim \pi_ {f} ^ {\eta} (\cdot | x)} \Big [ (r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {*}} (x, a)) ^ {2} \Big ].\tag{39}
$$

The key implication of (39) is that controlling the policy suboptimality $\mathrm { g a p }$ is reduced to controlling a reward estimation error term under an interpolating policy.

Step 3 (Change of measure under point-wise coverage). For the reward function f in Step 2 and the induced policy $\pi _ { f } ^ { \eta } \in \Pi$ , define

$$
w _ {f} (x, a) := \frac {\pi_ {f} ^ {\eta} (a \mid x)}{\pi_ {0} (a \mid x)},
$$

with the convention $0 / 0 = 0$ . By Assumption 4, for any $( x , a )$ with $d _ { 0 } ( x ) > 0$

$$
0 \leq w _ {f} (x, a) \leq C.
$$

Therefore,

$$
\begin{array}{r l} & {\mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {f} ^ {\eta} (a \mid x) \Big \{r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {*}} (x, a) \Big \} ^ {2}} \\ & {= \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {0} (a \mid x) w _ {f} (x, a) \Big \{r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {*}} (x, a) \Big \} ^ {2}} \\ & {\leq C \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {0} (a \mid x) \Big \{r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {*}} (x, a) \Big \} ^ {2}.} \end{array}
$$

Plugging this bound into (39) yields

$$
Q (\pi_ {\theta^ {\star}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}) \leq \eta   C   \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {0} (a \mid x) \Bigl \{r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {\star}} (x, a) \Bigr \} ^ {2}.
$$

Step 4 (Bound the π -reward MSE by statistical + DP errors). By Assumption 2, for any $( x , a )$ , we have

$$
r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {\star}} (x, a) = \langle \tilde {\theta} - \theta^ {\star}, \phi (x, a) \rangle .
$$

Hence,

$$
\begin{array}{r l} & {\mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {0} (a \mid x) \Bigl \{r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {\star}} (x, a) \Bigr \} ^ {2}} \\ & {= \mathbb {E} _ {x \sim d _ {0}} \mathbb {E} _ {a \sim \pi_ {0} (\cdot | x)} \Bigl [ \langle \tilde {\theta} - \theta^ {\star}, \phi (x, a) \rangle \Bigr ] ^ {2}} \\ & {\leq \mathbb {E} _ {x \sim d _ {0}} \mathbb {E} _ {a \sim \pi_ {0} (\cdot | x)} \Bigl [ \| \tilde {\theta} - \theta^ {\star} \| _ {2} ^ {2} \| \phi (x, a) \| _ {2} ^ {2} \Bigr ]} \\ & {\leq \left(\sup _ {x, a} \| \phi (x, a) \| _ {2} ^ {2}\right) \| \tilde {\theta} - \theta^ {\star} \| _ {2} ^ {2}.} \end{array}
$$

Let $\hat { \theta }$ denote the non-private MLE. Using the quadratic inequality,

$$
\begin{array}{c} \| \tilde {\theta} - \theta^ {\star} \| _ {2} ^ {2} = \| (\tilde {\theta} - \hat {\theta}) + (\hat {\theta} - \theta^ {\star}) \| _ {2} ^ {2} \\ \leq 2 \| \tilde {\theta} - \hat {\theta} \| _ {2} ^ {2} + 2 \| \hat {\theta} - \theta^ {\star} \| _ {2} ^ {2}. \end{array}
$$

Statistical term. Invoke Lemma 16 with failure probability $\rho / 2$ . With probability at least $1 - \rho / 2$ (over the data randomness),

$$
\| \hat {\theta} - \theta^ {\star} \| _ {2} ^ {2} \leq O \left(\frac {d + \log (2 / \rho)}{n}\right).
$$

DP term (high-probability, all randomness). We now control $\lVert \tilde { { \boldsymbol { \theta } } } - \hat { { \boldsymbol { \theta } } } \rVert _ { 2 } ^ { 2 }$ in high probability over both the data randomness and the algorithmic randomness. Invoke Theorem 11 with failure probability $\rho / 2$ . With probability at least $1 - \rho / 2$ , we have the excess empirical risk bound

$$
\bar {L} _ {n} (\tilde {\theta}) - \bar {L} _ {n} (\hat {\theta}) \leq \widetilde {O} \left(\frac {d L ^ {2}}{\mu n ^ {2} \varepsilon^ {2}}\right),
$$

ignoring polylogarithmic factors in $( n , 1 / \delta , 1 / \rho )$

On the same event, by µ-strong convexity of ${ \bar { L } } _ { n }$ on $\Theta$ ,

$$
\bar {L} _ {n} (\tilde {\theta}) - \bar {L} _ {n} (\hat {\theta}) \geq \frac {\mu}{2} \| \tilde {\theta} - \hat {\theta} \| _ {2} ^ {2},
$$

which implies

$$
\| \tilde {\theta} - \hat {\theta} \| _ {2} ^ {2} \leq \frac {2}{\mu} \left\{\bar {L} _ {n} (\tilde {\theta}) - \bar {L} _ {n} (\hat {\theta}) \right\} \leq \widetilde {O} \left(\frac {d L ^ {2}}{\mu^ {2} n ^ {2} \varepsilon^ {2}}\right).
$$

Combine. By a union bound over the statistical event and the DP event, with probability at least $1 - \rho$ (over both the data randomness and the algorithmic randomness),

$$
\begin{array}{l} \| \tilde {\theta} - \theta^ {\star} \| _ {2} ^ {2} \leq 2 \| \tilde {\theta} - \hat {\theta} \| _ {2} ^ {2} + 2 \| \hat {\theta} - \theta^ {\star} \| _ {2} ^ {2} \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

Consequently, with the same probability at least $1 - \rho ,$

$$
\begin{array}{l} \mathbb {E} _ {x \sim d _ {0}} \sum_ {a \in \mathcal {A}} \pi_ {0} (a \mid x) \Bigl \{r _ {\tilde {\theta}} (x, a) - r _ {\theta^ {\star}} (x, a) \Bigr \} ^ {2} \\ \leq \left(\sup _ {x, a} \| \phi (x, a) \| _ {2} ^ {2}\right) \| \tilde {\theta} - \theta^ {\star} \| _ {2} ^ {2} \\ \leq \left(\sup _ {x, a} \| \phi (x, a) \| _ {2} ^ {2}\right) \left\{O \bigg (\frac {d + \log (2 / \rho)}{n} \bigg) + \widetilde {O} \bigg (\frac {d   L ^ {2}}{\mu^ {2}   n ^ {2} \varepsilon^ {2}} \bigg) \right\}. \end{array}
$$

Conclusion. Combining Step 3 and Step 4 gives, with probability at least $1 - \rho$

$$
Q (\pi_ {\theta^ {*}} ^ {\eta}) - Q (\pi_ {\tilde {\theta}} ^ {\eta}) \leq \eta   C \left(\sup _ {x, a} \| \phi (x, a) \| _ {2} ^ {2}\right) \left\{O \bigg (\frac {d + \log (2 / \rho)}{n} \bigg) + \widetilde {O} \bigg (\frac {d   L ^ {2}}{\mu^ {2}   n ^ {2} \varepsilon^ {2}} \bigg) \right\}.
$$

## D.4 Proof of Theorem 10

In this section, we provide the rigorous proof of Theorem 10. Our analysis decomposes the lower bound into two fundamental barriers: the statistical complexity inherent to the reward class and the information-theoretic limit imposed by DP.

Step 1. Non-private minimax lower bound. We establish a non-private baseline by inverting the sample-complexity lower bound of Zhao et al. (2024). Throughout this step, “uniformly guarantees” means

$$
\sup _ {\theta^ {\star} \in \Theta} \mathbb {E} _ {\theta^ {\star}} [ \mathrm{Gap} (\hat {\pi}; \theta^ {\star}) ] \leq \mu ,\tag{40}
$$

where the expectation is over the algorithm’s randomness and the data generated under $\theta ^ { \star }$

Fast-rate (local-curvature) branch. For KL-regularized objectives, Proposition 17 implies that any algorithm satisfying (40) must have

$$
n \geq c _ {0} \min \biggl \{\frac {\eta \log \mathcal {N} _ {\mathcal {R}} (\mu)}{\mu}, \frac {\log \mathcal {N} _ {\mathcal {R}} (\mu)}{\mu^ {2}} \biggr \},\tag{41}
$$

for a universal constant $c _ { 0 } > 0$ , where $\mathcal { N } _ { \mathcal { R } } ( \mu )$ denotes the µ-covering number of $\mathcal { R }$ . Fix a universal constant $\bar { c } \in ( 0 , 1 ) ~ ( \mathrm { e . g . , } ~ \bar { c } = 1 / 6 4$ in Zhao et al. (2024)) and restrict to target gaps $\mu$ such that

$$
\eta \mu \leq \bar {c}.\tag{42}
$$

Then the first term in (41) is the active branch: indeed,

$$
\frac {\log \mathcal {N} _ {\mathcal {R}} (\mu) / \mu^ {2}}{\eta \log \mathcal {N} _ {\mathcal {R}} (\mu) / \mu} = \frac {1}{\eta \mu} \geq \frac {1}{\bar {c}},
$$

so log $\mathcal { N _ { R } } ( \mu ) / \mu ^ { 2 } \geq$ (η log $\mathcal { N } _ { \mathcal { R } } ( \mu ) ) / \mu$ . Hence (41) simplifies to

$$
n \geq c _ {0} \frac {\eta \log \mathcal {N} _ {\mathcal {R}} (\mu)}{\mu}, \quad \text { whenever } \eta \mu \leq \bar {c}.\tag{43}
$$

Recall that for the d-dimensional linear reward class considered in this work, standard volumetric arguments yield log $\mathcal { N } _ { \mathcal { R } } ( \mu ) \asymp d \log ( 1 / \mu )$ (up to universal constants). In particular, fixing any $\mu _ { \star } \in ( 0 , 1 )$ and restricting to $\mu \in ( 0 , \mu _ { \star } ]$ allows us to absorb the factor log $( 1 / \mu )$ into constants, so there exist constants $c _ { \mathrm { e n t } } > 0$ and $\mu _ { \star } \in ( 0 , 1 )$ such that

$$
\log \mathcal {N} _ {\mathcal {R}} (\mu) \geq c _ {\text {ent}} d, \quad \text {for all} \mu \in (0, \mu_ {\star} ].\tag{44}
$$

Fix

$$
\left\{ \begin{array}{l} \kappa := \frac {c _ {0} c _ {\mathrm{ent}} \eta}{2}, \\ \mu_ {n} := \kappa   \frac {d}{n}. \end{array} \right.
$$

Define the non-private threshold

$$
n _ {\mathrm{np}} := \max \left\{\frac {\kappa d}{\mu_ {\star}}, \frac {\eta \kappa d}{\bar {c}} \right\} = \max \left\{\frac {c _ {0} c _ {\text {ent}} \eta}{2 \mu_ {\star}} d, \frac {c _ {0} c _ {\text {ent}} \eta^ {2}}{2 \bar {c}} d \right\}.\tag{45}
$$

For any $n \geq n _ { \mathrm { n p } }$ , we have $\mu _ { n } \leq \mu _ { \star }$ and $\eta \mu _ { n } \leq { \bar { c } } ,$ , so that (44) and (43) apply at $\mu = \mu _ { n }$

Suppose, toward a contradiction, that there exists an algorithm satisfying $\begin{array} { r } { \operatorname* { s u p } _ { \theta ^ { \star } \in \Theta } \mathbb { E } _ { \theta ^ { \star } } [ \mathrm { G a p } ( \hat { \pi } ; \theta ^ { \star } ) ] \le } \end{array}$ $\mu _ { n }$ . Then (43) and (44) yield

$$
n \geq c _ {0} \frac {\eta \log \mathcal {N} _ {\mathcal {R}} (\mu_ {n})}{\mu_ {n}} \geq c _ {0} \frac {\eta (c _ {\mathrm{ent}} d)}{\kappa d / n} = c _ {0} \frac {\eta c _ {\mathrm{ent}}}{\kappa} n = 2 n,
$$

which is a contradiction. Therefore, no algorithm can satisfy (40) with $\mu = \mu _ { n } ,$ and hence

$$
R _ {n} ^ {\mathrm{np}} := \inf _ {\hat {\pi}} \sup _ {\theta^ {\star} \in \Theta} \mathbb {E} _ {\theta^ {\star}} [ \operatorname{Gap} (\hat {\pi}; \theta^ {\star}) ] \geq \mu_ {n} = \kappa \frac {d}{n} \gtrsim \frac {d}{n},\tag{46}
$$

for all $n \geq n _ { \mathrm { n p } } .$

Since the class of $( \varepsilon , \delta ) – \mathrm { D P }$ algorithms is a subset of all randomized algorithms,

$$
R _ {n} (\varepsilon , \delta) \geq R _ {n} ^ {\mathrm{np}}.
$$

Step 2. Hard instance construction. We instantiate the environment so that the context features align with the canonical basis of $\mathbb { R } ^ { d }$ , which ensures that informative observations occur with probability $1 / d .$ This makes the efective distinguishability scale as $n / d ,$ which is the mechanism by which the dimension d enters the privacy lower bound.

Concretely, we take a finite context space $\mathcal { X } = \{ x _ { 0 } , x _ { 1 } , . . . , x _ { d - 1 } \}$ with the uniform distribu tion $\rho ( x ) = 1 / d$ . Since a minimax lower bound only requires exhibiting one instance within the model class, this choice is fully admissible and directly captures the intrinsic dificulty induced by the ambient dimension d.

Let the action space be $\mathcal { A } = \{ 0 , 1 \}$ . We consider a d-dimensional linear reward class with feature map $\phi : \mathcal { X } \times \mathcal { A }  \mathbb { R } ^ { d } \mathrm { : }$ : set $\phi ( x _ { 0 } , 1 ) = e _ { 1 } , \phi ( x _ { 0 } , 0 ) = 0$ , and for all $x \neq x _ { 0 }$ set $\phi ( x , 1 ) =$ $\phi ( x , 0 ) = 0$ . For a signal level $c > 0$ , define two parameters

$$
\left\{ \begin{array}{l} \theta_ {+} := c   e _ {1}, \\ \theta_ {-} := - c   e _ {1}. \end{array} \right.
$$

Note that this is a valid hard instance within the d-dimensional linear reward class: the feature map $\phi ( \cdot , \cdot )$ takes values in $\mathbb { R } ^ { d }$ and both $\theta _ { + } = c e _ { 1 }$ and $\theta _ { - } ~ = - c e _ { 1 }$ belong to the admissible parameter set (hence we are lower bounding the minimax risk over a subclass of $\mathcal { R } )$ ).

Then the induced reward diference at the informative context is

$$
\left\{ \begin{array}{l} r _ {\theta_ {+}} (x _ {0}, 1) - r _ {\theta_ {+}} (x _ {0}, 0) = c, \\ r _ {\theta_ {-}} (x _ {0}, 1) - r _ {\theta_ {-}} (x _ {0}, 0) = - c, \end{array} \right.
$$

while all other contexts are deliberately uninformative. In particular, the data distributions $P _ { \theta _ { + } }$ and $P _ { \theta . }$ coincide on $\{ X \neq x _ { 0 } \}$ , and the two instances difer only through the rare event $\{ X = x _ { 0 } \}$ which occurs with probability $1 / d .$

We generate pairwise preference labels from a Bradley–Terry model.

Step 3. From the gap to a DP testing problem. Fix any $( \varepsilon , \delta ) – \mathrm { D P }$ algorithm A and let $\hat { \pi } = \mathcal { A } ( D )$ be its output policy. We first rewrite the KL-regularized suboptimality gap as a KL divergence to the regularized optimizer, and then relate this KL divergence to a two-point testing error.

Fix any $( \varepsilon , \delta ) – \mathrm { D P }$ algorithm A and let $\hat { \pi } = \mathcal { A } ( D )$ be its output policy. For KL-regularized objectives, the gap admits the KL representation as presented in the upper bound proof:

$$
\operatorname{Gap} (\hat {\pi}; \theta) = \frac {1}{\eta} \mathbb {E} _ {X \sim \rho} \bigl [ \operatorname{KL} \bigl (\hat {\pi} (\cdot | X) \| \pi_ {\theta} ^ {\star} (\cdot | X) \bigr) \bigr ] = \frac {1}{\eta d} \sum_ {x \in \mathcal {X}} \operatorname{KL} \bigl (\hat {\pi} (\cdot | x) \| \pi_ {\theta} ^ {\star} (\cdot | x) \bigr).
$$

All summands are nonnegative, hence

$$
\operatorname{Gap} (\hat {\pi}; \theta) \geq \frac {1}{\eta d} \operatorname{KL} \bigl (\hat {\pi} (\cdot | x _ {0}) \| \pi_ {\theta} ^ {\star} (\cdot | x _ {0}) \bigr).\tag{47}
$$

We now show that a wrong-side output at $x _ { 0 }$ forces a constant KL loss. Under our hard instance, $\mathcal { A } = \{ 0 , 1 \}$ , so $\pi ( \cdot | x _ { 0 } )$ is a Bernoulli distribution. Write $p : = \pi ( 1 | x _ { 0 } )$ . Under $\theta _ { + }$ we have $\pi _ { \theta _ { + } } ^ { \star } ( 1 | x _ { 0 } ) = \sigma ( \eta c )$ , hence

$$
\operatorname{KL} \left(\pi (\cdot | x _ {0}) \| \pi_ {\theta_ {+}} ^ {\star} (\cdot | x _ {0})\right) = \operatorname{KL} \left(\operatorname{Bern} (p) \| \operatorname{Bern} (\sigma (\eta c))\right).
$$

We now show that whenever $p \leq 1 / 2$

$$
\operatorname{KL} \left(\operatorname{Bern} (p) \| \operatorname{Bern} (\sigma (\eta c))\right) \geq \log \cosh (\eta c / 2).\tag{48}
$$

Since $p \leq 1 / 2$ and $q : = \sigma ( \eta c ) > 1 / 2$ , write

$$
\begin{array}{c} f (p) := \operatorname{KL} \bigl (\operatorname{Bern} (p) \parallel \operatorname{Bern} (q) \bigr) \\ = p \log \frac {p}{q} + (1 - p) \log \frac {1 - p}{1 - q}. \end{array}
$$

Diferentiating in p gives

$$
f ^ {\prime} (p) = \log \frac {p}{q} - \log \frac {1 - p}{1 - q} = \log \Big (\frac {p (1 - q)}{q (1 - p)} \Big).
$$

For $p \in ( 0 , 1 / 2 ]$ and $q \in ( 1 / 2 , 1 )$ , we have $p / ( 1 - p ) \leq 1$ and $( 1 - q ) / q < 1$ , hence

$$
\frac {p (1 - q)}{q (1 - p)} \leq \frac {1 - q}{q} <   1 \quad \Rightarrow \quad f ^ {\prime} (p) <   0.
$$

Therefore $f$ is decreasing on $[ 0 , 1 / 2 ]$ , and thus

$$
\operatorname{KL} \bigl (\operatorname{Bern} (p) \| \operatorname{Bern} (q) \bigr) = f (p) \geq f (1 / 2) = \operatorname{KL} \bigl (\operatorname{Bern} (1 / 2) \| \operatorname{Bern} (q) \bigr).
$$

Now compute, letting $t : = \eta c$ and $q = \sigma ( t )$ ,

$$
\begin{array}{l} \operatorname{KL} \bigl (\operatorname{Bern} (1 / 2) \parallel \operatorname{Bern} (q) \bigr) = \frac {1}{2} \log \frac {1 / 2}{q} + \frac {1}{2} \log \frac {1 / 2}{1 - q} \\ \qquad = \frac {1}{2} \bigl (\log (1 / 2) - \log q \bigr) + \frac {1}{2} \bigl (\log (1 / 2) - \log (1 - q) \bigr) \\ \qquad = \log (1 / 2) - \frac {1}{2} \bigl (\log q + \log (1 - q) \bigr) \\ \qquad = - \log 2 - \frac {1}{2} \log \bigl (q (1 - q) \bigr). \end{array}
$$

Moreover,

$$
\begin{array}{c} q (1 - q) = \sigma (t) (1 - \sigma (t)) = \sigma (t) \sigma (- t) = \frac {1}{1 + e ^ {- t}} \cdot \frac {1}{1 + e ^ {t}} \\ = \frac {1}{(1 + e ^ {- t}) (1 + e ^ {t})} = \frac {1}{2 + e ^ {t} + e ^ {- t}} = \frac {1}{2 (1 + \cosh t)}. \end{array}
$$

Plugging this into the previous display gives

$$
\begin{array}{r l} \mathrm{KL} \big (\mathrm{Bern} (1 / 2) \parallel \mathrm{Bern} (\sigma (t)) \big) & = - \log 2 + \frac {1}{2} \log \{2 (1 + \cosh t) \} \\ & = \frac {1}{2} \log (1 + \cosh t) - \frac {1}{2} \log 2 \\ & = \log \cosh (t / 2), \end{array}
$$

where we used 1 + cosh $t = 2 \cosh ^ { 2 } ( t / 2 )$ . This proves (48).

By symmetry, under $\theta _ { - }$ we have $\pi _ { \theta _ { - } } ^ { \star } ( 1 | x _ { 0 } ) = \sigma ( - \eta c ) < 1 / 2$ , and whenever $p \ge 1 / 2$ the same lower bound (48) holds with $\theta _ { - }$

We next connect the gap lower bound to a two-point testing error by explicitly constructing a test from the algorithm output πˆ.

Let $\hat { p } : = \hat { \pi } ( 1 | x _ { 0 } ) \in [ 0 , 1 ]$ . Under $\theta _ { + }$ we have

$$
\operatorname{Gap} (\hat {\pi}; \theta_ {+}) \geq \frac {1}{\eta d} \operatorname{KL} \bigl (\hat {\pi} (\cdot | x _ {0}) \| \pi_ {\theta_ {+}} ^ {\star} (\cdot | x _ {0}) \bigr) = \frac {1}{\eta d} \operatorname{KL} \bigl (\operatorname{Bern} (\hat {p}) \| \operatorname{Bern} (\sigma (\eta c)) \bigr).
$$

If $\hat { p } < 1 / 2$ , then (48) gives $\mathrm { K L } \big ( \mathrm { B e r n } ( \hat { p } ) \| \mathrm { B e r n } ( \sigma ( \eta c ) ) \big ) \geq \log \cosh ( \eta c / 2 )$ . Therefore,

$$
\mathrm{Gap} (\hat {\pi}; \theta_ {+}) \geq \frac {1}{\eta d} \log \cosh (\eta c / 2) \cdot \mathbf {1} \{\hat {p} <   1 / 2 \}.
$$

Taking expectations under $\theta _ { + }$ yields

$$
\mathbb {E} _ {\theta_ {+}} \left[ \operatorname{Gap} \left(\hat {\pi}; \theta_ {+}\right) \right] \geq \frac {1}{\eta d} \log \cosh (\eta c / 2) \cdot \mathbb {P} _ {\theta_ {+}} \left(\hat {\pi} \left(1 | x _ {0}\right) <   1 / 2\right).
$$

Similarly, under $\theta _ { - }$ we have $\pi _ { \theta _ { - } } ^ { \star } ( 1 | x _ { 0 } ) = \sigma ( - \eta c ) < 1 / 2$ . Using the symmetric version of (48) $( { \mathrm { i . e . } }$ , when $\hat { p } \ge 1 / 2 )$ ,

$$
\mathbb {E} _ {\theta_ {-}} \left[ \operatorname{Gap} \left(\hat {\pi}; \theta_ {-}\right) \right] \geq \frac {1}{\eta d} \log \cosh (\eta c / 2) \cdot \mathbb {P} _ {\theta_ {-}} \left(\hat {\pi} \left(1 \mid x _ {0}\right) \geq 1 / 2\right).
$$

Averaging the last two displays gives

$$
\frac {1}{2} \mathbb {E} _ {\theta_ {+}} [ \mathrm{Gap} (\hat {\pi}; \theta_ {+}) ] + \frac {1}{2} \mathbb {E} _ {\theta_ {-}} [ \mathrm{Gap} (\hat {\pi}; \theta_ {-}) ] \geq \frac {1}{\eta d} \log \cosh (\eta c / 2) \cdot P _ {e},\tag{49}
$$

where

$$
P _ {e} := \frac {1}{2} \mathbb {P} _ {\theta_ {+}} (\hat {\pi} (1 | x _ {0}) <   1 / 2) + \frac {1}{2} \mathbb {P} _ {\theta_ {-}} (\hat {\pi} (1 | x _ {0}) \geq 1 / 2).
$$

It is convenient to interpret $P _ { e }$ as the error probability of an explicit test for distinguishing $\theta _ { + } ~ \mathrm { v s . } ~ \theta .$ <sub>−</sub>. Define $\hat { \theta } = \hat { \theta } ( D ) \in \{ + , - \}$ by

$$
\hat {\theta} (D) = \left\{ \begin{array}{l l} +, & \text { if } \hat {\pi} (1 | x _ {0}) \geq 1 / 2, \\ -, & \text { if } \hat {\pi} (1 | x _ {0}) <   1 / 2. \end{array} \right.
$$

Then, under the uniform prior on $\{ \theta _ { + } , \theta _ { - } \}$ , the Bayes error of $\hat { \theta }$ is exactly $P _ { e }$ :

$$
\mathbb {P} (\hat {\theta} (D) \neq \theta) = \frac {1}{2} \mathbb {P} _ {\theta_ {+}} (\hat {\theta} (D) = -) + \frac {1}{2} \mathbb {P} _ {\theta_ {-}} (\hat {\theta} (D) = +) = P _ {e}.
$$

Moreover, since $\hat { \theta }$ is a deterministic function of the DP output $\hat { \pi } _ { i }$ , it is a post-processing of πˆ and hence $\hat { \theta }$ is also $( \varepsilon , \delta ) – \mathrm { D P }$

Step 4. Lower bound $P _ { e }$ via DP Le Cam. We now lower bound the testing error $P _ { e }$ established in (49), which is the Bayes error under the uniform prior on $\{ \theta _ { + } , \theta _ { - } \}$ . We invoke Lemma 19, the DP $\mathrm { L e }$ Cam inequality presented in Acharya et al. (2021), which requires bounding the expected Hamming distance under a coupling of the data generating distributions.

Let $P _ { \theta _ { + } }$ and $P _ { \theta _ { - } }$ denote the single-observation distributions induced by the two hard instances $\theta _ { + }$ and $\theta _ { - }$ , respectively. Thus $D _ { + } \sim P _ { \theta _ { + } } ^ { \otimes n }$ and $D _ { - } \sim P _ { \theta _ { - } } ^ { \otimes n }$ mean that $D _ { + }$ and $D .$ are independent size-n datasets generated i.i.d. under $\theta _ { + }$ and $\theta _ { - }$ , respectively. We now construct a coupling of $D _ { + }$ and $D _ { - }$ and compute

$$
D := \mathbb {E} \big [ d _ {\mathrm{Ham}} (D _ {+}, D _ {-}) \big ],
$$

where $d _ { \mathrm { H a m } } ( D _ { + } , D _ { - } )$ denotes the Hamming distance between the two datasets, that is, the number of coordinates at which the two coupled samples difer.

Couple the datasets record-wise. For each $i \in \{ 1 , \ldots , n \}$ , first couple contexts by setting $X _ { i } ^ { + } = X _ { i } ^ { - } \sim \rho$ (recall $\rho$ is uniform on $\mathcal { X } _ { : }$ , so $\mathbb { P } ( X _ { i } ^ { + } = x _ { 0 } ) = 1 / d )$ . Given $X _ { i } ^ { + } = X _ { i } ^ { - }$ :

• If $X _ { i } ^ { + } ~ \neq ~ x _ { 0 }$ , then under our hard instance the conditional label distributions coincide

under $\theta _ { + }$ and $\theta _ { - } ,$ so we set $Y _ { i } ^ { + } = Y _ { i } ^ { - }$ (this contributes zero to the Hamming distance).

• If $X _ { i } ^ { + } = x _ { 0 }$ , then under $\theta _ { + }$ and $\theta _ { - }$ <sub>−</sub> the conditional label laws difer. In this case we couple $Y _ { i } ^ { + } \sim \mathrm { B e r n } ( \sigma ( c ) )$ and $Y _ { i } ^ { - } \sim \mathrm { B e r n } ( \sigma ( - c ) )$ by a maximal coupling.

Under this construction,

$$
\begin{array}{r l} & {\mathbb {P} \big ((X _ {i} ^ {+}, Y _ {i} ^ {+}) \neq (X _ {i} ^ {-}, Y _ {i} ^ {-}) \big) = \mathbb {P} (X _ {i} ^ {+} = x _ {0}) \cdot \mathbb {P} (Y _ {i} ^ {+} \neq Y _ {i} ^ {-} | X _ {i} ^ {+} = x _ {0})} \\ & {\qquad = \frac {1}{d} \operatorname{TV} (\operatorname{Bern} (\sigma (c)), \operatorname{Bern} (\sigma (- c))),} \end{array}
$$

where maximal coupling gives $\mathbb { P } ( Y _ { i } ^ { + } \neq Y _ { i } ^ { - } ) = \mathrm { T V } ( \cdot , \cdot )$

For Bernoulli distributions, $\mathrm { T V } ( \mathrm { B e r n } ( u ) , \mathrm { B e r n } ( v ) ) = | u - v |$ . Hence,

$$
\begin{array}{r l} \mathrm{TV} (\mathrm{Bern} (\sigma (c)), \mathrm{Bern} (\sigma (- c))) & = | \sigma (c) - \sigma (- c) | \\ & = \left| \frac {1}{1 + e ^ {- c}} - \frac {1}{1 + e ^ {c}} \right| \\ & = \left| \frac {(1 + e ^ {c}) - (1 + e ^ {- c})}{(1 + e ^ {- c}) (1 + e ^ {c})} \right| \\ & = \left| \frac {e ^ {c} - e ^ {- c}}{2 + e ^ {c} + e ^ {- c}} \right| = \frac {e ^ {c} - e ^ {- c}}{e ^ {c} + 2 + e ^ {- c}} \\ & = \frac {2 \sinh (c)}{2 (1 + \cosh (c))} = \frac {\sinh (c)}{1 + \cosh (c)} = \tanh (c / 2). \end{array}
$$

Therefore,

$$
\mathbb {P} \big ((X _ {i} ^ {+}, Y _ {i} ^ {+}) \neq (X _ {i} ^ {-}, Y _ {i} ^ {-}) \big) = \frac {1}{d} \tanh (c / 2).
$$

Since

$$
d _ {\mathrm{Ham}} (D _ {+}, D _ {-}) = \sum_ {i = 1} ^ {n} \mathbf {1} \left\{\left(X _ {i} ^ {+}, Y _ {i} ^ {+}\right) \neq \left(X _ {i} ^ {-}, Y _ {i} ^ {-}\right) \right\},
$$

taking expectations and using linearity gives

$$
\begin{array}{l} D := \mathbb {E} \big [ d _ {\mathrm{Ham}} (D _ {+}, D _ {-}) \big ] = \mathbb {E} \bigg [ \sum_ {i = 1} ^ {n} \mathbf {1} \Big \{(X _ {i} ^ {+}, Y _ {i} ^ {+}) \neq (X _ {i} ^ {-}, Y _ {i} ^ {-}) \Big \} \bigg ] \\ \qquad = \sum_ {i = 1} ^ {n} \mathbb {E} \Big [ \mathbf {1} \Big \{(X _ {i} ^ {+}, Y _ {i} ^ {+}) \neq (X _ {i} ^ {-}, Y _ {i} ^ {-}) \Big \} \bigg ] \\ \qquad = \sum_ {i = 1} ^ {n} \mathbb {P} \Big ((X _ {i} ^ {+}, Y _ {i} ^ {+}) \neq (X _ {i} ^ {-}, Y _ {i} ^ {-}) \Big) \\ \qquad = \sum_ {i = 1} ^ {n} \frac {1}{d} \tanh (c / 2) = \frac {n}{d} \tanh (c / 2). \end{array}
$$

Now apply Lemma 19. For the $( \varepsilon , \delta ) – \mathrm { D P }$ test ${ \hat { \theta } } ,$

$$
\begin{array}{r l} P _ {e} \geq & \frac {1}{2} \Big [ 0. 9 e ^ {- 1 0 \varepsilon D} - 1 0 D \delta \Big ] _ {+} \\ = & \frac {1}{2} \Big [ 0. 9 \exp \Big (- 1 0 \varepsilon \cdot \frac {n}{d} \tanh (c / 2) \Big) - 1 0 \cdot \frac {n}{d} \tanh (c / 2) \cdot \delta \Big ] _ {+}. \end{array}
$$

Plugging this bound into (49) yields the privacy-dependent lower bound

$$
\begin{array}{r l} & {\frac {1}{2} \mathbb {E} _ {\theta_ {+}} [ \mathrm{Gap} (\hat {\pi}; \theta_ {+}) ] + \frac {1}{2} \mathbb {E} _ {\theta_ {-}} [ \mathrm{Gap} (\hat {\pi}; \theta_ {-}) ]} \\ & {\geq \frac {1}{\eta d} \log \cosh (\eta c / 2) \cdot \frac {1}{2} \Big [ 0. 9 \exp \Big (- 1 0 \varepsilon \cdot \frac {n}{d} \tanh (c / 2) \Big) - 1 0 \cdot \frac {n}{d} \tanh (c / 2) \cdot \delta \Big ] _ {+}.} \end{array}
$$

This completes the DP Le Cam step; it remains to choose c and simplify the expression to obtain the two rate regimes.

Step 5. Signal calibration and rate simplification. We now choose the signal level c and simplify the privacy-dependent lower bound. Set

$$
c := \frac {d}{K \varepsilon n},
$$

where $K > 0$ is a suficiently large universal constant to be specified below. Since tanh $( u ) \leq u$ for all $u \geq 0$ , we have

$$
1 0 \varepsilon D = 1 0 \varepsilon \cdot \frac {n}{d} \tanh (c / 2) \leq 1 0 \varepsilon \cdot \frac {n}{d} \cdot \frac {c}{2} = \frac {5}{K}.
$$

Likewise, using the theorem assumption $\delta \leq \varepsilon$ .

$$
1 0 D \delta = 1 0 \cdot \frac {n}{d} \tanh (c / 2) \cdot \delta \leq 1 0 \cdot \frac {n}{d} \cdot \frac {c}{2} \cdot \delta = \frac {5 \delta}{K \varepsilon} \leq \frac {5}{K}.
$$

Therefore

$$
0. 9 e ^ {- 1 0 \varepsilon D} - 1 0 D \delta \geq 0. 9 e ^ {- 5 / K} - \frac {5}{K}.
$$

Choosing K suficiently large makes the right-hand side bounded below by a universal positive constant $b _ { 0 } > 0$ . Hence

$$
P _ {e} \geq b _ {1}
$$

for some universal constant $b _ { 1 } > 0$ . Plugging this into (49) yields

$$
R _ {n} (\varepsilon , \delta) \geq \frac {b _ {2}}{\eta d} \log \cosh \left(\frac {\eta d}{2 K \varepsilon n}\right)\tag{50}
$$

for some universal constant $b _ { 2 } > 0$

We now simplify (50). Let

$$
u _ {n} := \frac {\eta d}{2 K \varepsilon n}.
$$

If $u _ { n } \geq 1$ , then log cosh $( u _ { n } ) \geq c _ { \mathrm { l i n } } u _ { n }$ for a universal constant $c _ { \mathrm { l i n } } > 0$ , and therefore

$$
R _ {n} (\varepsilon , \delta) \geq \frac {b _ {2} c _ {\mathrm{lin}}}{\eta d} u _ {n} = \frac {b _ {2} c _ {\mathrm{lin}}}{2 K} \cdot \frac {1}{n \varepsilon}.
$$

If $u _ { n } \leq 1$ , then log cosh $( u _ { n } ) \geq c _ { \mathrm { q u a d } } u _ { n } ^ { 2 }$ for a universal constant $c _ { \mathrm { q u a d } } > 0$ , and therefore

$$
R _ {n} (\varepsilon , \delta) \geq \frac {b _ {2} c _ {\mathrm{quad}}}{\eta d} u _ {n} ^ {2} = \frac {b _ {2} c _ {\mathrm{quad}} \eta}{4 K ^ {2}} \cdot \frac {d}{n ^ {2} \varepsilon^ {2}}.
$$

Since $\eta > 0$ is fixed, the last two displays imply that there exist constants $c _ { \eta } , C _ { \eta } > 0$ such that

$$
R _ {n} (\varepsilon , \delta) \geq c _ {\eta} \min \biggl \{\frac {1}{n \varepsilon}, \frac {d}{n ^ {2} \varepsilon^ {2}} \biggr \},
$$

and, moreover, whenever $n \geq n _ { \mathrm { P } } : = C _ { \eta } d / \varepsilon$ , the quadratic branch is active and

$$
R _ {n} (\varepsilon , \delta) \geq c _ {\eta} \frac {d}{n ^ {2} \varepsilon^ {2}}.
$$

Combining this privacy-dependent lower bound with the non-private lower bound from Step 1 yields

$$
R _ {n} (\varepsilon , \delta) \geq c _ {\eta} \max \left\{\frac {d}{n}, \min \left(\frac {1}{n \varepsilon}, \frac {d}{n ^ {2} \varepsilon^ {2}}\right) \right\},
$$

for all $n \geq n _ { \mathrm { N P } } : = C _ { \eta } d .$ , and

$$
R _ {n} (\varepsilon , \delta) \geq c _ {\eta} \max \left\{\frac {d}{n}, \frac {d}{n ^ {2} \varepsilon^ {2}} \right\},
$$

for all n ≥ max $\{ n _ { \mathrm { N P } } , n _ { \mathrm { P } } \}$ . This completes the proof.

## E Supporting Lemma

Lemma 12 (Remark 5.3 in (Tropp, 2012)). Let $\{ X _ { i } \} _ { i = 1 } ^ { n }$ be independent, random, positivesemidefinite, symmetric matrices in $\mathbb { R } ^ { d \times d }$ . Assume $\lambda _ { \operatorname* { m a x } } ( X _ { i } ) \leq R$ almost surely for all i, and define

$$
\mu_ {\mathrm{min}} := \lambda_ {\mathrm{min}} \Bigl (\sum_ {i = 1} ^ {n} \mathbb {E} [ X _ {i} ] \Bigr).
$$

Then for any $t \in [ 0 , 1 ]$ ，

$$
\mathbb {P} \left(\lambda_ {\min} \left(\sum_ {i = 1} ^ {n} X _ {i}\right) \leq t \mu_ {\min}\right) \leq d \cdot \exp \left(- \frac {(1 - t) ^ {2} \mu_ {\min}}{2 R}\right).
$$

Lemma 13 (Harvey last-iterate bound, scaled). Let f be µ-strongly convex and L-Lipschitz on a closed convex set Θ. Consider iterates

$$
x _ {t + 1} = \Pi_ {\Theta} (x _ {t} - \eta_ {t} (\nabla f (x _ {t}) - z _ {t})), \qquad \eta_ {t} = \frac {1}{\mu t},
$$

where $\left( { z _ { t } } \right)$ is adapted and satisfies $\mathbb { E } [ z _ { t } | \mathcal { F } _ { t - 1 } ] = 0$ and $\| z _ { t } \| _ { 2 } \leq Z$ a.s. for all t. Then there exists

a universal constant $c > 0$ such that for any $\delta \in ( 0 , 1 )$ ,

$$
\mathbb {P} \left(f (x _ {T + 1}) - f (x ^ {*}) \leq c \cdot \frac {\log T \cdot \log \frac {1}{\delta}}{T} \cdot \frac {(L + Z) ^ {2}}{\mu}\right) \geq 1 - \delta , \quad x ^ {*} := \arg \min _ {x \in \Theta} f (x).\tag{51}
$$

Proof of Lemma 13. This is a direct rescaling of Harvey et al. (2019, Theorem 3.1). Let $G : =$ $L + Z$ and define ${ \tilde { f } } : = f / G$ . Then $\tilde { f }$ is 1-Lipschitz and $( \mu / G ) \mathrm { { - s t r o n g l y } }$ convex. Also $\tilde { z } _ { t } : = z _ { t } / G$ satisfies $\| \tilde { z } _ { t } \| \leq 1 \ \mathrm { a . s . }$ . and $\mathbb { E } [ \tilde { z } _ { t } | \mathcal { F } _ { t - 1 } ] = 0$ . Writing the update in terms of $\tilde { f } \ \mathrm { g i }$ ves

$$
x _ {t + 1} = \Pi_ {\Theta} \Big (x _ {t} - \eta_ {t} G \big (\nabla \tilde {f} (x _ {t}) - \tilde {z} _ {t} \big) \Big).
$$

With $\eta _ { t } = 1 / ( \mu t )$ , we have $\eta _ { t } G = 1 / ( ( \mu / G ) t )$ , i.e., the step size used by Harvey et al. (2019, Theorem 3.1) for $\tilde { f } .$ . Applying that theorem yields

$$
\tilde {f} (x _ {T + 1}) - \tilde {f} (x ^ {*}) \leq c \cdot \frac {\log T \cdot \log (1 / \delta)}{T} \cdot \frac {1}{\mu / G}
$$

with probability at least $1 - \delta$ . Multiplying by G gives (51).

Lemma 14 (Tail bound for $\chi ^ { 2 }$ , (Laurent and Massart, 2000)). Let $U \sim \chi _ { d } ^ { 2 }$ . Then for every $x > 0$

$$
\mathbb {P} \left(U \geq d + 2 \sqrt {d x} + 2 x\right) \leq e ^ {- x}.\tag{52}
$$

Moreover, for every $x > 0$

$$
\mathbb {P} \left(U \leq d - 2 \sqrt {d x}\right) \leq e ^ {- x}.\tag{53}
$$

Lemma 15 (Utility of noisy SGD for strongly convex ERM, adapted from Theorem 2.4, (Bassily et al., 2014)). Fix a dataset $D = \{ z _ { i } \} _ { i = 1 } ^ { n }$ and define the average empirical loss

$$
\bar {L} _ {n} (\theta) = \frac {1}{n} \sum_ {i = 1} ^ {n} \ell (\theta ; z _ {i}), \quad \hat {\theta} \in \arg \min _ {\theta \in \Theta} \bar {L} _ {n} (\theta).
$$

Assume $\ell ( \cdot ; z )$ is G-Lipschitz on Θ for all z and ${ \bar { L } } _ { n }$ is µ-strongly convex on $\Theta .$ . Then there exists an $( \varepsilon , \delta ) \ – D P$ noisy SGD algorithm whose output $\tilde { \theta }$ satisfies

$$
\mathbb {E} \left[ \bar {L} _ {n} (\tilde {\theta}) - \bar {L} _ {n} (\hat {\theta}) \mid D \right] \leq O \left(\frac {G ^ {2} d \log^ {2} (n / \delta) \log (1 / \delta)}{\mu n ^ {2} \varepsilon^ {2}}\right),
$$

where the expectation is over the algorithmic randomness conditional on D. Equivalently, in ${ \widetilde { O } } ( \cdot )$ notation,

$$
\mathbb {E} \left[ \bar {L} _ {n} (\tilde {\theta}) - \bar {L} _ {n} (\hat {\theta}) \mid D \right] = \widetilde {O} \left(\frac {G ^ {2} d}{\mu n ^ {2} \varepsilon^ {2}}\right).
$$

Lemma 16 (Non-private MLE statistical error under PD design (via (Zhu et al., 2023))). Let $D = \{ ( x _ { i } , a _ { i } ^ { w } , a _ { i } ^ { \ell } , y _ { i } ) \} _ { i = 1 } ^ { n }$ be i.i.d. and let $\Delta \phi _ { i } : = \phi ( x _ { i } , a _ { i } ^ { w } ) - \phi ( x _ { i } , a _ { i } ^ { \ell } ) \in \mathbb { R } ^ { d }$ . Define the empirical

covariance

$$
\Sigma_ {D} := \frac {1}{n} \sum_ {i = 1} ^ {n} \Delta \phi_ {i} \Delta \phi_ {i} ^ {\top}.
$$

Assume $( i ) \ \| \Delta \phi _ { i } \| _ { 2 } \leq L _ { \Delta } \ a . s . , \ ( i i ) \ \| \theta \| _ { 2 } \leq R$ for all $\theta \in \Theta$ , and (iii) the population covariance is nondegenerate:

$$
\Sigma := \mathbb {E} [ \Delta \phi \Delta \phi^ {\top} ] \succeq \lambda I _ {d} \quad f o r s o m e \lambda > 0.
$$

Let $\hat { \theta } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \bar { L } _ { n } ( \theta )$ be the (non-private) MLE. Define the curvature constant

$$
\gamma := \inf _ {| t | \leq L _ {\Delta} R} \sigma (t) \bigl (1 - \sigma (t) \bigr) = \frac {1}{2 + e ^ {L _ {\Delta} R} + e ^ {- L _ {\Delta} R}}.
$$

Fix any $\rho \in ( 0 , 1 )$ . If

$$
n \geq \frac {8 L _ {\Delta} ^ {2}}{\lambda} \log \left(\frac {2 d}{\rho}\right),
$$

then with probability at least $1 - \rho _ { ; }$

$$
\| \hat {\theta} - \theta^ {\star} \| _ {2} \leq \frac {\sqrt {2}}{\sqrt {\lambda}} C _ {\mathrm{MJ}} \sqrt {\frac {d + \log (2 / \rho)}{\gamma^ {2} n}},
$$

where $C _ { \mathrm { M J } } > 0$ is the universal constant appearing in Zhu et al. (2023, Lemma $\it 3 . 1 $

Proof. Set $\delta : = \rho / 2$

Step 1 (MJ bound with $\lambda _ { \mathrm { r e g } } \ : = \ : 0 )$ . Apply Zhu et al. (2023, Lemma 3.1) with regularization parameter $\lambda _ { \mathrm { r e g } } = 0$ . With probability at least $1 - \delta$

$$
\| \hat {\theta} - \theta^ {\star} \| _ {\Sigma_ {D}} \leq C _ {\mathrm{MJ}} \sqrt {\frac {d + \log (1 / \delta)}{\gamma^ {2} n}} = C _ {\mathrm{MJ}} \sqrt {\frac {d + \log (2 / \rho)}{\gamma^ {2} n}}.
$$

Step 2 (Empirical PD from population PD). Since $\Delta \phi _ { i } \Delta \phi _ { i } ^ { \top } \succeq 0 , \| \Delta \phi _ { i } \| _ { 2 } \leq L _ { \Delta }$ implies $\lambda _ { \operatorname* { m a x } } ( \Delta \phi _ { i } \Delta \phi _ { i } ^ { \top } ) \leq$ $L _ { \Delta } ^ { 2 } ~ \mathrm { a . s } .$ . Moreover,

$$
\mathbb {E} [ \Delta \phi \Delta \phi^ {\top} ] = \Sigma \succeq \lambda I _ {d} \quad \Longrightarrow \quad \lambda_ {\min} (\mathbb {E} [ \Sigma_ {D} ]) = \lambda_ {\min} (\Sigma) \geq \lambda .
$$

Thus, a standard matrix Chernof bound yields that if $\begin{array} { r } { n \ge \frac { 8 L _ { \Delta } ^ { 2 } } { \lambda } \log \left( \frac { 2 d } { \rho } \right) } \end{array}$ , then with probability at least $1 - \delta$ ,

$$
\lambda_ {\mathrm{min}} (\Sigma_ {D}) \geq \frac {\lambda}{2}.
$$

Step $\mathcal { B }$ (Convert to $\ell _ { 2 } \rfloor$ . On the intersection of the two events above (which holds with probability at least $1 - 2 \delta = 1 - \rho )$ ),

$$
\| \hat {\theta} - \theta^ {\star} \| _ {\Sigma_ {D}} ^ {2} = (\hat {\theta} - \theta^ {\star}) ^ {\top} \Sigma_ {D} (\hat {\theta} - \theta^ {\star}) \geq \lambda_ {\min} (\Sigma_ {D}) \| \hat {\theta} - \theta^ {\star} \| _ {2} ^ {2} \geq \frac {\lambda}{2} \| \hat {\theta} - \theta^ {\star} \| _ {2} ^ {2},
$$

so

$$
\| \hat {\theta} - \theta^ {\star} \| _ {2} \leq \sqrt {\frac {2}{\lambda}} \| \hat {\theta} - \theta^ {\star} \| _ {\Sigma_ {D}} \leq \frac {\sqrt {2}}{\sqrt {\lambda}} C _ {\mathrm{MJ}} \sqrt {\frac {d + \log (2 / \rho)}{\gamma^ {2} n}}.
$$

This proves the claim.

Proposition 17 (Non-private minimax lower bound for KL-regularized RLHF (Theorem 4.6 in (Zhao et al., 2024))). Fix a target gap level $\mu \in ( 0 , 1 / 2 5 6 )$ and $\eta > 4$ . Consider the KL-regularized RLHF setting with preference feedback. Then for any (possibly randomized) algorithm A that, given n i.i.d. preference samples, outputs a policy $\hat { \pi } = \mathcal { A } ( D )$ , there exists a KL-regularized preference learning instance (with two actions, a finite context space, a reward function class $\mathcal { R } _ { \mathrm { : } }$ and coverage coeficient of order $O ( N _ { \mathcal { R } } ( \mu ) ) )$ such that achieving suboptimality gap at most µ requires

$$
n = \Omega \bigg (\min \left\{\frac {\eta \log N _ {\mathcal {R}} (\mu)}{\mu}, \frac {\log N _ {\mathcal {R}} (\mu)}{\mu^ {2}} \right\} \bigg).
$$

Equivalently, if n is smaller than the above order, then A cannot guarantee gap $\leq \mu$ uniformly over that problem class.

Lemma 18 (Pinsker’s Inequality). If $\mathbb { P } _ { 1 } , \mathbb { P } _ { 2 }$ are two probability measures on a common measurable space (Ω, F), then it holds that

$$
\delta (\mathbb {P} _ {1}, \mathbb {P} _ {2}) \leq \sqrt {\frac {1}{2} K L (\mathbb {P} _ {1} \| \mathbb {P} _ {2})},
$$

where $\delta ( \cdot , \cdot )$ is the total variation distance and $K L ( \mathbb { P } _ { 1 } \| \mathbb { P } _ { 2 } )$ is the Kullback-Leibler divergence.

Lemma 19 $\left( \left( \varepsilon , \delta \right)  – \mathrm { D P } \right)$ Le Cam lower bound with coupling (Acharya et al., 2021)). Let $p _ { 1 } \in$ $\mathrm { c o } ( P _ { 1 } )$ and $p _ { 2 } \in \mathrm { c o } ( P _ { 2 } )$ be two distributions on ${ \mathcal { Z } } ^ { n }$ . Let $( X , Y )$ be any coupling of $p _ { 1 }$ and p<sub>2</sub> such that $D : = \mathbb { E } [ d _ { \mathrm { H a m } } ( X , Y ) ] < \infty$ . Then for any $( \varepsilon , \delta ) \ – D P$ hypothesis testing algorithm $\hat { \theta }$ that outputs {1, 2}, the (minimax) testing risk satisfies

$$
P _ {e} (\hat {\theta}; P _ {1}, P _ {2}) \geq \frac {1}{2} \max \Big \{1 - d _ {\mathrm{TV}} (p _ {1}, p _ {2}), 0. 9 e ^ {- 1 0 \varepsilon D} - 1 0 D \delta \Big \}.\tag{54}
$$
