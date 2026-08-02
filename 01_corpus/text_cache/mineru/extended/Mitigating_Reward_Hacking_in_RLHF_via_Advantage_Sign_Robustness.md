# Mitigating Reward Hacking in RLHF via Advantage Sign Robustness

Shinnosuke Ono The University of Tokyo ono@ms.k.u-tokyo.ac.jp

Soichiro Nishimori The University of Tokyo and RIKEN AIP

Masashi Sugiyama RIKEN AIP and The University of Tokyo

Johannes Ackermann The University of Tokyo and RIKEN AIP

Takashi Ishida RIKEN AIP and The University of Tokyo

## Abstract

Reward models (RMs) used in reinforcement learning from human feedback (RLHF) are vulnerable to reward hacking: as the policy maximizes a learned proxy reward, true quality plateaus or degrades. We make the assumption that reward hacking is often caused by flipped advantage signs: instead of reducing the likelihood of a bad response, a flipped sign causes the update to increase it. By considering an adversarial perturbation in the RM parameter space, we can derive a certified sign-preservation radius, which is the smallest perturbation that can flip the advantage sign during policy optimization. Based on this formulation, we propose Sign-Certified Policy Optimization (SignCert-PO), down-weighting non-robust completions in the policy gradient update. Unlike prior approaches that require multiple RMs or access to the RM training data, SignCert-PO is lightweight and operates purely at the policy optimization stage using only the RM parameters and on-policy completions. On TL;DR summarization and AlpacaFarm benchmarks, SignCert-PO consistently achieves a better win rate than baselines and reduces reward hacking.

## 1 Introduction

Reinforcement learning from human feedback (RLHF) has become a standard paradigm for aligning large language models (LLMs) with human intent (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022). In the typical RLHF pipeline, a reward model (RM) is trained on pairwise human preference data and then used as a proxy for the true, unobservable preference signal during policy optimization. Because the RM is inherently an imperfect proxy, optimizing the policy against it often causes the policy to exploit local inaccuracies in the reward landscape. This phenomenon is known as reward hacking (Gao et al., 2023). As optimization progresses, proxy reward continues to rise while true quality stagnates or degrades. This is especially acute when the proxy RM is trained with limited preference data (Gao et al., 2023) or when the policy drifts from the RM’s training distribution during policy optimization by reinforcement learning (RL) (Ackermann et al., 2025).

Prior approaches to mitigating reward hacking can be organized along an axis from ante-hoc to post-hoc (Retzlaff et al., 2024). Ante-hoc methods incorporate uncertainty quantification into the reward modeling pipeline. For instance, ensemble-based methods (Coste et al., 2024; Eisenstein et al., 2024) train multiple RMs and optimize conservatively with respect to their disagreement, while distributional or uncertainty-aware RMs (Xu et al., 2024) modify the RM architecture or training objective. These approaches can directly quantify RM uncertainty, but require either multiple RM training and inference runs or changes to the RM pipeline, limiting their practical applicability. In contrast, in the post-hoc approach, the

(a)
(c)

(b)


(d)
Figure 1: We argue that the reliability of the proxy RM’s estimates differs by completion. Certified sign-preservation radius $\Delta _ { j }$ provides this reliability measure. (a) Proxy and true advantages. Completions 7 and 8 have opposite signs, showing the proxy RM is unreliable there. (b) $\Delta _ { j }$ is the smallest perturbation of the RM parameters that flips a completion’s advantage sign. Dashed lines are decision boundaries for the j- and k-th completions. (c) Per-completion $\Delta _ { j }$ . Completions 7 and 8 exhibit low $\Delta ,$ confirming $\Delta _ { j }$ identifies unreliable completions. SignCert-PO suppresses completions below $\Delta = \epsilon$ . (d) True (solid) and proxy (dashed) reward during RL. Using $\Delta _ { j }$ as a re-weighting mechanism, SignCert-PO prevents reward hacking and further improves the true reward. See Appendix F.1 for details.

RM is left unchanged and robustness is introduced only at the policy optimization stage. AdvPO (Zhang et al., 2024), the closest prior work to ours, is only partially post-hoc: it derives single-RM uncertainty estimates from last-layer embeddings without retraining, but still requires access to the entire RM training dataset to calibrate its uncertainty model. Our approach is entirely post-hoc: it operates solely with the current RM parameters and on-policy completions, and thus we do not need to train new RMs nor retain the dataset used for RM training. We discuss additional related work in Appendix A.2.

We argue that what matters during policy optimization is whether the RM correctly predicts the sign of each completion’s advantage, since that sign determines whether the policy increases or decreases the completion’s probability. Crucially, the reliability of this prediction varies per completion, and a naive robustness penalty fails to capture this variability. We therefore ask: for a given completion, how much would the RM need to change before that advantage sign flips? Based on randomized smoothing (Cohen et al., 2019), we formalize this quantity as a certified sign-preservation radius and down-weight completions whose advantage signs are easy to flip (Figure 1). We first show that directly considering perturbations of all RM parameters is computationally challenging, as it requires per-completion gradients. This quickly causes us to run out of memory, even for relatively small RMs. Instead, we propose to approximate this solution by only considering perturbations of the RM head, yielding a lightweight re-weighting of the standard policy gradient update. The resulting method, Sign-Certified Policy Optimization (SignCert-PO), prevents policy updates from being dominated by directions whose update signs are locally fragile under RM perturbations.

We evaluate SignCert-PO on TL;DR summarization (Stiennon et al., 2020) and Alpaca-Farm (Dubois et al., 2023) with Pythia (Biderman et al., 2023) and Qwen2.5 (Qwen et al., 2025) models. Using the gold-model evaluation setup (Gao et al., 2023), we compare against Dr.GRPO (Liu et al., 2025), uncertainty-weighted optimization (UWO) with RM ensembles (Coste et al., 2024), behavior-supported policy optimization (BSPO) (Dai et al., 2025), and AdvPO (Zhang et al., 2024). SignCert-PO achieves the highest gold-RM win rate in most settings, with improved RM accuracy during policy optimization.

## 2 Background

We denote the set of prompts by  and the set of responses by . We assume a prompt distribution $P$ over ${ \dot { \boldsymbol { \mathcal { X } } } } ,$ where $\breve { P } ( x )$ denotes the probability mass of a prompt $\hat { \boldsymbol { x } } \in \hat { \boldsymbol { \mathcal { X } } }$ An autoregressive language model $\pi _ { \phi }$ with parameters $\phi$ is then conditional probability $\pi _ { \phi } ( y | x ) = \Pi _ { t } \pi _ { \phi } ( y _ { t } | x , y _ { < t } )$ , where $x \in \mathcal { X }$ is a prompt, $y \in \mathcal { V }$ is the response, $y _ { t }$ is the t-th response token and $y _ { < t }$ are the response tokens before $y _ { t }$ . Following the RL literature (Sutton & Barto, 2018), we sometimes use the word policy, referring to the language model $\pi _ { \phi }$ . We denote the $\ell _ { 2 }$ norm by $\| \cdot \| _ { 2 }$

## 2.1 Reinforcement learning from human feedback

RLHF typically consists of three key steps: supervised fine-tuning (SFT), reward modeling, and reinforcement learning (Stiennon et al., 2020). We denote the SFT model as $\pi _ { \mathrm { S F T } }$

Reward modeling. Suppose the true reward function $r ^ { * } : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ that mirrors human preference exists. Since $r ^ { * }$ is generally unknown, an RM is trained as a proxy with the following procedure: given a prompt $x \ \in \ { \mathcal { X } } .$ , two responses $y _ { 0 } , y _ { 1 } \in \mathcal { V }$ are sampled from $\pi _ { \mathrm { S F T } } ( y | x )$ . A human annotator then selects their preferred response, yielding a dataset of triples $( x , y _ { \mathrm { w } } , y _ { \mathrm { l } } )$ where $y _ { \mathrm { { w } } }$ (winning) is preferred over $y _ { \mathrm { l } }$ (losing). A scalar RM $r _ { \theta } : \mathcal { X } \times \mathcal { Y }  \mathbf { \hat { R } }$ is trained under the Bradley-Terry (BT) model (Bradley $\&$ Terry, 1952) by minimizing the cross-entropy loss:

$$
\mathcal {L} _ {\mathrm{BT}} (\theta) = - \mathbb {E} _ {(x, y _ {\mathrm{w}}, y _ {\mathrm{l}})} \left[ \log \sigma \big (r _ {\theta} (x, y _ {\mathrm{w}}) - r _ {\theta} (x, y _ {\mathrm{l}}) \big) \right],\tag{1}
$$

where $\sigma : \mathbb { R }  \mathbb { R }$ is the sigmoid function: $\sigma ( z ) : = 1 / ( 1 + \exp ( - z ) )$

Reinforcement learning. The policy $\pi _ { \phi }$ is optimized to maximize the Kullback-Leibler (KL)-regularized expected reward (Ziegler et al., 2020):

$$
J (\phi , \theta) = \mathbb {E} _ {x \sim P (x), y \sim \pi_ {\phi}} \big [ r _ {\theta} (x, y) - \beta D _ {\mathrm{KL}} (\pi_ {\phi} (\cdot | x) \| \pi_ {\mathrm{ref}} (\cdot | x)) \big ],\tag{2}
$$

where $D _ { \mathrm { K L } } ( p \Vert q )$ is the KL divergence of probability densities $p$ from $q ,$ and $\beta \geq 0$ controls the strength of KL regularization against a reference policy $\pi _ { \mathrm { r e f } }$ . This regularization discourages the policy from diverging too far from the RM’s training distribution (Gao et al., 2023). We denote this regularized reward by $r _ { \theta } ^ { \prime } ( x , y ) : = r _ { \theta } ( x , y ) - \check { \beta D } _ { \mathrm { K L } } ( \pi _ { \phi } ( \cdot | x ) \| \pi _ { \mathrm { r e f } } ( \cdot | x ) )$ . Policy optimization proceeds via the policy gradient (PG) (Sutton et al., 1999):

$$
\nabla_ {\phi} J (\phi , \theta) = \mathbb {E} _ {x \sim P, y \sim \pi_ {\phi} (\cdot | x)} [ A _ {\theta} ^ {\pi_ {\phi}} (x, y) \nabla_ {\phi} \log \pi_ {\phi} (y | x) ],\tag{3}
$$

where $A _ { \theta } ^ { \pi _ { \phi } } ( x , y ) : = r _ { \theta } ^ { \prime } ( x , y ) - V _ { \theta } ^ { \pi _ { \phi } } ( x )$ is the advantage function with value function $V _ { \theta } ^ { \pi _ { \phi } } ( x ) : = \mathbb { E } _ { y \sim \pi _ { \phi } } [ r _ { \theta } ^ { \prime } ( x , y ) ]$ . Different PG-based algorithms for RLHF differ mainly in their update rules and advantage estimation. In this work, we adopt Dr.GRPO (Liu et al., 2025), a widely-used variant of GRPO (Shao et al., 2024). Given a prompt $x \in { \mathcal { X } } ,$ , we sample K com pletions $y ^ { ( 1 ) } , \ldots , y ^ { ( K ) } \in \mathcal { V }$ from the current policy $\pi _ { \phi } ( \cdot | x )$ . Each receives a reward $r _ { \theta } ( x , y ^ { ( j ) } )$ for $j \in [ \dot { K } ] : = \{ 1 , \ldots , K \}$ . Dr.GRPO then provides the following advantage estimation:

$$
A _ {j} (\theta) := r _ {\theta} (x, y ^ {(j)}) - \frac {1}{K} \sum_ {k = 1} ^ {K} r _ {\theta} (x, y ^ {(k)}).\tag{4}
$$

Since the RM $r _ { \theta }$ is only a proxy for true human preferences, the policy can exploit regions where it is inaccurate. As optimization progresses, proxy reward continues to rise while true quality degrades, especially when the proxy RM is trained with limited preference data. This phenomenon is known as reward hacking (Gao et al., 2023).

## 2.2 Randomized smoothing

Reward modeling reduces to a binary classification problem on reward margins (Tang et al., 2024; Nishimori et al., 2025). Randomized smoothing (RS) (Cohen et al., 2019) is a framework for certifying the robustness of a classifier’s prediction against perturbations. Consider a classifier $\mathcal { \bar { f } } : \bar { \mathbb { R } ^ { d } }  [ C ]$ over C classes. Given a noise scale $\sigma > 0$ for the Gaussian distribution $\mathcal { N } ,$ , RS creates the smoothed classifier

$$
g (x) := \underset {c \in [ C ]} {\arg \max} \operatorname * {P r} \left(f (x + \xi) = c\right), \quad \text { where } \xi \sim \mathcal {N} (0, \sigma^ {2} I _ {d}),
$$

which returns the class most likely to be predicted under random perturbations of the input. Cohen et al. (2019) showed that if one predicted class is sufficiently more likely than the others under these perturbations, then the prediction of the smoothed classifier is provably consistent throughout an $\ell _ { 2 }$ ball around x. The radius of this ball is called the certified radius. That is, the certified radius is the maximum $\ell _ { 2 }$ perturbation to the input under which the smoothed classifier’s prediction is guaranteed to remain unchanged.

## 3 Robust policy gradient method

We argue that when the RM $r _ { \theta }$ assigns incorrect advantage signs, the policy is not properly updated and thus reward hacking can occur. Guided by the notion of the certified radius (Cohen et al., 2019), we formalize this by introducing the certified sign-preservation radius, the smallest perturbation of the RM parameters that flips a completion’s advantage sign (§3.1). This leads to our proposed method of re-weighting the policy gradient updates, using the radius as a reliability measure (§3.2). Finally, we compare our method of using per-completion perturbations to the standard global perturbation setup, which we show suffers uniform pessimism unlike our method. We focus on the advantage estimate used in Dr.GRPO (4) throughout the rest of the paper, but we note that other PG methods can be used instead, as shown in Appendix E.

## 3.1 Certified sign-preservation radius

The PG (3) is a weighted sum of per-completion directions:

$$
\nabla_ {\phi} J (\phi , \theta) = \mathbb {E} _ {x \sim P, \{y ^ {(1)}, \dots , y ^ {(K)} \} \sim \pi_ {\phi}} \left[ \sum_ {j = 1} ^ {K} A _ {j} (\theta) \nabla_ {\phi} \log \pi_ {\phi} (y ^ {(j)} | x) \right].\tag{5}
$$

Each term pushes the policy to increase $( A _ { j } ( \theta ) > 0 )$ or decrease $( A _ { j } ( \theta ) < 0 )$ the probability of the j-th completion. Consider a single gradient step $\phi ^ { \prime } = \phi + \eta \overline { { \nabla } } _ { \phi } J ( \phi , \theta )$ ) with step size $\eta > 0$ . Then, a first-order expansion gives

$$
\log \pi_ {\phi^ {\prime}} (y ^ {(j)} | x) - \log \pi_ {\phi} (y ^ {(j)} | x) \approx \eta A _ {j} (\theta) \| \nabla_ {\phi} \log \pi_ {\phi} (y ^ {(j)} | x) \| _ {2} ^ {2}.
$$

Since $\eta > 0$ and $\parallel \nabla _ { \phi }$ log $\pi _ { \phi } \| _ { 2 } ^ { 2 } \geq 0 .$ , the sign of $A _ { j } ( \theta )$ alone determines the update direction, when considering the j-th term in isolation. Crucially, the reliability of this sign prediction varies per completion: since the RM is only a proxy, some predictions in a group may not agree with the true reward, while others do agree (Figure 1a).

To quantify the sensitivity for a given completion, we follow the idea of ${ \mathrm { R S } } ,$ which certifies robustness against input perturbations. As the input space of the RM is discrete, we cannot directly utilize input perturbations. Instead, we model epistemic uncertainty about the RM as perturbations in the parameter space: $\mathcal { U } _ { \tau } ^ { \theta } : = \{ \theta ^ { \prime } : \| \theta ^ { \prime } - \theta \| _ { 2 } \leq \tau \}$ (Ben-Tal et al., 2009).

Definition 3.1 (Certified sign-preservation radius). Under an uncertainty set $\mathcal { U } _ { \tau } ^ { \theta } .$ , the certified sign-preservation radius for the j-th completion is

$$
\Delta_ {j} := \sup \left\{\tau \geq 0: \operatorname{sign} \left(A _ {j} \left(\theta^ {\prime}\right)\right) = \operatorname{sign} \left(A _ {j} (\theta)\right) \forall \theta^ {\prime} \in \mathcal {U} _ {\tau} ^ {\theta} \right\}.\tag{6}
$$

This definition follows the certified-radius framework of Cohen et al. (2019), where the advantage sign is viewed as a binary classification decision in the RM parameter space (see Appendix C for the formal connection). As such, (6) has a natural interpretation as a sensitivity measure. A completion with large $\Delta _ { j }$ has a robust advantage sign: the RM would need to change substantially before the sign flips, so the corresponding gradient term in (5) can be trusted. On the other hand, for a completion with small $\Delta _ { j } ,$ a small perturbation to the RM could reverse whether the policy reinforces or suppresses it. The certified signpreservation radius thus provides a per-sample confidence measure: for each completion individually, it tells us how much the update direction depends on the exact choice of the proxy RM parameters θ.

For a general differentiable RM $r _ { \theta }$ with full-parameter uncertainty $\mathcal { U } _ { \tau } ^ { \theta } ,$ a first-order Taylor expansion of the advantage around θ gives $\bar { A _ { j } } ( \theta ^ { \prime } ) \approx A _ { i } ( \theta ) + \nabla _ { \theta } A _ { j } ( \theta ) ^ { \mathsf { T } } ( \theta ^ { \prime } - \theta )$ . The certified sign-preservation radius is then $\begin{array} { r } { \Delta _ { j } ( { \boldsymbol { \theta } } ) \gtrsim \frac { | \dot { \boldsymbol { A } } _ { j } ( { \boldsymbol { \theta } } ) | } { \| \nabla _ { \boldsymbol { \theta } } \boldsymbol { A } _ { j } ( { \boldsymbol { \theta } } ) \| _ { 2 } } } \end{array}$ . Unfortunately, computing $\| \nabla _ { \theta } A _ { j } ( \theta ) \|$ ∥2 requires the per-completion gradient norm of the RM, which our preliminary experiments found to be computationally infeasible for larger RMs. We thus propose to use a simpler perturbation model, for which we can efficiently calculate the certified sign-preservation radius, as we will show below.

## 3.2 Sign-Certified Policy Optimization (SignCert-PO)

We consider the RM architecture $\boldsymbol { r } _ { \theta } ( x , y ) = \boldsymbol { w } ^ { \top } h _ { \psi } ( x , y ) + \boldsymbol { b } ,$ consisting of a feature extractor $h _ { \psi } ( x , y )$ with parameters ψ and a linear head with parameters $( w , b )$ , as commonly used in practice (Stiennon et al., 2020; Ouyang et al., 2022). Instead of perturbing all parameters $( \psi , w , b )$ , we only perturb the linear head, and thus use the uncertainty set<sup>1</sup>

$$
\mathcal {U} _ {\epsilon} ^ {w} := \{w ^ {\prime} \in \mathbb {R} ^ {d}: \| w ^ {\prime} - w \| _ {2} \leq \epsilon \}.\tag{7}
$$

In what follows, $A _ { j } ( w )$ denotes $A _ { j } ( \theta )$ , stressing perturbations for the head parameters w.

Under the linear head model, the certified sign-preservation radius (6) admits a closed form that is easy to compute during policy optimization without additional information:

Theorem 3.1 (Certified radius). Under the linear head model and the uncertainty set ${ { \mathcal { U } } _ { \epsilon } ^ { w } \left( 7 \right) }$ , the certified sign-preservation radius is

$$
\Delta_ {j} = \frac {| A _ {j} (w) |}{\| h _ {\psi} (x , y ^ {(j)}) - \bar {h} \| _ {2}}, \quad w h e r e \bar {h} := \frac {1}{K} \sum_ {k = 1} ^ {K} h _ {\psi} (x, y ^ {(k)}).\tag{8}
$$

The proof follows from the linearity of the reward head and is given in Appendix B.

Based on the certified radius above, we propose SignCert-PO, a conservative variant of the standard PG (5). For each completion, let the adversary choose $w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { w }$ to maximally corrupt that completion’s gradient contribution. Concretely, the adversary minimizes the advantage for completions the policy with the nominal RM would reinforce $( A _ { j } ( w ) > 0 )$ and maximizes it for completions the policy would suppress $( A _ { j } ( w ) < 0 )$

$$
\tilde {A} _ {j} := \left\{ \begin{array}{l l} \inf _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} A _ {j} (w ^ {\prime}) & \text { if } A _ {j} (w) > 0, \\ \sup _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} A _ {j} (w ^ {\prime}) & \text { if } A _ {j} (w) <   0. \end{array} \right.\tag{9}
$$

This formulation asks: Given that the RM could be wrong within $\mathcal { U } _ { \epsilon } ^ { w }$ , what is the most conservative advantage we can still attribute to each completion?

Theorem 3.2 (Worst-case advantage under per-completion adversary). Under the linear head model and the uncertainty set ${ \mathcal { U } } _ { \epsilon } ^ { w } \left( 7 \right)$ , the worst-case advantage (9)for the j-th completion is

$$
\tilde {A} _ {j} = \rho_ {j} ^ {*} \cdot A _ {j} (w), \quad w h e r e \quad \rho_ {j} ^ {*} := 1 - \frac {\epsilon}{\Delta_ {j}} = 1 - \frac {\epsilon \| h _ {\psi} (x , y ^ {(j)}) - \bar {h} \| _ {2}}{| A _ {j} (w) |}.\tag{10}
$$

The proof is given in Appendix B. SignCert-PO thus replaces each advantage with its worst-case value:

$$
\widehat {\nabla} _ {\phi} J (\phi , \theta) = \mathbb {E} _ {x} \left[ \sum_ {j = 1} ^ {K} \rho_ {j} ^ {*} \cdot A _ {j} (w) \cdot \nabla_ {\phi} \log \pi_ {\phi} (y ^ {(j)} | x) \right].\tag{11}
$$

This is the standard GRPO gradient with each term re-weighted using its certified signpreservation radius. Setting $\overset { \smile } { \epsilon } = 0$ recovers standard GRPO. (11) can also be derived as the PG of a certain robust optimization problem but the discussion is deferred to Appendix D.

Practical implementation. The coefficient $\rho _ { j } ^ { * }$ (10) depends only on the advantage $A _ { j } ( w )$ and the feature deviation $\| h _ { \psi } ( x , y ^ { ( j ) } ) - \bar { h } \| _ { 2 }$ . Rather than fixing ϵ globally, we set it adaptively using the empirical distribution of the ratio $\| h _ { \psi } ( x , y ^ { ( j ) } ) - \bar { h } \| _ { 2 } / | A _ { j } ( w ) |$ within each batch. Specifically, we set ϵ to the $q _ { t } { \mathrm { - } } \mathrm { t h }$ quantile of $\{ \| h _ { \psi } ( x , y ^ { ( j ) } ) - \bar { h } \| _ { 2 } / | A _ { j } ( w ) | \} _ { i = 1 } ^ { B \cdot K }$ over all $B \cdot K$ completions in the current batch, where B is the number of prompts and K is the group size. As shown in Appendix G.3, the parameter $q _ { t } \in [ 0 , 1 ]$ provides a natural interpretation of controlling the proxy RM accuracy during policy optimization. The full pseudocode is provided in Algorithm 1 (Appendix F).

## 3.3 Comparison with the global robust objective

Finally, we compare our per-completion adversary (9) with the standard approach of taking a single worst-case reward for all completions over an uncertainty set (Ben-Tal et al., 2009; Zhang et al., 2024), which we show that cannot distinguish reliable from unreliable completions. Consider the global robust objective, with the shared perturbation $w ^ { \prime } ;$

$$
J _ {\text { global }} (\phi , \theta) = J _ {\text { global }} (\phi , \{\psi , w, b \}) := \inf _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} J (\phi , \{\psi , w ^ {\prime}, b \}).\tag{12}
$$

The following lemma shows that this approach falls short (the proof is given in Appendix B): Lemma 3.3 (Policy gradient of the global robust objective). Under the linear reward head, the PG of the worst-case objective (12) is

$$
\nabla_ {\phi} J _ {\mathrm{global}} (\phi , \theta) = \mathbb {E} _ {x} \left[ \sum_ {j = 1} ^ {K} A _ {j} (w) \nabla_ {\phi} \log \pi_ {\phi} (y ^ {(j)} | x) \right] - \epsilon \nabla_ {\phi} \big \| \mathbb {E} _ {x, y} [ h _ {\psi} (x, y) ] \big \| _ {2}.\tag{13}
$$

Here, the correction term $\nabla _ { \phi } \left\| \mathbb { E } _ { x , y } [ h _ { \psi } ( x , y ) ] \right\|$ <sub>2</sub> is a single direction shared across all comple tions $y \sim \pi _ { \phi }$ . It does not depend on which completions receive reliable reward signal and which do not: a completion whose ranking is highly certain receives the same penalty as one whose ranking could be flipped by a small perturbation of the RM parameters.

This uniform pessimism is also observed beyond the linear-head case. Zhang et al. (2024) empirically observed that their uncertainty set over the whole RM parameters θ led to overly pessimistic reward corrections. Eisenstein et al. (2024) similarly found that ensemblebased penalties fail when all members agree on the same spurious features, suggesting that completion-level discrimination is necessary.

In contrast, our per-completion formulation (9) and re-weighting (11) provide this discrimination: completions with robust advantage signs receive the gradient essentially unchanged, while completions with unreliable signs are effectively removed. SignCert-PO therefore selectively suppresses the unreliable gradient directions while preserving the reliable ones.

## 4 Experiments

To evaluate our proposed method in LLM alignment tasks, we adopt the gold-model setup of Gao et al. (2023), which allows us to investigate reward hacking by simulating the true reward $r ^ { * }$ with a larger “gold” RM. The pipeline proceeds in four stages:

<table><tr><td rowspan="2">Dataset Model</td><td colspan="4">TL;DR Summarization</td><td colspan="2">AlpacaFarm</td></tr><tr><td>P 1B</td><td>Q 1.5B</td><td>P 2.8B</td><td>Q 3B</td><td>Q 1.5B</td><td>Q 3B</td></tr><tr><td>SFT</td><td> $15.7 \pm 0.2$ </td><td> $27.3 \pm 0.3$ </td><td>25.2</td><td>35.6</td><td>37.8</td><td>47.9</td></tr><tr><td>Dr.GRPO</td><td> $21.0 \pm 5.4$ </td><td> $66.9 \pm 3.1$ </td><td>68.0</td><td>90.2</td><td>32.6</td><td>46.9</td></tr><tr><td>UWO</td><td> $6.1 \pm 3.6$ </td><td> $59.6 \pm 5.3$ </td><td>38.1</td><td>81.8</td><td>38.3</td><td>46.9</td></tr><tr><td>BSPO</td><td> $40.4 \pm 1.2$ </td><td> $78.3 \pm 0.7$ </td><td>50.0</td><td>63.1</td><td>31.3</td><td>43.8</td></tr><tr><td>AdvPO</td><td> $47.0 \pm 2.6$ </td><td> $58.1 \pm 1.0$ </td><td>69.1</td><td>90.0</td><td>37.5</td><td>50.0</td></tr><tr><td>SignCert-PO (ours)</td><td> $60.0 \pm 2.0$ </td><td> $66.0 \pm 1.9$ </td><td>73.8</td><td>91.8</td><td>47.9</td><td>52.3</td></tr></table>

Table 1: Win rate (%) vs. human-generated reference response after training, as judged by the gold RM. For TL;DR experiments on Pythia 1B and Qwen2.5 1.5B, we run three different random seeds and report mean standard deviation. The other experiments use a single seed. We abbreviate Pythia as P and Qwen2.5 as Q. Best result per column is shown in bold.
1. SFT. A base LLM is fine-tuned on reference responses to obtain the SFT policy π<sub>SFT</sub>.

2. Preference data generation. For each prompt $x ,$ two candidate responses are sampled from $\pi _ { \mathrm { S F T } }$ and labeled by the gold RM to form pairwise preference data.

3. Proxy RM training. A proxy reward model is trained on the generated preference data using the Bradley–Terry objective (1). We add a randomly initialized scalar head on top of the same base LLM backbone as the SFT model $\pi _ { \mathrm { S F T } }$ . During RM training, although we derived (6) under the frozen-backbone assumption, we train the full RM parameters to test broader applicability of our proposed method.

4. Policy optimization and evaluation. The SFT policy is optimized against the proxy RM using one of the methods described below. The resulting policy is evaluated by the gold RM that was not used during policy optimization.

We conduct experiments on the foundational summarization from human feedback task (TL;DR) (Stiennon et al., 2020) and on the AlpacaFarm benchmark (Dubois et al., 2023). We use Pythia (Biderman et al., 2023) and Qwen2.5 (Qwen et al., 2025) series as base models. For Pythia 1B and Qwen2.5 1.5B policies, the gold RM is Skywork-Reward-Llama-3.1-8Bv0.2 (Liu et al., 2024), one of the top-performing discriminative RMs on RewardBench (Malik et al., 2025). For larger policy models (Pythia 2.8B and Qwen2.5 3B), we use OpenAI’s GPT-4.1 Nano as the gold RM. More experimental details are given in Appendix F.

## 4.1 Comparison with baselines

We compare our proposed method with the SFT policy π<sub>SFT</sub>, Dr.GRPO (Liu et al., 2025), Uncertainty Weighted Optimization (UWO) (Coste et al., 2024), Behavior-Supported Policy Optimization (BSPO) (Dai et al., 2025), and AdvPO (Zhang et al., 2024). For UWO, we use an ensemble of three RMs and smaller individual RMs to control total model capacity, following the setup of Zhang et al. (2024).

<table><tr><td>Method</td><td>Pythia 1B</td><td>Qwen2.5 1.5B</td></tr><tr><td>Dr.GRPO</td><td>7.54</td><td>10.67</td></tr><tr><td>UWO</td><td>7.48</td><td>10.76</td></tr><tr><td>BSPO</td><td>9.92</td><td>13.37</td></tr><tr><td>AdvPO</td><td>10.93</td><td>16.32</td></tr><tr><td>SignCert-PO</td><td>7.47</td><td>10.78</td></tr></table>

Table 2: Average wall-clock time per policy optimization step (seconds) on 8 GPUs over three runs on TL;DR dataset, as in Table 1. SignCert-PO roughly matches Dr.GRPO.

Table 1 presents the gold-RM win rate for each method across both benchmarks and model scales. On TL;DR summarization, SignCert-PO achieves the highest win rate in most settings.

We observe larger gains when the proxy RM is relatively small, which we believe is due to smaller proxy RMs becoming inaccurate faster (Appendix G.2) and thus needing robustness more. As the policy and proxy RM scale up, the margin narrows but SignCert-PO remains competitive or superior. We also note that the performance of UWO on the TL;DR task with Pythia 1B does not improve even with larger proxy RMs (Table 7 in Appendix G.1).

(a) Pythia-1B: KL–RM accuracy

(b) Pythia-1B: KL–reward
Figure 2: SignCert-PO keeps the policy in regions where the proxy RM remains reliable, preventing reward hacking. KL divergence trade-offs on TL;DR. Left: proxy RM accuracy vs. KL. SignCert-PO maintains higher RM accuracy at every KL budget. Right: gold reward (solid) and proxy reward (dashed) vs. KL. Baselines exhibit reward hacking, whereas SignCert-PO avoids this divergence. The reference policy is the SFT model π<sub>SFT</sub>.

Runtime. Table 2 compares the per-step wall-clock time across methods. SignCert-PO adds negligible overhead to Dr.GRPO: the only additional computation is evaluating the coefficients $\left| \rho _ { j } ^ { * } \right|$ (10) from the RM’s last-layer features, which are already extracted during reward scoring. UWO matches Dr.GRPO in training time, but requires loading multiple ensemble members, increasing VRAM usage when using full-sized RMs. BSPO and AdvPO incur additional training cost due to their modified optimization procedures.

## 4.2 Additional experiments and analysis

We further perform additional experiments to better understand our proposed method.

How SignCert-PO mitigates reward hacking. Figure 2 (left) shows that as the policy drifts from π<sub>SFT</sub> (i.e., increasing KL divergence), the proxy RM’s accuracy on on-policy completions degrades. Dr.GRPO’s accuracy drops toward 50% at high KL as a result, while SignCert-PO follows a different trajectory. Our proposed method keeps the policy in regions where the proxy RM maintains accuracy around 60%, even as optimization progresses. This is a consequence of the re-weighting mechanism: completions whose advantage signs are easy to flip receive reduced gradient weight, preventing the policy from being pulled toward regions where the proxy RM is unreliable. This RM accuracy degradation can lead to reward hacking, as suggested by Figure 2 (right). Dr.GRPO’s proxy reward (dashed) continues to rise even as its gold reward (solid) collapses. SignCert-PO achieves a comparable or even better gold reward at a fraction of the KL budget and avoids the collapse.

Comparison with early stopping. The gold reward curves in Figure 2b also demonstrate the comparison with early stopping. Dr.GRPO with β=0.01 (the one also used for SignCert-PO) reaches peak gold reward around KL 100, after which the reward degrades. An ideal early-stopping mechanism that stops training at this point would recover its peak gold reward, which is still lower than the gold reward after training by SignCert-PO. In contrast, SignCert-PO converges to a favorable region without relying on external evaluation.

Empirical validation of the certified radius. In §3, we assumed linear head perturbation when deriving the certified sign-preservation radius $\Delta _ { j } \left( 8 \right)$ , in order to simplify computation during policy optimization. We thus test whether $\Delta _ { j }$ predicts robustness under the other perturbation classes discussed in §3.1 and show the results in Figure 3. We show (i) whole-model perturbation, where all RM parameters are perturbed, (ii) input embedding perturbation, where noise is added before the transformer backbone, and (iii) cross-model agreement, measuring whether the proxy RM’s advantage signs match those of the gold


Figure 3: $\Delta _ { j }$ (8) is predictive of sign robustness beyond the linear head assumption, on the TL;DR task for Pythia 1B. Left axis: agreement with other perturbation models, where $A _ { j } ^ { \prime }$ is the advantage recomputed under whole-RM or input embedding perturbation. Right axis: agreement with the gold RM. See Appendix F.5 for details.
Figure 4: SignCert-PO provides the largest gains when preference data is limited, with the gap narrowing as more data becomes available. Gold win rate vs. number of preference data epochs on TL;DR for the Pythia 1B proxy RM. We also observe overfitting of the proxy RM for 2.3M pairs.

RM. In all three settings, completions with larger $\Delta _ { j }$ consistently demonstrate higher signpreservation or agreement rates. Furthermore, the Spearman rank correlation coefficient between $\Delta _ { j }$ and sign preservation rate is 0.72 for whole-model perturbation and 0.67 for input-embedding perturbation. These results suggest that $\Delta _ { j }$ captures sensitivity structure beyond our theoretical assumption.

Sensitivity to proxy RM quality. The proxy RM’s reliability depends on the amount of preference data used for its training. Figure 4 varies the number of generated preference pairs (from 116K at 1 epoch to 2.3M at 20 epochs) and compares the win rate of Dr.GRPO and SignCert-PO on Pythia 1B. SignCert-PO provides the largest gains when the preference data is limited. With only 116K preference pairs (i.e., one pair per prompt in the dataset), SignCert-PO achieves significantly higher win rate compared to Dr.GRPO. This is the most practically relevant regime, since collecting large preference datasets is expensive. While the benefit of our method decreases for larger datasets, results show that it does not hurt performance when the RM is already reliable. Finally, for more than 1M pairs we observe overfitting of the proxy RM and the results for both methods degrade.

## 5 Conclusion

We presented SignCert-PO, a lightweight method for RLHF that mitigates reward hacking by down-weighting policy gradient contributions of completions whose advantage signs are fragile under RM parameter perturbations. Our approach derives a certified sign-preservation radius that quantifies, for each completion, the smallest perturbation to the RM that can flip the advantage sign. This per-completion formulation yields a simple re-weighting mechanism of the standard PG update, requiring only the current RM head parameters and on-policy completions. Experiments on TL;DR summarization and AlpacaFarm across Pythia and Qwen2.5 model families show that SignCert-PO achieves the highest gold-RM win rate in most settings while adding negligible computational overhead.

Our method makes several simplifying assumptions (§3.2). Further, our per-sample adversary formulation is more conservative than a shared-adversary variant. Analyzing the tighter coupled formulation is a future direction. Finally, extending the certified signpreservation radius to general RMs without incurring the cost of full-parameter gradient norm computation remains an open problem.

## Acknowledgement

This research was conducted using the Supermicro ARS-111GL-DNHR-LCC and FUJITSU Server PRIMERGY CX2550 M7 (Miyabi) at Joint Center for Advanced High Performance Computing (JCAHPC).

## Ethics Statement

This work proposes a new algorithm that mitigates reward hacking in RLHF, contributing to the development of more reliable and aligned language models. Our method does not involve the collection of new human preference data or human subjects. We use only publicly available datasets and models in our experiments. We do not foresee specific ethical risks arising from this work beyond those inherent to the RLHF framework.

## Reproducibility statement

All experiments are conducted on a shared compute cluster. Each training job uses 8 NVIDIA GH200 nodes, each equipped with an NVIDIA H100 GPU. Response generation during the gold-model pipeline uses vLLM for efficient batched inference. Complete hyperparameter settings for SFT, RM training and policy optimization are listed in Appendix F.4 (Tables 4–6). We use the publicly available dataset<sup>2</sup> for our TL;DR experiments. For BSPO and AdvPO, we did not find the public implementation and thus we implemented them ourselves, based on the respective papers. For the 1-epoch proxy RM experiments on Pythia-1B and <sup>Qwen2.5-1.5B,</sup> <sup>we</sup> <sup>report</sup> <sup>mean</sup> ± <sup>standard</sup> <sup>deviation</sup> <sup>over</sup> <sup>3</sup> <sup>random</sup> <sup>seeds,</sup> <sup>corresponding</sup> to different SFT models, sampled preference datasets, proxy RMs and resulting policies.

## References

Johannes Ackermann, Takashi Ishida, and Masashi Sugiyama. Off-policy corrected reward modeling for reinforcement learning from human feedback. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=0zxugBcgF5#discussion.

Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust Optimization. Princeton University Press, 2009. ISBN 978-1-4008-3105-0. URL https://press.princeton.edu/ books/hardcover/9780691143682/robust-optimization.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. Pythia: a suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of ICML’23, pp. 2397–2430. JMLR.org, 2023. URL https://proceedings.mlr.press/v202/biderman23a/biderman23a. pdf.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3):324–345, 1952. ISSN 0006-3444. doi: 10.2307/2334029. URL https://www.jstor.org/stable/2334029.

Daiki Chijiwa, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Susumu Takeuchi. Portable reward tuning: Towards reusable fine-tuning across different pretrained models. In Proceedings of the 42nd International Conference on Machine Learning, pp. 10428–10448. PMLR, 2025. URL https://proceedings.mlr.press/v267/chijiwa25a.html.

Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 4302–4310. Curran Associates Inc., 2017. URL https://dl.acm.org/doi/10.5555/3294996.3295184.

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In Proceedings of the 36th International Conference on Machine Learning, pp. 1310–1320. PMLR, 2019. URL https://proceedings.mlr.press/v97/cohen19c.html.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=dcjtMYkpXx.

Juntao Dai, Taiye Chen, Yaodong Yang, Qian Zheng, and Gang Pan. Mitigating reward over-optimization in RLHF via behavior-supported regularization. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=PNMv4r7s1i.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. AlpacaFarm: a simulation framework for methods that learn from human feedback. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NeurIPS’23, pp. 30039–30069. Curran Associates Inc., 2023. URL https://proceedings.neurips.cc/paper files/paper/2023/ file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alexander Nicholas D’Amour, Krishnamurthy Dj Dvijotham, Adam Fisch, Katherine A. Heller, Stephen Robert Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=5u1GpUkKtG.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of ICML’23, pp. 10835–10866. PMLR, 2023. URL https://proceedings.mlr.press/v202/ gao23h/gao23h.pdf.

Scott Geng, Hamish Ivison, Chun-Liang Li, Maarten Sap, Jerry Li, Ranjay Krishna, and Pang Wei Koh. The delta learning hypothesis: Preference tuning on weak data can yield strong gains. In ICLR 2025 Workshop on Navigating and Addressing Data Problemsfor Foundation Models, 2025. URL https://openreview.net/forum?id=cVlY21dIVE.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=nZeVKeeFYf9.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in LLMs, 2024. URL http://arxiv.org/abs/2410.18451.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025. URL http://arxiv.org/abs/2503.20783.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi, and Nathan Lambert. RewardBench 2: Advancing reward model evaluation, 2025. URL http://arxiv.org/abs/2506.01937.

Soichiro Nishimori, Yu-Jie Zhang, Thanawat Lodkaew, and Masashi Sugiyama. On symmetric losses for robust policy optimization with noisy preferences, 2025. URL https://arxiv.org/abs/2505.24709.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie

Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simon Posada Fishman,´ Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mely, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan,´ Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Fe lipe Ceron Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright,´ Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, C. J. Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. GPT-4 technical report, 2024. URL http://arxiv.org/abs/2303.08774.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NeurIPS’22, pp. 27730–27744. Curran Associates Inc., 2022. URL https://proceedings.neurips.cc/paper files/paper/2022/file/ b1efde53be364a73914f58805a001731-Paper-Conference.pdf.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang

Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL http://arxiv.org/abs/2412.15115.

Carl O. Retzlaff, Alessa Angerschmid, Anna Saranti, David Schneeberger, Richard Rottger,¨ Heimo Muller, and Andreas Holzinger. Post-hoc vs ante-hoc explanations: xAI design¨ guidelines for data scientists. Cognitive Systems Research, 86:101243, 2024. ISSN 1389- 0417. doi: 10.1016/j.cogsys.2024.101243. URL https://www.sciencedirect.com/science/ article/pii/S1389041724000378.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL http://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL http://arxiv.org/abs/ 2402.03300.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback. In Proceedings of the 34th International Conference on Neural Information Processing Sys tems, NeurIPS’20, pp. 3008–3021. Curran Associates Inc., 2020. URL https://proceedings. neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. Adaptive Computation and Machine Learning series. MIT Press, 2 edition, 2018. ISBN 978-0-262- 03924-6. URL https://mitpress.mit.edu/9780262039246/reinforcement-learning/.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, volume 12. MIT Press, 1999. URL https://papers.nips. cc/paper files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html.

Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Remi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Avila Pires, and Bilal Piot. Generalized preference optimization: A unified approach to offline alignment. In Proceedings of the 41st International Conference on Machine Learning, pp. 47725–47742. PMLR, 2024. URL https://proceedings.mlr.press/v235/tang24b.html.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.

Sheng Xu, Bo Yue, Hongyuan Zha, and Guiliang Liu. Uncertainty-aware preference alignment in reinforcement learning from human feedback. In ICML 2024 Workshop on Models of Human Feedbackfor AI Alignment, 2024. URL https://openreview.net/forum?id= oVVg7lug1F.

Yuanzhao Zhai, Yu Lei, Han Zhang, Yue Yu, Kele Xu, Dawei Feng, Bo Ding, and Huaimin Wang. Uncertainty-penalized reinforcement learning from human feedback with diversified reward LoRA ensembles. Information Processing & Management, 63(3):104548, 2026. ISSN 0306-4573. doi: 10.1016/j.ipm.2025.104548. URL https://www.sciencedirect.com/ science/article/pii/S0306457325004893.

Xiaoying Zhang, Jean-Franc¸ois Ton, Wei Shen, Hongning Wang, and Yang Liu. Mitigating reward overoptimization via lightweight uncertainty estimation. In Proceedings of the 38th International Conference on Neural Information Processing Systems, volume 37 of NeurIPS’24, pp. 81717–81747. Curran Associates Inc., 2024. URL https://proceedings.neurips.cc/paper files/paper/2024/file/ 94bbcb744bbada8808fda05b9d9290d6-Paper-Conference.pdf.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2020. URL http://arxiv.org/abs/1909.08593.

## A Additional background and related work

We present a more comprehensive discussion of related work here.

## A.1 Policy gradient algorithms

Proximal Policy Optimization (PPO). PPO (Schulman et al., 2017) is a canonical PGbased method for RLHF that uses a clipped surrogate objective and jointly trains a value network for advantage estimation. Given responses sampled from an old policy $\pi _ { \phi _ { \mathrm { o l d } } } ,$ PPO maximizes the clipped surrogate objective:

$$
L ^ {\mathrm{PPO}} (\phi) = \mathbb {E} _ {x \sim P, y \sim \pi_ {\phi_ {\mathrm{old}}}} \left[ \frac {1}{T} \sum_ {t = 1} ^ {T} \min \left(\nu_ {t} (\phi) \hat {A} _ {t}, \operatorname{clip} \left(\nu_ {t} (\phi), 1 - \varepsilon_ {\operatorname{clip}}, 1 + \varepsilon_ {\operatorname{clip}}\right) \hat {A} _ {t}\right) \right],\tag{14}
$$

where $T = | y |$ is the response length, $\varepsilon _ { \mathrm { c l i p } } > 0$ is the clipping hyperparameter, and

$$
\nu_ {t} (\phi) := \frac {\pi_ {\phi} (y _ {t} \mid x , y _ {<   t})}{\pi_ {\phi_ {\mathrm{old}}} (y _ {t} \mid x , y _ {<   t})}
$$

is the density ratio. The token-level advantage $\hat { A } _ { t }$ is estimated using a learned value network $V _ { \omega } ( x , y _ { < t } )$ with parameters $\omega ,$ trained jointly with the policy:

$$
\hat {A} _ {t} = r _ {t} + \gamma V _ {\omega} (x, y _ {\leq t}) - V _ {\omega} (x, y _ {<   t}),\tag{15}
$$

where $\gamma \in \ [ 0 , 1 ]$ is the discount factor and $r _ { t }$ is the per-token reward. In RLHF, $r _ { t } ~ =$ $- \beta \log \left( \pi _ { \phi } ( y _ { t } \mid x , y _ { < t } ) / \pi _ { \mathrm { r e f } } ( y _ { t } \mid x , y _ { < t } ) \right)$ for all $t ,$ with the RM score $r _ { \theta } ( x , y )$ added at the final token $t = T$ (Ziegler et al., 2020).

Group Relative Policy Optimization (GRPO). GRPO (Shao et al., 2024) replaces the learned value network in PPO with a group-based Monte Carlo baseline. For each prompt $x \in { \mathcal { X } } ,$ a group of $K$ completions $y ^ { ( 1 ) } , \ldots , y ^ { ( K ) } \in \mathcal { V }$ is sampled from the current policy $\pi _ { \phi } ( \cdot | x )$ , and each receives a reward $r _ { j } : = r _ { \theta } ( x , y ^ { ( j ) } )$ . In the original GRPO, the advantage is the normalized reward: $\widehat { A } _ { j } : = ( r _ { j } - \mathrm { m e a n } ( \mathbf { r } ) ) / \mathrm { s t d } ( \mathbf { r } )$ , where r is the vectorized reward $( r _ { 1 } , \ldots , r _ { K } ) ^ { \mathsf { T } }$ , and mean( ) and std( ) denote the mean and standard deviation of the elements of a vector, respectively.

GRPO can be viewed as replacing PPO’s learned value baseline with a within-group relative baseline computed from sampled completions. This removes the need to train a separate value network, allowing for preserving VRAM and scaling up the RLHF pipeline.

Dr.GRPO (Liu et al., 2025) is a widely-used variant that removes the standard-deviation normalization to avoid biased gradient estimation (see §2.1 in the main text).

## A.2 Additional related work

Reward hacking arises when the policy distribution drifts outside the region where the proxy RM is reliable. Prior work addresses this through ensemble-based uncertainty, modified RM training, or adversarial robustness.

RM ensembles. Ensemble-based methods mitigate reward hacking by optimizing a policy conservatively with respect to multiple RMs. Coste et al. (2024) train several proxy RMs and study conservative objectives such as worst-case and uncertainty-weighted optimization (UWO), showing that ensembles reduce reward hacking in the gold-RM setup of Gao et al. (2023). Eisenstein et al. (2024) observe that ensembles still exploit spurious features when all members agree on them. Zhai et al. (2026) propose LoRA (Hu et al., 2022)-based RM ensembles to reduce the cost of training multiple full RMs. These approaches require multiple RM training and inference runs.

Distribution shift and modified RM training. Reward hacking can also be viewed as a distribution shift problem: as the policy moves away from the SFT distribution, the RM’s training data becomes unrepresentative. Ackermann et al. (2025) propose off-policy corrected reward modeling, which uses importance weighting to make the RM consistent under the evolving policy distribution. Dai et al. (2025) propose BSPO, which regularizes the RM training objective to improve robustness. Xu et al. (2024) train distributional RMs with explicit uncertainty outputs. These methods modify the RM training pipeline, whereas SignCert-PO operates purely at policy optimization time using any pre-trained RM.

Adversarial policy optimization. The closest work to ours is AdvPO (Zhang et al., 2024), which derives single-RM uncertainty estimates from last-layer embeddings and formulates a distributionally robust policy optimization problem. Two differences distinguish our approach. First, AdvPO requires constructing a matrix $M _ { D }$ from the RM training dataset D at policy optimization time; in many practical pipelines, this data is unavailable when using third-party RMs (Malik et al., 2025; Chijiwa et al., 2025). SignCert-PO requires only the current head parameters and on-policy completions. Second, AdvPO constructs a confidence region for the reward function using that matrix and then optimizes the policy against a pessimistic reward within that region. In contrast, SignCert-PO models uncertainty as perturbations of the RM parameters and asks whether those perturbations can flip the sign of the group-relative advantage for each completion. This aligns our adversarial formulation directly with GRPO’s update geometry, as discussed in §3.3.

## B Proofs

In what follows, $h _ { \psi } ( x , y )$ is referred to as $h ( x , y )$ for brevity, and $h _ { j }$ denotes $h ( x , y ^ { ( j ) } )$ . We also write $A _ { j }$ instead of $A _ { j } ( w )$ when the context is clear. We also sometimes omit the outer expectation $x \sim P ( x )$

Theorem B.1 (Certified radius (Theorem 3.1, restated)). The certified sign-preservation radius for sample j under head parameter uncertainty is

$$
\Delta_ {j} ^ {\text { param }} = \frac {| A _ {j} |}{\| h _ {j} - \bar {h} \| _ {2}}.
$$

ProofofTheorem 3.1. We first establish how the advantage changes under head parameter perturbation. By linearity of the reward head,

$$
\begin{array}{l} A _ {j} (w ^ {\prime}) = (w ^ {\prime}) ^ {\mathsf {T}} h _ {j} + b - \frac {1}{K} \sum_ {k = 1} ^ {K} \left((w ^ {\prime}) ^ {\mathsf {T}} h _ {k} + b\right) \\ \qquad = (w + \delta) ^ {\mathsf {T}} h _ {j} - \frac {1}{K} \sum_ {k = 1} ^ {K} (w + \delta) ^ {\mathsf {T}} h _ {k} \\ \qquad = A _ {j} (w) + \delta^ {\mathsf {T}} (h _ {j} - \bar {h}), \end{array}
$$

where $\begin{array} { r } { \bar { h } \ = \ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } h _ { k } } \end{array}$ . The worst-case perturbation for flipping the sign of $A _ { j } ~ > ~ 0$ is $\delta ^ { * } = - \epsilon \cdot \left( h _ { j } - \bar { h } \right) / \| h _ { j } - \bar { h } \| _ { 2 } ,$ yielding $A _ { j } ( w + \delta ^ { * } ) = A _ { j } - \epsilon \| h _ { j } - \bar { h } \| _ { 2 }$ . The sign flips when this becomes non-positive, i.e., when $\epsilon \geq | A _ { j } | / \| h _ { j } - \bar { h } \| _ { 2 }$ . The case $A _ { j } < 0$ is symmetric.

Theorem B.2 (Worst-case advantage under per-completion adversary (Theorem 3.2, restated)). Under the linear head model and the uncertainty set $\mathcal { U } _ { \epsilon } ^ { w } \left( 7 \right)$ , the worst-case advantage (9) for the j-th completion is

$$
\tilde {A} _ {j} = \rho_ {j} ^ {*} \cdot A _ {j}, \quad w h e r e \quad \rho_ {j} ^ {*} := 1 - \frac {\epsilon}{\Delta_ {j}} = 1 - \frac {\epsilon \| h _ {j} - \bar {h} \| _ {2}}{| A _ {j} |}.
$$

Proof of Theorem 3.2. For $A _ { j } > 0 ,$ , the worst case under $w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { w }$ is $\begin{array} { r } { \operatorname* { i n f } _ { w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { w } } A _ { j } ( w ^ { \prime } ) = A _ { j } - } \end{array}$ $\epsilon \| h _ { j } - \bar { h } \| _ { 2 } = A _ { j } ( 1 - \epsilon / \Delta _ { j } )$ , where the first equality follows from the perturbation analysis in the proof of Theorem 3.1. For $A _ { j } < 0 ,$ , the analogous worst case is $\begin{array} { r } { \operatorname* { s u p } _ { w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { w } } A _ { j } ( w ^ { \prime } ) = } \end{array}$ $A _ { j } + \epsilon \| h _ { j } - \bar { h } \| _ { 2 } = A _ { j } ( 1 - \epsilon / \Delta _ { j } )$ , where we used $| A _ { j } | = - A _ { j }$ . In both cases, the worst-case advantage equals $\rho _ { j } ^ { * } \cdot A _ { j }$ with $\rho _ { j } ^ { * }$ given by (10). □

Lemma B.3 (Policy gradient of the global robust objective (Lemma $3 . 3 ,$ restated)). Under the GRPO sampling with K on-policy completions per prompt, the PG of the global robust objective is

$$
\nabla_ {\phi} J _ {\mathrm{global}} (\phi , \theta) = \mathbb {E} _ {x} \left[ \sum_ {j = 1} ^ {K} A _ {j} \nabla_ {\phi} \log \pi_ {\phi} (y ^ {(j)} | x) \right] - \epsilon \nabla_ {\phi} \big \| \mathbb {E} _ {y \sim \pi_ {\phi}} [ h (x, y) ] \big \| _ {2}.
$$

ProofofLemma 3.3. We first derive the worst-case objective. Since the KL term is independent of $w ^ { \prime } ,$ , the infimum reduces to in $\begin{array} { r } { \mathbf { f } _ { w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { w } } \mathbb { E } _ { y \sim \pi _ { \phi } } [ { w ^ { \prime } } ^ { \mathsf { T } } h ( x , y ) + b ] = w ^ { \mathsf { T } } \mathbb { E } _ { \pi } [ h ] + } \end{array}$ $\begin{array} { r } { b - \mathsf { s u p } _ { \parallel \delta \parallel _ { 2 } \le \epsilon } \delta ^ { \mathsf { T } } \mathbb { E } _ { \pi } [ h ] = \mathbb { E } _ { \pi } [ r _ { w } ( x , y ) ] - \epsilon \parallel \mathbb { E } _ { \pi } [ h ] \parallel _ { 2 } } \end{array}$ , where the supremum is attained at $\delta ^ { * } = \epsilon \mathbb { E } _ { \pi } ^ { * } [ \bar { h } ] / \| \mathbb { E } _ { \pi } [ h ] \| _ { 2 }$ by the Cauchy–Schwarz inequality. Thus,

$$
J _ {\mathrm{global}} (\phi , w) = J (\phi , w) - \epsilon \big \| \mathbb {E} _ {y \sim \pi_ {\phi}} [ h (x, y) ] \big \| _ {2}.
$$

The first term of the PG is the standard GRPO gradient of ${ \cal J } ( \phi , w )$ . The second term follows from differentiating the penalty $- \epsilon \| \mathbb { E } _ { \pi } [ h ] \| _ { 2 }$ through the policy, which depends on $\phi$ via $\pi _ { \phi }$ but not on any per-completion quantity.

## C Connection with randomized smoothing

In §3.1, we defined the certified sign-preservation radius $\Delta _ { j } ~ ( 6 )$ as the largest perturbation under which the sign of the j-th completion’s advantage remains constant. This section shows that $\Delta _ { j }$ admits a formal interpretation through RS in the binary classification case.

## C.1 Randomized smoothing interpretation of the certified sign-preservation radius

Consider the binary decision given by the sign of the advantage, and inject Gaussian noise $\xi \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ into the space where uncertainty is modeled $( \mathrm { e . g . }$ , the RM head parameter space). In RS, the certified radius for a binary decision is

$$
R = \sigma \Phi^ {- 1} (p),
$$

where p is the probability that the decision is preserved under the Gaussian perturbation (Cohen et al., 2019). For linear perturbations, the sign-preservation probability has the form $p = \Phi ( | A _ { i } | / ( \sigma \cdot \bar { c } ) )$ for some constant c. Substituting into the RS formula recovers exactly the certified radii in (8) and (16).

The above results imply that responses with low coefficients $\rho _ { j } ^ { * }$ , or equivalently low radii $\Delta _ { j } ^ { \mathrm { p a r a m } }$ , contribute less to the policy update with the robust policy optimization objective. The radius $\Delta _ { j } ^ { \mathrm { p a r a m } } = | A _ { j } | / \| h _ { j } - \bar { h } \| _ { 2 }$ is small when:

• The advantage $| A _ { j } |$ is small. In this case, the preference signal is weak and easily flipped.

• The feature deviation $| | h _ { j } - { \bar { h } } | | _ { 2 }$ is large. This means the response is an outlier in feature space, where the RM may be exploring.

The first category corresponds to the delta learning hypothesis (Geng et $\mathsf { a l . }$ , 2025). Our theoretical results suggest that reward hacking also exploits responses in the second category. Template-like or verbose outputs that achieve high rewards through spurious correlations often lie in sharp, high-norm feature directions, yielding small certified radii despite large advantages.

<table><tr><td>Model</td><td>Uncertainty set</td><td>Certified radius  $\Delta_{j}$ </td><td>Interpretation</td></tr><tr><td>Head param.</td><td> $\|w' - w\|_2 \leq \epsilon$ </td><td> $|A_j| / \|h_j - \bar{h}\|_2$ </td><td>Feature geometry</td></tr><tr><td>Feature</td><td> $\|h' - h\|_2 \leq \delta$ </td><td> $|A_j| / \|w\|_2$ </td><td>Advantage magnitude</td></tr><tr><td>Full param.</td><td> $\| \theta' - \theta \|_2 \leq \tau$ </td><td> $|A_j| / \| \nabla_\theta A_j\|_2$ </td><td>Gradient sharpness</td></tr></table>

Table 3: Summary of uncertainty models and their certified radii.

## C.2 Alternative uncertainty models

The same derivation applies to the other uncertainty models, yielding analogous certified radii.

Feature uncertainty. Under perturbation $h _ { j } ^ { \prime } = h _ { j } + \xi _ { j }$ with $\xi _ { j } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma _ { h } ^ { 2 } I )$ , the perturbed advantage is

$$
\tilde {A} _ {j} = A _ {j} + w ^ {\top} \left(\xi_ {j} - \operatorname{mean} (\boldsymbol {\xi})\right).
$$

Since $\xi _ { j } - \mathrm { m e a n } ( \xi ) \sim \mathcal { N } ( 0 , \sigma _ { h } ^ { 2 } ( 1 - 1 / K ) I )$ , the perturbed advantage is Gaussian:

$$
\tilde {A} _ {j} \sim \mathcal {N} \left(A _ {j}, \sigma_ {h} ^ {2} (1 - 1 / K) \| w \| _ {2} ^ {2}\right).
$$

The probability of sign preservation and certified radius are:

$$
p _ {j} ^ {\text { feat }} = \Phi \left(\frac {| A _ {j} |}{\sigma_ {h} \| w \| _ {2} \sqrt {1 - 1 / K}}\right), \quad \Delta_ {j} ^ {\text { feat }} = \frac {| A _ {j} |}{\| w \| _ {2}}.\tag{16}
$$

Note that $\Delta _ { j } ^ { \mathrm { f e a t } }$ is a monotone function of $| A _ { j } |$ , unlike $\Delta _ { j } ^ { \mathrm { p a r a m } }$ which also depends on feature geometry.

Full parameter uncertainty. Under first-order approximation with $\theta ^ { \prime } \sim \mathcal { N } ( \theta , \sigma _ { \theta } ^ { 2 } I ) ;$

$$
A _ {j} (\theta^ {\prime}) \approx A _ {j} (\theta) + \nabla_ {\theta} A _ {j} (\theta) ^ {\top} (\theta^ {\prime} - \theta).
$$

This yields:

$$
p _ {j} ^ {\text { full }} = \Phi \left(\frac {| A _ {j} |}{\sigma_ {\theta} \| \nabla_ {\theta} A _ {j} \| _ {2}}\right), \quad \Delta_ {j} ^ {\text { full }} = \frac {| A _ {j} |}{\| \nabla_ {\theta} A _ {j} \| _ {2}}.
$$

We expect spurious patterns to have large gradient norms $\| \nabla _ { \theta } A _ { j } \| _ { 2 } ,$ , as the RM assigns high rewards through sharp, non-robust computations. Such patterns receive small radii and are appropriately down-weighted.

Summary of uncertainty models. Table 3 summarizes the three models.

## D Robust optimization perspective and its limitations

The re-weighted gradient (11) can be viewed as the PG of a per-completion robust optimization problem. For each completion, the adversary chooses $w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { \dot { w } }$ to maximally corrupt that completion’s gradient contribution:

$$
J _ {\text {robust}} (\phi) = \mathbb {E} _ {x \sim P} \left[ \sum_ {j = 1} ^ {K} \inf _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} A _ {j} (w ^ {\prime}) \cdot \mathbf {1} [ A _ {j} (w) > 0 ] + \sup _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} A _ {j} (w ^ {\prime}) \cdot \mathbf {1} [ A _ {j} (w) <   0 ] \right].\tag{17}
$$

This formulation takes the inf and sup inside the sum over completions, allowing the adversary to choose different perturbations for each completion.

We now argue that this per-sample decomposition is a necessity: any formulation that robustifies an aggregate advantage is vacuous under GRPO’s mean-subtracted structure.

Zero-sum obstruction. A more standard robust optimization approach (Ben-Tal et al., 2009) would robustify the aggregate advantage directly, e.g., in $\begin{array} { r } { \dot { \mathbf { \sigma } } _ { w ^ { \prime } \in \mathcal { U } _ { \epsilon } ^ { w } } \sum _ { j = 1 } ^ { K } A _ { j } ( w ^ { \prime } ) } \end{array}$ . However, under the linear head model, for any $\boldsymbol w ^ { \prime } \in \mathbb R ^ { d }$

$$
\sum_ {j = 1} ^ {K} A _ {j} (w ^ {\prime}) = (w ^ {\prime}) ^ {\mathsf {T}} \sum_ {j = 1} ^ {K} (h _ {j} - \bar {h}) = 0,
$$

since $\begin{array} { r } { \sum _ { j = 1 } ^ { K } ( h _ { j } - \bar { h } ) = 0 } \end{array}$ by definition of <sup>¯</sup>h. Group-relative advantages sum to zero regardless of the RM parameters, so any robust objective of the form in $\begin{array} { r } { \mathrm { f } _ { w ^ { \prime } } \sum _ { j } A _ { j } ( w ^ { \prime } ) } \end{array}$ evaluates to zero trivially and provides no useful signal for policy optimization.

This complements Lemma 3.3, which showed that the global robust reward objective yields only a completion-agnostic penalty (13).

Conservatism of the per-sample formulation. The per-sample objective (17) circumvents the zero-sum obstruction by partitioning the sum according to the sign of ${ \dot { \boldsymbol { A } } } _ { j } ( { \boldsymbol { w } } )$ before applying the inf and sup. This is a stronger $( \mathrm { i . e . } $ , more conservative) setting than the standard robust optimization formulation, where a single worst-case parameter $\widetilde { w ^ { \prime } }$ would be shared across all terms:

$$
J _ {\text {robust}} ^ {\text {shared}} (\phi) = \mathbb {E} _ {x \sim P} \left[ \inf _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} \sum_ {j = 1} ^ {K} A _ {j} (w ^ {\prime}) \cdot \mathbf {1} [ A _ {j} (w) > 0 ] + \sup _ {w ^ {\prime} \in \mathcal {U} _ {\epsilon} ^ {w}} \sum_ {j = 1} ^ {K} A _ {j} (w ^ {\prime}) \cdot \mathbf {1} [ A _ {j} (w) <   0 ] \right].\tag{18}
$$

The per-sample formulation provides a tractable upper bound on the conservatism: because each term is optimized independently, the resulting coefficients $\rho _ { j } ^ { * }$ decompose into persample quantities, yielding the closed-form solution in Theorem 3.1. The shared-adversary variant (18), while tighter, couples all completions and requires solving a joint optimization problem over the group. The further analysis of different robust optimizations is left for future work.

## E Extension to other policy gradient methods

The sign-certification principle of $\ S 3$ applies beyond Dr.GRPO to any PG-based RLHF method whose update involves a scalar coefficient depending on the RM. Since the exact form of the certified radius depends on the update geometry, we only discuss RE-INFORCE (Williams, 1992), PPO (Schulman et al., 2017) and vanilla GRPO (Shao et al., 2024).

## E.1 REINFORCE

The simplest case is REINFORCE, or more generally any response-level PG method with advantage

$$
A ^ {\mathrm{R}} (x, y; w) = r _ {w} (x, y) - V (x),
$$

where $\boldsymbol { r } _ { w } ( x , y ) \ : = \ : w ^ { \mathsf { T } } h ( x , y ) \ : + \ : b$ is the linear reward head and $V ( x )$ is a baseline independent of the RM head w $( \mathrm { e . g . , ~ a ~ }$ learned value function, a Monte Carlo return, or simply $V ( x ) \equiv 0$ for vanilla REINFORCE). The PG then takes the form $\nabla _ { \phi } J ( \phi ) \ =$ $\mathbb { E } _ { x , y } [ A ^ { \mathrm { R } } ( x , y ; w ) \nabla _ { \phi } \log \pi _ { \phi } ( y | x ) ]$ , matching the form in (3).

Under the same linear-head uncertainty set ${ { \mathcal U } _ { \epsilon } ^ { w } \left( 7 \right) }$ as in $\ S 3$ , the certified sign-preservation radius admits an exact closed form.

Proposition E.1 (Certified sign-preservation radius for response-level PG). Under the linear reward head and the uncertainty set $\mathcal { U } _ { \epsilon } ^ { w } \left( 7 \right)$ , with a baseline $\dot { V ( x ) }$ independent of the RM head w, the certified sign-preservation radius for response y is

$$
\Delta^ {\mathrm{R}} (x, y) = \frac {| A ^ {\mathrm{R}} (x , y ; w) |}{\| h (x , y) \| _ {2}}.\tag{19}
$$

Proof. Under perturbation $w ^ { \prime } = w + \delta$ with $\| \delta \| _ { 2 } \leq \epsilon \colon$

$$
A ^ {R} (x, y; w ^ {\prime}) = A ^ {R} (x, y; w) + \delta^ {\top} h (x, y),
$$

since $V ( x )$ does not depend on w. By the Cauchy–Schwarz inequality, $| \delta ^ { \top } h ( x , y ) | ~ \leq$ $\epsilon \| h ( x , y ) \| _ { 2 }$ , with equality when $\delta = \perp \epsilon h ( x , y ) / \parallel \tilde { h } ( x , y ) \parallel _ { 2 }$ . The sign flips when $\epsilon \geq$ $| A ^ { \mathrm { R } } ( x , y ; w ) | / \| h ( x , y ) \| _ { 2 }$ □

We can also derive the worst-case advantage in the vein of Theorem 3.2:

Corollary E.2 (Worst-case advantage under per-completion adversary). Under the same setting as Proposition E.1, the worst-case advantage under the per-completion adversary (9) is

$$
\tilde {A} ^ {\mathrm{R}} (x, y) = \rho^ {\mathrm{R}} \cdot A ^ {\mathrm{R}} (x, y; w), \quad \text {where} \quad \rho^ {\mathrm{R}} := 1 - \frac {\epsilon}{\Delta^ {\mathrm{R}} (x , y)} = 1 - \frac {\epsilon \| h (x , y) \| _ {2}}{| A ^ {\mathrm{R}} (x , y ; w) |}.
$$

## E.2 Approximate extensions

Standard PPO and vanilla GRPO do not generally admit a single response-level coefficient whose sign alone determines the update for the whole completion. Sign certification for these methods therefore requires a first-order approximation.

## E.2.1 PPO

Standard PPO in RLHF uses a token-level clipped surrogate objective (14) with token-level advantages $\hat { A } _ { t } \left( 1 5 \right)$ estimated via a learned value network (§A.1). Unlike Dr.GRPO or the response-level PG of §E.1, PPO does not generally admit a single scalar response-level coefficient. Its per-token structure means that the sign of the effective coefficient can differ across tokens within a single response.

Sign certification for PPO should therefore be formulated for the token-level coefficient appearing in each surrogate term. Under the linear reward head, the token-level advantage $\hat { A } _ { t }$ depends on the RM head w only through the terminal reward $r _ { w } ( x , y )$ added at $t = T$ . A first-order approximation gives

$$
\Delta_ {t} ^ {\mathrm{PPO}} \approx \frac {| \hat {A} _ {t} |}{\| \nabla_ {w} \hat {A} _ {t} \| _ {2}},
$$

which is consistent with the general first-order extension in §3.1.

## E.2.2 Vanilla GRPO

In vanilla GRPO (Shao et al., 2024), the advantage is the normalized reward $\hat { A } _ { j } = ( r _ { j } -$ mean(r))/std(r), where $\mathbf { r } = ( r _ { 1 } , \ldots , r _ { K } ) ^ { \mathsf { T } }$ . Because std $\left( \mathbf { r } ( w ^ { \prime } ) \right)$ depends non-linearly on $\delta = w ^ { \prime } - w$ through all K rewards, the perturbed advantage $\hat { A } _ { j } ( w ^ { \prime } )$ is no longer affine in δ. This breaks the exact analysis that yields the Dr.GRPO closed form (8). A first-order expansion yields the approximate radius

$$
\Delta_ {j} ^ {\mathrm{GRPO}} \approx \frac {| \hat {A} _ {j} |}{\| \nabla_ {w} \hat {A} _ {j} \| _ {2}},
$$

which requires the gradient of the normalized advantage with respect to w. This is more expensive than the Dr.GRPO closed form but avoids the full-parameter gradients needed for the general RM extension (§3.1).

## F Training and implementation details

## F.1 Conceptual experiment (Figure 1)

Before presenting the full-scale LLM experiments, we design a controlled conceptual experiment to demonstrate advantage sign flipping causes reward hacking and to verify that SignCert-PO prevents it.

## F.1.1 Setup

We consider a stateless M-armed bandit (M = 8). Each action (completion) m is associated with a fixed feature vector $h _ { m } \in \mathbb { R } ^ { d } \left( d = 6 \right)$ , analogous to the last-layer hidden states $h ( x , y ^ { ( j ) } )$ of an RM backbone in $\ S 3 . 2$ . Both the proxy and true RMs use a linear head on top of these shared features:

$$
r _ {k} = w ^ {\mathsf {T}} h _ {k}, \qquad r _ {k} ^ {*} = w ^ {* \mathsf {T}} h _ {k},
$$

matching the linear-head architecture assumed in §3.2. The feature space $\mathbb { R } ^ { d }$ is split into two subspaces: shared dimensions (0–2), where both heads assign weight 1.0; divergent dimensions (3–5), where the proxy assigns weight 0.4 but the true RM assigns 0.1. Actions 1–3 (reliable positive) use only shared dimensions and have positive proxy and true rewards. Actions 4–6 (reliable negative) also use only shared dimensions but have negative rewards. Actions 7–8 (hacking) use only divergent dimensions. As a result, the proxy RM assigns them the highest advantages in the group, while their true advantages are negative.

The policy is a categorical stochastic policy $\pi ( a _ { m } | \phi ) = \operatorname { s o f t m a x } ( \phi ) _ { m }$ initialized to uniform $( \boldsymbol { \phi } = \mathbf { 0 } )$ . Because M is small and there is a single state, we use the exact expected policy gradient.

## F.1.2 Policy optimization methods

We compare three methods, all sharing the softmax policy gradient structure but differing in how advantages are weighted:

• Dr.GRPO: $\nabla _ { \phi _ { i } } J = \pi _ { i } ( A _ { i } - V _ { \pi } )$ , where $\begin{array} { r } { V _ { \pi } = \sum _ { k } \pi _ { k } A _ { k } } \end{array}$

• Global robust objective (Lemma 3.3): adds a penalty $- \epsilon \pi _ { i } ( h _ { i } ~ -$ $\mathbb { E } _ { \pi } [ h ] ) ^ { \mathsf { T } } ( \mathbb { E } _ { \pi } [ h ] / \| \mathbb { E } _ { \pi } [ h ] \| )$ that corrects all actions through a single direction. We sweep ϵ over 20 log-spaced values in $[ 1 0 ^ { - 2 } , 1 0 ]$ and report the best.

• SignCert-PO (Theorem 3.1): $\nabla _ { \phi _ { i } } J = \pi _ { i } ( \rho _ { i } A _ { i } - V _ { \rho } )$ , where $\rho _ { k } = 1 - \epsilon / \Delta _ { k }$ and ϵ is set to the $q _ { t } { \mathrm { - } } \mathrm { t h }$ quantile of $\{ \Delta _ { k } \}$

## F.1.3 Results

Figure 1(a) shows the proxy advantages per action. Red bars indicate completions whose advantage sign disagrees with the true RM. The hacking actions (7–8) have the largest proxy advantages. Figure 1(c) shows the per-action certified sign-preservation radius $\Delta _ { j } .$ The completions 7 and 8 cluster at low ∆ values because they are outliers in the feature-space (i.e., large $\| h _ { j } - \bar { h } \| )$ , confirming that $\Delta _ { j }$ identifies unreliable completions.

Figure 1(d) tracks true (solid) and proxy (dashed) expected reward during policy optimization. Dr.GRPO converges on the hacking actions, where true reward collapses. The global robust penalty partially mitigates reward hacking, but the single penalty direction cannot distinguish per-action reliability and is thus too conservative. SignCert-PO achieves the highest true reward, by suppressing the hacking actions’ gradient contributions via their small $\rho _ { k }$ weights.

## F.2 Algorithm pseudocode

Algorithm 1 implements the SignCert-PO re-weighting of GRPO derived in §3.2. We highlight the modifications over standard Dr.GRPO in blue

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 SignCert-PO Algorithm.

Require: Policy  $\pi_{\phi}$ , reward model  $r_{w}(h) = w^{\top} h + b$ , quantile threshold  $q_{t}$ , group size K

1: for each prompt x in batch do

2:    Sample K completions:  $y^{(1)}, \ldots, y^{(K)} \sim \pi_{\phi}(\cdot|x)$

3:    Compute  $h_{j} = h(x, y^{(j)}), r_{j} = w^{\top} h_{j} + b, A_{j} = r_{j} - \frac{1}{K} \sum_{k=1}^{K} r_{k}$  for  $j \in [K]$

4:    Compute  $\bar{h} = \frac{1}{K} \sum_{k=1}^{K} h_{k}, \Delta_{j} = |A_{j}| / \|h_{j} - \bar{h}\|_{2}$

5: Set  $\epsilon$  to the  $q_{t}$ -th quantile of  $\{1/\Delta_{j}\}$  over all completions in the batch

6: for each completion j do

7:    $\rho_{j}^{*} \leftarrow 1 - \epsilon/\Delta_{j}$ $\triangleright$  stop-gradient on  $A_{j}, \Delta_{j}$

8:    $\widetilde{A}_{j} \leftarrow \rho_{j}^{*} \cdot A_{j}$

9: Update policy with PG using  $\{\widetilde{A}_{j}\}$
</div>

## F.3 Training details

## F.3.1 Datasets

TL;DR summarization. We use the filtered version<sup>3</sup> of OpenAI’s Summarize from Feedback dataset (Stiennon et al., 2020) containing 116K samples. Each sample consists of a Reddit post and a human-written reference summary.

AlpacaFarm. We additionally evaluate on the AlpacaFarm benchmark (Dubois et al., 2023), a simulation framework for learning from human feedback built on 52K instructionfollowing samples.

## F.3.2 Models

Policy and proxy RM. We use Pythia (Biderman et al., 2023) and Qwen2.5 (Qwen et al., 2025) series as policy and proxy RM base models. We vary the policy size among Pythia 1B and 2.8B, and Qwen2.5 1.5B and 3B. The same model size as the policy is chosen for the proxy RM.

For TL;DR, the proxy RM is trained with 1 epoch of preference data. For AlpacaFarm, the proxy RM is trained with 3 epochs of preference data, roughly matching the number of queries for TL;DR (see §F.3.1).

Gold RM. For the TL;DR experiments with Pythia 1B and Qwen2.5 1.5B policies, the gold reward model is Skywork-Reward-Llama-3.1-8B-v0.2 (Liu et al., 2024), one of the top-performing discriminative reward models on RewardBench (Malik et al., 2025). All preference data and evaluation metrics for these settings are computed using this gold RM. For larger policy models (Pythia 2.8B and Qwen2.5 3B), we use GPT-4.1 Nano (OpenAI et al., 2024) as the gold RM, since a model with stronger capability than the proxy RM is required.

## F.3.3 Baselines

We compare SignCert-PO (Algorithm 1) against the following baselines:

• SFT: the SFT policy π<sub>SFT</sub> without any RL optimization.

• Dr.GRPO (Liu et al., 2025), as introduced in §2.1. The KL coefficient is swept over 0, 0.001, 0.005, 0.01, 0.1, 0.2 for Pythia-1B and Qwen2.5-1.5B, and the best configuration is selected per dataset. Due to computational constraints, we do not sweep the KL coefficient for larger models and instead use 0.005 for Pythia 2.8B and 0.01 for Qwen2.5 3B.

• UWO (Coste et al., 2024): an RM ensemble method that trains multiple proxy RMs with different random seeds and uses the mean reward minus a penalty proportional to the ensemble standard deviation. We use an ensemble of three RM members. Following the setup of Zhang et al. (2024), to control total model capacity, we use smaller individual RMs (Pythia 410M for 1B experiments and Qwen2.5 0.5B for 1.5B experiments). For larger policy models (Pythia 2.8B, Qwen2.5 3B), we use Pythia 1B and Qwen2.5 1.5B as proxy RMs in the ensemble, respectively.

• BSPO (Dai et al., 2025): a method that constrains policy updates to remain within the support of the behavior policy to prevent out-of-distribution exploitation.

• AdvPO (Zhang et al., 2024): an adversarial policy optimization method that augments the reward with a learned uncertainty penalty. We sweep the perturbation budget $B \in \{ 1 , 5 , 1 0 , 1 5 \}$ , following the range used in the original paper.

For both benchmarks, we set the learning rate to $1 0 ^ { - 6 }$ , with micro batch size 32 on 8 GPUs and $K = 8$ generations per prompt. Pythia policies are trained for 2,000 steps; Qwen2.5 for 500 steps since proxy reward converges faster. For SignCert-PO, we sweep the quantile threshold $\mathbf { \Phi } ^ { * } q _ { t } \ \in \ \{ 0 . 1 , 0 . \overset { . } { 2 } , 0 . 2 5 , 0 . 3 , 0 . 4 , 0 . \overset { \smile } { . } \}$ . For TL;DR experiments with Pythia 1B and Qwen2.5 1.5B, we report mean  standard deviation over 3 random seeds to account for variance from limited preference data.

## F.3.4 Evaluation metrics

We report the win rate of policy outputs against reference responses, judged by the gold RM. We also measure proxy RM accuracy and KL divergence during policy optimization, measured at every 50 global optimization steps. Proxy RM accuracy is calculated as agreement between the proxy RM and the gold RM on pairwise ranking comparisons of on-policy completions.

## F.4 Hyperparameters

Tables 4–6 list all hyperparameters used in our experiments.

<table><tr><td>Hyperparameter</td><td>Pythia 1B</td><td>Pythia 2.8B</td><td>Qwen2.5 1.5B</td><td>Qwen2.5 3B</td></tr><tr><td colspan="5">SFT</td></tr><tr><td>Epochs</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Learning rate</td><td> $3 \times 10^{-6}$ </td><td> $3 \times 10^{-6}$ </td><td> $3 \times 10^{-6}$ </td><td> $3 \times 10^{-6}$ </td></tr><tr><td>LR scheduler</td><td>cosine</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td>Per-device batch size</td><td>16</td><td>16</td><td>16</td><td>16</td></tr><tr><td>bf16</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td colspan="5">Preference data generation</td></tr><tr><td>Completions per prompt</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Sampling temperature</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Max new tokens</td><td>53</td><td>53</td><td>63</td><td>63</td></tr><tr><td>Gold RM</td><td>Skywork-8B</td><td>GPT-4.1 Nano</td><td>Skywork-8B</td><td>GPT-4.1 Nano</td></tr></table>

Table 4: SFT and preference data generation hyperparameters.

## F.5 Other experimental details

Certified radius validation (Figure 3). We test whether the certified sign-preservation radius $\Delta _ { j }$ (8), derived under a linear head perturbation assumption, predicts robustness under broader perturbation classes. We use a Pythia 1B SFT policy to generate K=8 completions (53 tokens each, temperature 0.7) for 500 prompts from the TL;DR validation split. The Pythia 1B proxy RM scores each completion and provides last-layer hidden states $h _ { j } \in \mathbb { R } ^ { 2 0 4 8 }$ . For the gold RM we use Skywork-Reward-Llama-3.1-8B-v0.2, the one used for RM training. Groups with reward standard deviation below $1 0 ^ { - 4 }$ are filtered. We consider three settings beyond the analytically tractable head-only case. Whole-model perturbation: for each parameter tensor $p ,$ we add $\delta _ { p } \sim \mathcal { N } ( 0 , ( \sigma \| p \| _ { F } / \sqrt { \mathrm { n u m e l } ( p ) } ) ^ { 2 } I )$ with $\sigma = 0 . 1$ and draw 10 samples. Input embedding perturbation: isotropic Gaussian noise $\delta \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { i n } } ^ { 2 } I )$ is added to the embedding layer output before the transformer backbone, with $\sigma _ { \mathrm { i n } } = 0 . 0 \ddot { 1 }$ and 10 samples. Gold RM agreement: we compare the advantage signs of the proxy and gold RMs without any perturbation. Completions are binned into 10 quantile groups (deciles) by $\Delta _ { j } .$ For each bin we report the mean sign-preservation rate or gold agreement rate with 95% confidence intervals.

<table><tr><td>Hyperparameter</td><td>Pythia 1B</td><td>Qwen2.5 1.5B</td></tr><tr><td>Preference data epochs</td><td colspan="2"> $\{1,3,5,10,20\}$ </td></tr><tr><td>Max length</td><td>512</td><td>512</td></tr><tr><td>Learning rate</td><td> $2 \times 10^{-5}$ </td><td> $2 \times 10^{-5}$ </td></tr><tr><td>LR scheduler</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.03</td><td>0.03</td></tr><tr><td>Weight decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Per-device batch size</td><td>16</td><td>16</td></tr><tr><td>Gradient accumulation steps</td><td>1</td><td>1</td></tr><tr><td>Gradient checkpointing</td><td>√</td><td>√</td></tr><tr><td>bf16</td><td>√</td><td>√</td></tr></table>

Table 5: Proxy reward model training hyperparameters. The same base model architecture as the SFT policy is used with a randomly initialized scalar head.

<table><tr><td>Hyperparameter</td><td>Pythia 1B</td><td>Pythia 2.8B</td><td>Qwen2.5 1.5B</td><td>Qwen2.5 3B</td></tr><tr><td>Training steps</td><td>2,000</td><td>2,000</td><td>500</td><td>500</td></tr><tr><td>Generations per prompt K</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Max new tokens</td><td>53</td><td>53</td><td>63</td><td>63</td></tr><tr><td>Temperature</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Per-device batch size</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Gradient accumulation steps</td><td>16</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Number of GPUs</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-6}$ </td></tr><tr><td>LR scheduler</td><td>constant</td><td>constant</td><td>constant</td><td>constant</td></tr><tr><td>Clip range ratio</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>KL coefficient β</td><td>0.01</td><td>0.005</td><td>0.1</td><td>0.01</td></tr><tr><td>SignCert-PO quantile qt</td><td></td><td colspan="3">{0.1, 0.2, 0.25, 0.3, 0.4, 0.5}</td></tr><tr><td>bf16</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 6: Policy optimization hyperparameters (Dr.GRPO / SignCert-PO). For Pythia 1B and Qwen2.5 1.5B, the KL coefficient is selected via sweep over 0, 0.001, 0.005, 0.01, 0.1, 0.2 (c.f., Appendix F.3.3). For the RM epoch ablation (Appendix G.3), the KL coefficient is re-tuned per epoch.

## G Additional results

## G.1 Additional comparison with baselines

Stronger KL penalty. Figure 5 sweeps the KL coefficient $\beta$ for Dr.GRPO on Pythia 1B, with the SignCert-PO trajectory. Increasing $\beta$ does constrain the policy to lower KL regions, but this trades off exploration for safety. SignCert-PO reaches the highest peak gold reward at a moderate KL budget. We argue that the KL penalty controls how far the policy moves from the reference distribution, uniformly penalizing all deviations. On the other hand, SignCert-PO controls which directions the policy moves, selectively suppressing updates where the RM is unreliable.

Figure 5: Increasing β constrains the policy to lower KL regions but trades off exploration for safety. KL coefficient sweep for Dr.GRPO on Pythia 1B and TL;DR with the SignCert-PO trajectory.

UWO with same-size RMs. In our experiments (Table 1), we followed Zhang et al. (2024) and use smaller individual RMs for UWO to control total capacity (Appendix F.3.3). As an ablation, we train an ensemble of three Pythia 1B RMs with different random seeds and run UWO under the same setting on TL;DR with the Pythia 1B policy. As shown in Table 7, increasing the individual RM size does not improve UWO.

<table><tr><td>Method</td><td>RM size</td><td>Win rate</td></tr><tr><td>UWO (×3)</td><td>410M×3</td><td>6.1 ± 3.6</td></tr><tr><td>UWO (×3)</td><td>1B×3</td><td>5.9</td></tr><tr><td>SignCert-PO</td><td>1B</td><td>60.0 ± 2.0</td></tr></table>

Table 7: UWO RM size ablation on TL;DR (Pythia 1B).

## G.2 Proxy RM accuracy during training

Figure 6 compares the proxy RM accuracy trajectory during policy optimization for Pythia 1B and 2.8B on TL;DR. Although their performance is comparable initially, the Pythia-1B proxy RM degrades substantially faster than the 2.8B proxy RM, confirming that smaller proxy RMs lose their accuracy more rapidly during policy optimization.

Figure 6: Smaller proxy RMs lose accuracy faster during policy optimization. Proxy RM accuracy over training steps for Pythia 1B and 2.8B on TL;DR. For Pythia 1B, mean over three seeds is shown with individual seed traces.

## G.3 Ablation on quantile threshold and the amount of preference data

Figure 7 shows the trade-off between KL divergence and RM accuracy across quantile hyperparameters for Pythia 1B on TL;DR. Higher quantile thresholds effectively constrain the policy to the regions where the proxy RM maintains higher accuracy. We note that SignCert-PO consistently achieves higher RM accuracy than Dr.GRPO at the same KL budget, suggesting its distinction from the KL penalty parameter.

Figure 7: The quantile parameter q<sub>t</sub> provides a tunable indicator of RM accuracy. Proxy RM accuracy vs. KL divergence for Dr.GRPO and SignCert-PO with varying q<sub>t</sub> on TL;DR (Pythia 1B). Thin lines show individual seeds and thick lines show the mean. Higher q<sub>t</sub> maintains RM accuracy.

Tables 8–9 report the average proxy RM accuracy over training for varying preference-data epochs and quantile thresholds. SignCert-PO consistently maintains higher proxy RM accuracy than Dr.GRPO across all settings, with the improvement most pronounced when the proxy RM is weak. We note that training a proxy RM with the preference data generated for 10–20 epochs can lead to overfitting, as we saw in §4.2. With approximately 123K preference pairs in the TL;DR dataset, 10 epochs corresponds to roughly 1.23M training samples, exceeding the 1–3 epochs typical in the literature (Ouyang et al., 2022; Stiennon et al., 2020). We include these settings as ablations.

<table><tr><td>RM Epochs</td><td>GRPO</td><td>SignCert-PO  $q_t=0.1$ </td><td> $q_t=0.2$ </td><td> $q_t=0.25$ </td><td> $q_t=0.3$ </td><td> $q_t=0.4$ </td><td> $q_t=0.5$ </td></tr><tr><td>1</td><td>52.89±0.22%</td><td>60.62±0.54%</td><td>60.25±0.41%</td><td>62.04±0.20%</td><td>62.09±0.92%</td><td>62.80±0.54%</td><td>64.63±0.15%</td></tr><tr><td>5</td><td>58.26%</td><td>56.97%</td><td>59.23%</td><td>60.82%</td><td>61.66%</td><td>64.77%</td><td>65.15%</td></tr><tr><td>10</td><td>57.32%</td><td>58.14%</td><td>60.47%</td><td>62.20%</td><td>61.08%</td><td>64.93%</td><td>65.32%</td></tr><tr><td>20</td><td>59.82%</td><td>61.28%</td><td>63.76%</td><td>62.53%</td><td>62.61%</td><td>69.18%</td><td>67.21%</td></tr></table>

Table 8: Pythia-1B: average proxy RM accuracy (%) over training. Higher is better.

<table><tr><td>RM Epochs</td><td>GRPO</td><td>SignCert-PO  $q_t=0.1$ </td><td> $q_t=0.2$ </td><td> $q_t=0.25$ </td><td> $q_t=0.3$ </td><td> $q_t=0.4$ </td><td> $q_t=0.5$ </td></tr><tr><td>1</td><td>65.82±0.88%</td><td>67.09±0.78%</td><td>68.78±0.80%</td><td>69.19±0.56%</td><td>69.32±0.33%</td><td>71.04±0.22%</td><td>71.72±0.70%</td></tr><tr><td>5</td><td>66.94%</td><td>67.05%</td><td>68.84%</td><td>68.55%</td><td>68.25%</td><td>70.54%</td><td>73.28%</td></tr><tr><td>10</td><td>68.22%</td><td>68.48%</td><td>69.99%</td><td>69.83%</td><td>71.86%</td><td>71.11%</td><td>75.94%</td></tr><tr><td>20</td><td>69.51%</td><td>67.95%</td><td>69.30%</td><td>71.22%</td><td>71.32%</td><td>71.96%</td><td>74.31%</td></tr></table>

Table 9: Qwen2.5-1.5B: average proxy RM accuracy (%) over training. Higher is better.

Tables 10–11 present the full quantile sweep on TL;DR. The optimal quantile threshold depends on the proxy RM quality and policy architecture. Across both models, higher thresholds monotonically improve proxy RM accuracy (Tables 8–9) but can reduce win rate when the threshold is too aggressive, reflecting the exploration–exploitation trade-off in robust optimization.

## G.4 Sample outputs

Tables 12–13 show example outputs by models trained with different methods. On TL;DR with Pythia-1B (1-epoch proxy RM), SignCert-PO produced the response with the highest gold reward (GR) in 326 out of 512 evaluation prompts. On AlpacaFarm with Qwen2.5-1.5B, SignCert-PO achieved the highest GR in 210 out of 512 prompts.

<table><tr><td>RM Epochs</td><td>GRPO</td><td>SignCert-PO qt=0.1</td><td>qt=0.2</td><td>qt=0.25</td><td>qt=0.3</td><td>qt=0.4</td><td>qt=0.5</td></tr><tr><td>1</td><td>20.96±4.38%</td><td>20.31±6.40%</td><td>30.08±7.16%</td><td>42.77±1.81%</td><td>49.81±8.98%</td><td>60.03±1.61%</td><td>53.39±3.86%</td></tr><tr><td>5</td><td>63.28%</td><td>50.98%</td><td>52.54%</td><td>67.58%</td><td>67.97%</td><td>66.41%</td><td>49.61%</td></tr><tr><td>10</td><td>55.27%</td><td>66.80%</td><td>73.44%</td><td>70.12%</td><td>71.29%</td><td>77.93%</td><td>67.38%</td></tr><tr><td>20</td><td>64.06%</td><td>53.32%</td><td>56.06%</td><td>48.24%</td><td>47.27%</td><td>42.97%</td><td>49.22%</td></tr></table>

Table 10: Pythia-1B: win rate (%) after 2,000 steps across quantile thresholds and RM epochs.

<table><tr><td>RM Epochs</td><td>GRPO</td><td>SignCert-PO qt=0.1</td><td>qt=0.2</td><td>qt=0.25</td><td>qt=0.3</td><td>qt=0.4</td><td>qt=0.5</td></tr><tr><td>1</td><td>66.86±3.13%</td><td>66.01±1.92%</td><td>65.11±1.27%</td><td>64.06±0.70%</td><td>63.74±1.07%</td><td>60.09±1.27%</td><td>53.58±1.30%</td></tr><tr><td>5</td><td>94.53%</td><td>94.92%</td><td>95.90%</td><td>95.51%</td><td>92.58%</td><td>94.34%</td><td>90.43%</td></tr><tr><td>10</td><td>94.34%</td><td>95.70%</td><td>94.92%</td><td>95.51%</td><td>94.53%</td><td>90.43%</td><td>91.21%</td></tr><tr><td>20</td><td>95.31%</td><td>96.09%</td><td>94.53%</td><td>96.09%</td><td>94.34%</td><td>96.88%</td><td>94.92%</td></tr></table>

Table 11: Qwen2.5-1.5B: win rate (%) after 500 steps across quantile thresholds and RM epochs.

<table><tr><td>Content</td><td>GR</td></tr><tr><td>SUBREDDIT: r/relationshipsTITLE: Me [17 M] with a girl [16 F] who I don't know how to let down.POST: There is a backstory behind this so let me give it to everyone straight. I'm 17 M never dated anyone before. I have made posts prior to this one based around my issues with being able to date. My biggest issue is that once a girl peeks my interest I pursue her until she develops the same feelings. Once I understand she is interested too I sort of back off. A friend of mine called me a “chaser” which is what I feel I am. Now I have this girl that I know, really sweet, really smart, talented, the whole thing. I found recently she has feelings for me and we talked a bit about it after I mentioned I was aware of them. When we talked I gave her the impression that I wanted to date her and that we should talk in person and hang out a big more to be dating. The only issue is my gut AGAIN like always is telling me I'm not interested even though I really want to be, I'm just not.</td><td>-</td></tr><tr><td>Reference: I don't know how to let this girl down after already giving her the impression I wanted to date her. I feel absolutely terrible about it.</td><td>-0.44</td></tr><tr><td>Dr.GRPO: talks/ possible dates/ feelings for girl/ situation changed but am I worried about telling/ situation affecting date/ how to let down girlfriend/ options available/ outcome. Advice please but worried about telling/ situation affecting date/ outcome.AdvPO: communication/dating situation, possibility of dating but am not sure if to let down friend/ possible date/ talk in person/ hang out/ possible outcome. Do I tell friend or risk hurting friendship? worried about outcome. worried about outcomeUWO: Don't know how to let her down. Help. Help. Help help help. help help help help. help help. not sure how to best approach help help help. not sure how to best approach help. not sure how to tell. not sure how toBSPO: I'm 17 M never dated before, have feelings for girl, want to date her, but not sure if I'm interested or not. not sure how to let her down. any advice or suggestions would be appreciated. any thoughts or comments would be appreciated. anySignCert-PO (Ours): dating/potential relationship, girl has feelings for me, I give impression I want to date but gut tells me I'm not interested. Am I overthinking this? Advice on letting down? Am worried about hurting relationship? Help!</td><td>-8.56-9.50-16.62-8.75-1.31</td></tr><tr><td>Generate a list of the seven most iconic locations in Colorado.</td><td>-</td></tr><tr><td>Reference: 1. Garden of the Gods - Colorado Springs2. Maroon Bells - Aspen3. Mesa Verde National Park - Montezuma4. Royal Gorge Bridge - Cañon City5. Red Rocks Amphitheater - Denver6. Pikes Peak - Colorado Springs7. Rocky Mountain National Park - Estes Park</td><td>7.81</td></tr><tr><td>Dr.GRPO: The seven most iconic locations in Colorado are: Rocky Mountain National Park, Denver, Colorado, Pikes Peak, Colorado, Colorado Springs, Colorado, and the Great Sand Dunes National Park. Explanation: These locations are iconic because they are known for their natural beauty, adventure, and history. They are also popular tourist destinations and have a lot of cultural and historical significance. They are also easily accessible and offer a variety of activities and experiences. Finally, they are all located in Colorado and are a great way to experience the state.AdvPO: The seven most iconic locations in Colorado are: Rocky Mountain National Park, Denver, Colorado Springs, Pikes Peak, Colorado Springs, Colorado Springs. These locations are all known for their natural beauty and are popular tourist destinations. They are also home to many famous landmarks and attractions.UWO: The seven most iconic locations in Colorado are: 1. Rocky Mountains 2. Grand Canyon 3. Mesa Verde 4. Garden of the Gods 5. Pikes Peak 6. Rocky Mountain National Park 7. Denver. These locations are iconic because they are beautiful, unique and have a rich history. They are also popular tourist destinations and have a significant impact on the state's economy. Therefore, they are considered to be the most iconic locations in Colorado.BSPO: The seven most iconic locations in Colorado are: Rocky Mountain National Park, Denver, Colorado, Denver International Airport, Denver, Colorado, Denver, Colorado, Denver, Colorado.SignCert-PO (Ours): 1. Rocky Mountains 2. Denver 3. Colorado Springs 4. Pikes Peak 5. Garden of the Gods 6. Mesa Verde 7. Telluride</td><td>-9.06-20.88-16.25-28.508.56</td></tr></table>

Table 12: Example output by a Pythia-1B model trained with different methods on the TL;DR task. SignCert-PO produces a coherent summary comparable to the reference. GR denotes gold reward.

Table 13: Example output by a Qwen2.5 1.5B model trained with different methods on the AlpacaFarm task. SignCert-PO produces a concise list comparable to the reference.
