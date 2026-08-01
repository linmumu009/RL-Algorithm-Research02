# On the Impossibility of Unbiased and Length-Invariant Policy Optimization with Outcome Rewards

Fei Ding<sup>1∗</sup>, Yongkang Zhang<sup>1</sup>, Runhao Liu<sup>1</sup>, Yuhao Liao<sup>2</sup>, Zijian Zeng<sup>2</sup>, Huiming Yang<sup>2</sup>

<sup>1</sup>Alibaba Group <sup>2</sup>Tsinghua University

## Abstract

Group Relative Policy Optimization (GRPO) is the dominant reinforcement learning algorithm for training reasoning capabilities in large language models, notably adopted by DeepSeek-R1. The recent improvement Dr. GRPO (COLM 2025) identifies the response-level length bias caused by pertrajectory length normalization in GRPO and proposes removing this normalization, claiming the resulting optimizer is “unbiased.” We show that this claim is incomplete. Specifically, we establish an impossibility theorem: under the standard outcome reward + GRPO setting, no length-based weighting scheme can simultaneously achieve the following two properties. (P1) Gradient unbiasedness: the gradient estimator is an unbiased estimate of the true policy gradient. (P2) Length invariance: each trajectory’s efective contribution to the gradient is independent of its token length. GRPO approximately satisfies P2 but violates P1; Dr. GRPO satisfies P1 but violates P2. We characterize the complete tradeof spectrum via the parametric family $f _ { \alpha } ( L ) = { \bf \hat { \Psi } } L ^ { \alpha - 1 }$ , where α = 0 recovers GRPO, α = 1 recovers Dr. GRPO, and provide quantitative analysis showing that Dr. GRPO’s length bias can cause longer trajectories to dominate gradient updates by a factor proportional to the length ratio. Our results reveal that neither algorithm is universally “done right”; they occupy opposite ends of a fundamental and unavoidable tradeof.

Figure 1: Complete unbiasedness is impossible.

## 1 Introduction

Reinforcement learning (RL) has become a core technique for improving reasoning capabilities of large language models (LLMs). DeepSeek-R1-Zero (DeepSeek-AI et al. 2026) demonstrated an important finding: without supervised finetuning, directly applying RL to a base LLM can elicit complex reasoning behaviors, including chain-of-thought and self-reflection. The core algorithm of this training paradigm is Group Relative Policy Optimization (GRPO) (Shao et al. 2024). GRPO is a critic-free RL algorithm that estimates advantages by comparing multiple responses sampled for the same prompt.

A salient empirical observation during GRPO training is the persistent growth of response length (DeepSeek-AI et al. 2026; Zeng et al. 2025; Hu et al. 2025). Liu et al. (2025) critically examined this phenomenon and identified two sources of optimization bias in GRPO. The first is response-level length bias caused by per-trajectory length normalization $\frac { 1 } { | \mathbf { o } _ { i } | }$ , and the second is question-level dificulty bias caused by standard deviation normalization. They proposed Dr. GRPO, removing both normalization terms, claiming to restore an “unbiased” optimization objective. Dr. GRPO has been widely adopted by the community and achieved stateof-the-art results on mathematical reasoning benchmarks at the time.

In this paper, we challenge the completeness of this claim. We confirm that Dr. GRPO’s gradient estimator is indeed an unbiased estimate of the policy gradient (as they rigorously proved in their Appendix A). However, we show that removing the length normalization term $\frac { 1 } { \left| \mathbf { o } _ { i } \right| }$ introduces another form of bias: length bias in the optimization dynamics. This bias causes longer trajectories to contribute disproportionately more to gradient updates. More fundamentally, we establish the following impossibility result:

Main Result (Informal). Under the outcome reward + GRPO setting, no length-based weighting scheme can simultaneously achieve gradient unbiasedness and length invariance. GRPO and Dr. GRPO represent the two extremes of this unavoidable tradeof.

Our contributions are as follows:

• We formalize two desirable properties of group-based RL optimizers, namely gradient unbiasedness (P1) and length invariance (P2), and prove they are mutually exclusive under outcome rewards (theorem 6).

• We characterize the tradeof spectrum via the parametric family $f _ { \alpha } ( L ) = L ^ { \alpha - 1 } \left( \alpha \in \bar { [ 0 , 1 ] } \right)$ , where $\alpha = 0$ corresponds to GRPO and $\alpha = 1$ corresponds to Dr. GRPO (theorem 8).

• We provide quantitative analysis showing that Dr. GRPO’s length bias can be severe: at length ratio $r ,$ the longer trajectory captures $\frac { r } { 1 + r }$ of the gradient signal (theorem 9).

## 2 Preliminaries

Token-level MDP. Language model generation is modeled as a token-level Markov Decision Process $\begin{array} { r l } { \mathcal { M } } & { { } = } \end{array}$ $( \boldsymbol { S } , \mathcal { A } , \boldsymbol { r } , p _ { \mathcal { Q } } )$ . At step t, the state $s _ { t } = [ \mathbf q , o _ { 1 } , \dots , o _ { t - 1 } ]$ is the concatenation of the prompt and previously generated tokens. The policy $\pi _ { \boldsymbol { \theta } } ( \cdot | s _ { t } )$ selects the next token $o _ { t }$ from the vocabulary A. Generation terminates upon producing an end-of-sequence token or exhausting the token budget. The objective is to maximize the expected return:

$$
J (\pi_ {\theta}) = \mathbb {E} _ {\mathbf {q} \sim p _ {\mathcal {Q}}} \left[ \mathbb {E} _ {\mathbf {o} \sim \pi_ {\theta} (\cdot | \mathbf {q})} \left[ R (\mathbf {q}, \mathbf {o}) \right] \right],\tag{1}
$$

where $\begin{array} { r } { R ( \mathbf { q } , \mathbf { o } ) \ = \ \sum _ { t = 1 } ^ { | \mathbf { o } | } r ( s _ { t } , o _ { t } ) } \end{array}$ is the trajectory return. Under the standard outcome reward setting for reasoning tasks (DeepSeek-AI et al. 2026), a scalar reward is assigned at the end of generation: $R ( \mathbf { q } , \mathbf { o } ) = 1$ if o contains the correct answer, and 0 otherwise.

Policy gradient. The Monte Carlo policy gradient (Williams 1992; Sutton and Barto 2018) of Eq. (1) is:

$$
\nabla_ {\theta} J (\pi_ {\theta}) = \mathbb {E} _ {\mathbf {q}, \mathbf {o} \sim \pi_ {\theta}} \left[ \sum_ {t = 1} ^ {| \mathbf {o} |} \nabla_ {\theta} \log \pi_ {\theta} (o _ {t} | \mathbf {q}, \mathbf {o} _ {<   t}) \cdot A (o _ {t} | \mathbf {q}, \mathbf {o} _ {<   t}) \right],\tag{2}
$$

where $A ( o _ { t } | \mathbf { q } , \mathbf { o } _ { < t } ) = R ( \mathbf { q } , \mathbf { o } ) - B ( \mathbf { q } , \mathbf { o } _ { < t } )$ is the advantage and B is any baseline independent of $o _ { t }$ (Sutton and Barto 2018). Under outcome rewards, the advantage is identical for all tokens in a trajectory since the return does not depend on $t .$

Group-relative baseline. Both GRPO and Dr. GRPO sample G responses $\{ \mathbf { o } _ { 1 } , \hdots , \mathbf { o } _ { G } \}$ for each prompt and use the group mean as the baseline: $B \ = \ \bar { \mathrm { m e a n } ( \mathbf { R } ) }$ , where ${ \bf R } = \{ R ( { \bf q } , { \bf o } _ { 1 } ) , \ldots , R ( { \bf q } , { \bf o } _ { G } ) \}$ . The advantage for all tokens in trajectory $\mathbf { o } _ { i }$ is:

$$
\tilde {A} _ {i} = R (\mathbf {q}, \mathbf {o} _ {i}) - \mathrm{mean} (\mathbf {R}).\tag{3}
$$

GRPO (Shao et al. 2024). GRPO maximizes the following surrogate objective (omitting the clipping mechanism as it does not afect our analysis):

$$
J _ {\mathrm{GRPO}} (\theta) = \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| \mathbf {o} _ {i} |} \sum_ {t = 1} ^ {| \mathbf {o} _ {i} |} \frac {\pi_ {\theta} (o _ {i , t} | \mathbf {q} , \mathbf {o} _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i , t} | \mathbf {q} , \mathbf {o} _ {i , <   t})} \cdot \frac {\tilde {A} _ {i}}{\mathrm{std} (\mathbf {R})}.\tag{4}
$$

Dr. GRPO (Liu et al. 2025). Dr. GRPO removes the pertrajectory length normalization $\frac { 1 } { | \mathbf { o } _ { i } | }$ and the standard deviation normalization std(R):

$$
J _ {\mathrm{Dr.GRPO}} (\theta) = \frac {1}{G} \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| \mathbf {o} _ {i} |} \frac {\pi_ {\theta} (o _ {i , t} | \mathbf {q} , \mathbf {o} _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i , t} | \mathbf {q} , \mathbf {o} _ {i , <   t})} \cdot \tilde {A} _ {i}.\tag{5}
$$

Liu et al. (2025) proved in their Appendix A that the gradient of Eq. (5) recovers the unbiased Monte Carlo policy gradient with a group-relative baseline. Furthermore, the advantage ${ \tilde { A } } _ { i }$ is equivalent to REINFORCE Leave-One-Out (RLOO) (Kool, van Hoof, and Welling 2019; Ahmadian et al. 2024) up to a constant factor.

Remark 1. Dr. GRPO’s advantage estimator is equivalent to RLOO (Ahmadian et al. 2024) up to a constant factor; see Liu et al. (2025) Appendix $\mathbf { A } .$ . Therefore our analysis also applies to RLOO, but we focus on Dr. GRPO since it explicitly claims to resolve the length bias issue.

Unified framework. To unify the analysis of both methods, we introduce a weighted gradient estimator parameterized by a weighting function $f : \mathrm { \bar { N } } \to \mathbb { R } _ { + }$

$$
\hat {g} _ {f} = \frac {1}{G} \sum_ {i = 1} ^ {G} f (| \mathbf {o} _ {i} |) \cdot \tilde {A} _ {i} \cdot \sum_ {t = 1} ^ {| \mathbf {o} _ {i} |} \nabla_ {\theta} \log \pi_ {\theta} (o _ {i, t} | \mathbf {q}, \mathbf {o} _ {i, <   t}).\tag{6}
$$

GRPO corresponds to $f ( L ) = 1 / L$ and Dr. GRPO corresponds to $f ( \bar { L ) } = 1$ (both omitting the std(R) factor since it is a question-level scalar orthogonal to length bias analysis).

## 3 Main Result: Impossibility Theorem

Notation and setup. Consider the length-weighted gradient estimator

$$
\hat {g} _ {f} = \frac {1}{G} \sum_ {i = 1} ^ {G} f (L _ {i}) \tilde {A} _ {i} S _ {i},\tag{7}
$$

where

$$
L _ {i} := | \mathbf {o} _ {i} |, \quad S _ {i} := \sum_ {t = 1} ^ {L _ {i}} \nabla_ {\theta} \log \pi_ {\theta} (o _ {i, t} \mid q, \mathbf {o} _ {i, <   t}),\tag{8}
$$

with the group mean baseline

$$
\tilde {A} _ {i} := R _ {i} - \frac {1}{G} \sum_ {j = 1} ^ {G} R _ {j}.\tag{9}
$$

In what follows, Π denotes the policy class under consideration. We assume all expectations below exist and that within-group trajectories are conditionally i.i.d. given the prompt and the current policy.

Assumption 2 (Fixed-length realizability). There exists a set of lengths $\mathcal { L } \subseteq \mathbb { N }$ such that for every $L \in { \mathcal { L } }$ , the policy class Π contains a policy $\pi ^ { ( L ) }$ under which, given the prompt, the trajectory length equals L almost surely, while the token content retains non-degenerate randomness.

Assumption 3 (Update scale functional). Fix an update scale functional

$$
\rho : \mathbb {R} ^ {d} \to \mathbb {R} _ {+},
$$

used to measure the magnitude of a single-trajectory score sum. We only require $\rho$ to be positively homogeneous of degree one for non-negative scalars, i.e., for all $\alpha \geq 0$ and all $\bar { \boldsymbol { v } } \in \mathbb { R } ^ { d }$

$$
\rho (\alpha v) = \alpha \rho (v).\tag{10}
$$

Typical examples include vector norms or the non-negative projection magnitude along a fixed direction.

Definition 4 (Trajectory-level correctness P1). The estimator $\hat { g } _ { f }$ satisfies trajectory-level correctness over the policy class Π if there exists a constant $c > 0$ , independent of the trajectory length distribution, such that for every policy $\pi \in \Pi$

$$
\mathbb {E} _ {\pi} [ \hat {g} _ {f} ] = c \nabla_ {\theta} J (\pi).\tag{11}
$$

Definition 5 (Length neutrality $\mathbf { P } 2 )$ . Let

$$
\Gamma_ {\pi , \rho} (L; a) := \mathbb {E} _ {\pi} \left[ \rho (S) \mid L (\tau) = L, \tilde {A} (\tau) = a \right],\tag{12}
$$

where a denotes a fixed efective training signal, i.e., a realized value of the group-relative advantage.

The estimator $\hat { \boldsymbol g } _ { f }$ satisfies length neutrality under the scale functional $\rho$ if for every policy $\pi \in \Pi$ , every realizable length $L ,$ and every fixed $^ { a , }$

$$
f (L) \Gamma_ {\pi , \rho} (L; a)\tag{13}
$$

is independent of $L .$

Theorem 6 (Structural conflict at the policy-class level). Under the outcome-level reward and group mean baseline setting, consider a weight function depending only on length,

$$
f: \mathbb {N} \to \mathbb {R} _ {+}.
$$

Suppose Assumptions 2 and 3 hold.

Ifthere exist a policy $\pi ^ { \star } \in \Pi$ , an efective training signal value $a ^ { \star }$ , and two distinct lengths $L _ { 1 } , L _ { 2 } \in \mathcal { L }$ such that

$$
\Gamma_ {\pi^ {\star}, \rho} (L _ {1}; a ^ {\star}) \neq \Gamma_ {\pi^ {\star}, \rho} (L _ {2}; a ^ {\star}),\tag{14}
$$

then no such f can simultaneously satisfy P1 (trajectorylevel correctness) and P2 (length neutrality) over the policy class Π.

Proof. We show that P1 and P2 impose mutually contradictory constraints on f.

Step 1: If P1 holds over the policy class Π, then $f ( L )$ must be a constant function.

Pick any $L _ { 0 } \in { \mathcal { L } }$ . By Assumption 2, there exists a policy $\pi ^ { ( L _ { 0 } ) } \in \Pi$ under which the trajectory length equals $L _ { 0 }$ almost surely given the prompt, while the token content remains random. Under this policy, for all i,

$$
L _ {i} = L _ {0},\tag{15}
$$

so the estimator can be written as

$$
\hat {g} _ {f} = f (L _ {0}) \cdot \frac {1}{G} \sum_ {i = 1} ^ {G} \tilde {A} _ {i} S _ {i}.\tag{16}
$$

Expanding the baseline,

$$
\tilde {A} _ {i} = R _ {i} - \frac {1}{G} \sum_ {j = 1} ^ {G} R _ {j} = \left(1 - \frac {1}{G}\right) R _ {i} - \frac {1}{G} \sum_ {j \neq i} R _ {j}.\tag{17}
$$

Therefore,

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} [ \tilde {A} _ {i} S _ {i} ] = \left(1 - \frac {1}{G}\right) \mathbb {E} _ {\pi^ {(L _ {0})}} [ R _ {i} S _ {i} ] - \frac {1}{G} \sum_ {j \neq i} \mathbb {E} _ {\pi^ {(L _ {0})}} [ R _ {j} S _ {i} ].\tag{18}
$$

For j $\neq i ,$ since within-group trajectories are conditionally i.i.d., $R _ { j }$ and $S _ { i }$ are independent; moreover, by the score function identity,

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} [ S _ {i} ] = 0.
$$

Hence,

(19)

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} [ R _ {j} S _ {i} ] = \mathbb {E} _ {\pi^ {(L _ {0})}} [ R _ {j} ] \mathbb {E} _ {\pi^ {(L _ {0})}} [ S _ {i} ] = 0.\tag{20}
$$

Thus,

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} [ \tilde {A} _ {i} S _ {i} ] = \left(1 - \frac {1}{G}\right) \mathbb {E} _ {\pi^ {(L _ {0})}} [ R _ {i} S _ {i} ].\tag{21}
$$

By the REINFORCE identity,

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} [ R _ {i} S _ {i} ] = \nabla_ {\theta} J (\pi^ {(L _ {0})}),\tag{22}
$$

yielding

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} [ \hat {g} _ {f} ] = f (L _ {0}) \frac {G - 1}{G} \nabla_ {\theta} J (\pi^ {(L _ {0})}).\tag{23}
$$

If P1 holds over the policy class Π, there exists a lengthindependent constant $c > 0$ such that

$$
\mathbb {E} _ {\pi^ {(L _ {0})}} \big [ \hat {g} _ {f} \big ] = c \nabla_ {\theta} J \big (\pi^ {(L _ {0})} \big).
$$

Therefore,

(24)

$$
f (L _ {0}) \frac {G - 1}{G} = c.\tag{25}
$$

Since $L _ { 0 }$ is arbitrary in ${ \mathcal { L } } , f ( L )$ must be the same for all $L \in { \mathcal { L } }$ . That is, there exists a constant $c _ { 0 } > 0$ such that

$$
f (L) \equiv c _ {0}, \quad \forall L \in \mathcal {L}.\tag{26}
$$

Step 2: If P2 holds, then under the theorem’s assumptions f cannot be a constant function.

By Definition 5, if the estimator $\hat { g } _ { f }$ satisfies length neutrality P2 under the scale functional $\rho ,$ then for every policy π ∈ Π and every efective training signal value a for which the conditional expectation is defined, there exists a constant $C _ { \pi , a }$ depending only on $( \pi , a )$ and not on the length $L ,$ such that for all realizable lengths $\dot { L } \in \mathcal { L } .$

$$
f (L) \Gamma_ {\pi , \rho} (L; a) = C _ {\pi , a}.\tag{27}
$$

Now fix the policy $\pi ^ { \star } \in \Pi$ , the efective training signal value $a ^ { \star }$ , and the two distinct lengths $L _ { 1 } , L _ { 2 } \in { \mathcal { L } }$ from the theorem’s assumptions, satisfying

$$
\Gamma_ {\pi^ {\star}, \rho} (L _ {1}; a ^ {\star}) \neq \Gamma_ {\pi^ {\star}, \rho} (L _ {2}; a ^ {\star}).\tag{28}
$$

Figure 2: The impossibility tradeof. The origin (zero bias on both axes) is unreachable. GRPO (α = 0) and Dr. GRPO (α = 1) occupy opposite ends of the Pareto frontier parameterized by $f _ { \alpha } ( L ) = L ^ { \alpha - 1 }$

We show that $f$ cannot be a constant function.

Suppose for contradiction that $f$ is constant, i.e., there exists a constant $c _ { 0 } > 0$ such that

$$
f (L) \equiv c _ {0}, \quad \forall L \in \mathcal {L}.\tag{29}
$$

Substituting (29) into (27) with $\pi = \pi ^ { \star }$ and $a = a ^ { \star }$ , we obtain for all realizable lengths $L \in { \mathcal { L } }$

$$
c _ {0} \Gamma_ {\pi^ {\star}, \rho} (L; a ^ {\star}) = C _ {\pi^ {\star}, a ^ {\star}}.\tag{30}
$$

In particular, for $L _ { 1 }$ and $L _ { 2 } .$ ,

$$
c _ {0} \Gamma_ {\pi^ {\star}, \rho} (L _ {1}; a ^ {\star}) = C _ {\pi^ {\star}, a ^ {\star}},\tag{31}
$$

and

$$
c _ {0} \Gamma_ {\pi^ {\star}, \rho} (L _ {2}; a ^ {\star}) = C _ {\pi^ {\star}, a ^ {\star}}.\tag{32}
$$

Since $c _ { 0 } > 0 ;$ these two equations imply

$$
\Gamma_ {\pi^ {\star}, \rho} (L _ {1}; a ^ {\star}) = \Gamma_ {\pi^ {\star}, \rho} (L _ {2}; a ^ {\star}),\tag{33}
$$

contradicting (28).

Therefore, under the theorem’s assumptions, any weight function f satisfying P2 cannot be a constant function.

## Step 3: Contradiction.

Step 1 shows: if P1 holds over the policy class Π, then f(L) must be a constant function. Step 2 shows: if P2 holds and there exists a policy for which $\Gamma _ { \pi , \rho } ( L ; a )$ varies non-trivially with length, then $\dot { f } ( L )$ cannot be a constant function.

These are contradictory. Therefore, under the theorem’s assumptions, no weight function f depending only on length can simultaneously satisfy P1 and P2 over the policy class Π. □

Illustrative example. Consider two trajectories for the same prompt with lengths $L _ { s } \ll L _ { \ell }$ , compared under the same efective training signal. If under some pre-specified scale functional $\rho ,$ the longer trajectory has a larger typical score-sum magnitude, i.e.,

$$
\Gamma_ {\pi , \rho} (L _ {\ell}; a) > \Gamma_ {\pi , \rho} (L _ {s}; a),
$$

then constant weights preserve this length-induced scale disparity, while any length compensation attempting to eliminate this disparity must deviate from constant weights. This example serves only to illustrate the structural conflict in the theorem and does not form part of the proof.

Scope of the theorem. Theorem 6 does not claim that a specific functional form (e.g., 1/L) is necessarily optimal; it merely states: when the typical score-sum magnitude under fixed efective training signal varies non-trivially with length, no unified weight function depending only on length can simultaneously satisfy P1 and P2.

Furthermore, the theorem only excludes weight functions that depend solely on length; more general estimator designs, such as weighting schemes that depend on token position, context, score geometry, or finer-grained credit assignment, are not within the scope of this exclusion.

Remark 7 (Essence of the conflict). P1 requires that a uniform length weight does not alter the original trajectory-level policy gradient objective; P2 requires that this weight compensates for the non-trivial variation of score-sum magnitude with length. When P1 constrains f(L) to be a constant function while P2 demands it to vary with length, the two become structurally irreconcilable.

## 3.1 Examples

For ease of understanding, see the supplementary material’s “Intuitive Examples of Asymmetric Length Behavior” and “Extreme Example” sections. They show that GRPO’s length bias manifests as correct responses tending to be shorter and incorrect responses tending to be longer, while Dr. GRPO’s length bias manifests as both correct and incorrect responses tending to be longer.

## 4 Corollaries and Analysis

## 4.1 Tradeof Spectrum

Corollary 8 (Parametric Tradeof Family). Consider the parametric family $f _ { \alpha } ( L ) = L ^ { \alpha - 1 } , \alpha \in [ 0 , 1 ] .$

• α = 0: f<sub>0</sub>(L) = 1/L — GRPO. Approximately satisfies P2 (length invariant) but violates P1 (biased gradient).

$\alpha = 1 \colon f _ { 1 } ( L ) = 1 - D r$ . GRPO. Satisfies P1 (unbiased gradient) but violates P2 (length biased).

$\alpha \in ( 0 , 1 )$ : intermediate tradeof. Partially biased gradient, partially length-dependent.

Table 1: Gradient weight shares of Dr. GRPO vs. GRPO at diferent length ratios $( G = 2 ,$ , binary reward). Under GRPO, both trajectories always receive equal weight.

<table><tr><td rowspan="2">Length ratio  $r$ </td><td colspan="2">Dr. GRPO</td><td colspan="2">GRPO</td></tr><tr><td> $w_{long}$ </td><td> $w_{short}$ </td><td> $w_{long}$ </td><td> $w_{short}$ </td></tr><tr><td>1:1</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>2:1</td><td>66.7%</td><td>33.3%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>5:1</td><td>83.3%</td><td>16.7%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>10:1</td><td>90.9%</td><td>9.1%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>50:1</td><td>98.0%</td><td>2.0%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>100:1</td><td>99.0%</td><td>1.0%</td><td>50.0%</td><td>50.0%</td></tr></table>

Gradient estimation bias isproportional to $| \alpha - 1 |$ and length bias is proportional to $\alpha ,$ establishing an inverse relationship.

fig. 2 visualizes this tradeof.

## 4.2 Quantifying Dr. GRPO’s Length Bias

Corollary 9 (Dr. GRPO’s length bias). Under Dr. GRPO $( f ( L ) = \mathrm { 1 } )$ with $G = 2$ and binary outcome reward, let $\mathbf { o } _ { 1 }$ and $\mathbf { o } _ { 2 }$ be two trajectories with lengths $L _ { 1 }$ and $L _ { 2 } .$ Their advantages satisfy $| \tilde { A } _ { 1 } | = | \tilde { A } _ { 2 } | = 0 . 5$ . The efective gradient weight oftrajectory $\mathbf { o } _ { i }$ is:

$$
w _ {i} = \frac {L _ {i}}{L _ {1} + L _ {2}}.\tag{34}
$$

For length ratio $r = L _ { \mathrm { m a x } } / L _ { \mathrm { m i n } } ,$ , the longer trajectory captures:

$$
w _ {\mathrm{long}} = \frac {r}{1 + r}\tag{35}
$$

of the total gradient magnitude, approaching 100% as $r  \infty .$ . Under GRPO $( f ( L ) = 1 / L ) , w _ { 1 } = w _ { 2 } = 0 . 5 ,$ independent of length.

Proof. With $G = 2$ and binary reward, exactly one trajectory is correct $( R = 1 )$ and one incorrect $( R = 0 )$ , giving $\mathrm { m e a n } ( \mathbf { R } ) = 0 . 5$ and $| \tilde { A } _ { 1 } | = | \tilde { A } _ { 2 } | = 0 . 5$ . Under Dr. GRPO, the gradient contribution magnitude of $\mathbf { o } _ { i }$ is proportional to $f ( | \mathbf { o } _ { i } | ) \cdot | \tilde { A } _ { i } | \cdot | \mathbf { o } _ { i } | = 1 \cdot 0 . 5 \cdot L _ { i }$ . The share is $w _ { i } = L _ { i } / ( \bar { L _ { 1 } } + \bar { L _ { 2 } } )$ . Under GRPO, the contribution is $( 1 / L _ { i } ) \cdot 0 . 5 \cdot L _ { i } = 0 . 5$ , independent of length. □

Example 10 (Extreme case). Let $G = 2 , \mathbf { o } _ { 1 }$ correct $( R = 1$ length 10 tokens), $\mathbf { o } _ { 2 }$ incorrect $( R = 0 ,$ , length 10,000 tokens). The advantages are $\tilde { A } _ { 1 } = + 0 . 5 , \tilde { A } _ { 2 } = - 0 . 5$ . Under Dr. GRPO, $\mathbf { o } _ { 2 } \ ' \mathbf { s }$ gradient contribution is $1 0 , 0 0 0 \times 0 . 5 =$ 5,000, while $\mathbf { o } _ { 1 } \mathbf { \ ' } _ { \mathbf { S } }$ is only $1 0 \times 0 . 5 = 5$ . The longer trajectory captures $\frac { 5 0 0 0 } { 5 0 0 5 } = 9 \dot { 9 } . 9 \%$ of the gradient signal, nearly completely drowning out the reinforcement of the correct answer. Under GRPO, both contribute 50%. A step-by-step derivation of this example (including gradient decomposition and its efect on parameter updates) is provided in the supplementary material’s “Extreme Example” section.

table 1 shows the severity of this efect at various length ratios.

Example 11 (Practical relevance). Liu et al. (2025) reported in their Table 5 that DeepSeek-R1-Zero produces correct answers averaging 4,965 tokens and incorrect answers averaging 8,206 tokens (a ratio of approximately 1:1.65). Under Dr. GRPO with $G \ : = \ : 2$ , the incorrect (longer) trajectory would capture approximately $\frac { 8 2 0 6 } { 4 9 6 5 + 8 2 0 6 } \approx 6 2 . 3 \%$ of the gradient, deviating 24.6 percentage points from the balanced 50%. While this proportion may appear moderate for a single update, the bias accumulates over hundreds of training iterations, systematically favoring longer responses.

## 4.3 Quantifying GRPO’s Gradient Bias

For completeness, we also characterize the gradient bias introduced by GRPO.

Corollary 12 (GRPO’s gradient bias). Under GRPO $( f ( L ) = 1 / L )$ , the gradient estimator satisfies:

$$
\mathbb {E} [ \hat {g} _ {1 / L} ] - \nabla_ {\theta} J = \mathbb {E} \left[ \frac {1}{G} \sum_ {i} \tilde {A} _ {i} \left(\frac {1}{| \mathbf {o} _ {i} |} - 1\right) \nabla_ {\theta} \log \pi_ {\theta} (\mathbf {o} _ {i} | \mathbf {q}) \right].\tag{36}
$$

This bias is non-zero when the trajectory length $\left| \mathbf { o } _ { i } \right|$ is correlated with the score function $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( \mathbf { o } _ { i } | \mathbf { q } )$ . This is generally always the case since the policy determines when the EOS token is generated.

$\begin{array} { r l r } { P r o o f . ~ \mathrm { B y } \qquad \mathrm { d i r e c t } \qquad \mathrm { c o m p u t a t i o n } ; \qquad \mathbb { E } \bigl [ \hat { g } _ { 1 / L } \bigr ] } & { { } } & { = } \\ { \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i } \tilde { A } _ { i } \frac { 1 } { | \mathbf { o } _ { i } | } \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { o } _ { i } | \mathbf { q } ) \right] \qquad \mathrm { a n d } \qquad \nabla _ { \theta } J } & { { } } & { = } \end{array}$ $\begin{array} { r } { \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i } \tilde { A } _ { i } \nabla _ { \theta } \log \pi _ { \theta } \big ( \mathbf { o } _ { i } \big | \mathbf { q } \big ) \right] } \end{array}$ The diference follows directly by linearity. Since $\left| { \bf { \bar { o } } } _ { i } \right|$ is determined by when π generates the EOS token, $\left| \mathbf { o } _ { i } \right|$ and $\nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { o } _ { i } | \mathbf { q } \big )$ are dependent, making the bias generally non-zero. □

## 4.4 Extension to General Group Size

Corollary 13 (General G + binary reward). For group size G with binary reward, if K out of G responses are correct, the advantages are $\tilde { A } _ { \mathrm { c o r r e c t } } = 1 - K / G$ and $\tilde { A } _ { \mathrm { { i n c o r r e c t } } } =$ $- K / G .$ Under Dr. GRPO, the efective weight of trajectory $\mathbf { o } _ { i }$ is still proportional to $| \mathbf { o } _ { i } | \cdot | \tilde { A } _ { i } |$ . The length bias exists for all G: longer trajectories always contribute more to the gradient, regardless of their correctness:

$$
w _ {i} = \frac {| \mathbf {o} _ {i} | \cdot | \tilde {A} _ {i} |}{\sum_ {j = 1} ^ {G} | \mathbf {o} _ {j} | \cdot | \tilde {A} _ {j} |}.\tag{37}
$$

## 5 Discussion

“Done Right” is a misnomer. Dr. GRPO (Liu et al. 2025), titled “Understanding R1-Zero-Like Training: A Critical Perspective,” positions its contribution as fixing GRPO’s optimization biases. The phrase “GRPO Done Right” implies a single correct formulation. Our impossibility theorem (theorem 6) shows this is not the case: GRPO and Dr. GRPO navigate diferent points on the inherent tradeof between gradient unbiasedness and length invariance. Calling one of them “done right” obscures the fact that both make legitimate but diferent tradeof choices.

DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.; Wang, P.; Zhu, Q.; Xu, R.; Zhang, R.; Ma, S.; Bi, X.; Zhang, X.; Yu,

X.; Wu, Y.; Wu, Z. F.; Gou, Z.; Shao, Z.; Li, Z.; Gao, Z.; Liu,

Qiu, J.; Li, J.; Cai, J. L.; Ni, J.; Liang, J.; Chen, J.; Dong,

Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang, M.; Li, M.;

Z.; Yan, Z.; Wu, Z.; Gu, Z.; Zhu, Z.; Liu, Z.; Li, Z.; Xie, Z.;

Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen, J.; Yuan, J.;

A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.; Zhao, C.;

Z. 2026. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

When does the tradeof matter? The practical importance of the tradeof depends on the variance of response lengths. When all responses to a given prompt have similar lengths (e.g., simple arithmetic), the diference between α = 0 and α = 1 is negligible. When response lengths vary substantially, the choice of α materially afects training dynamics. This situation is typical in reasoning tasks: correct solutions may be concise while incorrect attempts tend to be verbose (DeepSeek-AI et al. 2026).

Practical guidance. While we do not propose a specific algorithm, our analysis suggests: (i) When response length variance is high, a smaller α (closer to GRPO) may be preferable to prevent longer trajectories from dominating the gradient. (ii) When gradient bias is the primary concern (e.g., early in training when the policy changes rapidly), a larger α (closer to Dr. GRPO) provides more accurate gradient estimates. (iii) The optimal α may vary across training phases, suggesting that a curriculum approach could be beneficial.

Implications for training dynamics. A practical implication of theorem 9 deserves attention: when long correct responses receive L times more reinforcement signal than short correct responses, the policy may gradually shift toward generating longer outputs. The complete causal chain from gradient dominance to behavioral change also involves clipping, learning rate, and multi-step optimization, which lie beyond the scope of our single-step analysis. However, the systematic asymmetry in gradient signals provides a necessary condition for this trend. Conversely, under GRPO (α = 0), a 10-token short correct response and a 10,000- token long correct response receive the same total reinforcement signal. This provides no incentive at the gradient level to favor longer or shorter outputs.

Relationship to other biases. Our analysis complements Yang et al. (2026). The latter studies a diferent bias in GRPO: dificulty bias. This bias refers to the group-relative advantage estimator systematically underestimating advantages for difficult prompts and overestimating them for easy prompts. The length bias we identify is orthogonal, arising from withingroup length variation rather than between-group dificulty variation. The standard deviation normalization in GRPO contributes to dificulty bias (Liu et al. 2025); our impossibility result is independent of whether std(R) normalization is used.

Limitations. Our impossibility result is specific to the outcome reward setting, where each trajectory is assigned a scalar reward broadcast to all tokens. Under process reward (Schulman et al. 2018), diferent tokens receive diferent advantage estimates and the problem structure changes. The advantage is no longer constant across tokens, and the $\textstyle \sum _ { t }$ aggregation is no longer simply |o<sub>i</sub>| · A<sup>˜</sup><sub>i</sub>. Extending the impossibility analysis to process rewards is an interesting future direction. Furthermore, our analysis focuses on singlestep gradient estimators. The interaction between length bias and multi-step optimization dynamics (e.g., through PPOstyle clipping) warrants further investigation.

## 6 Conclusion

We have established a fundamental impossibility result for group-based policy optimization under outcome rewards: gradient unbiasedness and length invariance cannot coexist. This reveals that GRPO and Dr. GRPO are not in a “biased” vs. “correct” relationship, but instead represent two principled tradeof choices on the Pareto frontier. We hope this clarification helps the community make more informed algorithmic decisions, recognizing that the appropriate operating point depends on the specific characteristics of the training setting, especially the distribution of response lengths.

## References

Ahmadian, A.; Cremer, C.; Gallé, M.; Fadaee, M.; Kreutzer, J.; Pietquin, O.; Üstün, A.; and Hooker, S. 2024. Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12248–12267. Bangkok, Thailand: Association for Computational Linguistics.

H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin, H.; Gao, H.;

K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.;

Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen, Q.; Du, Q.;

Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen, R. J.; Jin, R. L.;

Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye, S.; Wang, S.; Yu,

S.; Zhou, S.; Pan, S.; Li, S. S.; Zhou, S.; Wu, S.; Ye, S.; Yun,

T.; Pei, T.; Sun, T.; Wang, T.; Zeng, W.; Zhao, W.; Liu, W.;

Liang, W.; Gao, W.; Yu, W.; Zhang, W.; Xiao, W. L.; An, W.;

Liu, X.; Wang, X.; Chen, X.; Nie, X.; Cheng, X.; Liu, X.;

Xie, X.; Liu, X.; Yang, X.; Li, X.; Su, X.; Lin, X.; Li, X. Q.;

Jin, X.; Shen, X.; Chen, X.; Sun, X.; Wang, X.; Song, X.;

Zhou, X.; Wang, X.; Shan, X.; Li, Y. K.; Wang, Y. Q.; Wei,

Y. X.; Zhang, Y.; Xu, Y.; Li, Y.; Zhao, Y.; Sun, Y.; Wang,

Y.; Yu, Y.; Zhang, Y.; Shi, Y.; Xiong, Y.; He, Y.; Piao, Y.;

Wang, Y.; Tan, Y.; Ma, Y.; Liu, Y.; Guo, Y.; Ou, Y.; Wang, Y.;

Gong, Y.; Zou, Y.; He, Y.; Xiong, Y.; Luo, Y.; You, Y.; Liu,

Y.; Zhou, Y.; Zhu, Y. X.; Xu, Y.; Huang, Y.; Li, Y.; Zheng, Y.;

Zhu, Y.; Ma, Y.; Tang, Y.; Zha, Y.; Yan, Y.; Ren, Z. Z.; Ren,

Z.; Sha, Z.; Fu, Z.; Xu, Z.; Xie, Z.; Zhang, Z.; Hao, Z.; Ma,

Song, Z.; Pan, Z.; Huang, Z.; Xu, Z.; Zhang, Z.; and Zhang,

Hu, J.; Zhang, Y.; Han, Q.; Jiang, D.; Zhang, X.; and Shum, H.-Y. 2025. Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model. arXiv:2503.24290.

Kool, W.; van Hoof, H.; and Welling, M. 2019. Buy 4 RE-INFORCE Samples, Get a Baseline for Free!

Liu, Z.; Chen, C.; Li, W.; Qi, P.; Pang, T.; Du, C.; Lee, W. S.; and Lin, M. 2025. Understanding R1-Zero-Like Training: A Critical Perspective. In Second Conference on Language Modeling.

Schulman, J.; Moritz, P.; Levine, S.; Jordan, M.; and Abbeel, P. 2018. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Sutton, R. S.; and Barto, A. G. 2018. Reinforcement Learning: An Introduction. The MIT Press, 2 edition.

Williams, R. J. 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4): 229–256.

Yang, F.; Chen, Z.; Wang, X.; Lu, X.; Chai, J.; Yin, G.; Lin, W.; Ma, S.; Zhuang, F.; Wang, D.; Yang, Y.; Li, J.; and Ban, Y. 2026. Your Group-Relative Advantage Is Biased. arXiv:2601.08521.

Zeng, W.; Huang, Y.; Liu, W.; He, K.; Liu, Q.; Ma, Z.; and He, J. 2025. 7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Efective and Eficient. https://hkust-nlp.notion.site/simplerl-reason. Notion Blog.
