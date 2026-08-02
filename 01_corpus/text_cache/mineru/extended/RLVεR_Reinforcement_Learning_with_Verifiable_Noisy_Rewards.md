# RLV<sup>ε</sup>R: Reinforcement Learning with Verifiable Noisy Rewards

Ali Rad<sup>1∗</sup> Khashayar Filom<sup>1</sup> Darioush Keivan<sup>1</sup> Peyman Mohajerin Esfahani<sup>2</sup> Ehsan Kamalinejad<sup>1</sup>

<sup>1</sup> Cognichip AI <sup>2</sup>University of Toronto  https://github.com/cognichip/Noisy-RL

January 9, 2026

## Abstract

Reinforcement learning with verifiable rewards (RLVR) is a simple but powerful paradigm for training LLMs: sample a completion, verify it, and update. In practice, however, the verifier is almost never clean–unit tests probe only limited corner cases; human and synthetic labels are imperfect; and LLM judges $( \mathrm { { { e . g . } } }$ , RLAIF) are noisy and can be exploited–and this problem worsens on harder domains (especially coding) where tests are sparse and increasingly model-generated. We ask a pragmatic question: does the verification noise merely slow down the learning (rate), or can it flip the outcome (fate)?

To address this, we develop an analytically tractable multi-armed bandit view of RLVR dynamics, instantiated with GRPO and validated in controlled experiments. Modeling false positives and false negatives and grouping completions into recurring reasoning modes yields a replicator-style (natural-selection) flow on the probability simplex. The dynamics decouples into within-correct-mode competition and a one-dimensional evolution for the mass on incorrect modes, whose drift is determined solely by Youden’s index $J = \mathrm { T P R } - \mathrm { F P R }$ . This yields a sharp phase transition: when $J > 0 ,$ the incorrect mass is driven toward extinction (learning); when $J = 0 ,$ , the process is neutral; and when $J < 0 ,$ incorrect modes amplify until they dominate (anti-learning and collapse). In the learning regime $J > 0 ,$ noise primarily rescales convergence time (“rate, not $\mathrm { f a t e ^ { \prime \prime } } )$ . Experiments on verifiable programming tasks under synthetic noise reproduce the predicted $J = 0$ boundary. Beyond noise, the framework offers a general lens for analyzing RLVR stability, convergence, and algorithmic interventions.


Figure 1: Phase transition under a sloppy oracle. We vary a single noise knob $J ,$ defined as $J = 1 - \delta _ { \mathrm { F P } } -$ $\delta _ { \mathrm { F N } } = \mathrm { T P R } - \mathrm { F P R } ,$ , which summarizes the net effect of false-positive and false-negative reward corruption induced by a noisy oracle. As J crosses zero, we observe a sharp transition from Phase I (learning; $\dot { J } > 0 ,$ accuracy improves with training and the bad-solution mass $p$ for a prompt converges to 0) to Phase II (unlearning; $\dot { J } < 0 .$ , accuracy systematically degrades and p converges to 1), with $\overset \vartriangle { \boldsymbol { J } } = \boldsymbol { 0 }$ as the critical boundary. Curves report validation performance during GRPO training on a Python code-generation task (two epochs) starting from Qwen-2.5-3B; solid lines denote the run-averaged E[pass@1] and shaded bands indicate 95% confidence intervals across runs. Although our mean-field ODE characterizes the trainingtime dynamics of $p ,$ the same sign-of-J transition is mirrored on held-out validation prompts in this setup, suggesting the drift is not merely a training-set artifact.

## 1 Introduction

Reinforcement Learning and LLMs. Recent breakthroughs in the reasoning capabilities of large language models (LLMs) through Reinforcement Learning (RL)–particularly Reinforcement Learning with Verifiable Rewards (RLVR) and group-normalized algorithms such as Group Relative Policy Optimization (GRPO) (Wen et al. (2025); Su et al. (2025))–have greatly expanded the frontier of model intelligence (Shao et al. (2024)). These methods provide further hope for the idea that true creativity and intelligence can emerge through self-play, interaction with an environment, and reward-based feedback.

Group-Normalized RL and RLVR. Group-normalized approaches in RL, like GRPO, eliminate the need for an explicit reward model (critic or PRM, Schulman et al. (2017); Lightman et al. (2023)) in verifiable domains such as mathematics and code generation, and demonstrate that even a few rollouts per prompt are often sufficient to approximate the advantage of each generated sequence (Wen et al. (2025); Su et al. (2025)). However, the cornerstone of these algorithms remains the sequence-level reward. This naturally leads to several key questions:

How sensitive is RL training to the quality of grand truth labels and rewards? Is the performance robust? Or does performance converge to afraction of the noise-free label as simply as ”you get what you pay for /garbage in, garbage out”?

Approaches such as LLM-as-Judge and RLAIF (Bai et al. (2022); Lee et al. (2023)) have shown some promise to replacing human preference labeling with AI feedback (synthetic preferences). Alternative methods attempt to remove the need for ground-truth supervision altogether, relying instead on label-free or self-rewarding mechanisms. However, these too are vulnerable to the same sources of noise, such as false positives and false negatives. For example, methods based on majority voting (Zuo et al. (2025)), using the consistency of reasoning traces (Zhang et al. (2025a)) or using it’s own model internal feedback (RLIF), like self-certainty– which leverage the log-probability confidence of generated sequences–aim to approximate reliable reward signals without external supervision (e.g., Zhao et al. (2025); Fu et al. (2025)).

Figure 2: The Winner Takes It All. The GRPO mean-field dynamics exhibit a striking structural property: in the absence of any symmetry among initial good arm masses, one arm ultimately dominates. When the oracle is in the learning regime $\bar { ( J } = \mathrm { T P R } - \mathrm { F P R } > 0 )$ , the total bad-arm mass p(t) converges to 0, and among the good arms the algorithm converges to the one with the highest initial probability (see Appendix I for a rigorous theoretical analysis of this phenomenon.). At the critical boundary $( J = 0 )$ , the probability vector remains fixed throughout training, yielding a continuum of equilibrium points. In the unlearning regime $( J < 0 ) .$ , the flow reverses and the bad arm becomes the almost-sure winner. This “winner-take-all” behavior is an inherent feature of the GRPO-style replicator dynamics.

## MOTIVATION

Learning without ground truth? Reinforcement learning is fundamentally driven by reward and environment feedback, and is therefore highly sensitive to their quality. Can an agent still learn and self-improve when the feedback is noisy and no direct ground-truth signal is available? To make this question precise, We begin by analyzing RL dynamics under noisy feedback.

Noisy Reward. Both major sources of supervision–human annotations and synthetically generated examples–are inherently prone to label noise, which directly induces reward noise in reinforcement learning settings. In addition, real-world automatic checkers are often imperfect: unit tests may be incomplete, edge cases may be missed, and multiple correct solutions may go unrecognized, etc. This problem becomes even more acute when LLM models produce the labels or rewards (Grattafiori et al. (2024)), since AI-generated supervision can introduce systematic biases rather than purely random noise.

Noisy Reward for Coding Tasks. These challenges are particularly severe in RLVR for coding, where verification relies on incomplete test suites and admits many semantically correct implementations, unlike short-answer math problems (e.g., AIME) or multiple-choice benchmarks with a fixed answer key (e.g., MMLU; Hendrycks et al. (2021)). Although current generative models perform impressively on many programming tasks, as we push toward increasingly difficult problems we should expect test coverage and fidelity to deteriorate. In the extreme, pass/fail outcomes can become essentially uncorrelated with true functional correctness, so that unit-test based verification approaches (and may even fall below) chance-level reliability. For this reason, we center our experiments on Python programming tasks, where imperfect verification is the norm and the resulting noise regime is both realistic and practically important.

In practice, these issues manifest primarily as False Positives (FP)–incorrect solutions that receive positive reward-and False Negatives (FN)–correct solutions that receive zero or negative reward. Together, these developments highlight an increasing dependence on potentially noisy supervision signals in RL training pipelines. This naturally raises a central question:

Is there a noise threshold beyond which learning becomes unreliable and may even collapse, especially under AI-generated supervision?

To address this question, we start by considering the per-prompt sequence-level graders, and the reward is a noisy binary signal $r \in \{ 0 , 1 \}$ . We model reward/label noise by two rates of flipping the true label with false–negative and false–positive rates

$$
\delta_ {\mathrm{FN}} = \operatorname * {P r} (r = 0 \mid \text { good }), \quad \delta_ {\mathrm{FP}} = \operatorname * {P r} (r = 1 \mid \text { bad }).\tag{1}
$$

Notice that these error rates in general can be time dependent, i.e $\delta _ { \mathrm { F P , F N } } = \delta _ { \mathrm { F P , F N } } ( t )$ . Based on these two levels of noise, we can define a scalar

$$
J := \mathbf {1} - \delta_ {\mathrm{FN}} - \delta_ {\mathrm{FP}} = \mathrm{TPR} - \mathrm{FPR} ^ {1} \in [ - 1, 1 ].\tag{2}
$$

which summarizes the net discriminative power of the checker. In statistical decision theory this is known as Youden’s index Youden (1950). Concretely,

• J = 1: Perfect rewarder (TPR = 1, FPR = 0);

• J = 0: Chance-level (uninformative) rewarder (TPR = FPR);

• J < 0: Inverted or anti-informative rewarder.

Geometrically, J equals the vertical distance between the ROC curve of the checker and the diagonal “random” line; thus it measures how far the verifier’s decisions deviate from random guessing.

Recently, controlled experiments such as Chen et al. (2025) have shown that imbalances between false-positive and false-negative rewards can induce severe mode collapse during RL training. Related phenomena have also been documented in mechanisms based on self-certainty, majority voting, or entropy regularization (Zhang et al. (2025b)), where systematic reward skew leads the model to collapse onto a narrow subset of outputs. Methods that rely on LLM-as-Judge further suffer from reward hacking, revealing structural vulnerabilities in model-generated supervision. Complementary theoretical analyses of noisy verifiers in RLVR have begun to emerge (Cai et al. (2025)), but despite this progress, a fundamental question remains:

## THIS PAPER ASKS

• How much reward sloppiness can RLVR tolerate? Concretely, under what levels of label/reward noise does GRPO/REINFORCE continue to improve accuracy, and when do characteristic failure modes (e.g., collapse or reward inversion) begin to emerge?

• What are the learning dynamics of RLVR under noisy rewards? How do the magnitude and structure of noise affect asymptotic accuracy? In the infinite-training limit, can learning under noisy rewards reach the same performance as with noise-free signals?

• When and how do these dynamics break down-is there a phase transition? Does the system exhibit a critical threshold beyond which learning abruptly fails or reverses direction, analogous to a phase transition in physical systems?

• Can we predict the rate of accuracy improvement analytically? Can we derive a closed-form drift (or mean-field ODE) for an accuracy-related state variable (e.g., total bad-mode mass p(t)) that yields time-to-accuracy predictions and cleanly separates rate effects from fate (the limiting performance)?

To address these questions, we introduce the framework RLV<sup>ε</sup>R: Reinforcement Learning with Verifiable Noisy Rewards, to study the effects of noise through the lens of analytical insight.

## 2 RLVR with Sloppy Rewards

Group-normalized reinforcement learning with verifiable rewards (RLVR) is an effective method to improve the policy optimization via simple, iterative loop. The goal is to maximize the expected reward by contrasting the performance of multiple completions generated from the same prompt. A single iteration proceeds as follows:

Figure 3: Rate, not Fate. Our mean-field multi-armed bandit analysis predicts that, whenever learning succeeds $( J > 0 )$ , training with noisy rewards asymptotically converges to the same final accuracy as training on clean rewards; the noise level controls only the convergence speed, not the eventual performance. The same phenomena happen for $J < 0 .$ . Our mean field model, along with experimental observation in Fig. 1 supports the “convergence rate is different, fate is the same” prediction.

1. Sampling. For a given prompt x, we draw a cohort of G independent completions from the current policy:

$$
y _ {1}, \dots , y _ {G} \sim \pi_ {\theta} (\cdot | x).
$$

Equivalently, sample indices $I _ { g } \stackrel { \mathrm { i i d } } { \sim }$ Categorical(p) with $\mathfrak { p } = \mathrm { s o f t m a x } ( z )$ , and let $y _ { g }$ be the completion associated with $I _ { g } .$

2. Scoring. We assign a raw reward $r _ { g } : = r ( x , y _ { g } )$ to each completion using a programmatic rule or a learned reward model.

3. Group Normalization. To isolate the relative quality of each completion, we compute per-sample advantages by standardizing rewards within the group:

$$
\widehat {A} _ {g} = \frac {r _ {g} - \bar {r}}{\sigma_ {r} + \varepsilon},
$$

where r¯ and $\sigma _ { r }$ are the empirical mean and standard deviation of the rewards in the current batch.

4. Update. We perform a policy-gradient step to reinforce high-advantage completions (deferring PPO-style clipping or KL penalties to §.6). The core update is:

$$
\Delta \theta = \eta \frac {1}{G} \sum_ {g = 1} ^ {G} \widehat {A} _ {g} \nabla_ {\theta} \log \pi_ {\theta} (y _ {g} | x).\tag{3}
$$

## 2.1 Learning Dynamics

To understand the essence of this update mechanism, let’s consider the simple binary outcome setup where the LLM generates either a “good” or “bad” solution. (In the §5, we will generalize the setup to the most general case.) Let p be the probability of generating a bad solution, controlled by a logit z such that $p = \sigma ( z ) = 1 / ( 1 + e ^ { - z } ) \equiv \pi ( \mathrm { B a } \bar { \mathrm { d } } )$

Good vs. Bad. For a per-sample normalized advantage ${ \widehat { A } } ,$ the update direction is determined by the correlation between the normalized advantage $\widehat { A }$ and the gradient of the log-probability such that the expected logit update is

$$
\Delta z \propto \mathbb {E} \left[ \widehat {A} \nabla_ {z} \log \pi (a) \right].
$$

In a binary setting, the score function simplifies to $\nabla _ { z } \log \pi ( \mathsf { b a d } ) = 1 - p$ and $\nabla _ { z } \log \pi ( \mathrm { g o o d } ) = - p$

By defining the conditional expected advantages $f ( \mathbf { b a d } ) = \mathbb { E } [ \widehat { A } \mid \mathbf { b a d } ]$ and $f ( \mathrm { g o o d } ) = \mathbb { E } [ \widehat { A } \mid \mathrm { g o o d } ] .$ , we can compute the full expectation over actions:

$$
\mathbb {E} [ \widehat {A} \nabla_ {z} \log \pi (a) ] = p f (\text {bad}) (1 - p) + (1 - p) f (\text {good}) (- p) = p (1 - p) \big (f (\text {bad}) - f (\text {good}) \big).
$$

Passing to continuous time using $\begin{array} { r } { \dot { p } = \frac { d p } { d z } \dot { z } = p ( 1 - p ) \dot { z } } \end{array}$ together with equation 3, we observe:

## THE LAW OF MOTION

Group normalization up-weights better-than-average samples and down-weights worse-thanaverage ones. In the simplest two-class reduction $( ^ { \prime \prime } \mathrm { g o o d ^ { \prime \prime } v s . \ ^ { \prime } b a d ^ { \prime \prime } } )$ , this induces a replicator-skeleton for the bad mass $p ( t ) = { \dot { ^ { \prime } } } \pi _ { \theta } ( \mathbf { b a d } \mid x )$ :

$$
\dot {p} (t) = - \eta [ p (t) (1 - p (t)) ] ^ {2} (f (\text { good }) - f (\text { bad })), \quad \text { GRPO   Dynamics }\tag{4}
$$

where $f ( \mathrm { c l a s s } ) : = \mathbb { E } [ \widehat { A } \mid \mathrm { c l a s s } ]$ . Thus if $f ( \mathrm { g o o d } ) > f ( \mathrm { b a d } )$ , then $\dot { p } < 0$ and accuracy rises.

Despite its simplicity, equation 4 provides profound insight into the behavior of RL algorithms under group normalization. As long as good solutions yield higher normalized scores than bad ones, that is, when $f ( \mathrm { g o o d } ) > f ( \mathrm { b a d } )$ , the system drains probability mass, $p ( t )$ , from the bad state.

Interpretation. More generally, the dynamics of the form

$$
\dot {p} _ {i} (t) = p _ {i} (t) \left(f _ {i} (p (t)) - \bar {f} (p (t))\right), \quad \bar {f} (p) = \sum_ {j} p _ {j} f _ {j} (p),
$$

Replicator Dynamics

are called the replicator dynamics. Here, $f$ is the fitness function, $p _ { i } \geq 0$ and $\textstyle \sum _ { i } p _ { i } = 1$ , and each type (or strategy) i is rewarded or penalized according to how its fitness compares with the population average: types with above-average fitness grow in frequency, while those with below-average fitness decline (Cressman (2003)).

GRPO is like a natural selection. The one with fitness above the average will survive eventually.

## 2.2 Noisy Rewards

To understand how reward noise shapes GRPO dynamics in equation 4, we consider the binary-reward setting $r \in \{ 0 , 1 \}$ in which the observed signal is corrupted by the false-positive and false-negative rates introduced in equation 1. As summarized by Youden’s index J in equation ${ \bf \Pi } _ { 2 , \bullet } ^ { \bullet }$ such a “sloppy” grader departs from the ground truth. Under this noise model, the effective success probability $q ( p )$ and the reward variance $\sigma ( p ) ^ { 2 }$ become functions of the bad-mass $p { : }$

$$
q (p) := \mathbb {E} [ r ] = (1 - \delta_ {\mathrm{FN}}) - (1 - \delta_ {\mathrm{FP}} - \delta_ {\mathrm{FN}}) p, \quad \sigma (p) ^ {2} := \operatorname{Var} [ r ] = q (p) (1 - q (p)).
$$

Signal vs. Noise. How does this corruption affect the learning dynamics? Remarkably, under group normalization, the advantage gap takes a purely geometric form (see Appendix C):

$$
\mathbb {E} [ \widehat {A} \mid \mathrm{good} ] - \mathbb {E} [ \widehat {A} \mid \mathrm{bad} ] = \frac {1 - \delta_ {\mathrm{FN}} - \delta_ {\mathrm{FP}}}{\sigma (\delta_ {\mathrm{FN}} , \delta_ {\mathrm{FP}} , p)} \equiv \frac {J}{\sigma (p)}.\tag{5}
$$

The Crucial Role of Youden’s Index. Substituting this back into the equation describing our dynamics, we see that J acts as a signed coefficient of friction for learning.

• If $J > 0$ (Signal): The grader is better than random chance. The update pushes mass toward the correct solution, though the speed is throttled by the noise level J.

• If $J < 0$ (Anti-Signal): The grader is systematically misleading. Learning actively reverses, optimizing for the wrong objective.

• If $J = 0$ (Noise): The gradient vanishes. The advantage gap collapses to zero, leaving the policy drifting in neutral territory regardless of the sample size.

## 3 Phase Transition

Let’s continue to work with the binary outcome setting of the previous section. Substituting the noisy reward gap from equation 5 into the two-state GRPO dynamics equation 4, we obtain the following scalar dynamics for the bad mass $p = { \mathrm { P r } } ( { \mathrm { b a d } } )$ :

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} p ^ {2} (1 - p) ^ {2}, \qquad J = \mathrm{TPR-FPR}.\tag{6}
$$

Here J is Youden’s index (the effective signal polarity), and $\sigma ( p ) > 0$ is the group-normalized reward standard deviation at bad-mass level p.

## THREE LEARNING REGIMES.

The sign of J completely determines learning behavior: (Assuming p<sub>0</sub> ̸= 0, 1 ≡ degenerate cases)

$$
\begin{array}{r l} J > 0 & \Rightarrow \dot {p} <   0 \Rightarrow \text {Learning: bad mass shrinks, accuracy rises to 1} \\ J = 0 & \Rightarrow \dot {p} = 0 \Rightarrow \text {Neutral: no systematic improvement, pure drift} \\ J <   0 & \Rightarrow \dot {p} > 0 \Rightarrow \text {Anti - learning: bad mass grows, accuracy decays to 0} \end{array}
$$

This reveals a sharp phase transition at the critical boundary TPR = FPR, where the reward signal crosses from informative to misleading.

Notice that in the noise-free setting (a perfect oracle), equation 6 simplifies to

$$
\dot {p} = - \eta p ^ {3 / 2} (1 - p) ^ {3 / 2}, \quad \text { GRPO   Dynamics   with   Perfect   Oracle }\tag{7}
$$

which represents the baseline GRPO dynamics in the absence of reward noise.

Note: In practice, TPR and FPR may drift as policy and rewarder co-evolve. without losing any generality, the equation above captures the instantaneous learning direction at each moment.

## 3.1 Bifurcation at the Critical Point

Fixed points and stability. For any $J \neq 0 .$ , the system has two boundary equilibria: $p ^ { \star } = 0$ (all good) and $p ^ { \star } = 1$ (all bad), where $p ^ { \bar { 2 } } ( 1 - p ) ^ { 2 }$ vanishes. Which equilibrium attracts depends entirely on the sign of J:

• When $J > 0$ (learning regime): $p = 0$ is globally attracting on (0, 1) while $p = 1$ repels. Starting from any nontrivial mixture, accuracy $1 \dot { - } p ( t )$ climbs monotonically toward perfection.

• When $J < 0$ (anti-learning regime): The basin structure inverts completely. Now $p = 1$ attracts and $p = 0$ repels, causing RLVR updates to systematically degrade performance until accuracy collapses to zero.

• At the knife edge $J = 0 :$ The entire interval [0, 1] becomes a continuum of neutrally stable fixed points— directional information vanishes entirely.

Crossing $J = 0$ induces a fundamental qualitative change in learning dynamics, delineating the boundary between reward signals that guide learning and those that actively mislead it. The ODE is identically zero in this case.

Special case: $J = 1 .$ . In the noise-free regime $( J = 1 )$ , where rewards are perfectly reliable, the closed-form solution of equation 7 gives an explicit trajectory for the bad-arm probability as a function of gradient steps:

$$
p (t) = \left\{ \begin{array}{l l} \frac {1}{2} + \frac {1}{2} \frac {\varphi (p _ {0}) - \frac {\eta}{2} t}{\sqrt {4 + \big (\varphi (p _ {0}) - \frac {\eta}{2} t \big) ^ {2}}}, & \text {if p_{0} \neq 0,1,\quad \varphi(p): = \frac {2p - 1}{\sqrt{p(1 - p)}}}. \\ 0, 1, & \text {if p_{0} = 0,1 (\equiv LLM has / hasn^{\prime} t absol} \end{array} \right.\tag{8}
$$

In particular, if $p _ { 0 } \neq 0 , 1$ , then the late–time asymptotic is

$$
p (t) \sim \frac {4}{\eta^ {2} t ^ {2}} \rightarrow 0, \qquad t \rightarrow \infty .
$$

Thus, under perfectly reliable labels, the bad mass decays with a universal $t ^ { - 2 }$ tail: the accuracy $1 - p ( t )$ races toward 1 at a polynomial rate determined entirely by the learning rate η.

RLVR limitation. Equation equation 8 exposes an support barrier: the boundary states $p _ { 0 } \in \{ 0 , 1 \}$ are absorbing, so modes with zero initial mass cannot be created by RLVR. In particular, the analysis assumes $1 - p _ { 0 } > 0$ so that correct solutions are sampled occasionally and the gradient has something to amplify; if instead $1 - p _ { 0 } = 0$ (the prompt lies beyond the base model’s capability), the dynamics are trapped at the degenerate equilibrium $p ( t ) \dot { = } 1$ and learning never takes off. Thus RLVR can sharpen and reweight reasoning paths already present in the base model, but it cannot reliably expand capability beyond its initial support, consistent with findings that RLVR mainly boosts pass@1 while large-k coverage can shrink (Yue et al. (2025)).

Asymptotics: which tail, and when. The late-time behavior depends critically on whether the reward variance $\sigma ( p )$ vanishes at the attracting equilibrium. Define the boundary variances. Let $\sigma _ { 0 } = \sqrt { \left( 1 - \delta _ { \mathrm { F N } } \right) \delta _ { \mathrm { F N } } }$ and $\sigma _ { 1 } : = \sqrt { \delta _ { \mathrm { F P } } ( 1 - \delta _ { \mathrm { F P } } ) }$ . The late-time decay is governed by whether the reward variance $\sigma ( p )$ vanishes at the attracting vertex.

Case $\mathbf { ( i ) } \colon J > 0$ with attractor at $p = 0 .$ . Two distinct regimes emerge:

• If Nondegenerate noise $( \delta _ { \mathrm { F N } } > 0 )$ , then $\sigma ( p ) \to \sigma _ { 0 } > 0$ and late-time behavior (0) of the dynamics will be

$$
\dot {p} = - \eta \frac {J}{\sigma_ {0}} p ^ {2} + o (p ^ {2}) \Rightarrow p (t) \sim \frac {\sigma_ {0}}{\eta J} \frac {1}{t}.
$$

• If Variance-degenerate case $( \delta _ { \mathrm { F N } } = 0 )$ , then $\sigma ( p ) = \sqrt { q ( 1 - q ) } \sim \sqrt { J p }$ and

$$
\dot {p} = - \eta \sqrt {J} p ^ {3 / 2} + o (p ^ {3 / 2}) \Rightarrow p (t) \sim \frac {4}{\eta^ {2} J} \frac {1}{t ^ {2}}.
$$

Case (ii): $J < 0$ with attractor at $p = 1$ . Defining the good mass $u ( t ) : = 1 - p ( t )$ , we find $q ( 1 ) = \delta _ { \mathrm { F P } }$ and $\sigma ( 1 ) = \sigma _ { 1 }$ . Since $\sigma _ { 1 } > 0$ whenever $\delta _ { \mathrm { F P } } \in ( 0 , 1 )$

$$
\dot {u} = - \frac {\eta | J |}{\sigma_ {1}} u ^ {2} + o (u ^ {2}) \Rightarrow u (t) = 1 - p (t) \sim \frac {\sigma_ {1}}{\eta | J |} \frac {1}{t}.
$$

Note that no $t ^ { - 2 }$ regime appears here: setting $\delta _ { \mathrm { F P } } = 0$ would force $J \geq 0 ,$ , precluding this case.

Therefore, the learning direction is determined by $\mathrm { s i g n } ( J )$ , while the asymptotic tail follows a universal pattern:

$$
\text { error } \sim \left\{ \begin{array}{l l} \mathcal {O} (t ^ {- 1}) & \text { when   reward   variance   is   nonzero   at   the   attractor }, \\ \mathcal {O} (t ^ {- 2}) & \text { in   the   variance - degenerate   case   (\delta_ {\mathrm{FN}} = 0, J > 0).} \end{array} \right.
$$

This provides a crisp analytical characterization of test-time error $p ( t )$ under noisy reward signals.

## 3.2 Rate, Not Fate

As long as $J > 0$ , the one–dimensional ODE for the bad mass has a single basin of attraction. In particular, both the noisy and the noise-free dynamics converge to the same limiting state (up to the critical knife-edge where the sign of J flips). In this sense, reward noise does not change thefate of training: the basin is the same, and so is the final performance. What changes is only the speed at which we flow toward that basin.

More precisely, comparing the noisy and perfectly reliable dynamics gives the simple time–rescaling

$$
\frac {\dot {p} _ {\text {noisy}}}{\dot {p} _ {\text {perfect}}} \propto \frac {1}{J}\tag{9}
$$

Thus, for example, when $J = 0 . 5 ,$ , the noisy system needs roughly twice compute steps to trace out the same trajectory in p. In other words, additional compute can compensate for imperfect data label/reward signal.

In the next sections, we generalize this analysis to the multi-solution setting and incorporate additional components such as importance sampling, the clipping ratio, and a possible KL penalty term, yielding a formulation that fully aligns with practical PPO/GRPO-style algorithms.

## 3.3 Maximal Learnability at Intermediate Bad Mass

In the noiseless regime $J = 1$ , our scalar dynamics for the bad mass $p$ satisfy

$$
| \Delta p | \propto \big [ p (1 - p) \big ] ^ {3 / 2}.
$$

The prefactor $p ( 1 - p )$ is maximized at

$$
p ^ {\star} = \frac {1}{2},
$$

so the largest single-step reduction in bad-arm mass occurs when the current bad probability is neither too small nor too large, but instead sits at the “intermediate” value $p \approx 1 / 2$ . As $p \overset { \_ } { \to } 0$ or $p  1$ , the factor $p ( 1 - p )$ vanishes, and the dynamics slow down: once a prompt is either almost always solved correctly or almost always answered incorrectly, additional GRPO steps make only marginal progress on that prompt.

Connection to prior $" p ( 1 - p ) "$ learnability observations. The emergence of an intermediate-difficulty optimum mirrors patterns observed in several seemingly distinct analyses. Bae et al. (2025) derive progress bounds proportional to $p ( 1 - p )$ and empirically find that prompts with $p ( x ) \approx 0 . 5$ are most learnable. Foster et al. (2025) relate learnability to reward variance; for Bernoulli rewards, $\operatorname { V a r } ( r ) = q ( 1 - q )$ is maximized at $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ . Our mean-field GRPO dynamics provides a complementary dynamical explanation: the same non-saturation phenomenon that yields high information content also governs the instantaneous rate at which bad mass is eliminated. See Appendix D for more details.

Training is most efficient on “medium-difficulty” questions, where the model is roughly 50–50 between good and bad solutions. Under asymmetric noise the optimum shifts but remains at an intermediate bad mass.

## 4 LLM as a Multi-Armed Bandit

In contemporary reinforcement learning (RL) training paradigms for LLMs, such as RLVR and related frameworks, the supervision signal is typically provided only after the model has produced an entire response. Because the reward is evaluated at the completion level rather than per token, it is often more appropriate to treat the entire output sequence as a single decision made by the policy. This viewpoint naturally suggests a bandit-style abstraction, where each sequence corresponds to one action (or ”arm”) and the learning signal is attached to that action as a whole (Kreutzer et al., 2017; Nguyen et al., 2017; Dang et al., 2025b). Earlier sequence-level policy gradient methods–such as RLOO and related REINFORCE variants (Ahmadian et al., 2024)–implicitly operated in this regime, while contemporary approaches like GRPO (Shao et al., 2024) make this perspective explicit by defining advantages and updates directly over full generations.

Adopting the bandit abstraction provides a clean and principled theoretical foundation for our methodology (see Appendix A for further details).

From sequences to modes. To ground this abstraction, consider a fixed prompt x (e.g., a math problem). Sampling an LLM at non-zero temperature yields many distinct token sequences, yet these typically collapse into a small number of reasoning modes–canonical solution paths, recurring chains of thought, or standard solver templates. This clustering exposes a low-dimensional structure: each mode acts as an “arm” in the bandit abstraction, carrying its own probability mass. (Recent work has begun to map the landscape of such reasoning modes (Zhou et al., 2025) and to leverage their structure for designing new training algorithms (Zhang et al., 2025a).)

We now formalize this intuition. Fix a prompt x and let $y \sim \pi _ { \omega } ( \cdot \mid x )$ be a sampled completion. With a length cutoff $L _ { \mathrm { m a x } } ,$ the effective support $\mathcal { Y } _ { \le L _ { \mathrm { m a x } } } = \cup _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } \mathcal { V } ^ { \ell }$ is finite. We define a coarse-graining map $\phi : \mathcal { V } _ { \leq L _ { \operatorname* { m a x } } } \to$ H that clusters sequences into reasoning modes $\mathcal { H } = \left\{ h _ { 1 } , \ldots , h _ { K + M } \right\}$ . This induces a categorical mode policy:

$$
\pi_ {\theta} (h \mid x) := \sum_ {y: \phi (y) = h} \pi_ {\omega} ^ {(L)} (y \mid x), \quad \pi_ {\omega} ^ {(L)} (y \mid x) \propto \pi_ {\omega} (y \mid x) \mathbf {1} \{y \in \mathcal {Y} _ {\leq L _ {\max}} \},
$$

which we parameterize by effective logits $\theta = \left( \theta _ { 1 } , \ldots , \theta _ { K + M } \right)$ such that $\pi _ { \theta } ( h _ { i } \mid x ) = \operatorname { s o f t m a x } ( \theta )$ <sub>i</sub>. The key quantity for our analysis is the total probability mass on incorrect modes, or the bad mass $p { : }$

$$
p = \operatorname * {P r} [ \text { LLM   chooses   an   incorrect   mode } \mid x ].
$$

Two boundary cases are notable: (i) String-matching, where $\phi$ is the identity and $\mathcal { H } = \mathcal { Y } _ { \le L _ { \mathrm { m a x } } } ;$ and (ii) Infinite horizon, where $L _ { \mathrm { m a x } }  \infty$ . In practice, our mode-level statements depend only on $\bar { \pi _ { \theta } } ( \cdot \mid x )$ over $\mathcal { \dot { H } }$ and remain invariant to the specific choice of a reasonable ϕ.

Good vs. bad families. To further analyze the distribution, we partition the set of modes into good (correct) and bad (incorrect) subsets:

$$
\mathcal {H} = \mathcal {H} ^ {+} \cup \mathcal {H} ^ {-}, | \mathcal {H} ^ {+} | = K, | \mathcal {H} ^ {-} | = M.
$$

Let α and p denote the aggregate mass of the correct and incorrect families, respectively:

$$
\alpha = \sum_ {h \in \mathcal {H} ^ {+}} \pi_ {\theta} (h \mid x), \quad p = \sum_ {h \in \mathcal {H} ^ {-}} \pi_ {\theta} (h \mid x) = 1 - \alpha .
$$

We define the relative distribution within the good modes as $y \in \Delta ^ { K - 1 }$ and within the bad modes as $z \in \Delta ^ { M - 1 }$ , where:

$$
y _ {i} := \frac {\pi_ {\theta} (h _ {i} \mid x)}{\alpha} \text {   for   } h _ {i} \in \mathcal {H} ^ {+}, \quad z _ {j} := \frac {\pi_ {\theta} (h _ {j} \mid x)}{p} \text {   for   } h _ {j} \in \mathcal {H} ^ {-}.
$$

Thus, the full distribution over all arms is given by the vector $\left( \alpha y _ { 1 } , \dots , \alpha y _ { K } , p z _ { 1 } , \dots , p z _ { M } \right)$ . This decomposition allows us to analyze the model’s performance (α vs. p) independently from its internal preference for specific reasoning paths (y and z).

## LLM AS MULTI-ARMED BANDIT

Treat each prompt as a tiny bandit problem. The model spreads probability across a few reasoning modes (arms). A sequence-level grader marks modes as correct (1) or incorrect (0). GRPO pushes probability toward above-average arms and away from below-average ones. Under noise-free rewards, the mass on incorrect modes shrinks monotonically.

## 5 Geometric Flow on the Probability Simplex

We model the policy as a probability vector p distributed over K “good” arms and a distinct “bad” arm comprising M internal modes (representing bad-mass). The state space is the simplex:

$$
\mathbf {p} = \left(p _ {1}, \dots , p _ {K}, \dots , p _ {K + M}\right) \in \Delta^ {K + M - 1}, \quad \text { where } \quad \Delta^ {d} = \left\{\mathbf {x} \in \mathbb {R} _ {\geq 0} ^ {d + 1}: \mathbf {1} ^ {\top} \mathbf {x} = 1 \right\}.
$$

We denote $p : = p _ { \mathrm { b } }$ as the aggregate bad-mass coordinate. Admissible velocity fields are constrained to the tangent space $T _ { \mathbf { p } } \dot { \Delta } ^ { K + M - 1 } = \big \{ v \in \mathbb { R } ^ { K + M } : \mathbf { 1 } ^ { \top } v = 0 \big \}$ , ensuring total probability mass is strictly conserved.

Figure 4: Geometry of the Probability Simplex. The policy p evolves on the non-Euclidean manifold $\Delta ^ { \check { K } + M - 1 }$ . The softmax Jacobian $\Im ( { \mathfrak { p } } )$ endows this space with the Shahshahani (Fisher) geometry, projecting updates onto the zero-sum tangent space $T _ { \pmb { \ p } } .$ . GRPO induces a mass-conserving replicator flow, $\dot { \pmb { \dot { \mathsf { p } } } } = \eta \widetilde { \pmb { \mathsf { y } } } ( \pmb { \mathsf { p } } ) \widetilde { \bf A } ,$ which dynamically redistributes probability mass based on relative advantage. In the local tangent space, the forward KL divergence manifests as the quadratic form $\begin{array} { r } { \frac { 1 } { 2 } \delta ^ { \top } \operatorname { D i a g } ( \mathbf { p } ) ^ { - 1 } \delta , } \end{array}$ , identifying $\Im ( { \mathfrak { p } } )$ as the inverse Riemannian metric.

The geometry of optimization on this manifold is governed by the softmax Jacobian, $\Im ( { \bf p } ) = \mathrm { D i a g } ( { \bf p } ) - { \bf p } { \bf p } ^ { \top }$ This operator acts as the inverse of the Riemannian metric tensor (associated with the Shahshahani or Fisher-Rao metric), mapping gradients into natural gradients on the tangent space:

$$
\mathfrak {J} (\mathbf {p}) v = \mathbf {p} \odot (v - \bar {v}), \quad \text { with } \quad \bar {v} := \mathbf {p} ^ {\top} v.\tag{10}
$$

Under the GRPO objective, the policy update follows a continuous-time natural gradient flow

$$
\dot {\mathbf {p}} = \eta \Im (\mathbf {p}) ^ {2} \mathbf {A}, \quad G R P O d y n a m i c s o n \Delta^ {K + M - 1},
$$

where the vector A comes from GRPO-style advantage computation. Equivalently, this can be expressed in the “replicator-flow” form:

$$
\dot {\mathbf {p}} = \eta \mathbf {p} \odot [ \mathfrak {J} (\mathbf {p}) \mathbf {A} - \langle \mathbf {p}, \mathfrak {J} (\mathbf {p}) \mathbf {A} \rangle \mathbf {1} ].\tag{11}
$$

This “replicator dynamics” form reveals two key properties: (1) Multiplicativity, ensuring the faces of the simplex remain invariant; and (2) Relative Performance, where mass flows between arms strictly based on their advantage relative to the mean (see Appendix. C for full details).

## 5.1 Decoupling the Dynamics: Shape vs. Good and Bad Mass

To disentangle the evolution of the “good” policy structure from the decay of the “bad” masses, we decompose the state vector. In the simplex interior, we parameterize p as:

$$
\mathbf {p} = \big ((1 - p) y, p z \big),
$$

where $y \in \Delta ^ { K - 1 }$ and $z \in \Delta ^ { M - 1 }$ are the normalized distributions over good and bad arms respectively, and $p \in ( 0 , \bar { 1 } )$ represents the total bad mass.

Applying this coordinate change to equation 11 decouples the system into shape dynamics (internal to y and z) and mass dynamics (governing p). The block-diagonal evolution equations are:

$$
\dot {y} = + \kappa (p) y \odot \left(y - \| y \| _ {2} ^ {2} \mathbf {1}\right),\tag{12a}
$$

$$
\dot {z} = - \kappa (p) z \odot \left(z - \| z \| _ {2} ^ {2} \mathbf {1}\right),\tag{12b}
$$

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} \left(\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}\right),\tag{12c}
$$

where the time-rescaling factor is $\begin{array} { r } { \kappa ( p ) : = \eta \frac { J } { \sigma ( p ) } p ( 1 - p ) } \end{array}$

Remark 5.1. In the noise-free regime $( J = 1 )$ with $\sigma ( p ) = { \sqrt { p ( 1 - p ) } }$ , this factor simplifies to

$$
\kappa (p) = \eta \sqrt {p (1 - p)}, \quad \frac {\eta J}{\sigma (p)} [ p (1 - p) ] ^ {2} = \eta   p (1 - p) ^ {\frac {3}{2}}.
$$

This specialization highlights that noise affects the effective time scale and sign (direction) of the dynamics, while the geometric structure of the flows remains unchanged.

Remark 5.2. In the mass dynamics equation 12c, the denominator is the standard deviation $\sigma ( p ) = ( q ( p ) ( 1 -$ $q ( p ) ) ) ^ { 1 / 2 }$ , where $q ( p ) : = ( 1 - \delta _ { \mathrm { F N } } ) - J p$ . Since $J = 1 - \delta _ { \mathrm { F P } } - \delta _ { \mathrm { F N } } ,$ we have $\sigma ( p ) \in ( 0 , 1 )$ for all $p \in ( 0 , 1 )$ except in the singular case where $J = 0 .$

Interpretation. The coupled system equation 12 highlights a competition between three distinct geometric forces:

Diversity Collapse in Good Arms: Equation equation 12a describes a self-reinforcing flow. The term $\| y \| _ { 2 } ^ { 2 }$ (the collision probability) acts as a threshold: arms with mass $y _ { i } > \| y \| _ { 2 } ^ { 2 }$ grow super-linearly, causing the distribution to sharpen and diversity to collapse onto the optimal arms. Let $S ^ { \star } : \doteq$ arg $\mathrm { m a \bar { x } } _ { i \in [ K ] } y _ { i } ( 0 )$ , then for $J > 0 \mathrm { j }$

$$
y _ {i} (t) \rightarrow 0 \quad (i \notin S ^ {\star}), \quad \text { GRPO's   Diversity   Collapse }
$$

Entropy Increase in Bad Arms: Conversely, the bad-mass distribution z evolves under a negative feedback loop equation 12b. This flow pushes z away from concentration and toward the uniform distribution (maximum entropy) on $\Delta ^ { M - \hat { 1 } }$

$$
z (t) \longrightarrow \frac {1}{M} {\bf 1},
$$

This “spreading” effect slows the decay of $p$ if the bad mass is diffuse (i.e., when $\| z \| _ { 2 } ^ { 2 } \approx 1 / M )$

Bad Mass Evolution: The total bad mass $p$ decays monotonically (provided $J > 0 )$ at a rate proportional to $[ p ( 1 - p ) ] ^ { 2 }$ , but the rate is modulated by the structural sparsity $\| y \| _ { 2 } ^ { 2 } + \| z \| _ { 2 } ^ { 2 }$ . In the late-time limit, as the good arms collapse $( \| y \| _ { 2 } ^ { 2 } \to 1 )$ and bad arms homogenize $( \| z \| _ { 2 } ^ { 2 } \to 1 / M )$ , the decay rate stabilizes, driven purely by the system’s reward gap J

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} \big (\| z \| _ {2} ^ {2} + \| y \| _ {2} ^ {2} \big) \xrightarrow [ t \to \infty ]{} \approx - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2},
$$

Motion is slow near the faces $p \approx 0$ and $p \approx 1$ and faster in the interior. In the learning regime $J > 0 ,$ the prefactor $J / \sigma ( p )$ is positive, ensuring p is monotonically driven down.

Thus, the simplex dynamics factor cleanly into a shape evolution within the good-arms simplex and a mass evolution that suppresses the bad arms.

## 5.2 The right geometry: Shahshahani metric on ∆

The geometry of the simplex is fundamentally non-Euclidean. Displacing probability mass from a rare arm incurs a “higher cost” than displacing the same amount from a common arm. The Shahshahani metric (Shahshahani $( 1 9 7 9 )$ formalizes this intuition by weighting tangent directions inversely proportional to the square root of probabilities (refer to Appendix H for details):

$$
\langle u, v \rangle_ {\mathrm{Shah}; \mathbf {p}} = \sum_ {i = 1} ^ {K + M} \frac {u _ {i} v _ {i}}{p _ {i}}, \qquad u, v \in T _ {\mathbf {p}} \Delta^ {K + M - 1}.
$$

This definition corresponds precisely to the Fisher-information metric (Amari et al. (2019)) of the categorical family, up to a constant factor. Given a smooth function $F : \Delta ^ { K + M } \xrightarrow { } \mathbb { R }$ , the associated natural gradient is derived by projecting and reweighting the Euclidean gradient:

$$
\operatorname{grad} _ {\text { Shah }} F (\mathbf {p}) = \mathfrak {J} (\mathbf {p}) \nabla F (\mathbf {p}) = \mathbf {p} \odot (\nabla F (\mathbf {p}) - \langle \mathbf {p}, \nabla F (\mathbf {p}) \rangle \mathbf {1}) \in T _ {\mathbf {p}} \Delta^ {K + M - 1}.
$$

This structure simplifies significantly for the good-arm subsystem. The evolution of $\dot { y }$ in equation 12a expresses a Shahshahani gradient flow of the potential Φ, scaled by $\kappa ( p )$

$$
\dot {y} = \kappa (p) \operatorname{grad} _ {\text {Shah}} \Phi (y), \quad \Phi (y) := \frac {1}{2} \| y \| _ {2} ^ {2} = \frac {1}{2} \sum_ {i = 1} ^ {K} y _ {i} ^ {2}.
$$

Consequently, within the good block, the dynamics perform natural-gradient ascent on a potential that incentivizes concentration. This mechanism drives the mass toward a single dominant good arm.

## TAKEAWAY

In the zero-noise limit, GRPO reduces to pure selection: centered rewards induce a replicator-type drift, with effective “fitness” given by the natural-gradient signal $\Im ( \mathfrak { p } ) \mathbf { A } ,$ shifting probability mass from below-average to above-average types, and introducing no additional nonlinearities beyond the softmax Jacobian.

KL-regularized mirror ascent and the replicator limit. A complementary discrete-time perspective employs entropic (KL) regularization. Let $\bar { \mathsf { p } } \in \Delta ^ { K + M - 1 }$ and $\mathbf { A } \in \dot { \mathbb { R } } ^ { K + M }$ be fixed. For a step size $\eta > 0$ , we consider the KL-regularized maximization problem (cf. Proposition H.7):

$$
\mathbf {p} ^ {+} = \arg \max _ {\mathbf {q} \in \Delta^ {K}} \left\{\langle \mathbf {A}, \mathbf {q} \rangle - \frac {1}{\eta} D _ {\mathrm{KL}} (\mathbf {q} \| \mathbf {p}) \right\} \Longrightarrow \mathbf {p} ^ {+} = \frac {\mathbf {p} \odot e ^ {\eta \mathbf {A}}}{\mathbf {1} ^ {\top} (\mathbf {p} \odot e ^ {\eta \mathbf {A}})}.\tag{13}
$$

Since the objective function is strictly concave with respect to $q ,$ the maximizer $p ^ { + }$ exists and is unique. Furthermore, it adopts the familiar multiplicative-weights (or exponentiated-gradient) form.

For small $\eta ,$ the first-order expansion of the mirror-ascent step equation $7 8$ corresponds to an Euler step of the natural-gradient (replicator) flow:

$$
\mathbf {p} ^ {+} - \mathbf {p} = \eta \mathbf {p} \odot (\mathbf {A} - \langle \mathbf {p}, \mathbf {A} \rangle \mathbf {1}) + O (\eta^ {2}),
$$

which implies

$$
\dot {\mathbf {p}} = \eta \Im (\mathbf {p}) A = \eta \mathbf {p} \odot (A - \langle \mathbf {p}, A \rangle \mathbf {1}).
$$

In essence, entropic mirror ascent discretizes the intrinsic geometry. It constitutes steepest ascent with respect to the Shahshahani metric, and its infinitesimal limit recovers replicator dynamics on the simplex.

In summary, the comparison is as follows:

$$
\left\{ \begin{array}{l l} \text { GRPO: } & \dot {\mathbf {p}} = \eta   \Im (\mathbf {p}) ^ {2}   \mathbf {A}, \\ \text { GRPO with KL regularization: } & \dot {\mathbf {p}} = \eta   \Im (\mathbf {p})   \mathbf {A}. \end{array} \right.
$$

The resulting analysis highlights three key structural properties:

• Preservation of the simplex constraint. The GRPO-induced dynamics are intrinsically tangent and multiplicative; probabilities remain nonnegative and sum to unity.

• Decoupling of good and bad arm dynamics. In the decomposition ${ \bf p } = ( ( 1 - p ) y , p z )$ , the shape y and y of the good/bad-arm distribution evolve via a Shahshahani gradient flow, whereas the total bad mass p is monotonically suppressed.

• Geometric consistency. The Shahshahani metric captures the natural geometry of ∆. Both the continuous-time GRPO flow equation 11 and the discrete-time KL-mirror step equation 78 represent steepest-ascent procedures under this geometry. The former utilizes the natural-gradient signal $\tilde { \mathfrak { J } } ( \mathfrak { p } ) \mathbf { A } .$ while the latter reduces to the replicator flow $\mathbf { p } \odot \left( \mathbf { A } - \langle \mathbf { p } , \mathbf { A } \rangle \mathbf { 1 } \right)$ in the small-step limit.

## 5.3 Finite Sampling Cause Genetic Drift Noise

If the GRPO mean-field limit is viewed as an analogue to replicator-style natural selection, then the finite sampling of rollouts introduces a stochasticity equivalent to genetic drift. GRPO-style updates rely on group estimates: for each prompt, one averages over a finite set of G rollouts to form a normalized advantage and then applies a policy-gradient step. Replacing population expectations by the empirical group mean introduces an additional randomness even if the underlying reward model were fixed. This $\prime \prime \mathrm { f i n i t e } { - \dot { G } ^ { \prime \prime } }$ effect is conceptually separate from reward noise $( \mathrm { e . g . } \delta _ { \mathrm { F N } } , \delta _ { \mathrm { F P } } ) { : }$ it is simply Monte Carlo error from having only G samples. Consequently, the learning dynamics do not follow the deterministic mean-field drift exactly; instead, they fluctuate around it, with a typical fluctuation size shrinking like $O ( G ^ { - 1 / 2 } )$ (and proportional to the learning rate η). For example, if yˆ is the empirical frequency from G i.i.d. categorical draws with mean y, then

$$
\sqrt {G} (\hat {y} - y) \Rightarrow \mathcal {N} (0, \operatorname{Diag} (y) - y y ^ {\top}).
$$

At the level of the probability vector p on the simplex, this sampling-induced randomness has the canonical Fisher/Wright–Fisher geometry. Indeed, if $I \sim \mathrm { C a t } ( \mathbf { p } )$ and $e _ { I }$ is the corresponding one-hot vector, the score feature $e _ { I } - \mathbf { p }$ has covariance

$$
\operatorname{Cov} \left[ \frac {1}{G} \sum_ {g = 1} ^ {G} \left(e _ {I _ {g}} - \mathbf {p}\right) \right] = \frac {1}{G} \left(\operatorname{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top}\right) := \frac {1}{G} \Sigma (\mathbf {p}).
$$

so averaging G i.i.d. rollouts produces fluctuations of order $\Sigma ( \mathbf { p } ) / G$ . In a diffusion (continuous-time) approximation, this appears as a Wright–Fisher-type noise with diffusion speed, $\nu ,$ added on top of the deterministic drift derived earlier, as demonstrated in Dang et al. (2025a):

$$
d p = \dot {p} d t + \frac {\eta \sqrt {\nu}}{\sqrt {G}} \Sigma (p) ^ {1 / 2} d W _ {t}.
$$

Advantage normalization and noisy rewards $\begin{array} { r } { ( \mathrm { i . e . , } \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \tilde { r } \big ( e _ { I _ { g } } - \mathfrak { p } \big ) ) } \end{array}$ ) primarily rescale the amplitude of this diffusion by an order-one, state-dependent factor, while the simplex-shaped matrix $\Sigma ( \mathbf { p } )$ (and thus the vanishing of noise near the boundary) remains the same.

## 6 Mean-Field Dynamics of Bad Mass in GRPO

In the preceding sections, we analyzed the mean-field probability dynamics induced by REINFORCE-style updates and identified the aggregate bad probability mass as a convenient one-dimensional summary of the learning trajectory. In this section we extend that analysis to GRPO, viewed as a concrete instantiation of a group-normalized policy-gradient method. Relative to the basic REINFORCE update, GRPO typically incorporates two additional mechanisms: (i) PPO-style importance sampling with ratio clipping, and (ii) KL regularization. Our goal here is to derive the corresponding mean-field evolution for the bad mass and to clarify which parts of the algorithm influence the leading-order drift.

The role of clipping and importance sampling. The GRPO update inherits PPO’s importance sampling and ratio clipping by modifying the objective in equation 3. Concretely, each per-sample advantage ${ \mathrm { \dot { A } } } _ { i }$ is reweighted by a clipped importance ratio,

$$
A _ {i} \longmapsto A _ {i} \operatorname{clip} \left(\frac {\pi_ {\text { new }} (i)}{\pi_ {\text { old }} (i)}, 1 - \varepsilon , 1 + \varepsilon^ {\prime}\right),
$$

where $\varepsilon , \varepsilon ^ { \prime } > 0$ are the PPO/GRPO clip thresholds Schulman et al. (2017). In the mean-field limit, our calculation in Appendix F shows that, in the small-step regime with fixed thresholds and $\eta \ll \varepsilon , \varepsilon ^ { \prime } ,$ , clipping and importance sampling do not alter the leading-order mean-field drift. Their contribution is absorbed into the $\mathcal { O } ( \dot { \eta } ^ { 2 } )$ ) remainder, and therefore does not affect the first-order phase portrait. We refer to Appendix F for the detailed expansion.

We now state the resulting closed mean-field equation for the aggregate bad mass, together with its internaltime logit form and the corresponding small-heterogeneity refinement:

In the multi-bad-arm setting, define the aggregate bad mass and the within-block normalized states by

$$
p (t) := \sum_ {m = 1} ^ {M} p _ {b _ {m}} (t) \in [ 0, 1 ], \qquad y _ {j} (t) := \frac {p _ {j} (t)}{1 - p (t)} \in \Delta^ {K - 1}, \qquad z _ {m} (t) := \frac {p _ {b _ {m}} (t)}{p (t)} \in \Delta^ {M - 1},
$$

and set the within-block collision masses

$$
s _ {2} (t) := \| y (t) \| _ {2} ^ {2} \in \left[ \frac {1}{K}, 1 \right], \qquad t _ {2} (t) := \| z (t) \| _ {2} ^ {2} \in \left[ \frac {1}{M}, 1 \right].
$$

The associated geometry factor is

$$
C _ {\mathrm{geo}} (t) := s _ {2} (t) + t _ {2} (t) \in \Big [ \frac {1}{K} + \frac {1}{M}, 2 \Big ],
$$

we have:

## THEOREM

Theorem 6.1 (Bad-mass ODE, internal-time logit form, and first-order geometry reduction). Under group-normalized GRPO with small stepsize $\eta \ll \breve { \varepsilon } , \varepsilon ^ { \prime } ( P P O$ -clipping factors) and fresh on-policy groups, the aggregate bad mass obeys the mean-field ODE

$$
\dot {p} (t) = - \eta \frac {J}{\sigma (p (t))} [ p (t) (1 - p (t)) ] ^ {2} C _ {\mathrm{geo}} (t) + \mathcal {O} (\eta^ {2}),\tag{14}
$$

where $J { = } T P R$ -FPR reflects the good–bad advantage gap, and $\sigma ( p ) > 0$ is the group-normalization scale.

Internal-time logit form. Assume $J \neq 0$ and define the logit

$$
L (t) := \log \frac {p (t)}{1 - p (t)},
$$

together with the internal time change

$$
\tau (t) := \int_ {0} ^ {t} \eta \frac {| J |}{\sigma (p (u))} p (u) (1 - p (u)) d u.\tag{15}
$$

Viewing $p , y , z$ asfunctions of τ via $t = t ( \tau )$ , one has, whenever $p ( t ) \in ( 0 , 1 )$

$$
\frac {d L}{d \tau} = - \operatorname{sign} (J) C _ {\mathrm{geo}} (\tau) = - \operatorname{sign} (J) \left(s _ {2} (\tau) + t _ {2} (\tau)\right).\tag{16}
$$

$I f J = 0$ , the deterministic drift term vanishes at this order.

Small-heterogeneity regime. Write the within-block states at $\tau = 0$ as

$$
y (0) = u _ {K} + v _ {0}, \quad \sum_ {j = 1} ^ {K} (v _ {0}) _ {j} = 0, \qquad z (0) = u _ {M} + w _ {0}, \quad \sum_ {m = 1} ^ {M} (w _ {0}) _ {m} = 0,
$$

where $u _ { K } = ( 1 / K , \dots , 1 / K )$ and $u _ { M } = ( 1 / M , \dots , 1 / M )$ . Define the heterogeneities

$$
\zeta_ {0} := \| v _ {0} \| _ {2} ^ {2} = s _ {2} (0) - \frac {1}{K}, \qquad \xi_ {0} := \| w _ {0} \| _ {2} ^ {2} = t _ {2} (0) - \frac {1}{M}.
$$

Then Theorem I.7 implies that, in the near-uniform regime (that $i s ,$ while the conditions of that theorem hold),

$$
L (\tau) = L (0) - \operatorname{sign} (J) \left(\frac {1}{K} + \frac {1}{M}\right) \tau - \operatorname{sign} (J) \frac {K}{2} \zeta_ {0} \left(e ^ {\frac {2}{K} \tau} - 1\right) - \operatorname{sign} (J) \frac {M}{2} \xi_ {0} \left(1 - e ^ {- \frac {2}{M} \tau}\right) + \widetilde {R} _ {L} (\tau),\tag{17}
$$

with a remainder $\widetilde { R } _ { L } ( \tau )$ controlled by the same $O ( \zeta _ { 0 } ^ { 3 / 2 } e ^ { 3 \tau / K } + \xi _ { 0 } ^ { 3 / 2 } )$ bound as in Theorem I.7.

In particular, to first order the dependence of the bad-mass drift on the within-block initialization $y ( 0 ) , z ( 0 )$ enters only through the two scalars $\left( \zeta _ { 0 } , \xi _ { 0 } \right)$ . Allfiner details contribute only at order $O ( \zeta _ { 0 } ^ { 3 / 2 } )$ and $O \big ( \xi _ { 0 } ^ { 3 / 2 } \big )$ and higher.

Sign structure. Since $C _ { \mathrm { g e o } } ( \tau ) > 0 f o r$ all interior states, equation 16 yields the global monotonicity:

$I f J > 0 ,$ , then L decreases in τ, hence $p ( t )$ decreases monotonically toward 0 (learning succeeds).

$I f J < 0 ,$ , then L increases in $\tau ,$ hence $p ( t )$ increases monotonically toward 1 (anti-learning).

$I f J = 0$ , the deterministic drift vanishes at this order (neutral evolution).

## 6.1 KL Regularization: From Phase Transition to Interior Equilibrium

The introduction of a KL penalty toward a reference bad mass $p _ { \mathrm { r e f } }$ incorporates a restoring drift within the probability space. When enforcing the KL term across the two classes (the forward-KL format), the contribution to the dynamics is given by:

$$
\dot {p} \big | _ {\mathrm{KL}} = - \beta   p (1 - p) \Big (\log \frac {p}{1 - p} - \log \frac {p _ {\mathrm{ref}}}{1 - p _ {\mathrm{ref}}} \Big).
$$

Consequently, the KL-regularized bad-mass ODE takes the form:

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} C (y, z) - \beta p (1 - p) \left(\log \frac {p}{1 - p} - \log \frac {p _ {\text {ref}}}{1 - p _ {\text {ref}}}\right).\tag{18}
$$

An analogous form holds for the full reverse-KL penalty, where the logit gap $\ell - \ell _ { \mathrm { r e f } }$ is replaced by $\ell - \ell _ { \mathrm { r e f } } -$ $D _ { \mathrm { K L } } ( y \vert \vert y ^ { \mathrm { r e f } } ) + D _ { \mathrm { K L } } ( z \vert \vert z ^ { \mathrm { r e f } } ) ,$ ; a detailed derivation is provided in Appendix $G .$

An immediate consequence of equation 18 is that the regularized dynamics now support interior fixed points $p ^ { \star } \in ( 0 , 1 )$ ). At these points, the reward-driven drift is exactly balanced by the KL anchoring:

$$
\beta \big (L (p ^ {\star}) - L (p _ {\mathrm{ref}}) \big) = - \eta \frac {J}{\sigma (p ^ {\star})} p ^ {\star} (1 - p ^ {\star}) C (y, z), \qquad L (p) := \log \frac {p}{1 - p}.\tag{19}
$$

Unique interior equilibrium. For any $\beta > 0$ and fixed $\left( y , z \right)$ , the regularized dynamics admit a unique globally stable fixed point $p ^ { \star } \in ( 0 , 1 )$ , as established in Appendix $\\\boldsymbol { G } .$ The position of this equilibrium is fundamentally determined by the sign of the alignment J:

$$
\begin{array}{l l} \text {If} J > 0: & 0 <   p ^ {\star} <   p _ {\text {ref}} \quad \text {and} \quad p ^ {\star} \searrow 0 \text {as} \beta \downarrow 0; \\ \text {If} J = 0: & p ^ {\star} = p _ {\text {ref}}; \\ \text {If} J <   0: & p _ {\text {ref}} <   p ^ {\star} <   1 \quad \text {and} \quad p ^ {\star} \nearrow 1 \text {as} \beta \downarrow 0. \end{array}
$$

In this sense, KL regularization transforms the sharp phase transition at $J = 0 ,$ , which previously resulted in boundary collapse to $p = 0 \mathrm { o r } p = 1$ , into a smooth and stabilized interior equilibrium for any $\dot { \boldsymbol { { \beta } } } > 0$

Asymptotic regimes. The behavior of the equilibrium depends on the relative strength of the KL anchoring. In the strong-KL regime $( \beta \to \infty )$ , we can linearize equation 19 around $p _ { \mathrm { r e f } }$ to find:

$$
p ^ {\star} \approx p _ {\mathrm{ref}} - \frac {\eta J}{\beta} \frac {\left[ p _ {\mathrm{ref}} (1 - p _ {\mathrm{ref}}) \right] ^ {2}}{\sigma (p _ {\mathrm{ref}})} C (y, z).
$$

Here, $p ^ { \star }$ converges to $p _ { \mathrm { r e f } }$ at a rate of $\mathcal { O } ( \beta ^ { - 1 } )$ . Conversely, in the weak-KL regime $( \beta \downarrow 0 )$ with $J < 0 ,$ , we let $\varepsilon : = 1 - p$ to obtain:

$$
1 - p ^ {\star} \sim \frac {\beta}{c} \log \frac {c}{\beta}, \quad c := \frac {- \eta J}{\sigma (1)} C (y, z) > 0.
$$

Notably, even an infinitesimal $\beta > 0$ is sufficient to prevent total collapse to $p ^ { \star } = 1$ , although the equilibrium may drift arbitrarily close to the bad vertex as $\beta$ vanishes. Complementary to our mean-field ODE view, prior work studies the noise-free GRPO update with KL anchoring and shows that the induced success-probability dynamics follow a fixed-point iteration whose limit depends explicitly on the KL strength Mroueh (2025). We refer the reader to Appendix G for more details on the results presented here.

## 7 Experiments

We validate our theoretical analysis on programmatically verifiable Python coding tasks, testing whether real-world RL training exhibits the predicted phase transition at $J = { \dot { 0 } } .$ , and whether noise merely rescales convergence speed without altering the basin of attraction.

## 7.1 Experimental Hypotheses

Our analysis makes two sharp predictions:

$( \mathcal { H } _ { 1 } )$ Phase Transition at $J = 0 :$ Learning improves accuracy only when Youden’s index $J = \mathrm { T P R } - \mathrm { F P R }$ is positive. $\mathrm { A t } \ J = 0$ , training yields no systematic improvement (neutral drift). For $J < 0$ , accuracy degrades as the system anti-learns.

$( \mathcal { H } _ { 2 } )$ Rate, not Fate: For any $J > 0$ , the sign of the reward signal determines the basin of attraction. Noise level affects only the convergence rate, more rollouts or training steps accelerate progress toward the same equilibrium without changing the asymptotic outcome.


Figure 5: KL regularization smooths the phase transition at $J = 0$ via an interior fixed point. Incorporating a KL penalty toward a reference mass $p _ { \mathrm { r e f } }$ introduces a restoring drift, resulting in the regularized ODE equation 18. For any $\beta > 0 ,$ the system possesses a unique stable interior equilibrium $p ^ { \star } \in ( 0 , 1 )$ ) defined by the balance condition equation 19. This equilibrium satisfies $p ^ { \star } < p _ { \mathrm { r e f } }$ for $J > 0 , \dot { p } ^ { \star } = \dot { p } _ { \mathrm { r e f } }$ at $J = 0 ,$ and $p ^ { \star } > p _ { \mathrm { r e f } }$ for $J < 0$ . The reward-driven component is modulated by the multi-block collision factor $C ( y , z ) ~ = ~ \| y \| _ { 2 } ^ { 2 } + \| z \| _ { 2 } ^ { 2 } ~ \in ~ [ 1 / K + 1 / M , 2 ]$ . As $\beta  \infty$ , the equilibrium approaches $p _ { \mathrm { r e f } }$ due to strong anchoring; as ${ \bar { \beta } } \downarrow 0 ,$ it approaches the reward-driven boundary, effectively smoothing the boundary collapse into a controlled interior state.

## 7.2 Setup

Task and data. We use Python code generation with programmatic verification via unit tests. Our corpus is a filtered subset of high-quality problems from OpenR1 Hugging Face (2025) with $N _ { \mathrm { t r a i n } } = 1 0 { , } 2 3 9$ training prompts and $N _ { \mathrm { v a l } } = \breve { 5 } 9 4$ validation prompts. Each instance includes a natural-language specification, input/output examples, and a test harness combining public and hidden test cases.

Model and evaluation. We fine-tune Qwen2.5-3B as the base policy, evaluating E[pass@1] (the fraction of problems solved on the first attempt) averaged over five independent runs for each hyperparameter configuration.

Training algorithm. We employ standard GRPO with per-group advantage standardization and importance ratio clipping via the VeRL library Sheng et al. (2024). Each group contains $G = 8$ rollouts per prompt, with returns normalized to zero mean and unit variance before computing advantages. We set the KL penalty to $\beta = 0$ to isolate the pure reward-driven dynamics. Complete hyperparameters appear in Appendix L.

Synthetic verifier noise. $\mathrm { L e t } z \in \{ 0 , 1 \}$ denote the true correctness of a rollout under an oracle checker. The operational reward $r \in \{ 0 , 1 \}$ is produced by a noisy checker characterized by

$$
\operatorname{TPR} = \operatorname * {P r} (r = 1 \mid z = 1), \quad \operatorname{FPR} = \operatorname * {P r} (r = 1 \mid z = 0), \quad J = \operatorname{TPR} - \operatorname{FPR}.
$$

We implement noise by independently flipping the oracle outcome with Bernoulli trials:

$$
r = \left\{ \begin{array}{l l} 1 \text {   with   probability   TPR } & \text { if   } z = 1, \text {   else   } 0 \\ 1 \text {   with   probability   FPR } & \text { if   } z = 0, \text {   else   } 0. \end{array} \right.
$$

We explore a grid $\mathcal { I } \subset [ - 0 . 1 , 1 ]$ with multiple (TPR, FPR) factorizations for each target J to disentangle the effects of signal quality from error type prevalence.

Protocol. For each $J \in { \mathcal { I } } ,$ , we train the base model for two epochs (1,410 gradient steps), logging metrics every 5 steps. Each configuration is run five times with different random seeds; we report mean and standard deviation of pass@1 on the validation set. All other hyperparameters remain fixed across noise conditions.

Baseline. Our primary baseline is GRPO with the noise-free oracle $( J = 1 )$ , representing the performance ceiling under perfect verification.

## 7.3 Results

Table 1: Validation Accuracy & Noise Sensitivity. Validation accuracy after two epochs across noise conditions. Improvement is visualized relative to the Baseline model performance. Bars to the left indicate degradation; bars to the right indicate gain.

<table><tr><td>J</td><td>(FPR, FNR)</td><td>E[Pass@1]</td><td>Improvement from the Base model</td><td></td></tr><tr><td>-0.1</td><td>(0.60, 0.50)</td><td>0.16%</td><td>-12.6%</td><td></td></tr><tr><td>0.0</td><td>(0.50, 0.50)</td><td>13.40%</td><td>+0.6%</td><td>1</td></tr><tr><td>0.3</td><td>(0.00, 0.70)</td><td>16.00%</td><td>+3.2%</td><td></td></tr><tr><td>0.3</td><td>(0.70, 0.00)</td><td>14.60%</td><td>+1.8%</td><td></td></tr><tr><td>0.7</td><td>(0.20, 0.10)</td><td>18.6%</td><td>+5.8%</td><td></td></tr><tr><td>1.0</td><td>(0.00, 0.00)</td><td>20.8%</td><td>+8.0%</td><td></td></tr></table>

Phase transition confirmed $( \mathcal { H } _ { 1 } ) .$ . Figure 1 and Table 1 confirms a sharp qualitative boundary at $J = 0$ For $J > 0$ , all configurations show monotonic improvement in pass@1, with stronger signal (J closer to 1) yielding faster convergence and higher final accuracy. At the critical point $J = 0 ,$ , training produces minimal improvement $( + 0 . 6 \% )$ , consistent with neutral drift where reward noise cancels directional information. For $J < 0 ,$ , accuracy actively degrades (−12.6%), demonstrating anti-learning as predicted, the system systematically moves toward the bad equilibrium.

Noise rescales speed, not fate $( \mathcal { H } _ { 2 } )$ . Figure 1 shows learning trajectories across noise levels. For all $J > 0$ conditions, the accuracy curves increase monotonically over the training horizon. Our experiments are limited to 1410 steps (two epochs), so we remain agnostic about the exact asymptotic behavior; however, the observed trajectories are consistent with the hypothesis that both the noise–free and noisy regimes converge to the same basin of attraction.

The basin of attraction thus appears qualitatively unchanged: noisy signals with $J > 0$ still drive the system toward the good equilibrium, only at a reduced velocity. This aligns with our theoretical prediction that noise rescales the multiplicative factor in $\dot { p }$ but preserves the sign structure that governs the long–term dynamics.

Notably, even heavily degraded signals $( J = 0 . 3 )$ still enable learning, though convergence is substantially slower. The asymmetry between false positives and false negatives matters: at fixed $J = \stackrel { \smile } { 0 . 3 } .$ , the configuration (FPR=0.00, FNR=0.70) achieves 15.98% while (FPR=0.70, FNR=0.00) reaches $1 4 . { \dot { 6 } } 4 \% ,$ , suggesting FNs are more tolerable than FPs in this regime. This supports the theoretical investigation that convergence rate is $O ( t ^ { - 2 } )$ if $\mathrm { F N } = 0$ vs. $O ( t ^ { - 1 } )$ for $\mathrm { F N } > 0$

## 7.4 Limitations and Future Directions

Oracle imperfection. Our verification relies on a finite test suite. While designed for high coverage, incomplete tests introduce systematic bias in estimated (TPR, FPR), particularly for edge cases.

Context length effects. As model performance declines (especially for $J < 0 )$ , there is potential to increase the generation of longer responses that exceed the maximum token limit. VeRL’s handling of truncated rollouts, assigning them reward zero and high clipping ratios, introduces systematic false negatives that can shift the effective J downward. This may explain some of the asymmetry between predicted and observed behavior in the anti-learning regime.

Generalization. Our experiments focus on Python coding with Qwen2.5-3B. The phase transition at $J = 0$ is a fundamental property of the learning dynamics and should generalize across domains and architectures, but the specific decay rates and noise tolerance may vary with task complexity, model capacity, and verifier characteristics. Extensions to mathematical reasoning, creative writing with LLM-as-Judge, and larger models remain important future directions.

Time-dependent noise. Although our ODE framework is generalizable to arbitrary time-dependent noise, our experiments employ fixed noise rates. In practical settings, the simultaneous evolution of policy and reward models may induce drift in TPR(t) and FPR(t). While our mean-field analysis theoretically supports time-dependency, a full investigation of these co-evolutionary dynamics is reserved for future work.

## 8 Conclusion

We asked a simple but operational question: how much slop is too much in RLVR, when the verifier is imperfect and the learning algorithm repeatedly amplifies its feedback. Our analysis shows that, for group-normalized policy-gradient methods (e.g., GRPO), the qualitative outcome is governed by a single scalar:

$$
J = \mathrm{TPR-FPR}.
$$

When $J > 0 ,$ the verifier is net-informative and RL pushes probability mass toward correct solutions; when $J = 0 ,$ the signal is effectively chance-level and learning becomes neutral drift; when $J < 0$ , the signal is net-misleading and the updates become anti-learning, systematically driving the policy toward incorrect modes.

## MAIN FINDING

Group-normalized RL is directionally consistent under noisy rewards whenever $J = \mathrm { T P R } - \mathrm { F P R } >$ 0. In that regime, the aggregate bad-mode mass decreases monotonically and accuracy improves; noise primarily reduces the speed of progress rather than changing the eventual basin of attraction (“rate, not fate”). When $J < 0$ , the direction flips and performance collapses.

What this paper contributed. Beyond identifying the $J = 0$ boundary, we developed a minimal and predictive framework for noisy-reward RLVR:

• A multi-armed bandit abstraction for LLM completions. We coarse-grain sequences into recurring reasoning modes (arms), making sequence-level RLVR analytically tractable.

• A mean-field probability-simplex view of GRPO. Group normalization induces a replicator-style (natural-selection) flow that redistributes probability mass based on relative advantage.

• A closed-form governing variable: the bad-mode mass $p ( t )$ . The dynamics decouple into (i) an outer evolution of $p ( t )$ (good vs. bad families) and (ii) an inner competition within each family. This yields an explicit drift whose sign depends only on sign(J), producing the observed phase transition at $\dot { J } = 0$

• Rate laws and learnability insights. In the learning regime $( J > 0 )$ , noise changes convergence rates $( \mathrm { e . g . , } t ^ { - 1 } \mathrm { v s . } t ^ { - 2 }$ tails depending on variance degeneracy) and predicts that prompts are most learnable at intermediate difficulty (roughly when the model is near 50–50 between good and bad).

• Geometry and diversity implications. The simplex geometry (Shahshahani/Fisher) clarifies why GRPO produces winner-take-all behavior inside the good manifold (symmetry breaking / diversity collapse), even when multiple correct modes exist.

• Where practical GRPO details enter. PPO-style importance sampling and clipping do not change the leading-order drift in the small-step mean-field regime, while KL anchoring turns boundary collapse into a unique interior equilibrium, smoothing the sharpness of the transition without eliminating the fundamental dependence on J.

## PRACTICAL TAKEAWAYS FOR RLVR WITH NOISY VERIFIERS

• Measure (or estimate) J = TPR − FPR early. $\mathrm { I f } \ J \leq 0 ,$ , scaling RL compute will not fix the problem, it will stagnate or actively degrade the policy.

• If $J > 0 ,$ , compute helps (mostly) by buying time. Noisy-but-informative rewards tend to slow training rather than changing its qualitative endpoint.

• False positives are especially dangerous. Holding J fixed, the noise structure can change speed; empirically, high FPR is often more damaging than high FNR.

• Use KL regularization for stability, not as a substitute for signal quality. KL anchoring can prevent extreme collapse and yields controlled interior behavior, but it cannot turn a net-misleading verifier into a learning signal.

Closing perspective. Overall, RLV<sup>ε</sup>R provides a simple analytic lens for understanding when RLVR is viable under imperfect verification: the verifier’s net discriminative power (captured by J) determines the fate, while algorithmic details and noise structure shape the rate and stability. This gives both a concrete diagnostic for verifier quality and a principled foundation for designing more robust RLVR pipelines in domains where clean ground-truth supervision is unavailable.

## References

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet ´ Ust <sup>¨</sup> un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human¨ feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Shun-ichi Amari, Ryo Karakida, and Masafumi Oizumi. Fisher information and natural gradient learning in random deep networks. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 694–702. PMLR, 2019.

Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. arXiv preprint arXiv:2504.03380, 2025.

Yuntao Bai, Siddhant Kadavath, Shantanu Kundu, Amanda Askell, Jonathan Kernion, Andrew Jones, Cameron Chen, Michael Goldie, Ali Mirhoseini, and C. McKinnon. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022. URL https://arxiv.org/abs/2212.08073.

Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, and Masashi Sugiyama. Reinforcement learning with verifiable yet noisy rewards under imperfect verifiers. arXiv preprint arXiv:2510.00915, 2025. URL https://arxiv.org/abs/2510.00915.

Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025.

Ross Cressman. Evolutionary dynamics and extensiveform games, volume 5. MIT Press, 2003.

Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Assessing diversity collapse in reasoning. In Scaling Self-Improving Foundation Models without Human Supervision, 2025a.

Xingyu Dang, Christina Baek, Kaiyue Wen, Zico Kolter, and Aditi Raghunathan. Weight ensembling improves reasoning in language models. arXiv preprint arXiv:2504.10478, 2025b.

Thomas Foster, Anya Sims, Johannes Forkel, Mattie Fellows, and Jakob Foerster. Learning to reason at the frontier of learnability. arXiv preprint arXiv:2502.12272, 2025.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github.com/ huggingface/open-r1.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! 2019.

Julia Kreutzer, Artem Sokolov, and Stefan Riezler. Bandit structured prediction for neural sequence-tosequence learning. arXiv preprint arXiv:1704.06497, 2017.

Tor Lattimore and Csaba Szepesvari. ´ Bandit algorithms. Cambridge University Press, 2020.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Youssef Mroueh. Reinforcement learning with verifiable rewards: Grpo’s effective loss, dynamics, and success amplification. arXiv preprint arXiv:2503.06639, 2025.

Khanh Nguyen, Hal Daume III, and Jordan Boyd-Graber. Reinforcement learning for bandit neural machine´ translation with simulated human feedback. arXiv preprint arXiv:1707.07402, 2017.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv.org/abs/1707.06347.

Siavash Shahshahani. A new mathematical framework for the study of linkage and selection. American Mathematical Soc., 1979.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains. arXiv preprint arXiv:2503.23829, 2025. doi: 10.48550/arXiv.2503.23829. URL https://arxiv.org/abs/2503.23829.

Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, Jiang Bian, and Mao Yang. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025. URL https: //arxiv.org/abs/2506.14245.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

William J Youden. Index for rating diagnostic tests. Cancer, 3(1):32–35, 1950.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, and Dacheng Tao. Consistent paths lead to truth: Self-rewarding reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.08745

Zizhuo Zhang, Jianing Zhu, Xinmu Ge, Zihua Zhao, Zhanke Zhou, Xuan Li, Xiao Feng, Jiangchao Yao, and Bo Han. Co-rewarding: Stable self-supervised rl for eliciting reasoning in large language models. arXiv preprint arXiv:2508.00410, 2025b.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

Zhanke Zhou, Zhaocheng Zhu, Xuan Li, Mikhail Galkin, Xiao Feng, Sanmi Koyejo, Jian Tang, and Bo Han. Landscape of thoughts: Visualizing the reasoning process of large language models. arXiv preprint arXiv:2503.22165, 2025.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

## A LLM as Multi-arm Bandit

## A.1 Multi-armed bandits.

The multi-armed bandit (MAB) problem is a model in optimization and probability that focuses on the exploration–exploitation trade-off. In this problem setup, a decision maker repeatedly selects one of $K$ actions (“arms”); upon pulling arm $a _ { t } \in [ K ]$ at round $t ,$ a stochastic reward $R _ { t } ( \dot { a _ { t } } )$ is observed, drawn from an unknown distribution with mean $\mu _ { a _ { t } }$ . The goal is to maximize cumulative reward or equivalently minimize regret, despite this uncertainty:

$$
R _ {T} = T \mu_ {*} - \sum_ {t = 1} ^ {T} \mu_ {a _ {t}}, \quad \mu_ {*} := \max _ {a \in [ K ]} \mu_ {a}.
$$

This setting captures a wide range of real systems where feedback is noisy, delayed, or partial: online recommendation, $\mathtt { A } / \mathtt { B }$ testing, adaptive science experiments, and (as emphasized in this work) coarsegrained evaluation of generative models. Classical algorithms balance information gathering with reward maximization (e.g., optimism/UCB, posterior sampling, or gradient-based updates), and their guarantees hinge on the number of arms, reward signal quality, and the horizon T. In our context, the bandit abstraction serves as a tractable surrogate for complex, high-dimensional decision spaces while preserving the essential statistical structure of learning under uncertainty (Lattimore & Szepesvari´ (2020).

## A.2 Bandit Abstraction for LLMs

In the context of generative AI, specifically large language models, a given problem (such as a request or prompt) can yield multiple potential solutions, particularly when these models operate with a non-zero temperature setting. Recall that the temperature parameter influences output of the final layer of the model, where it directly affects the selection of subsequent tokens from the logit vectors using the softmax mechanism adjusted by the specified temperature. This selection process at the token level results in the generation of various sequences, some of which are correct (if the domain is verifiable), while others may be incorrect. Although the space of possible sequences that an LLM can generate is theoretically infinite akin to the hypothetical scenario of a monkey randomly typing and eventually producing a proof of the Goldbach conjecture, the total number of answers is finite due to the maximum response length that is feasible for the model to generate these answers.

For a fixed prompt $x ,$ an LLM samples a completion y from $\pi _ { \omega } ( y \mid x )$ . With nonzero temperature, the raw support over all token sequences can be very large (in principle, unbounded). In practice, inference and training impose a maximum generation length $\bar { L _ { \mathrm { m a x } } } ( \mathrm { e . g . }$ , max new tokens) and an end-of-sequence token ⟨eos⟩. Let $\mathcal { \dot { V } }$ denote the finite vocabulary. The admissible completions are then drawn from the truncated set

$$
\mathcal {Y} _ {\leq L _ {\max}} = \bigcup_ {\ell = 1} ^ {L _ {\max}} \mathcal {V} ^ {\ell}, \quad | \mathcal {Y} _ {\leq L _ {\max}} | \leq \sum_ {\ell = 1} ^ {L _ {\max}} | \mathcal {V} | ^ {\ell} = \frac {| \mathcal {V} | ^ {L _ {\max} + 1} - | \mathcal {V} |}{| \mathcal {V} | - 1},
$$

so the effective support isfinite. (Stop-sequences and ⟨eos⟩ further reduce this set in practice.) Given a fixed prompt x, a large language model (LLM) samples an output sequence y from its conditional policy $\pi _ { \omega } ( y | x )$ (with base parameters ω). In case of truncation, we can write the truncated policy as

$$
\pi_ {\omega} ^ {(L)} (y \mid x) \propto \pi_ {\omega} (y \mid x) \mathbf {1} \{y \in \mathcal {Y} _ {\leq L _ {\max}} \}.
$$

## A.3 Coarse-graining into Modes

For a nonzero sampling temperature, the model typically admits many distinct answers to the same prompt, often spanning a very large (potentially infinite) support. However, in the practice, due to the limitation on the output length (controlled by max tokens), the space of possible solution is practically coarsen into a finite collection of representatives reasoning modes (or solution prototypes). By clustering the reasoning-equivalent response together as a one reasoning mode/arm, we can map outputs via a surjective map

$$
\phi : \mathcal {Y} _ {\leq L _ {\max}} \longrightarrow \mathcal {H} = \{h _ {1}, \dots , h _ {K + M} \},
$$

where each mode $h \in { \mathcal { H } }$ represents a literal or semantic/evaluative equivalence class (e.g., logically equivalent answers, rubric-equivalent or string matching equivalency).

In the next step, we can partition the modes into good (correct) and bad (incorrect) solutions,

$$
\mathcal {H} = \mathcal {H} ^ {+} \cup \mathcal {H} ^ {-}, \quad | \mathcal {H} ^ {+} | = K, | \mathcal {H} ^ {-} | = M.
$$

Without loss of generality, we index good modes by $i \in \{ 1 , \ldots , K \}$ and bad modes by $i \in \{ K + 1 , \ldots , K +$ $M \}$ . Sampling a response is now equivalent to pulling one arm from a $( K { + } M )$ -armed bandit with pull probabilities $\check { \pi } _ { \boldsymbol { \theta } } ( h _ { i } | \boldsymbol { \hat { x } } )$ . We then work with the induced categorical policy over modes,

$$
\pi_ {\theta} (h _ {i} \mid x) = \frac {\exp (\theta_ {i})}{\sum_ {j = 1} ^ {K + M} \exp (\theta_ {j})} = \operatorname{softmax} (\theta) _ {i},
$$

where $\boldsymbol { \theta } = \left( \theta _ { 1 } , \ldots , \theta _ { K + M } \right)$ are effective logits that summarize, for the fixed prompt $x ,$ the aggregate probability mass the base model places on each mode. These logits are not a one-to-one reparameterization of ω; rather, they are low-dimensional coordinates (unique up to an additive constant) on the probability simplex over H induced by $\pi _ { \omega } ( \cdot | x )$

In this work, since we are interested mostly in the total probability of bad arms, as we discussed in the §4, we define the bad arms mass probability by partitioning modes into good (correct) and bad (incorrect), $\mathcal { H } = \mathcal { H } ^ { + } \cup \mathcal { H } ^ { - } , | \mathcal { H } ^ { + } | = K , | \mathcal { H } ^ { - } | \mathit { \bar { \Psi } } = M ,$ , such that $\begin{array} { r } { p \ = \ \sum _ { h \in { \mathcal H } ^ { - } } \pi _ { \theta } ( h \mid x ) } \end{array}$

Interior optimum (J=+0.8) Anti-informative (J=-0.3) Boundary p = 0 (J=+0.2) Uninformative (J=+0.0) Boundary p = 1 (J=+0.1)



Figure 6: Reward-variance geometry under noisy Bernoulli rewards. (Left) Reward variance ${ \mathrm { V a r } } ( r ) =$ $q ( p ) { \bigl ( } 1 - q ( p ) { \bigr ) }$ as a function of bad mass $p$ for representative noise settings; markers indicate $p ^ { \star } =$ $\mathbf { a r g m a x } _ { p \in [ 0 , 1 ] } \operatorname { V a r } ( r )$ (equivalently $\begin{array} { r } { q ( p ) = \frac { 1 } { 2 } } \end{array}$ when attainable, otherwise the boundary $p \in \{ 0 , 1 \} )$ ). (Middle) Heatmap of $p ^ { \star } ( \delta _ { \mathrm { F N } } , \delta _ { \mathrm { F P } } )$ in the informative region $J > 0 ,$ with the dashed diagonal marking the phase boundary $\dot { J } = 1 - \delta _ { \mathrm { F N } } - \delta _ { \mathrm { F P } } = 0$ and contours showing level sets of $p ^ { \star }$ . (Right) Maximum achievable reward standard deviation $\sigma _ { \mathrm { m a x } } ( \delta _ { \mathrm { F N } } , \delta _ { \mathrm { F P } } ) = \operatorname* { m a x } _ { p \in [ 0 , 1 ] } \sqrt { q ( p ) ( 1 - q ( p ) ) }$ with contours. Throughout, $q ( p ) = ( 1 - \delta _ { \mathrm { F N } } ) - J p$ and $J = 1 - \delta _ { \mathrm { F N } } - \delta _ { \mathrm { F P } }$

## B Noisy Rewards and Youden’s J Index

Recall the definition of noise that we had in equation 1:

$$
\delta_ {\mathrm{FN}} = \operatorname * {P r} (r = 0 \mid \text { good }), \quad \delta_ {\mathrm{FP}} = \operatorname * {P r} (r = 1 \mid \text { bad }),
$$

and the Youden’s Index, equation $^ { 2 , }$ as

$$
J := 1 - \delta_ {\mathrm{FN}} - \delta_ {\mathrm{FP}} = \text { TPR } - \text { FPR } \in [ - 1, 1 ].
$$

where $p = { \mathrm { P r } } ( { \mathrm { b a d } } )$ denote the current bad mass (so $\operatorname* { P r } ( \operatorname* { g o o d } ) = 1 - p )$ . With this setup, the expected reward of a single pull is

$$
\begin{array}{r l} q (p) & := \mathbb {E} [ r ] = \mathbb {E} [ r \mid \text {good} ]   (1 - p) + \mathbb {E} [ r \mid \text {bad} ]   p \\ & = (1 - p)   (1 - \delta_ {\text {FN}}) + p   \delta_ {\text {FP}} = (1 - \delta_ {\text {FN}}) - J   p. \end{array}\tag{20}
$$

Since $r$ is Bernoulli with mean $q ,$ its variance is

$$
\sigma^ {2} (p) := \operatorname{Var} (r) = q (p) (1 - q (p)).\tag{21}
$$

we can also directly verify this property:

$$
\begin{array}{r l} & {\mathrm{Var} (r) = (1 - p) (1 - \delta_ {\mathrm{FN}}) \delta_ {\mathrm{FN}} + p \delta_ {\mathrm{FP}} (1 - \delta_ {\mathrm{FP}}) + p (1 - p) J ^ {2}} \\ & {\qquad = ((1 - \delta_ {\mathrm{FN}}) - J p) (\delta_ {\mathrm{FN}} + J p) = q (1 - q).} \end{array}\tag{22}
$$

Notice that $q , \sigma ( p ) \equiv q , \sigma ( \delta _ { \mathrm { F N } } , \delta _ { F P } , p )$ , but for brevity, we denote it as $\sigma ( p )$ and $q ( p )$

It is good to notice that since $q ( p ) \in [ \delta _ { \mathrm { F P } } , 1 - \delta _ { \mathrm { F N } } ] .$ , the variance term $\sigma ( p ) = \sqrt { q ( p ) \big ( 1 - q ( p ) \big ) }$ is maximized at $\begin{array} { r } { q ( p ) = \frac { 1 } { 2 } } \end{array}$ , provided $\frac 1 2$ lies inside this interval. Solving $\begin{array} { r } { q ( p ) = \frac { 1 } { 2 } } \end{array}$ for p yields

$$
p ^ {\star} = \mathrm{clip} (\frac {\frac {1}{2} - \delta_ {\mathrm{FN}}}{J}, 0, 1)
$$

such that the value of $p$ that maximizes $\sigma ( p )$ , assuming $J = 1 - \delta _ { \mathrm { F N } } - \delta _ { \mathrm { F P } } > 0$ . In the edge case where $\begin{array} { r } { \frac { 1 } { 2 } \notin \left[ \delta _ { \mathrm { F P } } , 1 - \delta _ { \mathrm { F N } } \right] ( \mathrm { i . e . } } \end{array}$ , for extremely noisy graders), the maximum of $\sigma ( p )$ occurs at the boundary: $q = \delta _ { \mathrm { F P } }$ or $q = 1 - \delta _ { \mathrm { F N } } ,$ whichever is closer to $\frac { 1 } { 2 }$ . Equivalently, $p ^ { \star }$ clips to 0 or 1 in this regime (see Fig. 6).

Group-based policy typically updates normalized rewards within a prompt-specific group of G rollouts. While alternatives exist $( \mathrm { e . g . }$ , leave-one-out baselines Kool et al. (2019) or centered-but-unstandardized variants Liu et al. (2025)), we adopt a simple z-score normalization (as in GRPO Shao et al. (2024)):

$$
\tilde {r} = \frac {r - q (p)}{\sigma (p)}.\tag{23}
$$

Conditioning on the latent correctness, this yields

$$
\tilde {r} \mid \text {good} = \left\{ \begin{array}{l l} \frac {1 - q}{\sigma}, & \text {w.p.} 1 - \delta_ {\mathrm{FN}}, \\ \frac {- q}{\sigma}, & \text {w.p.} \delta_ {\mathrm{FN}}, \end{array} \right. \quad \tilde {r} \mid \text {bad} = \left\{ \begin{array}{l l} \frac {1 - q}{\sigma}, & \text {w.p.} \delta_ {\mathrm{FP}}, \\ \frac {- q}{\sigma}, & \text {w.p.} 1 - \delta_ {\mathrm{FP}}. \end{array} \right.
$$

Taking expectations gives the block-symmetric conditional means

$$
\mathbb {E} [ \tilde {r} \mid \text { good } ] = \frac {J   p}{\sigma (p)}, \qquad \mathbb {E} [ \tilde {r} \mid \text { bad } ] = - \frac {J   (1 - p)}{\sigma (p)},\tag{24}
$$

and global centering holds automatically:

$$
\mathbb {E} [ \tilde {r} ] = (1 - p) \mathbb {E} [ \tilde {r} | \mathrm{good} ] + p \mathbb {E} [ \tilde {r} | \mathrm{bad} ] = 0,\tag{25}
$$

which is desirable for stable, scale-invariant updates. These expressions demonstrate that Youden’s index J governs the sign and magnitude of the expected normalized reward for good versus bad arms.

Remark B.1. If one omits division by $\sigma ( p )$ (the “centered-only” modification of GRPO Liu et al. (2025)), then

$$
\mathbb {E} [ \tilde {r} \mid \mathrm{good} ] = J p, \qquad \mathbb {E} [ \tilde {r} \mid \mathrm{bad} ] = - J (1 - p), \qquad \mathbb {E} [ \tilde {r} ] = 0.\tag{26}
$$

Remark B.2. Some works use a {±1}-valued reward $S : = 2 r - 1$ . Then $\mathbb { E } [ S ] = 2 q - 1$ and $\operatorname { V a r } ( S ) = 4 q ( 1 - q )$ Equations equation 23–equation 24 map to this reward by the linear rescaling $S = 2 r - \mathrm { \dot { 1 } } ;$ ; the resulting normalization differs only by a constant factor of 2.

Remark B.3. For noise-free case, $J = 1$ , the expectation values take a simpler form

$$
\mathbb {E} [ \tilde {r} \mid \text {good} ] = \sqrt {\frac {p}{1 - p}}, \qquad \mathbb {E} [ \tilde {r} \mid \text {bad} ] = \sqrt {\frac {1 - p}{p}}, \qquad \mathbb {E} [ \tilde {r} \mid \text {good} ] - \mathbb {E} [ \tilde {r} \mid \text {bad} ] = \frac {1}{\sqrt {p (1 - p)}}.
$$

## C Mean Field Dynamics..

In this section, we analyze the evolution of good and bad arms using a mean-field approximation of the multiarmed bandit (MAB) system (see Appendix A). Our derivation accounts for a general noisy environment where $J \in [ - 1 , 1 ]$ . The noise-free scenario is treated as a specialization of this framework; specifically, by setting $J = 1$ , we recover the standard clean-reward dynamics without requiring further modification.

## C.1 Dynamics of the Bad Arms

Consider a (K+M)-arm bandit comprising K good arms and M bad arms $\left\{ b _ { 1 } , \dotsc , b _ { M } \right\}$ . Let the policy be defined as $\mathbf { p } = { \mathrm { s o f t m a x } } ( \mathbf { \theta } ) \in \Delta ^ { K + M - 1 }$ where:

$$
\mathbf {p} = (p _ {1}, \dots , p _ {K}, p _ {b _ {1}}, \dots , p _ {b _ {M}}), \qquad p := \sum_ {m = 1} ^ {M} p _ {b _ {m}} \in [ 0, 1 ], \qquad \alpha := 1 - p,
$$

and we define the normalized within-block coordinates:

$$
p _ {j} = \alpha   y _ {j} (j \leq K), \quad y = (y _ {1}, \ldots , y _ {K}) \in \Delta^ {K - 1}, \qquad p _ {b _ {m}} = p   z _ {m} (m \leq M), \quad z = (z _ {1}, \ldots , z _ {M}) \in \Delta^ {M - 1}.
$$

Equivalently, the probability vector factors as follows:

$$
\mathbf {p} = \left[ \begin{array}{c} \alpha   y \\ p   z \end{array} \right], \qquad \| y \| _ {2} ^ {2} = \sum_ {j = 1} ^ {K} y _ {j} ^ {2} \in \Big [ \frac {1}{K}, 1 \Big ], \qquad \| z \| _ {2} ^ {2} = \sum_ {m = 1} ^ {M} z _ {m} ^ {2} \in \Big [ \frac {1}{M}, 1 \Big ].
$$

Let the conditional expected advantage reward be denoted by $\begin{array} { r } { \begin{array} { r c l } { A _ { i } } & { : = } & { \mathbb { E } [ \tilde { r } } \end{array} | \quad I \ = \ i ] } \end{array}$ , with ${ \bf A } \ =$ $( A _ { 1 } , \dots , A _ { K } , A _ { b _ { 1 } } , \dots , A _ { b _ { M } } )$ and the mean advantage ${ \bar { A } } : = \langle \mathbf { p } , \mathbf { A } \rangle = \sum _ { i } p _ { i } A _ { i }$ . We assume block symmetry within both blocks, a condition that arises naturally in the binary-reward scenario analyzed in Appendix B (refer to equation 24):

$$
A _ {j} = a _ {\mathrm{g}} (p) (j \leq K), \quad A _ {b _ {m}} = a _ {\mathrm{b}} (p) (m \leq M), \quad \Delta r (p) := a _ {\mathrm{b}} (p) - a _ {\mathrm{g}} (p).\tag{27}
$$

It follows that $\bar { A } = \alpha a _ { \mathrm { g } } ( p ) + p a _ { \mathrm { b } } ( p )$ , which implies $a _ { \mathrm { g } } ( p ) - \bar { A } = - p \Delta r ( p )$ and $a _ { \mathrm { b } } ( p ) - \bar { A } = \alpha \Delta r ( p )$

Proposition C.1 (Expected directions in θ and p space). Given $\mathfrak { p } = \mathrm { s o f t m a x } ( \pmb { \theta } )$ and the softmax Jacobian $\Im ( { \bf p } ) : = \mathrm { D i a g } ( { \bf p } ) - { \bf p } { \bf \bar { p } } ^ { \top }$ , for a step size η and group size G:

$$
\mathbb {E} [ \Delta \pmb {\theta} | \pmb {p} ] = \eta \mathfrak {J} (\pmb {p}) \mathbf {A}, \qquad \mathbb {E} [ \Delta \pmb {p} | \pmb {p} ] \approx \mathfrak {J} (\pmb {p}) \mathbb {E} [ \Delta \pmb {\theta} | \pmb {p} ] = \eta \mathfrak {J} (\pmb {p}) ^ {2} \mathbf {A},
$$

to the first order in ∆θ.

Proof Sketch. Let $e _ { i }$ denote the i-th standard basis vector. Define $\pi _ { \boldsymbol { \theta } } ( i ) = p _ { i } = \exp ( \theta _ { i } ) / \sum _ { k } \exp ( \theta _ { k } )$ as the softmax policy. For a realized arm $I ,$ the gradient is:

$$
\nabla_ {\boldsymbol {\theta}} \log \pi_ {\boldsymbol {\theta}} (I) = e _ {I} - \mathbf {p}.
$$

The REINFORCE estimator (Williams (1992)) for $\nabla _ { \pmb { \theta } } \mathbb { E } [ r ] \mathrm { i } \mathbf { s } g = \widetilde { r } \left( e _ { I } - \mathbf { p } \right)$ . Taking the conditional expectation given p yields:

$$
\mathbb {E} [ g \mid \mathbf {p} ] = \sum_ {i} p _ {i} \mathbb {E} [ \tilde {r} \mid I = i ] (e _ {i} - \mathbf {p}) = (\operatorname{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top}) \mathbf {A} = \mathfrak {J} (\mathbf {p}) \mathbf {A}.
$$

This confirms the stated form and the first identity in Proposition C.1. For the second part, refer to Lemma H.1.

Remark C.2 (The Importance of Coupling Terms). Retaining the full Jacobian, including the rank-one term $\mathsf { p } \mathsf { p } ^ { \top }$ , is essential because it couples all arms through collision terms. Specifically, the total bad-mass drift depends on the collisions within both the good and bad blocks via $\| y \| _ { 2 } ^ { 2 }$ and $\| z \| _ { 2 } ^ { 2 }$ (see equation 34). Omitting the $\mathbf { p } \mathbf { p } ^ { \top }$ term would spuriously decouple the blocks and result in incomplete mean-field dynamics.

Corollary C.3 (First-order softmax pushforward). For a small logit increment $\Delta \theta \mathrm { : }$

$$
\Delta \mathbf {p} = \Im (\mathbf {p}) \Delta \boldsymbol {\theta} = \operatorname{Diag} (\mathbf {p}) \Delta \boldsymbol {\theta} - \mathbf {p} \mu , \quad \mu := \langle \mathbf {p}, \Delta \boldsymbol {\theta} \rangle = \sum_ {k} p _ {k} \Delta \theta_ {k},
$$

which implies $\Delta p _ { i } = p _ { i } \big ( \Delta \theta _ { i } - \mu \big )$

Applying Proposition C.1 and the relation for ${ \bar { A } } ,$ we obtain the following expectations (where conditioning on p is suppressed for brevity):

$$
\mathbb {E} [ \Delta \theta_ {j} ] = - \eta p (1 - p) \Delta r (p) y _ {j}, \qquad j = 1, \ldots , K,\tag{28}
$$

$$
\mathbb {E} [ \Delta \theta_ {b _ {m}} ] = \eta p (1 - p) \Delta r (p) z _ {m}, \qquad m = 1, \ldots , M.\tag{29}
$$

The expected step therefore follows the block-form direction:

$$
\mathbb {E} [ \Delta \pmb {\theta} ] = \eta p (1 - p) \Delta r (p) \left[ \begin{array}{c} - y \\ z \end{array} \right].\tag{30}
$$

Since $\Im ( \mathbf { p } ) \mathbf { 1 } = 0 ,$ the update is centered:

$$
\sum_ {i} \mathbb {E} [ \Delta \theta_ {i} ] = \eta   \mathbf {1} ^ {\top} \mathfrak {J} (\mathbf {p}) \mathbf {A} = 0.\tag{31}
$$

Moreover, within each block the logit increment is collinear with the current within-block distribution:

$$
\mathbb {E} [ \Delta \theta_ {j} ] - y _ {j} \sum_ {k = 1} ^ {K} \mathbb {E} [ \Delta \theta_ {k} ] = 0, \qquad \mathbb {E} [ \Delta \theta_ {b _ {m}} ] - z _ {m} \sum_ {\ell = 1} ^ {M} \mathbb {E} [ \Delta \theta_ {b _ {\ell}} ] = 0.
$$

In other words, there is no arm-specific drift within a block in logit space; arms move in lockstep proportional to y (good block) and z (bad block).

Following equation 24, the expected advantages relative to the noise level $J = \mathrm { T P R } - \mathrm { F P R }$ are expressed as:

$$
a _ {\mathrm{g}} (p) = \frac {J p}{\sigma (p)}, \qquad a _ {\mathrm{b}} (p) = - \frac {J (1 - p)}{\sigma (p)}, \qquad \Delta r (p) = - \frac {J}{\sigma (p)}.\tag{32}
$$

Total Bad-Mass Drift By Corollary C.3, the softmax-centering scalar µ becomes:

$$
\mu = \eta p (1 - p) \Delta r (p) \left(p \| z \| _ {2} ^ {2} - (1 - p) \| y \| _ {2} ^ {2}\right).\tag{33}
$$

Summing the bad components provides the total bad-mass drift:

$$
\mathbb {E} [ \Delta p ] = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} \left(\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}\right).\tag{34}
$$

In the case where $M = 1$ , then $z = ( 1 )$ and $\| z \| _ { 2 } ^ { 2 } = 1$ , which recovers the (K+1) formula.

Within-Bad Dynamics in Normalized Coordinates Using the identity $\Delta z _ { m } = { \textstyle \frac { 1 } { p } } ( \Delta p _ { b _ { m } } - z _ { m } \Delta p )$ and substituting the first-order drift, we find:

$$
\mathbb {E} [ \Delta z _ {m} ] = - \eta \frac {J}{\sigma (p)} p (1 - p) z _ {m} \Big (z _ {m} - \| z \| _ {2} ^ {2} \Big), \qquad m = 1, \ldots , M.\tag{35}
$$

In vector form, this is expressed as ${ \mathbb E } [ \Delta z ] = \eta \ p ( 1 - p ) \Delta r ( p ) \left( z \odot z - \| z \| _ { 2 } ^ { 2 } z \right)$ . Consequently, for an informative grader $( J > 0 )$ , the bad-block dynamics exhibit the opposite sign of the good-block collision field, tending to spread bad mass toward a uniform distribution on $\hat { \Delta } ^ { M - 1 }$

## C.2 Dynamics of the Good Arms

Regarding the good arms, the combination of Corollary C.3 with equation 28 and equation 33 provides the probability increments for the good block:

$$
\mathbb {E} \left[ \Delta \mathbf {p} _ {\text { good }} \right] = - \eta p (1 - p) ^ {2} \Delta r (p) \left(y \odot y + [ p \| z \| _ {2} ^ {2} - (1 - p) \| y \| _ {2} ^ {2} ] y\right).\tag{36}
$$

In componentwise form, substituting $p _ { j } = ( 1 - p ) y _ { j } .$ , we obtain:

$$
\begin{array}{r l} & {\mathbb {E} [ \Delta p _ {j} ] = (1 - p) y _ {j} \Big (\mathbb {E} [ \Delta \theta_ {j} ] - \mu \Big)} \\ & {\qquad = - \eta p (1 - p) ^ {2} \Delta r (p) y _ {j} \Big (y _ {j} + p \| z \| _ {2} ^ {2} - (1 - p) \| y \| _ {2} ^ {2} \Big), \qquad j = 1, \ldots , K.} \end{array}\tag{37}
$$

Summing equation 37 over all j and applying the constraint $\begin{array} { r } { \sum _ { j } y _ { j } = 1 } \end{array}$ yields:

$$
\sum_ {j = 1} ^ {K} \mathbb {E} [ \Delta p _ {j} ] = - \eta [ p (1 - p) ] ^ {2} \Delta r (p) \left(\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}\right) = - \mathbb {E} [ \Delta p ].
$$

Consequently, the drift for the total good mass $\alpha : = 1 - p$ is given by:

$$
\mathbb {E} [ \Delta \alpha ] = - \mathbb {E} [ \Delta p ] = - \eta [ p (1 - p) ] ^ {2} \Delta r (p) (\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}).\tag{38}
$$

By utilizing the relationship $y _ { j } = p _ { j } / \alpha ,$ , we can apply the exact identity:

$$
\Delta y _ {j} = \frac {1}{\alpha} \Big (\Delta p _ {j} - y _ {j} \Delta \alpha \Big).\tag{39}
$$

Substituting equation 37 through equation 38 into equation 39 and simplifying leads to the within-good drift:

$$
\mathbb {E} [ \Delta y _ {j} ] = - \eta p \alpha \Delta r (p) y _ {j} \left(y _ {j} - \| y \| _ {2} ^ {2}\right), \quad j = 1, \dots , K.\tag{40}
$$

In vector form, this is expressed as:

$$
\mathbb {E} [ \Delta y ] = - \eta p (1 - p) \Delta r (p) \left(y \odot y - \| y \| _ {2} ^ {2} y\right).\tag{41}
$$

Notably, $\begin{array} { r } { \sum _ { j } \mathbb { E } [ \Delta y _ { j } ] = 0 . } \end{array}$ , which confirms that the simplex remains invariant as expected. The fixed points of equation 41 are located at the barycenter and the vertices. When $\Delta r ( p ) < 0 ,$ a condition signifying an informative grader that favors good arms over bad arms, the uniform point becomes unstable and the vertices act as attractors.

Substituting $\Delta r ( p ) = - J / \sigma ( p )$ from equation 32 into equation 40 results in:

$$
\mathbb {E} [ \Delta y _ {j} ] = \eta \frac {J}{\sigma (p)} p (1 - p) y _ {j} \left(y _ {j} - \| y \| _ {2} ^ {2}\right).
$$

For $J > 0 ,$ , arms where $y _ { j } > \| y \| _ { 2 } ^ { 2 }$ will grow while those where $y _ { j } < \| y \| _ { 2 } ^ { 2 }$ shrink, representing a deterministic sharpening within the good block.

## C.3 From Expectation-Based Updates to ODEs: The Small-Step Bridge

Consider an expectation-level logit update:

$$
\boldsymbol {\theta} ^ {(t + 1)} = \boldsymbol {\theta} ^ {(t)} + \eta g (\mathbf {p} ^ {(t)}), \quad \Delta \boldsymbol {\theta} = \eta g (\mathbf {p}),
$$

where g represents the per-step expected gradient. Through the softmax mapping $\mathfrak { p } = \mathrm { s o f t m a x } ( \pmb { \theta } )$ , a small logit update is defined as $\Delta { \bf p } = \Im ( { \bf p } ) \Delta \theta$ , with $\Im ( { \bf p } ) \ = \ \mathrm { D i a g } ( { \bf p } ) - { \bf p } { \bf p } ^ { \top }$ . Substituting $\Delta \theta = \eta g ( { \bf p } )$ yields the expected increment for the policy:

$$
\Delta \mathbf {p} = \eta (\mathrm{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top}) g (\mathbf {p}) + O (\eta^ {2}).
$$

Option 1: Unit Time per Iteration We may treat the iteration index itself as continuous time. Let $t \in \mathbb { R }$ denote a continuous extension of the discrete counter, where a single algorithmic update corresponds to a unit time step $\Delta t = 1$ . By defining $\mathbf { p } ( t ) \approx \mathbf { p } ^ { ( t ) }$ , the relationship is:

$$
\frac {\mathbf {p} ^ {(t + 1)} - \mathbf {p} ^ {(t)}}{\Delta t} \approx \frac {d \mathbf {p}}{d t} (t) = \dot {\mathbf {p}} (t).
$$

Aligning this with the discrete update $\mathbf { p } ^ { ( t + 1 ) } - \mathbf { p } ^ { ( t ) } = \Delta \mathbf { p }$ results in the following ordinary differential equation (ODE):

$$
\dot {\mathbf {p}} (t) = \eta (\operatorname{Diag} (\mathbf {p} (t)) - \mathbf {p} (t) \mathbf {p} (t) ^ {\top}) g (\mathbf {p} (t)).\tag{42}
$$

The expectation-level GRPO update thus serves as a forward-Euler discretization of the continuous-time dynamics in equation 42 with a unit step size.

This ODE provides an accurate proxy within the small-learning-rate regime. The local truncation error of the Euler step satisfies $\lVert \mathbf { p } ^ { + } - \mathbf { p } - \dot { \mathbf { p } } \rVert = O ( \eta ^ { 2 } )$ , and given that max<sub>a</sub> η $\mid g _ { a } ( \mathbf { p } ) \mid \ll 1$ , no coordinate of p shifts excessively in a single iteration. Geometrically, equation 42 remains a natural-gradient (Shahshahani) flow:

$$
\dot {\mathbf {p}} = \eta \mathbf {G} (\mathbf {p}) \nabla_ {\theta} \mathcal {L}, \quad \mathbf {G} (\mathbf {p}) = \operatorname{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top},
$$

where $\mathbf { G } ( \mathbf { p } )$ represents the Fisher metric tensor on the simplex. The factor η scales the velocity along this geometric flow. By approximating discrete differences with derivatives, equation 41 and equation 42 transform into the coupled ODEs outlined in equation 12:

$$
\dot {y} = \kappa (p) \left(y \odot y - \| y \| _ {2} ^ {2} y\right),
$$

$$
\dot {z} = - \kappa (p) \left(z \odot z - \| z \| _ {2} ^ {2} z\right),
$$

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} \left(\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}\right),
$$

where we define the proportionality factor $\begin{array} { r } { \kappa ( p ) : = \eta \frac { J } { \sigma ( p ) } p ( 1 - p ) } \end{array}$

Option 2: Alternative Time Rescaling An alternative approach involves absorbing the learning rate directly into the time variable. By defining a rescaled time $\mathrm { t } = \eta t \mathrm { . }$ , each discrete update advances t by $\Delta \mathfrak { t } = \eta$ Using the chain rule for $\mathbf { p } ( \mathrm { t } ) : = \mathbf { \dot { p } } ( t )$ , we find:

$$
\frac {d \mathbf {p}}{d t} = \frac {1}{\eta} \frac {d \mathbf {p}}{d t} = \left(\operatorname{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top}\right) g (\mathbf {p}),
$$

which simplifies equation 42 to:

$$
\frac {d \mathbf {p}}{d t} = \left(\operatorname{Diag} (\mathbf {p} (t)) - \mathbf {p} (t) \mathbf {p} (t) ^ {\top}\right) g (\mathbf {p} (t)).\tag{44}
$$

This represents the standard gradient-flow limit.

Remark C.4. While the trajectories in policy space remain identical across both time parametrizations, this work utilizes the unit time per iteration notation to maintain the visibility of mean-field correction terms as they relate to η.



Noiseless (J=+1.0) FP-heavy (J=+0.6) Symmetric noise (J=+0.6) Near-uninformative (J=+0.1) FN-heavy (J=+0.6)

Figure 7: Learnability-maximizing bad mass $p ^ { \dagger }$ under group z-scored rewards. We plot the instantaneous learnability speed $\begin{array} { r } { \mathcal { L } ( p ) = \frac { J } { \sigma ( p ) } [ p ( 1 - p ) ] ^ { 2 } } \end{array}$ , which controls the magnitude of the one-step bad-mass drift $| \Delta p | \propto \mathcal { L } ( p )$ (up to a positive constant), where $\sigma ( p ) = \sqrt { q ( p ) ( 1 - q ( p ) ) } , q ( p ) = ( 1 - \delta _ { \mathrm { F N } } ) - J p ,$ and $J = 1 - \delta _ { \mathrm { F N } } - \delta _ { \mathrm { F P } }$ . (A) Curves of $\mathcal { L } ( p )$ versus $p$ for representative noise settings; markers indicate $p ^ { \dagger } =$ arg $\mathrm { { \ m a x } } _ { p \in [ 0 , 1 ] } \mathcal { L } ( p )$ . (B) Heatmap of $p ^ { \dagger } \big ( \delta _ { \mathrm { F N } } , \delta _ { \mathrm { F P } } \big )$ in the informative region $J > 0$ (masked for $J \ \leq \ 0 ) ;$ the dashed diagonal marks $J = 0$ and contours indicate level sets of $p ^ { \dagger }$ . (C) Heatmap of the maximum instantaneous learnability ma $\mathfrak { c } _ { p } \mathcal { L } ( p )$ , showing how noise reduces the peak achievable drift. Symmetric noise $( \delta _ { \mathrm { F N } } = \delta _ { \mathrm { F P } } )$ preserves the midpoint optimum $\begin{array} { r } { p ^ { \dagger } = \frac { 1 } { 2 } } \end{array}$ , while asymmetric noise shifts the maximizer away from $\frac { 1 } { 2 }$ by reweighting the signal through $\sigma ( p ) ^ { - 1 }$

## D Maximal Learnability

We quantify a prompt’s learnability by the instantaneous rate at which GRPO reduces its latent bad-mode mass $p = { \dot { \operatorname* { P r } } } ( { \dot { \mathsf { b a d } } } \mid x )$ . Under the block-symmetric mean-field approximation derived in Appendix C (see equation 34, the (unregularized) one-step drift of p takes the form)

$$
| \Delta p | \propto \Delta (p) [ p (1 - p) ] ^ {2}, \quad \Delta (p) := \mathbb {E} [ \tilde {r} | \text {good} ] - \mathbb {E} [ \tilde {r} | \text {bad} ],\tag{45}
$$

up to an overall positive step-size constant and smooth factors that vary slowly with p. Here r˜ denotes the group-normalized reward.

Normalized separation under noisy rewards. With z-score normalization equation 23, the conditional means equation 24 imply a simple closed form for the separation in normalized units:

$$
\Delta (p) = \frac {J}{\sigma (p)}, \qquad J = 1 - \delta_ {\mathrm{FN}} - \delta_ {\mathrm{FP}}, \qquad \sigma (p) = \sqrt {q (p) (1 - q (p))}, \qquad q (p) = (1 - \delta_ {\mathrm{FN}}) - J p.\tag{46}
$$

Consequently, the learnability speed (i.e., the p-dependent component of the drift) is

$$
\mathcal {L} (p; \delta_ {\mathrm{FN}}, \delta_ {\mathrm{FP}}) := \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2}, \quad \text {so that} \quad | \Delta p | \propto \mathcal {L} (p; \delta_ {\mathrm{FN}}, \delta_ {\mathrm{FP}}).\tag{47}
$$

Throughout this section we focus on the informative regime $J > 0$ (the grader is better than random). When $J = 0$ the signal vanishes, and when $J < 0$ it is anti-informative and must be corrected (Remark D.1).

Noiseless case (J = 1). When $\delta _ { \mathrm { F N } } = \delta _ { \mathrm { F P } } = 0 .$ we have $q ( p ) = 1 - p$ and $\sigma ( p ) = { \sqrt { p ( 1 - p ) } }$ , hence

$$
\mathcal {L} (p; 0, 0) = \frac {1}{\sqrt {p (1 - p)}} [ p (1 - p) ] ^ {2} = [ p (1 - p) ] ^ {3 / 2}.\tag{48}
$$

This is maximized at $\begin{array} { r } { p ^ { \star } = ~ \frac { 1 } { \widehat { \varsigma } } } \end{array}$ . Thus, the largest single-step reduction in bad mass occurs on “mediumdifficulty” prompts where the model is roughly 50–50 between good and bad solutions. At the extremes $p  0$ (almost always good) or $p  1$ (almost always bad), the factor $p ( 1 - p )$ vanishes and learning slows down sharply: additional GRPO steps make only marginal progress on highly saturated prompts.

Noisy grading: what changes. With noise, the learnability speed becomes

$$
\mathcal {L} (p; \delta_ {\mathrm{FN}}, \delta_ {\mathrm{FP}}) = \frac {J}{\sqrt {q (p) (1 - q (p))}} [ p (1 - p) ] ^ {2}, \quad q (p) = (1 - \delta_ {\mathrm{FN}}) - J p.\tag{49}
$$

This expression highlights two distinct effects:

1. Global shrinkage via J. Increasing noise decreases $J = 1 - \delta _ { \mathrm { F N } } - \delta _ { \mathrm { F P } }$ , uniformly reducing learnability. In the limit $\bar { \delta } _ { \mathrm { F N } } + \delta _ { \mathrm { F P } }  1$ , the grader becomes uninformative and $\mathcal { L } \to 0$ for all $p .$ .

2. Reweighting via $\sigma ( p ) ^ { - 1 }$ . Group z-scoring divides by the reward standard deviation $\sigma ( p )$ , so the effective learning signal is amplified when the reward distribution is highly concentrated (small $\sigma )$ and attenuated when it is maximally noisy (large σ). Equivalently, the normalized separation is $\Delta ( p ) = J / \sigma ( p )$

Symmetric noise. If $\delta _ { \mathrm { F N } } = \delta _ { \mathrm { F P } } = \delta ,$ then $J = 1 - 2 \delta$ and $q ( 1 - p ) = 1 - q ( p )$ , implying $\sigma ( 1 - p ) = \sigma ( p )$ . Since $p ( 1 - p )$ is also symmetric, $\mathcal { L } ( 1 - p ) = \mathcal { L } ( p )$ , and the maximizer remains at the symmetry point $\begin{array} { r } { p ^ { \star } = \frac { 1 } { 2 } } \end{array}$ Moreover, $\begin{array} { r } { q ( \frac { 1 } { 2 } ) = \frac { 1 } { 2 } } \end{array}$ and $\begin{array} { r } { \sigma ( \frac { 1 } { 2 } ) = \frac { 1 } { 2 } . } \end{array}$ , yielding the explicit peak value

$$
\mathcal {L} _ {\max} (\delta , \delta) = \mathcal {L} \left(\frac {1}{2}; \delta , \delta\right) = \frac {J}{1 / 2} \left(\frac {1}{4}\right) ^ {2} = \frac {1 - 2 \delta}{8}.\tag{50}
$$

Hence symmetric noise does not shift the most-learnable difficulty, but it reduces the maximal attainable learning speed, collapsing to zero as $\delta  { \frac { 1 } { 2 } }$

Asymmetric noise. When $\delta _ { \mathrm { F N } } \neq \delta _ { \mathrm { F P } }$ , the symmetry $p  1 - p$ is broken and the maximizer shifts to an interior point $p ^ { \dag } \in ( 0 , 1 )$ that balances the mixture factor $[ p ( 1 - p ) ] ^ { 2 }$ against the normalization term $\sigma ( p )$ Differentiating log $\mathcal { L }$ yields the stationarity condition

$$
p ^ {\dagger} \text {solves} 2 \frac {1 - 2 p}{p (1 - p)} + \frac {J}{2} \frac {1 - 2 q (p)}{q (p) (1 - q (p))} = 0, \quad q (p) = (1 - \delta_ {\mathrm{FN}}) - J p,\tag{51}
$$

with boundary clipping if no interior maximizer exists. Intuitively, the term $[ p ( 1 - p ) ] ^ { 2 }$ favors intermediate difficulty, while the factor $1 / \sigma ( p )$ reweights the signal in a way that can skew the optimum when false negatives and false positives are imbalanced. Equation equation 51 admits an explicit solution but not a simple elementary one in general: after substituting $q ( p ) \\\\\\\\r = \left( 1 - \delta _ { \mathrm { F N } } \right) - J p$ and clearing denominators, the condition reduces to a cubic polynomial in $p .$ . Thus $p ^ { \dagger }$ can be written in closed form via Cardano’s formula, although the expression is cumbersome; in practice, we select the real root in $p \in ( 0 , 1 )$ (or clip to $\{ 0 , 1 \}$ if the maximizer lies on the boundary). A notable simplification occurs under symmetric noise $\bar { \delta _ { \mathrm { F N } } } = \bar { \delta _ { \mathrm { F P } } } ,$ where the invariance $\mathcal { L } ( 1 - p ) = \mathcal { L } ( p )$ forces $\begin{array} { r } { p ^ { \dagger } = \frac { 1 } { 2 } } \end{array}$ . Fig. 7 shows p† for full range of noises in the learning phase. Remark D.1 (Uninformative or adversarial graders). If $J = 0 \left( \mathrm { i . e . , } \delta _ { \mathrm { F N } } + \delta _ { \mathrm { F P } } = 1 \right)$ , then $\Delta ( p ) = J / \sigma ( p ) = 0$ and the mean update provides no systematic signal to reduce bad mass. If $J < 0$ , the grader is worse than random and $\Delta ( \hat { p } )$ flips sign; equivalently, swapping labels $r \mapsto 1 - r$ restores an effective $J ^ { \prime } > 0$ and recovers the analysis above.

## E Lyapunov analysis and the role of J

We consider a bandit configuration comprising K good arms and M bad arms, , as discussed in Appendix. C, defined by the probability vector:

$$
\mathbf {p} = (p _ {1}, \dots , p _ {K}, p _ {b _ {1}}, \dots , p _ {b _ {m}}) \in \Delta^ {K + M - 1}, \quad \sum_ {j = 1} ^ {K} p _ {j} + \sum_ {m = 1} ^ {M} p _ {b _ {m}} = 1.
$$

We define the total mass of the good and bad blocks respectively as:

$$
s (\mathbf {p}) := \sum_ {j = 1} ^ {K} p _ {j} \in [ 0, 1 ], \quad P _ {\text { bad }} (\mathbf {p}) := \sum_ {m = 1} ^ {M} p _ {b _ {m}} = 1 - s (\mathbf {p}), \quad \Im (\mathbf {p}) = \operatorname{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top}.
$$

Let $\mathbf { 1 } \in \mathbb { R } ^ { K + M }$ denote the all-ones vector and let

$$
\mathbf {1} _ {G} := (\underbrace {1 , \ldots , 1} _ {K}, \underbrace {0 , \ldots , 0} _ {M}) \in \mathbb {R} ^ {K + M}
$$

serve as the indicator vector for the good block.

For $s ( \mathbf { p } ) \in ( 0 , 1 )$ , it is convenient to introduce within-block normalized coordinates:

$$
y _ {j} := \frac {p _ {j}}{s (\mathbf {p})} (j \leq K), \quad z _ {m} := \frac {p _ {b _ {m}}}{P _ {\text { bad }} (\mathbf {p})} (m \leq M).
$$

In this representation, $y \in \Delta ^ { K - 1 }$ and $z \in \Delta ^ { M - 1 }$ , such that $\mathfrak { p } = \left( s y , \left( 1 - s \right) z \right)$

Block symmetry and GRPO parametrization. We assume a structure of block symmetry and state dependence defined by:

$$
A _ {j} (\mathbf {p}) = a _ {\mathrm{g}} (s (\mathbf {p})) \quad (j \leq K), \quad A _ {b _ {m}} (\mathbf {p}) = a _ {\mathrm{b}} (s (\mathbf {p})) \quad (m \leq M).
$$

The resulting gap between good and bad arms is denoted as:

$$
\Delta (s) := a _ {\mathrm{g}} (s) - a _ {\mathrm{b}} (s).
$$

Under the GRPO specialization examined in this work, we set:

$$
a _ {\mathrm{g}} (s) = \frac {J s}{\sigma (s)}, \qquad a _ {\mathrm{b}} (s) = - \frac {J (1 - s)}{\sigma (s)}, \qquad \sigma (s) > 0.
$$

This formulation implies that:

$$
\Delta (s) = \frac {J}{\sigma (s)}.\tag{52}
$$

The advantage vector can then be expressed as:

$$
\mathbf {A} (\mathbf {p}) = a _ {\mathrm{b}} (s (\mathbf {p})) \mathbf {1} + \Delta (s (\mathbf {p})) \mathbf {1} _ {G}.
$$

GRPO mean-field flow. Our analysis focuses on the GRPO mean-field ordinary differential equation (ODE):

$$
\dot {\mathbf {p}} = \eta \mathfrak {J} (\mathbf {p}) ^ {2} \mathbf {A} (\mathbf {p}), \quad \eta > 0.\tag{53}
$$

Theorem E.1 (Dichotomy by the sign of J; exchange of stability at $J = 0 )$ . Assume the block-symmetric structure defined above, with ∆ given by equation 52. Define a scalar potential $F : \Delta ^ { K + M - 1 } $ R such that:

$$
F (\mathbf {p}) := F (s (\mathbf {p})), \quad F ^ {\prime} (s) = \Delta (s).\tag{54}
$$

Given that $s ( \mathbf { p } ) \in [ 0 , 1 ]$ and ∆ is integrable on $[ 0 , 1 ]$ , F remains bounded on $\Delta ^ { K + M - 1 }$ . For any constant C satisfying $C \geq \operatorname* { s u p } _ { \mathfrak { p } } F ( \mathbf { \hat { p } } )$ , we define the standard decreasing Lyapunovfunction:

$$
V (\mathbf {p}) := C - F (\mathbf {p}) \geq 0.\tag{55}
$$

Along any trajectory p(t) of equation 53 originating at $s ( 0 ) \in ( 0 , 1 )$ , the following properties hold:

(i) Lyapunov identity. For all $t \geq 0 \colon$

$$
\frac {d}{d t} V (\mathbf {p} (t)) = - \eta \left\| \mathfrak {J} (\mathbf {p} (t)) \mathbf {A} (\mathbf {p} (t)) \right\| _ {2} ^ {2} \leq 0,\tag{56}
$$

where equality holds if and only $i f \Im { \big ( } \mathbf { p } ( t ) { \big ) } \mathbf { A } { \big ( } \mathbf { p } ( t ) { \big ) } = 0$ . This is equivalent to:

$$
\frac {d}{d t} F (\mathbf {p} (t)) = \eta \left\| \Im (\mathbf {p} (t)) \mathbf {A} (\mathbf {p} (t)) \right\| _ {2} ^ {2} \geq 0.\tag{57}
$$

(ii) Explicit field and the sign of s˙. Let $s = s ( \mathbf { p } )$ and $P _ { \mathrm { b a d } } : = 1 - s$ . The components of the field are given by:

$$
\left[ \mathfrak {J} (\mathbf {p}) \mathbf {A} (\mathbf {p}) \right] _ {j} = p _ {j} P _ {\text { bad }} \Delta (s) \quad (j \leq K), \quad \left[ \mathfrak {J} (\mathbf {p}) \mathbf {A} (\mathbf {p}) \right] _ {b _ {m}} = - p _ {b _ {m}} s \Delta (s) \quad (m \leq M).
$$

Consequently, for $J \neq 0 :$

$$
\dot {s} (t) = \frac {1}{\Delta (s (t))} \frac {d}{d t} F (\mathbf {p} (t)) = \eta \Delta (s (t)) \left(P _ {\text {bad}} (t) ^ {2} \sum_ {j = 1} ^ {K} p _ {j} (t) ^ {2} + s (t) ^ {2} \sum_ {m = 1} ^ {M} p _ {b _ {m}} (t) ^ {2}\right).\tag{58}
$$

In normalized coordinates, this simplifies to:

$$
\dot {s} = \eta \Delta (s) [ s (1 - s) ] ^ {2} \left(\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}\right).\tag{59}
$$

Thus, for any $J \neq 0 ,$ the sign $o f \dot { s } ( t )$ is identical to the sign of J.

(iii) Global behavior and exchange of stability.

$I f J > 0 , s ( t )$ is strictly increasing such $t h a t s ( t ) \uparrow 1$ . In this case, every ω–limit point lies in the good face $\Delta _ { G } : = \{ { \bf p } : p _ { b _ { 1 } } = \cdot \cdot \cdot = p _ { b _ { M } } = 0 \}$

$I f J < 0 , s ( t )$ is strictly decreasing such that $s ( t ) \downarrow ($ . Here, every ω–limit point lies in the bad face $\bar { \Delta } _ { B } : = \{ { \bf \nabla } { \bf p } : { \bf \nabla } s ( { \bf p } ) = 0 \}$

$I f = 0 ,$ , the advantage $A _ { i } ( \mathbf { p } )$ is constant across all arms, implying ${ \dot { \pmb { \mathsf { P } } } } \equiv 0$ . Under these conditions, every point in the simplex is an equilibrium.

As J crosses zero, the global attractor switches from $\Delta _ { B }$ to $\Delta _ { G }$ . This represents a parameter-driven exchange of stability, or phase transition, at $J = 0$

(iv) Quantitative tail behavior. For $J \neq 0$ and bounded $\sigma ,$ andfollowing the bounds established in Appendix B, we observe $\begin{array} { r } { 0 < \sigma _ { \operatorname* { m i n } } \le \sigma ( s ) \le \sigma _ { \operatorname* { m a x } } \le \frac { 1 } { 4 } } \end{array}$ . By Jensen’s inequality, we obtain:

$$
| \dot {s} (t) | \geq \eta \frac {| J |}{\sigma_ {\max}} \left(\frac {1}{K} + \frac {1}{M}\right) s (t) ^ {2} (1 - s (t)) ^ {2}.\tag{60}
$$

For $J > 0$ , there exists a time $T _ { 1 / 2 }$ such thatfor all $\begin{array} { r } { t \ge T _ { 1 / 2 } , s ( t ) \ge \frac { 1 } { 2 } } \end{array}$ and:

$$
P _ {\text { bad }} (t) \leq \left(\frac {1}{P _ {\text { bad }} (T _ {1 / 2})} + \frac {\eta J}{4 \sigma_ {\max}} \left(\frac {1}{K} + \frac {1}{M}\right) (t - T _ {1 / 2})\right) ^ {- 1}.\tag{61}
$$

Similarly, for $J < 0 , s ( t )$ follows an $\mathcal { O } ( 1 / t )$ decay after a finite transient period.

Proof. Let $\textstyle F ( s ) : = \int _ { 0 } ^ { s } \Delta ( u ) d u$ . Since $F ( \mathbf { p } ) = F ( s ( \mathbf { p } ) )$ and $s ( \mathbf { p } ) = \Sigma _ { j \leq K } p _ { j } .$ , it follows that $\nabla F ( \mathbf { p } ) = \Delta ( s ) \mathbf { 1 } _ { G }$ Utilizing the fact that $\Im ( \mathbf { p } ) \mathbf { 1 } = \mathbf { 0 }$ , we find:

$$
\mathfrak {J} (\mathbf {p}) \nabla F (\mathbf {p}) = \Delta (s) \mathfrak {J} (\mathbf {p}) \mathbf {1} _ {G} = \mathfrak {J} (\mathbf {p}) \mathbf {A} (\mathbf {p}).
$$

Applying the chain rule and the symmetry of $\Im ( { \mathfrak { p } } )$ yields:

$$
\frac {d}{d t} F (\mathbf {p} (t)) = \nabla F (\mathbf {p} (t)) ^ {\top} \dot {\mathbf {p}} (t) = \eta \left\| \mathfrak {J} (\mathbf {p} (t)) \mathbf {A} (\mathbf {p} (t)) \right\| _ {2} ^ {2},
$$

which confirms equation $^ { 5 7 }$ and, by extension, equation 56.

The coordinate-wise expressions in (ii) arise from the identity $[ \Im ( { \bf p } ) { \bf A } ] _ { i } = p _ { i } ( A _ { i } - \bar { A } )$ , where $\bar { A } = s a _ { \mathrm { g } } ( s ) +$ $( 1 - s ) a _ { \mathrm { b } } ( s )$ . The resulting differential inequality in (iv) is obtained by combining equation 58 with Jensen’s lower bounds and the definition $\Delta ( s ) = J / \sigma ( s )$ . Integrating over $\left[ T _ { 1 / 2 } , t \right]$ while assuming $\begin{array} { r } { s ( t ) \geq \frac { 1 } { 2 } } \end{array}$ produces the bound in equation 61. □

Remark E.2 (Interpretation). The dynamics described by equation 58 and equation 59 indicate that the direction of mass transfer between blocks is governed exclusively by the sign of J. For $J > 0 ,$ , the good face $\Delta _ { G }$ is globally attracting, whereas $J < 0$ renders the bad face $\Delta _ { B }$ the attractor. At the critical value $\stackrel { \smile } { J } = 0$ , the system undergoes a phase transition characterized by a degenerate continuum of equilibria.

Decomposition dynamics and Shahshahani structure. For $s \in ( 0 , 1 )$ , we decompose the flow into coordinates $\left( y , z , s \right)$ . Equation equation 53 then reduces to the following system:

$$
\dot {y} = \kappa (s) (y \odot y - \| y \| _ {2} ^ {2} y), \quad \dot {z} = - \kappa (s) (z \odot z - \| z \| _ {2} ^ {2} z), \quad \dot {s} = \eta \Delta (s) [ s (1 - s) ] ^ {2} (\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}),\tag{62}
$$

where $\kappa ( s ) : = \eta \Delta ( s ) s ( 1 - s )$ . This reveals that $\begin{array} { r } { \Phi ( y ) : = \frac { 1 } { 2 } \| y \| _ { 2 } ^ { 2 } } \end{array}$ serves as a state-scaled Shahshahani-gradient potential on $\Delta ^ { K - 1 }$ , while $\begin{array} { r } { \Psi ( z ) : = \frac { 1 } { 2 } \| z \| _ { 2 } ^ { 2 } } \end{array}$ acts as the corresponding potential on $\Delta ^ { M - 1 }$ but with an inverted sign. For $J > 0$ , we observe $\dot { \Phi } \geq 0$ and $\dot { \Psi } \le 0$ . These inequalities reverse when $J < 0$

Probabilistic interpretation of the potential $\Phi ( y )$ . The function $\begin{array} { r } { \Phi ( y ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { K } y _ { i } ^ { 2 } } \end{array}$ represents the collision probability, also known as the Herfindahl-Hirschman index. It quantifies the concentration within the good block, where $\Phi ( y ) = 1 / 2 K$ at the uniform distribution and $\Phi \dot { ( } y ) = 1 / 2$ at any vertex. An increase in Φ signifies that mass is condensing onto a smaller subset of arms, while a decrease indicates a more uniform distribution.

The induced intra-good dynamics are expressed as $\dot { y } = \kappa ( s ) \mathrm { \ g r a d } _ { \mathrm { S h a h } } \Phi ( y )$ . This shows that y follows a Shahshahani gradient flow of Φ. Consequently, if $J > { \mathsf { 0 } } ,$ the collision probability increases over time, leading to a rich-get-richer effect where the distribution concentrates toward a vertex. Conversely, if $J < 0 ,$ , the flow promotes diversity by pushing y toward a uniform distribution.

Coordination-game correspondence. The replicator field $\dot { y } _ { i } \propto y _ { i } \big ( y _ { i } - \| y \| _ { 2 } ^ { 2 } \big )$ corresponds to the replicator dynamics of a symmetric pure coordination game. In this context, the payoff for each arm is equal to its current population fraction. Our Shahshahani-gradient identity formally states that the replicator dynamics ascend this potential when $\kappa > 0$ and descend it when $\kappa < 0$

## F Bad arm dynamics under PPO/GRPO style importance sampling and clipping

We demonstrate that, within the small-step regime, importance sampling (IS) as employed in PPO and GRPO does not alter the leading-order mean-field ODE established in Appendix C. Specifically, IS modifies the conditional mean drift only at order $O ( \eta ^ { 2 } )$ , ensuring that the $O ( \eta )$ ODE limit remains identical to the on-policy (REINFORCE) system.

Setup. Consider a softmax policy $\pi _ { \theta }$ over arms $i \in \{ b , 1 , \ldots , K \}$ . During an update, samples are drawn from an $\mathrm { \Omega ^ { \prime \prime } o l d ^ { \prime \prime } }$ policy $\pi _ { \mathrm { o l d } }$ with probability vector $\boldsymbol { \mathsf { p } } = \boldsymbol { \mathsf { p } } _ { \mathrm { o l d } }$ . The parameters are then updated to $\mathrm { ~ \tt ~ { ~ 1 ~ } ~ } ^ { \prime \prime } \mathrm { { n e w } ^ { \prime \prime } }$ policy $\pi _ { \mathrm { n e w } }$ with probability vector ${ \mathfrak { p } } ^ { + } = { \mathfrak { p } } _ { \mathrm { n e w } }$ . We define the exact importance ratio as:

$$
\rho_ {i} := \frac {\pi_ {\mathrm{new}} (i)}{\pi_ {\mathrm{old}} (i)} = \frac {p _ {i} ^ {+}}{p _ {i}}.
$$

This ratio is well-defined because the softmax function has full support $( p _ { i } > 0 )$ . Let r˜ denote the scalar signal used in the update, and let:

$$
A _ {i} (\mathbf {p}) := \mathbb {E} [ \tilde {r} \mid I = i, \mathbf {p} ].
$$

We assume su $\geqslant _ { i , \mathbf { p } } | A _ { i } ( \mathbf { p } ) | < \infty$ . In this context, the small-step regime implies $\| \Delta \theta \| = O ( \eta )$ and consequently $\lVert \mathbf { p } ^ { + } - \mathbf { p } \rVert = O ( \bar { \eta } )$ , a result supported by Lemma H.1.

IS score-function update. The IS-corrected score-function update is given by:

$$
g _ {\mathrm{IS}} := \tilde {r} \rho_ {I} \nabla_ {\theta} \log \pi_ {\mathrm{new}} (I), \qquad I \sim \pi_ {\mathrm{old}}, \qquad \Delta \theta = \eta g _ {\mathrm{IS}}.
$$

For a softmax policy, where $\nabla _ { \boldsymbol { \theta } } \log \pi _ { \mathrm { n e w } } ( i ) = \boldsymbol { e } _ { i } - \mathbf { p } ^ { + }$ , the update simplifies to $g _ { \mathrm { I S } } = \tilde { r } \rho _ { I } \left( e _ { I } - \mathbf { p } ^ { + } \right)$

Proposition F.1 (IS affects the mean logit drift only at $O ( \eta ^ { 2 } ) )$ ). Define the on-policy vector field:

$$
G (\mathbf {p}) := \sum_ {i} p _ {i} A _ {i} (\mathbf {p}) (e _ {i} - \mathbf {p}).
$$

In the small-step regime, the following holds:

$$
\mathbb {E} [ \Delta \theta \mid \mathbf {p} ] = \eta G (\mathbf {p}) + O (\eta^ {2}).
$$

Proof. Taking the conditional expectation and applying $\mathbb { E } [ \tilde { r } \mid I = i , \mathbf { p } ] = A _ { i } ( \mathbf { p } )$ , we have:

$$
\mathbb {E} \left[ g _ {\mathrm{IS}} \mid \mathbf {p} \right] = \sum_ {i} p _ {i} \rho_ {i} A _ {i} (\mathbf {p}) \left(e _ {i} - \mathbf {p} ^ {+}\right).
$$

By utilizing the identity $p _ { i } \rho _ { i } = p _ { i } ^ { + }$ , the expression becomes:

$$
\mathbb {E} [ g _ {\mathrm{IS}} \mid \mathbf {p} ] = \sum_ {i} p _ {i} ^ {+}   A _ {i} (\mathbf {p})   (e _ {i} - \mathbf {p} ^ {+}) =: \widetilde {G} (\mathbf {p} ^ {+}; \mathbf {p}), \quad \widetilde {G} (\mathbf {q}; \mathbf {p}) := \sum_ {i} q _ {i}   A _ {i} (\mathbf {p})   (e _ {i} - \mathbf {q}).
$$

For a fixed p, the map $\mathbf { q } \mapsto \widetilde { G } ( \mathbf { q } ; \mathbf { p } )$ is a smooth polynomial in q with bounded coefficients. It is therefore locally Lipschitz in q. Given that $\| \boldsymbol { \mathsf { p } } ^ { + } - \boldsymbol { \mathsf { p } } \| = \bar { O ( \eta ) }$ , it follows that:

$$
\widetilde {G} (\mathbf {p} ^ {+}; \mathbf {p}) = \widetilde {G} (\mathbf {p}; \mathbf {p}) + O (\| \mathbf {p} ^ {+} - \mathbf {p} \|) = G (\mathbf {p}) + O (\eta).
$$

Multiplying by η yields the desired result: $\mathbb { E } [ \Delta \theta \ | \ \mathbf { p } ] = \eta G ( \mathbf { p } ) + O ( \eta ^ { 2 } )$

Expanded form and relation to Appendix C. This conclusion can also be derived by examining the identity $\rho _ { i } = 1 + \Delta p _ { i } / p _ { i } ,$ where $\Delta p _ { i } : = p _ { i } ^ { + } - p _ { i }$ . This implies that $\mathbb { E } [ \rho _ { i } \mid \mathbf { p } ] = 1 + \mathbb { E } [ \Delta p _ { i } \mid \mathbf { p } ] / p _ { i }$ . Since replacing $\mathbf { p } ^ { + }$ with p introduces only an $O ( \eta )$ error, its contribution after multiplying by η is $O ( \eta ^ { 2 } )$ . Consequently:

$$
\mathbb {E} [ \Delta \theta \mid \mathbf {p} ] = \eta \sum_ {i} \left(p _ {i} + \mathbb {E} [ \Delta p _ {i} \mid \mathbf {p} ]\right) A _ {i} (\mathbf {p}) (e _ {i} - \mathbf {p}) + O (\eta^ {2}).\tag{63}
$$

By letting $\begin{array} { r } { \bar { A } : = \sum _ { i } p _ { i } A _ { i } ( \mathbf { p } ) } \end{array}$ and $\begin{array} { r } { \bar { A } ^ { \prime } : = \sum _ { i } \mathbb { E } [ \Delta p _ { i } \ | \ \mathbf { p } ] A _ { i } ( \mathbf { p } ) } \end{array}$ , the coordinate-wise updates are expressed as:

$$
\mathbb {E} \left[ \Delta \theta_ {i} \mid \mathbf {p} \right] = \eta p _ {i} \left(A _ {i} (\mathbf {p}) - \bar {A}\right) + \eta \left(\mathbb {E} \left[ \Delta p _ {i} \mid \mathbf {p} \right] A _ {i} (\mathbf {p}) - p _ {i} \bar {A} ^ {\prime}\right) + O \left(\eta^ {2}\right).\tag{64}
$$

This structure highlights the ”extra terms” that appear when the IS mean update is expanded. To show these terms are second order, we apply the softmax pushforward from Lemma H.1, yielding $\mathbb { E } [ \Delta \mathbf { p } \mid \mathbf { p } ] =$ $\Im ( \mathbf { p } ) \mathbb { E } [ \Delta \theta \ | \ \mathbf { p } ] + O ( \eta ^ { 2 } )$ . Because $\bar { \mathbb { E } } [ \bar { \Delta } \bar { \theta } \mid \mathfrak { p } ] = O ( \eta )$ , the term $\mathbb { E } [ \Delta p _ { i } \mid \mathbf { p } ]$ is also ${ \cal { O } } ( \eta )$ , confirming that the correction in equation 64 is $O ( \eta ^ { 2 } )$ ).

Block-symmetric specialization. Under the assumption of block symmetry, we have $A _ { b _ { m } } ( { \bf { p } } ) = a _ { \bf { b } } ( p )$ and $A _ { j } ( \mathfrak { p } ) = a _ { \mathrm { g } } ( \mathfrak { p } )$ for $m = 1 , \ldots , M$ and $j = 1 , \dots , K .$ . The mean advantage is then $\bar { A } = p a _ { \mathrm { b } } ( \ddot { p } ) + ( 1 - p ) a _ { \mathrm { g } } ( p )$ The leading terms in equation 64 result in:

$$
\mathbb {E} [ \Delta \theta_ {b _ {m}} \mid \mathbf {p} ] = \eta p (1 - p) \big (a _ {\mathrm{b}} (p) - a _ {\mathrm{g}} (p) \big) z _ {m} + O (\eta^ {2}),
$$

$$
\mathbb {E} [ \Delta \theta_ {j} \mid \mathbf {p} ] = - \eta p (1 - p) \big (a _ {\mathrm{b}} (p) - a _ {\mathrm{g}} (p) \big) y _ {j} + O (\eta^ {2}).
$$

These expressions match the REINFORCE dynamics to first order in η.

Corollary F.2 (Invariance of the leading-order mean-field ODE). In the small-step regime, the conditional expectation of the probability update satisfies:

$$
\mathbb {E} [ \Delta \mathbf {p} | \mathbf {p} ] = \eta \Im (\mathbf {p}) G (\mathbf {p}) + O (\eta^ {2}).
$$

Consequently, the leading-order mean-field ODE for good and bad arm dynamics remains as derived in Appendix C.

Clipping in PPO and GRPO. When clipping is applied to the ratios such that $\widehat { \rho } _ { i } = \mathrm { c l i p } ( \rho _ { i } , 1 - \varepsilon , 1 + \varepsilon ^ { \prime } )$ , the ratio ρ<sub>i</sub> approaches 1 as $\eta  0$ . For sufficiently small η, the clipping mechanism remains inactive. Therefore, clipping does not influence the leading-order ODE.

Conclusion. In the small-step regime, the inclusion of importance sampling and clipping with fixed thresholds does not change the leading-order mean-field ODE. The impact of these techniques is confined to the $O ( \eta ^ { 2 } )$ terms.

## G KL Regularization

In this section, we analyze how KL regularization modifies the policy dynamics in our multi-armed bandit abstraction. Many practical algorithms (e.g., PPO/GRPO-style methods) add a KL penalty to keep the learned policy close to a reference policy, and we study the resulting mean-field drift in our multi-good/multi-bad model.

We begin by recalling the notation for the multi-good/bad-arm model introduced in Appendix C. Given K good arms and M bad arms, we represent the policy as

$$
\mathbf {p} = (p _ {1}, \ldots , p _ {K}, p _ {b _ {1}}, \ldots , p _ {b _ {M}}) \in \Delta^ {K + M - 1}.
$$

The total bad mass and the within-block compositions are

$$
p = \sum_ {m = 1} ^ {M} p _ {b _ {m}} \in [ 0, 1 ], \quad y _ {j} = \frac {p _ {j}}{1 - p} (j \leq K), \quad z _ {m} = \frac {p _ {b _ {m}}}{p} (m \leq M).
$$

Equivalently,

$$
p _ {j} = (1 - p) y _ {j} \quad (j \leq K), \quad p _ {b _ {m}} = p z _ {m} \quad (m \leq M),
$$

with $y \in \Delta ^ { K - 1 }$ and $z \in \Delta ^ { M - 1 }$

Reference policy. We define the KL reference policy $\displaystyle \mathbf { p } ^ { \mathrm { r e f } }$ analogously:

$$
\mathbf {p} ^ {\mathrm{ref}} = \big ((1 - p _ {\mathrm{ref}}) y _ {1} ^ {\mathrm{ref}}, \ldots , (1 - p _ {\mathrm{ref}}) y _ {K} ^ {\mathrm{ref}}, p _ {\mathrm{ref}} z _ {1} ^ {\mathrm{ref}}, \ldots , p _ {\mathrm{ref}} z _ {M} ^ {\mathrm{ref}} \big),
$$

where $p _ { \mathrm { r e f } } \in ( 0 , 1 ) , y ^ { \mathrm { r e f } } \in \Delta ^ { K - 1 }$ , and $z ^ { \mathrm { r e f } } \in \Delta ^ { M - 1 }$ . We also use the bad-mass log-odds

$$
\ell (p) := \log \frac {p}{1 - p}, \quad \ell_ {\text { ref }} := \log \frac {p _ {\text { ref }}}{1 - p _ {\text { ref }}}.
$$

Replicator/natural-gradient form (recall). Throughout, we view a penalty $\Phi ( \mathbf { p } )$ as inducing a Shahshahani (natural-gradient) flow on the simplex. Concretely, for the objective contribution $- \beta \Phi ( \mathbf { \bar { p } } )$ , the induced replicator flow is

$$
\dot {p} _ {i} = - \beta p _ {i} \left(\partial_ {p _ {i}} \Phi (\mathbf {p}) - \sum_ {k} p _ {k} \partial_ {p _ {k}} \Phi (\mathbf {p})\right).\tag{65}
$$

Equivalently, if we interpret $\Delta \theta _ { i }$ as a small logit increment, then to first order

$$
\Delta p _ {i} = p _ {i} \Big (\Delta \theta_ {i} - \sum_ {k} p _ {k} \Delta \theta_ {k} \Big) + O (\| \Delta \theta \| ^ {2}).\tag{66}
$$

Thus, choosing centered increments with $\begin{array} { r } { \sum _ { k } p _ { k } \Delta \theta _ { k } = 0 \mathrm { y i e l d s } \Delta p _ { i } = p _ { i } \Delta \theta _ { i } } \end{array}$ at first order, which makes it easy to realize a desired replicator drift.

Lemma G.1 (Exact reverse-KL decomposition (multi-bad setting)). The reverse KL divergence decomposes into a two-class term (bad vs. good) plus within-block terms:

$$
D _ {\mathrm{KL}} (\mathbf {p} \| \mathbf {p} ^ {\text {ref}}) = \underbrace {p \log \frac {p}{p _ {\text {ref}}} + (1 - p) \log \frac {1 - p}{1 - p _ {\text {ref}}}} _ {\text {two - class (bad vs. good)}} + \underbrace {(1 - p) D _ {\mathrm{KL}} (y \| y ^ {\text {ref}})} _ {\text {within - good}} + \underbrace {p D _ {\mathrm{KL}} (z \| z ^ {\text {ref}})} _ {\text {within - bad}}.
$$

Proof. Substituting $p _ { j } = ( 1 - p ) y _ { j }$ and $p _ { b _ { m } } = p z _ { m } ,$ we get

$$
\log \frac {p _ {j}}{p _ {j} ^ {\mathrm{ref}}} = \log \frac {1 - p}{1 - p _ {\mathrm{ref}}} + \log \frac {y _ {j}}{y _ {j} ^ {\mathrm{ref}}}, \quad \log \frac {p _ {b _ {m}}}{p _ {b _ {m}} ^ {\mathrm{ref}}} = \log \frac {p}{p _ {\mathrm{ref}}} + \log \frac {z _ {m}}{z _ {m} ^ {\mathrm{ref}}}.
$$

Summing $\textstyle \sum _ { j \leq K } p _ { j } ( \cdot )$ and $\begin{array} { r } { \sum _ { m \leq M } p _ { b _ { m } } ( \cdot ) } \end{array}$ yields the stated decomposition.

Two KL choices. In practice, one may penalize either (A) only the two-class (inter-block) divergence or (B) the full reverse-KL divergence.

Proposition G.2 (Two-class KL penalty (bad vs. good only)). Let

$$
\Phi_ {2 \mathrm{c}} (p) := p \log \frac {p}{p _ {\text { ref }}} + (1 - p) \log \frac {1 - p}{1 - p _ {\text { ref }}}.
$$

The Shahshahani/natural-gradient (replicator) flow for $- \beta \Phi _ { 2 \mathrm { c } }$ satisfies

$$
\dot {p} \Big | _ {\mathrm{KL}, 2 \mathrm{c}} = - \beta   p (1 - p) (\ell - \ell_ {\mathrm{ref}}), \qquad \dot {\ell} \Big | _ {\mathrm{KL}, 2 \mathrm{c}} = - \beta (\ell - \ell_ {\mathrm{ref}}).
$$

A centered small-step logit update realizing this flow $( i . e . , \sum _ { i } p _ { i } \Delta \theta _ { i } ^ { \mathrm { K L } } = 0 )$ is

$$
\Delta \theta_ {b _ {m}} ^ {\mathrm{KL}} = - \beta (1 - p) (\ell - \ell_ {\mathrm{ref}}) (m \leq M), \quad \Delta \theta_ {j} ^ {\mathrm{KL}} = + \beta p (\ell - \ell_ {\mathrm{ref}}) (j \leq K).
$$

Because $\Delta \theta _ { i } ^ { \mathrm { K L } }$ is constant within each block, this KL term induces no deterministic drift in y $o r z ;$ it acts only on the total bad mass p.

Proof sketch (with key steps). View $\Phi _ { 2 \mathrm { c } }$ as a function on the full simplex via $\begin{array} { r } { p = \sum _ { m = 1 } ^ { M } p _ { b _ { m } } } \end{array}$ . Then

$$
\partial_ {p _ {b _ {m}}} \Phi_ {2 c} (\mathbf {p}) = \Phi_ {2 c} ^ {\prime} (p), \quad \partial_ {p _ {j}} \Phi_ {2 c} (\mathbf {p}) = 0.
$$

Moreover,

$$
\sum_ {i} p _ {i} \partial_ {p _ {i}} \Phi_ {2 c} (\mathbf {p}) = \sum_ {m = 1} ^ {M} p _ {b _ {m}} \Phi_ {2 c} ^ {\prime} (p) = p \Phi_ {2 c} ^ {\prime} (p).
$$

Plugging into the replicator formula equation 65 gives the coordinate-wise drifts

$$
\dot {p} _ {b _ {m}} = - \beta p _ {b _ {m}} \Big (\Phi_ {2 c} ^ {\prime} (p) - p \Phi_ {2 c} ^ {\prime} (p) \Big) = - \beta p _ {b _ {m}} (1 - p) \Phi_ {2 c} ^ {\prime} (p),
$$

$$
\dot {p} _ {j} = - \beta p _ {j} \left(0 - p \Phi_ {2 c} ^ {\prime} (p)\right) = + \beta p _ {j} p \Phi_ {2 c} ^ {\prime} (p).
$$

Summing over bad arms yields

$$
\dot {p} = \sum_ {m = 1} ^ {M} \dot {p} _ {b _ {m}} = - \beta (1 - p) \Phi_ {2 c} ^ {\prime} (p) \sum_ {m = 1} ^ {M} p _ {b _ {m}} = - \beta p (1 - p) \Phi_ {2 c} ^ {\prime} (p).
$$

A direct derivative computation shows

$$
\Phi_ {2 \mathrm{c}} ^ {\prime} (p) = \log \frac {p}{p _ {\mathrm{ref}}} - \log \frac {1 - p}{1 - p _ {\mathrm{ref}}} = \log \frac {p}{1 - p} - \log \frac {p _ {\mathrm{ref}}}{1 - p _ {\mathrm{ref}}} = \ell - \ell_ {\mathrm{ref}},
$$

which gives the claimed p equation. Since˙ $\dot { \ell } = \dot { p } / [ p ( 1 - p ) ]$ , we obtain $\dot { \ell } = - \beta ( \ell - \ell _ { \mathrm { r e f } } )$

Finally, there is no within-block drift: for instance, for any $m , r \leq M ,$

$$
\frac {d}{d t} \log \frac {p _ {b _ {m}}}{p _ {b _ {r}}} = \frac {\dot {p} _ {b _ {m}}}{p _ {b _ {m}}} - \frac {\dot {p} _ {b _ {r}}}{p _ {b _ {r}}} = - \beta (1 - p) \Phi_ {2 c} ^ {\prime} (p) + \beta (1 - p) \Phi_ {2 c} ^ {\prime} (p) = 0,
$$

so all ratios ${ p _ { b } } _ { m } / p _ { b _ { r } }$ are constant and hence z is constant; the same argument applies to y.

For the logit realization, take the centered velocity

$$
\Delta \theta_ {i} ^ {\mathrm{KL}} = - \beta \Bigl (\partial_ {p _ {i}} \Phi_ {2 \mathrm{c}} - \sum_ {k} p _ {k} \partial_ {p _ {k}} \Phi_ {2 \mathrm{c}} \Bigr),
$$

which gives exactly the stated block-constant increments (substitute $\Phi _ { 2 \mathrm { c } } ^ { \prime } ( p ) = \ell - \ell _ { \mathrm { r e f } } )$ . By centering, $\begin{array} { r } { \sum _ { i } p _ { i } \Delta \theta _ { i } ^ { \mathrm { K L } } = 0 } \end{array}$ and then equation 66 implies

$$
\Delta p = \sum_ {m = 1} ^ {M} \Delta p _ {b _ {m}} = \sum_ {m = 1} ^ {M} p _ {b _ {m}} \Delta \theta_ {b _ {m}} ^ {\mathrm{KL}} = - \beta   p (1 - p) (\ell - \ell_ {\mathrm{ref}}),
$$

matching the continuous-time drift.

Proposition G.3 (Full reverse-KL penalty (multi-bad setting)). Let $\Phi _ { \mathrm { f u l l } } ( \mathbf { p } ) : = D _ { \mathrm { K L } } ( \mathbf { p } \| \mathbf { p } ^ { \mathrm { r e f } } )$ . The replicator $f l o w \bar { f } o r - \beta \Phi _ { \mathrm { f u l l } }$ induces the bad-mass drift

$$
\dot {p} \Big | _ {\mathrm{KL,full}} = - \beta p (1 - p) \Big (\ell - \ell_ {\mathrm{ref}} - D _ {\mathrm{KL}} (y \| y ^ {\mathrm{ref}}) + D _ {\mathrm{KL}} (z \| z ^ {\mathrm{ref}}) \Big),
$$

and hence

$$
\dot {\ell} \Big | _ {\mathrm{KL}, \text {full}} = - \beta \Big (\ell - \ell_ {\text {ref}} - D _ {\mathrm{KL}} (y \| y ^ {\text {ref}}) + D _ {\mathrm{KL}} (z \| z ^ {\text {ref}}) \Big).
$$

A canonical centered logit increment follows the natural-gradient form

$$
\Delta \theta_ {i} ^ {\mathrm{KL}} = - \beta \Big (\log \frac {p _ {i}}{p _ {i} ^ {\mathrm{ref}}} - \sum_ {k} p _ {k} \log \frac {p _ {k}}{p _ {k} ^ {\mathrm{ref}}} \Big),
$$

which satisfies $\begin{array} { r } { \sum _ { i } p _ { i } \Delta \theta _ { i } ^ { \mathrm { K L } } = 0 } \end{array}$ and yields the stated p-drift.

Moreover, in (y, z)-coordinates, the full reverse-KL induces independent within-block replicator pulls:

$$
\left. \dot {y} _ {j} \right| _ {\text {KL,full}} = - \beta y _ {j} \Big (\log \frac {y _ {j}}{y _ {j} ^ {\text {ref}}} - \sum_ {i = 1} ^ {K} y _ {i} \log \frac {y _ {i}}{y _ {i} ^ {\text {ref}}} \Big), \qquad \dot {z} _ {m} \Big | _ {\text {KL,full}} = - \beta z _ {m} \Big (\log \frac {z _ {m}}{z _ {m} ^ {\text {ref}}} - \sum_ {r = 1} ^ {M} z _ {r} \log \frac {z _ {r}}{z _ {r} ^ {\text {ref}}} \Big).
$$

Proof (outline with the main algebra). For $\begin{array} { r } { \Phi _ { \mathrm { f u l l } } ( \mathfrak { p } ) = \sum _ { i } p _ { i } \log \frac { p _ { i } } { p _ { i } ^ { \mathrm { r e f } } } } \end{array}$ , we have $\begin{array} { r } { \partial _ { p _ { i } } \Phi _ { \mathrm { f u l l } } = \log \frac { p _ { i } } { p _ { i } ^ { \mathrm { r e f } } } + 1 ; } \end{array}$ the constant +1 cancels under the mean-subtraction in equation 65, yielding

$$
\dot {p} _ {i} = - \beta p _ {i} \left(\log \frac {p _ {i}}{p _ {i} ^ {\text { ref }}} - \sum_ {k} p _ {k} \log \frac {p _ {k}}{p _ {k} ^ {\text { ref }}}\right).
$$

To get ${ \dot { p } } ,$ sum over bad indices:

$$
\dot {p} = \sum_ {m = 1} ^ {M} \dot {p} _ {b _ {m}} = - \beta \Big (\sum_ {m = 1} ^ {M} p _ {b _ {m}} \log \frac {p _ {b _ {m}}}{p _ {b _ {m}} ^ {\text { ref }}} - p \sum_ {k} p _ {k} \log \frac {p _ {k}}{p _ {k} ^ {\text { ref }}} \Big).
$$

Using the block parametrization,

$$
\log \frac {p _ {b _ {m}}}{p _ {b _ {m}} ^ {\text { ref }}} = \log \frac {p}{p _ {\text { ref }}} + \log \frac {z _ {m}}{z _ {m} ^ {\text { ref }}}, \qquad \log \frac {p _ {j}}{p _ {j} ^ {\text { ref }}} = \log \frac {1 - p}{1 - p _ {\text { ref }}} + \log \frac {y _ {j}}{y _ {j} ^ {\text { ref }}},
$$

we obtain

$$
\sum_ {m = 1} ^ {M} p _ {b _ {m}} \log \frac {p _ {b _ {m}}}{p _ {b _ {m}} ^ {\mathrm{ref}}} = p \log \frac {p}{p _ {\mathrm{ref}}} + p D _ {\mathrm{KL}} (z \| z ^ {\mathrm{ref}}),
$$

and

$$
\sum_ {k} p _ {k} \log \frac {p _ {k}}{p _ {k} ^ {\mathrm{ref}}} = \Phi_ {2 \mathrm{c}} (p) + (1 - p) D _ {\mathrm{KL}} (y \| y ^ {\mathrm{ref}}) + p D _ {\mathrm{KL}} (z \| z ^ {\mathrm{ref}}) \quad (b y L e m m a G. 1).
$$

Substituting and simplifying yields

$$
\dot {p} = - \beta p (1 - p) \Big (\log \frac {p}{p _ {\mathrm{ref}}} - \log \frac {1 - p}{1 - p _ {\mathrm{ref}}} - D _ {\mathrm{KL}} (y \| y ^ {\mathrm{ref}}) + D _ {\mathrm{KL}} (z \| z ^ {\mathrm{ref}}) \Big),
$$

and the log-oddsformfollows since log $\begin{array} { r } { \frac { p } { p _ { \mathrm { r e f } } } - \log \frac { 1 - p } { 1 - p _ { \mathrm { r e f } } } = \ell - \ell _ { \mathrm { r e f } } . } \end{array}$ Finally, the $( y , z )$ equationsfollow by differentiating $y _ { j } = p _ { j } / ( 1 - p )$ and $z _ { m } = p _ { b _ { m } } / p$ and substituting the above $\dot { p } _ { i }$ expressions; the same cancellations as in Proposition G.2 reduce each block to its own replicator pull toward the corresponding within-block reference. □

Remark G.4 (Comparison of KL penalties). The two-class penalty $\Phi _ { 2 \mathrm { c } } ( \boldsymbol { p } )$ depends only on the aggregate bad mass p (equivalently, on $\begin{array} { r } { \ell = \log { \frac { p } { 1 - p } } ) } \end{array}$ . Consequently, its Shahshahani field is block-constant: all good arms receive the same logit increment and all bad arms receive the same logit increment. This implies that the within-block compositions are invariant,

$$
\dot {y} \Big | _ {\mathrm{KL}, 2 \mathrm{c}} = 0, \qquad \dot {z} \Big | _ {\mathrm{KL}, 2 \mathrm{c}} = 0,
$$

and the KL regularizer acts solely as a logistic contraction of ℓ toward $\ell _ { \mathrm { r e f } } .$

$$
\left. \dot {\ell} \right| _ {\mathrm{KL}, 2 \mathrm{c}} = - \beta (\ell - \ell_ {\mathrm{ref}}).
$$

In contrast, the full reverse-KL splits as

$$
D _ {\mathrm{KL}} (\mathbf {p} \| \mathbf {p} ^ {\text { ref }}) = \Phi_ {2 c} (p) + (1 - p) D _ {\mathrm{KL}} (y \| y ^ {\text { ref }}) + p D _ {\mathrm{KL}} (z \| z ^ {\text { ref }}),
$$

so it produces two effects simultaneously: (i) independent within-block replicator pulls that damp deviations of y and z from $y ^ { \mathrm { r e f } }$ and $z ^ { \mathrm { r e f } } .$ , and (ii) an additional scalar feedback into the bad-mass drift. Concretely, the effective restoring force on the log-odds becomes

$$
\dot {\ell} \Big | _ {\mathrm{KL,full}} = - \beta \Big (\ell - \ell_ {\mathrm{ref}} - D _ {\mathrm{KL}} (y \| y ^ {\mathrm{ref}}) + D _ {\mathrm{KL}} (z \| z ^ {\mathrm{ref}}) \Big).
$$

Thus, if the bad block is more “misaligned” than the good block $( D _ { \mathrm { K L } } ( z \| z ^ { \mathrm { r e f } } ) > D _ { \mathrm { K L } } ( y \| y ^ { \mathrm { r e f } } ) )$ , the KL term strengthens the push to decrease p; whereas a larger within-good mismatch $( D _ { \mathrm { K L } } ( y \vert \vert y ^ { \mathrm { r e f } } )$ large) can partially counteract that push because it is weighted by $( 1 - p )$ in the decomposition. In particular, when $y = y ^ { \mathrm { r e f } }$ and $z = z ^ { \mathrm { r e f } }$ , the full reverse-KL reduces to the two-class behavior.

## G.1 Full Mean-Field ODE for the Bad Mass with KL

We now integrate the reward-driven mean-field drift with the KL penalties derived above. For this analysis, we omit the clipping or importance-ratio factors typically found in PPO/GRPO: our previous results indicate these contribute only second-order corrections and do not alter the fundamental phase portrait of the mean-field drift.

Define the following quantities:

$$
\alpha (p) := \eta \frac {J}{\sigma (p)} p (1 - p), \quad s _ {2} := \| y \| _ {2} ^ {2} \in \left[ \frac {1}{K}, 1 \right], \quad t _ {2} := \| z \| _ {2} ^ {2} \in \left[ \frac {1}{M}, 1 \right].
$$

Based on the reward term alone, the baseline drift of the total bad mass is

$$
\mathbb {E} [ \Delta p ] _ {\mathrm{reward}} = - \alpha (p) p (1 - p) (s _ {2} + t _ {2}) = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} (s _ {2} + t _ {2}).\tag{67}
$$

Combined Dynamics. Combining the reward drift with each KL choice yields the following governed ODEs:

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} (s _ {2} + t _ {2}) - \beta p (1 - p) (\ell - \ell_ {\mathrm{ref}})\tag{two-class KL}
$$

(68)

$$
\dot {p} = - \eta \frac {J}{\sigma (p)} [ p (1 - p) ] ^ {2} (s _ {2} + t _ {2}) - \beta p (1 - p) (\ell - \ell_ {\mathrm{ref}} - D _ {\mathrm{KL}} (y \| y ^ {\mathrm{ref}}) + D _ {\mathrm{KL}} (z \| z ^ {\mathrm{ref}})) \quad \text {(full reverse - KL).}\tag{69}
$$

## G.2 Nullcline, Interior Equilibrium, and the Prevention of Collapse

Using $s _ { 2 } = \| y \| _ { 2 } ^ { 2 } , t _ { 2 } = \| z \| _ { 2 } ^ { 2 }$ , and $\begin{array} { r } { \ell ( p ) = \log \frac { p } { 1 - p } , } \end{array}$ , the mean-field ODE regularized by the two-class KL penalty is

$$
\dot {p} = p (1 - p) \left\{- \eta \frac {J}{\sigma (p)} p (1 - p) \left(s _ {2} + t _ {2}\right) - \beta (\ell - \ell_ {\mathrm{ref}}) \right\}.
$$

This formulation allows us to characterize the stationary behavior of the system.

Definition G.5 (Nullcline). The interior nullcline $\{ { \dot { p } } = 0 \}$ on the interval (0, 1) is defined by the graph

$$
\beta (\ell (p) - \ell_ {\text { ref }}) = - \eta \frac {J}{\sigma (p)} p (1 - p) (s _ {2} + t _ {2}).\tag{70}
$$

Theorem G.6 (Existence, uniqueness, and stability of the interior equilibrium). For any penalty strength $\beta > 0$ andfixed within-block distributions $( y , z )$ :

1. There exists a unique equilibrium $p ^ { \star } \in ( 0 , 1 )$ satisfying equation $7 0 .$

2. This equilibrium $p ^ { \star }$ is globally asymptotically stable on $( 0 , 1 )$

Proof sketch. On the open interval $( 0 , 1 )$ , the right-hand side of equation $7 0$ is continuous and bounded. In contrast, the log-odds $\mathbf { \dot { \ell } } \ell ( p )$ is strictly increasing, with $\ell ( p ) \to - \infty$ as $p \downarrow 0$ and $\ell ( p ) $ +∞ as $p \uparrow 1$ . Existence and uniqueness follow from the intermediate value theorem and monotonicity. Asymptotic stability is verified by noting that $\partial _ { p } \dot { p } \big ( p ^ { \star } \big ) < 0$ upon linearization. □

Corollary G.7 (KL Regularization prevents collapse for $J < 0 )$ . When $J < 0 ,$ , the right-hand side of equation $\boldsymbol { 7 0 }$ is strictly positive, implying $\ell ( p ^ { \star } ) > \ell _ { \mathrm { r e f } }$ and thus $\bar { p } ^ { \star } > p _ { \mathrm { r e f } }$ . While the unregularized dynamics $( \check { \beta } = 0 )$ would drive $p ( \dot { t } ) \dot { } \uparrow 1$ (leading to a collapse onto the bad-arm block), any $\beta > 0$ ensures the existence of a stable $p ^ { \star } < 1$ Consequently, the long-run accuracy $1 - p ^ { \star }$ remains strictly positive and is necessarily higher than in the $\beta = 0$ regime.

Asymptotic behavior of $p ^ { \star }$ . Letting $\sigma _ { \star } : = \sigma ( p ^ { \star } )$ , we examine the equilibrium under varying KL strengths:

• Strong KL $( \beta  \infty ) :$ Expanding equation 70 around $p _ { \mathrm { r e f } } ,$ we find

$$
p ^ {\star} = p _ {\mathrm{ref}} - \frac {\eta J}{\beta} \frac {\left[ p _ {\mathrm{ref}} (1 - p _ {\mathrm{ref}}) \right] ^ {2}}{\sigma (p _ {\mathrm{ref}})} \left(s _ {2} + t _ {2}\right) + O \Bigl (\frac {1}{\beta^ {2}} \Bigr).\tag{71}
$$

For $J < 0 ,$ , the correction term is positive and scales as $O ( \beta ^ { - 1 } )$ , indicating that $p ^ { \star }$ approaches $p _ { \mathrm { r e f } }$ from above as $\beta$ increases.

• Weak KL $( \beta \downarrow 0 , J < 0 ) \colon$ By setting $\varepsilon : = 1 - p$ and balancing leading-order terms near $p = 1$ , we obtain

$$
1 - p ^ {\star} \sim \frac {\beta}{c} \log \frac {c}{\beta}, \qquad c := - \frac {\eta J}{\sigma (1)} \left(s _ {2} + t _ {2}\right) > 0.\tag{72}
$$

This confirms that any non-zero $\beta$ is sufficient to prevent total collapse $( p ^ { \star } < 1 )$ .

## H Properties of the Simplex

This section collects the geometric and algebraic facts about the probability simplex that we repeatedly use in the main text and appendix. Let $d \geq 2$ be an integer and let $\mathbf { 1 } \in \mathbb { R } ^ { d }$ denote the all-ones vector. We consider the probability simplex

$$
\Delta^ {d - 1} := \left\{\mathbf {p} \in \mathbb {R} _ {\geq 0} ^ {d}: \mathbf {1} ^ {\top} \mathbf {p} = 1 \right\},
$$

whose affine tangent space at any interior point p is

$$
T _ {\mathbf {p}} \Delta^ {d - 1} := \left\{\mathbf {v} \in \mathbb {R} ^ {d}: \mathbf {1} ^ {\top} \mathbf {v} = 0 \right\}.
$$

All definitions and statements below apply verbatim to any simplex-valued variable (e.g. p or y) by renaming. A recurring character in simplex geometry is the matrix

$$
\mathfrak {J} (\mathbf {p}) := \operatorname{Diag} (\mathbf {p}) - \mathbf {p p} ^ {\top},
$$

which simultaneously plays three roles: (i) it is the Jacobian of the softmax map, (ii) it projects directions back to the simplex tangent space, and (iii) it is the inverse of the Shahshahani metric when restricted to the tangent. We build these facts in steps.

Lemma H.1 (Softmax differential is tangent; Jacobian form). Let $\pmb { \mathrm { p } } = \mathrm { s o f t m a x } ( \pmb { \theta } ) \in \Delta ^ { d - 1 }$ with components $p _ { i } ( \pmb { \theta } ) = e ^ { \theta _ { i } } / Z ( \pmb { \theta } )$ , where $\begin{array} { r } { Z ( \pmb { \theta } ) = \sum _ { j = 1 } ^ { d } e ^ { \theta _ { j } } } \end{array}$ . Then

$$
\frac {\partial p _ {i}}{\partial \theta_ {j}} = p _ {i} (\delta_ {i j} - p _ {j}) \quad \Longrightarrow \quad \mathrm{d} \mathbf {p} = \Im (\mathbf {p}) \mathrm{d} \boldsymbol {\theta}.
$$

Moreover, $\Im ( \mathbf { p } ) \mathbf { 1 } = 0$ and $\mathbf { 1 } ^ { \top } \Im ( \mathbf { p } ) = 0 ^ { \top }$ , so Im $\Im ( \mathbf { p } ) \subseteq T _ { \mathbf { p } } \Delta ^ { d - 1 } .$ ; i.e., the softmax differential always lies in the tangent space.

Lemma H.1 already explains why replicator-like dynamics naturally appear when working in logits: any infinitesimal change in θ is automatically mapped to a tangent direction in probability space via $\Im ( \mathfrak { p } )$

Corollary H.2 (Rank, nullspace, and image). When p lies in the interior $( p _ { i } > 0 f o r$ all i), the matrix $\Im ( { \mathfrak { p } } )$ is symmetric positive semidefinite and satisfies

$$
\operatorname{null} \left(\mathfrak {J} (\mathbf {p})\right) = \operatorname{span} \{\mathbf {1} \}, \quad \operatorname{rank} \left(\mathfrak {J} (\mathbf {p})\right) = d - 1, \quad \operatorname{Im} \mathfrak {J} (\mathbf {p}) = T _ {\mathbf {p}} \Delta^ {d - 1}.
$$

In particular, $\Im ( { \mathfrak { p } } )$ acts as a linear automorphism of the tangent space.

The only “forbidden” direction is 1 (which corresponds to shifting all logits by a constant and hence does not change p). On the tangent space, the directions that actually change probabilities, $\Im ( { \mathfrak { p } } )$ is full rank.

Lemma H.3 (A right inverse of J on the tangent). For an interior point p, and for any vector $v \in T _ { \mathbf { p } } \Delta ^ { d - 1 }$

$$
\mathfrak {J} (\mathbf {p}) \operatorname{Diag} (\mathbf {p}) ^ {- 1} \boldsymbol {v} = (I - \mathbf {p} \mathbf {1} ^ {\top}) \boldsymbol {v} = \boldsymbol {v}.
$$

Equivalently, $\mathrm { D i a g } ( { \bf p } ) ^ { - 1 }$ behaves like $\Im ( \mathfrak { p } ) ^ { - 1 }$ once we restrict attention to tangent directions.

Proof. By direct computation,

$$
\mathfrak {J} (\boldsymbol {\mathsf {p}}) \operatorname{Diag} (\boldsymbol {\mathsf {p}}) ^ {- 1} = \left(\operatorname{Diag} (\boldsymbol {\mathsf {p}}) - \boldsymbol {\mathsf {p p}} ^ {\top}\right) \operatorname{Diag} (\boldsymbol {\mathsf {p}}) ^ {- 1} = I - \boldsymbol {\mathsf {p}} \boldsymbol {\mathsf {1}} ^ {\top},
$$

$$
\text { using } \mathbf {p} ^ {\top} \operatorname{Diag} (\mathbf {p}) ^ {- 1} = \mathbf {1} ^ {\top}. \text {   If   } v \in T _ {\mathbf {p}} \Delta^ {d - 1}, \text {   then   } \mathbf {1} ^ {\top} v = 0, \text {   so   } (I - \mathbf {p} \mathbf {1} ^ {\top}) v = v.
$$

Up to now, $\Im ( { \mathfrak { p } } )$ has appeared as a purely algebraic object (a Jacobian with a convenient nullspace). Next we endow the simplex with a Riemannian metric for which $\Im ( { \mathfrak { p } } )$ becomes the natural “inverse metric” on the tangent space.

Definition H.4 (Shahshahani metric). For an interior point p, the Shahshahani inner product on $T _ { \mathbf { p } } \Delta ^ { d - 1 }$ is defined as

$$
\langle \boldsymbol {u}, \boldsymbol {v} \rangle_ {\mathrm{Shah}} := \boldsymbol {u} ^ {\top} \operatorname{Diag} (\mathbf {p}) ^ {- 1} \boldsymbol {v} = \sum_ {i = 1} ^ {d} \frac {u _ {i} v _ {i}}{p _ {i}}, \qquad \boldsymbol {u}, \boldsymbol {v} \in T _ {\mathbf {p}} \Delta^ {d - 1}.\tag{73}
$$

Geometric and information-theoretic intuition. The Shahshahani metric rescales each coordinate by $1 / { \sqrt { p _ { i } } } \mathrm { : }$ moving a small component is “expensive,” and the induced norm

$$
\| \delta \| _ {\mathbf {p}} ^ {2} = \langle \delta , \delta \rangle_ {\mathrm{Shah}} = \sum_ {i = 1} ^ {d} \frac {\delta_ {i} ^ {2}}{p _ {i}}
$$

measures relative (multiplicative) change. A convenient way to visualize geodesics is the square-root embedding $p \mapsto { \sqrt { p } }$ (componentwise), which maps the simplex interior to the positive orthant of the unit sphere. Under this embedding, Shahshahani geodesics become great-circle arcs, and the induced geodesic distance is the Bhattacharyya angle

$$
d _ {\text { Shah }} (x, y) = 2 \arccos \left(\sum_ {i = 1} ^ {d} \sqrt {x _ {i} y _ {i}}\right).\tag{74}
$$

This geometry is also tied to information theory: the metric tensor is the Hessian of the convex potential $\begin{array} { r } { \psi ( \mathbf { p } ) = \sum _ { i = 1 } ^ { d } p _ { i } \log p _ { i } } \end{array}$ (negative Shannon entropy),

$$
\nabla^ {2} \psi (\mathbf {p}) = \operatorname{Diag} \left(\frac {1}{p _ {1}}, \dots , \frac {1}{p _ {d}}\right),\tag{75}
$$

so the forward KL divergence has the local quadratic expansion

$$
D _ {\mathrm{KL}} (\mathbf {p} + \delta \| \mathbf {p}) = \frac {1}{2} \delta^ {\top} \operatorname{Diag} (\mathbf {p}) ^ {- 1} \delta + o (\| \delta \| ^ {2}), \quad \mathbf {1} ^ {\top} \delta = 0,\tag{76}
$$

which is precisely why this metric underlies mirror descent and multiplicative-weights updates.

Once a metric is fixed, gradients become metric-dependent: the Shahshahani gradient is the unique tangent vector whose inner product against any tangent direction matches the directional derivative.

Riemannian gradient under the Shahshahani metric. For a function $F : \Delta ^ { d - 1 } \to \mathbb { R }$ , the Shahshahani gradient $\mathrm { g r a d } _ { \mathrm { S h a h } } ^ { } F ( \mathbf { p } ) \in T _ { \mathbf { p } } \Delta ^ { d - 1 }$ is defined by

$$
\langle \mathrm{grad} _ {\mathrm{Shah}} F (\mathbf {p}), \boldsymbol {u} \rangle_ {\mathrm{Shah}} = \nabla F (\mathbf {p}) ^ {\top} \boldsymbol {u}, \quad \forall \boldsymbol {u} \in T _ {\mathbf {p}} \Delta^ {d - 1},
$$

where $\nabla F ( \mathbf { p } )$ denotes the Euclidean gradient in the ambient space.

The next corollary shows the pleasant surprise: under the Shahshahani metric, the gradient is obtained by applying $\Im ( { \mathfrak { p } } )$ to the Euclidean gradient, so the same matrix that appears in the softmax Jacobian also governs natural-gradient flow on the simplex.

Corollary H.5 (Natural gradient on the simplex). $I f F : \Delta ^ { d - 1 } \to \mathbb { R } i s C ^ { 1 }$ and p is interior, then

$$
\operatorname{grad} _ {\text { Shah }} F (\mathbf {p}) = \Im (\mathbf {p}) \nabla F (\mathbf {p}) = \mathbf {p} \odot (\nabla F (\mathbf {p}) - \langle \mathbf {p}, \nabla F (\mathbf {p}) \rangle \mathbf {1}) \in T _ {\mathbf {p}} \Delta^ {d - 1}.
$$

Proof. Let $g : = \Im ( { \mathfrak { p } } ) \nabla F ( { \mathfrak { p } } )$ . From Lemma H.1, we have $g \in T _ { \mathbf { p } } \Delta ^ { d - 1 }$ . For any $\pmb { u } \in T _ { \mathbf { p } } \Delta ^ { d - 1 }$ , symmetry of $\Im ( { \mathfrak { p } } )$ and Lemma H.3 yield

$$
\langle g, \boldsymbol {u} \rangle_ {\text { Shah }} = g ^ {\top} \operatorname{Diag} (\mathbf {p}) ^ {- 1} \boldsymbol {u} = \nabla F (\mathbf {p}) ^ {\top} \mathfrak {J} (\mathbf {p}) \operatorname{Diag} (\mathbf {p}) ^ {- 1} \boldsymbol {u} = \nabla F (\mathbf {p}) ^ {\top} \boldsymbol {u}.
$$

Thus $g$ satisfies the defining property of $\operatorname { g r a d } _ { \mathrm { S h a h } } F ( { \mathfrak { p } } )$ , proving the claim. The componentwise form follows by expanding $\Im ( { \bf p } ) \nabla F ( { \bf p } ) = \mathrm { D i a g } ( { \bf p } ) \nabla F ( { \bf p } ) - { \bf p } ( { \bf p } ^ { \top } \nabla F ( { \bf p } ) )$ □

The following lemma states that $\Im ( { \mathfrak { p } } )$ is strictly positive on tangent directions, so it can safely serve as an inverse metric (and as a preconditioner) as long as we stay in the simplex interior.

Lemma H.6 (Positive definiteness of J on the tangent). For any interior p and any nonzero $v \in T _ { \mathbf { p } } \Delta ^ { d - 1 }$

$$
\boldsymbol {v} ^ {\top} \mathfrak {J} (\boldsymbol {p}) \boldsymbol {v} = \sum_ {i = 1} ^ {d} p _ {i} v _ {i} ^ {2} - \left(\sum_ {i = 1} ^ {d} p _ {i} v _ {i}\right) ^ {2} = \operatorname{Var} _ {i \sim p} (v _ {i}) > 0.
$$

Thus, $\Im ( { \mathfrak { p } } )$ is symmetric positive definite when restricted to the tangent space.

Remark on boundary points. If some components satisfy $p _ { i } = 0 .$ , the statements remain valid after restricting to the support of p and the corresponding lower-dimensional face (where $\mathrm { D i a g } ( { \bf p } ) ^ { - 1 }$ is welldefined).

So far we have described the geometry (metric) and the resulting notion of gradient (natural gradient). We now connect this viewpoint to the standard KL-regularized update used by mirror descent / multiplicative weights, and show that it matches the Shahshahani natural-gradient flow to first order.

Proposition H.7 (Entropic mirror-ascent on the simplex). For $\pmb { \mathrm { p } } \in \Delta ^ { d - 1 }$ , a vector $A \in \mathbb { R } ^ { d }$ , and a step size $\eta > 0 ,$ the KL-regularized maximization problem

$$
\mathbf {p} ^ {+} = \arg \max _ {\mathbf {q} \in \Delta^ {d - 1}} \left\{\langle A, \mathbf {q} \rangle - \frac {1}{\eta} D _ {\mathrm{KL}} (\mathbf {q} \| \mathbf {p}) \right\}\tag{77}
$$

satisfies the following properties:

(a) Existence and uniqueness. The objective is strictly concave on the relative interior of the face determined by p, hence the maximizer $\mathfrak { p } ^ { + }$ exists and is unique.

(b) Closed-form update (multiplicative weights). Let $\begin{array} { r } { Z : = \sum _ { j = 1 } ^ { d } p _ { j } \exp ( \eta A _ { j } ) } \end{array}$ . Then

$$
p _ {i} ^ {+} = \frac {p _ {i} \exp (\eta A _ {i})}{Z}, \quad i = 1, \dots , d.\tag{78}
$$

Equivalently, log $p _ { i } ^ { + } = \log p _ { i } + \eta A _ { i } - \log Z .$

(c) Trust-region equivalence. For any $\rho > 0$ , there exists $\lambda > 0$ such that $\mathbf { p } ^ { + }$ also solves $\operatorname* { m a x } _ { \mathbf { q } \in \Delta ^ { d - 1 } } \left\{ \left. A , \mathbf { q } \right. \right.$ $D _ { \mathrm { K L } } ( \mathbf { q } \| \mathbf { p } ) \leq \rho \}$ with $\eta = 1 / \lambda$

(d) Optimal value. The maximum value of the objective equals $\begin{array} { r } { \frac { 1 } { \eta } \log Z . } \end{array}$

(e) Invariance and support. The update is invariant under shifts $A \mapsto A + c \mathbf { 1 }$ and does not create new support in one step.

(f) Improvement via Jeffreys divergence. With ${ \bar { A } } : = \langle \mathbf { p } , A \rangle$ and $J _ { \mathrm { K L } } ( \pmb { \mathrm { p } } ^ { + } , \pmb { \mathrm { p } } ) : = D _ { \mathrm { K L } } ( \pmb { \mathrm { p } } ^ { + } | | \pmb { \mathrm { p } } ) + D _ { \mathrm { K L } } ( \pmb { \mathrm { p } } | | \pmb { \mathrm { p } } ^ { + } )$

$$
\eta \langle A, \mathbf {p} ^ {+} - \mathbf {p} \rangle = J _ {\mathrm{KL}} (\mathbf {p} ^ {+}, \mathbf {p}) \geq 0.
$$

(g) First-order expansion (replicator direction). For small $\eta ,$

$$
\mathbf {p} ^ {+} - \mathbf {p} = \eta \mathfrak {J} (\mathbf {p}) A + O (\eta^ {2}).\tag{79}
$$

(h) Local objective gain. $\langle A , \mathbf { p } ^ { + } - \mathbf { p } \rangle = \eta \operatorname { V a r } _ { i \sim p } ( A _ { i } ) + O ( \eta ^ { 2 } )$

Proof. These claims follow from standard Lagrangian optimality conditions and a Taylor expansion of equation 78. In particular, stationarity yields $q _ { i } \propto p _ { i } e ^ { \eta A _ { i } }$ and hence equation 78, while the Jeffreys identity follows by summing $D _ { \mathrm { K L } } ( \mathbf { p } ^ { + } \| \mathbf { p } )$ and $D _ { \mathrm { K L } } ( \mathfrak { p } \Vert \mathfrak { p } ^ { + } )$ ). □

Closing the loop. The final corollary makes the connection explicit: mirror-ascent is an Euler discretization of Shahshahani natural-gradient flow, which is exactly the replicator-form dynamics that appear in our mean-field ODEs.

Corollary H.8 (Natural-gradient interpretation). The mirror-ascent step equation 77 is equivalent, to first order, to an Euler step of the Shahshahani natural-gradientflow:

$$
\dot {\mathbf {p}} = \Im (\mathbf {p}) A = \mathbf {p} \odot (A - \langle \mathbf {p}, A \rangle \mathbf {1}).
$$

## I Inner Dynamics of the Good Arms

In Appendix $C ,$ we established the mean-field dynamics of the good and bad blocks in the explicit $( K { + } M ) .$ arm model (with K good arms and M bad arms), and in particular we introduced the within-block coordinates

$$
p (t) := \sum_ {m = 1} ^ {M} p _ {b _ {m}} (t) \in [ 0, 1 ], \quad y _ {j} (t) := \frac {p _ {j} (t)}{1 - p (t)} (j \leq K), \quad z _ {m} (t) := \frac {p _ {b _ {m}} (t)}{p (t)} (m \leq M).
$$

In this section, we go into the details of the inner good-arm dynamics, i.e. the evolution of $y ( t ) \in \Delta ^ { K - 1 }$ . (The corresponding inner bad-arm dynamics $\mathrm { f o r } z ( t )$ is the sign-reversed analogue and is recorded separately.)

To start, we re-derive the inner good dynamics ODE in a more intuitive way. Let $p _ { i } = \exp ( \theta _ { i } ) / Z$ with

$$
Z = \sum_ {k = 1} ^ {K} \exp (\theta_ {k}) + \sum_ {m = 1} ^ {M} \exp (\theta_ {b _ {m}}), \quad p := \sum_ {m = 1} ^ {M} p _ {b _ {m}},
$$

and define

$$
y _ {j} := \frac {p _ {j}}{1 - p}, \qquad j \in [ K ].
$$

Then

$$
y _ {j} = \frac {\exp (\theta_ {j})}{\sum_ {k = 1} ^ {K} \exp (\theta_ {k})} = \left(\operatorname{softmax} \left(\boldsymbol {\theta} _ {\text { good }}\right)\right) _ {j},\tag{80}
$$

so $y = \left( y _ { 1 } , \cdot \cdot \cdot , y _ { K } \right)$ depends only on $\theta _ { \mathrm { g o o d } }$ (the bad-block logits cancel by normalization).

Lemma I.1 (Pushforward from logits to within-good composition). For any small increment $\begin{array} { r l } { \Delta \pmb { \theta } } & { { } = } \end{array}$ $( \Delta \theta _ { \mathrm { g o o d } } , \Delta \theta _ { \mathrm { b a d } } )$

$$
\Delta y = \left(\mathrm{Diag} (y) - y y ^ {\top}\right) \Delta \pmb {\theta} _ {\mathrm{good}} = y \odot \Bigl (\Delta \pmb {\theta} _ {\mathrm{good}} - \langle y, \Delta \pmb {\theta} _ {\mathrm{good}} \rangle \mathbf {1} \Bigr),\tag{81}
$$

and in particular $\partial y / \partial \theta _ { b _ { m } } = 0 $ for all $m \in [ M ]$ (equivalently, $\partial y / \partial \pmb { \theta } _ { \mathrm { b a d } } = \mathbf { 0 } )$

Proof. From equation 80, write $y _ { j } = \exp ( \theta _ { j } - L )$ with $L : = \log \sum _ { k \leq K } \exp ( \theta _ { k } )$ , so $\Delta y _ { j } = y _ { j } ( \Delta \theta _ { j } - \Delta L )$ and $\begin{array} { r } { \Delta L = \sum _ { k } { < _ { K } y _ { k } \Delta \theta _ { k } } } \end{array}$ . Stacking over j gives equation 81.

Similar to our discussion in Appendix C, assume block symmetry with $A _ { j } = a _ { \mathrm { g } } ( \boldsymbol { p } )$ for $j \le K$ and $A _ { b _ { m } } = a _ { \mathrm { b } } ( p )$ for $m \leq M$ , and set $\Delta r ( p ) = a _ { \mathrm { b } } ( p ) - a _ { \mathrm { g } } ( p )$ . The expected logit step in the good block is

$$
\mathbb {E} [ \Delta \boldsymbol {\theta} _ {\mathrm{good}} ] = \kappa (p) y, \quad \kappa (p) := - \eta p (1 - p) \Delta r (p),\tag{82}
$$

so there is no arm-specific preference in θ-space. Applying Lemma I.1 leads to

$$
\mathbb {E} [ \Delta y ] = \kappa (p) \left(y \odot y - \| y \| _ {2} ^ {2} y\right), \qquad \mathbb {E} [ \Delta y _ {j} ] = \kappa (p) y _ {j} \bigl (y _ {j} - \| y \| _ {2} ^ {2} \bigr).\tag{83}
$$

If $\begin{array} { r } { a _ { \mathrm { g } } ( p ) = \frac { J p } { \sigma ( p ) } } \end{array}$ and $\begin{array} { r } { a _ { \mathrm { b } } ( p ) = - \frac { J ( 1 - p ) } { \sigma ( p ) } } \end{array}$ then $\Delta r ( p ) = - J / \sigma ( p )$ and equation 83 becomes

$$
\mathbb {E} [ \Delta y _ {j} ] = \eta \frac {J}{\sigma (p)} p (1 - p) y _ {j} \Bigl (y _ {j} - \| y \| _ {2} ^ {2} \Bigr),
$$

Consequences. If $J > 0 ,$ , arms with $y _ { j } > \| y \| _ { 2 } ^ { 2 }$ grow while those with $y _ { j } < \| y \| _ { 2 } ^ { 2 }$ shrink (deterministic sharpening within the good block). The fixed points in y are the uniform point and the vertices. If $\Delta r ( p ) < 0$ (informative grader favoring the good block), the uniform point is unstable and the vertices are attracting.<sup>2</sup>

Lemma I.2 (Geometry and Lyapunov structure). Consider the ODE on the simplex

$$
\dot {y} = \kappa (t) \left(y \odot y - \| y \| _ {2} ^ {2} y\right), \quad y = \left(y _ {1}, \dots , y _ {K}\right) \in \Delta^ {K - 1} := \{y \geq 0, \sum_ {i} y _ {i} = 1 \}.
$$

Let $\begin{array} { r } { \tau ( t ) : = \int _ { 0 } ^ { t } \kappa ( s ) } \end{array}$ ds and write $\begin{array} { r } { y ^ { \prime } = \frac { d y } { d \tau } . \left( E . g . \right. } \end{array}$ ., here in our noisy GRPO dynamics, $\begin{array} { r } { \kappa ( t ) \propto \frac { J } { \sigma ( p ( t ) ) } p ( t ) ( 1 - p ( t ) ) } \end{array}$ up to the chosen mean-field time scaling.)

(1) Simplex invariance. $\begin{array} { r } { \sum _ { i } \dot { y } _ { i } = 0 } \end{array}$ , and $i f y _ { i } ( 0 ) \geq 0$ then $y _ { i } ( t ) \geq 0$ for all t. Hence $\Delta ^ { K - 1 }$ is forward invariant.

(2) Gradient form. Define

$$
\mathcal {L} (y) := \frac {1}{3} \sum_ {i = 1} ^ {K} y _ {i} ^ {3} - \frac {1}{4} \left(\sum_ {i = 1} ^ {K} y _ {i} ^ {2}\right) ^ {2} = \frac {1}{3} s _ {3} - \frac {1}{4} s _ {2} ^ {2},
$$

such that $s _ { 2 } ( t ) : = \| y ( t ) \| _ { 2 } ^ { 2 } = \textstyle \sum _ { i } y _ { i } ^ { 2 }$ and $\begin{array} { r } { s _ { 3 } ( t ) : = \sum _ { i } y _ { i } ^ { 3 } . } \end{array}$ , then $\nabla \mathcal { L } ( y ) = ( y _ { i } ^ { 2 } - \| y \| _ { 2 } ^ { 2 } y _ { i } )$ <sub>i</sub> and

$$
\dot {y} = \kappa (t) \nabla \mathcal {L} (y), \quad \frac {d}{d t} \mathcal {L} (y (t)) = \kappa (t) \| \nabla \mathcal {L} (y) \| _ {2} ^ {2}.
$$

In particular, when $J > 0 ,$ , then $\kappa ( t ) \geq 0$ and theflow monotonically ascends L (and descends it when $J < 0 )$

(3) Monotone concentration.

$$
\dot {s} _ {2} = 2 \kappa (t) \left(\sum_ {i} y _ {i} ^ {3} - s _ {2} ^ {2}\right) \geq 0 \quad \text { whenever } \kappa (t) \geq 0,
$$

because by Cauchy–Schwarz, $\begin{array} { r } { \sum _ { i } y _ { i } ^ { 3 } \ge ( \sum _ { i } y _ { i } ^ { 2 } ) ^ { 2 } } \end{array}$ on the simplex, with equality $i f y$ is uniform on its support. Hence $f o r \kappa \geq 0$ the mass generically concentrates while when $\kappa < 0 ,$ s decreases.

(4) Equilibria. Stationary points satisfy $y _ { i } \in \{ 0 , s _ { 2 } \}$ for all i. Thus for any $m \in \{ 1 , \ldots , K \}$ , the points with exactly m nonzero coordinates all equal to 1/m are equilibria (uniform on a support of size m). For $\kappa > 0 ,$ , the $m = \dot { 1 }$ vertices are (Lyapunov) attractors and the others are saddles.

Remark I.3 (Time reparametrization). All phase-portrait statements are independent of t and depend only on the internal time τ. In τ-time the ODE is autonomous:

$$
\frac {d y}{d \tau} = y \odot y - \| y \| _ {2} ^ {2} y = y \odot y - s _ {2} y.\tag{84}
$$

Solutions in physical time are obtained by composing with $\tau ( t ) . ^ { 3 }$

Solving the system from equation 84:

$$
y _ {i} (\tau) = \frac {\frac {d \gamma}{d \tau}}{\frac {1}{y _ {i} (\tau)} - \gamma (\tau)}
$$

where $\gamma ( \tau )$ satisfies $\gamma ( 0 ) = 0$ and

$$
\prod_ {i = 1} ^ {K} \left(\frac {1}{y _ {i} (0)} - \gamma (\tau)\right) = \frac {1}{\prod_ {i = 1} ^ {K} y _ {i} (0)} e ^ {- \tau}.
$$

By the Implicit Function Theorem, there exists a unique strictly increasing function $\gamma : \ [ 0 , \infty ) \ $ $\bigl [ \dot { 0 } , \frac { 1 } { \operatorname* { m a x } _ { 1 \leq i \leq K } y _ { i } ( 0 ) } \bigr )$ with these properties, and it tends to $\begin{array} { r } { \frac { 1 } { \operatorname* { m a x } _ { 1 \leq i \leq K } y _ { i } ( 0 ) } \ \mathrm { a s } \ \tau  \ \infty } \end{array}$ . In particular, assuming we have m highest initial values among $y _ { 1 } ( 0 ) , \ldots , y _ { K } ( 0 ) , { \mathrm { a s ~ } } \tau { \overset {  } { \to } } \infty , y _ { i } ( \tau ) ^ { \prime } { \mathrm { s } }$ s with highest initial values tend to $\textstyle { \frac { 1 } { m } }$ and the rest tend to 0.

We will continue the dynamics of good arms in more details in the following subsection. I.5

## I.1 Evolution of Collision term, s<sub>2</sub>

Lemma I.4 (Evolution and bounds for $s _ { 2 } = \| y \| _ { 2 } ^ { 2 } )$ . In τ-time one has

$$
\frac {d}{d \tau} s _ {2} = 2 (s _ {3} - s _ {2} ^ {2}), \quad \frac {d}{d t} s _ {2} = 2 \kappa (t) (s _ {3} - s _ {2} ^ {2}).
$$

Consequently:

1. (Monotonicity) On the simplex, $s _ { 3 } \ \geq \ s _ { 2 } ^ { 2 }$ with equality iff y is uniform on its support. Hence $s _ { 2 } ( \tau )$ is nondecreasing (strictly, away from uniform-on-support points) when $\dot { \kappa } \geq 0$

2. $( \mathrm { R a n g e } ) \frac { 1 } { K } \leq s _ { 2 } ( \tau ) \leq 1$ . In the multi-bad setting, defining $t _ { 2 } ( \tau ) : = \| z ( \tau ) \| _ { 2 } ^ { 2 } \in [ \frac { 1 } { M } , 1 ] .$ , we have the uniform bound

$$
\frac {1}{K} + \frac {1}{M} \leq s _ {2} (\tau) + t _ {2} (\tau) \leq 2.
$$

3. (Logistic upper differential) Since $0 \leq y _ { i } \leq 1 , s _ { 3 } \leq s _ { 2 } .$ , hence

$$
\frac {d}{d \tau} s _ {2} \leq 2 s _ {2} (1 - s _ {2}), \Rightarrow s _ {2} (\tau) \leq \frac {1}{1 + \left(\frac {1 - s _ {2} (0)}{s _ {2} (0)}\right) e ^ {- 2 \tau}}.
$$

Integrating,

$$
\int_ {0} ^ {\tau} s _ {2} (u) d u \leq \frac {1}{2} \log \left(1 + s _ {2} (0) \left(e ^ {2 \tau} - 1\right)\right).
$$

Corollary I.5 (Exact logit representation and envelopes for p (multi-bad setting)). Let $p ( \tau ) \in ( 0 , 1 )$ be the total bad mass and let $z ( \tau ) \in \dot { \Delta } ^ { M - 1 }$ be the within-bad composition, so $t _ { 2 } ( \tau ) : = \| z ( \tau ) \| _ { 2 } ^ { 2 } \in [ 1 / M , 1 ]$ . In τ-time the p-equation reads

$$
\frac {d p}{d \tau} = - p (1 - p) \left(s _ {2} (\tau) + t _ {2} (\tau)\right).
$$

For the logit $\begin{array} { r } { L ( \tau ) : = \log \frac { p ( \tau ) } { 1 - p ( \tau ) } } \end{array}$

$$
\frac {d L}{d \tau} = - (s _ {2} (\tau) + t _ {2} (\tau)), \quad L (\tau) = L (0) - \int_ {0} ^ {\tau} (s _ {2} (u) + t _ {2} (u)) d u.
$$

Because $s _ { 2 } \in \left[ \frac { 1 } { K } , 1 \right]$ and $t _ { 2 } \in \left[ \frac { 1 } { M } , 1 \right]$ , comparison yields the envelopes

$$
\frac {p _ {0}}{1 - p _ {0}} e ^ {- 2 \tau} \leq \frac {p (\tau)}{1 - p (\tau)} \leq \frac {p _ {0}}{1 - p _ {0}} e ^ {- (\frac {1}{K} + \frac {1}{M}) \tau},
$$

equivalently

$$
\frac {1}{1 + \frac {1 - p _ {0}}{p _ {0}} e ^ {2 \tau}} \leq p (\tau) \leq \frac {1}{1 + \frac {1 - p _ {0}}{p _ {0}} e ^ {\left(\frac {1}{K} + \frac {1}{M}\right) \tau}}.
$$

Corollary I.6 (Hitting-time bracket in internal time (multi-bad setting)). Fix $p _ { \star } \in ( 0 , 1 )$ . The internal time to reach $p ( \dot { \tau _ { \star } } ) = p ,$ <sub>⋆</sub> is bounded by

$$
\frac {1}{2} \log \frac {p _ {0} (1 - p _ {\star})}{(1 - p _ {0}) p _ {\star}} \leq \tau_ {\star} \leq \frac {1}{\frac {1}{K} + \frac {1}{M}} \log \frac {p _ {0} (1 - p _ {\star})}{(1 - p _ {0}) p _ {\star}}.
$$

Thus the factor $\| y \| _ { 2 } ^ { 2 } + \| z \| _ { 2 } ^ { 2 }$ accelerates the decay of logit p by a multiplicative factor between $\textstyle { \frac { 1 } { K } } + { \frac { 1 } { M } }$ (near-uniform within both blocks) and 2 (maximally concentrated within both blocks).

Theorem I.7 (General- $\cdot ( K , M )$ small-heterogeneity expansion). Let $u _ { K } ~ = ~ ( 1 / K , \ldots , 1 / K )$ and $u _ { M } ~ =$ $( 1 / M , \dots , 1 / \dot { M } )$ and write

$$
y = u _ {K} + \boldsymbol {v}, \quad \sum_ {i = 1} ^ {K} v _ {i} = 0, \qquad z = u _ {M} + w, \quad \sum_ {m = 1} ^ {M} w _ {m} = 0.
$$

Define the heterogeneities

$$
\zeta (\tau) := \| \boldsymbol {v} (\tau) \| _ {2} ^ {2} = s _ {2} (\tau) - \frac {1}{K} \geq 0, \quad \zeta_ {0} := \zeta (0),
$$

$$
\xi (\tau) := \| w (\tau) \| _ {2} ^ {2} = t _ {2} (\tau) - \frac {1}{M} \geq 0, \quad \xi_ {0} := \xi (0).
$$

Then in internal time $\tau ,$

$$
\frac {d L}{d \tau} = - (s _ {2} (\tau) + t _ {2} (\tau)) = - (\frac {1}{K} + \frac {1}{M} + \zeta (\tau) + \xi (\tau)), L (\tau) = L (0) - \int_ {0} ^ {\tau} (s _ {2} (u) + t _ {2} (u)) d u.
$$

Moreover:

(i) Linearized ζ-law (good block). One has the identity

$$
\zeta^ {\prime} (\tau) = \frac {2}{K} \zeta (\tau) - 2 \zeta (\tau) ^ {2} + 2 \sum_ {i = 1} ^ {K} v _ {i} (\tau) ^ {3},
$$

and the bound $\textstyle \left| \sum _ { i } v _ { i } ^ { 3 } \right| \le \sum _ { i } | v _ { i } | ^ { 3 } \le \| v \| _ { 2 } ^ { 3 } = \zeta ^ { 3 / 2 }$ . Hence, uniformly while $\zeta ( \tau ) \leq 1$

$$
\zeta^ {\prime} (\tau) = \frac {2}{K} \zeta (\tau) + O \bigl (\zeta (\tau) ^ {3 / 2} \bigr).
$$

(ii) Linearized ξ-law (bad block). Because the within-bad dynamics are the sign-reversed collisionflow, one analogously has

$$
\xi^ {\prime} (\tau) = - \frac {2}{M} \xi (\tau) + 2 \xi (\tau) ^ {2} - 2 \sum_ {m = 1} ^ {M} w _ {m} (\tau) ^ {3},
$$

and $\begin{array} { r } { \left| \sum _ { m } w _ { m } ^ { 3 } \right| \leq \| w \| _ { 2 } ^ { 3 } = \xi ^ { 3 / 2 } } \end{array}$ . Hence, uniformly while $\xi ( \tau ) \leq 1$

$$
\xi^ {\prime} (\tau) = - \frac {2}{M} \xi (\tau) + O \bigl (\xi (\tau) ^ {3 / 2} \bigr).
$$

(iii) Asymptotic forms. There exist constants $C _ { K } , C _ { M } > 0$ (depending only on K and M) such that, for all $\tau \geq 0$ with $\begin{array} { r } { \sqrt { \zeta _ { 0 } } e ^ { \tau / K } \leq \frac { 1 } { 2 } . } \end{array}$ ,

$$
\zeta (\tau) = \zeta_ {0} e ^ {\frac {2}{K} \tau} + R _ {\zeta} (\tau), \qquad | R _ {\zeta} (\tau) | \leq C _ {K} \zeta_ {0} ^ {3 / 2} e ^ {\frac {3}{K} \tau},
$$

and, for all $\tau \geq 0$ with $\begin{array} { r } { \sqrt { \xi _ { 0 } } \le \frac { 1 } { 2 } } \end{array}$

$$
\xi (\tau) = \xi_ {0} e ^ {- \frac {2}{M} \tau} + R _ {\xi} (\tau), \qquad | R _ {\xi} (\tau) | \leq C _ {M} \xi_ {0} ^ {3 / 2} e ^ {- \frac {3}{M} \tau}.
$$

(iv) Impact on the p-logit. Consequently,

$$
\int_ {0} ^ {\tau} s _ {2} (u) d u = \frac {\tau}{K} + \frac {K}{2} \zeta_ {0} \Bigl (e ^ {\frac {2}{K} \tau} - 1 \Bigr) + R _ {I, y} (\tau), \qquad | R _ {I, y} (\tau) | \leq C _ {K} ^ {\prime} \zeta_ {0} ^ {3 / 2} e ^ {\frac {3}{K} \tau},
$$

$$
\int_ {0} ^ {\tau} t _ {2} (u) d u = \frac {\tau}{M} + \frac {M}{2} \xi_ {0} \Big (1 - e ^ {- \frac {2}{M} \tau} \Big) + R _ {I, z} (\tau), \qquad | R _ {I, z} (\tau) | \le C _ {M} ^ {\prime} \xi_ {0} ^ {3 / 2},
$$

and hence

$$
L (\tau) = L (0) - \left(\frac {1}{K} + \frac {1}{M}\right) \tau - \frac {K}{2} \zeta_ {0} \left(e ^ {\frac {2}{K} \tau} - 1\right) - \frac {M}{2} \xi_ {0} \left(1 - e ^ {- \frac {2}{M} \tau}\right) + R _ {L} (\tau),
$$

with $| R _ { L } ( \tau ) | \le C _ { K , M } ^ { \prime \prime } \bigl ( \zeta _ { 0 } ^ { 3 / 2 } e ^ { 3 \tau / K } + \xi _ { 0 } ^ { 3 / 2 } \bigr )$ for a constant $C _ { K , M } ^ { \prime \prime } .$

In particular, to first order the only dependence on the initial within-block states is through $\zeta _ { 0 } = \| y ( 0 ) - u _ { K } \| _ { 2 } ^ { 2 }$ and $\xi _ { 0 } = \| z ( 0 ) - u _ { M } \| _ { 2 } ^ { 2 } ;$ finer details $o f y ( 0 )$ and $z ( 0 )$ enter only at order $O ( \zeta _ { 0 } ^ { 3 / 2 } )$ and $O \big ( \xi _ { 0 } ^ { 3 / 2 } \big )$ and higher.

Corollary I.8 (Small-τ expansion; “only $\left( \zeta _ { 0 } , \xi _ { 0 } \right)$ matters” at first order). Expanding the expression in Theorem I.7 for small τ gives

$$
L (\tau) = L (0) - \left(\frac {1}{K} + \frac {1}{M}\right) \tau - (\zeta_ {0} + \xi_ {0})   \tau - \left(\frac {\zeta_ {0}}{K} - \frac {\xi_ {0}}{M}\right) \tau^ {2} + O (\zeta_ {0}   \tau^ {3}) + O (\xi_ {0}   \tau^ {3}) + O (\zeta_ {0} ^ {3 / 2}   \tau) + O (\xi_ {0} ^ {3 / 2}   \tau).
$$

Thus, to linear order in τ, the correction to the baseline $\begin{array} { r } { - ( \frac { 1 } { K } + \frac { 1 } { M } ) \tau } \end{array}$ is exactly $- ( \zeta _ { 0 } + \xi _ { 0 } ) \tau$ . Equivalently, at leading order,

$$
\frac {d L}{d \tau} = - (s _ {2} (0) + t _ {2} (0)) + h i g h e r - o r d e r c o r r e c t i o n s.
$$

Remark I.9 (Scope and sign). (i) The τ-phase portraits for y are independent of the path of $p ( t )$ (since p only reparametrizes time through τ). (ii) If $\kappa ( t ) \ { \stackrel { \cdot } { \leq } } \ 0 \ ( \mathbf { e . g . } \ J / { \stackrel { \cdot } { \sigma } } ( p ) \leq 0 ) ^ { . }$ , replace $\tau { \mathrm { ~ b y ~ } } { \bar { - } } | \tau |$ to flip directions in physical time; the internal-time identities remain valid. (iii) The window $\sqrt { \zeta _ { 0 } } e ^ { \tau / \bar { K } } \leq \frac { 1 } { 2 }$ describes the regime where the linearized law for $\zeta$ dominates; beyond it, the global envelopes in Cor. I.5 apply.

## I.2 Asymptotic General inner dynamics behavior

Lemma I.10 (Order preservation and winner identity). For any $i \neq j ,$ define $\delta _ { i j } ( \tau ) : = y _ { i } ( \tau ) - y _ { j } ( \tau )$ . Along $\begin{array} { r } { \frac { d y } { d \tau } = y \odot y - s _ { 2 } y , } \end{array}$

$$
\delta_ {i j} ^ {\prime} = \delta_ {i j} (y _ {i} + y _ {j} - s _ {2}), \quad \Rightarrow \quad \delta_ {i j} (\tau) = \delta_ {i j} (0) \exp \left(\int_ {0} ^ {\tau} (y _ {i} + y _ {j} - s _ {2}) d u\right).
$$

Hence sign $\delta _ { i j } ( \tau ) = \mathrm { s i g n } \delta _ { i j } ( 0 )$ for all τ. In particular,

$$
m := \arg \max _ {1 <   i <   K} y _ {i} (0)
$$

remains the unique maximizerfor all $\tau > 0 ,$ , and each hyperplane $\{ y _ { i } = y _ { j } \}$ is invariant.

Theorem I.11 (Global convergence and generic basins). Under equation 84, the trajectory stays in $\Delta ^ { K - 1 }$ and $\mathcal { L }$ is a strict Lyapunov function. The equilibria are

$$
\mathcal {E} = \bigcup_ {m = 1} ^ {K} \mathcal {E} _ {m}, \quad \mathcal {E} _ {m} := \left\{y: y _ {i _ {1}} = \dots = y _ {i _ {m}} = 1 / m, y _ {j} = 0 e l s e \right\}.
$$

For generic initial conditions (no coordinate ties), the trajectory converges to the vertex $\mathbf { e } _ { m }$ selected by Lemma I.10. The non-vertex equilibria (uniform on m-subsupports with $m \geq 2 )$ are saddles with co-dimension one stable manifolds coinciding with unions of the tie sets $\{ y _ { i } = { \overset { \cdot } { y _ { j } } } \}$

Proposition I.12 (Exponential polarization in internal time). Let $m = \arg \operatorname* { m a x } y _ { i } ( 0 )$ and write $\varepsilon _ { i } ( \tau ) : = y _ { i } ( \tau )$ $f o r i \neq m$ . Linearizing equation 84 at the vertex $\mathbf { e } _ { m }$ yields

$$
\varepsilon_ {i} ^ {\prime} = - \varepsilon_ {i} + O (\varepsilon^ {2}), \quad 1 - y _ {m} = \sum_ {i \neq m} \varepsilon_ {i}.
$$

Hence there exists $\tau _ { 0 }$ and constants $c _ { i } > 0$ such that,for all $\tau \geq \tau _ { 0 } .$

$$
y _ {i} (\tau) = c _ {i} e ^ {- \tau} (1 + o (1)), \qquad 1 - y _ {m} (\tau) = \Bigl (\sum_ {i \neq m} c _ {i} \Bigr) e ^ {- \tau} (1 + o (1)), \qquad 1 - s _ {2} (\tau) = \Theta (e ^ {- \tau}).
$$

Remark I.13 (Sharper monotonicity identity). The variance form

$$
s _ {3} - s _ {2} ^ {2} = \sum_ {i = 1} ^ {K} y _ {i} (y _ {i} - s _ {2}) ^ {2} \geq 0
$$

(with equality iff y is uniform on its support) implies $s _ { 2 } ^ { \prime } ( \tau ) = 2 \bigl ( s _ { 3 } - s _ { 2 } ^ { 2 } \bigr ) \geq 0 .$ , with strict increase away from the saddle sets.

## I.3 Stability of the Within–Good Equilibria

Consider our the inner-good ODE

$$
\dot {y} = \kappa (p) (y \odot y - s _ {2} y), \quad s _ {2} = \| y \| _ {2} ^ {2}, \quad \kappa (p) = \frac {J}{\sigma (p)} p (1 - p),\tag{85}
$$

with $y \in \Delta ^ { K - 1 }$ and $p \in ( 0 , 1 )$ treated as quasi–frozen on the slow time scale. We write $y ^ { \star } \in \Delta ^ { K - 1 }$ for an equilibrium of equation 85 and work on the tangent space $T _ { y ^ { \star } } \Delta ^ { K - 1 } = \{ \delta \in \mathbb { R } ^ { K } : \textstyle \sum _ { j } \delta _ { j } = 0 \}$

Lemma I.14 (Jacobian on the simplex tangent). Let $y ^ { \star }$ be an equilibrium ofequation 85 and write $s _ { 2 } ^ { \star } = \| \boldsymbol { y } ^ { \star } \| _ { 2 } ^ { 2 }$ . For perturbations $\stackrel { \cdot } { y } = y ^ { \star } + \delta$ with $\begin{array} { r } { \sum _ { j } \dot { \delta } _ { j } = 0 } \end{array}$ , the linearization is

$$
\dot {\delta} = J _ {y} \delta , \qquad J _ {y} = \kappa (p) \left(\mathrm{diag} (2 y ^ {\star}) - 2 y ^ {\star} (y ^ {\star}) ^ {\top} - s _ {2} ^ {\star} I\right),\tag{86}
$$

where $J _ { y }$ acts on $T _ { y ^ { \star } } \Delta ^ { K - 1 }$ (the subspace orthogonal to 1).

Proposition I.15 (Stability of the uniform equilibrium). Consider the uniform equilibrium $\begin{array} { r } { y ^ { \star } = \frac { 1 } { K } \mathbf { 1 } } \end{array}$ ofequation 85. Then $\begin{array} { r } { s _ { 2 } ^ { \star } = \frac { 1 } { K } } \end{array}$ and the Jacobian reduces to

$$
J _ {y} = \frac {\kappa (p)}{K} \left(I - \frac {2}{K} \mathbf {1 1} ^ {\top}\right).\tag{87}
$$

The eigenstructure is:

• Along the direction 1: a zero eigenvalue (simplex invariance).

• On the (K−1)-dimensional tangent space $T _ { y ^ { \star } } \Delta ^ { K - 1 }$ : eigenvalues $\begin{array} { r } { \lambda = \frac { \kappa ( p ) } { K } } \end{array}$

Hence:

$$
\left\{ \begin{array}{l l} \kappa (p) > 0 (i. e., J > 0): & y ^ {\star} \text {   is   linearly   unstable   (repelling); } \\ \kappa (p) <   0 (i. e., J <   0): & y ^ {\star} \text {   is   asymptotically   stable. } \end{array} \right.
$$

Proposition I.16 (Stability of the pure–arm equilibria). Consider a vertex / pure–arm equilibrium $\boldsymbol { y } ^ { \star } = \boldsymbol { e } _ { j }$ of equation 85, for some $j \in \{ 1 , \ldots , K \}$ . Then $s _ { 2 } ^ { \star } = 1$ and

$$
J _ {y} = \kappa (p) \left(\operatorname{diag} \left(2 e _ {j}\right) - 2 e _ {j} e _ {j} ^ {\top} - I\right).\tag{88}
$$

In coordinates this yields

$$
\dot {\delta} _ {j} = 0, \quad \dot {\delta} _ {i} = - \kappa (p)   \delta_ {i} \quad (i \neq j),
$$

so that perturbations orthogonal to $e _ { j }$ decay or grow according to the sign of $\kappa ( p )$ . In particular:

$$
\left\{ \begin{array}{l l} \kappa (p) > 0: & y ^ {\star} = e _ {j} \text {   is   locally   asymptotically   stable   (winner - take - all); } \\ \kappa (p) <   0: & y ^ {\star} = e _ {j} \text {   is   unstable   (flow   returns   toward   mixed   states). } \end{array} \right.
$$

Since $\begin{array} { r } { \kappa ( p ) = \frac { J } { \sigma ( p ) } p ( 1 - p ) } \end{array}$ and $p ( 1 - p ) > 0$ for $p \in \mathsf { \Gamma } ( 0 , 1 )$ , the sign of the Youden index J dictates the symmetry breaking based:

Corollary I.17 (Stability summary for within–good equilibria). For the inner dynamics equation 85, the stability types of the canonical equilibria are summarized in Table 2.

<table><tr><td>Equilibrium type</td><td>Condition on J</td><td>Stability type</td></tr><tr><td>Uniform  $y_{j} = 1/K$ </td><td>J &gt; 0</td><td>Unstable (diversity collapse)</td></tr><tr><td>Uniform  $y_{j} = 1/K$ </td><td>J &lt; 0</td><td>Stable (diversity preserved)</td></tr><tr><td>Vertex  $y = e_{j}$ </td><td>J &gt; 0</td><td>Stable (specialization)</td></tr><tr><td>Vertex  $y = e_{j}$ </td><td>J &lt; 0</td><td>Unstable (reverts to mixture)</td></tr></table>

Table 2: Stability of the uniform and pure–arm equilibria for the within–good ODE equation 85.

Remark I.18 (Role of the Youden index J). We can see the effect of noise as

$J > 0$ (reward alignment): the uniform mixture is destabilized, pure–arm vertices become attractors, and the good arms polarize; diversity collapses.

$J < 0$ (reward inversion): the uniform mixture is stabilized while vertices are repelling; diversity is maintained.

## I.4 Coupling back to physical time

Let $p ( t ) \in ( 0 , 1 )$ be the total bad mass and define $\begin{array} { r } { \kappa ( t ) = \frac { J } { \sigma ( p ( t ) ) } p ( t ) \bigl ( 1 - p ( t ) \bigr ) } \end{array}$ . Then $d \tau / d t = \kappa ( t )$ and (in the multi-bad model) the coupled equations read

$$
\dot {y} = \kappa (t) \left(y \odot y - s _ {2} y\right), \qquad \dot {p} = - \frac {J}{\sigma (p)} \left[ p (1 - p) \right] ^ {2} \left(s _ {2} + t _ {2}\right), \qquad s _ {2} = \| y \| _ {2} ^ {2}, t _ {2} = \| z \| _ {2} ^ {2}.
$$

In internal time,

$$
\frac {d p}{d \tau} = - p (1 - p) \left(s _ {2} (\tau) + t _ {2} (\tau)\right).
$$

Along the generic $J > 0$ branch we have $s _ { 2 } ( \tau ) \to 1$ (good-block polarization) and $t _ { 2 } ( \tau )  1 / M$ (bad-block mixing), hence

$$
\frac {d p}{d \tau} = - (1 + \frac {1}{M})   p   (1 + o (1)), \qquad p (\tau) = C   e ^ {- (1 + 1 / M) \tau}   (1 + o (1)) \quad (\tau \to \infty),
$$

for some $C > 0$ determined by the initial condition $( \mathrm { e . g . }$ . via the logit identity).

$_ { \mathrm { N e x t , } }$ since

$$
\frac {d t}{d \tau} = \frac {1}{\kappa (t)} = \frac {\sigma (p (\tau))}{J p (\tau) (1 - p (\tau))} = \frac {\sigma (p (\tau))}{J p (\tau)} (1 + o (1)),
$$

the physical-time asymptotics are governed by the local behavior of $\sigma ( p )$ near $p = 0$

Theorem I.19 (Physical-time rates via the local law of $\sigma ( p )$ (multi-bad setting)). Assume $\sigma ( p ) \sim \sigma _ { 0 } p ^ { \gamma }$ as $p \downarrow 0$ with $\sigma _ { 0 } > 0$ and $\gamma \in \mathbb { R }$ . Let $m = \arg \operatorname* { m a x } _ { i } y _ { i } ( 0 )$ (unique) and suppose we are on the $J > 0$ branch so that $p ( t ) \downarrow 0 .$ . Write $\textstyle a : = 1 + { \frac { 1 } { M } }$ (the asymptotic internal-time slope of logit p). Then, generically, as $t \to \infty$ or to afinite absorption time $t _ { \infty } ,$

$$
\textbf {(A)} \gamma <   1: p (t) \asymp t ^ {- 1 / (1 - \gamma)}, \qquad 1 - y _ {m} (t), y _ {i} (t) (i \neq m) \asymp t ^ {- 1 / [ a (1 - \gamma) ]};
$$

$$
\text { (B) } \gamma = 1: p (t) \asymp e ^ {- (a J / \sigma_ {0}) t}, \quad 1 - y _ {m} (t), y _ {i} (t) (i \neq m) \asymp e ^ {- (J / \sigma_ {0}) t};
$$

$$
(\mathbf {C}) \gamma > 1: p (t) \asymp (t _ {\infty} - t) ^ {1 / (\gamma - 1)}, \quad 1 - y _ {m} (t), y _ {i} (t) (i \neq m) \asymp (t _ {\infty} - t) ^ {1 / [ a (\gamma - 1) ]}.
$$

All implicit constants depend only on $( y ( 0 ) , z ( 0 ) , p ( 0 ) , J , \sigma _ { 0 } , \gamma )$ . The power-law exponents for $p ( t )$ are universal (independent of K and M), while the y-rates depend on M through $a = \mathrm { \ i } + 1 / M$ (reducing to the one-bad-arm case when $M = 1 )$

Corollary I.20 (Who wins, and how fast?). For a general initial condition $y ( 0 ) \in \Delta ^ { K - 1 }$ with a unique maximizer m, the winning arm is m. In internal time, the non-winners decay as $e ^ { - \tau }$ ; in physical time, the ratesfollow Theorem I.19.

Corollary I.21 (Sharp τ-envelopes for p). Using the logit $\begin{array} { r } { L ( \tau ) = \log \frac { p ( \tau ) } { 1 - p ( \tau ) } . } \end{array}$

$$
\frac {d L}{d \tau} = - (s _ {2} (\tau) + t _ {2} (\tau)).
$$

On the generic $J > 0$ branch, $s _ { 2 } ( \tau )$ ↑ 1 and $t _ { 2 } ( \tau ) \downarrow 1 / M$ (with exponentially decaying gaps), so

$$
L (\tau) = L (0) - \Big (1 + \frac {1}{M} \Big) \tau + O (1), \qquad p (\tau) = \Theta \big (e ^ {- (1 + 1 / M) \tau} \big).
$$

Thus physical-time rates reduce to integrating $d t / d \tau \sim \sigma ( p ( \tau ) ) / ( J p ( \tau ) )$ , i.e. to the local exponent $\gamma .$

## I.5 Dynamics of y-

Let $q : = y ( 0 ) \in \Delta ^ { K - 1 }$ be the initial composition. Reparametrize time by

$$
\tau (t) = \int_ {0} ^ {t} \kappa (s) d s, \quad \kappa (t) = \frac {J}{\sigma (p (t))} p (t) (1 - p (t)).
$$

In τ-time the inner flow is autonomous:

$$
\frac {d y _ {j}}{d \tau} = y _ {j} (y _ {j} - s _ {2}), \qquad s _ {2} = \sum_ {i} y _ {i} ^ {2}.
$$

Introduce $u _ { j } : = 1 / y _ { j }$ . Then $u _ { j } ^ { \prime } - s _ { 2 } u _ { j } = - 1$ , whose solution $\mathrm { i s } ^ { 4 }$

$$
u _ {j} (\tau) = e ^ {S (\tau)} \left(\frac {1}{q _ {j}} - I (\tau)\right), \quad S (\tau) := \int_ {0} ^ {\tau} s _ {2} (r) d r, \quad I (\tau) := \int_ {0} ^ {\tau} e ^ {- S (s)} d s = \int_ {0} ^ {\tau} e ^ {- \int_ {0} ^ {\tau} \sum_ {i} y _ {i} (r) ^ {2} d r} d s.
$$

notice that based on the simplex constraint $\begin{array} { r } { \sum _ { j = 1 } ^ { K } y _ { j } ( \tau ) = 1 } \end{array}$ , we can eliminate the common factor $e ^ { - S ( \tau ) }$

$$
1 = \sum_ {\ell = 1} ^ {K} y _ {\ell} (\tau) = e ^ {- S (\tau)} \sum_ {\ell = 1} ^ {K} \frac {1}{\frac {1}{q _ {\ell}} - I (\tau)}.\tag{95}
$$

Hence

$$
e ^ {- S (\tau)} = \left[ \sum_ {\ell = 1} ^ {K} \frac {1}{\frac {1}{q _ {\ell}} - I (\tau)} \right] ^ {- 1}.\tag{96}
$$

Substituting this back into the expression for $y _ { j } ( \tau )$ gives

$$
y _ {j} (\tau) = \frac {e ^ {- S (\tau)}}{\frac {1}{q _ {j}} - I (\tau)} = \frac {\frac {q _ {j}}{1 - I (\tau) q _ {j}}}{\sum_ {\ell = 1} ^ {K} \frac {q _ {\ell}}{1 - I (\tau) q _ {\ell}}}, \quad I \in [ 0, y (0) _ {\max}).\tag{97}
$$

<sup>4</sup>Writing it in the standard form $u _ { j } ^ { \prime } ( \tau ) + a ( \tau ) u _ { j } ( \tau ) = b ( \tau )$ , we have $a ( \tau ) = - s _ { 2 } ( \tau ) , b ( \tau ) = - 1$ .. For a linear ODE $u _ { j } ^ { \prime } + a ( \tau ) u _ { j } = b ( \tau )$ , the integrating factor is $\begin{array} { r } { \mu ( \tau ) = \exp \Bigl ( \int _ { 0 } ^ { \tau } a ( r ) d r \Bigr ) } \end{array}$ . Using $ { \boldsymbol { a } } ( \tau ) = - s _ { 2 } ( \tau )$ and the notation $\begin{array} { r } { S ( \tau ) : = \int _ { 0 } ^ { \tau } s _ { 2 } ( r ) } \end{array}$ dr, we obtain $\begin{array} { r } { \mu ( \tau ) \ = \ \mathrm { e x p } \Big ( \int _ { 0 } ^ { \tau } - s _ { 2 } ( r ) d r \Big ) \ = \ e ^ { - S ( \tau ) } } \end{array}$ . Multiplying the ODE by $\mu ( \tau )$ gives

$$
e ^ {- S (\tau)} u _ {j} ^ {\prime} (\tau) - s _ {2} (\tau) e ^ {- S (\tau)} u _ {j} (\tau) = - e ^ {- S (\tau)}.\tag{89}
$$

By the product rule and the definition of S,

$$
\frac {d}{d \tau} \big (e ^ {- S (\tau)} u _ {j} (\tau) \big) = e ^ {- S (\tau)} u _ {j} ^ {\prime} (\tau) - s _ {2} (\tau) e ^ {- S (\tau)} u _ {j} (\tau),\tag{90}
$$

so the left-hand side becomes an exact derivative and the equation reduces to

$$
\frac {d}{d \tau} \big (e ^ {- S (\tau)} u _ {j} (\tau) \big) = - e ^ {- S (\tau)}.\tag{91}
$$

Integrating from 0 to τ yields

$$
e ^ {- S (\tau)} u _ {j} (\tau) - e ^ {- S (0)} u _ {j} (0) = - \int_ {0} ^ {\tau} e ^ {- S (s)} d s.\tag{92}
$$

Since $S ( 0 ) = 0 .$ , we have $e ^ { - S ( 0 ) } = 1$ . Writing the initial composition as $y _ { j } ( 0 ) = q _ { j } ,$ , we have $u _ { j } ( 0 ) = 1 / q _ { j }$ . Thus

$$
e ^ {- S (\tau)} u _ {j} (\tau) = \frac {1}{q _ {j}} - \int_ {0} ^ {\tau} e ^ {- S (s)} d s.\tag{93}
$$

Defining $\begin{array} { r } { I ( \tau ) : = \int _ { 0 } ^ { \tau } e ^ { - S ( s ) } d s . } \end{array}$ , and multiplying both sides by $e ^ { S ( \tau ) }$ , we obtain the explicit solution

$$
u _ {j} (\tau) = e ^ {S (\tau)} \Bigl (\frac {1}{q _ {j}} - I (\tau) \Bigr),\tag{94}
$$

## I.6 Evolution of the collision term $s _ { 2 }$

Throughout, let $q : = y ( 0 ) \in \Delta ^ { K - 1 }$ be the initial within–good composition and $I ( \tau )$ the scalar from Eq. equation $9 { \breve { 7 } } .$ . For $r \in \{ 1 , 2 , 3 \}$ define the moment sums

$$
\mathsf {M} _ {r} (I) := \sum_ {j = 1} ^ {K} \frac {q _ {j} ^ {r}}{(1 - I q _ {j}) ^ {r}} \quad \left(\mathsf {M} _ {1} > 0 \text {   on   } I \in [ 0, 1 / q _ {*})\right), \quad q _ {*} := \max _ {j} q _ {j}.
$$

Lemma I.22 (Exact moment formulas and internal-time map). Under the change of variable $I = I ( \tau )$ from $E q .$ . equation 97,

$$
y _ {j} (\tau (I)) = \frac {\frac {q _ {j}}{1 - I q _ {j}}}{\mathsf {M} _ {1} (I)},\tag{98}
$$

$$
s _ {2} (\tau (I)) = \| y \| _ {2} ^ {2} = \frac {\mathrm{M} _ {2} (I)}{\mathrm{M} _ {1} (I) ^ {2}}, \quad s _ {3} (\tau (I)) = \sum_ {j} y _ {j} ^ {3} = \frac {\mathrm{M} _ {3} (I)}{\mathrm{M} _ {1} (I) ^ {3}},\tag{99}
$$

$$
\tau (I) = \int_ {0} ^ {I} \mathsf {M} _ {1} (z) d z = - \sum_ {j = 1} ^ {K} \log \left(1 - I q _ {j}\right),\tag{100}
$$

$$
S (\tau) := \int_ {0} ^ {\tau} s _ {2} (u) d u = \int_ {0} ^ {I (\tau)} \frac {\mathrm{M} _ {2} (z)}{\mathrm{M} _ {1} (z)} d z.\tag{101}
$$

Proof. equation 97 gives the first line immediately. The formulas for $s _ { 2 }$ and $s _ { 3 }$ follow by summing $y _ { j } ^ { 2 }$ and $y _ { j } ^ { 3 }$ . Since $I ^ { \prime } ( \tau ) = e ^ { - S ( \tau ) } = 1 / \mathsf { M } _ { 1 } ( I )$ , we get $d \tau / d I = \mathsf { M } _ { 1 } ( I )$ , and integrate to obtain $\tau ( I )$ . Finally, $\begin{array} { r } { S ( \tau ) \stackrel { } { = } \int s _ { 2 } d \tau = \int ( \mathsf { M } _ { 2 } / \mathsf { M } _ { 1 } ) d I } \end{array}$

Proposition I.23 (Two-sided integral bounds; refined logit envelope (multi-bad outer bounds)). For all $I \in \mathsf { \bar { [ 0 , 1 / q _ { * } ) } }$

$$
\frac {1}{K} \tau (I) \leq S (\tau (I)) \leq - \log (1 - I q _ {*}), \quad \tau (I) = - \sum_ {j = 1} ^ {K} \log (1 - I q _ {j}).\tag{102}
$$

Define also the bad-block collision integral

$$
T (\tau) := \int_ {0} ^ {\tau} t _ {2} (u) d u \quad \text { with } \quad t _ {2} (u) = \| z (u) \| _ {2} ^ {2} \in \left[ \frac {1}{M}, 1 \right],
$$

so that $\begin{array} { r } { \frac { \tau } { M } \leq T ( \tau ) \leq \tau } \end{array}$ . Then for the logit $\begin{array} { r } { L ( \tau ) = \log \frac { p ( \tau ) } { 1 - p ( \tau ) } } \end{array}$

$$
L (\tau) = L (0) - S (\tau) - T (\tau),\tag{103}
$$

$$
L (0) - 2 \tau \leq L (\tau) \leq L (0) - \left(\frac {1}{K} + \frac {1}{M}\right) \tau ,\tag{104}
$$

$$
L (\tau (I)) \geq L (0) - \tau (I) + \log (1 - I q _ {*}) \quad (i m p l i c i t l o w e r s i d e).\tag{105}
$$

Proof. The bounds on S are as in the one-bad case: Cauchy–Schwarz on $\begin{array} { r } { \{ a _ { j } \} = \{ \frac { q _ { j } } { 1 - I q _ { j } } \} } \end{array}$ gives $\mathsf { M } _ { 1 } ( I ) ^ { 2 } \leq$ $K \mathsf { M } _ { 2 } ( I )$ ; hence $\mathsf { M } _ { 2 } / \mathsf { M } _ { 1 } \geq \mathsf { M } _ { 1 } / K$ . Integrating: $\begin{array} { r } { S = \int ( \mathsf { M } _ { 2 } / \mathsf { M } _ { 1 } ) d I \geq \frac { 1 } { K } \int \mathsf { M } _ { 1 } d I = \tau / K } \end{array}$ . For the upper bound, monotonicity of $t \mapsto { \frac { t } { 1 - I t } }$ implies $\begin{array} { r } { \frac { q _ { j } ^ { 2 } } { ( 1 - I q _ { j } ) ^ { 2 } } \le \frac { q _ { * } } { 1 - I q _ { * } } \cdot \frac { q _ { j } } { 1 - I q _ { j } } , \mathrm { s o } \ \mathsf { M } _ { 2 } \le \frac { q _ { * } } { 1 - I q _ { * } } \mathsf { M } _ { 1 } } \end{array}$ . Integrating yields $\begin{array} { r } { S \le \int \frac { q _ { * } } { 1 - I q _ { * } } d I = - \log ( 1 - I q _ { * } ) } \end{array}$

For the logit, use Corollary I.5 to write $\begin{array} { r } { L ( \tau ) = L ( 0 ) - \int _ { 0 } ^ { \tau } ( s _ { 2 } + t _ { 2 } ) d u = L ( 0 ) - S ( \tau ) - T ( \tau ) } \end{array}$ , and then bound $S ( \bar { \tau } ) \in [ \tau / K , \tau ]$ and $T ( \tau ) \in [ \tau / M , \tau ]$ to get equation 104. For equation 105, combine $T ( \tau ) \leq \tau$ and $S ( \tau ( I ) ) \stackrel { . } { \leq } - \log ( 1 - \stackrel { . } { I q } _ { * } )$

Remark I.24 (What is $^ { \prime \prime } { \mathrm { c o l l i s i o n } } ^ { \prime \prime } ? )$ . Here $\begin{array} { r } { s _ { 2 } = \sum _ { i } y _ { i } ^ { 2 } } \end{array}$ is the usual collision probability on the simplex. The collision gap $s _ { 3 } - s _ { 2 } ^ { 2 } = \textstyle { \sum } _ { i } y _ { i } ( y _ { i } - s _ { 2 } ) ^ { 2 }$ drives the slope: in internal time, $\begin{array} { r } { \frac { d s _ { 2 } } { d \tau } = 2 \big ( s _ { 3 } - s _ { 2 } ^ { 2 } \big ) \geq 0 . } \end{array}$ , with strict increase off the uniform-on-support sets.

Proposition I.25 (Pointwise bounds for $s _ { 2 } ^ { \prime }$ (tighter than logistic)). Let $u ( \tau ) : = \sqrt { s _ { 2 } ( \tau ) } \in [ 1 / \sqrt { K } , 1 ]$ and $y _ { \mathrm { m a x } } ( \tau ) : = \mathrm { m a x } _ { i } y _ { i } ( \tau )$ . Thenfor all $\tau ,$

$$
\text {(Exact)} \quad \frac {d s _ {2}}{d \tau} = 2 \sum_ {i = 1} ^ {K} y _ {i} (y _ {i} - s _ {2}) ^ {2} = 2 (s _ {3} - s _ {2} ^ {2});\tag{106}
$$

$$
(\ell_ {p} \text {upper bound}) \quad \frac {d s _ {2}}{d \tau} \leq 2 (s _ {2} ^ {3 / 2} - s _ {2} ^ {2}) = 2 u ^ {3} (1 - u) \quad (s i n c e s _ {3} \leq s _ {2} ^ {3 / 2});\tag{107}
$$

$$
\text {(support - aware)} \quad \frac {d s _ {2}}{d \tau} \leq 2   y _ {\max} (\tau)   s _ {2} (\tau)   (K   s _ {2} (\tau) - 1), \quad \left[ u s i n g \sum_ {i} (y _ {i} - s _ {2}) ^ {2} = s _ {2} (K s _ {2} - 1) \right].\tag{108}
$$

Moreover, writing $y = u _ { K } + v$ with $\textstyle \sum v _ { i } = 0$ and $\eta : = \| v \| _ { 2 } ^ { 2 } = s _ { 2 } - \frac 1 K$

$$
\frac {d s _ {2}}{d \tau} = 2 \left(\frac {\eta}{K} - \eta^ {2} + \sum_ {i} v _ {i} ^ {3}\right), \quad \Rightarrow \quad 2 \left(\frac {\eta}{K} - \eta^ {2} - \eta^ {3 / 2}\right) \leq \frac {d s _ {2}}{d \tau} \leq 2 \left(\frac {\eta}{K} - \eta^ {2} + \eta^ {3 / 2}\right).\tag{109}
$$

Proof. equation 106 is the variance identity. For equation 107, use $\| y \| _ { 3 } \leq \| y \| _ { 2 }$ to get $s _ { 3 } \ \leq \ s _ { 2 } ^ { 3 / 2 }$ . For equation 108, bound $y _ { i } \leq y _ { \mathrm { { m a x } } }$ in equation 106. For equation 109, expand s<sub>3</sub> around u: $s _ { 3 } = 1 / K ^ { 2 } + 3 \eta / K +$ $\textstyle \sum { v _ { i } ^ { 3 } } ;$ subtract $s _ { 2 } ^ { 2 } = ( \dot { 1 } / K + \eta ) ^ { 2 }$

Corollary I.26 (Implicit envelope for $u ( \tau ) = \sqrt { s _ { 2 } ( \tau ) } )$ . Integrating the differential inequality $u ^ { \prime } ( \tau ) \leq u ( \tau ) ^ { 2 } \big ( 1 -$ $u ( \tau ) )$ from equation 107 yields the implicit bound

$$
\log \frac {u (\tau)}{1 - u (\tau)} - \frac {1}{u (\tau)} \leq \log \frac {u _ {0}}{1 - u _ {0}} - \frac {1}{u _ {0}} + \tau , \quad u _ {0} = \sqrt {s _ {2} (0)} = \| q \| _ {2}.
$$

This dominates the logistic envelope $s _ { 2 } ( \tau ) \leq \big ( 1 + [ ( 1 - s _ { 2 } ( 0 ) ) / s _ { 2 } ( 0 ) ] e ^ { - 2 \tau } \big ) ^ { - 1 }$ whenever $u _ { 0 }$ is close to $1 / \sqrt { K }$ (near-uniform start).

Corollary I.27 (Refined envelopes for $p ( \tau )$ (multi-bad outer bounds)). Using $\begin{array} { r } { L ( \tau ) = L ( 0 ) - \int _ { 0 } ^ { \tau } ( s _ { 2 } + t _ { 2 } ) } \end{array}$ du and Proposition I.23:

$$
\frac {1}{1 + \frac {1 - p _ {0}}{p _ {0}} e ^ {2 \tau}} \leq p (\tau) \leq \frac {1}{1 + \frac {1 - p _ {0}}{p _ {0}} e ^ {(\frac {1}{K} + \frac {1}{M}) \tau}},
$$

and, with the implicit time $I \mapsto \tau ( I )$

$$
p (\tau (I)) \geq \frac {1}{1 + \frac {1 - p _ {0}}{p _ {0}} \exp \left(\tau (I) - \log \left(1 - I q _ {*}\right)\right)}.
$$

The last (implicit) lower envelope becomes tight as the good mass polarizes to the maximizer of ${ \dot { \mathbf { \zeta } } } _ { q . }$

## J Inner Dynamics of the Bad Arms

This section is the companion to Section I. When we keep the M bad arms explicitly (instead of aggregating them into a single virtual bad arm), the within-bad composition obeys the same collision ODE as the withingood composition, but with an overall signflip. As a result, essentially every statement in Section I has a direct bad-block analogue obtained by the substitutions

$$
(y, K) \mapsto (z, M), \quad \kappa \mapsto - \kappa \quad (\text { equivalently }, \tau \mapsto - \tau \text { in   internal   time }).
$$

We record the corresponding results (without reproving them).

## J.1 Within-bad composition and pushforward

Let the policy over K good arms and M bad arms be

$$
\mathbf {p} = (p _ {1}, \ldots , p _ {K}, p _ {b _ {1}}, \ldots , p _ {b _ {M}}) \in \Delta^ {K + M - 1}, \qquad p := \sum_ {m = 1} ^ {M} p _ {b _ {m}} \in [ 0, 1 ].
$$

Define the within-bad normalized composition

$$
z _ {m} := \frac {p _ {b _ {m}}}{p}, \qquad m \in [ M ], \qquad z \in \Delta^ {M - 1}.
$$

Writing $p _ { i } = \exp ( \theta _ { i } ) / Z$ with $\begin{array} { r } { Z = \sum _ { j \le K } e ^ { \theta _ { j } } + \sum _ { m \le M } e ^ { \theta _ { b _ { m } } } } \end{array}$ , the bad normalization cancels the good logits:

$$
z _ {m} = \frac {\exp (\theta_ {b _ {m}})}{\sum_ {\ell = 1} ^ {M} \exp (\theta_ {b _ {\ell}})} = \left(\operatorname{softmax} (\boldsymbol {\theta} _ {\mathrm{bad}})\right) _ {m},\tag{110}
$$

so z depends only on $\theta _ { \mathrm { b a d } }$

Lemma J.1 (Pushforward from logits to within-bad composition). For any small increment $\begin{array} { r l } { \Delta \pmb { \theta } } & { { } = } \end{array}$ $( \Delta \theta _ { \mathrm { g o o d } } , \Delta \theta _ { \mathrm { b a d } } )$

$$
\Delta z = \left(\operatorname{Diag} (z) - z z ^ {\top}\right) \Delta \boldsymbol {\theta} _ {\text { bad }} = z \odot \left(\Delta \boldsymbol {\theta} _ {\text { bad }} - \langle z, \Delta \boldsymbol {\theta} _ {\text { bad }} \rangle \mathbf {1}\right),\tag{111}
$$

and in particular $\partial z / \partial \pmb { \theta } _ { \mathrm { g o o d } } = \mathbf { 0 }$

Proof. Identical to Lemma I.1 with $\left( y , \pmb \theta _ { \mathrm { g o o d } } , K \right)$ replaced by $( z , \theta _ { \mathrm { b a d } } , M )$

## J.2 Bad-block drift: the same collision field with opposite sign

Assume the same block symmetry as in Section I / Appendix C:

$$
A _ {j} = a _ {\mathrm{g}} (p) (j \leq K), \quad A _ {b _ {m}} = a _ {\mathrm{b}} (p) (m \leq M), \quad \Delta r (p) := a _ {\mathrm{b}} (p) - a _ {\mathrm{g}} (p).
$$

Recall the (good-block) scalar from equation 82:

$$
\kappa (p) := - \eta p (1 - p) \Delta r (p).
$$

Then the expected logit drift in the bad block is the sign-reversed analogue of equation 82:

$$
\mathbb {E} \left[ \Delta \boldsymbol {\theta} _ {\text { bad }} \right] = - \kappa (p) z, \quad \text { i.e. } \quad \mathbb {E} \left[ \Delta \theta_ {b _ {m}} \right] = - \kappa (p) z _ {m}.\tag{112}
$$

Proposition J.2 (Within-bad mean drift in z-coordinates). Applying Lemma J.1 to equation 112 yields

$$
\mathbb {E} [ \Delta z ] = - \kappa (p) \left(z \odot z - \| z \| _ {2} ^ {2} z\right), \quad \mathbb {E} [ \Delta z _ {m} ] = - \kappa (p) z _ {m} \left(z _ {m} - \| z \| _ {2} ^ {2}\right).\tag{113}
$$

In the noisy GRPO specialization $\begin{array} { r } { a _ { \mathsf { g } } ( p ) = \frac { J p } { \sigma ( p ) } , a _ { \mathsf { b } } ( p ) = - \frac { J ( 1 - p ) } { \sigma ( p ) } ( s o \Delta r ( p ) = - J / \sigma ( p ) a n d \kappa ( p ) = \eta \frac { J } { \sigma ( p ) } p ( 1 - \eta ) , } \end{array}$ p)), this becomes

$$
\mathbb {E} [ \Delta z _ {m} ] = - \eta \frac {J}{\sigma (p)} p (1 - p) z _ {m} \Big (z _ {m} - \| z \| _ {2} ^ {2} \Big).
$$

Consequences (bad-block smoothing vs. polarization). For $J > 0$ (hence $\kappa ( p ) > 0 ) .$ , the sign in equation 113 implies a smoothing effect: components with $z _ { m } > \| z \| _ { 2 } ^ { 2 }$ shrink while those with $z _ { m } < \| z \| _ { 2 } ^ { 2 }$ grow, pushing z toward uniformity on its support. For $J < 0$ the direction reverses and the bad block polarizes (winner-take-all among bad modes), mirroring the good-block behavior when $J > 0$

## J.3 Internal-time form and direct correspondence with Section I

As in Section I, define the internal time

$$
\tau (t) := \int_ {0} ^ {t} \kappa (p (s)) d s,
$$

so that (in mean-field ODE form) the bad composition satisfies

$$
\frac {d z}{d \tau} = - \Big (z \odot z - \| z \| _ {2} ^ {2} z \Big).\tag{114}
$$

Equivalently, with $\rho : = - \tau$ one has

$$
\frac {d z}{d \rho} = z \odot z - \| z \| _ {2} ^ {2} z,
$$

which is exactly the same autonomous ODE as equation 84 for y, with K replaced by M.

Lemma J.3 (Geometry / Lyapunov structure for the bad block (sign-reversed)). $L e t z ( \tau ) \in \Delta ^ { M - 1 }$ solve equation 114 and define

$$
t _ {2} (\tau) := \| z (\tau) \| _ {2} ^ {2} = \sum_ {m = 1} ^ {M} z _ {m} (\tau) ^ {2}, \quad t _ {3} (\tau) := \sum_ {m = 1} ^ {M} z _ {m} (\tau) ^ {3}.
$$

Then the statements of the “Geometry and Lyapunov structure” lemma in Section I carry over with $\left( y , s _ { 2 } , s _ { 3 } , K \right)$ replaced by $( z , t _ { 2 } , t _ { 3 } , \dot { M } )$ and with all monotonicities reversed. Concretely:

(1) Simplex invariance. $\begin{array} { r } { \sum _ { m } z _ { m } ( \tau ) = 1 } \end{array}$ and $z _ { m } ( \tau ) \geq 0$ are preserved.

(2) Gradientform with opposite sign. With the same potential

$$
\mathcal {L} (z) := \frac {1}{3} \sum_ {m = 1} ^ {M} z _ {m} ^ {3} - \frac {1}{4} \Big (\sum_ {m = 1} ^ {M} z _ {m} ^ {2} \Big) ^ {2}, \qquad \nabla \mathcal {L} (z) = \left(z _ {m} ^ {2} - t _ {2} z _ {m}\right) _ {m},
$$

we have

$$
\frac {d z}{d \tau} = - \nabla \mathcal {L} (z), \quad \frac {d}{d \tau} \mathcal {L} (z (\tau)) = - \| \nabla \mathcal {L} (z) \| _ {2} ^ {2} \leq 0.
$$

(3) Monotone de-concentration.

$$
\frac {d}{d \tau} t _ {2} (\tau) = - 2 \big (t _ {3} (\tau) - t _ {2} (\tau) ^ {2} \big) \leq 0,
$$

with equality $i f f z$ is uniform on its support.

(4) Equilibria. Stationary points are exactly the uniform points on a support of size m: for any m $\in \{ 1 , \ldots , M \}$ any point with exactly m nonzero entries, each equal to $1 / m ,$ is an equilibrium.

## J.4 Stability and global limits (bad-block counterpart of Section I.3)

Write the within-bad mean-field ODE in physical time as

$$
\dot {z} = - \kappa (p (t)) (z \odot z - t _ {2} z), \quad t _ {2} = \| z \| _ {2} ^ {2}, \quad \kappa (p) = \frac {J}{\sigma (p)} p (1 - p) (\text { noisy   GRPO }).\tag{115}
$$

Proposition J.4 (Stability of the uniform and vertex equilibria for the bad block). The stability conclusions of Propositions I.15 and I.16 carry over with $K \mapsto M$ and $\kappa \mapsto - \kappa \colon$

• Uniform equilibrium. For $\begin{array} { r } { z ^ { \star } = \frac { 1 } { M } \mathbf { 1 } , } \end{array}$ , the nontrivial eigenvalues on the simplex tangent space are $\lambda = - \kappa ( p ) / M$ Hence

$$
\left\{ \begin{array}{l l} J > 0 (\kappa (p) > 0): & z ^ {\star} \text {   is   asymptotically   stable   (bad   mass   spreads) }; \\ J <   0 (\kappa (p) <   0): & z ^ {\star} \text {   is   unstable }. \end{array} \right.
$$

<table><tr><td>Equilibrium type</td><td>Condition on J</td><td>Stability type</td></tr><tr><td>Uniform  $z_{m} = 1/M$ </td><td>J&gt;0</td><td>Stable (bad mass diffuses)</td></tr><tr><td>Uniform  $z_{m} = 1/M$ </td><td>J&lt;0</td><td>Unstable</td></tr><tr><td>Vertex  $z = e_{m}$ </td><td>J&gt;0</td><td>Unstable</td></tr><tr><td>Vertex  $z = e_{m}$ </td><td>J&lt;0</td><td>Stable (bad-mode collapse)</td></tr></table>

Table 3: Stability of canonical equilibria for the within–bad ODE equation 115.

• Vertex equilibria. For a vertex $z ^ { \star } = e _ { m } ,$ the transverse modes have eigenvalues $+ \kappa ( p )$ , hence

$$
\int J > 0 (\kappa (p) > 0): z ^ {\star} = e _ {m}
$$

$\begin{array} { r } { \boxed { J < 0 ( \kappa ( p ) < 0 ) : \quad z ^ { \star } = e _ { m } } } \end{array}$ is locally asymptotically stable (bad-mode collapse).

Corollary J.5 (Stability summary for within–bad equilibria). The within-bad stability types are the sign-reversed analogue of Corollary I.17:

Theorem J.6 (Global limit of z (bad-block counterpart of Theorem I.11)). Consider equation 115 with an interior initialization z(0) (all coordinates positive). Then:

$I f J > 0 ,$ , the flow is the reverse of the good-block collision flow in internal time. Consequently $z ( t )$ converges to the uniform point on thefull bad simplex:

$$
z (t) \longrightarrow \frac {1}{M} {\bf 1},
$$

and the collision probability $t _ { 2 } ( t ) = \| z ( t ) \| _ { 2 } ^ { 2 }$ decreases monotonically to $1 / M .$

$I f J < 0$ , the direction reverses and $z ( t )$ follows the forward collisionflow (on $\Delta ^ { M - 1 } )$ . For generic initial conditions $( n o t i e s ) , z ( t )$ converges to the vertex selected by the unique maximizer m = arg $\mathrm { m a x } _ { m } z _ { m } ( 0 )$ (winner-take-all among bad modes).

• If some coordinates $o f z ( 0 )$ are exactly zero, the support is invariant and the same statements hold with M replaced by the support size (uniform-on-support for $J > \dot { 0 } .$ , vertex-on-support for $J < 0 )$

Proof. Immediate from Theorem I.11 by the correspondence equation 114 (time reversal) and $K \mapsto M$

Proposition J.7 (Exponential approach in internal time). The rate statements in Proposition I.12 transfer to z with the same substitutions:

• For $J > 0$ (uniform stable), linearization at $\begin{array} { r } { z ^ { \star } = \frac { 1 } { M } \mathbf { 1 } } \end{array}$ yields

$$
\| z (\tau) - \frac {1}{M} \mathbf {1} \| _ {2} = \Theta \big (e ^ {- \tau / M} \big) (\tau \rightarrow + \infty),
$$

and hence $\begin{array} { r } { t _ { 2 } ( \tau ) - \frac { 1 } { M } = \Theta ( e ^ { - 2 \tau / M } ) } \end{array}$

• For $J < 0$ (vertices stable), after reversing internal time as in the sign discussion of Section I, the non-winners decay as e<sup>−|τ|</sup> toward the winning vertex, exactly as in Proposition I.12 with $K \mapsto M$

Remark J.8 ( scalar closed-form representation (sign-flipped analogue of equation 97)). The explicit “onescalar” representation for $y ( \tau )$ in Section I also carries over to the bad block with the sign flipped in the denominators. If $q : = z ( 0 ) \in \mathring { \Delta } ^ { M - 1 }$ and $I ( \tau )$ is a strictly increasing scalar with $I ( 0 ) = 0 .$ , then

$$
z _ {m} (\tau) = \frac {\frac {q _ {m}}{1 + I (\tau)   q _ {m}}}{\sum_ {\ell = 1} ^ {M} \frac {q _ {\ell}}{1 + I (\tau)   q _ {\ell}}}, \qquad I (\tau) \uparrow \infty   \Rightarrow   z (\tau) \to \frac {1}{M} \mathbf {1}.
$$

This is the direct sign-reversal analogue of equation $9 7 ;$ we omit the derivation.

Remark J.9 (Effect on the total bad-mass drift in the multi-bad model). In the multi-bad setting the total bad mass $p ( t )$ couples to both collision terms: in internal time $\tau ,$

$$
\frac {d p}{d \tau} = - p (1 - p) \big (\| y (\tau) \| _ {2} ^ {2} + \| z (\tau) \| _ {2} ^ {2} \big), \qquad \frac {d}{d \tau} \log \frac {p}{1 - p} = - (\| y \| _ {2} ^ {2} + \| z \| _ {2} ^ {2}).
$$

Thus $\| z \| _ { 2 } ^ { 2 } \in \left[ 1 / M , 1 \right]$ enters only as a bounded multiplicative factor in the decay of logit $p ,$ recovering the aggregated-bad model when $M = 1$ (where $\| z \| _ { 2 } ^ { 2 } \equiv 1 )$ .

## K Shahshahani geometry and the within–good flow

Let $y = ( y _ { 1 } , \dots , y _ { K } ) \in \Delta ^ { K - 1 }$ denote the within–good composition and define

$$
\dot {y} = \kappa (p) \left(y \odot y - \| y \| _ {2} ^ {2} y\right), \quad \kappa (p) := \frac {J}{\sigma (p)} p (1 - p),\tag{116}
$$

where J is the judge–separation, $\sigma ( p ) > 0$ is the group–normalization scale, and $p \in [ 0 , 1 ]$ is the bad–mass. Set $s _ { 2 } : = \| y \| _ { 2 } ^ { 2 } = \bar { \Sigma _ { i } } y _ { i } ^ { 2 }$ and $\begin{array} { r } { s _ { 3 } : = \sum _ { i } y _ { i } ^ { 3 } } \end{array}$

Shahshahani (Fisher) metric on the simplex. On the interior of the simplex $\Delta ^ { K - 1 }$ , the Shahshahani inner product is

$$
\langle u, v \rangle_ {y} = \sum_ {i = 1} ^ {K} \frac {u _ {i} v _ {i}}{y _ {i}} \quad \text { on   the   tangent   space } T _ {y} \Delta^ {K - 1} = \Big \{v \in \mathbb {R} ^ {K}: \sum_ {i} v _ {i} = 0 \Big \}.
$$

For a smooth potential $\phi : \Delta ^ { K - 1 } \to \mathbb { R }$ , the associated natural gradient (steepest ascent in this metric) has the replicator form

$$
\operatorname{grad} _ {\text { Shah }} \phi (y) = y \odot (\nabla \phi (y) - \langle \nabla \phi (y), y \rangle \mathbf {1}).\tag{117}
$$

Indeed, with $w : = y \odot \left( \nabla \phi - c \mathbf { 1 } \right)$ and $c = \langle \nabla \phi , y \rangle$ , we have $\textstyle \sum _ { i } w _ { i } = 0$ and, for any $v \in T _ { y } \Delta ^ { K - 1 } , \langle w , v \rangle _ { y } =$ $\Sigma _ { i } ( \nabla \phi _ { i } - c ) v _ { i } = \langle \nabla \phi , v \rangle$ , which characterizes the Riemannian gradient.

Proposition K.1 (Shahshahani gradient representation). Let $\begin{array} { r } { \Phi ( y ) : = \frac { 1 } { 2 } \| y \| _ { 2 } ^ { 2 } = \frac { 1 } { 2 } \sum _ { i } y _ { i } ^ { 2 } } \end{array}$ . Then equation 116 is the Shahshahani gradient flow of Φ, scaled by $\kappa ( p )$

$$
\dot {y} = \kappa (p) \operatorname{grad} _ {\text { Shah }} \Phi (y).
$$

Equivalently, in coordinates, $\dot { y } _ { i } = \kappa ( p ) y _ { i } \big ( y _ { i } - \| y \| _ { 2 } ^ { 2 } \big )$

Proof. We have $\nabla \Phi ( y ) = y$ and $\langle \nabla \Phi ( y ) , y \rangle = \| y \| _ { 2 } ^ { 2 }$ . Applying equation 117 gives $\mathrm { g r a d } _ { \mathrm { S h a h } } \Phi ( y ) = y \odot ( y -$ $\| y \| _ { 2 } ^ { 2 } \mathbf { 1 } )$ , hence the claim. □

Interpretation (Herfindahl ascent in Fisher units). The potential $\begin{array} { r } { \Phi ( y ) = \frac { 1 } { 2 } \| y \| _ { 2 } ^ { 2 } } \end{array}$ is the (half) Herfindahl–Hirschman concentration index. Thus equation 116 is the steepest way, in Fisher/Shahshahani geometry, to increase concentration when $\kappa ( p ) > 0$ (and to decrease it when $\kappa ( p ) < 0 )$ . The scalar $\begin{array} { r } { \kappa ( p ) = \frac { J } { \sigma ( p ) } p ( 1 - p ) } \end{array}$ gates the time–scale: the inner reshuffling freezes at $p \in \{ 0 , 1 \}$ and is fastest near $\begin{array} { r } { p = \frac { 1 } { 2 } ; } \end{array}$ its sign flips with J. Corollary K.2 (Lyapunov monotonicity and a variance identity). Along any trajectory of equation 116,

$$
\frac {d}{d t} \Phi (y (t)) = \kappa (p (t)) \left(s _ {3} - s _ {2} ^ {2}\right) = \kappa (p (t)) \operatorname{Var} _ {i \sim y} (y _ {i}) \geq 0 \quad w h e n e v e r \kappa \geq 0,\tag{118}
$$

with equality iff y is uniform on its support $( y _ { i } \in \{ 0 , 1 / m \}$ on some subset of size m). Equivalently, $\begin{array} { r } { \frac { d } { d t } \| y \| _ { 2 } ^ { 2 } = } \end{array}$ $2 \kappa ( p ) \mathrm { V a r } _ { i \sim y } ( y _ { i } )$

Proof. By the chain rule, $\begin{array} { r } { \frac { d } { d t } \Phi = \langle \nabla \Phi , \dot { y } \rangle = \kappa \sum _ { i } y _ { i } \big ( y _ { i } - \| y \| _ { 2 } ^ { 2 } \big ) y _ { i } = \kappa \left( s _ { 3 } - s _ { 2 } ^ { 2 } \right) } \end{array}$ . Since $\begin{array} { r } { s _ { 3 } - s _ { 2 } ^ { 2 } = \sum _ { i } y _ { i } ( y _ { i } - } \end{array}$ $s _ { 2 } ) ^ { 2 } \geq 0$ and vanishes exactly at support–uniform points, the claim follows. □

Proposition K.3 (Equilibria, support invariance, and stability). The rest points of equation 116 are exactly the barycenters offaces: for any subset $S \subseteq [ K ]$ of size m, $\begin{array} { r } { y _ { i } ^ { \star } = \frac { 1 } { m } f o r \ : i \in S } \end{array}$ and $y _ { i } ^ { \star } = 0$ otherwise. Moreover:

(i) Support invariance. $I f y _ { i } ( 0 ) = 0 .$ , then $y _ { i } ( t ) \equiv 0$ for all t (since $\dot { y } _ { i } = \kappa y _ { i } ( \cdot ) )$ .

(ii) Stability for $\kappa \ > \ 0$ . The unique asymptotically stable equilibria are the vertices $( m ~ = ~ 1 ) ;$ all higher–dimensional barycenters $( m \geq 2 )$ are saddles/unstable.

(iii) Stability for $\kappa < 0$ . The roles reverse: the full–uniform point $( m = K )$ is the unique asymptotically stable equilibrium; all others are unstable.

Table 4: Training configuration.

<table><tr><td colspan="2">Data Configuration</td></tr><tr><td>Base Model</td><td>Qwen2.5-3B</td></tr><tr><td>Global Batch Size</td><td>16</td></tr><tr><td>Train Steps</td><td>1410</td></tr><tr><td>Total Epochs</td><td>2</td></tr><tr><td colspan="2">Rollout Inference</td></tr><tr><td>Rollout Num per Prompt</td><td>8</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Top-p</td><td>1.0</td></tr><tr><td>Top-k</td><td>-1</td></tr><tr><td>Max Prompt Length</td><td>4000</td></tr><tr><td>Max Response Length</td><td>4000</td></tr><tr><td colspan="2">Actor Training</td></tr><tr><td>PPO Mini Batch Size</td><td>32</td></tr><tr><td>Advantage Estimation Type</td><td>GRPO</td></tr><tr><td>Clipping  $\varepsilon_{low}$ </td><td>0.2</td></tr><tr><td>Clipping  $\varepsilon_{high}$ </td><td>0.2</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning Rate</td><td> $10^{-6}$ </td></tr><tr><td>Weight Decay</td><td>0.1</td></tr><tr><td> $(\beta_1, \beta_2)$ </td><td>(0.9, 0.999)</td></tr><tr><td>Gradient Norm Clipping</td><td>1.0</td></tr><tr><td>Learning Rate Scheduler</td><td>constant</td></tr><tr><td>Warmup Steps</td><td>10</td></tr><tr><td>KL coefficient ( $\beta$ )</td><td>0.0</td></tr><tr><td colspan="2">Evaluation Setup</td></tr><tr><td>Temperature</td><td>0.0</td></tr><tr><td>Top-p</td><td>1.0</td></tr><tr><td>Top-k</td><td>-1</td></tr><tr><td>Max Generation Length</td><td>4000</td></tr></table>

Proof sketch. At equilibrium, $0 = { \dot { y } } _ { i } = \kappa y _ { i } ( y _ { i } - s _ { 2 } )$ implies: either $y _ { i } = 0 \mathrm { o r } y _ { i } = s _ { 2 }$ . If m coordinates are positive, then $\begin{array} { r } { 1 = \sum _ { i } y _ { i } = m s _ { 2 } \Rightarrow y _ { i } = s _ { 2 } = \frac { 1 } { m } } \end{array}$ on the support. For stability, note that equation 116 is a Shahshahani gradient system for the convex $\Phi ,$ scaled by κ. When $\kappa > 0$ the flow ascends Φ and converges to its maximizers on $\Delta ^ { K - 1 }$ , which are precisely the vertices; when κ $< 0$ it descends to the unique minimizer, the full–uniform point. Support invariance follows from the factor $y _ { i }$ in each coordinate. □

## L Hyperparameters and Training Details

Table 4 summarizes the training and evaluation configuration used in all experiments. We fine-tune a Qwen2.5-3B base model for 1410 optimization steps (2 epochs) with a global batch size of 16. For rollout generation, we sample 8 responses per prompt with temperature 1.0 $( { \mathrm { t o p } } { - } p = 1 . 0 , { \mathrm { t o p } } { - } k = - 1 )$ and truncate prompts/responses at a maximum length of 4000 tokens each. For actor optimization, we use GRPO with symmetric PPO clipping $( \varepsilon _ { \mathrm { l o w } } , \varepsilon _ { \mathrm { h i g h } } ) = ( 0 . 2 , 0 . 2 )$ and Adam (learning rate $1 0 ^ { - 6 } ,$ , weight decay 0.1, $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 , 0 . 9 9 9 ) )$ , with gradient-norm clipping 1.0 and a constant learning-rate schedule with 10 warmup steps. Evaluation is performed greedily (temperature 0.0) with the same decoding truncation limits. In our experimental setup, we skipped the KL-regularization term by setting its corresponding coefficient to zero.

## M Noise Injection Pseudocode

```txt
Algorithm 1 Noisy Verifier Wrapper
Inputs: Oracle checker Oracle(·) ∈ {0,1}, target (TPR,FPR)
1: function NOISYCHECK(program)
2:    z ← Oracle(program)    ▷ ground truth
3:    if z = 1 then
4:    r ← Bernoulli(TPR)
5:    else
6:    r ← Bernoulli(FPR)
7:    return r
```

## N Data Sample

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Example Coding Problem: kMarsh

Problem Statement. Solve the following coding problem using the programming language Python:
Mr. K has a rectangular plot of land which may contain marshes where fenceposts cannot be set. He wants you to find the perimeter of the largest rectangular fence that can be built on this land.
For example, in the following  $m \times n = 4 \times 4$  grid, x marks a marsh and . marks good land:
    ....
    ..x.
    ..x.
    x...
If we number the rows and columns starting with 1, there are two main areas that can be fenced: (1,1) - (3,2) and (1,2) - (4,4). The longest perimeter is 10.
Function Description. Complete the function kMarsh in the editor below. It should print either an integer or the word impossible.
kMarsh(grid): • Input: an array of strings that represent the grid.
• Output: an integer representing the largest perimeter, or the string impossible.
Input Format.
• The first line contains two space-separated integers m and n, the grid rows and columns.
• Each of the next m lines contains n characters describing the land: "x" (ASCII 120) if it is a marsh, and "." (ASCII 46) otherwise.
Constraints.
    2 ≤ m, n ≤ 500
Output Format. Print a single integer—the largest perimeter—or impossible if no rectangular fence can be built.
Sample Input 0
4 5
.....
.x.x.
.....
.....
Sample Output 0
14
Explanation 0. The fence can be built around the entire field.
Perimeter = 2(4 - 1) + 2(5 - 1) = 14.
Sample Input 1
</div>

```txt
2 2
.x
x.
Sample Output 1
impossible
Explanation 1. We need a minimum of four corner points to form a fence, hence it is impossible.
Sample Input 2
2 5
......
xxxx.
Sample Output 2
impossible
Explanation 2. The lower row prevents forming a valid rectangle.
The input is provided via stdin, and the solution should print its result to stdout.
Task. Now, solve the problem and return the code.
```

```python
Test Case List (JSON-like format):
[
    {
    'fn_name': None,
    'input': '4 5\n.....\n.x.x.\n.....\n.....\n',
    'output': '14\n',
    'type': 'stdin_stdout'
    },
    {
    'fn_name': None,
    'input': '2 2\n.x\nx.\n',
    'output': 'impossible\n',
    'type': 'stdin_stdout'
    },
    {
    'fn_name': None,
    'input': '2 5\n.....\nxxxx.\n',
    'output': 'impossible\n',
    'type': 'stdin_stdout'
    }
]
```
