# Anytime-valid of-policy inference for contextual bandits

Ian Waudby-Smith<sup>1</sup>, Lili Wu<sup>2</sup>, Aaditya Ramdas<sup>1</sup>, Nikos Karampatziakis<sup>2</sup>, and Paul Mineiro<sup>2</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Microsoft

ianws@cmu.edu, liliwu@microsoft.com, aramdas@cmu.edu, nikosk@microsoft.com, pmineiro@microsoft.com

## Abstract

Contextual bandit algorithms are ubiquitous tools for active sequential experimentation in healthcare and the tech industry. They involve online learning algorithms that adaptively learn policies over time to map observed contexts X<sub>t</sub> to actions A<sub>t</sub> in an attempt to maximize stochastic rewards R . This adaptivity raises interesting but hard statistical inference questions, especially counterfactual ones: for example, it is often of interest to estimate the properties of a hypothetical policy that is diferent from the logging policy that was used to collect the data — a problem known as “of-policy evaluation” (OPE). Using modern martingale techniques, we present a comprehensive framework for OPE inference that relax unnecessary conditions made in some past works (such as performing inference at prespecified sample sizes, uniformly bounded importance weights, constant logging policies, constant policy values, among others), significantly improving on them both theoretically and empirically. Importantly, our methods can be employed while the original experiment is still running (that is, not necessarily post-hoc), when the logging policy may be itself changing (due to learning), and even if the context distributions are a highly dependent time-series (such as if they are drifting over time). More concretely, we derive confidence sequences for various functionals of interest in OPE. These include doubly robust ones for time-varying of-policy mean reward values, but also confidence bands for the entire cumulative distribution function of the of-policy reward distribution. All of our methods (a) are valid at arbitrary stopping times (b) only make nonparametric assumptions, (c) do not require importance weights to be uniformly bounded and if they are, we do not need to know these bounds, and (d) adapt to the empirical variance of our estimators. In summary, our methods enable anytime-valid of-policy inference using adaptively collected contextual bandit data.

## 1 Introduction Introduction

2 1.1 Of-policy inference, confidence intervals, and confidence sequences 2 1.2 Desiderata for anytime-valid of-policy inference 4 1.3 Outline and contributions 6 1.4 Related work . 6 1.5 Notation: supermartingales, filtrations, and stopping times 7

3.3 Sequential testing and anytime $p$ -values for off-policy inference 20
4 Time-uniform inference for the off-policy CDF 22
5 Summary & extensions 24
A Proofs of the main results 31
A.1 A technical lemma 31
A.2 Proof of Theorem 1 31
A.3 Proof of Theorem 2 33
A.4 Proof of Proposition 2 34
A.5 Proof of Proposition 3 34
A.6 Proof of Theorem 3 36
B A causal view of contextual bandits via potential outcomes 41

## 1 Introduction

The so-called “contextual bandit” problem is an abstraction that can be used to describe several problem setups in statistics and machine learning [35, 36]. For example, it generalizes the multi-armed bandit problem by allowing for “contextual” side information, and it can be used to describe many adaptive sequential experiments. The general contextual bandit problem can be described informally as follows: an agent (such as a medical scientist in a clinical trial) views contextual information $X _ { t } \in \mathcal { X }$ for subject t (such as the clinical patient’s demographics, medical history, etc.), takes an action $A _ { t } \in { \mathcal { A } }$ (such as whether to administer a placebo, a low dose, or a high dose), and observes some reward $R _ { t }$ (such as whether their adverse symptoms have subsided). This description is made formal in the protocol for the generation of contextual bandit data in Algorithm 1. In the present paper, no restrictions are placed on the dimensionality or structure of the context and action spaces $\mathcal { X }$ and A beyond them being measurable, but it is often helpful to think about $\mathcal { X }$ as a d-dimensional Euclidean space, and A as t0, 1u for binary treatments, or R for diferent dosages, and so on. Indeed, while highdimensional settings often pose certain challenges in contextual bandits (such as computational ones, or inflated variances), none of these issues will afect the validity of our statistical inference methods. Throughout, we will require that the rewards are real-valued and bounded in r0, 1s — a common assumption in contextual bandits [50, 28] — except for Section 4 where we relax the boundedness constraint.

There are two main objectives that one can study in the contextual bandit setup: (1) policy optimization, and (2) of-policy evaluation (OPE) [36, 35, 11, 12]. Here, a “policy” $\pi ( \boldsymbol { a } \mid \boldsymbol { x } )$ is simply a conditional distribution over actions, such as the probability that patient t should receive various treatments given their context $X _ { t }$ . Policy optimization is concerned with finding policies that achieve high cumulative rewards (typically measured through regret), while of-policy evaluation is concerned with asking the counterfactual question: “how would we have done if we used some policy π instead of the policy that is currently collecting data?”. In this paper, we study the latter with a particular focus on statistical inference in adaptive, sequential environments under nonparametric assumptions.

## 1.1 Of-policy inference, confidence intervals, and confidence sequences

By far the most common parameter of interest in the OPE problem is the expected reward $\nu : =$ $\mathbb { E } _ { \pi } ( R )$ that would result from taking an action from the policy π. This expectation $\nu$ is called the “value” of the policy π. While several estimators for ν have been derived and refined over the years, many practical problems call for more than just a point estimate: we may also wish to quantify the uncertainty surrounding our estimates via statistical inference tools such as confidence intervals (CI).

However, a major drawback of CIs is the fact that they are only valid at fixed and prespecified sample sizes, while contextual bandit data are collected in a sequential and adaptive fashion over time.

We lay out the assumed protocol for the generation of contextual bandit data in Algorithm 1, and in particular, all of our results will assume access to the output of this algorithm, namely the (potentially infinite) sequence of tuples $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { T }$ for $T \in \mathbb { N } \cup \lbrace \infty \rbrace$ . As is standard in OPE, we will always assume that the policy $\pi$ is (almost surely) absolutely continuous with respect to $h _ { t }$ so that $\pi / h _ { t }$ is almost surely finite (without which, estimation and inference are not possible in general). Indeed, this permits many bandit techniques and in principle allows for Thompson sampling since it always assigns positive probability to an action (note that it may not always easy to compute the probability of taking that action via Thompson sampling, but if those probabilities can be computed, they can be used directly within our framework). However, $( h _ { t } ) _ { t = 1 } ^ { \infty }$ cannot be the result of Upper Confidence Bound (UCB)-style algorithms since they take conditionally deterministic actions given the past, violating the absolute continuity of π with respect to $h _ { t }$

In Algorithm 1, the term “exogenously time-varying” simply means that the context and reward distributions at time t can only depend on the past through $X _ { 1 } ^ { t - 1 } \equiv ( X _ { 1 } , \dots , X _ { t - 1 } )$ , and not on the actions taken (or rewards received). Formally, we allow for any joint distribution over $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ 1 as long as

$$
p _ {R _ {t}} (r \mid x, a, \mathcal {H} _ {t - 1}) = p _ {R _ {t}} (r \mid x, a, X _ {1} ^ {t - 1}) \quad \mathrm{and} \quad p _ {X _ {t}} (x \mid \mathcal {H} _ {t - 1}) = p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1}),\tag{1}
$$

where $\mathcal { H } _ { t }$ is all of the history $\sigma \left( ( X _ { i } , A _ { i } , R _ { i } ) _ { i = 1 } ^ { t } \right)$ up until time t. This conditional independence requirement (1) includes as a special case more classical setups where $X _ { t }$ is independent of all else given $A _ { t } ,$ such as those considered in Bibaut et al. [4] or iid scenarios [28], but is strictly more general, since, for example, $( X _ { t } ) _ { t = 1 } ^ { \infty }$ can be a highly dependent time-series. However, we do not go as far as to consider the adversarial setting that is sometimes studied in the context of regret minimization. We impose this conditional independence requirement since otherwise, the interpretation of $\mathbb { E } _ { \pi } ( R _ { t } \mid \mathcal { H } _ { t - 1 } )$ changes depending on which sequence of actions were played by the logging policy. Making matters more concrete, the conditional of-policy value $\mathbb { E } _ { \pi } ( R _ { t } \mid \mathcal { H } _ { t - 1 } )$ at time t is given by

$$
\begin{array}{l} \nu_ {t} := \mathbb {E} _ {\pi} (R _ {t} \mid \mathcal {H} _ {t - 1}) \equiv \int_ {\mathcal {X} \times \mathcal {A} \times \mathbb {R}} r \cdot p _ {R _ {t}} (r \mid a, x, \mathcal {H} _ {t - 1}) \pi (a \mid x) p _ {X _ {t}} (x \mid \mathcal {H} _ {t - 1})   \mathrm{d} x   \mathrm{d} a   \mathrm{d} r \\ = \int_ {\mathcal {X} \times \mathcal {A} \times \mathbb {R}} r \cdot p _ {R _ {t}} (r \mid a, x, X _ {1} ^ {t - 1}) \pi (a \mid x) p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1})   \mathrm{d} x   \mathrm{d} a   \mathrm{d} r, \end{array}\tag{2}
$$

(3)

and the equality (3) follows from (1). Notice that (2) could in principle depend on the logging policies and actions played, but (3) does not. Despite imposing the conditional independence assumption (1), the integral (2) is still a perfectly well-defined functional, and if (1) is not satisfied, then our CS will still cover a quantity in terms of this functional. However, its interpretation would no longer be counterfactual with respect to the entire sequence of actions (only conditional on the past).

While most prior work on OPE in contextual bandits is not written causally in terms of potential outcomes (e.g. [50, 28, 4, 5, 25]), it is nevertheless possible to write down a causal target $\nu _ { t } ^ { \star }$ (i.e. a functional of the potential outcome distribution) and show that it is equal to $\nu _ { t }$ under certain causal identification assumptions. These assumptions resemble the familiar consistency, exchangeability, and positivity conditions that are ubiquitous in the treatment efect estimation literature. Moreover, there is a close relationship between OPE and the estimation of so-called stochastic interventions in causal inference; indeed, they can essentially be seen as equivalent but with slightly diferent emphases and setups. However, given that neither the potential outcomes view nor the stochastic intervention interpretation of OPE are typically emphasized in the contextual bandit literature (with the exception of Zhan et al. [61], who use potential outcomes throughout), we leave this discussion to Appendix B.

To illustrate the shortcomings of CIs for OPE, suppose we run a contextual bandit algorithm and want to see whether π is better than the current state-of-the-art policy $h - \mathrm { e . g }$ . whether $\mathbb { E } _ { \pi } ( R ) >$ $\mathbb { E } _ { h } ( R )$ . (Here, we are implicitly assuming that $\mathbb { E } _ { \pi ^ { \prime } } ( R ) = \mathbb { E } _ { \pi ^ { \prime } } ( R _ { t } \mid \mathcal { H } _ { t - 1 } )$ for any policy $\pi ^ { \prime }$ for the sake of illustration.) Suppose we compute a CI for the value of π based on n samples (for some prespecified $n )$ , and while π seems promising, the CI turns out to be inconclusive (the CI for $\mathbb { E } _ { \pi } ( R )$ includes $\mathbb { E } _ { h } ( R )$ if the latter is known, or the two CIs overlap if the latter is unknown). It is tempting to collect more data, for a total of $n ^ { \prime }$ points, to see if the result is now conclusive; however the resulting sample size $n ^ { \prime }$ is now a data-dependent quantity, rendering the CI invalid. (This could happen more than once.)

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Protocol for the generation of contextual bandit data
// Here,  $T \in N \cup \{\infty\}$ .
for  $t = 1, 2, \ldots, T$  do
    // The agent selects a policy  $h_t$  based on the history  $H_{t-1} \equiv \sigma((X_i, A_i, R_i)^{t-1})_i=1$ .
    $h_t \in H_{t-1}$ .
    // The environment draws a context from an (exogenously time-varying) distribution.
    $X_t \sim p_{X_t}(\cdot)$
    // The agent plays a random action drawn from the selected policy.
    $A_t \sim h_t(\cdot \mid X_t)$ .
    // The environment draws a reward from an (exogenously time-varying) distribution based on the action and context.
    $R_t \sim p_{R_t}(\cdot \mid A_t, X_t)$ .
end for
// Return a (potentially infinite) sequence of contextual bandit data.
return  $(X_t, A_t, R_t)^T_{t=1}$
</div>

Fortunately, there exist statistical tools that permit adaptive stopping in these types of sequential data collection problems: confidence sequences (CSs [9, 34]). A CS is a sequence of confidence intervals, valid at all sample sizes uniformly (and hence at arbitrary stopping times). Importantly for the aforementioned OPE setup, CSs allow practitioners to collect additional data and continuously monitor ${ \mathrm { i t } } ,$ so that the resulting CI is indeed valid at the data-dependent stopped sample size $n ^ { \prime } .$ . More formally, we say that a sequence of intervals $[ L _ { t } , U _ { t } ] _ { t = 1 } ^ { \infty }$ is a CS for the parameter $\theta \in \mathbb { R }$ if

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, \theta \in \left[ L _ {t}, U _ {t} \right]\right) \geqslant 1 - \alpha , \text {or equivalently,} \mathbb {P} \left(\exists t \in \mathbb {N}: \theta \notin \left[ L _ {t}, U _ {t} \right]\right) \leqslant \alpha .\tag{4}
$$

Contrast (4) above with the definition of a CI which states that @n P N, $\mathbb { P } ( \theta \in [ L _ { n } , U _ { n } ] ) \geqslant 1 - \alpha ,$ so that the $\ " \forall n \ "$ is outside the probability Pp¨q rather than inside. A powerful consequence of (4) is that $[ L _ { \tau } , U _ { \tau } ]$ is a valid CI for any stopping time τ. In fact, $[ L _ { \tau } , U _ { \tau } ]$ being a valid CI is not just an implication of (4) but the two statements are actually equivalent; see Howard et al. [24, Lemma 3].

The consequence for the OPE practitioner is that they can continuously update and monitor a CS for the value of π while the contextual bandit algorithm is running, and deploy π as soon as they are confident that it is better than the current state-of-the-art h. Karampatziakis et al. [28] refer to this adaptive policy switching as “gated deployment”, and we will return to this motivating example through the paper. Let us now lay out five desiderata that we want all of our CSs to satisfy.

## 1.2 Desiderata for anytime-valid of-policy inference

Throughout this paper, we will derive methods for of-policy evaluation and inference in a variety of settings — including fixed policies (Section 2), time-varying policies (Section 3), and for entire cumulative distribution functions (Section 4). However, what all of these approaches will have in common is that they will satisfy five desirable properties which we enumerate here.

1. Nonasymptotic: Our confidence sets will satisfy exact coverage guarantees for any sample size, unlike procedures based on the central limit theorem which only satisfy approximate guarantees for large samples.<sup>1</sup>

2. Nonparametric: We will not make any parametric assumptions on the distribution of the contexts, policies, or rewards.

3. Time-uniform / anytime-valid: Our confidence sets will be uniformly valid for all sample sizes, and permit of-policy inference at arbitrary data-dependent stopping times.

4. Adaptive data collection (via online learning): All of our of-policy inference methods will allow for the sequence of logging policies $( h _ { t } ) _ { t = 1 } ^ { \infty }$ to be predictable $\left( \mathrm { i . e . ~ } h _ { t } \right.$ can depend on $\mathcal { H } _ { t - 1 } )$ . In particular $( h _ { t } ) _ { t = 1 } ^ { \infty }$ can be the result of an online learning algorithm.

5. Unknown and unbounded $w _ { \mathrm { m a x } } \mathbf { \cdot }$ In all of our algorithms, the maximal importance weight

$$
w _ {\max} := \operatorname * {e s s   s u p} _ {t \in \mathbb {N}, a \in \mathcal {A}, x \in \mathcal {X}} \frac {\pi (a \mid x)}{h _ {t} (a \mid x)}
$$

can be unknown, and need not be uniformly bounded $\mathrm { ( i . e . ~ } w _ { \mathrm { m a x } }$ can be infinite). Note that we do require that importance weights $\pi ( a \mid x ) / h _ { t } ( a \mid x )$ themselves are finite for each $( t , a , x )$ but their essential supremum need not be. Perhaps surprisingly, even if $w _ { \mathrm { m a x } }$ is infinite, it is still possible for our CSs to shrink to zero-width since they depend only on empirical variances. As an illustrative example, see Proposition 3 for a closed-form CS whose width can shrink at a rate of alog log $\overline { { t / t } }$ as long as the importance-weighted rewards are well-behaved (e.g. in the iid setting, if they have finite second moments).

In addition to the above, we will design procedures so that they have strong empirical performance and are straightforward to compute. While some of these desiderata are quite intuitive and common in statistical inference (such as nonasymptotic, nonparametric, and time-uniform validity, so as to avoid relying on large sample theory, unrealistic parametric assumptions, or prespecified sample sizes), desiderata 4 and 5 are more specific to OPE and have not been satisfied in several prior works as we outline in Sections $2 , 3 ,$ and 4. Given their central importance to our work, let us elaborate on them here.

Why allow for logging policies to be predictable? For the purpose of policy optimization, contextual bandit algorithms are tasked with balancing exploration and exploitation: simultaneously figuring out which policies will yield high rewards (at the expense of trying out suboptimal policies) and playing actions from the policies that have proven efective so far. On the other hand, in adaptive sequential trials, an experimenter might aim to balance context distributions between treatment arms (such as via Efron’s biased coin design [14]) or to adaptively find the treatment policy that yields the most eficient treatment efect estimators [29]. In both cases, the logging policies $( h _ { t } ) _ { t = 1 } ^ { \infty }$ are not only changing over time, but adaptively so based on previous observations. We strive to design procedures that permit inference in precisely these types of adaptive data collection regimes, despite most prior works on of-policy inference for contextual bandits having assumed that there is a fixed, prespecified logging policy [50, 28, 5, 25]. Of course, if a CS or CI is valid under adaptive data collection, they are also valid when fixed logging policies are used instead.

Why not rely on knowledge of $w _ { \mathrm { m a x } } ?$ Related to the previous discussion, it may not be known a priori how the range of the predictable logging policies will evolve over time. Moreover, one can imagine a situation where $\operatorname* { s u p } _ { a , x } \pi ( a \mid x ) / h _ { t } ( a \mid x ) \to \infty$ , even if every individual importance weight $\pi / h _ { t }$ is finite. In such cases, having CSs be agnostic to the value of $w _ { \mathrm { m a x } }$ is essential. However, even if $w _ { \mathrm { m a x } }$ is known, it may be preferable to design CSs that do not depend on this worst-case value. Suppose for the sake of illustration that a logging policy h assigns a novel treatment (denoted by $a = 1 )$ with probability $1 / 5$ and a placebo (denoted by $a = 0 )$ with probability $4 / 5$ for most subjects, except for a small but high-risk subpopulation, who receive the novel treatment with probability $1 / 1 0 0 0$ . To estimate the expected reward of the novel treatment, note that the importance weight for subject t will take on the value $w _ { t } : = 1 / h ( A _ { t } \mid X _ { t } )$ for treatment $A _ { t } \in \{ 0 , 1 \}$ and context $X _ { t } \in \mathcal { X }$ . Despite the fact that most of the importance weights are only 5, and hence most importance-weighted pseudooutcomes $w _ { t } R _ { t }$ will take values in r0, 5s, the worst-case $w _ { \mathrm { m a x } }$ is much larger at 1000. Consequently, we should expect a CS that scales with $w _ { \mathrm { m a x } } = 1 0 0 0$ to be much wider than one that only scales with a quantity like an empirical variance. For these reasons we prefer procedures that depend on an empirical variance term (defined later) rather than the worst-case importance weight $w _ { \mathrm { m a x } }$

## 1.3 Outline and contributions

Our fundamental contribution is in the derivation of CSs for various of-policy parameters, including fixed policy values, time-varying policy values, and quantiles of the of-policy reward distribution. We begin in Section 2 with the most common formulation of the OPE problem: estimating the value ν of a target policy π. Theorem 1 presents time-uniform CSs for $\nu ,$ a result that generalizes and improves upon the current state-of-the-art CSs for ν by Karampatziakis et al. [28]. In Section 3, we consider the more challenging problem of estimating a time-varying average policy value $\nu _ { t }$ , where the distribution of of-policy rewards can change over time in an arbitrary and unknown fashion. In Section 4, we derive CSs for quantiles of the of-policy reward distribution, and in particular, Theorem 3 presents a confidence band for the entire cumulative distribution function (CDF) that is both uniformly valid in time and in the quantiles. For the results of Sections 3 and 4, no other solutions to this problem exist in the literature, to the best of our knowledge. Finally, in Section 5, we summarize our results and describe some natural extensions and implications of them, namely false discovery rate control under arbitrary dependence when evaluating several policies, and diferentially private of-policy inference.

## 1.4 Related work

Throughout the paper, we will draw detailed comparisons to work that is most closely related to ours, i.e. papers that are broadly concerned with estimating policy values and/or the of-policy CDF from contextual bandit data in a model-free setting — here we are using the term “model-free” to mean that no restrictions are placed on the functional form between the rewards R and the actions A nor on the covariates X. Specifically, Table 1 and the preceding text provides a (selective) property-by-property comparison to the directly related works of Karampatziakis et al. [28], Bibaut et al. [4], Zhan et al. [61], and Howard et al. [24], and Table 2 provides a similar comparison to the works of Howard and Ramdas [22], Chandak et al. [5], and Huang et al. [25]. However, there are several other works that focus on fixed-n (i.e. not time-uniform) and asymptotic statistical inference from adaptively collected data (e.g. in the form of multi-armed bandits, contextual bandits, or more general reinforcement learning). For example, Ramprasad et al. [41] develop a bootstrap procedure for estimating policy values under Markov noise with temporal diference learning algorithms, Dimakopoulou et al. [10] perform adaptive inference in the multi-armed bandit setting, Hadad et al. [18] provide asymptotic confidence intervals for treatment efects in adaptive experiments, and Zhang et al. [63] provide distribution-uniform asymptotic procedures for M-estimation from contextual bandit data. Other works that consider more model-based approaches include Zhang et al. [62], Khamaru et al. [33], Shen et al. [48], and Chen et al. [6].

## 1.5 Notation: supermartingales, filtrations, and stopping times

Since all of our results will rely on the analysis of nonnegative (super)martingales, predictable processes, stopping times, and so on, it is worth defining some of these terms before proceeding. Consider a universe of distributions Π on a filtered probability space $( \Omega , { \mathcal { F } } )$ . A single draw from any distribution $P \in \Pi$ results in a sequence $Z _ { 1 } , Z _ { 2 } , . . .$ . of potentially dependent observations. (In the context of this paper, $Z _ { t }$ may represent $( X _ { t } , A _ { t } , R _ { t } )$ , for example, and the distribution $P$ may be induced by the policy, and not specified in advance.) If $Z _ { 1 } , Z _ { 2 } , \ldots$ are independent and identically distributed (iid), we will explicitly say $\mathrm { s o } ,$ , but in general we eschew iid assumptions in this paper.

As is common in the statistics literature, we will use upper-case letters like $Z$ to refer to random variables and lower-case letters z to refer to non-stochastic values in the same space that $Z$ takes values. Let $Z _ { 1 } ^ { t }$ denote the tuple $( Z _ { 1 } , \ldots , Z _ { t } )$ and let $\mathcal { H } \equiv ( \mathcal { H } _ { t } ) _ { t = } ^ { \infty }$ by default represent the data (or “canonical”) filtration, meaning that $\mathcal { H } _ { t } = \sigma ( Z _ { 1 } ^ { t } )$

A sequence of random variables $Y \equiv ( Y _ { t } ) _ { t = 1 } ^ { \infty }$ is called a process if it is adapted to $\mathcal { H } ,$ that is if $Y _ { t }$ is measurable with respect to $\mathcal { H } _ { t }$ for every t. A process Y is predictable if $Y _ { t }$ is measurable with respect to $\mathcal { H } _ { t - 1 }$ — informally $^ { 6 6 } Y _ { t }$ only depends on the past”. A process M is a martingale for $P$ with respect to H if

$$
\mathbb {E} _ {P} \big [ M _ {t} \mid \mathcal {H} _ {t - 1} \big ] = M _ {t - 1}\tag{5}
$$

for all $t \geqslant 1$ . M is a supermartingale for P if it satisfies (5) with $\cdots = ^ { 9 }$ relaxed to $^ { 6 6 } \leqslant ^ { 9 3 }$ . A (super)martingale is called a test (super)martingale if it is nonnegative and $M _ { 0 } = 1 . \mathrm { ~ A ~ }$ process M is called a test (super)martingale for $\mathcal { P } \subset \Pi$ if it is a test (super)martingale for every $P \in { \mathcal { P } }$

Throughout, if an expectation E operator is used without a subscript $P ,$ , or if a boldface $\mathbb { P }$ is used to denote a probability, these are always referring to the distribution of $( X _ { t } , A _ { t } , R _ { t } ) _ { t \geq 1 }$ induced by Algorithm 1 and the logging policies $( h _ { t } ) _ { t = 1 } ^ { \infty }$

An H-stopping time τ is $\mathrm { ~ a ~ N ~ } \cup \left\{ \infty \right\}$ -valued random variable such that $\{ \tau \leqslant t \} \in { \mathcal { H } } _ { t }$ for each $t \geqslant 0 .$ Informally and in the context of this paper, a stopping time can be thought of as a sample size that was chosen based on all of the information $\mathcal { H } _ { t }$ up until time t.

## 2 Warmup: Of-policy inference for constant policy values

This section deals with the case where $\nu _ { t }$ from (2) does not depend on $t ,$ meaning that it is constant as a function of time. We handle the time-varying case in the next section.

We begin by extending a result of Karampatziakis et al. [28, Section 5.2] which applied in the iid setting, meaning that the logging policy h is fixed and the contexts and rewards are assumed to be iid. Their paper derives several CSs for the value $\nu : = \operatorname { \mathbb { E } } _ { A \sim \pi } ( R )$ of the policy π for r0, 1s-bounded rewards $R ,$ but some of their CSs require knowledge of $w _ { \mathrm { m a x } }$ , which we would like to avoid as per our desiderata in Section 1.2. However, their so-called “scalar betting” approach in [28, Section 5.2] makes use of importance-weighted random variables and does not depend on knowing $w _ { \mathrm { m a x } }$ . To elaborate, let $w _ { t }$ be the importance weight for the target policy π versus the logging policy h given by

$$
w _ {t} := \frac {\pi (A _ {t} \mid X _ {t})}{h (A _ {t} \mid X _ {t})},\tag{6}
$$

and let $\phi _ { t } ^ { \mathrm { ( I W - } \ell \mathrm { ) } } : = w _ { t } R _ { t }$ and $\phi _ { t } ^ { \mathrm { ( I W - } u \mathrm { ) } } : = w _ { t } ( 1 - R _ { t } )$ be importance-weighted rewards that will be used to construct lower and upper bounds respectively. Note that $\phi _ { t } ^ { ( \mathrm { I W } - \ell ) }$ is ubiquitous in the bandit and causal inference literatures, and the authors were not concerned with deriving new estimators, but rather new confidence sequences using existing estimators. While $w _ { t } \equiv w _ { t } ( X _ { t } , A _ { t } )$ does depend on both $A _ { t }$ and $X _ { t } ,$ we leave the dependence on them implicit going forward to reduce notational clutter.

Proposition 1 (Scalar betting of-policy CS [28]). Suppose $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ are iid with r0, 1s-valued rewards $( R _ { t } ) _ { t = 1 } ^ { \infty }$ , and the logging policy h is fixed. For each $\nu ^ { \prime } \in [ 0 , 1 ]$ , let $( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) ) _ { t = } ^ { \infty }$ <sub>“1</sub> be any $[ 0 , 1 / \nu ^ { \prime } )$

valued predictable sequence. Then,

$$
L _ {t} ^ {\mathrm{IW}} := \inf \left\{\nu^ {\prime} \in [ 0, 1 ]: \prod_ {i = 1} ^ {t} \left(1 + \lambda_ {i} ^ {L} (\nu^ {\prime}) \cdot \left(\phi_ {t} ^ {(\mathrm{IW} - \ell)} - \nu^ {\prime}\right)\right) <   \frac {1}{\alpha} \right\}\tag{7}
$$

forms a lower $\left( 1 - \alpha \right) - C S f o r \nu _ { ; }$ , meaning $\mathbb { P } ( \forall t \in \mathbb { N } , \ \nu \geqslant L _ { t } ^ { \mathrm { I W } } ) \geqslant 1 - \alpha$ . Similarly, for any $[ 0 , 1 / ( 1 { - } \nu ^ { \prime } ) )$ valued predictable sequence $( \lambda _ { t } ^ { U } ( \nu ^ { \prime } ) ) _ { t = 1 } ^ { \infty }$

$$
U _ {t} ^ {\mathrm{IW}} := 1 - \inf \left\{1 - \nu^ {\prime} \in [ 0, 1 ]: \prod_ {i = 1} ^ {t} \left[ 1 + \lambda_ {i} ^ {U} (\nu^ {\prime}) \cdot \left(\phi_ {t} ^ {(\mathrm{IW} - u)} - (1 - \nu^ {\prime})\right) \right] <   \frac {1}{\alpha} \right\}\tag{8}
$$

forms an upper $( 1 - \alpha ) – C S$ for $\nu ,$ meaning $\mathbb { P } \left( \forall t \in \mathbb { N } , \ \nu \leqslant U _ { t } ^ { \mathrm { I W } } \right) \geqslant 1 - \alpha$ . A two-sided $C S$ can be formed using $[ L _ { t } ^ { \mathrm { I \dot { W } } } , U _ { t } ^ { \mathrm { I \dot { W } } } ] _ { t = 1 } ^ { \infty }$ combined with a union bound.

The above CS $[ L _ { t } ^ { \mathrm { I W } } , U _ { t } ^ { \mathrm { I W } } ] _ { t = 1 } ^ { \infty }$ due to Karampatziakis et al. [28] has a number of desirable properties. Namely, it satisfies the first four of five desiderata in Section 1.2, meaning it is a nonasymptotic, nonparametric, time-uniform confidence sequence that does not require knowledge of $w _ { \mathrm { m a x } }$ . Note that while infima appear in the definitions of $L _ { t } ^ { \mathrm { I W } }$ and $U _ { t } ^ { \mathrm { I W } }$ they are straightforward to compute (e.g. via line or grid search) when the product is quasiconvex in $\nu ^ { \prime } \in [ 0 , 1 ]$ which is often the case as we discuss in Section 2.2. The idea behind Proposition 1 is to show that the product inside the above infima are nonnegative martingales when $\nu ^ { \prime } = \nu$ and then apply Ville’s inequality to it [53]. Our main results in the coming sections use similar techniques albeit with very diferent (super)martingale tailored to diferent problem settings.

We also wish to highlight that indeed, $[ L _ { t } ^ { \mathrm { I W } } , U _ { t } ^ { \mathrm { I W } } ] _ { t = 1 } ^ { \infty }$ forms a valid $( 1 - \alpha )$ -CS regardless of how the sequences $\left( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) \right)$ <sub>t</sub> and $( \lambda _ { t } ^ { U } ( \nu ^ { \prime } ) ) _ { 1 }$ <sub>t</sub> are chosen. Such phenomena are common in martingale-based statistical procedures such as in Waudby-Smith and Ramdas [58] (see also the review paper of Ramdas et al. [40]) and will be seen in several of the results to follow. We will discuss some guiding principles for how to choose these sequences in Remark 1.

## 2.1 Tighter confidence sequences via doubly robust pseudo-outcomes

Here, we generalize and improve upon Proposition 1 in three ways. First, we show that the logging policy h can be replaced by a sequence of predictable logging policies $( h _ { t } ) _ { t = 1 } ^ { \infty }$ so that $h _ { t }$ can be built from the entire history $\mathcal { H } _ { t - 1 }$ up until time t ´ 1 (and in particular, $( h _ { t } ) _ { t = 1 } ^ { \infty }$ can be the result of an online learning algorithm), so that the importance weight $w _ { t }$ at time t is given by

$$
w _ {t} := \frac {\pi (A _ {t} \mid X _ {t})}{h _ {t} (A _ {t} \mid X _ {t})}.\tag{9}
$$

Second, we show how the importance-weighted pseudo-outcomes $( \phi _ { t } ) _ { t = 1 } ^ { \infty }$ can be made doubly robust in the sense of Dud´ık et al. [11, 12]. Indeed, define the lower and upper doubly robust pseudooutcomes $( \phi _ { t } ^ { ( \mathrm { D R - } \ell ) } ) _ { t = 1 } ^ { \infty }$ and $( \phi _ { t } ^ { ( \mathrm { D R - } u ) } ) _ { t = 1 } ^ { \infty }$ given by

$$
\phi_ {t} ^ {(\mathrm{DR-} \ell)} := w _ {t} \cdot \left(R _ {t} - \left[ \hat {r} _ {t} (X _ {t}; A _ {t}) \wedge \frac {k _ {t}}{w _ {t}} \right]\right) + \mathbb {E} _ {a \sim \pi (\cdot | X _ {t})} \left(\hat {r} _ {t} (X _ {t}; a) \wedge \frac {k _ {t}}{w _ {t}}\right),\tag{10}
$$

$$
\phi_ {t} ^ {(\mathrm{DR-} u)} := w _ {t} \cdot \left(1 - R _ {t} - \left[ (1 - \hat {r} _ {t} (X _ {t}; A _ {t})) \wedge \frac {k _ {t}}{w _ {t}} \right]\right) + \mathbb {E} _ {a \sim \pi (\cdot | X _ {t})} \left([ 1 - \hat {r} _ {t} (X _ {t}; a) ] \wedge \frac {k _ {t}}{w _ {t}}\right),\tag{11}
$$

where $\widehat { r } _ { t } ( X _ { t } ; A _ { t } )$ is any r0, 1s-valued predictor of $R _ { t }$ built from $\mathcal { H } _ { t - 1 }$ and $k _ { t }$ is a $\mathbb { R } _ { \geqslant 0 } \cup \{ \infty \}$ -valued tuning parameter built from $\mathcal { H } _ { t - 1 }$ that determines how “doubly robust” $\phi _ { t } ^ { \mathrm { D R } }$ should be. Note that $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) }$ and $\bar { \phi } _ { t } ^ { ( \mathrm { D R } - u ) }$ are both at least $- k _ { t }$ by construction, and have conditional means of ν and $1 - \nu$ respectively. Note that the phrase “doubly robust” is sometimes used to refer to properties of estimators (e.g. that their bias is second order and depends only on products of nuisance errors in observational studies where importance weights are unknown [32]) and sometimes to refer to types $o f$ estimators that enjoy variance-reduction without compromising validity in experiments where importance weights are known. We are using this phrase in the second sense following the conventions of Dud´ık et al. [11, 12].

Similar to the discussion surrounding Proposition 1, the doubly robust pseudo-outcomes in (10) are ubiquitous in the causal inference and bandit literatures [44, 11, 12, 51] with the minor tweak that we are truncating the reward predictor. Note that we are not doing this for the purposes of deriving better estimators — instead, we are doing so for the purposes of sharp concentration of measure in the pursuit of tighter CSs.

Setting $k _ { 1 } = k _ { 2 } = \cdot \cdot \cdot = 0$ recovers the IW outcomes exactly, while setting $k _ { 1 } = k _ { 2 } = \cdot \cdot \cdot = \infty$ recovers the classic doubly robust outcomes [11, 12] (this could also be achieved by setting $k _ { 1 } = k _ { 2 } =$ $\cdots = w _ { \mathrm { m a x } } ,$ provided $w _ { \mathrm { m a x } }$ is finite and known). We discuss the need to truncate $\widehat { r } _ { t }$ in Remark 2, but the motivation to include a reward predictor at all is to reduce the variance of $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) }$ and $\phi _ { t } ^ { \left( \mathrm { D R } - u \right) }$ if $R _ { t }$ can be well-predicted by $\widehat { r } _ { t } ,$ a well-known phenomenon in doubly robust estimation [44, 52, 7]. Indeed, we find that the resulting CSs are able to adapt to this reduced variance accordingly for large t (see Figure 1 for an illustration).

Third and finally, we relax the iid assumption, and only require that $\mathbb { E } _ { \pi } ( R _ { t } \mid \mathcal { H } _ { t - 1 } ) = \nu \equiv \mathbb { E } _ { \pi } ( R _ { t } )$ and $R _ { t } ~ \in ~ [ 0 , 1 ]$ almost surely. This relaxation of assumptions can be obtained for free, without any change to the resulting CSs whatsoever, and with only a slight modification to the proof. We summarize our extensions in the following theorem.

Theorem 1 (Doubly robust betting of-policy CS). Suppose $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ is an infinite sequence $o f$ contextual bandit data generated by the predictable policies $( h _ { t } ) _ { t = 1 } ^ { \infty }$ whose r0, 1s-valued reward $R _ { t }$ at time t has conditional mean $\mathbb { E } _ { \pi } ( R _ { t } \ | \ \mathcal { H } _ { t - 1 } ) = \nu \equiv \mathbb { E } _ { \pi } ( R _ { t } )$ . For any predictable sequence $( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) ) _ { t = 1 } ^ { \infty }$ such that $\lambda _ { t } ^ { L } ( \nu ^ { \prime } ) \in [ 0 , ( \nu ^ { \prime } + k _ { t } ) ^ { - 1 } )$ , we have

$$
L _ {t} ^ {\mathrm{DR}} := \inf \left\{\nu^ {\prime} \in [ 0, 1 ]: \prod_ {i = 1} ^ {t} \left[ 1 + \lambda_ {i} ^ {L} (\nu^ {\prime}) \cdot \left(\phi_ {i} ^ {(\mathrm{DR} - \ell)} - \nu^ {\prime}\right) \right] <   \frac {1}{\alpha} \right\}\tag{12}
$$

forms a lower $( 1 - \alpha ) – C S f o r \nu .$ Similarly, i $f \lambda _ { t } ^ { U } ( \nu ^ { \prime } ) \in \left[ 0 , ( 1 - \nu ^ { \prime } + k _ { t } ) ^ { - 1 } \right)$ is predictable, then

$$
U _ {t} ^ {\mathrm{DR}} := 1 - \inf \left\{1 - \nu^ {\prime} \in [ 0, 1 ]: \prod_ {i = 1} ^ {t} \left[ 1 + \lambda_ {i} ^ {U} (\nu^ {\prime}) \cdot \left(\phi_ {i} ^ {(\mathrm{DR} - u)} - (1 - \nu^ {\prime})\right) \right] <   \frac {1}{\alpha} \right\}\tag{13}
$$

forms an upper $( 1 - \alpha ) – C S f o r \ \nu$

The proof of Theorem 1 can be found in Appendix A.2 and relies on applying Ville’s inequality [53] to the products in (12) and (13). Note that the dimensionality of X does not in any way afect the validity of Theorem 1. Moreover, $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) }$ and $\phi _ { t } ^ { \left( \mathrm { D R } - u \right) }$ are always unbiased and yield valid CSs regardless of how $\widehat { r } _ { t }$ is chosen. The reason to introduce these doubly robust pseudo-outcomes is to obtain lower-variance CSs (as illustrated in Figure 1) since doubly robust estimators can be semiparametric eficient thereby attaining the optimal asymptotic mean squared error in a local minimax sense. These details are outside the scope of the present paper, but we direct the interested reader to Kennedy [32] and Uehara et al. [51] for modern reviews discussing this subject.

Notice that Theorem 1 is a generalization of Proposition 1. Indeed, if the logging policies do not change (i.e. $h _ { 1 } = h _ { 2 } = \cdot \cdot \cdot = h )$ , and if the observations $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ are iid, and if $k _ { t } = 0$ for each $t ,$ then Theorem 1 recovers Proposition 1 exactly. For this reason, we do not elaborate on empirical comparisons between Proposition 1 and Theorem 1 — any CS that can be derived using the former is a special case of the latter. Moreover, in the on-policy setting with all importance weights set to 1 and without a reward predictor, Theorem 1 recovers the betting-style CSs of Waudby-Smith and Ramdas [58, Theorem $3 ]$ . As alluded to in the discussion following Proposition 1, the infima above are straightforward to compute for many choices of predictable sequences $( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) ) _ { t = \cdot } ^ { \infty }$ and $( \lambda _ { t } ^ { U } ( \nu ^ { \prime } ) ) _ { t = } ^ { \infty }$ “1 including all of those discussed in the following section.

Confidence sequences and their widths when the policy value is $\nu = 0 . 6$


Confidence sequences and their widths when the policy value is $\nu = 0 . 1$

Figure 1: Three confidence sequences for a policies with values $\nu = 0 . 6$ and $\nu = 0 . 1$ . The first CS is built from importance-weighted pseudo-outcomes $\left(  { \mathrm { \Sigma } } ^ { \left( 6 \right)} \mathrm { I W } ^ { \prime 3 }  \right.$ , and the other two are built from doubly robust pseudo-outcomes $\mathrm { ( ^ { 6 6 } R ^ { 7 } ) }$ with k taking values 1 and 2, respectively. In these examples, the reward $R _ { t }$ can be predicted easily, a property that only the doubly robust CSs can exploit. Notice that a larger value of k allows the doubly robust CS to become narrower for large t, but it pays for this adaptivity with wider bounds at small t. Nevertheless, all three CSs are time-uniform, and nonasymptotically valid in both simulation scenarios.

## 2.2 Tuning, truncating, and mirroring

We make three remarks below, that are important on both theoretical and practical fronts.

Remark 1 (Tuning $( \lambda _ { t } ^ { L } ) _ { t = 1 } ^ { \infty }$ and $\left( k _ { t } \right) _ { t = 1 } ^ { \infty } )$ . As stated, Theorem 1 yields a valid lower CS for ν using any predictable sequence of $[ 0 , ( \nu ^ { \prime } + k _ { t } ) ^ { - 1 } )$ -valued tuning parameters $\left( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) \right)$ — referred to as $\mathrm { \Delta ^ { 6 6 } t s ^ { 7 } }$ by Karampatziakis et al. [28] and Waudby-Smith and Ramdas [58], but how should these bets be chosen? Waudby-Smith and Ramdas [58, Appendix B] discuss several possible options, but in practice none of them uniformly dominate the others. (This should not be surprising, since there is a certain formal sense in which diferent nontrivial nonnegative martingales cannot uniformly dominate each other for a given composite sequential testing problem; see Ramdas et al. [39] for a precise statement.) For a simple-to-implement and empirically compelling option, we suggest scaling $\bar { \phi } _ { t } ^ { \mathrm { { D R } } }$ as $\xi _ { t } : = \phi _ { t } ^ { \mathrm { D R } } / ( k _ { t } + 1 )$ and setting $\lambda _ { t } ^ { L } ( \nu ^ { \prime } )$ as

$$
\lambda_ {t} ^ {L} (\nu^ {\prime}) := \sqrt {\frac {2 \log (1 / \alpha)}{\hat {\sigma} _ {t - 1} ^ {2} t \log (1 + t)}} \wedge \frac {c}{k _ {t} + \nu^ {\prime}}, \quad \mathrm{where}\tag{14}
$$

$$
\widehat {\sigma} _ {t} ^ {2} := \frac {\sigma_ {0} ^ {2} + \sum_ {i = 1} ^ {t} (\xi_ {i} - \bar {\xi} _ {i}) ^ {2}}{t + 1}, \quad \text { and } \quad \bar {\xi} _ {t} := \left(\frac {1}{t} \sum_ {i = 1} ^ {t} \xi_ {i}\right) \wedge \frac {1}{k _ {t} + 1},\tag{15}
$$

with a similar definition for $\lambda _ { t } ^ { U } ( \nu ^ { \prime } )$ but with $c / ( k _ { t } + \nu ^ { \prime } )$ replaced by $c / ( k _ { t } + ( 1 - \nu ^ { \prime } ) )$ . Here, $c \in ( 0 , 1 )$ is some truncation scale, reasonable values of which may lie between $1 / 4$ and $3 / 4 .$ , but it is of relatively minor practical importance, and for suficiently large t, the choice of c will be inconsequential. A justification for why $\lambda _ { t } ^ { L } ( \nu ^ { \prime } )$ is a sensible choice can be found in Waudby-Smith and Ramdas [58, Section B.1], but the practitioner is nevertheless free to use any other sequence of bets, as long as they are predictable and satisfy the aforementioned boundedness constraints. Furthermore, as in Waudby-Smith and Ramdas [58], when $\lambda _ { t } ^ { L } ( \nu ^ { \prime } )$ is chosen as above, the product in (12) is quasiconvex in $\nu ^ { \prime } \in [ 0 , 1 ]$ and hence the infima in Theorem 1 (and Proposition 1) can be computed straightforwardly via line or grid search (see Waudby-Smith and Ramdas [58, Section A.5]).

The sequence of nonnegative $( k _ { t } ) _ { t = 1 } ^ { \infty }$ that truncate the reward predictors can also be chosen in any way as long as they are predictable. There are several heuristics that one might use, with increasing levels of complexity. One option is to have a prior guess for $w _ { \mathrm { m a x } }$ (or some value $w _ { \mathrm { m a x } } ^ { \prime }$ that the practitioner believes will upper-bound most importance weights) and set $k _ { t } ~ = ~ w _ { \mathrm { m a x } } ^ { \prime } / C$ for some $C \geqslant 1 , \mathrm { e . g . } C = 2$ For a more adaptive option, one could set $k _ { t } : = \mathrm { m e d i a n } ( w _ { 1 } , \dots , w _ { t - 1 } )$ , or even try out a grid of values $\{ k ^ { ( 1 ) } , \ldots , k ^ { ( J ) } \}$ and choose the $k ^ { ( j ) }$ that would have yielded the tightest CSs in hindsight. Nevertheless, all three of these options yield nonasymptotically valid $( 1 - \alpha ) – \mathrm { C S s }$ for ν.

Remark 2 (Why truncate the reward predictor $\widehat { r } _ { t } ? )$ . Readers familiar with doubly robust estimation in causal inference or contextual bandits will notice that if $k _ { 1 } = k _ { 2 } = \cdot \cdot \cdot = \infty$ , then $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) }$ takes the form of a classical doubly robust estimator of the policy value, and that such estimators often vastly outperform those based on importance weighting alone (and in many cases, are provably more eficient, at least in an asymptotic sense), so why would we want to truncate $\widehat { r } _ { t }$ at all?

The reason has to do with the fact that for nonasymptotic inference, we exploit the lowerboundedness of $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) }$ in order to show that the product in (12) is a nonnegative martingale. However, if we introduce a non-truncated reward predictor, we can only say that $\bar { \phi } _ { t } ^ { ( \mathrm { D R } - \ell ) }$ is lower-bounded by $- w _ { \mathrm { m a x } }$ , which we do not want to assume knowledge of (or that it is finite at all). Truncation of $\widehat { r } _ { t }$ allows us to occupy a middle ground, so that many of the eficiency gains from doubly robust estimation can be realized, without entirely losing the lower-boundedness structure of $\phi _ { t } ^ { ( \mathrm { D R } - \check { \ell } ) }$

This same line of thought helps to illustrate why including a reward predictor tightens our CSs for large t, potentially at the expense of tightness for smaller t. Notice that truncating the reward predictor at $k _ { t } / w _ { t }$ simply restricts the tuning parameter $\lambda _ { t } ^ { L } ( \nu ^ { \prime } )$ to lie in $[ 0 , ( \nu ^ { \prime } + k _ { t } ) ^ { - 1 } )$ instead of $[ 0 , \nu ^ { \prime - 1 } )$ for importance weighting. Without dwelling on the details too much, it is known that smaller values of $\lambda _ { t }$ correspond to CSs and CIs being tighter for larger $t - \mathrm { e . g }$ . the role that λ plays in Hoefding’s CIs looks like $\sqrt { \log ( 2 / \alpha ) / 2 n } \left[ 2 1 \right] -$ but we refer the reader to papers on CSs for more in-depth discussions [24, 58]. The important takeaway for our purposes here, is that larger $k _ { t }$ corresponds to more variance adaptivity via double robustness, but does more to restrict the CSs tightness at small t. Nevertheless, this tradeof is clearly worth it in some cases (see Figure 1).

We note that the idea to truncate $\widehat { r } _ { t }$ based on $k _ { t } / w _ { t }$ was inspired by the so-called “reducedvariance” estimators of Zimmert and Lattimore [64], Zimmert and Seldin [65]. However, their reduced variance estimators are slightly diferent since they multiply by an indicator $\mathbb { 1 } ( w _ { i } \leqslant \eta )$ for some $\eta \geqslant 0 .$ which sends the reward predictor to zero for large importance weights, whereas ours only truncates the reward predictor.

Remark 3 (Mirroring trick for upper CSs). Notice that in Proposition 1 and Theorem 1, upper CSs for ν were obtained by importance weighting $1 - R _ { t }$ rather than $R _ { t }$ to obtain a lower CS for $1 - \nu ,$ which was then translated into an upper CS for ν. This “mirroring trick” — first used in the OPE setting by Thomas et al. [50] to the best of our knowledge — applies to all of the results that follow, but for the sake of succinctness, we will only explicitly write the lower CSs.

## 2.3 Closed-form confidence sequences

In Theorem 1, we derived CSs for the policy value that generalize and improve on prior work [28]. These bounds are empirically tight and can be computed eficiently, but are not closed-form, which may be desirable in practice. In this section, we derive a simple, closed-form, variance-adaptive CS for the fixed policy value $\nu : = \mathbb { E } _ { \pi } ( R _ { t } \mid { \mathcal { H } } _ { t - 1 } )$ . Let

$$
\xi_ {t} := \frac {\phi_ {t} ^ {(\mathrm{DR} - \ell)}}{k _ {t} + 1}, \quad \text { and } \quad \widehat {\xi} _ {t - 1} := \left(\frac {1}{t - 1} \sum_ {i = 1} ^ {t - 1} \xi_ {i}\right) \wedge \frac {1}{k _ {t} + 1}.\tag{16}
$$

With the above notation in mind, we are ready to state the main result of this section.

Proposition 2 (Closed-form predictable plug-in CS for ν). Given contextual bandit data $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ with r0, 1s-valued rewards, choose nonnegative and predictable tuning parameters $( k _ { t } ) _ { t = 1 } ^ { \infty } ,$ and define $( \lambda _ { t } ) _ { t = \cdot } ^ { \infty }$ <sub>1</sub> as

$$
\lambda_ {t} := \sqrt {\frac {2 \log (1 / \alpha)}{\hat {\sigma} _ {t - 1} ^ {2} t \log (1 + t)}} \wedge c, \quad \hat {\sigma} _ {t} ^ {2} := \frac {\sigma_ {0} ^ {2} + \sum_ {i = 1} ^ {t} (\xi_ {i} - \bar {\xi} _ {i}) ^ {2}}{t + 1}, \quad \bar {\xi} _ {t} := \left(\frac {1}{t} \sum_ {i = 1} ^ {t} \xi_ {i}\right) \wedge \frac {1}{k _ {t} + 1},\tag{17}
$$

where $c \in ( 0 , 1 )$ is some truncation parameter (reasonable values of which may include $1 / 2$ or ${ \it 3 / 4 } ) ,$ and $\xi _ { 0 } \in ( 0 , 1 )$ and $\sigma _ { 0 } ^ { 2 } > 0$ are some user-chosen parameters that can be thought of as prior guesses for the mean and variance of ξ, respectively. Then,

$$
L _ {t} ^ {\mathrm{PrPl}} := \left(\frac {\sum_ {i = 1} ^ {t} \lambda_ {i} \xi_ {i}}{\sum_ {i = 1} ^ {t} \lambda_ {i} / (k _ {i} + 1)} - \frac {\log (1 / \alpha) + \sum_ {i = 1} ^ {t} \left(\xi_ {i} - \widehat {\xi} _ {i - 1}\right) ^ {2} \psi_ {E} (\lambda_ {i})}{\sum_ {i = 1} ^ {t} \lambda_ {i} / (k _ {i} + 1)}\right) \vee 0\tag{18}
$$

forms a lower $( 1 - \alpha ) – C S f o r \nu .$ . An analogous upper CS $( U _ { t } ^ { \mathrm { P r P l } } ) _ { t = 1 } ^ { \infty }$ follows by mirroring (Remark 3).

The proof can be found in Appendix A.4 and relies on Ville’s inequality [53] applied to a predictable plug-in empirical Bernstein supermartingale similar to that of Waudby-Smith and Ramdas [58, Section 3.2] but with a variant of Fan’s inequality [15] for lower-bounded random variables with upper-bounded means. As seen in Figure 2, the betting CSs of Theorem 1 still have better empirical performance than the closed-form predictable plug-in CSs of Proposition 2, but the latter are more computationally and analytically convenient, and are valid under the same set of assumptions. A similar phenomenon was observed by Waudby-Smith and Ramdas [58] for bounded random variables (outside the context of OPE). In the on-policy setting with all importance weights set to 1 and without a reward predictor, Proposition 2 recovers Waudby-Smith and Ramdas [58, Theorem 2].


Figure 2: Betting-based (Theorem 1) and predictable plug-in (PrPl) (Proposition 2) CSs for ν with both importance-weighted (IW) and doubly robust (DR) variants. Notice that for both IW and DR CSs, the betting-based approach of Theorem 1 outperforms the PrPl CSs. Nevertheless, the closedform PrPl CSs are simpler to implement, and can still benefit from doubly robust variance adaptation.

It is important not to confuse $\widehat { \xi } _ { t }$ with $\bar { \xi } _ { t }$ . The diference between them may seem minor since the former simply has access to one less data point than the latter, but they play two very diferent roles in Proposition $2 \colon \widehat { \xi } _ { t - 1 }$ is a predictable sample mean that shows up in the width of $\dot { L } _ { t } ^ { \mathrm { P r P l } }$ explicitly, and its predictability is fundamental to the proof technique, while $\bar { \xi } _ { t }$ is just used as a tool to obtain better estimates of $\dot { \widehat { \sigma } } _ { t } ^ { 2 }$ so that they can be plugged in to the tuning parameters $\lambda _ { t }$ . Consequently, $\widehat { \xi } _ { t }$ can be found in CSs that rely on a similar proof technique (such as Theorem 2), while $\xi _ { t }$ can be found in other CSs that make use of predictable tuning parameters (such as Theorem 1).

## 2.4 Fixed-time confidence intervals

While this paper is focused on time-uniform CSs for OPE, our methods also naturally give rise to fixedtime CIs that are not anytime-valid but can still benefit from our general techniques. In this section, we will briefly discuss what minor modifications are needed to derive sharp fixed-time instantiations of our otherwise time-uniform bounds. We will also compare our fixed-time CIs to the CIs of Thomas et al. [50], but this comparison is by no means comprehensive. Indeed, our goal is not to show that our methods are “better” than prior work, even if some simulations may suggest this — instead, we aim to provide the reader with some context as to how our fixed-time instantiations fit within the broader literature on CIs for OPE.

Confidence intervals for policy values. We begin by deriving fixed-time analogues of the CSs for ν presented in Theorem 1 and Proposition 2 — the former being a “betting-style” CS that is very tight in practice, and the latter being a closed-form predictable plug-in (PrPl) CS that is slightly more analytically and computationally convenient. In both cases, our suggested modification is essentially the same: choose a predictable sequence $( \lambda _ { t } ) _ { t = 1 } ^ { n }$ that is tuned for the desired sample size n — to be elaborated on shortly — and take the intersection of the implicit CS $( C _ { t } ) _ { t = 1 } ^ { n }$ that is formed from times 1 through n. Concretely, define the predictable sequence $( \lambda _ { t } ) _ { t = 1 } ^ { n }$ given by

$$
\dot {\lambda} _ {t, n} := \sqrt {\frac {2 \log (1 / \alpha)}{n \widehat {\sigma} _ {t - 1} ^ {2}}},\tag{19}
$$

where $\widehat { \sigma } _ { t } ^ { 2 }$ is given as in (15). Then, we have the following corollary for betting-style CIs for $\nu .$

Corollary 1. Let $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { n }$ be a finite sequence of contextual bandit data with r0, 1s-valued rewards and define $( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) ) _ { t = 1 } ^ { n }$ and $( \lambda _ { t } ^ { U } ( \nu ^ { \prime } ) ) _ { t = 1 } ^ { n }$ by

$$
\lambda_ {t} ^ {L} (\nu^ {\prime}) := \dot {\lambda} _ {t, n} \wedge \frac {c}{k _ {t} + \nu^ {\prime}}, \quad a n d \quad \lambda_ {t} ^ {U} (\nu^ {\prime}) := \dot {\lambda} _ {t, n} \wedge \frac {c}{k _ {t} + (1 - \nu^ {\prime})},\tag{20}
$$

where $c \in ( 0 , 1 )$ is some truncation scale as in Theorem 1. Let $L _ { t } ^ { \mathrm { D R } }$ and $U _ { t } ^ { \mathrm { D R } }$ be as in (12) and (13). Then,

$$
\dot {L} _ {n} := \max _ {1 \leqslant t \leqslant n} L _ {t} ^ {\mathrm{DR}} \quad a n d \quad \dot {U} _ {n} := \min _ {1 \leqslant t \leqslant n} U _ {t} ^ {\mathrm{DR}}\tag{21}
$$

form lower and upper $( 1 - \alpha ) – C I s ~ f o r ~ \nu ,$ respectively, meaning $\mathbb { P } ( \nu \in [ \dot { L } _ { n } , \dot { U } _ { n } ] ) \geqslant 1 - \alpha$


Figure 3: Fixed-time 90% confidence intervals for ν using three diferent methods: a betting-based CI (Corollary 1), a predictable plug-in (PrPl) CI (Corollary 2), and those presented in a paper entitled “High-confidence of-policy evaluation” (HCOPE15) by Thomas et al. [50]. Notice that the bettingbased CI outperforms the closed-form PrPl CI, which itself significantly outperforms the bounds in Thomas et al. [50].

Corollary 1 is an immediate consequence of Theorem 1 where the sequence of tuning parameters was chosen to tighten the CI for the sample size n. This particular choice of $( \lambda _ { t } ^ { L } ( \nu ^ { \prime } ) ) _ { t = 1 } ^ { n }$ is inspired by the fact that the product in (12) resembles an exponential supermartingale whose resulting CI can be tightened using tuning parameters that are well-estimated by (20). For more details, we refer the reader to Waudby-Smith and Ramdas [58, Section 3]. The fact that the maximum and minimum can be taken in (21) follows from the fact that $( L _ { t } ^ { \mathrm { D R } } ) _ { t = 1 } ^ { n }$ satisfies $\mathbb { P } \left( \forall t \in \{ 1 , \dots , n \} , \ \nu \geqslant L _ { t } ^ { \mathrm { D R } } \right) \geqslant 1 - \alpha$ and similarly for the upper CI $\dot { U } _ { n }$ . Figure 3 demonstrates what these CIs may look like in practice.

Similar to how Corollary 1 is a fixed-time instantiation of Theorem 1, the following corollary is a fixed-time instantiation of the closed-form PrPl CSs of Proposition 2.

Corollary 2. Let $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { n }$ be a finite sequence of contextual bandit data with r0, 1s-valued rewards and define $( \lambda _ { t } ) _ { t = 1 } ^ { n }$ by

$$
\lambda_ {t} := \dot {\lambda} _ {t, n} \wedge c,\tag{22}
$$

with c chosen as in Proposition 2. Then, with $L _ { t } ^ { \mathrm { P r P l } }$ and $U _ { t } ^ { \mathrm { P r P l } }$ defined in Proposition 2, we have that

$$
\dot {L} _ {n} ^ {\mathrm{PrPl}} := \max _ {1 \leqslant t \leqslant n} L _ {t} ^ {\mathrm{PrPl}} \quad a n d \quad \dot {U} _ {n} ^ {\mathrm{PrPl}} := \min _ {1 \leqslant t \leqslant n} U _ {t} ^ {\mathrm{PrPl}}\tag{23}
$$

form lower and upper $( 1 - \alpha ) – C I s ~ f o r ~ \nu ,$ respectively, meaning $\mathbb { P } ( \nu \in [ \dot { L } _ { n } ^ { \mathrm { P r P l } } , \dot { U } _ { n } ^ { \mathrm { P r P l } } ] ) \geqslant 1 - \alpha$

Corollary 2 is an immediate consequence of Proposition 2 instantiated for a diferent choice of $\lambda _ { t }$ and with an intersection being taken over the implicit CS from times 1 through n.

While the methods of this section improve on past work both theoretically and empirically, each of our results thus far have assumed that $\nu \equiv \mathbb { E } _ { \pi } ( R _ { t } )$ is fixed and does not change over time, an assumption that we may not always wish to make in practice (e.g. if the environment is nonstationary). Fortunately, it is still possible to design CSs that capture an interpretable parameter: the time-varying average policy value thus far. However, we will need completely diferent supermartingales to achieve this, which we outline in the following section.

## 3 Inference for time-varying policy values

Let us now consider the more challenging task of performing anytime-valid of-policy inference for a time-varying average policy value. Concretely, suppose that the value of the r0, 1s-bounded reward $R _ { t }$ under policy π is given by $\nu _ { t } : = \mathbb { E } _ { \pi } ( R _ { t } \mid \mathcal { H } _ { t - 1 } ) \in [ 0 , 1 ]$ , and hence $( \nu _ { t } ) _ { t = 1 } ^ { \infty }$ is now a sequence of conditional policy values. Our goal is to derive CSs for $( \widetilde \nu _ { t } ) _ { t = 1 } ^ { \infty }$ where $\begin{array} { r } { \tilde { \nu } _ { t } : = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \nu _ { i } } \end{array}$ is the average conditional policy value so far. In addition to satisfying desiderata 1–5 in Section 1.2 our CSs will impose no restrictions on how $( \nu _ { t } ) _ { t = 1 } ^ { \infty }$ changes over time. Unfortunately, the techniques of Proposition 1 and Theorem 1 will not work here because their underlying test statistics cannot be written explicitly as functions of a candidate average policy value $\begin{array} { r } { \widetilde \nu _ { t } ^ { \prime } : = \stackrel { \triangledown } { \frac { 1 } { t } } \sum _ { i = 1 } ^ { t } \nu _ { i } } \end{array}$ , but only of the entire candidate tuple $( \nu _ { 1 } \ldots , \nu _ { t } )$ . To remedy this, we will rely on test statistics that are functions of a candidate average $\widetilde { \nu } _ { t } ^ { \prime }$ to derive CSs for $\widetilde { \nu } _ { t }$ in Theorems 2 and 3. An empirical demonstration of the failure of Theorem 1 juxtaposed with the remedy provided by Theorem 2 can be seen in the left-hand side of Figure 4 and a comparison between Theorem 2 and Proposition 3 can be seen in the right-hand side of the same figure.

We will present two CSs for $\tilde { \nu } _ { t } \colon ( 1 )$ the “empirical Bernstein” CS in Theorem 2 whose underlying supermartingale is constructed using Robbins’ method of mixtures [42], and (2) the “iterated logarithm” CS in Proposition 3 which uses the stitching technique. Both yield time-uniform, nonasymptotically valid bounds, and are easy to compute. However, the former tends to have better empirical performance in finite samples, while the latter achieves the (optimal) rate of convergence, matching the law of the iterated logarithm. Nevertheless, both boundaries shrink to zero-width at a rate of $\widetilde { O } ( \sqrt { V _ { t } } / t )$ [24]. Here, ${ \widetilde O } ( \cdot )$ means $O ( \cdot )$ up to logarithmic factors.

In order to write down the empirical Bernstein CS, we first define the scaled doubly robust pseudooutcomes $\xi _ { t } : = \phi _ { t } / ( 1 + k )$ where $\bar { \phi } _ { t } \equiv \phi _ { t } ^ { ( \mathrm { D R } - \ell ) }$ is given in (10) and with all $k _ { t }$ equal to a fixed k. Then, define the corresponding centered sum process $( S _ { t } ( \widetilde { \nu } _ { t } ^ { \prime } ) ) _ { t = 1 } ^ { \infty }$ and variance process $( V _ { t } ) _ { t = 1 } ^ { \infty }$ given by

$$
S _ {t} (\widetilde {\nu} _ {t} ^ {\prime}) := \sum_ {i = 1} ^ {t} \xi_ {i} - \frac {t \widetilde {\nu} _ {t} ^ {\prime}}{1 + k}, \mathrm{and}\tag{24}
$$

$$
V _ {t} := \sum_ {i = 1} ^ {t} (\xi_ {i} - \widehat {\xi} _ {i - 1}) ^ {2}, \text {   where   } \widehat {\xi} _ {t} := \left(\frac {1}{t} \sum_ {i = 1} ^ {t} \xi_ {i}\right) \wedge \frac {1}{1 + k},\tag{25}
$$

and $\xi _ { 0 } \in [ 0 , ( 1 + k ) ^ { - 1 } ]$ is chosen by the user. With this setup and notation in mind, we are ready to state the empirical Bernstein CS for $\widetilde { \nu } _ { t }$

Theorem 2 (Empirical Bernstein confidence sequence for $\widetilde { \nu } _ { t } )$ . Let $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ be an infinite sequence of contextual bandit data with r0, 1s-valued rewards generated by the sequence of policies $\displaystyle \bigl ( h _ { t } \bigr ) _ { t = 1 } ^ { \infty }$ and let $( S _ { t } ( \tilde { \nu } _ { t } ^ { \prime } ) ) _ { t = 1 } ^ { \infty }$ and $( V _ { t } ) _ { t = 1 } ^ { \infty }$ be the centered sum and variance processes as in (24) and (25). Then for any $\rho > 0$

$$
M _ {t} ^ {\mathrm{EB}} (\widetilde {\nu} _ {t}) := \left(\frac {\rho^ {\rho} e ^ {- \rho}}{\Gamma (\rho) - \Gamma (\rho , \rho)}\right) \left(\frac {1}{V _ {t} + \rho}\right) _ {1} F _ {1} (1, V _ {t} + \rho + 1, S _ {t} (\widetilde {\nu} _ {t}) + V _ {t} + \rho)\tag{26}
$$

forms a nonnegative supermartingale starting at one, where ${ _ 1 F _ { 1 } } ( \cdot , \cdot , \cdot )$ is Kummer’s confluent hypergeometric function and $\Gamma ( \cdot , \cdot )$ is the upper incomplete gamma function. Consequently,

$$
L _ {t} ^ {\mathrm{EB}} := \inf \left\{\widetilde {\nu} _ {t} ^ {\prime} \in [ 0, 1 ]: M _ {t} ^ {\mathrm{EB}} (\widetilde {\nu} _ {t} ^ {\prime}) <   1 / \alpha \right\}\tag{27}
$$

forms a lower $( 1 - \alpha ) – C S$ for $\widetilde { \nu } _ { t }$ , meaning $\mathbb { P } ( \forall t \in \mathbb { N } , \ \tilde { \nu } _ { t } \geqslant L _ { t } ^ { \mathrm { E B } } ) \geqslant 1 - \alpha$ . Similarly, an upper CS can be derived by using $\phi _ { t } ^ { \left( \mathrm { D R } - u \right) }$ defined in (11) and employing the mirroring trick described in Remark 3.

The proof of Theorem 2 can be found in Appendix A.3 and relies on an inequality due to Fan et al. [15] along with a mixture supermartingale analogous to that of Howard et al. [24, Proposition 9]. In the on-policy setting with all importance weights set to 1 and with no reward predictor (and hence $k = 0 )$ , we have that Theorem 2 recovers the gamma-exponential mixture CS of Howard et al. [24, Proposition 9].

The tuning parameter $\rho > 0$ efectively dictates the neighborhood of intrinsic time — i.e. the value of $V _ { t } \mathrm { ~ - ~ } \mathrm { a t }$ which $L _ { t } ^ { \mathrm { E B } }$ is tightest, and it is rather straightforward to choose $\rho > 0$ given this interpretation. Following Howard et al. [24], $\rho > 0$ can be chosen to (approximately) tighten $L _ { t } ^ { \mathrm { E B } }$ at $V _ { t } = V ^ { \star }$ by setting

$$
\rho (V ^ {\star}) := \sqrt {\frac {- 2 \log \alpha + \log (- 2 \log \alpha + 1)}{V ^ {\star}}}.\tag{28}
$$

Nevertheless, $L _ { t } ^ { \mathrm { E B } }$ forms a valid lower $( 1 - \alpha ) – \mathrm { C S }$ for $\widetilde { \nu } _ { t }$ regardless of how $\rho > 0$ is chosen, as long as this is done data-independently.

Readers familiar with gamma-exponential mixture supermartingales such as those in Howard et al. [23, 24] and Choe and Ramdas [8] may have expected to see a lower incomplete gamma function $\gamma ( a , b )$ instead of $_ 1 F _ { 1 } ( 1 , a , b )$ . Indeed, $_ 1 F _ { 1 } ( 1 , a , b )$ reduces to $\gamma ( a , b )$ when $b \equiv S _ { t } ( \tilde { \nu } _ { t } ) + V _ { t } + \rho$ is nonnegative, but unlike the lower incomplete gamma, ${ _ 1 F _ { 1 } } ( 1 , a , b )$ is well-defined when $b < 0$ . Writing (26) in terms of a lower incomplete gamma would have required lower-bounding this term by a piece-wise function when $b \equiv S _ { t } + V _ { t } + \rho < 0$ as in Choe and Ramdas [8, Appendix C]; see Remark 5 for details.

While $L _ { t } ^ { \mathrm { E B } }$ is not a closed-form bound, it can be computed eficiently using root-finding algorithms, and it can also be shown to achieve an asymptotic width of $O ( \sqrt { V _ { t } \log V _ { t } } / t )$ (the justifications in Howard et al. [24] carry over to this scenario). While this rate is suficient for deriving CSs with strong empirical performance that shrink to zero-width, one can derive CSs that achieve an improved rate of $O ( { \sqrt { V _ { t } } }$ log log $\overline { { V _ { t } } } / t )$ using a diferent technique known as “stitching”.

Proposition 3 (Variance-adaptive iterated logarithm confidence sequence for $\widetilde { \nu } _ { t } )$ . Let $\bar { V } _ { t } : = V _ { t } \vee 1$ where $V _ { t }$ is given as in (25). Define the function $\ell _ { t } ( \alpha )$

$$
\ell_ {t} (\alpha) := 2 \log \left(\log \bar {V} _ {t} + 1\right) + \log \left(\frac {1 . 6 5}{\alpha}\right).\tag{29}
$$

Then we have that under the same conditions as Theorem ${ \it 2 } ,$

$$
L _ {t} ^ {\text { L   I   L }} := (k + 1) \left(\frac {1}{t} \sum_ {i = 1} ^ {t} \xi_ {i} - \frac {\sqrt {2 . 1 3 \ell_ {t} (\alpha) \bar {V} _ {t} + 1 . 7 6 \ell_ {t} (\alpha) ^ {2}}}{t} - \frac {1 . 3 3 \ell_ {t} (\alpha) ^ {2}}{t}\right) \vee 0\tag{30}
$$

forms a lower $\begin{array} { r } { ( 1 - \alpha ) – C S f o r \tilde { \nu } _ { t } : = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \nu _ { i } } \end{array}$ , meaning Pp@t, $\widetilde { \nu } _ { t } \geqslant L _ { t } ^ { \mathrm { L I L } } ) \geqslant 1 - \alpha$ . An analogous upper CS $( U _ { t } ^ { \mathrm { L I L } } ) _ { t = 1 } ^ { \infty }$ can be derived using the mirroring trick of Remark 3.

The proof in Appendix A.5 uses the “stitching” (sometimes called “peeling”) technique that is common in the derivation of LIL-type bounds [9, 27, 30] applied to linear sub-exponential boundaries. Importantly, $L _ { t } ^ { \mathrm { L I L } } \asymp \mathcal { O } \left( t ^ { - 1 } \sqrt { V _ { t } \log \log V _ { t } } \right)$ which matches the unimprovable rate implied by the law of the iterated logarithm. We take a maximum with 0 in $L _ { t } ^ { \mathrm { L I L } }$ (and hence an implicit minimum with 1 in $U _ { t } ^ { \mathrm { L I L } } )$ since $\mathcal { N } _ { t } \in [ 0 , 1 ]$ for every t P N by assumption. Of course, if one is in a stationary environment so that $\nu _ { 1 } = \nu _ { 2 } = \cdot \cdot \cdot = \nu _ { \cdot }$ , then $\widetilde { \nu } _ { t } = \nu .$ , and hence $( L _ { t } ^ { \mathrm { L I L } } , U _ { t } ^ { \mathrm { L I L } } )$ forms a $( 1 - \alpha ) – \mathrm { C S }$ for ν.


Figure 4: Various CSs for the time-varying policy value $\widetilde { \nu } _ { t }$ . The left-hand side plot illustrates that while the betting-style CS of Theorem 1 is tight when $\widetilde { \nu } _ { t }$ remains fixed, it fails to cover when $\widetilde { \nu } _ { t }$ changes (in this case, there is an abrupt change at $t = 1 0 0 0 )$ . The right-hand side plot illustrates how Theorem 2 and Proposition 3 compare, both using their importance-weighted (IW) and doubly robust (DR) variants. Notice that while LIL-IW and LIL-DR attain optimal rates of convergence, the empirical Bernstein CSs (EB-IW and EB-DR) are much tighter in practice. In both cases, the DR variant outperforms the IW variant due to the reward being easy to predict in this particular example.

Comparison of Theorems 1 and 2 with prior work. To the best of our knowledge, Thomas et al. [50] were the first to derive nonasymptotic confidence intervals for policy values in contextual bandits, and they did so without knowledge of $w _ { \mathrm { m a x } } .$ . However, their bounds are not time-uniform, and the authors do not consider time-varying policy values nor data-dependent logging policies $\left( h _ { t } \right) _ { t = 1 } ^ { \infty } .$ Four other prior works stand out as being related to the results of this section, namely Karampatziakis et al. [28], Howard et al. [24, Section 4.2], Bibaut et al. [4], and Zhan et al. [61] and we discuss each of them in some detail below. Note that in the last row labeled “Doubly robust” of Table 1, we are referring to the property of confidence sets to be potentially sharpened in the presence of regression estimators without compromising validity (as discussed in the paragraphs surrounding Theorem 1).

• KMR21:. The of-policy CSs of Karampatziakis et al. [28] — reviewed in Proposition 1 — can in several ways be seen as an improvement of Thomas et al. [50] since they are time-uniform, in addition to being empirically tight. As discussed in Section 2, their importance-weighted ofpolicy CSs are strictly generalized and extended in our Theorem 1, but their result nevertheless yields the current state-of-the-art CSs for ν.

• BDKCvdL21 & ZHHA21: Bibaut et al. [4] and Zhan et al. [61] both study the of-policy inference problem in contextual bandits from an asymptotic point of view. Their of-policy estimators take the form of sample averages of influence functions — what Bibaut et al. [4] refer to as the canonical gradient — to which martingale central limit theorems may be applied to obtain asymptotically valid inference.

In contrast to our work, the confidence intervals of Bibaut et al. [4] and Zhan et al. [61] (a) are asymptotic, and hence do not have finite-sample guarantees, (b) are not time-uniform, and hence cannot be used at stopping times, and (c) do not track time-varying policy values.

• HRMS21: Howard et al. [24, Section 4.2] derive time-uniform, nonasymptotic CSs for the average treatment efect (ATE) in randomized experiments. The main diference between our results and those of [24, Section 4.2] is that they do not study the contextual bandit of-policy evaluation problem. However, since estimating the ATE in randomized experiments can be seen as a special case of the contextual bandit problem, it is natural to wonder how our approach difers in this special case. The main diference here is that Howard et al. [24] require knowledge of $w _ { \mathrm { m a x } } - \mathrm { o r }$ equivalently in their setup, the maximal and minimal propensity scores — while ours do not. Moreover, our results allow $w _ { \mathrm { m a x } }$ to be infinite and nevertheless enjoy varianceadaptivity. See Section 3.2 for a more detailed discussion of the implications of our bounds for ATE estimation in randomized experiments.

Table 1: Comparison of various CSs and CIs for mean of-policy values.

<table><tr><td></td><td>KMR21</td><td>BDKCvdL21 &amp; ZHHA21</td><td>HRMS21</td><td>Thm. 1</td><td>Thm. 2</td></tr><tr><td>Contextual bandits</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>Time-varying rewards</td><td></td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>Nonasymptotic</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Time-uniform</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Predictable  $(h_t)_{t=1}^{\infty}$ </td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td> $w_{\text{max}}$ -free</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>Doubly robust</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 3.1 A remark on policy value diferences

The results in this paper have taken the form of CSs for policy values, e.g. a sequence of sets $[ L _ { t } , U _ { t } ] _ { t = 1 } ^ { \infty }$ such that Pp@t P N, $\nu \in [ L _ { t } , U _ { t } ] ) \geqslant 1 - \alpha$ , but it may be of interest to directly estimate policy value diferences — e.g. $\delta \equiv \nu _ { 1 } - \nu _ { 2 } \equiv \nu ( \pi _ { 1 } ) - \nu ( \pi _ { 2 } )$ , where $\nu ( \pi )$ is the value of some policy $\pi ,$ and $\pi _ { 1 }$ and $\pi _ { 2 }$ are two policies we would like to compare. In many cases including the gated deployment problem studied by Karampatziakis et al. [28], π<sub>1</sub> is some target policy of interest and $\pi _ { 2 } = h _ { 1 } = h _ { 2 } = \cdot \cdot \cdot = h$ is the logging policy so that $\nu ( \pi _ { 1 } ) - \nu ( h )$ can be interpreted as the additional value $\left( \mathrm { o r } \ ^ { \ast } \mathrm { l i f t } ^ { \dag } \right)$ in the target policy $\pi _ { 1 }$ over the logging policy. However, our setup allows for $\pi _ { 1 }$ and $\pi _ { 2 }$ to be any two policies that are absolutely continuous with respect to the logging policies.

Of course, one can always solve this problem by union bounding: construct $( 1 - \alpha / 2 )$ -CSs for $\nu _ { 1 }$ and $\nu _ { 2 }$ separately to yield a p1´αq-CS for their diference. However, it is possible to remove this small amount of slack introduced by union bounding and instead derive a CS for the diference directly.

The idea is simple: rather than only leverage lower-boundedness of importance-weighted rewards $w _ { t } R _ { t }$ , we construct a new random variable $\theta _ { t } : = w _ { t } ^ { ( 1 ) } R _ { t } - [ 1 - w _ { t } ^ { ( 2 ) } ( 1 - R _ { t } ) ]$ and leverage its lowerboundedness directly — here, $w _ { t } ^ { ( 1 ) } = \pi _ { 1 } / h _ { t }$ and $w _ { t } ^ { ( 2 ) } = \pi _ { 2 } / h _ { t }$ are the importance weights for policies $\pi _ { 1 }$ and $\pi _ { 2 }$ , respectively. In particular, notice that

$$
\mathbb {E} \left[ \theta_ {t} \right] = \delta , \quad \text { and } \quad \theta_ {t} \geqslant - 1, \text {   and   hence }\tag{31}
$$

$$
\frac {1}{2} \left[ \theta_ {t} - \delta \right] \geqslant - 1 \mathrm{almostsurely.}\tag{32}
$$

Consequently, (32) can be used in the proofs of our theorems to derive a CS for δ directly, since those proofs fundamentally rely on the centered (i.e. with their mean subtracted) random variables being almost surely lower-bounded $\mathrm { b y - 1 }$ . For instance, we have that for any p0, 1q-valued predictable sequence $( \lambda _ { t } ( \delta ) ) _ { t = 1 } ^ { \infty }$ 1

$$
M _ {t} (\delta) := \prod_ {i = 1} ^ {t} \left(1 + \lambda_ {i} (\delta) \cdot (\theta_ {t} - \delta) / 2\right)\tag{33}
$$

forms a test martingale and hence $L _ { t } : =$ inf $\left. \delta ^ { \prime } \in [ - 1 , 1 ] : M _ { t } ( \delta ^ { \prime } ) < 1 / \alpha \right.$ forms a lower $( 1 - \alpha ) – \mathrm { C S }$ for δ. As usual, the mirroring trick can be used to obtain an upper CS for this policy value diference. Moreover, the above discussion can be extended to time-varying policy value diferences and doubly robust pseudo-outcomes (rather than just their importance-weighted counterparts), as well as sequences of policies — i.e. analyzing the sequences $( \pi _ { 1 } ^ { ( t ) } ) _ { t = 1 } ^ { \infty }$ and $( \pi _ { 2 } ^ { ( t ) } ) _ { t = 1 } ^ { \infty } -$ but we omit these derivations for the sake of brevity.

## 3.2 Time-varying treatment efects in adaptive experiments

While this paper is focused on anytime-valid contextual bandit inference — i.e. inference for policy values or their CDFs from contextual bandit data — one can nevertheless view of-policy evaluation as a generalization of treatment efect estimation from adaptive experiments. Consequently, every single result in this paper also has powerful implications for nonasymptotic inference for treatment efects from such experiments. In this section, we will focus on adaptive experiments with binary treatments for simplicity, but the analogy extends to more general settings.

From contextual bandits to adaptive experiments with binary treatments. The contextual bandit problem can be seen as a generalization of adaptive experiments since the latter has three key notational diferences.

1. The “context” $X _ { t }$ is typically referred to as a “covariate” or a “feature”, and may be used to represent baseline demographics and medical history in a clinical trial, for example.

2. The “action” $A _ { t }$ (which is binary in this case) is referred to as a “treatment”, and the policy $h _ { t }$ is called the “propensity score”, and is simply the probability of a subject with covariates $X _ { t }$ receiving treatment $A _ { t } = 1$ at time t.

3. The “reward” $R _ { t }$ is often referred to as the “outcome” for subject t.

There are many reasons why one may wish to run an adaptive sequential experiment rather than a simple Bernoulli(h) experiment with a constant pre-specified propensity score h. Two simple examples include: (a) balancing designs such as Efron’s biased coin [14] which vary the propensity scores $( h _ { t } ) _ { t = 1 } ^ { \infty }$ 1 over time to ensure that treatment groups are “balanced” within certain levels of the covariates, and (b) the experimental designs of Kato et al. [29] which adaptively choose propensity scores to minimize the variance of the resulting doubly robust and inverse propensity-weighted (IPW) estimators, yielding sharper confidence sets. (In the language of contextual bandits and of-policy evaluation, IPW and importance weighting are equivalent.) Both (a) and (b) — or any other design that varies propensity scores adaptively over time — can be paired with the CSs of the current paper.

Implications for causal inference in adaptive experiments. From the perspective of treatment efect estimation, the current paper provides nonasymptotic, nonparametric, time-uniform inference for treatment efects, all without knowledge of the minimal propensity score $h _ { \operatorname* { m i n } } : = \mathrm { e s s } \ \mathrm { i n f } _ { t , a , x } h _ { t } ( a \ |$ xq and this essential infimum can even be 0 (as long as it is not attained, meaning each $h _ { t } ( \boldsymbol { a } \mid \boldsymbol { x } )$ is itself positive). Contrast this with prior work on nonasymptotic, nonparametric, time-uniform inference for treatment efects such as Howard et al. [24, Section 4.2], which require a priori knowledge of $h _ { \operatorname* { m i n } } > 0$ Their bounds necessarily scale with an implied upper-bound on the variance of $h _ { t } ^ { - 1 } R _ { t }$ implied by $h _ { \operatorname* { m i n } } ^ { - 1 }$ while ours only scale with the empirical variance of $( h _ { t } ^ { - 1 } R _ { t } ) _ { t = 1 } ^ { \infty } -$ the latter always being smaller.

Concretely, Theorem 1 can be used to derive CSs for the average treatment efect from adaptively collected data in experiments with binary treatments and bounded outcomes. Theorem 2 goes further, enabling the construction of CSs for time-varying average treatment $e f f e c t s$ similar to Howard et al. [24, Section 4.2]. Finally, Theorem 3 — to be presented in Section 4 — allows for the construction of time-uniform confidence bands for the CDF of the outcome distribution under a given treatment. Moreover, all of this is possible in a nonparametric, nonasymptotic framework, without knowledge (or strict positivity) of $h _ { \mathrm { m i n } }$ . To the best of our knowledge, all three of these implied results are new in the literature for treatment efect estimation.

## 3.3 Sequential testing and anytime p-values for of-policy inference

While we have thus far taken an estimation perspective (i.e. deriving CSs and CIs rather than p-values), all of our results have hypothesis testing analogues. In particular, the CSs and CIs developed in this paper have all been built by first deriving implicit e-processes. Formally, given a set of distributions $\mathcal { P } _ { 0 }$ (referred to as “the null hypothesis”), an e-process $E \equiv ( E _ { t } ) _ { t = 1 } ^ { \infty }$ for $\mathcal { P } _ { 0 }$ is a nonnegative process such that $\mathbb { E } _ { P } [ E _ { \tau } ] \leqslant 1$ for any $P \in \mathcal { P } _ { 0 }$ and any stopping time τ. (In particular, all test supermartingales for $\mathcal { P } _ { 0 }$ are e-processes by the optional stopping theorem, but not vice versa.)

While e-processes can serve as tools to derive CSs, they can also be used as interpretable testing tools in their own right, or as a way to derive anytime p-values — p-values that are uniformly valid over time in the same sense as CSs. Formally, an anytime p-value for $\mathcal { P } _ { 0 }$ is an H-adapted process $( p _ { t } ) _ { t = } ^ { \infty } .$ such that

$$
\sup _ {P \in \mathcal {P} _ {0}} P \left(\exists t \in \mathbb {N}: p _ {t} \leqslant \alpha\right) \leqslant \alpha .\tag{34}
$$

Compare (34) with a traditional fixed-time p-value $p _ { n }$ that satisfies $\forall n \in \mathbb { N }$ , sup $_ { P \in \mathcal { P } _ { 0 } } P ( p _ { n } \leqslant \alpha ) \leqslant \alpha$ On the other hand, an e-process $( E _ { t } ) _ { t = } ^ { \infty }$ for $\mathcal { P } _ { 0 }$ also satisfies Ville’s inequality:

$$
\sup _ {P \in \mathcal {P} _ {0}} P \left(\exists t \in \mathbb {N}: E _ {t} \geqslant 1 / \alpha\right) \leqslant \alpha .\tag{35}
$$

As a direct consequence of (35), notice that e-processes yield anytime p-values via the transformation $p _ { t } : = ( 1 / E _ { t } ) \wedge 1$

Remark 4 (Which should you choose: e or $p ? )$ . There are several philosophical and practical reasons why one may wish to use e-processes over anytime p-values, despite the fact that they can both be used for sequential hypothesis testing. Philosophically, e-processes (and hence test supermartingales) have game-theoretic interpretations and connections to Bayesian statistics [47, 17, 57], and have been argued to serve as a better foundation for statistical communication [46]. Practically, stopped eprocesses form e-values — nonnegative random variables with expectation at most one [54] — which have several attractive properties over p-values, including the fact that they are very straightforward to combine for the sake of testing a global null [54], to perform meta-analyses [49], or to control the false discovery rate under arbitrary dependence [55]. In this section, we remain agnostic as to which of the two one should use: we will simply derive e-processes and note that their philosophical and practical properties can be enjoyed within OPE, and that if anytime p-values are preferred, they are always available via the transformation $p _ { t } : = ( 1 / E _ { t } ) \wedge 1$

Following Section 3.1, let us now derive sequential tests for whether a policy $\pi _ { 1 }$ has a higher average value than some other policy $\pi _ { 2 }$ . Technically, one could also replace $\pi _ { 1 }$ or π with a sequence of predictable policies, but for simplicity we will only discuss fixed policies. Concretely, let $\Delta _ { t }$ denote the diference in the values of policies $\pi _ { 1 }$ and $\pi _ { 2 }$ at time $t ,$

$$
\delta_ {t} := \nu_ {t} (\pi_ {1}) - \nu_ {t} (\pi_ {2}) \equiv \mathbb {E} _ {A _ {t} \sim \pi_ {1}} (R _ {t}) - \mathbb {E} _ {A _ {t} \sim \pi_ {2}} (R _ {t}),\tag{36}
$$

and let $\begin{array} { r } { \Delta _ { t } : = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \delta _ { t } } \end{array}$ denote the running average diference. We are interested in testing the weak null hypothesis $H _ { 0 }$

$$
H _ {0}: \forall t, \Delta_ {t} \leqslant 0, \quad \text { vs } \quad H _ {1}: \exists t: \Delta_ {t} > 0.\tag{37}
$$

In words, $H _ { 0 }$ says that $^ { 6 6 } \pi _ { 1 }$ is no better than $\pi _ { 2 }$ on average thus $f a r ^ { \ " }$ and was used to compare sequential forecasters in Choe and Ramdas [8]. This “weak null” should be contrasted with the “strong null” that would posit $H _ { 0 } ^ { \star } : \forall t , \delta _ { t } \leqslant 0 \longrightarrow$ clearly, the latter implies the former, and hence any e-process (or anytime p-value) for $H _ { 0 }$ can also be used for $H _ { 0 } ^ { \star }$ . Mathematically, $H _ { 0 }$ is a composite superset of the point null $H _ { 0 } ^ { \star }$ . From a practical perspective, $H _ { 0 }$ may be a favorable null to test since it allows for $\nu _ { t } ( \pi _ { 1 } ) > \nu _ { t } ( \pi _ { 2 } )$ at various t as long as $\begin{array} { r } { \overline { { \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \nu _ { t } ( \pi _ { 1 } ) } } \leqslant \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \nu _ { t } ( \pi _ { 2 } ) } \end{array}$ for all $t ,$ whereas $H _ { 0 } ^ { \star }$ requires $\pi _ { 1 }$ to be uniformly dominated by $\pi _ { 2 }$

Let us now derive an explicit e-process for $H _ { 0 }$ . Using the techniques of Section 3.1, define

$$
\theta_ {t} := w _ {t} ^ {(1)} R _ {t} - (1 - w _ {t} ^ {(2)} (1 - R _ {t})),\tag{38}
$$

where $w _ { t } ^ { ( 1 ) } : = \pi _ { 1 } ( A _ { t } \mid X _ { t } ) / h _ { t } ( A _ { t } \mid X _ { t } )$ and $w _ { t } ^ { ( 2 ) } : = \pi _ { 2 } ( A _ { t } \mid X _ { t } ) / h _ { t } ( A _ { t } \mid X _ { t } )$ are the importance weights for policies $\pi _ { 1 }$ and $\pi _ { 2 }$ . As before, we note that $\theta _ { t } \geqslant - 1$ and $\mathbb { E } ( \theta _ { t } \mid \mathcal { H } _ { t - 1 } ) = \delta _ { t }$ , and hence

$$
\mathbb {E} \left(\frac {1}{t} \sum_ {i = 1} ^ {t} \theta_ {i}   \Big |   \mathcal {H} _ {t - 1}\right) = \Delta_ {t}.\tag{39}
$$

Given the above setup, we are ready to derive an e-process (and hence an anytime p-value) for the weak null $H _ { 0 }$ (an illustration is provided in Figure 5).

Proposition 4. Given contextual bandit data $( X _ { t } , A _ { t } , R _ { t } ) _ { t = } ^ { \infty }$ and two target policies $\pi _ { 1 }$ and $\pi _ { 2 }$ that we would like to compare, define $S _ { t } ( \Delta _ { t } ^ { \prime } )$ and $V _ { t }$ by

$$
S _ {t} (\Delta_ {t} ^ {\prime}) := \frac {1}{2} \left(\sum_ {i = 1} ^ {t} \theta_ {i} - t \Delta_ {t} ^ {\prime}\right) \quad a n d\tag{40}
$$

$$
V _ {t} := \frac {1}{2} \sum_ {i = 1} ^ {t} (\theta_ {i} - \widehat {\theta} _ {i - 1}) ^ {2}, \quad w h e r e \widehat {\theta} _ {t} := \frac {1}{2} \left[ \left(\frac {1}{t} \sum_ {i = 1} ^ {t} \theta_ {i}\right) \wedge 1 \right]\tag{41}
$$

Then, given any $\rho > 0$ , we have that

$$
M _ {t} ^ {\mathrm{EB}} (0) := \left(\frac {\rho^ {\rho} e ^ {- \rho}}{\Gamma (\rho) - \Gamma (\rho , \rho)}\right) \left(\frac {1}{V _ {t} + \rho}\right) _ {1} F _ {1} (1, V _ {t} + \rho + 1, S _ {t} (0) + V _ {t} + \rho)\tag{42}
$$

forms an e-process for $H _ { 0 }$ . Consequently, $p _ { t } : = ( 1 / M _ { t } ^ { \mathrm { E B } } ) \wedge 1$ forms an anytime p-value for the weak null $H _ { 0 } : \forall t , \Delta _ { t } \leqslant 0$ , meaning sup $_ { P \in H _ { 0 } } P ( \exists t : p _ { t } \leqslant \alpha ) \leqslant \alpha$


Figure 5: An illustration of how the anytime p-value derived in Proposition 4 can be used to test the weak null $H _ { 0 } : \forall t , \Delta _ { t } \leqslant 0$ . In the left-hand side plot, notice that $\delta _ { t }$ ventures above 0 at several points prior to $t = 2 0 3 7$ , but the average policy value diference is positive for the first time at $t = 2 0 3 7$ . In the right-hand side plot, we see that the anytime $p \textmd { - }$ value dips below α shortly after $\Delta _ { t } > 0$ , at which point the weak null can be safely rejected, with no penalties for the $p \mathrm { - }$ value having been continuously monitored.

The fact that $M _ { t } ^ { \mathrm { E B } } ( 0 )$ forms an e-process is easy to see: under $H _ { 0 }$ , we have that $\Delta _ { t } \leqslant 0$ and notice that $M _ { t } ^ { \mathrm { E B } }$ with $S _ { t } ( 0 )$ replaced with $S _ { t } ( \Delta _ { t } )$ forms a test supermartingale using the same techniques as Section 3. Since $S _ { t } ( \cdot )$ is nonincreasing and ${ } _ { 1 } F _ { 1 }$ is nondecreasing in its third argument, we have that $M _ { t } ^ { \mathrm { E B } }$ is upper-bounded by the aforementioned test supermartingale whenever $\Delta _ { t } \leqslant 0$ . The claimed e-process property is then an immediate consequence of the optional stopping theorem applied to the above test supermartingale.

## 4 Time-uniform inference for the of-policy CDF

Thus far we have focused on of-policy inference for mean policy values, i.e. functionals of the form $\nu : = \mathbb { E } _ { \pi } \left( R \right)$ . In some cases, however, it may be of interest to study quantiles (e.g. median or $7 5 ^ { \mathrm { t h } }$ percentile) or perhaps the entire cumulative distribution function (CDF) of the reward distribution under policy π. In this section, we focus on the latter, deriving confidence bands for the CDF $\mathbb { P } _ { \pi } ( R \leqslant r )$ of the reward R under policy π. Our confidence bands will be uniform in two senses: in time, and in the quantiles. Concretely, if $Q ( p )$ and $Q ^ { - } ( p )$ q are the right (standard) and left quantiles, respectively — meaning $Q ( p ) : = \operatorname* { s u p } \left\{ x \in \mathbb { R } : \mathbb { P } _ { \pi } ( R \leqslant x ) \right\}$ u and $Q ^ { - } ( p ) : = \operatorname* { s u p } \left\{ x \in \mathbb { R } : \mathbb { P } _ { \pi } ( R < x ) \right\} -$ then we will derive a sequence of confidence bands $[ L _ { t } ( p ) , U _ { t } ( p ) ] _ { t \in \mathbb { N } }$ such that

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, p \in (0, 1), L _ {t} (p) <   Q ^ {-} (p) \text { and } Q (p) <   U _ {t} (p)\right) \geqslant 1 - \alpha .\tag{43}
$$

Such a guarantee enables anytime-valid inference at arbitrary stopping times for all quantiles simultaneously, (as well as any functional thereof). In addition, all our confidence bands will satisfy all five desiderata laid out in Section 1.2, and they will consistently shrink to the true quantile $Q ( p )$ for all $p .$

In order to state our main result, we first need to define a few terms. Define $W _ { t } , \overline { { W } } _ { t } , \bar { q } _ { t } ( p )$ , and $\ell _ { t } ( p ; \alpha )$ given by

$$
W _ {t} := \sum_ {i = 1} ^ {t} w _ {i} ^ {2}, \overline {{W}} _ {t} := W _ {t} \vee 1,\tag{44}
$$

$$
\bar {q} _ {t} (p) := \log \mathrm{it} ^ {- 1} \left(\operatorname{logit} (p) + 4 \sqrt {\frac {e}{\overline {{W}} _ {t}}}\right),\tag{45}
$$

$$
\ell_ {t} (p; \alpha) := 2 \log \left(\log \overline {{W}} _ {t} + 1\right) + 2 \log \left(\left| \left\lceil \frac {\sqrt {\overline {{W}} _ {t}} \operatorname{logit} (p)}{4} \right\rceil \right| \vee 1\right) + \log \left(\frac {7 . 0 6}{\alpha}\right),\tag{46}
$$

$$
\text { and } \mathfrak {B} _ {t} (p; \alpha) := \frac {\sqrt {2 . 1 3 \ell_ {t} (p ; \alpha) \overline {{W}} _ {t} + 1 . 7 6 \bar {q} _ {t} (p) ^ {2} \ell_ {t} (p ; \alpha) ^ {2}}}{t}\tag{47}
$$

$$
+ \frac {1 . 3 3 \overline {{q}} _ {t} (p) \ell_ {t} (p ; \alpha) + t (\overline {{q}} _ {t} (p) - p)}{t}.\tag{48}
$$

While some of the above may seem complicated, they arise naturally from the proof technique discussed below, and it is straightforward to implement them into code. Given the above setup, we are ready to state the main result of this section.

Theorem 3 (Time-uniform confidence band for the of-policy CDF). Consider a sequence of contextual bandit data $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ with real-valued (i.e. not necessarily r0, 1s-bounded) rewards. Let $\begin{array} { r } { \widehat { F } _ { t } ^ { \pi } ( { \boldsymbol { x } } ) : = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } w _ { i } \mathbb { 1 } ( R _ { i } \leqslant { \boldsymbol { x } } ) } \end{array}$ be the importance-weighted empirical CDF, and let $\widehat { Q } _ { t } ( p )$ and $\widehat { Q } _ { t } ^ { - } ( \boldsymbol { p } )$ be the upper and lower empirical quantiles, meaning

$$
\widehat {Q} _ {t} (p) := \sup \left\{x \in \mathbb {R}: \widehat {F} _ {t} ^ {\pi} (x) \leqslant p \right\},\tag{49}
$$

and similarly for $\widehat { Q } _ { t } ^ { - } ( \boldsymbol { p } )$ w $i t h \leqslant$ in the above supremum replaced by a strict inequality ă. Then,

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, p \in (0, 1), Q (p) <   \widehat {Q} _ {t} ^ {-} \left([ p + \mathfrak {B} _ {t} (p; \alpha) ] \wedge 1\right)\right) \geqslant 1 - \alpha .\tag{50}
$$

Similarly, after applying the mirroring trick of Remark 3, we have that

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, p \in (0, 1), \hat {Q} _ {t} \left(\left[ p + \frac {1}{t} \sum_ {i = 1} ^ {t} w _ {i} - 1 - \mathfrak {B} _ {t} (1 - p; \alpha) \right] \vee 0\right) <   Q ^ {-} (p)\right) \geqslant 1 - \alpha .\tag{51}
$$

Figure 6: Time- and quantile-uniform 90% confidence band for the of-policy CDF $\mathbb { P } _ { \pi } ( R \leqslant Q )$ in a Bernoull $. ( 1 / 2 )$ experiment with the target policy set to $\pi ( a \mid x ) = \mathbb { 1 } ( a = 1 ) - { \mathrm { i . e . } }$ “always play action $1 ^ { \mathfrak { s } }$ . Here, the of-policy distribution of R is a beta distribution with $\alpha = \beta = 1 0$ . These confidence bands are simultaneously valid for all $Q \in \mathbb { R }$ and all $t \in \mathbb { N }$ (though we only display them at $t \in \{ 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$ above). In particular, notice that as t gets larger, the confidence bands shrink towards the true CDF (and will continue to do so in the limit).

The proof in Appendix A.6 modifies the “double stitching” technique of Howard and Ramdas [22, Theorem 5] to handle importance-weighted observations, and relies on a sub-exponential concentration inequality rather than a sub-Bernoulli one. Notice that (50) and (51) could be written without $\widehat { Q } _ { t } ^ { - }$ and $Q ^ { - } - \mathrm { i . e }$ . replacing $\widehat { Q } _ { t } ^ { - }$ and $Q ^ { - }$ with $\widehat { Q } _ { t }$ and $Q { \mathrm { . } }$ respectively — but this would never result in a tighter bound. Illustrations of the time-uniform confidence bands derived in Theorem 3 are can be found in Figure 6.

Many of the CSs throughout this paper have recovered prior CSs in the literature when specialized to the on-policy regime (that is, when all importance weights are set to 1 and reward predictors are set to 0). Examples include Theorem 1 recovering Waudby-Smith and Ramdas [58, Theorem 3] or Theorem 2 recovering Howard et al. [24, Proposition 9]. However, Theorem 3 does not recover the on-policy bound it most resembles (Howard and Ramdas [22, Theorem 5]). The reason for this is subtle, and has to do with the fact that in the on-policy regime, $1 ( R _ { t } \leqslant Q ( p ) ) - p$ is a $\operatorname { B e r n o u l l i } ( p )$ random variable, hence their partial sums $\begin{array} { r } { \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \mathsf { \bar { I } } ( R _ { i } \leqslant Q ( p ) ) - p \mathsf { I } } \end{array}$ form sub-Bernoulli processes [23, 24, 22] with variance process $t p ( 1 - p )$ regardless of the value of $Q ( p )$ . On the other hand, in the of-policy setting, we use importance-weighted indicators $w _ { t } \mathbb { 1 } ( R _ { t } \leqslant Q ( p ) ) - p$ whose partial sums are not sub-Bernoulli, but are instead sub-exponential with a variance process that depends on $Q ( p )$ . This fundamental diference changes the test supermartingales that we have access to, and consequently alters the downstream CSs.

Comparison with prior work. There are three prior works that are related to the results of this section, namely those of Howard and Ramdas [22], Chandak et al. [5], and Huang et al. [25], but it is important to note that none of them solve the problem that we are studying — time-uniform confidence bands for of-policy CDFs — and hence we focus on a theoretical comparison with these prior works, rather than an empirical one. We discuss each of them in detail below and summarize how they compare with the present paper in Table 2.

• HR22: Howard and Ramdas [22] derive time- and quantile-uniform confidence bands for the CDF of iid random variables in the on-policy setting, and in particular, Theorem 5 in their paper satisfies a guarantee of the form (43). However, Howard and Ramdas [22] do not consider the of-policy inference problem that we do here, and hence the “Predictable $\left( h _ { t } \right) _ { t = 1 } ^ { \infty } \ d t \ d t ^ { }$ and $^ { 6 } w _ { \mathrm { m a x } ^ { - } }$ free” rows are not applicable $\mathrm { ( N / A ) }$ . In addition, our setup can be seen as a generalization of theirs if all importance weights are set to 1.

• UnO21: In the paper entitled “Universal of-policy evaluation” (UnO), Chandak et al. [5] derive fixed-time quantile-uniform confidence bands for the of-policy CDF. Cleverly exploiting monotonicity of CDFs, they reduce the problem to computing finitely many confidence intervals for means of importance-weighted bounded random variables, and taking a union bound over them. Notably, their bounds do not require knowledge of $w _ { \mathrm { m a x } }$

The main diference between our bounds and those of Chandak et al. [5] is that ours are both time- and quantile-uniform, while theirs are only quantile-uniform. Note that Chandak et al. [5] do consider the more general setup of reinforcement learning in Markov Decision Processes (MDP) — an area to which we intend to extend all of our CSs in the future — and MDPs include the contextual bandit setting as a special case. When focusing on contextual bandits specifically, however, and even when ignoring time-uniformity, Theorem 3 improves on the fixed-time results of Chandak et al. [5] in the two following ways.

First, the confidence bands of Chandak et al. [5] are not guaranteed to shrink to the true CDF as the sample size grows to infinity while ours are (and at an explicit rate of $O ( \sqrt { \log t / t } ) )$ , which we refer to as “consistency” in Table 2.

Second, their bounds assume that the logging policy h is fixed, while ours can take the form of a sequence of data-dependent logging policies $\left( h _ { t } \right) _ { t = 1 } ^ { \infty } ,$ , such as those that result from online learning algorithms. However, since Chandak et al. [5, Theorem 2] simply requires taking a union bound over several CIs for importance-weighted bounded random variables, their Theorem 2 can presumably be extended to handle predictable logging policy sequences by employing the CIs provided in Corollaries 1 or 2.

• HLLA21: Huang et al. [25] derive impressive quantile-uniform confidence bands for the ofpolicy CDF. Their bounds are elegant and simple to state, resembling the famous Dvoretzky-Kiefer-Wolfowitz (DKW) inequalities and their sharpened forms [13, 37]. Notably, their bounds are consistent for the true CDF at $O ( 1 / \sqrt { n } )$ rates. Similar to Chandak et al. [5], however, their results are not time-uniform, and hence do not permit valid inference at stopping times, unlike ours. Moreover, all of their bounds require knowledge of $w _ { \mathrm { m a x } }$ (and for this value to be finite), while our bounds do not. Finally, similar to Chandak et al. [5], their bounds assume that the logging policy h is fixed.

## 5 Summary & extensions

This paper derived time-uniform confidence sequences for various parameters in of-policy evaluation which remain valid even in contextual bandit setups where data are collected adaptively and sequentially over time. We began in Section 1.2 by laying out our desiderata for of-policy inference: we sought methods that (1) are exact and nonasymptotically valid, (2) only make nonparametric assumptions such as boundedness, (3) are time-uniform, and hence valid at arbitrary stopping times, (4) do not require knowledge of extreme values of importance weights, and (5) allow data to be collected by data-dependent logging policies.

Table 2: Comparison of various uniform confidence bands for the CDF

<table><tr><td></td><td>HR22</td><td>UnO21</td><td>HLLA21</td><td>Thm. 3</td></tr><tr><td>Off-policy</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Time-uniform</td><td>√</td><td></td><td></td><td>√</td></tr><tr><td>Consistency</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>Predictable  $(h_t)_{t=1}^{\infty}$ </td><td>N/A</td><td></td><td></td><td>√</td></tr><tr><td> $w_{\text{max}}$ -free</td><td>N/A</td><td>√</td><td></td><td>√</td></tr></table>

In Section 2, we began by studying the most classical of-policy parameter — a fixed policy value ν — and we derived CSs that strictly generalize prior state-of-the-art CSs by weakening the required assumptions and allowing for variance reduction via double robustness. In the same section, we also develop the first closed-form confidence sequences for policy values, as well as some tight fixed-time confidence intervals that are instantiations of our time-uniform bounds. Section 3 then developed CSs for a more general parameter: the time-varying average policy value $\begin{array} { r } { \widetilde \nu _ { t } : = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \nu _ { i } } \end{array}$ , and we discussed what implications these bounds have for adaptive sequential experiments, such as online $\mathrm { A } / \mathrm { B }$ tests.

Finally, in Section 4, we derived simultaneously valid CSs for every quantile of the of-policy reward distribution. Said diferently, these bounds form time-uniform confidence bands for the CDF of the of-policy reward distribution.

There are a few other works that consider CSs and test supermartingales without reference to OPE or contextual bandits, but that nevertheless now have interesting consequences in the OPE problem once paired with the present paper. In particular, we want to highlight the implications that Wang and Ramdas [55] and Xu et al. [60] have on false discovery rate control in OPE, and how Waudby-Smith et al. [59] immediately yields algorithms for diferentially private OPE. We briefly discuss these implications below, but omit their full derivations since these extensions are rather simple and not central to the current paper.

False discovery/coverage rate control under arbitrary dependence. Suppose that rather than estimate a single policy value $\nu ,$ we are interested in a collection $( \nu _ { 1 } , \ldots , \nu _ { J } )$ containing the values of the policies $( \pi _ { 1 } , \ldots , \pi _ { J } )$ . When testing several hypothesis or constructing several CIs, etc., it is often of interest to control some multiple testing metric, such as the false discovery rate (FDR), or the false coverage rate (FCR), respectively [2, 3]. Rather surprisingly, Wang and Ramdas [55] and $\mathrm { X u }$ et al. [60] show that for tests and CIs built from e-values — nonnegative test statistics with expectation at most one — the FDR and FCR can be controlled under arbitrary dependence with virtually no modification to the famous Benjamini-Hochberg [2] and Benjamini-Yekutieli [3] procedures, while this fact is not true for generic tests and CIs. Relevant to the current paper, all of our CSs are fundamentally built from test supermartingales which form e-values at arbitrary stopping times. As a concrete consequence, we can take a collection of stopped CSs ${ C _ { \tau } ^ { ( 1 ) } , \dots , C _ { \tau } ^ { ( J ) } }$ for $( \pi _ { j } ) _ { j \in [ J ] }$ , adjust them via the e-BY procedure of Xu et al. [60] to produce $\widetilde { C } _ { \tau } ^ { ( 1 ) } , \ldots , \widetilde { C } _ { \tau } ^ { ( J ) }$ , so that the FCR is controlled at some desired level $\delta \in ( 0 , 1 )$ . The ability control the FCR under arbitrary dependence is crucial for our setting since the CSs $( C _ { \tau } ^ { ( j ) } ) _ { j \in [ J ] }$ are highly dependent and constructed from the same data, but with diferent importance weights. Similar implications hold for sequential tests and control of the FDR via the e-BH procedure of Wang and Ramdas [55].

Locally diferentially private of-policy evaluation in contextual bandits. Waudby-Smith et al. [59] developed nonparametric CSs and CIs for means of bounded random variables under privacy constraints. The authors developed a so-called “Nonparametric randomized response” (NPRR) mechanism that serves as a nonparametric generalization of Warner’s randomized response [56], mapping a r0, 1s-bounded random variable $Y _ { t }$ to a new random variable $Z _ { t }$ so that each $Z _ { t }$ is an ε-locally diferentially private view of $Y _ { t }$ with mean $r \mathbb { E } ( Y _ { t } ) + ( 1 - r ) / 2$ , where r is a known quantity that depends on ε (and hence it is possible to work out what EpY q is). While Waudby-Smith et al. [59] did not explicitly consider the contextual bandit setup, they did develop CSs for time-varying treatment efects in sequential experiments, similar to the discussion in Section 3.2. However, like other prior work, their CSs require a priori knowledge of the minimal propensity score (in the language of this paper: they require knowledge of $w _ { \mathrm { m a x } } )$ . Nevertheless, it is possible to derive locally private CSs for (time-varying) policy values without knowledge of $w _ { \mathrm { m a x } }$ using the techniques of the current paper. Moreover, several policies can be evaluated from a single application of NPRR, thereby avoiding in flation of the privacy parameter ε from evaluating multiple policies. That is, given r0, 1s-bounded rewards $( R _ { t } ) _ { t = 1 } ^ { \infty } ,$ we can use NPRR to generate private views $( Z _ { t } ) _ { t = 1 } ^ { \infty }$ of these rewards, and notice that $\mathbb { E } ( w _ { t } Z _ { t } ) = \mathbb { E } _ { A _ { t } \sim \pi } ( Z _ { t } ) = r \mathbb { E } _ { A _ { t } \sim \pi } ( R _ { t } ) + ( 1 - r ) / 2$ , and hence a CS for $\mathbb { E } ( w _ { t } Z _ { t } )$ can be translated into a CS for $\mathbb { E } _ { A _ { t } \sim \pi } ( R _ { t } )$ even though we only see a privatized version of $R _ { t }$ . In particular, practitioners can derive locally private CSs for time-varying policy values using Theorem 2 for several policies $\pi _ { 1 } , \ldots , \pi _ { J }$ , with only a single application of NPRR.

We believe that this paper presents a comprehensive treatment of OPE inference, yielding procedures that are theoretically valid under more general settings and yet deliver state-of-the-art practical performance. A challenging open problem is to extend these techniques to the of-policy MDP (Markov Decision Process) setting, where the actions at each step afect subsequent covariate and reward dis tributions, as captured by state variables. Another important open problem is to design practical OPE inference methods not just for one policy, but uniformly over an entire family of policies.

## Acknowledgements

IW-S thanks Martin Larsson, Alec McClean, Steve Howard, Ruohan Zhan, and Edward H. Kennedy for helpful discussions. The authors acknowledge support from NSF grants IIS-2229881 and DMS-2310718. AR acknowledges funding from NSF Grant DMS2053804 and from ARL IoBT REIGN. Research reported in this paper was sponsored in part by the DEVCOM Army Research Laboratory under Cooperative Agreement W911NF-17-2-0196 (ARL IoBTCRA). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the oficial policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

## References

[1] Alberto Abadie, Susan Athey, Guido W Imbens, and Jefrey M Wooldridge. Sampling-based versus design-based uncertainty in regression analysis. Econometrica, 88(1):265–296, 2020. 43

[2] Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300, 1995. 25

[3] Yoav Benjamini and Daniel Yekutieli. False discovery rate–adjusted multiple confidence intervals for selected parameters. Journal of the American Statistical Association, 100(469):71–81, 2005. 25

[4] Aur´elien Bibaut, Maria Dimakopoulou, Nathan Kallus, Antoine Chambaz, and Mark van der Laan. Post-contextual-bandit inference. Advances in Neural Information Processing Systems, 34: 28548–28559, 2021. 3, 6, 17

[5] Yash Chandak, Scott Niekum, Bruno da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S Thomas. Universal of-policy evaluation. Advances in Neural Information Processing Systems, 34:27475–27490, 2021. 3, 5, 6, 24

[6] Haoyu Chen, Wenbin Lu, and Rui Song. Statistical inference for online decision making: In a contextual bandit setting. Journal of the American Statistical Association, 116(533):240–255, 2021. 6

[7] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68, 2018. 9

[8] Yo Joong Choe and Aaditya Ramdas. Comparing sequential forecasters. Operations Research, 2023. 16, 20

[9] DA Darling and Herbert Robbins. Confidence sequences for mean, variance, and median. Proceedings of the National Academy of Sciences of the United States of America, 58(1):66, 1967. 4, 16

[10] Maria Dimakopoulou, Zhimei Ren, and Zhengyuan Zhou. Online multi-armed bandits with adaptive inference. Advances in Neural Information Processing Systems, 34:1939–1951, 2021. 6

[11] Miroslav Dud´ık, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. In Proceedings of the 28th International Conference on Machine Learning, pages 1097–1104, 2011. 2, 8, 9

[12] Miroslav Dud´ık, Dumitru Erhan, John Langford, and Lihong Li. Doubly robust policy evaluation and optimization. Statistical Science, 29(4):485–511, 2014. 2, 8, 9

[13] Aryeh Dvoretzky, Jack Kiefer, and Jacob Wolfowitz. Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator. The Annals of Mathematical Statistics, (3):642–669, 1956. 24

[14] Bradley Efron. Forcing a sequential experiment to be balanced. Biometrika, 58(3):403–417, 1971. 5, 19

[15] Xiequan Fan, Ion Grama, and Quansheng Liu. Exponential inequalities for martingales with applications. Electronic Journal of Probability, 20(1):1–22, 2015. 12, 16, 31

[16] Ronald Aylmer Fisher. Design of experiments. British Medical Journal, 1(3923):554, 1936. 43

[17] Peter Gr¨unwald, Rianne de Heide, and Wouter M Koolen. Safe testing. arXiv preprint arXiv:1906.07801, 2019. 20

[18] Vitor Hadad, David A Hirshberg, Ruohan Zhan, Stefan Wager, and Susan Athey. Confidence intervals for policy evaluation in adaptive experiments. Proceedings of the national academy of sciences, 118(15):e2014602118, 2021. 6

[19] Dae Woong Ham, Iavor Bojinov, Michael Lindon, and Martin Tingley. Design-based confidence sequences for anytime-valid causal inference. arXiv preprint arXiv:2210.08639, 2022. 43

[20] Sebastian Haneuse and Andrea Rotnitzky. Estimation of the efect of interventions that modify the received treatment. Statistics in medicine, 32(30):5260–5277, 2013. 42

[21] Wassily Hoefding. Probability Inequalities for Sums of Bounded Random Variables. Journal of the American Statistical Association, 58(301):13–30, 1963. 12

[22] Steven R Howard and Aaditya Ramdas. Sequential estimation of quantiles with applications to $\mathrm { A } / \mathrm { B }$ testing and best-arm identification. Bernoulli, 28(3):1704–1728, 2022. 6, 23, 24, 38

[23] Steven R. Howard, Aaditya Ramdas, Jon McAulife, and Jasjeet Sekhon. Time-uniform Chernof bounds via nonnegative supermartingales. Probability Surveys, 17:257–317, 2020. 16, 23, 31, 34, 36

[24] Steven R Howard, Aaditya Ramdas, Jon McAulife, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021. 4, 6, 12, 15, 16, 17, 18, 19, 23, 31, 33, 34, 35, 36, 40, 43

[25] Audrey Huang, Liu Leqi, Zachary Lipton, and Kamyar Azizzadenesheli. Of-policy risk assessment in contextual bandits. Advances in Neural Information Processing Systems, 34:23714–23726, 2021. 3, 5, 6, 24

[26] Guido W Imbens and Donald B Rubin. Causal inference in statistics, social, and biomedical sciences. Cambridge University Press, 2015. 43

[27] Kevin Jamieson, Matthew Malloy, Robert Nowak, and S´ebastien Bubeck. lil’ UCB: An optimal exploration algorithm for multi-armed bandits. In Conference on Learning Theory, pages 423– 439. PMLR, 2014. 16

[28] Nikos Karampatziakis, Paul Mineiro, and Aaditya Ramdas. Of-policy confidence sequences. International Conference on Machine Learning, 2021. 2, 3, 4, 5, 6, 7, 8, 11, 12, 17, 18, 31

[29] Masahiro Kato, Takuya Ishihara, Junya Honda, and Yusuke Narita. Adaptive experimental design for eficient treatment efect estimation. arXiv preprint arXiv:2002.05308, 2020. 5, 19

[30] Emilie Kaufmann, Olivier Capp´e, and Aur´elien Garivier. On the complexity of best-arm identification in multi-armed bandit models. The Journal of Machine Learning Research, 17(1):1–42, 2016. 16

[31] Edward H Kennedy. Nonparametric causal efects based on incremental propensity score interventions. Journal of the American Statistical Association, 114(526):645–656, 2019. 42

[32] Edward H Kennedy. Semiparametric doubly robust targeted double machine learning: a review. arXiv preprint arXiv:2203.06469, 2022. 9

[33] Koulik Khamaru, Yash Deshpande, Lester Mackey, and Martin J Wainwright. Near-optimal inference in adaptive linear regression. arXiv preprint arXiv:2107.02266, 2021. 6

[34] Tze Leung Lai. On confidence sequences. The Annals of Statistics, 4(2):265–280, 1976. 4

[35] John Langford and Tong Zhang. The epoch-greedy algorithm for contextual multi-armed bandits. Advances in neural information processing systems, 20(1):96–1, 2007. 2

[36] Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web, pages 661–670, 2010. 2

[37] Pascal Massart. The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality. The annals of Probability, 18(3):1269–1283, 1990. 24

[38] Jerzy Neyman. On the application of probability theory to agricultural experiments, essay on principles, section 9. Statistical Science, 5(4):465–472, 1923/1990. 41, 43

[39] Aaditya Ramdas, Johannes Ruf, Martin Larsson, and Wouter Koolen. Admissible anytime-valid sequential inference must rely on nonnegative martingales. arXiv preprint arXiv:2009.03167, 2020. 11

[40] Aaditya Ramdas, Peter Gr¨unwald, Vladimir Vovk, and Glenn Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science, 38(4):576–601, 2023. 8

[41] Pratik Ramprasad, Yuantong Li, Zhuoran Yang, Zhaoran Wang, Will Wei Sun, and Guang Cheng. Online bootstrap inference for policy evaluation in reinforcement learning. Journal of the American Statistical Association, 118(544):2901–2914, 2023. 6

[42] Herbert Robbins. Statistical methods related to the law of the iterated logarithm. The Annals of Mathematical Statistics, 41(5):1397–1409, 1970. 15

[43] James Robins. A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor efect. Mathematical modelling, 7 (9-12):1393–1512, 1986. 42

[44] James M Robins, Andrea Rotnitzky, and Lue Ping Zhao. Estimation of regression coeficients when some regressors are not always observed. Journal of the American Statistical Association, 89(427):846–866, 1994. 9

[45] Donald B Rubin. Estimating causal efects of treatments in randomized and nonrandomized studies. Journal of educational Psychology, 66(5):688, 1974. 41

[46] Glenn Shafer. Testing by betting: A strategy for statistical and scientific communication. Journal of the Royal Statistical Society: Series A (Statistics in Society), 184(2):407–431, 2021. 20

[47] Glenn Shafer, Alexander Shen, Nikolai Vereshchagin, and Vladimir Vovk. Test martingales, Bayes factors and p-values. Statistical Science, 26(1):84–101, 2011. 20

[48] Ye Shen, Hengrui Cai, and Rui Song. Doubly robust interval estimation for optimal policy evaluation in online learning. arXiv preprint arXiv:2110.15501, 2021. 6

[49] Judith ter Schure and Peter Gr¨unwald. All-in meta-analysis: breathing life into living systematic reviews. arXiv preprint arXiv:2109.12141, 2021. 20

[50] Philip Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. High-confidence ofpolicy evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015. 2, 3, 5, 12, 13, 14, 17

[51] Masatoshi Uehara, Chengchun Shi, and Nathan Kallus. A review of of-policy evaluation in reinforcement learning. arXiv preprint arXiv:2212.06355, 2022. 9

[52] Mark J van der Laan and Sherri Rose. Targeted learning: causal inference for observational and experimental data. Springer Science & Business Media, 2011. 9

[53] Jean Ville. Etude critique de la notion de collectif. Bull. Amer. Math. Soc, 45(11):824, 1939. 8, 9, 12, 32, 37

[54] Vladimir Vovk and Ruodu Wang. E-values: Calibration, combination and applications. The Annals of Statistics, 49(3):1736–1754, 2021. 20

[55] Ruodu Wang and Aaditya Ramdas. False discovery rate control with e-values. Journal of the Royal Statistical Society: Series B (Methodological), 84(3):822–852, 2022. 20, 25

[56] Stanley L Warner. Randomized response: A survey technique for eliminating evasive answer bias. Journal of the American Statistical Association, 60(309):63–69, 1965. 26

[57] Ian Waudby-Smith and Aaditya Ramdas. Confidence sequences for sampling without replacement. Advances in Neural Information Processing Systems, 33:20204–20214, 2020. 20

[58] Ian Waudby-Smith and Aaditya Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society, Series B (to appear with discussion), 86:1–27, 2024. 8, 9, 11, 12, 14, 23, 31

[59] Ian Waudby-Smith, Zhiwei Steven Wu, and Aaditya Ramdas. Nonparametric extensions of randomized response for private confidence sets. International Conference on Machine Learning, 202:36748–36789, 2023. 25, 26

[60] Ziyu Xu, Ruodu Wang, and Aaditya Ramdas. Post-selection inference for e-value based confidence intervals. arXiv preprint arXiv:2203.12572, 2022. 25

[61] Ruohan Zhan, Vitor Hadad, David A Hirshberg, and Susan Athey. Of-policy evaluation via adaptive weighting with data from contextual bandits. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 2125–2135, 2021. 3, 6, 17

[62] Kelly Zhang, Lucas Janson, and Susan Murphy. Inference for batched bandits. Advances in neural information processing systems, 33:9818–9829, 2020. 6

[63] Kelly Zhang, Lucas Janson, and Susan Murphy. Statistical inference with m-estimators on adaptively collected data. Advances in neural information processing systems, 34:7460–7471, 2021. 6

[64] Julian Zimmert and Tor Lattimore. Connections between mirror descent, Thompson sampling and the information ratio. Advances in Neural Information Processing Systems, 32, 2019. 12

[65] Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22(28):1–49, 2021. 12

## A Proofs of the main results

## A.1 A technical lemma

Lemma 1. Let Z and $\hat { Z }$ be H-adapted processes such that $Z _ { t } - \widehat { Z } _ { t - 1 } \geqslant - 1$ almost surely for all t. Denoting $\mu _ { t } : = \mathbb { E } ( Z _ { t } \mid \mathcal { H } _ { t - 1 } )$ , we have that for any p0, 1q-valued predictable process $( \lambda _ { t } ) _ { t = 1 } ^ { \infty }$ 2

$$
M _ {t} := \exp \left(\sum_ {i = 1} ^ {t} \lambda_ {i} (Z _ {t} - \mu_ {t}) - \sum_ {i = 1} ^ {t} \psi_ {E} (\lambda_ {i}) \left(Z _ {t} - \widehat {Z} _ {t - 1}\right) ^ {2}\right),\tag{52}
$$

forms a test supermartingale, where $\psi _ { E } ( \lambda ) : = - \log ( 1 - \lambda ) - \lambda .$

Above, $\widehat { Z } _ { t - 1 }$ is to be interpreted as an estimator of $Z _ { t }$ using the first t ´ 1 samples. Closely related lemmas have appeared in [15, 23, 24, 58], but those papers assumed $Z _ { t } - \mu _ { t } \geqslant - 1$ , which does not sufice for our purposes. What is somewhat surprising above is that we do not require a particular lower bound on $Z _ { t }$ or an upper bound on $\mu _ { t } ,$ , as long as $Z _ { t } - \widehat { Z } _ { t - 1 } \geqslant - 1$

Proof. First, note that $M _ { 0 } \equiv 1$ by construction, and $M _ { t }$ is always positive. It remains to show that $M _ { t }$ forms a supermartingale. Writing out the conditional expectation of $M _ { t }$ given $\mathcal { H } _ { t - 1 }$ , we have that

$$
\mathbb {E} (M _ {t} \mid \mathcal {H} _ {t - 1}) = M _ {t - 1} \underbrace {\mathbb {E} \left(\exp \left\{\lambda_ {t} (Z _ {t} - \mu_ {t}) - \psi_ {E} (\lambda_ {t}) (Z _ {t} - \hat {Z} _ {t - 1}) ^ {2} \right\} \mid \mathcal {H} _ {t - 1}\right)} _ {(\dagger)},\tag{53}
$$

and hence it sufices to prove that $( \dag ) \leqslant 1$ . Denote for the sake of succinctness,

$$
Y _ {t} := Z _ {t} - \mu_ {t} \quad \mathrm{and} \quad \delta_ {t} := \widehat {Z} _ {t - 1} - \mu_ {t},
$$

and note that $\mathbb { E } ( Y _ { t } \mid \mathcal { H } _ { t - 1 } ) = 0$ . Using the proof of Fan et al. [15, Proposition $4 . 1 ]$ , we have that $\exp \{ b \lambda - b ^ { 2 } \psi _ { E } ( \lambda ) \} \leqslant 1 + b \lambda$ for any $\lambda \in [ 0 , 1 )$ and $b \geqslant - 1$ . Setting $b : = Y _ { t } - \delta _ { t } = Z _ { t } - \widehat { Z } _ { t - 1 }$ 2

$$
\begin{array}{l} \mathbb {E} \left[ \exp \left\{\lambda_ {t} Y _ {t} - (Y _ {t} - \delta_ {t}) ^ {2} \psi_ {E} (\lambda_ {t}) \right\} \Big | \mathcal {H} _ {t - 1} \right] \\ = \mathbb {E} \left[ \exp \left\{\lambda_ {t} (Y _ {t} - \delta_ {t}) - (Y _ {t} - \delta_ {t}) ^ {2} \psi_ {E} (\lambda_ {t}) \right\} \Big | \mathcal {H} _ {t - 1} \right] \exp (\lambda_ {t} \delta_ {t}) \\ \leqslant \mathbb {E} \left[ 1 + (Y - \delta_ {t}) \lambda_ {t} \mid \mathcal {H} _ {t - 1} \right] \exp (\lambda_ {t} \delta_ {t}) \\ = \mathbb {E} \left[ 1 - \delta_ {t} \lambda_ {t} \mid \mathcal {H} _ {t - 1} \right] \exp (\lambda_ {t} \delta_ {t}) \leqslant 1, \end{array}
$$

where the last line follows from the fact that $Y _ { t }$ is conditionally mean zero and the inequality $1 - x \leqslant$ $\exp ( - x )$ for all $x \in \mathbb { R }$ . This completes the proof.

## A.2 Proof of Theorem 1

We will only derive the lower CS for $\nu ,$ since the upper CS follows analogously. Consider the process $( M _ { t } ( \nu ) ) _ { t = 1 } ^ { \infty }$ <sub>1</sub> given by

$$
M _ {t} (\nu) := \prod_ {i = 1} ^ {t} \left[ 1 + \lambda_ {i} ^ {L} (\nu) \cdot \left(\phi_ {i} ^ {(\mathrm{DR-} \ell)} - \nu\right) \right].\tag{54}
$$

The proof proceeds in three steps, following the strategy of [24, 58] and [28]. In Step 1, we show that the pseudo-outcomes have conditional mean $\nu ,$ i.e. $\mathbb { E } ( \phi _ { t } ^ { ( \mathrm { D R - } \ell ) } \mid \mathcal { H } _ { t - 1 } ) = \mathbb { E } _ { \pi } ( R _ { t } \mid \mathcal { H } _ { t - 1 } ) = \nu .$ . In Step 2, we use Step 1 to show that $M _ { t } ( \nu )$ forms a test martingale and apply Ville’s inequality to it. In Step 3, we “invert” this test martingale to obtain the lower CS found in Theorem 1.

Step 1: Computing the conditional mean of the doubly robust pseudo-outcomes. Writing out the conditional expectation of $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) }$ , we have

$$
\begin{array}{l} \mathbb {E} [ \phi_ {t} ^ {(\mathrm{DR-} \ell)} \mid \mathcal {H} _ {t - 1} ] \\ = \mathbb {E} (w _ {t} R _ {t} \mid \mathcal {H} _ {t - 1}) - \mathbb {E} \left\{w _ {t} \cdot \left(\widehat {r} _ {t} (X _ {t}; A _ {t}) \wedge \frac {k _ {t}}{w _ {t}}\right) - \mathbb {E} _ {a \sim \pi (\cdot | X _ {t})} \left(\widehat {r} _ {t} (X _ {t}; a) \wedge \frac {k _ {t}}{w _ {t}}\right) \mid \mathcal {H} _ {t - 1} \right\} \\ = \mathbb {E} (w _ {t} R _ {t} \mid \mathcal {H} _ {t - 1}) \\ = \int_ {(x, a, r)} \frac {\pi (a \mid x)}{h _ {t} (a \mid x)} r \cdot p _ {R _ {t}} (r \mid a, x, \mathcal {H} _ {t - 1}) h _ {t} (a \mid x) p _ {X _ {t}} (x \mid \mathcal {H} _ {t - 1})   \mathrm{d} x   \mathrm{d} a   \mathrm{d} r \\ = \int_ {(x, a, r)} r \cdot p _ {R _ {t}} (r \mid a, x, \mathcal {H} _ {t - 1}) \pi (a \mid x) p _ {X _ {t}} (x \mid \mathcal {H} _ {t - 1})   \mathrm{d} x   \mathrm{d} a   \mathrm{d} r \\ = \mathbb {E} _ {\pi} (R _ {t} \mid \mathcal {H} _ {t - 1}) = \nu . \end{array}
$$

Step 2: Showing that $M _ { t } ( \nu )$ forms a test martingale. First, note that $M _ { 0 } \equiv 1$ by construction. To show that $M _ { t }$ is nonnegative, notice that since $R _ { t } , \widehat { r } _ { t } \in [ 0 , 1 ]$ almost surely, we have that $\phi _ { t } ^ { \left( \mathrm { D R - } \ell \right) } \geqslant$ $- k _ { t }$ . Therefore, for any $\nu \in [ 0 , 1 ]$ ，

$$
\begin{array}{r l} & 1 + \lambda_ {t} ^ {L} (\nu) \cdot (\phi_ {t} ^ {(\mathrm{DR-} \ell)} - \nu) \geqslant 1 + \lambda_ {t} ^ {L} (\nu) \cdot (- k _ {t} - \nu) \\ & \qquad > 1 + \frac {- k _ {t} - \nu}{k _ {t} + \nu} \quad (\text {since} \lambda_ {t} ^ {L} (\nu) \in [ 0, (\nu + k _ {t}) ^ {- 1})) \\ & \qquad = 0. \end{array}
$$

Lastly, it remains to show that $\mathbb { E } \left[ M _ { t } ( \nu ) \ | \ \mathcal { H } _ { t - 1 } \right] = M _ { t - 1 } ( \nu )$ . Writing out the conditional expectation of $M _ { t } ( \nu )$ , we have

$$
\begin{array}{r l} & {\mathbb {E} \left[ M _ {t} (\nu) \mid \mathcal {H} _ {t - 1} \right] = \mathbb {E} \left[ M _ {t - 1} (\nu) \left\{1 + \lambda_ {t} ^ {L} (\nu) \cdot (\phi_ {t} ^ {(\mathrm{DR-} \ell)} - \nu) \right\} \mid \mathcal {H} _ {t - 1} \right]} \\ & {\qquad = M _ {t - 1} (\nu) \cdot \left[ 1 + \lambda_ {t} ^ {L} (\nu) \cdot \mathbb {E} \left\{(\phi_ {t} ^ {(\mathrm{DR-} \ell)} - \nu) \mid \mathcal {H} _ {t - 1} \right\} \right]} \\ & {\qquad = M _ {t - 1} (\nu) \cdot (1 + \lambda_ {t} ^ {L} (\nu) \cdot 0) = M _ {t - 1} (\nu),} \end{array}
$$

where the second line follows from the fact that $M _ { t - 1 } , \lambda _ { t } ^ { L }$ are predictable, and the third line follows from Step 1. Therefore, by Ville’s inequality for nonnegative supermartingales [53], we have

$$
\mathbb {P} \left(\exists t \in \mathbb {N}, M _ {t} (\nu) \geqslant \frac {1}{\alpha}\right) \leqslant \alpha .\tag{55}
$$

Step 3: Inverting Ville’s inequality to obtain a lower CS. Recall the lower boundary given by (12),

$$
L _ {t} ^ {\mathrm{DR}} := \inf \left\{\nu^ {\prime} \in [ 0, 1 ]: \prod_ {i = 1} ^ {t} \left[ 1 + \lambda_ {i} ^ {L} (\nu^ {\prime}) \cdot (\phi_ {i} ^ {(\mathrm{DR-} \ell)} - \nu^ {\prime}) \right] <   \frac {1}{\alpha} \right\}
$$

and notice that if $\nu < L _ { t } ^ { \mathrm { D R } }$ , then $M _ { t } ( \nu ) \geqslant 1 / \alpha$ by definition of $L _ { t } ^ { \mathrm { D R } }$ . Consequently,

$$
\mathbb {P} (\exists t \in \mathbb {N}, \nu <   L _ {t} ^ {\mathrm{DR}}) \leqslant \mathbb {P} \left(\exists t \in \mathbb {N}, M _ {t} (\nu) \geqslant \frac {1}{\alpha}\right) \leqslant \alpha .
$$

Therefore, we have $\mathbb { P } ( \forall t \in \mathbb { N } , \ \nu \geqslant L _ { t } ^ { \mathrm { D R } } ) \geqslant 1 - \alpha$ , so $L _ { t } ^ { \mathrm { D R } }$ forms a lower $( 1 - \alpha ) – \mathrm { C S }$ for $\nu ,$ which completes the proof.

## A.3 Proof of Theorem 2

Proof of Theorem 2. The proof proceeds in three steps, following the high level outline of the conjugate mixture method in [24]. First, we invoke Lemma 1 to derive a test supermartingale for each $\lambda \in ( 0 , 1 )$ . Second, we mix over $\lambda \in ( 0 , 1 )$ using the truncated gamma density to obtain (26). Third and finally, we invert this test supermartingale to obtain a lower CS for $\widetilde { \nu } _ { t }$

Step 1: Deriving a test supermartingale indexed by $\lambda \in ( 0 , 1 )$ . Let $Z _ { t } : = \xi _ { t }$ and $\widehat { Z } _ { t - 1 } : = \widehat { \xi } _ { t - 1 }$ as in the setup of Theorem 2. First, notice that $\mathbb { E } ( \xi _ { t } \mid \mathcal { H } _ { t - 1 } ) = \nu _ { t } \colon$

$$
\begin{array}{l} \mathbb {E} (\xi_ {t} \mid \mathcal {H} _ {t - 1}) = \mathbb {E} (w _ {t} R _ {t} \mid \mathcal {H} _ {t - 1}) \\ \qquad = \int_ {x, a, r} \frac {\pi (a \mid x)}{h _ {t} (a \mid x)} r p _ {R _ {t}} (r \mid a, x, \mathcal {H} _ {t - 1}) h _ {t} (a \mid x) p _ {X _ {t}} (x \mid \mathcal {H} _ {t - 1})   \mathrm{d} x   \mathrm{d} a   \mathrm{d} r. \end{array}\tag{56}
$$

(57)

Notice that $\xi _ { t } - \widehat { \xi } _ { t - 1 } \geqslant - 1$ , and hence by Lemma 1, we have that for any $\lambda \in ( 0 , 1 )$

$$
M _ {t} (\widetilde {\nu} _ {t}; \lambda) := \exp \left\{\lambda S _ {t} (\widetilde {\nu} _ {t}) - V _ {t} \psi_ {E} (\lambda) \right\}
$$

forms a test supermartingale.

Step 2: Mixing over λ using the truncated gamma density. For any distribution $F$ on p0, 1q,

$$
M _ {t} ^ {\mathrm{EB}} (\widetilde {\nu} _ {t}) := \int_ {\lambda \in (0, 1)} M _ {t} (\widetilde {\nu} _ {t}; \lambda) \mathrm{d} F (\lambda)\tag{58}
$$

forms a test supermartingale by Fubini’s theorem. In particular, we will use the truncated gamma density $f ( \lambda )$ given by

$$
f (\lambda) = \frac {\rho^ {\rho} e ^ {- \rho (1 - \lambda)} (1 - \lambda) ^ {\rho - 1}}{\Gamma (\rho) - \Gamma (\rho , \rho)},\tag{59}
$$

as the mixing density. Writing out $M _ { t } ( \nu )$ using $d F ( \lambda ) : = f ( \lambda ) d \lambda$ , we have

$$
\begin{array}{l}M_{t}^{\mathrm{EB}}(\widetilde{\nu}_{t}):= \int_{0}^{1}\exp \left\{\lambda S_{t}(\widetilde{\nu}_{t}) - V_{t}\psi_{E}(\lambda)\right\} f(\lambda)  \mathrm{d}\lambda \\ = \int_{0}^{1}\exp \left\{\lambda S_{t}(\widetilde{\nu}_{t}) - V_{t}\psi_{E}(\lambda)\right\} \frac{\rho^{\rho}e^{-\rho(1 - \lambda)}(1 - \lambda)^{\rho - 1}}{\Gamma(\rho) - \Gamma(\rho,\rho)}  \mathrm{d}\lambda \\ = \frac{\rho^{\rho}e^{-\rho}}{\Gamma(\rho) - \Gamma(\rho,\rho)}\int_{0}^{1}\exp \left\{\lambda   (\rho +S_{t} + V_{t})\right\}(1 - \lambda)^{V_{t} + \rho -1}  \mathrm{d}\lambda \\ = \left(\frac{\rho^{\rho}e^{-\rho}}{\Gamma(\rho) - \Gamma(\rho,\rho)}\right)\left(\frac{1}{V_{t} + \rho}\right)\left(\frac{\Gamma(b)}{\Gamma(a)\Gamma(b - a)}\int_{0}^{1}e^{z u}u^{a - 1}(1 - u)^{b - a - 1}  \mathrm{d}u\right)\bigg|_{\substack{a = 1\\ b = V_{t} + \rho +1\\ z = S_{t} + V_{t} + \rho}}\\ = \left(\frac{\rho^{\rho}e^{-\rho}}{\Gamma(\rho) - \Gamma(\rho,\rho)}\right)\left(\frac{1}{V_{t} + \rho}\right)_{1}F_{1}(1,V_{t} + \rho +1,S_{t} + V_{t} + \rho), \end{array}
$$

which completes this step.

Step 3: Inverting the mixture test supermartingale to obtain (26). Similar to Step 3 of the proof of Theorem 1, we have that $\tilde { \nu } _ { t } < L _ { t } ^ { \mathrm { E B } }$ if and only if $M _ { t } ( \widetilde { \nu } _ { t } ) \geqslant 1 / \alpha$ , and hence by Ville’s inequality for nonnegative supermartingales, we have that

$$
\mathbb {P} (\exists t: \widetilde {\nu} _ {t} <   L _ {t} ^ {\mathrm{EB}}) = \mathbb {P} (\exists t: M _ {t} ^ {\mathrm{EB}} (\widetilde {\nu} _ {t}) \geqslant 1 / \alpha) \leqslant \alpha ,
$$

and hence $L _ { t } ^ { \mathrm { E B } }$ forms a lower $( 1 - \alpha ) – \mathrm { C S }$ for $\widetilde { \nu } _ { t }$ . This completes the proof.

Remark 5 (Writing (26) in terms of the lower incomplete gamma function). For readers familiar with Howard et al. [24, Proposition 9], we can rewrite (26) in terms of the lower incomplete gamma function via the identity ${ } _ { 1 } F _ { 1 } ( 1 , b , z ) = ( b - 1 ) e ^ { z } z ^ { 1 - b } ( \Gamma ( b - 1 ) - \Gamma ( b - 1 , z ) )$ q, resulting in

$$
\begin{array}{l} \left(\frac {\rho^ {\rho} e ^ {- \rho}}{\Gamma (\rho) - \Gamma (\rho , \rho)}\right) \left(\frac {1}{v + \rho}\right) _ {1} F _ {1} (1, v + \rho + 1, s + v + \rho) \\ = \left(\frac {\rho^ {\rho}}{\Gamma (\rho) \gamma (\rho , \rho)}\right) \frac {\Gamma (v + \rho) \gamma (v + \rho , s + v + \rho)}{(s + v + \rho) ^ {v + \rho}} \exp {(s + v)}, \end{array}
$$

where $\gamma ( \cdot , \cdot )$ is the lower regularized incomplete gamma function and $v = V _ { t }$ and $s = S _ { t } ( \widetilde { \nu } _ { t } )$ . This matches Howard et al. [24, Eq. (66)] when setting $c = 1$ . The final representation above is realvalued after some complex terms are cancelled (in the case where $( s + v + \rho )$ is negative), but the representation in terms of ${ _ 1 F _ { 1 } } ( 1 , \cdot , \cdot )$ sidesteps this subtlety altogether, which is why we prefer to use it in Theorem 2.

## A.4 Proof of Proposition 2

Proof. Consider the process $M \equiv ( M _ { t } ) _ { t = 1 } ^ { \infty }$ given by

$$
M _ {t} := \exp \left\{\sum_ {i = 1} ^ {t} \lambda_ {i} \left(\xi_ {i} - \frac {\nu}{k _ {i} + 1}\right) - \sum_ {i = 1} ^ {t} \left(\xi_ {i} - \widehat {\xi} _ {i - 1}\right) ^ {2} \psi_ {E} (\lambda_ {i}) \right\}.\tag{60}
$$

Then by Lemma 1, we have that M is a test supermartingale, and hence by Ville’s inequality, $\mathbb { P } ( \exists t :$ $M _ { t } \geqslant 1 / \alpha ) \leqslant \alpha$ . Inverting this time-uniform concentration inequality, we have that with probability at least $( 1 - \alpha )$ and for all $t \in \mathbb { N }$ ,

$$
\begin{array}{l} M _ {t} <   1 / \alpha \iff \exp \left\{\sum_ {i = 1} ^ {t} \lambda_ {i} \left(\xi_ {i} - \frac {\nu}{k _ {i} + 1}\right) - \sum_ {i = 1} ^ {t} \left(\xi_ {i} - \widehat {\xi} _ {i - 1}\right) ^ {2} \psi_ {E} (\lambda_ {i}) \right\} <   \frac {1}{\alpha} \\ \iff \sum_ {i = 1} ^ {t} \lambda_ {i} \left(\xi_ {i} - \frac {\nu}{k _ {i} + 1}\right) - \sum_ {i = 1} ^ {t} \left(\xi_ {i} - \widehat {\xi} _ {i - 1}\right) ^ {2} \psi_ {E} (\lambda_ {i}) <   \log (1 / \alpha) \\ \iff \sum_ {i = 1} ^ {t} \lambda_ {i} \xi_ {i} - \nu \sum_ {i = 1} ^ {t} \frac {\lambda_ {i}}{k _ {i} + 1} - \sum_ {i = 1} ^ {t} \left(\xi_ {i} - \widehat {\xi} _ {i - 1}\right) ^ {2} \psi_ {E} (\lambda_ {i}) <   \log (1 / \alpha) \\ \iff \nu > \frac {\sum_ {i = 1} ^ {t} \lambda_ {i} \xi_ {i}}{\sum_ {i = 1} ^ {t} \lambda_ {i} / (k _ {i} + 1)} - \frac {\log (1 / \alpha) + \sum_ {i = 1} ^ {t} \left(\xi_ {i} - \widehat {\xi} _ {i - 1}\right) ^ {2} \psi_ {E} (\lambda_ {i})}{\sum_ {i = 1} ^ {t} \lambda_ {i} / (k _ {i} + 1)}, \end{array}
$$

which completes the proof.

## A.5 Proof of Proposition 3

We will prove a more general result below for arbitrary $\eta , s > 1$ , but the exact constants in Proposition 3 can be obtained by setting $\eta = e , s = 2$ . By Lemma 1 combined with Howard et al. [23, Table 5, row 7], we have that $S _ { t } ( \widetilde { \nu } _ { t } )$ is a sub-gamma process with scale parameter $c = 1$ , meaning for any $\lambda \in [ 0 , 1 )$

$$
M _ {t} ^ {G} (\lambda) := \exp \left\{\lambda S _ {t} (\widetilde {\nu} _ {t}) - V _ {t} \psi_ {G} (\lambda) \right\},\tag{61}
$$

where $\begin{array} { r } { \psi _ { G } ( \lambda ) \equiv \psi _ { G , 1 } ( \lambda ) = \frac { \lambda ^ { 2 } } { 2 ( 1 - \lambda ) } } \end{array}$ . Define the following parameters:

$$
\lambda_ {k} := \psi^ {- 1} (\log (1 / \alpha) / \eta^ {k + 1 / 2}), \text {   where   } \psi_ {G} ^ {- 1} (a) := \frac {2}{1 + \sqrt {1 + 2 / a}},
$$

$$
\begin{array}{c} \alpha_ {k} := \frac {\alpha}{(k + 1) ^ {s} \zeta (s)}, \text { and } \\ b _ {t, k} := \frac {V _ {t} \psi_ {G} (\lambda_ {k}) + \log (1 / \alpha_ {k})}{\lambda_ {k}}. \end{array}
$$

Taking a union bound over $k \in \mathbb N$ , we have that

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, k \in \mathbb {N}, S _ {t} (\widetilde {\nu} _ {t}) \leqslant b _ {t, k}\right) \geqslant 1 - \alpha .
$$

It remains to find a deterministic upper bound on $b _ { t , k }$ that does not depend on k. Indeed, similar to Howard et al. [24, Eq. (39)], we have that

$$
b _ {t, k} = A \left(\frac {\log (1 / \alpha_ {k})}{\eta^ {k + 1 / 2}}\right) \underbrace {\left[ \sqrt {\frac {\eta^ {k + 1 / 2}}{V _ {t}}} + \sqrt {\frac {V _ {t}}{\eta^ {k + 1 / 2}}} \right]} _ {(\star)} \sqrt {\frac {\log (1 / \alpha_ {k}) V _ {t}}{2}},\tag{62}
$$

where $A ( a ) : = \sqrt { 2 a } / \psi _ { G } ^ { - 1 } ( a ) = \sqrt { 1 + a / 2 } + \sqrt { a / 2 }$ . Now, notice that p‹q is convex in $V _ { t }$ on $V _ { t } \in  { }$ $[ \eta ^ { k } , \eta ^ { k + 1 } ]$ , and hence $( \star )$ is maximized at the endpoints $\eta ^ { k }$ and $\eta ^ { k + 1 }$ . Consequently, on the $k ^ { \mathrm { t h } }$ epoch $- \mathrm { i . e . }$ . when $\eta ^ { k } \leqslant V _ { t } \leqslant \eta ^ { k + 1 } \_$ we have that

$$
\begin{array}{l} b _ {t, k} \leqslant A \left(\frac {\log (1 / \alpha_ {k})}{\eta^ {k + 1 / 2}}\right) \left[ \eta^ {- 1 / 4} + \eta^ {1 / 4} \right] \sqrt {\frac {\log (1 / \alpha_ {k}) V _ {t}}{2}} \\ \leqslant A \left(\frac {\sqrt {\eta} \log (1 / \alpha_ {k})}{V _ {t}}\right) \left[ \eta^ {- 1 / 4} + \eta^ {1 / 4} \right] \sqrt {\frac {\log (1 / \alpha_ {k}) V _ {t}}{2}} \\ = \left[ \sqrt {1 + \frac {\sqrt {\eta} \log (1 / \alpha_ {k})}{2 V _ {t}}} + \sqrt {\frac {\sqrt {\eta} \log (1 / \alpha_ {k})}{2 V _ {t}}} \right] \cdot \left[ \eta^ {- 1 / 4} + \eta^ {1 / 4} \right] \cdot \sqrt {\frac {\log (1 / \alpha_ {k}) V _ {t}}{2}} \end{array}\tag{63}
$$

where the first inequality follows from our analysis of $( \star )$ , the second follows from monotonicity of $A ( \cdot )$ and the fact that $\bar { V _ { t } } \leqslant \bar { \eta } ^ { k + 1 }$ on the $k ^ { \mathrm { t h } }$ epoch, the third follows from the definition of $A ( \cdot )$ . Rewriting the final line (63) more succinctly, we have the following upper bound on $b _ { t , k }$ for every $k \in \mathbb N$

$$
b _ {t, k} \leqslant \sqrt {\gamma_ {1} ^ {2} \log (1 / \alpha_ {k}) V _ {t} + \gamma_ {2} ^ {2} \log^ {2} (1 / \alpha_ {k})} + \gamma_ {2} \log (1 / \alpha_ {k}),\tag{64}
$$

$$
\text { where } \gamma_ {1} := \frac {\eta^ {1 / 4} + \eta^ {- 1 / 4}}{\sqrt {2}}, \text { and } \gamma_ {2} := \frac {\sqrt {\eta} + 1}{2}.\tag{65}
$$

Now, notice that the above bound only depends on k through $\log ( 1 / \alpha _ { k } )$ . As such, we will upper bound $\log ( 1 / \alpha _ { k } )$ solely in terms of $V _ { t }$ and other constants. Indeed, on the $k ^ { \mathrm { t h } }$ epoch, we have

$$
\log (1 / \alpha_ {k}) \equiv \log \left(\frac {(k + 1) ^ {s} \zeta (s)}{\alpha}\right) = s \log (k + 1) + \log \left(\frac {\zeta (s)}{\alpha}\right)
$$

$$
\leqslant s \log (\log_ {\eta} V _ {t} + 1) + \log \left(\frac {\zeta (s)}{\alpha}\right) \equiv \ell_ {t}\tag{66}
$$

where the final line used the upper bound k ď lo $\mathrm { g } _ { \eta } V _ { t }$ which follows because $\eta ^ { k } \leqslant V _ { t }$ on the $k ^ { \mathrm { t h } }$ epoch. Combining (64) and (66), we have that

$$
b _ {t, k} \leqslant \sqrt {\gamma_ {1} ^ {2} \ell_ {t} V _ {t} + \gamma_ {2} ^ {2} \ell_ {t} ^ {2}} + \gamma_ {2} \ell_ {t}, \text {where} \ell_ {t} := s \log (\log_ {\eta} V _ {t} + 1) + \log \left(\frac {\zeta (s)}{\alpha}\right),\tag{67}
$$

which no longer depends on k. Consequently, we have that

$$
\begin{array}{l} 1 - \alpha \leqslant \mathbb {P} \left(\forall t \in \mathbb {N}, \sum_ {i = 1} ^ {t} \xi_ {i} - \sum_ {i = 1} ^ {t} \nu_ {i} \leqslant \sqrt {\gamma_ {1} ^ {2} \ell_ {t} V _ {t} + \gamma_ {2} ^ {2} \ell_ {t} ^ {2}} + \gamma_ {2} \ell_ {t}\right) \\ = \mathbb {P} \left(\forall t \in \mathbb {N}, \widetilde {\nu} _ {t} \geqslant \underbrace {\frac {1}{t} \sum_ {i = 1} ^ {t} \xi_ {i} - \frac {\sqrt {\gamma_ {1} ^ {2} \ell_ {t} V _ {t} + \gamma_ {2} ^ {2} \ell_ {t} ^ {2}}}{t} - \frac {\gamma_ {2} \ell_ {t}}{t}} _ {(\dagger)}\right), \end{array}
$$

and hence p:q forms a lower p1 ´ αq-CS for $\widetilde { \nu } _ { t }$ .

## A.6 Proof of Theorem 3

We will prove a more general result below for arbitrary $\eta , s , \delta > 1$ , but the exact constants in Proposition 3 can be obtained by setting $\eta = e ,$ and $s = \delta = 2$ . The proof will proceed in five steps. First, we derive an exponential e-process $- \mathrm { i . e . }$ an adapted process upper-bounded by a test supermartingale — from $\begin{array} { r } { S _ { t } ( p ) : = \sum _ { i = 1 } ^ { t } w _ { i } \Im \left( R _ { i } \leqslant Q ( p ) \right) - t p } \end{array}$ . Second, we apply Ville’s inequality to the aforementioned e-process to obtain a level-α linear boundary $b _ { t } ( p )$ on $S _ { t } ( p )$ , meaning $\mathbb { P } ( \exists t \in \mathbb { N } : S _ { t } ( p ) \geqslant b _ { t } ( p ) ) \leqslant \alpha$ Third, we derive one $\mathrm { l e v e l } { - \alpha _ { k , j } }$ linear boundary for each $k \in \mathbb { N } , j \in \mathbb { Z }$ using the techniques of Step 2 so that $\textstyle \sum _ { k \in \mathbb { N } } \sum _ { j \in \mathbb { Z } } \alpha _ { k , j } \leqslant a$ and take a union bound over all of them. Here, $k \in \mathbb N$ will index exponentially spaced epochs of time $t \in \mathbb { N }$ , while $j \in \mathbb Z$ will index evenly-spaced log-odds of $p \in ( 0 , 1 )$ . Fourth, we modify the boundaries derived in Step 3 to obtain a boundary that is uniform in both $t \in \mathbb { N }$ and in $p \in ( 0 , 1 )$ . Fifth and finally, we obtain an analytic upper bound on the boundary derived in Step 4.

At several points throughout the proof, we will make use of various functions that depend on k and $j$ . While we will define them as they are needed, we also list them here for reference.

$$
W _ {t} := \sum_ {i = 1} ^ {t} w _ {i} ^ {2},\tag{68a}
$$

$$
\alpha_ {k, j} := \frac {\alpha}{(k + 1) ^ {s} (| j | \vee 1) ^ {s} \zeta (s) (2 \zeta (s) + 1)},   1\tag{68b}
$$

$$
q (k, j) := \frac {1}{1 + \exp \left\{- 2 j \delta / \eta^ {k / 2} \right\}},\tag{68c}
$$

$$
j (k, p) := \left\lceil \frac {\eta^ {k / 2} \operatorname{logit} (p)}{2 \delta} \right\rceil ,\tag{68d}
$$

$$
\lambda (k, j) := \psi_ {G, q (k, j)} ^ {- 1} (\log (1 / \alpha_ {k, j}) / \eta^ {k + 1 / 2}), \text {where} \psi_ {G, c} ^ {- 1} (a) := \frac {2}{c + \sqrt {c ^ {2} + 2 / a}}, \text {and}\tag{68e}
$$

$$
b _ {t, k} (p) := \frac {W _ {t} \psi_ {G , p} (\lambda_ {k , j}) + \log (1 / \alpha_ {k , j})}{\lambda_ {k , j}}.\tag{68f}
$$

Step 1: Deriving an e-process. Invoking Lemma 1 combined with Howard et al. [23, Table 5, row 7] we have that for any $p \in ( 0 , 1 ) , S _ { t } ( p )$ is sub-gamma [23, 24] with variance process $V _ { t } ( p ) : =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { t } ( w _ { i } \mathbb { 1 } \{ R _ { i } \leqslant Q ( p ) \} ) ^ { 2 } } \end{array}$ and scale $c = p$ , meaning we have that for any $\lambda \in [ 0 , 1 / c )$

$$
M _ {t} ^ {G} (\lambda ; p) := \exp \left\{\lambda S _ {t} (p) - V _ {t} (p) \psi_ {G, p} (\lambda) \right\}\tag{69}
$$

forms a test supermartingale. Now, since $\begin{array} { r } { V _ { t } ( p ) \leqslant \sum _ { i = 1 } ^ { t } w _ { i } ^ { 2 } \equiv W _ { t } } \end{array}$ almost surely, we have that

$$
E _ {t} ^ {G} (\lambda ; p) := \exp \left\{\lambda S _ {t} (p) - W _ {t} \psi_ {G, p} (\lambda) \right\} \leqslant M _ {t} ^ {G} (\lambda ; p)\tag{70}
$$

forms an e-process — i.e. it is upper-bounded by a test supermartingale. This completes the first step of the proof.

Step 2: Applying Ville’s inequality to $E _ { t } ^ { G } ( \lambda ; p )$ , yielding a time-uniform linear boundary. In Step 1, we showed that $E _ { t } ^ { G } ( \lambda ; p )$ forms an e-process. By Ville’s maximal inequality for nonnegative supermartingales [53], we have that

$$
\mathbb {P} (\exists t \in \mathbb {N}: E _ {t} ^ {G} (\lambda ; p) \geqslant 1 / \alpha) \leqslant \mathbb {P} (\exists t \in \mathbb {N}: M _ {t} ^ {G} (\lambda ; p) \geqslant 1 / \alpha) \leqslant \alpha .\tag{71}
$$

Now, we will rewrite the inequality $E _ { t } ^ { G } ( \lambda ; p ) \geqslant 1 / \alpha$ slightly more conveniently so that we can derive a time-uniform concentration inequality for $S _ { t } ( p )$ . Indeed,

$$
\begin{array}{r l} E _ {t} ^ {G} (\lambda ; p) \geqslant 1 / \alpha & \Longleftrightarrow \lambda S _ {t} (p) - W _ {t} \psi_ {G, p} (\lambda) \geqslant \log (1 / \alpha) \\ & \Longleftrightarrow S _ {t} (p) \geqslant \underbrace {\frac {W _ {t} \psi_ {G , p} (\lambda) + \log (1 / \alpha)}{\lambda}} _ {b _ {t} (p)}. \end{array}
$$

In summary, we have the following time-uniform concentration inequality on $S _ { t } ( p )$ for any $p \in ( 0 , 1 )$ $\alpha \in ( 0 , 1 )$ and $\lambda \in [ 0 , 1 / p )$ ，

$$
\mathbb {P} \left(\exists t \in \mathbb {N}: S _ {t} (p) \geqslant b _ {t} (p)\right) \leqslant \alpha , \quad \text { where } \quad b _ {t} (p) := \frac {W _ {t} \psi_ {G , p} (\lambda) + \log (1 / \alpha)}{\lambda},\tag{72}
$$

which could also be written as a time-uniform high-probability upper bound on $S _ { t } ( p )$ :

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, S _ {t} (p) <   b _ {t} (p)\right) \geqslant 1 - \alpha .\tag{73}
$$

Step 3: Union-bounding over infinitely many choices of $\lambda , \alpha ,$ and $p .$ In Step 2, we showed that $b _ { t } ( p )$ forms a time-uniform high-probability upper bound for $S _ { t } ( p )$ . We will now take a union bound over a countably infinite two-dimensional grid of t and $p .$ Concretely, for each k P N and $j \in \mathbb { Z } ,$ recall $\alpha _ { k , j } , q ( k , j )$ , and $\lambda ( k , j )$ as in (68b), (68c), and (68e). The exact choices of $q ( k , j )$ and $\lambda ( k , j )$ will become relevant later. For now, note that by (72) from Step 2 combined with a union bound, we have that

$$
\mathbb {P} \left(\exists t \in \mathbb {N}, k \in \mathbb {N}, j \in \mathbb {Z}: S _ {t} (q (k, j)) \geqslant b _ {t, k} (q (k, j))\right) \leqslant \sum_ {k \in \mathbb {N}} \sum_ {j \in \mathbb {Z}} \alpha_ {k, j},\tag{74}
$$

$$
\text { where } \quad b _ {t, k} (q (k, j)) := \frac {W _ {t} \psi_ {G , q (k , j)} (\lambda_ {k , j}) + \log (1 / \alpha_ {k , j})}{\lambda_ {k , j}}.\tag{75}
$$

We will now show that $\textstyle \sum _ { k \in \mathbb { N } } \sum _ { j \in \mathbb { Z } } \alpha _ { k , j } = \alpha$ so that (74) holds with probability at most $\alpha .$ . Indeed,

$$
\begin{array}{l} \sum_ {k \in \mathbb {N}} \sum_ {j \in \mathbb {Z}} \alpha_ {k, j} = \frac {\alpha}{\zeta (s) (2 \zeta (s) + 1)} \sum_ {k \in \mathbb {N}} \frac {1}{(k + 1) ^ {s}} \sum_ {j \in \mathbb {Z}} \frac {1}{(| j | \vee 1) ^ {s}} \\ \qquad = \frac {\alpha}{\zeta (s) (2 \zeta (s) + 1)} \underbrace {\sum_ {k = 0} \frac {1}{(k + 1) ^ {s}}} _ {= \zeta (s)} \underbrace {\left(1 + 2 \sum_ {m = 1} ^ {\infty} \frac {1}{m ^ {s}}\right)} _ {= 2 \zeta (s) + 1} \\ \qquad = \alpha . \end{array}
$$

Therefore, in summary, we have that

$$
\mathbb {P} \left(\exists t \in \mathbb {N}, k \in \mathbb {N}, j \in \mathbb {Z}: S _ {t} (q (k, j)) \geqslant b _ {t, k} (q (k, j))\right) \leqslant \alpha .\tag{76}
$$

Step 4: Removing dependence on $j \in \mathbb Z$ and obtaining p-uniformity. We will obtain a bound that is uniform in $p \in ( 0 , 1 )$ by replacing j with $j ( k , p ) - \mathrm { a }$ function of both $k \in \mathbb N$ and $p \in ( 0 , 1 )$ . For any $k \in \mathbb N$ and any $p \in ( 0 , 1 )$ q, define $j ( k , p )$ as

$$
j (k, p) := \left\lceil \frac {\eta^ {k / 2} \operatorname{logit} (p / (1 - p)}{2 \delta} \right\rceil .\tag{77}
$$

Of course, $j ( k , p ) \in \mathbb { Z }$ is not unique. It is easy to check that $p \leqslant q ( k , j ( k , p ) )$ , a fact that we will use shortly. Abusing notation slightly, let $j _ { 1 } , j _ { 2 } , \ldots$ . denote the integers generated by $j ( k , p )$ for every $k \in \mathbb N$ and $p \in ( 0 , 1 )$ , and let $\mathcal { I } : = \{ j _ { 1 } , j _ { 2 } , . . . \} \subseteq \mathbb { Z }$ denote their image. Given this setup and applying (76) from Step 3, we have that

$$
\begin{array}{l} \mathbb {P} \left(\exists t \in \mathbb {N}, k \in \mathbb {N}, p \in (0, 1): S _ {t} (q (k, j (k, p))) \geqslant b _ {t, k} (q (k, j (k, p))\right) \\ = \mathbb {P} \left(\exists t \in \mathbb {N}, k \in \mathbb {N}, j \in \mathcal {J}: S _ {t} (q (k, j)) \geqslant b _ {t, k} (q (k, j))\right) \\ \leqslant \mathbb {P} \left(\exists t \in \mathbb {N}, k \in \mathbb {N}, j \in \mathbb {Z}: S _ {t} (q (k, j)) \geqslant b _ {t, k} (q (k, j))\right) \\ \leqslant \alpha , \end{array}
$$

where the second line follows from the definition of ${ \mathcal { I } } .$ the third follows from the fact that ${ \mathcal { I } } \subseteq \mathbb { Z }$ and the last follows from (76). In summary, we have the time- and p-uniform concentration inequality given by

$$
\mathbb {P} \left(\exists t \in \mathbb {N}, k \in \mathbb {N}, p \in (0, 1): S _ {t} (q (k, j (k, p))) \geqslant b _ {t, k} (q (k, j (k, p))))\right) \leqslant \alpha , \text { or   equivalently },\tag{78}
$$

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, k \in \mathbb {N}, p \in (0, 1), S _ {t} (q (k, j (k, p))) <   b _ {t, k} (q (k, j (k, p))))\right) \geqslant 1 - \alpha\tag{79}
$$

Step 5: Obtaining a time- and p-uniform upper bound on $S _ { t } ( p )$ . While $( 7 9 )$ is now written to be p-uniform, the quantity $b _ { t , k } ( q ( k , j ( k , p ) ) )$ is only a high-probability upper bound on $S _ { t } ( q ( k , j ( k , p ) ) )$ but what we need is a high-probability upper bound on $S _ { t } ( p )$ . To this end, we use a similar technique to Howard and Ramdas [22] to bound the distance between $S _ { t } ( q ( k , j ( k , p ) ) )$ and $S _ { t } ( p )$ for any $p \in ( 0 , 1 )$ . Indeed, consider the following representation of $S _ { t } ( p )$ in terms of $S _ { t } ( q ( k , j ( k , p ) ) )$ :

$$
\begin{array}{l} S _ {t} (p) := \sum_ {i = 1} ^ {t} w _ {i} \mathbb {1} (R _ {i} \leqslant Q (p)) - t p \\ \qquad \leqslant \sum_ {i = 1} ^ {t} w _ {i} \mathbb {1} (R _ {i} \leqslant Q (q (k, j (k, p)))) - t p \\ \qquad = S _ {t} (q (k, j (k, p))) + t (q (k, j (k, p)) - p), \end{array}
$$

where the first line follows by definition of $S _ { t } ( p )$ , the second by monotonicity of $Q \mapsto \mathbb { 1 } ( R _ { t } \leqslant Q )$ and the fact that $p \leqslant p _ { k , j ( k , p ) }$ , and the third follows from the definition of $S _ { t } ( q ( k , j ( k , p ) ) )$ . Combining (79) with the above representation of $S _ { t } ( p )$ , we have that

$$
\mathbb {P} \left(\forall t \in \mathbb {N}, k \in \mathbb {N}, p \in (0, 1), S _ {t} (p) <   \underbrace {b _ {t , k} (q (k , j (k , p)))} _ {\text {(i)}} + \underbrace {t (q (k , j (k , p)) - p)} _ {\text {(ii)}}\right) \geqslant 1 - \alpha ,\tag{80}
$$

where $\mathrm { ( i ) } \equiv b _ { t , k } ( q ( k , j ( k , p ) ) )$ is given by

$$
b _ {t, k} (q (k, j (k, p))) := \frac {W _ {t} \psi_ {G , q (k , j (k , p))} (\lambda_ {k , j (k , p)}) + \log (1 / \alpha_ {k , j (k , p)})}{\lambda_ {k , j (k , p)}}.\tag{81}
$$

Step 5(i): Upper-bounding piq without dependence on $k .$ . Applying Lemma 2 but with $j ( k , p )$ in place of $j$ , we have that for every $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$ ，

$$
\begin{array}{r l} & b _ {t, k} (q (k, j (k, p))) \leqslant \sqrt {\gamma_ {1} ^ {2} \log (1 / \alpha_ {k , j (k , p)}) W _ {t}} + \gamma_ {2} ^ {2} q (k, j (k, p)) ^ {2} \log^ {2} (1 / \alpha_ {k, j (k, p)}) \\ & \qquad + \gamma_ {2} q (k, j (k, p)) \log (1 / \alpha_ {k, j (k, p)}). \end{array}
$$

Now notice that the above upper bound depends on k solely through $q ( k , j ( k , p ) )$ and $\log ( 1 / \alpha _ { k , j ( k , p ) } )$ each of which we will upper-bound independently of k. By Lemma 3, we have that

$$
q (k, j (k, p)) \leqslant \bar {q} _ {t} (p) \equiv \operatorname{logit} ^ {- 1} \left(\operatorname{logit} (p) + 2 \delta \sqrt {\frac {\eta}{W _ {t}}}\right) \quad \text { for   all } \eta^ {k} \leqslant W _ {t} \leqslant \eta^ {k + 1},\tag{82}
$$

so it remains to upper-bound log $( 1 / \alpha _ { k , j ( k , p ) } )$ q. Recall the definition of $\alpha _ { k , j }$ for any $k \in \mathbb { N } , j \in \mathbb { Z }$ given in (68b). Then we can write $\log ( 1 / \alpha _ { k , j ( k , p ) } )$ as

$$
\log (1 / \alpha_ {k, j (k, p)}) = \underbrace {s \log (k + 1)} _ {(\star k)} + \underbrace {2 \log (| j (k , p) | \vee 1)} _ {(\star j)} + \log \zeta (s) + \log (2 \zeta (s) + 1) + \log (1 / \alpha),\tag{83}
$$

and we observe that p‹kq and $( \star j )$ are the only terms depending on k. Firstly, notice that $( \star j )$ can be upper bounded for every $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$ as

$$
\begin{array}{c}(\star j) \equiv 2 \log (| j (k, p) | \vee 1) = 2 \log \left(\left| \left\lceil \frac {\eta^ {k / 2} \operatorname{logit} (p)}{2 \delta} \right\rceil \right| \vee 1\right)\\\leqslant 2 \log \left(\left| \left\lceil \frac {\sqrt {W _ {t}} \operatorname{logit} (p)}{2 \delta} \right\rceil \right| \vee 1\right).\end{array}
$$

Second, notice that we can easily upper-bound p‹kq on epoch $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$ as

$$
s \log (k + 1) \leqslant s \log \left(\log_ {\eta} W _ {t} + 1\right).
$$

Therefore, we have the following upper-bound on log $\big ( 1 / \alpha _ { k , j ( k , p ) } \big )$ :

$$
\begin{array}{l}\log (1 / \alpha_ {k, j}) \leqslant s \log \left(\log_ {\eta} W _ {t} + 1\right) + 2 \log \left(\left| \left\lceil \frac {\sqrt {W _ {t}} \operatorname{logit} (p)}{2 \delta} \right\rceil \right| \vee 1\right) + \log \zeta (s) + \log (2 \zeta (s) + 1) + \log (1 / \alpha),\\\equiv \ell_ {t} (p),\end{array}
$$

which no longer depends on k. In summary, we have that

$$
b _ {t, k} (q (k, j (k, p))) \leqslant \sqrt {\gamma_ {1} ^ {2} \ell_ {t} (p) W _ {t} + \gamma_ {2} ^ {2} \overline {{q}} _ {t} (p) ^ {2} \ell_ {t} (p) ^ {2}} + \gamma_ {2} \overline {{q}} _ {t} (p) \ell_ {t} (p).
$$

Step 5(ii): Upper-bounding piiq without dependence on k. By Lemma 3, we have that $\begin{array} { r } { q ( k , j ( k , p ) ) \leqslant \bar { q } _ { t } ( p ) \equiv \log \mathrm { i t } ^ { - 1 } \left( \log \mathrm { i t } ( p ) + 2 \delta \sqrt { \frac { \eta } { W _ { t } } } \right) } \end{array}$ Therefore, we can upper bound piiq as

$$
\text {(ii)} \equiv t (q (k, j (k, p)) - p) \leqslant t (\bar {q} _ {t} (p) - p) \equiv t \left[ \operatorname{logit} ^ {- 1} \left(\operatorname{logit} (p) + 2 \delta \sqrt {\frac {\eta}{W _ {t}}}\right) - p \right],\tag{84}
$$

where the final inequality no longer depends on k. In sum, with probability at least $1 - \alpha$

$$
\forall t \in \mathbb {N}, p \in (0, 1), S _ {t} (p) <   \sqrt {\gamma_ {1} ^ {2} \ell_ {t} (p) W _ {t} + \gamma_ {2} ^ {2} \overline {{q}} _ {t} (p) ^ {2} \ell_ {t} (p) ^ {2}} + \gamma_ {2} \overline {{q}} _ {t} (p) \ell_ {t} (p) + t (\overline {{q}} _ {t} (p) - p).\tag{85}
$$

Lemma 2. For any $k \in \mathbb N$ and any $j \in \mathbb { Z }$ , we have that for all $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$

$$
b _ {t, k} (q (k, j)) \leqslant \sqrt {\gamma_ {1} ^ {2} \log (1 / \alpha_ {k , j}) W _ {t} + \gamma_ {2} ^ {2} q (k , j) ^ {2} \log^ {2} (1 / \alpha_ {k , j})} + q (k, j) \gamma_ {2} \log (1 / \alpha_ {k, j}),\tag{86}
$$

where $\gamma _ { 1 } , \gamma _ { 2 }$ are constants defined as

$$
\gamma_ {1} := \frac {\eta^ {1 / 4} + \eta^ {- 1 / 4}}{\sqrt {2}} a n d \gamma_ {2} := \frac {\sqrt {\eta} + 1}{2}.\tag{87}
$$

Proof. Recall the chosen value of $\lambda _ { k , j }$ given in (68e),

$$
\lambda (k, j) := \psi_ {G, q (k, j)} ^ {- 1} (\log (1 / \alpha_ {k, j}) / \eta^ {k + 1 / 2}), \text {where} \psi_ {G, c} ^ {- 1} (a) := \frac {2}{c + \sqrt {c ^ {2} + 2 / a}}\tag{88}
$$

Similar to Howard et al. [24, Eq. (39)], some algebra will reveal that for any $t , k \in \mathbb { N } , j \in \mathbb { Z }$ , we have that

$$
b _ {t, k} (q (k, j)) = A _ {q (k, j)} \left(\frac {\log (1 / \alpha_ {k , j})}{\eta^ {k + 1 / 2}}\right) \underbrace {\left[ \sqrt {\frac {\eta^ {k + 1 / 2}}{W _ {t}}} + \sqrt {\frac {W _ {t}}{\eta^ {k + 1 / 2}}} \right]} \sqrt {\frac {\log (1 / \alpha_ {k , j}) W _ {t}}{2}},
$$

where $A _ { c } ( a ) : = \sqrt { 2 a } / \psi _ { G . c } ^ { - 1 } ( a ) = \sqrt { 1 + c ^ { 2 } a / 2 } + c \sqrt { a / 2 }$ . Now, notice that the second derivative of $( \star )$ with respect to $W _ { t }$ is positive on $W _ { t } \in [ \eta ^ { k } , \eta ^ { k + 1 } ]$ , and hence p‹q is convex in $W _ { t }$ . As such, for every $W _ { t } \in [ \eta ^ { \bar { k } } , \eta ^ { k + 1 } ] - \mathrm { i . e }$ . the $k ^ { \mathrm { t h } }$ epoch — we have that p‹q is maximized at the endpoints $W _ { t } = \eta ^ { k }$ and $W _ { t } = \eta ^ { k + 1 }$ , and we thus have the following upper bound on $b _ { t , k } ( q ( k , j ) )$ on the $k ^ { \mathrm { t h } }$ epoch:

$$
b _ {t, k} (q (k, j)) \leqslant A _ {q (k, j)} \left(\frac {\log (1 / \alpha_ {k , j})}{\eta^ {k + 1 / 2}}\right) \left[ \eta^ {1 / 4} + \eta^ {- 1 / 4} \right] \sqrt {\frac {\log (1 / \alpha_ {k , j}) W _ {t}}{2}}.\tag{89}
$$

Furthermore, since $W _ { t } / \sqrt { \eta } \leqslant \eta ^ { k + 1 / 2 }$ on the $k ^ { \mathrm { t h } }$ epoch, we also have that

$$
A _ {q (k, j)} \left(\frac {\log (1 / \alpha_ {k , j})}{\eta^ {k + 1 / 2}}\right) \leqslant A _ {q (k, j)} \left(\frac {\sqrt {\eta} \log (1 / \alpha_ {k , j})}{W _ {t}}\right) \quad \text { for   all } \eta^ {k} \leqslant W _ {t} \leqslant \eta^ {k + 1}.\tag{90}
$$

Putting (89) and (90) together, we have that for all $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$ ，

$$
b _ {t, k} (q (k, j)) \leqslant \frac {\eta^ {1 / 4} + \eta^ {- 1 / 4}}{\sqrt {2}} \left(\sqrt {\log (1 / \alpha_ {k , j}) W _ {t} + \frac {\sqrt {\eta} q (k , j) ^ {2} \log^ {2} (1 / \alpha_ {k , j})}{2}} + q (k, j) \frac {\eta^ {1 / 4} \log (1 / \alpha_ {k , j})}{\sqrt {2}}\right)
$$

$$
= \sqrt {\gamma_ {1} ^ {2} \log (1 / \alpha_ {k , j}) W _ {t}} + \gamma_ {2} ^ {2} q (k, j) ^ {2} \log^ {2} (1 / \alpha_ {k, j}) + q (k, j) \gamma_ {2} \log (1 / \alpha_ {k, j}),\tag{91}
$$

where $\gamma _ { 1 } , \gamma _ { 2 }$ are constants defined in (87). This completes the proof of Lemma 2.

Lemma 3. Define $\overline { { q } } _ { t } ( \boldsymbol { p } )$ as

$$
\bar {q} _ {t} (p) := \operatorname{logit} ^ {- 1} \left(\operatorname{logit} (p) + 2 \delta \sqrt {\frac {\eta}{W _ {t}}}\right).\tag{92}
$$

For all $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$ , we have that $q ( k , j ( k , p ) ) \leqslant \bar { q } _ { t } ( p )$

Proof. The result follows by definition of $q ( k , j ( k , p ) )$ . Indeed, we have that for all $\eta ^ { k } \leqslant W _ { t } \leqslant \eta ^ { k + 1 }$

$$
\begin{array}{l}q (k, j (k, p)) := \frac {1}{1 + \exp \left\{- 2 j (k , p) \delta / \eta^ {k / 2} \right\}}\\\qquad = \left(1 + \exp \left\{- 2 \left\lceil \frac {\eta^ {k / 2} \operatorname{logit} (p)}{2 \delta} \right\rceil \delta / \eta^ {k / 2} \right\}\right) ^ {- 1}\\\qquad \leqslant \left(1 + \exp \left\{- 2 \left(\frac {\eta^ {k / 2} \operatorname{logit} (p)}{2 \delta} + 1\right) \delta / \eta^ {k / 2} \right\}\right) ^ {- 1}\\\qquad = \left(1 + \exp \left\{- (\operatorname{logit} (p) - 2 \delta / \eta^ {k / 2}) \right\}\right) ^ {- 1}\\\qquad = \operatorname{logit} ^ {- 1} (\operatorname{logit} (p) + 2 \delta / \eta^ {k / 2})\\\qquad \leqslant \operatorname{logit} ^ {- 1} \left(\operatorname{logit} (p) + 2 \delta \sqrt {\frac {\eta}{W _ {t}}}\right),\end{array}
$$

which completes the proof.

## B A causal view of contextual bandits via potential outcomes

In Section 1, we discussed how the OPE problem can be interpreted as asking a counterfactual question, such as “how would the rewards have been, had we used a diferent policy π than the logging policy h that collected the data?”. While it is somewhat reasonable to think about the functional $\nu _ { t } = \mathbb { E } _ { \pi } ( R _ { t } \mid X _ { 1 } ^ { t - 1 } )$ in a counterfactual sense, the Neyman-Rubin potential outcomes framework was designed for the rigorous study of precisely these types of causal questions [38, 45]. In this section, we will define a target causal functional $\nu _ { t } ^ { \star }$ in terms of potential outcomes, and outline the identification assumptions under which $\nu _ { t } ^ { \star }$ is equal to $\nu _ { t }$ (and hence, the conditions under which our CSs can be interpreted as covering the causal quantity $\begin{array} { r } { \dot { \tilde { \nu } } _ { t } ^ { \star } : = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \nu _ { i } ^ { \star } ) } \end{array}$ . We emphasize that these identification assumptions are not required for the CSs to be useful or sensible — indeed, $\widetilde { \nu } _ { t }$ is still an interpretable statistical quantity that we may wish to estimate — but they cannot otherwise be said to cover a causal functional defined in terms of potential outcomes.

Making our setup precise, we posit that for each time t, there is one potential outcome $R _ { t } ( a )$ for every action $a \in A .$ The functional we are ultimately interested in estimating is the conditional mean potential outcome reward under the policy π, i.e.

(93)

$$
\begin{array}{r l} & {\nu_ {t} ^ {\star} \equiv \mathbb {E} _ {\pi} (R _ {t} (G) \mid X _ {1} ^ {t - 1}) := \mathbb {E} \left\{\mathbb {E} _ {G \sim \pi (\cdot | X _ {t})} (R _ {t} (G) \mid X _ {t}, X _ {1} ^ {t - 1}) \mid X _ {1} ^ {t - 1} \right\}} \\ & {\qquad = \int_ {\mathcal {A} \times \mathcal {X}} \mathbb {E} (R _ {t} (g) \mid G = g, X _ {t} = x, X _ {1} ^ {t - 1}) \pi (g \mid x) p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1}) \mathrm{d} g \mathrm{d} x.} \end{array}\tag{94}
$$

In words, $\nu _ { t }$ is the average of the potential outcomes $\{ R _ { t } ( g ) \} _ { g \in \mathcal { A } }$ conditional on $X _ { 1 } ^ { t - 1 }$ with respect to the distribution $\pi ( \cdot \ | \ X _ { t } )$ . We use $g$ and $G$ in place of a and A to avoid confusion between the actual (random) action $A _ { t }$ played according to the logging policy $h _ { t } ( \cdot \ | \ X _ { t } )$ and the hypothetical (random) action $G .$ Without further assumptions, however, the counterfactual quantity $\nu _ { t } ^ { \star }$ is not necessarily identified, meaning it cannot necessarily be written as a functional of the distribution of the observed data $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ . This is simply due to the fact that the potential outcome $R _ { t } ( g )$ is not directly observable from $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ . To remedy this, consider the following causal identification assumptions for every subject t,

(IA1): Consistency: $A _ { t } = a \implies R _ { t } ( a ) = R _ { t }$ for every a $\in { \mathcal { A } }$ with positive π-density,

(IA2): Sequential exchangeability: $A _ { t } \perp \perp R _ { t } ( a ) \mid X _ { 1 } ^ { t }$ , and

(IA3): Positivity: $\pi \ll h _ { t } .$ , meaning $h _ { t } ( A _ { t } \mid X _ { t } ) = 0 \implies \pi ( A _ { t } \mid X _ { t } ) = 0$ almost surely.

Notice that in the contextual bandit setup, IA2 and IA3 are known to hold by design, while IA1 is more subtle (e.g. IA1 may not hold even in a randomized experiment due to interference between subjects, such as in a vaccine trial). Nevertheless, with IA1, IA2, and IA3 in mind, we are ready to state the main identification result of this section.

Lemma 4. Under causal assumptions IA1, IA2, and IA3, we have that

$$
\nu_ {t} ^ {\star} = \nu_ {t}, \quad a n d h e n c e \quad \tilde {\nu} _ {t} ^ {\star} := \frac {1}{t} \sum_ {i = 1} ^ {t} \nu_ {i} ^ {\star} = \frac {1}{t} \sum_ {i = 1} ^ {t} \nu_ {i} =: \tilde {\nu} _ {t}.\tag{95}
$$

In other words, the counterfactual conditional mean $\nu _ { t } ^ { \star }$ can be represented as a function of the distribution of observed data $( X _ { t } , A _ { t } , R _ { t } ) _ { t = 1 } ^ { \infty }$ , and that representation is given by $\nu _ { t }$ .

Proof. The proof is an exercise in causal identification and essentially follows that of Kennedy [31, Theorem 1] and Robins’ g-formula [43], but we nevertheless provide a derivation here for completeness. Writing out the definition of $\nu _ { t } ^ { \star }$ , we have

$$
\nu_ {t} ^ {\star} := \int_ {\mathcal {A} \times \mathcal {X}} \mathbb {E} (R _ {t} (g) \mid G = g, X _ {t} = x, X _ {1} ^ {t - 1}) \pi (g \mid x _ {t}) p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1}) \mathrm{d} g \mathrm{d} x\tag{96}
$$

$$
= \int_ {\mathcal {A} \times \mathcal {X}} \mathbb {E} (R _ {t} (g) \mid G = g, A _ {t} = g, X _ {t} = x, X _ {1} ^ {t - 1}) \pi (g \mid x) p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1}) \mathrm{d} g \mathrm{d} x\tag{97}
$$

$$
= \int_ {\mathcal {A} \times \mathcal {X}} \mathbb {E} (R _ {t} (g) \mid A _ {t} = g, X _ {t} = x, X _ {1} ^ {t - 1}) \pi (g \mid x) p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1}) \mathrm{d} g \mathrm{d} x\tag{98}
$$

$$
= \int_ {\mathcal {A} \times \mathcal {X}} \mathbb {E} (R _ {t} \mid A _ {t} = g, X _ {t} = x, X _ {1} ^ {t - 1}) \pi (g \mid x) p _ {X _ {t}} (x \mid X _ {1} ^ {t - 1}) \mathrm{d} g \mathrm{d} x\tag{99}
$$

$$
= \mathbb {E} \left\{\mathbb {E} _ {A _ {t} \sim \pi (\cdot | X _ {t})} (R _ {t} \mid X _ {t}, X _ {1} ^ {t - 1}) \mid X _ {1} ^ {t - 1} \right\}\tag{100}
$$

$$
\equiv \mathbb {E} _ {\pi} (R _ {t} \mid X _ {1} ^ {t - 1}) =: \nu_ {t},\tag{101}
$$

where (97) follows from IA2 (sequential exchangeability), (98) follows from the fact that $R _ { t } ( G )$ KK $G \mid X _ { 1 } ^ { t }$ (by definition), and (99) follows from IA1 (consistency). Throughout, we implicitly used IA3 (positivity) so that the outer integral is well-defined. That is, we conditioned on $A _ { t } \ = \ g$ at several points, which implicitly leaves us with a factor of $\pi ( { \boldsymbol { g } } \mid x ) / h _ { t } ( { \boldsymbol { g } } \mid x )$ — positivity ensures that this quantity is well-defined with probability one. This completes the proof of Lemma 4.

Remark 6 (On the relationship between OPE and stochastic intervention efect estimation). It is no surprise that the proof of Lemma 4 follows Robins [43] and Kennedy [31] who study stochastic intervention efects in causal inference. Indeed, OPE and estimation of stochastic interventions are two diferent framings of essentially the same problem, and use the same importance-weighted and doubly robust estimators. The main diferences between these fields lie in their emphases: the former is focused on adaptive experiments where logging policies are data-adaptive and known, whereas the latter typically places more emphasis on potential outcomes and causal identification, observational studies (i.e. where logging policies must be estimated), and more complex causal functionals, such as those of Haneuse and Rotnitzky [20] and Kennedy [31]. Of course, these are incomplete characterizations made with broad strokes; for a more detailed summary of prior work in stochastic interventions, see Kennedy [31, Section 1].

Remark 7 (Implications for design-based confidence sequences). As an alternative to estimating treatment efects in superpopulations, one can opt to consider a so-called “design-based” approach to causal inference where the potential outcomes of all individuals are conditioned on, and confi dence intervals are constructed for the sample average treatment efect (SATE) given by $\mathrm { S A T E } _ { t } : =$ $\begin{array} { r } { \frac { 1 } { t } \sum _ { i = 1 } ^ { t } ( R _ { i } ( 1 ) - R _ { i } ( 0 ) ) } \end{array}$ where $R _ { i } ( a )$ is subject i’s potential outcome under treatment $a \in \{ 0 , 1 \}$ . Here, the resulting confidence intervals cover the SATE with high probability, where the probability is taken with respect to the randomness in the treatment assignment mechanism only. The design-based approach goes back to Fisher and has a deep and extensive literature [16, 38, 26], and more recent work has constructed nonasymptotic CSs for the time-varying efect $( \mathrm { S A T E } _ { t } ) _ { t = 1 } ^ { \infty }$ in Howard et al. [24, Section 4.2] and asymptotic ones in Ham et al. [19]. For a more comprehensive literature review, we direct readers to Abadie et al. [1] and Ham et al. [19] as well as the references therein.

We simply remark here that the results of Section 3 simultaneously apply to the design-based and superpopulation settings as immediate corollaries. Indeed, in the stochastic (non-design-based) setting for binary experiments and under the causal identification assumptions IA1–IA3, we have that Lemma 4 yields

$$
\Delta_ {t} ^ {\star} := \frac {1}{t} \sum_ {i = 1} ^ {t} \mathbb {E} \left[ R _ {i} (1) - R _ {i} (0) \right] = \frac {1}{t} \sum_ {i = 1} ^ {t} \left[ \mathbb {E} (R _ {i} \mid A _ {i} = 1) - \mathbb {E} (R _ {i} \mid A _ {i} = 0) \right] =: \Delta_ {t}.\tag{102}
$$

Now, to recover CSs for $( { \mathrm { S A T E } } _ { t } ) _ { t = 1 } ^ { \infty }$ we simply condition on $( R _ { t } ( 1 ) , R _ { t } ( 0 ) , X _ { t } ) _ { t = 1 } ^ { \infty }$ so that $( A _ { t } ) _ { t = 1 } ^ { \infty }$ are the only non-degenerate random variables here. The techniques for time-varying treatment efects described in Section 3.1 and Section 3.2, yield a $\left( 1 - \alpha \right) - \mathrm { C S } \left[ L _ { t } , U _ { t } \right] _ { t = 1 } ^ { \infty }$ for $( \Delta _ { t } ) _ { t = 1 } ^ { \infty } \equiv ( \Delta _ { t } ^ { \star } ) _ { t = 1 } ^ { \infty }$ and hence for $( { \mathrm { S A T E } } _ { t } ) _ { t = 1 } ^ { \infty }$ . Going further, when instantiated for the design-based setting, our CSs substantially improve on Howard et al. [24, Section 4.2], both practically and theoretically. Indeed, as discussed in Ham et al. [19, Section 3.2], one of the drawbacks of existing nonasymptotic CSs in the literature is that the minimal propensity score — i.e. $\begin{array} { r } { p _ { \operatorname* { m i n } } : = \exp \operatorname { i n f } _ { t , a , x } \mathbb P ( A _ { t } = a \ | \ x \ | \ \mathcal H _ { t - 1 } ) \ - } \end{array}$ must be specified in advance, and the downstream CSs always scale with $p _ { \mathrm { m i n } } ^ { - 1 }$ . However, as we have emphasized throughout this paper, beginning with desideratum 5 in Section 1.2, none of our CSs sufer from this limitation.

Simultaneously, if we consider the superpopulation setting where $\mathbb { E } ( R _ { t } ( 1 ) ) - \mathbb { E } ( R _ { t } ( 0 ) ) = \delta$ for all $t \geqslant 1$ and for some $\delta \in [ - 1 , 1 ]$ , then under identification assumptions IA1–IA3, the same CS $[ L _ { t } , U _ { t } ] _ { t = 1 } ^ { \infty }$ also covers δ by Lemma 4. In this way, our time-varying CSs simultaneously handle the stationary superpopulation setting where treatment efects do not change over time, as well as the design-based setting where all potential outcomes are conditioned on, since these are both special cases of the time-varying stochastic setting considered in Section 3.
