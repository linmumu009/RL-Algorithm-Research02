# Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning

Anthony GX-Chen<sup>1,\*,+</sup>, Ankit Anand<sup>2,\*</sup>, Gheorghe Comanici<sup>2,\*</sup>, Zaheer Abbas<sup>2</sup>, Eser Aygün<sup>2</sup>, David Smalling<sup>2</sup>, Shibl Mourad<sup>2</sup>, Doina Precup<sup>2</sup>, André Barreto<sup>2,\*</sup> and Mark Rowland<sup>2,</sup>

<sup>1</sup>New York University, <sup>2</sup>Google DeepMind, <sup>+</sup>Work done as a student researcher at Google DeepMind, <sup>\*</sup>Core Contributor

Classical reinforcement learning (RL) typically seeks a deterministic policy that maximizes the expected sum of a scalar reward. Yet, modern applications such as language model fine-tuning or scientific discovery demand diversity. Existing remedies such as entropy regularization or diversity bonuses often require fragile trade-ofs that sacrifice performance for stochasticity or rely on heuristic metrics that can misalign policy rankings. We argue that diversity is more naturally understood as the rational response to uncertainty in the reward. When the reward function is not perfectly known—as is the case with ambiguous preferences or imperfect reward models—committing to a single action can be sub-optimal. Building on this, we propose a fundamental reformulation of the RL objective by replacing the scalar reward with a distribution over reward functions, and applying a non-linear objective over sets of actions. The result is a framework in which calibrated behavioural diversity emerges naturally, remains controllable through the reward function distribution, and is obtained without sacrificing expected reward. Focusing on the contextual bandit setting, we derive a principled gradient estimator for this objective and prove that our formulation naturally generalizes both vanilla policy gradient and more recently developed action-set approaches. Our empirical results demonstrate that this framework ofers a robust and theoretically grounded alternative for complex RL tasks where the traditional formulation of the problem fails to induce the desired breadth of agent behaviour.

## 1. Introduction

Classical reinforcement learning (RL) theory rests on the foundational assumption that preferences can be captured by the expected sum of a scalar reward function. A well-known consequence of this assumption is the existence of an optimal deterministic policy (Puterman, 1994), and, accordingly, most standard RL methods are designed to converge to such a solution. However, as RL is increasingly applied to open-ended domains, convergence to a single behaviour has become a significant limitation. In settings such as the fine-tuning of large language models (LLMs), it is often essential to maintain the diversity of the pre-trained policy in order to preserve its usefulness for creative tasks and chainof-thought reasoning (Chung et al., 2025; Cui et al., 2025; Ismayilzada et al., 2025; Kirk et al., 2024; Shypula et al., 2025; West and Potts, 2025; Yang and Holtzman, 2025; Yun et al., 2025; Zhao et al., 2025). Beyond language, the capacity for inference-time exploration is vital for scientific discovery, where agents must navigate vast search spaces to identify rare, high-value solutions (Aygün et al., 2025; Hubert et al., 2026; Romera-Paredes et al., 2024). Furthermore, convergence to a deterministic optimal policy is detrimental when the reward model itself is only an imperfect proxy for some true underlying preference, and over-optimization results in the deterioration of generation quality (Coste et al., 2024; Gao et al., 2022; Lambert, 2026; Moskovitz et al., 2024; Rafailov et al., 2024).

Existing approaches for inducing diversity include using entropy regularization or diversity bonuses. Both families of approaches can be thought of as modifications to the original reward function, resulting in policies that are highly sensitive to the explicit choice of diversity measures, and must be carefully tuned. A variety of approaches make use of policy entropy penalties (Haarnoja et al., 2017;

Mnih et al., 2016; Todorov, 2006; Williams and Peng, 1991; Ziebart et al., 2008), although this forces a trade-of, in which a more stochastic policy can be obtained at the cost of reduced expected rewards (Jhaveri et al., 2025). Diversity bonuses that get added to the rewards similarly introduce a tension between optimizing the original reward function and focusing on the bonus (Hamid et al., 2026; Li et al., 2025; Orney et al., 2026), and may result in undesirable ranking of sub-optimal policies, as we discuss further below.

In this work, we propose a shift in perspective: diversity should be understood as the rational response to reward uncertainty. For instance, in the case of generating diverse, creative model outputs, the uncertainty arises from ambiguity in the user’s preferences, which are often under-specified by the input prompt alone. In the context of mitigating reward model over-optimization, this uncertainty reflects an epistemic gap concerning the definitive reward function. In each of these cases, the learning objective is more holistically described by considering a distribution over reward functions, rather than by augmenting a single reward function.

However, standard policy gradient updates over a distribution of reward functions often collapse the policy to average, non-specialized behaviour. In response to this, we propose to use policy gradients that depend non-linearly on sets of actions (Hamid et al., 2026; Tang et al., 2025) and sampled reward functions. This approach allows the policy to flexibly learn diverse behaviours corresponding to diferent regions of the distribution of reward functions, without collapsing behaviour onto optimal actions for the most frequently encountered reward function. Concretely, focusing on the contextual bandit setting, we derive a principled gradient estimator for this objective and prove that our formulation naturally generalizes both vanilla policy gradient and recently developed approaches based on action sets. Our empirical results demonstrate that this framework provides a more robust and natural foundation for complex RL settings, where simple return maximization fails to induce the desired breadth of agent behaviour.

## Our core contributions are as follows:

• We introduce a new family of RL objectives that provides calibrated diversity control through a distribution over reward functions, which we refer to as Randomized Objectives, Set Actions (ROSA) (Sec. 3).

• We provide an analysis of this new family of objectives, both in terms of the global optimizing policies and the landscape of the objectives themselves. This simultaneously provides theoretical guarantees for the objectives and informs useful hyperparameter settings in practice (Sec. 4).

• We complement this analysis with a variety of empirical studies, both in toy domains, to develop a fine-grained understanding of the method’s performance, and in larger-scale domains, to demonstrate the broad applicability of the proposed objectives in practical scenarios (Sec. 5).

## 2. Background

We consider a contextual bandit scenario (Langford and Zhang, 2007; Sutton and Barto, 2018) defined over action space Y and state space X. This encompasses both abstract bandit problems, as well as large-scale generative modelling such as single-turn fine-tuning of language models. A typical RL problem is specified via a reward function $R : X \times y  \mathbb { R }$ , and the goal is to optimize a policy $\pi : \chi \to { \mathcal { P } } ( y )$ according to the mean reward criterion for some state distribution $\mu ,$

$$
\mathcal {J} (\pi) = \mathbb {E} _ {X \sim \mu , Y \sim \pi (\cdot | X)} [ R (X, Y) ].\tag{1}
$$

Policy gradient (PG) methods are common means of optimizing the policy � with respect to the objective in Eq. (1). The prototypical PG method is REINFORCE (Williams, 1992); given a parametrized policy $\pi _ { \theta }$ and a state–action sample $( X , Y )$ drawn according to $\pi _ { \boldsymbol { \theta } _ { \mathrm { : } } }$ , the algorithm updates the parameters � in the direction

$$
\nabla_ {\theta} \mathcal {J} _ {\mathrm{PG}} (\pi_ {\theta}) = \mathbb {E} _ {X \sim \mu , Y \sim \pi_ {\theta} (\cdot | X)} \left[ R (X, Y) \nabla_ {\theta} \log \pi_ {\theta} (Y | X) \right].\tag{2}
$$

## 2.1. Inducing diversity with policy gradients

As described in Section 1, there are many applications of reinforcement learning where we want to maintain some level of diverse behaviour after training, rather than collapsing to a deterministic policy. Before introducing the core algorithmic proposal of the paper, we briefly review several existing families of approaches that aim to achieve this.

Entropy regularization (Haarnoja et al., 2017; Mnih et al., 2016; Todorov, 2006; Williams and Peng, 1991; Ziebart et al., 2008) applies a (scaled) entropy bonus $\begin{array} { r } { \mathcal { H } ( \pi ( \cdot | x ) ) = - \sum _ { y \in y } \pi ( y | x ) \log ( \pi ( y | x ) ) } \end{array}$ to the basic objective in Eq. (1). This prevents policy gradient updates from reaching a deterministic policy, and the optimal policy is guaranteed to place non-zero mass on all actions.

Diversity bonuses introduce an additional diversity function, quantifying a level of diversity among a given collection of actions. Typically, the integrand in Eq. (1) is then modified by considering a sequence of sampled actions $Y _ { 1 : n }$ at a given state �, and modifying rewards based on the diversity function. This can be implemented on a per-sample basis (see, e.g., Li et al., 2025), with a pairwise diversity function $d : y ^ { 2 } $ ℝ and modified reward $R ( X , Y _ { i } ) \textstyle \sum _ { j \neq i } d ( Y _ { i } , Y _ { j } )$ , or at a group level (Hamid et al., 2026), with a group diversity function $D : { \bf { \mathcal { Y } } } ^ { n }  [ 0 , { \stackrel { . } { \infty } } )$ , and modified reward $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } R ( X , Y _ { i } ) D ( Y _ { 1 : n } ) } \end{array}$

Multi-objective reinforcement learning (Hayes et al., 2022; Roijers et al., 2013) specifies a finite collection of reward functions $R _ { 1 } , \ldots , R _ { d }$ , and then optimizes a scalarized objective. This is typically done via scalarized expected returns (SER) or expected scalarized returns (ESR), taking the form $s ( \mathbb { E } _ { Y \sim \pi } [ \mathbf { R } ] )$ , or, respectively, $\mathbb { E } _ { Y \sim \pi } \left[ s ( \mathbf { R } ) \right]$ . Here, $\mathbf { R } = ( R _ { 1 } ( Y ) , \ldots , R _ { d } ( Y ) )$ is the vector of rewards, and $s : \mathbb { R } ^ { d } \to \mathbb { R }$ is a (possibly non-linear) scalarization function.

## 3. Optimizing behaviour for distributions of reward functions

Our goal is to develop a policy gradient method that learns diverse behaviours capable of reflecting three key criteria: (a) uncertainty in the objective function; (b) variance in human judgment regarding what constitutes a high-quality action $( \mathrm { e . g . }$ , scenarios where a single query yields multiple valid answers); and (c) a strategic desire to maintain policy flexibility for future fine-tuning, such as adapting to individual user preferences. While the existing approaches outlined in Section 2.1 serve foundational algorithmic roles in reinforcement learning, they exhibit several shortcomings when applied to the specific use cases described above. We begin by analyzing these limitations in detail (further discussed in Appendix C and summarized in Table 1).

Inclusion of poor-quality actions in the optimal policy. The optimal policy for the entropyregularized objective puts non-zero mass on all actions, including ones with very bad rewards. This is also true for some variants of diversity-bonus methods.

Undesirable ordering of sub-optimal policies. Many approaches induce undesirable orderings of sub-optimal policies: a policy with lower average reward (but higher diversity) can be preferred over a policy with higher average reward.

Inability to express a stochastic optimal policy. A number of objectives, including vanilla policy gradient, pass-at-�/best-of-� gradient, and the expected scalarized return (ESR) always have an optimal deterministic policy, and so cannot be guaranteed to induce diversity.

Unavailability of unbiased gradients via direct sampling. The scalarized expected return (SER) multi-objective RL formulation typically is dificult to optimize, due to the unavailability of unbiased sample-based gradient estimators of nonlinear objectives over expectations, $s ( \mathbb { E } [ \cdot ] )$

In summary, for our problem of learning a policy that reflects diverse behaviour under the motivations outlined above, these existing approaches sufer several drawbacks: inexpressivity of the objective, incorrect optimal/sub-optimal policy orderings, and optimization problems.



(a) Examples of distributions of reward functions (b) ROSA+Max optimizing a policy for a distribution of and the desired policy. (Left) A binary reward set- reward functions). Given some policy (orange) and reward ting with a variety of correct actions; we prefer a functions $R _ { k }$ (pink), we first compute the policy’s max-of-� policy that places probability mass on all correct performance for each reward function, $J _ { k }$ (green). We then actions. (Right) We only have access to approxi- take the dot product with the reward function probabilities mate reward functions $( \hat { R } ^ { \prime } s ,$ , shaded lines). There is $\left( \rho ( R _ { k } ) \right.$ , blue) to get the ROSA+Max score. Optimizing this epistemic uncertainty about the value of the true score calibrates the policy to the reward function uncertainunderlying reward (bold line), and we wish to avoid ties. Appendix C contains more details and comparisons overfitting to the approximate �ˆ. with regular PG.

Figure 1 | Many complex tasks are naturally expressed as distributions over reward functions.

## 3.1. The Randomized Objectives, Set Actions (ROSA) criterion

We now make our core algorithmic proposal, aiming to close the drawbacks described in the previous section. We build on the approach of working with multiple reward functions that express our desired diverse behaviours, and develop an approach that (i) naturally allows for arbitrary distributions over reward functions, (ii) avoids inducing learnt policies to take bad actions solely for the sake of increased diversity, (iii) has straightforwardly-implementable, unbiased policy gradient estimates. The key idea of our proposal is to generalize Eq. (1) by considering (i) multiple actions sampled i.i.d. from the policy � (Hamid et al., 2026; Tang et al., 2025); and (ii) a distribution $\rho$ over reward functions �. This gives rise to the family of Randomized Objectives, Set Actions (ROSA) criteria.

Formally, we let $\rho \in \mathcal { P } ( \mathbb { R } ^ { X \times y } )$ be a distribution over reward functions, and let $\mathbf { Y } = ( Y _ { i } ) _ { i = 1 } ^ { n }$ denote a multiset of actions induced by $Y _ { 1 } , \dots , Y _ { n }$ . We use tuple notation to signify multiplicities are retained, but any orders are ignored. We use the shorthand $\mathbf { Y } \sim \pi _ { \theta }$ to denote the � actions are sampled i.i.d. from policy $\pi _ { \theta }$ . Let $R \sim \rho$ a sample from the distribution over reward functions. For brevity, we denote the reward multiset as $R ( \mathbf { Y } ) = \left( R ( Y _ { i } ) \right) _ { i = 1 } ^ { n }$ . Our primary objective is the ROSA+Max criterion,

$$
\mathcal {J} _ {\mathrm{ROSA+Max}} (\pi_ {\theta}) = \mathbb {E} _ {X \sim \mu , \mathbf {Y} \stackrel {\text {i.i.d.}} {\sim} \pi_ {\theta} (\cdot | X)} \left[ \mathbb {E} _ {R \sim \rho} \left[ \max _ {1 \leq i \leq n} R (X, Y _ {i}) \right] \right],\tag{3}
$$

where the max function aggregates over the reward of the � actions in a permutation-invariant way for each reward function. Figure 1b contains an example in the case of a discrete set of reward functions.

Intuitively, the ROSA+Max criterion provides an asymmetric summary of performance on each sampled reward function, encouraging at least one action in the multiset to perform well in each case. This is related to the expected utilities used in multi-objective RL, described above, but difers in several important ways: (i) the form of optimism used is strictly more expressive than that is typically used in MORL (see Appendix C) leading to more desirable optimal policies, (ii) the objective has a straightforwardly derivable unbiased policy gradient estimator (see below), and (iii) the objective naturally supports arbitrary distributions over reward functions (rather than an explicitly declared finite number of reward functions). We will see shortly in Section 3.2 that the optimum of this objective is indeed a policy that assigns action probabilities over the set of actions that are optimal under each $R \sim \rho$

We now derive an unbiased, variance-reduced policy gradient estimator to optimize our proposed objective. Given � sampled reward functions $( R _ { k } ) _ { k = 1 } ^ { m }$ from $\rho _ { z }$ , and � sampled actions $( Y _ { i } ) _ { i = 1 } ^ { n }$ from $\pi _ { \theta } ( \cdot | X )$ , an unbiased gradient estimator is given by:

$$
\hat{g} (\theta) = \frac{1}{m}\sum_{k = 1}^{m}\left[\sum_{i = 1}^{n}\Big(\max_{1\leq j\leq n}R_{k}(X,Y_{j}) - \max_{\substack{1\leq j\leq n,\\ j\neq i}}R_{k}(X,Y_{j})\Big)\nabla_{\theta}\log \pi_{\theta}(Y_{i}|X)\right],\tag{4}
$$

which has the efect of only the maximal reward action contributing a non-zero term to the estimator. Note the term $\scriptstyle \operatorname* { m a x } _ { 1 \leq j \leq n , j \neq i } R _ { k } ( Y _ { j } )$ is an optional control variate for variance reduction, which is analogous (in the single reward function setting) to the estimator used in Tang et al. (2025). We provide derivations for a generalization of the above gradient estimator (for general set functions and control variates) and all other theoretical results in the main paper in Appendix A.

Remark 3.1. While set-function objectives based on best-of-� have been investigated in the singlereward setting as a means of inducing diversity (see, e.g., Tang et al. 2025; Walder and Karkhanis 2026), we emphasize that these objectives do not induce a stochastic optimal policy. There remains a deterministic optimal policy for the single reward function, though the set-function objective may induce a re-ordering of sub-optimal policies that increases the objective value of certain stochastic policies. By contrast, in our setting, the combination of randomized reward functions and set function objectives induces an optimal stochastic policy, which balances the incentives of the randomized reward functions.

## 3.2. Theoretical analysis of ROSA+Max

Having introduced the ROSA+Max update in Eq. (4), we now examine it theoretically. Without loss of generality, we consider the single-state case, and drop state-dependence from our notation. Our theoretical analysis will assume each reward function $R \sim \rho$ is binary and “one-hot”, assigning reward 1 to a distinct optimal action $y ^ { * } \in \mathcal { Y }$ and 0 otherwise. We note that the ROSA objective and gradient estimator support arbitrary reward functions in practice.

Proposition 3.2. Optimal policy of ROSA+Max with uniform reward function distributions. Consider � binary reward functions $( r _ { i } ) _ { i = 1 } ^ { m } .$ , each with a single distinct optimal action $y _ { i } ^ { * } \in \mathcal { Y } .$ , and set $\rho$ uniform over $( r _ { i } ) _ { i = 1 } ^ { m }$ . Then, writing $\delta _ { y }$ for the Dirac delta distribution at $y ,$ the ROSA+Max objective with any action set size $n \geq 2$ has unique optimal policy

$$
\pi^ {*} = \sum_ {i = 1} ^ {m} \frac {1}{m} \delta_ {y _ {i} ^ {*}}.
$$

This result verifies that the ROSA+Max objective achieves our intended goal of having the maximally diverse policy over correct responses at its unique optimizer. To develop intuition further, we numerically simulate the global objective landscape over all policies in an example with two reward functions (Figure 2). We compare ROSA with other diversity-inducing RL objectives, as well as the standard PG objective (Figure 2a), that does not induce any kind of diversity among optimal actions.

(a) Vanilla PG

(b) Entropy Reg

(c) Mult. Diversity

(d) ROSA+Max
Figure 2 | Simplices of global objective landscapes over 3-action categorical policies for four RL objectives. Actions A and B receive reward +1, and C receives reward 0. Details in Appendix D.1.

Firstly, adding entropy regularization to a standard RL objective (Haarnoja et al., 2017; Todorov, 2006; Ziebart et al., 2008) changes the ordering of preferred policies to favour non reward-maximizing policies with higher entropy (Figure 2b), and disproportionally favour actions with minor improvements in rewards (GX-Chen et al., 2026). We also consider diversity bonuses (Hamid et al., 2026; Li et al., 2025; Orney et al., 2026), such as in the form of a multiplicative diversity bonus, e.g., $\bar { r } ( X , Y _ { i } ) = r ( X , Y _ { i } ) \mathrm { D i v } ( X , { \bf Y } )$ where Div(�, Y) is the average distance to all other samples in the action multiset. This family of objectives can induce a global optimum near/at a reward maximizing policy, but re-orders sub-optimal policies where a high-diversity, low-reward policy can be favoured over a high-reward policy (Figure 2c). Finally, ROSA+Max does not have either of these issues: its gradient points solely in the direction of both higher reward and higher action diversity (Figure 2d). It should be noted that both “diversity bonus” methods and ROSA require more information than the original reward function, while entropy regularization does not; diversity bonus requires a diversity function, while ROSA requires a distribution of reward functions.

## 3.3. Choosing the action set size

It is interesting to note that the action sampling parameter � in Prop. 3.2 does not afect the optimal policy, meaning that for a uniform reward function distribution, any $n \geq 2$ induces a maximally diverse policy over all reward functions; thus, the global optimum can be obtained with small parallel sampling budgets (e.g., just � = 2 i.i.d. action samples from $\pi _ { \boldsymbol { \theta } } )$ . However, in practice we expect the selection of this parameter to be important for algorithmic performance. To build some intuition for why this is, we can calculate the curvature of the objective at the optimal policy.

Lemma 3.3. Under the assumptions of Proposition 3.2, the Hessian of the objective at the optimal policy is diagonal, with all diagonal elements given $b y - n ( n - 1 ) ( 1 - { \frac { 1 } { m } } ) ^ { n - 2 }$

The term in Lemma 3.3 has absolute value of 2 when � = 2 for all �, is maximized (in absolute value) when $n \approx 2 m$ , and decays toward 0 as $n \gg 2$ . This fits the intuition that a large � results in a flat landscape around the optimum, slowing optimization. It also suggests $2 \leq n < 2 m$ as a sensible hyperparameter range to obtain a steep objective landscape around the optimal policy (when � is known). We visualize and discuss this more in Appendix B.1.

## 4. Beyond the max: ROSA with general distributions and set functions

Having established a result on the optimal solution in the case of uniform distribution over rewards and the max set function in Proposition 3.2, we now generalize it in two ways: to a general family of set functions, and to non-uniform reward function distributions.

## 4.1. A family of set functions with maximally diverse optimal ROSA policies

So far, we have shown that the ROSA+Max criterion has the desirable property of having an optimal policy that places mass equally over all optimal answers. The reader may wonder if set functions other than “max” can be used to aggregate over rewards $( R ( Y _ { i } ) ) _ { i = 1 } ^ { n }$ . Here, we present an additional theoretical result which complements Proposition 3.2 and establishes this optimal policy guarantee for a more general family of set functions.

We focus our analysis on the binary reward setting. Under binary rewards, any general multi-set function $f ( R ( X , Y _ { 1 } ) , \dots , R ( X , Y _ { n } ) )$ can be simplified into a success-count rewardfunction $\scriptstyle { \tilde { f } } ( \sum _ { i = 1 } ^ { n } R ( X , Y _ { i } ) )$ depending only on the success count $\scriptstyle \sum _ { i = 1 } ^ { n } R ( X , Y _ { i } ) $ ; because the individual rewards are binary, the function’s value is determined entirely by the number of correct actions.

Theorem 4.1. Optimal policy for general $f .$ Consider � uniformly distributed (binary, distinct) reward functions $( r _ { i } ) _ { i = 1 } ^ { m } ,$ each with a single distinct optimal action $y _ { i } ^ { * } \in \mathcal { Y } ,$ , and a set function $f ,$ with corresponding success-count reward function $\tilde { f }$ (defined above) which is strictly increasing as well as strictly concave in the sum of rewards. Then the $R O S A + f$ objective has a global optimum which samples correct actions from all reward functions uniformly,

$$
\pi^ {*} = \frac {1}{m} \sum_ {i = 1} ^ {m} \delta_ {y _ {i} ^ {*}}.
$$

Theorem 4.1 allows the construction of many diferent kinds of set functions which all have the same optimum at the maximally diverse policy, yet can have vastly diferent optimization properties. One function from this family is the Softmax, which is a natural extension of the max that varies smoothly between $f _ { \mathrm { m a x } }$ and $f _ { \mathrm { m e a n . } }$ ,

$$
f _ {\text { Softmax }} \left(\left(R (X, Y _ {i})\right) _ {i = 1} ^ {n}\right) = \sum_ {i = 1} ^ {n} \frac {\exp \left(R (X , Y _ {i})\right)}{\sum_ {j = 1} ^ {n} \exp \left(R (X , Y _ {j})\right)} R (X, Y _ {i}).\tag{5}
$$

Numerical simulations in Figure 3 illustrate that, like the Max, the Softmax also has a maximally diverse global optimum, though with diferent optimization landscape. Notably, the Softmax function also removes the sparsity of the variance-reduced gradient in Eq. (4) in the case of the max set function. We leave exploration of other set functions for future work and focus primarily on $f _ { \mathrm { M a x } }$ and $f _ { \mathrm { S o f t m a x } }$ in the next sections. See also Verdun et al. (2025) for the use of softmax distributions as an alternative to best-of-� sampling in the single-reward setting.

Figure 3 | ROSA + Softmax simplex in same setting as Figure 2.

## 4.2. Non-uniform reward function distribution induces controllable diversity

We now consider non-uniform weightings in the reward function distribution $\rho .$ In this setting, the relationship between $\rho$ and optimal policy action probabilities is more complex. We derive a specific result for the ROSA+Max objective and leave the extension to other set functions for future work.

Proposition 4.2. Consider � distinct, binary rewardfunctions $( r _ { i } ) _ { i = 1 } ^ { m }$ (each with a single distinct optimal action $y _ { i } ^ { * } \in \mathcal { Y } )$ with probabilities $( \alpha _ { i } ) _ { i = 1 } ^ { m }$ under $\rho ,$ and assume without loss ofgenerality that $\alpha _ { 1 } \geq \cdots \geq \alpha _ { m } .$ The ROSA+Max objective with $n \geq 2$ action samples has an optimal policy which samples optimal actions from the top-k reward functions (ordered by �’s), each with probability,

(a) Vanilla PG
(b) ROSA+Max
Figure 4 | Global objective landscape over simplex of 3-category policies, with non-uniformly weighted reward functions $R _ { A }$ and $R _ { B }$ . Vanilla PG can only induce global optima at deterministic policies, while ROSA+Max naturally induces global optima at optimal stochastic policies. The probability of sampling each unique optimal action is characterized in Proposition 4.2.

$$
p _ {i} ^ {*} = 1 - \frac {(k - 1) \alpha_ {i} ^ {- 1 / (n - 1)}}{\sum_ {j = 1} ^ {k} \alpha_ {j} ^ {- 1 / (n - 1)}},\tag{6}
$$

where � is the largest integer between 1 and � where $p _ { k } ^ { * }$ is positive.

Proposition 4.2 allows us to predict, given reward function probabilities $( \alpha _ { 1 } , . . . , \alpha _ { m } )$ , how frequently the optimal policy under ROSA+Max will sample from each of the distinct correct answers (corresponding to each reward function). This gives us the ability to precisely predict and/or control the diversity in the optimal policy through the reward function distribution. Figure 4 numerically illustrates this point: the ROSA+Max optimum smoothly interpolates between uniformly sampling optimal actions, to preferentially sampling the optimal action under the more probable reward function.

## 5. Experiments

We now evaluate empirically how our proposed objective function, ROSA+Max, and its variant ROSA+Softmax compare with the standard policy gradient objective and other variants in experiments meant to illustrate a variety of real-world applications.

A practical ROSA+Max gradient update is provided in Alg. 1. Although we write it iteratively for clarity, it can be implemented eficiently via matrix operations (over � actions and � sampled reward functions). Note that the same algorithm can be applied to any (multi)set functions by replacing the max with the desired set function.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 ROSA+Max single-state policy update

Require: Policy  $\pi_{\theta}$ , state X, reward functions  $\{R_{k}\}_{k=1}^{m}$  each with probabilities  $\{\alpha_{k}\}_{k=1}^{m}$ .

1: Sample actions  $Y_{1}, \ldots, Y_{n} \stackrel{\text{i.i.d.}}{\sim} \pi_{\theta}(\cdot \mid X)$ .

2: for each action  $Y_{i}$  and reward function  $R_{k}$  do

3: Calculate the effective advantage:

 $A_{i}^{(k)} = \max_{1 \leq j \leq n} R_{k}(X, Y_{j}) - \max_{\substack{1 \leq j \leq n \\ j \neq i}} R_{k}(X, Y_{j})$

4: end for

5: Compute gradient estimator:

 $\hat{g} = \sum_{i=1}^{n} \left( \sum_{k=1}^{m} \alpha_{k} \cdot A_{i}^{(k)} \right) \nabla_{\theta} \log \pi_{\theta}(Y_{i} \mid X)$

6: Update weights  $\theta' \leftarrow Optimizer(\theta, \hat{g})$
</div>

## 5.1. Diverse, competing preferences

We first investigate the regime of conflicting preferences, where $R \sim \rho$ represents a distribution over diverse, contradictory reward functions. In this setting, vanilla policy gradient is fundamentally incapable of optimization: by aggregating the return as $\bar { r } ( Y ) = \mathbb { E } _ { R \sim \rho } [ R ( Y ) ]$ , opposing reward signals mutually cancel out, resulting in no reward.

(a) Sequence generation task with competing reward functions. $R _ { 1 }$ prefers answers with the same parity as the last question token, $R _ { 2 }$ prefers opposite parity.

(b) Evaluation result of diferent gradient estimators.

Figure 5 | Comparison of vanilla PG, ROSA+Max and ROSA+Softmax in scenario of competing preferences. PG (even if augmented with a multiplicative diversity reward, Mult. Div. Bonus) sufers from reward cancellation, resulting in no learning—its pass@� curve is identical to the Base (untrained) model. However, ROSA+Max and ROSA+Softmax cleanly balance both preferences: starting at ∼50% accuracy and generating both answer types as � increases. Details in Appendix D.2.

We construct a didactic experiment with a small transformer outputting only integers. There are two competing reward functions: $R _ { 1 }$ is +1 if all generated integers have the same parity (i.e. same odd/even-ness) as the last integer in the prompt, −1 if all generated integers have the opposite parity as the last integer in the prompt, and 0 if the generated integers have mixed parities. �<sub>2</sub> is the negative of $R _ { 1 }$ (see Figure 5a for example and explanation). As Figure 5 demonstrates, ROSA is the only method capable of making any progress at all in this setting.

## 5.2. Multiple correct answers

Many tasks in the real world can naturally contain multiple correct answers (or correct actions). For instance, a coding task asking the user to “reverse a Python list” can be implemented in a number of syntactically distinct, yet functionally equivalent ways, or a task of discovering a drug having a high reward (defined by some properties) may have multiple drug molecules achieving the highest reward. In these scenarios, the user is not interested in finding a deterministic policy which achieves the highest reward on one solution but finding a stochastic policy which discovers many correct solutions.

We can model this scenario by defining a reward function distribution over multiple reward functions, each corresponding to a unique, correct answer. Note that optimizing the sum/average i.e. “joint” reward function $\begin{array} { r } { \bar { R } ( Y _ { i } ) = \sum _ { a \in \mathcal { A } ^ { * } } R _ { a } ( Y _ { i } ) } \end{array}$ , reduces to a regular reward function that gives $^ { \mathfrak { s } } \mathbb { 1 } ^ { \mathfrak { p } }$ to all correct answers $Y _ { i } \in \mathcal { A } ^ { * }$ , and 0 otherwise. We construct a setting in mathematical reasoning where we prefer answers that are correct, but we want diversity in the answer lengths: short answers are concise and to the point, while long answers can be more didactic and detailed. Specifically, we train with four diferent reward functions with diferent length preferences. Figure 6 shows the output pass@� from sampling the resulting policies, where ROSA achieves the same overall correctness as regular policy gradient training, but is able to sample from all length preferences specified. We show short and long example responses from a ROSA trained policy in Figure 17.

## 5.3. Learning with unknown ground-truth rewards

We now consider the case where we are uncertain about the true reward function. This naturally occurs in reward modelling, or if LLMs are used as the reward function. Here, due to finite data or stochasticity in the judge, we only have an approximately correct reward function. This epistemic uncertainty is naturally expressed as a reward function distribution $R \sim \rho ,$ and we will subsequently show that ROSA+Max/Softmax naturally captures the uncertainty in the reward function in the resulting policy’s action distribution. We first set up a didactic task where there are multiple reward functions that do not always agree. Specifically, we consider the case of one-dimensional state � and action �; see Figure 7(left), illustrating multiple diverse, non-equally weighted reward functions. Figure 7(right) displays the results of training ROSA+Max as well as PG with low/high entropy regularization. PG training with low-entropy regularization converges to the highest weighted reward functions, while high-entropy regularization results in wider coverage but also action probabilities over many bad actions that are not optimal under any reward functions. ROSA+Max covers all reward modes in proportion to their reward function’s weight. See Appendix D.4 for more experimental details and additional figures comparing specific slices of �(�|�) between methods.


Figure 6 | Comparison of ROSA+Max/Softmax to PG in a setting where we train on MATH and with multiple reward functions preferring diferent lengths. Each reward function gives 1 if the answer is both correct and within its preferred character length. Training with vanilla PG results in generations mostly in the 400-1.6k length range, while the ROSA-trained policy samples from all four preferred length ranges, without sacrificing any accuracy (overall pass@� rate).
Figure 7 | Left: Reward functions; white lines denote maximum reward $y$ for each $x ,$ , with rewards decaying exponentially away from the maximum. Right: Trained policy �(�|�).

Finally, we extend our evaluation to an “ensemble of judges” setting for LLM post-training. We construct a mathematical reasoning task in which each problem is paired with a panel of eight judges. A response receives reward of 1 from a judge if it matches the judge’s target, though only some judges are correct, and the learner is not told which ones. We train under varying levels of judge noise, with 6, 3, or 1 correct judges out of 8 (corresponding to 75%, 37.5%, and 12.5% judges’ accuracy in Figure 8a). Across these settings, ROSA is more robust to noisy rewards than both standard PG and ensemble-regularized pessimistic baselines (Figure 8a). To test out-of-distribution generalization, we further evaluate models trained with noisy MATH judges on AIME2025. ROSA-trained models achieve higher out-of-distribution pass@� (Figure 8b).

(a) In domain (MATH) evaluation with diferentjudge mixtures

(b) Out-of-distribution pass@� performance on unseen AIME 2025 problem set

Figure 8 | Gemma 2B performance when trained on MATH where the reward is given by an ensemble of noisy judges. (8a) In distribution evaluation performance. (8b) Out-of-distribution pass@� evaluation on unseen AIME 2025 set. Ensemble min and stddev refer to taking min $( ( r _ { i } ) _ { i = 1 } ^ { m } )$ or $\arg ( ( r _ { i } ) _ { i = 1 } ^ { m } ) - \mathsf { s t d d e v } ( ( r _ { i } ) _ { i = 1 } ^ { m } )$ of the � rewards. All details are in Appendix D.5

## 6. Related work

ROSA generalizes several existing RL objectives. Specifically, in the single reward function case, ROSA with the set mean function $\begin{array} { r } { f _ { \mathrm { m e a n } } \big ( ( U _ { i } ) _ { i = 1 } ^ { n } \big ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } U _ { i } } \end{array}$ recovers the vanilla PG criterion (Eq. (1)), while the set max $f _ { \operatorname* { m a x } } \big ( ( U _ { i } ) _ { i = 1 } ^ { n } \big ) = \operatorname* { m a x } _ { 1 \leq i \leq n } \bar { U _ { i } }$ coincides with the “best-of-�” training objective (Stiennon et al., 2020). With multiple reward functions, the set mean function $f _ { \mathrm { m e a n } }$ recovers linear scalarization from multi-objective RL (Vamplew et al., 2011).

Improving diversity in post-training. Many recent works (GX-Chen et al., 2026; Huang et al., 2025; Padmakumar and He, 2023) have pointed to the problem of mode collapse in RL post-trained models compared to supervised fine-tuned models. Li et al. (2025) proposes to address this by jointly training the model on a reward function which is a product of diversity term and task reward. The diversity term is computed as a function of pairwise distances over a batch of samples for each prompt where the more unique samples are assigned a higher reward. While this approach preserves the diversity during post-training and shows significant gains in performance, we observe that this diversity bonus undesirably distorts the ordering of sub-optimal policies (Figure 2c). Similarly, Hamid et al. (2026); Orney et al. (2026) train on product of reward and diversity defined at a set level, at the cost of having a non reward-maximizing policy as the optimum of this objective (Figure 13). We discuss in detail in Appendix C.3.

Optimization over a set of responses. The best-of-� approach to inference-time policy improvement is a popular approach to improving LLM performance (Stiennon et al., 2020). Several recent works have studied the modification of RL objectives to take this inference-time technique into account (Balashankar et al., 2024; Beirami et al., 2025; Chen et al., 2026; Chow et al., 2024; Tang et al., 2025; Walder and Karkhanis, 2026), and other works use the implicit resulting improved policy in training objectives (Gui et al., 2024; Sessa et al., 2025). Distinct from our contributions of dealing with distribution of reward functions, these methods are concerned with optimization of a single reward function, and thus have deterministic policies as objective optima. Earlier related work in this vein is motivated by recommender systems (Radlinski et al., 2008), including online learning methods (Dimakopoulou et al., 2019; Kale et al., 2010; Yue and Guestrin, 2011) and deep reinforcement learning (Sunehag et al., 2015). Fard and Pineau (2011) study the problem of identifying all nearoptimal actions in MDPs via mixed-integer programming methods.

Multi-objective reinforcement learning. Multi-objective reinforcement learning (Hayes et al., 2022; Roijers et al., 2013) focuses on modelling problems that naturally have several, potentially conflicting, notions of reward. Key questions include optimization of reward subject to cost constraints (Altman, 1999), and discovery of policies on the Pareto frontier (Abdolmaleki et al., 2020; Bahlous-Boldi et al., 2026; Gábor et al., 1998; Mannor and Shimkin, 2001). Most closely connected to the problem we study is scalarization, in which a single scalar objective is constructed from several reward signals. However, these approaches are typically limited to expected utilities, which we show in Appendix C.2.1 are either insuficiently expressive to capture our desired diverse policies as solutions, or otherwise sufer from bias issues, unlike the ROSA objective proposed here. Concurrent with this paper, Bahlous-Boldi et al. (2026) propose a method for Pareto optimization that combines multiple reward functions and sets of actions; we discuss comparison with this work in more detail in Appendix C.2.2.

## 7. Conclusion

We presented a novel, generalized reinforcement learning framework that unifies optimization for sets of actions over distributions of reward functions. By shifting the objective away from maximizing expected scalar returns, we obtain policy gradient objectives whose reward-maximizing solutions are by design distributions, thereby resolving the tension between reward maximization and policy entropy. This addresses a variety of limitations with standard policy gradients, such as mode collapse and brittleness under reward function errors. We provide rigorous theoretical analysis for the behaviour, optimization eficiency, and optimal solution of this objective, and also provide empirical evidence at small and large scales for how ROSA can be applied to a wide variety of example settings. Ultimately, this work provides a principled foundation for training stochastic policies that are as diverse and adaptable as the environments they inhabit.

## Acknowledgement

The authors are grateful to Tom Schaul and Guillaume Desjardins for their thoughtful comments on a draft of the paper, and for insightful discussions with Yinlam Chow, Arian Hosseini, Tom Zahavy, Kalesha Bullard, and Andreas Kirsch. We would further like to thank the Agency team and the Montreal team at large for various inspirational conversations.

## References

A. Abdolmaleki, S. Huang, L. Hasenclever, M. Neunert, F. Song, M. Zambelli, M. Martins, N. Heess, R. Hadsell, and M. Riedmiller. A distributional view on multi-objective policy optimization. In Proceedings of the International Conference on Machine Learning, 2020.

M. Agarwal, V. Aggarwal, and T. Lan. Multi-objective reinforcement learning with non-linear scalarization. In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems, 2022.

E. Altman. Constrained Markov decision processes. Routledge, 1999.

E. Aygün, A. Belyaeva, G. Comanici, M. Coram, H. Cui, J. Garrison, R. Johnston, A. Kast, C. Y. McLean, P. Norgaard, Z. Shamsi, D. Smalling, J. Thompson, S. Venugopalan, B. P. Williams, C. He, S. Martinson, M. Plomecka, L. Wei, Y. Zhou, Q.-Z. Zhu, M. Abraham, E. Brand, A. Bulanova, J. A. Cardille, C. Co, S. Ellsworth, G. Joseph, M. Kane, R. Krueger, J. Kartiwa, D. Liebling, J.-M. Lueckmann, P. Raccuglia, X. Wang, K. Chou, J. Manyika, Y. Matias, J. C. Platt, L. Dorfman, S. Mourad, and M. P. Brenner. An AI system to help scientists write expert-level empirical software. arXiv preprint arXiv:2509.06503, 2025.

R. Bahlous-Boldi, I. Puri, I. Shenfeld, A. Kumar, M. Damani, S. Risi, O. Khattab, Z.-W. Hong, and P. Agrawal. Vector policy optimization: Training for diversity improves test-time search. arXiv preprint arXiv:2605.22817, 2026.

Q. Bai, M. Agarwal, and V. Aggarwal. Joint optimization of concave scalarized multi-objective reinforcement learning with policy gradient based algorithm. Journal of Artificial Intelligence Research, 74:1565–1597, 2022.

A. Balashankar, Z. Sun, J. Berant, J. Eisenstein, M. Collins, A. Hutter, J. Lee, C. Nagpal, F. Prost, A. Sinha, A. T. Suresh, and A. Beirami. InfAlign: Inference-aware language model alignment. In Proceedings of the International Conference on Machine Learning, 2024.

A. Beirami, A. Agarwal, J. Berant, A. D’Amour, J. Eisenstein, C. Nagpal, and A. T. Suresh. Theoretical guarantees on the best-of-� alignment policy. In Proceedings of the International Conference on Machine Learning, 2025.

V. J. Bowman Jr. On the relationship of the Tchebychef norm and the eficient frontier of multiplecriteria objectives. In Multiple Criteria Decision Making: Proceedings of a Conference Jouy-en-Josas, France May 21–23, 1975, pages 76–86. Springer, 1976.

Y. Chen, S. Chakraborty, L. Wolf, Y. Paschalidis, and A. Pacchiano. Post-training large language models for diverse high-quality responses. In Proceedings of the International Conference on Learning Representations, 2026.

T. Cho, S. Han, H. Lee, K. Lee, and J. Lee. Pitfall of optimism: Distributional reinforcement learning by randomizing risk criterion. In Advances in Neural Information Processing Systems, 2023.

Y. Chow and M. Ghavamzadeh. Algorithms for CVaR optimization in MDPs. In Advances in Neural Information Processing Systems, 2014.

Y. Chow, G. Tennenholtz, I. Gur, V. Zhuang, B. Dai, S. Thiagarajan, C. Boutilier, R. Agarwal, A. Kumar, and A. Faust. Inference-aware fine-tuning for best-of-� sampling in large language models. In Proceedings of the International Conference on Learning Representations, 2024.

J. J. Y. Chung, V. Padmakumar, M. Roemmele, Y. Sun, and M. Kreminski. Modifying large language model post-training for diverse creative writing. In Proceedings of the Conference on Language Modeling, 2025.

A. Coache and S. Jaimungal. Robust reinforcement learning with dynamic distortion risk measures. SIAM Journal on Mathematics of Data Science, 8(1):1–22, 2026.

T. Coste, U. Anwar, R. Kirk, and D. Krueger. Reward model ensembles help mitigate overoptimization. In Proceedings of the International Conference on Learning Representations, 2024.

G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, Z. Liu, H. Peng, L. Bai, W. Ouyang, Y. Cheng, B. Zhou, and N. Ding. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

W. Dabney, G. Ostrovski, D. Silver, and R. Munos. Implicit quantile networks for distributional reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2018.

M. Dimakopoulou, N. Vlassis, and T. Jebara. Marginal posterior sampling for slate bandits. In Proceedings of the International Joint Conference on Artificial Intelligence, 2019.

M. Ehrgott. Multicriteria optimization. Springer, 2005.

M. M. Fard and J. Pineau. Non-deterministic policies in Markovian decision processes. Journal of Artificial Intelligence Research, 40:1–24, 2011.

Z. Gábor, Z. Kalmár, and C. Szepesvári. Multi-criteria reinforcement learning. In Proceedings of the International Conference on Machine Learning, 1998.

S. Ganesh and V. Aggarwal. Breaking the bias barrier in concave multi-objective reinforcement learning. arXiv preprint arXiv:2603.08518, 2026.

L. Gao, J. Schulman, and J. Hilton. Scaling laws for reward model overoptimization. arXiv preprint arXiv:2210.10760, 2022.

X. Gu, S. De, L. Markeeva, P. Veličković, and R. Pascanu. Understanding performance gap between parallel and sequential sampling in large reasoning models. arXiv preprint arXiv:2604.05868, 2026.

L. Gui, C. Gârbacea, and V. Veitch. Bonbon alignment for large language models and the sweetness of best-of-� sampling. In Advances in Neural Information Processing Systems, 2024.

A. GX-Chen, J. Prakash, J. Guo, R. Fergus, and R. Ranganath. KL-regularized reinforcement learning is designed to mode collapse. In Proceedings ofthe International Conference on Learning Representations, 2026.

T. Haarnoja, H. Tang, P. Abbeel, and S. Levine. Reinforcement learning with deep energy-based policies. In Proceedings of the International Conference on Machine Learning, 2017.

J. I. Hamid, I. H. Orney, E. Xu, C. Finn, and D. Sadigh. Polychromic objectives for reinforcement learning. In Proceedings of the International Conference on Learning Representations, 2026.

C. F. Hayes, R. Rădulescu, E. Bargiacchi, J. Källström, M. Macfarlane, M. Reymond, T. Verstraeten, L. M. Zintgraf, R. Dazeley, F. Heintz, E. Howley, A. A. Irissappane, P. Mannion, A. Nowé, G. Ramos, M. Restelli, P. Vamplew, and D. M. Roijers. A practical guide to multi-objective reinforcement learning and planning. Autonomous Agents and Multi-Agent Systems, 36(1):26, 2022.

A. Huang, A. Block, D. J. Foster, D. Rohatgi, C. Zhang, M. Simchowitz, J. T. Ash, and A. Krishnamurthy. Self-improvement in language models: The sharpening mechanism. In Proceedings of the International Conference on Learning Representations, 2025.

T. Hubert, R. Mehta, L. Sartran, M. Z. Horváth, G. Žužić, E. Wieser, A. Huang, J. Schrittwieser, Y. Schroecker, H. Masoom, O. Bertolli, T. Zahavy, A. Mandhane, J. Yung, I. Beloshapka, B. Ibarz, V. Veeriah, L. Yu, O. Nash, P. Lezeau, S. Mercuri, C. Sönne, B. Mehta, A. Davies, D. Zheng, F. Pedregosa, Y. Li, I. von Glehn, M. Rowland, S. Albanie, A. Velingker, S. Schmitt, E. Lockhart, E. Hughes, H. Michalewski, N. Sonnerat, D. Hassabis, P. Kohli, and D. Silver. Olympiad-level formal mathematical reasoning with reinforcement learning. Nature, 651:607–613, 2026.

M. Ismayilzada, A. Laverghetta Jr, S. A. Luchini, R. Patel, A. Bosselut, L. van der Plas, and R. Beaty. Creative preference optimization. In Findings of the Association for Computational Linguistics, 2025.

Y. Jhaveri, H. Wiltzer, P. Shafto, M. G. Bellemare, and D. Meger. Convergence theorems for entropyregularized and distributional reinforcement learning. In Advances in Neural Information Processing Systems, 2025.

S. Kale, L. Reyzin, and R. E. Schapire. Non-stochastic bandit slate problems. In Advances in Neural Information Processing Systems, 2010.

R. Kirk, I. Mediratta, C. Nalmpantis, J. Luketina, E. Hambro, E. Grefenstette, and R. Raileanu. Understanding the efects of RLHF on LLM generalisation and diversity. In Proceedings of the International Conference on Learning Representations, 2024.

N. Lambert. Over-optimization. In Reinforcement Learning from Human Feedback. Online, 2026.

J. Langford and T. Zhang. The epoch-greedy algorithm for multi-armed bandits with side information. In Advances in Neural Information Processing Systems, 2007.

H. Levy. Stochastic dominance: Investment decision making under uncertainty. Springer, third edition, 2016.

T. Li, Y. Zhang, P. Yu, S. Saha, D. Khashabi, J. Weston, J. Lanchantin, and T. Wang. Jointly reinforcing diversity and quality in language model generations. arXiv preprint arXiv:2509.02534, 2025.

X. Lin, X. Zhang, Z. Yang, F. Liu, Z. Wang, and Q. Zhang. Smooth Tchebychef scalarization for multi-objective optimization. In Proceedings of the International Conference on Machine Learning, 2024.

S. Mannor and N. Shimkin. The steering approach for multi-criteria reinforcement learning. In Advances in Neural Information Processing Systems, 2001.

K. Miettinen. Nonlinear multiobjective optimization, volume 12. Springer Science & Business Media, 1999.

V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2016.

T. Moskovitz, A. K. Singh, D. Strouse, T. Sandholm, R. Salakhutdinov, A. D. Dragan, and S. McAleer. Confronting reward model overoptimization with constrained RLHF. In Proceedings of the International Conference on Learning Representations, 2024.

I. H. Orney, J. I. Hamid, S. S. Ramanujam, S. Wu, H. Hu, N. Goodman, D. Sadigh, and C. Finn. Poly-EPO: Training exploratory reasoning models. arXiv preprint arXiv:2604.17654, 2026.

V. Padmakumar and H. He. Does writing with language models reduce content diversity? In Proceedings of the International Conference on Learning Representations, 2023.

M. L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley, 1994.

F. Radlinski, R. Kleinberg, and T. Joachims. Learning diverse rankings with multi-armed bandits. In Proceedings of the International Conference on Machine Learning, 2008.

R. Rafailov, Y. Chittepu, R. Park, H. S. Sikchi, J. Hejna, B. Knox, C. Finn, and S. Niekum. Scaling laws for reward model overoptimization in direct alignment algorithms. In Advances in Neural Information Processing Systems, 2024.

D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley. A survey of multi-objective sequential decision-making. Journal ofArtificial Intelligence Research, 48:67–113, 2013.

B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. R. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, P. Kohli, and A. Fawzi. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024.

P. G. Sessa, R. Dadashi, L. Hussenot, J. Ferret, N. Vieillard, A. Ramé, B. Shariari, S. Perrin, A. Friesen, G. Cideron, S. Girgin, P. Stanczyk, A. Michi, D. Sinopalnikov, S. Ramos, A. Héliou, A. Severyn, M. Hofman, N. Momchev, and O. Bachem. BOND: Aligning LLMs with best-of-� distillation. In Proceedings of the International Conference on Learning Representations, 2025.

A. Shypula, S. Li, B. Zhang, V. Padmakumar, K. Yin, and O. Bastani. Evaluating the diversity and quality of LLM generated content. In Proceedings of the Conference on Language Modeling, 2025.

R. E. Steuer and E.-U. Choo. An interactive weighted Tchebychef procedure for multiple objective programming. Mathematical programming, 26(3):326–344, 1983.

N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, 2020.

P. Sunehag, R. Evans, G. Dulac-Arnold, Y. Zwols, D. Visentin, and B. Coppin. Deep reinforcement learning with attention for slate Markov decision processes with high-dimensional states and actions. arXiv preprint arXiv:1512.01124, 2015.

R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018.

Y. Tang, K. Zheng, G. Synnaeve, and R. Munos. Optimizing language models for inference time objectives using reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2025.

E. Todorov. Linearly-solvable Markov decision problems. In Advances in Neural Information Processing Systems, 2006.

A. Tversky and D. Kahneman. Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and Uncertainty, 5(4):297–323, 1992.

P. Vamplew, R. Dazeley, A. Berry, R. Issabekov, and E. Dekker. Empirical evaluation methods for multiobjective reinforcement learning algorithms. Machine learning, 84(1):51–80, 2011.

C. M. Verdun, A. Oesterling, H. Lakkaraju, and F. P. Calmon. Soft best-of-� sampling for model alignment. In IEEE International Symposium on Information Theory (ISIT). IEEE, 2025.

C. Walder and D. T. Karkhanis. Pass@� policy optimization: Solving harder reinforcement learning problems. In Advances in Neural Information Processing Systems, 2026.

S. Wang. Premium calculation by transforming the layer premium density. ASTIN Bulletin: The Journal of the IAA, 26(1):71–92, 1996.

P. West and C. Potts. Base models beat aligned models at randomness and creativity. In Proceedings of the Conference on Language Modeling, 2025.

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.

R. J. Williams and J. Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268, 1991.

M. E. Yaari. The dual theory of choice under risk. Econometrica: Journal of the Econometric Society, pages 95–115, 1987.

C. Yang and A. Holtzman. LLM probability concentration: How alignment shrinks the generative horizon. arXiv preprint arXiv:2506.17871, 2025.

Y. Yue and C. Guestrin. Linear submodular bandits and their application to diversified retrieval. In Advances in Neural Information Processing Systems, 2011.

L. Yun, C. An, Z. Wang, L. Peng, and J. Shang. The price of format: Diversity collapse in LLMs. arXiv preprint arXiv:2505.18949, 2025.

Y. Zhang, H. Diddee, S. Holm, H. Liu, X. Liu, V. Samuel, B. Wang, and D. Ippolito. NoveltyBench: Evaluating language models for humanlike diversity. In Proceedings of the Conference on Language Modeling, 2025.

R. Zhao, A. Meterez, S. Kakade, C. Pehlevan, S. Jelassi, and E. Malach. Echo chamber: RL post-training amplifies behaviors learned in pretraining. In Proceedings of the Conference on Language Modeling, 2025.

B. D. Ziebart, A. L. Maas, J. A. Bagnell, and A. K. Dey. Maximum entropy inverse reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2008.

## APPENDICES

## A. Proofs

In this section, we collect statements and proofs of results from the main paper.

## A.1. ROSA policy gradient

Proposition A.1. Let $\mathbf { Y } = ( Y _ { i } ) _ { i = 1 } ^ { n }$ denote a multiset of actions induced by $Y _ { 1 } , \dots , Y _ { n }$ . We write $\mathbf { Y } \sim \pi _ { \theta }$ to mean the � actions are sampled i.i.d. from policy $\pi _ { \theta } .$ Denote by � a distribution over rewardfunctions $R : X { \times } y  \mathbb { R }$ , and let $R \sim \rho$ be a drawfrom this distribution. For brevity, we write $R ( X , \mathbf { Y } ) = ( R ( X , Y _ { i } ) ) _ { i = 1 } ^ { n }$ as the reward multiset. Let � be a function defined over multisets of actions. The generic ROSA objective is

$$
\mathcal {J} _ {\mathrm{ROSA}} (\boldsymbol {\pi} _ {\theta}) = \mathbb {E} _ {X \sim \mu , \mathbf {Y} _ {\sim} ^ {i. i. d} \boldsymbol {\pi} _ {\theta} (\cdot | X)} \left[ \mathbb {E} _ {R \sim \rho} \Big [ f (R (X, \mathbf {Y})) \Big ] \right],\tag{7}
$$

and the gradient of the objective at a sampled state $X \sim \mu$ is

$$
\mathbb {E} _ {\mathbf {Y} \stackrel {i. i. d.} {\sim} \pi_ {\theta} (\cdot | X)} \left[ \mathbb {E} _ {R \sim \rho} \Big [ \sum_ {i = 1} ^ {n} \Big (f \big (R (X, \mathbf {Y}) \big) - b \big (R (X, \mathbf {Y} _ {- i}) \big) \Big) \nabla_ {\theta} \log \pi_ {\theta} (Y _ {i} | X) \Big ] \right],\tag{8}
$$

where � is an arbitrary multiset function that forms an optional control variate, and $\mathbf { Y } _ { - i }$ indicates the multiset $\mathbf { Y } = ( Y _ { j } ) _ { j = 1 } ^ { n }$ with element $Y _ { i }$ removed.

Proof. The proof is a straightforward extension of analysis by Tang et al. (2025) in the single-reward setting, to deal with randomization over reward functions. The core technique (in the single-reward setting) is to treat the vector $( Y _ { 1 } , \ldots , Y _ { n } )$ as a single “macro action”, with reward $f ( R ( X , \mathbf { Y } ) )$ . Generally, for any multiset function $^ { g , }$ we have per the usual score function trick (Williams, 1992),

$$
\nabla_ {\theta} \mathbb {E} _ {\mathbf {Y} \stackrel {\text {i.i.d.}} {\sim} \pi_ {\theta} (\cdot | X)} \left[ g (X, \mathbf {Y}) \right] = \mathbb {E} _ {\mathbf {Y} \stackrel {\text {i.i.d.}} {\sim} \pi_ {\theta} (\cdot | X)} \left[ g (X, \mathbf {Y})   \nabla_ {\theta} \log \pi_ {\theta} (\mathbf {Y} | X) \right],
$$

as an expression for the gradient. The statement then follows by noting that since the actions comprising Y are independent, log $\textstyle \pi _ { \theta } ( \mathbf { Y } | X ) = \sum _ { i = 1 } ^ { n }$ log $\pi _ { \theta } ( Y _ { i } | X )$ , which gives us,

$$
\mathbb {E} _ {\mathbf {Y} \stackrel {\text {i.i.d.}} {\sim} \pi_ {\theta} (\cdot | X)} \left[ g (X, \mathbf {Y}) \sum_ {i = 1} ^ {n} \nabla_ {\theta} \log \pi_ {\theta} (Y _ {i} | X) \right].
$$

Extending this to $g ( X , \mathbf { Y } ) = \mathbb { E } _ { R \sim \rho } [ f ( R ( X , \mathbf { Y } ) ) ]$ yields,

$$
\nabla_ {\theta} \mathcal {J} _ {\mathrm{ROSA}} (\pi_ {\theta}) = \mathbb {E} _ {\mathbf {Y} _ {\sim} ^ {\text {i.i.d.}} \pi_ {\theta} (\cdot | X)} \left[ \mathbb {E} _ {R \sim \rho} \Big [ f \big (R (X, \mathbf {Y}) \big) \nabla_ {\theta} \log \pi_ {\theta} (\mathbf {Y} | X) \Big ] \right].
$$

Finally, by independence, any random variable independent of $Y _ { i }$ can stand as a control variate for $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( Y _ { i } | \boldsymbol { X } )$ per the score function property $\mathbb { E } _ { Y _ { i } \sim \pi _ { \theta } } [ \nabla _ { \theta }$ log $\pi _ { \theta } ( Y _ { i } | X ) ] = 0$ . It follows that,

$$
\mathbb {E} _ {\mathbf {Y} \stackrel {\mathrm{i.i.d.}} {\sim} \pi_ {\theta} (\cdot | X)} \left[ b (X, \mathbf {Y} _ {- i}) \nabla_ {\theta} \log \pi_ {\theta} (Y _ {i} | X) \right] = 0.
$$

Subtracting this expectation-zero quantity from the per-sample score gives us the gradient as expressed in the statement of the result. □

Note � can be any multiset function, including a function which only make use of a subset of $\mathbf { Y } _ { - i }$ . We opt to use $b = f$ in the present work for simplicity.

## A.2. Uniform rewards, ROSA+Max optimal policy

Proposition 3.2. Optimal policy of ROSA+Max with uniform reward function distributions. Consider � binary reward functions $\left( r _ { i } \right) _ { i = 1 } ^ { m } ,$ , each with a single distinct optimal action $y _ { i } ^ { * } \in \mathcal { Y } ,$ , and set $\rho$ uniform over $( r _ { i } ) _ { i = 1 } ^ { m }$ . Then, writing $\delta _ { y }$ for the Dirac delta distribution at $y ,$ the ROSA+Max objective with any action set size $n \geq 2$ has unique optimal policy

$$
\pi^ {*} = \sum_ {i = 1} ^ {m} \frac {1}{m} \delta_ {y _ {i} ^ {*}}.
$$

Proof. The objective value for a policy putting probability $p _ { i }$ on $y _ { i } ^ { * }$ is

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \left(1 - (1 - p _ {i}) ^ {n}\right).
$$

The constraint is that we have $\textstyle \sum _ { i = 1 } ^ { m } p _ { i } \in [ 0 , 1 ]$ . We will not directly impose constraints of non-negativity on the $( p _ { i } ) _ { i = 1 } ^ { m } ,$ and will instead verify that the solution we obtain below automatically satisfies these conditions. Clearly, it is optimal to take $\textstyle \sum _ { i = 1 } ^ { m } p _ { i } = 1$ . We can then consider the Lagrangian associated with this optimization problem to understand which policies are optimal. The Lagrangian, with Lagrange multiplier �, is

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \left(1 - (1 - p _ {i}) ^ {n}\right) - \lambda \left(\sum_ {i = 1} ^ {m} p _ {i} - 1\right).
$$

The derivative of the Lagrangian with respect to $p _ { i }$ is

$$
- \frac {n}{m} (1 - p _ {i}) ^ {n - 1} - \lambda .
$$

Equating these to 0 for all $p _ { i }$ means that the $p _ { i }$ must be equal, and hence the optimizer to this linearly-constrained concave maximization problem is when the $p _ { i }$ are all equal, as required.

Lemma 3.3. Under the assumptions of Proposition 3.2, the Hessian of the objective at the optimal policy is diagonal, with all diagonal elements given by $- n ( n - 1 ) ( 1 - { \frac { 1 } { m } } ) ^ { n - { \dot { 2 } } }$

Proof. As in the proof of Proposition 3.2, note that the objective value for a policy putting probability $p _ { i }$ on \_ $y _ { i } ^ { * }$ is

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \left(1 - (1 - p _ {i}) ^ {n}\right).
$$

Also by Proposition 3.2, the optimal policy has $p _ { i } = 1 / m$ for $i = 1 , \ldots , m$ . We can then immediately calculate the entries of the Hessian as required. □

## A.3. Non-uniform rewards, ROSA+Max optimal policy

Proposition 4.2. Consider � distinct, binary rewardfunctions $( r _ { i } ) _ { i = 1 } ^ { m }$ (each with a single distinct optimal action $y _ { i } ^ { * } \in \mathcal { Y } )$ with probabilities $( \alpha _ { i } ) _ { i = 1 } ^ { m }$ under $\rho ,$ and assume without loss ofgenerality that $\alpha _ { 1 } \geq \cdots \geq \alpha _ { m } .$ The ROSA+Max objective with $n \geq 2$ action samples has an optimal policy which samples optimal actions from the top-k reward functions (ordered by �’s), each with probability,

$$
p _ {i} ^ {*} = 1 - \frac {(k - 1) \alpha_ {i} ^ {- 1 / (n - 1)}}{\sum_ {j = 1} ^ {k} \alpha_ {j} ^ {- 1 / (n - 1)}},\tag{6}
$$

where � is the largest integer between 1 and � where $p _ { k } ^ { * }$ is positive.

Proof. Following a similar procedure as Proposition 3.2, we have the objective $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \alpha _ { i } \left( 1 - ( 1 - p _ { i } ) ^ { n } \right) } \end{array}$ where $p _ { i }$ is the probability the policy samples the correct action under the �-th reward function. The Lagrangian, including the nonnegative constraint, is,

$$
\sum_ {i = 1} ^ {m} \alpha_ {i} \left(1 - (1 - p _ {i}) ^ {n}\right) - \lambda \Big (\sum_ {i = 1} ^ {m} p _ {i} - 1 \Big) + \sum_ {i = 1} ^ {m} \mu_ {i} p _ {i}.
$$

The partial derivatives with respect to each $p _ { i }$ have the form $\alpha _ { i } n \big ( 1 - p _ { i } \big ) ^ { n - 1 } - \lambda + \mu _ { i }$ . Setting the derivatives to zero, we get the generic relationship describing the constrained optimum,

$$
n \alpha_ {i} (1 - p _ {i} ^ {*}) ^ {n - 1} = \lambda - \mu_ {i} ^ {*},\tag{9}
$$

with conditions $p _ { i } ^ { * } \geq 0 , \mu _ { i } ^ { * } \geq 0 , \mu _ { i } ^ { * } p _ { i } ^ { * } = 0 .$ , and $\begin{array} { r } { \sum _ { i } p _ { i } ^ { * } = 1 } \end{array}$

If $p _ { i } ^ { * } > 0$ , then $\mu _ { i } ^ { * } = 0$ (complementary slackness), and the solution is (denoting as $A _ { i }$ for brevity),

$$
p _ {i} ^ {*} = A _ {i} := 1 - \left(\frac {\lambda}{n \alpha_ {i}}\right) ^ {\frac {1}{n - 1}}.\tag{10}
$$

Note $A _ { i } > 0$ also implies $p _ { i } ^ { * } > 0$ (and therefore $p _ { i } ^ { * } = A _ { i } )$ . To see this, note $A _ { i } > 0$ implies $n \alpha _ { i } > \lambda$ . If $p _ { i } ^ { * } = 0 .$ , then $n \alpha _ { i } = \lambda - \mu _ { i } ^ { * } \leq \lambda$ (per Equation (9) and $\mu _ { i } ^ { * } \geq 0 )$ which is a contradiction. Since $p _ { i } ^ { * } > 0$ if and only if $A _ { i } > 0$ , this also means $p _ { i } ^ { * } = 0$ when $A _ { i } \leq 0$ . Taken together, we have,

$$
p _ {i} ^ {*} = \max \left(0, 1 - \left(\frac {\lambda}{n \alpha_ {i}}\right) ^ {\frac {1}{n - 1}}\right),
$$

with a unique value of � such that $\begin{array} { r } { \sum _ { i } p _ { i } ^ { * } = 1 } \end{array}$ . Since this function is strictly increasing in $\alpha _ { i }$ , we know that the top-� largest �’s will have non-zero $\boldsymbol { p } _ { i } ^ { * }$ , for some value of � (including $k = m )$ ). Consider only this active set $\{ p _ { 1 } ^ { * } , \ldots , p _ { k } ^ { * } \} , p _ { i } ^ { * } > 0$ . Equation (10) and $\begin{array} { r } { \sum _ { i } p _ { i } ^ { * } = 1 } \end{array}$ gives,

$$
\lambda = n \left(\frac {k - 1}{\sum_ {i = 1} ^ {k} \alpha_ {i} ^ {- 1 / (n - 1)}}\right) ^ {n - 1}.
$$

Substituting this back into Equation (10) gives us the probabilities for the active set,

$$
p _ {i} ^ {*} = 1 - \frac {(k - 1) \alpha_ {i} ^ {- 1 / (n - 1)}}{\sum_ {j = 1} ^ {k} \alpha_ {j} ^ {- 1 / (n - 1)}},
$$

as required.

## A.4. Uniform rewards, ROSA+General set function optimal policy

Before proving Theorem 4.1, we state and prove an auxiliary result that will be helpful in our eventual argument. We remind the reader we consider set functions $f ( R ( X , Y _ { 1 } ) , \dots , R ( X , Y _ { n } ) )$ , and their corresponding success-count functions $\scriptstyle { \tilde { f } } ( \sum _ { i = 1 } ^ { n } R ( X , Y _ { i } ) )$ .

Lemma A.2. Assume we have � rewardfunctions $( r _ { i } ) _ { i = 1 } ^ { m } ,$ with each $r _ { i }$ one-hot for a distinct action $y _ { i } ^ { * }$ and probability $\rho ( r _ { i } )$ under the distribution $\rho .$ . The overall objective can be written $^ { a s , }$

$$
\mathbb {E} _ {\mathbf {Y} \sim \pi} \bigg [ \mathbb {E} _ {R \sim \rho} \big [ f (R (\mathbf {Y})) \big ] \bigg ] = \mathbb {E} _ {\mathbf {Y} \sim \pi} \bigg [ \sum_ {i = 1} ^ {m} \rho (r _ {i}) \tilde {f} (S _ {i}) \bigg ] \bigg ],\tag{11}
$$

where $\begin{array} { r } { S _ { i } = \sum _ { j = 1 } ^ { n } r _ { i } ( Y _ { j } ) } \end{array}$ . Further, define $p _ { i } = \mathbb { E } _ { Y \sim \pi } [ r _ { i } ( Y ) ]$ to be the probability of sampling an action obtaining a reward of 1 under $r _ { i \cdot }$ . The partial derivative of the overall objective w.r.t. $p _ { i }$ is,

$$
\frac {\partial}{\partial p _ {i}} \mathbb {E} _ {\mathbf {Y} \sim \pi} \bigg [ \sum_ {i = 1} ^ {m} \rho (r _ {i}) \tilde {f} (S _ {i}) \bigg ] = \rho (r _ {i}) n \mathbb {E} _ {S _ {i} ^ {\prime}} \bigg [ \tilde {f} (S _ {i} ^ {\prime} + 1) - \tilde {f} (S _ {i} ^ {\prime}) \bigg ],\tag{12}
$$

where $S _ { i } ^ { \prime } \sim$ Binom $( n - 1 , p _ { i } )$ .

Proof of Lemma A.2. The first statement follows by writing the expectation over reward functions out explicitly, and using the definition of the success-count function $\tilde { f }$ from $f$ in the binary reward setting introduced in the main paper. Next, since $r _ { i }$ is binary, $r _ { i } ( Y _ { j } )$ is a Bernoulli random variable, $r _ { i } ( Y ) \sim \mathrm { B e r n o u l l i } ( p _ { i } )$ . It follows that $S _ { i } \sim$ Binom $( n , p _ { i } )$ , so that

$$
\mathbb {P} (S _ {i} = k) = \binom {n} {k} p _ {i} ^ {k} (1 - p _ {i}) ^ {n - k}.\tag{13}
$$

We can write the derivative of the binomial; with some algebra we get,

$$
\begin{array}{r l} & {\frac {\partial}{\partial p _ {i}} \mathbb {P} (S _ {i} = k) = \binom {n} {k} \bigg [ p _ {i} ^ {k - 1} (1 - p _ {i}) ^ {n - k - 1} \Big [ k (1 - p _ {i}) - (n - k) p _ {i} \Big ] \bigg ],} \\ & {\qquad = k \binom {n} {k} p _ {i} ^ {k - 1} (1 - p _ {i}) ^ {n - k} - (n - k) \binom {n} {k} p _ {i} ^ {k} (1 - p _ {i}) ^ {(n - 1) - k},} \\ & {\qquad = n \bigg [ \mathbb {P} (S _ {i} ^ {\prime} = k - 1) - \mathbb {P} (S _ {i} ^ {\prime} = k) \bigg ],} \end{array}
$$

where $S _ { i } ^ { \prime } \sim$ Binom $( n - 1 , p _ { i } )$ . Further, we can write,

$$
\begin{array}{r l} & {\frac {\partial}{\partial p _ {i}} \mathbb {E} _ {S _ {i}} \Big [ f (S _ {i}, n) \Big ] = \frac {\partial}{\partial p _ {i}} \sum_ {k = 0} ^ {n} \mathbb {P} (S _ {i} = k) \tilde {f} (k),} \\ & {\qquad = \sum_ {k = 0} ^ {n} \tilde {f} (k) \frac {\partial}{\partial p _ {i}} \mathbb {P} (S _ {i} = k),} \\ & {\qquad = \sum_ {k = 0} ^ {n} \tilde {f} (k) n \big [ \mathbb {P} (S _ {i} ^ {\prime} = k - 1) - \mathbb {P} (S _ {i} ^ {\prime} = k) \big ],} \\ & {\qquad = n \Big [ \sum_ {k = 0} ^ {n} \tilde {f} (k) \mathbb {P} (S _ {i} ^ {\prime} = k - 1) - \sum_ {k = 0} ^ {n} \tilde {f} (k) \mathbb {P} (S _ {i} ^ {\prime} = k) \Big ],} \\ & {\qquad = n \Big [ \sum_ {h = 0} ^ {n - 1} \tilde {f} (h + 1) \mathbb {P} (S _ {i} ^ {\prime} = h) - \sum_ {k = 0} ^ {n - 1} \tilde {f} (k) \mathbb {P} (S _ {i} ^ {\prime} = k) \Big ],} \\ & {\qquad = n \sum_ {k = 0} ^ {n - 1} \mathbb {P} (S _ {i} ^ {\prime} = k) \Big [ \tilde {f} (k + 1) - \tilde {f} (k) \Big ],} \\ & {\qquad = n \mathbb {E} _ {S _ {i} ^ {\prime}} \Big [ \tilde {f} (S _ {i} ^ {\prime} + 1) - \tilde {f} (S _ {i} ^ {\prime}) \Big ].} \end{array}
$$

Putting things together, we have,

$$
\frac {\partial}{\partial p _ {i}} \mathbb {E} _ {\mathbf {Y} \sim \pi} \bigg [ \sum_ {j = 1} ^ {m} \rho (r _ {j}) \tilde {f} (S _ {j}) \bigg ] = \rho (r _ {i}) \cdot n \mathbb {E} _ {S _ {i} ^ {\prime}} \Big [ \tilde {f} (S _ {i} ^ {\prime} + 1) - \tilde {f} (S _ {i} ^ {\prime}) \Big ],\tag{14}
$$

as required.

We are now ready to prove Theorem 4.1.

Theorem 4.1. Optimal policy for general �. Consider � uniformly distributed (binary, distinct) reward functions $\left( r _ { i } \right) _ { i = 1 } ^ { m } .$ , each with a single distinct optimal action $y _ { i } ^ { * } \in \mathcal { Y } ,$ , and a set function $f ,$ with corresponding success-count reward function $\tilde { f }$ (defined above) which is strictly increasing as well as strictly concave in the sum of rewards. Then the $R O S A + f$ objective has a global optimum which samples correct actions from all reward functions uniformly,

$$
\pi^ {*} = \frac {1}{m} \sum_ {i = 1} ^ {m} \delta_ {y _ {i} ^ {*}}.
$$

Proof. Under the assumptions of the statement, Lemma A.2 shows that the partial derivative of the objective with respect to $p _ { i }$ is proportional to

$$
\mathbb {E} _ {S _ {i} ^ {\prime}} \left[ \tilde {f} (S _ {i} ^ {\prime} + 1) - \tilde {f} (S _ {i} ^ {\prime}) \right],\tag{15}
$$

where $S _ { i } ^ { \prime } \sim$ Binom $( n - 1 , p _ { i } )$ . If $\tilde { f }$ is strictly increasing and concave, then the function $h ( k ) = \tilde { f } ( k +$ $1 ) - { \tilde { f } } ( k )$ is strictly decreasing and positive. It then follows that if $S ^ { \prime } \sim$ Binom $( n - 1 , p _ { i } )$ , we have that

$$
\mathbb {E} _ {S _ {i} ^ {\prime}} \Big [ \tilde {f} (S _ {i} ^ {\prime} + 1) - \tilde {f} (S _ {i} ^ {\prime}) \Big ] = \mathbb {E} _ {S _ {i} ^ {\prime}} \Big [ h (S _ {i} ^ {\prime}) \Big ],
$$

is strictly monotone decreasing in $p _ { i }$ . The intuition behind this latter statement is that as we increase the parameter $p _ { i }$ in the Binom $( n - 1 , p _ { i } )$ distribution, probability mass is shifted to the right, and since ℎ is strictly decreasing, the expectation itself must also be strictly monotone decreasing. To make this rigorous, we observe that when $p _ { i } < p _ { i } ^ { \prime } ,$ , we have Binom $( n - 1 , p _ { i } )$ is strictly stochastically dominated by Binom $( n - 1 , p _ { i } ^ { \prime } )$ . We combine this observation with the general result that if a distribution � is strictly stochastically dominated by another distribution $\mu ^ { \prime } { } _ { : }$ , then for any strictly decreasing function $g : \mathbb { R } \to \mathbb { R }$ , we have

$$
\mathbb {E} _ {Z \sim \mu} [ g (Z) ] > \mathbb {E} _ {Z \sim \mu^ {\prime}} [ g (Z) ]
$$

see, for example, Levy (2016) for further background. Therefore, given the component of the Lagrangian in Lemma $\mathsf { A } . 2 .$ we require all $p _ { i }$ equal in order for all derivatives of the Lagrangian to be 0, and hence the constrained optimum is attained with all $p _ { i }$ equal, as required. □

## B. Additional Results

## B.1. Objective Landscape of ROSA+Max

As discussed in relation to Proposition 3.2, the action sampling parameter � does not afect the optimal policy, meaning that for a uniform reward function distribution, any $n \geq 2$ induces a maximally diverse policy over all reward functions, thus the global optimum can be optimized with small parallel sampling budgets (e.g. just $n = 2 \mathrm { i . i . d }$ . action samples from �<sub>�</sub>). However, in practice we expect the selection of this parameter to be important to algorithmic performance.

To see this, consider the case of � uniformly distributed reward functions giving binary reward to � distinct actions.

Figure 9 | Objective landscape slice from zero reward to the optimal, maximally diverse policy, given � reward functions and � sampled actions.

The optimal solution is to put $1 / m$ mass uniformly over the � good actions. Now, consider a straight line going from all negative actions (at $t = 0 )$ to this optimum (at � = 1),

$$
1 - \left(1 - \frac {t}{m}\right) ^ {n}.\tag{16}
$$

This can be thought of as a slice of the objective landscape. We plot this in Figure 9. Observe that while all slices have maximum at � = 1, they follow diferent curvatures, with � ≫ � resulting in the objective flattening out too early which can slow optimization due to the shallow landscape.

Expected advantage of ROSA and vanilla PG. Another way to conceptualize the above trade-of is to consider the “efective advantage” term in Equation (4) (i.e. $\begin{array} { r } { \operatorname* { m a x } _ { i \in \{ 1 , \dots , n \} } R ( Y _ { i } ) \ - \operatorname* { m a x } _ { i \neq j } R ( Y _ { i } ) ) } \end{array}$ which multiplies the gradient of the log-probability. This term becomes 0 as soon as there are two optimal �’s in the sampled action set Y. This can be illustrated clearly by plotting the expected efective advantage for the vanilla PG $\left( f _ { \mathrm { m e a n } } \right)$ compared against that of the set max function $\left( f _ { \operatorname* { m a x } } \right)$ in a binary reward setting (Figure 10, left). Observe that the efective advantage of both correct and incorrect actions quickly decay and converge to 0 as the policy improves for ROSA+Max, while the gap between the two is constant for vanilla PG. Fundamentally, we do want training to slow down as we approach a better policy (as ROSA+Max induces), though not so much that it slows learning. Other set functions, such as the ROSA+Softmax, can help induce diferent optimization landscapes.


Correct (R=1) Incorrect (R=0) Average


Correct (R=1) Incorrect (R=0) Average

Figure 10 | Expected advantage of a positive (green) and negative (red) sample in a binary reward task, as function of policy performance (x axis), for set size $n = 8 .$ , and diferent set functions with leave-one-out baseline. Error bar denotes the middle 50% percentile interval of data. We see the diference in expected advantage between positive and negative samples (i) is constant for vanilla PG, and (ii) quickly decays to zero for Set Max. Moreover, Set Softmax interpolates between these two behaviours depending on the choice of inverse temperature parameter �.

## C. Additional discussion of related work

In this section, we provide more technical discussion with related subfields of RL.

## C.1. Risk-sensitive reinforcement learning

The set max function used in ROSA+Max can be interpreted as optimizing a function of the reward distribution that is not expressible as an expected utility, but rather as a distortion risk measure (Wang, 1996; Yaari, 1987). The particular distributional property is summarized below (Wang, 1996).

Lemma C.1. The expected max-of-n reward can be expressed through a distorted expectation,

$$
\mathbb {E} _ {Y _ {1}, \ldots , Y _ {n} \stackrel {i. i. d.} {\sim} \pi} \big [ \max (R (Y _ {1}), \ldots , R (Y _ {n})) \big ] = \int_ {0} ^ {1} F ^ {- 1} (u ^ {1 / n}) \mathrm{d} u,\tag{17}
$$

where � is the CDF of $R ( Y ) , Y \sim \pi .$

Proof. Recall that $R ( Y )$ is equal in distribution to $F ^ { - 1 } ( U )$ , where � ∼ Uniform([0, 1]), and $F ^ { - 1 }$ is the quantile function of $R ( Y )$ . We then have that $( R ( Y _ { 1 } ) \ . \ . . , R ( Y _ { n } ) )$ is equal in distribution to $( F ^ { - 1 } ( U _ { 1 } ) , \dots , F ^ { - 1 } ( U _ { n } ) )$ , where $( U _ { i } ) _ { i = 1 } ^ { n } \stackrel { \mathrm { i . i . d . } } { \sim }$ Uniform( [0, 1]). It follows that,

$$
\max (F ^ {- 1} (U _ {1}), \ldots , F ^ {- 1} (U _ {n})) = F ^ {- 1} (\max (U _ {1}, \ldots , U _ {n})),
$$

due to monotonicity of $F ^ { - 1 }$ . Now, max $( U _ { 1 } , \ldots , U _ { n } )$ is distributed equally to $U _ { 1 } ^ { 1 / n }$ , since

$$
\mathbb {P} (\max (U _ {1}, \ldots , U _ {n}) \leq z) = \mathbb {P} (\forall i, U _ {i} \leq z) = \mathbb {P} (U _ {1} \leq z) ^ {n} = z ^ {n} = \mathbb {P} (U _ {1} \leq z ^ {n}) = \mathbb {P} (U _ {1} ^ {1 / n} \leq z).
$$

Hence, we have

$$
\mathbb {E} _ {Y _ {1: n} \stackrel {{i. i. d.}} {{\sim}} \pi} [ \max (R (Y _ {1}), \ldots , R (Y _ {n})) ] = \mathbb {E} _ {U \sim \text {Uniform} ([ 0, 1 ])} \left[ F ^ {- 1} (U ^ {1 / n}) \right],
$$

as required.

Note that without the $1 / n$ exponent, the expression on the right-hand side of Eq. (17) would evaluate to the mean of the distribution, $\mathbb { E } _ { Y \sim \pi } \big [ R ( Y ) \big ]$ . The integral in Lemma C.1 therefore reports an asymmetric summary of the distribution by distorting the variable of integration.

(a) The Quantile transform $( F ^ { - 1 } )$ commutes with set max operation

(b) Max-of-n (right) as transforms of distorted uniform (left)

Figure 11 | Max-of-� as an estimator of a distortion risk measure. (a) The quantile transform allows for sampling of any 1D random variable � by mapping a uniform random variable through its quantile function $( \mathrm { i . e . }$ , the inverse CDF). Because this transformation commutes with order-preserving operations like the maximum, a max-of-� distribution can be identically modelled by taking the max-of-� of uniform variables before applying the quantile transform. (b) Consequently, the max-of-� distribution of � can be viewed as a quantile transform of a distorted uniform distribution. Specifically, we sample $U \sim \mathrm { U n i f } ( 0 , 1 )$ , apply the distortion $U ^ { 1 / n }$ , and pass it to the inverse CDF to obtain $\bar { F } ^ { - 1 } ( U ^ { 1 / n } )$

Figure 11 further illustrates the intuition of Lemma C.1, where the max-of-� operation can be viewed as a quantile transform of a distorted uniform distribution. Specifically, transforming a standard uniform sample $U \sim \mathrm { U n i f } ( 0 , 1 )$ into $U ^ { 1 / n }$ yields a Beta(�, 1) distribution, which is identically distributed to the maximum of � independent uniforms. More generally, evaluating expectations under a distorted probability space defines a distortion risk measure (Wang, 1996; Yaari, 1987). This framework is widely used to model risk sensitivity, such as in cumulative prospect theory (Tversky and Kahneman, 1992), and extensively in risk-aware reinforcement learning (Cho et al., 2023; Chow and Ghavamzadeh, 2014; Coache and Jaimungal, 2026; Dabney et al., 2018). All in all, the implicit use of distributional properties beyond expected utilities, such as the distortion risk measure above, is key to the ROSA framework being able to express objectives with the desired optimal policies.

## C.2. Multi-objective reinforcement learning

In the main paper, we mention comparison with multi-objective reinforcement learning (Hayes et al., 2022; Roijers et al., 2013). Problem formulations in this domain typically focus on finding good policies in settings with a finite collection of reward functions $R _ { 1 } , \ldots , R _ { d }$ . We discuss comparisons to scalarization approaches in the bandit setting below, which are the most closely related variety of MORL problems to our setting, though often in the literature these approaches are expressed more generally in the MDP setting.

## C.2.1. Scalarization approaches

The principal approaches to handling multiple reward functions—and those closely related to the ROSA framework—find optimal policies with respect to a scalar objective constructed from the individual rewards. The first approach is termed expected scalarized return (ESR), and selects a scalarization function $s : \mathbb { R } ^ { d } \to \mathbb { R }$ to obtain an objective

$$
\mathbb {E} _ {Y \sim \pi} \big [ s (R _ {1} (Y), \ldots , R _ {d} (Y)) \big ].\tag{18}
$$

The second approach is termed scalarized expected return (SER), and instead inverts the ordering of the expectation and scalarization function, leading to an objective of the form

$$
s \big (\mathbb {E} _ {Y \sim \pi} \big [ R _ {1} (Y) \big ], \ldots , \mathbb {E} _ {Y \sim \pi} \big [ R _ {d} (Y) \big ] \big).\tag{19}
$$

Both approaches have been applied to a variety of problems in reinforcement learning that are naturally expressed through multiple reward functions. For the problems we are concerned with in this paper, however, these approaches have several drawbacks which the ROSA framework circumvents.

Lack of expressiveness. ESR deals solely with expected utilities; properties of distributions that can be expressed as the expectation of a (possibly non-linear function) applied to a random variable. This is a limited class of distributional properties, and is not expressive enough to capture the notions of diversity that we consider in this paper. As a concrete example, we consider the case where $R _ { 1 } , \ldots , R _ { d }$ are one-hot over distinct actions, and would like an objective that is uniquely maximized by the policy which is uniform over this set of actions. The only possible values that the vector $( R _ { 1 } ( Y ) , \ldots , R _ { d } ( Y ) )$ can take on are the zero-vector, and one-hot vectors. By symmetry, our non-linearity � must assign equal utility to each one-hot vector. But then the objective simply reduces to the amount of probability mass allocated to the set of actions that are optimal for one of the $( R _ { i } ) _ { i = 1 } ^ { d } ,$ , with no incentive for uniformity of the distribution on this set. On the other hand, SER captures quantities that are functions of expected rewards, and thus cannot capture distributional properties of the random variable $R _ { i } ( Y )$ that require more than the mean, such as the ones discussed in Section C.1.

Implementation complexity. SER additionally encounters complexities in implementation, owing to the objective applying non-linearities to expectations of rewards; the naive approach to forming a policy gradient for this objective induces bias, requiring more intricate algorithm design, such as two-timescale approaches (Agarwal et al., 2022; Bai et al., 2022; Ganesh and Aggarwal, 2026). By contrast, since the ROSA framework expresses the objective in terms of expectations over set actions, the non-linear function of the reward distribution is made implicit, and unbiased policy gradients for the objective are straightforward to compute.

Accessing all rewards functions. The MORL approaches described above typically assume access to all reward functions when computing an objective estimator, and thus scale computationally with $d ,$ the number of reward functions specified in the problem. In particular, they do not generally handle infinitely-many reward functions. By contrast, as the ROSA objective is phrased with an oute expectation over the reward function, unbiased estimates of objective and gradient can be calculated straightforwardly with samples from the reward function distribution.

Standard policy gradient on marginalized reward function as a special case. Given a distribution $\rho$ over a finite number of reward functions $( r _ { i } ) _ { i = 1 } ^ { m }$ , both the ESR and ROSA objectives have standard policy gradient on the reward function $\textstyle \sum _ { i = 1 } ^ { m } \rho ( r _ { i } ) r _ { i }$ as a special case. For ESR, given a distribution this is obtained by taking $s : \mathbb { R } ^ { d } \to \mathbb { R }$ to be

$$
s (z _ {1}, \ldots , z _ {n}) = \sum_ {i = 1} ^ {n} \rho (r _ {i}) z _ {i}.
$$

For ROSA, this is obtained for any � by taking the set function � to be the mean function. In the context of Figure 1 in the main paper, illustrating the general ROSA framework, this is mathematically equivalent to collapsing the plot along the reward function axis and applying a standard policy gradient. By contrast, the general ROSA framework applies a non-linear transformation, based on the policy, to each reward function before performing the reduction along this axis.

## C.2.2. Pareto optimization

While not explored in this work, we highlight a concurrent work which explores randomized objectives for Pareto optimization (Bahlous-Boldi et al., 2026). Concretely, the authors consider a sequence of vector utility functions, $\mathbf { u } = ( u _ { 1 } , \ldots , u _ { d } )$ , where $\mathbf { u } ( x , y ) \in \mathbb { R } ^ { d }$ is the �-dimensional utility feedback for inputs $( x , y )$ , and aim to find a set of actions which are Pareto-optimal for the vector utilities (Ehrgott, 2005; Miettinen, 1999). Note that in our discussion here, we have intentionally used the term utility functions, $u _ { i } : X \times { \mathcal { Y } }  \mathbb { R }$ , to distinguish from the reward functions, $R : X \times y  \mathbb { R }$ in the ROSA framework. The primary objective in Bahlous-Boldi et al. (2026) computes the score for some action multiset $( y _ { 1 } , \ldots , y _ { n } )$ as,

$$
\mathbb {E} _ {\mathbf {w} \sim \mathrm{Dir} (\mathbf {1})} \left[ \max _ {1 \leq i \leq n} \mathbf {w} ^ {\top} \mathbf {u} (x, y _ {i}) \right],\tag{20}
$$

where Dir(1) is a uniform distribution over �-dimensional vectors that sum to 1, which linearly combines the � dimensional utilities.

We note that this is in fact a special case of ROSA+Max (Eq. (3)), where the randomized reward function is defined via the following reparameterization,

$$
R (x, y) = \mathbf {w} ^ {\top} \mathbf {u} (x, y), \quad \mathbf {w} \sim \operatorname{Dir} (\mathbf {1}).\tag{21}
$$

Substituting the above distribution $R \sim \rho$ into the ROSA+Max criterion (Eq. (3)) recovers Equation (20). Thus, since the primary objective in Bahlous-Boldi et al. (2026) is an instance of ROSA, the theoretical understanding we provide in this work can be applied to understand their work as well. From this perspective, Bahlous-Boldi et al. (2026) elegantly apply distributions over scalarizations (of utilities u) to induce the reward-function distribution, and demonstrate an additional application of the ROSA framework: to do Pareto-optimization.

It is worth noting that the approach of Bahlous-Boldi et al. (2026) difers operationally from standard action-set methods. Rather than sampling the action multiset i.i.d. in parallel—as done in Hamid et al. (2026); Orney et al. (2026); Tang et al. (2025) and our work—their action multiset is generated sequentially within a single rollout. Sequential generation allows later action samples to condition on earlier ones and may, in principle, enjoy greater representation flexibility and improved multiset diversity.<sup>1</sup> Yet, it imposes a number of additional restrictions: (i) prompts have to be modified to specify multiset format and size, (ii) the model must have suficient instruction-following ability and generate following a rigidly parsable multi-action schema, and (iii) using large multisets requires long autoregressive generations. Conversely, parallel sampling requires no structural assumptions on the prompt and generation beyond compatibility with a reward function, while better exploiting modern parallelized hardware. Credit assignment also difers operationally for the two approaches: because the sequential approach optimizes a single concatenated generation containing multiple actions, its current operationalization lacks action-specific credit assignment. In contrast, paralle sampling naturally admits action-level variance-reduction techniques (Appendix A.1).

Ultimately, the generalized ROSA formulation remains orthogonal and complementary to the choice of scalarization distributions, which are seamlessly absorbed into the reward function distribution $\rho .$ Identifying the right reward function distribution for Pareto-diversity stands as a promising direction for future research. For instance, Bahlous-Boldi et al. (2026) uses linear scalarization, while nonlinear scalarization methods can have diferent coverage properties that better suit Pareto geometries (Bowman Jr, 1976; Ehrgott, 2005; Lin et al., 2024; Steuer and Choo, 1983). Promisingly, any such exploration can be viewed as ROSA with distinct choices of reward distributions, and stand to benefit directly from the theoretical understanding we have established in the current work.

## C.3. Diversity bonus approaches

While ROSA induces diversity as naturally emergent from the underlying reward distribution, a separate, distinct paradigm enforces output diversity by pairing a single reward function � with an explicit, external diversity mechanism. Intuitively, these approaches manually alter the optimization objective to “boost” distinct actions and penalize redundant ones. In this section, we describe their properties and draw contrasts with the ROSA framework.

Per-sample multiplicative diversity. A common approach is to scale the reward of each individual action by its average pairwise distance to all other concurrent actions in the multiset (Li et al., 2025). The modified per-sample reward ¯�(�<sub>�</sub>) is (again dropping state dependence to lighten notation),

$$
\bar {r} (Y _ {i}) = r (Y _ {i}) \cdot \left(\frac {1}{n - 1} \sum_ {1 \leq j \leq n, j \neq i} d (Y _ {i}, Y _ {j})\right),\tag{22}
$$

where $d : y \times y  \mathbb { R }$ is a pairwise diversity function quantifying the distance between a pair of actions. For instance, � can be a binary classifier encoding whether � and $y ^ { \prime }$ are semantically equivalent (Li et al., 2025; Zhang et al., 2025).

While intuitive, this approach sufers from two primary pitfalls. First, optimizing Equation (22) via standard policy gradient (Eq. (2)) introduces gradient bias. This is because an individual action’s modified reward dynamically depends on all other sampled actions, and standard PG ignores crossderivatives tracking how changing one action’s probability afects the diversity scores of concurrent actions. Second, even if the set policy gradient is correctly computed, per-sample multiplicative diversity warps the optimization landscape such that a worse policy (in terms of reward) can be ranked much higher than a nearly optimal policy with low diversity, therefore complicating policy improvement. This can be observed in Figure 12c: a policy which assign 50-50 probability to actions A and $\textsf { C } ( \mathbb { E } [ R ( Y ) ] \approx 0 . 5 )$ is preferred over one that puts nearly all probabilities on A $( \mathbb { E } [ R ( Y ) ] \approx 1 )$

Set-level multiplicative diversity. An alternative approach applies a global diversity modifier directly to the collective action set reward (Hamid et al., 2026; Orney et al., 2026). This alters the

(a) Reward only

(b) Diversity only

(c) Mult. diversity
Figure 12 | Per-sample multiplicative diversity. We numerically simulate the average policy score using (a) reward function only, (b) pairwise diversity function only with diversity function $d ( y , y ^ { \prime } ) = \mathbf { 1 } _ { [ y \neq y ^ { \prime } ] } ,$ and (c) per-sample multiplicative between the reward and diversity. Actions A and B receive a reward of 1, and C receives 0. Colours denote the averaged score for diferent policy on the 3-category simplex. Details in Appendix D.1.

total multiset objective to the form:

$$
\bar {R} (Y _ {1: n}) = \left(\frac {1}{n} \sum_ {i = 1} ^ {n} r (Y _ {i})\right) \cdot D (Y _ {1: n}),\tag{23}
$$

where $D ( Y _ { 1 : n } )$ is a global set-level metric. A number of diversity metrics are available here, which we again study in the three-category simplex to visualize the average score of Eq. (23) with diferent action probabilities (Figure 13). One intuitive set diversity metric is the set average distance (i.e. set version of Equation (22)),

$$
D _ {\mathrm{AvgDist}} (Y _ {1: n}) = \frac {1}{n (n - 1)} \sum_ {i = 1} ^ {n} \sum_ {1 \leq j \leq n, j \neq i} d (Y _ {i}, Y _ {j}),
$$

which we visualize in Figure 13a. Another set diversity metric is the number of distinct clusters (Hamid et al., 2026; Orney et al., 2026),

$$
D _ {\text {Clusters}} (Y _ {1: n}) = \frac {\text {number of distinct clusters in} Y _ {1 : n}}{n},
$$

which we compute as the number of unique actions, visualized in Figure 13b. Both approaches inherit the landscape-distortion pitfall of multiplicative scaling, and have optimal policies that are reward sub-optimal, as the objective prefers highly diverse sets of incorrect actions over redundant sets of correct actions. In contrast, ROSA+Max (Fig. 2d) and ROSA+Softmax (Fig. 3) naturally preserve sub-optimal policy ordering, ensuring that policy improvement gradients always point in a direction that simultaneously maximizes correctness and diversity.

Additive diversity. Finally, we also study additive bonuses in Figure 14 of the form,

$$
\bar {r} _ {\text { additive }} (Y _ {i}) = r (Y _ {i}) + \alpha \left(\frac {1}{n - 1} \sum_ {1 \leq j \leq n, j \neq i} \mathbf {1} _ {[ Y _ {i} \neq Y _ {j} ]}\right).
$$

While additive formulations can mitigate some of the policy-ordering distortions found in multiplicative methods, they introduce a separate practical pitfall: extreme sensitivity to the scale of the reward and diversity bonuses. The hyperparameter � requires extensive tuning: a value too small may fail to induce diversity, while a value too large similarly results in low-reward optimal policies.

(a) Average distance

(b) Unique actions

Figure 13 | Set multiplicative diversity. We numerically simulate the average policy score using average reward multiplied with (a) average set distance, and (b) number of distinct actions. Actions A and B receive a reward of 1, and C receives 0. Colours denote the averaged score for diferent policy on the 3-category simplex. Details in Appendix D.1.
(a) � = 0.1

(b) � = 1

(c) $\alpha = 2$
Figure 14 | Numerical simulation of additively combined reward and diversity scores. Actions A and B receive a reward of 1, while C receives 0. Colours denote the average action multiset score for diferent policy distributions on the simplex.

Interpreting diversity bonuses under the ROSA framework. Interestingly, in the case of binary rewards, the multiplicative diversity bonus (Eq. (22)) can be interpreted under the ROSA framework, albeit with a set function not supported by the theoretical guarantees of Theorem 4.1. Specifically, we will assume that actions rewarded under diferent reward functions are diferent. Suppose we have taken a multiset of � actions with � of them being rewarded under reward function $r ,$ the sum of quality × diversity scores for this set of correct actions is,

$$
k \times \frac {n - k}{n - 1}.
$$

The multiplicative term is because the other $n - k$ actions are diferent from the � rewarded actions, and therefore contribute to the diversity bonus. In other words, multiplicative diversity can be realized under the ROSA framework with a success-count reward function $\tilde { f } ( k ) ~ = ~ k ( n - k ) / ( n - 1 )$ . We plot this in Figure 15, alongside the success-count function for ROSA+Max, and ROSA+Softmax. Note that for ROSA+Softmax, the success-count function can be calculated by summing over the � contributions of a reward-1 action in Equation (5), to obtain,

$$
\frac {k e}{k e + (n - k)}.
$$

Figure 15 | Success-count functions $\tilde { f }$ for ROSA+Max, ROSA+Softmax, and the multiplicative diversity objective appearing in Equation (22), in the binary reward setting.

Interestingly, the set function corresponding to multiplicative diversity does not satisfy the criteria of Theorem 4.1; in particular, monotonicity is not satisfied. This can manifest certain issues: if only one unique correct action is observed (� times) in a collection of � actions, values of � around $n / 2$ actually receive a higher score than values of � closer to �. This forces optimization to prefer a policy that has low expected reward $( \sim k / n )$ in order to be diverse, and is an instance where the diversity-modified objective distorts the ordering of sub-optimal policies. On the other hand, we know from Theorem 4.1 that there exists a family of $\bar { \tilde { f } }$ that provides both high expected reward and a diverse optimal policy.

We conclude this section with Table 1, which summarizes, in the context of our central problem, some key properties of ROSA and a variety of comparator methods discussed in this section.

<table><tr><td>Desiderata</td><td>Entropy Regularization</td><td>Diversity Bonus</td><td>MORL (ESR)</td><td>MORL (SER)</td><td>ROSA</td></tr><tr><td>Preserves reward-maximizing optimum</td><td>✕</td><td>√/✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Ranks sub-optimal policies correctly</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Has stochastic optimal policies (diverse)</td><td>√</td><td>√</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Unbiased gradient estimation</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>√</td></tr><tr><td>Handles arbitrary reward distributions</td><td>N/A</td><td>N/A</td><td>✕</td><td>✕</td><td>√</td></tr></table>

Table 1 | Comparison of ROSA to other methods in diversity-promoting and multi-objective RL. ROSA is the only approach that satisfies all desiderata: (i) its objective optimum aligns with an expected reward optimum, (ii) it correctly orders sub-optimal policies by their rewards, (iii) it maintains a stochastic, diversity-preserving optimal policy, (iv) it admits straightforward optimization via unbiased gradient estimators from action samples $Y \sim \pi ( \cdot | X )$ , and (v) it naturally accommodates arbitrary distributions over potentially infinite reward functions. Note that depending on the type of diversity bonus approach, it may or may not have a reward-maximizing optimal policy (see Appendix C.3).

## D. Experimental details

In this section, we collect together additional experimental details and results, to aid reproduction and intuition regarding the core method of the paper and associated hyperparameter selection.

## D.1. Global landscape

To generate Figure 2, we considered categorical distributions over the three-action simplex, $\pi =$ $( \pi _ { A } , \pi _ { B } , \pi _ { C } ) \in \Delta ^ { 2 }$ . For each value of �, we drew � = 4 independent actions $Y _ { 1 } , \dots , Y _ { n } \sim { \mathrm { C a t e g o r i c a l } } ( \pi )$ and evaluated a set-level score. We estimated the expected score at each � by averaging up to 100,000 independent trials. The simplex landscape was obtained by evaluating this Monte Carlo estimate over approximately 5000 values of � tiled across the simplex.

There are three possible actions, $Y _ { i } \in \{ A , B , C \}$ , and two binary reward functions:

$$
R _ {A} (Y) = \mathbf {1} _ {[ Y = A ]}, \qquad R _ {B} (Y) = \mathbf {1} _ {[ Y = B ]}.\tag{24}
$$

Let $m \ : = \ : 2$ denote the number of reward functions and write $Y _ { 1 : n } = ( Y _ { 1 } , \ldots , Y _ { n } ) $ . By default the reward distribution is $\rho ( R _ { A } ) = \rho ( R _ { B } ) = 0 . 5$ unless otherwise specified (in Figure 4). The standard policy-gradient-style score sums rewards across sampled actions:

$$
f _ {\mathrm{PG}} (Y _ {1: n}) = \sum_ {k \in \{A, B \}} \rho (R _ {k}) \sum_ {i = 1} ^ {n} R _ {k} (Y _ {i}).\tag{25}
$$

The entropy-regularized score adds the entropy of the sampling distribution:

$$
f _ {\mathrm{PG+Ent}} (Y _ {1: n}, \pi) = \sum_ {k \in \{A, B \}} \sum_ {i = 1} ^ {n} \left[ R _ {k} (Y _ {i}) + \mathcal {H} (\pi) \right].\tag{26}
$$

The max-based set score rewards whether each reward function is achieved at least once in the sampled set:

$$
f _ {\mathrm{ROSA+Max}} (Y _ {1: n}) = \sum_ {k \in \{A, B \}} \rho (R _ {k}) \max _ {1 \leq i \leq n} R _ {k} (Y _ {i}).\tag{27}
$$

The diversity-weighted score weights each rewarded action by the fraction of other samples in the set that difer from it (Li et al., 2025):

$$
f _ {\mathrm{Div}} (Y _ {1: n}) = \sum_ {k \in \{A, B \}} \sum_ {i = 1} ^ {n} R _ {k} (Y _ {i}) \cdot \Big (\frac {1}{n - 1} \sum_ {j \neq i} \mathbf {1} _ {[ Y _ {j} \neq Y _ {i} ]} \Big).\tag{28}
$$

In Figure 16, we further illustrate optimization landscapes in a simple example with three actions (�, �, and �), and two one-hot rewards (on each of � and �). The figure shows how optimization landscapes vary depending on the objective in question (standard marginalized reward function/PG, PG with entropy regularization, ROSA+Max, ROSA+Softmax), as well as the weights of each reward function in the problem.

(a) PG






(b) PG with entropy regularization



(c) ROSA+Max




(d) ROSA+Softmax



Figure 16 | Optimization landscapes for a variety of objectives and reward function distributions.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Vocabulary size</td><td>20</td></tr><tr><td>Prompt length</td><td>3</td></tr><tr><td>Total sequence length</td><td>8</td></tr><tr><td>Train dataset size</td><td>512</td></tr><tr><td>Number of prompts per batch</td><td>256</td></tr><tr><td>Samples per prompt</td><td>16</td></tr><tr><td>Learning rate</td><td> $2 \times 10^{-4}$ </td></tr><tr><td>Training steps</td><td>2048</td></tr><tr><td>Entropy coefficient</td><td>0.01</td></tr><tr><td>Eval dataset size</td><td>512</td></tr><tr><td>Eval samples per prompt</td><td>512</td></tr></table>

Table 2 | Hyperparameters for the binary parity/anti-parity experiment.

## D.2. Small transformer experiment with competing preferences

In Figure 5, we use a small transformer in which each token’s vocabulary index corresponds to its integer representation. The reward function checks if the fixed length output sequence gives parity or anti-parity matched answers,

$$
R _ {1} (X, Y _ {i}) = \left\{ \begin{array}{l l} + 1 & \text { if   parity } (X, Y _ {i}), \\ - 1 & \text { if   anti - parity } (X, Y _ {i}), \\ 0 & \text { otherwise } \end{array} \right. \qquad R _ {2} (X, Y _ {i}) = - R _ {1} (X, Y _ {i}).\tag{29}
$$

The transformer is initialized and trained from scratch. We sample fixed-size, non-overlapping training and evaluation datasets of prompts (i.e. randomly generated integers). The training details are in Table 2.

## D.3. Multiple correct answers

Example generations for ROSA trained policies on MATH with reward function distribution over diferent generations lengths are shown in Figure 17.

## D.4. Uncertain reward functions

In Figure 7, we design an experiment with 6 reward functions defined over a continuous 2D domain $( x ~ \in ~ [ 0 , 1 0 ] , ~ y ~ \in ~ [ 0 , 4 ] )$ . Each reward function is parametrized by an optimal curve $g _ { i } ( x )$ , and $r _ { i } ( x , y ) = \exp ( - ( y - g _ { i } ( x ) ) ^ { 2 } / ( 2 \sigma ^ { 2 } ) )$ . In other words, each reward function has maximum of 1 when $y = g _ { i } ( x )$ , and decaying as a Gaussian with distance from the optimum, with $\sigma = 0 . 1 5$ (see Figure 18 for the reward landscape). The optimal $y s$ for each reward function do not always agree, creating regions where the reward function agree on a single optimum, and regions where they diverge into distinct modes. The aggregated reward is a weighted combination of the 6 functions with weights (0.3, 0.2, 0.2, 0.1, 0.1, 0.1). We represent the policy $\pi _ { \theta } ( Y | X )$ as a categorical distribution over 128 uniformly spaced bins covering the � domain (bin width $\approx 0 . 0 3 1 )$ , parametrized by a 2-layer ReLU MLP (64 hidden units per layer) implemented in JAX with Equinox. We optimize using Adam (setting the learning rate to $3 \times 1 0 ^ { - 4 } )$ for 100,000 steps with batch size 128 and group size 8 (i.e., 8 action samples � per context � per batch element). See Figure 18 for extended figures showing cross-sections of the learned policy $\pi ( \cdot | x )$ at diferent values of �.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Here's how to evaluate the expression:
1. Multiply:
•  $(1 + 2i)6 = 6 + 12i$
2. Subtract:
•  $6 + 12i - 3i = 6 + 12i - 3i = 6 + 9i$
Therefore,  $(1 + 2i)6 - 3i = \boxed{6 + 9i}$ .
(a) Short Generation (length = 196)
</div>

(b) Long Generation (length = 4245)
Figure 17 | Comparison of length-diverse responses from a ROSA trained policy. Question: Evaluate \$(1+2i)6-3i\$.


(a) PG (entropy coef = 0.1)







(b) PG (entropy coef = 0.8)

(c) ROSA+Max (entropy coef = 0.01)

Figure 18 | Policy trained on 6 non-uniformly weighted reward functions $R : X \times y $ ℝ. Left plot shows the reward functions and their maximum (white lines). Right plot shows the policy over Y as a function of �. Observe PG training leads to collapse, while PG with a high entropy coeficient converges to the dominant reward functions. On the other hand, ROSA+Max covers all reward modes in proportion to their reward function weight.

## D.5. Noisy MATH judges

In Figure 8a, we construct an ensemble of reward judges in MATH. We train with ROSA (Max and Softmax), vanilla PG, as well as ensemble-based methods suggested by Coste et al. (2024). Specifically, given � rewards $( r _ { i } ) _ { i = 1 } ^ { m }$ , ensemble min optimizes the minimum min $( r _ { i } ) _ { i = 1 } ^ { m } )$ reward, while ensemble stddev optimizes the mean minus the standard deviation $\begin{array} { r } { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } r _ { i } - s \mathrm { t d } \bar { \mathrm { d e v } } ( ( r _ { i = 1 } ^ { m } ) ) } \end{array}$

In Figure 8b, we directly take a ROSA+Max trained checkpoint from the above MATH task, and evaluate it zero shot in AIME2025.
