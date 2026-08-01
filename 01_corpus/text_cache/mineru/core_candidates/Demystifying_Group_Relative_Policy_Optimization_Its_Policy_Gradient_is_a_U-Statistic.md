# Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic

Hongyi Zhou<sup>1,∗</sup> Kai Ye<sup>2,∗</sup> Erhan Xu<sup>2</sup> Jin Zhu<sup>3</sup> Ying Yang<sup>4</sup> Shijin Gong<sup>5,†</sup> Chengchun Shi<sup>2,†</sup>

## Abstract

Group relative policy optimization (GRPO), a core methodological component of DeepSeek-Math and DeepSeek-R1, has emerged as a cornerstone for scaling reasoning capabilities of large language models. Despite its widespread adoption and the proliferation of follow-up works, the theoretical properties of GRPO remain less studied. This paper provides a unified framework to understand GRPO through the lens of classical U-statistics. We demonstrate that the GRPO policy gradient is inherently a U-statistic, allowing us to characterize its mean squared error (MSE), derive the finite-sample error bound and asymptotic distribution of the suboptimality gap for its learned policy. Our findings reveal that GRPO is asymptotically equivalent to an oracle policy gradient algorithm – one with access to a value function that quantifies the goodness of its learning policy at each training iteration – and achieves asymptotically optimal performance within a broad class of policy gradient algorithms. Furthermore, we establish a universal scaling law that ofers principled guidance for selecting the optimal group size. Empirical experiments further validate our theoretical findings, demonstrating that the optimal group size is universal, and verify the oracle property of GRPO.

## 1 Introduction

Since the birth of ChatGPT in 2022, large language models (LLMs) have been increasingly integrated into our work and everyday life. These models are evolving at an extremely fast pace and can now perform a broad range of tasks at or beyond human level, ushering in an era of rapid technological advancement across scientific, industrial, and societal domains. One of the underlying technologies that enhance the capabilities of these LLMs is reasoning. The idea is straightforward: because LLMs generate human language, they can be guided to perform multi-step reasoning in a manner similar to humans. Initially, this is often achieved through some simple magical prompts [Wei et al., 2022] – for example, “Let us think step by step” – which encourage the model to decompose complex questions into intermediate reasoning steps and produce more accurate solutions; see Figure 1 for an illustration.

Figure 1: An illustration of LLM reasoning. The example shows a prompt and the corresponding model output, consisting of a reasoning trace and a final solution.

Early progress in LLM reasoning was driven largely by inference-time techniques that avoid retraining the language model, focusing on eliciting or searching for more efective intermediate reasoning steps. In parallel, Ouyang et al. [2022] established a scalable reinforcement learning with human feedback (RLHF) pipeline for retraining – more precisely, post-training – LLMs to align their outputs with human values. Its main idea is to (i) collect multiple responses for each prompt and obtain human preference over these responses; (ii) learn a reward model from the resulting preference data [Christiano et al., 2017]; and (iii) apply reinforcement learning (RL), specifically the proximal policy optimization algorithm [PPO, Schulman et al., 2017] to fine-tune the LLM parameters so as to maximize the cumulative estimated reward. This work provided a practical demonstration that RL, when combined with human preference feedback, can reliably steer LLM behavior at scale.

The RLHF pipeline is applicable to LLM reasoning. However, it introduces two challenges. First, reasoning tasks often produce long intermediate trajectories, making human supervision for labeling responses time-consuming. This challenge can be addressed by reinforcement learning with verifiable rewards [RLVR, Lambert et al., 2025], which similarly employs RL for post-training but replaces the reward function learned from subjective human preference with objective verifiers in applications such as mathematics problems, where solutions are unique, or programming tasks, where the generated code can be executed to verify its correctness. The second challenge is more technical: PPO requires to learn a critic network that quantifies the quality of the model at each step to reduce the variance of the stochastic gradient. Nonetheless, estimating and storing such a network in reasoning tasks is computationally expensive.

Group relative policy optimization [GRPO, Shao et al., 2024] represents a major milestone for RL algorithms tailored to LLM reasoning. It provides a highly efective solution to the second challenge by eliminating the critic network entirely and instead sampling multiple outputs for each prompt, using their group average as a proxy for the critic; refer to Figure 2 for an overview of the GRPO pipeline, and Section 3 for details. This approach leads to the development of DeepSeek-R1, a prominent large reasoning model at the time, whose post-training requires only about 147K H800 GPU-hours – an order of magnitude lower than many contemporary large reasoning models, including OpenAI’s o1 series [Jaech et al., 2024].

The methodology was further formalized in a paper published in Nature [Guo et al., 2025], marking the first peer-reviewed LLM article in the journal, and was subsequently deployed by various open-source LLMs. These eforts established GRPO as a foundational RLVR algorithm for LLM reasoning, with numerous follow-up methods built upon it shortly after its introduction (see

Figure 2: Overview of the GRPO pipeline. For each prompt, GRPO samples multiple reasoning traces to generate outputs, which are then evaluated by a reward model to measure their quality. These rewards are compared against the group mean and standardized to compute the advantage function, which is used to update the policy model, subject to KL regularization with respect to a reference model.

Section 2.2 for a review). Beyond LLM reasoning, GRPO has also been applied as a general-purpose RL engine in other domains, including agent settings, where the goal extends beyond chatbots to enabling the model to use external tools or perform more complex tasks [e.g., Qian et al., 2025, Ding and Ye, 2026].

All of the aforementioned developments highlight the practical appeal of GRPO, but they also reveal a theoretical gap, as formal analyses remain limited in the literature. In particular:

Q1. Why is GRPO so efective?

Q2. What is the rationale for using the group mean to approximate the critic network?

Q3. Can we provide finite-sample or asymptotic analyses regarding its convergence?

Q4. How many outputs shall we sample per prompt?

This paper addresses these questions by demystifying GRPO from a statistical perspective. Our key observation is that GRPO is deeply connected to U-statistics [Hoefding, 1948], a connection that has not been explicitly recognized in prior works. Building on this observation, we conduct a comprehensive analysis of GRPO, covering both finite-sample and asymptotic properties, examining both the original algorithm and its variants, and analyzing both the policy gradient estimator used to update model parameters at each iteration and the suboptimality gap, which measures the diference between the learned policy and the optimal policy; see Figure 3 for a visualization of our theoretical framework.

Our theoretical contributions are as follows:

1. We establish the first connection between GRPO and U-statistics by showing that the GRPO policy gradient is inherently a U-statistic (Lemma 1). This addresses Q2 by providing a principled explanation for using the group mean to approximate the critic network through classical U-statistics theory.

Figure 3: Roadmap of our theoretical results.

2. To address Q3, we provide finite-sample analyses that characterize the mean squared error (MSE) of the GRPO policy gradient (Theorem 2 & Proposition 3) and derive error bounds on its suboptimality gap (Lemma 6). We further establish parameter consistency and derive the asymptotic distribution of the suboptimality gap without requiring parameter identifiability (Theorem 8). This result is novel in two respects: (i) existing literature primarily focuses on error bounds for the suboptimality gap, which characterize its order but are not as accurate as its distribution; and (ii) classical asymptotic analyses rely on parameter identifiability, an assumption clearly violated in overparameterized LLMs.

3. Our finite-sample and asymptotic analyses lead to two desirable properties of GRPO: (i) the oracle property (Corollaries 4 & 9), whereby GRPO is asymptotically equivalent to an oracle algorithm with access to the true critic network; and (ii) optimality (Corollaries 5 & 10), whereby GRPO asymptotically minimizes both the MSE and the suboptimality gap among a broad class of RL algorithms. These theoretical findings are further supported by empirical evidence (Figure 4), which addresses Q1 and explains GRPO’s efectiveness.

4. Finally, to address Q4, we derive a scaling law that delineates how GRPO’s performance depends on the number of sampled outputs per group, and identifies the optimal group size that maximizes its performance (Theorem 7). Notably, this optimal choice depends only on the training data and model architecture, and is independent of other factors such as the training budget or number of iterations. This universality makes our scaling law particularly appealing, as it does not require retuning when these factors change. Its universality is further validated empirically (Table 2 & Figure 5).

## 2 Related works

Our work connects two modern areas in artificial intelligence – reinforcement learning (Section 2.1), and its emerging application to LLM reasoning through RLVR (Section 2.2) – with the classical theory of U-statistics (Section 2.3).

## 2.1 Reinforcement learning

The literature on RL is vast. Existing algorithms are broadly distinguished as planning or learning, based on whether the data generating process is known [e.g., Sutton and Barto, 2018, Chapter 8].

Within learning, model-based approaches explicitly estimate the MDP model [e.g., Jiang, 2024], while model-free methods do not. The latter can be further divided into value-based algorithms that learn a value function to measure the goodness of a policy and derive the optimal policy by maximizing this value, and policy-based algorithms that directly searches the optimal policy over a restricted policy class. Over time, RL research has evolved across four phases, summarized below in chronological order:

1. Classical RL: Early works studied non-deep-learning algorithms, in the form of tabular methods that store estimates in lookup tables, or using classical ML models for function approximation. Two foundational examples are tabular Q-learning [Watkins and Dayan, 1992] and fitted Qiteration [Ernst et al., 2005]. Among these algorithms, GRPO is closely related to policy-based algorithms such as REINFORCE [Williams, 1992] and actor-critic [Konda and Tsitsiklis, 1999]. As demonstrated later, GRPO is in essence a policy-based algorithm adapted for LLM reasoning.

2. Deep RL: Spurred by the deep learning revolution of the 2010s, a line of advanced deep RL algorithms emerged. This era was largely catalyzed by the success of the deep Q-network in mastering video games [Mnih et al., 2015] and the development of AlphaGo for the game of Go [Silver et al., 2016]. Since then, deep RL has become a cornerstone of modern AI research both methodologically [e.g., Mnih et al., 2016, Van Hasselt et al., 2016, Dabney et al., 2018, Zhou et al., 2020, Chen et al., 2021] and theoretically [e.g., Fan et al., 2020, Feng et al., 2023, Shen et al., 2025, Sun et al., 2025]. Of particular relevance to GRPO are trust region policy optimization [TRPO, Schulman et al., 2015] and its successor, PPO. The latter has been a prominent policybased algorithm widely applied across various domains, including robotics [Andrychowicz et al., 2020] and LLM fine-tuning [Ouyang et al., 2022].

3. Ofline RL: In high-stakes applications where safety concerns make online exploration prohibitive, it is more practical to employ ofline RL that learns exclusively from static, historical datasets [Levine et al., 2020]. The core principle of ofline RL is to apply the pessimistic principle [e.g., Jin et al., 2021, Rashidinejad et al., 2021] for conservative policy learning. While this principle traces back to the seminal works of Swaminathan and Joachims [2015b,a], the paradigm saw a resurgence in the early 2020s [e.g., Kumar et al., 2019, Wu et al., 2019, Yu et al., 2020, Xie et al., 2021, Uehara and Sun, 2022]. Parallel to ofline policy learning, of-policy evaluation seeks to evaluate the impact of adopting a target policy based on the historical data generated by a diferent policy [e.g., Thomas et al., 2015, Jiang and Li, 2016, Thomas and Brunskill, 2016, Liu et al., 2018, Xie et al., 2019, Kallus and Uehara, 2022]; see Uehara et al. [2022] for a recent review.

4. RLHF: Following the seminal work by Ouyang et al. [2022], the field has seen an unprecedented proliferation of literature dedicated to RLHF [Christiano et al., 2017], with the goal of aligning the output of LLMs with human preferences through RL [e.g., Munos et al., 2023, Rafailov et al., 2023, Wu et al., 2024, 2025b, Zeng et al., 2024]. Theoretically, existing works have explored both the asymptotic distribution of the parameter estimates [Liu et al., 2024] and finite-sample error bounds for the sub-optimality gap [e.g., Chowdhury et al., 2024, Zhong et al., 2024, Aminian et al., 2025, Ye et al., 2025, Xu et al., 2025a] and regret [Zhang et al., 2025d]. Despite these analyses, GRPO – a key driver of recent breakthroughs in LLM reasoning – remains largely under-theorized.

Finally, RL is closely related to two branches of research in statistics, primarily motivated by healthcare applications; see Chakraborty and Moodie [2013], Kosorok and Laber [2019], Tsiatis et al. [2019], Li et al. [2023], Shi [2025], Ge et al. [2025], Gazi et al. [2026] for reviews. (i) Early works develop RL algorithms to learn optimal dynamic treatment regimes (DTRs), which are individualized strategies that tailor medical interventions to a patient’s unique characteristics and evolving clinical state. Some representative algorithms include Q-learning [Qian and Murphy, 2011, Song et al., 2015], A-learning [Murphy, 2003, Robins, 2004, Shi et al., 2018] and policy-based methods [Zhang et al., 2013, Zhao et al., 2015]. These studies focus on short-horizon settings, where treatment decisions are made at a single stage or over a small, finite number of stages. (ii) More recently, the literature has expanded to study MDPs under long-horizon settings [e.g., Ertefaie and Strawderman, 2018, Luckett et al., 2020, Liao et al., 2022, Chen et al., 2024, Li et al., 2024a, Shi et al., 2024a,b, Zhou et al., 2024, Ma et al., 2025, Jin et al., 2025, Li et al., 2025b, Zhong et al., 2025] as well as RLHF [e.g., Lee et al., 2024, Liu et al., 2025b,a, Lu et al., 2025, Xiao et al., 2025b].

## 2.2 Reinforcement learning from verifiable rewards

RLVR, an LLM post-training strategy introduced by Lambert et al. [2025], directly optimizes LLMs against verifiable outcomes. Unlike RLHF, which requires to learn a reward function from subjective human preferences across multiple candidate responses, RLVR leverages objective feedback signals – for example, by verifying whether a model’s answer matches a ground-truth mathematical solution or by executing the model’s generated code in programming tasks. The original algorithm in Lambert et al. [2025] relied on PPO for policy learning, which learns a separate critic model that evaluates the quality of the learning policy.

GRPO, a major breakthrough in RLVR, drastically scales the reasoning capacities of existing LLMs. Its key ingredient lies in completely eliminating the critic model, sampling multiple reasoning traces, and using their average reward as a proxy for the critic (see Section 3 for details). This approach, originally introduced in DeepSeekMath [Shao et al., 2024], was largely popularized by DeepSeek-R1 [Guo et al., 2025] and later validated by a series of pioneering open-source reasoning models like Qwen2.5 [Yang et al., 2025]. Together, these works have sparked a surge of followup RLVR algorithms [see, e.g., Zhang et al., 2025a, for a recent review], which can be broadly categorized into three types:

1. The first line of works refined the GRPO policy gradient estimator by (i) modifying [Hao et al., 2025, Xiao et al., 2025a, Zeng et al., 2025] or replacing [Li et al., 2024b, Ahmadian et al., 2024, Hu et al., 2025] the baseline term (the empirical group mean in GRPO); (ii) revising the importance sampling ratio used in GRPO [Zheng et al., 2025a, Pang and Jin, 2025] or removing it altogether [Chu et al., 2025]; and (iii) applying diferent normalizations to the reward [Xiong et al., 2025, Liu et al., 2025c, Xiao et al., 2025a]. In summary, these algorithms preserve the critic-free GRPO framework while modifying how policy gradient estimators are computed or normalized.

2. The second line of works considered diferent optimization objectives, including entropy-regularized objectives that encourage exploration to prevent the model from entropy collapse [Zhang et al., 2025b, Cheng et al., 2025, Chen et al., 2025], length- or dificulty-aware objectives that balance correctness with concise reasoning [Zhang and Zuo, 2025, Dai et al., 2025a], risk-sensitive targets [Ren et al., 2026] and objectives that guide the model to adaptively select reasoning formats [Wu et al., 2025a].

3. The third line of works focused on improving GRPO’s training eficiency, either statistically or computationally [Dai et al., 2025b, Yu et al., 2025, Lin et al., 2025, Xu et al., 2025b, Zhang et al., 2025c]. For example, Yan et al. [2025] leveraged of-policy data from stronger models to provide informative reasoning trajectories and enhance learning. Li et al. [2025a] and Zhan et al. [2026] reused previously generated trajectories instead of discarding all old samples after each policy update. Zheng et al. [2025b] explored multiple reasoning paths concurrently to accelerate reasoning speed, while Xu and Ding [2026] generated only a single sampled output per prompt to reduce computational cost.

Despite the popularity in developing practical RLVR algorithms, their theoretical foundations – and those of GRPO specifically – remain largely unexplored. Among those available, Liu et al. [2025c] and Yang et al. [2026] studied the biases of GRPO’s policy gradient and advantage function (see Section 3 for formal definitions). Pang and Jin [2025] upper bounded the expected squared $\ell _ { 2 } \cdot$ -norm of the gradient for GRPO’s KL-regularized objective. Davis and Recht [2025] and Vojnovic and Yun [2025] characterized GRPO’s objective function. In particular, Davis and Recht [2025] proved GRPO optimizes an arcsin transformation of the expected reward, rather than the expected reward in its original form. More recently, Yao et al. [2026] discussed the of-policy nature of GRPO. However, these results do not deliver a unified finite-sample and asymptotic characterization of GRPO, nor do they establish a connection to U-statistics – both of which are precisely what our theory establishes.

## 2.3 U-statistics

Our work builds upon the classical theory of U-statistics, a class of estimators introduced by Hoefding [1948] that generalize the sample mean to averages over functions of multiple random variables or vectors. Let $\{ X _ { i } \} _ { i = 1 } ^ { n }$ denote a sequence of independent and identically distributed (i.i.d.) random vectors. A U-statistic of order m is defined by a symmetric kernel function $h ( X _ { 1 } , \ldots , X _ { m } )$ For instance, a second-order U-statistic with a bivariate kernel h is defined as:

$$
U = \frac {1}{n (n - 1)} \sum_ {i \neq j} h (X _ {i}, X _ {j}).\tag{1}
$$

Its statistical property is largely characterized through the Hoefding decomposition, which expresses the U-statistic as a sum of several orthogonal components:

$$
U = h _ {0} + \underbrace {\frac {2}{n} \sum_ {i = 1} ^ {n} [ h _ {1} (X _ {i}) - h _ {0} ]} _ {\text {first - order term}} + \underbrace {\frac {1}{n (n - 1)} \sum_ {i \neq j} [ h (X _ {i} , X _ {j}) - h _ {1} (X _ {i}) - h _ {1} (X _ {j}) + h _ {0} ]} _ {\text {second - order term}},\tag{2}
$$

where $h _ { 0 } = \mathbb { E } [ h ( X _ { 1 } , X _ { 2 } ) ]$ is the expectation of the kernel and $h _ { 1 } ( x ) = \mathbb { E } [ h ( x , X _ { 2 } ) ]$ denotes the first-order projection.

In the non-degenerate case where $\mathrm { V a r } ( h _ { 1 } ( X _ { 1 } ) ) > 0$ , the first-order term dominates the variance and fluctuates at an order of $O _ { p } ( n ^ { - 1 / 2 } )$ , whereas the second-order term decays at a faster rate of $O _ { p } ( n ^ { - 1 } )$ . In essence, this decomposition establishes the asymptotic equivalence between a Ustatistic and a simple average of i.i.d. random variables (captured by the first-order term), allowing classical limit theorems for sample averages to be directly applied to U-statistics. As we will demonstrate in Section 4, this decomposition is instrumental in characterizing the behavior of GRPO’s policy gradient and its estimated optimal policy.

Owing to their favorable theoretical properties, U-statistics are widely employed across disciplines, including their extension to U-processes in probability theory [Nolan and Pollard, 1987], their use in semiparametric statistics via higher-order influence functions [Liu et al., 2017], their role in econometrics through the maximum rank correlation estimator [Han, 1987, Sherman, 1993], and their application in estimating optimal individualized treatment regimes via concordance-assisted learning [Fan et al., 2017, Liang et al., 2018].

## 3 Preliminaries

In this section, we first introduce the sequential decision making problem in RL and formulates LLM reasoning as a sequential decision making problem (Section 3.1). We next present a meta-algorithm (see Algorithm 1 for the pseudocode) that unifies a range of policy-based approaches, ranging from the classical REINFORCE algorithm to more advanced advantage actor-critic methods and variants of GRPO (Section 3.2).

## 3.1 Problem setup

RL is a powerful machine learning framework for solving sequential decision making problems. In this framework, an agent repeatedly interacts with an environment. At each time step, the agent receives an observation which represents the environment’s current state, and selects an action based on this information. In response, the environment provides feedback in the form of a reward and transitions to a new state, yielding the next observation. The agent’s objective is to utilize the collected observation-action-reward tuples to learn an optimal policy – a mapping from its observed data history to the space of actions – that maximizes the expected cumulative reward in the long run.

In natural language processing, text is represented as a sequence of discrete units known as tokens. Let V denote the vocabulary consisting of all such tokens. LLMs produce text via autoregressive next token prediction. Specifically, given an input prompt X (e.g., a user query) represented as a sequence of tokens, the model generates the first token $Y _ { 1 } \in \mathcal { V }$ sampled from a conditional distribution $\pi _ { \boldsymbol { \theta } } ( \bullet \vert X ) = \mathbb { P } ( Y _ { 1 } = \bullet \vert X )$ , parameterized by $\theta \in \Theta$ . Next, the second token $Y _ { 2 }$ is produced according to $\pi _ { \boldsymbol { \theta } } ( \bullet | X , Y _ { 1 } )$ . At each time step $t ,$ the model generates $Y _ { t }$ according to $\pi _ { \boldsymbol { \theta } } ( \bullet \vert X , Y _ { < t } )$ , where $Y _ { < t } = ( Y _ { 1 } , \ldots , Y _ { t - 1 } )$ represents the previously generated prefix. This iterative procedure continues until the model generates a complete output $Y = ( Y _ { 1 } , \dots , Y _ { T } )$ , terminating with an “end-of-sequence” token $Y _ { T }$ . To handle conditioning sets of varying lengths, a Transformer architecture [Vaswani et al., 2017] is commonly employed to parameterize $\pi _ { \theta }$ . For reasoning tasks, the output Y contains both a reasoning trace and a final solution (denoted by S); see Figure 1 for an illustration.

The above procedure can be naturally formulated as a sequential decision making problem. Specifically, the observation is fixed as the input prompt X, the action at time step t corresponds to the generated token $Y _ { t } .$ , and the policy is given by $\pi _ { \theta }$ . Unlike standard RL settings where rewards may be provided at each step, the reward here is sparse and observed only upon completion of the entire sequence. In GRPO, the terminal reward, denoted by $Z ,$ evaluates both the format consistency of the response (i.e., its adherence to a pre-specified template) and the accuracy of the final solution S. All intermediate rewards are set to zero. Our goal is to optimize the policy $\pi _ { \theta } ,$ , or equivalently, the parameter θ, to maximize the expected reward of the generated output $\mathbb { E } ^ { \pi _ { \theta } } ( Z )$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 A meta-algorithm for LLM reasoning.

1: Input: Prompt distribution  $f(X)$ , initial parameters  $\theta_{0} \in \Theta$ , sequence of learning rates  $\{\eta_{i}\}_{i \in N}$ , batch size B, per-prompt group size G, and baseline functions  $\{C_{i}^{(b,g)}\}_{i,b,g}$ .

2: for  $i = 0, 1, 2, \ldots, n - 1$  do

3: Sample a batch of prompts  $\{X^{(b)}\}_{b=1}^{B} \stackrel{iid}{\sim} f(\bullet)$ .

4: For each  $X^{(b)}$ , generate a group of G outputs  $\{Y^{(b,g)}\}_{g=1}^{G} \sim \pi_{\theta_{i}}(\cdot | X^{(b)})$ .

5: For each  $Y^{(b,g)}$ , obtain its reward  $Z^{(b,g)}$ .

6: Compute the gradient

 $\widehat{g}(\theta_{i}) = \frac{1}{BG} \sum_{b=1}^{B} \sum_{g=1}^{G} \sum_{t} \nabla_{\theta} \log \pi_{\theta_{i}}(Y_{t}^{(b,g)} | X^{(b)}, Y_{&lt;t}^{(b,g)}) (Z^{(b,g)} - C_{i}^{(b,g)})$ 

7: Parameter update:  $\theta_{i+1} \leftarrow \theta_{i} + \eta_{i+1} \widehat{g}(\theta_{i})$ .

8: end for

9: Output:  $\pi_{\theta_{n}}$ .
</div>

Alternatively, this problem can be viewed from a diferent perspective by treating the entire output sequence $Y$ as a single action. This efectively collapses the time horizon T to 1. Consequently, the problem is recast as a bandit problem [e.g., Lai and Robbins, 1985], where the data is summarized by the context-action-reward $( X , Y , Z )$ tuples.

## 3.2 A meta-algorithm

Existing RLVR algorithms are policy-based algorithms, grounded in the policy gradient theorem [Sutton et al., 1999]. This theorem is instrumental as it provides a closed-form expression for the gradient of the expected reward with respect to the policy parameters, which enables the application of gradient-based algorithms for parameter optimization. In our specific formulation, the gradient of the expected reward can be expressed as:

$$
g (\theta) := \nabla_ {\theta} [ \mathbb {E} ^ {\pi_ {\theta}} (Z) ] = \mathbb {E} \Big [ \sum_ {t} \nabla_ {\theta} \log \pi_ {\theta} (Y _ {t} | X, Y _ {<   t}) Z \Big ].\tag{3}
$$

Equation (3) gives rise to the classical REINFORCE algorithm [Williams, 1992]. At the ith iteration, the algorithm (i) samples a prompt X from a reference distribution $f ( \bullet ) ^ { 1 }$ , (ii) utilizes the current parameter $\theta _ { i }$ to generate an output Y, (iii) computes its reward $Z , \mathrm { ( i v ) }$ constructs a plug-in estimator $\begin{array} { r } { \widehat { g } ( \theta _ { i } ) = \sum _ { t } \nabla _ { \theta } \log \pi _ { \theta } ( Y _ { t } | X , Y _ { < t } ) Z } \end{array}$ based on these samples, and (v) updates the parameters via stochastic gradient ascent $\theta _ { i + 1 }  \theta _ { i } + \eta _ { i } \widehat { g } ( \theta _ { i } )$ with learning rate $\eta _ { i } > 0$

While REINFORCE’s gradient estimator is unbiased, it is notoriously susceptible to high variance. A common trick in the literature to reduce its variance is to subtract a baseline term $C _ { i }$ (which can vary across iterations i) from the reward Z. The rationale lies in that the score function $\textstyle \sum _ { t } \nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( Y _ { t } | \boldsymbol { X } , Y _ { < t } )$ has an expectation of zero. Consequently, provided that $C _ { i }$ is a function exclusively of the random variable X (being conditionally independent of Y and Z given X), the expectation in (3) remains invariant when Z is replaced by its “centered” version, $Z - C _ { i }$ . As such, the resulting gradient estimator remains unbiased. However, a proper choice of $C _ { i }$ can efectively reduce the variance of the estimator.

The most widely adopted $C _ { i }$ is the value function $V ^ { \pi _ { \theta _ { i } } } ( X ) = \mathbb { E } ^ { \pi _ { \theta _ { i } } } ( Z | X )$ . This choice gives rise to advantage actor-critic (A2C) algorithms [e.g., Mnih et al., 2016], which, in addition to the policy (the actor), learn a value function (the critic) to estimate the baseline and calculate the advantage function (the diference between the return and the critic) for variance reduction. However, in reasoning tasks, maintaining and updating a separate critic network is extremely computationally intensive. GRPO addresses this ineficiency by eliminating the critic network entirely. It samples multiple outputs for each prompt and utilizes their group mean as the baseline term (see Equation (4)). Because sampling from the policy is drastically more eficient than training a critic model, GRPO ofers a highly scalable solution to LLM reasoning. In Section 4, we show that this solution is also mathematically elegant.

To unify these methods, we introduce the meta-algorithm in Algorithm 1. It incorporates (i) the centering trick for variance reduction, (ii) minibatch sampling of B prompts per iteration, and (iii) G sampled outputs per prompt for evaluating the gradient. Several aforementioned algorithms arise as special cases:

• REINFORCE: Recovered by setting $B = 1 , G = 1$ , and $C _ { i } ^ { ( 1 , 1 ) } = 0$

• A2C: Recovered by setting $B = 1 , G = 1$ , and $C _ { i } ^ { ( 1 , 1 ) }$ to the critic network.

• GRPO-type: Recovered by allowing $B , G > 1$ and setting $C _ { i } ^ { ( b , g ) }$ to the leave-one-out group $m e a n ^ { 2 }$

$$
C _ {i} ^ {(b, g)} = \bar {Z} ^ {(b, - g)} := \frac {\sum_ {k \neq g} Z ^ {(b , g)}}{G - 1}.\tag{4}
$$

We conclude this section by noting that while we focus on analyzing Algorithm 1 for building intuition, it simplifies the standard production-level implementation in three aspects: (i) it omits the reward normalization used in GRPO; (ii) it does not incorporate the importance sampling strategy used in GRPO to enhance sample eficiency through multiple gradient updates per batch; (iii) it excludes the Kullback–Leibler (KL) divergence penalty, which prevents the policy from deviating excessively from the reference model. These simplifications were implemented in GRPO variants such as Dr-GRPO [Liu et al., 2025c] and GPG [Chu et al., 2025], which we analyze rigorously in Sections 4.1 and 4.2. In Section A of the Supplementary Material, we will bridge these gaps to align our theoretical analyses with standard GRPO implementations.

## 4 Main results

This section presents our main theoretical results, where we compare GRPO with two alternatives to demonstrate its efectiveness. To ensure a fair comparison, all algorithms use the same batch size B and group size G. As summarized in Table 1, their diferences lie solely in the choice of the baseline term. Specifically, we analyze:

Table 1: Three meta-algorithms compared in Section 4, difering in their choice of baseline term.

<table><tr><td>Algorithm</td><td>Vanilla algorithm</td><td>GRPO-type algorithm</td><td>Oracle algorithm</td></tr><tr><td>Baseline term</td><td> $C_i^{(b,g)} = 0$ </td><td> $C_i^{(b,g)} = \bar{Z}^{(b,-g)}$ </td><td> $C_i^{(b,g)} = V^{\pi_{\theta_i}}(X^{(b)})$ </td></tr></table>

(i) a vanilla algorithm, a minibatch variant of REINFORCE that sets $C _ { i } ^ { ( b , g ) } = 0 ;$

(ii) a GRPO-type algorithm, which adopts the group mean defined in $( 4 ) ;$

(iii) an oracle algorithm, which uses the true value function $V ^ { \pi _ { \theta _ { i } } } ( X )$ as the baseline;

Notice that the oracle algorithm is not practically implementable: the true value function is unknown and estimating it via a separate network is computationally expensive, as discussed earlier. Nevertheless, this algorithm serve as a benchmark whose performance practical algorithms aim to match. We say that an algorithm achieves the oracle property if it attains asymptotically equivalent performance to the oracle algorithm.

We next provide a high-level summary of our findings, covering both finite-sample and asymptotic results (see Figure 3 for a roadmap of our results):

• Gradient evaluation: Section 4.1 investigates the properties of policy gradient estimators employed by GRPO-type algorithms. Specifically, Lemma 1 formally establishes the connection between the estimator and the U-statistic. Theorem 2 and Proposition 3 derive bounds on its MSE, which in turn demonstrating the estimator’s superiority over the vanilla algorithm, yielding its oracle property (Corollary 4) and optimality (Corollary 5).

• Policy optimization: Section 4.2 shifts the focus to the sub-optimality gap of the learned policy. In particular, Lemma 6 presents a finite-sample upper bound on this gap, upon which Theorem 7 establishes a scaling law that ofers insights into the optimal choice of the group size. Theorem 8 further establishes the consistency of the parameter estimates and derives the asymptotic distribution of the sub-optimality gap as a mixture of $\chi ^ { 2 }$ random variables without assuming parameter identifiability. These results verify the oracle property of the GRPO policy (Corollary 9) as well as its optimality (Corollary 10).

• Practical considerations: Section A of the Supplementary Material extends the aforementioned analyses to accommodate (i) reward standardization; (ii) importance sampling and (iii) the KL divergence penalty. This section connects the gradient estimator with U-statistics (Lemma 11), characterizes the gradient estimator’s MSE (Theorem 12), and derives the consistency of the parameter estimates (Proposition 13).

## 4.1 Group relative gradient evaluation

In this section, we consider the task of evaluating the gradient $g ( \theta )$ (see Equation (3)). To build intuition, we begin with estimating $g ( \theta )$ for a fixed prompt x. This corresponds to the setting where the prompt distribution f in Algorithm 1 is degenerate at x. The observed data consists of G i.i.d. output-reward pairs $\{ ( Y ^ { ( g ) } , Z ^ { ( g ) } ) \} _ { g = 1 } ^ { G }$ sampled from the model’s policy $\pi _ { \boldsymbol { \theta } } .$ . Here, we omit the superscript (b) as the prompt is fixed. We denote the resulting gradient estimators for the vanilla,

GRPO-type and oracle algorithms as ${ \widehat { g } } _ { \mathrm { v a n i l l a } } ( x ; \theta )$ , gbGRPO(x; θ) and ${ \widehat { g } } _ { \mathrm { o r a c l e } } ( x ; \theta )$ , respectively. These estimators follow the general form,

$$
\widehat {g} (x; \theta) = \frac {1}{G} \sum_ {g = 1} ^ {G} \nabla_ {\theta} \log \pi_ {\theta} (Y ^ {(g)} \mid x) [ Z ^ {(g)} - C ^ {(g)} ],\tag{5}
$$

with diferent choices of the baseline term $C ^ { ( g ) }$ summarized in Table 1.

We begin by demonstrating that the GRPO’s group relative gradient estimator, which contrasts a realized reward Z against its group average, admits a second-order U-statistic representation.

Lemma 1 (Gradient estimator as a U-statistic). ${ \widehat { g } } _ { \mathrm { G R P O } } ( x ; \theta )$ can be written as a second-order U-statistic:

$$
\widehat {g} _ {\mathrm{GRPO}} (x; \theta) = \binom {G} {2} ^ {- 1} \sum_ {1 \leq i <   j \leq G} h \big ((Y ^ {(i)}, Z ^ {(i)}), (Y ^ {(j)}, Z ^ {(j)}) \big),\tag{6}
$$

with a symmetric kernel

$$
h \big ((Y ^ {(i)}, Z ^ {(i)}), (Y ^ {(j)}, Z ^ {(j)}) \big) := \frac {1}{2} [ \nabla_ {\theta} \log \pi_ {\theta} (Y ^ {(i)} | x) - \nabla_ {\theta} \log \pi_ {\theta} (Y ^ {(j)} | x) ] (Z ^ {(i)} - Z ^ {(j)}).
$$

The proof of Lemma 1 is straightforward. By setting $C ^ { ( g ) }$ in (5) to the group mean baseline $\begin{array} { r } { \bar { Z } ^ { ( - g ) } = \bar { ( } G - 1 ) ^ { - 1 } \sum _ { k \neq g } Z ^ { ( g ) } } \end{array}$ , each individual term in the sum satisfies

$$
\nabla_ {\theta} \log \pi_ {\theta} (Y ^ {(g)} \mid x) [ Z ^ {(g)} - \bar {Z} ^ {(- g)} ] = \frac {1}{G - 1} \sum_ {k \neq g} \nabla_ {\theta} \log \pi_ {\theta} (Y ^ {(g)} \mid x) [ Z ^ {(g)} - Z ^ {(k)} ].
$$

Averaging over all $g ~ \in ~ \{ 1 , \ldots , G \}$ and applying a standard symmetrization argument allows ${ \widehat { g } } _ { \mathrm { G R P O } } ( x ; \theta )$ to be reformulated as the average of the symmetric kernel h over all pairs, yielding the U-statistic in (6).

Next, applying the Hoefding decomposition from (2) partitions the group relative gradient estimator into three orthogonal components:

(i) the expectation of the kernel, which equals the true gradient

$$
g (x; \theta) = \mathbb {E} ^ {\pi_ {\theta}} [ \nabla_ {\theta} \log \pi_ {\theta} (Y \mid x) \mathbb {E} (Z | X = x) ];
$$

(ii) a first-order term, which can be represented as the diference between the oracle gradient estimator and $g ( x ; \theta )$ , since the first-order projection $h _ { 1 }$ satisfies

$$
2 h _ {1} (y, z) = \mathbb {E} \Big [ h \big ((y, z), (Y, Z) \big) \Big ] = \nabla_ {\theta} \log \pi_ {\theta} (y \mid x) \big [ z - \mathbb {E} ^ {\pi_ {\theta}} (Z \mid X = x) \big ],
$$

which is precisely each summand in (5) under the oracle baseline $C ^ { ( g ) } = V ^ { \pi _ { \theta } } ( x )$ (x);

(iii) a second-order degenerate term.

Since these components are uncorrelated, this decomposition leads to the following MSE bound, as summarized in Theorem 2.

Assumption 1 (Bounded reward). Z is almost surely bounded.

Theorem 2 (MSE conditional on the prompt). Under Assumption 1, we have

$$
\begin{array}{r l} & M S E (\widehat {g} _ {G R P O} (x; \theta)) := \mathbb {E} \| \widehat {g} _ {G R P O} (x; \theta) - g (x; \theta) \| ^ {2} \\ & = \frac {t r a c e [ \Sigma_ {o r a c l e} (x ; \theta) ]}{G} + O \Big (\frac {\mathbb {E} \| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2}}{G ^ {2}} \Big), \end{array}\tag{7}
$$

where trace[•] denotes the trace of a matrix, $\Sigma _ { o r a c l e } ( x ; \theta )$ denotes the asymptotic covariance matrix of the oracle gradient estimator $G C o v ( \widehat { g } _ { o r a c l e } ( x ; \theta ) )$ and $\| \bullet \|$ denotes the $\ell _ { 2 }$ -norm of a vector.

We make two remarks. First, Assumption 1 is generally satisfied. In GRPO, the reward Z is typically the sum of a format reward and an accuracy reward, which evaluate the format consistency and the correctness of the output, respectively. Since both components are bounded within [0, 1], the final reward Z remains bounded within [0, 2].

Second, the two terms on the second line of (7) correspond to the expected squared $\ell _ { 2 } { \mathrm { - n o r m } }$ of the first- and second-order projections from the Hoefding decomposition. Crucially, the first-order projection serves as the leading term that scales at a rate of $G ^ { - 1 }$ and coincides with the MSE of the oracle gradient estimator. In contrast, the second-order term decays at a faster rate of $G ^ { - 2 }$ confirming its role as a higher-order residual. This observation establishes the oracle property of ${ \widehat { g } } _ { \mathrm { G R P O } } ( \theta )$ , which we formalize in Corollary 4: as the group size $G  \infty$ , the estimator’s MSE becomes equivalent to that of the oracle estimator with access to the true value function. In the oficial DeepSeekMath implementation, G is set to 64 [Shao et al., 2024], which is suficiently large for the residual term to be negligible relative to the leading term.

Next, we extend Theorem 2 to the minibatch setting, where the prompt is no longer fixed at x. Instead, we sample B i.i.d. prompts $\{ X ^ { ( b ) } \} _ { b = 1 } ^ { B }$ and G output-reward pairs per prompt to estimate

$$
g (\theta) = \mathbb {E} _ {X \sim f (\bullet)} [ g (X; \theta) ].\tag{8}
$$

Accordingly, we define the vanilla, oracle, and GRPO-type minibatch gradient estimators as the empirical average of the prompt-specific estimators:

$$
\widehat {g} (\theta) = \frac {1}{B} \sum_ {b = 1} ^ {B} \widehat {g} (X ^ {(b)}; \theta).
$$

Proposition 3 (MSE in the minibatch setting). Under Assumption 1, we have

$$
\begin{array}{r l} & M S E (\widehat {g} _ {G R P O} (\theta)) := \mathbb {E} \| \widehat {g} _ {G R P O} (\theta) - g (\theta) \| ^ {2} \\ & = \frac {\mathbb {E} \| g (X ; \theta) - g (\theta) \| ^ {2}}{B} + \frac {t r a c e [ \Sigma_ {o r a c l e} (\theta) ]}{B G} + O \Big (\frac {\mathbb {E} \| \nabla_ {\theta} \log \pi_ {\theta} (Y | X) \| ^ {2}}{B G ^ {2}} \Big), \end{array}\tag{9}
$$

where $\Sigma _ { o r a c l e } ( \theta ) = \mathbb { E } [ \Sigma _ { o r a c l e } ( X ; \theta ) ]$

In the minibatch setting, the MSE is further reduced by a factor of $B ^ { - 1 }$ . The first term in the second line of (9) represents the inherent variance arising from prompt sampling, which is independent of G and shared by all gradient estimators (e.g., vanilla, oracle). Equation (9) is instrumental in deriving our scaling law detailed in the next section. To elaborate, suppose we fix the sampling budget per prompt as $N = B \times G$ . Then $B = N / G$ , so the first term scales linearly with G, the second term is constant when N is fixed, and the third term scales inversely with $G .$ Consequently, there exists an optimal choice of G that is neither too small nor too large, balancing the first and third terms to minimize the MSE. As we will show later, the MSE is closely related to the suboptimality gap of the learned policy, and thus there exists a corresponding G that minimizes this suboptimality gap.

Additionally, as $G  \infty .$ , the MSE of the GRPO-type gradient again becomes equivalent to that of the oracle gradient, and we formalize such oracle property below.

Corollary 4 (Oracle property of gradient estimator). Let $M S E _ { A } ( \bullet )$ denote the asymptotic MSE of a gradient estimator, obtained by removing errors that are high-order in the group size $G$ , we have $M S E _ { A } ( \widehat { g } _ { \mathrm { G R P O } } ( x ; \theta ) ) = M S E ( \widehat { g } _ { o r a c l e } ( x ; \theta ) )$ and $M S E _ { A } ( \widehat { g } _ { \mathrm { G R P O } } ( \theta ) ) = M S E ( \widehat { g } _ { o r a c l e } ( \theta ) )$

Finally, we establish the optimality of the GRPO estimator in Corollary 5 by demonstrating the following properties: (i) it asymptotically minimizes the MSE within the class of gradient estimators of the form (5) where the baseline term is a function of the prompt x only; and (ii) its asymptotic MSE is strictly smaller than that of the vanilla algorithm.

Assumption 2 (Conditional uncorrelation). $\| \nabla _ { \theta } \log \pi _ { \theta } ( Y | X ) \| ^ { 2 }$ is conditionally uncorrelated of Z given X.

Corollary 5 (Optimality of gradient estimator). Suppose Assumptions 1 and 2 hold. For any gradient estimator $\widehat { g } ( x ; \theta )$ of the form (5) whose baseline term is a function of the prompt x only, we have

$$
M S E _ {A} (\widehat {g} _ {\mathrm{GRPO}} (x; \theta)) \leq M S E (\widehat {g} (x; \theta)) \quad a n d \quad M S E _ {A} (\widehat {g} _ {\mathrm{GRPO}} (\theta)) \leq M S E (\widehat {g} (\theta)).
$$

In particular, provided that the score functions $\nabla _ { \theta } \log \pi _ { \theta } ( Y \mid x ) , \nabla _ { \theta } \log \pi _ { \theta } ( Y \mid X )$ and the value function $V ^ { \pi _ { \theta } } ( X )$ are not almost surely zero, and $V ^ { \pi _ { \theta } } ( x ) \neq 0$

$$
M S E _ {A} (\widehat {g} _ {\mathrm{GRPO}} (x; \theta)) <   M S E (\widehat {g} _ {v a n i l l a} (x; \theta)) \quad a n d \quad M S E _ {A} (\widehat {g} _ {\mathrm{GRPO}} (\theta)) <   M S E (\widehat {g} _ {v a n i l l a} (\theta)).
$$

To conclude this section, we remark that Assumption 2 ensures that the oracle gradient estimator minimizes the MSE within the class of unbiased policy gradient estimators [Greensmith et al., 2004]. This assumption is well-supported by the empirical success of modern RL algorithms such as advantage actor-critic and PPO, which prioritize the value function as the optimal baseline for variance reduction.

## 4.2 Group relative policy optimization

This section turns to policy optimization and investigates the properties of the policy learned by Algorithm 1. For a given policy $\pi ,$ we evaluate its performance via the suboptimality gap, defined as the diference in the expected return between $\pi$ and the optimal policy:

$$
\Delta (\pi) = \max _ {\theta \in \Theta} \mathbb {E} ^ {\pi_ {\theta}} (Z) - \mathbb {E} ^ {\pi} (Z)
$$

By definition, a smaller gap indicates a policy closer to the optimal one. Additionally, the subop timality gap measures only the quality of the final policy obtained at the last iteration. In the RL literature, an alternative metric is regret, which quantifies the cumulative diference between the optimal policy and the sequence of intermediate policies across iterations. However, we adopt the suboptimality gap because it better reflects practical LLM applications: once training is complete, only the final policy is deployed in practice. This criterion also aligns with the evaluation metric in deployment-eficient RL [e.g., Huang et al., 2022].

We begin with the following set of conditions under which we establish a finite-sample error bound for the suboptimality gap of the meta-algorithm in Lemma 6.

Assumption 3 (L-smoothness). The expected return $\mathbb { E } ^ { \pi _ { \theta } } ( Z )$ is L-smooth with respect to θ. That is, its gradient $g ( \bullet )$ is diferentiable and Lipschitz continuous with some constant $L > 0$ such that:

$$
\left\| g \left(\theta_ {1}\right) - g \left(\theta_ {2}\right) \right\| \leq L \left\| \theta_ {1} - \theta_ {2} \right\|, \quad \forall \theta_ {1}, \theta_ {2} \in \Theta .
$$

Additionally, the score function $\nabla _ { \theta }$ log $\pi _ { \theta }$ is uniformly bounded and continuous in θ.

Assumption 4 (Polyak-Lojasiewicz (PL) condition). There exists some constant $\mu > 0$ such that

$$
\| g (\theta) \| ^ {2} \geq 2 \mu \Delta (\pi_ {\theta}), \quad \forall \theta \in \Theta .\tag{10}
$$

Assumption 5 (Learning rate). The sequence of learning rates $\{ \eta _ { i } \} _ { i \ge 1 }$ in Algorithm 1 satisfies either (a) a constant schedule where $\eta _ { i } = \beta$ for some $0 < \beta < ( 2 L ) ^ { - 1 }$ $o r \left( b \right)$ an $1 / i$ schedule where $\eta _ { i } = i ^ { - 1 } \beta$ for some constant $\beta > ( 2 \mu ) ^ { - 1 }$

The L-smoothness condition (Assumption 3) is standard in the optimization literature $[ \mathrm { e . g . }$ Nesterov, 2013]. It requires the gradient of the expected return to be a Lipschitz continuous function of θ. The PL condition (Assumption 4) is substantially weaker than the strong concavity condition commonly imposed in the literature $[ \mathrm { e . g . }$ , Boyd and Vandenberghe, 2004]. Specifically, strong concavity requires a unique global optimizer and a strictly negative definite Hessian matrix $\nabla _ { \theta } ^ { 2 } \mathbb { E } ^ { \pi _ { \theta } } ( Z )$ – both requirements are likely violated in over-parameterized models such as LLMs.

The PL condition, to the contrary, accommodates landscapes with multiple global optimizers and singular Hessians. To see this, consider a hypothetical two-dimensional example where $\theta =$ $( \theta _ { 1 } , \theta _ { 2 } ) , \Theta = \mathbb { R } ^ { 2 }$ , and the expected return is $\mathbb { E } ^ { \pi _ { \theta } } ( Z ) = - \theta _ { 1 } ^ { 2 }$ . Its Hessian matrix $\binom { - 2 } { 0 } 0$ has an eigenvalue of 0 and is therefore not negative definite, violating strong concavity. Nonetheless, its gradient is $g ( \theta ) = ( - 2 \theta _ { 1 } , 0 ) ^ { \top }$ , which satisfies (10) for any $\mu \leq 2$

Finally, Assumption 5 considers two standard learning rate schedules. The constant schedule (a) is common in practical applications [Sheng et al., 2025]. To the contrary, the $1 / i$ schedule (b) is motivated by stochastic approximation theory: it ensures that the suboptimality gap theoretically achieves the optimal convergence rate [Zhang, 2016].

Lemma 6 (Finite-sample sub-optimality gap). Suppose Assumptions 1 and 3–5 hold. Suppose each $C _ { i } ^ { ( b , g ) }$ in Algorithm 1 is either a function of $X ^ { ( b ) }$ or a GRPO-type baseline in (4). Let $M =$ sup<sub>θ</sub> $M S E ( \widehat { g } ( \theta ) )$ denote the uniform upper bound on the MSE of this algorithm’s gradient estimator. Then for any $n \geq 1$ , under the constant schedule $( a )$ , the output policy of $\pi _ { \theta _ { n } }$ satisfies

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n}}) ] \leq (1 - 2 \mu \beta + L \mu \beta^ {2}) ^ {n} \mathbb {E} [ \Delta (\pi_ {\theta_ {0}}) ] + \frac {L \beta^ {2} M}{4 \mu \beta - 2 L \mu \beta^ {2}},
$$

where the expectation is taken with respect to the randomness in the estimator $\theta _ { n }$ and any randomness in the initial parameter $\theta _ { 0 }$ . Under the $1 / i$ schedule $( b )$ , we have

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n}}) ] \leq \max \Big (\frac {(1 + \epsilon) L \beta^ {2} M}{(4 \mu \beta - 2) n}, \frac {c}{n} \Big),
$$

for any suficiently small $\epsilon > 0$ , where c is a positive constant depending only on $( \beta , \mu , L , \epsilon )$

Lemma 6 provides a unified analysis for the meta-algorithm, covering vanilla, oracle, and GRPOtype estimators under both learning rate schedules. The bounds reveal that the sub-optimality gap: (i) scales with the smoothness constant L; (ii) decreases with the PL constant $\mu$ and the number of iterations $n ;$ and (iii) is crucially governed by the MSE of the gradient estimator M. As this MSE is the dominant factor (for the constant schedule) and the convergence rate constant (for the $1 / i$ schedule), the oracle property and optimality we established for the GRPO gradient estimator directly translates into the learned policy’s performance. By substituting the MSE bounds derived in Proposition 3 into Lemma 6, we can explicitly characterize how the group size G and batch size B afects the suboptimality gap, leading to the following scaling law for GRPO:

Theorem 7 (Scaling law for GRPO). For GRPO-type algorithms, the sub-optimality upper bounds established in Lemma 6 depend on B and G through the following quantity

$$
\frac {c _ {1}}{B} + \frac {c _ {2}}{B G} + \frac {c _ {3}}{B G ^ {2}},\tag{11}
$$

where the constants are given by $c _ { 1 } = \operatorname* { s u p } _ { \theta } \mathbb { E } \| g ( X ; \theta ) - g ( \theta ) \| _ { 2 } ^ { 2 } , \ c _ { 2 } = \operatorname* { s u p } _ { \theta } t r a c e [ \Sigma _ { o r a c l e } ( \theta ) ]$ and $c _ { 3 } = O ( \operatorname* { s u p } _ { \theta } \mathbb { E } \| \nabla _ { \theta } \log \pi _ { \theta } ( Y | X ) \| ^ { 2 } )$

Under a fixed sampling budget $N = B G$ per iteration or a total sampling budget $N = n B G$ , the optimal group size $G ^ { * }$ that minimizes the sub-optimality upper bounds is:

$$
G ^ {*} = \sqrt {\frac {c _ {3}}{c _ {1}}}.\tag{12}
$$

In GRPO, sampling reasoning traces from the learning policy (Line 4 of Algorithm 1) is computationally expensive. With a fixed sampling budget, Equation (11) reveals a trade-of in the allocation of computational resources. A small G allows for a larger batch size B and more iterations $n ,$ which reduces the first variance term in (11) and accelerates the convergence rates established in Lemma 6. Conversely, a large $G$ reduces the higher-order residual term (the third term in (11)).

Our scaling law proposed in (12) explicitly balances this trade-of. Crucially, the optimal group size $G ^ { * }$ is universal: it is independent of the budget N, the number of iterations n and the learning rate schedule. Instead, $G ^ { * }$ depends solely on the underlying data generating process and the geometry of the policy space. This makes our result highly practical for implementation, as the optimal group size remains constant for a given task regardless of the total compute available. We will verify this observation empirically in Section 5. While the constants $c _ { i }$ are formally defined as suprema over the parameter space, they may yield conservative theoretical bounds. In practice, these suprema can be relaxed by estimating the constants at a few representative parameter values (e.g., those of the reference model).

We remark that the aforementioned results apply only to upper bounds on the suboptimality gap. Although GRPO attains a sharper upper bound, this does not necessarily imply that it achieves a strictly smaller suboptimality gap. To rigorously establish its superior performance, a more refined characterization of its suboptimality gap – beyond bounds – is required. This motivates our study of its asymptotic distribution in Theorem 8 below.

Nonetheless, the asymptotic analysis presents substantial theoretical challenges. Most classical asymptotic results rely on the uniqueness of the population-level optimizer (i.e., a unique $\theta ^ { * }$ that maximizes $\mathbb { E } ^ { \pi _ { \theta } } ( Z ) )$ and the strict negative definiteness of the Hessian matrix $H ( \theta ^ { * } )$ at that optimum. In overparameterized models such as LLMs, both assumptions are inherently violated. This leads to two complications: (i) Parameter convergence is not well-defined in the classical sense: the estimators may oscillate within an optimal manifold of population-level maximizers and approach diferent maximizers across iterations. This lack of identifiability prevents the sequence of estimators from converging to a single fixed point. (ii) The parameter’s asymptotic distribution becomes analytically intractable, since the estimator fails to converge to a fixed point in the first place.

To address the first challenge, we shift our focus from point-wise convergence to set-wise convergence. We define $\Theta ^ { * }$ as the set of all population-level maximizers,

$$
\Theta^ {*} = \{\theta \in \Theta : \theta \in \arg \max _ {\theta^ {*} \in \Theta} \mathbb {E} ^ {\pi_ {\theta^ {*}}} (Z) \},
$$

and characterize parameter convergence by the distance of the estimator $\theta _ { n }$ to the set $\Theta ^ { * }$

$$
d (\theta_ {n}, \Theta^ {*}) = \inf _ {\theta^ {*} \in \Theta^ {*}} \| \theta_ {n} - \theta^ {*} \|.
$$

We say the estimator $\theta _ { n }$ is consistent if $d ( \theta _ { n } , \Theta ^ { \ast } ) \stackrel { P } { \to } 0 \mathrm { a s } n \to \infty$ . To address the second challenge, we recognize that while the parameter estimates are non-identifiable, the suboptimality gap remains identifiable. We thus analyze the asymptotic distribution of the suboptimality gap rather than the parameters themselves.

Consider the asymptotic regime where the number of iterations $n \to \infty$ and other parameters such as the batch size B and group size G are constant. For this analysis, we adopt the $1 / i$ schedule for the learning rate. This is motivated by Lemma 6: under the constant learning rate schedule, the sub-optimality gap might not converge to zero when B and G are fixed. While one could theoretically allow B and $G$ to grow with n or explore alternative decay schedules, these configurations introduce substantial technical complexity into the proofs without altering our major findings. We focus on this specific regime to more clearly elucidate the performance of GRPO, and impose the following conditions.

Assumption 6 (Compact support). The parameter space Θ is compact.

Assumption 7 (Hessian matrices). $\Theta ^ { * }$ is closed and convex, and all Hessian matrices at $\Theta ^ { * }$ share the same rank r. Moreover, there exists an orthogonal projection matrix Q and a strictly negative definite matrix $H ^ { * }$ such that $Q ^ { \top } Q = I _ { r }$ and $Q ^ { \top } H ( \theta ^ { * } ) Q = H ^ { * } , \forall \theta ^ { * } \in \Theta ^ { * }$

Assumption 8 (Convergence of covariance matrix). There exists a (possibly random) matrix Γ such that the following holds almost surely as $n \to \infty$ ，

$$
\mathbb {E} \left\{(\widehat {g} (\theta_ {n}) - g (\theta_ {n})) (\widehat {g} (\theta_ {n}) - g (\theta_ {n})) ^ {\top} \Big | \theta_ {n} \right\} \to \Gamma .
$$

Assumption 9 (Weak strong concavity). Let Π<sub>Θ</sub>∗ denote the projection operator such that $\Pi _ { \Theta ^ { * } } ( \theta ) : =$ arg $\mathrm { m i n } _ { x \in \Theta ^ { * } } d ( \theta , x )$ , we have

$$
\langle g (\theta), \Pi_ {\Theta^ {*}} (\theta) - \theta \rangle \geq \mu d ^ {2} (\theta , \Theta^ {*}), \quad \forall \theta \in \Theta .
$$

Assumption 6 is mild and commonly imposed in the literature [e.g., Schmidt-Hieber, 2020]. Assumption 7 relaxes the standard identifiability condition by allowing the population parameter to form a connected manifold rather than a unique point. It assumes that, after projection onto a suitable r-dimensional subspace, the parameter becomes identifiable with a strictly negative definite Hessian $H ^ { * }$ . This efectively decomposes the parameter space into an identifiable component defined through the transformation matrix $Q$ and a non-identifiable component along its orthogonal complement. Assumption 8 is not overly restrictive, as the limiting matrix Γ is allowed to be random. The resulting asymptotic distribution can be interpreted as a mixture distribution: first, Γ is realized, and then the limiting distribution is obtained conditionally on Γ. Finally, Assumption 9 imposes a weak strong concavity (WSC) condition. It is weaker than strong concavity (which entails a unique optimizer) but stronger than PL (Assumption 4). In contrast to strong concavity, WSC does not require the objective function to be globally concave in all directions, but only along directions that point toward the optimal set $\Theta ^ { * }$ . Thus, the objective function $\mathbb { E } ^ { \pi _ { \theta } } ( Z )$ may remain flat along other directions, allowing multiple optimizers to exist. Under the L-smoothness condition (Assumption 3), it can be shown that WSC implies the PL condition.

Theorem 8 (Consistency & asymptotic distribution). Suppose Assumptions 1, ${ \mathit { 3 , 4 , 5 ( b ) } }$ and 6 hold. Suppose each $C _ { i } ^ { ( b , g ) }$ in Algorithm 1 is either a function of $X ^ { ( b ) }$ or a GRPO-type baseline in (4). Then the estimator $\theta _ { n }$ is consistent in the sense that $d ( \theta _ { n } , \Theta ^ { \ast } ) \stackrel { P } {  } 0$

Suppose Assumptions $\gamma - g$ hold additionally. Then the output policy satisfies

$$
n \Delta (\pi_ {\theta_ {n}}) \xrightarrow {d} \sum_ {k = 1} ^ {r} w _ {k} \chi_ {1, k} ^ {2},
$$

where r is the rank of the Hessian matrix, $\{ \chi _ { 1 , k } ^ { 2 } \} _ { k = 1 } ^ { r }$ are $i . i . d . \ \chi _ { 1 } ^ { 2 }$ random variables, and $\{ w _ { k } \} _ { k = 1 } ^ { r }$ are positive weights arranged in non-increasing order, determined by the asymptotic covariance matrix of the gradient estimators (see step (iv) of the proof of Theorem 8 in the Supplementary Material).

Theorem 8 is novel in two respects: (i) It establishes the asymptotic distribution of the suboptimality gap for policy gradient estimators as a weighted sum of independent $\chi ^ { 2 }$ random variables, whereas most existing literature focuses primarily on finite-sample error bounds. (ii) It derives this limiting distribution in the overparameterized regime, which goes beyond the classical assumption of a unique optimizer with a non-singular Hessian.

Nonetheless, establishing such a result in the overparameterized regime is non-trivial. While classical results have established CLTs for parameter estimates in stochastic gradient algorithms [e.g., Zhang, 2016], these results do not directly apply due to overparameterization. Our key idea is to first establish a CLT for the subset of parameters corresponding to directions in which the Hessian is negative definite. A second-order Taylor expansion then shows that the suboptimality gap is driven entirely by the asymptotic behavior of these directions, which facilitates the derivation of its limit.

As mentioned earlier, the weights $w _ { k } \mathrm { s }$ are determined by the covariance matrices of the gradient estimators. The oracle and optimality properties of GRPO gradient estimators (Corollaries 4 and 5) directly afect these weights, which in turn establishes the oracle and optimality properties of the GRPO policy. We summarize these results in Corollaries 9 and 10 below.

Corollary 9 (Oracle property of the policy). Suppose the assumptions in Theorem 8 hold. Suppose the projected covariance matrix $\Omega ( \theta ^ { * } ) = C o v ( Q ^ { \top } \widehat { g } ( \theta ^ { * } ) )$ ) depends on $\theta ^ { * } \in \Theta ^ { * }$ only through $Q ^ { \top } \theta ^ { * }$ and is non-singular. Then GRPO’s weights and oracle weights satisfy $w _ { k , \mathrm { G R P O } } - w _ { k , \mathrm { o r a c l e } } = O ( G ^ { - 2 } )$ Consequently, the suboptimality gap of GRPO is asymptotically equivalent to that of the oracle algorithm as $G  \infty$

(a) Base model

(b) Instruct model

(c) ICL model  
Figure 4: MSEs of three policy gradient estimators (vanilla, GRPO-type, and oracle) under three model configurations (base, instruct and in-context learning (ICL)) for diferent group sizes. Error bars represent 95% confidence intervals of the empirically estimated MSEs.

Corollary 10 (Optimality of the policy). Suppose the assumptions in Corollary 9 hold. Suppose $Q ^ { \top } \nabla _ { \theta } \log \pi _ { \theta ^ { * } } ( Y | X ) [ \nabla _ { \theta } \log \pi _ { \theta ^ { * } } ( Y | X ) ] ^ { \top } Q$ is conditionally uncorrelated with Z given X, for some $\theta ^ { * } \in \Theta ^ { * }$ . Then for any meta-algorithm whose baseline term is a function of the prompt only, its weights satisfy $w _ { k } \geq w _ { k , \mathrm { o r a c l e } } + O ( G ^ { - 2 } )$ . Consequently, as $G  \infty$ , its suboptimality gap is asymptotically no smaller than that of GRPO.

## 5 Experiments

We conduct two sets of experiments in this section to validate our theoretical findings. In Section 5.1, we empirically compare GRPO with the vanilla and oracle algorithms in terms of gradien evaluation, to verify the oracle property of the GRPO gradient estimator (Corollary 4) and its su periority over the vanilla estimator (Corollary 5). In Section 5.2, we investigate the optimal group size for policy optimization, to verify the universality of the optimal group size $G ^ { * }$ established b our scaling law (Theorem 7).

## 5.1 Oracle property in gradient evaluation

We first evaluate the MSE of the three gradient estimators introduced in Section 4.1: vanilla, GRPO-type, and oracle. To conduct this comparison, we construct a synthetic arithmetic dataset consisting of 500 questions. Each question is a medium-dificulty integer arithmetic problem, sampled uniformly from five categories: (i) two-step addition or subtraction, (ii) three-step addition or subtraction, (iii) single-step multiplication, (iv) integer division, and (v) addition or subtraction with parentheses. For each generated problem, we record the ground-truth solution. We then query an LLM (the prompt is provided in Section C of the Supplementary Material), extract the solution from its output, and compute a binary reward indicating whether the solution equals the ground truth.

We evaluate gradients aggregated over the 500 questions (see Equation (8)) with respect to three target policies, derived from three models with progressively stronger reasoning capabilities: (i) Qwen/Qwen2.5-0.5B Base model, (ii) Qwen/Qwen2.5-0.5B Instruct model, and (iii) Qwen/Qwen2.5-0.5B Instruct model with in-context learning [ICL, Brown et al., 2020]. For the third model, in addition to the prompt, we provide a few-shot in-context demonstration containing several arithmetic questions with solutions (see Section C of the Supplementary Material). These models produce target policies of diferent quality: the base model is expected to achieve the lowest accuracy, while the ICL model attains the highest. We also consider multiple groups sizes $G \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ to estimate the gradient and use Monte Carlo simulations to evaluate each gradient estimator’s MSE and report their associated 95% confidence interval in Figure 4.

We make the following observations:

1. Across all combinations of group size and target policy, the vanilla estimator (blue line) exhibits the largest MSE, reflecting the well-known high-variance nature of REINFORCE. In contrast, the GRPO-type estimator (orange line) achieves significantly smaller MSEs in all cases, which verifies the second assertion of Corollary 5 and demonstrates its superiority over the vanilla estimator.

2. The oracle estimator (green line) achieves the smallest MSE, as expected. Nevertheless, with a moderately large group size $( G = 8 )$ , the MSE of the GRPO-type estimator is already close to that of the oracle estimator. As G increases further $( \mathrm { e . g . , } G = 3 2 ~ \mathrm { o r } ~ 6 4 )$ , the two estimators become nearly indistinguishable, confirming the oracle property in Corollary 4.

3. Finally, the MSE of all estimators decreases as the group size G increases. It also decreases with the model’s reasoning capability. This behavior is expected as well: as the model becomes stronger, its outputs are more likely to be correct and thus more deterministic. Conversely, weaker models generate more random outputs, which enlarges the variance of the gradient estimator.

## 5.2 Optimal group size for policy optimization

As shown in our theoretical analysis and empirical results (Section 5.2), increasing the group size G reduces the MSE of the GRPO-type gradient estimator. However, a larger G also increases computational cost, as more outputs must be sampled per prompt. Under a fixed sampling budget (i.e., a fixed total number of sampled outputs), our scaling law in Theorem 7 shows that the optimal group size $G ^ { * }$ depends only on the data and the policy model, and is universal with respect to other parameters such as the number of iterations n and the total budget per prompt N. In this section, we verify this universality using two widely adopted math reasoning benchmark datasets, GSM8K [Cobbe et al., 2021] and MATH [Hendrycks et al., 2021].

We first utilize GSM8K to examine the universality of $G ^ { * }$ across diferent training iterations n. Specifically, we fix the total sampling budget per prompt at $N = B G = 1 0 2 4$ and evaluate six candidate group sizes: $G \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ }. For each choice of $G ,$ , we apply GRPO to fine-tune the Qwen2.5-1.5B Instruct model and calculate its test accuracy, defined as the percentage of correctly solved problems in the test dataset. In addition to reporting the final accuracy at the last iteration, we also record accuracy at intermediate checkpoints where $n \in \{ 2 0 0 , 3 0 0 , 4 0 0 , 6 0 0 , 8 0 0 \}$ Given the inherent stochasticity of RLVR – arising, for example, from randomness in sampled outputs – which often leads to high variance across runs, we repeat the training five times for each value of G. We report the mean accuracy along with its associated confidence interval in Figure 5. We note that much of the existing literature reports results from a single training run due to the substantial computational cost of training, which can lead to results that are dificult to reproduce.

We make the following observations from Figure 5:

1. Except at training step $n = 2 0 0$ , the test accuracy generally increases with the group size $G$ and then decreases as G becomes larger. This trend is consistent with the scaling law in Theorem 7.

Step 200

Step 300

Step 400

Step 600

Step 800

Final Step  
Figure 5: Test accuracy of GRPO-fine-tuned models at diferent training steps with a fixed sampling budget of $N = B \times G = 1 0 2 4$ per prompt. Both training and evaluation are conducted on GSM8K. Each curve shows accuracy as a function of the group size $G \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ . Results are averaged over five independent runs, with shaded regions visualizing 95% confidence bands.

When G is small, the second-order residual term can be large, inflating the variance of the gradient estimator. Conversely, when $G$ is large, fixing the total sampling budget forces the batch size B to be small, which enlarges the first variance term in (11). As a result, the optimal group size $G ^ { * }$ lies between these two extremes.

2. With the exception of $n = 2 0 0$ , the optimal group size is consistently $G ^ { * } = 3 2$ across all training steps, demonstrating the universality of $G ^ { * }$ with respect to n. For $n = 3 0 0$ and $n = 8 0 0$ , the performance with $G = 1 6$ is very close to that with $G = 3 2$ . However, its accuracy deteriorates more noticeably for other values of n.

3. Finally, the diferences in test accuracy across diferent values of G are mostly statistically insignificant, as only five independent runs are conducted for each setting. This limitation is due to the high computational cost of training. While increasing the number of runs could yield more statistically meaningful conclusions, doing so is not computationally feasible at this stage.

We next use the MATH dataset to assess the universality of the optimal group size $G ^ { * }$ across diferent sampling budgets N. The procedure closely follows that used for GSM8K: we consider the same candidate group sizes $G \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and train the model separately for each choice of $G .$ The diferences are that, for MATH, we fine-tune a larger and more powerful Qwen2.5-Math-7B model, and we evaluate three diferent sampling budgets, $N \in \{ 1 0 2 4 , 2 0 4 8 , 4 0 9 6 \}$ Test accuracies of the resulting trained models are reported in Table 2.

It can be seen that the optimal group size is mostly 64, but increases to 128 as the sampling budget grows. We suspect this shift is due to the finite number of prompts, which remains constant with respect to the batch size B and group size G, and is not accounted for in our theoretical analysis for simplicity. Meanwhile, when the sampling budget is fixed at 1024, the optimal $G ^ { * }$ is larger than that for GSM8K. This shift is expected, as the constants in (11) depend on the data and the model, and thus the optimal $G ^ { * }$ varies accordingly. These results suggest that larger models may benefit from a larger group size during training.

Table 2: Test accuracy of GRPO-fine-tuned models at the final training step. Each row reports accuracy as a function of the group size $G \in { 4 , 8 , 1 6 , 3 2 , 6 4 }$ , 128 with a fixed sampling budget per prompt; the sampling budget varies across rows. Both training and evaluation are conducted on MATH. Due to the high computational cost of training a 7B model, results are reported from a single run, with the highest accuracy highlighted in bold.

<table><tr><td rowspan="2">Sampling budget</td><td colspan="6">Group size G</td></tr><tr><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td>1024</td><td>0.7677</td><td>0.7491</td><td>0.7753</td><td>0.7627</td><td>0.7817</td><td>0.7743</td></tr><tr><td>2048</td><td>0.7703</td><td>0.7679</td><td>0.7697</td><td>0.7793</td><td>0.7819</td><td>0.7665</td></tr><tr><td>4096</td><td>0.7691</td><td>0.7713</td><td>0.7701</td><td>0.7673</td><td>0.7703</td><td>0.7757</td></tr></table>

## 6 Conclusion

This paper provides a rigorous theoretical analysis of GRPO, a cornerstone algorithm for enhancing the reasoning capabilities of LLMs. We show that GRPO is a statistically principled policy gradient algorithm whose gradient estimator naturally forms a U-statistic (Lemma 1). Leveraging Hoefding’s decomposition, we characterize its MSE (Theorem 2, Proposition 3) and establish both oracle (Corollary 4) and optimality (Corollary 5) properties. For policy optimization, we derive explicit bounds on GRPO’s suboptimality gap (Lemma 6), leading to a scaling law that informs the optimal choice of group size (Theorem 7). We characterize the asymptotic distribution of the suboptimality gap (Theorem 8), confirming GRPO’s oracle (Corollary 9) and optimality (Corollary 10) properties in policy learning. We further extend these results to practical settings by accommodating reward normalization, importance sampling, and KL-divergence penalties (Lemma 11, Theorem 12, Proposition 13). Finally, we empirically validate GRPO’s gradient estimator’s oracle and optimality properties (Figure 4), and verify our scaling law with respect to group size G, demonstrating its universality across training iterations (Figure 5) and sampling budgets (Table 2).

## References

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨un, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learn-<sup>¨</sup> ing from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, 2024.

Gholamali Aminian, Amir R. Asadi, Idan Shenfeld, and Youssef Mroueh. Kl-regularized rlhf with multiple reference models: Exact solutions and sample complexity. In Advances in Neural Infor-

mation Processing Systems (NeurIPS), volume 38, 2025. URL https://neurips.cc/virtual/ 2025/poster/116496.

OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3–20, 2020.

Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Bibhas Chakraborty and Erica E Moodie. Statistical methods for dynamic treatment regimes. Springer-Verlag. doi, 10(978-1):4–1, 2013.

Elynn Y Chen, Rui Song, and Michael I Jordan. Reinforcement learning in latent heterogeneous environments. Journal of the American Statistical Association, 119(548):3113–3126, 2024.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In Advances in Neural Information Processing Systems, pages 15084–15097, 2021.

Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. arXiv preprint arXiv:2505.12346, 2025.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025.

Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning language models with noisy feedback. arXiv preprint arXiv:2403.00409, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Milos Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, pages 4299–4307, 2017.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Will Dabney, Mark Rowland, Marc Bellemare, and R´emi Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Muzhi Dai, Shixuan Liu, and Qingyi Si. Stable reinforcement learning for eficient reasoning. arXiv preprint arXiv:2505.18086, 2025a.

Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, et al. Cde: Curiosity-driven exploration for eficient reinforcement learning in large language models. In The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025b.

Damek Davis and Benjamin Recht. What is the objective of reasoning with reinforcement learning? arXiv preprint arXiv:2510.13651, 2025.

Zheng Ding and Weirui Ye. TreeGRPO: Tree-advantage GRPO for online RL post-training of difusion models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=3rZdp4TmUb.

Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research, 6, 2005.

Ashkan Ertefaie and Robert L Strawderman. Constructing dynamic treatment regimes over indefinite time horizons. Biometrika, 105(4):963–977, 2018.

Caiyun Fan, Wenbin Lu, Rui Song, and Yong Zhou. Concordance-assisted learning for estimating optimal individualized treatment regimes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79(5):1565–1582, 2017.

Jianqing Fan, Zhaoran Wang, Yuchen Xie, and Zhuoran Yang. A theoretical analysis of deep q-learning. In Learning for dynamics and control, pages 486–489. PMLR, 2020.

Xingdong Feng, Yuling Jiao, Lican Kang, Baqun Zhang, and Fan Zhou. Over-parameterized deep nonparametric regression for dependent data with its applications to reinforcement learning. Journal of Machine Learning Research, 24(383):1–40, 2023.

Asim H Gazi, Yongyi Guo, Daiqi Gao, Ziping Xu, Kelly W Zhang, and Susan A Murphy. Statistical reinforcement learning in the real world: A survey of challenges and future directions. arXiv preprint arXiv:2601.15353, 2026.

Lin Ge, Hengrui Cai, Runzhe Wan, Yang Xu, and Rui Song. A review of causal decision making. arXiv preprint arXiv:2502.16156, 2025.

Evan Greensmith, Peter L Bartlett, and Jonathan Baxter. Variance reduction techniques for gradient estimates in reinforcement learning. Journal of Machine Learning Research, 5(Nov):1471– 1530, 2004.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

Aaron K. Han. Non-parametric analysis of a generalized regression model: The maximum rank correlation estimator. Journal of Econometrics, 35(2):303–316, 1987.

Yaru Hao, Li Dong, Xun Wu, Shaohan Huang, Zewen Chi, and Furu Wei. On-policy rl with optimal reward baseline. arXiv preprint arXiv:2505.23585, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Wassily Hoefding. A class of statistics with asymptotically normal distribution. The Annals of Mathematical Statistics, 19(3):293–325, 1948.

Jian Hu, Jason Klein Liu, and Wei Shen. Reinforce++: An eficient rlhf algorithm with robustness to both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025.

Jiawei Huang, Jinglin Chen, Li Zhao, Tao Qin, Nan Jiang, and Tie-Yan Liu. Towards deploymenteficient reinforcement learning: Lower bound and optimality. In International Conference on Learning Representations, 2022.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Nan Jiang. A note on loss functions and error compounding in model-based reinforcement learning. arXiv preprint arXiv:2404.09946, 2024.

Nan Jiang and Lihong Li. Doubly robust of-policy value evaluation for reinforcement learning. In International conference on machine learning, pages 652–661. PMLR, 2016.

Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably eficient for ofline rl? In International conference on machine learning, pages 5084–5096. PMLR, 2021.

Ying Jin, Zhimei Ren, Zhuoran Yang, and Zhaoran Wang. Policy learning “without” overlap: Pessimism and generalized empirical bernstein’s inequality. The Annals of Statistics, 53(4): 1483–1512, 2025.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan. github. io/posts/muon, 6(3):4, 2024.

Nathan Kallus and Masatoshi Uehara. Eficiently breaking the curse of horizon in of-policy eval uation with double reinforcement learning. Operations Research, 70(6):3282–3302, 2022.

Vijay Konda and John Tsitsiklis. Actor-critic algorithms. Advances in neural information processing systems, 12, 1999.

Michael R Kosorok and Eric B Laber. Precision medicine. Annual review of statistics and its application, 6(1):263–286, 2019.

Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing of-policy q-learning via bootstrapping error reduction. Advances in neural information processing systems, 32, 2019.

Tze Leung Lai and Herbert Robbins. Asymptotically eficient adaptive allocation rules. Advances in applied mathematics, 6(1):4–22, 1985.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=i1uGbfHHpH.

Seong Jin Lee, Will Wei Sun, and Yufeng Liu. Low-rank contextual reinforcement learning from heterogeneous human feedback. arXiv preprint arXiv:2412.19436, 2024.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Ofline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Gen Li, Laixi Shi, Yuxin Chen, Yuejie Chi, and Yuting Wei. Settling the sample complexity of model-based ofline reinforcement learning. The Annals of Statistics, 52(1):233–260, 2024a.

Siheng Li, Zhanhui Zhou, Wai Lam, Chao Yang, and Chaochao Lu. Repo: Replay-enhanced policy optimization. arXiv preprint arXiv:2506.09340, 2025a.

Yuhan Li, Eugene Han, Yifan Hu, Zhengling Qi, Yifan Cui, and Ruoqing Zhu. Reinforcement learning with continuous actions under unmeasured confounding. Journal of the American Statistical Association, To appear, 2025b.

Zhen Li, Jie Chen, Eric Laber, Fang Liu, and Richard Baumgartner. Optimal treatment regimes: a review and empirical comparison. International Statistical Review, 91(3):427–463, 2023.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: a simple, efective, and eficient reinforcement learning method for aligning large language models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024b.

Shuhan Liang, Wenbin Lu, Rui Song, and Lan Wang. Sparse concordance-assisted learning for optimal treatment decision. Journal of Machine Learning Research, 18(202):1–26, 2018.

Peng Liao, Zhengling Qi, Runzhe Wan, Predrag Klasnja, and Susan A Murphy. Batch policy learning in average reward markov decision processes. Annals of statistics, 50(6):3364, 2022.

Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. Cppo: Accelerating the training of group relative policy optimization-based reasoning models. arXiv preprint arXiv:2503.22342, 2025.

Kaizhao Liu, Qi Long, Zhekun Shi, Weijie J Su, and Jiancong Xiao. Statistical impossibility and possibility of aligning llms with human preferences: From condorcet paradox to nash equilibrium. arXiv preprint arXiv:2503.10990, 2025a.

Lin Liu, Rajarshi Mukherjee, Whitney K Newey, and James M Robins. Semiparametric eficient empirical higher order influence function estimators. arXiv preprint arXiv:1705.07577, 2017.

Pangpang Liu, Chengchun Shi, and Will Wei Sun. Dual active learning for reinforcement learning from human feedback. arXiv preprint arXiv:2410.02504, 2024.

Pangpang Liu, Junwei Lu, and Will Wei Sun. Uncertainty quantification for large language model reward learning under heterogeneous human feedback. arXiv preprint arXiv:2512.03208, 2025b.

Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. Breaking the curse of horizon: Infinitehorizon of-policy estimation. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling, 2025c.

Nan Lu, Ethan X Fang, and Junwei Lu. Contextual online uncertainty-aware preference learning for human feedback. arXiv preprint arXiv:2504.19342, 2025.

Daniel J Luckett, Eric B Laber, Anna R Kahkoska, David M Maahs, Elizabeth Mayer-Davis, and Michael R Kosorok. Estimating dynamic treatment regimes in mobile health using v-learning. Journal of the American Statistical Association, 115(530):692, 2020.

Tao Ma, Jin Zhu, Hengrui Cai, Zhengling Qi, Yunxiao Chen, Chengchun Shi, and Eric B. Laber. Sequential knockofs for variable selection in reinforcement learning. Journal of the American Statistical Association, accepted, 2025.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pages 1928–1937. PMLR, 2016.

R´emi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 18, 2023.

Susan A Murphy. Optimal dynamic treatment regimes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 65(2):331–355, 2003.

Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2013.

Deborah Nolan and David Pollard. U-processes: Rates of convergence. The Annals of Statistics, 15(2):780–799, 1987.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in neural information processing systems, pages 27730–27744, 2022.

Lei Pang and Ruinan Jin. On the theory and practice of grpo: A trajectory-corrected approach with fast convergence. arXiv preprint arXiv:2508.02833, 2025.

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru WANG, Xiusi Chen, Dilek Hakkani-T¨ur, Gokhan Tur, and Heng Ji. ToolRL: Reward is all tool learning needs. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/ forum?id=eOLdGbXT6t.

Min Qian and Susan A Murphy. Performance guarantees for individualized treatment rules. Annals of statistics, 39(2):1180, 2011.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging ofline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems, 34:11702–11716, 2021.

Tao Ren, Jinyang Jiang, Hui Yang, Wan Tian, Minhao Zou, Guanghao Li, Zishi Zhang, Qinghao Wang, Shentao Qin, Yanjun Zhao, Rui Tao, Hui Shao, and Yijie Peng. RiskPO: Risk-based policy optimization with verifiable reward for LLM post-training. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=KjHB7rebQO.

Herbert Robbins and David Siegmund. A convergence theorem for non negative almost supermartingales and some applications. In Optimizing methods in statistics, pages 233–257. Elsevier, 1971.

James M Robins. Optimal structural nested models for optimal sequential decisions. In Proceedings of the Second Seattle Symposium in Biostatistics: analysis of correlated data, pages 189–326. Springer, 2004.

Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. The Annals of Statistics, 48(4):1875, 2020.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guohao Shen, Runpeng Dai, Guojun Wu, Shikai Luo, Chengchun Shi, and Hongtu Zhu. Deep distributional learning with non-crossing quantile network. arXiv preprint arXiv:2504.08215, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

Robert P. Sherman. The limiting distribution of the maximum rank correlation estimator. Econometrica, 61(1):123–137, 1993.

Chengchun Shi. Statistical inference in reinforcement learning: A selective survey. arXiv preprint arXiv:2502.16195, 2025.

Chengchun Shi, Alin Fan, Rui Song, and Wenbin Lu. High-dimensional a-learning for optimal dynamic treatment regimes. Annals of statistics, 46(3):925, 2018.

Chengchun Shi, Shikai Luo, Yuan Le, Hongtu Zhu, and Rui Song. Statistically eficient advantage learning for ofline reinforcement learning in infinite horizons. Journal of the American Statistical Association, 119(545):232–245, 2024a.

Chengchun Shi, Zhengling Qi, Jianing Wang, and Fan Zhou. Value enhancement of reinforcement learning via eficient and robust trust region optimization. Journal of the American Statistical Association, 119(547):2011–2025, 2024b.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016.

Rui Song, Weiwei Wang, Donglin Zeng, and Michael R Kosorok. Penalized q-learning for dynamic treatment regimens. Statistica Sinica, 25(3):901, 2015.

Ke Sun, Yingnan Zhao, Enze Shi, Yafei Wang, Xiaodong Yan, Bei Jiang, and Linglong Kong. Intrinsic benefits of categorical distributional loss: Uncertainty-aware regularized exploration in reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. A Bradford Book, 2018.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Proceedings of the 12th International Conference on Neural Information Processing Systems, pages 1057–1063, 1999.

Adith Swaminathan and Thorsten Joachims. Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research, 16(1):1731–1755, 2015a.

Adith Swaminathan and Thorsten Joachims. The self-normalized estimator for counterfactual learning. In advances in neural information processing systems, volume 28, 2015b.

Philip Thomas and Emma Brunskill. Data-eficient of-policy policy evaluation for reinforcement learning. In International Conference on Machine Learning, pages 2139–2148. PMLR, 2016.

Philip Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. High-confidence of-policy evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.

Anastasios A Tsiatis, Marie Davidian, Shannon T Holloway, and Eric B Laber. Dynamic treatment regimes: Statistical methods for precision medicine. Chapman and Hall/CRC, 2019.

Masatoshi Uehara and Wen Sun. Pessimistic model-based ofline reinforcement learning under partial coverage. In International Conference on Learning Representations, 2022.

Masatoshi Uehara, Chengchun Shi, and Nathan Kallus. A review of of-policy evaluation in reinforcement learning. arXiv preprint arXiv:2212.06355, 2022.

Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double qlearning. In Proceedings of the AAAI conference on artificial intelligence, volume 30, 2016.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, 2017.

Milan Vojnovic and Se-Young Yun. What is the alignment objective of grpo? arXiv preprint arXiv:2502.18548, 2025.

Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279–292, 1992.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Siye Wu, Jian Xie, Yikai Zhang, Aili Chen, Kai Zhang, Yu Su, and Yanghua Xiao. ARM: Adaptive reasoning model. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://openreview.net/forum?id=z9oeQrcNh9.

Tianhao Wu, Banghua Zhu, Ruoyu Zhang, Zhaojin Wen, Kannan Ramchandran, and Jiantao Jiao. Pairwise proximal policy optimization: Language model alignment with comparative rl. In First Conference on Language Modeling, 2024.

Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized ofline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. In The Thirteenth International Conference on Learning Representations (ICLR), 2025b. URL https://arxiv.org/abs/2405.00675.

Changyi Xiao, Mengdi Zhang, and Yixin Cao. Bnpo: Beta normalization policy optimization. arXiv preprint arXiv:2506.02864, 2025a.

Jiancong Xiao, Ziniu Li, Xingyu Xie, Emily Getzen, Cong Fang, Qi Long, and Weijie Su. On the algorithmic bias of aligning large language models with RLHF: Preference collapse and matching regularization. Journal of the American Statistical Association, 120(552):2154–2164, 2025b.

Tengyang Xie, Yifei Ma, and Yu-Xiang Wang. Towards optimal of-policy evaluation for reinforcement learning with marginalized importance sampling. Advances in neural information processing systems, 32, 2019.

Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for ofline reinforcement learning. Advances in neural information processing systems, 34:6683–6694, 2021.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

Erhan Xu, Kai Ye, Hongyi Zhou, Luhan Zhu, Francesco Quinzan, and Chengchun Shi. Doubly robust alignment for large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2025a.

Yixuan Even Xu, Yash Savani, Fei Fang, and J Zico Kolter. Not all rollouts are useful: Downsampling rollouts in llm reinforcement learning. arXiv preprint arXiv:2504.13818, 2025b.

Zhongwen Xu and Zihan Ding. Single-stream policy optimization. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=b61UW62K7W.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under of-policy guidance. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=vO8LLoNWWk.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Fengkai Yang, Zherui Chen, Xiaohan Wang, Xiaodong Lu, Jiajun Chai, Guojun Yin, Wei Lin, Shuai Ma, Fuzhen Zhuang, Deqing Wang, et al. Your group-relative advantage is biased. arXiv preprint arXiv:2601.08521, 2026.

Chaorui Yao, Yanxi Chen, Yuchang Sun, Yushuo Chen, Wenhao Zhang, Xuchen Pan, Yaliang Li, and Bolin Ding. Group-relative REINFORCE is secretly an of-policy algorithm: Demystifying some myths about GRPO and its friends. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=7CFlXvCoN6.

Kai Ye, Hongyi Zhou, Jin Zhu, Francesco Quinzan, and Chengchun Shi. Robust reinforcement learning from human feedback for large language models fine-tuning. arXiv preprint arXiv:2504.03784, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. Mopo: Model-based ofline policy optimization. Advances in Neural Information Processing Systems, 33:14129–14142, 2020.

Guanning Zeng, Zhaoyi Zhou, Daman Arora, and Andrea Zanette. Shrinking the variance: Shrinkage baselines for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2511.03710, 2025.

Y. Zeng, G. Liu, W. Ma, N. Yang, H. Zhang, and J. Wang. Token-level direct preference optimization. In International conference on machine learning, pages 58348–58365. PMLR, 2024.

Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, and Yu Cheng. ExGRPO: Learning to reason from prior successes. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=701tjQXWVk.

Baqun Zhang, Anastasios A Tsiatis, Eric B Laber, and Marie Davidian. Robust estimation of optimal dynamic treatment regimes for sequential treatment decisions. Biometrika, 100(3):681– 694, 2013.

Duzhen Zhang, Zhong-Zhi Li, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Xiuyi Chen, Yingying Zhang, et al. From system 1 to system 2: a survey of reasoning large language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025a.

Jixiao Zhang and Chunsheng Zuo. Grpo-lead: A dificulty-aware reinforcement learning approach for concise mathematical reasoning in language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5642–5665, 2025.

Li-Xin Zhang. Central limit theorems of a recursive stochastic algorithm with applications to adaptive designs. Annals of applied probability: an oficial journal of the Institute of Mathematical Statistics, 26(6):3630–3658, 2016.

Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised LLM reasoning incentivization. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025b. URL https:// openreview.net/forum?id=k8Mim6RI5O.

Xiaojiang Zhang, Jinghui Wang, Zifei Cheng, Wenhao Zhuang, Zheng Lin, Minglei Zhang, Shaojie Wang, Yinghan Cui, Chao Wang, Junyi Peng, et al. Srpo: A cross-domain implementation of large-scale reinforcement learning on llm. arXiv preprint arXiv:2504.14286, 2025c.

Yuheng Zhang, Dian Yu, Baolin Peng, Linfeng Song, Ye Tian, Mingyue Huo, Nan Jiang, Haitao Mi, and Dong Yu. Iterative nash policy optimization: Aligning LLMs with general preferences via no-regret learning. In The Thirteenth International Conference on Learning Representations, 2025d. URL https://openreview.net/forum?id=Pujt3ADZgI.

Ying-Qi Zhao, Donglin Zeng, Eric B Laber, and Michael R Kosorok. New statistical learning methods for estimating optimal dynamic treatment regimes. Journal of the American Statistical Association, 110(510):583–598, 2015.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025a.

Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, and Dong Yu. Parallel-r1: Towards parallel thinking via reinforcement learning. arXiv preprint arXiv:2509.07980, 2025b.

Han Zhong, Xun Deng, Ethan X. Fang, Zhuoran Yang, Zhaoran Wang, and Runze Li. Risk-sensitive deep rl: Variance-constrained actor-critic provably finds globally optimal policy. Journal of the American Statistical Association, To appear, 2025.

Huiying Zhong, Zhun Deng, Weijie J. Su, Zhiwei Steven Wu, and Linjun Zhang. Provable multiparty reinforcement learning with diverse human feedback. arXiv preprint arXiv:2403.05006, 2024.

Fan Zhou, Jianing Wang, and Xingdong Feng. Non-crossing quantile regression for distributional reinforcement learning. Advances in neural information processing systems, 33:15909–15919, 2020.

Wenzhuo Zhou, Ruoqing Zhu, and Annie Qu. Estimating optimal infinite horizon dynamic treatment regimes via pt-learning. Journal of the American Statistical Association, 119(545):625–638, 2024.

## Appendices

## A Practical considerations

As commented in the main paper, the GRPO-type algorithm analyzed in Section 4 difers from the implementation proposed in the original DeepSeekMath paper. To bridge this gap, we present a more consistent formulation in Algorithm 2. There are three major diferences compared to Algorithm 1: (i) reward normalization, (ii) importance sampling (IS), and (iii) KL penalty. Specifically:

(i) Reward normalization. According to (15), the advantage function $A ^ { ( b , g ) }$ is not simply the diference between the reward $Z ^ { ( b , g ) }$ and the baseline $\bar { Z } ^ { ( b , - g ) }$ . Instead, this diference is normalized by the standard error of the within-group rewards (see (16)). Consequently, the algorithm assigns larger weight to prompts whose rewards are more uncertain. Intuitively, when a prompt is either too easy or too dificult, most responses receive a score of zero or achieve the maximum score, and the normalized advantage is small. In contrast, when the level of dificulty lies in the middle — so that some responses are correct while others are wrong — the prompt receives large weight.

(ii) Importance sampling. The practical implementation includes a minibatch parameter. Whenever the minibatch size is smaller than the full batch size B, the sampled prompts are divided into m minibatches, and the parameter is updated multiple times within each iteration (see Lines 8–10 of Algorithm 2). This is intended to improve learning eficiency: smaller minibatches lead to more frequent parameter updates, although each update is based on less data and is therefore noisier. Since the outputs are generated under the initial parameter $\theta _ { \mathrm { o l d } }$ (see Line 4), whereas the gradient is evaluated at the current parameter θ, a distributional shift arises. To correct for this mismatch, the advantage function is multiplied by an IS ratio $\mu _ { t } ^ { ( b , g ) } = \pi _ { \theta , t } ^ { ( b , g ) } / \pi _ { \theta _ { \mathrm { o l d } } , t } ^ { ( b , g ) }$ in (14). Ideally, one would instead use the sequential IS ratio up to time $t , \Pi _ { k = 1 } ^ { t } \mu _ { k } ^ { ( b , g ) }$ , which yields an unbiased correction. However, it is well known that such sequential IS ratio sufers from the curse of horizon [Liu et al., 2018], in the sense that their variance grows exponentially fast with t. As such, the practical algorithm uses only the token-level ratio, which substantially reduces variance at the cost of introducing bias, as we show formally in Theorem 12. The use of the token-level ratio is also in line with the PPO algorithm.

(iii) KL penalty. The second and third terms inside the square brackets in (14) arise from imposing a KL penalty

$$
\kappa \mathrm{KL} (\pi_ {\theta} \| \pi_ {\theta_ {\mathrm{ref}}}) = \kappa \mathbb {E} _ {X \sim f, Y \sim \pi_ {\theta} (\bullet | X)} \log \frac {\pi_ {\theta} (Y | X)}{\pi_ {\theta_ {\mathrm{ref}}} (Y | X)}
$$

on the objective function, for some regularization parameter $\kappa > 0$ . The purpose is to discourage the learning policy to departure too far away from a reference policy, an idea that shares similar spirits with the pessimistic principle in ofline RL. In particular, GRPO uses Schulman’s K3 estimator<sup>3</sup>

$$
\frac {1}{G} \sum_ {g = 1} ^ {G} \left[ \frac {\pi_ {\theta_ {\mathrm{ref}}} (Y ^ {(b , g)} | X)}{\pi_ {\theta} (Y ^ {(b , g)} | X)} - \log \frac {\pi_ {\theta_ {\mathrm{ref}}} (Y ^ {(b , g)} | X)}{\pi_ {\theta} (Y ^ {(b , g)} | X)} - 1 \right]\tag{13}
$$

for the KL divergence, which leads to the gradient estimator $\begin{array} { r } { G ^ { - 1 } \sum _ { t } \nabla _ { \theta } \log \pi _ { \theta , t } ^ { ( b , g ) } ( 1 - \pi _ { \theta , t } ^ { ( b , g ) } / \pi _ { \theta _ { \mathrm { r e f } } , t } ^ { ( b , g ) } ) } \end{array}$ in (14). When $Y ^ { ( g ) }$ is sampled from $\pi _ { \boldsymbol { \theta } } ( \bullet | X )$ , it is immediate that the first term inside the square brackets of (13) has mean one, so the expectation of (13) reduces to the second term alone and is thus unbiased for the KL divergence. The advantage of using the K3 estimator over solely the second term (known as the K1 estimator) is that K3 achieves lower variance and is guaranteed to be non-negative, both of which help stabilize training [Shao et al., 2024].

We begin by analyzing the gradient estimator ${ \widehat { g } } ( x ; \theta )$ in (14). The following lemma shows that, due to reward normalization, this gradient estimator is not exactly a U-statistic – as the kernel function depends on $\mathrm { s e } ( Z ^ { ( b , \bullet ) } )$ , which is data-dependent – but admits a U-statistic representation asymptotically as $G  \infty$

Assumption 10 (Coverage). $\pi _ { \theta }$ is lower bounded by ϵ for some $\epsilon > 0$ and any $\theta \in \Theta$

Conditions similar to 10 are frequently imposed in ofline RL [e.g., Uehara et al., 2022].

Lemma 11 (Gradient estimator asymptotically equivalent to a U-statistic). Suppose Assumptions 1, 3 and 10 hold. Suppose $s t d _ { X } ( Z ) : = \sqrt { { \cal V } a r _ { Z \sim \pi _ { \theta _ { o l d } } ( \bullet | X ) } } \big ( Z | X \big )$ is bounded away from zero. Then ${ \widehat { g } } ( x ; \theta )$ converges in probability to a second-order U-statistic

$$
\widehat {g} (x; \theta) = \binom {G} {2} ^ {- 1} \sum_ {1 \leq i <   j \leq G} h \big ((Y ^ {(i)}, Z ^ {(i)}), (Y ^ {(j)}, Z ^ {(j)}) \big),
$$

as $G  \infty$ , with symmetric kernel

$$
\begin{array}{l} h \big ((Y ^ {(i)}, Z ^ {(i)}), (Y ^ {(j)}, Z ^ {(j)}) \big) \\ = \frac {1}{2} \Big [ \sum_ {t} \mu_ {t} ^ {(i)} W _ {t} ^ {(i)} - \sum_ {t} \mu_ {t} ^ {(j)} W _ {t} ^ {(j)} \Big ] \frac {Z ^ {(i)} - Z ^ {(j)}}{\mathrm{std} _ {x} (Z)} + \frac {\kappa}{2} \Big [ \sum_ {t} (\mu_ {t} ^ {(i)} - 1) W _ {t} ^ {(i)} + \sum_ {t} (\mu_ {t} ^ {(j)} - 1) W _ {t} ^ {(j)} \Big ], \end{array}
$$

where $W _ { t } ^ { ( g ) } = \nabla _ { \theta }$ <sub>θ</sub> log $\pi _ { \boldsymbol { \theta } } ( Y _ { t } ^ { ( g ) } | x , Y _ { < t } ^ { ( g ) } )$ and the superscript (b) on Y and Z is removed for simplicity.

We next analyze the MSE of ${ \widehat { g } } ( x , \theta )$ , defined as $\mathrm { M S E } ( \widehat { g } ( x , \theta ) ) = \mathbb { E } [ \| \widehat { g } ( x ; \theta ) - g ^ { \dagger } ( x ; \theta ) \| ^ { 2 } ]$ where

$$
g ^ {\dagger} (x; \theta) = \mathbb {E} \Big [ \mu^ {(g)} \nabla_ {\theta} \log \pi_ {\theta} (Y ^ {(g)} | x) \frac {Z ^ {(g)} - \bar {Z} ^ {(- g)}}{\mathrm{std} _ {x} (Z)} + \kappa \mu^ {(g)} \sum_ {t} (\mu_ {t} ^ {(g)} - 1) W _ {t} ^ {(g)} \mid X = x \Big ],
$$

and $\mu ^ { ( g ) }$ denotes the sequential IS (SIS) ratio $\textstyle \prod _ { t } \mu _ { t } ^ { ( g ) }$ . This ratio appears in the ground-truth gradient to account for the distributional shift: the gradient is evaluated under the current policy π<sub>θ</sub>, whereas the outcomes were generated under $\pi _ { \theta _ { \mathrm { o l d } } }$ . As discussed earlier, the practical implementation does not use the SIS ratio for gradient estimation due to its extraordinarily large variance; instead, only the token-level IS ratio $\bar { \mu _ { t } ^ { ( b , g ) } }$ is used to multiply the score and the advantage function. Meanwhile, the KL divergence term does not include any IS ratio and is therefore biased as well. Thus, bias arises from both sources. In Theorem 12 below, we characterize the MSE of the gradient estimator, upper bounding both its squared bias and variance.

Theorem 12 (MSE upper bounds). Under Assumptions 1, 3, 10 and the same conditions in lemma 11, we have

$$
M S E (\widehat {g} (x; \theta)) = O \Big (\frac {m ^ {2} \eta_ {i + 1} ^ {2}}{\epsilon} \Big) + O \Big (\frac {\kappa^ {2} m ^ {2} \eta_ {i + 1} ^ {2}}{\epsilon^ {3}} \Big) + \frac {t r a c e [ \Sigma_ {1} (x ; \theta) ]}{G} + O \Big (\frac {1}{\epsilon G ^ {2}} \Big),\tag{17}
$$

where

$$
\Sigma_ {1} (x; \theta) = C o v \Big [ \sum_ {t} \mu_ {t} ^ {(g)} W _ {t} ^ {(g)} \frac {Z ^ {(g)} - V ^ {\pi_ {\theta_ {o l d}}} (x)}{s t d _ {x} (Z)} + \kappa \sum_ {t} (\mu_ {t} ^ {(g)} - 1) W _ {t} ^ {(g)} | X = x \Big ].\tag{18}
$$

Theorem 12 gives a bias-variance decomposition regarding the gradient estimator’s MSE:

1. The first two terms on the right hand side (RHS) of (17) are bias terms that upper bound the squared biases arising from the two sources mentioned above, respectively. Notably, these bias terms are proportional to the learning rate $\eta _ { i + 1 }$ and vanish as $\eta _ { i + 1 }$ approaches zero.

2. The last two terms are variance terms. In particular, trace $[ \Sigma _ { 1 } ( x ; \theta ) ] / G$ is the leading term, which, by its definition in (18), corresponds to the variance of an oracle estimator that knows the value function $V ^ { \pi \theta _ { \mathrm { o l d } } } \left( x \right)$ a priori.

3. The last term is a higher-order variance term that measures the error of the U-statistic in approximating the oracle estimator, and is of the same order $O ( G ^ { - 2 } )$ as in Theorem 2. As G grows to infinity, this term decays to zero at a faster rate, so that the estimator achieves the same asymptotic variance as the oracle estimator.

Finally, we establish the consistency of the estimated policy parameter $\widehat { \theta } _ { n }$

Proposition 13. Suppose the reward signal Z is binary and the value function $V ^ { \pi _ { \theta } }$ is bounded away from 0 and 1. Then under Assumptions $1 , \ 3 , \ 4 , \ 5 ( b )$ and 6, we have

$$
d (\widehat {\theta} _ {n}, \widetilde {\Theta} ^ {*}) \xrightarrow {P} 0,
$$

where $\widetilde { \Theta } ^ { * } = \arg \operatorname* { m a x } _ { \theta } \mathcal { I } ( \theta )$ and

$$
\mathcal {J} (\theta) = \mathbb {E} \left[ 2 \arcsin \left(\sqrt {V ^ {\pi_ {\theta}} (X)}\right) \right] - \kappa K L \left(\pi_ {\theta} \| \pi_ {\theta_ {r e f}}\right).\tag{19}
$$

We make a few remarks. First, we adopt the same asymptotic framework as in Section 4 for establishing consistency, by allowing multiple optimizers to exist and by letting $n \to \infty$ while fixing all other parameters such as $B , G , \kappa ,$ m and ϵ. Second, consider the case where $\kappa = 0$ , so that the KL penalty is not imposed. Interestingly, according to (19), the policy parameter does not directly maximize the value function but rather an arcsin transformation of it. To see why, note that when Z is binary, std $_ x ( Z ) = \sqrt { V ^ { \pi _ { \theta _ { \mathrm { o l d } } } } ( x ) [ 1 - V ^ { \pi _ { \theta _ { \mathrm { o l d } } } } ( x ) ] }$ . Under the stated conditions, we can show that the ground-truth gradient

$$
g ^ {\dagger} (\theta) = \mathbb {E} [ g ^ {\dagger} (X; \theta) ] = \mathbb {E} ^ {\pi_ {\theta}} \Big [ W _ {t} ^ {(g)} \frac {Z}{\mathrm{std} _ {x} (Z)} \Big ] \to \mathbb {E} ^ {\pi_ {\theta}} \Big [ \nabla_ {\theta} \log \pi_ {\theta} (Y _ {t} ^ {(g)} | X, Y _ {<   t} ^ {(g)}) \frac {\sqrt {V ^ {\pi_ {\theta}} (X)}}{\sqrt {1 - V ^ {\pi_ {\theta_ {\mathrm{old}}}} (X)}} \Big ],
$$

which is precisely the derivative of ${ \mathcal { I } } ( \theta )$ in (19). This arises from reward normalization. This relationship was identified by Davis and Recht [2025], who did not consider the use of minibatches or KL penalties. Nor did they establish parameter consistency. We formally verify their result by establishing parameter consistency under more practical settings.

To conclude this section, we remark that although we aim to match practical settings as closely as possible, two gaps remain: (i) the original GRPO uses length normalization, which further divides each advantage function $\overrightharpoon { A } ^ { ( b , g ) }$ by the length of $Y ^ { ( g ) }$ , although this is not used in later variants such as Dr. GRPO and GPG; (ii) simple stochastic gradient descent algorithm is no longer used in practice, and has been replaced by more sophisticated optimizers such as Muon [Jordan et al., 2024]. We leave these gaps for future research.

## B Proofs

## B.1 Group relative gradient evaluation

In this section, we provide the proofs of Lemma 1, Theorem 2, Proposition 3, and Corollaries 4 and 5 stated in Section 4.1.

Proof of Lemma 1: To ease notation, we denote $W ^ { ( g ) } = \nabla _ { \theta } \log \pi _ { \theta } ( { \cal Y } ^ { ( g ) } | x )$ . As discussed below Lemma 1, by setting $C ^ { ( g ) }$ in (5) to the leave-one-out group mean baseline, each individual term in the sum satisfies

$$
W ^ {(g)} [ Z ^ {(g)} - \bar {Z} ^ {(- g)} ] = \frac {1}{G - 1} \sum_ {k \neq g} W ^ {(g)} [ Z ^ {(g)} - Z ^ {(k)} ].
$$

By definition, we obtain

$$
\widehat {g} _ {\mathrm{GRPO}} (x; \theta) = \frac {1}{G (G - 1)} \sum_ {k \neq g} W ^ {(g)} [ Z ^ {(g)} - Z ^ {(k)} ].
$$

Applying a standard symmetrization argument, it follows that

$$
\begin{array}{r c l} \widehat {g} _ {\mathrm{GRPO}} (x; \theta) & = & \frac {1}{2 G (G - 1)} \sum_ {k \neq g} W ^ {(g)} [ Z ^ {(g)} - Z ^ {(k)} ] + \frac {1}{2 G (G - 1)} \sum_ {k \neq g} W ^ {(k)} [ Z ^ {(k)} - Z ^ {(g)} ] \\ & = & \frac {1}{2 G (G - 1)} \sum_ {k \neq g} [ W ^ {(g)} Z ^ {(g)} - W ^ {(g)} Z ^ {(k)} - W ^ {(k)} Z ^ {(g)} + W ^ {(k)} Z ^ {(k)} ] \\ & = & \frac {1}{2 G (G - 1)} \sum_ {k \neq g} [ (W ^ {(g)} - W ^ {(k)}) (Z ^ {(g)} - Z ^ {(k)}) ] \\ & = & \frac {1}{2 G (G - 1)} \sum_ {k \neq g} h ((Y ^ {(k)}, Z ^ {(k)}), (Y ^ {(g)}, Z ^ {(g)})). \end{array}
$$

The conclusion of Lemma 1 thus follows.

Proof of Theorem 2: Recall that the Hoefding decomposition (2) expresses a U-statistic as the sum of the expectation of the kernel $h _ { 0 }$ , a first-order component,

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \zeta_ {1} (X _ {i}) = \frac {2}{n} \sum_ {i = 1} ^ {n} \bigl [ h _ {1} (X _ {i}) - h _ {0} \bigr ],
$$

and a second-order component

$$
\frac {1}{n (n - 1)} \sum_ {i \neq j} \zeta_ {2} (X _ {i}, X _ {j}) = \frac {1}{n (n - 1)} \sum_ {i \neq j} \left[ h (X _ {i}, X _ {j}) - h _ {1} (X _ {i}) - h _ {1} (X _ {j}) + h _ {0} \right].
$$

With some calculations, it is straightforward to show that

(i) Each of $\zeta _ { 1 } ( X _ { i } )$ and $\zeta _ { 2 } ( X _ { i } , X _ { j } )$ has mean zero and is therefore orthogonal (i.e., uncorrelated) to $h _ { 0 } ;$

(ii) The first-order term $\zeta _ { 1 } ( X _ { i } )$ is orthogonal to the second-order term $\zeta _ { 2 } ( X _ { k } , X _ { g } )$ for any $i , k , g ;$

(iii) Diferent first-order and second-order terms are mutually orthogonal, i.e., $\zeta _ { 1 } ( X _ { i } ) \perp \zeta _ { 1 } ( X _ { j } )$ for $i \neq j$ , and $\zeta _ { 2 } ( X _ { i } , X _ { j } ) \perp \zeta _ { 2 } ( X _ { k } , X _ { g } )$ whenever $\{ i , j \} \neq \{ k , g \}$

These orthogonality relations lead to the following MSE decomposition for the U-statistic:

$$
\begin{array}{r c l} \mathrm{MSE} (U) & = & \frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} \mathrm{trace} \big [ \mathrm{Var} \big (\zeta_ {1} (X _ {i}) \big) \big ] + \frac {4}{n ^ {2} (n - 1) ^ {2}} \sum_ {1 \leq i <   j \leq n} \mathrm{trace} \big [ \mathrm{Var} \big (\zeta_ {2} (X _ {i}, X _ {j}) \big) \big ] \\ & = & \frac {1}{n} \mathrm{trace} \big [ \mathrm{Var} \big (\zeta_ {1} (X _ {1}) \big) \big ] + \frac {2}{n (n - 1)} \mathrm{trace} \big [ \mathrm{Var} \big (\zeta_ {2} (X _ {1}, X _ {2}) \big) \big ]. \end{array}
$$

Next, by Jensen’s inequality,

$$
\begin{array}{r l} & {\mathrm{trace} \big [ \mathrm{Var} \big (\zeta_ {2} (X _ {1}, X _ {2}) \big) \big ] = \mathbb {E} \big \| h (X _ {1}, X _ {2}) - \mathbb {E} [ h (X _ {1}, X _ {2}) \mid X _ {1} ] - \mathbb {E} [ h (X _ {2}, X _ {1}) \mid X _ {2} ] + \mathbb {E} h (X _ {1}, X _ {2}) \big \| ^ {2}} \\ & {\qquad \leq 4 \mathbb {E} \| h (X _ {1}, X _ {2}) \| ^ {2} + 4 \mathbb {E} \big \| \mathbb {E} [ h (X _ {1}, X _ {2}) \mid X _ {1} ] \big \| ^ {2}} \\ & {\qquad + 4 \mathbb {E} \big \| \mathbb {E} [ h (X _ {2}, X _ {1}) \mid X _ {2} ] \big \| ^ {2} + 4 \left\| \mathbb {E} h (X _ {1}, X _ {2}) \right\| ^ {2}} \\ & {\qquad \leq 1 6 \mathbb {E} \| h (X _ {1}, X _ {2}) \| ^ {2}.} \end{array}
$$

It follows that

$$
\operatorname{MSE} (U) = \frac {1}{n} \operatorname{trace} \left[ \operatorname{Var} \left(\zeta_ {1} \left(X _ {1}\right)\right) \right] + \frac {O \left(\mathbb {E} \| h \left(X _ {1} , X _ {2}\right) \| ^ {2}\right)}{n (n - 1)}.\tag{20}
$$

By setting U to the GRPO gradient estimator ${ \widehat { g } } _ { \mathrm { G R P O } } ( \theta ; x )$ , it is immediate to see that the first-order component becomes the oracle gradient estimator so that

$$
\operatorname{trace} \left[ \operatorname{Var} \left(\zeta_ {1} \left(X _ {1}\right)\right) \right] = \operatorname{trace} \left[ \Sigma_ {\text { oracle }} (x; \theta) \right].\tag{21}
$$

Additionally, under the boundedness assumption 1, we obtain

$$
\mathbb {E} \| W ^ {(1)} - W ^ {(2)} \| ^ {2} (Z ^ {(1)} - Z ^ {(2)}) ^ {2} = O \big (\mathbb {E} \| W ^ {(1)} - W ^ {(2)} \| ^ {2} \big) = O \big (\mathbb {E} \| W \| ^ {2} \big),\tag{22}
$$

where the last equality follows from the fact that for any random vectors X and Y , $\mathbb { E } \| X + Y \| ^ { 2 } \leq$ $2 ( \mathbb { E } \| X \| ^ { 2 } + \mathbb { E } \| Y \| ^ { 2 } )$ . Combining (22) with (20) and (21), we have

$$
\operatorname{MSE} \left(\widehat {g} _ {\text { GRPO }} (x; \theta)\right) = \frac {1}{G} \operatorname{trace} \left[ \Sigma_ {\text { oracle }} (x; \theta) \right] + O \left(\frac {\mathbb {E} \| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2}}{G ^ {2}}\right).
$$

This completes the proof of Theorem 2.

Proof of Proposition 3. By definition, $\operatorname { M S E } ( { \widehat { g } } _ { \mathrm { G R P O } } ( \theta ) )$ equals

$$
\begin{array}{r l} & {\mathbb {E} \| \widehat {g} _ {\mathrm{GRPO}} (\theta) - g (\theta) \| ^ {2}} \\ {=} & {\mathbb {E} \left\| \frac {1}{B} \sum_ {b = 1} ^ {B} [ \widehat {g} _ {\mathrm{GRPO}} (X ^ {(b)}; \theta) - g (X ^ {(b)}; \theta) ] + \frac {1}{B} \sum_ {b = 1} ^ {B} [ g (X ^ {(b)}; \theta) - g (\theta) ] \right\| ^ {2}.} \end{array}
$$

Since $g ( X ^ { ( b ) } ; \theta )$ equals the conditional mean of $\widehat { g } _ { \mathrm { G R P O } } ( X ^ { ( b ) } ; \theta )$ given $X ^ { ( b ) } , { \widehat { g } } _ { \mathrm { G R P O } } ( X ^ { ( b ) } ; \theta ) - g ( X ^ { ( b ) } ; \theta )$ is orthogonal to $g ( X ^ { ( b ) } ; \theta ) - g ( \theta )$ . It follows that

$$
\begin{array}{r l} & {\mathrm{MSE} (\widehat {g} _ {\mathrm{GRPO}} (\theta)) = \mathbb {E} \left\| \frac {1}{B} \sum_ {b = 1} ^ {B} [ \widehat {g} _ {\mathrm{GRPO}} (X ^ {(b)}; \theta) - g (X ^ {(b)}; \theta) ] \right\| ^ {2} + \mathbb {E} \left\| \frac {1}{B} \sum_ {b = 1} ^ {B} [ g (X ^ {(b)}; \theta) - g (\theta) ] \right\| ^ {2}} \\ & {\qquad = \frac {1}{B} \mathbb {E} \left\| \widehat {g} _ {\mathrm{GRPO}} (X; \theta) - g (X; \theta) \right\| ^ {2} + \frac {1}{B} \mathbb {E} \left\| g (X; \theta) - g (\theta) \right\| ^ {2}.} \end{array}
$$

Applying the conclusion of Theorem 2, we obtain

$$
\begin{array}{r c l} \mathbb {E} \| \widehat {g} _ {\mathrm{GRPO}} (\theta) - g (\theta) \| ^ {2} & = & \frac {\mathbb {E} \left\| g (X ; \theta) - g (\theta) \right\| ^ {2}}{B} \\ & & + \frac {\mathrm{trace} [ \Sigma_ {\mathrm{oracle}} (\theta) ]}{B G} + O \left(\frac {\mathbb {E} \| \nabla_ {\theta} \log \pi_ {\theta} (Y | X) \| ^ {2}}{B G ^ {2}}\right). \end{array}
$$

This completes the proof of Proposition 3.

Proof of Corollary $\it 4 .$ The conclusion directly follows from Theorem $2$ and Proposition 3 by letting $G  \infty$ □

Proof of Corollary 5. Any gradient estimator ${ \widehat { g } } ( x ; \theta )$ whose baseline term $b ( x )$ is a function of the prompt x only is an unbiased estimator of $g ( x ; \theta )$ . Therefore, its MSE can be represented as:

$$
\mathrm{MSE} (\widehat {g} (\theta ; x)) = \mathrm{trace} \big [ \mathrm{Var} (\widehat {g} (\theta ; x)) \big ] = \mathbb {E} \| \widehat {g} (\theta ; x) \| ^ {2} - \| g (\theta ; x) \| ^ {2}.\tag{23}
$$

The first term on the RHS can be further decomposed as

$$
\begin{array}{l} \mathbb {E} \| \widehat {g} (\theta ; x) \| ^ {2} = \frac {1}{G} \mathbb {E} \left\{\| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2} [ Z - V ^ {\pi_ {\theta}} (x) + V ^ {\pi_ {\theta}} (x) - b (x) ] ^ {2} \right\} \\ \qquad = \frac {1}{G} \mathbb {E} \left\{\| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2} [ Z - V ^ {\pi_ {\theta}} (x) ] ^ {2} \right\} \\ \qquad + \frac {1}{G} \mathbb {E} \left\{\| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2} [ V ^ {\pi_ {\theta}} (x) - b (x) ] ^ {2} \right\} \\ \qquad + \frac {1}{G} \mathbb {E} \left\{\| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2} [ V ^ {\pi_ {\theta}} (x) - b (x) ] [ Z - V ^ {\pi_ {\theta}} (x) ] \right\}. \end{array}\tag{24}
$$

Notice that the second line of (24) equals trace $[ \Sigma _ { \mathrm { o r a c l e } } ( x ; \theta ) ]$ . Under Assumption 2 that $Z$ and $\| \nabla _ { \theta } \log \pi _ { \theta } ( Y | x ) \|$ are conditionally independent given $x ,$ the interaction term (i.e., the last line of (24)) equals zero. This together with (23) and (24) yields that

$$
\begin{array}{r l} & {\mathrm{MSE} (\widehat {g} (x; \theta)) = \frac {1}{G} \mathrm{trace} [ \Sigma_ {\mathrm{oracle}} (x; \theta) ] + \frac {1}{G} \mathbb {E} \left\{\| \nabla_ {\theta} \log \pi_ {\theta} (Y | x) \| ^ {2} [ V ^ {\pi_ {\theta}} (x) - b (x) ] ^ {2} \right\}} \\ & {\qquad \geq \frac {1}{G} \mathrm{trace} [ \Sigma_ {\mathrm{oracle}} (x; \theta) ].} \end{array}\tag{25}
$$

Since Corollary 4 implies $\mathrm { M S E } _ { A } ( \widehat { g } _ { \mathrm { G R P O } } ( x ; \theta ) ) = G ^ { - 1 } \mathrm { t r a c e } [ \Sigma _ { \mathrm { o r a c l e } } ( x ; \theta ) ]$ , we obtain

$$
\operatorname{MSE} _ {A} (\widehat {g} _ {\mathrm{GRPO}} (x; \theta)) \leq \operatorname{MSE} (\widehat {g} (x; \theta)).
$$

Moreover, when the baseline term $b ( x ) \equiv 0$ , the above inequality becomes strict given that $V ^ { \pi _ { \theta } } ( x ) \neq$ 0 and $\nabla _ { \theta } \log \pi _ { \theta } ( Y | x )$ is not almost surely zero.

Next, consider $\begin{array} { r } { \widehat { g } ( \theta ) = \sum _ { b } \widehat { g } ( X ^ { ( b ) } ; \theta ) / B } \end{array}$ in the minibatch setting. Following a similar argument to the proof of Proposition 3, we obtain

$$
\mathbb {E} \| \widehat {g} (\theta) - g (\theta) \| ^ {2} = \frac {\mathbb {E} \| g (X ; \theta) - g (\theta) \| ^ {2}}{B} + \frac {1}{B} \mathbb {E} \| \widehat {g} (X; \theta) - g (X; \theta) \| ^ {2}.
$$

Applying inequality (25), we obtain

$$
\mathrm{MSE} (\widehat {g} (\theta)) \geq \frac {\mathbb {E} \left\| g (X ; \theta) - g (\theta) \right\| ^ {2}}{B} + \frac {1}{B G} \mathrm{trace} [ \Sigma_ {\mathrm{oracle}} (\theta) ] = \mathrm{MSE} _ {A} (\widehat {g} _ {\mathrm{GRPO}} (\theta)).
$$

For a zero baseline function, the above inequality becomes strict given that $V ^ { \pi _ { \theta } } ( X )$ and $\nabla _ { \theta } \log \pi _ { \theta } ( Y | X )$ are not almost surely zero. This completes the proof of Corollary 5. □

## B.2 Group relative policy optimization

This section presents the proofs of Lemma 6, Theorems 7 and 8, and Corollaries 9 and 10.

Proof of Lemma 6. Let $J ( \theta ) = \mathbb { E } ^ { \pi _ { \theta } } ( Z )$ denote our objective function defined on $\theta \in \Theta$ . Its gradient is given by $\nabla _ { \theta } J ( \theta ) = g ( \theta )$ Recall that the parameter is updated according to $\theta _ { n + 1 } = \theta _ { n } + \eta _ { n } \widehat { g } ( \theta _ { n } )$ Under the L-smoothness condition (Assumption 3), we have

$$
J (\theta_ {n + 1}) = J (\theta_ {n} + \eta_ {n} \widehat {g} (\theta_ {n})) \geq J (\theta_ {n}) + \eta_ {n} g ^ {\top} (\theta_ {n}) \widehat {g} (\theta_ {n}) - \frac {1}{2} L \eta_ {n} ^ {2} \| \widehat {g} (\theta_ {n}) \| ^ {2}.
$$

Notice that ${ \widehat { g } } ( \theta )$ is unbiased to $g ( \theta )$ . Conditioned on $\theta _ { n }$ and take conditional expectation on both sides, we obtain

$$
\mathbb {E} [ J (\theta_ {n + 1}) | \theta_ {n} ] \geq J (\theta_ {n}) + \eta_ {n} \| g (\theta_ {n}) \| ^ {2} - \frac {1}{2} L \eta_ {n} ^ {2} \mathbb {E} \big [ \| \widehat {g} (\theta_ {n}) \| ^ {2} | \theta_ {n} \big ].\tag{26}
$$

The last term on the RHS can be represented by $- L \eta _ { n } ^ { 2 } \big [ \mathrm { M S E } ( \widehat { g } ( \theta _ { n } ) ) + \| g ( \theta _ { n } ) \| ^ { 2 } \big ] / 2$ . Let $\theta ^ { * }$ be a maximizer of $J ( \theta )$ . Then, rearranging the terms in (26) yields:

$$
\begin{array}{r c l} J (\theta^ {*}) - \mathbb {E} [ J (\theta_ {n + 1}) | \theta_ {n} ] & \leq & J (\theta^ {*}) - J (\theta_ {n}) - \eta_ {n} \| g (\theta_ {n}) \| ^ {2} \\ & & + \frac {1}{2} L \eta_ {n} ^ {2} \| g (\theta_ {n}) \| ^ {2} + \frac {1}{2} L \eta_ {n} ^ {2} \mathrm{MSE} (\widehat {g} _ {n} (\theta_ {n})). \end{array}
$$

It follows that

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n + 1}}) | \theta_ {n} ] \leq \Delta (\pi_ {\theta_ {n}}) - \left(\eta_ {n} - \frac {L \eta_ {n} ^ {2}}{2}\right) \| g (\theta_ {n}) \| ^ {2} + \frac {L \eta_ {n} ^ {2}}{2} \mathrm{MSE} (\widehat {g} (\theta_ {n}))
$$

Under the PL condition (Assumption 4), we obtain

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n + 1}}) | \theta_ {n} ] \leq \left(1 - 2 \mu \eta_ {n} + \mu L \eta_ {n} ^ {2}\right) \Delta (\pi_ {\theta_ {n}}) + \frac {1}{2} L \eta_ {n} ^ {2} M,\tag{27}
$$

where M denotes the uniform upper bound of the MSEs.

Based on this inequality, we derive suboptimality gap bounds for the two learning-rate schedules separately, as follows.

(i) The constant schedule. Under the constant schedule where $\eta _ { i } = \beta ,$ , (27) becomes

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n + 1}}) | \theta_ {n} ] \leq (1 - 2 \mu \beta + \mu L \beta^ {2}) \Delta (\pi_ {\theta_ {n}}) + \frac {1}{2} L \beta^ {2} M.\tag{28}
$$

By Lemma 16 in Section B.4, we have $\mu \leq L$ . Consider the quadratic equation $1 - 2 \mu \beta + \mu L \beta ^ { 2 } = 0$ with $\beta$ treated as the variable. Its discriminant equals $\mu ^ { 2 } - 4 \mu L \leq 0$ so that the equation has at most one real solution. As such, we have $\rho : = 1 - 2 \mu \beta + \mu L \beta ^ { 2 } \geq 0$ . Additionally, since $\beta < ( 2 L ) ^ { - 1 }$ (Assumption $\mathrm { 5 ( a ) } )$ ), ρ is strictly smaller than 1.

Taking expectations on both sides of (28) with respect to $\theta _ { n }$ and unrolling the recursion yields

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n + 1}}) ] \leq \rho^ {n} \mathbb {E} [ \Delta (\pi_ {\theta_ {0}}) ] + \frac {1}{2} L \beta^ {2} M \sum_ {k = 1} ^ {n} \rho^ {k - 1} = \rho^ {n} \mathbb {E} [ \Delta (\pi_ {\theta_ {0}}) ] + \frac {L \beta^ {2} M}{2 (1 - \rho)}.
$$

This completes the proof under the constant schedule.

(ii) The $1 / i$ schedule. When $\eta _ { i } = \beta / i$ for some constant $\beta ,$ (27) becomes

$$
\mathbb {E} \left[ \Delta \left(\pi_ {\theta_ {n + 1}}\right) \mid \theta_ {n} \right] \leq \left(1 - \frac {2 \mu \beta}{n} + \frac {\mu L \beta^ {2}}{n ^ {2}}\right) \Delta \left(\pi_ {\theta_ {n}}\right) + \frac {L \beta^ {2} M}{2 n ^ {2}}.
$$

Taking expectation on both sides yields

$$
\mathbb {E} \left[ \Delta \left(\pi_ {\theta_ {n + 1}}\right) \right] \leq \left(1 - \frac {2 \mu \beta}{n} + \frac {\mu L \beta^ {2}}{n ^ {2}}\right) \mathbb {E} \left[ \Delta \left(\pi_ {\theta_ {n}}\right) \right] + \frac {L \beta^ {2} M}{2 n ^ {2}}.
$$

To this end, we apply Lemma 14 (Section B.4) by letting $a _ { n } = \mathbb { E } [ \Delta ( \pi _ { \theta _ { n } } ) ] , A = 2 \mu \beta , B = \mu L \beta ^ { 2 }$ and $C = { \textstyle { \frac { 1 } { 2 } } } L \beta ^ { 2 } M$ . Since $\beta > ( 2 \mu ) ^ { - 1 }$ , we have $A > 1$ . Notice that the sequence $a _ { n }$ is bounded under Assumption 1. Therefore, lemma 14 implies that for any constant $\varepsilon > 0$ ,

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n}}) ] \leq \max \left\{\frac {(1 + \varepsilon) L \beta^ {2} M}{(4 \mu \beta - 2) n}, \frac {c}{n} \right\}
$$

holds for all $n \geq 1$ , where c is a constant only depends on $\mu , \beta , L$ and ε. This finishes the proof of Lemma 6. □

Proof of Theorem 7. According to Lemma 6, the sub-optimality upper bounds depend on $\operatorname* { s u p } _ { \theta } \operatorname { M S E } ( \widehat { g } ( \theta ) ) / n$ By Proposition 3, we know that for GRPO-type algorithm,

$$
\sup _ {\theta} \mathrm{MSE} (\widehat {g} (\theta)) \leq \frac {c _ {1}}{B} + \frac {c _ {2}}{B G} + \frac {c _ {3}}{B G ^ {2}},\tag{29}
$$

where $c _ { 1 } = \operatorname* { s u p } _ { \theta } \mathbb { E } \| g ( X ; \theta ) - g ( \theta ) \| _ { 2 } ^ { 2 } , c _ { 2 } = \operatorname* { s u p } _ { \theta }$ t ${ \mathrm { \cdot a c e } } [ \Sigma _ { \mathrm { o r a c l e } } ( \theta ) ]$ and $c _ { 3 } = O ( \operatorname* { s u p } _ { \theta } \mathbb { E } \| \nabla _ { \theta }$ log $\pi _ { \theta } ( Y | X ) \| ^ { 2 } )$

Under a fixed sampling budget $N = B G$ per iteration, the RHS becomes

$$
\frac {c _ {1} G}{N} + \frac {c _ {2}}{N} + \frac {c _ {3}}{N G} \geq \frac {c _ {2}}{N} + \frac {2}{N} \sqrt {c _ {1} c _ {3}},
$$

where the inequality follows from that $a ^ { 2 } + b ^ { 2 } \geq 2 a b .$ , and the equality holds if and only if $G =$ $\sqrt { c _ { 3 } / c _ { 1 } }$ . Similarly, under a fixed total sampling budget $\mathbf { N } = n B G$ , the MSE is upper bounded by

$$
\frac {c _ {1} n G}{\mathbf {N}} + \frac {c _ {2} n}{\mathbf {N}} + \frac {c _ {3} n}{\mathbf {N} G} \geq \frac {c _ {2} n}{\mathbf {N}} + \frac {2}{\mathbf {N}} \sqrt {c _ {1} c _ {3} n}.
$$

Again, the the equality holds if and only if $G = { \sqrt { c _ { 3 } / c _ { 1 } } }$

Proof of Theorem 8. We first prove consistency. For any $\theta _ { 1 } , \theta _ { 2 } \in \Theta$ , the distance function satisfies

$$
\left| d \left(\theta_ {1}, \Theta^ {*}\right) - d \left(\theta_ {2}, \Theta^ {*}\right) \right| \leq \| \theta_ {1} - \theta_ {2} \|,
$$

and is therefore continuous in θ. Hence, for any $\varepsilon > 0$ , the level set

$$
\mathcal {N} _ {\varepsilon} := \{\theta \in \Theta : d (\theta , \Theta^ {*}) \geq \varepsilon \}
$$

is closed. Under the compactness assumption of Θ in Assumption $6 , \mathcal { N } _ { \varepsilon }$ is compact as well. Combined with the continuity of the policy learning objective J (implied by Assumption 3), we obtain that the supremum $\operatorname* { s u p } _ { \theta \in { \mathcal { N } } _ { \varepsilon } } J ( \theta )$ is attainable at some $\theta _ { \varepsilon } \in \mathcal { N } _ { \varepsilon }$

Since $\theta _ { \varepsilon } \notin \Theta ^ { \ast }$ , we have $\delta _ { \varepsilon } : = J ^ { \ast } - J ( \theta _ { \varepsilon } ) > 0$ . Therefore,

$$
\mathbb {P} \big (d (\theta_ {n}, \Theta^ {*}) \geq \varepsilon \big) \leq \mathbb {P} \big (\Delta (\pi_ {\theta_ {n}}) \geq \delta_ {\varepsilon} \big).
$$

Applying Markov’s inequality and invoking Lemma $6 ,$ we obtain

$$
\mathbb {P} \big (d (\theta_ {n}, \Theta^ {*}) \geq \varepsilon \big) \leq \frac {\mathbb {E} \big [ \Delta (\pi_ {\theta_ {n}}) \big ]}{\delta_ {\varepsilon}} \leq \frac {\bar {c}}{n \delta_ {\varepsilon}} \to 0 \quad \text {as} n \to \infty ,
$$

where ¯c is a positive constant depending on $( \beta , \mu , L , M , \varepsilon )$ . Thus,

$$
d (\theta_ {n}, \Theta^ {*}) \xrightarrow {P} 0,
$$

which establishes the consistency of $\theta _ { n }$ .

Next, we derive the asymptotic distribution of $\Delta ( \pi _ { \theta _ { n } } )$ . The proof proceeds in four steps:

• Step (i): We strengthen consistency to almost sure convergence, i.e., $d ( \theta _ { n } , \Theta ^ { \ast } ) \stackrel { \mathrm { a . s . } } {  } 0$

• Step (ii): We derive the convergence rate of $\theta _ { n }$ by showing that $\mathbb { E } [ d ^ { 2 } ( \theta _ { n } , \Theta ^ { * } ) ] = O ( n ^ { - 1 } )$

• Step (iii): We establish the asymptotic normality of the identifiable part of $\theta _ { n } , Q ^ { \top } \theta _ { n }$

• Step (iv): We derive the asymptotic distribution of $\Delta ( \pi _ { \theta _ { n } } )$

Step (i): Almost sure convergence. To prove this result, recall the inequality established in equation (27):

$$
\mathbb {E} [ \Delta (\pi_ {\theta_ {n + 1}}) | \theta_ {n} ] \leq \left(1 - 2 \mu \eta_ {n} + \mu L \eta_ {n} ^ {2}\right) \Delta (\pi_ {\theta_ {n}}) + \frac {1}{2} L \eta_ {n} ^ {2} M,
$$

where $\eta _ { n } = \beta / n$ . We apply the Robbins-Siegmund theorem (see Lemma 17) by setting $V _ { n } = \Delta ( \pi _ { \theta _ { n } } )$ $A _ { n } = \mu L \eta _ { n } ^ { 2 } , B _ { n } = \textstyle { \frac { 1 } { 2 } } L \eta _ { n } ^ { 2 } M$ and $C _ { n } = 2 \mu \eta _ { n } V _ { n }$ . It is immediate to see that

$$
\sum_ {n = 1} ^ {\infty} A _ {n} = \sum_ {n = 1} ^ {\infty} \frac {\mu L \beta^ {2}}{n ^ {2}} <   \infty
$$

and

$$
\sum_ {n = 1} ^ {\infty} B _ {n} = \sum_ {n = 1} ^ {\infty} \frac {L M \beta^ {2}}{2 n ^ {2}} <   \infty .
$$

By the Robbins-Siegmund theorem, it follows that $V _ { n } = \Delta ( \pi _ { \theta _ { n } } )$ ) converges to some random variable $V _ { \infty }$ almost surely.

Additionally, by the suboptimality gap bound established in Lemma 6, an application of Fatou’s lemma yields

$$
\mathbb {E} V _ {\infty} \leq \operatorname * {l i m i n f} _ {n} \mathbb {E} \Delta (\pi_ {\theta_ {n}}) \leq \operatorname * {l i m i n f} _ {n} c ^ {*} / n = 0,
$$

for some constant $c ^ { * } > 0$ . Since $V _ { \infty }$ is non-negative, it follows that $V _ { \infty } = 0$ almost surely. Consequently, $\Delta ( \pi _ { \theta _ { n } } )  0$ almost surely. This implies for any $\delta > 0$

$$
\mathbb {P} \left(\{\Delta (\pi_ {\theta_ {n}}) \geq \delta , \mathrm{i.o.} \}\right) = 0,
$$

where $\{ A _ { n } , \ i . . 0 . \}$ represents the event that $A _ { n }$ occurs infinitely often. Using the same argument as in the proof of consistency of $\theta _ { n } .$ we obtain for any $\varepsilon > 0$ that,

$$
\mathbb {P} \left(\left\{d (\theta_ {n}, \Theta^ {*}) \geq \varepsilon , \text {   i.o. } \right\}\right) \leq \mathbb {P} \left(\left\{\Delta (\pi_ {\theta_ {n}}) \geq \delta_ {\varepsilon}, \text {   i.o. } \right\}\right) = 0.
$$

This establishes the almost sure convergence of $\theta _ { n }$ .

Step (ii): Convergence rate of $\theta _ { n } \mathfrak { i }$ : Recall that the projection operator $\Pi _ { \Theta ^ { * } }$ is defined as:

$$
\Pi_ {\Theta^ {*}} (\theta) := \arg \min _ {x \in \Theta^ {*}} d (\theta , x).
$$

Since $\Theta ^ { * }$ is closed and convex (Assumption 7), the projection operator is well-defined. It follows from its definition that

$$
d ^ {2} (\theta_ {n + 1}, \Theta^ {*}) \leq \| \theta_ {n + 1} - \Pi_ {\Theta^ {*}} (\theta_ {n}) \| ^ {2}.
$$

According to the update rule $\theta _ { n + 1 } = \theta _ { n } + \eta _ { n } \widehat { g } ( \theta _ { n } )$ , we have

$$
\begin{array}{r c l} {d ^ {2} (\theta_ {n + 1}, \Theta^ {*})} & \leq & {\| \theta_ {n} + \eta_ {n} \widehat {g} (\theta_ {n}) - \Pi_ {\Theta^ {*}} (\theta_ {n}) \| ^ {2}} \\ & = & {d ^ {2} (\theta_ {n}, \Theta^ {*}) + 2 \eta_ {n} \widehat {g} (\theta_ {n}) ^ {\top} (\theta_ {n} - \Pi_ {\Theta^ {*}} (\theta_ {n})) + \eta_ {n} ^ {2} \| \widehat {g} (\theta_ {n}) \| ^ {2}.} \end{array}
$$

Taking conditional expectation with respect to $\theta _ { n }$ on both sides yields

$$
\begin{array}{r l} & {\mathbb {E} [ d ^ {2} (\theta_ {n + 1}, \Theta^ {*}) | \theta_ {n} ] \leq d ^ {2} (\theta_ {n}, \Theta^ {*}) + 2 \eta_ {n} g ^ {\top} (\theta_ {n}) (\theta_ {n} - \Pi_ {\Theta^ {*}} (\theta_ {n})) + \eta_ {n} ^ {2} \mathbb {E} \big [ \| \widehat {g} (\theta_ {n}) \| ^ {2} | \theta_ {n} \big ]} \\ & {\qquad = d ^ {2} (\theta_ {n}, \Theta^ {*}) + 2 \eta_ {n} g ^ {\top} (\theta_ {n}) (\theta_ {n} - \Pi_ {\Theta^ {*}} (\theta_ {n})) + \eta_ {n} ^ {2} \big [ \mathrm{MSE} (\widehat {g} (\theta_ {n})) + \| g (\theta_ {n}) \| ^ {2} \big ].} \end{array}\tag{30}
$$

Under L-smoothness (Assumption 3) and weak strong concavity (Assumption 9), we have

$$
\| g (\theta_ {n}) \| ^ {2} = \| g (\theta_ {n}) - g (\Pi_ {\Theta^ {*}} (\theta_ {n})) \| ^ {2} \leq L ^ {2} \| \theta_ {n} - \Pi_ {\Theta^ {*}} (\theta_ {n}) \| ^ {2} = L ^ {2} d ^ {2} (\theta_ {n}, \Theta^ {*}),
$$

and

$$
g ^ {\top} (\theta_ {n}) (\theta_ {n} - \Pi_ {\Theta^ {*}} (\theta_ {n})) \leq - \mu d ^ {2} (\theta_ {n}, \Theta^ {*}).
$$

Setting $\eta _ { n } = \beta / n$ , and combining these inequalities with (30) yields

$$
\mathbb {E} [ d ^ {2} (\theta_ {n + 1}, \Theta^ {*}) ] \leq \left(1 - \frac {2 \mu \beta}{n} + \frac {L ^ {2} \beta^ {2}}{n ^ {2}}\right) \mathbb {E} [ d ^ {2} (\theta_ {n}, \Theta^ {*}) ] + \frac {\beta^ {2} M}{n ^ {2}}.
$$

Under the assumption that $\beta > ( 2 \mu ) ^ { - 1 }$ , we can apply Lemma 14 again to obtain

$$
\mathbb {E} [ d ^ {2} (\theta_ {n}, \Theta^ {*}) ] = O (n ^ {- 1}).
$$

This establishes the desired convergence rate of $\theta _ { n }$ .

Step (iii) Asymptotic normality of $Q ^ { \top } \theta _ { n }$ : In overparameterized models such as LLMs, the entire parameter vector is not asymptotically normal because it is not fully identifiable. To address this, we decompose the parameter into its identifiable and non-identifiable components. Specifically, the matrix Q introduced in Assumption 7 forms an orthonormal basis for the identifiable subspace.

Without loss of generality, we assume the Hessian matrix $H ^ { * }$ in Assumption 7 is a diagonal matrix dia $\mathrm { g } ( - \lambda _ { 1 } , \ldots , - \lambda _ { r } )$ , where $\lambda _ { 1 } , \ \cdots , \ \lambda _ { r }$ are the r positive eigenvalues of $- H ^ { * }$ . Indeed, since $H ( \theta ^ { * } )$ is symmetric, so is $H ^ { * }$ . As such, there exists an orthogonal matrix P such that

$P ^ { \top } H ^ { * } P = \mathrm { d i a g } ( - \lambda _ { 1 } , \ldots , - \lambda _ { r } )$ . Consequently, we can replace $H ^ { * }$ and the projection matrix Q by $P ^ { \top } H ^ { * } P$ (which is diagonal) and $Q P ^ { \top }$ , respectively. Let $\bar { Q } = ( Q , \widetilde { Q } )$ denote an orthogonal completion of Q so that $\bar { Q }$ is a d × d orthogonal matrix and

$$
\bar {Q} ^ {\top} H (\theta^ {*}) \bar {Q} = \mathrm{diag} (- \lambda_ {1}, \ldots , - \lambda_ {r}, 0 \ldots , 0).\tag{31}
$$

By Lemma 15, the set $Q ^ { \top } \Theta ^ { * } = \{ Q ^ { \top } \theta : \theta \in \Theta ^ { * } \}$ consists of a single point, which we denote by $v ^ { * }$ We aim to establish the asymptotic normality of $Q ^ { \top } \theta _ { n } - v ^ { * }$ in this step.

Let $\gamma _ { n } = \bar { Q } ^ { \top } \theta _ { n }$ . Then $\gamma _ { n }$ satisfies the recursion

$$
\gamma_ {n + 1} = \gamma_ {n} + \eta_ {n} \bar {Q} ^ {\top} g (\bar {Q} \gamma_ {n}) + \eta_ {n} \bar {Q} ^ {\top} \big (\widehat {g} (\theta_ {n}) - g (\theta_ {n}) \big).\tag{32}
$$

Let $v _ { n } = Q ^ { \top } \theta _ { n }$ denote the first r components of $\gamma _ { n }$ (the identifiable component), and let $u _ { n }$ denote the remaining $d - r$ components (the non-identifiable component), where d is the dimension of θ. Then

$$
\begin{array}{r l} & v _ {n + 1} = v _ {n} + \eta_ {n} Q ^ {\top} g (Q v _ {n} + \widetilde {Q} u _ {n}) + \eta_ {n} Q ^ {\top} \big (\widehat {g} (\theta_ {n}) - g (\theta_ {n}) \big) \\ & \qquad = v _ {n} + \frac {h (v _ {n} , u _ {n})}{n} + \frac {\Delta M _ {n + 1}}{n}, \end{array}\tag{33}
$$

where

$$
h (v, u) = \beta Q ^ {\top} g (Q v + \widetilde {Q} u), \quad \Delta M _ {n + 1} = \beta Q ^ {\top} \left(\widehat {g} (\theta_ {n}) - g (\theta_ {n})\right)
$$

is a martingale diference satisfying $\mathbb { E } ( \Delta M _ { n + 1 } \mid \theta _ { n } ) = 0 .$

We aim to apply central limit theorems (CLTs) for stochastic approximation $[ \mathrm { e . g . }$ , Zhang, 2016, Proposition B.2] to establish the asymptotic normality of $v _ { n }$ . To invoke this CLT (see Lemma 18), we need to represent $v _ { n }$ in the form

$$
v _ {n + 1} - v ^ {*} = \Big (I _ {r} - \frac {\Sigma + o (1)}{n} \Big) (v _ {n} - v ^ {*}) + \frac {e _ {n + 1}}{n},
$$

for some martingale diference sequence $\{ e _ { n } \}$ . Comparing with (33), we can set $e _ { n + 1 }$ to $\Delta M _ { n + 1 }$ It remains to show that

$$
h (v _ {n}, u _ {n}) = - H (v _ {n} - v ^ {*}) + o (\| v _ {n} - v ^ {*} \|),\tag{34}
$$

for some deterministic matrix H.

Nonetheless, it remains challenging to prove (34). Although the orthogonal transformation isolates the identifiable component, the update still depends on the full parameter vector. Consequently, the recursion for $v _ { n }$ involves not only $v _ { n }$ but also the non-identifiable component $u _ { n }$ , as reflected in (33). However, (34) requires to approximate $h ( v _ { n } , u _ { n } )$ using only $v _ { n }$ . While this may seem impossible at first glance, we show below that it can be achieved under the WSC condition (Assumption 9) by setting $H = - \beta H ^ { * }$

Let $\boldsymbol { u } _ { n } ^ { * }$ denote the projection of the non-identifiable component $u _ { n }$ onto $\Theta ^ { * }$ , defined through the following sequence of mappings:

$$
u _ {n} \mapsto \theta_ {n} \mapsto \theta_ {n} ^ {*} \mapsto \gamma_ {n} ^ {*} \mapsto u _ {n} ^ {*}.
$$

Specifically, recall that $\theta _ { n } = \bar { Q } \gamma _ { n } = \bar { Q } ( v _ { n } ^ { \top } , u _ { n } ^ { \top } ) ^ { \top }$ . Let $\theta _ { n } ^ { * } = \Pi _ { \Theta ^ { * } } ( \theta _ { n } )$ denote the projection of $\theta _ { n }$ onto $\Theta ^ { * }$ , and define $\gamma _ { n } ^ { * } = \bar { Q } ^ { \top } \theta _ { n } ^ { * }$ . We decompose $\gamma _ { n } ^ { * }$ as $\gamma _ { n } ^ { * } = ( v ^ { * \top } , u _ { n } ^ { * \top } ) ^ { \top }$ , where $v ^ { * }$ and $\boldsymbol { u } _ { n } ^ { * }$ correspond to the first r and last $d - r$ components of $\gamma _ { n } ^ { * }$ , respectively.

Note that by Lemma $1 5 , v ^ { * }$ is unique and does not depend on n, whereas $\boldsymbol { u } _ { n } ^ { * }$ may vary with n. By definition, we have $h ( v ^ { * } , u _ { n } ^ { * } ) = 0$ . It follows from Taylor’s theorem that

$$
\begin{array}{r l} & h (v _ {n}, u _ {n}) = h (v _ {n}, u _ {n}) - h (v ^ {*}, u _ {n} ^ {*}) = \beta Q ^ {\top} [ g (Q v _ {n} + \widetilde {Q} u _ {n}) - g (Q v ^ {*} + \widetilde {Q} u _ {n} ^ {*}) ] \\ = & \beta Q ^ {\top} H (\theta^ {*}) [ Q (v _ {n} - v ^ {*}) + \widetilde {Q} (u _ {n} - u _ {n} ^ {*}) ] + o (\| Q (v _ {n} - v ^ {*}) \| + \| \widetilde {Q} (u _ {n} - u _ {n} ^ {*}) \|), \end{array}
$$

where the first term on the second line equals $\beta Q ^ { \top } H ( \theta ^ { * } ) Q ( v _ { n } - v ^ { * } ) \ : = \ : \beta H ^ { * } ( v _ { n } - v ^ { * } )$ by (31). Moreover, the second term $= o ( \left\| v _ { n } - v ^ { * } \right\| + \left\| u _ { n } - u _ { n } ^ { * } \right\| )$ , and somewhat surprisingly, this term further simplifies to $o ( \left\| v _ { n } - v ^ { * } \right\| )$ under the WSC condition (Assumption 9), as we show below.

Under Assumption 9, we have

$$
\mu \big [ \| v _ {n} - v ^ {*} \| ^ {2} + \| u _ {n} - u ^ {*} \| ^ {2} \big ] \leq g ^ {\top} (Q \gamma_ {n}) \bar {Q} (\gamma_ {n} ^ {*} - \gamma_ {n}).
$$

Applying Taylor’s theorem again, the RHS becomes

$$
- (\gamma_ {n} - \gamma_ {n} ^ {*}) ^ {\top} Q ^ {\top} H Q (\gamma_ {n} - \gamma_ {n} ^ {*}) + o (\| \gamma_ {n} - \gamma_ {n} ^ {*} \| ^ {2}) = - \sum_ {k = 1} ^ {r} \lambda_ {k} (v _ {n k} - v _ {k} ^ {*}) ^ {2} + o (\| \gamma_ {n} - \gamma_ {n} ^ {*} \| ^ {2}).
$$

Consequently, we have for suficiently large n that

$$
\sum_ {k = 1} ^ {r} (\epsilon - \lambda_ {k} - \mu) (v _ {n k} - v _ {k} ^ {*}) ^ {2} \geq (\mu - \epsilon) \| u _ {n} - u ^ {*} \| ^ {2},
$$

for some $\epsilon < \mu .$ . This leads to $\| u _ { n } - u ^ { * } \| = O ( \| v _ { n } - v ^ { * } \| )$

To summarize, we have shown that

$$
h (v _ {n}, u _ {n}) = \beta H ^ {*} (v _ {n} - v ^ {*}) + o (\| v _ {n} - v ^ {*} \|).\tag{35}
$$

This proves (34) with $\Sigma = - \beta H ^ { * }$ . Define

$$
H _ {n} = \left\{ \begin{array}{l l} - \beta H ^ {*} - \frac {[ h (v _ {n} , u _ {n}) - \beta H ^ {*} (v _ {n} - v ^ {*}) ] (v _ {n} - v ^ {*}) ^ {\top}}{\| v _ {n} - v ^ {*} \| ^ {2}}, & \text {if (v_{n} ,u_{n})\notin\bar {Q}^{\top}\Theta^{*}}, \\ - \beta H ^ {*}, & \text {if (v_{n} ,u_{n})\in\bar {Q}^{\top}\Theta^{*}}. \end{array} \right.
$$

It follows from (35) that $H _ { n }$ is well-defined and satisfies $H _ { n } \to - \beta H ^ { * }$ almost surely, given the almost sure convergence of the estimated parameters.

Additionally, we have $h ( v _ { n } , u _ { n } ) = H _ { n } ( v _ { n } - v ^ { * } )$ by definition. This together with (33) leads to

$$
v _ {n + 1} - v ^ {*} = \left(I _ {r} - \frac {H _ {n}}{n}\right) (v _ {n} - v ^ {*}) + \frac {\Delta M _ {n + 1}}{n}.
$$

Based on this equation, according to Lemma 18, it remains to verify the following conditions to establish the CLT:

(i) The smallest eigenvalue of the limit of $H _ { n } , { \mathrm { i . e . , ~ } } - \beta H ^ { * }$ , should be larger than $1 / 2$

(ii) The Lindeberg condition for martingale diferences: for any $\epsilon > 0$ ，

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} [ \| \Delta M _ {i} \| ^ {2} \mathbb {I} \{\| \Delta M _ {i} \| \geq \epsilon \sqrt {n} | \mathcal {F} _ {i - 1} ] \to 0 \text {a.s.},
$$

where ${ \mathcal { F } } _ { i }$ denotes the filtration generated by the martingale process.

(iii) Convergence of the quadratic variation:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} \left[ (\Delta M _ {i}) (\Delta M _ {i}) ^ {\top} | \mathcal {F} _ {i - 1} \right] \to \Omega \quad \mathrm{a.s.},
$$

for some symmetric positive semidefinite (random) matrix Ω.

To verify Condition (i), notice that by Lemma 16, the smallest eigenvalue of $- H ^ { * }$ is lower bounded by $\mu .$ Condition (i) thus holds under the assumption that $\beta > ( 2 \mu ) ^ { - 1 }$ (Assumption 5). Condition (ii) follows from the bounded reward assumption (Assumption 1) and the bounded score assumption (Assumption 3), which together imply the boundedness of $\nabla M _ { i }$ . Condition (iii) is directly implied by Assumption 8, with $\Omega = Q ^ { \top } \Gamma Q$ . Applying Lemma 18, we obtain

$$
v _ {n} \xrightarrow {d} N (0, \Sigma),
$$

where

$$
\Sigma = \int_ {0} ^ {\infty} (e ^ {- (- \beta H ^ {*} - \frac {1}{2} I _ {r}) u}) ^ {\top} Q ^ {\top} \Gamma Q e ^ {- (- \beta H ^ {*} - \frac {1}{2} I _ {r}) u} d u.
$$

This completes the proof of this step.

Step (iv) Asymptotic distribution of $\Delta ( \pi _ { \theta _ { n } } )$ : Finally, we derive the asymptotic distribution of $\Delta ( \pi _ { \theta _ { n } } )$ . Recall that $\theta _ { n } ^ { * } = \arg \operatorname* { m i n } _ { \theta \in \Theta ^ { * } } d ( \theta _ { n } , \theta )$ . It follows from Taylor’s theorem that

$$
\begin{array}{r c l} \Delta (\pi_ {\theta_ {n}}) & = & - [ J (\theta_ {n}) - J (\theta_ {n} ^ {*}) ] \\ & = & - \frac {1}{2} (\theta_ {n} - \theta_ {n} ^ {*}) ^ {\top} H (\theta_ {n} ^ {*}) (\theta_ {n} - \theta_ {n} ^ {*}) + o (\| \theta_ {n} - \theta_ {n} ^ {*} \| ^ {2}) \\ & = & - \frac {1}{2} (\theta_ {n} - \theta_ {n} ^ {*}) ^ {\top} H (\theta_ {n} ^ {*}) (\theta_ {n} - \theta_ {n} ^ {*}) + o _ {p} (n ^ {- 1}), \end{array}
$$

where the last equation follows from the rate of convergence derived in Step (ii).

Consider the leading term on the RHS. Notice that

$$
\begin{array}{r l} & {(\theta_ {n} - \theta_ {n} ^ {*}) ^ {\top} H (\theta_ {n} ^ {*}) (\theta_ {n} - \theta_ {n} ^ {*})} \\ {=} & {(\theta_ {n} - \theta_ {n} ^ {*}) ^ {\top} \bar {Q} H ^ {*} \bar {Q} ^ {\top} (\theta_ {n} - \theta_ {n} ^ {*})} \\ {=} & {- (v _ {n} - v ^ {*}) ^ {\top} \mathrm{diag} (\lambda_ {1}, \ldots , \lambda_ {r}) (v _ {n} - v ^ {*})} \end{array}
$$

Combining this with the asymptotic normality ${ \sqrt { n } } ( v _ { n } - v ^ { * } ) \ { \stackrel { d } { \to } } \ N ( 0 , \Sigma )$ established in Step (iii), we obtain

$$
n \Delta (\pi_ {\theta_ {n}}) \xrightarrow {d} \sum_ {k = 1} ^ {r} w _ {k} \chi_ {1, k} ^ {2}
$$

where $w _ { k }$ are eigenvalues of $\frac { 1 } { 2 } \Sigma ^ { 1 / 2 } \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { r } ) \Sigma ^ { 1 / 2 }$ . This completes the proof of Theorem 8.

Proof of Corollary 9: Under Assumption 3, the policy score $\nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } }$ is both continuous (as a function of θ) and uniformly bounded. This together with the bounded reward assumption (Assumption 1) enables us to invoke the dominated convergence theorem to show that $| \Omega ( \theta _ { n } ) - \Omega ( \theta _ { n } ^ { * } ) |  0$ whenever $d ( \theta _ { n } , \Theta ^ { * } ) \to 0$ , where we recall $\theta _ { n } ^ { * } = \Pi _ { \Theta ^ { * } } ( \theta _ { n } )$ . By assumption, $\Omega ( \theta ^ { * } )$ depends on $\theta ^ { * }$ only through $Q ^ { \top } \theta ^ { * } = v ^ { * }$ , which is unique and invariant over $\Theta ^ { * }$ by Lemma 15. Fixing any $\theta ^ { * } \in \Theta ^ { * }$ , we thus obtain $\Omega ( \theta _ { n } ) \to \Omega ( \theta ^ { * } )$ , which under Assumption 8 yields $Q ^ { \top } \Gamma Q = \Omega ( \theta ^ { * } )$ .

Using the same calculations as in the proof of Proposition 3, we have

$$
\mathrm{Cov} (\widehat {g} _ {\mathrm{GRPO}} (\theta^ {*})) - \mathrm{Cov} (\widehat {g} _ {\mathrm{oracle}} (\theta^ {*})) = O (G ^ {- 2}).
$$

Consequently, the same order of magnitude holds for $\Omega _ { \mathrm { G R P O } } ( \theta ^ { * } ) - \Omega _ { \mathrm { o r a c l e } } ( \theta ^ { * } )$ , and hence for $Q ^ { \top } ( \Gamma _ { \mathrm { G R P O } } - \Gamma _ { \mathrm { o r a c l e } } ) Q$ , where $\Gamma _ { \mathrm { o r a c l e } }$ and $\Gamma _ { \mathrm { G R P O } }$ denote the matrices Γ in Assumption 8 under the oracle and GRPO algorithms, respectively.

It follows that $\Sigma _ { \mathrm { G R P O } } \mathrm { ~ - ~ } \Sigma _ { \mathrm { o r a c l e } } = O ( G _ { \cdot } ^ { - 2 } )$ , where $\Sigma _ { \mathrm { G R P O } }$ and $\Sigma _ { \mathrm { o r a c l e } }$ denote the asymptotic covariance matrices of $Q ^ { \top } \widehat { \theta } _ { n , \mathrm { G R P O } }$ and $Q ^ { \top } \widehat { \theta } _ { n , \mathrm { o r a c l e } }$ defined in Step (iii) of the proof of Theorem 8. Now, define

$$
W _ {a} := - \frac {1}{2} \Sigma_ {a} ^ {1 / 2} (\theta^ {*}) H ^ {*} \Sigma_ {a} ^ {1 / 2} (\theta^ {*}), \qquad a \in \{\text { GRPO,oracle } \}.
$$

By assumption, both $\Omega _ { \mathrm { G R P O } } ( \theta ^ { * } )$ and $\Omega _ { \mathrm { o r a c l e } } ( \theta ^ { * } )$ are positive definite. So are $\Sigma _ { \mathrm { o r a c l e } }$ and $\Sigma _ { \mathrm { G R P O } }$ Therefore, the matrix square-root map is locally Lipschitz so that

$$
\left\| \Sigma_ {\mathrm{GRPO}} ^ {1 / 2} - \Sigma_ {\mathrm{oracle}} ^ {1 / 2} \right\| = O \big (\left\| \Sigma_ {\mathrm{GRPO}} - \Sigma_ {\mathrm{oracle}} \right\| \big) = O (G ^ {- 2}).
$$

Hence, it follows from the negative definiteness of $H ^ { * }$ (Assumption 7) that

$$
\begin{array}{r} W _ {\mathrm{GRPO}} - W _ {\mathrm{oracle}} = \frac {1}{2} \Big (\Sigma_ {\mathrm{GRPO}} ^ {1 / 2} (- H ^ {*}) \Sigma_ {\mathrm{GRPO}} ^ {1 / 2} - \Sigma_ {\mathrm{oracle}} ^ {1 / 2} (- H ^ {*}) \Sigma_ {\mathrm{oracle}} ^ {1 / 2} \Big) \\ = \frac {1}{2} \Big [ \big (\Sigma_ {\mathrm{GRPO}} ^ {1 / 2} - \Sigma_ {\mathrm{oracle}} ^ {1 / 2} \big) (- H ^ {*}) \Sigma_ {\mathrm{GRPO}} ^ {1 / 2} + \Sigma_ {\mathrm{oracle}} ^ {1 / 2} (- H ^ {*}) \big (\Sigma_ {\mathrm{GRPO}} ^ {1 / 2} - \Sigma_ {\mathrm{oracle}} ^ {1 / 2} \big) \Big ], \end{array}
$$

if of the order $O ( G ^ { - 2 } )$ as well.

By Weyl’s inequality,

$$
| w _ {k, \mathrm{GRPO}} - w _ {k, \mathrm{oracle}} | \leq \| W _ {\mathrm{GRPO}} - W _ {\mathrm{oracle}} \|, \qquad k = 1, \ldots , r.
$$

Therefore,

$$
w _ {k, \mathrm{GRPO}} - w _ {k, \mathrm{oracle}} = O (G ^ {- 2}).
$$

The proof is hence completed.

Proof of Corollary 10: Using the same calculations as in the proof of Corollary 5, we can show that $\Omega _ { \mathrm { o r a c l e } } ( \theta ^ { * } ) \preceq \Omega ( \theta ^ { * } )$ in the positive semidefinite order, under the conditional uncorrelation assumption in Corollary 10, where $\Omega ( \theta ^ { * } )$ denotes the projected covariance matrix $\operatorname { C o v } ( \widehat { g } ( \theta ^ { * } ) )$ for a given meta-algorithm whose baseline is a function of X only. The remainder of the proof follows very similarly to that of Corollary 9. □

## B.3 Practical considerations

This section presents the proofs of Lemma 11, Theorem 12 and Proposition 13.

Proof of Lemma 11. By the law of large numbers, $\sec ( Z ) \stackrel { P } { \to } \mathrm { s t d } _ { x } ( Z )$ as $G  \infty$ . Under the assumption that std<sub>x</sub>(Z) is bounded away from zero, it follows from the continuous mapping theorem that $1 / { \mathrm { s e } } ( Z ) \stackrel { P } {  } 1 / { \mathrm { s t d } } _ { x } ( Z )$ . Under the bounded reward assumption (Assumption 1), the bounded score assumption (Assumption 3) and the coverage assumption (Assumption 10), it follows from that the gradient estimator is asymptotically equivalent to a version with $\mathrm { s e } ( Z ^ { ( \bullet ) } )$ ) replaced by $\operatorname { s t d } ( Z )$ , as $G  \infty$ . Following arguments similar to those in the proof of Lemma 1, this version is a U-statistic with the kernel function given in Lemma 11. This completes the proof. □

Proof of Theorem 12. According to the bias-variance decomposition, we have

$$
\mathrm{MSE} (\widehat {g} (x; \theta)) = \| b (x; \theta) \| ^ {2} + \mathrm{trace} (\mathrm{Var} (\widehat {g} (x; \theta)),
$$

where $b ( x ; \theta ) : = \mathbb { E } [ \widehat { g } ( x ; \theta ) ] - g ^ { \dagger } ( x ; \theta )$

Following similar arguments to those in the proof of Theorem 2, we can derive that

$$
\operatorname{trace} \left(\operatorname{Var} \left(\widehat {g} ^ {\dagger} (x; \theta)\right) = \frac {\operatorname{trace} \left[ \Sigma_ {1} (x ; \theta) \right]}{G} + O \left(\frac {1}{\epsilon G ^ {2}}\right). \right.
$$

It remains to upper bound the squared bias term. Notice that the bias can be represented by

$$
b (x; \theta) = \sum_ {t} \mathbb {E} \Big [ (\mu_ {t} ^ {(g)} - \mu^ {(g)}) W _ {t} ^ {(g)} \frac {Z ^ {(g)} - \bar {Z} ^ {(- g)}}{\mathrm{std} (Z)} \Big ] + \kappa \mathbb {E} \Big [ (1 - \mu^ {(g)}) \sum_ {t} (\mu_ {t} ^ {(g)} - 1) W _ {t} ^ {(g)} \Big ].
$$

Since $\nabla _ { \theta } \pi _ { \theta } = \pi _ { \theta } \nabla$ log $\pi _ { \boldsymbol { \theta } } .$ , it follows from the bounded score condition in Assumption 3 that $\pi _ { \theta }$ is Lipchitz continuous as a function of θ. Under Assumptions 1, 3 and 10, the gradient $\widehat g$ is bounded. As such,

$$
\pi_ {\theta} - \pi_ {\theta_ {\mathrm{old}}} = O (\| \theta - \theta_ {\mathrm{old}} \|) = O (m \eta_ {i + 1}).\tag{36}
$$

This together with the coverage assumption yields

$$
\mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} | \mu_ {t} ^ {(g)} - 1 | ^ {2} = \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} \left\{\frac {(\pi_ {\theta} (Y _ {t} ^ {(g)} | x , Y _ {<   t} ^ {(g)}) - \pi_ {\theta_ {\mathrm{old}}} (Y _ {t} ^ {(i)} | x , Y _ {<   t} ^ {(i)})) ^ {2}}{\pi_ {\theta_ {\mathrm{old}}} ^ {2} (Y _ {t} ^ {(i)} | x , Y _ {<   t} ^ {(i)})} \right\}\tag{37}
$$

$$
\leq \frac {c m ^ {2} \eta_ {i + 1} ^ {2}}{\epsilon} \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} [ \pi_ {\theta_ {\mathrm{old}}} ^ {- 1} (Y _ {t} ^ {(i)} | x, Y _ {<   t} ^ {(i)}) ] = O \Big (\frac {m ^ {2} \eta_ {i + 1} ^ {2}}{\epsilon} \Big),\tag{38}
$$

for some constant $c > 0$

Following a similar argument, we have

$$
\mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} | \mu^ {(g)} - 1 | ^ {2} = O \Big (\frac {m ^ {2} \eta_ {i + 1} ^ {2}}{\epsilon} \Big).\tag{39}
$$

It follows the Cauchy-Schwarz inequality that

$$
\begin{array}{r l} & b ^ {2} (x; \theta) \leq 2 \Big \| \sum_ {t} \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} \Big \{(1 - \mu^ {(g)}) W _ {t} ^ {(g)} \Big [ \frac {Z ^ {(g)} - \bar {Z} ^ {(- g)}}{\mathrm{std} _ {x} (Z)} + \kappa (\mu_ {t} ^ {(g)} - 1) \Big ] \Big \} \Big \| ^ {2} \\ & \qquad + 2 \Big \| \sum_ {t} \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} \Big [ (1 - \mu_ {t} ^ {(g)}) W _ {t} ^ {(g)} \frac {Z ^ {(g)} - \bar {Z} ^ {(- g)}}{\mathrm{std} _ {x} (Z)} \Big ] \Big \| ^ {2} \\ & \qquad = O \Big (\frac {\kappa^ {2}}{\epsilon^ {2}} \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} | \mu^ {(g)} - 1 | ^ {2} \Big) + O \Big (\sum_ {t} \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} | \mu_ {t} ^ {(g)} - 1 | ^ {2} \Big), \end{array}\tag{40}
$$

where the last equality holds under the bounded reward assumption (Assumption 1), the bounded score assumption (Assumption 3), the coverage assumption (Assumption 10) and that $\operatorname { s t d } _ { X } ( Z )$ is uniformly bounded away from 0.

Combining (37) – (40), we obtain

$$
b ^ {2} (x; \theta) = O \Big (\Big [ \frac {\kappa^ {2}}{\epsilon^ {2}} + 1 \Big ] \frac {m ^ {2} \eta_ {i + 1} ^ {2}}{\epsilon} \Big).
$$

This completes the proof of Theorem 12.

Proof of Proposition 13. As discussed below Proposition 13, the ground-truth gradient $g ^ { \dagger } ( \theta )$ is asymptotically equivalent to the derivative of $\mathcal { I } ( \boldsymbol { \theta } )$ by (36). The rest of the proof follows very similarly to those of Lemma 6 and Theorem 8. We omit the details for brevity. □

## B.4 Auxiliary lemmas and their proofs

Lemma 14. Consider a sequence $\{ a _ { n } \} _ { n }$ that satisfies $0 \leq a _ { n } \leq a$ for some $a > 0$ and

$$
a _ {n + 1} \leq \left(1 - \frac {A}{n} + \frac {B}{n ^ {2}}\right) a _ {n} + \frac {C}{n ^ {2}}, \quad \forall n \geq 0,\tag{41}
$$

where $A , B , C$ are three constants satisfying $A > 1$ and $B , C \geq 0$ . Then for any constant $\varepsilon > 0$ , we have

$$
a _ {n} \leq \max \left\{\frac {(1 + \varepsilon) C}{(A - 1) n}, \frac {a n _ {0}}{n} \right\}\tag{42}
$$

holds for all $n ,$ , where $n _ { 0 }$ is a constant that depends only on A, B and $\varepsilon .$ .

Proof of Lemma $1 \not \angle \cdot$ For any fixed $\varepsilon ,$ let $n _ { 0 }$ denote $\begin{array} { r } { \lceil \frac { B ( 1 + \varepsilon ) } { \varepsilon ( A - 1 ) } \rceil + 1 } \end{array}$ where $\lceil x \rceil$ represents the smallest integer that is larger than x. We prove (42) by induction:

1. Base step: For any $n \leq n _ { 0 }$ , it directly follows from the condition $a _ { n } \leq a$ that $a _ { n } \leq a n _ { 0 } / n$ The inequality thus holds for any $n \leq n _ { 0 }$

2. Induction step: Suppose inequality (42) holds for $n = k$ . We aim to show it holds for $n = k + 1$ as well. To ease notation, let $M = \operatorname* { m a x } \{ ( 1 + \varepsilon ) C / ( A - 1 ) , a n _ { 0 } \}$ . Then according to inequality (41),

$$
a _ {k + 1} - \frac {M}{k + 1} \leq \left(1 - \frac {A}{k} + \frac {B}{k ^ {2}}\right) a _ {k} + \frac {C}{k ^ {2}} - \frac {M}{k + 1}.
$$

Applying the induction hypothesis that $a _ { k } \le M / k$ yields

$$
\begin{array}{r c l} a _ {k + 1} - \frac {M}{k + 1} & \leq & \left(1 - \frac {A}{k} + \frac {B}{k ^ {2}}\right) \frac {M}{k} + \frac {C}{k ^ {2}} - \frac {M}{k + 1} \\ & = & \frac {M}{k (k + 1)} - \frac {A M - C}{k ^ {2}} + \frac {B M}{k ^ {3}} \\ & = & \frac {1}{k ^ {3}} \left[ \frac {k ^ {2} M}{k + 1} - (A M - C) k + B M \right] \\ & \leq & \frac {1}{k ^ {3}} [ k M - (A M - C) k + B M ] \\ & = & \frac {1}{k ^ {3}} \left\{k [ (1 - A) M + C ] + B M \right\}. \end{array}
$$

Since $M \geq \frac { ( 1 + \varepsilon ) C } { A - 1 }$ , we have $( 1 - A ) M + C \leq - \varepsilon C \leq 0$ . Combining the fact that $\begin{array} { r } { k > \frac { B \left( 1 + \varepsilon \right) } { \varepsilon \left( A - 1 \right) } } \end{array}$ , we obtain

$$
\begin{array}{r c l} a _ {k + 1} - \frac {M}{k + 1} & \leq & \frac {1}{k ^ {3}} \left\{\frac {B (1 + \varepsilon)}{\varepsilon (A - 1)} [ (1 - A) M + C ] + B M \right\} \\ & = & \frac {B}{k ^ {3}} \left\{\frac {C (1 + \varepsilon)}{A - 1} - M \right\} \leq 0, \end{array}
$$

where the last inequality follows from the definition of M as well.

Combining the results from both steps completes the proof of Lemma 14.

Lemma 15. Assume the Hessian matrix $H ^ { * }$ in Assumption 7 is a diagonal matrix diag( $- \lambda _ { 1 } , \ldots , - \lambda _ { r } )$ where $\lambda _ { 1 } , \cdots , \lambda _ { r }$ are the r positive eigenvalues $o f - H ^ { * }$ . Under Assumption $^ { 7 , }$ the set $Q ^ { \top } \Theta ^ { * } =$ $\{ Q ^ { \top } \theta : \theta \in \Theta ^ { * } \}$ consists of a single point $v ^ { * }$

Proof of Lemma 15. Suppose there are two diferent points $v _ { 1 } , v _ { 2 } \in Q ^ { \top } \Theta ^ { * }$ with $v _ { 1 } = Q ^ { \top } \theta _ { 1 } ^ { * } , v _ { 2 } =$ $Q ^ { \top } \theta _ { 2 } ^ { * }$ . Applying Taylor’s expansion leads to,

$$
J (\theta_ {1} ^ {*}) - J (\theta_ {2} ^ {*}) = \nabla_ {\theta} J (\theta_ {2} ^ {*}) + \frac {1}{2} (\theta_ {2} ^ {*} - \theta_ {1} ^ {*}) ^ {\top} H (\widetilde {\theta} ^ {*}) (\theta_ {2} ^ {*} - \theta_ {1} ^ {*}),\tag{43}
$$

for some $\widetilde { \theta } ^ { * }$ lying between $\theta _ { 1 } ^ { * }$ and $\theta _ { 2 } ^ { * }$ . Since $\Theta ^ { * }$ is convex (Assumption 7), we have $\smash { \widetilde { \theta } ^ { * } \in \Theta ^ { * } }$ . Given that $J ( \theta _ { 1 } ^ { * } ) = J ( \theta _ { 2 } ^ { * } )$ and $\nabla J ( \theta _ { 2 } ^ { * } ) = 0$ , it follows from (43) that

$$
0 = (\theta_ {2} ^ {*} - \theta_ {1} ^ {*}) ^ {\top} H (\widetilde {\theta} ^ {*}) (\theta_ {2} ^ {*} - \theta_ {1} ^ {*}) = (\theta_ {2} ^ {*} - \theta_ {1} ^ {*}) ^ {\top} \bar {Q} \mathrm{diag} (\lambda_ {1}, \dots , \lambda_ {r}, \mathbf {0} ^ {\top}) \bar {Q} ^ {\top} (\theta_ {2} ^ {*} - \theta_ {1} ^ {*}).
$$

Expanding further, we obtain

$$
0 = \sum_ {k = 1} ^ {r} \lambda_ {k} (v _ {2 k} - v _ {1 k}) ^ {2},\tag{44}
$$

where $v _ { i k }$ denotes the k-th element of $v _ { i }$ . From (44), it is immediate to see that $v _ { 1 } = v _ { 2 }$ , confirming that $Q ^ { \top } \Theta ^ { * }$ is a singleton. This completes the proof. □

Lemma 16. Consider a function $J ( \theta )$ defined on a compact parameter set $\Theta \subseteq \mathbb { R } ^ { d }$ . Suppose $J ( \theta )$ is twice continuously diferentiable, and its gradient vector $g ( \theta )$ satisfies PL with constant µ (refer to Assumption $\it 4 )$ . Then, for any maximizer $\theta ^ { * } \in \Theta$ of J, every positive eigenvalue of the negative Hessian matrix $- \nabla _ { \theta } g ( \theta ^ { * } )$ is no smaller than $\mu .$ In particular, under the L-smoothness condition (Assumption 3), we have $\mu \leq L$

Proof of Lemma 16. Let $\theta ^ { * } \in \Theta$ be a maximizer of $J ( \theta )$ , λ be a positive eigenvalue of $- \nabla _ { \theta } g ( \theta ^ { * } )$ and v be its eigenvector with $\lVert \boldsymbol { v } \rVert = 1$ . Applying Talyor’s theorem to the function $f ( t ) = J ( \theta ^ { * } + t v )$ at $t = 0$ , we obtain:

$$
J (\theta^ {*} + t v) - J (\theta^ {*}) = t v ^ {\top} g (\theta^ {*}) + \frac {1}{2} t ^ {2} v ^ {\top} \nabla_ {\theta} g (\theta^ {*}) v + o (t ^ {2}).
$$

Since $\theta ^ { * }$ maximizes $J ( \theta )$ , we have $g ( \theta ^ { * } ) = 0$ . Moreover, since v is an eigenvector with unit $\ell _ { 2 } { \mathrm { - n o r m } }$ we have $v ^ { \top } \nabla _ { \theta } g ( \theta ^ { * } ) v = - \lambda$ . Therefore,

$$
J (\theta^ {*}) - J (\theta^ {*} + t v) = \frac {1}{2} t ^ {2} \lambda + o (t ^ {2}).\tag{45}
$$

Applying the PL condition to the left hand side, we obtain

$$
\begin{array}{r l} & J (\theta^ {*}) - J (\theta^ {*} + t v) \leq \frac {1}{2 \mu} \| g (\theta^ {*} + t v) \| ^ {2} \\ & \qquad = \frac {1}{2 \mu} \| g (\theta^ {*} + t v) - g (\theta^ {*}) \| ^ {2} \\ & \qquad = \frac {t ^ {2}}{2 \mu} \| v ^ {T} \nabla_ {\theta} g (\theta^ {*}) \| ^ {2} + o (t ^ {2}) \\ & \qquad = \frac {\lambda^ {2} t ^ {2}}{2 \mu} + o (t ^ {2}). \end{array}\tag{46}
$$

Combining (45) and (46) yields

$$
\frac {\lambda^ {2} t ^ {2}}{2 \mu} \geq \frac {1}{2} \lambda t ^ {2} + o (t ^ {2}).
$$

Dividing both sides by $t ^ { 2 }$ and letting $t  0$ gives $\lambda ^ { 2 } \geq \mu \lambda$ . Since $\lambda > 0$ , it follows that $\lambda \geq \mu$ Therefore, all positive eigenvalues of $- \nabla _ { \theta } g ( \theta ^ { * } )$ are lower bounded by $\mu .$ . Using similar arguments, we can show that the smoothness constant L must also satisfy $L \geq \mu$ □

Lemma 17 (Robins-Siegmund Theorem). Let $( \Omega , { \mathcal { F } } , ( { \mathcal { F } } _ { n } ) _ { n \geq 1 } , P )$ be a filtered probability space. For each $n \geq 1 , V _ { n } , A _ { n } , B _ { n } , C _ { n }$ are four positive ${ \mathcal { F } } _ { n }$ -measurable random variable such that

$$
\mathbb {E} [ V _ {n + 1} | \mathcal {F} _ {n} ] \leq (1 + A _ {n}) V _ {n} + B _ {n} - C _ {n}.
$$

$i f \ \sum _ { n = 1 } ^ { \infty } A _ { n } \ < \ \infty$ and $\textstyle \sum _ { n = 1 } ^ { \infty } B _ { n } ~ < ~ \infty$ holds almost surely, then $V _ { n }$ converges to some random variable $V _ { \infty }$ almost surely. Moreover, $\textstyle \sum _ { n = 1 } ^ { \infty } C _ { n } < \infty$ almost surely.

Proof of Lemma 17. Refer to Theorem 1 of Robbins and Siegmund [1971].

Lemma 18. Let $\{ \Delta M _ { n } , \mathcal { F } _ { n } , n \ge 1 \}$ is a martingale diference sequence satisfying

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} \left[ (\Delta M _ {i}) (\Delta M _ {i}) ^ {\top} | \mathcal {F} _ {i - 1} \right] \to \Lambda \quad a. s.,
$$

where Λ is a symmetric positive semidefinite random matrix and

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} \left[ \| \Delta M _ {i} \| ^ {2} \mathbb {I} \{\Delta M _ {i} \geq \varepsilon \sqrt {n} \} | \mathcal {F} _ {i - 1} \right] \to 0 \quad a. s. f o r \forall \varepsilon > 0.
$$

Let $\{ H _ { n } \} _ { n }$ denote a sequence of random matrices satisfying $H _ { n } \to H$ almost surely, where H is a nonrandom matrix with all its eigenvalues larger than $1 / 2$ . Then for a sequence of random vectors $\Delta _ { n }$ that satisfies

$$
\Delta_ {n + 1} = \Big (I - \frac {H _ {n}}{n} \Big) \Delta_ {n} + \Delta M _ {n},
$$

we have ${ \sqrt { n } } \Delta _ { n } \ { \stackrel { d } { \to } } \ N ( 0 , \Sigma )$ where

$$
\Sigma = \int_ {0} ^ {\infty} (e ^ {- (H - \frac {1}{2} I) u}) ^ {\top} \Lambda e ^ {- (H - \frac {1}{2} I) u} d u.
$$

Proof of Lemma 18. The proof follows immediately from Propositions B.1 and B.2 of Zhang [2016].

## C Experiment Details

This section provides the prompts we used in Section 5.

Prompt template for the In-Context Learning model.

```txt
Solve the arithmetic problem. Return ONLY the final integer.
<arithmetic problem>
```

Prompt template for the Base and Instruct models.

```txt
You are a calculator. Solve the arithmetic problem and return ONLY the final integer.

Examples:
Problem: 15 + 25 =
Answer: 40
Problem: 20 + 30 - 15 =
Answer: 35
Problem: 3 * 4 =
Answer: 12
Problem: 24 / 6 =
Answer: 4
Problem: 25 - (10 + 5) =
Answer: 10

Now solve:
Problem: <arithmetic problem>
```

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Group relative policy optimization
1: Input: prompt distribution $f(X)$, initial parameter $\theta\in\Theta$, learning rates $\{\eta_{i}\}_{i\in\mathbb{N}}$, batch size $B$, per-prompt group size $G$, number of minibatches $m$, KL regularization parameter $\kappa&gt;0$ and reference model $\theta_{\text{ref}}$.
2: for $i=0,1,2,\ldots,n-1$ do
3: Set $\theta_{\text{old}}\leftarrow\theta$.
4: Sample prompts $\{X^{(b)}\}_{b=1}^{B}\stackrel{\text{iid}}{\sim}f(\cdot)$.
5: For each prompt $X^{(b)}$, generate a group of $G$ outputs
$\{Y^{(b,g)}\}_{g=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|X^{(b)})$.
6: For each output $Y^{(b,g)}$, obtain reward $Z^{(b,g)}$.
7: Partition $\{1,\ldots,B\}$ into $m$ disjoint minibatches $\mathcal{B}_{1},\ldots,\mathcal{B}_{m}$ of equal size.
8: for $j=1,\ldots,m$ do
9: Compute the minibatch gradient estimator $\widehat{g}(\theta)=|\mathcal{B}_{j}|^{-1}\sum_{b\in\mathcal{B}_{j}}\widehat{g}(X^{(b)};\theta)$ where
where
$\pi_{\theta,t}^{(b,g)}:=\pi_{\theta}\left(Y_{t}^{(b,g)}|X^{(b)},Y_{&lt;t}^{(b,g)}\right)$,
and
$A^{(b,g)}:=\frac{Z^{(b,g)}-\bar{Z}^{(b,-g)}}{\text{se}(Z^{(b,\bullet)})}$,
where $Z^{(b,\bullet)}=\{Z^{(b,g)}\}_{g=1}^{G}$ and
$\text{se}(Z^{(b,\bullet)})=\sqrt{\frac{1}{G-1}\sum_{g}[Z^{(b,g)}-\bar{Z}^{(b)}]^{2}}$,
denotes its standard error and $\bar{Z}^{(b)}$ denotes the empirical group mean $G^{-1}\sum_{g=1}^{G}Z^{(b,g)}$.
10: Update the parameter:
$\theta\leftarrow\theta+\eta_{i+1}\widehat{g}(\theta)$.
11: end for
12: end for
13: Output: policy $\pi_{\theta}$.
</div>
