# Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards

Xiaoyuan Liu<sup>∗,1,2</sup> , Tian Liang<sup>1</sup> , Zhiwei He<sup>1</sup> , Jiahao Xu<sup>1</sup> , Wenxuan Wang<sup>1</sup> , Pinjia He<sup>†,2</sup> , Zhaopeng Tu<sup>†</sup> <sup>,1</sup> , Haitao Mi<sup>1</sup> , and Dong Yu<sup>1</sup>

<sup>1</sup>Tencent <sup>2</sup>The Chinese University of Hong Kong, Shenzhen

Problem Solution & Verification Generation

Figure 1: Illustration of RISE, which consists of two stages: (i) Problem Solution and Verification Generation: problems from the training batch are used to generate chain-of-thought solutions from <sup>y</sup>v<sup>\*</sup> <sup>=</sup> <sup>r</sup>gthe model. Problems and model solutions are then formatted as verification prompts to generate Template T(x, y)verifications of the solutions. (ii) RL Optimization: the original generation data and their verification are mixed as the new batch, and the model is optimized based on the RL objective.

## Abstract

Large Language Models (LLMs) show great promise in complex reasoning, with Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement strategy. However, a prevalent issue is “superficial self-reflection”, where models fail to robustly verify their own outputs. We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online RL framework designed to tackle this. RISE explicitly and simultaneously trains an LLM to improve both its problem-solving and self-verification abilities within a single, integrated RL process. The core mechanism involves leveraging verifiable rewards from an outcome verifier to provide on-the-fly feedback for both solution generation and self-verification tasks. In each iteration, the model generates solutions, then critiques its own on-policy generated solutions, with both trajectories contributing to the policy update. Extensive experiments on diverse mathematical reasoning benchmarks show that RISE consistently improves model’s problem-solving accuracy while concurrently fostering strong self-verification skills. Our analyses high light the advantages of online verification and the benefits of increased verification compute. Additionally, RISE models exhibit more frequent and accurate self-verification behaviors during reasoning. These advantages reinforce RISE as a flexible and effective path towards developing more robust and self-aware reasoners.

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable potential in complex reasoning tasks. A promising avenue for further enhancing these capabilities is Reinforcement Learning (RL), particularly methods that utilize verifiable rewards (RLVR) from outcome verifiers (Gao et al., 2024; DeepSeek-AI et al., 2025; Lambert et al., 2024; Yue et al., 2025). This paradigm, often applied to domains like mathematics where solution correctness can be programmatically evaluated, enabling models to improve through direct feedback on their generated solutions.

However, even with outcome-based RL, models may learn to generate correct answers without deeply understanding the underlying reasoning process or robust self-assessment skills. This can lead to “superficial self-reflection” (Liu et al., 2025), where models struggle to reliably identify flaws in their own reasoning and verify the correctness of their outputs. While some approaches explicitly incorporate self-critique (Xi et al., 2024; Xie et al., 2025) to provide additional signals, the process of learning to solve problems and learning to verify solutions are often decoupled or lack direct, contemporaneous feedback for the verification skill itself within the RL loop.

To address this limitation and foster more robust reasoning, we introduce RISE (Reinforcing Reasoning with Self-Verification) as a novel online reinforcement learning framework. RISE is designed to explicitly and simultaneously train an LLM to improve both its problem-solving ability and its capacity to verify its own generated solutions within a single, integrated RL process. The key idea is to leverage the verifiable reward signal from a rule-based outcome verifier not only to guide the generation of correct solutions but also to align the model’s self-verification ability on-the-fly.

In the RISE framework, during each training iteration, the model first generates solutions for a batch of problems. Subsequently, using these on-policy generated solutions and the original problems, verification problems are constructed with a predefined template, prompting the model to critique its own solution and provide a score. The same outcome verifier used to assess problem solutions also provides ground-truth supervision for the verification task, based on an exact match between the predicted verification score and the ground-truth solution score. Both the problem-solving trajectories and the self-verification trajectories, along with their respective verifiable rewards, are then combined to update the model’s parameters using a unified RL objective. This tight coupling enables the model to learn not only to solve problems, but also to critique and verify its own outputs, fostering a more dynamic and grounded self-improvement loop.

In our experiments, we implement and evaluate RISE using the Proximal Policy Optimization (PPO) algorithm, applying it to the 1.5B, 3B, and 7B base models from the Qwen2.5 series. Compared to a Zero-RL baseline, which incorporates only problem-solving supervision, RISE consistently improves reasoning accuracy and achieves up to a 2.8× increase in verification accuracy on challenging mathematical benchmarks. Moreover, RISE outperforms instruction-tuned models across both tasks. For instance, RISE-3B achieves a 3.7% average improvement in reasoning accuracy and a 33.4% gain in self-verification accuracy over Qwen-3B-Instruct.

We also find that this enhanced self-verification ability contributes to improved test-time performance. Specifically, RISE-3B and RISE-7B outperform standard majority voting by +0.2% and +1.9%, respectively, under a k=4 inference budget. Further analysis reveals that RISE enhances the internal reasoning process by encouraging more frequent and effective verification behaviors. Finally, our ablations demonstrate that online verification is crucial to the success of RISE.

Our main contributions are as follows:

• We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online reinforcement learning framework that explicitly and simultaneously trains LLMs to improve both problemsolving and self-verification capabilities within a single, integrated RL process, leveraging verifiable rewards for both tasks on-the-fly.

• We demonstrate, through extensive experiments on challenging mathematical reasoning benchmarks using a PPO-based implementation, that RISE significantly boosts problem-solving performance while instilling robust self-verification skills in the LLM.

• We provide comprehensive analyses elucidating the critical role of RISE’s online verification mechanism, the benefits of scaling verification training compute, and how the developed selfverification capability contributes to more accurate and reliable solution generation.

## 2 Related Work

RLVR for LLM Reasoning In the literature, reinforcement learning has been widely used to align language models with human preferences, typically through reward models or pairwise preference comparisons (Christiano et al., 2017; Ouyang et al., 2022; Rafailov et al., 2023). More Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach for improving the reasoning capabilities of LLMs in domains such as mathematics and programming (Jaech et al., 2024; DeepSeek-AI et al., 2025). Using only outcome rewards, recent work has demonstrated the scalability of RL algorithms for LLM reasoning (DeepSeek-AI et al., 2025; Team et al., 2025; Zeng et al., 2025; Hu et al., 2025). However, leveraging verifiable rewards not only for reasoning supervision but also as a direct training signal for self-verification remains underexplored, which is the main focus of RISE.

Learning to Solve and Verify Solution generation and verification are two foundational capabilities of LLMs (Huang et al., 2024; Song et al., 2024), echoing the classic P versus NP dichotomy in computer science (Wikipedia contributors, 2025). In the context of LLM reasoning, previous work has focused on teaching models either to solve problems (DeepSeek-AI et al., 2025; Zelikman et al., 2022), to verify solutions (Wang et al., 2023a; Lightman et al., 2023; Shi and Jin, 2025; Zhang et al., 2025), or to leverage the verification capability to perform for self-improvement (Yuan et al., 2024; Xiong et al., 2025). More recently, Lin et al. (2025) proposed a self-play framework that jointly teaches LLMs to generate code and corresponding test cases through two-stage training. In contrast, we introduce an online RL framework that explicitly leverages verifiable reward signals to jointly align the model’s problem-solving and self-verification abilities in a unified training process.

## 3 Reinforcement Learning Preliminaries

Policy Gradient Methods The goal of RL is to learn a policy that maximizes the expected cumulative reward (namely return), denoted as the performance measure J. Policy gradient methods learn a parameterized policy that can select actions to maximize J without consulting other value functions. Grounded by the policy gradient theorem (Sutton and Barto, 2018), the optimization is performed as gradient ascent based on the gradient of J(θ) with respect to the policy parameter θ.

A large language model is naturally a parametrized policy $\pi _ { \theta }$ . The state at time $t ,$ denoted as $s _ { t } ,$ is the concatenation of the prompt x and the response $\mathbf { y } _ { < t }$ generated so far, while the action $a _ { t }$ is the next token $y _ { t }$ . T refers to total timestamps (response length + 1). Thus, the gradient can be expressed as:

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}, \mathbf {y} \sim \pi_ {\theta}} \left[ \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid \mathbf {x}, \mathbf {y} _ {<   t}) A _ {t} \right].
$$

The core part of this method is the advantage function $A _ { t , }$ , which determines the extent to increase or decrease the probability of selecting this action (token) in the given state. In practice, the advantage function is implemented as cumulative discounted rewards subtracting an optional baseline, representing how much better an action is compared to the alternatives:

$$
A _ {t} = \sum_ {t = t _ {0}} ^ {T} \gamma^ {t - t _ {0}} r _ {t} - b (s _ {t _ {0}}),\tag{1}
$$

where $\gamma \in [ 0 , 1 ]$ is the discount factor for the future rewards and $r _ { t } = R ( s _ { t } , a _ { t } , s _ { t + 1 } )$ is the reward from the environment at time t. Different implementations of the baseline formulate multiple variants of policy gradient methods, including using learned state-value functions (e.g., REINFORCE (Williams,

1992), Actor-Critic (Barto et al., 1983)), group-level reward means (e.g., GRPO (Shao et al., 2024)), and leave-one-out (e.g. RLOO (Ahmadian et al., 2024)).

Proximal Policy Optimization Proximal Policy Optimization (PPO) (Schulman et al., 2017) is a popular algorithm of Actor-Critic method, which incorporates a critic model $\phi$ to help estimate advantage for training the actor model θ (i.e., policy). One major improvement of PPO is penalizing excessive policy updates and thereby maintaining training stability. In practice, the objective of the actor model is defined as follows:

$$
\mathcal {J} (\theta) = \mathbb {E} _ {t} \left[ \min (r _ {t} (\theta) \hat {A} _ {t}, \operatorname{clip} \left(r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {t}) - \beta K L \left(\pi_ {\theta} | | \pi_ {r e f}\right)\right),
$$

(2)

where $\begin{array} { r } { r _ { t } ( \theta ) = \frac { \pi _ { \theta } ( y _ { t } | x , y _ { < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { t } | x , y _ { < t } ) } } \end{array}$ . Clip() and $\mathrm { K L } ( )$ are two techniques used for limiting update magnitudes. With Generalized Advantage Estimation (GAE) (Schulman et al., 2016), the advantage is estimated as a λ-weighted sum of step-emporal-Difference (TD) errors:

$$
\hat {A} _ {t} = \delta_ {t} + (\gamma \lambda) \delta_ {t + 1} + \dots + (\gamma \lambda) ^ {T - t + 1} \delta_ {T - 1},\tag{3}
$$

$$
\text { where } \delta_ {t} = r _ {t} + \gamma V _ {\phi} (s _ {t + 1}) - V _ {\phi} (s _ {t}).
$$

T denotes response length with token indexes from 0 to $T - 1 . \ V _ { \phi } { \left( s _ { t } \right) }$ is the value predicted by the critic model $\phi$ at state $s _ { t } , r _ { t }$ is the scalar reward from the environment at time t, and $\lambda \in [ 0 , 1 ]$ is the GAE parameter that trades off between bias and variance. In practice, we set $\lambda = \gamma = \bar { 1 } .$ , thus making the per-token loss averaged over the full response length T. By design, $r _ { t } = 0$ for $t < T - 1 ,$ and $r _ { t } = r$ for $t = T - 1 ( \mathrm { i . e . }$ , outcome reward). After we update the actor model, the critic model should also be updated for accurate value estimations. In practice, we use Mean Squared Error (MSE) to measure the prediction loss and perform the update:

$$
\mathcal {J} (\phi) = \mathbb {E} _ {t} \left[ \max \left((V _ {\phi} (s _ {t}) - V _ {t} ^ {t a r g}) ^ {2}, \left(\operatorname{clip} (V _ {\phi} (s _ {t}), V _ {\phi_ {\mathrm{old}}} (s _ {t}) - \epsilon , V _ {\phi_ {\mathrm{old}}} (s _ {t}) + \epsilon) - V _ {t} ^ {t a r g}\right) ^ {2}\right) \right],\tag{4}
$$

$$
\mathrm{where} V _ {t} ^ {t a r g} = V _ {\phi_ {\mathrm{old}}} (s _ {t}) + \hat {A} _ {t}.
$$

Verifiable Reward Unlike the rewards from conventional reward models which are continuous numbers denoting the goodness of the response, verifiable rewards are usually discrete numbers representing the correctness of the final result (Lambert et al., 2024; DeepSeek-AI et al., 2025). Given the prompt x and the complete response y from the LLM $\pi _ { \theta } ,$ , the verifiable reward is given as a binary number by a deterministic outcome verifier OV: $r = \stackrel { \cdot } { O V } ( \mathbf { x } , \mathbf { y } ) \in \{ 0 , 1 \}$ , where $r = 1$ if and only if the final answer is exactly correct $( \mathrm { e . g . }$ , the numeric result is mathematically identical to the ground truth answer) and $r = 0$ otherwise. In practice, an auxiliary format reward can be included to encourage the model to present its answer in a prescribed style.

## 4 Methodology: Reinforcing Reasoning with Self-Verification (RISE)

To address the challenge of superficial self-reflection, we propose RISE for self-improving reasoners, which is a scalable online RL method with explicit verification objective. The key idea of RISE is the use of the verifiable reward signal from the rule-based outcome verifier to align the model’s verification ability on-the-fly. This enables us to teach the model to verify its own response at the same time it solves the problem, as depicted in Figure 1 and Algorithm 1.

## 4.1 Online Reasoning and Verification

Problem Solution Generation Given an initial model $\pi _ { \theta }$ and a training set $D = \{ ( \mathbf { x _ { i } } , \mathbf { y } _ { i } ^ { * } ) \}$ consisting of problems $\mathbf { x } _ { i } ,$ and their corresponding ground-truth answers $\mathbf { y } _ { i } ^ { * }$ , we begin each RL iteration by sampling a data batch. At iteration $t ,$ the model first generates k solutions for each problem in the batch, each comprising a chain-of-thought reasoning followed by a final answer.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 RISE (PPO)

Input Language model  $\pi_{\theta_{init}}$ : outcome verifier OV; dataset D; rollout number K; generation batch size  $B_{g}$ , verification batch size  $B_{v}$ ; verification prompt template T; total iteration N.

1: Initialize: actor  $\pi_{\theta} \leftarrow \pi_{\theta_{init}}$ , old-actor  $\pi_{\theta_{old}}$ , critic  $\pi_{\phi}$ , reference  $\pi_{ref}$ 

2: for iteration = 1 to N do

3: Sample  $B_{g}$  samples for generation  $\mathcal{P}_{g} = \{(\mathbf{x}_{i}, \mathbf{y}_{i}^{*})\}_{i=1}^{\mathcal{B}_{g}} \sim \mathcal{D}$ 

4: Get generation batch: ▷ Generate solutions

 $\mathcal{G} \leftarrow \{(\mathbf{x}_{i}, \mathbf{y}_{i}^{(k)}, r_{\mathrm{ov}}(\mathbf{y}_{i}^{(k)}, \mathbf{y}_{i}^{*})) \mid \mathbf{y}_{i}^{(k)} \sim \pi_{\theta}(\cdot | \mathbf{x}_{i}), i \leq \mathcal{B}_{g}, k \leq K\}$ 

5: Select  $B_{v}$  triples  $P' = \{(\mathbf{x}_{i}, \mathbf{y}_{i}, r_{i})\}_{i=1}^{\mathcal{B}_{v}} \subseteq G$  for verification

6:  $\mathcal{P}_{v} \leftarrow \{(\mathcal{T}(\mathbf{x}, \mathbf{y}), r) \mid (\mathbf{x}, \mathbf{y}, r) \in \mathcal{P}'\} \quad // each element is a new prob-ans tuple (\mathbf{x}, \mathbf{y}^{*})$ 

7: Get verification batch: ▷ Verify generations

 $\mathcal{V} \leftarrow \{(\mathbf{x}_{j}, \mathbf{y}_{j}^{(k)}, r_{\mathrm{ov}}(\mathbf{y}_{j}^{(k)}, \mathbf{y}_{j}^{*})) \mid \mathbf{y}_{j}^{(k)} \sim \pi_{\theta}(\cdot | \mathbf{x}_{j}), j \leq B_{v}, k \leq K\}$ 

8: Get complete training batch  $B \leftarrow G \cup V$ 

9: Estimate advantages  $\hat{A}$  using Eq. (3) ▷ Joint optimization

10: Update critic  $\pi_{\phi}$  by critic loss in Eq. (4)

11: Update actor  $\pi_{\theta}$  by actor loss in Eq. (2); update  $\theta_{old} \leftarrow \theta$ 

12: end for

Output Optimized actor model  $\pi_{\theta}$
</div>

Next, the reward is computed for each generated response. Following prior RLVR approaches, we define a rule-based outcome verifier (OV) that incorporates both answer and format correctness:

$$
r _ {o} (\mathbf {y}, \mathbf {y} ^ {*}) = \left\{ \begin{array}{l l} 1, & \text {   boxed   and   matched   } \\ - 0. 5, & \text {   boxed   but   not   matched   } \\ - 1, & \text {   unboxed   } \end{array} \right.
$$

Here “matched” means the final answer in the generated solution y is mathematically identical to the provided ground truth $\mathbf { y } ^ { * }$ , and “boxed” means the final answer in y is wrapped in the \boxed{}.

This produces the generation batch $\mathcal { G } = ( \mathbf { x } , \mathbf { y } , r )$ , where each element includes the input problem, a model-generated solution, and its associated reward.

Online Solution Verification To construct verification data, we apply a fixed prompt template (Figure 8) to ${ \mathcal { G } } ,$ , formatting the problem-solution pair into a new verification prompt $\pmb { x } _ { \mathrm { v e r } }$ that explicitly states the verification criteria and asks the model to critique the provided solution and assign a score. Since the criteria used in the prompt are exactly those employed by the outcome verifier, the original reward r from the generation phase is reused as the ground-truth score for the verification task. Thus, for each triple $( \mathbf { x } , \mathbf { y } , r ) \in { \mathcal { G } }$ , we construct the verification data as $( \mathbf { x } _ { \mathrm { v e r } } = \mathcal { T } ( \mathbf { x } , \mathbf { y } ) , \mathbf { y } _ { \mathrm { v e r } } ^ { * } = r )$ . In practice, the amount of verification data is controlled by the verification batch size.

For each verification prompt, the model generates K responses, each containing a natural language critique and a final score. These responses are evaluated using the same $\boldsymbol { \mathrm { O V } }$ criteria, where the reward is determined by whether the extracted score from the model’s response matches the groundtruth score. This process yields the verification batch ${ \boldsymbol { \nu } } = ( \mathbf { x } , \mathbf { y } , r )$ , maintaining the same structure.

## 4.2 RL Integration

The preceding Online Reasoning and Verification stage is architecturally agnostic to the choice of the underlying policy-gradient algorithm; its only algorithm-specific interface is the advantage estimator $\hat { A }$ used in the policy update. In our formulation, advantage values are computed from a concatenated mini-batch $B ^ { ^ { \bullet } } = \mathcal { G } \cup \dot { \mathcal { V } }$ , which aggregates rewards from both the generation and verification tasks. Since every sample in B is annotated with a scalar reward and the action log-probability under the current policy $\pi _ { \theta } ,$ , any estimator that maps a sequence of state–action–reward tuples to an advantage can be incorporated without further structural change.

For our main experiments with PPO (see Algorithm 1), we apply GAE (Eq. 3) independently to each trajectory. The generation and verification trajectories are jointly processed within the same stochastic gradient descent (SGD) step, enabling the actor to be optimized with respect to both types of data. Meanwhile, the shared critic learns a unified value function across tasks. PPO’s clipping mechanism further ensures that updates remain stable within a consistent trust region.

## 5 Experiment

## 5.1 Experiment Setup

Dataset We follow the previous study (Zeng et al., 2025) to utilize MATH-Hard (Level 3–5) (Hendrycks et al., 2021) as our training set, which in total comprising 8,523 problems. This training set is used for all SFT baselines, Zero-RL baselines, and RISE models.

Models We conduct our RL training experiments on three Qwen2.5 models (Yang et al., 2024) with different sizes (i.e., 1.5B, 3B, and 7B) for their strong reasoning capabilities. The RL training of our models is based on the verl (Sheng et al., 2024) framework with a train batch size of 1024 and a mini-batch size of 128. We follow (Zeng et al., 2025) by setting the sampling temperature to 1.0 and rollout 8 responses for each problem. The RISE models have a default verification batch size 128. We set the RL configurations same for RISE models and Zero-RL models, ensuring a fair comparison.

Benchmarks We evaluate model performance on standard mathematical reasoning benchmarks: MATH500 (Hendrycks et al., 2021; Lightman et al., 2023), Minerva Math (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), and competition-level benchmarks AIME 2024 and AMC 2023. Following (Zeng et al., 2025), we generate 8 responses per problem using a sampling temperature of 1.0, and report Pass@1 accuracy (Chen et al., 2021) as the evaluation metric. Reasoning correctness is based on exact match of the final answer, and verification correctness depends on exact match between the predicted verification score and the score from the outcome verifier.

## 5.2 Experimental Results

Table 1 presents the results of RISE across model sizes and benchmarks.

RISE significantly enhances self-verification capabilities while improving reasoning performance. RISE models consistently outperform their Zero-RL counterparts across both reasoning and selfverification tasks on all model sizes. The improvement in self-verification is particularly dramatic - RISE-1.5B achieves 74.5% average verification accuracy compared to just 26.8% for Zero-RL, representing a 47.7 percentage point improvement. This demonstrates that our integrated approach successfully develops robust self-verification skills while simultaneously enhancing problem-solving capabilities. Notably, the verification improvements are particularly pronounced on the most challenging benchmarks like AIME24 and OlympiadBench, suggesting that RISE enables models to better recognize their limitations and errors on difficult problems.

Scaling model size improves reasoning performance while maintaining strong verification capabilities. Scaling model size from 1.5B to 7B parameters consistently enhances reasoning performance across all benchmarks. Interestingly, the verification performance of RISE models remains consistently high across model sizes, with all models achieving over 69% average accuracy. The ability to maintain strong verification capabilities while scaling reasoning performance aligns with our contribution of developing a framework that simultaneously improves both critical capabilities.

RISE models outperform standard SFT and base models by a substantial margin. The results clearly demonstrate that RISE models substantially outperform their SFT and base model counterparts. For instance, RISE-7B achieves 42.9% average reasoning accuracy compared to just 11.3% for

Table 1: Detailed results of RISE and other baseline methods on various math benchmarks. Zero-RL models are trained with the setting with RISE except the verification objective.

<table><tr><td rowspan="2">Model</td><td colspan="6">Reasoning</td><td colspan="6">Self-Verification</td></tr><tr><td>MATH</td><td>AIME</td><td>AMC</td><td>Mine.</td><td>Olym.</td><td>Avg.</td><td>MATH</td><td>AIME</td><td>AMC</td><td>Mine.</td><td>Olym.</td><td>Avg.</td></tr><tr><td>GPT-4o</td><td>79.0</td><td>13.3</td><td>55.0</td><td>50.0</td><td>42.5</td><td>48.0</td><td>83.4</td><td>33.3</td><td>67.5</td><td>50.4</td><td>54.4</td><td>57.8</td></tr><tr><td colspan="13">Qwen2.5-1.5B</td></tr><tr><td>Base</td><td>2.0</td><td>0.0</td><td>1.9</td><td>0.8</td><td>0.6</td><td>1.1</td><td>19.4</td><td>21.9</td><td>22.7</td><td>15.9</td><td>21.1</td><td>20.2</td></tr><tr><td>Instruct</td><td>37.5</td><td>0.8</td><td>19.4</td><td>8.3</td><td>11.7</td><td>15.5</td><td>48.8</td><td>22.1</td><td>36.5</td><td>36.9</td><td>29.6</td><td>34.8</td></tr><tr><td>SFT</td><td>10.1</td><td>0.0</td><td>4.1</td><td>1.8</td><td>2.0</td><td>3.6</td><td>19.0</td><td>5.8</td><td>12.3</td><td>10.5</td><td>10.9</td><td>11.7</td></tr><tr><td>Zero-RL</td><td>55.3</td><td>2.1</td><td>25.9</td><td>17.4</td><td>19.5</td><td>24.0</td><td>54.1</td><td>5.0</td><td>30.7</td><td>21.0</td><td>23.0</td><td>26.8</td></tr><tr><td>RISE-1.5B</td><td>54.6</td><td>2.9</td><td>27.5</td><td>17.2</td><td>19.8</td><td>24.4</td><td>75.9</td><td>85.0</td><td>70.6</td><td>66.0</td><td>74.9</td><td>74.5</td></tr><tr><td colspan="13">Qwen2.5-3B</td></tr><tr><td>Base</td><td>32.7</td><td>1.3</td><td>15.3</td><td>10.3</td><td>10.7</td><td>14.1</td><td>39.5</td><td>13.6</td><td>22.5</td><td>29.9</td><td>21.2</td><td>25.3</td></tr><tr><td>Instruct</td><td>61.0</td><td>3.8</td><td>34.1</td><td>25.6</td><td>24.6</td><td>29.8</td><td>65.6</td><td>21.0</td><td>45.5</td><td>37.6</td><td>35.0</td><td>40.9</td></tr><tr><td>SFT</td><td>14.4</td><td>0.4</td><td>5.3</td><td>2.9</td><td>2.8</td><td>5.2</td><td>21.5</td><td>2.1</td><td>10.9</td><td>17.9</td><td>13.2</td><td>13.1</td></tr><tr><td>Zero-RL</td><td>64.2</td><td>6.7</td><td>37.5</td><td>27.4</td><td>26.6</td><td>32.5</td><td>64.9</td><td>13.0</td><td>39.7</td><td>30.3</td><td>31.2</td><td>35.8</td></tr><tr><td>RISE-3B</td><td>64.3</td><td>7.9</td><td>42.5</td><td>26.2</td><td>26.6</td><td>33.5</td><td>81.0</td><td>86.3</td><td>74.4</td><td>56.1</td><td>73.6</td><td>74.3</td></tr><tr><td colspan="13">Qwen2.5-7B</td></tr><tr><td>Base</td><td>38.3</td><td>2.1</td><td>21.9</td><td>11.9</td><td>13.2</td><td>17.5</td><td>58.4</td><td>45.9</td><td>51.5</td><td>48.4</td><td>48.4</td><td>50.5</td></tr><tr><td>Instruct</td><td>73.8</td><td>10.0</td><td>50.6</td><td>35.9</td><td>35.8</td><td>41.2</td><td>77.2</td><td>26.3</td><td>57.0</td><td>40.2</td><td>45.2</td><td>49.2</td></tr><tr><td>SFT</td><td>28.7</td><td>0.8</td><td>13.8</td><td>6.2</td><td>7.2</td><td>11.3</td><td>40.5</td><td>36.6</td><td>47.4</td><td>39.2</td><td>36.1</td><td>40.0</td></tr><tr><td>Zero-RL</td><td>74.5</td><td>12.1</td><td>51.3</td><td>34.2</td><td>36.7</td><td>41.7</td><td>75.9</td><td>21.7</td><td>56.5</td><td>37.3</td><td>41.6</td><td>46.6</td></tr><tr><td>RISE-7B</td><td>74.8</td><td>12.5</td><td>55.9</td><td>34.6</td><td>36.7</td><td>42.9</td><td>83.8</td><td>75.0</td><td>72.5</td><td>48.6</td><td>65.9</td><td>69.2</td></tr></table>

SFT-7B and 17.5% for the base model. This dramatic improvement highlights the effectiveness of our integrated RL approach that incorporates both problem-solving and self-verification objectives.

## 5.3 Test-Time Scaling with Self-verification

(a) k=4

(b) k=8  
<sup>Gen.</sup> <sup>Verifier</sup> <sup>(GPT-4o)Gen.</sup> <sup>Verifier</sup> <sup>(GPT-4o)</sup>Figure 2: Test-time scaling performance across different sampling budgets (“k”).

To further evaluate the benefits of the enhanced self-verification capabilities developed by RISE, c<sup>y</sup> 70 c<sup>y</sup> 70 we investigate its impact at test-time using self-consistency majority voting (“maj@k”) (Wang et al., c<sup>u</sup> 65.665.5c<sup>u</sup> 65.665.52023b) and verification-weighted majority voting. In the latter, following (Wang et al., 2023a), <sup>A 60</sup> <sup>A 60</sup>the model’s self-generated verification scores for each candidate solution are used to weight its ti<sup>o</sup> 52.9ti<sup>o</sup> 52.9contribution in the majority vote. The results, presented in Figure 2, compare RISE models against f<sup>ic 49.0</sup>f<sup>ic</sup>Zero-RL models across different sampling budgets (“k=4” and “k=8”).

RISE consistently improves test-time scaling performance with self-verification and majority voting. RISE models outperform their Zero-RL counterparts when employing test-time strategies such as majority voting and verification-weighted selection. Across model sizes and sampling budgets, RISE achieves higher average accuracy, with the largest relative gains observed when self-verification scores are used to re-rank majority votes. For example, RISE-7B achieves an average score of 49.8% with k = 8 + self-verify, surpassing Zero-RL’s 48.3% under the same conditions. This consistent improvement substantiates the effectiveness of integrating self-verification during both training and inference, fulfilling our objective of developing more robust and self-aware reasoners.

Verification-weighted voting delivers further accuracy gains. Incorporating self-verification scores as weights in the voting process leads to additional accuracy improvements for all RISE models. For instance, RISE-3B and RISE-7B models see improvements of +0.2% and +1.9% over standard majority voting at the k = 4 budget, respectively. These results indicate that the selfverification policy learned by RISE provides meaningful confidence signals for answer calibration.

## 5.4 Comparison with Off-the-shelf Verifiers

We further compare the verification accuracy between our RISE models as self-verifiers and off-the-shelf verifiers, including a discriminative verifier (Math-Shepherd-7B (Wang et al., 2023a)) and a generative verifier (GPT-4o (OpenAI, 2024)). Specifically, we use the verification prompt in Figure 8 for both RISE models and GPT-4o and adhere to the original logic for Math-Shepherd to verify the generated solutions. The results of RISE-1.5B, 3B and 7B are presented in Figure 3, which show that RISE models consistently outperform existing outcome<sup>41.9</sup> <sup>41.7</sup>  <sup>41.8</sup> verifiers in judge their solutions’ correctness. This serves<sup>y</sup> as a great advantage for the model to further improve itsu<sup>r</sup> 35 <sup>33.5</sup> 32.5 test-time performance, by leveraging the self-verification <sup>A</sup> signal either externally or internally. Detailed results andni evaluation implementation can be found in Appendix D.a<sup>s</sup>

Figure 3: Comparisons between RISEti<sup>o</sup> (self-verify) and off-the-shelf verifiers.f<sup>i</sup>

## 5.5 Analysis

In this section, we provide some insights into how RISE improves performance.<sup>Verification</sup> <sup>Data</sup> <sup>Portion</sup> <sup>(%) Ve</sup>

RISE demonstrates robust and simultaneous learning of problemsolving and self-verification, with self-verification skills developing notably faster across different model scales. The learning curves, illustrated by the reward trends in Figure 4, reveal a consistent and steady improvement in both reasoning (problem-solving) and selfverification rewards throughout the RL training process for all evaluated models. This uniform positive progression across varying model sizes

Figure 4: Reasoning and verification reward at train time.

highlights the robustness of the RISE framework in co-training these two abilities, a core contribution of our work. A key observation is that the self-verification reward generally exhibits a more rapid increase and reaches a higher relative level compared to the problem-solving reward within the same training duration. This aligns with the “Generation-Verification Gap” posited by Song et al.

(2024), suggesting that models might acquire verification capabilities more readily than complex reasoning.

Impact of Verification Compute In the main experiment, we trained our RISE models with a verification batch size of 128, which is 12.5% of the generation batch 1024. We further explore the model performance by scaling up the verification data batch, i.e., the train-time compute, up to 100% of the generation batch. In practice, we choose the percentages S of {0, 12.5%, 25%, 50%, 100%} and perform experiment on our RISE models. The results are shown in Figure 5. The problem-solving performance first in-


Figure 5: Impact of verification data ratio.

creases, and then slightly decreases, and finally increases across the benchmark, maintaining at a <sub>a</sub>nhigh level. Furthermore, the verification performance keeps scaling with more training compute, Mindicating the robustness of scalability of our RISE method.

<sup>R</sup>Online and Offline Verification We validate <sup>.6 49.5</sup>48.3<sup>Zero-RL</sup>the effectiveness of online verification by com--1c<sup>y</sup> RISE + Self-Verifyparing it to a offline variant, where the verifica-K=8 <sup>0</sup> u<sup>r</sup> 40 40.7tion data are collected from a distant policy and Adirectly added to the training set. In practice,.6 <sup>49.849.5</sup>48.3<sup>Zero-RL</sup> i<sup>n</sup>we select the policy at step 96 (final step) of the<sub>c</sub><sup>y</sup> <sub>RISE</sub> <sub>+</sub> <sub>Self-Verify</sub> s<sup>o 30 31.4</sup>Zero-RL model and use its generated responsesu<sup>r</sup> 40 R<sup>e</sup>to construct offline verification set. In the ex-A periment, we keep the portion of verificationn <sup>20</sup>data and the training batch size same to elim-<sup>o</sup> 30 <sup>31.731.4</sup> inate other influence factors, making the only<sub>R</sub><sup>e</sup> changing variable the source of the verification data. Figure 6 shows the results. While the<sup>20</sup> <sup>Gen.</sup> <sup>Verifier</sup> <sup>(GPT-4o)</sup>problem-solving performance of offline verifica-


Figure 6: Comparisons between online and offline<sup>24.4</sup> <sub>25.3</sub> verification.

RISE Self-Verifytion models are on par with the online ones, they have a significant drop in terms of self-verification accuracy, which indicates the importance of online verification designed in our RISE method.

<sup>ur</sup> 65.665.5RISE Self-VerifyEnhanced Verification for Reasoning Besides A <sup>60</sup>leveraging the self-verification ability externally i<sup>oy 74.5</sup> <sup>74.3</sup>during the test-time as in § 5.3, such ability is i<sup>c 50.1</sup> <sup>49.0</sup>u<sup>r</sup>also internalized by the model to enhance its e<sup>r</sup> 40A <sup>60</sup>reasoning generation process. To analysis this o<sup>n</sup>effect from the quantitative perspective, we mea-<sup>30</sup>c<sup>a 50.1</sup> 49.0sure the Verification Frequency and Self-Verified e<sup>ri</sup> <sub>40</sub>Reasoning Accuracy in models problem-solving process. Inspired by (Yeo et al., 2025), we use <sup>30</sup>a set of verification keywords to select the responses containing self-verification behaviors, namely {“verify”, “verifying”, “recheck”, “validate”, “re-evaluate”}.


Figure 7: Verification frequency (left panel) and its effectiveness in reasoning (right panel).

Figure 7 presents the results, where the propor-

tion of responses that contain an explicit verification phrase consistently rises after RISE training. The increase is modest for the 1.5 B model (+0.05 absolute) but becomes substantial as scale grows (+1.09 for 3 B and +1.05 for 7 B). Because both systems share the same decoding hyper-parameters, the higher frequency stems from the policy update that jointly optimizes for solution reward and verification reward, validating our first contribution: RISE integrates verification into the RL loop instead of treating it as a post-hoc heuristic.

Simply asking the model to “verify” is not sufficient; the verification needs to be correct. RISE yields clear improvements over Zero-RL for self-verified reasoning accuracy at every scale: +3.3% (1.5 B), +1.7% (3 B), and a striking +6.5% (7 B). These gains show that RISE’s online verifier reward shapes the policy toward not only producing more verifications, but also ones that align with ground truth.

The case in Table 2 illustrates this distinction. Zero-RL “verifies” 222 by merely restating divisibility rules, overlooking the necessity of a trailing zero for multiples of 30. RISE, in contrast, recomputes both the units-digit constraint and the digit-sum divisibility test, ultimately validating the answer of 2220. Such structured, multi-step verification reflects a genuinely internalized skill and explains the quantitative trend that higher verification frequency correlates with higher reasoning accuracy.

Table 2: Examples of verification behaviors of different methods on Qwen2.5-7B: RISE can incur more effectiveness self-verification in the reasoning.

<table><tr><td>Problem</td><td>What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?</td></tr><tr><td>Zero-RL</td><td>To find the least positive integer multiple of 30 that can be written with only the digits 0 and 2, we need to consider the properties of multiples of 30 ... Since 222 is divisible by both 2 and 3, it is a multiple of 30. Let&#x27;s verify: 222 is even, so it is divisible by 2. The sum of the digits of 222 is  $2 + 2 + 2 = 6$ , which is divisible by 3. Thus, 222 is indeed a multiple of 30. Therefore, the least positive integer multiple of 30 that can be written with only the digits 0 and 2 is [222].</td></tr><tr><td>RISE</td><td>To find the least positive integer multiple of 30 that can be written with only the digits 0 and 2, we need to ... Therefore, the smallest number N is 2220. To verify: 2220 ends in 0, so it is a multiple of 10. The sum of the digits of 2220 is  $2 + 2 + 2 + 0 = 6$ , which is a multiple of 3. Since 2220 satisfies both conditions, it is the least positive integer multiple of 30 that can be written with only the digits 0 and 2. Thus, the final answer is [2220].</td></tr></table>

## 6 Conclusion

In this work, we introduced RISE, a novel online reinforcement learning framework that integrates problem-solving with explicit self-verification training for LLMs. By leveraging verifiable rewards for both generation and verification tasks within a unified RL objective, RISE aims to overcome superficial self-reflection and foster more robust reasoning capabilities. Our experiments, primarily using PPO with Qwen2.5 models on diverse mathematical reasoning benchmarks, demonstrate that RISE significantly improves problem-solving accuracy while concurrently developing strong self-verification skills. Notably, RISE models learned to verify their own on-policy generations more effectively than off-the-shelf verifiers. Together, RISE provides a promising direction for building more reliable and self-aware LLM reasoners, adaptable to various policy-gradient algorithms and extendable to other domains with verifiable rewards.

Future work includes: (1) exploring other challenging reasoning domains beyond mathematical reasoning, such as code generation, physics reasoning, and real-world reasoning-intensive domains (Su et al., 2025); (2) examining additional policy-gradient algorithms besides PPO (such as GRPO (Shao et al., 2024) and RLOO (Ahmadian et al., 2024)) within the RISE framework; and (3) investigating potential synergies between RISE and complementary enhancements like retrieval-augmented generation (RAG) or external tools for knowledge verification.

## References

A. Ahmadian, C. Cremer, M. Galle, M. Fadaee, J. Kreutzer, O. Pietquin, A.´ Ust<sup>¨</sup> un, and S. Hooker.¨ Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

A. G. Barto, R. S. Sutton, and C. W. Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics, SMC-13(5):834–846, 1983. doi: 10.1109/TSMC.1983.6313077.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

J. Gao, S. Xu, W. Ye, W. Liu, C. He, W. Fu, Z. Mei, G. Wang, and Y. Wu. On designing effective rl reward at training time for llm reasoning. ArXiv, abs/2410.15115, 2024.

C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

J. Hu, Y. Zhang, Q. Han, D. Jiang, X. Zhang, and H.-Y. Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.

A. Huang, A. Block, D. J. Foster, D. Rohatgi, C. Zhang, M. Simchowitz, J. T. Ash, and A. Krishnamurthy. Self-improvement in language models: The sharpening mechanism. arXiv preprint arXiv:2412.01951, 2024.

A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

N. Lambert, J. D. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, X. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi. Tulu 3: Pushing frontiers in open¨ language model post-training. ArXiv, abs/2411.15124, 2024.

A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Z. Lin, S. Shen, J. Shang, J. Weston, and Y. Nie. Learning to solve and verify: A self-play framework for code and test generation. arXiv preprint arXiv:2502.14948, 2025.

Z. Liu, C. Chen, W. Li, T. Pang, C. Du, and M. Lin. There may not be aha moment in r1-zero-like training — a pilot study. https://oatllm.notion.site/oat-zero, 2025. Notion Blog.

OpenAI. Gpt-4o, 2024. URL https://openai.com/index/hello-gpt-4o/. Accessed: 2024-06-15.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

W. Shi and X. Jin. Heimdall: test-time scaling on the generative verification. arXiv preprint arXiv:2504.10337, 2025.

Y. Song, H. Zhang, C. Eisenach, S. Kakade, D. Foster, and U. Ghai. Mind the gap: Examining the self-improvement capabilities of large language models. arXiv preprint arXiv:2412.02674, 2024.

Y. Su, D. Yu, L. Song, J. Li, H. Mi, Z. Tu, M. Zhang, and D. Yu. Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains, 2025. URL https://arxiv.org/ abs/2503.23829.

R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

K. Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

S. Toshniwal, W. Du, I. Moshkov, B. Kisacanin, A. Ayrapetyan, and I. Gitman. OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data. In ICLR, 2025.

L. von Werra, Y. Belkada, L. Tunstall, E. Beeching, T. Thrush, N. Lambert, S. Huang, K. Rasul, and Q. Gallouedec. Trl: Transformer reinforcement learning.´ https://github.com/huggingface/trl, 2020.

P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y.Wu, and Z. Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. ArXiv, abs/2312.08935, 2023a.

X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023b.

Wikipedia contributors. P versus np problem — Wikipedia, the free encyclopedia, 2025. URL https: //en.wikipedia.org/w/index.php?title=P\_versus\_NP\_problem&oldid=1287287306. [Online; accessed 13-May-2025].

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

Z. Xi, D. Yang, J. Huang, J. Tang, G. Li, Y. Ding, W. He, B. Hong, S. Do, W. Zhan, et al. Enhancing llm reasoning via critique models with test-time and training-time supervision. arXiv preprint arXiv:2411.16579, 2024.

Z. Xie, L. Chen, W. Mao, J. Xu, L. Kong, et al. Teaching language models to critique via reinforcement learning. arXiv preprint arXiv:2502.03492, 2025.

W. Xiong, H. Zhang, C. Ye, L. Chen, N. Jiang, and T. Zhang. Self-rewarding correction for mathematical reasoning. arXiv preprint arXiv:2502.19613, 2025.

A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

E. Yeo, Y. Tong, M. Niu, G. Neubig, and X. Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.

W. Yuan, R. Y. Pang, K. Cho, S. Sukhbaatar, J. Xu, and J. E. Weston. Self-rewarding language models. ArXiv, abs/2401.10020, 2024. URL https://api.semanticscholar.org/CorpusID:267035293.

Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, Y. Yue, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. URL https: //arxiv.org/abs/2504.13837.

E. Zelikman, Y. Wu, J. Mu, and N. Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

W. Zeng, Y. Huang, Q. Liu, W. Liu, K. He, Z. Ma, and J. He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/ abs/2503.18892.

L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, and R. Agarwal. Generative verifiers: Reward modeling as next-token prediction, 2025. URL https://arxiv.org/abs/2408.15240.

## A Limitations

On experiments: After training the LLM to self-verify its outputs, it may serve as a pseudo–rulebased verifier to further guide RL training without relying on ground-truth labels. This opens the possibility of self-improvement on unlabeled data, which we leave for future exploration as it is beyond the current scope. Our experiments focus exclusively on math reasoning tasks across diverse problem types. While the generalization of the method to other domains remains underexplored, we argue that its strong performance on math reasoning is a compelling demonstration of its effectiveness. RISE is expected to generalize as long as a well-defined verifiable reward is accessible.

On algorithm: RISE trains the LLM as a generative verifier that produces natural language critiques, which has shown benefits for both reasoning and test-time scaling. An alternative design is to train a discriminative verifier with a separate classification head. While it remains unclear how RISE would perform in that setting, we believe this does not affect our main contributions which demonstrate the effectiveness of generative verification in improving problem-solving capabilities.

## B Prompt Templates

Figure 8: Verification prompt used in the experiment.

## C Training Details

During RL training, we set the actor’s clipping ratio to 0.2 and disable the KL penalty loss. The critic uses a clipping range of 0.5. The learning rates are fixed at $5 \times 1 0 ^ { - 7 }$ for the actor and $9 \times 1 0 ^ { - 6 }$ for the critic. The KL divergence coefficient is set to $1 \times 1 0 ^ { - 2 }$ . We limit the maximum response length to 3000 tokens, which already results in a negligible clip ratio. The full dataset is trained for 12 epochs. This configuration is shared across both the Zero-RL and RISE models.


Figure 9: Prompt template used in the training and evaluation.  
```txt
Prompt Template
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{Input} Please reason step by step, and put your final answer within
\\boxed{}.<|im_end|>
<|im_start|>assistant
```  
Figure 10: Prompt template used for Qwen base model evaluation.

For the SFT baseline models, we use a batch size of 32 and apply a cosine learning rate scheduler with a learning rate of $2 \times 1 0 ^ { - 5 }$ and a warm-up ratio of $1 \times 1 0 ^ { - \hat { 3 } }$ . The dataset is trained for 3 epochs.

## D Evaluation Details

## D.1 Verification Evaluation with Other Verifiers

To evaluate the verification accuracy of RISE and GPT-4o (prompted as a verifier), we extract the final verification score from each response and normalize it to either +1 (predicted correct) or 0 (predicted incorrect). The normalization is defined as:

$$
s _ {\text { normalized }} = \left\{ \begin{array}{l l} 1, & s = 1 \\ 0, & \text { otherwise } \end{array} \right.
$$

which aligns with the criteria used by the rule-based outcome verifier. For the Math-Shepherd model, which outputs a continuous score in the range [0, 1] (with 0 indicating the solution/step is predicted to be incorrect and 1 indicating correct), we apply a threshold of 0.5 for normalization:

$$
s _ {\text { normalized }} = \left\{ \begin{array}{l l} 1, & s > 0. 5 \\ 0, & \text { otherwise } \end{array} \right.
$$

After normalization, we compute verification accuracy by directly comparing the predicted scores against those returned by the outcome verifier.

## D.2 Weighted Majority Voting with Self-Verification

In § 5.3, we explore the combination of self-consistency and self-verification in test time following (Wang et al., 2023a). In practice, we initially classify solutions into distinct groups according to their final answers. Following that, we extract and normalize the self-verification scores and normalize them as +1 (correct) and 0 (incorrect) as in D.1. Since our score are binary and could lead to an unexpected zero sum, we integrate Laplace smoothing for computing the mean score for the answer. Formally, the final selected answer based on N candidate solutions is:

$$
a _ {\text {maj@N + self - verify}} = \operatorname{argmax} _ {a} \underbrace {\sum_ {i = 1} ^ {N} \mathbb {I} (a _ {i} = a)} _ {\text {frequency}} \cdot \underbrace {\frac {\alpha + \sum_ {i = 1} ^ {N} S (p , s _ {i})}{N + \alpha d}} _ {\text {smoothed mean score}}.\tag{5}
$$

where $S ( p , S _ { i } )$ is the score of the i-th solution assigned by the self-verification. In practice, we set $\alpha = 2$ and $d = 2$ empirically, suggesting a prior belief of a 0.5 average score.

## E Detailed Experiment Results

## E.1 Detailed Comparison with off-the-shelf verifiers

In § 5.4, we report the average verification accuracy across the five benchmarks. Here, we present the detailed verification accuracy comparison between RISE models, Math-Shepherd, and GPT-4o on each evaluation benchmark.

Figure 11: Detailed comparisons of verification accuracy between RISE-1.5B and other verifiers.

## E.2 Detailed Analysis for Enhanced Verification

In Figure 7, we report the average verification frequency and accuracy of self-verified solutions on the five benchmarks. Here, we present the fine-grained results between RISE models and Zero-RL baseline on each evaluation benchmark.

## E.3 Reflection Keywords Analysis

Following Yeo et al. (2025), we track the self-reflection keywords {“wait”, “however”, “alternatively”, “retry”, “recheck”} to quantitatively measure the general reflection behaviors beyond the selfverification among the model problem-solving responses. In practice, we sum the total word counts for the keyword set and normalize it by the dataset size.

The results in Table 4 show that our RISE model constantly have a higher level of reflection-related behaviors than the Zero-RL models, indicating the positive effect of self-verification training.

Figure 12: Detailed comparisons of verification accuracy between RISE-3B and other verifiers.

Figure 13: Detailed comparisons of verification accuracy between RISE-7B and other verifiers.

## F Licenses

Datasets and Benchmarks. The training dataset is derived from MATH (MIT License). We evaluate on five benchmarks: MATH 500 (MIT License), AIME 2024 (CC0: Public Domain), AMC 2023 (Apache License 2.0), Minerva Math (license not found), and Olympiad Bench (MIT License).

Framework. RL training is based on verl v0.2 (Apache-2.0 license), and SFT training is based on trl (von Werra et al., 2020) v0.14.0 (Apache-2.0 license). Evaluation is performed using vllm framework (Kwon et al., 2023) v0.7.2 (Apache-2.0 License) and the script is based on OpenMathInstruct-2 (Toshniwal et al., 2025).

Table 3: Performance comparison between RISE models and Zero-RL models on verification frequency and effectiveness for the generation.

<table><tr><td rowspan="2">Method</td><td colspan="6">Verification Frequency</td></tr><tr><td>MATH</td><td>AIME</td><td>AMC</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Qwen2.5-1.5B-Zero-RL</td><td>6.45</td><td>6.67</td><td>7.81</td><td>2.25</td><td>15.59</td><td>7.75</td></tr><tr><td>RISE-1.5B</td><td>7.10</td><td>8.75</td><td>5.31</td><td>2.53</td><td>15.31</td><td>7.80</td></tr><tr><td>Qwen2.5-3B-Zero-RL</td><td>4.90</td><td>8.33</td><td>14.29</td><td>2.99</td><td>8.72</td><td>7.85</td></tr><tr><td>RISE-3B</td><td>4.63</td><td>9.17</td><td>18.18</td><td>3.08</td><td>9.67</td><td>8.94</td></tr><tr><td>Qwen2.5-7B-Zero-RL</td><td>5.30</td><td>5.00</td><td>7.19</td><td>1.56</td><td>8.19</td><td>5.45</td></tr><tr><td>RISE-7B</td><td>6.08</td><td>7.92</td><td>8.13</td><td>1.79</td><td>8.57</td><td>6.50</td></tr><tr><td></td><td colspan="6">Self-Verified Solution Accuracy</td></tr><tr><td>Qwen2.5-1.5B-Zero-RL</td><td>37.21</td><td>0.00</td><td>24.00</td><td>24.49</td><td>15.59</td><td>20.26</td></tr><tr><td>RISE-1.5B</td><td>38.73</td><td>4.76</td><td>35.29</td><td>23.64</td><td>15.31</td><td>23.55</td></tr><tr><td>Qwen2.5-3B-Zero-RL</td><td>45.92</td><td>0.00</td><td>14.29</td><td>20.00</td><td>19.96</td><td>20.03</td></tr><tr><td>RISE-3B</td><td>43.78</td><td>4.55</td><td>18.18</td><td>22.39</td><td>19.54</td><td>21.69</td></tr><tr><td>Qwen2.5-7B-Zero-RL</td><td>62.74</td><td>0.00</td><td>8.70</td><td>35.29</td><td>26.70</td><td>26.68</td></tr><tr><td>RISE-7B</td><td>65.43</td><td>5.26</td><td>38.46</td><td>28.21</td><td>28.73</td><td>33.22</td></tr></table>

Table 4: Reflection Keywords Rate between RISE models and Zero-RL models.

<table><tr><td rowspan="2">Method</td><td colspan="6">Verification Frequency in Generation</td></tr><tr><td>MATH</td><td>AIME</td><td>AMC</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Qwen2.5-1.5B-Zero-RL</td><td>0.16</td><td>0.40</td><td>0.26</td><td>0.16</td><td>0.29</td><td>0.25</td></tr><tr><td>RISE-1.5B</td><td>0.19</td><td>0.45</td><td>0.29</td><td>0.16</td><td>0.32</td><td>0.28</td></tr><tr><td>Qwen2.5-3B-Zero-RL</td><td>0.14</td><td>0.40</td><td>0.24</td><td>0.11</td><td>0.27</td><td>0.23</td></tr><tr><td>RISE-3B</td><td>0.16</td><td>0.45</td><td>0.20</td><td>0.13</td><td>0.29</td><td>0.25</td></tr><tr><td>Qwen2.5-7B-Zero-RL</td><td>0.13</td><td>0.38</td><td>0.23</td><td>0.08</td><td>0.23</td><td>0.21</td></tr><tr><td>RISE-7B</td><td>0.14</td><td>0.50</td><td>0.29</td><td>0.10</td><td>0.27</td><td>0.26</td></tr></table>

Models. We train our models based on the Qwen2.5 series. Specifically, Qwen2.5-1.5B<sup>1</sup> and Qwen2.5- $7 \mathrm { B } ^ { 2 }$ are released under the Apache License $2 . 0 ,$ while Qwen2.5-3B<sup>3</sup> is released under a custom Qwen Research license. We also compare against Math-Shepherd<sup>4</sup> model (license not found), and GPT-4o (accessed via OpenAI API, governed by OpenAI Terms of Use<sup>5</sup>).

Figure 14: Example verification returned by RISE-7B on AIME 24.
