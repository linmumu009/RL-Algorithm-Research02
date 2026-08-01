# Reinforcing General Reasoning without Verifiers

Xiangxin Zhou<sup>∗23</sup>, Zichen Liu<sup>∗14</sup>, Anya Sims<sup>∗15</sup>, Haonan Wang<sup>14</sup>, Tianyu Pang<sup>1</sup>, Chongxuan Li<sup>6</sup>, Liang Wang<sup>23</sup>, Min Lin<sup>1</sup>, Chao Du<sup>†1</sup>

<sup>1</sup>Sea AI Lab, Singapore; <sup>2</sup>University of Chinese Academy of Sciences;

<sup>3</sup>Institute of Automation, Chinese Academy of Sciences;

<sup>4</sup>National University of Singapore; <sup>5</sup>University of Oxford; <sup>6</sup>Renmin University of China zhouxiangxin1998@gmail.com; {liuzc, tianyupang, linmin, duchao}@sea.com

## Abstract

The recent paradigm shift towards training large language models (LLMs) using DeepSeek-R1-Zero-style reinforcement learning (RL) on verifiable rewards has led to impressive advancements in code and mathematical reasoning. However, this methodology is limited to tasks where rule-based answer verification is possible and does not naturally extend to real-world domains such as chemistry, healthcare, engineering, law, biology, business, and economics. Current practical workarounds use an additional LLM as a model-based verifier; however, this introduces issues such as reliance on a strong verifier LLM, susceptibility to reward hacking, and the practical burden of maintaining the verifier model in memory during training. To address this and extend DeepSeek-R1-Zero-style training to general reasoning domains, we propose a verifier-free method (VeriFree) that bypasses answer verification and instead uses RL to directly maximize the probability of generating the reference answer. We compare VeriFree with verifier-based methods and demonstrate that, in addition to its significant practical benefits and reduced compute requirements, VeriFree matches and even surpasses verifier-based methods on extensive evaluations across MMLU-Pro, GPQA, SuperGPQA, and math-related benchmarks. Moreover, we provide insights into this method from multiple per spectives: as an elegant integration of training both the policy and implicit verifier in a unified model, and as a variational optimization approach. Code is available at https://github.com/sail-sg/VeriFree.

Qwen3-4B model performance  

Qwen3-8B model performance  
Figure 1: The general reasoning capability is significantly improved when we apply VeriFree to fine-tune Qwen3 base models on a general reasoning dataset. Notably, VeriFree can perform on par with or even surpass the instruct models and models RL-tuned from base with a specialized LLM as a verifier.

## 1 Introduction

DeepSeek-R1-Zero [10] recently demonstrated that training large language models (LLMs) using reinforcement learning (RL) with verifiable rewards can be extremely effective in improving reasoning capabilities. In this RL with verifiable rewards (RLVR) framework [17], the LLM generates a reason ing trace (i.e., chain of thoughts, CoT) followed by a final answer. A rule-based program then extracts and evaluates the final answer, assigning a reward of 1 to the response if the final answer is correct and 0 otherwise. The model is trained with RL using GRPO [37]—a simplified variant of PPO [36].

The simplicity of this approach, coupled with impressive performance improvements in mathematical reasoning tasks, has sparked a wave of follow-up works in this paradigm of RL with rule-based verifiable rewards [24, 26, 45], which we will refer to as the R1-Zero-style training in the following. However, these methods remain limited to domains such as mathematics and code, where rule-based verification is feasible. Reasoning is critical far beyond math and coding; however, the difficulty of answer verification in general reasoning tasks poses a major obstacle to applying this training paradigm to broader domains. To address this limitation, we investigate how to extend R1-Zero-style training to tasks where rule-based answer verification is not possible.

A natural extension, as explored in recent general reasoning works [38, 27], is to introduce a specialized LLM as a verifier, similar to the reward model used in RL from human feedback (RLHF) [51, 30]. In these methods, the model-based verifier is queried to determine whether the generated answer is equivalent to the reference answer. Although this approach bypasses the need for rule-based evaluation, it introduces several potential drawbacks (as in standard RLHF): it depends on the availability of a strong verifier LLM, it converts the R1-Zero-style paradigm into optimizing a model-based reward, which makes it vulnerable to reward hacking [7], and it adds significant computational overhead by requiring an additional model to be held in memory and queried during training.

In this work, we propose an alternative: a verifier-free approach that preserves the benefits of the RL paradigm while eliminating the reliance on explicit verification, either performed by rules or by models. Our method proceeds as follows. Given a question, we only generate the reasoning trace and concatenate it with the reference answer from the dataset. We then evaluate the likelihood of the reference answer conditioned on the question and the generated reasoning trace. This likelihood serves both as a reward signal for policy gradients on the reasoning trace and as a weighting term for supervised training of the reference answer. We term our method VeriFree since it does not rely on rule- or model-based verifiers, and give an illustration in Fig. 2.

This approach has several appealing properties. First, when there is a unique correct answer string, our method is equivalent in expectation to the objective in RLVR, but with lower variance, which can be viewed as a form of reward shaping [28, 34]. Even when multiple valid answers exist, we show empirically that using just one as a reference provides a sufficient learning signal to elicit strong reasoning behavior. Additionally, this framework can be viewed through a variational lens as a neat way of optimizing over latent reasoning traces.

To make this work in practice, we identify and address several subtle challenges, including effective variance reduction and precise handling of tokenization at the reasoning-answer patching point. We conduct comprehensive ablations to understand the impact of each design choice. We benchmark our method across a diverse set of general reasoning tasks, and the results are striking: as shown in Fig. 1, VeriFree not only matches but often outperforms verifier-based alternatives, while being simpler, faster, less memory-intensive, and more robust.

## 2 Methodology

## 2.1 Preliminaries: Verifier-Based Reinforcement Learning

In RL applied to LLMs, the language model is treated as a policy $\pi _ { \theta }$ that generates an output o autoregressively in response to an input question x. The goal is typically to optimize $\pi _ { \theta }$ to maximize a given reward function $R ( { \pmb x } , { \pmb o } )$

$$
\theta \in \arg \max _ {\theta} \mathbb {E} _ {\boldsymbol {o} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} [ R (\boldsymbol {x}, \boldsymbol {o}) ].\tag{1}
$$

In R1-Zero-style $\mathrm { R L }$ , the reward is computed by first parsing the response o into a reasoning trace z and a final answer y. A verifier then checks y against the ground-truth reference answer $\boldsymbol { y } ^ { \star }$ and assigns a binary reward based on correctness, namely $R _ { \mathrm { V e r i f i e r } } ( { \pmb y } ; { \pmb y } ^ { \star } ) = \mathbb { 1 } _ { \{ { \pmb y } \equiv { \pmb y } ^ { \star } \} } . ^ { 1 }$ Decomposing the model output as $\begin{array} { r } { \pmb { o } = ( z , \pmb { y } ) } \end{array}$ , we can rewrite the objective in Eq. (1) as:

Figure 2: VeriFree enables R1-Zero-style LLM training without requiring access to a verifier. In the case of a single correct answer format, VeriFree optimizes exactly the same objective as R1-Zero with a lower variance gradient estimator.

$$
J _ {\text { Verifier }} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} [ R _ {\text { Verifier }} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) ],\tag{2}
$$

which separates the sampling of the reasoning trace and the final answer. To maximize this objective, the model $\pi _ { \theta }$ is typically updated using the policy gradient estimator [39]:

$$
\nabla_ {\theta} J _ {\text {Verifier}} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) = \underset {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} {\mathbb {E}} \underset {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} {\mathbb {E}} \Big [ R _ {\text {Verifier}} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) \big [ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \big ] \Big ].\tag{3}
$$

However, this approach requires evaluating answer correctness via $R _ { \mathrm { V e r i f i e r } } ( { \pmb y } ; { \pmb y } ^ { \star } ) = { \mathbb 1 } _ { \{ { \pmb y } \equiv { \pmb y } ^ { \star } \} }$ , which is often nontrivial. While in domains such as math and code, this evaluation is feasible via rules [10, 24] or test cases [8], accurate verification in general reasoning tasks is substantially more difficult. As a result, recent advances in R1-Zero-style training have largely been restricted to verifiable domains, leaving reasoning tasks in general domains underexplored. In light of this, we present an alternative verifier-free approach which naturally extends this training paradigm to broader reasoning domains.

## 2.2 VeriFree Policy Optimization

We begin with the standard objective in $\operatorname { E q . } \left( 2 \right)$ and show that, in the case of a single correct answer, we can derive an equivalent objective that does not require a verifier. Moreover, we compare the gradient estimators of this new objective and the verifier-based objective (Eq. (3)), and demonstrate that our verifier-free gradient estimator has the additional benefit of lower variance.

Starting from Eq. (2) and assuming a unique correct answer such that $R _ { \mathrm { V e r i f i e r } } ( { \pmb y } ; { \pmb y } ^ { * } ) = \mathbb { 1 } _ { \{ { \pmb y } = { \pmb y } ^ { \star } \} } ( \mathrm { i . e . }$ exact match rather than semantic equivalence $\mathbb { 1 } _ { \{ y \equiv y ^ { \star } \} } )$ ), the VeriFree objective is derived as follows:

$$
\begin{array}{l} J _ {\text {Verifier}} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \Big [ \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} [ \overbrace {\mathbb {1} _ {\{\boldsymbol {y} = \boldsymbol {y} ^ {\star} \}}} ^ {R _ {\text {Verifier}}} ] \Big ] \\ = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \Big [ \sum_ {\boldsymbol {y}} \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \mathbb {1} _ {\{\boldsymbol {y} = \boldsymbol {y} ^ {\star} \}} \Big ] \\ = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \Big [ \underbrace {\pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x} , \boldsymbol {z})} _ {R _ {\text {VeriFree}}} \Big ] \triangleq J _ {\text {VeriFree}} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}). \end{array}\tag{4}
$$

This can be interpreted as follows: if only one answer $y ^ { \star }$ is correct and receives a reward of 1 (while all others receive 0), then the expected reward given a reasoning trace z can be computed directly as the probability assigned to $\boldsymbol { y } ^ { \star }$ , effectively marginalizing out y. The corresponding gradient estimator is given by (see Appendix A.1 for a full derivation):

<table><tr><td>Verifier-based (R1-Zero)</td><td>VeriFree (Ours)</td></tr><tr><td>Model generates the reasoning trace z and answer y. Extract the answer y. Check answer using a verifier. Reward R Verifier = 1 if correct, 0 otherwise. Train with gradient estimator ∇θ J Verifier (Eq. 3).</td><td>Model generates the reasoning trace z. Patch in the correct answer y*. Evaluate probability πθ(y*|x, z). Reward R VeriFree = πθ(y*|x, z). Train with gradient estimator ∇θ J VeriFree (Eq. 5).</td></tr></table>

Figure 3: A pseudocode-like comparison of VeriFree (ours) and the standard R1-Zero approach.

$$
\nabla_ {\theta} J _ {\text {VeriFree}} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ R _ {\text {VeriFree}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \left[ \underbrace {\nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x})} _ {\text {reasoning term}} + \underbrace {\nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x} , \boldsymbol {z})} _ {\text {reference answer term}} \right] \right].\tag{5}
$$

Both the objective and its gradient estimator (Eq. (4) and $( 5 ) )$ are equivalent in expectation to their verifier-based counterparts (Eq. (2) and (3)). Intuitively, the “reasoning term” in Eq. (5) can be interpreted as a policy gradient where the reward for a reasoning trace z is the probability that the policy will generate the correct answer $\boldsymbol { y } ^ { \star }$ given z, while “reference answer term” can be viewed as a reward-weighted supervised learning term for $\boldsymbol { y } ^ { \star }$ given z. We will further elaborate on this interpretation in Sec. 2.3. In addition to bypassing the need for a verifier, our VeriFree gradient estimator also benefits from reduced variance:

Theorem 1. (Variance Reduction) Let $\hat { G } _ { \mathrm { V e r i f i e r } } ( x , y ^ { \star } , z )$ and $\hat { G } _ { \mathrm { V e r i F r e e } } ( x , y ^ { \star } , z , y )$ denote the single sample Monte Carlo estimators of $\nabla _ { \theta } J _ { \mathrm { V e r i f i e r } }$ and $\nabla _ { \theta } J _ { \mathrm { V e r i F r e e } }$ given x and $y ^ { \star }$ , respectively. Then we have

$$
\operatorname{Var} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ \hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \right] \leq \operatorname{Var} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}), \boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} \left[ \hat {G} _ {\text { Verifier}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right].\tag{6}
$$

This reduction in variance arises from Rao-Blackwellization [2]. For intuition, the variance in the Monte Carlo estimate of $\nabla _ { \theta } J _ { \mathrm { V e r i f i e t } }$ stems from the randomness in sampling $z \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } ) , y \sim$ $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { x } , z )$ , while for estimating $\nabla _ { \theta } J _ { \mathrm { V e r i F r e e } }$ we analytically marginalizes out y, thereby removing this source of randomness. We provide a full proof in Appendix A.2.

Our gradient estimator $\nabla _ { \theta } J _ { \mathrm { V e r i F r e e } } ( \theta ; x , y ^ { \star } )$ is fully compatible with other variance reduction techniques, including RLOO [1], GRPO [37] reward normalizations, and the PPO [36] clipping operation. As such, we sample multiple responses for each prompt and apply the RLOO baseline to the reasoning term in Eq. (5). We also adopt the corrected response-length normalization from Liu et al. [24]. The final on-policy gradient estimator is as follows:

$$
\nabla_ {\theta} J _ {\text { VeriFree }} (\theta) = \frac {1}{G} \sum_ {i = 1} ^ {G} \left[ A _ {i} \cdot \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} _ {i} | \boldsymbol {x}) + R _ {i} \cdot \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z} _ {i}) \right],\tag{7}
$$

where $z _ { i } \sim \pi _ { \theta } ( \cdot | x ) , R _ { i } = \pi _ { \theta } ( y ^ { \star } | x , z _ { i } )$ , and $\begin{array} { r } { A _ { i } = \pi _ { \boldsymbol { \theta } } ( \pmb { y } ^ { \star } | \pmb { x } , \pmb { z } _ { i } ) - \frac { 1 } { G - 1 } \sum _ { j \neq i } \pi _ { \boldsymbol { \theta } } ( \pmb { y } ^ { \star } | \pmb { x } , \pmb { z } _ { j } ) } \end{array}$ . We also provide the PPO-based off-policy variant in Appendix B.

## 2.3 Comparison to Existing Approaches

There have been two main prior works that, although derived from a different perspective, arrive at related alternative gradient estimators: Tang et al. [40] (JLB), and Chen et al. [4] (LaTRO).

$$
\nabla_ {\theta} J _ {\mathrm{Verifier}} = \mathbb {E} _ {\boldsymbol {z}, \boldsymbol {y}} \big [ \mathbb {1} _ {\{\boldsymbol {y} \equiv \boldsymbol {y} ^ {*} \}} \overbrace {\nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x})} ^ {\text {reasoning term}} + \mathbb {1} _ {\{\boldsymbol {y} \equiv \boldsymbol {y} ^ {*} \}} \overbrace {\nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x} , \boldsymbol {z})} ^ {\text {answer term}} \big ]\tag{R1-Zero}
$$

$$
\nabla_ {\theta} J _ {\text { VeriFree }} = \mathbb {E} _ {\boldsymbol {z}} \left[ \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \overbrace {\nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x} , \boldsymbol {z})} ^ {\text { reference   answer   term }} \right]\tag{Ours}
$$

$$
\nabla_ {\theta} J _ {\mathrm{JLB}} = \mathbb {E} _ {\boldsymbol {z}} \left[ \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + 1 \cdot \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \right] \quad (\text {Tang et al. [40]})
$$

$$
\nabla_ {\theta} J _ {\mathrm{LaTRO}} = \mathbb {E} _ {\boldsymbol {z}} \big [ (\log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) - \log \frac {\pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x})}{\pi_ {\mathrm{ref}} (\boldsymbol {z} | \boldsymbol {x})}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + 1 \cdot \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \big ]\tag{Chen et al. [4]}
$$

Both JLB and LaTRO treat the reasoning trace z as a latent variable and extend the standard supervised learning objective (log-likelihood) to optimize lower bounds on log $\left. E _ { z \sim \pi _ { \theta } ( \cdot | \pmb { x } ) } \left[ \pi _ { \theta } ( \pmb { y } ^ { \star } | \pmb { x } , \dot { z } ) \right] \right.$ and $\log \left( E _ { z \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } \left[ \pi _ { \theta } \bigl ( \pmb { y } ^ { \star } | \pmb { x } , z \bigr ) \right] \right)$ , respectively. The primary difference is that JLB samples z from the learned policy $\pi _ { \theta } ,$ while LaTRO uses a fixed reference policy $\pi _ { \mathrm { r e f } }$ . Despite originating from different perspectives, these approaches arrive at alternative gradient estimators that can be used similarly to ours, as shown in the above comparisons. However, as reported in Tang et al. [40], these verifier-free, variational-inference-based methods consistently underperform the standard verifier-based R1-Zero approach. In contrast, our formulation matches or outperforms the verifier-based baseline.

One possible explanation is that our method exactly recovers the original verifier-based objective under the single-correct-answer assumption, whereas JLB and LaTRO optimize subtly different objectives. For example, JLB effectively uses a reward of $R = \log \pi _ { \theta } ( y ^ { \star } | x , z )$ , as highlighted in the gradient expressions above. Another distinction lies in the weighting of the “reference answer term” $\nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \mathbf { y } ^ { \star } | \mathbf { x } , z )$ . In our method, this term is weighted by the probability $\pi _ { \boldsymbol { \theta } } ( \pmb { y } ^ { \star } | \pmb { x } , z )$ , which is the likelihood of the reference answer given the sampled reasoning trace. In contrast, both JLB and LaTRO use a fixed weight of 1, thereby increasing the probability of $\boldsymbol { y } ^ { \star }$ regardless of the quality of the reasoning trace z. We hypothesize that this behavior could promote poor reasoning. For instance, suppose the model generates the reasoning trace “... minus 2 apples, finally resulting in a total of 7 apples” when the correct answer is $" 6 "$ . The JLB and LaTRO objectives would still push the model to output “6” from that flawed trace, reinforcing a mismatch between reasoning and answer. Our method avoids this by down-weighting contributions from low-quality traces.

## 2.4 How to handle the tokenization at patching point?

A critical consideration when extracting reasoning traces z from model responses $( z , y )$ and subsequently replacing y with $\boldsymbol { y } ^ { \star }$ stems from the fact that LLMs operate on token sequences, not raw text strings. While human-readable outputs $( \mathrm { e . g . , \Sigma ^ { 6 6 } } \cdot \cdot \cdot$ .<answer> \\boxed{...} </answer>” as in Template 1) suggest splitting reasoning traces z at specific text patterns like “<answer>”, such textbased splitting strategy may cause tokenization inconsistencies. For example, the “>” character might be tokenized differently depending on its surrounding context in y versus $\boldsymbol { y } ^ { \star }$ . While one potential solution is to introduce special tokens to enforce consistent tokenization boundaries, these novel tokens could harm model performance due to their absence from the base model’s pretraining vocabulary.

Instead, we resolve this by defining the end of z at the token corresponding to <answer” (i.e., without $\mathbf { \vec { \cdot } } \mathbf { \vec { \cdot } } > \mathbf { \vec { \cdot } } )$ , leveraging the fact that the pattern $\mathbf { \dot { \bar { \rho } } } \mathbf { r } > \mathbf { \vec { \rho } }$ does not appear in standard tokenizer vocabularies. This ensures consistent token-space alignment between sampling and optimization in RL, avoiding instability due to off-policy mismatches. Notably, this approach is operationally equivalent to setting “<answer” (instead of “<answer>”) as the stop word during sampling, a mechanism natively supported by modern inference engines like vLLM [16]. In this case, we can sample reasoning traces z directly, instead of first generating the full response (z, y) and then extracting z post hoc.

## 3 Experiments

## 3.1 Setup

Training. Following the “zero” setting widely adopted in recent work [10, 14, 24, 27], we directly finetune the base LLM, skipping the intermediate stage of supervised fine-tuning (SFT). We implement our RL training pipeline using Oat [22] by instantiating their base modules and incorporate our algorithmic changes. Our experiments are conducted using Qwen3 [41] base models across multiple scales, including 1.7B, 4B, and 8B parameters. We adopt the prompt template shown in Template 1. We do not employ KL regularization losses or KL penalties in rewards, as recent studies suggest that removing KL terms does not have a significant impact [24, 14]. As a result, our method does not require maintaining a reference model in memory.

```txt
Template 1 (for Ours).
<|im_start|>user\n{question}\nPlease reason step by step, and put your final answer within <answer> \\boxed{} </answer>.<|im_end|>\n<|im_start|>assistant\n
```

For the 1.7B and 4B models, we conduct fine-tuning for approximately 4,000 policy gradient steps; for the 8B models, we fine-tune for around 3,000 policy gradient steps. During each step, the policy model (i.e., the LLM) generates 8 responses for each question (i.e., group\_size=8), with 16 questions processed per step. We use the sampling configurations temperature=1.0, top\_p=1, and max\_tokens=3000 for the rollout process. The responses are then parsed into reasoning traces and model-predicted answers. We replace the model-predicted answers with the reference answers from the training dataset. Subsequently, a single forward pass is executed to compute the conditional probability of the reference answer, conditioned on all preceding tokens including the prompt and the reasoning trace. This procedure introduces only a minimal additional computational cost, as the forward pass of the LLM does not require autoregressive decoding and does not require storing intermediate states for backpropagation. All collected samples from each step are used for one optimization step. The training is conducted on a single node with 8×H100 GPUs.

Table 1: Accuracy comparison on the MMLU-Pro benchmark.

<table><tr><td>Method</td><td>Len.</td><td>Avg.</td><td>CS</td><td>Math</td><td>Chem</td><td>Eng</td><td>Law</td><td>Bio</td><td>Health</td><td>Phys</td><td>Bus</td><td>Phil</td><td>Econ</td><td>Other</td><td>Psy</td><td>Hist</td></tr><tr><td>Qwen3-1.7B-Base</td><td>618</td><td>33.3</td><td>34.6</td><td>38.9</td><td>32.2</td><td>21.0</td><td>17.3</td><td>56.1</td><td>33.5</td><td>32.0</td><td>38.5</td><td>21.8</td><td>45.7</td><td>28.4</td><td>44.4</td><td>21.0</td></tr><tr><td>Qwen3-1.7B (w/o thinking)</td><td>893</td><td>46.1</td><td>49.5</td><td>64.4</td><td>48.0</td><td>35.9</td><td>22.9</td><td>64.9</td><td>38.0</td><td>49.7</td><td>53.5</td><td>33.7</td><td>53.9</td><td>36.4</td><td>51.6</td><td>31.2</td></tr><tr><td>Qwen3-1.7B (w/ thinking)</td><td>3904</td><td>52.0</td><td>56.1</td><td>76.4</td><td>57.6</td><td>27.0</td><td>21.9</td><td>67.9</td><td>47.7</td><td>57.5</td><td>61.3</td><td>38.9</td><td>64.5</td><td>42.5</td><td>59.2</td><td>32.3</td></tr><tr><td>Qwen3-1.7B-Base-Verifier</td><td>875</td><td>47.0</td><td>48.8</td><td>64.4</td><td>52.7</td><td>38.1</td><td>18.7</td><td>62.9</td><td>41.2</td><td>51.9</td><td>54.9</td><td>31.9</td><td>55.2</td><td>38.6</td><td>53.3</td><td>30.2</td></tr><tr><td>Qwen3-1.7B-Base-VeriFree</td><td>856</td><td>46.9</td><td>46.8</td><td>64.1</td><td>51.7</td><td>41.8</td><td>20.0</td><td>64.0</td><td>39.7</td><td>52.1</td><td>55.6</td><td>29.5</td><td>53.1</td><td>37.5</td><td>53.0</td><td>29.9</td></tr><tr><td>Qwen3-4B-Base</td><td>825</td><td>47.2</td><td>42.9</td><td>67.1</td><td>55.5</td><td>40.0</td><td>22.5</td><td>56.9</td><td>43.6</td><td>55.4</td><td>54.9</td><td>27.5</td><td>52.7</td><td>34.3</td><td>48.6</td><td>34.7</td></tr><tr><td>Qwen3-4B (w/o thinking)</td><td>838</td><td>60.0</td><td>65.9</td><td>79.1</td><td>65.8</td><td>45.7</td><td>29.0</td><td>76.6</td><td>57.0</td><td>65.1</td><td>66.7</td><td>48.9</td><td>69.2</td><td>52.1</td><td>64.3</td><td>44.6</td></tr><tr><td>Qwen3-4B (w/ thinking)</td><td>3456</td><td>62.7</td><td>70.0</td><td>84.8</td><td>66.6</td><td>38.6</td><td>28.7</td><td>81.3</td><td>60.4</td><td>67.4</td><td>69.2</td><td>53.7</td><td>75.1</td><td>57.8</td><td>67.9</td><td>49.6</td></tr><tr><td>Qwen3-4B-Base-Verifier</td><td>921</td><td>63.0</td><td>66.1</td><td>81.3</td><td>69.7</td><td>52.8</td><td>29.1</td><td>79.8</td><td>62.8</td><td>67.6</td><td>71.2</td><td>48.5</td><td>73.1</td><td>52.8</td><td>68.5</td><td>45.4</td></tr><tr><td>Qwen3-4B-Base-VeriFree</td><td>1241</td><td>63.5</td><td>64.4</td><td>82.2</td><td>70.1</td><td>55.6</td><td>30.7</td><td>81.7</td><td>59.2</td><td>71.0</td><td>71.0</td><td>47.1</td><td>71.7</td><td>53.4</td><td>66.8</td><td>47.5</td></tr><tr><td>Qwen2.5-7B</td><td>519</td><td>47.8</td><td>48.3</td><td>59.5</td><td>44.4</td><td>33.4</td><td>25.1</td><td>63.6</td><td>50.4</td><td>48.0</td><td>55.9</td><td>34.7</td><td>60.6</td><td>46.0</td><td>58.2</td><td>38.3</td></tr><tr><td>Qwen2.5-7B-SimpleRL-Zoo</td><td>705</td><td>51.2</td><td>51.2</td><td>52.0</td><td>50.2</td><td>40.8</td><td>30.5</td><td>69.5</td><td>54.3</td><td>52.5</td><td>57.3</td><td>41.9</td><td>62.8</td><td>52.6</td><td>60.8</td><td>42.3</td></tr><tr><td>Qwen2.5-Math-7B-Oat-Zero</td><td>556</td><td>40.5</td><td>47.6</td><td>47.7</td><td>46.9</td><td>32.1</td><td>18.1</td><td>53.6</td><td>25.7</td><td>49.4</td><td>52.9</td><td>29.5</td><td>54.7</td><td>32.8</td><td>43.0</td><td>22.8</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>481</td><td>55.3</td><td>56.6</td><td>70.4</td><td>55.6</td><td>42.7</td><td>29.8</td><td>69.3</td><td>55.1</td><td>57.9</td><td>63.5</td><td>41.5</td><td>63.4</td><td>53.6</td><td>62.4</td><td>43.6</td></tr><tr><td>General-Reasoner-7B</td><td>867</td><td>58.7</td><td>63.4</td><td>73.7</td><td>63.3</td><td>44.9</td><td>35.2</td><td>72.0</td><td>56.6</td><td>61.5</td><td>66.7</td><td>43.1</td><td>68.1</td><td>52.8</td><td>62.8</td><td>47.8</td></tr><tr><td>Qwen3-8B-Base</td><td>613</td><td>59.8</td><td>61.2</td><td>75.0</td><td>66.2</td><td>46.7</td><td>31.4</td><td>75.9</td><td>60.4</td><td>62.1</td><td>65.9</td><td>48.7</td><td>69.0</td><td>54.3</td><td>63.9</td><td>47.2</td></tr><tr><td>Qwen3-8B (w/o thinking)</td><td>1032</td><td>61.9</td><td>65.6</td><td>71.9</td><td>62.8</td><td>46.2</td><td>34.7</td><td>79.9</td><td>66.1</td><td>63.7</td><td>69.3</td><td>55.9</td><td>72.9</td><td>58.9</td><td>67.9</td><td>52.5</td></tr><tr><td>Qwen3-8B (w/ thinking)</td><td>3952</td><td>66.9</td><td>71.5</td><td>83.8</td><td>68.0</td><td>38.7</td><td>39.2</td><td>85.2</td><td>72.1</td><td>69.8</td><td>73.3</td><td>57.5</td><td>79.2</td><td>66.3</td><td>71.8</td><td>57.7</td></tr><tr><td>Qwen3-8B-Base-Verifier</td><td>594</td><td>65.9</td><td>63.9</td><td>81.8</td><td>71.1</td><td>56.9</td><td>35.4</td><td>81.9</td><td>64.9</td><td>71.6</td><td>74.1</td><td>53.9</td><td>74.2</td><td>58.4</td><td>68.4</td><td>54.3</td></tr><tr><td>Qwen3-8B-Base-VeriFree</td><td>776</td><td>67.2</td><td>71.5</td><td>85.3</td><td>73.5</td><td>55.7</td><td>37.3</td><td>81.9</td><td>64.3</td><td>73.1</td><td>74.1</td><td>54.9</td><td>74.8</td><td>59.6</td><td>67.7</td><td>54.6</td></tr></table>

Dataset. To support general reasoning, we begin with the dataset curated by Ma et al. [27], sourced from WebInstruct [49]. To improve data quality and reliability and reduce size, we retain samples with answers that consist of fewer than seven tokens, and use Qwen2.5-72B-Instruct [43] to filter out low-quality and noisy data. This process results in approximately 61,000 data samples spanning diverse domains, which we refer to as WebData. The category distribution is visualized in Fig. 7.

Evaluation. In line with prior work [27], we employ multiple-choice questions for evaluation to facilitate verification. To assess general reasoning abilities, we utilize the following benchmarks: MMLU-Pro [42], a challenging multi-task understanding benchmark designed to evaluate the capabilities of LLMs across various domains; SuperGPQA [6], a large-scale benchmark consisting of graduate-level reasoning questions spanning 285 diverse disciplines; and GPQA [35], which focuses on graduate-level question-answering and is designed to resist shallow pattern-matching and memorization. While our primary focus is not on enhancing mathematical abilities, we also evaluate math reasoning using a suite of standard math reasoning benchmarks. This suite includes MATH-500 [13], OlympiadBench [11], Minerva Math [18], GSM8K [5], AMC and AIME24 [19]. We utilize Math-Verify<sup>2</sup> to check for answer equivalence. Except for AIME24, where we employ a temperature=1.0 and repeat each question 32 times, all other benchmarks are evaluated using temperature=0.0. We use max\_tokens=8192 for all evaluations.

Baselines. Our primary baseline, denoted Verifier, is a verifier-based approach using the verifier from Ma et al. [27]. The verifier is initialized from Qwen2.5-Math-1.5B [44] and fine-tuned on data generated by Gemini 2.0 Flash to assess equivalence between the reference and predicted answers, conditioned on the question. We apply Dr. GRPO [24] as the optimization algorithm for the baseline, ensuring that all other settings are consistent with our approach. Following Ma et al. [27], the reward definition incorporates additional factors beyond verifier correctness, including format compliance and the length of generated answers. If the format is incorrect (e.g., missing \\boxed{} in the model response), a negative reward of -0.5 is applied. Moreover, a length penalty of -0.05 × min(10, abs(length\_of\_correct\_answer - length\_of\_answer)) is added.

Table 2: Accuracy comparison on the SuperGPQA benchmark.

<table><tr><td>Method</td><td>Len.</td><td>Avg.</td><td>Eng.</td><td>Med.</td><td>Sci.</td><td>Phil.</td><td>M.S.</td><td>Econ.</td><td>Mgmt.</td><td>Socio.</td><td>L/A</td><td>Hist.</td><td>Agron.</td><td>Law</td><td>Edu.</td></tr><tr><td>Qwen3-1.7B-Base</td><td>997</td><td>17.4</td><td>17.7</td><td>18.6</td><td>16.0</td><td>27.4</td><td>27.3</td><td>20.5</td><td>22.4</td><td>23.1</td><td>15.5</td><td>11.1</td><td>18.1</td><td>20.6</td><td>21.3</td></tr><tr><td>Qwen3-1.7B (w/o thinking)</td><td>1152</td><td>23.3</td><td>22.6</td><td>22.8</td><td>24.3</td><td>30.3</td><td>31.2</td><td>24.3</td><td>27.5</td><td>23.8</td><td>19.0</td><td>18.1</td><td>20.8</td><td>24.5</td><td>28.1</td></tr><tr><td>Qwen3-1.7B (w/ thinking)</td><td>4799</td><td>23.6</td><td>21.8</td><td>25.3</td><td>23.6</td><td>33.1</td><td>33.7</td><td>29.6</td><td>27.5</td><td>32.2</td><td>19.0</td><td>18.0</td><td>25.0</td><td>26.2</td><td>31.6</td></tr><tr><td>Qwen3-1.7B-Base-Verifier</td><td>1049</td><td>24.5</td><td>26.0</td><td>23.9</td><td>24.4</td><td>30.8</td><td>26.8</td><td>26.9</td><td>27.0</td><td>26.6</td><td>18.9</td><td>16.6</td><td>22.3</td><td>22.6</td><td>27.3</td></tr><tr><td>Qwen3-1.7B-Base-VeriFree</td><td>964</td><td>24.8</td><td>25.7</td><td>24.7</td><td>24.9</td><td>26.5</td><td>30.2</td><td>25.9</td><td>27.9</td><td>28.0</td><td>20.4</td><td>15.9</td><td>22.9</td><td>25.0</td><td>28.9</td></tr><tr><td>Qwen3-4B-Base</td><td>902</td><td>24.7</td><td>25.7</td><td>23.6</td><td>26.0</td><td>23.6</td><td>25.4</td><td>28.8</td><td>28.4</td><td>19.6</td><td>16.4</td><td>16.8</td><td>20.6</td><td>25.6</td><td>24.0</td></tr><tr><td>Qwen3-4B (w/o thinking)</td><td>1397</td><td>31.6</td><td>32.0</td><td>31.5</td><td>32.3</td><td>37.5</td><td>36.1</td><td>37.8</td><td>33.7</td><td>33.6</td><td>24.3</td><td>20.6</td><td>28.3</td><td>31.4</td><td>33.5</td></tr><tr><td>Qwen3-4B (w/ thinking)</td><td>4568</td><td>31.7</td><td>30.7</td><td>33.2</td><td>32.1</td><td>41.2</td><td>31.7</td><td>41.7</td><td>35.9</td><td>32.9</td><td>24.5</td><td>22.4</td><td>30.9</td><td>35.7</td><td>35.3</td></tr><tr><td>Qwen3-4B-Base-Verifier</td><td>1045</td><td>34.3</td><td>35.4</td><td>35.5</td><td>34.5</td><td>39.2</td><td>41.0</td><td>39.1</td><td>36.7</td><td>37.1</td><td>26.6</td><td>22.3</td><td>33.8</td><td>33.1</td><td>35.3</td></tr><tr><td>Qwen3-4B-Base-VeriFree</td><td>1451</td><td>35.1</td><td>36.3</td><td>34.5</td><td>36.9</td><td>35.7</td><td>37.1</td><td>39.1</td><td>38.3</td><td>31.5</td><td>24.7</td><td>22.0</td><td>33.0</td><td>33.2</td><td>34.1</td></tr><tr><td>Qwen2.5-7B</td><td>716</td><td>23.8</td><td>24.2</td><td>27.0</td><td>21.8</td><td>28.8</td><td>31.2</td><td>27.6</td><td>29.1</td><td>22.4</td><td>20.8</td><td>20.2</td><td>24.5</td><td>27.4</td><td>30.2</td></tr><tr><td>Qwen-2.5-7B-SimpleRL-Zoo</td><td>850</td><td>26.3</td><td>26.4</td><td>30.5</td><td>23.8</td><td>32.6</td><td>32.2</td><td>33.0</td><td>31.9</td><td>28.7</td><td>24.1</td><td>21.4</td><td>27.2</td><td>29.6</td><td>32.9</td></tr><tr><td>Qwen2.5-Math-7B-Oat-Zero</td><td>638</td><td>21.3</td><td>23.1</td><td>16.4</td><td>21.5</td><td>23.1</td><td>21.5</td><td>25.9</td><td>27.2</td><td>25.2</td><td>17.7</td><td>15.9</td><td>21.4</td><td>18.8</td><td>24.8</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>604</td><td>28.4</td><td>27.7</td><td>32.2</td><td>27.6</td><td>33.7</td><td>32.2</td><td>32.4</td><td>32.9</td><td>32.9</td><td>24.5</td><td>22.1</td><td>29.7</td><td>30.6</td><td>32.4</td></tr><tr><td>General-Reasoner-7B</td><td>1047</td><td>30.8</td><td>31.5</td><td>32.2</td><td>29.9</td><td>35.2</td><td>41.5</td><td>38.4</td><td>33.1</td><td>35.0</td><td>25.5</td><td>22.7</td><td>28.9</td><td>32.5</td><td>35.5</td></tr><tr><td>Qwen3-8B-Base</td><td>825</td><td>31.0</td><td>31.3</td><td>34.0</td><td>30.6</td><td>36.0</td><td>37.1</td><td>34.7</td><td>37.5</td><td>35.0</td><td>24.2</td><td>20.0</td><td>28.5</td><td>31.4</td><td>36.4</td></tr><tr><td>Qwen3-8B (w/o thinking)</td><td>1638</td><td>32.4</td><td>32.6</td><td>36.5</td><td>31.2</td><td>39.5</td><td>42.0</td><td>37.7</td><td>37.3</td><td>38.5</td><td>25.0</td><td>22.6</td><td>33.2</td><td>34.3</td><td>38.4</td></tr><tr><td>Qwen3-8B (w/ thinking)</td><td>4995</td><td>35.0</td><td>33.6</td><td>42.1</td><td>33.5</td><td>44.4</td><td>37.6</td><td>44.2</td><td>42.7</td><td>42.7</td><td>27.9</td><td>24.9</td><td>37.9</td><td>38.7</td><td>40.7</td></tr><tr><td>Qwen3-8B-Base-Verifier</td><td>713</td><td>37.1</td><td>38.2</td><td>39.5</td><td>37.2</td><td>39.5</td><td>39.5</td><td>43.0</td><td>40.1</td><td>38.5</td><td>28.9</td><td>24.8</td><td>34.2</td><td>34.8</td><td>38.2</td></tr><tr><td>Qwen3-8B-Base-VeriFree</td><td>951</td><td>38.0</td><td>38.3</td><td>39.1</td><td>39.6</td><td>37.5</td><td>42.9</td><td>41.8</td><td>41.7</td><td>44.8</td><td>28.6</td><td>23.3</td><td>33.6</td><td>36.3</td><td>38.6</td></tr></table>

We also report the results for the base and the instruct models of Qwen-3-1.7B/4B/8B [41] and Qwen2.5-7B [43], as well as the checkpoints released by Qwen2.5-7B-SimpleRL-Zoo [50], Qwen2.5- Math-7B-Oat-Zero [24], and General-Reasoner-7B [27].<sup>3</sup> Notably, Qwen3 integrates both a thinking mode (for complex, multi-step reasoning) and a non-thinking mode (for rapid, context-driven responses) within a unified framework. We report results of both modes on Qwen3 instruct models.

## 3.2 Main Results

VeriFree improves general reasoning capabilities. We begin by evaluating the effectiveness of VeriFree in enhancing the general reasoning capabilities of LLMs using the MMLU-Pro and SuperGPQA benchmarks. Table 1 presents a detailed comparison across model scales and domains in the MMLU-Pro benchmark. Starting from base models, applying RL with VeriFree yields substantial gains in average accuracy—ranging from 12% to 40%—demonstrating that VeriFree effectively fine-tunes LLMs to improve general reasoning performance.

Similar improvements are observed on the SuperGPQA benchmark, as shown in Table 2, where VeriFree consistently enhances the performance of base models by a significant margin. Notably, VeriFree achieves performance comparable to, or even surpassing, that of the instruct model in thinking mode and the Verifier baseline (i.e., the RL-tuned model learned with an additional modelbased verifier)—without relying on any explicit verification signals.

In addition to accuracy gains, we also observe an increase in response length after tuned by VeriFree, suggesting that the model explores longer reasoning traces to arrive at more accurate answers—a behavior reminiscent of DeepSeek-R1-Zero [10]. Results on the GPQA benchmark are provided in Appendix D due to space constraints.

VeriFree leads to better learning efficiency. We compare VeriFree with the baseline that learns from a model-based verifier reward (i.e., Verifier). As shown in Fig. 4 (Left), VeriFree consistently outperforms the baseline, achieving higher accuracy with fewer training steps. We attribute this improved learning efficiency to reduced gradient variance, enabled by VeriFree’s continuous reward signals and the RLOO objective. While both approaches optimize the same reward signal in expectation, VeriFree provides more stable and informative policy gradients, which accelerate convergence and leads to better final performance.


Figure 4: Left: MMLU-Pro accuracy of VeriFree and the baseline fine-tuned from Qwen3-8B base model along training steps. The curve is smoothed by a moving average with an interval of 384. Right: The dynamics of MMLU-Pro evaluation accuracy and average model confidence π<sub>θ</sub> $( \boldsymbol { y } ^ { \star } | \boldsymbol { x } , z )$ along training based on Qwen3-8B base model. Raw data points are depicted with more transparency, while smoothed data using a Gaussian filter is shown with less transparency for emphasis. Darker colors represent larger training steps.

Model confidence is a good reasoning capability proxy. Our analysis based on Qwen3-8B base model reveals a strong positive correlation $( \rho = 0 . 8 2 )$ between MMLU-Pro accuracy and the average model confidence $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { \star } | \boldsymbol { x } , z )$ during training (Fig. 4, Right). This empirically demonstrates that the model’s self-estimated confidence in the correct answer, $\mathrm { i } . \mathrm { e } . , \pi _ { \theta } ( y ^ { \star } | x , z )$ , serves as an effective metric for quantifying emergent reasoning capabilities in LLMs.

VeriFree learns transferable reasoning skills. To evaluate the transferability of reasoning acquired through VeriFree, we train a model on a dataset with all math-related examples removed, and assess its performance on both general and math-specific benchmarks. As shown in Fig. 5, VeriFree not only improves reasoning performance on general tasks, as expected, but also demonstrates strong transfer to math benchmarks—despite the absence of math supervision during training. This highlights VeriFree’s ability to induce general reasoning capabilities that extend across domains.

Figure 5: Reasoning transfer to math without math training. VeriFree enhances reasoning transfer: when trained only on non-math data, the model improves on general benchmarks and effectively transfers to math-specific tasks.

## 3.3 Ablation Study

To systematically evaluate method components and offer a comprehensive understanding of VeriFree, we conduct ablation studies based on Qwen3-1.7B base models as follows.

Effects of extraction strategy for reasoning trace z. Our method requires precise separation between reasoning path z and answer y to enable answer replacement. While human-readable splits using <answer> seem intuitive, we instead define z to end at “<answer” (omitting “>”), ensuring consistent tokenization boundaries (see Sec. 2.4). We compare with a variant using text-based splitting (denoted as “VeriFree w/o token split”) on Qwen3-1.7B via MMLU-Pro (Fig. 6). Our tokenization-aware approach achieves superior convergence, while the variant suffers optimization instability due to effectively introducing off-policy data.

Effects of RLOO. As observed in Fig. 6 (Left), removing RLOO leads to a consistent drop in performance throughout training, with final accuracy more than 3% lower than that of the full method. This highlights the importance of RLOO in stabilizing learning and guiding the model toward better generalization. Without RLOO, the model converges prematurely and fails to reach the same peak accuracy.

Effects of equivalence class. As mentioned in Sec. 2.1, verifier-based RL typically assesses answer correctness as rewards. Correct answers within a specific class often form an equivalence class. Our method, however, utilizes model confidence by focusing on a single reference answer for a given question and the model’s reasoning trace. To explore the potential advantages of integrating an “equivalence class” into our approach, we conducted ablation studies as follows. We employed a model fine-tuned on the MATH-12k dataset [12, 20] from Qwen3-8B base model through Dr. GRPO [24] with rule-based verification to sample answers on MATH-12k, subsequently verifying answer correctness using Math-Verify. This approach enabled us to create an extended dataset with a set of equivalent correct answers for each question. We then fine-tuned Qwen3-1.7B base models using our method on both the original and the extended MATH-12k datasets incorporating equivalence classes. These models are evaluated on GSM8K, MATH-500, Minerva Math, and OlympiadBench to assess the impact of including equivalence classes. The results, shown in Fig. 6 (Right), indicate that considering equivalence classes in our method offers slight performance improvements, aligning with our expectations. This highlights a minor limitation of our current formulation and motivates future work on algorithms that can better leverage answer equivalence.


Figure 6: Left: MMLU-Pro Evaluation Accuracy of VeriFree against ablation variants (w/o RLOO, w/o tokenization-aware split strategy) along training steps. Note that all these models are based on Qwen3-1.7B base models. Right: Effects of introducing the equivalent class to VeriFree on model performance.

## 4 Related Works

DeepSeek-R1-Zero-like reinforcement learning. DeepSeek-R1-Zero [10], Tülu [17] and OpenAI’s o1 [29] recently demonstrated that applying RL to learn directly from binary verification-based rewards can be extremely powerful in enhancing the reasoning capabilities of base LLMs. Since then several works have reproduced R1-Zero-like training on smaller scales [50, 31, 23, 14, 26, 25]. The aforementioned works all focus on math and coding, domains where verifiable rewards are readily available. By contrast, our work aims to extend the R1-Zero-like training paradigm to general domains where verifiable rewards are not available.

Reasoning beyond verifiable domains. Previous work on reasoning without access to verifiable rewards has been based around employing an additional LLM to act as a proxy verifier or reward model. NaturalReasoning [47] introduces a large, multi-domain dataset and presents baselines trained using RFT [48] and DPO [33], leveraging a second LLM as a reward model, while Su et al. [38] and General-Reasoner [27] similarly incorporate a separate LLM to serve as a verifier. X-Reasoner [21] also inves tigates general reasoning but circumvents the lack of rule-based verification by dropping the R1-Zerostyle training paradigm, instead training via SFT [9] on responses sampled from more capable models.

Self-improving language models. Several works have explored training LLMs using signals based on the model’s own outputs. Yuan et al. [46] propose to prompt the model to judge and rank different responses and select the best and worst for preference learning. Chen et al. [3] leverage the DPO implicit rewards for more efficient and robust self-alignment via iterative DPO [33]. Zuo et al. [52] use majority voting to construct self-labeled rewards for RL to further improve well-trained models during test time, which can be understood as a form of sharpening [15]. Another line of research [32, 4, 40] approaches LLM reasoning from the direction of variational optimization, treating the reasoning trace as a latent variable. Despite starting from a different viewpoint our method has interesting and close connections to this perspective which we discuss in detail in Sec. 2.3.

## 5 Conclusions

In this paper, we rethink reinforcement learning with verifiable rewards (RLVR) for LLMs from a novel perspective. By leveraging the gradient equivalence under the unique answer assumption, we derive a new optimization objective that eliminates the need for explicit verification. Our proposed method, VeriFree, is particularly well-suited for general reasoning tasks, where rule-based verifiers are infeasible and model-based verifiers are both expensive and vulnerable to reward hacking. Through extensive experiments and ablations, we demonstrate the effectiveness and robustness of VeriFree on a wide range of general reasoning benchmarks. We hope our work offers a fresh viewpoint for the LLM RL community and provides a practical approach for building future general-purpose reasoners.

## References

[1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

[2] George Casella and Christian P. Robert. Rao-blackwellisation of sampling schemes. Biometrika, 83(1):81–94, 1996.

[3] Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakantham, and Min Lin. Bootstrapping language models with dpo implicit rewards. In International Conference on Learning Representations (ICLR), 2025.

[4] Haolin Chen, Yihao Feng, Zuxin Liu, Weiran Yao, Akshara Prabhakar, Shelby Heinecke, Ricky Ho, Phil Mui, Silvio Savarese, Caiming Xiong, et al. Language models are hidden reasoners: Unlocking latent reasoning capabilities via self-rewarding. arXiv preprint arXiv:2411.04282, 2024.

[5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[6] Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739, 2025.

[7] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

[8] Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Quentin Carbonneaux, Taco Cohen, and Gabriel Synnaeve. Rlef: Grounding code llms in execution feedback with reinforcement learning. arXiv preprint arXiv:2410.02089, 2024.

[9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[11] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.

[12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

[14] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025. URL https://arxiv.org/abs/2503.24290.

[15] Audrey Huang, Adam Block, Dylan J Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T Ash, and Akshay Krishnamurthy. Self-improvement in language models: The sharpening mechanism. arXiv preprint arXiv:2412.01951, 2024.

[16] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[17] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

[18] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

[19] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13:9, 2024.

[20] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=v8L0pN6EOi.

[21] Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, and Hoifung Poon. X-reasoner: Towards generalizable reasoning across modalities and domains, 2025. URL https://arxiv.org/ abs/2505.03981.

[22] Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee, and Min Lin. Oat: A research-friendly framework for llm online alignment. https://github.com/sail-sg/oat, 2024.

[23] Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. There may not be aha moment in r1-zero-like training — a pilot study. https://oatllm.notion.site/ oat-zero, 2025. Notion Blog.

[24] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

[25] Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level, 2025. Notion Blog.

[26] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

[27] Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. Generalreasoner: Advancing llm reasoning across all domains. arXiv preprint arXiv:2505.14652, 2025.

[28] A. Ng, Daishi Harada, and Stuart J. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In International Conference on Machine Learning, 1999. URL https://api.semanticscholar.org/CorpusID:5730166.

[29] OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.

[30] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[31] Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.

[32] Du Phan, Matthew Douglas Hoffman, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, and Rif A Saurous. Training chain-ofthought via latent-variable inference. Advances in Neural Information Processing Systems, 36: 72819–72841, 2023.

[33] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

[34] Jette Randlov and Preben Alstrøm. Learning to drive a bicycle using reinforcement learning and shaping. pages 463–471, 01 1998.

[35] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

[36] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[37] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[38] Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Expanding rl with verifiable rewards across diverse domains. arXiv preprint arXiv:2503.23829, 2025.

[39] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018.

[40] Yunhao Tang, Sid Wang, and Rémi Munos. Learning to chain-of-thought with jensen’s evidence lower bound. arXiv preprint arXiv:2503.19618, 2025.

[41] Qwen Team. Qwen3, April 2025. URL https://qwenlm.github.io/blog/qwen3/.

[42] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[43] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[44] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

[45] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[46] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. International Conference on Machine Learning, 2024. doi: 10.48550/arXiv.2401.10020.

[47] Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E Weston, et al. Naturalreasoning: Reasoning in the wild with 2.8 m challenging questions. arXiv preprint arXiv:2502.13124, 2025.

[48] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.

[49] Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. Advances in Neural Information Processing Systems, 37:90629–90660, 2024.

[50] Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. https://hkust-nlp.notion.site/simplerl-reason, 2025. Notion Blog.

[51] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

[52] Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

## Appendix

## Table of Contents

A Theoretical Analysis 14  
A.1 Derivation of Gradient Estimators 14  
A.2 Proof of Lower Variance 15  
B Off-policy Gradient Estimators 16  
C Dataset Details 16  
D Extended Empirical Results 16  
E Case Study 17  
F Broader Impact 21

## A Theoretical Analysis

## A.1 Derivation of Gradient Estimators

Here we provide derivations for Eq. (5)—the gradient estimator of VeriFree. We also include the corresponding derivation for the standard verifier-based gradient estimator for completeness.

Proof. The gradient estimator for $J _ { \mathrm { V e r i f i e r } }$ is derived as follows:

$$
\begin{array}{l} \nabla_ {\theta} J _ {\text {Verifier}} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) \\ = \nabla_ {\theta} \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} \left[ R _ {\text {Verifier}} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) \right] \right] \\ = \nabla_ {\theta} \sum_ {\boldsymbol {z}, \boldsymbol {y}} R _ {\text {Verifier}} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \\ = \sum_ {\boldsymbol {z}, \boldsymbol {y}} R _ {\text {Verifier}} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) \left[ \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \nabla_ {\theta} \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \nabla_ {\theta} \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \right] \\ = \sum_ {\boldsymbol {z}, \boldsymbol {y}} R _ {\text {Verifier}} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) \left[ \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \right] \\ = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} \left[ R _ {\text {Verifier}} (\boldsymbol {y}; \boldsymbol {y} ^ {\star}) [ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) ] \right] \right] \end{array}
$$

The gradient estimator for $J _ { \mathrm { V e r i F r e e } }$ is derived as follows:

$$
\begin{array}{l} \nabla_ {\theta} J _ {\text {VeriFree}} (\theta ; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) \\ = \nabla_ {\theta} \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} [ R _ {\text {VeriFree}} (\boldsymbol {z}; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) ] \\ = \nabla_ {\theta} \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} [ \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) ] \\ = \nabla_ {\theta} \sum_ {\boldsymbol {z}} \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \\ = \sum_ {\boldsymbol {z}} \left[ \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \nabla_ {\theta} \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \nabla_ {\theta} \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \right] \\ = \sum_ {\boldsymbol {z}} \left[ \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \right] \\ = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) [ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) ] \right] \\ = \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ R _ {\text {VeriFree}} (\boldsymbol {z}; \boldsymbol {x}, \boldsymbol {y} ^ {\star}) [ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) ] \right] \end{array}
$$

## A.2 Proof of Lower Variance

Here we provide a full proof of Theorem 1, the reduced variance property of VeriFree. We show that the policy gradient estimator derived from $J _ { \mathrm { V e r i F r e e } } ( \theta ; x , y ^ { \star } )$ has variance less than or equal to that of the estimator derived from $J _ { \mathrm { V e r i f i e r } } ( \theta ; x , y ^ { \star } )$ for any given $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } , \mathbf { \boldsymbol { y } } ^ { \star } }$ . The same relationship will hold for the global objectives averaged over $( \pmb { x } , \pmb { y } ^ { \star } ) \sim \mathcal { D }$

Proof. The global objective functions are:

$$
\begin{array}{l} J _ {\text {Verifier}} (\theta) = \mathbb {E} _ {(\boldsymbol {x}, \boldsymbol {y} ^ {\star}) \sim \mathcal {D}} \left[ \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \Big [ \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} \big [ \mathbb {1} \{\boldsymbol {y} = \boldsymbol {y} ^ {\star} \} \big ] \Big ] \right] \\ J _ {\text {VeriFree}} (\theta) = \mathbb {E} _ {(\boldsymbol {x}, \boldsymbol {y} ^ {\star}) \sim \mathcal {D}} \Big [ \mathbb {E} _ {\boldsymbol {z} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \big [ \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \big ] \Big ] \end{array}
$$

For a given $( \pmb { x } , \pmb { y } ^ { \star } ) \sim \mathcal { D }$ , the single-sample Monte Carlo gradient estimators are:

$$
\hat {G} _ {\text { Verifier }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) = \mathbb {1} \left\{\boldsymbol {y} = \boldsymbol {y} ^ {\star} \right\} \left[ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \right]
$$

where $z \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } ) , y \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } , z )$ , and

$$
\hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) = \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \left[ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \right]
$$

where $z \sim \pi _ { \theta } ( \cdot | x )$

The proof relies on the law of total variance and the relationship between $\hat { G } _ { \mathrm { V e r i f i e r } }$ and $\hat { G } _ { \mathrm { V e r i F r e e } }$

First, we show that $\hat { G } _ { \mathrm { V e r i F r e e } } ( x , y ^ { \star } , z )$ is the conditional expectation of $\hat { G } _ { \mathrm { V e r i f i e r } } ( \ b { x } , \ b { y } ^ { \star } , \ b { z } , \ b { y } )$ given $x , y ^ { \star } , z$ . The expectation is taken over $\pmb { y } \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } , z )$ :

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} \left[ \hat {G} _ {\text { Verifier}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) | \boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z} \right] \\ = \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {z})} \left[ \mathbb {1} \{\boldsymbol {y} = \boldsymbol {y} ^ {\star} \} \left[ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}, \boldsymbol {z}) \right] | \boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z} \right] \\ = \sum_ {\boldsymbol {y} ^ {\prime}} \pi_ {\theta} (\boldsymbol {y} ^ {\prime} | \boldsymbol {x}, \boldsymbol {z}) \left[ \mathbb {1} \{\boldsymbol {y} ^ {\prime} = \boldsymbol {y} ^ {\star} \} \left[ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\prime} | \boldsymbol {x}, \boldsymbol {z}) \right] \right] \\ = \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \left[ \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {z} | \boldsymbol {x}) + \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {y} ^ {\star} | \boldsymbol {x}, \boldsymbol {z}) \right] \\ = \hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \end{array}
$$

We denote $\begin{array} { r } { \mathbb { E } _ { { y } \sim \pi _ { \boldsymbol { \theta } } ( . | \mathbf { x } , z ) } \big [ \hat { G } _ { \mathrm { V e r i f i e r } } ( \mathbf { x } , \pmb { y } ^ { \star } , z , \pmb { y } ) \big | \mathbf { x } , \pmb { y } ^ { \star } , z \big ] } \end{array}$ as $\mathbb { E } _ { { \pmb y } | z } \big [ \hat { G } _ { \mathrm { V e r i f i e r } } ( { \pmb x } , { \pmb y } ^ { \star } , z , { \pmb y } ) \big ]$ for brevity in the following, since x and $y ^ { \star }$ are already given and fixed. Thus, we have

$$
\mathbb {E} _ {\boldsymbol {y} | \boldsymbol {z}} \left[ \hat {G} _ {\text { Verifier }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] = \hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z})\tag{8}
$$

The law of total variance states that for a random variable W and conditioning variables S, $\operatorname { v a r } ( W ) =$ $\mathbb { E } _ { S } [ \operatorname { V a r } [ W | S ] ] + \operatorname { V a r } _ { S } [ \mathbb { E } [ W | S ] ]$ . Let $W = \hat { G } _ { \mathrm { V e r i f i e r } } ( x , y ^ { \star } , z , y )$ . Given x and $y ^ { \star }$ , the sources of randomness for $\hat { G } _ { \mathrm { V e r i f i e r } }$ are z and y. Let $S = z$ be the conditioning variables. The randomness in $\hat { G } _ { \mathrm { V e r i f i e r } }$ given $S$ comes from $\pmb { y } \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } , z )$ . Applying the law:

$$
\operatorname{Var} _ {\boldsymbol {z}, \boldsymbol {y}} \left[ \hat {G} _ {\text { Verifier}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] = \mathbb {E} _ {\boldsymbol {z}} \left[ \operatorname{Var} _ {\boldsymbol {y} | \boldsymbol {z}} \left[ \hat {G} _ {\text { Verifier}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] \right] + \operatorname{Var} _ {\boldsymbol {z}} \left[ \mathbb {E} _ {\boldsymbol {y} | \boldsymbol {z}} \left[ \hat {G} _ {\text { Verifier}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] \right]
$$

The expectation $\mathbb { E } _ { z }$ is over $z \sim \pi _ { \theta } ( \cdot | x )$ . The conditional variance $\operatorname { V a r } _ { y \mid z }$ and expectation $\mathbb { E } _ { y \mid z }$ are over $\pmb { y } \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } , z )$ for fixed $x , y ^ { \star } , z$

Substituting the result from Eq. (8) into the law of total variance:

$$
\operatorname{Var} _ {\boldsymbol {z}, \boldsymbol {y}} \left[ \hat {G} _ {\text { Verifier }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] = \mathbb {E} _ {\boldsymbol {z}} \left[ \operatorname{Var} _ {\boldsymbol {y} | \boldsymbol {z}} \left[ \hat {G} _ {\text { Verifier }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] \right] + \operatorname{Var} _ {\boldsymbol {z}} \left[ \hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \right]
$$

The second term, $\begin{array} { r } { \operatorname { V a r } _ { z } \left\lceil \hat { G } _ { \mathrm { V e r i F r e e } } ( x , y ^ { \star } , z ) \right\rceil } \end{array}$ , is the definition of the variance of the estimator $\hat { G } _ { \mathrm { V e r i F r e e } }$ The first term, $\begin{array} { r } { \mathbb { E } _ { z } \left[ \operatorname { V a r } _ { \pmb { y } | z } \left[ \hat { G } _ { \mathrm { V e r i f i e r } } ( \pmb { x } , \pmb { y } ^ { \star } , z ) \right] \right] } \end{array}$ , is an expectation of a variance. Since variance is always non-negative, $\begin{array} { r } { \operatorname { V a r } _ { \pmb { y } | \mathcal { z } } \left[ \hat { G } _ { \mathrm { V e r i f i e r } } ( \pmb { x } , \pmb { y } ^ { \star } , \pmb { z } ) \right] \geq 0 } \end{array}$ . Therefore, its expectation is also non-negative:

$$
\mathbb {E} _ {\boldsymbol {z}} \left[ \operatorname{Var} _ {\boldsymbol {y} | \boldsymbol {z}} \left[ \hat {G} _ {\text { Verifier}} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \right] \right] \geq 0
$$

Thus, we have:

$$
\operatorname{Var} _ {\boldsymbol {z}, \boldsymbol {y}} \left[ \hat {G} _ {\text { Verifier }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] = (\text { a   non - negative   term }) + \operatorname{Var} _ {\boldsymbol {z}} \left[ \hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \right]
$$

This implies:

$$
\operatorname{Var} _ {\boldsymbol {z}, \boldsymbol {y}} \left[ \hat {G} _ {\text { Verifier }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}, \boldsymbol {y}) \right] \geq \operatorname{Var} _ {\boldsymbol {z}} \left[ \hat {G} _ {\text { VeriFree }} (\boldsymbol {x}, \boldsymbol {y} ^ {\star}, \boldsymbol {z}) \right]
$$

The variance of the policy gradient estimator $\hat { G } _ { \mathrm { V e r i F r e e } }$ is less than or equal to that of $\hat { G } _ { \mathrm { V e r i f i e r } }$ . This is an instance of Rao-Blackwellization, where analytically integrating out a source of randomness (the sampling of y) by using its conditional expectation reduces variance. □

## B Off-policy Gradient Estimators

In the main paper we provide an expression for the gradient estimator when the data is fully on-policy. VeriFree is also fully compatible with PPO-style gradient clipping for the case when data is reused to improve sample efficiency. In this case the gradient estimator is:

$$
\nabla_ {\theta} J _ {\text { VeriFree }} (\theta) = \frac {1}{G} \sum_ {i = 1} ^ {G} \left[ \sum_ {t = 1} ^ {| z _ {i} |} \operatorname{Clip} \left\{\frac {\pi_ {\theta} \left(\boldsymbol {z} _ {i , t} \mid \boldsymbol {x} , \boldsymbol {z} _ {i , <   t}\right)}{\pi_ {\theta_ {\text { old}}} \left(\boldsymbol {z} _ {i , t} \mid \boldsymbol {x} , \boldsymbol {z} _ {i , <   t}\right)} \right\} A _ {i} + \sum_ {t ^ {\prime} = 1} ^ {| \boldsymbol {y} ^ {\star} |} \operatorname{Clip} \left\{\frac {\pi_ {\theta} \left(\boldsymbol {y} _ {t} ^ {\star} \mid \boldsymbol {x} , \boldsymbol {z} _ {i}\right)}{\pi_ {\theta_ {\text { old}}} \left(\boldsymbol {y} _ {t} ^ {\star} \mid \boldsymbol {x} , \boldsymbol {z} _ {i}\right)} \right\} R _ {i} \right],
$$

where $\pi _ { \theta _ { \mathrm { o l d } } }$ is the sampling policy, $\begin{array} { r } { A _ { i } \ = \ \pi _ { \theta _ { \mathrm { o l d } } } ( { \pmb y } ^ { \star } | { \pmb x } , z _ { i } ) - \frac { 1 } { G - 1 } \sum _ { j \neq i } \pi _ { \theta _ { \mathrm { o l d } } } ( { \pmb y } ^ { \star } | { \pmb x } , z _ { j } ) , \ R _ { i } \ = \ } \end{array}$ $\pi _ { \theta _ { \mathrm { o l d } } } ( \pmb { y } ^ { \star } | \pmb { x } , z _ { i } )$ , and $\mathrm { C l i p } \{ \cdot \}$ denotes the PPO clipping operation.

## C Dataset Details

The category distribution in WebData (our training data) is visualized in Fig. 7.

## D Extended Empirical Results

In Tables 1 and 2, we provide detailed benchmark results for MMLU-Pro and SuperGPQA using domain name abbreviations. The full nomenclature is as follows:

MMLU-Pro (Table 1):

CS (Computer Science), Math (Mathematics), Chem (Chemistry), Eng (Engineering), Law (Law), Bio (Biology), Health (Health), Phys (Physics), Bus (Business), Phil (Philosophy), Econ (Economics), Other (Other), Psy (Psychology), Hist (History).

Main Categories

Grouped Categories Detail

Figure 7: Category distributions in WebData. A breakdown of the “grouped” category (right) shows detailed distribution of various domains with fewer data samples.

## SuperGPQA (Table 2):

Eng. (Engineering), Med. (Medicine), Sci. (Science), Phil. (Philosophy),

M.S. (Military Science), Econ. (Economics), Mgmt. (Management), Socio. (Sociology), L/A (Literature and Arts), Hist. (History), Agron. (Agronomy), Law (Law), Edu (Education).

In Table 3, we present an accuracy comparison across six math evaluation benchmarks and GPQA. Models trained with VeriFree demonstrate consistent and significant improvements over the base models, further validating the effectiveness of our approach.

Table 3: Accuracy comparison on math evaluation suite and GPQA.

<table><tr><td rowspan="2">Method</td><td colspan="2">AIME24</td><td colspan="2">AMC</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">Minerva</td><td colspan="2">Olympiad</td><td colspan="2">GPQA</td></tr><tr><td>Acc.</td><td>Len.</td><td>Acc.</td><td>Len.</td><td>Acc.</td><td>Len.</td><td>Acc.</td><td>Len.</td><td>Acc.</td><td>Len.</td><td>Acc.</td><td>Len.</td><td>Acc.</td><td>Len.</td></tr><tr><td>Qwen3-1.7B-Base</td><td>1.7</td><td>1287</td><td>40.0</td><td>1001</td><td>71.4</td><td>343</td><td>58.8</td><td>1094</td><td>19.5</td><td>1071</td><td>23.3</td><td>1737</td><td>17.7</td><td>1355</td></tr><tr><td>Qwen3-1.7B (w/o thinking)</td><td>10.9</td><td>2552</td><td>40.0</td><td>2291</td><td>83.3</td><td>312</td><td>72.8</td><td>1021</td><td>27.9</td><td>776</td><td>39.1</td><td>1970</td><td>19.2</td><td>1885</td></tr><tr><td>Qwen3-1.7B (w/ thinking)</td><td>20.9</td><td>7855</td><td>57.5</td><td>5973</td><td>89.0</td><td>2220</td><td>77.4</td><td>4525</td><td>36.8</td><td>5606</td><td>40.9</td><td>6497</td><td>17.7</td><td>7104</td></tr><tr><td>Qwen3-1.7B-Base-Verifier</td><td>8.4</td><td>2317</td><td>42.5</td><td>1712</td><td>81.7</td><td>405</td><td>66.2</td><td>1057</td><td>30.5</td><td>1357</td><td>29.8</td><td>1885</td><td>36.4</td><td>1317</td></tr><tr><td>Qwen3-1.7B-Base-VeriFree</td><td>10.7</td><td>1783</td><td>37.5</td><td>1522</td><td>76.2</td><td>447</td><td>63.8</td><td>972</td><td>21.0</td><td>867</td><td>29.8</td><td>1484</td><td>30.3</td><td>1147</td></tr><tr><td>Qwen3-4B-Base</td><td>5.2</td><td>1312</td><td>50.0</td><td>1001</td><td>73.1</td><td>393</td><td>73.4</td><td>724</td><td>29.4</td><td>1027</td><td>39.6</td><td>1202</td><td>24.7</td><td>1275</td></tr><tr><td>Qwen3-4B (w/o thinking)</td><td>20.4</td><td>3146</td><td>67.5</td><td>2047</td><td>92.1</td><td>308</td><td>82.2</td><td>1143</td><td>41.2</td><td>822</td><td>49.5</td><td>2593</td><td>29.8</td><td>2127</td></tr><tr><td>Qwen3-4B (w/ thinking)</td><td>33.0</td><td>7750</td><td>62.5</td><td>6123</td><td>92.9</td><td>2261</td><td>84.4</td><td>4370</td><td>41.5</td><td>5352</td><td>47.3</td><td>6432</td><td>31.8</td><td>6504</td></tr><tr><td>Qwen3-4B-Base-Verifier</td><td>15.6</td><td>2407</td><td>57.5</td><td>1566</td><td>72.5</td><td>377</td><td>73.8</td><td>1023</td><td>24.6</td><td>1088</td><td>45.9</td><td>1576</td><td>44.4</td><td>1266</td></tr><tr><td>Qwen3-4B-Base-VeriFree</td><td>16.9</td><td>2706</td><td>65.0</td><td>1904</td><td>87.5</td><td>682</td><td>74.8</td><td>1269</td><td>25.4</td><td>1444</td><td>44.9</td><td>1899</td><td>42.4</td><td>1619</td></tr><tr><td>Qwen2.5-7B</td><td>2.2</td><td>869</td><td>32.5</td><td>922</td><td>84.6</td><td>260</td><td>63.2</td><td>582</td><td>26.8</td><td>819</td><td>30.2</td><td>963</td><td>24.2</td><td>595</td></tr><tr><td>Qwen-2.5-7B-SimpleRL-Zoo</td><td>15.5</td><td>1285</td><td>57.5</td><td>1097</td><td>92.1</td><td>331</td><td>78.6</td><td>690</td><td>36.4</td><td>795</td><td>43.4</td><td>1083</td><td>23.7</td><td>990</td></tr><tr><td>Qwen2.5-Math-7B-Oat-Zero</td><td>28.3</td><td>1115</td><td>65.0</td><td>846</td><td>90.8</td><td>386</td><td>79.0</td><td>652</td><td>33.1</td><td>655</td><td>43.0</td><td>860</td><td>24.7</td><td>722</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>11.2</td><td>993</td><td>52.5</td><td>986</td><td>91.7</td><td>318</td><td>78.2</td><td>649</td><td>37.9</td><td>690</td><td>39.9</td><td>1123</td><td>31.8</td><td>643</td></tr><tr><td>General-Reasoner-7B</td><td>13.1</td><td>1363</td><td>52.5</td><td>1083</td><td>81.7</td><td>408</td><td>74.6</td><td>858</td><td>23.5</td><td>968</td><td>39.3</td><td>1303</td><td>34.8</td><td>1199</td></tr><tr><td>Qwen3-8B-Base</td><td>6.5</td><td>1213</td><td>65.0</td><td>917</td><td>91.7</td><td>304</td><td>77.0</td><td>671</td><td>38.2</td><td>884</td><td>41.3</td><td>1189</td><td>38.9</td><td>887</td></tr><tr><td>Qwen3-8B (w/o thinking)</td><td>24.6</td><td>2897</td><td>62.5</td><td>1919</td><td>93.3</td><td>293</td><td>82.2</td><td>1087</td><td>36.4</td><td>751</td><td>49.5</td><td>2422</td><td>27.8</td><td>2204</td></tr><tr><td>Qwen3-8B (w/ thinking)</td><td>31.0</td><td>7789</td><td>62.5</td><td>6201</td><td>95.3</td><td>2203</td><td>83.0</td><td>4414</td><td>43.8</td><td>5410</td><td>45.2</td><td>6554</td><td>37.4</td><td>6681</td></tr><tr><td>Qwen3-8B-Base-Verifier</td><td>17.8</td><td>1954</td><td>57.5</td><td>1265</td><td>93.4</td><td>307</td><td>76.2</td><td>784</td><td>39.0</td><td>698</td><td>41.8</td><td>1201</td><td>43.9</td><td>733</td></tr><tr><td>Qwen3-8B-Base-VeriFree</td><td>25.2</td><td>2461</td><td>67.5</td><td>1542</td><td>91.6</td><td>375</td><td>80.8</td><td>1024</td><td>30.5</td><td>902</td><td>50.2</td><td>1778</td><td>44.4</td><td>1080</td></tr></table>

## E Case Study

In this section, we provide some examples of reasoning traces produced by the model trained by our method on training questions. And we also provide an example with both our model’s and base model’s responses on the test dataset.

## Example question in training dataset (category: Chemistry)

## Prompt:

<|im\_start|>user

Label or a placard colored black and white with a skull and crossbones falls under which class

Please reason step by step, and put your final answer within <answer> \boxed </answer>.<|im\_end|>

<|im\_start|>assistant

## Our Model’s Response:

A label or placard colored black and white with a skull and crossbones indicates the class of hazardous materials known as \*\*Class 6\*\*. Class 6 is specifically divided into two divisions:

\- \*\*Division 6.1\*\*: Toxic substances. These are materials that can cause injury or death if ingested or inhaled.

\- \*\*Division 6.2\*\*: Infectious substances. These are materials that contain pathogens and can cause disease in humans or animals.

The skull and crossbones symbol is universally recognized as representing toxicity, which corresponds to \*\*Division 6.1\*\* of Class 6 hazardous materials. <answer> \boxed{Class 6} </answer>

## Reference Answer:

Class 6

## Example question in training dataset (category: Physics)

## Prompt:

## <|im\_start|>user

For what type of object is the period-luminosity relation used for determining distances? Please reason step by step, and put your final answer within <answer> \boxed{} </answer>.<|im\_end|>

<|im\_start|>assistant

## Our Model’s Response:

The period-luminosity relation is a fundamental tool in astronomy used to determine distances, particularly for a specific type of star known as Cepheid variables.

Cepheid variables are pulsating stars that exhibit a regular pattern of expansion and contraction. This pulsation causes them to vary in luminosity, and the key insight about these stars is that the period of their luminosity variation is directly related to their intrinsic luminosity (absolute magnitude). In other words, the longer the period of pulsation, the more intrinsically luminous the star is.

This period-luminosity relation was first discovered by Henrietta Leavitt in 1908 by studying Cepheid variables in the Large Magellanic Cloud. Because Cepheids are so bright intrinsically, they can be used as "standard candles" to measure distances within our galaxy and to nearby galaxies.

Here’s how the period-luminosity relation is used to determine distances:

1. \*\*Observe the Period\*\*: First, astronomers measure the period of the star’s brightness variation, which is the time it takes for the star to go from brightest to dimmest and back to brightest.

2. \*\*Determine Intrinsic Luminosity\*\*: Using the period-luminosity relation, they can then determine the star’s intrinsic luminosity. This relation is usually given in a form where a graph or a formula relates period to absolute magnitude (a measure of intrinsic brightness).

3. \*\*Compare to Apparent Luminosity\*\*: The apparent magnitude (how bright the star appears from Earth) can be measured directly.

4. \*\*Calculate Distance\*\*: Using the inverse square law for light, which relates the intrinsic luminosity, apparent brightness, and distance, astronomers can then calculate the distance to the Cepheid variable.

This method is crucial for measuring distances in astronomy because it provides a way to gauge how far away these stars are with a high degree of accuracy. The distances determined using Cepheid variables are themselves used to calibrate other distance measurement techniques, such as those involving supernovae.

Given these details, the type of object for which the period-luminosity relation is used for determining distances is Cepheid variables. <answer> \boxed{Cepheid variables} </answer>

## Reference Answer:

cepheid variable stars

## Example question in MMLU-Pro benchmark (category: Psychology)

Prompt (Here we ignore the prompt template and only show the question for brevity): During the second stage of Kohlberg’s preconventional level of moral development, children obey rules because:

Options are:

A. they are taught that rules are unbreakable.

B. they enjoy following the rules.

C. they are rewarded for following rules.

D. they feel they have a personal duty to uphold rules and laws.

E. doing so helps them satisfy their personal needs.

F. they understand the concept of fairness and justice.

G. doing so helps them avoid punishment.

H. """everyone else is doing it."""

I. they believe in the inherent goodness of rules.

J. they fear societal disapproval.

## Our Model’s Response:

In Kohlberg’s theory of moral development, the preconventional level is the first stage of moral reasoning, and it is primarily focused on self-interest. This level is divided into two stages:

## 1. \*\*Stage 1: Obedience and Punishment Orientation\*\*

In this stage, children obey rules to avoid punishment. Their reasoning is fear-based, and they view rules as set by authority figures. The focus is on the consequences of following or breaking rules, speci fically the avoidance of punishment.

## 2. \*\*Stage 2: Individualism and Exchange\*\*

In this stage, which is the second stage of the preconventional level, children recognize that different people have different needs and interests. They obey rules only when it is in their own interest or when they can benefit from doing so. This stage is often described as the "what’s in it for me?" stage. Children at this level understand the concept of exchange and may follow rules if it leads to a reciprocal benefit.

Now, let’s evaluate the options:

(A) they are taught that rules are unbreakable.

\- This does not align with either stage of the preconventional level.

(B) they enjoy following the rules.

\- Enjoyment of rules is more characteristic of conventional levels of moral development.

(C) they are rewarded for following rules.

\- While this could be somewhat related to Stage 2, the primary focus in Stage 2 is on personal benefit rather than external rewards.

(D) they feel they have a personal duty to uphold rules and laws.

\- This is more aligned with the conventional level, specifically the intermediate stage.

(E) doing so helps them satisfy their personal needs.

\- This could be seen as related to Stage 2, but it is not the most precise description.

(F) they understand the concept of fairness and justice.

\- Understanding fairness and justice is more associated with the conventional and post-conventional levels.

(G) they avoid punishment.

\- This describes Stage 1 of the preconventional level, not Stage 2.

<table><tr><td>Cont.</td></tr><tr><td>(H) everyone else is doing it.- This is more related to conformity and is not a key characteristic of Kohlberg&#x27;s stages.</td></tr><tr><td>(I) they believe in the inherent goodness of rules.- This is more characteristic of the post-conventional level.</td></tr><tr><td>(J) they fear societal disapproval.- Fear of societal disapproval is more related to the conventional level.</td></tr><tr><td>The best fit for Stage 2 of the preconventional level is (E) doing so helps them satisfy their personal needs, as it reflects the idea of mutual benefit and personal interest. However, (G) they avoid punishment describes Stage 1 more accurately. Given the typical interpretations of Kohlberg&#x27;s stages, (E) is the more appropriate choice for Stage 2.</td></tr><tr><td>But let&#x27;s double-check: According to Kohlberg, in Stage 2, children&#x27;s moral reasoning is based on the notion of exchange and the satisfaction of self-interest. Therefore, (E) is the most accurate option.</td></tr><tr><td>&lt;answer&gt;\boxed{E} &lt;/answer&gt;</td></tr><tr><td>Base Model&#x27;s Response:During the second stage of Kohlberg&#x27;s preconventional level of moral development, children obey rules because they understand the concept of fairness and justice. This stage is characterized by a focus on the consequences of actions and the desire to avoid punishment or gain rewards. Children at this stage begin to recognize that rules are not absolute and can be negotiated based on the situation and the needs of the individual. They start to consider the perspectives of others and the fairness of the rules, which is a significant step towards more advanced moral reasoning.</td></tr><tr><td>Therefore, the correct answer is:</td></tr><tr><td>\boxed{F}</td></tr><tr><td>Reference Answer:E</td></tr></table>

## F Broader Impact

Improving the general reasoning capabilities of large language models (LLMs) has the potential to significantly expand their applicability across domains such as education, scientific discovery, legal analysis, and decision support. By removing reliance on explicit verifiers—often expensive or domain-specific—our approach democratizes access to advanced reasoning systems, enabling broader use in resource-constrained settings.

However, improved reasoning can amplify both beneficial and harmful behaviors, depending on the intent and oversight of the user. As such, responsible usage, transparent reporting of model limitations, and value alignment remain critical.

We encourage future work that complements our method with safeguards for misuse prevention and performance guarantees under distributional shifts, particularly as general-purpose reasoning models become more widely integrated into real-world workflows.
