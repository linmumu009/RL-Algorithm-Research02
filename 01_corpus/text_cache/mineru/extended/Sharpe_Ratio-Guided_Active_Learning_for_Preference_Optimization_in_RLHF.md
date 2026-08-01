# Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF

Syrine Belakaria<sup>1</sup>, Joshua Kazdan<sup>1</sup>, Charles Marx<sup>1</sup>, Chris Cundy<sup>1</sup>, Willie Neiswanger<sup>3</sup>, Sanmi Koyejo<sup>1</sup>, Barbara E. Engelhardt<sup>1,</sup> <sup>2</sup>, and Stefano Ermon<sup>1</sup>

<sup>1</sup>Stanford University <sup>2</sup>Gladstone Institutes <sup>3</sup>University of Southern California

## Abstract

Reinforcement learning from human feedback (RLHF) has become a cornerstone of the training and alignment pipeline for large language models (LLMs). Recent advances, such as direct preference optimization (DPO), have simplified the preference learning step. However, collecting preference data remains a challenging and costly process, often requiring expert annotation. This cost can be mitigated by carefully selecting the data points presented for annotation. In this work, we propose an active learning approach to eficiently select prompt and preference pairs using a risk assessment strategy based on the Sharpe Ratio. To address the challenge of unknown preferences prior to annotation, our method evaluates the gradients of all potential preference annotations to assess their impact on model updates. These gradient-based evaluations enable risk assessment of data points regardless of the annotation outcome By leveraging the DPO loss derivations, we derive a closed-form expression for computing these Sharpe ratios on a per-tuple basis, ensuring our approach remains both tractable and computationally eficient. We also introduce two variants of our method, each making diferent assumptions about prior information Experimental results demonstrate that our method outperforms the baseline by up to 5% in win rates against the chosen completion with limited human preference data across several language models and real-world datasets.

## 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) constitutes the final step of training for modern large language models (LLMs) [9]. RLHF ensures that language models align with human preferences in many aspects, including response length [33], helpfulness [21], and lack of harmfulness. RLHF can be used to align models according to any criterion of choice from the user and has been extended beyond language to vision [39, 43] and scientific models [16]. However, unlike pretraining data, which can be scraped in large quantities from sources such as books, archives, and the internet without requiring annotation, RLHF data is costly to gather, as it necessitates expert labeling depending on the specific domain [4, Lee et al.].

RLHF data is generally structured as tuples consisting of a single prompt and multiple candidate responses. In an ideal setup, one response within each tuple is labeled as preferred, while the remaining responses are marked as rejected. Due to the potentially large volume of such tuples, however, labeling them all is prohibitively expensive and impractical. As a result, only a limited subset of these data points can typically be presented to expert annotators. Established RLHF datasets, for instance, often include only a few tens of thousands of these expert-labeled preference pairs, despite the much larger volume of unlabeled data available [3, 13].

The high cost of producing RLHF fine-tuning data leads to investigating more eficient data collection strategies. Models generate millions of responses to human prompts each day; among these, which prompts—if labeled with preference pairs—would provide the greatest benefit during additional RLHF training? Identifying the prompt-responses triplets that yield the highest impact on training could substantially reduce both the time and monetary costs associated with human annotation. This question falls under the broader umbrella of Active Learning(AL), which aims to determine the most informative samples for model training. [27].

Figure 1: Workflow for pool-based active learning in DPO. First, a user asks the LLM questions. The LLM generates two candidate answers to each question. A subset of the question-responses tuples are chosen for labeling by the user. Then, the model is updated using the collected human preferences.

Figure 2: An illustration of the steps of active learning for RLHF using Sharpe Ratio selection criteria.

Active learning algorithms have demonstrated success for both general statistical models [7, 36] and deep learning models under supervised learning [27]. However, relatively few approaches focus on applying AL to RLHF for LLMs. In recent work, Muldrew et al. [23] selected prompt-responses triplets by prioritizing higher reward gaps, while Mehta et al. [22] used uncertainty metrics to target data where the model appeared less confident. Both of these methods implicitly rely on predicting which response will be preferred, incorporating that assumption directly into the selection process. In contrast, our approach accounts for all potential preference outcomes, enabling the assessment of data points regardless of which response is chosen by the expert.

More recently, direct preference optimization (DPO) was proposed as an alternative to the traditional RLHF pipeline that simplifies the process of learning from preference-labeled data [25]. In this work, we present a novel active learning technique that targets an efective selection of data for DPO. We propose to leverage information about the magnitude of gradient update as a selection criterion. Before gathering human preferences about which of two responses is favored, we note that the gradient update will assume one of two forms, depending on which response is set as chosen. As each of these responses is equally likely to be preferred, the resulting gradient update can be seen as a random variable that will settle to one of two values. Rather than relying solely on the expectation or variance of this random variable, we draw inspiration from statistical finance and adopt the Sharpe ratio to characterize and compare the potential updates[32]. The Sharpe ratio naturally balances the expected improvement (mean) against the uncertainty (standard deviation), making it well-suited to pinpoint samples that promise substantial gains while managing risk. Accordingly, we select prompt–responses triplets that yield the highest Sharpe ratios, focusing on cases with the greatest potential for informativeness.

Importantly, we propose a derivation that allows us to obtain a closed-form expression for per-tuple Sharpe ratios, circumventing the need for the memory and computationally intensive multiple backpropagations and keeping our method tractable and eficient. We further introduce two variants of our approach. The first, SHARP (SHarpe Ratio-based Active Requested Preferences), assumes all possible annotations are equally likely. The second, W-SHARP, incorporates the implicit reward model as a prior, producing a weighted version of SHARP that accounts for varying annotation likelihoods.

By applying our procedures, we achieve up to 5% improvement in win rate over the benchmark dataset’s preferred completions, even with a highly constrained data budget, less than 18% of available training tuples in the HH [2] and SHP [13] datasets. We demonstrate the efectiveness of our algorithm across diferent mode scales—specifically Llama-3-8B and Pythia-2.8B—using two state-of-the-art benchmarks: the Helpful-Harmless (HH) dataset [2] and the Stanford Human Preferences (SHP) dataset [13].

To summarize our contributions:

• Drawing inspiration from statistical finance, we introduce a risk assessment approach for active learning in RLHF/DPO. Our method uses the Sharpe ratio of gradient magnitudes to determine which data points are most valuable for labeling.

• We propose two instantiations of our proposed method. The first assumes that each response is equally likely to be chosen as preferred, while the second uses a prior derived from an implicit reward model to weigh the likelihood of each response.

• Leveraging the DPO loss function, we derive fast and memory-eficient closed-form expressions of our acquisition functions.

• We demonstrate improvements in win rates on popular RLHF datasets using three diferent LLMs of varying sizes.

## 2 Background

In this section, we review the details of RLHF and direct preference optimization (DPO). Reinforcement Learning from Human Feedback (RLHF) has emerged as a key approach for aligning language models with human preferences. Originally popularized by works such as Christiano et al. [10] and Stiennon et al. [35], the standard RLHF pipeline begins with a supervised fine-tuning (SFT) phase using high-quality data, followed by training a reward model on preference-labeled examples. In the final phase, the policy is further refined through reinforcement learning, where the reward model, reflecting human feedback, serves as a learned utility function guiding the policy updates via algorithms like Proximal Policy Optimization (PPO) [28, 30].

A major drawback of traditional RLHF was the need to train a reward function, which increases the computational complexity of the alignment step due to the overhead of a separate model. Additionally, reward models are often large, unstable, and might overfit to the preference data [8, 34, 42]. To obviate the need to train a reward function, Rafailov et al. [26] developed direct preference optimization (DPO), an adaptation of the Bradley-Terry model [5] that converts the RLHF pipeline into a preference classification problem and uses the language model and the reference model to form an implicit reward model. Specifically, let x be a prompt and $y$ be a response to this prompt. Denoting the policy model by $\varphi _ { \theta }$ and the reference model by $\varphi _ { \mathrm { r e f } } ,$ The RLHF optimization problem is expressed as:

$$
\max _ {\varphi_ {\theta}} \mathbb {E} _ {x \sim \mathcal {D}, y \sim \varphi_ {\theta} (y | x)} \big [ r _ {\phi} (x, y) \big ] - \beta \mathbb {D} _ {\mathrm{KL}} \big [ \varphi_ {\theta} (y \mid x) \mid \mid \varphi_ {\mathrm{ref}} (y \mid x) \big ]\tag{1}
$$

The optimal solution to the KL-constrained reward maximization objective leads to an expression of the reward model as:

$$
r (x, y) = \beta \log \frac {\varphi_ {\theta} (y | x)}{\varphi_ {\mathrm{ref}} (y | x)} + \beta \log Z (x).\tag{2}
$$

In this equation, $\beta$ is a hyper-parameter that controls the deviation of the policy from the reference policy, and $Z ( x )$ is the partition function that depends only on x. Let $r ^ { * }$ be the ground-truth reward, and $\varphi ^ { * }$ be the optimal policy. Under the Bradley-Terry model [5], the probability that one response is preferred over another is:

$$
p ^ {*} (y _ {1} \succ y _ {2} | x) = \sigma (r ^ {*} (x, y _ {1}) - r ^ {*} (x | y _ {2})).\tag{3}
$$

Substituting in Equation equation 2, the preference probabilities under Bradley-Terry model can be expressed as a function of the optimal RLHF policy $\varphi ^ { * }$ as follows:

$$
p ^ {*} (y _ {1} \succ y _ {2} | x) = \frac {1}{1 + \exp \left(\beta \log \frac {\varphi^ {*} (y _ {2} | x)}{\varphi_ {\mathrm{ref}} (y _ {2} | x)} - \beta \log \frac {\varphi^ {*} (y _ {1} | x)}{\varphi_ {\mathrm{ref}} (y _ {1} | x)}\right)}.\tag{4}
$$

Since we can express the probability of human preference data in terms of the optimal policy rather than a separate reward model, we can construct a maximum likelihood objective for a parameterized policy φ<sub>θ</sub> in terms of the chosen $y _ { w }$ and rejected $y _ { \ell }$ rewards. This produces a preference classification loss function:

$$
\mathcal {L} _ {\mathrm{DPO}} (x, y _ {w}, y _ {l}) = - \log \sigma \left(\beta \log \frac {\varphi_ {\theta} (y _ {w} \mid x)}{\varphi_ {\mathrm{ref}} (y _ {w} \mid x)} - \beta \log \frac {\varphi_ {\theta} (y _ {l} \mid x)}{\varphi_ {\mathrm{ref}} (y _ {l} \mid x)}\right).\tag{5}
$$

While training using the DPO objective, one simultaneously trains the language model and an implicit reward model. This saves substantial time and computation by removing the need to train a separate reward model. In this work, we develop an active learning method for RLHF. Although we experimentally focus on DPO due to its lower computational overhead, our method also applies to RLHF.

## 3 Related Work

Some estimates suggest that over 80% of engineering eforts in machine learning concern data preparation and labeling [14]. Active learning (AL), also referred to as optimal experimental design [24], aims to achieve strong model performance with fewer training samples [1]. The most common use case for active learning occurs when there is a large pool of unlabeled data, and the scientist training a machine learning model must choose which of these data points should be labeled, subject to a labeling budget. In $\mathrm { A L } ,$ an acquisition function applied to the unlabeled data points is used to perform this selection. AL techniques have been applied across various machine learning domains such as support vector machines (SVM) [37], image classification [15], and other areas [29]. Recent eforts in deep active learning (DAL) have focused on text classification [38], image analysis [40], and NLP [17]. Many active learning methods are based on the principle of uncertainty [37], wherein the algorithm prioritizes labeling data points that the model is most uncertain about. Other active learning methods emphasize the importance of diversity and exploration when choosing diferent types of examples to label [12].

Across domains, AL is a notoriously dificult problem [6, 18]. Active learning is especially challenging for RLHF in large-scale models that lack convexity guarantees or bounded noise. Currently, few works tackle the design of acquisition functions in this context. Recently, Mehta et al. [22] and Das et al. [11], Ji et al. [19] formulated active learning for RLHF and DPO as an ofline contextual dueling bandit problem. Mehta et al. [22] proposed an uncertainty-based approach, measuring variance in predicted logits under dropout, while Ji et al. [19] introduced an algorithm with theoretical guarantees on regret and query complexity. Das et al. [11] improved the theoretical guarantees with reduced assumptions. In parallel, Muldrew et al. [23] explored active learning in DPO by first selecting a sub-batch of prompts with high predictive entropy, then further filtering based on large reward gaps, interpreted as lower uncertainty in the DPO model. Beyond dueling bandit frameworks, Zhang et al. [44] introduced a bilevel optimization approach for DPO that favors potentially high-reward responses. Xiong et al. [41] proposed an online exploration method and a rejection sampling strategy for ofline settings, formulated as a reverse-KL-regularized contextual bandit. Although these methods employ diferent exploration or exploitation strategies, they often require a prior assumption about which response is preferred, computing acquisition scores under that assumption. Ideally, an active learning approach should consider all possible preference outcomes without relying on a predefined guess. Our method fulfills this criterion, ofering a first attempt at a risk-based perspective that balances exploration and exploitation more comprehensively.

## 4 Problem Setting

Consider a practitioner who wishes to fine-tune a large language model (LLM) via reinforcement learning from human feedback (RLHF) in a specific domain. The practitioner has access to a large pool of unlabeled data,

$$
\mathcal {D} = \left\{\left(x _ {i}, y _ {i 1}, y _ {i 2}\right) \right\} _ {i = 1} ^ {n}
$$

where n is large, and each entry consists of a prompt $x _ { i }$ along with two candidate responses $y _ { i 1 }$ and $y _ { i 2 }$ . Owing to the high cost and impracticality of labeling every entry in $\mathcal { D } ,$ only a small subset $\mathcal { D } _ { L } \subseteq \mathcal { D }$ can be annotated with expert preferences (i.e., which of $y _ { i 1 }$ or $y _ { i 2 }$ is preferred).

Once the practitioner obtains b labeled triplets from $\mathcal { D } _ { L }$ , a direct preference optimization $( D P O )$ update is performed on the LLM. The model is then used to query a new batch of unlabeled data for expert feedback, and this iterative process continues until the labeling budget is exhausted. The key challenge is to select the most informative triplets for labeling to maximize the final performance of the RLHF-fine-tuned model under strict budget constraints.

To closely mirror practical scenarios of collecting and deploying preference data, we require a criterion that identifies the most valuable prompts for human annotation. In our experimental setup, we model this situation as follows:

1. For each prompt and response pair in a large batch of size $b \times p .$ , evaluate a designed selection criterion, where $p$ is a user-defined fraction indicating the annotation budget. We use this strategy as a practical search procedure.

2. Rank all triplets based on the selection criterion and select the top b to label.

3. Using the labeled preference pairs and perform a single DPO update.

## 5 Sharpe Ratio for Active Preference Learning

## 5.1 Method Description

We propose a novel method to eficiently collect human preference data in an online setting. Our strategy maximizes the gradient magnitude derived from the DPO objective on the selected data, thereby using information about model parameters when deciding which samples will have the greatest training impact.

A key challenge arises because we cannot compute the DPO gradient without knowing which response is actually preferred. However, we do know that, for each prompt x with candidate responses $y _ { 1 }$ and $y _ { 2 }$ , the gradient will assume exactly one of two possible forms: one if $y _ { 1 }$ is preferred, and another if $y _ { 2 }$ is preferred. Let $\varphi _ { \mathrm { r e f } }$ denote the reference model. Depending on which response is ultimately chosen, the DPO update takes one of the following two forms:

$$
\begin{array}{c} G _ {1} = \nabla_ {\theta} \mathcal {L} _ {\mathrm{DPO}} (x, y _ {1}, y _ {2}) = - \nabla_ {\theta} \log \sigma \Big (\beta \log \frac {\varphi_ {\theta} (y _ {1} | x)}{\varphi_ {\mathrm{ref}} (y _ {1} | x)} - \beta \log \frac {\varphi_ {\theta} (y _ {2} | x)}{\varphi_ {\mathrm{ref}} (y _ {2} | x)} \Big) \\ = - \beta \sigma (\hat {r} _ {\theta} (x, y _ {2}) - \hat {r} _ {\theta} (x, y _ {1})) \times \left[ \nabla_ {\theta} \log \varphi_ {\theta} (y _ {1} | x) - \nabla_ {\theta} \log \varphi_ {\theta} (y _ {2} | x) \right] \end{array}
$$

$$
\begin{array}{c} G _ {2} = \nabla_ {\theta} \mathcal {L} _ {\mathrm{DPO}} (x, y _ {2}, y _ {1}) = - \nabla_ {\theta} \log \sigma \Big (\beta \log \frac {\varphi_ {\theta} (y _ {2} | x)}{\varphi_ {\mathrm{ref}} (y _ {2} | x)} - \beta \log \frac {\varphi_ {\theta} (y _ {1} | x)}{\varphi_ {\mathrm{ref}} (y _ {1} | x)} \Big) \\ = - \beta \sigma (\hat {r} _ {\theta} (x, y _ {1}) - \hat {r} _ {\theta} (x, y _ {2})) \times \big [ \nabla_ {\theta} \log \varphi_ {\theta} (y _ {2} | x) - \nabla_ {\theta} \log \varphi_ {\theta} (y _ {1} | x) \big ]. \end{array}
$$

Let G be the random variable defined by the magnitude of the gradient update that is obtained by soliciting human feedback for the $( x , y _ { 1 } , y _ { 2 } )$ triplet. Let $p _ { 1 } = p ( y _ { 1 } \succ y _ { 2 } | x )$ be the probability that $y _ { 1 }$ is preferred to y<sub>2</sub> and $p _ { 2 } = p ( y _ { 2 } \succ y _ { 1 } | x )$ be the probability that $y _ { 2 }$ is preferred to $y _ { 1 }$

The expectation of G is defined as:

$$
\mathbb {E} [ G ] = p _ {1} \| G _ {1} \| + p _ {2} \| G _ {2} \|.\tag{6}
$$

The variance of G is defined as:

$$
\sigma^ {2} (G) = p _ {1} (\| G _ {1} \| - \mathbb {E} [ G ]) ^ {2} + p _ {2} (\| G _ {2} \| - \mathbb {E} [ G ]) ^ {2}.\tag{7}
$$

The expectation alone is not a good decision metric when selecting which responses should be labeled for several reasons. First, suppose that one response is gibberish, and the other is sensible. The gradient in which the gibberish response is the preferred response would likely be large, and therefore, the expectation would be high. However, selecting a tuple where one of the responses is gibberish will not lead to an informative update to the model. Thus, we need some way to account for the variance of G. To do this, we use a tool from statistical finance: the Sharpe ratio. The Sharpe ratio [31], invented by William Sharpe in the 1960s, evaluates not just the expected value of an investment portfolio but also the risk. For example, one would likely eschew investment opportunities that could result in losing one’s entire life savings, even if these investmen opportunities had a high expected value. We apply the same logic when selecting which preference pairs to label. We want to maximize the expected magnitude of our gradient updates but reduce the risk of getting a small gradient update if a certain response is preferred. By choosing to label the preference pairs that yield the highest Sharpe ratio, we accomplish this goal. Because we drew inspiration for our method of active learning from the Sharpe ratio metric, we name our method SHarpe Ratio-based Active Requested Preferences, or SHARP for short. The Sharp ratio of a triplet $( x , y _ { 1 } , y _ { 2 } )$ is defined as:

$$
S R (G) = \frac {\mathbb {E} [ G ]}{\sigma (G)}\tag{8}
$$

In our active learning setting, we select triplets that yield the highest Sharpe ratio. We define an acquisition function for the current policy $\varphi _ { \theta }$ as:

$$
\alpha_ {\varphi_ {\theta}} (x, y _ {1}, y _ {2}) = S R (G).\tag{9}
$$

SHARP: No Prior Acquisition Function: Before querying the expert labeling of the preference, we might have no prior assumption about which response might be preferred to the other. In this case, we can assume that $y _ { 1 }$ and $y _ { 2 }$ are equally likely to be the better response, and therefore $p _ { 1 } = p _ { 2 } = 0 . 5$ . We consider this the no prior version of our method, and we refer to it as SHARP.

W-SHARP: Prior-based Acquisition Function: The RLHF/DPO pipeline usually initializes the policy φ<sub>θ</sub> to the SFT policy previously finetuned on data related to the same domain or topic of interest. This model can provide us with a prior for the preference probabilities $p _ { 1 }$ and $p _ { 2 }$ . For instance, in the DPO setting, we can derive an implicit reward model from $\varphi _ { \theta }$ and $\begin{array} { r } { \varphi _ { \mathrm { r e f } } , r _ { \theta } ( x , y ) = \beta \log \frac { \varphi _ { \theta } ( y | x ) } { \varphi _ { \mathrm { r e f } } ( y | x ) } } \end{array}$ and then set the probabilitie $p _ { 1 }$ and $p _ { 2 }$ based on Equation 4 during the active learning iterations. We refer to this version of our method as weighted SHARP (W-SHARP).

## 5.2 Eficient Execution of SHARP with DPO

In practice, computing the Sharpe ratio would require computing the gradient for each element in the dataset twice and consequently backpropagating through the $\mathrm { L L M 2 } \times B$ times for each batch of size B instead of a single batch-wise backpropagation. This procedure is computationally expensive in terms of both time and memory. To overcome this bottleneck, we use the closed-form expression of the gradient of the DPO loss function to simplify the final expression of the SHARP acquisition functions. Given the final expression of the gradient of the DPO loss, we can express $G _ { 2 }$ as a function of $G _ { 1 }$ :

$$
\begin{array}{r l} & G _ {2} = - \beta \sigma (\hat {r} _ {\theta} (x, y _ {1}) - \hat {r} _ {\theta} (x, y _ {2})) \times \left[ \nabla_ {\theta} \log \varphi_ {\theta} (y _ {2} | x) - \nabla_ {\theta} \log \varphi_ {\theta} (y _ {1} | x) \right] \\ & \quad = - \beta \big [ \sigma (\hat {r} _ {\theta} (x, y _ {2}) - \hat {r} _ {\theta} (x, y _ {1})) - 1 \big ] \times \left[ \nabla_ {\theta} \log \varphi_ {\theta} (y _ {1} | x) - \nabla_ {\theta} \log \varphi_ {\theta} (y _ {2} | x) \right] \\ & \quad = G _ {1} \Big [ 1 - \frac {1}{\sigma (\hat {r} _ {\theta} (x , y _ {2}) - \hat {r} _ {\theta} (x , y _ {1}))} \Big ]. \end{array}
$$

Consequently, we have:

$$
\| G _ {2} \| = \| G _ {1} \| \cdot \| \gamma \|,\tag{10}
$$

with $\begin{array} { r } { \left\| \gamma \right\| = \left\| 1 - \frac { 1 } { \sigma ( \hat { r } _ { \theta } ( x , y _ { 2 } ) - \hat { r } _ { \theta } ( x , y _ { 1 } ) ) } \right\| . } \end{array}$

Combining Equations 8, Equation 6, and Equation 7, we get an expression of the Sharpe ratio as follows:

$$
S R (G) = \frac {p _ {1} \| G _ {1} \| + p _ {2} \| G _ {2} \|}{\sqrt {p _ {1} (\| G _ {1} \| - \mathbb {E} [ G ]) ^ {2} + p _ {2} (\| G _ {2} \| - \mathbb {E} [ G ]) ^ {2}}}.
$$

By substituting the expression of $\| G _ { 2 } \|$ from Equation 10, we obtain the final form of the Sharpe ratio, in which the gradient terms $\| G _ { 1 } \|$ cancel out in both the numerator and the denominator.

$$
\begin{array}{c} S R (G) = \frac {\| G _ {1} \| (p _ {1} + p _ {2} \| \gamma \|)}{\sqrt {p _ {1} (\| G _ {1} \| - \| G _ {1} \| (p _ {1} + p _ {2} \| \gamma \|)) ^ {2} + p _ {2} (\| G _ {1} \| \cdot \| \gamma \| - \| G _ {1} \| (p _ {1} + p _ {2} \| \gamma \|)) ^ {2}}} \\ = \frac {(p _ {1} + p _ {2} \| \gamma \|)}{\sqrt {p _ {1} (1 - (p _ {1} + p _ {2} \| \gamma \|)) ^ {2} + p _ {2} (\| \gamma \| - (p _ {1} + p _ {2} \| \gamma \|)) ^ {2}}}. \end{array}\tag{11}
$$

(12)

In the case of W-SHARP, we substitute the probabilities $p _ { 1 }$ and $p _ { 2 }$ by the preference probabilities obtained by combining the implicit reward model and the Bradley-Terry preference model [5]:

$$
\alpha_ {\varphi_ {\theta}} ^ {W - S H A R P} (x, y _ {1}, y _ {2}) = S R (G),\tag{13}
$$

with SR(G) defined in Equation 12. In the case of SHARP, where we assume that we do not have any prior about the preference probabilities, we have $\begin{array} { r } { p _ { 1 } = p _ { 2 } = \frac { 1 } { 2 } } \end{array}$ . The acquisition function expression can be further simplified as follows:

$$
\alpha_ {\varphi_ {\theta}} ^ {S H A R P} (x, y _ {1}, y _ {2}) = \frac {\frac {1}{2} (1 + \| \gamma \|)}{\sqrt {\frac {1}{2} (1 - \frac {1}{2} (1 + \| \gamma \|)) ^ {2} + \frac {1}{2} (\| \gamma \| - \frac {1}{2} (1 + \| \gamma \|)) ^ {2}}}.\tag{14}
$$

By simplifying Equation 14, we obtain the final expression:

$$
\alpha_ {\varphi_ {\theta}} ^ {S H A R P} (x, y _ {1}, y _ {2}) = \frac {1 + \| \gamma \|}{| 1 - \| \gamma \| |}.\tag{15}
$$

By leveraging the gradient expression of the DPO loss and the relationship between swapped-preference gradients and the Sharpe ratio, our derivation provides a \*\*closed-form formula\*\* for per-tuple Sharpe ratios. This circumvents the need for multiple backpropagations, significantly reducing both memory and computation costs and keeping the method tractable. Crucially, without this derivation, although the approach might still be conceptually valid and useful, it would be prohibitively impractical in real-world applications.

We execute SHARP and W-SHARP on each batch of incoming unlabeled prompt-responses triplets to select a sub-batch for human labeling. SHARP proceeds as in Algorithm 1.

## 6 Experiments

In this section, we provide the details of our evaluation pipeline. Our main goal is to determine if we can achieve better or comparable performance as DPO while using a smaller amount of labeled data. To assess whether our approaches enhance data selection in DPO, we conduct experiments training large language models (LLMs) on two datasets applied to three diferent models with diferent ranges of sizes. When comparing the two approaches, we keep all parameters of the experiments identical besides the data selection method to isolate and verify its impact on performance.

Datasets We evaluate both methods on two public datasets: the Anthropic Helpful-Harmless (HH) dataset [2] and the Stanford Human Preferences (SHP) dataset [13].

Anthropic Helpful-Harmless (HH): The HH dataset is designed to measure an AI assistant’s ability to be both helpful and harmless. It contains two main types of examples: queries where the user’s request is reasonable, and the assistant should provide a helpful response, and queries where the user’s request may be harmful or inappropriate, requiring the assistant to prioritize safety by giving a non-harmful response.

Stanford Human Preferences (SHP): The SHP dataset consists of Reddit posts and corresponding humangenerated comments spanning 18 diferent categories. This broad coverage provides diverse human writing styles and topics. SHP focuses on modeling general human preferences across a wide range of real-world conversations.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 SHARP Data Selection Algorithm

Inputs: policy  $\varphi_{\theta}$ , reference policy  $\varphi_{ref}$ , exploration parameter  $\beta$ , batch size b, number of iterations N, a dataset  $\mathcal{D} = \{(x_i, y_{i1}, y_{i2})\}_{i=1}^n$ , the fraction p of the batch that we can afford to label.

Output: A subset of the data  $D_L \subset D$  of triplets of expert labeling with  $|D_L| = b \times N$ , Updated  $\varphi_{\theta}$ .

1: for  $t = 1, \ldots, N$  do

2: Draw a large batch of triplets  $B = \{(x_i, y_{i1}, y_{i2})_{i=1}^{(b.p)}\} \sim D$ .

3: for  $(x_i, y_{i1}, y_{i2}) \in B$  do

4: If using SHARP method, compute  $\alpha_{\varphi_{\theta}}^{SHARP}(x_i, y_{i1}, y_{i2})$ 

5: If using W-SHARP method, compute  $\alpha_{\varphi_{\theta}}^{W-SHARP}(x_i, y_{i1}, y_{i2})$ 

6: end for

7: Let  $B_L$  be the top-b elements of B by the value of the acquisition function  $\alpha$ .

8: Request the preferences labels from the expert and add them to  $D_L$ 

9: Update the policy  $\varphi_{\theta}$  using a gradient step of the  $L_{DPO}$  using  $B_L$ 

10: end for

11: return  $D_L$  and  $\varphi_{\theta}$ .
</div>

```txt
Win Rate of GPT-2 Policies on HH, Win Rate of Pythia-2.8B Policies on HH, Win Rate of Llama-3-8B Policies on HH, Win Rate of GPT-2 Policies on SHP, Win Rate of Pythia-2.8B Policies on SHP, Win Rate of Llama-3-8B Policies on SHP
Average Win Rate Over Chosen
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHAPP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPO
W-SHARP-DPO
DPO
SHARP-DPRO
```  
Figure 3: Comparison of W-SHARP-DPO and SHARP-DPO against DPO across diferent models and datasets. The metric is the average win rate over chosen completions, computed with GPT-4o under swapped evaluation orders to reduce positional bias. Error bars indicate the standard error.

LLMs: We explore the impact of active learning by evaluating three models of varying size and capacity: GPT-2, Pythia-2.8-B, and Llama-3-8B. These models span a broad range of resource requirements and capabilities, allowing us to assess how active learning strategies perform under diferent constraints. We conduct six distinct experiments using the above datasets to provide a comprehensive analysis of each model’s

## performance.

Pipeline Setup: In the DPO pipeline, we begin by splitting each dataset into training and test sets. During the Supervised Fine-Tuning (SFT) phase, we finetune each model on the training split, updating all parameters in each gradient step. In the subsequent DPO phase, to eficiently manage computational resources, we apply a quantized Low-Rank Adaptation (LoRA) of each LLM. This approach reduces memory footprint and speeds up experimentation without sacrificing the model’s overall performance. We apply 4-bit quantization using a double quantization strategy under the NF4 scheme while computing in bfloat16. In addition, we use a LoRA configuration with rank 16, alpha 32, and a dropout rate of 0.05, tailored for causal language modeling tasks and omitting additional bias. We set the batch size of our training to 64 and set the fraction that would be labeled to p = 6.

We evaluate model performance using the winrate against the dataset’s designated “chosen” completions. Formally, the winrate indicates the proportion of generated responses that are deemed preferable to those labeled as chosen in the dataset. We recompute this metric after every 4,096 training samples to track performance trends over time.

To underscore the benefits of active learning under constrained resources, we limit the DPO phase to a total of 28,672 training points across all experiments. Additionally, we use GPT-4o as an evaluation oracle to compare each newly generated response with the baseline DPO generation. To mitigate position bias, each pair of responses is evaluated twice with reversed ordering, and we report the average winrate across these two evaluations.

Both W-SHARP-DPO and SHARP-DPO consistently outperform the standard DPO baseline (Figure 3). We attribute this improvement to our acquisition function α, which takes the risk (i.e., all possible gradient outcomes) into account when selecting data points. Interestingly, W-SHARP-DPO and SHARP-DPO achieve similar performance, suggesting that incorporating the implicit reward model as a prior does not necessarily yield further gains in this setting. This could indicate that while using a prior might help in other contexts, it is not required for efective data selection, and making no prior assumption could be beneficial for risk assessment.

The accuracy of the implicit reward model for experiments conducted on both datasets echos this result (Figure 4, Appendix). Although this metric is not our primary focus, the results indicate that both SHARP and W-SHARP tend to attain higher accuracy more quickly on the test data, suggesting that these methods guide the model toward more efective reward predictions.

## 7 Summary, Future Directions, and Limitations

We present a novel active learning strategy for RLHF/DPO in large language models, designed to prioritize and label the most impactful data points under limited human annotation budgets. Central to our method is the use of a Sharpe ratio-based acquisition function to evaluate potential gradient updates. By selecting examples with the highest Sharpe ratios, we aim to target those most likely to produce substantial improvements in policy performance. Our empirical results suggest that this risk-aware selection can reduce annotation costs while enhancing the quality of the learned policy.

Our current approach focuses exclusively on high Sharpe ratio data, which may bias the distribution of selected examples. Although such selective sampling is typical in active learning scenarios, if a practical setting requires an unbiased estimate of the underlying data distribution, future methods could address potential deviations arising from risk-based sampling. Potentially, future methods could combine our Sharpe ratio-based approach with techniques like importance sampling or explicit expectation balancing to address such requirements. Moreover, our computational study was limited by relatively modest resources, restricting the scale of DPO training and the range of datasets evaluated. While our findings demonstrate the promise of a Sharpe ratio-based framework, additional investigation—across larger tasks and more extensive experiments would establish its robustness and generalizability.

## References

[1] Alizadeh, A., Tavallali, P., Khosravi, M. R., and Singhal, M. (2021). Survey on recent active learning methods for deep learning.

[2] Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022a). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

[3] Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., Mann, B., and Kaplan, J. (2022b). Training a helpful and harmless assistant with reinforcement learning from human feedback.

[4] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. (2022c). Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

[5] Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345.

[6] Castro, R. M. and Nowak, R. D. (2007). Minimax bounds for active learning. IEEE Transactions on Information Theory, 54:2339–2353.

[7] Castro, R. M., Willett, R. M., and Nowak, R. D. (2005). Faster rates in regression via active learning. In Neural Information Processing Systems.

[8] Chaudhari, S., Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K., Deshpande, A., and da Silva, B. C. (2024). Rlhf deciphered: A critical analysis of reinforcement learning from human feedback for llms. ArXiv, abs/2404.08555.

[9] Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. (2017a). Deep reinformcement learning from human preferences. In Advances in Neural Information Processing Systems 30 (NIPS 2017).

[10] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017b). Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

[11] Das, N., Chakraborty, S., Pacchiano, A., and Chowdhury, S. R. (2024). Provably sample eficient rlhf via active preference optimization. arXiv preprint arXiv:2402.10500.

[12] Doucet, P., Estermann, B., Aczel, T., and Wattenhofer, R. (2024). Bridging diversity and uncertainty in active learning with self-supervised pre-training. arXiv preprint arXiv:2403.03728.

[13] Ethayarajh, K., Choi, Y., and Swayamdipta, S. (2022). Understanding dataset dificulty with V-usable information. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S., editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5988–6008. PMLR.

[14] Fredriksson, T., Mattos, D. I., Bosch, J., and Olsson, H. H. (2020). Data labeling: An empirical investigation into industrial challenges and mitigation strategies. In Morisio, M., Torchiano, M., and Jedlitschka, A., editors, Product-Focused Software Process Improvement, pages 202–216, Cham. Springer International Publishing.

[15] Gal, Y., Islam, R., and Ghahramani, Z. (2017). Deep bayesian active learning with image data. In International conference on machine learning, pages 1183–1192. PMLR.

[16] Gu, S., Xu, M., Powers, A., Nie, W., Gefner, T., Kreis, K., Leskovec, J., Vahdat, A., and Ermon, S. (2025). Aligning target-aware molecule difusion models with exact energy optimization. Advances in Neural Information Processing Systems, 37:44040–44063.

[17] Hadian, H. and Sameti, H. (2014). Active learning in noisy conditions for spoken language understanding. In Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers, pages 1081–1090, Dublin, Ireland. Dublin City University and Association for Computational Linguistics.

[18] Hanneke, S. and Yang, L. (2010). Negative results for active learning with convex losses. In Teh, Y. W. and Titterington, M., editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 321–325, Chia Laguna Resort, Sardinia, Italy. PMLR.

[19] Ji, K., He, J., and Gu, Q. (2024). Reinforcement learning from human feedback with active queries. arXiv preprint arXiv:2402.09401.

[Lee et al.] Lee, H., Phatale, S., Mansoor, H., Mesnard, T., Ferret, J., Lu, K. R., Bishop, C., Hall, E., Carbune, V., Rastogi, A., et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. In Forty-first International Conference on Machine Learning.

[21] Li, A., Xiao, Q., Cao, P., Tang, J., Yuan, Y., Zhao, Z., Chen, X., Zhang, L., Li, X., Yang, K., Guo, W., Gan, Y., Yu, X., Wang, D., and Shan, Y. (2024). Hrlaif: Improvements in helpfulness and harmlessness in open-domain reinforcement learning from ai feedback.

[22] Mehta, V., Das, V., Neopane, O., Dai, Y., Bogunovic, I., Schneider, J., and Neiswanger, W. (2023). Sample eficient reinforcement learning from human feedback via active exploration.

[23] Muldrew, W., Hayes, P., Zhang, M., and Barber, D. (2024). Active preference learning for large language models. arXiv preprint arXiv:2402.08114.

[24] Olsson, F. (2009). A literature survey of active machine learning in the context of natural language processing.

[25] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. (2023a). Direct preference optimization: Your language model is secretly a reward model.

[26] Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2023b). Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

[27] Ren, P., Xiao, Y., Chang, X., Huang, P.-Y., Li, Z., Gupta, B. B., Chen, X., and Wang, X. (2021). A survey of deep active learning.

[28] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. ArXiv, abs/1707.06347.

[29] Settles, B. (2009). Active learning literature survey.

[30] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. (2024). Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

[31] Sharpe, W. F. (1966). Mutual fund performance. The Journal of Business, 39(1):119–138.

[32] Sharpe, W. F. (1998). The sharpe ratio. Streetwise–the Best of the Journal of Portfolio Management, 3:169–185.

[33] Singhal, P., Goyal, T., Xu, J., and Durrett, G. (2024). A long way to go: Investigating length correlations in rlhf.

[34] Skalse, J., Howe, N. H. R., Krasheninnikov, D., and Krueger, D. (2022). Defining and characterizing reward hacking. ArXiv, abs/2209.13085.

[35] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. (2020). Learning to summarize from human feedback.

[36] Tong, S. and Koller, D. (2000). Active learning for parameter estimation in bayesian networks. In Neural Information Processing Systems.

[37] Tong, S. and Koller, D. (2001). Support vector machine active learning with applications to text classification. Journal of machine learning research, 2(Nov):45–66.

[38] Tuia, D., Volpi, M., Copa, L., Kanevski, M., and Munoz-Mari, J. (2011). A survey of active learning algorithms for supervised remote sensing image classification. IEEE Journal of Selected Topics in Signal Processing, 5(3):606–617.

[39] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. (2024). Difusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238.

[40] Wang, H., Jin, Q., Li, S., Liu, S., Wang, M., and Song, Z. (2023). A comprehensive survey on deep active learning in medical image analysis. arXiv preprint arXiv:2310.14230.

[41] Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. (2024). Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In Forty-first International Conference on Machine Learning.

[42] Yan, Y., Lou, X., Li, J., Zhang, Y., Xie, J., Yu, C., Wang, Y., Yan, D., and Shen, Y. (2024). Reward-robust rlhf in llms. ArXiv, abs/2409.15360.

[43] Yang, K., Tao, J., Lyu, J., Ge, C., Chen, J., Shen, W., Zhu, X., and Li, X. (2024). Using human feedback to fine-tune difusion models without any reward model. In CVPR, pages 8941–8951.

[44] Zhang, S., Yu, D., Sharma, H., Yang, Z., Wang, S., Hassan, H., and Wang, Z. (2024). Self-exploring language models: Active preference elicitation for online alignment. arXiv preprint arXiv:2405.19332.

## A Additional Results

In this section, we provide additional results reporting the accuracy of the implicit reward model on the test set. To provide informative results, we use exponential moving average smoothing.






Figure 4: The accuracy of the implicit reward model for models.
