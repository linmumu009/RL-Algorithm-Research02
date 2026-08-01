# Active Learning for Direct Preference Optimization

Branislav Kveton <sup>1</sup> Xintong Li <sup>2</sup> Julian McAuley <sup>2</sup> Ryan Rossi <sup>1</sup> Jingbo Shang <sup>2</sup> Junda Wu <sup>2</sup> Tong Yu <sup>1</sup>

## Abstract

Direct preference optimization (DPO) is a form of reinforcement learning from human feedback (RLHF) where the policy is learned directly from preferential feedback. Although many models of human preferences exist, the critical task of selecting the most informative feedback for training them is under-explored. We propose an active learning framework for DPO, which can be applied to collect human feedback online or to choose the most informative subset of already collected feedback offline. We propose efficient algorithms for both settings. The key idea is to linearize the DPO objective at the last layer of the neural network representation of the optimized policy and then compute the D-optimal design to collect preferential feedback. We prove that the errors in our DPO logit estimates diminish with more feedback. We show the effectiveness of our algorithms empirically in the setting that matches our theory and also on large language models.

## 1. Introduction

Reinforcement learningfrom humanfeedback (RLHF) has been effective in aligning and fine-tuning large language models (LLMs) (Ouyang et al., 2022; Rafailov et al., 2023). The main difference from classic reinforcement learning (RL) (Sutton and Barto, 1998) is that the agent learns from human feedback, which is expressed as preferences for different potential choices (Christiano et al., 2017). The human feedback allows LLMs to be adapted beyond the distribution of data that was used for their pre-training and generate more human-like responses. The feedback can be incorporated by learning a reward model (Ouyang et al., 2022) from preferences over two (Bradley and Terry, 1952) or multiple (Plackett, 1975; Luce, 2005) choices. Proximal policy optimization (PPO) (Schulman et al., 2017) is then used to maximize the expected reward of the LLM policy under the reward model. Learning of reward models can be avoided by directly optimizing the policy with preferential feedback, known as direct preference optimization (DPO) (Rafailov et al., 2023).

Learning of human preferences for LLM optimization has two main components: preference modeling (Rafailov et al., 2023; Ethayarajh et al., 2024) and how the preferences are elicited (Lightman et al., 2024). We focus on the latter and note that this problem is analogous to classic active learning (Bishop, 2006). Prior works formulated this problem as identifying a subset of prompts with candidate responses, either online or offline, where preferential feedback would improve policy learning by RLHF, either through a reward model or DPO. These works differ in how the prompts are selected: Mehta et al. (2023); Ji et al. (2024); Muldrew et al. (2024) choose prompts based on differences of estimated rewards to their responses; Mukherjee et al. (2024); Scheid et al. (2024); Thekumparampil et al. (2024) derive optimal policies for offline exploration using D-optimal designs (Pukelsheim, 2006); and Das et al. (2024); Liu et al. (2024) solve D-optimal designs online using a greedy algorithm. Most works prove that the errors in learned reward models diminish with more feedback. Interestingly, many works propose two kinds of algorithms (Mehta et al., 2023; Das et al., 2024; Ji et al., 2024), which are either analyzable or practical. We present the first analysis of active learning in DPO and our algorithms are practical.

We study active learning in direct preference optimization. At a high level, we collect preferential feedback to improve DPO policies learned from it. We study two settings: online and offline. In the online setting, the input is a dataset of N prompts with two candidate responses per prompt. The human feedback is unknown in advance and we elicit it online. This setting is motivated by statistical efficiency; we elicit the most informative feedback within a fixed budget on human labor. In the offline setting, the input is a dataset of N prompts with two candidate responses per prompt, and logged preferential feedback for the responses. This setting is motivated by computational efficiency; even if the human feedback is known in advance, we may not have computational resources to learn from all of it. We solve both settings in a unified way. The key idea in our work is to linearize the DPO objective at the last layer of the neural network representation of the optimized policy and identify the most informative subset of n prompts out of N using a

D-optimal design (Pukelsheim, 2006). D-optimal designs are a well-established tool in adaptive learning (Lattimore and Szepesvari, 2019) for near-optimal information gathering. Several recent papers applied them to learning reward models in RLHF (Das et al., 2024; Mukherjee et al., 2024; Liu et al., 2024; Scheid et al., 2024).

We make the following contributions:

1. We formalize active learning for DPO as choosing a subset of n data points out of N such the error in DPO logits, the log odds of preferring one response to the other, is minimized (Section 3).

2. This is the first work that derives a D-optimal design for DPO (Section 4). The key idea is to assume loglinear policies, which linearize the DPO objective at the last layer of the neural network policy representation. The derived D-optimal design resembles tha of logistic regression, with additional terms due to the reference policy and regularization by it. We propose two computationally-efficient algorithms, ADPO and $\mathtt { A D P 0 ^ { + } }$ , which select the most informative data points for DPO. ADPO elicits preferential feedback online and $\tt A D P 0 ^ { + }$ leverages previously logged preferential feedback to have a better design.

3. We analyze ADPO and $\mathrm { \ A D P 0 ^ { + } }$ , and show that their logit errors are ${ \tilde { O } } ( d / { \sqrt { n } } )$ , where d is the number of features in the linearized DPO policies and n is the budget on preferential human feedback. This is the first analysis for DPO and has several novel technical aspects. The main technical trick is relating the feedback model and policy parameter under the assumption of log-linear policies. Therefore, we can argue for concentration of the policy parameter with more feedback. The analysis is also under a practical assumption that preferential feedback can be elicited at most once per prompt. To attain a ${ \tilde { O } } ( d / { \sqrt { n } } )$ rate in this setting, we introduce a novel assumption on the sufficient diversity of prompts and candidate responses.

4. We evaluate ADPO and $\mathtt { A D P 0 ^ { + } }$ empirically. We experiment with both log-linear DPO policies, which match our theory, and on LLMs. Our methods perform well empirically, despite the fact that they are the first ones with an analysis for active learning in DPO.

The paper is structured as follows. In Section 2, we introduce classic methods for training LLMs. In Section 3, we introduce active learning for DPO. We introduce our algorithms in Section 4 and analyze them in Section 5. In Section $^ { 6 , }$ we evaluate our algorithms empirically. We review related work in detail in Appendix C and conclude in Section 7.

## 2. Background

We start by introducing our notation. The prompt is a string $x \in { \mathcal { Z } }$ , where Z is the space of all strings. The response is a string $y \in { \mathcal { Z } }$ . A large language model (LLM) is a policy that maps x to y. We denote the probability of generating response y to prompt x by a policy parameterized by $\theta \in \Theta$ by $\pi ( \boldsymbol { y } \mid \boldsymbol { x } ; \boldsymbol { \theta } )$ , where Θ is the space of policy parameters. To simplify terminology, we call θ a policy when it is clear that we refer to $\pi ( \cdot \mid \cdot ; \theta )$ . Pre-trained LLMs can be optimized by supervised fine-tuning (Mangrulkar et al., 2022; Hu et al., 2022) and reinforcement learning from human feedback, which may require learning of a reward model (Ouyang et al., 2022) or not (Rafailov et al., 2023). These methods are introduced next.

## 2.1. Supervised Fine-Tuning

Supervisedfine-tuning (SFT) (Mangrulkar et al., 2022; Hu et al., 2022) is a direct application of supervised learning to LLMs. The objective of SFT is to minimize the negative log-likelihood (loglik) of response y given prompt x,

$$
\mathcal {L} _ {\mathrm{SFT}} (\theta) = - \mathbb {E} _ {x, y} \left[ \log \pi (y \mid x; \theta) \right],\tag{1}
$$

in expectation over prompt-response pairs $( x , y )$ sampled from a training set. One limitation of SFT is that we learn only from positive examples. Therefore, it is hard to learn not to generate certain y given x. This motivates learning of policies through rewards in Section 2.2.

## 2.2. Reinforcement Learning from Human Feedback

Reinforcement learningfrom humanfeedback (RLHF) has two stages: reward model learning and policy optimization. The reward model $r : \mathcal { X } \times \mathcal { Y } $ R is learned from human feedback (Ouyang et al., 2022). The LLM policy is then optimized to maximize the expected reward under the reward model using proximal policy optimization (PPO) (Schulman et al., 2017). The objective is

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{RLHF}} (\theta) \\ = \mathbb {E} _ {x, y \sim \pi (\cdot | x; \theta)} \left[ r (x, y) - \beta \log \frac {\pi (y \mid x ; \theta)}{\pi_ {0} (y \mid x)} \right], \end{array}\tag{2}
$$

where x is a prompt sampled from a training set. The first term is the reward of response y to prompt x. The second term penalizes for deviations of policy θ from a reference policy $\pi _ { 0 } .$ , usually obtained by SFT (Section 2.1). The regularization is needed because the reward model is usually learned from data collected by $\pi _ { 0 }$ and thus cannot estimate the value of significantly different policies well. The parameter $\beta \geq 0$ trades off the two terms. We define the optimal RLHF policy as $\theta _ { \mathrm { R L H F } } = \arg \operatorname* { m a x } _ { \theta \in \Theta } \mathcal { L } _ { \mathrm { R L H F } } ( \theta )$

## 2.3. Direct Preference Optimization

Direct preference optimization (DPO) (Rafailov et al., 2023) recasts RLHF as follows. Under the Bradley-Terry-Luce (BTL) model (Bradley and Terry, 1952; Luce, 2005) of human feedback, a response with reward $r ( x , y _ { 1 } )$ is preferred to that with reward $r ( x , y _ { 2 } )$ with probability

$$
p (y _ {1} \succ y _ {2} \mid x) = \mu (r (x, y _ {1}) - r (x, y _ {2}))  ,
$$

where $\mu ( v ) = 1 / ( 1 + \exp [ - v ] )$ is a sigmoidfunction. The key observation in DPO is that the policy that maximizes (2) has a closed form

$$
\pi (y \mid x; \theta_ {\mathrm{RLHF}}) = \frac {1}{Z (x)} \pi_ {0} (y \mid x) \exp \left[ \frac {1}{\beta} r (x, y) \right],
$$

where $Z ( x )$ is the normalizer (Rafailov et al., 2023). This holds for any prompt x and response y, under the assumption that the space of optimized policies can represent each conditional distribution exactly. This can be rearranged as $r ( x , y ) = \beta \log \frac { \pi ( y \mid x ; \theta _ { \mathrm { R L H F } } ) } { \pi _ { 0 } ( y \mid x ) } + \beta Z ( x )$ and thus

$$
\begin{array}{l} p (y _ {1} \succ y _ {2} \mid x; \theta) \\ = \mu \left(\beta \log \frac {\pi (y _ {1} \mid x ; \theta)}{\pi_ {0} (y _ {1} \mid x)} - \beta \log \frac {\pi (y _ {2} \mid x ; \theta)}{\pi_ {0} (y _ {2} \mid x)}\right) \end{array}\tag{3}
$$

holds when $\theta = \theta _ { \mathrm { R L H F } } . \mathrm { A }$ nice property of this substitution is that the normalizers $Z ( x )$ , which are difficult to estimate when the space of responses is infinite, cancel out.

Therefore, instead of learning a reward model and optimizing (2), we can directly optimize the policy in (3). Specifically, let $s \in \{ 0 , 1 \}$ be a random variable such that $s = 1$ when $y _ { 1 }$ is preferred to $y _ { 2 }$ given $x ,$ and $s = 0$ when $y _ { 2 }$ is preferred to $y _ { 1 }$ given x. This problem can be viewed as fitting (3) to the distribution of $s \mid x , y _ { 1 } , y _ { 2 }$ and written as maximizing the negative loglik

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{DPO}} (\theta) = - \mathbb {E} [ s \log p (y _ {1} \succ y _ {2} \mid x; \theta) + \\ (1 - s) \log p (y _ {2} \succ y _ {1} \mid x; \theta) ], \end{array}\tag{4}
$$

where the expectation is over prompt-candidate response pairs $( x , y _ { 1 } , y _ { 2 } )$ sampled from a training set, and stochastic preferential feedback $s \mid x , y _ { 1 } , y _ { 2 }$ . We define the optimal DPO policy as

$$
\theta_ {*} = \arg \min _ {\theta \in \Theta} \mathcal {L} _ {\mathrm{DPO}} (\theta)\tag{5}
$$

and note that it is the maximum likelihood estimate $( M L E )$ for (4). Note that (4) is equivalent to a more classic

$$
\mathcal {L} _ {\mathrm{DPO}} (\theta) = - \mathbb {E} \left[ \log p (y _ {w} \succ y _ {l} \mid x; \theta) \right]
$$

when the winning response is $y _ { w } = s y _ { 1 } + ( 1 - s ) y _ { 2 }$ and the losing response is $y _ { l } = ( 1 - s ) y _ { 1 } + s y _ { 2 }$ . We use the reparameterized objective in (4) because it clearly separates the random variable s from the rest of the objective.

We also note that (3) can be rewritten as

$$
\begin{array}{l} p (y _ {1} \succ y _ {2} \mid x; \theta) \\ = \mu \left(\beta \log \frac {\pi (y _ {1} \mid x ; \theta)}{\pi (y _ {2} \mid x ; \theta)} - \beta \frac {\pi_ {0} (y _ {1} \mid x)}{\pi_ {0} (y _ {2} \mid x)}\right), \end{array}
$$

where log $\frac { \pi _ { 0 } ( y _ { 1 } | x ) } { \pi _ { 0 } ( y _ { 2 } | x ) }$ depends on the reference policy $\pi _ { 0 }$ but not on the optimized policy θ. We use this algebraic form because it separates the optimized part of the objective from essentially constants.

## 3. Setting

We study active learning in DPO (Section 2.3). Simply put, instead of assuming that (4) is approximated using a fixed dataset, we choose the dataset actively with the objective of learning policies that are close to $\theta _ { * }$ . We study two variants of this problem, offline and online, which we present next.

Offline feedback. The input to this setting is a dataset of size $N$ with preferential human feedback for all data points. The dataset is $\mathcal { D } = \{ ( x _ { i } , y _ { i , 1 } , y _ { i , 2 } , s _ { i } ) \} _ { i = 1 } ^ { N }$ , where $x _ { i }$ is the prompt in data point $i \in [ N ] , y _ { i , 1 }$ and $y _ { i , 2 }$ are the candidate responses, and $s _ { i }$ is the preferential feedback. Specifically, $s _ { i } = 1$ if the preferred response is $y _ { i , 1 }$ , and $s _ { i } = 0$ if the preferred response is $y _ { i , 2 }$ . Our goal is to select a subset of $\mathcal { D }$ of size n so that the DPO policy on this subset is “close” to $\theta _ { * }$ . This setting is motivated by computational efficiency. In particular, even if preferential feedback $s _ { i }$ is known, we may not have computational resources to learn from all of it. Choosing the most informative subset of D of size n is a natural way of maximizing the information gain within the computational cost constraint.

Online feedback. The input to this setting is a dataset of size N without preferential human feedback. The dataset is $\mathcal { D } = \{ ( x _ { i } , y _ { i , 1 } , y _ { i , 2 } ) \} _ { i = 1 } ^ { N }$ , where $x _ { i }$ is the prompt in data point $i \in [ N ]$ , and $y _ { i , 1 }$ and $y _ { i , 2 }$ are the candidate responses. The human feedback $s _ { i }$ is elicited online. This setting is motivated by statistical efficiency. We want to collect the most informative feedback using only information about prompts $x _ { i } .$ , and candidate responses $y _ { i , 1 }$ and $y _ { i , 2 } .$

Let $S _ { n } \subseteq [ N ]$ be a subset of n data point indices from $\mathcal { D } ,$ either collected online or offline. After the algorithm selects ${ \cal { S } } _ { n } ,$ , we minimize an empirical approximation to (4) on $s _ { n } .$ Before we define it, we introduce a more compact notation. Let

$$
\mu_ {i} (\theta) = \mu \left(\beta \log \frac {\pi (y _ {i , 1} \mid x _ {i} ; \theta)}{\pi (y _ {i , 2} \mid x _ {i} ; \theta)} - \beta b _ {i}\right)
$$

be the probability that response y<sub>i,1</sub> is preferred to $y _ { i , 2 }$ given $x _ { i }$ under policy θ, where $\begin{array} { r } { b _ { i } = \log \left( \frac { \pi _ { 0 } \left( y _ { i , 1 } | x _ { i } \right) } { \pi _ { 0 } \left( y _ { i , 2 } | x _ { i } \right) } \right) } \end{array}$ is the bias due to the reference policy $\pi _ { 0 } .$ . Let

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DPO}} (\theta ; \mathcal {S}) \\ = - \sum_ {i \in \mathcal {S}} s _ {i} \log \mu_ {i} (\theta) + (1 - s _ {i}) \log (1 - \mu_ {i} (\theta)) \end{array}\tag{6}
$$

be the DPO negative loglik on $S \subseteq [ N ]$ . Then (4) can be approximated on $S _ { n }  { \mathrm { \ b y } } \frac { 1 } { n }  { \mathcal { L } } _ { \mathrm { D P O } } ( \theta ;  { S } _ { n } )$ . We propose algorithms for choosing $S _ { n }$ in Section 4.

Objective. Now we are ready to state our objective. Let $\theta _ { * }$ be the optimal DPO policy in (5). Let $\mathcal { E } ( \theta , \theta _ { * } ) =$

$$
\max _ {i \in [ N ]} \left| \beta \log \frac {\pi (y _ {i , 1} \mid x _ {i} ; \theta)}{\pi (y _ {i , 2} \mid x _ {i} ; \theta)} - \beta \log \frac {\pi (y _ {i , 1} \mid x _ {i} ; \theta_ {*})}{\pi (y _ {i , 2} \mid x _ {i} ; \theta_ {*})} \right|\tag{7}
$$

be the maximum logit error under policy θ, the difference of DPO logits under θ and $\theta _ { * }$ . Note that the biases cancel. Let $\begin{array} { r } { \hat { \theta } _ { n } = \arg \operatorname* { m i n } _ { \theta \in \Theta } \mathcal { L } _ { \mathrm { D P O } } ( \theta ; \mathcal { S } _ { n } ) } \end{array}$ denote the optimal DPO policy on $S _ { n }$ . We want $\ddot { \theta } _ { n }$ to be close to $\theta _ { * }$ in terms of (7). Specifically, we want $\mathcal { E } ( \hat { \theta } _ { n } , \theta _ { * } )$ to decrease with n with a high probability. The motivation for (7) is that it can bound many other errors. For instance, since the Lipschitz factor of $\mu$ is $1 / 4$ , we get

$$
\max _ {i \in [ N ]} | \mu_ {i} (\hat {\theta} _ {n}) - \mu_ {i} (\theta_ {*}) | \leq \frac {1}{4} \mathcal {E} (\hat {\theta} _ {n}, \theta_ {*}).
$$

Therefore, when the maximum logit error is small, the estimated probability that $y _  i , $ <sub>1</sub> is preferred to $y _ { i , 2 }$ under policy $\widehat { \theta } _ { n } .$ , for any data point $i \in [ N ]$ , is close to that under $\theta _ { * }$ .

## 4. Algorithms

The key idea in our paper is to linearize the policy at the last layer of its neural network representation and use linear algebra for active learning. Active learning on linearized neural networks was popularized in regret minimization by Riquelme et al. (2018). Das et al. (2024); Mukherjee et al. (2024); Thekumparampil et al. (2024); Liu et al. (2024); Scheid et al. (2024) applied it recently to learning reward models. In our work, we linearize policies and formalize it as follows.

Assumption 1. All policies are log-linear,

$$
\pi (y \mid x; \theta) \propto \exp [ \phi (x, y) ^ {\top} \theta ],\tag{8}
$$

where $\phi ( x , y ) \in \mathbb { R } ^ { d }$ is the feature vectorfor pair $( x , y )$ and $\theta \in \mathbb { R } ^ { d }$ is a policy parameter.

We make this assumption for the rest of the paper. Under this assumption, $\mu _ { i } ( \boldsymbol { \theta } )$ in (6) becomes

$$
\mu_ {i} (\theta) = \mu (\beta (\phi_ {i} ^ {\top} \theta - b _ {i}))  ,\tag{9}
$$

where $\phi _ { i } = \phi ( x _ { i } , y _ { i , 1 } ) - \phi ( x _ { i } , y _ { i , 2 } )$ is the difference of the feature vectors of responses $y _ { i , 1 }$ and $y _ { i , 2 }$ given $x _ { i }$ . We note that the normalizers of $\pi ( \boldsymbol { y } \mid \boldsymbol { x } ; \boldsymbol { \theta } )$ cancel out. We also note that when (9) is substituted into (6), we obtain a similar expression to the negative loglik of logistic regression, except for the bias $b _ { i }$ and $\beta .$ The key idea in our algorithms is to optimize the Hessian of the DPO negative loglik.

Lemma 1. Let $\pi ( \boldsymbol { y } \mid \boldsymbol { x } ; \boldsymbol { \theta } )$ be a log-linear policy. Then the Hessian of $\mathcal { L } _ { \mathrm { D P O } } ( \theta ; S )$ in (6) with respect to θ is

$$
\nabla^ {2} \mathcal {L} _ {\mathrm{DPO}} (\theta ; \mathcal {S}) = \beta^ {2} \sum_ {i \in \mathcal {S}} \mu_ {i} (\theta) (1 - \mu_ {i} (\theta)) \phi_ {i} \phi_ {i} ^ {\top}.
$$

It is also positive semi-definite.

Proof. The proof is in Appendix A.1.

The Hessian $\nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta ; S )$ can be used to derive the covariance matrix of the MLE of ${ \mathcal { L } } _ { \mathrm { D P O } } ( \theta ; S )$ and is also known as the Fisher information matrix (Fisher, 1922). Therefore, it can be used for both uncertainty quantification and information gathering (Lattimore and Szepesvari, 2019). Since the MLE of ${ \mathcal { L } } _ { \mathrm { D P O } } ( \theta ; S )$ is a policy, we can use the Hessian to select a subset of data points to learn better policies.

Specifically, let $S _ { n }$ be a subset of n data point indices and $\hat { \theta } _ { n } =$ arg min $\theta \in \Theta  ^ { \angle _ { \mathrm { D P O } } ( \theta ; S _ { n } ) }$ be the corresponding MLE. We show in Theorem 2 that the error in the logit estimate at data point $i \in [ N ]$ is bounded with a high probability as

$$
| \phi_ {i} ^ {\top} (\hat {\theta} _ {n} - \theta_ {*}) | \leq \sqrt {d \phi_ {i} ^ {\top} (\nabla^ {2} \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*} ; \mathcal {S} _ {n})) ^ {- 1} \phi_ {i}}
$$

up to logarithmic factors. To minimize it, we want to maximize all eigenvalues of $\nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; \mathcal { S } _ { n } )$ . We achieve this by maximizing log $\operatorname* { d e t } ( \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; S _ { n } ) )$ over $S _ { n }$

This optimization problem is challenging for two reasons. First, it is a discrete optimization problem over $S _ { n }$ . In our work, we maximize log det $( \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; S _ { n } ) )$ ) greedily. An informal justification for this approach is that log det(X) is monotone and concave in X for $X \succeq 0$ , and thus a greedy algorithm should be near optimal (Nemhauser et al., 1978). We prove this formally in Section 5. Second, $\theta _ { * }$ is unknown. We overcome this by using its plug-in estimates (Stufken and Yang, 2012)

## 4.1. Active DPO with Online Preferential Feedback

Our first algorithm does not have access to any preferential feedback initially. It collects it online, re-estimates $\theta _ { * }$ , and approximately maximizes log det $( \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; S _ { n } ) )$

The pseudo-code of the algorithm is in Algorithm 1 and we call it active DPO (ADPO). ADPO chooses data points in n rounds. The indices of the chosen data points in the first t rounds are denoted by $S _ { t }$ and the corresponding Hessian is $H _ { t }$ . We refer to it as the design matrix since it is used to select next data points. The design matrix is initialized to $\gamma I _ { d } ,$ where $\gamma > 0$ is a constant that guarantees that all $H _ { t }$ are well defined. In round t, ADPO selects the index $I _ { t }$ that greedily maximizes the information gain given $H _ { t }$ and the empirical estimate of $\theta _ { * }$ up to round $t , \widehat { \theta } _ { t - 1 }$ (line 6). This is because

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 ADPO: Active DPO with online feedback.

1: Input: Dataset  $\mathcal{D}=\{(x_{i},y_{i,1},y_{i,2})\}_{i=1}^{N}$ 

2:  $H_{0}\leftarrow\gamma I_{d},S_{0}\leftarrow\emptyset$ 

3: for  $t=1,\ldots,n$  do

4: Solve  $\hat{\theta}_{t-1}\leftarrow\arg\min_{\theta\in\Theta}\mathcal{L}_{\mathrm{DPO}}(\theta;\mathcal{S}_{t-1})$ 

5: Let  $v_{t,i}\leftarrow\beta\sqrt{\mu_{i}(\hat{\theta}_{t-1})(1-\mu_{i}(\hat{\theta}_{t-1}))}\phi_{i}$ 

6:  $I_{t}\leftarrow\arg\max_{i\in[N]\setminus S_{t-1}}\log\det(H_{t-1}+v_{t,i}v_{t,i}^{\top})$ 

7: Get preferential feedback  $s_{I_{t}}$  on  $(x_{I_{t}},y_{I_{t},1},y_{I_{t},2})$ 

8:  $H_{t}\leftarrow H_{t-1}+v_{t,I_{t}}v_{t,I_{t}}$ 

9:  $S_{t}\leftarrow S_{t-1}+\{I_{t}\}$ 

10: Output: Data point indices  $S_{n}$  for learning a model

Algorithm 2 ADPO+: Active DPO for offline feedback.

1: Input: Dataset  $\mathcal{D}=\{(x_{i},y_{i,1},y_{i,2},s_{i})\}_{i=1}^{N}$ 

2:  $H_{0}\leftarrow\gamma I_{d},S_{0}\leftarrow\emptyset$ 

3: Solve  $\hat{\theta}\leftarrow\arg\min_{\theta\in\Theta}\mathcal{L}_{\mathrm{DPO}}(\theta;[N])$ 

4: for  $t=1,\ldots,n$  do

5:  $\hat{\theta}_{t-1}\leftarrow\hat{\theta}$ 

6: Let  $v_{t,i}\leftarrow\beta\sqrt{\mu_{i}(\hat{\theta}_{t-1})(1-\mu_{i}(\hat{\theta}_{t-1}))}\phi_{i}$ 

7:  $I_{t}\leftarrow\arg\max_{i\in[N]\setminus S_{t-1}}\log\det(H_{t-1}+v_{t,i}v_{t,i}^{\top})$ 

8:  $H_{t}\leftarrow H_{t-1}+v_{t,I_{t}}v_{t,I_{t}}$ 

9:  $S_{t}\leftarrow S_{t-1}+\{I_{t}\}$ 

10: Output: Data point indices  $S_{n}$  for learning a model
</div>

$$
v _ {t, i} v _ {t, i} ^ {\top} = \beta^ {2} \mu_ {i} (\hat {\theta} _ {t - 1}) (1 - \mu_ {i} (\hat {\theta} _ {t - 1})) \phi_ {i} \phi_ {t} ^ {\top}
$$

can be viewed as the incremental gain due to data point i in Lemma 1. After the data point $I _ { t }$ is chosen, we observe preferential feedback on it (line 7) and update all statistics (lines 8-9). Finally, after n rounds, ADPO outputs n chosen indices (line 10) and an LLM policy is optimized on them using DPO.

The time complexity of ADPO is $O ( n ^ { 2 } + n N )$ . The former term is due to training on all past feedback in each round (line 4) and the latter is due to maximizing exactly in line 6. In experiments, we reduce the former to $O ( n \log n )$ by estimating $\hat { \theta } _ { t - 1 }$ only a logarithmic number of times, when $t = 2 ^ { i }$ for some integer $i > 0$ . We reduce the latter to $O ( n )$ by replacing $[ N ] \setminus S _ { t - 1 }$ with its random subset of a fixed size 256. Finally, note that $I _ { t }$ in line 6 can be equivalently expressed (Appendix A.3) as

$$
I _ {t} = \arg \max _ {i \in [ N ] \backslash \mathcal {S} _ {t - 1}} v _ {t, i} H _ {t - 1} ^ {- 1} v _ {t, i} ^ {\top}.\tag{10}
$$

Therefore, the determinant does not need to be computed. The inverse $H _ { t - 1 } ^ { - 1 }$ can be computed incrementally using the Sherman-Morrison formula, with $O ( d ^ { 2 } )$ update time. The statistical efficiency of ADPO is analyzed in Section 5.

## 4.2. Active DPO with Offline Preferential Feedback

Our second algorithm has access to preferential feedback initially. All feedback is used to estimate $\theta _ { * }$ , which is then used to approximately maximize log det $( \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; S _ { n } ) )$

The pseudo-code of our algorithm is in Algorithm 2 and we call it $\tt A D P O ^ { + }$ , where + indicates that $\mathtt { A D P 0 ^ { + } }$ has access to more information than ADPO. $\mathtt { A D P 0 ^ { + } }$ differs from ADPO in two steps. First, $\theta _ { * }$ is estimated initially (line 3) from all preferential feedback. Second, no preferential feedback is collected online. Similarly to ADPO, the time complexity of $\mathtt { A D P 0 ^ { + } }$ is $O ( n N )$ because of the exact maximization in line 7. We reduce it to $O ( n )$ in experiments as in Section 4.1.

## 5. Analysis

In this section, we provide a unified analysis for ADPO and $\mathtt { A D P 0 ^ { + } }$ . This is possible because the algorithms only differ in how the instance-specific factors in the design matrix are estimated. In $\tt A D P O ^ { + }$ , they are estimated from all preferential feedback. In ADPO, only the online elicited feedback up to round t is used. We state our assumptions first.

We assume that all policies are log-linear (Assumption 1) and that the collected feedback $s _ { I _ { t } }$ is conditionally independent given all feedback up to round t, for all $t \in [ n ]$ . Under this assumption, the negative loglik in (6) is similar to that of logistic regression and we can use existing concentration inequalities (Abbasi-Yadkori et al., 2011).

Assumption 2. [Boundedness] For any $i \in [ N ] , \| \phi _ { i } \| _ { 2 } \leq 1$ and $| b _ { i } | \leq 1$ . We assume that Θ is a unit sphere, and hence $\| \theta _ { * } \| _ { 2 } \leq 1 a n d \| \hat { \theta } _ { n } \| _ { 2 } \leq 1 .$

Assumptions on feature vectors, comprising $\phi _ { i }$ and $b _ { i } ,$ are standard in the analyses of generalized linear models (Li et al., 2017; Kveton et al., 2020; Mukherjee et al., 2024). Our assumption on $\theta _ { * }$ and $\ddot { \theta } _ { n }$ can be guaranteed by applying DPO to a unit sphere Θ. The assumption can be weakened to $\lVert \hat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } _ { * } \rVert _ { 2 } \leq 1$ using initial exploration (Li et al., 2017; Kveton et al., 2020).

We can analyze ADPO and $\mathtt { A D P 0 ^ { + } }$ in a unified way because the instance-specific factors in their design matrices can be bounded from below by $c _ { \mathrm { m i n } }$ and above by $c _ { \mathrm { m a x } }$

Assumption 3. [Design matrix] For any $i \in [ N ]$ and $\theta \in \Theta$ we have $0 \leq c _ { \operatorname* { m i n } } \leq \beta ^ { 2 } \mu _ { i } ( \theta ) ( 1 - \mu _ { i } ( \theta ) ) \leq c _ { \operatorname* { m a x } } .$

These constants obviously exist and can be easily derived. For instance, since ma $\mathrm { x } _ { x \in \mathbb { R } } \mu ( x ) ( 1 - \mu ( x ) ) = 0 . 2 5$ , we get $c _ { \operatorname* { m a x } } = 0 . 2 5 \beta ^ { 2 }$ . Moreover, under Assumption 2, we have for any $\mu _ { i } ( \theta ) \leq 0 . 5$ that

$$
\beta^ {2} \mu_ {i} (\theta) (1 - \mu_ {i} (\theta)) \geq \beta^ {2} \mu_ {i} ^ {2} (\theta) \geq \beta^ {2} \mu (- 4 \beta) = c _ {\mathrm{min}}.
$$

The argument for $\mu _ { i } ( \theta ) \geq 0 . 5$ is similar. The constants $c _ { \mathrm { m i n } }$ and $c _ { \mathrm { m a x } }$ appear in our bounds.

The last assumption is that the dataset is sufficiently diverse. Assumption 4. [Diverse dataset] There exists a constant $\kappa \geq 1$ such that $v _ { t , i } ^ { \top } H _ { t - 1 } ^ { - 1 } v _ { t , i } \leq \kappa v _ { t , I _ { t } } ^ { \top } H _ { t - 1 } ^ { - 1 } v _ { t , I _ { t } }$ holds for any $i \in [ N ]$ and $t \in [ n ]$

This assumption says that the maximizer in (10) is an approximate upper bound, up to a multiplicative $\kappa \geq 1$ , on the information gain at each data point, including those previously chosen that cannot be chosen again. We note that the assumption holds for $\kappa = 1$ when repeated independent observations of the data points are allowed, as in all prior works (Appendix C). In this case, the maximization in (10) would be over $i \in [ N ]$

## 5.1. Main Result

We state our main claim below.

Theorem 2. Let $\hat { \theta } _ { n } = \mathrm { a r g }$ min $\theta \in \Theta  ^ { \angle _ { \mathrm { D P O } } ( \theta ; S _ { n } ) }$ . Then the maximum logit error under ADPO and ADPO<sup>+</sup> is

$$
\mathcal {E} (\hat {\theta} _ {n}, \theta_ {*}) = \tilde {O} (d \sqrt {\log (1 / \delta) / n})
$$

with probability at least $1 - \delta ,$ where $\tilde { O }$ hides all logarithmic factors but those in δ.

We prove the claim as follows. For log-linear policies, (7) reduces to max $\mathsf { \Omega } _ { i \in [ N ] } \left| \phi _ { i } ^ { \top } ( \hat { \theta } _ { n } { - } \theta _ { * } ) \right.$ |. By the Cauchy-Schwarz inequality, for any data point $i \in [ N ]$

$$
| \phi_ {i} ^ {\top} (\hat {\theta} _ {n} - \theta_ {*}) | \leq \| \phi_ {i} \| _ {\Sigma_ {n} ^ {- 1}} \| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}},\tag{11}
$$

where $\Sigma _ { n } = \gamma I _ { d } + \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; \mathcal { S } _ { n } )$ a regularized Hessian at the optimal DPO policy $\theta _ { * }$ . To bound the first term, we note that the feedback at data point i is distributed as

$$
s _ {i} \sim \mu_ {i} (\theta_ {*}) = \mu (\beta (\phi_ {i} ^ {\top} \theta_ {*} - b _ {i})).\tag{12}
$$

This assumption follows from the definition of DPO in (3), which says that $\mu _ { i } (  { \boldsymbol { \theta } } _ { * } )$ is the probability that response $y _ { i , 1 }$ is preferred to $y _ { i , 2 }$ given $x _ { i }$ . Thus we can build on existing concentration results for sub-Gaussian random variables to prove the following.

Theorem 3. For any set of n indices $S _ { n } \subseteq [ N ]$

$$
\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} \leq \sqrt {\frac {\beta^ {2} d}{c _ {\min}} \log \left(\frac {1 + c _ {\min} n / \gamma}{\delta}\right)} + 2 \gamma^ {\frac {1}{2}}
$$

holds with probability at least $1 - \delta$

To bound the second term in (11), we use the fact that the standard errors of the logit estimates do not increase over time and decrease at a desired rate if Assumption 4 holds for some constant $\kappa \geq 1$

Theorem 4. For any data point $i \in [ N ]$

$$
\phi_ {i} ^ {\top} \Sigma_ {n} ^ {- 1} \phi_ {i} \leq \frac {c _ {\mathrm{max}} ^ {3} \log \left(1 + \frac {c _ {\mathrm{max}} n}{\gamma d}\right)}{c _ {\mathrm{min}} \gamma \log (1 + c _ {\mathrm{max}} / \gamma)} \frac {\kappa d}{n}.
$$

All proofs are in Appendix A.

## 5.2. Discussion

The bound in Theorem 2 is $\tilde { O } ( d \sqrt { \log ( 1 / \delta ) / n } )$ and holds with probability at least $1 - \delta$ . As a result, the maximum logit error decreases with more feedback n and increases with the number of learned policy parameters d. The bound is not directly comparable to prior works in Appendix C because they bound reward model errors, while we bound a policy learning error. That being said, the dependence on n and δ is similar. The linear dependence on d arises because Theorem 4 is proved through a self-normalizing bound in Theorem 3 that would apply even to infinitely-large datasets. We would get an $\tilde { O } ( \sqrt { d \log ( N ) \log ( 1 / \delta ) / n } )$ bound, where $N$ is the dataset size, if we followed the analysis of Kveton et al. (2020) and applied a union bound over all data points.

## 6. Experiments

We experiment with both log-linear (Section 6.1) and LLM (Section 6.2) policies. The log-linear experiments validate that ADPO and ADPO<sup>+</sup> work as analyzed. The LLM experiments show that ADPO and ADPO<sup>+</sup> perform well in practice when applied to LLMs. We conduct more experiments with log-linear policies in Appendix B.

## 6.1. Log-Linear Policies

This experiment is designed as follows. First, we take an existing multi-class classification dataset and turn it into a preferential feedback dataset. More specifically, we choose a random positive label and generate N vectors $\left\{ \phi _ { i } \right\} _ { i = 1 } ^ { N } ,$ where $\phi _ { i } \in \mathbb { R } ^ { d }$ is the difference of feature vectors of random positive and negative examples. Second, we label all $\phi _ { i }$ with 1 and learn a logistic regression model to simulate preferential feedback. Let <sup>¯</sup>θ and Σ<sup>¯</sup> be the learned model parameter and its covariance, respectively. Third, we generate preferential feedback $s _ { i } \sim \mathrm { B e r } ( \mu ( \phi _ { i } ^ { \top } \bar { \theta } ) )$ for all $\phi _ { i }$ and get a dataset $\mathcal { D } = \{ ( \phi _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N }$ . Fourth, we generate a reference policy as $\theta _ { 0 } \sim \mathcal { N } ( \bar { \theta } , \bar { \Sigma } )$ and set the bias as $b _ { i } = \phi _ { i } ^ { \top } \theta _ { 0 }$ Simply put, $\theta _ { 0 }$ is close ${ \bar { \theta } } ,$ as measured by the uncertainty of ${ \bar { \theta } } .$ Finally, we compute the optimal DPO policy $\theta _ { * }$ on $\mathcal { D }$ All compared methods apply DPO to their selected subset $S _ { n }$ of D and learn $\begin{array} { r } { \hat { \theta } _ { n } = \arg \operatorname* { m i n } _ { \theta \in \Theta } \mathcal { L } _ { \mathrm { D P O } } ( \theta ; \mathcal { S } _ { n } ) } \end{array}$

Figure 1. Experiments with log-linear policies on the CIFAR-10 (first row) and CIFAR-100 (second row) datasets.

We compare $\hat { \theta } _ { n }$ to $\theta _ { * }$ in three metrics. The first metric is the maximum logit error, max $\smash { i \in [ N ] } \left| \phi _ { i } ^ { \top } ( \hat { \boldsymbol { \theta } } _ { n } - \boldsymbol { \theta } _ { * } ) \right|$ , which we bound in Theorem $2 .$ The second metric is the mean logit error $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | \boldsymbol { \phi } _ { i } ^ { \top } ( \hat { \boldsymbol { \theta } } _ { n } - \boldsymbol { \theta } _ { * } ) } \end{array}$ |. Although we do not analyze $\mathrm { i t , }$ our methods minimize it indirectly through the maximum error. The last metric is the error rate,

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \left\{\mathrm{sgn} (\phi_ {i} ^ {\top} \hat {\theta} _ {n} - b _ {i}) \neq \mathrm{sgn} (\phi_ {i} ^ {\top} \theta_ {*} - b _ {i}) \right\},
$$

which is the fraction of incorrectly ordered responses by $\hat { \theta } _ { n }$ when θ<sub>∗</sub> is the ground truth.

We compare five algorithms. The first two algorithms are ADPO and $\mathtt { A D P 0 ^ { + } }$ . We expect $\mathtt { A D P 0 ^ { + } }$ to perform better because it has access to more information. We consider three baselines: Uniform, APO, and PMC. Uniform selects data points uniformly at random. While simple, it is known to be competitive in real-world problems where feature vectors may cover the feature space close to uniformly (Ash et al., 2020; 2021; Mukherjee et al., 2024; Muldrew et al., 2024). APO is the practical incremental D-optimal design for linear models proposed in Das et al. (2024). The main difference from ADPO is that APO neglects logistic model factors and $\beta$ (Lemma 1). Therefore, while it selects diverse $\phi _ { i }$ , they do not necessarily maximize the information gain in DPO. The last baseline is PMC of Muldrew et al. (2024), which selects data points with the highest differences between estimated rewards of their responses.

We experiment with CIFAR-10 and CIFAR-100 datasets (Krizhevsky, 2009). The features are a random subset of ResNet-50 embeddings (He et al., 2016) of size $d = 3 8 4$ The dataset size is $N = 2 ^ { 1 6 }$ . We set the DPO regularizer to $\beta = 1$ and experiment with other $\beta$ in Appendix B. Our CIFAR-10 results are reported in the first row of Figure 1. $\mathtt { A D P 0 ^ { + } }$ is the best performing method in all metrics. Many improvements are major. For instance, the lowest maximum logit error of Uniform $( n = 2 ^ { 1 5 } )$ is attained by $\mathtt { A D P 0 ^ { + } }$ at $n < 2 ^ { 1 3 }$ . The lowest maximum logit error of APO $( n = 2 ^ { 1 5 } )$ is attained by $\mathtt { A D P 0 ^ { + } }$ at $n < 2 ^ { 1 4 }$ . ADPO is the second best method in the maximum logit error. It is never worse than Uniform, APO, and PMC. ADPO improves in all metrics over all baselines at larger sample sizes. Our CIFAR-100 results are reported in the second row of Figure 1 and we observe the same trends as on the CIFAR-10 dataset.

## 6.2. LLM Policies

We also experiment with a real-world preference dataset Nectar (Zhu et al., 2023) and two LLM policies: Llama-3.2 (3B parameters) (Dubey et al., 2024) and Phi-3 (Abdin et al., 2024). We sample $N = 5 0 0 0$ prompts $\{ x _ { i } \} _ { i = 1 } ^ { N }$ from the dataset, each with two responses. The accepted $\left\{ y _ { i , w } \right\} _ { i = 1 } ^ { N }$ and rejected $\left\{ y _ { i , l } \right\} _ { i = \cdot } ^ { N }$ responses are determined based on the ground truth in the dataset. The feature vector $\phi ( x , y )$ is the embedding of the concatenated prompt and response from the last hidden layer of the LLM, of size $d = 4 0 9 6$ The bias term is $b _ { i } = \log \pi _ { 0 } ( y _ { i , w } \mid x _ { i } ) - \log \pi _ { 0 } ( y _ { i , l } \mid x _ { i } )$

Figure 2. Experiments with LLM policies on the Nectar dataset. We use Llama-3.2 (first row) and Phi-3 (second row) models.

where $\pi _ { 0 }$ is the initial LLM reference policy.

We report three metrics. The accuracy measures how wel we distinguish between positive and negative responses,

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \left\{\log \frac {\pi (y _ {i , w} \mid x _ {i} ; \theta)}{\pi_ {0} (y _ {i , w} \mid x _ {i})} > \log \frac {\pi (y _ {i , l} \mid x _ {i} ; \theta)}{\pi_ {0} (y _ {i , l} \mid x _ {i})} \right\}.
$$

This metric is 1 minus the error rate in Figure 1 and thus identical, up to how we plot it. We could not plot the two other metrics in Figure 1 because they require knowing $\theta _ { * }$ Therefore, we decided to plot two other metrics that reflect the confidence in distinguishing the responses. The margin is the advantage of a positive response over a negative one,

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \beta \log \frac {\pi (y _ {i , w} \mid x _ {i} ; \theta)}{\pi_ {0} (y _ {i , w} \mid x _ {i})} - \beta \log \frac {\pi (y _ {i , l} \mid x _ {i} ; \theta)}{\pi_ {0} (y _ {i , l} \mid x _ {i})}.
$$

The negative loglik is the logistic regression loss,

$$
- \frac {1}{N} \sum_ {i = 1} ^ {N} \log \mu \left(\beta \log \frac {\pi (y _ {i , w} \mid x _ {i} ; \theta)}{\pi_ {0} (y _ {i , w} \mid x _ {i})} - \beta \log \frac {\pi (y _ {i , l} \mid x _ {i} ; \theta)}{\pi_ {0} (y _ {i , l} \mid x _ {i})}\right)
$$

Our results with Llama-3.2 and Phi-3 models are reported in Figure 2. We observe similar trends to Figure 1. ADPO<sup>+</sup> is clearly the best performing method in both the margin and negative loglik. ADPO is among the best three methods for larger sample sizes. The least clear trend is in accuracy. We believe that this is because many responses are of a similar quality. Therefore, they cannot be easily distinguished and lie close to the decision boundary, which can be impacted by even minor changes in the LLM.

## 7. Conclusions

We propose an active learning framework for DPO. The key idea is to linearize the DPO objective at the last layer of the neural network representation of the optimized policy and then compute the D-optimal design to collect preferential feedback. We propose two algorithms. One is for the online setting, where the human feedback is elicited online, and the other is for the offline setting, where the feedback has already been collected and we choose its subset to improve the computation efficiency of DPO. We analyze both algo rithms and also evaluate them empirically, in the setting that matches our theory and on LLMs.

This is the first work that applies optimal designs to DPO. The main difference from prior works is that the optimal design is applied to policy optimization. A natural direction .for future work are other policy optimization frameworks, such as KTO (Ethayarajh et al., 2024). Our analysis could also be improved in several aspects. For instance, it is for log-linear policies and we have not derived an upper bound on κ in Assumption 4. In the setting of prior works, where multiple independent observations of preferential feedback for the same prompt are possible, κ = 1.

## References

Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems 24, pages 2312–2320, 2011.

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.

Jordan Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. In Proceedings of the 8th International Conference on Learning Representations, 2020.

Jordan Ash, Surbhi Goel, Akshay Krishnamurthy, and Sham Kakade. Gone fishing: Neural active learning with Fisher embeddings. In Advances in Neural Information Processing Systems 34, 2021.

Jean-Yves Audibert, Sebastien Bubeck, and Remi Munos. Best arm identification in multi-armed bandits. In Proceedings of the 23rd Annual Conference on Learning Theory, pages 41–53, 2010.

Mohammad Javad Azizi, Branislav Kveton, and Mohammad Ghavamzadeh. Fixed-budget best-arm identification in structured bandits. In Proceedings ofthe 31st International Joint Conference on Artificial Intelligence, 2022.

Markus Bayer and Christian Reuter. Activellm: Large language model-based active learning for textual few-shot scenarios. arXiv preprint arXiv:2405.10808, 2024.

Christopher Bishop. Pattern Recognition and Machine Learning. Springer, New York, NY, 2006.

Ralph Allan Bradley and Milton Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3-4):324–345, 1952.

Sebastien Bubeck, Remi Munos, and Gilles Stoltz. Pure exploration in multi-armed bandits problems. In Proceedings of the 20th International Conference on Algorithmic Learning Theory, pages 23–37, 2009.

Yifang Chen, Shuohang Wang, Ziyi Yang, Hiteshi Sharma, Nikos Karampatziakis, Donghan Yu, Kevin Jamieson, Simon Shaolei Du, and Yelong Shen. Cost-effective proxy reward model construction with on-policy and active learning. arXiv preprint arXiv:2407.02119, 2024.

Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems 30, 2017.

Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano, and Sayak Ray Chowdhury. Active preference optimization for sample efficient RLHF. CoRR, abs/2402.10500, 2024. URL https://arxiv.org/abs/2402.10500.

Paul Doucet, Benjamin Estermann, Till Aczel, and Roger Wattenhofer. Bridging diversity and uncertainty in active learning with self-supervised pre-training. arXiv preprint arXiv:2403.03728, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In Proceedings ofthe 41th International Conference on Machine Learning, 2024.

Ronald Fisher. On the mathematical foundations of theoretical statistics. Philosophical Transactions of the Royal Society ofLondon: Series A, 222:309–368, 1922.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792, 2024.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2016.

Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In Proceedings ofthe 10th International Conference on Learning Representations, 2022.

Kaixuan Ji, Jiafan He, and Quanquan Gu. Reinforcement learning from human feedback with active queries. arXiv preprint arXiv:2402.09401, 2024.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Branislav Kveton, Csaba Szepesvari, Zheng Wen, and Azin Ashkan. Cascading bandits: Learning to rank in the cascade model. In Proceedings of the 32nd International Conference on Machine Learning, 2015.

Branislav Kveton, Manzil Zaheer, Csaba Szepesvari, Lihong Li, Mohammad Ghavamzadeh, and Craig Boutilier. Randomized exploration in generalized linear bandits.

In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, 2020.

Paul Lagree, Claire Vernade, and Olivier Cappe. Multipleplay bandits in the position-based model. In Advances in Neural Information Processing Systems 29, pages 1597– 1605, 2016.

Tor Lattimore and Csaba Szepesvari. Bandit Algorithms. Cambridge University Press, 2019.

Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear contextual bandits. In Proceedings ofthe 34th International Conference on Machine Learning, pages 2071–2080, 2017.

Shuai Li, Baoxiang Wang, Shengyu Zhang, and Wei Chen. Contextual combinatorial cascading bandits. In Proceedings of the 33rd International Conference on Machine Learning, pages 1245–1253, 2016.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In Proceedings ofthe 12th International Conference on Learning Representations, 2024.

Pangpang Liu, Chengchun Shi, and Will Wei Sun. Dual active learning for reinforcement learning from human feedback. CoRR, abs/2410.02504, 2024. URL https: //arxiv.org/abs/2410.02504.

Robert Duncan Luce. Individual Choice Behavior: A Theoretical Analysis. Dover Publications, 2005.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. https://github.com/huggingface/ peft, 2022.

Katerina Margatina, Timo Schick, Nikolaos Aletras, and Jane Dwivedi-Yu. Active learning principles for incontext learning with large language models. arXiv preprint arXiv:2305.14264, 2023.

Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, and Willie Neiswanger. Sample efficient reinforcement learning from human feedback via active exploration. CoRR, abs/2312.00267, 2023. URL https://arxiv.org/abs/2312.00267.

Subhojyoti Mukherjee, Anusha Lalitha, Kousha Kalantari, Aniket Deshmukh, Ge Liu, Yifei Ma, and Branislav Kveton. Optimal design for human preference elicitation. In Advances in Neural Information Processing Systems 37, 2024.

William Muldrew, Peter Hayes, Mingtian Zhang, and David Barber. Active preference learning for large language models. arXiv preprint arXiv:2402.08114, 2024.

G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher. An analysis of approximations for maximizing submodular set functions - I. Mathematical Programming, 14(1): 265–294, 1978.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35, 2022.

Robin Lewis Plackett. The analysis of permutations. Journal of the Royal Statistical Society: Series C (Applied Statistics), 24(2):193–202, 1975.

Friedrich Pukelsheim. Optimal Design of Experiments. Society for Industrial and Applied Mathematics, 2006.

Filip Radlinski, Robert Kleinberg, and Thorsten Joachims. Learning diverse rankings with multi-armed bandits. In Proceedings of the 25th International Conference on Machine Learning, pages 784–791, 2008.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems 36, 2023.

Carlos Riquelme, George Tucker, and Jasper Snoek. Deep Bayesian bandits showdown: An empirical comparison of Bayesian deep networks for Thompson sampling. In Proceedings of the 6th International Conference on Learning Representations, 2018.

Antoine Scheid, Etienne Boursier, Alain Durmus, Michael Jordan, Pierre Menard, Eric Moulines, and Michal Valko. Optimal design for reward modeling in RLHF. CoRR, abs/2410.17055, 2024. URL https://arxiv.org/ abs/2410.17055.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL https://arxiv.org/abs/1707.06347.

John Stufken and Min Yang. Optimal designs for generalized linear models. In Design and Analysis of Experiments, pages 137–164. John Wiley & Sons, 2012.

Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 1998.

Kiran Thekumparampil, Gaurush Hiranandani, Kousha Kalantari, Shoham Sabach, and Branislav Kveton. Comparing few to rank many: Active human preference learning using randomized Frank-Wolfe. CoRR, abs/2412.19396, 2024. URL https://arxiv.org/ abs/2412.19396.

Jiahao Wang, Bolin Zhang, Qianlong Du, Jiajun Zhang, and Dianhui Chu. A survey on data selection for llm instruction tuning. arXiv preprint arXiv:2402.05123, 2024.

Junwen Yang and Vincent Tan. Minimax optimal fixedbudget best arm identification in linear bandits. In Advances in Neural Information Processing Systems 35, 2022.

Yiming Zhang, Shi Feng, and Chenhao Tan. Active example selection for in-context learning. arXiv preprint arXiv:2211.04486, 2022.

Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. Starling-7b: Improving llm helpfulness & harmlessness with rlaif, November 2023.

Shi Zong, Hao Ni, Kenny Sung, Nan Rosemary Ke, Zheng Wen, and Branislav Kveton. Cascading bandits for largescale recommendation problems. In Proceedings ofthe 32nd Conference on Uncertainty in Artificial Intelligence, 2016.

## A. Proofs and Supporting Lemmas

This section contains proofs of our main claims and supporting lemmas.

## A.1. Proof of Lemma 1

Let $v \in \mathbb { R }$ and $\mu ( v ) = 1 / ( 1 + \exp [ - v ] )$ . Then

$$
\frac {\partial}{\partial v} \mu (v) = - \frac {1}{(1 + \exp [ - v ]) ^ {2}} \frac {\partial}{\partial v} \exp [ - v ] = \frac {\exp [ - v ]}{(1 + \exp [ - v ]) ^ {2}} = \mu (v) (1 - \mu (v)).
$$

We start with computing the gradient of (6),

$$
\begin{array}{l} \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta ; \mathcal {S}) = - \sum_ {i \in \mathcal {S}} s _ {i} \frac {\nabla \mu_ {i} (\theta)}{\mu_ {i} (\theta)} - (1 - s _ {i}) \frac {\nabla \mu_ {i} (\theta)}{1 - \mu_ {i} (\theta)} = \beta \sum_ {i \in \mathcal {S}} (1 - s _ {i}) \mu_ {i} (\theta) \phi_ {i} - s _ {i} (1 - \mu_ {i} (\theta)) \phi_ {i} \\ \qquad = \beta \sum_ {i \in \mathcal {S}} (\mu_ {i} (\theta) - s _ {i}) \phi_ {i}. \end{array}
$$

It follows that the Hessian is

$$
\nabla^ {2} \mathcal {L} _ {\mathrm{DPO}} (\theta ; \mathcal {S}) = \nabla (\nabla \mathcal {L} _ {\mathrm{DPO}} (\theta ; \mathcal {S})) = \beta \sum_ {i \in \mathcal {S}} \phi_ {i} \nabla \mu_ {i} (\theta) = \beta^ {2} \sum_ {i \in \mathcal {S}} \mu_ {i} (\theta) (1 - \mu_ {i} (\theta)) \phi_ {i} \phi_ {i} ^ {\top}.
$$

The term $\phi _ { i } \phi _ { i } ^ { \top }$ is an outer product, which is positive semi-definite. Because $\mu _ { i } ( \theta ) ( 1 - \mu _ { i } ( \theta ) ) \geq 0$ , the Hessian is a weighted sum of positive semi-definite matrices, and thus a positive semi-definite matrix.

## A.2. Proof of Theorem 3

Let $\hat { \Sigma } _ { n } = \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } _ { * } ; \mathcal { S } _ { n } )$ . We start by noting that $\hat { \Sigma } _ { n }$ is a positive semi-definite matrix (Lemma 1). Therefore, $\mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ; \mathcal { S } _ { n } )$ is strongly convex in θ and

$$
\mathcal {L} _ {\mathrm{DPO}} (\hat {\theta} _ {n}; \mathcal {S} _ {n}) \geq \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}) + \langle \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}), \hat {\theta} _ {n} - \theta_ {*} \rangle + \frac {1}{2} \| \hat {\theta} _ {n} - \theta_ {*} \| _ {\hat {\Sigma} _ {n}} ^ {2}
$$

holds. Now we use that $\mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; S _ { n } ) \ge \mathcal { L } _ { \mathrm { D P O } } ( \hat { \theta } _ { n } ; S _ { n } )$ and that $\hat { \Sigma } _ { n } = \Sigma _ { n } - \gamma I _ { d }$ , rearrange the inequality, and get

$$
\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} ^ {2} \leq 2 \langle \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}), \theta_ {*} - \hat {\theta} _ {n} \rangle + \gamma \| \hat {\theta} _ {n} - \theta_ {*} \| _ {2} ^ {2}.
$$

Then we apply the Cauchy–Schwarz inequality to the right-hand side and get

$$
\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} ^ {2} \leq 2 \| \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}) \| _ {\Sigma_ {n} ^ {- 1}} \| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} + \gamma \| \hat {\theta} _ {n} - \theta_ {*} \| _ {2} ^ {2}.
$$

Now we divide both sides by $\Vert \hat { \theta } _ { n } - \theta _ { * } \Vert _ { \Sigma _ { n } } > 0$ and get

$$
\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} \leq 2 \| \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}) \| _ {\Sigma_ {n} ^ {- 1}} + \frac {\gamma \| \hat {\theta} _ {n} - \theta_ {*} \| _ {2} ^ {2}}{\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}}} \leq 2 \| \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}) \| _ {\Sigma_ {n} ^ {- 1}} + 2 \gamma^ {\frac {1}{2}}.
$$

The last inequality follows from

$$
\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} = \sqrt {(\hat {\theta} _ {n} - \theta_ {*}) ^ {\top} \Sigma_ {n} (\hat {\theta} _ {n} - \theta_ {*})} \geq \sqrt {\gamma} \| \hat {\theta} _ {n} - \theta_ {*} \| _ {2} ^ {2},
$$

which is proved using $\Sigma _ { n } \succeq \gamma I _ { d } .$ , and that $\lVert \hat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } _ { * } \rVert _ { 2 } \leq 2$

Therefore, to bound $\| \hat { \theta } _ { n } - \theta _ { * } \| _ { \Sigma _ { n } }$ , it suffices to show that $\| \nabla \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { * } ; \mathcal { S } _ { n } ) \| _ { \Sigma _ { n } ^ { - } }$ 1 is small with a high probability. We show this next. We start by recalling from Lemma 1 that

$$
\nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}) = \beta \sum_ {i \in \mathcal {S} _ {n}} (\mu_ {i} (\theta_ {*}) - s _ {i}) \phi_ {i},
$$

where $s _ { i }$ is a binary random variable with mean $\mathbb { E } \left[ s _ { i } \right] = \mu _ { i } ( \theta _ { * } )$ , as described in (12). Let $Z _ { i } = \mu _ { i } ( \theta _ { * } ) - s _ { i }$ . Since

$$
\Sigma_ {n} \succeq c _ {\mathrm{min}} \left(\frac {\gamma}{c _ {\mathrm{min}}} I _ {d} + \sum_ {i \in \mathcal {S} _ {n}} \phi_ {i} \phi_ {i} ^ {\top}\right),
$$

we get

$$
\left\| \nabla \mathcal {L} _ {\mathrm{DPO}} \left(\theta_ {*}; \mathcal {S} _ {n}\right) \right\| _ {\Sigma_ {n} ^ {- 1}} \leq \frac {\beta}{\sqrt {c _ {\min}}} \left\| \sum_ {i \in \mathcal {S} _ {n}} Z _ {i} \phi_ {i} \right\| _ {V _ {n} ^ {- 1}}
$$

for $\begin{array} { r } { V _ { n } = \gamma I _ { d } / c _ { \operatorname* { m i n } } + \sum _ { i \in S _ { n } } \phi _ { i } \phi _ { i } ^ { \top } } \end{array}$ . Finally, since $s _ { i }$ are conditionally independent given the history and their variance proxy is 0.25, we can use Theorem 1 of Abbasi-Yadkori et al. (2011) and get that

$$
\left\| \sum_ {i \in \mathcal {S} _ {n}} Z _ {i} \phi_ {i} \right\| _ {V _ {n} ^ {- 1}} \leq \sqrt {\frac {d}{4} \log \left(\frac {1 + c _ {\min} n / \gamma}{\delta}\right)}
$$

holds with probability at least $1 - \delta .$ . Finally, we collect all inequalities and get that

$$
\| \hat {\theta} _ {n} - \theta_ {*} \| _ {\Sigma_ {n}} \leq \| \nabla \mathcal {L} _ {\mathrm{DPO}} (\theta_ {*}; \mathcal {S} _ {n}) \| _ {\Sigma_ {n} ^ {- 1}} + 2 \gamma^ {\frac {1}{2}} \leq \sqrt {\frac {\beta^ {2} d}{c _ {\mathrm{min}}} \log \left(\frac {1 + c _ {\mathrm{min}} n / \gamma}{\delta}\right)} + 2 \gamma^ {\frac {1}{2}}
$$

holds with probability at least $1 - \delta .$

## A.3. Proof of Theorem 4

First, we introduce $\mu _ { t , i } = \mu _ { i } (  { \hat { \theta } } _ { t - 1 } )$ , and note that $v _ { t , i }$ in ADPO and $\mathtt { A D P 0 ^ { + } }$ can be redefined as

$$
v _ {t, i} = \beta \sqrt {\mu_ {t , i} (1 - \mu_ {t , i})} \phi_ {i}.
$$

Now note that

$$
\| \phi_ {i} \| _ {\Sigma_ {n} ^ {- 1}} ^ {2} = \phi_ {i} ^ {\top} \Sigma_ {n} ^ {- 1} \phi_ {i} \leq \frac {c _ {\mathrm{max}}}{c _ {\mathrm{min}}} \phi_ {i} ^ {\top} H _ {n} ^ {- 1} \phi_ {i}
$$

because $\begin{array} { r } { H _ { t } = \gamma I _ { d } + \sum _ { i \in { \cal S } _ { t } } v _ { t , i } v _ { t , i } ^ { \top } } \end{array}$ . Next we utilize the fact that the standard errors of the estimates decrease with more observations.

Lemma 5. For any $i \in [ N ]$ and $t \in [ n ]$

$$
\phi_ {i} ^ {\top} H _ {t} ^ {- 1} \phi_ {i} \leq \phi_ {i} ^ {\top} H _ {t - 1} ^ {- 1} \phi_ {i}.
$$

Proof. The proof follows from the Sherman–Morrison formula. Specifically, since

$$
H _ {t} ^ {- 1} = H _ {t - 1} ^ {- 1} - \frac {H _ {t - 1} ^ {- 1} \phi_ {i} \phi_ {i} ^ {\top} H _ {t - 1} ^ {- 1}}{1 + \phi_ {i} ^ {\top} H _ {t - 1} ^ {- 1} \phi_ {i}} \preceq H _ {t - 1} ^ {- 1},
$$

we get $v ^ { \top } H _ { t } ^ { - 1 } v \leq v ^ { \top } H _ { t - 1 } ^ { - 1 }$ v for any vector $v \in \mathbb { R } ^ { d }$ . This completes the proof.

Lemma 5 implies that

$$
\phi_ {i} ^ {\top} H _ {n} ^ {- 1} \phi_ {i} \leq \frac {1}{n} \sum_ {t = 1} ^ {n} \phi_ {i} ^ {\top} H _ {t - 1} ^ {- 1} \phi_ {i} \leq \frac {c _ {\max}}{n} \sum_ {t = 1} ^ {n} v _ {t, i} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, i}
$$

holds for any $i \in [ N ]$ . This allows us to attribute the quality of the solution to individual greedy steps in ADPO and $\mathtt { A D P 0 ^ { + } }$ The next step is to relate $v _ { t , i } ^ { \top } H _ { t - 1 } ^ { - 1 } v _ { t , i } \mathrm { ~ t o ~ } v _ { t , I _ { t } } ^ { \top } H _ { t - 1 } ^ { - 1 } v _ { t , I _ { t } }$ . The key observation is that

$$
\begin{array}{r l} & I _ {t} = \underset {i \in [ N ] \setminus \mathcal {S} _ {t - 1}} {\arg \max} \log \det (H _ {t - 1} + v _ {t, i} v _ {t, i} ^ {\top}) = \underset {i \in [ N ] \setminus \mathcal {S} _ {t - 1}} {\arg \max} \log \det (I _ {d} + H _ {t - 1} ^ {- \frac {1}{2}} v _ {t, i} v _ {t, i} ^ {\top} H _ {t - 1} ^ {- \frac {1}{2}}) \\ & \quad = \underset {i \in [ N ] \setminus \mathcal {S} _ {t - 1}} {\arg \max} \log (1 + v _ {t, i} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, i}) = \underset {i \in [ N ] \setminus \mathcal {S} _ {t - 1}} {\arg \max} v _ {t, i} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, i}. \end{array}
$$

The second equality holds because $H _ { t - 1 }$ is fixed when $I _ { t }$ is selected. The last equality holds because the logarithm is a monotone function. It follows that $I _ { t }$ is the index of the feature vector with the maximum variance.

If the scope of the maximization was $i \in [ N ]$ , the inequality $v _ { t , i } ^ { \top } H _ { t - 1 } ^ { - 1 } v _ { t , i } \le v _ { t , I _ { t } } ^ { \top } H _ { t - 1 } ^ { - 1 } v _ { t , I _ { t } }$ would hold for any $i \in [ N ]$ Since the scope is $i \in [ N ] \backslash S _ { t - 1 }$ , we make Assumption 4, which equates to assuming that $\phi _ { i }$ are sufficiently diverse. We also use the following logarithmic transformation.

Lemma 6. For any $v \in \mathbb { R } ^ { d }$ and t ∈ [n], $t \in [ n ]$

$$
v ^ {\top} H _ {t - 1} ^ {- 1} v \leq \frac {c _ {\max}}{\gamma \log (1 + c _ {\max} / \gamma)} \log (1 + v ^ {\top} H _ {t - 1} ^ {- 1} v).
$$

Proof. We start with an upper bound on $v ^ { \top } H _ { t - 1 } ^ { - 1 } v .$ By Weyl’s inequalities, we have

$$
\lambda_ {1} (H _ {t - 1} ^ {- 1}) = \lambda_ {d} ^ {- 1} (H _ {t - 1}) \leq \lambda_ {d} ^ {- 1} (\gamma I _ {d}) = 1 / \gamma .
$$

Thus, under the assumption that $\| v \| _ { 2 } ^ { 2 } \leq c _ { \operatorname* { m a x } } .$ we have $v ^ { \top } H _ { t - 1 } ^ { - 1 } v \leq c _ { \operatorname* { m a x } } / \gamma$ . Now note that for $y \in [ 0 , y _ { \mathrm { m a x } } ]$

$$
y = \frac {y}{\log (1 + y)} \log (1 + y) \leq \left(\max _ {y \in [ 0, y _ {\max} ]} \frac {y}{\log (1 + y)}\right) \log (1 + y) = \frac {y _ {\max}}{\log (1 + y _ {\max})} \log (1 + y).
$$

Finally, we set $y = v ^ { \top } H _ { t - 1 } ^ { - 1 }$ v and $y _ { \mathrm { m a x } } = c _ { \mathrm { m a x } } / \gamma$ , and get our claim.

Now we apply Assumption 4 and Lemma $^ { 6 , }$ use the telescoping property of the sum, and get

$$
\begin{array}{l} \sum_ {t = 1} ^ {n} v _ {t, i} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, i} \leq \kappa \sum_ {t = 1} ^ {n} v _ {t, I _ {t}} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, I _ {t}} \leq c \sum_ {t = 1} ^ {n} \log (1 + v _ {t, I _ {t}} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, I _ {t}}) = c \sum_ {t = 1} ^ {n} \log \det (I _ {d} + H _ {t - 1} ^ {- \frac {1}{2}} v _ {t, I _ {t}} v _ {t, I _ {t}} ^ {\top} H _ {t - 1} ^ {- \frac {1}{2}}) \\ \qquad = c \sum_ {t = 1} ^ {n} \log \det (H _ {t - 1} + v _ {t, I _ {t}} v _ {t, I _ {t}} ^ {\top}) - \log \det (H _ {t - 1}) = c \sum_ {t = 1} ^ {n} \log \det (H _ {t}) - \log \det (H _ {t - 1}) \\ \qquad = c (\log \det (H _ {n}) - \log \det (H _ {0})) = c \log \det (H _ {0} ^ {- \frac {1}{2}} H _ {n} H _ {0} ^ {- \frac {1}{2}}), \end{array}
$$

where $\begin{array} { r } { c = \frac { c _ { \mathrm { m a x } } \kappa } { \gamma \log ( 1 + c _ { \mathrm { m a x } } / \gamma ) } } \end{array}$ . Furthermore,

$$
\begin{array}{c} \log \det (H _ {0} ^ {- \frac {1}{2}} H _ {n} H _ {0} ^ {- \frac {1}{2}}) \leq d \log \left(\frac {1}{d} \operatorname{tr} (H _ {0} ^ {- \frac {1}{2}} H _ {n} H _ {0} ^ {- \frac {1}{2}})\right) = d \log \left(1 + \frac {1}{d} \sum_ {t = 1} ^ {n} \operatorname{tr} (H _ {0} ^ {- \frac {1}{2}} v _ {t, I _ {t}} v _ {t, I _ {t}} ^ {\top} H _ {0} ^ {- \frac {1}{2}})\right) \\ = d \log \left(1 + \frac {1}{d} \sum_ {t = 1} ^ {n} v _ {t, I _ {t}} ^ {\top} H _ {0} ^ {- 1} v _ {t, I _ {t}}\right) \leq d \log \left(1 + \frac {c _ {\max} n}{\gamma d}\right). \end{array}
$$

Finally, we combine all claims and get

$$
\phi_ {i} ^ {\top} H _ {n} ^ {- 1} \phi_ {i} \leq \frac {1}{n} \sum_ {t = 1} ^ {n} \phi_ {i} ^ {\top} H _ {t - 1} ^ {- 1} \phi_ {i} \leq \frac {c _ {\max} \kappa}{n} \sum_ {t = 1} ^ {n} v _ {t, I _ {t}} ^ {\top} H _ {t - 1} ^ {- 1} v _ {t, I _ {t}} \leq \frac {c _ {\max} ^ {2} \log \left(1 + \frac {c _ {\max} n}{\gamma d}\right)}{\gamma \log (1 + c _ {\max} / \gamma)} \frac {\kappa d}{n}.
$$

This completes the proof.

## B. Ablation Study

In Section 6.1, we experiment with $\beta = 1$ . There is nothing specific about this choice. In Figure 3, we report results for $\beta \in \{ 2 , 5 \}$ and observe improvements in both settings.

To increase the stability of our algorithms at small sample sizes, we replace $\mu _ { i } (  { \hat { \theta } } _ { t } ) ( 1 - \mu _ { i } (  { \hat { \theta } } _ { t } ) )$ with a high probability upper confidence bound (UCB). Let $\hat { \Sigma } _ { t }$ be the covariance matrix for $\widehat { \theta } _ { t }$ . Then the UCB is computed as

$$
U _ {i} = \mu (z _ {i}) (1 - \mu (z _ {i})), \quad z _ {i} = \max \left\{\left| \beta (\phi_ {i} ^ {\top} \hat {\theta} _ {t} - b _ {i}) \right| - \alpha \sqrt {\phi_ {i} ^ {\top} \hat {\Sigma} _ {t} \phi_ {i}}, 0 \right\}\tag{13}
$$

for some $\alpha > 0$ . We set $\alpha = 3$ in Section 6. In Figure 4, we set $\alpha = 0$ and observe that this has no major impact on our trends as the number of data points n increases.






Figure 3. Experiments with log-linear policies on the CIFAR-10 dataset, with $\beta = 2$ (first row) and $\beta = 5$ (second row).






Figure 4. Experiments with log-linear policies on the CIFAR-10 (first row) and CIFAR-100 (second row) datasets with $\alpha = 0$ in (13).

## C. Related Work

The closest related works are on active learning with preferential feedback, and we review them first (Appendix C.1). Then we review active learning for fine-tuning (Appendix C.2) and other related works (Appendix C.3).

## C.1. Active Learning for Preferential Feedback

Mehta et al. (2023) applied active learning to DPO in Section 5. Their acquisition function is

$$
I _ {t} = \underset {i \in [ N ]} {\arg \max} \left(\underset {j \in [ 2 ]} {\max} U (x _ {i}, y _ {i, j}) - \underset {j \in [ 2 ]} {\max} L (x _ {i}, y _ {i, j})\right),
$$

where $U ( x , y )$ is the UCB and $L ( x , y )$ is the LCB of $r ( x , y )$ . The analysis is for dueling the UCB response with a random response. Their optimized metric is the maximum gap

$$
\max _ {i \in [ N ]} (\max _ {j \in [ 2 ]} r (x _ {i}, y _ {i, j}) - r (x _ {i}, \hat {y} _ {i}))  ,\tag{14}
$$

where $\hat { r }$ is the estimated reward model and $\hat { y } _ { i } = \arg \operatorname* { m a x } _ { j \in [ 2 ] } \hat { r } ( x _ { i } , y _ { i , j } )$ is the best response given $x _ { i } .$ . They prove that the maximum gap is $O ( 1 / \sqrt { n } )$ for sampling with replacement.

Das et al. (2024) proposed two algorithms for active RLHF. The acquisition function in APO is

$$
I _ {t} = \underset {i \in [ N ]} {\arg \max} \left\| \phi_ {i} \right\| _ {H _ {t} ^ {- 1} (\hat {\theta} _ {t - 1})},
$$

where $H _ { t } (  { \hat { \theta } } _ { t - 1 } )$ is a logistic regression Hessian in round $t ,$ which is re-estimated in each round. They prove that (14) is $O ( 1 / \sqrt { n } )$ for sampling with replacement. APO is not evaluated. This is the closest algorithm design to ADPO. The main difference in ADPO is that we maximize the information gain (line 6) and do not compute $H _ { t } ^ { - 1 } ( \hat { \theta } _ { t - 1 } )$ . Das et al. (2024) also proposed a practical APO,

$$
I _ {t} = \underset {i \in [ N ]} {\arg \max} \| \phi_ {i} \| _ {H _ {t} ^ {- 1}},
$$

where $H _ { t }$ is a linear regression Hessian in round t. Practical APO is not analyzed. We use it as a baseline in Section 6.

Mukherjee et al. (2024) studied active learning with absolute and ranking feedback with $K \geq 2$ responses. For $K = 2$ , their algorithm Dope is $I _ { t } \sim \pi _ { * }$ , where $\pi _ { * }$ is a distribution over $N$ prompts with 2 responses obtained by the D-optimal design. They prove that

$$
\underset {i \in [ N ]} {\arg \max} | \phi_ {i} ^ {\top} (\hat {\theta} - \theta_ {*}) | = O (1 / \sqrt {n})
$$

for sampling with replacement, where $\theta _ { * }$ is the true model parameter and $\hat { \theta }$ is its estimate from n observations. Dope is evaluated on RLHF datasets. Thekumparampil et al. (2024) extended Mukherjee et al. (2024) to ranking N items from $K \leq N$ responses.

Liu et al. (2024) extended APO of Das et al. (2024) to selecting both the prompt and teacher model. They prove that (14) is $O ( 1 / \sqrt { n } )$ for sampling with replacement. The proposed algorithm is empirically evaluated.

Scheid et al. (2024) proposed offline and online algorithms for active learning of reward models in RLHF. The offline algorithm, which is in the same setting as our work, computes the D-optimal design, similarly to Mukherjee et al. (2024) for $K = 2$ , and explores by sampling with replacement. They prove a $O ( 1 / \sqrt { n } )$ bound on (14). The paper does not contain any experiments.

Ji et al. (2024) proposed two active learning algorithms: APPO and ADPO. APPO is a regret minimizing algorithm similar to those in dueling bandits. In round t, APPO is given a prompt as an input and proposes two responses to duel. APPO is analyzed. ADPO is a heuristic that queries responses on prompts where the agent is uncertain. The response is uncertain if $| r ( x _ { i } , y _ { i , 1 } ) - r ( x _ { i } , y _ { i , 2 } ) |$ in the DPO objective is high.

Muldrew et al. (2024) proposed an active learning algorithm for DPO that repeatedly acquires labels and fine-tunes on them. The data are acquired in batches until a budget is met. The acquisition function is

$$
I _ {t} = \underset {i \in [ N ]} {\arg \max} | \hat {r} (x _ {i}, y _ {i, 1}) - \hat {r} (x _ {i}, y _ {i, 2}) |,
$$

where $\hat { r }$ is the estimated reward model. We use it as a baseline in Section 6.

Guo et al. (2024) proposed online DPO from AI feedback. The key is to elicit AI feedback instead of human feedback and then use it in DPO. This is an empirical paper.

Chen et al. (2024) proposed active learning with coresets for reward models. They learn cluster centroids in the space of prompt embeddings that minimize the maximum distance of the prompt to its closest centroid. This is an empirical paper.

## C.2. Active Learning for Fine-Tuning

There are many related works on active learning in LLMs (Margatina et al., 2023; Bayer and Reuter, 2024; Zhang et al., 2022). A recent survey by Wang et al. (2024) categorizes existing methods for data selection in instruction tuning. Most of these methods rely on heuristic approaches, such as uncertainty sampling, clustering, or diversity-based strategies, which often lack theoretical grounding. Doucet et al. (2024) proposed a method that bridges diversity and uncertainty in active learning by leveraging self-supervised pre-training to address the cold-start problem and enhance data efficiency. However, these approaches do not align data selection directly with the task-specific objective, limiting their effectiveness in optimizing downstream performance. Zhang et al. (2022) used LLMs for selecting instances for in-context learning. More recently, Bayer and Reuter (2024) proposed ActiveLLM, which is a pool-based sampling method that leverages LLMs to select batches of instances for humans to label. Despite this fundamental difference, they also study two variants of their approach, one that incorporates feedback and another one that does not.

## C.3. Multi-Armed Bandits

Our setting is also related to multi-armed bandits. Due to the budget n, it is reminiscent of fixed-budget best arm identification (BAI) (Bubeck et al., 2009; Audibert et al., 2010; Azizi et al., 2022; Yang and Tan, 2022). The main difference is that we do not want to identify the best arm. We want to get a good estimate for a set of arms, essentially pairs of items, in the worst case. Online learning to rank has also been studied extensively (Radlinski et al., 2008; Kveton et al., 2015; Zong et al., 2016; Li et al., 2016; Lagree et al., 2016). We do not minimize cumulative regret or try to identify the best arm.
