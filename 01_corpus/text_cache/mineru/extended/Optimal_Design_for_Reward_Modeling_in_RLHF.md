# Optimal Design for Reward Modeling in RLHF

Antoine Scheid CMAP - CNRS Ecole polytechnique <sup>´</sup> Palaiseau, France

Etienne Boursier INRIA Saclay Universit´e Paris Saclay, LMO Orsay, France

Alain Durmus CMAP - CNRS Ecole polytechnique<sup>´</sup> Palaiseau, France

Michael I. Jordan U.C., Berkeley INRIA, ENS Paris, France

Pierre M´enard ENS Lyon Lyon, France

Eric Moulines CMAP - CNRS Ecole polytechnique <sup>´</sup> Palaiseau, France

Michal Valko INRIA

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has become a popular approach to align language models (LMs) with human preferences. This method involves collecting a large dataset of human pairwise preferences across various text generations and using it to infer (implicitly or explicitly) a reward model. Numerous methods have been proposed to learn the reward model and align a LM with it. However, the costly process of collecting human preferences has received little attention and could benefit from theoretical insights. This paper addresses this issue and aims to formalize the reward training model in RLHF. We frame the selection of an efective dataset as a simple regret minimization task, using a linear contextual dueling bandit method. Given the potentially large number of arms, this approach is more coherent than the best-arm identification setting. We then propose an ofline framework for solving this problem. Under appropriate assumptions — linearity of the reward model in the embedding space, and boundedness of the reward parameter — we derive bounds on the simple regret. Finally, we provide a lower bound that matches our upper bound up to constant and logarithmic terms. To our knowledge, this is the first theoretical contribution in this area to provide an ofline approach as well as worst-case guarantees.

## 1 Introduction

In learning from human feedback (Christiano et al., 2017; Naveed et al., 2023; Wei et al., 2023), an agent learns to act based on a preference signal. This subject has recently seen a surge of interest due to its efectiveness for aligning pre-trained Large Language Models (LLMs) with human preferences. Typically, the human feedback is gathered by constructing a large dataset of contexts (prompts), pairs of language model outputs (completions), and human preference between the pairs. Given this preference dataset, several meth ods have been proposed to align pre-trained large language models (LLMs). For instance, reinforcement learning from human feedback (RLHF, Ziegler et al. 2019) involves training a reward model from the preference dataset and then fine-tuning the pre-trained LLM using reinforcement learning, typically with the PPO algorithm (Schulman et al., 2017), to maximize the reward model. Another approach is the direct preference optimization (DPO) procedure (Rafailov et al., 2024), where the pre-trained LLM is directly fine-tuned with the preference dataset by learning an implicit reward model.

Learning from human feedback has proven to have extraordinary abilities in its application to various fields, including robotics (Tucker et al., 2020; Bıyık et al., 2020, 2024), language models (Ouyang et al., 2022; Touvron et al., 2023; Bubeck et al., 2023), and recommendations (Chen et al., 2024b; Zhao et al., 2023). However, the majority of works in this area focus on preference optimization, and little is known about how to eficiently construct a human preferences dataset. In this work, we provide a theoretically grounded insight into the data collection process for learning a reward model.

In general, a very large dataset of prompts and associated generations $\mathcal { D } _ { \mathrm { i n i } }$ is sampled. Among this dataset, a smaller one is selected $\left( \mathcal { D } _ { \mathrm { s e l e c t } } \right)$ and receives feedback from human labelers, due to the cost of labeling the generations. In practice, the dataset selection is achieved without too much care, hence a loss in the information that could have been retrieved from the original dataset $\mathcal { D } _ { \mathrm { i n i } }$ . Some techniques to improve the selection rely on heuristics or black-box methods (Shen et al., 2024; Dong et al., 2024; Chang et al., 2024) but lack of provable bounds on the optimality of the procedure.

Based on these considerations, we study the ofline selection of the optimal dataset $\mathcal { D } _ { \mathrm { s e l e c t } }$ Our goal is to minimize the number of samples that need to be rated by labelers while retaining as much valuable information as possible from the initial dataset $\mathcal { D } _ { \mathrm { i n i } }$ . To achieve this, we propose a new method called ODPO: Optimal Design for Policy Optimization. This method guides the dataset selection process using the solution to an optimal design problem. We prove that ODPO is optimal from a worst-case perspective. Interestingly, ODPO can be applied to select pairs in the dataset $\mathcal { D } _ { \mathrm { s e l e c t } }$ at a low cost before running any preference optimization procedure.

## We summarize our contributions as follows:

• Under the Bradley-Terry model and the assumption of a contextual linear bandit for the reward model, we formalize a pure exploration bandit framework for the collection of samples used to train the reward model in RLHF.

• Within that setting, we introduce ODPO: Optimal Design for Preference Optimization, which optimally selects the best arms to learn the reward model and we upper bound its simple regret.

• We prove the optimality of our technique with a lower bound which matches our upper bound up to logarithmic factors.

Note that few works study the optimal way of choosing the dataset of human preferences. Thus, we are very enthusiastic about the potential impact of our method and theoretical results on the reward training steps. As mentioned by Casper et al. (2023), collecting data that is representative of human preferences is an open problem in RLHF and deserves more attention, hence our attempt in this direction.

## 2 Setting

## 2.1 Background on RLHF

For what follows, $\mathcal { X }$ represents the set of contexts (or prompts) and $\mathcal { V }$ the set of generations (or completions). Human labelers are presented with pairs of prompt-completion tuples, denoted $( x , y )$ and $( x , y ^ { \prime } )$ which we can express as $\{ x , y , y ^ { \prime } \}$ . Formally, a language model ϕ is a mapping from the set of contexts $\mathcal { X }$ to probability distributions over the set of possible generations $y .$ The task of the labelers (annotators) is to determine which completion between y and $y ^ { \prime }$ is more accurate or preferable in the context of $x ,$ denoted as $y \succ y ^ { \prime } | x .$ , when $( x , y )$ is preferred over $( x , y ^ { \prime } )$ . To account for human uncertainty, we model the binary feedback $\mathbb { 1 } ( y \succ y ^ { \prime } | x )$ process probabilistically by assuming a preference probability P: the event $\{ y \succ y ^ { \prime } | x \}$ coded as a binary variable $\mathbb { 1 } ( y \succ y ^ { \prime } | x )$ occurs with probability $\mathbb { P } ( y \succ y ^ { \prime } | x )$

The Bradley-Terry model (Bradley and Terry, 1952) provides a framework for modeling preferences based on real-valued rewards. Given a reward function $r ( x , y )$ that assigns a score to each context-generation pair $( x , y )$ , the probability of favoring one generation over another is expressed as follows

$$
\begin{array}{c} \mathbb {P} (y \succ y ^ {\prime} | x) = \sigma (r (x, y) - r (x, y ^ {\prime})) \\ = 1 / (1 + e ^ {- (r (x, y) - r (x, y ^ {\prime}))}) , \end{array}\tag{1}
$$

where $\sigma$ is the sigmoid function. It is worth noting that alternative preference models, such as the Plackett-Luce model (Plackett, 1975), can be used instead of this one. After having selected a dataset of pairs ${ \mathcal D } _ { \mathrm { s e l e c t } } ~ = ~ \{ X _ { t } , Y _ { t } ^ { ( 1 ) } , Y _ { t } ^ { ( \bar { 2 } ) } \} _ { t \in [ T ] }$ and the associated human preferences to make it $\tilde { \mathcal { D } } _ { \mathrm { s e l e c t e d } } ~ =$ $\{ X _ { t } , Y _ { t } ^ { ( 1 ) } , Y _ { t } ^ { ( 2 ) } , \mathbb { 1 } ( Y _ { t } ^ { ( 1 ) } \succ Y _ { t } ^ { ( 2 ) } | X _ { t } ) \} _ { t \in [ T ] }$ , the estimated reward function ˆr is computed as the minimum of the loss

$$
\mathcal {J} (r) = - \mathbb {E} _ {(x, y, y ^ {\prime}) \sim \mathcal {D} _ {\mathrm{select}}} [ \log (\mathbb {P} (y \succ y ^ {\prime} | x)) ] ,\tag{2}
$$

and is then used to fine-tune the model $\phi$ which needs to maximize this reward while staying close from the initial model $\phi _ { 0 }$ , which is achieved by minimizing the following loss

$$
\mathcal {L} (\phi) = \mathbb {E} _ {\phi} [ \hat {r} (x, y) ] - \gamma \mathrm{D} _ {\mathrm{KL}} (\phi | | \phi_ {0}),\tag{3}
$$

where $\gamma$ is some constant and $\mathrm { D } _ { \mathrm { K I } }$ stands for the Kullback-Leibler divergence.

Despite a growing literature around the optimization procedures (2), (3) (see, e.g., Schulman et al., 2017; Rafailov et al., 2024; Azar et al., 2024), little has been done to select optimally the human-labelled dataset $\mathcal { D } _ { \mathrm { s e l e c t } }$ , although it crucially impacts the reward training or the policy optimization.

In practice, the selection of the pairs $( y _ { n } ^ { 1 } , \ldots , y _ { n } ^ { K } )$ associated with the n-th prompt $x _ { n }$ is achieved with the initial model $\phi _ { 0 }$ sampling several generations for the same context (with a change of the temperature between the diferent ones) from which two generations are randomly selected. The full dataset is then given to labelers for rating. T and N are of order 1000 to 100000 for usual datasets while K is around one or a few dozens.

## 2.2 RLHF as a Dueling Bandit Problem

We now introduce our ofline setting, which is one of the main novelty of our approach as compared to previous works around this topic. Relying on optimal design and statistical foundations, the objective is to choose a dataset $\mathcal { D } _ { \mathrm { s e l e c t } }$ of prompts-generations that maximizes the information gained from human label-$\mathrm { e r s ^ { \prime } }$ feedback. Our approach to minimize the size of the collected dataset is notable for two reasons. First, it aligns with common practice, as it is impractical to operate online and get labelers’ feedback before choosing the next pair. Second, we establish matching upper and lower bounds, up to constant and logarithmic factors, which ensures the eficiency of ODPO.

We make the assumption of a contextual linear reward, hence the existence of a known feature map ψ : $\mathcal { X } \times$ $\mathcal { V } \to \mathbb { R } ^ { d } , x , y \mapsto \psi ( x , y )$ such that for any $x , y \in \mathcal { X } \times \mathcal { Y }$

$$
r (x, y) = \langle \theta^ {\star}, \psi (x, y) \rangle .\tag{4}
$$

The reward is given with respect to the embedding of the promt-completion pair. Simple encoder models such as BERT, RoBERTa or SBERT (Reimers, 2019; Devlin, 2018; Liu, 2019) can be used for the embedding step. Usually, the feature map can be obtained by removing the last layer of the initially trained model.

Figure 1: Illustration of ODPO among the whole RLHF framework.

We now work in the embedding space $\mathbb { R } ^ { d }$ for the dataset selection. Defining N as the number of available prompts, and K the number of associated generations for each prompt (although our results still hold for N, K tending to infinity), we have access to an initial dataset that can be written $\{ ( x _ { n } , y _ { n } ^ { k } ) \} _ { n \in [ N ] , k \in [ K ] } .$ We model it as a union of N sets $\mathcal { A } _ { n } , n \in [ N ]$ and write ${ \mathcal { D } } _ { \mathrm { i n i } } = \cup _ { n \in [ N ] } { \mathcal { A } } _ { n }$ . A subset $\mathcal { A } _ { n }$ represents the set of all the prompts associated with the same generation $x _ { n }$ : for any $n \in [ N ] , \mathcal { A } _ { n } = \{ a _ { n } ^ { 1 } , . . . , a _ { n } ^ { K } \}$ and $a _ { n } ^ { k } = \psi ( x _ { n } , y _ { n } ^ { k } )$ with $n \in [ N ] , k \in [ K ]$ . The goal of our procedure is to select a subset of $T \in \mathbb { N } ^ { \star }$ pairs of generations, each pair being associated with the same context. Formally, it boils down to choosing a sequence $\{ ( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) \} _ { t \in [ T ] }$ of pairs in $\mathcal { D } _ { \mathrm { i n i } }$ that receive a feedback from labelers such that for any $t \in [ T ] , ( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) =$ $( a _ { n } ^ { i } , a _ { n } ^ { j } )$ for some $n \in [ N ] , i , j \in [ K ]$ . The selected dataset $\mathcal { D } _ { \mathrm { s e l e c t } } = \{ ( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) \} _ { t \in [ T ] }$ is given to labelers who give the winner of the duel between $A _ { t } ^ { 1 }$ and $A _ { t } ^ { 2 }$ for any $t \in [ T ]$ . The preference feedback is then received as

$\{ A _ { t } ^ { 1 } \succ A _ { t } ^ { 2 } \}$ with probability $\sigma ( \langle { \theta ^ { \star } , A _ { t } ^ { 1 } - A _ { t } ^ { 2 } } \rangle )$ or $\{ A _ { t } ^ { 2 } \succ A _ { t } ^ { 1 } \}$ with probability $\sigma ( \langle { \theta ^ { \star } , A _ { t } ^ { 2 } - A _ { t } ^ { 1 } } \rangle )$ and we encode it as a random variable $Y _ { t }$ , following

$$
Y _ {t} = \mathbb {1} (A _ {t} ^ {1} \succ A _ {t} ^ {2}),\tag{5}
$$

which therefore follows the distribution

$$
\begin{array}{c} \mathbb {P} _ {\theta^ {\star}} (Y _ {t} = 1 | \mathcal {F} _ {t - 1}) = \sigma (\langle \theta^ {\star}, A _ {t} ^ {1} - A _ {t} ^ {2} \rangle) \\ = 1 / (1 + e ^ {- \langle \theta^ {\star}, A _ {t} ^ {1} - A _ {t} ^ {2} \rangle})  , \end{array}\tag{6}
$$

for the unknown reward parameter $\theta ^ { \star }$ . We also define the set of diferences between all possible action pairs ${ \mathcal { B } } = \{ a _ { n } ^ { i } - a _ { n } ^ { j } \} _ { n \in [ N ] , i , j \in [ K ] } = \cup _ { n \in [ N ] } \{ A _ { n } - { \mathcal { A } } _ { n } \}$ , as well as its cardinal $L = \dot { \mathrm { C a r d } } ( B ) \leqslant \dot { T } \dot { K } ^ { 2 }$ . Choosing an arbitrary ordering among the elements of B, we can write $\beta = \{ b _ { l } \} _ { l \in [ L ] }$

It is important to note that we considered a finite action space for the sake of clarity. However, our theory still holds for an arbitrary action space of possibly infinite size. We would keep the same bounds depending solely on T and d due to the leverage of optimal design theory which circumvents the burden of having a large action space and relies on a distribution $\pi ^ { \star }$ with a finite and bounded support.

Based on the feedback $( Y _ { t } ) _ { t \in [ T ] }$ from the preference pairs, our procedure first estimates $\theta ^ { \star }$ . Then, for any context $x _ { n }$ given as an argument, a completion $y _ { n } ^ { i } , i \in [ K ]$ or equivalently an action $\hat { a } _ { T } ( \mathcal { A } _ { n } ) \in \mathcal { A } _ { n }$ can be chosen. We now define the simple regret of an algorithm ALG, as

$$
\mathfrak {R} _ {\mathrm{ALG}} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) = \max _ {n \in [ N ]} \max _ {a \in \mathcal {A} _ {n}} \left\langle \theta^ {\star}, a - \hat {a} _ {T} (\mathcal {A} _ {n}) \right\rangle ,\tag{7}
$$

and defining $a _ { n } ^ { \star } = \mathrm { a r g m a x } _ { a \in \mathcal { A } _ { n } } \langle \theta ^ { \star } , a \rangle$ , we have that

$$
\mathfrak {R} _ {\mathrm{ALG}} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) = \max _ {n \in [ N ]} \left\langle \theta^ {\star}, a _ {n} ^ {\star} - \hat {a} _ {T} (\mathcal {A} _ {n}) \right\rangle ,
$$

Note that in our setup, we are looking for an algorithm that converges for any possible parameter $\theta ^ { \star }$ in the unit ball. We also consider any set of actions $( { \mathcal { A } } _ { n } ) _ { n \in [ N ] }$ - which makes our results robust in an adversarial, non i.i.d. setting. Minimizing the simple regret is a coherent objective to study optimal dataset selection. Since we only care about selecting informative arms and about the quality of the prediction after all the arms have been sampled, it makes more sense than looking at the cumulative reward. Imagine that labelers need to rate a pair of bad completions: there is no harm for anyone. Secondly, it is hard to make hypotheses on the form of the actions sets or the reward parameters for embeddings of LM generations, hence the fact that we do not make any i.i.d. assumption.

Objectives. Note that we could have thought about diferent objectives for our problem, instead of minimizing the simple regret:

• Best-arm identification: one formulation of it within our setup would be to maximize $\mathbb { P } ( \hat { a } _ { T } ( \mathcal { A } _ { n } ) = a _ { n } ^ { \star } )$ over any $n \in [ N ]$

• Arm distance minimization: for any $T \in \mathbb { N } , n \in$ $[ N ]$ , minimize $\Vert \hat { a } _ { T } ( \mathcal { A } _ { n } ) - a _ { n } ^ { \star } \Vert$ for some well-chosen norm $\| \cdot \|$

We do not focus on arm distance minimization here, as it is dificult to quantify the diference in quality between two LM generations based on their distance or cosine similarity in $\mathbb { R } ^ { d }$ (Steck et al., 2024). Additionally, since the reward gap between two arms can be arbitrarily small and approach zero, the concept of best-arm does not really apply to our setup. Instead, seeking simple regret minimization efectively captures the quality of the dataset selection process by measuring how well the sampled pairs align the reward model with human preferences - with the one-step final reward becoming close from optimality.

We make the following assumption about the reward parameter as well as the embeddings of the pairs in R<sup>d</sup>.

H1 (Boundedness of action and parameter). For any $x , y \in \mathcal { X } { \times } \mathcal { Y } , \psi ( x , y ) \in \mathrm { B } ( 0 , 1 )$ , where $\mathrm { B } ( 0 , 1 )$ stands for the unit ball in $\mathbb { R } ^ { d }$ . We also suppose that $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$

## 2.3 Parameter Estimation

Log-likelihood and design matrix. Several useful quantities appear in the rest of the paper. Since they are at the core of our algorithms and results, we introduce them now. We define the regularized log-likelihood L for the collected samples up to time t and a reward parameter $\theta \in \mathbb { R } ^ { d }$ as

$$
\begin{array}{r l} \mathcal {L} _ {t} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta) & = \sum_ {s = 1} ^ {t - 1} \log (\mathbb {P} _ {\theta} (A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s})) \\ & \quad - \lambda \| \theta \| _ {2} ^ {2} / 2, \end{array} \tag {8}\tag{8}
$$

which can be rewritten as

$$
\begin{array}{l} \mathcal {L} _ {t} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta) = \sum_ {s = 1} ^ {t - 1} Y _ {s} \log (\sigma (\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle)) \\ \qquad + \sum_ {s = 1} ^ {t - 1} (1 - Y _ {s}) \log (\sigma (- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle)) - \lambda \| \theta \| _ {2} ^ {2} / 2, \end{array}
$$

and the maximum likelihood estimator at step t (MLE) $\widehat { \theta } _ { t }$ is computed following

$$
\hat {\theta} _ {t} \in \operatorname{argmax} _ {\theta \in \mathbb {R} ^ {d}} \mathcal {L} _ {t} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta).\tag{9}
$$

Lemma 1. We can diferentiate the likelihood defined in (8), and obtain

$$
\begin{array}{l} \nabla_ {\theta} \mathcal {L} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta) \\ = \sum_ {s = 1} ^ {t - 1} (Y _ {s} - \mathbb {P} _ {\theta} (Y _ {s} = 1)) (A _ {s} ^ {1} - A _ {s} ^ {2}) - \lambda \theta , \end{array}\tag{10}
$$

which gives by definition of the maximum likelihood estimator in (9), that $\widehat { \theta } _ { t }$ must satisfy

$$
\sum_ {s = 1} ^ {t - 1} (Y _ {s} - \mathbb {P} _ {\hat {\theta} _ {t}} (Y _ {s} = 1)) (A _ {s} ^ {1} - A _ {s} ^ {2}) - \lambda \hat {\theta} _ {t} = 0.\tag{11}
$$

Defining for any $t \in [ T + 1 ]$ the function $H _ { t } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ $\begin{array} { r } { \theta \mapsto \lambda \theta + \sum _ { s = 1 } ^ { t - 1 } \sigma ( \langle \theta , A _ { s } ^ { 1 } - A _ { s } ^ { 2 } \rangle ) ( A _ { s } ^ { 1 } - A _ { s } ^ { 2 } ) } \end{array}$ , we obtain by definition of $\widehat { \theta } _ { t }$ that

$$
\sum_ {s = 1} ^ {t - 1} Y _ {s} (A _ {s} ^ {1} - A _ {s} ^ {2}) = H _ {t} (\hat {\theta} _ {t}).\tag{12}
$$

Note that we define the MLE as the maximizer of the likelihood over the whole set $\mathbb { R } ^ { d }$ although we know that under H1, $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ . This is why we define the projected MLE estimator $\widehat { \theta } _ { t } ^ { P } \cdot \mathrm { a }$ similar quantity is also used by Faury et al. (2020) - as

$$
\hat {\theta} _ {t} ^ {P} = \underset {\theta \in \mathrm{B} (0, 1)} {\operatorname{argmin}} \| H _ {t} (\theta) - H _ {t} (\hat {\theta} _ {t}) \| _ {V _ {t} ^ {- 1}},\tag{13}
$$

where $V _ { t }$ stands for the design matrix in our problem, defined as

$$
V _ {t} = \lambda I + \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) (A _ {s} ^ {1} - A _ {s} ^ {2}) ^ {T},\tag{14}
$$

where $\lambda > 0$ is a regularization parameter - the same as for the likelihood. Since we work in an ofline fashion, we have access to all the data $\{ a _ { n } ^ { k } \} _ { n \in [ N ] , k \in [ K ] }$ and work on it, trying to extract as much knowledge as possible from the pairwise comparisons.

Note that a lot of applied works help to circumvent the burden of manipulating very large set of parameters in language modeling (Hu et al., 2021; Houlsby et al., 2019; Lester et al., 2021), such as the computation of inversion of matrices.

## 3 Ofline and Online Algorithms

Algorithms for our setting sample T pairs from the set $\mathcal { D } _ { \mathrm { i n i } }$ (possibly with repetition). For any $t \in [ T ]$ , we write $( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } )$ for the pair of actions sampled by ODPO at iteration $t ,$ even though there is no ”time ordering” in the procedure. This choice corresponds to taking an action $B _ { t } \in B \colon B _ { t } = A _ { t } ^ { 1 } - A _ { t } ^ { 2 } .$ . (6) shows that for a reward parameter θ, we have $Y _ { t } \sim \mathrm { B e r } ( \sigma ( \langle \theta , A _ { t } ^ { 1 } -$ $A _ { t } ^ { 2 } \rangle ) ) = \mathrm { B e r } ( \sigma ( \theta ^ { T } B _ { t } ) )$

## 3.1 ODPO: Optimal Design Policy

We now introduce ODPO: Optimally Designed Policy Optimization, with the pseudocode provided in Algorithm 1. The core idea behind is to work ofline using the entire dataset $\mathcal { D } _ { \mathrm { i n i } }$ The strength of optimal design techniques comes from the $K i e f e r \mathrm { - }$ Wolfowitz theorem Appendix B) to select an optimal core subset of samples. This makes it ideal for selecting $\mathcal { D } _ { \mathrm { s e l e c t } }$ without requiring any online feedback. In our approach, the approximate optimal design policy ˆπ is obtained using the Frank-Wolfe algorithm, given in Appendix B . Instead of requiring T steps of computations of the likelihood and most informative pairs, our setup only requires to run the Frank- $W o l f e$ algorithm to know which subset $\mathcal { D } _ { \mathrm { s e l e c t } }$ to select from $\mathcal { D } _ { \mathrm { i n i } }$ . Then, after the human preferences over $\mathcal { D } _ { \mathrm { s e l e c t } }$ are given, the likelihood and the MLE are only computed once.

After sampling T informative pairs $\{ A _ { t } ^ { 1 } , A _ { t } ^ { 2 } \} _ { t \in [ T ] }$ from $\mathcal { D } _ { \mathrm { i n i } }$ , ODPO constructs the maximum likelihood estimator $\hat { \theta } _ { T }$ for the regularized log-likelihood relying on $\tilde { \mathcal { D } } _ { \mathrm { s e l e c t e d } } = \{ A _ { t } ^ { 1 } , A _ { t } ^ { 2 } , Y _ { t } \} _ { t \in [ T ] }$ and the maximum likelihood estimator $\widehat { \theta } _ { T + 1 } ^ { P }$ projected on the unit ball - see (13). Then, for any context $x _ { n } , n \ \in \ [ N ]$ associated with the embedded set $A _ { n } .$ , and the estimated reward parameter $\widehat { \theta } _ { T + 1 } ^ { P }$ output by ODPO, we can estimate the best-arm in this set $\hat { a } _ { T } ( \mathcal { A } _ { n } )$ , following an optimistic procedure

$$
\hat {a} _ {T} (\mathcal {A} _ {n}) = \mathrm{argmax} _ {a \in \mathcal {A} _ {n}} \langle \hat {\theta} _ {T + 1} ^ {P}, a \rangle .\tag{15}
$$

Lemma 2. Under H1, for any $\delta \in ( 0 , 1 )$ , with proba-

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 ODPO: Optimally Designed Policy Optimization
1: Input: Number of samples T, set of actions  $B = \cup_{n \in [N]} \{a_n^i - a_n^j\}_{n \in [N], i,j \in [K]}$  of size L, regularization parameters  $\lambda$ , approximation parameter,  $\varepsilon$ .
2: Compute the history  $H_{0} = \varnothing$ , as well as t = 0,  $\hat{\theta}_{0} = \varnothing$  and  $V_{0} = \lambda I$ ,  $D_{\text{select}} = \varnothing$ .
3: Use FW to compute an  $(1 + \varepsilon)$  approximation  $\hat{\pi}$  of the optimal design  $\pi^{\star}$  with  $\mathcal{B}, \mathcal{U}(\mathcal{B}), \lambda$ .
4: for  $b \in B$  do
5: Append b to  $D_{select} \left[ T \hat{\pi}_{b} \right]$  times.
6: end for
7: Compute  $\hat{\theta}_{T+1}$  according to (9) and  $\hat{\theta}_{T+1}^{P}$  according to (13).
8: Return  $D_{select}$  and  $\hat{\theta}_{T+1}^{P}$ .
</div>

Figure 2: Note that on this figure, $a _ { 2 }$ and $a _ { 3 }$ are optimal, alghough playing both of them for the duel will provide feedback of very low value since they lie in the same region of $\mathbb { R } ^ { d }$ . This is why a duel between $a _ { 1 }$ and another arm is of greater interest in our exploration setup: sampling good arms is not the optimal strategy, hence the link with pure exploration.

bility at least $1 - \delta _ { i }$ , we have that

$$
\begin{array}{l} \| \hat {\theta} _ {t} ^ {P} - \theta^ {\star} \| _ {V _ {t}} \leqslant \\ 2 0 \left[ \sqrt {2 \log (1 / \delta) + d \log \left(\lambda^ {1 - 1 / d} + 4 t / d \lambda^ {1 / d}\right)} + \sqrt {\lambda} \right]. \end{array}
$$

Lemma 2 allows to control the gap between the true reward parameter $\theta ^ { \star }$ and the estimation $\widehat { \theta } _ { T + 1 }$ . We postpone the proof to Appendix A.

Decomposition of the Regret. Since R can be written $\begin{array} { r l r } { \Re ( T , ( { \mathcal A } _ { n } ) _ { n \in [ N ] } , \theta ^ { \star } ) } & { = } & { \langle \theta ^ { \star } , a _ { n ^ { \star } } ^ { \star } \ - \ \hat { a } _ { T + 1 } ( { \mathcal A } _ { n ^ { \star } } ) \rangle } \end{array}$ with $\begin{array} { r l r } { a _ { n } ^ { \star } } & { { } = } & { \mathrm { \ a r g m a x } _ { a \in \mathcal { A } _ { n } } \langle \theta ^ { \star } , a \rangle } \end{array}$ and $n ^ { \star } \quad = \quad$ argmax $\cdot _ { n \in [ N ] } \langle \theta ^ { \star } , a _ { n } ^ { \star } - \hat { a } _ { T + 1 } ( \mathcal { A } _ { n ^ { \star } } ) \rangle$ , we have that

$$
\begin{array}{l} \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant \langle \theta^ {\star}, a _ {n ^ {\star}} ^ {\star} - \hat {a} _ {T} (\mathcal {A} _ {n ^ {\star}}) \rangle \\ \quad - \langle \hat {\theta} _ {T + 1}, a _ {n ^ {\star}} ^ {\star} - \hat {a} _ {T} (\mathcal {A} _ {n ^ {\star}}) \rangle \\ = \langle \theta^ {\star} - \hat {\theta} _ {T + 1}, a _ {n ^ {\star}} ^ {\star} - \hat {a} _ {T} (\mathcal {A} _ {n ^ {\star}}) \rangle \\ \leqslant \| \theta^ {\star} - \hat {\theta} _ {T + 1} \| _ {V _ {T + 1}} \\ \quad \times \| a _ {n ^ {\star}} ^ {\star} - \hat {a} _ {T} (\mathcal {A} _ {n ^ {\star}}) \| _ {V _ {T + 1} ^ {- 1}} \\ \leqslant \| \theta^ {\star} - \hat {\theta} _ {T + 1} \| _ {V _ {T + 1}} \max _ {b \in \mathcal {B}} \| b \| _ {V _ {T + 1} ^ {- 1}}, \end{array} \tag {16}
$$

where the third line holds thanks to H¨older inequality. Let $\pi \colon B  [ 0 , 1 ]$ be a distribution over the set of actions. We define the application $g \colon \Delta ( B )  \mathbb { R }$ and the design matrix $\tilde { V } ( \pi )$ for the distribution π as

$$
\tilde {V} (\pi) = \sum_ {b \in \mathcal {B}} \pi (b) b b ^ {T} \text { and } g (\pi) = \max _ {b \in \mathcal {B}} \| b \| _ {\tilde {V} (\pi) ^ {- 1}} ^ {2}.\tag{17}
$$

A design π for our problem is a probability distribution over the set of actions B. An optimal design $\pi ^ { \star }$ is a solution in the Kiefer-Wolfowitz theorem while the distribution ˆπ over B is a $( 1 + \varepsilon )$ -approximation of $\pi ^ { \star } \ \mathrm { i f } \ g ( \hat { \pi } ) \leqslant ( 1 + \varepsilon ) g ( \pi ^ { \star } )$

Theorem 1. Let $\varepsilon > 0$ and suppose that we collect at least $T \geqslant d ^ { 2 }$ samples according to an $( 1 + \varepsilon )$ approximation πˆ of the optimal design policy $\pi ^ { \star } ~ f o r$ the problem. Then, for any B and $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ , with probability at least $1 - \delta , \delta \in ( 0 , 1 )$ , we have that

$$
\begin{array}{l} \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 2 0 (1 + \varepsilon) \sqrt {d / T} \times \\ \left[ \sqrt {2 \log (1 / \delta) + d \log \left(\lambda^ {1 - 1 / d} + 4 T / d \lambda^ {1 / d}\right)} + \sqrt {\lambda} \right]. \end{array} \tag {18}\tag{18}
$$

Corollary 1. Suppose that we have selected the samples $\mathcal { D } _ { \mathrm { s e l e c t } }$ to label under πˆ, a 3/2-approximation of the optimal design policy $\pi ^ { \star }$ . Choosing $\lambda = 1 / d \ f o r$ the regularization, for any $\delta \in ( 0 , 1 )$ , under the conditions of Theorem 1, with probability at least $1 - \delta$ , we have that

$$
\begin{array}{l} \mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 3 0 \sqrt {d / T} \times \\ \left[ \sqrt {2 \log (1 / \delta) + d \log ((1 + 4 T) / d ^ {1 - 1 / d})} + 1 / \sqrt {d} \right], \end{array}
$$

and as a consequence, choosing $\delta = d ^ { 1 - 1 / d } / ( 4 T + 1 )$ ， we can bound the expectation of the regret as

$$
\begin{array}{c} \mathbb {E} [ \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) ] \leqslant 3 0 \frac {d + 2}{\sqrt {T}} \sqrt {\log \left(\frac {4 T + 1}{d ^ {1 - 1 / d}}\right)} \\ + 3 1 / \sqrt {T}. \end{array}
$$

The lower bound in Section 4 matches the upper bound in Corollary 1 up to constant or logarithmic factors.

Our procedure achieves optimal selection without online feedback, simplifying dataset design and reducing computational costs, as key components like the design matrix or likelihood are computed only once.

## 3.2 Changing Action Set

Setup. The term online as opposed to ofline is not always clear in the RLHF literature. We can consider a version of our process with changing action sets: at each step $n \in [ N ]$ , where $N \in \mathbb { N } .$ a prompt $x _ { n }$ is drawn from the set of prompts $x ,$ , and the initial lan guage model $\phi _ { 0 }$ generates K completions $y _ { n } ^ { 1 } , \ldots , y _ { n } ^ { K }$ associated with $x _ { n }$ . At each iteration, from this set of K completions, two samples, $Y _ { n } ^ { ( 1 ) }$ and $Y _ { n } ^ { ( 2 ) }$ , must be selected for evaluation by a human labeler. Given that rating samples is costly, the goal is to design an algorithm that optimally selects T pairs, $( x _ { t } , Y _ { t } ^ { ( \bar { 1 } ) } )$ and $( x _ { t } , Y _ { t } ^ { ( 2 ) } )$ at each step, to form the most efective dataset before training the reward model while keeping T relatively small. Here, note that $T = N$ based on the previous notation from the ofline setting.

Objectives and metrics. As before, we are interested in the precision of the reward estimation after the T steps, which guides us towards selecting an optimal dataset $\mathcal { D } _ { \mathrm { s e l e c t } }$ . This is why we keep the same objective as before and thus consider the simple regret of the procedure, defined here as

$$
\mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta) = \max _ {n \in [ N ]} \max _ {a \in \mathcal {A} _ {n}} \left\langle \theta , a - \hat {a} (\mathcal {A} _ {n}) \right\rangle ,\tag{19}
$$

where $\hat { a } ( \mathcal { A } _ { n } ) \in \mathcal { A } _ { n }$ is the best-arm prediction of the procedure among the set ${ \mathcal { A } } _ { n }$ . Considering the same kind of objective also allows us to compare both kinds of procedures.

Formally, at each step $t \in [ T ]$ , an action set $A _ { t } , t \in [ T ]$ is provided and the algorithm must select a pair of samples $( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) \in \mathcal { A } _ { t } ^ { 2 }$ . We can define the learner’s history as $\mathcal { \dot { H } } _ { t } = \sigma ( \{ X _ { s } , A _ { s } ^ { 1 } , A _ { s } ^ { 2 } \} _ { s = 1 , \dots , t } ) , \mathcal { H } _ { 0 } = \emptyset$ and the learner uses an algorithm ALG to choose the action pair $( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } )$ based on $( \mathcal { H } _ { t - 1 } , \mathcal { A } _ { t } , U _ { t } ) , \ ( U _ { t } ) _ { t \in \mathbb { N } ^ { \star } }$ being a family of independent uniform random variables on [0, 1] allowing randomization in ALG. A commonly proposed strategy to explore is to pick the pair $( A _ { t } ^ { 1 } , \bar { A } _ { t } ^ { 2 } ) \in \bar { \mathcal { A } } _ { t } ^ { 2 }$ at each iteration following

$$
A _ {t} ^ {1}, A _ {t} ^ {2} \in \operatorname{argmax} _ {a, a ^ {\prime} \in \mathcal {A} _ {t}} \| a - a ^ {\prime} \| _ {V _ {t} ^ {- 1}}.
$$

However, such a strategy, as well as any procedure ALG for a setup with changing arms cannot converge, since we do not make i.i.d. assumptions throughout this work and allow adversarial action sets. The choice of specific action sets that prevent convergence for any algorithm is explained in the proof of Theorem 2.

Theorem 2. Consider the 2-dimensional euclidian space $S p a n ( e _ { 1 } , e _ { 2 } )$ as the whole action space. In that case, there exists a set of actions $( A _ { t } ) _ { t \in [ T ] }$ and some $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ such the regret defined in (19) for any algorithm ALG satisfies

$$
\mathfrak {R} _ {\mathrm{ALG}} (T, (\mathcal {A} _ {t}) _ {t \in [ T ]}, \theta^ {\star}) \geqslant \mathrm{e} ^ {- c} / 2,
$$

for some $c > 0$ independent of T and d.

## 3.3 Extensions of the Ofline Scenario

An online version of our setup could involve gathering the entire action set $\mathcal { D } _ { \mathrm { i n i } }$ before the learner iteratively selects samples from $\mathcal { D } _ { \mathrm { i n i } } .$ , using feedback at each step to inform the next selection.

However, it may be unrealistic to assume online feedback at each step. A more feasible approach would involve sampling a batch of pairs ofline, sending them to labelers for evaluation, and then selecting the next batch based on the feedback from the entire batch of preferences. This setup is more practical, as it allows a dynamic process where labelers provide preferences for a batch of pairs before the next batch is sampled, although they do not have to provide feedback at each step (too complex in practice). The eficiency of the method depends on the batch size - a batch of size 1 mirrors the online setting and a batch of size T, which recovers our ofline setup.

A lot of approaches in practice use pairs sampled offline, followed by training DPO (Rafailov et al., 2024) on the resulting dataset. Our ofline strategy, allowing the selection of pairs that are statistically the most informative, could significantly enhance DPO’s performance, without requiring too many additional computations. It would be interesting to investigate whether the optimal dataset selection depends on the reward modeling or the choice of the ψ-function within the particular setup of Azar et al. (2024).

Another potential sampling strategy of the dataset could be to select points based on a binary search procedure in $\mathbb { R } ^ { d }$ (Lobel et al., 2018), combined with bandit feedback. The reward parameter can be eficiently identified through cuts of the unit ball in $\mathbb { R } ^ { d }$ which is why this approach is of some interest. Finally, we believe that our work has strong connections with best-arm identification in linear bandits (Soare et al., 2014; Degenne et al., 2020), particularly where ideas from Optimal Design are applied. The challenge arises from the absence of online feedback in our setup, and one should look for relaxations in order to circumvent this issue.

## 4 Lower Bound

As before, after $N \in \mathbb { N }$ sets of prompt-completions have been sampled, we consider again $B ~ = ~ \{ a _ { n } ^ { i } ~ -$ $a _ { n } ^ { j } \bigr \} _ { n \in [ N ] , i , j \in [ K ] } = \cup _ { n \in [ N ] } \bigl \{ \mathcal { A } _ { n } - \mathcal { A } _ { n } \bigr \}$ , the set of all possible action pairs. We consider an algorithm ALG that samples $T$ pairs from this set and receives feedback $Y _ { t }$ from a labeler. For any $t \in [ T ]$ , we write $( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } )$ for the pair of actions sampled by ALG at iteration t, even though there is no ”time ordering” in our procedure. This choice corresponds to taking an action

$$
B _ {t} \in \mathcal {B} \colon B _ {t} = A _ {t} ^ {1} - A _ {t} ^ {2}.
$$

Formally, ALG selects an action $B _ { t }$ at step t and observes a logistic feedback $Y _ { t } \sim$ Ber $( \sigma ( \theta ^ { T } B _ { t } ) )$ for reward parameter θ. Based on $( Y _ { t } ) _ { t \in [ T ] }$ ALG estimates θ and then for any input $\mathcal { A } _ { n } , n \in [ N ]$ , plays some action $\hat { a } _ { T } ( \mathcal { A } _ { n } ) \in \mathcal { A } _ { n }$ . The performance is still evaluated with the simple regret

$$
\Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta) = \max _ {n \in \mathcal {A} _ {n}} \langle \theta , a _ {n} ^ {\star} - \hat {a} (\mathcal {A} _ {n}) \rangle .
$$

We write $\mathbb { P } _ { \theta }$ for the probability distribution of our linear contextual bandit instance with reward parameter θ over the whole possible action space A and $( P _ { b _ { 1 } } ^ { \theta } , \ldots , P _ { b _ { L } } ^ { \theta } )$ the probability distribution associated with the diferent pairs of arms from B with the parameter θ. $\mathbb { P } _ { \theta ^ { \prime } }$ as well as $( P _ { b _ { 1 } } ^ { \theta ^ { \prime } } , \dots , P _ { b _ { L } } ^ { \theta ^ { \prime } } )$ stand for the same objects with parameter ${ \bf { \dot { \boldsymbol { \theta } } } } ^ { \prime }$ . Within this setup, we can use the divergence bound for general spaces from Lattimore and Szepesv´ari (2020, 15.8), and write

$$
\begin{array}{l} \mathrm{D} _ {\mathrm{KL}} (\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}}) = \sum_ {t = 1} ^ {T} \mathbb {E} _ {\theta} [ \mathrm{D} _ {\mathrm{KL}} (P _ {A _ {t} ^ {1} - A _ {t} ^ {2}} ^ {\theta}, P _ {A _ {t} ^ {1} - A _ {t} ^ {2}} ^ {\theta^ {\prime}}) ] \\ = \sum_ {t = 1} ^ {T} \mathbb {E} _ {\theta} [ \mathrm{D} _ {\mathrm{KL}} (\mathrm{Ber} (\sigma (\theta^ {T} B _ {t})), \mathrm{Ber} (\sigma (\theta^ {' T} B _ {t})) ]. \end{array}\tag{20}
$$

Lemma 3. Assume that P and Q are probability measures on a measurable space $x , A$ such that P is absolutely continuous with respect to Q. Then

$$
D _ {K L} (\mathbb {P}, \mathbb {Q}) \leqslant \log (1 + D _ {\chi^ {2}} (\mathbb {P}, \mathbb {Q})) \leqslant D _ {\chi^ {2}} (\mathbb {P}, \mathbb {Q}).
$$

$I f \mathbb { P } \ll \mathbb { Q }$ does not hold, then the result is trivial.

The result of this lemma is of great help to upper bound the divergence of our Bernoulli random variables since the $\chi ^ { 2 }$ divergence is easier to use with Bernoulli distributions. The proof is postponed to appendix A.

We now present our main theorem from this section, which gives a lower bound for our setup whiwh matches our upper bound from Corollary 1 up to constant or logarithmic factors.

Theorem 3. Suppose that $d \geqslant 1 6$ and that $T \geqslant d ^ { 2 }$ For any algorithm ALG which samples T pairs from B and receives a preference feedback before outputing an action $\hat { a } ( \mathcal { A } _ { n } ) \in \mathcal { A } _ { n }$ for an input $\mathcal { A } _ { n }$ , there exists $( \mathcal { A } _ { n } ) _ { n \in [ N ] } \subseteq \mathrm { B } ( 0 , 1 )$ as well as $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ such that

$$
\Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \geqslant d \mathrm{e} ^ {- 5} / 4 \sqrt {T}.
$$

The self-concordance property of the sigmoid function (Bach, 2010) is of some importance in this bound since its properties play a role inequalities we work this. This lower bound allows us to claim that our proposed method is close from optimality. To our knowledge, our work is the first contribution with such an evidence of optimality for preferences dataset selection in RLHF.

## 5 Related Work

The extraordinary capabilities of RLHF to fine-tune large language models (Brown, 2020; Bubeck et al., 2023; Casper et al., 2023) are a reason for the growing attention in the field. The modeling of RLHF as Markov Decision Processes (Wang et al., 2023) or bandits is a common assumption and has been proposed for various goals (Zhu et al., 2024; Mehta et al., 2023). Our theoretical work involves comparing the feedback given as one action preferred to another, which is directly related to the problem of dueling bandits (Yue et al., 2012; Gabillon et al., 2012; Sui et al., 2018). Some foundations have been laid out to study preference-based and dueling bandits or RL (Pacchiano et al., 2021; Novoseller et al., 2020) while other works consider ofline RL where the learning does not result from interactions with the environment (Zhan et al., 2023). As mentioned in the latter, an issue in ofline RL is the insuficient coverage of the space with the collected data. Interestingly, this is an issue that we address here through optimal design.

A lot of works compare online to ofline RLHF (Hu et al., 2023; Tang et al., 2024; Cen et al., 2024) but here, we make the online-ofline distinction for the dataset generation before the reward modeling and policy optimization start; something that always needs to be done ofline in practice.

The human feedback in RLHF is usually given as a preference between a pair of generations associated with a same context, hence the link with with contextual dueling bandits (Dud´ık et al., 2015). Our setting considers a binary feedback, which relates it to generalized linear bandits (Filippi et al., 2010) and more precisely to logistic bandits (Lee and Oh, 2024; Lee et al., 2024; Abeille et al., 2021; Faury et al., 2020). Bengs et al. (2022) provide an interesting setup for contextual dueling bandits but do not provide all the proofs and rely on the work by Vaswani et al. (2019) to bound the distance between their estimate and the true reward vector, although the bounds of Vaswani et al. (2019) hold for an ordinary least square estimator and not a maximum likelihood estimator - a harder task because of the lack of explicit form of the estimator. Saha (2021) propose a very interesting approach to transform a contextual dueling bandit setting into a linear contextual linear bandit setting but leverage the iterative structure of the problem to do so, which is not possible in our case. Finally, Gabillon et al. (2012) give important ideas around pure exploration since it is one of the first and most important works in best-arm identification for multi-armed bandits.

A lot of empirical works have been done around the problem of active learning and optimal choice of samples for diverse and informative collection of data (Metz et al., 2023). In an online setup, Chen et al. (2024a) propose an interesting approach to improve the alignment with a reweighing of the generations to improve the collected information while some theoretical foundations have been have already been laid out by Lindner (2023); Wang et al. (2023). Our work has hope to stand at the crossroad of algorithmic foundations and practical considerations. There are already seminal works in new directions which involve working with of-policy evaluation for preference learning (Bhargava et al., 2024) or active learning for choosing teaching examples (Wang et al., 2021). The few theoretical attempts in our direction (Das et al., 2024; Ji et al., 2024) consider an active learning setting where the selection of the sample pairs is done concomitantly with the received feedback (online setup), which is an unrealistic assumption due to the practical operation of the work with human labelers. Also, they propose interesting methods but without lower bounds nor a deep theoretical analysis.

More generally, recent works consider learning from human preferences, such as Mukherjee et al. (2024) where the preference ordering over a list is learnt or Munos et al. (2023), where a Nash equilibrium is learnt. Instead of learning the preferences based on a score, they can be learnt with the data being some preference pairs, hence the link with the dueling bandit framework as some theoretical model (Yan et al., 2022).

Finally, we propose a lower bound for our setup. Such bounds already exist for dueling bandits but rely on the sequential structure of the problem and a global regret objective (Saha, 2021; Yue et al., 2012; Komiyama et al., 2015), something that we cannot do with the simple regret that we are looking for. This is why we see our problem as a logistic bandit and go back to the traditional Bretagnolle-Huber inequality (Bretagnolle and Huber, 1979) to control the bad events and obtain the lower bound for our .

## 6 Conclusion

This paper addresses the problem of selecting pairs of language model generations to present to labelers in order to maximize the information gathered from their feedback. The goal is to develop an eficient strategy for selecting which generations - or arms - should be rated to retrieve the most valuable information before fine-tuning the model. To tackle this, we build on the framework of pure exploration in linear contextual dueling bandits, a well-suited approach for the specific task that we are looking for. We operate under several key assumptions: a linear reward; the Bradley-Terry model that governs the preferences between pairs and the boundedness of the action set as well as the reward parameter.

The core of our approach lies in leveraging optimal design techniques, which allow us to strategically choose the arms. By doing so, we maximize the information gained from each comparison, making the rating process highly eficient. Furthermore, by applying information-theoretic tools, we derive a lower bound for the performance of our method. Remarkably, this lower bound matches our upper bound up to constant and logarithmic factors, thereby demonstrating the optimality of our approach.

Finally, we highlight that the techniques developed in this work are not only theoretical but also closely related to practical methods used for selecting the pairs and applying RLHF. The results suggest that our procedure can be both practical and highly efective, ofering a significant advancement in how LM generations are selected before receiving human feedback preferences.

## Acknowledgements

Funded by the European Union (ERC, Ocean, 101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

Abbasi-Yadkori, Y., P´al, D., and Szepesv´ari, C. (2011). Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24.

Abeille, M., Faury, L., and Calauz\`enes, C. (2021). Instance-wise minimax-optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence and Statistics, pages 3691–3699. PMLR.

Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. (2024). A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR.

Bach, F. (2010). Self-concordant analysis for logistic regression.

Bengs, V., Saha, A., and H¨ullermeier, E. (2022). Stochastic contextual dueling bandits under linear stochastic transitivity models. In International Conference on Machine Learning, pages 1764–1786. PMLR.

Bhargava, A., Jain, L., Kveton, B., Liu, G., and Mukherjee, S. (2024). Of-policy evaluation from logged human feedback. arXiv preprint arXiv:2406.10030.

Bıyık, E., Huynh, N., Kochenderfer, M. J., and Sadigh, D. (2020). Active preference-based gaussian process regression for reward learning. arXiv preprint arXiv:2005.02575.

Bıyık, E., Huynh, N., Kochenderfer, M. J., and Sadigh, D. (2024). Active preference-based gaussian process regression for reward learning and optimization. The International Journal of Robotics Research, 43(5):665–684.

Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345.

Bretagnolle, J. and Huber, C. (1979). Estimation des densit´es: risque minimax. Zeitschrift f¨ur Wahrscheinlichkeitstheorie und verwandte Gebiete, 47:119–137.

Brown, T. B. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. (2023). Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. (2023). Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217.

Cen, S., Mei, J., Goshvadi, K., Dai, H., Yang, T., Yang, S., Schuurmans, D., Chi, Y., and Dai, B. (2024). Value-incentivized preference optimization: A unified approach to online and ofline rlhf. arXiv preprint arXiv:2405.19320.

Chang, J. D., Shan, W., Oertell, O., Brantley, K., Misra, D., Lee, J. D., and Sun, W. (2024). Dataset reset policy optimization for rlhf. arXiv preprint arXiv:2404.08495.

Chen, L., Chen, J., Liu, C., Kirchenbauer, J., Soselia, D., Zhu, C., Goldstein, T., Zhou, T., and Huang, H. (2024a). Optune: Eficient online preference tuning. arXiv preprint arXiv:2406.07657.

Chen, Y., Tan, J., Zhang, A., Yang, Z., Sheng, L., Zhang, E., Wang, X., and Chua, T.-S. (2024b). On softmax direct preference optimization for recommendation. arXiv preprint arXiv:2406.09215.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

Das, N., Chakraborty, S., Pacchiano, A., and Chowdhury, S. R. (2024). Provably sample eficient rlhf via active preference optimization. arXiv preprint arXiv:2402.10500.

Degenne, R., M´enard, P., Shang, X., and Valko, M. (2020). Gamification of pure exploration for linear bandits. In International Conference on Machine Learning, pages 2432–2442. PMLR.

Devlin, J. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Di, Q., Jin, T., Wu, Y., Zhao, H., Farnoud, F., and Gu, Q. (2023). Variance-aware regret bounds for stochastic contextual dueling bandits. arXiv preprint arXiv:2310.00968.

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. (2024). Rlhf workflow: From

reward modeling to online rlhf. arXiv preprint arXiv:2405.07863.

Dud´ık, M., Hofmann, K., Schapire, R. E., Slivkins, A., and Zoghi, M. (2015). Contextual dueling bandits. In Conference on Learning Theory, pages 563–587. PMLR.

Faury, L., Abeille, M., Calauz\`enes, C., and Fercoq, O. (2020). Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning, pages 3052–3060. PMLR.

Filippi, S., Cappe, O., Garivier, A., and Szepesv´ari, C. (2010). Parametric bandits: The generalized linear case. Advances in neural information processing systems, 23.

Gabillon, V., Ghavamzadeh, M., and Lazaric, A. (2012). Best arm identification: A unified approach to fixed budget and fixed confidence. Advances in Neural Information Processing Systems, 25.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and Gelly, S. (2019). Parameter-eficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Hu, J., Tao, L., Yang, J., and Zhou, C. (2023). Aligning language models with ofline reinforcement learning from human feedback. arXiv preprint arXiv:2308.12050.

Ji, K., He, J., and Gu, Q. (2024). Reinforcement learning from human feedback with active queries. arXiv preprint arXiv:2402.09401.

Komiyama, J., Honda, J., Kashima, H., and Nakagawa, H. (2015). Regret lower bound and optimal algorithm in dueling bandit problem. In Conference on learning theory, pages 1141–1154. PMLR.

Lattimore, T. and Szepesv´ari, C. (2020). Bandit algorithms. Cambridge University Press.

Lee, J. and Oh, M.-h. (2024). Nearly minimax optimal regret for multinomial logistic bandit. arXiv preprint arXiv:2405.09831.

Lee, J., Yun, S.-Y., and Jun, K.-S. (2024). Improved regret bounds of (multinomial) logistic bandits via regret-to-confidence-set conversion. In International Conference on Artificial Intelligence and Statistics, pages 4474–4482. PMLR.

Lester, B., Al-Rfou, R., and Constant, N. (2021). The power of scale for parameter-eficient prompt tuning. arXiv preprint arXiv:2104.08691.

Lindner, D. (2023). Algorithmic Foundations for Safe and Eficient Reinforcement Learning from Human Feedback. PhD thesis, ETH Zurich.

Liu, Y. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Lobel, I., Leme, R. P., and Vladu, A. (2018). Multidimensional binary search for contextual decisionmaking. Operations Research, 66(5):1346–1361.

Mehta, V., Das, V., Neopane, O., Dai, Y., Bogunovic, I., Schneider, J., and Neiswanger, W. (2023). Sample eficient reinforcement learning from human feedback via active exploration.

Metz, Y., Lindner, D., Baur, R., Keim, D., and El-Assady, M. (2023). Rlhf-blender: A configurable interactive interface for learning from diverse human feedback. arXiv preprint arXiv:2308.04332.

Mukherjee, S., Lalitha, A., Kalantari, K., Deshmukh, A., Liu, G., Ma, Y., and Kveton, B. (2024). Optimal design for human feedback. arXiv preprint arXiv:2404.13895.

Munos, R., Valko, M., Calandriello, D., Azar, M. G., Rowland, M., Guo, Z. D., Tang, Y., Geist, M., Mesnard, T., Michi, A., et al. (2023). Nash learning from human feedback. arXiv preprint arXiv:2312.00886.

Naveed, H., Khan, A. U., Qiu, S., Saqib, M., Anwar, S., Usman, M., Akhtar, N., Barnes, N., and Mian, A. (2023). A comprehensive overview of large language models. arXiv preprint arXiv:2307.06435.

Novoseller, E., Wei, Y., Sui, Y., Yue, Y., and Burdick, J. (2020). Dueling posterior sampling for preferencebased reinforcement learning. In Conference on Uncertainty in Artificial Intelligence, pages 1029–1038. PMLR.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Pacchiano, A., Saha, A., and Lee, J. (2021). Dueling rl: reinforcement learning with trajectory preferences. arXiv preprint arXiv:2111.04850.

Plackett, R. L. (1975). The analysis of permutations. Journal of the Royal Statistical Society Series C: Applied Statistics, 24(2):193–202.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.

Reimers, N. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Saha, A. (2021). Optimal algorithms for stochastic contextual preference bandits. Advances in Neural Information Processing Systems, 34:30050–30062.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Shen, J. H., Sharma, A., and Qin, J. (2024). Towards data-centric rlhf: Simple metrics for preference dataset comparison. arXiv preprint arXiv:2409.09603.

Soare, M., Lazaric, A., and Munos, R. (2014). Bestarm identification in linear bandits. Advances in Neural Information Processing Systems, 27.

Steck, H., Ekanadham, C., and Kallus, N. (2024). Is cosine-similarity of embeddings really about similarity? In Companion Proceedings of the ACM on Web Conference 2024, pages 887–890.

Sui, Y., Zoghi, M., Hofmann, K., and Yue, Y. (2018). Advancements in dueling bandits. In IJCAI, pages 5502–5510.

Tang, Y., Guo, D. Z., Zheng, Z., Calandriello, D., Cao, Y., Tarassov, E., Munos, R., Pires, B. A., Valko, M.,<sup>´</sup> Cheng, Y., et al. (2024). Understanding the performance gap between online and ofline alignment algorithms. arXiv preprint arXiv:2405.08448.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi\`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. (2023). Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971.

Tucker, M., Novoseller, E., Kann, C., Sui, Y., Yue, Y., Burdick, J. W., and Ames, A. D. (2020). Preferencebased learning for exoskeleton gait optimization. In 2020 IEEE international conference on robotics and automation (ICRA), pages 2351–2357. IEEE.

Vaswani, S., Mehrabian, A., Durand, A., and Kveton, B. (2019). Old dog learns new tricks: Randomized ucb for bandit problems. arXiv preprint arXiv:1910.04928.

Wang, C., Singla, A., and Chen, Y. (2021). Teaching an active learner with contrastive examples. Advances in Neural Information Processing Systems, 34:17968–17980.

Wang, Y., Liu, Q., and Jin, C. (2023). Is rlhf more dificult than standard rl? a theoretical perspective. Advances in Neural Information Processing Systems, 36:76006–76032.

Wei, C., Wang, Y.-C., Wang, B., and Kuo, C.-C. J. (2023). An overview on language models: Recent developments and outlook. arXiv preprint arXiv:2303.05759.

Yan, X., Luo, C., Clarke, C. L., Craswell, N., Voorhees, E. M., and Castells, P. (2022). Human preferences as dueling bandits. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 567–577.

Yue, Y., Broder, J., Kleinberg, R., and Joachims, T. (2012). The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538– 1556.

Zhan, W., Uehara, M., Kallus, N., Lee, J. D., and Sun, W. (2023). Provable ofline reinforcement learning with human feedback. In ICML 2023 Workshop The Many Facets of Preference-Based Learning.

Zhao, Z., Fan, W., Li, J., Liu, Y., Mei, X., Wang, Y., Wen, Z., Wang, F., Zhao, X., Tang, J., et al. (2023). Recommender systems in the era of large language models (llms). arXiv preprint arXiv:2307.02046.

Zhu, B., Jordan, M. I., and Jiao, J. (2024). Iterative data smoothing: Mitigating reward overfitting and overoptimization in rlhf. arXiv preprint arXiv:2401.16335.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. (2019). Fine-tuning language models from human preferences. ArXiv, abs/1909.08593.

## A Proofs

Lemma 4. For any $x \in \mathbb { R } , \sigma ^ { \prime } ( x ) > 0$ and for any interval of the form $[ - \alpha , \beta ]$ for $\alpha , \beta > 0$ , we have that $\sigma ^ { \prime }$ is increasing on $[ - \alpha , 0 ]$ and decreasing on $[ 0 , \beta ]$

Proof of Lemma $\it 4 .$ Note that for any $x \in \mathbb { R }$ we have

$$
\sigma^ {\prime} (x) = e ^ {- x} / (1 + e ^ {- x}) ^ {2} \quad \mathrm{and} \quad \sigma^ {\prime \prime} (x) = e ^ {- x} (e ^ {- x} - 1) / (1 + e ^ {- x}) ^ {3},
$$

and we observe that $\sigma ^ { \prime \prime }$ cancels out in $0 ,$ is positive on $\mathbb { R } _ { - } ^ { \star }$ and negative on $\mathbb { R } _ { + } ^ { \star }$ , hence the result.

Lemma 1. We can diferentiate the likelihood defined in (8), and obtain

$$
\begin{array}{l} \nabla_ {\theta} \mathcal {L} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta) \\ = \sum_ {s = 1} ^ {t - 1} (Y _ {s} - \mathbb {P} _ {\theta} (Y _ {s} = 1)) (A _ {s} ^ {1} - A _ {s} ^ {2}) - \lambda \theta , \end{array}\tag{10}
$$

which gives by definition of the maximum likelihood estimator in (9), that $\widehat { \theta } _ { t }$ must satisfy

$$
\sum_ {s = 1} ^ {t - 1} (Y _ {s} - \mathbb {P} _ {\hat {\theta} _ {t}} (Y _ {s} = 1)) (A _ {s} ^ {1} - A _ {s} ^ {2}) - \lambda \hat {\theta} _ {t} = 0.\tag{11}
$$

Proof of Lemma 1. As mentioned in the main text, a direct computation gives that

$$
\begin{array}{l} \mathcal {L} _ {t} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta) = \sum_ {s = 1} ^ {t - 1} \bigl \{Y _ {s} \log (\sigma (\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle)) + (1 - Y _ {s}) \log (\sigma (- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle)) \bigr \} - \lambda \| \theta \| _ {2} ^ {2} / 2 \\ = \sum_ {s = 1} ^ {t - 1} \biggl \{Y _ {s} \log \biggl (\frac {1}{1 + \mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}} \biggr) + (1 - Y _ {s}) \log \biggl (\frac {1}{1 + \mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}} \biggr) \biggr \} - \lambda \| \theta \| _ {2} ^ {2} / 2, \end{array}
$$

and Lemma 4 ofers an expression for the diferential of the sigmoid, which gives that

$$
\begin{array}{r l} & {\nabla_ {\theta} \mathcal {L} _ {t} (\{A _ {s} ^ {1}, A _ {s} ^ {2}, Y _ {s} \} _ {s \in [ t - 1 ]}, \theta) = \sum_ {s = 1} ^ {t - 1} \Bigg \{Y _ {s} \frac {(A _ {s} ^ {1} - A _ {s} ^ {2}) \mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle} (1 + \mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}) ^ {- 2}}{(1 + \mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}) ^ {- 1}} \Bigg \}} \\ & {\qquad + \sum_ {s = 1} ^ {t - 1} \Bigg \{- (1 - Y _ {s}) \frac {(A _ {s} ^ {1} - A _ {s} ^ {2}) \mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle} (1 + \mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}) ^ {- 2}}{(1 + \mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}) ^ {- 1}} \Bigg \} - \lambda \theta} \\ & {\qquad = \sum_ {s = 1} ^ {t - 1} \Bigg \{Y _ {s} (A _ {s} ^ {1} - A _ {s} ^ {2}) \frac {\mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}}{1 + \mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}} + (Y _ {s} - 1) (A _ {s} ^ {1} - A _ {s} ^ {2}) \frac {\mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}}{1 + \mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}} \Bigg \} - \lambda \theta} \\ & {\qquad = \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) \Bigg \{Y _ {s} \frac {1}{1 + \mathrm{e} ^ {\langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}} + (Y _ {s} - 1) \frac {1}{1 + \mathrm{e} ^ {- \langle \theta , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle}} \Bigg \} - \lambda \theta} \\ & {\qquad = \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) \{Y _ {s} (1 - \mathbb {P} _ {\theta} (Y _ {s} = 1)) + (Y _ {s} - 1) \mathbb {P} _ {\theta} (Y _ {s} = 1) \} - \lambda \theta} \\ & {\qquad = \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) (Y _ {s} - \mathbb {P} _ {\theta} (Y _ {s} = 1)) - \lambda \theta ,} \end{array}
$$

hence the result. By definition of the maximum likelihood estimator, $\widehat { \theta } _ { t }$ satisfies $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { t } ( \{ A _ { s } ^ { 1 } , A _ { s } ^ { 2 } , Y _ { s } \} _ { s \in [ t - 1 ] } , \hat { \theta } _ { t } ) = 0 ,$ and therefore we obtain (11). □

Lemma 2. Under H1, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , we have that

$$
\begin{array}{l} \| \hat {\theta} _ {t} ^ {P} - \theta^ {\star} \| _ {V _ {t}} \leqslant \\ 2 0 \left[ \sqrt {2 \log (1 / \delta) + d \log (\lambda^ {1 - 1 / d} + 4 t / d \lambda^ {1 / d})} + \sqrt {\lambda} \right]. \end{array}
$$

Proof of Lemma 2. Our proof is inspired by results from Di et al. (2023); Ji et al. (2024). For any $t \in [ T + 1 ]$ recall the definition of $H _ { t } ;$ we now define

$$
X _ {t} = Y _ {t} - \mathbb {P} (Y _ {t} = 1) = \mathbb {1} (A _ {t} ^ {1} \succ A _ {t} ^ {2}) - \sigma (\langle \theta^ {\star}, A _ {t} ^ {1} - A _ {t} ^ {2} \rangle) \quad \mathrm{and} \quad Z _ {t} = \sum_ {s = 1} ^ {t - 1} X _ {s} (A _ {s} ^ {1} - A _ {s} ^ {2}).
$$

As we saw in (12), by definition of the maximum likelihood estimator, and $\theta ^ { \star }$ as the true reward parameter, we have that

$$
H _ {t} (\hat {\theta} _ {t}) = \sum_ {s = 1} ^ {t - 1} Y _ {s} (A _ {s} ^ {1} - A _ {s} ^ {2}) \quad \mathrm{and} \quad H _ {t} (\theta^ {\star}) = \lambda \theta^ {\star} + \sum_ {s = 1} ^ {t - 1} \mathbb {P} (Y _ {s} = 1) (A _ {s} ^ {1} - A _ {s} ^ {2}),
$$

where $\mathbb { P }$ stands for the true reward distribution, according to the parameter $\theta ^ { \star }$ . Therefore

$$
H _ {t} (\hat {\theta} _ {t}) - H _ {t} (\theta^ {\star}) = \sum_ {s = 1} ^ {t - 1} [ Y _ {s} - \mathbb {P} (Y _ {s} = 1) ] (A _ {s} ^ {1} - A _ {s} ^ {2}) - \lambda \theta^ {\star} = Z _ {t} - \lambda \theta^ {\star}.\tag{21}
$$

We now consider the diference $H _ { t } ( \theta _ { 1 } ) - H _ { t } ( \theta _ { 2 } )$ for arbitrary $\theta _ { 1 } , \theta _ { 2 } \mathrm { i n } \mathbb { R } ^ { d }$ , and apply a first order Taylor expansion with integral remainder to the function $\theta \mapsto \sigma ( \langle \theta , A _ { s } ^ { 1 } - A _ { s } ^ { 2 } \rangle )$ ) on the space $\mathbb { R } ^ { d }$ in each term of the sum, which leads to

$$
\begin{array}{l} H _ {t} (\theta_ {1}) - H _ {t} (\theta_ {2}) = \lambda \theta_ {1} - \lambda \theta_ {2} + \sum_ {s = 1} ^ {t - 1} (\sigma (\theta_ {1} ^ {T} (A _ {s} ^ {1} - A _ {s} ^ {2})) - \sigma (\theta_ {2} ^ {T} (A _ {s} ^ {1} - A _ {s} ^ {2}))) (A _ {s} ^ {1} - A _ {s} ^ {2}) \\ \qquad = \lambda (\theta_ {1} - \theta_ {2}) + \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) \int_ {u = 0} ^ {1} (A _ {s} ^ {1} - A _ {s} ^ {2}) ^ {T} \sigma^ {\prime} (\langle \theta_ {1} + u (\theta_ {2} - \theta_ {1}), A _ {s} ^ {1} - A _ {s} ^ {2} \rangle) (\theta_ {1} - \theta_ {2}) d u \\ \qquad = \left[ \lambda I + \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) (A _ {s} ^ {1} - A _ {s} ^ {2}) ^ {T} \int_ {u = 0} ^ {1} \sigma^ {\prime} (\langle \theta_ {1} + u (\theta_ {2} - \theta_ {1}), A _ {s} ^ {1} - A _ {s} ^ {2} \rangle) d u \right] (\theta_ {1} - \theta_ {2}). \end{array}
$$

We define for any $t \in [ T ] , \theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { d }$

$$
G _ {t} (\theta_ {1}, \theta_ {2}) = \lambda I + \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) (A _ {s} ^ {1} - A _ {s} ^ {2}) ^ {T} \int_ {u = 0} ^ {1} \underbrace {\sigma^ {\prime} (\langle \theta_ {2} + u (\theta_ {1} - \theta_ {2}) , A _ {s} ^ {1} - A _ {s} ^ {2} \rangle)} _ {\geqslant 0} d u \succ 0,
$$

since $\lambda$ is chosen such that $\lambda > 0$ . Note that for any $t , \theta _ { 1 } , \theta _ { 2 } \colon G _ { t } ( \theta _ { 1 } , \theta _ { 2 } )$ is symmetric. By definition, we have that for any $\theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { d }$

$$
H _ {t} (\theta_ {1}) - H _ {t} (\theta_ {2}) = G _ {t} (\theta_ {1}, \theta_ {2}) (\theta_ {1} - \theta_ {2}),
$$

which gives that

$$
\| \theta_ {1} - \theta_ {2} \| _ {G _ {t} (\theta_ {1}, \theta_ {2})} = \sqrt {(\theta_ {1} , \theta_ {2}) G _ {t} G _ {t} ^ {- 1} G _ {t} (\theta_ {1} , \theta_ {2})} = \| H _ {t} (\theta_ {1}) - H _ {t} (\theta_ {2}) \| _ {G _ {t} ^ {- 1} (\theta_ {1}, \theta_ {2})}.\tag{22}
$$

Using Lemma 4, since $\sigma ^ { \prime } ( - 2 ) \geqslant 0 . 1$ and $\sigma ^ { \prime } ( 2 ) \geqslant 0 . 1$ , we have that for any $x \in [ - 2 , 2 ] , \sigma ^ { \prime } ( x ) \geqslant 0 . 1$ . Note that for any $s \in [ T ] , \theta \in \mathrm { B } ( 0 , 1 ) , \langle \theta , A _ { s } ^ { 1 } - A _ { s } ^ { 2 } \rangle \in [ - 2 , 2 ]$ . Therefore, under $\mathbf { H } 1$ , a convexity argument gives that for any $\theta \in \mathrm { B } ( 0 , 1 ) , u \in [ 0 , 1 ] , \left. \theta + u ( \theta ^ { \star } - \theta ) , A _ { s } ^ { 1 } - A _ { s } ^ { 2 } \right. \in [ - 2 , 2 ]$ , and we obtain

$$
\int_ {u = 0} ^ {1} \sigma^ {\prime} (\langle \hat {\theta} _ {t} ^ {P} + u (\theta^ {\star} - \hat {\theta} _ {t} ^ {P}), A _ {s} ^ {1} - A _ {s} ^ {2} \rangle) d u \geqslant 0. 1,
$$

since $\hat { \theta } _ { t } ^ { P } \in \mathrm { B } ( 0 , 1 )$ . Note that we use here the fact that $\widehat { \theta } _ { t } ^ { P }$ lies in the unit ball to control the boundedness of the sigmoid function. This leads to

$$
V _ {t} \prec 1 0 G _ {t} (\theta^ {\star}, \hat {\theta} _ {t} ^ {P}) \quad \mathrm{and} \quad G _ {t} (\theta^ {\star}, \hat {\theta} _ {t} ^ {P}) ^ {- 1} \preceq 1 0 \Bigg (\lambda I + \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) (A _ {s} ^ {1} - A _ {s} ^ {2}) ^ {T} \Bigg) ^ {- 1} = 1 0 V _ {t} ^ {- 1},\tag{23}
$$

and we obtain

$$
\begin{array}{l} \| \theta^ {\star} - \hat {\theta} _ {t} ^ {P} \| _ {V _ {t}} \leqslant \sqrt {1 0} \| \theta^ {\star} - \hat {\theta} _ {t} ^ {P} \| _ {G _ {t} (\theta^ {\star}, \hat {\theta} _ {t} ^ {P})} \\ \quad = \sqrt {1 0} \| H _ {t} (\theta^ {\star}) - H _ {t} (\hat {\theta} _ {t} ^ {P}) \| _ {G _ {t} ^ {- 1} (\theta^ {\star}, \hat {\theta} _ {t} ^ {P})} \\ \quad \leqslant 1 0 \| H _ {t} (\hat {\theta} _ {t} ^ {P}) - H _ {t} (\theta^ {\star}) \| _ {V _ {t} ^ {- 1}} \\ \quad \leqslant 1 0 (\| H _ {t} (\hat {\theta} _ {t}) - H _ {t} (\theta^ {\star}) \| _ {V _ {t} ^ {- 1}} + \| H _ {t} (\hat {\theta} _ {t} ^ {P}) - H _ {t} (\hat {\theta} _ {t}) \| _ {V _ {t} ^ {- 1}}) \\ \quad \leqslant 2 0 \| H _ {t} (\hat {\theta} _ {t}) - H _ {t} (\theta^ {\star}) \| _ {V _ {t} ^ {- 1}} \\ \quad = 2 0 \| Z _ {t} - \lambda \theta^ {\star} \| _ {V _ {t} ^ {- 1}} \\ \quad \leqslant 2 0 (\| Z _ {t} \| _ {V _ {t} ^ {- 1}} + \sqrt {\lambda}), \end{array}
$$

where the second line holds by (22), the third line by Equation (23), the fourth by the triangular inequality, the fifth by definition of $\hat { \theta } _ { t } ^ { P } \ ( 1 3 )$ and the penultimate by (21). Note that for any $t \in [ T ] , X _ { t } \in [ - 1 , 1 ]$ and is therefore 1-subgaussian by Hoefding inequality. Therefore, we apply Abbasi-Yadkori et al. (2011, Theorem 1), which gives that with probability at least $1 - \delta$ , we have

$$
\left\| Z _ {t} \right\| _ {V _ {t} ^ {- 1}} ^ {2} = \left\| \sum_ {s = 1} ^ {t - 1} X _ {s} (A _ {s} ^ {1} - A _ {s} ^ {2}) \right\| _ {V _ {t} ^ {- 1}} ^ {2} \leqslant 2 \log \Big (\sqrt {\det (V _ {t})} / (\sqrt {\det (V _ {0})} \delta) \Big) \leqslant 2 \log (1 / \delta) + \log (\det V _ {t} / \det V _ {0}).
$$

By definition, we have that for any $t \geqslant 1$

$$
V _ {t} = V _ {0} + \sum_ {s = 1} ^ {t - 1} (A _ {s} ^ {1} - A _ {s} ^ {2}) (A _ {s} ^ {1} - A _ {s} ^ {2}) ^ {T},
$$

and now using Lattimore and Szepesv´ari (2020, Lemma 19.4), we can write

$$
\log (\det V _ {t} / \det V _ {0}) \leqslant d \log \left(\frac {\operatorname{Tr} (V _ {0}) + 4 (t - 1)}{d \det (V _ {0}) ^ {1 / d}}\right) = d \log \left(\frac {d \lambda + 4 (t - 1)}{d \lambda^ {1 / d}}\right).
$$

We finally obtain that with probability at least $1 - \delta$

$$
\| \hat {\theta} _ {t} ^ {P} - \theta^ {\star} \| _ {V _ {t}} \leqslant 2 0 \left[ \sqrt {2 \log (1 / \delta) + d \log (\lambda^ {1 - 1 / d} + 4 t / d \lambda^ {1 / d})} + \sqrt {\lambda} \right],
$$

hence the result.

Theorem 1. Let $\varepsilon > 0$ and suppose that we collect at least $T \geqslant d ^ { 2 }$ samples according to an $( 1 + \varepsilon )$ approximation ${ \hat { \pi } } \ O f$ the optimal design policy $\pi ^ { \star }$ for the problem. Then, for any $\boldsymbol { B }$ and $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ , with probability at least $1 - \delta , \delta \in ( 0 , 1 )$ , we have that

$$
\begin{array}{l} \mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 2 0 (1 + \varepsilon) \sqrt {d / T} \times \\ \left[ \sqrt {2 \log (1 / \delta) + d \log (\lambda^ {1 - 1 / d} + 4 T / d \lambda^ {1 / d})} + \sqrt {\lambda} \right]. \end{array}\tag{18}
$$

Proof of Theorem 1. The condition $T \geqslant d ( d + 1 ) / 2$ ensures that we collect enough points so that the optimal design policy $\hat { \pi }$ satisfies the results from Theorem 4. Using the decomposition of the regret (16) and Lemma 2, we obtain that with a probability at least $1 - \delta$

$$
\mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 2 0 \left[ \sqrt {2 \log (1 / \delta) + d \log \left(\lambda^ {1 - 1 / d} + 4 T / d \lambda^ {1 / d}\right)} + \sqrt {\lambda} \right] \max _ {(a, a ^ {\prime}) \in \mathcal {D} _ {\mathrm{ini}} ^ {2}} \| a - a ^ {\prime} \| _ {V _ {T + 1} ^ {- 1}}.\tag{24}
$$

Let ˆπ be an $1 + \varepsilon$ approximation of the optimal design policy $\pi ^ { \star }$ . For any distribution $\pi ,$ we have $\tilde { V } ( \pi ) =$ $\textstyle \sum _ { b \in B } \pi ( b ) b b ^ { T }$ The regularized design matrix based on the collected samples from ˆπ is defined as $V _ { T + 1 } ~ =$ $\begin{array} { r } { \lambda I + \sum _ { t = 1 } ^ { T } B ( t ) B ( t ) ^ { T } = \lambda I + \sum _ { b \in B } \lceil T \hat { \pi } _ { b } \rceil b b ^ { T } } \end{array}$ with $B ( t ) = A _ { t } ^ { 1 } - A _ { t } ^ { 2 }$ , since the samples $( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) _ { t \in [ T ] }$ are chosen according to ODPO. Therefore, for any $b \in B$ , we have that

$$
\begin{array}{l} \| b \| _ {V _ {T + 1} ^ {- 1}} ^ {2} = b ^ {T} \Bigg (\lambda I + \sum_ {\tilde {b} \in \mathcal {B}} [ T \hat {\pi} _ {\tilde {b}} ] \tilde {b} \tilde {b} ^ {T} \Bigg) ^ {- 1} b \\ \leqslant b \Bigg (\sum_ {\tilde {b} \in \mathcal {B}} T \hat {\pi} _ {\tilde {b}} \tilde {b} \tilde {b} ^ {T} \Bigg) ^ {- 1} b \\ = \frac {1}{T} b ^ {T} \Bigg (\sum_ {\tilde {b} \in \mathcal {B}} \hat {\pi} _ {\tilde {b}} \tilde {b} \tilde {b} ^ {T} \Bigg) ^ {- 1} b \\ = \frac {1}{T} \| b \| _ {\tilde {V} ^ {- 1} (\hat {\pi})} ^ {2} \\ = (1 + \varepsilon) d / T, \end{array}
$$

where the last line holds thanks to Algorithm 2 and results on its convergence $( \mathrm { s e e } , \mathrm { e . g . }$ , Lattimore and Szepesv´ari, 2020, 21.2). It gives that

$$
\max _ {(a, a ^ {\prime}) \in \mathcal {D} _ {\mathrm{ini}}} \| a - a ^ {\prime} \| _ {V _ {T + 1} ^ {- 1}} \leqslant \sqrt {(1 + \varepsilon) d / T},
$$

and plugging this bound in (24), we obtain the result.

Corollary 1. Suppose that we have selected the samples $\mathcal { D } _ { \mathrm { s e l e c t } }$ to label under $\hat { \pi } , \ a \ 3 / 2$ -approximation of the optimal design policy $\pi ^ { \star }$ . Choosing $\lambda = 1 / d$ for the regularization, for any $\delta \in ( 0 , 1 )$ , under the conditions of Theorem 1, with probability at least $1 - \delta$ , we have that

$$
\begin{array}{l} \mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 3 0 \sqrt {d / T} \times \\ \left[ \sqrt {2 \log (1 / \delta) + d \log ((1 + 4 T) / d ^ {1 - 1 / d})} + 1 / \sqrt {d} \right], \end{array}
$$

and as a consequence, choosing $\delta = d ^ { 1 - 1 / d } / ( 4 T + 1 )$ , we can bound the expectation of the regret as

$$
\begin{array}{c} \mathbb {E} [ \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) ] \leqslant 3 0 \frac {d + 2}{\sqrt {T}} \sqrt {\log \left(\frac {4 T + 1}{d ^ {1 - 1 / d}}\right)} \\ + 3 1 / \sqrt {T}. \end{array}
$$

Proof of Corollary 1. By Equation (18), we have that for any $\delta \in ( 0 , 1 )$

$$
\Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 2 0 (1 + \varepsilon) \sqrt {d / T} \bigg [ \sqrt {2 \log (1 / \delta) + d \log \bigl (\lambda^ {1 - 1 / d} + 4 T / d \lambda^ {1 / d} \bigr)} + \sqrt {\lambda} \bigg ],\tag{25}
$$

and we now choose $\hat { \pi }$ to be a $3 / 2 .$ -approximation of the optimal policy $\pi ^ { \star }$ as well as $\lambda = 1 / d$ . Plugging these quantities in (25) gives

$$
\begin{array}{c} \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant 2 0 \times 3 / 2 \times \sqrt {d / T} \times \left[ \sqrt {2 \log (1 / \delta) + d \log \bigl (d ^ {1 / d - 1} + 4 T   d ^ {1 / d} / d \bigr)} + 1 / \sqrt {d} \right] \\ = 3 0 \sqrt {d / T} \times \left[ \sqrt {2 \log (1 / \delta) + d \log ((1 + 4 T) / d ^ {1 - 1 / d})} + 1 / \sqrt {d} \right], \end{array}
$$

hence the first part of the corollary. Observe that since $\begin{array} { r l r } { \Re ( T , ( \mathcal { A } _ { n } ) _ { n \in [ N ] } , \theta ^ { \star } ) } & { { } } & { = } \end{array}$ max $\begin{array} { r l } & { \cdot _ { n \in [ N ] } \operatorname* { m a x } _ { a _ { n } ^ { \star } \in A _ { n } } \langle \theta ^ { \star } , a _ { n } ^ { \star } - \hat { a } _ { T } ( \mathcal { A } _ { n } ) \rangle } \end{array}$ , we have that $\Re ( T , ( { \mathcal { A } } _ { n } ) _ { n \in [ N ] } , \theta ^ { \star } ) \ \leqslant \ 2$ under H 1. Therefore, for any $\mathrm { C } > 0$ , we can write

$$
\mathbb {E} \big [ \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \big ] \leqslant \mathbb {P} \big (\Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant C \big) \cdot C + 2 \cdot (1 - \mathbb {P} \big (\Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \leqslant C \big)) ,\tag{26}
$$

and we now apply (26) with $\begin{array} { r }  \mathrm { ~  ~ C ~ } = ~ 3 0 \sqrt { d / T } ~ \times ~ \left[ \sqrt { 2 \log ( 1 / \delta ) + d \log ( ( 1 + 4 T ) / d ^ { 1 - 1 / d } ) } + 1 / \sqrt { d } \right] ~ \mathrm { a n d } ~ \delta ~ = ~ 1 8 0 ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot } ~ \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 1 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 1 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~ \scriptstyle ~ C ~ } _ { 0 } ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } = ~ \mathrm { ~  ~ \cdot ~ } \mathrm { ~  ~ \scriptstyle ~ C ~ } _ { 0 } ^  \mathrm \end{array}$ $d ^ { 1 - 1 / d } / ( 4 T + 1 )$ . The first part of the corollary that we already proved gives

$$
\begin{array}{l} \mathbb {E} \left[ \Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \right] \leqslant (1 - d ^ {1 - 1 / d} / (4 T + 1)) \times 3 0 \sqrt {d / T} \times \left[ \sqrt {2 \log ((4 T + 1) / d ^ {1 - 1 / d}) + d \log ((1 + 4 T) / d ^ {1 - 1 / d})} + 1 / \sqrt {d} \right] \\ \quad + 2 \times d ^ {1 - 1 / d} / (4 T + 1) \\ \leqslant 3 0 \sqrt {d / T} \left[ \sqrt {(d + 2) \log \left(\frac {4 T + 1}{d ^ {1 - 1 / d}}\right)} + 1 / \sqrt {d} \right] + d / 2 T \\ \leqslant 3 0 \frac {d + 2}{\sqrt {T}} \sqrt {\log \left(\frac {4 T + 1}{d ^ {1 - 1 / d}}\right)} + 3 0 / \sqrt {T} + \frac {d}{2 T} \\ \leqslant 3 0 \frac {d + 2}{\sqrt {T}} \sqrt {\log \left(\frac {4 T + 1}{d ^ {1 - 1 / d}}\right)} + 3 1 / \sqrt {T}, \end{array}
$$

where the last line holds since $T \geqslant d ( d + 1 ) / 2$ . Hence the second part of the result.

Theorem 2. Consider the 2-dimensional euclidian space $S p a n ( e _ { 1 } , e _ { 2 } )$ as the whole action space. In that case, there exists a set of actions $( A _ { t } ) _ { t \in [ T ] }$ and some $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ such the regret defined in (19) for any algorithm ALG satisfies

$$
\mathfrak {R} _ {\mathrm{ALG}} (T, (\mathcal {A} _ {t}) _ {t \in [ T ]}, \theta^ {\star}) \geqslant \mathrm{e} ^ {- c} / 2,
$$

for some $c > 0$ independent of T and d.

Proof of Theorem 2. Consider the space $\mathbb { R } ^ { 2 }$ with an orthonormal basis $( e _ { 1 } , e _ { 2 } )$ . Suppose that $\theta ^ { \star } \in \Theta = \{ \pm e _ { 2 } \}$ and that the action sets are $\mathcal { A } _ { 1 } = . . . = . A _ { T - 1 } = \{ \pm e _ { 1 } \}$ and $\mathcal { A } _ { T } = \{ \pm e _ { 2 } \}$ . After sampling data and preferences, ALG outputs an action ˆa and is then evaluated with the simple regret

$$
\mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta) = \max _ {t \in [ T ]} \max _ {a \in \mathcal {A} _ {t}} \langle \theta , a - \hat {a} (\mathcal {A} _ {t}) \rangle ,
$$

and we consider that ALG sampled arms $\{ ( A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) \} _ { t \in [ T ] }$ . We write $A _ { t } ^ { 1 } - A _ { t } ^ { 2 } = B _ { t }$ . For any $t \in [ T - 1 ] , B _ { t } \in$ $\mathrm { S p a n } ( e _ { 1 } )$ . Now consider the events $\{ \hat { a } = e _ { 2 } \}$ and $\{ \hat { a } = - e _ { 2 } \}$ . We write $\theta \ : = \ : e _ { 2 }$ and $\theta ^ { \prime } = - e _ { 2 }$ . We have $\begin{array} { r } { \operatorname* { m a x } _ { a \in \mathrm { B } _ { 2 } ( 0 , 1 ) } \langle \theta , a \rangle = \operatorname* { m a x } _ { a \in \mathrm { B } _ { 2 } ( 0 , 1 ) } \langle \theta ^ { \prime } , a \rangle = 1 } \end{array}$ . We now use the Bretagnolle-Huber inequality and write

$$
\mathbb {P} _ {\theta} \left(\left\{\hat {a} = - e _ {2} \right\}\right) + \mathbb {P} _ {\theta^ {\prime}} \left(\left\{\hat {a} = e _ {2} \right\}\right) = \mathbb {P} _ {\theta} \left(\left\{\hat {a} = - e _ {2} \right\}\right) + \mathbb {P} _ {\theta^ {\prime}} \left(\left\{\hat {a} = - e _ {2} \right\} ^ {c}\right) \geqslant \exp \left(- D _ {K L} \left(\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}}\right)\right) / 2,\tag{27}
$$

as well as Lattimore and Szepesv´ari (2020, 15.8) to obtain

$$
\begin{array}{l} \mathrm{D} _ {\mathrm{KL}} (\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}}) = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathrm{D} _ {\mathrm{KL}} (P _ {B _ {t}} ^ {\theta}, P _ {B _ {t}} ^ {\theta^ {\prime}}) \right] \\ \qquad = \mathbb {E} _ {\theta} \left[ \sum_ {t = 1} ^ {T - 1} \mathrm{D} _ {\mathrm{KL}} (\mathrm{Ber} (\sigma (\theta^ {T} B _ {t})), \mathrm{Ber} (\sigma (\theta^ {\prime T} B _ {t}))) \right] + \mathbb {E} _ {\theta} [ \mathrm{D} _ {\mathrm{KL}} (\mathrm{Ber} (\sigma (\theta^ {T} B _ {T})), \mathrm{Ber} (\sigma (\theta^ {\prime T} B _ {T}))) ] \\ \qquad = \mathbb {E} _ {\theta} [ \mathrm{D} _ {\mathrm{KL}} (\mathrm{Ber} (\sigma (\theta^ {T} B _ {T})), \mathrm{Ber} (\sigma (\theta^ {\prime T} B _ {T}))) ], \end{array}
$$

since $\theta , \theta ^ { \prime } \ \in \ \mathrm { S p a n } ( e _ { 2 } ) , B _ { t } \ \in \ \mathrm { S p a n } ( e _ { 1 } )$ for any $t \ \in \ [ T \ - \ 1 ]$ , which gives $\theta ^ { \prime T } B _ { t } ~ = ~ \theta ^ { T } B _ { t } ~ = ~ 0$ Therefore $\mathrm { D } _ { \mathrm { K L } } ( \mathrm { B e r } ( \sigma ( \theta ^ { T } B _ { t } ) )$ $\mathrm { B e r } ( \sigma ( \theta ^ { \prime T } B _ { t } ) ) ) = \mathrm { D } _ { \mathrm { K L } } ( \mathrm { B e r } ( 1 / 2 ) , \mathrm { B e r } ( 1 / 2 ) ) = 0$ for any $t \in [ T - 1 ] .$ There exists a constant $c > 0$ independent of T and the dimension such that $\mathrm { D } _ { \mathrm { K L } } ( \mathrm { B e r } ( \sigma ( \theta ^ { T } B _ { T } ) )$ ), Ber $( \sigma ( \theta ^ { \prime T } B _ { T } ) ) ) \leqslant c \mathrm { . }$ . Thus, at least one of the terms in the left-hand side of (27) is bigger than exp(−c)/4 - say $\mathbb { P } _ { \theta ^ { \prime } } ( \{ \hat { a } = e _ { 2 } \} ) \geqslant \exp ( - c ) / 4$ Which one being bigger than $\exp ( - c ) / 4$ does not matter by symmetry. The regret incurred under $\theta ^ { \prime }$ for $\{ \hat { a } = e _ { 2 } \}$ holding is 2 and we finally obtain that

$$
\mathfrak {R} _ {\mathrm{ALG}} (T, (\mathcal {A} _ {t}) _ {t \in [ T ]}, \theta^ {\prime}) \geqslant \mathrm{e} ^ {- c} / 2.
$$

Lemma 3. Assume that P and Q are probability measures on a measurable space $x , A$ such that $\mathbb { P }$ is absolutely continuous with respect to $\mathbb { Q }$ . Then

$$
D _ {K L} (\mathbb {P}, \mathbb {Q}) \leqslant \log (1 + D _ {\chi^ {2}} (\mathbb {P}, \mathbb {Q})) \leqslant D _ {\chi^ {2}} (\mathbb {P}, \mathbb {Q}).
$$

$I f \mathbb { P } \ll \mathbb { Q }$ does not hold, then the result is trivial.

Proof of Lemma 3. $\mathrm { B y }$ definition of the KL-divergence, we can write

$$
\mathrm{D} _ {\mathrm{KL}} (\mathbb {P}, \mathbb {Q}) = \int_ {\mathcal {X}} \log \left(\frac {d \mathbb {P}}{d \mathbb {Q}}\right) d \mathbb {P},
$$

and applying Jensen’s inequality with the logarithm, we obtain

$$
\mathrm{D} _ {\mathrm{KL}} (\mathbb {P}, \mathbb {Q}) \leqslant \log \left(\int_ {\mathcal {X}} \frac {d \mathbb {P}}{d \mathbb {Q}} d \mathbb {P}\right) = \log \left(\int_ {\mathcal {X}} \left(\frac {d \mathbb {P}}{d \mathbb {Q}}\right) ^ {2} d \mathbb {Q}\right) = \log \left(\int_ {\mathcal {X}} \left(\frac {d \mathbb {P}}{d \mathbb {Q}}\right) ^ {2} d \mathbb {Q} - 1 + 1\right) = \log (\mathrm{D} _ {\chi^ {2}} + 1).
$$

Finally, using the inequality log $( 1 + x ) \leqslant x$ for any $x > - 1$ allows us to conclude.

Theorem 3. Suppose that $d \geqslant 1 6$ and that $T \geqslant d ^ { 2 }$ . For any algorithm ALG which samples T pairs from B and receives a preference feedback before outputing an action $\hat { a } ( \mathcal { A } _ { n } ) \in \mathcal { A } _ { n }$ for an input $A _ { n }$ , there exists $( \mathcal { A } _ { n } ) _ { n \in [ N ] } \subseteq \mathrm { B } ( 0 , 1 )$ as well as $\theta ^ { \star } \in \mathrm { B } ( 0 , 1 )$ such that

$$
\Re (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \geqslant d \mathrm{e} ^ {- 5} / 4 \sqrt {T}.
$$

Proof of Theorem 3. We first restrict θ to belong to the set $\Theta = \{ \pm \sqrt { d / T } \} ^ { d } \subseteq \mathrm { B } ( 0 , 1 )$ . Let $i \in [ d ]$ and $\theta , \theta ^ { \prime } \in \Theta$ such that for any $j \in [ d ] , j \neq i , \theta _ { j } = \theta _ { j } ^ { \prime }$ and $\theta _ { i } ^ { \prime } = - \theta _ { i }$ . For any prediction ˆa output by ALG, we define the event

$$
A _ {i, \theta} = \left\{\operatorname{sgn} (\hat {a} _ {i}) = - \operatorname{sgn} (\theta_ {i}) \right\},
$$

as well as the corresponding probability

$$
p (\theta , i) = \mathbb {P} _ {\theta} (A _ {i, \theta}) = \mathbb {P} _ {\theta} (\{\mathrm{sgn} (\hat {a} _ {i}) = - \mathrm{sgn} (\theta_ {i}) \}).
$$

Consider the action set $\pmb { \mathscr { A } } = [ \pm 1 / \sqrt { d } ] ^ { d }$ . Note that $A _ { i , \theta } ^ { \mathrm { c } } = \left\{ \mathrm { s g n } ( { \hat { a } } _ { i } ) = \mathrm { s g n } ( \theta _ { i } ) \right\} = \left\{ \mathrm { s g n } ( { \hat { a } } _ { i } ) = - \mathrm { s g n } ( \theta _ { i } ^ { \prime } ) \right\}$ . We now apply Bretagnolle-Huber’s inequality to obtain

$$
\mathbb {P} _ {\theta} (A _ {i, \theta}) + \mathbb {P} _ {\theta^ {\prime}} (A _ {i, \theta} ^ {\mathrm{c}}) \geqslant \exp (- \mathrm{D} _ {\mathrm{KL}} (\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}})) / 2.\tag{28}
$$

Now using the expression of the divergence from (20) as well as Lemma 3, we can write

$$
\mathrm{D} _ {\mathrm{KL}} \left(\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}}\right) \leqslant \sum_ {t = 1} ^ {T} \mathbb {E} _ {\theta} \left[ \mathrm{D} _ {\chi^ {2}} \left(\operatorname{Ber} \left(\sigma \left(\theta^ {T} B _ {t}\right)\right), \operatorname{Ber} \left(\sigma \left(\theta^ {\prime T} B _ {t}\right)\right)\right) \right].
$$

Since $\mathrm { D } _ { \chi ^ { 2 } } ( \mathrm { B e r } ( p ) , \mathrm { B e r } ( q ) ) = ( p - q ) ^ { 2 } / q ^ { 2 }$ , we can write

$$
\mathrm{D} _ {\mathrm{KL}} (\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}}) \leqslant \sum_ {t = 1} ^ {T} \mathbb {E} _ {\theta} \Big [ (\sigma (\theta^ {T} B _ {t}) - \sigma (\theta^ {' T} B _ {t})) ^ {2} / \sigma (\theta^ {' T} B _ {t}) (1 - \sigma (\theta^ {' T} B _ {t})) \Big ] .
$$

For any $x \ \in \ \mathbb { R } ^ { d } , \sigma ( x ) \ = \ 1 / ( 1 + \mathrm { e } ^ { - x } )$ , which gives that $1 / \sigma ( x ) ( 1 - \sigma ( x ) ) = \mathrm { e } ^ { x } ( 1 + \mathrm { e } ^ { - x } ) ^ { 2 }$ and we define $f \colon \mathbb { R } \to \mathbb { R } , x \mapsto \mathrm { e } ^ { x } ( 1 + \mathrm { e } ^ { - x } ) ^ { 2 }$ . A derivation shows that $f ^ { \prime }$ cancels out in 0, is negative on R<sub>−</sub> and positive on $\mathbb { R } _ { + }$ Therefore, for any $x \in [ - 1 / 2 , 1 / 2 ] , f ( x ) \leqslant \operatorname* { m a x } \{ f ( - 1 / 2 ) , f ( 1 / 2 ) \} \leqslant 5 .$

We now define $g \colon [ 0 , 1 ] \ \to \ \mathbb { R } , v \ \mapsto \ \sigma ( \theta ^ { ' T } B _ { t } + v ( \theta - \theta ^ { ' T } ) ^ { T } B _ { t } )$ . We have that $\begin{array} { r } { g ( 1 ) \ = \ \sigma ( \theta ^ { T } B _ { t } ) } \end{array}$ while $g ( 0 ) = \sigma ( \boldsymbol { \theta } ^ { ' T } B _ { t } )$ , which allows us to write

$$
\begin{array}{r l} & {\sigma (\theta^ {T} B _ {t}) - \sigma (\theta^ {' T} B _ {t}) = g (1) - g (0)} \\ & {\qquad = \int_ {u = 0} ^ {1} (\theta - \theta^ {'}) ^ {T} B _ {t} \sigma^ {\prime} (\theta^ {' T} B _ {t} + v (\theta - \theta^ {' T}) ^ {T} B _ {t}) \mathrm{d} v} \\ & {\qquad = \int_ {u = 0} ^ {1} \sigma^ {\prime} (\theta^ {' T} B _ {t} + v (\theta - \theta^ {' T}) ^ {T} B _ {t}) \mathrm{d} v (\theta - \theta^ {'}) ^ {T} B _ {t}.} \end{array}
$$

As we showed in the proof of Theorem 1, we have that for any $x \in [ - 2 , 2 ] , \sigma ^ { \prime } ( x ) \leqslant \sigma ^ { \prime } ( 0 ) = 1 / 4$ and therefore, we obtain

$$
\sigma (\theta^ {T} B _ {t}) - \sigma (\theta^ {' T} B _ {t}) \leqslant (\theta - \theta^ {'}) ^ {T} B _ {t} / 4.
$$

Plugging the diferent inequalities together gives that

$$
\begin{array}{l} \mathrm{D} _ {\mathrm{KL}} (\mathbb {P} _ {\theta}, \mathbb {P} _ {\theta^ {\prime}}) \leqslant \sum_ {t = 1} ^ {T} \mathbb {E} _ {\theta} \Big [ 5 ((\theta - \theta^ {'}) ^ {T} B _ {t}) ^ {2} / 1 6 \Big ] \\ \qquad = 5 / 1 6 \sum_ {t = 1} ^ {T} \mathbb {E} _ {\theta} \Big [ ((\theta - \theta^ {'}) ^ {T} B _ {t}) ^ {2} \Big ] \\ \qquad = 5 / 1 6 \sum_ {t = 1} ^ {T} 1 6 \theta_ {i} ^ {2} / d \\ \qquad = 5, \end{array}
$$

where the last penultimate line holds since $\| B _ { t } \| _ { \infty } ^ { 2 } \leqslant 4 / d$ because of $\pmb { \mathscr { A } } \subseteq [ \pm 1 / \sqrt { d } ] ^ { d }$ and $\theta - \theta ^ { \prime } = 2 \theta _ { i } e _ { i }$ where $e _ { i }$ stands for the i-th basis vector. The last line holds since $\theta _ { i } \in \{ \pm \sqrt { d / T } \}$ . Finally, plugging this inequality in (28) gives that

$$
p (\theta , i) + p (\theta^ {\prime}, i) \geqslant \mathrm{e} ^ {- 5} / 2.\tag{29}
$$

We now apply the ”averaging hammer” technique, which consists in summing all the $p ( \theta , i )$ for $\theta \in \Theta , i \in [ d ]$ and group the term that difer in only one coordinate. It gives that

$$
\sum_ {\theta \in \Theta} 1 / | \Theta | \sum_ {i = 1} ^ {d} p (\theta , i) = 1 / | \Theta | \sum_ {i = 1} ^ {d} \sum_ {\theta \in \Theta} p (\theta , i),
$$

and we reckon that $2 ^ { d - 1 }$ pairs appear as in (29), which gives that

$$
\sum_ {\theta \in \Theta} 1 / | \Theta | \sum_ {i = 1} ^ {d} p (\theta , i) \geqslant 1 / | \Theta | \sum_ {i = 1} ^ {d} 2 ^ {d - 1} \mathrm{e} ^ {- 5} / 2 = d \mathrm{e} ^ {- 5} / 4,
$$

since $\mathrm { C a r d } ( \Theta ) = 2 ^ { d }$ (hypercube). Therefore, there exists at least on $\theta ^ { \star } \in \Theta$ such that ${ \textstyle \sum _ { i = 1 } ^ { d } p ( \theta , i ) \geq d \mathrm { e } ^ { - 5 } / 4 }$ Still considering the action set $\mathcal { A } = [ - 1 / \sqrt { d } , 1 / \sqrt { d } ] ^ { d } \subseteq \mathrm { B } ( 0 , 1 )$ , we can lower bound the regret

$$
\begin{array}{l} \mathfrak {R} (T, (\mathcal {A} _ {n}) _ {n \in [ N ]}, \theta^ {\star}) \geqslant \mathbb {E} _ {\theta^ {\star}} \left[ \sum_ {i = 1} ^ {d} (\mathrm{sgn} (\theta_ {i} ^ {\star}) / \sqrt {d} - \hat {a} _ {i}) \theta_ {i} ^ {\star} \right] \\ \quad \geqslant \sum_ {i = 1} ^ {d} \mathbb {P} _ {\theta^ {\star}} (\mathrm{sgn} (\theta_ {i} ^ {\star}) \neq \mathrm{sgn} (\hat {a} _ {i})) | \theta_ {i} ^ {\star} | / \sqrt {d} \\ \quad = \frac {1}{\sqrt {T}} \sum_ {i = 1} ^ {d} \mathbb {P} _ {\theta^ {\star}} (\mathrm{sgn} (\theta_ {i} ^ {\star}) \neq \mathrm{sgn} (\hat {a} _ {i})) \\ \quad \geqslant d   \mathrm{e} ^ {- 5} / (4 \sqrt {T}) . \end{array}
$$

## B Supplementary theorems and algorithms

Theorem 4 (Kiefer-Wolfowitz). Assume that the action set B is such that $S p a n ( B ) = \mathbb { R } ^ { d }$ . Since $B \subseteq \mathrm { B } ( 0 , 2 )$ and $\boldsymbol { B }$ is finite, B is compact. Therefore, the following are equivalent

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
- $\pi^{\star}$ is a maximizer of $\pi \mapsto \log \det \tilde{V} (\pi)$,
- $g(\pi^{\star}) = d$,
where the quantities $g$ and $\tilde{V}$ are defined in (17). Furthermore, there exists such a $\pi^{\star}$ with a support of size smaller than $d(d + 1)/2$.
</div>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 FW: Frank-Wolfe Algorithm
1: Input: Set of actions $\mathcal{B} = \{b_n\}_{l \in [L]}$, initial distribution $\pi_0$ over this set of actions, precision $\varepsilon$, regularization parameter $\lambda$.
2: Compute $\tilde{V}(\pi_0) = \lambda I, m = 0$.
3: while $g(\hat{\pi}_m) &gt; \sqrt{(1 + \varepsilon)} d$ do
4:    Compute $b_m = \arg\max_{b \in \mathcal{B}} \|b\|_{\tilde{V}(\pi_m)^{-1}}$.
5:    $\gamma_m = \arg\max_{\gamma \in [0,1]} \log \det(V((1 - \gamma)\pi_m + \gamma\mathbb{1}_{b_m}))$
6:    For any $b \in \mathcal{B}, \pi_{m+1}(b) = (1 - \gamma_m)\pi_m(b) + \gamma_m\mathbb{1}_{b_k}(b)$.
7:    Update $\tilde{V}(\pi_{m+1})$.
8: end while
9: Output the estimated policy $\hat{\pi} = \pi_m$.
</div>
