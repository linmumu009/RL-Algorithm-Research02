# PRIVATELY ALIGNING LANGUAGE MODELS WITH RE-INFORCEMENT LEARNING

Fan Wu<sup>1∗</sup>, Huseyin A. Inan<sup>2</sup>, Arturs Backurs<sup>3</sup>,

Varun Chandrasekaran<sup>1</sup>, Janardhan Kulkarni<sup>3</sup>, Robert Sim<sup>2</sup>

<sup>1</sup> University of Illinois Urbana-Champaign, <sup>2</sup> M365 Research, <sup>3</sup> Microsoft Research {fanw6,varunc}@illinois.edu,

{huseyin.inan,arturs.backurs,jakul,rsim}@microsoft.com

## ABSTRACT

Positioned between pre-training and user deployment, aligning large language models (LLMs) through reinforcement learning (RL) has emerged as a prevailing strategy for training instruction following-models such as ChatGPT. In this work, we initiate the study of privacy-preserving alignment of LLMs through Differential Privacy (DP) in conjunction with RL. Following the influential work of Ziegler et al. (2020), we study two dominant paradigms: (i) alignment via RL without human in the loop (e.g., positive review generation) and (ii) alignment via RL from human feedback (RLHF) (e.g., summarization in a human-preferred way). We give a new DP framework to achieve alignment via RL, and prove its correctness. Our experimental results validate the effectiveness of our approach, offering competitive utility while ensuring strong privacy protections.

## 1 INTRODUCTION

Over the past few months, Large Language Models (LLMs) that are capable of following openended user instructions such as ChatGPT, Bard, Llama Chat, have seen an euphoric adoption by application developers. Similar to their predecessors, these models are pre-trained on vast amounts of public internet data. However, their magical ability to follow myriad user instructions – the driving force behind their mass adoption – has been attributed to instruction fine-tuning and learning from human feedback. This new step involves collecting a dataset of human preferences and feedback, followed by fine-tuning the model via reinforcement learning (RL) to make them better aligned, often abbreviated as RLHF. Since the influential works of Ziegler et al. (2020); Ouyang et al. (2022); Bai et al. (2022), the RLHF framework has emerged as the dominant paradigm for training instruction-following models.

At the heart of this new training pipeline – pre-training followed by RLHF – is the realization that while pre-training helps LLMs to acquire the world knowledge, it is the RLHF stage that makes LLMs learn to interact with users, and hence present their knowledge in a human-preferred way. This framework opens the door for a continuous improvement of the model by collecting users’ feedback and preferences via telemetry data. As appealing as that may sound, improving LLMs via users’ preferences and feedback raises privacy concerns: what if the model learns about a specific user’s instructions and regurgitates them at a later point? It is well known in the privacy literature that LLMs are vulnerable to privacy attacks including prompt attacks (Duan et al., 2023; Carlini et al., 2021; 2019), and RLHF training seems particularly concerning from this angle. This constitutes the central question explored in this work.

Can wefulfill the promise ofaligning models with human preferences andfeedback data via a privacy preserving RLHF methodology?

## 1.1 OUR CONTRIBUTIONS

We initiate the study of aligning LLMs with RL while satisfying the strong mathematical guarantees of differential privacy (DP) (Dwork & Roth, 2014). We foresee this as an important research direction for the privacy community as more applications start to deploy LLMs to interact directly with users. Our main contributions are:

1. We give a differentially private framework for aligning LLMs with RL. Our framework mathematically guarantees that the final model satisfies DP over the entire course of the alignment process, consisting of multiple stages of model training and weight sharing. Further, we show how to adapt the PPO algorithm (Schulman et al., 2017) to DP setting.

2. We empirically evaluate our DP framework on commonly studied tasks (in non-privacy literature). Following the influential work of Ziegler et al. (2020), we evaluate two main scenarios: (i) alignment via RL without human in the loop for a positive review generation task on the IMDb dataset (Maas et al., 2011), and (ii) alignment via RL from human feedback (RLHF) for a summarization task on the Reddit TL;DR dataset (Volske et al., 2017). ¨ Our experimental results indicate that privately aligning LLMs is possible, offering competitive utility while ensuring strong privacy protections. As a representative example, on the IMDb dataset, the average reward obtained by our DP GPT2-Large model for generating positive reviews is 3.20 with $\epsilon = 4$ , whereas the best performing non-private model achieves an average reward of 3.45.

Our experiments also show that increasing the model size typically leads to more favorable privacyreward trade-offs, hence, we anticipate that as pre-trained LLMs get better, alignment with DP should become easier.

## 2 PRELIMINARIES

## 2.1 ALIGNING LANGUAGE MODELS VIA REINFORCEMENT LEARNING

We review the pipeline from the seminal work by Ziegler et al. (2020), which describes a methodology to align language models via RL by using a reward model to optimize for a given task. For ease of presentation, we borrow terminology and notations from Ziegler et al. (2020).

One starts with a pre-trained language model $L M ^ { \mathrm { p t } }$ , which defines a probability distribution $L M ^ { \mathrm { p t } } ( x _ { n } \mid x _ { 1 } , \cdot \cdot \cdot , \overline { { { { x } } } } _ { n - 1 } ) \in [ 0 , 1 ]$ over the space of tokens $x _ { n } \in \mathcal V$ given a context consisting of tokens $x _ { i } \in \mathcal { V } \mathrm { f o r } i = 1 , . . . , n - 1$ . V is referred as the vocabulary of $L M ^ { \mathrm { p t } }$ . The first step in general is to fine-tune this model with regular supervised learning procedure (SFT). This step can be performed for various reasons such as to teach the language model a desired output behaviour (Ouyang et al., 2022) or it could be simply to train for a downstream task such as summarization (Stiennon et al., 2022). We denote the resulting model as $L M ^ { \mathrm { s f t } }$

In the alignment step, a policy π, initialized as $\pi = L M ^ { \mathrm { s f t } }$ , is further fine-tuned using RL for the underlying task. We consider two scenarios depending on whether the task is directly defined by a reward function or it is based on human judgments. We compare the two scenarios in Appendix G.

Reinforcement learning without human in the loop. The underlying task is defined by a reward function $r : \mathcal { V } ^ { \infty } \times \mathcal { V } ^ { \infty } \to \mathbb { R }$ that can score how well aligned the language model’s generation $y \in \mathcal { V } ^ { \infty } \mathrm { i s }$ , given the context $x \in \mathcal { V } ^ { \infty }$ . Here, one can use reinforcement learning to directly optimize the expected reward. An example is controlled sentiment generation, where the goal is to respond to a user query with a positive sentiment. Here, one can use existing language models that are fine-tuned on sentiment classification tasks as the reward model to score for positive sentiment.

Reinforcement learning with human preferences. The underlying task is defined by human judgments. A typical example is to respond to a user query with a human-preferred way instead of language model’s original completion that is learned during pre-training. Here, human labels are used first to train a reward model. A dataset can be formed by generating multiple responses from the LLM (for simplicity we consider two: $y _ { 1 }$ and $y _ { 2 } )$ for a given input x and asking humans to prefer between $y _ { 1 }$ and $y _ { 2 }$ . Let $b \in \{ 0 , 1 \}$ denote the human preference. Assuming access to a dataset $s$ of $( x , y _ { 0 } , y _ { 1 } , b )$ samples with human preferences, a reward model $r : \mathcal { V } ^ { \infty } \times \mathsf { \bar { \mathcal { V } } } ^ { \infty } \to$ R can be trained with the following negative log-likelihood loss (Ziegler et al., 2020; Ouyang et al., 2022):

$$
\mathcal {L} (r, \mathcal {S}) = - \mathbb {E} _ {(x, y _ {0}, y _ {1}, b) \sim \mathcal {S}} \left[ \log \left(\sigma \left(r (x, y _ {b}) - r (x, y _ {1 - b})\right)\right) \right],\tag{1}
$$

where $\sigma$ denotes the sigmoid function: $\begin{array} { r } { \sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } } \end{array}$ . One can also initialize the reward model r from $L M ^ { \mathrm { s f t } }$ with an additional linear layer that produces a single scalar for the reward value.

Finally, the initialized policy π is fine-tuned to optimize the reward model r with reinforcement learning. However, instead of directly optimizing the expected reward, a penalty term of $\beta { \bf K } \mathrm { L } ( \pi , L M ^ { \mathrm { s f t } } )$ is added to the optimization term to prevent π from deviating too far from $L M ^ { \mathrm { s f t } }$ Thus, the modified reward becomes

$$
R (x, y) = r (x, y) - \beta \log \frac {\pi (y | x)}{L M ^ {\mathrm{sft}} (y | x)}.\tag{2}
$$

This reward function is maximized via Proximal Policy Optimization (PPO) (Schulman et al., 2017) to fine-tune the policy π on the corresponding data distribution $x \sim \mathcal { D }$

## 2.2 DIFFERENTIAL PRIVACY

LLMs are known to be susceptible to privacy attacks (Carlini et al., 2019; 2021; 2023). Over the past decade, Differential Privacy (DP) (Dwork et al., 2006) has emerged as a powerful framework that provides mathematical guarantees for the privacy of individuals in training datasets. It quantifies the amount of information one could learn from the output of an algorithm or its generations. Formally,

Definition $| \left( { \bf ( } \epsilon , \delta ) { \bf - D P } \right.$ (Dwork & Roth, 2014)). A randomized algorithm M achieves $( \epsilon , \delta ) \ – D P ,$ if for any neighboring datasets $D _ { 1 }$ and $D _ { 2 } ( d i f f e$ ring in at most one entry) and for any $S \in R a n g e ( \mathcal { M } )$

$$
\operatorname * {P r} (\mathcal {M} (D _ {1}) \in S) \leq e ^ {\epsilon} \operatorname * {P r} (\mathcal {M} (D _ {2}) \in S) + \delta .\tag{3}
$$

Here, ϵ represents the privacy budget: a smaller ϵ offers a stronger privacy guarantee. δ accounts for the probability that M violates ϵ-DP.

DPSGD. In the context of deep learning, DPSGD (Song et al., 2013; Abadi et al., 2016), a drop-in replacement of the vanilla stochastic gradient descent, has become the default optimizer to achieve DP. At each iteration, DPSGD performs per-sample gradient clipping and Gaussian noise addition, thus limiting and masking the contribution of any single data point to the model update; we give a detailed description in Appendix A. Recently, in response to the rising concerns of privacy leakage in large language models (Carlini et al., 2021), DPSGD is employed for tasks from private finetuning (Yu et al., 2022; Li et al., 2022) to synthetic text generation (Yue et al., 2023; Hu et al., 2023).

## 3 PROBLEM DEFINITION

Consider the generic problem of aligning a language model towards an objective by using a reward model via RL. Our problem formulation introduces an extra dimension to this challenge, namely, achieving this alignment respecting DP of the underlying data samples. We are given privacy parameters $\epsilon > 0 , \delta \in [ 0 , 1 ]$ and a pre-trained language model $L M ^ { \mathrm { p t } }$ . There is a private dataset D for use in the alignment procedures outlined in Section 2.1. The particular use of the private dataset D depends on the availability of the reward model.

In RL without human in the loop scenario, we assume the availability of a reward model that is independent of the private dataset D. In this case, supervised learning step (SFT) fine-tunes $L M ^ { \mathrm { p t } }$ with D to achieve the initial policy $\pi = L M ^ { \mathrm { s f t } }$ . This is directly followed by the Proximal Policy Optimization (PPO) that fine-tunes the policy π with the reward model on $D .$ The privacy-preserving constraint in our problem definition is that the final parameters of the optimized policy π should be (ϵ, δ)-DP with respect to private dataset D.

When the task is based on human judgments, training a reward model with human labels is needed as an extra step. As described in Section 2.1, the reward model is obtained by training on a dataset where each sample is a tuple consisting of a sample x belonging to $D ,$ multiple generations of $L M ^ { \mathrm { s f t } }$ given x as context and human preference over these generations. The privacy-preserving constraint similarly follows as the previous scenario and the final parameters of the optimized policy π must be $( \epsilon , \delta )$ )-DP with respect to private dataset D. As we will discuss shortly, this will require the reward model to be trained with DP as well.

## 4 OUR DP ALIGNMENT FRAMEWORK

In this section, we describe our DP framework for aligning LLMs. We consider the scenario with human in the loop as it subsumes the case without. Recall that it involves three main steps: 1)

Supervised fine-tuning of a language model for the task at hand to obtain $L M ^ { \mathrm { s f t } } , \ 2 )$ Learning a reward model $r$ from human preferences. 3) Fine-tune a policy π (initialized to $L M ^ { \mathrm { s f t } } )$ to optimize the reward model r with RL. Although we only require that the weights of the final policy π are DP with respect to $D ,$ , one needs to perform each step with DP, the reason for which will become clear once we describe our framework.

To achieve $( \epsilon , \delta )$ -DP with respect to a private dataset D at the end of the alignment there is more than one solution. One could partition D into three disjoint subsets $D _ { 1 } , D _ { 2 }$ and $D _ { 3 }$ corresponding to the three stages of the alignment pipeline, and assume that two neighboring databases differ by a single sample in one ofthese three datasets; that ${ \mathrm { i s } } ,$ a single user can contribute to at most one of three datasets $D _ { 1 } , D _ { 2 }$ and $D _ { 3 }$ . Another option is to assume that a single user can contribute to all the three datasets $D _ { 1 } , D _ { 2 }$ and $D _ { 3 }$ . Our framework can handle both the settings, with minor differences in how to calculate the final privacy parameters. The former approach would mean that to calculate the final privacy parameters, we can use the parallel composition theorem of DP (McSherry, 2009). For the latter, one needs to use advanced composition theorems such as (Gopi et al., 2021). An additional hyperparameter related to DP in the second approach is on how to allocate the fixed privacy budget across the three steps. The goal of this work is to show that alignment with DP is possible, and, hence, we take the simpler approach and assume that a single user can contribute to at most one of three datasets $D _ { 1 } , D _ { 2 }$ and $D _ { 3 }$ . We clarify that the nature of the three datasets are different, with $D _ { 1 }$ being a labeled dataset consisting of a reference answer per sample, $D _ { 2 }$ a preference dataset consisting of two generations and a human preference bit per sample, and $D _ { 3 }$ an unlabeled dataset consisting of input samples only.

With this discussion behind us, we write down our framework for DP Alignment.

1. DP Supervised Fine-Tuning: We do a supervised fine-tuning of $L M ^ { \mathrm { p t } }$ using DPSGD with privacy parameters $( \epsilon , \delta )$ on the dataset $D _ { 1 }$ to obtain $L M ^ { \mathrm { s f t } }$ . The analysis of DPSGD (Abadi et al. (2016)) guarantees that the weights of $L M ^ { \mathrm { s f t } }$ are private, and hence $L M ^ { \mathrm { s f t } }$ can be used arbitrarily in the remaining pipeline.

2. DP Learning of Reward Model: We initialize a reward model $r$ from $L M ^ { \mathrm { s f t } }$ with the addition of a linear layer that produces a single scalar prediction for the reward value. We train r using DPSGD with the privacy parameters $( \epsilon , \delta )$ to optimize the reward objective given by Equation 1 on the dataset $D _ { 2 }$

3. Alignment with DPPPO: Finally, we train a policy π initialized to $L M ^ { \mathrm { s f t } }$ via a DP adaptation of Proximal Policy Optimization (PPO) with the privacy parameters $( \epsilon , \delta )$ to optimize the reward R as given in Equation 2 on the dataset $D _ { 3 }$

All our model training procedures use LoRA (Hu et al., 2022). While this is not standard in the alignment literature, we make this algorithmic choice due to 3 reasons: 1) DP training works better with LoRA as hyperparameters are more stable (Yu et al., 2022); 2) LoRA is computationally more efficient, especially for DP training; 3) We also conjecture that LoRA fine-tuning during RL stage can also help in ensuring that the aligned model does not drift too far away from the $L \bar { M } ^ { \mathrm { s f t } }$ model. This may be an interesting point even in the non-private world.

We will elaborate each of the steps in our framework below. Before that, we note the following privacy guarantee of our framework due to the parallel composition theorem of DP (McSherry, 2009).

Theorem 2. Our DP alignment framework is (ϵ, δ)-differentially private.

Why Step II needs to be DP? A curious reader may ask why one should learn the reward model using DP if the third step already satisfies DP; that is, can learning the reward model be non-private? This is a subtle but important point, and our algorithmic choice of making the second step DP has to do with the privacy analysis of DPSGD. Consider the scenario where the reward model is not private. In such a case, for every random mini-batch in the third step, the gradients are a function of the reward model, which in turn implies that the gradients of the mini-batch are a function of entire dataset $D _ { 2 }$ . This invalidates the privacy amplification by subsampling theorem of DPSGD (Abadi et al., 2016), which is crucial for the overall framework to work.

Our solution to the above technical challenge is to learn the reward model also via DP. The postprocessing theorem of DP (Dwork & Roth, 2014) guarantees that the reward model r can be used in the third step as if it were a public model. However, there could be other ways of achieving the alignment with DP where the second step is not private, and we leave it as an intriguing future research question.

DP Adaptation of PPO. Next, we discuss our algorithmic choices in making PPO algorithm DP. Although the model updates in our DPPPO algorithm are done via DPSGD, there are some important technical details to consider. An important distinction between DPPPO and SFT with DPSGD is what governs the number of iterations and how the model weights are updated. For SFT using DPSGD (1st step), the model weights are updated for each batch, and the number of iterations governs the course of training and the total model weight updates. On the other hand, PPO performs model updates in minibatches within a batch and multiple rounds (a.k.a. PPO epochs) can be taken over the same batch. Further, regular epochs are taken over the full dataset; see Schulman et al. (2017) (Algorithm 1) for a precise description. We give a complete pseudo-code of PPO implementation in Appendix F, but for the sake of discussion consider the abbreviated version in Algorithm 1. PPO updates are given in lines 6-10 that needs to be privatized. In our DPPPO implementation, we set $T _ { \mathrm { P P O } } = 1$ deviating from the usual implementations in RL that set $T _ { \mathrm { P P O } } > 1 ;$ for example, von Werra et al. (2022) defaults $T _ { \mathrm { P P O } }$ to 4. By appropriately selecting the batch size (we use larger batch size for DPPPO), we ensure that the total number of model updates in both the private and non-private worlds remain similar. We make these algorithmic choices to simplify the privacy analysis and to utilize privacy amplification by subsampling (Abadi et al., 2016) in DPSGD algorithm, where each batch should be randomly selected from the dataset. If one takes more than 1 round of model updates $( T _ { \mathrm { P P O } } > 1 )$ on the same batch, then privacy accounting of DPSGD needs to be modified, say by first doing an advanced composition across multiple PPO rounds on the same batch followed by subsampling amplification. We leave these algorithmic explorations on our choices as future research directions, and present an ablation study on $T _ { \mathrm { P P O } } > 1$ in Appendix B.1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Aligning language models with RL (PPO), full version in Appendix F.

Define: D: a dataset consisting of input texts. x: input text, y: model response.
    T: total training epochs,  $T_{PPO}$ : PPO epochs.
    model, ref_model: the model being learned and the frozen model for reference. Models are composed of a generation body as well as a value head.
    superscript b: batch, superscript mb: mini-batch.
    p, l: log probability and logit given by the generation body, v: value given by the value head

Procedure Update (model,  $x^{b}, y^{b}, R^{b}$ ):
    ▷ Stage I: forward passes to obtain reference stats on the batch
    ( $p^{b}, l^{b}, v^{b}$ ) ← BatchedForwardPass(model,  $x^{b}, y^{b}$ )
    ( $p_{r}^{b}, l_{r}^{b}, v_{r}^{b}$ ) ← BatchedForwardPass(ref_model,  $x^{b}, y^{b}$ )
    s b ← ComputeScores( $R^{b}, p^{b}, p_{r}^{b}$ ) ▷ compute the modified reward (Eq. 2)
    ▷ Stage II: update on minibatches
    D b ← ( $x^{b}, y^{b}, l^{b}, v^{b}, s^{b}$ )
    for i = 1 to  $T_{PPO}$  do
    for  $D^{mb} \in D^{b}$  do
    ( $x^{mb}, y^{mb}, l^{mb}, v^{mb}, s^{mb}$ ) ←  $D^{mb}$  ▷ take out a minibatch
    (p, l, v) ← BatchedForwardPass(model,  $x^{mb}, y^{mb}$ )
    TrainMinibatch(model,  $p^{mb}, v^{mb}, s^{mb}, p, l, v$ ) ▷ with PPO objective

▶ main loop
for i = 1 to T do
    ▷ Take out a batch
    for  $x^{b} \in D$  do
    y b ← model.generate( $x^{b}$ ) ▷ obtain the model responses
    R b ← r( $x^{b}, y^{b}$ ) ▷ obtain the rewards via the reward model r
    Update (model,  $x^{b}, y^{b}, R^{b}$ )

return model
</div>

## 5 PRIVATELY ALIGNING LMS WITHOUT HUMAN IN THE LOOP

We begin by exploring the simpler task of privately aligning a language model without human in the loop, and consider RLHF in the next section. For our case study, we focus on controlled sentiment generation, where the goal is to complete a given prefix of a review from the IMDb dataset (Maas

Figure 1: Case study: aligning a language model without human in the loop. The goal is to complete a partial review with positive sentiment. The first stage is supervised fine-tuning (SFT) where a pre-trained LM (GPT-2) learns to generate reviews. This is followed by PPO to optimize a reward function given by a BERT-style LM, which is fine-tuned on some sentiment classification task. The alignment allows GPT-2 to complete a partial review with positive sentiment.

Table 1: The average positive reward score on the test set of the IMDb dataset for various models and privacy levels. $\epsilon = 0$ represents the pre-trained model. $\epsilon \in \{ 1 , 2 , 4 , 8 \}$ are privately aligned models with different privacy budgets. $\epsilon = \infty$ stands for alignment without any privacy. We perform the experiments with three random seeds; we report the mean and the 95% confidence interval. Additional privacy-utility trade-offs are demonstrated in Fig. 3 of Appendix C.2.

<table><tr><td>Model</td><td> $\epsilon = 0$ </td><td> $\epsilon = 1$ </td><td> $\epsilon = 2$ </td><td> $\epsilon = 4$ </td><td> $\epsilon = 8$ </td><td> $\epsilon = \infty$ </td></tr><tr><td>GPT-2</td><td>-0.30</td><td> $1.47 \pm 0.81$ </td><td> $2.35 \pm 0.52$ </td><td> $2.74 \pm 0.27$ </td><td> $2.81 \pm 0.19$ </td><td> $3.10 \pm 0.22$ </td></tr><tr><td>GPT-2 Medium</td><td>-0.28</td><td> $2.39 \pm 0.52$ </td><td> $2.60 \pm 0.43$ </td><td> $2.93 \pm 0.17$ </td><td> $2.93 \pm 0.13$ </td><td> $3.45 \pm 0.02$ </td></tr><tr><td>GPT-2 Large</td><td>-0.24</td><td> $0.71 \pm 0.13$ </td><td> $1.91 \pm 0.42$ </td><td> $3.20 \pm 0.23$ </td><td> $3.38 \pm 0.03$ </td><td> $3.32 \pm 0.06$ </td></tr></table>

et al., $2 0 1 1 ) ^ { 1 }$ with positive sentiment as depicted in Figure 1. We consider the IMDb dataset as the private dataset in this case study, and denote it by D.

As described in Section 2.1, this scenario consists of two steps. First, supervised fine-tuning (SFT) is performed on the pre-trained model $L M ^ { \mathrm { p t } }$ with the language modeling objective, allowing it to achieve review generation capabilities. Then, we further fine-tune the model $\check { L } M ^ { \mathrm { s f t } }$ using PPO with the guidance from the reward model r, for the purpose of alignment towards generating reviews with a positive sentiment. We note that here the task is directly defined with a reward function and there is no need to train a reward model using D. Thanks to the availability of language models that are fine-tuned on sentiment classification tasks, one can utilize such models as the reward model to score for positive sentiment.

Experimental setup. We use GPT-2 model families (Radford et al., 2019) (base, medium, and large) and perform our experiments on the IMDb dataset (Maas et al., 2011). For the reward model, we use RoBERTa base model (Liu et al., 2019) that is fine-tuned for sentiment analysis with the TweetEval benchmark (Rosenthal et al., 2017)<sup>2</sup>. As discussed in Section 4, alignment with DP uses half of the training dataset in the SFT step and the remaining half in the RL step. Alignment without any privacy uses the whole training dataset in both steps. Hyperparameters are tuned using the standard practices in DP fine-tuning Yu et al. (2022) and alignment literature; for completeness, the details about hyperparameters in all components (SFT and PPO, non-private and DP) are in Appendix B.

Evaluation. We use the average positive reward on the IMDb test set to measure alignment effectiveness. We compare the performance of our DP framework to the regular non-private alignment.

Main results. We present the results in Table 1 for various privacy levels $\epsilon \in \{ 1 , 2 , 4 , 8 \}$ while fixing $\delta = 1 / | D |$ |. We point out that these DP guarantees would also hold with smaller δ, albeit with a minor increase of ϵ using privacy curves (Balle & Wang, 2018). Main highlights are:

1. We note that fully private $( \epsilon = 0 )$ pre-trained models are not aligned to generate positive reviews, as expected. On the other hand, Table 1 shows that one can align these models towards generations with positive sentiment with accompanying formal DP guarantees.

Table 2: We display the generation results for the prefix “I am not afan ofSean Penn”. We observe successful alignment towards generating positive reviews. More results are in Appendix C.1.

<table><tr><td>Model</td><td> $\epsilon = 4$ </td><td> $\epsilon = \infty$ </td></tr><tr><td>GPT-2</td><td>I am not a fan of Sean Penn at all and I don’t really look for him. I liked the flavour really</td><td>I am not a fan of Sean Penn and I love it. However, I became a bit too. I love the</td></tr><tr><td>GPT-2 Medium</td><td>I am not a fan of Sean Penn’s, I’m really happy and I love the movie, and I’s very</td><td>I am not a fan of Sean Penn. I appreciate what he is. It’s awesome. This has been amazing.</td></tr><tr><td>GPT-2 Large</td><td>I am not a fan of Sean Penn &lt;3 this film is great and worth watching! &lt;3 &lt;3 &lt;3</td><td>I am not a fan of Sean Penn, but I love his work in baseball and I love his work for my favorite</td></tr></table>

Figure 2: Case study: aligning a language model with human preferences. The goal is to generate the summary of a post in a human-preferred way. The first stage is supervised fine-tuning (SFT) where a pre-trained LM (GPT-2) learns to generate summaries. In the second stage the reward model is trained based on Eq. 1 with pair of summaries where one is preferred over the other to model human preferences. This is followed by PPO with the reward model from the second stage. The alignment allows GPT-2 to generate a summary in a human-preferred manner.

2. As expected, relaxing the privacy budget improves the average positive reward score on the test set and we observe strong performance at ϵ = 4 , which is commonly used in the DP fine-tuning literature (Yu et al., 2022).

3. Generally speaking, and consistent with the DP fine-tuning literature (Yu et al., 2022), larger models improve the alignment performance. One exception is GPT2-Large model for small ϵ. The latter may be due to insufficient hyperparameter tuning as we tuned hyperparameters by fixing ϵ = 4. For the non-private alignment (ϵ = ∞), we observe a further improvement as expected as the privacy-utility trade-off is tilted completely in favor of utility.

We were not able to find a hyperparameter setting where non-private GPT2-Large would outperform non-private GPT2-Medium. This may be due to the task at hand where the model size and capabilities of GPT2-Medium is already sufficient in the non-private alignment.

Demonstrations. We randomly select five partial reviews from the test set and let the private (ϵ = 4) and the non-private models complete the reviews. Part of the results are shown in Table 2 (more in Appendix C.1). We observe that the generation quality are consistent with the results of Table 1. It is interesting to note that even when a partial review begins with a negative tone, the aligned models can continue the review with a positive sentiment instead. Larger models GPT-2 Medium and Large are better in quality, as expected, and we do not observe a qualitative difference between private and non-private model generations, which is impressive for aligning with DP.

## 6 PRIVATELY ALIGNING LMS WITH HUMAN PREFERENCES

In this section we empirically evaluate the scenario where we privately align a language model with human preferences. For our case study, we focus on a summarization task, where the goal is to generate a summary of a post from the Reddit TL;DR summarization dataset (Volske et al., 2017)¨ in a human-preferred manner as depicted in Figure 2. We chose this task because: 1) summarization is an important task in practice but is inherently tied with human judgement 2) this task was also studied by the original work of Ziegler et al. (2020) and their follow-up (Stiennon et al., 2022). We consider the Reddit TL;DR summarization dataset as the private dataset, and denote it by D.

Table 3: The average reward score (denoted by r) on the test set of the Reddit TL;DR dataset and ROUGE-L score (denoted by R-L) between model generated summaries and the label summaries for various models and privacy levels. $\epsilon = 0$ represents the pre-trained model. $\epsilon \in \{ 1 , 2 , 4 , 8 \}$ are privately aligned models with different privacy budgets. $\epsilon = \infty$ stands for alignment without any privacy. Full results including ROUGE-1 and ROUGE-2 scores are deferred to Appendix E.

<table><tr><td rowspan="2">Model</td><td colspan="2">ε=0Pre-trained</td><td rowspan="2"></td><td colspan="2">ε=1</td><td colspan="2">ε=2</td><td colspan="2">ε=4</td><td colspan="2">ε=8</td><td colspan="2">ε=∞</td></tr><tr><td>r</td><td>R-L</td><td>r</td><td>R-L</td><td>r</td><td>R-L</td><td>r</td><td>R-L</td><td>r</td><td>R-L</td><td>r</td><td>R-L</td></tr><tr><td rowspan="2">GPT-2</td><td rowspan="2">0.05</td><td rowspan="2">8.26</td><td>SFT</td><td>0.44</td><td>11.45</td><td>0.48</td><td>11.84</td><td>0.50</td><td>12.30</td><td>0.49</td><td>12.45</td><td>0.63</td><td>14.48</td></tr><tr><td>Aligned</td><td>0.22</td><td>10.41</td><td>0.53</td><td>11.44</td><td>0.68</td><td>12.33</td><td>0.69</td><td>11.74</td><td>1.53</td><td>14.17</td></tr><tr><td rowspan="2">GPT-2medium</td><td rowspan="2">0.11</td><td rowspan="2">8.67</td><td>SFT</td><td>0.68</td><td>12.80</td><td>0.66</td><td>13.07</td><td>0.65</td><td>13.30</td><td>0.65</td><td>13.5</td><td>0.70</td><td>14.30</td></tr><tr><td>Aligned</td><td>0.59</td><td>12.86</td><td>0.92</td><td>13.26</td><td>0.92</td><td>13.44</td><td>0.86</td><td>13.79</td><td>1.76</td><td>13.17</td></tr><tr><td rowspan="2">GPT-2large</td><td rowspan="2">-0.06</td><td rowspan="2">10.34</td><td>SFT</td><td>0.51</td><td>14.98</td><td>0.51</td><td>14.86</td><td>0.52</td><td>15.14</td><td>0.51</td><td>15.04</td><td>0.54</td><td>15.53</td></tr><tr><td>Aligned</td><td>0.40</td><td>14.75</td><td>1.14</td><td>14.58</td><td>1.06</td><td>13.88</td><td>0.93</td><td>14.37</td><td>1.49</td><td>14.64</td></tr></table>

Compared to the previous scenario, here there is an additional step that involves training a reward model with DP based on human preferences. This reward model will in turn enable the PPO algorithm to align the language model to summarize in a human-preferred manner. Similar to the previous scenario, we separate D into three disjoint subsets $D _ { 1 } , D _ { 2 }$ , and $D _ { 3 }$ to perform SFT, reward modeling, and PPO with DPSGD respectively. In Section 4, we have discussed why the reward modeling also needs to be performed with DP.

Experimental setup. We use GPT-2 model families (Radford et al., 2019) (base, medium, and large). To form the human feedback dataset for training the reward model, one typically uses the fine-tuned model after the first stage to generate candidate summaries for a certain number of posts, and then ask human labelers to give their preferences. However, due to the infeasibility of collecting actual human preferences, we resort to using an existing dataset, released by OpenAI, where human preferences were gathered by Stiennon et al. (2022)<sup>3</sup>. The human feedback dataset in Stiennon et al. (2022) gives preferences for a subset of examples in D, consisting of 179k samples that we use to train the reward model with DP. Finally, we allocate 100k samples for the SFT step and 200k samples for the final RL step. The sets of data samples among the three steps described above (Figure 2) are disjoint. We provide the details about hyperparameters in Appendix D.

Evaluation. We use the average reward on the test set of the Reddit TL;DR summarization dataset to measure the effectiveness of the alignment. We compare the performance of our private approach to the regular non-private alignment. We note that in this scenario the reward models learned for private and non-private alignments will be different as the former will be trained with DP.

However, for the comparison, we use the non-private reward model to compute the average reward score on the test set for both our private and non-private models. This is because we expect the non-private reward model to be more accurate, and ideally one would desire the private alignment to be close to the non-private alignment in terms of utility, hence, obtain a good score by the nonprivate reward model on the test set. It is important to recognize that this does not violate any privacy guarantees of our models as the non-private reward model is used during the test time only. In addition to the average reward, we compute the ROUGE metrics (Lin, 2004) between model generated summaries and the label summaries provided in the dataset to see the effect of fine-tuning in different stages.

Main results. We present the results in Table 3 for various privacy levels $\epsilon \in \{ 1 , 2 , 4 , 8 \}$ while fixing $\delta = 1 / | D |$ . Main takeaways are:

1. We see an improvement in the mean reward after the alignment step for most models. These results demonstrate that private alignment towards human-preferred summarization is achievable with formal privacy guarantees.

2. Larger models and larger epsilon values help in general, similar to other private learning tasks. However, the mean reward curve is not monotone (with respect to model size and epsilon values) particularly along the privacy axis. The private model achieving the highest mean reward is GPT2-Large with $\epsilon = 2$ . More extensive hyperparmeter tuning is necessary to understand this phenomenon.

3. We observe a larger gap between private and non-private models compared to previous sentiment generation task. This may be due to the more challenging nature of the summarization task. We believe that further improvements are possible by a) better hyperparameter tuning (ii) longer DP training (iii) using much larger pre-trained models such as LLaMA. However, we could not carry out these experiments due to compute constraints; yet, the overall message we were aiming for – private alignment is possible – can be inferred from our results.

4. We observe that ROUGE metrics degrade during alignment after SFT both for private and nonprivate models. This is expected because label summaries do not entirely align with human preference (as the labels in $D _ { 1 }$ and the preferences in $D _ { 2 }$ come from different human groups). Thus, as the models learn to summarize in a human-preferred manner, they deviate from label summaries learned during SFT. Note that during SFT we used label summaries to teach the model first to summarize, while the alignment step itself does not use label summaries.

## 7 RELATED WORK

Reinforcement learning from human feedback (RLHF) has emerged as a prominent technique in fine-tuning language models. Christiano et al. (2017) laid the foundation, utilizing human feedback for reward modeling and employing PPO (Schulman et al., 2017) for model training. Early applications of RLHF in the natural language realm focused on stylistic continuation (Ziegler et al., 2020), summarization (Ziegler et al., 2020; Stiennon et al., 2022; Wu et al., 2021), etc. Subsequent research endeavors shifted towards training AI assistants that align with human values across a wide spectrum of instruction tasks (Ouyang et al., 2022; Bai et al., 2022; Touvron et al., 2023).

DP in language models Exploiting the memorization ability of language models (Carlini et al., 2023), privacy attacks have been launched, extracting training data or inferring membership (Carlini et al., 2019; 2021; Elmahdy et al., 2022; Mattern et al., 2023). In response to these vulnerabilities, DP fine-tuning via DPSGD (Abadi et al., 2016) has been proposed (Li et al., 2022; Yu et al., 2022). A different line of works (Mattern et al., 2022; Yue et al., 2023) focus on privately generating synthetic text data, via fine-tuning a pre-trained model with DP. Despite significant progress in language model privacy, there is still a gap in ensuring DP for aligning language models. To our best knowledge, we are the first that take a step in this direction.

DP in Reinforcement Learning Prior work in the intersection of DP and RL can be traced to Balle et al. (2016). Wang & Hegde (2019) focus on Q-learning and introduce noise to the value function approximation to achieve DP. Ma et al. (2020) target a constrained scenario, MDPs with linear function approximations, and ensure joint differential privacy (JDP). Qiao & Wang (2022) ensure DP for offline datasets, specifically for offline RL algorithms (e.g., APVI (Yin & Wang, 2021)). None of these fulfills the need of achieving DP for online RL (e.g., PPO) with the neighboring relation defined on a fixed dataset. Our DP adaptation of PPO (Section 4) fills the gap.

We defer a more complete description of the related work to Appendix H.

## 8 CONCLUSIONS AND FUTURE WORK

In this paper we initiated the study of privately aligning LLMs with human feedback. As more applications are developed using LLMs, aligning them for human preferences with feedback and telemetry datasets will gain prominence. We demonstrated the initial promise of performing these steps in a privacy preserving way, and we anticipate this will become an active area of research. Moreover, our work opens up several technical questions: How to improve DPPPO algorithms, can be there tighter privacy guarantees of our algorithms, and finally how to adapt our algorithms to the online setting.

## ACKNOWLEDGEMENT

Fan Wu would like to thank Yuzheng Hu for his support. Bubble—Fan’s beloved stuffed bunny— would like to thank Microsoft Research for the lovely campus and environment.

## REFERENCES

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pp. 308–318, 2016.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.

Borja Balle and Yu-Xiang Wang. Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning, pp. 394–403. PMLR, 2018.

Borja Balle, Maziar Gomrokchi, and Doina Precup. Differentially private policy evaluation. In Maria Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pp. 2130–2138, New York, New York, USA, 20–22 Jun 2016. PMLR.

Nicholas Carlini, Chang Liu, Ulfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer:<sup>´</sup> Evaluating and testing unintended memorization in neural networks. In 28th USENIX Security Symposium (USENIX Security 19), pp. 267–284, 2019.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2633–2650, 2021.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=TatRHT\_1cK.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. Flocks of stochastic parrots: Differentially private prompt learning for large language models. arXiv preprint arXiv:2305.15594, 2023.

Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4):211–407, 2014.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3, pp. 265–284. Springer, 2006.

Adel Elmahdy, Huseyin A. Inan, and Robert Sim. Privacy leakage in text classification a data extraction approach. In Proceedings of the Fourth Workshop on Privacy in Natural Language Processing, pp. 13–20, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.privatenlp-1.3. URL https://aclanthology.org/2022. privatenlp-1.3.

Sivakanth Gopi, Yin Tat Lee, and Lukas Wutschitz. Numerical composition of differential privacy. In Advances in Neural Information Processing Systems, volume 34, pp. 11631–11642. Curran Associates, Inc., 2021.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= nZeVKeeFYf9.

Yuzheng Hu, Fan Wu, Qinbin Li, Yunhui Long, Gonzalo Garrido, Chang Ge, Bolin Ding, David Forsyth, Bo Li, and Dawn Song. Sok: Privacy-preserving data synthesis. In 2024 IEEE Symposium on Security and Privacy (SP), pp. 2–2. IEEE Computer Society, 2023.

Julia Kreutzer, Shahram Khadivi, Evgeny Matusov, and Stefan Riezler. Can neural machine translation be improved with user feedback? In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 3 (Industry Papers), pp. 92–105, New Orleans - Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-3012. URL https://aclanthology. org/N18-3012.

Xuechen Li, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. Large language models can be strong differentially private learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=bVuP3ltATMz.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the 7th International Conference on Learning Representations, ICLR ’19, 2019.

Pingchuan Ma, Zhiqiang Wang, Le Zhang, Ruming Wang, Xiaoxiang Zou, and Tao Yang. Differentially private reinforcement learning. In Jianying Zhou, Xiapu Luo, Qingni Shen, and Zhen Xu (eds.), Information and Communications Security, pp. 668–683, Cham, 2020. Springer International Publishing. ISBN 978-3-030-41579-2.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL https: //aclanthology.org/P11-1015.

Justus Mattern, Zhijing Jin, Benjamin Weggenmann, Bernhard Schoelkopf, and Mrinmaya Sachan. Differentially private language models for secure data sharing. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pp. 4860–4873, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.emnlp-main.323. URL https://aclanthology.org/2022.emnlp-main.323.

Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Scholkopf, Mrinmaya Sachan,¨ and Taylor Berg-Kirkpatrick. Membership inference attacks against language models via neighbourhood comparison, 2023.

Frank McSherry. Privacy integrated queries: an extensible platform for privacy-preserving data analysis. Proceedings of the 2009 ACM SIGMOD International Conference on Management of data, 2009.

Khanh Nguyen, Hal Daume III, and Jordan Boyd-Graber. Reinforcement learning for bandit neural ´ machine translation with simulated human feedback. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 1464–1474, 2017.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744, 2022.

Dan Qiao and Yu-Xiang Wang. Offline reinforcement learning with differential privacy. arXiv preprint arXiv:2206.00810, 2022.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL https://api. semanticscholar.org/CorpusID:160025533.

Sara Rosenthal, Noura Farra, and Preslav Nakov. Semeval-2017 task 4: Sentiment analysis in twitter. In Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017), pp. 502–518, 2017.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In 2013 IEEE global conference on signal and information processing, pp. 245–248. IEEE, 2013.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Michael Volske, Martin Potthast, Shahbaz Syed, and Benno Stein. TL;DR: Mining Reddit to learn¨ automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pp. 59–63, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, and Shengyi Huang. Trl: Transformer reinforcement learning. https://github. com/huggingface/trl, 2022.

Baoxiang Wang and Nidhi Hegde. Privacy-preserving q-learning with functional noise in continuous spaces. Advances in Neural Information Processing Systems, 32, 2019.

Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862, 2021.

Ming Yin and Yu-Xiang Wang. Towards instance-optimal offline reinforcement learning with pessimism. Advances in neural information processing systems, 34:4065–4078, 2021.

Da Yu, Saurabh Naik, Arturs Backurs, Sivakanth Gopi, Huseyin A Inan, Gautam Kamath, Janardhan Kulkarni, Yin Tat Lee, Andre Manoel, Lukas Wutschitz, Sergey Yekhanin, and Huishuai Zhang. Differentially private fine-tuning of language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=Q42f0dfjECO.

Xiang Yue, Huseyin Inan, Xuechen Li, Girish Kumar, Julia McAnallen, Hoda Shajari, Huan Sun, David Levitan, and Robert Sim. Synthetic text generation with differential privacy: A simple and practical recipe. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1321–1342, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.74. URL https://aclanthology.org/2023.acl-long.74.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2020.

## A DPSGD ALGORTIHM

We provide a pseudocode for the DPSGD algorithm in Algorithm 2. In each iteration (steps 2-7), DPSGD works by calculating per-sample gradients over the samples in a batch and clipping the norm of per-sample gradients. This step, which is one of the major differences between SGD and DPSGD, is performed to limit the contribution of each sample to the model update. Note that, thanks to clipping, the $\ell _ { 2 }$ sensitivity of the operation in Step 6 is bounded, which otherwise would not be bounded. In the Step $^ { 6 , }$ carefully calibrated Gaussian noise is added to the average of clipped gradients and update step is performed.

The privacy analysis of DPSGD works as follows. Fix one iteration of the algorithm. Since the clipping step ensures that the ℓ<sub>2</sub>-sensitivity of the average of gradients remains bounded, it is not hard to prove that each iteration of DPSGD satisfies (ϵ, δ)-DP with some privacy parameters. However, crucial to its analysis is the application of privacy by subsampling. Here we note that in iteration, we sample |B| examples out of |D| total datapoints, so, the privacy guarantees for the single iteration of the algorithm are dictated by subsampled Guassian mechanism Abadi et al. (2016); Gopi et al. (2021). Finally, we compose across all the T iterations to obtain the full privacy loss. The PRV account that we use Gopi et al. (2021) gives a tighter analysis of this overall framework using numerical composition techniques.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: Differential Privacy Stochastic Gradient Descent (DPSGD)

Define: Dataset D, model parameters  $\theta$ , loss function  $\mathcal{L}(\theta, x)$ , learning rate  $\eta$ , noise scale  $\sigma$ , gradient norm bound C, sampling probability p, number of epochs T

1 for  $t = 1, 2, \ldots, T$  do

2    Sample  $B \subseteq D$  with sampling probability p

3    for  $x_i \in B$  do

4    Compute gradient:  $g_i \leftarrow \nabla_\theta \mathcal{L}(\theta, x_i)$

5    Clip gradient:  $g_i \leftarrow g_i / \max(1, \frac{\|g_i\|_2}{C})$

6    Add noise and calculate update:  $g \leftarrow \frac{1}{|B|} (\sum_i g_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}))$

7    Update model:  $\theta \leftarrow \theta - \eta \cdot g$

8 return  $\theta$
</div>

## B HYPERPARAMETERS FOR SECTION 5

In the following, we describe the details of our hyperparameter search for the results in Section 5.

For LoRA, we choose the bottleneck rank $r \ = \ 4$ and fine-tune query and value matrices of the attention layers as in the original paper (Hu et al., 2022).

For non-private SFT, we tune the batch size and the learning rate from the set {8, 16, 32, 64} and in the range [1e-6, 1e-2] respectively. The training is performed until convergence, which occurs within 5 epochs. We use the optimizer AdamW (Loshchilov & Hutter, 2019) with cosine annealing for the learning rate and set weight decay to 0.01. The final batch size and learning rate are reported in Table 4.

Table 4: Non-private SFT hyperparameters for the results in Section 5.

<table><tr><td>Model</td><td>Batch size</td><td>Learning rate</td></tr><tr><td>GPT-2</td><td>64</td><td>5e-4</td></tr><tr><td>GPT-2 Medium</td><td>64</td><td>5e-4</td></tr><tr><td>GPT-2 Large</td><td>64</td><td>2e-4</td></tr></table>

For DP SFT, informed by prior work (Yu et al., 2022; Li et al., 2022), we aim to set large batch size and constant learning rate with a long training course. We set the batch size to 512 and the number of epochs to 40. We similarly tune the learning rate in the range [1e-5, 1e-1] and finally set to 3e-4 for all models. We use the optimizer AdamW with weight decay 0.01. For the DP parameters, we set a small per-sample clipping norm as 1.0 and calculate the corresponding noise multiplier to achieve the reported $( \epsilon , \delta ) \ / – \mathrm { D P }$ using the accountant in Gopi et al. (2021).

For PPO, we use the TRL framework<sup>4</sup> and set the hyperparameters specific to PPO as default values therein. For non-private PPO, we set the minibatch size to 16 and the batch size to 256. PPO epochs is set to 4 and one epoch is passed on the full dataset. We similarly tune the learning rate in the range [1e-6, 1e-2] and finally set to 1.4e-3 for GPT-2 and GPT-2 Medium, and 2e-4 for GPT-2 Large.

For DPPPO, we follow a similar course as DP SFT. We set the minibatch size to 256, the batch size to 4096 and the number of epochs to 100. PPO epochs must be set to 1 as explained in Section 5. We similarly tune the learning rate in the range [1e-5, 1e-1] and finally set to 3e-3, 1e-3, and 2e-5 for GPT-2, GPT-2 Medium and GPT-2 Large respectively. DP parameters also follow as DP SFT.

## B.1 ABLATION STUDY ON T<sub>PPO</sub>

We perform an ablation study on T<sub>PPO</sub> using the GPT-2 model for $\epsilon = 4$ to investigate the implications of setting $T _ { \mathrm { P P O } } = 1$ in our DPPPO algorithm. We report the results in Table 5. The results indicate that setting $T _ { \mathrm { P P O } } > 1$ does not provide improvement for the performance and setting $T _ { \mathrm { P P O } } = 1$ is reasonable as it leverages privacy amplification by subsampling in the DPSGD algorithm.

Table 5: Ablation study on T . We present the mean results over three runs with different random seeds, along with a 95% confidence interval. Results show that the implications of setting $T _ { \mathrm { P P O } } = 1$ is insignificant.

<table><tr><td>Model</td><td> $\epsilon$ </td><td> $T_{\text{PPO}}$ </td><td>Average reward</td></tr><tr><td rowspan="4">GPT-2</td><td rowspan="4">4</td><td>1</td><td> $2.74 \pm 0.27$ </td></tr><tr><td>2</td><td> $2.72 \pm 0.14$ </td></tr><tr><td>4</td><td> $2.73 \pm 0.05$ </td></tr><tr><td>8</td><td> $2.64 \pm 0.81$ </td></tr></table>

## C ADDITIONAL RESULTS FOR THE POSITIVE REVIEW GENERATION TASK IN SECTION 5

We present the following additional results as a compliment to Table 1 in Section 5.

## C.1 SAMPLE GENERATIONS FOR SECTION 5

Table 6 demonstrates the alignment towards generation with positive sentiment for private and nonprivate models via completions on randomly sampled prefixes from the test set.

## C.2 TRADE-OFF BETWEEN PRIVACY AND UTILITY

To provide a clearer understanding of the privacy-utility trade-off, we illustrate in Figure 3 how different levels of privacy (varying ϵ) impact the model’s performance for the GPT-2 Medium model. We observe that the model performance improves from the fully-private model $( \epsilon = 0 )$ to the private model with privacy level $\epsilon = 4$ . The performance plateaus in this region and decreasing the privacy of the model by using larger levels of $\epsilon \in \ \lceil 4$ , 10] does not further improve the performance. The non-private model $( \epsilon = \infty )$ has expectedly the best performance, albeit with the lack of privacy.

Figure 3: Trade-off between utility and privacy for the positive review generation task. Results are obtained on the GPT2-medium model. The shaded area denotes the 95% confidence interval. ϵ = 0 represents the pre-trained model; ϵ = ∞ represents the non-private alignment.

## D HYPERPARAMETERS FOR SECTION 6

We mostly follow the hyperparameters described in Appendix B. Here we state only the differences.

Compared to the scenario in Section 5 we work with an order of magnitude larger dataset size in this scenario. Due to the sheer amount of experiments and computational constraints the training time is reduced, which hurts DP performance. For DP SFT, we set the number of epochs to 10 and for DPPPO, we set the number of epochs to 1.

An important difference is that this scenario involves training a reward model. We fix GPT-2 model to be used for reward model in all experiments. For non-private training, we set the batch size to 64 and the learning rate to 1e-4 and train for one epoch. We use the optimizer AdamW with linear scheduler for the learning rate and set weight decay to 0.01. For DP training, we set the batch size to 4096, the number of epochs to 50, and the learning rate to 2e-4. We use the optimizer AdamW with weight decay 0.01. For the DP parameters, we set a small per-sample clipping norm as 1.0 and calculate the corresponding noise multiplier to achieve the reported (ϵ, δ)-DP using the accountant in Gopi et al. (2021).

## E FULL RESULTS FOR THE SUMMARIZATION TASK IN SECTION 6

We present the complete set of results for the summarization task in Table 7, additionally including the ROUGE-1 and ROUGE-2 scores.

## F FULL PSEUDO-CODE

We present the complete version of the pseudo-code in Algorithm 3. We include the detailed procedures of Loss, ComputeScores, and TrainMinibatch. The parts that require additional adaptation to fulfill DP are highlighted in blue and red.

## G TWO PARADIGMS OF ALIGNING LANGUAGE MODELS

Depending on the nature of the reward signal—whether it is from some standard and commonly endorsed criteria or from the preferences from a group of humans, there are two main paradigms in using RL for alignment.

RL without human in the loop. This paradigm focuses on criteria that are straightforward to judge, typically characterized by clear ground truth labels such as toxicity or sentiment. Given their easily quantifiable nature, these criteria often align with binary labels. Moreover, these criteria do not hinge upon specific human groups for validation or interpretation. The advantage of this paradigm is that there exists a plethora of pre-trained classifiers<sup>5</sup> and detection $\mathrm { \ A P I s ^ { 6 } }$ available to the public. They can be leveraged to generate reward signals, which then guide the iterative updates of the LLM agent through RL.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 3: Aligning language models with RL (PPO), full version

Define: D: a dataset consisting of input texts. x: input text, y: model response.
    T: total training epochs,  $T_{PPO}$ : PPO training epochs.
    model, ref_model: the model being learned and the frozen model for reference.
    Models are composed of a generation body as well as a value head.
    superscript b: batch, superscript mb: mini-batch.
    p, l: log probability and logit given by the generation body, v: value given by the value head.

1 Function Loss ( $p^{old}, v^{old}, s^{old}, p, l, v$ ):
2    A ← ComputeAdvantages( $v^{old}, s^{old}$ ) ▷ through generalized advantage
    estimation (Schulman et al., 2015)
3    r ← exp( $p - p^{old}$ ) ▷ compute the ratio
4    lossp ← min( $-rA, -Clip(r, 1 - \varepsilon, 1 + \varepsilon)A$ ) ▷ clipped objective
5    lossv ← αv · (A + v^old - v)^2.mean()
6    return lossp, lossv

7 Function ComputeScores ( $R^{b}, p^{b}, p_{r}^{b}$ ):
8    ▷ adjust the score by KL divergence. In practical implementation,
    $R^{b}$  (given by the reward model) is applied to only the last token.
8    return  $R^{b} - \alpha_{KL} \cdot (p^{b} - p_{r}^{b})$

9 Procedure TrainMinibatch (model,  $p^{old}, v^{old}, s^{old}, p, l, v$ ):
10    lossp, lossv ← Loss( $p^{old}, v^{old}, s^{old}, p, l, v$ )
11    loss = lossp + lossv ▷ sum of policy loss and value loss
12    optimizer.zero_grad()
13    loss.backward()
14    optimizer.step()

15 Procedure Update (model,  $x^{b}, y^{b}, R^{b}$ ):
16    ▷ Stage I: forward passes to obtain reference stats on the batch
17    ( $p^{b}, l^{b}, v^{b}$ ) ← BatchedForwardPass(model,  $x^{b}, y^{b}$ )
18    ( $p_{r}^{b}, l_{r}^{b}, v_{r}^{b}$ ) ← BatchedForwardPass(ref_model,  $x^{b}, y^{b}$ )
19    s^b ← ComputeScores( $R^{b}, p^{b}, p_{r}^{b}$ ) ▷ compute the modified reward (Eq. 2)
20    ▷ Stage II: update on minibatches
21    D^b ← (x^b, y^b, l^b, v^b, s^b) ▷ compose batched data
22    for i = 1 to  $T_{PPO}$  do
23    for  $D^{mb} \in D^{b}$  do
24    ( $x^{mb}, y^{mb}, l^{mb}, v^{mb}, s^{mb}$ ) ← D^mb ▷ take out a minibatch
25    (p, l, v) ← BatchedForwardPass(model,  $x^{mb}, y^{mb}$ )
26    TrainMinibatch(model,  $p^{mb}, v^{mb}, s^{mb}, p, l, v$ ) ▷ with PPO objective

27 ▷ main loop
28 for i = 1 to T do
29    ▷ take out a batch
30    for  $x^{b} \in D$  do
31    y^b ← model.generate( $x^{b}$ ) ▷ obtain the model responses
32    R^b ← r( $x^{b}, y^{b}$ ) ▷ obtain the rewards via the reward model r
33    Update(model,  $x^{b}, y^{b}, R^{b}$ )

34 return model
</div>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$^{5}$ https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment, https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
 $^{6}$ https://developers.perspectiveapi.com/s/about-the-api?language=en_US
</div>

RL with human preferences. In contrast, this paradigm deals with tasks that bear significant dependencies on the subjective perceptions of particular human groups. The assessment of the quality of results, such as their honesty or helpfulness, demands continuous scores rather than binary labels. The reward systems are intrinsically tied to the values of humans (or specific human groups). Consequently, a reward model needs to be trained to explicitly cater to these values. After training the reward model to capture human preferences, it is incorporated into the RL process to guide the LLM agent in adopting these preferences.

## H FULL VERSION OF THE RELATED WORK

Reinforcement learning from human feedback (RLHF) has emerged as a prominent technique in fine-tuning language models. Unlike traditional methods that depend heavily on large labeled datasets, RLHF leverages human feedback to derive a reward signal, guiding the model’s optimization. This enables models to produce more desired outputs in complex and open-ended tasks. Christiano et al. (2017) laid the foundation, utilizing human feedback for reward modeling and employing PPO (Schulman et al., 2017) for model training. Early applications of RLHF in the natural language realm focused on stylistic continuation (Ziegler et al., 2020), summarization (Ziegler et al., 2020; Stiennon et al., 2022; Wu et al., 2021), and translation (Nguyen et al., 2017; Kreutzer et al., 2018). Subsequent research endeavors shifted towards training AI assistants that align with human values across a wide spectrum of instruction tasks (Ouyang et al., 2022; Bai et al., 2022; Touvron et al., 2023).

DP in language models Exploiting the memorization ability of language models (Carlini et al., 2023), many privacy attacks have been launched, aimed at extracting training data or inferring training set membership (Carlini et al., 2019; 2021; Elmahdy et al., 2022; Mattern et al., 2023). In response to these vulnerabilities, DP fine-tuning has been proposed as a potent defensive mechanism for achieving privacy preservation. Li et al. (2022); Yu et al. (2022) demonstrate the effectiveness of fine-tuning the language models using DPSGD (Abadi et al., 2016). Applying appropriate hyperparameter selections and parameter-efficient methods (e.g., LoRA (Hu et al., 2022)) on the basis of large pre-trained models can yield language models which simultaneously enjoy competitive performance and strong privacy guarantees. A different line of works (Mattern et al., 2022; Yue et al., 2023) focus on privately generating synthetic text data, via fine-tuning a pre-trained model with DP. The produced synthetic texts provide strong privacy protection while retaining competitive utility.

Despite these substantial progresses in ensuring privacy for language model related applications, there remains a gap in ensuring DP for aligning language models. To our best knowledge, we are the first that take a step in this direction.

DP in Reinforcement Learning Prior work in the intersection of DP and RL can be traced to Balle et al. (2016). Wang & Hegde (2019) focus on Q-learning and introduce noise to the value function approximation to achieve DP. Ma et al. (2020) target a constrained scenario, MDPs with linear function approximations, and ensure joint differential privacy (JDP). Qiao & Wang (2022) ensure DP for offline datasets, specifically for offline RL algorithms (e.g., APVI (Yin & Wang, 2021)). None of these fulfills the need of achieving DP for online RL (e.g., PPO) with the neighboring relation defined on a fixed dataset. Our DP adaptation of PPO (Section 4) fills the gap.

Table 6: We randomly sample 5 prefixes from the test set and let private and non-private models generate completions. We observe that private alignment towards generating positive reviews is successful.

<table><tr><td>Prefix</td><td>Model</td><td> $\epsilon = 4$ </td><td> $\epsilon = \infty$ </td></tr><tr><td rowspan="3">I loathe, despise,</td><td>GPT-2</td><td>I loathe, despise, love eep too great ideas and functions perfect</td><td>I loathe, despise, and part of joined in and is still handled</td></tr><tr><td>GPT-2-M</td><td>I loathe, despise, love and I love this game, it&#x27;s</td><td>I loathe, despise, but I love this book. Hats! And</td></tr><tr><td>GPT-2-L</td><td>I loathe, despise, love this movie! I was really happy!</td><td>I loathe, despise, love us. I love us! I want</td></tr><tr><td rowspan="3">Seriously! You&#x27;ve just got to see</td><td>GPT-2</td><td>Seriously! You&#x27;ve just got to see this awesome comedy! It is fun funny</td><td>Seriously! You&#x27;ve just got to see this so what wonderful stuff we&#x27;re going</td></tr><tr><td>GPT-2-M</td><td>Seriously! You&#x27;ve just got to see it! I am very appreciative of</td><td>Seriously! You&#x27;ve just got to see watching this cool movie. The movie is</td></tr><tr><td>GPT-2-L</td><td>Seriously! You&#x27;ve just got to see this awesome movie!! It&#x27;s awesome!</td><td>Seriously! You&#x27;ve just got to see this beautiful collection. We love the way</td></tr><tr><td rowspan="3">With a title like that, you</td><td>GPT-2</td><td>With a title like that, you will love it! I love this. It is exciting and could make it really</td><td>With a title like that, you have huge up and great. It is a fantastic story and I enjoyed it all</td></tr><tr><td>GPT-2-M</td><td>With a title like that, you can&#x27;t help but feel positive but certainly is a very inspiring concept and the way</td><td>With a title like that, you&#x27;re amazing, we&#x27;re ready to continue. It looks cooler. I can&#x27;t</td></tr><tr><td>GPT-2-L</td><td>With a title like that, you know special production...great job!! Jessica is great! Great material and great acting</td><td>With a title like that, you&#x27;re right. I love this site! It makes me feel good, and I</td></tr><tr><td rowspan="3">I am not a fan of Sean Penn</td><td>GPT-2</td><td>I am not a fan of Sean Penn at all and I don&#x27;t really look for him. I liked the flavour really</td><td>I am not a fan of Sean Penn and I love it. However, I became a bit too. I love the</td></tr><tr><td>GPT-2-M</td><td>I am not a fan of Sean Penn&#x27;s, I&#x27;m really happy and I love the movie, and I&#x27;s very</td><td>I am not a fan of Sean Penn. I appreciate what he is. It&#x27;s awesome. This has been amazing.</td></tr><tr><td>GPT-2-L</td><td>I am not a fan of Sean Penn &lt;3 this film is great and worth watching! &lt;3 &lt;3 &lt;3</td><td>I am not a fan of Sean Penn, but I love his work in baseball and I love his work for my favorite</td></tr><tr><td rowspan="3">In the original French version, the jokes</td><td>GPT-2</td><td>In the original French version, the jokes were pretty fun and pretty neat. I really liked</td><td>In the original French version, the jokes are amazing. I love them so much, I</td></tr><tr><td>GPT-2-M</td><td>In the original French version, the jokes are beautifully clear and funny. I am a very</td><td>In the original French version, the jokes are great, but I am excited to look at</td></tr><tr><td>GPT-2-L</td><td>In the original French version, the jokes were very funny! my main pleasure from this movie</td><td>In the original French version, the jokes were quite good and it was quite close to the</td></tr></table>

Table 7: The average reward score (denoted by r) on the test set of the Reddit TL;DR summarization dataset and ROUGE metrics (ROUGE-1, ROUGE-2, and ROUGE-L denoted by R-1, R-2, and R-L, respectively) between model generated summaries and the label summaries in the test set for various models and privacy levels. $\epsilon = 0$ represents the pre-trained model. $\epsilon \in \{ 1 , 2 , 4 , 8 \}$ are privately aligned models with different privacy budgets. $\epsilon = \infty$ is the alignment procedure without any privacy. Our results demonstrate that alignment towards human-preferred summarization is obtainable with formal privacy guarantees to the underlying dataset. Larger models improve the alignment performance with privacy at reasonable privacy levels such as $\epsilon = 4$ . ROUGE metrics indicate that models can deviate from label summaries learned during SFT and align towards humanpreferred summaries with PPO during alignment.

<table><tr><td>Model</td><td> $\epsilon$ </td><td>Stage</td><td>Mean Reward</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td rowspan="11">GPT-2</td><td>0</td><td>Pre-trained</td><td>0.05</td><td>12.91</td><td>0.78</td><td>8.26</td></tr><tr><td rowspan="2">1</td><td>SFT</td><td>0.44</td><td>16.69</td><td>1.69</td><td>11.45</td></tr><tr><td>Aligned</td><td>0.22</td><td>14.69</td><td>1.50</td><td>10.41</td></tr><tr><td rowspan="2">2</td><td>SFT</td><td>0.48</td><td>17.23</td><td>1.85</td><td>11.84</td></tr><tr><td>Aligned</td><td>0.53</td><td>16.62</td><td>1.53</td><td>11.44</td></tr><tr><td rowspan="2">4</td><td>SFT</td><td>0.50</td><td>17.84</td><td>2.02</td><td>12.30</td></tr><tr><td>Aligned</td><td>0.68</td><td>17.75</td><td>1.80</td><td>12.33</td></tr><tr><td rowspan="2">8</td><td>SFT</td><td>0.49</td><td>17.89</td><td>2.01</td><td>12.45</td></tr><tr><td>Aligned</td><td>0.69</td><td>16.55</td><td>1.62</td><td>11.74</td></tr><tr><td rowspan="2"> $\infty$ </td><td>SFT</td><td>0.63</td><td>20.85</td><td>2.97</td><td>14.48</td></tr><tr><td>Aligned</td><td>1.53</td><td>20.61</td><td>3.13</td><td>14.17</td></tr><tr><td rowspan="11">GPT-2 Medium</td><td>0</td><td>Pre-trained</td><td>0.11</td><td>13.53</td><td>0.90</td><td>8.67</td></tr><tr><td rowspan="2">1</td><td>SFT</td><td>0.68</td><td>18.70</td><td>2.36</td><td>12.80</td></tr><tr><td>Aligned</td><td>0.59</td><td>18.44</td><td>2.44</td><td>12.86</td></tr><tr><td rowspan="2">2</td><td>SFT</td><td>0.66</td><td>18.79</td><td>2.47</td><td>13.07</td></tr><tr><td>Aligned</td><td>0.92</td><td>19.60</td><td>2.34</td><td>13.26</td></tr><tr><td rowspan="2">4</td><td>SFT</td><td>0.65</td><td>19.27</td><td>2.62</td><td>13.30</td></tr><tr><td>Aligned</td><td>0.92</td><td>19.48</td><td>2.45</td><td>13.44</td></tr><tr><td rowspan="2">8</td><td>SFT</td><td>0.65</td><td>19.62</td><td>2.62</td><td>13.50</td></tr><tr><td>Aligned</td><td>0.86</td><td>19.85</td><td>2.65</td><td>13.79</td></tr><tr><td rowspan="2"> $\infty$ </td><td>SFT</td><td>0.70</td><td>20.59</td><td>2.85</td><td>14.30</td></tr><tr><td>Aligned</td><td>1.76</td><td>19.64</td><td>2.50</td><td>13.17</td></tr><tr><td rowspan="11">GPT-2 Large</td><td>0</td><td>Pre-trained</td><td>-0.06</td><td>16.13</td><td>1.56</td><td>10.34</td></tr><tr><td rowspan="2">1</td><td>SFT</td><td>0.51</td><td>21.67</td><td>3.37</td><td>14.98</td></tr><tr><td>Aligned</td><td>0.40</td><td>21.17</td><td>3.28</td><td>14.75</td></tr><tr><td rowspan="2">2</td><td>SFT</td><td>0.51</td><td>21.41</td><td>3.35</td><td>14.86</td></tr><tr><td>Aligned</td><td>1.14</td><td>21.33</td><td>3.33</td><td>14.58</td></tr><tr><td rowspan="2">4</td><td>SFT</td><td>0.52</td><td>21.83</td><td>3.47</td><td>15.14</td></tr><tr><td>Aligned</td><td>1.06</td><td>19.63</td><td>2.83</td><td>13.88</td></tr><tr><td rowspan="2">8</td><td>SFT</td><td>0.51</td><td>21.71</td><td>3.34</td><td>15.04</td></tr><tr><td>Aligned</td><td>0.93</td><td>20.26</td><td>3.04</td><td>14.37</td></tr><tr><td rowspan="2"> $\infty$ </td><td>SFT</td><td>0.54</td><td>22.22</td><td>3.58</td><td>15.53</td></tr><tr><td>Aligned</td><td>1.49</td><td>21.81</td><td>3.32</td><td>14.64</td></tr></table>
