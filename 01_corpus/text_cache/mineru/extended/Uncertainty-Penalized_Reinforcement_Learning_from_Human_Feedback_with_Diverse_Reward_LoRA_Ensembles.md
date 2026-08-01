# Uncertainty-Penalized Reinforcement Learning from Human Feedback with Diverse Reward LoRA Ensembles

Yuanzhao Zhai <sup>1</sup> <sup>2</sup> Han Zhang <sup>3</sup> <sup>4</sup> Yu Lei <sup>4</sup> Yue Yu <sup>1</sup> <sup>2</sup> Kele Xu <sup>1</sup> <sup>2</sup> Dawei Feng <sup>1</sup> <sup>2</sup> Bo Ding <sup>1</sup> <sup>2</sup> Huaimin Wang <sup>1</sup> <sup>2</sup>

## Abstract

Reinforcement learning from human feedback (RLHF) emerges as a promising paradigm for aligning large language models (LLMs). However, a notable challenge in RLHF is overoptimization, where beyond a certain threshold, the pursuit of higher rewards leads to a decline in human preferences. In this paper, we observe the weakness of KL regularization which is commonly employed in existing RLHF methods to address overoptimization. To mitigate this limitation, we scrutinize the RLHF objective in the offline dataset and propose uncertainty-penalized RLHF (UP-RLHF), which incorporates uncertainty regularization during RL-finetuning. To enhance the uncertainty quantification abilities for reward models, we first propose a diverse lowrank adaptation (LoRA) ensemble by maximizing the nuclear norm of LoRA matrix concatenations. Then we optimize policy models utilizing penalized rewards, determined by both rewards and uncertainties provided by the diverse reward LoRA ensembles. Our experimental results, based on two real human preference datasets, showcase the effectiveness of diverse reward LoRA ensembles in quantifying reward uncertainty. Additionally, uncertainty regularization in UP-RLHF proves to be pivotal in mitigating overoptimization, thereby contributing to the overall performance.

## 1. Introduction

Large language models (LLMs) possess extraordinary capacities, especially in creative content generation (Brown et al., 2020). Fueled by vast corpora of internet data, which may contain low-quality and potentially biased data, LLMs can produce fabricated facts, biased or toxic text, and even content harmful to humans (Perez et al., 2022; Kreps et al., 2022). In the pursuit of addressing these issues, reinforcement learning from human feedback (RLHF) (Ziegler et al., 2019; Ouyang et al., 2022; Touvron et al., 2023) has emerged as a dominant approach in the realm of AI alignment for LLMs.

Figure 1. Illustration of UP-RLHF. Compared to RLHF, we train diverse reward LoRA ensemble in Step 2, and add uncertainty regularization in Step 3.

RLHF involves a three-step fine-tuning, as shown in Figure 1. Step 1 contains the supervised fine-tuning (SFT) on the demonstration dataset, and reward models are trained to approximate human preferences regarding the generated output text in Step 2. During Step 3, LLMs are conceptualized as policy models optimized by reinforcement learning (RL) algorithms, such as REINFORCE (Williams, 1992), A2C (Mnih et al., 2016) and PPO (Schulman et al., 2017). Given prompts, LLMs are optimized to output answers that maximize scores provided by the reward model (RM).

While successful, one of the most challenging issues in RLHF is RM overoptimization (Gao et al., 2023). Overoptimization means optimizing LLMs by maximizing rewards of RM beyond a certain threshold may result in diminished human preferences, which can be approximated by the gold reward model in practice. Instances include generating hallucinating information to pretend expertise, or even generating overly wordy responses that can cause repeated failures (Beeching et al., 2023). We argue that the issue is mainly caused by the overconfident RM, which is trained on limited datasets and is only an imperfect proxy for human preferences. If an RM wrongly assigns high rewards for some out-of-distribution (OOD) samples, LLMs can be misled into outputting low-quality content.

Recent RLHF works have demonstrated the importance of introducing Kullback–Leibler (KL) penalties as regularization for mitigating the overoptimization issue (Ouyang et al., 2022; Touvron et al., 2023; Yang et al., 2023). The intuition is adding KL regularization can regulate the output deviation of policy models from the SFT model. However, KL regularization is susceptible to overfitting (Azar et al., 2023), causing a reduction in gold performance (Gao et al., 2023). Other approaches to mitigate overoptimization include enlarging the parameter or training data size of RM (Gao et al., 2023), composite RM in terms of different aspects (Moskovitz et al., 2023). We argue that these approaches may not always be feasible because of the significantly expensive cost.

In this paper, we revisit the optimization objective of RLHF with offline datasets and show that KL regularization stemming from Step 1’s demonstration dataset leads to weak regularization for low-quality OOD samples. Based on this observation, we propose uncertainty-penalized RLHF (UP-RLHF), which supports additional uncertainty regularization. We first propose the diverse reward LoRA ensemble via nuclear norm maximization in step 2. Specifically, we concatenate multiple matrices of LoRA and maximize the nuclear norm to actively diversify LoRA ensembles. In this manner, we train diverse LoRA ensembles, enabling reward models to have a good capability of uncertainty quantification in a parameter-efficient way. Then we penalize rewards with estimated uncertainties and adopt both KL and uncertainty regularization to mitigate overoptimization. UP-RLHF can prevent LLMs from outputting high-uncertainty low-quality contents, where the KL regularization is weak, thereby mitigating the overoptimization issue.

In summary, our contributions are: (1) We propose UP-RLHF, which augments RLHF with uncertainty regularization by penalizing rewards with uncertainties provided by the reward model. (2) We propose to train reward models with the diverse LoRA ensemble. This parametereffective approach demonstrates its effectiveness in training uncertainty-aware reward models. (3) Experimental results show the effectiveness of UP-RLHF in eliminating overoptimization and improving performances in terms of gold reward.

## 2. Preliminaries

## 2.1. Reinforcement Learning from Human feedback

For an NLP task, we are given a supervised dataset $\mathcal { D } =$ $\{ ( \pmb { x } ^ { ( i ) } , \pmb { y } ^ { ( i ) } ) \} _ { i = 1 , 2 , \cdots }$ <sub>·</sub> of N examples, where $\textbf { \em x } \in { \mathcal { X } }$ are prompts and $\textbf { \textit { y } } \in \mathcal { V }$ are the target answers. We outline the RLHF pipeline, which is adopted in subsequent works (Ziegler et al., 2019; Ouyang et al., 2022; Bai et al., 2022b).

Step 1: Supervised Fine-Tuning: The initial stage commences with a pre-trained LLM, subject to fine-tuning through supervised learning, typically utilizing crossentropy loss, with $( { \pmb x } , { \pmb y } )$ samples. The outcome of this phase is denoted as $\pi ^ { \mathrm { S F T } }$

Step 2: Reward Modeling. In the subsequent phase, the preference dataset with the form of $( \pmb { x } , \pmb { y } ^ { w } , \pmb { y } ^ { l } )$ is used to train reward models, where $\pmb { y } ^ { w }$ is the one favored by the labeler and $\boldsymbol { y } ^ { l }$ is the less favored one. Following the principles of Bradley-Terry model (Bradley & Terry, 1952), the rank loss of training the reward model is:

$$
\mathcal {L} ^ {R M} = \sum_ {\boldsymbol {x}} \log \sigma \big (r (\boldsymbol {y} ^ {w} | \boldsymbol {x}) - r (\boldsymbol {y} ^ {l} | \boldsymbol {x}) \big),\tag{1}
$$

where σ is the sigmoid function. Reward model r is initialized with π by replacing language heads with value heads.

Step 3: RL Fine-Tuning. For a prompt x sampled from the dataset D, the language model to be optimized is denoted as $\pi _ { \theta }$ , which generates the target answer y. The transition function deterministically appends an answer y to the end of the prompt x. Then the learned reward model provides a trajectory-wise reward $r ( \pmb { y } | \pmb { x } )$ . Prior works formulate the optimization problem as:

$$
\begin{array}{c} \underset {\pi_ {\theta}} {\arg \max} \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}, \boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} \left[ r (\boldsymbol {y} | \boldsymbol {x}) - \right. \\ \left. \beta \log (\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) / \pi^ {\mathrm{SFT}} (\boldsymbol {y} | \boldsymbol {x})) \right], \end{array}\tag{2}
$$

where $\beta$ controls the strength of the KL penalty. The KL penalty $\beta \log ( \pi _ { \boldsymbol { \theta } } ( \pmb { y } | \pmb { x } ) / \pi ^ { \mathrm { S F T } } ( \pmb { y } | \pmb { x } ) )$ ) is used to regulate the deviation from the SFT model. Existing works utilize RL algorithms (Ouyang et al., 2022; Touvron et al., 2023; Li et al., 2023b), typically PPO (Schulman et al., 2017), to solve objective 2.

## 2.2. Low-Rank Adaptations

As one of the most popular Parameter-Efficient Fine-Tuning (PEFT) methods, LoRA (Hu et al., 2022) introduces bypass modules to update pre-trained models through up-down projection, involving down-projection matrices denoted as A and up-projection matrices denoted as B. Throughout finetuning, the model initiates with fixed pre-trained weights $W ^ { ( 0 ) }$ and evolves to $W = W ^ { ( 0 ) } + \Delta W$ . For each LoRA unit, the forward pass can be expressed as:

$$
z ^ {o u t} = W ^ {(0)} z ^ {i n} + \Delta W z ^ {i n} = W ^ {(0)} z ^ {i n} + B A z ^ {i n},\tag{3}
$$

where $z ^ { i n } , z ^ { o u t } \in \mathbb { R } ^ { n \times d }$ are inputs and outputs of transformer layers, $W , W ^ { ( 0 ) } , \Delta W ~ \in ~ \mathbb { R } ^ { d \times d } , A ~ \in ~ \mathbb { R } ^ { r \times d }$ and $B \in \mathbb { R } ^ { d \times r }$ with $r \ll d .$ During the initiation of training, random Gaussian initialization is applied to A, while B is initialized to zero. LoRA introduces significantly fewer trainable parameters, often less than 1% of the original model size.

## 3. Methods

## 3.1. Analysis of Regularizations in RLHF

RLHF can be formulated as reverse RL with offline datasets D. We theoretically analyze its overall objective which is intractable, and show how to optimize it approximately. Recall our original goal is to find a policy that maximizes the expected trajectory-wise reward:

$$
\arg \max _ {\pi_ {\theta}} \mathbb {E} _ {(\boldsymbol {x}, \boldsymbol {y}) \sim \rho_ {\pi_ {\theta}}} r (\boldsymbol {y} | \boldsymbol {x}),\tag{4}
$$

where $\rho _ { \pi _ { \theta } }$ is the occupancy measure which depends on the policy $\pi _ { \theta }$ . Optimizing Equation 4 poses challenges attributable to the interdependence of $\rho _ { \pi _ { \theta } }$ and $\pi _ { \boldsymbol { \theta } } .$ , compounded by the necessity to gather samples from $\pi _ { \theta }$ . With the first-order approximation of the objective (Schulman et al., 2015; Peng et al., 2019), we can formulate the following constrained policy optimization problem:

$$
\begin{array}{l} \underset {\pi_ {\theta}} {\arg \max} \int_ {\boldsymbol {x}} \mathcal {D} (\boldsymbol {x}) \int_ {\boldsymbol {y}} \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) r (\boldsymbol {y} | \boldsymbol {x}) d \boldsymbol {y} d \boldsymbol {x} \\ \text {s.t.} \quad \int_ {\boldsymbol {x}} \mathcal {D} (\boldsymbol {x}) D _ {\mathrm{KL}} \left(\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) | | \pi_ {\mathcal {D}} (\boldsymbol {y} | \boldsymbol {x})\right) d \boldsymbol {x} \leq \epsilon , \end{array}\tag{5}
$$

where $\pi _ { \mathcal { D } }$ is the behavior policy induced by D. The constraint in Equation 5 ensures that the new policy $\pi _ { \theta }$ is close to the data distribution of $\pi _ { \mathcal { D } }$ , and therefore the surrogate objective remains a reasonable approximation.

Forming the Lagrangian of the constrained optimization problem presented above, we obtain the loss function:

$$
\begin{array}{l} \mathcal {L} _ {\theta} = \int_ {\boldsymbol {x}} \mathcal {D} (\boldsymbol {x}) \int_ {\boldsymbol {y}} \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) r (\boldsymbol {y} | \boldsymbol {x}) d \boldsymbol {y} d \boldsymbol {x} \\ \qquad + \beta \left(\int_ {\boldsymbol {x}} \mathcal {D} (\boldsymbol {x}) \mathrm{D} _ {\mathrm{KL}} \left(\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) | | \pi_ {\mathcal {D}} (\boldsymbol {y} | \boldsymbol {x})\right) d \boldsymbol {x}\right), \end{array}\tag{6}
$$

where $\beta$ is a Lagrange multiplier. Upon differentiating the objective function $\mathcal { L } ( \pi , \beta )$ with respect to $\pi _ { \boldsymbol { \theta } } ( \pmb { y } | \pmb { x } )$ and subsequently solving for the optimal policy $\pi ^ { \star }$ , the resultant expression for the optimal policy is as follows:

$$
\pi^ {\star} (\boldsymbol {y} | \boldsymbol {x}) = \frac {1}{Z (\boldsymbol {x})} \pi_ {\mathcal {D}} (\boldsymbol {y} | \boldsymbol {x}) \exp \left(\frac {1}{\beta} (r (\boldsymbol {y} | \boldsymbol {x})\right),\tag{7}
$$

where

$$
Z (\boldsymbol {x}) = \sum_ {\boldsymbol {y}} \pi_ {\mathcal {D}} (\boldsymbol {y} | \boldsymbol {x}) \exp \left(\frac {1}{\beta} (r (\boldsymbol {y} | \boldsymbol {x})\right)
$$

is the partition function or normalizing constant. Following (Korbak et al., 2022; Go et al., 2023), we utilize the reverse KL divergence between $\pi _ { \theta }$ and $\pi ^ { \star }$ for distribution matching:

$$
\begin{array}{r l} & D _ {\mathrm{KL}} (\pi_ {\theta}, \pi^ {\star}) = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})} \log \frac {\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})}{\pi^ {\star} (\boldsymbol {y} | \boldsymbol {x})} \\ & \qquad = - \frac {1}{\beta} \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})} (r (\boldsymbol {y} | \boldsymbol {x}) \\ & \qquad \qquad - \beta \log \frac {\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})}{\pi_ {\mathcal {D}} (\boldsymbol {y} | \boldsymbol {x})} - \beta \log Z (\boldsymbol {x})), \end{array}\tag{8}
$$

Following the analysis of previous works (Peng et al., 2019; Zhu et al., 2023), the partition function $Z ( { \pmb x } ) \approx 1$ . According to Equation 8, minimizing $D _ { \mathrm { K L } } ( \pi _ { \theta } , \pi ^ { \star } )$ coincides with the objective:

$$
\begin{array}{c} \underset {\pi_ {\theta}} {\arg \max} \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})} [ r (\boldsymbol {y} | \boldsymbol {x}) - \\ \beta \log (\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) / \pi_ {\mathcal {D}} (\boldsymbol {y} | \boldsymbol {x})) ]. \end{array}\tag{9}
$$

We note that $\pi _ { \mathcal { D } }$ is intractable to obtain, as the generation of D can be diverse, e.g., by either $\pi ^ { \mathrm { S F T } }$ , powerful LLMs like GPT-4, or humans. Therefore, the distribution of the behavior policy $\pi _ { D }$ is not accessible. Since $\pi ^ { \mathrm { S F T } }$ has been fine-tuned on part of D, we can approximate $\pi _ { \mathcal { D } }$ with $\pi ^ { \mathrm { S F T } }$ and then obtain the objective as in Equation 2.

Considering a low-quality answer y, even if its generation probability is small for a satisfactory policy model 7, we may sample such y during RL training. In this case, the KL penalty in Equation 2 becomes weaker or even negative, which would cause overoptimization. This problem would be exacerbated when the RM wrongly assigns high rewards for such OOD low-quality samples.

Trained on $\mathcal { D } ,$ reward models should be well-calibrated and be greatly uncertain for OOD (x, y) samples, which correspond to small $\pi _ { \boldsymbol { D } } ( \pmb { y } | \pmb { x } )$ . Given an answer y generated by $\pi _ { \boldsymbol { \theta } } ( \pmb { y } | \pmb { x } )$ , the more OOD the sample is, the larger the penalty term should be. Therefore, we can approximate the intractable term in 9 with the uncertainty estimation of reward models $u ( \pmb { y } | \pmb { x } )$ , which induces the following objectives:

$$
\begin{array}{c} \arg \max _ {\pi_ {\theta}} \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})} [ r (\boldsymbol {y} | \boldsymbol {x}) - \\ \beta_ {1} \log (\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) / \pi_ {\mathrm{SFT}} (\boldsymbol {y} | \boldsymbol {x})) - \beta_ {2} u (\boldsymbol {y} | \boldsymbol {x}) ], \end{array}\tag{10}
$$

where $\beta _ { 1 }$ and $\beta _ { 2 }$ are coefficients to control the KL and uncertainty regularization respectively.

## 3.2. Training Diverse Reward LoRA Ensembles

To estimate the reward uncertainty $u ( \pmb { y } | \pmb { x } )$ , we investigate the ensemble approach, which is widely adopted for enhancing the uncertainty of deep learning methods. Since reward models (RM) are also initialized from LLMs, we train multiple LoRAs instead of reward models for ensembles, which is more parameter-effective. Then the forward pass can be formulated as:

$$
\begin{array}{c} z ^ {o u t} = \frac {1}{N} \sum_ {n = 1} ^ {N} (W ^ {(0)} x + \Delta W _ {n} z ^ {i n}) \\ = \frac {1}{N} \sum_ {n = 1} ^ {N} (W ^ {(0)} x + B _ {n} A _ {n} z ^ {i n}), \end{array}\tag{11}
$$

where $\Delta W _ { n }$ are different LoRAs of the ensemble. Though LoRA-ensemble members have random initialization, we observe that LoRA ensembles can not exhibit satisfactory uncertainty quantification abilities We hypothesize this is due to a lack of diversity between LoRA ensembles. Recall that LoRA only learns parameter-update, the output of different ensemble members can be more homogeneous compared to traditional deep ensembles. Similar phenomena are also observed in other fine-tuning methods of LLMs’ ensembles (Gleave & Irving, 2022; Eisenstein et al., 2023).

Figure 2. Illustration of training diverse reward LoRA ensembles.

To actively diversify reward LoRA ensembles, we propose a diversity regularization via Nuclear Norm Maximization when training LoRA ensembles. As shown in Figure 2, we first concatenate multiple $A _ { n }$ along the LoRA dimension r and obtain matrix $A \in \mathbb { R } ^ { N r \times d }$ . If LoRA-ensemble members are totally homogeneous, the rank of A equals the rank of LoRA member $A _ { n } .$ . On the contrary, diverse members mean linearly independent along the first dimension of A. Therefore, we could measure the diversity (or the homogeneity) of the LoRA ensemble with the matrix rank of the matrix A. Since the rank optimization problem is known to be NP-hard, we leverage the convex surrogate, nuclear norm, as a computationally efficient approximation of matrix rank, which is calculated via singular value decomposition (SVD). In addition to the rank loss in Equation 1, the loss function of training diverse reward LoRA Ensemble is:

$$
\begin{array}{l} \mathcal {L} ^ {R M} = \underbrace {\sum_ {\boldsymbol {x}} \log \sigma \left(\frac {1}{N} \sum_ {n = 1} ^ {N} r _ {n} (\boldsymbol {y} ^ {w} | \boldsymbol {x}) - \frac {1}{N} \sum_ {n = 1} ^ {N} r _ {n} r (\boldsymbol {y} ^ {l} | \boldsymbol {x})\right)} _ {\text {Rank loss}} \\ + \underbrace {\lambda \frac {1}{M} \sum_ {m} ^ {M} \| A \| _ {*} / \| A \| _ {F}} _ {\text {Diversity regularization}}, \end{array} \tag {12}
$$

where λ is the NNM weight to control the diversity loss, $\| A \|$ <sub>∗</sub> is the nuclear norm of A, and $\| A \| _ { F }$ is the Frobenius norm of A, which is used to control the value of weights not to be too large.

After training reward models with the diverse LoRA ensemble, we can estimate the reward uncertainty using the standard deviation:

$$
u (\boldsymbol {y} | \boldsymbol {x}) = \sqrt {\frac {1}{N} \sum_ {n = 1} ^ {N} \left(r _ {n} (\boldsymbol {y} | \boldsymbol {x}) - \frac {1}{N} \sum_ {n = 1} ^ {N} r _ {n} (\boldsymbol {y} | \boldsymbol {x})\right) ^ {2}}.\tag{13}
$$

## 3.3. Overall Optimization Objectives

In Equation 10, three scalars including reward, KL penalty, and uncertainty penalty are to be optimized with the RL objective. To prevent the three terms from interfering with each other, we make the KL regularization independent of the actor loss. Specifically, we only optimize the uncertainty penalized rewards using RL algorithms:

$$
\begin{array}{c} \mathcal {J} _ {\theta} ^ {R L} = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})} \big [ r (\boldsymbol {y} | \boldsymbol {x}) \\ - \beta_ {2} \big (u (\boldsymbol {y} | \boldsymbol {x}) - \bar {u} (\boldsymbol {y} | \boldsymbol {x}) \big) \big ], \end{array}\tag{14}
$$

where $\bar { u } ( \boldsymbol { y } | \boldsymbol { x } )$ represent the uncertainty of rewards models for $( { \pmb x } , { \pmb y } )$ due to the different scales of ensemble members. In practice, we use the mean uncertainty of all previously seen samples to approximate $\bar { u } ( \boldsymbol { y } | \boldsymbol { x } )$

For KL regularization, the objective is:

$$
\mathcal {J} _ {\theta} ^ {K L} = - \beta_ {1} \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}} \mathbb {E} _ {\boldsymbol {y} \sim \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})} [ (\log \frac {\pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x})}{\pi_ {\mathrm{SFT}} (\boldsymbol {y} | \boldsymbol {x})}) ^ {2} ],\tag{15}
$$

where we utilize the KL estimator with lower variance, low bias, and positive assurance. Since the objective 15 is differentiable, we directly optimize it via gradient descent. Overall, the objective of UP-RLHF is as:

$$
\mathcal {J} _ {\theta} ^ {\mathrm{UP-RLHF}} = \mathcal {J} _ {\theta} ^ {\mathrm{RL}} + \mathcal {J} _ {\theta} ^ {\mathrm{KL}}.\tag{16}
$$

The KL regularization can be seen as the regularization from step 1 of the RLHF pipeline, while the uncertainty penalty can be seen as the regularization from step 2.

## 4. Experimental Results

In this section, we conduct empirical experiments to evaluate the alignment of UP-RLHF on two extensively utilized RLHF tasks, namely summarization and questionanswering. We aim to investigate three primary research questions (RQs):

• RQ1 (Step 2: Reward modeling): How well does diverse reward LoRA Ensemble improve the uncertainty quantification of reward models?

• RQ2 (Step 3: RL Fine-Tuning): How well does uncertainty penalization mitigate the overoptimization issue?

• RQ3 (Performance): How does UP-RLHF perform compared to existing RLHF methods?

To answer the above questions, we will first provide a concise introduction to the datasets and training setups. The subsequent discussion includes evaluations of both reward models and policy models.

## 4.1. Datasets and Training Setups

Datasets. For the summarization task, we employ the “TL;DR” (Too Long; Didn’t Read) dataset introduced by Volske et al. (2017). In this dataset,¨ x represents a forum post sourced from Reddit, and y corresponds to the respective summary. Notably, we use the gold reward to relabel the dataset in terms of preference, ensuring that the gold reward is the perfect proxy for the relabeled dataset.

In the question-answering task, following prior work, we use the Anthropic Helpful dataset (Bai et al., 2022b) with human preference without additional relabeling. x signifies a fragment of a conversation involving interactions between a human and a digital assistant. The model is specifically trained to generate the helpful subsequent turn of the assistant, denoted as y.

Training Setups. In the summarization task, the policy model is established using OPT-1.3B (Zhang et al., 2022), and the reward model is established using OPT-350m. In the question-answering task, both the policy model and the reward model are established using Llama2-7B (Touvron et al., 2023).

According to the scaling law of the reward model (Gao et al., 2023), RMs with larger parameters and more training data are more robust to optimization. Therefore, we use fine-tuned GPT-J-6B <sup>1</sup> as the gold reward model in the summarization task because of its larger RM parameter size and satisfactory accuracy (75% on the test set). In the context of the question-answering task, 3B SteamSHP-XL <sup>2</sup> is chosen as the gold reward model because of its larger RM training data size than the reward model, which is fine-tuned on both the HH and SHP (Ethayarajh et al., 2022) datasets.

Following (Yao et al., 2023), for both tasks, we perform a random partitioning for the datasets into three segments: 20% for step 1, 40% for step 2, and the remaining 40% for step 3.

## 4.2. Reward Model Evaluation

To study the uncertainty quantification ability of the reward model, we study ECE (Naeini et al., 2015), which is a metric used to assess model miscalibration. It involves binning assigned probability scores and comparing them to the average accuracies within these bins. Following the Bradley–Terry model, the probability score of preferring an answer $\pmb { y } ^ { w }$ over $\boldsymbol { y } ^ { l }$ can be calculated as:

$$
\begin{array}{c} P (\boldsymbol {y} ^ {w} > \boldsymbol {y} ^ {l} | \boldsymbol {x}) = \frac {\exp (r (\boldsymbol {y} ^ {w} | \boldsymbol {x}))}{\exp (r (\boldsymbol {y} ^ {w} | \boldsymbol {x})) + \exp (r (\boldsymbol {y} ^ {l} | \boldsymbol {x}))} \\ = \frac {1}{1 + \exp (r (\boldsymbol {y} ^ {w} | \boldsymbol {x}) - r (\boldsymbol {y} ^ {l} | \boldsymbol {x}))} \end{array}\tag{17}
$$

Then we can define the Expected Calibration Error (ECE) for the reward model:

$$
\mathrm{ECE} = \sum_ {m} ^ {M} \frac {\left| B _ {m} \right|}{\sum_ {m} \left| B _ {m} \right|} \left| \operatorname{ACC} \left(B _ {m}\right) - \operatorname{CONF} \left(B _ {m}\right) \right|,\tag{18}
$$

where we divide samples into M = 15 bins, $B _ { m } ,$ according to the reward difference, and

$$
\begin{array}{c} \operatorname{ACC} (B _ {m}) = | B _ {m} | ^ {- 1} \sum_ {i \in B _ {m}} \mathbb {I} [ r (\boldsymbol {y} _ {i} ^ {w} | \boldsymbol {x}) > r (\boldsymbol {y} _ {i} ^ {l} | \boldsymbol {x}) ], \\ \operatorname{CONF} (B _ {m}) = | B _ {m} | ^ {- 1} \sum_ {i \in B _ {m}} P (\boldsymbol {y} _ {i} ^ {w} > \boldsymbol {y} _ {i} ^ {l} | \boldsymbol {x}), \end{array}\tag{19}
$$

where I is the indicator function. We observe that different reward models have different reward scales. To calculate ECE, we scale reward differences to ensure that the largest reward difference in the test dataset corresponds to 0.99 confidence, which induces the calibrated ACC.

We establish reward models using OPT-330M on TL;DR and using Llama2-7B on the Anthropic Helpful dataset. Table 1 details the performance of reward models with different training methods and it can be observed that LoRA Ensemble benefits both accuracy and ECE on the test dataset. Utilizing NNM, the overall performance in terms of the two metrics can be further improved.

We use two reward models, which are trained with LoRA ensemble and diverse LoRA ensemble to train the policy model respectively utilizing the RLHF objective 2.

Table 1. Accuracy and ECE of different training methods for reward modeling on two datasets. The best-performing values are highlighted. All ensemble methods have 5 members.

<table><tr><td>Base Model</td><td>Training Method</td><td>ACC ↑</td><td>ECE ↓</td></tr><tr><td rowspan="3">OPT-330M</td><td>Full FT</td><td>0.694</td><td>0.485</td></tr><tr><td>LoRA Ensemble</td><td>0.697</td><td>0.480</td></tr><tr><td>Diverse LoRA Ensemble</td><td>0.697</td><td>0.481</td></tr><tr><td rowspan="3">Llama2-7B</td><td>Full FT</td><td>0.685</td><td>0.515</td></tr><tr><td>LoRA Ensemble</td><td>0.710</td><td>0.496</td></tr><tr><td>Diverse LoRA Ensemble</td><td>0.720</td><td>0.485</td></tr></table>

(a)

(b)  
Figure 3. With diversity regularization, our proposed diverse reward LoRA ensemble achieves better OOD detection capabilities.

Following (Gao et al., 2023), we utilize the KL divergence between the policy model and the SFT model $\mathrm { D } _ { \mathrm { K L } } \left( \pi _ { \boldsymbol { \theta } } ( \pmb { y } | \pmb { x } ) | | \pi _ { \mathrm { S F T } } ( \pmb { y } | \pmb { x } ) \right)$ to measure the degree of policy optimization. As shown in Figure 3, the uncertainty provided by the reward LoRA ensemble grows rapidly in the range of KL divergence from 0 to 50, which makes it difficult to distinguish between samples with high gold rewards and samples generated by over-optimized models (KL divergence roughly from 50 to 100). On the contrary, our proposed diverse reward LoRA ensemble provides gradually increased uncertainty as the optimization process, indicating better OOD detection capabilities.

## 4.3. Effect of Uncertainty Penalty

Even with diverse reward LoRA ensembles, we observe significant overoptimization during the optimization process to the mean reward of the ensembles, as shown in Figure 4(a). When incorporating uncertainty penalties into rewards, the uncertainty of generated samples is well-controlled within a reasonable range, and the overoptimization issue is eliminated. This demonstrates the effectiveness of uncertainty regularization in mitigating overoptimization.

Interestingly, we observe that though utilizing uncertainty regularization can improve the overall performance in terms of gold RM, the RM score is diminished. This may be because uncertainty-penalized rewards limit the exploration of OOD output by the policy model, whether these outputs are high-quality or low-quality. In this case, using additional uncertainty regularization may restrict the exploration of policy models, which corresponds to the exploration-exploitation dilemma in RL.

(a) Dashed lines represent RM scores and solid lines represent gold RM scores.

(b) Reward uncertainty of LoRA ensembles.  
Figure 4. Uncertainty penalty ablation on policy model evaluation in the summarization task over 4 different seeds.

## 4.4. Policy Model Evaluation

In this section, we compare our proposed UP-RLHF with existing RLHF methods in both summarization and questionanswering tasks. We compare gold RM scores instead of RM scores because different RMs have different scaling, thus making no sense to compare RM scores directly.

As shown in Figure 5, UP-RLHF outperforms RLHF in terms of gold performance with a large margin in both tasks. Especially in the summarization task, compared to RLHF, UP-RLHF can achieve higher performance with less KL divergence cost. Note that the RLHF method utilized the full fine-tuning for reward modeling, while our diverse reward LoRA ensemble in UP-RLHF only fine-tunes 4.53% parameters for OPT-350M and 1.25% parameters for Llama2-7B.

(a) OPT-1.3B in the summarization task over 4 seeds.

(b) OPT-1.3B in the summarization task over 4 seeds.

(c) Llama2-7B in the questionanswering task.

(d) Llama2-7B in the questionanswering task.  
Figure 5. Comparison of UP-RLHF and RLHF.

## 5. Related Works

## 5.1. Reinforcement Learning from Human Feedback

RLHF is a pivotal approach for fine-tuning language models to align with human preferences. Researchers have applied RLHF to diverse tasks (Ramamurthy et al., 2023) such as text summarization (Stiennon et al., 2020) and enhancing the harmlessness and helpfulness of language models (Ba et al., 2022b). Notably, InstructGPT introduces the threestep RLHF pipeline using a supervised approach and the PPO algorithm (Schulman et al., 2017), demonstrating its effectiveness on ChatGPT. While successful, RLHF faces various challenges (Casper et al., 2023). One of the most pressing challenges is overoptimization, which is caused by imperfect RMs (Gao et al., 2023). The author in (Gao et al., 2023) provides the scaling law of RMs, which shows the effect of increasing RM parameters and data size in mitigating the issue.

RLHF heavily relies on reward modeling to proxy human preferences. Some recent works aim to bypass the reward modeling step (Yuan et al., 2023; Rafailov et al., 2023; Song et al., 2023). Specifically, DPO directly optimizes the policy towards the objective 2 by solving a classification problem on the human preference data. Although bypassing the reward modeling step benefits from easy implementation and training stability, more recent works reveal several advantages of using reward models. (Azar et al., 2023) analyzes the robustness of reward-model-based methods against overfitting caused by the weakness of the KL regularization. Besides, compared to DPO, reward-model-based RLHF shows great advantages on out-of-preference samples (Li et al., 2023b;a).

There are many works to address the challenge in RLHF such as computational overhead (Li et al., 2023b), sample efficiency (Snell et al., 2023; Gulcehre et al., 2023), unstable training (Wu et al., 2023), and overoptimization (Moskovitz et al., 2023; Coste et al., 2023; Eisenstein et al., 2023). We also focus on the overoptimization issue. While most recent works focus only on the RL fine-tuning step, we first introduce uncertainty quantification to the reward modeling step and make the RL fine-tuning uncertainty aware.

## 5.2. Uncertainty Aware Reinforcement Learning

Uncertainty is a pivotal factor in the realm of RL. The Optimism in the Face of Uncertainty (OFU) principle (Abbasi-Yadkori et al., 2011) in online RL strategies is widely adopted for facilitating active and efficient exploration of the environment (Lockwood & Si, 2022). In offline RL (Levine et al., 2020), uncertainty is typically utilized for conservative to control the prediction errors caused by imperfect dynamics models. Uncertainty is usually estimated by value networks in model-free RL (Pathak et al., 2019; Bai et al.,

2022a) and by dynamics models in model-based RL (Janner et al., 2019; Yu et al., 2020).

RLHF can be formulated as reverse RL with offline datasets, where reward models trained on an offline limited preference dataset are imperfect. Inspired by recent model-based offline RL methods (Yu et al., 2020; Kidambi et al., 2020; Lu et al., 2022), we propose to penalize rewards with the model uncertainty for conservative policy optimization, aiming for mitigating the overoptimization issue. Concurrent work by (Coste et al., 2023; Eisenstein et al., 2023) also shows reward model ensemble helps mitigate overoptimization. However, utilizing reward model ensembles increases RM parameters several times, and may lack diversity between ensemble members (Gleave & Irving, 2022). To diversify reward ensembles, (Eisenstein et al., 2023) propose to use different seeds in the pre-training phase. We propose to train diverse LoRA ensembles with NNM for reward modeling, which is much cheaper and parameter-effective. Besides, we analyze the relations between KL and uncertainty regularization and make them affect independently.

## 5.3. Uncertainty for LLMs

Uncertainty quantification for deep neural networks has been well studied (Gawlikowski et al., 2023). Popular methods include deep ensemble, MC dropout (Gal & Ghahramani, 2016), and so on. In the context of LLMs, some new challenges arise. Diversity plays an important role in ensemble-based methods (Breiman, 2001). However, finetuning LLMs for ensembles (Sun et al., 2022) not only is too expensive to scale up but also lacks diversity (Gleave & Irving, 2022). Therefore, we adopt a popular PEFT technology, LoRA (Hu et al., 2022) for training the ensemble of reward models. Different from the concurrent work (Wang et al., 2023) which also proposes LoRA ensemble for LLMs fine-tuning and different regularization techniques for each LoRA, we propose a diversity regularization to encourage diversity between ensemble members. Besides, we mainly focus on the reward modeling in the context of RLHF.

## 6. Conclusion and Limitations

In this paper, we propose UP-RLHF, an uncertainty-aware RLHF framework that contributes to the uncertainty of AI systems based on LLMs. Our proposed diverse reward LoRA ensemble can provide satisfactory uncertainty quantification for samples in RLHF. Leveraging the reward uncertainty, we highlight the pivotal role of uncertainty regularization in effectively addressing the overoptimization challenge in the alignment of LLMs.

Our work has limitations. While the diverse reward LoRA ensemble proves to be parameter-effective, the computation of the nuclear norm for concatenated LoRA matrices introduces additional time overhead. Moreover, uncertainty regularization may exhibit over-conservatism, particularly in cases involving near-distribution high-quality outputs. As a future direction, exploring methods to strike a balance between KL and uncertainty regularization for specific samples could further refine the framework’s performance.

## References

Abbasi-Yadkori, Y., Pal, D., and Szepesv ´ ari, C. Improved´ algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011.

Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., and Munos, R. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036, 2023.

Bai, C., Wang, L., Yang, Z., Deng, Z., Garg, A., Liu, P., and Wang, Z. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. International Conference on Learning Representations, 2022a.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022b.

Beeching, E., Belkada, Y., Rasul, K., Tunstall, L., von Werra, L., Rajani, N., and Lambert, N. Stackllama: An rl fine-tuned llama model for stack exchange question and answering, 2023. URL https://huggingface. co/blog/stackllama, 1, 2023.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Breiman, L. Random forests. Machine learning, 45:5–32, 2001.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Coste, T., Anwar, U., Kirk, R., and Krueger, D. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Ethayarajh, K., Choi, Y., and Swayamdipta, S. Understanding dataset difficulty with V-usable information. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 5988–6008. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/ethayarajh22a.html.

Gal, Y. and Ghahramani, Z. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016.

Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Gawlikowski, J., Tassi, C. R. N., Ali, M., Lee, J., Humt, M., Feng, J., Kruspe, A., Triebel, R., Jung, P., Roscher, R., et al. A survey of uncertainty in deep neural networks. Artificial Intelligence Review, 56(Suppl 1):1513–1589, 2023.

Gleave, A. and Irving, G. Uncertainty estimation for language reward models. arXiv preprint arXiv:2203.07472, 2022.

Go, D., Korbak, T., Kruszewski, G., Rozen, J., Ryu, N., and Dymetman, M. Aligning language models with preferences through f-divergence minimization. In International conference on machine learning. PMLR, 2023.

Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Janner, M., Fu, J., Zhang, M., and Levine, S. When to trust your model: Model-based policy optimization. Advances in Neural Information Processing Systems, 32, 2019.

Kidambi, R., Rajeswaran, A., Netrapalli, P., and Joachims, T. Morel: Model-based offline reinforcement learning. Advances in neural information processing systems, 33: 21810–21823, 2020.

Korbak, T., Elsahar, H., Kruszewski, G., and Dymetman, M. On reinforcement learning and distribution matching for fine-tuning language models with no catastrophic forgetting. Advances in Neural Information Processing Systems, 35:16203–16220, 2022.

Kreps, S., McCain, R. M., and Brundage, M. All the news that’s fit to fabricate: Ai-generated text as a tool of media misinformation. Journal ofexperimental political science, 9(1):104–117, 2022.

Levine, S., Kumar, A., Tucker, G., and Fu, J. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Li, Z., Xu, T., and Yu, Y. Policy optimization in rlhf: The impact of out-of-preference data. arXiv preprint arXiv:2312.10584, 2023a.

Li, Z., Xu, T., Zhang, Y., Yu, Y., Sun, R., and Luo, Z. Remax: A simple, effective, and efficient method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023b.

Lockwood, O. and Si, M. A review of uncertainty for deep reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, volume 18, pp. 155–162, 2022.

Lu, C., Ball, P., Parker-Holder, J., Osborne, M., and Roberts, S. J. Revisiting design choices in offline model based reinforcement learning. In International Conference on Learning Representations, 2022.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pp. 1928– 1937. PMLR, 2016.

Moskovitz, T., Singh, A. K., Strouse, D., Sandholm, T., Salakhutdinov, R., Dragan, A. D., and McAleer, S. Confronting reward model overoptimization with constrained rlhf. arXiv preprint arXiv:2310.04373, 2023.

Naeini, M. P., Cooper, G., and Hauskrecht, M. Obtaining well calibrated probabilities using bayesian binning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 29, 2015.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Pathak, D., Gandhi, D., and Gupta, A. Self-supervised exploration via disagreement. In International Conference on Machine Learning, pp. 5062–5071. PMLR, 2019.

Peng, X., Kumar, A., Zhang, G., and Levine, S. Advantageweighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N., and Irving, G. Red teaming language models with language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 3419–3448, 2022.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 2023.

Ramamurthy, R., Ammanabrolu, P., Brantley, K., Hessel, J., Sifa, R., Bauckhage, C., Hajishirzi, H., and Choi, Y. Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization. In The Eleventh International Conference on Learning Representations, 2023.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International conference on machine learning, pp. 1889–1897. PMLR, 2015.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Snell, C. V., Kostrikov, I., Su, Y., Yang, S., and Levine, S. Offline rl for natural language generation with implicit language q learning. In The Eleventh International Conference on Learning Representations, 2023.

Song, F., Yu, B., Li, M., Yu, H., Huang, F., Li, Y., and Wang, H. Preference ranking optimization for human alignment. arXiv preprint arXiv:2306.17492, 2023.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33: 3008–3021, 2020.

Sun, M., Yan, W., Abbeel, P., and Mordatch, I. Quantifying uncertainty in foundation models via ensembles. In NeurIPS 2022 Workshop on Robustness in Sequence Modeling, 2022.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Wang, X., Aitchison, L., and Rudolph, M. Lora ensembles for large language model fine-tuning. arXiv preprint arXiv:2310.00035, 2023.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

Wu, T., Zhu, B., Zhang, R., Wen, Z., Ramchandran, K., and Jiao, J. Pairwise proximal policy optimization: Harnessing relative feedback for llm alignment. arXiv preprint arXiv:2310.00212, 2023.

Yang, A., Xiao, B., Wang, B., Zhang, B., Bian, C., Yin, C., Lv, C., Pan, D., Wang, D., Yan, D., et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.

Yao, Z., Aminabadi, R. Y., Ruwase, O., Rajbhandari, S., Wu, X., Awan, A. A., Rasley, J., Zhang, M., Li, C., Holmes, C., et al. Deepspeed-chat: Easy, fast and affordable rlhf training of chatgpt-like models at all scales. arXiv preprint arXiv:2308.01320, 2023.

Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J. Y., Levine, S., Finn, C., and Ma, T. Mopo: Model-based offline policy optimization. Advances in Neural Information Processing Systems, 33:14129–14142, 2020.

Yuan, Z., Yuan, H., Tan, C., Wang, W., Huang, S., and Huang, F. Rrhf: Rank responses to align language models with human feedback without tears. Advances in neural information processing systems, 2023.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Zhu, B., Sharma, H., Frujeri, F. V., Dong, S., Zhu, C., Jordan, M. I., and Jiao, J. Fine-tuning language models with advantage-induced policy alignment. arXiv preprint arXiv:2306.02231, 2023.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
