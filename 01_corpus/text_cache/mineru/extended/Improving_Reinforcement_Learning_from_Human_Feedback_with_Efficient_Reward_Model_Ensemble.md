# Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble

Shun Zhang<sup>1</sup>, Zhenfang Chen<sup>1</sup>, Sunli Chen<sup>2</sup>, Yikang Shen<sup>1</sup>, Zhiqing Sun<sup>3</sup>, and Chuang Gan<sup>1,4</sup> <sup>1</sup>MIT-IBM Watson AI Lab, <sup>2</sup>Tsinghua University, <sup>3</sup>Carnegie Mellon University, <sup>4</sup>UMass Amherst

## Abstract

Reinforcement Learning from Human Feedback (RLHF) is a widely adopted approach for aligning large language models with human values. However, RLHF relies on a reward model that is trained with a limited amount of human preference data, which could lead to inaccurate predictions. As a result, RLHF may produce outputs that are misaligned with human values. To mitigate this issue, we contribute a reward ensemble method that allows the reward model to make more accurate predictions. As using an ensemble of large language model-based reward models can be computationally and resource-expensive, we explore efficient ensemble methods including linear-layer ensemble and LoRA-based ensemble. Empirically, we run Best-of-n and Proximal Policy Optimization with our ensembled reward mod els, and verify that our ensemble methods help improve the alignment performance of RLHF outputs.

## 1 Introduction

Large language models (LLMs) (Vaswani et al., 2017) have become prominent in the field of artificial intelligence in solving a wide range of complex tasks in question answering (Ouyang et al., 2022; Guu et al., 2020), code generation (Li et al., 2022; Nijkamp et al., 2023), reasoning and planning (Kojima et al., 2023; Liu et al., 2023), and various other domains. However, as large language models are trained using data from various sources, they may generate outputs that are biased, inappropriate, or even harmful, which are misaligned with human values. Therefore, it is crucial to align large language models with human values for them to be safely deployed.

Recently, Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022) has shown to be a promising approach to mitigate the misalignment issue. Concretely, RLHF first runs supervised fine-tuning (SFT) to train a large language model to generate responses that follow instructions. It then trains a reward model using a preference dataset that reflects human values. Lastly, it runs reinforcement learning using the learned reward model to finetune the SFT model. In this way, the finetuned model would generate sequences with higher rewards, which are presumably more aligned with human values.

The RLHF algorithm has demonstrated success in improving the alignment of large language models (Dubois et al., 2023; Wu et al., 2023). However, it also has some recognized issues. As the reward model is trained on offline-collected preference data, its reward predictions may not be accurate on out-of-distribution data. If we use reinforcement learning to optimize a large language model with an inaccurate reward model, it may generate misaligned outputs with incorrectly high estimated rewards. This problem is observed in the value alignment literature and is usually known as reward hacking or reward overoptimization (Amodei et al., 2016; Leike et al., 2017; Gao et al., 2022; Casper et al., 2023), and is also a well-known problem in model-based offline reinforcement learning (Levine et al., 2020).

In offline reinforcement learning, a prevalent strategy to mitigate reward overoptimization is to estimate rewards conservatively under uncertainty (Kumar et al., 2020). Building upon this strategy, we consider an ensemble approach that employs a set of reward models to make better predictions, in line with concurrent works (Coste et al., 2023; Eisenstein et al., 2023). To achieve reward model ensemble, a straightforward approach is to train multiple reward models independently and then ensemble them. However, this approach presents some challenges in our setting. As reward mod els are usually based on large language models, training all the reward models and then loading all of them during inference time can be computationally expensive and resource-consuming. Therefore, we contribute to designing efficient ensemble approaches. Lastly, we empirically confirm the effectiveness of our ensemble methods using well-accepted evaluation benchmarks, AlpacaEval (Dubois et al., 2023) and MT-Bench (Zheng et al., 2023). The contributions of this paper are summarized as follows.

• We design reward ensemble algorithms that improve reward estimation accuracy, and hence improve the alignment of large language models.

• We propose two ensemble approaches, linearlayer ensemble and LoRA-based ensemble, that offer trade-offs between computational efficiency and alignment performance.

• We use the ensembled reward models for both Best-of-n and Proximal Policy Optimization (PPO) algorithms, evaluate them on AlpacaEval and MT-Bench, and empirically confirm that RLHF with our ensembled reward models outperforms the standard RLHF algorithm.

## 2 Background and Related Work

Reinforcement learning from human feedback (RLHF). RLHF was originally considered in the TAMER framework (Knox and Stone, 2009), in which an agent learns the reward function from a human user’s positive or negative feedback. This setting is later considered in deep reinforcement learning (Christiano et al., 2017), and recently employed for finetuning large language models (Ouyang et al., 2022) to align the model’s behavior with human values and preferences. We overviewed the RLHF framework in the introduction and leave more details of the standard RLHF algorithm in Sec. A.1.

Ensemble models and uncertainty estimation. Model ensembling has been an accepted approach to improving a model’s accuracy and estimating a model’s uncertainty. Lakshminarayanan et al. (2017) quantify predictive uncertainty in deep neural networks using ensembling, which performs better than Bayesian neural networks. In reinforcement learning, Liang et al. (2022) use ensembled reward models to estimate the model’s uncertainty for more informed exploration. Gleave and Irving (2022) use ensembled reward models for active learning.

Concurrently, Coste et al. (2023) also use reward model ensemble to mitigate the reward model overoptimization problem and draw similar conclusions to our paper. The key difference is that our work focuses on developing efficient ensemble approaches, since ensembling multiple independently-trained reward models can be expensive. Eisenstein et al. (2023) focus on comparing ensembling during pretraining and finetuning. Zhai et al. (2023) also consider LoRA-based ensemble, while their work focuses on an uncertaintypenalized objective in RL-finetuning. Ramé et al. (2024) consider a different approach of averaging the weights of multiple reward models instead of ensembling their predictions. Wang et al. (2023) use LoRA ensembles for improving predictive accuracy and uncertainty quantification. Ahmed et al. (2024) propose a scalable ensemble approach that shares an encoder backbone but uses separate linear heads, achieving similar performance to full ensembling.

Offline reinforcement learning. Uncertainty estimation is also a key problem in offline reinforcement learning (Levine et al., 2020; Janner et al., 2019). For example, conservative Q-learning (CQL) (Kumar et al., 2020) learns a conservative Q function to mitigate the overestimation bias for out-of-distribution state, action pairs. Inspired by this algorithm, we also design an ensemble algo rithm that uses conservative predictions by using the lower confidence bound of the ensembled predictions.

## 3 Reward Model Ensemble

The reward model in RLHF is a (large language model-based) model that takes as input an instruction and a response, and outputs a reward prediction that indicates the alignment performance of the response. In this section, we provide a formal descirption for our ensemble algorithms.

## 3.1 Architecture Design of Reward Model Ensemble

In the literature, reward models are commonly finetuned from pretrained large language models. Following the convention (Dubois et al., 2023), we assume a reward model is a Transformer model and a linear layer, where the linear layer takes as input the last hidden layer of the Transformer model, and outputs a reward value.

In this subsection, we discuss possible ways to ensemble multiple large language model-based reward models and discuss their advantages. The ensemble algorithms are illustrated in Fig. 1 and the pseudocode is provided in Alg. 1.

Figure 1: Illustration of the reward model ensemble algorithms.

Ensemble of single reward models. A straightforward way to achieve reward model ensemble is to train multiple reward models independently using different random seeds. At inference time, we simply load all the reward models and ensemble them. However, such a method can be both expensive in training and inference – during training, we need to run reward training for multiple times; during inference, we need to load multiple large language model-based reward models simultaneously to GPUs, which can be resourceconsuming. Specifically, an ensemble of k independently trained reward models needs to train and load k(M+L) parameters, where M is the number of parameters of the Transformer model, and L is the number of parameters of a linear layer (shown in Fig. 1 (a)).

Linear-layer ensemble. To make the ensembling more efficient both in training and inference, we can make all the ensembled models share the same Transformer model, while each ensembled model has its own linear layer that outputs its reward prediction. During training, both the Transformer model and the linear layers of all the ensembled models are being trained. Note that the Transformer model is the same for all the ensembled reward models. In this way, a linear-layer ensemble model of size k only requires M + kL parameters (shown in Fig. 1 (b)).

LoRA-based ensemble. In linear-layer ensemble, allowing all the ensembled models to share the same Transformer model indeed reduces the total number of parameters requiring training. However, it may considerably limit the diversity of the ensembled models. Therefore, we allow each ensembled model to slightly finetune the Transformer model. Specifically, each ensembled model trains its own linear layer, and a LoRA adapter (Hu et al., 2021) that is added to the Transformer model. The LoRA adapter only has a small number of parameters and can be trained efficiently. (The background on LoRA is provided in Sec. A.2.) In this way, a LoRA-based ensemble model of size k requires M + kL + kA parameters, where A is the number of parameters of an adapter (shown in Fig. 1 (c)).

Empirically, we find that only LoRA-finetuning the Transformer model in the reward model does not perform well, as the Transformer model is not trained for reward prediction at all. So in our experiments, we first finetune the Transformer model in the same way as linear-layer ensemble using a subset of preference data before ensemble model training (Line 14 in Alg. 1). We then use the rest of the data for ensemble training. We provide more details on the dataset split in Sec. 4.

## 3.2 Predictions of Ensembled Reward Models

Now we need to ensemble the predictions of different ensembled reward models. We explore two algorithms for ensembling these predictions, which use the mean predicted value and the lower confidence bound of the predicted values, respectively.

Let R be the set of ensemble model predictions. $R = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { k } \}$ . Mean value prediction simply uses mean(R), the mean value of the ensemble reward model prediction. This is inherently a lower-variance estimation of the reward. On the other hand, lower confidence bound (LCB) is a conservative estimation of the reward. It considers the standard deviation of the ensemble model predictions, defined as $\mathrm { L C B } ( R ) = \operatorname { m e a n } ( R ) - \beta$ std(R), where $\beta$ is a hyperparameter. However, we empirically find that the performance of LCB is comparable to that of mean value prediction.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Reward Model Ensemble Algorithms
Require: k: number of ensemble models, M: parameters of Transformer model
1: Initialize ensemble as empty set: ensemble ← {}
2: ▶ Option 1: Ensemble of Single Reward Models
3: for i = 1 to k do
4:    $M_i \leftarrow \text{clone}(M)$
5:    Initialize linear layer with random parameters $L_i$
6:    $train(M_i \cup L_i)$
7:    Add model $M_i \cup L_i$ to ensemble
8: end for
9: ▶ Option 2: Linear-layer Ensemble
10: Initialize linear layer with random parameters $L_i$ for i ∈ [0, k)
11: Concurrently $train(M \cup L_i)$ for i ∈ [0, k)
12: Add $M \cup L_i$, i ∈ [0, k) to ensemble
13: ▶ Option 3: LoRA-based Ensemble
14: Finetune $M, L_i, \ldots, L_k$ using linear-layer ensemble with a subset of data
15: for i = 1 to k do
16:    Add LoRA adapter to the Transformer model M with random parameters $A_i$
17:    $train(A_i \cup L_i)$
18:    Add $M \cup A_i \cup L_i$ to ensemble
19: end for
20: return ensemble
</div>

## 4 Empirical Evaluation

In this section, we empirically answer the following questions. Q1: Compared with using a single reward model, do ensembled reward models help improve alignment performance in RLHF? Q2: Which ensemble architecture in Sec. 3.1 has the best performance? Q3: Which prediction algorithm in Sec. 3.2 has a better performance?

Algorithms. We consider using the ensembled reward models with Best-of-n and Proximal Policy Optimization (PPO) (Schulman et al., 2017), which are standard approaches in RLHF (Ouyang et al., 2022; Dubois et al., 2023; Coste et al., 2023). Specifically, Best-of-n generates n samples from the SFT model for each input and selects the sample with the highest predicted reward. Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that finetunes the SFT model using the reward model.

For each reward ensemble method, we conduct the experiments multiple times using different random seeds. Specifically, we repeat the experiments 10 times for Best-of-n and 5 times for PPO. For all experiments except experiments without reward ensemble, we use three reward models for ensembling (k = 3).

Models. We use the pretrained models provided in AlpacaFarm (Dubois et al., 2023). Specifically, we use SFT10k as the base model for generation, which is a Llama-7b model (Touvron et al., 2023) finetuned on the alpaca\_instructions dataset. So the model can follow instructions, while it has not been aligned with human preferences. To be consistent with Dubois et al. (2023), the Transformer model in our reward model is also initialized using SFT10k, which is not yet trained for reward prediction.

Datasets. We use the AlpacaFarm datasets (Dubois et al., 2023) for training and evaluation, which provide utilities to evaluate the alignment performance of the model outputs using GPT-4 APIs and can easily compare our models with other benchmarking models and algorithms. We train all the reward models using both alpaca\_noisy\_multi\_preference and alpaca\_human\_preference datasets, and use alpaca\_farm\_evaluation for evaluation.

Specifically, for LoRA-based ensemble, we use the two training datasets for different phases: We first use alpaca\_noisy\_multi\_preference to fully finetune the Transformer model with k linear layers, in the same way as linear-layer ensemble (Line 14 in Alg. 1), as we find fullyfinetuning the Transformer is necessary for it to make reward predictions. Then we use alpaca\_human\_preference for training the linear layers and the adapters for the ensemble members (Line 15-19 in Alg. 1).

Results. Our main results are reported in Figure 2. For Best-of-n, we choose n = 50, 100, 200. For PPO, we evaluate checkpoints at every 100 training steps. To evaluate the alignment performance, we use the win rate metric provided in AlpacaEval (shown as the vertical axis), which measures the chances that our methods’ outputs are preferred by GPT-4 compared with a GPT-3 generated baseline. All the ensemble approaches use the mean value prediction, which uses the mean of the predicted rewards of the ensemble members as their final predictions.

Overall, we find that the win rates are consistently higher when using reward method ensemble (answering Q1). For Best-of-n, both ensemble of single reward models and LoRA-based ensemble have the best performance. For PPO, we are unable to run ensemble of single reward models as it requires loading multiple reward models during

(a) Best-of-n results.

(b) PPO results.  
Single Reward Model Linear-layer Ensemble (Mean) LoRA-based Ensemble (Mean) Ensemble of Single Reward Models (Mean)

Figure 2: Win rates of model responses using Best-of-n and PPO on AlpacaEval. Different lines represent different reward ensemble algorithms. The shaded areas represent standard errors.

<table><tr><td>Ens. Method</td><td>First Turn</td><td>Second Turn</td><td>Average</td></tr><tr><td>Ens. of Single</td><td>4.70 ± 0.12</td><td>3.63 ± 0.22</td><td>4.16 ± 0.14</td></tr><tr><td>Linear-layer</td><td>4.73 ± 0.22</td><td>3.67 ± 0.19</td><td>4.20 ± 0.10</td></tr><tr><td>LoRA-based</td><td>4.86 ± 0.09</td><td>3.84 ± 0.21</td><td>4.35 ± 0.12</td></tr></table>

Table 1: Alignment scores on MT-Bench for different ensemble methods.

PPO training. Nonetheless, we find LoRA-based ensemble has the best performance.

We also conduct the experiments on MT-Bench (Zheng et al., 2023), which is a benchmark for multiturn questions. We evaluate our PPO-trained models with the most training steps, and report the alignment scores after the first turn, after the second turn, and the average of both. The results are reported in Table 1, and our findings are consistent with the AlpacaEval results. The results on both benchmarks suggest that, although LoRA does not fully finetune the Transformer models, it is effective for reward model ensemble and can improve the alignment performance (answering Q2).

In terms of the prediction methods, we find that the mean reward prediction and LCB have similar performance. Detailed results are presented in Sec. B (answering Q3).

## 5 Discussion and Conclusion

In summary, our paper presents a novel approach to enhancing the alignment of large language models through efficient reward model ensemble in RLHF. Specifically, the LoRA-based ensemble method demonstrates effectiveness under computational constraints. In future work, we will extend this approach to other steps of LLM training and inference, such as sample-efficient training of reward models (Gleave and Irving, 2022).

## References

Ahmed M. Ahmed, Rafael Rafailov, Stepan Sharkov, Xuechen Li, and Sanmi Koyejo. 2024. Scalable Ensembling For Mitigating Reward Overoptimisation. arXiv preprint arXiv:2406.01013.

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 2016. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Wang, Samuel Marks, Charbel-Raphaël Segerie, Micah Carroll, Andi Peng, Phillip Christoffersen, Mehul Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J. Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter Hase, Erdem Bıyık, Anca Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. 2023. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback. ArXiv:2307.15217 [cs].

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, pages 4299–4307.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. 2023. Reward Model Ensembles Help Mitigate Overoptimization. ArXiv:2310.02743 [cs].

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpaca-Farm: A Simulation Framework for Methods that Learn from Human Feedback.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, D. J. Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. 2023. Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking. ArXiv:2312.09244 [cs].

Leo Gao, John Schulman, and Jacob Hilton. 2022. Scaling Laws for Reward Model Overoptimization.

Adam Gleave and Geoffrey Irving. 2022. Uncertainty Estimation for Language Reward Models. ArXiv:2203.07472 [cs].

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: Retrieval-Augmented Language Model Pre-Training. ArXiv:2002.08909 [cs].

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. 2019. When to trust your model: Modelbased policy optimization. In Advances in Neural Information Processing Systems, pages 12519–12530.

W. Bradley Knox and Peter Stone. 2009. Interactively Shaping Agents via Human Reinforcement: The TAMER Framework. In Proceedings of the Fifth International Conference on Knowledge Capture, K-CAP ’09, pages 9–16, New York, NY, USA.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2023. Large Language Models are Zero-Shot Reasoners. ArXiv:2205.11916 [cs].

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. 2020. Conservative Q-Learning for Offline Reinforcement Learning. Neural Information Pro cessing Systems.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. ArXiv:1612.01474 [cs, stat].

Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, and Shane Legg. 2017. AI safety gridworlds. arXiv preprint arXiv:1711.09883.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. 2020. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. arXiv:2005.01643 [cs, stat]. ArXiv: 2005.01643.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, and Agustin Dal Lago. 2022. Competition-level code generation with alphacode. Science, 378(6624):1092–1097. Publisher: American Association for the Advancement of Science.

Xinran Liang, Katherine Shu, Kimin Lee, and Pieter Abbeel. 2022. Reward Uncertainty for Exploration in Preference-based Reinforcement Learning. ArXiv:2205.12401 [cs].

Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. 2023. LLM+P: Empowering Large Language Models with Optimal Planning Proficiency. ArXiv:2304.11477 [cs].

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2023. CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis. ArXiv:2203.13474 [cs].

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. ArXiv:2203.02155 [cs].

Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. 2024. WARM: On the Benefits of Weight Averaged Reward Models. ArXiv:2401.12187 [cs].

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs]. ArXiv: 1707.06347

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. arXiv:1706.03762 [cs]. ArXiv: 1706.03762.

Xi Wang, Laurence Aitchison, and Maja Rudolph. 2023. LoRA ensembles for large language model finetuning. arXiv preprint arXiv:2310.00035.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. 2023. Fine-Grained Human Feedback Gives Better Rewards for Language Model Training. ArXiv:2306.01693 [cs].

Yuanzhao Zhai, Han Zhang, Yu Lei, Yue Yu, Kele Xu, Dawei Feng, Bo Ding, and Huaimin Wang. 2023. Uncertainty-Penalized Reinforcement Learning from Human Feedback with Diverse Reward LoRA Ensembles. ArXiv:2401.00243 [cs].

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, and Eric Xing. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. arXiv preprint arXiv:2306.05685.

## A Preliminaries

For the completeness of the paper, we provide more background details on reinforcement learning from human feedback and LoRA finetuning in this section.

## A.1 Reinforcement Learning from Human Feedback

Reinforcement learning from human feedback (RLHF) follows a three-step process: supervised fine-tuning (SFT), reward modeling, and reinforcement learning using the learned reward model.

In this paper, we focus on the second step of RLHF, which trains a reward model that re flects human preferences. We denote the reward model by $r _ { \theta }$ , parameterized by θ. To train the reward model, we have a preference dataset, D = $\{ ( x , y _ { w } , y _ { l } ) , \dotsc \}$ , where x is a context (a question or an instruction); $y _ { w }$ is a preferred output, and y is a less preferred output. The reward model is then trained to predict the preference score $r _ { \theta } ( x , y )$ for an input-output pair, where a larger score indicates that the output is more preferred by a human. The loss function for reward model training is

$$
\begin{array}{l} \text { loss } (\theta) = -   \mathbb {E} _ {(x, y _ {w}, y _ {l}) \sim \mathcal {D}} \\ \quad \left[ \log \left(\sigma \left(r _ {\theta} (x, y _ {w}) - r _ {\theta} (x, y _ {l})\right)\right) \right], \end{array}\tag{1}
$$

where $\sigma$ represents the sigmoid function. This approach effectively captures humans’ preferences, so that $r _ { \theta }$ predicts higher rewards for responses that are preferred by humans.

In the last step of RLHF, we can finetune the supervised finetuned (SFT) model using reinforcement learning, typically using the Proximal Policy Optimization (PPO) algorithm. In essence, PPO iteratively improves the policy by simultaneously minimizing the divergence between new and old policies and maximizing the expected cumulative rewards. We refer readers to find algorithm details in the original paper (Schulman et al., 2017).

## A.2 LoRA Finetuning

When finetuning a large language model, finetuning all the parameters can be computationally expensive and resource-demanding. To this end, Low-Rank Adaptation (LoRA) (Hu et al., 2021) is a well-accepted algorithm to efficiently finetune a pretrained large language model. Concretely, for each Transformer layer, LoRA learns

$$
\Delta W = A _ {1} A _ {2},\tag{2}
$$

where $\Delta W$ is the change applied to the weight matrix W of a transformer layer, and $A _ { 1 }$ and $A _ { 2 }$ are smaller matrices. Let the dimension of $W$ be $d _ { 1 } \times d _ { 2 }$ . Then the dimension of $A _ { 1 }$ is $d _ { 1 } \times r$ and the dimension of $A _ { 2 }$ is $r \times d _ { 2 }$ , where r is the rank of the decomposition, which is smaller than $d _ { 1 }$ and $d _ { 2 }$

When applying LoRA to a transformer layer, the modified weight matrix $W ^ { \prime }$ is used in the forward pass as follows,

$$
W ^ {\prime} = W + \Delta W,\tag{3}
$$

where W is the original weight matrix of the transformer layer. It is worth noting that during training, only the matrices $A _ { 1 }$ and $A _ { 2 }$ are updated. The original weights W remain frozen. This approach requires training only a substantially smaller set of parameters in matrices $A _ { 1 }$ and $A _ { 2 }$ , compared to the original weights W, while all the model weights will be influenced during training.

## B Additional Empirical Results

In addition to our results in the main paper, we also explored different prediction algorithms, including using the mean and lower confidence bound (LCB) of the reward predictions of the ensembled models. We perform the experiments on AlpacaEval. The differences between these prediction algorithms are insignificant, as shown in Figures 3 and 4.

## C Hyperparameters

We include the key hyperparameters for our experiments in the following tables.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>seed</td><td>42, 43, ..., 51</td></tr><tr><td>num_train_epochs</td><td>1</td></tr><tr><td>gradient_accumulation_steps</td><td>2</td></tr><tr><td>learning_rate</td><td>3e-6</td></tr><tr><td>weight_decay</td><td>0.0</td></tr><tr><td>warmup_ratio</td><td>0.03</td></tr><tr><td>optimizer</td><td>adamw_torch</td></tr><tr><td>lr_scheduler_type</td><td>cosine</td></tr><tr><td>learning_rate</td><td>5e-5</td></tr><tr><td>weight_decay</td><td>0.0</td></tr><tr><td>warmup_ratio</td><td>0.03</td></tr><tr><td>optimizer</td><td>adamw_torch</td></tr><tr><td>lr_scheduler_type</td><td>constant</td></tr></table>

Table 2: Parameters for reward modeling.

Table 3: Parameters for reward modeling with LoRA.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>rollout_batch_size</td><td>64</td></tr><tr><td>step_batch_size</td><td>32</td></tr><tr><td>learning_rate</td><td>1e-5</td></tr><tr><td>warmup_steps</td><td>5</td></tr><tr><td>epoch_num</td><td>2</td></tr><tr><td>optimizer</td><td>adamw_torch</td></tr><tr><td>kl_divergence_coefficient</td><td>0.02</td></tr><tr><td>value_function_coefficient</td><td>0.1</td></tr></table>

Table 4: PPO parameters.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>temperature</td><td>1.0</td></tr><tr><td>max_new_tokens</td><td>300</td></tr><tr><td>top_p</td><td>0.9</td></tr></table>

Table 5: Decoding parameters for Best-of-n.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>temperature</td><td>0.7</td></tr><tr><td>max_new_tokens</td><td>300</td></tr><tr><td>top_p</td><td>0.9</td></tr></table>

Table 6: Decoding parameters for PPO.

Figure 3: Lora-based ensemble with LCB and different β values.

Figure 4: Linear-layer ensemble with LCB and different β values.
