# Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization

Sunghwan Kim<sup>∗</sup> Dongjin Kang<sup>∗</sup> Taeyoon Kwon Hyungjoo Chae Dongha Lee Jinyoung Yeo

Yonsei University

{kimsh8564, hard1010, jinyeo}@yonsei.ac.kr

## Abstract

Reward models (RMs) play a crucial role in reinforcement learning from human feedback (RLHF), aligning model behavior with human preferences. However, existing benchmarks for reward models show a weak correlation with the performance of optimized policies, suggesting that they fail to accurately assess the true capabilities of RMs. To bridge this gap, we explore several evaluation designs through the lens of reward overoptimization—a phenomenon that captures both how well the reward model aligns with human preferences and the dynamics of the learning signal it provides to the policy. The results highlight three key findings on how to construct a reliable benchmark: (i) it is important to minimize differences between chosen and rejected responses beyond correctness, (ii) evaluating reward models requires multiple comparisons across a wide range of chosen and rejected responses, and (iii) given that reward models encounter responses with diverse representations, responses should be sourced from a variety of models. However, we also observe that a extremely high correlation with degree of overoptimization leads to comparatively lower correlation with certain downstream performance. Thus, when design ing a benchmark, it is desirable to use the degree of overoptimization as a useful tool, rather than the end goal. We make our code and data publicly available.<sup>1</sup>

## 1 Introduction

Reinforcement learning from human feedback (RLHF) through reward models (RMs) has been a key driver of the remarkable progress of large language models (LLMs) (Ouyang et al., 2022; Touvron et al., 2023; Lightman et al., 2024; OpenAI, 2024). However, a learned reward model is merely a proxy for human preferences and does not present the true reward (Stiennon et al., 2020; Gao et al., 2023). While the policy is trained to maximize the reward, over-optimizing for this proxy can lead to divergence from the true reward, a phenomenon referred to as reward overoptimization (Gao et al., 2023; Coste et al., 2024; Rafailov et al., 2024).


Figure 1: Relationship between existing benchmark scores for reward models (x-axis) and downstream performance (y-axis) for BoN sampling and PPO. Each dashed line indicates a best-fit trend with its r<sup>2</sup> value, demonstrating that existing benchmarks exhibit low r<sup>2</sup>. Relying on such unreliable benchmarks to develop reward models is likely to hinder progress in RLHF.

Previous studies have attempted to mitigate overoptimization by applying early stopping during training (Moskovitz et al., 2024) or using ensembles of reward models (Coste et al., 2024), while another line of research has focused on developing benchmarks to evaluate reward model’s capabilities (Lambert et al., 2024). Recently, Ivison et al. (2024) demonstrated that a reward model’s performance on benchmarks has only a weak correlation with the downstream performance (Figure 1). Building a reward model based on such unreliable benchmarks can unintentionally lead to models that accelerate reward overoptimization.

In this paper, we underscore the need for further exploration into benchmarks that exhibit a stronger correlation with downstream performance. However, directly designing benchmarks based on their correlation with downstream performance might be harmful, as the results can vary significantly depending on the policy optimization method used, such as best-of-n (BoN) sampling and proximal policy optimization (PPO). This work thus focuses on the reward overoptimization, which not only reflects how well the reward model aligns with human preferences but also captures the dynamics of learning signals provided to policies. To this end, we define a metric to quantify the degree of overoptimization, making it easier to analyze its correlation with downstream performance.

However, as directly measuring it comes with significant costs, our next objective is to investigate the evaluation design for reward models that well estimates the degree of overoptimization. Based on the 16 difference types of evaluation designs, including RewardBench (Lambert et al., 2024) and RM-Bench (Liu et al., 2024a), our investigation leads to several important insights on how to construct a benchmark that strongly correlates with the degree of overoptimization:

• Large distribution differences between chosen and rejected responses, e.g., response length and the number of steps, make it easier for reward models to achieve high scores.

• Multi-pairwise comparison allows for a more reliable evaluation than single comparison—i.e., comparing multiple chosen and rejected responses rather than a single pair.

• Multiple responses sourced from a wide range of LLMs are preferable to simply augmenting a single response, which aligns with the nature of reward models—encountering responses from a wide range of models at test time.

Upon these findings, a natural question is how to evaluate the success of a reward model under the multiple responses. Interestingly, this lead us to Goodhart’s Law (Goodhart, 1984), which states that when a measure becomes a target, it ceases to be a good measure. We observe that evaluation designs with the highest correlation to degree of overoptimization tend to exhibit lower correlations with downstream performance compared to other designs. For instance, while the evaluation design with strong correlation to PPO achieves $r ^ { 2 } > 0 . 8 3$ a design with a correlation close to 0.9 with the degree of overoptimization shows a lower correlation of 0.74 with PPO. Furthermore, we also find a trend where higher correlation with degree of overoptimization generally improves correlation with downstream performance, but excessive optimization results in a decline.

This suggests that blindly designing benchmarks to achieve a perfect correlation with the degree of overoptimization in a specific optimization method requires caution. In other words, a benchmark that shows the strongest correlation with the degree of overoptimization does not guarantee the highest correlation with downstream performance, as various factors—such as the choice of RL algorithm or hyperparameters—can influence the results. So, when designing benchmarks for reward models, the degree of overoptimization should be monitored as a useful tool rather than the end goal.

## 2 Background & Related Work

## 2.1 Reward Model Evaluation

In RLHF, RMs serve as proxy reward functions, providing a learning signal for policy optimization instead of human feedback (Ouyang et al., 2022; Bai et al., 2022). These models are typically implemented by adding a classification head to a pretrained language model (Ouyang et al., 2022; Zhu et al., 2023) and trained on human-annotated datasets (Cui et al., 2023; Wang et al., 2024c; Lightman et al., 2024) to assign rewards of responses based on given prompts. In mathematical reasoning, Uesato et al. (2022) and Lightman et al. (2024) propose process reward model (PRM), which predicts the correctness of each step in a solution rather than the overall response. Additionally, recent studies enhance RMs by integrating multi-objective ratings (Wang et al., 2024a) and generating critiques before assigning rewards (Ankner et al., 2024), improving generalization and interpretability.

The evaluation of reward models relies mainly on downstream performance, validating their efficacy by observing enhancements in the optimized policy (Dubois et al., 2024; Zheng et al., 2024). However, these evaluation approaches are questionable due to numerous ad-hoc choices in the policy optimization process, including the selection of the RL algorithm, computational resources, and hyperparameters (Gao et al., 2023; Casper et al., 2023). To better understand the behavior of the reward model, Lambert et al. (2024) proposed Reward-Bench, which uses pairwise comparisons between chosen and rejected responses. Recently, Liu et al. (2024b) introduced RM-Bench, which evaluates reward models by assessing their sensitivity to subtle content variations and resistance to style biases.

Figure 2: An example of reward overoptimization. The KL divergence represents the degree of optimization.

## 2.2 Reward Overoptimization

Reward overoptimization arises when a policy π is optimized to maximize rewards from a learned RM, leading to a decrease in performance even though achieving higher rewards from RM. This discrepancy occurs because (proxy) RM serves as an imperfect proxy for human preference, mak ing it vulnerable to Goodhart’s Law (Goodhart, 1984)—when a measure becomes a target, it ceases to be a good measure.

To systematically investigate this issue, Gao et al. (2023) introduced a synthetic setup that replace human annotator with a reward model (gold RM) for scoring human preference, i.e., true reward. This controlled experiment enables researchers to observe how the gold RM score evolves as optimization progresses against the proxy RM. Empirical evidence reveals a clear pattern: while the gold RM score initially increases with KL divergence between the initial and optimized policy π, it eventually declines—marking the onset of overoptimization (Figure 2). By exploring this phenomenon, several studies have demonstrated the effectiveness of their optimization methods (Gao et al., 2023; Coste et al., 2024; Yang et al., 2024; Rafailov et al., 2024). In this work, we analyze RM evaluation through the degree of overoptimization.

## 3 Evaluation Setup

## 3.1 Task and Focus

Task: mathematical reasoning. We focus on mathematical reasoning task because it provides a clear and objective human preference: accuracy. Prior works (Ivison et al., 2024; Rafailov et al., 2024) and our results (Figure 1) have highlighted a weak correlation between RM accuracy and downstream performance. This discrepancy may be attributed to two key factors: (1) limitations in the evaluation methods for RMs and (2) the lack of consensus on human preference could make it impossible to develop a truly effective RM. By focusing on mathematical reasoning, we can isolate the limitation of evaluation methods, eliminating the confounding influence of the second factor.

Figure 3: The results of the degree of overoptimization $( \gamma _ { \mathrm { o r a c l e } } )$ under oracle reward setting. Higher values of γ indicate a greater tendency for reward to degrade as optimized continues. The value of $\gamma$ for each RMs are reported in Table 10.

Focus: reward overoptimization. Among the various perspectives for evaluating RMs, this work focuses on reward overoptimization, which we believe is a crucial factor in assessing RMs. To underscore its importance, we outline why the degree of overoptimization matters (Section 4) and discuss how to design evaluation for RMs that effectively captures it (Section 5). This work utilizes two policy optimization methods: best-of-n (BoN) sampling and proximal policy optimization (PPO).

## 3.2 Dataset

To ensure a fair comparison with existing RM evaluations, we construct RM evaluation set based on MATH500, a widely used as a standard benchmark in mathematical reasoning, and used in Reward-Bench (Lambert et al., 2024). The MATH500 is derived from the test set of MATH dataset (Hendrycks et al., 2021), which includes 7.5k training problems and 5k test problems at the high school level. For evaluating downstream performance, we utilize two test sets: MATH500 (in-distribution) and Gaokao-math (out-of-distribution) (Zhang et al., 2023). The Gaokao-math is a math subset of the Chinese College Entrance Exam, and in our study, we use 390 problems that have been translated into



Figure 4: Relationship between degree of overoptimization $\gamma _ { \mathrm { o r a c l e } }$ and downstream performance (BoN and PPO) using MetaMATH-Mistral-7B as policy model. The results demonstrate that $\gamma _ { \mathrm { o r a c l e } }$ strongly correlates with the downstream performance. Results for $\gamma _ { \mathrm { g o l d } }$ and Llama3-8B-Instruct are provided in Figure 13.

English from Tang et al. (2024).

## 3.3 Degree of Overoptimization

Previous studies have illustrated the relationship between the gold reward and the KL divergence $D _ { \mathrm { K L } } ( \pi | | \pi _ { \mathrm { i n i t } } )$ through trend graphs (Gao et al., 2023; Coste et al., 2024). However, deriving a clear correlation from these graph is challenging. To address this, we propose a new metric, degree of overoptimization γ, which quantifies the extent to which optimization deviatesfrom the true reward.

For γ, we fit our BoN experimental data to function: $R _ { \mathrm { b o n } } ( x ) = x ( \alpha _ { \mathrm { b o n } } - \beta _ { \mathrm { b o n } } x )$ , where x represents $\sqrt { D _ { \mathrm { K L } } ( \pi | | \pi _ { \mathrm { i n i t } } ) }$ , following the empirical findings of Gao et al. (2023). We define the fitted curve for the gold RM as $f ( x )$ and the proxy RM as $g ( x )$ . The degree of overoptimization, γ, is then formulated as the difference between the areas under these two curves, normalized by the area under the gold RM:

$$
\gamma = \frac {\int_ {0} ^ {k} | f (x) - g (x) | d x}{\int_ {0} ^ {k} f (x) d x}
$$

where k represents the largest $D _ { \mathrm { K L } } ( \pi | | \pi _ { \mathrm { i n i t } } )$ . This metric implies that the closer γ is to 0, the better the proxy reward model aligns with the true reward, providing a more accurate and reliable learning signal. An example of $\gamma$ computed under various optimized trends is shown in Figure 3. More details on metric are provided in Appendix B.2.

## 4 Understanding the Overoptimization in Reward Models

In this section, we examine the relationship between reward overoptimization and the effectiveness of the reward model in providing a learning signal to the policy. To this end, we quantify the degree of overoptimization for each RM and analyze its correlation with downstream performance.

## 4.1 Models

Reward models. We conduct our experiments on 14 RMs widely used in mathematical reasoning task, such as classifier-based RMs and process reward models (PRMs). From the RewardBench leaderboard, we adopt top-ranked classifier-based RMs (Wang et al., 2024a; Cai et al., 2024; Yuan et al., 2024; Dai et al., 2024; Liu and Zeng, 2024; Yang et al., 2024), as well as available open-source PRMs (Wang et al., 2024b; Sun et al., 2024; Xia et al., 2024; o1 Team, 2024; Zhang et al., 2025). We use the geometric mean as the aggregation function to combine step-level rewards for PRMs.

Policy models. We employ one math expert model (MetaMATH-Mistral-7B, Wang et al. 2024b) and one general-purpose model (LLaMA-3-8B-Instruct, AI@Meta 2024) as policy models. More detailed explanations of the reward models and the policy models are in Appendix B.1.

## 4.2 Exploring Reward Overoptimization

Unlike previous works employing a synthetic setup (Gao et al., 2023) to examine reward overoptimization using proxy RMs trained from the gold reward, we explore this phenomenon in a more realistic scenario. To achieve this, we exclusively utilize open-source RMs in all our experiments, rather than training a proxy RM from a gold RM. We designate Skywork-o1-Open-PRM-Qwen-2.5-7B as the gold RM, which produces a gold reward, as it achieves the highest performance under the best-of-64 sampling. In mathematics, where a clear human preference exists, we extend beyond the gold reward by using accuracy as the oracle reward.

<table><tr><td rowspan="2">Design</td><td rowspan="2">Chosen</td><td rowspan="2">Rejected</td><td rowspan="2">Metric</td><td colspan="2">MetaMATH-Mistral-7B</td><td colspan="2">Llama3-8B-Instruct</td></tr><tr><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td></tr><tr><td colspan="8">Single Pairwise Comparisons</td></tr><tr><td>A</td><td rowspan="4">Human</td><td>unaligned GPT-4</td><td rowspan="4">1:1 acc</td><td>0.015</td><td>0.032</td><td>0.009</td><td>0.010</td></tr><tr><td>B</td><td>Gemma2-27B</td><td>0.102</td><td>0.117</td><td>0.101</td><td>0.102</td></tr><tr><td>C</td><td>Qwen1.5-7B</td><td>0.256</td><td>0.285</td><td>0.215</td><td>0.218</td></tr><tr><td>D</td><td>random, 1</td><td>0.224</td><td>0.252</td><td>0.216</td><td>0.216</td></tr><tr><td>E</td><td rowspan="4">GPT-4o*</td><td>unaligned GPT-4</td><td rowspan="4">1:1 acc</td><td>0.145</td><td>0.179</td><td>0.113</td><td>0.117</td></tr><tr><td>F</td><td>Gemma2-27B</td><td>0.485</td><td>0.497</td><td>0.394</td><td>0.399</td></tr><tr><td>G</td><td>Qwen1.5-7B</td><td>0.601</td><td>0.683</td><td>0.448</td><td>0.453</td></tr><tr><td>H</td><td>random, 1</td><td>0.720</td><td>0.820</td><td>0.597</td><td>0.601</td></tr><tr><td colspan="8">Multi-Pairwise Comparisons</td></tr><tr><td>I</td><td rowspan="2">GPT-4o*</td><td>GPT-4o, style</td><td rowspan="2">1:3 acc</td><td>0.143</td><td>0.214</td><td>0.067</td><td>0.068</td></tr><tr><td>J</td><td>random, 3</td><td>0.838</td><td>0.915</td><td>0.738</td><td>0.740</td></tr><tr><td>K</td><td rowspan="2">GPT-4o, style</td><td>GPT-4o, style</td><td rowspan="2">3:3 matrix</td><td>0.329</td><td>0.408</td><td>0.317</td><td>0.318</td></tr><tr><td>L</td><td>random, 3</td><td>0.620</td><td>0.678</td><td>0.691</td><td>0.687</td></tr><tr><td>M</td><td rowspan="2">GPT-4o*</td><td rowspan="2">random, 9</td><td>1:9 acc</td><td>0.870</td><td>0.936</td><td>0.805</td><td>0.804</td></tr><tr><td>N</td><td>1:9 matrix</td><td>0.758</td><td>0.854</td><td>0.632</td><td>0.637</td></tr><tr><td>O</td><td rowspan="2">random, 3</td><td rowspan="2">random, 3</td><td>3:3 acc</td><td>0.877</td><td>0.943</td><td>0.840</td><td>0.841</td></tr><tr><td>P</td><td>3:3 matrix</td><td>0.783</td><td>0.892</td><td>0.705</td><td>0.711</td></tr></table>

Table 1: The correlation $( r ^ { 2 } )$ between evaluation results from various design and the degree of overoptimization (γ) with two policy models. $\mathbf { G P T - 4 0 ^ { * } }$ indicates that a human solution is converted into a machine-generated solution using GPT-4o. Random refers to responses generated by random models. Style refers to augmenting a single response by applying style variations—concise, detailed, and detailed with markdown (Liu et al., 2024a).

Measurement of $\gamma _ { \mathbf { g o l d } }$ and $\gamma _ { \mathbf { 0 r a c l e } } .$ To measure $\gamma ,$ , we first generate a set of n responses from policy model $\pi .$ , denoted as $A = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { n } \}$ , using MATH500, with $n = 1 0 2 4 ^ { 2 }$ . Next, the proxy RM $R _ { \mathrm { p r o x y } }$ assigns a score to each response. The response with the highest assigned score is selected: $\hat { a } = \underset { a \in A } { \operatorname { a r g m a x } } R _ { \mathrm { p r o x y } } ( a )$ . We then compute the degree of overoptimization (γ) in two ways. If we use the gold reward corresponding to aˆ, we can obtain $\gamma _ { \mathrm { g o l d } } ;$ alternatively, if we use the accuracy of aˆ as an oracle reward, we can derive $\gamma _ { \mathrm { o r a c l e } }$ . This allows us to rigorously assess the alignment between the proxy RM and both the gold and oracle rewards.

## 4.3 Results

To examine the impact of overoptimization in RMs, we analyze the correlation between each RM’s degree of overoptimization (Figure 3) and the results from BoN $( n = 6 4 )$ and PPO experiments. As shown in Figure 4, BoN experiments on MATH500 and Gaokao-math yield a high coefficient of determination $( r ^ { 2 } )$ , indicating a strong relationship between $\gamma$ and downstream performance. Similarly,

PPO experiments demonstrate a strong positive correlation $( r ^ { 2 } > 0 . 7 )$ , even in the presence of various confounding factors beyond the reward signal.

Reward model evaluation and the degree of overoptimization. Our results highlight that the degree of overoptimization γ is a strong indicator of a reward model’s performance. However, directly measuring γ can be computationally expensive; for instance, evaluating a single RM in this experiment requires over 500K inferences. Therefore, developing a benchmark that strongly correlates with $\gamma$ would not only provide a reliable evaluation but also offer a cost-efficient approach to evaluate RMs. Building on this, our next step is to explore how to design an RM evaluation that effectively captures reward overoptimization.

## 5 Rethinking Evaluation Design for Reward Models

To evaluate a reward model, it is essential to compare the scores assigned by the reward model to chosen and rejected responses. It consists of three key components: chosen responses, rejected responses, and the evaluation metric. In this section, we explore how the design of each component influences the degree of overoptimization.


Figure 5: The correlation $( r ^ { 2 } )$ between evaluation results across different designs and downstream performance with two policy models. $( L e f t )$ Results for single pairwise evaluation designs. (Right) Results for multi-pairwise evaluation designs, demonstrating stronger correlation compare to single pairwise comparisons.

## 5.1 Experimental Setup

Chosen & Rejected response. In the math domain of RewardBench (Lambert et al., 2024), chosen responses consist of human-written solutions, while rejected responses are generated by an unaligned GPT-4—a base model without RLHF. Moreover, RM-Bench (Liu et al., 2024a) collects both correct and incorrect responses from GPT-4o and applies style modifications to create three versions response: concise, detailed, and detailed with markdown. To explore a variety of designs, we convert human solutions into machine-generated responses using GPT-4o for generating step-bystep solutions. Additionally, we incorporate chosen and rejected responses from various models. More details are provided in Appendix B.4.

Metric. We leverage two types of evaluation metric: accuracy and matrix. Accuracy assigns a true classification label when the reward for the chosen response is higher than that for all rejected responses. Following Liu et al. (2024a), matrix is defined as the average of all 1:1 pairwise comparisons between chosen and rejected responses. More details on metrics are in Appendix B.5.

## 5.2 How Should We Collect Chosen and Rejected Responses?

To understand how the configurations of chosen and rejected responses affect the estimation of degree of overoptimization, we examine 16 evaluation designs. The results for all RM across different designs and their degree of overoptimization are provided in Appendix C.


(a) Human vs. unalinged GPT-4 (Design A)  
(b) $\mathrm { G P T - 4 o ^ { * } }$ vs. random, 1 (Design H)  
Figure 6: The length distribution of chosen and rejected responses. In Design A, chosen responses are written by humans, while rejected responses are generated by an unaligned GPT-4, which tends to produce more verbose outputs—resulting in a large difference in response length.

Large differences in distribution between chosen and rejected responses hinder RM evaluation. Table 1 shows that using human-written solutions as chosen responses (A–E) or unaligned GPT-4 output as rejected responses (A, E) results in a weaker correlation with γ compared to other designs. Furthermore, Figure 5 (left) illustrates a weak correlation between these evaluation designs and downstream performance. One contributing factor is that human solutions often omit detailed reasoning, relying on mental calculations. As shown in Figure 6, human solutions contain an average of 218.9 tokens, whereas machine-generated solutions provide step-by-step explanations, averaging 407.1 tokens. This discrepancy in response length not only hinders RM evaluation but also risks of reward hacking due to length bias. A more detailed analysis are provided in Appendix B.7.

<table><tr><td>Response Type</td><td>Method</td><td>Diversity (↑)</td></tr><tr><td rowspan="2">Chosen</td><td>style</td><td>0.0107</td></tr><tr><td>random</td><td>0.0151</td></tr><tr><td rowspan="2">Rejected</td><td>style</td><td>0.0056</td></tr><tr><td>random</td><td>0.0265</td></tr></table>

Table 2: Diversity of chosen and rejected responses applying style variations and using random selection from various model responses. The details of metrics are provided in Appendix B.5.

Limitation of collecting response from a single model. We collect rejected responses using two approaches: sourcing them from a single model (F, G) and randomly selecting them from various models (H). To evaluate their impact on correlation with γ, we compared these approaches. As shown in Table 1 and Figure 5 (left), relying on a single model introduces unstable performance variations and results in a lower correlation with γ than random selection. In contrast, selecting responses from multiple models yields a more robust correlation and remains consistent across different random seeds (see Appendix B.7). These findings suggest that collecting responses from a wide range of models improves the evaluation reliability.

## 5.3 Improving RM Evaluation: The Impact of Diversity and Metric

Upon the findings that single pairwise comparisons can lead to unreliable results, we construct multiple responses to design a more robust evaluation set. We investigate how response diversity and evaluation metrics influence the reliability of evaluation.

Diversity of response can make evaluation more reliable. To explore how diversity influences benchmark reliability, we compare two approaches: (1) applying style variations to a single response (I, K) and (2) collecting responses from various models (J, L). Table 2 shows that style modification—simply rephrasing surface-level features (i.e., style)—results in low diversity. As a result, RM evlauation with style variations struggle to accurately reflect overoptimization and are thus weakly correlated with downstream performance, as demonstrated in Figure 5 (right). In contrast, generating responses from diverse models significantly enhances diversity and yields a stronger correlation with downstream performance. These results underscore the importance of response diversity in benchmarks, highlighting the limitations of style variations and the benefits of leveraging responses from multiple sources. However, because the three variants of a single response—concise, detailed, detailed with markdown—differ in length, as demonstrated in Section 5.2, this can raise a concern about their effects on the correlation of I and K. We address this further in Appendix B.7.

Figure 7: Relationship between RM evaluation results and PPO performance across various design (A–P). The figure shows a trend where higher correlation with γ generally improves correlation with PPO, but excessive optimization leads to lower correlation with PPO.

Is accuracy a good metric of RM evaluation? We find that minimizing distributional differences between chosen and rejected responses—while collecting responses from various models for multipairwise comparisons—yields a more reliable estimation of overoptimization and demonstrates a stronger correlation with downstream performance. These findings naturally raise a question: what is the most appropriate evaluation metric?

To explore this, we compare the correlation of accuracy and matrix with γ in both 1:9 and 3:3 comparisons. Table 1 reveals that accuracy more effectively captures overoptimization. However, as shown in Figure 5 (right), while accuracy exhibits a strong correlation with BoN sampling, its correlation with PPO is relatively lower. This discrepancy arises because accuracy evaluates correctness based on whether all chosen responses receive higher than rejected ones, whereas PPO prioritizes stable learning signals over strict ranking precision.



Figure 8: The correlation $( r ^ { 2 } )$ between benchmark scores and downstream performance on MBPP+ (code generation) and WildGuardTest (safety) across RewardBench, RM-Bench, and our method. We use LLaMA3-8B-Instruct as the policy model. RM-Bench shows a strong negative correlation in the safety domain, suggesting potential misalignment between benchmark scores and safe behavior, whereas our evaluation achieves positive correlation in both domains.

Implications for Goodhart’s Law Figure 7 demonstrates that designing an evaluation to better capture overoptimization initially enhances its correlation with downstream performance. However, as the $r ^ { 2 }$ continue to increase, the correlation with downstream performance begins to decline. This observation bring to mind Goodhart’s Law (Goodhart, 1984; Manheim and Garrabrant, 2018): when a metric becomes a target, it can lose its effectiveness as a true measure of performance, often leading to unintended consequences. In the context of RM evaluation, designing a benchmark solely to maximize its correlation with γ may finally fail to evaluate true performance of RMs. These findings suggest that, when developing benchmarks for reward models, the degree of overoptimization should be used as an insightful tool to monitor rather than treated as an end goal.

## 6 Application: Code and Safety

## 6.1 Experimental Setup

Building on the findings from the previous section, we extend our analysis to additional domains, specifically code and safety. We examine the correlation between the results of BoN sampling and the performance of 9 RMs on RewardBench, RM-Bench, and our evaluation design.

RM evaluation. Following Lambert et al. (2024); Liu et al. (2024a), we construct our evaluation sets using HumanEvalPack (Muennighoff et al., 2023) for the code domain and XSTest (Röttger et al., 2023), donotanswer (Wang et al., 2023), and AI2 Refusal datasets (Lambert et al., 2024) for the safety. Based on our findings, we collect 3 chosen and rejected responses from various models and evaluate them using the matrix metric.

Downstream performance. To assess downstream performance, we conduct BoN sampling with n = 16. For the code domain, we employ MBPP+ (Liu et al., 2023) and measure performance using the pass@1 metric. In the safety domain, we use WildGuardTest (Han et al., 2024) and evaluate the accuracy by using GPT-4o-mini to determine whether the response of the policy model showed refusal or compliance when given harmful/unharmful prompts. More experimental details are provided in Appendix B.6.

## 6.2 Generalization beyond Mathematics

Figure 8 illustrates the correlation between BoN sampling results and benchmark score across the code and safety domains. In the code domain, our evaluation design demonstrates a stronger correlation (both $r ^ { 2 }$ and Spearman) compared to existing benchmarks. In contrast, for the safety domain, reliance on the LLM’s judgment introduces instability in downstream performance, leading to an insignificant $r ^ { 2 }$ . Nevertheless, we still achieves a Spearman correlation exceeding 0.6, outperforming other benchmarks. These findings indicate that our evaluation design effectively reflects the performance of RMs. Consequently, we confirm that the findings from the previous section—derived from mathematics, where human preference is welldefined—can be generalized to other domains.

## 7 Discussion and Conclusion

In this work, we explore reliable RM evaluation through the lens of reward overoptimization, a critical issue in RLHF. We reveal that the degree of overoptimization (γ) is strongly correlated with downstream performance, underscoring its potential as a metric for evaluating RMs. However, directly leveraging γ is computationally expensive, making it impractical for benchmarking purposes. To address this limitation, we analyze 16 different types of evaluation design to examine how can we construct a benchmark that estimates overoptimization. The results underscore three key factors for designing RM evaluation that effectively capture overoptimization: distribution differences between chosen and rejected responses, response diversity, and multi-pairwise comparisons.

Interestingly, we find that extremely high correlation (above 0.9) with does not necessarily guarantee a strong correlation with downstream performance, which led us to Goodhart’s law. This occurs because γ itself is influenced by choices such as RL algorithm or hyperparameters. Therefore, intentionally engineering a benchmark to achieve perfect correlation with is not desirable, as this could result in benchmarks that are valid only for specific RL settings. Instead, we argue that γ should be used as a tool to validate benchmark reliability rather than treated as an end goal. We hope that our work inspires further research into developing more reliable RM evaluations. Furthermore, a deeper understanding of overoptimization and its impact on policy optimization will be crucial for advancing RLHF systems.

## Limitations

This work focuses on reward overoptimziation for designing reliable RM evaluations. While our study provides valuable insights, several limitations must be acknowledged. First, due to resource constraints, we were unable to experiment with larger reward models or policy models. Exploring a wider range of models could offer further insights into the scalability and generalizability of our findings. Second, our experiments for assessing policy performance were limited of BoN sampling and PPO. Although these methods are widely used in RLHF, exploring alternative optimization methods might provide a more comprehensive understanding of the impact of reward overoptimization. Finally, while we analyzed accuracy and a matrix-based metirc, further inverstivateion into additional metrics for multi-pairwise comparisons is needed.

## Acknowledgments

This work was supported by STEAM R&D Project (RS-2024-00454458) and Global Young Connect Project (RS-2024-00407282), and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT)(No.RS-2020-II201361, Artificial Intelligence Graduate School Program (Yonsei University)). Jinyoung Yeo is a corresponding author.

## References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, and 1 others. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

AI@Meta. 2024. Llama 3 model card.

Zachary Ankner, Mansheej Paul, Brandon Cui, Jonathan D Chang, and Prithviraj Ammanabrolu. 2024. Critique-out-loud reward models. arXiv preprint arXiv:2408.11791.

Anthropic. 2024. Claude 3.5 sonnet. https://www. anthropic.com/news/claude-3-5-sonnet.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, and 29 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, and 1 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324– 345.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, and 81 others. 2024. Internlm2 technical report. Preprint, arXiv:2403.17297.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier

Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Tong Wang, Samuel Marks, Charbel-Raphael Segerie, Micah Carroll, Andi Peng, Phillip Christoffersen, Mehul Damani, Stewart Slocum, Usman Anwar, and 13 others. 2023. Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research. Survey Certification.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. 2024. Reward model ensembles help mitigate overoptimization. In The Twelfth International Conference on Learning Representations.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. 2024. Safe rlhf: Safe reinforcement learning from human feedback. In The Twelfth International Conference on Learning Representations.

DeepSeek-AI. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. Preprint, arXiv:2405.04434.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. 2024. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR.

Charles AE Goodhart. 1984. Problems of monetary management: the UK experience. Springer.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. 2024. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. arXiv preprint arXiv:2406.18495.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. NeurIPS.

Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert, Noah A Smith, Yejin Choi, and Hannaneh Hajishirzi. 2024. Unpack ing dpo and ppo: Disentangling best practices for learning from preference feedback. arXiv preprint arXiv:2406.09279

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and 1 others. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, and 1 others. 2024. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Chris Yuhao Liu and Liang Zeng. 2024. Skywork reward model series. https://huggingface.co/ Skywork.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chat-GPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. 2024a. Rm-bench: Benchmarking reward models of language models with subtlety and style. Preprint, arXiv:2410.16184.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. 2024b. Rm-bench: Benchmarking reward models of language models with subtlety and style. arXiv preprint arXiv:2410.16184.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583.

David Manheim and Scott Garrabrant. 2018. Categorizing variants of goodhart’s law. arXiv preprint arXiv:1803 04585

Ted Moskovitz, Aaditya K Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca Dragan, and Stephen Marcus McAleer. 2024. Confronting reward model overoptimization with constrained RLHF. In The Twelfth International Conference on Learning Representations.

Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro Von Werra, and Shayne Longpre. 2023. Octopack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, and 1 others. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

Skywork o1 Team. 2024. Skywork-o1 open series. https://huggingface.co/Skywork.

OpenAI. 2023a. Chatgpt. https://openai.com/ blog/chatgpt.

OpenAI. 2023b. Gpt-4 technical report. Preprint, arXiv:2303.08774.

OpenAI. 2024. Learning to reason with large language models. https://openai.com/index/ learning-to-reason-with-llms/.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, W. Bradley Knox, Chelsea Finn, and Scott Niekum. 2024. Scaling laws for reward model overoptimization in direct alignment algorithms. In ICML 2024 Workshop on Models ofHuman Feedbackfor AI Alignment.

Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. 2023. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. arXiv preprint arXiv:2308.01263.

Meng Rui, Liu Ye, Joty Shafiq Rayhan, Xiong Caiming, Zhou Yingbo, and Yavuz Semih. 2024. Sfrembedding-mistral:enhance text retrieval with transfer learning. Salesforce AI Research Blog.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proxi mal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learn ing to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008– 3021.

Zhiqing Sun, Longhui Yu, Yikang Shen, Weiyang Liu, Yiming Yang, Sean Welleck, and Chuang Gan. 2024. Easy-to-hard generalization: Scalable align ment beyond human supervision. arXiv preprint arXiv:2403.09472.

Zhengyang Tang, Xingxing Zhang, Benyou Wang, and Furu Wei. 2024. Mathscale: Scaling instruction tuning for mathematical reasoning. arXiv preprint arXiv:2403.02884.

Gemma Team. 2024. Gemma.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. 2022. Solving math word problems with process-and outcomebased feedback. arXiv preprint arXiv:2211.14275.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. 2024a. Interpretable preferences via multi-objective reward modeling and mixture-ofexperts. arXiv preprint arXiv:2406.12845.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024b. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, Bangkok, Thailand. Association for Computational Linguistics.

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. 2023. Do-not-answer: A dataset for evaluating safeguards in llms. arXiv preprint arXiv:2308.13387.

Zhilin Wang, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, Jimmy J Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. 2024c. Helpsteer2: Open-source dataset for training top-performing reward models. arXiv preprint arXiv:2406.08673.

Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. 2024. Evaluating mathematical reasoning beyond accuracy. arXiv preprint arXiv:2404.05692.

Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and Tong Zhang. 2024. Regularizing hidden states enables learning generalizable reward model for llms. arXiv preprint arXiv:2406.10216.

Longhui Yu, Weisen Jiang, Han Shi, YU Jincheng, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large

language models. In The Twelfth International Conference on Learning Representations.

Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun. 2024. Advancing llm reasoning generalists with preference trees. Preprint, arXiv:2404.02078.

Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. 2023. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2024. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Pro cessing Systems, 36.

Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, and 1 others. 2023. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364.

Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. 2023. Starling-7b: Improving llm helpfulness & harmlessness with rlaif.

## A Additional Background

## A.1 Policy Optimization Method

Best-of-n Sampling Best-of-n (BoN) sampling is an inference-time method used to optimize the responses generated by a policy (Nakano et al., 2021; Stiennon et al., 2020). This work generates n completions from the policy π and select the completion with the highest score assigned by the proxy reward model. Following Stiennon et al. (2020), we can evaluate the degree of optimization analytically by defining the Kullback-Leibler (KL) divergence of BoN: $\begin{array} { r } { \mathrm { K L } _ { \mathrm { b o n } } = \log n - \frac { n - 1 } { n } } \end{array}$

Proximal Policy Optimization (PPO) Proximal policy optimization (PPO) (Schulman et al., 2017) is employed to update the policy π with a reward model $r _ { \phi }$ in RLHF (Ouyang et al., 2022; Bai et al.,

2022; Zheng et al., 2023). PPO aims to maximize the expected reward, which is adjusted by a KL penalty term to ensure that the optimized policy $\pi _ { \theta }$ does not deviate significantly from the reference policy $\pi _ { \mathrm { r e f } } \cdot$

$$
\begin{array}{r l} & {\underset {\pi_ {\theta}} {\max} \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\theta} (. | x)} \left[ r _ {\phi} (x, y) \right]} \\ & {-   \beta \mathbb {D} _ {\mathrm{KL}} \big [ \pi_ {\theta} (y | x)   | |   \pi_ {\mathrm{ref}} (y | x) \big ],} \end{array}
$$

where $\beta$ is a scaling factor for the KL penalty.

## A.2 Reward Models

Classifier-based Reward Model Using annotated preference data $\mathcal { D } \ = \ ( x ^ { i } , y _ { c } ^ { i } , y _ { r } ^ { i } ) _ { i = 1 } ^ { \bar { M } }$ , the classifier-based reward model is trained to assign higher reward to the chosen completion $y _ { c }$ over the rejected completion $y _ { r }$ . This training process involves maximizing the log-likelihood under the Bradley-Terry (BT) model (Bradley and Terry, 1952) for preference estimation:

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{reward}} \\ = - \mathbb {E} _ {(x, y _ {c}, y _ {r}) \sim \mathcal {D}} \big [ \log (\sigma (r _ {\phi} (x, y _ {c}) - r _ {\phi} (x, y _ {r}))) \big ] \end{array}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. In general, the reward model is obtained by replacing the final output layer of causal language model with a linear head to predict a scalar.

Process Reward Model (PRM) Uesato et al. (2022) and Lightman et al. (2024) propose the process reward model (PRM), which predicts the correctness of each intermediate step $s _ { i }$ in a solution. The PRM is trained with the following objective function:

$$
\begin{array}{l} \mathcal {L} _ {\text { pointwise }} = \sum_ {i = 1} ^ {K} \hat {y} _ {s _ {i}} \log y _ {s _ {i}} \\ \qquad + (1 - \hat {y} _ {s _ {i}}) \log (1 - y _ {s _ {i}}) \end{array}
$$

where $\hat { y } _ { s _ { i } }$ is the correctness label of $s _ { i } ,$ and $y _ { s _ { i } }$ is the sigmoid score of $s _ { i }$ assigned by PRM.

## A.3 Benchmark for Mathematical Reasoning

MATH500 The MATH dataset, introduced by Hendrycks et al. (2021), was designed to evaluate the mathematical problem solving capabilities. It originally consisted of 7.5K training problems and 5K test problems spanning various mathemat ical domains. However, to address the limitation of insufficient training data, Lightman et al. (2024)

proposed using a subset of the MATH test set, leading to the creation of MATH500. Since its introduction, MATH500 has been widely used in recent work as a standard benchmark. In this work, we use MATH500 both for the RM evaluation and as a test data set to assess the downstream performance.

Gaokao-math Gaokao-math is a mathematical subset of the Chinese College Entrance Exam. In this work , we utilize the dataset from Tang et al. (2024), which translated the math subset originally provided by Zhong et al. (2023) into English <sup>3</sup>. Additionally, Tang et al. (2024) reformulated traditional multiple-choice questions into math word problems, resulting in a dataset of 390 test samples. Notably, many LLMs exhibit lower performance on Gaokao-Math compared to their performance on MATH500.

## B Details of Experiments

## B.1 Models

Classifier-based Reward Model We use nine classifier-based reward models for evaluating the solutions. Among these, Yuan et al. (2024) and Dai et al. (2024) release human-annotated preference dataset, with Eurus-RM-7b using a mixture of the UltraInteract, UltraFeedback, and UltraSafety dataset,<sup>4</sup> and Beaver-7b-v2.0- reward using the PKU-SafeRLHF dataset.<sup>5</sup> Additionally, ArmoRM-Llama3-8B-v0.1,<sup>6</sup> Internlm2- 7b/20b-reward,<sup>7,8</sup> Oasst-rm-2.1-pythia-1.4b<sup>9</sup> are top-ranked in RewardBench. Notably, ArmoRM-Llama3-8B-v0.1 and Internlm2-7b-reward (Wang et al., 2024a; Cai et al., 2024) have been developed to prevent reward hacking. Furthermore, we employ Skywork-Reward-Llama-3.1-8B (Liu and Zeng, 2024), GRM-llama3-8B and GRM-gemma-2B (Yang et al., 2024), which are recently released models that rank at the top on RewardBench.

Process Reward Model (PRM) We employ process reward models (PRM), which assign a score to each intermediate step of a solution. Lightman et al. (2024) releases PRM800K, a comprehensive dataset comprising 800,000 step-level human feedback labels for training PRMs. Xia et al. (2024)

use PRM800K to train models on WizardMath-7B-V1.1, proposing ReasonEval 7B. Additionally, Sun et al. (2024) introduce an easy-to-hard generation approach, guided by the observation that evaluation is easier than generation. They provide PRM trained on the easier data (e.g., level 1-3 problems from the MATH dataset) within PRM800K.<sup>10</sup> However, due to the substantial human resources required to construct PRM training data, Wang et al. (2024b) propose a framework to automatically construct process supervision datasets without human annotations for math reasoning tasks, and train Mistral-7B using the MathShepherd dataset.<sup>11</sup>

Policy Model We emxploy MetaMATH-Mistral-7B<sup>12</sup> and Llama3-8B-Instruct<sup>13</sup> as policy models across all of our experiments. Specifically, MetaMATH-Mistral-7B is trained on MetaMATH dataset (Yu et al., 2023) and incorporates a separator (i.e., special step token) to distinguish reasoning steps.

## B.2 Degree of Overoptimization

In this work, we assess the reliability of RM evaluation not by directly correlating it with downstream performance, but by examining it through the lens of reward overoptimization. Below, we discuss why we employ the degree of overoptimization (γ) for a comprehensive analysis and outline our approach to measruing the γ.

Why this metric matters? The choice of policy optimization method profoundly influences downstream performance. As a result, an RM evaluation that correlates well with one down stream task might display weak—or even meaningless—correlations with another. For example, BoN sampling selects the highest scoring response among multiple candidates, inherently favoring reward models that emphasize ranking precision. In contrast, PPO emphasizes providing stable learning signals rather than strictly preserving ranking order. Given these differences, we aim to design RM evaluation based on reward overoptimization, which not only reflects how well the reward model aligns with human preferences but also captures the dynamics of learning signals provided to policies.

(a) Skywork-Reward-Llama-3.1-8B-v0.2

(b) internlm2-7b-reward  
Figure 9: The results of the curve fit analysis on our BoN experimental data for degree of overoptimization (γ) metric. This demonstrates strong alignment between our experimental results and proposed function from Gao et al. (2023).

How to quantify the degree of overoptimization $( \gamma ) ?$ Consider a set $A = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { n } \}$ of n responses generated by the policy model π in response to a given query. Each response is scored by a proxy RM $R _ { \mathrm { p r o x y } }$ , and we select the highest-scoring response as follows: aˆ = argmax $R _ { \mathrm { p r o x y } } ( a )$ . Subsequently, the gold RM a∈A $R _ { \mathrm { g o l d } }$ evaluates aˆ, yielding $R _ { \mathrm { g o l d } } ( \hat { a } )$ . Following the methodology of Stiennon et al. (2020), we analytically assess the degree of optimization by defining the Kullback-Leibler (KL) divergence for BoN as $\begin{array} { r } { D _ { \mathrm { K L } } = \log n - \frac { n - 1 } { n } } \end{array}$ . This enables us to generate an experimental plot:

$$
(x, y) = \left(\sqrt {D _ {K L} (n)}, \frac {1}{L} \sum_ {0} ^ {L} R _ {\text { gold }} (\hat {a})\right),
$$

where L denotes the dataset size. We then fit our BoN experimental data to the function $R _ { \mathrm { b o n } } ( x )$ = $x ( \alpha - \beta x )$ , where x represents $D _ { \mathrm { K L } } ( \pi | | \pi _ { \mathrm { i n i t } } )$ , as introduced by the empirical observations of Gao et al. (2023) (Figure 9). We define the fitted curve for the gold RM as $f ( x )$ and the corresponding curve for the proxy RM as $g ( x )$ . Based on this fitting, we compute the degree of overoptimization $( \gamma )$ as follows:

<table><tr><td>Hyperparameters</td><td>Value</td></tr><tr><td>Actor Learning Rate</td><td>1e-7 (M) , 2e-7 (L)</td></tr><tr><td>Critic Learning Rate</td><td>1e-6 (M) , 2e-6 (L)</td></tr><tr><td>Batch Size</td><td>64</td></tr><tr><td>Gradient Accumulation</td><td>2</td></tr><tr><td>KL Penalty Coefficient</td><td>0.05 (M), 0.08 (L)</td></tr><tr><td>Value Function Coefficient</td><td>0.05</td></tr><tr><td>Generate Max Length</td><td>1024</td></tr><tr><td>Training Epochs</td><td>2</td></tr></table>

Table 3: Hyperparameters used in PPO experiments, where (M) represents MetaMATH-Mistral-7B and (L) represents Llama3-8B-Instruct.

$$
\gamma = \frac {\int_ {0} ^ {k} | f (x) - g (x) | d x}{\int_ {0} ^ {k} f (x) d x}
$$

where k is the maximum observed $D _ { \mathrm { K L } } ( \pi | | \pi _ { \mathrm { i n i t } } )$

## B.3 Evaluating Downstream Performance

To evaluate downstream performance, we employ BoN sampling and PPO as policy optimization methods. We conduct experiments using two policy models: MetaMATH-Mistral-7B and Llama3- 8B-Instruct. All experiments are conducted on 8 NVIDIA RTX A6000 GPUs and 1 NVIDIA A100 GPU.

BoN experiments. In our BoN experiments, we generate $n = 6 4$ candidate solutions for each problem using two policy models. These experiments are conducted on the MATH500 and Gaokao-math dataset with a temperature of 1.0 and a top-p of 0.95. Each RM assigns scores to the generated candidates, and the highest-scoring response is selected for evaluation. A total of 14 RMs are used in this experiments.

PPO experimnets. Due to computational constraints, we could not evaluate all reward models under PPO. In particular, PRMs require parsing solutions step by step for stable usage. However, Llama3-8B-Instruct is hard to controll the outputs, making PPO with PRMs challenging. As a result, we use 7 classifier-based RMs and 3 PRMs for MetaMATH-Mistral-7B, and use 9 classifier-based RMs for Llama3-8B-Instruct. Morover, we use 30K dataset from the MetMATH dataset (Yu et al., 2023) for PPO experimtents. The hyperparameters used in our experiments are provided in Table 3, and we implemented PPO training using the Open-RLHF.<sup>14</sup>

<table><tr><td rowspan="2">LLMs</td><td colspan="4">Proportion (%)</td></tr><tr><td>Random, 3 (C)</td><td>Random, 1 (R)</td><td>Random, 3 (R)</td><td>Random, 9 (R)</td></tr><tr><td>GPT-4o-2024-05-13 (OpenAI, 2023b)</td><td>12.8</td><td>3.2</td><td>5.7</td><td>5.0</td></tr><tr><td>GPT-3.5-turbo-0125 (OpenAI, 2023a)</td><td>9.2</td><td>13.3</td><td>10.2</td><td>11.2</td></tr><tr><td>Claude-3-sonnet-20240229 (Anthropic, 2024)</td><td>8.0</td><td>9.9</td><td>9.3</td><td>9.2</td></tr><tr><td>Meta-Llama-3-70B (AI@Meta, 2024)</td><td>11.6</td><td>8.3</td><td>10.4</td><td>9.6</td></tr><tr><td>Mixtral-8x7B (Jiang et al., 2024)</td><td>9.7</td><td>14.4</td><td>13.7</td><td>12.6</td></tr><tr><td>Gemma-2-27b-it (Team, 2024)</td><td>10.9</td><td>11.5</td><td>8.2</td><td>9.2</td></tr><tr><td>DeepSeek-V2 (DeepSeek-AI, 2024)</td><td>5.2</td><td>10.4</td><td>11.3</td><td>11.1</td></tr><tr><td>Phi-3-medium (Abdin et al., 2024)</td><td>11.4</td><td>10.7</td><td>10.5</td><td>11.0</td></tr><tr><td>Qwen-1.5-7B-Chat (Bai et al., 2023)</td><td>7.2</td><td>7.7</td><td>9.7</td><td>10.3</td></tr><tr><td>Gemma-7b-it (Team, 2024)</td><td>5.0</td><td>5.3</td><td>6.7</td><td>6.5</td></tr><tr><td>WizardMath-7B-v1.1 (Luo et al., 2023)</td><td>9.0</td><td>5.3</td><td>4.3</td><td>4.3</td></tr></table>

Table 4: Proportion of chosen (C) and rejected (R) responses across different LLMs under random selection design.

## B.4 Dataset Construction for Reward Model Evaluation.

Following Lambert et al. (2024), we use MATH500 as the evaluation set for evaluating reward models in mathematical reasoning.

Chosen response. In this work, we construct four types of chosen responses:

• Human: Responses that are originally written by humans and sourced directly from the MATH500 dataset.

• GPT-4o\*: Machine-generated responses produced by converting the human-written solutions using GPT-4o. For these outputs, we prompt the model with: Based on the given solution, please regenerate your own solution. Solve the problem step-by-step.

• GPT-4o, style: Responses initially generated by GPT-4o that are further refined into three stylistic variations—concise, detailed, and markdown format.

• Random: A collection of responses randomly selected from a pool of 13 different large language models (see Table 4), from which only the correct solutions are retained. In this category, we vary the number of responses by selecting 1, 3, or 9 outputs per instance.

Rejected response. We also construct four types of rejected responses:

• Unaligned GPT-4: Responses generated by GPT-4 without RLHF. These outputs typically lack detailed reasoning steps and often deviate from the expected problem-solving approach, resulting in answers that are less coherent or complete.

• GPT-4o, style: Same as chosen response.

• Single model: Responses are generated using only one specific LLM from our pool.

• Random: Similar to the chosen responses, only the incorrect solutions are retained here, ensuring that the reward model is exposed to a diverse set of erroneous reasoning examples.

Distribution of each evaluation design. Figure 10 presents the length distribution across difference style modification (concise, detailed and markdownformat). For chosen responses, we observe noticeable difference in length depending on the style. However, for rejected responses, the difference between the concise and detailed style is relatively small. This is because, in simple problems, instructing the model to generate a concise response typically results in a correct solution, whereas the artificially incorrect responses tend to be more verbose. Additionally, Table B.1 presents the distribution of chosen and rejected responses across different models, illustrating the proportion of responses generated by each LLM.

## B.5 Details of Reward Model Evaluation

RM evaluation. For inference classifier-based RMs, we use the code provided by the official RewardBench repository.<sup>15</sup> For PRMs, given a solution $S = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , the PRMs assign a score to each step $s _ { i } ,$ , necessitating an aggregation function to calculate the final reward. In this work. we propose new aggregation function geo\_mean to mitigate step count bias from prod.


(a) Chosen Response  
(b) Rejected Response  
Figure 10: The distribution of chosen and rejected responses across different styles in the reproduced RM-Bench. While chosen responses demonstrate differences in token counts across three styles, rejected responses show relatively similar token distributions.

$$
\text { geo\_mean } = \left(\prod_ {i} s _ {i}\right) ^ {\frac {1}{n}}
$$

Accuracy. Accuracy quantifies the proportion of instances in which the reward model correctly ranks the chosen response above all rejected responses. Formally, for a given set of chosen and rejected responses $\left\{ a _ { \mathrm { c h o s e n } } , a _ { \mathrm { r e j e c t e d _ { 1 } } } , \dots , a _ { \mathrm { r e j e c t e d _ { \boldsymbol { k } } } } \right\}$ the accuracy metric assigns a correct classification label if:

$$
R (a _ {\text { chosen }}) > R (a _ {\text { rejected } _ {i}}), \quad \forall i \in \{1,..., k \}.
$$

This metric provides a strict evaluation criterion, as a single incorrect ranking among the rejected responses leads to a failure in classification.

Matrix. The matrix metric adopts a pairwise comparison approach, offering a more granular evaluation of ranking quality. Rather than requiring the chosen response to be ranked higher than all rejected responses, it computes the average correctness across all possible pairwise comparisons. Given the n chosen responses and m rejected responses, the matrix score is calculated as:

$$
\frac {1}{n m} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \mathbb {1} [ R (a _ {\text { chosen } _ {i}}) > R (a _ {\text { rejected } _ {j}}) ].
$$

This approach provides a more fine-grained assessment of the performance of the reward model, capturing partial ranking improvements even when the chosen response is not consistently ranked highest among all rejected responses.

Calculation of response diversity. For a given problem $p _ { i } ,$ let $A = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { n } \}$ represent the collected responses. To measure the semantic diversity between responses generated through style variations and those collected randomly from multiple models, we use cosine similarity. The diversity score is computed as follows:

$$
\begin{array}{c} \tilde {A} _ {i} = \text { Embedding } (A _ {i}) \\ D i v e r s i t y = \frac {1}{K} \sum_ {i = 1} ^ {K} (1 - \text { cosine\_similarity } (\tilde {A} _ {i})) \end{array}
$$

where K is the total number of problems. A higher diversity score allows for a more comprehensive and robust assessment of RM performance, and we confirm that greater diversity is positively correlated with $\gamma .$ . We use Salesforce/SFR-Embedding-Mistral (Rui et al., 2024) as our embedding model.

## B.6 Application for Other Domains

For the extended analysis in both code and safety domain, we employ Llama3-8b-Instruct as the policy to examine the correlation between the results of BoN sampling (n = 16) on downstream tasks and 9 classifier-based RMs (Appendix B.1) on RewardBench, RM-Bench and our evaluation design.

Constructing RM evaluation set. Following Lambert et al. (2024); Liu et al. (2024a), we use HumanEvalPack (Muennighoff et al., 2023) for code domain, and XSTest (Röttger et al., 2023), donotanswer (Wang et al., 2023), and AI2 Refusal Dataset (Lambert et al., 2024) for safety domain as the evaluation set to evaluate the reward models.

To ensure response diversity when collecting 3 chosen and rejected responses on each domain, we collected responses from various LLMs. We used 5 LLMs in the code domain, while for the safety domain, we expanded to 9 LLMs by adding

<table><tr><td rowspan="2">LLMs</td><td colspan="4">Proportion (%)</td></tr><tr><td>Code (C)</td><td>Code (R)</td><td>Safety (C)</td><td>Safety (R)</td></tr><tr><td>GPT-4o-2024-05-13 (OpenAI, 2023b)</td><td>24.5</td><td>14.2</td><td>8.5</td><td>5.7</td></tr><tr><td>Claude-3-sonnet-20240229 (Anthropic, 2024)</td><td>12.9</td><td>11.2</td><td>11.6</td><td>1.3</td></tr><tr><td>Meta-Llama-3-70B (AI@Meta, 2024)</td><td>22.5</td><td>23.2</td><td>11.1</td><td>9.9</td></tr><tr><td>Mixtral-8x7B (Jiang et al., 2024)</td><td>17.6</td><td>27.1</td><td>9.3</td><td>17.7</td></tr><tr><td>Gemma-2-27b-it (Team, 2024)</td><td>22.5</td><td>24.3</td><td>11.2</td><td>2.5</td></tr><tr><td>Mistral-7b-v0.1 (Jiang et al., 2023)</td><td>-</td><td>-</td><td>14.0</td><td>23.6</td></tr><tr><td>Gemma-2-2b-it (Team, 2024)</td><td>-</td><td>-</td><td>11.3</td><td>5.3</td></tr><tr><td>Meta-Llama-3-8B-Instruct (AI@Meta, 2024)</td><td>-</td><td>-</td><td>11.7</td><td>10.5</td></tr><tr><td>Dolphin-2.0-mistral-7b</td><td>-</td><td>-</td><td>11.3</td><td>23.5</td></tr></table>

Table 5: Proportions of models included in the reward model evaluation set for code and safety domains. C represents chosen response and R represents rejected response. Following Lambert et al. (2024), we use the Dolphin-2.0-mistral-7b model from cognitivecomputations/dolphin-2.0-mistral-7b.

4 more relatively underperforming models to obtain harmful responses. After evaluatin the LLMs responses for each problem, we randomly selected 3 responses across different LLMs and removed any problem that did not yield a complete set of 6 responses. Table 5 shows the LLMs used in each domain and their proportions in the evaluation set.

To robustly select chosen and rejected response, we use HumanEvalPackGenerative function from the official Github repository<sup>16</sup> to verify pass/fail status of the LLM responses in the code domain. For the safety domain, we employed GPT-4o-mini to perform binary classification based on four criteria, using the midpoint as the threshold. For shouldrespond tasks, we checked for over-refusal, while for should-refuse tasks, we evaluated compliance. The prompt used for the safety evaluation can be found in Figure 14, and Figure 15.

Downstream tasks. We selected different tasks with the RM evaluation set to avoid overlap. For the code domain, we utilize MBPP+ (Liu et al., 2023) and use the evaluation function from the official repository<sup>17</sup> to measure the pass@1 metric.

For the safety domain, we employ Wild-GuardTest (Han et al., 2024), which is designed to evaluate safety moderation models by assessing its ability to identify prompt harmfulness, response harmfulness, and LLM refusal. Since our goal is to evaluate BoN results in downstream tasks, we only consider prompt harmfulness. Therefore, from the 1.73k WildGuardTest dataset, we ramdomly sample 50 instances for each combination of adversarial (true/false) and harmfulness (true/false) prompts to increase task difficulty. We use accuracy as our evaluation metric by utilizing GPT-4o-mini to classify whether the LLM’s response is a refusal or compliance.

## B.7 Additional Analysis

Why does the math domain of RewardBench exhibit low correlation with downstream performance? Figure 11 illustrates the significant length distribution difference between humanwritten chosen response and unaligned GPT-4 rejected response in RewardBench. Notably, even after converting human response into machinegenerated response, the gap remain substantial. This discrepancy contributes to the low correlation with γ. As demonstrated in Table 13 (Design A, E), this disparity leads to the overassessment for many RMs, potentially resulting in reward hacking and inflated scores that do not reflect actual performance. Interestingly, for oasst-rm-2.1-pythia-1.4b-epoch-2.5, merely replacing human responses causes a significant performance drop, suggesting that this RM exhibits a length bias. Consequently, these results contribute to lower correlation with downstream perofrmance in RewardBench. In contrast, when rejected responses are randomly selected from various models, the differences in length distribution are minimized, indicating a more reliable evaluation setting.

The effectiveness of random selection. There are two primary approaches to collecting rejected responses: gathering responses from a specific model or randomly selecting responses from a diverse set of models. We compare these methods to assess their correlation with the degree of overoptimization (γ). As shown in Table 15, response collected from single models exhibit lower correlation with $\gamma$ and higher result variability compared to random selection from multiple models (Table $^ { 1 5 , }$ Design H). This suggest that results are highly sensitive to the choice of the source model, underscoring the importance of incorporating responses from multiple models to ensure a more reliable evaluation.

(a) Human vs. unalinged GPT-4


(b) GPT-4o\* vs. unaligned GPT-4  
(c) GPT-4o\* vs. Random, 1  
Figure 11: Comparison of token number distributions among RewardBench evaluation set, $\mathrm { G P T - 4 o ^ { * } }$ , and Random. A substantial distributional difference can be observed between Human and unaligned GPT-4, while $\mathrm { G P T - 4 o ^ { * } }$ and unaligned GPT-4 show moderate differences. In contrast, the distributions between $\mathrm { G P T - 4 o ^ { * } }$ and Random remain relatively similar. These distributional differences correlate with the degree of overoptimization (γ) results.

An analysis on how length differences across different styles affect the results of the 3×3 matrix. As shown in Figure 11, RM-Bench has a relatively small length difference between chosen and rejected responses within each style (i.e., concise, detailed, or detailed with markdown). However, there remain substantial differences across different styles. As we noted in Section 5.2, this raises a natural concern that length differences might have influenced the results of the 3x3 matrix. To investigate this, we additionally assess a $3 \times 3$ diagonal matrix using only the chosen and rejected responses within the same style from the original 3×3 matrix. As a result, the computed degree of overoptimization scores are $\gamma _ { \mathrm { g o l d } } = 0 . 2 7 0$ and $\gamma _ { \mathrm { o r a c l e } } = 0 . 3 4 0$ which are notably lower than those obtained using the full matrix (Design K). Although these scores exhibit a higher correlation than Design I, which also use three comparisons, they still remains low. Therefore, we conclude that incorporating style modification for reward model evaluation does not improve reliability of benchmark.

## C Additional Results

A case study for each design of RM evaluation. We present examples of chosen and rejected solutions of RewardBench (Table 6), RM-bench (Table 7, Table 8), and Random Selection from various models (Table 9).

Degree of overoptimization. We analyze the degree of overoptimization γ patterns in our experiments. Figure 12 shows the relationship between KL divergence and rewards (both gold and oracle) across different models. Figure 13 illustrates how variations in $\gamma _ { \mathrm { o r a c l e } }$ and $\gamma _ { \mathrm { g o l d } }$ correlate with downstream performance for MetaMATH-Mistral-7B and Llama3-8B-Instruct models.

Results of RM for all experiments. We demonstrate the detailed results of RM for all experiments. The degree of overoptimizaiton for RMs is shown in Table 10. Downstream performance results are presented in Table 11 and Table 12 for MetaMATH-Mistral-7b and Llama3-8b-Instruct, respectively.

The full result of RM performance for each evaluation design is shown in Table 13 and 14. The analysis of rejected responses collected from a single model is presented in Table 15. Finally, Table 16 shows the results with different random seeds.

Prompts used for safety experiments. We present the prompts used for safety tasks in Figure 14 and Figure 15 for should-respond and should-refuse tasks, respectively.

<table><tr><td>Problem</td><td>The product of integers 240 and  $k$  is a perfect cube. What is the smallest possible positive value of  $k$ ?</td></tr><tr><td>Response Types</td><td>Solution</td></tr><tr><td>Chosen (Human)</td><td> $1. 240 = 2^{4} \cdot 3 \cdot 5 = 2^{3}(2 \cdot 3 \cdot 5)$  $2. For\ 240k \text{ to be a perfect cube (and not a perfect square), } k \text{ must be at least } 2^{2} \cdot 3^{2} \cdot 5^{2} = \boxed{900}.$ </td></tr><tr><td>Rejected (Unalinged GPT-4)</td><td> $1. To start, I want to find the prime factorization of 240, since that will help me see what factors I need to multiply it by to get a perfect cube.\( 2. Using a factor tree, I can write 240 as 2 \times 120, then 2 \times 2 \times 60, then 2 \times 2 \times 2 \times 15, then 2 \times 2 \times 2 \times 3 \times 5.$  $3. So the prime factorization of 240 is 2^{3} \times 3 \times 5.$  $4. Now, to make a perfect cube, I need each prime factor to have a power that is a multiple of 3.$  $5. I already have 2^{3}, which is good, but 3 and 5 have powers of 1, which are not multiples of 3.$  $6. So I need to multiply 240 by some integer k that will give 3 and 5 higher powers that are multiples of 3.$  $7. The smallest way to do that is to multiply them by 3^{2} and 5^{2}, respectively.$  $8. Therefore, the smallest possible value of k is 3^{2} \times 5^{2}, which is 9 \times 25, which is \boxed{225}.$ </td></tr></table>

Table 6: An example of chosen and rejected solution from RewardBench.

(a) γ , curve fit


(c) $\gamma _ { \mathrm { g o l d } }$ , curve fit

(b) γ<sub>oracle</sub>  
(d) $\gamma _ { \mathrm { g o l d } }$  
Figure 12: The trend of gold rewards and oracle rewards as KL divergence increases for each model. Curve-fitted lines represent fits to the function proposed by Gao et al. (2023), while non-fitted points show the actual experimental results.

<table><tr><td colspan="2">Problem</td><td>The product of integers 240 and k is a perfect cube. What is the smallest possible positive value of k?</td></tr><tr><td>Response Types</td><td>Style</td><td>Solution</td></tr><tr><td rowspan="3">Chosen</td><td>Concise</td><td>1. To solve the problem, find the smallest positive integer k such that 240k is a perfect cube.2. Prime Factorization of 240: Factor 240 into its prime factors by dividing by the smallest primes:  $240 \div 2 = 120, 120 \div 2 = 60, 60 \div 2 = 30, 30 \div 2 = 15, 15 \div 3 = 5$ . So,  $240 = 2^{4} \times 3^{1} \times 5^{1}$ .3. Conditions for a Perfect Cube: Each prime factor's exponent must be a multiple of 3: For  $2^{4}$ , add 2 (4 + 2 = 6). For  $3^{1}$ , add 2 (1 + 2 = 3). For  $5^{1}$ , add 2 (1 + 2 = 3).4. Calculate k:  $k = 2^{2} \times 3^{2} \times 5^{2}$  Calculating:  $2^{2} = 4, 3^{2} = 9, 5^{2} = 25, k = 4 \times 9 \times 25, 4 \times 9 = 36, 36 \times 25 = 900$  Therefore, the smallest k is [900].</td></tr><tr><td>Detailed</td><td>1. To solve the problem, we need to find the smallest positive integer k such that the product 240k is a perfect cube. This requires analyzing the prime factorization of 240 and determining any necessary factors to make the entire product a perfect cube.2. The first step is to factor 240 into its prime factors. We do this by progressively dividing by the smallest prime numbers.3. 240 is even, so divide by 2:  $240 \div 2 = 120. 120$  is even, so divide by 2:  $120 \div 2 = 60. 60$  is even, so divide by 2:  $60 \div 2 = 30 30$  is even, so divide by 2:  $30 \div 2 = 15. 15$  is divisible by 3, so divide by 3:  $15 \div 3 = 5. 5$  is a prime number. Thus, the factorization of 240 is:  $240 = 2^{4} \times 3^{1} \times 5^{1}$ 4. The exponent on each prime factor in the factorization of a perfect cube must be a multiple of 3.5. For  $2^{4}$ , we need  $4 + a \equiv 0 \mod 3$ . Here, a = 2 (since  $4 + 2 = 6$ ). For  $3^{1}$ , we need  $1 + b \equiv 0 \mod 3$ . Here, b = 2 (since  $1 + 2 = 3$ ).6. Using these conditions, the smallest k will include additional factors:  $k = 2^{2} \times 3^{2} \times 5^{2}$ 7. Calculating this, we find:  $2^{2} = 4, 3^{2} = 9, 5^{2} = 25. Thus, k = 4 \times 9 \times 25$ 8. Calculating these products progressively: First,  $4 \times 9 = 36. Then, 36 \times 25 = 900$ 9. Therefore, the smallest possible positive value of k that makes 240k a perfect cube is: [900]</td></tr><tr><td>Markdown Format</td><td>1. To solve the problem, we need to find the smallest positive integer k such that the product 240k is a perfect cube. This requires analyzing the prime factorization of 240 and determining any necessary factors to make the entire product a perfect cube.2. **Prime Factorization of 240**: The first step is to factor 240 into its prime factors. We do this by progressively dividing by the smallest prime numbers. - 240 is even, so divide by 2: $240 \div 2 = 120$ - 120 is even, so divide by 2:  $120 \div 2 = 60$ - 60 is even, so divide by 2:  $60 \div 2 = 30$ - 30 is even, so divide by 2:  $30 \div 2 = 15$ - 15 is divisible by 3, so divide by 3:  $15 \div 3 = 5$ - 5 is a prime number. Thus, the factorization of 240 is: $240 = 2^{4} \times 3^{1} \times 5^{1}$ 3. **Define the Conditions for a Perfect Cube**: The exponent on each prime factor in the factorization of a perfect cube must be a multiple of 3. - For  $2^{4}$ , we need  $4 + a \equiv 0 \pmod{3}$ . Here, a = 2 (since  $4 + 2 = 6$ ). - For  $3^{1}$ , we need  $1 + b \equiv 0 \pmod{3}$ . Here, b = 2 (since  $1 + 2 = 3$ ). - For  $5^{1}$ , we need  $1 + c \equiv 0 \pmod{3}$ . Here, c = 2 (since  $1 + 2 = 3$ ).4. **Calculate k**: Using these conditions, the smallest k will include additional factors: $k = 2^{2} \times 3^{2} \times 5^{2}$ Calculating this, we find:  $-2^{2} = 4 - 3^{2} = 9 - 5^{2} = 25$  Thus, $k = 4 \times 9 \times 25$ Calculating these products progressively: - First,  $4 \times 9 = 36$  - Then,  $36 \times 25 = 900$  Therefore, the smallest possible positive value of k that makes 240k a perfect cube is:[900]</td></tr><tr><td rowspan="3">Rejected</td><td>Concise</td><td>1. To find the smallest positive integer k such that  $240 \times k$  is a perfect cube: Each prime exponent in a perfect cube must be a multiple of 3.2. Adjust Factors to Form a Perfect Cube:-  $2^{4}$  needs one more factor of  $2(2^{1})$  to reach  $2^{5}$ .-  $3^{1}$  needs two more factors of  $3(3^{2})$  to reach  $3^{2}$ .-  $5^{1}$  needs two more factors of  $5(5^{2})$  to reach  $5^{3}$ .3. Calculate k:  $k = 2^{1} \times 3^{2} \times 5^{2} = 2 \times 9 \times 25 = 450$ .4. Thus, the smallest k such that  $240 \times k$  is a perfect cube is [450].</td></tr><tr><td>Detailed</td><td>1. To find the smallest positive integer value of k such that the product of 240 and k is a perfect cube, we need to follow these steps:2. First, Prime Factorization of 240: Begin by factoring 240 into its prime components:  $240 = 2^{4} x 3^{1} x 5^{1}$ . Each prime power in the factorization of a number that forms a perfect cube should be a multiple of 3.3. Second, Adjust Factors of 240 to Make a Perfect Cube: Currently, the factors  $2^{4}$ ,  $3^{1}$ , and  $5^{1}$  need to be adjusted to match the requirements of being perfect cubes. For  $2^{4}$ , we need at least  $2^{5}$  to have a power that is a multiple of 3. Thus, we need one more factor of  $2(2^{1})$ . For  $3^{1}$ , we need at least  $3^{3}$ . Thus, we need two more factors of  $3(3^{2})$ . For  $5^{1}$ , we need at least  $5^{3}$ . Thus, we need two more factors of  $5(5^{2})$ .4. Third, Calculation of k: Now, calculate k by multiplying the additional factors needed:  $k = 2^{1} x 3^{2} x 5^{2}$ . Calculate each component:  $2^{1} = 2$ ,  $3^{2} = 9$ ,  $5^{2} = 25$ . Multiply these together to find k:  $k = 2x9x25$ . Perform step-by-step multiplication:  $2x9 = 18$ ,  $18x25 = 450$ .5. Thus, the smallest possible positive value of k such that 240 x k is a perfect cube is: [450].</td></tr><tr><td>Markdown Format</td><td>1. To find the smallest positive integer value of k such that the product of 240 and k is a perfect cube, we need to follow these steps:2. **Prime Factorization of 240:** Begin by factoring 240 into its prime components: $240 = 2^{4} \times 3^{1} \times 5^{1}$ Each prime power in the factorization of a number that forms a perfect cube should be a multiple of 3.3. **Adjust Factors of 240 to Make a Perfect Cube:** Currently, the factors  $2^{4}$ ,  $3^{1}$ , and  $5^{1}$  need to be adjusted to match the requirements of being perfect cubes. - For  $2^{4}$ , we need at least  $2^{5}$  to have a power that is a multiple of 3. Thus, we need one more factor of  $2(2^{1})$ . - For  $3^{1}$ , we need at least  $3^{3}$ . Thus, we need two more factors of  $3(3^{2})$ . - For  $5^{1}$ , we need at least  $5^{3}$ . Thus, we need two more factors of  $5(5^{2})$ .4. **Calculation of k:** Now, calculate k by multiplying the additional factors needed: $k = 2^{1} \times 3^{2} \times 5^{2}$ Calculate each component: $2^{1} = 2$ ,  $3^{2} = 9$ ,  $5^{2} = 25$ Multiply these together to find k: $k = 2 \times 9 \times 25$ Perform step-by-step multiplication: $2 \times 9 = 18$  $18 \times 25 = 450$ 5. Thus, the smallest possible positive value of k such that  $240 \times k$  is a perfect cube is:[450]</td></tr><tr><td>Response Types</td><td>Models</td><td>Solution</td></tr><tr><td>Chosen (GPT-4o*)</td><td>GPT-4o</td><td>1. We express 240 as a product of its prime factors:  $240 = 2^{4} \cdot 3 \cdot 5 = 2^{3}(2 \cdot 3 \cdot 5)$ .2. For the product 240k to be a perfect cube (and not a perfect square), the integer  $k$  must be at least  $2^{2} \cdot 3^{2} \cdot 5^{2} = \boxed{900}$ .</td></tr><tr><td rowspan="3">Chosen (Random)</td><td>GPT-3.5-turbo-0125</td><td>1. We are given that the product of integers 240 and  $k$  is a perfect cube. This implies that 240k should be in the form of  $m^{3}$  for some integer  $m$ .2. We can express 240 as  $2^{4} \cdot 3 \cdot 5$ . To make 240k a perfect cube, we need to add multiples of 2, 3, and 5 to make the powers of all prime factors multiples of 3.3.he smallest  $k$  that would make 240k a perfect cube would be  $2^{2} \cdot 3^{2} \cdot 5^{2} = 900$ .4. Therefore, the smallest possible positive value of  $k$  is  $\boxed{900}$ .</td></tr><tr><td>Llama3-70B-Instruct</td><td>1. Since 240 is not a perfect cube,  $k$  must contain the remaining factors necessary to make the product a perfect cube.2. The prime factorization of 240 is  $2^{4} \times 3 \times 5$ .3. To make this a perfect cube, we need to multiply by  $2^{2} \times 3^{2} \times 5^{2}$ , which equals 900.4.Therefore, the smallest possible positive value of  $k$  is  $\boxed{900}$ .</td></tr><tr><td>Gemma2-27B-it</td><td>1. In order for a number to be a perfect cube, each of its prime factors must appear a multiple of 3 times.2. Prime factorize 240:  $240 = 2^{4} \cdot 3 \cdot 5$ .3. To make this a perfect cube, we need to include the prime factors needed to raise each of these prime factors to a power divisible by 3.4. We already have  $2^{4}$ , so we need  $2^{2}$  to make the exponent of 2 a multiple of 3.5. We need  $3^{2}$ , and we need  $5^{2}$ .6. Therefore, the smallest possible value of  $k$  is  $k = 2^{2} \cdot 3^{2} \cdot 5^{2} = \boxed{900}$ .</td></tr><tr><td rowspan="3">Rejected (Random)</td><td>GPT-3.5-turbo-0125</td><td>1. We know that  $240 \cdot k = 2^{4} \cdot 3 \cdot 5 \cdot k$  needs to be a perfect cube.2. To make this a perfect cube, we need to divide by  $2^{4} \cdot 3 \cdot 5 = 120$  to get  $k$  as a perfect cube.3. The smallest possible positive value of  $k$  is  $120 = 2^{3} \cdot 3 \cdot 5$ .4. Therefore, the smallest possible positive value of  $k$  is  $\boxed{120}$ .</td></tr><tr><td>Gemma2-27B-it</td><td>1. To find the smallest possible positive value of  $k$  such that the product 240 and  $k$  is a perfect cube, we need to determine the prime factorization of 240.2. The prime factorization of 240 is  $2^{4} \cdot 3 \cdot 5$ .3. To make this a perfect cube, each prime factor's exponent should be a multiple of 3.4. We can achieve this by increasing the exponent of 2 to 6 (since  $2^{6} = 64$  is a perfect cube) and making sure the exponents of 3 and 5 are also multiples of 3.5. So, the smallest possible value of  $k$  is  $3^{2} \cdot 5^{2} = \boxed{225}$ .</td></tr><tr><td>WizardMath-7B-v1.1</td><td>1. The problem is asking for the smallest possible positive integer  $k$  such that the product of 240 and  $k$  is a perfect cube.2. Prime factorization of 240. First, we need to find the prime factorization of 240. The prime factorization of 240 is  $2^{4} \cdot 3 \cdot 5$ .3. Understanding perfect cubes. A perfect cube is a number that can be expressed as the cube of an integer. Therefore, the prime factorization of a perfect cube must have all its exponents divisible by 3.4. Finding the smallest  $k$ . We need to find the smallest  $k$  such that the product of 240 and  $k$  is a perfect cube. This means that the prime factorization of  $k$  must be such that when multiplied with the prime factorization of 240, all exponents are divisible by 3.5. The prime factorization of  $k$  would then need to be in the form  $2^{a} \cdot 3^{b} \cdot 5^{c}$ , where  $3a + b + c$  is divisible by 3.6. To minimize  $k$ , we want to minimize the exponents  $a, b$ , and  $c$ . The smallest possible values for  $a, b$ , and  $c$  that make  $3a + b + c$  divisible by 3 are  $a = 0, b = 1$ , and  $c = 0$ .7. Therefore, the smallest possible positive value of  $k$  is  $2^{a} \cdot 3^{b} \cdot 5^{c} = 2^{0} \cdot 3^{1} \cdot 5^{0} = 3 \cdot 1 = 3$ .8. So, the smallest possible positive value of  $k$  is  $\boxed{3}$ .</td></tr></table>

Table 7: An example of chosen solution using style variation with GPT-4o.

Table 8: An example of rejected solution using style variation with GPT-4o.

Table 9: An example of chosen (GPT-4\* and Random, 3) and rejected (Random, 3) solution.




(a) γ <sub>ld</sub>, MetaMATH-Mistral-7B  



(b) γ<sub>oracle</sub>, Llama3-8B-Instruct  


(c) $\gamma _ { \mathrm { g o l d } } ,$ Llama3-8B-Instruct  
Figure 13: Relationship between $\gamma _ { \mathrm { o r a c l e } } , \gamma _ { \mathrm { g o l d } } .$ , and downstream performance for MetaMATH-Mistral-7B and Llama3- 8B-Instruct. The plot illustrates how variations in both $\gamma _ { \mathrm { o r a c l e } }$ and $\gamma _ { \mathrm { g o l d } }$ correspond to downstream task performance, with MetaMATH-Mistral- ${ \cdot } 7 \mathrm { B } ^ { \prime } \mathrm { s } \ \gamma _ { \mathrm { o r a c l e } }$ specifically detailed in Figure 4.

<table><tr><td rowspan="2">Reward Model</td><td colspan="2">MetaMATH-Mistral-7B</td><td colspan="2">Llama3-8B-Instruct</td></tr><tr><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td></tr><tr><td>Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B</td><td>0</td><td>0.416</td><td>0</td><td>0.435</td></tr><tr><td>Skywork/Skywork-Reward-Llama-3.1-8B-v0.2</td><td>0.663</td><td>0.752</td><td>0.427</td><td>0.736</td></tr><tr><td>RLHFlow/ArmoRM-Llama3-8B-v0.1</td><td>0.720</td><td>0.782</td><td>0.463</td><td>0.817</td></tr><tr><td>internlm/internlm2-7b-reward</td><td>0.510</td><td>0.598</td><td>0.306</td><td>0.540</td></tr><tr><td>internlm/internlm2-20b-reward</td><td>0.548</td><td>0.661</td><td>0.339</td><td>0.598</td></tr><tr><td>PKU-Alignment/beaver-7b-v2.0-reward</td><td>0.971</td><td>0.970</td><td>0.538</td><td>0.949</td></tr><tr><td>Ray2333/GRM-llama3-8B-sftreg</td><td>0.667</td><td>0.736</td><td>0.438</td><td>0.772</td></tr><tr><td>Ray2333/GRM-Gemma-2B-sftreg</td><td>0.886</td><td>0.924</td><td>0.483</td><td>0.852</td></tr><tr><td>openbmb/Eurus-RM-7b</td><td>0.777</td><td>0.801</td><td>0.421</td><td>0.743</td></tr><tr><td>OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5</td><td>0.994</td><td>0.994</td><td>0.577</td><td>1.018</td></tr><tr><td>Qwen/Qwen2.5-Math-PRM-7B</td><td>0.307</td><td>0.457</td><td>0.198</td><td>0.349</td></tr><tr><td>peiyi9979/math-shepherd-mistral-7b-prm</td><td>0.634</td><td>0.757</td><td>0.353</td><td>0.623</td></tr><tr><td>ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf</td><td>0.815</td><td>0.888</td><td>0.516</td><td>0.910</td></tr><tr><td>GAIR/ReasonEval-7B</td><td>0.668</td><td>0.772</td><td>0.390</td><td>0.688</td></tr></table>

Table 10: The degree of overoptimization $( \gamma )$ of RMs for two policy models. Note that the Skywork-o1-Open-PRM-Qwen2.5-7B is used as the gold RM, so its $\gamma _ { \mathrm { g o l d } }$ is 0.

<table><tr><td rowspan="2">Reward Model</td><td colspan="3">MetaMATH-Mistral-7B</td></tr><tr><td>BoN (MATH500)</td><td>BoN (Gaokao-math)</td><td>PPO (MATH500)</td></tr><tr><td colspan="4">Classifier-based Reward Models</td></tr><tr><td>Skywork/Skywork-Reward-Llama-3.1-8B-v0.2</td><td>39.4</td><td>14.6</td><td>28.8</td></tr><tr><td>RLHFlow/ArmoRM-Llama3-8B-v0.1</td><td>37.8</td><td>13.3</td><td>27.2</td></tr><tr><td>internlm/internlm2-7b-reward</td><td>46.0</td><td>20.8</td><td>29.4</td></tr><tr><td>internlm/internlm2-20b-reward</td><td>41.6</td><td>18.4</td><td>-</td></tr><tr><td>PKU-Alignment/beaver-7b-v2.0-reward</td><td>29.6</td><td>11.8</td><td>26.8</td></tr><tr><td>Ray2333/GRM-llama3-8B-sftreg</td><td>39.6</td><td>16.7</td><td>28.0</td></tr><tr><td>Ray2333/GRM-Gemma-2B-sftreg</td><td>29.6</td><td>13.6</td><td>-</td></tr><tr><td>openbmb/Eurus-RM-7b</td><td>35.6</td><td>12.6</td><td>29.2</td></tr><tr><td>OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5</td><td>26.2</td><td>8.21</td><td>26.0</td></tr><tr><td colspan="4">Process Reward Models</td></tr><tr><td>Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B</td><td>56.2</td><td>28.2</td><td>29.4</td></tr><tr><td>Qwen/Qwen2.5-Math-PRM-7B</td><td>52.8</td><td>22.3</td><td>30.2</td></tr><tr><td>peiyi9979/math-shepherd-mistral-7b-prm</td><td>37.4</td><td>13.8</td><td>-</td></tr><tr><td>ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf</td><td>32.2</td><td>10.5</td><td>27.4</td></tr><tr><td>GAIR/ReasonEval-7B</td><td>37.0</td><td>13.1</td><td>-</td></tr><tr><td rowspan="2">Reward Model</td><td colspan="3">Llama3-8B-Instruct</td></tr><tr><td>BoN (MATH500)</td><td>BoN (Gaokao-math)</td><td>PPO (MATH500)</td></tr><tr><td colspan="4">Classifier-based Reward Models</td></tr><tr><td>Skywork/Skywork-Reward-Llama-3.1-8B-v0.2</td><td>38.2</td><td>15.9</td><td>28.5</td></tr><tr><td>RLHFlow/ArmoRM-Llama3-8B-v0.1</td><td>32.8</td><td>14.1</td><td>28.2</td></tr><tr><td>internlm/internlm2-7b-reward</td><td>45.2</td><td>24.6</td><td>28.9</td></tr><tr><td>internlm/internlm2-20b-reward</td><td>42.6</td><td>19.2</td><td>29.4</td></tr><tr><td>PKU-Alignment/beaver-7b-v2.0-reward</td><td>27.6</td><td>10.5</td><td>25.6</td></tr><tr><td>Ray2333/GRM-llama3-8B-sftreg</td><td>34.4</td><td>15.9</td><td>27.8</td></tr><tr><td>Ray2333/GRM-Gemma-2B-sftreg</td><td>30.4</td><td>11.5</td><td>27.5</td></tr><tr><td>openbmb/Eurus-RM-7b</td><td>38.4</td><td>16.7</td><td>27.2</td></tr><tr><td>OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5</td><td>26.6</td><td>10.8</td><td>25.2</td></tr><tr><td colspan="4">Process Reward Models</td></tr><tr><td>Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B</td><td>55.0</td><td>31.0</td><td>-</td></tr><tr><td>Qwen/Qwen2.5-Math-PRM-7B</td><td>52.8</td><td>27.4</td><td>-</td></tr><tr><td>peiyi9979/math-shepherd-mistral-7b-prm</td><td>40.2</td><td>13.8</td><td>-</td></tr><tr><td>ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf</td><td>30.8</td><td>12.6</td><td>-</td></tr><tr><td>GAIR/ReasonEval-7B</td><td>36.8</td><td>15.9</td><td>-</td></tr></table>

Table 11: Downstream performance of MetaMATH-Mistral-7B with policy optimization methods using various RMs.

Table 12: Downstream performance of Llama3-8B-Instruct with policy optimization methods using various RMs.

<table><tr><td rowspan="2">Reward Model</td><td colspan="8">RM Evaluation Design</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td></tr><tr><td colspan="9">Classifier-based Reward Models</td></tr><tr><td>Skywork/Skywork-Reward-Llama-3.1-8B-v0.2</td><td>97.8</td><td>58.2</td><td>82.1</td><td>66.4</td><td>98.1</td><td>55.9</td><td>85.1</td><td>68.9</td></tr><tr><td>RLHFlow/ArmoRM-Llama3-8B-v0.1</td><td>98.7</td><td>61.2</td><td>87.8</td><td>73.4</td><td>87.8</td><td>50.3</td><td>83.9</td><td>67.5</td></tr><tr><td>internlm/internlm2-7b-reward</td><td>94.9</td><td>73.7</td><td>95.2</td><td>88.5</td><td>79.3</td><td>52.2</td><td>87.5</td><td>75.1</td></tr><tr><td>internlm/internlm2-20b-reward</td><td>95.1</td><td>69.0</td><td>92.8</td><td>88.2</td><td>84.2</td><td>54.6</td><td>83.7</td><td>74.1</td></tr><tr><td>PKU-Alignment/beaver-7b-v2.0-reward</td><td>60.4</td><td>63.3</td><td>59.0</td><td>63.5</td><td>37.2</td><td>47.2</td><td>35.7</td><td>48.0</td></tr><tr><td>Ray2333/GRM-llama3-8B-sftreg</td><td>89.3</td><td>26.3</td><td>38.9</td><td>27.8</td><td>98.6</td><td>55.6</td><td>82.7</td><td>68.9</td></tr><tr><td>Ray2333/GRM-Gemma-2B-sftreg</td><td>66.9</td><td>26.6</td><td>27.4</td><td>27.3</td><td>98.4</td><td>32.4</td><td>45.4</td><td>42.8</td></tr><tr><td>openbmb/Eurus-RM-7b</td><td>79.9</td><td>48.1</td><td>65.8</td><td>56.5</td><td>58.4</td><td>44.1</td><td>73.5</td><td>62.6</td></tr><tr><td>OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5</td><td>95.1</td><td>77.9</td><td>79.9</td><td>79.5</td><td>20.2</td><td>47.5</td><td>45.6</td><td>50.6</td></tr><tr><td colspan="9">Process Reward Models</td></tr><tr><td>Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B</td><td>79.9</td><td>74.3</td><td>94.3</td><td>87.1</td><td>80.0</td><td>74.7</td><td>94.6</td><td>86.1</td></tr><tr><td>Qwen/Qwen2.5-Math-PRM-7B</td><td>66.4</td><td>70.7</td><td>94.3</td><td>88.5</td><td>65.2</td><td>69.8</td><td>93.6</td><td>88.9</td></tr><tr><td>peiyi9979/math-shepherd-mistral-7b-prm</td><td>81.4</td><td>40.3</td><td>79.6</td><td>64.7</td><td>68.7</td><td>25.9</td><td>67.4</td><td>48.2</td></tr><tr><td>ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf</td><td>29.1</td><td>38.5</td><td>72.2</td><td>52.9</td><td>32.7</td><td>45.4</td><td>79.9</td><td>59.1</td></tr><tr><td>GAIR/ReasonEval-7B</td><td>14.8</td><td>17.6</td><td>34.8</td><td>32.7</td><td>38.8</td><td>39.8</td><td>65.2</td><td>60.9</td></tr><tr><td colspan="9">Classifier-based Reward Models</td></tr><tr><td>Skywork/Skywork-Reward-Llama-3.1-8B-v0.2</td><td>53.5</td><td>43.1</td><td>70.3</td><td>59.1</td><td>20.1</td><td>42.7</td><td>20.2</td><td>72.5</td></tr><tr><td>RLHFlow/ArmoRM-Llama3-8B-v0.1</td><td>63.2</td><td>40.2</td><td>74.5</td><td>58.7</td><td>20.5</td><td>42.2</td><td>18.6</td><td>68.6</td></tr><tr><td>internlm/internlm2-7b-reward</td><td>60.7</td><td>55.1</td><td>78.3</td><td>80.9</td><td>37.3</td><td>55.2</td><td>31.3</td><td>76.4</td></tr><tr><td>internlm/internlm2-20b-reward</td><td>56.5</td><td>52.7</td><td>74.4</td><td>76.7</td><td>34.0</td><td>52.0</td><td>26.4</td><td>74.0</td></tr><tr><td>PKU-Alignment/beaver-7b-v2.0-reward</td><td>45.1</td><td>21.2</td><td>72.6</td><td>65.0</td><td>7.2</td><td>26.6</td><td>9.0</td><td>56.4</td></tr><tr><td>Ray2333/GRM-llama3-8B-sftreg</td><td>66.0</td><td>44.9</td><td>75.1</td><td>66.8</td><td>24.4</td><td>45.8</td><td>23.0</td><td>72.2</td></tr><tr><td>Ray2333/GRM-Gemma-2B-sftreg</td><td>37.6</td><td>17.9</td><td>68.1</td><td>62.5</td><td>5.0</td><td>24.1</td><td>13.2</td><td>62.4</td></tr><tr><td>openbmb/Eurus-RM-7b</td><td>42.9</td><td>37.9</td><td>70.2</td><td>74.4</td><td>17.0</td><td>38.3</td><td>19.5</td><td>70.6</td></tr><tr><td>OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5</td><td>48.7</td><td>21.4</td><td>59.0</td><td>33.3</td><td>7.0</td><td>27.0</td><td>8.1</td><td>56.8</td></tr><tr><td colspan="9">Process Reward Models</td></tr><tr><td>Skywork/Skywork-o1-Open-PRM-Qwen2.5-7B</td><td>56.8</td><td>72.9</td><td>79.3</td><td>88.5</td><td>55.1</td><td>70.6</td><td>39.4</td><td>79.7</td></tr><tr><td>Qwen/Qwen2.5-Math-PRM-7B</td><td>65.2</td><td>74.4</td><td>76.1</td><td>83.3</td><td>59.4</td><td>73.1</td><td>34.1</td><td>78.5</td></tr><tr><td>peiyi9979/math-shepherd-mistral-7b-prm</td><td>28.7</td><td>28.0</td><td>64.8</td><td>71.5</td><td>15.7</td><td>33.6</td><td>17.4</td><td>67.8</td></tr><tr><td>ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf</td><td>51.3</td><td>36.0</td><td>57.8</td><td>53.2</td><td>16.4</td><td>37.5</td><td>10.4</td><td>62.6</td></tr><tr><td>GAIR/ReasonEval-7B</td><td>27.9</td><td>35.5</td><td>56.4</td><td>69.4</td><td>20.3</td><td>39.3</td><td>13.9</td><td>65.9</td></tr></table>

Table 13: Benchmark scores of RMs across RM evaluation designs (A-H). The highest scores are shown in bold.

Table 14: Benchmark scores of RMs across RM evaluation designs (I-P). The highest scores are shown in bold.

<table><tr><td rowspan="2">Chosen</td><td rowspan="2">Rejected</td><td rowspan="2">Metric</td><td colspan="2">MetaMATH-Mistral-7B</td><td colspan="2">Llama3-8B-Instruct</td></tr><tr><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td></tr><tr><td rowspan="4">Human</td><td>Gemma2-27B</td><td rowspan="4">1:1 acc</td><td>0.102</td><td>0.117</td><td>0.101</td><td>0.102</td></tr><tr><td>Qwen1.5-7B</td><td>0.256</td><td>0.285</td><td>0.2154</td><td>0.218</td></tr><tr><td>Mixtral-8x7B-Instruct-v0.1</td><td>0.225</td><td>0.247</td><td>0.218</td><td>0.196</td></tr><tr><td>DeepSeek-V2-Lite-Chat</td><td>0.287</td><td>0.329</td><td>0.272</td><td>0.289</td></tr><tr><td rowspan="4">GPT-4*</td><td>Gemma2-27B</td><td rowspan="4">1:1 acc</td><td>0.485</td><td>0.497</td><td>0.394</td><td>0.399</td></tr><tr><td>Qwen1.5-7B</td><td>0.601</td><td>0.683</td><td>0.448</td><td>0.453</td></tr><tr><td>Mixtral-8x7B-Instruct-v0.1</td><td>0.674</td><td>0.752</td><td>0.561</td><td>0.544</td></tr><tr><td>DeepSeek-V2-Lite-Chat</td><td>0.558</td><td>0.703</td><td>0.477</td><td>0.546</td></tr></table>

Table 15: Correlation with degree of overoptimization (γ) for rejected responses collected from a specific model. In the case of single pairwise comparisons, the results vary significantly depending on which model the rejected responses are collected from.

<table><tr><td rowspan="2">Design</td><td rowspan="2">Chosen</td><td rowspan="2">Rejected</td><td rowspan="2">Metric</td><td colspan="2">MetaMATH-Mistral-7B</td><td colspan="2">Llama3-8B-Instruct</td></tr><tr><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td><td> $\gamma_{gold}$ </td><td> $\gamma_{oracle}$ </td></tr><tr><td rowspan="2">H</td><td rowspan="2">GPT-4o*</td><td rowspan="2">random, 1 random, 1 (other seed)</td><td rowspan="2">1:1 acc</td><td>0.720</td><td>0.820</td><td>0.597</td><td>0.601</td></tr><tr><td>0.719</td><td>0.819</td><td>0.593</td><td>0.598</td></tr><tr><td rowspan="2">J</td><td rowspan="2">GPT-4o*</td><td rowspan="2">random, 3 random, 3 (other seed)</td><td rowspan="2">1:3 acc</td><td>0.838</td><td>0.915</td><td>0.738</td><td>0.740</td></tr><tr><td>0.847</td><td>0.914</td><td>0.737</td><td>0.740</td></tr><tr><td rowspan="2">L</td><td rowspan="2">GPT-4o, style</td><td rowspan="2">random, 3 random, 1 (other seed)</td><td rowspan="2">3:3 matrix</td><td>0.620</td><td>0.678</td><td>0.691</td><td>0.687</td></tr><tr><td>0.643</td><td>0.704</td><td>0.737</td><td>0.699</td></tr></table>

Table 16: Correlation with degree of overoptimization (γ) across different random seeds. The results demonstrate consistent correlation patterns regardless of random seed variations.

Figure 14: Prompt used for should-respond task in safety evaluation.

Figure 15: Prompt used for should-refuse task in safety evaluation.
