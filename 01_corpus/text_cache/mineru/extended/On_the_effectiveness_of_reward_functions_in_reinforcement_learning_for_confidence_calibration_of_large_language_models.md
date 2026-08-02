# On the effectiveness of reward functions in reinforcement learning for confidence calibration of large language models

Chee Heng Tan School of Computing National University of Singapore e0764286@u.nus.edu

Mehul Motani Department of Electrical & Computer Engineering National University of Singapore motani@nus.edu.sg

Institute of Advanced Intelligence and Computing Agency for Science, Technology and Research lin\_zhuoyi@a-star.edu.sg

Wee Sun Lee School of Computing National University of Singapore leews@comp.nus.edu.sg

## Abstract

In this paper, we consider the setting where large language models (LLMs) are trained using reinforcement learning (RL) to simultaneously improve reasoning accuracy and verbalize its confidence. Our reward scheme uses two functions for rewarding confidence verbalized by the LLM: one when the LLM is correct and a different one when the LLM is incorrect. With a poorly designed reward scheme, the LLM may be incentivized to answer incorrectly so that it can be confident that its answer is indeed incorrect, a phenomenon that we call confidence reward hacking. We propose the concept of non-hackable confidence reward schemes and define a spectrum of such reward schemes for RL confidence calibration training in LLMs. We demonstrate that selective confidence reward hacking can occur in practical datasets with reward schemes that are not designed to be non-hackable. We also demonstrate that the reward scheme with the best calibration to accuracy tradeoff depends on the dataset and the application, and propose using the reward scheme as a hyperparameter to optimize the tradeoffs in accordance to what is important for the application. The code of our experiments is available in https://anonymous. 4open.science/r/rl-confidence-calibration-9ED4/README.md.

## 1 Introduction

Large language models (LLM)s have made significant recent advances in reasoning, especially when using reinforcement learning (RL) [1]. However, much of the RL literature has mainly focused on the RL with verifiable rewards setting (RLVR) [2], [3], which does not directly attempt to estimate the likelihood of hallucinations via confidence calibration.

Confidence calibration has applications such as event forecasting [4] and determining whether the user is trying to factually deceive the LLM [5]. Pretrained LLMs have been demonstrated to be overconfident [6], [7] and the same is true for LLMs trained with RL without confidence calibration [8], [9]. Overconfidence in LLM can mislead end-users into thinking that the output of the LLM is almost certainly correct and can be trusted even though the output should still be viewed with skepticism. For instance, in medical LLM assistants, LLM overconfidence in a negative diagnosis may result in doctors underestimating the risk of cancer development during screening, which may lead to insufficient precautions to manage cancer risk.

Confidence calibration using RL shows potential as RL usually exhibits better out-of-distribution generalization capabilities [10] and lesser forgetting [11]. However, it remains an underexplored topic. RL calibration fine-tuning using proper scoring methods such as log loss [12] or Brier score [13] is possible, but doing so in isolation may improve calibration at the expense of answer accuracy. It is possible to define a function that rewards a combination of accuracy and calibration. However, this may incentivize the LLM to answer incorrectly so that it can be confident that its answer is indeed incorrect. We call this phenomenon confidence reward hacking. In [6], a provably non-hackable scheme, combining a constant correctness reward with the Brier score, is proposed.

In this paper, we define a reward scheme as a pair of confidence-dependent functions that, respectively, provide the rewards for answering correctly and answering incorrectly. We theoretically characterized the set of non-hackable confidence reward schemes, which are reward schemes that reward more for being confidently incorrect, penalize more for being confidently incorrect and encourage the LLM to be honest with its confidence and to answer to the best of its ability. To support our theoretical analysis, we empirically demonstrate that LLMs trained with RL using reward schemes outside the set can answer some questions incorrectly to maximize the reward.

Subsequently, we define a spectrum of reward schemes containing reward schemes where underconfi dence is preferred and where overconfidence is preferred. Such schemes may be useful when there is a need to trade off accuracy with calibration, e.g. when the LLM has limited capacity or where the optimization process is difficult, resulting in the LLM being unable to provide the optimal solution to both requirements.

Lastly, we examine the tradeoffs and demonstrate that different reward schemes can lead to better accuracy-calibration tradeoffs, suggesting that the choice of the reward scheme is a hyperparameter that can be used to optimize the tradeoff depending on what is important for the application.

## 2 Related Work

Proper scoring methods have been used for confidence calibration in RL of LLM. In [12], a clipped log-loss reward function is used over the confidence scores of the LLM while in [13], RL with a Brier-based reward function is used after supervised finetuning (SFT) using empirical correctness rates. In both papers, accuracy is not optimized and is treated as secondary to confidence calibration.

In [14], it is shown that two stage training, first for accuracy, then for confidence calibration results in reward hacking when using the standard Group Relative Policy Optimization (GRPO) [15] loss. To address the issue, a combination of a correctness reward and a confidence alignment reward denoting how close the output confidence is to the question-wise empirical accuracy of the model is proposed. While experimentally successful, this work did not show theoretically that the reward functions are non-hackable. We provide mathematical foundations for understanding non-hackability in this paper.

Recently, Damani et al. [6] proposed a reward scheme which combines accuracy with Brier score and proved that it incentivizes both answering correctly and accurate confidence calibration. More generally, in their theoretical derivation, they considered reward schemes with constant correctness rewards. Notably, [16] had also examined the theory of proper scoring reward schemes that also incentivize answering correctly, and obtained theoretical results similar to [6] in their analysis. To the best of our knowledge, Damani et al. [6] were the first to successfully utilize RL for simultaneously learning to reason and performing confidence calibration. Despite experimenting with reward schemes that allow for confidence reward hacking, they did not show concrete evidence of reward hacking in practical datasets when trained simultaneously for correctness and calibration [6], leaving open the question of whether reward hacking occurs in practice. Moreover, they considered only reward schemes with constant correctness reward. We generalize their work to more general reward schemes where the correctness reward can vary by confidence. We show the presence of reward hacking in practical datasets and further examine the trade-off between accuracy and calibration for different non-hackable confidence reward schemes.

Similar reward functions have been proposed in [17] and [18] based on the behavioral calibration framework of [19]. While [18] proposed a family of reward schemes proven to incentivize both answering accurately and accurate confidence calibration, we demonstrate the existence of other reward schemes satisfying both properties and characterize the set of all such reward schemes. More details can be found in Appendices A.1, A.2 and A.3.

## 3 Problem Formulation

Our goal is to investigate the effect of reward functions in RL when LLMs are trained for confidence calibration. We consider the following RL setting:

For each question, the LLM is tasked to answer the question and provide a confidence value $c \in ( 0 , 1 )$ We assume that the questions have a deterministic and well-defined set of correct answers. The LLM obtains a reward of $\bar { \boldsymbol { f } } ( \boldsymbol { c } )$ for answering correctly and a reward of $g ( c )$ for answering incorrectly. The objective of the LLM is to maximize its reward. Note that this differs from the setting in [6] since the reward for correctness is no longer constrained to be independent of c.

Definition 1. We define a “binary correctness confidence reward scheme”, or “reward scheme”for short, as a pair offunctions $( f ( \dot { c } ) , g ( c ) )$ defined over $( 0 , 1 )$ , where f and g have continuous first derivatives. The corresponding "binary correctness confidence rewardfunction", or "rewardfunction" for short, is $r ( q , a , c ) = i s C o r r e c t ( q , a ) f ( c ) + ( 1 - i s C o r r e c t ( q , a ) ) g ( c )$ where $i s C o r r e c t ( q , a )$ is 1 if the answer a to question q is correct and 0 otherwise.

A reward function can be viewed as the composition of the reward scheme $( f ( c ) , g ( c ) )$ and the evaluator isCorrect, where $f ( c )$ is returned if isCorrect $( q , a ) = 1$ and g(c) otherwise

If the reward scheme is not properly designed, the LLM may select some questions to intentionally answer wrongly with low confidence to obtain a higher reward, a phenomenon we term “confidence reward hacking”. For more details on how this may occur, refer to Remark 1 in Appendix A.1. Therefore, our first research question aims to investigate the possibility of selective confidence reward hacking over practical reasoning datasets.

Infinitely many reward schemes exist such that confidence reward hacking never happens. We term such reward schemes as “non-hackable confidence”. Such reward schemes may not offer the same accuracy and confidence calibration performance due to the limited approximation capability of neural networks and optimization capability of gradient descent optimizers. Therefore, our second research question aims to examine the accuracy and calibration tradeoffs as well as the convergence speed for different non-hackable confidence reward schemes.

We aim to investigate and answer the following research questions:

• RQ1: Can the use of a hackable confidence reward scheme on $( 0 , 1 )$ result in LLMs selectively choosing questions to intentionally answer incorrectly?

• RQ2: What are the accuracy and confidence calibration tradeoffs in using different nonhackable confidence reward schemes for RL training in LLM?

## 4 Methodology

In this paper, we propose the concept of non-hackable confidence reward schemes, designed to encourage the LLM to answer to the best of its ability and to be honest about its confidence. Subsequently, we designed a spectrum of reward schemes with one end where it is better to be overconfident when unsure, and the other end where it is better to be underconfident when unsure.

## 4.1 Non-hackable Confidence Reward Schemes

A confident and correct answer should be rewarded more than an unconfident and correct answer. Likewise, a confident and incorrect answer should be rewarded less than an unconfident and incorrect answer. Ideally, we would want the LLM to answer to the best of its ability, and yet understand its own limitations by providing an informative estimate of its epistemic uncertainty. This notion, refined from [6] via the inclusion of the Interpretability property, is formalized as follows:

Definition 2. We define a “non-hackable confidence reward scheme” over an open interval $( a , b ) \subseteq$ (0, 1) as a pair of functions $( f ( c ) , g ( c ) )$ satisfying the following properties:

(i) Interpretability: $f ( c )$ is non-decreasing on $( 0 , 1 )$ and $g ( c )$ is non-increasing on $( 0 , 1 )$

(ii) Proper Scoring: Let $R ( c , p ) = \mathbb { E } _ { q \sim B e r n o u l l i ( p ) } [ q f ( c ) + ( 1 - q ) g ( c ) ] = p f ( c ) + ( 1 - p ) g ( c ) ,$ defined over $c \in ( 0 , 1 )$ and $p \in [ 0 , 1 ]$ . Then,for all $p \in ( 0 , 1 ) , p \in a r g m a x _ { c } R ( c , p )$

(iii) Best Effort: Let $R _ { m a x } ( c ) = R ( c , c )$ . Then, $R _ { m a x } ( c )$ is non-decreasing in $( a , b )$

If the open interval is not explicitly mentioned, it is taken to be over $( 0 , 1 )$ . Corresponding reward functions of non-hackable confidence reward schemes are termed “non-hackable confidence reward functions”.

Non-hackable confidence reward schemes are characterized as follows, with proof in Appendix A.1: Theorem 1. A reward scheme is non-hackable confidence over $( a , b ) \subseteq ( 0 , 1 )$ ifand only ifall ofthe following hold:

(i) There exists a function $h ( c )$ defined over (0, 1) such that $\begin{array} { r } { h ( c ) = \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$

(ii) $h ( c ) \leq 0 .$ for all $c \in ( 0 , 1 )$

(iii) $f ( a ^ { + } ) \geq g ( a ^ { + } )$

Intuitively, property (iii) means that the LLM is never incentivized to answer incorrectly, e.g. by intentionally answering incorrectly, provided the confidence output of the LLM is restricted to $( a , b )$ The standard binary reward for correctness (1, 0) and the Brier-based reward scheme $( 1 - { \overset { . } { ( } } 1 -$ $c ) ^ { 2 } , - c ^ { 2 } )$ , originally proposed by [6], are examples of non-hackable confidence reward schemes. Table 1 shows more examples of reward schemes and whether they are non-hackable confidence.

## 4.2 Overconfidence to Underconfidence Spectrum

In this subsection, we define an overconfidence to underconfidence spectrum of non-hackable confidence reward schemes. To understand the spectrum, we first define the miscalibration penalty. Intuitively, the miscalibration penalty is the loss in expected reward due to outputting a sub-optimal confidence value c instead of the true probability p of answering the question correctly.

Definition 3. The miscalibration penaltyfor a reward scheme $( f ( c ) , g ( c ) )$ , denoted as $R _ { p e n } ( c , p )$ , is defined as $R _ { p e n } ( c , p ) = R ( p , p ) - R ( c , p )$ .

$H c > p ,$ , the miscalibration penalty can be termed as “overconfidence penalty". $H c < p ,$ the miscalibration penalty can be termed as “underconfidence penalty".

Definition 4. We define an overconfidence to underconfidence spectrum ofnon-confidence hackable reward schemes asfollows, where the mentioned reward schemes are as defined in Table 1. In order from the overconfident end to the underconfident end of the spectrum, reading from top to bottom, we have thefollowing reward schemes:

• Correctness-only

• Overconfidence-k, $k > 0 ,$ , larger k indicates more towards the overconfident end.

• Brier-1

• Underconfidence- $k , k > 0$ , larger k indicates more towards the underconfident end.

• Brier-log Hybrid<sup>1</sup>

We also introduce the notion of overconfidence and underconfidence bias. Intuitively, a reward scheme exhibits overconfidence bias when the LLM is penalized less by being slightly overconfident compared to being slightly underconfident. Definition 5 formalizes this notion.

Definition 5. A non-hackable confidence reward scheme $( f ( p ) , g ( p ) )$ has overconfidence bias (respectively, underconfidence bias) ifthe miscalibration penalty $R _ { p e n }$ satisfies thefollowing property: For all $\delta > 0 a n d p \in ( 0 , 1 )$ ) such that $p { + } \delta \in ( 0 , 1 )$ $a n d p \ – \delta \in ( 0 , 1 ) , R _ { p e n } ( p + \delta , p ) < R _ { p e n } ( p - \delta , p )$ (respectively $R _ { p e n } ( p + \delta , p ) > R _ { p e n } ( p - \delta , p ) )$

Note that $R _ { p e n }$ is as defined in Definition 3.

Figure 1 shows the miscalibration penalty plot for some reward schemes along the overconfidence to underconfidence spectrum. Brier-log Hybrid and Underconfidence-k are towards the underconfident end of the spectrum and exhibit underconfidence bias. Overconfidence-k is towards the overconfident end of the spectrum and exhibits overconfidence bias.

Both Brier-1 and Correctness-only have neither overconfidence nor underconfidence bias. Even though Correctness-only does not have overconfidence bias, Overconfidence-k converges pointwise to Correctness-only over $( 0 , 1 )$ as $k \to \infty$ , which justifies the position of Correctness-only furthest in the overconfident end of the spectrum. A rigorous derivation can be found in Appendix D.

Figure 1: Miscalibration penalty of Correctness-only, Overconfidence-1000, Overconfidence-4, Overconfidence-1, Brier-1, Underconfidence-1, Underconfidence-4 and Brier-log Hybrid reward schemes. The reward schemes are as defined in Table 1. Darker greys mean higher miscalibration penalty, which occur when the confidence c differs more significantly from the probability p of answering correctly. Blue line represents ideal confidence calibration. Green and red line represents being consistently underconfident and overconfident by 0.2 respectively.

## 5 Experimental Settings

In this section, we describe the datasets, reward schemes and the procedure for our experiments.

## 5.1 Datasets Used

For our experiments, we used the HotpotQA [20], HotpotQA-Modified (inspired by [6]), BigMath [21] and DeepMath-103K [22] datasets. HotpotQA is a textual reasoning dataset where the LLM is given sources and required to reason about the sources to provide the answer. HotpotQA-Modified is a variant of HotpotQA where two sources are removed at random. Questions with one or both removed sources being relevant are harder to answer because the LLM is required to answer based on its trained knowledge. BigMath and DeepMath-103K are math reasoning datasets. More details of the datasets used and their preprocessing can be found on Appendix E.

## 5.2 Reward Schemes Used

For our experiments, we use the reward schemes shown in Table 1. Let $R _ { R Q 2 } = \{ \mathrm { C o r r e c t n e s s } -$ only, Overconfidence-1000, Overconfidence-4, Overconfidence-1, Brier-1, Underconfidence-1, Underconfidence-4, Brier-log Hybrid} denote the set of non-hackable confidence reward schemes that we will examine in our experiments. Justifications behind why each of the reward schemes are or are not hackable can be found in Appendix C.

For $\mathrm { L o g } { - } k ,$ we tested $k \in \{ 1 , \frac { 1 } { \ln { 2 0 2 } } \}$ . For Brier- $\cdot k ,$ we tested $k \in \{ 1 , 2 \}$ . For Overconfidence- $\cdot k ,$ , we tested $\bar { k } \in \{ 1 , 4 , 1 0 0 0 \}$ . For Underconfidence-k, we tested $k \in \{ 1 , 4 \}$ . All reward schemes in Table 1 except Log Loss and Brier Score satisfy $f ( 1 ^ { - } ) = 1$ and $g ( 0 ^ { + } ) = 0 .$ , which implies $R _ { m a x } ( 1 ^ { - } ) = 1$ and $\bar { R _ { m a x } } ( \bar { 0 } ^ { + } ) = 0$ . This ensures a consistent reward scale for a fair comparison among reward schemes with correctness rewards.

## 5.3 Experimental Procedure

We ran our experiments on Qwen 2.5 (3B) Instruct [23], [24]. The LLM was fine-tuned using SFT on a reformatted version of its own responses to help it follow the RL answering format. For each reward scheme, we ran RL training on the LLM using Dr GRPO [25]. We evaluated the model on Accuracy, Expected Calibration Error (ECE), Area Under Receiver Operating Characteristic Curve (AUROC), Brier score, average Brier-1 reward and Calibration bias. Calibration bias is defined as accuracy minus average confidence. A negative calibration bias indicates overconfidence while a positive calibration bias indicates underconfidence. More details on the metrics used can be found on Appendix E.7. For full details on the experimental procedure, refer to Appendix E.1.

Table 1: Table of reward schemes used in experiments. $f ( c )$ is the reward for answering correctly while g(c) is the reward for answering incorrectly. Details on whether reward schemes listed are non-hackable confidence can be found in Appendix C.

<table><tr><td>Reward Scheme</td><td> $f(c)$ </td><td> $g(c)$ </td><td>Non-hackable confidence?</td></tr><tr><td>Correctness-only</td><td>1</td><td>0</td><td>Yes</td></tr><tr><td>Log- $k$ </td><td> $1 + k \ln(c)$ </td><td> $k \ln(1 - c)$ </td><td>No</td></tr><tr><td>Log Loss</td><td> $\ln(c)$ </td><td> $\ln(1 - c)$ </td><td>No</td></tr><tr><td>Brier- $k$ </td><td> $1 - k(1 - c)^{2}$ </td><td> $-kc^{2}$ </td><td>Only for  $k \leq 1$ </td></tr><tr><td>Brier Score</td><td> $-(1 - c)^{2}$ </td><td> $-c^{2}$ </td><td>No</td></tr><tr><td>Brier-Log Hybrid</td><td> $c$ </td><td> $c + \ln(1 - c)$ </td><td>Yes</td></tr><tr><td>Overconfidence- $k$ </td><td> $\frac{(k+1) \ln(ck+1)-ck}{(k+1) \ln(k+1)-k}$ </td><td> $\frac{\ln(ck+1)-ck}{(k+1) \ln(k+1)-k}$ </td><td>Yes</td></tr><tr><td>Underconfidence- $k$ </td><td> $\frac{kc+\ln(1-\frac{kc}{1+k})}{k-\ln(1+k)}$ </td><td> $\frac{kc+(k+1)\ln(1-\frac{kc}{1+k})}{k-\ln(1+k)}$ </td><td>Yes</td></tr></table>

## 6 Experimental Results

In this section, we analyze our experimental results to answer RQ1 and RQ2. The full results, containing performance statistics by difficulty split, can be found in Appendix F.

Table 2: Accuracy results for BigMath, DeepMath-103K, HotpotQA-Modified and HotpotQA for different reward schemes, split by difficulty. Bolded reward schemes are non-hackable confidence. The reward schemes are subdivided into three categories from top to bottom: independent of confidence, hackable confidence and other non-hackable confidence.

<table><tr><td rowspan="2">Reward Scheme</td><td colspan="3">BigMath</td><td colspan="3">DeepMath-103K</td><td colspan="3">HotpotQA-Modified</td><td>HotpotQA</td></tr><tr><td>easy</td><td>medium</td><td>hard</td><td>easy</td><td>medium</td><td>hard</td><td>easy</td><td>medium</td><td>hard</td><td>hard</td></tr><tr><td>Correctness-only</td><td>0.9540</td><td>0.7545</td><td>0.3229</td><td>0.5308</td><td>0.3940</td><td>0.4189</td><td>0.6273</td><td>0.4250</td><td>0.2100</td><td>0.6571</td></tr><tr><td>Log Loss</td><td>0.0001</td><td>0.0001</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0003</td><td>0.0001</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Brier Score</td><td>0.0003</td><td>0.0001</td><td>0.0000</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.0000</td><td>0.0001</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Log-1</td><td>0.9388</td><td>0.6981</td><td>0.2748</td><td>0.0001</td><td>0.0000</td><td>0.0000</td><td>0.2022</td><td>0.0619</td><td>0.0204</td><td>0.6472</td></tr><tr><td>Brier-2</td><td>0.9530</td><td>0.7520</td><td>0.3185</td><td>0.0000</td><td>0.0001</td><td>0.0004</td><td>0.5296</td><td>0.2323</td><td>0.0958</td><td>0.6507</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.9501</td><td>0.7427</td><td>0.3139</td><td>0.5334</td><td>0.4059</td><td>0.4260</td><td>0.6271</td><td>0.4203</td><td>0.2093</td><td>0.6537</td></tr><tr><td>Overconfidence-1000</td><td>0.9529</td><td>0.7506</td><td>0.3172</td><td>0.5262</td><td>0.3886</td><td>0.4197</td><td>0.6228</td><td>0.4135</td><td>0.2050</td><td>0.6548</td></tr><tr><td>Overconfidence-4</td><td>0.9509</td><td>0.7474</td><td>0.3169</td><td>0.5324</td><td>0.4054</td><td>0.4167</td><td>0.6248</td><td>0.4187</td><td>0.2070</td><td>0.6526</td></tr><tr><td>Overconfidence-1</td><td>0.9521</td><td>0.7532</td><td>0.3211</td><td>0.5205</td><td>0.3855</td><td>0.4063</td><td>0.6328</td><td>0.4238</td><td>0.2055</td><td>0.6507</td></tr><tr><td>Brier-1</td><td>0.9529</td><td>0.7480</td><td>0.3117</td><td>0.5108</td><td>0.3796</td><td>0.3976</td><td>0.6317</td><td>0.4184</td><td>0.2028</td><td>0.6504</td></tr><tr><td>Underconfidence-1</td><td>0.9505</td><td>0.7444</td><td>0.3146</td><td>0.5191</td><td>0.3876</td><td>0.4105</td><td>0.6245</td><td>0.4197</td><td>0.2110</td><td>0.6551</td></tr><tr><td>Underconfidence-4</td><td>0.9552</td><td>0.7573</td><td>0.3279</td><td>0.4948</td><td>0.3613</td><td>0.3911</td><td>0.6199</td><td>0.4158</td><td>0.2008</td><td>0.6529</td></tr><tr><td>Brier-log-hybrid</td><td>0.9513</td><td>0.7460</td><td>0.3169</td><td>0.3312</td><td>0.2263</td><td>0.2919</td><td>0.6111</td><td>0.4003</td><td>0.1915</td><td>0.6439</td></tr></table>

## 6.1 RQ1: Presence of Selective Reward Hacking

To answer RQ1, we demonstrate the presence of confidence reward hacking in reward schemes that allow such hacking, which can occur to varying degrees in practice. Table 2 shows the accuracy statistics for BigMath, DeepMath-103K, HotpotQA-Modified and HotpotQA, split by question difficulty, where the LLM is trained under different reward schemes.

The LLM trained with the Log Loss and Brier Score reward schemes obtained a near-zero accuracy, highlighting the necessity of adding a correctness reward (which may vary by confidence) when both answering ability and confidence calibration ability are simultaneously trained.

Compared to the Correctness-only reward scheme, Log-1 reward scheme demonstrates more severe accuracy loss in HotpotQA-Modified and BigMath than in the other examined non-hackable confidence reward schemes. The accuracy decrease in relative terms is more significant with increasing difficulty in the test set (e.g. 14.9% on hard, 7.5% on medium and 1.6% on easy relative to Correctness-only for BigMath), suggesting that the LLM has learnt to give up answering the questions it finds harder to answer. The result for Brier-2 reward scheme is qualitatively similar to Log-1 for HotpotQA-Modified but Brier-2 maintained similar accuracy as Correctness-only for BigMath, suggesting that Brier-2 is less prone to confidence reward hacking than Log-1.

Both Brier-2 and Log-1 satisfy $R _ { m a x } ( 0 ^ { + } ) = 0 , R _ { m a x } ( c ) < 0$ for all $c \in ( 0 , p )$ and $R _ { m a x } ( c ) > 0$ for all $c \in \mathsf { \Gamma } ( p , 1 )$ for some $p \in \mathsf { \Gamma } ( 0 , 1 )$ . The value of $p$ for Brier-2 is ${ \frac { 1 } { 2 } } ,$ , which is smaller than $p \approx 0 . 6 4 8 4 2$ for Log-1. The confidence threshold below which it is optimal for the LLM to always answer incorrectly with confidence approaching $0 ^ { + }$ is lower for Brier-2, hence Brier-2 is less prone to confidence reward hacking.

During the final step of the training run, as evidence of selective confidence reward hacking, the LLM has occasionally given up on answering the question, giving responses such as “Unknown" with confidence 0. Appendix G shows a case study of selective confidence reward hacking.

In contrast, no evidence of confidence reward hacking occurred for $\mathrm { L o g } { - } { \frac { 1 } { \ln { 2 0 2 } } }$ as the accuracy remained similar to Correctness-only. $\mathrm { L o g  – \frac { 1 } { l n 2 0 2 } }$ is a non-hackable confidence reward scheme over $\textstyle { \left( { \frac { 1 } { 2 0 3 } } , 1 \right) }$ but not $( 0 , 1 )$ . In practice, we can clip the confidence values to $\begin{array} { r } { [ \frac { 1 } { 2 0 3 } , 1 - \frac { 1 } { 2 0 3 } ] } \end{array}$ to prevent confidence reward hacking, similar to what is proposed in Equation 4 of [17], and the “truncated logarithmic scoring system” in [16, p. 137].

Overall, this suggests that the presence and extent of reward hacking depends on the difficulty distribution of the dataset and the reward scheme used. Model size influences reasoning ability of the model [26]. Both reasoning ability of the model and dataset distribution influence the difficulty distribution of the dataset. Therefore, the findings indirectly support the hypothesis in [6, p. 23].

## Answer to RQ1:

Confidence reward hacking, selective or total, can occur when a hackable confidence reward scheme is used. Its occurrence and type depends on the difficulty distribution of the dataset and the reward scheme used.

## 6.2 RQ2: Accuracy and Calibration Tradeoffs

To answer RQ2, we demonstrate using Figures 2, 3 and 4 that the best reward scheme within the overconfidence to underconfidence spectrum depends on both the dataset and the metric used. Subsequently, we demonstrate using Figure 5 that the reward scheme influences the training speed of the LLM in both accuracy and confidence calibration.

Figure 2 shows the accuracy, AUROC and Brier-1 reward metric for the reward schemes in $R _ { R Q 2 }$ when ordered from the overconfident end to the underconfident end of the spectrum (from left to right of plot). Brier-1 is an integrated metric which takes into account both accuracy and confidence calibration, simulating one possible application where accuracy and confidence are both important.

As one moves towards the underconfident end of the spectrum, the accuracy is either maintained or generally decreases. The decrease in accuracy is most notable in DeepMath-103K and is the least pronounced in BigMath. In contrast, AUROC and Brier-1 initially improve, but may deteriorate or level off as one moves towards the underconfident end of the spectrum. The degree of deterioration depends on the dataset and the metric. Underconfidence-4 and Overconfidence-4 have the best


Figure 2: Accuracy, AUROC and Brier-1 performance metrics for BigMath, DeepMath, HotpotQA and HotpotQA-Modified datasets (from left to right) for non-hackable confidence reward schemes ordered in the spectrum from the overconfident end to the underconfident end (from left to right). For all three performance metrics, higher is better.

Brier-1 performance in BigMath and DeepMath-103K respectively while Brier-log Hybrid and Overconfidence-1 have the best AUROC in BigMath and DeepMath-103K respectively, highlighting that the best reward scheme depends on both the dataset and the metric.




Figure 3: Brier score, ECE (10 bins) and Calibration bias statistics for BigMath, DeepMath-103K, HotpotQA and HotpotQA-Modified datasets (from left to right) for non-hackable confidence reward schemes ordered in the spectrum from the overconfident end to the underconfident end (from left to right). For all three performance metrics, closer to 0 (red dotted line) is better.


Figure 4: Confidence calibration plots of the LLM after RL training with Correctness-only, Overconfidence-1000, Brier-1 and Brier-log Hybrid reward schemes (from left to right, in order from overconfident end to underconfident end of spectrum) from the BigMath dataset. Each question in the test set was evaluated with 16 different responses. Confidence values are subdivided into 10 equally spaced bins, with tie-breaking at boundaries to the right. The label of each point represents the size of each bin. The ideal calibration line is in red.

Figure 3 shows the Brier score, ECE (10 bins) and calibration bias for the reward schemes in $R _ { R Q 2 }$ when ordered from the overconfident end of the spectrum to the underconfident end (from left to right of plot). As one moves towards the underconfident end of the spectrum, both Brier score and ECE generally improve before leveling off around Overconfidence-4. As one moves toward the underconfident end of the spectrum, the calibration bias generally becomes less negative. This is further supported by the confidence calibration plots in Figure 4 depicting the LLM accuracy in each confidence bin when trained and evaluated on the BigMath dataset.

Figure 5: Training accuracy (left) and training confidence (right) statistics for DeepMath-103K from Step 10 to Step 499 (0-indexed), smoothed using an exponential moving average [27, p. 15], both initialized at 0.5 with smoothing factor of 0.2.

Figure 5 shows the training accuracy and training confidence of Qwen 2.5 (3B) Instruct when trained with different non-hackable confidence reward schemes in DeepMath-103K. The training speed is generally slower towards the underconfident end of the spectrum, while the confidence calibration is generally faster apart from a significant anomaly in Brier-Log Hybrid. Speed of confidence calibration is important in resource-limited applications where confidence calibration is prioritized over accuracy.

While the initial LLM, i.e. just after the SFT stage, is overconfident, its AUROC is above 0.5 (see Appendix F for full results), suggesting that answers with lower confidence are less likely to be correct. With higher confidence penalty at high confidences towards the underconfident end of the spectrum, the LLM is incentivized to generate responses with lower confidence as these responses have higher expected reward. Such answers are less likely to be correct, leaving the LLM with fewer correct responses to learn from, which slows down the improvements in training accuracy.

The general trend in the confidence calibration speed is likely due to the higher miscalibration penalties towards the underconfident end of the spectrum when initial confidence is high. The anomaly is likely due to relatively low miscalibration penalties at low confidences and low probability of answering correctly in DeepMath-103K when using the Brier-log Hybrid reward scheme.

As the priority metric varies by application, the optimal choice of reward scheme depends on the application. Compute budget may influence the optimal choice as different reward schemes have different training speeds and confidence calibration speeds. Therefore, we suggest adopting the reward scheme as a hyperparameter to tune the tradeoffs.

## Answer to RQ2:

Accuracy and speed of training accuracy improvements generally improve towards the overconfident end of the spectrum. Brier score generally improves and calibration bias generally becomes less negative towards the underconfident end of the spectrum. The optimal reward scheme for AUROC and Brier-1 is generally in between Overconfidence-4 and Brier-log Hybrid (both inclusive), and depends on both the dataset and the metric. As one moves towards the underconfident end of the spectrum, the speed of confidence calibration generally improves but may slow down if the initial probability of answering correctly is low. The optimal tradeoff within the spectrum depends on the dataset and the application since different applications require prioritization of different metrics.

## 7 Limitations and Future Work

In this section, we analyze the limitations of our work and provide future research directions.

First, correctness evaluation is not perfect for practical datasets. Therefore, evaluation noise may seep into confidence estimates, leading to post-training biases.

Second, this work only applies to scenarios where an answer can be considered completely correct, or completely wrong. Future work can generalize to scenarios involving partially correct answers earning partial credit, categorical distribution of possible correct answers and long-form answers where only some sentences, but not others, are factually correct.

Third, this work is dependent on the stability of RL training of LLM. Future work can investigate into the loss landscapes of different reward schemes to improve learning stability.

## 8 Conclusion

In this paper, we examined RL for LLMs in the setting where the LLM is trained to jointly improve reasoning accuracy and verbalized confidence calibration. We identified a failure mode, which we call confidence reward hacking, in which a poorly designed reward scheme can incentivize the LLM to provide an answer it can confidently recognize as incorrect. Therefore, rewarding confidence naively can create incentives that conflict with accuracy.

To address this issue, we introduced non-hackable confidence reward schemes, which removes the incentive for an LLM to trade correctness for high confidence in incorrectness. Furthermore, we introduced an overconfidence to underconfidence spectrum of reward schemes which contain reward schemes that favor overconfidence by δ over underconfidence by δ and rewards that favor underconfidence by δ over overconfidence by δ.

Empirically, we showed that hackable confidence reward schemes can lead to selective confidence reward hacking on practical datasets, with accuracy degradation concentrated on harder questions compared with training for accuracy alone. We also found that no single non-hackable confidence reward scheme consistently performs best across different metrics and different datasets. Thus, the reward scheme should be tuned as a hyperparameter according to the calibration and accuracy requirements of the target application.

## References

[1] K. Zhang et al., A Survey of Reinforcement Learning for Large Reasoning Models, 2025. arXiv: 2509.08827 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2509.08827

[2] C. Wang, Z. Li, J. Bai, Y. Zhang, S. Cui, Z. Zhao, and Y. Wang, Arbitrary entropy policy optimization breaks the exploration bottleneck ofreinforcement learning, 2025. arXiv: 2510. 08141 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2510.08141

[3] Y. Wu et al., It Takes Two: Your GRPO Is Secretly DPO, 2026. arXiv: 2510.00977 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2510.00977

[4] B. Turtel, D. Franklin, K. Skotheim, L. Hewitt, and P. Schoenegger, “Outcome-based Reinforcement Learning to Predict the Future,” Transactions on Machine Learning Research, 2025, ISSN: 2835-8856. [Online]. Available: https://openreview.net/forum?id=bbhdeL8EUX

[5] Y. Li, Y. Miao, X. Ding, R. Krishnan, and R. Padman, “Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds., Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 6679– 6700, ISBN: 979-8-89176-256-5. DOI: 10.18653/v1/2025.findings-acl.347 [Online]. Available: https://aclanthology.org/2025.findings-acl.347/

[6] M. Damani, I. Puri, S. Slocum, I. Shenfeld, L. Choshen, Y. Kim, and J. Andreas, “Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=ASQ649zdHm

[7] Y. Li, R. Krishnan, and R. Padman, Consistency ofLarge Reasoning Models Under Multi-Turn Attacks, 2026. arXiv: 2602.13093 [cs.AI]. [Online]. Available: https://arxiv.org/ abs/2602.13093

[8] X. Xie, X. Liu, and B. Yao, Know When You’re Wrong: Aligning Confidence with Correctness for LLM Error Detection, 2026. arXiv: 2603.06604 [cs.LG]. [Online]. Available: https: //arxiv.org/abs/2603.06604

[9] J. Achiam et al., GPT-4 Technical Report, 2024. arXiv: 2303.08774 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2303.08774

[10] T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma, “SFT memorizes, RL generalizes: A comparative study of foundation model post-training,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=dYur3yabMj

[11] I. Shenfeld, J. Pari, and P. Agrawal, “RL’s razor: Why online reinforcement learning forgets less,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=7HNRYT4V44

[12] D. Bani-Harouni, C. Pellegrini, P. Stangel, E. Özsoy, K. Zaripova, M. Keicher, and N. Navab, “Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=yResLmrVO1

[13] T. Xu, S. Wu, S. Diao, X. Liu, X. Wang, Y. Chen, and J. Gao, “SaySelf: Teaching LLMs to express confidence with self-reflective rationales,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds., Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 5985–5998. DOI: 10.18653/v1/2024.emnlp-main.343 [Online]. Available: https://aclanthology.org/2024.emnlp-main.343/

[14] C. Li, J. Wu, H. Zhang, Z. Chen, G. An, J. Qiu, X. Wang, and Q. Tian, Confidence Before Answering: A Paradigm Shiftfor Efficient LLM Uncertainty Estimation, 2026. arXiv: 2603. 05881 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2603.05881

[15] Z. Shao et al., DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. arXiv: 2402 . 03300 [cs.CL]. [Online]. Available: https : / / arxiv.org/abs/2402.03300

[16] E. H. Shuford Jr, A. Albert, and H. E. Massengill, “Admissible probability measurement procedures,” Psychometrika, vol. 31, no. 2, pp. 125–145, 1966.

[17] J. Wu, J. Liu, Z. Zeng, T. Zhan, T. Cai, and W. Huang, Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning, 2026. arXiv: 2512.19920 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2512.19920

[18] S. Wu, F. K. Gustafsson, E. Phillips, B. Gao, A. Thakur, and D. A. Clifton, BAS: A Decision-Theoretic Approach to Evaluating Large Language Model Confidence, 2026. arXiv: 2604. 03216 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2604.03216

[19] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, Why language models hallucinate, 2025. arXiv: 2509.04664 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2509.04664

[20] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning, “HotpotQA: A dataset for diverse, explainable multi-hop question answering,” in Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

[21] A. Albalak et al., Big-Math: A Large-Scale, High-Quality Math Datasetfor Reinforcement Learning in Language Models, 2025. arXiv: 2502 . 17387 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2502.17387

[22] Z. He et al., “DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning,” 2025. arXiv: 2504.11456 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2504.11456

[23] A. Yang et al., “Qwen2 Technical Report,” arXiv preprint arXiv:2407.10671, 2024.

[24] Qwen Team, Qwen2.5: A Party of Foundation Models, Sep. 2024. [Online]. Available: https: //qwenlm.github.io/blog/qwen2.5/

[25] Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, Understanding R1- Zero-Like Training: A Critical Perspective, 2025. arXiv: 2503.20783 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2503.20783

[26] Qwen Team, Qwen3 Technical Report, 2025. arXiv: 2505.09388 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2505.09388

[27] R. G. Brown, Exponential smoothingfor predicting demand. Arthur D. Little Inc., 1956.

[28] M. J. Schervish, “A General Method for Comparing Probability Assessors,” The Annals of Statistics, vol. 17, no. 4, pp. 1856–1879, 1989.

[29] R. C. Buck and E. F. Buck, Advanced Calculus (International Series in Pure and Applied Mathematics), 3rd ed. New York: McGraw-Hill, 1978.

[30] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” Journal ofthe American Statistical Association, vol. 102, no. 477, pp. 359–378, 2007.

[31] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, LoRA: Low-Rank Adaptation ofLarge Language Models, 2021. arXiv: 2106.09685 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2106.09685

[32] M. Bereket and J. Leskovec, Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes, 2025. arXiv: 2508.11800 [cs.LG]. [Online]. Available: https:// arxiv.org/abs/2508.11800

[33] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, On Calibration of Modern Neural Networks, 2017. arXiv: 1706.04599 [cs.LG]. [Online]. Available: https://arxiv.org/abs/1706. 04599

[34] J. Marcum, “A statistical theory of target detection by pulsed radar,” IRE Transactions on Information Theory, vol. 6, no. 2, pp. 59–267, 1960. DOI: 10.1109/TIT.1960.1057560

[35] W. Peterson, T. Birdsall, and W. Fox, “The theory of signal detectability,” Transactions of the IRE Professional Group on Information Theory, vol. 4, no. 4, pp. 171–212, 1954. DOI: 10.1109/TIT.1954.1057460

[36] G. W. Brier, “Verification of forecasts expressed in terms of probability,” Monthly weather review, vol. 78, no. 1, pp. 1–3, 1950. DOI: 10.1175/1520-0493(1950)078<0001:vofeit> 2.0.co;2

[37] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out, Barcelona, Spain: Association for Computational Linguistics, Jul. 2004, pp. 74– 81. [Online]. Available: https://aclanthology.org/W04-1013/

[38] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient Memory Management for Large Language Model Serving with PagedAttention,” in Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[39] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview. net/forum?id=Bkg6RiCqY7

[40] D. Han, M. Han, and Unsloth team, Unsloth, 2023. [Online]. Available: https://github. com/unslothai/unsloth

[41] S. Devic, C. Peale, A. Bradley, S. Williamson, P. Nakkiran, and A. Gollakota, Trace Length is a Simple Uncertainty Signal in Reasoning Models, 2025. arXiv: 2510.10409 [cs.AI]. [Online]. Available: https://arxiv.org/abs/2510.10409

[42] A. Grattafiori et al., The Llama 3 Herd of Models, 2024. arXiv: 2407 . 21783 [cs.AI]. [Online]. Available: https://arxiv.org/abs/2407.21783

## A Proofs of Theorems In Paper

## A.1 Non-hackable confidence reward schemes

In these proofs, R is as defined in Proper Scoring property of Definition 2, and $R _ { m a x }$ is as defined in Best Effort property of Definition 2. We first motivate our definition of non-hackable confidence reward schemes, then proceed to characterize the set of non-hackable confidence reward schemes.

## Remark 1. Why we consider non-hackable confidence reward schemes in thefirst place.

Consider the strategy where the LLM performs at its best and honestly reports its confidence c, aligned with the Bayesian probability of answering correctly. We want this strategy to be optimal. Its expected reward is $R _ { m a x } ( c )$

Suppose $( f ( c ) , g ( c ) )$ is a hackable confidence reward scheme. Then, there exists $c \in ( 0 , 1 )$ and $p \in ( 0 , c )$ such that $R _ { m a x } ( p ) > R _ { m a x } ( c )$ . Consider the strategy where the LLM reports a confidence ofp, and answers to the best ofits ability with probability <sup>p</sup> and answers gibberish, assumed to be a wrong, otherwise. Then, the probability of the LLM answering correctly is $\textstyle { \frac { p } { c } } c + ( 1 - { \frac { p } { c } } ) ( 0 ) = p .$ Therefore, the expected reward of this strategy is $R _ { m a x } ( p ) > \bar { R _ { m a x } } ( c )$

Therefore, the LLM can obtain a higher maximum expected reward by intentionally answering wrongly with some probability, for instance, by answering some gibberish. In practice, this defeats the purpose ofdoing RL to improve LLM reasoning, as this loophole encourages the LLM to answer incorrectly. Since such a p may only existfor a proper subset ofquestions in the dataset, it is possible that the LLM will selectively choose the subset of questions to answer incorrectly, which may lead to difficult debugging or even non-detection of bugs if the overall accuracy happens to increase during the training.

Lemma 1. Reward schemes $( f ( c ) , g ( c ) )$ that satisfy the Proper Scoring property over $( 0 , 1 )$ must necessarily satisfy $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$ over (0, 1).

Remark 2. A related result, i.e. both Lemma 1 and its converse hold assuming the Interpretability property, was proven in [16] and generalized by [28], with varying assumptions about $f ( c )$ and $g ( c )$ . In this section, we take a slightly different approach and prove Lemma 1 in the context of characterizing non-hackable confidence reward schemes.

## Proof of Lemma 1.

Consider $R ( c , p ) = p f ( c ) + ( 1 - p ) g ( c )$ , where p is treated as constant. Since f and g are assumed to be have continuous first derivatives, R is partially differentiable with respect to c. Differentiating with respect to $R ,$ we obtain $\begin{array} { r } { \frac { \partial R } { \partial c } = p f ^ { \prime } ( c ) \dot { + } ( 1 - \dot { p } ) g ^ { \prime } ( c ) } \end{array}$

By the Proper Scoring property, since $R ( c , p )$ obtains its maximum at $c = p$ when keeping p constant, $p \mathbf { \bar { f } } ^ { \prime } ( p ) + ( \mathbf { \bar { 1 } } - p ) g ^ { \prime } ( \bar { p } ) = \mathbf { \bar { 0 } }$ for all $p \in ( 0 , 1 )$ . Therefore, $p f ^ { \prime } ( p ) = ( p - 1 ) g ^ { \prime } ( p )$ . Since $p \in ( 0 , 1 )$ , $p \neq 0$ and $p - 1 \neq 0$ , hence we divide by $p ( p - 1 )$ on both sides to obtain $\begin{array} { r } { { \frac { f ^ { \prime } ( p ) } { p - 1 } } = { \frac { g ^ { \prime } ( p ) } { p } } } \end{array}$ . A change of variable from p to c yields the result in the lemma.

Definition 6. For any reward scheme such that $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$ for all $c \in ( 0 , 1 )$ , thefunction $h ( c )$ is well-defined over (0, 1) as $\begin{array} { r } { h ( c ) = \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$ . In such a scenario, $f ( c )$ and $g ( c )$ can be expressed as $\begin{array} { r } { f ( c ) = \int ( c - 1 ) h ( c ) } \end{array}$ dc and $\begin{array} { r } { g ( c ) = \int c h ( c ) } \end{array}$ dc, up to arbitrary additive constants.

Lemma 2. Assuming h(c) is well-defined (as defined in Definition 6), the Interpretability property holds ifand only $i f \bar { h ( c ) } \overset { \cdot } { \leq } 0 f o r a l l c \in ( 0 , 1 )$

## Proof of Lemma 2.

Recall that the Interpretability property is that f is non-decreasing on (0, 1) and $g$ is non-increasing on (0, 1). Because $\bar { f }$ and g are differentiable on (0, 1):

f is non-decreasing on (0, 1)

$\iff f ^ { \prime } ( c ) = ( c - 1 ) h ( c ) \geq 0$ for all $c \in ( 0 , 1 )$

$\iff h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ since $c - 1 < 0$ for all $c \in ( 0 , 1 )$ 1

Similarly,

g is non-increasing on (0, 1)

$\iff g ^ { \prime } ( c ) = c h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$

$\iff h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ since $c > 0$ for all $c \in ( 0 , 1 )$

Thus, the Interpretability property holds if and only if $h ( c ) \leq 0 \mathrm { o v e r } ( 0 , 1 )$

## Lemma 3. Assuming h is well-defined, $\begin{array} { r } { \frac { \partial } { \partial c } R ( c , p ) = ( c - p ) h ( c ) f o r a l l c \epsilon \in ( 0 , 1 ) } \end{array}$

## Proof of Lemma 3.

By the definition of the expected reward R in the Proper Scoring property of Definition 2, $R ( c , p ) =$ $p f ( c ) + ( 1 - p ) g ( c )$ . Therefore, $\begin{array} { r } { \frac { \partial } { \partial c } R ( c , p ) = p f ^ { \prime } ( c ) + ( 1 - p ) g ^ { \prime } ( c ) } \end{array}$

From Definition 6, $f ^ { \prime } ( c ) = ( c - 1 ) h ( c )$ and $g ^ { \prime } ( c ) = c h ( c )$ for all $c \in ( 0 , 1 )$

Therefore, $\begin{array} { r } { \frac { \partial } { \partial c } R ( c , p ) = p ( c - 1 ) h ( c ) + ( 1 - p ) c h ( c ) = ( p c - p ) h ( c ) + ( c - p c ) h ( c ) = ( c - p ) h ( c ) } \end{array}$

Corollary 1. For any fixed p, if the reward scheme satisfies both the Interpretability and the Proper Scoring property, $R ( c , p )$ is non-decreasing on $c \in ( 0 , p )$ and non-increasing on $c \in ( p , 1 )$ .

## Proof of Corollary 1.

Since the reward scheme satisfies the Proper Scoring property, by Lemma 1, h is well-defined.

Since the reward scheme additionally satisfies the Interpretability property, by Lemma 2, h is non-positive.

By Lemma 3, $\begin{array} { r } { \frac { \partial } { \partial c } R ( c , p ) = ( c - p ) h ( c ) } \end{array}$ . For $c \in ( 0 , p ) , c - p < 0$ . For $c \in ( p , 1 ) , c - p > 0 .$

This implies that keeping p fixed, $R ( c , p )$ is non-decreasing on $c \in ( 0 , p )$ and non-increasing on $c \in ( p , 1 )$ .

## Corollary 2.

Now, we assume the LLM can only output confidences $c \in ( 0 , 1 )$ in some finite set $C _ { v a l i d }$ . All elements in $C _ { v a l i d }$ are assumed to be in $( 0 , 1 )$ . Let $c _ { m i n } = \operatorname* { m i n } { C _ { v a l i d } }$ and $c _ { m a x } = \operatorname* { m a x } C _ { v a l i d } .$

Take any $p \in ( 0 , 1 )$ . Let $C _ { \leq } = \{ c \in C _ { v a l i d } | c \leq p \}$ and $C _ { > } = \{ c \in C _ { v a l i d } | c > p \}$

Let $c _ { l e f t } = \operatorname* { m a x } C \leq i f C \leq$ is not empty, otherwise set to $c _ { m i n } .$

Let $c _ { r i g h t } = \operatorname* { m i n } C _ { > } i f C _ { > }$ is not empty, otherwise set to $c _ { m a x }$

Suppose the reward scheme satisfies both the Interpretability property and the Proper Scoring property, then an optimal value of $\dot { \boldsymbol { \cdot } } c \in C _ { v a l i d }$ that maximizes $R ( c , p )$ is either $c _ { l e f t } o r c _ { r i g h t } . $

## Proof of Corollary 2.

We consider three cases based on where $p \in ( 0 , 1 )$ lies relative to $c _ { m i n }$ and $c _ { m a x }$

Case 1: $p \in ( 0 , c _ { m i n } )$

In this case, $c _ { l e f t } = c _ { m i n }$ since $C _ { \leq }$ is empty and $c _ { r i g h t } = c _ { m i n }$

Since $c _ { m i n } \in ( p , 1 )$ , all values in $C _ { v a l i d }$ are in $( p , 1 )$ . Therefore, since $R ( c , p )$ is non-increasing on $c \in ( p , 1 )$ by $\mathrm { C o r o l l a r y } \ 1 , c _ { m i n } \in \mathrm { a r g m a x } _ { c \in C _ { v a l i d } } R ( c , p )$

Case 2: $p \in [ c _ { m i n } , c _ { m a x } )$

In this case, since $C _ { \leq }$ and $C _ { > }$ are both non-empty, $c _ { l e f t } = \operatorname* { m a x } C _ { \leq }$ and $c _ { r i g h t } = \operatorname* { m i n } C _ { > }$

If $\begin{array} { l } { { c _ { l e f t } } } \end{array} = \begin{array} { r } { p } \end{array}$ , then $c _ { l e f t } ~ \in ~ \mathrm { a r g m a x } _ { c \in ( 0 , 1 ) } R ( c , p )$ implies $c _ { l e f t } ~ \in ~ \mathrm { a r g m a x } _ { c \in C _ { v a l i d } } R ( c , p )$ since $C _ { v a l i d } \subseteq ( 0 , 1 )$

Otherwise, since all elements in $C _ { < }$ are in $( 0 , p )$ and $R ( c , p )$ is non-decreasing on $c \in ( 0 , p )$ by Corollary $1 , c _ { l e f t } = \operatorname* { m a x } C _ { \leq } \in \arg \operatorname* { m a x } _ { c \in C _ { < } } R ( c , p )$

Likewise, since all elements in $C _ { > }$ are in $( p , 1 )$ and $R ( c , p )$ is non-increasing on $c \in \mathsf { \Gamma } ( p , 1 )$ by Corollary $1 , c _ { r i g h t } = \operatorname* { m i n } C _ { > } \in \arg \operatorname* { m a x } _ { c \in C _ { > } } R ( c , p )$

Since $\begin{array} { r } { C _ { v a l i d } = C _ { \le } \cup C _ { > } , \mathrm { e i t h e r } \ c _ { l e f t } \in \arg \operatorname* { m a x } _ { c \in C _ { v a l i d } } R ( c , p ) \ \mathrm { o r } \ c _ { r i g h t } \in \arg \operatorname* { m a x } _ { c \in C _ { v a l i d } } R ( c , p ) . } \end{array}$ Case 3: $p \in [ c _ { m a x } , 1 )$

In this case, $c _ { r i g h t } = c _ { m a x }$ since $C _ { > }$ is empty and $c _ { l e f t } = c _ { m a x }$

If $p = c _ { m a x } ,$ then $c _ { l e f t } = c _ { m a x } \in \mathrm { a r g m a x } _ { c \in ( 0 , 1 ) } R ( c , p )$ implies $c _ { l e f t } \in \mathrm { \ a r g m a x } _ { c \in C _ { v a l i d } } R ( c , p )$ since $C _ { v a l i d } \subseteq ( 0 , 1 )$

Otherwise, $p \in \mathsf { \Gamma } ( c _ { m a x } , 1 )$ . Therefore, all values in $C _ { v a l i d }$ are in $( 0 , p )$ . Since $R ( c , p )$ is nondecreasing on $c \in ( 0 , p )$ by Corollary $1 , c _ { m a x } \in \operatorname { a r g m a x } _ { c \in C _ { v a l i d } } R ( c , p )$

Remark 3. For our experiments, $\begin{array} { r } { C _ { v a l i d } = \left\{ \frac { \tilde { c } + 0 . 5 } { 1 0 1 } | \tilde { c } \in \{ 0 , 1 , \cdots , 1 0 0 \} \right\} } \end{array}$

Remark 4. Intuitively, Corollary 2 implies that even in discretization ofconfidences,for non-hackable confidence reward schemes, the LLM is incentivized to output a confidence value that is the closest to (eitherjust lesser than orjust greater than) the probability ofanswering the question correctly.

Lemma 4. Assuming h is well-defined, $R _ { m a x } ^ { \prime } ( c ) = f ( c ) - g ( c ) f o r a l l c \in ( 0 , 1 )$

## Proof of Lemma 4.

By definition of $R _ { m a x }$ and R, $R _ { m a x } ( c ) = R ( c , c ) = c f ( c ) + ( 1 - c ) g ( c ) \ :$

$$
R _ {m a x} ^ {\prime} (c) = f (c) + c f ^ {\prime} (c) - g (c) + (1 - c) g ^ {\prime} (c)
$$

Since h is assumed to be well-defined, $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$ , which implies that $c f ^ { \prime } ( c ) + ( 1 - c ) g ^ { \prime } ( c ) = 0$ Therefore, $R _ { m a x } ^ { \prime } ( c ) = f ( c ) - g ( c )$ ■

For ease of reference, Theorem 1, which characterizes the non-hackable confidence reward schemes, is restated as follows:

Theorem 1. A reward scheme is non-hackable confidence over $( a , b ) \subseteq ( 0 , 1 )$ if and only if all of the following hold:

(i) There exists afunction h(c) defined over (0, 1) such that $\begin{array} { r } { h ( c ) = \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$

(ii) $h ( c ) \leq 0 .$ for all $c \in ( 0 , 1 )$

(iii) $f ( a ^ { + } ) \geq g ( a ^ { + } )$

## Proof of Theorem 1.

(⇒)

If a reward scheme is non-hackable confidence, then three properties must be proven to be true.

We are given that $( f , g )$ is a non-reward hackable scheme. Since $( f , g )$ satisfies the proper scoring property over (0, 1), by Lemma 1, h(c) exists and is well-defined, satisfying property (i).

By the Interpretability property, since we assume f to be differentiable and non-decreasing on (0, 1), since $\begin{array} { r } { c \in ( 0 , 1 ) , f ^ { \prime } ( c ) \geq 0 \Rightarrow h ( c ) = \frac { f ^ { \prime } ( c ) } { c - 1 } \leq 0 } \end{array}$ , which fulfills property (ii).

By Lemma 4, $R _ { m a x } ^ { \prime } ( c ) = f ( c ) - g ( c )$ for all $c \in ( 0 , 1 )$ . By Best Effort property, since $R _ { m a x } ^ { \prime }$ is nondecreasing on $( a , b ) , R _ { m a x } ^ { \prime } ( c ) \geq 0$ for all $c \in ( a , b )$ . Therefore, since ${ \overline { { ( a , b ) \subseteq ( 0 , 1 ) , f ( c ) } } } \geq g ( c )$ for all $c \in ( a , b )$

Take any $x , y \in ( a , b )$ . By Interpretability property, f is non-decreasing and g is non-increasing.

Case 1: $x \leq y . \ f ( x ) \geq g ( x ) \geq g ( y )$

Case 2: $x > y . \ f ( x ) \geq f ( y ) \geq g ( y )$

Therefore, $f ( x ) \geq g ( y )$ for all $x , y \in ( a , b )$

This implies in $\operatorname { i f } _ { x \in ( a , b ) } f ( x ) \geq g ( y )$ for all $y \in ( a , b )$

This implies in $\begin{array} { r } { \mathrm { f } _ { x \in ( a , b ) } f ( x ) \geq \operatorname* { s u p } _ { y \in ( a , b ) } g ( y ) } \end{array}$

By Interpretability property, since f is non-decreasing on $( a , b ) , \operatorname* { i n f } _ { x \in ( a , b ) } f ( x ) = f ( a ^ { + } )$

By Interpretability property, since g is non-increasing on $( a , b ) , \operatorname* { s u p } _ { y \in ( a , b ) } g ( y ) = g ( a ^ { + } )$

Therefore, $f ( a ^ { + } ) \geq g ( a ^ { + } )$ , satisfying property (iii).

(⇐)

If the following three properties are satisfied, then we need to prove that the scheme is non-confidence hackable.

(i) There exists a function h(c) defined over (0, 1) such that $\begin{array} { r } { h ( c ) = \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$

(ii) $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$

(iii) $f ( a ^ { + } ) \geq g ( a ^ { + } )$

From property $( \mathrm { i } ) , h$ is well-defined. Therefore, by Lemma 2 and property (ii), the Interpretability property holds.

Next, we aim to show the Proper Scoring property. Fix any $p \in ( 0 , 1 )$ and let $r ( c ) = R ( c , p )$ , where r is defined over $c \in ( 0 , 1 )$ . By Lemma $\begin{array} { r } { 3 , r ^ { \prime } ( c ) = \frac { \partial R } { \partial c } = ( c - p ) h ( c ) } \end{array}$ for all $c \mathrm { i n } \left( 0 , 1 \right)$

Therefore, from property (ii), $r ^ { \prime } ( c ) \geq 0 \Leftrightarrow c \leq p$ and $r ^ { \prime } ( c ) \leq 0 \Leftrightarrow c \geq p$

We aim to show that $r ( c ) \leq r ( p )$ for all $c \in ( 0 , 1 )$

Case 1: $c \in ( 0 , p ]$ Since $r ^ { \prime } ( x ) \geq 0$ for all $\begin{array} { r } { x \in ( 0 , p ] , r ( p ) - r ( c ) = \int _ { c } ^ { p } r ^ { \prime } ( x ) \mathrm { d } x \geq 0 } \end{array}$ . This implies $r ( p ) \geq r ( c )$

Case $2 \colon c \in ( p , 1 )$ Since $r ^ { \prime } ( x ) \leq 0$ for all $\begin{array} { r } { x \in ( p , 1 ) , r ( c ) - r ( p ) = \int _ { p } ^ { c } r ^ { \prime } ( x ) \mathrm { d } x \leq 0 } \end{array}$ . This implies $r ( c ) \leq r ( p )$

In both cases, $r ( p )$ is an upper bound of $r ( c )$ . Therefore, the maximum value of R when p is fixed is achieved when c is set to p. Hence, the Proper Scoring property is satisfied.

Last, we aim to show the Best Effort property. By Lemma 4, since h is well-defined, $R _ { m a x } ^ { \prime } ( c ) =$ $f ( c ) - g ( c )$ for all $c \in ( 0 , 1 )$ .

By the Interpretability property, for all $c \in ( a , b ) , f ( c ) \geq \operatorname* { i n f } _ { c ^ { \prime } \in ( a , b ) } f ( c ^ { \prime } ) = f ( a ^ { + } )$ . Similarly, $g ( a ^ { + } ) = \operatorname* { s u p } _ { c ^ { \prime } \in ( a , b ) } g ( c ^ { \prime } ) \geq g ( c )$ . Since property (iii) states that $f ( a ^ { + } ) \geq g ( a ^ { + } )$ , we have $f ( c ) -$ $g ( c ) = R _ { m a x } ^ { \prime } ( c ) \geq 0$ . Therefore, $R _ { m a x }$ is non-decreasing on $( a , b )$ , which is the Best Effort property.

Corollary 3. If a reward scheme $( f ( c ) , g ( c ) )$ is a non-hackable confidence reward scheme over $( a , b ) \subseteq ( 0 , 1 )$ , then it must be non-hackable confidence over $( a , 1 )$ .

## Proof of Corollary 3.

The corollary follows from the fact that properties (i), (ii) and (iii) in Theorem 1 do not rely on the endpoint b. Here is an alternative proof which does not rely on Theorem 1.

Since $( f ( c ) , g ( c ) )$ is a non-hackable confidence reward scheme, it satisfies proper scoring, hence the corresponding function h is well-defined and Lemma 4 applies. By the Interpretability property, both $f ( c )$ and $- g ( c )$ are non-decreasing on $( 0 , 1 )$ . Therefore, using Lemma $4 , \bar { R } _ { m a x } ^ { \prime } \stackrel { . } { = } \bar { f } ( c ) - \stackrel { . } { g } ( c )$ is non-decreasing on $( 0 , 1 )$ . By the Best Effort property, $R _ { m a x } ( \bar { c } )$ is non-decreasing on $( a , b )$ , which implies $R _ { m a x } ^ { \prime } ( c ) \geq 0$ for all $c \in ( a , b )$

Since $R _ { m a x } ^ { \prime }$ is non-decreasing on $( 0 , 1 )$ and is non-negative on $( a , b ) , R _ { m a x } ^ { \prime }$ must be non-negative on $( a , 1 )$ . Therefore, $R _ { m a x } ( c )$ must be non-decreasing on $( a , 1 )$ , satisfying the Best Effort property. The Interpretability and Proper Scoring properties are automatically satisfied as the properties remain the same when the interval is changed from $( a , b ) \mathrm { t o } ( a , 1 )$

Therefore, $( f ( c ) , g ( c ) )$ is a non-hackable confidence reward scheme on $( a , 1 )$

We then proceed to derive the general form of non-hackable confidence reward schemes. But we have to first show that certain improper integrals are convergent to establish well-definedness of the general form.

Lemma 5. Let $f _ { 1 } , f _ { 2 }$ be continuous on $( a , b ]$ , where $a , b \in \mathbb { R }$ and $0 \leq f _ { 1 } ( x ) \leq f _ { 2 } ( x )$ for all $\begin{array} { r } { x \in ( a , b ] . \ I f \int _ { a } ^ { b } f _ { 2 } ( x ) } \end{array}$ dx converges, then $\textstyle \int _ { a } ^ { b } f _ { 1 } ( x )$ dx converges.

Lemma 6. Let $f _ { 1 } , f _ { 2 }$ be continuous on $( a , b ] ,$ , where $a , b \in \mathbb { R }$ and $f _ { 1 } ( x ) , f _ { 2 } ( x ) \geq 0$ for all $x \in ( a , b ]$ $\begin{array} { r } { I f \operatorname* { l i m } _ { x  a ^ { + } } \frac { f _ { 1 } ( x ) } { f _ { 2 } ( x ) } } \end{array}$ exists and is positive and finite, then $\textstyle \int _ { a } ^ { b } f _ { 1 } ( x )$ dx and $\textstyle \int _ { a } ^ { b } f _ { 2 } ( x )$ dx either both converge or both diverge.

The proofs of Lemmas 5 and 6 are analogous to the proofs of Theorem 15 and its corollary respectively in [29, pp. 214-215].

Lemma 7. Let $f _ { 1 }$ be a continuousfunction on (0, 1). $H \int _ { 0 } ^ { c } f _ { 1 } ( x )$ dx converges for some $c \in ( 0 , 1 )$ then $\begin{array} { r } { \int _ { 0 } ^ { p } f _ { 1 } ( x ) } \end{array}$ dx converges for all $p \in ( 0 , 1 )$ .

## Proof of Lemma 7.

We are given that $\textstyle \int _ { 0 } ^ { c } f _ { 1 } ( x )$ dx converges for some $c \in ( 0 , 1 )$ . Take any $p \in ( 0 , 1 )$

Case 1: $p = c$

Trivially proven as $\textstyle \int _ { 0 } ^ { c } f _ { 1 } ( x )$ dx is known to converge.

Case 2: $p < c$

Since $\textstyle \int _ { 0 } ^ { c } f _ { 1 } ( x )$ dx converges and $f _ { 1 }$ is continuous on $( 0 , c ]$ , both $\textstyle \int _ { p } ^ { c } f _ { 1 } ( x )$ dx and $\begin{array} { r } { \int _ { 0 } ^ { p } f _ { 1 } ( x ) } \end{array}$ dx must converge.

Case $3 \colon p > c$

Since $f _ { 1 }$ is continuous on $[ c , p ] , \int _ { c } ^ { p } f _ { 1 } ( x )$ dx is a proper integral and therefore converges.

Since $\textstyle \int _ { 0 } ^ { c } f _ { 1 } ( x )$ dx converges by hypothesis, $\begin{array} { r } { \int _ { 0 } ^ { p } f _ { 1 } ( x ) } \end{array}$ dx converges and is equal to $\textstyle \int _ { c } ^ { p } f _ { 1 } ( x )$ dx + $\textstyle \int _ { 0 } ^ { c } f _ { 1 } ( x ) d x$

Therefore, combining the three cases, $\textstyle \int _ { 0 } ^ { p } f ( x )$ dx converges for all $p \in ( 0 , 1 )$

Corollary 4. Let $h ( c )$ be any continuous function on $( 0 , 1 )$ such that $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ Then, $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc convergesfor all $p \in ( 0 , 1 ) \Leftrightarrow \int _ { 0 } ^ { p } ( c - 1 ) h ( c )$ dc converges for all $p \in ( 0 , 1 )$

## Proof of Corollary 4.

Let $f _ { 1 }$ and $f _ { 2 }$ be defined on $( 0 , 1 )$ such that $f _ { 1 } ( c ) = ( c - 1 ) h ( c )$ and $f _ { 2 } ( c ) = - h ( c )$

Since h is continuous on $( 0 , 1 ) , f _ { 1 }$ and $f _ { 2 }$ are continuous on $( 0 , 1 )$ .

Since $h ( c ) \leq 0$ and $c - 1 < 0$ for all c ∈ (0, 1), f<sub>1</sub>(c) ≥ 0 and $f _ { 2 } ( c ) \geq 0$ for all $c \in ( 0 , 1 )$

$\begin{array} { r } { \operatorname* { l i m } _ { c  0 ^ { + } } \frac { f _ { 1 } ( c ) } { f _ { 2 } ( c ) } = \operatorname* { l i m } _ { c  0 ^ { + } } ( 1 - c ) = 1 } \end{array}$ is positive and finite.

Therefore, by Lemma $6 , \ \int _ { 0 } ^ { p } \ f _ { 1 } ( c )$ dc and $\textstyle \int _ { 0 } ^ { p } f _ { 2 } ( c )$ dc either both converge or both diverge for all $p \in \mathsf { \Gamma } ( 0 , 1 )$ . Since $h ( c ) = - f _ { 2 } ( c )$ and $( c - 1 ) h ( c ) = f _ { 1 } ( c )$ , it follows that $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc and $\textstyle \int _ { 0 } ^ { p } ( c - 1 ) h ( c )$ dc either both converge or both diverge for all $p \in ( 0 , 1 )$

Note that since we did not rule out the possibility where $\textstyle \int _ { 0 } ^ { p _ { 1 } } h ( c )$ dc converges but $\int _ { 0 } ^ { p _ { 2 } } h ( c )$ dc diverges for some $p _ { 1 } , p _ { 2 } \in ( 0 , 1 )$ , we are not done yet.

Take any $p \in ( 0 , 1 )$ and consider whether $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc diverges.

Case 1: $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc diverges

By Lemma 7, $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc must diverge for all $p \in ( 0 , 1 )$

Hence, $\textstyle \int _ { 0 } ^ { p } ( c - 1 ) h ( c )$ dc must diverge for all $p \in ( 0 , 1 )$

Case 2: $\begin{array} { r } { \int _ { 0 } ^ { p } f _ { 1 } ( x ) } \end{array}$ dx converges

By Lemma $7 , \int _ { 0 } ^ { p } h ( c )$ dc must converge for all $p \in ( 0 , 1 )$

Hence, $\textstyle \int _ { 0 } ^ { p } ( c - 1 ) h ( c )$ dc must converge for all $p \in ( 0 , 1 )$

Therefore, combining the two cases, $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc converges for all $p \in ( 0 , 1 ) \Leftrightarrow \int _ { 0 } ^ { p } ( c - 1 ) h ( c )$ dc converges for all $p \in ( 0 , 1 )$

Hence, $\textstyle \int _ { 0 } ^ { p } h ( c )$ dc either both converge or both diverge.

Corollary 5. Let $h ( c )$ be any continuous function on $( 0 , 1 )$ such that $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ Then, $\textstyle \int _ { 0 } ^ { c } h ( x )$ dx convergesfor all $c \in ( 0 , 1 ) \implies \int _ { 0 } ^ { c } x h ( x )$ dx convergesfor all $c \in ( 0 , 1 )$

## Proof of Corollary 5.

Suppose $\textstyle \int _ { 0 } ^ { c } h ( x )$ dx converges for all $c \in ( 0 , 1 )$

Since $h ( c )$ is continuous on (0, 1), ch(c) is continuous on $( 0 , 1 )$

Since $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ and $\textstyle \int _ { 0 } ^ { c } h ( x )$ dx converges for all $\begin{array} { r } { c \in \ ( 0 , 1 ) , \ \int _ { 0 } ^ { c } \left| h ( x ) \right| d x \ = } \end{array}$ $\textstyle \int _ { 0 } ^ { c } - h ( x )$ dx converges for all $c \in ( 0 , 1 )$ .

Since $0 \leq | c h ( c ) | \leq | h ( c ) |$ | and $\int _ { 0 } ^ { c } { \lvert h ( x ) }$ | dx converges for all $c \in ( 0 , 1 )$ , by Lemma 5 $, \int _ { 0 } ^ { c } | x h ( x ) |$ dx converges for all $c \in ( 0 , 1 )$ .

For all $c \in ( 0 , 1 )$ , since $c > 0$ and $h ( c ) \leq 0 , c h ( c ) \leq 0 .$ . Therefore, $\begin{array} { r } { \int _ { 0 } ^ { c } x h ( x ) d x = \int _ { 0 } ^ { c } - \vert x h ( x ) \vert d x } \end{array}$ converges for all $c \in ( 0 , 1 )$ .

Theorem 2. The generalform ofnon-hackable confidence reward schemes is $\begin{array} { r l } { ( \int _ { 0 } ^ { c } ( x - 1 ) h ( x ) d x + } \end{array}$ $\textstyle f _ { 0 } , \int _ { 0 } ^ { c } x h ( x ) d x + g _ { 0 } )$ , where thefollowing conditions hold:

(i) $h ( c ) \leq 0 f o r a l l c \in ( 0 , 1 )$

(ii) h is continuous on (0, 1)

(iii) $\textstyle \int _ { 0 } ^ { c } h ( x )$ dx converges for all $c \in ( 0 , 1 )$

(iv) $f _ { 0 } , g _ { 0 }$ are constants in R satisfying $f _ { 0 } \geq g _ { 0 }$

Remark 5. A related theorem that describes the generalform ofreward schemes (not necessarily non-hackable confidence) that satisfy Proper Scoring property can befound in Theorem 4.2 of[28].

## Proof of Theorem 2.

(⇒)

We aim to show that if a reward scheme $( f ( c ) , g ( c ) )$ is non-hackable confidence, then it must be of the given form.

Since $( f ( c ) , g ( c ) )$ is non-hackable confidence, by Theorem 1, h(c) is well-defined and $h ( c ) =$ $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( \dot { c } ) } { c } } \end{array}$ , which implies that $f ^ { \prime } ( c ) = ( c - 1 ) h ( c )$ and $g ^ { \prime } ( c ) = c h ( c )$ . Moreover, $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ , satisfying (i).

By Interpretability property, since g is non-increasing on (0, 1), it is impossible for $g ( 0 ^ { + } )$ to diverge $\mathrm { t o } \mathrm { - } \infty$ . Since ${ \bar { f ( 0 ^ { + } ) } } \stackrel { \cdot } { = } \stackrel { \cdot } { g ( 0 ^ { + } ) }$ , it is impossible for $f ( 0 ^ { + } )$ to diverge $\mathrm { \bf t o } - \infty ,$ . Likewise, since f is non-decreasing on $( 0 , 1 )$ , it is impossible for $f ( 0 ^ { + } )$ to diverge to $\infty .$ . Since $f ( 0 ^ { + } ) \geq g ( 0 ^ { + } )$ it is impossible for $g ( 0 ^ { + } )$ to diverge to $\infty$ . Therefore, the antiderivatives of $f ^ { \prime }$ and $g ^ { \prime }$ must have well-defined real number limits at $0 ^ { + }$

Since f and g are assumed to have continuous first order derivatives, $h ( c )$ is continuous on $( 0 , 1 )$ • satisfying (ii).

Therefore, $f ^ { \prime } ( c ) = ( c - 1 ) h ( c )$ and $g ^ { \prime } ( c ) = c h ( c )$ are continuous on $( 0 , 1 )$ . This implies that $\textstyle \int _ { 0 } ^ { c } ( x - 1 ) h ( x )$ dx and $\textstyle \int _ { 0 } ^ { c } x h ( x )$ dx are well-defined and must converge for all $c \in ( 0 , 1 )$ . Hence, $\begin{array} { r } { f ( c ) = \int _ { 0 } ^ { c } ( x - 1 ) h ( x ) \mathop { d x } + f _ { 0 } } \end{array}$ and $\begin{array} { r } { g ( c ) = \int _ { 0 } ^ { c } x h ( x ) \ : d x + g _ { 0 } } \end{array}$ for some real constants $f _ { 0 }$ and $g _ { 0 }$ , satisfying the general form.

By Corollary 4, since $\textstyle \int _ { 0 } ^ { c } ( x - 1 ) h ( x )$ dx converges for all $c \in ( 0 , 1 ) , \int _ { 0 } ^ { c } h ( x )$ dx must converge for all $c \in ( 0 , 1 )$ , satisfying (iii).

Since $f ( 0 ^ { + } ) \geq g ( 0 ^ { + } )$ by Theorem 1, lim $\begin{array} { r } { \mathfrak { i } _ { c  0 ^ { + } } ( \int _ { 0 } ^ { c } ( x - 1 ) h ( x ) d x + f _ { 0 } ) \geq \operatorname* { l i m } _ { c  0 ^ { + } } ( \int _ { 0 } ^ { c } x h ( x ) } \end{array}$ dx + $g _ { 0 } )$ , which implies that $f _ { 0 } \geq g _ { 0 }$ since both integrals converge, satisfying (iv).

We want to show that if $\begin{array} { r } { ( f ( c ) , g ( c ) ) = ( \int _ { 0 } ^ { c } ( x - 1 ) h ( x ) d x + f _ { 0 } , \int _ { 0 } ^ { c } x h ( x ) d x + g _ { 0 } ) } \end{array}$ such that the given constraints on $h , f _ { 0 }$ and g are satisfied, then $( f ( c ) , g ( c ) )$ is non-hackable confidence.

Since $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ from (i), h is continuous on (0, 1) from (ii) and $\textstyle \int _ { 0 } ^ { c } h ( x )$ dx converges for all $c \in ( 0 , 1 )$ from (iii), it follows that $\textstyle \int _ { 0 } ^ { c } ( x - 1 ) h ( x )$ dx converges for all $c \in ( 0 , 1 )$ by Corollary 4 and $\textstyle \int _ { 0 } ^ { c } x h ( x )$ dx converges for all $c \in ( 0 , 1 )$ by Corollary 5.

Therefore, both f and g are well-defined functions.

Since h is continuous on (0, 1) from (ii), $f ^ { \prime } ( c ) = ( c - 1 ) h ( c )$ and $g ^ { \prime } ( c ) = c h ( c )$ are both continuous on (0, 1), hence both f and g have continuous first derivatives on (0, 1).

$$
f (0 ^ {+}) = \lim _ {c \to 0} (\int_ {0} ^ {c} (x - 1) h (x) d x + f _ {0}) = f _ {0}
$$

$$
g (0 ^ {+}) = \lim _ {c \to 0} (\int_ {0} ^ {c} x h (x) d x + g _ {0}) = g _ {0}
$$

Since $f _ { 0 } \geq g _ { 0 }$ from (iv), we have $f ( 0 ^ { + } ) \geq g ( 0 ^ { + } )$ . In addition, we also have $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ , therefore, $( f ( c ) , g ( c ) )$ is non-hackable confidence on (0, 1).

Definition 7. A reward scheme $( f ( c ) , g ( c ) )$ is strictly non-hackable confidence ifall ofthefollowing are satisfied:

(i) Strict Interpretability: $f ( c )$ is strictly increasing in (0, 1) and $g ( c )$ is strictly decreasing on $( 0 , 1 )$

(ii) Strict Proper Scoring: $( f ( c ) , g ( c ) )$ is a strictly proper scoring rule, i.e. $a r g m a x _ { c } R ( c , p ) =$ $\{ p \} \forall p \in ( 0 , \bar { 1 } )$

(iii) Strict Best Effort: $R _ { m a x }$ is strictly increasing on (0, 1).

Remark 6. Strict Interpretability ensures that correct answers with higher confidence are always given a higher reward and incorrect answers with higher confidence are always penalized more heavily.

Strict Proper Scoring provides a stronger theoretical guarantee that the only optimal solution is for the LLM to be honest about its epistemic uncertainty, assuming the answer to each question is deterministic and the evaluator is deterministic. Without the strict condition, the LLM can output an inaccurate confidence estimate and still obtain the same reward.

Strict Best Effort ensures that the LLM is incentivized to always answer to the best ofits ability as an increase the probability for answering correctly is always rewarded with an increase with the maximum possible expected rewardfor accurate confidence calibration.

Remark 7. In general, strictly non-hackable confidence reward schemes are non-hackable confidence. But the converse is not true. Correctness-only, i.e. the reward scheme (1, 0), is an example of a non-hackable confidence reward scheme that is not strictly non-hackable confidence. Theorem 3 describes a condition under which the converse is true.

Theorem 3. Ifa non-hackable confidence reward scheme $( f ( c ) , g ( c ) )$ further satisfies $h ( c ) < 0 f o r$ $a l l c \in ( 0 , 1 )$ , then $( f ( c ) , g ( c ) )$ is strictly non-hackable confidence.

## Proof of Theorem 3.

Let p be any real number in (0, 1) and $r ( c ) = R ( c , p )$ , where r is defined over $c \in ( 0 , 1 )$ . By Lemma $\textstyle 3 , r ^ { \prime } ( c ) = { \frac { \partial R } { \partial c } } = ( c - p ) h ( c )$ for all c in (0, 1). Since $h ( c ) < 0$ for all $c \in ( 0 , 1 ) , c = p$ is the only critical point in $( 0 , 1 )$ . Since $r ^ { \prime }$ changes from positive to negative at $c = p , c = p$ is a local maximum by first derivative test. This implies that $c = p$ is the global maximum of r on (0, 1). Therefore, $\operatorname { a r g m a x } _ { c } R ( c , p ) = \{ p \}$ , satisfying the Strict Proper Scoring property.

From Definition 6, $f ^ { \prime } ( c ) = ( c - 1 ) h ( c )$ and $g ^ { \prime } ( c ) = c h ( c )$ . Since $c - 1 < 0$ and $c > 0$ for all $c \in ( 0 , 1 ) , f ^ { \prime } ( c ) > 0$ and $g ^ { \prime } ( c ) < 0$ for all $c \in ( 0 , 1 )$ ). Therefore, $f$ is strictly increasing and $g$ is strictly decreasing on $( 0 , 1 )$ , satisfying the Strict Interpretability property.

In addition, by Lemma 4, since h is well-defined by definition of non-hackable confidence reward schemes (Definition 2), $R _ { m a x } ^ { \prime } ( c ) = f ( c ) - g ( c )$ for all $c \in ( 0 , 1 )$ . Therefore, $R _ { m a x } ^ { \prime }$ is strictly increasing on (0, 1).

Since Best Effort property is satisfied in non-hackable confidence reward schemes, $R _ { m a x } ^ { \prime } ( c ) \geq 0$ for all $c \in ( 0 , 1 )$ . Suppose $\dot { R _ { m a x } ^ { \prime } } ( \epsilon ) = 0$ for some $\epsilon \in ( 0 , 1 )$ . Then $\frac \epsilon 2 \in ( 0 , 1 )$ and $\begin{array} { r } { \frac { \epsilon } { 2 } < \epsilon , } \end{array}$ , which implies that $R _ { m a x } ^ { \prime } \bigl ( \frac { \epsilon } { 2 } \bigr ) < 0$ since $R _ { m a x } ^ { \prime }$ is strictly increasing. This contradicts the Best Effort Property. Hence, $R _ { m a x } ^ { \prime } ( c ) > 0$ for all $c \in ( 0 , 1 )$ . This implies that $R _ { m a x }$ is strictly increasing on $( 0 , 1 )$ , satisfying the Strict Best Effort property.

## A.2 Alternative Formulation of Non-hackable Confidence Reward Schemes

As mentioned in Section 2, [17] and [18] proposed reward functions based on the behavioral calibration framework of [19].

In this subsection, we show that the abstention-based risk-thresholding framework leads to a nonconfidence hackable reward scheme and can be used to construct all non-hackable confidence reward schemes such that $f ( 0 ^ { + } ) = g ( 0 ^ { + } ) = 0$

In the behavioral calibration framework of [19, p. 13], the LLM is awarded 1 point for a correct answer, 0 for an abstention, and $- { \frac { t } { 1 - t } }$ points for an incorrect answer. This is formally described in Definition 8, and termed “abstention reward function”.

Definition 8. Let isAbstention $( q , a )$ be 1 if the LLM answer a abstains from answering question q and 0 otherwise. Let isCorrect(q, a) be equal to 1 ifanswer a is a correct answer to question q and 0 otherwise. Then,for any $t \in [ 0 , 1 )$ , the abstention rewardfunction with confidence threshold t is defined asfollows:

$$
R _ {a b s} (q, a) = \left\{ \begin{array}{l l} 1, & \text {   if   } i s C o r r e c t (q, a) = 1 \land i s A b s t e n t i o n (q, a) = 0 \\ 0, & \text {   if   } i s A b s t e n t i o n (q, a) = 1 \\ - \frac {t}{1 - t}, & \text {   if   } i s C o r r e c t (q, a) = 0 \land i s A b s t e n t i o n (q, a) = 0 \end{array} \right.
$$

Theorem 4. Assuming that the confidence output c ofthe LLM is well-calibrated, under the abstention reward scheme with confidence threshold t, it is optimal on expectation for the LLM to answer is $c \geq t$ and to abstain $i f c < t$

## Proof of Theorem 4.

Note that E[LLM answers] $= c - { \frac { ( 1 - c ) t } { 1 - t } }$ and E[LLM abstains] = 0 since we assume that the confidence outputs are well-calibrated.

Since $1 - t \in ( 0 , 1 ]$

E[LLM answers] ≥ E[LLM abstains]

$$
\Leftrightarrow c - \frac {(1 - c) t}{1 - t} \geq 0
$$

$$
\Leftrightarrow c \geq \frac {(1 - c) t}{1 - t}
$$

$$
\Leftrightarrow c \geq \frac {t}{1 - t} - \frac {c t}{1 - t}
$$

$$
\Leftrightarrow c + \frac {c t}{1 - t} \geq \frac {t}{1 - t}
$$

$$
\Leftrightarrow c (1 + \frac {t}{1 - t}) \geq \frac {t}{1 - t}
$$

$$
\Leftrightarrow c \bigl (\frac {(1 - t) + t}{1 - t} \bigr) \geq \frac {t}{1 - t}
$$

$$
\Leftrightarrow c \bigl (\frac {1}{1 - t} \bigr) \geq \frac {t}{1 - t}
$$

$$
\Leftrightarrow c \geq t
$$

Therefore, it is optimal for the LLM to answer the question if $c \geq t$ and abstain otherwise.

We define a function $w ( t )$ of confidence thresholds, where w is defined, non-negative and continuous on (0, 1) and $\textstyle \int _ { 0 } ^ { c } w ( t )$ dt converges for all $c \in ( 0 , 1 )$ . Note that this generalizes from [18] by no longer restricting $w ( t )$ to be a probability distribution over [0, 1).

Following [18], we aim to maximize the expected reward of the LLM under the abstention reward function with confidence threshold t weighted by $w ( t )$ . Under the optimal strategy as described in Theorem 4, we derive the reward scheme $\bar { \ b { ( f ( c ) , g ( c ) ) } }$ as follows:

$$
f (c) = \int_ {c} ^ {1} 0 d t + \int_ {0} ^ {c} w (t) d t = \int_ {0} ^ {c} w (t) d t
$$

$$
g (c) = \int_ {c} ^ {1} 0 d t + \int_ {0} ^ {c} w (t) \frac {- t}{1 - t} d t = \int_ {0} ^ {c} w (t) \frac {t}{t - 1} d t
$$

Definition 9 formally summarizes this notion as “weighted abstention confidence threshold reward schemes”.

Definition 9. $( f ( c ) , g ( c ) )$ is a weighted abstention confidence threshold reward scheme ifthere exists w(c) defined on (0, 1) such that all of the following hold:

(i) $w ( c ) \geq 0$ for all $c \in ( 0 , 1 )$

(ii) w is continuous on (0, 1)

(iii) $\textstyle \int _ { 0 } ^ { c } w ( t )$ dt convergesfor all $c \in ( 0 , 1 )$

(iv) $\begin{array} { r } { f ( c ) = \int _ { 0 } ^ { c } w ( t ) } \end{array}$ dt

(v) $\begin{array} { r } { g ( c ) = \int _ { 0 } ^ { c } w ( t ) \frac { t } { t - 1 } d } \end{array}$ t

Remark 8. [16] derived a generalform ofreward schemes $( f ( c ) , g ( c ) )$ that satisfy Interpretability and Proper Scoring, but not necessarily Best Effort. Up to notational differences, the expressions off(c) and $g ( c )$ are almost identical to weighted abstention confidence threshold reward schemes except for the integration endpoints for $g ( c )$ . In this work, we aim to characterize non-hackable confidence reward schemes in terms ofweighted abstention confidence threshold reward schemes, which satisfies Best Effort in addition to Interpretability and Proper Scoring

Theorem 5. The weighted abstention confidence threshold reward scheme $( f ( c ) , g ( c ) ) =$ $\begin{array} { r } { ( \int _ { 0 } ^ { c } w ( t ) ~ d t , \int _ { 0 } ^ { c } w ( t ) \frac { t } { t - 1 } ~ \ d t ) } \end{array}$ is a non-hackable confidence reward scheme for any non-negative and continuous $w ( t )$ over (0, 1) such that $\textstyle \int _ { 0 } ^ { c } w ( t )$ dt convergesfor all $c \in ( 0 , 1 )$

## Proof of Theorem 5.

Since $f ^ { \prime } ( c ) = w ( c )$ and $\begin{array} { r } { g ^ { \prime } ( c ) = w ( c ) \frac { c } { c - 1 } } \end{array}$ from (iv) and (v) respectively and w is continuous on (0, 1) from (ii), f and $g$ have continuous first derivatives on $( 0 , 1 )$ . Furthermore, $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { w ( c ) } { c - 1 } } \end{array}$ and $\begin{array} { r } { \frac { g ^ { \prime } ( c ) } { c } = \frac { w ( c ) } { c - 1 } } \end{array}$ . Hence, h is well defined and $\begin{array} { r } { h ( c ) = \frac { w ( c ) } { c - 1 } } \end{array}$

Since $w ( c ) \geq 0$ for all $c \in ( 0 , 1 )$ from (i) and $c - 1 < 0$ for all $c \in ( 0 , 1 ) , h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ Since w(c) is continuous on (0, 1) from (ii), it follows that $h ( c )$ must also be continuous on (0, 1).

By Corollary 4, since $w ( c ) = ( c - 1 ) h ( c )$ and $\textstyle \int _ { 0 } ^ { c } w ( t )$ dt converges for all $c \in ( 0 , 1 )$ from (iii), $\textstyle \int _ { 0 } ^ { c } h ( t )$ dt converges for all $c \in ( 0 , 1 )$ . Therefore, by Corollary $5 , \int _ { 0 } ^ { c } t h ( t )$ dt converges for all $c \in ( 0 , 1 )$

Since $w ( c ) = ( c - 1 ) h ( c )$ ) and $\begin{array} { r } { w ( c ) \frac { c } { c - 1 } = c h ( c ) } \end{array}$ for all $\begin{array} { r } { c \in ( 0 , 1 ) , ( \int _ { 0 } ^ { c } w ( t ) d t , \int _ { 0 } ^ { c } w ( t ) \frac { t } { t - 1 } d t ) = } \end{array}$ $\begin{array} { r l } {  { ( \int _ { 0 } ^ { c } ( t - 1 ) h ( t ) d t , \int _ { 0 } ^ { c } t h ( t ) d t ) } \quad } & { { } } \end{array}$ is a non-hackable confidence reward scheme on $( 0 , 1 )$ by Theorem 2.

As the proof of Theorem 5 suggests, the formulation of the weighted abstention confidence threshold reward scheme has $f _ { 0 } = g _ { 0 } = 0$ , hence does not take into account possible additive constant rewards for answering correctly and answering incorrectly. Therefore, we generalize the notion of weighted abstention confidence threshold reward schemes in Definition 10.

Definition 10. $( f ( c ) , g ( c ) )$ is a generalized weighted abstention confidence threshold reward scheme ifthere exists w(t) defined on (0, 1) and $f _ { 0 } , g _ { 0 } \in \mathbb { R }$ such that all ofthefollowing hold:

(i) $w ( c ) \geq 0 .$ for all $c \in ( 0 , 1 )$

(ii) w is continuous on (0, 1)

(iii) $\textstyle \int _ { 0 } ^ { c } w ( t )$ dt converges for all $c \in ( 0 , 1 )$

(iv) $\begin{array} { r } { f ( c ) = \int _ { 0 } ^ { c } w ( t ) d t + f _ { 0 } } \end{array}$

(v) $\begin{array} { r } { g ( c ) = \int _ { 0 } ^ { c } w ( t ) \frac { t } { t - 1 } d t + g _ { 0 } } \end{array}$

(vi) $f _ { 0 } \geq g _ { 0 }$

Theorem 6. The set ofnon-hackable confidence reward schemes is the set ofgeneralized weighted abstention confidence threshold reward schemes.

## Proof of Theorem 6.

(⇒)

We aim to show non-hackable confidence reward schemes are generalized weighted abstention confidence threshold reward schemes. Let $( f ( c ) , g ( c ) )$ ) be a non-hackable confidence reward scheme.

By Theorem $\begin{array} { r } { 2 , f ( c ) = \int _ { 0 } ^ { c } ( t - 1 ) h ( t ) d t + f _ { 0 } \mathrm { a n d } g ( c ) = \int _ { 0 } ^ { c } t h ( t ) d t + g _ { 0 } } \end{array}$ , where all of the following hold:

(a) $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$

(b) h is continuous on (0, 1)

(c) $\textstyle \int _ { 0 } ^ { c } h ( t )$ dt converges for all $c \in ( 0 , 1 )$

(d) $f _ { 0 } , g _ { 0 } \in \mathbb { R }$ such that $f _ { 0 } \geq g _ { 0 }$

We aim to show that $( f ( c ) , g ( c ) )$ is a generalized weighted abstention confidence threshold reward scheme.

Let $w ( c ) = ( c - 1 ) h ( c )$ for all $c \in ( 0 , 1 )$ . Since $h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ ) from (a) and $c - 1 < 0$ for all $c \in ( 0 , 1 ) , w ( c ) \geq 0$ for all $c \in ( 0 , 1 )$ , satisfying (i) in Definition 10.

Since h is continuous on (0, 1) from (b), w is continuous on (0, 1), satisfying (ii) in Definition 10.

Since in addition, $\textstyle \int _ { 0 } ^ { c } h ( t )$ dt converges for all $c \in ( 0 , 1 )$ from (c), by Corollary 4, $\textstyle \int _ { 0 } ^ { c } ( t - 1 ) h ( t ) d t$ t converges for all $c \in ( 0 , 1 )$ from (c), satisfying (iii) in Definition 10. Therefore, by Corollary 5, $\textstyle \int _ { 0 } ^ { c } t h ( t )$ dt converges for all $c \in ( 0 , 1 )$ . This ensures both f and g are well-defined on (0, 1).

$\begin{array} { r } { f ( c ) = \int _ { 0 } ^ { c } ( t - 1 ) h ( t ) d t + f _ { 0 } = \int _ { 0 } ^ { c } w ( t ) d t + f _ { 0 } } \end{array}$ , satisfying (iv) in Definition 10.

For all $c \in ( 0 , 1 )$ , since $\begin{array} { r } { w ( c ) = ( c - 1 ) h ( c ) , h ( c ) = \frac { w ( c ) } { c - 1 } } \end{array}$ . Therefore,

$\begin{array} { r } { g ( c ) = \int _ { 0 } ^ { c } t h ( t ) d t + g _ { 0 } = \int _ { 0 } ^ { c } w ( t ) \frac { t } { t - 1 } d t + g _ { 0 } } \end{array}$ , which satisfies (v) in Definition 10.

(vi) in Definition 10 is automatically satisfied from (d).

Therefore, $( f ( c ) , g ( c ) )$ is a generalized weighted abstention confidence threshold reward scheme.

We aim to show that generalized weighted abstention confidence threshold reward schemes are non-hackable confidence reward schemes. Let $( f ( c ) , g ( c ) )$ be a generalized weighted abstention confidence threshold reward scheme.

Since $f ^ { \prime } ( c ) = w ( c )$ and $\begin{array} { r } { g ^ { \prime } ( c ) = w ( c ) \frac { c } { c - 1 } } \end{array}$ from (iv) and (v) in Definition 10 respectively and w is continuous on (0, 1) from (ii) in Definition $1 0 , f$ and g have continuous first derivatives on (0, 1). Furthermore, $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { w ( c ) } { c - 1 } } \end{array}$ and $\begin{array} { r } { \frac { g ^ { \prime } ( c ) } { c } = \frac { w ( c ) } { c - 1 } } \end{array}$ . Hence, h is well defined and $\begin{array} { r } { h ( c ) = \frac { w ( c ) } { c - 1 } } \end{array}$

Since $w ( c ) \geq 0$ for all $c \in ( 0 , 1 )$ from (i) in Definition 10 and $c - 1 < 0$ for all $c \in ( 0 , 1 ) , h ( c ) \leq 0$ for all $c \in ( 0 , 1 )$ ). Since $w ( c )$ is continuous on (0, 1) from (ii) in Definition 10, it follows that $h ( c )$ must also be continuous on (0, 1).

By Corollary 4, since $w ( c ) = ( c - 1 ) h ( c )$ and $\textstyle \int _ { 0 } ^ { c } w ( t )$ dt converges for all $c \in ( 0 , 1 )$ from (iii) in Definition 10, $\textstyle \int _ { 0 } ^ { c } h ( t ) d t$ converges for all $c \in ( 0 , 1 )$ . Therefore, by Corollary $5 , \int _ { 0 } ^ { c } t h ( t ) d t$ converges for all $c \in ( 0 , 1 )$

Since $f _ { 0 } \geq g _ { 0 }$ from (vi) in Definition 10, $w ( c ) = ( c - 1 ) h ( c )$ for all $c \in ( 0 , 1 )$ and $\begin{array} { r } { w ( c ) \frac { c } { c - 1 } = c h ( c ) } \end{array}$ for all $\begin{array} { r } { c \in ( 0 , 1 ) , ( \int _ { 0 } ^ { c } w ( t ) d t + f _ { 0 } , \int _ { 0 } ^ { c } w ( t ) \frac { t } { t - 1 } d t + g _ { 0 } ) = ( \int _ { 0 } ^ { c } ( t - 1 ) h ( t ) d t + f _ { 0 } , \int _ { 0 } ^ { c } t h ( t ) d t + g _ { 0 } ) } \end{array}$ is a non-hackable confidence reward scheme on $( 0 , 1 )$ by Theorem 2.

As shown in Theorem 6, generalized weighted abstention confidence threshold reward schemes form an alternative parametrization of non-hackable confidence reward schemes, providing an alternative perspective from the decision theoretic point of view.

The scenario in [18] is a special case with all of the following additional constraints:

(i) $w ( t )$ has a well-defined continuous extension at $0 ^ { + }$ by setting $w ( 0 ) = w ( 0 ^ { + } )$

(ii) $f _ { 0 } = g _ { 0 } = 0$

(iii) $\textstyle \int _ { 0 } ^ { 1 } w ( t ) d t = 1$

## A.3 Importance of Interpretability Property

While both [6] and [18] considered the Proper Scoring and Best Effort properties, they did not consider the Interpretability property. Nonetheless, their proposed Brier-1 [6] and Brier-log Hybrid [18] reward schemes satisfy the Interpretability property. As demonstrated in Corollary 2, proper scoring reward schemes that also satisfy the Interpretability property encourage a response confidence value close to the true probability of answering the question correctly even when the allowed confidence values are discretized. This is useful as the difference in the optimal confidence value to the true probability of answering the question correctly can be bounded by the maximum gap of two consecutive allowable confidence values when the set of allowable confidence values is finite. For instance, this applies to settings where the LLM is tasked to output a raw unnormalized confidence value as an integer between 0 and 100 inclusive.

This section provides a proof that reward schemes that satisfy the Proper Scoring property must also satisfy the Interpretability property, demonstrating that we did not further restrict the set of possible non-hackable confidence reward schemes by inclusion of the Interpretability property.

Theorem 7. Reward schemes $( f ( c ) , g ( c ) )$ that satisfy the Proper Scoring property must also satisfy the Interpretability property.

Remark 9. A more general version of Theorem 7, which does not assume continuity of $f ( c )$ and $g ( c )$ , and its corresponding proof, can be found in Lemma A.1 of [28]. A different explanation of why Theorem 7 holds can befound in [30].

## Proof of Theorem 7.

Since $( f ( c ) , g ( c ) )$ satisfies the Proper Scoring property, by Lemma 1, h is well-defined and $h ( c ) =$ $\begin{array} { r } { \frac { f ^ { \prime } ( c ) } { c - 1 } = \frac { g ^ { \prime } ( c ) } { c } } \end{array}$ for all $c \in ( 0 , 1 )$ .

Suppose the Interpretability property is not satisfied. Then, by Lemma 2, there exists $p \in ( 0 , 1 )$ such that $h ( p ) > 0$

Since f and g are assumed to have continuous first derivatives on (0, 1), h is continuous on $( 0 , 1 )$ Therefore, there exist $\epsilon > 0$ such that $| h ( c ) - h ( p ) | < h ( p )$ for all $c \in ( p - \epsilon , p + \epsilon )$ . Note that since $h ( c )$ must be well-defined for al $c \in ( p - \epsilon , p + \epsilon ) , p - \epsilon , p + \epsilon \in [ 0 , 1 ]$ . Since $\epsilon > 0$ $\begin{array} { r } { 0 \le p - \epsilon < p - \frac { \epsilon } { 2 } < p < p + \epsilon \le 1 } \end{array}$

$$
c \in [ p - \frac {\epsilon}{2}, p), | h (c) - h (p) | <   h (p) \implies h (c) - h (p) > - h (p) \implies h (c) > 0.
$$

Let $r ( c ) = R ( c , p )$ . By Lemma $\begin{array} { r } { 3 , r ^ { \prime } ( c ) = \frac { \partial } { \partial c } R ( c , p ) = ( c - p ) h ( c ) < 0 } \end{array}$ for all $c \in [ p - \textstyle { \frac { \epsilon } { 2 } } , p )$

Therefore, $\begin{array} { r } { r ( p ) - r ( c ) = \int _ { c } ^ { p } ( c - p ) h ( c ) < 0 \implies r ( c ) > r ( p ) } \end{array}$ for all $c \in [ p - \frac { \epsilon } { 2 } , p )$ . Hence, $p \notin$ argmax ${ \bf \Phi } _ { c } r ( c ) \implies p \notin$ argmax ${ } _ { c } R ( c , p )$ . Therefore, we obtain a contradiction on the Proper Scoring property and all reward schemes that satisfy the Proper Scoring property must also satisfy the Interpretability property.

## B Brier-log Hybrid Naming

This section of the appendix aims to justify why the Brier-log Hybrid reward scheme is named as such. To recall, Brier-log Hybrid reward scheme is the reward scheme $( f ( c ) , g ( c ) )$ , where $f ( c ) = c$ and $g ( c ) = c + \ln ( 1 - c )$ . In short, the Brier-log Hybrid reward scheme resembles the log loss at higher confidences and a scaled version of Brier-1 score at lower confidences.

## B.1 Calculations for Brier-1 reward scheme

To understand the comparison with the Brier-1 reward scheme when the probability of answering correctly is low, we need to compute $R _ { m a x } ( c )$ and $R ( c , p )$ . To recall, the Brier-1 reward scheme, originally proposed (but not named as such) by [6], is defined as $( 1 - ( 1 - c ) ^ { 2 } , - c ^ { 2 } )$

For the Brier-1 reward scheme, the maximum expected reward with probability of correctness c is

$$
\begin{array}{r l} & R _ {m a x} (c) \\ & = c (1 - (1 - c) ^ {2}) + (1 - c) (- c ^ {2}) \\ & = c (1 - 1 + 2 c - c ^ {2}) - c ^ {2} + c ^ {3} \\ & = 2 c ^ {2} - c ^ {3} - c ^ {2} + c ^ {3} \\ & = c ^ {2}. \end{array}
$$

The expected reward with confidence c and probability of correctness p is

$$
\begin{array}{r l} & R (c, p) \\ & = p (1 - (1 - c) ^ {2}) + (1 - p) (- c ^ {2}) \\ & = p (1 - 1 - 2 c + c ^ {2}) - c ^ {2} + p c ^ {2} \\ & = - 2 p c + p c ^ {2} - c ^ {2} + p c ^ {2} \\ & = 2 p c - c ^ {2} \end{array}
$$

## B.2 Low confidence

When $c \in ( 0 , 1 )$ is close to 0, the incorrectness reward is $\begin{array} { r } { c + \ln ( 1 - c ) \approx c - c - \frac { c ^ { 2 } } { 2 } = - \frac { c ^ { 2 } } { 2 } } \end{array}$ . The correctness reward is approximately c.

The maximum possible expected reward is $\begin{array} { r } { R _ { m a x } ( c ) \approx c ^ { 2 } + ( 1 - c ) ( - \frac { c ^ { 2 } } { 2 } ) \approx c ^ { 2 } - \frac { c ^ { 2 } } { 2 } = \frac { c ^ { 2 } } { 2 } } \end{array}$ , which is exactly half of that in the Brier-1 reward scheme.

Up to the second order terms, for the Brier-log Hybrid function, assuming both p and c are small,

$$
\begin{array}{r l} & {R (c, p)} \\ & {= p (c) + (1 - p) (c + \ln (1 - c))} \\ & {\approx p c + (1 - p) (c - c - \frac {c ^ {2}}{2})} \\ & {= p c + (1 - p) (- \frac {c ^ {2}}{2})} \\ & {= p c - \frac {c ^ {2}}{2} + \frac {p c ^ {2}}{2}} \\ & {\approx p c - \frac {c ^ {2}}{2}} \end{array}
$$

which is one-half of the corresponding value for Brier-1.

## B.3 High confidence

As observed by [18], both Brier-log Hybrid and Log Loss have a heavy overconfidence penalty. In this short subsection, we explain from an intuitive mathematical perspective how Brier-log Hybrid resembles the Log Loss reward scheme when the confidence c is high.

Table 3: List of corresponding h(c) for each reward scheme

<table><tr><td>Reward Scheme</td><td> $h(c)$ </td></tr><tr><td>Correctness-only</td><td>0</td></tr><tr><td>Log- $k$ </td><td> $\frac{k}{c(c-1)}$ </td></tr><tr><td>Log Loss</td><td> $\frac{1}{c(c-1)}$ </td></tr><tr><td>Brier- $k$ </td><td> $-2k$ </td></tr><tr><td>Brier Score</td><td> $-2$ </td></tr><tr><td>Brier-Log Hybrid</td><td> $\frac{1}{c-1}$ </td></tr><tr><td>Overconfidence- $k$ </td><td> $-\frac{k}{(\frac{1}{k}+c)((k+1)\ln(k+1)-k)}$ </td></tr><tr><td>Underconfidence- $k$ </td><td> $-\frac{1}{(\frac{1}{k}+(1-c))(1-\frac{\ln(k+1)}{k})}$ </td></tr></table>

When $c \in ( 0 , 1 )$ is close to 1, ln $c \approx 0$ . Therefore, the correctness reward of c approximates $c + \ln ( c )$ The incorrectness reward of $c + \ln ( 1 - c )$ stays the same as Log Loss reward scheme except for an additional reward of c. Hence, Brier-log Hybrid approximates Log Loss with an additional constant reward of c regardless of the correctness of the answer.

## C Determination of whether reward schemes in experiments are non-hackable confidence

For all reward schemes parametrized by k, we assume that k is non-negative.

The reward schemes used in experiments all have $h ( c )$ , as defined in Definition $^ { 6 , }$ well defined. The corresponding values of $h ( c )$ are in Table 3. Note that all the $h ( c )$ in Table 3 are non-positive for $c \in ( 0 , 1 )$ , assuming positive k. Moreover, assuming positive $k , { \dot { h } } ( c ) < 0$ for all $c \in ( 0 , 1 )$ for all examined examined reward schemes except Correctness-only. Hence, by Theorem 3, apart from Correctness-only, all other examined reward schemes are strictly non-hackable confidence over the same interval they are non-hackable confidence.

By Theorem 1, it remains to check the values of a such that $f ( a ^ { + } ) \geq g ( a ^ { + } )$ holds

## C.1 Correctness-only

Since $f ( c ) = 1 > g ( c ) = 0$ for all $c \in ( 0 , 1 )$ , Correctness-only is a non-hackable confidence reward scheme on (0, 1).

## C.2 Log-k

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Recall that for Log-k,  $f(c) = 1 + k \ln(c)$  and  $g(c) = k \ln(1 - c)$ .
 $f(c)-g(c)$ $=1+k\ln(c)-k\ln(1-c)$ $=1+k(\ln(c)-\ln(1-c))$ $=1+k\ln(\frac{c}{1-c})$
Therefore,  $f(c) \geq g(c)$ $\Leftrightarrow 1 + k \ln(\frac{c}{1-c}) \geq 0$ $\Leftrightarrow k \ln(\frac{c}{1-c}) \geq -1$ $\Leftrightarrow \ln(\frac{c}{1-c}) \geq -\frac{1}{k}$
</div>

⇔ $\frac { c } { 1 - c } \geq e ^ { - \frac { 1 } { k } }$

⇔ c ≥ (1 − c)e<sup>−</sup> <sup>1</sup>k (which holds since $c \in ( 0 , 1 ) \implies 1 - c > 0 )$

$$
\Leftrightarrow c \geq e ^ {- \frac {1}{k}} - c e ^ {- \frac {1}{k}}
$$

$$
\Leftrightarrow c + c e ^ {- \frac {1}{k}} \geq e ^ {- \frac {1}{k}}
$$

$$
\Leftrightarrow c (1 + e ^ {- \frac {1}{k}}) \geq e ^ {- \frac {1}{k}}
$$

$$
\Leftrightarrow c \geq \frac {e ^ {- \frac {1}{k}}}{1 + e ^ {- \frac {1}{k}}}
$$

Therefore, Log-k is a non-hackable confidence reward scheme on $\left( \frac { e ^ { - \frac { 1 } { k } } } { 1 + e ^ { - \frac { 1 } { k } } } , 1 \right)$ . An alternative explanation for why Log-k is a hackable confidence reward scheme on $( 0 , 1 )$ can be found in [6].

In particular, when $\begin{array} { r } { k = { \frac { 1 } { \ln { 2 0 2 } } } , e ^ { - { \frac { 1 } { k } } } = e ^ { - \ln { 2 0 2 } } = { \frac { 1 } { 2 0 2 } } } \end{array}$ . Therefore, $\mathrm { L o g } { - } { \frac { 1 } { \ln { 2 0 2 } } }$ is a non-hackable confidence reward scheme on $\left( { \frac { 1 } { 2 0 3 } } , 1 \right)$ . For our experiments, the minimum possible representable confidence value is $\frac { 1 } { 2 0 2 }$ , hence confidence reward hacking cannot take place in $\operatorname { L o g - } \frac { 1 } { \ln { 2 0 2 } }$ . In practice, with a sufficiently small k, which need not be too small (e.g. less than 0.1), confidence reward hacking is unlikely to take place.

## C.3 Log Loss

Recall that $f ( c ) = \ln ( c )$ and $g ( c ) = \ln ( 1 - c )$

Hence, $f ( c ) \geq g ( c ) \Leftrightarrow \ln ( c ) \geq \ln ( 1 - c ) \Leftrightarrow c \geq 1 - c \Leftrightarrow 2 c \geq 1 .$

Therefore, the Log Loss reward scheme is non-hackable confidence only on (0.5, 1).

## C.4 Brier-k

Recall that for Brier- $k , f ( c ) = 1 - k ( 1 - c ) ^ { 2 } { \mathrm { ~ a n d ~ } } g ( c ) = - k c ^ { 2 }$

$$
f (c) - g (c)
$$

$$
= 1 - k (1 - c) ^ {2} - (- k c ^ {2})
$$

$$
= 1 - k (1 - 2 c + c ^ {2}) + k c ^ {2}
$$

$$
= 1 - k + 2 k c - k c ^ {2} + k c ^ {2}
$$

$$
= 1 - k + 2 k c
$$

Therefore, $\begin{array} { r } { f ( c ) \geq g ( c ) \Leftrightarrow 1 - k + 2 k c \geq 0 \Leftrightarrow 2 k c \geq k - 1 \Leftrightarrow c \geq \frac { k - 1 } { 2 k } . } \end{array}$

Hence, Brier-k is non-confidence hackable on (0, 1) for all $k \in \mathsf { \Gamma } ( 0 , 1 ]$ and on $\textstyle { \left( { \frac { k - 1 } { 2 k } } , 1 \right) }$ for all $k \in ( 1 , \infty )$

An alternative explanation for why Brier-k is a non-hackable confidence reward scheme over $( 0 , 1 )$ if and only i ${ \mathrm { ~ f ~ } } k \in ( 0 , 1 ]$ (excluding the Interpretability property) can be found in [6].

## C.5 Brier Score

Recall that the Brier Score reward scheme has $f ( c ) = - ( 1 - c ) ^ { 2 } { \mathrm { ~ a n d ~ } } g ( c ) = - c ^ { 2 }$

$$
f (c) - g (c) = - (1 - c) ^ {2} - (- c ^ {2}) = - (1 - 2 c + c ^ {2}) + c ^ {2} = - 1 + 2 c
$$

Therefore, $f ( c ) \geq g ( c )$ if and only ${ \mathrm { i f ~ } } c \in [ 0 . 5 , 1 )$

Hence, Brier score is a non-hackable confidence reward scheme only over (0.5, 1).

## C.6 Brier-log Hybrid

Since $f ( c ) - g ( c ) = - \ln ( 1 - c )$ is always non-negative for all $c \in ( 0 , 1 )$ , Brier-Log Hybrid is non-hackable confidence.

An alternative explanation for why Brier-log Hybrid is a strict non-hackable confidence reward scheme over $( 0 , 1 \bar { ) }$ (excluding the strict Interpretability property) can be found on [18].

## C.7 Overconfidence-k

Let $l ( k ) = ( k + 1 ) \ln ( k + 1 ) - k$ . Since $l ( 0 ^ { + } ) = 0$ and $l ^ { \prime } ( k ) = \ln ( k + 1 ) + 1 - 1 = \ln ( k + 1 ) > 0$ for all positive $k , l ( k ) > 0$ for all positive k.

Since $\begin{array} { r } { f ( c ) - g ( c ) = \frac { k \ln ( c k + 1 ) } { ( k + 1 ) \ln ( k + 1 ) - k } } \end{array}$ is always non-negative for all $c \in ( 0 , 1 )$ , Overconfidence-k is non-hackable confidence.

## C.8 Underconfidence-k

Let $l ( k ) = k - \ln ( 1 + k )$ . Since $l ( 0 ^ { + } ) = 0$ and $\begin{array} { r } { l ^ { \prime } ( k ) = 1 - \frac { 1 } { 1 + k } > 0 } \end{array}$ for all positive $k , l ( k )$ is positive for all positive k.

Therefore, $\begin{array} { l } { f ( c ) ~ - ~ g ( c ) ~ = ~ \frac { - k \ln ( 1 - \frac { k c } { 1 + k } ) } { k - \ln ( 1 + k ) } } \end{array}$ is always non-negative for all $c \in \mathsf { \Gamma } ( 0 , 1 )$ . Hence, Underconfidence-k is a non-hackable confidence reward scheme on (0, 1).

## D Determination of Non-hackable Confidence Reward Schemes with Overconfidence and Underconfidence Bias

Intuitively, overconfidence bias means that the LLM will receive a better reward when overconfident relative to the true probability of answering correctly by δ, compared to being underconfident by δ. When unsure about the well-calibrated subjective probability of answering correctly, LLM is incentivized to be overconfident about its correctness.

Likewise, underconfidence bias means that the LLM will receive a better reward when underconfident relative to the true probability of answering correctly by δ, compared to being overconfident by δ. When unsure about the well-calibrated subjective probability of answering correctly, LLM is incentivized to be underconfident about its correctness.

Definition 5 is restated below. Note that for simplicity, in this section of the appendix, we only consider non-hackable confidence reward schemes.

Definition 5. A non-hackable confidence reward scheme $( f ( p ) , g ( p ) )$ has overconfidence bias (respectively, underconfidence bias) if the miscalibration penalty $R _ { p e n }$ satisfies thefollowing property: For all $\delta > 0$ and $p \in ( 0 , 1 )$ ) such that $p { + } \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 ) , R _ { p e n } ( p + \delta , p ) < R _ { p e n } ( p - \delta , p )$ (respectively $R _ { p e n } ( p + \delta , p ) > R _ { p e n } ( p - \delta , p ) )$

Note that $R _ { p e n }$ is as defined in Definition 3.

[18] proposed Brier-log Hybrid and alluded to its underconfidence bias. In Appendix D.1, we generalize this finding by proving a sufficient condition for a non-hackable reward confidence scheme to exhibit overconfidence or underconfidence bias.

In Appendix D.2, we determine whether each of the non-reward hackable schemes examined in Appendix C exhibit overconfidence bias or underconfidence bias.

## D.1 Sufficient Condition for Overconfidence/Underconfidence Bias

For this subsection, we further assume that both the reward for answering correctly $f ( c )$ and the reward for answering incorrectly $g ( c )$ have continuous second-order derivatives. This allows for the expected reward function $R ( c , p )$ to have continuous second-order derivatives.

Lemma 8. Let r be afunction on (0, 1) such that r has continuous second-order derivatives on (0, 1) and $r ^ { \prime \prime }$ is strictly decreasing on $( 0 , 1 )$ . Suppose r has a maximum at $p .$

Then, for all $\delta > 0$ such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 ) , - r ^ { \prime } ( p + \delta ) > r ^ { \prime } ( p - \delta )$

## Proof of Lemma 8.

Take any $\delta > 0$ such that such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 )$

Since r has continuous second-order derivatives in $( 0 , 1 ) , r ^ { \prime }$ must have continuous first-order derivatives in (0, 1). Since r has a maximum at $p , r ^ { \prime } ( p ) = 0$ and $r ^ { \prime \prime } ( p ) \leq 0$

$$
r ^ {\prime} (p + \delta) = r ^ {\prime} (p + \delta) - r ^ {\prime} (p) = \int_ {p} ^ {p + \delta} r ^ {\prime \prime} (c) d c
$$

$$
- r ^ {\prime} (p - \delta) = r ^ {\prime} (p) - r ^ {\prime} (p - \delta) = \int_ {p - \delta} ^ {p} r ^ {\prime \prime} (c) d c
$$

Let $u = c + \delta$ . Then, when $c = p , u = p + \delta ;$ and when $c = p - \delta , u = p$ . Moreover, $d u = d c$ Therefore, since $r ^ { \prime \prime } ( u - \delta ) > r ^ { \prime \prime } ( \bar { u } )$ for all $u \in [ p , p + \delta ]$

$$
- r ^ {\prime} (p - \delta) = \int_ {p} ^ {p + \delta} r ^ {\prime \prime} (u - \delta) d u > \int_ {p} ^ {p + \delta} r ^ {\prime \prime} (u) d u = r ^ {\prime} (p + \delta)
$$

Hence, $r ^ { \prime } ( p - \delta ) < - r ^ { \prime } ( p + \delta )$ , yielding the result in the lemma.

Lemma 9. Let r be afunction on (0, 1) such that r has continuous second-order derivatives on (0, 1) and $r ^ { \prime \prime }$ is strictly decreasing on (0, 1). Suppose r has a maximum at p.

Then, for all $\delta > 0$ such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 ) , r ( p ) - r ( p + \delta ) > r ( p ) - r ( p - \delta )$

## Proof of Lemma 9.

Take any $\delta > 0$ such that such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 )$

For all $x \in ( 0 , \delta ]$ , since $p - \delta \leq p - x < p < p + x \leq p + \delta ,$ , both $p - x$ and p + x are in (0, 1).

Therefore, by Lemma $8 , - r ^ { \prime } ( p + x ) > r ^ { \prime } ( p - x )$ for all $x \in ( 0 , \delta ]$

Note that $\begin{array} { r } { r ( p ) - r ( p + \delta ) = \int _ { p } ^ { p + \delta } - r ^ { \prime } ( c _ { + } ) d c _ { + } \mathrm { a n d } r ( p ) - r ( p - \delta ) = \int _ { p - \delta } ^ { p } r ^ { \prime } ( c _ { - } ) d c _ { - } } \end{array}$

Let $u _ { + } = c _ { + } - p .$ . Then, when ${ c _ { + } } = p , { u _ { + } } = 0$ and when $c _ { + } = p + \delta , u _ { + } = \delta$ . Moreover, $d u _ { + } = d c _ { + }$

let $u _ { - } = p - c _ { - }$ . Then, when $c _ { - } = p , u _ { - } = 0$ and when $c _ { - } = p - \delta , u _ { - } = \delta$ . Moreover, $d u _ { - } = - d c _ { - } .$

Hence, $\begin{array} { r } { r ( p ) - r ( p + \delta ) = \int _ { p } ^ { p + \delta } - r ^ { \prime } ( c _ { + } ) d c _ { + } = \int _ { 0 } ^ { \delta } - r ^ { \prime } ( p + u _ { + } ) d u _ { + } } \end{array}$ and

$$
r (p) - r (p - \delta) = \int_ {p - \delta} ^ {p} r ^ {\prime} (c _ {-}) d c _ {-} = \int_ {\delta} ^ {0} - r ^ {\prime} (p - u _ {-}) d u _ {-} = \int_ {0} ^ {\delta} r ^ {\prime} (p - u _ {-}) d u _ {-}.
$$

Since $- r ^ { \prime } ( p + u ) \ > \ r ^ { \prime } ( p - u )$ for all $u \in \mathsf { \Gamma } ( 0 , \delta ]$ , we conclude that $\begin{array} { r } { \int _ { 0 } ^ { \delta } - r ^ { \prime } ( p + u _ { + } ) \ d u _ { + } \ > } \end{array}$ $\begin{array} { r } { \int _ { 0 } ^ { \delta } r ^ { \prime } ( p - u _ { - } ) } \end{array}$ du<sub>−</sub>, therefore, $r ( p ) - r ( p + \delta ) > r ( p ) - r ( p - \delta )$ ■

Similarly, using the idea of the proofs of Lemmas 8 and 9, we can prove the following result:

Lemma 10. Let r be afunction on (0, 1) such that r has continuous second-order derivatives on (0, 1) and $r ^ { \prime \prime }$ is strictly increasing on (0, 1). Suppose r has a maximum at p.

Then, for all $\delta > 0$ such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 ) , r ( p ) - r ( p + \delta ) < r ( p ) - r ( p - \delta )$

Theorem 8. Let $( f ( c ) , g ( c ) )$ be a non-hackable confidence reward scheme such that f and g have continuous second-order derivatives. Let $R ( c , p )$ be as defined in Definition 2. Then, the following hold:

(i) If for all $\textstyle p \in ( 0 , 1 ) , \frac { \partial ^ { 2 } R } { \partial c ^ { 2 } }$ is strictly decreasing with respect to c on (0, 1) when keeping p fixed, then $\bar { ( f ( c ) , g ( c ) ) }$ has underconfidence bias.

(ii) Iffor all $\textstyle p \in ( 0 , 1 ) , \frac { \partial ^ { 2 } R } { \partial c ^ { 2 } }$ is strictly increasing with respect to c on (0, 1) when keeping p fixed, then $( f ( c ) , g ( c ) )$ has overconfidence bias.

## Proof of Theorem 8.

Since the proofs for (i) and (ii) are similar, we show only the proof of (i) here.

Fix any $p \in ( 0 , 1 )$ . Note that when p is fixed, by Proper Scoring property of non-hackable confidence reward schemes in Definition $2 , R ( c , p )$ has a maximum at $c = p .$

Let $r _ { p } ( c ) = R ( c , p )$ for all $c \in ( 0 , 1 )$ . Note that by Definition 2 $, R ( c , p ) = p f ( c ) + ( 1 - p ) g ( c )$ Therefore, since $f ( c )$ and $g ( c )$ are assumed to have continuous second-order derivatives, $R ( c , p )$ has continuous second-order derivatives.

Since $r _ { p }$ has a maximum at $c = p$ and $\begin{array} { r } { r _ { p } ^ { \prime \prime } ( c ) = \frac { \partial ^ { 2 } } { \partial c ^ { 2 } } ( R ( c , p ) ) } \end{array}$ is strictly decreasing on $( 0 , 1 )$ , by Lemma $9 , r _ { p } ( p ) - r ( p + \delta ) > r _ { p } ( p ) - r _ { p } ( p - \delta )$ for all $\delta > 0$ such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , \bar { 1 } )$

Therefore, for all $\delta > 0$ such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 ) , R ( p , p ) - R ( p + \delta , p ) >$ $R ( p , p ) - R ( p - \delta , p )$ , which implies that $R _ { p e n } ( p + \delta , p ) > R _ { p e n } ( p - \delta , p )$

Therefore, $( f ( c ) , g ( c ) )$ has underconfidence bias.

Theorem 8 provides a sufficient condition of overconfidence and underconfidence bias based on the second derivative of the expected reward function $R ( c , p )$ , allowing us to show that Overconfidence-k has overconfidence bias and Underconfidence-k has underconfidence bias in the next subsection.

## D.2 Determination of Reward Schemes with Overconfidence/Underconfidence Bias

## D.2.1 Correctness-only

Since Correctness-only reward scheme has a reward that is independent of the confidence of the LLM, the miscalibration penalty is always zero. Therefore, the Correctness-only reward scheme has neither overconfidence nor underconfidence bias.

## D.2.2 Overconfidence-k

$$
\begin{array}{l} R (c, p) \\ = p f (c) + (1 - p) g (c) \\ = p \frac {(k + 1) \ln (c k + 1) - c k}{(k + 1) \ln (k + 1) - k} + (1 - p) \frac {\ln (c k + 1) - c k}{(k + 1) \ln (k + 1) - k} \\ = \frac {p ((k + 1) \ln (c k + 1) - c k) + (1 - p) (\ln (c k + 1) - c k)}{(k + 1) \ln (k + 1) - k} \\ = \frac {p (k + 1) \ln (c k + 1) + (1 - p) \ln (c k + 1) - c k}{(k + 1) \ln (k + 1) - k} \\ = \frac {(p k + p + 1 - p) (\ln (c k + 1)) - c k}{(k + 1) \ln (k + 1) - k} \\ = \frac {(p k + 1) \ln (c k + 1) - c k}{(k + 1) \ln (k + 1) - k} \\ \frac {\partial R}{\partial c} = \frac {(p k + 1) \frac {k}{c k + 1} - k}{(k + 1) \ln (k + 1) - k} \\ \frac {\partial^ {2} R}{\partial c ^ {2}} = \frac {- (p k + 1) \frac {k ^ {2}}{(c k + 1) ^ {2}}}{(k + 1) \ln (k + 1) - k} \end{array}
$$

As shown in Appendix $\mathbf { C . 7 } , ( k + 1 ) \ln ( k + 1 ) - k$ is always positive for all $k > 0$ . Note that $p k + 1 > 0$ and $k ^ { 2 } > 0$ since p is non-negative and k is positive.

As c increases in (0, 1), ck + 1 increases and is positive, hence $( c k + 1 ) ^ { 2 }$ increases and is positive. This implies that $\frac { k ^ { 2 } } { ( c k + 1 ) ^ { 2 } }$ decreases and is positive.

Therefore, keeping p fixed, $\frac { \partial ^ { 2 } R } { \partial c ^ { 2 } }$ increases with respect to c on the interval $( 0 , 1 )$ . Hence, by Theorem 8, Overconfidence-k has overconfidence bias.

## D.2.3 Brier-k, where $k \leq 1$

The expected reward with confidence c and probability of correctness p is

$$
\begin{array}{l} {R (c, p)} \\ {= p (1 - k (1 - c) ^ {2}) + (1 - p) (- k c ^ {2})} \\ {= p (1 - k (1 - 2 c + c ^ {2})) - k c ^ {2} + p k c ^ {2}} \\ {= p (1 - k + 2 c k - c ^ {2} k) - k c ^ {2} + p k c ^ {2}} \\ {= p - p k + 2 p c k - p c ^ {2} k - k c ^ {2} + p k c ^ {2}} \\ {= p - p k + 2 p c k - k c ^ {2}} \end{array}
$$

Therefore, the miscalibration penalty $R _ { p e n } ( c , p )$ is equal to

$$
\begin{array}{r l} & R (p, p) - R (c, p) \\ & = (p - p k + 2 p ^ {2} k - k p ^ {2}) - (p - p k + 2 p c k - k c ^ {2}) \\ & = p - p k + 2 p ^ {2} k - k p ^ {2} - p + p k - 2 p c k + k c ^ {2}) \\ & = 2 p ^ {2} k - k p ^ {2} - 2 p c k + k c ^ {2}) \\ & = k (2 p ^ {2} - p ^ {2} - 2 p c + c ^ {2}) \\ & = k (p ^ {2} - 2 p c + c ^ {2}) \\ & = k (p - c) ^ {2} \end{array}
$$

Therefore, for all $\delta > 0$ and $p \in ( 0 , 1 )$ such that $p + \delta \in ( 0 , 1 )$ and $p - \delta \in ( 0 , 1 ) , R _ { p e n } ( p + \delta , p ) =$ $k \delta ^ { 2 } = R _ { p e n } ( p - \delta , p )$ . Therefore, Brier-k has neither overconfidence bias nor underconfidence bias.

## D.2.4 Underconfidence-k

$$
\begin{array}{r l} & {R (c, p)} \\ & {= p f (c) + (1 - p) g (c)} \\ & {= p \frac {k c + \ln (1 - \frac {k c}{1 + k})}{k - \ln (1 + k)} + (1 - p) \frac {k c + (k + 1) \ln (1 - \frac {k c}{1 + k})}{k - \ln (1 + k)}} \\ & {= \frac {p (k c + \ln (1 - \frac {k c}{1 + k})) + (1 - p) (k c + (k + 1) \ln (1 - \frac {k c}{1 + k}))}{k - \ln (1 + k)}} \\ & {= \frac {p k c + p \ln (1 - \frac {k c}{1 + k}) + (1 - p) k c + (1 - p) (k + 1) \ln (1 - \frac {k c}{1 + k})}{k - \ln (1 + k)}} \\ & {= \frac {k c + (p + (1 - p) (k + 1)) \ln (1 - \frac {k c}{1 + k})}{k - \ln (1 + k)}} \\ & {= \frac {k c + (p + (1 - p) k + (1 - p)) \ln (1 - \frac {k c}{1 + k})}{k - \ln (1 + k)}} \\ & {= \frac {k c + (1 + k - p k) \ln (1 - \frac {k c}{1 + k})}{k - \ln (1 + k)}} \\ & {\frac {\partial R}{\partial c}} \\ & {= \frac {k + (1 + k - p k) \frac {- \frac {k}{1 + k}}{1 - \frac {k c}{1 + k}}}{k - \ln (1 + k)}} \\ & {= \frac {k + (1 + k - p k) \frac {- k}{1 + k - k c}}{k - \ln (1 + k)}} \\ & {= \frac {k}{k - \ln (1 + k)} + \frac {1 + k - p k}{k - \ln (1 + k)} \frac {- k}{1 + k - k c}} \\ & {\frac {\partial^ {2} R}{\partial c ^ {2}}} \\ & {= \frac {1 + k - p k}{k - \ln (1 + k)} \frac {(- k) (- 1) (- k)}{(1 + k - k c) ^ {2}}} \\ & {= \frac {1 + k - p k}{k - \ln (1 + k)} \frac {- k ^ {2}}{(1 + k - k c) ^ {2}}} \\ & {= \frac {1 + k (1 - p)}{k - \ln (1 + k)} \frac {- k ^ {2}}{(1 + k (1 - c)) ^ {2}}} \end{array}
$$

Let $p \in ( 0 , 1 )$ be fixed. As shown in Appendix C.8, $k - \ln ( 1 + k ) > 0$ for all $k > 0 .$ . Furthermore, since $p \in ( 0 , \dot { 1 } ) , 1 + k ( 1 - p ) > 1 > 0 \mathrm { a n d } - k ^ { 2 } < 0$ . Likewise, for all $c \in ( 0 , 1 ) , 1 + k ( 1 - c ) > 1 > 0$ Therefore, $\begin{array} { r } { \frac { \partial ^ { 2 } R } { \partial c ^ { 2 } } < 0 } \end{array}$ for all $p , c \in ( 0 , 1 )$ .

Keeping p fixed, since $1 + k ( 1 - c )$ decreases and is positive as c increases within the interval (0, 1), $\frac { 1 } { ( 1 + k ( 1 - c ) ) ^ { 2 } }$ increases as c increases within the interval (0, 1). Therefore, $\frac { \partial ^ { 2 } R } { \partial c ^ { 2 } }$ is strictly decreasing with respect to $c \in ( 0 , 1 )$ . Therefore, by Theorem 8, Underconfidence-k has underconfidence bias.

## D.2.5 Brier-log Hybrid

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$R(c,p)$ $= pf(c) + (1 - p)g(c)$ $= pc + (1 - p)(c + \ln (1 - c))$ $= pc + c + \ln (1 - c) - pc - p\ln (1 - c)$ $= c + \ln (1 - c) - p\ln (1 - c)$ $= c + (1 - p)\ln (1 - c)$ $\frac{\partial R}{\partial c} = 1 - \frac{1 - p}{1 - c}$ $\frac{\partial^2R}{\partial c^2} = -\frac{1 - p}{(1 - c)^2}$
</div>

For all $p \in \mathsf { ( 0 , 1 ) }$ , keeping p fixed, $\frac { \partial ^ { 2 } R } { \partial c ^ { 2 } }$ is strictly decreasing as c increases in the range (0, 1). Therefore, by Theorem 8, Brier-log Hybrid has underconfidence bias.

## E Experimental Setting Details

## E.1 Experimental Procedure

Our experiment methodology can be described as follows:

Stage 1 - Dataset preparation: For each dataset obtained, we split the data into training and testing sets, and classified the questions according to their difficulty in easy, medium and hard, as elaborated in Appendix E.2.

Stage 2 - Fine-tuning preparation: For each dataset, the LLM with temperature 1 is tasked to answer 1024 questions in the training set and to provide its confidence score in JSON format. Its responses are collected for the next stage. More details on how this is done can be found on Appendix E.3.

Stage 3 - Supervised Fine-tuning (SFT): The responses from the previous step are reformatted to an XML-like format to help the LLM follow instructions in the next step. The LLM is fine-tuned on the reformatted responses using LoRA [31] for 1 epoch.

Stage 4 - RL: We fine-tuned the LLM to perform confidence calibration using RL. The LLM is tasked to answer the question and to provide a confidence value c˜, an integer in [0, 100], to its answer, along with some reasoning on how it arrived at the answer and c˜. To determine the reward, the answer is graded for correctness and c˜ is normalized into the implied confidence c by setting $\begin{array} { r } { c = \frac { \tilde { c } + 0 . 5 } { 1 0 1 } } \end{array}$

Instead of Group Relative Policy Optimization (GRPO) [15], we used GRPO Done Right (Dr GRPO) [25] loss, which removes the standard deviation normalization term and gives each token equal weightage. This loss removes the confidence calibration bias found in GRPO [32]. We ran our experiments for 500 steps, with 8 generations per question and 64 questions per batch. Since we aim to train the LLM in both confidence calibration and reasoning improvements, the loss is taken over the entire output.

Step 5 - Evaluation: The LLM is then evaluated using the test set of the dataset, with the answers and verbalized confidences recorded. The LLM samples 16 responses for each question at a temperature of 1. The base model and the model after Stage 3 (i.e., the SFT model) are evaluated as baselines. We evaluated the model using accuracy, Expected Calibration Error (10 bins) [33], Area Under

Receiver Operating Characteristic Curve [34], [35], Brier score [36], the average Brier-1 reward [6] and calibration bias.

Details regarding the exact prompt formats can be found on Appendix E.8.

## E.2 Dataset details and preprocessing

• BigMath [21]: We took the filtered dataset from Hugging Face found at open-r1/Big-Math-RL-Verified-Processed<sup>2</sup>. The filtered dataset contains only questions where the answers are parseable by math-verify<sup>3</sup> and the Llama 3.1 (8B) solve rate is provided by [21].

We labeled the questions with the highest tercile of Llama 3.1 (8B) solve rate as easy, middle tercile of Llama 3.1 (8B) solve rate as medium and lower tercile of Llama 3.1 (8B) solve rate as difficult. The easy questions have Llama 3.1 (8B) solve rates greater than 0.765625, the hard questions have corresponding solve rates below 0.1875, while the medium questions have solve rates in between 0.1875 and 0.765625 inclusive.

We randomly split the dataset into train and test splits in an approximately 97:3 ratio.

To verify LLM answers, we used math-verify, with an accuracy tolerance of two decimal places for non-exact answers represented in decimal.

• DeepMath-103K [22]: This dataset consists of difficult math questions and has been used to finetune LLM via RL to improve reasoning performance [22]. The difficulty rating is based on estimates from GPT-4o after being prompted with the difficulty level guidance in the Art of Problem Solving<sup>4</sup>[22].

We labeled the questions with the lowest tercile of difficulty rating as easy, middle tercile of difficulty rating as medium and highest tercile of difficulty rating as difficult. The easy questions have difficulty under 5, the hard questions have difficulty above 6 while the medium questions have difficulty in between 5 and 6 inclusive.

The train and test dataset splits were taken from trl-lib/DeepMath-103K at Hugging Face<sup>5</sup>. The questions were matched with the original dataset from zwhe99/DeepMath-103K<sup>6</sup> to obtain the difficulty values.

We attempted to filter off multiple choice questions and yes/no questions by filtering off the questions whose case-insensitive answer after removing the dollar sign in LaTeX or the full stop at the back is in the set {yes, no, true, false, a, b, c, d}. In such questions, the question format and the ground truth format are often inconsistent, complicating reliable grading of the answers.

In particular, the yes/no questions often require proof to ensure that true understanding of mathematical concepts is rewarded. Automating the grading of the reasoning accuracy in proofs is difficult and can be unreliable.

The verification method is the same as in BigMath.

• HotpotQA [20]: This is a textual reasoning dataset where LLMs are given multiple sources and are asked reading comprehension questions about the sources. The entries in the dataset have 10 sources, where 2 sources are relevant and 8 are irrelevant. The difficulties of easy, medium and hard are already provided in the dataset.

We used the distractor subset.

For training, we used the provided training set, while for validation, we used the provided validation set. The training set contains easy, medium and hard questions while the validation set only contains hard questions.

An answer is marked as correct if the ROUGE-1 [37] score exceeds 0.7.

• HotpotQA-Modified: Modified from the procedure in [6] with the most important difference being the inclusion of dataset difficulty, for each dataset entry in the HotpotQA dataset, we perform the following at random:

Table 4: Number of questions with valid JSON responses in Qwen 2.5 (3B) Instruct out of the 1024 questions in the fine-tuning preparation stage.

<table><tr><td>Dataset</td><td>Questions with valid JSON</td></tr><tr><td>BigMath</td><td>1019</td></tr><tr><td>DeepMath-103K</td><td>1015</td></tr><tr><td>HotpotQA</td><td>1015</td></tr><tr><td>HotpotQA-Modified</td><td>1019</td></tr></table>

– With probability $_ { 3 } ^ { \mathrm { ~ 1 ~ } } , 2$ irrelevant sources are removed. These are relabeled as easy questions.

– With probability $\textstyle { \frac { 1 } { 3 } }$ , 1 irrelevant source and 1 relevant source are removed. These are relabeled as medium questions.

– With probability $\frac 1 3$ , both relevant sources are removed, leaving the LLM with no choice but to deduce from prior knowledge. These are relabeled as hard questions.

The evaluation method is the same as in HotpotQA.

## E.3 Procedure details for fine-tuning preparation

The LLM was instructed to answer a 1024-question subset of questions in the training set in JavaScript Object Notation (JSON) format. A format enforcer in vLLM [38] was used to guide the LLM to produce valid a JSON response. The JSON contains the reasoning, answer, confidence analysis and confidence attributes. The LLM is instructed to place its reasoning on the reasoning attribute, its final answer on the answer attribute, its justification of its confidence value on the confidence analysis attribute and the confidence value on the confidence attribute. The LLM is instructed to output an integer value between 0 and 100 inclusive in the confidence attribute. For BigMath and DeepMath-103K, the output token limit was 2048. For HotpotQA and HotpotQA-Modified, the output token limit was 1024.

If a valid JSON response was produced, each of the attributes were extracted and rearranged into an Extensible Markup Language (XML) format for the SFT stage. As the LLM may not realize that its LaTeX expressions involving backslashes were being escaped, we attempt to undo the escapes to recover the original string.

If the JSON response was invalid, the question will be discarded and its response will not be included in the SFT dataset.

As shown in Table 4, our experiments in Qwen 2.5 (3B) Instruct had the LLM output valid JSON responses for over 95% of the 1024-question subset for each of the datasets tested, ensuring a sufficient dataset size for the SFT stage.

For the exact prompt formats used, please refer to Appendix E.8.

## E.4 Training details for SFT

The LLM was loaded in 16-bit. We used LoRA [31] with a rank of 32 and an α of 32. We used the AdamW [39] optimizer with a weight decay of 0.01. The learning rate was set to 0.0002 with 5 warmup steps. The effective batch size is 16. We trained over the SFT dataset, prepared from the previous fine-tuning preparation stage, for 1 epoch.

## E.5 Training details for RL

The LLM was loaded in 16-bit. We used the AdamW [39] 8-bit optimizer with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9$ . The learning rate was set to $1 0 ^ { - 5 }$ with 25 warmup steps. The maximum gradient norm was set to 0.1. The experiments were run for 500 steps. We used an effective batch size of 512 with 8 generations per question and 64 questions per batch.

For consistency with the SFT stage, we used LoRA [31] with a rank of 32 and an α of 32. The LLM was initialized from the LoRA adapter in the supervised finetuning step because the LLM has been fine-tuned in the supervised finetuning step to follow the XML output format which is used in the current RL step.

The RL code was implemented using Unsloth [40], significantly decreasing the video random access memory (VRAM) requirements.

For consistency, the output token limits were the same as during the fine-tuning preparation stage.

Our implementation of RL contained additional formatting rewards in addition to the reward in the reward scheme. These were graded independently of the answer. Our formatting reward was based on an Unsloth GRPO tutorial [40]<sup>7</sup>. To discourage overly long answers which would highly likely be graded as incorrect, a valid answer reward of 0.5 was provided if the length did not exceed 1000 characters. A reward of 0.5 was awarded for fully following the XML format. An additional reward of 0.1 was awarded for each opening and closing tag in the XML format that appears exactly once. A reward of 1 was provided by providing a valid confidence value, i.e. an integer between 0 and 100 inclusive. In the event of an invalid confidence value, the worst possible confidence value for the answer was assigned, further discouraging reward hacking of the LLM by not following the format.

For the exact prompt formats used, please refer to Appendix E.8.

## E.6 Details on Evaluation Protocol

During the evaluation stage, we tested the base model on both the JSON format and XML format. The JSON format was evaluated to ensure that the SFT training in XML format did not significantly decrease performance of the base model. The XML format was evaluated to demonstrate that the base model is prompt-sensitive, i.e. performance varies by the exact prompt format used. We also evaluated on the SFT models and each of the RL models, corresponding to the different reward schemes, on the SFT model.

For both the JSON and XML formats, we provided a system prompt instructing the LLM on the format to use and used a format enforcer in vLLM [38] to guide the LLM towards following the instructed format. For consistency, the output token limits were the same as in the RL stage and the fine-tuning preparation stage.

Inspired by [6], in the event of invalid output, we give the LLM a second chance during evaluation and probe them for the final answer and confidence.

If the model was instructed to format in JSON, the LLM will additionally be guided to provide a valid confidence value, i.e. an integer in between 0 and 100 inclusive and an answer not exceeding 1000 characters. Nearly all ground truths in the BigMath, DeepMath-103K and the HotpotQA dataset are under 1000 characters long. Since an invalid JSON string cannot be parsed, in the event the JSON string is invalid, the LLM will be asked to output its answer in a follow-up prompt, then the confidence.

If the model was instructed to output in XML, in the event of an invalid XML output, the evaluation code will attempt to extract valid XML tags for answer and confidence as parts of the output may be salvaged. In the event of multiple valid tags, only the last tag will be taken. If both the answer and the confidence can be retrieved, the output will be graded as normal. If the answer cannot be retrieved, the LLM will be asked for the answer in a follow-up. If a valid confidence cannot be retrieved, the LLM will be asked for the confidence in a follow-up. If both the answer and the confidence cannot be taken, the LLM will be asked for the answer, followed by a separate prompt asking for the confidence in a follow-up.

The follow-up prompts are identical for both JSON and XML formats. Since the initial insructions to output in JSON and XML no longer apply, the system prompt was removed with the remaining conversational trace left intact.

To prevent the LLM from taking extra time to answer the question via extra reasoning beyond the original given token limit, the LLM was tasked to output only the final answer when asked in a follow-up, and in LaTeX format for mathematical datasets. In addition, to reduce the chances of LLM failing to follow instructions, we prefixed the response with "Final Answer:" using a format enforcer. The output token limit, including the prefix, was set to 64 in BigMath and DeepMath-103K and 32 for HotpotQA and HotpotQA-Modified.

When the LLM is asked for its confidence in a follow-up, the LLM will be guided using a format enforcer to output an integer in between 0 and 100 inclusive, representing the confidence value.

Just like in [6], once the answer and confidence values were retrieved, the same dataset-dependent correctness checking algorithm is used, allowing for a fair comparison.

For the exact prompt formats used, please refer to Appendix E.8.

## E.7 Evaluation Metrics Used

To evaluate our model, we used the following metrics.

• Accuracy: The proportion of sampled responses that were judged as correct answers to the corresponding questions.

• Expected Calibration Error (ECE) [33]: Following [6], let M be the number of bins and N questions in total. The ith bin is denoted as $B _ { i } ,$ defined as the set of question-answerconfidence tuples whose confidence lies within the ith bin. Let the average confidence of the tuples in the ith bin be denoted as con $\mathsf { f } ( B _ { i } )$ ) and the accuracy of the tuples in the ith bin be denoted as $\operatorname { a c c } ( B _ { i } )$ . Then, the ECE is defined as $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \frac { | B _ { i } | } { N } | \operatorname { a c c } ( B _ { i } ) - \operatorname { c o n f } ( B _ { i } ) | [ 6 ] } \end{array}$ Following [6], we used $M = 1 0$ bins. In addition, we use equally spaced bins.

• Area Under Receiver Operating Characteristic Curve (AUROC) [34], [35]: Since a zero ECE can be obtained by uninformedly aligning the confidence score with the accuracy of the model [41], we use the AUROC metric to measure the ability of the LLM to differentiate between the answers that are likely correct and the answers that are likely wrong via its verbalized confidence. Where TPR stands for the true positive rate and FPR stands for the false positive rate, AUROC is defined as $\begin{array} { r l } {  { \int _ { 0 } ^ { 1 } \mathrm { T P R } \big ( \mathrm { F P R } ^ { - 1 } ( t ) \big ) } } \end{array}$ dt [6].

• Brier score [36]: Let N be the number of questions. For question $i ,$ let the question be denoted as $q _ { i } ,$ , the corresponding answer denoted as $a _ { i } .$ , and the confidence, normalized in the range $[ 0 , 1 ] .$ , be denoted as $c _ { i }$ . Let isCorrec $( q _ { i } , a _ { i } )$ denote the function that validates whether $a _ { i }$ is the correct answer to $q _ { i } ,$ returning 1 if the answer is correct and 0 if the answer is wrong. Brier score is a proper scoring metric that is used to evaluate probability calibration abilities, given by the formula: $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \mathrm { i s C o r r e c t } ( q _ { i } , a _ { i } ) - c _ { i } ) ^ { 2 } } \end{array}$

• Average Brier-1 reward [6]: A non-hackable confidence reward scheme repurposed as an integrated metric that takes into account both the accuracy and the confidence calibration of the model. The average Brier-1 reward is equal to the difference between the accuracy and the Brier score since Brier-1 offers an additional reward of 1 to the negative of the Brier score if and only if the answer is correct.

• Calibration bias: Calibration bias is defined as the accuracy minus the average confidence over the evaluated answers. The average confidence over the evaluated answers is the expected accuracy assuming the confidence values are well-calibrated. A negative value means that the model is overconfident on average, while a positive value means that the model is underconfident on average.

## E.8 Prompt formats

The JSON system prompt format is used for the fine-tuning preparation stage, and when evaluating base models.

The XML prompt format is used during the SFT stage to train the LLM to follow instructions for the RL stage. It is also used during the RL stage and the XML evaluation for consistency.

The JSON and XML system prompt formats are as shown below.

<table><tr><td>JSON system prompt format</td></tr></table>

Respond in the JSON format: {"reasoning": string, "answer": string, "confidence\_analysis": string, "confidence": int}

Provide your internal thought process in "reasoning" attribute.

Provide only your final answer in "answer" attribute.

For mathematical questions, provide your final answer in LaTeX format. Provide your analysis on the uncertainty of your answer in "confidence\_analysis" attribute, taking into account various factors that may lead to your answer being different or incorrect.

In "confidence" attribute, express your confidence as an integer between 0 and 100 inclusive.

## XML system prompt format

When answering questions, follow these instructions:

1) Enclose your internal thought process with <reasoning> and </reasoning> tags.

2) Enclose your final answer with <answer> and </answer> tags. For mathematical answers, answer in LaTeX format.

3) Enclose your analysis on the uncertainty of your answer with <confidence\_analysis> and </confidence\_analysis> tags, taking into account various factors that may lead to your answer being different or incorrect.

4) Enclose your confidence with <confidence> and </confidence> tags. Confidence is an integer between 0 and 100 inclusive, with higher values indicating higher confidence. Higher confidence means higher score if answer is correct but lower score if answer is incorrect. Your aim is to maximize your score considering the confidence of your answer given your internal thought process to the question.

Respond in the following format:

<reasoning>

</reasoning>

</answer>

<confidence\_analysis>

</confidence\_analysis>

</confidence>

For BigMath and DeepMath-103K, the prompt is simply the question in the dataset. For HotpotQA and HotpotQA-Modified, there are multiple sources which need to be presented to form a prompt. The sources are presented in random order to mitigate the effects of data leakage. The LLM prompt for questions in the HotpotQA and HotpotQA-Modified datasets are as follows, where N is the number of sources in the question:

Question prompt format for HotpotQA and HotpotQA-Modified

```txt
Based on the information sources given below and your existing knowledge, answer the following question: [QUESTION]

Source 1: [TITLE OF SOURCE 1]
[TEXT OF SOURCE 1]

Source 2: [TITLE OF SOURCE 2]
[TEXT OF SOURCE 2]

...

Source [N]: [TITLE OF SOURCE N]
[TEXT OF SOURCE N]
```

For the evaluation, sometimes, the LLM may produce an invalid response despite the help of a format enforcer. Therefore, the LLM may be prompted a second time for the answer or the confidence. Their respective prompt formats are as shown below.

Prompt format for second LLM probe for answer (BigMath and DeepMath-103K)

Reasoning token limit reached. Please output only your final answer within 64 tokens. Express your answer in LaTeX.

Prompt format for second LLM probe for answer (HotpotQA and HotpotQA-Modified)

Reasoning token limit reached. Please output only your final answer within 32 tokens.

Note that the token limit for the second probe of the answer is 64 for BigMath and DeepMath-103K and 32 for HotpotQA and HotpotQA-Modified.

Prompt format for second LLM probe for confidence

Please output your confidence as an integer between 0 and 100 inclusive.

## E.9 Hardware used

Only central processing unit (CPU) power is required for dataset preparation and result analysis.

For our finetuning dataset preparation, SFT, RL and evaluation stages, we used a single instance of 3g.47gb in an NVIDIA H100 graphics processing unit (GPU). For finetuning dataset preparation, supervised finetuning, RL, we used 48 GiB of CPU random access memory (RAM). For the evaluation stage, we used 128 GiB of CPU RAM.

For Qwen 2.5 (3B) Instruct, the preprocessing stage and the finetuning stage each took under 10 minutes per dataset. The RL experiment took anywhere between 20 hours and 3 days depending on the dataset and the reward scheme. The evaluation stage completed in under 3 days per dataset, with execution time varying depending on dataset.

## E.10 Dataset and Model Licenses

DeepMath-103K [22] is MIT licensed, BigMath [21] has Apache 2.0 license, and HotpotQA [20] has CC-BY-SA-4.0 licence, which allows derivatives of the work such as HotpotQA-Modified.

Qwen 2.5 (3B) Instruct [24] is licensed using Qwen Research Agreement license, which permits use only for research and evaluation purposes. Llama 3.2 (3B) Instruct [42] is licensed using the Llama

3.2 Community License Agreement. Llama 3.1 (8B) Instruct [42] is licensed using the Llama 3.1 Community License Agreement.

All the licenses involved permit use for research.

## F Full results

This section provides the accuracy, AUROC, Brier-1, Brier score, ECE (10-bins) and calibration bias of all datasets used in the experiment for Qwen 2.5 (3B) Instruct. For accuracy, AUROC and Brier-1, higher is better, while for Brier score, ECE (10-bins) and calibration bias, closer to 0 is better.

The results are split by dataset difficulty. Note that the HotpotQA test set only contains questions labeled as hard, therefore, only the overall results in HotpotQA are shown.

The AUROC is not shown if the accuracy is outside the range (0.001, 0.999) because the small number of correct answers or the small number of wrong answers may not form a reliable sample of the ability of the LLM to differentiate correct answers from wrong answers.

For all tables in the section, base models and SFT models are included for reference at the top, while the middle section contains hackable confidence reward schemes and the bottom contains the reward schemes in the overconfidence to underconfidence spectrum, arranged from the overconfident end at the top to the underconfident end at the bottom.

The SFT (XML) model performed similarly to the Base (JSON) model. The SFT (XML) model was fine-tuned via SFT from reformatted outputs of Base (JSON) to XML format. This shows that the SFT stage did not result in a significant loss of accuracy from the original model.

The Base (XML) format was shown to highlight the prompt sensitivity of LLM, demonstrating the importance of the exact prompt format used in evaluation. This draws parallels to [25], which highligted that including a prompt template for Qwen 2.5 models negatively impacts performance prior to RL.

Tables 5, 6, 7 and 8 respectively show the performance statistics of Qwen 2.5 (3B) Instruct when trained on BigMath and tested on the entire test set, easy subset, medium subset and hard subset, of the test set of BigMath.

Tables 9, 10, 11 and 12 respectively show the performance statistics of Qwen 2.5 (3B) Instruct when trained on DeepMath-103K and tested on the entire test set, easy subset, medium subset and hard subset, of the test set of DeepMath-103K.

Table 13 shows the performance statistics of Qwen 2.5 (3B) Instruct when trained on HotpotQA and tested on the test set of HotpotQA.

Tables 14, 15, 16 and 17 respectively show the performance statistics of Qwen 2.5 (3B) Instruct when trained on HotpotQA-Modified and tested on the entire test set, easy subset, medium subset and hard subset, of the test set of HotpotQA-Modified.

Table 5: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the test set of BigMath when trained with BigMath dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.2531</td><td>0.5908</td><td>-0.4212</td><td>0.6743</td><td>0.6988</td><td>-0.6983</td></tr><tr><td>Base (XML)</td><td>0.5644</td><td>0.5982</td><td>0.1736</td><td>0.3908</td><td>0.3910</td><td>-0.3844</td></tr><tr><td>SFT (XML)</td><td>0.2499</td><td>0.5959</td><td>-0.4266</td><td>0.6765</td><td>0.7018</td><td>-0.7017</td></tr><tr><td>Log Loss</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0002</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Log-1</td><td>0.6336</td><td>0.8602</td><td>0.4841</td><td>0.1495</td><td>0.0826</td><td>0.0826</td></tr><tr><td>Brier-2</td><td>0.6713</td><td>0.8434</td><td>0.5278</td><td>0.1436</td><td>0.0388</td><td>0.0028</td></tr><tr><td> $\text{Log-} \frac{1}{\ln 202}$ </td><td>0.6656</td><td>0.7849</td><td>0.4885</td><td>0.1771</td><td>0.0649</td><td>-0.0503</td></tr><tr><td>Correctness-only</td><td>0.6739</td><td>0.5277</td><td>0.3565</td><td>0.3174</td><td>0.3171</td><td>-0.3171</td></tr><tr><td>Overconfidence-1000</td><td>0.6704</td><td>0.6681</td><td>0.4364</td><td>0.2340</td><td>0.1826</td><td>-0.1825</td></tr><tr><td>Overconfidence-4</td><td>0.6685</td><td>0.7492</td><td>0.4908</td><td>0.1777</td><td>0.0564</td><td>-0.0376</td></tr><tr><td>Overconfidence-1</td><td>0.6723</td><td>0.7669</td><td>0.5080</td><td>0.1643</td><td>0.0560</td><td>-0.0151</td></tr><tr><td>Brier-1</td><td>0.6676</td><td>0.8195</td><td>0.5151</td><td>0.1525</td><td>0.0244</td><td>-0.0243</td></tr><tr><td>Underconfidence-1</td><td>0.6666</td><td>0.8282</td><td>0.5132</td><td>0.1534</td><td>0.0307</td><td>-0.0192</td></tr><tr><td>Underconfidence-4</td><td>0.6769</td><td>0.8343</td><td>0.5272</td><td>0.1498</td><td>0.0441</td><td>0.0423</td></tr><tr><td>Brier-log Hybrid</td><td>0.6682</td><td>0.8521</td><td>0.5253</td><td>0.1429</td><td>0.0474</td><td>-0.0054</td></tr></table>

Table 6: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the easy subset of the test set of BigMath when trained with BigMath dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.4391</td><td>0.5595</td><td>-0.0842</td><td>0.5233</td><td>0.5321</td><td>-0.5315</td></tr><tr><td>Base (XML)</td><td>0.8587</td><td>0.5524</td><td>0.7228</td><td>0.1359</td><td>0.1175</td><td>-0.1093</td></tr><tr><td>SFT (XML)</td><td>0.4494</td><td>0.5681</td><td>-0.0628</td><td>0.5122</td><td>0.5219</td><td>-0.5217</td></tr><tr><td>Log Loss</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0002</td><td>0.0048</td><td>-0.0048</td></tr><tr><td>Brier Score</td><td>0.0003</td><td>-</td><td>0.0000</td><td>0.0004</td><td>0.0047</td><td>-0.0047</td></tr><tr><td>Log-1</td><td>0.9388</td><td>0.7847</td><td>0.8574</td><td>0.0814</td><td>0.1645</td><td>0.1645</td></tr><tr><td>Brier-2</td><td>0.9530</td><td>0.7412</td><td>0.8981</td><td>0.0549</td><td>0.0866</td><td>0.0743</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.9501</td><td>0.6862</td><td>0.8837</td><td>0.0664</td><td>0.1351</td><td>0.1351</td></tr><tr><td>Correctness-only</td><td>0.9540</td><td>0.5181</td><td>0.9090</td><td>0.0450</td><td>0.0405</td><td>-0.0405</td></tr><tr><td>Overconfidence-1000</td><td>0.9529</td><td>0.5987</td><td>0.9035</td><td>0.0495</td><td>0.0670</td><td>0.0621</td></tr><tr><td>Overconfidence-4</td><td>0.9509</td><td>0.6201</td><td>0.8804</td><td>0.0705</td><td>0.1540</td><td>0.1533</td></tr><tr><td>Overconfidence-1</td><td>0.9521</td><td>0.6179</td><td>0.8836</td><td>0.0685</td><td>0.1597</td><td>0.1596</td></tr><tr><td>Brier-1</td><td>0.9529</td><td>0.6756</td><td>0.9012</td><td>0.0517</td><td>0.0923</td><td>0.0921</td></tr><tr><td>Underconfidence-1</td><td>0.9505</td><td>0.7292</td><td>0.8926</td><td>0.0579</td><td>0.1001</td><td>0.0996</td></tr><tr><td>Underconfidence-4</td><td>0.9552</td><td>0.7240</td><td>0.8925</td><td>0.0626</td><td>0.1233</td><td>0.1225</td></tr><tr><td>Brier-log Hybrid</td><td>0.9513</td><td>0.7566</td><td>0.8994</td><td>0.0519</td><td>0.1010</td><td>0.1009</td></tr></table>

Table 7: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the medium subset of the test set of BigMath when trained with BigMath dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.2387</td><td>0.5701</td><td>-0.4519</td><td>0.6905</td><td>0.7135</td><td>-0.7130</td></tr><tr><td>Base (XML)</td><td>0.6096</td><td>0.5734</td><td>0.2559</td><td>0.3538</td><td>0.3472</td><td>-0.3381</td></tr><tr><td>SFT (XML)</td><td>0.2262</td><td>0.5702</td><td>-0.4767</td><td>0.7030</td><td>0.7263</td><td>-0.7262</td></tr><tr><td>Log Loss</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0002</td><td>0.0048</td><td>-0.0048</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Log-1</td><td>0.6981</td><td>0.7655</td><td>0.5074</td><td>0.1907</td><td>0.1417</td><td>0.1417</td></tr><tr><td>Brier-2</td><td>0.7520</td><td>0.7580</td><td>0.5902</td><td>0.1618</td><td>0.0929</td><td>0.0735</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.7427</td><td>0.6964</td><td>0.5712</td><td>0.1715</td><td>0.0311</td><td>0.0227</td></tr><tr><td>Correctness-only</td><td>0.7545</td><td>0.5231</td><td>0.5149</td><td>0.2395</td><td>0.2376</td><td>-0.2374</td></tr><tr><td>Overconfidence-1000</td><td>0.7506</td><td>0.6171</td><td>0.5610</td><td>0.1896</td><td>0.1088</td><td>-0.1080</td></tr><tr><td>Overconfidence-4</td><td>0.7474</td><td>0.6718</td><td>0.5761</td><td>0.1712</td><td>0.0354</td><td>0.0272</td></tr><tr><td>Overconfidence-1</td><td>0.7532</td><td>0.6829</td><td>0.5903</td><td>0.1629</td><td>0.0428</td><td>0.0391</td></tr><tr><td>Brier-1</td><td>0.7480</td><td>0.7244</td><td>0.5823</td><td>0.1657</td><td>0.0507</td><td>0.0318</td></tr><tr><td>Underconfidence-1</td><td>0.7444</td><td>0.7373</td><td>0.5785</td><td>0.1660</td><td>0.0461</td><td>0.0445</td></tr><tr><td>Underconfidence-4</td><td>0.7573</td><td>0.7436</td><td>0.5832</td><td>0.1741</td><td>0.1143</td><td>0.1134</td></tr><tr><td>Brier-log Hybrid</td><td>0.7460</td><td>0.7600</td><td>0.5878</td><td>0.1582</td><td>0.0532</td><td>0.0532</td></tr></table>

Table 8: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the hard subset of the test set of BigMath when trained with BigMath dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.0892</td><td>0.5631</td><td>-0.7134</td><td>0.8027</td><td>0.8440</td><td>-0.8436</td></tr><tr><td>Base (XML)</td><td>0.2358</td><td>0.5598</td><td>-0.4376</td><td>0.6734</td><td>0.6995</td><td>-0.6957</td></tr><tr><td>SFT (XML)</td><td>0.0823</td><td>0.5723</td><td>-0.7250</td><td>0.8073</td><td>0.8496</td><td>-0.8496</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier Score</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Log-1</td><td>0.2748</td><td>0.7581</td><td>0.1019</td><td>0.1729</td><td>0.0569</td><td>-0.0561</td></tr><tr><td>Brier-2</td><td>0.3185</td><td>0.7265</td><td>0.1085</td><td>0.2100</td><td>0.1387</td><td>-0.1380</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.3139</td><td>0.6824</td><td>0.0248</td><td>0.2891</td><td>0.3028</td><td>-0.3028</td></tr><tr><td>Correctness-only</td><td>0.3229</td><td>0.5225</td><td>-0.3355</td><td>0.6584</td><td>0.6639</td><td>-0.6639</td></tr><tr><td>Overconfidence-1000</td><td>0.3172</td><td>0.6105</td><td>-0.1393</td><td>0.4565</td><td>0.4935</td><td>-0.4934</td></tr><tr><td>Overconfidence-4</td><td>0.3169</td><td>0.6635</td><td>0.0296</td><td>0.2873</td><td>0.2869</td><td>-0.2869</td></tr><tr><td>Overconfidence-1</td><td>0.3211</td><td>0.6949</td><td>0.0633</td><td>0.2577</td><td>0.2382</td><td>-0.2382</td></tr><tr><td>Brier-1</td><td>0.3117</td><td>0.7084</td><td>0.0758</td><td>0.2359</td><td>0.1933</td><td>-0.1932</td></tr><tr><td>Underconfidence-1</td><td>0.3146</td><td>0.7136</td><td>0.0822</td><td>0.2324</td><td>0.1991</td><td>-0.1984</td></tr><tr><td>Underconfidence-4</td><td>0.3279</td><td>0.7136</td><td>0.1191</td><td>0.2087</td><td>0.1081</td><td>-0.1073</td></tr><tr><td>Brier-log Hybrid</td><td>0.3169</td><td>0.7330</td><td>0.1022</td><td>0.2148</td><td>0.1674</td><td>-0.1671</td></tr></table>

Table 9: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the test set of DeepMath-103K when trained with DeepMath-103K dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.1796</td><td>0.5713</td><td>-0.5268</td><td>0.7063</td><td>0.7427</td><td>-0.7415</td></tr><tr><td>Base (XML)</td><td>0.3228</td><td>0.5577</td><td>-0.2743</td><td>0.5971</td><td>0.6145</td><td>-0.6078</td></tr><tr><td>SFT (XML)</td><td>0.1634</td><td>0.5732</td><td>-0.5626</td><td>0.7260</td><td>0.7636</td><td>-0.7633</td></tr><tr><td>Log Loss</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0002</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Log-1</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier-2</td><td>0.0002</td><td>-</td><td>0.0000</td><td>0.0002</td><td>0.0048</td><td>-0.0048</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.4327</td><td>0.6119</td><td>0.1952</td><td>0.2375</td><td>0.0556</td><td>0.0527</td></tr><tr><td>Correctness-only</td><td>0.4238</td><td>0.5034</td><td>-0.1437</td><td>0.5675</td><td>0.5686</td><td>-0.5684</td></tr><tr><td>Overconfidence-1000</td><td>0.4204</td><td>0.5446</td><td>0.1737</td><td>0.2468</td><td>0.0822</td><td>-0.0821</td></tr><tr><td>Overconfidence-4</td><td>0.4294</td><td>0.6609</td><td>0.2052</td><td>0.2242</td><td>0.0155</td><td>-0.0140</td></tr><tr><td>Overconfidence-1</td><td>0.4138</td><td>0.6752</td><td>0.1930</td><td>0.2207</td><td>0.0341</td><td>0.0299</td></tr><tr><td>Brier-1</td><td>0.4063</td><td>0.6558</td><td>0.1828</td><td>0.2235</td><td>0.0468</td><td>0.0082</td></tr><tr><td>Underconfidence-1</td><td>0.4159</td><td>0.6375</td><td>0.1870</td><td>0.2289</td><td>0.0296</td><td>-0.0153</td></tr><tr><td>Underconfidence-4</td><td>0.3921</td><td>0.6452</td><td>0.1684</td><td>0.2237</td><td>0.0422</td><td>0.0360</td></tr><tr><td>Brier-log Hybrid</td><td>0.2635</td><td>0.6302</td><td>0.0781</td><td>0.1854</td><td>0.0273</td><td>0.0271</td></tr></table>

Table 10: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the easy subset of the test set of DeepMath-103K when trained with DeepMath-103K dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.1958</td><td>0.5777</td><td>-0.5148</td><td>0.7107</td><td>0.7412</td><td>-0.7404</td></tr><tr><td>Base (XML)</td><td>0.4331</td><td>0.5909</td><td>-0.0689</td><td>0.5020</td><td>0.5118</td><td>-0.5070</td></tr><tr><td>SFT (XML)</td><td>0.1785</td><td>0.5796</td><td>-0.5570</td><td>0.7355</td><td>0.7660</td><td>-0.7660</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0051</td><td>-0.0051</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>-0.0001</td><td>0.0001</td><td>0.0051</td><td>-0.0051</td></tr><tr><td>Log-1</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier-2</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0050</td><td>-0.0050</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.5334</td><td>0.6897</td><td>0.2933</td><td>0.2401</td><td>0.1270</td><td>0.1249</td></tr><tr><td>Correctness-only</td><td>0.5308</td><td>0.5040</td><td>0.0689</td><td>0.4619</td><td>0.4618</td><td>-0.4617</td></tr><tr><td>Overconfidence-1000</td><td>0.5262</td><td>0.5899</td><td>0.2904</td><td>0.2358</td><td>0.0120</td><td>0.0039</td></tr><tr><td>Overconfidence-4</td><td>0.5324</td><td>0.7571</td><td>0.3292</td><td>0.2032</td><td>0.0786</td><td>0.0359</td></tr><tr><td>Overconfidence-1</td><td>0.5205</td><td>0.7646</td><td>0.3118</td><td>0.2087</td><td>0.0938</td><td>0.0860</td></tr><tr><td>Brier-1</td><td>0.5108</td><td>0.7142</td><td>0.2843</td><td>0.2265</td><td>0.1289</td><td>0.0834</td></tr><tr><td>Underconfidence-1</td><td>0.5191</td><td>0.6986</td><td>0.2907</td><td>0.2284</td><td>0.1143</td><td>0.0695</td></tr><tr><td>Underconfidence-4</td><td>0.4948</td><td>0.7209</td><td>0.2692</td><td>0.2256</td><td>0.1062</td><td>0.1061</td></tr><tr><td>Brier-log Hybrid</td><td>0.3313</td><td>0.7306</td><td>0.1347</td><td>0.1966</td><td>0.0750</td><td>0.0649</td></tr></table>

Table 11: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the medium subset of the test set of DeepMath-103K when trained with DeepMath-103K dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.1508</td><td>0.5724</td><td>-0.5790</td><td>0.7298</td><td>0.7706</td><td>-0.7695</td></tr><tr><td>Base (XML)</td><td>0.3016</td><td>0.5557</td><td>-0.3137</td><td>0.6153</td><td>0.6351</td><td>-0.6280</td></tr><tr><td>SFT (XML)</td><td>0.1398</td><td>0.5677</td><td>-0.6080</td><td>0.7479</td><td>0.7876</td><td>-0.7874</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Log-1</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0049</td><td>-0.0049</td></tr><tr><td>Brier-2</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0049</td><td>-0.0049</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.4059</td><td>0.6051</td><td>0.1732</td><td>0.2328</td><td>0.0328</td><td>0.0283</td></tr><tr><td>Correctness-only</td><td>0.3940</td><td>0.5037</td><td>-0.2030</td><td>0.5970</td><td>0.5986</td><td>-0.5984</td></tr><tr><td>Overconfidence-1000</td><td>0.3886</td><td>0.5359</td><td>0.1399</td><td>0.2487</td><td>0.1123</td><td>-0.1123</td></tr><tr><td>Overconfidence-4</td><td>0.4054</td><td>0.6549</td><td>0.1817</td><td>0.2237</td><td>0.0335</td><td>-0.0335</td></tr><tr><td>Overconfidence-1</td><td>0.3855</td><td>0.6683</td><td>0.1681</td><td>0.2174</td><td>0.0253</td><td>0.0057</td></tr><tr><td>Brier-1</td><td>0.3796</td><td>0.6575</td><td>0.1611</td><td>0.2185</td><td>0.0489</td><td>-0.0170</td></tr><tr><td>Underconfidence-1</td><td>0.3876</td><td>0.6344</td><td>0.1616</td><td>0.2260</td><td>0.0402</td><td>-0.0401</td></tr><tr><td>Underconfidence-4</td><td>0.3613</td><td>0.6433</td><td>0.1450</td><td>0.2163</td><td>0.0180</td><td>0.0088</td></tr><tr><td>Brier-log Hybrid</td><td>0.2263</td><td>0.6178</td><td>0.0568</td><td>0.1695</td><td>0.0079</td><td>-0.0039</td></tr></table>

Table 12: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the hard subset of the test set of DeepMath-103K when trained with DeepMath-103K dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.2204</td><td>0.5671</td><td>-0.4435</td><td>0.6639</td><td>0.6957</td><td>-0.6941</td></tr><tr><td>Base (XML)</td><td>0.3015</td><td>0.5334</td><td>-0.3140</td><td>0.6155</td><td>0.6331</td><td>-0.6260</td></tr><tr><td>SFT (XML)</td><td>0.1958</td><td>0.5793</td><td>-0.4876</td><td>0.6835</td><td>0.7212</td><td>-0.7207</td></tr><tr><td>Log Loss</td><td>0.0003</td><td>-</td><td>0.0000</td><td>0.0004</td><td>0.0047</td><td>-0.0047</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0002</td><td>0.0048</td><td>-0.0048</td></tr><tr><td>Log-1</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier-2</td><td>0.0004</td><td>-</td><td>0.0000</td><td>0.0004</td><td>0.0046</td><td>-0.0046</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.4260</td><td>0.5671</td><td>0.1818</td><td>0.2441</td><td>0.0586</td><td>0.0567</td></tr><tr><td>Correctness-only</td><td>0.4189</td><td>0.5024</td><td>-0.1533</td><td>0.5722</td><td>0.5731</td><td>-0.5728</td></tr><tr><td>Overconfidence-1000</td><td>0.4197</td><td>0.5273</td><td>0.1704</td><td>0.2493</td><td>0.0776</td><td>-0.0754</td></tr><tr><td>Overconfidence-4</td><td>0.4167</td><td>0.6007</td><td>0.1807</td><td>0.2360</td><td>0.0304</td><td>-0.0068</td></tr><tr><td>Overconfidence-1</td><td>0.4063</td><td>0.6217</td><td>0.1736</td><td>0.2327</td><td>0.0476</td><td>0.0420</td></tr><tr><td>Brier-1</td><td>0.3976</td><td>0.6062</td><td>0.1670</td><td>0.2306</td><td>0.0184</td><td>0.0122</td></tr><tr><td>Underconfidence-1</td><td>0.4105</td><td>0.5993</td><td>0.1762</td><td>0.2342</td><td>0.0179</td><td>-0.0172</td></tr><tr><td>Underconfidence-4</td><td>0.3911</td><td>0.5921</td><td>0.1559</td><td>0.2352</td><td>0.0523</td><td>0.0459</td></tr><tr><td>Brier-log Hybrid</td><td>0.2919</td><td>0.5800</td><td>0.0850</td><td>0.2069</td><td>0.0637</td><td>0.0603</td></tr></table>

Table 13: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the test set of HotpotQA when trained with HotpotQA dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.4404</td><td>0.5850</td><td>-0.0286</td><td>0.4689</td><td>0.4730</td><td>-0.4693</td></tr><tr><td>Base (XML)</td><td>0.0479</td><td>0.5315</td><td>-0.7309</td><td>0.7788</td><td>0.8399</td><td>-0.8399</td></tr><tr><td>SFT (XML)</td><td>0.4362</td><td>0.6044</td><td>-0.0208</td><td>0.4570</td><td>0.4644</td><td>-0.4634</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0049</td></tr><tr><td>Brier Score</td><td>0.0000</td><td>-</td><td>-0.0001</td><td>0.0001</td><td>0.0058</td><td>-0.0058</td></tr><tr><td>Log-1</td><td>0.6472</td><td>0.6420</td><td>0.4277</td><td>0.2194</td><td>0.0777</td><td>-0.0776</td></tr><tr><td>Brier-2</td><td>0.6507</td><td>0.5920</td><td>0.4221</td><td>0.2286</td><td>0.1057</td><td>-0.1057</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.6537</td><td>0.5455</td><td>0.4077</td><td>0.2461</td><td>0.1491</td><td>-0.1491</td></tr><tr><td>Correctness-only</td><td>0.6571</td><td>0.5095</td><td>0.3207</td><td>0.3363</td><td>0.3346</td><td>-0.3345</td></tr><tr><td>Overconfidence-1000</td><td>0.6548</td><td>0.5273</td><td>0.3220</td><td>0.3328</td><td>0.3307</td><td>-0.3305</td></tr><tr><td>Overconfidence-4</td><td>0.6526</td><td>0.5725</td><td>0.4134</td><td>0.2393</td><td>0.1319</td><td>-0.1319</td></tr><tr><td>Overconfidence-1</td><td>0.6507</td><td>0.5854</td><td>0.4059</td><td>0.2448</td><td>0.1563</td><td>-0.1562</td></tr><tr><td>Brier-1</td><td>0.6504</td><td>0.5790</td><td>0.4144</td><td>0.2360</td><td>0.1225</td><td>-0.1225</td></tr><tr><td>Underconfidence-1</td><td>0.6551</td><td>0.6046</td><td>0.4238</td><td>0.2313</td><td>0.1175</td><td>-0.1175</td></tr><tr><td>Underconfidence-4</td><td>0.6529</td><td>0.6158</td><td>0.4242</td><td>0.2287</td><td>0.1093</td><td>-0.1093</td></tr><tr><td>Brier-log Hybrid</td><td>0.6439</td><td>0.6098</td><td>0.4128</td><td>0.2311</td><td>0.1081</td><td>-0.1080</td></tr></table>

Table 14: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the test set of HotpotQA-Modified when trained with HotpotQA-Modified dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.2806</td><td>0.6559</td><td>-0.2226</td><td>0.5032</td><td>0.5335</td><td>-0.5317</td></tr><tr><td>Base (XML)</td><td>0.0363</td><td>0.6050</td><td>-0.6437</td><td>0.6800</td><td>0.7703</td><td>-0.7703</td></tr><tr><td>SFT (XML)</td><td>0.2787</td><td>0.6472</td><td>-0.2270</td><td>0.5057</td><td>0.5347</td><td>-0.5325</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier Score</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Log-1</td><td>0.0950</td><td>0.9681</td><td>0.0536</td><td>0.0415</td><td>0.0235</td><td>-0.0235</td></tr><tr><td>Brier-2</td><td>0.2862</td><td>0.8900</td><td>0.1677</td><td>0.1185</td><td>0.0414</td><td>-0.0414</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.4186</td><td>0.7081</td><td>0.1922</td><td>0.2264</td><td>0.1038</td><td>-0.1026</td></tr><tr><td>Correctness-only</td><td>0.4204</td><td>0.5292</td><td>-0.1282</td><td>0.5486</td><td>0.5508</td><td>-0.5470</td></tr><tr><td>Overconfidence-1000</td><td>0.4135</td><td>0.6856</td><td>0.1402</td><td>0.2734</td><td>0.2300</td><td>-0.2290</td></tr><tr><td>Overconfidence-4</td><td>0.4166</td><td>0.7405</td><td>0.2149</td><td>0.2016</td><td>0.0223</td><td>-0.0129</td></tr><tr><td>Overconfidence-1</td><td>0.4204</td><td>0.7332</td><td>0.2012</td><td>0.2192</td><td>0.1182</td><td>-0.1152</td></tr><tr><td>Brier-1</td><td>0.4173</td><td>0.7474</td><td>0.2131</td><td>0.2043</td><td>0.0637</td><td>-0.0608</td></tr><tr><td>Underconfidence-1</td><td>0.4181</td><td>0.7519</td><td>0.2140</td><td>0.2041</td><td>0.0794</td><td>-0.0775</td></tr><tr><td>Underconfidence-4</td><td>0.4118</td><td>0.7481</td><td>0.2088</td><td>0.2030</td><td>0.0616</td><td>-0.0565</td></tr><tr><td>Brier-log Hybrid</td><td>0.4007</td><td>0.7440</td><td>0.1994</td><td>0.2013</td><td>0.0503</td><td>-0.0459</td></tr></table>

Table 15: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the easy subset of the test set of HotpotQA-Modified when trained with HotpotQA-Modified dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.4493</td><td>0.5908</td><td>-0.0125</td><td>0.4618</td><td>0.4672</td><td>-0.4642</td></tr><tr><td>Base (XML)</td><td>0.0588</td><td>0.5386</td><td>-0.7216</td><td>0.7804</td><td>0.8369</td><td>-0.8368</td></tr><tr><td>SFT (XML)</td><td>0.4421</td><td>0.5898</td><td>-0.0213</td><td>0.4634</td><td>0.4680</td><td>-0.4655</td></tr><tr><td>Log Loss</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0049</td></tr><tr><td>Brier Score</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Log-1</td><td>0.2022</td><td>0.9386</td><td>0.1275</td><td>0.0747</td><td>0.0271</td><td>-0.0271</td></tr><tr><td>Brier-2</td><td>0.5296</td><td>0.7723</td><td>0.3546</td><td>0.1750</td><td>0.0162</td><td>-0.0162</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.6271</td><td>0.6098</td><td>0.3953</td><td>0.2318</td><td>0.0725</td><td>-0.0320</td></tr><tr><td>Correctness-only</td><td>0.6273</td><td>0.5059</td><td>0.2612</td><td>0.3661</td><td>0.3648</td><td>-0.3634</td></tr><tr><td>Overconfidence-1000</td><td>0.6228</td><td>0.5957</td><td>0.3723</td><td>0.2506</td><td>0.1348</td><td>-0.1299</td></tr><tr><td>Overconfidence-4</td><td>0.6248</td><td>0.6326</td><td>0.3958</td><td>0.2290</td><td>0.0698</td><td>0.0697</td></tr><tr><td>Overconfidence-1</td><td>0.6328</td><td>0.6198</td><td>0.4081</td><td>0.2247</td><td>0.0506</td><td>-0.0310</td></tr><tr><td>Brier-1</td><td>0.6317</td><td>0.6359</td><td>0.4082</td><td>0.2235</td><td>0.0419</td><td>0.0025</td></tr><tr><td>Underconfidence-1</td><td>0.6245</td><td>0.6511</td><td>0.4033</td><td>0.2212</td><td>0.0414</td><td>-0.0083</td></tr><tr><td>Underconfidence-4</td><td>0.6199</td><td>0.6488</td><td>0.3972</td><td>0.2226</td><td>0.0433</td><td>0.0044</td></tr><tr><td>Brier-log Hybrid</td><td>0.6111</td><td>0.6461</td><td>0.3864</td><td>0.2247</td><td>0.0371</td><td>0.0210</td></tr></table>

Table 16: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the medium subset of the test set of HotpotQA-Modified when trained with HotpotQA-Modified dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.2726</td><td>0.6221</td><td>-0.2435</td><td>0.5161</td><td>0.5411</td><td>-0.5382</td></tr><tr><td>Base (XML)</td><td>0.0352</td><td>0.5796</td><td>-0.6395</td><td>0.6746</td><td>0.7681</td><td>-0.7681</td></tr><tr><td>SFT (XML)</td><td>0.2726</td><td>0.6152</td><td>-0.2452</td><td>0.5178</td><td>0.5407</td><td>-0.5381</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier Score</td><td>0.0001</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0049</td></tr><tr><td>Log-1</td><td>0.0619</td><td>0.9708</td><td>0.0272</td><td>0.0347</td><td>0.0268</td><td>-0.0268</td></tr><tr><td>Brier-2</td><td>0.2323</td><td>0.8839</td><td>0.1149</td><td>0.1174</td><td>0.0583</td><td>-0.0583</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.4203</td><td>0.6393</td><td>0.1704</td><td>0.2499</td><td>0.1243</td><td>-0.0881</td></tr><tr><td>Correctness-only</td><td>0.4250</td><td>0.5209</td><td>-0.1237</td><td>0.5487</td><td>0.5498</td><td>-0.5447</td></tr><tr><td>Overconfidence-1000</td><td>0.4135</td><td>0.6110</td><td>0.1207</td><td>0.2929</td><td>0.2229</td><td>-0.2217</td></tr><tr><td>Overconfidence-4</td><td>0.4187</td><td>0.6659</td><td>0.1923</td><td>0.2264</td><td>0.0541</td><td>0.0033</td></tr><tr><td>Overconfidence-1</td><td>0.4238</td><td>0.6575</td><td>0.1805</td><td>0.2433</td><td>0.1206</td><td>-0.1053</td></tr><tr><td>Brier-1</td><td>0.4184</td><td>0.6743</td><td>0.1875</td><td>0.2309</td><td>0.0896</td><td>-0.0424</td></tr><tr><td>Underconfidence-1</td><td>0.4197</td><td>0.6762</td><td>0.1902</td><td>0.2295</td><td>0.0777</td><td>-0.0615</td></tr><tr><td>Underconfidence-4</td><td>0.4158</td><td>0.6766</td><td>0.1869</td><td>0.2289</td><td>0.0816</td><td>-0.0363</td></tr><tr><td>Brier-log Hybrid</td><td>0.4003</td><td>0.6699</td><td>0.1747</td><td>0.2257</td><td>0.0691</td><td>-0.0296</td></tr></table>

Table 17: Accuracy, AUROC, Brier-1, Brier, ECE (10 bins) and calibration bias statistics for the hard subset of the test set of HotpotQA-Modified when trained with HotpotQA-Modified dataset. Calib. Bias is short for Calibration bias.

<table><tr><td>Model/Reward Scheme</td><td>Accuracy</td><td>AUROC</td><td>Brier-1</td><td>Brier</td><td>ECE</td><td>Calib. Bias</td></tr><tr><td>Base (JSON)</td><td>0.1203</td><td>0.6546</td><td>-0.4118</td><td>0.5320</td><td>0.5926</td><td>-0.5926</td></tr><tr><td>Base (XML)</td><td>0.0149</td><td>0.6180</td><td>-0.5702</td><td>0.5851</td><td>0.7062</td><td>-0.7062</td></tr><tr><td>SFT (XML)</td><td>0.1217</td><td>0.6452</td><td>-0.4144</td><td>0.5361</td><td>0.5953</td><td>-0.5938</td></tr><tr><td>Log Loss</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0000</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Brier Score</td><td>0.0000</td><td>-</td><td>0.0000</td><td>0.0001</td><td>0.0050</td><td>-0.0050</td></tr><tr><td>Log-1</td><td>0.0204</td><td>0.9829</td><td>0.0054</td><td>0.0149</td><td>0.0166</td><td>-0.0166</td></tr><tr><td>Brier-2</td><td>0.0958</td><td>0.9398</td><td>0.0325</td><td>0.0633</td><td>0.0503</td><td>-0.0503</td></tr><tr><td> $\text{Log-\frac{1}{\ln 202}}$ </td><td>0.2093</td><td>0.7020</td><td>0.0109</td><td>0.1984</td><td>0.1896</td><td>-0.1870</td></tr><tr><td>Correctness-only</td><td>0.2100</td><td>0.5276</td><td>-0.5202</td><td>0.7302</td><td>0.7370</td><td>-0.7321</td></tr><tr><td>Overconfidence-1000</td><td>0.2050</td><td>0.6807</td><td>-0.0722</td><td>0.2772</td><td>0.3357</td><td>-0.3346</td></tr><tr><td>Overconfidence-4</td><td>0.2070</td><td>0.7824</td><td>0.0567</td><td>0.1504</td><td>0.1160</td><td>-0.1110</td></tr><tr><td>Overconfidence-1</td><td>0.2055</td><td>0.7701</td><td>0.0152</td><td>0.1904</td><td>0.2106</td><td>-0.2085</td></tr><tr><td>Brier-1</td><td>0.2028</td><td>0.7788</td><td>0.0434</td><td>0.1594</td><td>0.1448</td><td>-0.1416</td></tr><tr><td>Underconfidence-1</td><td>0.2110</td><td>0.7925</td><td>0.0486</td><td>0.1624</td><td>0.1638</td><td>-0.1618</td></tr><tr><td>Underconfidence-4</td><td>0.2008</td><td>0.7764</td><td>0.0424</td><td>0.1584</td><td>0.1413</td><td>-0.1366</td></tr><tr><td>Brier-log Hybrid</td><td>0.1915</td><td>0.7612</td><td>0.0371</td><td>0.1544</td><td>0.1326</td><td>-0.1283</td></tr></table>

Table 18: Table of responses to three example questions in the final RL training batch of Qwen 2.5 (3B) Instruct for Brier-2 in the HotpotQA-Modified dataset. Con. stands for the corresponding confidence level of the response.

<table><tr><td colspan="2">Example Question A</td><td colspan="2">Example Question B</td><td colspan="2">Example Question C</td></tr><tr><td>Response</td><td>Con.</td><td>Response</td><td>Con.</td><td>Response</td><td>Con.</td></tr><tr><td>W. Chump &amp; Sons Limited</td><td>70</td><td>Unknown</td><td>0</td><td>Unknown</td><td>0</td></tr><tr><td>W. Chump &amp; Sons Limited</td><td>60</td><td>No information</td><td>0</td><td>Not provided</td><td>0</td></tr><tr><td>W. Chump &amp; Sons Limited</td><td>70</td><td>Hog</td><td>60</td><td>Unknown</td><td>0</td></tr><tr><td>W. Chump &amp; Sons</td><td>70</td><td>Unknown</td><td>0</td><td>Not enough information</td><td>0</td></tr><tr><td>W. Chump &amp; Sons Limited</td><td>70</td><td>Hog</td><td>70</td><td>Unknown</td><td>0</td></tr><tr><td>W. Chump &amp; Sons</td><td>70</td><td>Unknown</td><td>0</td><td>Unknown</td><td>0</td></tr><tr><td>W. Chump &amp; Sons Limited</td><td>70</td><td>unspecified</td><td>0</td><td>Unknown</td><td>0</td></tr><tr><td>W. Chump &amp; Sons Limited</td><td>70</td><td>Unknown</td><td>0</td><td>Not provided</td><td>0</td></tr></table>

## G Case study of selective confidence reward hacking

This appendix section shows a case study of selective confidence reward hacking in Brier-2 reward scheme for the HotpotQA-Modified dataset. This was recorded during the 500th and final RL training step of Qwen 2.5 (3B) Instruct. Table 18 shows the responses of a cherry-picked sample of 3 questions in the final RL training batch of the LLM but representative of the scenarios on how the LLM may respond when selective confidence reward hacking is present. Note that the exact question is irrelevant because the main focus is to show how the LLM chooses not to provide an educated guess to maximize its chances of getting the correct answer.

The LLM chose to answer Example Question A as it was moderately confident of the correct answer. The confidence values have largely converged to 70, showing a high degree of internal consistency of its confidence.

The LLM has selectively given up on answering Example Question B in 6 out of the 8 responses. This shows that the decision to be confident about an incorrect answer may not be deterministic.

In the third example question, the LLM has decided to give up on Example Question C as there was insufficient information to answer the question. This occurs in HotpotQA-Modified as relevant sources can be removed during dataset preprocessing.

## H Results for other LLM

In addition to Qwen 2.5 (3B) Instruct, we ran our experiments on Llama 3.2 (3B) Instruct and Llama 3.1 (8B) Instruct [42] on the BigMath dataset.

## H.1 Llama 3.2 (3B) Instruct

For Llama 3.2 (3B) Instruct, we ran RL on BigMath on 1000 steps and the learning rate lowered to 10<sup>−6</sup>. The other hyperparameters are the same as in Qwen 2.5 (3B) Instruct. We tested the LLM on Brier-1, Log-1, Brier-2 and Brier-log Hybrid reward schemes, of which only Brier-1 and Brier-log Hybrid are non-hackable confidence. The training accuracies and the training losses for each experiment are as shown in Figure 6.

Dr GRPO training collapsed, as evidenced by the sudden spike in training loss in Figure 6. As this empirically holds independent of the reward scheme used, it is likely a limitation of Dr GRPO RL rather than the reward scheme. Notably, Brier-2 was the first and Brier-log Hybrid was the last to experience training collapse.


Figure 6: Training accuracy (left) and training loss (right) statistics for BigMath in Llama 3.2 (3B) Instruct from Step 10 to Step 999 (0-indexed). The training accuracy has been smoothed using an exponential moving average [27, p. 15], initialized at 0.5 with smoothing factor of 0.2. Dotted lines represent hackable confidence reward schemes.


Figure 7: Training accuracy (left) and training loss (right) statistics for BigMath in Llama 3.1 (8B) Instruct from Step 10 to Step 299 (0-indexed). The training accuracy has been smoothed using an exponential moving average [27, p. 15], initialized at 0.5 with smoothing factor of 0.2. Dotted lines represent hackable confidence reward schemes.

## H.2 Llama 3.1 (8B) Instruct

We ran Dr GRPO RL on BigMath with the same hyperparameters as Qwen 2.5 (3B) Instruct except that the maximum gradient norm has been conservatively set to 0.01 to rule out issues regarding too large a gradient. We ran our experiments on Correctness-only, Overconfidence-1000, Overconfidence-1, Brier-1, Underconfidence-1, Brier-log Hybrid, Log-1 and Brier-2 reward schemes. All the reward schemes mentioned are non-hackable confidence except Log-1 and Brier-2. Figure 7 shows the results of our experiments.

Likewise, the model initially trains well before a sharp decrease to training accuracy, likely due to conflicting signals in the Dr GRPO loss. This occurred regardless of the reward scheme used. The peaks in the training loss occur during the significant decrease in training accuracy.

Similarly, Brier-log Hybrid was the last to suffer the training collapse. However, unlike the experiment in Llama 3.2 (3B) Instruct, the training collapse occurs when the LLM has reached approximately the same reasoning accuracy apart from the hackable confidence reward scheme Log-1, which suggests a learning problem. A more detailed investigation on the phenomenon of training collapse during confidence calibration of RL will be left to future research.

As Brier-log Hybrid is the furthest along the underconfident end of the spectrum, based on the results in Figure 5 of Section 6.2, Brier-log Hybrid can be expected to take the longest time to train to the approximate accuracy level threshold, after which, RL training collapse would occur. Indeed, as shown in Figure 7, Brier-log Hybrid takes the longest time to do so among the examined non-hackable confidence reward schemes.

## I Experimental Anomalies

This section highlights experimental anomalies that researchers may face while attempting to reproduce this work. Currently, only one has been identified, i.e. the confidence may converge to approximately 0.1 while performing RL training on confidence calibration.

## I.1 Confidence Converges to Approximately 0.1

As shown in Figure 8, while training Qwen 2.5 (3B) Instruct on the HotpotQA-Modified dataset using RL on the reward schemes Correctness-only, Brier-1 and Overconfidence-1, the average LLM training confidence appeared to converge to approximately 0.1 mid-run, and did not significantly change for hundreds of steps despite improving training accuracy, suggesting poor confidence calibration due to the LLM getting stuck in a suboptimal local maximum. These runs were not taken for evaluation in the main paper.

We reran the anomalous RL experiments starting from the same fine-tuned model at the supervised fine-tuning stage. Upon rerun, the anomalies mostly did not resurface, and we took these runs for the evaluation in the main paper. As the problem was not reproducible across reruns, this was likely due to the stochasticity of the RL algorithm and issues with the reliability of the trust region in RL.

Figure 8: Training accuracy (left) and training confidence (right) statistics for both anomalous runs and their corresponding reruns during RL training Qwen 2.5 (3B) Instruct of the HotpotQA-Modified dataset from Step 10 to Step 499 (0-indexed), smoothed using an exponential moving average [27, p. 15], both initialized at 0.5 with smoothing factor of 0.2. Dotted lines represent the anomalous runs while solid lines of the same color represent the corresponding reruns.

The LLM has often shown high confidence on its confidence analysis despite providing a confidence of 10 on the scale from 0 to 100. Therefore, the LLM has likely misinterpreted the instruction to output the confidence on a scale from 0 to 100 as a scale from 0 to 10 instead, and the misinterpretation was likely reinforced due to the lower confidence of 10 resulting in a higher expected reward. To discourage this from happening, there should be checks to ensure that the confidence analysis correctly justifies the confidence value provided. We leave the checking to future work.

In the example below, we provide one question and one corresponding LLM response in the final training step of the anomalous experimental run when RL training on the HotpotQA-Modified dataset and using the Brier-1 reward scheme. The response demonstrates LLM confidence in its answer despite indicating a low confidence of 10 on the scale of 0 to 100, as instructed in the system prompt. The system prompt used is the XML system prompt format in Section E.8.

## Question (excluding system prompt)

Based on the information sources given below and your existing knowledge, answer the following question: The Holy Trinity, an episode of British motoring series "The Grand Tour" was produced by which independent television production company founded in 2015 in the UK?

Source 1: James May

James Daniel May (born 16 January 1963) is an English television presenter and journalist. He is best known as a co-presenter of the motoring programme "Top Gear" alongside Jeremy Clarkson and Richard Hammond from 2003 until 2015. As of 2016 he is a director of the production company W. Chump & Sons (founded July 2015) and is also a co-presenter in the television series "The Grand Tour" for Amazon Video as an exclusive for Amazon Prime customers, alongside his former "Top Gear" colleagues, Clarkson and Hammond, as well as former producer Andy Wilman.

## Source 2: G-Unit Films and Television Inc.

G-Unit Films and Television Inc. is an American film and television production company founded by rapper 50 Cent and Interscope in 2003. In 2008, 50 Cent stated in an interview that he has created his own independent film production company with Randall Emmett called Cheetah Vision, technically scrapping G-Unit Films. In 2010, Jackson revived G-Unit Films, renaming the company to G-Unit Films and Television Inc. The company has joint ventures with Will Packer’s production company Will Packer Productions and Universal Television. In over 18 months, Jackson has sold projects to six different networks. Among them was "Power", a STARZ drama in which he not only co-stars but also serves as co-creator and executive producer. “Power” debuted in June 2014 and was renewed for a second season after one episode. “Power’s” August 2 season finale garnered the high ratings through the season, more than doubling the premiere and it generated 71% of the African-American viewership of any scripted premium series since 2006. Jackson serves as a co-star, co-creator and executive television producer of the STARZ network drama where he signed a 2 year contract with representation coming from the Agency for the Performing Arts. Ratings have been a success for Starz. with the second season premiere being the highest-ever season with 1.43 million people tuning in live. Jackson also serves as an executive television producer for "Dream School" for SundanceTV, a series that follows fifteen high school dropouts as they are taught by a series of celebrity "teachers".

## Source 3: WestWind Pictures

WestWind Pictures Ltd. is an independent television production company founded in 1989 in Regina, Saskatchewan. The company, now based in Toronto, Ontario, is co-owned by CEO Mary Darling and President Clark Donnelly. WestWind currently has programs airing in over 80 countries around the world. The company is currently involved in scripted television series, documentaries, lifestyle programming and feature films.

## Source 4: W. Chump and Sons

W. Chump & Sons Limited is an independent television production company founded in July 2015 in the United Kingdom. The company is headquartered on Power Road , Chiswick, London, and owned by Jeremy Clarkson, Richard Hammond, James May and Andy Wilman. The directors of the company are the former presenters and executive producer of the MPH Show and "Top Gear" Live exhibition tours, as well as the former "Top Gear" series produced for BBC Television from October 2002 until March 2015. s of 2016 the company was engaged in producing "The Grand Tour," a motoring television series that debuted on Amazon Video on 18 November 2016.

## Source 5: The Automobile Association

AA plc (The AA, originally The Automobile Association) is a British motoring association founded in 1905, which currently provides car insurance, driving lessons, breakdown cover, loans, motoring advice, road maps and other services. The association demutualised in 1999, to become a private limited company, and in 2002, the AA Motoring Trust was created to continue its public interest and road safety activities. The company is listed on the London Stock Exchange, and is a constituent of the FTSE 250 Index.

## Source 6: Kapital Entertainment

Kapital Entertainment is an American independent television production company founded by Aaron Kaplan in 2009. Prior to founding Kapital, Kaplan was the worldwide head of scripted television at William Morris Agency to which he joined in 1991. He exited the company in 2009 after 18 years when WMA was going through the regulatory process to finalize their merger with Endeavor to form WME Entertainment.

## Source 7: The Grand Tour (TV series)

The Grand Tour is a British motoring television series for Amazon Video presented by Jeremy Clarkson, Richard Hammond and James May and produced by Andy Wilman.

<table><tr><td>Source 8: Desilu Productions</td></tr><tr><td>Desilu Productions () was an American production company founded and co-owned by husband and wife Desi Arnaz and Lucille Ball, best known for shows such as &quot;I Love Lucy&quot;, &quot;&quot;, and &quot;The Untouchables&quot;. Until 1962, Desilu was the second-largest independent television production company in the U.S. behind MCA&#x27;s Revue Productions until MCA bought Universal Pictures, and Desilu became and remained the number-one independent production company until being sold in 1967. Ball and Arnaz jointly owned the majority stake in Desilu from its inception until 1962, when Ball bought out Arnaz and ran the company by herself for several years. Ball had succeeded in making Desilu profitable again by 1967, when she sold her shares of Desilu to Gulf+Western for $17 million ($ in 2016 dollars). Gulf+Western then transformed Desilu into the television production arm of Paramount Pictures, rebranding the company as the original Paramount Television.</td></tr><tr><td>Response</td></tr><tr><td>&lt;reasoning&gt;</td></tr><tr><td>The question asks for the independent television production company founded in 2015 in the UK producing The Holy Trinity, an episode of the British motoring series &#x27;The Grand Tour&#x27;. &#x27;The Grand Tour&#x27; is produced by W. Chump and Sons, which is stated to be an independent television production company founded in July 2015 in the United Kingdom.</td></tr><tr><td>&lt;/reasoning&gt;</td></tr><tr><td>&lt;answer&gt;</td></tr><tr><td>W. Chump and Sons</td></tr><tr><td>&lt;/answer&gt;</td></tr><tr><td>&lt;confidence_analysis&gt;</td></tr><tr><td>The answer is directly derived from Source 4, which states that W. Chump &amp; Sons Limited is an independent television production company founded in July 2015 in the United Kingdom and is the producer of &#x27;The Grand Tour&#x27;. All other sources either do not mention &#x27;The Grand Tour&#x27; or are unrelated to the production company question.</td></tr><tr><td>&lt;/confidence_analysis&gt;</td></tr><tr><td>&lt;confidence&gt;</td></tr><tr><td>10</td></tr><tr><td>&lt;/confidence&gt;</td></tr></table>
