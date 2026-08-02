# An Imperfect Verifier is Good Enough: Learning with Noisy Rewards

Andreas Plesner<sup>1,2</sup>, Francisco Guzmán<sup>1</sup> and Anish Athalye<sup>1</sup>

<sup>1</sup>Handshake AI, <sup>2</sup>ETH Zurich

Reinforcement Learning with Verifiable Rewards (RLVR) has become a prominent method for post-training Large Language Models (LLMs). However, verifiers are rarely error-free; even deterministic checks can be inaccurate, and the growing dependence on model-based judges exacerbates the issue. The extent to which RLVR is robust to such noise and the verifier accuracy required for efective training remain unresolved questions. We investigate these questions in the domains of code generation and scientific reasoning by introducing noise into RL training. Noise rates up to 15% yield peak validation accuracy within 2 percentage points of the clean baseline. These findings are consistent across controlled and model-based noise types, three model families (Qwen3, GLM4, Llama 3.1), and model sizes from 4B to 9B. Overall, the results indicate that imperfect verification does not constitute a fundamental barrier to RLVR. Furthermore, our findings suggest that practitioners should prioritize moderate accuracy with high precision over perfect verification.

## 1. Introduction

Reinforcement learning (RL) has grown in popularity as a post-training method to improve large language models (LLMs) in various domains, particularly after the release of DeepSeek-R1 (DeepSeek-AI, 2025), which demonstrated that Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) (Shao et al., 2024) can produce a frontier-level model at relatively low cost. Early eforts focused on verifiable domains such as math and coding, where deterministic checks provide the reward signal. Since then, RLVR has been extended to semi-verifiable domains such as finance and law, where rubrics and LLM-as-a-Judge provide the reward signal (Viswanathan et al., 2025; Zhou et al., 2025).

RLVR is motivated by the ideal of a perfect verifier: a deterministic oracle that consistently rewards good outputs and penalizes bad ones. In practice, though, such a verifier does not exist. While some datasets, such as GPQA (answers are multiple-choice) and AIME (answers are integers between 000 and 999), admit highly accurate verifiers, verification can be challenging and error-prone even in ostensibly verifiable domains such as mathematics. For example, a string-equality-based verifier might incorrectly reject $\textstyle { \frac { 1 } { e ^ { 2 } } } - { \frac { 1 } { 6 } }$ as an answer with respect to a ground truth $\frac { 6 - e ^ { 2 } } { 6 e ^ { 2 } }$ , despite mathematical equivalence (Xu et al., 2025b; Huang et al., 2025). This problem is exacerbated in semi-verifiable domains where non-deterministic model-based judges are used (Tan et al., 2025; Pan et al., 2026).

While the accuracy of verifiers has been improved through a myriad of eforts such as domain-specific techniques (e.g., as in Hugging Face (2025b)’s Math-Verify library) and finetuned judges (Zhu et al., 2025a; He et al., 2025), fundamental questions remain unanswered: what accuracy does RLVR actually require of its verifier, and is there a point where the verifier is “good enough”? Despite their importance, these questions remain poorly understood, even as the field shifts toward LLM-as-a-Judge and Agent-as-a-Judge (Vidgen et al., 2026; Zhuge et al., 2025), where measurement error is amplified.

To study these questions in a controlled setting, we focus on RLVR for coding, and we measure the impact of verifier noise on model training. We focus on coding for two reasons: because the reward structure mirrors rubric-based rewards (a set of unit tests is analogous to a set of text-based rubric criteria), and because the verifiers can be highly accurate, providing a noise-free baseline as a point of comparison for controlled and realistic noise.

Our experiments reveal that RLVR is robust to imperfect verifiers: noise rates up to 15% produce no significant drop in peak validation performance. The results hold across noise types—controlled noise patterns as well as realistic noise—and the results generalize across domains to scientific reasoning. Our results also show that not all errors are equal: precision matters more than recall. But engineering efort improving a verifier beyond a certain point has diminishing returns: an imperfect verifier is good enough.

## 2. Related work

Reinforcement learning with verifiable rewards. Silver & Sutton (2025) articulated a vision for scaling RL to language agents, crystallizing an emerging trend of increasing model abilities through RL (OpenAI, 2024; Zeng et al., 2025). A key example of this is RLVR, which was popularized by DeepSeek-R1 (DeepSeek-AI, 2025). They showed that GRPO with outcome-based rewards on math and code tasks can produce strong reasoning LLMs while having less risk of reward hacking (Hutter, 2005). Verification can also be applied at the process level rather than just the outcome to provide stronger learning signals (Lightman et al., 2023; Hübotter et al., 2026). Our work takes RLVR as a given (see Appendix D or DeepSeek-AI (2025) for background) and investigates what happens when the outcome-based reward signal is imperfect.

LLM-as-a-Judge / model-based verifiers. Using models as evaluators has become standard practice for evaluation and training (Bai et al., 2022; Gu et al., 2026). Zheng et al. (2023) introduced the LLM-as-a-Judge paradigm for evaluation, and Lambert et al. (2025) created a benchmark to rank judges’ alignment with human preferences. Liu et al. (2026) show that reasoning models are better verifiers than non-reasoning models. For training, rubricbased approaches use model-based verifiers to provide reward signals in domains without deterministic verifiers (Gunjal et al., 2025; He et al., 2025).

Model-based verifiers have well-documented failure modes, including positional bias (Thakur et al., 2025) and sensitivity to prompt phrasing (Chen et al., 2024; Shankar et al., 2024), meaning that model-based verifier noise is neither uniform nor independent.

Reinforcement learning with noisy rewards. The problem oflearning under corrupted/noisy rewards is not new. Everitt et al. (2017) formalize the corrupted reward channel as a POMDP (Kaelbling et al., 1998) and identify conditions under which optimal behavior is still recoverable. Wang et al. (2020) propose estimators robust to reward perturbation in deep RL. In the preference-based setting, Gong et al. (2025) study noisy feedback and Li et al. (2026) analyze feature-dependent noise. Separately, Fortunato et al. (2018) add parametric noise directly to network weights to drive exploration.

Noisy rewards in RLVR. Rad et al. (2026) provide a theoretical analysis of reward noise in GRPO and validate their analysis on Qwen2.5 3B for code generation; our work complements and challenges their work with a broader empirical study. In contrast to them, we find that low to medium levels of noise do not negatively impact post-training. Cai et al. (2025) also study noisy RLVR like us, but test a narrower selection of noise types and models; also, their results do not align with other works, so we omit them in later discussions (see Appendix K). Shao et al. (2025) study spurious rewards in RLVR for math, finding that Qwen models can learn from incorrect reward signals. Zhu & Kang (2026) follow up with a rebuttal saying that noise does hurt model performance. Chen et al. (2026) revisit spurious rewards through the lens of exploration vs. exploitation, and Xu et al. (2025a) propose a two-stage entropy-based approach for noise-tolerant RLVR training.

However, these works focus on very aggressive amounts of noise where up to 100% of the labels are incorrect. In contrast, we focus on the robustness threshold and the structure of noise. Mansouri et al. (2025) model reward corruption as Bernoulli noise and derive a correction that yields provably unbiased GRPO gradients; our work is complementary, as we empirically characterize when such corrections are needed.

## 3. Methodology

We begin with a noise-free baseline of RLVR in the coding domain, where we have an error-free verifier. We corroborate this quality of the verifier by running the verifier for ground truth solutions, requiring that they all pass.

Next, we train models using noisy verifiers, evaluating the trained model against the held-out test set using the error-free verifier. For the verifier used during training, we vary noise rates as well as noise type: we study a more controlled setting of specific noise patterns as well as more realistic noise distributions by using a model-based verifier.

## 3.1. Controlled noise for RLVR

In the code-generation setting we study, each prompt has � unit tests, and each training step generates a group of � rollouts per prompt. The reward for a single rollout is the fraction of unit tests passed, so the full reward computation can be represented as a $G \times T$ binary matrix M, where $M _ { i , j } = 1$ if rollout � passes test �.

We inject noise by flipping entries of M with probability $p ,$ independently of whether the original value is pass or fail. This bitflipping means the false positive rate and false negative rate are both equal to �. We identify two orthogonal axes that determine the structure of the noise, yielding four modes (illustrated in Figure 1):

(a) Sample $\times$ unit test. Each cell $M _ { i , j }$ is flipped independently with probability $p .$ This noise is the most fine-grained; every test outcome for every rollout has an independent chance of corruption.

(b) Sample × rollout. Each row (rollout) is selected with probability $p ,$ and all test outcomes in that row are flipped. This noise models a scenario where the verifier completely misclassifies a solution.

(c) Group × unit test. Each column (test index) is selected with probability $p ,$ and that test’s outcome is flipped for all rollouts in the group. This noise models a faulty test case that is consistently wrong across all solutions for a given prompt.

(d) Group × rollout. The entire matrix is flipped with probability $p .$ . When triggered, every test outcome for every rollout is inverted; otherwise, the matrix is left intact. This noise is the coarsest; it either corrupts everything or nothing.

During training, we have multiple epochs over the data, but the noise is sampled each time independently. Thus, the noise in epoch � might difer from the noise in epoch $j \neq i .$ . The noise is always applied symmetrically in terms of the false-negative and false-positive rates. We leave it to future work to address fixed noise and asymmetric noise.

## 3.2. Model-based verifier

While controlled noise allows precise exploration over the error rate and structure of noise, its failure modes are artificial—real verifiers do not flip outcomes uniformly at random. Thus, to study more realistic noise distributions, we replace the unit-test executor with a model-based verifier that predicts whether each unit test would pass given the generated code. Concretely, the verifier receives the generated code and a single test case (formatted as an assert statement) and returns a binary pass/fail judgment (see Appendix C for the full prompt). The per-rollout reward is then the fraction of tests that the verifier marks as passing, matching how the ground-truth reward is computed. In this setting, we vary the noise rate by using models of diferent sizes/strengths.

Figure 1. The four controlled noise injection modes. Rows are rollouts (� –� ), columns are unit tests $\left( t _ { 1 } - t _ { 3 } \right)$ . Red cells indicate flipped outcomes. (a) Each cell flipped independently. (b) Entire rows flipped. (c) Entire columns flipped. (d) Entire matrix flipped.

This setup mirrors how model-based verifiers are used in rubric-based post-training (Gunjal et al., 2025; He et al., 2025), where the model’s output is graded using a rubric by a judge on a per-criterion basis that is aggregated into a final score. The setup has two key advantages for our study. First, the failure modes are realistic—the verifier may struggle with edge cases, subtle bugs, or ambiguous specifications, rather than producing uniform random errors. Second, because the ground-truth unit tests remain available, we can compute the verifier’s accuracy, precision, recall, and F1 score on every training batch. These metrics let us track how verifier quality evolves as the trained model’s output distribution shifts during training.<sup>1</sup> Being able to capture these changing metrics cost-efectively is distinctive to our setting of using model-based verifiers in place of existing deterministic checks; capturing these online metrics for semi-verifiable tasks would require costly human labeling at each evaluation step.

## 4. Experimental setup

Dataset. Our experiments focus on the Mostly Basic Python Problems (MBPP) dataset (Austin et al., 2021). Each problem provides a natural-language description and three unit tests. We train on the standard MBPP training split (374 problems) and evaluate on the validation split (90 problems) using a maximum response length of 8192 tokens. The validation metric is the mean unit-test pass rate across samples.

Models and compute. We use an internal training framework built on SLIME (Zhu et al., 2025b). For our main experiments, we focus on Qwen3 8B (Qwen Team, 2025) and GLM4 9B (Team GLM, 2024), with ablations using Llama 3.1 8B and Qwen3 4B. Training uses Group Relative Policy Optimization (GRPO) (Shao et al., 2024); full hyperparameters are in


Figure 2. Best validation reward across noise levels for group rollout noise. Shaded regions indicate ±1 standard deviation across multiple seeds. We only run multiple seeds for �≤0.10 to save compute (see Section 4).

Appendix B.

Each training run requires approximately 64 GPU-hours on H100 GPUs. Multi-seed experiments are therefore limited to the most important configurations to manage compute costs. Unless otherwise noted, results are from a single seed; key results (baselines and group rollout noise at $p { \le } 0 . 1 0 )$ use 2–3 seeds.

## 5. Results

We include here the key results showing that an imperfect verifier is good enough and that the noise type matters less than the amount of noise. For the former, we focus on the group-level entire-rollout noise structure at diferent rates. For the latter, we focus on all four controlled noise types at $_ { p = 0 . 1 0 }$ , and the model-based verifier.

## 5.1. Imperfect is good enough

We gradually sweep the controlled noise rate � from 0.01 to 0.50 while fixing the noise type to group-level entire-rollout noise. Figure 2 shows that the best validation performance is remarkably robust to verifier errors; performance remains within 2 points of the clean baseline for $\scriptstyle p \leq 0 . 1 5$ , and degrades gracefully up to $\scriptstyle { p = 0 . 3 0 }$ . The drop only becomes severe at $\scriptstyle { p = 0 . 4 0 }$ and above, with $\scriptstyle { p = 0 . 5 0 }$ approaching random noise.

Figure 3 indicates that low to moderate noise can lead to the same or slightly higher final-checkpoint performance on the validation set than the clean baseline. GLM4 9B with group-level entire-rollout noise at $_ { p = 0 . 1 0 }$ achieves a final validation reward of 0.900, compared to 0.905 for the clean baseline. Qwen3 8B shows a similar pattern (0.886 vs 0.901). Overall, we see that Qwen3 8B has a slightly higher final performance when noise is applied with $p { \le } 0 . 2 0$ compared to when there is no noise.

We posit that when there is no noise, the model starts to overfit to the training data, but slight amounts of noise act as a regularizer, which prevents overfitting to the training data. The regularization hypothesis is consistent with the broader observation that RL training generalizes better than supervised fine-tuning (Chu et al., 2025), and reward noise may amplify this efect by further discouraging memorization of the training distribution. In Section 6.2, we validate this hypothesis through additional experimentation.


Figure 3. Final checkpoint validation reward across noise levels for group rollout noise. Shaded regions indicate ±1 standard deviation across seeds. We only run multiple seeds for �≤0.10 to save compute (see Section 4).

## 5.2. The type of noise does not matter

Table 1 shows the final and best validation reward for each noise type. We compare the four controlled noise modes at a fixed noise rate of $_ { p = 0 . 1 0 }$ , a moderate level, and the model-based verifiers, to see how the type of noise impacts the results.

Controlled noise. Interestingly, when comparing diferent types of controlled noise, grouplevel noise (modes c and d) slightly outperforms sample-level noise (modes a and b) across the models. However, we see that the controlled noise performance is comparable to the baseline performance and does not lead to any substantial changes.

Model noise. In contrast, when considering the model-based verifier results, we observe some diferences: with the 30B verifier, we achieve a maximum validation reward of 0.871, compared to the 0.901 of the baseline, while with a less capable model, the 4B verifier, we only reach 0.704. We posit that the key diference is a result of the accuracy and precision of the model, which are approximately 15 percentage points higher for the 30B verifier, as we will discuss in detail in Section 6.3.

We include results for convergence of GLM4 9B and Qwen3 8B in Appendix G, Llama 3.1 8B in Table 1 and appendix J, and Qwen3 4B in Appendix H.

## 6. Discussion

Following the results in the previous section, which show robustness in RLVR to noise introduced by the verifiers, there are natural questions that arise: (1) Do the results hold for other domains? (2) Why can models learn with noisy rewards and show improvements over the baseline? (3) Should model-based verifiers optimize for precision or recall?

Here we revisit those questions in detail and provide more evidence that sheds more light on our findings.

## 6.1. Do the results hold for other domains?

To test other domains, we consider Graduate-Level Google-Proof Q&A (GPQA) (Rein et al., 2023) with graduate-level scientific reasoning (multiple choice) problems. We opt to exclude math, as prior Qwen models have shown peculiar RLVR behavior in that domain (Shao et al., 2025; Zhu & Kang, 2026), and because Xu et al. (2025b) demonstrated that verifying

<table><tr><td>Model</td><td>Setup</td><td>Seeds</td><td>Best</td><td>Final</td><td>Steps-to-best</td></tr><tr><td colspan="6">Base model (before training)</td></tr><tr><td>GLM4 9B</td><td>Base model</td><td>3</td><td>0.609 ± 0.010</td><td>0.609 ± 0.010</td><td>NA</td></tr><tr><td>Qwen3 8B</td><td>Base model</td><td>3</td><td>0.487 ± 0.011</td><td>0.487 ± 0.011</td><td>NA</td></tr><tr><td colspan="6">Baselines (ground-truth unit tests)</td></tr><tr><td>GLM4 9B</td><td>Baseline</td><td>3</td><td>0.905 ± 0.002</td><td>0.884 ± 0.006</td><td>159 ± 34.6</td></tr><tr><td>Qwen3 8B</td><td>Baseline</td><td>2</td><td>0.901 ± 0.009</td><td>0.864 ± 0.007</td><td>179 ± 84.8</td></tr><tr><td>Llama 3.1 8B</td><td>Baseline</td><td>2</td><td>0.658 ± 0.001</td><td>0.505 ± 0.025</td><td>59 ± 28.2</td></tr><tr><td colspan="6">Controlled noise (p=0.10)</td></tr><tr><td rowspan="4">GLM4 9B</td><td>Group rollout</td><td>3</td><td>0.900 ± 0.005</td><td>0.895 ± 0.010</td><td>212 ± 80.8</td></tr><tr><td>Group unit test</td><td>1</td><td>0.891</td><td>0.869</td><td>139</td></tr><tr><td>Sample rollout</td><td>1</td><td>0.866</td><td>0.835</td><td>59</td></tr><tr><td>Unit test</td><td>1</td><td>0.875</td><td>0.864</td><td>119</td></tr><tr><td rowspan="4">Qwen3 8B</td><td>Group rollout</td><td>3</td><td>0.886 ± 0.003</td><td>0.880 ± 0.003</td><td>219 ± 20.0</td></tr><tr><td>Group unit test</td><td>2</td><td>0.893 ± 0.003</td><td>0.881 ± 0.007</td><td>229 ± 14.1</td></tr><tr><td>Sample rollout</td><td>2</td><td>0.864 ± 0.011</td><td>0.863 ± 0.012</td><td>249 ± 14.1</td></tr><tr><td>Unit test</td><td>2</td><td>0.854 ± 0.013</td><td>0.852 ± 0.014</td><td>229 ± 14.1</td></tr><tr><td>Llama 3.1 8B</td><td>Group rollout</td><td>1</td><td>0.651</td><td>0.542</td><td>19</td></tr><tr><td colspan="6">Model-based verifier</td></tr><tr><td rowspan="2">Qwen3 8B</td><td>Verifier: Qwen3 4B</td><td>1</td><td>0.704</td><td>0.652</td><td>39</td></tr><tr><td>Verifier: Qwen3 30B-A3B</td><td>1</td><td>0.871</td><td>0.869</td><td>179</td></tr></table>

Table 1. Overview of baseline and model-based verifier results on MBPP. Best and final refer to the highest and last checkpoint validation reward (mean unit-test pass rate), evaluated against ground-truth unit tests. ± values are standard deviations across seeds when available.
mathematical equivalence is itself dificult for model-based verifiers.

Experimental setup. We remove the diamond problems (198 problems) from the main subset (448 problems before, 250 problems after). We then use the 250 main problems for training and the 198 diamond problems for validation. We train Qwen3 8B with group rollout noise at $\scriptstyle { p = 0 . 0 5 }$ and $\scriptstyle { p = 0 . 3 0 }$ along with a noise-free baseline. We use the same setup as for MBPP, except we use Group Sequence Policy Optimization (GSPO) (Zheng et al., 2025) instead of GRPO for this task to stabilize training convergence in the sparse-reward setting (see Appendix D for details).

Results. Table 2 shows the results. Post-training improves the base model from 0.540 to 0.600 without noise. With group rollout noise at $\scriptstyle { p = 0 . 0 5 }$ , performance is 0.604, and at $\scriptstyle { p = 0 . 3 0 }$ it is 0.603—slightly exceeding the clean baseline. These findings are consistent with our MBPP findings that GRPO-style algorithms are robust to group-level noise, and extend the result to a fundamentally diferent task type with a simpler (binary) reward signal.

<table><tr><td>Setup</td><td>Best</td></tr><tr><td>Base model</td><td>0.540</td></tr><tr><td>No noise</td><td>0.600</td></tr><tr><td>Noise p=0.05</td><td>0.604</td></tr><tr><td>Noise p=0.30</td><td>0.603</td></tr></table>

Table 2. Best validation reward on GPQA Diamond for Qwen3 8B trained with GSPO under group rollout noise. Both noisy settings match the clean baseline.

The fact that $\scriptstyle { p = 0 . 3 0 }$ causes no degradation on GPQA is surprising, but aligns with how prior work has found that Qwen models are robust to noise in the math domain (Shao et al., 2025). The lack of degradation opens the possibility that binary rewards might be more robust to noise than unit test-based rewards. However, we do not have enough data to make any solid conclusions. We leave it to future work to explore this.

Observation. The above shows that the results generalize to scientific reasoning, and that the verifier used for post-training with RLVR does not have to be perfect. It can handle at least 15% noise rates in the tested domains.

## 6.2. Why can models learn with noisy rewards?

We have seen that low to moderate noise levels may not impact the post-training results negatively, which may be surprising. We discuss below why the group-level entire-rollout noise can have positive efects. Then we use the Ackley function to demonstrate the potentially beneficial efects of noise during optimization.

Controlled noise. Group-level entire-rollout noise, when triggered, inverts the ranking of advantages within the GRPO group and thus flips the direction of the policy gradient. At first glance, a gradient step in the opposite direction of the intended minimization should be harmful. However, this efect has a natural interpretation through the lens of loss landscape geometry: when the policy is near a sharp minimum, an inverted gradient step moves the policy away from that basin, functioning as an implicit escape mechanism. We include in Appendix E.1 a detailed discussion of prior works highlighting why noise can be beneficial.

In contrast to the group-level entire-rollout noise, sample-level noise distorts the relative ranking of rollouts within the group, corrupting the advantage estimates in a way that neither preserves the original gradient direction nor cleanly inverts it. This corruption results in a random perturbation that lacks the structured exploration benefit of a full inversion. We corroborate this local-minima-escape hypothesis with a controlled toy experiment below, where we show that reward noise helps GRPO escape local minima on the Ackley function.

The Ackley function. We hypothesize that reward noise acts as a regularizer, preventing the policy from overfitting to the training distribution, and to better understand this, we turn to a classic optimization theory problem. The advantages of such a problem are that the landscape and local curvature are well understood, and the function only has two inputs, so we can easily visualize the space; thus, we can test the hypothesis in a controlled setting. We construct a toy experiment using a simplified version of $\scriptstyle { \mathrm { \dot { G } R P O } ^ { 2 } }$ to optimize the Ackley function. We initialize the optimization at 3 random points on a circle around the global minimum and run the optimization method for 500 steps.

Figure 4 shows the optimization trajectories. Without noise $( \sigma _ { \mathrm { n o i s e } } { = } 0 )$ , the optimizer gets trapped in minima near the starting points—the Ackley landscape has many minima that the exploration cannot escape. With moderate noise $( \sigma _ { \mathrm { n o i s e } } { = } 2 . 0 )$ , the noisy reward signal allows the optimizer to escape local minima and make progress toward the global minimum. At high noise $( \sigma _ { \mathrm { n o i s e } } { = } 1 0 . 0 )$ , the optimization quality degrades, though trajectories still explore more than the noiseless case.

This toy experiment provides intuition for why moderate noise can be beneficial in the LLM training setting. When the reward landscape has local optima (e.g., the model learns a specific coding pattern that passes training tests but generalizes poorly), noise in the reward signal can prevent the policy from committing too strongly to these suboptimal solutions.

Observation. Noise can have beneficial attributes, and while it must be suficiently accurate, it need not be perfect. One can even theoretically expect benefits with slight noise.



Figure 4. Optimization trajectories on the Ackley function from 3 random starting points. Without noise, the optimizer gets trapped in local minima. Moderate noise (�=2.0) helps escape local basins, while excessive noise (�=10.0) degrades optimization quality.



Figure 5. Training Qwen3 8B with model-based verifier rewards on MBPP. Left: validation reward against ground-truth unit tests (solid) and raw rollout reward (dashed). The 30B verifier recovers most of the ground-truth signal (0.871 vs. 0.901 without noise), while the 4B verifier peaks at 0.704. Center and right: verifier precision and recall throughout training. Both verifiers maintain high recall (> 90%), but the 4B verifier has substantially lower precision, frequently marking incorrect code as passing.

## 6.3. Should verifiers optimize for precision or recall?

Figure 5 shows metrics from training on MBPP<sup>3</sup> with the model-based verifiers. As noted earlier, we see that training stagnates quickly when using the 4B verifier, while the 30B verifier is able to get final performance comparable with training with the noise-free verifier. From the figure, we also see that recall remains high for both (> 90%, with the 4B verifier exhibiting higher recall), indicating both reliably identify correct solutions; the bottleneck is precision—the 4B verifier frequently assigns passing scores to incorrect code, introducing false-positive noise into the training signal. This diference in precision and post-training outcome would imply that the false positives are more harmful than the false negatives.

One intuitive understanding of why this is the case is the exploration vs. exploitation trade-of that is often found in reinforcement learning. When there are many false positives, the model learns to exploit these bad solutions. We can see this in Figure 5 on the left panel.

The train reward with the 4B verifier is much higher than the reward when using the stronger 30B verifier. Meanwhile, false negatives force the model to try many diferent ways to solve the problem. Many problems have many equally correct solutions, so this forced exploration is not a problem. Overall, these efects connect to the broader reward overoptimization phenomenon (Gao et al., 2023).

These findings contrast with Xu et al. (2025b), who state that “our findings highlight the critical importance of addressing verifier false negatives.” However, their analysis focused on mathematics, where verifiers often fail to recognize equivalent statements as correct, and they did not examine false-positive rates. We discuss this in detail in Appendix E.2.

Observation. When implementing a verifier, it is important to increase the precision, even at the cost of decreasing the recall.

## 7. Conclusion

We have presented a systematic study of reward noise in RLVR across three model families (GLM4, Qwen3, and Llama 3.1), multiple model sizes (4B–9B), multiple noise types (controlled and model-based), and two task domains (coding and scientific reasoning). Our main finding is that a verifier need not be perfect: a model-based verifier with ≈85% accuracy and precision recovers most of the ground-truth training signal, and RLVR tolerates up to 15% controlled noise with no significant performance loss. These findings lower the bar for extending RLVR to semi-verifiable domains: practitioners can target moderate verifier accuracy with high precision rather than pursuing perfect verification.

Limitations. l h d l f l b f d GLM4 9B; whether the robustness thresholds generalize to larger models, other architectures, or domains beyond coding and scientific reasoning remains open. The controlled noise is symmetric (equal FPR and FNR) and resampled for each epoch. Real verifiers have asymmetric errors and persistent biases, as evident in our model-based verifier experiments. A systematic controlled study varying FPR and FNR independently, including fixed noise, would strengthen our conclusions about the importance of precision over recall.

## References

Austin, Jacob; Augustus Odena; Maxwell Nye; Maarten Bosma; Henryk Michalewski; David Dohan; Ellen Jiang; Carrie Cai; Michael Terry; Quoc Le & Charles Sutton (Aug. 2021). Program Synthesis with Large Language Models. doi: 10.48550/arXiv.2108.07732. arXiv: 2108.07732 [cs].

Bai, Yuntao; Saurav Kadavath; Sandipan Kundu; Amanda Askell; Jackson Kernion; Andy Jones; Anna Chen; Anna Goldie; Azalia Mirhoseini; Cameron McKinnon; Carol Chen; Catherine Olsson; Christopher Olah; Danny Hernandez; Dawn Drain; Deep Ganguli; Dustin Li; Eli Tran-Johnson; Ethan Perez; Jamie Kerr; Jared Mueller; Jefrey Ladish; Joshua Landau; Kamal Ndousse; Kamile Lukosuite; Liane Lovitt; Michael Sellitto; Nelson Elhage; Nicholas Schiefer; Noemi Mercado; Nova DasSarma; Robert Lasenby; Robin Larson; Sam Ringer; Scott Johnston; Shauna Kravec; Sheer El Showk; Stanislav Fort; Tamera Lanham; Timothy Telleen-Lawton; Tom Conerly; Tom Henighan; Tristan Hume; Samuel R. Bowman; Zac Hatfield-Dodds; Ben Mann; Dario Amodei; Nicholas Joseph; Sam McCandlish; Tom Brown & Jared Kaplan (Dec. 2022). Constitutional AI: Harmlessness from AI Feedback. doi: 10.48550/arXiv.2212.08073. arXiv: 2212.08073 [cs].

Cai, Xin-Qiang; Wei Wang; Feng Liu; Tongliang Liu; Gang Niu & Masashi Sugiyama (Dec. 2025). Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers. doi: 10.48550/arXiv.2510.00915. arXiv: 2510.00915 [cs].

Chaudhari, Pratik; Anna Choromanska; Stefano Soatto; Yann LeCun; Carlo Baldassi; Christian Borgs; Jennifer Chayes; Levent Sagun & Riccardo Zecchina (Feb. 2017). “Entropy-SGD: Biasing Gradient Descent Into Wide Valleys”. In: International Conference on Learning Representations.

Chen, Guiming Hardy; Shunian Chen; Ziche Liu; Feng Jiang & Benyou Wang (Nov. 2024). “Humans or LLMs as the Judge? A Study on Judgement Bias”. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Ed. by Yaser Al-Onaizan; Mohit Bansal & Yun-Nung Chen. Miami, Florida, USA: Association for Computational Linguistics, pp. 8301–8327. doi: 10.18653/v1/2024.emnlp-main.474.

Chen, Peter; Xiaopeng Li; Ziniu Li; Wotao Yin; Xi Chen & Tianyi Lin (Jan. 2026). Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward. doi: 10.48550/arXiv.2512.16912. arXiv: 2512.16912 [cs].

Chu, Tianzhe; Yuexiang Zhai; Jihan Yang; Shengbang Tong; Saining Xie; Dale Schuurmans; Quoc V. Le; Sergey Levine & Yi Ma (May 2025). SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. doi: 10.48550/arXiv.2501.17161. arXiv: 2501.17161 [cs].

Cui, Ganqu; Lifan Yuan; Zefan Wang; Hanbin Wang; Yuchen Zhang; Jiacheng Chen; Wendi Li; Bingxiang He; Yuchen Fan; Tianyu Yu; Qixin Xu; Weize Chen; Jiarui Yuan; Huayu Chen; Kaiyan Zhang; Xingtai Lv; Shuo Wang; Yuan Yao; Xu Han; Hao Peng; Yu Cheng; Zhiyuan Liu; Maosong Sun; Bowen Zhou & Ning Ding (2025). Process Reinforcement through Implicit Rewards. doi: 10.48550/ARXIV.2502.01456.

DeepSeek-AI et al. (Sept. 2025). “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”. In: Nature 645.8081, pp. 633–638. issn: 0028-0836, 1476-4687. doi: 10.1038/s41586-025-09422-z. arXiv: 2501.12948 [cs].

Dziugaite, Gintare Karolina & Daniel M. Roy (Oct. 2017). Computing Nonvacuous Generalization Boundsfor Deep (Stochastic) Neural Networks with Many More Parameters than Training Data. doi: 10.48550/arXiv.1703.11008. arXiv: 1703.11008 [cs].

Everitt, Tom; Victoria Krakovna; Laurent Orseau & Shane Legg (Aug. 2017). “Reinforcement Learning with a Corrupted Reward Channel”. In: Proceedings of the 26th International Joint Conference on Artificial Intelligence. IJCAI’17. Melbourne, Australia: AAAI Press, pp. 4705– 4713. isbn: 978-0-9992411-0-3.

Foret, Pierre; Ariel Kleiner; Hossein Mobahi & Behnam Neyshabur (Oct. 2020). “Sharpness-Aware Minimization for Eficiently Improving Generalization”. In: International Conference on Learning Representations.

Fortunato, Meire; Mohammad Gheshlaghi Azar; Bilal Piot; Jacob Menick; Matteo Hessel; Ian Osband; Alex Graves; Volodymyr Mnih; Remi Munos; Demis Hassabis; Olivier Pietquin; Charles Blundell & Shane Legg (Feb. 2018). “Noisy Networks For Exploration”. In: International Conference on Learning Representations.

Gao, Leo; John Schulman & Jacob Hilton (July 2023). “Scaling Laws for Reward Model Overoptimization”. In: Proceedings of the 40th International Conference on Machine Learning. PMLR, pp. 10835–10866.

Gong, Yuhao; Zhenbo Lu; Wanxuan Lu; Wengang Zhou & Houqiang Li (May 2025). “Adaptive Confidence-aware Preference-based Reinforcement Learning with Noisy Feedback”. In: Companion Proceedings of the ACM on Web Conference 2025. WWW ’25. New York, NY, USA: Association for Computing Machinery, pp. 2120–2128. isbn: 979-8-4007-1331-6. doi: 10.1145/3701716.3717570.

Gu, Jiawei; Xuhui Jiang; Zhichao Shi; Hexiang Tan; Xuehao Zhai; Chengjin Xu; Wei Li; Yinghan Shen; Shengjie Ma; Honghao Liu; Saizhuo Wang; Kun Zhang; Zhouchi Lin; Bowen Zhang; Lionel Ni; Wen Gao; Yuanzhuo Wang & Jian Guo (Jan. 2026). “A Survey on LLM-as-a-Judge”. In: The Innovation, p. 101253. issn: 26666758. doi: 10.1016/j.xinn.2025.101253.

Gunjal, Anisha; Anthony Wang; Elaine Lau; Vaskar Nath; Yunzhong He; Bing Liu & Sean Hendryx (Oct. 2025). Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. doi: 10.48550/arXiv.2507.17746. arXiv: 2507.17746 [cs].

He, Chaoqun; Renjie Luo; Yuzhuo Bai; Shengding Hu; Zhen Thai; Junhao Shen; Jinyi Hu; Xu Han; Yujie Huang; Yuxiang Zhang; Jie Liu; Lei Qi; Zhiyuan Liu & Maosong Sun (Aug. 2024). “OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems”. In: Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers). Ed. by Lun-Wei Ku; Andre Martins & Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, pp. 3828–3850. doi: 10.18653/v1/2024.acl-long.211.

He, Yun; Wenzhe Li; Hejia Zhang; Songlin Li; Karishma Mandyam; Sopan Khosla; Yuanhao Xiong; Nanshu Wang; Xiaoliang Peng; Beibin Li; Shengjie Bi; Shishir G. Patil; Qi Qi; Shengyu Feng; Julian Katz-Samuels; Richard Yuanzhe Pang; Sujan Gonugondla; Hunter Lang; Yue Yu; Yundi Qian; Maryam Fazel-Zarandi; Licheng Yu; Amine Benhalloum; Hany Awadalla & Manaal Faruqui (Nov. 2025). AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learningfor Advancing LLM Instruction Following. doi: 10.48550/arXiv.2511.10507. arXiv: 2511.10507 [cs].

Hendrycks, Dan; Collin Burns; Saurav Kadavath; Akul Arora; Steven Basart; Eric Tang; Dawn Song & Jacob Steinhardt (Nov. 2021). Measuring Mathematical Problem Solving With the MATH Dataset. doi: 10.48550/arXiv.2103.03874. arXiv: 2103.03874 [cs].

Hochreiter, Sepp & Jürgen Schmidhuber (Jan. 1997). “Flat Minima”. In: Neural Computation 9.1, pp. 1–42. issn: 0899-7667. doi: 10.1162/neco.1997.9.1.1.

Huang, Yuzhen; Weihao Zeng; Xingshan Zeng; Qi Zhu & Junxian He (Oct. 2025). From Accuracy to Robustness: A Study ofRule- and Model-based Verifiers in Mathematical Reasoning. arXiv:2505.22203v2 [cs.LG]. url: https://arxiv.org/abs/2505.22203v2.

Hübotter, Jonas; Frederike Lübeck; Lejs Behric; Anton Baumann; Marco Bagatella; Daniel Marta; Ido Hakimi; Idan Shenfeld; Thomas Kleine Buening; Carlos Guestrin & Andreas Krause (Jan. 2026). Reinforcement Learning via Self-Distillation. doi: 10.48550/arXiv. 2601.20802. arXiv: 2601.20802 [cs].

Hugging Face (2025a). Chat templates. https://huggingface.co/docs/transformers/ en/chat\_templating. Accessed: 2026-03-24.

Hugging Face (Jan. 2025b). Math-Verify. url: https://github.com/huggingface/Math-Verify.

Hutter, Marcus (2005). Universal Artificial Intelligence: Sequential Decisions based on Algorithmic Probability. Berlin: Springer, 300 pages. isbn: 3-540-22139-5. doi: 10.1007/b138233. url: http://www.hutter1.net/ai/uaibook.htm.

Jin, Chi; Rong Ge; Praneeth Netrapalli; Sham M. Kakade & Michael I. Jordan (July 2017). “How to Escape Saddle Points Eficiently”. In: Proceedings of the 34th International Conference on Machine Learning. PMLR, pp. 1724–1732.

Kaelbling, Leslie Pack; Michael L. Littman & Anthony R. Cassandra (May 1998). “Planning and Acting in Partially Observable Stochastic Domains”. In: Artificial Intelligence 101.1, pp. 99–134. issn: 0004-3702. doi: 10.1016/S0004-3702(98)00023-X.

Keskar, Nitish Shirish; Dheevatsa Mudigere; Jorge Nocedal; Mikhail Smelyanskiy & Ping Tak Peter Tang (Feb. 2017). “On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima”. In: International Conference on Learning Representations.

Kingma, Diederik P. & Jimmy Ba (2015). “Adam: A Method for Stochastic Optimization”. In: International Conference on Learning Representations (ICLR). url: https://arxiv.org/ abs/1412.6980.

Kleinberg, Bobby; Yuanzhi Li & Yang Yuan (July 2018). “An Alternative View: When Does SGD Escape Local Minima?” In: Proceedings of the 35th International Conference on Machine Learning. PMLR, pp. 2698–2707.

Lambert, Nathan; Valentina Pyatkin; Jacob Morrison; LJ Miranda; Bill Yuchen Lin; Khyathi Chandu; Nouha Dziri; Sachin Kumar; Tom Zick; Yejin Choi; Noah A. Smith & Hannaneh Hajishirzi (Apr. 2025). “RewardBench: Evaluating Reward Models for Language Modeling”. In: Findings of the Association for Computational Linguistics: NAACL 2025. Ed. by Luis Chiruzzo; Alan Ritter & Lu Wang. Albuquerque, New Mexico: Association for Computational Linguistics, pp. 1755–1797. isbn: 979-8-89176-195-7. doi: 10 . 18653 / v1 / 2025 . findings-naacl.96.

Le, Samuel L. Smith and Quoc V. (Feb. 2018). “A Bayesian Perspective on Generalization and Stochastic Gradient Descent”. In: International Conference on Learning Representations.

Lewkowycz, Aitor; Anders Andreassen; David Dohan; Ethan Dyer; Henryk Michalewski; Vinay Ramasesh; Ambrose Slone; Cem Anil; Imanol Schlag; Theo Gutman-Solo; Yuhuai Wu; Behnam Neyshabur; Guy Gur-Ari & Vedant Misra (2022). “Solving Quantitative Reasoning Problems with Language Models”. In: Advances in Neural Information Processing Systems. Ed. by S. Koyejo; S. Mohamed; A. Agarwal; D. Belgrave; K. Cho & A. Oh. Vol. 35. Curran Associates, Inc., pp. 3843–3857.

Li, Yuxuan; Harshith Reddy Kethireddy & Srijita Das (Jan. 2026). Evaluating Feature Dependent Noise in Preference-based Reinforcement Learning. doi: 10.48550/arXiv.2601.01904. arXiv: 2601.01904 [cs].

Lightman, Hunter; Vineet Kosaraju; Yura Burda; Harri Edwards; Bowen Baker; Teddy Lee; Jan Leike; John Schulman; Ilya Sutskever & Karl Cobbe (May 2023). Let’s Verify Step by Step. doi: 10.48550/arXiv.2305.20050. arXiv: 2305.20050 [cs].

Liu, Yixin; Yue Yu; DiJia Su; Sid Wang; Xuewei Wang; Song Jiang; Bo Liu; Arman Cohan; Yuandong Tian & Zhengxing Chen (2026). Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training. doi: 10.48550/ARXIV.2603.12246.

Liu, Zichen; Changyu Chen; Wenjun Li; Penghui Qi; Tianyu Pang; Chao Du; Wee Sun Lee & Min Lin (Oct. 2025). Understanding R1-Zero-Like Training: A Critical Perspective. doi: 10.48550/arXiv.2503.20783. arXiv: 2503.20783 [cs].

Mansouri, Omar El; Mohamed El Amine Seddik & Salem Lahlou (2025). Noise-Corrected GRPO: From Noisy Rewards to Unbiased Gradients. doi: 10.48550/ARXIV.2510.18924.

AI-MO (2025a). aimo-validation-aime Dataset. https://huggingface.co/datasets/AI-MO/aimo-validation-aime. Accessed: 2026-03-24.

AI-MO (2025b). aimo-validation-amc Dataset. https://huggingface.co/datasets/AI-MO/aimo-validation-amc. Accessed: 2026-03-24.

Muennighof, Niklas; Zitong Yang; Weijia Shi; Xiang Lisa Li; Li Fei-Fei; Hannaneh Hajishirzi; Luke Zettlemoyer; Percy Liang; Emmanuel Candes & Tatsunori Hashimoto (Nov. 2025). “S1: Simple Test-Time Scaling”. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Ed. by Christos Christodoulopoulos; Tanmoy Chakraborty; Carolyn Rose & Violet Peng. Suzhou, China: Association for Computational Linguistics, pp. 20275–20321. isbn: 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 1025.

Natarajan, Nagarajan; Inderjit S Dhillon; Pradeep K Ravikumar & Ambuj Tewari (2013). “Learning with Noisy Labels”. In: Advances in Neural Information Processing Systems. Vol. 26. Curran Associates, Inc.

OpenAI (Sept. 2024). Learning to Reason with LLMs. https://openai.com/index/learning-toreason-with-llms/.

Pan, Tianjun; Xuan Lin; Wenyan Yang; Qianyu He; Shisong Chen; Licai Qi; Wanqing Xu; Hongwei Feng; Bo Xu & Yanghua Xiao (Mar. 2026). RubricEval: A Rubric-Level Meta-Evaluation Benchmark for LLM Judges in Instruction Following. url: https://arxiv. org/abs/2603.25133.

Proceedings of the 13th International Conference on Learning Representations (ICLR) (Apr. 2025). Singapore.

Rad, Ali; Khashayar Filom; Darioush Keivan; Peyman Mohajerin Esfahani & Ehsan Kamalinejad (Jan. 2026). Rate or Fate? RLV<sup>�</sup>R: Reinforcement Learning with Verifiable Noisy Rewards. doi: 10.48550/arXiv.2601.04411. arXiv: 2601.04411 [cs].

Rein, David; Betty Li Hou; Asa Cooper Stickland; Jackson Petty; Richard Yuanzhe Pang; Julien Dirani; Julian Michael & Samuel R. Bowman (Nov. 2023). GPQA: A Graduate-Level Google-ProofQ&A Benchmark. doi: 10 48550/ Xi 2311 12022. arXiv: 2311 12022 [cs].

Schulman, John; Filip Wolski; Prafulla Dhariwal; Alec Radford & Oleg Klimov (Aug. 2017). Proximal Policy Optimization Algorithms. doi: 10.48550/arXiv.1707.06347. arXiv: 1707.06347 [cs].

Shankar, Shreya; J.D. Zamfirescu-Pereira; Bjoern Hartmann; Aditya Parameswaran & Ian Arawjo (Oct. 2024). “Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences”. In: Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology. UIST ’24. New York, NY, USA: Association for Computing Machinery, pp. 1–14. isbn: 979-8-4007-0628-8. doi: 10.1145/3654777. 3676450.

Shao, Rulin; Shuyue Stella Li; Rui Xin; Scott Geng; Yiping Wang; Sewoong Oh; Simon Shaolei Du; Nathan Lambert; Sewon Min; Ranjay Krishna; Yulia Tsvetkov; Hannaneh Hajishirzi; Pang Wei Koh & Luke Zettlemoyer (June 2025). Spurious Rewards: Rethinking Training Signals in RLVR. doi: 10.48550/arXiv.2506.10947. arXiv: 2506.10947 [cs].

Shao, Zhihong; Peiyi Wang; Qihao Zhu; Runxin Xu; Junxiao Song; Xiao Bi; Haowei Zhang; Mingchuan Zhang; Y. K. Li; Y. Wu & Daya Guo (Apr. 2024). DeepSeekMath: Pushing the Limits ofMathematical Reasoning in Open Language Models. doi: 10.48550/arXiv.2402. 03300. arXiv: 2402.03300 [cs].

Silver, David & Richard S Sutton (2025). Welcome to the Era of Experience.

Song, Hwanjun; Minseok Kim; Dongmin Park; Yooju Shin & Jae-Gil Lee (Nov. 2023). “Learning From Noisy Labels With Deep Neural Networks: A Survey”. In: IEEE Transactions on Neural Networks and Learning Systems 34.11, pp. 8135–8153. issn: 2162-2388. doi: 10 1109/TNNLS 2022 3152527.

Szegedy, Christian; Vincent Vanhoucke; Sergey Iofe; Jon Shlens & Zbigniew Wojna (June 2016). “Rethinking the Inception Architecture for Computer Vision”. In: 2016 IEEE Confer-

ence on Computer Vision and Pattern Recognition (CVPR), pp. 2818–2826. doi: 10.1109/ CVPR.2016.308.

Tan, Sijun; Siyuan Zhuang; Kyle Montgomery; William Y. Tang; Alejandro Cuadron; Chenguang Wang; Raluca Ada Popa & Ion Stoica (Apr. 2025). “JudgeBench: A Benchmark for Evaluating LLM-based Judges”. In: Proceedings of the 13th International Conference on Learning Representations (ICLR). Singapore.

Team GLM; Aohan Zeng; Bin Xu; Bowen Wang; Chenhui Zhang; Da Yin; Dan Zhang; Diego Rojas; Guanyu Feng; Hanlin Zhao; Hanyu Lai; Hao Yu; Hongning Wang; Jiadai Sun; Jiajie Zhang; Jiale Cheng; Jiayi Gui; Jie Tang; Jing Zhang; Jingyu Sun; Juanzi Li; Lei Zhao; Lindong Wu; Lucen Zhong; Mingdao Liu; Minlie Huang; Peng Zhang; Qinkai Zheng; Rui Lu; Shuaiqi Duan; Shudan Zhang; Shulin Cao; Shuxun Yang; Weng Lam Tam; Wenyi Zhao; Xiao Liu; Xiao Xia; Xiaohan Zhang; Xiaotao Gu; Xin Lv; Xinghan Liu; Xinyi Liu; Xinyue Yang; Xixuan Song; Xunkai Zhang; Yifan An; Yifan Xu; Yilin Niu; Yuantao Yang; Yueyan Li; Yushi Bai; Yuxiao Dong; Zehan Qi; Zhaoyu Wang; Zhen Yang; Zhengxiao Du; Zhenyu Hou & Zihan Wang (2024). ChatGLM: A Family ofLarge Language Modelsfrom GLM-130B to GLM-4 All Tools. arXiv: 2406.12793 [cs.CL]. url: https://arxiv.org/abs/2406.12793.

Thakur, Aman Singh; Kartik Choudhary; Venkat Srinik Ramayapally; Sankaran Vaidyanathan & Dieuwke Hupkes (July 2025). “Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges”. In: Proceedings of the Fourth Workshop on Generation, Evaluation and Metrics (GEM<sup>2</sup>). Ed. by Ofir Arviv; Miruna Clinciu; Kaustubh Dhole; Rotem Dror; Sebastian Gehrmann; Eliya Habba; Itay Itzhak; Simon Mille; Yotam Perlitz; Enrico Santus; João Sedoc; Michal Shmueli Scheuer; Gabriel Stanovsky & Oyvind Tafjord. Vienna, Austria and virtual meeting: Association for Computational Linguistics, pp. 404–430. isbn: 979-8-89176-261-9.

Vemuri, Gautam & Rajarshi Roy (May 1989). “Stochastic Resonance in a Bistable Ring Laser”. In: Physical Review A 39.9, pp. 4668–4674. doi: 10.1103/PhysRevA.39.4668.

Vidgen, Bertie; Austin Mann; Abby Fennelly; John Wright Stanly; Lucas Rothman; Marco Burstein; Julien Benchek; David Ostrofsky; Anirudh Ravichandran; Debnil Sur; Neel Venugopal; Alannah Hsia; Isaac Robinson; Calix Huang; Olivia Varones; Daniyal Khan; Michael Haines; Zach Richards; Chirag Mahapatra; Brendan Foody & Osvald Nitski (Jan. 2026). APEX-Agents. doi: 10.48550/arXiv.2601.14242. arXiv: 2601.14242 [cs].

Viswanathan, Vijay; Yanchao Sun; Xiang Kong; Meng Cao; Graham Neubig & Tongshuang Wu (Oct. 29, 2025). “Checklists Are Better Than Reward Models For Aligning Language Models”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems. url: https://openreview.net/forum?id=RPRqKhjrr6.

Wang, Jingkang; Yang Liu & Bo Li (Apr. 2020). “Reinforcement Learning with Perturbed Rewards”. In: Proceedings of the AAAI Conference on Artificial Intelligence 34.04, pp. 6202– 6209. issn: 2374-3468. doi: 10.1609/aaai.v34i04.6086.

Ward, Lawrence M & Priscilla E Greenwood (May 2016). “Stochastic Facilitation in the Brain?” In: Journal of Statistical Mechanics: Theory and Experiment 2016.5, p. 054033. issn: 1742-5468. doi: 10.1088/1742-5468/2016/05/054033.

Xu, Donglai; Hongzheng Yang; Yuzhi Zhao; Pingping Zhang; Jinpeng Chen; Wenao Ma; Zhijian Hou; Mengyang Wu; Xiaolei Li; Senkang Hu; Ziyi Guan; Jason Chun Lok Li & Lai Man Po (Nov. 2025a). From Exploration to Exploitation: A Two-Stage Entropy RLVR Approachfor Noise-Tolerant MLLM Training. doi: 10.48550/arXiv.2511.07738. arXiv: 2511.07738 [cs].

Xu, Zhangchen; Yuetai Li; Fengqing Jiang; Bhaskar Ramasubramanian; Luyao Niu; Bill Yuchen Lin & Radha Poovendran (May 2025b). TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning. doi: 10 48550/ X 2505 14625. arXiv: 2505 14625 [cs].

Yang, An; Anfeng Li; Baosong Yang; Beichen Zhang; Binyuan Hui; Bo Zheng; Bowen Yu; Chang Gao; Chengen Huang; Chenxu Lv; Chujie Zheng; Dayiheng Liu; Fan Zhou; Fei Huang; Feng Hu; Hao Ge; Haoran Wei; Huan Lin; Jialong Tang; Jian Yang; Jianhong Tu; Jianwei Zhang; Jianxin Yang; Jiaxi Yang; Jing Zhou; Jingren Zhou; Junyang Lin; Kai Dang; Keqin Bao; Kexin Yang; Le Yu; Lianghao Deng; Mei Li; Mingfeng Xue; Mingze Li; Pei Zhang; Peng Wang; Qin Zhu; Rui Men; Ruize Gao; Shixuan Liu; Shuang Luo; Tianhao Li; Tianyi Tang; Wenbiao Yin; Xingzhang Ren; Xinyu Wang; Xinyu Zhang; Xuancheng Ren; Yang Fan; Yang Su; Yichang Zhang; Yinger Zhang; Yu Wan; Yuqiong Liu; Zekun Wang; Zeyu Cui; Zhenru Zhang; Zhipeng Zhou & Zihan Qiu (May 2025). Qwen3 Technical Report. doi: 10.48550/arXiv.2505.09388. arXiv: 2505.09388 [cs].

Yang, An; Beichen Zhang; Binyuan Hui; Bofei Gao; Bowen Yu; Chengpeng Li; Dayiheng Liu; Jianhong Tu; Jingren Zhou; Junyang Lin; Keming Lu; Mingfeng Xue; Runji Lin; Tianyu Liu; Xingzhang Ren & Zhenru Zhang (Sept. 2024). Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. doi: 10.48550/arXiv.2409.12122. arXiv: 2409.12122 [cs].

Zeng, Weihao; Yuzhen Huang; Qian Liu; Wei Liu; Keqing He; Zejun Ma & Junxian He (Aug. 2025). “SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild”. In: Second Conference on Language Modeling.

Zheng, Chujie; Shixuan Liu; Mingze Li; Xiong-Hui Chen; Bowen Yu; Chang Gao; Kai Dang; Yuqiong Liu; Rui Men; An Yang; Jingren Zhou & Junyang Lin (2025). Group Sequence Policy Optimization. doi: 10.48550/ARXIV.2507.18071.

Zheng, Lianmin; Wei-Lin Chiang; Ying Sheng; Siyuan Zhuang; Zhanghao Wu; Yonghao Zhuang; Zi Lin; Zhuohan Li; Dacheng Li; Eric P. Xing; Hao Zhang; Joseph E. Gonzalez & Ion Stoica (Dec. 2023). “Judging LLM-as-a-judge with MT-bench and Chatbot Arena”. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. NIPS ’23. Red Hook, NY, USA: Curran Associates Inc., pp. 46595–46623.

Zhou, Yang; Sunzhu Li; Shunyu Liu; Wenkai Fang; Kongcheng Zhang; Jiale Zhao; Jingwen Yang; Yihe Zhou; Jianwei Lv; Tongya Zheng; Hengtong Lu; Wei Chen; Yan Xie & Mingli Song (Oct. 2025). Breaking the Exploration Bottleneck: Rubric-Scafolded Reinforcement Learning for General LLM Reasoning. doi: 10.48550/arXiv.2508.16949. arXiv: 2508.16949 [cs].

Zhu, Lianghui; Xinggang Wang & Xinlong Wang (Apr. 2025a). “JudgeLM: Fine-tuned Large Language Models are Scalable Judges”. In: Proceedings of the 13th International Conference on Learning Representations (ICLR). Singapore.

Zhu, Yuxuan & Daniel Kang (Mar. 2026). Noisy Data Is Destructive to Reinforcement Learning with Verifigble Rewards, po1: 10.48550/arXiy.2603.16140, arXiv: 2603.16140 [cs]

Zhu, Zhanxing; Jingfeng Wu; Bing Yu; Lei Wu & Jinwen Ma (June 2019). The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior ofEscapingfrom Sharp Minima and Regularization Efects. doi: 10.48550/arXiv.1803.00195. arXiv: 1803.00195 [stat].

Zhu, Zilin; Chengxing Xie; Xin Lv & slime Contributors (2025b). slime: An LLM post-training framework for RL Scaling. https://github.com/THUDM/slime. GitHub repository. Corresponding author: Xin Lv.

Zhuge, Mingchen; Changsheng Zhao; Dylan Ashley; Wenyi Wang; Dmitrii Khizbullin; Yunyang Xiong; Zechun Liu; Ernie Chang; Raghuraman Krishnamoorthi; Yuandong Tian; Yangyang Shi; Vikas Chandra & Jürgen Schmidhuber (July 2025). “Agent-as-a-Judge: Evaluate Agents with Agents”. In: Proceedings of the 42nd International Conference on Machine Learning (ICML). Vancouver, Canada, pp. 80569–80611.

## A. Reproducibility

Note that the base Qwen3 (before training) performs worse than in the Qwen team’s technical report (Qwen Team, 2025). We provide a script that reproduces our numbers with tinker.<sup>4</sup> The cause of the diference is that the post-training framework we used applies the chat template (Hugging Face, 2025a), which injects extra tags. The base model has not been trained for this and is confused by the changes.

## B. Training details

## B.1. Final hyperparameters

Table 3 lists the hyperparameters used across all experiments unless otherwise noted.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-6}$ </td></tr><tr><td>LR schedule</td><td>Constant</td></tr><tr><td>Weight decay</td><td>0.10</td></tr><tr><td> $\beta_1, \beta_2$ </td><td>0.90, 0.98</td></tr><tr><td colspan="2">GRPO / GSPO</td></tr><tr><td>Clip range ( $\epsilon_{low}, \epsilon_{high}$ )</td><td>0.20, 0.28</td></tr><tr><td>KL coefficient</td><td>0.00</td></tr><tr><td>Entropy coefficient</td><td>0.00</td></tr><tr><td>Rollouts per prompt</td><td>16</td></tr><tr><td>Global batch size</td><td>96</td></tr><tr><td colspan="2">Sampling (training)</td></tr><tr><td>Temperature</td><td>1.00</td></tr><tr><td>Top-p</td><td>1.00</td></tr><tr><td>Max response length</td><td>4096 (MBPP) / 8192 (GPQA)</td></tr><tr><td colspan="2">Evaluation</td></tr><tr><td>Samples per prompt</td><td>16</td></tr><tr><td>Temperature</td><td>0.70</td></tr><tr><td>Top-p</td><td>1.00</td></tr><tr><td>Max response length</td><td>8192</td></tr><tr><td>Eval interval</td><td>Every 20 steps</td></tr></table>

Table 3. Hyperparameters used for all experiments. MBPP experiments use GRPO; GPQA experiments use GSPO.

## B.2. Hyperparameter sensitivity

Table 4 compares training performance under alternative sampling hyperparameters for GLM4 9B and Qwen3 8B. The main setting (top-�=1.00, temperature=1.00) is reproduced from Table 1 for reference.

Reducing top-� from 1.00 to 0.95 has a dramatic efect on GLM4 9B: the best reward drops from 0.905 to 0.817 and the final reward drops from 0.888 to 0.764. Qwen3 8B is also sensitive

to top-�, with best reward dropping from 0.901 to 0.829. Lowering the temperature from 1.00 to 0.70 hurts both models, particularly Qwen3 8B (0.803 vs. 0.901 best), likely because the reduced sampling diversity limits GRPO’s ability to estimate meaningful advantages within each group. Increasing the global batch size from 96 to 768 slightly reduces peak performance for both models (GLM4 9B: 0.888 vs. 0.905; Qwen3 8B: 0.845 vs. 0.901), though it eliminates the gap between best and final reward, suggesting more stable but slower optimization.

<table><tr><td>Model</td><td>Hyperparameters</td><td>Seeds</td><td>Best</td><td>Final</td></tr><tr><td colspan="5">GLM4 9B</td></tr><tr><td></td><td>top-p=1.00, T=1.00, B=96 (main)</td><td>3</td><td>0.905 ± 0.002</td><td>0.884 ± 0.006</td></tr><tr><td></td><td>top-p=0.95, T=1.00, B=96</td><td>2</td><td>0.817 ± 0.005</td><td>0.764 ± 0.002</td></tr><tr><td></td><td>top-p=1.00, T=0.70, B=96</td><td>1</td><td>0.895</td><td>0.881</td></tr><tr><td></td><td>top-p=1.00, T=1.00, B=768</td><td>1</td><td>0.888</td><td>0.888</td></tr><tr><td colspan="5">Qwen3 8B</td></tr><tr><td></td><td>top-p=1.00, T=1.00, B=96 (main)</td><td>2</td><td>0.901 ± 0.009</td><td>0.864 ± 0.007</td></tr><tr><td></td><td>top-p=0.95, T=1.00, B=96</td><td>2</td><td>0.829 ± 0.025</td><td>0.827 ± 0.022</td></tr><tr><td></td><td>top-p=1.00, T=0.70, B=96</td><td>1</td><td>0.803</td><td>0.769</td></tr><tr><td></td><td>top-p=1.00, T=1.00, B=768</td><td>1</td><td>0.845</td><td>0.845</td></tr></table>

Table 4. Sensitivity to sampling hyperparameters for baseline (no noise) training on MBPP. � denotes temperature, � the global batch size. All runs use 16 rollouts per prompt. The main setting rows are reproduced from Table 1.

## C. Prompts

We list the prompts used in our experiments. For MBPP and GPQA, the task description is wrapped using the model’s chat template (apply\_chat\_template (Hugging Face, 2025a)). For the model-based verifier experiments, we use the following system and user prompts.

## C.1. Model-based verifier: unit test evaluation

The model-based verifier is given the generated code and a single unit test and asked whether the assert statement would pass.

System: You are an expert code evaluator evaluating the quality of a model’s response against a test case. Your task is to determine whether the GENERATED CODE satisfies the TEST CASE. I.e., whether the assert statement would pass. Answer with a JSON object with the key “status” and a boolean value.

Instructions:

• Carefully read the TEST CASE to understand what is required.

• Evaluate the GENERATED CODE against this TEST CASE only (ignore anything outside it).

User:

GENERATED CODE:

{generated\_code}

TEST CASE:

$$
\text { assert } \{\text { function\_name } \} (\{\text { test\_input } \}) = = \{\text { test\_output } \}
$$

## C.2. GPQA: answer extraction

For GPQA, the model is presented with the question and multiple-choice options via the chat template. The reward is computed by extracting the selected option letter from the model’s response using a series of regex patterns and comparing it against the correct answer. No model-based verifier is used for GPQA evaluation.

## D. Background

In RLVR, a language model is trained using reinforcement learning, where a verifier determines the reward for each generated response—for example, executing unit tests for code or checking a multiple-choice answer. This section describes the training objective.

GRPO. We use GRPO as our training objective. Unlike the earlier Proximal Policy Optimization (PPO) (Schulman et al., 2017), GRPO omits the value network and instead estimates advantages by comparing rollouts within a group sampled for the same prompt.

Concretely, for each prompt �, GRPO samples a group of � responses $\{ y _ { 1 } , . . . , y _ { G } \} \sim \pi _ { \theta _ { \mathrm { o l d } } } ( . ~ |$ �) and computes a reward $r _ { i } = R ( x , y _ { i } )$ for each response. The advantage for response �<sub>�</sub> is computed by normalizing rewards within the group:

$$
\hat {A} _ {i} = \frac {r _ {i} - \mu (\mathbf {r})}{\sigma (\mathbf {r}) + \epsilon}\tag{1}
$$

where $\mu ( \mathbf { r } )$ and $\sigma ( \mathbf { r } )$ are the mean and standard deviation of the group rewards, and � is a small constant for numerical stability.

GSPO. For the scientific reasoning experiments, we use Group Sequence Policy Optimization (GSPO) (Zheng et al., 2025) instead of GRPO to stabilize training convergence. GSPO modifies the advantage computation to reduce variance in sparse-reward settings, which we found helped for GPQA. We refer the reader to (Zheng et al., 2025) for technical details.

## E. In-depth discussion

## E.1. Benefits of noise and flat minima

We discussed in Section 6.2 that the noise can cause the gradients to be flipped. This mechanism connects to a broad body of work on the role of noise and perturbation in optimization, showing why it is not detrimental. Keskar et al. (2017) showed that the noise inherent in small-batch SGD biases optimization toward flat minima that generalize better, an efect that Le (2018) formalized by showing SGD noise acts as an implicit temperature parameter in a Bayesian framework. Crucially, Zhu et al. (2019) demonstrated that SGD noise is anisotropic—aligned with the curvature of the loss landscape—making it more efective than isotropic random perturbation, which is consistent with our observation that structured gradient inversion outperforms the unstructured perturbation induced by sample-level noise. The benefit of explicitly perturbing parameters uphill is central to Sharpness-Aware Minimization (Foret et al., 2020), which takes an adversarial step to maximize the loss before computing the gradient, biasing optimization toward flatter regions; Entropy-SGD (Chaudhari et al., 2017) achieves a similar efect by optimizing a smoothed “local entropy” objective. From a theoretical perspective, Kleinberg et al. (2018) showed that SGD noise enables escape from local minima in non-convex landscapes, and Jin et al. (2017) proved that adding perturbation to gradient descent allows eficient escape from saddle points. The preference for flat minima itself traces back to Hochreiter & Schmidhuber (1997), who argued that flat minima correspond to low-complexity solutions and thus better generalization—a connection that has since been formalized through PAC-Bayes bounds (Dziugaite & Roy, 2017).

Additionally, our findings echo the phenomenon of stochastic resonance—where noise enhances signal detection in nonlinear systems—studied extensively in physics and neuroscience (Vemuri & Roy, 1989; Ward & Greenwood, 2016). The beneficial efects of noise are well-established in supervised learning; e.g., with dropout, label smoothing (Szegedy et al., 2016; Song et al., 2023), and techniques for learning from noisy labels (Natarajan et al., 2013).

## E.2. The results of Xu et al. (2025b)

Following the results we presented in Section 6.3, we compare our conclusion with that of Xu et al. (2025b).

Xu et al. (2025b) did not report their verifier’s false-negative rate. The only reported metric is that, conditional on the prime verifier marking a response incorrect, 38.5% were actually correct. Thus, the only available information is that TinyV’s false-negative rate is upper-bounded by the prime verifier’s (Cui et al., 2025), as dictated by design (Xu et al., 2025b, Figure 5). Similarly, the design suggests the false-positive rate will be equal to or higher than the prime verifier’s, since the LLM may incorrectly classify negatives as positives.

This absence of metrics is significant: the false-negative and false-positive rates of both verifiers remain unknown. While their experiments show greater gains with TinyV, the underlying cause is not conclusively established. Although theoretical arguments support reducing false negatives, the unreported false-positive rates prevent ruling out that observed improvements are partially or entirely due to changes in false-positive rates.

## F. Detailed metrics for the model-based verifiers

Figure 6. Training Qwen3 8B with model-based verifier rewards on MBPP. Solid lines show eval metrics; dashed lines show exponentially smoothed rollout metrics. Top row, left to right: validation reward (against ground-truth unit tests) with rollout reward from the (model-based) verifier, verifier accuracy, and verifier F1 score. Bottom row: verifier precision and recall. Both verifiers maintain high recall (>90%), but the Qwen3 4B verifier (blue) has substantially lower precision than the 30B verifier (orange), injecting false-positive noise that limits the trained model to a peak validation reward of 0.704 vs. 0.871 with the 30B verifier.

We show in Figure 6 the reward for training and validation when using the model-based verifiers; these overlap with the results in Section 6.3. For training, the rewards are given by the model-based verifier, while the validation rewards come from running the actual unit tests. The accuracy, F1 score, precision, and recall are for the model-based verifier when comparing the pass/fail decisions with running the unit tests.

Figure 7. Best validation reward for Qwen3 4B and 8B under group rollout noise at varying noise levels. Both models degrade gracefully up to $\scriptstyle p = 0 . 3 0 ;$ the drop at $\scriptstyle { p = 0 . 4 0 }$ is more pronounced for the smaller model.

As mentioned in the main text in Section 6.3, the 4B verifier has a much lower precision, but a slightly higher recall. We see that the accuracy and F1 score are also much lower (to be expected from the precision and recall metrics). Thus, we can see that the recall is not the most important metric to maximize when building a verifier, and the model is somewhat robust to false negatives in contrast with Xu et al. (2025b). Also, it is interesting to note that the F1 score for the 4B verifier is in the mid-70s to mid-80s. This range is noteworthy because He et al. (2025) post-trained a verifier where they report the uplift in the F1 score, which increased from 0.515 to 0.728 (He et al., 2025, Table 4). Thus, based on our results, their verifier might not be good enough to deliver the full uplift available from post-training.

## G. Convergence rate

Figure 8 compares the training curves of the clean baseline and group rollout noise at $p { = } 0 . 1$ The noisy runs track the baseline closely, reaching comparable peak performance with only a slight delay. Notably, the noisy runs exhibit less overfitting: their final-checkpoint reward is closer to their peak than the baseline’s, where performance degrades after the peak. At higher noise levels, training curves show increased oscillation and slower convergence (see Figure 3 for final-checkpoint results).

## H. Model size

To examine whether robustness to noise varies with model capacity, we compare Qwen3 4B and Qwen3 8B under group rollout noise at several noise levels (Figure 7). Both models degrade gracefully up to $\scriptstyle { p = 0 . 3 0 }$ , with the 8B model maintaining a consistent advantage. At $\scriptstyle { p = 0 . 4 0 }$ , the 8B model drops sharply to 0.769 (a loss of 0.13 from its baseline), while the 4B model is more stable at 0.804 (a loss of 0.065). At low noise $\scriptstyle ( p = 0 . 0 5 )$ , both models nearly match their clean baselines, confirming that moderate noise is well-tolerated regardless of model size.


Figure 8. Training curves for group rollout noise at $p { = } 0 . 1$ . Shaded regions indicate ±1 standard deviation across seeds.

## I. Response length

We show in Figure 9 the median response length of Qwen3 8B and GLM4 9B during training of the baseline noise-free setting and when training with group rollout noise with �=0.1. We see from the figure that the noise does not impact how the models learn to write shorter answers over time. The answer length and how quickly it decreases are the same regardless of the noise.

Why does the response length decrease over time? One of the key observations in the DeepSeek R1 paper was that the model tends to generate longer responses during training, which they link to the model’s ability to solve harder problems more accurately (DeepSeek-AI, 2025, Figure 1 (b)). Specifically, the authors write the following.

As shown in Figure 1(b), DeepSeek-R1-Zero exhibits a steady increase in thinking time throughout training, driven solely by intrinsic adaptation rather than external modifications. Leveraging long CoT, the model progressively refines its reasoning, generating hundreds to thousands of tokens to explore and improve its problemsolving strategies.

The increase in thinking time fosters the autonomous development of sophisticated behaviors. Specifically, DeepSeek-R1-Zero increasingly exhibits advanced reasoning strategies such as reflective reasoning and systematic exploration of alternative solutions. (DeepSeek-AI, 2025, p. 5)

However, Liu et al. (2025) noted that output length does not imply better downstream performance. When we go through early (in terms of training steps) rollouts, we notice that the models generate very long, verbose, and at times circular reasoning chains. The long outputs cause the models not to finish generation within the token budget. Many of the paragraphs start with expressions like “Alternatively,” “But wait,” “Wait,” “So how do I approach this?”, and “So the approach.” One likely explanation is that the models are using the ideas of Muennighof et al. (2025), who showed that reasoning performance could be improved by extending the model’s thinking process by appending “Wait,” when it was about to exit thinking mode.

Thus, one of the key elements the models learn in the beginning is to shorten the thinking process to ensure they provide an answer.

Qwen3-8B

Best Final Baseline Noisy (�=0.10) Base model =0.05 =0.10 =0.15 =0.20
Noise level (�)

Figure 9. Median response length over training steps for group rollout noise at �=0.1 compared to the no-noise baseline. Shaded regions indicate ±1 standard deviation across seeds.



Figure 10. Llama 3.1 8B with group rollout noise on MBPP. Left: best and final validation reward vs. noise level. Center: training curves by noise level. Right: median response length comparison.

## J. Llama results

We run a subset of experiments with Llama 3.1 8B to verify that our findings are not specific to the Qwen/GLM model families. Figure 10 shows the noise sweep and training curves for Llama 3.1 8B with group rollout noise. The clean baseline achieves a best validation reward of 0.658, lower than both GLM4 9B and Qwen3 8B, consistent with Llama 3.1 8B’s weaker base coding performance. The noise robustness pattern is consistent with our main results: moderate noise (�≤0.10) causes minimal degradation, while higher noise levels lead to progressive performance loss.

## K. Cai et al. (2025)’s inconsistent results

Cai et al. (2025) show in their Figure 2 pass@1 results for Qwen2.5-Math-1.5B, DeepSeek-R1-Distill-Qwen-1.5B, Llama-3.2-3B-Instruct, and Qwen2.5-Math-7B on 6 math benchmarks. However, their results are not consistent with other works that test on the same benchmark.

Yang et al. (2024) and DeepSeek-AI (2025) report numbers for some of the same benchmarks. Since these are the oficial technical reports for the models, we work with the assumption that they have done a proper evaluation of the models. We collect the relevant numbers in Table 5. We managed to get comparisons for the results on the Math (Hendrycks et al., 2021), MinervaMath (Lewkowycz et al., 2022), Olympiad Bench (He et al., 2024), AIME 2024 (AI-MO, 2025a), and AMC 2023 (AI-MO, 2025b) benchmarks.

<table><tr><td rowspan="2"></td><td colspan="2">Qwen2.5-Math-1.5B</td><td colspan="2">Qwen2.5-Math-7B</td><td colspan="2">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Reference</td><td>Range</td><td>Reference</td><td>Range</td><td>Reference</td><td>Range</td></tr><tr><td>Math</td><td>69.4</td><td>[48, 70]</td><td>75.1</td><td>[52, 80]</td><td>83.9</td><td>[60, 78]</td></tr><tr><td>MinervaMath</td><td>29.4</td><td>[6, 18]</td><td>34.6</td><td>[10, 25]</td><td></td><td></td></tr><tr><td>Olympiad Bench</td><td>31.3</td><td>[24, 32]</td><td>38.2</td><td>[26, 38]</td><td></td><td></td></tr><tr><td>AIME 2024</td><td>3.3</td><td>[7, 16]</td><td>13.3</td><td>[12, 30]</td><td>28.9</td><td>[10, 20]</td></tr><tr><td>AMC 2023</td><td>45.0</td><td>[34, 50]</td><td>62.5</td><td>[45, 63]</td><td></td><td></td></tr></table>

Table 5. Pass@1 results for Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and DeepSeek-R1- Distill-Qwen-1.5B. References: Yang et al. (2024) and DeepSeek-AI (2025) for Qwen2.5 and DeepSeek-R1-Distill numbers, respectively. The range is given by the base and best post-trained results in (Cai et al., 2025, Figure 2). Notice that Cai et al. (2025)’s results are usually below the reference for the entire range. Thus, even after post-training the model, they report numbers worse than the base model’s oficial values.

The lower end of the intervals in the ranges is the base numbers reported by Cai et al. (2025), while the upper end of the intervals is the results after post-training the models with ground-truth labels. Importantly, we see that the reference numbers are often well above the interval. Thus, even after post-training the models, Cai et al. (2025) cannot beat the base models.

## L. Ackley function optimization

## L.1. Setup

The “policy” is a diagonal Gaussian with fixed standard deviation:

$$
\pi (\cdot \mid \mu) = \mathcal {N} (\mu , \sigma^ {2} I), \qquad \mu \in \mathbb {R} ^ {2}.
$$

$\mu$ is the only learnable parameter, optimized by Adam (Kingma & Ba, 2015).

## L.2. Forward pass (one iteration)

## L.2.1. Sampling

Samples are generated with $\mu$ detached from the computation graph:

$$
s _ {i} = \mu_ {\mathrm{detach}} + \sigma \varepsilon_ {i}, \qquad \varepsilon_ {i} \sim \mathcal {N} (0, I), \quad i = 1, \ldots , G.
$$

Because $\mu$ is detached, the samples $s _ { i }$ are constants with no gradient connection to $\mu$ .

## L.2.2. Rewards and advantages

The Ackley cost is evaluated on these constant samples and converted to group-relative advantages:

$$
r _ {i} = - \operatorname{ackley} (s _ {i}) + \text {noise}, \quad A _ {i} = \frac {r _ {i} - \bar {r}}{\operatorname{std} (r) + 1 0 ^ {- 8}}.
$$

There are still no gradients flowing through this computation.

## L.2.3. Log-probabilities (where gradients enter)

A distribution ${ \cal N } ( \mu , \sigma ^ { 2 } I )$ is constructed with the live $\mu ,$ , and log-probabilities are evaluated on the constant samples:

$$
\log \pi (s _ {i} \mid \mu) = \sum_ {d = 1} ^ {2} \left[ - \frac {(s _ {i , d} - \mu_ {d}) ^ {2}}{2 \sigma^ {2}} - \log \sigma - \frac {1}{2} \log 2 \pi \right].
$$

The above expression is the only point where $\mu$ enters the computation graph.

## L.2.4. Policy loss

The loss is the standard REINFORCE objective with group-relative advantages:

$$
L = - \frac {1}{G} \sum_ {i = 1} ^ {G} A _ {i} \cdot \log \pi (s _ {i} \mid \mu).
$$

Diferentiating the log-probability with respect to $\mu _ { d }$ :

$$
\frac {\partial \log \pi (s _ {i} \mid \mu)}{\partial \mu_ {d}} = \frac {s _ {i , d} - \mu_ {d}}{\sigma^ {2}}.
$$

Since $s _ { i , d } - \mu _ { d } = \sigma \varepsilon _ { i , d }$ (the detached and live $\mu$ values share the same value at evaluation time):

$$
\frac {\partial \log \pi (s _ {i} \mid \mu)}{\partial \mu_ {d}} = \frac {\varepsilon_ {i , d}}{\sigma}.
$$

## L.3. Gradient

$$
\nabla_ {\mu} L = - \frac {1}{G \sigma} \sum_ {i = 1} ^ {G} A _ {i} \varepsilon_ {i}
$$

Each step pushes $\mu$ toward samples that received above-average rewards and away from below-average ones, weighted by the normalized noise directions. Adam applies adaptive learning rates on top of this raw gradient.
