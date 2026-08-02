# Adversarial Reward Auditing for Active Detection and Mitigation of Reward Hacking

Mohammad Beigi <sup>1</sup> Ming Jin <sup>2</sup> Junshan Zhang <sup>1</sup> Qifan Wang <sup>3</sup> Lifu Huang <sup>1</sup>

## Abstract

Reinforcement Learning from Human Feedback (RLHF) remains vulnerable to reward hacking, where models exploit spurious correlations in learned reward models to achieve high scores while violating human intent. Existing mitigations rely on static defenses that cannot adapt to novel exploitation strategies. We propose Adversarial Reward Auditing (ARA), a framework that reconceptualizes reward hacking as a dynamic, competitive game. ARA operates in two stages: first, a Hacker policy discovers reward model vulnerabilities while an Auditor learns to detect exploitation from latent representations; second, Auditor-Guided RLHF (AG-RLHF) gates reward signals to penalize detected hacking, transforming reward hacking from an unobservable failure into a measurable, controllable signal. Experiments across three hacking scenarios demonstrate that ARA achieves the best alignment-utility tradeoff among all baselines: reducing sycophancy to near-SFT levels while improving helpfulness, decreasing verbosity while achieving the highest ROUGE-L, and suppressing code gaming while improving Pass@1. Beyond single-domain evaluation, we show that reward hacking, detection, and mitigation all generalize across domains—a Hacker trained on code gaming exhibits increased sycophancy despite no reward for this behavior, and an Auditor trained on one domain effectively suppresses exploitation in others, enabling efficient multi-domain defense with a single model.

## 1. Introduction

Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022) has become a dominant paradigm for aligning Large Language Models (LLMs) with human preferences. In typical RLHF pipelines, a reward model (RM) is first trained from human-ranked response pairs to approximate preference judgments, and the LLM policy is then optimized to maximize this learned reward. However, since the reward models are usually imperfect proxies for true human intent, this pipeline suffers from a fundamental vulnerability: reward hacking, where models exploit spurious correlations in the reward model to achieve high scores while violating genuine human preferences (Gao et al., 2023; Geirhos et al., 2020; Amodei et al., 2016; Everitt et al., 2021; Liu et al., 2024). Known manifestations of reward hacking include length bias (i.e., overly verbose but inaccurate generations) (Chen et al., 2024; Gao et al., 2023), “gaming” in coding tasks (e.g., modifying unit tests rather than solving problems) (Baker et al., 2025; MacDiarmid et al., 2025), and sycophancy that prioritizes agreement over accuracy (Denison et al., 2024; Beigi et al., 2025). Importantly, recent work (MacDiarmid et al., 2025) demonstrates that reward hacking may generalize beyond isolated settings: once a model learns to exploit a proxy reward in one domain, it can acquire more strategic misaligned behaviors, such as appearing compliant under monitoring while pursuing misaligned objectives when unobserved.

There has been a growing body of work that seeks to mitigate reward hacking, broadly categorized into three directions: (1) Regularization methods, which constrain optimization through KL divergence penalties (Chen et al., 2024), or information-theoretic constraints that filter taskirrelevant features (Miao et al., 2024); (2) Reward model improvements, such as scaling reward models (Gao et al., 2023), employing ensemble methods for more robust preference estimation (Liu et al., 2024; Eisenstein et al., 2023), and reward clipping (Engstrom et al., 2020; Zheng et al., 2023; Chen et al., 2024); and (3) Bias-specific interventions, which directly penalize known artifacts such as excessive response length (Singhal et al., 2024; Chen et al., 2024). While these strategies mitigate specific failure modes, they share a fundamental limitation: they are largely static defenses against an inherently dynamic problem. In particular, they provide no explicit mechanism to audit whether a high reward reflects genuine task quality or exploitation of spurious correlations in the reward model. Furthermore, these approaches can only suppress previously identified hacking patterns, but cannot detect or adapt to sophisticated, emergent forms of reward exploitation that arise during optimization.

Figure 1. Adversarial Reward Auditing (ARA) trains a Hacker to exploit a frozen proxy reward model while an Auditor learns to detect reward-hacked outputs. The Auditor then gates the reward used for RL, making exploitation detectable and unprofitable.

Motivated by this perspective, we propose Adversaria Reward Auditing (ARA), a framework that formulates reward hacking as a competitive two-player game between a Hacker and an Auditor. As illustrated in Figure 1, ARA operates in two stages: In Stage 1 (Hacker-Auditor Game), we train two competing components against a frozen reward model. The Hacker is a language model (initialized from supervised fine-tuning) that learns to generate responses with high proxy reward through exploitation. The Auditor is a classifier that operates on the reward model’s internal representations and learns to distinguish genuinely high-quality responses from exploitative ones. These components are trained adversarially: the Hacker is incentivized to evade detection while maximizing reward, and the Auditor continually updates to recognize the Hacker’s evolving strategies. This co-evolution exposes the Auditor to hard, adaptive adversarial examples, preventing overfitting to simple hacking patterns. In Stage 2 (Auditor-Guided RLHF (AG-RLHF)), we deploy the trained Auditor to guide standard RLHF training. Rather than using the proxy reward directly, AG-RLHF gates the reward signal based on the Auditor’s exploitation probability: responses flagged as exploitative receive suppressed rewards, making hacking strategies unprofitable. This transforms reward hacking from an unobservable failure into a measurable signal that actively shapes policy optimization. This two-stage design provides two key bene fits. First, the competitive dynamics create a self-improving mechanism where the Hacker continuously exposes reward model vulnerabilities, providing exactly the challenging cases needed to strengthen detection. Second, unlike static defenses against predetermined failure modes, our framework actively discovers emerging forms of reward hacking, adapting to novel exploitation strategies as they develop during training. Building on these advantages, ARA mitigates reward hacking in RLHF by making exploitation detectable and thus unprofitable.

We conduct extensive experiments across three reward hacking scenarios including sycophancy (Sharma et al., 2025; Denison et al., 2024), length bias (Chen et al., 2024; Gao et al., 2023), and code gaming (MacDiarmid et al., 2025; Baker et al., 2025), demonstrating that ARA consistently achieves the best alignment-utility tradeoff by reducing hacking while maintaining task performance, compared to all existing strong baselines. Furthermore, our analysis reveals that reward hacking generalizes across domains: exploits learned in one setting transfer to others. A hacker trained only on code gaming exhibits 22.5% higher sycophancy despite receiving no reward for this behavior, with cross-domain transfer accelerating precisely when indomain exploitation saturates. Critically, we find that ARA’s detection and mitigation capabilities also generalize: the Auditor’s ability to detect and suppress exploits transfers across domains. Beyond single-domain evaluation, we demonstrate that both reward hacking and its mitigation generalize across domains.

Our contributions are summarized as follows:

• We propose ARA, the first framework that formulates reward hacking as a competitive two-player game, jointly training a hacker to discover reward exploits and an auditor to detect them, transforming reward hack ing from an unobservable failure into a measurable, controllable signal.

• We demonstrate through extensive experiments on multiple benchmarks that our framework detects and significantly mitigates reward hacking compared to existing methods, achieving better alignment with human preferences.

• We show that reward hacking, detection, and mitigation all generalize across domains: a Hacker trained on one domain exhibits exploitation in others, and an Auditor trained on a single domain effectively suppresses hacking across multiple domains.

## 2. Related Works

Reward Hacking in Large Language Models. Reward hacking, which presents a prominent challenge in alignment and safe deployment of AI systems, occurs when policy models exploit proxy reward models to achieve high scores without fulfilling true human objectives (Gao et al., 2023; Geirhos et al., 2020; Amodei et al., 2016; Everitt et al., 2021; Wen et al., 2024). This stems from reward misgeneralization, where RMs trained on finite preference datasets serve as imperfect proxies for human preference, incorrectly associating rewards with spurious features that correlate with training preferences but not actual quality (Gao et al., 2023; Liu et al., 2024; Skalse et al., 2025). As policies optimize these flawed proxies, proxy metrics increase while true alignment deteriorates (Skalse et al., 2025; Miao et al.,

2024; Amodei et al., 2016). Common known manifestations include length bias, (Chen et al., 2024), sycophancy (Denison et al., 2024; Beigi et al., 2025), and coding exploitation (Baker et al., 2025). As model capabilities advance, reward hacking evolves from passive exploitation to sophisticated strategic behavior, necessitating adaptive rather than static solutions (Taylor et al., 2025; Denison et al., 2024; MacDiarmid et al., 2025).

Mitigating Reward Hacking in Large Language Models. Current mitigation approaches treat reward hacking as an optimization problem to constrain rather than an adversarial challenge to counter. Regularization-based methods, primarily KL divergence penalties (Chen et al., 2024) and information-theoretic constraints that filter task-irrelevant features (Miao et al., 2024). Reward model improvements through scaling (Gao et al., 2023) and ensembles (Eisenstein et al., 2023) attempt to create more robust reward signals, yet scaling up network size or quantity presents limited feasibility and may incur significant costs, especially for models with billions of parameters. Targeted debiasing approaches (Singhal et al., 2024; Chen et al., 2024) address specific known biases such as response length, bu remain limited to predetermined failure modes and cannot generalize to novel exploitation strategies. Our approach is distinct from existing methods since it specifically targets reward misgeneralization through an adversarial auditor that actively probes for and exposes spurious correlations as they emerge during training. Furthermore, it provides automatic detection of whether high-scoring responses achieve their rewards through genuine quality or exploitation, transforming reward hacking from an unobservable failure into a measurable signal.

## 3. ARA: Adversarial Reward Auditing

We operate within the standard RLHF setting. Given an input prompt x and a model-generated response y, a reward model $R _ { \theta } ( x , y )$ assigns a scalar score intended to reflect how well y aligns with human preferences for x. In practice, $R _ { \theta }$ is generally trained on a human-preference dataset $\mathcal { D } _ { \mathrm { p r e f } } = \{ ( x ^ { ( i ) } , y _ { w } ^ { ( i ) } , y _ { l } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , where $y _ { w }$ and $y _ { l }$ denote preferred and dispreferred responses for the same prompt x. The reward model thus serves as a learned proxy for an inaccessible ground-truth preference function $R ^ { * }$ . RLHF then optimizes a policy $\pi _ { \phi }$ , initialized from a supervised fine-tuned model $\pi _ { \mathrm { S F T } }$ , via

$$
\mathcal {J} _ {\mathrm{RLHF}} (\phi) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\phi}} \left[ R _ {\theta} (x, y) - \beta D _ {\mathrm{KL}} \left(\pi_ {\phi} \| \pi_ {\mathrm{SFT}}\right) \right].
$$

Reward hacking occurs when $\pi _ { \phi }$ exploits misspecifications in $R _ { \theta }$ such that $R _ { \theta } ( x , y )$ increases while $R ^ { * } ( x , y )$ decreases.

We introduce Adversarial Reward Auditing $\mathbf { ( A R A ) }$ , a framework that formulates reward hacking as a competitive two-player game. ARA operates in two stages: Stage 1 (Hacker-Auditor Game), conducted prior to policy optimization, where we train a Hacker to discover exploits in the frozen reward model $R _ { \theta }$ while an Auditor learns to detect these exploits from the reward model’s internal representations. The Hacker is initialized from the same supervised fine-tuned model $\pi _ { \mathrm { S F T } }$ that will later be optimized in Stage $^ { 2 , }$ ensuring it explores the same exploitation strategies the policy would naturally discover during RLHF. Once the Auditor is trained, we proceed to Stage 2 (Auditor-Guided RLHF), where the trained Auditor gates reward signals during standard policy optimization—responses flagged as exploitative receive suppressed rewards, making hacking strategies unprofitable. Throughout both stages, the reward model $R _ { \theta }$ remains frozen, ensuring the Auditor is calibrated to a fixed feature space.

## 3.1. Stage 1: The Hacker-Auditor Game

We formulate an adversarial game between a Hacker policy $H _ { \psi }$ and an Auditor network $A _ { \xi }$ . The Hacker, initialized from $\pi _ { \mathrm { S F T } } .$ , learns to generate responses that achieve high proxy rewards by exploiting spurious correlations in $R _ { \theta }$ . The Auditor learns to distinguish genuinely aligned responses from exploitative ones.

## 3.1.1. THE AUDITOR

The Auditor $A _ { \xi }$ is designed to distinguish genuinely aligned responses from those that exploit $R _ { \theta }$ . Since both may yield high rewards, discrimination requires analyzing the internal mechanism of reward generation rather than reward values alone. Our key insight is that exploitation manifests distinctly in the reward model’s latent space: models trained on finite preference data inevitably encode both task-relevant features and spurious correlates, and exploitative responses activate the latter disproportionately. Let $R _ { \theta }$ decompose into a feature extractor $f _ { \theta }$ and a reward head, and define $h _ { x , y } = f _ { \theta } ( x , y ) \in \mathbb { R } ^ { d }$ as the penultimate layer activation. The Auditor is a multilayer perceptron $A _ { \xi } : \mathbb { R } ^ { d }  [ 0 , 1 ]$ that takes $h _ { x , y }$ as input and estimates P(genuine $\mid h _ { x , y } )$ the probability that response y reflects genuine alignment rather than exploitation.

Training the Auditor requires labeled examples of both classes. Positive samples $\mathcal { D } ^ { + } = \{ ( x _ { i } , y _ { i } ^ { w } ) \in \mathcal { D } _ { \mathrm { p r e f } } \}$ consist of preferred responses from the original preference dataset, representing the ground-truth distribution that $R _ { \theta }$ was trained to model. Negative samples $\mathcal { D } ^ { - }$ are drawn from two sources: responses generated by the current Hacker $H _ { \psi }$ , and historical exploits stored in a replay buffer B. While both $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - }$ contain high-reward responses, they achieve reward through different activation patterns: $\mathcal { D } ^ { + }$ activates features aligned with human judgment, whereas $\mathcal { D } ^ { - }$ increasingly activates spurious features as the Hacker learns to exploit. This activation-space divergence is initially small but grows throughout training, providing the Auditor with progressively stronger signal. We maintain B as a fixedsize buffer, adding only Hacker responses that achieve high proxy reward $( R _ { \theta } > \tau _ { R } )$ and successfully evade the current Auditor $( A _ { \xi } > \tau _ { A } )$ . When the buffer reaches capacity, we remove examples that the Auditor now easily detects, keeping the buffer focused on hard negatives. During training, we sample negatives as a mixture: $( 1 - \alpha )$ from the current Hacker and $\alpha$ from $B ,$ with buffer samples prioritized by Auditor confidence to focus on cases the Auditor still fails to detect. To prevent the Auditor from learning superficial heuristics rather than exploitation signatures, all prompts for $\mathcal { D } ^ { - }$ are sampled from the same distribution as $\mathcal { D } ^ { + }$

The Auditor minimizes: $\begin{array} { r } { \mathcal { L } _ { A } ( \xi ; \psi ) = \mathcal { L } _ { \mathrm { B C E } } ( \xi ) + \lambda _ { C } \mathcal { L } _ { \mathrm { C o n } } ( \xi ) } \end{array}$ where the dependence on ψ arises because $\mathcal { D } ^ { - }$ contains samples from the current Hacker $H _ { \psi }$ . The binary crossentropy loss is:

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{BCE}} (\xi) = - \mathbb {E} _ {(x, y ^ {+}) \sim \mathcal {D} ^ {+}} \left[ \log A _ {\xi} (h _ {x, y ^ {+}}) \right] \\ - \mathbb {E} _ {(x, y ^ {-}) \sim \mathcal {D} ^ {-}} \left[ \log (1 - A _ {\xi} (h _ {x, y ^ {-}})) \right] \end{array}
$$

The supervised contrastive loss operates on normalized embeddings $z _ { i } = g _ { \xi } ( h _ { x _ { i } , y _ { i } } ) / \| g _ { \xi } ( h _ { x _ { i } , y _ { i } } ) \|$ from the Auditor’s penultimate layer $g _ { \xi } ,$ which transforms the reward model representations $h \ = \ f _ { \theta } ( x , y )$ into a space optimized for exploitation detection:

$$
\mathcal {L} _ {\mathrm{Con}} (\xi) = \sum_ {i \in \mathcal {B}} \frac {- 1}{| S (i) |} \sum_ {s \in S (i)} \log \frac {\exp (z _ {i} \cdot z _ {s} / \tau)}{\sum_ {a \neq i} \exp (z _ {i} \cdot z _ {a} / \tau)}
$$

where $\boldsymbol { B }$ is a mini-batch, $S ( i ) = \{ s \in \mathcal { B } : s \neq i , c _ { s } = c _ { i } \}$ is the set of samples sharing the same class label $c _ { i } \in$ {genuine, exploit} as sample i, and τ is a temperature hyperparameter. The contrastive term encourages geometrically structured representations where genuine responses cluster together and exploits cluster separately, improving robustness to the Hacker’s evolving strategies.

## 3.1.2. THE HACKER

The Hacker $H _ { \psi }$ , initialized from π , is trained to discover exploits in the reward model by maximizing proxy reward while evading Auditor detection. We optimize:

$$
\begin{array}{c} \mathcal {J} _ {H} (\psi ; \xi) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim H _ {\psi} (\cdot | x)} \Big [ R _ {\theta} (x, y) + \lambda_ {A} \log A _ {\xi^ {*}} (h _ {x, y}) \\ - \beta_ {H} D _ {\mathrm{KL}} (H _ {\psi} \| \pi_ {\mathrm{SFT}}) \Big ] \end{array}
$$

using Proximal Policy Optimization (PPO), where $R _ { \theta } ( x , y ) + \lambda _ { A }$ log $A _ { \xi ^ { * } } ( h _ { x , y } )$ serves as the reward signal. The three terms serve complementary purposes. The proxy reward $R _ { \theta } ( x , y )$ incentivizes the Hacker to find high-reward responses, including those that exploit spurious correlations. The evasion term $\lambda _ { A }$ log $A _ { \xi ^ { * } } ( h _ { x , y } )$ penalizes responses that the Auditor identifies as exploitative: since $A _ { \xi ^ { * } } \in [ 0 , 1 ]$ estimates the probability of genuine alignment, log $A _ { \xi ^ { * } }$ is negative when the Auditor suspects exploitation and approaches zero for responses deemed genuine. The KL regularization $\beta _ { H } D _ { \mathrm { K L } } ( H _ { \psi } \Vert \pi _ { \mathrm { S F T } } )$ constrains the Hacker to remain within a plausible language distribution, preventing degenerate outputs that trivially achieve high reward. The target Auditor $A _ { \xi ^ { * } }$ ∗ is a Polyak-averaged copy of the Auditor, updated as $\xi ^ { * }  \rho \xi ^ { * } + ( 1 - \rho ) \xi$ after each Auditor update with $\rho = 0 . 9 9 5$ . This exponential moving average provides a stable optimization target: the Hacker optimizes against a slowly-evolving Auditor rather than the current one, preventing oscillatory dynamics where the Hacker overfits to transient Auditor states.

## 3.1.3. GAME DYNAMICS AND STABILIZATION

The coupled optimization—RL for the Hacker, supervised learning for the Auditor—requires careful stabilization: if the Auditor advances too quickly, the Hacker receives uninformative gradients; if too slowly, it fails to detect emerging exploits. We employ three mechanisms to balance learning dynamics.

Two-Phase Update Schedule. We adopt a two-phase schedule to balance learning speeds. In Phase 1 (warmup), for the first $T _ { \mathrm { w a r m } }$ steps, we update the Auditor at fixed intervals— once every K Hacker PPO updates—allowing both players to establish initial strategies. In Phase 2 (confidence-gated), we switch to adaptive updates: let A<sup>¯</sup> denote the moving average of Auditor confidence on recent Hacker outputs over a window of m steps. We update the Auditor only when $\bar { A } > \tau _ { \mathrm { g a t e } }$ , indicating the Hacker is successfully evading detection; otherwise, the Auditor remains frozen to let the Hacker catch up. This gating prevents the Auditor from overpowering the Hacker prematurely.

Replay Buffer. We maintain a buffer B of historical exploits, sampling negatives as a mixture: $( 1 - \alpha )$ from the current Hacker and α from B. This implements a form of fictitious play—the Auditor learns to detect the average historical strategy, not just the current one—preventing catastrophic forgetting of past vulnerabilities.

Target Network. The Hacker optimizes against a Polyakaveraged target Auditor $A _ { \xi ^ { * } }$ , updated as $\xi ^ { * } \gets \rho \xi ^ { * } { + } ( 1 { - } \rho ) \xi$ with $\rho = 0 . 9 9 5$ . This smooths the Hacker’s learning signal, preventing overfitting to transient Auditor states.

## 3.2. Stage 2: Auditor-Guided RLHF (AG-RLHF)

Once Stage 1 produces a trained Auditor $A _ { \xi }$ that reliably distinguishes genuine quality from exploitation, we deploy it to guide standard RLHF training. Rather than optimizing the proxy reward $R _ { \theta }$ directly, AG-RLHF gates the reward signal based on the Auditor’s confidence that a response

reflects genuine alignment:

$$
R _ {\text { gated }} (x, y) = R _ {\theta} (x, y) \cdot A _ {\xi} (h _ {x, y}) ^ {\gamma}
$$

where $\gamma > 0$ controls gating severity. When the Auditor is confident that a response is genuine $( A _ { \xi } \approx 1 )$ , the gated reward approaches the full proxy reward. When the Auditor detects exploitation $( A _ { \xi } \approx 0 )$ , the gated reward is suppressed toward zero regardless of how high the proxy reward is. The exponent γ modulates this tradeoff: higher values impose stricter penalties on suspected exploits, while lower values are more permissive.

The policy $\pi _ { \phi } .$ , initialized from $\pi _ { \mathrm { S F T } } ,$ is optimized via:

$$
\mathcal {J} _ {\mathrm{AG-RLHF}} (\phi) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\phi}} \left[ R _ {\mathrm{gated}} (x, y) - \beta D _ {\mathrm{KL}} (\pi_ {\phi} \| \pi_ {\mathrm{SFT}}) \right]
$$

using PPO. The Auditor remains frozen throughout Stage 2. This separation is deliberate: the Auditor was trained adversarially against a Hacker explicitly optimizing for evasion, making it robust to the milder exploitation pressures that arise during standard policy optimization. Freezing also ensures stable reward signals—if the Auditor continued adapting, the policy would chase a moving target, destabilizing training. We provided the details of hyperparameter selection in Appendix A.1.

## 4. Experimental Setup

Tasks and Datasets. To comprehensively evaluate ARA, we cover three known reward hacking scenarios: (1) Sycophancy: We train the reward model on Anthropic HH-RLHF (Bai et al., 2022) and measure Sycophancy Rate on SycophancyEval (Sharma et al., 2025), measuring whether models maintain correct answers when users express disagreement. (2) Length Bias: we follow (Chen et al., 2024) setting and measure average response length and ROUGE-L. (3) Code Gaming: We construct a coding environment enabling the reward hacking behaviors using the unit test dataset from (?), models receive problems with visible test cases and can either (i) implement general solutions or (ii) exploit test visibility through hardcoding outputs, manipulating assertions. We measure alignment through GPT-4-classified Gaming Rate (fraction of solutions exploiting rather than solving) and utility through Pass@1 on held-out tests invisible during training.

Baselines. We comprehensively compare against methods spanning four mitigation categories: (1) Unmitigated SFT: Supervised fine-tuned model without RLHF (preoptimization baseline). (2) PPO-KL Regularization: Explicit KL penalty constraining drift from SFT (Chen et al., 2024; Miao et al., 2024). (3) ODIN (Chen et al., 2024): Orthogonal disentanglement via dual reward heads (4) In-FoRM (Miao et al., 2024): Variational information bottleneck filtering task-irrelevant features. (5) Reward Model

Improvements with Enesemble: Averages n independently trained reward models to reduce exploitable variance (Eisenstein et al., 2023; Miao et al., 2024). (6) Training Interventions: Removes high-reward outliers $( R > \mu + 2 \sigma )$ following scaling law findings of (Gao et al., 2023). The SFT model (i.e., Llama-2-7B (Touvron et al., 2023; hug) serves as the baseline policy $\pi _ { 0 }$ . Each experiment is repeated with three random seeds; we report mean ± 95% confidence intervals.

## 5. Result and Discussion

## 5.1. Reward Hacking Mitigation with ARA

Table 1 summarizes results across all three hacking scenarios, where ARA consistently achieves the best alignment—utility tradeoff. (1) For sycophancy, standard PPO increases the rate from 36.2% (SFT) to 72.4% while improving helpfulness by 35.5%, confirming that optimizing the proxy reward improves surface metrics at the cost of true alignment. Regularization methods (PPO-KL, ODIN, InFoRM) reduce sycophancy to 47–58% but sacrifice helpfulness, whereas ARA significantly reduces sycophancy to 38.4% while improving helpfulness to 77.2%. (2) For length bias, PPO inflates response length from 148 to 347 tokens (+134%) with minimal ROUGE-L improvement (+1.7 points)—a clear signature of verbosity exploitation. ODIN and InFoRM, designed specifically for this setting, achieve 195–208 tokens, while ARA reduces length to 162 tokens and achieves the highest ROUGE-L (24.1), indicating that the Auditor learned to distinguish verbose padding from substantive content. (3) Code gaming presents the most sophisticated exploitation, where models manipulate unit tests rather than solving problems. PPO achieves a 61.3% gaming rate, and existing methods only reduce this to 40– 47%. ARA achieves a 19.6% gaming rate while improving Pass@1 to 35.8%, demonstrating that suppressing exploitation redirects optimization toward genuine problem-solving.

Goodhart Divergence Prevention. Figure 2 visualizes the training dynamics of proxy and task-specific performance across all three scenarios. Standard PPO exhibits the classic Goodhart pattern: proxy reward increases throughout training while gold reward peaks early and then declines, indicating that continued optimization actively degrades true performance. The timing of this divergence varies by scenario. Code gaming shows the earliest divergence at around 1,000 steps because test-gaming strategies are discovered and exploited quickly, leading to earlier saturation. Length bias diverges around step 3,500, reflecting a period where increased length initially correlates with quality before verbosity dominates. Sycophancy diverges latest at step 4,000, suggesting that agreement-seeking behavior emerges more gradually. In contrast, ARA maintains substantially better alignment between proxy and gold rewards throughout training, suggesting that Auditor-gated rewards successfully reduce reward over-optimization pressure toward genuine quality improvements rather than reward exploitation.

Submission and Formatting Instructions for ICML 2026

<table><tr><td rowspan="2">METHOD</td><td colspan="2">SYCOPHANCY</td><td colspan="2">LENGTH BIAS</td><td colspan="2">CODE GAMING</td></tr><tr><td>SYCOPHANCY ↓</td><td>HELPFULNESS ↑</td><td>AVG. LENGTH ↓</td><td>ROUGE-L ↑</td><td>GAMING RATE ↓</td><td>PASS@1 ↑</td></tr><tr><td>SFT (No RLHF)</td><td>36.2</td><td>41.3</td><td>148</td><td>21.4</td><td>4.2</td><td>28.5</td></tr><tr><td>Unmitigated PPO</td><td>72.4</td><td>76.8</td><td>347</td><td>23.1</td><td>61.3</td><td>34.2</td></tr><tr><td colspan="7">Regularization Methods</td></tr><tr><td>PPO w/KL</td><td>58.3</td><td>68.2</td><td>268</td><td>22.8</td><td>48.5</td><td>31.8</td></tr><tr><td>ODIN</td><td>51.6</td><td>63.4</td><td>195</td><td>23.4</td><td>42.1</td><td>33.5</td></tr><tr><td>INFORM</td><td>47.2</td><td>62.1</td><td>208</td><td>23.2</td><td>39.8</td><td>33.1</td></tr><tr><td colspan="7">Reward Model Improvements</td></tr><tr><td>RM ENSEMBLE</td><td>52.8</td><td>64.6</td><td>224</td><td>23.0</td><td>44.3</td><td>34.0</td></tr><tr><td colspan="7">Training Interventions</td></tr><tr><td>FILTERING ( $R > \mu + 2\sigma$ )</td><td>55.1</td><td>50.3</td><td>241</td><td>22.6</td><td>46.7</td><td>32.4</td></tr><tr><td>ARA(Ours)</td><td>38.4</td><td>77.2</td><td>162</td><td>24.1</td><td>19.6</td><td>35.8</td></tr></table>

Table 1. Main Results. Comparison of AG-RLHF against baselines across three reward hacking scenarios. We report alignment and utility metrics and utility metrics. Results averaged over 3 seeds.



Figure 2. ARA mitigates Goodhart’s Law during RLHF optimization. Proxy reward (solid) is the learned reward model score; gold reward (dashed) measures true task-specific performance—GPT-4 factual accuracy for sycophancy (given the golden answer), ROUGE-L for length bias, and Pass@1 on held-out tests for code gaming.

## 5.2. Hacker’s Reward Hacking Dynamics

Figure 3 compares the temporal dynamics of three reward hacking behaviors, each measured during training on its respective task. During steps 0–1,500, all three metrics remain near baseline as models learn basic task competence before discovering exploitable patterns. Beginning at step 1,500, all three hacking behaviors start to emerge: code gaming rises most steeply, while chat sycophancy and length bias follow similar trajectories. This synchronized onset across independent training runs suggests that reward hacking emerges at a consistent phase of optimization, regardless of the specific exploit type. The critical transition occurs around step 4,000, where code gaming saturates at approximately 59% having exhausted available exploits, while chat sycophancy and length bias continue climbing. The parallel trajectories across distinct hacking scenarios suggest a common underlying mechanism: models first acquire task competence, then discover exploitation strategies at a characteristic optimization phase, with saturation rates varying by the complexity of available exploits.

Figure 3. Reward hacking emerges across tasks

## 5.3. Auditor’s Performance Evaluation

A core aspect of our framework is that the Auditor can distinguish whether high reward scores reflect genuine quality or exploitation. The critical test is whether the Auditor succeeds where proxy reward fails: distinguishing genuinely high-quality responses from high-reward exploits. We construct a balanced evaluation set of 500 genuine responses (from D<sup>+</sup>) and 500 exploitative responses (from known hacked responses), all filtered to have comparable high proxy rewards $( R _ { \theta } ~ > ~ 0 . 8 5 )$ . As shown in Table 2, we compare three detection approaches. The learned reward model cannot distinguish the two groups—by construction, both achieve similarly high scores (0.91 vs 0.89, ∆ = 0.02). We also prompt GPT-4 to classify whether each response is genuine or exploitative based on the text alone, outputting a score from 0 (likely exploit) to 1 (likely genuine). GPT-4 achieves moderate separation $( \Delta = 0 . 4 4 )$ , confirming that exploits are detectable from surface features but not trivially so. The Auditor achieves the strongest separation $( \Delta = 0 . 6 3 )$ : on average, it assigns probability 0.82 to genuine responses and only 0.19 to exploits, indicating that the Auditor is highly confident in distinguishing the two groups.

<table><tr><td>DETECTION METHOD</td><td>GENUINE</td><td>EXPLOIT</td><td> $\Delta$ </td></tr><tr><td>PROXY REWARD ( $R_{\theta}$ )</td><td>0.91</td><td>0.89</td><td>0.02</td></tr><tr><td>GPT-4</td><td>0.78</td><td>0.34</td><td>0.44</td></tr><tr><td>AUDITOR CONFIDENCE</td><td>0.82</td><td>0.19</td><td>0.63</td></tr></table>

Table 2. Genuine vs. exploitative response detection. Mean scores (probability) for 500 genuine responses and 500 high-reward exploits, all with comparable proxy rewards $( R _ { \theta } > 0 . 8 \bar { 5 } )$ .
Figure 4. Representation structure of exploitation types. t-SNE of reward model hidden states $h ( x , y )$ for exploitative responses.

## 5.4. Cross-Domain Generalization of Reward Hacking

## 5.4.1. REWARD MODEL MISALIGNMENT STRUCTURE

Figure 4 visualizes reward model representations $h _ { x , y }$ for exploitative responses across three hacking types using t-SNE. The visualization reveals unexpected structure: chat sycophancy and length bias occupy nearly identical regions of representation space, while code gaming forms a distinct but proximate cluster. This geometry suggests that sycophancy and verbosity exploit similar features in the reward model—perhaps both manipulate surface-level indicators of helpfulness or engagement—while code gaming employs a mechanistically different strategy involving test-case manipulation.

## 5.4.2. HACKER GENERALIZATION

We examine whether the Hacker exhibits the emergent misalignment patterns across known hacks. We train the Hacker on a single domain until convergence, then evaluate its hacking rate on all three domains without any additional training. Table 3 shows hacking rates when the Hacker is trained on one domain and evaluated on others. The off-diagonal entries reveal that reward hacking does not remain domainspecific: training on code gaming alone increases chat sycophancy from 36.2% to 58.7% (+22.5 points) despite the Hacker never being rewarded for sycophantic behavior or chat datasets. The reverse also holds—training on hackable chat environment increases code gaming from 4.2% to 28.3%—confirming bidirectional transfer between semantically distinct exploitation strategies. Among singledomain conditions, code gaming produces the strongest cross-domain generalization (58.7% chat, 31.4% summarization). Training on all three domains amplifies these effects: combined hacking rates exceed any single-domain condition, suggesting that diverse exploitation experience compounds rather than competes, accelerating the acquisition of general reward-seeking behavior. This observation aligns with recent findings from MacDiarmid et al. (2025), who demonstrate that reward hacking in production RL systems leads to emergent misalignment that generalizes beyond the original exploitation context.

<table><tr><td rowspan="2">TRAIN DOMAIN</td><td colspan="3">EVAL DOMAIN</td></tr><tr><td>SYCOPHANCY</td><td>LENGTH BIAS</td><td>CODE GAMING</td></tr><tr><td>NONE (SFT)</td><td>36.2</td><td>8.1</td><td>4.2</td></tr><tr><td>SYCOPHANCY</td><td>72.4</td><td>35.2</td><td>28.3</td></tr><tr><td>LENGTH BIAS</td><td>41.8</td><td>52.3</td><td>22.6</td></tr><tr><td>CODE GAMING</td><td>58.7</td><td>31.4</td><td>61.3</td></tr><tr><td>ALL THREE</td><td>68.9</td><td>78.1</td><td>59.7</td></tr></table>

Table 3. Cross-Domain Generalization of Hacking. Hacking rate (%) on held-out domains after training on a single source domain. Diagonal entries (gray) show in-domain performance.

## 5.4.3. AUDITOR’S REWARD HACKING DETECTION GENERALIZATION

Table 4 reports the Auditor’s detection AUC for both indomain performance and cross-domain transfer. The transfer behavior closely mirrors the underlying representational geometry. An Auditor trained on chat sycophancy attains 0.77 AUC when transferred to length-bias detection—approaching its in-domain performance—whereas transfer to code gaming is markedly weaker (0.58 AUC). The pattern is symmetric: an Auditor trained on length bias achieves 0.75 AUC on chat sycophancy but only 0.55 on code gaming. In contrast, an Auditor trained on code gaming exhibits only modest transfer to both chat and length (0.59–0.62 AUC), consistent with code gaming occupying a more distinct region in representation space. These findings have practical implications. When sycophancy or verbosity are the primary concerns, training on either domain provides reasonable coverage of both. By comparison, mitigating code gaming likely requires dedicated training data, as its exploitation mechanism is sufficiently distinct that crossdomain generalization is limited.



Figure 5. Gating severity sensitivity analysis. Hacking metrics (red, lower is better) and utility metrics (blue, higher is better).

<table><tr><td rowspan="2">TRAIN DOMAIN</td><td colspan="3">TEST DOMAIN</td></tr><tr><td>SYCOPHANCY</td><td>LENGTH BIAS</td><td>CODE GAMING</td></tr><tr><td>SYCOPHANCY</td><td>0.82</td><td>0.77</td><td>0.58</td></tr><tr><td>LENGTH BIAS</td><td>0.75</td><td>0.79</td><td>0.55</td></tr><tr><td>CODE GAMING</td><td>0.62</td><td>0.59</td><td>0.84</td></tr></table>

Table 4. Auditor Transfer Matrix. Detection AUC when training the Auditor on one domain and testing on others.

## 5.4.4. ARA’S REWARD HACKING MITIGATIONGENERALIZATION

We further evaluate the cross-domain generalization of reward-hacking mitigation through ARA’s two-stage training procedure. As shown in Table 5, the transfer behavior closely mirrors the Auditor detection results in Table 4. Sycophancy and length bias exhibit strong bidirectional transfer: ARA trained on sycophancy reduces response length from 347 to 178 tokens (49% reduction), while ARA trained on length bias reduces sycophancy from 72.4% to 42.1%. Transfer involving code gaming is weaker but still meaningful: ARA trained on sycophancy lowers the gaming rate from 61.3% to 48.2%, and ARA trained on code gaming reduces sycophancy from 72.4% to 54.8%. Although these cross-domain gains fall short of in-domain performance, they provide substantial mitigation relative to unmitigated PPO, indicating partial overlap in exploitation signatures even across mechanistically distinct hacking strategies. Finally, training on all three domains yields nearoptimal mitigation across metrics (39.2% sycophancy, 165 tokens, 21.4% gaming rate), with only marginal degradation compared to domain-specific training.

<table><tr><td rowspan="2">TRAIN DOMAIN</td><td colspan="3">TEST DOMAIN (HACKING METRIC)</td></tr><tr><td>SYCO. (%) ↓</td><td>LENGTH BIAS ↓</td><td>CODE GAMING (%) ↓</td></tr><tr><td>PPO</td><td>72.4</td><td>347</td><td>61.3</td></tr><tr><td>SYCOPHANCY</td><td>38.4</td><td>178</td><td>48.2</td></tr><tr><td>LENGTH BIAS</td><td>42.1</td><td>162</td><td>51.6</td></tr><tr><td>CODE GAMING</td><td>54.8</td><td>246</td><td>19.6</td></tr><tr><td>ALL THREE</td><td>39.2</td><td>165</td><td>21.4</td></tr></table>

Table 5. Cross-Domain Mitigation Transfer. Hacking metrics when ARA is trained on one domain and deployed on others. Diagonal entries (bold) show in-domain performance.

## 5.5. AG-RLHF Gating Severity

Figure 5 shows performance across gating severity $\gamma \in$ [0, 8], where $\gamma = 0$ recovers standard PPO. Sycophancy and length bias achieve optimal utility at $\gamma = 2 ,$ while code gaming requires stronger suppression at $\gamma = 3 ,$ , likely because test manipulation represents more severe exploitation. Notably, utility improves as γ increases from 0 to the optimum, confirming that suppressing spurious shortcuts redirects optimization toward genuine task performance rather than merely trading off against it.

## 6. Ablation Studies

We examine how Auditor capacity affects detection and mitigation by varying model size from 5M to 85M parameters. Different exploitation types require different capacities: length bias saturates at 25M, sycophancy at 35M, and code gaming at 50M—reflecting increasing complexity from surface-level features to subtle test-manipulation patterns. Beyond these optimal points, larger Auditors provide marginal improvements (details in Appendix A.2.1).

## 7. Conclusion

We introduced Adversarial Reward Auditing (ARA), a framework that formulates reward hacking as a competitive game between a Hacker that discovers exploits and an Auditor that detects them, transforming exploitation from an unobservable failure into a measurable signal. Experiments across sycophancy, length bias, and code gaming demonstrate that ARA achieves the best alignment-utility tradeoff among all baselines, reducing hacking while improving task performance. Beyond single-domain evaluation, we show that reward hacking, detection, and mitigation all generalize across domains, enabling efficient multi-domain defense with a single Auditor. These findings suggest that adversarial auditing offers a promising direction for building RLHF systems robust to novel exploitation strategies.

## Impact Statement

This work aims to improve AI safety by detecting and mitigating reward hacking in RLHF systems. While the Hacker component discovers exploits, these vulnerabilities are already documented in prior work, and our primary contribution is the defensive Auditor framework. We believe the benefits of understanding and mitigating reward hacking outweigh potential dual-use concerns.

## References

meta-llama/Llama-2-7b · Hugging Face — huggingface.co. https://huggingface.co/meta-llama/ Llama-2-7b. [Accessed 02-01-2026].

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Mane, D. Concrete problems in ´ ai safety, 2016. URL https://arxiv.org/abs/ 1606.06565.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J., and Farhi, D. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation, 2025. URL https: //arxiv.org/abs/2503.11926.

Beigi, M., Shen, Y., Shojaee, P., Wang, Q., Wang, Z., Reddy, C. K., Jin, M., and Huang, L. Sycophancy mitigation through reinforcement learning with uncertainty-aware adaptive reasoning trajectories. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13090–13103, 2025.

Chen, L., Zhu, C., Soselia, D., Chen, J., Zhou, T., Goldstein, T., Huang, H., Shoeybi, M., and Catanzaro, B. Odin: Disentangled reward mitigates hacking in rlhf. arXiv preprint arXiv:2402.07319, 2024.

Denison, C., MacDiarmid, M., Barez, F., Duvenaud, D., Kravec, S., Marks, S., Schiefer, N., Soklaski, R., Tamkin, A., Kaplan, J., Shlegeris, B., Bowman, S. R., Perez, E., and Hubinger, E. Sycophancy to subterfuge: Investigating reward-tampering in large language models, 2024. URL https://arxiv.org/abs/2406.10162.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., and Madry, A. Implementation matters in deep policy gradients: A case study on ppo and trpo, 2020. URL https://arxiv.org/abs/2005.12729.

Everitt, T., Hutter, M., Kumar, R., and Krakovna, V. Reward tampering problems and solutions in reinforcement learning: A causal influence diagram perspective, 2021. URL https://arxiv.org/abs/1908.04734.

Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, November 2020. ISSN 2522-5839. doi: 10.1038/s42256-020-00257-z. URL http://dx. doi.org/10.1038/s42256-020-00257-z.

Liu, T., Xiong, W., Ren, J., Chen, L., Wu, J., Joshi, R., Gao, Y., Shen, J., Qin, Z., Yu, T., et al. Rrm: Robust reward model training mitigates reward hacking. arXiv preprint arXiv:2409.13156, 2024.

MacDiarmid, M., Wright, B., Uesato, J., Benton, J., Kutasov, J., Price, S., Bouscal, N., Bowman, S., Bricken, T., Cloud, A., Denison, C., Gasteiger, J., Greenblatt, R., Leike, J., Lindsey, J., Mikulik, V., Perez, E., Rodrigues, A., Thomas, D., Webson, A., Ziegler, D., and Hubinger, E. Natural emergent misalignment from reward hacking in production rl, 2025. URL https: //arxiv.org/abs/2511.18397.

Miao, Y., Zhang, S., Ding, L., Bao, R., Zhang, L., and Tao, D. Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling, 2024. URL https://arxiv.org/abs/2402.09345.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., and Perez, E. Towards understanding sycophancy in language models, 2025. URL https://arxiv.org/abs/2310.13548.

Singhal, P., Goyal, T., Xu, J., and Durrett, G. A long way to go: Investigating length correlations in rlhf, 2024. URL https://arxiv.org/abs/2310.03716.

Skalse, J., Howe, N. H. R., Krasheninnikov, D., and Krueger, D. Defining and characterizing reward hacking, 2025. URL https://arxiv.org/abs/2209.13085.

Taylor, M., Chua, J., Betley, J., Treutlein, J., and Evans, O. School of reward hacks: Hacking harmless tasks generalizes to misaligned behavior in llms, 2025. URL https://arxiv.org/abs/2508.17511.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Wen, J., Zhong, R., Khan, A., Perez, E., Steinhardt, J., Huang, M., Bowman, S. R., He, H., and Feng, S. Language models learn to mislead humans via rlhf, 2024. URL https://arxiv.org/abs/2409.12822.

Zheng, R., Dou, S., Gao, S., Hua, Y., Shen, W., Wang, B., Liu, Y., Jin, S., Liu, Q., Zhou, Y., Xiong, L., Chen, L., Xi, Z., Xu, N., Lai, W., Zhu, M., Chang, C., Yin, Z., Weng, R., Cheng, W., Huang, H., Sun, T., Yan, H., Gui, T., Zhang, Q., Qiu, X., and Huang, X. Secrets of rlhf in large language models part i: Ppo, 2023. URL https://arxiv.org/abs/2307.04964.

## A. Appendix

## A.1. Implementation Details

Table 6. Hyperparameters.

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="6">Hacker (PPO)</td><td>Learning rate</td><td> $1 \times 10^{-6}$ </td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>PPO epochs</td><td>4</td></tr><tr><td>Clip ratio  $\epsilon$ </td><td>0.2</td></tr><tr><td>Evasion weight  $\lambda_A$ </td><td>0.5</td></tr><tr><td>KL penalty  $\beta_H$ </td><td>0.1</td></tr><tr><td rowspan="5">Auditor (MLP)</td><td>Input dimension</td><td>4,096</td></tr><tr><td>Hidden layers</td><td>3–4</td></tr><tr><td>Hidden dimension</td><td>2,048–4,096</td></tr><tr><td>Activation</td><td>GELU</td></tr><tr><td>Parameters</td><td>25M–50M</td></tr><tr><td rowspan="6">Game Dynamics</td><td>Polyak coefficient  $\rho$ </td><td>0.995</td></tr><tr><td>Hacker updates per cycle  $K$ </td><td>5</td></tr><tr><td>Auditor updates per cycle  $M$ </td><td>1</td></tr><tr><td>Warmup steps  $T_{warm}$ </td><td>200</td></tr><tr><td>Evasion threshold  $\tau_{evade}$ </td><td>0.5</td></tr><tr><td>Replay buffer size  $|B|$ </td><td>1,000</td></tr><tr><td rowspan="4">AG-RLHF</td><td>Learning rate</td><td> $5 \times 10^{-7}$ </td></tr><tr><td>KL penalty  $\beta$ </td><td>0.05</td></tr><tr><td>Gating severity  $\gamma$  (Syco./Length)</td><td>2.0</td></tr><tr><td>Gating severity  $\gamma$  (Code)</td><td>3.0</td></tr><tr><td rowspan="3">Infrastructure</td><td>GPUs</td><td> $8 \times A40$  (48GB)</td></tr><tr><td>Training time</td><td>~72 hours</td></tr><tr><td>Random seeds</td><td>3</td></tr></table>

## A.2. Ablation Study

## A.2.1. AUDITOR SIZE

Figure 6 shows how Auditor capacity affects both detection and mitigation across all three scenarios. Under-capacity Auditors (5M–10M) achieve weak detection AUC (0.65–0.76) and correspondingly poor mitigation, leaving substantial hacking behavior. Interestingly, different exploitation types require different Auditor capacities: length bias achieves strong performance at 25M, sycophancy requires 35M, while code gaming benefits from additional capacity up to 50M before saturating. This ordering reflects exploitation complexity—length bias involves surface-level features easily captured by smaller models, whereas code gaming requires detecting more subtle test-manipulation patterns. Our chosen configurations (marked in gold) provide strong detection (0.79–0.84 AUC) and effective mitigation across all tasks. Larger Auditors (85M, shaded region) provide marginal improvements (<2%) despite additional parameters, indicating diminishing returns.






Figure 6. Auditor size ablation across all scenarios. Top row: Detection AUC. Bottom row: Mitigation performance (red = hacking metric ↓, blue = utility metric ↑). Gold markers indicate optimal configuration for each task. Different tasks require different capacities: length bias saturates at 25M, sycophancy at 35M, and code gaming at 50M, reflecting increasing exploitation complexity.
