# A Unifying Lens on Reward Uncertainty in RLHF

Ely Hahami <sup>1</sup> <sup>\*</sup> Yoel Zimmermann <sup>2</sup> <sup>\*</sup> Ray Zhou <sup>1</sup> Jack Benarroch Jedlicki <sup>2</sup>

## Abstract

Reinforcement learning from human feedback (RLHF) is bottlenecked by reward hacking, where the policy exploits errors in a proxy reward model (RM) and produces high RM scores without genuine quality gains. A natural mitigation is pessimism: lowering rewards in regions where the RM is uncertain. However, standard scalar RMs provide no principled notion of uncertainty. We argue that the right object is a distributional reward model p(r | x, y). Under either a Bayesian inference or a KL-distributionally robust optimization (KL-DRO) lens, the KL-regularized RLHF objective admits a closed-form effective reward $\widetilde { r } ( x , y ) ~ = ~ \pm \beta \log { \mathbb { E } _ { p } [ e ^ { \pm r / \beta } ] }$ The pessimistic branch unifies the prior heuristics for RM ensemble aggregation: mean aggregation, worst-case optimization (WCO), and uncertainty-weighted optimization (UWO) all emerge as limits or truncations of this single expression. This also clarifies the implicit assumptions of each existing rule.

## 1. Introduction

Reinforcement learning from human feedback (RLHF) is the dominant paradigm for aligning large language models with user intent (Christiano et al., 2017; Ouyang et al., 2022; Bai et al., 2022; Stiennon et al., 2020). Pairwise preferences are first used to fit a reward model (RM) and a policy is then optimized against the RM with a KL penalty to a reference policy, via policy-gradient methods such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024). Because the RM is an imperfect proxy for human preferences, policy optimization can overoptimize into out-of-distribution regions, producing high RM scores without improving true response quality. This is an instance of Goodhart’s law (Gao et al., 2023; Skalse et al., 2022; Amodei et al., 2016; Pan et al., 2022) and has been documented across forms including sycophancy and (Sharma et al., 2025; Perez et al., 2023) length bias (Singhal et al., 2024).

A natural mitigation is pessimism: when the RM is uncertain, a lower effective reward should be assigned. Prior work in RLHF instantiates this idea by training ensembles of RMs and aggregating their outputs in different ways. Coste et al. (2024) introduce three rules: mean aggregation (averaging members), worst-case optimization (WCO, taking the minimum across members), and uncertainty-weighted optimization (UWO, mean minus a tunable multiple of the empirical variance). Eisenstein et al. (2024) make a similar case for ensembles, and follow-up work has explored adversarial (Zhang et al., 2024) and robustness-driven (Yan et al., 2024) variants. These methods empirically help, but are typically presented as heuristics with little principled guidance on which to prefer and how their hyperparameters relate.

We argue here that to properly handle uncertainty, one should replace the scalar RM with a distributional reward model p(r | x, y). Once the RM is distributional, a single principle—either marginalization over reward uncertainty through a Bayesian inference lens (Korbak et al., 2022; Levine, 2018) or KL-distributionally robust optimization (KL-DRO) (Ben-Tal et al., 2009; Hansen & Sargent, 2008)— yields a closed-form effective reward to substitute into the standard KL-RL objective. The aggregation rules used in practice emerge as truncations or limits of this expression. This perspective also connects RLHF pessimism to distributional reinforcement learning (Bellemare et al., 2017; Dabney et al., 2018b;a; Keramati et al., 2020) and to offline-RL pessimism (Jin et al., 2021; Xie et al., 2021; Rigter et al., 2022).

## Contributions.

• A unifying lens. From either a Bayesian inference or KL-DRO perspective, the RLHF objective with reward uncertainty admits a closed-form effective reward $\boldsymbol { \tilde { r } } ( x , y ) = \pm \beta \log \mathbb { E } _ { p } [ e ^ { \pm r / \beta } ]$ (Sec. 3).

• Unification of prior heuristics. Mean, WCO, and UWO emerge as the $\beta \to \infty$ limit, β → 0 limit, and Gaussian truncation, respectively, of the pessimistic branch (Sec. 3.5).

• Implementation guidance. We discuss what each heuristic implicitly assumes about the reward distribution $p ( r \mid x , y )$ , and what is needed (calibration, posterior families) to instantiate the framework with a real RM (Sec. 4).

## 2. Background and Related Work

Reward overoptimization. Gao et al. (2023) established that the proxy RM score and a stronger “gold” RM score (which is used to model the true objective we seek to optimize, i.e., how a human would score the response) diverge as PPO progresses, the canonical signature of reward hacking. Skalse et al. (2022) formalize reward hacking, and Amodei et al. (2016) situate it among broader AI safety concerns. Empirically, hacked policies exhibit sycophancy (Sharma et al., 2025), length inflation (Singhal et al., 2024), and other surface artifacts that the proxy RM cannot distinguish from true quality (Wang et al., 2026).

Mitigating reward hacking with pessimism. Coste et al. (2024) introduce Mean, WCO, and UWO aggregation over RM ensembles and find that pessimistic variants reduce reward hacking. Eisenstein et al. (2024) reach similar conclusions for best-of-n and PPO, while noting that ensembles can share failure modes. Lightweight ensembles share a frozen backbone and diversify only a head (Dwaracherla et al., 2024). Weight averaging the members instead of ensembling them (Rame et al.´ , 2024) trades the distributional signal for robustness and is complementary to our analysis. Pessimism can also be built into the reward model itself, by fine-tuning it to be pessimistic (Xu et al., 2025), or imposed through pessimistic preference-based policy objectives (Gupta et al., 2025). Xu et al. (2026) use RM uncertainty to route hard cases to a stronger LLM judge.

Distributional reward models. An alternative to ensembles is to place a posterior directly over a single head. Bayesian last-layer methods (Yang et al., 2026; Zhai et al., 2023) yield a closed-form Gaussian posterior over the reward $\mathcal { N } ( \mu ( x , y ) , \sigma ^ { 2 } ( x , y ) )$ at each prompt-response pair. Dorka (2024) learns a full, potentially multimodal reward distribution by quantile regression and optimizes its lower quantiles for risk-aware RLHF, and Sun et al. (2025) generalizes the Bradley-Terry likelihood to output a per-pair reward distribution whose uncertainty penalizes the policy. Our framework specifies how to use any such distributional RM in the standard KL-regularized objective.

Control-as-inference and robust RL. The Bayesian view of KL-regularized RL (Korbak et al., 2022; Levine, 2018) interprets policies as posteriors under an “optimality” observation. Robust optimization (Ben-Tal et al., 2009; Hansen & Sargent, 2008) and pessimism in offline RL (Jin et al., 2021;

Xie et al., 2021; Rigter et al., 2022) provide the complementary, adversarial framing. Our derivation shows that, in the RLHF setting, these two seemingly disparate views produce the samefunctionalform up to a sign.

## 3. Theory: A Unifying Lens

## 3.1. Setup

Standard KL-regularized RLHF (Stiennon et al., 2020; Ouyang et al., 2022) maximizes

$$
\max _ {\theta} \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {E} _ {y \sim \pi_ {\theta} (\cdot | x)} [ r (x, y) ] - \beta D _ {\mathrm{KL}} \left(\pi_ {\theta} \| \pi_ {\text {ref}}\right) \right],\tag{1}
$$

with the well-known optimum $\pi ^ { * } ( y | x ) \propto \pi _ { \mathrm { r e f } } ( y |$ $x ) \exp ( r ( x , y ) / \beta )$

In the standard pipeline, $r ( x , y ) = r _ { \phi } ( x , y )$ is a deterministic scalar from a learned RM. We instead treat the reward as a random variable $\boldsymbol { r } \sim p ( \boldsymbol { r } \mid x , y )$ , arising, for example, from a deep ensemble or a Bayesian head, and ask: to best mitigate reward hacking, what scalar effective reward $\tilde { r } ( x , y )$ should replace $r _ { \phi } ( x , y )$ in Eq. 1?

## 3.2. Optimistic Branch: Bayesian Marginalization

Following the control-as-inference view (Korbak et al., 2022; Levine, 2018), we introduce a binary optimality variable $O \in \{ 0 , 1 \}$ with

$$
p (O = 1 \mid x, y, r) = e ^ {r / \beta},\tag{2}
$$

where r is shifted<sup>1</sup> so that sup $r = 0 .$ . Marginalizing over reward uncertainty gives

$$
p (O = 1 \mid x, y) = \int e ^ {r / \beta} p (r \mid x, y) d r = M _ {x} (1 / \beta),\tag{3}
$$

where $M _ { x } ( t ) : = \mathbb { E } _ { p } [ e ^ { t r } ]$ is the moment generating function of $p ( r \mid x , y )$ . Bayes’ rule with prior $\pi _ { \mathrm { r e f } }$ then yields $\pi _ { \mathrm { o p t } } ^ { * } ( y \mid x ) \propto \pi _ { \mathrm { r e f } } ( y \mid x ) M _ { x } ( 1 / \beta )$ . Matching to the KL-RL form π<sup>∗</sup> ∝ π<sub>ref</sub> exp(˜r/β) identifies

$$
\boxed {\tilde {r} _ {\mathrm{opt}} (x, y) = \beta \log \mathbb {E} _ {p (r | x, y)} \left[ e ^ {r / \beta} \right].}\tag{4}
$$

By Jensen’s inequality, since exp is convex,

$$
\tilde {r} _ {\mathrm{opt}} (x, y) \geq \mu (x, y),\tag{5}
$$

so uncertainty inflates the effective reward. This is the Bayesian-optimal use of an uncertain RM under the controlas-inference likelihood in Eq. 2 in the absence of any robustness concern.

## 3.3. Pessimistic Branch: KL-DRO

To mitigate reward hacking we instead want a worst-case effective reward. We let an adversary choose the reward distribution $Q .$ , paying a KL cost to deviate from the believed posterior p (Ben-Tal et al., 2009; Hansen & Sargent, 2008):

$$
\tilde {r} _ {\mathrm{rob}} (x, y) = \inf _ {Q} \left\{\mathbb {E} _ {Q} [ r ] + \beta D _ {\mathrm{KL}} (Q \| p (r \mid x, y)) \right\}.\tag{6}
$$

The parameter $\beta$ controls the adversary’s reach: large $\beta$ pins $Q$ close to $p ,$ small $\beta$ permits arbitrary tilting. Solving the variational problem (see Appendix A) yields the unique optimizer

$$
Q ^ {*} (r) \propto p (r \mid x, y) e ^ {- r / \beta},\tag{7}
$$

which is p exponentially tilted toward low rewards. Plugging back in,

$$
\boxed {\tilde {r} _ {\mathrm{rob}} (x, y) = - \beta \log \mathbb {E} _ {p (r | x, y)} \Big [ e ^ {- r / \beta} \Big ].}\tag{8}
$$

This is the entropic risk measure (Follmer & Schied¨ , 2025) of $p ( r \mid x , y )$ . By Jensen’s inequality, $\tilde { r } _ { \mathrm { r o b } } ( x , y ) \le \mu ( x , y )$ uncertainty deflates the effective reward.

Choice of KL coefficients. The KL coefficient $\beta$ in $\operatorname { E q } .$ . 1 controls the deviation of the policy from the reference, while $\beta$ in Eq. 6 controls the robustness to reward uncertainty. These play distinct roles and need not coincide. In this work we tie the two for simplicity, yielding the simple coefficient $\lambda = 1 / ( 2 \beta )$ . More generally one may consider $c \beta$ , with c reflecting the desired degree of pessimism relative to policy regularization.

## 3.4. Cumulant Expansion

Let $K _ { x } ( t ) : = \log M _ { x } ( t ) = \log \mathbb { E } _ { p } [ e ^ { t r } ]$ be the cumulant generating function $( \mathbf { C G F } ) \operatorname { o f } p ( r \mid x , y )$ , with cumulants $\kappa _ { n }$ defined by $\begin{array} { r } { K _ { x } ( t ) = \sum _ { n > 1 } \kappa _ { n } t ^ { n } / n ! } \end{array}$ . Then $ \kappa _ { 1 } = \mu , \kappa _ { 2 } =$ $\sigma ^ { 2 } , \kappa _ { 3 }$ is the third central moment, $\kappa _ { 4 } = \mathbb { E } [ ( r - \mu ) ^ { 4 } ] - 3 \sigma ^ { 4 }$ (excess kurtosis $\times \sigma ^ { 4 } )$ , and so on.

Substituting into Eqs. 4 and 8:

$$
\tilde {r} _ {\mathrm{opt/rob}} = \mu \pm \frac {\sigma^ {2}}{2 \beta} + \frac {\kappa_ {3}}{6 \beta^ {2}} \pm \frac {\kappa_ {4}}{2 4 \beta^ {3}} + \dots\tag{9}
$$

Even cumulants $( \kappa _ { 2 } , \kappa _ { 4 } , \ldots )$ flip sign between the two while odd cumulants $( \kappa _ { 3 } , \kappa _ { 5 } , \ldots )$ do not. For symmetric distributions $( \kappa _ { 2 n + 1 } = 0$ for $n \geq 1 ) , \tilde { r } _ { \mathrm { o p t } }$ and $\tilde { r } _ { \mathrm { r o b } }$ are exact mirror images about $\mu$ .

Gaussian case. If $\cdot p ( r \mid x , y ) = \mathcal { N } ( \mu , \sigma ^ { 2 } )$ , all cumulants beyond $\kappa _ { 2 }$ vanish, $K _ { x } ( t ) = \mu t + \sigma ^ { 2 } t ^ { 2 } / 2$ , and the series in Eq. 9 terminate exactly:

$$
\tilde {r} _ {\mathrm{opt}} = \mu + \frac {\sigma^ {2}}{2 \beta}, \qquad \tilde {r} _ {\mathrm{rob}} = \mu - \frac {\sigma^ {2}}{2 \beta}.\tag{10}
$$

(a)

(b)
Figure 1. The pessimistic effective reward interpolates between Mean and WCO. Illustration for a $K { = } 1 0 \ \bar { \mathrm { R M } }$ ensemble that is bimodal on a suspect response (eight members assign high reward, two flag it). (a) The adversarially tilted distribution $Q ^ { * } ( r ) \propto p ( r ) e ^ { - r / \beta }$ (Eq. 7): as $\beta$ decreases, the adversary concentrates mass on the worst members. (b) The exact effective reward $\tilde { r } _ { \mathrm { r o b } } \left( \mathrm { E q . } \ 8 \right)$ recovers Mean as $\beta $ ∞ and WCO as $\beta \to 0 ( \sec . 3 . 5 )$ . The Gaussian truncation $\hat { \mu } - \hat { \sigma } ^ { 2 } / 2 \beta$ (UWO with $\lambda { = } 1 / 2 \beta )$ agrees at large β but is over-pessimistic at small $\beta ,$ dropping below min<sub>i</sub> $R _ { i }$ which no distribution supported on the ensemble can justify (Sec. 3.6).

Under any Gaussian distributional RM, the principled pessimistic reward is $\mu - \sigma ^ { 2 } / 2 \beta$ , with the variance coefficient set by the KL coefficient $\beta$ rather than tuned as a free hyperparameter.

## 3.5. Unification of Prior Methods

The aggregation rules of Coste et al. (2024) emerge as limits and truncations of $\tilde { r } _ { \mathrm { r o b } }$

Mean $( \beta \to \infty ) . \quad \operatorname { A s } \beta \to \infty$ , the variance term in Eq. 9 vanishes $( \sigma ^ { 2 } / 2 \beta \to 0 )$ , and similarly for all higher orders. Hence $\tilde { r } _ { \mathrm { r o b } }  \mu .$ . Equivalently, an infinite KL penalty forces the adversary’s distribution $Q ^ { * }$ back onto $p ,$ so the worst case coincides with the mean. Mean aggregation is the risk-neutral limit of the pessimistic effective reward.

WCO $( \beta \to 0 ) . \quad \operatorname { A s } \beta \to 0 .$ , the tilted distribution $Q ^ { * } ( r )$ ∝ $p ( r ) e ^ { - r / \beta }$ concentrates all of its mass on inf $\operatorname { s u p p } ( p )$ . For an empirical ensemble $\{ R _ { i } \} _ { i = } ^ { K }$ where $p$ is a uniform mixture of point masses, the infimum is min $R _ { i }$ . Hence

Table 1. Existing pessimistic aggregation rules (Coste et al., 2024) as limiting cases or truncations of $\tilde { r } _ { \mathrm { r o b } }$ . The fourth row gives the principled default set by the KL coefficient $\beta$ under a Gaussian posterior, with no extra hyperparameter.

<table><tr><td>METHOD</td><td>FORMULA</td><td>RECOVERED AS</td></tr><tr><td>MEAN</td><td> $\hat{\mu}$ </td><td> $\beta \rightarrow \infty$  LIMIT</td></tr><tr><td>WCO</td><td> $\min_{i} R_{i}$ </td><td> $\beta \rightarrow 0$  LIMIT</td></tr><tr><td>UWO</td><td> $\hat{\mu} - \lambda \hat{\sigma}^{2}$ </td><td>GAUSSIAN,  $\lambda$  FREE</td></tr><tr><td>OURS</td><td> $\hat{\mu} - \hat{\sigma}^{2}/2\beta$ </td><td>GAUSSIAN,  $\lambda = 1/2\beta$ </td></tr></table>

$$
\lim _ {\beta \to 0} \tilde {r} _ {\mathrm{rob}} (x, y) = \min _ {i} R _ {i} (x, y).\tag{11}
$$

Worst-case optimization is the unbounded-adversary limit of the pessimistic effective reward.

UWO (Gaussian truncation). The UWO rule of Coste et al. (2024) is

$$
R _ {\mathrm{UWO}} (x, y) = \hat {\mu} (x, y) - \lambda \hat {\sigma} ^ {2} (x, y),\tag{12}
$$

with λ treated as a free hyperparameter. Intuitively, UWO works by penalizing the policy for generating responses for which there is high disagreement among reward models within the ensemble, which helps prevent the exploitation of a single reward model which might be erroneously assigning high rewards to incorrect or low-quality responses. Comparing with the Gaussian case of Eq. 10, UWO is exactly the pessimistic effective reward under a Gaussian posterior, with $\lambda = 1 / ( 2 \beta )$ . UWO’s implicit assumptions are: (i) $p ( r \mid x , y )$ is approximately Gaussian and (ii) the variance coefficient can be decoupled from the KL coefficient $\beta .$

Table 1 and Figure 1 summarize the unification. The takeaway is that the choice of aggregation rule is a choice about the assumed shape of $p ( r \mid x , y )$ and the adversary’s KL budget. Mean assumes no uncertainty matters, WCO assumes the worst possible distribution, and UWO is the middle ground in a Gaussian approximation. Our framework makes these assumptions explicit and supplies a principled default $( \lambda = 1 / 2 \beta )$ when working with a Gaussian distributional RM.

## 3.6. Estimation from a Finite Sample

In practice $p ( r \mid x , y )$ is approximated by a finite sample $\{ r _ { i } ( x , y ) \} _ { i = 1 } ^ { K }$ , either ensemble members or posterior samples:

$$
\hat {\mu} = \frac {1}{K} \sum_ {i = 1} ^ {K} r _ {i}, \quad \hat {\sigma} ^ {2} = \frac {1}{K - 1} \sum_ {i = 1} ^ {K} (r _ {i} - \hat {\mu}) ^ {2}.\tag{13}
$$

Two natural estimators of $\tilde { r } _ { \mathrm { r o b } }$ arise.

Direct log-MGF estimator.

$$
\hat {\tilde {r}} _ {\mathrm{rob}} ^ {(\log \mathrm{MGF})} = - \beta \log \left(\frac {1}{K} \sum_ {i = 1} ^ {K} e ^ {- r _ {i} / \beta}\right).\tag{14}
$$

This estimator is consistent but biased at finite $K \colon$ since log is concave, Jensen gives $\mathbb { E } [ \hat { \tilde { r } } _ { \mathrm { r o b } } ^ { ( \mathrm { l o g M G F } ) } ] \geq \tilde { r } _ { \mathrm { r o b } }$ , with bias growing in $\sigma / \beta$ . It is sensitive to outliers when $\beta$ is small (the $\exp ( - r _ { i } / \beta )$ term blows up for any single low $r _ { i } )$

Gaussian-truncated estimator.

$$
\hat {\tilde {r}} _ {\mathrm{rob}} ^ {(\mathrm{Gauss})} = \hat {\mu} - \frac {\hat {\sigma} ^ {2}}{2 \beta}.\tag{15}
$$

This is exact when $p$ is Gaussian and drops higher cumulants otherwise. We note that for small K (as is typical for RM ensembles, $K \in \{ 3 , \ldots , 1 0 \}$ in Coste et al. (2024)), higher cumulants are poorly estimated. The relative error on $\hat { \sigma } ^ { 2 }$ alone is $\sqrt { 2 / ( K - 1 ) } \approx 7 1 \%$ for $K = 5 ,$ so we recommend Eq. 15 as a default.

## 4. Discussion

The pessimistic effective reward $\tilde { r } _ { \mathrm { r o b } }$ is only as informative as the posterior p(r $\mid x , y )$ used to compute it. Two concrete prerequisites:

(i) A genuine distributional RM. Scalar RMs trained with the Bradley-Terry likelihood (Bradley & Terry, 1952) provide no uncertainty. Two practical options are: (a) a deep ensemble (Lakshminarayanan et al., 2017; Coste et al., 2024; Dwaracherla et al., 2024), whose empirical variance approximates $\sigma ^ { 2 } ( x , y ) ;$ ; and (b) a Bayesian last-layer construction (Yang et al., 2026; Zhai et al., 2023), which yields a closedform Gaussian posterior at the cost of a single forward pass and an $O ( d ^ { 2 } )$ matrix-vector product. The Bayesian construction is especially interesting as its posterior is Gaussian and Eq. 10 applies without truncation.

(ii) Calibration. The variance term $\sigma ^ { 2 } ( x , y )$ only modulates $\tilde { r } _ { \mathrm { r o b } }$ usefully if it tracks true epistemic uncertainty. The Bradley–Terry loss collapses soft preferences (0.51 and 0.99) onto the same hard label and degrades calibration. Strictly proper scoring rules (Gneiting & Raftery, 2007), in particular the Brier score (Brier, 1950), preserve annotator confidence and are known to improve calibration. Using AI feedback (Bai et al., 2022; Lee et al., 2024) provides soft labels naturally by prompting for a confidence score during AI feedback.

## Implications for existing methods.

• When Mean should suffice. The cumulant expansion (Eq. 9) shows that pessimism corrections decay as $1 / \beta ^ { n - 1 }$ . For large $\beta$ (heavy KL regularization), Mean is essentially optimal as all rules collapse to it.

• When WCO is appropriate. WCO is the $\beta ~  ~ 0$ limit, suitable when the adversary is essentially unconstrained. In RLHF, this corresponds to a setting where one expects the RM to be arbitrarily wrong in some direction. Eisenstein et al. (2024) report that WCO can underperform Mean when the ensemble is too small or correlated, which our framework explains: WCO discards all distributional information beyond the minimum.

• Why UWO works. UWO is the right shape under a Gaussian posterior. Its empirical success suggests that practical RM ensembles are reasonably symmetric and unimodal. The free λ may be absorbing miscalibration; under the theory-prescribed value $\lambda = 1 / 2 \beta ,$ , calibration becomes essential.

Beyond Gaussian. The cumulant expansion in Eq. 9 suggests three directions when $p ( r \mid \mathbf { \theta } x , y )$ is not wellapproximated by a Gaussian. (i) For asymmetric but lighttailed distributions, third- and fourth-cumulant corrections to Eq. 15 can be estimated from larger ensembles. (ii) For genuinely heavy-tailed distributions where the MGF estimator is unstable, quantile-based pessimism (e.g., CVaR) is a robust alternative, connecting to quantile distributiona RL (Dabney et al., 2018b;a; Keramati et al., 2020). (iii) Flexible posterior families, such as mixtures of Gaussians or normalizing flows fit directly to held-out preference data could replace the ensemble or Bayesian linear head.

Connections to broader pessimism. The KL-DRO derivation connects RLHF reward hacking to pessimism in offline RL (Jin et al., 2021; Xie et al., 2021; Rigter et al., 2022), where similar log-MGF / entropic-risk formulations arise from analogous principles. Our contribution is to localize this connection: in RLHF, the relevant random variable is the reward (not the transition dynamics, which are deterministic for an autoregressive policy), and the relevant penalty parameter is the standard KL coefficient β already present in Eq. 1.

## 5. Conclusion

We have given a single derivation that captures the essence of reward uncertainty in RLHF. From either a Bayesian or a KL-DRO starting point, the closed-form effective reward is $\widetilde { r } ( x , y ) = \pm \beta \log { \mathbb { E } _ { p } [ e ^ { \pm r / \beta } ] }$ The pessimistic branch unifies mean aggregation, WCO, and UWO as limits and truncations, and yields a principled default $\tilde { r } = \mu - \sigma ^ { 2 } / 2 \beta$ under a Gaussian posterior in which the variance coefficient is set by the KL coefficient rather than tuned. The framework clarifies what each existing heuristic implicitly assumes and gives a recipe for going beyond Gaussian via higher cumulants or quantile-based pessimism.

## Acknowledgements

We thank Kiante Brantley for initial guidance and helpful discussions during the early stages of this project. OpenAI GPT-5.5 and Refine.ink were used to assist with language editing and improve the clarity and readability of the manuscript.

## References

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Mane, D. Concrete problems in´ ai safety, 2016. URL https://arxiv.org/abs/ 1606.06565.

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., Chen, C., Olsson, C., Olah, C., Hernandez, D., Drain, D., Ganguli, D., Li, D., Tran-Johnson, E., Perez, E., Kerr, J., Mueller, J., Ladish, J., Landau, J., Ndousse, K., Lukosuite, K., Lovitt, L., Sellitto, M., Elhage, N., Schiefer, N., Mercado, N., DasSarma, N., Lasenby, R., Larson, R., Ringer, S., Johnston, S., Kravec, S., Showk, S. E., Fort, S., Lanham, T., Telleen-Lawton, T., Conerly, T., Henighan, T., Hume, T., Bowman, S. R., Hatfield-Dodds, Z., Mann, B., Amodei, D., Joseph, N., McCandlish, S., Brown, T., and Kaplan, J. Constitutional ai: Harmlessness from ai feedback, 2022. URL https://arxiv.org/abs/2212.08073.

Bellemare, M. G., Dabney, W., and Munos, R. A distributional perspective on reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, pp. 449–458, 2017.

Ben-Tal, A., El Ghaoui, L., and Nemirovski, A. Robust Optimization. Princeton University Press, 2009.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Brier, G. W. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3, 1950.

Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Coste, T., Anwar, U., Kirk, R., and Krueger, D. Reward model ensembles help mitigate overoptimization. In International Conference on Learning Representations, volume 2024, pp. 50905–50931, 2024.

Dabney, W., Ostrovski, G., Silver, D., and Munos, R. Implicit quantile networks for distributional reinforcement

learning. In International conference on machine learning, pp. 1096–1105. PMLR, 2018a.

Dabney, W., Rowland, M., Bellemare, M., and Munos, R. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018b.

Dorka, N. Quantile regression for distributional reward models in rlhf, 2024. URL https://arxiv.org/ abs/2409.10164.

Dwaracherla, V., Asghari, S. M., Hao, B., and Van Roy, B. Efficient exploration for LLMs. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 12215–12227. PMLR, 21–27 Jul 2024.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., Shaw, P., and Berant, J. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking, 2024. URL https://arxiv.org/abs/2312.09244.

Follmer, H. and Schied, A. ¨ Stochastic finance: an introduction in discrete time. Walter de Gruyter GmbH & Co KG, 2025.

Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102(477):359–378, 2007.

Gupta, D., Fisch, A., Dann, C., and Agarwal, A. Mitigating preference hacking in policy optimization with pessimism, 2025. URL https://arxiv.org/abs/ 2503.06810.

Hansen, L. P. and Sargent, T. J. Robustness. Princeton University Press, 2008.

Jin, Y., Yang, Z., and Wang, Z. Is pessimism provably efficient for offline rl? In International conference on machine learning, pp. 5084–5096. PMLR, 2021.

Keramati, R., Dann, C., Tamkin, A., and Brunskill, E. Being optimistic to be conservative: Quickly learning a cvar policy. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pp. 4436–4443, 2020.

Korbak, T., Perez, E., and Buckley, C. L. Rl with kl penalties is better viewed as bayesian inference, 2022. URL https://arxiv.org/abs/2205.11275.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles, 2017. URL https://arxiv.org/abs/ 1612.01474.

Lee, H., Phatale, S., Mansoor, H., Mesnard, T., Ferret, J., Lu, K., Bishop, C., Hall, E., Carbune, V., Rastogi, A., and Prakash, S. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback, 2024. URL https://arxiv.org/abs/2309.00267.

Levine, S. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018. URL https://arxiv.org/abs/1805.00909.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Pan, A., Bhatia, K., and Steinhardt, J. The effects of reward misspecification: Mapping and mitigating misaligned models, 2022. URL https://arxiv.org/ abs/2201.03544.

Perez, E., Ringer, S., Lukosiute, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., et al. Discovering language model behaviors with modelwritten evaluations. In Findings of the association for computational linguistics: ACL 2023, pp. 13387–13434, 2023.

Rame, A., Vieillard, N., Hussenot, L., Dadashi, R., Cideron,´ G., Bachem, O., and Ferret, J. Warm: on the benefits of weight averaged reward models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Rigter, M., Lacerda, B., and Hawes, N. Rambo-rl: Robust adversarial model-based offline reinforcement learning. Advances in neural information processing systems, 35: 16082–16097, 2022.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/ 1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T.,

McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., and Perez, E. Towards understanding sycophancy in language models, 2025. URL https://arxiv.org/abs/2310.13548.

Singhal, P., Goyal, T., Xu, J., and Durrett, G. A long way to go: Investigating length correlations in rlhf, 2024. URL https://arxiv.org/abs/2310.03716.

Skalse, J., Howe, N., Krasheninnikov, D., and Krueger, D. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Sun, W., Cheng, X., Yu, X., Xu, H., Yang, Z., He, S., Zhao, J., and Liu, K. Probabilistic uncertain reward model, 2025. URL https://arxiv.org/abs/2503.22480.

Wang, A., Arcuschin, I., and Conmy, A. Automatically finding reward model biases, 2026. URL https:// arxiv.org/abs/2602.15222.

Xie, T., Cheng, C.-A., Jiang, N., Mineiro, P., and Agarwal, A. Bellman-consistent pessimism for offline reinforcement learning. Advances in neural information processing systems, 34:6683–6694, 2021.

Xu, Y., Kang, H., Suresh, T., Wan, Y., and Singh, G. Learning a pessimistic reward model in rlhf, 2025. URL https://arxiv.org/abs/2505.20556.

Xu, Z., Lu, Q., Zhang, Q., Qiu, L., Hong, I., Yu, C., Yao, W., Liu, Y., Jiang, H., Li, L., et al. Ask a strong llm judge when your reward model is uncertain. Advances in Neural Information Processing Systems, 38:74639–74664, 2026.

Yan, Y., Lou, X., Li, J., Zhang, Y., Xie, J., Yu, C., Wang, Y., Yan, D., and Shen, Y. Reward-robust rlhf in llms, 2024. URL https://arxiv.org/abs/2409.15360.

Yang, D., Stante, S., Redhardt, F., Libon, L., Kassraie, P., Hakimi, I., Pasztor, B., and Krause, A. Rewarduq: A´ unified framework for uncertainty-aware reward models, 2026. URL https://arxiv.org/abs/2602. 24040.

Zhai, Y., Zhang, H., Lei, Y., Yu, Y., Xu, K., Feng, D., Ding, B., and Wang, H. Uncertainty-penalized reinforcement learning from human feedback with diverse reward lora ensembles, 2023. URL https://arxiv.org/abs/ 2401.00243.

Zhang, X., Ton, J.-F., Shen, W., Wang, H., and Liu, Y. Overcoming reward overoptimization via adversarial policy optimization with lightweight uncertainty estimation, 2024. URL https://arxiv.org/abs/2403. 05171.

## A. Proofs and Derivations

A.1. Optimistic Effective Reward

Setup. Let $\pi _ { 0 } = \pi _ { \mathrm { r e f } }$ . For each $( x , y ) , r \sim p ( r \mid x , y )$ . The KL-regularized objective with effective reward r˜ is

$$
J (\pi) = \mathbb {E} _ {\pi} [ \tilde {r} (x, y) ] - \beta D _ {\mathrm{KL}} (\pi \| \pi_ {0}), \quad \beta > 0.\tag{16}
$$

Step 1: Optimality variable. Introduce $O \in \{ 0 , 1 \}$ with $p ( O = 1 \mid x , y , r ) = e ^ { r / \beta }$ where r is shifted so sup $r = 0$

Step 2: Marginalize.

$$
p (O = 1 \mid x, y) = \int e ^ {r / \beta} p (r \mid x, y) d r = M _ {x} (1 / \beta),\tag{17}
$$

where $M _ { x } ( t ) : = \mathbb { E } _ { p } [ e ^ { t r } ]$ is the MGF.

Step 3: Bayes’ rule. With prior $\pi _ { 0 }$ ,

$$
\pi_ {\mathrm{opt}} ^ {*} (y \mid x) \propto \pi_ {0} (y \mid x) M _ {x} (1 / \beta).\tag{18}
$$

Step 4: Identify r˜. Matching to $\pi ^ { * } \propto \pi _ { 0 }$ exp(˜r/β) gives

$$
\tilde {r} _ {\mathrm{opt}} (x, y) = \beta \log \mathbb {E} _ {p} [ e ^ {r / \beta} ].\tag{19}
$$

Step 5: Jensen. Since exp is convex, $\mathbb { E } _ { p } [ e ^ { r / \beta } ] \ge e ^ { \mu / \beta }$ , so $\tilde { r } _ { \mathrm { o p t } } \geq \mu$

## A.2. Pessimistic Effective Reward (KL-DRO)

Step 1: Definition.

$$
\tilde {r} _ {\mathrm{rob}} (x, y) = \inf _ {Q} \{\mathbb {E} _ {Q} [ r ] + \beta D _ {\mathrm{KL}} (Q \| p) \}.\tag{20}
$$

Step 2: Lagrangian. With multiplier λ for $\textstyle \int Q = 1$

$$
\mathcal {L} (Q, \lambda) = \int Q (r) r d r + \beta \int Q \log Q d r - \beta \int Q \log p d r - \lambda (\int Q d r - 1).
$$

(21)

Step 3: Stationarity. Using $\textstyle { \frac { d } { d Q } } [ Q \log Q ] = \log Q + 1$

$$
\frac {\delta \mathcal {L}}{\delta Q (r)} = r + \beta (\log Q + 1) - \beta \log p - \lambda = 0,\tag{22}
$$

giving $Q ^ { * } ( r ) \propto p ( r ) e ^ { - r / \beta }$ . The normalization is $C ( x , y ) : = \mathbb { E } _ { p } [ e ^ { - r / \beta } ]$

Step 4: Evaluate the infimum. log $\begin{array} { r } { Q ^ { * } ( r ) - \log p ( r ) = - r / \beta - \log C , } \end{array}$ so

$$
D _ {\mathrm{KL}} (Q ^ {*} \| p) = - \frac {1}{\beta} \mathbb {E} _ {Q ^ {*}} [ r ] - \log C.\tag{23}
$$

Therefore $\mathbb { E } _ { Q ^ { * } } [ r ] + \beta D _ { \mathrm { K L } } ( Q ^ { * } \Vert p ) = - \beta \log C$ (the reward terms cancel exactly), giving

$$
\tilde {r} _ {\mathrm{rob}} (x, y) = - \beta \log \mathbb {E} _ {p} [ e ^ {- r / \beta} ].\tag{24}
$$

Step 5: Jensen. $\mathbb { E } _ { p } [ e ^ { - r / \beta } ] \ge e ^ { - \mu / \beta }$ ; taking −β log reverses the inequality, so $\tilde { r } _ { \mathrm { r o b } } \le \mu$

## A.3. Cumulant Expansion

Let $\begin{array} { r } { K _ { x } ( t ) = \log \mathbb { E } _ { p } [ e ^ { t r } ] = \sum _ { n > 1 } \kappa _ { n } t ^ { n } / n ! } \end{array}$ . Then

$$
\tilde {r} _ {\mathrm{opt}} = \beta K _ {x} (1 / \beta) = \sum_ {n \geq 1} \frac {\kappa_ {n}}{n ! \beta^ {n - 1}} = \mu + \frac {\sigma^ {2}}{2 \beta} + \frac {\kappa_ {3}}{6 \beta^ {2}} + \frac {\kappa_ {4}}{2 4 \beta^ {3}} + \dots ,\tag{25}
$$

$$
\tilde {r} _ {\mathrm{rob}} = - \beta K _ {x} (- 1 / \beta) = \sum_ {n > 1} \frac {(- 1) ^ {n + 1} \kappa_ {n}}{n ! \beta^ {n - 1}} = \mu - \frac {\sigma^ {2}}{2 \beta} + \frac {\kappa_ {3}}{6 \beta^ {2}} - \frac {\kappa_ {4}}{2 4 \beta^ {3}} - \dots .\tag{26}
$$

For Gaussian $p , \kappa _ { n \geq 3 } = 0$ and the series terminate at $\tilde { r } = \mu \pm \sigma ^ { 2 } / 2 \beta$

## A.4. Recovery of Mean and WCO

$\beta  \infty$ (Mean). Each term $\kappa _ { n } / ( n ! \beta ^ { n - 1 } )$ for $n \geq 2$ vanishes, leaving $\tilde { r } _ { \mathrm { r o b } }  \kappa _ { 1 } = \mu$ . Equivalently, $Q ^ { * } ( r )$ ∝ $p ( r ) e ^ { - r / \beta } \to p ( r )$ as $\beta \to \infty$

$\beta \to 0 ( \mathbf { W C O } )$ . For an empirical ensemble $\{ R _ { i } \} _ { i = 1 } ^ { K }$ with $\begin{array} { r } { p ( r \mid x , y ) = \frac { 1 } { K } \sum _ { i } \delta ( r - R _ { i } ) } \end{array}$

$$
\tilde {r} _ {\mathrm{rob}} = - \beta \log \left(\frac {1}{K} \sum_ {i = 1} ^ {K} e ^ {- R _ {i} / \beta}\right).\tag{27}
$$

Let $R _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i } R _ { i }$ and factor out $e ^ { - R _ { \mathrm { m i n } } / \beta }$

$$
\tilde {r} _ {\mathrm{rob}} = R _ {\min} - \beta \log \left(\frac {1}{K} \sum_ {i = 1} ^ {K} e ^ {- (R _ {i} - R _ {\min}) / \beta}\right).\tag{28}
$$

As $\beta  0 ,$ , all terms in the sum with $R _ { i } > {  { R _ { \mathrm { m i n } } } }$ are exponentially suppressed; the sum approaches the count of minimizing members m divided by K. The logarithm of this sum approaches the constant $\log ( m / K )$ , so the overall correction term scales as $O ( \beta ) \to 0$ . Hence $\tilde { r } _ { \mathrm { r o b } }  R _ { \mathrm { m i n } } = \mathrm { m i n } _ { i } R _ { i }$ , recovering WCO.

Gaussian truncation (UWO). For $p = \mathcal { N } ( \mu , \sigma ^ { 2 } ) , K _ { x } ( t ) = \mu t + \sigma ^ { 2 } t ^ { 2 } / 2$ exactly, so $\tilde { r } _ { \mathrm { r o b } } = \mu - \sigma ^ { 2 } / 2 \beta$ . This matches $R _ { \mathrm { U W O } } = \hat { \mu } - \lambda \hat { \sigma } ^ { 2 }$ with $\lambda = 1 / ( 2 \beta )$
