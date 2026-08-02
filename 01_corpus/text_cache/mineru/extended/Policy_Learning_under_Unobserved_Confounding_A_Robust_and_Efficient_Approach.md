# Policy Learning under Unobserved Confounding: A Robust and Efficient Approach<sup>\*</sup>

Zequn Jin<sup>†</sup> Gaoqian Xu<sup>‡</sup> Xi Zheng<sup>§</sup> Yahong Zhou<sup>¶</sup>

July 9, 2026

## Abstract

This paper develops a robust and efficient method for policy learning from observational data in the presence of unobserved confounding, complementing existing instrumental variable based approaches. We employ the marginal sensitivity model (MSM) to relax the commonly used yet restrictive unconfoundedness assumption by introducing a sensitivity parameter that captures the extent of selection bias induced by unobserved confounders. Building on this framework, we consider two distributionally robust welfare criteria, defined as the worst-case welfare and policy improvement functions, evaluated over an uncertainty set of counterfactual distributions charac terized by the MSM. Closed-form expressions for both welfare criteria are derived. Leveraging these identification results, we construct doubly robust scores and estimate the robust policies by maximizing the proposed criteria. Our approach accommodates flexible machine learning meth ods for estimating nuisance components, even when these converge at moderately slow rates. We establish asymptotic regret bounds for the resulting policies, providing a robust guarantee against the most adversarial confounding scenario. The proposed method is evaluated through extensive simulation studies and empirical applications to the JTPA study and Head Start program.

JEL codes: C14, C31, C54

Keywords: Partial Identification, Targeted Policy, Treatment Effects, Sensitivity Analysis

## 1 Introduction

Policy targeting, allocating interventions based on individual characteristics, has become increasingly influential in applied economics and econometrics. For example, a local government may need to decide which workers receive job training, a school district may identify students most in need of educational support, and a regulatory agency has to determine which workplaces warrant safety inspections.

Policymakers often rely on observational data to guide targeting decisions. However, unobserved confounding, unobserved factors that simultaneously affect both treatment and potential outcomes, can severely bias causal inference from such data. Ignoring these confounders can result in misleading estimates and suboptimal policy decisions. For example, Johnson et al. (2023) note that safety inspections tend to target workplaces with a history of accidents or informal complaints, events that may reflect unobserved traits, such as poor safety culture or weak management, which also affect injury risk. Given the potentially large benefits of well-targeted inspections,<sup>1</sup>designing targeting without accounting for unobserved confounding may misidentify high-risk workplaces, leading to the inefficient use of limited regulatory resources. Similar challenges arise in education, where observational data guide targeting decisions for specific academic programs. However, actual enrollment often hinges on parental choice, which is based on factors inaccessible to policymakers, such as a childs latent abilities, behavioral traits, or home environment. This self-selection process based on unobserved characteristics complicates the design of effective targeting policies.

Many existing policy learning (or targeting) methods using observational data assume no unobserved confounding, that is, treatment assignment is random conditional on the observed covariates; see (Kitagawa and Tetenov, 2018; Athey and Wager, 2021). This assumption, commonly referred to as unconfoundedness, is often unreasonable, as historical policies or decisions may have been based on additional, unobserved information. When this assumption breaks down, researchers often turn to instrumental variable (IV) methods, which enable causal inference and policy targeting; see (Cui and Tchetgen Tchetgen, 2021; Qiu et al., 2021; Sasaki and Ura, 2024). Yet in practice, identifying a valid and credible instrument is often challenging. Moreover, even with a valid binary instrument, heterogeneous treatment effects are typically only partially identified, thereby limiting the ability to effectively target interventions.

As a supplement to IV methods, sensitivity analysis is used to assess the impact of unobserved confounding on policy effects, as originally proposed by Rosenbaum and Rubin (1983). Building on this idea, Kallus and Zhou (2021) incorporate sensitivity analysis into policy learning. Leveraging the marginal sensitivity model (MSM) of Tan (2006), they relax the unconfoundedness assumption and formulate a robust optimization framework to learn policies that remain effective under unobserved confounding. Specifically, building on the framework proposed by Zhao et al. (2019), they construct the objective function as the largest inverse propensity weighting (IPW) estimator of expected welfare, where the putative propensity scores are restricted by the MSM. While the method offers robustness to unobserved confounding, it is not without limitations. First, Kallus and Zhou (2021) derive a loose upper bound on expected welfare improvement, which leads to overly conservative policies. Second, the method requires propensity scores to be estimated at the parametric rate, which precludes the use of many popular machine learning (ML) methods, such as random forests and LASSO. Third, the policy optimization algorithm is tailored to differentiable parametric policy classes and relies on a computationally intensive iterative routine.

## 1.1 Main Contributions

This work proposes a method for policy learning that is robust to unobserved confounding, serving as a complement to existing IV-based approaches. Building on the MSM framework, we construct a distributionally robust welfare criterion that captures the worst-case policy value over a plausible set of counterfactual outcome distributions. Based on this criterion, we develop a two-stage policy learning procedure: the first stage estimates nuisance components using flexible nonparametric or modern ML methods; the second stage optimizes the estimated criterion over a constrained policy class. The resulting procedure is both computationally efficient and straightforward to implement.

In Section 3, building on Dorn and Guo (2023) and Dorn et al. (2025), we derive closed-form expressions for the worst-case (i.e., sharp lower bounds of) average welfare and welfare improvement<sup>2</sup> compatible with the MSM. Although our main analysis focuses on binary compulsory assignment, Appendix A examines selection-driven policy targeting under the MSM, following Ida et al. (2026), with self-selection as a policy option. This extension is relevant when the MSM-implied treatmenteffect bounds are insufficiently informative for compulsory assignment, in which case preserving self-selection may help exploit individuals’ private information for policy targeting. Given the identification of these two robust criteria/objectives using moment conditions, we can construct the doubly robust estimation of them, even when the nuisance components are estimated at moderately slow rate, i.e., $o \big ( n ^ { - 1 / 4 } \big )$ in the $L ^ { 2 } { \mathrm { - n o r m } }$

Section 4 presents algorithms for learning optimal confounding-robust policies, where the policy class may be constrained by practical considerations such as implementability, cost, and interpretability. Our method follows the mainstream two-stage paradigm of Athey and Wager (2021); Zhou et al. (2023). Unlike Kallus and Zhou (2021), our second-stage policy optimization supports standard ML methods and readily available software packages, such as logistic regression, policy trees, and neural networks, to learn the optimal policy by maximizing the estimated criterion. For example, given the doubly robust scores, our method can be implemented using the widely adopted R-package policytree (Sverdrup et al., 2020). Moreover, Section 4 provides asymptotic upper regret bounds as performance guarantees for our estimated confounding-robust policies.

We validate our method via simulations in Section 5, comparing it to Kallus and Zhou (2021) and the empirical welfare maximization (EWM) of Athey and Wager (2021). We then demonstrate its practical relevance in Section 6 by applying it to two classic empirical settings where unobserved confounding is a primary concern, focusing on the contrast with the EWM.

Our applications suggest that ignoring such confounding may lead to misguided policy recommendations. In the National Job Training Partnership Act (JTPA) study, our method refines the naive EWM policy which treats nearly all participants. As concerns about unobserved confounding grow, our method shifts to identifying more selective policies that target subgroups with higher education but lower prior earnings. In our Head Start application, the EWM policy assigns no children to Head Start. In contrast, our robust method prioritizes disadvantaged children with lower family income and maternal education, aligning with the program’s mission. These findings demonstrate that accounting for unobserved confounding can fundamentally alter policy targeting, highlighting the importance of robust methods in practice.

## 1.2 Related Literature

Our work contributes to the growing literature on policy learning. Many policy learning works focus on mean-optimal policies under unconfoundedness (Kitagawa and Tetenov, 2018; Athey and Wager, 2021; Zhou et al., 2023). Alternative objectives under the same assumption have also been explored, including quantile-optimal policies (Wang et al., 2018; Leqi and Kennedy, 2021) as well as those motivated by fairness (Kitagawa and Tetenov, 2021; Fang et al., 2023; Viviano and Bradic, 2024; Fan et al., 2025).

To address unobserved confounding, several studies have proposed IV-based approaches for policy targeting. To avoid partial identification of the welfare function, Sasaki and Ura (2024) assume the availability of a continuous instrument with sufficiently large support and identify the average welfare via the marginal treatment effect (MTE). A similar approach has been applied to the design of optimal encouragement rules under endogeneity; see (Chen and Xie, 2022; Liu, 2022). In the statistics literature, other works investigate optimal treatment rules or encouragement interventions using binary instruments; see (Cui and Tchetgen Tchetgen, 2021; Qiu et al., 2021). However, these methods rely on quite stringent assumptions for a valid IV to point identify the conditional average treatment effect (CATE).

Binary instruments, though widely used, typically allow identification only of the local average treatment effect (LATE) for the subpopulation of compliers; see (Imbens and Angrist, 1994; Angrist et al., 1996; Athey and Imbens, 2017). This presents major challenges for policy learning: the resulting policy intervention is effective only for compliers, a subpopulation that cannot be identified in advance in a new sample, thus limiting external validity. Moreover, compliance status is instrument-dependent, leading to instability and lack of generalizability across different IV designs. To address this issue, Pu and Zhang (2021) and d’Adamo (2021), propose methods for binary IV settings that partially identify heterogeneous treatment effects, learning robust policies by optimizing policy criteria constructed from IV-identified bounds on the CATE.

More broadly, our work contributes to the growing literature on policy learning and treatment choice under partial identification, where the relevant welfare function is only set-identified. To address this ambiguity, the literature commonly relies on the minimax principle (Manski, 2000, 2004, 2025; Stoye, 2012; Yata, 2025; Montiel Olea et al., 2026), while Christensen et al. (2026) proposes a hybrid approach that combines minimax criteria with quasi-Bayesian methods. Partial identification may also arise from distributional ambiguity, in the sense that the target population may differ from the observed sample. In this setting, some works adopt distributionally robust optimization to learn policies that remain valid across new environments (Kido, 2022; Si et al., 2023; Qi et al., 2023; Lei et al., 2023; Zhang et al., 2024b; Adjaho and Christensen, 2025). These works mainly maximize worst-case expected outcomes over ambiguity sets defined by Wasserstein distance or Kullback-Leibler divergence. Moreover, learning externally valid policies may require extrapolating causal parameters that cannot be point-identified from the available sample. To address this, recent studies (Khan et al., 2023; Zhang et al., 2024a; Ben-Michael et al., 2025; Higbee, 2025) impose functional or shape restrictions to partially identify treatment effects and derive robust policies. Finally, we focus on learning robust policies when causal parameters are partially identified due to unmeasured confounding, viewed through the lens of sensitivity analysis (Tan, 2006; Masten and Poirier, 2018; Zhao et al., 2019; Kallus and Zhou, 2021; Dorn and Guo, 2023; Dorn et al., 2025).

## 2 Problem Statement and Preliminaries

We consider observational data consisting of random samples $\{ Z _ { i } \} _ { i = 1 } ^ { n } = \{ X _ { i } , Y _ { i } , A _ { i } \} _ { i = 1 } ^ { n }$ , where $X _ { i } \in \mathcal { X } \subseteq \mathbb { R } ^ { d }$ represents the observed covariates, $A _ { i } \in \{ 0 , 1 \}$ denotes a binary intervention/treatment, and $Y _ { i } \in \mathbb { R }$ is the real-valued observed outcome. Within the Neyman-Rubin potential outcomes framework, let $Y _ { i } ( 0 )$ and $Y _ { i } ( 1 )$ denote the potential outcomes under control $( A _ { i } = 0 )$ and treatment $( A _ { i } = 1 )$ , respectively. The observed outcome satisfies $Y _ { i } = Y _ { i } ( A _ { i } )$ . Throughout this work, we interpret $Y _ { i }$ as a measure of welfare or utility, where higher values correspond to more desirable outcomes.

Our objective is to use the observational data $\{ Z _ { i } \} _ { i = 1 } ^ { n }$ to guide personalized policy interventions in settings where unobserved confounding may be present

Example 1 (Job Training). Let $A _ { i }$ denote whether individual i participates in a job training program. Let $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ denote individual i’s post-program earnings under participation and nonparticipation, respectively. To improve labor market outcomes, the policymaker decides whether individual i should receive the job training based on observed covariates $X _ { i } ,$ such as years of education and pre-program earnings.

Example 2 (Head Start Enrollment). Let $A _ { i }$ indicate whether child i is enrolled in the Head Start program. Let $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ denote academic outcomes under enrollment and non-enrollment (e.g., test scores or school readiness). To improve early childhood development, the policymaker allocates slots based on characteristics $X _ { i }$ , including household income, parental education, and number of siblings.

## 2.1 The Marginal Sensitivity Model

Our analysis builds on the marginal sensitivity model (MSM) introduced by Tan (2006) to control the selection bias caused by unobserved confounders. This model relaxes the unconfoundedness assumption by allowing for unobserved confounders $U \in \mathbb { R } ^ { k }$ , where $k \in \mathbb { N } ^ { + }$ is unknown. Here, we assume that U is a vector but have no prior knowledge of its dimension or other characteristics. We denote $P _ { o }$ as the true distribution of $O \equiv ( X , Y ( 1 ) , Y ( 0 ) , A , U )$ . The true propensity score is given by $e _ { o } ( x , u ) = \mathbb { P } _ { P _ { o } } [ A = 1 | X = x , U = u ]$ , which accounts for both observed covariates and unobserved confounders. In contrast, the nominal propensity score, defined as $e ( x ) = \mathbb { P } _ { P _ { o } } [ A =$ $1 | X = x ]$ In the MSM, unobserved confounders are assumed to have a bounded influence on the odds of treatment assignment, restricting the extent of selection bias introduced by unobserved confounders.

We now present the formal specification of the MSM, a widely used framework for sensitivity analysis in the causal inference literature (see, e.g., Zhao et al. (2019); Dorn and Guo (2023); Oprescu et al. (2023); Dorn et al. (2025)).

Assumption 2.1 (Marginal Sensitivity Model). Suppose there exists a vector of unobserved confounders $U \in \mathbb { R } ^ { k }$ such that

$$
(Y (1), Y (0)) \perp A \mid (X, U).
$$

The distribution of $O \equiv ( X , Y ( 1 ) , Y ( 0 ) , A , U )$ satisfies the selection bias condition with $1 \leq \Lambda <$ ∞ if the following inequality holds $P _ { o } .$ -almost surely,

$$
\frac {1}{\Lambda} \leq \frac {e _ {o} (x , u) / (1 - e _ {o} (x , u))}{e (x) / (1 - e (x))} \leq \Lambda .\tag{2.1}
$$

Remark 2.1. In Example 1, the latent variable U may capture unobserved factors, such as motivation, expected gains from training, or participation constraints, that affect both training take-up and potential earnings. Similarly, in Example 2, U may reflect unobserved characteristics of the child and family, including aspects of socioeconomic background, parental preferences, and investments in early human capital.

In practice, the sensitivity parameter Λ in Assumption 2.1 is selected by policymakers based on their prior beliefs regarding the extent of unobserved confounding and the resulting selection bias. Implicitly, it is assumed that policymakers choose Λ such that Assumption 2.1 is satisfied. The special case of $\Lambda = 1$ corresponds to the unconfoundedness, also known as the selection-onobservables assumption. Higher values of Λ impose fewer restrictions on the degree of unobserved confounding, allowing for greater potential selection bias.

Remark 2.2. The sensitivity parameter, Λ, measures the allowed deviation from the unconfoundedness benchmark. Following Hsu and Small (2013), we calibrate Λ assuming the effect of an unobserved confounder on the treatment-assignment odds is comparable in magnitude to that of observed covariates. In practice, we quantify this magnitude by examining how the estimated treatmentassignment odds change when each observed covariate is omitted in turn.

When credible bounds from an alternative identification strategy are available, they can also serve as a reference for calibrating Λ. Following Masten and Poirier (2018), we ask how much the baseline unconfoundedness assumption must be relaxed before the MSM-implied bounds become conservative relative to these reference bounds. For the JTPA study in Section 6.1, we use the IV bounds induced by randomized eligibility assignment as this benchmark. Specifically, we choose the smallest value of Λ such that the MSM-implied lower bound do not exceed the IV lower bound.

## 2.2 Policy Learning under the MSM

In this subsection, we provide a brief review of policy learning under the assumption of unconfoundedness $( \Lambda = 1 )$ . We then explore how to learn a robust policy when this assumption is violated, within the framework of the marginal sensitivity model. Throughout the rest of this work, we fix the sensitivity parameter $\Lambda \geq 1$

A possibly randomized policy π maps covariates $x \in \mathcal { X }$ to treatment assignment probabilities, with $\pi ( x )$ denoting the probability of receiving treatment $A = 1$ . The policymaker can pre-specify a policy class Π, defined as a collection of Borel measurable functions mapping from X to $[ 0 , 1 ]$ This class typically incorporates application-specific constraints, including budgetary limitations, structural assumptions, and fairness requirements; see Section 4.2 for concrete examples.

We begin by recalling the framework of policy learning under the unconfoundedness assumption. Given a predefined policy class Π, the mean-optimal policy learning aims to identify a policy that maximizes the expected outcome:

$$
\pi^ {\star} \in \underset {\pi \in \Pi} {\operatorname{argmax}}   \mathbb {E} \left[ Y \left(\pi (X)\right) \right],\tag{2.2}
$$

where $Y \left( \pi \left( X \right) \right) = \pi ( X ) Y ( 1 ) + ( 1 - \pi ( X ) ) Y ( 0 )$ . For simplicity, let $W _ { 1 } ( \pi ) = \mathbb { E } \left[ Y \left( \pi ( X ) \right) \right]$ denote the expected welfare function. Moreover, the regret of deploying a policy $\pi \in \Pi$ is defined as

$$
\operatorname{Reg} (\pi) = \sup _ {\pi^ {\prime} \in \Pi} \mathbb {E} [ Y (\pi^ {\prime} (X)) ] - \mathbb {E} [ Y (\pi (X)) ].\tag{2.3}
$$

It is evident that a policy maximizing $W _ { 1 } ( \pi )$ simultaneously minimizes the regret given in Eq. (2.3).

A central objective in policy learning is policy evaluation, that is, identifying and estimating the criterion function. Once an estimate of $W _ { 1 } ( \cdot )$ is available, the optimal policy can be estimated by maximizing the estimated welfare over the policy class. Unconfoundedness $( \mathrm { i . e . , } \Lambda = 1 )$ is a widely adopted assumption for identifying the expected welfare function; see Kitagawa and Tetenov (2018); Athey and Wager (2021); Zhou et al. (2023). Under this assumption, various methods such as inverse probability weighting (IPW) can be employed to identify and estimate $W _ { 1 } ( \pi )$ . In particular, the expected welfare function can be expressed as

$$
W _ {1} (\pi) = \mathbb {E} [ Y (0) ] + \mathbb {E} [ \tau (X) \pi (X) ]\tag{2.4}
$$

$$
= \mathbb {E} \left[ \frac {Y A \pi (X)}{e (X)} + \frac {Y (1 - A) (1 - \pi (X))}{1 - e (X)} \right],\tag{2.5}
$$

where $e ( x )$ is the nominal propensity score, and $\tau ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ]$ is the conditional average treatment effect (CATE). When $\Lambda = 1$ , the outcome regression, nominal propensity score, and CATE are all identifiable.

When unconfoundedness fails $( \mathrm { i . e . , } \Lambda > 1 )$ , the expected welfare $W _ { 1 } ( \pi )$ becomes unidentifiable, and policies based on Eq. (2.4) or Eq. (2.5) lack reliable guarantees. As a result, $W _ { 1 } ( \pi )$ is not a suitable criterion for policy learning in the presence of unobserved confounding.

We adopt a distributionally robust optimization (DRO) framework for policy learning, replacing $W _ { 1 } ( \pi )$ with its worst-case counterpart over distributions consistent with MSM specified in Assumption 2.1. Within this framework, we construct two welfare criteria that are both robust to unobserved confounding and identifiable, and derive the policy by optimizing one of them.

To implement the DRO, we first characterize the distributional uncertainty set over the counterfactual distribution of $( Y ( 1 ) , Y ( 0 ) , X , U , A )$ subject to the constraints imposed by Assumption $2 . 1$ Formally, let ${ \mathcal { P } } ( \Lambda )$ denote the set of all probability distributions Q on $\mathbb { R } ^ { 2 } \times \mathcal { X } \times \{ 0 , 1 \} \times \mathbb { R } ^ { k }$ satisfying the following conditions:

(1) If (X, Y(0), Y(1), A, U) ∼ Q, then (Y(0), Y(1)) |= A | (X, U) under $Q ;$

(2) If $Y = A Y ( 1 ) + ( 1 - A ) Y ( 0 )$ , then the distribution of $( X , Y , A )$ under $Q$ is identical to the observed-data distribution $P ;$

(3) The odds ratio between the true propensity score and the nominal propensity score lies in $[ 1 / \Lambda , \Lambda ] , \mathrm { i } . \mathbf { e } .$

$$
\frac {1}{\Lambda} \leq \frac {\mathbb {P} _ {Q} (A = 1 | X , U) / \mathbb {P} _ {Q} (A = 0 | X , U)}{\mathbb {P} _ {Q} (A = 1 | X) / \mathbb {P} _ {Q} (A = 0 | X)} \leq \Lambda .
$$

We now present two complementary policy learning methods that are robust to unobserved confounding under the MSM framework. Our first approach is the max-min expected welfare method. Specifically, the worst-case welfarefunction $W _ { \Lambda } : \Pi  \mathbb { R }$ is defined as

$$
W _ {\Lambda} (\pi) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (\pi (X)) \right].
$$

The corresponding max-min welfare (MMW) policy is given by

$$
\pi_ {W, \Lambda} \equiv \pi_ {W} (\cdot ; \Lambda) \in \underset {\pi \in \Pi} {\operatorname{argmax}} W _ {\Lambda} (\pi).\tag{2.6}
$$

Under the worst-case welfare function $W _ { \Lambda }$ , we define the confounding-robust welfare regret (CRWregret) of a policy $\pi \in \Pi$ , relative to the best possible policy in Π, as

$$
\operatorname{Reg} _ {W} (\pi) = \sup _ {\pi^ {\prime} \in \Pi} W _ {\Lambda} (\pi^ {\prime}) - W _ {\Lambda} (\pi).\tag{2.7}
$$

The second approach builds on the conditional average treatment effect (CATE). When $\Lambda = 1$ unconfoundedness holds and, the policy learning problem reduces to maximizing the policy improvement function $\mathbb { E } [ \tau ( X ) \pi ( X ) ]$ ]. To handle the more general case with $\Lambda \geq 1$ , we extend this objective to the worst-case policy improvementfunction $\Delta _ { \Lambda } : \Pi  \mathbb { R }$ , defined as

$$
\Delta_ {\Lambda} (\pi) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ \pi (X) (Y (1) - Y (0)) \right].\tag{2.8}
$$

The corresponding max-min improvement (MMI) policy is given by

$$
\pi_ {\Delta , \Lambda} \equiv \pi_ {\Delta} (\cdot ; \Lambda) \in \underset {\pi \in \Pi} {\operatorname{argmax}}   \Delta_ {\Lambda} (\pi).\tag{2.9}
$$

Under the criterion $\Delta _ { \Lambda }$ , we define the confounding-robust policy improvement regret (CRI-regret)

of a policy $\pi \in \Pi$ , relative to the best possible policy in Π, as

$$
\operatorname{Reg} _ {\Delta} (\pi) = \sup _ {\pi^ {\prime} \in \Pi} \Delta_ {\Lambda} (\pi^ {\prime}) - \Delta_ {\Lambda} (\pi).\tag{2.10}
$$

The quantity $\begin{array} { r } { \Delta _ { \Lambda } ( \pi ) = \operatorname* { i n f } _ { Q \in { \mathcal P } ( \Lambda ) } \mathbb { E } _ { Q } \left[ Y ( \pi ( X ) ) - Y ( 0 ) \right] } \end{array}$ represents the worst-case welfare gain of policy π relative to the baseline policy $\pi _ { 0 } ( x ) = 0$ . In fact, this approach can be generalized to the policy improvement of π against any given baseline policy $\pi _ { 0 }$ , with the optimal policy obtained by:

$$
\max _ {\pi \in \Pi} \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (\pi (X)) - Y \left(\pi_ {0} (X)\right) \right].\tag{2.11}
$$

We refer to any solution to (2.11) as a baseline-relative MMI policy. Since $Y ( \pi ( X ) ) - Y \left( \pi _ { 0 } ( X ) \right) =$ $\left( \pi ( X ) - \pi _ { 0 } ( X ) \right) \left( Y ( 1 ) - Y ( 0 ) \right)$ , both (2.11) and the resulting optimal policy can be derived using the similar identification and estimation strategy as in (2.9). This objective is closely related to that in Kallus and Zhou (2021). Their method proceeds by constructing a conservative lower bound for the inner minimization in (2.11) based on an uncertainty set for putative propensity weights. See (B.1) for the sharp characterization of (2.11), and Remarks D.1 and D.2 provides a detailed comparison with Kallus and Zhou (2021).

Remark 2.3. When $\Lambda \ = \ 1$ , the MMW policy in Eq. (2.6) coincides with the MMI policy in Eq. (2.9). However, in the presence of unobserved confounding (i.e. $\Lambda > 1 )$ , the two policies may differ. A detailed discussion of the differences between these policies is deferred to Section 3.

## 3 Identification

Within our framework, learning a confounding-robust policy requires identifying and estimating either the worst-case welfare function $W _ { \Lambda } ( \pi )$ or the worst-case policy improvement function $\Delta _ { \Lambda } ( \pi )$ In this section, we formally characterize these criterion functions by leveraging partial identification results for conditional means and CATE under the MSM. We then develop doubly robust/orthogonal moment functions for identifying $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ , which enable the construction of efficient estimators.

For notational simplicity, we define two quantile functions $q _ { \Lambda } ^ { \pm } ( x , a )$ as

$$
\begin{array}{l} q _ {\Lambda} ^ {+} (x, a) = \inf \left\{q: F (q | x, a) \geq \frac {\Lambda}{1 + \Lambda} \right\}, \\ q _ {\Lambda} ^ {-} (x, a) = \inf \left\{q: F (q | x, a) \geq \frac {1}{1 + \Lambda} \right\}, \end{array}
$$

where $F ( \cdot | x , a )$ denotes the CDF of Y given $X = x$ and $A = a$ . Moreover, let $e _ { a } ( x ) = \mathbb { P } ( A =$ $a | X = x )$ for $a \in \{ 0 , 1 \}$ , so that $e _ { 1 } ( x ) = e ( x )$ and $e _ { 0 } ( x ) = 1 - e ( x )$

## 3.1 Identification of Robust Criteria

In this subsection, we identify and characterize the robust criterion functions $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ Additionally, we derive the first-best policies optimized under these robust criteria. Let $\Pi _ { o }$ denote the set of all measurable policies:

$$
\Pi_ {o} = \left\{\pi : \mathcal {X} \rightarrow [ 0, 1 ] \text {   is   Borel   measurable } \right\}.
$$

A first-best policy refers to a policy that maximizes the criterion function over the unrestricted policy class $\Pi _ { o }$

We begin by identifying the worst-case welfare $W _ { \Lambda } ( \pi )$ . When $\Lambda > 1$ , the true conditional mean function $\mu _ { o } ( x , a ) = \mathbb { E } _ { P _ { o } } [ Y ( a ) | X = x ]$ is no longer point identified due to unobserved confounding. Instead, we characterize the sharp bounds for $\mu _ { o } ( x , a )$ using Proposition D.1 in Section D, which also implies a sharp lower bound for the true welfare function $\mathbb { E } _ { P _ { o } } [ Y ( \pi ) ]$ . Specifically, the proposition provides closed-form expressions for the upper and lower bounds, denoted $\mu _ { \Lambda } ^ { \pm } ( x , a )$ which are given by

$$
\mu_ {\Lambda} ^ {\pm} (x, a) = \mathbb {E} \left[ Y \mathbb {1} \{A = a \} \left[ 1 + \frac {1 - e _ {a} (X)}{e _ {a} (X)} \Lambda^ {\pm \operatorname{sgn} \left(Y - q _ {\Lambda} ^ {\pm} (X, a)\right)} \right] | X = x \right],
$$

where $\operatorname { s g n } ( t ) = 1 \operatorname { i f } t \geq 0$ and −1 otherwise.

Theorem 3.1. Under Assumption 2.1, for any policy $\pi \in \Pi$ , we have

$$
W _ {\Lambda} (\pi) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (\pi (X)) \right] = \mathbb {E} \left[ \mu_ {\Lambda} ^ {-} (X, 1) \pi (X) + \mu_ {\Lambda} ^ {-} (X, 0) \left(1 - \pi (X)\right) \right].
$$

Moreover, a first-best MMW policy that solves Eq. (2.6) with $\Pi = \Pi _ { o }$ is given by

$$
\pi_ {W, \Lambda} ^ {\star} (x) = \mathbb {1} \left\{\mu_ {\Lambda} ^ {-} (x, 1) - \mu_ {\Lambda} ^ {-} (x, 0) > 0 \right\}.
$$

The identification of $\Delta _ { \Lambda } ( \pi )$ is established in Theorem 3.2, which follows as a corollary of Proposition D.2. To formalize the result, define $\tau _ { \Lambda } ^ { - } ( x ) = \mu _ { \Lambda } ^ { - } ( x , 1 ) - \mu _ { \Lambda } ^ { + } ( x , 0 )$

Theorem 3.2. Under Assumption 2.1, for any policy $\pi \in \Pi$ , we have

$$
\Delta_ {\Lambda} (\pi) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ (Y (1) - Y (0)) \pi (X) \right] = \mathbb {E} \left[ \tau_ {\Lambda} ^ {-} (X) \pi (X) \right].
$$

Moreover, a first-best MMI policy that solves Eq. (2.9) with $\Pi = \Pi _ { o }$ is given by

$$
\pi_ {\Delta , \Lambda} ^ {\star} (x) = \mathbb {1} \left\{\tau_ {\Lambda} ^ {-} (x) > 0 \right\} = \mathbb {1} \left\{\mu_ {\Lambda} ^ {-} (x, 1) - \mu_ {\Lambda} ^ {+} (x, 0) > 0 \right\}.
$$

Theorem 3.1 characterizes the first-best MMW policy $\pi _ { W , \Lambda } ^ { \star } ( x )$ , which assigns treatment by comparing the lower bounds $\mu _ { \Lambda } ^ { - } ( x , 1 )$ and $\mu _ { \Lambda } ^ { - } ( x , 0 )$ . In contrast, Theorem 3.2 shows that the firstbest MMI policy $\pi _ { \Delta , \Lambda } ^ { \star } ( x )$ assigns treatment by comparing $\mu _ { \Lambda } ^ { - } ( x , 1 )$ with the upper bound $\mu _ { \Lambda } ^ { + } ( x , 0 )$ As a result, the MMI policy is more conservative: it treats only when the worst-case treated outcome exceeds the best-case control outcome. Although the unrestricted policy class Π permits randomized policies, the first-best MMW and MMI policies are deterministic and unique up to tie-breaking on indifference sets, because the corresponding robust criteria are affine functionals of $\pi . ^ { 3 }$

We compare our first-best MMW and MMI policies with the Bayes decision rule (referred to as a policy in our framework) of Pu and Zhang (2021), hereafter referred to as PZ-policy.

Remark 3.1. The first-best MMW policy can also be interpreted through a CATE-based approach, as it assigns treatment by comparing the worst-case potential outcomes. Since this rule is confined to a comparison of identified lower bounds, it is uninformative about the magnitude of the true treatment effect. For this reason, we turn our focus to the MMI policy, and a comparison between the first-best MMI policy and PZ-policy is more instructive.

While both approaches assign treatment based on CATE bounds, they differ in two aspects. First is the construction of the bounds: MMI policy uses the MSM framework, whereas the PZ-policy employs an IV approach. Second, and more importantly, the two policies differ in how they apply decision rules to these bounds. The first-best MMI policy is more conservative and follows a strict criterion: treatment is assigned only when the lower bound of the CATE exceeds zero. In contrast, the PZ-policy adopts a three-case decision rule: it treats if the lower bound is positive, withholds if the upper bound is negative, and when the bounds straddle zero, treats if the upper bound has a larger absolute value than the lower bound. This contrast highlights why our MMI policy is more conservative: it avoids intervention in the presence of ambiguity, whereas the PZ-policy allows for treatment under uncertainty (see Fig. 1).

(a) When $\tau _ { \Lambda } ^ { - } < 0 < \frac { 1 } { 2 } ( \tau _ { \Lambda } ^ { + } + \tau _ { \Lambda } ^ { - } )$ , the PZ policy recommends treatment $( \pi = 1 ) .$ , while the MMI policy recommends no treatment $( \pi = 0 )$ .
(b) When $\begin{array} { r } { \frac { 1 } { 2 } ( \tau _ { \Lambda } ^ { + } + \tau _ { \Lambda } ^ { - } ) < 0 , } \end{array}$ both the PZ and MMI policies recommend no treatment $( \pi = 0 )$
Figure 1: Comparison of the MMI and PZ decision rules when applied to a given ambiguous CATE interval $[ \tau _ { \Lambda } ^ { - } , \tau _ { \Lambda } ^ { + } ]$

## 3.2 Doubly/Locally Robust Scores

To effectively learn the confounding-robust policy, it is essential to estimate the entire worst-case welfare function with minimal estimation error. To facilitate the use of modern ML methods, we derive doubly robust scores for the worst-case criteria, $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ , such that the estimation of nuisance parameters has no first-order influence on the resulting policy evaluation. For notational convenience, we may write $\pi ( a | x ) = a \pi ( x ) + ( 1 - a ) \left( 1 - \pi ( x ) \right)$ , which compactly represents the probability that treatment $a \in \{ 0 , 1 \}$ is assigned under policy π given covariate x. Moreover, we assume that the conditional distribution of $Y _ { i } | X _ { i } , A _ { i }$ admits a bounded density, as in Dorn and Guo (2023) and Dorn et al. (2025).

Assumption 3.1. For each $( x , a ) \in \mathcal { X } \times \{ 0 , 1 \}$ , the conditional distribution $\textstyle F ( y | x , a )$ is continuous with a uniformly bounded density $f ( \boldsymbol { y } | \boldsymbol { x } , a )$ that is positive on the interior of its support.

## Doubly Robust Score for $W _ { \Lambda } ( \pi )$

We begin by presenting the doubly robust score for estimating $W _ { \Lambda } ( \pi )$ . As shown in Theorem 3.1, this naturally leads to the following moment condition:

$$
\mathbb {E} \left[ \mu_ {\Lambda} ^ {-} (X, 1) \pi (1 | X) + \mu_ {\Lambda} ^ {-} (X, 0) \pi (0 | X) \right] - W _ {\Lambda} (\pi) = 0.
$$

For $t \in \{ 0 , 1 \}$ , let

$$
g _ {t} \left(z; e, q _ {\Lambda} ^ {-}\right) = y \mathbb {1} \{a = t \} \left[ 1 + \frac {1 - e _ {t} (x)}{e _ {t} (x)} \Lambda^ {- \operatorname{sgn} \left(y - q _ {\Lambda} ^ {-} (x, t)\right)} \right].
$$

Using explicit expressions for $\mu _ { \Lambda } ^ { \pm }$ given in Proposition D.1, the moment condition above can be rewritten as 1 1

$$
\mathbb {E} \left[ \sum_ {t \in \{0, 1 \}} g _ {t} (Z; e, q _ {\Lambda} ^ {-}) \pi (t | X) \right] - W _ {\Lambda} (\pi) = 0.\tag{3.1}
$$

Here, $W _ { \Lambda } ( \pi )$ is the parameter of interest, while $e ( \cdot )$ and $q _ { \Lambda } ^ { - } ( \cdot , \cdot )$ are two unknown nuisance parameters that can be estimated using ML/nonparametric methods.

A plug-in estimator for $W _ { \Lambda } ( \pi )$ can be constructed by substituting estimates into the score function $g _ { t } ( Z ; \cdot )$ and averaging over the sample, yielding $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } \sum _ { t \in \{ 0 , 1 \} } g _ { t } \left( Z _ { i } ; \widehat { e } , \widehat { q _ { \Lambda } } \right) \pi ( t | X _ { i } ) } \end{array}$ b bHowever, this plug-in estimator is sensitive to estimators of the nominal propensity score and the quantile functions, and may suffer from severe bias. This motivates the doubly robust estimator proposed in this section.

Following Chernozhukov et al. (2018, 2022), we construct the doubly robust score for $W _ { \Lambda } ( \pi )$ as follows:

$$
\begin{array}{l} \phi_ {t} ^ {-} (z; e, q _ {\Lambda} ^ {-}, \rho_ {1, \Lambda} ^ {-}, \rho_ {0, \Lambda} ^ {-}) \\ = y \mathbb {1} \{a = t \} \left[ 1 + \frac {1 - e _ {t} (x)}{e _ {t} (x)} \Lambda^ {- \mathrm{sgn} (y - q _ {\Lambda} ^ {-} (x, t))} \right] \\ + q _ {\Lambda} ^ {-} (x, t) \mathbb {1} \{a = t \} \frac {1 - e _ {t} (x)}{e _ {t} (x)} (\Lambda - \Lambda^ {- 1}) \left[ \frac {1}{1 + \Lambda} - \mathbb {1} \{y <   q _ {\Lambda} ^ {-} (x, t) \} \right] \\ - \frac {1}{e _ {t} (x)} [ \Lambda \rho_ {1, \Lambda} ^ {-} (x, t) + \Lambda^ {- 1} \rho_ {0, \Lambda} ^ {-} (x, t) ] [ \mathbb {1} \{a = t \} - e _ {t} (x) ], \end{array}\tag{3.2}
$$

where

$$
\begin{array}{r l} & {\rho_ {1, \Lambda} ^ {\pm} (x, t) = \mathbb {E} \left[ Y \mathbb {1} \left\{Y <   q _ {\Lambda} ^ {\pm} (X, A) \right\} | X = x, A = t \right],} \\ & {\rho_ {0, \Lambda} ^ {\pm} (x, t) = \mathbb {E} \left[ Y \mathbb {1} \left\{Y > q _ {\Lambda} ^ {\pm} (X, A) \right\} | X = x, A = t \right].} \end{array}\tag{3.3}
$$

Let $\eta _ { W , \Lambda } = \left( e , q _ { \Lambda } ^ { - } , \rho _ { 1 , \Lambda } ^ { - } , \rho _ { 0 , \Lambda } ^ { - } \right)$ denote the tuple of nuisance functions, the doubly robust score for $W _ { \Lambda } ( \pi )$ is given by

$$
\psi_ {W} (z, \pi ; \eta_ {W, \Lambda}) = \sum_ {t \in \{0, 1 \}} \phi_ {t} ^ {-} (z; \eta_ {W, \Lambda}) \pi (t | x).
$$

The following Proposition 3.1 establishes the Neyman orthogonality for the moment condition. Moreover, the function $\psi _ { W } ( z , \pi ; \eta _ { W , \Lambda } ) - W _ { \Lambda } ( \pi )$ serves as the efficient influence function for estimating $W _ { \Lambda } ( \pi )$

Proposition 3.1. Under Assumptions 2.1 and 3.1, for any policy $\pi \in \Pi$ , the score function ψ<sub>W</sub> satisfies:

$$
\mathbb {E} \left[ \psi_ {W} (Z, \pi ; \eta_ {W, \Lambda}) \right] = W _ {\Lambda} (\pi). \tag {1}
$$

(2) For any $\widetilde { \eta } _ { W , \Lambda } = ( { \widetilde { e } } , { \widetilde { q } } _ { \Lambda } ^ { - } , { \widetilde { \rho } } _ { 1 , \Lambda } ^ { - } , { \widetilde { \rho } } _ { 0 , \Lambda } ^ { - } )$ with $\widetilde { e } : \mathcal { X } \to ( 0 , 1 ) , \widetilde { q } _ { \Lambda } ^ { - } : \mathcal { X } \times \{ 0 , 1 \} \to \mathbb { R }$ , and $\widetilde { \rho } _ { 1 , \Lambda } ^ { - } , \widetilde { \rho } _ { 0 , \Lambda } ^ { - } : \mathcal { X } \times \{ 0 , 1 \}  \mathbb { R }$ e e e, the pathwise (or Gateaux) derivative satisfies

$$
\frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \psi_ {W} \left(Z, \pi ; \eta_ {W, \Lambda} + r \left(\widetilde {\eta} _ {W, \Lambda} - \eta_ {W, \Lambda}\right)\right) \right] _ {r = 0} = 0.
$$

Remark 3.2. The framework of Kallus and Zhou (2021) relies on estimating the nominal propensity score at the parametric rate of $O _ { P } ( n ^ { - 1 / 2 } )$ to achieve an $n ^ { - 1 / 2 }$ regret bound. However, this rate can only be attained under correct model specification, ruling out the use of ML and nonparametric methods. In contrast, our framework allows for the nuisance components to be estimated at a slower rate of $o _ { P } { \left( n ^ { - 1 / 4 } \right) }$ , enabling the use of data-adaptive, model-agnostic estimation techniques.

## Doubly Robust Score for $\Delta _ { \Lambda } ( \pi )$

We then construct the doubly robust score of $\Delta _ { \Lambda } ( \pi )$ . Theorem 3.2 implies the moment condition as follows:

$$
\mathbb {E} \left[ \tau_ {\Lambda} ^ {-} (X) \pi (X) \right] - \Delta_ {\Lambda} (\pi) = 0.
$$

Let $\eta _ { \Delta , \Lambda } = \left( e , q _ { \Lambda } ^ { \pm } , \rho _ { 1 , \Lambda } ^ { \pm } , \rho _ { 0 , \Lambda } ^ { \pm } \right)$ denote the tuple of nuisance parameters. The doubly robust score for $\Delta _ { \Lambda } ( \pi )$ can be constructed as

$$
\psi_ {\Delta} \left(z, \pi ; \eta_ {\Delta , \Lambda}\right) = \pi (x) \left[ \phi_ {1} ^ {-} \left(z; e, q _ {\Lambda} ^ {-}, \rho_ {1, \Lambda} ^ {-}, \rho_ {0, \Lambda} ^ {-}\right) - \phi_ {0} ^ {+} \left(z; e, q _ {\Lambda} ^ {+}, \rho_ {1, \Lambda} ^ {+}, \rho_ {0, \Lambda} ^ {+}\right) \right],
$$

where

$$
\begin{array}{r l} & {\phi_ {t} ^ {+} \left(z; e, q _ {\Lambda} ^ {+}, \rho_ {1, \Lambda} ^ {+}, \rho_ {0, \Lambda} ^ {+}\right)} \\ & {= y \mathbb {1} \left\{a = t \right\} \left[ 1 + \frac {1 - e _ {t} (x)}{e _ {t} (x)} \Lambda^ {\mathrm{sgn} \left(y - q _ {\Lambda} ^ {+} (x, t)\right)} \right]} \\ & {- q _ {\Lambda} ^ {+} (x, t) \mathbb {1} \left\{a = t \right\} \frac {1 - e _ {t} (x)}{e _ {t} (x)} \left(\Lambda - \Lambda^ {- 1}\right) \left[ \frac {\Lambda}{1 + \Lambda} - \mathbb {1} \left\{y <   q _ {\Lambda} ^ {+} (x, t) \right\} \right]} \\ & {- \frac {1}{e _ {t} (x)} \left[ \Lambda^ {- 1} \rho_ {1, \Lambda} ^ {+} (x, t) + \Lambda \rho_ {0, \Lambda} ^ {+} (x, t) \right] [ \mathbb {1} \left\{a = t \right\} - e _ {t} (x) ].} \end{array}\tag{3.4}
$$

We summarize several crucial properties of $\psi _ { \Delta }$ in the following proposition.

Proposition 3.2. Under Assumptions 2.1 and 3.1, for any policy $\pi \in \Pi$ , the score function $\psi _ { \Delta }$ satisfies:

$$
\mathbb {E} \left[ \psi_ {\Delta} (Z, \pi ; \eta_ {\Delta , \Lambda}) \right] = \Delta_ {\Lambda} (\pi). \tag {1}
$$

(2) For any $\widetilde { \eta } _ { \Delta , \Lambda } = \left( \widetilde { e } , \widetilde { q } _ { \Lambda } ^ { \pm } , \widetilde { \rho } _ { 1 , \Lambda } ^ { \pm } , \widetilde { \rho } _ { 0 , \Lambda } ^ { \pm } \right)$ with $\widetilde { e } : \mathcal { X } \to ( 0 , 1 ) , \widetilde { q } _ { \Lambda } ^ { \pm } : \mathcal { X } \times \{ 0 , 1 \} \to \mathbb { R } ,$ and $\widetilde { \rho } _ { 1 , \Lambda } ^ { \pm } , \widetilde { \rho } _ { 0 , \Lambda } ^ { \pm } : \mathcal { X } \times \{ 0 , 1 \}  \mathbb { R }$ e e e, the pathwise (or Gateaux) derivative satisfies

$$
\frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \psi_ {\Delta} (Z, \pi ; \eta + r (\widetilde {\eta} _ {\Delta , \Lambda} - \eta_ {\Delta , \Lambda})) \right] _ {r = 0} = 0.
$$

Remark 3.3. The efficient influence functions for $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ are given by $\psi _ { W } ( z , \pi ; \eta _ { W , \Lambda } ) -$ $W _ { \Lambda } ( \pi )$ and $\psi _ { \Delta } ( z , \pi ; \eta _ { \Delta , \Lambda } ) - \Delta _ { \Lambda } ( \pi )$ , respectively; see Newey (1994).

## 4 Algorithm and Performance Guarantees

In this section, we focus on the algorithms for estimating the MMW and MMI policies, as defined in Section 2.2. Let ${ \widehat \pi } _ { W , \Lambda }$ and $\widehat { \pi } _ { \Delta , \Lambda }$ denote the estimated MMW and MMI policies, respectively. b bWe also provide theoretical guarantees for these estimated policies in the form of upper bounds on their associated regrets. Specifically, we derive asymptotic upper bounds on the CRW-regret $\mathrm { R e g } _ { W } \left( \widehat { \pi } _ { W , \Lambda } \right)$ and the CRI-regret $\mathrm { R e g } _ { \Delta } \left( \widehat { \pi } _ { \Delta , \Lambda } \right)$ , as defined in Eq. (2.7) and Eq. (2.10).

## 4.1 Algorithm

We estimate the MMW and MMI policies using a two-stage procedure: (1) estimation of the robust criterion, either $W _ { \Lambda } ( \pi )$ or $\Delta _ { \Lambda } ( \pi )$ , using K-fold cross-fitting; and (2) policy optimization based on the estimated objective.

Given the doubly robust scores in Section 3.2, we first estimate $e ( \cdot ) , q _ { \Lambda } ^ { \pm } ( \cdot , \cdot )$ and $\rho _ { t , \Lambda } ^ { \pm } ( \cdot , \cdot )$ for $t \in \{ 0 , 1 \}$ . To this end, we divide the sample into K evenly-sized folds $\cup _ { k = 1 } ^ { K } \mathcal { T } _ { k }$ . For each fold

$$
k \in [ K ]
$$

$$
e, q _ {\Lambda} ^ {\pm}
$$

$$
\rho_ {t, \Lambda} ^ {\pm}
$$

$$
(K - 1)
$$

$$
\widehat {\eta} _ {W, \Lambda} ^ {- k} = \left(\widehat {e} ^ {- k}, q _ {\Lambda} ^ {-, - k}, \rho_ {1, \Lambda} ^ {-, - k}, \rho_ {0, \Lambda} ^ {-, - k}\right) \quad \text { and } \quad \widehat {\eta} _ {\Delta , \Lambda} ^ {- k} = \left(\widehat {e} ^ {- k}, q _ {\Lambda} ^ {\pm , - k}, \rho_ {1, \Lambda} ^ {\pm , - k}, \rho_ {0, \Lambda} ^ {\pm , - k}\right)
$$

Using these estimates, we construct the cross-fitted estimator for $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ as

$$
\widehat {W} _ {\Lambda , n} (\pi) = \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \psi_ {W} \left(Z _ {i}, \pi ; \widehat {\eta} _ {W, \Lambda} ^ {- k}\right) \text {and} \widehat {\Delta} _ {\Lambda , n} (\pi) = \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \psi_ {\Delta} \left(Z _ {i}, \pi ; \widehat {\eta} _ {\Delta , \Lambda} ^ {- k}\right)\tag{4.1}
$$

The final policy is selected from a pre-specified policy class $\Pi _ { n }$ by maximizing the estimated objective, either $\widehat { W } _ { \Lambda , n } ( \pi )$ or $\widehat { \Delta } _ { \Lambda , n } ( \pi )$ . The complete procedures for estimating the MMW and MMI b bpolicies are summarized in Algorithm 1 and Algorithm 2, respectively.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Max-Min Welfare (MMW) Policy Learning
1: Input: Sample $\{Y_i, X_i, A_i\}_{i=1}^n$, sensitivity parameter $\Lambda \geq 1$, and a policy class $\Pi_n$
2: Choose $K \in \mathbb{N}^+$, and partition the sample into $K$ equally sized folds $\cup_{k=1}^K \mathcal{I}_k$
3: for each $k = 1, \ldots, K$ do
4: Fit estimators for $\eta_{W,\Lambda} = \left(e, q_\Lambda^-, \rho_{1,\Lambda}^-, \rho_{0,\Lambda}^-\right)$ using the other $(K-1)$ folds, $\mathcal{I}_k^c \equiv [n] \backslash \mathcal{I}_k$, denoted by $\widehat{\eta}_{W,\Lambda}^{-k}$.
5: end for
6: Estimate $W_\Lambda(\pi)$ as
$\widehat{W}_{\Lambda,n}(\pi) = \frac{1}{n} \sum_{k=1}^{K} \sum_{i \in \mathcal{I}_k} \psi_W(Z_i, \pi; \widehat{\eta}_{W,\Lambda}^{-k})$.
7: Return $\widehat{\pi}_{W,\Lambda} = \argmax_{\pi \in \Pi_n} \widehat{W}_{\Lambda,n}(\pi)$.
</div>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Max-Min Improvement (MMI) Policy Learning
1: Input: Sample $\{Y_i, X_i, A_i\}_{i=1}^n$, sensitivity parameter $\Lambda \geq 1$, and a policy class $\Pi_n$
2: Choose $K \in \mathbb{N}^+$, and partition the sample into $K$ equally sized folds $\cup_{k=1}^K \mathcal{I}_k$
3: for each $k = 1, \ldots, K$ do
4: Fit estimators for $\eta_{\Delta,\Lambda} = \left(e, q_{\Lambda}^{\pm}, \rho_{1,\Lambda}^{\pm}, \rho_{0,\Lambda}^{\pm}\right)$ using the other $(K-1)$ folds, $\mathcal{I}_k^c \equiv [n] \backslash \mathcal{I}_k$, denoted by $\widehat{\eta}_{\Delta,\Lambda}^{-k}$.
5: end for
6: Estimate $\Delta_\Lambda(\pi)$ as
$\widehat{\Delta}_{\Lambda,n}(\pi) = \frac{1}{n} \sum_{k=1}^{K} \sum_{i \in \mathcal{I}_k} \psi_\Delta \left(Z_i, \pi; \widehat{\eta}_{\Delta,\Lambda}^{-k}\right)$. (4.2)
7: Return $\widehat{\pi}_{\Delta,\Lambda} = \argmax_{\pi \in \Pi_n} \widehat{\Delta}_{\Lambda,n}(\pi)$.
</div>

## 4.2 Assumptions about the Policy Class

In policy design, it is crucial to account for multiple constraints, including budget, simplicity, interpretability, and functional form. Statistically speaking, to achieve regret bounds that decay at the rate of $n ^ { \stackrel { - } { - } 1 / 2 }$ , one must control the complexity of the policy class. In the remainder of the paper, we allow the policy class $\Pi \equiv \Pi _ { n }$ to vary with n. Consequently, we restrict $\Pi _ { n }$ to be VC-subgraph class; see van der Vaart and Wellner (1996); Giné and Nickl (2021) for further details.

Assumption 4.1. The policy class $\Pi _ { n }$ is $\mathrm { V C ^ { - } }$ subgraph with VC dimension satisfying $\mathrm { V C } ( \Pi _ { n } ) \leq$ $n ^ { \zeta _ { \Pi } }$ , where $\begin{array} { r } { 0 < \zeta _ { \Pi } < \frac { 1 } { 2 } } \end{array}$

Remark 4.1. In the statistical learning literature, ${ \mathrm { V C } } ( \Pi _ { n } )$ is often referred to as the pseudo-dimension, which generalizes the classical Vapnik-Chervonenkis (VC) dimension from binary-valued function classes to real-valued ones; see Anthony and Bartlett (2009); Bartlett et al. (2019). For binary-valued function classes, the VC dimension and pseudo-dimension are identical. In this work, we do not distinguish between the classical VC dimension and the pseudo-dimension, and refer to both simply as the VC dimension.

Various ML models can be used as policy classes. Below, we present several examples of such models along with their corresponding VC dimensions, covering both deterministic and randomized policy classes.

Example 3 (Linear Policies). The class of (deterministic) linear policies is defined as

$$
\Pi_ {n} = \left\{\mathbb {1} \{T (x) ^ {\prime} \beta > 0 \}: \beta \in \mathbb {R} ^ {d _ {n}} \right\},
$$

where $T ( x ) \in \mathbb { R } ^ { d _ { n } }$ denotes a vector of transformed covariates constructed from the raw features X using basis expansions such as polynomial terms, B-splines, or interaction terms. The VC dimension of linear policy class $\Pi _ { n }$ is given by $d _ { n } + 1$

Example 4 (Decision Trees). A decision tree is a classifier $\pi : { \mathcal { X } } \to \{ 0 , 1 \}$ that recursively partitions the feature space $\mathcal { X }$ into disjoint rectangular regions, assigning a label to each partition. The class of depth-L decision trees in $\mathbb { R } ^ { d }$ has VC dimension bounded on the order of $O \left( 2 ^ { L } \log d \right)$

Next, we provide several examples of randomized policy classes.

Example 5 (Neural Networks). Deep neural networks have achieved remarkable success in solving complex classification and regression problems. Formally, a neural network defines a class of functions mapping from $\mathcal { X }$ to R. Such networks can be employed to represent both deterministic and randomized policy classes. Formally, neural networks model the relationship between inputs and outputs through layers of interconnected computational units (neurons), inspired by biological neural systems. The class of networks with L layers, $p$ parameters and a piecewise linear activation function has a VC dimension bounded on the order of $O \left( L p \log p \right)$ ; see Bartlett et al. (2019).

Example 6 (Logistic Policies). The class of logistic policies is defined as

$$
\Pi_ {n} = \left\{\sigma \left(T (x) ^ {\prime} \beta\right): \beta \in \mathbb {R} ^ {d _ {n}} \right\},
$$

where $\sigma$ is standard logistic function and $T ( x )$ denotes a transformation of covariates x as in Example 3. It is noted that $\sigma$ is strictly increasing, and $\Pi _ { n } = \sigma \circ \{ T ( x ) ^ { \prime } \beta : \beta \in \mathbb { R } ^ { d _ { n } } \}$ . As a result, Theorem 2.6.18 in van der Vaart and Wellner (1996) implies $\mathrm { V C } ( \Pi _ { n } ) = d _ { n } + 1$

## 4.3 Nuisance Estimators and Uniform Coupling

In this section, we present a key lemma demonstrating that the estimation error of the nuisance parameters becomes asymptotically negligible when the objective functions $W _ { \Lambda } ( \cdot )$ and $\Delta _ { \Lambda } ( \cdot )$ are estimated using the doubly robust score with cross-fitting.

Assumption 4.2. Suppose that Y and $\boldsymbol { q } _ { \Lambda } ^ { \pm } ( \boldsymbol { X } , \boldsymbol { a } )$ , for $a \in \{ 0 , 1 \}$ have finite second moments; that is, $\mathbb { E } | Y | ^ { 2 } < \infty$ and $\vec { \Xi } \left| q _ { \Lambda } ^ { \pm } ( X , a ) \right| ^ { 2 } < \infty$

Assumption 4.3. There exist $\kappa \in ( 0 , 1 / 2 )$ such that the nominal propensity score satisfies $e ( x ) \in$ $\left( \kappa , 1 - \kappa \right)$ for all $x \in \mathcal { X }$

Assumption 4.2 is straightforward to verify. Assumption 4.3 is the standard strict overlap condition in the literature, which is essential for establishing the asymptotic regret upper bounds. Moreover, we adopt an agnostic stance on how the nuisance components, $\widehat { e } , \widehat { q } _ { \Lambda }$ and $\widehat { \rho } _ { t , \Lambda }$ , are estimated. b b bRather than specifying particular estimation procedures, we impose high-level assumptions on their convergence ratio, as stated below.

Assumption 4.4. Suppose we have uniformly consistent estimators of the nuisance parameters such that

$$
\sup _ {x} \left| \widehat {e} (x) - e (x) \right|, \quad \sup _ {x, a} \left| \widehat {q} _ {\Lambda} ^ {\pm} (x, a) - q _ {\Lambda} ^ {\pm} (x, a) \right|, \quad \sup _ {x, a, t} \left| \widehat {\rho} _ {t, \Lambda} ^ {\pm} (x, a) - \rho_ {t, \Lambda} ^ {\pm} (x, a) \right| \stackrel {{P}} {{\to}} 0.\tag{4.3}
$$

Furthermore, there exist constants $\zeta _ { e } , \zeta _ { q } , \zeta _ { \rho } \geq 1 / 2$ and a sequence $b ( n ) = o ( 1 )$ such that

$$
\begin{array}{r l} & {\mathbb {E} \left[ | \widehat {e} (X _ {i}) - e (X _ {i}) | ^ {2} \right] \leq \frac {b (n)}{n ^ {\zeta_ {e}}}, \qquad \mathbb {E} \left[ | \widehat {q} _ {\Lambda} ^ {\pm} (X _ {i}, A _ {i}) - q _ {\Lambda} ^ {\pm} (X _ {i}, A _ {i}) | ^ {2} \right] \leq \frac {b (n)}{n ^ {\zeta_ {q}}},} \\ & {\mathbb {E} \left[ \left| \widehat {\rho} _ {t, \Lambda} ^ {\pm} (X _ {i}, A _ {i}) - \rho_ {t, \Lambda} ^ {\pm} (X _ {i}, A _ {i}) \right| ^ {2} \right] \leq \frac {b (n)}{n ^ {\zeta_ {\rho}}}.} \end{array}\tag{4.4}
$$

Remark 4.2. The regression and quantile functions, $e ( x )$ and $\scriptstyle q ^ { \pm } ( x , a )$ can be estimated using kernel and sieve methods; see (Li and Racine, 2007; Chen, 2007; Belloni et al., 2019). Recent advances have widely adopted ML methods for estimating nuisance parameters, with desirable asymptotic properties well established in the literature; see, e.g., Lasso-based approaches (Belloni and Chernozhukov, 2011; Belloni et al., 2014, 2017) and deep neural network estimators (Schmidt-Hieber, 2020; Farrell et al., 2021; Kohler and Langer, 2021; Padilla et al., 2022). For estimating $\rho _ { t , \Lambda } ^ { \pm }$ , which corresponds to the conditional value at risk (CVaR) or expected shortfall, a variety of methods have been developed, including kernel, local linear, and deep neural network approaches; see (Fissler et al., 2023; Olma, 2021; Yu et al., 2025).

Remark 4.3. Assumption 4.4 is similar to Assumption 2 in Athey and Wager (2021). First, the condition (4.4) closely follows standard semiparametric assumptions, which typically require an $n ^ { - 1 / 4 }$ -convergence rate in $L ^ { 2 } { \mathrm { - n o r m } }$ for nuisance parameters (Belloni et al., 2017; Chernozhukov et al., $2 0 1 8 , 2 0 2 2 ) . ^ { 4 }$ Second, the condition (4.3) is essential in policy learning with observational data to withstand the policy optimization over large class $\Pi _ { n }$ . Specifically, this ensures that the welfare estimator with plug-in nuisance parameter estimates uniformly approximates its oracle counterpart, as established in Lemma 4.1.

The use of doubly robust scores combined with cross-fitting ensures that errors from nuisance parameter estimation become asymptotically negligible for policy learning, provided ${ \mathrm { V C } } ( \Pi _ { n } )$ does not grow too quickly with n. To formalize this, suppose the true nuisance functions $\eta _ { W , \Lambda }$ and $\eta _ { \Delta , \Lambda }$ defined in Section 3.2 are known, the oracle estimators for $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ are given by

$$
\begin{array}{l} W _ {\Lambda , n} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \psi_ {W} (Z _ {i}, \pi ; \eta_ {W, \Lambda}), \\ \Delta_ {\Lambda , n} (\pi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \psi_ {\Delta} (Z _ {i}, \pi ; \eta_ {\Delta , \Lambda}). \end{array}
$$

We conclude this subsection by demonstrating that $\widehat { W } _ { \Lambda , n } ( \pi )$ is a valid approximation to $W _ { \Lambda , n } ( \pi )$ and similarly, $\widehat { \Delta } _ { \Lambda , n } ( \pi )$ approximates $\Delta _ { \Lambda , n } ( \pi )$ b, both with a convergence rate faster than $n ^ { - 1 / 2 }$

Lemma 4.1. Under Assumptions 3.1, 4.1, 4.2, 4.3 and 4.4, we have

$$
\begin{array}{r} \mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| \widehat {W} _ {\Lambda , n} (\pi) - W _ {\Lambda , n} (\pi) \right| \right] = o (n ^ {- 1 / 2}), \\ \mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| \widehat {\Delta} _ {\Lambda , n} (\pi) - \Delta_ {\Lambda , n} (\pi) \right| \right] = o (n ^ {- 1 / 2}). \end{array}
$$

## 4.4 Asymptotic Regret Bounds

In this subsection, we will demonstrate that, under appropriate bounds on potential hidden confounding, the CRW-regret and CRI-regret of the learned MMW and MMI policies decay at a rate that is upper bounded by $\sqrt { \mathrm { V C } ( \Pi _ { n } ) / n }$ . Recall that the CRW-regret and CRI-regret of the learned optimal policies are defined as

$$
\begin{array}{r} \mathrm{Reg} _ {W} (\widehat {\pi} _ {W, n}) = \sup _ {\pi \in \Pi_ {n}} W _ {\Lambda} (\pi) - W _ {\Lambda} (\widehat {\pi} _ {W, n}), \\ \mathrm{Reg} _ {\Delta} (\widehat {\pi} _ {\Delta , n}) = \sup _ {\pi \in \Pi_ {n}} \Delta_ {\Lambda} (\pi) - \Delta_ {\Lambda} (\widehat {\pi} _ {\Delta , n}), \end{array}
$$

where ${ \widehat { \pi } } _ { W , n }$ and $\widehat { \pi } _ { \Delta , n }$ are learned from Algorithm 1 and Algorithm 2.

b bTheorem 4.1. Under Assumptions 2.1, 3.1, 4.1, 4.2, 4.3 and 4.4, then

$$
\mathbb {E} \left[ \operatorname{Reg} _ {W} \left(\widehat {\pi} _ {W, n}\right) \right] = O \left(\sqrt {\operatorname{VC} \left(\Pi_ {n}\right) / n}\right),\tag{4.5}
$$

$$
\mathbb {E} \left[ \mathrm{Reg} _ {\Delta} \left(\widehat {\pi} _ {\Delta , n}\right) \right] = O \left(\sqrt {\mathrm{VC} (\Pi_ {n}) / n}\right).\tag{4.6}
$$

Theorem 4.1 improves upon the result in Kallus and Zhou (2021), where Proposition 6 shows that estimating the nominal propensity score leads to a regret bound that exceeds the oracle regret bound $( \mathrm { i . e . }$ ., the regret bound assuming a known nominal propensity score) by an additional term of the order

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} | 1 / \widehat {e} (X _ {i}) - 1 / e (X _ {i}) |.
$$

With the application of the doubly robust score and cross-fitting, the regrets of the learned MMW and MMI policies decay at the rate $\sqrt { \mathrm { V C } ( \Pi _ { n } ) / n }$ , and the estimation errors of the nuisance parameters have no asymptotic effect on the regret bound.

Remark 4.4. Let us analyze Eq. (4.5) in Theorem 4.1 in detail; the analysis of Eq. (4.6) proceeds in the same fashion. To be more precise, there is a universal constant $K > 0$ not depending on Λ such that

$$
\limsup _ {n \to \infty} \frac {\mathrm{Reg} _ {W} \left(\widehat {\pi} _ {W , n}\right)}{\sqrt {\mathrm{VC} (\Pi_ {n}) / n}} \leq K \sigma_ {W} (\Lambda)\tag{4.7}
$$

where $\sigma _ { W } ( \Lambda ) ^ { 2 } = { \mathbb E } \big | \big ( \phi _ { 0 } ^ { - } - \phi _ { 1 } ^ { - } \big ) \big ( Z , \eta _ { W , \Lambda } \big ) \big | ^ { 2 }$ , and ${ \phi } _ { t } ^ { - }$ for $t \in \{ 0 , 1 \}$ are defined in Eq. (3.2).


The asymptotic upper bound given in Eq. (4.7) depends on the universal constant K and a variance complexity term $\sigma _ { W } ( \Lambda )$ that captures the second moments of $\phi _ { t } ^ { - } ( Z , \eta _ { W , \Lambda } )$ for $t \in \{ 0 , 1 \}$ . As shown in the proof of Theorem 4.1, the constant $K$ is universal in the sense that it does not depend on sample size $n ,$ the sensitivity parameter $\Lambda _ { i }$ , the nuisance function $\eta _ { W , \Lambda }$ , or even the underlying distribution of $P _ { o } \in \mathcal { P } ( \Lambda )$ ). The variance complexity term $\sigma _ { W } ( \Lambda )$ reflects the level of uncertainty induced by the unobserved confounding. When $\Lambda = 1$ , then $\sigma _ { W } ^ { 2 } ( 1 )$ coincides with the semiparametric efficient variance for the average treatment effect under the selection-on-observables assumption;

see Newey (1994); Robins et al. (1994)

## 5 Simulation Studies

In this section, we conduct simulation studies to evaluate the performance of the MMW and MMI policies, which are learned using Algorithm 1 and 2, respectively. To facilitate comparison with the results in Kallus and Zhou (2021), we use the logistic policies given in Example 6.

## 5.1 Simulation Design

The data-generating process (DGP) is specified as follows. Let log ${ \Lambda } ^ { * } = 1 . 5 .$ which implies that the true sensitivity parameter is $\Lambda ^ { * } = 4 . 4 8 2$ . The observed covariates $X \in \mathbb { R } ^ { 2 }$ , treatment assignment $A \in \{ 0 , 1 \}$ , and outcome $Y \in \mathbb { R }$ are generated according to:

$$
\begin{array}{l} X \sim \mathcal {N} (\mu_ {X}, I _ {2}), \quad U \mid X \sim \text {Bern} \left(\frac {\Lambda^ {*}}{1 + \Lambda^ {*}} e (X) + \frac {1}{1 + \Lambda^ {*}} (1 - e (X))\right), \\ A \mid U, X \sim \text {Bern} \left(\frac {e (X)}{e (X) + \Lambda^ {* (1 - 2 U)} (1 - e (X))}\right), \end{array}
$$

where $\mu _ { X } = [ - 1 , 1 ] ^ { \prime }$ and $I _ { 2 } \in \mathbb { R } ^ { 2 \times 2 }$ is the identity matrix. The nominal propensity score $e ( X )$ is defined as:

$$
e (X) = \mathbb {P} (A = 1 \mid X) = \sigma (\zeta (X) ^ {\prime} \theta), \quad \mathrm{with} \quad \sigma (z) = \frac {1}{1 + e ^ {- z}},
$$

where the parameter vector θ and the nonlinear feature map ζ(X) are given by

$$
\theta = [ 0. 2, 0. 4, 0. 1, - 0. 1, 0. 5, - 0. 5 ], \quad \zeta (X) = \left[ \max (x _ {1}, 0), \frac {x _ {1} x _ {2} ^ {2}}{1 0}, \sin (x _ {2} ^ {2}), x _ {1}, x _ {2}, 1 \right] ^ {\prime}.
$$

It can be readily verified that U perturbs the nominal odds ratio $e ( X ) / ( 1 - e ( X ) )$ by a multiplicative factor of $\Lambda ^ { * \pm 1 }$ , ensuring the DGP satisfies the MSM with $\Lambda ^ { * } = 4 . 4 8 2$ . Finally, the potential outcome is generated as:

$$
Y (A) = \beta_ {\mathrm{cons}} + \beta_ {A} A + X ^ {\prime} \beta_ {X} + (A \cdot X) ^ {\prime} \beta_ {X, A} + \beta_ {U} U + \epsilon , \quad \epsilon \sim \mathcal {N} (0, 1),
$$

where:

$$
\beta_ {\mathrm{cons}} = - 0. 2, \quad \beta_ {A} = - 0. 1, \quad \beta_ {X} = [ 1, - 1 ] ^ {\prime}, \quad \beta_ {X, A} = [ 0. 2, 0. 4 ] ^ {\prime}, \quad \beta_ {U} = 1. 5.
$$

We generate i.i.d. samples $\{ X _ { i } , A _ { i } , U _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n }$ according to the DGP described above, with $n = 2 , 0 0 0$ . The observed data $\{ X _ { i } , A _ { i } , Y _ { i } \} _ { i = 1 } ^ { n }$ are used to learn the policies. The conditional quantile functions $q _ { \Lambda } ^ { \pm }$ are estimated using gradient boosted trees and the nominal propensity score e and CVaR $\rho _ { t , \Lambda } ^ { \pm }$ are estimated using random forests.

We evaluate the performance of estimated policy π using three metrics: (i) the expected welfare, defined in Eq. (2.2); (ii) the worst-case welfare $W _ { \Lambda } ( \widehat { \pi } )$ , as given in Theorem 3.1; and (iii) the worst-case policy improvement $\Delta _ { \Lambda } ( \widehat { \pi } )$ bgiven in Theorem 3.2. All three metrics are estimated using bsample averages computed on an independent out-of-sample dataset of 100,000 randomly drawn observations. To assess the stability of our method, we evaluate each metric over 100 repeated experiments. We report the mean from these repetitions and visualize the variability using shaded bands that represent the 95% confidence bands, calculated from the standard deviation across the

100 experiments.

## 5.2 Simulation Analysis

We report the performance of the MMW and MMI policies, as well as the robust IPW-based policy of Kallus and Zhou (2021) (KZ for short). As a benchmark, we include the policy proposed by Athey and Wager (2021) (AW), which assumes no unobserved confounding $( \Lambda = 1 )$ To evaluate the MMW, MMI, and KZ policies, we conduct a sensitivity analysis by varying the sensitivity parameter over log $\Lambda \in \{ 0 . 1 , 0 . 2 , . . . , 3 . 5 \}$ . This assesses their robustness to miscalibration relative to the true level of unobserved confounding in the true DGP with log $\Lambda ^ { * } = 1 . 5$ . By design, the MMW and MMI policies reduce to the AW policy at the $\Lambda = 1$ setting. In contrast, the KZ policy remains different from AW even when $\Lambda = 1$ , as its objective function does not include a bias-correction term for the estimated propensity scores.

Figure 2a shows how each policy adjusts its treatment probability in response to the assumed level of unobserved confounding, and helps explain their relative performance in terms of expected welfare, presented in Figure 2b. The AW policy, assuming no unobserved confounding( $\Lambda = 1 )$ yields an aggressive strategy, assigning treatment to 95% of individuals. In contrast, MMI, MMW and KZ are sensitivity-aware, adopting a conservative approach that is responsive to Λ. As log Λ increases, they systematically reduce treatment rates to hedge against selection bias. However, their conservative dynamics differ significantly. The KZ policy is overly conservative, exhibiting a precipitous drop in assignments due to its reliance on the loose lower bound on policy improvement from Zhao et al. (2019). In contrast, the MMI policy is precisely conservative; by identifying the sharp lower bound of the CATE, it establishes a sharper bound on policy improvement, leading to a more calibrated and gradual reduction in treatment. Finally, the MMW policy’s aggressiveness under high uncertainty is a direct consequence of its decision rule, illustrated in Section 3.2, which favors treatment whenever $\mu _ { \Lambda } ^ { - } ( x , 1 )$ exceeds $\mu _ { \Lambda } ^ { - } ( x , 0 )$ .

Figure 2b shows the resulting expected welfare. The MMI policy achieves its peak expected welfare at log $\Lambda = 1$ . It outperforms the MMW policy for assumed confounding levels below the true value of log $\Lambda ^ { * } = 1 . 5$ . In regimes of greater assumed uncertainty log $\Lambda > 1 . 5$ , however, the MMW policy’s performance is superior, yielding significantly higher welfare. Their performance crossover occurs precisely at this true value, confirming their complementary nature. This divergence occurs because, in high-uncertainty regimes, the MMI policy becomes overly cautious, whereas the MMW policy adopts a more risk-tolerant strategy.

Finally, Figures 3a and 3b validate that the observed behaviors are consequences of each policy’s specific design. As expected, Figure 3a shows the MMW policy achieving the highest worst-case welfare, and Figure 3b confirms the MMI policy’s success in maximizing the worst-case policy improvement. This confirms that each policy effectively optimizes its intended objective, leading to the distinct and complementary performance profiles we identified.

(a) Average Treatment Probability

(b) Expected Welfare

Figure 2: Average treatment probability and expected welfare as a function of the sensitivity parameter log(Λ). The MMI and MMW policies are compared against the KZ and AW benchmarks. Shaded areas are 95% confidence bands.
(a) Worst-case Welfare

(b) Worst-case Policy Improvement
Figure 3: Performance on Intended Objectives. The figures plot (a) the MMW policy’s objective of worst-case welfare and (b) the MMI policy’s objective of worst-case policy improvement against log(Λ). Shaded areas denote 95% confidence bands.

## 6 Empirical Application

In this section, we apply our method to two empirical applications. Section 6.1 revisits the JTPA study and examines assignment to job training programs, while Section 6.2 explores enrollment decisions in the Head Start program. In both settings, unobserved confounders likely give rise to endogenous selection.

## 6.1 The JTPA Study

We apply the MMW and MMI policy learning methods to the National Job Training Partnership Act (JTPA) Study, a large-scale randomized controlled trial evaluating training services for disadvantaged adults (Bloom et al., 1997). In the experiment, eligibility for services was randomly assigned, and participants’ earnings were tracked over the 30-month period after the assignment. This dataset was used by Kitagawa and Tetenov (2018) to estimate an intent-to-treat optimal policy for 9,223 applicants using their Empirical Welfare Maximization (EWM) approach, based on covariates such as education and prior earnings.

While eligibility was randomly assigned, compliance in the JTPA study was imperfect: approximately 23% of individuals did not adhere to their assigned eligibility status, as shown in Table 1.

This imperfect compliance introduces self-selection into actual participation decisions, raising concerns about unobserved confounding. For instance, eligible individuals who opt into the program may exhibit stronger job search motivation or face fewer personal constraints, such as childcare responsibilities, health issues, or transportation barriers, than those who decline participation. These unobserved factors likely influence both program participation and post-program earnings, violating the unconfoundedness assumption required by conventional policy learning methods.

Table 1: Joint distribution of eligibility and participation, JTPA study

<table><tr><td rowspan="2">Participation ( $A_i$ )</td><td colspan="3">Eligibility (IV)</td></tr><tr><td>0</td><td>1</td><td>Total</td></tr><tr><td>0</td><td>3047</td><td>2118</td><td>5165</td></tr><tr><td>1</td><td>43</td><td>4015</td><td>4058</td></tr><tr><td>Total</td><td>3090</td><td>6133</td><td>9223</td></tr></table>

Data source: Kitagawa and Tetenov (2018) and Abadie et al. (2002).

We focus on actual program participation, as the policy-relevant objective is to target individuals who are most likely to benefit from job training and to allocate participation accordingly. Building on the perspective of d’Adamo (2021), who frame policy learning as deriving optimal treatment rules under potential unobserved confounding, we apply the MMW and MMI methods, both of which are explicitly designed to address such confounding. Unlike d’Adamo (2021), who use a binary instrumental variable (eligibility) to construct partial identification intervals for the CATE, we adopt a complementary strategy by employing with a pre-specified sensitivity parameter Λ.

Calibration of Sensitivity Parameter. We calibrate the Λ following the breakdown frontier perspective of Masten and Poirier (2020). The IV bounds discussed in d’Adamo (2021) provide a credible benchmark for this purpose. Let $\tau _ { \mathrm { { I V } } } ^ { - } ( x )$ be the lower bound for CATE $\tau ( x )$ , and $\mu _ { \mathrm { I V } } ^ { - } ( x , a )$ be the lower bound for the conditional response function $\mu ( x , a )$ , constructed using $\mathrm { I V } . ^ { 5 }$ We select Λ as the smallest relaxation level at which the MSM lower bounds are at least as conservative as the IV lower bounds. In practice, we select Λ based on the coverage share, that is, the fraction of sample observations for which the MSM-implied lower bound is no larger than the IV-based lower bound. For the MMI policy, the coverage share is computed using $\tau _ { \Lambda } ^ { - } ( x )$ and $\tau _ { \mathrm { { I V } } } ^ { - } ( x )$ . We select the smallest value of Λ for which this coverage share reaches the pre-specified threshold. For the MMW policy, we apply the same rule to $\mu _ { \Lambda } ^ { - } ( x , a )$ and $\mu _ { \mathrm { I V } } ^ { - } ( x , a )$ for $a \in 0 , 1$ . As shown in Figure 4, targeting an 80% coverage share yields $\Lambda = 2 . 5$ for the MMI policy and $\Lambda = 4 . 5$ for the MMW policy.

We estimate the optimal MMW and MMI policies using Algorithms 1 and 2, with $K = 1 0 \mathrm { f o l d s }$ for cross-fitting. The conditional quantile functions $q _ { \Lambda } ^ { \pm }$ are estimated using gradient boosted trees, and the nominal propensity score e and CVaR $\rho _ { t , \Lambda } ^ { \pm }$ are estimated using random forests. The nominal propensity score estimates range from 0.27 to 0.73, so we retain the full sample without trimming. To account for program costs, we subtract \$1,216 from the outcomes of treated individuals. This amount corresponds to the average cost of services per actual participant, as reported in Table 5 of Bloom et al. (1997). Following the approach of Kitagawa and Tetenov (2018) and d’Adamo (2021), we consider the class of quadrant treatment policies due to its simplicity and interpretability:

$$
\Pi \equiv \left\{\mathbb {1} \left\{s _ {1} (\mathrm{edu} - t _ {1}) > 0, s _ {2} (\text { earnings } - t _ {2}) > 0 \right\}: s _ {1}, s _ {2} \in \{- 1, 1 \}, t _ {1}, t _ {2} \in \mathbb {R} \right\}.
$$

Figure 4: Calibration of the sensitivity parameter Λ against the IV-based lower bounds. Panel (a) plots, as functions of $\Lambda ,$ the coverage shares $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } 1 \mathbb { 1 } \left\{ \mu _ { \Lambda } ^ { - } ( x _ { i } , a ) \le \mu _ { \mathrm { I V } } ^ { - } ( x _ { i } , a ) \right\} } \end{array}$ for $a \in \{ 0 , 1 \}$ Panel (b) plots the coverage share $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \left\{ \tau _ { \Lambda } ^ { - } ( x _ { i } ) \stackrel { . } { \le } \bar { \tau } _ { \mathrm { I V } } ^ { - } ( x _ { i } ) \right\} } \end{array}$ . The plotted coverage shares  	are computed using the estimated MSM and IV lower bounds. The horizontal dotted line marks the 80% target threshold, attained at $\Lambda = 4 . 5$ for the MMW policy, determined by the treated-arm comparison, and $\Lambda = 2 . 5$ for the MMI policy.

Given the cross-fitted doubly robust scores, we maximize the estimated robust criterion by searching over all values of $( s _ { 1 } , s _ { 2 } )$ and all threshold pairs $\left( t _ { 1 } , t _ { 2 } \right)$ induced by the observed values of education and pre-program earnings, following the implementation in Kitagawa and Tetenov (2018).

Figure 5 presents the estimated upper and lower bounds of the CATE under the MSM framework, evaluated across different values of the sensitivity parameter $\Lambda \in \{ 1 , 1 . 5 , 2 \}$ . When $\Lambda = 1$ , the estimated upper and lower bounds coincide. Choosing a larger value of Λ can yield more robust policy targeting by ensuring that the identified set contains the true causal effect. However, this robustness comes at the cost of producing less informative estimates and more conservative policy intervention. Therefore, we recommend that practitioners examine results across a range of Λ values to assess the sensitivity of their conclusions to unobserved confounding.

Figure 6 illustrates the estimated optimal quadrant policies for MMW and MMI , using the IV minimax-regret (MMR) policy of d’Adamo (2021) as a benchmark. When $\Lambda = 1$ , both the MMW and MMI policies are exactly the AW policy, but with a treated fraction approaching one. As sensitivity parameter $\Lambda$ increases, the MMI policy becomes increasingly conservative, with the treatment rate dropping rapidly to near zero when $\Lambda \geq 1 . 5$ . The extremely low treatment rates under the MMI policy arise because the estimated lower bounds of the CATE fall below zero for nearly all individuals when $\Lambda \geq 1 . 5$ . Consequently, our subsequent analysis focuses on the MMW policy. When $\Lambda = 1$ , the quadrant policy targets individuals with more than 7.5 years of education and pre-program earnings below \$40,800. In contrast, at $\Lambda = 2 .$ , the MMW quadrant policy assigns treatment to those with at least 10.5 years of education and earnings below \$35,144. As the sensitivity parameter Λ increases, the estimated MMW policy becomes more selective, prioritizing individuals with higher educational level and lower earnings. <sup>6</sup> In sharp contrast, the IV MMR policy assigns treatment to roughly 96% of the sample. Such an excessively high treatment rate fails to provide meaningful targeting guidance, offering little practical relevance for selective policy design.


Figure 5: Upper and lower bounds of the CATE under uncertainty levels Λ = 1, 1.5, 2.
Figure 6: Optimal MMW and MMI quadrant policies under different uncertainty levels: Λ = 1, 1.5, 2 , compared with the Λ-invariant IV MMR policy.

## 6.2 The Head Start Program

We further demonstrate our confounding-robust policy learning method by applying it to enrollment decisions for the Head Start program. The objective is to develop a policy that improves academic outcomes by targeting admission to children most likely to benefit. This application presents a multi-valued treatment setting with three alternatives, Head Start, other preschools, or no preschool, necessitating the use of our MMW approach extended for multiple treatments as detailed in Section E.

Head Start is a federal matching grant initiative designed to foster the cognitive, social, and health development of children from low-income families, with the aim of preparing them to enter school on a more equal footing with their more advantaged peers. A substantial body of research has evaluated the impact of Head Start participation on outcomes such as academic achievement, physical health, and long-term socioeconomic status; see, e.g., (Currie and Thomas, 1995; Deming, 2009; Ludwig and Miller, 2007; Walters, 2015).

A key challenge in evaluating Head Start’s impact is that enrollment is not randomly assigned. First, families self-select into Head Start or other preschool programs based on eligibility and perceived benefits. Those who value early education may also make unobserved investments in their children, such as fostering enriched home environments or providing supplemental learning. Second, limited program capacity in many areas gives administrators discretion over admissions, leading to selection based on both observed and unobserved characteristics, some of which also affect child outcomes. As a result, failing to account for such confounding can lead to targeting policies that misidentify the children most likely to benefit.

We draw on data from the National Longitudinal Survey of Youth (NLSY) and its child supplement, the National Longitudinal Survey of Youth 1979 Children and Young Adults (NLSCYA). The NLSY began in 1979 with a nationally representative sample of young women, and the NLSCYA has tracked their children since 1986. Our analysis sample is constructed by linking child-level outcomes from the NLSCYA with maternal background characteristics from the original NLSY cohort.

We use the child’s standardized score on the Peabody Picture Vocabulary Test (PPVT), a widely used measure of early verbal ability, as a proxy for our policy objective of improving academic achievement. The pre-treatment covariates used in our analysis capture a range of child, maternal, and household characteristics. These include: (1) the percentile rank of average net family income between ages 0 and 4 (Income Pctl); (2) child’s gender (Gender); (3) birth weight (Birth Wt); (4) weight at preschool entry (Preschool Entry Wt); (5) firstborn status (Firstborn); (6) mother’s highest completed grade (Mother’s Grade); (7) mother’s AFQT percentile score (Mother’s AFQT); (8) number of the mother’s biological siblings (Bio. Siblings); (9-10) number of household members with less than 12 years (HH Edu <12) and at least 16 years of education (HH Edu ≥16); and (11) race (White, Hispanic, Black). After excluding observations with missing data, the final sample consists of 3,826 children: 755 attended Head Start, 2,020 enrolled in other preschool programs, and 1,051 did not attend any preschool.

Calibration of Sensitivity Parameter. Since valid instruments are unavailable in the Head Start study, we cannot calibrate the sensitivity parameter Λ using the IV-based procedure in Section 6.1. Following Hsu and Small (2013) and Kallus and Zhou (2021), we instead calibrate Λ using observed covariates by computing the effect of omitting each observed covariate on the odds ratio of the propensity score. Specifically, for each treatment arm $a \in \{ 0 , 1 , 2 \}$ , we estimate two propensity score models using random forests: a full model $\widehat { e _ { a } } ( x )$ that includes all covariates, and a leaveone-out model $\widehat { e _ { a } } ^ { - j } ( { x } _ { - j } )$ bthat omits the j-th covariate. The individual-level calibrated sensitivity bparameter associated with omitting the j-th covariate in arm a is defined as

$$
\Lambda_ {j, a} (x) = \max \left(\frac {\widehat {e} _ {a} (x) / (1 - \widehat {e} _ {a} (x))}{\widehat {e} _ {a} ^ {- j} (x _ {- j}) / (1 - \widehat {e} _ {a} ^ {- j} (x _ {- j}))}, \frac {\widehat {e} _ {a} ^ {- j} (x _ {- j}) / (1 - \widehat {e} _ {a} ^ {- j} (x _ {- j}))}{\widehat {e} _ {a} (x) / (1 - \widehat {e} _ {a} (x))}\right).
$$

We take the 95th percentile of $\Lambda _ { j , a } ( X _ { i } )$ for $i \in [ n ]$ as the calibrated value associated with omitting covariate j for arm a, denoted as $\Lambda _ { j , a }$ . We then compute the covariate-specific calibrated sensitivity parameter as $\Lambda _ { j } = { \tt m a x } _ { a } \Lambda _ { j , a }$ , that is reported in Table 2. Finally, taking the maximum across covariates gives the overall calibrated sensitivity parameter $\Lambda = 3 . 6$ . The largest covariate-specific value is associated with the family income percentile.

Table 2: Calibrated sensitivity parameters for each covariate (no propensity truncation).

<table><tr><td></td><td>Firstborn</td><td>Birth Wt</td><td>Preschool Entry Wt</td><td>Gender</td><td>Mother&#x27;s AFQT</td><td>Mother&#x27;s Grade</td><td>Bio. Siblings</td><td>HH Edu ≥16</td><td>HH Edu &lt;12</td><td>Income Pctl</td><td>Hispanic</td><td>Black</td></tr><tr><td> $\Lambda_{j}$ </td><td>1.746</td><td>1.802</td><td>1.833</td><td>1.485</td><td>2.550</td><td>1.655</td><td>1.718</td><td>1.683</td><td>1.576</td><td>3.596</td><td>1.691</td><td>2.049</td></tr></table>

We estimate the nuisance functions following the same estimation procedures used in Sections 5.2 and 6.1. The initial propensity score estimates range from 0.0005 to 0.95, so we truncate them to [α, 1 − α] with α = 0.05 to enforce the strict overlap condition. While the full set of pretreatment covariates is used to estimate doubly robust scores, we restrict the variables used for policy optimization to a small, interpretable subset: percentile rank of family income, mother’s AFQT score, birth weight, and weight at preschool entry. These are selected based on fairness and ethical considerations. To ensure interpretability, we restrict the policy class to depth-2 decision trees, using R-package policytree.

The policy tree estimated under Λ = 1 serves as our benchmark and reproduces the recommendation that would be obtained by methods such as (Athey and Wager, 2021; Zhou et al., 2023), which assume no unobserved confounding. An analysis of this benchmark policy reveals several features that present challenges for direct implementation. First, it completely excludes Head Start, contradicting the program’s mandate to serve disadvantaged households. Second, it misallocates these economically vulnerable households to other tuition-charging preschools rather than the subsidized public program. This motivates our subsequent sensitivity analysis to assess how the policy recommendations change when accounting for unobserved confounding. Once we account for unobserved confounding, the policy recommendation changes markedly. Specifically, at the calibrated value Λ = 3.6, the robust policy targets Head Start toward a clearly defined disadvantaged group, while taking a more conservative non-intervention stance for the majority.<sup>7</sup>

Figure 7: This figure presents optimal policy trees under unconfoundedness and calibrated value Λ = 3.6. While the unconfoundedness benchmark completely excludes Head Start, incorporating unobserved confounding enables the optimal policy to effectively target this program toward the disadvantaged subgroup.

## 7 Concluding Remarks

Formulating effective policies from observational data is challenged by unobserved confounding, which can lead to biased and potentially harmful policy intervention. This paper proposes a robust and efficient framework for policy learning that explicitly accounts for such endogeneity. Our contributions are twofold: first, we derive sharp, closed-form identification results for robust welfare criteria under the Marginal Sensitivity Model; second, we develop doubly robust scores for their estimation. This innovation enables the use of flexible ML methods for nuisance estimation within our framework. Building on these components, we establish regret bounds for the resulting confounding-robust policies.

Our work also offers insights into the practical challenges of applying policy learning methods. While powerful, ML-based approaches that optimize a single objective can yield policies that are difficult to trust and interpret, particularly when built on fragile assumptions. Our work underscores that sensitivity analysis is a critical step for examining a policy’s foundational reliability.

Finally, several directions remain open for future research. First, while our framework primarily focuses on binary treatments, extending it to multivalued or continuous treatments is a promising direction that would broaden its applicability and introduce new methodological challenges. Second, the choice of the sensitivity parameter Λ remains an important yet unresolved issue. Although our empirical analysis illustrates two calibration strategies, namely observed-covariate calibration following Hsu and Small (2013) and IV-based calibration, the choice of Λ in a given application may still require substantive domain knowledge. Developing more systematic, data-driven methods for selecting Λ remains a key avenue for future investigation.

## Appendix A Policy Learning with Self-Selection as an Option

Most policy-learning work studies compulsory treatment rules, using observable characteristics to mandate program participation or exclusion (Kitagawa and Tetenov, 2018; Athey and Wager, 2021). However, individuals often possess private information, unobserved by the social planner, regarding their potential benefits from the program. So, self-selection serves as a valuable source of information for policy targeting (Manski, 2013; Alatas et al., 2016; Ito et al., 2023). Building on these insights, Ida et al. (2026) integrate targeting by observables with targeting through self-selection. Specifically, they use (quasi)-experimental data to assign individuals to one of three arms: compulsory treatment, compulsory un-treatment, or self-selection, fulfilling the policymaker’s objective.

Section 3 studies robust policy learning under unobserved confounding, focusing on compulsory intervention. This section extends this framework to incorporate self-selection as a policy option. In observational studies, unmeasured confounding makes it harder to assess precisely the welfare gains from counterfactual compulsory assignments. Therefore, absent robust evidence that mandated assignments improve welfare, preserving the status quo self-selection is a natural policy option.

Motivated by this, we formalize the planner’s goal as assigning each individual to one of three arms: compulsorily treated (indexed as 1), compulsorily untreated (indexed as 0), and self-selection (indexed as S). An individual assigned to 1 or 0 is exposed to or excluded from the program with no opt-out or opt-in option, whereas an individual assigned to S retains the autonomy to choose whether to take it up. For clarity, we focus on deterministic policies mapping $\mathcal { X }$ to {0, 1, S} in this section. Let $\Pi _ { o } ^ { \dagger }$ denote the deterministic policy class incorporating the self-selection option

$$
\Pi_ {o} ^ {\dagger} = \left\{\pi : \mathcal {X} \rightarrow \{0, 1, S \} \text {   is   Borel   measurable } \right\}.
$$

Since the observational data reflects the status quo of self-selection, the counterfactual outcome under arm S coincides with the realized outcome $Y = A Y ( 1 ) + ( 1 - A ) Y ( 0 )$ . For any policy $\pi ^ { \dagger } \in \Pi _ { o } ^ { \dagger }$ , the resulting post-treatment outcome $Y ( \pi ^ { \dagger } )$ can be expressed as:

$$
\begin{array}{r l} & Y (\pi^ {\dagger}) = \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} Y (1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} Y (0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} Y \\ & \qquad = \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} + A \mathbb {1} \{\pi^ {\dagger} (X) = S \} \right] Y (1) \\ & \qquad + \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} + (1 - A) \mathbb {1} \{\pi^ {\dagger} (X) = S \} \right] Y (0). \end{array}\tag{A.1}
$$

We now generalize the MMW and MMI criteria from Section 2.2 to an augmented policy class $\Pi _ { o } ^ { \dagger }$ that incorporates self-selection. The worst-case welfare $W _ { \Lambda }$ and policy improvement $\Delta _ { \Lambda }$ functions are defined as:

$$
W _ {\Lambda} (\pi^ {\dagger}) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \big [ Y (\pi^ {\dagger}) \big ] \quad \text {and} \quad \Delta_ {\Lambda} (\pi^ {\dagger}) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \big [ Y (\pi^ {\dagger}) - Y (0) \big ].
$$

Over the policy class $\Pi _ { o } ^ { \dagger }$ , the first best MMW and MMI policies are given, respectively, by

$$
\pi_ {W, \Lambda} ^ {\dagger} \in \underset {\pi^ {\dagger} \in \Pi_ {o} ^ {\dagger}} {\operatorname{argmax}} W _ {\Lambda} (\pi^ {\dagger}),\tag{A.2a}
$$

$$
\pi_ {\Delta , \Lambda} ^ {\dagger} \in \underset {\pi^ {\dagger} \in \Pi_ {o} ^ {\dagger}} {\operatorname{argmax}}   \Delta_ {\Lambda} (\pi^ {\dagger}).\tag{A.2b}
$$

Theorem A.1 characterizes the worst-case welfare when self-selection is incorporated as a feasible policy option, subsequently deriving the first-best MMW policy. To this end, recall the notation $\mu _ { \Lambda } ^ { \pm } ( X , a )$ for $a \in \{ 0 , 1 \}$ as defined in Section 3.1. Moreover, we define the worst-case conditional average treatment effect on the untreated (ATU) and the best-case conditional average treatment effect on the treated (ATT):

$$
\begin{array}{l} \mathrm{ATU} _ {\Lambda} ^ {-} (x) \equiv \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (1) - Y (0) | X = x, A = 0 \right], \\ \mathrm{ATT} _ {\Lambda} ^ {+} (x) \equiv \sup _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (1) - Y (0) | X = x, A = 1 \right]. \end{array}
$$

These quantities are point-identified through the following closed-form relations:

$$
\mathrm{ATU} _ {\Lambda} ^ {-} (x) = \frac {\mu_ {\Lambda} ^ {-} (x , 1) - \mathbb {E} [ Y | X = x ]}{1 - e (x)} \text {and} \mathrm{ATT} _ {\Lambda} ^ {+} (x) = \frac {\mathbb {E} [ Y | X = x ] - \mu_ {\Lambda} ^ {-} (x , 0)}{e (x)}.
$$

Theorem A.1. Under Assumption 2.1, for any policy $\pi ^ { \dagger } \in \Pi _ { o } ^ { \dagger }$ , the worst-case welfare is given by

$$
\begin{array}{r l} & W _ {\Lambda} (\pi^ {\dagger}) = \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \mu_ {\Lambda} ^ {-} (X, 1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \mu_ {\Lambda} ^ {-} (X, 0) \right] \\ & \qquad + \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right]. \end{array}
$$

Moreover, a first-best MMW policy that solves Eq. (A.2a) is given by

$$
\pi_ {W, \Lambda} ^ {\dagger} (x) = \left\{ \begin{array}{l l} 1, & \text {if} \mu_ {\Lambda} ^ {-} (x, 1) > \mu_ {\Lambda} ^ {-} (x, 0) \text {and} \mathrm{ATU} _ {\Lambda} ^ {-} (x) > 0, \\ 0, & \text {if} \mu_ {\Lambda} ^ {-} (x, 0) > \mu_ {\Lambda} ^ {-} (x, 1) \text {and} \mathrm{ATT} _ {\Lambda} ^ {+} (x) <   0, \\ S, & \text {if} \mathrm{ATU} _ {\Lambda} ^ {-} (x) \leq 0 \text {and} \mathrm{ATT} _ {\Lambda} ^ {+} (x) \geq 0. \end{array} \right.\tag{A.3}
$$

Remark A.1. Theorem A.1 establishes a confounding-robust selection-driven targeting rule analogous to Ida et al. (2026). The social planner assigns individuals to {0, 1, S}, based on robust, subpopulation-specific treatment effects. Compared to the selection-absent MMW policy in Theorem 3.1, the selection-driven MMW policy in Theorem A.1 establishes a higher threshold for intervention.

Specifically, compulsory treatment is assigned only when $\mathrm { A T U } _ { \Lambda } ^ { - } ( x ) > 0 .$ , ensuring a guaranteed benefit even for non-takers under the most adversarial confounding.<sup>8</sup> Conversely, compulsory untreatment is assigned only when $\mathrm { A T T } _ { \Lambda } ^ { + } ( x ) < 0$ , implying that treatment may be harmful for takers even under the most favorable confounding scenario. When neither compulsory assignment can be robustly justified, the social planner defaults to self-selection, recognizing that there is no sufficient basis to override individual discretion.

Theorem A.2 characterizes the selection-driven MMI policy. Define the worst-case conditional

average treatment effect on the treated (ATT):

$$
\mathrm{ATT} _ {\Lambda} ^ {-} (x) \equiv \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (1) - Y (0) | X = x, A = 1 \right],
$$

which is point-identified as:

$$
\mathrm{ATT} _ {\Lambda} ^ {-} (x) = \frac {\mathbb {E} [ Y | X = x ] - \mu_ {\Lambda} ^ {+} (x , 0)}{e (x)}.
$$

Theorem A.2. Under Assumption 2.1, for any policy $\pi ^ { \dagger } \in \Pi _ { o } ^ { \dagger }$ , we have that

$$
\begin{array}{r l} & {\Delta_ {\Lambda} (\pi^ {\dagger}) = \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \mu_ {\Lambda} ^ {-} (X, 1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \mu_ {\Lambda} ^ {+} (X, 0) \right]} \\ & {\qquad + \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right] - \mathbb {E} \left[ \mu_ {\Lambda} ^ {+} (X, 0) \right],} \end{array}
$$

Moreover, a first-best MMI policy that solves Eq. (A.2b) is given by

$$
\pi_ {\Delta , \Lambda} ^ {+} (x) = \left\{ \begin{array}{l l} 1, & \text {if} \mu_ {\Lambda} ^ {-} (x, 1) > \mu_ {\Lambda} ^ {+} (x, 0) \text {and} \mathrm{ATU} _ {\Lambda} ^ {-} (x) > 0, \\ 0, & \text {if} \mu_ {\Lambda} ^ {+} (x, 0) > \mu_ {\Lambda} ^ {-} (x, 1) \text {and} \mathrm{ATT} _ {\Lambda} ^ {-} (x) <   0, \\ S, & \text {if} \mathrm{ATU} _ {\Lambda} ^ {-} (x) \leq 0 \text {and} \mathrm{ATT} _ {\Lambda} ^ {-} (x) \geq 0. \end{array} \right.\tag{A.4}
$$

Remark A.2. Following Remark A.1, the selection-driven MMI policy in Eq. (A.4) imposes a stricter criterion for compulsory intervention than the MMW policy in Eq. (A.3). By evaluating the expected outcome under self-selection $\mathbb { E } [ Y | X = x ]$ against the best-case control outcome $\mu _ { \Lambda } ^ { + } ( x , 0 )$ it further expands the ambiguity zone and defaults a larger share of the population to self-selection.

## Appendix B Baseline-Relative MMI and Minimax Regret

This appendix collects two extensions of the robust policy learning framework: a baseline-relative version of the MMI criterion introduced at the end of Section 2.2, and a minimax regret criterion.

## B.1 Baseline-Relative Max-Min Improvement

In this subsection, we investigate the baseline-relative policy improvement criterion introduced in Eq. (2.11). For an arbitrary baseline policy $\pi _ { 0 } \in \Pi$ , we define

$$
\Delta_ {\Lambda} (\pi , \pi_ {0}) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mathbb {E} _ {Q} \left[ Y (\pi (X)) - Y \left(\pi_ {0} (X)\right) \right].
$$

Given $\pi _ { 0 } \in \Pi$ , the optimal policy under this criterion is defined as

$$
\pi_ {\Delta , \Lambda} (\cdot ; \Lambda , \pi_ {0}) \in \underset {\pi \in \Pi} {\operatorname{argmax}} \Delta_ {\Lambda} (\pi , \pi_ {0}).\tag{B.1}
$$

The following corollary provides a sharp characterization of $\Delta _ { \Lambda } ( \pi , \pi _ { 0 } )$ and its associated first-best policy.

Corollary B.1. Under Assumption 2.1, for any policies $\pi , \pi _ { 0 } \in \Pi$ , we have

$$
\Delta_ {\Lambda} (\pi , \pi_ {0}) = \mathbb {E} \left[ \tau_ {\Lambda} ^ {-} (X) \max \left\{\pi (X) - \pi_ {0} (X), 0 \right\} - \tau_ {\Lambda} ^ {+} (X) \max \left\{\pi_ {0} (X) - \pi (X), 0 \right\} \right].
$$

Moreover, a first-best MMI policy relative to the baseline $\pi _ { 0 } .$ , which solves Eq. (B.1) over the unconstrained policy space $\Pi = \Pi _ { o } .$ , is given by

$$
\pi_ {\Delta , \Lambda} ^ {\star} (x; \pi_ {0}) = \left\{ \begin{array}{l l} 1, & \text {if} \tau_ {\Lambda} ^ {-} (x) \geq 0, \\ \pi_ {0} (x), & \text {if} \tau_ {\Lambda} ^ {-} (x) <   0 <   \tau_ {\Lambda} ^ {+} (x), \\ 0, & \text {if} \tau_ {\Lambda} ^ {+} (x) \leq 0. \end{array} \right.
$$

Corollary B.1 shows that, under ambiguity, the first-best baseline-relative MMI policy has a baseline-preserving property: whenever the identified interval for the CATE contains zero, the optimal rule retains the baseline decision $\pi _ { 0 } ( x )$ . Since the standard no-treatment baseline, $\pi _ { 0 } ( x ) \equiv 0$ is nested as a special case, Corollary B.1 directly generalizes Theorem 3.2.

## B.2 Minimax Regret

Following Manski (2004), we consider an alternative criterion that minimizes the worst-case regret over the MSM uncertainty set. For any distribution $Q \in { \mathcal { P } } ( \Lambda )$ , the regret of a policy π is defined as its welfare shortfall relative to the optimal policy within the class Π:

$$
\operatorname{Reg} _ {Q} (\pi) = \sup _ {\pi^ {\prime} \in \Pi} \mathbb {E} _ {Q} \left[ Y (\pi^ {\prime} (X)) \right] - \mathbb {E} _ {Q} \left[ Y (\pi (X)) \right].
$$

The MMR policy, denoted by $\pi _ { R , \Lambda } ^ { \star }$ , minimizes this welfare loss uniformly over the ambiguity set ${ \mathcal { P } } ( \Lambda )$ . Formally, it is defined as the solution to the following minimax optimization problem:

$$
\pi_ {R, \Lambda} ^ {\star} \in \underset {\pi \in \Pi} {\operatorname{argmin}}   \sup _ {Q \in \mathcal {P} (\Lambda)} \operatorname{Reg} _ {Q} (\pi).\tag{B.2}
$$

Theorem B.1. Under Assumption 2.1, we have

$$
\inf _ {\pi \in \Pi} \sup _ {Q \in \mathcal {P} (\Lambda)} \operatorname{Reg} _ {Q} (\pi) = \inf _ {\pi \in \Pi} \sup _ {\pi^ {\prime} \in \Pi} \mathbb {E} \left[ b _ {\Lambda} (\pi , \pi^ {\prime}) (X) \right],\tag{B.3}
$$

where

$$
b _ {\Lambda} \left(\pi , \pi^ {\prime}\right) (x) = \tau_ {\Lambda} ^ {+} (x) \max \left\{\pi^ {\prime} (x) - \pi (x), 0 \right\} - \tau_ {\Lambda} ^ {-} (x) \max \left\{\pi (x) - \pi^ {\prime} (x), 0 \right\}.
$$

In particular, i $\begin{array} { r } { \mathrm { f } \Pi = \Pi _ { o } . } \end{array}$ , the first-best MMR policy

$$
\pi_ {R, \Lambda} ^ {\star} (x) = \left\{ \begin{array}{l l} 1, & \text {if} \tau_ {\Lambda} ^ {-} (x) \geq 0, \\ \tau_ {\Lambda} ^ {+} (x) / \left(\tau_ {\Lambda} ^ {+} (x) - \tau_ {\Lambda} ^ {-} (x)\right), & \text {if} \tau_ {\Lambda} ^ {-} (x) <   0 <   \tau_ {\Lambda} ^ {+} (x), \\ 0, & \text {if} \tau_ {\Lambda} ^ {+} (x) \leq 0, \end{array} \right.
$$

solves the minimax regret problem of Eq. (B.3).

Remark B.1. Theorem B.1 highlights a distinction between the MMR criterion and the max-min criteria considered in the main text. If $\tau _ { \Lambda } ^ { - } ( x ) > 0 \mathrm { o r } \tau _ { \Lambda } ^ { + } ( x ) < 0$ , the sign of the treatment effect is robustly determined, and the first-best MMR policy is deterministic. $\mathrm { I f } 0 \in \bigl ( \tau _ { \Lambda } ^ { - } ( x ) , \tau _ { \Lambda } ^ { + } ( x ) \bigr )$ , neither  treatment nor control uniformly dominates, and randomization arises as a way to balance the worstcase regret from choosing the wrong action. Additionally, when Π is restricted to be deterministic, the corresponding first-best MMR policy in this scenario becomes <sup>1</sup> $\{ \tau _ { \Lambda } ^ { + } ( x ) + \tau _ { \Lambda } ^ { - } ( x ) > 0 \}$

## Appendix C Additional Empirical Analyses

This section provides supplementary empirical results and robustness checks to complement the results in Section 6. Specifically, we extend our confounding-robust policy learning framework to incorporate self-selection as an explicit policy option for both the JTPA and Head Start studies. We then conduct sensitivity analyses to verify that our empirical conclusions remain robust to alternative choices of the truncation threshold α and the sensitivity parameter Λ.

## C.1 JTPA Application

We revisit the JTPA application from Section 6.1 and incorporate the self-selection as a policy option. Specifically, we apply the MMW policy with self-selection, as defined in Eq. (A.2a) and characterized by Theorem A.1. Relative to Section 6.1, the only additional component is $\mathbb { E } [ Y | X = x ]$ which is estimated using random forests. We restrict the policy class to depth-2 decision trees over {0, 1, S}, and estimate the MMW policy using policytree R package. Fig. 8 reports assignment fractions across policy arms and the corresponding welfare performance over the Λ-grid $\{ 1 , 1 . 5 , \ldots , 5 \}$ , which includes the sensitivity level calibrated using IV-bounds, $\Lambda = 4 . 5 .$

Panel (a) of Figure 8 reports the assignment fraction of the estimated depth-2 policy tree across the specified Λ-grid. At Λ = 1, the policy mirrors the unconfoundedness benchmark, assigning 93.5% of applicants to compulsory treatment and 6.5% to compulsory control, with no assignment to the self-selection arm. For $\Lambda > 1$ , the policy assigns most applicants to the self-selection arm: 98.5% at $\Lambda = 1 . 5 ,$ and more than 99.6% for all $\Lambda \geq 3 .$ . This pattern is consistent with the robust decision rule in Remark A.1. As Λ increases, the worst-case welfare bounds for compulsory assignments become less informative, and the resulting MMW policy tends to favor the self-selection arm, allowing applicants to make their own take-up decisions.

Panel (b) reports welfare gains relative to the compulsory un-treatment-only baseline. We compare compulsory treatment only, self-selection only, selection-absent targeting, and selection-driven targeting, with the compulsory un-treatment-only policy normalized to zero. We evaluate these policies using the worst-case welfare criterion $W _ { \Lambda }$ characterized in Theorem A.1. When $\Lambda = 1$ selection-driven and selection-absent targeting coincide and deliver higher welfare gains than the self-selection-only policy. For $\Lambda \geq 1 . 5$ , the selection-driven policy assigns almost all applicants to the self-selection arm, so its welfare gain becomes close to that of the self-selection-only policy.


Figure 8: Policy assignment fractions and welfare performance across values of Λ. Panel (a) plots assignment fractions across policy arms. Panel (b) plots welfare gains of four candidate policies relative to the compulsory un-treatment-only baseline.

## C.2 Head Start Application

## C.2.1 Robustness Checks

In this subsection, we examine the robustness of the Head Start results to two implementation choices: the propensity-score truncation level and the sensitivity parameter in the MSM. We first assess robustness to the propensity-score truncation level α by re-running the empirical analysis under $\alpha \in \left. 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 \right.$ }, holding all other implementation details fixed.

Table 3 reports the number and percentage of observations with estimated propensity scores outside $\left[ \alpha , 1 - \alpha \right]$ , both in the full sample and separately by treatment arms. The table shows that only a small fraction of observations have estimated propensity scores outside the truncation interval. Specifically, this fraction is 0.03% when $\alpha = 0 . 0 1$ , 0.39% under our baseline choice $\alpha = 0 . 0 5$ , and 3.24% under the more conservative truncation level $\alpha = 0 . 1$ . These small fractions suggest that the estimated policy tree and the subsequent empirical findings are not sensitive to the particular choice of α.

Table 3: Number and percentage of observations outside the propensity score truncation interval across choices of α with sample size $n = 3 , 8 2 6$

<table><tr><td></td><td> $\alpha = 0$ </td><td> $\alpha = {0.01}$ </td><td> $\alpha = {0.05}$ </td><td> $\alpha = {0.1}$ </td></tr><tr><td>Head Start</td><td>0</td><td>1</td><td>14</td><td>51</td></tr><tr><td>Other preschool</td><td>0</td><td>0</td><td>1</td><td>53</td></tr><tr><td>No preschool</td><td>0</td><td>0</td><td>0</td><td>20</td></tr><tr><td>Full sample</td><td>0</td><td>1</td><td>15</td><td>124</td></tr><tr><td>Percentage</td><td>0%</td><td>0.03%</td><td>0.39%</td><td>3.24%</td></tr></table>

We next turn to the robustness check regarding the sensitivity parameter Λ. We re-estimate the depth-2 MMW policy tree over the grid $\Lambda \in \{ 1 , 1 . 5 , 2 , 2 . 5 , 3 , 3 . 5 , 3 . 6 , 4 \}$ ; the resulting trees are displayed in Figure 9. For $\Lambda \geq 1 . 5$ , the learned policy trees share the same qualitative structure: Head Start is targeted to children from lower-income families whose mothers have relatively low AFQT percentile scores. The main change across Λ is the AFQT cutoff, which rises from 4.31 to 7.36, increasing the Head Start assignment share from 5.10% to 7.61%. For larger values of Λ, the MMW policy assigns a larger fraction of children to Head Start. In particular, the Head Start assignment share increases from 5.10% under $\Lambda \in \{ 1 . 5 , 2 , 2 . 5 , 3 \}$ to 7.61% under $\Lambda \in \{ 3 . 5 , 3 . 6 , 4 \}$

Figure 9: Sensitivity analysis of learned depth-2 policy trees across values of Λ. The trees have the same qualitative structure across the reported values of Λ, with changes only in the splitting thresholds. For larger values of Λ, the higher AFQT cutoff expands the Head Start assignment probability.

## C.2.2 Head Start program with self-selection as an option

We revisit the Head Start application in Section 6.2, now allowing self-selection as a policy option. We use the same sample, covariates, and estimations as in Section 6.2; the additional component, E[Y|X], is estimated by random forest regression. We report results over $\Lambda \in \{ 1 , 1 . 5 , \ldots , 4 \}$ , a grid that includes the sensitivity level $\Lambda = 3 . 6$ calibrated using the method of Hsu and Small (2013).

Panel (a) of Figure 10 reports the assignment fractions of the estimated depth-2 policy tree across values of Λ. At Λ = 1, the policy assigns 1.23% of children to Head Start, 86.83% to other preschool, and 11.94% to no preschool, with no assignment to the self-selection arm. For values of Λ above the unconfoundedness benchmark, the policy assigns most children to self-selection: 91.19% at $\Lambda = 1 . 5 .$ , and 95.71% for all $\Lambda \geq 2$

Panel (b) reports welfare performance. We compare six candidate policies: no preschool only, Head Start only, other preschool only, self-selection only, selection-absent targeting, and selectiondriven targeting. As in the JTPA application, welfare gains are measured relative to the no-preschoolonly baseline using the worst-case welfare criterion. The pattern is similar to that in the JTPA application. For values of Λ above the unconfoundedness benchmark, the selection-driven policy assigns most children to the self-selection arm, so its welfare gain is closed to that of the selfselection-only policy and exceeds that of selection-absent targeting.

(a) Share of treatment assignment

(b) Welfare gains
Figure 10: Policy assignment shares and welfare performance across Λ-grid. The figures plot (a) the share of treatment allocation for each policy arm, and (b) the welfare gains of candidate policies relative to the no-preschool-only.

## References

Abadie, A., Angrist, J., and Imbens, G. (2002). Instrumental variables estimates of the effect of subsidized training on the quantiles of trainee earnings. Econometrica, 70(1):91–117.

Adjaho, C. and Christensen, T. (2025). Externally valid policy choice. arXiv preprint arXiv:2205.05561.

Alatas, V., Purnamasari, R., Wai-Poi, M., Banerjee, A., Olken, B. A., and Hanna, R. (2016). Self-targeting: Evidence from a field experiment in Indonesia. Journal of Political Economy, 124(2):371–427.

Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of causal effects using instrumental variables. Journal ofthe American Statistical Association, 91(434):444–455.

Anthony, M. and Bartlett, P. L. (2009). Neural network learning: Theoretical foundations. Cambridge University Press.

Athey, S. and Imbens, G. W. (2017). The state of applied econometrics: Causality and policy evaluation. Journal of Economic Perspectives, 31(2):3–32.

Athey, S. and Wager, S. (2021). Policy learning with observational data. Econometrica, 89(1):133– 161.

Bartlett, P. L., Harvey, N., Liaw, C., and Mehrabian, A. (2019). Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research, 20(63):1–17.

Belloni, A. and Chernozhukov, V. (2011). ℓ1-penalized quantile regression in high-dimensional sparse models. The Annals of Statistics, 39(1):82–130.

Belloni, A., Chernozhukov, V., Chetverikov, D., and Fernández-Val, I. (2019). Conditional quantile processes based on series or many regressors. Journal ofEconometrics, 213(1):4–29.

Belloni, A., Chernozhukov, V., Fernandez-Val, I., and Hansen, C. (2017). Program evaluation and causal inference with high-dimensional data. Econometrica, 85(1):233–298.

Belloni, A., Chernozhukov, V., and Hansen, C. (2014). High-dimensional methods and inference on structural and treatment effects. Journal of Economic Perspectives, 28(2):29–50.

Ben-Michael, E., Greiner, D. J., Imai, K., and Jiang, Z. (2025). Safe policy learning through extrapolation: Application to pre-trial risk assessment. Journal ofthe American Statistical Association, 120(551):1386–1399.

Bloom, H. S., Orr, L. L., Bell, S. H., Cave, G., Doolittle, F., Lin, W., and Bos, J. M. (1997). The benefits and costs of JTPA Title II-A programs: Key findings from the National Job Training Partnership Act study. Journal ofHuman Resources, 32(3):549–576.

Chen, X. (2007). Large sample sieve estimation of semi-nonparametric models. Handbook of Econometrics, 6:5549–5632.

Chen, Y.-C. and Xie, H. (2022). Personalized subsidy rules. arXiv preprint arXiv:2202.13545.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68.

Chernozhukov, V., Escanciano, J. C., Ichimura, H., Newey, W. K., and Robins, J. M. (2022). Locally robust semiparametric estimation. Econometrica, 90(4):1501–1535.

Christensen, T., Moon, H. R., and Schorfheide, F. (2026). Optimal decision rules when payoffs are partially identified. The Review ofEconomic Studies.

Cui, Y. and Tchetgen Tchetgen, E. (2021). A semiparametric instrumental variable approach to optimal treatment regimes under endogeneity. Journal of the American Statistical Association, 116(533):162–173.

Currie, J. and Thomas, D. (1995). Does Head Start make a difference? American Economic Review, 85(3):341–364.

d’Adamo, R. (2021). Orthogonal policy learning under ambiguity. arXiv preprint arXiv:2111.10904.

Deming, D. (2009). Early childhood intervention and life-cycle skill development: Evidence from Head Start. American Economic Journal: Applied Economics, 1(3):111–134.

Dorn, J. and Guo, K. (2023). Sharp sensitivity analysis for inverse propensity weighting via quantile balancing. Journal of the American Statistical Association, 118(544):2645–2657.

Dorn, J., Guo, K., and Kallus, N. (2025). Doubly-valid/doubly-sharp sensitivity analysis for causal inference with unmeasured confounding. Journal of the American Statistical Association, 120(549):331–342.

Fan, Y., Qi, Y., and Xu, G. (2025). Policy learning with α-expected welfare. arXiv preprint arXiv:2505.00256.

Fang, E. X., Wang, Z., and Wang, L. (2023). Fairness-oriented learning for optimal individualized treatment rules. Journal of the American Statistical Association, 118(543):1733–1746.

Farrell, M. H., Liang, T., and Misra, S. (2021). Deep neural networks for estimation and inference. Econometrica, 89(1):181–213.

Fissler, T., Merz, M., and Wüthrich, M. V. (2023). Deep quantile and deep composite triplet regression. Insurance: Mathematics and Economics, 109:94–112.

Giné, E. and Nickl, R. (2021). Mathematicalfoundations ofinfinite-dimensional statistical models. Cambridge University Press.

Higbee, S. D. (2025). Policy learning with new treatments. Quantitative Economics, 16(4):1409– 1456.

Hsu, J. Y. and Small, D. S. (2013). Calibrating sensitivity analyses to observed covariates in observational studies. Biometrics, 69(4):803–811.

Ida, T., Ishihara, T., Ito, K., Kido, D., Kitagawa, T., Sakaguchi, S., and Sasaki, S. (2026). Choosing who chooses: Selection-driven targeting in energy rebate programs. Econometrica, 94(1):225– 247.

Imbens, G. W. and Angrist, J. D. (1994). Identification and estimation of local average treatment effects. Econometrica, 62(2):467–475.

Ito, K., Ida, T., and Tanaka, M. (2023). Selection on welfare gains: Experimental evidence from electricity plan choice. American Economic Review, 113(11):2937–2973.

Johnson, M. S., Levine, D. I., and Toffel, M. W. (2023). Improving regulatory effectiveness through better targeting: Evidence from OSHA. American Economic Journal: Applied Economics, 15(4):30–67.

Kallus, N. and Zhou, A. (2021). Minimax-optimal policy learning under unobserved confounding. Management Science, 67(5):2870–2890.

Khan, S., Saveski, M., and Ugander, J. (2023). Off-policy evaluation beyond overlap: partial identification through smoothness. arXiv preprint arXiv:2305.11812.

Kido, D. (2022). Distributionally robust policy learning with Wasserstein distance. arXiv preprint arXiv:2205.04637.

Kitagawa, T. and Tetenov, A. (2018). Who should be treated? Empirical welfare maximization methods for treatment choice. Econometrica, 86(2):591–616. Supplementary materials and online appendix available at https://doi.org/10.3982/ECTA13288.

Kitagawa, T. and Tetenov, A. (2021). Equality-minded treatment choice. Journal of Business & Economic Statistics, 39(2):561–574.

Kohler, M. and Langer, S. (2021). On the rate of convergence of fully connected deep neural network regression estimates. The Annals of Statistics, 49(4):2231–2249.

Lei, L., Sahoo, R., and Wager, S. (2023). Policy learning under biased sample selection. arXiv preprint arXiv:2304.11735.

Leigh, J. P. (2011). Economic burden of occupational injury and illness in the United States. The Milbank Quarterly, 89(4):728–772.

Leqi, L. and Kennedy, E. H. (2021). Median optimal treatment regimes. arXiv preprint arXiv:2103.01802.

Li, Q. and Racine, J. S. (2007). Nonparametric econometrics: theory and practice. Princeton University Press.

Liu, Y. (2022). Policy learning under endogeneity using instrumental variables. arXiv preprint arXiv:2206.09883.

Ludwig, J. and Miller, D. L. (2007). Does Head Start improve children’s life chances? Evidence from a regression discontinuity design. The Quarterly Journal ofEconomics, 122(1):159–208.

Manski, C. F. (2000). Identification problems and decisions under ambiguity: Empirical analysis of treatment response and normative analysis of treatment choice. Journal of Econometrics, 95(2):415–442.

Manski, C. F. (2004). Statistical treatment rules for heterogeneous populations. Econometrica, 72(4):1221–1246.

Manski, C. F. (2013). Public policy in an uncertain world: analysis and decisions. Harvard University Press.

Manski, C. F. (2025). Identification and statistical decision theory. Econometric Theory, 41(4):977– 993.

Masten, M. A. and Poirier, A. (2018). Identification of treatment effects under conditional partial independence. Econometrica, 86(1):317–351.

Masten, M. A. and Poirier, A. (2020). Inference on breakdown frontiers. Quantitative Economics, 11(1):41–111.

Montiel Olea, J. L., Qiu, C., and Stoye, J. (2026). Decision theory for treatment choice problems with partial identification. Review of Economic Studies, page rdag015.

Newey, W. (1994). The asymptotic variance of semiparametric estimators. Econometrica, 62(6):1349–82.

Olma, T. (2021). Nonparametric estimation of truncated conditional expectation functions. arXiv preprint arXiv:2109.06150.

Oprescu, M., Dorn, J., Ghoummaid, M., Jesson, A., Kallus, N., and Shalit, U. (2023). B-learner: Quasi-oracle bounds on heterogeneous causal effects under hidden confounding. In International Conference on Machine Learning, pages 26599–26618. PMLR.

Padilla, O. H. M., Tansey, W., and Chen, Y. (2022). Quantile regression with ReLU networks: Estimators and minimax rates. Journal ofMachine Learning Research, 23(247):1–42.

Pu, H. and Zhang, B. (2021). Estimating optimal treatment rules with an instrumental variable: A partial identification learning approach. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(2):318–345.

Qi, Z., Pang, J.-S., and Liu, Y. (2023). On robustness of individualized decision rules. Journal of the American Statistical Association, 118(543):2143–2157.

Qiu, H., Carone, M., Sadikova, E., Petukhova, M., Kessler, R. C., and Luedtke, A. (2021). Optimal individualized decision rules using instrumental variable methods. Journal of the American Statistical Association, 116(533):174–191.

Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. Journal ofthe American Statistical Association, 89(427):846– 866.

Rosenbaum, P. R. and Rubin, D. B. (1983). Assessing sensitivity to an unobserved binary covariate in an observational study with binary outcome. Journal ofthe Royal Statistical Society: Series B (Methodological), 45(2):212–218.

Sasaki, Y. and Ura, T. (2024). Welfare analysis via marginal treatment effects. Econometric Theory, pages 1–24.

Schmidt-Hieber, J. (2020). Nonparametric regression using deep neural networks with ReLU activation function. The Annals ofStatistics, 48(4):1875.

Si, N., Zhang, F., Zhou, Z., and Blanchet, J. (2023). Distributionally robust batch contextual bandits. Management Science, 69(10):5772–5793.

Stoye, J. (2012). Minimax regret treatment choice with covariates or with limited validity of experiments. Journal ofEconometrics, 166(1):138–156.

Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., and Wager, S. (2020). policytree: Policy learning via doubly robust empirical welfare maximization over trees. Journal of Open Source Software, 5(50):2232.

Tan, Z. (2006). A distributional approach for causal inference using propensity scores. Journal of the American Statistical Association, 101(476):1619–1637.

van der Vaart, A. W. and Wellner, J. A. (1996). Weak Convergence and Empirical Processes: With Applications to Statistics. Springer.

Viviano, D. and Bradic, J. (2024). Fair policy targeting. Journal of the American Statistical Association, 119(545):730–743.

Walters, C. R. (2015). Inputs in the production of early childhood human capital: Evidence from Head Start. American Economic Journal: Applied Economics, 7(4):76–102.

Wang, L., Zhou, Y., Song, R., and Sherwood, B. (2018). Quantile-optimal treatment regimes. Journal ofthe American Statistical Association, 113(523):1243–1254.

Yata, K. (2025). Optimal decision rules under partial identification. arXiv preprint arXiv:2111.04926.

Yu, M., Wang, Y., Xie, S., Tan, K. M., and Zhou, W.-X. (2025). Estimation and inference for nonparametric expected shortfall regression over RKHS. Journal of the American Statistical Association, pages 1–12.

Zhang, Y., Ben-Michael, E., and Imai, K. (2024a). Safe policy learning under regression discontinuity designs with multiple cutoffs. arXiv preprint arXiv:2208.13323.

Zhang, Y., Huang, M., and Imai, K. (2024b). Minimax regret estimation for generalizing heterogeneous treatment effects with multisite data. arXiv preprint arXiv:2412.11136.

Zhao, Q., Small, D. S., and Bhattacharya, B. B. (2019). Sensitivity analysis for inverse probability weighting estimators via the percentile bootstrap. Journal of the Royal Statistical Society Series B: Statistical Methodology, 81(4):735–761.

Zhou, Z., Athey, S., and Wager, S. (2023). Offline multi-action policy learning: Generalization and optimization. Operations Research, 71(1):148–183.

## Appendix D Identification Results under MSM

In this section, we present partial identification results for the conditional mean and CATE under the MSM, drawing primarily from (Dorn and Guo, 2023; Oprescu et al., 2023; Dorn et al., 2025). Recall that $\mu _ { o } ( x , a ) = \mathbb { E } _ { P _ { o } } \left[ Y ( a ) | X = x \right]$ denotes the conditional mean of the potential outcome, and the true CATE is defined as $\tau _ { o } ( x ) = \mu _ { o } ( x , 1 ) - \mu _ { o } ( x , 0 )$

To formally characterize partial identification, we describe the identified sets of these functions under distributional uncertainty, where the true counterfactual distribution may deviate within an uncertainty set ${ \mathcal { P } } ( \Lambda )$ . For any distribution $Q \in { \mathcal { P } } ( \Lambda )$ , define $\mu _ { Q } ( x , a ) = \mathbb { E } _ { Q } [ Y ( a ) | X = x ]$ and $\tau _ { Q } ( x ) = \mu _ { Q } ( x , 1 ) - \mu _ { Q } ( x , 0 )$ , which represent the conditional mean and CATE under Q. The identified sets for these functions are then

$$
\begin{array}{r} \Theta_ {\mu , \Lambda} (x, a) \equiv \left\{\mu_ {Q} (x, a): Q \in \mathcal {P} (\Lambda) \right\}, \\ \Theta_ {\tau , \Lambda} (x) \equiv \left\{\tau_ {Q} (x): Q \in \mathcal {P} (\Lambda) \right\}, \end{array}
$$

for any $x \in \mathcal { X }$ and $a \in \{ 0 , 1 \}$ . When $\Lambda = 1$ , the uncertainty set ${ \mathcal { P } } ( \Lambda )$ collapses to a singleton, implying point identification: $\mu _ { o } ( x , a )$ and $\tau _ { o } ( x )$ are uniquely determined by $\Theta _ { \mu , \Lambda } ( x , a )$ and $\Theta _ { \tau , \Lambda } ( x )$

For general Λ, let $\mu _ { \Lambda } ^ { \pm } ( x , a )$ denote the endpoints of $\Theta _ { \mu , \Lambda } ( x , a )$ , i.e.,

$$
\mu_ {\Lambda} ^ {-} (x, a) = \inf _ {Q \in \mathcal {P} (\Lambda)} \mu_ {Q} (x, a) \quad \text { and } \quad \mu_ {\Lambda} ^ {+} (x, a) = \sup _ {Q \in \mathcal {P} (\Lambda)} \mu_ {Q} (x, a).
$$

The following Propositions D.1 and D.2 provide sharp upper and lower bounds for both $\Theta _ { \mu , \Lambda } ( x , a )$ and $\Theta _ { \tau , \Lambda } ( x )$

Proposition D.1. Under Assumption 2.1, there are distributions $P _ { \mu , a } ^ { \pm } \in \mathcal { P } ( \Lambda )$ such that $\mu _ { P _ { \mu , a } ^ { + } } ( x , a ) =$ $\mu _ { \Lambda } ^ { + } ( x , a )$ and $\mu _ { P _ { u , a } ^ { - } } ( x , a ) = \mu _ { \Lambda } ^ { - } ( x , a )$ , almost surely. In particular, these sharp bounds can be expressed as:

$$
\mu_ {\Lambda} ^ {\pm} (x, a) = \mathbb {E} \left[ Y \mathbb {1} \{A = a \} \left[ 1 + \frac {1 - e _ {a} (X)}{e _ {a} (X)} \Lambda^ {\pm \mathrm{sgn} \left(Y - q _ {\Lambda} ^ {\pm} (X, a)\right)} \right] \Big | X = x \right],\tag{D.1}
$$

where $\operatorname { s g n } ( t ) = 1 \operatorname { i f } t \geq 0$ and −1 otherwise.

We next characterize the upper and lower bounds for $\Theta _ { \tau , \Lambda } ( x )$ . Specifically, the endpoints of $\Theta _ { \tau , \Lambda } ( x )$ are defined as

$$
\tau_ {\Lambda} ^ {-} (x) = \inf _ {Q \in \mathcal {P} (\Lambda)} \tau_ {Q} (x) \quad \text { and } \quad \tau_ {\Lambda} ^ {+} (x) = \sup _ {Q \in \mathcal {P} (\Lambda)} \tau_ {Q} (x).
$$

The following result, Proposition D.2, builds on Proposition D.1.

Proposition D.2. Under Assumption 2.1, there are distributions $P _ { \tau } ^ { \pm } \in \mathcal { P } ( \Lambda )$ such that $\tau _ { P _ { \tau } ^ { + } } ( x ) =$ $\tau _ { \Lambda } ^ { + } ( x )$ and $\tau _ { P _ { \tau } ^ { - } } ( x ) = \tau _ { \Lambda } ^ { - } ( x )$ , almost surely. Moreover, the sharp bounds for the CATE satisfy:

$$
\tau_ {\Lambda} ^ {\pm} (x) = \mu_ {\Lambda} ^ {\pm} (x, 1) - \mu_ {\Lambda} ^ {\mp} (x, 0).\tag{D.2}
$$

We illustrate the intuition behind Proposition D.1. Suppose we had oracle access to the true propensity score $e _ { o } ( X , U )$ . Under Assumption 2.1, the true conditional mean $\mu _ { o } ( x , a )$ would then be point-identified via IPW:

$$
\mu_ {o} (x, a) = \mathbb {E} _ {P _ {o}} \left[ \frac {Y \mathbb {1} \{A = a \}}{a e _ {o} (X , U) + (1 - a) (1 - e _ {o} (X , U))} \Big | X = x \right],
$$

However, when $\Lambda > 1$ , the true propensity score $e _ { o } ( X , U )$ is no longer identifiable. Instead, Assumption 2.1 restricts $e _ { o } ( X , U )$ to an uncertainty set, which in turn induces a partially identified set for $\mu _ { o } ( x , a )$ . Specifically, the uncertainty set consists of all random variables E that satisfy both the sensitivity restriction and the usual balancing constraints:

$$
\Lambda^ {- 1} \leq \frac {E / (1 - E)}{e (X) / (1 - e (X))} \leq \Lambda ,\tag{D.3}
$$

$$
\mathbb {E} \left[ A / E | X \right] = \mathbb {E} \left[ (1 - A) / (1 - E) | X \right] = 1.\tag{D.4}
$$

Formally, we define

$$
\mathcal {E} (\Lambda) = \left\{E: \text { Eq.   (D.3)   and   Eq.   (D.4)   hold   a.s. } \right\}.
$$

Given this uncertainty set, the identified set $\Theta _ { \mu , \Lambda } ( x , a )$ can be equivalently expressed as:

$$
\Theta_ {\mu , \Lambda} (x, a) = \left\{\mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \mid X = x \right]: E \in \mathcal {E} (\Lambda) \right\}.
$$

For completeness, the formal proofs of Propositions D.1 and D.2 are provided in Section F for interested readers.

Remark D.1. Our Propositions D.1 and D.2 extend the pointwise identification results of Dorn and Guo (2023); Oprescu et al. (2023) to a uniform setting, ensuring that the identified bounds hold simultaneously for almost all x and a. Such uniform bounds are crucial for policy evaluation and learning under the MSM. First, computing the robust criteria $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ involves integration over the covariate and treatment spaces, which requires uniformly valid bounds on $\Theta _ { \mu } ( x , a )$ and $\Theta _ { \tau } ( x )$ . Second, the closed-form sharp bounds enable the construction of orthogonal moments for the efficient estimation of $W _ { \Lambda } ( \pi )$ and $\Delta _ { \Lambda } ( \pi )$ . As shown by Kallus and Zhou (2021), the policy improvement guarantee can be severely biased by the slow convergence of propensity score estimation; by leveraging our identification results, this bias can be mitigated via orthogonal moments, yielding ${ \sqrt { n } } .$ -consistent estimates of $W _ { \Lambda } ( \cdot )$ and $\Delta _ { \Lambda } ( \cdot )$

Remark D.2. Kallus and Zhou (2021), building on Zhao et al. (2019), constructs an uncertainty set for the putative propensity scores by imposing unconditional balancing constraints, specifically $\mathbb { E } [ A / E ] = \mathbb { E } [ ( 1 - A ) / ( 1 - E ) ] = 1$ . In contrast, our uncertainty set ${ \mathcal { E } } ( \Lambda )$ incorporates the conditional balancing constraints defined in Eq. (D.4), which are strictly stronger than those employed in Kallus and Zhou (2021). According to Dorn and Guo (2023), such conditional balancing constraints are necessary for obtaining sharp bounds on the CATE, as established in Proposition D.2. Consequently, the policy learned in Kallus and Zhou (2021) is more conservative than our MMI policy.

## Appendix E Extending MMW Policy to Multi-Valued Treatments

This section extends our max-min welfare (MMW) policy learning approach to the multi-valued treatment setting, providing theoretical justification for its application in Section 6.2. Our identification here is constructive, directly informing a procedure for estimating the lower bound of the worst-case welfare function. This, in turn, allows for policy optimization using a method directly analogous to that in Algorithm 1. As these procedures and the corresponding doubly robust estimation closely follow the binary treatment case, we omit a detailed discussion here.

The identification results, including sharp upper and lower bounds for conditional mean outcomes and a lower bound for the worst-case welfare function, are established in Section E.1. Detailed proofs are presented in Section H.

We first extend the MSM framework to the setting where the treatment variable A takes more than two values. Let A denote the set of d possible treatment levels: $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { d } \}$ . Define the true and nominal propensity scores, respectively, as $e _ { o , a } ( x , u ) = \mathbb { P } _ { P _ { o } } [ A = a | X = x , U = u ]$ and $e _ { a } ( x ) = \mathbb { P } _ { P _ { o } } [ A = a | X = x ]$ for $a \in { \mathcal { A } }$

Assumption E.1. Suppose there exists a vector of unobserved confounders $U \in \mathbb { R } ^ { k }$ such that

$$
\left(Y \left(a _ {1}\right), Y \left(a _ {2}\right), \dots , Y \left(a _ {d}\right)\right) \perp A \mid (X, U).
$$

The distribution of $( X , Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \dots , Y ( a _ { d } ) , A , U )$ satisfies the selection bias condition with $1 \leq \Lambda < \infty$ if the following inequality holds $P _ { o }$ -almost surely,

$$
\frac {1}{\Lambda} \leq \frac {e _ {o , a} (x , u) / (1 - e _ {o , a} (x , u))}{e _ {a} (x) / (1 - e _ {a} (x))} \leq \Lambda \quad \text { for } \quad a \in \mathcal {A}.\tag{E.1}
$$

Assumption E.1 naturally induces a distributional uncertainty set over the counterfactual distribution of $( X , Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \dots , Y ( a _ { d } ) , A , U )$ . Formally, let $\mathcal { P } _ { \mathrm { M } }$ denote the set of all probability distributions Q on $\mathcal { X } \times \mathbb { R } ^ { d } \times \mathcal { A } \times \mathbb { R } ^ { k }$ that satisfy:

(1) If $( X , Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \dots , Y ( a _ { d } ) , A , U ) \sim Q$ , then $( Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \ldots , Y ( a _ { d } ) ) \bot \bot A$ (X, U) under Q;

(2) If $Y = Y ( A )$ , then the distribution of $\left( X , Y , A \right)$ under Q coincides with the observed-data distribution $P ;$

(3) For all $a \in { \mathcal { A } }$ , the odds ratio between the true propensity score and the nominal propensity score lies in $[ 1 / \Lambda , \Lambda ] , \mathrm { i . e . }$ •

$$
\frac {1}{\Lambda} \leq \frac {\mathbb {P} (A = a | X , U) / (1 - \mathbb {P} (A = a | X , U))}{\mathbb {P} (A = a | X) / (1 - \mathbb {P} (A = a | X))} \leq \Lambda , Q \text {-a.s.}
$$

## E.1 Identifying the Worst-Case Welfare Function

For any $( x , a ) \in { \mathcal { X } } \times { \mathcal { A } }$ , let $\mu _ { Q } ( x , a ) = \mathbb { E } _ { Q } [ Y ( a ) | X = x ]$ denote the conditional mean under a counterfactual distribution Q. The identified set for $\mu _ { o } ( x , a ) = \mathbb { E } _ { P _ { o } } [ Y ( a ) | X = x ]$ is formally given by

$$
\Theta_ {\mathrm{M}, \mu} (x, a) \equiv \left\{\mu_ {Q} (x, a): Q \in \mathcal {P} _ {\mathrm{M}} \right\}.
$$

Let $\mu ^ { \pm } ( x , a )$ denote the endpoints of $\Theta _ { \mathrm { M } , \mu } ( x , a )$ , i.e.,

$$
\mu^ {-} (x, a) = \inf _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mu_ {Q} (x, a) \quad \text { and } \quad \mu^ {+} (x, a) = \sup _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mu_ {Q} (x, a).
$$

The following Proposition E.1 establishes the sharp upper and lower bounds for the identified set $\Theta _ { \mathrm { M } , \mu } ( x , a )$

Proposition E.1. Under Assumption E.1, there exist distributions $P _ { \mu , a } ^ { \pm } \in \mathcal { P } _ { \mathrm { M } }$ such that $\mu _ { P _ { \mu , a } ^ { + } } ( x , a ) =$ $\mu ^ { + } ( x , a )$ and $\mu _ { P _ { u , a } ^ { - } } ( x , a ) = \mu ^ { - } ( x , a )$ ) for almost all $( x , a ) \in { \mathcal { X } } \times { \mathcal { A } }$ , respectively. In particular, the sharp bounds admit the closed-form representation:

$$
\mu^ {\pm} (x, a) = \mathbb {E} \left[ Y \mathbb {1} \{A = a \} \left[ 1 + \frac {1 - e _ {a} (X)}{e _ {a} (X)} \Lambda^ {\pm \operatorname{sgn} \left(Y - q ^ {\pm} (X, a)\right)} \right] | X = x \right].\tag{E.2}
$$

In the multi-valued treatment setting, a policy is a mapping from the input space $\mathcal { X }$ to a decision $a \in A .$ . A randomized policy can be represented as a function from $\mathcal { X }$ to the probability simplex $\Delta ( \mathcal { A } )$ over the action space ${ \mathcal { A } } .$ In contrast, a deterministic policy is a function $\pi : \mathcal { X }  \{ 0 , 1 \} ^ { d }$ where the output indicates a deterministic choice among d possible actions.

The worst-case welfare function $W ( \pi )$ introduced in Section 2.2 can be naturally extended to the multi-valued treatment setting:

$$
W (\pi) = \inf _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mathbb {E} _ {Q} [ Y (\pi (X)) ] = \inf _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mathbb {E} _ {Q} \left[ \sum_ {a \in \mathcal {A}} Y (a) \pi_ {a} (X) \right],
$$

Let $\Pi _ { \mathrm { M } }$ denote the set of multi-valued policies specified by the policy maker. The corresponding max-min welfare (MMW) policy is obtained by solving $\mathrm { m a x } _ { \pi \in \Pi _ { \mathrm { M } } } W ( \pi )$ . The following theorem characterizes a lower bound for $W ( \pi )$

Theorem E.1. Under Assumption E.1, for any policy $\pi \in \Pi _ { \mathrm { M } }$ •

$$
W (\pi) \geq \mathbb {E} \left[ \sum_ {a \in \mathcal {A}} \mu^ {-} (X, a) \pi_ {a} (X) \right].
$$

Remark E.1. Whether the lower bound for $W ( \pi )$ in Theorem E.1 is sharp remains an open question. While sharp lower bounds for $\mu _ { o } ( x , a )$ are available for any fixed $( x , a )$ , identifying a single distribution that simultaneously attains these bounds for all $( x , a )$ may be challenging and lies beyond the scope of this study.

## Appendix F Proofs for Results in the Main Text

Since the sensitivity parameter Λ is treated as fixed throughout, we omit the subscript Λ for notational simplicity. For instance, we may write $\mathcal { P } \equiv \mathcal { P } ( \Lambda ) , \mu ^ { + } ( x , a ) \equiv \mu _ { \Lambda } ^ { + } ( x , a )$ and $\rho ^ { - } ( x , a ) \equiv$ $\rho _ { \Lambda } ^ { - } ( x , a )$ . This convention will be used throughout the remainder of the appendix.

## F.1 Proof of Proposition D.1

Proof of Proposition D.1. Step 1. Preliminary Results. We begin by showing that for any $( x , a ) \in$ ${ \mathcal { X } } \times \{ 0 , 1 \}$ , the partially identified set $\Theta _ { \mu } ( x , a )$ is an interval whose endpoints solve the following optimization problems:

$$
\begin{array}{l} \mu^ {+} (x, a) = \sup _ {E \in \mathcal {E}} \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right], \\ \mu^ {-} (x, a) = \inf _ {E \in \mathcal {E}} \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right]. \end{array}\tag{F.1}
$$

We focus on verifying the result for $\mu ^ { + } ( x , a )$ ; the argument for $\mu ^ { - } ( x , a )$ is analogous and omitted for brevity.

By Proposition 1B of Dorn and Guo (2023), for any random variable E defined on the same probability space as $\left( X , Y , A \right)$ and satisfying E $\left[ \mathbb { 1 } \left\{ A = a \right\} / \left( a E + ( 1 - a ) ( 1 - E ) \right) | X \right] = 1$ , we can construct a distribution $Q \in { \mathcal { P } }$ such that

$$
\mathbb {E} _ {Q} \left[ Y (a) | X = x \right] = \mathbb {E} _ {Q} \left[ \frac {Y \mathbb {1} \{A = a \}}{\mathbb {P} _ {Q} (A = a | X , U)} \Big | X = x \right] = \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right].\tag{F.2}
$$

Thus,

$$
\mu^ {+} (x, a) = \sup _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} [ Y (a) | X = x ] \geq \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right].
$$

Since this inequality holds for any E satisfying the balancing condition, it holds in particular for the supremum over all such E.

Conversely, for any $Q \in { \mathcal { P } }$

$$
\begin{array}{r l} & {\mathbb {E} _ {Q} \left[ Y (a) | X = x \right] = \mathbb {E} _ {Q} \left[ \frac {Y \mathbb {1} \left\{A = a \right\}}{\mathbb {P} _ {Q} (A = a | X , U)} \Big | X = x \right]} \\ & {\qquad = \mathbb {E} _ {Q} \left[ Y \mathbb {1} \left\{A = a \right\} \mathbb {E} \left[ \frac {1}{\mathbb {P} _ {Q} (A = a | X , U)} \Big | X, Y, A = a \right] \Big | X = x \right].} \end{array}
$$

Define $e _ { Q } ( X , Y , A ) = 1 / \mathbb { E } \left[ 1 / \mathbb { P } _ { Q } ( A = a | X , U ) | X , Y , A \right]$ , and introduce a random variable $E$ on the same probability space $P$ -such that $A E + ( 1 - A ) ( 1 - \bar { E } ) = e _ { Q } ( X , Y , A )$ . It is straightforward to verify that $E \in { \mathcal { E } }$ and

$$
\mathbb {E} _ {Q} \left[ Y (a) | X = x \right] = \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{e _ {Q} (X , Y , a)} \Big | X = x \right] = \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right].
$$

Thus,

$$
\mathbb {E} _ {Q} \left[ Y (a) | X = x \right] \leq \sup _ {E \in \mathcal {E}} \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right].
$$

Since $Q$ is arbitrary, the inequality continues to hold after taking the supremum over $Q \in { \mathcal { P } }$

Finally, the proof that the partially identified set $\Theta _ { \mu } ( x , a )$ forms an interval follows the same argument as in Dorn and Guo (2023).

Step 2. The Closed-form Expression. For $a \in \{ 0 , 1 \}$ , define $E _ { a } ^ { \pm }$ by

$$
\frac {1}{a E _ {a} ^ {\pm} + (1 - a) \left(1 - E _ {a} ^ {\pm}\right)} = 1 + \frac {1 - e _ {a} (X)}{e _ {a} (X)} \Lambda^ {\pm \operatorname{sgn} \left(Y - q ^ {\pm} (X, a)\right)}.\tag{F.3}
$$

It is straightforward to verify that $E _ { a } ^ { \pm } \in \mathcal { E }$ , and thus $E _ { a } ^ { \pm }$ is feasible for Eq. (F.1). For any $( x , a ) \in$ ${ \mathcal { X } } \times \{ 0 , 1 \}$ and $E \in { \mathcal { E } }$ , we have

$$
\begin{array}{l} \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right] \\ = q ^ {+} (x, a) \mathbb {E} \left[ \frac {\mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right] + \mathbb {E} \left[ \frac {(Y - q ^ {+} (X , a)) \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \Big | X = x \right] \\ \leq q ^ {+} (x, a) \mathbb {E} \left[ \frac {\mathbb {1} \{A = a \}}{a E _ {a} ^ {+} + (1 - a) (1 - E _ {a} ^ {+})} \Big | X = x \right] + \mathbb {E} \left[ \frac {(Y - q ^ {+} (X , a)) \mathbb {1} \{A = a \}}{a E _ {a} ^ {+} + (1 - a) (1 - E _ {a} ^ {+})} \Big | X = x \right] \\ = \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E _ {a} ^ {+} + (1 - a) (1 - E _ {a} ^ {+})} \Big | X = x \right], \end{array}
$$

where the inequality follows because $1 / \left( a E _ { a } ^ { + } + ( 1 - a ) ( 1 - E _ { a } ^ { + } ) \right)$ attains the maximum (minimum) allowable value when $\left( Y - q ^ { + } ( X , a ) \right) \mathbb { 1 } \left\{ A = a \right\}$ is positive (negative). By a similar argument, we can show that for any $( x , a ) \in \mathcal { X } \times \{ 0 , 1 \}$ and $E \in { \mathcal { E } }$

$$
\mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E + (1 - a) (1 - E)} \mid X = x \right] \geq \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E _ {a} ^ {-} + (1 - a) (1 - E _ {a} ^ {-})} \mid X = x \right].
$$

Since E is arbitrary, this completes the proof.

Step 3. The Existence of Counterfactual Distribution. Eqs. (F.1) and (F.3) together imply that random variables $E _ { a } ^ { \pm } \in \mathcal { E }$ satisfy

$$
\begin{array}{l} \mu^ {+} (x, a) = \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E _ {a} ^ {+} + (1 - a) (1 - E _ {a} ^ {+})} \Big | X = x \right], \\ \mu^ {-} (x, a) = \mathbb {E} \left[ \frac {Y \mathbb {1} \{A = a \}}{a E _ {a} ^ {-} + (1 - a) (1 - E _ {a} ^ {-})} \Big | X = x \right], \end{array}
$$

in $( x , a ) \in \mathcal { X } \times \{ 0 , 1 \}$ almost surely. By Proposition 1B in Dorn and Guo (2023) and Eq. (F.2), we conclude that there exist distributions $P _ { \mu , a } ^ { \pm } \in \mathcal { P }$ such that $\mu _ { P _ { \mu , a } ^ { + } } ( x , a ) = \mu ^ { + } ( x , a )$ and $\mu _ { P _ { \mu , a } ^ { - } } ( x , a ) =$ $\mu ^ { - } ( x , a ) \mathrm { i n } ( x , a ) \in \mathcal { X } \times \{ 0 , 1 \}$ almost surely. □

## F.2 Proof of Proposition D.2

ProofofProposition D.2. We complete the proof by constructing a data-compatible distribution in $\mathcal { P }$ that simultaneously attains $\mu ^ { + } ( x , 1 )$ and $\mu ^ { - } ( x , 0 )$ . Similar arguments can be applied to derive the lower bound, which we omit for brevity.

Define $E _ { \tau } ^ { + } = A E _ { 1 } ^ { + } + ( 1 - A ) E _ { 0 } ^ { - }$ . It is straightforward to verify that $E _ { \tau } ^ { + } ~ \in ~ \mathcal { E }$ . Applying Proposition D.1 and similar arguments as C.4.2 in Dorn and Guo (2023), it is straightforward to construct the distribution $P _ { \tau } ^ { + } \in \mathcal { P }$ such that

$$
\mathbb {E} _ {P _ {\tau} ^ {+}} [ Y (1) - Y (0) | X = x ] = \mathbb {E} \left[ \frac {Y A}{E _ {\tau} ^ {+}} \mid X = x \right] - \mathbb {E} \left[ \frac {Y (1 - A)}{1 - E _ {\tau} ^ {+}} \mid X = x \right] = \mu^ {+} (x, 1) - \mu^ {-} (x, 0)
$$

in $x \in \mathcal { X }$ almost surely. Since

$$
\begin{array}{l} \tau^ {+} (x) = \sup _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \left[ Y (1) - Y (0) | X = x \right] \\ \qquad \leq \sup _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \left[ Y (1) | X = x \right] - \inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \left[ Y (0) | X = x \right] \\ \qquad = \mu^ {+} (x, 1) - \mu^ {-} (x, 0), \end{array}
$$

in $x \in \mathcal { X }$ almost surely. We conclude that $\tau ^ { + } ( x ) = \mu ^ { + } ( x , 1 ) - \mu ^ { - } ( x , 0 )$ in $x \in \mathcal { X }$ almost surely. □

## F.3 Proof of Theorem 3.1

Proof of Theorem 3.1. Notice that

$$
\begin{array}{r l} & {\inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} [ Y (\pi (X)) ] = \inf _ {Q \in \mathcal {P}} \Big \{\mathbb {E} \Big [ \pi (X) \mu_ {Q} (X, 1) \Big ] + \mathbb {E} \Big [ (1 - \pi (X)) \mu_ {Q} (X, 0) \Big ] \Big \}} \\ & {\qquad \geq \inf _ {Q \in \mathcal {P}} \mathbb {E} \Big [ \pi (X) \mu_ {Q} (X, 1) \Big ] + \inf _ {Q \in \mathcal {P}} \mathbb {E} \Big [ (1 - \pi (X)) \mu_ {Q} (X, 0) \Big ].} \end{array}
$$

Then we will show that inf and expectation operators are exchangeable. On one hand, for any distribution $Q \in { \mathcal { P } }$ ，

$$
\mathbb {E} \Big [ \pi (X) \mu_ {Q} (X, 1) \Big ] \geq \mathbb {E} \left[ \pi (X) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) \right].
$$

Since Q is arbitrary, the inequality continues to hold after taking the infimum over $\mathcal { P }$ on both sides. For the other side, Proposition D.1 has shown that there exists a distribution $P _ { \mu , 1 } ^ { - } \in \mathcal { P }$ satisfying

$\mu ^ { - } ( x , 1 ) = \mu _ { P _ { \mu , 1 } ^ { - } } ( x , 1 )$ for any $x \in \mathcal { X }$ . Thus,

$$
\inf _ {Q \in \mathcal {P}} \mathbb {E} \left[ \pi (X) \mu_ {Q} (X, 1) \right] \leq \mathbb {E} \left[ \pi (X) \mu_ {P _ {\mu , 1} ^ {-}} (X, 1) \right] = \mathbb {E} \left[ \pi (X) \mu^ {-} (X, 1) \right] = \mathbb {E} \left[ \pi (X) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) \right].
$$

Combining the results above, we can conclude that

$$
\inf _ {Q \in \mathcal {P}} \mathbb {E} \left[ \pi (X) \mu_ {Q} (X, 1) \right] = \mathbb {E} \left[ \pi (X) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) \right].
$$

Similar arguments can be applied to derive that

$$
\inf _ {Q \in \mathcal {P}} \mathbb {E} \Big [ (1 - \pi (X)) \mu_ {Q} (X, 0) \Big ] = \mathbb {E} \left[ (1 - \pi (X)) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 0) \right].
$$

Define $E ^ { - } = A E _ { 1 } ^ { - } + ( 1 - A ) E _ { 0 } ^ { - }$ satisfying $E ^ { - } \in \mathcal { E }$ . Applying Proposition D.1 and similar arguments as C.4.3 in Dorn and Guo (2023), we can construct the distribution $P ^ { - }$ fulfilling

$$
\mathbb {E} _ {P ^ {-}} [ Y (1) | X = x ] = \mathbb {E} \left[ \frac {Y A}{E ^ {-}} \mid X = x \right] = \mu^ {-} (x, 1)
$$

and

$$
\mathbb {E} _ {P ^ {-}} [ Y (0) | X = x ] = \mathbb {E} \left[ \frac {Y (1 - A)}{1 - E ^ {-}} \Big | X = x \right] = \mu^ {-} (x, 0).
$$

which imply that

$$
\mathbb {E} _ {P ^ {-}} [ Y (\pi (X)) ] = \mathbb {E} \left[ \pi (X) \mu^ {-} (X, 1) + (1 - \pi (X)) \mu^ {-} (X, 0) \right].
$$

Thus, we conclude that

$$
\begin{array}{c} \inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} [ Y (\pi (X)) ] \leq \mathbb {E} _ {P ^ {-}} [ Y (\pi (X)) ] \\ = \mathbb {E} \left[ \pi (X) \mu^ {-} (X, 1) + (1 - \pi (X)) \mu^ {-} (X, 0) \right] \\ = \mathbb {E} \left[ \pi (X) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) \right] + \mathbb {E} \left[ (1 - \pi (X)) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 0) \right]. \end{array}
$$

Results above together indicate that

$$
\begin{array}{c} \inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} [ Y (\pi (X)) ] = \mathbb {E} \left[ \pi (X) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) \right] + \mathbb {E} \left[ (1 - \pi (X)) \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 0) \right] \\ = \mathbb {E} \Big [ \pi (X) \left(\mu^ {-} (X, 1) - \mu^ {-} (X, 0)\right) \Big ] + \mathbb {E} [ \mu^ {-} (X, 0) ], \end{array}
$$

which completes the proof.

## F.4 Proof of Theorem 3.2

ProofofTheorem 3.2. Applying similar arguments as the proof of Theorem 3.1, we have that

$$
\inf _ {Q \in \mathcal {P}} \mathbb {E} \left[ \tau_ {Q} (X) \pi (X) \right] = \mathbb {E} \left[ \inf _ {Q \in \mathcal {P}} \tau_ {Q} (X) \pi (X) \right] = \mathbb {E} \left[ \tau^ {-} (X) \pi (X) \right].\tag{F.4}
$$

According to Eq. (F.4), the first-best policy is given by $\pi _ { \Delta } ^ { \star } ( x ) = 1 \{ \tau ^ { - } ( x ) > 0 \}$

## F.5 Proof of Proposition 3.1

Recall that

$$
\psi_ {W} (z, \pi ; \eta_ {W}) = \sum_ {t \in \{0, 1 \}} \phi_ {t} ^ {-} (z; \eta_ {W}) \pi (t | x),
$$

where $\eta _ { W } = \big ( e , q ^ { - } , \rho _ { 1 } ^ { - } , \rho _ { 0 } ^ { - } \big ) , \phi _ { t } ^ { - } \left( z ; \eta _ { W } \right)$ are defined in Eq. (3.2), and the functions $\rho _ { 1 } ^ { \pm } ( x , t )$ and $\rho _ { 0 } ^ { \pm } ( x , t )$  are defined in Eq. (3.3).

Proof of Proposition 3.1. Step 1. The proof of part (1) follows directly from Theorem 3.1, together with the following derivation:

$$
\mathbb {E} \left[ \mathbb {1} \{A = t \} \left(\frac {1}{1 + \Lambda} - \mathbb {1} \left\{Y <   q ^ {-} (X, t) \right\}\right) | X \right] = \mathbb {E} [ \mathbb {1} \{A = t \} - e _ {t} (X) | X ] = 0.\tag{F.5}
$$

Step 2. To show part (2), given the linearity of pathwise derivative, it suffices to show the pathwise derivative of $r \mapsto \mathbb { E } \left[ \psi _ { W } ( Z , \pi ; \eta _ { W , \Lambda } + r \bar { \eta } ) \right] = 0$ for all perturbation directions $\bar { \eta }$ varying only in the components corresponding to $e , q ^ { - } , \rho _ { 1 } ^ { - }$ and $\rho _ { 0 } ^ { - }$ , respectively. We only show that

$$
\frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \psi_ {W} (Z, \pi ; e, q ^ {-} + r (\widetilde {q} ^ {-} - q ^ {-}), \rho_ {1} ^ {-}, \rho_ {0} ^ {-}) \right] _ {r = 0} = 0,\tag{F.6}
$$

for any $\widetilde { q } ^ { - }$ belonging to some small neighborhood of $q ^ { - }$ . The derivations of the pathwise derivatives ewith respect to the perturbation directions in $e , \rho _ { 1 } ^ { - }$ and $\rho _ { 0 } ^ { - }$ are analogous and thus omitted for brevity. Notice that

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \psi_ {W} \left(Z, \pi ; e, q ^ {-} + r (\widetilde {q} ^ {-} - q ^ {-}), \rho_ {1} ^ {-}, \rho_ {0} ^ {-}\right) \right] \\ = \sum_ {t \in \{0, 1 \}} \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \phi_ {t} ^ {-} \left(Z; e, q ^ {-} + r (\widetilde {q} ^ {-} - q ^ {-}), \rho_ {1} ^ {-}, \rho_ {0} ^ {-}\right) \pi (t | X) \right]. \end{array}
$$

To derive Eq. (F.6), it suffices to show that

$$
\frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \phi_ {t} ^ {-} (Z; e, q ^ {-} + r (\widetilde {q} ^ {-} - q ^ {-}), \rho_ {1} ^ {-}, \rho_ {0} ^ {-}) \pi (t | X) \right] _ {r = 0} = 0,
$$

for all $t \in \{ 0 , 1 \}$ . By expanding the term for t = 1 and using Eq. (F.5), we have

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ \phi_ {1} ^ {-} (Z; e, q ^ {-} + r (\widetilde {q} ^ {-} - q ^ {-}), \rho_ {1} ^ {-}, \rho_ {0} ^ {-}) \pi (X) \right] _ {r = 0} \\ = \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ Y A \left(1 + \frac {1 - e (X)}{e (X)} \Lambda^ {- \operatorname{sgn} (Y - q ^ {-} (X, 1) - r (\widetilde {q} ^ {-} (X, 1) - q ^ {-} (X, 1)))}\right) \pi (X) \right] _ {r = 0} \\ + \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ q ^ {-} (X, 1) A \frac {1 - e (X)}{e (X)} (\Lambda - \Lambda^ {- 1}) \left(\frac {1}{1 + \Lambda} - \mathbb {1} \{Y <   q ^ {-} (X, 1) + r (\widetilde {q} ^ {-} (X, 1) - q ^ {-} (X, 1)) \}\right) \pi (X) \right] _ {r = 0}, \end{array}
$$

where $\pi ( 1 | X ) = \pi ( X )$ by notation. Let $\mathrm { I I } _ { A . 1 }$ and $\mathrm { I I } _ { A . 2 }$ denote the two summands in the expression

above. It is not difficult to verify that

$$
\begin{array}{l} \Pi_ {A. 1} = \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ Y A \frac {1 - e (X)}{e (X)} (\Lambda - \Lambda^ {- 1}) \mathbb {1} \left\{Y <   q ^ {-} (X, 1) + r (\widetilde {q} ^ {-} (X, 1) - q ^ {-} (X, 1)) \right\} \pi (X) \right] _ {r = 0} \\ = \frac {\mathrm{d}}{\mathrm{d} r} \mathbb {E} \left[ (1 - e (X)) (\Lambda - \Lambda^ {- 1}) \pi (X) \mathbb {E} \left[ Y \mathbb {1} \left\{Y <   q ^ {-} (X, 1) + r (\widetilde {q} ^ {-} (X, 1) - q ^ {-} (X, 1)) \right\} | X, A = 1 \right] \right] _ {r = 0} \\ = \mathbb {E} \left[ (1 - e (X)) (\Lambda - \Lambda^ {- 1}) \pi (X) \frac {\mathrm{d}}{\mathrm{d} r} \left[ \int_ {- \infty} ^ {q ^ {-} (X, 1) + r (\widetilde {q} ^ {-} (X, 1) - q ^ {-} (X, 1))} y f _ {Y} (y | X, 1) d y \right] _ {r = 0} \right] \\ = \mathbb {E} \left[ q ^ {-} (X, 1) (1 - e (X)) (\Lambda - \Lambda^ {- 1}) f _ {Y} (q ^ {-} (X, 1) | X, 1) (\widetilde {q} ^ {-} (X, 1) - q ^ {-} (X, 1)) \pi (X) \right] \\ = - \Pi_ {A. 2}, \end{array}
$$

where the fourth equality follows from Assumption 3.1. This implies that the pathwise derivative with respect to $q ^ { - }$ <sup>−</sup> is zero. The results for $t \ : = \ : 0$ follow from an analogous procedure, thereby completing the proof of Part (2). □

## F.6 Proof of Proposition 3.2

Proof of Proposition 3.2. The proof is identical to that of Proposition 3.1 and is omitted here.

## F.7 Proof of Lemma 4.1

Proof of Lemma 4.1. We focus on deriving the result for $\widehat { W } _ { n } ( \pi ) - W _ { n } ( \pi )$ ; the argument for $\widehat { \Delta } _ { n } ( \pi ) -$ $\Delta _ { n } ( \pi )$ is similar and omitted for brevity. Notice that

$$
\begin{array}{l} \widehat {W} _ {n} (\pi) - W _ {n} (\pi) = \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) \left(\phi_ {1} ^ {-} \left(Z _ {i}; \widehat {\eta} _ {W} ^ {- k}\right) - \phi_ {1} ^ {-} \left(Z _ {i}; \eta_ {W}\right)\right) \\ \qquad + \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} (1 - \pi (X _ {i})) \left(\phi_ {0} ^ {-} \left(Z _ {i}; \widehat {\eta} _ {W} ^ {- k}\right) - \phi_ {0} ^ {-} \left(Z _ {i}; \eta_ {W}\right)\right). \end{array}
$$

We bound the first term at the right-hand-side. The analysis of the second term is similar and omitted. To simplify notation, we suppress the superscript −k of the nuisance estimators, $\mathrm { e . g . , } \widehat { \eta } _ { W } = \widehat { \eta } _ { W } ^ { - k }$

By definition,

$$
\begin{array}{l} \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) \left(\phi_ {1} ^ {-} \left(Z _ {i}; \widehat {\eta} _ {W}\right) - \phi_ {1} ^ {-} \left(Z _ {i}; \eta_ {W}\right)\right) \\ = \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) Y _ {i} A _ {i} \left(\frac {1 - \widehat {e} (X _ {i})}{\widehat {e} (X _ {i})} \Lambda^ {- \mathrm{sgn} \left(Y _ {i} - \widehat {q} ^ {-} (X _ {i}, 1)\right)} - \frac {1 - e (X _ {i})}{e (X _ {i})} \Lambda^ {- \mathrm{sgn} \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right)}\right) \\ + \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \left(\frac {1}{1 + \Lambda} - \mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\}\right) \left(\widehat {q} ^ {-} (X _ {i}, 1) \frac {1 - \widehat {e} (X _ {i})}{\widehat {e} (X _ {i})} - q ^ {-} (X _ {i}, 1) \frac {1 - e (X _ {i})}{e (X _ {i})}\right) \\ - \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} q ^ {-} (X _ {i}, 1) \frac {1 - e (X _ {i})}{e (X _ {i})} \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) \\ - \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \left(\Lambda \widehat {\rho_ {1}} (X _ {i}, 1) + \Lambda^ {- 1} \widehat {\rho_ {0}} (X _ {i}, 1)\right) \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right) \\ - \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) \frac {A _ {i} - e (X _ {i})}{e (X _ {i})} \left[ \Lambda (\widehat {\rho_ {1}} (X _ {i}, 1) - \rho_ {1} ^ {-} (X _ {i}, 1)) + \Lambda^ {- 1} (\widehat {\rho_ {0}} (X _ {i}, 1) - \rho_ {0} ^ {-} (X _ {i}, 1)) \right]. \end{array}
$$

Denote these five summands by $D _ { j } ( \pi )$ for $1 \le j \le 5$

First term. To bound the first term, it is useful to separate the contributions of each of the K folds:

$$
\begin{array}{l} D _ {1, k} (\pi) = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) Y _ {i} A _ {i} \left(\frac {1 - \widehat {e} (X _ {i})}{\widehat {e} (X _ {i})} \Lambda^ {- \mathrm{sgn} \left(Y _ {i} - \widehat {q} ^ {-} (X _ {i}, 1)\right)} - \frac {1 - e (X _ {i})}{e (X _ {i})} \Lambda^ {- \mathrm{sgn} \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right)}\right) \\ = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) Y _ {i} A _ {i} \Lambda^ {- \mathrm{sgn} \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right)} \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right) \\ + \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) Y _ {i} A _ {i} \frac {1 - e (X _ {i})}{e (X _ {i})} \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) \\ + \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) Y _ {i} A _ {i} \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right) \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) \\ = D _ {1, k} ^ {(1)} (\pi) + D _ {1, k} ^ {(2)} (\pi) + D _ {1, k} ^ {(3)} (\pi). \end{array}
$$

We can use the Cauchy-Schwartz inequality to verify that

$$
\begin{array}{l} \sup _ {\pi \in \Pi_ {n}} \left| D _ {1, k} ^ {(3)} (\pi) \right| \leq \left(\Lambda - \Lambda^ {- 1}\right) \sqrt {\frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} Y _ {i} ^ {2} \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right) ^ {2}} \\ \qquad \times \sqrt {\frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} (\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i} , 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i} , 1) \right\}) ^ {2}} \\ = O _ {P} \left(\frac {b (n)}{n ^ {(\zeta_ {e} + \zeta_ {q}) / 2}}\right) = o _ {P} \left(n ^ {- 1 / 2}\right), \end{array}
$$

where the first equality holds by Markov’s inequality, Assumption 4.2 and Assumption 4.4. Then, uniformly over $\pi \in \Pi _ { n }$

$$
D _ {1, k} (\pi) = D _ {1, k} ^ {(1)} (\pi) + D _ {1, k} ^ {(2)} (\pi) + o _ {P} \left(n ^ {- 1 / 2}\right).
$$

Second term. For bounding the second term, we still separate out the contributions of the K different folds. After applying similar arguments as the preceding one, we obtain that uniformly over $\pi \in \Pi _ { n }$

$$
D _ {2, k} (\pi) = \widetilde {D} _ {2, k} (\pi) + o _ {P} \left(n ^ {- 1 / 2}\right),\tag{F.7}
$$

where

$$
\begin{array}{c} \widetilde {D} _ {2, k} (\pi) = \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \left(\frac {1}{1 + \Lambda} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) \\ \times \left(\widehat {q} ^ {-} (X _ {i}, 1) \frac {1 - \widehat {e} (X _ {i})}{\widehat {e} (X _ {i})} - q ^ {-} (X _ {i}, 1) \frac {1 - e (X _ {i})}{e (X _ {i})}\right). \end{array}
$$

Since $\widehat { q } ^ { - }$ and $\widehat { e }$ were computed using data $\{ Z _ { i } \} _ { i \in \mathcal { T } _ { k } ^ { c } }$ , one has

$$
\mathbb {E} \left[ \widetilde {D} _ {2, k} (\pi) \Big | \widehat {q} ^ {-}, \widehat {e} \right] = 0.
$$

By Assumption 4.4, we have that uniformly over i,

$$
\left| \widehat {q} ^ {-} (X _ {i}, 1) \frac {1 - \widehat {e} (X _ {i})}{\widehat {e} (X _ {i})} - q ^ {-} (X _ {i}, 1) \frac {1 - e (X _ {i})}{e (X _ {i})} \right| \leq 1
$$

with probability approaching to 1. So the individual summands in $\widetilde { D } _ { 2 , k } ( \pi )$ are all ν-sub Gaussian with probability approaching to 1. Define

$$
\widehat {V} _ {2, k} = \frac {\Lambda}{(1 + \Lambda) ^ {2}} \mathbb {E} \left[ e (X _ {i}) \left(\widehat {q} ^ {-} (X _ {i}, 1) \frac {1 - \widehat {e} (X _ {i})}{\widehat {e} (X _ {i})} - q ^ {-} (X _ {i}, 1) \frac {1 - e (X _ {i})}{e (X _ {i})}\right) ^ {2} \Big | \widehat {q} ^ {-}, \widehat {e} \right].
$$

We can apply Corollary 3 in Athey and Wager (2021) to establish that

$$
\begin{array}{l} \mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| \widetilde {D} _ {2, k} (\pi) \right| \left| \widehat {q ^ {-}}, \widehat {e} \right. \right] = \mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| \widetilde {D} _ {2, k} (\pi) - \mathbb {E} \left[ \widetilde {D} _ {2, k} (\pi) \left| \widehat {q ^ {-}}, \widehat {e} \right. \right] \right| \left| \widehat {q ^ {-}}, \widehat {e} \right. \right] \\ = O \left(\sqrt {\operatorname{VC} (\Pi_ {n}) \frac {\widehat {V} _ {2 , k}}{n}}\right). \end{array}\tag{F.8}
$$

Assumption 4.4, an application of Jensen’s inequality and Eqs. (F.7) and (F.8) result that

$$
\mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} | D _ {2} (\pi) | \right] = O \left(\sqrt {\operatorname{VC} \left(\Pi_ {n}\right) \frac {b (n)}{n ^ {1 + \min \{\zeta_ {e} , \zeta_ {q} \}}}}\right).
$$

Third term. Denote

$$
D _ {3, k} (\pi) = - \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} q ^ {-} (X _ {i}, 1) \frac {1 - e (X _ {i})}{e (X _ {i})} \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right).
$$

as the k-th fold in $D _ { 3 } ( \pi )$ . Then

$$
\begin{array}{c} D _ {1, k} ^ {(2)} (\pi) + D _ {3, k} (\pi) \\ = \frac {\Lambda - \Lambda^ {- 1}}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \frac {1 - e (X _ {i})}{e (X _ {i})} \Big (Y _ {i} - q ^ {-} (X _ {i}, 1) \Big) \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right). \end{array}
$$

Define

$$
\widehat {V} _ {3, k} = \mathbb {E} \left[ A _ {i} \Big (Y _ {i} - q ^ {-} (X _ {i}, 1) \Big) ^ {2} \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) ^ {2} \Big | \widehat {q} ^ {-} \right].
$$

By the law of iterated expectation, E $\left[ \widehat { V } _ { 3 , k } \right]$ can be expressed as

$$
\mathbb {E} \left[ \widehat {V} _ {3, k} \right] = \mathbb {E} \left[ e (X _ {i}) \mathbb {E} \left[ \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right) ^ {2} \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) ^ {2} \mid X _ {i}, A _ {i} = 1, \widehat {q} ^ {-} \right] \right].
$$

Notice that

$$
\begin{array}{l} \mathbb {E} \left[ \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right) ^ {2} \left(\mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\}\right) ^ {2} \Big | X _ {i}, A _ {i} = 1, \widehat {q} ^ {-} \right] \\ = \int_ {\widehat {q} ^ {-} (X _ {i}, 1)} ^ {q ^ {-} (X _ {i}, 1)} \left(y - q ^ {-} (X _ {i}, 1)\right) ^ {2} d F _ {Y _ {i}} \left(y \Big | X _ {i}, A _ {i} = 1, \widehat {q} ^ {-}\right) + \int_ {q ^ {-} (X _ {i}, 1)} ^ {\widehat {q} ^ {-} (X _ {i}, 1)} \left(y - q ^ {-} (X _ {i}, 1)\right) ^ {2} d F _ {Y _ {i}} \left(y \Big | X _ {i}, A _ {i} = 1, \widehat {q} ^ {-}\right) \\ = O _ {P} \left(\left(\widehat {q} ^ {-} (X _ {i}, 1) - q ^ {-} (X _ {i}, 1)\right) ^ {3}\right), \end{array}
$$

where the last equality follows by the mean value theorem. Hence, by Assumption 4.4,

$$
\mathbb {E} \left[ \widehat {V} _ {3, k} \right] = O \left(\frac {b (n)}{n ^ {3 \zeta_ {q} / 2}}\right).
$$

After applying a similar argument as the proof of $\widetilde { D } _ { 2 , k } ( \pi )$ , we have that for any k,

$$
\mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| D _ {1, k} ^ {(2)} (\pi) + D _ {3, k} (\pi) - \mathbb {E} \left[ D _ {1, k} ^ {(2)} (\pi) + D _ {3, k} (\pi) \mid \widehat {q} ^ {-} \right] \right| \mid \widehat {q} ^ {-} \right] = O \left(\sqrt {\operatorname{VC} \left(\Pi_ {n}\right) \frac {b (n)}{n ^ {1 + 3 \zeta_ {q} / 2}}}\right).\tag{F.9}
$$

By Assumption 4.3, $\pi ( X _ { i } ) \left( 1 - e ( X _ { i } ) \right) / e ( X _ { i } )$ is bounded uniformly over i and π. Thus,

$$
\begin{array}{l} \mathbb {E} \left[ D _ {1, k} ^ {(2)} (\pi) + D _ {3, k} (\pi) \Big | \widehat {q} ^ {-} \right] \\ \leq \frac {\Lambda - \Lambda^ {- 1}}{K} \frac {1 - \kappa}{\kappa} \mathbb {E} \left[ A _ {i} \Big | Y _ {i} - q ^ {-} (X _ {i}, 1) \Big | \left| \mathbb {1} \left\{Y _ {i} <   \widehat {q} ^ {-} (X _ {i}, 1) \right\} - \mathbb {1} \left\{Y _ {i} <   q ^ {-} (X _ {i}, 1) \right\} \right| \Big | \widehat {q} ^ {-} \right] \\ = O \left(\frac {b (n)}{n ^ {\zeta_ {q}}}\right) \end{array}\tag{F.10}
$$

uniformly over $\pi \in \Pi _ { n }$ . Applying Jensen’s inequality, triangle inequality, and Eqs. (F.9) and (F.10) yield

$$
\mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| D _ {1} ^ {(2)} (\pi) + D _ {3} (\pi) \right| \right] = O \left(\sqrt {\operatorname{VC} \left(\Pi_ {n}\right) \frac {b (n)}{n ^ {1 + 3 \zeta_ {q} / 2}}} + \frac {b (n)}{n ^ {\zeta_ {q}}}\right).
$$

Fourth term. Define the k-th fold of $D _ { 4 } ( \pi )$ as

$$
D _ {4, k} (\pi) = - \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \left(\Lambda \widehat {\rho_ {1} ^ {-}} (X _ {i}, 1) + \Lambda^ {- 1} \widehat {\rho_ {0} ^ {-}} (X _ {i}, 1)\right) \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right).
$$

Applying similar arguments as the preceding one, we can derive that

$$
D _ {4, k} (\pi) = \widetilde {D} _ {4, k} (\pi) + O _ {P} \left(n ^ {- 1 / 2}\right)
$$

uniformly over $\pi \in \Pi _ { n }$ , where

$$
\widetilde {D} _ {4, k} (\pi) = - \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \left(\Lambda \rho_ {1} ^ {-} (X _ {i}, 1) + \Lambda^ {- 1} \rho_ {0} ^ {-} (X _ {i}, 1)\right) \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right).
$$

Then we consider

$$
\begin{array}{c} D _ {1, k} ^ {(1)} (\pi) + \widetilde {D} _ {4, k} (\pi) \\ = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) A _ {i} \left(Y _ {i} \Lambda^ {- \mathrm{sgn} \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right)} - \Lambda \rho_ {1} ^ {-} (X _ {i}, 1) - \Lambda^ {- 1} \rho_ {0} ^ {-} (X _ {i}, 1)\right) \left(\frac {1}{\widehat {e} (X _ {i})} - \frac {1}{e (X _ {i})}\right). \end{array}
$$

It is not difficult to verify that

$$
\mathbb {E} \left[ D _ {1, k} ^ {(1)} (\pi) + \widetilde {D} _ {4, k} (\pi) \Big | \widehat {e} \right] = 0,
$$

because

$$
\mathbb {E} \left[ Y _ {i} \Lambda^ {- \operatorname{sgn} \left(Y _ {i} - q ^ {-} (X _ {i}, 1)\right)} - \Lambda \rho_ {1} ^ {-} (X _ {i}, 1) - \Lambda^ {- 1} \rho_ {0} ^ {-} (X _ {i}, 1) \Big | X _ {i}, A _ {i} = 1, \widehat {e} \right] = 0.
$$

Similar arguments are used to show that

$$
\mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| D _ {1} ^ {(1)} (\pi) + D _ {4} (\pi) \right| \right] = O \left(\sqrt {\operatorname{VC} \left(\Pi_ {n}\right) \frac {b (n)}{n ^ {1 + \zeta_ {e}}}}\right).
$$

Fifth term. Recall that E $\begin{array} { r } { \left\lceil A _ { i } - e ( X _ { i } ) \right\rceil X _ { i } , \widehat { \rho _ { 1 } } , \widehat { \rho _ { 0 } } \rceil = 0 . } \end{array}$ Applying a similar argument to bound b bthe fourth term, we can similarly bound the fifth term as follows:

$$
\mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} | D _ {5} (\pi) | \right] = O \left(\sqrt {\operatorname{VC} \left(\Pi_ {n}\right) \frac {b (n)}{n ^ {1 + \zeta_ {\rho}}}}\right).
$$

Wrapping Up Lemma 4.1. Combining five terms above with Assumption 4.1 and Assumption 4.4 gives

$$
\begin{array}{l} \mathbb {E} \left[ \sup _ {\pi \in \Pi_ {n}} \left| \frac {1}{n} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {I} _ {k}} \pi (X _ {i}) \left(\phi_ {1} ^ {-} \left(Z _ {i}; \widehat {\eta} _ {W}\right) - \phi_ {1} ^ {-} \left(Z _ {i}; \eta_ {W}\right)\right) \right| \right] \\ = O \left(\sqrt {\frac {\mathrm{VC} (\Pi_ {n}) b (n)}{n ^ {1 + \min \{\zeta_ {\rho} , \zeta_ {q} , \zeta_ {e} \}}}}\right) + O \left(\frac {b (n)}{n ^ {\zeta_ {q}}}\right) \\ = o (n ^ {- 1 / 2}). \end{array}
$$

## F.8 Proof of Theorem 4.1

ProofofTheorem 4.1. We prove Eq. (4.5); the proof of Eq. (4.6) follows by a similar argument and is omitted here for brevity.

Suppose $\pi _ { W , n } \in \mathrm { a r g m a x } _ { \pi \in \Pi _ { n } } W _ { n } ( \pi )$ . Otherwise, the proof can be modified by incorporating an ε-approximate optimizer argument. By the definitions of $\pi _ { W , n }$ and ${ \widehat { \pi } } _ { W , n }$ and Assumption 4.4, it follows that for any $\pi \in \Pi _ { n }$ :

$$
\begin{array}{l} W (\pi) - W (\widehat {\pi} _ {W, n}) = W (\pi) - W _ {n} (\pi) + \underbrace {W _ {n} (\pi) - W _ {n} (\pi_ {W , n})} _ {\leq 0} + \underbrace {W _ {n} (\pi_ {W , n}) - \widehat {W} _ {n} (\pi_ {W , n})} _ {= o _ {P} (n ^ {- 1 / 2})} \\ \quad + \underbrace {\widehat {W} _ {n} (\pi_ {W , n}) - \widehat {W} _ {n} (\widehat {\pi} _ {W , n})} _ {\leq 0} + \underbrace {\widehat {W} _ {n} (\widehat {\pi} _ {W , n}) - W _ {n} (\widehat {\pi} _ {W , n})} _ {o _ {P} (n ^ {- 1 / 2})} + W _ {n} (\widehat {\pi} _ {W, n}) - W (\widehat {\pi} _ {W, n}) \\ \quad \leq W (\pi) - W _ {n} (\pi) + W _ {n} (\widehat {\pi} _ {W, n}) - W (\widehat {\pi} _ {W, n}) + r _ {n} \\ \quad \leq 2 \sup _ {\pi \in \Pi_ {n}} | W (\pi) - W _ {n} (\pi) | + r _ {n} \end{array}
$$

where $r _ { n } = o _ { P } ( n ^ { - 1 / 2 } )$ and ${ \sqrt { n } } \mathbb { E } | r _ { n } | = o ( 1 )$ . For notational simplicity, we write

$$
\psi_ {W, \pi}: z \mapsto \psi_ {W} (z, \pi ; \eta_ {W}).
$$

Thus, for all $\pi \in \Pi _ { n }$

$$
\begin{array}{c} W (\pi) - W (\widehat {\pi} _ {W, n}) \leq 2 \sup _ {\pi \in \Pi_ {n}} | W _ {n} (\pi) - W (\pi) | + r _ {n} \\ = 2 \sup _ {\pi \in \Pi_ {n}} | (\mathbb {E} _ {n} - \mathbb {E})   \psi_ {W, \pi} | + r _ {n}. \end{array}\tag{F.11}
$$

Without loss of generality, suppose that there exists $\pi _ { W , n } ^ { * } \in \Pi _ { n }$ such that $W ( \pi _ { W , n } ^ { * } ) = \mathrm { m a x } _ { \pi \in \Pi _ { n } } W ( \pi )$ If no such $\pi _ { W , n } ^ { * }$ exists, the proof can be adapted using an ε-approximate optimizer, where $\varepsilon  0$ Substituting $\pi _ { W , n } ^ { * }$ into the preceding expression yields

$$
0 \leq \operatorname{Reg} _ {W} (\widehat {\pi} _ {W, n}) = W (\pi_ {W, n} ^ {*}) - W (\widehat {\pi} _ {W, n}) \leq 2 \sup _ {\pi \in \Pi_ {n}} | (\mathbb {E} _ {n} - \mathbb {E})   \psi_ {W, \pi} | + r _ {n}.
$$

Define ${ \mathcal { F } } _ { n } = \{ \psi _ { W , \pi } : \pi \in \Pi _ { n } \}$ . To complete the proof, it suffices to show the empirical process term decaying at rate $\sqrt { \mathrm { V C } ( \Pi _ { n } ) / n }$ , i.e.,

$$
\sup _ {\psi \in \mathcal {F} _ {n}} | (\mathbb {E} _ {n} - \mathbb {E}) \psi | = \sup _ {\pi \in \Pi_ {n}} | (\mathbb {E} _ {n} - \mathbb {E}) \psi_ {W, \pi} | = O _ {P} \left(\sqrt {\operatorname{VC} (\Pi_ {n}) / n}\right).\tag{F.12}
$$

We show Eq. (F.12) using a similar argument in Theorem 2.14.1 in van der Vaart and Wellner (1996), while allowing the function class $\mathcal { F } _ { n }$ to vary with n.

Step 1. Envelope function for ${ \mathcal { F } } _ { n } .$ As a first step for upper bounding $\operatorname* { s u p } _ { \psi \in { \mathcal { F } } _ { n } } | ( \mathbb { E } _ { n } - \mathbb { E } ) \psi |$ we construct an envelope function for the function class ${ \mathcal { F } } _ { n }$ . Under Assumption 4.3, we have the

following bound:

$$
\begin{array}{l} \mathbb {E} | Y | ^ {2} = \sum_ {a \in \{0, 1 \}} \int \mathbb {E} \left[ Y ^ {2} | X = x, A = a \right] e _ {a} (x) \mathrm{d} F _ {X | A} (x | a) \\ \qquad \qquad \qquad \geq \kappa \int \mathbb {E} \left[ Y ^ {2} | X = x, A = a \right] \mathrm{d} F _ {X | A} (x | a), \end{array}
$$

where $F _ { X \mid A } ( x | a )$ denote the conditional distribution of X given $A = a$ . Now, for any fixed $\Lambda \ \in$ $[ 1 , \infty )$ and $j , a \in \{ 0 , 1 \}$ , applying Jensen’s inequality along with the inequality above yields:

$$
\begin{array}{l} \mathbb {E} \big | \rho_ {j, \Lambda} ^ {-} (X, a) \big | ^ {2} \leq \mathbb {E} \left| \mathbb {E} \left[ Y ^ {2} | X, A = a \right] \right| ^ {2} = \int \mathbb {E} \left[ Y ^ {2} | X = x, A = a \right] \mathrm{d} F _ {X | A = a} (x) \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

As a result, the functions $e ( x ) , q ^ { - } ( x , a ) , \rho _ { 1 } ^ { - } ( x , a )$ , and $\rho _ { 0 } ^ { - } ( x , a )$ are all square-integrable, and hence $\phi _ { t } ^ { - } \left( z ; \eta _ { W } \right)$ ) defined in Eq. (3.2) for $t \in \{ 0 , 1 \}$ is also square-integrable by Minkowski inequality. The function $F ( \cdot ) \equiv \left| \phi _ { 0 } ^ { - } ( \cdot ; \eta _ { W } ) \right| + \left| \phi _ { 1 } ^ { - } ( \cdot ; \eta _ { W } ) \right|$ is an envelop function for ${ \mathcal { F } } _ { n }$ satisfying $| \psi | \leq F$ for all $\psi \in { \mathcal { F } } _ { n }$ and $\| F \| _ { \mathbb { P } , 2 } < \infty$

Step 2. Uniform entropy bound. Second, we derive a uniform upper bound on the covering number of $\mathcal { F } _ { n }$ for all n. Lemma A.1 in supplement of Kitagawa and Tetenov (2018) implies that $\mathrm { V C } ( { \mathcal { F } } _ { n } ) \leq \mathrm { V C } ( \Pi _ { n } )$ . By Theorem 2.6.7 in van der Vaart and Wellner (1996), there is a universal $K > 0$ such that for all n, the following inequality holds

$$
\sup _ {Q} \log N \left(\epsilon \| F \| _ {Q, 2}, \mathcal {F} _ {n}, L ^ {2} (Q)\right) \leq K \mathrm{VC} (\mathcal {F} _ {n}) \log \epsilon^ {- 1}, \quad \forall 0 <   \epsilon <   1,
$$

where the supremum is taken over all discrete probability measures Q with $\| F \| _ { Q , 2 } \ > \ 0$ . As a consequence, the uniform entropy can be upper bounded by:

$$
\sup _ {Q} \int_ {0} ^ {1} \sqrt {1 + \log N (\epsilon \| F \| _ {Q , 2} , \mathcal {F} _ {n} , L ^ {2} (Q))} d \epsilon \leq 1 + \sqrt {K} (1 + \sqrt {\pi} / 2) \sqrt {\operatorname{VC} (\Pi_ {n})}.
$$

It is important to note that the universal constant K is independent of the function class ${ \mathcal { F } } _ { n } ;$ see Theorem 2.6.4 and 2.6.7 in van der Vaart and Wellner (1996) for details.

Step 3. Bounding the supremum of the symmetrized process. Given the sample $\{ Z _ { i } \} _ { i = 1 } ^ { n } .$ define the symmetrized empirical process $\left\{ \mathbb { G } _ { n } ^ { o } f : f \in { \mathcal { F } } _ { n } \right\}$ as

$$
\mathbb {G} _ {n} ^ {o}: f \mapsto \frac {1}{\sqrt {n}} \sum_ {i = 1} ^ {n} \varepsilon_ {i} f (Z _ {i}),
$$

where the $\varepsilon _ { i }$ are independent Rademacher random variables such that $\varepsilon _ { i } = \pm 1$ with probability 1/2 each. Moreover, let $\phi ( x ) = e ^ { x ^ { 2 } } - 1$ and the conditional Orlicz norm on $\mathcal { F } _ { n }$ is defined as

$$
\left\| \mathbb {G} _ {n} ^ {o} f \right\| _ {\phi , n} = \inf \left\{c > 0: \mathbb {E} \left[ \phi \left(\left| \mathbb {G} _ {n} ^ {o} f \right| / c\right) \mid \left\{Z _ {i} \right\} _ {i = 1} ^ {n} \right] \leq 1 \right\}.
$$

For more details on Orlicz norms, see in Chapter 2.2 in van der Vaart and Wellner (1996).

Conditionally on $\{ Z _ { i } \} _ { i = 1 } ^ { n }$ , the process $\mathbb { G } _ { n } ^ { o }$ is sub-Gaussian for the $L ^ { 2 } ( \mathbb { P } _ { n } )$ -seminorm $\| \cdot \| _ { n }$ by Hoeffding’s inequality. Formally, for any $f , g \in { \mathcal { F } } _ { n }$

$$
\| \mathbf {G} _ {n} ^ {o} f - \mathbb {G} _ {n} ^ {o} g \| _ {\phi , n} \leq \| f - g \| _ {n} \equiv \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} | f (Z _ {i}) - g (Z _ {i}) | ^ {2}}.
$$

The value $\eta _ { n } = \operatorname* { s u p } _ { f \in { \mathcal { F } } _ { n } } \| f \| _ { n }$ is an upper bound for the radius of ${ \mathcal { F } } _ { n } \cup \{ 0 \}$ with respect to this norm. The maximal inequality Theorem 2.2.4 in van der Vaart and Wellner (1996) gives

$$
\begin{array}{l} \left\| \sup _ {f \in \mathcal {F} _ {n}} | \mathbb {G} _ {n} ^ {o} f | \right\| _ {\phi , n} \leq K _ {\phi} \int_ {0} ^ {\eta_ {n}} \sqrt {1 + \log N (\epsilon , \mathcal {F} _ {n} , L ^ {2} (\mathbb {P} _ {n}))} \mathrm{d} \epsilon \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

where $K _ { \phi }$ is a universal constant only depending on the function $\phi ;$ see Theorem 2.2.4 in van der Vaart and Wellner (1996) for more details. By Problem 2.2.5 in van der Vaart and Wellner (1996), we have

$$
\mathbb {E} \left[ \sup _ {f \in \mathcal {F} _ {n}} | \mathbb {G} _ {n} ^ {o} f | ^ {2} \Big | \{Z _ {i} \} _ {i = 1} ^ {n} \right] \leq 4 \log 2 \left\| \sup _ {f \in \mathcal {F} _ {n}} | \mathbb {G} _ {n} ^ {o} f | \right\| _ {\phi , n} ^ {2}.
$$

Consequently, applying Jensen’s inequality gives that there is a constant $K > 0$ not depending on n such that for all $n \in \mathbb { N } ^ { + }$

$$
\begin{array}{l} \mathbb {E} \left[ \sup _ {f \in \mathcal {F} _ {n}} | \mathbb {G} _ {n} ^ {o} f |   \Big | \{Z _ {i} \} _ {i = 1} ^ {n} \right] \leq K \| F \| _ {n} \int_ {0} ^ {1} \sqrt {1 + \log N (\epsilon \| F \| _ {n} , \mathcal {F} _ {n} , L ^ {2} (\mathbb {P} _ {n}))} \mathrm{d} \epsilon \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

where $K > 0$ is a constant independent of n. Taking expectation on both hand sides and applying Lemma 2.3.1 in van der Vaart and Wellner (1996) gives

$$
\begin{array}{l} \mathbb {E} \left[ \sup _ {\psi \in \mathcal {F} _ {n}} | (\mathbb {E} _ {n} - \mathbb {E})   \psi | \right] \leq 2 n ^ {- 1 / 2} \mathbb {E} \left[ \sup _ {f \in \mathcal {F} _ {n}} | \mathrm{G} _ {n} ^ {o} f | \right] \\ \quad \lesssim \sqrt {\mathrm{VC} (\Pi_ {n}) / n} \mathbb {E} \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} | F (Z _ {i}) | ^ {2}} \\ \quad \leq \| F \| _ {\mathbb {P}, 2} \sqrt {\mathrm{VC} (\Pi_ {n}) / n}, \end{array}
$$

where the last step follows from Jensen’s inequality. The desired result follows.

## Appendix G Proofs for the Results in Sections A and B

## G.1 Proof of Theorem A.1

Proof of Theorem A.1. By definition of $Y ( \pi ^ { \dagger } )$ , the worst-case welfare $W ( \pi ^ { \dagger } )$ can be decomposed as follows:

$$
\begin{array}{r l} & W (\pi^ {\dagger}) = \inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \left[ Y (\pi^ {\dagger}) \right] \\ & \quad = \inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} Y (1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} Y (0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} Y \right] \\ & \quad = \inf _ {Q \in \mathcal {P}} \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \mu_ {Q} (X, 1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \mu_ {Q} (X, 0) \right] + \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right], \end{array}
$$

where the third equality holds because the counterfactual outcome under the self-selection arm $\mathsf { S }$ is observed, whose distribution is identified from the data and thus invariant to $Q .$ . Applying similar arguments as the proof of Theorem 3.1, we have that

$$
\begin{array}{c} W (\pi^ {\dagger}) = \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right] \\ = \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \mu^ {-} (X, 1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \mu^ {-} (X, 0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right], \end{array}
$$

which completes the proof.

## G.2 Proof of Theorem A.2

ProofofTheorem A.2. Notice that $\Delta ( \pi ^ { \dagger } )$ can be decomposed as

$$
\begin{array}{l} \Delta (\pi^ {\dagger}) = \inf _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \Big [ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} Y (1) + \Big (\mathbb {1} \{\pi^ {\dagger} (X) = 0 \} - 1 \Big) Y (0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} Y \Big ] \\ = \inf _ {Q \in \mathcal {P}} \mathbb {E} \Big [ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \mu_ {Q} (X, 1) - \Big (1 - \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \Big) \mu_ {Q} (X, 0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \Big ]. \end{array}
$$

Applying similar arguments as the proof of Theorem 3.1, we obtain

$$
\begin{array}{l} \Delta (\pi^ {\dagger}) = \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \inf _ {Q \in \mathcal {P}} \mu_ {Q} (X, 1) - \Big (1 - \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \Big) \sup _ {Q \in \mathcal {P}} \mu_ {Q} (X, 0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right] \\ = \mathbb {E} \left[ \mathbb {1} \{\pi^ {\dagger} (X) = 1 \} \mu^ {-} (X, 1) + \mathbb {1} \{\pi^ {\dagger} (X) = 0 \} \mu^ {+} (X, 0) + \mathbb {1} \{\pi^ {\dagger} (X) = S \} \mathbb {E} [ Y | X ] \right] - \mathbb {E} [ \mu^ {+} (X, 0) ]. \end{array}
$$

This completes the proof.

## G.3 Proof of Corollary B.1

ProofofCorollary B.1. Consider the following derivation:

$$
\begin{array}{r l} & {\mathbb {E} _ {Q} \left[ Y (\pi (X)) - Y (\pi_ {0} (X)) \right]} \\ & {= \mathbb {E} _ {Q} \left[ (Y (1) - Y (0)) (\pi (X) - \pi_ {0} (X)) \right] = \mathbb {E} \left[ \tau_ {Q} (X) (\pi (X) - \pi_ {0} (X)) \right]} \\ & {= \mathbb {E} \left[ \tau_ {Q} (X) \max \{\pi (X) - \pi_ {0} (X), 0 \} \right] - \mathbb {E} \left[ \tau_ {Q} (X) \max \{\pi_ {0} (X) - \pi (X), 0 \} \right].} \end{array}
$$

Since the regions $\{ x : \pi ( x ) > \pi _ { 0 } ( x ) \}$ and $\{ x : \pi ( x ) < \pi _ { 0 } ( x ) \}$ are disjoint, after applying similar arguments as the proof of Theorem 3.1, we have

$$
\begin{array}{r l} & {\inf _ {Q \in \mathcal {P}} \left\{\mathbb {E} \left[ \tau_ {Q} (X) \max \{\pi (X) - \pi_ {0} (X), 0 \} \right] - \mathbb {E} \left[ \tau_ {Q} (X) \max \{\pi_ {0} (X) - \pi (X), 0 \} \right] \right\}} \\ & {= \inf _ {Q \in \mathcal {P}} \mathbb {E} \left[ \tau_ {Q} (X) \max \{\pi (X) - \pi_ {0} (X), 0 \} \right] - \sup _ {Q \in \mathcal {P}} \mathbb {E} \left[ \tau_ {Q} (X) \max \{\pi_ {0} (X) - \pi (X), 0 \} \right]} \\ & {= \mathbb {E} \left[ \tau^ {-} (X) \max \{\pi (X) - \pi_ {0} (X), 0 \} \right] - \mathbb {E} \left[ \tau^ {+} (X) \max \{\pi_ {0} (X) - \pi (X), 0 \} \right],} \end{array}
$$

which completes the proof.

## G.4 Proof of Theorem B.1

ProofofTheorem B.1. We start the proof with the following derivation:

$$
\begin{array}{l} \inf _ {\pi \in \Pi} \sup _ {Q \in \mathcal {P}} \operatorname{Reg} _ {Q} (\pi) = \inf _ {\pi \in \Pi} \sup _ {Q \in \mathcal {P}} \sup _ {\pi^ {\prime} \in \Pi} \mathbb {E} _ {Q} [ Y (\pi^ {\prime} (X)) ] - \mathbb {E} _ {Q} [ Y (\pi (X)) ] \\ = \inf _ {\pi \in \Pi} \sup _ {\pi^ {\prime} \in \Pi} \sup _ {Q \in \mathcal {P}} \mathbb {E} _ {Q} \left[ (Y (1) - Y (0)) (\pi^ {\prime} (X) - \pi (X)) \right] \\ = \inf _ {\pi \in \Pi} \sup _ {\pi^ {\prime} \in \Pi} \sup _ {Q \in \mathcal {P}} \mathbb {E} \left[ \tau_ {Q} (X) (\pi^ {\prime} (X) - \pi (X)) \right]. \end{array}
$$

Notice that

$$
\begin{array}{r l} & {\mathbb {E} \left[ \tau_ {Q} (X) \left(\pi^ {\prime} (X) - \pi (X)\right) \right]} \\ & {= \mathbb {E} \left[ \tau_ {Q} (X) \max \left\{\pi^ {\prime} (X) - \pi (X), 0 \right\} \right] - \mathbb {E} \left[ \tau_ {Q} (X) \max \left\{\pi (X) - \pi^ {\prime} (X), 0 \right\} \right],} \end{array}
$$

and the fact that $\{ x : \pi ( x ) > \pi ^ { \prime } ( x ) \}$ and $\{ x : \pi ( x ) < \pi ^ { \prime } ( x ) \}$ are disjoint. Applying similar arguments as the proof of Theorem 3.1, we have

$$
\begin{array}{l} \sup _ {Q \in \mathcal {P}} \mathbb {E} \left[ \tau_ {Q} (X) \left(\pi^ {\prime} (X) - \pi (X)\right) \right] \\ = \mathbb {E} \left[ \tau^ {+} (X) \max \left\{\pi^ {\prime} (X) - \pi (X), 0 \right\} \right] - \mathbb {E} \left[ \tau^ {-} (X) \max \left\{\pi (X) - \pi^ {\prime} (X), 0 \right\} \right], \end{array}
$$

which completes the proof of Eq. (B.3).

Then we turn to derive the first-best policy. When $\Pi = \Pi _ { o }$ consists of all measurable policies, it suffices to consider the following optimizations for any $x \in \mathcal { X }$

$$
\sup _ {\pi^ {\prime} (x) \in [ 0, 1 ]} \tau^ {+} (x) \max \left\{\pi^ {\prime} (x) - \pi (x), 0 \right\} - \tau^ {-} (x) \max \left\{\pi (x) - \pi^ {\prime} (x), 0 \right\}.\tag{G.1}
$$

It is straightforward to show that

$$
\tilde {\pi} ^ {\prime} (x) = \mathbb {1} \left\{\left(1 - \pi (x)\right) \tau^ {+} (x) + \pi (x) \tau^ {-} (x) > 0 \right\}
$$

is an optimal solution to Eq. (G.1). Substituting $\tilde { \pi } ^ { \prime } ( \cdot )$ into Eq. (G.1), we find that the first-best policy minimizing

$$
\tau^ {+} (x) \max \left\{\tilde {\pi} ^ {\prime} (x) - \pi (x), 0 \right\} - \tau^ {-} (x) \max \left\{\pi (x) - \tilde {\pi} ^ {\prime} (x), 0 \right\}
$$

is given by

$$
\pi_ {R} ^ {\star} (x) = \left\{ \begin{array}{l l} 1, & \text { if } \tau^ {-} (x) \geq 0, \\ \tau^ {+} (x) / (\tau^ {+} (x) - \tau^ {-} (x)), & \text { if } \tau^ {-} (x) <   0 <   \tau^ {+} (x). \\ 0, & \text { if } \tau^ {+} (x) \leq 0, \end{array} \right.
$$

# Appendix H Proofs for the Identification under Multi-Valued Treatment

Before presenting the formal proofs of Proposition E.1 and Theorem E.1, we first establish a more general result that extends Proposition 1B of Dorn and Guo (2023) to the multi-valued treatment setting. As shown below, Proposition E.1 follows as a direct corollary of Lemma H.1.

Lemma H.1. For any $a \in { \mathcal { A } }$ and any random variable $E \in ( 0 , 1 )$ satisfying

$$
\mathbb {E} \left[ \mathbb {1} \{A = a \} / E | X = x \right] = 1,
$$

and

$$
\frac {\left(e _ {a} (X) + \left[ 1 - e _ {a} (X) \right] / \Lambda\right) \mathbb {1} \{A = a \}}{e _ {a} (X)} \leq \frac {\mathbb {1} \{A = a \}}{E} \leq \frac {\left(e _ {a} (X) + \left[ 1 - e _ {a} (X) \right] \Lambda\right) \mathbb {1} \{A = a \}}{e _ {a} (X)},\tag{H.1}
$$

we can construct random variables $( X , Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \dots , Y ( a _ { d } ) , A , U )$ defined on the same probability space as $\left( X , Y , A , E \right)$ and an associated putative propensity score $e _ { a } ( x , u ) = \mathbb { P } [ A = a | X =$ $x , U = u ]$ such that

$$
Y = \sum_ {j = 1} ^ {d} \mathbb {1} \{A = a _ {j} \} Y (a _ {j}). \tag {1}
$$

(2) (Y(a<sub>1</sub>), Y(a<sub>2</sub>), . . . , Y(a )) = A | (X, U) and

$$
e _ {a} (X) / \left(e _ {a} (X) + [ 1 - e _ {a} (X) ] \Lambda\right) \leq e _ {a} (X, U) \leq e _ {a} (X) / \left(e _ {a} (X) + [ 1 - e _ {a} (X) ] / \Lambda\right).
$$

$$
(3) \mathbb {1} \{A = a \} / e _ {a} (X, U) = \mathbb {1} \{A = a \} / E.
$$

Proof of Lemma H.1. Define the following conditional distribution functions:

$$
\begin{array}{c} F (y | x, a) = \mathbb {P} (Y \leq y | X = x, A = a), \\ G (y | x, a, e) = \mathbb {P} (Y \leq y | X = x, A = a, E = e), \\ H (e | x, a) = \mathbb {P} (E \leq e | X = x, A = a), \\ K _ {a} (u | x) = \int_ {- \infty} ^ {u} \frac {e _ {a} (x)}{1 - e _ {a} (x)} \frac {1 - e}{e} \mathrm{d} H (e | x, a). \end{array}
$$

Given E $[ \mathbb { 1 } \left\{ A = a \right\} / E \mid X = x ] = 1 , K _ { a } ( u | x )$ is a valid CDF for any $x \in \mathcal { X }$ and $a \in A .$ . For simplicity, we provide the construction for $a = a _ { 1 } ;$ ; identical arguments apply to other treatments. Let $V , V _ { a _ { 1 } } , V _ { a _ { 2 } } , \ldots , V _ { a _ { d } }$ be i.i.d. Uniform $[ 0 , 1 ]$ random variables, independent of $\left( X , Y , A , E \right)$ . We construct $Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \dots , Y ( a _ { d } )$ and U as:

$$
Y (a _ {1}) = \mathbb {1} \{A = a _ {1} \} Y + \mathbb {1} \{A \neq a _ {1} \} G ^ {- 1} (V _ {a _ {1}} | X, a _ {1}, U),
$$

$$
\begin{array}{c} Y (a _ {j}) = \mathbb {1} \{A \neq a _ {j} \} F ^ {- 1} (V _ {a _ {j}} | X, a _ {j}) + \mathbb {1} \{A = a _ {j} \} Y \quad \text {for} \quad j \in \{2, 3, \ldots , d \}, \\ U = \mathbb {1} \{A = a _ {1} \} E + \mathbb {1} \{A \neq a _ {1} \} K _ {a _ {1}} ^ {- 1} (V | X). \end{array}
$$

We verify that the constructed random variables satisfy the three required properties. Property (1) follows directly from the construction. We now proceed to examine the remaining properties. To verify property (2), we consider the following derivation. Conditional on $X , U , A = a _ { 1 }$ , we have

$$
\begin{array}{l} \mathbb {P} (Y (a _ {1}) \leq y _ {1}, Y (a _ {2}) \leq y _ {2}, \ldots , Y (a _ {d}) \leq y _ {d} | X, U, A = a _ {1}) \\ = \mathbb {P} \left(Y \leq y _ {1}, F ^ {- 1} (V _ {a _ {2}} | X, a _ {2}) \leq y _ {2}, \ldots , F ^ {- 1} (V _ {a _ {d}} | X, a _ {d}) \leq y _ {d} | X, U, A = a _ {1}\right) \\ = \mathbb {P} \left(Y \leq y _ {1} | X, U, A = a _ {1}\right) \prod_ {j = 2} ^ {d} \mathbb {P} \left(F ^ {- 1} (V _ {a _ {j}} | X, a _ {j}) \leq y _ {j} | X, U, A = a _ {1}\right) \\ = G (y _ {1} | X, a _ {1}, U) \prod_ {j = 2} ^ {d} F (y _ {j} | X, a _ {j}). \end{array}
$$

Conditional on $\quad X , U , A = a _ { i } \operatorname { f o r } j \neq 1$ , we have

$$
\begin{array}{l} \mathbb {P} (Y (a _ {1}) \leq y _ {1}, Y (a _ {2}) \leq y _ {2}, \ldots , Y (a _ {d}) \leq y _ {d} | X, U, A = a _ {j}) \\ = \mathbb {P} \left(G ^ {- 1} (V _ {a _ {1}} | X, a _ {1}, U) \leq y _ {1}, F ^ {- 1} (V _ {a _ {2}} | X, a _ {2}) \leq y _ {2}, \ldots , Y \leq y _ {j}, \ldots , F ^ {- 1} (V _ {a _ {d}} | X, a _ {d}) \leq y _ {d} | X, U, A = a _ {j}\right) \\ = \mathbb {P} \left(G ^ {- 1} (V _ {a _ {1}} | X, a _ {1}, U) \leq y _ {1} | X, U, A = a _ {j}\right) \prod_ {\ell = 2, \ell \neq j} ^ {d} \mathbb {P} \left(F ^ {- 1} (V _ {a _ {\ell}} | X, a _ {\ell}) \leq y _ {\ell} | X, U, A = a _ {\ell}\right) \\ \quad \times \mathbb {P} \left(Y \leq y _ {j} | X, U, A = a _ {j}\right) \\ = G (y _ {1} | X, a _ {1}, U) \prod_ {\ell = 2, \ell \neq j} ^ {d} F (y _ {\ell} | X, a _ {\ell}) \times \mathbb {P} \left(Y \leq y _ {j} | X, K ^ {- 1} (V | X), A = a _ {j}\right) \\ = G (y _ {1} | X, a _ {1}, U) \prod_ {j = 2} ^ {d} F (y _ {j} | X, a _ {j}), \end{array}
$$

Combining the two expressions above gives $( Y ( a _ { 1 } ) , Y ( a _ { 2 } ) , \ldots , Y ( a _ { d } ) ) \bot \bot A \mid ( X , U )$ . By Bayes’ rule,

$$
\begin{array}{l} e _ {a _ {1}} (x, u) = e _ {a _ {1}} (x) \frac {\mathrm{d} \mathbb {P} (u | X = x , A = a _ {1})}{\mathrm{d} \mathbb {P} (u | X = x)} \\ \qquad = e _ {a _ {1}} (x) \frac {\mathrm{d} \mathbb {P} (u | X = x , A = a _ {1}) / \mathrm{d} H (u | x , a _ {1})}{\mathrm{d} \mathbb {P} (u | X = x) / \mathrm{d} H (u | x , a _ {1})} \\ \qquad = \frac {e _ {a _ {1}} (x)}{e _ {a _ {1}} (x) + \frac {e _ {a _ {1}} (x)}{1 - e _ {a _ {1}} (x)} \frac {1 - u}{u} (1 - e _ {a _ {1}} (x))} \\ \qquad = u. \end{array}
$$

Since the support of $K _ { a _ { 1 } } ( \cdot | x )$ is contained in that of $H ( \cdot | x , a _ { 1 } )$ , Eq. (H.1) implies

$$
e _ {a _ {1}} (X) / \left(e _ {a _ {1}} (X) + \left[ 1 - e _ {a _ {1}} (X) \right] \Lambda\right) \leq U \leq e _ {a _ {1}} (X) / \left(e _ {a _ {1}} (X) + \left[ 1 - e _ {a _ {1}} (X) \right] / \Lambda\right) \quad \text { a   .   s   . }
$$

Finally, property (3) is also easy to verify. The event $\mathbb { 1 } \{ A = a _ { 1 } \}$ implies $U = E ,$ , so

$$
\mathbb {1} \left\{A = a _ {1} \right\} / e _ {a _ {1}} (X, U) = \mathbb {1} \left\{A = a _ {1} \right\} / U = \mathbb {1} \left\{A = a _ {1} \right\} / E.
$$

Proof of Proposition E.1. The proof follows directly from that of Lemma H.1, using a similar argument as in Proposition D.1, and is therefore omitted for brevity. □

We now turn to proof Theorem E.1.

Proof of Theorem E.1. According to Proposition E.1, for any $Q \in \mathcal { P } _ { \mathrm { M } }$ and $( x , a ) \in \mathcal { X } \times \mathcal { A }$ , we have

$$
\mu_ {Q} (x, a) \geq \inf _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mu_ {Q} (x, a) = \mu^ {-} (x, a).
$$

It follows that

$$
\mathbb {E} \left[ \sum_ {a \in \mathcal {A}} \mu_ {Q} (X, a) \pi_ {a} (X) \right] \geq \mathbb {E} \left[ \sum_ {a \in \mathcal {A}} \mu^ {-} (X, a) \pi_ {a} (X) \right]
$$

Since $Q \in \mathcal { P } _ { \mathrm { M } }$ is arbitrary, the inequality remains valid after taking the infimum over $Q \in \mathcal { P } _ { \mathrm { M } }$ on both sides. Therefore, we conclude that

$$
\begin{array}{l} W (\pi) = \inf _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mathbb {E} _ {Q} \left[ \sum_ {a \in \mathcal {A}} Y (a) \pi_ {a} (X) \right] = \inf _ {Q \in \mathcal {P} _ {\mathrm{M}}} \mathbb {E} \left[ \sum_ {a \in \mathcal {A}} \mu_ {Q} (X, a) \pi_ {a} (X) \right] \\ \geq \mathbb {E} \left[ \sum_ {a \in \mathcal {A}} \mu^ {-} (X, a) \pi_ {a} (X) \right]. \end{array}
$$
