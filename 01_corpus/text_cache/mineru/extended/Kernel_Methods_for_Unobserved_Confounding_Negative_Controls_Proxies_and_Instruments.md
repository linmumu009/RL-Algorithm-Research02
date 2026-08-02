# Kernel Methods for Unobserved Confounding: Negative Controls, Proxies, and Instruments

Rahul Singh<sup>∗</sup> MIT Economics

Original draft: December 18, 2020. This draft: March 23, 2023.

## Abstract

Negative control is a strategy for learning the causal relationship between treatment and outcome in the presence of unmeasured confounding. The treatment efect can nonetheless be identified if two auxiliary variables are available: a negative control treatment (which has no efect on the actual outcome), and a negative control out come (which is not afected by the actual treatment). These auxiliary variables can also be viewed as proxies for a traditional set of control variables, and they bear resemblance to instrumental variables. I propose a family of algorithms based on kernel ridge regression for learning nonparametric treatment efects with negative controls. Examples include dose response curves, dose response curves with distribution shift, and heterogeneous treatment efects. Data may be discrete or continuous, and low, high, or infinite dimensional. I prove uniform consistency and provide finite sample rates of convergence. I estimate the dose response curve of cigarette smoking on infant birth weight adjusting for unobserved confounding due to household income, using a data set of singleton births in the state of Pennsylvania between 1989 and 1991.

Keywords: potential outcome, reproducing kernel Hilbert space, dose response

## 1 Introduction

Selection on observables is the popular assumption in causal inference that the assignment of treatment D is as good as random after conditioning on covariates X. It is a strong causal assumption which is often violated even in laboratory settings. Negative controls, widely used in laboratory science, guard against unobserved confounding. The idea is to check for spurious relationships that would only be nonzero in the presence of an unobserved confounder U–an approach sometimes called falsification or specificity testing. Consider two auxiliary variables: a negative control treatment Z (which a priori has no efect on the actual outcome Y ), and a negative control outcome W (which a priori is not afected by the actual treatment D). [Miao and Tchetgen Tchetgen, 2018] and [Deaner, 2018] carefully formalize a learning problem in which negative controls (Z, W) can not only check for the presence of unobserved confounding U, but also recover the causal relationship of interest.

As a concrete example, consider the empirical strategy of [Lousdal et al., 2020]. The goal is to measure the efect of mammography screening D on death from breast cancer Y. The set of covariates X includes marriage status, number of children, age at first birth, years of education, annual income, and hormone drug use. The authors used negative controls to document that, even after taking into account the covariates X, unobserved confounding U drives spurious correlations. Specifically, dental care participation Z decreases the likelihood of death from breast cancer Y in the dataset. Mammography screening D decreases the likelihood of death from other causes W in the dataset. The authors conclude that unobserved confounding contaminates treatment efect estimation in this setting.

In the present work, I propose a family of nonparametric algorithms based on kernel ridge regression that use negative controls to not only detect but also adjust for unobserved confounding. I consider treatment efects of the population, of subpopulations, and of alternative populations with alternative covariate distributions. Moreover, I allow for treatments, covariates, and negative controls that may be discrete or continuous, and low, high, or infinite dimensional. Due to the intuitive nature of negative controls, I use such terminology throughout the paper. In recent work, [Tchetgen Tchetgen et al., 2020] refer to negative controls as proxy variables in order to emphasize that they may arise in not only experimental but also observational settings. Due to the formal resemblance between negative controls and instrumental variables, the new statistical results I provide also apply to nonparametric instrumental variable regression (NPIV). Altogether, I provide conceptual, algorithmic, and statistical contributions.

Conceptual. I unify a variety of learning problems with unobserved confounding into one general nonparametric learning problem. In semiparametric causal inference, treatment D is restricted to be binary. I consider nonparametric causal inference, allowing the treatment D to be binary, discrete, or continuous. It appears that this is the first work on conditional dose response curves and heterogeneous treatment efects using negative controls. See Section 2 for a discussion of related work on related estimands, e.g. [Mastouri et al., 2021, Kallus et al., 2021, Ghassami et al., 2021]. I provide a template for future epidemiology research to estimate dose response curves and heterogeneous treatment efects from medical records despite unobserved confounding.

Algorithmic. I propose a family of novel estimators with closed form solutions that are straightforward to implement by matrix operations. To do so, I assume that the true causal relationship is a function in a reproducing kernel Hilbert space (RKHS), which is a popular nonparametric setting in machine learning. The hyperparameters are ridge regression penalties and kernel hyperparameters. For the former, I derive the closed form solution for leave-one-out cross validation. The latter have well known heuristics. I evaluate the estimators in simulations against alternative estimators that ignore unobserved confounding.

Statistical. I prove uniform (sup norm) consistency with finite sample rates. A uniform guarantee encodes caution about worst case scenarios when informing policy decisions. The finite sample rates of convergence do not directly depend on the data dimension but rather the smoothness of the true causal relationship. An important intermediate result is finite sample analysis of NPIV in sup norm. Of independent interest, I relate assumptions required in ill posed inverse problems–existence and completeness–to the RKHS setting. This characterization appears to be absent from previous work on NPIV in the RKHS.

To illustrate how the proposed estimators are useful, I conduct a case study. Estimating the efect of cigarette smoking on infant birth weight is challenging for several reasons. First, pregnant women are classified as a vulnerable population, so they are typically excluded from clinical trials; observational data are the only option. Second, pregnancy induces many physiological changes, so medical knowledge predicts diferent dose response curves for women who are pregnant compared to women who are not pregnant. Third, medical records exclude an unobserved confounder known to be crucial for maternal-fetal health: household income. I argue that medical records include variables that satisfy the properties of negative controls for unobserved income, and discuss what issues may arise if there are additional unobserved confounders. I provide preliminary results and outline directions for future work on this important topic.

The structure of the paper is as follows. Section 2 describes related work. Section 3 formalizes the learning problem. Section 4 proposes the new algorithms. Section 5 proves uniform consistency. Section 6 conducts simulation experiments and estimates the dose response curve of cigarette smoking on infant birth weight, adjusting for unobserved confounding due to household income. Section 7 concludes.

## 2 Related work

I view dose response curves and heterogeneous treatment efects as reweightings of a structural function defined by an ill posed inverse problem. As such, I extend the partial means framework [Newey, 1994]. Existing work on partial means considers consumer surplus [Newey, 1994] and certain causal parameters [Singh et al., 2020] to be reweightings of a regression function. By contrast, I consider causal parameters that are reweightings of a structural function. My uniform analysis therefore generalizes uniform analysis in previous work. To express causal parameters in this way, I generalize identification theorems for treatment efects that use negative controls [Miao et al., 2018, Miao and Tchetgen Tchetgen, 2018, Deaner, 2018, Tchetgen Tchetgen et al., 2020].

Early work on negative controls emphasized their role in detection of unobserved confounding. As early as the 1950s, epidemiologists proposed the principle of causal specificity as a diagnostic tool [Berkson, 1958, Yerushalmy and Palmer, 1959, Hill, 1965]. Subsequent work formalized these concepts [Rosenbaum, 1989, Weiss, 2002, Lipsitch et al., 2010]. A more recent literature emphasizes the role of negative controls in adjustment for unobserved confounding. Many papers eliminate the bias from unobserved confounding by imposing additional structure: linearity and normality [Gagnon-Bartsch and Speed, 2012,

Wang et al., 2017]; joint normality [Kuroki and Pearl, 2014]; rank preservation of individual potential outcomes [Tchetgen Tchetgen, 2014]; or monotonicity of confounding efects [Sofer et al., 2016]. I generalize identification results that relax such additional structure.

In econometrics, closely related strategies adjust for unobserved confounding in dynamic settings: diference-in-diference [Card, 1990, Meyer, 1995, Abadie, 2005], and panel proxy control [Deaner, 2018]. Traditional diference-in-diference analysis requires strong assumptions such as linearity and additive separability of confounding. [Athey and Imbens, 2006] present a more general approach, called changes-in-changes, articulated in terms of a nonseparable, nonlinear structural model. A key assumption is monotonicity of confounding efects. Importantly, the model I present allows nonlinearity without requiring monotonicity of confounding efects. The panel proxy control approach is also articulated in terms of a nonseparable, nonlinear structural model, and its static special case closely resembles the negative control model. [Deaner, 2018] presents a series estimator as well as innovative strategies to handle ill posedness and completeness for both static and dynamic settings. It is straightforward to use the techniques developed in this paper to derive an RKHS estimator for the panel proxy setting. See [Sofer et al., 2016] and [Deaner, 2018] for explicit comparisons of negative control with diference-in-diference and panel proxy control, respectively.

As previewed above, the causal parameters studied in this work are reweightings of a structural function called a confounding bridge, which closely resembles a nonparametric instrumental variable regression (NPIV). NPIV has a rich literature, including the seminal works of [Newey and Powell, 2003, Hall and Horowitz, 2005, Blundell et al., 2007, Darolles et al., 2011, Chen and Reiss, 2011, Chen and Pouzo, 2012], among others. The kernel ridge regression approach in this work employs RKHS-norm Tikhonov regularization over an infinite dimensional RKHS with a low efective dimension. See e.g. [Darolles et al., 2011, Hall and Horowitz, 2005, Horowitz and Lee, 2005, Carrasco et al., 2007, Chen and Pouzo, 2012], and references therein, for a rich variety NPIV estimators that employ various types of Tikhonov regularizations over various infinite dimensional function spaces. In Appendix D, I compare my approximation assumptions to the approximation assumptions employed in this literature, building on the discussion of [Chen and Reiss, 2011].

I contribute to a growing literature that adapts RKHS methods to treatment efect estimation. [Nie and Wager, 2021] propose an RKHS estimator of heterogeneous treatment efects under selection on observables and prove mean square error rates. I pursue a more general definition of heterogeneous treatment efects conditional on some interpretable subvector $V \subset X$ [Abrevaya et al., 2015] and allow for unobserved confounding. [Singh et al., 2019] present an RKHS approach for nonparametric instrumental variable regression and prove projected mean square error rates in the sense of [Ai and Chen, 2003]. [Singh et al., 2020] present an RKHS approach for treatment efects identified by selection on observables and prove uniform rates. I unify the RKHS constructions in both works in order to handle both ill posedness and reweighting. My work is complementary in that I consider a new causal setting. My uniform analysis applies to not only negative control treatment efects but also nonparametric instrumental variable regression, providing alternative results under the same assumptions as [Singh et al., 2019]. I build on fundamental statistical contributions from [Smale and Zhou, 2005, Smale and Zhou, 2007, Fischer and Steinwart, 2020].

This draft subsumes [Singh, 2020]. Several other works have proposed alternative RKHS estimators for the negative control setting. Independently and contemporaneously to [Singh, 2020], [Mastouri et al., 2021] propose estimators for the dose response curve. [Mastouri et al., 2021] formulate a kernel two stage regression approach and a kernel moment restriction approach, and formalize connections between them. [Mastouri et al., 2021] analyze excess risk of a surrogate loss for the former, and consistency for the latter. Excess risk of a surrogate loss corresponds to projected mean square error for the confounding bridge. See Sections 4 and 5 for further comparisons. [Kallus et al., 2021, Ghassami et al., 2021 study the semiparametric problem rather than the nonparametric problem considered here. Both works propose doubly robust estimators that combine nuisance functions estimated by a minimax procedure. [Chernozhukov et al., 2021] provide abstract conditions to translate learning theory rates into semiparametric inference when treatment is binary. I summarize the connection to semiparametrics in Appendix A.

The emphasis of this work is uniform consistency for the nonparametric case. The main theoretical results of this paper are (i) uniform consistency of the confounding bridge and (ii) uniform consistency of causal functions estimated using a kernel two stage regression approach. Uniform nonparametric inference remains an open question for future research.

## 3 Learning problem

## 3.1 Treatment efects

Treatment efects are statements about counterfactual outcomes given hypothetical interventions. Though we observe outcome Y, we seek to infer means of counterfactual outcomes $\{ Y ^ { ( d ) } \}$ , where $Y ^ { ( d ) }$ is the potential outcome given the hypothetical intervention $D = d .$ The treatment efect literature aims to measure a rich variety of treatment efects, which I quote from [Singh et al., 2020, Definition 3.1].

Definition 1 (Treatment efects). I define the following treatment efects.

1. Dose response: $\theta _ { 0 } ^ { A T E } ( d ) : = \mathbb { E } [ Y ^ { ( d ) } ]$ is the counterfactual mean outcome given intervention $D = d$ for the entire population.

2. Dose response with distribution shift: $\theta _ { 0 } ^ { D S } ( d , \tilde { \mathbb { P } } ) : = \mathbb { E } _ { \tilde { \mathbb { P } } } [ Y ^ { ( d ) } ]$ is the counterfactual mean outcome given intervention $D = d$ for an alternative population with data distribution $\tilde { \mathbb { P } }$ (elaborated in Assumption 3).

3. Conditional dose response: $\theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } ) : = \mathbb { E } [ Y ^ { ( d ^ { \prime } ) } | D = d ]$ is the counterfactual mean outcome given intervention $D = d ^ { \prime }$ for the subpopulation who actually received treatment $D = d$

4. Heterogeneous treatment efect: $\theta _ { 0 } ^ { C A T E } ( d , v ) : = \mathbb { E } [ Y ^ { ( d ) } | V = v ]$ is the counterfactual mean outcome given intervention $D = d$ for the subpopulation with covariate value $V = v$

The superscipt of each nonparametic treatment efect corresponds to its semiparametric analogue. If treatment is binary, then average treatment efect (ATE) is $\mathbb { E } [ Y ^ { ( 1 ) } - Y ^ { ( 0 ) } ]$ average treatment efect with distribution shift (DS) is $\mathbb { E } _ { \tilde { \mathbb { P } } } [ Y ^ { ( 1 ) } - Y ^ { ( 0 ) } ]$ ; average treatment on the treated (ATT) is $\mathbb { E } [ Y ^ { ( 1 ) } - Y ^ { ( 0 ) } | D = 1 ]$ ; and conditional average treatment efect (CATE) is $\mathbb { E } [ Y ^ { ( 1 ) } - Y ^ { ( 0 ) } | V = v ]$ . Rather than diferences of potential outcomes indexed by binary treatment, I analyze potential outcomes indexed by discrete or continuous treatment.

$\theta _ { 0 } ^ { A T E } ( d )$ has many names: dose response curve, continuous treatment efect, and average structural function. If treatment is binary, then $\theta _ { 0 } ^ { A T E } ( d )$ is a vector in $\mathbb { R } ^ { 2 }$ and the learning problem is semiparametric. If the treatment is discrete or continuous, then $\theta _ { 0 } ^ { A T E } ( d )$ is a function and the learning problem is nonparametric. $\theta _ { 0 } ^ { D S } ( d , \tilde { \mathbb { P } } )$ is a closely related variant that handles the scenario where the covariate distribution has shifted. This variant may be called distribution shift, covariate shift, policy efect, or transfer learning.

Both $\theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } )$ and $\theta _ { 0 } ^ { C A T E } ( d , v )$ involve conditioning on a particular subpopulation. If treatment is binary, then $\theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } )$ is a matrix in $\mathbb { R } ^ { 2 \times 2 }$ and the learning problem is semiparametric. If the treatment is discrete or continuous, then $\theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } )$ is a surface and the learning problem is nonparametric. Likewise for $\theta _ { 0 } ^ { C A T E } ( d , v ) . \ \theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } )$ is called the conditional dose response, and $\theta _ { 0 } ^ { C A T E } ( d , v )$ is called the heterogeneous treatment efect. The possibility for D to be discrete or continuous and for V to be a particular covariate, rather than the full set of covariates required for identification, is more general than the typical heterogeneous treatment efect [Nie and Wager, 2021]. For $\theta _ { 0 } ^ { C A T E }$ , I slightly abuse notation by denoting the complete set of identifying covariates as $( V , X )$

## 3.2 Negative control identification

In pioneering work, [Tchetgen Tchetgen et al., 2020] propose a potential outcome model in which treatment efects can be measured from outcomes Y , treatments $D ,$ and covariates $( V , X )$ despite unobserved confounding U. The technique involves two auxiliary variables: negative control treatment $Z _ { i }$ and negative control outcome W. In this model, potential outcomes $\{ Y ^ { ( d , z ) } \}$ and potential negative control outcomes $\{ W ^ { ( d , z ) } \}$ are initially indexed by both the treatment value $D = d$ and the negative control treatment value $Z = z$ . The identification strategy requires prior knowledge of how the unobserved confounder, which may be a vector, relates to the observed variables. The validity of negative controls as articulated in Assumptions 1 and 2 is relative to a conjectured unobserved confounder.

## Assumption 1 (Negative controls). Assume

1. No interference: if $D = d$ and $Z = z$ then $Y = Y ^ { ( d , z ) }$ and $W = W ^ { ( d , z ) }$

2. Latent exchangeability: $\{ Y ^ { ( d , z ) } \} , \{ W ^ { ( d , z ) } \} \bot D , Z | U , X .$

3. Overlap: if $f ( u , x ) ~ > ~ 0$ then $f ( d , z | u , x ) \ > \ 0$ , where $f ( u , x )$ and $f ( d , z | u , x )$ are densities.

4. Negative control treatment and outcome: $Y ^ { ( d , z ) } = Y ^ { ( d ) }$ and $W ^ { ( d , z ) } = W$ For $\theta _ { 0 } ^ { C A T E }$ , replace X with $( V , X )$

No interference is also called consistency or the stable unit treatment value assumption in causal inference, and it rules out network efects. Latent exchangeability states that conditional on covariates X and unobserved confounder U, treatment assignment and negative control treatment assignment are as good as random. Latent exchangeability relaxes the classic assumption of conditional exchangeability in which U = ∅, i.e. in which there is no unobserved confounder. Overlap ensures that there is no confounder-covariate stratum such that treatment and negative control treatment have a restricted support; for any stratum, any value of treatment or negative control treatment can occur.

In a graphical causal model, there could be many sets of observed variables that could serve as covariates X and many sets of unobserved variables that could serve as the unobserved confounder U based on these initial criteria. The subsequent criteria provide guidance in how to choose (X, U). We will see that the set of covariates X should be chosen to block as much unobserved confounding as possible, because the variation in unobserved confounding that remains must be tied to variation in negative controls.

The negative control treatment condition imposes that the negative control treatment Z only afects the outcome Y via actual treatment D. It is identical to the exclusion restriction assumed for instrumental variables [Angrist et al., 1996]. The negative control outcome condition imposes that the negative control outcome W is unafected by the treatment D and negative control treatment Z. It is an even stronger exclusion restriction. Altogether, Assumption 1 formalizes the intuition that if there are spurious correlations then there is unobserved confounding. It also implies $Y \bot \bot | D , U , X$ and $W \bot \bot D , Z | U , X$ , which are weaker conditions used in the identification argument [Miao et al., 2018].

Figure 1 visualizes a representative directed acyclic graph (DAG). Despite access to covariates X, unobserved confounding U has unblocked paths to treatment D and outcome Y. In the DAG, we also see the proxy interpretation of this learning problem. Covariates X, negative control treatment Z, and negative control outcome W are all imperfect proxies for a set of control variables that would block unobserved confounding. Covariates X are proxies that induce treatment and outcome; negative control treatment Z is a proxy that induces treatment only; and negative control outcome W is a proxy that induces outcome only.

Next, I quote a high level technical condition, which I will later verify for the RKHS setting. Define the regression $\gamma _ { 0 } ( d , x , z ) : = \mathbb { E } [ Y | D = d , X = x , Z = z ] .$

## Assumption 2 (Confounding bridge). Assume

1. Existence: there exists a solution $h _ { 0 }$ to the operator equation


$$
\gamma_ {0} (d, x, z) = \mathbb {E} [ h (D, X, W) | D = d, X = x, Z = z ].
$$

2. Completeness: for any function $f ,$

Figure 1: Negative control DAG

$$
\mathbb {E} [ f (U) | D = d, X = x, Z = z ] = 0 \quad \forall (d, x, z) \iff f (U) = 0.
$$

I call $h _ { 0 }$ the confounding bridge, following [Miao and Tchetgen Tchetgen, 2018]. Here, we see the formal resemblance to the nonparametric instrumental variable regression problem (NPIV) [Newey and Powell, 2003]. In the language of NPIV, the LHS $\gamma _ { 0 } ( d , x , z )$ is the reduced form, while the RHS is a composition of a stage 1 conditional expectation operator and stage 2 structural function $h _ { 0 }$ . In the language of functional analysis, the operator equation is a Fredholm integral equation of the first kind. Solving this operator equation for $h _ { 0 }$ involves inverting a linear operator with infinite dimensional domain; it is an ill posed problem. Indeed, existence will require conditions on the spectrum of the conditional expectation operator formalized in Appendix B. Completeness is a technical condition from the NPIV literature. Taking $f = f _ { 1 } - f _ { 2 }$ , it states that the observed variables $( D , X , Z )$ have suficiently rich variation in the sense that diferent functions of unobserved confounding $f _ { 1 } ( U )$ and $f _ { 2 } ( U )$ lead to diferent projections onto $( D , X , Z )$ ; if they lead to the same projections, then $f _ { 1 } = f _ { 2 }$

There is a subtle yet fundamental connection between the existence of the confounding bridge and the relevance of negative controls.

Proposition 1 (Relevance). Suppose Assumption 1 holds and $\gamma _ { 0 }$ varies in z, i.e. there exist $( z , z ^ { \prime } )$ such that $\gamma _ { 0 } ( d , x , z ) \neq \gamma _ { 0 } ( d , x , z ^ { \prime } )$

1. If the negative control treatment is irrelevant to the unobserved confounder in the sense that $Z \bot \bot U | D , X$ , then no confounding bridge exists.

2. If the negative control outcome is irrelevant to the unobserved confounder in the sense that $W { \underline { { \parallel U } } } | X$ , then no confounding bridge exists.

See Appendix B for the proof, as well as further discussion of the cases in which $( U , Z , W )$ are discrete or continuous. Assumption 2 formalizes the converse intuition of Assumption 1: if there is unobserved confounding, then there are spurious correlations. In order to use (Z, W) to adjust for unobserved confounding U, it must be the case that (Z, W) can detect U well enough. In practice, an analyst must collaborate with domain experts in order to assess (i) what are the sources of unobserved confounding, (ii) whether the negative control exclusion restrictions hold, and (iii) whether the negative controls are relevant. In summary, the key assumptions of negative control identification apply to settings where there are spurious correlations if and only if there is unobserved confounding. When domain knowledge is insuficient to verify these assumptions, then the strategy of negative controls is inappropriate.

To handle $\theta _ { 0 } ^ { D S }$ , I generalize a standard assumption in transfer learning.

## Assumption 3 (Distribution shift). Assume

1. The diference in population distributions P and $\tilde { \mathbb { P } }$ is only in the marginal distribution of treatments, negative control treatments, and covariates:

$$
\tilde {\mathbb {P}} (Y, W, D, X, Z) = \mathbb {P} (Y, W | D, X, Z) \tilde {\mathbb {P}} (D, X, Z).
$$

2. P<sup>˜</sup> $( D , X , Z )$ is absolutely continuous with respect to $\mathbb { P } ( D , X , Z )$

Proposition 2 (Invariance of confounding bridge). Under Assumptions 2 and $^ { 3 , }$ the confounding bridge $h _ { 0 }$ remains the same across the diferent populations P and $\tilde { \mathbb { P } }$ .

See Appendix C for the proof. It appears that Assumption 3 and Proposition 2 are the first formalization of distribution shift in negative control and NPIV settings.

In summary, I place three assumptions: availability of negative controls (Assumption 1); existence and completeness of the confounding bridge (Assumption 2); and invariance of the confounding bridge for transfer learning (Assumption 3). Formally, the theorem that uses these assumptions to express treatment efects in terms of data is known as an identification result. I present the main identification result below, extending the powerful insights of [Miao et al., 2018, Miao and Tchetgen Tchetgen, 2018, Deaner, 2018, Tchetgen Tchetgen et al., 2020] to additional treatment efects beyond $\theta _ { 0 } ^ { A T E } ( d )$

Theorem 1 (Identification of treatment efects). If Assumptions 1 and 2 hold then

1. $\begin{array} { r } { \theta _ { 0 } ^ { A T E } ( d ) = \int h _ { 0 } ( d , x , w ) \mathrm { d } \mathbb { P } ( x , w ) . } \end{array}$

2. If in addition Assumption 3 holds, then $\begin{array} { r } { \theta _ { 0 } ^ { D S } ( d , \tilde { \mathbb { P } } ) = \int h _ { 0 } ( d , x , w ) \mathrm { d } \tilde { \mathbb { P } } ( x , w ) } \end{array}$

3. $\begin{array} { r } { \theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } ) = \int h _ { 0 } ( d ^ { \prime } , x , w ) \mathrm { d } \mathbb { P } ( x , w | d ) } \end{array}$

4. $\begin{array} { r } { \theta _ { 0 } ^ { C A T E } ( d , v ) = \int h _ { 0 } ( d , v , x , w ) \mathrm { d } \mathbb { P } ( x , w | v ) . } \end{array}$

See Appendix C for the proof. In Theorem 1, we see how negative controls (Z, W) allow us to adjust for unobserved confounding U and to thereby recover the treatment efect of interest. Each treatment efect is a reweighting of the confounding bridge $h _ { 0 }$ defined in Assumption 2 with respect to some distribution $\mathbb { Q } .$ In this sense, each treatment efect is an example of the same general nonparametric learning problem: $\begin{array} { r } { \int h _ { 0 } ( d , x , w ) \mathrm { d } \mathbb { Q } } \end{array}$ ,where Q may be an unconditional distribution such as $\mathbb { P } ( x , w )$ or a conditional distribution such as $\mathbb { P } ( x , w | d )$

## 3.3 RKHS background

Until this point, I have placed only causal assumptions formalized in Assumptions 1, 2, and 3. For computational and analytical tractability, I now place additional structure on the learning problem: I assume key quantities are elements of a reproducing kernel Hilbert space (RKHS). The RKHS is a canonical setting in machine learning, and it is a space of smooth functions that generalizes the Sobolev space. For a broad statistical audience, I organize ideas from RKHS learning theory that underpin the algorithm derivation (Section 4) and consistency guarantee (Section 5) to follow.

Kernel and feature map. I begin with basic kernel and feature map notation. Consider the RKHS  which consist of functions of the form $f : { \mathcal { A } }  \mathbb { R }$ , where is a Polish space (defined formally below). An RKHS  is characterized by its feature map $\phi ( a )$ which can be interpreted as the dictionary of basis functions for the RKHS in the sense that, for any $f \in \mathcal { H } , f ( a ) = \langle f , \phi ( a ) \rangle _ { \mathcal { H } }$ . The kernel $k : \mathcal { A } \times \mathcal { A } \to \mathbb { R }$ is a positive definite, symmetric, and continuous function such that $\begin{array} { r } { k ( { a } , { a } ^ { \prime } ) = \langle \phi ( { a } ) , \phi ( { a } ^ { \prime } ) \rangle _ { \mathcal { H } } ; } \end{array}$ it is the inner product of features, so it encodes the geometry of the RKHS. Alternatively, one may define the kernel first, then define the feature map as $\phi : { \mathcal { A } }  { \mathcal { H } } , a \mapsto k ( a , \cdot )$ . The feature map perspective is helpful for theory, but the kernel perspective is helpful for practice, since $k ( a , a ^ { \prime } )$ is a scalar that can be computed. Ultimately, I will reduce the algorithm to kernel evaluations.

Kernel mean embedding. We have seen how the feature map helps to evaluate a function. A related object, called the kernel mean embedding, helps to take the expectation of a function. Suppose we wish to calculate $\mathbb { E } [ f ( A ) ]$ . The idea of kernel mean embedding is to write

$$
\mathbb {E} [ f (A) ] = \int f (a) \mathrm{d} \mathbb {P} (a) = \int \langle f, \phi (a) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (a) = \left\langle f, \int \phi (a) \mathrm{d} \mathbb {P} (a) \right\rangle_ {\mathcal {H}} = \langle f, \mu \rangle_ {\mathcal {H}}
$$

where the exchange of expectation and inner product requires weak regularity conditions (defined formally below). The object $\textstyle \mu : = \int \phi ( a ) \mathrm { d } \mathbb { P } ( a )$ is called the mean embedding of the distribution $\mathbb { P } ( a )$ . A kernel is characteristic when the mapping $\mathbb { P } ( a ) \mapsto \mu$ is injective. The geometry of the RKHS implies that, to calculate the expectation of a function, it sufices to take the product of the function and the mean embedding of the corresponding distribution. This idea extends to conditional distributions over a subset of the arguments of the function. I will use this idea extensively when deriving the algorithm.

Tensor product. What if a function is defined over multiple variables, e.g. $f : { \mathcal { A } } \times { \mathcal { B } } $ R? A natural approach is to define an RKHS for such functions as the combination of RKHSs $\mathcal { H } _ { A }$ and $\mathcal { H } _ { B }$ that contain functions of the form $f _ { 1 } : { \mathcal { A } }  \mathbb { R }$ and $f _ { 2 } : B \to \mathbb { R }$ 2 respectively. Denote the individual feature maps by $\phi _ { A } ( a )$ and $\phi _ { B } ( b )$ , then define the tensor product feature map $\phi ( a , b ) = \phi _ { A } ( a ) \otimes \phi _ { B } ( b )$ for the tensor product RKHS . The tensor product is a generalization of the outer product; formally, $[ a \otimes b ] c = a \langle b , c \rangle$ . Then, for any $f \in \mathcal { H } , f ( a , b ) = \langle f , \phi _ { \mathcal { A } } ( a ) \otimes \phi _ { \mathcal { B } } ( b ) \rangle _ { \mathcal { H } }$ . It turns out that the kernel of this RKHS is simply the product of the kernels of the individual RKHSs: $k ( a , b ; a ^ { \prime } , b ^ { \prime } ) = k _ { \cal A } ( a , a ^ { \prime } ) \cdot k _ { \cal B } ( b , b ^ { \prime } )$ As such, $k ( a , b ; a ^ { \prime } , b ^ { \prime } )$ is a scalar that can be computed. In this work, I will extensively use tensor product constructions. For this reason, the algorithm statements will have the symbol  for the elementwise product of objects that contain kernel evaluations.

RKHS for operators. So far, I have defined RKHSs for functions of one or more variables. RKHSs also exist for operators. I denote by $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { A } } , \mathcal { H } _ { B } )$ the space of Hilbert-Schmidt operators of the form $E : \mathcal { H } _ { \mathcal { A } } \to \mathcal { H } _ { B }$ . It turns out that this space is an RKHS in its own right. The operators of interest are conditional expectation operators, which correspond to conditional mean embeddings. For example, consider the goal of calculating $\mathbb { E } [ f ( a ) | B = b ]$ . As before, one can express

$$
\int f (a) \mathrm{d} \mathbb {P} (a | b) = \int \langle f, \phi_ {\mathcal {A}} (a) \rangle_ {\mathcal {H} _ {\mathcal {A}}} \mathrm{d} \mathbb {P} (a | b) = \left\langle f, \int \phi_ {\mathcal {A}} (a) \mathrm{d} \mathbb {P} (a | b) \right\rangle_ {\mathcal {H} _ {\mathcal {A}}} = \langle f, \mu_ {a} (b) \rangle_ {\mathcal {H} _ {\mathcal {A}}}
$$

where $\begin{array} { r } { \mu _ { a } ( b ) : = \int \phi _ { \mathcal { A } } ( a ) \mathrm { d } \mathbb { P } ( a | b ) } \end{array}$ is called the conditional mean embedding of the distribution $\mathbb { P } ( a | b )$ . Observe that

$$
\mu_ {a} (b) = \int \phi (a) \mathrm{d} \mathbb {P} (a | b) = [ E \phi (\cdot) ] (b) = [ E ^ {*} \phi (b) ] (\cdot)
$$

where $E ^ { * } : \mathcal { H } _ { B } \to \mathcal { H } _ { A }$ is the adjoint of $E : { \mathcal { H } } _ { { \mathcal { A } } } \to { \mathcal { H } } _ { B }$ , and both are conditional expectation operators. Formally, the operators $E : f ( \cdot ) \mapsto \mathbb { E } [ f ( A ) | B = \cdot ]$ and $E ^ { * } : g ( \cdot ) \mapsto \mathbb { E } [ g ( B ) | A = \cdot ]$ encode the same information as the conditional mean embedding $\mu _ { a } ( b )$ . This relationship will facilitate estimation and analysis.

Closed form solution. The final piece of RKHS machinery necessary for the algorithm derivation (Section 4) is the so-called kernel trick. We have seen how a kernel evaluation is, in the end, simply a scalar. So, if an analyst can express an algorithm exclusively in terms of kernel evaluations, then the algorithm can be easily computed. A virtue of kernel methods is that they tend to have closed form solutions in terms of kernel evaluations. Conceptually, an RKHS algorithm $\hat { f } ( a ) = \langle \hat { f } , \phi ( a ) \rangle _ { \mathcal { H } }$ involves a possibly nonlinear feature map $\phi ( \cdot )$ applied to the data, so such an algorithm maintains computational simplicity while allowing for rich nonlinearity. Consider, for example, the kernel ridge regression

$$
\hat {f} = \underset {f \in \mathcal {H}} {\mathrm{argmin}} \frac {1}{n} \sum_ {i = 1} ^ {n} \{y _ {i} - \langle f, \phi (a _ {i}) \rangle_ {\mathcal {H}} \} ^ {2} + \lambda \| f \| _ {\mathcal {H}} ^ {2}\tag{1}
$$

with regularization hyperparameter $\lambda > 0$ . Its closed form solution is

$$
\hat {f} (a) = \langle \hat {f}, \phi (a) \rangle_ {\mathcal {H}} = \mathbf {Y} ^ {\top} (\mathbf {K} _ {A A} + n \lambda \mathbf {I}) ^ {- 1} \mathbf {K} _ {A a}\tag{2}
$$

where $\mathbf { Y } \in \mathbb { R } ^ { n }$ is the vector of outcomes with i-th entry $y _ { i }$ , $\mathbf { K } _ { A A } \in \mathbb { R } ^ { n \times n }$ is the kernel matrix with $( i , j )$ -th entry $k ( a _ { i } , a _ { j } )$ , and ${ \bf K } _ { A a } \in \mathbb { R } ^ { n }$ is the evaluation vector with i-th entry $k ( a _ { i } , a )$ . The algorithms I propose generalize kernel ridge regression. Sometimes, instead of regressing the outcome Y on features $\phi ( A )$ , I will regress the outcome $Y$ on mean embeddings $\mu _ { a } ( B )$ . At other times, I will regress one collection of features $\phi ( A )$ on another collection of features $\phi ( B )$ . These generalizations haves losses and closed form solutions that generalize those of kernel ridge regression. As such, they are simple combinations of kernel matrices and evaluation vectors despite being nonparametric.

Spectral view. The statistical guarantees of Section 5 require the spectral view of the RKHS $\mathcal { H }$ . The spectral view is more challenging, but it is the only way to articular RKHS learning theory. Let $\mathbb { L } _ { 2 }$ denote the space of square integrable functions mapping from  to R with respect to measure $\mathbb { P } .$ . For a fixed kernel $k ,$ define the convolution operator $L : \mathbb { L } _ { 2 } \to \mathbb { L } _ { 2 }$ 2 $\begin{array} { r } { f \mapsto \int k ( a , \cdot ) f ( a ) \mathrm { d } \mathbb { P } ( a ) } \end{array}$ . By the spectral theorem, we can express the operator L in terms of its countable eigenvalues $\{ \eta _ { j } \}$ and eigenfunctions $\{ \varphi _ { j } \}$ : $\begin{array} { r } { L f = \sum _ { j = 1 } ^ { \infty } \eta _ { j } \langle f , \varphi _ { j } \rangle \cdot \varphi _ { j } } \end{array}$ . Without loss of generality, $\{ \eta _ { j } \}$ is a weakly decreasing sequence and $\{ \varphi _ { j } \}$ forms an orthonormal basis of $\mathbb { L } _ { 2 }$ . With this spectral notation, we are ready to formalize the sense in which the RKHS $\mathcal { H }$ is a smooth subset of $\mathbb { L } _ { 2 }$ . Since $\{ \varphi _ { j } \}$ forms an orthonormal basis of $\mathbb { L } _ { 2 }$ , any $f , g \in \mathbb { L } _ { 2 }$ can be expressed as $\begin{array} { r } { f = \sum _ { j = 1 } ^ { \infty } f _ { j } \varphi _ { j } } \end{array}$ and $\begin{array} { r } { g = \sum _ { j = 1 } ^ { \infty } g _ { j } \varphi _ { j } } \end{array}$ . By [Cucker and Smale, 2002, Theorem 4], $\mathbb { L } _ { 2 }$ <sup>and</sup> <sup>the</sup> <sup>RKHS</sup> H <sup>can</sup> <sup>be</sup> <sup>explicitly</sup> <sup>represented</sup> <sup>as</sup>

$$
\mathbb {L} _ {2} = \left\{f = \sum_ {j = 1} ^ {\infty} f _ {j} \varphi_ {j}: \sum_ {j = 1} ^ {\infty} f _ {j} ^ {2} <   \infty \right\}, \quad \langle f, g \rangle_ {\mathbb {L} _ {2}} = \sum_ {j = 1} ^ {\infty} f _ {j} g _ {j}
$$

$$
\mathcal {H} = \left\{f = \sum_ {j = 1} ^ {\infty} f _ {j} \varphi_ {j}: \sum_ {j = 1} ^ {\infty} \frac {f _ {j} ^ {2}}{\eta_ {j}} <   \infty \right\}, \quad \langle f, g \rangle_ {\mathcal {H}} = \sum_ {j = 1} ^ {\infty} \frac {f _ {j} g _ {j}}{\eta_ {j}}.
$$

The RKHS  is the subset of $\mathbb { L } _ { 2 }$ for which higher order terms in the series $\{ \varphi _ { j } \}$ have a smaller contribution. In the RKHS, there is a penalty on higher order coeficients, and the magnitude of the penalty corresponds to how small the eigenvalue is.

Main assumptions. Finally, I articulate the main approximation assumptions of this paper. Formally, to analyze bias, I assume that a statistical target $f _ { 0 }$ satisfies

$$
f _ {0} \in \mathcal {H} ^ {c} := \left\{f = \sum_ {j = 1} ^ {\infty} f _ {j} \varphi_ {j}: \sum_ {j = 1} ^ {\infty} \frac {f _ {j} ^ {2}}{\eta_ {j} ^ {c}} <   \infty \right\}, \quad c \in (1, 2 ].\tag{3}
$$

For $c = 1$ , we see that $\mathcal { H } ^ { 1 } = \mathcal { H } ;$ I am simply assuming that $f _ { 0 }$ is correctly specified as an element of the RKHS. For $c > 1$ , I am assuming that $f _ { 0 }$ is in the interior of the RKHS. This assumption is called the source condition in statistical learning theory and econometrics [Smale and Zhou, 2007, Caponnetto and De Vito, 2007, Carrasco et al., 2007]. As we will see, a larger value of c corresponds to a smoother target $f _ { 0 }$ and a faster uniform rate. I allow c to be as large as $c = 2$ , which is the highest degree of smoothness to which kernel ridge estimators can adapt. In Appendix D, I compare this main approximation assumption with alternative approximation assumptions in the negative control and NPIV literatures.

The second main approximation is a spectral decay assumption called the efective dimension of the basis $\{ \varphi _ { j } \}$ . I quantify the efective dimension as the rate at which the eigenvalues $\{ \eta _ { j } \}$ decay. Formally, to analyze variance, I assume that there exists some constant $C$ such that each $\eta _ { j }$ satisfies

$$
\eta_ {j} \leq C j ^ {- b}, \quad b \geq 1.\tag{4}
$$

[Fischer and Steinwart, 2020, Lemma 10] shows that a bounded kernel k satisfies this condition with b that is at least one. A higher value of b corresponds to a lower efective dimension, better control of the variance, and a faster rate. The limit $b \to \infty$ corresponds to an RKHS with finite dimension [Caponnetto and De Vito, 2007].

Special case: Sobolev space. The abstract approximation conditions are easy to interpret in the context of Sobolev spaces. Denote by H<sup>s</sup> the Sobolev space of functions of the form $f : { \mathcal { A } }  \mathbb { R }$ with $\mathcal { A } \subset \mathbb { R } ^ { p }$ . The parameter s denotes how many derivatives of f are square integrable. The Sobolev space H<sup>s</sup><sub>2</sub> is an RKHS if and only if $s > p / 2$ [Berlinet and Thomas-Agnan, 2011, Theorem 132]. Its kernel is known as the Matérn kernel. Suppose we take $\mathcal { H } = \mathbb { H } _ { 2 } ^ { s }$ with $s > p / 2$ as the RKHS for estimation. If the true target $f _ { 0 }$ is in $\mathbb { H } _ { 2 } ^ { s _ { 0 } }$ , then $c = s _ { 0 } / s$ [Fischer and Steinwart, 2020]. In the notation of (3), $\mathbb { H } _ { 2 } ^ { s _ { 0 } } = [ \mathbb { H } _ { 2 } ^ { s } ] ^ { c }$ . Clearly $c > 1$ means that the target $f _ { 0 }$ is in the interior of $\mathbb { H } _ { 2 } ^ { s }$ . Moreover, the efective dimension of the RKHS H<sup>s</sup> is quantified by $b = 2 s / p$ [Fischer and Steinwart, 2020]. Rates in terms of $( b , c )$ adapt to the smoothness of $f _ { 0 }$ and the efective dimension of the RKHS . They are invariant to dimension as long as $s _ { 0 } > s > p / 2$

## 4 Algorithm

## 4.1 RKHS construction

I provide a new RKHS construction for negative control treatment efect estimation, generalizing and unifying the constructions in [Singh et al., 2019, Singh et al., 2020]. In my construction, I define RKHSs for treatment $D ,$ negative controls $( Z , W )$ , and covariates $( V , X )$ For example, for treatment D define the RKHS $\mathcal { H } _ { \mathcal { D } }$ with feature map $\phi _ { \mathcal { D } } ( d )$ and kernel $k _ { \mathcal { D } } ( d , d ^ { \prime } ) = \langle \phi _ { \mathcal { D } } ( d ) , \phi _ { \mathcal { D } } ( d ^ { \prime } ) \rangle _ { \mathcal { H } _ { \mathcal { D } } }$ . Formally, $\phi _ { \mathcal { D } } : \mathcal { D }  \mathcal { H } _ { \mathcal { D } }$ and $k _ { \mathcal { D } } : \mathcal { D } \times \mathcal { D }  \mathbb { R }$ . To lighten notation, I suppress subscripts when arguments are provided, e.g. I write $\phi ( d ) = \phi _ { \mathcal { D } } ( d )$

From these individual RKHSs, I construct a tensor product RKHS  for the confounding bridge $h _ { 0 }$ . For clarity of exposition, I initially focus on the case without V, i.e. excluding $\theta _ { 0 } ^ { C A T E }$ . I assume the confounding bridge $h _ { 0 }$ is an element of the RKHS with tensor product feature map $\phi ( d , x , w ) : = \phi ( d ) \otimes \phi ( x ) \otimes \phi ( w )$ , i.e. $h _ { 0 } \in \mathcal { H } : = \mathcal { H } _ { D } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } }$ As before, the feature map can be interpreted as the dictionary of basis functions since $h _ { 0 } ( d , x , w ) = \langle h _ { 0 } , \phi ( d , x , w ) \rangle _ { \mathcal { H } } = \langle h _ { 0 } , \phi ( d ) \otimes \phi ( x ) \otimes \phi ( w ) \rangle _ { \mathcal { H } }$ . In Appendix B, I discuss how an analogous assumption for the regression $\gamma _ { 0 }$ relates to Assumption 2.

This tensor product RKHS construction plays a central role in deriving simple representations of treatment efects and hence deriving simple estimators, under weak regularity conditions. The RKHS aspect allows for the technique of kernel mean embedding: an analyst can reweight a function $h _ { 0 } ( d , x , w )$ according to some counterfactual distribution by reweighting its feature map $\phi ( d , x , w )$ by that distribution. The tensor product aspect ensures separability of the diferent variables in the feature map; since $\phi ( d , x , w ) \ = \ \phi ( d ) \otimes \phi ( x ) \otimes \phi ( w )$ , the reweighting can apply to the specific variables $\phi ( x ) \otimes \phi ( w )$ . Using these properties, I represent the causal quantities defined in Definition 1 and identified in Theorem 1 in a more tractable form. To begin, I state the regularity conditions.

Assumption 4 (RKHS regularity conditions). Assume

1. $\displaystyle k _ { \mathcal { D } } , k _ { \mathcal { X } } , k _ { \mathcal { W } }$ , and $k _ { \mathcal { Z } }$ are continuous and bounded:

$$
\sup _ {d \in \mathcal {D}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \leq \kappa_ {d}, \sup _ {x \in \mathcal {X}} \| \phi (x) \| _ {\mathcal {H} _ {\mathcal {X}}} \leq \kappa_ {x}, \sup _ {w \in \mathcal {W}} \| \phi (w) \| _ {\mathcal {H} _ {\mathcal {W}}} \leq \kappa_ {w}, \sup _ {z \in \mathcal {Z}} \| \phi (z) \| _ {\mathcal {H} _ {\mathcal {Z}}} \leq \kappa_ {z};
$$

2. $\phi ( d ) , \phi ( x ) , \phi ( w )$ , and $\phi ( z )$ are measurable;

3. $k _ { \mathcal { X } }$ and $k _ { \mathcal { W } }$ are characteristic.

For $\theta _ { 0 } ^ { C A T E }$ , extend the stated assumptions from X to $( V , X )$

Continuity, boundedness, and measurability are weak conditions satisfied by commonly used kernels. The characteristic property is a regularity condition for embedding a distribution in an RKHS, and it is satisfied by commonly used kernels as well [Sriperumbudur et al., 2010]. Formally, $k _ { \mathcal { X } }$ is characteristic if and only if, for all Borel probability measures $\mathbb { Q } ,$ the mapping $\begin{array} { r } { \mathbb { Q } \mapsto \int \phi ( x ) \mathrm { d } \mathbb { Q } } \end{array}$ is injective. I explain the role of the characteristic property below.

Theorem 2 (Representation via kernel mean embedding). Suppose the conditions of Theorem 1 hold. Further suppose Assumption 4 holds and $h _ { 0 } \in \mathcal { H }$ . Then

$$
\gamma_ {0} (d, x, z) = \left\langle h _ {0}, \phi (d) \otimes \phi (x) \otimes \mu_ {w} (d, x, z) \right\rangle_ {\mathcal {H}} \text {where} \mu_ {w} (d, x, z) := \int \phi (w) \mathrm{d} \mathbb {P} (w | d, x, z).
$$

Moreover

1. $\theta _ { 0 } ^ { A T E } ( d ) = \langle h _ { 0 } , \phi ( d ) \otimes \mu \rangle _ { \mathcal { H } }$ where $\begin{array} { r } { \mu : = \int [ \phi ( x ) \otimes \phi ( w ) ] \mathrm { d } \mathbb { P } ( x , w ) } \end{array}$ ;

2. $\theta _ { 0 } ^ { D S } ( d , \tilde { \mathbb { P } } ) = \langle h _ { 0 } , \phi ( d ) \otimes \nu \rangle _ { \mathcal { H } }$ where $\begin{array} { r } { \nu : = \int [ \phi ( x ) \otimes \phi ( w ) ] \mathrm { d } \tilde { \mathbb { P } } ( x , w ) } \end{array}$ ;

3. $\theta _ { 0 } ^ { A T T } ( d , d ^ { \prime } ) = \langle h _ { 0 } , \phi ( d ^ { \prime } ) \otimes \mu ( d ) \rangle _ { \mathcal { H } }$ where $\begin{array} { r } { \mu ( d ) : = \int [ \phi ( x ) \otimes \phi ( w ) ] \mathrm { d } \mathbb { P } ( x , w | d ) } \end{array}$ ;

$$
4. \theta_ {0} ^ {C A T E} (d, v) = \left\langle h _ {0}, \phi (d) \otimes \phi (v) \otimes \mu (v) \right\rangle_ {\mathcal {H}} \text {where} \mu (v) := \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \mathbb {P} (x, w | v).
$$

I present the proof in Appendix E. Whereas the expressions in Theorem 1 are reweightings of the confounding bridge $h _ { 0 }$ , the expressions in Theorem 2 are inner products of $h _ { 0 }$ . The quantity $\mu _ { w } ( d , x , z )$ encodes the conditional distribution $\mathbb { P } ( w | d , x , z )$ from the integral equation that defines the confounding bridge $h _ { 0 }$ . The quantities $\mu , \nu , \mu ( d ) , \mu ( v )$ embed various reweighting distributions: $\mathbb { P } ( x , w ) , \tilde { \mathbb { P } } ( x , w ) , \mathbb { P } ( x , w | d )$ , and $\mathbb { P } ( x , w | v )$ , respectively. In general, the quantity $\begin{array} { r } { \int [ \phi ( \boldsymbol { x } ) \otimes \phi ( \boldsymbol { w } ) ] \mathrm { d } \mathbb { Q } } \end{array}$ encodes the distribution $\mathbb { Q }$ as a function in $\mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } }$ The characteristic property for $k _ { \mathcal { X } }$ and $k _ { \mathcal { W } }$ ensures that the mapping $\begin{array} { r } { \mathbb { Q } \mapsto \int [ \phi ( x ) \otimes \phi ( w ) ] \mathrm { d } \mathbb { Q } } \end{array}$ is injective, so that the RKHS representation of the reweighting distribution Q is unique.

These representations abstractly suggest estimators. For example, for $\theta _ { 0 } ^ { A T E } ( d )$ the estimator should be of the form $\hat { \theta } ^ { A T E } ( d ) = \langle \hat { h } , \phi ( d ) \otimes \hat { \mu } \rangle _ { \mathcal { H } }$ , where $\hat { h }$ is an estimator of the confounding bridge $h _ { 0 }$ and $\hat { \mu }$ is an estimator of the mean embedding $\mu .$ . I propose a regularized kernel estimator of the confounding bridge function in the spirit of two stage least squares (2SLS): first project $\phi ( W )$ onto $\phi ( D ) \otimes \phi ( X ) \otimes \phi ( Z )$ to obtain $\hat { \mu } _ { w } ( D , X , Z )$ , then project Y onto $\phi ( D ) \otimes \phi ( X ) \otimes { \hat { \mu } } _ { w } ( D , X , Z )$ to obtain $\hat { h }$ . I estimate unconditional mean embeddings $\hat { \mu } , \hat { \nu }$ with simple averages, and I estimate conditional mean embeddings ${ \hat { \mu } } ( d ) , { \hat { \mu } } ( v )$ with projections similar to $\hat { \mu } _ { w } ( d , x , z )$

## 4.2 Generalized regression loss

It is not obvious that $\hat { \theta } ^ { A T E } ( d ) = \langle \hat { h } , \phi ( d ) \otimes \hat { \mu } \rangle _ { \mathcal { H } }$ has a closed form expression in terms of kernel matrices. In this section, I state the generalized regression losses that define the estimator. These losses generalize the loss in (1). Along the way, I discuss the connection to 2SLS and provide intuition for these techniques. In the next section, I formally prove that a closed form expression exists and then solve for it, generalizing the expression in (2).

Similar to 2SLS, I estimate the confounding bridge $\hat { h }$ in two stages. In the first stage, I estimate the conditional mean embedding $\hat { \mu } _ { w } ( d , x , z )$ . Let n be the number of observations of $( d _ { i } , x _ { i } , w _ { i } , z _ { i } )$ used to estimate the conditional mean embedding $\hat { \mu } _ { w } ( d , x , z )$ with regularization parameter $\lambda .$ The generalized regression loss for the regression of $\phi ( W )$ on $\phi ( D , X , Z )$ is

$$
\hat {E} = \underset {E \in \mathcal {L} _ {2} (\mathcal {H} _ {\mathcal {W}}, \mathcal {H} _ {\mathcal {D}} \otimes \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {Z}})} {\mathrm{argmin}} \frac {1}{n} \sum_ {i = 1} ^ {n} \| \phi (w _ {i}) - E ^ {*} \phi (d _ {i}, x _ {i}, z _ {i}) \| _ {\mathcal {H} _ {\mathcal {W}}} ^ {2} + \lambda \| E \| _ {\mathcal {L} _ {2} (\mathcal {H} _ {\mathcal {W}}, \mathcal {H} _ {\mathcal {D}} \otimes \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {Z}})} ^ {2}
$$

so that $\hat { \mu } _ { w } ( d , x , z ) = \hat { E } ^ { * } \phi ( d , x , z )$ . In this notation, $E ^ { * }$ is the adjoint operator of $E ,$ and $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { Z } } )$ is an RKHS whose elements are operators of the form $E : \mathcal { H } _ { \mathcal { W } } $ $\mathcal { H } _ { D } \otimes \mathcal { H } _ { X } \otimes \mathcal { H } _ { Z }$

In the second stage, I estimate the confounding bridge $\hat { h } .$ . Let $m$ be the number of observations of $( { \dot { y } } _ { i } , { \dot { d } } _ { i } , { \dot { x } } _ { i } , { \dot { z } } _ { i } )$ used to estimate the confounding bridge $\hat { h }$ with regularization parameter $\xi .$ . This notation allows the analyst to use diferent quantities of observations $( n , m )$ to estimate $\hat { \mu } _ { w } ( d , x , z )$ and $\hat { h } .$ , or to reuse the same observations. To estimate $\hat { h } .$ I regress $Y$ on $\hat { \mu } ( D , X , Z ) : = \phi ( D ) \otimes \phi ( X ) \otimes \hat { \mu } _ { w } ( D , X , Z )$ . The generalized regression loss

for the regression of $Y$ on $\hat { \mu } ( D , X , Z )$ is

$$
\hat {h} = \underset {h \in \mathcal {H}} {\mathrm{argmin}} \frac {1}{m} \sum_ {i = 1} ^ {m} \left\{\dot {y} _ {i} - \langle h, \hat {\mu} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \right\} ^ {2} + \xi \| h \| _ {\mathcal {H}} ^ {2}.
$$

[Mastouri et al., 2021, eqs. 5, 6] propose the same generalized regression losses for the confounding bridge. The original draft of this paper [Singh, 2020] misquoted the regression loss for $\hat { E }$ from [Singh et al., 2019, Section 4.1].<sup>2</sup> This correction was pointed out by [Mastouri et al., 2021] and an anonymous referee prior to the current draft. However, the closed form [Mastouri et al., 2021, Proposition 2] difers as detailed below.

Finally, when estimating $\hat { \theta } ^ { A T T }$ and $\hat { \theta } ^ { C A T E }$ , one must also estimate the conditional mean embeddings $\hat { \mu } ( d )$ and $\hat { \mu } ( v )$ . The losses for these conditional mean embeddings mirror the loss for the conditional mean embedding $\hat { \mu } _ { w } ( d , x , z )$

## 4.3 Closed form

I present a closed form solution for the confounding bridge estimator, generalizing kernel instrumental variable regression [Singh et al., 2019, Algorithm 1] to my extended RKHS construction.

Algorithm 1 (Estimation of confounding bridge). Let mean elementwise product. Then

$$
\begin{array}{r l} & {\mathbf {A} = \mathbf {K} _ {D D} \odot \mathbf {K} _ {X X} \odot \mathbf {K} _ {Z Z} \in \mathbb {R} ^ {n \times n}, \quad \dot {\mathbf {A}} = \mathbf {K} _ {D \dot {D}} \odot \mathbf {K} _ {X \dot {X}} \odot \mathbf {K} _ {Z \dot {Z}} \in \mathbb {R} ^ {n \times m},} \\ & {\mathbf {B} = (\mathbf {A} + n \lambda \mathbf {I}) ^ {- 1} \dot {\mathbf {A}} \in \mathbb {R} ^ {n \times m}, \qquad \mathbf {M} = \mathbf {K} _ {\dot {D} \dot {D}} \odot \mathbf {K} _ {\dot {X} \dot {X}} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W W} \mathbf {B} \} \in \mathbb {R} ^ {m \times m},} \\ & {\hat {\pmb {\alpha}} = (\mathbf {M M} ^ {\top} + m \xi \mathbf {M}) ^ {- 1} \mathbf {M} \dot {\mathbf {Y}} \in \mathbb {R} ^ {m}, \quad \hat {h} (d, x, w) = \hat {\pmb {\alpha}} ^ {\top} [ \mathbf {K} _ {\dot {D} d} \odot \mathbf {K} _ {\dot {X} x} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W w} \} ] \in \mathbb {R}} \end{array}
$$

where (λ, ξ) are ridge penalty hyperparameters.

See Appendix E for the derivation, which begins with an original proof that such an $\hat { \pmb { \alpha } } \in$ $\mathbb { R } ^ { m }$ even exists. [Mastouri et al., 2021, Proposition 2] show that a matrix representation $\hat { \pmb { \alpha } } \in \mathbb { R } ^ { n \times m }$ exists, rather than the vector representation $\hat { \pmb { \alpha } } \in \mathbb { R } ^ { m }$ in Algorithm 1. The vector of representation of Algorithm 1 is similar to kernel ridge regression. The elementwise products arise because tensor product RKHSs correspond to product kernels. For example, the kernel of $\mathcal { H } _ { \mathcal { D } } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { Z } } \mathrm { ~ i s ~ } k ( d , x , z ; d ^ { \prime } , x ^ { \prime } , z ^ { \prime } ) = k _ { \mathcal { D } } ( d , d ^ { \prime } ) k _ { \mathcal { X } } ( x , x ^ { \prime } ) k _ { \mathcal { Z } } ( z , z ^ { \prime } )$ so its kernel matrix is ${ \bf K } _ { D D } \odot { \bf K } _ { X X } \odot { \bf K } _ { Z Z }$ . M is essentially the kernel matrix for the conditional mean embedding $\hat { \mu } ( d , x , z ) = \phi ( d ) \otimes \phi ( x ) \otimes \hat { \mu } _ { w } ( d , x , z )$ . Interpreting these expressions, $\hat { \alpha }$ is clearly the regularized empirical projection of Y onto $\phi ( D ) \otimes \phi ( X ) \otimes { \hat { \mu } } _ { w } ( D , X , Z )$ in the spirit of 2SLS.

Next, I present closed form solutions for treatment efects, e.g. $\hat { \theta } ^ { A T E } ( d ) = \langle \hat { h } , \phi ( d ) \otimes \hat { \mu } \rangle _ { \mathcal { H } }$ building on $\hat { h }$ from Algorithm 1. Whereas [Singh et al., 2020, Algorithm 3.1] estimate treatment efects assuming selection on observables, Algorithm 2 estimates treatment efects assuming access to negative controls. For $\theta _ { 0 } ^ { D S }$ , let n˜ be the number of observations of $( \tilde { x } _ { i } , \tilde { w } _ { i } )$ drawn from population $\tilde { \mathbb { P } } .$

Algorithm 2 (Estimation of treatment efects). Treatment efect estimators have the closed form solutions

$$
1. \hat {\theta} ^ {A T E} (d) = n ^ {- 1} \sum_ {i = 1} ^ {n} \hat {\pmb {\alpha}} ^ {\top} [ \mathbf {K} _ {\dot {D} d} \odot \mathbf {K} _ {\dot {X} x _ {i}} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W w _ {i}} \} ]
$$

$$
\hat {\theta} ^ {D S} (d, \tilde {\mathbb {P}}) = \tilde {n} ^ {- 1} \sum_ {i = 1} ^ {\tilde {n}} \hat {\boldsymbol {\alpha}} ^ {\top} [ \mathbf {K} _ {\dot {D} d} \odot \mathbf {K} _ {\dot {X} \tilde {x} _ {i}} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W \tilde {w} _ {i}} \} ]
$$

$$
3. \hat {\theta} ^ {A T T} (d, d ^ {\prime}) = \hat {\pmb {\alpha}} ^ {\top} [ \mathbf {K} _ {\dot {D} d ^ {\prime}} \odot \{[ \mathbf {K} _ {\dot {X} X} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W W} \} ] (\mathbf {K} _ {D D} + n \lambda_ {1} \mathbf {I}) ^ {- 1} \mathbf {K} _ {D d} \} ]
$$

$$
4. \hat {\theta} ^ {C A T E} (d, v) = \hat {\pmb {\alpha}} ^ {\top} [ \mathbf {K} _ {\dot {D} d} \odot \mathbf {K} _ {\dot {V} v} \odot \{[ \mathbf {K} _ {\dot {X} X} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W W} \} ] (\mathbf {K} _ {V V} + n \lambda_ {2} \mathbf {I}) ^ {- 1} \mathbf {K} _ {V v} \} ]
$$

where $( \lambda _ { 1 } , \lambda _ { 2 } )$ are ridge regression penalty hyperparameters. In $\widehat { \theta } ^ { C A T E } ( d , v )$ , αˆ is the coeficient for the confounding bridge that includes V .

See Appendix E for the derivation. I give theoretical values for the regularization parameters that balance bias and variance in Section 5 below. In particular, I specify $( \lambda , \xi )$ in Theorem 3 and $( \lambda _ { 1 } , \lambda _ { 2 } )$ in Theorem 4. In Appendix F, I propose a practical tuning procedure based on the closed form solution of leave-one-out cross validation (LOOCV) to empirically balance bias and variance, and I discuss the time complexity.

## 4.4 Summary

To fix ideas, I summarize the end-to-end procedure for $\hat { \theta } ^ { A T E } ( d )$ . For simplicity, I suppose that the analyst re-uses observations in the two stages of confounding bridge estimation. I provide additional discussion in Appendix E for researchers who are new to kernel methods.

Algorithm 3 (End-to-end details). Given n observations of outcome $Y ,$ , treatment $D _ { : }$ covariates X, negative control outcome W, and negative control treatment $Z ,$

1. Specify the kernels $k _ { D } , k _ { X } , k _ { \mathcal { W } } , k _ { \mathcal { Z } }$

(a) For multivariate objects, e.g. $\boldsymbol { X } = ( X _ { 1 } , . . . , X _ { p } )$ , use the product of scalar kernels

$$
k _ {\mathcal {X}} (x, x ^ {\prime}) = \prod_ {j = 1} ^ {p} k _ {\mathcal {X} _ {1}} (x _ {1}, x _ {1} ^ {\prime}) \cdot \ldots \cdot k _ {\mathcal {X} _ {p}} (x _ {p}, x _ {p} ^ {\prime}).
$$

(b) Tune the scalar kernel hyperparameters. For example, if the treatment kernel $k _ { D }$ is chosen as the Gaussian kernel, a standard heuristic is to use the median interpoint distance among observed treatment values.

(c) Compute the kernel matrices, e.g. ${ \bf K } _ { D D } \in \mathbb { R } ^ { n \times n }$ with $( i , j ) -$ th entry $k _ { \mathcal { D } } ( d _ { i } , d _ { j } )$

2. Specify the regularization hyperparameters $( \lambda , \xi )$

(a) For $\lambda ,$ I derive the closed form solution of LOOCV in Appendix F.

(b) The same procedure applies to $\xi ,$ plugging in the chosen value of $\lambda .$ .

3. Estimate the confounding bridge $\hat { h }$ in two stages, using $( \lambda , \xi )$

(a) Estimate the distribution in the integral equation $\hat { \mathbb { P } } ( w | d , x , z )$ via its mean embedding $\hat { \mu } _ { w } ( d , x , z )$ with regularization λ as

$$
[ \hat {\mu} _ {w} (d, x, z) ] (w) = \mathbf {K} _ {w W} (\mathbf {K} _ {D D} \odot \mathbf {K} _ {X X} \odot \mathbf {K} _ {Z Z} + n \lambda \mathbf {I}) ^ {- 1} [ \mathbf {K} _ {D d} \odot \mathbf {K} _ {X x} \odot \mathbf {K} _ {Z z} ].
$$

(b) Regress Y onto $\phi ( D ) \otimes \phi ( X ) \otimes { \hat { \mu } } _ { w } ( D , X , Z )$ with regularization $\xi .$ It turns out that $\hat { h } ( d , x , w ) = \hat { \pmb { \alpha } } ^ { \top } [ \mathbf { K } _ { D d } \odot \mathbf { K } _ { X x } \odot \{ \mathbf { B } ^ { \top } \mathbf { K } _ { W w } \} ] \in \mathbb { 1 }$ R where

$$
\mathbf {A} = \mathbf {K} _ {D D} \odot \mathbf {K} _ {X X} \odot \mathbf {K} _ {Z Z} \in \mathbb {R} ^ {n \times n}, \quad \mathbf {B} = (\mathbf {A} + n \lambda \mathbf {I}) ^ {- 1} \mathbf {A} \in \mathbb {R} ^ {n \times n},
$$

$$
\mathbf {M} = \mathbf {K} _ {D D} \odot \mathbf {K} _ {X X} \odot \left\{\mathbf {B} ^ {\top} \mathbf {K} _ {W W} \mathbf {B} \right\} \in \mathbb {R} ^ {n \times n}, \quad \hat {\boldsymbol {\alpha}} = (\mathbf {M M} ^ {\top} + n \xi \mathbf {M}) ^ {- 1} \mathbf {M Y} \in \mathbb {R} ^ {n}.
$$

4. Estimate the counterfactual distribution $\hat { \mathbb { P } } ( x , w )$ via its mean embedding $\hat { \mu }$ as

$$
[ \hat {\mu} ] (x, w) = \frac {1}{n} \sum_ {i = 1} ^ {n} k _ {\mathcal {X}} (x _ {i}, x) k _ {\mathcal {W}} (w _ {i}, w) \in \mathbb {R}.
$$

5. Estimate the dose response $\hat { \theta } ^ { A T E } ( d )$ by combining $\hat { h }$ and $\hat { \mu }$ according to $\hat { \theta } ^ { A T E } ( d ) =$ $\langle \hat { h } , \phi ( d ) \otimes \hat { \mu } \rangle _ { \mathcal { H } }$ . To do so, match the common arguments $( x , w )$ of $\hat { h }$ and $\hat { \mu }$ . In summary,

$$
\hat {\theta} ^ {A T E} (d) = \frac {1}{n} \sum_ {i = 1} ^ {n} \hat {\boldsymbol {\alpha}} ^ {\top} [ \mathbf {K} _ {D d} \odot \mathbf {K} _ {X x _ {i}} \odot \{\mathbf {B} ^ {\top} \mathbf {K} _ {W w _ {i}} \} ] \in \mathbb {R}.
$$

## 5 Consistency

To define the learning problem in Section 3, I placed three assumptions: availability of negative controls (Assumption 1); existence and completeness of the confounding bridge (Assumption 2); and invariance of the confounding bridge for transfer learning (Assumption 3). To construct an algorithm in Section 4, I assumed RKHS regularity (Assumption 4). To guarantee uniform consistency in this section, I place three final assumptions: original space regularity (Assumption 5); smoothness and efective dimension of conditional expectation operators (Assumption 6); and smoothness and efective dimension of the confounding bridge (Assumption 7). I first prove uniform consistency of the confounding bridge, then uniform consistency of treatment efects. As before, for $\theta _ { 0 } ^ { C A T E }$ I extend the stated assumptions from X to (V, X).

[Mastouri et al., 2021, Theorem 2] analyze excess risk of a surrogate loss for a kernel two stage regression estimator of the confounding bridge, with finite sample rates, under smoothness and efective dimension assumptions. Excess risk of a surrogate loss corresponds to projected mean square error for the confounding bridge.

## 5.1 Confounding bridge

I require weak regularity conditions on the original spaces of the outcome $Y$ , treatment D, covariates (V, X), and negative controls (W, Z).

Assumption 5 (Original space regularity conditions). Assume

1. $Y \in \mathcal { V } \subset \mathbb { R }$ is bounded, i.e. there exists $C < \infty$ such that $| Y | \le C$ almost surely.

2. ,  , , and $\mathcal { Z }$ are Polish spaces.

To simplify notation and analysis, I require that the outcome $Y \in \mathbb { R }$ is a bounded scalar. <sup>More</sup> <sup>generally,</sup> Y <sup>could</sup> <sup>be</sup> <sup>a</sup> <sup>separable</sup> <sup>Hilbert</sup> <sup>space.</sup> <sup>I</sup> <sup>preserve</sup> <sup>generality</sup> <sup>for</sup> <sup>treatment,</sup> covariates, and negative controls. A Polish space is a separable and completely metrizable topological space. Random variables with support in a Polish space may be discrete or continuous and low, high, or infinite dimensional. As such, I allow for treatment, covariates, and negative controls that could even be texts, graphs, or images.

Next, I place a smoothness and efective dimension conditions in the sense of (3) and (4) for the conditional mean embedding $\mu _ { w } ( d , x , z )$ . In anticipation of later analysis, I articulate this assumption abstractly. Consider the abstract conditional mean embedding $\mu _ { a } ( b ) : =$ $\textstyle \int \phi ( a ) \mathrm { d } \mathbb { P } ( a | b )$ where $a \in \mathcal A _ { \ell }$ and $b \in \ B _ { \ell }$ . I will ultimately consider the three diferent conditional mean embeddings $\mu _ { w } ( d , x , z ) , \ \mu ( d )$ , and $\mu ( v )$ indexed by $\ell \in \{ 0 , 1 , 2 \}$ . As previewed in Section 3, the conditional expectation operator $E _ { \ell } : \mathcal { H } _ { A _ { \ell } } \to \mathcal { H } _ { B _ { \ell } } , \ f ( \cdot ) \ \mapsto$ $\mathbb { E } [ f ( A _ { \ell } ) | B _ { \ell } = \cdot ]$ encodes the same information as $\mu _ { a } ( b )$ . In particular,

$$
\mu_ {a} (b) = \int \phi (a) \mathrm{d} \mathbb {P} (a | b) = [ E _ {\ell} \phi (\cdot) ] (b) = [ E _ {\ell} ^ {*} \phi (b) ] (\cdot), \quad a \in \mathcal {A} _ {\ell}, \quad b \in \mathcal {B} _ {\ell}
$$

where $E _ { \ell } ^ { * } : \mathcal { H } _ { \mathcal { B } _ { \ell } } \to \mathcal { H } _ { \mathcal { A } _ { \ell } } , g ( \cdot ) \mapsto \mathbb { E } [ g ( B _ { \ell } ) | A _ { \ell } = \cdot ]$ is the adjoint of $E _ { \ell }$ . I denote the space of Hilbert-Schmidt operators between $\mathcal { H } _ { A _ { \ell } }$ and $\mathcal { H } _ { B _ { \ell } }$ by $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { A } _ { \ell } } , \mathcal { H } _ { B _ { \ell } } )$ , which is an RKHS in its own right.

Assumption 6 (Smoothness and spectral decay for conditional expectation operator). Assume $E _ { \ell } \in [ \mathcal { L } _ { 2 } ( \mathcal { H } _ { A _ { \ell } } , \mathcal { H } _ { B _ { \ell } } ) ] ^ { c _ { \ell } }$ and $\eta _ { j } ( \mathcal { H } _ { B _ { \ell } } ) \le C j ^ { - b _ { \ell } }$

To specialize the assumption, all one has to do is specify $A _ { \ell }$ and $B _ { \ell }$ . For example, for $\mu _ { w } ( d , x , z ) , \mathcal { A } _ { 0 } = \mathcal { W }$ and $\boldsymbol { B } _ { 0 } = \mathcal { D } \times \boldsymbol { \mathcal { X } } \times \boldsymbol { \mathcal { Z } }$ . By assuming smoothness and efective dimension of $E _ { 0 }$ , I assume smoothness and efective dimension of $\mu _ { w } ( d , x , z )$ . I explicitly specialize the assumption in Appendices G and H.

The final assumption that I place in order to prove uniform consistency of the confounding bridge estimator $\hat { h }$ is that the confounding bridge $h _ { 0 }$ is smooth in the sense of (3) with low efective dimension in the sense of (4). Recall that the features for the RKHS are $\phi ( d , x , w ) = \phi ( d ) \otimes \phi ( x ) \otimes \phi ( w )$ . Recall from Theorem 2 that we must solve the integral equation $\gamma _ { 0 } ( d , x , z ) = \langle h _ { 0 } , \mu ( d , x , z ) \rangle _ { \mathcal { H } }$ where $\mu ( d , x , z ) = \phi ( d ) \otimes \phi ( x ) \otimes \mu _ { w } ( d , x , z )$ is a mean embedding. By construction,

$$
\langle \mu (d, x, z), \mu (d ^ {\prime}, x ^ {\prime}, z ^ {\prime}) \rangle_ {\mathcal {H}} = k _ {\mathcal {D}} (d, d ^ {\prime}) k _ {\mathcal {X}} (x, x ^ {\prime}) \int k _ {\mathcal {W}} (w, w ^ {\prime}) \mathrm{d} \mathbb {P} (w | d, x, z) \mathrm{d} \mathbb {P} (w ^ {\prime} | d ^ {\prime}, x ^ {\prime}, z ^ {\prime}),
$$

which we may interpret as the inner product of a space $\mathcal { H } _ { \mu } \subset \mathcal { H }$ . In words, the kernel of ${ \mathcal { H } } _ { \mu }$ is simply the kernel of  after integrating out $( w , w ^ { \prime } )$ according to the conditional distribution $\mathbb { P } ( w | d , x , z )$ from the integral equation. Formally, ${ \mathcal { H } } _ { \mu }$ can be viewed as an RKHS of functions evaluated on mean embeddings instead of features [Szabó et al., 2016]. Assumption 7 (Smoothness and spectral decay for confounding bridge). Assume $h _ { 0 } \in \mathcal { H } _ { \mu } ^ { c }$ and $\eta _ { j } ( \mathcal { H } _ { \mu } ) \leq C j ^ { - b }$

As a technical aside, an analyst may introduce additional nonlinearity by enriching the model and enriching this assumption. A richer model would instead allow $\gamma _ { 0 } ( d , x , z ) =$ $H _ { 0 } \mu ( d , x , z )$ where $H _ { 0 } : \mathcal { H }  \mathbb { R }$ is a nonlinear mapping in a richer RKHS. In such case, the smoothness and efective dimension assumptions would be placed on $H _ { 0 }$ rather than $h _ { 0 }$ and a notion of Hölder continuity is required [Szabó et al., 2016, Table 1]. For clarity, I omit this complexity.

Under these conditions, I arrive at the first main result: uniform consistency of the confounding bridge. This result appears to be the first finite sample uniform analysis of nonparametric instrumental variable regression in the RKHS. By allowing $n \neq m$ , I allow the possibility of asymmetric sample splitting and the use of observations from diferent data sets. The finite sample rates are expressed in terms of $( n , m )$ . The parameter $a > 0$ characterizes the ratio between the sample sizes.

Theorem 3 (Consistency of confounding bridge). Suppose Assumptions 2, 4, 5, 6 with $\mathbf { \mathcal { A } } _ { 0 } = \mathcal { W }$ and $\boldsymbol { B } _ { 0 } = \mathcal { D } \times \boldsymbol { \mathcal { X } } \times \boldsymbol { \mathcal { Z } }$ , and 7 hold. Set $\lambda = n ^ { - \frac { 1 } { c _ { 0 } + 1 / b _ { 0 } } }$ and $n = m ^ { \frac { a ( c _ { 0 } + 1 / b _ { 0 } ) } { c _ { 0 } - 1 } }$ where $a > 0$

1. If $a \le ( c + 3 ) / ( c + 1 / b )$ then $\| \hat { h } - h _ { 0 } \| _ { \infty } = O _ { p } ( m ^ { - \frac { 1 } { 2 } \frac { a ( c - 1 ) } { c + 3 } } )$ with $\xi = m ^ { - \frac { a } { c + 3 } }$

2. If $a \ge ( c + 3 ) / ( c + 1 / b )$ then $\| \hat { h } - h _ { 0 } \| _ { \infty } = O _ { p } ( m ^ { - \frac { 1 } { 2 } \frac { c - 1 } { c + 1 / b } } )$ with $\xi = m ^ { - \frac { 1 } { c + 1 / b } }$

See Appendix G for exact finite sample rates and intermediate results in RKHS norm. At $a = ( c + 3 ) / ( c + 1 / b )$ , the convergence rate $m ^ { - \frac { 1 } { 2 } \frac { c - 1 } { c + 1 / b } }$ attains the rate of single stage kernel ridge regression with respect to m [Fischer and Steinwart, 2020]. This rate is calibrated by ${ \mathit { c } } ,$ the smoothness of confounding bridge and b, the efective dimension of its RKHS.

The rate of Theorem 3 requires the ratio between sample sizes to be $n = m ^ { \frac { c + 3 } { c + 1 / b } \cdot \frac { ( c _ { 0 } + 1 / b _ { 0 } ) } { c _ { 0 } - 1 } }$ ， implying $n \gg m$ . In practice, the analyst often uses the same observations to estimate $\hat { \mu } _ { w } ( d , x , z )$ and $\hat { h }$ .

Corollary 1 (Reusing samples). If samples are reused to estimate $\hat { \mu } _ { w } ( d , x , z )$ and $\hat { h } .$ , then $\begin{array} { r } { n = m , a = ( c _ { 0 } - 1 ) / ( c _ { 0 } + 1 / b _ { 0 } ) , \xi = n ^ { - \frac { c _ { 0 } - 1 } { ( c _ { 0 } + 1 / b _ { 0 } ) ( c + 3 ) } } } \end{array}$ , and $\| \hat { h } - h _ { 0 } \| _ { \infty } = O _ { p } ( n ^ { - \frac { 1 } { 2 } \frac { c _ { 0 } - 1 } { c _ { 0 } + 1 / b _ { 0 } } \frac { c - 1 } { c + 3 } } )$

This rate adapts to the smoothness $c _ { 0 }$ of the conditional distribution as well as the smoothness $c$ of the confounding bridge. The slow rate reflects the challenge of a uniform norm guarantee in an ill posed inverse problem. A faster rate could be possible under further assumptions, which I leave to future work.

## 5.2 Treatment efects

Recall from Theorem 2 that $\theta _ { 0 } ^ { A T T }$ and $\theta _ { 0 } ^ { C A T E }$ contain conditional mean embeddings $\mu ( d )$ and $\mu ( v )$ , respectively. I estimate these conditional mean embeddings by regularized projections in Algorithm 2. To control bias and variance, I place smoothness and efective dimension conditions in the sense of (3) and (4) for $\mu ( d )$ and $\mu ( v )$ as well. As previewed in the discussion about $\mu _ { w } ( d , x , z )$ and $E _ { 0 }$ , a conditional mean embedding corresponds to a conditional expectation operator. As before, all one has to do is specify $A _ { \ell }$ and $B _ { \ell }$ to specialize the assumption. For $\mu ( d ) , { \mathcal { A } } _ { 1 } = \chi \times \mathcal { W }$ and $\begin{array} { r } { B _ { 1 } = \mathcal { D } ; } \end{array}$ ; for $\mu ( v ) , \mathcal { A } _ { 2 } = \mathcal { X } \times \mathcal { W }$ and $\begin{array} { r } { B _ { 2 } = \nu } \end{array}$

Under these conditions, I arrive at the second main result: uniform consistency of the treatment efect estimators. For simplicity, I specialize to the scenario with the fastest rates. Recall $\tilde { n }$ is the number of observations drawn from the alternative population $\tilde { \mathbb { P } }$

Theorem 4 (Consistency of treatment efects). Suppose Assumption 1 holds, as well as the conditions of Theorem 3. Set $( \lambda , \lambda _ { 1 } , \lambda _ { 2 } ) = ( n ^ { - { \frac { 1 } { c _ { 0 } + 1 / b _ { 0 } } } } , n ^ { - { \frac { 1 } { c _ { 1 } + 1 / b _ { 1 } } } } , n ^ { - { \frac { 1 } { c _ { 2 } + 1 / b _ { 2 } } } } ) , \xi = m ^ { - { \frac { 1 } { c + 1 / b } } }$ 2 and $n = m ^ { \frac { c + 3 } { c + 1 / b } \cdot \frac { ( c _ { 0 } + 1 / b _ { 0 } ) } { c _ { 0 } - 1 } }$

1. Then

$$
\| \hat {\theta} ^ {A T E} - \theta_ {0} ^ {A T E} \| _ {\infty} = O _ {p} \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}} + n ^ {- \frac {1}{2}}\right).
$$

2. If in addition Assumption 3 holds, then

$$
\| \hat {\theta} ^ {D S} (\cdot , \tilde {\mathbb {P}}) - \theta_ {0} ^ {D S} (\cdot , \tilde {\mathbb {P}}) \| _ {\infty} = O _ {p} \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}} + \tilde {n} ^ {- \frac {1}{2}}\right).
$$

3. If in addition Assumption 6 holds with $\mathcal { A } _ { 1 } = \mathcal { X } \times \mathcal { W }$ and $\boldsymbol { B } _ { 1 } = \boldsymbol { D }$ , then

$$
\| \hat {\theta} ^ {A T T} - \theta_ {0} ^ {A T T} \| _ {\infty} = O _ {p} \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}} + n ^ {- \frac {1}{2} \frac {c _ {1} - 1}{c _ {1} + 1 / b _ {1}}}\right).
$$

4. If in addition Assumption 6 holds with $\mathcal { A } _ { 2 } = \mathcal { X } \times \mathcal { W }$ and $\begin{array} { r } { B _ { 2 } = \mathcal { V } . } \end{array}$ , then

$$
\| \hat {\theta} ^ {C A T E} - \theta_ {0} ^ {C A T E} \| _ {\infty} = O _ {p} \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}} + n ^ {- \frac {1}{2} \frac {c _ {2} - 1}{c _ {2} + 1 / b _ {2}}}\right).
$$

See Appendix H for exact finite sample rates. Inspecting the rates, we see that each one is a sum of the rate for the confounding bridge from Theorem 3 with the rate for the appropriate mean embedding estimation procedure. The rates adapt to the smoothness parameters $( c , c _ { 0 } , c _ { 1 } , c _ { 2 } )$ and efective dimension parameters $( b , b _ { 0 } , b _ { 1 } , b _ { 2 } )$ of the confounding bridge $h _ { 0 }$ and the conditional expectation operators $E _ { 0 } , \ E _ { 1 }$ , and $E _ { 2 }$ . Equivalently, the rates adapt to the smoothness and efective dimension of the confounding bridge $h _ { 0 }$ and the conditional distributions $\mathbb { P } ( w | d , x , z ) , \mathbb { P } ( x , w | d )$ , and $\mathbb { P } ( x , w | v )$

The goal of this project is to propose dose response and heterogeneous treatment efect estimators to ultimately inform policy and medical decisions. For this reason, I prove a uniform guarantee that strictly controls error for any level of treatment, rather than a mean square guarantee that controls error for the average level of treatment. Uniform guarantees come at the cost of slower rates. In negative control treatment efect estimation, the ill posedness of the confounding bridge learning problem compounds this phenomenon. Theorem 4 appears to be the first finite sample analysis for nonparametric negative control treatment efects, and it holds under weak assumptions. Obtaining faster rates, perhaps via further assumptions, is an important direction for future work.

## 6 Simulation and application

## 6.1 Simulations

I evaluate the empirical performance of the new estimators. I focus on dose response with negative controls, and consider various designs with varying sample sizes. Specifically, I compare the new algorithm that uses negative controls (N.C.) with an existing RKHS algorithm for nonparametric treatment efects (T.E.) [Singh et al., 2020] that ignores unobserved confounding and instead classifies negative controls as additional covariates. Whereas the new algorithm involves reweighting a confounding bridge, the previous algorithm involves reweighting a regression. For each design, sample size, and algorithm,

I implement 100 simulations and calculate mean square error (MSE) with respect to the true counterfactual function.

(a) Quadratic

(b) Sigmoid

(c) Peaked  
Figure 2: Simulation results for various designs

Specifically, I adapt the continuous treatment efect design proposed by [Colangelo and Lee, 2020]. Whereas the original setting studied by [Colangelo and Lee, 2020] has no unobserved confounding, my modification does have unobserved confounding. The goal is to learn the counterfactual function $\theta _ { 0 } ^ { A T E } ( d )$ , which may be a quadratic, sigmoid, or peaked function. A single observation consists of the tuple $( Y , W , D , Z , X )$ for outcome, negative control outcome, treatment, negative control treatment, and covariates. $( Y , D )$ are continuous scalars. In the baseline experiment, $X \in \mathbb { R } ^ { 5 }$ and $( Z , W ) \in \mathbb { R }$

To explore the role of sample size, I consider $n \in \{ 1 0 0 , 5 0 0 , 1 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \}$ . To explore the role of dimension, I focus on the quadratic design, fix sample size at $n = 1 0 0 0$ and then vary dim $( X ) \in \{ 1 , 5 , 1 0 , 5 0 , 1 0 0 \}$ , $d i m ( Z ) \in \{ 1 , 5 , 1 0 \}$ , or $d i m ( W ) \in \{ 1 , 5 , 1 0 \}$ This range of sample sizes and dimensions is common in epidemiology research. Figures 2 and 3 visualize results. Across designs, sample sizes, and dimensions, the use of negative controls to adjust for unobserved confounding improves performance. The improvement is generally increasing in n and $d i m ( Z )$ but decreasing in $d i m ( X )$ and dim(W). Intuitively, $( X , W )$ are the variables used in the reweighting step, which is common across the two estimators N.C. and T.E.; as this step becomes relatively more important, the estimators become more similar. See Appendix I for implementation details as well as additional simulations that confirm: (i) robustness to tuning; (ii) improvement when treatment is discrete; and (iii) ineficiency in the absence of unobserved confounding.

(a) Covariate

(b) N.C. treatment

(c) N.C. outcome  
Figure 3: Simulation results for various dimensions

## 6.2 Dose response of cigarette smoking

Estimating the efect of cigarette smoking on infant birth weight is challenging for several reasons. First, pregnant women are classified as a vulnerable population, so they are typically excluded from clinical trials of any kind. When the treatment of interest causes harm, ethical considerations preclude randomization. Therefore observational data are the only option. Second, pregnancy induces many physiological changes, so medical knowledge predicts diferent dose response curves for women who are pregnant compared to women who are not pregnant. For example, plasma volume increases 35%, cardiac output increases 40%, and glomerular filtration rate (a measure of kidney function) increases 50% during pregnancy [Cunningham et al., 2014]. Therefore the shape of the dose response curve for pregnant women is an unknown, nonparametric quantity. Third, medical records exclude an unobserved confounder known to be crucial for maternal-fetal health: household income [Joseph et al., 2007].

In this section, I argue that medical records include variables that satisfy the properties of negative controls for unobserved income. I provide preliminary results and outline directions for future work on this topic. Finally, I discuss what issues may arise if there are additional unobserved confounders. The purpose of this case study is to illustrate how the proposed estimators may be useful in epidemiology research, though the findings are not conclusive.

I estimate the dose response curve of cigarette smoking on infant birth weight using a data set of singleton births in the state of Pennsylvania between 1989 and 1991 assembled by [Almond et al., 2005] and subsequently analyzed by [Cattaneo, 2010]. I focus on Pennsylvania because smoking data are available for over 95% of mothers. I focus on singleton births because multiple gestations reflect a variety of factors and result in diferent fetal growth trajectories. 21% of women report smoking during pregnancy, and I subset to this sample. I consider the subpopulations of (a) nonhispanic white women who smoke $( n = 7 3 , 8 3 4 )$ ， (b) nonhispanic black women who smoke $( n = 1 7 , 6 2 5 )$ , and (c) hispanic women who smoke $( n = 2 , 1 5 2 )$ . Formally, I estimate $\theta _ { 0 } ^ { C A T E } ( d , v )$ where $D \in \mathbb { R }$ is the number of cigarettes smoked per day, and V concatenates mother’s race $V _ { 1 } ~ \in ~ \{ \mathrm { w h i t e } $ , black, hispanic and mother’s smoking status $V _ { 2 } \in \{ 0 , 1 \}$ . See Appendix J for further discussion.

The classification of variables extensively relies on domain knowledge, so I sought the expertise of physicians from the Department of Obstetrics, Gynecology & Reproductive Biology at Harvard Medical School. Together, we arrived at the classification given in Appendix J, based on a canonical textbook [Cunningham et al., 2014]. Figure 4 illustrates the model. Demographics, alcohol consumption, prenatal care, existing medical conditions, county, and year serve as covariates X since they may be associated with both smoking D and birth weight Y.

Education serves as a negative control treatment Z because it reflects unobserved confounding due to household income U but has no direct medical efect on birth weight Y. Formally, we require $Z \underline { { | | } } Y | D , U , X$ : education is independent of birth weight after conditioning on smoking, income, and observed covariates. Prenatal care and weight gain are the observed covariates that, along with smoking and income, justify the conditional independence between education and birth weight.

Infant birth order and sex serve as a negative control outcomes W because family size reflects household income U but is not directly caused by smoking D or education Z. Formally, we require $W \bot \bot D , Z | U , X$ : family size is independent of smoking and education after conditioning on income and observed covariates. Age and marriage status are the observed covariates that, along with income, justify the conditional independence between education and family size. We also include Rh sensitization as a negative control outcome because it is one of the few medical conditions not afected by smoking (it is caused by blood type).

(a) White smoking mothers

(b) Black smoking mothers

Figure 5: Efect of cigarette smoking on birth weight for diferent subpopulations; $\theta _ { 0 } ^ { C A T E } ( d , v )$ where $D \in \mathbb { R }$ is the number of cigarettes smoked per day, and V concatenates mother’s race $V _ { 1 } \in \{ \mathrm { w h i t e }$ , black, hispanic and mother’s smoking status $V _ { 2 } \in \{ 0 , 1 \}$ .  
Figure 4: Smoking DAG

I implement both the new algorithm (N.C.) and an existing RKHS algorithm for continuous treatment efect (T.E) [Singh et al., 2020] that ignores unobserved confounding. For the method that ignores unobserved confounding, I classify negative controls as additional covariates. Figures 5 and 6 visualize results for white, black, and hispanic smoking mothers.

The efect of cigarettes smoked per day D on birth weight in grams $Y$ is generally negative with similar shapes across subpopulations. The counterfactual birth weights for black and hispanic mothers are lower than for white mothers when the number of cigarettes is high. The main finding is that using negative controls leads to higher dose response curves. Under the stated causal assumptions, the gap between N.C. and T.E. is the magnitude of unobserved confounding due to income. These preliminary results support the clinical hypothesis that poverty is an unmeasured confounder that afects infant birth weight. Unobserved poverty may substantially mislead observational studies that fail to account for it. See Appendix J for implementation details.

An unanticipated result is that the dose response curves appear nonmonotonic; estimated counterfactual birth weight increases before it decreases. This phenomenon prevails across subpopulations, and it can be seen in not only N.C. but also T.E. and the raw data. We propose two conjectures based on the data and domain knowledge. Both of these conjectures are ways in which the data generating process may violate the causal assumptions in Assumption 1.

First, it may be that measurement error contaminates observations. In the raw data, it appears that when the number of cigarettes was between one and 10 it may have been rounded up to 10. Indeed, Figures 5 and 6 document substantial point masses at multiples of 10. This phenomenon would violate our causal model,

(c) Hispanic smoking mothers

Figure 6: Efect of cigarette smoking on birth weight for diferent subpopulations; $\theta _ { 0 } ^ { C A T E } ( d , v )$ where $D \in$ R is the number of cigarettes smoked per day, and V concatenates mother’s race $V _ { 1 } ~ \in ~ \{ \mathrm { w h i t e } $ , black, hispanic and mother’s smoking status $V _ { 2 } \in \{ 0 , 1 \}$

since it would mean that when the true treatment value d was less than 10, we observe $D = 1 0$ $Z = z .$ , yet $Y = Y ^ { ( d , z ) }$ . In such case, estimates of the dose response for $d < 1 0$ may be unreliable. How to account for measurement error in negative control estimation remains an open question.

Second, it could be that another unobserved confounder exists, is not detected by the negative controls, and disproportionately afects women who reported smoking less than 10 cigarettes. Previous studies suggest that rural-urban classification, poverty, and psychosocial stress are possible confounders [Hobel et al., 2008]. In our analysis, we account for rural-urban classification as an observed covariate and we account for poverty via negative controls, but we did not find plausible negative controls for stress in this data set; see Appendix J for further discussion. Indeed, psychosocial stress is notoriously dificult to measure, and it may cause both smoking and low birth weight. We pose for future work a further analysis that adjusts for unobserved confounding due to both income and stress.

## 7 Conclusion

I propose a new family of nonparametric algorithms for learning treatment efects with negative controls. The estimators are easily implemented and uniformly consistent. As a contribution to the negative control literature, I propose methods to estimate dose response curves and heterogeneous treatment efects under the assumption that treatment efects are smooth. As a contribution to the kernel methods literature, I show how the RKHS is well suited to causal inference in the presence of unobserved confounding. As a contribution to maternal-fetal medicine, I propose a toolkit for estimating dose response curves for pregnant women from medical records despite unobserved confounding. The results suggest that RKHS methods may be an efective bridge between epidemiology and machine learning.

## References

[Abadie, 2005] Abadie, A. (2005). Semiparametric diference-in-diferences estimators. The Review of Economic Studies, 72(1):1–19.

[Abrevaya et al., 2015] Abrevaya, J., Hsu, Y.-C., and Lieli, R. P. (2015). Estimating conditional average treatment efects. Journal of Business & Economic Statistics, 33(4):485– 505.

[Ai and Chen, 2003] Ai, C. and Chen, X. (2003). Eficient estimation of models with conditional moment restrictions containing unknown functions. Econometrica, 71(6):1795– 1843.

[Almond et al., 2005] Almond, D., Chay, K. Y., and Lee, D. S. (2005). The costs of low birth weight. The Quarterly Journal of Economics, 120(3):1031–1083.

[Angrist et al., 1996] Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of causal efects using instrumental variables. Journal of the American Statistical Association, 91(434):444–455.

[Athey and Imbens, 2006] Athey, S. and Imbens, G. W. (2006). Identification and inference in nonlinear diference-in-diferences models. Econometrica, 74(2):431–497.

[Berkson, 1958] Berkson, J. (1958). Smoking and lung cancer: Some observations on two recent reports. Journal of the American Statistical Association, 53(281):28–38.

[Berlinet and Thomas-Agnan, 2011] Berlinet, A. and Thomas-Agnan, C. (2011). Reproducing Kernel Hilbert Spaces in Probability and Statistics. Springer Science & Business Media.

[Blundell et al., 2007] Blundell, R., Chen, X., and Kristensen, D. (2007). Seminonparametric IV estimation of shape-invariant Engel curves. Econometrica, 75(6):1613– 1669.

[Caponnetto and De Vito, 2007] Caponnetto, A. and De Vito, E. (2007). Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7(3):331–368.

[Card, 1990] Card, D. (1990). The impact of the Mariel boatlift on the Miami labor market. Industrial and Labor Relations Review, 43(2):245–257.

[Carrasco et al., 2007] Carrasco, M., Florens, J.-P., and Renault, E. (2007). Linear inverse problems in structural econometrics estimation based on spectral decomposition and regularization. Handbook of Econometrics, 6:5633–5751.

[Cattaneo, 2010] Cattaneo, M. D. (2010). Eficient semiparametric estimation of multivalued treatment efects under ignorability. Journal of Econometrics, 155(2):138–154.

[Chen and Christensen, 2018] Chen, X. and Christensen, T. M. (2018). Optimal sup-norm rates and uniform inference on nonlinear functionals of nonparametric IV regression. Quantitative Economics, 9(1):39–84.

[Chen and Pouzo, 2012] Chen, X. and Pouzo, D. (2012). Estimation of nonparametric conditional moment models with possibly nonsmooth generalized residuals. Econometrica, 80(1):277–321.

[Chen and Reiss, 2011] Chen, X. and Reiss, M. (2011). On rate optimality for ill-posed inverse problems in econometrics. Econometric Theory, 27(3):497–521.

[Chernozhukov et al., 2021] Chernozhukov, V., Newey, W. K., and Singh, R. (2021). A simple and general debiased machine learning theorem with finite sample guarantees. arXiv:2105.15197.

[Colangelo and Lee, 2020] Colangelo, K. and Lee, Y.-Y. (2020). Double debiased machine learning nonparametric inference with continuous treatments. arXiv:2004.03036.

[Cucker and Smale, 2002] Cucker, F. and Smale, S. (2002). On the mathematical foundations of learning. Bulletin of the American Mathematical Society, 39(1):1–49.

[Cui et al., 2020] Cui, Y., Pu, H., Shi, X., Miao, W., and Tchetgen Tchetgen, E. J. (2020). Semiparametric proximal causal inference. arXiv:2011.08411.

[Cunningham et al., 2014] Cunningham, F. G., Leveno, K. J., Bloom, S. L., Spong, C. Y., Dashe, J. S., Hofman, B. L., Casey, B. M., and Shefield, J. S. (2014). Williams Obstetrics, volume 7. McGraw-Hill Medical New York.

[Darolles et al., 2011] Darolles, S., Fan, Y., Florens, J.-P., and Renault, E. (2011). Nonparametric instrumental regression. Econometrica, 79(5):1541–1565.

[Deaner, 2018] Deaner, B. (2018). Proxy controls and panel data. arXiv:1810.00283.

[Dikkala et al., 2020] Dikkala, N., Lewis, G., Mackey, L., and Syrgkanis, V. (2020). Minimax estimation of conditional moment models. Advances in Neural Information Processing Systems, 33:12248–12262.

[Fischer and Steinwart, 2020] Fischer, S. and Steinwart, I. (2020). Sobolev norm learning rates for regularized least-squares algorithms. Journal of Machine Learning Research, 21:205–1.

[Gagnon-Bartsch and Speed, 2012] Gagnon-Bartsch, J. A. and Speed, T. P. (2012). Using control genes to correct for unwanted variation in microarray data. Biostatistics, 13(3):539–552.

[Ghassami et al., 2021] Ghassami, A., Ying, A., Shpitser, I., and Tchetgen Tchetgen, E. J. (2021). Minimax kernel machine learning for a class of doubly robust functionals. arXiv:2104.02929.

[Hall and Horowitz, 2005] Hall, P. and Horowitz, J. L. (2005). Nonparametric methods for inference in the presence of instrumental variables. The Annals of Statistics, 33(6):2904– 2929.

[Hill, 1965] Hill, A. B. (1965). The environment and disease: Association or causation? Proceedings of the Royal Society of Medicine, 58:295–300.

[Hobel et al., 2008] Hobel, C. J., Goldstein, A., and Barrett, E. S. (2008). Psychosocial stress and pregnancy outcome. Clinical Obstetrics and Gynecology, 51(2):333–348.

[Horowitz and Lee, 2005] Horowitz, J. L. and Lee, S. (2005). Nonparametric estimation of an additive quantile regression model. Journal of the American Statistical Association, 100(472):1238–1249.

[Joseph et al., 2007] Joseph, K., Liston, R. M., Dodds, L., Dahlgren, L., and Allen, A. C. (2007). Socioeconomic status and perinatal outcomes in a setting with universal access to essential health care services. Canadian Medical Association Journal, 177(6):583–590.

[Kallus et al., 2021] Kallus, N., Mao, X., and Uehara, M. (2021). Causal inference under unmeasured confounding with negative controls: A minimax learning approach. arXiv:2103.14029.

[Kress, 1989] Kress, R. (1989). Linear Integral Equations, volume 3. Springer.

[Kuroki and Pearl, 2014] Kuroki, M. and Pearl, J. (2014). Measurement bias and efect restoration in causal inference. Biometrika, 101(2):423–437.

[Lipsitch et al., 2010] Lipsitch, M., Tchetgen Tchetgen, E. J., and Cohen, T. (2010). Negative controls: A tool for detecting confounding and bias in observational studies. Epidemiology, 21(3):383.

[Lousdal et al., 2020] Lousdal, M. L., Lash, T. L., Flanders, W. D., Brookhart, M. A., Kristiansen, I. S., Kalager, M., and Støvring, H. (2020). Negative controls to detect uncontrolled confounding in observational studies of mammographic screening comparing participants and non-participants. International Journal of Epidemiology.

[Mastouri et al., 2021] Mastouri, A., Zhu, Y., Gultchin, L., Korba, A., Silva, R., Kusner, M. J., Gretton, A., and Muandet, K. (2021). Proximal causal learning with kernels: Two-stage estimation and moment restriction. arXiv:2105.04544.

[Meyer, 1995] Meyer, B. D. (1995). Natural and quasi-experiments in economics. Journal of Business & Economic Statistics, 13(2):151–161.

[Miao et al., 2018] Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. (2018). Identifying causal efects with proxy variables of an unmeasured confounder. Biometrika, 105(4):987– 993.

[Miao and Tchetgen Tchetgen, 2018] Miao, W. and Tchetgen Tchetgen, E. J. (2018). A confounding bridge approach for double negative control inference on causal efects. arXiv:1808.04945.

[Newey, 1994] Newey, W. K. (1994). Kernel estimation of partial means and a general variance estimator. Econometric Theory, pages 233–253.

[Newey and Powell, 2003] Newey, W. K. and Powell, J. L. (2003). Instrumental variable estimation of nonparametric models. Econometrica, 71(5):1565–1578.

[Nie and Wager, 2021] Nie, X. and Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment efects. Biometrika, 108(2):299–319.

[Rosenbaum, 1989] Rosenbaum, P. R. (1989). The role of known efects in observational studies. Biometrics, 45(2):557–569.

[Shi et al., 2020] Shi, X., Miao, W., Nelson, J. C., and Tchetgen Tchetgen, E. J. (2020). Multiply robust causal inference with double-negative control adjustment for categorical unmeasured confounding. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(2):521–540.

[Singh, 2020] Singh, R. (2020). Kernel methods for unobserved confounding: Negative controls, proxies, and instruments. arXiv:2012.10315.

[Singh et al., 2019] Singh, R., Sahani, M., and Gretton, A. (2019). Kernel instrumental variable regression. In Advances in Neural Information Processing Systems, pages 4595– 4607.

[Singh et al., 2020] Singh, R., Xu, L., and Gretton, A. (2020). Kernel methods for causal functions: Dose, heterogeneous, and incremental response curves. arXiv:2010.04855.

[Smale and Zhou, 2005] Smale, S. and Zhou, D.-X. (2005). Shannon sampling II: Connections to learning theory. Applied and Computational Harmonic Analysis, 19(3):285–302.

[Smale and Zhou, 2007] Smale, S. and Zhou, D.-X. (2007). Learning theory estimates via integral operators and their approximations. Constructive Approximation, 26(2):153– 172.

[Sofer et al., 2016] Sofer, T., Richardson, D. B., Colicino, E., Schwartz, J., and Tchetgen Tchetgen, E. J. (2016). On negative outcome control of unobserved confounding as a generalization of diference-in-diferences. Statistical Science, 31(3):348.

[Sriperumbudur et al., 2010] Sriperumbudur, B., Fukumizu, K., and Lanckriet, G. (2010). On the relation between universality, characteristic kernels and RKHS embedding of measures. In International Conference on Artificial Intelligence and Statistics, pages 773–780.

[Steinwart and Christmann, 2008] Steinwart, I. and Christmann, A. (2008). Support Vector Machines. Springer Science & Business Media.

[Sutherland, 2017] Sutherland, D. J. (2017). Fixing an error in Caponnetto and de Vito (2007). arXiv:1702.02982.

[Szabó et al., 2016] Szabó, Z., Sriperumbudur, B. K., Póczos, B., and Gretton, A. (2016). Learning theory for distribution regression. The Journal of Machine Learning Research, 17(1):5272–5311.

[Tchetgen Tchetgen, 2014] Tchetgen Tchetgen, E. J. (2014). The control outcome calibration approach for causal inference with unobserved confounding. American Journal of Epidemiology, 179(5):633–640.

[Tchetgen Tchetgen et al., 2020] Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., and Miao, W. (2020). An introduction to proximal causal learning. arXiv:2009.10982.

[Wang et al., 2017] Wang, J., Zhao, Q., Hastie, T., and Owen, A. B. (2017). Confounder adjustment in multiple hypothesis testing. Annals of Statistics, 45(5):1863.

[Weiss, 2002] Weiss, N. S. (2002). Can the “specificity” of an association be rehabilitated as a basis for supporting a causal hypothesis? Epidemiology, 13(1):6–8.

[Yerushalmy and Palmer, 1959] Yerushalmy, J. and Palmer, C. E. (1959). On the methodology of investigations of etiologic factors in chronic diseases. Journal of Chronic Diseases, 10(1):27–40.

## A Semiparametric inference

While the main contribution of this paper is uniform consistency in nonparametric settings, in this section I present complementary results for semiparametric settings. In particular, I articulate conditions that imply Gaussian approximation and valid confidence intervals for the setting with binary treatment, appealing to the semiparametric theory of [Cui et al., 2020, Kallus et al., 2021, Ghassami et al., 2021, Chernozhukov et al., 2021].

To lighten notation, fix the treatment value $d ^ { \prime } \in \{ 0 , 1 \}$ and denote $\theta _ { 0 } = \theta _ { 0 } ^ { A T E } ( d ^ { \prime } )$ Observe that for each treatment value,

$$
\theta_ {0} = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (x, w) = \int h _ {0} (d, x, w) \tau_ {0} (d, x, w) \mathrm{d} \mathbb {P} (d, x, w)
$$

where by standard propensity score arguments

$$
\tau_ {0} (d, x, w) = \frac {1 \{d = d ^ {\prime} \}}{\pi_ {0} (x , w)}, \quad \pi_ {0} (x, w) = \mathbb {P} (D = d ^ {\prime} | X = x, W = w).
$$

Just as the primary confounding bridge $h _ { 0 }$ is defined as the solution to the operator equation

$$
\gamma_ {0} (d, x, z) = \mathbb {E} [ h (D, X, W) | D = d, X = x, Z = z ]
$$

one may define a secondary confounding bridge $\alpha _ { 0 }$ as the solution to the operator equation

$$
\tau_ {0} (d, x, w) = \mathbb {E} [ \alpha (D, X, Z) | D = d, X = x, W = w ].
$$

Proposition 3 (Secondary confounding bridge). Suppose Assumptions 1 and 2 hold. In addition, suppose there exists a solution $\alpha _ { 0 }$ to the operator equation

$$
\tau_ {0} (d, x, w) = \mathbb {E} [ \alpha (D, X, Z) | D = d, X = x, W = w ].
$$

Then

$$
\theta_ {0} = \int y \alpha_ {0} (d, x, z) \mathrm{d} \mathbb {P} (y, d, x, z).
$$

Proof. The result follows from the law of iterated expectations and the definitions of the

primary and secondary confounding bridges. Write

$$
\begin{array}{l} \theta_ {0} = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (x, w) \\ \quad = \int h _ {0} (d, x, w) \tau_ {0} (d, x, w) \mathrm{d} \mathbb {P} (d, x, w) \\ \quad = \int h _ {0} (d, x, w) \mathbb {E} [ \alpha_ {0} (D, X, Z) | D = d, X = x, W = w ] \mathrm{d} \mathbb {P} (d, x, w) \\ \quad = \int h _ {0} (d, x, w) \alpha_ {0} (d, x, z) \mathrm{d} \mathbb {P} (d, z, x, w) \\ \quad = \int \mathbb {E} [ h _ {0} (D, X, W) | D = d, X = x, Z = z ] \alpha_ {0} (d, x, z) \mathrm{d} \mathbb {P} (d, x, z) \\ \quad = \int \gamma_ {0} (d, x, z) \alpha_ {0} (d, x, z) \mathrm{d} \mathbb {P} (d, x, z) \\ \quad = \int y \alpha_ {0} (d, x, z) \mathrm{d} \mathbb {P} (y, d, z, x). \end{array}
$$

The secondary confounding bridge formulation in Proposition 3 generalizes the familiar inverse propensity weight formulation for treatment efects. Instead of using the propensity score $\pi _ { 0 }$ encoded by $\tau _ { 0 } .$ , the formulation uses a secondary confounding bridge $\alpha _ { 0 }$ defined as the solution to an ill posed inverse problem that involves the propensity score. Analogously, the primary confounding bridge formulation in Theorem 1 generalizes the familiar g-formula for treatment efects, using the confounding bridge $h _ { 0 }$ rather than the regression $\gamma _ { 0 }$

Both the primary and secondary confounding bridges $\left( h _ { 0 } , \alpha _ { 0 } \right)$ appear in the semiparametrically eficient asymptotic variance, so it is natural to incorporate them into semiparametric estimation and inference [Cui et al., 2020]. The targeted and debiased machine learning literatures provide a meta algorithm to do so, as well as rate conditions that suffice for Gaussian approximation. I quote the meta algorithm and rate conditions, adapting them to the treatment efect $\theta _ { 0 }$ identified by negative controls.

Algorithm 4 (Debiased machine learning). Given a sample $( Y _ { i } , W _ { i } , D _ { i } , X _ { i } , Z _ { i } ) ( i = 1 , . . . , n )$ partition the sample into folds $( I _ { \ell } ) \ ( \ell = 1 , . . . , L )$ . Denote by $I _ { \ell } ^ { c }$ the complement of $I _ { \ell }$ .

1. For each fold $\ell ,$ estimate $\hat { h } _ { \ell }$ and $\hat { \alpha } _ { \ell }$ from observations in $I _ { \ell } ^ { c }$

2. Estimate $\begin{array} { r } { \hat { \theta } = n ^ { - 1 } \sum _ { \ell = 1 } ^ { L } \sum _ { i \in I _ { \ell } } [ \hat { h } _ { \ell } ( d , X _ { i } , W _ { i } ) + \hat { \alpha } _ { \ell } ( D _ { i } , X _ { i } , Z _ { i } ) \{ Y _ { i } - \hat { h } _ { \ell } ( D _ { i } , X _ { i } , W _ { i } ) \} ] . } \end{array}$

3. Estimate its 95% confidence interval as $\hat { \theta } \pm 1 . 9 6 \hat { \sigma } n ^ { - 1 / 2 }$ , where

$$
\hat {\sigma} ^ {2} = n ^ {- 1} \sum_ {\ell = 1} ^ {L} \sum_ {i \in I _ {\ell}} [ \hat {h} _ {\ell} (d, X _ {i}, W _ {i}) + \hat {\alpha} _ {\ell} (D _ {i}, X _ {i}, Z _ {i}) \{Y _ {i} - \hat {h} _ {\ell} (D _ {i}, X _ {i}, W _ {i}) \} - \hat {\theta} ] ^ {2}.
$$

Towards a formal statement of the inference result, define the following moments:

$$
\sigma^ {2} = \mathbb {E} [ \psi_ {0} (Y, W, D, Z, X) ^ {2} ], \quad \chi^ {3} = \mathbb {E} [ | \psi_ {0} (Y, W, D, Z, X) | ^ {3} ], \quad \omega^ {4} = \mathbb {E} [ \psi_ {0} (Y, W, D, Z, X) ^ {4} ],
$$

where

$$
\psi_ {0} (Y, W, D, Z, X) = h _ {0} (d, X, W) + \alpha_ {0} (D, X, Z) \{Y - h _ {0} (D, X, W) \} - \theta_ {0}
$$

is the asymptotic influence of each observation. Next, I define mean square error.

Definition 2 (Mean square error). Write the mean square error $\mathcal { R } ( \hat { h } _ { \ell } )$ and the projected mean square error $\mathcal { P } ( \hat { h } _ { \ell } )$ of $\hat { h } _ { \ell }$ trained on observations indexed by $I _ { \ell } ^ { c }$ as

$$
\mathcal {R} (\hat {h} _ {\ell}) = \mathbb {E} [ \{\hat {h} _ {\ell} (D, X, W) - h _ {0} (D, X, W) \} ^ {2} \mid I _ {\ell} ^ {c} ]
$$

$$
\mathcal {P} (\hat {h} _ {\ell}) = \mathbb {E} ([ \mathbb {E} \{\hat {h} _ {\ell} (D, X, W) - h _ {0} (D, X, W) \mid D, X, Z, I _ {\ell} ^ {c} \} ] ^ {2} \mid I _ {\ell} ^ {c}).
$$

Likewise define $\mathcal { R } ( \hat { \alpha } _ { \ell } )$ and $\mathcal { P } ( \hat { \alpha } _ { \ell } )$

$$
\mathcal {R} (\hat {\alpha} _ {\ell}) = \mathbb {E} [ \{\hat {\alpha} _ {\ell} (D, X, Z) - \alpha_ {0} (D, X, Z) \} ^ {2} | I _ {\ell} ^ {c} ]
$$

$$
\mathcal {P} (\hat {\alpha} _ {\ell}) = \mathbb {E} ([ \mathbb {E} \{\hat {\alpha} _ {\ell} (D, X, Z) - \alpha_ {0} (D, X, Z) \mid D, X, W, I _ {\ell} ^ {c} \} ] ^ {2} \mid I _ {\ell} ^ {c}).
$$

Suficiently fast rates of mean square error and projected mean square error imply Gaussian approximation. The following lemma summarizes the rate conditions.

Lemma 1 (Semiparametric inference; Corollary 5.1 of [Chernozhukov et al., 2021]). Assume the propensity score $\pi _ { 0 } ( x , w )$ is bounded away from zero and one and the following regularity conditions hold for some absolute constant $C < \infty \colon$

$$
\mathbb {E} \left[ \left\{Y - h _ {0} (D, X, W) \right\} ^ {2} \mid D, X, W \right] \leq C, \quad \| \alpha_ {0} \| _ {\infty} \leq C, \quad \| \hat {\alpha} _ {\ell} \| _ {\infty} \leq C.
$$

Further assume the following learning rate conditions, as $n \to \infty$

1. (χ/σ)<sup>3</sup> + ω<sup>2</sup> n−<sup>1/2</sup> = o(1);

2. $\{ \mathcal { R } ( \hat { h } _ { \ell } ) \} ^ { 1 / 2 } = o _ { p } ( 1 )$ and $\{ \mathcal { R } ( \hat { \alpha } _ { \ell } ) \} ^ { 1 / 2 } = o _ { p } ( 1 )$ ;

$$
3. \{n \mathcal {R} (\hat {h} _ {\ell}) \mathcal {R} (\hat {\alpha} _ {\ell}) \} ^ {1 / 2} \wedge \{n \mathcal {P} (\hat {h} _ {\ell}) \mathcal {R} (\hat {\alpha} _ {\ell}) \} ^ {1 / 2} \wedge \{n \mathcal {R} (\hat {h} _ {\ell}) \mathcal {P} (\hat {\alpha} _ {\ell}) \} ^ {1 / 2} = o _ {p} (1).
$$

Then the estimator $\hat { \theta }$ in Algorithm 4 is consistent and asymptotically Gaussian, and the confidence interval in Algorithm 4 includes $\theta _ { 0 }$ with probability approaching the nominal level. Formally,

$$
\hat {\theta} \stackrel {p} {\rightarrow} \theta_ {0}, \quad \sigma^ {- 1} n ^ {1 / 2} (\hat {\theta} - \theta_ {0}) \stackrel {d} {\rightarrow} \mathcal {N} (0, 1), \quad \mathbb {P} \left\{\theta_ {0} \in \left(\hat {\theta} \pm 1. 9 6 \hat {\sigma} n ^ {- 1 / 2}\right)\right\}\rightarrow 0. 9 5.
$$

In summary, the algorithmic techniques developed in this paper may be combined with semiparametric theory as long as the rate conditions in Lemma 1 are satisfied. Theorem 3 in the main text provides a sup norm guarantee for the primary confounding bridge $\hat { h } .$ , which implies slow rates of $\mathcal { P } ( \hat { h } _ { \ell } )$ and $\mathcal { R } ( \hat { h } _ { \ell } )$ . [Singh et al., 2019, Theorem 4] directly provides fast rates of $\mathcal { P } ( \hat { h } _ { \ell } )$ . [Kallus et al., 2021, Theorems 4 and 8] and [Ghassami et al., 2021, Theorem 5 and Lemma 1] provide fast rates of $\mathcal { P } ( \hat { \alpha } _ { \ell } )$ and slow rates of $\mathcal { R } ( \hat { \alpha } _ { \ell } )$ for minimax kernel estimators of the secondary confounding bridge. An interesting direction for future work would be to develop a kernel ridge regression estimator for the secondary confounding bridge.

## B Relevance and existence

In this appendix, I revisit the high level conditions in Assumption 2. These conditions are standard in the negative control and instrumental variable literatures. I begin by illustrating how irrelevance of negative controls violates existence. Then I characterize existence and completeness in the RKHS. This characterization appears to be absent from previous work on nonparametric instrumental variable regression in the RKHS.

## B.1 Relevance

Existence of the confounding bridge is fundamentally connected to the relevance of negative controls, as articulated in Proposition 1. The heart of the argument is as follows.

Lemma 2 (Rephrasing existence). Suppose Assumption 1 holds. The existence condition in Assumption 2 holds if and only if there exists a solution $h _ { 0 }$ to the operator equation

$$
\gamma_ {0} (d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u | d, x, z).
$$

$$
W \bot D, Z | U, X.
$$

Hence

$$
\begin{array}{r l} & {\mathbb {P} (w | d, x, z) = \int \mathbb {P} (w, u | d, x, z) \mathrm{d} u} \\ & {\qquad = \int \mathbb {P} (w | u, d, x, z) \mathrm{d} \mathbb {P} (u | d, x, z)} \\ & {\qquad = \int \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u | d, x, z).} \end{array}
$$

Therefore

$$
\int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u | d, x, z).
$$

□

For interpretation, consider the special case where $( U , Z , W )$ are discrete with finite supports $( \mathcal { U } , \mathcal { Z } , \mathcal { W } )$ , respectively. Then for any fixed $( \bar { d } , \bar { x } )$ , the following representations are possible [Miao et al., 2018, Shi et al., 2020]:

$$
\gamma_ {0} (\bar {d}, \bar {x}, z) \in \mathbb {R} ^ {1 \times | \mathcal {Z} |}, \quad h _ {0} (\bar {d}, \bar {x}, w) \in \mathbb {R} ^ {1 \times | \mathcal {W} |}, \quad \mathbb {P} (w | u, \bar {x}) \in \mathbb {R} ^ {| \mathcal {W} | \times | \mathcal {U} |}, \quad \mathbb {P} (u | \bar {d}, \bar {x}, z) \in \mathbb {R} ^ {| \mathcal {U} | \times | \mathcal {Z} |}.
$$

For this special case, it is clear from the expression in Lemma 2

$$
\gamma_ {0} (\bar {d}, \bar {x}, z) = \int h _ {0} (\bar {d}, \bar {x}, w) \mathrm{d} \mathbb {P} (w | u, \bar {x}) \mathrm{d} \mathbb {P} (u | \bar {d}, \bar {x}, z)
$$

that there exists a solution $h _ { 0 }$ when $| \mathcal { W } | \ge | \mathcal { U } | , | \mathcal { Z } | \ge | \mathcal { U } |$ , and both matrices $\mathbb { P } ( w | u , \bar { x } )$ and $\mathbb { P } ( u | \bar { d } , \bar { x } , z )$ are full rank for any fixed values $( \bar { d } , \bar { x } )$ [Shi et al., 2020, Lemma 1]. $| \mathcal { W } | \geq$ $| \mathcal { U } |$ and $| { \mathcal { Z } } | \geq | { \mathcal { U } } |$ mean that the negative controls are expressive enough relative to the unobserved confounder. Full rank $\mathbb { P } ( w | u , \bar { x } )$ and $\mathbb { P } ( u | \bar { d } , \bar { x } , z )$ mean that the conditional distributions are non-degenerate; variation in the negative controls is relevant for recovering variation in the unobserved confounder. When $| \mathcal { W } | > | \mathcal { U } | \mathrm { ~ o r ~ } | \mathcal { Z } | > | \mathcal { U } |$ , the solution $h _ { 0 }$ is not unique. Nonetheless, the completeness condition ensures that treatment efects are point identified.<sup>3</sup>

Next, I interpret the special case in which $( U , Z , W )$ are continuous with supports $( \mathcal { U } , \mathcal { Z } , \mathcal { W } )$ , respectively. I uncover the sense in which the existence assumption may be quite stringent. As before, fix $( \bar { d } , \bar { x } )$ . Now, fix $( z , z ^ { \prime } )$ such that $\gamma _ { 0 } ( \bar { d } , \bar { x } , z ) \neq \gamma _ { 0 } ( \bar { d } , \bar { x } , z ^ { \prime } )$ . If, for these choices, $\mathbb { P } ( u | \bar { d } , \bar { x } , z ) = \mathbb { P } ( u | \bar { d } , \bar { x } , z ^ { \prime } )$ , then by Lemma 2 the confounding bridge does not exist. In other words, to violate Assumption 2, there simply needs to exist some $( \bar { d } , \bar { x } )$ stratum such that $\gamma _ { 0 }$ takes on diferent values at z versus $z ^ { \prime }$ yet the conditional densities of unobserved confounding coincide.

Finally, I prove the result given in the main text.

Proof of Proposition 1. With Lemma 2, I prove each claim separately.

1. $Z \bot U | D ,$ X means that $\mathbb { P } ( u | d , x , z ) = \mathbb { P } ( u | d , x )$ . Towards a contradiction, suppose $h _ { 0 }$ exists. Then by Lemma 2,

$$
\gamma_ {0} (d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u | d, x).
$$

The RHS does not depend on $z ,$ implying $\gamma _ { 0 } ( d , x , z ) = \gamma _ { 0 } ( d , x , z ^ { \prime } )$ , which violates the hypothesis that $\gamma _ { 0 }$ varies in z.

2. W $\bot \vert { U } \vert { X }$ means that $\mathbb { P } ( w | u , x ) = \mathbb { P } ( w | x )$ . Towards a contradiction, suppose $h _ { 0 }$ exists. Then by Lemma 2,

$$
\gamma_ {0} (d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | x) \mathrm{d} \mathbb {P} (u | d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | x).
$$

The RHS does not depend on z, implying $\gamma _ { 0 } ( d , x , z ) = \gamma _ { 0 } ( d , x , z ^ { \prime } )$ , which violates the hypothesis that $\gamma _ { 0 }$ varies in z.

## B.2 Existence

Next I revisit the existence condition. In particular, I present (i) a lemma to characterize existence in ill posed inverse problems; (ii) a proposition verifying existence in the nonparametric negative control problem; and (iii) a proposition verifying existence in the RKHS negative control problem.

## B.2.1 Picard’s criterion

Lemma 3 (Picard’s criterion; Theorem 15.18 of [Kress, 1989]). Let $K : H _ { 1 }  H _ { 2 }$ be a compact operator from the Hilbert space $H _ { 1 }$ to the Hilbert space $H _ { 2 }$ , with singular value decomposition $( e _ { k } ^ { 1 } , \eta _ { k } , e _ { k } ^ { 2 } ) _ { k = 1 } ^ { \infty }$ . Given the function $g \in H _ { 2 }$ , the equation $K f \ = \ g$ has a solution if and only if

1. Inclusion: $g \in \mathcal { N } ( K ^ { * } ) ^ { \perp }$ i.e. g is an element of the orthogonal complement to the null space of the adjoint operator $K ^ { * }$ ;

2. Penalized square summability: $\begin{array} { r } { \sum _ { k = 1 } ^ { \infty } \eta _ { k } ^ { - 2 } \langle g , e _ { k } ^ { 2 } \rangle _ { H _ { 2 } } < \infty } \end{array}$

Note that $e _ { k } ^ { 1 } \in H _ { 1 } , e _ { k } ^ { 2 } \in H _ { 2 }$ , and $K ^ { * } : H _ { 2 } \to H _ { 1 }$ by construction. Various works use this technical lemma to prove existence of the confounding bridge [Miao et al., 2018, Deaner, 2018]. Inclusion essentially means that $g$ is in the appropriate row space. Penalized square summability means that K−g has a finite norm since $g$ is not too aligned with the right singular functions of K relative to the spectral decay.

In what follows, I provide conditions in $\mathbb { L } _ { 2 }$ and subsequently conditions in the RKHS that verify the abstract conditions in Lemma 3. The RKHS conditions are stronger versions of the $\mathbb { L } _ { 2 }$ conditions, analogous to how the RKHS is a subset of $\mathbb { L } _ { 2 }$ as we have seen in Section 3.

## B.2.2 Previous work

I quote a proposition verifying existence of the confounding bridge in the negative control problem. To begin, fix the values $( \bar { d } , \bar { x } )$ . Given $( \bar { d } , \bar { x } )$ , denote by $\mathbb { L } _ { 2 } ( \mathbb { P } ( w | \bar { d } , \bar { x } ) )$ the space of functions $g : \mathcal { W } \to \mathbb { R }$ that are square integrable with respect to the conditional distribution $\mathbb { P } ( w | \bar { d } , \bar { x } )$ , i.e. $\begin{array} { r } { \int g ( w ) ^ { 2 } \mathrm { d } \mathbb { P } ( w | \bar { d } , \bar { x } ) < \infty } \end{array}$ . Likewise for $\mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ . Then define the operator

$$
E _ {\bar {d}, \bar {x}}: \mathbb {L} _ {2} (\mathbb {P} (w | \bar {d}, \bar {x})) \to \mathbb {L} _ {2} (\mathbb {P} (z | \bar {d}, \bar {x})), \quad f (\cdot) \mapsto \mathbb {E} [ f (W) | D = \bar {d}, X = \bar {x}, Z = \cdot ].
$$

Matching symbols with Lemma 3, $H _ { 1 } = \mathbb { L } _ { 2 } ( \mathbb { P } ( w | \bar { d } , \bar { x } ) ) , H _ { 2 } = \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ , and the adjoint operator is

$$
E _ {\bar {d}, \bar {x}} ^ {*}: \mathbb {L} _ {2} (\mathbb {P} (z | \bar {d}, \bar {x})) \to \mathbb {L} _ {2} (\mathbb {P} (w | \bar {d}, \bar {x})), \quad g (\cdot) \mapsto \mathbb {E} [ g (Z) | D = \bar {d}, X = \bar {x}, W = \cdot ].
$$

Proposition 4 (Existence in $\mathbb { L } _ { 2 }$ for fixed $( \bar { d } , \bar { x } )$ ; Proposition 1 of [Miao et al., 2018]). Fix $( \bar { d } , \bar { x } )$ . Denote by $f ( w | \bar { d } , \bar { x } , z )$ and $f ( z | \bar { d } , \bar { x } , w )$ the densities of P $\ P ( w | \bar { d } , \bar { x } , z )$ and $\mathbb { P } ( z | \bar { d } , \bar { x } , w )$ Suppose

1. Regularity: $\begin{array} { r } { \int f ( w | \bar { d } , \bar { x } , z ) f ( z | \bar { d } , \bar { x } , w ) \mathrm { d } w \mathrm { d } z < \infty ; } \end{array}$

2. Completeness: for any function $g \in \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$

$$
\mathbb {E} [ g (Z) | D = \bar {d}, X = \bar {x}, W = w ] = 0 \quad \forall w \iff g (Z) = 0;
$$

3. Correct specification: $\gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) \in \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ ;

4. Penalized square summability: $\begin{array} { r } { \sum _ { k = 1 } ^ { \infty } \eta _ { k } ^ { - 2 } \langle \gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) , e _ { k } ^ { z } \rangle _ { \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) ) } < \infty } \end{array}$ , where the singular value decomposition exists due to the regularity condition.

Then for any fixed $( \bar { d } , \bar { x } )$ , there exists some $f _ { \bar { d } , \bar { x } } \in \mathbb { L } _ { 2 } ( \mathbb { P } ( w | \bar { d } , \bar { x } ) )$ such that

$$
\gamma_ {0} (\bar {d}, \bar {x}, z) = [ E _ {\bar {d}, \bar {x}} f _ {\bar {d}, \bar {x}} ] (z) = \mathbb {E} [ f _ {\bar {d}, \bar {x}} (W) | D = \bar {d}, X = \bar {x}, Z = z ].
$$

Regularity pertains to the conditional distribution $\mathbb { P } ( w | d , x , z )$ in the integral operator equation. It ensures compactness of the corresponding conditional expectation operator in order to appeal to Lemma 3. Completeness is a technical condition from the NPIV literature which, together with correct specification of $\gamma _ { 0 }$ , implies the inclusion condition of Lemma 3. Penalized square summability is identical to Lemma 3.

Proof. I verify the conditions of Lemma 3. Rewriting the regularity condition,

$$
\int f (w | \bar {d}, \bar {x}, z) f (z | \bar {d}, \bar {x}, w) \mathrm{d} w \mathrm{d} z = \int \frac {f (z , w | \bar {d} , \bar {x}) ^ {2}}{f (w | \bar {d} , \bar {x}) f (z | \bar {d} , \bar {x})} \mathrm{d} w \mathrm{d} z
$$

which is a suficient condition for compactness of $E _ { \bar { d } , \bar { x } }$ [Carrasco et al., 2007, Example 2.3]. By compactness, the singular value decomposition $( e _ { k } ^ { w } , \eta _ { k } , e _ { k } ^ { z } ) _ { k = 1 } ^ { \infty }$ exists, where $e _ { k } ^ { w } \in$ $\mathbb { L } _ { 2 } ( \mathbb { P } ( w | \bar { d } , \bar { x } ) )$ and $e _ { k } ^ { z } \in \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ .

Next, I verify inclusion by appealing to completeness and correct specification. Towards this end, I first argue that $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) ^ { \perp } = \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ . It sufices to show $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) = 0$ , i.e. the null space only consists of the zero function. Consider any $g \in \mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } )$ . By definition of the null space,

$$
\begin{array}{r l} & 0 (\cdot) = [ E _ {\bar {d}, \bar {x}} ^ {*} g ] (\cdot) \\ & \qquad = \mathbb {E} [ g (Z) | D = \bar {d}, X = \bar {x}, W = \cdot ]. \end{array}
$$

By completeness, I conclude that $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) = 0$ . Therefore $\mathcal { N } ( E _ { \bar { d } . \bar { x } } ^ { * } ) ^ { \perp } = \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ . Finally, correct specification implies $\gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) \in \mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ .

Penalized square summability is immediate.

Corollary 2 (Existence in $\mathbb { L } _ { 2 } ;$ Proposition 1 of [Miao et al., 2018]). If the conditions of Proposition 4 hold for each $( \bar { d } , \bar { x } )$ , then there exists a confounding bridge $h _ { 0 }$ such that

$$
\gamma_ {0} (d, x, z) = \mathbb {E} [ h _ {0} (D, X, W) | D = d, X = x, Z = z ].
$$

Proof. For each $( \bar { d } , \bar { x } )$ , set $h _ { 0 } ( \bar { d } , \bar { x } , w ) = f _ { \bar { d } , \bar { x } } ( w )$

## B.2.3 RKHS

In $\mathbb { L } _ { 2 } .$ , I introduced the function spaces $\mathbb { L } _ { 2 } ( \mathbb { P } ( w | \bar { d } , \bar { x } ) )$ and $\mathbb { L } _ { 2 } ( \mathbb { P } ( z | \bar { d } , \bar { x } ) )$ which are induced by conditioning on $( \bar { d } , \bar { x } )$ . Now, I introduce analogous function spaces $\mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ and $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ . In particular, I demonstrate that these function spaces are RKHSs and characterize their kernels. For now, I simply state their kernels. Later in this section, I will demonstrate how these kernels relate to the tensor product RKHS construction.

Recall that the kernels for $\mathcal { H } _ { \mathcal { W } }$ and $\mathcal { H } _ { \mathcal { Z } }$ are $k ( w , w ^ { \prime } )$ and $k ( z , z ^ { \prime } )$ , respectively. The kernels for $\mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ and $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ are

$$
k _ {\bar {d}, \bar {x}} (w, w ^ {\prime}) = k (\bar {d}, \bar {d}) k (\bar {x}, \bar {x}) k (w, w ^ {\prime}), \quad k _ {\bar {d}, \bar {x}} (z, z ^ {\prime}) = k (\bar {d}, \bar {d}) k (\bar {x}, \bar {x}) k (z, z ^ {\prime}).
$$

Define the scalar $c _ { \bar { d } , \bar { x } } = k ( \bar { d } , \bar { d } ) k ( \bar { x } , \bar { x } )$ . Then

$$
k _ {\bar {d}, \bar {x}} (w, w ^ {\prime}) = c _ {\bar {d}, \bar {x}} \cdot k (w, w ^ {\prime}), \quad k _ {\bar {d}, \bar {x}} (z, z ^ {\prime}) = c _ {\bar {d}, \bar {x}} \cdot k (z, z ^ {\prime})
$$

and it is clear that these induced kernels are simply rescaled versions of the original kernels $k ( w , w ^ { \prime } )$ and $k ( z , z ^ { \prime } )$ according to the conditioned value $( \bar { d } , \bar { x } )$ , so they remain positive definite and hence valid. Then define the operator

$$
E _ {\bar {d}, \bar {x}}: \mathcal {H} _ {\mathcal {W}} ^ {\bar {d}, \bar {x}} \to \mathcal {H} _ {\mathcal {Z}} ^ {\bar {d}, \bar {x}}, f (\cdot) \mapsto \mathbb {E} [ f (W) | D = \bar {d}, X = \bar {x}, Z = \cdot ].
$$

Matching symbols with Lemma 3, $H _ { 1 } = \mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } } , H _ { 2 } = \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ , and the adjoint operator is

$$
E _ {\bar {d}, \bar {x}} ^ {*}: \mathcal {H} _ {\mathcal {Z}} ^ {\bar {d}, \bar {x}} \to \mathcal {H} _ {\mathcal {W}} ^ {\bar {d}, \bar {x}}, g (\cdot) \mapsto \mathbb {E} [ g (Z) | D = \bar {d}, X = \bar {x}, W = \cdot ].
$$

Proposition 5 (Existence in the RKHS for fixed $( \bar { d } , \bar { x } ) )$ . Suppose Assumption 4 holds, as well as

1. Regularity: $E _ { \bar { d } , \bar { x } } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } } , \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } } )$ , i.e. the space of Hilbert-Schmidt operators from $\mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ to $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$

2. Completeness: closure(span $\{ \mu _ { z } ^ { \bar { d } , \bar { x } } ( \bar { d } , \bar { x } , w ) \} _ { w \in { \mathcal W } } ) = { \mathcal H } _ { \mathcal Z } ^ { \bar { d } , \bar { x } }$

3. Correct specification: $\gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) \in \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$

4. Penalized square summability: $\begin{array} { r } { \sum _ { k = 1 } ^ { \infty } \eta _ { k } ^ { - 2 } \langle \gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) , e _ { k } ^ { z } \rangle _ { \mathcal { H } _ { z } ^ { \bar { d } , \bar { x } } } ^ { 2 } < \infty . } \end{array}$

Then for any fixed $( \bar { d } , \bar { x } )$ , there exists some $f _ { \bar { d } , \bar { x } } \in \mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ such that

$$
\gamma_ {0} (\bar {d}, \bar {x}, z) = [ E _ {\bar {d}, \bar {x}} f _ {\bar {d}, \bar {x}} ] (z) = \mathbb {E} [ f _ {\bar {d}, \bar {x}} (W) | D = \bar {d}, X = \bar {x}, Z = z ].
$$

I name the conditions of Proposition 5 to match the conditions of Proposition 4, which are standard in the literature. Regularity is again a condition on the smoothness of the distribution $\mathbb { P } ( w | d , x , z )$ . Completeness becomes a condition on how well the encodings of conditional distributions of $Z$ can recover the full space. Correct specification is with respect to the RKHS rather than $\mathbb { L } _ { 2 }$ . Penalized square summability is as before, though using the appropriate inner product.

Proof. Hilbert-Schmidt operators are compact [Carrasco et al., 2007, Theorem 2.32], verifying the compactness requirement. By compactness, the singular value decomposition $( e _ { k } ^ { w } , \eta _ { k } , e _ { k } ^ { z } ) _ { k = 1 } ^ { \infty }$ exists, where $e _ { k } ^ { w } \in \mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ and $e _ { k } ^ { z } \in \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$

Next, I verify inclusion by appealing to completeness and correct specification. Towards this end, I first argue that $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) ^ { \perp } = \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ . It sufices to show $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) = 0$ , i.e. the null space is the zero function. Consider any $g \in \mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } )$ . By definition of the null space,

$$
\begin{array}{r l} & 0 (\cdot) = [ E _ {\bar {d}, \bar {x}} ^ {*} g ] (\cdot) \\ & \qquad = \mathbb {E} [ g (Z) | D = \bar {d}, X = \bar {x}, W = \cdot ]. \end{array}
$$

In Assumption 4, I impose that the kernels are bounded. This assumption has several implications. First, the feature maps are Bochner integrable [Steinwart and Christmann, 2008, Definition A.5.20]. Bochner integrability permits the exchange of expectation and inner product. Second, the mean embeddings exist. Third, the induced kernel $k _ { \bar { d } , \bar { x } } ( z , z ^ { \prime } )$ is also bounded and hence the induced RKHS $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ inherits these favorable properties. Therefore

$$
\begin{array}{r l} & {\mathbb {E} [ g (Z) | D = \bar {d}, X = \bar {x}, W = \cdot ] = \int g (z) \mathrm{d} \mathbb {P} (z | \bar {d}, \bar {x}, \cdot)} \\ & {\qquad = \int \langle g, \phi^ {\bar {d}, \bar {x}} (z) \rangle_ {\mathcal {H} _ {\mathcal {Z}} ^ {\bar {d}, \bar {x}}} \mathrm{d} \mathbb {P} (z | \bar {d}, \bar {x}, \cdot)} \\ & {\qquad = \left\langle g, \int \phi^ {\bar {d}, \bar {x}} (z) \mathrm{d} \mathbb {P} (z | \bar {d}, \bar {x}, \cdot) \right\rangle_ {\mathcal {H} _ {\mathcal {Z}} ^ {\bar {d}, \bar {x}}}} \\ & {\qquad = \langle g, \mu_ {z} ^ {\bar {d}, \bar {x}} (\bar {d}, \bar {x}, \cdot) \rangle_ {\mathcal {H} _ {\mathcal {Z}} ^ {\bar {d}, \bar {x}}}} \end{array}
$$

where $\phi ^ { \bar { d } , \bar { x } } ( z )$ is the feature map of $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ and $\begin{array} { r } { \mu _ { z } ^ { \bar { d } , \bar { x } } ( \bar { d } , \bar { x } , w ) = \int \phi ^ { \bar { d } , \bar { x } } ( w ) \mathrm { d } \mathbb { P } ( z | \bar { d } , \bar { x } , w ) } \end{array}$ is the conditional mean embedding of $\mathbb { P } ( z | \bar { d } , \bar { x } , w )$ . In summary,

$$
0 (\cdot) = \langle g, \mu_ {z} ^ {\bar {d}, \bar {x}} (\bar {d}, \bar {x}, \cdot) \rangle_ {\mathcal {H} _ {\mathcal {Z}} ^ {\bar {d}, \bar {x}}}.
$$

By hypothesis, $g \in \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } } = c l o s u r e ( s p a n \{ \mu _ { z } ^ { \bar { d } , \bar { x } } ( \bar { d } , \bar { z } , w ) \} _ { w \in \mathcal { W } } )$ . Since $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ is a Hilbert space,

$$
\langle g, \mu_ {z} ^ {\bar {d}, \bar {x}} (\bar {d}, \bar {x}, w) \rangle_ {\mathcal {H} _ {z} ^ {\bar {d}, \bar {x}}} = 0 \quad \forall w \in \mathcal {W} \iff g = 0.
$$

Combining these results, $g \in \mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } )$ implies $g = 0$ , so $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) = 0$ and $\mathcal { N } ( E _ { \bar { d } , \bar { x } } ^ { * } ) ^ { \perp } = \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ Finally, correct specification implies $\gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) \in \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$

Penalized square summability is immediate.

Corollary 3 (Existence in the RKHS). If the conditions of Proposition 5 hold for each $( \bar { d } , \bar { x } )$ , then there exists a confounding bridge $h _ { 0 }$ such that

$$
\gamma_ {0} (d, x, z) = \mathbb {E} [ h _ {0} (D, X, W) | D = d, X = x, Z = z ].
$$

Proof. For each $( \bar { d } , \bar { x } )$ , set $h _ { 0 } ( \bar { d } , \bar { x } , w ) = f _ { \bar { d } , \bar { x } } ( w )$

Finally, I relate the spaces $\mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ and $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ with the tensor product construction in the main text. In particular, I show how the spaces $\mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$ and $\mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$ can be induced by a tensor product construction.

Proposition 6 (Relation among kernels). Suppose Assumption 4 holds.

1. If $\gamma _ { 0 } \in \mathcal { H } _ { D } \otimes \mathcal { H } _ { X } \otimes \mathcal { H } _ { Z }$ then $\gamma _ { 0 } ( \bar { d } , \bar { x } , \cdot ) \in \mathcal { H } _ { \mathcal { Z } } ^ { \bar { d } , \bar { x } }$

2. If $h _ { 0 } \in \mathcal { H } _ { D } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } }$ then $h _ { 0 } ( \bar { d } , \bar { x } , \cdot ) \in \mathcal { H } _ { \mathcal { W } } ^ { \bar { d } , \bar { x } }$

The succinct assumption that that the reduced form belongs to a tensor product RKHS, i.e. $\gamma _ { 0 } \in \mathcal { H } _ { D } \otimes \mathcal { H } _ { X } \otimes \mathcal { H } _ { Z }$ , implies the correct specification condition in Proposition 5 for each value $( \bar { d } , \bar { x } )$ . In the main text, I simply assume that the confounding bridge exists and belongs to a tensor product RKHS, i.e. $h _ { 0 } \in \mathcal { H } _ { D } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } }$ . Proposition 6 demonstrates that this assumption is coherent with conditions that imply existence in Proposition 5. I pose as a question for future research how to characterize the conditions with which one can prove, rather than assume, that the confounding bridge is an element of an RKHS.

Proof. I prove each result separately.

1. If $\gamma _ { 0 } \in \mathcal { H } _ { D } \otimes \mathcal { H } _ { X } \otimes \mathcal { H } _ { Z }$ , then

$$
\gamma_ {0} (\bar {d}, \bar {x}, z) = \langle \gamma_ {0}, \phi (\bar {d}) \otimes \phi (\bar {x}) \otimes \phi (z) \rangle_ {\mathcal {H} _ {\mathcal {D}} \otimes \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {Z}}}.
$$

Moreover,

$$
\begin{array}{r l} & k _ {\bar {d}, \bar {x}} (z, z ^ {\prime}) = k (\bar {d}, \bar {d}) k (\bar {x}, \bar {x}) k (z, z ^ {\prime}) \\ & \qquad = \langle \phi (\bar {d}) \otimes \phi (\bar {x}) \otimes \phi (z), \phi (\bar {d}) \otimes \phi (\bar {x}) \otimes \phi (z ^ {\prime}) \rangle_ {\mathcal {H} _ {\mathcal {D}} \otimes \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {Z}}} \end{array}
$$

so the feature maps coincide.

2. If $h _ { 0 } \in \mathcal { H } _ { D } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } }$ , then

$$
h _ {0} (\bar {d}, \bar {x}, w) = \langle \gamma_ {0}, \phi (\bar {d}) \otimes \phi (\bar {x}) \otimes \phi (w) \rangle_ {\mathcal {H} _ {\mathcal {D}} \otimes \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}}.
$$

Moreover,

$$
\begin{array}{r l} & k _ {\bar {d}, \bar {x}} (w, w ^ {\prime}) = k (\bar {d}, \bar {d}) k (\bar {x}, \bar {x}) k (w, w ^ {\prime}) \\ & \qquad = \langle \phi (\bar {d}) \otimes \phi (\bar {x}) \otimes \phi (w), \phi (\bar {d}) \otimes \phi (\bar {x}) \otimes \phi (w ^ {\prime}) \rangle_ {\mathcal {H} _ {\mathcal {D}} \otimes \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \end{array}
$$

so the feature maps coincide.

## C Identification proof

To lighten notation, I abbreviate conditional expectations and conditional probabilities. For example, I write

$$
\gamma_ {0} (d, x, z) = \mathbb {E} [ Y | D = d, X = x, Z = z ] = \mathbb {E} [ Y | d, x, z ].
$$

Proof of Proposition 2. $h _ { 0 }$ is defined as a solution to $\gamma _ { 0 } ( d , x , z ) \ = \ { \mathbb E } [ h ( D , X , W ) | D \ =$ $d , X = x , Z = z ]$ , which exists by Assumption 2. The reduced form $\gamma _ { 0 } ( d , x , z ) = \mathbb { E } [ Y | D =$ $d , X = x , Z = z ]$ is the same across populations since $\mathbb { P } ( Y | D , X , Z )$ is invariant in Assumption 3. The conditional expectation operator is also the same across populations since $\mathbb { P } ( W | D , X , Z )$ is invariant in Assumption 3. □

Proposition 7. Suppose the conditions of Theorem 1 hold. Then

$$
\mathbb {E} [ Y | d, u, x ] = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x)
$$

For $\theta _ { 0 } ^ { C A T E }$ , replace x with (v, x).

Proof. I generalize [Miao et al., 2018, Theorem 1]. By Assumption 1

$$
Y \bot Z | U, D, X
$$

$$
W \perp D, Z | U, X.
$$

The former implies

$$
\begin{array}{l} \gamma_ {0} (d, x, z) = \mathbb {E} [ Y | d, x, z ] \\ \qquad = \int \mathbb {E} [ Y, u | d, x, z ] \mathrm{d} u \\ \qquad = \int \mathbb {E} [ Y | u, d, x, z ] \mathrm{d} \mathbb {P} (u | d, x, z) \\ \qquad = \int \mathbb {E} [ Y | u, d, x ] \mathrm{d} \mathbb {P} (u | d, x, z). \end{array}
$$

The latter implies

$$
\begin{array}{l} \mathbb {P} (w | d, x, z) = \int \mathbb {P} (w, u | d, x, z) \mathrm{d} u \\ \qquad = \int \mathbb {P} (w | u, d, x, z) \mathrm{d} \mathbb {P} (u | d, x, z) \\ \qquad = \int \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u | d, x, z). \end{array}
$$

Using these results and existence

$$
\begin{array}{r l} \int \mathbb {E} [ Y | u, d, x ] \mathrm{d} \mathbb {P} (u | d, x, z) & = \gamma_ {0} (d, x, z) \\ & = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | d, x, z) \\ & = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u | d, x, z). \end{array}
$$

Hence by completeness

$$
\mathbb {E} [ Y | u, d, x ] = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x).
$$

Proof of Theorem 1. I prove each result, appealing to Assumption 1 and Proposition 7.

$$
\begin{array}{l} \theta_ {0} ^ {A T E} (d) = \mathbb {E} [ Y ^ {(d)} ] \\ \qquad = \int \mathbb {E} [ Y ^ {(d)} | u, x ] \mathrm{d} \mathbb {P} (u, x) \\ \qquad = \int \mathbb {E} [ Y ^ {(d)} | d, u, x ] \mathrm{d} \mathbb {P} (u, x) \\ \qquad = \int \mathbb {E} [ Y | d, u, x ] \mathrm{d} \mathbb {P} (u, x) \\ \qquad = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u, x) \\ \qquad = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w, u, x) \\ \qquad = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w, x); \end{array}
$$

$$
\begin{array}{l} \theta_ {0} ^ {D S} (d, \tilde {\mathbb {P}}) = \mathbb {E} _ {\tilde {\mathbb {P}}} [ Y ^ {(d)} ] \\ \qquad = \int \mathbb {E} _ {\tilde {\mathbb {P}}} [ Y ^ {(d)} | u, x ] \mathrm{d} \tilde {\mathbb {P}} (u, x) \\ \qquad = \int \mathbb {E} _ {\tilde {\mathbb {P}}} [ Y ^ {(d)} | d, u, x ] \mathrm{d} \tilde {\mathbb {P}} (u, x) \\ \qquad = \int \mathbb {E} _ {\tilde {\mathbb {P}}} [ Y | d, u, x ] \mathrm{d} \tilde {\mathbb {P}} (u, x) \\ \qquad = \int h _ {0} (d, x, w) \mathrm{d} \tilde {\mathbb {P}} (w | u, x) \mathrm{d} \tilde {\mathbb {P}} (u, x) \\ \qquad = \int h _ {0} (d, x, w) \mathrm{d} \tilde {\mathbb {P}} (w, u, x) \\ \qquad = \int h _ {0} (d, x, w) \mathrm{d} \tilde {\mathbb {P}} (w, x); \end{array}
$$

$$
\begin{array}{l} \theta_ {0} ^ {A T T} (d, d ^ {\prime}) = \mathbb {E} [ Y ^ {(d ^ {\prime})} | D = d ] \\ \qquad = \int \mathbb {E} [ Y ^ {(d ^ {\prime})} | d, u, x ] \mathrm{d} \mathbb {P} (u, x | d) \\ \qquad = \int \mathbb {E} [ Y ^ {(d ^ {\prime})} | d ^ {\prime}, u, x ] \mathrm{d} \mathbb {P} (u, x | d) \\ \qquad = \int \mathbb {E} [ Y | d ^ {\prime}, u, x ] \mathrm{d} \mathbb {P} (u, x | d) \\ \qquad = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (w | u, x) \mathrm{d} \mathbb {P} (u, x | d) \\ \qquad = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (w | d, u, x) \mathrm{d} \mathbb {P} (u, x | d) \\ \qquad = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (w, u, x | d) \\ \qquad = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (w, x | d); \end{array}
$$

$$
\begin{array}{l} \theta_ {0} ^ {C A T E} (d, v) = \mathbb {E} [ Y ^ {(d)} | V = v ] \\ \qquad = \int \mathbb {E} [ Y ^ {(d)} | u, v, x ] \mathrm{d} \mathbb {P} (u, x | v) \\ \qquad = \int \mathbb {E} [ Y ^ {(d)} | d, u, v, x ] \mathrm{d} \mathbb {P} (u, x | v) \\ \qquad = \int \mathbb {E} [ Y | d, u, v, x ] \mathrm{d} \mathbb {P} (u, x | v) \\ \qquad = \int h _ {0} (d, v, x, w) \mathrm{d} \mathbb {P} (w | u, v, x) \mathrm{d} \mathbb {P} (u, x | v) \\ \qquad = \int h _ {0} (d, v, x, w) \mathrm{d} \mathbb {P} (w, u, x | v) \\ \qquad = \int h _ {0} (d, v, x, w) \mathrm{d} \mathbb {P} (w, x | v). \end{array}
$$

## D Discussion of the source condition

In this appendix, I provide further discussion of the smoothness assumption that drives my analysis: the source condition. The source condition pertains to estimation of $h _ { 0 }$ rather than existence of $h _ { 0 }$ . In this appendix, I discuss estimation assumptions; see Appendix B for a thorough discussion of existence assumptions. This discussion is more intuitive than formal. See [Chen and Reiss, 2011] for a formal discussion of source conditions in the NPIV literature.

## D.1 Source conditions in this work

A source condition is an approximation assumption that helps to control the bias from ridge regularization. I place four source conditions in this work, parametrized by $( c , c _ { 0 } , c _ { 1 } , c _ { 2 } )$ The source condition parametrized by c quantifies the smoothness of the confounding bridge $h _ { 0 }$ . The source conditions parametrized by $( c _ { 0 } , c _ { 1 } , c _ { 2 } )$ quantify the smoothness of the conditional distributions $\mathbb { P } ( w | d , x , z ) , \mathbb { P } ( x , w | d )$ , and $\mathbb { P } ( x , w | v )$ . Due to the similarity of $( c _ { 0 } , c _ { 1 } , c _ { 2 } )$ in this discussion I focus on only $( c , c _ { 0 } )$

To understand the role of $( c , c _ { 0 } )$ , recall the integral equation

$$
\gamma_ {0} (d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | d, x, z).
$$

The source conditions apply to both objects on the RHS. In Section 4, I provide an RKHS construction to express the integral equation as

$$
\gamma_ {0} (d, x, z) = \left\langle h _ {0}, \phi (d) \otimes \phi (x) \otimes E _ {0} ^ {*} [ \phi (d) \otimes \phi (x) \otimes \phi (z) ] \right\rangle_ {\mathcal {H}}.
$$

The conditional expectation operator $E _ { 0 }$ encodes the conditional distribution $\mathbb { P } ( w | d , x , z )$ R and I quantify its smoothness by $c _ { 0 }$ . I quantify the smoothness of the confounding bridge $h _ { 0 }$ by c. Recall that $\phi ( d ) , \phi ( x )$ , and $\phi ( z )$ can be interpreted as dictionaries of basis functions.

I propose an estimator of the confounding bridge $h _ { 0 }$ that proceeds in two stages similar to two stage least squares (2SLS): (i) estimate the conditional expectation operator $E _ { 0 }$ with a generalized kernel ridge regression; (ii) estimate $h _ { 0 }$ with a generalized kernel ridge regression, using the estimator of $E _ { 0 }$ from the first stage. See Section 4 for discussion of the estimation procedure. Because both $( E _ { 0 } , h _ { 0 } )$ involve ridge regularization, I place the source conditions $( c _ { 0 } , c )$ to control the regularization bias. See Appendix G for the formal arguments by which the source conditions imply bounds on regularization bias.

The source condition $c _ { 0 }$ placed on $E _ { 0 }$ means that the conditional expectation operator for P $( w | d , x , z )$ has rapidly decaying Fourier coeficients as formalized in Section 3. In particular, $c _ { 0 }$ quantifies how fast the Fourier coeficients of $E _ { 0 }$ decay relative to the eigenvalues of the kernel of $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { Z } } )$ . Similarly, the source condition c placed on $h _ { 0 }$ means that the confounding bridge has rapidly decaying Fourier coeficients as formalized in Section 3. In particular, c quantifies how fast the Fourier coeficients of $h _ { 0 }$ decay relative to the eigenvalues of the kernel of ${ \mathcal { H } } _ { \mu }$

## D.2 Source conditions in ill posed inverse problems

As documented in the main text, the confounding bridge learning problem is precisely the nonparametric instrumental variable (NPIV) regression problem where $( D , X )$ are exogenous regressors, W is an endogenous regressor, and Z is the instrument. Several works in the NPIV literature deploy source conditions or similar assumptions including [Hall and Horowitz, 2005, Darolles et al., 2011, Singh et al., 2019]. I demonstrate that the source conditions in this work generalize those of [Singh et al., 2019] and depart from those of [Darolles et al., 2011, Hall and Horowitz, 2005].

The source conditions in this paper most closely resemble those in [Singh et al., 2019], who study the NPIV problem without exogenous covariates. In particular, the integral equation studied in that work is

$$
\gamma_ {0} (z) = \int h _ {0} (w) \mathrm{d} \mathbb {P} (w | z) = \langle h _ {0}, E _ {0} ^ {*} \phi (z) \rangle_ {\mathcal {H}}.
$$

As such, the source conditions in [Singh et al., 2019] are a special case of the source conditions in this work.

[Darolles et al., 2011] also study the NPIV problem without exogenous covariates, but assume a diferent source condition. Whereas I assume two source conditions for $E _ { 0 }$ and $h _ { 0 }$ , [Darolles et al., 2011] assume one source condition for $E _ { 0 }$ and $h _ { 0 }$ . Consider the NPIV problem without exogenous covariates. If $E _ { 0 }$ is compact, then its singular value decomposition is $\{ e _ { j } ^ { w } , \eta _ { j } , e _ { j } ^ { z } \} _ { j = 1 } ^ { \infty }$ where $e _ { j } ^ { w } \in \mathbb { L } _ { 2 } ( \mathbb { P } ( w ) )$ ) and $e _ { j } ^ { z } \in \mathbb { L } _ { 2 } ( \mathbb { P } ( z ) ) ,$ ). The authors assume

$$
h _ {0} \in \mathcal {H} ^ {\beta} := \left\{f = \sum_ {j = 1} ^ {\infty} f _ {j} e _ {j} ^ {z}: \sum_ {j = 1} ^ {\infty} \frac {f _ {j} ^ {2}}{\eta_ {j} ^ {2 \beta}} <   \infty , f _ {j} = \langle f, e _ {j} ^ {w} \rangle_ {\mathbb {L} _ {2} (\mathbb {P} (w))} \right\}, \quad \beta \in (0, 2 ].
$$

This source condition involves both the smoothness of $E _ { 0 }$ , encoded by the rate at which the singular values $\{ \eta _ { j } \}$ decay, and the smoothness of $h _ { 0 }$ , encoded by the rate at which the Fourier coeficients $\{ f _ { j } \}$ decay.

While [Hall and Horowitz, 2005] do not explicitly place a source condition, the assumptions in that work are closely related to the source condition in [Darolles et al., 2011]. Specifically, [Hall and Horowitz, 2005] directly place assumptions on the rates at which $\{ \eta _ { j } \}$ and $\{ f _ { j } \}$ and decay: $\eta _ { j } \sim j ^ { - s }$ and $f _ { j } \sim j ^ { - r }$ . Note that if these conditions hold and $\beta < s ^ { - 1 } ( r - 1 / 2 )$ then $h _ { 0 } \in \mathcal { H } ^ { \beta }$ . Moreover, s essentially pins down the degree of smoothness of the joint density of (W, Z).

## D.3 Alternatives to source conditions

Two recent works prove uniform consistency of estimators in ill posed settings. Whereas this work proposes machine learning estimators with finite sample guarantees for the negative control problem, previous work proposes series estimators with asymptotic guarantees for related problems: NPIV [Chen and Christensen, 2018] and panel proxy control [Deaner, 2018]. I now compare the source conditions with the alternatives assumptions in these related works.

[Chen and Christensen, 2018] assume that (W, Z) have compact rectangular supports and bounded densities. Denoting by J the dimension of the series and $\tau _ { J }$ the series measure of ill posedness, the authors place restrictions on how J evolves relative to $n , \ \tau _ { J } .$ , and the norms of inverse covariances applied to the series. The main assumption is a collection of high level conditions about ill posedness, the projection of $E _ { 0 }$ onto the series, and the projection of $h _ { 0 }$ onto the series [Chen and Christensen, 2018, Assumption 4]. The high level conditions are satisfied if the series are taken to be the singular functions $( e _ { j } ^ { w } )$ and $\left( e _ { j } ^ { z } \right)$ of the conditional expectation operator $E _ { 0 }$ . In such case, the stated conditions amount to joint assumptions on $\{ f _ { j } \}$ and $\{ \eta _ { j } \}$ as before, which we have seen to be a type of source condition.

The conditions of [Deaner, 2018] are again in terms of series analysis. [Deaner, 2018] assumes (i) the series approximates Hölder functions well; (ii) joint densities are bounded away from zero and one; (iii) several ratios of joint over marginal densities are Hölder smooth, as is the reduced form $\gamma _ { 0 } ;$ (iv) the series dimension J grows at a certain rate relative to $n _ { : }$ , the approximation quality, and the norms of inverse covariances applied to the series. The smoothness of joint over marginal density ratios is akin to assumptions on $\{ \eta _ { j } \}$ . The restrictions on norms of inverse covariances applied to the series is akin to assumptions on $\{ f _ { j } \}$ . In this sense, the assumptions of [Chen and Christensen, 2018, Deaner, 2018] resemble the assumptions of [Hall and Horowitz, 2005].

## E Algorithm derivation

## E.1 Overview

I present an overview of the end-to-end procedure, first in principle then in practice. For simplicity, I focus on the dose response curve $\begin{array} { r } { \theta _ { 0 } ^ { A T E } ( d ) = \int h _ { 0 } ( d , x , w ) \mathrm { d } \mathbb { P } ( x , w ) } \end{array}$ , where $h _ { 0 }$ is the confounding bridge and $\mathbb { P } ( x , w )$ is the counterfactual distribution with which we wish to reweight $h _ { 0 }$ . To further simplify the discussion, I abstract from sample splitting. By Theorem 2,

$$
\theta_ {0} ^ {A T E} (d) = \langle h _ {0}, \phi (d) \otimes \mu \rangle_ {\mathcal {H}}, \quad \mu := \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \mathbb {P} (x, w)
$$

where $\mu$ is the mean embedding of the counterfactual distribution $\mathbb { P } ( x , w )$ . This representation suggests an estimator of the form

$$
\hat {\theta} ^ {A T E} (d) = \langle \hat {h}, \phi (d) \otimes \hat {\mu} \rangle_ {\mathcal {H}}, \quad \hat {\mu} := \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \hat {\mathbb {P}} (x, w)
$$

combining appropriate estimators $\hat { h }$ and $\hat { \mu }$ . Estimation of $\hat { \mu }$ is simply an average: $\hat { \mu } =$ $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } \phi ( x _ { i } ) \otimes \phi ( w _ { i } ) \quad } \end{array}$ . Estimation of $\hat { h }$ is more involved.

Towards an estimator $\hat { h } .$ I recast $h _ { 0 }$ as a nonparametric instrumental variable (NPIV) regression. The confounding bridge is defined as the solution to the integral equation

$$
\gamma_ {0} (d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | d, x, z), \quad \gamma_ {0} (d, x, z) := \mathbb {E} [ Y | D = d, X = x, Z = z ].
$$

For expositional purposes, define the residual $\epsilon : = Y - \gamma _ { 0 } ( D , X , Z )$ so that

$$
Y = \int h _ {0} (D, X, w) \mathrm{d} \mathbb {P} (w | D, X, Z) + \epsilon , \quad \mathbb {E} [ \epsilon | D, X, Z ] = 0.
$$

Furthermore,

$$
Y = h _ {0} (D, X, W) + \epsilon , \mathbb {E} [ \epsilon | D, X, Z ] = 0.
$$

This representation is precisely the NPIV problem where $W$ is endogenous and $Z$ is the instrument.

In this light, I propose a two stage estimator for $\hat { h }$ that generalizes two stage least squares (2SLS) and appeals to the RKHS construction. In 2SLS, an analyst would estimate $\hat { h } ^ { 2 S L S }$ by (i) projecting W onto $( D , X , Z )$ , obtaining an estimator $\hat { W } ( D , X , Z )$ ; then (ii) projecting $Y$ onto $( D , X , { \hat { W } } ( D , X , Z ) )$ ). I propose something similar. Define the mean embedding

$$
\mu_ {w} (d, x, z) := \int \phi (w) \mathrm{d} \mathbb {P} (w | d, x, z)
$$

which encodes the distribution $\mathbb { P } ( w | d , x , z )$ in the integral equation we wish to solve. I propose estimating $\hat { h }$ by (i) projecting $\phi ( W )$ onto $\phi ( D ) \otimes \phi ( X ) \otimes \phi ( W )$ , obtaining an estimator $\hat { \mu } _ { w } ( D , X , Z )$ ; then (ii) projecting $Y$ onto $\phi ( D ) \otimes \phi ( X ) \otimes { \hat { \mu } } _ { w } ( D , X , Z )$

Algorithm 5 (End-to-end: Principles). Given n observations of outcome Y, treatment $D ,$ covariates X, negative control outcome W, and negative control treatment $Z .$ ,

1. Specify the kernels $k _ { D } , k _ { X } , k _ { \mathcal { W } } , k _ { \mathcal { Z } }$

2. Specify the regularization hyperparameters $( \lambda , \xi )$

3. Estimate the confounding bridge $\hat { h }$ in two stages, using $( \lambda , \xi )$

(a) Estimate the distribution in the integral equation $\hat { \mathbb { P } } ( w | d , x , z )$ via its mean embedding $\hat { \mu } _ { w } ( d , x , z )$ with regularization λ.

$$
\phi (D) \otimes \phi (X) \otimes \hat {\mu} _ {w} (D, X, Z)
$$

$$
\xi .
$$

4. Estimate the counterfactual distribution $\hat { \mathbb { P } } ( x , w )$ via its mean embedding $\hat { \mu } .$

5. Estimate the dose response $\hat { \theta } ^ { A T E } ( d )$ by combining $\hat { h }$ and $\hat { \mu }$ according to $\hat { \theta } ^ { A T E } ( d ) =$ $\langle \hat { h } , \phi ( d ) \otimes \hat { \mu } \rangle _ { \mathcal { H } }$

With these principles in mind, I now fill in the details. Objects in an RKHS are infinite dimensional, and I have reasoned about them abstractly in Algorithm 5. To actually compute the estimator, I must express the procedure exclusively as inner products of RKHS objects, i.e. as scalar evaluations of kernels. I provide such details in Algorithm 3 in the main text.

## E.2 Representation

Proof of Theorem 2. In Assumption 4, I impose that the scalar kernels are bounded. This assumption has several implications. First, the feature maps are Bochner integrable [Steinwart and Christm

Definition A.5.20]. Bochner integrability permits the exchange of expectation and inner product. Second, the mean embeddings exist. Third, the product kernel is also bounded and hence the tensor product RKHS inherits these favorable properties. Since $h _ { 0 } \in \mathcal { H }$ 2

$$
\begin{array}{l} \gamma_ {0} (d, x, z) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (w | d, x, z) \\ \qquad = \int \langle h _ {0}, \phi (d) \otimes \phi (x) \otimes \phi (w) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (w | d, x, z) \\ \qquad = \left\langle h _ {0}, \phi (d) \otimes \phi (x) \otimes \int \phi (w) \mathrm{d} \mathbb {P} (w | d, x, z) \right\rangle_ {\mathcal {H}} \\ \qquad = \langle h _ {0}, \phi (d) \otimes \phi (x) \otimes \mu_ {w} (d, x, z) \rangle_ {\mathcal {H}}. \end{array}
$$

Next, I generalize [Singh et al., 2020, Theorem 3.2], replacing the prediction function with the confounding bridge $h _ { 0 }$ . By Theorem 1 and linearity of expectation,

$$
\begin{array}{l} \theta_ {0} ^ {A T E} (d) = \int h _ {0} (d, x, w) \mathrm{d} \mathbb {P} (x, w) \\ \qquad = \int \langle h _ {0}, \phi (d) \otimes \phi (x) \otimes \phi (w) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (x, w) \\ \qquad = \left\langle h _ {0}, \phi (d) \otimes \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \mathbb {P} (x, w) \right\rangle_ {\mathcal {H}} \\ \qquad = \langle h _ {0}, \phi (d) \otimes \mu \rangle_ {\mathcal {H}}; \end{array}
$$

$$
\begin{array}{l} \theta_ {0} ^ {D S} (d, \tilde {\mathbb {P}}) = \int h _ {0} (d, x, w) \mathrm{d} \tilde {\mathbb {P}} (x, w) \\ \qquad = \int \langle h _ {0}, \phi (d) \otimes \phi (x) \otimes \phi (w) \rangle_ {\mathcal {H}} \mathrm{d} \tilde {\mathbb {P}} (x, w) \\ \qquad = \left\langle h _ {0}, \phi (d) \otimes \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \tilde {\mathbb {P}} (x, w) \right\rangle_ {\mathcal {H}} \\ \qquad = \langle h _ {0}, \phi (d) \otimes \nu \rangle_ {\mathcal {H}}; \end{array}
$$

$$
\begin{array}{r l} & {\theta_ {0} ^ {A T T} (d, d ^ {\prime}) = \int h _ {0} (d ^ {\prime}, x, w) \mathrm{d} \mathbb {P} (x, w | d)} \\ & {\qquad = \int \langle h _ {0}, \phi (d ^ {\prime}) \otimes \phi (x) \otimes \phi (w) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (x, w | d)} \\ & {\qquad = \bigg \langle h _ {0}, \phi (d ^ {\prime}) \otimes \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \mathbb {P} (x, w | d) \bigg \rangle_ {\mathcal {H}}} \\ & {\qquad = \langle h _ {0}, \phi (d ^ {\prime}) \otimes \mu (d) \rangle_ {\mathcal {H}};} \end{array}
$$

$$
\begin{array}{l} \theta_ {0} ^ {C A T E} (d, v) = \int h _ {0} (d, v, x, w) \mathrm{d} \mathbb {P} (x, w | v) \\ \qquad = \int \langle h _ {0}, \phi (d) \otimes \phi (v) \otimes \phi (x) \otimes \phi (w) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (x, w | v) \\ \qquad = \left\langle h _ {0}, \phi (d) \otimes \phi (v) \otimes \int [ \phi (x) \otimes \phi (w) ] \mathrm{d} \mathbb {P} (x, w | v) \right\rangle_ {\mathcal {H}} \\ \qquad = \langle h _ {0}, \phi (d) \otimes \phi (v) \otimes \mu (v) \rangle_ {\mathcal {H}}. \end{array}
$$

## E.3 Confounding bridge

Let n be the number of observations of $( d _ { i } , x _ { i } , w _ { i } , z _ { i } )$ used to estimate the conditional mean embedding $\mu _ { w } ( d , x , z )$ by kernel ridge regression with regularization parameter λ. Let m be the number of observations of $( { \dot { y } } _ { i } , { \dot { d } } _ { i } , { \dot { x } } _ { i } , { \dot { z } } _ { i } )$ used to estimate the confounding bridge $h _ { 0 }$ by kernel ridge regression with regularization parameter $\xi .$ This notation allows the analyst to use diferent quantities of observations $( n , m )$ to estimate $( E _ { 0 } , h _ { 0 } )$ , or to reuse the same observations.

Derivation of Algorithm 1. I proceed in steps, generalizing the derivation of [Singh et al., 2019, Algorithm 1].

## 1. Closed form for stage 1.

By [Singh et al., 2019, Algorithm 1], the closed form solution for the stage 1 conditional mean embedding is

$$
\hat {\mu} _ {w} (d, x, z) = \sum_ {i = 1} ^ {n} \beta_ {i} (d, x, z) \phi (w _ {i})
$$

where

$$
\beta (d, x, z) = \left(K _ {D D} \odot K _ {X X} \odot K _ {Z Z} + n \lambda I\right) ^ {- 1} \left[ K _ {D d} \odot K _ {X x} \odot K _ {Z z} \right] \in \mathbb {R} ^ {n}.
$$

Slightly abusing notation, one may write

$$
\hat {\mu} _ {w} (d, x, z) = K. _ {W} \beta (d, x, z).
$$

## 2. Closed form for stage 2.

Next, I argue that $\begin{array} { r } { \hat { h } = \sum _ { i = 1 } ^ { m } \alpha _ { i } [ \phi ( \dot { d } _ { i } ) \otimes \phi ( \dot { x } _ { i } ) \otimes \hat { \mu } _ { w } ( \dot { d } _ { i } , \dot { x } _ { i } , \dot { z } _ { i } ) ] } \end{array}$ for some $\alpha \in \mathbb { R } ^ { m }$ . Write the objective as

$$
\hat {h} = \underset {h \in \mathcal {H}} {\operatorname{argmin}} \mathcal {E} _ {\xi} ^ {m} (h), \quad \mathcal {E} _ {\xi} ^ {m} (h) = \frac {1}{m} \sum_ {i = 1} ^ {m} \| \dot {y} _ {i} - \langle h, \phi (\dot {d} _ {i}) \otimes \phi (\dot {x} _ {i}) \otimes \hat {\mu} _ {w} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \| _ {\mathcal {Y}} ^ {2} + \xi \| h \| _ {\mathcal {H}} ^ {2}.
$$

Due to the ridge penalty, the stated objective is coercive and strongly convex with respect to h. Hence it has a unique minimizer that obtains the minimum.

Write $\hat { h } = \hat { h } _ { m } + \hat { h } _ { m } ^ { \perp }$ where $\hat { h } _ { m } \in s p a n \{ \phi ( \dot { d } _ { i } ) \otimes \phi ( \dot { x } _ { i } ) \otimes \hat { \mu } _ { w } ( \dot { d } _ { i } , \dot { x } _ { i } , \dot { z } _ { i } ) \}$ and $\hat { h } _ { m } ^ { \perp }$ is an element of the orthogonal complement. Therefore

$$
\mathcal {E} _ {\xi} ^ {m} (\hat {h}) = \mathcal {E} _ {\xi} ^ {m} (\hat {h} _ {m}) + \xi \| \hat {h} _ {m} ^ {\perp} \| _ {\mathcal {H}} ^ {2}
$$

which implies $\mathcal { E } _ { \xi } ^ { m } ( \hat { h } ) \geq \mathcal { E } _ { \xi } ^ { m } ( \hat { h } _ { m } )$ . Since $\hat { h }$ is the unique minimizer, $\hat { h } = \hat { h } _ { m }$

## 3. Substitution.

I substitute the functional form of $\hat { h }$ into its objective. Note that

$$
\begin{array}{r l} & {\| \hat {h} \| _ {\mathcal {H}} ^ {2}} \\ & {= \langle \hat {h}, \hat {h} \rangle_ {\mathcal {H}}} \\ & {= \left\langle \sum_ {i = 1} ^ {m} \alpha_ {i} [ \phi (\dot {d} _ {i}) \otimes \phi (\dot {x} _ {i}) \otimes \hat {\mu} _ {w} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) ], \sum_ {j = 1} ^ {m} \alpha_ {j} [ \phi (\dot {d} _ {j}) \otimes \phi (\dot {x} _ {j}) \otimes \hat {\mu} _ {w} (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}) ] \right\rangle_ {\mathcal {H}}} \\ & {= \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {m} \alpha_ {i} \alpha_ {j} k (\dot {d} _ {i}, \dot {d} _ {j}) k (\dot {x} _ {i}, \dot {x} _ {j}) \beta (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) ^ {\top} K _ {W W} \beta (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j})} \\ & {= \alpha^ {\top} M \alpha} \end{array}
$$

where in the last line I define the matrix

$$
M = K _ {\dot {D} \dot {D}} \odot K _ {\dot {X} \dot {X}} \odot \left\{\dot {A} ^ {\top} (A + n \lambda I) ^ {- 1} K _ {W W} (A + n \lambda I) ^ {- 1} \dot {A} \right\} \in \mathbb {R} ^ {m \times m}
$$

with

$$
A = K _ {D D} \odot K _ {X X} \odot K _ {Z Z} \in \mathbb {R} ^ {n \times n}, \quad \dot {A} = K _ {D \dot {D}} \odot K _ {X \dot {X}} \odot K _ {Z \dot {Z}} \in \mathbb {R} ^ {n \times m}.
$$

Further define

$$
B = (A + n \lambda I) ^ {- 1} \dot {A} \in \mathbb {R} ^ {n \times m}
$$

whose j-th column is $\beta ( \dot { d } _ { j } , \dot { x } _ { j } , \dot { z } _ { j } )$ . In terms of $B _ { : }$ ,

$$
M = K _ {\dot {D} \dot {D}} \odot K _ {\dot {X} \dot {X}} \odot \{B ^ {\top} K _ {W W} B \}.
$$

Next note that

$$
\begin{array}{l} \langle \hat {h}, \phi (d) \otimes \phi (x) \otimes \hat {\mu} _ {w} (d, x, z) \rangle_ {\mathcal {H}} \\ = \left\langle \sum_ {j = 1} ^ {m} \alpha_ {j} [ \phi (\dot {d} _ {j}) \otimes \phi (\dot {x} _ {j}) \otimes \hat {\mu} _ {w} (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}) ], \phi (d) \otimes \phi (x) \otimes \hat {\mu} _ {w} (d, x, z) \right\rangle_ {\mathcal {H}} \\ = \sum_ {j = 1} ^ {m} \alpha_ {j} \left\langle \phi (\dot {d} _ {j}) \otimes \phi (\dot {x} _ {j}) \otimes \hat {\mu} _ {w} (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}), \phi (d) \otimes \phi (x) \otimes \hat {\mu} _ {w} (d, x, z) \right\rangle_ {\mathcal {H}} \\ = \sum_ {j = 1} ^ {m} \alpha_ {j} k (\dot {d} _ {j}, d) k (\dot {x} _ {j}, x) \beta (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}) ^ {\top} K _ {W W} \beta (d, x, z). \end{array}
$$

Hence

$$
\begin{array}{l} \langle \hat {h}, \phi (\dot {d} _ {i}) \otimes \phi (\dot {x} _ {i}) \otimes \hat {\mu} _ {w} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \\ = \sum_ {j = 1} ^ {n} \alpha_ {j} k (\dot {d} _ {j}, \dot {d} _ {i}) k (\dot {x} _ {j}, \dot {x} _ {i}) \beta (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}) ^ {\top} K _ {W W} \beta (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \\ = \sum_ {j = 1} ^ {m} \alpha_ {j} k (\dot {d} _ {i}, \dot {d} _ {j}) k (\dot {x} _ {i}, \dot {x} _ {j}) \beta (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) ^ {\top} K _ {W W} \beta (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}). \end{array}
$$

In terms of $M$ ,

$$
\begin{array}{l} [ M ^ {\top} \alpha ] _ {i} = \left[ \left(K _ {\dot {D} \dot {D}} \odot K _ {\dot {X} \dot {X}} \odot \{B ^ {\top} K _ {W W} B \}\right) \alpha \right] _ {i} \\ = \left[ \sum_ {j = 1} ^ {m} \alpha_ {j} \left(K _ {\dot {D} \dot {d} _ {j}} \odot K _ {\dot {X} \dot {x} _ {j}} \odot B ^ {\top} K _ {W W} \beta (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j})\right) \right] _ {i} \\ = \sum_ {j = 1} ^ {m} \alpha_ {j} k (\dot {d} _ {i}, \dot {d} _ {j}) k (\dot {x} _ {i}, \dot {x} _ {j}) \beta (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) ^ {\top} K _ {W W} \beta (\dot {d} _ {j}, \dot {x} _ {j}, \dot {z} _ {j}) \\ = \langle \hat {h}, \phi (\dot {d} _ {i}) \otimes \phi (\dot {x} _ {i}) \otimes \hat {\mu} _ {w} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}}. \end{array}
$$

In summary,

$$
\mathcal {E} _ {\xi} ^ {m} (\hat {h}) = \frac {1}{m} \| \dot {Y} - M ^ {\top} \alpha \| _ {2} ^ {2} + \xi \alpha^ {\top} M \alpha .
$$

4. Optimization.

Express the objective, scaled by m, as

$$
\begin{array}{r l} & m \mathcal {E} _ {\xi} ^ {m} (\hat {h}) = \| \dot {Y} - M ^ {\top} \alpha \| _ {2} ^ {2} + m \xi \cdot \alpha^ {\top} M \alpha \\ & \qquad = \dot {Y} ^ {\top} \dot {Y} - 2 \alpha^ {\top} M \dot {Y} + \alpha^ {\top} M M ^ {\top} \alpha + m \xi \cdot \alpha^ {\top} M \alpha \\ & \qquad = \dot {Y} ^ {\top} \dot {Y} - 2 \alpha^ {\top} M \dot {Y} + \alpha^ {\top} (M M ^ {\top} + m \xi M) \alpha \end{array}
$$

Solving the first order condition with respect to α,

$$
\hat {\alpha} = (M M ^ {\top} + m \xi M) ^ {- 1} M \dot {Y}.
$$

Therefore

$$
\begin{array}{r l} & {\hat {h} (d, x, w) = \langle \hat {h}, \phi (d) \otimes \phi (x) \otimes \phi (w) \rangle_ {\mathcal {H}}} \\ & {\qquad = \bigg \langle \sum_ {i = 1} ^ {m} \hat {\alpha} _ {i} [ \phi (\dot {d} _ {i}) \otimes \phi (\dot {x} _ {i}) \otimes \hat {\mu} _ {w} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) ], \phi (d) \otimes \phi (x) \otimes \phi (w) \bigg \rangle_ {\mathcal {H}}} \\ & {\qquad = \sum_ {i = 1} ^ {m} \hat {\alpha} _ {i} \left\langle \phi (\dot {d} _ {i}) \otimes \phi (\dot {x} _ {i}) \otimes \hat {\mu} _ {w} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}), \phi (d) \otimes \phi (x) \otimes \phi (w) \right\rangle_ {\mathcal {H}}} \\ & {\qquad = \sum_ {i = 1} ^ {m} \hat {\alpha} _ {i} k (\dot {d} _ {i}, d) k (\dot {x} _ {i}, x) \beta (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) ^ {\top} K _ {W w}} \\ & {\qquad = \hat {\alpha} ^ {\top} [ K _ {\dot {D} d} \odot K _ {\dot {X} x} \odot \{B ^ {\top} K _ {W w} \} ].} \end{array}
$$

## E.4 Treatment efects

Derivation of Algorithm 2. By Algorithm 1,

$$
\langle \hat {h}, \phi (d) \otimes \phi (\cdot) \otimes \phi (\cdot) \rangle_ {\mathcal {H}} = \hat {\alpha} ^ {\top} [ K _ {\dot {D} d} \odot K _ {\dot {X}.} \odot \{B ^ {\top} K _ {W}. \} ]
$$

where I leave the arguments $( x , w )$ blank in order to showcase how the confounding bridge estimator will be combined with the kernel mean embedding estimators. By Theorem 2, it is suficient to obtain expressions for kernel mean embedding estimators and to substitute them in. In particular,

$$
1. \theta_ {0} ^ {A T E} \colon \hat {\mu} = n ^ {- 1} \sum_ {i = 1} ^ {n} [ \phi (x _ {i}) \otimes \phi (w _ {i}) ] = n ^ {- 1} \sum_ {i = 1} ^ {n} k (\cdot , x _ {i}) k (\cdot , w _ {i});
$$

$$
2. \theta_ {0} ^ {D S} \colon \hat {\nu} = \tilde {n} ^ {- 1} \sum_ {i = 1} ^ {\tilde {n}} [ \phi (\tilde {x} _ {i}) \otimes \phi (\tilde {w} _ {i}) ] = \tilde {n} ^ {- 1} \sum_ {i = 1} ^ {\tilde {n}} k (\cdot , \tilde {x} _ {i}) k (\cdot , \tilde {w} _ {i});
$$

$$
3. \theta_ {0} ^ {A T T} \colon \hat {\mu} (d) = [ K _ {. X} \odot K _ {. W} ] (K _ {D D} + n \lambda_ {1} I) ^ {- 1} K _ {D d};
$$

$$
4. \theta_ {0} ^ {C A T E} \colon \hat {\mu} (v) = [ K _ {. X} \odot K _ {. W} ] (K _ {V V} + n \lambda_ {2} I) ^ {- 1} K _ {V v}.
$$

I use the n observations of $( d _ { i } , x _ { i } , w _ { i } , z _ { i } )$ to estimate the kernel mean embeddings, and I do not use the m observations of $( { \dot { y } } _ { i } , { \dot { d } } _ { i } , { \dot { x } } _ { i } , { \dot { z } } _ { i } )$ , because the former contain the negative control outcome while the latter do not. The conditional mean embedding expressions follow from [Singh et al., 2019, Algorithm 1]. Matching the blank arguments yields the desired result. □

## F Tuning

## F.1 Simplified setting

The dose response and heterogeneous treatment efect estimators I propose are composed of kernel ridge regressions. The hyperparameters of the new estimators are therefore the same hyperparameters of kernel ridge regression. I present practical tuning procedures for the hyperparameters of (i) ridge regression penalties and (ii) the kernel itself. This appendix is an elaboration of [Singh et al., 2020, Appendix F].

## F.2 Ridge penalty

To begin, I quote a tuning procedure for kernel ridge regression. For simplicity, I focus on the regression of Y on A. It is convenient to tune λ by leave-one-out cross validation (LOOCV), since the validation loss has a closed form solution.

Algorithm 6 (Tuning kernel ridge regression; Algorithm F.1 of [Singh et al., 2020]). Construct the matrices

$$
H _ {\lambda} := I - K _ {A A} (K _ {A A} + n \lambda I) ^ {- 1} \in \mathbb {R} ^ {n \times n}, \quad \tilde {H} _ {\lambda} := d i a g (H _ {\lambda}) \in \mathbb {R} ^ {n \times n}
$$

where ${ \tilde { H } } _ { \lambda }$ has the same diagonal entries as $H _ { \lambda }$ and of diagonal entries of 0. Then set

$$
\lambda^ {*} = \underset {\lambda \in \Lambda} {\operatorname{argmin}} \frac {1}{n} \| \tilde {H} _ {\lambda} ^ {- 1} H _ {\lambda} Y \| _ {2} ^ {2}, \quad \Lambda \subset \mathbb {R}.
$$

The same principles govern the tuning of conditional mean embeddings, which can be viewed as vector valued regressions. Moreover, the same principles govern the tuning of kernel instrumental variable regression, which consists of two kernel ridge regressions. The extension of these tuning procedures to kernel instrumental variable regression is an innovation, and it difers from the tuning procedure in [Singh et al., 2019, Appendix A.5.2]. To simplify the discussion, I focus on the conditional mean embedding $\begin{array} { r } { \mu ( a ) = \int \phi ( y ) \mathrm { d } \mathbb { P } ( y | a ) } \end{array}$ Recall that the closed form solution of the conditional mean embedding estimator using all observations is

$$
\hat {\mu} (a) = K _ {. Y} (K _ {A A} + n \lambda I) ^ {- 1} K _ {A a}.
$$

Algorithm 7 (Tuning conditional mean embedding). Construct the matrices

$$
\begin{array}{l} R := K _ {A A} (K _ {A A} + n \lambda I) ^ {- 1} \in \mathbb {R} ^ {n \times n} \\ S \in \mathbb {R} ^ {n \times n} \text {s.t.} S _ {i j} = 1 \{i = j \} \left\{\frac {1}{1 - R _ {i i}} \right\} ^ {2} \\ T := S (K _ {Y Y} - 2 K _ {Y Y} R ^ {\top} + R K _ {Y Y} R ^ {\top}) \in \mathbb {R} ^ {n \times n} \end{array}
$$

where $R _ { i i }$ is the i-th diagonal element of R. Then set

$$
\lambda^ {*} = \underset {\lambda \in \Lambda} {\operatorname{argmin}} \frac {1}{n} t r (T), \quad \Lambda \subset \mathbb {R}.
$$

Derivation. I prove that $n ^ { - 1 } t r ( T )$ is the LOOCV loss. By definition, the LOOCV loss is

$$
\mathcal {E} (\lambda) := \frac {1}{n} \sum_ {i = 1} ^ {n} \| \phi (y _ {i}) - \hat {\mu} _ {- i} (a _ {i}) \| _ {\mathcal {H} _ {\mathcal {Y}}} ^ {2}
$$

where $\hat { \mu } _ { - i }$ is the conditional mean embedding estimator using all observations except the i-th observation.

Informally, let Φ be the matrix of features for $\{ a _ { i } \}$ , with i-th row $\phi ( a _ { i } ) ^ { \top }$ , and let $Q : = \Phi ^ { \top } \Phi + n \lambda$ . Let Ψ be the matrix of features for $\{ y _ { i } \}$ , with i-th row $\phi ( y _ { i } ) ^ { \top }$ . By the regression first order condition

$$
\begin{array}{c} \hat {\mu} (a) ^ {\top} = \phi (a) ^ {\top} Q ^ {- 1} \Phi^ {\top} \Psi \\ \hat {\mu} _ {- i} (a) ^ {\top} = \phi (a) ^ {\top} \{Q - \phi (a _ {i}) \phi (a _ {i}) ^ {\top} \} ^ {- 1} \{\Phi^ {\top} \Psi - \phi (a _ {i}) \phi (y _ {i}) ^ {\top} \}. \end{array}
$$

The Sherman-Morrison formula for rank one updates gives

$$
\{Q - \phi (a _ {i}) \phi (a _ {i}) ^ {\top} \} ^ {- 1} = Q ^ {- 1} + \frac {Q ^ {- 1} \phi (a _ {i}) \phi (a _ {i}) ^ {\top} Q ^ {- 1}}{1 - \phi (a _ {i}) ^ {\top} Q ^ {- 1} \phi (a _ {i})}.
$$

Let $\beta _ { i } : = \phi ( a _ { i } ) ^ { \top } Q ^ { - 1 } \phi ( a _ { i } )$ . Then

$$
\begin{array}{r l} & {\hat {\mu} _ {- i} (a) ^ {\top} = \phi (a _ {i}) ^ {\top} \left\{Q ^ {- 1} + \frac {Q ^ {- 1} \phi (a _ {i}) \phi (a _ {i}) ^ {\top} Q ^ {- 1}}{1 - \beta_ {i}} \right\} \{\Phi^ {\top} \Psi - \phi (a _ {i}) \phi (y _ {i}) ^ {\top} \}} \\ & {\qquad = \phi (a _ {i}) ^ {\top} \left\{I + \frac {Q ^ {- 1} \phi (a _ {i}) \phi (a _ {i}) ^ {\top}}{1 - \beta_ {i}} \right\} \{\hat {\mu} ^ {\top} - Q ^ {- 1} \phi (a _ {i}) \phi (y _ {i}) ^ {\top} \}} \\ & {\qquad = \left\{1 + \frac {\beta_ {i}}{1 - \beta_ {i}} \right\} \phi (a _ {i}) ^ {\top} \{\hat {\mu} ^ {\top} - Q ^ {- 1} \phi (a _ {i}) \phi (y _ {i}) ^ {\top} \}} \\ & {\qquad = \left\{1 + \frac {\beta_ {i}}{1 - \beta_ {i}} \right\} \{\hat {\mu} (a _ {i}) ^ {\top} - \beta_ {i} \phi (y _ {i}) ^ {\top} \}} \\ & {\qquad = \frac {1}{1 - \beta_ {i}} \{\hat {\mu} (a _ {i}) ^ {\top} - \beta_ {i} \phi (y _ {i}) ^ {\top} \}} \end{array}
$$

i.e. $\hat { \mu } _ { - i }$ can be expressed in terms of $\hat { \mu }$ . Note that

$$
\begin{array}{r l} \phi (y _ {i}) - \hat {\mu} _ {- i} (a _ {i}) & = \phi (y _ {i}) - \frac {1}{1 - \beta_ {i}} \{\hat {\mu} (a _ {i}) - \beta_ {i} \phi (y _ {i}) \} \\ & = \phi (y _ {i}) + \frac {1}{1 - \beta_ {i}} \{\beta_ {i} \phi (y _ {i}) - \hat {\mu} (a _ {i}) \} \\ & = \frac {1}{1 - \beta_ {i}} \{\phi (y _ {i}) - \hat {\mu} (a _ {i}) \}. \end{array}
$$

Substituting back into the LOOCV loss

$$
\begin{array}{c} \frac {1}{n} \sum_ {i = 1} ^ {n} \| \phi (y) _ {i} - \hat {\mu} _ {- i} (a _ {i}) \| _ {\mathcal {H} _ {\mathcal {Y}}} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\| \{\phi (y _ {i}) - \hat {\mu} (a _ {i}) \} \left\{\frac {1}{1 - \beta_ {i}} \right\} \right\| _ {\mathcal {H} _ {\mathcal {Y}}} ^ {2} \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{\frac {1}{1 - \beta_ {i}} \right\} ^ {2} \| \phi (y _ {i}) - \hat {\mu} (a _ {i}) \| _ {\mathcal {H} _ {\mathcal {Y}}} ^ {2}. \end{array}
$$

By arguments in [Singh et al., 2020, Appendix F],

$$
\beta_ {i} = [ K _ {A A} (K _ {A A} + n \lambda I) ^ {- 1} ] _ {i i}
$$

i.e. $\beta _ { i }$ can be calculated as the i-th diagonal element of $K _ { A A } ( K _ { A A } + n \lambda I ) ^ { - 1 }$ . Moreover

$$
\begin{array}{r l} & {\| \phi (y _ {i}) - \hat {\mu} (a _ {i}) \| _ {\mathcal {H}} ^ {2}} \\ & {= k (y _ {i}, y _ {i}) - 2 K _ {y _ {i} Y} (K _ {A A} + n \lambda I) ^ {- 1} K _ {A a _ {i}} + K _ {a _ {i} A} (K _ {A A} + n \lambda I) ^ {- 1} K _ {Y Y} (K _ {A A} + n \lambda I) ^ {- 1} K _ {A a _ {i}}} \\ & {= [ K _ {Y Y} - 2 K _ {Y Y} (K _ {A A} + n \lambda I) ^ {- 1} K _ {A A} + K _ {A A} (K _ {A A} + n \lambda I) ^ {- 1} K _ {Y Y} (K _ {A A} + n \lambda I) ^ {- 1} K _ {A A} ] _ {i i}} \end{array}
$$

i.e. $\lVert \phi ( y _ { i } ) - \hat { \mu } ( a _ { i } ) \rVert _ { \mathcal { H } } ^ { 2 }$ can be calculated as the i-th diagonal element of a matrix as well. Substituting these results back into the LOOCV loss gives the final expression. □

## F.3 Kernel

The Gaussian kernel satisfies the requirements of Assumption 4. Formally, the kernel

$$
k (a, a ^ {\prime}) = \exp \left\{- \frac {1}{2} \frac {\| a - a ^ {\prime} \| _ {\mathcal {A}} ^ {2}}{\sigma^ {2}} \right\}
$$

is continuous, bounded, and characteristic. The kernel hyperparameter $\sigma$ is called the lengthscale. A simple heuristic is to set σ as the median interpoint distance of $\{ a _ { i } \} _ { i = 1 } ^ { n }$ 2 where the interpoint distance between observations i and j is $\| a _ { i } - a _ { j } \| _ { A }$

I use the Gaussian kernel in experiments. When the input a is a vector rather than a scalar, I use the kernel obtained as the product of scalar kernels for each input dimension, following [Singh et al., 2019, Singh et al., 2020]. For example, if $\mathcal { A } \subset \mathbb { R } ^ { d }$ then

$$
k (a, a ^ {\prime}) = \prod_ {j = 1} ^ {d} \exp \left\{- \frac {1}{2} \frac {[ a _ {j} - a _ {j} ^ {\prime} ] ^ {2}}{\sigma_ {j} ^ {2}} \right\}.
$$

Each lengthscale $\sigma _ { j }$ is set as the median interpoint distance for that input dimension. In principle, one could instead use LOOCV to tune kernel hyperparameters as above. The LOOCV approach to tuning lengthscales $\{ \sigma _ { j } \}$ is impractical in high dimensions, since there is a lengthscale $\sigma _ { j }$ for each input dimension.

## F.4 Time complexity

As in classic kernel ridge regression, the time consuming step is tuning. We see in Algorithms 6 and 7 that to choose the ridge penalty hyperparameter $\lambda ^ { * }$ , one must invert the matrix

$$
K _ {A A} + n \lambda I \in \mathbb {R} ^ {n \times n}
$$

for each value λ in the grid Λ. Inversion of such a matrix has complexity $O ( n ^ { 3 } )$ ; the sample size n is the limiting factor. The same is true for the two ridge penalty hyperparameters in the confounding bridge estimated by Algorithm 1, and for the additional ridge penalty hyperparameter that appears when estimating the heterogeneous treatment efect in Algorithm 2. Therefore tuning of the heterogeneous treatment efect in Algorithm 2 takes roughly three times as long as tuning of a kernel ridge regression, whose runtime scales as $O ( n ^ { 3 } )$ .

In the simulations of Section 6 and Appendix I, I implement kernel methods with and without negative controls, across various designs, with several iterations, which compounds the time complexity of the tuning step. It is therefore feasible to implement only 100 iterations when the sample size is $n = 1 0 , 0 0 0$

In practice, an analyst would implement the method once for one sample size, which is feasible on a personal laptop. The dose response curve can be estimated in a matter of seconds for $n \in \{ 1 0 0 , 5 0 0 \}$ , in a matter of minutes for $n \in \{ 1 0 0 0 , 5 0 0 0 \}$ , and a matter of hours for $n = 1 0 , 0 0 0$ . A vast literature considers how to speed up kernel methods by replacing the kernel matrix with a low rank approximation. Appendix E.3 of [Dikkala et al., 2020] discusses popular techniques and their implementation in NPIV. I pose as a question for future work how to extend the main results of this paper to accommodate kernel matrix approximations.

## G Confounding bridge consistency proof

In this appendix, I (i) state a probability lemma, (ii) explicitly specialize the smoothness assumptions, (iii) provide regression lemmas, (iv) prove technical bounds, and (v) prove uniform consistency of the confounding bridge. This is the most technically demanding appendix.

## G.1 Probability lemma

Lemma 4 (Lemma 2 of [Smale and Zhou, 2007]). Let ξ be a random variable taking values in a real separable Hilbert space $\kappa .$ Suppose there exists $\tilde { M }$ and $\sigma ^ { 2 }$ such that

$$
\| \xi \| _ {\mathcal {K}} \leq \tilde {M} <   \infty \quad \text { almost   surely }, \quad \mathbb {E} \| \xi \| _ {\mathcal {K}} ^ {2} \leq \sigma^ {2}.
$$

Then $\forall n \in \mathbb { N } , \forall \eta \in ( 0 , 1 )$ 1),

$$
\mathbb {P} \bigg [ \bigg \| \frac {1}{n} \sum_ {i = 1} ^ {n} \xi_ {i} - \mathbb {E} \xi \bigg \| _ {\mathcal {K}} \leq \frac {2 \tilde {M} \ln (2 / \eta)}{n} + \sqrt {\frac {2 \sigma^ {2} \ln (2 / \eta)}{n}} \bigg ] \geq 1 - \eta .
$$

## G.2 Smoothness assumptions

Let the symbol  mean composition. I use it to emphasize the composition of operators. To lighten notation, let $\mathcal { H } _ { R F } = \mathcal { H } _ { \mathcal { D } } \otimes \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { Z } }$ , which is named for the reduced form.

Assumption 8 (Smoothness of conditional expectation). Assume

1. The conditional expectation operator $E _ { 0 }$ is well specified as a Hilbert-Schmidt operator between RKHSs, i.e. $E _ { 0 } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { R F } )$ , where

$$
E _ {0}: \mathcal {H} _ {\mathcal {W}} \to \mathcal {H} _ {R F}, \quad h (\cdot) \mapsto \mathbb {E} [ h (W) | D = \cdot , X = \cdot , Z = \cdot ].
$$

2. The conditional expectation operator is a particularly smooth element of $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { R F } )$ Formally, define the covariance operator $T _ { 0 } : = \mathbb { E } [ \phi ( D , X , Z ) \otimes \phi ( D , X , Z ) ]$ for $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { R F } )$ I assume there exists $G _ { 0 } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { R F } )$ such that $E _ { 0 } = ( T _ { 0 } ) ^ { \frac { c _ { 0 } - 1 } { 2 } } \circ G _ { 0 } , c _ { 0 } \in ( 1 , 2 ]$ and $\| G _ { 0 } \| _ { \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { R F } ) } ^ { 2 } \le \zeta _ { 0 }$

Assumption 9 (Smoothness of confounding bridge). Assume

1. The confounding bridge $h _ { 0 }$ is well specified, i.e. $h _ { 0 } \in { \mathcal { H } } _ { \mu } \subset { \mathcal { H } } .$

2. The confounding bridge is a particularly smooth element of ${ \mathcal { H } } _ { \mu }$ . Formally, define the covariance operator $T : = \mathbb { E } [ \mu ( D , X , Z ) \otimes \mu ( D , X , Z ) ]$ , where $\mu ( d , x , z ) = \phi ( d ) \otimes $ $\phi ( x ) \otimes \mu _ { w } ( d , x , z )$ , for ${ \mathcal { H } } _ { \mu }$ . I assume there exists $g \in { \mathcal { H } }$ such that $h _ { 0 } = T ^ { \frac { c - 1 } { 2 } } \circ g$ $c \in ( 1 , 2 ]$ , and $\| g \| _ { \mathcal { H } } ^ { 2 } \leq \zeta$

Proposition 8. The following assumptions are equivalent

1. Assumption 6 with $\mathcal { A } _ { 0 } = \mathcal { W }$ and $\mathcal { B } _ { 0 } = \mathcal { D } \times \mathcal { X } \times \mathcal { Z }$ is equivalent to Assumption 8

2. Assumption 7 is equivalent to Assumption 9

Proof. The result is immediate from [Caponnetto and De Vito, 2007, Remark 2]. The expanded expressions are more convenient for analysis. □

## G.3 Regression lemmas

Let n be the number of observations of $( d _ { i } , x _ { i } , w _ { i } , z _ { i } )$ used to estimate the stage 1 conditional mean embedding $\mu ( d , x , z ) = \phi ( d ) \otimes \phi ( x ) \otimes \mu _ { w } ( d , x , z )$ by kernel ridge regression with regularization parameter λ. Let m be the number of observations of $( { \dot { y } } _ { i } , { \dot { d } } _ { i } , { \dot { x } } _ { i } , { \dot { z } } _ { i } )$ used to estimate the stage 2 confounding bridge operator $h _ { 0 }$ by kernel ridge regression with regularization parameter $\xi .$ .

Proposition 9. Suppose Assumptions 2 and 4 hold, and $h _ { 0 } \in \mathcal { H }$ . Then

$$
\mathbb {E} [ \mu (D, X, Z) Y ] = T h _ {0}.
$$

Proof. Appealing to the definition of $T .$ , the argument in the proof of Theorem 2, and the law of iterated expectations,

$$
\begin{array}{l} T h _ {0} = \mathbb {E} [ \mu (D, X, Z) \otimes \mu (D, X, Z) ] h _ {0} \\ \qquad = \mathbb {E} [ \mu (D, X, Z) \langle \mu (D, X, Z), h _ {0} \rangle_ {\mathcal {H}} ] \\ \qquad = \mathbb {E} [ \mu (D, X, Z) \gamma_ {0} (D, X, Z) ] \\ \qquad = \mathbb {E} [ \mu (D, X, Z) Y ]. \end{array}
$$

口

To facilitate analysis, define the following quantities.

## Definition 3 (Confounding bridge risk). Define

1. Target bridge

$$
h _ {0} \in \underset {h \in \mathcal {H}} {\operatorname{argmin}} \mathcal {E} (h), \quad \mathcal {E} (h) = \mathbb {E} [ \{Y - \langle h, \mu (D, X, Z) \rangle_ {\mathcal {H}} \} ^ {2} ].
$$

2. Regularized bridge

$$
h _ {\xi} = \underset {h \in \mathcal {H}} {\operatorname{argmin}}   \mathcal {E} _ {\xi} (h), \quad \mathcal {E} _ {\xi} (h) = \mathcal {E} (h) + \xi \| h \| _ {\mathcal {H}} ^ {2}.
$$

3. Empirical regularized bridge

$$
h _ {\xi} ^ {m} = \underset {h \in \mathcal {H}} {\operatorname{argmin}} \mathcal {E} _ {\xi} ^ {m} (h), \quad \mathcal {E} _ {\xi} ^ {m} (h) = \frac {1}{m} \sum_ {i = 1} ^ {m} \{\dot {y} _ {i} - \langle h, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \} ^ {2} + \xi \| h \| _ {\mathcal {H}} ^ {2}.
$$

## 4. Estimated bridge

$$
\hat {h} _ {\xi} ^ {m} = \underset {h \in \mathcal {H}} {\mathrm{argmin}} \hat {\mathcal {E}} _ {\xi} ^ {m} (h), \quad \hat {\mathcal {E}} _ {\xi} ^ {m} (h) = \frac {1}{m} \sum_ {i = 1} ^ {m} \{\dot {y} _ {i} - \langle h, \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \} ^ {2} + \xi \| h \| _ {\mathcal {H}} ^ {2}
$$

where $\mu _ { \lambda } ^ { n } ( d , x , z )$ is an estimator of the conditional mean embedding $\mu ( d , x , z )$

Proposition 10 (Closed form). $\forall \xi > 0$ , the solution $h _ { \xi } ^ { m }$ to $\mathcal { E } _ { \xi } ^ { m }$ and the solution $\hat { h } _ { \xi } ^ { m }$ to $\hat { \mathcal { E } } _ { \xi } ^ { m }$ both exist, are unique, and

$$
h _ {\xi} ^ {m} = (\mathbf {T} + \boldsymbol {\xi}) ^ {- 1} \mathbf {g}, \quad \mathbf {T} = \frac {1}{m} \sum_ {i = 1} ^ {m} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}), \quad \mathbf {g} = \frac {1}{m} \sum_ {i = 1} ^ {m} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \dot {y} _ {i},
$$

$$
\hat {h} _ {\xi} ^ {m} = (\mathbf {\hat {T}} + \boldsymbol {\xi}) ^ {- 1} \mathbf {\hat {g}}, \quad \mathbf {\hat {T}} = \frac {1}{m} \sum_ {i = 1} ^ {m} \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}), \quad \mathbf {\hat {g}} = \frac {1}{m} \sum_ {i = 1} ^ {m} \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \dot {y} _ {i}.
$$

Proof. The result is a simplification of [Singh et al., 2019, Theorem 3].

## G.4 Bias and variance of second stage

Proposition 11 (Bias). Suppose Assumptions 2, 4, 5, and 9 hold. Then

$$
\| h _ {\xi} - h _ {0} \| _ {\mathcal {H}} \leq \xi^ {\frac {c - 1}{2}} \sqrt {\zeta}.
$$

Proof. I generalize [Smale and Zhou, 2005, Theorem 4]. By Assumption 9, there exists a function $g \in { \mathcal { H } }$ such that

$$
g = T ^ {\frac {1 - c}{2}} h _ {0} = \sum_ {k} \eta_ {k} ^ {\frac {1 - c}{2}} e _ {k} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}}
$$

where $\{ \eta _ { k } \}$ are the eigenvalues and $\{ e _ { k } \}$ are the eigenfunctions of $T$ . By Proposition 9, write

$$
h _ {\xi} - h _ {0} = [ (T + \xi) ^ {- 1} T - I ] h _ {0} = \sum_ {k} \left(\frac {\eta_ {k}}{\eta_ {k} + \xi} - 1\right) e _ {k} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}}.
$$

Therefore

$$
\begin{array}{r l} & {\| h _ {\xi} - h _ {0} \| _ {\mathcal {H}} ^ {2} = \sum_ {k} \left(\frac {\eta_ {k}}{\eta_ {k} + \xi} - 1\right) ^ {2} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}} ^ {2}} \\ & {\quad = \sum_ {k} \left(\frac {\xi}{\eta_ {k} + \xi}\right) ^ {2} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}} ^ {2}} \\ & {\quad = \sum_ {k} \left(\frac {\xi}{\eta_ {k} + \xi}\right) ^ {2} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}} ^ {2} \left(\frac {\xi}{\xi} \cdot \frac {\eta_ {k}}{\eta_ {k}} \cdot \frac {\eta_ {k} + \xi}{\eta_ {k} + \xi}\right) ^ {c - 1}} \\ & {\quad = \xi^ {c - 1} \sum_ {k} \eta_ {k} ^ {1 - c} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}} ^ {2} \left(\frac {\xi}{\eta_ {k} + \xi}\right) ^ {3 - c} \left(\frac {\eta_ {k}}{\eta_ {k} + \xi}\right) ^ {c - 1}} \\ & {\quad \leq \xi^ {c - 1} \sum_ {k} \eta_ {k} ^ {1 - c} \langle e _ {k}, h _ {0} \rangle_ {\mathcal {H}} ^ {2}} \\ & {\quad = \xi^ {c - 1} \| g \| _ {\mathcal {H}} ^ {2}} \\ & {\quad \leq \xi^ {c - 1} \zeta .} \end{array}
$$

Lemma 5 (Helpful bounds). Suppose Assumptions 4, 5, and 7 hold. I adopt the language of [Caponnetto and De Vito, 2007].

1. The generalized reconstruction error is $\begin{array} { r } { B ( \xi ) = \| h _ { \xi } - h _ { 0 } \| _ { \mathcal { H } } ^ { 2 } \le \zeta \cdot \xi ^ { c - 1 } } \end{array}$

2. The generalized efective dimension is $\mathcal { N } ( \xi ) = \mathrm { t r } \{ ( T + \xi ) ^ { - 1 } T \} \le C ( \pi / b ) \{ \sin ( \pi / b ) \} ^ { - 1 } \xi ^ { - 1 / b }$

Proof. The first result is a corollary of Proposition 11. The second result follows from [Sutherland, 2017, eq. f], appealing to the efective dimension condition in Assumption 7.

Lemma 6 (Decomposition of variance). The following bound holds:

$$
\begin{array}{r l} & {\| h _ {\xi} ^ {m} - h _ {\xi} \| _ {\mathcal {H}} \leq \| (T + \xi) ^ {- 1 / 2} \{\mathbf {g} - (\mathbf {T} + \xi) h _ {\xi} \} \| _ {\mathcal {H}}} \\ & {\qquad \cdot \| (T + \xi) ^ {1 / 2} (\mathbf {T} + \xi) ^ {- 1} (T + \xi) ^ {1 / 2} \| _ {o p}} \\ & {\qquad \cdot \| (T + \xi) ^ {- 1 / 2} \| _ {o p}.} \end{array}
$$

Moreover, in the first factor,

$$
(T + \xi) ^ {- 1 / 2} \{\mathbf {g} - (\mathbf {T} + \xi) h _ {\xi} \} = \frac {1}{m} \sum_ {i = 1} ^ {m} \dot {\xi} _ {i} - \mathbb {E} [ \dot {\xi} ]
$$

where

$$
\begin{array}{r l} & {\dot {\xi} _ {i} = (T + \xi) ^ {- 1 / 2} \{\mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \dot {y} _ {i} - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) h _ {\xi} \}} \\ & {\qquad = (T + \xi) ^ {- 1 / 2} \{\mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) [ \dot {y} _ {i} - \langle h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} ] \}.} \end{array}
$$

Proof. The result mirrors [Fischer and Steinwart, 2020, eq. 44]. For the decomposition of the first factor, the definitions in Proposition 10 give

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \dot {\xi} _ {i} = (T + \xi) ^ {- 1 / 2} (\mathbf {g} - \mathbf {T} h _ {\xi}).
$$

Meanwhile

$$
\begin{array}{l} \mathbb {E} [ \dot {\xi} ] = (T + \xi) ^ {- 1 / 2} \{\mathbb {E} [ \mu (D, X, Z) Y ] - T h _ {\xi} \} \\ \qquad = (T + \xi) ^ {- 1 / 2} \{\mathbb {E} [ \mu (D, X, Z) Y ] - T h _ {\xi} - \xi h _ {\xi} + \xi h _ {\xi} \} \\ \qquad = (T + \xi) ^ {- 1 / 2} \{\mathbb {E} [ \mu (D, X, Z) Y ] - (T + \xi) h _ {\xi} + \xi h _ {\xi} \} \\ \qquad = (T + \xi) ^ {- 1 / 2} \{\mathbb {E} [ \mu (D, X, Z) Y ] - \mathbb {E} [ \mu (D, X, Z) Y ] + \xi h _ {\xi} \} \\ \qquad = (T + \xi) ^ {- 1 / 2} \{\xi h _ {\xi} \} \end{array}
$$

as desired.

Lemma 7 (Bounding the first factor). Suppose Assumptions 4 and 5 hold. Then with probability $1 - \delta / 2 .$ , the first factor in Lemma 6 is bounded as

$$
\begin{array}{l} \| (T + \xi) ^ {- 1 / 2} \{\mathbf {g} - (\mathbf {T} + \xi) h _ {\xi} \} \| _ {\mathcal {H}} \\ \leq 4 \log (4 / \delta) \left\{\frac {\kappa C + \kappa^ {2} \| h _ {0} \| _ {\mathcal {H}}}{m \xi^ {1 / 2}} + \frac {\kappa^ {2} \mathcal {B} (\xi) ^ {1 / 2}}{m \xi^ {1 / 2}} + \frac {(C + \kappa \| h _ {0} \| _ {\mathcal {H}}) \mathcal {N} (\xi) ^ {1 / 2}}{m ^ {1 / 2}} + \frac {\kappa \mathcal {B} (\xi) ^ {1 / 2} \mathcal {N} (\xi) ^ {1 / 2}}{m ^ {1 / 2}} \right\}. \end{array}
$$

Proof. I verify the conditions of Lemma 4. Let

$$
\dot {\xi} _ {i} = (T + \xi) ^ {- 1 / 2} \{\mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) [ \dot {y} _ {i} - \langle h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} ] \}.
$$

I proceed in steps.

1. First moment.

Observe that

$$
\begin{array}{r l} & {\| \dot {\xi} _ {i} \| _ {\mathcal {H}} = \| (T + \xi) ^ {- 1 / 2} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) [ \dot {y} _ {i} - \langle h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} ] \| _ {\mathcal {H}}} \\ & {\qquad \leq \| (T + \xi) ^ {- 1 / 2} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} \cdot | \dot {y} _ {i} - \langle h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} |.} \end{array}
$$

Moreover

$$
\| (T + \xi) ^ {- 1 / 2} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} \leq \| (T + \xi) ^ {- 1 / 2} \| _ {o p} \| \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} \leq \frac {\kappa}{\xi^ {1 / 2}}
$$

and

$$
\begin{array}{r l} & {| \dot {y} _ {i} - \langle h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} | \leq | \dot {y} _ {i} - \langle h _ {0}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} | + | \langle h _ {0} - h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} |} \\ & {\qquad \leq C + \kappa \| h _ {0} \| _ {\mathcal {H}} + \kappa \| h _ {0} - h _ {\xi} \| _ {\mathcal {H}}} \\ & {\qquad \leq C + \kappa \{\| h _ {0} \| _ {\mathcal {H}} + \mathcal {B} (\xi) ^ {1 / 2} \}.} \end{array}
$$

In summary,

$$
\| \dot {\xi} _ {i} \| _ {\mathcal {H}} \leq \frac {\kappa}{\xi^ {1 / 2}} [ C + \kappa \{\| h _ {0} \| _ {\mathcal {H}} + \mathcal {B} (\xi) ^ {1 / 2} \} ].
$$

## 2. Second moment.

Next, write

$$
\begin{array}{l} \mathbb {E} (\| \dot {\xi} _ {i} \| _ {\mathcal {H}} ^ {2}) \\ = \int [ \dot {y} _ {i} - \langle h _ {\xi}, \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} ] ^ {2} \langle \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}), (T + \xi) ^ {- 1} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (\dot {y} _ {i}, \dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \\ \leq \sup _ {y, d, x, z} [ y - \langle h _ {\xi}, \mu (d, x, z) \rangle_ {\mathcal {H}} ] ^ {2} \int \langle \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}), (T + \xi) ^ {- 1} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}). \end{array}
$$

Focusing on the former factor, as argued above,

$$
\sup _ {y, d, x, z} \left[ y - \langle h _ {\xi}, \mu (d, x, z) \rangle_ {\mathcal {H}} \right] ^ {2} \leq \left[ C + \kappa \{\| h _ {0} \| _ {\mathcal {H}} + \mathcal {B} (\xi) ^ {1 / 2} \} \right] ^ {2}.
$$

Focusing on the latter factor,

$$
\begin{array}{l} \int \langle \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}), (T + \xi) ^ {- 1} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \rangle_ {\mathcal {H}} \mathrm{d} \mathbb {P} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \\ = \int \mathrm{tr} [ (T + \xi) ^ {- 1} \{\mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \} ] \mathrm{d} \mathbb {P} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \\ = \mathrm{tr} \{(T + \xi) ^ {- 1} T \} \\ = \mathcal {N} (\xi). \end{array}
$$

In summary,

$$
\mathbb {E} (\| \dot {\xi_ {i}} \| _ {\mathcal {H}} ^ {2}) \leq \mathcal {N} (\xi) \left[ C + \kappa \{\| h _ {0} \| _ {\mathcal {H}} + \mathcal {B} (\xi) ^ {1 / 2} \} \right] ^ {2}.
$$

3. Concentration.

Therefore with probability $1 - \delta / 2$

$$
\begin{array}{r l} & {\left\| \frac {1}{m} \sum_ {i = 1} ^ {m} \dot {\xi} _ {i} - \mathbb {E} [ \dot {\xi} ] \right\| _ {\mathcal {H}}} \\ & {\leq \frac {2 \log (4 / \delta)}{m} \frac {\kappa}{\xi^ {1 / 2}} [ C + \kappa \{\| h _ {0} \| _ {\mathcal {H}} + \mathcal {B} (\xi) ^ {1 / 2} \} ] + \left[ \frac {2 \log (4 / \delta)}{m} \mathcal {N} (\xi) \left[ C + \kappa \{\| h _ {0} \| _ {\mathcal {H}} + \mathcal {B} (\xi) ^ {1 / 2} \} \right] ^ {2} \right] ^ {1 / 2}} \\ & {\leq 4 \log (4 / \delta) \left\{\frac {\kappa C + \kappa^ {2} \| h _ {0} \| _ {\mathcal {H}}}{m \xi^ {1 / 2}} + \frac {\kappa^ {2} \mathcal {B} (\xi) ^ {1 / 2}}{m \xi^ {1 / 2}} + \frac {(C + \kappa \| h _ {0} \| _ {\mathcal {H}}) \mathcal {N} (\xi) ^ {1 / 2}}{m ^ {1 / 2}} + \frac {\kappa \mathcal {B} (\xi) ^ {1 / 2} \mathcal {N} (\xi) ^ {1 / 2}}{m ^ {1 / 2}} \right\}.} \end{array}
$$

Remark 1 (Suficiently large m). In the finite sample, I assume a certain inequality holds when bounding the second factor:

$$
m \geq 8 \kappa^ {2} \log (4 / \delta) \cdot \xi \cdot \log \left\{2 e \cdot \mathcal {N} (\xi) \frac {\| T \| _ {o p} + \xi}{\| T \| _ {o p}} \right\}, \quad \kappa = \kappa_ {d} \cdot \kappa_ {x} \cdot \kappa_ {w}.\tag{5}
$$

Ultimately, I will choose $\xi = m ^ { - 1 / ( c + 1 / b ) }$ in Theorem 4. This choice of $\xi$ together with the bound on generalized efective dimension $\mathcal { N } ( \xi )$ in Lemma 5 imply that there exists an $m _ { 0 }$ such that for all $m \geq m _ { 0 }$ , (5) holds, as argued by [Fischer and Steinwart, 2020, Proof of Theorem 1]. I use the phrase “m suficiently large” when I appeal to this logic, and I summarize the final bound using $O ( \cdot )$ notation.

Lemma 8 (Bounding the second factor). Suppose Assumptions 4 and 5 hold. Further assume (5) holds. Then probability $1 - \delta / 2$ , the second factor in Lemma 6 is bounded as

$$
\| (T + \xi) ^ {1 / 2} (\mathbf {T} + \xi) ^ {- 1} (T + \xi) ^ {1 / 2} \| _ {o p} \leq 3.
$$

Proof. The result follows from [Fischer and Steinwart, 2020, eq. 44b, 47]. In particular, my assumptions sufice for the properties used in [Fischer and Steinwart, 2020, Lemma 17] to hold, using the same argument as [Singh et al., 2020, Lemma I.4]. Separability of original <sup>spaces</sup> <sup>together</sup> <sup>with</sup> <sup>boundedness</sup> <sup>of</sup> <sup>kernels</sup> <sup>imply</sup> <sup>that</sup> H <sup>is</sup> <sup>separable</sup> <sup>[Steinwart</sup> <sup>and</sup> <sup>Christmann,</sup> <sup>2008,</sup> Lemma 4.33]. Next, I verify the assumptions called EMB, EVD, and SRC. Boundedness of the kernel implies EMB with $a = 1$ . EVD is the assumption I call efective dimension, parametrized by $b \geq 1$ . SRC is the assumption I call the source condition, parametrized by $c \in ( 1 , 2 ]$ □

Lemma 9 (Bounding the third factor). With probability one, the third factor in Lemma 6 is bounded as

$$
\| (T + \xi) ^ {- 1 / 2} \| _ {o p} \leq \xi^ {- 1 / 2}.
$$

Proof. The result follows from the definition of operator norm.

Proposition 12 (Variance). Suppose Assumptions 2, 4, 5, and 6 hold. Then $\forall \delta \in ( 0 , 1 )$ ， for m suficiently large, the following holds with probability $1 - \delta \colon$

$$
\| h _ {\xi} ^ {m} - h _ {\xi} \| _ {\mathcal {H}} \leq C \log (4 / \delta) \left\{\frac {1}{m \xi} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} \right\}.
$$

Proof. I combine the previous lemmas to generalize [Fischer and Steinwart, 2020, Theorem 16]. By Lemmas 6, 7, 8, and 9, if (5) holds, then with probability $1 - \delta$

$$
\begin{array}{l} \| h _ {\xi} ^ {m} - h _ {\xi} \| _ {\mathcal {H}} \\ \leq \frac {1 2 \log (4 / \delta)}{\xi^ {1 / 2}} \left\{\frac {\kappa C + \kappa^ {2} \| h _ {0} \| _ {\mathcal {H}}}{m \xi^ {1 / 2}} + \frac {\kappa^ {2} \mathcal {B} (\xi) ^ {1 / 2}}{m \xi^ {1 / 2}} + \frac {(C + \kappa \| h _ {0} \| _ {\mathcal {H}}) \mathcal {N} (\xi) ^ {1 / 2}}{m ^ {1 / 2}} + \frac {\kappa \mathcal {B} (\xi) ^ {1 / 2} \mathcal {N} (\xi) ^ {1 / 2}}{m ^ {1 / 2}} \right\}. \end{array}
$$

Next, recall the bounds in Lemma 5. When $\xi \le 1$

$$
\mathcal {B} (\xi) ^ {1 / 2} \leq \zeta^ {1 / 2} \xi^ {\frac {c - 1}{2}} \leq \zeta^ {1 / 2}.
$$

For brevity, write

$$
\mathcal {N} (\xi) ^ {1 / 2} \leq C ^ {\prime} \xi^ {- \frac {1}{2 b}}.
$$

Therefore when $\xi \le 1$ the bound simplifies as

$$
\| h _ {\xi} ^ {m} - h _ {\xi} \| _ {\mathcal {H}} \leq C \log (4 / \delta) \left\{\frac {1}{m \xi} + \frac {1}{m ^ {1 / 2} \xi^ {1 / (2 b) + 1 / 2}} \right\}.
$$

## G.5 Bounds

Proposition 13. Suppose Assumption 4 holds. Assume that $\forall d \in \mathcal { D } , x \in \mathcal { X } , z \in \mathcal { Z }$ 2 $\begin{array} { r } { \| \mu _ { \lambda } ^ { n } ( d , x , z ) - \mu ( d , x , z ) \| _ { \mathcal { H } } \leq r _ { \mu } ( n , \delta , b _ { 0 } , c _ { 0 } ) . } \end{array}$

1. Then $\begin{array} { r } { \| \hat { \mathbf { T } } - \mathbf { T } \| _ { \mathcal { L } ( \mathcal { H } ) } \leq \{ 2 \kappa + r _ { \mu } ( n , \delta , b _ { 0 } , c _ { 0 } ) \} r _ { \mu } ( n , \delta , b _ { 0 } , c _ { 0 } ) . } \end{array}$

2. If in addition Assumption 5 holds then $\| \hat { \mathbf { g } } - \mathbf { g } \| _ { \mathcal { H } } \leq C r _ { \mu } ( n , \delta , b _ { 0 } , c _ { 0 } )$


Proof. For the former result, write

$$
\begin{array}{r l} & {\hat {\mathbf {T}} - \mathbf {T} = \frac {1}{m} \sum_ {i = 1} ^ {m} \{\mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \}} \\ & {\qquad = \frac {1}{m} \sum_ {i = 1} ^ {m} \{\mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) + \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) + \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \}.} \end{array}
$$

Hence

$$
\begin{array}{l} \| \mathbf {T} - \hat {\mathbf {T}} \| _ {\mathcal {L} (\mathcal {H})} \leq \frac {1}{m} \sum_ {i = 1} ^ {m} \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {L} (\mathcal {H})} \\ \qquad + \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {L} (\mathcal {H})}. \end{array}
$$

In the first term,

$$
\begin{array}{r l} & {\| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {L} (\mathcal {H})}} \\ & {= \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} \cdot \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}}} \\ & {\leq \left(\| \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} + \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}}\right) \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}}} \\ & {\leq \{\kappa + r _ {\mu} (n, \delta , b _ {0}, c _ {0}) \} r _ {\mu} (n, \delta , b _ {0}, c _ {0}).} \end{array}
$$

In the second term

$$
\begin{array}{l} \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \otimes \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {L} (\mathcal {H})} \\ = \| \mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} \cdot \| \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \| _ {\mathcal {H}} \\ \leq \kappa \cdot r _ {\mu} (n, \delta , b _ {0}, c _ {0}). \end{array}
$$

In summary,

$$
\| \mathbf {T} - \hat {\mathbf {T}} \| _ {\mathcal {L} (\mathcal {H})} \leq \left\{2 \kappa + r _ {\mu} (n, \delta , b _ {0}, c _ {0}) \right\} r _ {\mu} (n, \delta , b _ {0}, c _ {0}).
$$

For the latter result, write

$$
\hat {\mathbf {g}} - \mathbf {g} = \frac {1}{m} \sum_ {i = 1} ^ {m} \{\mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \} \dot {y} _ {i}.
$$

Hence

$$
\| \hat {\mathbf {g}} - \mathbf {g} \| _ {\mathcal {H}} \leq \frac {1}{m} \sum_ {i = 1} ^ {m} \| \{\mu_ {\lambda} ^ {n} (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) - \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \} \| _ {\mathcal {H}} \cdot | \dot {y} _ {i} | \leq C r _ {\mu} (n, \delta , b _ {0}, c _ {0}).
$$

Proposition 14. Suppose Assumption 4 holds. If $\forall d \in { \mathcal { D } } , x \in { \mathcal { X } } , z \in { \mathcal { Z } } , \| \mu _ { \lambda } ^ { n } ( d , x , z ) -$ $\mu ( d , x , z ) \| _ { \mathcal { H } } \leq r _ { \mu } ( n , \delta , b _ { 0 } , c _ { 0 } )$ then

$$
\| (\hat {\mathbf {T}} + \xi) ^ {- 1} - (\mathbf {T} + \xi) ^ {- 1} \| _ {\mathcal {L} (\mathcal {H})} \leq \frac {\{2 \kappa + r _ {\mu} (n , \delta , b _ {0} , c _ {0}) \} r _ {\mu} (n , \delta , b _ {0} , c _ {0})}{\xi^ {2}}.
$$

Proof. Since $A ^ { - 1 } - B ^ { - 1 } = B ^ { - 1 } ( B - A ) A ^ { - 1 }$

$$
(\hat {\mathbf {T}} + \xi) ^ {- 1} - (\mathbf {T} + \xi) ^ {- 1} = (\mathbf {T} + \xi) ^ {- 1} (\mathbf {T} - \hat {\mathbf {T}}) (\hat {\mathbf {T}} + \xi) ^ {- 1}.
$$

Therefore

$$
\| (\hat {\mathbf {T}} + \xi) ^ {- 1} - (\mathbf {T} + \xi) ^ {- 1} \| _ {\mathcal {L} (\mathcal {H})} \leq \frac {1}{\xi^ {2}} \| \mathbf {T} - \hat {\mathbf {T}} \| _ {\mathcal {L} (\mathcal {H})} \leq \frac {\left\{2 \kappa + r _ {\mu} (n , \delta , b _ {0} , c _ {0}) \right\} r _ {\mu} (n , \delta , b _ {0} , c _ {0})}{\xi^ {2}}
$$

where the final inequality appeals to Proposition 13.

Proposition 15. Suppose Assumptions 4 and 5 hold. Then

$$
\| \mathbf {g} \| _ {\mathcal {H}} \leq \kappa C.
$$

Proof.

$$
\| \mathbf {g} \| _ {\mathcal {H}} = \left\| \frac {1}{m} \sum_ {i = 1} ^ {m} \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \dot {y} _ {i} \right\| _ {\mathcal {H}} \leq \frac {1}{m} \sum_ {i = 1} ^ {m} \left\| \mu (\dot {d} _ {i}, \dot {x} _ {i}, \dot {z} _ {i}) \right\| _ {\mathcal {H}} | \dot {y} _ {i} | \leq \kappa C.
$$

## G.6 Main result

Theorem 5 (Collecting results). Suppose Assumptions 2, 4, 5, and 7 hold. Suppose $\forall d \in \mathcal { D } , x \in \mathcal { X } , z \in \mathcal { Z } , \| \mu _ { \lambda } ^ { n } ( d , x , z ) - \mu ( d , x , z ) \| _ { \mathcal { H } } \leq r _ { \mu } ( n , \delta , b _ { 0 } , c _ { 0 } )$ with probability $1 - \delta$ Then $\forall \delta \in ( 0 , 1 )$ and $\forall \eta \in ( 0 , 1 )$ , the following holds with probability $1 - \eta - \delta \colon$

$$
\begin{array}{l} \| \hat {h} _ {\xi} ^ {m} - h _ {0} \| _ {\mathcal {H}} \\ \leq r _ {h} (n, \delta , c _ {0}; m, \eta , c) \\ := \frac {C r _ {\mu} (n , \delta , b _ {0} , c _ {0})}{\xi} + \frac {\kappa C}{\xi^ {2}} \left\{2 \kappa + r _ {\mu} (n, \delta , b _ {0}, c _ {0}) \right\} r _ {\mu} (n, \delta , b _ {0}, c _ {0}) \\ + C \log (4 / \eta) \left\{\frac {1}{m \xi} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} \right\} + \xi^ {\frac {c - 1}{2}} \sqrt {\zeta}. \end{array}
$$

Proof. I begin with a decomposition using Proposition 10.

$$
\hat {h} _ {\xi} ^ {m} - h _ {0} = \left[ (\hat {\mathbf {T}} + \boldsymbol {\xi}) ^ {- 1} \hat {\mathbf {g}} - (\hat {\mathbf {T}} + \boldsymbol {\xi}) ^ {- 1} \mathbf {g} \right] + \left[ (\hat {\mathbf {T}} + \boldsymbol {\xi}) ^ {- 1} \mathbf {g} - (\mathbf {T} + \boldsymbol {\xi}) ^ {- 1} \mathbf {g} \right] + \left[ (\mathbf {T} + \boldsymbol {\xi}) ^ {- 1} \mathbf {g} - h _ {0} \right].
$$

Consider the first term.

$$
\| (\hat {\mathbf {T}} + \xi) ^ {- 1} (\hat {\mathbf {g}} - \mathbf {g}) \| _ {\mathcal {H}} \leq \frac {1}{\xi} \| \hat {\mathbf {g}} - \mathbf {g} \| _ {\mathcal {H}} \leq \frac {1}{\xi} C r _ {\mu} (n, \delta , b _ {0}, c _ {0})
$$

by Proposition 13.

Consider the second term.

$$
\begin{array}{r l} \Big \| \Big [ (\hat {\mathbf {T}} + \xi) ^ {- 1} - (\mathbf {T} + \xi) ^ {- 1} \Big ] \mathbf {g} \Big \| _ {\mathcal {H}} & \leq \| (\hat {\mathbf {T}} + \xi) ^ {- 1} - (\mathbf {T} + \xi) ^ {- 1} \| _ {\mathcal {L} (\mathcal {H})} \| \mathbf {g} \| _ {\mathcal {H}} \\ & \leq \frac {1}{\xi^ {2}} \left\{2 \kappa + r _ {\mu} (n, \delta , b _ {0}, c _ {0}) \right\} r _ {\mu} (n, \delta , b _ {0}, c _ {0}) \cdot \kappa C. \end{array}
$$

with probability $1 - \delta$ by Propositions 14 and 15.

Consider the third term.

$$
\left\| h _ {\xi} ^ {m} - h _ {0} \right\| _ {\mathcal {H}} \leq \left\| h _ {\xi} ^ {m} - h _ {\xi} \right\| _ {\mathcal {H}} + \| h _ {\xi} - h _ {0} \| _ {\mathcal {H}} \leq C \log (4 / \eta) \left\{\frac {1}{m \xi} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} \right\} + \xi^ {\frac {c - 1}{2}} \sqrt {\zeta}
$$

with probability $1 - \eta$ , appealing to triangle inequality and Propositions 11 and 12.

Theorem 6 (Conditional mean embedding). Suppose Assumptions 4, 5, and 6 hold. Set $\lambda = n ^ { - \frac { 1 } { c _ { 0 } + 1 / b _ { 0 } } }$ . Then with probability $1 - \delta , \forall d \in \mathcal { D } , x \in \mathcal { X } , z \in \mathcal { Z }$

$$
\| \hat {\mu} (d, x, z) - \mu (d, x, z) \| _ {\mathcal {H}} \leq r _ {\mu} (n, \delta , b _ {0}, c _ {0})
$$

where $\kappa _ { R F } : = \kappa _ { d } \kappa _ { x } \kappa _ { z }$ and

$$
r _ {\mu} (n, \delta , b _ {0}, c _ {0}) := \kappa_ {d} \kappa_ {x} \cdot \kappa_ {R F} \cdot C \log (4 / \delta) n ^ {- \frac {1}{2} \frac {c _ {0} - 1}{c _ {0} + 1 / b _ {0}}}.
$$

Proof. Write

$$
\begin{array}{c} \hat {\mu} (d, x, z) - \mu (d, x, z) = [ \phi (d) \otimes \phi (x) \otimes \hat {\mu} _ {w} (d, x, z) ] - [ \phi (d) \otimes \phi (x) \otimes \mu_ {w} (d, x, z) ] \\ = \phi (d) \otimes \phi (x) \otimes [ \hat {\mu} _ {w} (d, x, z) - \mu_ {w} (d, x, z) ] \end{array}
$$

so that

$$
\| \hat {\mu} (d, x, z) - \mu (d, x, z) \| _ {\mathcal {H}} \leq \kappa_ {d} \kappa_ {x} \cdot \| \hat {\mu} _ {w} (d, x, z) - \mu_ {w} (d, x, z) \| _ {\mathcal {H} _ {\mathcal {W}}}.
$$

The bound on the final factor follows from [Singh et al., 2020, Proposition H.3], observing that

$$
E _ {0}: \mathcal {H} _ {\mathcal {W}} \to \mathcal {H} _ {R F}, \| \phi (d, x, z) \| _ {\mathcal {H} _ {R F}} \leq \kappa_ {R F}.
$$

Proof of Theorem 3. Summarize the bound in Theorem 5 as

$$
\begin{array}{r l} & {r _ {h} = O \left(\frac {r _ {\mu}}{\xi} + \frac {r _ {\mu}}{\xi^ {2}} + \frac {r _ {\mu} ^ {2}}{\xi^ {2}} + \frac {1}{m \xi} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} + \xi^ {\frac {c - 1}{2}}\right)} \\ & {\quad = O \left(\frac {r _ {\mu}}{\xi^ {2}} + \frac {1}{m \xi} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} + \xi^ {\frac {c - 1}{2}}\right)} \\ & {\quad = O \left(\frac {r _ {\mu}}{\xi^ {2}} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} + \xi^ {\frac {c - 1}{2}}\right)} \end{array}
$$

where the last statement holds when $m \xi \ge 1$ . Summarize the bound in Theorem 6 as

$$
r _ {\mu} = O \left(n ^ {- \frac {1}{2} \frac {c _ {0} - 1}{c _ {0} + 1 / b _ {0}}}\right) = O (m ^ {- \frac {a}{2}}), \quad n = m ^ {\frac {a (c _ {0} + 1 / b _ {0})}{(c _ {0} - 1)}}.
$$

Combining results,

$$
r _ {h} = O \left(\frac {1}{m ^ {\frac {a}{2}} \xi^ {2}} + \frac {1}{m ^ {1 / 2} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} + \xi^ {\frac {c - 1}{2}}\right), \quad \text {   such   that   } \quad \xi^ {2} \geq r _ {\mu}, \quad m \xi^ {\frac {1}{b} + 1} \geq 1.
$$

This choice of $( n , m )$ ratio generalizes the parametrization of [Singh et al., 2019, Theorem 4] to allow $b _ { 0 } > 1$

I have choice over ξ as a function of m to achieve the single stage rate of $m ^ { - { \frac { 1 } { 2 } } \frac { c - 1 } { c + 1 / b } }$ . I choose ξ to match the bias $\xi ^ { \frac { c - 1 } { 2 } }$ with the variance $\begin{array} { r } { \frac { 1 } { m ^ { \frac { a } { 2 } } \xi ^ { 2 } } + \frac { 1 } { m ^ { 1 / 2 } \xi ^ { \frac { 1 } { 2 b } + \frac { 1 } { 2 } } } } \end{array}$ . I set bias equal to each term in the variance.

1. $\begin{array} { r } { \xi ^ { \frac { c - 1 } { 2 } } = \frac { 1 } { m ^ { \frac { a } { 2 } } \xi ^ { 2 } } } \end{array}$ . Rearranging, $\xi = m ^ { - \frac { a } { c + 3 } }$ . The bias term becomes

$$
\xi^ {\frac {c - 1}{2}} = \left(m ^ {- \frac {a}{c + 3}}\right) ^ {\frac {c - 1}{2}}
$$

and the remaining term becomes

$$
\frac {1}{\sqrt {m} \xi^ {\frac {1}{2 b} + \frac {1}{2}}} = \frac {m ^ {\frac {1}{2} (1 + 1 / b) \frac {a}{c + 3}}}{\sqrt {m}} = m ^ {\frac {(1 + 1 / b) a - (c + 3)}{2 (c + 3)}}.
$$

Note that the former dominates the latter if and only if

$$
- \frac {a}{c + 3} \frac {c - 1}{2} \geq \frac {(1 + 1 / b) a - (c + 3)}{2 (c + 3)} \iff a \leq \frac {c + 3}{c + 1 / b}.
$$

2. $\xi ^ { \frac { c - 1 } { 2 } } = \frac { 1 } { \sqrt { m } \xi ^ { \frac { 1 } { 2 b } + \frac { 1 } { 2 } } }$ . Rearranging, $\xi = m ^ { - \frac { 1 } { c + 1 / b } }$ . The bias term becomes

$$
\xi^ {\frac {c - 1}{2}} = \left(m ^ {- \frac {1}{c + 1 / b}}\right) ^ {\frac {c - 1}{2}}
$$

and the remaining term becomes

$$
\frac {1}{m ^ {\frac {a}{2}} \xi^ {2}} = m ^ {- \frac {a}{2}} \left(m ^ {- \frac {1}{c + 1 / b}}\right) ^ {- 2} = m ^ {\frac {4 - a (c + 1 / b)}{2 (c + 1 / b)}}.
$$

Note that the former dominates the latter if and only if

$$
- \frac {1}{c + 1 / b} \frac {c - 1}{2} \geq \frac {4 - a (c + 1 / b)}{2 (c + 1 / b)} \iff a \geq \frac {c + 3}{c + 1 / b}.
$$

## H Treatment efect consistency proof

In this appendix, I (i) explicitly specialize the smoothness assumptions, (ii) provide rates for unconditional mean embeddings, (iv) provide rates for conditional mean embeddings, and (iv) prove uniform consistency of negative control treatment efects.

## H.1 Smoothness assumptions

## Assumption 10 (Smoothness for $\theta _ { 0 } ^ { A T T } )$ . Assume

1. The conditional expectation operator $E _ { 1 }$ is well specified as a Hilbert-Schmidt operator between RKHSs, i.e. $E _ { 1 } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } )$ , where

$$
E _ {1}: \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}} \to \mathcal {H} _ {\mathcal {D}}, \quad f (\cdot , \cdot) \mapsto \mathbb {E} [ f (X, W) | D = \cdot ].
$$

2. The conditional expectation operator is a particularly smooth element of $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } }$ ⊗ $\mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } )$ . Formally, define the covariance operator $T _ { 1 } : = \mathbb { E } [ \phi ( D ) \otimes \phi ( D ) ]$ for $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes$ $\mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } )$ . I assume there exists $G _ { 1 } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } )$ such that $E _ { 1 } = ( T _ { 1 } ) ^ { \frac { c _ { 1 } - 1 } { 2 } } \circ G _ { 1 }$ 2 $c _ { 1 } \in ( 1 , 2 ]$ , and $\| G _ { 1 } \| _ { \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { D } } ) } ^ { 2 } \le \zeta _ { 1 }$

## Assumption 11 (Smoothness for $\theta _ { 0 } ^ { C A T E } )$ . Assume

1. The conditional expectation operator $E _ { 2 }$ is well specified as a Hilbert-Schmidt operator between RKHSs, i.e. $E _ { 2 } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { X } \otimes \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { V } )$ , where

$$
E _ {2}: \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}} \to \mathcal {H} _ {\mathcal {V}}, f (\cdot , \cdot) \mapsto \mathbb {E} [ f (X, W) | V = \cdot ].
$$

2. The conditional expectation operator is a particularly smooth element of $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes$ $\mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { V } } )$ . Formally, define the covariance operator $T _ { 2 } : = \mathbb { E } [ \phi ( V ) \otimes \phi ( V ) ]$ for $\mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes$ $\mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { V } } )$ . I assume there exists $G _ { 2 } \in \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { V } } )$ such that $E _ { 2 } = ( T _ { 2 } ) ^ { \frac { c _ { 2 } - 1 } { 2 } } \circ G _ { 2 }$ 2 $c _ { 2 } \in ( 1 , 2 ]$ , and $\| G _ { 2 } \| _ { \mathcal { L } _ { 2 } ( \mathcal { H } _ { \mathcal { X } } \otimes \mathcal { H } _ { \mathcal { W } } , \mathcal { H } _ { \mathcal { V } } ) } ^ { 2 } \le \zeta _ { 2 }$

Proposition 16. The following assumptions are equivalent

1. Assumption 6 with $\mathcal { A } _ { 1 } = \mathcal { X } \times \mathcal { W }$ and $\boldsymbol { B } _ { 1 } = \boldsymbol { D }$ is equivalent to Assumption 10

2. Assumption 6 with $\mathcal { A } _ { 2 } = \mathcal { X } \times \mathcal { W }$ and $\begin{array} { r } { B _ { 2 } = \nu } \end{array}$ is equivalent to Assumption 11

Proof. The result is immediate from [Caponnetto and De Vito, 2007, Remark 2]. The expanded expressions are more convenient for analysis. □

## H.2 Unconditional mean embedding

Theorem 7 (Unconditional mean embedding). Suppose Assumptions 4 and 5 hold. Then with probability $1 - \delta$

$$
\| \hat {\mu} - \mu \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \leq r _ {\mu} (n, \delta) := \frac {4 \kappa_ {x} \kappa_ {w} \ln (2 / \delta)}{\sqrt {n}}.
$$

Likewise, with probability $1 - \delta$

$$
\| \hat {\nu} - \nu \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \leq r _ {\nu} (\tilde {n}, \delta) := \frac {4 \kappa_ {x} \kappa_ {w} \ln (2 / \delta)}{\sqrt {\tilde {n}}}.
$$

Proof. The first result follows from Lemma 4 with $\xi _ { i } = \phi ( x _ { i } ) \otimes \phi ( w _ { i } )$ , since

$$
\begin{array}{r l} \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} [ \phi (x _ {i}) \otimes \phi (w _ {i}) ] - \mathbb {E} [ \phi (X) \otimes \phi (W) ] \right\| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} & \leq \frac {2 \kappa_ {x} \kappa_ {w} \ln (2 / \delta)}{n} + \sqrt {\frac {2 \kappa_ {x} ^ {2} \kappa_ {w} ^ {2} \ln (2 / \delta)}{n}} \\ & \leq \frac {4 \kappa_ {x} \kappa_ {w} \ln (2 / \delta)}{\sqrt {n}}. \end{array}
$$

The argument for ν is identical, using $\xi _ { i } = \phi ( \tilde { x } _ { i } ) \otimes \phi ( \tilde { w } _ { i } )$

## H.3 Conditional mean embedding

Theorem 8 (Conditional mean embedding rate). Suppose Assumptions 4 and 5 hold. Set $( \lambda _ { 1 } , \lambda _ { 2 } ) = ( n ^ { - \frac { 1 } { c _ { 1 } + 1 / b _ { 1 } } } , n ^ { - \frac { 1 } { c _ { 2 } + 1 / b _ { 2 } } } )$

1. If in addition Assumption 10 holds then with probability $1 - \delta .$ for n suficiently large, $\forall d \in { \mathcal { D } }$

$$
\| \hat {\mu} (d) - \mu (d) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \leq r _ {\mu} ^ {A T T} (n, \delta , b _ {1}, c _ {1})
$$

where

$$
r _ {\mu} ^ {A T T} (n, \delta , b _ {1}, c _ {1}) := \kappa_ {d} \cdot C \log (4 / \delta) n ^ {- \frac {1}{2} \frac {c _ {1} - 1}{c _ {1} + 1 / b _ {1}}}.
$$

2. If in addition Assumption 11 holds then with probability $1 - \delta .$ for n suficiently large, $\forall v \in \mathcal { V }$

$$
\| \hat {\mu} (v) - \mu (v) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \leq r _ {\mu} ^ {C A T E} (n, \delta , b _ {2}, c _ {2})
$$

where

$$
r _ {\mu} ^ {C A T E} (n, \delta , b _ {2}, c _ {2}) := \kappa_ {v} \cdot C \log (4 / \delta) n ^ {- \frac {1}{2} \frac {c _ {2} - 1}{c _ {2} + 1 / b _ {2}}}.
$$

Proof. The proof immediately follows from [Singh et al., 2020, Proposition H.3], observing that

$$
E _ {1}: \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}} \to \mathcal {H} _ {\mathcal {D}}, \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \leq \kappa_ {d}
$$

and

$$
E _ {2}: \mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}} \to \mathcal {H} _ {\mathcal {V}}, \| \phi (v) \| _ {\mathcal {H} _ {\mathcal {V}}} \leq \kappa_ {v}.
$$


## H.4 Main result

In summary, the rates are

$$
\| \hat {h} - h _ {0} \| _ {\mathcal {H}} = O _ {p} \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}}\right)
$$

$$
\| \hat {\mu} - \mu \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} = O _ {p} \left(n ^ {- \frac {1}{2}}\right)
$$

$$
\left\| \hat {\nu} - \nu \right\| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} = O _ {p} \left(\tilde {n} ^ {- \frac {1}{2}}\right)
$$

$$
\| \hat {\mu} (d) - \mu (d) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} = O _ {p} \left(n ^ {- \frac {1}{2} \frac {c _ {1} - 1}{c _ {1} + 1 / b _ {1}}}\right)
$$

$$
\| \hat {\mu} (v) - \mu (v) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} = O _ {p} \left(n ^ {- \frac {1}{2} \frac {c _ {2} - 1}{c _ {2} + 1 / b _ {2}}}\right).
$$

Proof of Theorem 4. I generalize the argument in [Singh et al., 2020, Theorem 3.3]. I write out the finite sample bounds. Consider the decomposition

$$
\begin{array}{r l} & {\hat {\theta} ^ {A T E} (d) - \theta_ {0} ^ {A T E} (d)} \\ & {= \langle \hat {h}, \phi (d) \otimes \hat {\mu} \rangle_ {\mathcal {H}} - \langle h _ {0}, \phi (d) \otimes \mu \rangle_ {\mathcal {H}}} \\ & {= \langle \hat {h}, \phi (d) \otimes [ \hat {\mu} - \mu ] \rangle_ {\mathcal {H}} + \langle [ \hat {h} - h _ {0} ], \phi (d) \otimes \mu \rangle_ {\mathcal {H}}} \\ & {= \langle [ \hat {h} - h _ {0} ], \phi (d) \otimes [ \hat {\mu} - \mu ] \rangle_ {\mathcal {H}} + \langle h _ {0}, \phi (d) \otimes [ \hat {\mu} - \mu ] \rangle_ {\mathcal {H}} + \langle [ \hat {h} - h _ {0} ], \phi (d) \otimes \mu \rangle_ {\mathcal {H}}.} \end{array}
$$

Therefore with probability $1 - 2 \delta - \eta$

$$
\begin{array}{l} | \hat {\theta} ^ {A T E} (d) - \theta_ {0} ^ {A T E} (d) | \\ \leq \| \hat {h} - h _ {0} \| _ {\mathcal {H}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \hat {\mu} - \mu \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} + \| h _ {0} \| _ {\mathcal {H}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \hat {\mu} - \mu \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \\ \quad + \| \hat {h} - h _ {0} \| _ {\mathcal {H}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \mu \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}} \\ \leq \kappa_ {d} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c) \cdot r _ {\mu} (n, \delta) + \kappa_ {d} \cdot \| h _ {0} \| _ {\mathcal {H}} \cdot r _ {\mu} (n, \delta) \\ \quad + \kappa_ {d} \kappa_ {x} \kappa_ {w} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c) \\ = O \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}} + n ^ {- \frac {1}{2}}\right). \end{array}
$$

By the same argument as for $\theta _ { 0 } ^ { A T E }$ , with probability $1 - 2 \delta - \eta$

$$
\begin{array}{l} | \hat {\theta} ^ {D S} (d, \tilde {\mathbb {P}}) - \theta_ {0} ^ {D S} (d, \tilde {\mathbb {P}}) | \\ \leq \kappa_ {d} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c) \cdot r _ {\nu} (\tilde {n}, \delta) + \kappa_ {d} \cdot \| h _ {0} \| _ {\mathcal {H}} \cdot r _ {\nu} (\tilde {n}, \delta) \\ \quad + \kappa_ {d} \kappa_ {x} \kappa_ {w} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c) \\ = O \left(m ^ {- \frac {1}{2} \frac {c - 1}{c + 1 / b}} + \tilde {n} ^ {- \frac {1}{2}}\right). \end{array}
$$

Next, I turn to the nonparametric treatment efects with conditional mean embeddings. Consider the decomposition

$$
\begin{array}{l} \hat {\theta} ^ {A T T} (d, d ^ {\prime}) - \theta_ {0} ^ {A T T} (d, d ^ {\prime}) \\ = \langle \hat {h}, \phi (d ^ {\prime}) \otimes \hat {\mu} (d) \rangle_ {\mathcal {H}} - \langle h _ {0}, \phi (d ^ {\prime}) \otimes \mu (d) \rangle_ {\mathcal {H}} \\ = \langle \hat {h}, \phi (d ^ {\prime}) \otimes [ \hat {\mu} (d) - \mu (d) ] \rangle_ {\mathcal {H}} + \langle [ \hat {h} - h _ {0} ], \phi (d ^ {\prime}) \otimes \mu (d) \rangle_ {\mathcal {H}} \\ = \langle [ \hat {h} - h _ {0} ], \phi (d ^ {\prime}) \otimes [ \hat {\mu} (d) - \mu (d) ] \rangle_ {\mathcal {H}} + \langle h _ {0}, \phi (d ^ {\prime}) \otimes [ \hat {\mu} (d) - \mu (d) ] \rangle_ {\mathcal {H}} \\ + \langle [ \hat {h} - h _ {0} ], \phi (d ^ {\prime}) \otimes \mu (d) \rangle_ {\mathcal {H}}. \end{array}
$$

Therefore with probability $1 - 2 \delta - \eta$

$$
\begin{array}{r l} & {| \hat {\theta} ^ {A T T} (d, d ^ {\prime}) - \theta_ {0} ^ {A T T} (d, d ^ {\prime}) |} \\ & {\leq \| \hat {h} - h _ {0} \| _ {\mathcal {H}} \| \phi (d ^ {\prime}) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \hat {\mu} (d) - \mu (d) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}}} \\ & {\quad + \| h _ {0} \| _ {\mathcal {H}} \| \phi (d ^ {\prime}) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \hat {\mu} (d) - \mu (d) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}}} \\ & {\quad + \| \hat {h} - h _ {0} \| _ {\mathcal {H}} \| \phi (d ^ {\prime}) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \mu (d) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}}} \\ & {\leq \kappa_ {d} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c) \cdot r _ {\mu} ^ {A T T} (n, \delta , b _ {1}, c _ {1}) + \kappa_ {d} \cdot \| h _ {0} \| _ {\mathcal {H}} \cdot r _ {\mu} ^ {A T T} (n, \delta , b _ {1}, c _ {1})} \\ & {\quad + \kappa_ {d} \kappa_ {x} \kappa_ {w} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c)} \\ & {= O \left(m ^ {- \frac 12 \frac {c - 1}{c + 1 / b}} + n ^ {- \frac 12 \frac {c _ {1} - 1}{c _ {1} + 1 / b _ {1}}}\right).} \end{array}
$$

Similarly, consider the decomposition

$$
\begin{array}{l} \hat {\theta} ^ {C A T E} (d, v) - \theta_ {0} ^ {C A T E} (d, v) \\ = \langle \hat {h}, \phi (d) \otimes \phi (v) \otimes \hat {\mu} (v) \rangle_ {\mathcal {H}} - \langle h _ {0}, \phi (d) \otimes \phi (v) \otimes \mu (v) \rangle_ {\mathcal {H}} \\ = \langle \hat {h}, \phi (d) \otimes \phi (v) \otimes [ \hat {\mu} (v) - \mu (v) ] \rangle_ {\mathcal {H}} + \langle [ \hat {h} - h _ {0} ], \phi (d) \otimes \phi (v) \otimes \mu (v) \rangle_ {\mathcal {H}} \\ = \langle [ \hat {h} - h _ {0} ], \phi (d) \otimes \phi (v) \otimes [ \hat {\mu} (v) - \mu (v) ] \rangle_ {\mathcal {H}} \\ + \langle h _ {0}, \phi (d) \otimes \phi (v) \otimes [ \hat {\mu} (v) - \mu (v) ] \rangle_ {\mathcal {H}} \\ + \langle [ \hat {h} - h _ {0} ], \phi (d) \otimes \phi (v) \otimes \mu (v) \rangle_ {\mathcal {H}}. \end{array}
$$

Therefore with probability $1 - 2 \delta - \eta$

$$
\begin{array}{r l} & {| \hat {\theta} ^ {C A T E} (d, v) - \theta_ {0} ^ {C A T E} (d, v) |} \\ & {\leq \| \hat {h} - h _ {0} \| _ {\mathcal {H}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \phi (v) \| _ {\mathcal {H} _ {\mathcal {V}}} \| \hat {\mu} (v) - \mu (v) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}}} \\ & {\quad + \| h _ {0} \| _ {\mathcal {H}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \phi (v) \| _ {\mathcal {H} _ {\mathcal {V}}} \| \hat {\mu} (v) - \mu (v) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {w}}}} \\ & {\quad + \| \hat {h} - h _ {0} \| _ {\mathcal {H}} \| \phi (d) \| _ {\mathcal {H} _ {\mathcal {D}}} \| \phi (v) \| _ {\mathcal {H} _ {\mathcal {V}}} \| \mu (v) \| _ {\mathcal {H} _ {\mathcal {X}} \otimes \mathcal {H} _ {\mathcal {W}}}} \\ & {\leq \kappa_ {d} \kappa_ {v} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c) \cdot r _ {\mu} ^ {C A T E} (n, \delta , b _ {2}, c _ {2}) + \kappa_ {d} \kappa_ {v} \cdot \| h _ {0} \| _ {\mathcal {H}} \cdot r _ {\mu} ^ {C A T E} (n, \delta , b _ {2}, c _ {2})} \\ & {\quad + \kappa_ {d} \kappa_ {v} \kappa_ {x} \kappa_ {w} \cdot r _ {h} (n, \delta , b _ {0}, c _ {0}; m, \eta , b, c)} \\ & {= O \left(m ^ {- \frac 12 \frac {c - 1}{c + 1 / b}} + n ^ {- \frac 12 \frac {c _ {2} - 1}{c _ {2} + 1 / b _ {2}}}\right).} \end{array}
$$

## I Simulation details

## I.1 General simulation design

A single observation is generated as follows. Recall that $( Y , D ) \in \mathbb { R } , \ X \in \mathbb { R } ^ { d i m ( X ) } , \ Z \in$ $\mathbb { R } ^ { d i m ( Z ) }$ , and $W \in \mathbb { R } ^ { d i m ( W ) }$

1. Draw unobserved noise as

$$
\{\epsilon_ {i} \} _ {i \in [ 3 ]} \stackrel {{i. i. d}} {{\sim}} \mathcal {N} (0, 1), \quad \nu_ {z} \sim \mathcal {U} [ - 1, 1 ] ^ {d i m (Z)}, \quad \nu_ {w} \sim \mathcal {U} [ - 1, 1 ] ^ {d i m (W)}.
$$

2. Set unobserved confounders as $u _ { z } = \epsilon _ { 1 } + \epsilon _ { 3 }$ and $u _ { w } = \epsilon _ { 2 } + \epsilon _ { 3 }$

3. Set the negative controls as

$$
Z = \nu_ {z} + 0. 2 5 \cdot u _ {z} \cdot 1 _ {d i m (Z)}, W = \nu_ {w} + 0. 2 5 \cdot u _ {w} \cdot 1 _ {d i m (W)},
$$

where $1 _ { p } \in \mathbb { R } ^ { p }$ is the vector of ones of length $p .$

4. Draw covariates $X \sim { \mathcal { N } } ( 0 , \Sigma )$ where the covariance matrix $\Sigma \in \mathbb { R } ^ { d i m ( X ) \times d i m ( X ) }$ is such that $\Sigma _ { i i } = 1$ and $\begin{array} { r } { \Sigma _ { i j } = \frac { 1 } { 2 } \cdot 1 \{ | i - j | = 1 \} } \end{array}$ for $i \neq j$

5. Then set treatment as

$$
D = \Lambda (3 X ^ {\top} \beta_ {x} + 3 Z ^ {\top} \beta_ {z}) + 0. 2 5 \cdot u _ {w}.
$$

$\beta _ { x } \in \mathbb { R } ^ { d i m ( X ) }$ and $\beta _ { z } \in \mathbb { R } ^ { d i m ( Z ) }$ are quadratically decaying coeficients, $\mathrm { e . g . } ~ [ \beta _ { x } ] _ { j } = j ^ { - 2 }$ Λ is the truncated logistic link function $\begin{array} { r } { \Lambda ( t ) = ( 0 . 9 - 0 . 1 ) \frac { \exp ( t ) } { 1 + \exp ( t ) } + 0 . 1 } \end{array}$

6. Finally set the outcome as

$$
Y = \theta_ {0} ^ {A T E} (D) + 1. 2 (X ^ {\top} \beta_ {x} + W ^ {\top} \beta_ {w}) + D X _ {1} + 0. 2 5 \cdot u _ {z},
$$

where $\beta _ { w } \in \mathbb { R } ^ { d i m ( W ) }$ is a quadratically decaying coeficient, i.e. $[ \beta _ { w } ] _ { j } = j ^ { - 2 }$

For the quadratic design, $\theta _ { 0 } ^ { A T E } ( d ) = d ^ { 2 } + 1 . 2 d$ as in [Colangelo and Lee, 2020]. For the sigmoid design, $\theta _ { 0 } ^ { A T E } ( d ) = \ln ( | 1 6 d - 8 | + 1 ) \cdot s i g n ( d - 0 . 5 ) + 1 . 2 d$ similar to [Singh et al., 2019]. Finally, for the peaked design, $\theta _ { 0 } ^ { A T E } ( d ) = 2 \{ d ^ { 4 } / 6 0 0 + \exp ( - 4 d ^ { 2 } ) + d / 1 0 - 2 \} + 1 . 2 d$ similar to [Singh et al., 2019].

I implement the estimator $\hat { \theta } ^ { A T E } ( d ) \ ( \mathtt { N } . { \mathsf { C } } . )$ described in Section 4, with the tuning procedure described in Appendix F. Specifically, I use ridge penalties determined by leave-one-out cross validation, and product Gaussian kernel with lengthscales set by the median heuristic. I implement the continuous treatment efect estimator of [Singh et al., 2020] (T.E.) using the same principles. The latter tuning procedure is simpler; whereas the new estimator involves reweighting a confounding bridge (with two ridge penalty hyperparameters), the previous estimator involves reweighting a regression (with one ridge penalty hyperparameter).

## I.2 Robustness to tuning

As explained in Appendix F, the tuning procedure for the ridge penalties involves leave-oneout cross validation. I now confirm that the proposed estimator’s performance is robust to improper tuning. Figures $\mathrm { 7 ( a ) }$ and 7(b) summarize results.

(a) Forcing λ

(b) Forcing ξ

(c) Unfounded design  
Figure 7: Robustness studies

I conduct two robustness studies, each corresponding to a ridge penalty hyperparameter. In the first robustness study (Figure 7(a)), I force λ to take a particular value in a grid, then tune $\xi$ by leave-one-out cross validation: $\xi = \xi ^ { * } ( \lambda )$ . In the second robustness study (Figure 7(b)), I tune $\lambda ~ = ~ \lambda ^ { * }$ by leave-one-out cross validation, then force $\xi$ to take a particular value in a grid. Across all levels of improper tuning of N.C., it continues to outperform the properly tuned T.E. estimator.

## I.3 No unobserved confounding

Next, I study a setting where there is no unobserved confounding. I modify the data generating process as follows.

1. Draw unobserved noise as

$$
\{\epsilon_ {i} \} _ {i \in [ 4 ]} \stackrel {{i. i. d}} {{\sim}} \mathcal {N} (0, 1), \quad \nu_ {z} \sim \mathcal {U} [ - 1, 1 ] ^ {d i m (Z)}, \quad \nu_ {w} \sim \mathcal {U} [ - 1, 1 ] ^ {d i m (W)}.
$$

2. Set unobserved confounders as $u _ { z } = \epsilon _ { 1 } + \epsilon _ { 2 }$ and $u _ { w } = \epsilon _ { 3 } + \epsilon _ { 4 }$

3. Set the negative controls as

$$
Z = \nu_ {z} + 0. 2 5 \cdot u _ {z} \cdot 1 _ {d i m (Z)}, W = \nu_ {w} + 0. 2 5 \cdot u _ {w} \cdot 1 _ {d i m (W)}.
$$

4. Draw covariates $X \sim { \mathcal { N } } ( 0 , \Sigma )$

5. Then set treatment as

$$
D = \Lambda (3 X ^ {\top} \beta_ {x}) + 0. 2 5 \cdot u _ {w}.
$$

6. Finally set the outcome as

$$
Y = \theta_ {0} ^ {A T E} (D) + 1. 2 (X ^ {\top} \beta_ {x}) + D X _ {1} + 0. 2 5 \cdot u _ {z}.
$$

Figure 7(c) visualizes results. As expected, the previously existing method, which assumes no unobserved confounding, outperforms the proposed method. The proposed method solves an ill posed inverse problem. One pays a cost in terms of statistical eficiency when the ill posed inverse problem is unnecessary. From a practical perspective, an analyst should only use negative controls when the analyst firmly believes that unobserved confounding is present and that the negatives controls satisfy Assumptions 1 and 2.

## I.4 Discrete treatment

So far, I have focused on the case with continuous treatment. I now study empirical performance when treatment is discrete. I modify the data generating process as follows.

1. Draw unobserved noise as

$$
\{\epsilon_ {i} \} _ {i \in [ 3 ]} \stackrel {i. i. d} {\sim} \mathcal {N} (0, 1), \quad \nu_ {z} \sim \mathcal {U} [ - 1, 1 ] ^ {d i m (Z)}, \quad \nu_ {w} \sim \mathcal {U} [ - 1, 1 ] ^ {d i m (W)}.
$$

2. Set unobserved confounders as $u _ { z } = \epsilon _ { 1 } + \epsilon _ { 3 }$ and $u _ { w } = \epsilon _ { 2 } + \epsilon _ { 3 }$

3. Set the negative controls as

$$
Z = \nu_ {z} + 0. 5 \cdot u _ {z} \cdot 1 _ {d i m (Z)}, W = \nu_ {w} + 0. 5 \cdot u _ {w} \cdot 1 _ {d i m (W)}.
$$

4. Draw covariates $X \sim { \mathcal { N } } ( 0 , \Sigma )$

5. Then set treatment as

$$
D \sim B e r n o u l l i (\Lambda (X ^ {\top} \beta_ {x} + Z ^ {\top} \beta_ {z} + u _ {w})).
$$

6. Finally set the outcome as

$$
Y = 2. 2 + 1. 2 (X ^ {\top} \beta_ {x} + W ^ {\top} \beta_ {w}) + D X _ {1} + 0. 5 \cdot u _ {z}.
$$

By construction, $\theta _ { 0 } ^ { A T E } ( 1 ) - \theta _ { 0 } ^ { A T E } ( 0 ) = 2 . 2 - 0 = 2 . 2$ . Table 1 summarizes results. The proposed method (N.C.) outperforms the previous approach (T.E.) that ignores unobserved confounding, when the sample size is suficiently large. Importantly, the sample size n must be at least 1000 for the estimator to detect and correct for the unobserved confounding. Substantial bias remains, suggesting that the debiased semiparametric estimators subsequently proposed by [Kallus et al., 2021, Ghassami et al., 2021] may be more appropriate when treatment is discrete.

## J Application details

## J.1 Dose response of cigarette smoking

As described in the main text, I estimate the dose response curves for the subpopulations of white, black, and Hispanic mothers who smoke. Formally, I estimate $\theta _ { 0 } ^ { C A T E } ( d , v )$ where $D \in$ R is the number of cigarettes smoked per day, and $V \in \mathbb { R } ^ { 2 }$ concatenates mother’s race $V _ { 1 }$ and mother’s smoking status $V _ { 2 }$ . Observe that race is, for our purposes, a discrete variable with three values while smoking status is a binary variable. Consider the subpopulation of white mothers who smoke, i.e. $v = ( \mathrm { w h i t e } , 1 )$ . I implement the product of indicator kernels

<table><tr><td>Sample size</td><td>Mean</td><td>S.D.</td><td>M.S.E.</td></tr><tr><td>100</td><td>2.61</td><td>0.23</td><td>0.05</td></tr><tr><td>500</td><td>2.59</td><td>0.11</td><td>0.01</td></tr><tr><td>1000</td><td>2.55</td><td>0.06</td><td>0.00</td></tr><tr><td>5000</td><td>2.42</td><td>0.03</td><td>0.00</td></tr></table>

(a) Treatment efect (T.E.)

<table><tr><td>Sample size</td><td>Mean</td><td>S.D.</td><td>M.S.E.</td></tr><tr><td>100</td><td>3.07</td><td>0.27</td><td>0.07</td></tr><tr><td>500</td><td>2.62</td><td>0.11</td><td>0.01</td></tr><tr><td>1000</td><td>2.42</td><td>0.09</td><td>0.01</td></tr><tr><td>5000</td><td>1.99</td><td>0.05</td><td>0.00</td></tr></table>

(b) Negative control (N.C.)  
Table 1: Discrete design

$$
k _ {\mathcal {V}} (v, v ^ {\prime}) = k _ {\mathcal {V} _ {1}} (v _ {1}, v _ {1} ^ {\prime}) k _ {\mathcal {V} _ {2}} (v _ {2}, v _ {2} ^ {\prime}) = \mathbb {1} \{v _ {1} = v _ {1} ^ {\prime} \} \mathbb {1} \{v _ {2} = v _ {2} ^ {\prime} \}.
$$

Therefore the estimator of the confounding bridge $h _ { 0 } ( d , v , x , w )$ at the value $v = ( \mathrm { w h i t e } , 1 )$ simply zeroes out observations with $V \neq { \mathrm { ( w h i t e , 1 ) } }$ . Since the covariate of interest is discrete rather than continuous, a natural choice of the estimator $\hat { \mu } ( v )$ that encodes $\mathbb { P } ( x , w | v )$ simply takes the average for the corresponding subpopulation, e.g.

$$
\hat {\mu} (\text { white }, 1) = \frac {1}{| \{i : V _ {i} = (\text { white } , 1) \} |} \sum_ {i: V _ {i} = (\text { white }, 1)} \phi (x _ {i}) \otimes \phi (w _ {i}).
$$

Observe that this estimator also zeroes out observations with $V \neq { \mathrm { ( w h i t e , 1 ) } }$

In summary, when V is discrete, the described estimator of $\theta _ { 0 } ^ { C A T E } ( d , v )$ is equivalent to the following procedure: (i) subset to the observations such that $V = v$ , then (ii) implement $\hat { \theta } ^ { A T E } ( d )$ . I implement the estimator $\hat { \theta } ^ { A T E } ( d ) \ ( \mathtt { N } . { \mathsf { C } } . )$ described in Section 4, with the tuning procedure described in Appendix F, on the appropriate subset of observations. I implement the continuous treatment efect estimator of [Singh et al., 2020] (T.E.) using the same principles.

To speed up computation, I incorporate the kernlab package in R. Due to the large sample size for nonhispanic white women $( n = 7 3 , 8 3 4 )$ , I split the observations according to the year (1989, 1990, 1991) then take the average.

## J.2 Variable classification

Table 2 summarizes the variable classification. The reason why this choice of negative controls (Z, W) is appropriate when U is income, but inappropriate when U is stress, is the relevance condition detailed in Proposition 1. Based on clinical expertise and previously published results such as [Hobel et al., 2008], we posit that parental education and family size are relevant for income in the sense that variation in income U can be recovered from variation in parental education Z and family size W. The same cannot be said for stress; it is less justified by the existing literature that variation in stress U can be recovered from variation in parental education Z and family size W. Figure 8 visualizes a representative DAG.

Table 2: Variable classification

<table><tr><td>Symbol</td><td>Definition</td><td>Empirical application</td></tr><tr><td>Y</td><td>outcome</td><td>infant birth weight (grams)</td></tr><tr><td>D</td><td>treatment</td><td># cigarettes smoked per day during pregnancy</td></tr><tr><td>U</td><td>unobserved confounding</td><td>income stress</td></tr><tr><td>Z</td><td>n.c. treatment</td><td>mother&#x27;s educational attainment (years) father&#x27;s educational attainment (years)</td></tr><tr><td>W</td><td>n.c. outcome</td><td>infant birth order infant sex Rh sensitization</td></tr><tr><td>X</td><td>covariates</td><td>mother&#x27;s demographics: age, marriage status, foreign born father&#x27;s demographics: age, race alcohol consumption during pregnancy # prenatal visits during pregnancy trimester of first prenatal care visit hypertension: chronic, gestational other medical: diabetes, herpes, eclampsia, weight gain county year</td></tr><tr><td>V</td><td>covariate of interest</td><td>mother&#x27;s racemother&#x27;s smoking status</td></tr></table>

Figure 8: Smoking DAG with stress
