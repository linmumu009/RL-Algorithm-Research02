# The Generic Holdout: Preventing False-Discoveries in Adaptive Data Science

Preetum Nakkiran<sup>∗</sup>

Jarosław Błasiok <sup>†</sup>

## Abstract

The traditional framework of science is “non-adaptive”, in the sense that the scientist first fixes a hypothesis, and then collects data to test it. However, modern science is often “adaptive”, in that first large amounts of data are collected, then the scientist explores this data to propose hypotheses. Such adaptive data analysis has posed a challenge to science due to its ability to generate false hypotheses on moderately large data sets. In general, with non-adaptive data analyses (where the queries to the data are generated without being influenced by answers to previous queries) a data set containing n samples may support exponentially many queries in n. This number reduces to linearly many under naive adaptive data analysis, and even sophisticated remedies such as the Reusable Holdout (Dwork et. al 2015) only allow quadratically many queries in n.

In this work, we propose a new framework for adaptive science which exponentially improves on this number of queries under a restricted yet scientifically relevant setting in data analysis, where the goal of the scientist is to find a single (or a few) true hypotheses about the universe based on the samples. Such a setting may describe the search for predictive factors of some disease based on medical data, where the analyst may wish to try a number of predictive models until a satisfactory one is found.

Our solution, which we refer to as the Generic Holdout methodology, involves two extremely simple ingredients: (1) a partitioning of the data into a exploration set and a holdout set (a methodology that is already widely practiced) and (2) a limited exposure strategy for the holdout set (something that is widely violated, but easy to fix). An analyst is free to use the exploration set arbitrarily, but when testing hypotheses against the holdout set, the analyst only learns the answer to the question: “Is the given hypothesis true (empirically) on the holdout set?” It is common for holdout sets to provide much more information, such as “how well” the hypothesis fit the holdout set, and this is where a common fallacy lies. The resulting scheme is immediate to analyze, and reverts to the setting where exponentially many hypotheses can be tested against the data. Despite this simplicity we do not believe our method is obvious, as evidenced by the many violations in practice.

Our proposal is best seen as an alternative to pre-registration, where journals require scientists to commit to their hypotheses and analysis procedure before collecting any data. Pre-registration preserves statistical validity, but at the cost of adaptivity: researchers are not allowed to explore the structure of data to generate hypotheses. In this setting, the Generic Holdout allows researchers to get the benefits of adaptive data analysis, without the problems of adaptivity.

## 1 Introduction

In science, it is natural to first collect data, and then form informed hypotheses based on it. This is arguably how much of science was historically done — after all, one is unlikely to come up with a correct theory of physics without first observing the world. In order for the result to be statistically valid, a scientist must then collect independent data, after fixing their hypothesis. However, this is not done in practice: in modern experimental science, it is common to first collect data, and then explore it to generate plausible hypotheses. This is what we consider adaptive science: wherein the scientist generates hypotheses after somehow interacting with the data set. Doing this naively could lead to being convinced of false hypotheses, since a scientist may “overfit” — that is, derive a false hypothesis that appears to be true on the data. This could occur if the scientist implicitly or explicitly tests many hypotheses before finding one that happens to fit the experimental data. For example, if the first hypothesis tested proves false on the data set, a scientist may want to use information about this first hypothesis test to revisit the data set, and form a new hypothesis, and so on.

There is growing recognition in the sciences that this “adaptive” method of doing science is not statistically sound, and leads to invalid claims. There is a recognized “reproducibility crisis” in Psychology, for example, after a collaboration failed to replicate 62 out of 97 studies with positive findings in three prominent psychology journals [8]. The problem of adaptivity (also known as “p-hacking” or “researcher degrees of freedom”) is recognized as a key contributor to this crisis, since it is often not correctly accounted for in standard statistical analyses [17, 22]. Other experimental areas such as neuroscience [2], economics [5], and social sciences [6] are also aware of this issue, after conducting reproducibility studies following the example of Psychology. In general, any methodology in which a researcher decides which hypothesis to test after somehow interacting with the data set is susceptible to this problem of adaptivity. Note that this includes scenarios like testing a second hypothesis after the initial hypothesis test failed, or doing a “data-dependent analysis” in which the hypothesis formed depends on some structure of the data set itself. One notable example of such data-dependent analysis is using Principal Component Analysis on the data set to find a correlation structure, and using this in turn to define the hypothesis, that is tested against the same data set. The review article [17] provides many further examples of this problem of adaptivity arising in science. We provide an explicit, formal example of this problem in Section 2.1.

The preceding discussion motivates our scientific goal: we would like to have a statistically-valid scientific methodology that allows researchers to explore their data before generating hypotheses. We are considering the setting where the scientist is trying to derive a small number of true hypotheses, and would like to adaptively check if the proposed hypotheses are true. We want to guarantee that in this process, false hypotheses which are proposed are unlikely to be validated as true — that is, we would like to prevent false discoveries. Before describing our proposal, we first briefly discuss three proposed solutions to the problem of adaptivity in the sciences.

1. Preregistration. There has been a recent push in the scientific community towards preregistration: requiring scientists to commit to their scientific methods and hypotheses before conducting a study. Several prominent journals now encourage scientists to preregister their research, in order to preserve statistical rigor (for example: Open Science of the Royal Society, and Psychological Science of the Association for Psychological Science). An open letter published by more than 80 signatories calls for preregistration in the sciences [7]. Preregistration does preserve statistical validity, but at the cost of adaptivity: the researcher is not allowed to explore the structure of data to generate hypotheses, and thus preregistration has been critiqued for slowing down the overall scientific process.

2. Naive Holdout. Keeping a holdout set is another way of addressing the adaptivity problem. The idea is, the scientist holds out part of the data set (without looking at it to form hypotheses), and is free to explore the remainder of the data set. Then, after exploring and proposing a hypothesis, the scientist checks the hypothesis against the holdout set. This is statistically valid, since the holdout data is independent of the hypothesis being tested, but has the disadvantage that the holdout can only be used once: if the scientist now wants to test another hypothesis, he/she must collect additional independent data to use as a holdout. This is sample-ineficient and impractical in settings where collecting data is expensive. In particular, the naive holdout can only handle linearly many hypothesis tests in the total size of the holdout sets. Note that it is not valid to naively re-use the same holdout set after seeing the results of the first hypothesis test (say, its p-value) — this can quickly lead to overfitting on the holdout set, in a way made precise in an extended

example in Section 2.1.

3. Reusable Holdout. The Reusable Holdout [11, 10], a recent development in the field of Adaptive Data Analysis, manages to improve on the naive holdout, and handles up to quadratically many hypothesis tests in the holdout size<sup>1</sup>. The insight of the Reusable Holdout was to leak less information than the naive holdout, in part by only releasing a noisy estimate of how well the hypothesis fits the holdout set (instead of, say, the exact p-value), thus preventing overfitting to the holdout set.

We extend the reusable holdout idea, and propose that the holdout set should in fact leak no information about the hypothesis test, except for what is absolutely necessary: whether the hypothesis passed or not on the holdout set. In particular, it should not release any indication of “how well” the hypothesis fits the holdout set, such as a p-value. By this simple modification, our proposal allows exponentially-many adaptivelychosen false hypotheses to be invalidated, before the scientist discovers a true hypothesis. Moreover, it comes at no cost: in our scientific setting, there is no reason to leak more information from the holdout set, since ultimately we are only interested in preventing false discoveries. If we want to report p-values for the confirmed hypothesis, this can be done by simply keeping another holdout set, to use after the Generic Holdout, specifically for the purpose of finding the p-values for validated hypotheses that will be published.

Proposed Method: The Generic Holdout. To recap, our method is simple: first, the scientist collects the data, and keeps a holdout set for validation (without looking at it). The scientist is free to explore the rest of the data (exploration set) to come up with hypotheses that he/she deems plausible. Each time the scientist proposes a hypothesis, the validation procedure only returns “True” or “False”: whether the hypothesis passed validation on the holdout set, or not — it should not release more information, like the p-value of the hypothesis. The scientist can revisit the rest of the data, and adaptively propose hypotheses, and continue until he/she proposes a hypothesis that is confirmed by validation. In general, the scientist can continue this process until a small number of hypotheses are confirmed.

We stress that our method applies specifically to the case where the scientist’s goal is to derive a small number of true hypotheses, and will stop adaptively proposing hypotheses once several of them are validated to be true.

These ideas are not technically novel, but we are not aware of the problem of how to prevent false discoveries in science being phrased and addressed with such minimal assumptions, and with the guarantees we provide.

Remark 1. Here we point out a benefit of the Generic Holdout, and clarify the sense in which it is “adaptive,” by contrasting it with a related method.

An alternative way of using the holdout data set to provide a statistically sound and sample eficient methodology is the following: The data analyst interacts with the exploration set in an arbitrary way, and without looking at the holdout set produces a family of hypotheses, that is then simultaneously validated against the holdout set. In fact, this scenario is technically very similar to the Generic Holdout (as described below), but practically very diferent.

Indeed, let us consider the following thought experiment. The scientist is using the exploration set to come up with a hypotheses, and yet whenever they have a hypothesis in mind, the scientist pretends that it has been invalidated on the holdout set, and proceeds to come up with the next hypothesis. Eventually, he/she would come up with a family of hypotheses independent of the holdout set, and could validate all of those simultaneously. This is exactly the same sequence that would be generated in the real interaction with the Generic Holdout, except potentially longer.

However, implementing the above thought experiment in practice is infeasible: generating the set of hypotheses up front requires the scientist to simulate their behavior in the (hypothetical) case that every hypothesis they propose to the holdout is false. This is infeasible in settings where generating hypotheses is very expensive (in CPU-hours, or scientist-hours), or settings where the scientist cannot properly simulate themselves.

Moreover, it is unclear how many hypotheses should be generated in such a way, before moving to the validation stage. For example, consider large-scale physics experiments, where a scientific process is to first collect large amounts of data, and then explore the data to find interesting structures, and propose physical theories. Here, the process of investigating the data and coming up with theories of physics is extremely expensive. In this case, our proposal allows scientists to only invest efort in this process while they still have not derived a true hypothesis.

Applications. The primary application of the Generic Holdout, as discussed above, is to allow for adaptivity while preventing false discoveries — in particular, as an alternative to pre-registration. Here, our proposed method does not require the scientist to specify hypotheses ahead of time, before analyzing the data. Instead, the scientist only needs to specify how to determine if a hypothesis is significant or not. Using the Generic Holdout, the scientist can use any data analysis method (valid, or not) on their exploration set, and as long as they check with the holdout mechanism before publishing, they will not publish a false hypothesis except with small probability. Moreover, it is sample-eficient: they can ask up to exponentially-many false hypotheses to the holdout set, before a true hypothesis is confirmed.

Journals could even require that researchers submit their holdout set to the journal, without looking at it, and then journals implement the Generic Holdout mechanism themselves. That is, researchers (potentially interactively) submit hypotheses (with associated hypothesis tests) to the journals, which respond with a single bit.

The Generic Holdout also naturally applies to any data analysis procedure that involves several steps, each of which needs to be validated. For example, suppose the data analyst would like to first check “is the data well-clustered into 10 clusters?”, and then based on this, search for a good kernel embedding of the data, et. cetera. The Generic Holdout allows for such procedures to be statistically sound, without requiring any understanding of the exact statistical properties of the analyst’s queries.

Another application of the Generic Holdout is in fields where the existing scientific process appears to be working, but journals would like to have statistical soundness guarantees. This is especially relevant when a large data set is collected once and made public, and many research groups subsequently investigate and publish findings about the data set (for example, as in Genome-Wide Association Studies [23, 9]). Here, the proposal is: journals request some holdout data from the group initially collecting the data set, and put it in a vault (without publishing it, or looking at it). For every submitted study using the common data set, journals first do their usual review process. Then, when a paper passes their usual review, they do a final validation check on the holdout data. In this setting, the Generic Holdout guarantees that the journal can validate exponentially-many true hypotheses, and the holdout is only extinguished once it catches several false hypotheses. (Note, this is a complementary setting to the first application.)

Organization. In Section 1.1, we discuss lines of prior work on adaptivity in the sciences, related to our proposal. We formally describe our scientific setup and goals in Section 2, phrased in the language of statistical hypothesis testing. We provide an extended formal example of problems that arise due to adaptivity in data analysis in Section 2.1. In Section 3 we describe our proposed method, the Generic Holdout, and formally state its statistical guarantees. We include an example instantiation of our generic framework in Section 3.2, illustrating a common setting where exponentially-many hypotheses can be tested until several true ones are discovered.

## 1.1 Related Works

There are many related works surrounding the problem of adaptivity in the sciences; we discuss and compare the most relevant ones below.

Reusable Holdout. The proposal most similar to ours is the Reusable Holdout [11, 10], which developed out of ideas from Diferential Privacy [13] and Adaptive Data Analysis [12]. The Reusable Holdout addresses a similar scientific problem — preventing false discoveries in data analysis — and proposes a very similar methodology. However, there are several key diferences that allow us to improve on the Reusable Holdout in our setting.

The Reusable Holdout is a mechanism for interacting with the holdout set in (informally) the following way. When the scientist proposes a hypothesis, the mechanism first checks if the hypothesis “looks similar” on the holdout and exploration sets (i.e., if they have similar p-values, a measure of how well the hypothesis fit the data). If they are indeed similar, the mechanisms essentially releases the p-value of the hypothesis on the exploration set, not involving the holdout data. If they are very diferent, the mechanism releases a noisy p-value on the holdout set. This mechanism leaks information about the holdout set (a noisy p-value) whenever the scientist proposes “bad” hypotheses, which are overfit to the exploration data. As a result, the Reusable Holdout can handle only quadratically-many “bad” hypotheses in the size of the holdout set. In our setting, where the scientist may use an arbitrary exploration procedure to generate hypotheses, many of the proposed hypotheses may in fact be “bad”, and the Reusable Holdout would quickly become unusable. In contrast, our method allows for up to an exponential number of “bad” hypotheses, as long as the scientist stops after discovering a few true ones.

We stress that the technical details of the Generic Holdout are not novel (e.g., the SparseValidate mechanism of [10] is essentially the complement of our mechanism), but we believe our formalization of the scientific problem is meaningful, and our proposal cleanly solves this problem.

As an aside, note that the Reusable Holdout also releases estimated p-values of hypotheses, while the Generic Holdout only releases binary responses. However, to solve the scientific problem of preventing false discoveries, releasing binary responses is suficient. Moreover, if we would like p-values for the validated hypotheses, we can estimate these by simply keeping another small holdout set. Finally, the Generic Holdout allows for testing a much more general class of hypotheses than the Reusable Holdout, and does not require any specialized analysis per hypothesis class.

Adaptive Data Analysis. The recently-developed theory of Adaptive Data Analysis [10, 12, 3, 18] also addresses issues of how to do valid, adaptive science. At a high level, the goal of Adaptive Data Analysis is much more ambitious than our goal: there, the goal (informally) is to address the question of how to form good, generalizable hypotheses based on data. In contrast, our goal is merely to prevent a scientist from being convinced of false hypotheses; we do not give a procedure for deriving true hypotheses in the first place.

More formally, in Adaptive Data Analysis we have some underlying distribution D on the universe U, and the scientist would like to (approximately) know the result of statistical queries $\mathbb { E } { \boldsymbol { { X } } } { \sim } { \boldsymbol { { D } } } \big [ \phi _ { i } ( { \boldsymbol { x } } ) \big ]$ for some adaptivelychosen sequence of queries $\phi _ { i } : U \to [ 0 , 1 ]$ . The mechanism has access only to samples $X _ { 1 } , \dots X _ { n } \sim D$ , and must answer the queries with an estimate $\hat { \mu } _ { i } \approx \mathbb { E } _ { X \sim D } [ \phi _ { i } ( X ) ]$ . Thus, a scientist only interacting with the data through such a mechanism will always receive answers close to the true answers on the distribution, and thus will not generate hypotheses which are overfit to the data.

The tools developed in this area give computationally-eficient mechanisms for providing such estimates. However, due to the strong guarantees provided, these mechanisms can only correctly answer quadraticallymany queries in the size of the data set [3]. Moreoever, it is computationally intractable to answer more than polynomially-many queries in this setting [18].

We note that the Generic Holdout is well-suited to be used in conjunction with the methods of Adaptive Data Analysis. That is, we imagine a scientist using the tools of Adaptive Data Analysis (amongst possibly other methods) to generate hypotheses using the exploration set, and using the Generic Holdout to confirm them before publication.

Inference after Model Selection This recent line of work [14, 15] focuses on a specific kind of analysis procedure, which proceeds in two stages (model selection, and inference given the model). For example, if the analyst first selects influential variables via L -regularized regression (“model selection”) and then forms hypotheses based on these variables (“inference”). Adaptivity arises as a problem here for the same reason, since model-selection and inference are both performed on the same data set, and thus the hypotheses are dependent on the data they are tested against. This is essentially “2 rounds of adaptivity” in our setting. For specific kinds of data distributions and model-selection procedures, these works are able to precisely analyze how the hypotheses depend on the data, and thus give bounds on the performance of the overall procedure.

Our proposal is more general, in that it allows for multiple stages of adaptivity, each stage of which could be arbitrary. For example, our proposal would allow for model-selection in multiple stages, each of which needs to be validated with respect to the population distribution (e.g., “find a good embedding, then select influential variables, then cluster according to these...”). Moreover, our proposal makes no assumptions on the data distribution or hypothesis class, and can handle cases that may be hard to fully understand in the “inference after model selection” framework. Of course, for specific cases that can be understood, this framework could lead to tighter results than our generic framework.

Adaptive FDR Control. There is a related line of work that is interested in controlling the False-Discovery-Rate (FDR) of hypothesis testing. The setting here is as follows. We have a fixed large set of hypotheses, and we want to simultaneously test all of them on the same data set, while bounding the False-Discovery-Rate: the fraction of false hypotheses among all hypotheses that passed validation (i.e., the “false discoveries” among all discoveries). The scientific motivation here is often that we want to prune our hypotheses to a small set of “interesting” ones, on which we will then conduct further independent testing. For example, if we are interested in finding genes which cause a disease, we may first test all the hypotheses “gene X is correlated with the disease” for every value of gene X, using a testing procedure with bounded FDR. Then, among the returned hypotheses, we can do further experiments to determine their efect (say, looking physically at the mechanism of the gene expression). Here, controlling the FDR is important, since we do not want to invest too many resources into experiments which are likely to be null.

There are various proposed methods for controlling FDR in diferent settings [4, 1], and in particular, recently there have been proposals to control FDR by adaptively deciding the order in which to test hypotheses, based on the results of past hypothesis tests [19, 21, 20]. This could potentially yield more powerful tests, i.e. tests that are more likely to discover true hypotheses.

These works on [adaptive] FDR control operate in a diferent setting from our work, first because their notion of “adaptive” is diferent (the large set of hypotheses is usually assumed to be fixed beforehand), and second because they are interested in a diferent notion of error (the controlling the false-discovery rate, instead of preventing false discoveries overall). In the statistical terminology, our proposal controls the “family-wise error rate (FWER)” instead of the FDR.

## 2 The Scientific Framework

In this section, we define our scientific framework, in the language of hypothesis testing.

There exists some universe U and underlying true distribution D on U, specified by Nature. (For example, U could be genomic sequences, and D the true distribution of human genome.)

The scientist can form hypotheses $H \in { \mathcal { H } }$ about the true distribution (eg, “gene X is correlated to disease Y”). Each hypothesis corresponds to a partition of set of all distributions into a Null class $\mathcal { D } _ { H } ^ { N u l l }$ and an Alternative class $\mathcal { D } _ { H } ^ { A l t }$ . (The Null class defines distributions where the hypothesis is false, and the Alternative where the hypothesis is true).

For each hypothesis H in the hypothesis class , we have a hypothesis test Test ${ \bf \Pi } _ { H } ^ { ( n ) } : U ^ { n }  \{ 0 , 1 \}$ which takes n independent samples from a distribution and is supposed to accept under distributions in $\mathcal { D } _ { H } ^ { \bar { A } l t }$ and reject under those in $\mathcal { D } _ { H } ^ { N u l l }$ . The false-positive probability of each test is known as its p-value, and is given by

$$
p := \sup _ {D \in \mathcal {D} _ {H} ^ {N u l l}} \left\{\operatorname * {P r} _ {X _ {1}, \dots X _ {n} \sim D} [ \text {Test} _ {H} (X _ {1}, \dots X _ {n}) \text {accepts} ] \right\}
$$

In classical (non-adaptive) science, the scientific process is: we first fix some hypothesis H, and then col lect independent samples $X _ { 1 } , X _ { 2 } , . . . X _ { n } \sim D$ from the true distribution D, and run the hypothesis test $\mathrm { T e s t } _ { H } ^ { ( n ) } ( X _ { 1 } , . . . X _ { n } )$

For a single fixed hypothesis, we are usually interested in controlling the false-positive probability of the hypothesis test. This gives evidence for believing in hypotheses which pass the hypothesis test, in the following sense: Suppose a hypothesis test for hypothesis H has false-positive probability $p \ll 1$ . Then, if the hypothesis were false, our experimental procedure would have invalidated it with large probability $( 1 - p )$

The setting where we have a fixed set of hypotheses, and want to test them all simultaneously, is known as multiple hypothesis testing. In this setting, we could want to control diferent notions of error — for example, controlling the overall probability of confirming a false hypothesis, or controlling the fraction of false hypotheses among confirmed hypotheses. Throughout this work, we will consider controlling the overall probability of confirming a false hypotheses (and further, our hypotheses will be generated adaptively).

In particular, we consider the general adaptive scientific process as follows. The scientist first collects a data set $X _ { 1 } , X _ { 2 } , . . . X _ { n } \sim D$ of n independent samples from D. Then, the scientist is interested in exploring the data set to find true hypotheses, and will eventually propose a hypothesis (or small set of hypothesis) that s/he believes to be true. We would like to guarantee that the finally proposed hypotheses are in fact true that is, we want to bound the false-positive probability of the proposed hypotheses.

The Generic Holdout is a general, sample-eficient method to achieve this.

## 2.1 The Problem with Adaptivity

In this section, we give an extended formal example that illustrates the problem of adaptivity in data analysis (a version of what is known a “Freedman’s paradox” [16].)

Naively, if we collect a data set, form a hypothesis based on it, and then test the hypothesis on the same data set, we lose all guarantees of correctness. This is essentially because if we are allowed to adapt to our data set (and choose among many hypotheses), we can easily “overfit” to our data set, and find some hypothesis that is true about the data but not true in Nature. As an informal example, say we collect data on a set of 20 random people. Let their set of names be S. Then we form the hypothesis “At least 99% of people have names in $\mathrm { S } _ { \cdot } ^ { \mathfrak { n } }$ Clearly this hypothesis is well-supported by the data, but entirely false. Moreover, this hypothesis would be correctly rejected if it were formed a priori, and tested on an independent set of people.

The above problem still exists if we do not look at the data set directly, but we are allowed to adaptively choose hypotheses to test. That is, as a scientist we are not committed to a set of hypotheses beforehand, but rather we are interested in exploring the data set to find interesting structures. So we will first test some hypothesis $H _ { 0 }$ against the data set, and then seeing the results of this test (say, its p-value), we pick another hypothesis $H _ { 1 }$ to test, and so on. In the example below, we will see that this can easily lead to a scientist being convinced of a false hypothesis $H _ { k } ,$ , which appears to be true on the data set (i.e. passes validation with low p-value). Roughly what happens is, the scientist will test a series of “weak” hypotheses, and seeing the results of these hypotheses tests, will combine them into a single “strong” hypothesis which is over-fit to the data set.

Formal Example. Let us consider the universe $U : = \mathbb { R } ^ { d + 1 }$ , and distributions over $( x _ { 1 } , \dots x _ { d } , y ) \in \mathbb { R } ^ { d + 1 }$

We will form a sequence of hypothesis $H _ { i }$ . Each hypothesis is of the form: $y$ is positively correlated with $\langle w , x \rangle$ for $| | w | | _ { 2 } = 1$ . That is, each hypothesis ${ \cal H } ^ { ( w ) }$ is specified by $w ,$ and the Alternative class for $H ^ { ( w ) }$ corresponds to distributions on $( \vec { x } , y )$ for which $\mathbb { E } [ y \cdot \langle w , x \rangle ] > 0$ . (The Null class for $H ^ { ( w ) }$ is the complement of the Alternative class).

Note that the distribution where $( \vec { x } , y )$ are i.i.d. Gaussians $\mathcal { N } ( 0 , 1 )$ belongs to the Null class for all hypotheses. Call this distribution the “Global $\mathrm { N u l l } .$ 2

For a single, a priori fixed hypothesis $H ^ { ( w ) }$ , it is suficient to take $n = O ( \log ( 1 / p ) )$ independent samples from the distribution in order to test this hypothesis with false-positive probability $p .$ That is, the hypothesis test for $H ^ { ( w ) }$ takes n samples $\{ ( \boldsymbol x ^ { ( i ) } , \boldsymbol y ^ { ( i ) } ) \}$ , and tests if the empirical correlation $\begin{array} { r } { \big ( \frac { 1 } { n } \sum _ { i } y ^ { ( i ) } \langle w , x ^ { ( i ) } \rangle \big ) > 1 } \end{array}$ . This test has p-value $p ,$ meaning that under any distribution from the Null class of $H ^ { ( w ) }$ , this test rejects except with probability p (said another way, p is the “false-positive” probability).

Similarly, for any a priori fixed set of k hypotheses, it is suficient to take $n = O ( \log ( k / p ) )$ samples. In statistical parlance, this is equivalent to the “Bonferroni Procedure”, i.e. the Union Bound, which says that to test k fixed hypotheses simultaneously with error level $p ,$ one should test each individual hypothesis using at level $( p / k )$

Now, suppose we are in the Global Null distribution, and consider the following scientist who is trying to find a positive hypothesis in the class defined above. We will make only $k = d { + } 1$ queries total, so we decide to take $n = O ( \log ( d / p ) )$ ) samples from the distribution (this is incorrect as we will see, since it assumes our queries were fixed in advance). For the first d queries, the scientist tests the hypotheses $H ^ { ( w _ { 0 } ) } , H ^ { ( w _ { 1 } ) } , \dots H ^ { ( w _ { d } ) }$ for $w _ { i } = \vec { e } _ { i }$ the i-th standard basis vector. Knowing the $p \textmd { - }$ values from these tests, the scientist knows the empirical correlations $\begin{array} { r } { c _ { i } : = \tilde { \mathbb { E } } [ y x _ { i } ] : = \frac { 1 } { n } \sum _ { j } y ^ { ( j ) } x _ { i } ^ { ( j ) } } \end{array}$ between each of the coordinates $x _ { i }$ and $y$ on the samples. Each of these empirical correlations will have magnitude $| \tilde { \mathbb { E } } [ y x _ { i } ] | \gtrapprox 1 / \sqrt { n }$ in expectation. Now for the final query, the scientist checks the hypothesis $\begin{array} { r } { w ^ { * } : = \frac { 1 } { \sqrt { d } } \mathrm { s i g n } ( c ) } \end{array}$ . This has empirical correlation $\begin{array} { r } { \tilde { E } [ y \langle w ^ { * } , x \rangle ] > \frac { \sqrt { d } } { \sqrt { n } } } \end{array}$ by construction, since we sum all the coordinate-wise correlations. Note that with our choice of $n _ { \colon }$ we have $\begin{array} { r } { \frac { \sqrt { d } } { \sqrt { n } } \gg 1 } \end{array}$ , meaning this hypothesis test passes, even though we were in the Null distribution.

Conclusions. The above shows that methods to do hypothesis-testing with a fixed set of hypothesis (eg, controlling the p-values using the “Bonferroni Procedure”/union bound) can fail catastrophically when these hypotheses are chosen adaptively, knowing the results of previous hypothesis tests. In particular, a method for a priori testing may require exponentially more samples to be correct for adaptive testing. Note that this counterexample continues to hold if hypotheses are tested using cross-validation (ie, each hypothesis is tested on a diferent random subset of the data set.)

Remark 2. Looking closer at the above example, what happened is that the scientist first tested many “weak” hypotheses, which failed validation, but then combined the results of these weak hypotheses into a “strong” hypothesis, which passed validation. The Generic Holdout prevents such failures, by not releasing any additional information about weak hypotheses which do not validate.

The first (naive) example discussed in this section is a trivial manifestation of the problem with adaptivity, where the scientist is ridiculously malevolent. The second example, however, is much more enlightening and could serve as an abstraction for a mistake done by an honest, yet not careful enough scientist!

## 3 Proposed Method: The Generic Holdout

We propose the following scientific methodology (the “Generic Holdout”).

1. Take n independent samples, and partition them into a exploration set and a holdout set.

2. Set aside the holdout set and never look at it directly.

3. Use the exploration set freely, in any way, to adaptively explore and propose hypotheses.

4. When you have a plausible hypothesis H in hand, prepare a hypothesis test for H, with desired p-value, and apply this test on the holdout set, observing only the outcome of the test (whether it rejects the null hypothesis or not).

It is crucial that the binary outcome of the test is the only information observed from the holdout set. One must not observe more information, for example the actual p-value of the test on the holdout set.

5. You are free to adaptively repeat steps 3 and 4 to discover small number of true hypotheses.

## 3.1 Statistical Guarantees of the Generic Holdout

Here we set up some notation regarding the methodology proposed above that will be useful in further discussion. We consider some universe $U$ , and collect a data set $U _ { 1 } , \dots U _ { n } \in U$ , assumed to be a sequence of independent samples from some underlying population — probability distribution $\mathcal { D }$ over U. We partition it into $U _ { 1 } , \dots U _ { h }$ the holdout set, and $U _ { h + 1 } , . . . U _ { n } \gets$ the exploration set. The scientist uses exploration set to propose hypotheses $H _ { 1 } , H _ { 2 } , \dots , H _ { s }$ together with tests $\mathrm { T e s t } _ { H _ { i } }$ for each of them — each of hypotheses $H _ { i }$ can depend arbitrarily on the exploration set, and on results of all the previous tests.

When a scientist commits to use this mechanism until the number of validated hypotheses exceeds specific threshold $k ,$ or number of hypotheses tested altogether exceeds some specific threshold s, we wish to give strong statistical guarantee on the false positive rate for validated hypotheses. We focus on the scenario where $k \ll s ,$ i.e. we wish to discover only several true hypotheses, and we show that in this situation, the necessary size of the holdout set to achieve a fixed false positive probability scales gracefully with the total number of trials s.

The choice of the size of exploration set is not relevant to this discussion; clearly larger exploration set makes it easier for the scientist to produce valid hypotheses in the first place, but the acquisition and maintenance of larger data set is often related with additional costs.

We will now formally define the adaptive hypothesis selection mechanism.

Definition 1 (Adaptive hypothesis selection). We define the k-bounded adaptive hypothesis selection to be a sequence $o f$ (randomized) functions $\mathrm { A l g _ { 1 } } , \ldots . . \mathrm { A l g _ { \it s } }$ such that ${ \mathrm { A l g } } _ { i } : U ^ { n - h } \times \{ 0 , 1 \} ^ { s - 1 } \to { \mathcal { H } } \cup \{ \perp \}$ . We think of $\mathrm { A l g } _ { i }$ as a randomized scheme specifying how to pick $H _ { i } ,$ based on the exploration set, and results of all previous hypotheses tests. We assume that after finding k valid hypotheses, the researcher stops exploration, i.e. $\mathrm { A l g } _ { i } ( \vec { U } , x _ { 1 } , \dots x _ { i - 1 } ) = \perp$ whenever there are k ones among $x _ { 1 } , \ldots . x _ { i - 1 }$

Our main theorem quantifies the false-positive guarantees of the generic holdout test.

Theorem 1. Consider a sequence of hypotheses $H _ { 1 } , H _ { 2 } , H _ { 3 } , \dots H _ { s } \in { \mathcal { H } } \cup \{ \bot \}$ generated as in Definition 1, that is, the scientist adaptively generates up to s hypotheses, and stops once k hypotheses are confirmed. If the $p \cdot$ -value of each test $H _ { i }$ is bounded by $p ,$ then probability of false discovery in this workflow is bounded by $s ^ { k } p$ . More formally,

$$
\begin{array}{l} \forall D \in \mathcal {D}, \operatorname * {P r} [ \text {Scientist accepts a false hypothesis} ] \\ = \operatorname * {P r} _ {U _ {1}, \ldots U _ {n} \sim D} \big (\exists i \leq s, D \in D _ {H _ {i}} ^ {N u l l} \wedge \mathrm{Test} _ {H _ {i}} (U _ {1}, \ldots U _ {h}) = 1 \big) \leq s ^ {k} p. \end{array}
$$

The proof of this theorem is elementary, before we proceed with it let us state explicitly important interpretation of its statement.

Discussion. In order to achieve some target statistical significance, say $p _ { 0 } = 0 . 0 5$ , over the whole process described above, we want to use holdout set such that the guaranteed false-positive probability $p$ for each specific test $H _ { i }$ is of the order of $p _ { 0 } \mathord { \left/ { \vphantom { p _ { 0 } \varepsilon } } \right. \kern - delimiterspace } s ^ { k }$ . Often for standard statistical tests the required samples size scales like $\begin{array} { r } { \mathcal { O } ( \log { \frac { 1 } { p } } ) } \end{array}$ with the desired p-value, and as such it is enough to use the holdout set of size $\mathcal { O } ( k \log s )$

To put it diferently, once we have fixed holdout set of size $h ,$ desired p-value $p _ { 0 }$ and bound k on the number of discovered $^ { 6 6 } \mathrm { t r u e } ^ { 9 5 }$ hypotheses (after which we stop using collected holdout set for verification), we can issue $s = 2 ^ { \Omega ( h / k ) }$ queries in the workflow described above, and still have confidence $p _ { 0 }$ on the validity of all discovered hypotheses.

Remark. For $k = 1$ , this bound exactly matches the “Bonferroni Procedure” (ie, the union bound) for testing a fixed set of s non-adaptive hypotheses.

Remark. The statement of the theorem remains unafected in the complementary setting, where we expect number of rejected hypotheses to be bounded by k. Here the scientist. Here, we expect scientist to use the mechanism until at most k hypotheses are rejected, or at most s queries are issued. In this scenario, we can again bound the probability of any false discovery by $s ^ { k } p$

Remark. Note that simply providing a mechanism for validating hypotheses with small probability of false discoveries is trivial: the mechanism can just respond that every hypothesis tested is false. We would like mechanisms to also be $u s e f u l ,$ in that they allow for true discoveries. One possible formalization of usefulness guarantees of the Generic Holdout, for $k = 1$ , is as follows. Intuitively, we want to say that a scientist who follows a strategy that eventually proposes a valid hypothesis, will discover this hypothesis while using the Generic Holdout. More formally, for a hypothesis $H \in { \mathcal { H } } .$ , distribution $D \in { \mathcal { D } }$ and some associated test $\mathrm { T e s t } _ { H }$ we define $p _ { H , D } : = \operatorname* { P r } _ { X _ { 1 } , \ldots X _ { h } \sim D } ( \operatorname { T e s t } _ { H } ( X _ { 1 } , \ldots X _ { m } ) = 1 )$ . For $\mathrm { A l g } _ { 1 } , \ldots . . \mathrm { A l g } _ { s }$ as in Definition 1, and some distribution $D \in \mathcal { D }$ , we have

$$
\Pr_{\substack{U_{1},\ldots U_{h}\sim D\\ H_{1},\ldots H_{s}\leftarrow \operatorname{Alg}(U_{1},\ldots U_{h})}}(\exists i\leq s,  \operatorname{Test}_{H_{i}}(U_{1},\ldots U_{h}) = 1)\geq \mathop{\mathbb{E}}_{\substack{U_{1},\ldots U_{h}\sim D\\ H_{1},\ldots H_{s}\leftarrow \operatorname{Alg}(U_{1},\ldots U_{h})}}\max_{i}p_{H_{i},D}.
$$

Proof of Theorem 1. Observe that, as $U _ { 1 } , \dots , U _ { h }$ are assumed to be independent from $U _ { h + 1 } , \dots U _ { n } .$ , and the internal randomness of the scientist. Let us, for now, assume that the selection of the i-th hypothesis depends only on the results of all previous tests $\operatorname { A l g } _ { i } : \{ 0 , 1 \} ^ { i - 1 } \to { \mathcal { H } } \cup \{ \bot \}$ in a deterministic way.

Note that for a fixed sequence $\operatorname { A l g } _ { 1 } , \operatorname { A l g } _ { 2 } , . . . \operatorname { A l g } _ { 4 }$ as above (i.e. we assume that $\mathrm { A l g } _ { i } ( x _ { 1 } , \dots x _ { i - 1 } ) = \perp$ if there are at least $k$ ones among $x _ { 1 } , \ldots x _ { i - 1 } )$ , there is at most s $\textstyle \sum _ { i < k } { \binom { s } { k } } \leq s ^ { k }$ hypotheses that will ever be tested by this algorithm — this is a bound on the total range of all those functions. Consider the set $\tilde { \mathcal { H } } \subset \mathcal { H }$ given by union of all the ranges of ${ \mathrm { A l g } } _ { i }$ . We know that $| \tilde { \mathcal { H } } | \leq s ^ { k }$ , and moreover if we fix $D \in \mathcal { D }$ , we have

$$
\begin{array}{l l} \operatorname * {P r} _ {U _ {1}, \ldots U _ {h} \sim D} (\exists i \leq s, H _ {i} \in \mathcal {D} _ {H _ {i}} ^ {N u l l} \wedge \mathrm{Test} _ {H _ {i}} (U _ {1}, \ldots U _ {h}) = 1) \\ & \leq \operatorname * {P r} _ {U _ {1}, \ldots , U _ {h} \sim D} (\exists H \in \tilde {\mathcal {H}}, \mathcal {D} _ {H} ^ {N u l l} \wedge \mathrm{Test} _ {H _ {i}} = 1) \\ & \leq | \mathcal {H} | p \\ & \leq s ^ {k} p. \end{array}
$$

For general case, where ${ \mathrm { A l g } } _ { i }$ is a randomized function that depends also on the exploration set $U _ { h + 1 } , \dots U _ { n }$ we can use the linearity of expectation — conditioning on any deterministic realization of ${ \mathrm { A l g } } _ { i }$ , and the value of exploration set $U _ { h + 1 } , \dots U _ { n } ,$ the statement is true by the argument above, and therefore it is true, in expectation over those random variables. □

## 3.2 Example: Gapped Empirical Losses

In many natural situations, the hypothesis test takes a special form: thresholding an empirical loss evaluated on the sample at hand. Our general framework specializes to this case, and here we can give quantitative bounds on the number of samples n required to bound false-positive rate.

Specifically, suppose that with each hypothesis H we have some associated loss function $\ell _ { H } : U \to [ - 1 , 1 ]$ such that

$$
\forall D \in \mathcal {D} _ {H} ^ {N u l l}: \underset {x \sim D} {\mathbb {E}} \left[ \ell_ {H} (x) \right] \leq 0,
$$

moreover, suppose the hypothesis test is simply

$$
\operatorname{Test} _ {H} \left(X _ {1}, \dots X _ {h}\right) = \mathbf {1} \left\{\left(\frac {1}{h} \sum_ {i} \ell_ {H} \left(X _ {i}\right)\right) > 1 / 2 \right\}.\tag{1}
$$

In this case, we give quantitative bounds on the number of samples n required for constant statistical confidence on validated hypotheses within the Generic Holdout framework.

Theorem 2. If the scientist makes s adaptive hypothesis test queries (generated as in Definition 1) on the holdout set, including at most k that are confirmed to be valid, where each hypothesis test is of form (1) then using holdout set of size $h = O ( t \log ( k / p _ { 0 } ) )$ is suficient to guarantee that the probability of confirming a false hypothesis is at most $p _ { 0 }$

One concrete realization of this class of hypothesis tests is following. Consider the class of multivariate normal distributions with covariance matrix bounded in spectral norm by 1, and the problem of finding a linear predictor $h ( x ) : = \langle w , x \rangle$ that is correlated with target feature y. With each vector w of unit norm, we can consider associated loss function $\ell _ { w } ( x , y ) = \mathrm { t r u n c a t e } _ { [ - 1 , 1 ] } ( \langle w , x \rangle )$ , where trunca $\begin{array} { l } { \mathbf { \dot { e } } _ { [ - 1 , 1 ] } ( x ) = } \end{array}$ min $( \operatorname* { m a x } ( x , - 1 ) , 1 )$ . Hypotheses of this form can be generated by using linear regression on the exploration set, and then verified on the holdout set. Theorem 2 states that we can validate exponentially many hypotheses (with respect to the size of given holdout set), as long as we stop upon discovering few true hypotheses of this form.

## 4 Acknowledgements

We would like to thank Madhu Sudan, Boaz Barak, Lucas Janson, Jonathan Shi, and Thibaut Horel for helpful discussions during the course of this work.

## References

[1] Rina Foygel Barber, Emmanuel J Candès, et al. Controlling the false discovery rate via knockofs. The Annals of Statistics, 43(5):2055–2085, 2015.

[2] Deanna M Barch and Tal Yarkoni. Introduction to the special issue on reliability and replication in cognitive and afective neuroscience research, 2013.

[3] Raef Bassily, Kobbi Nissim, Adam Smith, Thomas Steinke, Uri Stemmer, and Jonathan Ullman. Algorithmic stability for adaptive data analysis. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pages 1046–1059. ACM, 2016.

[4] Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the royal statistical society. Series B (Methodological), pages 289–300, 1995.

[5] Colin F Camerer, Anna Dreber, Eskil Forsell, Teck-Hua Ho, Jürgen Huber, Magnus Johannesson, Michael Kirchler, Johan Almenberg, Adam Altmejd, Taizan Chan, et al. Evaluating replicability of laboratory experiments in economics. Science, 351(6280):1433–1436, 2016.

[6] Colin F Camerer, Anna Dreber, Felix Holzmeister, Teck-Hua Ho, Jürgen Huber, Magnus Johannesson, Michael Kirchler, Gideon Nave, Brian A Nosek, Thomas Pfeifer, et al. Evaluating the replicability of social science experiments in nature and science between 2010 and 2015. Nature Human Behaviour, page 1, 2018.

[7] C. Chambers, M. Munafo, et al. Trust in science would be improved by study pre-registration. Guardian US, June 2013. https://www.theguardian.com/science/blog/2013/jun/05/trust-in-science-study-pre-registration.

[8] Open Science Collaboration et al. Estimating the reproducibility of psychological science. Science, 349(6251):aac4716, 2015.

[9] Wellcome Trust Case Control Consortium et al. Genome-wide association study of 14,000 cases of seven common diseases and 3,000 shared controls. Nature, 447(7145):661, 2007.

[10] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toni Pitassi, Omer Reingold, and Aaron Roth. Generalization in adaptive data analysis and holdout reuse. In Advances in Neural Information Processing Systems, pages 2350–2358, 2015.

[11] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. The reusable holdout: Preserving validity in adaptive data analysis. Science, 349(6248):636–638, 2015.

[12] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Leon Roth. Preserving statistical validity in adaptive data analysis. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing, pages 117–126. ACM, 2015.

[13] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pages 265–284. Springer, 2006.

[14] William Fithian, Dennis Sun, and Jonathan Taylor. Optimal inference after model selection. arXiv preprint arXiv:1410.2597, 2014.

[15] William Fithian, Jonathan Taylor, Robert Tibshirani, and Ryan Tibshirani. Selective sequential model selection. arXiv preprint arXiv:1512.02565, 2015.

[16] David A Freedman and David A Freedman. A note on screening regression equations. the american statistician, 37(2):152–155, 1983.

[17] Andrew Gelman and Eric Loken. The statistical crisis in science. American scientist, 102(6):460, 2014.

[18] Moritz Hardt and Jonathan Ullman. Preventing false discovery in interactive data analysis is hard. In Foundations of Computer Science (FOCS), 2014 IEEE 55th Annual Symposium on, pages 454–463. IEEE, 2014.

[19] Lihua Lei and William Fithian. Adapt: an interactive procedure for multiple testing with side information. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 80(4):649–679, 2018.

[20] Ang Li and Rina Foygel Barber. Multiple testing with the structure adaptive benjamini-hochberg algorithm. arXiv preprint arXiv:1606.07926, 2016.

[21] Ang Li and Rina Foygel Barber. Accumulation tests for fdr control in ordered hypothesis testing. Journal of the American Statistical Association, 112(518):837–849, 2017.

[22] Joseph P Simmons, Leif D Nelson, and Uri Simonsohn. False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. Psychological science, 22(11):1359–1366, 2011.

[23] Danielle Welter, Jacqueline MacArthur, Joannella Morales, Tony Burdett, Peggy Hall, Heather Junkins, Alan Klemm, Paul Flicek, Teri Manolio, Lucia Hindorf, et al. The nhgri gwas catalog, a curated resource of snp-trait associations. Nucleic acids research, 42(D1):D1001–D1006, 2013.
