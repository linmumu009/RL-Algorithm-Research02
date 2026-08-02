# Generalization in Adaptive Data Analysis and Holdout Reuse

Cynthia Dwork<sup>∗</sup> Vitaly Feldman<sup>†</sup> Moritz Hardt<sup>‡</sup> Toniann Pitassi<sup>§</sup>

Omer Reingold<sup>¶</sup>

Aaron Roth<sup>k</sup>

September 28, 2015

## Abstract

Overfitting is the bane of data analysts, even when data are plentiful. Formal approaches to understanding this problem focus on statistical inference and generalization of individual analysis procedures. Yet the practice of data analysis is an inherently interactive and adaptive process: new analyses and hypotheses are proposed after seeing the results of previous ones, parameters are tuned on the basis of obtained results, and datasets are shared and reused. An investigation of this gap has recently been initiated by the authors in [DFH<sup>+</sup>14], where we focused on the problem of estimating expectations of adaptively chosen functions.

In this paper, we give a simple and practical method for reusing a holdout (or testing) set to validate the accuracy of hypotheses produced by a learning algorithm operating on a training set. Reusing a holdout set adaptively multiple times can easily lead to overfitting to the holdout set itself. We give an algorithm that enables the validation of a large number of adaptively chosen hypotheses, while provably avoiding overfitting. We illustrate the advantages of our algorithm over the standard use of the holdout set via a simple synthetic experiment.

We also formalize and address the general problem of data reuse in adaptive data analysis. We show how the diferential-privacy based approach given in [DFH<sup>+</sup>14] is applicable much more broadly to adaptive data analysis. We then show that a simple approach based on description length can also be used to give guarantees of statistical validity in adaptive settings. Finally, we demonstrate that these incomparable approaches can be unified via the notion of approximate max-information that we introduce. This, in particular, allows the preservation of statistical validity guarantees even when an analyst adaptively composes algorithms which have guarantees based on either of the two approaches.

## 1 Introduction

The goal of machine learning is to produce hypotheses or models that generalize well to the unseen instances of the problem. More generally, statistical data analysis is concerned with estimating properties of the underlying data distribution, rather than properties that are specific to the finite data set at hand. Indeed, a large body of theoretical and empirical research was developed for ensuring generalization in a variety of settings. In this work, it is commonly assumed that each analysis procedure (such as a learning algorithm) operates on a freshly sampled dataset – or if not, is validated on a freshly sampled holdout (or testing) set.

Unfortunately, learning and inference can be more dificult in practice, where data samples are often reused. For example, a common practice is to perform feature selection on a dataset, and then use the features for some supervised learning task. When these two steps are performed on the same dataset, it is no longer clear that the results obtained from the combined algorithm will generalize. Although not usually understood in these terms, “Freedman’s paradox” is an elegant demonstration of the powerful (negative) efect of adaptive analysis on the same data [Fre83]. In Freedman’s simulation, variables with significant t-statistic are selected and linear regression is performed on this adaptively chosen subset of variables, with famously misleading results: when the relationship between the dependent and explanatory variables is non-existent, the procedure overfits, erroneously declaring significant relationships.

Most of machine learning practice does not rely on formal guarantees of generalization for learning algorithms. Instead a dataset is split randomly into two (or sometimes more) parts: the training set and the testing, or holdout, set. The training set is used for learning a predictor, and then the holdout set is used to estimate the accuracy of the predictor on the true distribution<sup>1</sup>. Because the predictor is independent of the holdout dataset, such an estimate is a valid estimate of the true prediction accuracy (formally, this allows one to construct a confidence interval for the prediction accuracy on the data distribution). However, in practice the holdout dataset is rarely used only once, and as a result the predictor may not be independent of the holdout set, resulting in overfitting to the holdout set [Reu03, RF08, CT10]. One well-known reason for such dependence is that the holdout data is used to test a large number of predictors and only the best one is reported. If the set of all tested hypotheses is known and independent of the holdout set, then it is easy to account for such multiple testing or use the more sophisticated approach of Ng [Ng97].

However such static approaches do not apply if the estimates or hypotheses tested on the holdout are chosen adaptively: that is, if the choice of hypotheses depends on previous analyses performed on the dataset. One prominent example in which a holdout set is often adaptively reused is hyperparameter tuning (e.g.[DFN07]). Similarly, the holdout set in a machine learning competition, such as the famous ImageNet competition, is typically reused many times adaptively. Other examples include using the holdout set for feature selection, generation of base learners (in aggregation techniques such as boosting and bagging), checking a stopping condition, and analyst-in-the-loop decisions. See [Lan05] for a discussion of several subtle causes of overfitting.

The concrete practical problem we address is how to ensure that the holdout set can be reused to perform validation in the adaptive setting. Towards addressing this problem we also ask the more general question of how one can ensure that the final output of adaptive data analysis generalizes to the underlying data distribution. This line of research was recently initiated by the authors in [DFH<sup>+</sup>14], where we focused on the case of estimating expectations of functions from i.i.d. samples (these are also referred to as statistical queries). They show how to answer a large number of adaptively chosen statistical queries using techniques from diferential privacy [DMNS06](see Sec. 1.3 and Sec. 2.2 for more details).

## 1.1 Our Results

We propose a simple and general formulation of the problem of preserving statistical validity in adaptive data analysis. We show that the connection between diferentially private algorithms and generalization from [DFH<sup>+</sup>14] can be extended to this more general setting, and show that similar (but sometimes incomparable) guarantees can be obtained from algorithms whose outputs can be described by short strings. We then define a new notion, approximate max-information, that unifies these two basic techniques and gives a new perspective on the problem. In particular, we give an adaptive composition theorem for max-information, which gives a simple way to obtain generalization guarantees for analyses in which some of the procedures are diferentially private and some have short description length outputs. We apply our techniques to the problem of reusing the holdout set for validation in the adaptive setting.

## 1.1.1 A Reusable Holdout

We describe a simple and general method, together with two specific instantiations, for reusing a holdout set for validating results while provably avoiding overfitting to the holdout set. The analyst can perform any analysis on the training dataset, but can only access the holdout set via an algorithm that allows the analyst to validate her hypotheses against the holdout set. Crucially, our algorithm prevents overfitting to the holdout set even when the analysts hypotheses are chosen adaptively on the basis of the previous responses of our algorithm.

Our first algorithm, referred to as Thresholdout, derives its guarantees from diferential privacy and the results in [DFH<sup>+</sup>14, NS15]. For any function $\phi : \mathcal { X }  [ 0 , 1 ]$ given by the analyst, Thresholdout uses the holdout set to validate that φ does not overfit to the training set, that is, it checks that the mean value of $\phi$ evaluated on the training set is close to the mean value of $\phi$ evaluated on the distribution $\mathcal { P }$ from which the data was sampled. The standard approach to such validation would be to compute the mean value of $\phi$ on the holdout set. The use of the holdout set in Thresholdout difers from the standard use in that it exposes very little information about the mean of $\phi$ on the holdout set: if $\phi$ does not overfit to the training set, then the analyst receives only the confirmation of closeness, that is, just a single bit. On the other hand, if φ overfits then Thresholdout returns the mean value of $\phi$ on the training set perturbed by carefully calibrated noise.

Using results from [DFH<sup>+</sup>14, NS15] we show that for datasets consisting of i.i.d. samples these modifications provably prevent the analyst from constructing functions that overfit to the holdout set. This ensures correctness of Thresholdout’s responses. Naturally, the specific guarantees depend on the number of samples n in the holdout set. The number of queries that Thresholdout can answer is exponential in n as long as the number of times that the analyst overfits is at most quadratic in n.

Our second algorithm SparseValidate is based on the idea that if most of the time the analysts procedures generate results that do not overfit, then validating them against the holdout set does not reveal much information about the holdout set. Specifically, the generalization guarantees of this method follow from the observation that the transcript of the interaction between a data analyst and the holdout set can be described concisely. More formally, this method allows the analyst to pick any Boolean function of a dataset $\psi$ (described by an algorithm) and receive back its value on the holdout set. A simple example of such a function would be whether the accuracy of a predictor on the holdout set is at least a certain value α. (Unlike in the case of Thresholdout, here there is no need to assume that the function that measures the accuracy has a bounded range or even Lipschitz, making it qualitatively diferent from the kinds of results achievable subject to diferential privacy). A more involved example of validation would be to run an algorithm on the holdout dataset to select an hypothesis and check if the hypothesis is similar to that obtained on the training set (for any desired notion of similarity). Such validation can be applied to other results of analysis; for example one could check if the variables selected on the holdout set have large overlap with those selected on the training set. An instantiation of the SparseValidate algorithm has already been applied to the problem of answering statistical (and more general) queries in the adaptive setting [BSSU15]. We describe the formal guarantees for SparseValidate in Section 4.2.

In Section 5 we describe a simple experiment on synthetic data that illustrates the danger of reusing a standard holdout set, and how this issue can be resolved by our reusable holdout. The design of this experiment is inspired by Freedman’s classical experiment, which demonstrated the dangers of performing variable selection and regression on the same data [Fre83].

## 1.2 Generalization in Adaptive Data Analysis

We view adaptive analysis on the same dataset as an execution of a sequence of steps $\mathcal { A } _ { 1 } \to \mathcal { A } _ { 2 } \to \cdot \cdot \cdot \to \mathcal { A } _ { m }$ Each step is described by an algorithm $A _ { i }$ that takes as input a fixed dataset $S = ( x _ { 1 } , \ldots , x _ { n } ) $ drawn from some distribution D over $\mathcal { X } ^ { n }$ , which remains unchanged over the course of the analysis. Each algorithm $A _ { i }$ also takes as input the outputs of the previously run algorithms $\mathcal { A } _ { 1 }$ through $\mathcal { A } _ { i - 1 }$ and produces a value in some range $\mathcal { \mathrm { V } } _ { i }$ . The dependence on previous outputs represents all the adaptive choices that are made at step i of data analysis. For example, depending on the previous outputs, $A _ { i }$ can run diferent types of analysis on S. We note that at this level of generality, the algorithms can represent the choices of the data analyst, and need not be explicitly specified. We assume that the analyst uses algorithms which individually are known to generalize when executed on a fresh dataset sampled independently from a distribution D. We formalize this by assuming that for every fixed value $y _ { 1 } , \dots , y _ { i - 1 } \in { \mathcal { V } } _ { 1 } \times \dots \times { \mathcal { V } } _ { i - 1 }$ , with probability at least $1 - \beta _ { i }$ over the choice of $S$ according to distribution $\mathcal { D } _ { : }$ the output of $A _ { i }$ on inputs $y _ { 1 } , \ldots , y _ { i - 1 }$ and S has a desired property relative to the data distribution D (for example has low generalization error). Note that in this assumption $y _ { 1 } , \ldots , y _ { i - 1 }$ are fixed and independent of the choice of $S _ { \mathrm { { ; } } }$ whereas the analyst will execute $\mathbf { \mathcal { A } } _ { i }$ on values $Y _ { 1 } , \dots , Y _ { i - 1 }$ , where $Y _ { j } = { \mathcal { A } } _ { j } ( S , Y _ { 1 } , \ldots , Y _ { j - 1 } )$ . In other words, in the adaptive setup, the algorithm $A _ { i }$ can depend on the previous outputs, which depend on $S ,$ and thus the set $S$ given to $\mathbf { \mathcal { A } } _ { i }$ is no longer an independently sampled dataset. Such dependence invalidates the generalization guarantees of individual procedures, potentially leading to overfitting.

Diferential privacy: First, we spell out how the diferential privacy based approach from $\mathrm { [ D F H ^ { + } 1 4 ] }$ can be applied to this more general setting. Specifically, a simple corollary of results in $\mathrm { [ D F H ^ { + } 1 4 ] }$ is that for a dataset consisting of i.i.d. samples any output of a diferentially-private algorithm can be used in subsequent analysis while controlling the risk of overfitting, even beyond the setting of statistical queries studied in $\mathrm { [ D F H ^ { + } 1 4 ] }$ . A key property of diferential privacy in this context is that it composes adaptively: namely if each of the algorithms used by the analyst is diferentially private, then the whole procedure will be diferentially private (albeit with worse privacy parameters). Therefore, one way to avoid overfitting in the adaptive setting is to use algorithms that satisfy (suficiently strong) guarantees of diferential-privacy. In Section 2.2 we describe this result formally.

Description length: We then show how description length bounds can be applied in the context of guaranteeing generalization in the presence of adaptivity. If the total length of the outputs of algorithms $\mathcal { A } _ { 1 } , \dotsc , \mathcal { A } _ { i - 1 }$ can be described with k bits then there are at most $2 ^ { k }$ possible values of the input $y _ { 1 } , \ldots , y _ { i - 1 }$ to $A _ { i }$ . For each of these individual inputs $A _ { i }$ generalizes with probability $1 - \beta _ { i }$ . Taking a union bound over failure probabilities implies generalization with probability at least $1 - 2 ^ { k } \beta _ { i }$ . Occam’s Razor famously implies that shorter hypotheses have lower generalization error. Our observation is that shorter hypotheses (and the results of analysis more generally) are also better in the adaptive setting since they reveal less about the dataset and lead to better generalization of subsequent analyses. Note that this result makes no assumptions about the data distribution D. We provide the formal details in Section 2.3. In Section B we also show that description length-based analysis sufices for obtaining an algorithm (albeit not an eficient one) that can answer an exponentially large number of adaptively chosen statistical queries. This provides an alternative proof for one of the results in $\mathrm { [ D F H ^ { + } 1 4 ] }$

Approximate max-information: Our main technical contribution is the introduction and analysis of a new information-theoretic measure, which unifies the generalization arguments that come from both diferential privacy and description length, and that quantifies how much information has been learned about the data by the analyst. Formally, for jointly distributed random variables $( S , Y )$ , the max-information is the maximum of the logarithm of the factor by which uncertainty about S is reduced given the value of $\mathbf { Y } .$ , namely $\begin{array} { r } { I _ { \infty } ( S , Y ) \doteq \log \operatorname* { m a x } \frac { \mathbb { P } [ S = S \mathrm { ~ \lvert ~ } \mathbf { Y = } y ] } { \mathbb { P } [ S = S ] } } \end{array}$ , where the maximum is taken over all S in the support of S and $y$ in the support Y. Informally, β-approximate max-information requires that the logarithm above be bounded with probability at least $1 - \beta$ over the choice of (S, Y) (the actual definition is slightly weaker, see Definition 10 for details).In our use, S denotes a dataset drawn randomly from the distribution D and Y denotes the output of a (possibly randomized) algorithm on S. We prove that approximate max-information has the following properties

• An upper bound on (approximate) max-information gives generalization guarantees.

• Diferentially private algorithms have low max-information for any distribution D over datasets. A stronger bound holds for approximate max-information on i.i.d. datasets. These bounds apply only to so-called pure diferential privacy (the $\delta = 0 ~ \mathrm { c a s e } )$

• Bounds on the description length of the output of an algorithm give bounds on the approximate max-information of the algorithm for any D.

• Approximate max-information composes adaptively.

• Approximate max-information is preserved under post-processing.

Composition properties of approximate max-information imply that one can easily obtain generalization guarantees for adaptive sequences of algorithms, some of which are diferentially private, and others of which have outputs with short description length. These properties also imply that diferential privacy can be used to control generalization for any distribution D over datasets, which extends its generalization guarantees beyond the restriction to datasets drawn i.i.d. from a fixed distribution, as in [DFH<sup>+</sup>14].

We remark that (pure) diferential privacy and description length are otherwise incomparable – low description length is not a suficient condition for diferential privacy, since diferential privacy precludes revealing even a small number of bits of information about any single individual in the data set. At the same time diferential privacy does not constrain the description length of the output. Bounds on max-information or diferential privacy of an algorithm can, however, be translated to bounds on randomized description length for a diferent algorithm with statistically indistinguishable output. Here we say that a randomized algorithm has randomized description length of k if for every fixing of the algorithm’s random bits, it has description length of k. Details of these results and additional discussion appear in Sections 3 and A.

## 1.3 Related Work

This work builds on [DFH<sup>+</sup>14] where we initiated the formal study of adaptivity in data analysis. The primary focus of [DFH<sup>+</sup>14] is the problem of answering adaptively chosen statistical queries. The main technique is a strong connection between diferential privacy and generalization: diferential privacy guarantees that the distribution of outputs does not depend too much on any one of the data samples, and thus, diferential privacy gives a strong stability guarantee that behaves well under adaptive data analysis. The link between generalization and approximate diferential privacy made in [DFH<sup>+</sup>14] has been subsequently strengthened, both qualitatively — by [BSSU15], who make the connection for a broader range of queries — and quantitatively, by [NS15] and [BSSU15], who give tighter quantitative bounds. These papers, among other results, give methods for accurately answering exponentially (in the dataset size) many adaptively chosen queries, but the algorithms for this task are not eficient. It turns out this is for fundamental reasons – Hardt and Ullman [HU14] and Steinke and Ullman [SU14] prove that, under cryptographic assumptions, no eficient algorithm can answer more than quadratically many statistical queries chosen adaptively by an adversary who knows the true data distribution.

Diferential privacy emerged from a line of work [DN03, DN04, BDMN05], culminating in the definition given by [DMNS06]. There is a very large body of work designing diferentially private algorithms for various data analysis tasks, some of which we leverage in our applications. See [Dwo11] for a short survey and [DR14] for a textbook introduction to diferential privacy.

The classical approach in theoretical machine learning to ensure that empirical estimates generalize to the underlying distribution is based on the various notions of complexity of the set of functions output by the algorithm, most notably the VC dimension(see e.g. [SSBD14] for a textbook introduction). If one has a sample of data large enough to guarantee generalization for all functions in some class of bounded complexity, then it does not matter whether the data analyst chooses functions in this class adaptively or non-adaptively. Our goal, in contrast, is to prove generalization bounds without making any assumptions about the class from which the analyst can choose query functions. In this case the adaptive setting is very diferent from the non-adaptive setting.

An important line of work [BE02, MNPR06, PRMN04, SSSSS10] establishes connections between the stability of a learning algorithm and its ability to generalize. Stability is a measure of how much the output of a learning algorithm is perturbed by changes to its input. It is known that certain stability notions are necessary and suficient for generalization. Unfortunately, the stability notions considered in these prior works do not compose in the sense that running multiple stable algorithms sequentially and adaptively may result in a procedure that is not stable. The measure we introduce in this work (max information), like diferential privacy, has the strength that it enjoys adaptive composition guarantees. This makes it amenable to reasoning about the generalization properties of adaptively applied sequences of algorithms, while having to analyze only the individual components of these algorithms. Connections between stability, empirical risk minimization and diferential privacy in the context of learnability have been recently explored in [WLF15].

Freund gives an approach to obtaining data-dependent generalization bounds that takes into account the set of statistical queries that a given learning algorithm can produce for the distribution from which the data was sampled [Fre98]. A related approach of Langford and Blum also allows to obtain data-dependent generalization bounds based on the description length of functions that can be output for a data distribution [LB03]. Unlike our work, these approaches require the knowledge of the structure of the learning algorithm to derive a generalization bound. More importantly, the focus of our framework is on the design of new algorithms with better generalization properties in the adaptive setting.

Finally, inspired by our work, Blum and Hardt [BH15] showed how to reuse the holdout set to maintain an accurate leaderboard in a machine learning competition that allows the participants to submit adaptively chosen models in the process of the competition (such as those organized by Kaggle Inc.). Their analysis also relies on the description length-based technique we used to analyze SparseValidate.

## 2 Preliminaries and Basic Techniques

In the discussion below log refers to binary logarithm and ln refers to the natural logarithm. For simplicity we restrict our random variables to finite domains (extension of the claims to continuous domains is straightforward using the standard formalism). For two random variables X and Y over the same domain X the max-divergence of X from Y is defined as

$$
D _ {\infty} (\boldsymbol {X} \| \boldsymbol {Y}) = \log \max _ {x \in \mathcal {X}} \frac {\mathbb {P} [ \boldsymbol {X} = x ]}{\mathbb {P} [ \boldsymbol {Y} = x ]}.
$$

δ-approximate max-divergence is defined as

$$
D _ {\infty} ^ {\delta} (\boldsymbol {X} \| \boldsymbol {Y}) = \log \max _ {\mathcal {O} \subseteq \mathcal {X}, \mathbb {P} [ \boldsymbol {X} \in \mathcal {O} ] > \delta} \frac {\mathbb {P} [ \boldsymbol {X} \in \mathcal {O} ] - \delta}{\mathbb {P} [ \boldsymbol {Y} \in \mathcal {O} ]}.
$$

We say that a real-valued function over datasets $f : \mathcal { X } ^ { n }  \mathbb { R }$ has sensitivity c for all $i \in [ n ]$ and $x _ { 1 } , x _ { 2 } , \ldots , x _ { n } , x _ { i } ^ { \prime } \in \mathcal { X } , f ( x _ { 1 } , \ldots , x _ { i } , \ldots , x _ { n } ) - f ( x _ { 1 } , \ldots , x _ { i } ^ { \prime } , \ldots , x _ { n } ) \leq c$ . We review McDiarmid’s concentration inequality for functions of low-sensitivity.

Lemma 1 (McDiarmid’s inequality). Let $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ be independent random variables taking values in the set X. Further let $f : \mathcal { X } ^ { n }  \mathbb { R }$ be a function of sensitivity $c > 0$ . Then for all $\alpha > 0$ , and $\mu = \mathbb { E } \left[ f ( X _ { 1 } , \ldots , X _ { n } ) \right]$ ],

$$
\mathbb {P} \left[ f _ {\underline {{}}} (\boldsymbol {X} _ {1}, \dots , \boldsymbol {X} _ {n}) - \mu \geq \alpha \right] \leq \exp \left(\frac {- 2 \alpha^ {2}}{n \cdot c ^ {2}}\right).
$$

For a function $\phi : \mathcal { X }  \mathbb { R }$ and a dataset $S = ( x _ { 1 } , \ldots , x _ { n } ) $ , let $\begin{array} { r } { \mathcal { E } _ { S } [ \phi ] \doteq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi ( x _ { i } ) } \end{array}$ . Note that if the range of $\phi$ is in some interval of length α then $f ( S ) = { \mathcal { E } } _ { S } [ \phi ]$ has sensitivity $\alpha / n$ . For a distribution $\mathcal { P }$ over $\mathcal { X }$ and a function $\phi : \mathcal { X }  \mathbb { R }$ , let ${ \mathcal { P } } [ \phi ] \doteq \mathbb { E } _ { x \sim { \mathcal { P } } } [ \phi ( x ) ]$

## 2.1 Diferential Privacy

On an intuitive level, diferential privacy hides the data of any single individual. We are thus interested in pairs of datasets $S , S ^ { \prime }$ that difer in a single element, in which case we say S and $S ^ { \prime }$ are adjacent.

Definition 2. [DMNS06, DKM<sup>+</sup>06] A randomized algorithm A with domain X<sup>n</sup> for $n > 0$ is $( \varepsilon , \delta )$ diferentially private if for all pairs of datasets that difer in a single element S $, S ^ { \prime } \in \mathcal { X } ^ { n } \colon D _ { \infty } ^ { \delta } ( \mathcal { A } ( S ) \| \mathcal { A } ( S ^ { \prime } ) ) \le$ log(e<sup>ε</sup>). The case when $\delta = 0$ is sometimes referred to as pure diferential privacy, and in this case we may say simply that A is ε-diferentially private.

Diferential privacy is preserved under adaptive composition. Adaptive composition of algorithms is a sequential execution of algorithms on the same dataset in which an algorithm at step i can depend on the outputs of previous algorithms. More formally, let $\mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } , \ldots , \mathcal { A } _ { m }$ be a sequence of algorithms. Each algorithm $A _ { i }$ outputs a value in some range $\mathcal { \mathrm { V } } _ { i }$ and takes as an input dataset in $\mathcal { X } ^ { n }$ as well as a value in $\bar { y } _ { i - 1 } \doteq y _ { 1 } \times \ldots \times y _ { i - 1 }$ . Adaptive composition of these algorithm is the algorithm that takes as an input a dataset $S \in \mathcal { X } ^ { n }$ and executes $\mathcal { A } _ { 1 } \to \mathcal { A } _ { 2 } \to \cdot \cdot \cdot \to \mathcal { A } _ { m }$ sequentially with the input to $A _ { i }$ being S and the outputs $y _ { 1 } , \ldots , y _ { i - 1 }$ of $\mathcal { A } _ { 1 } , \dotsc , \mathcal { A } _ { i - 1 }$ . Such composition captures the common practice in data analysis of using the outcomes of previous analyses (that is $y _ { 1 } , \dotsc , y _ { i - 1 } )$ to select an algorithm that is executed on $S$

For an algorithm that in addition to a dataset has other input we say that it is $( \varepsilon , \delta )$ -diferentially private if it is $( \varepsilon , \delta )$ -diferentially private for every setting of additional parameters. The basic property of adaptive composition of diferentially private algorithms is the following result $\left( e . g . [ \mathrm { D L 0 9 } ] \right)$ :

Theorem 3. Let $\mathcal { A } _ { i } : \mathcal { X } ^ { n } \times \mathcal { Y } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { Y } _ { i - 1 }  \mathcal { Y } _ { i }$ be an $( \varepsilon _ { i } , \delta _ { i } )$ -diferentially private algorithm for $i \in [ m ]$ Then the algorithm $B : \mathcal { X } ^ { n }  \mathcal { Y } _ { m }$ obtained by composing $\boldsymbol { A } _ { i } ^ { \prime } \boldsymbol { s }$ adaptively is $\begin{array} { r } { \big ( \sum _ { i = 1 } ^ { m } \varepsilon _ { i } , \sum _ { i = 1 } ^ { m } \delta _ { i } \big ) \ – d i f f e r e n t i a l l y } \end{array}$ private.

A more sophisticated argument yields significant improvement when $\varepsilon < 1 \ ( e . g . [ \mathrm { D R 1 } 4 ] )$

Theorem 4. For all $\varepsilon , \delta , \delta ^ { \prime } \geq 0$ , the adaptive composition of m arbitrary (ε, δ)-diferentially private algorithms is $( \varepsilon ^ { \prime } , m \delta + \delta ^ { \prime } )  – d i f f e r e n t i a l l y$ private, where

$$
\varepsilon^ {\prime} = \sqrt {2 m \ln (1 / \delta^ {\prime})} \cdot \varepsilon + m \varepsilon (e ^ {\varepsilon} - 1).
$$

Another property of diferential privacy important for our applications is preservation of its guarantee under post-processing (e.g.[DR14, Prop. 2.1]):

Lemma 5. If A is an (, δ)-diferentially private algorithm with domain $\mathcal { X } ^ { n }$ and range $\mathcal { V } _ { i }$ , and $\boldsymbol { B }$ is any, possibly randomized, algorithm with domain $\mathcal { V }$ and range $\mathcal { { V } } ^ { \prime \prime }$ , then the algorithm $B \circ A$ with domain $\mathcal { X } ^ { n }$ and range $\mathcal { V } ^ { \prime }$ is also (, δ)-diferentially private.

## 2.2 Generalization via Diferential Privacy

Generalization in special cases of our general adaptive analysis setting can be obtained directly from results in $\mathrm { [ D F H ^ { + } 1 4 ] }$ and composition properties of diferentially private algorithms. For the case of pure diferentiall private algorithms with general outputs over i.i.d. datasets, in $\mathrm { [ D F H ^ { + } 1 4 ] }$ we prove the following result.

Theorem 6. Let A be an ε-diferentially private algorithm with range Y and let S be a random variable drawn from a distribution ${ \mathcal { P } } ^ { n }$ over $\mathcal { X } ^ { n }$ . Let $\pmb { Y } = \pmb { \mathcal { A } } ( \pmb { S } )$ be the corresponding output distribution. Assume that for each element $y \in \mathcal { V }$ there is a subset $R ( y ) \subseteq \mathcal X ^ { n }$ so that ma $\mathfrak { c } _ { y \in \mathcal { V } } \mathbb { P } [ S \in R ( y ) ] \le \beta$ . Then, $\begin{array} { r } { f o r \varepsilon \le \sqrt { \frac { \ln ( 1 / \beta ) } { 2 n } } } \end{array}$ we have $\mathbb { P } [ S \in R ( { \pmb Y } ) ] \le 3 \sqrt { \beta }$

An immediate corollary of Thm. 6 together with Lemma 1 is that diferentially private algorithms that output low-sensitivity functions generalize.

Corollary 7. Let A be an algorithm that outputs a c-sensitive function $f : \mathcal { X } ^ { n }  \mathbb { R }$ . Let S be a random dataset chosen according to distribution ${ \mathcal { P } } ^ { n }$ over $\mathcal { X } ^ { n }$ and let $\pmb { f } = \mathcal { A } ( \pmb { S } )$ ). If A is $\tau / ( c n )$ -diferentially private then $\begin{array} { r } { \mathbb { P } [ \pmb { f } ( \pmb { S } ) - \pmb { \mathcal { P } } ^ { n } [ \pmb { f } ] \geq \tau ] \leq 3 \exp \left( - \tau ^ { 2 } / ( c ^ { 2 } n ) \right) } \end{array}$

By Theorem 3, pure diferential privacy composes adaptively. Therefore, if in a sequence of algorithms $\mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } , \ldots , \mathcal { A } _ { m }$ algorithm $\mathbf { \mathcal { A } } _ { i }$ is $\varepsilon _ { i } – \mathrm { d i f f e r e n t i a l l y }$ private for all $i \leq m - 1$ then composition of the first $i - 1$ algorithms is $\varepsilon _ { i - 1 } ^ { \prime } \mathrm { - d i f f e r e n t i a l l y }$ private for $\begin{array} { r } { \varepsilon _ { i - 1 } ^ { \prime } = \left( \sum _ { j = 1 } ^ { i - 1 } \varepsilon _ { j } \right) } \end{array}$ . Theorem 6 can be applied to preserve the generalization guarantees of the last algorithm $A _ { m }$ (that does not need to be diferentially private). For example, assume that for every fixed setting of $\bar { y } _ { m - 1 } , A _ { m }$ has the property that it outputs a hypothesis function h such that, $\begin{array} { r } { \mathbb { P } [ \mathcal { E } _ { S } [ L ( h ) ] - \mathcal { P } [ L ( h ) ] \ge \tau ] \le e ^ { - n \tau ^ { 2 } / d } } \end{array}$ , for some notion of dimension d and a real-valued loss function L. Generalization bounds of this type follow from uniform convergence arguments based on various notions of complexity of hypotheses classes such as VC dimension, covering numbers, fat-shattering dimension and Rademacher complexity (see [SSBD14] for examples). Note that, for diferent settings of $\bar { y } _ { m - 1 }$ diferent sets of hypotheses and generalization techniques might be used. We define $R ( \bar { y } _ { m - 1 } )$ be all datasets S for which $\mathscr { A } _ { m } ( S , \bar { y } _ { m - 1 } )$ outputs h such that $\mathcal { E } _ { S } [ L ( h ) ] - \mathcal { P } [ L ( h ) ] \ge \tau$ . Now if $\varepsilon _ { m - 1 } ^ { \prime } \leq \sqrt { \tau ^ { 2 } / ( 2 d ) }$ , then even for the hypothesis output in the adaptive execution of $A _ { m }$ on a random i.i.d. dataset S (denoted by h) we have $\begin{array} { r } { \mathbb { P } \left[ \mathscr { E } _ { S } [ L ( \pmb { h } ) ] - \mathscr { P } [ L ( \pmb { h } ) ] \geq \tau \right] \leq 3 e ^ { - \tau ^ { 2 } n / ( 2 d ) } } \end{array}$

For approximate $( \varepsilon , \delta )$ -diferential privacy, strong preservation of generalization results are currently known only for algorithms that output a function over X of bounded range (for simplicity we use range $[ 0 , 1 ] )$ $[ \mathrm { D F H ^ { + } 1 4 }$ , NS15]. The following result was proved by Nissim and Stemmer [NS15] (a weaker statement is also given in $[ \mathrm { D F H ^ { + } 1 4 }$ , Thm. 10]).

Theorem 8. Let A be an (ε, δ)-diferentially private algorithm that outputs a function from $\mathcal { X }$ to $[ 0 , 1 ]$ . For a random variable S distributed according to ${ \mathcal { P } } ^ { n }$ we let $\phi = \mathcal { A } ( S )$ . Then for $n \geq 2 \ln ( 8 / \delta ) / \varepsilon ^ { 2 }$

$$
\mathbb {P} \left[ | \mathcal {P} [ \phi ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi ] | \geq 1 3 \varepsilon \right] \leq \frac {2 \delta}{\varepsilon} \ln \left(\frac {2}{\varepsilon}\right).
$$

Many learning algorithms output a hypothesis function that aims to minimize some bounded loss function $L$ as the final output. If algorithms used in all steps of the adaptive data analysis are diferentially private and the last step (that is, $A _ { m } )$ outputs a hypothesis $h ,$ then generalization bounds for the loss of h are implied directly by Theorem 8. We remark that this application is diferent from the example for pure diferential privacy above since there we showed preservation of generalization guarantees of arbitrarily complex learning algorithm $A _ { m }$ which need not be diferentially private. In Section 4 we give an application of Theorem 8 to the reusable holdout problem.

## 2.3 Generalization via Description Length

Let $\mathcal { A } : \mathcal { X } ^ { n }  \mathcal { Y }$ and $B : \mathcal { X } ^ { n } \times \mathcal { Y }  \mathcal { Y } ^ { \prime }$ be two algorithms. We now give a simple application of bounds on the size of $\mathcal { V } \ ( \mathrm { o r }$ , equivalently, the description length of A’s output) to preserving generalization of B. Here generalization can actually refer to any valid or desirable output of B for a given given dataset S and input $y \in \mathcal { V }$ . Specifically we will use a set $R ( y ) \subseteq \mathcal X ^ { n }$ to denote all datasets for which the output of B on $y$ and S is “bad” $( e . g .$ overfits). Using a simple union bound we show that the probability (over a random choice of a dataset) of such bad outcome can be bounded.

Theorem 9. Let $\mathcal { A } : \mathcal { X } ^ { n }  \mathcal { Y }$ be an algorithm and let S be a random dataset over $\mathcal { X } ^ { n }$ . Assume that $R : \mathcal { V } \to 2 ^ { \mathcal { X } ^ { n } }$ is such that for every $y \in \mathcal { Y } , \mathbb { P } [ S \in R ( y ) ] \le \beta$ . Then $\mathbb { P } [ { \pmb S } \in \boldsymbol { R } ( \boldsymbol { \mathcal { A } } ( \pmb { S } ) ) ] \le | \mathcal { V } | \cdot \beta$

Proof.

$$
\mathbb {P} [ \boldsymbol {S} \in R (\mathcal {A} (\boldsymbol {S})) ] \leq \sum_ {y \in \mathcal {Y}} \mathbb {P} [ \boldsymbol {S} \in R (y) ] \leq | \mathcal {Y} | \cdot \beta .
$$

The case of two algorithms implies the general case since description length composes (adaptively). Namely, let $\mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } , \ldots , \mathcal { A } _ { m }$ be a sequence of algorithms such that each algorithm $\mathbf { \mathcal { A } } _ { i }$ outputs a value in some range $\mathcal { \mathrm { V } } _ { i }$ and takes as an input dataset in $\mathcal { X } ^ { n }$ as well as a value in $\bar { \mathcal { D } } _ { i - 1 }$ . Then for every i, we can view the execution of $\mathcal { A } _ { 1 }$ through $\mathcal { A } _ { i - 1 }$ as the first algorithm $\bar { \mathcal { A } } _ { i - 1 }$ with an output in $\bar { \mathcal { D } } _ { i - 1 }$ and $\mathbf { \mathcal { A } } _ { i }$ as the second algorithm. Theorem 9 implies that if for every setting of $\bar { y } _ { i - 1 } = y _ { 1 } , \dots , y _ { i - 1 } \in \bar { \mathcal { D } } _ { i - 1 } , R ( \bar { y } _ { i - 1 } ) \subseteq \mathcal { X } ^ { n }$ satisfies that $\mathbb { P } [ S \in R ( \bar { y } _ { i - 1 } ) ] \le \beta _ { i }$ then

$$
\mathbb {P} [ \boldsymbol {S} \in R (\bar {\mathcal {A}} _ {i - 1} (\boldsymbol {S})) ] \leq | \bar {\mathcal {Y}} _ {i - 1} | \cdot \beta_ {i} = \prod_ {j = 1} ^ {i - 1} | \mathcal {Y} _ {j} | \cdot \beta_ {i}.
$$

In Section A we describe a generalization of description length bounds to randomized algorithms and show that it possesses the same properties.

## 3 Max-Information

Consider two algorithms $\mathcal { A } : \mathcal { X } ^ { n }  \mathcal { Y }$ and $B : { \mathcal { X } } ^ { n } \times { \mathcal { Y } }  { \mathcal { Y } } ^ { \prime }$ that are composed adaptively and assume that for every fixed input $y \in \mathcal { V }$ , B generalizes for all but fraction β of datasets. Here we are speaking of generalization informally: our definitions will support any property of input $y \in \mathcal { V }$ and dataset S. Intuitively, to preserve generalization of B we want to make sure that the output of A does not reveal too much information about the dataset S. We demonstrate that this intuition can be captured via a notion of max-information and its relaxation approximate max-information.

For two random variables X and Y we use $X \times Y$ to denote the random variable obtained by drawing X and Y independently from their probability distributions.

Definition 10. Let X and Y be jointly distributed random variables. The max-information between X and Y, denoted $I _ { \infty } ( X ; Y )$ , is the minimal value of k such that for every x in the support of X and y in the support of Y we have $\mathbb { P } [ X = x \mid Y = y ] \leq 2 ^ { k } \mathbb { P } [ X = x ]$ . Alternatively, $I _ { \infty } ( X ; Y ) = D _ { \infty } ( ( X , Y ) \| X \times Y )$ The β-approximate max-information is defined as $I _ { \infty } ^ { \beta } ( \dot { \bf { X } } ; { \bf { Y } } ) = D _ { \infty } ^ { \beta } ( ( { \bf { X } } , { \bf { Y } } ) \| { \bf { X } } \times { \bf { Y } } )$

It follows immediately from Bayes’ rule that for all $\beta \ge 0 , I _ { \infty } ^ { \beta } ( X ; Y ) = I _ { \infty } ^ { \beta } ( Y ; X )$ . Further, $I _ { \infty } ( X ; Y ) \le k$ if and only if for all x in the support of $X , D _ { \infty } ( Y \mid X = x \parallel Y ) \le k$ . Clearly, max-information upper bounds the classical notion of mutual information: $I _ { \infty } ( X ; Y ) \ge I ( X ; Y )$

In our use (X, Y ) is going to be a joint distribution $( S , { \mathcal { A } } ( S ) )$ ), where S is a random n-element dataset and $\mathcal { A }$ is a (possibly randomized) algorithm taking a dataset as an input. If the output of an algorithm on any distribution S has low approximate max-information then we say that the algorithm has low max-information. More formally:

Definition 11. We say that an algorithm A has β-approximate max-information of k if for every distribution $s$ over n-element datasets, $I _ { \infty } ^ { \beta } ( S ; { \mathcal { A } } ( S ) ) \leq k$ , where $_ { s }$ is a dataset chosen randomly according to $s$ . We denote this by $I _ { \infty } ^ { \beta } ( { \mathcal { A } } , n ) \leq k$

An alternative way to define the (pure) max-information of an algorithm is using the maximum of the infinity divergence between distributions on two diferent inputs.

Lemma 12. Let A be an algorithm with domain $\mathcal { X } ^ { n }$ and range Y. Then $\begin{array} { r } { I _ { \infty } ( A , n ) = \operatorname* { m a x } _ { S , S ^ { \prime } \in \mathcal { X } ^ { n } } D _ { \infty } ( A ( S ) \| A ( S ^ { \prime } ) ) } \end{array}$

Proof. For the first direction let $k = \mathrm { m a x } _ { S , S ^ { \prime } \in \mathcal { X } ^ { n } } D _ { \infty } ( A ( S ) \| A ( S ^ { \prime } ) )$ . Let S be any random variable over n-element input datasets for A and let Y be the corresponding output distribution $\pmb { Y } = \pmb { \mathcal { A } } ( \pmb { S } )$ . We will argue that $I _ { \infty } ( Y ; S ) \le k _ { \mathrm { : } }$ , that $I _ { \infty } ( S ; Y ) \le k$ follows immediately from the Bayes’ rule. For every $y \in \mathcal { V }$ , there must exist a dataset $S _ { y }$ such that $\mathbb { P } [ Y = y \mid S = S _ { y } ] \subseteq \mathbb { P } [ Y = y ]$ . Now, by our assumption, for every $S _ { ; }$ $\mathbb { P } [ \pmb { Y } = y \ | \ S = S ] \le 2 ^ { k } \cdot \mathbb { P } [ \pmb { Y } = y \ | \ S = S _ { y } ]$ . We can conclude that for every S and every $y ,$ it holds that $\mathbb { P } [ Y = y ~ \vert ~ S = S ] \le 2 ^ { k } \mathbb { P } [ Y = y ]$ . This yields $I _ { \infty } ( Y ; S ) \le k .$

For the other direction let $k = I _ { \infty } ( \mathcal { A } , n )$ , let $S , S ^ { \prime } \in \mathcal { X } ^ { n }$ and $y \in \mathcal { V }$ . For $\alpha \in ( 0 , 1 )$ , let $_ { s }$ be the random variable equal to S with probability α and to $S ^ { \prime }$ with probability $1 - \alpha$ and let $\pmb { Y } = \pmb { \mathcal { A } } ( \pmb { S } )$ . By our assumption, $I _ { \infty } ( Y ; S ) = I _ { \infty } ( S ; Y ) \le k$ . This gives

$$
\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] \leq 2 ^ {k} \mathbb {P} [ \boldsymbol {Y} = y ] \leq 2 ^ {k} (\alpha \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] + (1 - \alpha) \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ^ {\prime} ])
$$

and implies

$$
\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] \leq \frac {2 ^ {k} (1 - \alpha)}{1 - 2 ^ {k} \alpha} \cdot \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ^ {\prime} ].
$$

This holds for every $\alpha > 0$ and therefore

$$
\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] \leq 2 ^ {k} \cdot \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ^ {\prime} ].
$$

Using this inequality for every $y \in \mathcal { V }$ we obtain $D _ { \infty } ( \boldsymbol { \mathcal { A } } ( \boldsymbol { S } ) \| \boldsymbol { \mathcal { A } } ( \boldsymbol { S } ^ { \prime } ) ) \le k$

Generalization via max-information: An immediate corollary of our definition of approximate maxinformation is that it controls the probability of “bad events” that can happen as a result of the dependence of $\boldsymbol { \mathcal { A } } ( \boldsymbol { S } )$ on S.

Theorem 13. Let S be a random dataset in $\mathcal { X } ^ { n }$ and A be an algorithm with range Y such that for some $\beta \geq 0 , I _ { \infty } ^ { \beta } ( S ; \mathcal { A } ( S ) ) = k$ . Then for any event ${ \mathcal { O } } \subseteq { \mathcal { X } } ^ { n } \times { \mathcal { Y } }$ ，

$$
\mathbb {P} [ (\boldsymbol {S}, \mathcal {A} (\boldsymbol {S})) \in \mathcal {O} ] \leq 2 ^ {k} \cdot \mathbb {P} [ \boldsymbol {S} \times \mathcal {A} (\boldsymbol {S}) \in \mathcal {O} ] + \beta .
$$

In particular, $\begin{array} { r } { \operatorname { \mathbb { P } } [ ( S , A ( S ) ) \in \mathcal { O } ] \leq 2 ^ { k } \cdot \operatorname* { m a x } _ { y \in \mathcal { V } } \operatorname { \mathbb { P } } [ ( S , y ) \in \mathcal { O } ] + \beta . } \end{array}$

We remark that mutual information between $\pmb { S }$ and $\boldsymbol { \mathcal { A } } ( \boldsymbol { S } )$ would not sufice for ensuring that bad events happen with tiny probability. For example mutual information of k allows $\mathbb { P } [ ( S , \mathcal { A } ( S ) ) \in \mathcal { O } ]$ to be as high as $k / ( 2 \log ( 1 / \delta ) )$ ), where $\delta = \mathbb { P } [ S \times \mathcal { A } ( S ) \in \mathcal { O } ]$

Composition of max-information: Approximate max-information satisfies the following adaptive composition property:

Lemma 14. Let $\mathcal { A } : \mathcal { X } ^ { n }  \mathcal { Y }$ be an algorithm such that $I _ { \infty } ^ { \beta _ { 1 } } ( { \mathcal { A } } , n ) \leq k _ { 1 }$ , and let $B : \mathcal { X } ^ { n } \times \mathcal { Y }  \mathcal { Z }$ be an algorithm such that for every $y \in \mathcal { V } , \ : B ( \cdot , y )$ has β<sub>2</sub>-approximate max-information $k _ { 2 }$ . Let ${ \mathcal { C } } : { \mathcal { X } } ^ { n }  { \mathcal { Z } }$ be defined such that $\mathcal { C } ( S ) = B ( S , \mathcal { A } ( S ) )$ ). Then $I _ { \infty } ^ { \beta _ { 1 } + \beta _ { 2 } } ( { \mathcal { C } } , n ) \leq k _ { 1 } + k _ { 2 }$

Proof. Let D be a distribution over $\mathcal { X } ^ { n }$ and S be a random dataset sampled from D. By hypothesis, $I _ { \infty } ^ { \beta _ { 1 } } ( S ; { \mathcal { A } } ( S ) ) \leq k _ { 1 }$ . Expanding out the definition for all ${ \mathcal { O } } \subseteq { \mathcal { X } } ^ { n } \times { \mathcal { Y } }$ :

$$
\mathbb {P} [ (\boldsymbol {S}, \mathcal {A} (\boldsymbol {S})) \in \mathcal {O} ] \leq 2 ^ {k _ {1}} \cdot \mathbb {P} [ \boldsymbol {S} \times \mathcal {A} (\boldsymbol {S}) \in \mathcal {O} ] + \beta_ {1}.
$$

We also have for all ${ \mathcal { Q } } \subseteq { \mathcal { X } } ^ { n } \times { \mathcal { Z } }$ and for all $y \in \mathcal { V }$

$$
\mathbb {P} [ (\boldsymbol {S}, \mathcal {B} (\boldsymbol {S}, y)) \in \mathcal {Q} ] \leq 2 ^ {k _ {2}} \cdot \mathbb {P} [ \boldsymbol {S} \times \mathcal {B} (S, y) \in \mathcal {Q} ] + \beta_ {2}.
$$

For every $\mathcal { O } \subseteq \mathcal { X } ^ { n } \times \mathcal { Y }$ , define

$$
\mu (\mathcal {O}) = \left(\mathbb {P} [ (\boldsymbol {S}, \mathcal {A} (\boldsymbol {S})) \in \mathcal {O} ] - 2 ^ {k _ {1}} \cdot \mathbb {P} [ \boldsymbol {S} \times \mathcal {A} (\boldsymbol {S}) \in \mathcal {O} ]\right) _ {+}.
$$

Observe that $\mu ( \mathcal { O } ) \leq \beta _ { 1 }$ for all ${ \mathcal { O } } \subseteq { \mathcal { X } } ^ { n } \times { \mathcal { Y } }$ . For any event ${ \mathcal { Q } } \subseteq { \mathcal { X } } ^ { n } \times { \mathcal { Z } }$ , we have:

$$
\begin{array}{l l} & \mathbb {P} [ (\boldsymbol {S}, \mathcal {C} (\boldsymbol {S})) \in \mathcal {Q} ] \\ = & \mathbb {P} [ (\boldsymbol {S}, \mathcal {B} (\boldsymbol {S}, \mathcal {A} (\boldsymbol {S}))) \in \mathcal {Q} ] \\ = & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \mathbb {P} [ (S, \mathcal {B} (S, y)) \in \mathcal {Q} ] \cdot \mathbb {P} [ \boldsymbol {S} = S, \mathcal {A} (\boldsymbol {S}) = y ] \\ \leq & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \min \left(\left(2 ^ {k _ {2}} \cdot \mathbb {P} [ S \times \mathcal {B} (S, y) \in \mathcal {Q} ] + \beta_ {2}\right), 1\right) \cdot \mathbb {P} [ \boldsymbol {S} = S, \mathcal {A} (\boldsymbol {S}) = y ] \\ \leq & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \left(\min \left(2 ^ {k _ {2}} \cdot \mathbb {P} [ S \times \mathcal {B} (S, y) \in \mathcal {Q} ], 1\right) + \beta_ {2}\right) \cdot \mathbb {P} [ \boldsymbol {S} = S, \mathcal {A} (\boldsymbol {S}) = y ] \\ \leq & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \min \left(2 ^ {k _ {2}} \cdot \mathbb {P} [ S \times \mathcal {B} (S, y) \in \mathcal {Q} ], 1\right) \cdot \mathbb {P} [ \boldsymbol {S} = S, \mathcal {A} (\boldsymbol {S}) = y ] + \beta_ {2} \\ \leq & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \min \left(2 ^ {k _ {2}} \cdot \mathbb {P} [ S \times \mathcal {B} (S, y) \in \mathcal {Q} ], 1\right) \cdot (2 ^ {k _ {1}} \cdot \mathbb {P} [ \boldsymbol {S} = S ] \cdot \mathbb {P} [ \mathcal {A} (\boldsymbol {S}) = y ] + \mu (S, y)) + \beta_ {2} \\ \leq & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \min \left(2 ^ {k _ {2}} \cdot \mathbb {P} [ S \times \mathcal {B} (S, y) \in \mathcal {Q} ], 1\right) \cdot 2 ^ {k _ {1}} \cdot \mathbb {P} [ \boldsymbol {S} = S ] \cdot \mathbb {P} [ \mathcal {A} (\boldsymbol {S}) = y ] + \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \mu (S, y) + \beta_ {2} \\ \leq & \sum_ {S \in \mathcal {X} ^ {n}, y \in \mathcal {Y}} \min \left(2 ^ {k _ {2}} \cdot \mathbb {P} [ S \times \mathcal {B} (S, y) \in \mathcal {Q} ], 1\right) \cdot 2 ^ {k _ {1}}   -   {\mathbb P} [ {\boldsymbol S} = S ]   -   {\mathbb P} [ {\mathcal A} ({\boldsymbol S}) = y ] + {\beta_ {1}} + {\beta_ {2}} \\ \leq & 2 ^ {k _ {1} + k _ {2}}   -   {\left(\sum_ {S \in \mathcal {X} ^ {n}, y \in {\mathcal Y}} {\mathbb P} [ S   {\times}   {\mathcal B} (S, y)   {\in}   {\mathcal Q} ]   {\cdot}   {\mathbb P} [ {\boldsymbol S} = S ]   {\cdot}   {\mathbb P} [ {\mathcal A} ({\boldsymbol S}) = y ]\right)} + {\beta_ {1}} + {\beta_ {2}} \\ = & 2 ^ {k _ {1} + k _ {2}}   -   {\mathbb P} [ {\boldsymbol S}   {\times}   {\mathcal B} ({\boldsymbol S}, {\mathcal A} ({\boldsymbol S}))   {\in}   {\mathcal Q} ] + ({\beta_ {1}} + {\beta_ {2}}) . \end{array}
$$

Applying the definition of max-information, we see that equivalently, $I _ { \infty } ^ { \beta _ { 1 } + \beta _ { 2 } } ( S ; { \mathcal { C } } ( S ) ) \leq k _ { 1 } + k _ { 2 }$ , which is what we wanted. □

This lemma can be iteratively applied, which immediately yields the following adaptive composition theorem for max-information:

Theorem 15. Consider an arbitrary sequence of algorithms $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { k }$ with ranges $y _ { 1 } , \ldots , y _ { k }$ such that for all $i , \mathcal { A } _ { i } : \mathcal { X } ^ { n } \times \mathcal { Y } _ { 1 } \times . . . \times \mathcal { Y } _ { i - 1 } \to \mathcal { Y } _ { i }$ is such that $\mathcal { A } _ { i } ( \cdot , y _ { 1 } , \ldots , y _ { i - 1 } )$ has β<sub>i</sub>-approximate max-information $k _ { i }$ for all choices of $y _ { 1 } , \dots , y _ { i - 1 } \in { \mathcal { V } } _ { 1 } \times \dots \times { \mathcal { V } } _ { i - 1 }$ . Let the algorithm $B : \mathcal { X } ^ { n }  \mathcal { Y } _ { k }$ be defined as follows: $B ( S )$ :

1. Let $y _ { 1 } = A _ { 1 } ( S )$

2. For i = 2 to k: Let $y _ { i } = { A _ { i } } ( S , y _ { 1 } , \dots , y _ { i - 1 } )$

3. Output $y _ { k }$

Then B has $( \sum _ { i } \beta _ { i } )$ -approximate max-information $( \sum _ { i } { k _ { i } } )$

Post-processing of Max-information: Another useful property that (approximate) max-information shares with diferential privacy is preservation under post-processing. The simple proof of this lemma is identical to that for diferential privacy (Lemma 5) and hence is omitted.

Lemma 16. If A is an algorithm with domain $\mathcal { X } ^ { n }$ and range $\mathcal { V } _ { i }$ , and $\boldsymbol { B }$ is any, possibly randomized, algorithm with domain $\mathcal { V }$ and range $\mathcal { V } ^ { \prime \prime }$ , then the algorithm $B \circ A$ with domain $\mathcal { X } ^ { n }$ and range $\mathcal { V } ^ { \prime }$ satisfies: for every random variable $_ { s }$ over $\mathcal { X } ^ { n }$ and every $\displaystyle \beta \geq 0 , I _ { \infty } ^ { \beta } ( S ; B \circ {  \mathcal { A } } ( S ) ) \leq I _ { \infty } ^ { \beta } ( S ; {  \mathcal { A } } ( S ) )$ .

## 3.1 Bounds on Max-information

We now show that the basic approaches based on description length and (pure) diferential privacy are captured by approximate max-information.

## 3.1.1 Description Length

Description length k gives the following bound on max-information.

Theorem 17. Let A be a randomized algorithm taking as an input an n-element dataset and outputting a value in a finite set Y. Then for every $\beta > 0 , I _ { \infty } ^ { \beta } ( \mathcal { A } , n ) \leq \log ( | \mathcal { V } | / \beta )$

We will use the following simple property of approximate divergence $( e . g . [ \mathrm { D R 1 4 } ] )$ in the proof. For a random variable X over X we denote by $p ( X )$ the probability distribution associated with X.

Lemma 18. Let X and Y be two random variables over the same domain X. If

$$
\underset {x \sim p (\mathbf {X})} {\mathbb {P}} \left[ \frac {\mathbb {P} [ \mathbf {X} = x ]}{\mathbb {P} [ \mathbf {Y} = x ]} \geq 2 ^ {k} \right] \leq \beta
$$

then $D _ { \infty } ^ { \beta } ( \pmb { X } \| \pmb { Y } ) \le k$

Proof Thm. 17. Let S be any random variable over n-element input datasets and let Y be the corresponding output distribution $\pmb { Y } = \pmb { \mathcal { A } } ( \pmb { S } )$ . We prove that for every $\beta > 0 , I _ { \infty } ^ { \beta } ( S ; { \cal Y } ) \le \log ( | { \cal y } | / \beta )$

For $y \in \mathcal { V }$ we say that y is “bad” if exists S in the support of S such that

$$
\frac {\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ]}{\mathbb {P} [ \boldsymbol {Y} = y ]} \geq | \mathcal {Y} | / \beta .
$$

Let B denote the set of all $ { ^ { \circ } }  { \mathrm { b a d } } ^ { \prime \prime } \ y  { ^ { \prime } }  { \mathrm { s } }$ . From this definition we obtain that for a $\mathrm { { ^ { 4 9 } b a d } ^ { \prime \prime } } \ y , \mathbb { P } [ \boldsymbol { Y } = y ] \le \beta / | \boldsymbol { y } |$ and therefore $\mathbb { P } [ \pmb { Y } \in B ] \le \beta$ . Let $B = \mathcal { X } ^ { n } \times B$ . Then

$$
\mathbb {P} [ (\boldsymbol {S}, \boldsymbol {Y}) \in \mathcal {B} ] = \mathbb {P} [ \boldsymbol {Y} \in B ] \leq \beta .
$$

For every $( S , y ) \notin B$ we have that

$$
\mathbb {P} [ \boldsymbol {S} = S, \boldsymbol {Y} = y ] = \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] \cdot \mathbb {P} [ \boldsymbol {S} = S ] \leq \frac {| \mathcal {Y} |}{\beta} \cdot \mathbb {P} [ \boldsymbol {Y} = y ] \cdot \mathbb {P} [ \boldsymbol {S} = S ],
$$

and hence

$$
\underset {(S, y) \sim p (\boldsymbol {S}, \boldsymbol {Y})} {\mathbb {P}} \left[ \frac {\mathbb {P} [ \boldsymbol {S} = S , \boldsymbol {Y} = y ]}{\mathbb {P} [ \boldsymbol {S} = S ] \cdot \mathbb {P} [ \boldsymbol {Y} = y ]} \geq \frac {| \mathcal {Y} |}{\beta} \right] \leq \beta .
$$

This, by Lemma 18, gives that $I _ { \infty } ^ { \beta } ( S ; Y ) \le \log ( | y | / \beta )$

We note that Thms. 13 and 17 give a slightly weaker form of Thm. 9. Defining event $\mathcal { O } \doteq \{ ( S , y ) | S \in R ( y ) \}$ the assumptions of Thm. 9 imply that $\mathbb { P } [ S \times \mathcal { A } ( S ) \in \mathcal { O } ] \le \beta$ . For $\beta ^ { \prime } = \sqrt { | \mathcal { V } | \beta } .$ , by Thm. 17, we have that $I _ { \infty } ^ { \beta ^ { \prime } } ( S ; { \mathcal { A } } ( S ) ) \leq \log ( | \mathcal { V } | / \beta ^ { \prime } )$ . Now applying, Thm. 13 gives that $\mathbb { P } [ S \in R ( { \cal A } ( S ) ) ] \le | \mathcal { V } | / \beta ^ { \prime } \cdot \beta + \beta ^ { \prime } = 2 \sqrt { | \mathcal { V } | \beta }$

In Section A we introduce a closely related notion of randomized description length and show that it also provides an upper bound on approximate max-information. More interestingly, for this notion a form of the reverse bound can be proved: A bound on (approximate) max-information of an algorithm A implies a bound on the randomized description length of the output of a diferent algorithm with statistically indistinguishable from A output.

## 3.1.2 Diferential Privacy

We now show that pure diferential privacy implies a bound on max information. We start with a simple bound on max-information of diferentially private algorithms that applies to all distributions over datasets. In particular, it implies that the diferential privacy-based approach can be used beyond the i.i.d. setting in $\mathrm { [ D F H ^ { + } 1 4 ] }$

Theorem 19. Let A be an -diferentially private algorithm. Then $I _ { \infty } ( A , n ) \leq \log e \cdot \epsilon n$

Proof. Clearly, any two datasets S and $S ^ { \prime }$ difer in at most n elements. Therefore, for every y we have $\mathbb { P } [ \pmb { Y } = \pmb { y } \ | \ \pmb { S } = \pmb { S } ] \le e ^ { \epsilon n } \mathbb { P } [ \pmb { Y } = \pmb { y } \ | \ \pmb { S } = \pmb { S } ^ { \prime } ]$ (this is a direct implication of Definition 2 referred to as group privacy [DR14]), or equivalently, $D _ { \infty } ( A ( S ) \| A ( S ^ { \prime } ) ) \le \log e \cdot \epsilon n$ . By Lemma 3 we obtain the claim. □

Finally, we prove a stronger bound on approximate max-information for datasets consisting of i.i.d. samples using the technique from $\mathrm { [ D F H ^ { + } 1 4 ] }$ . This bound, together with Thm. 13, generalizes Thm. 6.

Theorem 20. Let A be an ε-diferentially private algorithm with range Y. For a distribution P over X, let $_ { s }$ be a random variable drawn from ${ \mathcal { P } } ^ { n }$ . Let $\pmb { Y } = \pmb { \mathcal { A } } ( \pmb { S } )$ denote the random variable output by A on input S. Then for any $\beta > 0 , I _ { \infty } ^ { \beta } ( S ; \mathcal { A } ( S ) ) \le \log e ( \varepsilon ^ { 2 } n / 2 + \varepsilon \sqrt { n \ln ( 2 / \beta ) / 2 } )$

Proof. Fix $y \in \mathcal { V }$ . We first observe that by Jensen’s inequality,

$$
\underset {S \sim \mathcal {P} ^ {n}} {\mathbb {E}} [ \ln (\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ]) ] \leq \ln \left(\underset {S \sim \mathcal {P} ^ {n}} {\mathbb {E}} [ \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] ]\right) = \ln (\mathbb {P} [ \boldsymbol {Y} = y ]).
$$

Further, by definition of diferential privacy, for two databases $S , S ^ { \prime }$ that difer in a single element,

$$
\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ] \leq e ^ {\varepsilon} \cdot \mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ^ {\prime} ].
$$

Now consider the function $\begin{array} { r } { g ( S ) = \ln \left( \frac { \mathbb { P } \left[ \mathbf { Y } = y ~ | ~ S = S \right] } { \mathbb { P } \left[ \mathbf { Y } = y \right] } \right) } \end{array}$ . By the properties above we have that $\mathbb { E } [ g ( S ) ] \leq$ $\ln ( \mathbb { P } [ Y = y ] ) - \ln ( \mathbb { P } [ Y = y ] ) = 0$ and $| g ( S ) \dot { - } g ( S ^ { \prime } ) | \leq \varepsilon .$ . This, by McDiarmid’s inequality (Lemma 1), implies that for any $t > 0$

$$
\mathbb {P} [ g (\boldsymbol {S}) \geq t ] \leq e ^ {- 2 t ^ {2} / (n \varepsilon^ {2})}.\tag{1}
$$

For an integer $i \geq 1$ , let $t _ { i } \doteq \varepsilon ^ { 2 } n / 2 + \varepsilon \sqrt { n \ln ( 2 ^ { i } / \beta ) / 2 }$ and let

$$
B _ {i} \doteq \left\{S \mid t _ {i} <   g (S) \leq t _ {i + 1} \right\}.
$$

Let

$$
B _ {y} \doteq \{S | g (S) > t _ {1} \} = \bigcup_ {i \geq 1} B _ {i}.
$$

By inequality (1), we have that for $i \geq 1$ ，

$$
\mathbb {P} [ g (\boldsymbol {S}) > t _ {i} ] \leq \exp \left(- 2 \left(\varepsilon \sqrt {n} / 2 + \sqrt {\ln (2 ^ {i} / \beta) / 2}\right) ^ {2}\right).
$$

By Bayes’ rule, for every $S \in B _ { i }$ ，

$$
\frac {\mathbb {P} [ \boldsymbol {S} = S \mid \boldsymbol {Y} = y ]}{\mathbb {P} [ \boldsymbol {S} = S ]} = \frac {\mathbb {P} [ \boldsymbol {Y} = y \mid \boldsymbol {S} = S ]}{\mathbb {P} [ \boldsymbol {Y} = y ]} = \exp (g (S)) \leq \exp (t _ {i + 1}).
$$

Therefore,

$$
\begin{array}{l} \mathbb {P} [ \boldsymbol {S} \in B _ {i} \mid \boldsymbol {Y} = y ] = \sum_ {S \in B _ {i}} \mathbb {P} [ \boldsymbol {S} = S \mid \boldsymbol {Y} = y ] \\ \qquad \leq \exp (t _ {i + i}) \cdot \sum_ {S \in B _ {i}} \mathbb {P} [ \boldsymbol {S} = S ] \\ \qquad \leq \exp (t _ {i + i}) \cdot \mathbb {P} [ g (\boldsymbol {S}) \geq t _ {i} ] \\ \qquad = \exp \left(\varepsilon^ {2} n / 2 + \varepsilon \sqrt {n \ln (2 ^ {i + 1} / \beta) / 2} - 2 \left(\varepsilon \sqrt {n} / 2 + \sqrt {\ln (2 ^ {i} / \beta) / 2}\right) ^ {2}\right) \\ \qquad \leq \exp \left(\varepsilon \sqrt {n / 2} \left(\sqrt {\ln (2 ^ {i + 1} / \beta)} - 2 \sqrt {\ln (2 ^ {i} / \beta)}\right) - \ln (2 ^ {i} / \beta)\right) \\ \qquad <   \exp (- \ln (2 ^ {i} / \beta)) = \beta / 2 ^ {i}. \end{array}
$$

An immediate implication of this is that

$$
\mathbb {P} [ \boldsymbol {S} \in B _ {y} \mid \boldsymbol {Y} = y ] = \sum_ {i} \mathbb {P} [ \boldsymbol {S} \in B _ {i} \mid \boldsymbol {Y} = y ] \leq \sum_ {i \geq 1} \beta / 2 ^ {i} \leq \beta .
$$

Let $\boldsymbol { { B } } = \{ ( S , y ) ~ | ~ \boldsymbol { { y } } \in \mathcal { V } , S \in B _ { y } \}$ . Then

$$
\mathbb {P} [ (\boldsymbol {S}, \boldsymbol {Y}) \in \mathcal {B} ] = \mathbb {P} [ (\boldsymbol {S}, \boldsymbol {Y}) \in B _ {\boldsymbol {Y}} ] \leq \beta .\tag{2}
$$

For every $( S , y ) \notin B$ we have that

$$
\mathbb {P} [ \boldsymbol {S} = S, \boldsymbol {Y} = y ] = \mathbb {P} [ \boldsymbol {S} = S \mid \boldsymbol {Y} = y ] \cdot \mathbb {P} [ \boldsymbol {Y} = y ] \leq \exp (t _ {1}) \cdot \mathbb {P} [ \boldsymbol {S} = S ] \cdot \mathbb {P} [ \boldsymbol {Y} = y ],
$$

and hence by $\mathrm { e q . } ( 2 )$ we get that

$$
\underset {(S, y) \sim p (\boldsymbol {S}, \boldsymbol {Y})} {\mathbb {P}} \left[ \frac {\mathbb {P} [ \boldsymbol {S} = S , \boldsymbol {Y} = y ]}{\mathbb {P} [ \boldsymbol {S} = S ] \cdot \mathbb {P} [ \boldsymbol {Y} = y ]} \geq \exp (t _ {1}) \right] \leq \beta .
$$

This, by Lemma 18, gives that

$$
I _ {\infty} ^ {\beta} (\boldsymbol {S}; \boldsymbol {Y}) \leq \log (\exp (t _ {1})) = \log e (\varepsilon^ {2} n / 2 + \varepsilon \sqrt {n \ln (2 / \beta) / 2}).
$$

Applications: We give two simple examples of using the bounds on max-information obtained from diferential privacy to preserve bounds on generalization error that follow from concentration of measure inequalities. Strong concentration of measure results are at the core of most generalization guarantees in machine learning. Let A be an algorithm that outputs a function $f : \mathcal { X } ^ { n }  \mathbb { R }$ of sensitivity c and define the “bad event” O<sub>τ</sub> is when the empirical estimate of $f$ is more than τ away from the expectation of $f ( S )$ for S distributed according to some distribution $\mathcal { D }$ over $\mathcal { X } ^ { n }$ . Namely,

$$
\mathcal {O} _ {\tau} = \left\{(S, f) \colon f (S) - \mathcal {D} [ f ] \geq \tau \right\},\tag{3}
$$

where $\mathcal { D } [ f ]$ denotes $\mathbb { E } _ { S \sim \mathcal { D } } [ f ( S ) ]$

By McDiarmid’s inequality (Lem. 1) we know that, if S is distributed according to ${ \mathcal { P } } ^ { n }$ then $\begin{array} { r } { \operatorname* { s u p } _ { f : \mathcal { X } ^ { n } \to \mathbb { R } } \mathbb { P } [ ( S , f ) \in } \end{array}$ $\mathcal { O } _ { \tau } ] \leq \exp ( - 2 \tau ^ { 2 } / ( c ^ { 2 } n ) )$ ). The simpler bound in Thm. 19 implies following corollary.

Corollary 21. Let A be an algorithm that outputs a c-sensitive function $f : \mathcal { X } ^ { n }  \mathbb { R }$ . Let S be a random dataset chosen according to distribution ${ \mathcal { P } } ^ { n }$ over $\mathcal { X } ^ { n }$ and let $\pmb { f } = \pmb { \mathcal { A } } ( \pmb { S } )$ . If for $\beta \geq 0$ and $\tau > 0$ $I _ { \infty } ^ { \beta } ( S ; \pmb { f } ) \le \log e \cdot \tau ^ { 2 } / c ^ { 2 }$ , then $\mathbb { P } [ \pmb { \mathscr { f } } ( \pmb { S } ) - \pmb { \mathscr { P } } ^ { n } [ \pmb { \mathscr { f } } ] \geq \tau ] \leq \exp \left( - \tau ^ { 2 } / ( c ^ { 2 } n ) \right) + \beta$ . In particular, $i f \mathcal { A } \ i s \ \tau ^ { 2 } / ( c ^ { 2 } n ^ { 2 } )$ diferentially private then $\begin{array} { r } { \mathbb { P } [ \pmb { f } ( \pmb { S } ) - \pmb { \mathcal { P } } ^ { n } [ \pmb { f } ] \geq \tau ] \leq \exp \left( - \tau ^ { 2 } / ( c ^ { 2 } n ) \right) } \end{array}$

Note that for $f ( S ) = { \mathcal { E } } _ { S } [ \phi ]$ , where $\phi : \mathcal { X } \to [ 0 , 1 ]$ this result requires $\varepsilon = \tau ^ { 2 }$ . The stronger bound allows to preserve concentration of measure even when $\varepsilon = \tau / ( c n )$ which corresponds to $\tau = \varepsilon$ when $f ( S ) = { \mathcal { E } } _ { S } [ \phi ]$

Corollary 22. Let A be an algorithm that outputs a c-sensitive function $f : \mathcal { X } ^ { n }  \mathbb { R }$ . Let S be a random dataset chosen according to distribution ${ \mathcal { P } } ^ { n }$ over $\mathcal { X } ^ { n }$ and let $\pmb { f } = \mathcal { A } ( \pmb { S } )$ ). If A is τ/(cn)-diferentially private then $\begin{array} { r } { \mathbb { P } [ \pmb { f } ( \pmb { S } ) - \pmb { \mathcal { P } } ^ { n } [ \pmb { f } ] \geq \tau ] \leq \exp \left( - 3 \tau ^ { 2 } / ( 4 c ^ { 2 } n ) \right) } \end{array}$

Proof. We apply Theorem 20 with $\beta = 2 \exp { \left( - \tau ^ { 2 } / ( c ^ { 2 } n ) \right) }$ to obtain that

$$
I _ {\infty} ^ {\beta} (\boldsymbol {S}; \boldsymbol {f}) \leq \log e \cdot (\varepsilon^ {2} n / 2 + \varepsilon \sqrt {n \ln (2 / \beta) / 2})) \leq \log e \cdot (\tau^ {2} / (c ^ {2} n) / 2 + \tau^ {2} / (c ^ {2} n) / \sqrt {2}).
$$

Applying Thm. 13 to McDiarmid’s inequality we obtain that

$$
\begin{array}{r l} & {\mathbb {P} [ \boldsymbol {f} (\boldsymbol {S}) - \mathcal {P} ^ {n} [ \boldsymbol {f} ] \geq \tau ] \leq \exp \left((1 / 2 + 1 / \sqrt {2}) \tau^ {2} / (c ^ {2} n)\right) \cdot \exp \left(- 2 \tau^ {2} / (c ^ {2} n)\right) + 2 \exp \left(- \tau^ {2} / (c ^ {2} n)\right)} \\ & {\qquad \leq \exp \left(- 3 \tau^ {2} / (4 c ^ {2} n)\right),} \end{array}
$$

where the last inequality holds when $\tau ^ { 2 } / ( c ^ { 2 } n )$ is larger than a fixed constant.

## 4 Reusable Holdout

We describe two simple algorithms that enable validation of analyst’s queries in the adaptive setting.

## 4.1 Thresholdout

Our first algorithm Thresholdout follows the approach in $\mathrm { [ D F H ^ { + } 1 4 ] }$ where diferentially private algorithms are used to answer adaptively chosen statistical queries. This approach can also be applied to any low-sensitivity functions <sup>2</sup> of the dataset but for simplicity we present the results for statistical queries. Here we address an easier problem in which the analyst’s queries only need to be answered when they overfit. Also, unlike in $\mathrm { [ D F H ^ { + } 1 4 ] }$ , the analyst has full access to the training set and the holdout algorithm only prevents overfitting to holdout dataset. As a result, unlike in the general query answering setting, our algorithm can eficiently validate an exponential in n number of queries as long as a relatively small number of them overfit.

Thresholdout is given access to the training dataset $S _ { t }$ and holdout dataset $S _ { h }$ and a budget limit B. It allows any query of the form φ : $: \mathcal { X }  [ 0 , 1 ]$ and its goal is to provide an estimate of ${ \mathcal { P } } [ \phi ]$ . To achieve this the algorithm gives an estimate of $\mathcal { E } _ { S _ { h } } [ \phi ]$ in a way that prevents overfitting of functions generated by the analyst to the holdout set. In other words, responses of Thresholdout are designed to ensure that, with high probability, E [φ] is close to ${ \mathcal { P } } [ \phi ]$ and hence an estimate of $\mathcal { E } _ { S _ { h } } [ \phi ]$ gives an estimate of the true expectation ${ \mathcal { P } } [ \phi ]$ . Given a function $\phi ,$ Thresholdout first checks if the diference between the average value of $\phi$ on the training set $S _ { t }$ (or $\mathcal { E } _ { S _ { t } } [ \phi ] ,$ and the average value of $\phi$ on the holdout set $S _ { h } \ \left( \mathrm { o r } \ \mathcal { E } _ { S _ { h } } [ \phi ] \right)$ is below a certain threshold $T + \eta$ . Here, $T$ is a fixed number such as 0.01 and $\eta$ is $\mathrm { a }$ Laplace noise variable whose standard deviation needs to be chosen depending on the desired guarantees (The Laplace distribution is a symmetric exponential distribution.) If the diference is below the threshold, then the algorithm returns $\mathcal { E } _ { S _ { t } } [ \phi ]$ . If the diference is above the threshold, then the algorithm returns $\mathcal { E } _ { S _ { h } } [ \phi ] + \xi$ for another Laplacian noise variable $\xi .$ Each time the diference is above threshold the “overfitting” budget B is reduced by one. Once it is exhausted, Thresholdout stops answering queries. In Fig. 1 we provide the pseudocode of Thresholdout.

We now establish the formal generalization guarantees that Thresholdout enjoys. As the first step we state what privacy parameters are achieved by Thresholdout.

Lemma 23. Thresholdout satisfies $( 2 B / ( \sigma n ) , 0 ) – d i f f$ erential privacy. Thresholdout also satisfies $( \sqrt { 3 2 B \ln ( 2 / \delta ) } / ( \sigma n ) , \delta )$ diferential privacy for any $\delta > 0$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm Thresholdout Input: Training set  $S_{t}$ , holdout set  $S_{h}$ , threshold T, noise rate  $\sigma$ , budget B

1. sample  $\gamma \sim \text{Lap}(2 \cdot \sigma)$ ;  $\hat{T} \leftarrow T + \gamma$

2. For each query  $\phi$  do

(a) if B &lt; 1 output “⊥”

(b) else

i. sample  $\eta \sim \text{Lap}(4 \cdot \sigma)$

ii. if  $|\mathcal{E}_{S_{h}}[\phi] - \mathcal{E}_{S_{t}}[\phi]| &gt; \hat{T} + \eta$

A. sample  $\xi \sim \text{Lap}(\sigma)$ ,  $\gamma \sim \text{Lap}(2 \cdot \sigma)$

B.  $B \leftarrow B - 1$  and  $\hat{T} \leftarrow T + \gamma$

C. output  $E_{S_{h}}[\phi] + \xi$

iii. else output  $E_{S_{t}}[\phi]$ .
</div>

Figure 1: The details of Thresholdout algorithm

Proof. Thresholdout is an instantiation of a basic tool from diferential privacy, the “Sparse Vector Algorithm” ([DR14, Algorithm 2]), together with the Laplace mechanism ([DR14, Defn. 3.3]). The sparse vector algorithm takes as input a sequence of c sensitivity $1 / n$ queries<sup>3</sup> (here $c = B$ , the budget), and for each query, attempts to determine whether the value of the query, evaluated on the private dataset, is above a fixed threshold T or below it. In our instantiation, the holdout set $S _ { h }$ is the private data set, and each function φ corresponds to the following query evaluated on $S _ { h } \colon f _ { \phi } ( S _ { h } ) : = | \mathcal { E } _ { S _ { h } } [ \phi ] - \mathcal { E } _ { S _ { t } } [ \phi ] |$ . (Note that the training set $S _ { t }$ is viewed as part of the definition of the query). Thresholdout then is equivalent to the following procedure: we run the sparse vector algorithm [DR14, Algorithm 2] with $c = B$ , queries $f _ { \phi }$ for each function φ, and noise rate $2 \sigma$ . Whenever an above-threshold query is reported by the sparse vector algorithm, we release its value using the Laplace mechanism [DR14, Defn. 3.3] with noise rate σ (this is what occurs every time Thresholdout answers by outputting $\xi _ { S _ { h } } [ \phi ] + \xi )$ . By the privacy guarantee of the sparse vector algorithm ([DR14, Thm. 3.25]), the sparse vector portion of Thresholdout satisfies $( B / ( \sigma n ) , 0 )$ -diferential privacy, and simultaneously satisfies $( \frac { \sqrt { 8 B \ln ( 2 / \delta ) } } { \sigma n } , \delta / 2 )$ -diferential privacy. The Laplace mechanism portion of Thresholdout satisfies $( B / ( \sigma n ) , 0 )$ -diferential privacy by [DR14, Thm. 3.6], and simultaneously satisfies $\big ( \frac { \sqrt { 8 B \ln ( 2 / \delta ) } } { \sigma n } , \delta / 2 \big ) \mathrm { . }$ diferential privacy by [DR14, Thm. 3.6] and [DR14, Cor. 3.21]. Finally, the composition of two mechanisms, the first of which is (<sub>1</sub>, δ<sub>1</sub>)-diferentially private, and the second of which is $( \epsilon _ { 2 } , \delta _ { 2 } )$ -diferentially private is itself $( \epsilon _ { 1 } + \epsilon _ { 2 } , \delta _ { 1 } + \delta _ { 2 } )$ -diferentially private (Thm. 3). Adding the privacy parameters of the Sparse Vector portion of Thresholdout and the Laplace mechanism portion of Thresholdout yield the parameters of our theorem. □

We note that tighter privacy parameters are possible $( \mathrm { e . g . }$ . by invoking the parameters and guarantees of the algorithm “NumericSparse” ([DR14, Algorithm 3]), which already combines the Laplace addition step) – we chose simpler parameters for clarity.

Note the seeming discrepancy between the guarantee provided by Thresholdout and generalization guarantees in Theorem 8 and Corollary 7: while Theorem 8 promises generalization bounds for functions that are generated by a diferentially private algorithm, here we allow an arbitrary data analyst to generate query functions in any way she chooses, with access to the training set and diferentially private estimates of the means of her functions on the holdout set. The connection comes from preservation of diferential privacy guarantee under post-processing (Lem. 5).

We can now quantify the number of samples necessary to achieve generalization error τ with probability

at least $1 - \beta .$

Lemma 24. Let $\tau , \beta , T , B > 0$ . Let S denote the holdout dataset of size n drawn i.i.d. from a distribution $\mathcal { P } .$ Consider an algorithm that is given access to $S _ { t }$ and adaptively chooses functions $\phi _ { 1 } , \ldots , \phi _ { m } : \mathcal { X } \to [ 0 , 1 ]$ while interacting with Thresholdout which is given datasets $s , S _ { t }$ and parameters $\sigma , B , { \cal T } .$ . If

$$
n \geq n _ {0} (B, \sigma , \tau , \beta) \doteq \max \{2 B / (\sigma \tau), \ln (6 / \beta) / \tau^ {2} \}
$$

or

$$
n \geq n _ {1} (B, \sigma , \tau , \beta) \doteq \frac {8 0 \cdot \sqrt {B \ln (1 / (\tau \beta))}}{\tau \sigma}
$$

then for every $i \in [ m ] , \mathbb { P } \left[ | { \mathcal { P } } [ \phi _ { i } ] - { \mathcal { E } } _ { S } [ \phi _ { i } ] | \geq \tau \right] \leq \beta .$

Proof. Consider the first guarantee of Lemma 23. In order to achieve generalization error τ via Corollary 7 (i.e. in order to guarantee that for every function φ we have: $\begin{array} { r } { \mathbb { P } \left[ | \mathcal { P } [ \breve { \phi } _ { i } ] - \mathcal { E } _ { S } [ \phi _ { i } ] | \geq \tau \right] \leq 6 e ^ { - \tau ^ { 2 } n } \Big ) } \end{array}$ we need to have n large enough to achieve $( \varepsilon , 0 )$ -diferential privacy for $\varepsilon = \tau$ . To achieve this it sufices to have $n \geq 2 B / ( \sigma \tau )$ . By ensuring that $n \geq \ln ( 6 / \beta ) / \tau ^ { 2 }$ we also have that $6 e ^ { - \tau ^ { 2 } n } \leq \beta$

We can also make use of the second guarantee in Lemma 23 together with the results of Nissim and Stemmer [NS15] (Thm. 8). In order to achieve generalization error τ with probability $1 - \beta \ ( \mathrm { i . e }$ . in order to guarantee for every function φ we have: $\mathbb { P } \left[ | \bar { \mathcal { P } } [ \phi _ { i } ] - \mathcal { E } _ { S } [ \phi _ { i } ] | \ge \tau \right] \le \beta )$ , we can apply Thm. 8 by setting $\epsilon = \sqrt { 3 2 B \ln ( 2 / \delta ) } / ( \sigma n ) = \tau / 1 3$ and $\begin{array} { r } { \delta = \frac { \beta \tau ^ { - } } { 2 6 \ln { ( 2 6 / \tau ) } } } \end{array}$ . We can obtain these privacy parameters from Lemma 23 by choosing any $n \geq \frac { 8 0 \cdot \sqrt { B \ln ( 1 / ( \tau \beta ) ) } } { \tau \sigma }$ (for suficiently small β and τ). We remark that a somewhat worse bound of $\begin{array} { r } { n _ { 1 } ( B , \sigma , \tau , \beta ) = \frac { \sqrt { 2 0 4 8 \ln ( 8 / \beta ) } } { \tau ^ { 3 / 2 } \sigma } } \end{array}$ follows by setting $\epsilon = \tau / 4$ and $\delta = ( \beta / 8 ) ^ { 4 / \tau }$ in $[ \mathrm { D F H ^ { + } 1 4 } ,$ , Thm. 10]. □

Both settings lead to small generalization error and so we can pick whichever gives the larger bound. The first bound has grows linearly with B but is simpler can be easily extended to other distributions over datasets and to low-sensitivity functions. The second bound has quadratically better dependence on B at the expense of a slightly worse dependence on τ . We can now apply our main results to get a generalization bound for the entire execution of Thresholdout.

Theorem 25. Let $\beta , \tau > 0$ and $m \ge B > 0$ . We set $T = 3 \tau / 4$ and $\sigma = \tau / ( 9 6 \ln ( 4 m / \beta ) )$ . Let S denote a holdout dataset of size n drawn i.i.d. from a distribution $\mathcal { P }$ and $S _ { t }$ be any additional dataset over $\mathcal { X } .$ Consider an algorithm that is given access to $S _ { t }$ and adaptively chooses functions $\phi _ { 1 } , \ldots , \phi _ { m }$ while interacting with Thresholdout which is given datasets $S , S _ { t }$ and values $\sigma , B , { \cal T }$ . For every $i \in [ m ]$ , let $\mathbf { a } _ { i }$ denote the answer of Thresholdout on function $\phi _ { i } : \mathcal { X } \to [ 0 , 1 ]$ . Further, for every $i \in [ m ]$ , we define the counter of overfitting

$$
\mathbf {Z} _ {i} \doteq | \{j \leq i: | \mathcal {P} [ \phi_ {j} ] - \mathcal {E} _ {S _ {t}} [ \phi_ {j} ] | > \tau / 2 \} |.
$$

Then

$$
\mathbb {P} \left[ \exists i \in [ m ], \boldsymbol {Z} _ {i} <   B \& | \boldsymbol {a} _ {i} - \mathcal {P} [ \phi_ {i} ] | \geq \tau \right] \leq \beta
$$

$$
\text {whenever} n \geq \min \{n _ {0} (B, \sigma , \tau / 8, \beta / (2 m)), n _ {1} (B, \sigma , \tau / 8, \beta / (2 m)) \} = O \left(\frac {\ln (m / \beta)}{\tau^ {2}}\right) \cdot \min \{B, \sqrt {B \ln (\ln (m / \beta) / \tau)} \}.
$$

Proof. There are two types of error we need to control: the deviation between $\mathbf { a } _ { i }$ and the average value of $\phi _ { i }$ on the holdout set $\xi _ { S } [ \phi _ { i } ]$ , and the deviation between the average value of $\phi _ { i }$ on the holdout set and the expectation of $\phi _ { i }$ on the underlying distribution, $\mathcal { P } [ \phi _ { i } ]$ . Specifically, we decompose the error as

$$
\mathbb {P} \left[ \boldsymbol {a} _ {i} \neq \bot \& | \boldsymbol {a} _ {i} - \mathcal {P} [ \phi_ {i} ] | \geq \tau \right] \leq \mathbb {P} \left[ \boldsymbol {a} _ {i} \neq \bot \& | \boldsymbol {a} _ {i} - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] | \geq 7 \tau / 8 \right] + \mathbb {P} \left[ | \mathcal {P} [ \phi_ {i} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] | \geq \tau / 8 \right].\tag{4}
$$

To control the first term we need to bound the values of noise variables used by Thresholdout. For the second term we will use the generalization properties of Thresholdout given in Lemma 24.

We now deal with the errors introduced by the noise variables. For $i \in [ m ]$ , let $\eta _ { i } , \xi _ { i }$ and $\gamma _ { i }$ denote the random variables $\eta , \xi$ and $\gamma ,$ respectively, at step i of the execution of Thresholdout. We first note that each of these variables is chosen from Laplace distribution at most m times. By properties of the Laplace distribution with parameter 4σ, we know that for every $t > 0 , \mathbb { P } [ | \pmb { \eta } _ { i } | \geq t \cdot 4 \sigma ] = e ^ { - t / 2 }$ . Therefore for $t = 2 \ln ( 4 m / \beta )$ we obtain

$$
\mathbb {P} [ | \pmb {\eta} _ {i} | \geq 2 \ln (4 m / \beta) \cdot 4 \sigma ] \leq e ^ {- t / 2} = \frac {\beta}{4 m}.
$$

By the definition of $\sigma , 8 \ln ( 4 m / \beta ) \cdot \sigma = \tau / 1 2$ . Applying the union bound we obtain that

$$
\mathbb {P} [ \exists i, | \boldsymbol {\eta} _ {i} | \geq \tau / 1 2 ] \leq \beta / 4,
$$

where by $\exists i$ we refer to $\exists i \in [ m ]$ for brevity. Similarly, $\pmb { \xi } _ { i }$ and $\gamma _ { i }$ are obtained by sampling from the Laplace distribution and each is re-randomized at most $B$ times. Therefore

$$
\mathbb {P} [ \exists i, | \boldsymbol {\gamma} _ {i} | \geq \tau / 2 4 ] \leq B \cdot \beta / 4 m \leq \beta / 4
$$

and

$$
\mathbb {P} [ \exists i, | \boldsymbol {\xi} _ {i} | \geq \tau / 4 8 ] \leq B \cdot \beta / 4 m \leq \beta / 4.
$$

For answers that are diferent from ⊥, we can now bound the first term of Equation (4) by considering two cases, depending on whether Thresholdout answers query $\phi _ { i }$ by returning ${ \pmb a } _ { i } = \mathcal { E } _ { S } [ \phi _ { i } ] + { \pmb \xi } _ { i }$ or by returning $a _ { i } = \mathcal { E } _ { S _ { t } } [ \phi _ { i } ]$ . First, consider queries whose answers are returned using the former condition. Under this condition, $| \bar { \pmb { a } } _ { i } - \mathcal { E } _ { S } [ \phi _ { i } ] | = | \pmb { \xi } _ { i } |$ . Next, we consider the second case, those queries whose answers are returned using $\mathcal { E } _ { S _ { t } } [ \phi _ { i } ]$ . By definition of the algorithm, we have

$$
\left| \boldsymbol {a} _ {i} - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] \right| = \left| \mathcal {E} _ {S _ {t}} [ \phi_ {i} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] \right| \leq T + \boldsymbol {\gamma} _ {i} + \boldsymbol {\eta} _ {i} \leq 3 \tau / 4 + | \boldsymbol {\gamma} _ {i} | + | \boldsymbol {\eta} _ {i} |.
$$

Combining these two cases implies that

$$
\mathbb {P} \left[ \exists i, \boldsymbol {a} _ {i} \neq \bot \& | \boldsymbol {a} _ {i} - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] | \geq 7 \tau / 8 \right] \leq \max \{\mathbb {P} \left[ \exists i, | \boldsymbol {\xi} _ {i} | \geq 7 \tau / 8 \right], \mathbb {P} \left[ \exists i, | \boldsymbol {\gamma} _ {i} | + | \boldsymbol {\eta} _ {i} | \geq \tau / 8 \right] \}.
$$

Noting that $\tau / 2 4 + \tau / 1 2 = \tau / 8$ and applying our bound on variables $\eta _ { i } , \xi _ { i }$ and $\gamma _ { i }$ we get

$$
\mathbb {P} \left[ \exists i, \boldsymbol {a} _ {i} \neq \bot \& | \boldsymbol {a} _ {i} - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] | \geq 7 \tau / 8 \right] \leq \beta / 2.\tag{5}
$$

By Lemma 24, for $n \geq \operatorname* { m i n } \{ n _ { 0 } ( B , \sigma , \tau / 8 , \beta / 2 m ) , n _ { 1 } ( B , \sigma , \tau / 8 , \beta / 2 m ) \}$

$$
\mathbb {P} \left[ | \mathcal {P} [ \phi_ {i} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] | \geq \tau / 8 \right] \leq \beta / 2 m.
$$

Applying the union bound we obtain

$$
\mathbb {P} \left[ \exists i, | \mathcal {P} [ \phi_ {i} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] | \geq \tau / 8 \right] \leq \beta / 2.
$$

Combining this with Equation (5) and using in Equation (4) we get that

$$
\mathbb {P} [ \exists i, \boldsymbol {a} _ {i} \neq \bot \& | \boldsymbol {a} _ {i} - \mathcal {P} [ \phi_ {i} ] | \geq \tau ] \leq \beta .
$$

To finish the proof we show that under the conditions on the noise variables and generalization error used above, we have that if $Z _ { i } < B$ then $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf $ . To see this, observe that for every $j \le i$ that reduces Thresholdout’s budget, we have

$$
\begin{array}{r l} & {| \mathcal {P} [ \phi_ {j} ] - \mathcal {E} _ {S _ {t}} [ \phi_ {j} ] | \geq | \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {j} ] - \mathcal {E} _ {S _ {t}} [ \phi_ {j} ] | - | \mathcal {P} [ \phi_ {j} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {j} ] |} \\ & {\qquad \geq | T + \boldsymbol {\gamma} _ {j} + \boldsymbol {\eta} _ {j} | - | \mathcal {P} [ \phi_ {j} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {j} ] |} \\ & {\qquad \geq T - | \boldsymbol {\gamma} _ {j} | - | \boldsymbol {\eta} _ {j} | - | \mathcal {P} [ \phi_ {j} ] - \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {j} ] |.} \end{array}
$$

This means that for every $j \le i$ that reduces the budget we have $| \mathcal { P } [ \phi _ { j } ] - \mathcal { E } _ { S _ { t } } [ \phi _ { j } ] | \geq 3 \tau / 4 - \tau / 2 4 - \tau / 1 2 - \tau / 8 =$ $\tau / 2$ and hence (when the conditions on the noise variables and generalization error are satisfied) for every $i ,$ if $Z _ { i } < B$ then Thresholdout’s budget is not yet exhausted and $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Xi \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \Xi \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \Xi \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf { } \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \mathbf$ . We can therefore conclude that

$$
\mathbb {P} [ \exists i, \boldsymbol {Z} _ {i} <   B \& | \boldsymbol {a} _ {i} - \mathcal {P} [ \phi_ {i} ] | \geq \tau ] \leq \beta .
$$

Note that in the final bound on $n ,$ the term $O \left( { \frac { \ln ( m / \beta ) } { \tau ^ { 2 } } } \right)$ is equal (up to a constant factor) to the number of samples that are necessary to answer m non-adaptively chosen queries with tolerance τ and confidence $1 - \beta$ In particular, as in the non-adaptive setting, achievable tolerance τ scales as $1 / \sqrt { n }$ (up to the logarithmic factor). Further, this bound allows m to be exponentially large in n as long as B grows sub-quadratically in n (that is, $B \leq n ^ { 2 - c }$ for a constant $c > 0 )$

Remark 26. In Thm. 25 $S _ { h }$ is used solely to provide a candidate estimate of the expectation of each query function. The theorem holds for any other way to provide such estimates. In addition, the one-sided version of the algorithm can be used when catching only the one-sided error is necessary. For example, in many cases overfitting is problematic only if the training error estimate is larger than the true error. This is achieved by using the condition $\mathcal { E } _ { S _ { h } } [ \phi ] - \mathcal { E } _ { S _ { t } } [ \phi ] > \hat { T } + \eta$ to detect overfitting. In this case only one-sided errors will be caught $b y$ Thresholdout and only one-sided overfitting will decrease the budget.

## 4.2 SparseValidate

We now present a general algorithm for validation on the holdout set that can validate many arbitrary queries as long as few of them fail the validation. The algorithm which we refer to as SparseValidate only reveals information about the holdout set when validation fails and therefore we use bounds based on description length to analyze its generalization guarantees.

More formally, our algorithm allows the analyst to pick any Boolean function of a dataset ψ (or even any algorithm that outputs a single bit) and provides back the value of $\psi$ on the holdout set $\psi ( S _ { h } )$ . SparseValidate has a budget m for the total number of queries that can be asked and budget B for the number of queries that returned 1. Once either of the budgets is exhausted, no additional answers are given. We now give a general description of the guarantees of SparseValidate.

Theorem 27. Let S denote a randomly chosen holdout set of size n. Let A be an algorithm that is given access to SparseValidate(m, B) and outputs queries $\psi _ { 1 } , \ldots , \psi _ { m }$ such that each $\psi _ { i }$ is in some set $\Psi _ { i }$ of functions from $\mathcal { X } ^ { n } \ t o \ \{ 0 , 1 \}$ . Assume that for every $i \in [ m ]$ and $\psi _ { i } \in \Psi _ { i } , \mathbb { P } [ \psi _ { i } ( S ) = 1 ] \leq \beta _ { i }$ . Let ψ be the random variable equal to the i’th query of A on S. Then $\mathbb { P } [ \psi _ { i } ( S ) = 1 ] \le \ell _ { i } \cdot \beta _ { i }$ , where $\begin{array} { r } { \ell _ { i } = \sum _ { j = 0 } ^ { \operatorname* { m i n } \{ i - 1 , B \} } \binom { i } { j } \leq m ^ { B } } \end{array}$

Proof. Let B denote the algorithm that represents view the interaction of A with $\mathsf { S p a r s e V a l i d a t e } ( m , B )$ up until query i and outputs the all the i − 1 responses of SparseValidate $( m , B )$ in this interaction. If there are B responses with value 1 in the interaction then all the responses after the last one are meaningless and can be assumed to be equal to 0. The number of binary strings of length i − 1 that contain at most $B$ ones is exactly $\begin{array} { r } { \ell _ { i } = \sum _ { j = 0 } ^ { \operatorname* { m i n } \{ i - 1 , B \} } \binom { i } { j } } \end{array}$ . Therefore we can assume that the output domain of B has size $\ell _ { i }$ and we denote it by Y. Now, for $y \in \mathcal { V }$ let $R ( y )$ be the set of datasets S such that $\psi _ { i } ( S ) = 1$ , where $\psi _ { i }$ is the function that A generates when the responses of SparseValidate(m, B) are y and the input holdout dataset is S (for now assume that A is deterministic). By the conditions of the theorem we have that for every $y , \mathbb { P } [ S \in R ( y ) ] \leq \beta _ { i }$ Applying Thm. 9 to B, we get that $\mathbb { P } [ S \in R ( B ( S ) ) ] \le \ell _ { i } \beta _ { i }$ , which is exactly the claim. We note that to address the case when A is randomized (including dependent on the random choice of the training set) we can use the argument above for every fixing of all the random bits of A. From there we obtain that the claim holds when the probability is taken also over the randomness of A.

We remark that the proof can also be obtained via a more direct application of the union bound over all strings in Y. But the proof via Thm. 9 demonstrates the conceptual role that short description length plays in this application. □

In this general formulation it is the analyst’s responsibility to use the budgets economically and pick query functions that do not fail validation often. At the same time, SparseValidate ensures that (for the appropriate values of the parameters) the analyst can think of the holdout set as a fresh sample for the purposes of validation. Hence the analyst can pick queries in such a way that failing the validation reliably indicates overfitting. To relate this algorithm to Thresholdout, consider the validation query function that is the indicator of the condition $| \mathcal { E } _ { S _ { h } } [ \phi ] - \mathcal { E } _ { S _ { t } } [ \phi ] | > \hat { T } + \eta$ (note that this condition can be evaluated using an algorithm with access to $S _ { h } )$ . This is precisely the condition that consumes the overfitting budget of Thresholdout. Now, as in Thresholdout, for every fixed $\phi , \mathbb { P } [ | \mathcal { E } _ { S _ { h } } [ \phi ] - \mathcal { P } [ \phi ] | \ge \tau ] \le 2 e ^ { - 2 \tau ^ { 2 } n }$ . If $B \leq \tau ^ { 2 } n /$ ln m, then we obtain that for every query $\phi$ generated by the analyst, we still have strong concentration of the mean on the holdout set around the expectation: $\begin{array} { r } { \mathbb { P } [ | \mathcal { E } _ { S _ { h } } [ \phi ] - \mathcal { P } [ \phi ] | \ge \tau ] \le 2 e ^ { - \tau ^ { 2 } n } } \end{array}$ . This implies that if the condition $| \mathcal { E } _ { S _ { h } } [ \phi ] - \mathcal { E } _ { S _ { t } } [ \phi ] | > \hat { T } + \eta$ holds, then with high probability also the condition $| \mathcal { P } [ \phi ] - \mathcal { E } _ { S _ { t } } [ \phi ] | > \hat { T } + \eta - \tau$ holds, indicating overfitting. One notable distinction of Thresholdout from SparseValidate is that SparseValidate does not provide corrections in the case of overfitting. One way to remedy that is simply to use a version of SparseValidate that allows functions with values in $\{ 0 , 1 , \ldots , L \}$ . It is easy to see that for such functions we would obtain the bound of the form $\ell _ { i } \cdot L ^ { \operatorname* { m i n } \{ B , i - 1 \} } \cdot \beta _ { i }$ . To output a value in [0, 1] with precision $\tau ,$ $L = \lfloor 1 / \tau \rfloor$ would sufice. However, in many cases a more economical solution would be to have a separate dataset which is used just for obtaining the correct estimates.

An example of the application of SparseValidate for answering statistical and low-sensitivity queries that is based on our analysis can be found in [BSSU15]. The analysis of generalization on the holdout set in [BH15] and the analysis of the Median Mechanism we give in Section B also rely on this sparsity-based technique.

An alternative view of this algorithm is as a general template for designing algorithms for answering some specific type of adaptively chosen queries. Generalization guarantees specific to the type of query can then be obtained from our general analysis. For example, an algorithm that fits a mixture of Gaussians model to the data could define the validation query to be an algorithm that fits the mixture model to the holdout and obtains a vector of parameters $\Theta _ { h }$ . The validation query then compares it with the vector of parameters $\Theta _ { t }$ obtained on the training set and outputs 1 if the parameter vectors are “not close” (indicating overfitting). Given guarantees of statistical validity of the parameter estimation method in the static setting one could then derive guarantees for adaptive validation via Thm. 27.

## 5 Experiments

We describe a simple experiment on synthetic data that illustrates the danger of reusing a standard holdout set and how this issue can be resolved by our reusable holdout. In our experiment the analyst wants to build a classifier via the following common strategy. First the analyst finds a set of single attributes that are correlated with the class label. Then the analyst aggregates the correlated variables into a single model of higher accuracy (for example using boosting or bagging methods). More formally, the analyst is given a d-dimensional labeled data set S of size 2n and splits it randomly into a training set $S _ { t }$ and a holdout set $S _ { h }$ of equal size. We denote an element of S by a tuple $( x , y )$ where x is a d-dimensional vector and $y \in \{ - 1 , 1 \}$ is the corresponding class label. The analyst wishes to select variables to be included in her classifier. For various values of the number of variables to select k, she picks k variables with the largest absolute correlations with the label. However, she verifies the correlations (with the label) on the holdout set and uses only those variables whose correlation agrees in sign with the correlation on the training set and both correlations are larger than some threshold in absolute value. She then creates a simple linear threshold classifier on the selected variables using only the signs of the correlations of the selected variables. A final test evaluates the classification accuracy of the classifier on both the training set and the holdout set.

Formally, the algorithm is used to build a linear threshold classifier:

1. For each attribute $i \in [ d ]$ compute the correlation with the label on the training and holdout sets: $\begin{array} { r } { w _ { i } ^ { t } = \sum _ { ( x , y ) \in S _ { t } } x _ { i } y } \end{array}$ and $\begin{array} { r } { \dot { w } _ { i } ^ { h } = \sum _ { ( x , y ) \in S _ { h } } x _ { i } y } \end{array}$ . Let

$$
W = \left\{i \mid w _ {i} ^ {t} \cdot w _ {i} ^ {h} > 0; | w _ {i} ^ {t} | \geq 1 / \sqrt {n}; | w _ {i} ^ {h} | \geq 1 / \sqrt {n} | \right\}
$$

that is the set of variables for which $\boldsymbol { w } _ { i } ^ { t }$ and $w _ { i } ^ { h }$ have the same sign and both are at least $1 / \sqrt { n }$ in absolute value (this is the standard deviation of the correlation in our setting). Let $V _ { k }$ be the subset of variables in W with k largest values of $| w _ { i } ^ { t } |$

2. Construct the classifier $\begin{array} { r } { f ( x ) = \mathrm { s g n } \left( \sum _ { i \in V _ { k } } \mathrm { s g n } ( w _ { i } ^ { t } ) \cdot x _ { i } \right) } \end{array}$

In the experiments we used an implementation of Thresholdout that difers somewhat from the algorithm we analyzed theoretically (given in Figure 1). Specifically, we set the parameters to be $T = 0 . 0 4$ and $\tau = 0 . 0 1$ This is lower than the values necessary for the proof (and which are not intended for direct application) but sufices to prevent overfitting in our experiment. Second, we use Gaussian noise instead of Laplacian noise as it has stronger concentration properties (in many diferential privacy applications similar theoretical guarantees hold mechanisms based on Gaussian noise).

No correlation between labels and data: In our first experiment, each attribute is drawn independently from the normal distribution $N ( 0 , 1 )$ and we choose the class label $y \in \{ - 1 , 1 \}$ uniformly at random so that there is no correlation between the data point and its label. We chose $n = 1 0 , 0 0 0 , d = 1 0 , 0 0 0$ and varied the number of selected variables k. In this scenario no classifier can achieve true accuracy better than 50%. Nevertheless, reusing a standard holdout results in reported accuracy of over 63% for $k = 5 0 0$ on both the training set and the holdout set (the standard deviation of the error is less than 0.5%). The average and standard deviation of results obtained from 100 independent executions of the experiment are plotted in Figure 2 which also includes the accuracy of the classifier on another fresh data set of size n drawn from the same distribution. We then executed the same algorithm with our reusable holdout. The algorithm Thresholdout was invoked with $T = 0 . 0 4$ and $\tau = 0 . 0 1$ explaining why the accuracy of the classifier reported by Thresholdout is of by up to 0.04 whenever the accuracy on the holdout set is within 0.04 of the accuracy on the training set. Thresholdout prevents the algorithm from overfitting to the holdout set and gives a valid estimate of classifier accuracy.


Figure 2: No correlation between class labels and data points. The plot shows the classification accuracy of the classifier on training, holdout and fresh sets. Margins indicate the standard deviation.

High correlation between labels and some of the variables: In our second experiment, the class labels are correlated with some of the variables. As before the label is randomly chosen from $\{ - 1 , 1 \}$ and each of the attributes is drawn from $N ( 0 , 1 )$ aside from 20 attributes which are drawn from $N ( y \cdot 0 . 0 6 , 1 )$ where y is the class label. We execute the same algorithm on this data with both the standard holdout and Thresholdout and plot the results in Figure 3. Our experiment shows that when using the reusable holdout, the algorithm still finds a good classifier while preventing overfitting. This illustrates that the reusable holdout simultaneously prevents overfitting and allows for the discovery of true statistical patterns.

In Figures 2 and 3, simulations that used Thresholdout for selecting the variables also show the accuracy on the holdout set as reported by Thresholdout. For comparison purposes, in Figure 4 we plot the actual accuracy of the generated classifier on the holdout set (the parameters of the simulation are identical to those used in Figures 2 and 3). It demonstrates that there is essentially no overfitting to the holdout set. Note that the advantage of the accuracy reported by Thresholdout is that it can be used to make further data dependent decisions while mitigating the risk of overfitting.

Discussion of the results: Overfitting to the standard holdout set arises in our experiment because the analyst reuses the holdout after using it to measure the correlation of single attributes. We first note that neither cross-validation nor bootstrap resolve this issue. If we used either of these methods to validate the correlations, overfitting would still arise due to using the same data for training and validation (of the final classifier). It is tempting to recommend other solutions to the specific problem we used in our experiment. Indeed, a significant number of methods in the statistics and machine learning literature deal with inference for fixed two-step procedures where the first step is variable selection (see [HTF09] for examples). Our experiment demonstrates that even in such simple and standard settings our method avoids overfitting without the need to use a specialized procedure – and, of course, extends more broadly. More importantly, the reusable holdout gives the analyst a general and principled method to perform multiple validation steps where previously the only known safe approach was to collect a fresh holdout set each time a function depend on the outcomes of previous validations.



Figure 3: Some variables are correlated with the label.
Figure 4: Accuracy of the classifier produced with Thresholdout on the holdout set.

## 6 Conclusions

In this work, we give a unifying view of two techniques (diferential privacy and description length bounds) which preserve the generalization guarantees of subsequent algorithms in adaptively chosen sequences of data analyses. Although these two techniques both imply low max-information – and hence can be composed together while preserving their guarantees – the kinds of guarantees that can be achieved by either alone are incomparable. This suggests that the problem of generalization guarantees under adaptivity is ripe for future study on two fronts. First, the existing theory is likely already strong enough to develop practical algorithms with rigorous generalization guarantees, of which Thresholdout is an example. However additional empirical work is needed to better understand when and how the theory should be applied in specific application scenarios. At the same time, new theory is also needed. As an example of a basic question we still do not know the answer to: even in the simple setting of adaptively reusing a holdout set for computing the expectations of boolean-valued predicates, is it possible to obtain stronger generalization guarantees (via any means) than those that are known to be achievable via diferential privacy?

## References

[BDMN05] Avrim Blum, Cynthia Dwork, Frank McSherry, and Kobbi Nissim. Practical privacy: the SuLQ framework. In PODS, pages 128–138, 2005.

[BE02] Olivier Bousquet and Andr´e Elisseef. Stability and generalization. JMLR, 2:499–526, 2002.

[BH15] Avrim Blum and Moritz Hardt. The ladder: A reliable leaderboard for machine learning competitions. CoRR, abs/1502.04585, 2015.

[BNS13] Amos Beimel, Kobbi Nissim, and Uri Stemmer. Characterizing the sample complexity of private learners. In ITCS, pages 97–110, 2013.

[BSSU15] Raef Bassily, Adam Smith, Thomas Steinke, and Jonathan Ullman. More general queries and less generalization error in adaptive data analysis. CoRR, abs/1503.04843, 2015.

[CT10] Gavin C. Cawley and Nicola L. C. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11:2079–2107, 2010.

[DFH<sup>+</sup>14] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. Preserving statistical validity in adaptive data analysis. CoRR, abs/1411.2664, 2014. Extended abstract in STOC 2015.

[DFN07] Chuong B. Do, Chuan-Sheng Foo, and Andrew Y. Ng. Eficient multiple hyperparameter learning for log-linear models. In NIPS, pages 377–384, 2007.

[DKM<sup>+</sup>06] Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In EUROCRYPT, pages 486–503, 2006.

[DL09] C. Dwork and J. Lei. Diferential privacy and robust statistics. In Proceedings of the 2009 International ACM Symposium on Theory of Computing (STOC), 2009.

[DMNS06] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography, pages 265–284. Springer, 2006.

[DN03] Irit Dinur and Kobbi Nissim. Revealing information while preserving privacy. In PODS, pages 202–210. ACM, 2003.

[DN04] Cynthia Dwork and Kobbi Nissim. Privacy-preserving datamining on vertically partitioned databases. In CRYPTO, pages 528–544, 2004.

[DR14] Cynthia Dwork and Aaron Roth. The algorithmic foundations of diferential privacy. Foundations and Trends in Theoretical Computer Science, 9(34):211–407, 2014.

[Dwo11] Cynthia Dwork. A firm foundation for private data analysis. CACM, 54(1):86–95, 2011.

[Fre83] David A. Freedman. A note on screening regression equations. The American Statistician, 37(2):152–155, 1983.

[Fre98] Yoav Freund. Self bounding learning algorithms. In COLT, pages 247–258, 1998.

[HTF09] Trevor Hastie, Robert Tibshirani, and Jerome H. Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer series in statistics. Springer, 2009.

[HU14] Moritz Hardt and Jonathan Ullman. Preventing false discovery in interactive data analysis is hard. In FOCS, pages 454–463, 2014.

[Lan05] John Langford. Clever methods of overfitting. http://hunch.net/?p=22, 2005.

[LB03] John Langford and Avrim Blum. Microchoice bounds and self bounding learning algorithms. Machine Learning, 51(2):165–179, 2003.

[MMP<sup>+</sup>10] Andrew McGregor, Ilya Mironov, Toniann Pitassi, Omer Reingold, Kunal Talwar, and Salil P. Vadhan. The limits of two-party diferential privacy. In FOCS, pages 81–90. IEEE Computer Society, 2010.

[MNPR06] Sayan Mukherjee, Partha Niyogi, Tomaso Poggio, and Ryan Rifkin. Learning theory: stability is suficient for generalization and necessary and suficient for consistency of empirical risk minimization. Advances in Computational Mathematics, 25(1-3):161–193, 2006.

[Ng97] Andrew Y. Ng. Preventing ”overfitting” of cross-validation data. In ICML, pages 245–253, 1997.

[NS15] Kobbi Nissim and Uri Stemmer. On the generalization properties of diferential privacy. CoRR, abs/1504.05800, 2015.

[PRMN04] Tomaso Poggio, Ryan Rifkin, Sayan Mukherjee, and Partha Niyogi. General conditions for predictivity in learning theory. Nature, 428(6981):419–422, 2004.

[Reu03] Juha Reunanen. Overfitting in making comparisons between variable selection methods. Journal of Machine Learning Research, 3:1371–1382, 2003.

[RF08] R. Bharat Rao and Glenn Fung. On the dangers of cross-validation. an experimental evaluation. In International Conference on Data Mining, pages 588–596. SIAM, 2008.

[RR10] Aaron Roth and Tim Roughgarden. Interactive privacy via the median mechanism. In 42nd ACM STOC, pages 765–774. ACM, 2010.

[SSBD14] Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.

[SSSSS10] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. The Journal of Machine Learning Research, 11:2635–2670, 2010.

[SU14] Thomas Steinke and Jonathan Ullman. Interactive fingerprinting codes and the hardness of preventing false discovery. arXiv preprint arXiv:1410.1228, 2014.

[WLF15] Yu-Xiang Wang, Jing Lei, and Stephen E. Fienberg. Learning with diferential privacy: Stability, learnability and the suficiency and necessity of ERM principle. CoRR, abs/1502.06309, 2015.

## A From Max-information to Randomized Description Length

In this section we demonstrate additional connections between max-information, diferential privacy and description length. These connections are based on a generalization of description length to randomized algorithms that we refer to as randomized description length.

Definition 28. For a universe Y let A be a randomized algorithm with input in X and output in Y. We say that the output of A has randomized description length k if for every fixed setting of random coin flips of A the set of possible outputs of A has size at most 2<sup>k</sup>.

We first note that just as the (deterministic) description length, randomized description length implies generalization and gives a bound on max-information.

Theorem 29. Let $\mathcal { A } : \mathcal { X } ^ { n }  \mathcal { Y }$ be an algorithm with randomized description length k and let S be a random dataset over $\mathcal { X } ^ { n }$ . Assume that $R : { \dot { \mathcal { V } } }  2 ^ { \mathcal { X } ^ { n } }$ is such that for every $y \in \mathcal { V } , \mathbb { P } [ S \in R ( y ) ] \le \beta$ . Then $\mathbb { P } [ S \in R ( { \cal A } ( S ) ) ] \le 2 ^ { k } \cdot \beta$

Theorem 30. Let A be an algorithm with randomized description length k taking as an input an n-element dataset and outputting a value in Y. Then for every $\beta > 0 , I _ { \infty } ^ { \beta } ( \mathcal { A } , n ) \leq \log ( | \mathcal { V } | / \beta )$

Proof. Let S be any random variable over n-element input datasets and let Y be the corresponding output distribution $\pmb { Y } = \pmb { \mathcal { A } } ( \pmb { S } )$ . It sufices to prove that for every $\beta > 0 , I _ { \infty } ^ { \beta } ( S ; Y ) \leq k + \log ( 1 / \beta )$

Let R be the set of all possible values of the random bits of A and let R denote the uniform distribution over a choice of $r \in R .$ . For $r \in R ,$ let $A _ { r }$ denote $\mathcal { A }$ with the random bits set to r and let $\pmb { Y } _ { r } = \pmb { \mathcal { A } } _ { r } ( \pmb { S } )$ . Observe that by the definition of randomized description length, the range of $\boldsymbol { A } _ { r }$ has size at most $2 ^ { k }$ . Therefore, by Theorem 17, we obtain that $I _ { \infty } ^ { \beta } ( S ; Y _ { r } ) \le \log ( 2 ^ { k } / \beta )$ .

For any event $\mathcal { O } \subseteq \mathcal { X } ^ { n } \times \mathcal { Y }$ we have that

$$
\begin{array}{l} \mathbb {P} [ (\boldsymbol {S}, \boldsymbol {Y}) \in \mathcal {O} ] = \underset {r \sim \mathcal {R}} {\mathbb {E}} [ \mathbb {P} [ (\boldsymbol {S}, \boldsymbol {Y} _ {r}) \in \mathcal {O} ] ] \\ \qquad \leq \underset {r \sim \mathcal {R}} {\mathbb {E}} \left[ \frac {2 ^ {k}}{\beta} \cdot \mathbb {P} [ \boldsymbol {S} \times \boldsymbol {Y} _ {r} \in \mathcal {O} ] + \beta \right] \\ \qquad = \frac {2 ^ {k}}{\beta} \cdot \underset {r \sim \mathcal {R}} {\mathbb {E}} [ \mathbb {P} [ \boldsymbol {S} \times \boldsymbol {Y} _ {r} \in \mathcal {O} ] ] + \beta \\ \qquad = \frac {2 ^ {k}}{\beta} \cdot \mathbb {P} [ \boldsymbol {S} \times \boldsymbol {Y} \in \mathcal {O} ] + \beta . \end{array}
$$

By the definition of β-approximate max-information, we obtain that $I _ { \infty } ^ { \beta } ( S ; Y ) \le \log ( 2 ^ { k } / \beta )$

We next show that if the output of an algorithm A has low approximate max-information about its input then there exists a (diferent) algorithm whose output is statistically close to that of A while having short randomized description. We remark that this reduction requires the knowledge of the marginal distribution $\boldsymbol { \mathcal { A } } ( \boldsymbol { S } )$

Lemma 31. Let A be a randomized algorithm taking as an input a dataset of n points from X and outputting a value in Y. Let Z be a random variable over Y. For $k > 0$ and a dataset S let $\beta _ { S } = \operatorname* { m i n } \{ \beta \mid D _ { \infty } ^ { \beta } ( A ( S ) \| Z ) \leq$ k}. There exists an algorithm A<sup>0</sup> that given $S \in { \mathcal { X } } ^ { n } , \beta , k$ and any $\beta ^ { \prime } > 0$

1. the output of A<sup>0</sup> has randomized description length $k + \log \ln ( 1 / \beta ^ { \prime } )$

2. for every S, $\Delta ( \mathcal { A } ^ { \prime } ( S ) , \mathcal { A } ( S ) ) \le \beta _ { S } + \beta ^ { \prime }$

Proof. Let S denote the input dataset. By definition of $\beta _ { S } , ~ D _ { \infty } ^ { \beta _ { S } } ( A ( S ) \| Z ) \le k$ . By the properties of approximate divergence $( e . g . [ \mathrm { D R 1 4 } ] ) , D _ { \infty } ^ { \beta _ { S } } ( A ( S ) \| Z ) \le k$ implies that there exists a random variable $\mathbf { Y }$ such that $\Delta ( \mathcal { A } ( S ) , Y ) \le \beta _ { S }$ and $D _ { \infty } ( Y \| Z ) \le k$

For $t = 2 ^ { k } \ln ( 1 / \beta ^ { \prime } )$ the algorithm $\mathcal { A } ^ { \prime }$ randomly and independently chooses t samples from Z. Denote them by $y _ { 1 } , y _ { 2 } , \ldots , y _ { t }$ . For $i = 1 , 2 , \dots , t , A ^ { \prime }$ outputs $y _ { i }$ with probability $\begin{array} { r } { p _ { i } = \frac { \mathbb { P } [ { \bf Y } = y _ { i } ] } { 2 ^ { k } \cdot \mathbb { P } [ { \bf Z } = y _ { i } ] } } \end{array}$ and goes to the next sample otherwise. Note that $p _ { i } \in [ 0 , 1 ]$ and therefore this is a legal choice of probability. When all samples are exhausted the algorithm outputs y<sub>1</sub>.

We first note that by the definition of this algorithm its output has randomized description length log $t = k + \log \ln ( 1 / \beta ^ { \prime } )$ . Let $T$ denote the event that at least one of the samples was accepted. Conditioned on this event the output of $\mathcal { A } ^ { \prime } ( S )$ is distributed according to $\mathbf { Y }$ . For each $i ,$

$$
\underset {y _ {i} \sim p (\mathbf {Z})} {\mathbb {E}} [ p _ {i} ] = \underset {y _ {i} \sim p (\mathbf {Z})} {\mathbb {E}} \left[ \frac {\mathbb {P} [ \mathbf {Y} = y _ {i} ]}{2 ^ {k} \cdot \mathbb {P} [ \mathbf {Z} = y _ {i} ]} \right] = \sum_ {y _ {i} \in \mathcal {Y}} \frac {\mathbb {P} [ \mathbf {Y} = y _ {i} ]}{2 ^ {k}} = \frac {1}{2 ^ {k}}.
$$

This means that the probability that none of t samples will be accepted is $( 1 - 2 ^ { - k } ) ^ { t } \leq e ^ { t / 2 ^ { k } } \leq \beta ^ { \prime }$ . Therefore $\Delta ( { \mathcal { A } } ^ { \prime } ( S ) , Y ) \leq \beta ^ { \prime }$ and, consequently, $\Delta ( \mathcal { A } ^ { \prime } ( S ) , \mathcal { A } ( S ) ) \le \beta _ { S } + \beta ^ { \prime }$ □

We can now use Lemma 31 to show that if for a certain random choice of a dataset $s ,$ the output of $\mathcal { A }$ has low approximate max-information then there exists an algorithm $\mathcal { A } ^ { \prime }$ whose output on $\pmb { S }$ has low randomized description length and is statistically close to the output distribution of ${ \mathcal { A } } .$

Theorem 32. Let S be a random dataset in $\mathcal { X } ^ { n }$ and A be an algorithm taking as an input a dataset in $\mathcal { X } ^ { n }$ and having a range $\mathcal { V } .$ Assume that for some $\beta \ge 0 , I _ { \infty } ^ { \beta } ( S ; { \mathcal A } ( S ) ) = k$ . For any $\beta ^ { \prime } > 0$ , there exists an algorithm A<sup>0</sup> taking as an input a dataset in $\mathcal { X } ^ { n }$ such that

1. the output of A<sup>0</sup> has randomized description length $k + \log \ln ( 1 / \beta ^ { \prime } )$ ;

$$
2. \Delta ((\boldsymbol {S}, \mathcal {A} ^ {\prime} (\boldsymbol {S})), (\boldsymbol {S}, \mathcal {A} (\boldsymbol {S})) \leq \beta + \beta^ {\prime}.
$$

Proof. For a dataset S let $\beta _ { S } = \operatorname* { m i n } \{ \beta \ \mid D _ { \infty } ^ { \beta } ( A ( S ) \| A ( S ) ) \leq k \}$ . To prove this result it sufices to observe that $\mathbb { E } [ \beta _ { S } ] \le \beta$ and then apply Lemma 31 with $\boldsymbol Z = \boldsymbol A ( \boldsymbol S )$ To show that $\mathbb { E } [ \beta _ { S } ] \le \beta$ let $\mathcal { O } _ { S } \subseteq \mathcal { V }$ denote an event such that $\mathbb { P } [ \mathcal { A } ( S ) \in \mathcal { O } _ { S } ] = 2 ^ { k } \cdot \mathbb { P } [ \mathcal { A } ( S ) \in \mathcal { O } _ { S } ] + \beta _ { S }$ . Let $\textstyle { \mathcal { O } } = \bigcup _ { S \in { \mathcal { X } } ^ { n } } \{ ( { \bar { S } } , { \bar { O } } _ { S } ) \}$ . Then,

$$
\begin{array}{r l} & {\mathbb {P} [ (\boldsymbol {S}, \mathcal {A} (\boldsymbol {S}) \in \mathcal {O} ] = \underset {S \sim p (\boldsymbol {S})} {\mathbb {E}} \left[ \mathbb {P} [ (S, \mathcal {A} (S) \in \mathcal {O} _ {S} ] \right]} \\ & {\qquad = \underset {S \sim p (\boldsymbol {S})} {\mathbb {E}} \left[ 2 ^ {k} \cdot \mathbb {P} [ \mathcal {A} (\boldsymbol {S}) \in \mathcal {O} _ {S} ] + \beta_ {S} \right]} \\ & {\qquad = 2 ^ {k} \cdot \mathbb {P} [ \boldsymbol {S} \times \mathcal {A} (\boldsymbol {S}) \in \mathcal {O} ] + \mathbb {E} [ \beta_ {\boldsymbol {S}} ].} \end{array}
$$

If $\mathbb { E } [ \beta _ { S } ] > \beta$ then it would hold for some $k ^ { \prime } > k$ that $\mathbb { P } [ ( S , { \mathcal { A } } ( S ) \in { \mathcal { O } } ] = 2 ^ { k ^ { \prime } } \cdot \mathbb { P } [ S \times { \mathcal { A } } ( S ) \in { \mathcal { O } } ] + \beta$ contradicting the assumption $I _ { \infty } ^ { \beta } ( S ; { \mathcal { A } } ( S ) ) = k$ . We remark that, it is also easy to see that $\mathbb { E } [ \beta _ { S } ] = \beta$ □

It is important to note that Theorem 32 is not the converse of Theorem 30 and does not imply equivalence between max-information and randomized description length. The primary diference is that Theorem 32 defines a new algorithm rather than arguing about the original algorithm. In addition, the new algorithm requires samples of $\boldsymbol { \mathcal { A } } ( \boldsymbol { S } )$ , that is, it needs to know the marginal distribution on Y. As a more concrete example, Theorem 32 does not allow us to obtain a description-length-based equivalent of Theorem 20 for all i.i.d. datasets. On the other hand, any algorithm that has bounded max-information for all distributions over datasets can be converted to an algorithm with low randomized description length.

Theorem 33. Let A be an algorithm over $\mathcal { X } ^ { n }$ with range $\mathcal { V }$ and let $k = I _ { \infty } ( \mathcal { A } , n )$ . For any $\beta > 0$ , there exists an algorithm A<sup>0</sup> taking as an input a dataset in $\mathcal { X } ^ { n }$ such that

1. the output of A<sup>0</sup> has randomized description length $k + \log \ln ( 1 / \beta )$ ;

2. for every dataset $S \in \mathcal { X } ^ { n } , \Delta ( \mathcal { A } ^ { \prime } ( S ) , \mathcal { A } ( S ) ) \le \beta$

Proof. Let $S _ { 0 } = ( x , x , \dots , x )$ be an n-element dataset for an arbitrary $x \in \mathcal { X }$ . By Lemma 3 we know that for every $S \in { \mathcal { X } } ^ { n } , D _ { \infty } ( A ( S ) \| A ( S _ { 0 } ) ) \leq k$ . We can now apply Lemma 31 with $\pmb { Z } = \mathcal { A } ( S _ { 0 } )$ , and $\beta ^ { \prime } = \beta$ to obtain the result. □

The conditions of Theorem 33 are satisfied by any ε-diferentially private algorithm with $\boldsymbol { k } = \log \boldsymbol { e } \cdot \varepsilon \boldsymbol { n }$ This immediately implies that the output of any ε-diferentially private algorithm is β-statistically close to the output of an algorithm with randomized description length of log $e \cdot \varepsilon n + \log \ln ( 1 / \beta )$ . Special cases of this property have been derived (using a technique similar to Lemma 31) in the context of proving lower bounds for learning algorithms [BNS13] and communication complexity of diferentially private protocols $[ \mathrm { M M P ^ { + } 1 0 } ]$

## B Answering Queries via Description Length Bounds

In this section, we show a simple method for answering any adaptively chosen sequence of m statistical queries, using a number of samples that scales only polylogarithmically in $m .$ . This is an exponential improvement over what would be possible by naively evaluating the queries exactly on the given samples. Algorithms that achieve such dependence were given in $\mathrm { [ D F H ^ { + } 1 4 ] }$ and [BSSU15] using diferentially private algorithms for answering queries and the connection between generalization and diferential privacy (in the same way as we do in Section 4.1). Here we give a simpler algorithm which we analyze using description length bounds. The resulting bounds are comparable to those achieved in $\mathrm { [ D F H ^ { + } 1 4 ] }$ using pure diferential privacy but are somewhat weaker than those achieved using approximate diferential privacy $[ \mathrm { D F H ^ { + } 1 4 } . $ , BSSU15, NS15].

The mechanism we give here is based on the Median Mechanism of Roth and Roughgarden [RR10]. A diferentially private variant of this mechanism was introduced in [RR10] to show that it was possible to answer exponentially many adaptively chosen counting queries (these are queries for an estimate of the empirical mean of a function $\phi : \mathcal { X } \to [ 0 , 1 ]$ on the dataset). Here we analyze a noise-free version and establish its properties via a simple description length-based argument. We remark that it is possible to analogously define and analyze the noise-free version of the Private Multiplicative Weights Mechanism of Hardt and Rothblum [?]. This somewhat more involved approach would lead to better (but qualitatively similar) bounds.

Recall that statistical queries are defined by functions $\phi : \mathcal { X }  [ 0 , 1 ]$ , and our goal is to correctly estimate their expectation ${ \mathcal { P } } [ \phi ]$ . The Median Mechanism takes as input a dataset S and an adaptively chosen sequence of such functions $\phi _ { 1 } , \ldots , \phi _ { m } ;$ , and outputs a sequence of answers $a _ { 1 } , \ldots , a _ { m } .$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm Median Mechanism
Input: An upper bound m on the total number of queries, a dataset S and an accuracy parameter  $\tau$
1. Let  $\alpha = \frac{\tau}{3}$ .
2. Let Consistent $_{0}$  =  $X^{\frac{\log m}{\alpha^{2}}}$
3. For a query  $\phi_{i}$  do
(a) Compute  $a_{i}^{pub} = median(\{\mathcal{E}_{S'}[\phi_{i}] : S' \in Consistent_{i-1}\})$ .
(b) Compute  $a_{i}^{priv} = E_{S}[\phi_{i}]$ .
(c) If  $\left|a_{i}^{pub}-a_{i}^{priv}\right|\leq2\alpha$  Then:
    i. Output  $a_{i}=a_{i}^{pub}$ .
    ii. Let Consistent $_{i}$  = Consistent $_{i-1}$ .
(d) Else:
    i. Output  $a_{i}=\lfloor a_{i}^{priv}\rfloor_{\alpha}$ .
    ii. Let Consistent $_{i}=\{S'\in Consistent_{i-1}:|a_{i}-E_{S'}[\phi_{i}]\leq2\alpha\}$ .
</div>

Figure 5: Noise-free version of the Median Mechanism from [RR10]

The guarantee we get for the Median Mechanism is as follows:

Theorem 34. Let $\beta , \tau > 0$ and $m \geq B > 0$ . Let S denote a dataset of size n drawn i.i.d. from a distribution $\mathcal { P }$ over X. Consider any algorithm that adaptively chooses functions $\phi _ { 1 } , \ldots , \phi _ { m }$ while interacting with the Median Mechanism which is given $\pmb { S }$ and values $\tau , \beta$ . For every $i \in [ m ]$ , let $\mathbf { a } _ { i }$ denote the answer of the Median Mechanism on function $\phi _ { i } : \mathcal { X } \to [ 0 , 1 ]$ . Then

$$
\mathbb {P} \left[ \exists i \in [ m ], | \boldsymbol {a} _ {i} - \mathcal {P} [ \phi_ {i} ] | \geq \tau \right] \leq \beta
$$

whenever

$$
n \geq n _ {0} = \frac {8 1 \cdot \log | \mathcal {X} | \cdot \log m \cdot \ln (3 m / \tau)}{2 \tau^ {4}} + \frac {9 \ln (2 m / \beta)}{2 \tau^ {2}}.
$$

Proof. We begin with a simple lemma which informally states that for every distribution ${ \mathcal { P } } _ { : }$ , and for every set of m functions $\phi _ { 1 } , \ldots , \phi _ { m } ;$ , there exists a small dataset that approximately encodes the answers to each of the corresponding statistical queries.

Lemma 35 ([DR14] Theorem 4.2). For every dataset S over $x ,$ , any set $C = \{ \phi _ { 1 } , \dots , \phi _ { m } \}$ of m functions $\phi _ { i } : \mathcal { X } \to [ 0 , 1 ]$ , and any $\alpha \in [ 0 , 1 ]$ , there exists a data set $S ^ { \prime } \in \mathcal { X } ^ { t }$ of size $\begin{array} { r } { t = \frac { \log ^ { - } m } { \alpha ^ { 2 } } } \end{array}$ such that:

$$
\max _ {\phi_ {i} \in C} | \mathcal {E} _ {S} [ \phi_ {i} ] - \mathcal {E} _ {S ^ {\prime}} [ \phi_ {i} ] | \leq \alpha .
$$

Next, we observe that by construction, the Median Mechanism (as presented in Figure 5) always returns answers that are close to the empirical means of respective query functions.

Lemma 36. For every sequence of queries $\phi _ { 1 } , \ldots , \phi _ { m }$ and dataset S given to the Median Mechanism, we have that for every $_ { i ; }$

$$
\left| a _ {i} - \mathcal {E} _ {S} [ \phi_ {i} ] \right| \leq 2 \alpha .
$$

Finally, we give a simple lemma from [RR10] that shows that the Median Mechanism only returns answers computed using the dataset $S$ in a small number of rounds – for any other round $i ,$ the answer returned is computed from the set Consistent<sub>i−1</sub>.

Lemma 37 ([RR10], see also Chapter 5.2.1 of [DR14]). For every sequence of queries $\phi _ { 1 } , \ldots , \phi _ { m }$ and a dataset S given to the Median Mechanism:

$$
\left| \{i: a _ {i} \neq a _ {i} ^ {p u b} \} \right| \leq \frac {\log | \mathcal {X} | \log m}{\alpha^ {2}}.
$$

Proof. We simply note several facts. First, by construction, $| \mathrm { C o n s i s t e n t _ { 0 } } | = | { \mathcal { X } } | ^ { \log m / \alpha ^ { 2 } }$ . Second, by Lemma 35, for every i, $| \mathrm { C o n s i s t e n t } _ { i } | \geq 1$ (because for every set of m queries, there is at least one dataset $S ^ { \prime }$ of size log $m / \alpha ^ { 2 }$ that is consistent up to error α with $S$ on every query asked, and hence is never removed from Consistent<sub>i</sub> on any round). Finally, by construction, on any round i such that $a _ { i } \neq a _ { i } ^ { \mathrm { p u b } }$ , we have |Consistent<sub>i</sub>| $\leq \frac { 1 } { 2 }$ · |Consistent<sub>i−1</sub>| (because on any such round, the median dataset $S ^ { \prime }$ – and hence at least half of the datasets in Consistent were inconsistent with the answer given, and hence removed.) The lemma follows from the fact that there can therefore be at most log $\left( | \mathcal { X } | ^ { \log m / \alpha ^ { 2 } } \right)$ many such rounds.

Our analysis proceeds by viewing the interaction between the data analyst and the Median Mechanism $\left( { \mathrm { F i g . ~ 5 } } \right)$ as a single algorithm A. A takes as input a dataset S and outputs a set of queries and answers $\mathcal { A } ( S ) = \{ \phi _ { i } , a _ { i } \} _ { i = 1 } ^ { m }$ . We will show that A’s output has short randomized description length (the data analyst is a possibly randomized algorithm and hence $\mathcal { A }$ might be randomized).

Lemma 38. Algorithm A has randomized description length of at most

$$
b \leq \frac {\log | \mathcal {X} | \cdot \log m}{\alpha^ {2}} \cdot \left(\log m + \log \frac {1}{\alpha}\right)
$$

bits.

Proof. We observe that for every fixing of the random bits of the data analyst the entire sequence of queries asked by the analyst, together with the answers he receives, can be reconstructed from a record of the indices i of the queries $\phi _ { i }$ such that $a _ { i } \neq a _ { i } ^ { \mathrm { p u b } }$ , together with their answers $a _ { i }$ (i.e. it is suficient to encode $M : = \{ ( i , a _ { i } ) \mid a _ { i } \neq a _ { i } ^ { \mathrm { p u b } } \}$ . Once this is established, the lemma follows because by Lemma $^ { 3 7 , }$ there are at most $\left( { \frac { \log | { \mathcal { X } } | \log m } { \alpha ^ { 2 } } } \right)$ such queries, and for each one, its index can be encoded with log m bits, and its answer with $\begin{array} { r } { \log { \frac { 1 } { \alpha } } } \end{array}$ bits.

To see why this is $\mathrm { s o } ,$ consider the following procedure for reconstructing the sequence $( \phi _ { 1 } , a _ { 1 } , \ldots , \phi _ { m } , a _ { m } )$ of queries asked and answers received. For every fixing of the random bits of the data analyst, her queries can be expressed as a sequence of functions $\left( f _ { 1 } , \ldots , f _ { m } \right)$ that take as input the queries previously asked to the Median Mechanism, and the answers previously received, and output the next query to be asked. That is, we have:

$$
f _ {1} () := \phi_ {1}, f _ {2} (\phi_ {1}, a _ {1}) := \phi_ {2}, f _ {3} (\phi_ {1}, a _ {1}, \phi_ {2}, a _ {2}) := \phi_ {3}, \dots , f _ {m} (\phi_ {1}, a _ {1}, \dots , \phi_ {m - 1}, a _ {m - 1}) := \phi_ {m}.
$$

Assume inductively that at stage i, the procedure has successfully reconstructed $\left( \phi _ { 1 } , a _ { 1 } , \ldots , \phi _ { i - 1 } , a _ { i - 1 } , \phi _ { i } \right)$ and the set $\mathrm { C o n s i s t e n t } _ { i - 1 }$ (This is trivially satisfied at stage $i = 1 )$ . For the inductive case, we need to recover $a _ { i } , \phi _ { i + 1 }$ , and Consistent<sub>i</sub>. There are two cases we must consider at stage i. In the first case, i is such that $a _ { i } \neq a _ { i } ^ { \mathrm { p u b } }$ . But in this case, $( i , a _ { i } ) \in M$ by definition, and so we have recovered ${ { a } _ { i } } ,$ and we can compute $\phi _ { i + 1 } = f _ { i + 1 } ( \phi _ { 1 } , a _ { 1 } , \dots , \phi _ { 1 } , a _ { i } )$ , and can compute Consisten $\mathrm { t } _ { i } = \left\{ S ^ { \prime } \in \mathrm { C o n s i s t e n t } _ { i - 1 } : \left. a _ { i } - \mathcal { E } _ { S ^ { \prime } } [ \phi _ { i } ] \right. \leq 2 \alpha \right\}$ . In the other case, $a _ { i } = a _ { i } ^ { \mathrm { p u b } }$ . But in this case, by definition of $a _ { i } ^ { \mathrm { p u b } }$ , we can compute $a _ { i } = \mathrm { m e d i a n } ( \{ \mathcal { E } _ { S ^ { \prime } } [ \phi _ { i } ]$ $S ^ { \prime } \in \operatorname { C o n s i s t e n t } _ { i - 1 } \} ) , \phi _ { i + 1 } = f _ { i + 1 } ( \phi _ { 1 } , a _ { 1 } , \dots , \phi _ { i } , a _ { i } )$ , and Conisiste $\mathrm { { \Phi } _ { { \mathrm { l } } t _ { i } } } = \mathrm { { C o n s i s t e n t } } _ { i - 1 }$ . This completes the argument – by induction, M is enough to reconstruct the entire query/answer sequence. □

Finally, we can complete the proof. By Hoefding’s concentration inequality and the union bound we know that for any every sequence of queries $\phi _ { 1 } , \ldots , \phi _ { m }$ and a dataset S of size n drawn from the distribution ${ \mathcal { P } } ^ { n }$

$$
\mathbb {P} \left[ \exists i, | \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] - \mathcal {P} [ \phi_ {i} ] | \geq \alpha \right] \leq 2 m \cdot \exp \left(- 2 n \alpha^ {2}\right).
$$

Applying Theorem 29 to the set $R ( \phi _ { 1 } , a _ { 1 } , \ldots , \phi _ { m } , a _ { m } ) = \{ S \mid \exists i , \ \vert { \mathcal { E } } _ { S } [ \phi _ { i } ] - { \mathcal { P } } [ \phi _ { i } ] \vert \geq \alpha \}$ we obtain that for the queries $\phi _ { 1 } , \ldots , \phi _ { m }$ generated on the dataset S and corresponding answers of the Median Mechanism $\pmb { a } _ { 1 } , \ldots , \pmb { a } _ { m }$ we have

$$
\begin{array}{r c l} \mathbb {P} \left[ \exists i, | \mathcal {E} _ {\boldsymbol {S}} [ \phi_ {i} ] - \mathcal {P} [ \phi_ {i} ] | \geq \alpha \right] & \leq & 2 ^ {b} \cdot 2 m \cdot \exp \left(- 2 n \alpha^ {2}\right). \\ & \leq & 2 ^ {\frac {\log | \mathcal {X} | \log m}{\alpha^ {2}} \cdot \log (m / \alpha)} \cdot 2 m \cdot \exp \left(- 2 n \alpha^ {2}\right). \end{array}
$$

Solving, we have that whenever:

$$
n \geq \frac {\log | \mathcal {X} | \cdot \log m}{2 \alpha^ {4}} \cdot \ln (m / \alpha) + \frac {\ln (2 m / \beta)}{2 \alpha^ {2}},
$$

we have: $\mathbb { P } \left[ \exists i , | { \mathcal { E } } _ { S } [ \phi _ { i } ] - { \mathcal { P } } [ \phi _ { i } ] | \geq \alpha \right] \leq \beta$ . Combining this with Lemma 36 we have:

$$
\mathbb {P} \left[ \exists i \in [ m ], | \boldsymbol {a} _ {i} - \mathcal {P} [ \phi_ {i} ] | \geq 3 \alpha \right] \leq \beta .
$$

Plugging in τ = 3α gives the theorem.
