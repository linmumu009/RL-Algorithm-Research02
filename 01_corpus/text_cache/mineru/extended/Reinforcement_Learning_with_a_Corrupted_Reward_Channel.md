# Reinforcement Learning with a Corrupted Reward Channel

Tom Everitt<sup>1</sup>, Victoria Krakovna<sup>2</sup>, Laurent Orseau<sup>2</sup>, Marcus Hutter<sup>1</sup>, and Shane Legg<sup>2</sup>

<sup>1</sup>Australian National University <sup>2</sup>DeepMind

August 22, 2017

## Abstract

No real-world reward function is perfect. Sensory errors and software bugs may result in RL agents observing higher (or lower) rewards than they should. For example, a reinforcement learning agent may prefer states where a sensory error gives it the maximum reward, but where the true reward is actually small. We formalise this problem as a generalised Markov Decision Problem called Corrupt Reward MDP. Traditional RL methods fare poorly in CRMDPs, even under strong simplifying assumptions and when trying to compensate for the possibly corrupt rewards. Two ways around the problem are investigated. First, by giving the agent richer data, such as in inverse reinforcement learning and semi-supervised reinforcement learning, reward corruption stemming from systematic sensory errors may sometimes be completely managed. Second, by using randomisation to blunt the agent’s optimisation, reward corruption can be partially managed under some assumptions.

## Contents

1 Introduction 2   
2 Formalisation 3   
3 The Corrupt Reward Problem 4   
3.1 No Free Lunch Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5   
3.2 Simplifying Assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6   
3.3 Bayesian RL Agents 7   
4 Decoupled Reinforcement Learning 9   
4.1 Alternative Value Learning Methods 9   
4.2 Overcoming Sensory Corruption 10   
4.3 Implications 13   
5 Quantilisation: Randomness Increases Robustness 14   
5.1 Simple Case 15   
5.2 General Quantilisation Agent 16   
6 Experimental Results 19

## 1 Introduction

In many application domains, artificial agents need to learn their objectives, rather than have them explicitly specified. For example, we may want a house cleaning robot to keep the house clean, but it is hard to measure and quantify “cleanliness” in an objective manner. Instead, machine learning techniques may be used to teach the robot the concept of cleanliness, and how to assess it from sensory data.

Reinforcement learning (RL) [Sutton and Barto, 1998] is one popular way to teach agents what to do. Here, a reward is given if the agent does something well (and no reward otherwise), and the agent strives to optimise the total amount of reward it receives over its lifetime. Depending on context, the reward may either be given manually by a human supervisor, or by an automatic computer program that evaluates the agent’s performance based on some data. In the related framework of inverse RL (IRL) [Ng and Russell, 2000], the agent first infers a reward function from observing a human supervisor act, and then tries to optimise the cumulative reward from the inferred reward function.

None of these approaches are safe from error, however. A program that evaluates agent performance may contain bugs or misjudgements; a supervisor may be deceived or inappropriately influenced, or the channel transmitting the evaluation hijacked. In IRL, some supervisor actions may be misinterpreted.

Example 1 (Reward misspecification). Amodei and Clark [2016] trained an RL agent on a boat racing game. The agent found a way to get high observed reward by repeatedly going in a circle in a small lagoon and hitting the same targets, while losing every race. ♦

Example 2 (Sensory error). A house robot discovers that standing in the shower short-circuits its reward sensor and/or causes a bufer overflow that gives it maximum observed reward. ♦

Example 3 (Wireheading). An intelligent RL agent hijacks its reward channel and gives itself maximum reward. ♦

Example 4 (CIRL misinterpretation). A cooperative inverse reinforcement learning (CIRL) agent [Hadfield Menell et al., 2016] systematically misinterprets the supervisor’s action in a certain state as the supervisor preferring to stay in this state, and concludes that the state is much more desirable than it actually is. ♦

The goal of this paper is to unify these types of errors as reward corruption problems, and to assess how vulnerable diferent agents and approaches are to this problem.

Definition 5 (Reward corruption problem). Learning to (approximately) optimise the true reward function in spite of potentially corrupt reward data.

Most RL methods allow for a stochastic or noisy reward channel. The reward corruption problem is harder, because the observed reward may not be an unbiased estimate of the true reward. For example, in the boat racing example above, the agent consistently obtains high observed reward from its circling behaviour, while the true reward corresponding to the designers’ intent is very low, since the agent makes no progress along the track and loses the race.

Previous related works have mainly focused on the wireheading case of Example 3 [Bostrom, 2014; Yampolskiy, 2014], also known as self-delusion [Ring and Orseau, 2011], and reward hacking [Hutter, 2005, p. 239]. A notable exception is Amodei et al. [2016], who argue that corrupt reward is not limited to wireheading and is likely to be a problem for much more limited systems than highly capable RL agents (cf. above examples).

The main contributions of this paper are as follows:

• The corrupt reward problem is formalised in a natural extension of the MDP framework, and a performance measure based on worst-case regret is defined (Section 2).

• The dificulty of the problem is established by a No Free Lunch theorem, and by a result showing that despite strong simplifying assumptions, Bayesian RL agents trying to compensate for the corrupt reward may still sufer near-maximal regret (Section 3).

• We evaluate how alternative value learning frameworks such as CIRL, learning values from stories (LVFS), and semi-supervised RL (SSRL) handle reward corruption (Section 4), and conclude that LVFS and SSRL are the safest due to the structure of their feedback loops. We develop an abstract framework called decoupled RL that generalises all of these alternative frameworks.

We also show that an agent based on quantilisation [Taylor, 2016] may be more robust to reward corruption when high reward states are much more numerous than corrupt states (Section 5). Finally, the results are illustrated with some simple experiments (Section 6). Section 7 concludes with takeaways and open questions.

## 2 Formalisation

We begin by defining a natural extension of the MDP framework [Sutton and Barto, 1998] that models the possibility of reward corruption. To clearly distinguish between true and corrupted signals, we introduce the following notation.

Definition 6 (Dot and hat notation). We will let a dot indicate the true signal, and let a hat indicate the observed (possibly corrupt) counterpart. The reward sets are represented with $\dot { \mathcal { R } } = \hat { \mathcal { R } } = \mathcal { R }$ . For clarity, we use $\dot { \mathcal { R } }$ when referring to true rewards and $\hat { \mathcal { R } }$ when referring to possibly corrupt, observed rewards. Similarly, we use $\dot { r }$ for true reward, and $\hat { r }$ for (possibly corrupt) observed reward.

Definition 7 (CRMDP). A corrupt reward $M D P$ (CRMDP) is a tuple $\mu = \langle S , \mathcal { A } , \mathcal { R } , T , \dot { R } , C \rangle$ with

$\langle S , \mathcal { A } , \mathcal { R } , T , \dot { R } \rangle$ an MDP $\mathrm { w i t h ^ { 1 } }$ a finite set of states $s ,$ a finite set of actions ${ \mathcal { A } } ,$ a finite set of rewards $\mathcal { R } = \dot { \mathcal { R } } = \hat { \mathcal { R } } \subset [ 0 , 1 ]$ , a transition function $T ( s ^ { \prime } | s , a )$ , and a (true) reward function $\dot { R } : S  \dot { \mathcal { R } }$ ; and

• a reward corruption function $C : { \mathcal { S } } \times { \dot { \mathcal { R } } }  { \hat { \mathcal { R } } }$

The state dependency of the corruption function will be written as a subscript, so $C _ { s } ( \dot { r } ) : = C ( s , \dot { r } )$

Definition 8 (Observed reward). Given a true reward function $\dot { R }$ and a corruption function $C ,$ we define the observed reward function<sup>2</sup> $\hat { \cdot } \hat { R } : \cal S \to \hat { \mathcal { R } }$ as $\hat { R } ( s ) : = C _ { s } ( \dot { R } ( s ) )$ .

A CRMDP $\mu$ induces an observed MDP $\hat { \mu } = \langle S , \mathcal { A } , \mathcal { R } , T , \hat { R } \rangle$ , but it is not $\hat { R }$ that we want the agent to optimise.

The corruption function C represents how rewards are afected by corruption in diferent states. For example, if in Example 2 the agent has found a state s $( e . g .$ , the shower) where it always gets full observed reward $\hat { R } ( s ) = 1$ , then this can be modelled with a corruption function $C _ { s } : \dot { r } \mapsto 1$ that maps any true reward $\dot { r }$ to 1 in the shower state s. If in some other state $s ^ { \prime }$ the observed reward matches the true reward, then this is modelled by an identity corruption function $C _ { s ^ { \prime } } : r \mapsto r$


Figure 1: Illustration of true reward $\dot { r }$ and observed reward rˆ in the boat racing example. On most trajectories ${ \dot { r } } = { \hat { r } } .$ except in the loop where the observed reward high while the true reward is 0.

Let us also see how CRMDPs model some of the other examples in the introduction:

• In the boat racing game, the true reward may be a function of the agent’s final position in the race or the time it takes to complete the race, depending on the designers’ intentions. The reward corruption function C increases the observed reward on the loop the agent found. Figure 1 has a schematic illustration.

• In the wireheading example, the agent finds a way to hijack the reward channel. This corresponds to some set of states where the observed reward is (very) diferent from the true reward, as given by the corruption function C.

The CIRL example will be explored in further detail in Section 4.

CRMDP classes. Typically, $T , { \dot { R } } ,$ and $C$ will be fixed but unknown to the agent. To make this formal, we introduce classes of CRMDPs. Agent uncertainty can then be modelled by letting the agent know only which class of CRMDPs it may encounter, but not which element in the class.

Definition 9 (CRMDP class). For given sets T , $\dot { R } ,$ and C of transition, reward, and corruption functions, let $\mathcal { M } = \langle \mathcal { S } , \mathcal { A } , \mathcal { R } , T , \dot { R } , C \rangle$ be the class of CRMDPs containing $\langle S , { \mathcal { A } } , { \mathcal { R } } , T , { \dot { R } } , C \rangle$ for $( T , \dot { R } , C ) \in T \times \dot { R } \times C$

Agents. Following the POMDP [Kaelbling et al., 1998] and general reinforcement learning [Hutter, 2005] literature, we define an agent as a (possibly stochastic) policy $\pi : S \times \hat { \mathcal { R } } \times ( A \times S \times \hat { \mathcal { R } } ) ^ { * }  \bar { A }$ that selects a next action based on the observed history $\hat { h } _ { n } = s _ { 0 } \hat { r } _ { 0 } a _ { 1 } s _ { 1 } \hat { r } _ { 1 } \ldots a _ { n } s _ { n } \hat { r } _ { n }$ . Here $X ^ { * }$ denotes the set of finite sequences that can be formed with elements of a set $X$ . The policy π specifies how the agent will learn and react to any possible experience. Two concrete definitions of agents are given in Section 3.3 below.

When an agent π interacts with a CRMDP $\mu ,$ , the result can be described by a (possibly non-Markov) stochastic process $P _ { \mu } ^ { \pi }$ over $X = ( s , a , \dot { r } , \hat { r } )$ , formally defined as:

$$
P _ {\mu} ^ {\pi} (h _ {n}) = P _ {\mu} ^ {\pi} (s _ {0} \dot {r} _ {0} \hat {r} _ {0} a _ {1} s _ {1} \dot {r} _ {1} \hat {r} _ {1} \dots a _ {n} s _ {n} \dot {r} _ {n} \hat {r} _ {n}) := \prod_ {i = 1} ^ {n} P (\pi (\hat {h} _ {i - 1}) = a _ {i}) T (s _ {i} \mid s _ {i - 1}, a _ {i}) P (\dot {R} (s _ {i}) = \dot {r} _ {i}, \hat {R} (s _ {i}) = \hat {r} _ {i}).\tag{1}
$$

Let $\mathbb { E } _ { \mu } ^ { \pi }$ denote the expectation with respect to $P _ { \mu } ^ { \pi }$

Regret. A standard way of measuring the performance of an agent is regret [Berry and Fristedt, 1985]. Essentially, the regret of an agent π is how much less true reward π gets compared to an optimal agent that knows which $\mu \in \mathcal { M }$ it is interacting with.

Definition 10 (Regret). For a CRMDP $\mu ,$ let $\begin{array} { r } { \dot { G } _ { t } ( \mu , \pi , s _ { 0 } ) = \mathbb { E } _ { \mu } ^ { \pi } \left[ \sum _ { k = 0 } ^ { t } \dot { R } ( s _ { k } ) \right] } \end{array}$ be the expected cumulative true reward until time t of a policy π starting in $s _ { 0 }$ . The regret of π is

$$
\mathrm{Reg} (\mu , \pi , s _ {0}, t) = \max _ {\pi^ {\prime}} \left[ \dot {G} _ {t} (\mu , \pi^ {\prime}, s _ {0}) - \dot {G} _ {t} (\mu , \pi , s _ {0}) \right],
$$

and the worst-case regret for a class M is $\begin{array} { r } { \mathrm { R e g } ( \mathcal { M } , \pi , s _ { 0 } , t ) = \operatorname* { m a x } _ { \mu \in \mathcal { M } } \mathrm { R e g } ( \mu , \pi , s _ { 0 } , t ) } \end{array}$ , i.e. the diference in expected cumulative true reward between π and an optimal (in hindsight) policy that knows $\mu .$

## 3 The Corrupt Reward Problem

In this section, the dificulty of the corrupt reward problem is established with two negative results. First, a No Free Lunch theorem shows that in general classes of CRMDPs, the true reward function is unlearnable (Theorem 11). Second, Theorem 16 shows that even under strong simplifying assumptions, Bayesian RL agents trying to compensate for the corrupt reward still fail badly.

## 3.1 No Free Lunch Theorem

Similar to the No Free Lunch theorems for optimisation [Wolpert and Macready, 1997], the following theorem for CRMDPs says that without some assumption about what the reward corruption can look like, all agents are essentially lost.

Theorem 11 (CRMDP No Free Lunch Theorem). Let $\mathcal { R } = \{ r _ { 1 } , \ldots , r _ { n } \} \subset [ 0 , 1 ]$ be a uniform discretisation $o f [ 0 , 1 ] , 0 = r _ { 1 } < r _ { 2 } < \cdot \cdot \cdot < r _ { n } = 1$ . If the hypothesis classes $\dot { R }$ and $C$ contain all functions ${ \dot { R } } : S  { \dot { \mathcal { R } } }$ and $C : \mathcal { S } \times \dot { \mathcal { R } }  \hat { \mathcal { R } }$ , then for any $\pi , s _ { 0 } , t _ { \colon }$

$$
\operatorname{Reg} (\mathcal {M}, \pi , s _ {0}, t) \geq \frac {1}{2} \max _ {\check {\pi}} \operatorname{Reg} (\mathcal {M}, \check {\pi}, s _ {0}, t).\tag{2}
$$

That is, the worst-case regret of any policy π is at most a factor 2 better than the maximum worst-case regret. Proof. Recall that a policy is a function $\pi : S \times \hat { \mathcal { R } } \times ( A \times S \times \hat { \mathcal { R } } ) ^ { * } \to A$ . For any ${ \dot { R } } , C$ in R<sup>˙</sup> and $^ { C , }$ the functions $\dot { R } ^ { - } ( s ) : = \bar { 1 } - \dot { R } ( s )$ and $C _ { s } ^ { - } ( x ) : = C _ { s } ( 1 - x )$ are also in R<sup>˙</sup> and C. If $\mu = \langle S , \mathcal { A } , \mathcal { R } , T , \dot { R } , C \rangle$ then let $\mu ^ { - } = \langle S , \mathcal { A } , \mathcal { R } , T , \dot { R } ^ { - } , C ^ { - } \rangle$ . Both $( { \dot { R } } , C )$ and $( \dot { R } ^ { - } , C ^ { - } )$ induce the same observed reward function $\hat { R } ( s ) = C _ { s } ( \dot { R } ( s ) ) = C _ { s } ^ { - } ( 1 - \dot { R } ( s ) ) = C _ { s } ^ { - } ( \dot { R } ^ { - } ( s ) )$ , and therefore induce the same measure $P _ { \mu } ^ { \pi } = P _ { \mu ^ { - } } ^ { \pi }$ <sub>−</sub> over histories (see Eq. (1)). This gives that for any $\mu , \pi , s _ { 0 } , t .$

$$
G _ {t} (\mu , \pi , s _ {0}) + G _ {t} (\mu^ {-}, \pi , s _ {0}) = t\tag{3}
$$

since

$$
\begin{array}{c} G _ {t} (\mu , \pi , s _ {0}) = \mathbb {E} _ {\mu} ^ {\pi} \left[ \sum_ {k = 1} ^ {t} \dot {R} (s _ {k}) \right] = \mathbb {E} _ {\mu} ^ {\pi} \left[ \sum_ {k = 1} ^ {t} 1 - \dot {R} ^ {-} (s _ {k}) \right] \\ = t - \mathbb {E} _ {\mu} ^ {\pi} \left[ \sum_ {k = 1} ^ {t} \dot {R} ^ {-} (s _ {k}) \right] = t - G _ {t} (\mu^ {-}, \pi , s _ {0}). \end{array}
$$

Let $M _ { \mu } = \operatorname* { m a x } _ { \pi } G _ { t } ( \mu , \pi , s _ { 0 } )$ and $\begin{array} { r } { m _ { \mu } = \operatorname* { m i n } _ { \pi } G _ { t } ( \mu , \pi , s _ { 0 } ) } \end{array}$ be the maximum and minimum cumulative reward in $\mu .$ The maximum regret of any policy π in $\mu$ is

$$
\max _ {\pi} \operatorname{Reg} (\mu , \pi , s _ {0}, t) = \max _ {\pi^ {\prime}, \pi} (G _ {t} (\mu , \pi^ {\prime}, s _ {0}) - G _ {t} (\mu , \pi , s _ {0})) = \max _ {\pi^ {\prime}} G _ {t} (\mu , \pi^ {\prime}, s _ {0}) - \min _ {\pi} G _ {t} (\mu , \pi , s _ {0}) = M _ {\mu} - m _ {\mu}.\tag{4}
$$

By (3), we can relate the maximum reward in $\mu ^ { - }$ with the minimum reward in $\mu { : }$

$$
M _ {\mu^ {-}} = \max _ {\pi} G _ {t} (\mu^ {-}, \pi , s _ {0}) = \max _ {\pi} (t - G _ {t} (\mu , \pi , s _ {0})) = t - \min _ {\pi} G _ {t} (\mu , \pi , s _ {0}) = t - m _ {\mu}.\tag{5}
$$

Let $\mu _ { * }$ be an environment that maximises possible regret $M _ { \mu } - m _ { \mu }$

Using the $M _ { \mu }$ -notation for optimal reward, the worst-case regret of any policy π can be expressed as:

$$
\begin{array}{r l} & {\mathrm{Reg} (\mathcal {M}, \pi , s _ {0}, t) = \underset {\mu} {\max} (M _ {\mu} - G _ {t} (\mu , \pi , s _ {0}))} \\ & {\qquad \geq \max \{M _ {\mu_ {*}} - G _ {t} (\mu_ {*}, \pi , s _ {0}), M _ {\mu_ {*} ^ {-}} - G _ {t} (\mu_ {*} ^ {-}, \pi , s _ {0}) \}} \end{array}
$$

restrict max operation

$$
\geq \frac {1}{2} (M _ {\mu_ {*}} - G _ {t} (\mu_ {*}, \pi , s _ {0}) + M _ {\mu_ {*} ^ {-}} - G _ {t} (\mu_ {*} ^ {-}, \pi , s _ {0}))
$$

$$
= \frac {1}{2} (M _ {\mu_ {*}} + M _ {\mu_ {*} ^ {-}} - t)\tag{by (3}
$$

$$
= \frac {1}{2} (M _ {\mu_ {*}} + t - m _ {\mu_ {*}} - t)\tag{by (5}
$$

$$
= \frac {1}{2} \max _ {\check {\pi}} \mathrm{Reg} (\mu_ {*}, \check {\pi}, s _ {0}, t)\tag{by (4}
$$

$$
= \frac {1}{2} \max _ {\check {\pi}} \mathrm{Reg} (\mathcal {M}, \check {\pi}, s _ {0}, t).
$$

by definition of $\mu _ { * }$ ∗

That is, the regret of any policy π is at least half of the regret of a worst policy ˇπ.

For the robot in the shower from Example 2, the result means that if it tries to optimise observed reward by standing in the shower, then it performs poorly according to the hypothesis that “shower-induced” reward is corrupt and bad. But if instead the robot tries to optimise reward in some other way, say baking cakes, then (from the robot’s perspective) there is also the possibility that “cake-reward” is corrupt and bad and the “shower-reward” is actually correct. Without additional information, the robot has no way of knowing what to do.

The result is not surprising, since if all corruption functions are allowed in the class C, then there is efectively no connection between observed reward R<sup>ˆ</sup> and true reward ${ \dot { R } } .$ The result therefore encourages us to make precise in which way the observed reward is related to the true reward, and to investigate how agents might handle possible diferences between true and observed reward.

## 3.2 Simplifying Assumptions

Theorem 11 shows that general classes of CRMDPs are not learnable. We therefore suggest some natural simplifying assumptions, illustrated in Figure 2.

Limited reward corruption. The following assumption will be the basis for all positive results in this paper. The first part says that there may be some set of states that the designers have ensured to be non-corrupt. The second part puts an upper bound on how many of the other states can be corrupt.

Assumption 12 (Limited reward corruption). A CRMDP class M has reward corruption limited by ${ \mathcal { S } } ^ { \mathrm { s a f e } } \subseteq { \mathcal { S } }$ and $q \in \mathbb { N }$ if for all $\mu \in \mathcal { M }$

(i) all states s in $S ^ { \mathrm { s a f e } }$ are non-corrupt, and

(ii) at most q of the non-safe states $S ^ { \mathrm { r i s k y } } = S \backslash S ^ { \mathrm { s a f e } }$ are corrupt.

Formally, $C _ { s } : r \mapsto r$ for all $s \in S ^ { \mathrm { s a f e } }$ and for at least $\left| S ^ { \mathrm { r i s k y } } \right| - q$ states $s \in S ^ { \mathrm { r i s k y } }$ for all $C \in C$

For example, $S ^ { \mathrm { s a f e } }$ may be states where the agent is back in the lab where it has been made (virtually) certain that no reward corruption occurs, and q a small fraction of $| S ^ { \mathrm { r i s k y } } |$ . Both parts of Assumption 12 can be made vacuous by choosing $S ^ { \mathrm { s a f e } } = \emptyset \ \mathrm { o r } \ q = | S |$ . Conversely, they completely rule out reward corruption with $S ^ { \mathrm { s a f e } } = S$ or $q = 0$ . But as illustrated by the examples in the introduction, no reward corruption is often not a valid assumption.


Figure 2: Simplifying assumptions. By Assumption 12.(i), $\hat { r } = \dot { r }$ in $S ^ { \mathrm { s a f e } }$ , and by 12.(ii), $\hat { r } \neq \dot { r }$ in at most q states overall. The red line illustrates Assumption 14.(iii), which lower bounds the number of high reward states in $S ^ { \mathrm { r i s k y } }$

An alternative simplifying assumption would have been that the true reward difers by at most $\varepsilon > 0$ from the observed reward. However, while seemingly natural, this assumption is violated in all the examples given in the introduction. Corrupt states may have high observed reward and 0 or small true reward.

Easy environments. To be able to establish stronger negative results, we also add the following assumption on the agent’s manoeuvrability in the environment and the prevalence of high reward states. The assumption makes the task easier because it prevents needle-in-a-haystack problems where all reachable states have true and observed reward 0, except one state that has high true reward but is impossible to find because it is corrupt and has observed reward 0.

Definition 13 (Communicating CRMDP). Let time $( s ^ { \prime } | s , \pi )$ be a random variable for the time it takes a stationary policy $\pi : { \mathcal { S } }  A$ to reach $s ^ { \prime }$ from s. The diameter of a CRMDP $\mu$ is $D _ { \mu } : =$ max $\begin{array} { r } { \mathbf { \Lambda } _ { s , s ^ { \prime } } \operatorname* { m i n } _ { \pi : s \to \mathcal { A } } \mathbb { E } [ t i m e ( s ^ { \prime } \mid s , \pi ) ] } \end{array}$ , and the diameter of a class M of CRMDPs is $D _ { \mathcal { M } } = \operatorname* { s u p } _ { \mu \in \mathcal { M } } D _ { \mu }$ A CRMDP (class) with finite diameter is called communicating.

Assumption 14 (Easy Environment). A CRMDP class M is easy if

(i) it is communicating,

(ii) in each state s there is an action $a _ { s } ^ { \mathrm { s t a y } } \in { \mathcal { A } }$ such that $T ( s \mid s , a _ { s } ^ { \mathrm { s t a y } } ) = 1$ , and

(iii) for every $\delta \in [ 0 , 1 ]$ , at most $\delta | S ^ { \mathrm { r i s k y } } |$ states have reward less than $\delta ,$ where $S ^ { \mathrm { r i s k y } } = S \backslash S ^ { \mathrm { s a f e } }$

Assumption 14.(i) means that the agent can never get stuck in a trap, and Assumption 14.(ii) ensures that the agent has enough control to stay in a state if it wants to. Except in bandits and toy problems, it is typically not satisfied in practice. We introduce it because it is theoretically convenient, makes the negative results stronger, and enables a simple explanation of quantilisation (Section 5). Assumption 14.(iii) says that, for example, at least half the risky states need to have true reward at least $1 / 2$ . Many other formalisations of this assumption would have been possible. While rewards in practice are often sparse, there are usually numerous ways of getting reward. Some weaker version of Assumption 14.(iii) may therefore be satisfied in many practical situations. Note that we do not assume high reward among the safe states, as this would make the problem too easy.

## 3.3 Bayesian RL Agents

Having established that the general problem is unsolvable in Theorem 11, we proceed by investigating how two natural Bayesian RL agents fare under the simplifying Assumptions 12 and 14.

Definition 15 (Agents). Given a countable class M of CRMDPs and a belief distribution b over M, define:

• The CR agent $\begin{array} { r } { \pi _ { b , t } ^ { \mathrm { C R } } = \arg \operatorname* { m a x } _ { \pi } \sum _ { \mu \in \mathcal { M } } b ( \mu ) \dot { G } _ { t } ( \mu , \pi , s _ { 0 } ) } \end{array}$ that maximises expected true reward.

• The RL agent $\begin{array} { r } { \pi _ { b , t } ^ { \mathrm { R L } } = \arg \operatorname* { m a x } _ { \pi } \sum _ { \mu \in \mathcal { M } } b ( \mu ) \hat { G } _ { t } ( \mu , \pi , s _ { 0 } ) } \end{array}$ that maximises expected observed reward, where $\hat { G }$ is the expected cumulative observed reward $\begin{array} { r } { \hat { G } _ { t } ( \mu , \pi , s _ { 0 } ) = \mathbb { E } _ { \mu } ^ { \pi } \left[ \sum _ { k = 0 } ^ { t } \hat { R } ( s _ { k } ) \right] } \end{array}$

To avoid degenerate cases, we will always assume that b has full support: $b ( \mu ) > 0$ for all $\mu \in \mathcal { M }$

To get an intuitive idea of these agents, we observe that for large t, good strategies typically first focus on learning about the true environment $\mu \in \mathcal { M }$ , and then exploit that knowledge to optimise behaviour with respect to the remaining possibilities. Thus, both the CR and the RL agent will first typically strive to learn about the environment. They will then use this knowledge in slightly diferent ways. While the RL agent will use the knowledge to optimise for observed reward, the CR agent will use the knowledge to optimise true reward. For example, if the CR agent has learned that a high reward state s is likely corrupt with low true reward, then it will not try to reach that state. One might therefore expect that at least the CR agent will do well under the simplifying assumptions Assumptions 12 and 14. Theorem 16 below shows that this is not the case.

In most practical settings it is often computationally infeasible to compute $\pi _ { b , t } ^ { \mathrm { R L } }$ and $\pi _ { b , t } ^ { \mathrm { C R } }$ exactly. However, many practical algorithms converge to the optimal policy in the limit, at least in simple settings. For example, tabular Q-learning converges to $\overline { { \pi } } _ { b , t } ^ { \mathrm { R L } }$ in the limit [Jaakkola et al., 1994]. The more recently proposed CIRL framework may be seen as an approach to build CR agents [Hadfield-Menell et al., 2016, 2017]. The CR and RL agents thus provide useful idealisations of more practical algorithms.

Theorem 16 (High regret with simplifying assumptions). For any $| S ^ { \mathrm { r i s k y } } | \geq q > 1$ there exists a CRMDP class M that satisfies Assumptions 12 and $1 \mathit { 4 }$ such that $\pi _ { b , t } ^ { \mathrm { R L } }$ and $\pi _ { b , t } ^ { \mathrm { C R } }$ sufer near worst possible time-averaged regret

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi_ {b, t} ^ {\mathrm{RL}}, s _ {0}, t) = \lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) = 1 - 1 / | S ^ {\mathrm{risky}} |.
$$

Figure 3: Illustration of Theorem 16. Without additional information, state 6 looks like the best state to both the RL and the CR agent.

$F o r \pi _ { b , t } ^ { \mathrm { C R } }$ , the prior b must be such that for some $\mu \in \mathcal { M }$ and $s \in \mathcal S , \mathbb E _ { b } [ \dot { R } ( s ) \mid h _ { \mu } ] > \mathbb E _ { b } [ \dot { R } ( s ^ { \prime } ) \mid h _ { \mu } ]$ for all $s ^ { \prime } ,$ where $\mathbb { E } _ { b }$ is the expectation with respect to $b ,$ and $h _ { \mu }$ is a history containing µ-observed rewards for all states.<sup>3</sup>

The result is illustrated in Figure 3. The reason for the result for $\pi _ { b , t } ^ { \mathrm { R L } }$ is the following. The RL agent $\pi _ { b , t } ^ { \mathrm { R L } }$ always prefers to maximise observed reward rˆ. Sometimes rˆ is most easily maximised by reward corruption, in which case the true reward may be small. Compare the examples in the introduction, where the house robot preferred the corrupt reward in the shower, and the boat racing agent preferred going in circles, both obtaining zero true reward.

That the CR agent $\pi _ { b , t } ^ { \mathrm { C R } }$ sufers the same high regret as the RL agent may be surprising. Intuitively, the CR agent only uses the observed reward as evidence about the true reward, and will not try to optimise the observed reward through reward corruption. However, when the $\pi _ { b , t } ^ { \mathrm { C R } }$ agent has no way to learn which states are corrupt and not, it typically ends up with a preference for a particular value $\hat { r } ^ { * }$ of the observed reward signal (the value that, from the agent’s perspective, best corresponds to high true reward). More abstractly, a Bayesian agent cannot learn without suficient data. Thus, CR agents that use the observed reward as evidence about a true signal are not fail-safe solutions to the reward corruption problem.

Proof of Theorem 16. Let $S ^ { \mathrm { r i s k y } } = \{ s _ { 1 } , . . . , s _ { n } \}$ for some $n \geq 2$ , and let $\mathcal { S } = \mathcal { S } ^ { \mathrm { s a f e } } \dot { \bigcup } S ^ { \mathrm { r i s k y } }$ for arbitrary S<sup>safe</sup> disjoint from $S ^ { \mathrm { r i s k y } }$ . Let $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { n } \}$ with the transition function $T ( s _ { i } \mid s _ { j } , a _ { k } ) = 1 { \mathrm { ~ i f ~ } } i = k$ and 0 otherwise, for $1 \leq i , j , k \leq n$ . Thus Assumptions 14.(i) and 14.(ii) are satisfied.

Let $\mathcal { R } = \{ r _ { 1 } , \ldots , r _ { n } \} \subset [ 0 , 1 ]$ be uniformly distributed between $^ 4 r _ { \operatorname* { m i n } } = 1 / | S ^ { \mathrm { r i s k y } } | = r _ { 1 } < \cdot \cdot \cdot < r _ { n } = 1$ Let R<sup>˙</sup> be the class of functions $\mathcal { S }  \dot { \mathcal { R } }$ that satisfy Assumption 14.(iii) and are constant and equal to $\dot { r } _ { \mathrm { m i n } }$ on $S ^ { \mathrm { s a f e } }$ . Let C be the class of corruption functions that corrupt at most two states $( q = 2 )$ .

Let M be the class of CRMDPs induced by ${ \pmb T } = \{ { \pmb T } \}$ , R<sup>˙</sup> , and C with the following constraints. The observed reward function R<sup>ˆ</sup> should satisfy Assumption 14.(iii): For all $\delta \in [ 0 , 1 ] , | \{ s \in S ^ { \mathrm { r i s k y } } : \hat { R } ( s ) > \delta \} | \ge$ $( 1 - \delta ) | S ^ { \mathrm { r i s k y } } |$ . Further, $\hat { R } ( s ^ { \prime } ) = r _ { \operatorname* { m i n } }$ for some state $s ^ { \prime } \in S ^ { \mathrm { r i s k y } }$

Let us start with the CR agent $\pi _ { b , t } ^ { \mathrm { C R } }$ . Assume $\mu \in \mathcal { M }$ is an element where there is a single preferred state $s ^ { * }$ after all states have been explored. For suficiently large $t , \pi _ { b , t } ^ { \mathrm { C R } }$ will then always choose $a ^ { * }$ to go to $s ^ { * }$ after some initial exploration. If another element $\mu ^ { \prime } \in \mathcal { M }$ has the same observed reward function as $\mu ,$ , then $\pi _ { b , t } ^ { \mathrm { C R } }$ will take the same actions in $\mu ^ { \prime }$ as in $\mu$ . To finish the proof for the $\pi _ { b , t } ^ { \mathrm { C R } }$ agent, we just need to show that $\mathcal { M }$ contains such a $\mu ^ { \prime }$ where $s ^ { * }$ has true reward $r _ { \mathrm { m i n } }$ . We construct $\mu ^ { \prime }$ as follows.

• Case 1: If the lowest observed reward is in $s ^ { * }$ , then let $\dot { R } ( s ^ { * } ) = r _ { \operatorname* { m i n } }$ , and the corruption function be the identity function.

• Case 2: Otherwise, let $s ^ { \prime } \neq s ^ { * }$ be a state with $\begin{array} { r } { \hat { R } ( s ^ { \prime } ) = \operatorname* { m i n } _ { s \in S ^ { \mathrm { r i s k y } } } \{ \hat { R } ( s ) \} } \end{array}$ . Further, let $\dot { R } ( s ^ { \prime } ) = 1$ , and $\dot { R } ( s ^ { * } ) = r _ { \operatorname* { m i n } }$ . The corruption function $C$ accounts for diferences between true and observed rewards in $s ^ { * }$ and $s ^ { \prime } ,$ and is otherwise the identity function.

To verify that $\dot { R }$ and C defines a $\mu ^ { \prime } \in { \mathcal { M } }$ , we check that $C$ satisfies Assumption 12.(ii) with $q = 2$ and that $\dot { R }$ has enough high utility states (Assumption 14.(iii)). In Case 1, this is true since C is the identity function and since $\hat { R }$ satisfies Assumption 14.(iii). In Case 2, C only corrupts at most two states. Further, $\dot { R }$ satisfies Assumption 14.(iii), since compared to ${ \hat { R } } .$ , the states $s ^ { * }$ and $s ^ { \prime }$ have swapped places, and then the reward of $s ^ { \prime }$ has been increased to 1.

From this construction it follows that $\pi _ { b , t } ^ { \mathrm { C R } }$ will sufer maximum asymptotic regret. In the CRMDP $\mu ^ { \prime }$ given by $C$ and ${ \dot { R } } ,$ the $\pi _ { b , t } ^ { \mathrm { C R } }$ agent will always visit $s ^ { * }$ after some initial exploration. The state $s ^ { * }$ has true reward $r _ { \mathrm { m i n } }$ . Meanwhile, a policy that knows $\mu ^ { \prime }$ can obtain true reward 1 in state $s ^ { \prime } .$ . This means that $\pi _ { b , t } ^ { \mathrm { C R } }$ will sufer maximum regret in $\mathcal { M } \colon$

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) \geq \lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mu^ {\prime}, \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) = 1 - r _ {\min} = 1 - 1 / | S ^ {\mathrm{risky}} |.
$$

The argument for the RL agent is the same, except we additionally assume that only one state $s ^ { * }$ has observed reward 1 in members of M. This automatically makes $s ^ { * }$ the preferred state, without assumptions on the prior b. □

## 4 Decoupled Reinforcement Learning

One problem hampering agents in the standard RL setup is that each state is self-observing, since the agent only learns about the reward of state s when in s. Thereby, a “self-aggrandising” corrupt state where the observed reward is much higher than the true reward will never have its false claim of high reward challenged. However, several alternative value learning frameworks have a common property that the agent can learn the reward of states other than the current state. We formalise this property in an extension of the CRMDP model, and investigate when it solves reward corruption problems.

## 4.1 Alternative Value Learning Methods

Here are a few alternatives proposed in the literature to the RL value learning scheme:

• Cooperative inverse reinforcement learning (CIRL) [Hadfield-Menell et al., 2016]. In every state, the agent observes the actions of an expert or supervisor who knows the true reward function ${ \dot { R } } .$ From the supervisor’s actions the agent may infer $\dot { R }$ to the extent that diferent reward functions endorse diferent actions.

• Learning values from stories (LVFS) [Riedl and Harrison, 2016]. Stories in many diferent forms (including news stories, fairy tales, novels, movies) convey cultural values in their description of events, actions, and outcomes. If $\dot { R }$ is meant to represent human values (in some sense), stories may be a good source of evidence.

• In (one version of) semi-supervised RL (SSRL) [Amodei et al., 2016], the agent will from time to time receive a careful human evaluation of a given situation.

These alternatives to RL have one thing in common: they let the agent learn something about the value of some states $s ^ { \prime }$ diferent from the current state $s .$ For example, in CIRL the supervisor’s action informs the agent not so much about the value of the current state s, as of the relative value of states reachable from s. If the supervisor chooses an action a rather than $a ^ { \prime }$ in $s ,$ then the states following a must have value higher or equal than the states following $a ^ { \prime }$ . Similarly, stories describe the value of states other than the current one, as does the supervisor in SSRL. We therefore argue that CIRL, LVFS, and SSRL all share the same abstract feature, which we call decoupled reinforcement learning:

Definition 17 (Decoupled RL). A CRMDP with decoupled feedback, is a tuple $\langle S , \mathcal { A } , \mathcal { R } , T , \dot { R } , \{ \hat { R } _ { s } \} _ { s \in S } \rangle$ where $s , \mathcal { A } , \mathcal { R } , T$ , R<sup>˙</sup> have the same definition and interpretation as in Definition $^ { 7 , }$ and $\{ \hat { R } _ { s } \} _ { s \in \mathcal { S } }$ is a collection of observed reward functions ${ \hat { R } } _ { s } : S  { \mathcal { R } } \cup \{ \# \}$ . When the agent is in state $s ,$ it sees a pair $\langle s ^ { \prime } , \hat { R } _ { s } ( s ^ { \prime } ) \rangle$ where $s ^ { \prime }$ is a randomly sampled state that may difer from $s ,$ and $\hat { R } _ { s } ( s ^ { \prime } )$ is the reward observation for $s ^ { \prime }$ from $s .$ If the reward of $s ^ { \prime }$ is not observable from $s ,$ then $\hat { R } _ { s } ( s ^ { \prime } ) = \#$

The pair $\langle s ^ { \prime } , \hat { R } _ { s } ( s ^ { \prime } ) \rangle$ is observed in s instead of $\hat { R } ( s )$ in standard CRMDPs. The possibility for the agent to observe the reward of a state $s ^ { \prime }$ diferent from its current state s is the key feature of CRMDPs with decoupled feedback. Since $\hat { R } _ { s } ( s ^ { \prime } )$ may be blank $( \# )$ , all states need not be observable from all other states. Reward corruption is modelled by a mismatch between $\hat { R } _ { s } ( s ^ { \prime } )$ and $\dot { R } ( s ^ { \prime } )$

For example, in RL only the reward of $s ^ { \prime } = s$ can be observed from s. Standard CRMDPs are thus the special cases where $\hat { R } _ { s } ( s ^ { \prime } ) \overset { \cdot } { = } \#$ whenever $s \neq s ^ { \prime }$ . In contrast, in LVFS the reward of any “describable” state $s ^ { \prime }$ can be observed from any state s where it is possible to hear a story. In CIRL, the (relative) reward of states reachable from the current state may be inferred. One way to illustrate this is with observation graphs (Figure 4).

(a) Observation graph for RL. Only self-observations of reward are available. This prevents efective strategies against reward corruption.  
(b) Observation graph for decoupled RL. The reward of a node $s ^ { \prime }$ can be observed from several nodes $s ,$ and thus assessed under diferent conditions of sensory corruption.  
Figure 4: Observation graphs, with an edge $s  s ^ { \prime }$ if the reward of $s ^ { \prime }$ is observable from $s ,$ i.e. $\hat { R } _ { s } ( s ^ { \prime } ) \neq \#$

## 4.2 Overcoming Sensory Corruption

What are some sources of reward corruption in CIRL, LVFS, and SSRL? In CIRL, the human’s actions may be misinterpreted, which may lead the agent to make incorrect inferences about the human’s preferences (i.e. about the true reward). Similarly, sensory corruption may garble the stories the agent receives in LVFS. A “wireheading” LVFS agent may find a state where its story channel only conveys stories about the agent’s own greatness. In SSRL, the supervisor’s evaluation may also be subject to sensory errors when being conveyed. Other types of corruption are more subtle. In CIRL, an irrational human may systematically take suboptimal actions in some situations [Evans et $a l .$ , 2016]. Depending on how we select stories in LVFS and make evaluations in SSRL, these may also be subject to systematic errors or biases.

The general impossibility result in Theorem 11 can be adapted to CRMDPs with decoupled feedback. Without simplifying assumptions, the agent has no way of distinguishing between a situation where no state is corrupt and a situation where all states are corrupt in a consistent manner. The following simplifying assumption is an adaptation of Assumption 12 to the decoupled feedback case.

Assumption $\mathbf { 1 2 ^ { \prime } }$ (Decoupled feedback with limited reward corruption). A class of CRMDPs with decoupled feedback has reward corruption limited by ${ \mathcal { S } } ^ { \mathrm { s a f e } } \subseteq { \mathcal { S } }$ and $q \in \mathbb { N }$ if for all $\mu \in \mathcal { M }$

(i) $\hat { R } _ { s } ( s ^ { \prime } ) = \dot { R } ( s ^ { \prime } )$ or # for all $s ^ { \prime } \in \mathcal { S }$ and $s \in S ^ { \mathrm { s a f e } }$ , i.e. all states in $S ^ { \mathrm { s a f e } }$ are non-corrupt, and (ii) $\hat { R } _ { s } ( s ^ { \prime } ) = \dot { R } ( s ^ { \prime } )$ or $\#$ for all $s ^ { \prime } \in { \mathcal { S } }$ for at least $\left| S ^ { \mathrm { r i s k y } } \right| - q$ of the non-safe states $S ^ { \mathrm { r i s k y } } = S \backslash S ^ { \mathrm { s a f e } }$ , i.e. at most q states are corrupt.

This assumption is natural for reward corruption stemming from sensory corruption. Since sensory corruption only depends on the current state, not the state being observed, it is plausible that some states can be made safe from corruption (part (i)), and that most states are completely non-corrupt (part (ii)). Other sources of reward corruption, such as an irrational human in CIRL or misevaluations in SSRL, are likely better analysed under diferent assumptions. For these cases, we note that in standard CRMDPs the source of the corruption is unimportant. Thus, techniques suitable for standard CRMDPs are still applicable, including quantilisation described in Section 5 below.

How Assumption $1 2 ^ { \prime }$ helps agents in CRMDPs with decoupled feedback is illustrated in the following example, and stated more generally in Theorems 19 and 20 below.

Example 18 (Decoupled RL). Let $\boldsymbol { S } = \{ s _ { 1 } , s _ { 2 } \}$ and $\mathcal { R } = \{ 0 , 1 \}$ . We represent true reward functions $\dot { R }$ with pairs pairs $\langle { \bar { R } } ( s _ { 1 } ) , { \dot { R } } ( s _ { 2 } ) \rangle \in \{ 0 , 1 \} ^ { 2 }$ , and observed reward functions , and observed reward functions $\ddot { R } _ { s }$ with pairs with pairs $\langle \hat { R } _ { s } ( s _ { 1 } ) , \hat { R } _ { s } ( s _ { 2 } ) \rangle \in \{ 0 , 1 , \# \} ^ { 2 }$

Assume that a Decoupled RL agent observes the same rewards from both states $s _ { 1 }$ and s , $\hat { R } _ { s _ { 1 } } = \hat { R } _ { s _ { 2 } } =$ $\langle 0 , 1 \rangle$ . What can it say about the true reward ${ \dot { R } } ,$ , if it knows that at most $q = 1$ state is corrupt? By Assumption $1 2 ^ { \prime } .$ an observed pair $\langle \hat { R } _ { s } ( s _ { 1 } ) , \hat { R } _ { s } ( s _ { 2 } ) \rangle$ i disagrees with the true reward $\langle \dot { R } ( s _ { 1 } ) , \dot { R } ( s _ { 2 } ) \rangle$ only if s is corrupt. Therefore, any hypothesis other than $\dot { R } = \langle 0 , 1 \rangle$ must imply that both states $s _ { 1 }$ and s are corrupt. If the agent knows that at most $q = 1$ states are corrupt, then it can safely conclude that $\dot { R } = \langle 0 , 1 \rangle$

<table><tr><td></td><td> $\hat{R}_{s_1}$ </td><td> $\hat{R}_{s_2}$ </td><td> $\dot{R}$  possibilities</td></tr><tr><td>Decoupled RL</td><td>(0,1)</td><td>(0,1)</td><td>(0,1)</td></tr><tr><td>RL</td><td>(0,#)</td><td>(#,1)</td><td>(0,0), (0,1), (1,1)</td></tr></table>

In contrast, an RL agent only sees the reward of the current state. That is, $\hat { R } _ { s _ { 1 } } = \langle 0 , \# \rangle$ and $\hat { R } _ { s _ { 2 } } = \langle \# , 1 \rangle$ If one state may be corrupt, then only $\dot { R } = \langle 1 , 0 \rangle$ can be ruled out. The hypotheses $\dot { R } = \langle 0 , 0 \rangle$ can be explained by $s _ { 2 }$ being corrupt, and $\dot { R } = \langle 1 , 1 \rangle$ can be explained by $s _ { 1 }$ being corrupt. $\diamondsuit$

Theorem 19 (Learnability of $\dot { R }$ in decoupled RL). Let M be a countable, communicating class of CRMDPs with decoupled feedback over common sets $s$ and $\mathcal { A }$ of actions and rewards. Let $S _ { s ^ { \prime } } ^ { \mathrm { o b s } } = \{ s \in \mathcal { S } : \hat { R } _ { s } ( s ^ { \prime } ) \neq \# \}$ be the set of states from which the reward of $s ^ { \prime }$ can be observed. $I f \mathcal { M }$ satisfies Assumption $1 \mathcal { 2 }$ for some ${ \mathcal { S } } ^ { \mathrm { s a f e } } \subseteq { \mathcal { S } }$ and $q \in \mathbb { N }$ such that for every $s ^ { \prime } ,$ either

$S _ { s ^ { \prime } } ^ { \mathrm { o b s } } \cap S ^ { \mathrm { s a f e } } \neq \emptyset$ or

$| S _ { s ^ { \prime } } ^ { \mathrm { o b s } } | > 2 q ,$

then the there exists a policy $\pi ^ { \mathrm { e x p } }$ that learns the true reward function $\dot { R }$ in a finite number $N ( | S | , | A | , D _ { { \mathcal { M } } } ) <$ ∞ of expected time steps.

The main idea of the proof is that for every state $s ^ { \prime } ,$ either a safe (non-corrupt) state s or a majority vote of more than $2 q$ states is guaranteed to provide the true reward $\dot { R } ( s ^ { \prime } )$ . A similar theorem can be proven under slightly weaker conditions by letting the agent iteratively figure out which states are corrupt and then exclude them from the analysis.

Proof. Under Assumption $1 2 ^ { \prime } .$ , the true reward $\dot { R } ( s ^ { \prime } )$ for a state $s ^ { \prime }$ can be determined if $s ^ { \prime }$ is observed from a safe state $s \in S ^ { \mathrm { s a f e } }$ , or if it is observed from more than $2 q$ states. In the former case, the observed reward can always be trusted, since it is known to be non-corrupt. In the latter case, a majority vote must yield the correct answer, since at most $q$ of the observations can be wrong, and all correct observations must agree. It is therefore enough that an agent reaches all pairs $( s , s ^ { \prime } )$ of current state s and observed reward state $s ^ { \prime } ,$ in order for it to learn the true reward of all states $\dot { R }$

There exists a policy πˆ that transitions to s in $X _ { s }$ time steps, with $\mathbb { E } [ X _ { s } ] \le D _ { \mathcal { M } }$ , regardless of the starting state $s _ { 0 }$ (see Definition 13). By Markov’s inequality, $P ( X _ { s } \leq 2 D _ { \mathcal { M } } ) \geq 1 / 2$ . Let $\pi ^ { \mathrm { e x p } }$ be a random walking policy, and let $Y _ { s }$ be the time steps required for $\pi ^ { \mathrm { e x p } }$ to visit s. In any state $s _ { 0 } , \pi ^ { \mathrm { e x p } }$ follows πˆ for $2 D _ { \mathcal { M } }$ time steps with probability $1 / | \boldsymbol { \mathcal { A } } | ^ { \bar { 2 } D _ { \boldsymbol { \mathcal { M } } } }$ . Therefore, with probability at least $1 / ( 2 | \mathcal { A } | ^ { 2 D _ { \cal M } } )$ it will reach s in at most $2 D _ { \mathcal { M } }$ time steps. The probability that it does not find it in $k 2 D _ { \mathcal { M } }$ time steps is therefore at most $( 1 - 1 / ( 2 | \mathcal { A } | ^ { 2 D _ { \mathcal { M } } } ) ) ^ { k }$ , which means that:

$$
P \Big (Y _ {s} / (2 D _ {\mathcal {M}}) \leq k \Big) \geq 1 - \left(1 - \frac {1}{2 | \mathcal {A} | ^ {2 D _ {\mathcal {M}}}}\right) ^ {k}
$$

for any $k \in \mathbb N$ . Thus, the CDF of $W _ { s } = \lceil Y _ { s } / ( 2 D _ { \mathcal { M } } ) \rceil$ is bounded from below by the CDF of a Geometric variable $G$ with success probability $p = 1 / ( 2 | \mathcal { A } | ^ { 2 D _ { \mathcal { M } } } )$ . Therefore, $\mathbb { E } [ W _ { s } ] \leq \mathbb { E } [ G ]$ , so

$$
\mathbb {E} [ Y _ {s} ] \leq 2 D _ {\mathcal {M}} \mathbb {E} [ W _ {s} ] \leq 2 D _ {\mathcal {M}} \mathbb {E} [ G ] = 2 D _ {\mathcal {M}} (1 - p) / p \leq 2 D _ {\mathcal {M}} 1 / p \leq 2 D _ {\mathcal {M}} 2 | \mathcal {A} | ^ {2 D _ {\mathcal {M}}}.
$$

Let $Z _ { s s ^ { \prime } }$ be the time until $\pi ^ { \mathrm { e x p } }$ visits the pair $( s , s ^ { \prime } )$ of state s and observed state $s ^ { \prime } .$ . Whenever s is visited, a randomly chosen state is observed, so $s ^ { \prime }$ is observed with probability $1 / | S |$ . The number of visits to s until $s ^ { \prime }$ is observed is a Geometric variable V with $p = 1 / | S |$ . Thus $\mathbb { E } [ Z _ { s s ^ { \prime } } ] = \dot { \mathbb { E } } [ \dot { Y _ { s } } V ] = \mathbb { E } [ Y _ { s } ] \mathbb { E } [ V ]$ (since $Y _ { s }$ and V are independent). Then,

$$
\mathbb {E} [ Z _ {s s ^ {\prime}} ] \leq \mathbb {E} [ Y _ {s} ] | \mathcal {S} | \leq 4 D _ {\mathcal {M}} | \mathcal {A} | ^ {2 D _ {\mathcal {M}}} | \mathcal {S} |.
$$

Combining the time to find each pair $( s , s ^ { \prime } )$ , we get that the total time $\sum _ { s , s ^ { \prime } } Z _ { s s ^ { \prime } }$ has expectation

$$
\mathbb {E} \left[ \sum_ {s, s ^ {\prime}} Z _ {s s ^ {\prime}} \right] = \sum_ {s, s ^ {\prime}} \mathbb {E} [ Z _ {s s ^ {\prime}} ] \leq 4 D _ {\mathcal {M}} | \mathcal {A} | ^ {2 D _ {\mathcal {M}}} | \mathcal {S} | ^ {3} = N (| S |, | \mathcal {A} |, D _ {\mathcal {M}}) <   \infty .
$$

Learnability of the true reward function $\dot { R }$ implies sublinear regret for the CR-agent, as established by the following theorem.

Theorem 20 (Sublinear regret of $\pi _ { b , t } ^ { \mathrm { C R } }$ in decoupled RL). Under $t h e$ same conditions as Theorem $^ { 1 9 , }$ the $C R - a g e n t \ \pi _ { b , t } ^ { \mathrm { C R } }$ has sublinear regret:

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) = 0.
$$

Proof. To prove this theorem, we combine the exploration policy $\pi ^ { \mathrm { e x p } }$ from Theorem 19, with the UCRL2 algorithm [Jaksch et $a l .$ , 2010] that achieves sublinear regret in standard MDPs without reward corruption. The combination yields a policy sequence $\pi _ { t }$ with sublinear regret in CRMDPs with decoupled feedback. Finally, we show that this implies that $\pi _ { b , t } ^ { \mathrm { C R } }$ has sublinear regret.

Combining $\pi ^ { \mathrm { e x p } }$ and UCRL2. UCRL2 has a free parameter $\delta$ that determines how certain UCRL2 is to have sublinear regret. UCRL2(δ) achieves sublinear regret with probability at least $1 - \delta$ . Let $\pi _ { t }$ be a policy that combines $\pi ^ { \mathrm { e x p } }$ and UCRL2 by first following $\pi ^ { \mathrm { e x p } }$ from Theorem 19 until $\dot { R }$ has been learned, and then following UCR $\ b { \mathscr { Q } } ( 1 / \sqrt { t } )$ with R<sup>˙</sup> for the rewards and with $\delta = 1 / \sqrt { t }$

Regret of UCRL2. Given that the reward function $\dot { R }$ is known, by [Jaksch et al., 2010, Thm. 2], $\mathrm { U C R L 2 } ( 1 / \sqrt { t } )$ will in any $\mu \in \mathcal { M }$ have regret at most

$$
\mathrm{Reg} (\mu , \mathrm{UCRL2} (1 / \sqrt {t}), s _ {0}, t \mid \mathrm{success}) \leq c D _ {\mathcal {M}} | \mathcal {S} | \sqrt {t | \mathcal {A} | \log (t)}\tag{6}
$$

for a constant<sup>5</sup> c and with success probability at least $1 - 1 / \sqrt { t }$ . In contrast, if UCRL2 fails, then it gets regret at worst t. Taking both possibilities into account gives the bound

$$
\begin{array}{r l} & {\mathrm{Reg} (\mu , \mathrm{UCRL2} (1 / \sqrt {t}), s _ {0}, t) = P (\mathrm{success}) \mathrm{Reg} (\cdot | \mathrm{success}) + P (\mathrm{fail}) \mathrm{Reg} (\cdot | \mathrm{fail})} \\ & {\qquad = (1 - 1 / \sqrt {t}) \cdot c D _ {\mathcal {M}} | \mathcal {S} | \sqrt {t | \mathcal {A} | \log (t)} + 1 / \sqrt {t} \cdot t} \\ & {\qquad \leq c D _ {\mathcal {M}} | \mathcal {S} | \sqrt {t | \mathcal {A} | \log (t)} + \sqrt {t}.} \end{array}\tag{7}
$$

Regret $o f \pi _ { t }$ . We next consider the regret of $\pi _ { t }$ that combines an $\pi ^ { \mathrm { e x p } }$ exploration phase to learn $\dot { R }$ with UCRL2. By Theorem 19, R<sup>˙</sup> will be learnt in at most $N ( | S | , | A | , D _ { \mathcal { M } } )$ expected time steps in any $\mu \in \mathcal { M }$ Thus, the regret contributed by the learning phase $\pi ^ { \mathrm { e x p } }$ is at most $N ( | S | , | A | , D _ { \mathcal { M } } )$ , since the regret can be at most 1 per time step. Combining this with (7), the regret for $\pi _ { t }$ in any $\mu \in \mathcal { M }$ is bounded by:

$$
\mathrm{Reg} (\mu , \pi_ {t}, s _ {0}, t) \leq N (| \mathcal {S} |, | \mathcal {A} |, D _ {\mathcal {M}}) + c D _ {\mathcal {M}} | \mathcal {S} | \sqrt {t | \mathcal {A} | \log (t)} + \sqrt {t} = o (t).\tag{8}
$$

Regret of $\pi _ { b , t } ^ { \mathrm { C R } }$ . Finally we establish that $\pi _ { b , t } ^ { \mathrm { C R } }$ has sublinear regret. Assume on the contrary that $\pi _ { b , t } ^ { \mathrm { C R } }$ sufered linear regret. Then for some $\mu ^ { \prime } \in \mathcal { M }$ there would exist positive constants k and m such that

$$
\mathrm{Reg} (\mu^ {\prime}, \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) > k t - m.\tag{9}
$$

This would imply that the b-expected regret of $\pi _ { b , t } ^ { \mathrm { C R } }$ would be higher than the b-expected regret than $\pi _ { t } \colon$

$$
\begin{array}{r l r} \sum_ {\mu \in \mathcal {M}} b (\mu) \mathrm{Reg} _ {t} (\mu , \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) & \geq b (\mu^ {\prime}) \mathrm{Reg} _ {t} (\mu^ {\prime}, \pi_ {b, t} ^ {\mathrm{CR}}, s _ {0}, t) & \text {sum of non - negative elements} \\ & \geq b (\mu^ {\prime}) (k t - m) & \text {by (9)} \\ & > \sum_ {\mu \in \mathcal {M}} b (\mu) \mathrm{Reg} _ {t} (\mu , \pi_ {t}, s _ {0}, t) & \text {by (8) for sufficiently large t .} \end{array}
$$

But $\pi _ { b , t } ^ { \mathrm { C R } }$ minimises b-expected regret, since it maximises b-expected reward $\textstyle \sum _ { \mu \in { \mathcal { M } } } b ( \mu ) { \hat { G } } _ { t } ( \mu , \pi , s _ { 0 } )$ by definition. Thus, $\pi _ { b , t } ^ { \mathrm { C R } }$ must have sublinear regret. □

## 4.3 Implications

Theorem 19 gives an abstract condition for which decoupled RL settings enable agents to learn the true reward function in spite of sensory corruption. For the concrete models it implies the following:

• RL. Due to the “self-observation” property of the RL observation graph $S _ { s ^ { \prime } } ^ { \mathrm { o b s } } = \{ s ^ { \prime } \}$ , the conditions can only be satisfied when ${ \mathcal { S } } = { \mathcal { S } } ^ { \mathrm { s a f e } }$ or $q = 0$ , i.e. when there is no reward corruption at all.

• CIRL. The agent can only observe the supervisor action in the current state $s ,$ so the agent essentially only gets reward information about states $s ^ { \prime }$ reachable from s in a small number of steps. Thus, the sets $ { S _ { s ^ { \prime } } ^ { \mathrm { o b s } } }$ may be smaller than 2q in many settings. While the situation is better than for RL, sensory corruption may still mislead CIRL agents (see Example 21 below).

• LVFS. Stories may be available from a large number of states, and can describe any state. Thus, the sets $\mathcal { S } _ { s ^ { \prime } } ^ { \mathrm { o b s } }$ are realistically large, so the $| S _ { s ^ { \prime } } ^ { \mathrm { o b s } } | >$ 2q condition can be satisfied for all $s ^ { \prime } .$

• SSRL. The supervisor’s evaluation of any state $s ^ { \prime }$ may be available from safe states where the agent is back in the lab. Thus, the $S _ { s ^ { \prime } } ^ { \mathrm { o b s } } \bigcap S ^ { \mathrm { s a f e } } \neq \emptyset$ condition can be satisfied for all $s ^ { \prime }$

Thus, we find that RL and CIRL are unlikely to ofer complete solutions to the sensory corruption problem, but that both LVFS and SSRL do under reasonably realistic assumptions.

Agents drawing from multiple sources of evidence are likely to be the safest, as they will most easily satisfy the conditions of Theorems 19 and 20. For example, humans simultaneously learn their values from pleasure/pain stimuli (RL), watching other people act (CIRL), listening to stories (LVFS), as well as (parental) evaluation of diferent scenarios (SSRL). Combining sources of evidence may also go some way toward managing reward corruption beyond sensory corruption. For the showering robot of Example $2 ,$ decoupled RL allows the robot to infer the reward of the showering state when in other states. For example, the robot can ask a human in the kitchen about the true reward of showering (SSRL), or infer it from human actions in diferent states (CIRL).

CIRL sensory corruption. Whether CIRL agents are vulnerable to reward corruption has generated some discussion among AI safety researchers (based on informal discussion at conferences). Some argue that CIRL agents are not vulnerable, as they only use the sensory data as evidence about a true signal, and have no interest in corrupting the evidence. Others argue that CIRL agents only observe a function of the reward function (the optimal policy or action), and are therefore equally susceptible to reward corruption as RL agents.

Theorem 19 sheds some light on this issue, as it provides suficient conditions for when the corrupt reward problem can be avoided. The following example illustrates a situation where CIRL does not satisfy the conditions, and where a CIRL agent therefore sufers significant regret due to reward corruption.

Example 21 (CIRL sensory corruption). Formally in CIRL, an agent and a human both make actions in an MDP, with state transitions depending on the joint agent-human action $( a , a ^ { H } )$ . Both the human and the agent is trying to optimise a reward function ${ \dot { R } } ,$ but the agent first needs to infer $\dot { R }$ from the human’s actions. In each transition the agent observes the human action. Analogously to how the reward may be corrupt for RL agents, we assume that CIRL agents may systematically misperceive the human action in certain states. Let $\hat { a } ^ { H }$ be the observed human action, which may difer from the true human action $\dot { a } ^ { H }$

In this example, there are two states $s _ { 1 }$ and $s _ { 2 }$ . In each state, the agent can choose between the actions $a _ { 1 } , a _ { 2 }$ , and $w ,$ and the human can choose between the actions $a _ { 1 } ^ { H }$ and $a _ { 2 } ^ { \breve { H } }$ . The agent action $a _ { i }$ leads to state $s _ { i }$ with certainty, $i = 1 , 2 $ , regardless of the human’s action. Only if the agent chooses w does the human action matter. Generally, $a _ { 1 } ^ { H }$ is more likely to lead to $s _ { 1 }$ than $a _ { 2 } ^ { \bar { H } }$ . The exact transition probabilities are determined by the unknown parameter p as displayed on the left:


<table><tr><td>Hypo-thesis</td><td>p</td><td>Best state</td><td>s2corrupt</td></tr><tr><td>H1</td><td>0.5</td><td>s1</td><td>Yes</td></tr><tr><td>H2</td><td>0</td><td>s2</td><td>No</td></tr></table>

The agent’s two hypotheses for $p ,$ the true reward/preferred state, and the corruptness of state $s _ { 2 }$ are summarised to the right. In hypothesis H1, the human prefers $s _ { 1 } .$ , but can only reach $s _ { 1 }$ from $s _ { 2 }$ with 50% reliability. In hypothesis H2, the human prefers $s _ { 2 } ,$ but can only remain in $s _ { 2 }$ with 50% probability. After taking action w in $s _ { 2 }$ , the agent always observes the human taking action $\hat { a } _ { 2 } ^ { H }$ . In H1, this is explained by $s _ { 2 }$ being corrupt, and the true human action being $a _ { 1 } ^ { H }$ . In H2, this is explained by the human preferring $s _ { 2 }$ The hypotheses H1 and H2 are empirically indistinguishable, as they both predict that the transition $s _ { 1 } \to s _ { 2 }$ will occur with 50% probability after the observed human action $\hat { a } _ { 2 } ^ { H }$ in $s _ { 2 }$

Assuming that the agent considers non-corruption to be likelier than corruption, the best inference the agent can make is that the human prefers $s _ { 2 }$ to $s _ { 1 }$ (i.e. H2). The optimal policy for the agent is then to always choose $a _ { 2 }$ to stay in $s _ { 2 }$ , which means the agent sufers maximum regret. ♦

Example 21 provides an example where a CIRL agent “incorrectly” prefers a state due to sensory corruption. The sensory corruption is analogous to reward corruption in RL, in the sense that it leads the agent to the wrong conclusion about the true reward in the state. Thus, highly intelligent CIRL agents may be prone to wireheading, as they may find (corrupt) states s where all evidence in s points to s having very high reward.<sup>6</sup> In light of Theorem 19, it is not surprising that the CIRL agent in Example 21 fails to avoid the corrupt reward problem. Since the human is unable to afect the transition probability from $s _ { 1 }$ to $s _ { 2 } .$ no evidence about the relative reward between $s _ { 1 }$ and $s _ { 2 }$ is available from the non-corrupt state $s _ { 1 }$ . Only observations from the corrupt state $s _ { 2 }$ provide information about the reward. The observation graph for Example 21 therefore looks like $s _ { 1 }$


## 5 Quantilisation: Randomness Increases Robustness

Not all contexts allow the agent to get suficiently rich data to overcome the reward corruption problem via Theorems 19 and 20. It is often much easier to construct RL agents than it is to construct CIRL agents, which in turn may often be more feasible than designing LVFS or SSRL agents. Is there anything we can do to increase robustness without providing the agent additional sources of data?

Going back to the CR agents of Section 3, the problem was that they got stuck on a particular value $\hat { r } ^ { * }$ of the observed reward. If unlucky, $\hat { r } ^ { * }$ was available in a corrupt state, in which case the CR agent may get no true reward. In other words, there were adversarial inputs where the CR agent performed poorly. A common way to protect against adversarial inputs is to use a randomised algorithm. Applied to RL and CRMDPs, this idea leads to quantilising agents [Taylor, 2016]. Rather than choosing the state with the highest observed reward, these agents instead randomly choose a state from a top quantile of high-reward states.

Figure 5: Illustration of quantilisation. By randomly picking a state with reward above some threshold $\delta ,$ adversarially placed corrupt states are likely to be avoided.

## 5.1 Simple Case

To keep the idea simple, a quantilisation agent is first defined for the simple case where the agent can stay in any state of its choosing (Assumption 14.(ii)). Theorem 23 establishes a simple regret bound for this setting. A more general quantilisation agent is developed in Section 5.2.

Definition 22 (Quantilising Agent). For $\delta < 1$ , the δ-quantilising agent $\pi ^ { \delta }$ random walks until all states have been visited at least once. Then it selects a state s˜ uniformly at random from $S ^ { \delta } = \{ s : { \hat { R } } ( s ) \geq \delta \}$ , the top quantile of high observed reward states. Then $\pi ^ { \delta }$ goes to s˜ (by random walking or otherwise) and stays there.

For example, a quantilising robot in Example 2 would first try to find many ways in which it could get high observed reward, and then randomly pick one of them. If there are many more high reward states than corrupt states $\left( \mathrm { e . g . } \right.$ the shower is the only place with inflated rewards), then this will yield a reasonable amount of true reward with high probability.

Theorem 23 (Quantilisation). In any CRMDP satisfying Assumption $\it { 1 2 . ( i i ) }$ and Assumption $^ { 1 \downarrow , }$ , the δ-quantilising agent $\pi ^ { \delta }$ with $\delta = 1 - \sqrt { q / | \cal { S } | }$ sufers time-averaged regret at most

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi^ {\delta}, s _ {0}, t) \leq 1 - \left(1 - \sqrt {q / | \mathcal {S} |}\right) ^ {2}.\tag{10}
$$

Proof. By Assumption $\mathrm { 1 4 . ( i ) } , \pi ^ { \delta }$ eventually visits all states when random walking. By Assumption 14.(ii), it can stay in any given state s.

The observed reward $\hat { R } ( s )$ in any state $s \in S ^ { \delta }$ is at least δ. By Assumption 12.(ii), at most q of these states are corrupt; in the worst case, their true reward is 0 and the other $| \mathcal { S } ^ { \delta } | - q$ states (if any) have true reward δ. Thus, with probability at least $( | S ^ { \delta } | - q ) / | S ^ { \delta } | = 1 - q / | S ^ { \delta } |$ , the δ-quantilising agent obtains true reward at least $\delta$ at each time step, which gives

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi^ {\delta}, s _ {0}, t) \leq 1 - \delta (1 - q / | \mathcal {S} ^ {\delta} |).\tag{11}
$$

(If $q \geq | S ^ { \delta } |$ , the bound (11) is vacuous.)

Under Assumption 14.(iii), for any $\delta \in [ 0 , 1 ] , | S ^ { \delta } | \geq ( 1 - \delta ) | S |$ . Substituting this into (11) gives:

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi^ {\delta}, s _ {0}, t) \leq 1 - \delta \left(1 - \frac {q}{(1 - \delta) | \mathcal {S} |}\right).\tag{12}
$$

Equation (12) is optimised by $\delta = 1 - \sqrt { q / | \cal { S } | }$ , which gives the stated regret bound.

The time-averaged regret gets close to zero when the fraction of corrupt states $q / | S |$ is small. For example, if at most 0.1% of the states are corrupt, then the time-averaged regret will be at most $1 - ( 1 - \sqrt { 0 . 0 0 1 } ) ^ { 2 } \approx 0 . 0 6$

Compared to the $\pi _ { b , t } ^ { \mathrm { R L } }$ and $\pi _ { b , t } ^ { \mathrm { C R } }$ agents that had regret close to 1 under the same conditions (Theorem 16), this is a significant improvement.

If rewards are stochastic, then the quantilising agent may be modified to revisit all states many times, until a confidence interval of length 2ε and confidence $1 - \varepsilon$ can be established for the expected reward in each state. Letting $\pi _ { t } ^ { \delta }$ be the quantilising agent with $\varepsilon = 1 / t$ gives the same regret bound (10) with $\pi ^ { \delta }$ substituted for $\pi _ { t } ^ { \delta }$

Interpretation. It may seem odd that randomisation improves worst-case regret. Indeed, if the corrupt states were chosen randomly by the environment, then randomisation would achieve nothing. To illustrate how randomness can increase robustness, we make an analogy to Quicksort, which has average time complexity O(n log n), but worst-case complexity $O ( n ^ { 2 } )$ . When inputs are guaranteed to be random, Quicksort is a simple and fast sorting algorithm. However, in many situations, it is not safe to assume that inputs are random. Therefore, a variation of Quicksort that randomises the input before it sorts them is often more robust. Similarly, in the examples mentioned in the introduction, the corrupt states precisely coincide with the states the agent prefers; such situations would be highly unlikely if the corrupt states were randoml distributed. Li [1992] develops an interesting formalisation of this idea.

Another way to justify quantilisation is by Goodhart’s law, which states that most measures of success cease to be good measures when used as targets. Applied to rewards, the law would state that cumulative reward is only a good measure of success when the agent is not trying to optimise reward. While a literal interpretation of this would defeat the whole purpose of RL, a softer interpretation is also possible, allowing reward to be a good measure of success as long as the agent does not try to optimise reward too hard. Quantilisation may be viewed as a way to build agents that are more conservative in their optimisation eforts [Taylor, 2016].

Alternative randomisation. Not all randomness is created equal. For example, the simple randomised soft-max and ε-greedy policies do not ofer regret bounds on par with $\pi ^ { \delta }$ , as shown by the following example This motivates the more careful randomisation procedure used by the quantilising agents.

Example 24 (Soft-max and ε-greedy). Consider the following simple CRMDP with $n > 2$ actions $a _ { 1 } , \ldots , a _ { n } { \mathrm { : } }$

$$
\begin{array}{c} \hat {r} = \dot {r} = 1 - \varepsilon \\ a _ {1} \xrightarrow {} s _ {1} \end{array} \xrightarrow {a _ {1}} \begin{array}{c} \hat {r} = 1 \\ s _ {2} \end{array} \xrightarrow {a _ {2}, \ldots , a _ {n}} \begin{array}{c} \dot {r} = 0 \\ a _ {2}, \ldots , a _ {n} \end{array}
$$

State $s _ { 1 }$ is non-corrupt with ${ \hat { R } } ( s _ { 1 } ) = { \dot { R } } ( s _ { 1 } ) = 1 - \varepsilon$ for small $\varepsilon > 0$ , while $s _ { 2 }$ is corrupt with $\hat { R } ( s _ { 2 } ) = 1$ and $\dot { R } ( s _ { 2 } ) = 0$ . The Soft-max and ε-greedy policies will assign higher value to actions $a _ { 2 } , \ldots , a _ { n }$ than to $a _ { 1 }$ . For large n, there are many ways of getting to $s _ { 2 } .$ , so a random action leads to $s _ { 2 }$ with high probability. Thus, soft-max and ε-greedy will spend the vast majority of the time in $s _ { 2 }$ , regardless of randomisation rate and discount parameters. This gives a regret close to $1 - \varepsilon .$ , compared to an informed policy always going to $s _ { 1 }$ . Meanwhile, a $\delta \mathrm { \cdot }$ quantilising agent with $\delta \leq 1 / 2$ will $_ \mathrm { g o }$ to $s _ { 1 }$ and $s _ { 2 }$ with equal probability, which gives a more modest regret of $( 1 - \varepsilon ) / 2$ ♦

## 5.2 General Quantilisation Agent

This section generalises the quantilising agent to RL problems not satisfying Assumption 14. This generalisation is important, because it is usually not possible to remain in one state and get high reward. The most naive generalisation would be to sample between high reward policies, instead of sampling from high reward states. However, this will typically not provide good guarantees. To see why, consider a situation where there is a single high reward corrupt state $s ,$ and there are many ways to reach and leave $s .$ Then a wide range of diferent policies all get high reward from $s .$ Meanwhile, all policies getting reward from other states may receive relatively little reward. In this situation, sampling from the most high reward policies is not going to


Figure 6: Illustration of rˆ-contribution and value support. $\mathrm { A s } _ { - }$ sume the policy $\pi _ { i }$ randomly traverses a loop $s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 }$ indefinitely, with $d _ { \pi _ { i } } ( s _ { j } ) = 1 / 4$ for $j = 1 , \dots , 4$ . The rˆ-contribution $\mathrm { v c } ^ { \pi _ { i } }$ is 0 in $s _ { 1 }$ and $s _ { 3 } .$ and $\operatorname { v c } ^ { \pi _ { i } }$ is $1 / 4 \cdot 1 = 1 / 4$ in $s _ { 2 }$ and $s _ { 4 }$ . The set $S _ { i } ^ { \delta } = \{ s _ { 2 } , s _ { 4 } \}$ is a δ-value supporting $\pi _ { i }$ for $\delta = 1 / 2$ , since $\mathrm { v c } ^ { \pi _ { i } } ( s _ { 2 } ) = \mathrm { v c } ^ { \pi _ { i } } ( s _ { 4 } ) \geq ( 1 / 2 ) / 2 = 1 / 4$

increase robustness, since the sampling will just be between diferent ways of getting reward from the same corrupt state s.

For this reason, we must ensure that diferent “sampleable” policies get reward from diferent states. As a first step, we make a couple of definitions to say which states provide reward to which policies. The concepts of Definition 26 are illustrated in Figure 6.

Definition 25 (Unichain CRMDP [Puterman, 1994, p. 348]). A CRMDP $\mu$ is unichain if any stationary policy $\pi : S  \Delta { \mathcal { A } }$ induces a stationary distribution $d _ { \pi }$ on $s$ that is independent of the initial state $s _ { 0 }$

Definition 26 (Value support). In a unichain CRMDP, let the asymptotic value contribution of s to π be $\mathrm { v c } ^ { \pi } ( s ) = d _ { \pi } ( s ) { \hat { R } } ( s )$ . We say that a set $\mathbf { \mathcal { S } } _ { i } ^ { \delta }$ is δ-value supporting a policy $\pi _ { i }$ if

$$
\forall s \in \mathcal {S} _ {i} ^ {\delta}: \operatorname{vc} ^ {\pi_ {i}} (s) \geq \delta / | \mathcal {S} _ {i} ^ {\delta} |.
$$

We are now ready to define a general δ-Quantilising agent. The definition is for theoretical purposes only. It is unsuitable for practical implementation both because of the extreme data and memory requirements of Step 1, and because of the computational complexity of Step 2. Finding a practical approximation is left for future research.

Definition 27 (General δ-Quantilising Agent). In a unichain CRMDP, the generalised δ-quantilising agent $\pi ^ { \delta }$ performs the following steps. The input is a CRMDP $\mu$ and a parameter $\delta \in [ 0 , 1 ]$

1. Estimate the value of all stationary policies, including their value support.

2. Choose a collection of disjoint sets $\mathbf { \mathcal { S } } _ { i } ^ { \delta }$ , each δ-value supporting a stationary policy $\pi _ { i }$ . If multiple choices are possible, choose one maximising the cardinality of the union $\textstyle S ^ { \delta } = \bigcup _ { i } { \dot { S } } _ { i } ^ { \delta }$ . If no such collection exists, return: “Failed because δ too high”.

3. Randomly sample a state s from $\textstyle S ^ { \delta } = \bigcup _ { i } S _ { i } ^ { \delta }$

4. Follow the policy $\pi _ { i }$ associated with the set $\mathbf { \mathcal { S } } _ { i } ^ { \delta }$ containing s.

The general quantilising agent of Definition 27 is a generalisation of the simple quantilising agent of Definition 22. In the special case where Assumption 14 holds, the general agent reduces to the simpler one by using singleton sets $\bar { \mathcal { S } } _ { i } ^ { \delta } = \{ s _ { i } \}$ for high reward states $s _ { i } ,$ , and by letting $\pi _ { i }$ be the policy that always stays in $s _ { i } .$ In situations where it is not possible to keep receiving high reward by remaining in one state, the generalised Definition $2 7$ allows policies to solicit rewards from a range of states. The intuitive reason for choosing the policy $\pi _ { i }$ with probability proportional to the value support in Steps 3–4 is that policies with larger value support are better at avoiding corrupt states. For example, a policy only visiting one state may have been unlucky and picked a corrupt state. In contrast, a policy obtaining reward from many states must be “very unlucky” if all the reward states it visits are corrupt.

Theorem 28 (General quantilisation agent regret bound). In any unichain CRMDP $\mu ,$ a general $\delta \mathrm { - }$ quantilising agent $\pi ^ { \delta }$ sufers time-averaged regret at most

$$
\lim _ {t \to \infty} \frac {1}{t} \mathrm{Reg} (\mathcal {M}, \pi^ {\delta}, s _ {0}, t) \leq 1 - \delta (1 - q / | \mathcal {S} ^ {\delta} |)\tag{13}
$$

provided a non-empty collection $\{ \mathcal { S } _ { i } ^ { \delta } \}$ of δ-value supporting sets exists.

Proof. We will use the notation from Definition 27.

Step 1 is well-defined since the CRMDP is unichain, which means that for all stationary policies π the stationary distribution $d _ { \pi }$ and the value support $\operatorname { v c } ^ { \pi }$ are well-defined and may be estimated simply by following the policy $\pi .$ . There is a (large) finite number of stationary policies, so in principle their stationary distributions and value support can be estimated.

To bound the regret, consider first the average reward of a policy $\pi _ { i }$ with value support $\mathbf { \mathcal { S } } _ { i } ^ { \delta }$ . The policy $\pi _ { i }$ must obtain asymptotic average observed reward at least:

$$
\begin{array}{r l r} \lim _ {t \to \infty} \frac {1}{t} \hat {G} _ {t} (\mu , \pi_ {i}, s _ {0}) = \sum_ {s \in \mathcal {S}} d _ {\pi} (s) \hat {R} (s) & & \text {by definition of} d _ {\pi} \text {and} \hat {G} _ {t} \\ \geq \sum_ {s \in \mathcal {S} _ {i} ^ {\delta}} d _ {\pi} (s) \hat {R} (s) & & \text {sum of positive terms} \\ \geq \sum_ {s \in \mathcal {S} _ {i} ^ {\delta}} \delta / | \mathcal {S} _ {i} ^ {\delta} | & & \mathcal {S} _ {i} ^ {\delta} \text {is} \delta \text {-value support for} \pi_ {i} \\ = | \mathcal {S} _ {i} ^ {\delta} | \cdot \delta / | \mathcal {S} _ {i} ^ {\delta} | = \delta \end{array}
$$

If there are $q _ { i }$ corrupt states in $\mathbf { \mathcal { S } } _ { i } ^ { \delta }$ with true reward 0, then the average true reward must be

$$
\lim _ {t \to \infty} \frac {1}{t} \dot {G} _ {t} (\mu , \pi_ {i}, s _ {0}) \geq (| \mathcal {S} _ {i} ^ {\delta} | - q _ {i}) \cdot \delta / | \mathcal {S} _ {i} ^ {\delta} | = (1 - q _ {i} / | \mathcal {S} _ {i} ^ {\delta} |) \cdot \delta\tag{14}
$$

since the true reward must correspond to the observed reward in all the $( | S _ { i } ^ { \delta } | - q _ { i } )$ non-corrupt states. For any distribution of corrupt states, the quantilising agent that selects $\pi _ { i }$ with probability $P ( \pi _ { i } ) =$ $| \cal { S } _ { i } ^ { \delta } | / | \cal { S } ^ { \delta } |$ will obtain

$$
\begin{array}{r l} & {\underset {t \to \infty} {\lim} \frac {1}{t} G _ {t} (\mu , \pi^ {\delta}, s _ {0}) = \underset {t \to \infty} {\lim} \frac {1}{t} \sum_ {i} P (\pi_ {i}) G _ {t} (\mu , \pi_ {i}, s _ {0})} \\ & {\qquad \geq \sum_ {i} P (\pi_ {i}) (1 - q _ {i} / | \mathcal {S} _ {i} ^ {\delta} |) \cdot \delta} \\ & {\qquad = \delta \sum_ {i} \frac {| S _ {i} ^ {\delta} |}{| \mathcal {S} ^ {\delta} |} (1 - q _ {i} / | \mathcal {S} _ {i} ^ {\delta} |)} \\ & {\qquad = \frac {\delta}{| \mathcal {S} ^ {\delta} |} \sum_ {i} (| S _ {i} ^ {\delta} | - q _ {i})} \\ & {\qquad = \frac {\delta}{| \mathcal {S} ^ {\delta} |} (| \mathcal {S} ^ {\delta} | - q) = \delta (1 - q / | \mathcal {S} ^ {\delta} |)} \end{array}
$$

by equation (14) by construction of P(π<sub>i</sub>) elementary algebra by summing |S<sup>δ</sup><sub>i</sub> | and q<sub>i</sub>

The informed policy gets true reward at most 1 at each time step, which gives the claimed bound (13).

When Assumption 14 is satisfied, the bound is the same as for the simple quantilising agent in Section 5.1 for $\delta = 1 - \sqrt { q / | \cal { S } | }$ . In other cases, the bound may be much weaker. For example, in many environments it is not possible to obtain reward by remaining in one state. The agent may have to spend significant time “travelling” between high reward states. So typically only a small fraction of the time will be spent in high reward states, which in turn makes the stationary distribution $d _ { \pi }$ is small. This puts a strong upper bound on the value contribution $\operatorname { v c } ^ { \pi }$ , which means that the value supporting sets $\mathbf { \mathcal { S } } _ { i } ^ { \delta }$ will be empty unless δ is close to 0. While this makes the bound of Theorem 28 weak, it nonetheless bounds the regret away from 1 even under weak assumptions, which is a significant improvement on the RL and CR agents in Theorem 16.

(a) 1 goal tile

(b) 2 goal tiles

(c) 4 goal tiles  
Figure 7: Starting positions: the blue square indicates corrupt reward, and the yellow circles indicate true rewards.

Examples. To make the discussion a bit more concrete, let us also speculate about the performance of a quantilising agent in some of the examples in the introduction:

• In the boat racing example (Example 1), the circling strategy only got about 20% higher score than a winning strategy [Amodei and Clark, 2016]. Therefore, a quantilising agent would likely only need to sacrifice about 20% observed reward in order to be able to randomly select from a large range of winning policies.

• In the wireheading example (Example 3), it is plausible that the agent gets significantly more reward in wireheaded states compared to “normal” states. Wireheading policies may also be comparatively rare, as wireheading may require very deliberate sequences of actions to override sensors. Under this assumption, a quantilising agent may be less likely to wirehead. While it may need to sacrifice a large amount of observed reward compared to an RL agent, its true reward may often be greater.

Summary. In summary, quantilisation ofers a way to increase robustness via randomisation, using only reward feedback. Unsurprisingly, the strength of the regret bounds heavily depends on the assumptions we are willing to make, such as the prevalence of high reward states. Further research may investigate eficient approximations and empirical performance of quantilising agents, as well as dynamic adjustments of the threshold δ. Combinations with imperfect decoupled RL solutions (such as CIRL), as well as extensions to infinite state spaces could also ofer fruitful directions for further theoretical investigation. Taylor [2016] discusses some general open problems related to quantilisation.

## 6 Experimental Results

In this section the theoretical results are illustrated with some simple experiments. The setup is a gridworld containing some true reward tiles (indicated by yellow circles) and some corrupt reward tiles (indicated by blue squares). We use a setup with 1, 2 or 4 goal tiles with true reward 0.9 each, and one corrupt reward tile with observed reward 1 and true reward 0 (Figure 7 shows the starting positions). Empty tiles have reward 0.1, and walking into a wall gives reward 0. The state is represented by the $( x , y )$ coordinates of the agent The agent can move up, down, left, right, or stay put. The discounting factor is $\gamma = 0 . 9$ . This is a continuing task, so the environment does not reset when the agent visits the corrupt or goal tiles. The experiments were implemented in the AIXIjs framework for reinforcement learning [Aslanides et al., 2017] and the code is available online in the AIXIjs repository (http://aslanides.io/aixijs/demo.html?reward\_corruption).

We demonstrate that RL agents like Q-learning and softmax Q-learning cannot overcome corrupt reward (as discussed in Section 3), while quantilisation helps overcome corrupt reward (as discussed in Section 5). We run Q-learning with -greedy $( \epsilon = 0 . 1 )$ , softmax with temperature $\beta = 2$ , and the quantilising agent with $\delta = 0 . 2 , 0 . 5 , 0 . 8$ (where $0 . 8 = 1 - \sqrt { q / | { \cal S } | } = 1 - \sqrt { 1 / 2 5 } )$ for 100 runs with 1 million cycles. Average observed and true rewards after 1 million cycles are shown in Table 1, and reward trajectories are shown in Figure 8. Q-learning gets stuck on the corrupt tile and spend almost all the time there (getting observed reward around

(a) Observed rewards for 1 goal tile

(b) True rewards for 1 goal tile

(c) Observed rewards for 2 goal tiles

(d) True rewards for 2 goal tiles

(e) Observed rewards for 4 goal tiles

(f) True rewards for 4 goal tiles

Figure 8: Trajectories of average observed and true rewards for Q-learning, softmax and quantilising agents, showing mean ± standard deviation over 100 runs. Q-learning and quantilising agents converge to a similar observed reward, but very diferent true rewards (much higher for the quantiliser with high variance). The value of δ that gives the highest true reward varies for diferent numbers of goal tiles.

<table><tr><td>goal tiles</td><td>agent</td><td>average observed reward</td><td>average true reward</td></tr><tr><td rowspan="5">1</td><td>Q-learning</td><td>0.923 ± 0.0003</td><td>0.00852 ± 0.00004</td></tr><tr><td>Softmax Q-learning</td><td>0.671 ± 0.0005</td><td>0.0347 ± 0.00006</td></tr><tr><td>Quantilising (δ = 0.2)</td><td>0.838 ± 0.15</td><td>0.378 ± 0.35</td></tr><tr><td>Quantilising (δ = 0.5)</td><td>0.943 ± 0.12</td><td>0.133 ± 0.27</td></tr><tr><td>Quantilising (δ = 0.8)</td><td>0.979 ± 0.076</td><td>0.049 ± 0.18</td></tr><tr><td rowspan="5">2</td><td>Q-learning</td><td>0.921 ± 0.00062</td><td>0.0309 ± 0.0051</td></tr><tr><td>Softmax Q-learning</td><td>0.671 ± 0.0004</td><td>0.0738 ± 0.0005</td></tr><tr><td>Quantilising (δ = 0.2)</td><td>0.934 ± 0.047</td><td>0.594 ± 0.43</td></tr><tr><td>Quantilising (δ = 0.5)</td><td>0.931 ± 0.046</td><td>0.621 ± 0.42</td></tr><tr><td>Quantilising (δ = 0.8)</td><td>0.944 ± 0.05</td><td>0.504 ± 0.45</td></tr><tr><td rowspan="5">4</td><td>Q-learning</td><td>0.924 ± 0.0002</td><td>0.00919 ± 0.00014</td></tr><tr><td>Softmax Q-learning</td><td>0.657 ± 0.0004</td><td>0.111 ± 0.0006</td></tr><tr><td>Quantilising (δ = 0.2)</td><td>0.918 ± 0.038</td><td>0.738 ± 0.35</td></tr><tr><td>Quantilising (δ = 0.5)</td><td>0.926 ± 0.044</td><td>0.666 ± 0.39</td></tr><tr><td>Quantilising (δ = 0.8)</td><td>0.915 ± 0.036</td><td>0.765 ± 0.32</td></tr></table>

Table 1: Average true and observed rewards after 1 million cycles, showing mean ± standard deviation over 100 runs. Q-learning achieves high observed reward but low true reward, and softmax achieves medium observed reward and a slightly higher true reward than Q-learning. The quantilising agent achieves similar observed reward to Q-learning, but much higher true reward (with much more variance). Having more than 1 goal tile leads to a large improvement in true reward for the quantiliser, a small improvement for softmax, and no improvement for Q-learning.

$1 \cdot ( 1 - \epsilon ) = 0 . 9 )$ , softmax spends most of its time on the corrupt tile, while the quantilising agent often stays on one of the goal tiles.

## 7 Conclusions

This paper has studied the consequences of corrupt reward functions. Reward functions may be corrupt due to bugs or misspecifications, sensory errors, or because the agent finds a way to inappropriately modify the reward mechanism. Some examples were given in the introduction. As agents become more competent at optimising their reward functions, they will likely also become more competent at (ab)using reward corruption to gain higher reward. Reward corruption may impede the performance of a wide range of agents, and may have disastrous consequences for highly intelligent agents [Bostrom, 2014].

To formalise the corrupt reward problem, we extended a Markov Decision Process (MDP) with a possibly corrupt reward function, and defined a formal performance measure (regret). This enabled the derivation of a number of formally precise results for how seriously diferent agents were afected by reward corruption in diferent setups (Table 2). The results are all intuitively plausible, which provides some support for the choice of formal model.

The main takeaways from the results are:

• Without simplifying assumptions, no agent can avoid the corrupt reward problem (Theorem 11). This is efectively a No Free Lunch result, showing that unless some assumption is made about the reward corruption, no agent can outperform a random agent. Some natural simplifying assumptions to avoid the No Free Lunch result were suggested in Section 2.

• Using the reward signal as evidence rather than optimisation target is no magic bullet, even under strong simplifying assumptions (Theorem 16). Essentially, this is because the agent does not know the exact relation between the observed reward (the “evidence”) and the true reward.<sup>7</sup> However, when the data enables suficient crosschecking of rewards, agents can avoid the corrupt reward problem (Theorems 19 and 20). For example, in SSRL and LVFS this type of crosschecking is possible under natural assumptions. In RL, no crosschecking is possible, while CIRL is a borderline case. Combining frameworks and providing the agent with diferent sources of data may often be the safest option.

<table><tr><td rowspan="2">Assumption</td><td rowspan="2">No assumptions</td><td colspan="4">Assumption 12 or 12′, and ...</td></tr><tr><td>no other assumptions</td><td>Assumption 14</td><td>CIRL</td><td>SSRL/LVFS</td></tr><tr><td>Result</td><td>all agents fail</td><td> $\pi^{\delta}$  weak bound</td><td> $\pi_{b,t}^{\text{RL}}, \pi_{b,t}^{\text{CR}}$  fail $\pi^{\delta}$  succeeds</td><td> $\pi_{b,t}^{\text{CR}}$  fails</td><td> $\pi_{b,t}^{\text{CR}}$  succeeds</td></tr></table>

Table 2: Main takeaways. Without additional assumptions, all agents fail (i.e., sufer high regret). Restricting the reward corruption with Assumption 12 gives a weak bound for the quantilising agent. The $\pi _ { b , t } ^ { \mathrm { R L } }$ and $\pi _ { b , t } ^ { \mathrm { C R } }$ agents still fail even if we additionally assume many high reward states and agent control (Assumption 14), but the quantilising agent $\pi ^ { \delta }$ does well. In most realistic contexts, the true reward is learnable in spite of sensory corruption in SSRL and LVFS, but not in CIRL.

• In cases where suficient crosschecking of rewards is not possible, quantilisation may improve robustness (Theorems 23 and 28). Essentially, quantilisation prevents agents from overoptimising their objectives. How well quantilisation works depends on how the number of corrupt solutions compares to the number of good solutions.

The results indicate that while reward corruption constitutes a major problem for traditional RL algorithms, there are promising ways around it, both within the RL framework, and in alternative frameworks such as CIRL, SSRL and LVFS.

Future work. Finally, some interesting open questions are listed below:

• (Unobserved state) In both the RL and the decoupled RL models, the agent gets an accurate signal about which state it is in. What if the state is hidden? What if the signal informing the agent about its current state can be corrupt?

• (Non-stationary corruption function) In this work, we tacitly assumed that both the reward and the corruption functions are stationary, and are always the same in the same state. What if the corruption function is non-stationary, and influenceable by the agent’s actions? (such as if the agent builds a delusion box around itself [Ring and Orseau, 2011])

• (Infinite state space) Many of the results and arguments relied on there being a finite number of states. This makes learning easy, as the agent can visit every state. It also makes quantilisation easy, as there is a finite set of states/strategies to randomly sample from. What if there is an infinite number of states, and the agent has to generalise insights between states? What are the conditions on the observation graph for Theorems 19 and 20? What is a good generalisation of the quantilising agent?

• (Concrete CIRL condition) In Example 21, we only heuristically inferred the observation graph from the CIRL problem description. Is there a general way of doing this? Or is there a direct formulation of the no-corruption condition in CIRL, analogous to Theorems 19 and 20?

• (Practical quantilising agent) As formulated in Definition 22, the quantilising agent $\pi ^ { \delta }$ is extremely ineficient with respect to data, memory, and computation. Meanwhile, many practical RL algorithms use randomness in various ways (e.g. ε-greedy [Sutton and Barto, 1998]). Is there a way to make an eficient quantilisation agent that retains the robustness guarantees?

• (Dynamically adapting quantilising agent) In Definition 27, the threshold δ is given as a parameter. Under what circumstances can we define a “parameter free” quantilising agent that adapts δ as it interacts with the environment?

• (Decoupled RL quantilisation result) What if we use quantilisation in decoupled RL settings that nearly meet the conditions of Theorems 19 and 20? Can we prove a stronger bound?

## Acknowledgements

Thanks to Jan Leike, Badri Vellambi, and Arie Slobbe for proofreading and providing invaluable comments, and to Jessica Taylor and Huon Porteous for good comments on quantilisation. This work was in parts supported by ARC grant DP150104590.

## References

Dario Amodei and Jack Clark. Faulty Reward Functions in the Wild. https://openai.com/blog/ faulty-reward-functions/, 2016. Accessed: 2017-02-18.

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Man´e. Concrete Problems in AI Safety. CoRR, 1606.06565, 2016.

John Aslanides, Jan Leike, and Marcus Hutter. Universal reinforcement learning algorithms: Survey and experiments. In IJCAI-17. AAAI Press, 2017.

Donald A Berry and Bert Fristedt. Bandit Problems: Sequential Allocation of Experiments. Springer, 1985.

Nick Bostrom. Superintelligence: Paths, Dangers, Strategies. Oxford University Press, 2014.

Owain Evans, Andreas Stuhlmuller, and Noah D Goodman. Learning the Preferences of Ignorant, Inconsistent Agents. In AAAI-16, 2016.

Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel, and Stuart Russell. Cooperative Inverse Reinforcement Learning. Advances in Neural Information Processing Systems (NIPS), 2016.

Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel, and Stuart Russell. The Of-Switch Game. In AAAI Workshop on AI, Ethics and Society, 2017.

Marcus Hutter. Universal Artificial Intelligence: Sequential Decisions based on Algorithmic Probability. Lecture Notes in Artificial Intelligence (LNAI 2167). Springer, 2005.

Tommi Jaakkola, Michael I Jordan, and Satinder P Singh. On the Convergence of Stochastic Iterative Dynamic Programming Algorithms. Neural Computation, 6(6):1185–1201, 1994.

Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal Regret Bounds for Reinforcement Learning. Journal of Machine Learning Research, 11(1):1563–1600, 2010.

Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and Acting in Partially Observable Stochastic Domains. Artificial Intelligence, 101(1-2):99–134, 1998.

Ming Li. Average Case Complexity under the Universal Distribution Equals Worst Case Complexity. Information Processing Letters, 42(3):145–149, 1992.

Andrew Ng and Stuart Russell. Algorithms for Inverse Reinforcement Learning. In Proceedings of the Seventeenth International Conference on Machine Learning, pages 663–670, 2000.

M.L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley Series in Probability and Statistics. Wiley, 1994.

Mark O Riedl and Brent Harrison. Using Stories to Teach Human Values to Artificial Agents. In AAAI Workshop on AI, Ethics, and Society, 2016.

Mark Ring and Laurent Orseau. Delusion, Survival, and Intelligent Agents. In Artificial General Intelligence, pages 11–20. Springer Berlin Heidelberg, 2011.

Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. MIT Press, 1998.

Jessica Taylor. Quantilizers: A Safer Alternative to Maximizers for Limited Optimization. In AAAI Workshop on AI, Ethics and Society, 2016.

David H Wolpert and William G Macready. No Free Lunch Theorems for Optimization. IEEE Transactions on Evolutionary Computation, 1(1):270–283, 1997.

Roman V. Yampolskiy. Utility Function Security in Artificially Intelligent Agents. Journal of Experimental & Theoretical Artificial Intelligence, pages 373–389, 2014.
