---
layout: page
title: Fairness
tagline: notation, definitions, data, legality
description: 
---

The source for this site is [here](https://github.com/speak-statistics-to-power/fairness). We welcome [pull requests](https://yangsu.github.io/pull-request-tutorial/) or emails (<sam942@mail.harvard.edu>) with additions and corrections!

## Notation
$Y =$ outcome, for simplicity only binary (e.g. arrest, [a mismeasurement of offense](#data))  
$A =$ race, for simplicity only $ = b$ or $w$  
$X =$ observed covariates (excludes $A$)  
$S = \hat{P}(Y = 1 | A, X) =$ estimated probability of the outcome  
&nbsp;&nbsp;&nbsp;&nbsp;$= f(A, X, \mathbf{Y}^{\textrm{train}}, \mathbf{A}^{\textrm{train}}, \mathbf{X}^{\textrm{train}})$ where $f$ can be stochastic and takes training data  
$d =$ decision/classifier, e.g. $=I(S > s_{\textrm{high-risk}})$ where $s_{\textrm{high-risk}} =$ cutoff/threshold. More generally,  
&nbsp;&nbsp;&nbsp;&nbsp;$= f_d(A, X, \mathbf{Y}^{\textrm{train}}, \mathbf{A}^{\textrm{train}}, \mathbf{X}^{\textrm{train}})$ where $f_d$ can be stochastic and takes training data  
&nbsp;&nbsp;&nbsp;&nbsp;(Some papers aren't clear whether they mean $S$ or $d$)  
$d(b) =$ decision had the person been black (a [potential outcome](http://stat.cmu.edu/~fienberg/Rubin/Rubin-JASA-05.pdf))  
$d(w) =$ decision had the person been white  
The observed $d = d(b)$ if the person is black and $= d(w)$ if the person is white. We can also define potential outcomes for $S$.  
Index people by $i,j$, e.g. $Y_i$ is the outcome for person $i$. Imagine drawing people ([independently and identically](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)) from a population distribution $p(Y,A,X,S,S(b),S(w),d,d(b),d(w) | \mathbf{Y}^{\textrm{train}}, \mathbf{A}^{\textrm{train}}, \mathbf{X}^{\textrm{train}})$. Many fairness definitions are properties of this distribution. From now on we assume everything is conditional on the training data.

## Definitions
Here are definitions of fairness found in the literature, along with some important results related to them. We link to papers using the definition or making the claim.  

#### Definitions based on $p(Y, A, S)$

1. *Calibration*:
$P(Y = 1 \| S = s, A = b) = P(Y = 1 \| S = s, A = w)$ [[Chouldechova]]  
([Hardt et al.] call this *matching conditional frequencies*)  
*Strong calibration*: also requires $= s$ [[Kleinberg et al.]]  
<button class="link" onclick="show('calibration_context')">(context)</button>
    <div id="calibration_context" style="display:none" markdown="1">
    > Strongly calibrated risk scores mean what they claim to mean. In other words, for people who get a risk score of 80%, there is an 80% probability they actually get arrested. Moreover, we require this to hold within groups: 80% of black people and 80% of white people with scores of 80% would end up getting arrested.  
    The weaker version of calibration allows the fraction arrested to differ from the risk score, as long as it is the same for both groups, e.g.: 70% of black people and 70% of white people with scores of 80% end up getting arrested.  
    > [COMPAS]\: satisfies [[Flores et al.]]  
    </div>
2. *Balance for the negative class*: 
$E(S \| Y = 0, A = b) = E(S \| Y = 0 , A = w)$ [[Kleinberg et al.]]  
<button class="link" onclick="show('balance_neg_context')">(context)</button>
    <div id="balance_neg_context" style="display:none" markdown="1">
    > If a score is balanced for the negative class, then among people who do not get arrested, risk scores for black and white people have the same average. In other words, for people who did not get arrested, we require that they have been judged to be equally risky, on average.  
    > [COMPAS]\: I assume not?
    </div>
3. *Balance for the positive class*: 
$E(S \| Y = 1, A = b) = E(S \| Y = 1 , A = w)$ [[Kleinberg et al.]]  
<button class="link" onclick="show('balance_pos_context')">(context)</button>
    <div id="balance_pos_context" style="display:none" markdown="1">
    > If a score is balanced for the positive class, then among people who did get arrested, risk scores for black and white people have the same average. In other words, for people who did get arrested, we require that they have been judged to be equally risky, on average.  
    > [COMPAS]\: I assume not?
    </div>

<div class="result" markdown="1">
MAY NEED RELAXING: Cannot have 1 + 2 + 3 unless $S=Y$ (perfect prediction) or $P(Y=1\|A=b) = P(Y=1\|A=w)$ (equal base rates) [[Kleinberg et al.]]
</div>

#### Definitions based on $p(Y, A, d)$

The next four definitions come from the four margins of the *Confusion Matrix*:

|-------------+----------------------------------+----------------------------------+------------------------------+-----------------------------|
|             |                                  |                                  |        Prevalence            |                             |
|             |            $Y=1$                 |             $Y=0$                |       "base rate"            |                             |
|             |                                  |                                  |         $P(Y=1)$             |                             |
|-------------|:--------------------------------:|:--------------------------------:|:----------------------------:|:---------------------------:|
|             |                                  |                                  |  Positive Predictive Value   |     False Discovery Rate    |
|    $d=1$    |        True Positive             |        False Positive            |        ([PPV](#PPV))         |       ([FDR](#FDR))         |
|             |            (TP)                  |             (FP)                 |     $P(Y = 1 \| d = 1)$      |     $P(Y = 0 \| d = 1)$     |
|             |                                  |                                  |                              |                             |
|-------------+----------------------------------+----------------------------------+------------------------------+-----------------------------|
|             |                                  |                                  |     False Omission Rate      |  Negative Predictive Value  |
|    $d=0$    |        False Negative            |        True Negative             |        ([FOR](#FOR))         |        ([NPV](#NPV))        |
|             |             (FN)                 |            (TN)                  |   $P(Y = 1 \| d = 0)$  |  $P(Y = 0 \| d = 0)$  |
|             |                                  |                                  |                              |                             |
|-------------+----------------------------------+----------------------------------+------------------------------+-----------------------------|
|             |      True Positive Rate          |     False Positive Rate          |                              |                             |
|             |([TPR](#TPR))<!--- sensitivity -->|        ([FPR](#FPR))             |  [Accuracy](#accuracy)       |                             |
|             |       $P(d = 1 \| Y = 1)$        |      $P(d = 1 \| Y = 0)$         |         $P(d = Y)$           |                             |
|             |                                  |                                  |                              |                             |
|-------------+----------------------------------+----------------------------------+------------------------------+-----------------------------|
|             |      False Negative Rate         |      True Negative Rate          |                              |                             |
|             |         ([FNR](#FNR))            |([TNR](#TNR))<!--- specificity -->|                              |                             |
|             |       $P(d = 0 \| Y = 1)$        |      $P(d = 0 \| Y = 0)$         |                              |                             |
|-------------+----------------------------------+----------------------------------+------------------------------+-----------------------------|

<br>
those given special names have citations:

{:start="4"}
4.  *equal NPVs<a name="NPV"></a>*: $P(Y=0 \| d = 0, A = b) = P(Y=0 \| d = 0, A = w)$  
    $\Leftrightarrow$ *equal FORs<a name="FOR"></a>*: $P(Y=1 \| d = 0, A = b) = P(Y=1 \| d = 0, A = w)$  
    $\Leftrightarrow Y \perp A | d = 0$  
<button class="link" onclick="show('NPV_context')">(context)</button>
    <div id="NPV_context" style="display:none" markdown="1">
    > [COMPAS]\: ?
    </div>
5.  *Predictive parity (equal PPVs<a name="PPV"></a>)*: $P(Y=1 \| d = 1, A = b) = P(Y=1 \| d = 1, A = w)$ [[Chouldechova]]  
    $\Leftrightarrow$ *equal FDRs<a name="FDR"></a>*: $P(Y=0 \| d = 1, A = b) = P(Y=0 \| d = 1, A = w)$  
    $\Leftrightarrow Y \perp A | d = 1$  
<button class="link" onclick="show('PPV_context')">(context)</button>
    <div id="PPV_context" style="display:none" markdown="1">
    > Intuition for predictive parity... calibrated risk scores can produce decisions that do not satisfy predictive parity, depending on the threshold used.  
    > [COMPAS]\: satisfies [[Northpointe]] 
    </div>
6. *Error rate balance (equal FPRs<a name="FPR"></a>)*: $P(d = 1 \| Y = 0, A = b) = P(d = 1 \| Y = 0, A = w)$ [[Chouldechova]\]  
<!--- [Corbett-Davies et al.] calls this *Predictive equality* but may not need to include this term -->
    $\Leftrightarrow$ *equal TNRs<a name="TNR"></a>*: $P(d = 0 \| Y = 0, A = b) = P(d = 0 \| Y = 0, A = w)$  
    $\Leftrightarrow d \perp A | Y = 0$  
<button class="link" onclick="show('FPR_context')">(context)</button>
    <div id="FPR_context" style="display:none" markdown="1">
    > Intuition for error rate balance (FPRs)  
    > [COMPAS]\: $P(d = 1 \| Y = 0, A = b) \approx 2*P(d = 1 \| Y = 0, A = w)$ [[ProPublica]]  
    </div>
7. *Error rate balance (equal FNRs<a name="FNR"></a>)*: $P(d = 0 \| Y = 1, A = b) = P(d = 0 \| Y = 1, A = w)$ [[Chouldechova]]  
    $\Leftrightarrow$ *Equal opportunity (equal TPRs<a name="TPR"></a>)*:
    $P(d = 1 \| Y = 1, A = b) = P(d = 1 \| Y = 1, A = w)$ [[Hardt et al.], [Kusner et al.]]  
    $\Leftrightarrow d \perp A | Y = 1$  
<button class="link" onclick="show('FNR_context')">(context)</button>
    <div id="FNR_context" style="display:none" markdown="1">
    > Intuition for error rate balance (FNRs)  
    > [COMPAS]\: $2*P(d = 0 \| Y = 1, A = b) \approx P(d = 0 \| Y = 1, A = w)$ [[ProPublica]] 
    </div>
    
<div class="result" markdown="1">
Cannot have 5 + 6 + 7 unless $d=Y$ (perfect prediction) or $P(Y=1\|A=b) = P(Y=1\|A=w)$ (equal base rates) because $\textrm{FPR} = \frac{p}{1-p} \frac{1-\textrm{PPV}}{\textrm{PPV}} (1 - \textrm{FNR})$ [[Chouldechova]]
</div>  

   4+5. *Conditional use accuracy equality*: $Y \perp A \| d$ [[Berk et al.]]  
   6+7. *Equalized odds* [[Hardt et al.]], *Conditional procedure accuracy equality* [[Berk et al.]]: $d \perp A \| Y$  
    
{:start="8"}
8. *Statistical/Demographic parity*<a name="parity"></a>: $P(d = 1 \| A = b) = P(d = 1 \| A = w)$ [[Chouldechova], [Kusner et al.]]  
or $E(S \| A = b) = E(S \| A = w)$ [[Kleinberg et al.]\]  
*Conditional statistical parity*: $P(d = 1 \| L = l, A = b) = P(d = 1 \| L = l, A = w)$, for *legitimate factors* $L$ [[Corbett-Davies et al.]]  
<button class="link" onclick="show('parity_context')">(context)</button>
    <div id="parity_context" style="display:none" markdown="1">
    > [Johndrow and Lum] advocate for this definition because...
    The causal analogy to legitimate factors are [*resolving variables*](#resolving))  
    > [COMPAS]\: I assume not?
    </div>
9.  *Overall accuracy<a name="accuracy"></a> equality*: $P(d = Y \| A = b) = P(d = Y \| A = w)$ [[Berk et al.]]  
<button class="link" onclick="show('accuracy_context')">(context)</button>
    <div id="accuracy_context" style="display:none" markdown="1">
    > Intuition for accuracy  
    > [COMPAS]\: ?
    </div>
10. *Treatment equality (equal FN/FP)*: $\frac{P(d=0, Y=1 \| A = b)}{P(d=1, Y=0 \| A = b)} = \frac{P(d=0, Y=1 \| A = w)}{P(d=1, Y=0 \| A = w)}$ [[Berk et al.]]  
<button class="link" onclick="show('FN/FP_context')">(context)</button>
    <div id="FN/FP_context" style="display:none" markdown="1">
    > Intuition for treatment equality  
    > [COMPAS]\: ?
    </div>

#### Definitions that do not marginalize over $X$ ("individual" definitions)

[aaronsadventures] describes how the above definitions can hide a lot of sins by marginalizing (i.e. averaging) over $X$. For example, suppose an algorithm gives $d = 1$ if and only if either $A = b$ and $X = 1$ or $A = w$ and $X = 0$. By the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability):

$$P(d = 1 | A = b) = \underbrace{P(d = 1 | A = b, X = 1)}_{1} P(X = 1 | A = b) + \underbrace{P(d = 1 | A = b, X = 0)}_{0} P(X = 0 | A = b)$$  

$$P(d = 1 | A = w) = \underbrace{P(d = 1 | A = w, X = 1)}_{0} P(X = 1 | A = w) + \underbrace{P(d = 1 | A = w, X = 0)}_{1} P(X = 0 | A = w)$$  

So if $P(X = 1 \| A = b) = P(X = 0 \| A = w)$ then we satisfy [parity](#parity). But within strata defined by $X$, we do not:

$$1 = P(d = 1 | A = b, X = 1) \ne P(d = 1 | A = w, X = 1) = 0$$

As [aaronsadventures] puts it:
> If you know some fact about yourself that makes you think you are not a uniformly random white person, the guarantee can lack bite.

The next definitions consider these facts, i.e. the $X$ variables.

{:start="11"}
11. *Fairness through unawareness*: $d_i = d_j$ if $X_i = X_j$ (no use of $A$) [[Kusner et al.]]  
A property of $p(d,X)$? Maybe: $\exists f$ such that $p(d|X) = 1$ if $d = f(X)$ and $= 0$ otherwise.  
<button class="link" onclick="show('unawareness_context')">(context)</button>
    <div id="unawareness_context" style="display:none" markdown="1">
    > Intuition for unawareness  
    > [COMPAS]\: ?
    </div>
12. *Individual Fairness*: $\|\|d_i - d_j\|\| \leq \epsilon'$ if $\|\|X_i^{\textrm{true}} - X_j^{\textrm{true}}\|\| \leq \epsilon$ where we think $X^{\textrm{true}}$ are fair to use for decisions [[Friedler et al.]]  
([Dwork et al.] call this the *Lipschitz property*.)  
This is possible if *What You See Is What You Get*: $\exists$ an observation process $g: X^{\textrm{true}} \rightarrow X$ that doesn't distort too much, i.e. $\bigg \| \|\|X_i^{\textrm{true}} - X_j^{\textrm{true}}\|\| - \|\|g(X_i^{\textrm{true}}) - g(X_j^{\textrm{true}}) \|\|\bigg \| < \delta$  
AND  
if you use an *Individual Fairness Mechanism* $\textrm{IFM}: X \rightarrow d$ that doesn't distort too much, i.e. $\bigg \| \|\|X_i - X_j\|\| - \|\|\textrm{IFM}(X_i) - \textrm{IFM}(X_j) \|\|\bigg \| < \delta'$  
A property of $p(d,X,X^{\textrm{true}})$?  
<button class="link" onclick="show('indiv_context')">(context)</button>
    <div id="indiv_context" style="display:none" markdown="1">
    > Intuition for individual fairness  
    > [COMPAS]\: ?
    </div>

#### Causal definitions
Papers that use causal definitions often do not make explicit the difference between $S$ and $d$. Below we write all definitions with $d$ for convenience.

{:start="13"}
13. *Fair inference*: no unfair path-specific effects (PSEs) of $A$ on $d$ [[Nabi and Shpitser]]  
<!--- construction: in the real world, a causal model induces some distribution on observed variables $p(d,X,A)$. Fair inference constructs $d^{fair}$ using $p^*(d,X,A)$, the [KL](https://twitter.com/KLdivergence)-closest distribution to $p(d,X,A)$ that blocks certain path-specific effects (PSEs) of $A$ on $d$ -->
<button class="link" onclick="show('fair_inference_context')">(context)</button>
    <div id="fair_inference_context" style="display:none" markdown="1">
    > Intuition for fair inference: See Section 4 of [Nabi and Shpitser], summarize?  
    > [COMPAS]\: ?
    </div>
14. *Counterfactual fairness*<a name="counterfactual_fairness"></a>: $p(d(b) \| A = a, X = x) = p(d(w) \| A = a, X = x)$ [[Kusner et al.]]  
To satisfy this, [Kusner et al.] propose $d$ be a function of non-descendents of $A$, but if arrows in the causal diagram represent individual-level effects, this implies a stronger definition of fairness: $d(b) = d(w)$  
<button class="link" onclick="show('counterfactual_fairness_context')">(context)</button>
    <div id="counterfactual_fairness_context" style="display:none" markdown="1">
    > Intuition for counterfactual fairness  
    > [COMPAS]\: ?
    </div>
15. *No unresolved discrimination*: there is no directed path from $A$ to $d$ unless through a resolving variable [[Kilbertus et al.]]  
*Resolving variable*<a name="resolving"></a>: a variable whose influence from $A$ we accept as non-discriminatory (e.g. we decide that it is acceptable for race to affect admission rates through department choice)  
<button class="link" onclick="show('no_unresolved_context')">(context)</button>
    <div id="no_unresolved_context" style="display:none" markdown="1">
    > Intuition for no unresolved discrimination  
    > [COMPAS]\: ?
    </div>
16. *No proxy discrimination*: $p(d(a)) = p(d(a'))$ for all $a,a'$ for a proxy variable $A^\textrm{proxy}$ [[Kilbertus et al.]]  
*Proxy variable*: descendent of $A$ we choose to label as a proxy (e.g. we decide to label skin color as a proxy, and disallow it from affecting admissions)  
<button class="link" onclick="show('no_proxy_context')">(context)</button>
    <div id="no_proxy_context" style="display:none" markdown="1">
    > Intuition for no proxy discrimination  
    > [COMPAS]\: ?
    </div>

    <div class="textbox" markdown="1">
    **Causal diagrams.**  
    The causal fairness literature differs on whether $Y$, $d$, or both are included in the causal diagrams. [Nabi and Shpitser] include $d$ in the causal diagrams. [Kusner et al.] include $Y$ in the causal diagrams (even though [they define fairness](#counterfactual_fairness) in terms of effects on $d$). [Kilbertus et al.] include both $Y$ and $d$ in the causal diagrams (in their Figure 2 only).
    </div>

    <div class="textbox" markdown="1">
    **Causal interpretation of race.**  
    [VanderWeele and Robinson]...
    </div>

## <a name="data"></a> Data

It is important to note that $\mathbf{Y}^{\textrm{train}}$ is what is measured (e.g. arrest), which is often different from the $\mathbf{Y}^{\textrm{true, train}}$ (e.g. offense). [Lum] highlights this issue, largely overlooked in the literature, which often uses the words "arrest" and "offense" interchangeably. [Corbett-Davies et al.] make the distinction in their discussion section:
> But arrests are an imperfect proxy. Heavier policing in minority neighborhoods might lead to black defendants being arrested more often than whites who commit the same crime [31 - [Lum and Isaac]]. Poor outcome data might thus cause one to systematically underestimate the risk posed by white defendants. This concern is mitigated when the outcome $y$ is serious crime — rather than minor offenses — since such incidents are less susceptible to biased observation. In particular, [Skeem and Lowencamp] [38] note that the racial distribution of individuals arrested for violent offenses is in line with the racial distribution of offenders inferred from victim reports and also in line with self-reported offending data.

[Skeem and Lowencamp]:
> In our view, official records of arrest — particularly for violent offenses — are a valid criterion. First, surveys of victimization yield “essentially the same racial differentials as do official statistics. For example, about 60 percent of robbery victims describe their assailants as black, and about 60 percent of victimization data also consistently show that they fit the official arrest data” (Walsh, 2009, p. 22). Second, self-reported offending data reveal similar race differentials, particularly for serious and violent crimes (see [Piquero], 2015).

But [Piquero] says:
> In official (primarily arrest) records, research has historically revealed that minorities (primarily Blacks because of the lack of other race/ethnicity data) are overrepresented in crime, especially serious crimes such as violence. More recently, this conclusion has started to be questioned, as researchers have started to better study and document the potential Hispanic effect [49]. Analyses of self-reported offending data reveal a much more similar set of estimates regarding offending across race, again primarily in Black-White comparisons...

Are there analytical ways around this measurement error issue? In v1 of their paper, [Nabi and Shpitser] suggested the following:
> For instance, recidivism in the United States is defined to be a *subsequent arrest*, rather than a *subsequent conviction*. It is well known that certain minority groups in the US are arrested disproportionately, which can certainly bias recidivism risk prediction models to unfairly target those minority groups. We can address this issue in our setting by using a missingness model, a simple version of which is shown in Fig. 1 (d), which is a version of Fig. 1 (a) with two additional variables, $Y(1)$ and $R$. $Y(1)$ represents the underlying true outcome, which is not always observed, $R$ is a missingness indicator, and $Y$ is equal to $Y(1)$ if $R = 1$ and equal to "?" otherwise...Fig. 1 (d) corresponds to a missing at random (MAR) assumption...

They distinguish between *arrest* and *conviction* (which is not identical to *offense*, the true variable of interest, but we ignore this for now). They use $Y$ to denote observed arrests and $Y(1)$ to denote unobserved conviction. The measurement problem is that $Y \ne Y(1)$. [Nabi and Shpitser] describe an observed indicator $R$ such that when $R = 1$, we have $Y = Y(1)$. Thus, conviction data $Y(1)$ are known for a subset ($R = 1$) of people. Missing at random (MAR)<sup>[1](#MAR)</sup> assumes $Y(1) \| X, R = 1 \overset{\textrm{MAR}}{=} Y(1) \| X$, allowing use of known conviction data for valid inference. More generally, we can let $R$ be an indicator for observing convictions, without assuming arrest equals conviction, i.e. $Y = Y(1)$, when $R = 1$. Are any risk tools trained on conviction data?

Differential measurement in the covariates $X$ could make them differently predictive across groups, as [Corbett-Davies et al.] write:
> One might similarly worry that the features x are biased in the sense that factors are not equally predictive across race groups, a phenomenon known as *subgroup validity* [[4](http://journals.sagepub.com/doi/pdf/10.3818/JRP.4.1.2002.131)]

For example, prior arrests for black defendents may be less predictive of future offense than prior arrests for white defendents.

## Legality - very rough (need help from lawyers)

U.S. courts have applied [strict scrutiny](https://en.wikipedia.org/wiki/Strict_scrutiny) if a law or policy either
1. infringes on a [fundamental right](https://en.wikipedia.org/wiki/Fundamental_rights#United_States), or
2. uses [race, national origin, religion, or alienage](https://en.wikipedia.org/wiki/Suspect_classification).

Strict scrutiny: the law or policy must be
1. justified by a compelling government interest,
2. narrowly tailored to achieve that goal, and
3. the least restrictive means to achieve that goal.

[Equal Protection Clause](https://en.wikipedia.org/wiki/Equal_Protection_Clause) of the [Fourteenth Amendment](https://en.wikipedia.org/wiki/Fourteenth_Amendment_to_the_United_States_Constitution):
> All persons born or naturalized in the United States, and subject to the jurisdiction thereof, are citizens of the United States and of the State wherein they reside. No State shall make or enforce any law which shall abridge the privileges or immunities of citizens of the United States; nor shall any State deprive any person of life, liberty, or property, without due process of law; nor deny to any person within its jurisdiction the equal protection of the laws.

Strict scrutiny has been applied in some cases (but not all, e.g. not in [gender cases](https://en.wikipedia.org/wiki/United_States_v._Virginia)) that involve the Equal Protection Clause. Strict scrutiny is also applied in some cases that involve [Free Speech](https://en.wikipedia.org/wiki/Reed_v._Town_of_Gilbert). 

The first case in which the Supreme Court determined strict scrutiny to be satisfied was [Korematsu v. United States (1944)](https://en.wikipedia.org/wiki/Korematsu_v._United_States), regarding Japanese internment. Since then, laws and policies using race have been subject to strict scrutiny, rarely satisfying it (although some [affirmative action plans have been upheld](https://en.wikipedia.org/wiki/Grutter_v._Bollinger), for example).

[Corbett-Davies et al.] warn that including race as an input feature in estimating risk scores or using race-specific thresholds for decisions (both of which can be used in efforts to satisfy certain fairness definitions) would likely trigger strict scrutiny.

<br>
<br>
<br>

---

<a name="MAR">1</a>: For more about MAR, see e.g. p.202 of [BDA3] or p.398 of [LDA].

[COMPAS]: http://www.equivant.com/solutions/inmate-classification
[ProPublica]: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
[Flores et al.]: http://www.ajc.state.ak.us/sites/default/files/commission-recommendations/rejoinder7.11.pdf  
[Northpointe]: http://go.volarisgroup.com/rs/430-MBX-989/images/ProPublica_Commentary_Final_070616.pdf 

[Chouldechova]: http://www.andrew.cmu.edu/user/achoulde/files/disparate_impact.pdf  
[Kleinberg et al.]: https://arxiv.org/abs/1609.05807   
[Hardt et al.]: https://arxiv.org/abs/1610.02413
[Corbett-Davies et al. Monkey Cage]: https://www.washingtonpost.com/news/monkey-cage/wp/2016/10/17/can-an-algorithm-be-racist-our-analysis-is-more-cautious-than-propublicas/?utm_term=.128a239d2221
<!--- reached out: -->
[Corbett-Davies et al.]: https://5harad.com/papers/fairness.pdf
[Berk et al.]: https://arxiv.org/pdf/1703.09207.pdf
<!--- reached out: -->
[Johndrow and Lum]: https://arxiv.org/abs/1703.04957

[aaronsadventures]: http://aaronsadventures.blogspot.com/2017/11/between-statistical-and-individual.html
<!--- reached out: -->
[Friedler et al.]: https://arxiv.org/abs/1609.07236
<!--- reached out: -->
[Dwork et al.]: https://arxiv.org/pdf/1104.3913.pdf

<!--- reached out: -->
[Nabi and Shpitser]: https://arxiv.org/abs/1705.10378
<!--- reached out: -->
[Kusner et al.]: https://arxiv.org/abs/1703.06856
<!--- reached out: -->
[Kilbertus et al.]: https://arxiv.org/abs/1706.02744
[VanderWeele and Robinson]: http://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC4125322&blobtype=pdf

<!--- reached out: -->
[Lum]: https://www.nature.com/articles/s41562-017-0141.epdf?author_access_token=zp6NdnflZBcbq4u8QE13i9RgN0jAjWel9jnR3ZoTv0OrWCV-_5fNRpT89FpEf7bpDfmwoV6oPboJp9g43OZUEZ3Jvuivgqlqr1rjq9C3M62_Ady8s_dGfhXTCFeIUphULML1P_InKq12GOvN0UKaRw%3D%3D
<!--- reached out: -->
[Lum and Isaac]: http://onlinelibrary.wiley.com/doi/10.1111/j.1740-9713.2016.00960.x/epdf
<!--- reached out: -->
[Skeem and Lowencamp]: http://risk-resilience.berkeley.edu/sites/default/files/journal-articles/files/criminology_proofs_archive.pdf
[Piquero]: https://link.springer.com/content/pdf/10.1007%2Fs40865-015-0004-3.pdf

[BDA3]: https://books.google.co.il/books?id=ZXL6AQAAQBAJ&vq=%22missing+at+random%22&source=gbs_navlinks_s
[LDA]: https://books.google.co.il/books?id=zVBjCvQCoGQC&source=gbs_navlinks_s