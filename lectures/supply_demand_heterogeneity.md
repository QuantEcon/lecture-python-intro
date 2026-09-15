---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(supply_demand_heterogeneity)=
# Market Equilibrium with Heterogeneity

## Overview

In the {doc}`previous lecture
<supply_demand_multiple_goods>`, we studied competitive equilibria in an economy with many goods.

That economy contained a single representative consumer.

Households, firms and other economic agents differ from one another along many dimensions.

This lecture introduces heterogeneity across consumers by letting their preferences and endowments differ.

We set production aside and study a pure exchange economy throughout.

We compute competitive equilibria, construct a representative consumer, and verify both welfare theorems.

Here are some imports:

```{code-cell} ipython3
import numpy as np
from scipy.linalg import inv
```

## A simple example

Let's study a simple example of a **pure exchange** economy without production.

Two consumers differ in their endowment vectors $e_i$ and their bliss point vectors $b_i$ for $i=1,2$.

Both consumers share the same matrix $\Pi$.

The shared $\Pi$ lets us aggregate the two consumers into a single representative consumer below.

The total endowment is $e_1 + e_2$.

A competitive equilibrium requires that

$$
c_1 + c_2 = e_1 + e_2
$$

Recall from {doc}`supply_demand_multiple_goods` that each consumer's demand curve is

$$
    c_i = (\Pi^\top \Pi )^{-1}(\Pi^\top b_i -  \mu_i p )
$$

Competitive equilibrium then requires that

$$
e_1 + e_2 =
    (\Pi^\top \Pi)^{-1}(\Pi^\top (b_1 + b_2) - (\mu_1 + \mu_2) p )
$$

which, after a line or two of linear algebra, implies that

$$
(\mu_1 + \mu_2) p = \Pi^\top(b_1+ b_2) - \Pi^\top \Pi (e_1 + e_2)
$$ (eq:old6)

We take $\Pi$ to be square and invertible, as in {doc}`supply_demand_multiple_goods`, so that $(\Pi^\top \Pi)^{-1}\Pi^\top = \Pi^{-1}$.

We normalize prices by setting $\mu_1 + \mu_2 =1$ and then solve

$$
\mu_i(p,e) = \frac{p^\top (\Pi^{-1} b_i - e_i)}{p^\top (\Pi^\top \Pi )^{-1} p}
$$ (eq:old7)

for $\mu_i, i = 1,2$.

```{exercise-start}
:label: sdh_ex1
```

Show that, up to normalization by a positive scalar, the same competitive equilibrium price vector that you derived in the preceding two-consumer economy would prevail in a single-consumer economy in which a single **representative consumer** has utility function

$$
- \frac{1}{2} (\Pi c -b) ^\top (\Pi c -b )
$$

and endowment vector $e$,  where

$$
b = b_1 + b_2
$$

and

$$
e = e_1 + e_2 .
$$

```{exercise-end}
```

```{solution-start} sdh_ex1
:class: dropdown
```

For the two-consumer economy, equation {eq}`eq:old6` says that the competitive equilibrium price vector satisfies

$$
(\mu_1 + \mu_2)\, p = \Pi^\top (b_1 + b_2) - \Pi^\top \Pi (e_1 + e_2)
  = \Pi^\top b - \Pi^\top \Pi e
$$

Now consider the single representative consumer with bliss point $b = b_1 + b_2$ and endowment $e = e_1 + e_2$.

Because that consumer is alone in the economy, market clearing requires $c = e$, so their demand curve implies the price vector $\tilde p$ that satisfies

$$
\tilde \mu \, \tilde p = \Pi^\top b - \Pi^\top \Pi e
$$

The right sides of the two displayed equations are identical, so

$$
\tilde p = \frac{\mu_1 + \mu_2}{\tilde \mu}\, p
$$

The two price vectors are therefore proportional, with a positive factor of proportionality because marginal utilities of wealth are positive.

Since only relative prices matter, the two economies have the same competitive equilibrium prices; normalizing $\mu_1 + \mu_2 = \tilde\mu = 1$ makes the two price vectors equal.

We verify this more carefully, and allow for transfers of wealth, in the section {ref}`deducing a representative consumer <rep_consumer>` below.

```{solution-end}
```

## Pure exchange economy

Let's further explore a pure exchange economy with $n$ goods and $m$ people.

In this lecture $m$ denotes the number of consumers.

In {doc}`supply_demand_multiple_goods` $m$ denoted the number of rows of $\Pi$.

### Competitive equilibrium

We'll compute a competitive equilibrium.

To compute a competitive equilibrium of a pure exchange economy, we use the fact that

- Relative prices in a competitive equilibrium are the same as those in a special single person or  representative consumer economy with preference $\Pi$ and $b=\sum_i b_i$, and endowment $e = \sum_i e_{i}$.

This aggregation requires that all consumers share the same matrix $\Pi$.

Consumer $i$ faces the budget constraint

$$
p^{\top}\left(c_{i}-e_{i}\right)=W_{i}
$$ (eq:budget_i)

where $W_{i}$ is a lump-sum transfer of wealth measured in units of the numeraire good, and the transfers satisfy $\sum_i W_{i}=0$.

Setting $W_{i}=0$ for every $i$ returns a pure exchange economy in which a consumer's endowment is their only source of income.

Substituting the demand curve into {eq}`eq:budget_i` and solving for $\mu_i$ gives

$$
\mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$ (eq:mu_i)

These are **Marshallian** demand curves in the terminology of {doc}`supply_demand_multiple_goods`: we solve for $\mu_i$ from a budget constraint instead of fixing it.

We compute a competitive equilibrium in three steps.

- First we solve the single representative consumer economy by normalizing $\mu = 1$, then renormalize the price vector by using the first consumption good as a numeraire.

- Next we compute each consumer's marginal utility of wealth from {eq}`eq:mu_i`.

- Finally we compute a competitive equilibrium allocation from the demand curves:

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p
$$


### Designing some Python code


Below we shall construct a Python class with the following attributes:

 * **Preferences** in the form of

     * an $n \times n$ invertible matrix $\Pi$, shared by all consumers, which makes $\Pi^\top \Pi$ positive definite
     * an $n \times 1$ vector of bliss points $b_i$ for each consumer $i$

 * **Endowments** in the form of

     * an $n \times 1$ vector $e_i$ for each consumer $i$
     * a scalar "wealth" $W_i$ for each consumer $i$, with default value $0$


The class will include a test to make sure that $b_i \gg \Pi e_i $ and raise an exception if it is violated
(at some threshold level we'd have to specify).

 * **A Pure Exchange Economy** will consist of

    * a collection of $m$ **persons**, each described by a bliss point $b_i$, an endowment $e_i$ and a wealth $W_i$

       * $m=2$ in all of the economies that we study below

    * an equilibrium price vector $p$ (normalized somehow)
    * an equilibrium allocation $c_1, c_2, \ldots, c_m$ -- a collection of $m$ vectors of dimension $n \times 1$

Now let's proceed to code.

```{code-cell} ipython3
class ExchangeEconomy:
    def __init__(self,
                 Π,
                 bs,
                 es,
                 Ws=None,
                 thres=1.5):
        """
        Set up the environment for an exchange economy

        Args:
            Π (np.array): shared matrix of substitution
            bs (list): all consumers' bliss points
            es (list): all consumers' endowments
            Ws (list): all consumers' wealth
            thres (float): a threshold used to test whether b >> Π e is violated
        """
        n, m = Π.shape[0], len(bs)

        # check non-satiation
        for b, e in zip(bs, es):
            if np.min(b / np.max(Π @ e)) <= thres:
                raise Exception('set bliss points further away')

        if Ws is None:
            Ws = np.zeros(m)
        elif not np.isclose(np.sum(Ws), 0):
            raise Exception('invalid wealth distribution')

        self.Π, self.bs, self.es, self.Ws, self.n, self.m = Π, bs, es, Ws, n, m

    def competitive_equilibrium(self):
        """
        Compute the competitive equilibrium prices and allocation
        """
        Π, bs, es, Ws = self.Π, self.bs, self.es, self.Ws
        n, m = self.n, self.m
        slope_dc = inv(Π.T @ Π)
        Π_inv = inv(Π)

        # aggregate
        b = sum(bs)
        e = sum(es)

        # compute price vector with mu=1 and renormalize
        p = Π.T @ b - Π.T @ Π @ e
        p = p / p[0]

        # compute marginal utility of wealth
        μ_s = []
        c_s = []
        A = p.T @ slope_dc @ p

        for i in range(m):
            μ_i = (-Ws[i] + p.T @ (Π_inv @ bs[i] - es[i])) / A
            c_i = Π_inv @ bs[i] - μ_i * slope_dc @ p
            μ_s.append(μ_i)
            c_s.append(c_i)

        for c_i in c_s:
            if any(c_i < 0):
                print('allocation: ', c_s)
                raise Exception('negative allocation: equilibrium does not exist')

        return p, c_s, μ_s
```

## Implementation

Next we use the class ``ExchangeEconomy`` defined above to study

* a two-person economy without production,
* a dynamic economy, and
* an economy with risk and Arrow securities.

### Two-person economy without production

Here we study how competitive equilibrium $p, c_1, c_2$ respond to different $b_i$ and $e_i$, $i \in \{1, 2\}$.

We also report each consumer's marginal utility of wealth $\mu_i$.

A high $\mu_i$ identifies a consumer who is poor at equilibrium prices, in the sense that an extra unit of wealth would be worth a lot to them.

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1]])

bs = [np.array([5, 5]),  # first consumer's bliss points
      np.array([5, 5])]  # second consumer's bliss points

es = [np.array([0, 2]),  # first consumer's endowment
      np.array([2, 0])]  # second consumer's endowment

EE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Marginal utilities of wealth:', μ_s)
```

The two consumers have identical preferences and differ only in which good they are endowed with.

They trade to a symmetric allocation, and their marginal utilities of wealth are equal.

What happens if the first consumer likes the first good more and the second consumer likes the second good more?

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # first consumer's bliss points
         np.array([5, 6])]  # second consumer's bliss points

p, c_s, μ_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Marginal utilities of wealth:', μ_s)
```

Each consumer now ends up with more of the good that they like better.

Let the first consumer be poorer.

Each cell below sets both the bliss points and the endowments, so that no experiment inherits values from the preceding one.

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # first consumer's bliss points
         np.array([5, 6])]  # second consumer's bliss points

EE.es = [np.array([0.5, 0.5]),  # first consumer's endowment
         np.array([1, 1])]  # second consumer's endowment

p, c_s, μ_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Marginal utilities of wealth:', μ_s)
```

The poorer consumer has the higher marginal utility of wealth.

The first consumer consumes none of the second good.

Our quadratic utility function does not rule out such corners.

A poorer first consumer would drive the allocation negative, and our class would raise an exception.

Now let's construct an autarky (i.e., no-trade) equilibrium.

```{code-cell} ipython3
EE.bs = [np.array([4, 6]),  # first consumer's bliss points
      np.array([6, 4])]  # second consumer's bliss points

EE.es = [np.array([0, 2]),  # first consumer's endowment
      np.array([2, 0])]  # second consumer's endowment

p, c_s, μ_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Marginal utilities of wealth:', μ_s)
```

The equilibrium allocation reproduces the endowments, and no trade occurs.

Each consumer likes the good with which they are endowed enough to keep it at the equilibrium price vector.

The aggregate endowment and the aggregate bliss point have been symmetric across the two goods in every experiment so far, and the equilibrium price vector has been $(1, 1)$.

With $\Pi = I$, equation {eq}`eq:old6` makes $p$ proportional to $b - e$, so symmetry of both aggregates produces equal prices.

Let's now make the second good scarcer in the aggregate.

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # first consumer's bliss points
         np.array([5, 6])]  # second consumer's bliss points

EE.es = [np.array([1.25, 0.75]),  # first consumer's endowment
         np.array([1.25, 0.75])]  # second consumer's endowment

p, c_s, μ_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Marginal utilities of wealth:', μ_s)
```

The relative price of the second good now exceeds $1$, because the second good is the scarcer of the two.

The two consumers have identical endowments, so their differing tastes drive all trade.

Now let's redistribute endowments before trade.

```{code-cell} ipython3
bs = [np.array([5, 5]),  # first consumer's bliss points
      np.array([5, 5])]  # second consumer's bliss points

es = [np.array([1, 1]),  # first consumer's endowment
      np.array([1, 1])]  # second consumer's endowment

Ws = [0.5, -0.5]
EE_new = ExchangeEconomy(Π, bs, es, Ws)
p, c_s, μ_s = EE_new.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Marginal utilities of wealth:', μ_s)
```

A lump-sum transfer of wealth from the second consumer to the first leaves the equilibrium price vector at $(1,1)$.

It moves the allocation from $(1,1)$ for each consumer to $(1.25, 1.25)$ and $(0.75, 0.75)$.

Redistributing wealth before trade has supported a different Pareto optimal allocation as a competitive equilibrium, an instance of the **second welfare theorem** of {doc}`supply_demand_multiple_goods`.

```{exercise}
:label: sdh_ex2

Take $\Pi = I$ and bliss points $b_1 = (6,5)^\top$ and $b_2 = (5,6)^\top$.

Consider two ways of dividing the *same* aggregate endowment $e_1 + e_2 = (3, 1.5)^\top$:

* distribution A: $e_1 = (2, 0.5)^\top$ and $e_2 = (1, 1)^\top$
* distribution B: $e_1 = (0.5, 1.5)^\top$ and $e_2 = (2.5, 0)^\top$

a. Compute the equilibrium price vector, allocation and marginal utilities of wealth under each distribution.

What changes and what does not?

b. Verify that $\sum_i \mu_i$ is also the same under the two distributions, and explain why by using equation {eq}`eq:old6`.

c. Which consumer has the higher marginal utility of wealth under distribution A, and why?

```

```{solution-start} sdh_ex2
:class: dropdown
```

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1]])

bs = [np.array([6, 5]),
      np.array([5, 6])]

distributions = {'A': [np.array([2, 0.5]), np.array([1, 1])],
                 'B': [np.array([0.5, 1.5]), np.array([2.5, 0])]}

for name, es in distributions.items():
    EE_d = ExchangeEconomy(Π, bs, es)
    p, c_s, μ_s = EE_d.competitive_equilibrium()
    print(f'distribution {name}: aggregate endowment = {sum(es)}')
    print(f'  price vector : {p}')
    print(f'  allocation   : {[np.round(c, 4) for c in c_s]}')
    print(f'  μ_1, μ_2     : {np.round(μ_s, 4)}, sum = {np.sum(μ_s):.4f}\n')
```

The price vector is identical under the two distributions.

The allocations differ.

Equation {eq}`eq:old6` makes the price vector depend on the individual $b_i$ and $e_i$ only through the sums $b = \sum_i b_i$ and $e = \sum_i e_i$, and redistributing a fixed aggregate endowment leaves both sums unchanged.

All consumers share the same $\Pi$, so every demand curve has the same slope $-(\Pi^\top \Pi)^{-1} p$ with respect to the marginal utility of wealth.

Moving wealth from one consumer to another shifts one demand curve down and the other up by offsetting amounts.

For part b, $\sum_i \mu_i = 8.0$ under both distributions.

Equation {eq}`eq:old6` pins that sum down given $p$, $b$ and $e$.

It equals the marginal utility of wealth $\tilde\mu$ of the representative consumer constructed below.

For part c, the second consumer has the higher marginal utility of wealth under distribution A.

Their endowment sells for less at equilibrium prices, so an extra unit of wealth is worth more to them.

```{solution-end}
```

### A dynamic economy

Now let's use the tricks described in {doc}`supply_demand_multiple_goods` to study a dynamic economy, one with two periods.

We give our two consumers identical preferences but endow them at different dates: the first consumer is endowed early and the second consumer late.

```{code-cell} ipython3
beta = 0.95

Π = np.array([[1, 0],
              [0, np.sqrt(beta)]])

bs = [np.array([5, np.sqrt(beta) * 5]),
      np.array([5, np.sqrt(beta) * 5])]

es = [np.array([2, 0]),   # first consumer is endowed at date 1
      np.array([0, 2])]   # second consumer is endowed at date 2

EE_DE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_DE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Gross interest rate R = p1/p2:', p[0] / p[1])
```

The gross interest rate is $R = \beta^{-1} = 1.0526$.

The first consumer is endowed early and lends to the second consumer, who is endowed late.

Each consumer's consumption is constant across the two dates.

Neither consumer's endowment is constant.

The exercise below derives that flatness from the condition $\beta R = 1$ of {doc}`cons_smooth`.

{doc}`olg` studies an economy with the same two-period structure, in which the young and the old trade at a gross interest rate.

{doc}`olg` adds production and an unending sequence of overlapping generations.

Our economy has a single pair of consumers and no production.

```{exercise}
:label: sdh_ex3

Consider the two-period economy just computed, in which both consumers have bliss point $\Pi (5,5)^\top$.

a. Show that consumer $i$'s consumption is constant across the two dates if and only if $\beta R = 1$, where $R = p_1 / p_2$.

b. Now let the aggregate endowment grow, by setting $e_1 = (1.5, 0)^\top$ and $e_2 = (0, 2.5)^\top$.

Recompute the equilibrium.  What happens to $R$ and to the shapes of the two consumption paths?

```

```{solution-start} sdh_ex3
:class: dropdown
```

For part a, the demand curve is $c_i = \Pi^{-1} b_i - \mu_i (\Pi^\top \Pi)^{-1} p$.

Here $\Pi^{-1} b_i = (5,5)^\top$ and $(\Pi^\top \Pi)^{-1} = \operatorname{diag}(1, \beta^{-1})$, so

$$
c_{i,2} - c_{i,1} = - \mu_i \left( \frac{p_2}{\beta} - p_1 \right)
                  = - \mu_i p_1 \left( \frac{1}{\beta R} - 1 \right)
$$

This difference is zero for every consumer if and only if $\beta R = 1$, because $\mu_i > 0$ and $p_1 > 0$.

{doc}`cons_smooth` obtains a flat consumption path from the same condition.

```{code-cell} ipython3
bs = [np.array([5, np.sqrt(beta) * 5]),
      np.array([5, np.sqrt(beta) * 5])]

es = [np.array([1.5, 0]),
      np.array([0, 2.5])]

EE_growth = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_growth.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', [np.round(c, 4) for c in c_s])
print(f'R       = {p[0] / p[1]:.6f}')
print(f'1 / beta = {1 / beta:.6f}')
```

The aggregate endowment grows from $1.5$ at date $1$ to $2.5$ at date $2$.

The gross interest rate rises to $R = 1.1930$ and exceeds $\beta^{-1} = 1.0526$.

Both consumption paths tilt upward over time, because $\beta R > 1$.

The two consumers share the consumption growth; they do not share the endowment growth.

```{solution-end}
```

### Risk economy with Arrow securities

We use the tricks described in {doc}`supply_demand_multiple_goods` to interpret  $c_1, c_2$ as "Arrow securities" that are state-contingent claims to consumption goods.

The first consumer is endowed only in state $1$ and the second consumer only in state $2$.

The economy-wide endowment is therefore one unit in each state, so the economy carries **individual** risk and no **aggregate** risk.

```{code-cell} ipython3
prob = 0.7

Π = np.array([[np.sqrt(prob), 0],
              [0, np.sqrt(1 - prob)]])

bs = [np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5]),
      np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])]

es = [np.array([1, 0]),
      np.array([0, 1])]

EE_AS = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_AS.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
print('Ratio of state prices p1/p2:', p[0] / p[1])
print('Ratio of probabilities λ/(1-λ):', prob / (1 - prob))
```

The relative price of the two Arrow securities equals the odds ratio $\lambda/(1-\lambda) = 0.7/0.3$, so state-contingent claims trade at **actuarially fair odds**.

Each consumer's consumption is the same in both states: $c_1 = (0.7, 0.7)$ and $c_2 = (0.3, 0.3)$.

The market in Arrow securities delivers **complete insurance**.

No consumer's consumption depends on which state occurs.

Consumption differs across the two consumers because they are not equally rich.

The first consumer owns the endowment in the state that occurs with probability $0.7$.

That endowment sells for more at equilibrium prices, and the market converts its value into a safe consumption stream.

```{exercise}
:label: sdh_ex4

Return to the Arrow securities economy just computed.

a. Introduce **aggregate** risk by setting $e_1 = (1.5, 0)^\top$ while leaving $e_2 = (0,1)^\top$, so that the economy-wide endowment is $1.5$ in state $1$ and $1$ in state $2$.

Recompute the equilibrium.  Is consumption still state-independent?  Does $p_1/p_2$ still equal the odds ratio?

b. Explain which kind of risk a complete set of Arrow securities can eliminate and which kind it cannot.

```

```{solution-start} sdh_ex4
:class: dropdown
```

```{code-cell} ipython3
es = [np.array([1.5, 0]),
      np.array([0, 1])]

EE_agg = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_agg.competitive_equilibrium()

print('aggregate endowment:', sum(es))
print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', [np.round(c, 4) for c in c_s])
print(f'p1/p2 = {p[0] / p[1]:.4f},  odds λ/(1-λ) = {prob / (1 - prob):.4f}')
```

Insurance is no longer complete.

Consumption is state-dependent for both consumers.

Both consume more in state $1$, the state in which the economy as a whole has more.

The price ratio falls to $2.2037$, below fair odds, because state $1$ consumption is now abundant and cheap.

For part b, a complete set of Arrow securities eliminates individual risk, which concerns who receives the goods.

It cannot eliminate aggregate risk, which concerns how many goods the economy has.

Consumers share aggregate risk, and it shows up in state-contingent consumption and in state prices that depart from the odds.

```{solution-end}
```

(rep_consumer)=
## Deducing a representative consumer

In the class of multiple consumer economies that we are studying here, it turns out that there
exists a single **representative consumer** whose preferences and endowments can be deduced from lists of preferences and endowments for separate individual consumers.

Consider a multiple consumer economy with an initial distribution of wealth $W_i$ satisfying $\sum_i W_{i}=0$.

We allow an initial redistribution of wealth.

We have the following objects:


- The demand curve:

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p
$$

- The marginal utility of wealth:

$$
\mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$

- Market clearing:

$$
\sum c_{i}=\sum e_{i}
$$

Denote aggregate consumption $\sum_i c_{i}=c$ and $\sum_i \mu_i = \mu$.

Market clearing requires

$$
\Pi^{-1}\left(\sum_{i}b_{i}\right)-(\Pi^{\top}\Pi)^{-1}p\left(\sum_{i}\mu_{i}\right)=\sum_{i}e_{i}
$$
which, after a few steps, leads to

$$
p=\mu^{-1}\left(\Pi^{\top}b-\Pi^{\top}\Pi e\right)
$$

where

$$
\mu = \sum_i\mu_{i}
    = \frac{-\sum_i W_{i} + p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
    = \frac{p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$

Here the wealth transfers have dropped out of the sum because $\sum_i W_{i}=0$.

Now consider the representative consumer economy specified above.

Denote the marginal utility of wealth of the representative consumer by $\tilde{\mu}$.

The demand function is

$$
c=\Pi^{-1}b-(\Pi^{\top}\Pi)^{-1}\tilde{\mu} p
$$

Substituting this into the representative consumer's budget constraint $p^{\top}(c-e)=0$, which contains no wealth term because $\sum_i W_{i}=0$, gives

$$
\tilde{\mu}=\frac{p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$

In an equilibrium $c=e$, so

$$
p=\tilde{\mu}^{-1}(\Pi^{\top}b-\Pi^{\top}\Pi e)
$$

Thus, we have  verified that, up to the choice of a numeraire in which to express absolute prices, the price
vector in our representative consumer economy is the same as that in an underlying  economy with multiple consumers.

```{note}
Aggregation worked because all of our consumers share the same matrix $\Pi$, which gives every demand curve a common slope in wealth.

Preferences with this property take the **Gorman form**.

Without it, the equilibrium price vector would depend on the distribution of wealth, and no representative consumer would exist.

The representative consumer computes prices.

It tells us nothing about how goods are distributed among individual consumers, and its utility is not a social welfare function.
```

## Negishi weights and the first welfare theorem

{doc}`supply_demand_multiple_goods` verified a version of the **first welfare theorem**: a competitive equilibrium quantity vector solves a planning problem.

That lecture contained a single consumer, so its planner had nobody to choose among.

A planner with many consumers must attach a weight to each of them.

Let $\theta_i > 0$ be the weight that a planner attaches to consumer $i$ and consider the planning problem

$$
\max_{\{c_i\}} \ \sum_i \theta_i \left(-\frac{1}{2}\right)(\Pi c_{i}-b_{i})^{\top}(\Pi c_{i}-b_{i})
\quad \text{subject to} \quad \sum_i c_{i}=\sum_i e_{i}
$$

Form the Lagrangian

$$
L = \sum_i \theta_i \left(-\frac{1}{2}\right)(\Pi c_{i}-b_{i})^{\top}(\Pi c_{i}-b_{i})
    + \eta^{\top}\left(\sum_i e_{i}-\sum_i c_{i}\right)
$$

where $\eta$ is a vector of Lagrange multipliers on the resource constraint.

First-order conditions with respect to $c_i$ are

$$
-\theta_{i}\Pi^{\top}(\Pi c_{i}-b_{i})-\eta=0
$$

so that

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\frac{\eta}{\theta_{i}}
$$ (eq:negishi_demand)

Now compare {eq}`eq:negishi_demand` with the competitive equilibrium demand curve

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p
$$

The two coincide when

$$
\theta_{i}=\frac{1}{\mu_{i}} \quad \text{and} \quad \eta = p
$$

A competitive equilibrium allocation solves a planning problem whose welfare weights are the reciprocals of the consumers' marginal utilities of wealth.

[Takashi Negishi](https://en.wikipedia.org/wiki/Takashi_Negishi) introduced weights of this kind, and they carry his name.

A consumer who is poor at equilibrium prices has a high $\mu_i$, and a planner attaches a low weight $\theta_i$ to them to justify the market outcome.

The multiplier $\eta$ on the resource constraint equals the equilibrium price vector.

Shadow prices of scarce resources are market prices, as {doc}`lp_intro` discusses.

```{exercise}
:label: sdh_ex5

Verify these claims numerically.

a. Write a function that solves the planner's problem for arbitrary weights $\theta_i$ by combining {eq}`eq:negishi_demand` with the resource constraint.

b. Check that setting $\theta_i = 1/\mu_i$ reproduces the competitive equilibrium allocation, and that $\eta$ is proportional to $p$.

Do this both for the two-person economy in which the second good is scarce and for the Arrow securities economy.

c. Now set equal weights $\theta_1 = \theta_2$ and describe how the planner's allocation differs from the competitive one.

```

```{solution-start} sdh_ex5
:class: dropdown
```

```{code-cell} ipython3
def planner_allocation(Π, bs, es, θs):
    """
    Solve the planner's problem for welfare weights θs
    """
    slope_dc = inv(Π.T @ Π)
    Π_inv = inv(Π)
    e = sum(es)

    # the resource constraint pins down the multiplier η
    A = slope_dc * sum(1 / θ for θ in θs)
    rhs = sum(Π_inv @ b for b in bs) - e
    η = np.linalg.solve(A, rhs)

    c_s = [Π_inv @ bs[i] - slope_dc @ η / θs[i] for i in range(len(bs))]
    return η, c_s
```

```{code-cell} ipython3
# the two-person economy in which the second good is scarce
Π_1 = np.array([[1, 0],
                [0, 1]])
bs_1 = [np.array([6, 5]), np.array([5, 6])]
es_1 = [np.array([1.25, 0.75]), np.array([1.25, 0.75])]

# the Arrow securities economy
prob = 0.7
Π_2 = np.array([[np.sqrt(prob), 0],
                [0, np.sqrt(1 - prob)]])
bs_2 = [np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5]),
        np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])]
es_2 = [np.array([1, 0]), np.array([0, 1])]

economies = [('scarce good 2', Π_1, bs_1, es_1),
             ('Arrow securities', Π_2, bs_2, es_2)]

for name, Π_i, bs_i, es_i in economies:
    p, c_s, μ_s = ExchangeEconomy(Π_i, bs_i, es_i).competitive_equilibrium()
    η, c_pl = planner_allocation(Π_i, bs_i, es_i, θs=[1 / μ for μ in μ_s])

    print(name)
    print(f'  competitive allocation : {[np.round(c, 4) for c in c_s]}')
    print(f'  planner allocation     : {[np.round(c, 4) for c in c_pl]}')
    print(f'  p                      : {np.round(p, 4)}')
    print(f'  η / η[0]               : {np.round(η / η[0], 4)}')
    print(f'  allocations agree      : {np.allclose(c_s, c_pl)}\n')
```

Setting $\theta_i = 1/\mu_i$ reproduces the competitive allocation, and the normalized multiplier $\eta$ equals the competitive price vector.

Now give the two consumers equal weights.

```{code-cell} ipython3
for name, Π_i, bs_i, es_i in economies:
    η, c_pl = planner_allocation(Π_i, bs_i, es_i, θs=[1, 1])
    print(f'{name}: equal-weight allocation = {[np.round(c, 4) for c in c_pl]}')
```

The equal-weight planner disregards who owns what.

In the Arrow securities economy it gives each consumer $(0.5, 0.5)$, an equal share of the aggregate endowment in each state.

The market delivers $(0.7, 0.7)$ and $(0.3, 0.3)$ and respects each consumer's ownership of their state-contingent endowment.

A lump-sum redistribution of wealth would decentralize the equal-weight allocation as a competitive equilibrium, by the second welfare theorem.

The first welfare theorem makes the market outcome a planner's outcome for some weights.

The second welfare theorem turns a planner's outcome for any weights into a market outcome, given the right transfers.

```{solution-end}
```

## Concluding remarks

This lecture studied a competitive equilibrium of a pure exchange economy whose consumers differ in their preferences and their endowments.

The equilibrium price vector depends on individual consumers only through the **aggregate** bliss point and the **aggregate** endowment.

A single representative consumer reproduces the equilibrium prices of an economy with many consumers.

A competitive equilibrium allocation solves a planning problem whose welfare weights are the reciprocals of the consumers' marginal utilities of wealth, an instance of the first welfare theorem.

Redistributing wealth before trade changes the allocation and leaves prices unchanged, an instance of the second welfare theorem.

The same apparatus describes borrowing and lending when goods are dated, and insurance when goods are state-contingent.

A competitive equilibrium delivers complete consumption smoothing when $\beta R = 1$ and complete insurance when no aggregate risk exists.

Neither result survives variation in the aggregate endowment across dates or across states.

Consumers share aggregate risk and aggregate growth; they cannot eliminate them.
