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

(supply_demand_multiple_goods)=
# Supply and Demand with Many Goods

## Overview

In a {doc}`previous lecture <intro_supply_demand>` we studied supply, demand
and welfare in a market with a single consumption good.

In this lecture, we study a setting with $n$ goods and $n$ corresponding prices.

Key infrastructure concepts that we'll encounter in this lecture are

* inverse demand curves
* marginal utilities of wealth
* inverse supply curves
* consumer surplus
* producer surplus
* social welfare as a sum of consumer and producer surpluses
* competitive equilibrium


We will provide a version of the [first fundamental welfare theorem](https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics), which was formulated by 

* [Leon Walras](https://en.wikipedia.org/wiki/L%C3%A9on_Walras)
* [Francis Ysidro Edgeworth](https://en.wikipedia.org/wiki/Francis_Ysidro_Edgeworth)
* [Vilfredo Pareto](https://en.wikipedia.org/wiki/Vilfredo_Pareto)

Important extensions to the key ideas were obtained by

* [Abba Lerner](https://en.wikipedia.org/wiki/Abba_P._Lerner)
* [Harold Hotelling](https://en.wikipedia.org/wiki/Harold_Hotelling)
* [Paul Samuelson](https://en.wikipedia.org/wiki/Paul_Samuelson)
* [Kenneth Arrow](https://en.wikipedia.org/wiki/Kenneth_Arrow) 
* [Gérard Debreu](https://en.wikipedia.org/wiki/G%C3%A9rard_Debreu)


We shall describe two classic welfare theorems:

* **first welfare theorem:** for a given distribution of wealth among consumers, a competitive  equilibrium  allocation of goods solves a  social planning problem.

* **second welfare theorem:** An allocation of goods to consumers that solves a social planning problem can be supported by a competitive equilibrium with an appropriate initial distribution of  wealth.

This lecture studies a single representative consumer, so the distribution of wealth plays no role in it.

We verify a version of the **first** welfare theorem only.

{doc}`supply_demand_heterogeneity` takes up the distribution of wealth and both welfare theorems.

As usual, we start by importing some Python modules.

```{code-cell} ipython3
# import some packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
```

## Formulas from linear algebra

{doc}`linear_equations` describes tools for solving the linear systems that appear in this lecture and in {doc}`supply_demand_heterogeneity`.

We shall apply formulas from linear algebra that

* differentiate an inner product with respect to each vector
* differentiate a product of a matrix and a vector with respect to the vector
* differentiate a quadratic form in a vector with respect to the vector

Where $a$ is an $n \times 1$ vector, $A$ is an $n \times n$ matrix, and $x$ is an $n \times 1$ vector:

$$
\frac{\partial a^\top x }{\partial x} = \frac{\partial x^\top a }{\partial x} = a
$$

$$
\frac{\partial A x} {\partial x} = A
$$

$$
\frac{\partial x^\top A x}{\partial x} = (A + A^\top)x
$$

```{note}
The first and third formulas differentiate a scalar with respect to a vector and return $n \times 1$ vectors.

The second differentiates a vector with respect to a vector and returns an $n \times n$ matrix.

We use the first and third formulas below.
```

## From utility function to demand curve

Our study of consumers will use the following primitives

* $\Pi$, an $m \times n$ matrix,
* $b$, an $m \times 1$ vector of bliss points,
* $e$, an $n \times 1$ vector of endowments

+++

We will analyze endogenous objects $c$ and $p$, where

* $c$ is an $n \times 1$ vector of consumptions of various goods,
* $p$ is an $n \times 1$ vector of prices

+++

The matrix $\Pi$ describes a consumer's willingness to substitute one good for every other good.

We assume that $\Pi$ has linearly independent columns, which implies that $\Pi^\top \Pi$ is a positive definite matrix.

* it follows that $\Pi^\top \Pi$ has an inverse.

In all of our examples we shall set $m = n$ and take $\Pi$ to be invertible, so that $(\Pi^\top \Pi)^{-1} \Pi^\top = \Pi^{-1}$.

The matrix $-\mu (\Pi^\top \Pi)^{-1}$ gives the slopes of demand curves for $c$ with respect to $p$, holding fixed the marginal utility of wealth $\mu$ that we define below:

$$
    \frac{\partial c } {\partial p} = - \mu (\Pi^\top \Pi)^{-1}
$$

$(\Pi^\top \Pi)^{-1}$ is positive definite, so its diagonal elements are positive and the diagonal elements of $-\mu (\Pi^\top \Pi)^{-1}$ are negative.

Each good's own-price demand curve slopes downward.

The off-diagonal cross-price effects take either sign.

A demand curve that holds $\mu$ fixed is a **Frisch** demand curve, named after [Ragnar Frisch](https://en.wikipedia.org/wiki/Ragnar_Frisch).

A consumer faces $p$ as a price taker and chooses $c$ to maximize the utility function

$$
    - \frac{1}{2} (\Pi c -b) ^\top (\Pi c -b )
$$ (eq:old0)

subject to the budget constraint

$$
    p^\top (c -e ) = 0
$$ (eq:old2)

We shall specify examples in which $\Pi$ and $b$ are such that it typically happens that

$$
    \Pi c \ll b
$$ (eq:bversusc)

This means that the consumer has much less of each good than they want.

The deviation in {eq}`eq:bversusc` will ultimately assure us that competitive equilibrium prices are positive, provided that $\Pi$ has non-negative entries, as it does in all of our examples.

+++

### Demand curve implied by constrained utility maximization

For now, we assume that the budget constraint is {eq}`eq:old2`.

So we'll be deriving what is known as  a **Marshallian** demand curve.

Our aim is to maximize [](eq:old0) subject to [](eq:old2).

Form a Lagrangian

$$ L = - \frac{1}{2} (\Pi c -b)^\top (\Pi c -b ) + \mu [p^\top (e-c)] $$

where $\mu$ is a Lagrange multiplier that is often called a **marginal utility of wealth**.

The consumer chooses $c$ to maximize $L$ and $\mu$ to minimize it.

First-order conditions for $c$ are

$$
    \frac{\partial L} {\partial c}
    = - \Pi^\top \Pi c + \Pi^\top b - \mu p = 0
$$

so that, given $\mu$, the consumer chooses

$$
    c = (\Pi^\top \Pi )^{-1}(\Pi^\top b -  \mu p )
$$ (eq:old3)

Substituting {eq}`eq:old3` into budget constraint {eq}`eq:old2` and solving for $\mu$ gives

$$
    \mu(p,e) = \frac{p^\top ( \Pi^\top \Pi )^{-1} \Pi^\top b - p^\top e}{p^\top (\Pi^\top \Pi )^{-1} p}
$$ (eq:old4)

Equation {eq}`eq:old4` tells how marginal utility of wealth depends on the endowment vector $e$ and the price vector $p$.

```{note}
Equation {eq}`eq:old4` is a consequence of imposing $p^\top (c - e) = 0$.
```

## Marshallian, Hicksian, and Frisch demand curves

Sometimes we'll use budget constraint {eq}`eq:old2` in situations in which a consumer's endowment vector $e$ is their **only** source of income.

Other times we'll instead assume that the consumer has another source of income (positive or negative) and write their budget constraint as

$$
p ^\top (c -e ) = w
$$ (eq:old2p)

where $w$ is measured in "dollars" (or some other **numeraire**) and component $p_i$ of the price vector is measured in dollars per unit of good $i$.

Whether the consumer's budget constraint is {eq}`eq:old2` or {eq}`eq:old2p` and whether we take $w$ as a free parameter or instead as an endogenous variable will affect the consumer's marginal utility of wealth.

Consequently, how we set $\mu$ determines which of the following three demand curves we are constructing:

* a **Marshallian** demand curve, as when we use {eq}`eq:old2` and solve for $\mu$ using equation {eq}`eq:old4` above

* a **Frisch** demand curve, as when we treat $\mu$ as a fixed parameter and solve {eq}`eq:old2p` for $w$

* a **Hicksian** demand curve, as when we hold the consumer's *utility* fixed and let $w$ be whatever is required to attain it

These three demand curves contemplate different mental experiments.

For a Marshallian demand curve, hypothetical changes in a price vector have both **substitution** and **income** effects

* income effects are consequences of changes in $p^\top e$ associated with the change in the price vector

For a Frisch demand curve, the marginal utility of wealth $\mu$ is frozen while $w$ adjusts to finance the resulting consumption bundle

* differentiating {eq}`eq:old3` while holding $\mu$ fixed gives the slopes $\frac{\partial c}{\partial p} = - \mu (\Pi^\top \Pi)^{-1}$ that we met above

For a Hicksian demand curve, $w$ instead adjusts to keep **utility** constant, which is why a Hicksian demand curve is often called a **compensated** demand curve

* the compensation is designed to disarm the income (or wealth) effect associated with a price change

```{note}
Frisch and Hicksian demand curves are distinct objects.

A Hicksian demand curve minimizes expenditure $p^\top c$ subject to attaining a given utility level, so scaling all prices by a common positive factor leaves it unchanged.

Its matrix of slopes $S$ satisfies $S p = 0$.

The Frisch slopes $-\mu (\Pi^\top \Pi)^{-1}$ do not.

Freezing the marginal utility of wealth differs from freezing utility.
```

In the endowment economy below, the budget constraint {eq}`eq:old2` holds and $\mu$ normalizes the price level.

In the production economy below, no budget constraint restrains the consumer, so $\mu$ is a free parameter of a Frisch inverse demand curve.

The planning problem there uses that free parameter as the weight that a planner attaches to the consumer.

+++

## Endowment economy

We now study a pure-exchange economy, or what is sometimes called an endowment economy.

Consider a single-consumer, multiple-goods economy without production.

The only source of goods is the single consumer's endowment vector $e$.

A competitive equilibrium price vector induces the consumer to choose $c=e$.

This implies that the equilibrium price vector satisfies

$$
p = \mu^{-1} (\Pi^\top b - \Pi^\top \Pi e)
$$

In the present case, where we have imposed the budget constraint in the form {eq}`eq:old2`, we are free to normalize the price vector by setting the marginal utility of wealth $\mu =1$ (or any other value for that matter).

This amounts to choosing a common unit (or numeraire) in which prices of all goods are expressed.

(Doubling all prices will affect neither quantities nor relative prices.)

We'll set $\mu=1$.

```{exercise}
:label: sdm_ex1

Verify that setting $\mu=1$ in {eq}`eq:old3` implies that formula {eq}`eq:old4` is satisfied.

```

```{solution-start} sdm_ex1
:class: dropdown
```

Setting $\mu = 1$ in {eq}`eq:old3` and imposing the equilibrium condition $c = e$ gives the price vector

$$
p = \Pi^\top b - \Pi^\top \Pi e
$$

Multiplying this by $(\Pi^\top \Pi)^{-1}$ and rearranging yields the useful identity

$$
(\Pi^\top \Pi)^{-1} \Pi^\top b - e = (\Pi^\top \Pi)^{-1} p
$$

Now substitute this into the numerator of {eq}`eq:old4`:

$$
p^\top \left[ (\Pi^\top \Pi)^{-1} \Pi^\top b - e \right]
    = p^\top (\Pi^\top \Pi)^{-1} p
$$

The numerator of {eq}`eq:old4` therefore equals its denominator, so $\mu(p,e) = 1$, as required.

```{solution-end}
```

```{exercise}
:label: sdm_ex2

Verify that setting  $\mu=2$ in {eq}`eq:old3` also implies that formula
{eq}`eq:old4` is satisfied.

```

```{solution-start} sdm_ex2
:class: dropdown
```

Setting $\mu = 2$ in {eq}`eq:old3` and again imposing $c = e$ gives

$$
p = \frac{1}{2}\left(\Pi^\top b - \Pi^\top \Pi e\right)
$$

so that the identity used in the previous exercise becomes

$$
(\Pi^\top \Pi)^{-1} \Pi^\top b - e = 2 (\Pi^\top \Pi)^{-1} p
$$

The numerator of {eq}`eq:old4` is now $2 p^\top (\Pi^\top \Pi)^{-1} p$, which is twice the denominator, so $\mu(p,e) = 2$.

The same argument works for any $\mu > 0$.

Doubling $\mu$ halves the equilibrium price vector and leaves relative prices and the allocation $c = e$ unchanged.

$\mu$ normalizes the price level.

```{solution-end}
```

Here is a class that computes competitive equilibria for our economy.

```{note}
Our code forms matrix inverses explicitly with `inv`, so that each line of code mirrors a line of algebra.

For larger problems, solve the linear system directly with `numpy.linalg.solve`, as in {doc}`linear_equations`.
```

```{code-cell} ipython3
class ExchangeEconomy:
    
    def __init__(self, 
                 Π, 
                 b, 
                 e,
                 thres=1.5):
        """
        Set up the environment for an exchange economy

        Args:
            Π (np.array): shared matrix of substitution
            b (list):  the consumer's bliss point
            e (list):  the consumer's endowment
            thres (float): a threshold to check the b >> Π e condition
        """

        # check non-satiation
        if np.min(b / np.max(Π @ e)) <= thres:
            raise Exception('set bliss points further away')


        self.Π, self.b, self.e = Π, b, e

    
    def competitive_equilibrium(self):
        """
        Compute the competitive equilibrium prices and allocation
        """
        Π, b, e = self.Π, self.b, self.e

        # compute price vector with μ=1
        p = Π.T @ b - Π.T @ Π @ e
        
        # compute consumption vector
        slope_dc = inv(Π.T @ Π)
        Π_inv = inv(Π)
        c = Π_inv @ b - slope_dc @ p

        if any(c < 0):
            print('allocation: ', c)
            raise Exception('negative allocation: equilibrium does not exist')

        return p, c
```

## Dynamics and risk as special cases

Special cases of our $n$-good pure exchange model can be created to represent

* **dynamics** --- by putting different dates on different commodities
* **risk** --- by interpreting delivery of goods as being contingent on states of the world whose realizations are described by a *known probability distribution*

Let's illustrate how.

### Dynamics

Suppose that we want to represent a utility function

$$
  - \frac{1}{2} [(c_1 - b_1)^2 + \beta (c_2 - b_2)^2]
$$

where $\beta \in (0,1)$ is a discount factor, $c_1$ is consumption at time $1$ and $c_2$ is consumption at time 2.

To capture this with our quadratic utility function {eq}`eq:old0`, set

$$
\Pi = \begin{bmatrix} 1 & 0 \cr
         0 & \sqrt{\beta} \end{bmatrix}
$$

$$
e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}
$$

and

$$
b = \begin{bmatrix} b_1 \cr \sqrt{\beta} b_2
\end{bmatrix}
$$

The budget constraint {eq}`eq:old2` becomes

$$
p_1 c_1 + p_2 c_2 = p_1 e_1 + p_2 e_2
$$

The left side is the **discounted present value** of consumption.

The right side is the **discounted present value** of the consumer's endowment.

The relative price  $\frac{p_1}{p_2}$ has units of time $2$ goods per unit of time $1$ goods.

Consequently, 

$$
    (1+r) := R := \frac{p_1}{p_2}
$$ 

is the **gross interest rate** and $r$ is the **net interest rate**.

{doc}`pv` computes present values by discounting future payoffs with a discount factor $\delta \in (0,1)$.

Here $p_2 / p_1 = R^{-1}$ is that discount factor, determined inside the model as the relative price of goods at two dates.

{doc}`cons_smooth` studies a consumer who smooths consumption completely when $\beta R = 1$, a condition that Milton Friedman and Robert Hall assumed.

An exercise below derives $R = \beta^{-1}$ from equilibrium when the endowment is the same at both dates.

Here is an example.

```{code-cell} ipython3
beta = 0.95

Π = np.array([[1, 0],
              [0, np.sqrt(beta)]])

b = np.array([5, np.sqrt(beta) * 5])

e = np.array([1, 1])

dynamics = ExchangeEconomy(Π, b, e)
p, c = dynamics.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c)
```

```{exercise}
:label: sdm_ex5

Consider the dynamic interpretation above, with $\Pi = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{\beta}\end{bmatrix}$,
$b = \begin{bmatrix} \bar b \\ \sqrt{\beta}\,\bar b \end{bmatrix}$ and $e = \begin{bmatrix} e_1 \\ e_2\end{bmatrix}$.

a. Show analytically that the gross interest rate is

$$
R = \frac{p_1}{p_2} = \frac{1}{\beta} \cdot \frac{\bar b - e_1}{\bar b - e_2}
$$

and hence that $R = \beta^{-1}$ whenever the endowment is flat, that is, whenever $e_1 = e_2$.

b. Let $e_2 = (1+g) e_1$ with $e_1 = 1$, $\bar b = 5$ and $\beta = 0.95$.

Compute $R$ numerically for $g \in [-0.2, 0.2]$ and plot it against $g$, marking $\beta^{-1}$ with a dashed line.

c. Explain, in terms of the consumer's desire to smooth consumption, why $R$ rises with $g$.

```

```{solution-start} sdm_ex5
:class: dropdown
```

For part a, the equilibrium price vector is $p = \Pi^\top b - \Pi^\top \Pi e$.

With this $\Pi$ and $b$,

$$
\Pi^\top b = \begin{bmatrix} \bar b \cr \beta \bar b \end{bmatrix}, \qquad
\Pi^\top \Pi e = \begin{bmatrix} e_1 \cr \beta e_2 \end{bmatrix}, \qquad \text{so} \qquad
p = \begin{bmatrix} \bar b - e_1 \cr \beta (\bar b - e_2) \end{bmatrix}
$$

Taking the ratio of the two components gives the formula, and setting $e_1 = e_2$ gives $R = \beta^{-1}$.

```{code-cell} ipython3
beta, b_bar, e1 = 0.95, 5, 1

def R_of_g(g):
    Π = np.array([[1, 0],
                  [0, np.sqrt(beta)]])
    b = np.array([b_bar, np.sqrt(beta) * b_bar])
    e = np.array([e1, (1 + g) * e1])
    p, c = ExchangeEconomy(Π, b, e).competitive_equilibrium()
    return p[0] / p[1]

gs = np.linspace(-0.2, 0.2, 41)
Rs = np.array([R_of_g(g) for g in gs])
R_formula = np.array([(b_bar - e1) / (beta * (b_bar - (1 + g) * e1)) for g in gs])

print(f'max |numerical - formula| = {np.max(np.abs(Rs - R_formula)):.2e}')

fig, ax = plt.subplots()
ax.plot(gs, Rs, label='$R = p_1 / p_2$')
ax.axhline(1 / beta, linestyle='--', color='red', label=r'$\beta^{-1}$')
ax.set_xlabel('endowment growth rate $g$')
ax.set_ylabel('gross interest rate $R$')
ax.legend()
plt.show()

for g in [0, 0.1, 0.2]:
    print(f'g = {g:>4}:  R = {R_of_g(g):.6f}')
```

The numerical and analytical answers agree, and $R = \beta^{-1} = 1.052632$ when $g = 0$.

For part c, $g > 0$ tilts the endowment toward period $2$, and the consumer wants a flat consumption path.

At $R = \beta^{-1}$ the consumer would borrow against period $2$ income.

A single consumer has nobody to borrow from, so $R$ rises until the consumer willingly consumes the endowment.

```{solution-end}
```

### Risk and state-contingent claims

We study risk in the context of a **static** environment, meaning that there is only one period.

By **risk** we mean that an outcome is not known in advance, but that it is governed by a known probability distribution.

That our consumer confronts **risk** means in particular that

  * there are two states of nature, $1$ and $2$.

  * the consumer knows that the probability that state $1$ occurs is $\lambda$.

  * the consumer knows that the probability that state $2$ occurs is $(1-\lambda)$.

Before the outcome is realized, the consumer's **expected utility** is

$$
- \frac{1}{2} [\lambda (c_1 - b_1)^2 + (1-\lambda)(c_2 - b_2)^2]
$$

where

* $c_1$ is consumption in state $1$
* $c_2$ is consumption in state $2$

To capture these preferences we set

$$
\Pi = \begin{bmatrix} \sqrt{\lambda} & 0 \cr
                     0  & \sqrt{1-\lambda} \end{bmatrix}
$$

$$
e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}
$$

+++

$$
b = \begin{bmatrix} \sqrt{\lambda}b_1 \cr \sqrt{1-\lambda}b_2 \end{bmatrix}
$$

A consumer's consumption vector is

$$
c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}
$$

A price vector is

$$
p = \begin{bmatrix} p_1 \cr p_2 \end{bmatrix}
$$

where $p_i$ is the price of one unit of consumption in state $i \in \{1, 2\}$.

The state-contingent goods being traded are often called **Arrow securities**.

Before the random state of the world $i$ is realized, the consumer sells their state-contingent endowment bundle and purchases a state-contingent consumption bundle.

Trading such state-contingent goods is one way economists often model **insurance**.

+++

Here is an instance of the risk economy:

```{code-cell} ipython3
prob = 0.2

Π = np.array([[np.sqrt(prob), 0],
              [0, np.sqrt(1 - prob)]])

b = np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])

e = np.array([1, 1])

risk = ExchangeEconomy(Π, b, e)
p, c = risk.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c)
```

```{exercise}
:label: sdm_ex3

Consider the instance above.

Please numerically study how each of the following cases affects the equilibrium prices and allocations:

* the consumer gets poorer,
* they like the first good more, or
* the probability that state $1$ occurs is higher.

Hint: for each case, choose a value of $e$, $b$, or $\lambda$ that differs from the one used in the instance above.

```

+++

```{solution-start} sdm_ex3
:class: dropdown
```

A single-consumer endowment economy has no production and no trading partner, so the allocation equals the endowment.

Only prices respond to the experiments below.

We build a fresh economy for each experiment.

```{code-cell} ipython3
def risk_economy(prob=0.2, b_scale=(5, 5), e=(1, 1)):
    """
    Build the risk economy, allowing each element of the baseline to be changed
    """
    Π = np.array([[np.sqrt(prob), 0],
                  [0, np.sqrt(1 - prob)]])
    b = np.array([np.sqrt(prob) * b_scale[0],
                  np.sqrt(1 - prob) * b_scale[1]])
    return ExchangeEconomy(Π, b, np.array(e))


def show(economy, label):
    p, c = economy.competitive_equilibrium()
    print(f'{label}')
    print(f'  price vector: {p}')
    print(f'  allocation  : {c}\n')


show(risk_economy(), 'baseline')
show(risk_economy(e=(0.5, 0.5)), 'the consumer is poorer')
show(risk_economy(b_scale=(6, 5)), 'the consumer likes the first good more')
show(risk_economy(prob=0.8), 'state 1 is more likely')
```

When the consumer is poorer, goods are scarcer relative to the bliss point, so **both** state-contingent prices rise.

When the consumer likes the first good more, the price of a claim on state $1$ rises while the price of a claim on state $2$ is unchanged.

When state $1$ becomes more likely, a claim that pays in state $1$ becomes more valuable and a claim that pays in state $2$ becomes less valuable.

```{solution-end}
```

```{exercise}
:label: sdm_ex6

In the risk interpretation, $p_i$ is the price of one unit of consumption contingent on state $i$.

a. Show that, with $\Pi = \operatorname{diag}(\sqrt{\lambda}, \sqrt{1-\lambda})$ and
    $b = (\sqrt{\lambda}\,\bar b, \ \sqrt{1-\lambda}\,\bar b)^\top$,

$$
\frac{p_1}{p_2} = \frac{\lambda}{1-\lambda} \cdot \frac{\bar b - e_1}{\bar b - e_2}
$$

b. Conclude that when the endowment is the same in both states, state prices are proportional to probabilities, so that claims trade at **actuarially fair odds**.

c. Verify both claims numerically for $\lambda \in \{0.2, 0.5, 0.8\}$, first with $e = (1,1)$ and then with $e = (0.5, 1.5)$.

Explain the sign of the departure from fair odds in the second case.

```

```{solution-start} sdm_ex6
:class: dropdown
```

For part a, again $p = \Pi^\top b - \Pi^\top \Pi e$, and here

$$
p = \begin{bmatrix} \lambda (\bar b - e_1) \cr (1-\lambda)(\bar b - e_2) \end{bmatrix}
$$

from which the ratio follows immediately.

Part b is then immediate too: if $e_1 = e_2$, the second factor equals $1$, so $p_1/p_2 = \lambda/(1-\lambda)$.

```{code-cell} ipython3
b_bar = 5

def price_ratio(λ, e):
    Π = np.array([[np.sqrt(λ), 0],
                  [0, np.sqrt(1 - λ)]])
    b = np.array([np.sqrt(λ) * b_bar, np.sqrt(1 - λ) * b_bar])
    p, c = ExchangeEconomy(Π, b, np.array(e)).competitive_equilibrium()
    return p[0] / p[1]

print(f"{'λ':>5} | {'e':>12} | {'p1/p2':>9} | {'odds λ/(1-λ)':>13}")
print('-' * 48)
for λ in [0.2, 0.5, 0.8]:
    for e in [[1, 1], [0.5, 1.5]]:
        print(f'{λ:>5} | {str(e):>12} | {price_ratio(λ, e):>9.4f} | {λ / (1 - λ):>13.4f}')
```

With $e = (1,1)$ the price ratio equals the odds ratio exactly.

With $e = (0.5, 1.5)$ the good is scarcer in state $1$, and $p_1/p_2$ exceeds $\lambda/(1-\lambda)$ for every $\lambda$.

State prices reflect probabilities and scarcity.

Asset prices are expectations taken with respect to a probability distribution twisted toward states in which consumption is low.

```{solution-end}
```

+++

## Economies with endogenous supplies of goods

Up to now we have described a pure exchange economy in which endowments of goods are exogenous, meaning that they are taken as given from outside the model.

### Supply curve of a competitive firm

A competitive firm that can produce goods takes a price vector $p$ as given and chooses a quantity $q$
to maximize total revenue minus total costs.

The firm's total revenue equals $p^\top q$ and its total cost equals $C(q)$ where $C(q)$ is a total cost function

$$
C(q) = h ^\top q +  \frac{1}{2} q^\top J q
$$


and $J$ is a positive definite matrix.


So the firm's profits are

$$
p^\top q - C(q)
$$ (eq:compprofits)



An $n\times 1$ vector of **marginal costs** is

$$
\frac{\partial C(q)}{\partial q} = h + H q
$$

where

$$
H =  \frac{1}{2} (J + J^\top)
$$

The firm maximizes total profits by setting **marginal revenue equal to marginal costs**.

An $n \times 1$ vector of marginal revenues for the price-taking firm is $\frac{\partial p^\top q}
{\partial q} = p $.

So **price equals marginal revenue** for our price-taking competitive firm.

This leads to the following **inverse supply curve** for the competitive firm:


$$
p = h + H q
$$




### Competitive equilibrium


We equate the inverse supply curve to the inverse demand curve and solve for the equilibrium quantity vector.

We then compute the equilibrium price vector from either curve.

That quantity vector also solves a planning problem, as we show below.

#### $\mu=1$ warmup

As a special case, let's pin down a demand curve by setting the marginal utility of wealth $\mu =1$.

Equating supply price to demand price and letting $q=c$ we get

$$
p = h + H c = \Pi^\top b - \Pi^\top \Pi c ,
$$

which implies the equilibrium quantity vector

$$
c = (\Pi^\top \Pi + H )^{-1} ( \Pi^\top b - h)
$$ (eq:old5)

This equation is the counterpart, for the scalar $n=1$ model of {doc}`intro_supply_demand`, of the equilibrium quantity derived there.

#### General $\mu\neq 1$ case

Now let's extend the preceding analysis to a more
general case by allowing $\mu \neq 1$.

Then the inverse demand curve is

$$
p = \mu^{-1} [\Pi^\top b - \Pi^\top \Pi c]
$$ (eq:old5pa)

Equating this to the inverse supply curve, letting $q=c$ and solving
for $c$ gives

$$
c = [\Pi^\top \Pi + \mu H]^{-1} [ \Pi^\top b - \mu h]
$$ (eq:old5p)

+++

### Multi-good welfare maximization problem

Our welfare maximization problem -- also sometimes called a social planning problem  -- is to choose $c$ to maximize

$$
    - \frac{1}{2} \mu^{-1}(\Pi c -b) ^\top (\Pi c -b )
$$

minus the area under the inverse supply curve, namely,

$$
    h^\top c +  \frac{1}{2} c^\top J c  
$$

So the welfare criterion is

$$
    - \frac{1}{2} \mu^{-1}(\Pi c -b)^\top (\Pi c -b ) -h^\top c 
        -  \frac{1}{2} c^\top J c
$$

In this formulation, $\mu$ is a parameter that describes how the planner weighs interests of outside suppliers and our representative consumer.

The first-order condition with respect to $c$ is

$$
- \mu^{-1} \Pi^\top \Pi c + \mu^{-1}\Pi^\top b - h -  H c = 0
$$

which implies {eq}`eq:old5p`.

Thus, as for the single-good case, with multiple goods a competitive equilibrium quantity vector solves a planning problem.

(This is another version of the first welfare theorem.)

#### Welfare as consumer surplus plus producer surplus

{doc}`intro_supply_demand` measured social welfare by consumer surplus plus producer surplus.

Our welfare criterion equals that measure up to a constant.

Consumer surplus is the area under the inverse demand curve minus expenditure.

Producer surplus is revenue minus the area under the inverse supply curve.

Adding them cancels revenue against expenditure:

$$
CS + PS = \mu^{-1} b^\top \Pi c - \frac{1}{2}\mu^{-1} c^\top \Pi^\top \Pi c
        - h^\top c - \frac{1}{2} c^\top J c
$$

Expanding the quadratic form in the welfare criterion gives

$$
- \frac{1}{2} \mu^{-1}(\Pi c -b)^\top (\Pi c -b ) - h^\top c - \frac{1}{2} c^\top J c
  = CS + PS - \frac{1}{2}\mu^{-1} b^\top b
$$

The constant $\frac{1}{2}\mu^{-1} b^\top b$ does not depend on $c$.

Maximizing the welfare criterion and maximizing total surplus are therefore the same problem, and the competitive equilibrium quantity vector {eq}`eq:old5p` maximizes consumer surplus plus producer surplus.

An exercise below computes both for a single good.

### Implementation

A Production Economy will consist of

* a single **person** that we'll interpret as a representative consumer
* a single set of **production costs**
* a multiplier $\mu$ that weights "consumers" versus "producers" in the planner's welfare function described above
* an $n \times 1$ vector $p$ of competitive equilibrium prices
* an $n \times 1$ vector $c$ of competitive equilibrium quantities
* **consumer surplus**
* **producer surplus**

Here we define a class ``ProductionEconomy``.

```{code-cell} ipython3
class ProductionEconomy:
    
    def __init__(self, 
                 Π, 
                 b, 
                 h, 
                 J, 
                 μ):
        """
        Set up the environment for a production economy

        Args:
            Π (np.ndarray): matrix of substitution
            b (np.array): bliss points
            h (np.array): h in cost func
            J (np.ndarray): J in cost func
            μ (float): welfare weight of the corresponding planning problem
        """
        self.n = len(b)
        self.Π, self.b, self.h, self.J, self.μ = Π, b, h, J, μ
        
    def competitive_equilibrium(self):
        """
        Compute a competitive equilibrium of the production economy
        """
        Π, b, h, μ, J = self.Π, self.b, self.h, self.μ, self.J
        H = .5 * (J + J.T)

        # allocation
        c = inv(Π.T @ Π + μ * H) @ (Π.T @ b - μ * h)

        # price
        p = 1 / μ * (Π.T @ b - Π.T @ Π @ c)

        # check non-satiation
        if any(Π @ c - b >= 0):
            raise Exception('invalid result: set bliss points further away')

        return c, p

    def compute_surplus(self):
        """
        Compute consumer and producer surplus for single good case
        """
        if self.n != 1:
            raise Exception('not single good')
        h, J, Π, b, μ = self.h.item(), self.J.item(), self.Π.item(), self.b.item(), self.μ
        H = J

        # supply/demand curve coefficients
        s0, s1 = h, H
        d0, d1 = 1 / μ * Π * b, 1 / μ * Π**2

        # competitive equilibrium
        c, p = self.competitive_equilibrium()

        # calculate surplus
        c_surplus = d0 * c - .5 * d1 * c**2 - p * c
        p_surplus = p * c - s0 * c - .5 * s1 * c**2

        return c_surplus, p_surplus
```

Then define a function that plots demand and supply curves and labels surpluses and equilibrium.

```{code-cell} ipython3
:tags: [hide-input]

def plot_competitive_equilibrium(PE):
    """
    Plot demand and supply curves, producer/consumer surpluses, and equilibrium for
    a single good production economy

    Args:
        PE (class): An initialized production economy class
    """
    # get singleton value
    J, h, Π, b, μ = PE.J.item(), PE.h.item(), PE.Π.item(), PE.b.item(), PE.μ
    H = J

    # compute competitive equilibrium
    c, p = PE.competitive_equilibrium()
    c, p = c.item(), p.item()

    # inverse supply/demand curve
    supply_inv = lambda x: h + H * x
    demand_inv = lambda x: 1 / μ * (Π * b - Π * Π * x)

    xs = np.linspace(0, 2 * c, 100)
    ps = np.ones(100) * p
    supply_curve = supply_inv(xs)
    demand_curve = demand_inv(xs)

    # plot
    plt.figure()
    plt.plot(xs, supply_curve, label='Supply', color='#020060')
    plt.plot(xs, demand_curve, label='Demand', color='#600001')

    plt.fill_between(xs[xs <= c], demand_curve[xs <= c], ps[xs <= c], label='Consumer surplus', color='#EED1CF')
    plt.fill_between(xs[xs <= c], supply_curve[xs <= c], ps[xs <= c], label='Producer surplus', color='#E6E6F5')

    plt.vlines(c, 0, p, linestyle="dashed", color='black', alpha=0.7)
    plt.hlines(p, 0, c, linestyle="dashed", color='black', alpha=0.7)
    plt.scatter(c, p, zorder=10, label='Competitive equilibrium', color='#600001')

    plt.legend(loc='upper right')
    plt.margins(x=0, y=0)
    plt.ylim(0)
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.show()
```

#### Example: single agent with one good and production

Now let's construct an example of a production economy with one good.

To do this we

  * specify a single **person** and a **cost curve** in a way that lets us replicate the simple single-good supply and demand example of {doc}`intro_supply_demand`

  * compute equilibrium $p$ and $c$ and consumer and producer surpluses

  * draw graphs of both surpluses

  * do experiments in which we shift $b$ and watch what happens to $p, c$.

```{code-cell} ipython3
Π = np.array([[1]])  # the matrix now is a singleton
b = np.array([10])
h = np.array([0.5])
J = np.array([[1]])
μ = 1

PE = ProductionEconomy(Π, b, h, J, μ)
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p.item())
print('Competitive equilibrium allocation:', c.item())

# plot
plot_competitive_equilibrium(PE)
```

```{code-cell} ipython3
c_surplus, p_surplus = PE.compute_surplus()

print('Consumer surplus:', c_surplus.item())
print('Producer surplus:', p_surplus.item())
```

Let's give the consumer a lower welfare weight by raising $\mu$.

```{code-cell} ipython3
PE.μ = 2
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p.item())
print('Competitive equilibrium allocation:', c.item())

# plot
plot_competitive_equilibrium(PE)
```

```{code-cell} ipython3
c_surplus, p_surplus = PE.compute_surplus()

print('Consumer surplus:', c_surplus.item())
print('Producer surplus:', p_surplus.item())
```

Raising $\mu$ from $1$ to $2$ halves the consumer's willingness to pay for each unit, $\mu^{-1}(\Pi b - \Pi^2 c)$.

The equilibrium quantity falls from $4.75$ to $3$ and the price falls from $5.25$ to $3.5$.

Both surpluses fall.

The two experiments hold $h$ and $J$ fixed, so both surpluses are measured in the same units.

The two experiments weight the consumer differently, so their surpluses do not rank welfare across them.

Within each experiment, consumer surplus plus producer surplus is maximized at the competitive quantity.

Now we change the bliss point so that the consumer derives more utility from consumption.

```{code-cell} ipython3
PE.μ = 1
PE.b = PE.b * 1.5
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p.item())
print('Competitive equilibrium allocation:', c.item())

# plot
plot_competitive_equilibrium(PE)
```

This raises both the equilibrium price and quantity.


#### Example: single agent two-good economy with production

Now we do experiments like those above, but with two goods.

We begin with a **diagonal** $\Pi$, so that the two goods are independent in preferences, together with a cost matrix $J$ whose off-diagonal element is positive, so that producing more of one good raises the marginal cost of the other.

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1]])

b = np.array([10, 10])

h = np.array([0.5, 0.5])

J = np.array([[1, 0.5],
              [0.5, 1]])
μ = 1

PE = ProductionEconomy(Π, b, h, J, μ)
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)
```

Now let's raise the bliss point for the first good from $10$ to $12$.

```{code-cell} ipython3
PE.b = np.array([12, 10])

c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)
```

The quantity of good $1$ rises while the quantity of good $2$ **falls**, even though preferences for good $2$ have not changed.

That cross-effect comes entirely from the cost side: producing more of good $1$ raises the marginal cost of good $2$.

Next we make $\Pi$ **non-diagonal**, so that the two goods interact in preferences as well as in costs.

```{code-cell} ipython3
PE.Π = np.array([[1, 0.3],
                 [0.3, 1]])

PE.b = np.array([10, 10])

c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)
```

```{code-cell} ipython3
PE.b = np.array([12, 10])
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)
```

The same shift in $b_1$ now moves both quantities by more: good $1$ rises by $1.11$ rather than $1.07$, and good $2$ falls by $0.30$ rather than $0.27$.

A non-diagonal $\Pi$ puts a positive off-diagonal element in $\Pi^\top \Pi$.

The cross-partial derivative of utility becomes $\partial^2 u / \partial c_1 \partial c_2 = -0.6$, so the two goods substitute for each other in preferences.

Preference substitution reinforces cost substitution and pushes $c_2$ down.

A non-diagonal $\Pi$ also raises both components of $\Pi^\top b$ when $b_1$ rises, which pushes $c_2$ up.

The first effect dominates.

```{note}
The two effects can be separated in $\Delta c = (\Pi^\top \Pi + H)^{-1} \Pi^\top \Delta b$.

Holding $\Pi^\top \Pi$ at $I$ and letting only $\Pi^\top b$ change gives $\Delta c = (0.99, 0.05)$.

Holding $\Pi^\top \Delta b$ at $(2,0)$ and letting only $\Pi^\top \Pi$ change gives $\Delta c = (1.32, -0.70)$.

The two together give $\Delta c = (1.11, -0.30)$.
```

### A monopolist supplier

A competitive firm is a **price-taker** who regards the price and therefore its marginal revenue as being beyond its control.

A monopolist knows that it has no competition and can influence the price and its marginal revenue by
setting quantity.

A monopolist takes a **demand curve** and not the **price** as beyond its control.

Thus, instead of being a price-taker, a monopolist chooses a quantity to maximize profits subject to the inverse demand curve
{eq}`eq:old5pa`.

So the monopolist's total profits as a function of its output $q$ is

$$
[\mu^{-1} \Pi^\top (b - \Pi q)]^\top  q - h^\top q -  \frac{1}{2} q^\top J q
$$ (eq:monopprof)

After finding
first-order necessary conditions for maximizing monopoly profits with respect to $q$
and solving them for $q$, we find that the monopolist sets

$$
q = (H + 2 \mu^{-1} \Pi^\top \Pi)^{-1} (\mu^{-1} \Pi^\top b - h)
$$ (eq:qmonop)

```{exercise}
:label: sdm_ex4

Please  verify the monopolist's supply curve {eq}`eq:qmonop`.

```

```{solution-start} sdm_ex4
:class: dropdown
```

Write the monopolist's profits {eq}`eq:monopprof` as

$$
\mu^{-1} b^\top \Pi q - \mu^{-1} q^\top \Pi^\top \Pi q - h^\top q - \frac{1}{2} q^\top J q
$$

Now apply the formulas for differentiating an inner product and a quadratic form, recalling that $H = \frac{1}{2}(J + J^\top)$:

$$
\frac{\partial}{\partial q}\left(\mu^{-1} b^\top \Pi q\right) = \mu^{-1}\Pi^\top b, \qquad
\frac{\partial}{\partial q}\left(\mu^{-1} q^\top \Pi^\top \Pi q\right) = 2 \mu^{-1}\Pi^\top \Pi q
$$

$$
\frac{\partial}{\partial q}\left(h^\top q\right) = h, \qquad
\frac{\partial}{\partial q}\left(\frac{1}{2} q^\top J q\right) = H q
$$

The first-order condition is therefore

$$
\mu^{-1}\Pi^\top b - 2 \mu^{-1}\Pi^\top \Pi q - h - H q = 0
$$

Collecting terms in $q$ gives

$$
\left(H + 2\mu^{-1}\Pi^\top\Pi\right) q = \mu^{-1}\Pi^\top b - h
$$

which is {eq}`eq:qmonop`.

The Hessian of profits with respect to $q$ is $-(H + 2\mu^{-1}\Pi^\top \Pi)$.

Because $H$ and $\Pi^\top \Pi$ are both positive definite, this Hessian is negative definite, so the first-order condition does indeed describe a maximum.

```{solution-end}
```

Let's now compare a monopolist with a competitive supplier.

Recall that in a competitive equilibrium, a price-taking supplier equates marginal revenue $p$ to marginal cost $h + Hq$.

A monopolist's marginal revenue is not constant but instead is a non-trivial function of the quantity it sets.

The monopolist's marginal revenue is

$$
MR(q) = -2\mu^{-1}\Pi^{\top}\Pi q+\mu^{-1}\Pi^{\top}b,
$$

which the monopolist equates to its marginal cost.

Below we define a class `Monopoly` that inherits from `ProductionEconomy` and adds a method that computes an equilibrium price and allocation when the supplier is a monopolist.

Since the supplier now has price-setting power

- we first compute the optimal quantity that solves the monopolist's profit maximization problem
- then we back out an equilibrium price from the consumer's inverse demand curve

```{code-cell} ipython3
class Monopoly(ProductionEconomy):
    
    def __init__(self, 
                 Π, 
                 b, 
                 h, 
                 J, 
                 μ):
        """
        Inherit all properties and methods from class ProductionEconomy
        """
        super().__init__(Π, b, h, J, μ)
        

    def equilibrium_with_monopoly(self):
        """
        Compute the equilibrium price and allocation when there is a monopolist supplier
        """
        Π, b, h, μ, J = self.Π, self.b, self.h, self.μ, self.J
        H = .5 * (J + J.T)

        # allocation
        q = inv(μ * H + 2 * Π.T @ Π) @ (Π.T @ b - μ * h)

        # price
        p = 1 / μ * (Π.T @ b - Π.T @ Π @ q)

        if any(Π @ q - b >= 0):
            raise Exception('invalid result: set bliss points further away')

        return q, p
```

Define a function that plots the demand, marginal cost and marginal revenue curves with surpluses and equilibrium labelled.

```{code-cell} ipython3
:tags: [hide-input]

def plot_monopoly(M):
    """
    Plot demand curve, marginal production cost and revenue, surpluses and the
    equilibrium in a monopolist supplier economy with a single good

    Args:
        M (class): An instance of a class that inherits from ProductionEconomy
    """
    # get singleton value
    J, h, Π, b, μ = M.J.item(), M.h.item(), M.Π.item(), M.b.item(), M.μ
    H = J

    # compute competitive equilibrium
    c, p = M.competitive_equilibrium()
    q, pm = M.equilibrium_with_monopoly()
    c, p, q, pm = c.item(), p.item(), q.item(), pm.item()

    # inverse supply/demand curve
    marg_cost = lambda x: h + H * x
    marg_rev = lambda x: -2 * 1 / μ * Π * Π * x + 1 / μ * Π * b
    demand_inv = lambda x: 1 / μ * (Π * b - Π * Π * x)

    xs = np.linspace(0, 2 * c, 100)
    pms = np.ones(100) * pm
    marg_cost_curve = marg_cost(xs)
    marg_rev_curve = marg_rev(xs)
    demand_curve = demand_inv(xs)

    # plot
    plt.figure()
    plt.plot(xs, marg_cost_curve, label='Marginal cost', color='#020060')
    plt.plot(xs, marg_rev_curve, label='Marginal revenue', color='#E55B13')
    plt.plot(xs, demand_curve, label='Demand', color='#600001')

    plt.fill_between(xs[xs <= q], demand_curve[xs <= q], pms[xs <= q], label='Consumer surplus', color='#EED1CF')
    plt.fill_between(xs[xs <= q], marg_cost_curve[xs <= q], pms[xs <= q], label='Producer surplus', color='#E6E6F5')

    plt.vlines(c, 0, p, linestyle="dashed", color='black', alpha=0.7)
    plt.hlines(p, 0, c, linestyle="dashed", color='black', alpha=0.7)
    plt.scatter(c, p, zorder=10, label='Competitive equilibrium', color='#600001')

    plt.vlines(q, 0, pm, linestyle="dashed", color='black', alpha=0.7)
    plt.hlines(pm, 0, q, linestyle="dashed", color='black', alpha=0.7)
    plt.scatter(q, pm, zorder=10, label='Equilibrium with monopoly', color='#E55B13')

    plt.legend(loc='upper right')
    plt.margins(x=0, y=0)
    plt.ylim(0)
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.show()
```

#### A single-good example

```{code-cell} ipython3
Π = np.array([[1]])  # the matrix now is a singleton
b = np.array([10])
h = np.array([0.5])
J = np.array([[1]])
μ = 1

M = Monopoly(Π, b, h, J, μ)
c, p = M.competitive_equilibrium()
q, pm = M.equilibrium_with_monopoly()

print('Competitive equilibrium price:', p.item())
print('Competitive equilibrium allocation:', c.item())

print('Equilibrium with monopolist supplier price:', pm.item())
print('Equilibrium with monopolist supplier allocation:', q.item())

# plot
plot_monopoly(M)
```

The monopolist sets output below the competitive equilibrium quantity, and in this single-good economy that lower quantity is associated with a higher price.

```{exercise}
:label: sdm_ex7

For the single-good economy just studied, with $\Pi = [1]$, $b = [10]$, $h = [0.5]$, $J = [[1]]$ and $\mu = 1$, define **total surplus** at quantity $x$ as the area under the inverse demand curve minus the area under the marginal cost curve:

$$
TS(x) = \mu^{-1}\left(\Pi b x - \frac{1}{2}\Pi^2 x^2\right) - \left(h x + \frac{1}{2} J x^2\right)
$$

a. Compute $TS$ at the competitive quantity and at the monopoly quantity, and report the **deadweight loss** caused by monopoly.

b. Maximize $TS$ over a fine grid of values of $x$ and check that the maximizer is the *competitive* quantity rather than the monopoly quantity.

Which welfare theorem does this illustrate?

```

```{solution-start} sdm_ex7
:class: dropdown
```

```{code-cell} ipython3
Π = np.array([[1]])
b = np.array([10])
h = np.array([0.5])
J = np.array([[1]])
μ = 1

M = Monopoly(Π, b, h, J, μ)
c, p = M.competitive_equilibrium()
q, pm = M.equilibrium_with_monopoly()
c, p, q, pm = c.item(), p.item(), q.item(), pm.item()

Π_, b_, h_, J_ = Π.item(), b.item(), h.item(), J.item()

def total_surplus(x):
    return (Π_ * b_ * x - .5 * Π_**2 * x**2) / μ - (h_ * x + .5 * J_ * x**2)

print(f'competitive: q = {c:.4f}, p = {p:.4f}, total surplus = {total_surplus(c):.4f}')
print(f'monopoly   : q = {q:.4f}, p = {pm:.4f}, total surplus = {total_surplus(q):.4f}')
print(f'deadweight loss = {total_surplus(c) - total_surplus(q):.4f}')

xs = np.linspace(0, 2 * c, 100001)
print(f'\nquantity that maximizes total surplus: {xs[np.argmax(total_surplus(xs))]:.4f}')
```

Let's draw the deadweight loss as the area between the demand curve and the marginal cost curve, over the output that the monopolist declines to produce.

```{code-cell} ipython3
:tags: [hide-input]

xs = np.linspace(0, 2 * c, 200)
demand_inv = lambda x: (Π_ * b_ - Π_**2 * x) / μ
marg_cost = lambda x: h_ + J_ * x

fig, ax = plt.subplots()
ax.plot(xs, demand_inv(xs), label='Demand', color='#600001')
ax.plot(xs, marg_cost(xs), label='Marginal cost', color='#020060')

mask = (xs >= q) & (xs <= c)
ax.fill_between(xs[mask], marg_cost(xs[mask]), demand_inv(xs[mask]),
                color='#BBBBBB', label='Deadweight loss')

ax.scatter(c, p, zorder=10, label='Competitive equilibrium', color='#600001')
ax.scatter(q, pm, zorder=10, label='Equilibrium with monopoly', color='#E55B13')

ax.set_xlabel('Quantity')
ax.set_ylabel('Price')
ax.legend(loc='upper right')
plt.show()
```

The competitive quantity $4.75$ maximizes total surplus, while the monopolist restricts output to $3.1667$ and so destroys $2.5069$ of surplus, the shaded triangle between the demand curve and the marginal cost curve.

That the competitive quantity solves the planner's problem is the **first welfare theorem**, here in its single-good form.

The monopolist violates the hypothesis of that theorem because it is not a price taker.

```{solution-end}
```

#### A multiple good example

Let's compare competitive equilibrium and monopoly outcomes in a multiple goods economy.

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1.2]])

b = np.array([10, 10])

h = np.array([0.5, 0.5])

J = np.array([[1, 0.5],
              [0.5, 1]])
μ = 1

M = Monopoly(Π, b, h, J, μ)
c, p = M.competitive_equilibrium()
q, pm = M.equilibrium_with_monopoly()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)

print('Equilibrium with monopolist supplier price:', pm)
print('Equilibrium with monopolist supplier allocation:', q)
```

Here the monopolist restricts output of both goods.

```{note}
With a single good, a monopolist produces less than a competitive industry does.

With several goods, off-diagonal elements of $\Pi$ or $J$ can lead a monopolist to produce more of one good than a competitive industry would.

The welfare comparison survives.

The monopoly quantity vector does not maximize the planner's criterion, so total surplus falls under monopoly.
```

## Concluding remarks

This lecture studied competitive equilibria in an economy with many goods, first in pure exchange and then with production.

The same mathematical structure describes **dynamics** and **risk**, once we index goods by dates or by states of the world.

A competitive equilibrium quantity vector solves a planning problem, a version of the first welfare theorem.

A monopolist violates the hypothesis of that theorem and destroys surplus.

Our economy contained a single representative consumer, so no question about the distribution of wealth arose.

{doc}`supply_demand_heterogeneity` lets consumers differ in their preferences and endowments and studies how a competitive equilibrium distributes goods among them.
