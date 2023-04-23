# Supply and Demand with Many Goods

## Overview

In a {doc}`previous lecture <intro_supply_demand>` we studied supply, demand
and welfare in a market with just one good.

In this lecture, we study a setting with $n$ goods and $n$ corresponding prices.

We shall describe two classic welfare theorems:

* **first welfare theorem:** for a  given a distribution of wealth among consumers,  a competitive  equilibrium  allocation of goods solves a  social planning problem.

* **second welfare theorem:** An allocation of goods to consumers that solves a social planning problem can be supported by a competitive equilibrium with an appropriate initial distribution of  wealth.



## Formulas from linear algebra

We  shall apply  formulas from linear algebra that

* differentiate an inner product with respect to each vector
* differentiate a product of a matrix and a vector with respect to the vector
* differentiate a quadratic form in a vector with respect to the vector

Where $a$ is an $n \times 1$ vector, $A$ is an $n \times n$ matrix, and $x$ is an $n \times 1$ vector:

$$
\frac{\partial a^\top x }{\partial x} = a
$$

$$
\frac{\partial A x} {\partial x} = A
$$

$$
\frac{\partial x^\top A x}{\partial x} = (A + A^\top)x
$$

## From utility function to demand curve

Let

* $\Pi$ be an $m \times n$ matrix,
* $c$ be an $n \times 1$ vector of consumptions of various goods,
* $b$ be an $m \times 1$ vector of bliss points,
* $e$ be an $n \times 1$ vector of endowments, and
* $p$ be an $n \times 1$ vector of prices

We assume that $\Pi$ has linearly independent columns, which implies that $\Pi^\top \Pi$ is a positive definite matrix.

* it follows that $\Pi^\top \Pi$ has an inverse.

The matrix $\Pi$ describes a consumer's willingness to substitute one good for every other good.

We shall see below that $(\Pi^T \Pi)^{-1}$ is a matrix of slopes of (compensated) demand curves for $c$ with respect to a vector of prices:

$$
    \frac{\partial c } {\partial p} = (\Pi^T \Pi)^{-1}
$$

A consumer faces $p$ as a price taker and chooses $c$ to maximize the utility function

$$
    -.5 (\Pi c -b) ^\top (\Pi c -b )
$$ (eq:old0)

subject to the budget constraint

$$
    p^\top (c -e ) = 0
$$ (eq:old2)

We shall specify examples in which  $\Pi$ and $b$ are such that it typically happens that

$$
    \Pi c < < b
$$ (eq:bversusc)

so that utility function {eq}`eq:old2` tells us that the consumer has much less  of each good than he wants.

Condition {eq}`eq:bversusc` will ultimately  assure us that competitive equilibrium prices  are  positive.

### Demand Curve Implied  by Constrained Utility Maximization

For now, we assume that the budget constraint is {eq}`eq:old2`.

So we'll be deriving what is known as  a **Marshallian** demand curve.

Form a Lagrangian

$$ L = -.5 (\Pi c -b) ^\top (\Pi c -b ) + \mu [p^\top (e-c)] $$

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
    \mu(p,e) = \frac{p^\top ( \Pi^\top \Pi )^{-1} \Pi^\top b - p^\top e}{p^\top (\Pi^\top \Pi )^{-1} p}.
$$ (eq:old4)

Equation {eq}`eq:old4` tells how marginal utility of wealth depends on  the endowment vector  $e$ and the price vector  $p$.

**Remark:** Equation {eq}`eq:old4` is a consequence of imposing that $p^\top (c - e) = 0$.  We could instead take $\mu$ as a parameter and use {eq}`eq:old3` and the budget constraint {eq}`eq:old2p` to solve for $W.$ Which way we proceed determines whether we are constructing a **Marshallian** or **Hicksian** demand curve.


## Endowment economy

We now study a pure-exchange economy, or what is sometimes called an  endowment economy.

Consider a single-consumer, multiple-goods economy without production.

The only source of goods is the single consumer's endowment vector   $e$.

A competitive equilibrium price vector  induces the consumer to choose  $c=e$.

This implies that the equilibrium price vector satisfies

$$
p = \mu^{-1} (\Pi^\top b - \Pi^\top \Pi e)
$$

In the present case where we have imposed budget constraint in the form {eq}`eq:old2`, we are free to normalize the price vector by setting the marginal utility of wealth $\mu =1$ (or any other value for that matter).

This amounts to choosing a common  unit (or numeraire) in which prices of all goods are expressed.

(Doubling all prices will affect neither quantities nor relative prices.)

We'll set $\mu=1$.

**Exercise:** Verify that setting $\mu=1$ in {eq}`eq:old3` implies that   formula {eq}`eq:old4` is satisfied.

**Exercise:** Verify that setting  $\mu=2$ in {eq}`eq:old3` also implies that formula {eq}`eq:old4` is satisfied.


## Digression: Marshallian and Hicksian Demand Curves

**Remark:** Sometimes we'll use budget constraint {eq}`eq:old2` in situations in which a consumers's endowment vector $e$ is his **only** source of income. Other times we'll instead assume that the consumer has another source of income (positive or negative) and write his budget constraint as

$$
p ^\top (c -e ) = W
$$ (eq:old2p)

where $W$ is measured in "dollars" (or some other **numeraire**) and component $p_i$ of the price vector is measured in dollars per unit of good $i$.

Whether the consumer's budget constraint is  {eq}`eq:old2` or {eq}`eq:old2p` and whether we take $W$ as a free parameter or instead as an endogenous variable   will  affect the consumer's marginal utility of wealth.

Consequently, how we set $\mu$  determines whether we are constructing

* a **Marshallian** demand curve, as when we use {eq}`eq:old2` and solve for $\mu$ using equation {eq}`eq:old4` below, or
* a **Hicksian** demand curve, as when we  treat $\mu$ as a fixed parameter and solve for $W$ from {eq}`eq:old2p`.

Marshallian and Hicksian demand curves contemplate different mental experiments:

* For a Marshallian demand curve, hypothetical changes in a price vector  have  both **substitution** and **income** effects

  * income effects are consequences of  changes in $p^\top e$ associated with the change in the price vector

* For a Hicksian demand curve, hypothetical price vector  changes  have only **substitution**  effects

  * changes in the price vector leave the $p^\top e + W$ unaltered because we freeze $\mu$ and solve for $W$

Sometimes a Hicksian demand curve is called a **compensated** demand curve in order to emphasize that, to disarm the income (or wealth) effect associated with a price change, the consumer's wealth $W$ is adjusted.

We'll discuss these distinct demand curves more  below.



## Dynamics and Risk as Special Cases of Pure Exchange Economy

Special cases of our $n$-good pure exchange  model can be created to represent

* dynamics
  - by putting different dates on different commodities
* risk
  - by interpreting delivery  of goods as being contingent on states of the world whose realizations are described by a **known probability distribution**

Let's illustrate how.

### Dynamics

Suppose that we want to represent a utility function

$$
  -.5 [(c_1 - b_1)^2 + \beta (c_2 - b_2)^2]
$$

where $\beta \in (0,1)$ is a discount factor, $c_1$ is consumption at time $1$ and $c_2$ is consumption at time 2.

To capture this with our quadratic utility function {eq}`eq:old0`,  set

$$
\Pi = \begin{bmatrix} 1 & 0 \cr
         0 & \sqrt{\beta} \end{bmatrix}
$$

$$
c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}
$$

and

$$
b = \begin{bmatrix} b_1 \cr \sqrt{\beta} b_2
\end{bmatrix}
$$

The  budget constraint {eq}`eq:old2` becomes

$$
p_1 c_1 + p_2 c_2 = p_1 e_1 + p_2 e_2
$$

The left side is the **discounted present value** of consumption.

The right side is the **discounted present value** of the consumer's endowment.

The relative price  $\frac{p_1}{p_2}$ has units of time $2$ goods per unit of time $1$ goods.

Consequently, $(1+r) = R \equiv \frac{p_1}{p_2}$ is the  **gross interest rate** and $r$ is the **net interest rate**.

### Risk and state-contingent claims

We study risk in the context of a **static** environment,  meaning that there is only one period.

By **risk** we mean that an outcome is not known in advance, but that it is governed by a known probability distribution.

As an example, our consumer confronts **risk** meaning in particular that

  * there are two states of nature, $1$ and $2$.

 * the consumer knows that  probability that state $1$ occurs is $\lambda$.

 * the consumer knows that the probability that state $2$ occurs is $(1-\lambda)$.

Before the outcome is realized, the the consumer's **expected utility** is

$$
-.5 [\lambda (c_1 - b_1)^2 + (1-\lambda)(c_2 - b_2)^2]
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
c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}
$$

$$
b = \begin{bmatrix} b_1 \cr b_2 \end{bmatrix}
$$

<!-- #region -->
$$
b = \begin{bmatrix} \sqrt{\lambda}b_1 \cr \sqrt{1-\lambda}b_2 \end{bmatrix}
$$

A consumer's  endowment vector is

$$
e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}
$$

A price vector is

$$
p = \begin{bmatrix} p_1 \cr p_2 \end{bmatrix}
$$

where $p_i$ is the price of one unit of consumption in state $i$.

The state-contingent goods being traded are often called **Arrow securities**.

Before the random state of the world $i$ is realized, the consumer  sells his/her state-contingent endowment bundle and purchases a state-contingent consumption bundle.

Trading such state-contingent goods  is one  way economists often model **insurance**.

## Exercises we can do

To illustrate consequences of demand and supply shifts, we have lots of parameters to shift

* distribution of endowments $e_1, e_2$
* bliss point vectors $b_1, b_2$
* probability $\lambda$

We can study how these things affect equilibrium prices and allocations.

## Economies with Endogenous Supplies of Goods

Up to now we have described a pure exchange economy in which endowments of good are exogenous, meaning that they are taken as given from outside the model.

### Supply Curve of a Competitive Firm

A competitive firm that can produce goods  takes a price vector $p$ as given and chooses a quantity $q$
to maximize total revenue minus total costs.

The firm's total  revenue equals $p^\top q$ and its total cost equals $C(q)$  where $C(q)$ is a total cost function

$$
C(q) = h ^\top q + .5 q^\top J q
$$


and  $J$ is a positive definite matrix.


So the firm's profits are

$$
p^\top q  - C(q)
$$ (eq:compprofits)



An $n\times 1$ vector of **marginal costs** is

$$
\frac{\partial C(q)}{\partial q} = h + H q
$$

where

$$
H = .5 (J + J')
$$

An $n \times 1$ vector of marginal revenues for the price-taking firm is $\frac{\partial p^\top q}
{\partial q} = p $.

So **price equals marginal revenue** for our price-taking competitive firm.

The firm maximizes total profits by setting  **marginal revenue to marginal costs**.

This leads to the following **inverse supply curve** for the competitive firm:


$$
p = h + H q
$$




### Competitive Equilibrium

#### $\mu=1$ warmup

As a special case, let's pin down a demand curve by setting the marginal utility of wealth  $\mu =1$.

Equating  supply price to demand price we get

$$
p = h + H c = \Pi^\top b - \Pi^\top \Pi c ,
$$

which implies the equilibrium quantity vector

$$
c = (\Pi^\top \Pi + H )^{-1} ( \Pi^\top b - h)
$$ (eq:old5)

This equation is the counterpart of equilibrium quantity {eq}`eq:old1` for the scalar $n=1$ model with which we began.

#### General $\mu\neq 1$ case

Now let's extend the preceding analysis to a more
general case by allowing $\mu \neq 1$.

Then the inverse demand curve is

$$
p = \mu^{-1} [\Pi^\top b - \Pi^\top \Pi c]
$$ (eq:old5pa)

Equating this to the inverse supply curve and solving
for $c$ gives

$$
c = [\Pi^\top \Pi + \mu H]^{-1} [ \Pi^\top b - \mu h]
$$ (eq:old5p)


### Digression: A  Supplier Who is a Monopolist

A competitive firm is a **price-taker** who regards the price and therefore its marginal revenue as being beyond its control.

A monopolist knows that it has no competition and can influence the price and its marginal revenue by
setting quantity.

A monopolist takes a **demand curve** and not the **price** as beyond its control.

Thus, instead of being a price-taker, a monopolist sets prices to maximize profits subject to the inverse  demand curve
{eq}`eq:old5pa`.

So the monopolist's total profits as a function of its  output $q$ is

$$
[\mu^{-1} \Pi^\top (b - \Pi q)]^\top  q - h^\top q - .5 q^\top J q
$$ (eq:monopprof)

After finding
first-order necessary conditions for maximizing monopoly profits with respect to $q$
and solving them for $q$, we find that the monopolist sets

$$
q = (H + 2 \mu^{-1} \Pi^T \Pi)^{-1} (\mu^{-1} \Pi^\top b - h)
$$ (eq:qmonop)

We'll soon  see that a monopolist sets a **lower output** $q$ than does either a

 * planner who chooses $q$ to maximize social welfare

 * a competitive equilibrium


**Exercise:** Please  verify the monopolist's supply curve {eq}`eq:qmonop`.




## Multi-good  Welfare Maximization Problem

Our welfare maximization problem -- also sometimes called a  social planning  problem  -- is to choose $c$ to maximize

$$
-.5 \mu^{-1}(\Pi c -b) ^\top (\Pi c -b )
$$

minus the area under the inverse supply curve, namely,

$$
h c + .5 c^\top J c  .
$$

So the welfare criterion is

$$
-.5 \mu^{-1}(\Pi c -b) ^\top (\Pi c -b ) -h c - .5 c^\top J c
$$

In this formulation, $\mu$ is a parameter that describes how the planner weights interests of outside suppliers and our representative consumer.

The first-order condition with respect to $c$ is

$$
- \mu^{-1} \Pi^\top \Pi c + \mu^{-1}\Pi^\top b - h -  H c = 0
$$

which implies {eq}`eq:old5p`.

Thus,  as for the single-good case, with  multiple goods   a competitive equilibrium quantity vector solves a planning problem.

(This is another version of the first welfare theorem.)

We can deduce a competitive equilibrium price vector from either

  * the inverse demand curve, or

  * the inverse supply curve

<!-- #endregion -->

<!-- #region -->

