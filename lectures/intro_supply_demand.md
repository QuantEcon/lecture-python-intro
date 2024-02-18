---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction to Supply and Demand

## Overview

This lecture is about some models of equilibrium prices and quantities, one of
the core topics of elementary microeconomics.

Throughout the lecture, we focus on models with one good and one price.

```{seealso}
In a {doc}`subsequent lecture <supply_demand_multiple_goods>` we will investigate settings with
many goods.
```

### Why does this model matter?

In the 15th, 16th, 17th and 18th centuries, mercantilist ideas held sway among most rulers of European countries.

Exports were regarded as good because they brought in bullion (gold flowed into the country).

Imports were regarded as bad because bullion was required to pay for them (gold flowed out).

This [zero-sum](https://en.wikipedia.org/wiki/Zero-sum_game) view of economics was eventually overturned by the work of the classical economists such as [Adam Smith](https://en.wikipedia.org/wiki/Adam_Smith) and [David Ricado](https://en.wikipedia.org/wiki/David_Ricardo), who showed how freeing domestic and international trade can enhance welfare.

There are many different expressions of this idea in economics.

This lecture discusses one of the simplest: how free adjustment of prices can maximize a measure of social welfare in the market for a single good.


### Topics and infrastructure

Key infrastructure concepts that we will encounter in this lecture are:

* inverse demand curves
* inverse supply curves
* consumer surplus
* producer surplus
* integration
* social welfare as the sum of consumer and producer surpluses
* the relationship between  equilibrium quantity and social welfare optimum

In our exposition we will use the following Python imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Consumer surplus

Before we look at the model of supply and demand, it will be helpful to have some background on (a) consumer and producer surpluses and (b) integration.

(If you are comfortable with both topics you can jump to the next section.)

### A discrete example

Regarding consumer surplus, suppose that we have a single good and 10 consumers.

These 10 consumers have different preferences; in particular, the amount they would be willing to pay for one unit of the good differs.

Suppose that the willingness to pay for each of the 10 consumers is as follows:

| consumer       | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10  |
|----------------|----|----|----|----|----|----|----|----|----|-----|
| willing to pay | 98 | 72 | 41 | 38 | 29 | 21 | 17 | 12 | 11 | 10  |

(We have ordered consumers by willingness to pay, in descending order.)

If $p$ is the price of the good and  $w_i$ is the amount that consumer $i$ is willing to pay, then $i$ buys when $w_i \geq p$.

```{note}
If $p=w_i$ the consumer is indifferent between buying and not buying; we arbitrarily assume that they buy.
```

The **consumer surplus** of the $i$-th consumer is $\max\{w_i - p, 0\}$

* if $w_i \geq p$, then the consumer buys and gets surplus $w_i - p$
* if $w_i < p$, then the consumer does not buy and gets surplus $0$

For example, if the price is $p=40$, then consumer 1 gets surplus $98-40=58$.

The bar graph below shows the surplus of each consumer when $p=25$.

The total height of each bar $i$ is willingness to pay by consumer $i$.

The orange portion of some of the bars shows consumer surplus.


```{code-cell} ipython3
fig, ax = plt.subplots()
consumers = range(1, 11) # consumers 1,..., 10
# willingness to pay for each consumer
wtp = (98, 72, 41, 38, 29, 21, 17, 12, 11, 10)
price = 25
ax.bar(consumers, wtp, label="consumer surplus", color="darkorange", alpha=0.8)
ax.plot((0, 12), (price, price), lw=2, label="price $p$")
ax.bar(consumers, [min(w, price) for w in wtp], color="black", alpha=0.6)
ax.set_xlim(0, 12)
ax.set_xticks(consumers)
ax.set_ylabel("willingness to pay, price")
ax.set_xlabel("consumer, quantity")
ax.legend()
plt.show()
```

The total consumer surplus in this market is 

$$ 
\sum_{i=1}^{10} \max\{w_i - p, 0\}
= \sum_{w_i \geq p} (w_i - p)
$$

Since consumer surplus $\max\{w_i-p,0\}$ of consumer $i$ is a measure of her gains from trade (i.e., extent to which the good is valued over and above the amount the consumer had to pay), it is reasonable to consider total consumer surplus as a measurement of consumer welfare.

Later we will pursue this idea further, considering how different prices lead to different welfare outcomes for consumers and producers.

### A comment on quantity.

Notice that in the figure, the horizontal axis is labeled "consumer, quantity".

We have added "quantity" here because we can read the number of units sold from this axis, assuming for now that there are sellers who are willing to sell as many units as the consumers demand, given the current market price $p$.

In this example, consumers 1 to 5 buy, and the quantity sold is 5.

Below we drop the assumption that sellers will provide any amount at a given price and study how this changes outcomes.

### A continuous approximation

It is often convenient to assume that there is a "very large number" of consumers, so that willingness to pay becomes a continuous curve.

As before, the vertical axis measures willingness to pay, while the horizontal axis measures quantity.

This kind of curve is called an **inverse demand curve**

An example is provided below, showing both an inverse demand curve and a set price.

The inverse demand curve is given by 

$$
p = 100 e^{-q} 
$$

```{code-cell} ipython3
def inverse_demand(q):
    return 100 * np.exp(- q)

# build a grid to evaluate the function at different values of q
q_min, q_max = 0, 5
q_grid = np.linspace(q_min, q_max, 1000)

# plot the inverse demand curve
fig, ax = plt.subplots()
ax.plot((q_min, q_max), (price, price), lw=2, label="price")
ax.plot(q_grid, inverse_demand(q_grid), 
        color="k", label="inverse demand curve")
ax.set_ylabel("willingness to pay, price")
ax.set_xlabel("quantity")
ax.set_xlim(q_min, q_max)
ax.set_ylim(0, 110)
ax.legend()
plt.show()
```

Reasoning by analogy with the discrete case, the area under the demand curve and above the price is called the **consumer surplus**, and is a measure of total gains from trade on the part of consumers.

The consumer surplus is shaded in the figure below.

```{code-cell} ipython3
# solve for the value of q where demand meets price
q_star = np.log(100) - np.log(price)

fig, ax = plt.subplots()
ax.plot((q_min, q_max), (price, price), lw=2, label="price")
ax.plot(q_grid, inverse_demand(q_grid), 
        color="k", label="inverse demand curve")
small_grid = np.linspace(0, q_star, 500)
ax.fill_between(small_grid, np.full(len(small_grid), price),
                inverse_demand(small_grid), color="darkorange",
                alpha=0.6, label="consumer surplus")
ax.vlines(q_star, 0, price, ls="--")
ax.set_ylabel("willingness to pay, price")
ax.set_xlabel("quantity")
ax.set_xlim(q_min, q_max)
ax.set_ylim(0, 110)
ax.text(q_star, -10, "$q^*$")
ax.legend()
plt.show()
```

The value $q^*$ is where the inverse demand curve meets price.

## Producer surplus

Having discussed demand, let's now switch over to the supply side of the market.

### The discrete case

The figure below shows the price at which a collection of producers, also numbered 1 to 10, are willing to sell one unit of the good in question

```{code-cell} ipython3
fig, ax = plt.subplots()
producers = range(1, 11) # producers 1,..., 10
# willingness to sell for each producer
wts = (5, 8, 17, 22, 35, 39, 46, 57, 88, 91)
price = 25
ax.bar(producers, wts, label="willingness to sell", color="blue", alpha=0.8)
ax.set_xlim(0, 12)
ax.set_xticks(producers)
ax.set_ylabel("willingness to sell")
ax.set_xlabel("producer")
ax.legend()
plt.show()
```

Let $v_i$ be the price at which producer $i$ is willing to sell the good.

When the price is $p$, producer surplus for producer $i$ is $\max\{p - v_i, 0\}$.

For example, a producer willing to sell at \$10 and selling at price \$20 makes a surplus of \$10. 

Total producer surplus is given by

$$
\sum_{i=1}^{10} \max\{p - v_i, 0\}
= \sum_{p \geq v_i} (p - v_i)
$$

As for the consumer case, it can be helpful for analysis if we approximate producer willingness to sell into a continuous curve.

This curve is called the **inverse supply curve**

We show an example below where the inverse supply curve is

$$
p = 2 q^2
$$

The shaded area is the total producer surplus in this continuous model.

```{code-cell} ipython3
def inverse_supply(q):
    return 2 * q**2

# solve for the value of q where supply meets price
q_star = (price / 2)**(1/2)

# plot the inverse supply curve
fig, ax = plt.subplots()
ax.plot((q_min, q_max), (price, price), lw=2, label="price")
ax.plot(q_grid, inverse_supply(q_grid), 
        color="k", label="inverse supply curve")
small_grid = np.linspace(0, q_star, 500)
ax.fill_between(small_grid, inverse_supply(small_grid), 
                np.full(len(small_grid), price), 
                color="darkgreen",
                alpha=0.4, label="producer surplus")
ax.vlines(q_star, 0, price, ls="--")
ax.set_ylabel("willingness to sell, price")
ax.set_xlabel("quantity")
ax.set_xlim(q_min, q_max)
ax.set_ylim(0, 60)
ax.text(q_star, -10, "$q^*$")
ax.legend()
plt.show()
```

## Integration

How can we calculate the consumer and producer surplus in the continuous case?

The short answer is: by using [integration](https://en.wikipedia.org/wiki/Integral).

Some readers will already be familiar with the basics of integration.

For those who are not, here is a quick introduction.

In general, for a function $f$, the **integral** of $f$ over the interval $[a, b]$ is the area under the curve $f$ between $a$ and $b$.

This value is written as $\int_a^b f(x) dx$ and illustrated in the figure below when $f(x) = \cos(x/2) + 1$.

```{code-cell} ipython3
def f(x):
    return np.cos(x/2) + 1

xmin, xmax = 0, 5
a, b = 1, 3
x_grid = np.linspace(xmin, xmax, 1000)
ab_grid = np.linspace(a, b, 400)

fig, ax = plt.subplots()
ax.plot(x_grid, f(x_grid), label="$f$", color="k")
ax.fill_between(ab_grid, [0] * len(ab_grid), f(ab_grid), 
                alpha=0.5, label="$\int_a^b f(x) dx$")
ax.legend()
plt.show()
```

There are many rules for calculating integrals, with different rules applying to different choices of $f$.

Many of these rules relate to one of the most beautiful and powerful results in all of mathematics: the [fundamental theorem of calculus](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus).

We will not try to cover these ideas here, partly because the subject is too big, and partly because you only need to know one rule for this lecture, stated below.

If $f(x) = c + d x$, then 

$$ 
\int_a^b f(x) dx = c (b - a) + \frac{d}{2}(b^2 - a^2) 
$$

In fact this rule is so simple that it can be calculated from elementary geometry -- you might like to try by graphing $f$ and calculating the area under the curve between $a$ and $b$.

We use this rule repeatedly in what follows.

## Supply and demand

Let's now put supply and demand together.

This leads us to the all important notion of market equilibrium, and from there onto a discussion of equilibria and welfare.

For most of this discussion, we'll assume that inverse demand and supply curves are **affine** functions of quantity.

```{note}
"Affine" means "linear plus a constant" and [here](https://math.stackexchange.com/questions/275310/what-is-the-difference-between-linear-and-affine-function) is a nice discussion about it.
```

We'll also assume affine inverse supply and demand functions when we study models with multiple consumption goods in our {doc}`subsequent lecture <supply_demand_multiple_goods>`.

We do this in order to simplify the exposition and enable us to use just a few tools from linear algebra, namely, matrix multiplication and matrix inversion.

We study a market for a single good in which buyers and sellers exchange a quantity $q$ for a price $p$.

Quantity $q$ and price $p$ are  both scalars.

We assume that inverse demand and supply curves for the good are:

$$
p = d_0 - d_1 q, \quad d_0, d_1 > 0
$$

$$
p = s_0 + s_1 q , \quad s_0, s_1 > 0
$$

We call them inverse demand and supply curves because price is on the left side of the equation rather than on the right side as it would be in a direct demand or supply function.

Here is a class that stores parameters for our single good market, as well as
implementing the inverse demand and supply curves.

```{code-cell} ipython3
class Market:

    def __init__(self,
                 d_0=1.0,      # demand intercept
                 d_1=0.6,      # demand slope
                 s_0=0.1,      # supply intercept
                 s_1=0.4):     # supply slope

        self.d_0, self.d_1 = d_0, d_1
        self.s_0, self.s_1 = s_0, s_1

    def inverse_demand(self, q):
        return self.d_0 - self.d_1 * q

    def inverse_supply(self, q):
        return self.s_0 + self.s_1 * q
```

Let's create an instance.

```{code-cell} ipython3
market = Market()
```

Here is a plot of these two functions using `market`.

```{code-cell} ipython3
:tags: [hide-input]

market = Market()

grid_min, grid_max, grid_size = 0, 1.5, 200
q_grid = np.linspace(grid_min, grid_max, grid_size)
supply_curve = market.inverse_supply(q_grid)
demand_curve = market.inverse_demand(q_grid)

fig, ax = plt.subplots()
ax.plot(q_grid, supply_curve, label='supply')
ax.plot(q_grid, demand_curve, label='demand')
ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((0, 1))
ax.set_yticks((0, 1))
ax.set_xlabel('quantity')
ax.set_ylabel('price')
plt.show()
```

In the above graph, an **equilibrium** price-quantity pair occurs at the intersection of the supply and demand curves. 

### Consumer surplus

Let a quantity $q$ be given and let $p := d_0 - d_1 q$ be the
corresponding price on the inverse demand curve.

We define **consumer surplus** $S_c(q)$ as the area under an inverse demand
curve minus $p q$:

$$
S_c(q) := 
\int_0^{q} (d_0 - d_1 x) dx - p q 
$$ (eq:cstm_spls)

The next figure illustrates

```{code-cell} ipython3
:tags: [hide-input]

q = 1.25
p = market.inverse_demand(q)
ps = np.ones_like(q_grid) * p

fig, ax = plt.subplots()
ax.plot(q_grid, demand_curve, label='demand')
ax.fill_between(q_grid[q_grid <= q],
                demand_curve[q_grid <= q],
                ps[q_grid <= q],
                label='consumer surplus',
                color='#EED1CF')
ax.vlines(q, 0, p, linestyle="dashed", color='black', alpha=0.7)
ax.hlines(p, 0, q, linestyle="dashed", color='black', alpha=0.7)

ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((q,))
ax.set_xticklabels(("$q$",))
ax.set_yticks((p,))
ax.set_yticklabels(("$p$",))
ax.set_xlabel('quantity')
ax.set_ylabel('price')
plt.show()
```

Consumer surplus provides a measure of total consumer welfare at quantity $q$.

The idea is that the inverse demand curve $d_0 - d_1 q$ shows a consumer's willingness to 
pay for an additional increment of the good at a given quantity $q$.

The difference between willingness to pay and the actual price is consumer surplus.

The value $S_c(q)$ is the "sum" (i.e., integral) of these surpluses when the total
quantity purchased is $q$ and the purchase price is $p$.

Evaluating the integral in the definition of consumer surplus {eq}`eq:cstm_spls` gives

$$
S_c(q) 
= d_0 q - \frac{1}{2} d_1 q^2 - p q
$$

### Producer surplus

Let a quantity $q$ be given and let $p := s_0 + s_1 q$ be the
corresponding price on the inverse supply curve.

We define **producer surplus** as $p q$ minus the area under an inverse supply curve

$$
S_p(q) 
:= p q - \int_0^q (s_0 + s_1 x) dx 
$$ (eq:pdcr_spls)

The next figure illustrates

```{code-cell} ipython3
:tags: [hide-input]

q = 0.75
p = market.inverse_supply(q)
ps = np.ones_like(q_grid) * p

fig, ax = plt.subplots()
ax.plot(q_grid, supply_curve, label='supply')
ax.fill_between(q_grid[q_grid <= q],
                supply_curve[q_grid <= q],
                ps[q_grid <= q],
                label='producer surplus',
                color='#E6E6F5')
ax.vlines(q, 0, p, linestyle="dashed", color='black', alpha=0.7)
ax.hlines(p, 0, q, linestyle="dashed", color='black', alpha=0.7)

ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((q,))
ax.set_xticklabels(("$q$",))
ax.set_yticks((p,))
ax.set_yticklabels(("$p$",))
ax.set_xlabel('quantity')
ax.set_ylabel('price')
plt.show()
```

Producer surplus measures total producer welfare at quantity $q$ 

The idea is similar to that of consumer surplus.

The inverse supply curve $s_0 + s_1 q$ shows the price at which producers are
prepared to sell, given quantity $q$.

The difference between willingness to sell and the actual price is producer surplus.

The value $S_p(q)$ is the integral of these surpluses.

Evaluating the integral in the definition of producer surplus {eq}`eq:pdcr_spls` gives

$$
S_p(q) = pq - s_0 q -  \frac{1}{2} s_1 q^2
$$


### Social welfare

Sometimes economists measure social welfare by a **welfare criterion** that
equals consumer surplus plus producer surplus, assuming that consumers and
producers pay the same price:

$$
W(q)
= \int_0^q (d_0 - d_1 x) dx - \int_0^q (s_0 + s_1 x) dx  
$$

Evaluating the integrals gives

$$
W(q) = (d_0 - s_0) q -  \frac{1}{2} (d_1 + s_1) q^2
$$

Here is a Python function that evaluates this social welfare at a given
quantity $q$ and a fixed set of parameters.

```{code-cell} ipython3
def W(q, market):
    # Unpack
    d_0, d_1, s_0, s_1 = market.d_0, market.d_1, market.s_0, market.s_1
    # Compute and return welfare
    return (d_0 - s_0) * q - 0.5 * (d_1 + s_1) * q**2
```

The next figure plots welfare as a function of $q$.

```{code-cell} ipython3
:tags: [hide-input]

q_vals = np.linspace(0, 1.78, 200)
fig, ax = plt.subplots()
ax.plot(q_vals, W(q_vals, market), label='welfare')
ax.legend(frameon=False)
ax.set_xlabel('quantity')
plt.show()
```

Let's now give a social planner the task of maximizing social welfare.

To compute a quantity that  maximizes the welfare criterion, we differentiate
$W$ with respect to $q$ and then set the derivative to zero.

$$
\frac{d W(q)}{d q} = d_0 - s_0 - (d_1 + s_1) q  = 0
$$

Solving for $q$ yields

$$
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:old1)

Let's remember the quantity $q$ given by equation {eq}`eq:old1` that a social planner would choose to maximize consumer surplus plus producer surplus.

We'll compare it to the quantity that emerges in a competitive equilibrium that equates supply to demand.

### Competitive equilibrium

Instead of equating quantities supplied and demanded, we can accomplish the
same thing by equating demand price to supply price:

$$
p =  d_0 - d_1 q = s_0 + s_1 q 
$$

If we solve the equation defined by the second equality in the above line for
$q$, we obtain 

$$
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:equilib_q)


This is the competitive equilibrium quantity. 

Observe that the equilibrium quantity equals the same $q$ given by equation  {eq}`eq:old1`.

The outcome that the quantity determined by equation {eq}`eq:old1` equates
supply to demand brings us a **key finding:**

*  a competitive equilibrium quantity maximizes our welfare criterion

This is a version of the [first fundamental welfare theorem](https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics), 

It also brings a useful **competitive equilibrium computation strategy:**

* after solving the welfare problem for an optimal quantity, we can read a competitive equilibrium price from either supply price or demand price at the competitive equilibrium quantity

## Generalizations

In a {doc}`later lecture <supply_demand_multiple_goods>`, we'll derive
generalizations of the above demand and supply curves from other objects.

Our generalizations will extend the preceding analysis of a market for a single good to the analysis of $n$ simultaneous markets in $n$ goods.

In addition

* we'll derive  **demand curves** from a consumer problem that maximizes a
 **utility function** subject to a **budget constraint**.

* we'll derive  **supply curves** from the problem of a producer who is price
 taker and maximizes his profits minus total costs that are described by a **cost function**.

## Exercises

Suppose now that the inverse demand and supply curves are modified to take the
form

$$
p = i_d(q) := d_0 - d_1 q^{0.6} 
$$

$$
p = i_s(q) := s_0 + s_1 q^{1.8} 
$$

All parameters are positive, as before.

```{exercise}
:label: isd_ex1

Define a new `Market` class that holds the same parameter values as before by
changing the `inverse_demand` and `inverse_supply` methods to
match these new definitions.

Using the class, plot the inverse demand and supply curves $i_d$ and $i_s$

```


```{solution-start} isd_ex1
:class: dropdown
```

```{code-cell} ipython3
class Market:

    def __init__(self,
                 d_0=1.0,      # demand intercept
                 d_1=0.6,      # demand slope
                 s_0=0.1,      # supply intercept
                 s_1=0.4):     # supply slope

        self.d_0, self.d_1 = d_0, d_1
        self.s_0, self.s_1 = s_0, s_1

    def inverse_demand(self, q):
        return self.d_0 - self.d_1 * q**0.6

    def inverse_supply(self, q):
        return self.s_0 + self.s_1 * q**1.8
```

Let's create an instance.

```{code-cell} ipython3
market = Market()
```

Here is a plot of inverse supply and demand.

```{code-cell} ipython3
:tags: [hide-input]

grid_min, grid_max, grid_size = 0, 1.5, 200
q_grid = np.linspace(grid_min, grid_max, grid_size)
supply_curve = market.inverse_supply(q_grid)
demand_curve = market.inverse_demand(q_grid)

fig, ax = plt.subplots()
ax.plot(q_grid, supply_curve, label='supply')
ax.plot(q_grid, demand_curve, label='demand')
ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((0, 1))
ax.set_yticks((0, 1))
ax.set_xlabel('quantity')
ax.set_ylabel('price')
plt.show()
```

```{solution-end}
```


```{exercise}
:label: isd_ex2

As before, consumer surplus at $q$ is the area under the demand curve minus
price times quantity:

$$
S_c(q) = \int_0^{q} i_d(x) dx - p q 
$$

Here $p$ is set to $i_d(q)$

Producer surplus is price times quantity minus the area under the inverse
supply curve:

$$
S_p(q) 
= p q - \int_0^q i_s(x) dx 
$$

Here $p$ is set to $i_s(q)$.

Social welfare is the sum of consumer and producer surplus under the
assumption that the price is the same for buyers and sellers:

$$
W(q)
= \int_0^q i_d(x) dx - \int_0^q i_s(x) dx  
$$

Solve the integrals and write a function to compute this quantity numerically
at given $q$. 

Plot welfare as a function of $q$.

```


```{solution-start} isd_ex2
:class: dropdown
```

Solving the integrals gives 

$$
W(q) 
= d_0 q - \frac{d_1 q^{1.6}}{1.6}
    - \left( s_0 q + \frac{s_1 q^{2.8}}{2.8} \right)
$$

Here's a Python function that computes this value:

```{code-cell} ipython3
def W(q, market):
    # Unpack
    d_0, d_1 = market.d_0, market.d_1
    s_0, s_1 = market.s_0, market.s_1
    # Compute and return welfare
    S_c = d_0 * q - d_1 * q**1.6 / 1.6
    S_p = s_0 * q + s_1 * q**2.8 / 2.8
    return S_c - S_p
```

The next figure plots welfare as a function of $q$.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(q_vals, W(q_vals, market), label='welfare')
ax.legend(frameon=False)
ax.set_xlabel('quantity')
plt.show()
```

```{solution-end}
```


```{exercise}
:label: isd_ex3

Due to non-linearities, the new welfare function is not easy to maximize with
pencil and paper.

Maximize it using `scipy.optimize.minimize_scalar` instead.

```


```{solution-start} isd_ex3
:class: dropdown
```

```{code-cell} ipython3
from scipy.optimize import minimize_scalar

def objective(q):
    return -W(q, market)

result = minimize_scalar(objective, bounds=(0, 10))
print(result.message)
```

```{code-cell} ipython3
maximizing_q = result.x
print(f"{maximizing_q: .5f}")
```

```{solution-end}
```


```{exercise}
:label: isd_ex4

Now compute the equilibrium quantity by finding the price that equates supply
and demand.

You can do this numerically by finding the root of the excess demand function

$$
e_d(q) := i_d(q) - i_s(q) 
$$

You can use `scipy.optimize.newton` to compute the root.

Initialize `newton` with a starting guess somewhere close to 1.0.

(Similar initial conditions will give the same result.)

You should find that the equilibrium price agrees with the welfare maximizing
price, in line with the first fundamental welfare theorem.

```


```{solution-start} isd_ex4
:class: dropdown
```

```{code-cell} ipython3
from scipy.optimize import newton

def excess_demand(q):
    return market.inverse_demand(q) - market.inverse_supply(q)

equilibrium_q = newton(excess_demand, 0.99)
print(f"{equilibrium_q: .5f}")
```

```{solution-end}
```
