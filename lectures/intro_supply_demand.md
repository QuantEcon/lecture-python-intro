---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction to Supply and Demand

## Overview

This lecture is about some models of equilibrium prices and quantities, one of
the main topics of elementary microeconomics.

Throughout the lecture, we focus on models with one good and one price.


In a {doc}`subsequent lecture <supply_demand_multiple_goods>` we will investigate settings with
many goods.

Key infrastructure concepts that we'll encounter in this lecture are

* inverse demand curves
* inverse supply curves
* consumer surplus
* producer surplus
* social welfare as the sum of consumer and producer surpluses
* relationship between  equilibrium quantity and social welfare optimum

Throughout the lectures, we'll assume that inverse demand and supply curves are **affine** functions of quantity.

("Affine" means "linear plus a constant" and [here](https://math.stackexchange.com/questions/275310/what-is-the-difference-between-linear-and-affine-function) is a nice discussion about it.)

We'll also assume affine inverse supply and demand functions when we study models with multiple consumption goods in our {doc}`subsequent lecture <supply_demand_multiple_goods>`.

We do this in order to simplify the exposition and enable us to use just a few tools from linear algebra, namely, matrix multiplication and matrix inversion.

In our exposition we will use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Supply and demand

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

Due to nonlinearities, the new welfare function is not easy to maximize with
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


```{solution-start} isd_ex3
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
