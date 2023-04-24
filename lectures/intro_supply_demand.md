---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction to Supply and Demand

## Outline

This lecture is about some models of equilibrium prices and quantities, one of
the main topics of elementary microeconomics.

Throughout the lecture, we focus on models with one good and one price.

({doc}`Later <supply_demand_multiple_goods>` we will investigate settings with
many goods.)

We shall describe two classic welfare theorems:

* **first welfare theorem:** for a given distribution of wealth among consumers, a competitive equilibrium allocation of goods solves a social planning problem.

* **second welfare theorem:** An allocation of goods to consumers that solves a social planning problem can be supported by a competitive equilibrium with an appropriate initial distribution of  wealth.

Key infrastructure concepts that we'll encounter in this lecture are

* inverse demand curves
* marginal utilities of wealth
* inverse supply curves
* consumer surplus
* producer surplus
* social welfare as a sum of consumer and producer surpluses
* competitive equilibrium

We will use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Supply and Demand

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

### Surpluses and Welfare

We define **consumer surplus** as the area under an inverse demand curve minus $p q$:

$$
\int_0^q (d_0 - d_1 x) dx - pq = d_0 q -.5 d_1 q^2 - pq
$$

We define **producer surplus** as $p q$ minus the area under an inverse supply curve:

$$
p q - \int_0^q (s_0 + s_1 x) dx = pq - s_0 q - .5 s_1 q^2
$$

Sometimes economists measure social welfare by a **welfare criterion** that equals consumer surplus plus producer surplus

$$
\int_0^q (d_0 - d_1 x) dx - \int_0^q (s_0 + s_1 x) dx  \equiv \textrm{Welf}
$$

or

$$
\textrm{Welf} = (d_0 - s_0) q - .5 (d_1 + s_1) q^2
$$

To compute a quantity that maximizes welfare criterion $\textrm{Welf}$, we differentiate $\textrm{Welf}$ with respect to $q$ and then set the derivative to zero.

We get

$$
\frac{d \textrm{Welf}}{d q} = d_0 - s_0 - (d_1 + s_1) q  = 0
$$

which implies

$$
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:old1)

Let's remember the quantity $q$ given by equation {eq}`eq:old1` that a social planner would choose to maximize consumer surplus plus producer surplus.

We'll compare it to the quantity that emerges in a competitive equilibrium that equates supply to demand.

### Competitive Equilibrium

Instead of equating quantities supplied and demanded, we can accomplish the same thing by equating demand price to supply price:

$$
p =  d_0 - d_1 q = s_0 + s_1 q ,
$$

+++

If we solve the equation defined by the second equality in the above line for $q$, we obtain the
competitive equilibrium quantity, which equals the same $q$ given by equation  {eq}`eq:old1`.

The outcome that the quantity determined by equation {eq}`eq:old1` equates
supply to demand brings us a **key finding:**

*  a competitive equilibrium quantity maximizes our  welfare criterion

It also brings a useful **competitive equilibrium computation strategy:**

* after solving the welfare problem for an optimal  quantity, we can read a competitive equilibrium price from either supply price or demand price at the competitive equilibrium quantity

### Generalizations

In later lectures, we'll derive generalizations of the above demand and supply curves from other objects.

Our generalizations will extend the preceding analysis of a market for a single good to the analysis of $n$ simultaneous markets in $n$ goods.

In addition

 * we'll derive **demand curves** from a consumer problem that maximizes a **utility function** subject to a **budget constraint**.

 * we'll derive **supply curves** from the problem of a producer who is price taker and maximizes his profits minus total costs that are described by a  **cost function**.

## Code

```{code-cell} ipython3
class SingleGoodMarket:

    def __init__(self,
                 d_0=1.0,  # demand intercept
                 d_1=0.5,  # demand slope
                 s_0=0.1,  # supply intercept
                 s_1=0.4):  # supply slope

        self.d_0, self.d_1 = d_0, d_1
        self.s_0, self.s_1 = s_0, s_1

    def inverse_demand(self, q):
        return self.d_0 - self.d_1 * q

    def inverse_supply(self, q):
        return self.s_0 + self.s_1 * q

    def equilibrium_quantity(self):
        return (self.d_0 - self.s_0) / (self.d_1 + self.s_1)

    def equilibrium_price(self):
        q = self.equilibrium_quantity()
        return self.s_0 + self.s_1 * q
```

```{code-cell} ipython3
def plot_supply_demand(market):
    # Unpack
    d_0, d_1 = market.d_0, market.d_1
    s_0, s_1 = market.s_0, market.s_1
    q = market.equilibrium_quantity()
    p = market.equilibrium_price()
    grid_size = 200
    x_grid = np.linspace(0, 2 * q, grid_size)
    ps = np.ones_like(x_grid) * p
    supply_curve = market.inverse_supply(x_grid)
    demand_curve = market.inverse_demand(x_grid)

    fig, ax = plt.subplots()

    ax.plot(x_grid, supply_curve, label='Supply', color='#020060')
    ax.plot(x_grid, demand_curve, label='Demand', color='#600001')

    ax.fill_between(x_grid[x_grid <= q],
                    demand_curve[x_grid <= q],
                    ps[x_grid <= q],
                    label='Consumer surplus',
                    color='#EED1CF')
    ax.fill_between(x_grid[x_grid <= q],
                    supply_curve[x_grid <= q],
                    ps[x_grid <= q],
                    label='Producer surplus',
                    color='#E6E6F5')

    ax.vlines(q, 0, p, linestyle="dashed", color='black', alpha=0.7)
    ax.hlines(p, 0, q, linestyle="dashed", color='black', alpha=0.7)

    ax.legend(loc='upper center', frameon=False)
    ax.margins(x=0, y=0)
    ax.set_ylim(0)
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')

    plt.show()
```

```{code-cell} ipython3
market = SingleGoodMarket()
```

```{code-cell} ipython3
plot_supply_demand(market)
```

