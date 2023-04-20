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

# Market Equilibrium with Heterogeneity


## An Endowment Economy

Let's study a **pure exchange** economy without production.

There are two consumers who differ in their endowment vectors $e_i$ and their bliss-point vectors $b_i$ for $i=1,2$.

The total endowment is $e_1 + e_2$.

A competitive equilibrium  requires that

$$
c_1 + c_2 = e_1 + e_2
$$

Assume the demand curves

$$
    c_i = (\Pi^\top \Pi )^{-1}(\Pi^\top b_i -  \mu_i p )
$$

Competitive equilibrium  then requires that

$$
e_1 + e_2 =
    (\Pi^\top \Pi)^{-1}(\Pi^\top (b_1 + b_2) - (\mu_1 + \mu_2) p )
$$

which after a line or two of linear algebra implies that

$$
(\mu_1 + \mu_2) p = \Pi^\top(b_1+ b_2) - \Pi^\top \Pi (e_1 + e_2)
$$ (eq:old6)

We can normalize prices by setting $\mu_1 + \mu_2 =1$ and then solving

$$
\mu_i(p,e) = \frac{p^\top (\Pi^{-1} b_i - e_i)}{p^\top (\Pi^\top \Pi )^{-1} p}
$$ (eq:old7)

for $\mu_i, i = 1,2$.

**Exercise:** Show that, up to normalization by a positive scalar,  the same competitive equilibrium price vector that you computed in the preceding two-consumer economy would prevail in a single-consumer economy in which a single **representative consumer** has utility function

$$
-.5 (\Pi c -b) ^\top (\Pi c -b )
$$

and endowment vector $e$,  where

$$
b = b_1 + b_2
$$

and

$$
e = e_1 + e_2 .
$$

## Designing some Python code


Below we shall construct a Python class with the following attributes:

 * **Preferences** in the form of

     * an $n \times n$  positive definite matrix $\Pi$
     * an $n \times 1$ vector of bliss points $b$

 * **Endowments** in the form of

     * an $n \times 1$ vector $e$
     * a scalar "wealth" $W$ with default value $0$

 * **Production Costs**  pinned down  by

     * an $n \times 1$ nonnegative vector $h$
     * an $n \times n$ positive definite matrix $J$

The class will include  a test to make sure that $b  > > \Pi e $ and raise an exception if it is violated
(at some threshold level we'd have to specify).

 * **A Person** in the form of a pair that consists of

    * **Preferences** and **Endowments**

 * **A Pure Exchange Economy** will  consist of

    * a collection of $m$ **persons**

       * $m=1$ for our single-agent economy
       * $m=2$ for our illustrations of a pure exchange economy

    * an equilibrium price vector $p$ (normalized somehow)
    * an equilibrium allocation $c^1, c^2, \ldots, c^m$ -- a collection of $m$ vectors of dimension $n \times 1$

 * **A Production Economy** will consist of

    * a single **person** that we'll interpret as a representative consumer
    * a single set of **production costs**
    * a multiplier $\mu$ that weights "consumers" versus "producers" in a planner's welfare function, as described above in the main text
    * an $n \times 1$ vector $p$ of competitive equilibrium prices
    * an $n \times 1$ vector $c$ of competitive equilibrium quantities
    * **consumer surplus**
    * **producer surplus**

Now let's proceed to code.
<!-- #endregion -->

```{code-cell} ipython3
# import some packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
```

<!-- #region -->
### Pure exchange economy

Let's first explore a pure exchange economy with $n$ goods and $m$ people.

We'll  compute a competitive equilibrium.

To compute a competitive equilibrium of a pure exchange economy, we use the fact that

- Relative prices in a competitive equilibrium are the same as those in a special single person or  representative consumer economy with preference $\Pi$ and $b=\sum_i b_i$, and endowment $e = \sum_i e_{i}$.

We can use the following steps to compute a competitive equilibrium:

- First, we solve the single representative consumer economy by normalizing $\mu = 1$. Then, we renormalize the price vector by using the first consumption good as numeraire.

- Next, we use the competitive equilibrium prices to compute each consumer's marginal utility of wealth:
$$ \mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}.$$

- Finally, we compute a competitive equilibrium allocation by  using the demand curves:
$$ c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p. $$



## Deducing a representative consumer

In the class of multiple consumer economies that we are studying here, it turns out that there
exists a single **representative consumer** whose preferences and endowments can be deduced from lists of preferences and endowments for the separate individual consumers.

Consider a multiple consumer economy with initial distribution of wealth $W_i$ satisfying $\sum_i W_{i}=0$

We allow an initial  redistribution of wealth.

We have the following objects


- The demand curve:
$$ c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p $$

- The marginal utility of wealth:
$$ \mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}$$

- Market clearing:
$$ \sum c_{i}=\sum e_{i}$$

Denote aggregate consumption $\sum_i c_{i}=c$ and $\sum_i \mu_i = \mu$.

Market  clearing requires

$$ \Pi^{-1}\left(\sum_{i}b_{i}\right)-(\Pi^{\top}\Pi)^{-1}p\left(\sum_{i}\mu_{i}\right)=\sum_{i}e_{i}$$
which, after a few steps, leads to
$$p=\mu^{-1}\left(\Pi^{\top}b-\Pi^{\top}\Pi e\right)$$

where
$$ \mu = \sum_i\mu_{i}=\frac{0 + p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}.
$$

Now consider the representative consumer economy specified above.

Denote the marginal utility of wealth of the representative consumer by $\tilde{\mu}$.

The demand function is

$$c=\Pi^{-1}b-(\Pi^{\top}\Pi)^{-1}\tilde{\mu} p.$$

Substituting this into the budget constraint gives
$$\tilde{\mu}=\frac{p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}.$$

In an equilibrium $c=e$, so
$$p=\tilde{\mu}^{-1}(\Pi^{\top}b-\Pi^{\top}\Pi e).$$

Thus, we have  verified that,  up to choice of a numeraire in which to express absolute prices,  the price vector in our representative consumer economy is the same as that in an underlying  economy with multiple consumers.
<!-- #endregion -->

```{code-cell} ipython3
class Exchange_economy:
    def __init__(self, Pi, bs, es, Ws=None, thres=4):
        """
        Set up the environment for an exchange economy

        Args:
            Pis (np.array): shared matrix of substitution
            bs (list): all consumers' bliss points
            es (list): all consumers' endowments
            Ws (list): all consumers' wealth
        """
        n, m = Pi.shape[0], len(bs)

        # check non-satiation
        for b, e in zip(bs, es):
            if np.min(b / np.max(Pi @ e)) <= 1.5:
                raise Exception('set bliss points further away')

        if Ws==None:
            Ws = np.zeros(m)
        else:
            if sum(Ws)!=0:
                raise Exception('invalid wealth distribution')

        self.Pi, self.bs, self.es, self.Ws, self.n, self.m = Pi, bs, es, Ws, n, m

    def competitive_equilibrium(self):
        """
        Compute the competitive equilibrium prices and allocation
        """
        Pi, bs, es, Ws = self.Pi, self.bs, self.es, self.Ws
        n, m = self.n, self.m
        slope_dc = inv(Pi.T @ Pi)
        Pi_inv = inv(Pi)

        # aggregate
        b = sum(bs)
        e = sum(es)

        # compute price vector with mu=1 and renormalize
        p = Pi.T @ b - Pi.T @ Pi @ e
        p = p/p[0]

        # compute marg util of wealth
        mu_s = []
        c_s = []
        A = p.T @ slope_dc @ p

        for i in range(m):
            mu_i = (-Ws[i] + p.T @ (Pi_inv @ bs[i] - es[i]))/A
            c_i = Pi_inv @ bs[i] - mu_i*slope_dc @ p
            mu_s.append(mu_i)
            c_s.append(c_i)

        for c_i in c_s:
            if any(c_i < 0):
                print('allocation: ', c_s)
                raise Exception('negative allocation: equilibrium does not exist')

        return p, c_s, mu_s
```

### Example: Two-person economy **without** production
  * Study how competitive equilibrium $p, c^1, c^2$ respond to  different

     * $b^i$'s
     * $e^i$'s



```{code-cell} ipython3
Pi = np.array([[1, 0],
               [0, 1]])

bs = [np.array([5, 5]),   # first consumer's bliss points
      np.array([5, 5])]   # second consumer's bliss points

es = [np.array([0, 2]),     # first consumer's endowment
      np.array([2, 0])]     # second consumer's endowment

example = Exchange_economy(Pi, bs, es)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

What happens if the first consumer likes the first good more and the second consumer likes the second good more?

```{code-cell} ipython3
bs = [np.array([6, 5]),   # first consumer's bliss points
      np.array([5, 6])]   # second consumer's bliss points

es = [np.array([0, 2]),     # first consumer's endowment
      np.array([2, 0])]     # second consumer's endowment


example = Exchange_economy(Pi, bs, es)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

Let the first consumer be poorer.

```{code-cell} ipython3
bs = [np.array([5, 5]),   # first consumer's bliss points
      np.array([5, 5])]   # second consumer's bliss points

es = [np.array([0.5, 0.5]),     # first consumer's endowment
      np.array([1, 1])]     # second consumer's endowment


example = Exchange_economy(Pi, bs, es)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

Now let's construct an autarky (i.e, no-trade) equilibrium.

```{code-cell} ipython3
bs = [np.array([4, 6]),   # first consumer's bliss points
      np.array([6, 4])]   # second consumer's bliss points

es = [np.array([0, 2]),     # first consumer's endowment
      np.array([2, 0])]     # second consumer's endowment


example = Exchange_economy(Pi, bs, es)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

Now let's  redistribute endowments  before trade.

```{code-cell} ipython3
bs = [np.array([5, 5]),   # first consumer's bliss points
      np.array([5, 5])]   # second consumer's bliss points

es = [np.array([1, 1]),     # first consumer's endowment
      np.array([1, 1])]     # second consumer's endowment

Ws = [0.5, -0.5]
example = Exchange_economy(Pi, bs, es, Ws)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

### A **dynamic economy**

Now let's use the tricks described above to study a dynamic economy, one with two periods.


```{code-cell} ipython3
beta = 0.95

Pi = np.array([[1, 0],
               [0, np.sqrt(beta)]])

bs = [np.array([5, np.sqrt(beta)*5])]

es = [np.array([1,1])]

example = Exchange_economy(Pi, bs, es)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)

```

### Example:  **Arrow securities**

We use the tricks described above to interpret  $c_1, c_2$ as "Arrow securities" that are state-contingent claims to consumption goods.



```{code-cell} ipython3
prob = 0.7

Pi = np.array([[np.sqrt(prob), 0],
               [0, np.sqrt(1-prob)]])

bs = [np.array([np.sqrt(prob)*5, np.sqrt(1-prob)*5]),
      np.array([np.sqrt(prob)*5, np.sqrt(1-prob)*5])]

es = [np.array([1, 0]),
      np.array([0, 1])]

example = Exchange_economy(Pi, bs, es)
p, c_s, mu_s = example.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)

```

### Production Economy

To compute a competitive equilibrium for a production economy where demand curve is pinned down by the marginal utility of wealth $\mu$, we first compute an allocation by solving a planning problem.

Then we compute the equilibrium  price vector using the inverse demand or supply curve.


```{code-cell} ipython3

class Production_economy:
    def __init__(self, Pi, b, h, J, mu):
        """
        Set up the environment for a production economy

        Args:
            Pi (np.ndarray): matrix of substitution
            b (np.array): bliss points
            h (np.array): h in cost func
            J (np.ndarray): J in cost func
            mu (float): welfare weight of the corresponding planning problem
        """
        self.n = len(b)
        self.Pi, self.b, self.h, self.J, self.mu = Pi, b, h, J, mu

    def competitive_equilibrium(self):
        """
        Compute a competitive equilibrium of the production economy
        """
        Pi, b, h, mu, J = self.Pi, self.b, self.h, self.mu, self.J
        H = .5*(J+J.T)

        # allocation
        c = inv(Pi.T@Pi + mu*H) @ (Pi.T@b - mu*h)

        # price
        p = 1/mu * (Pi.T@b - Pi.T@Pi@c)

        # check non-satiation
        if any(Pi @ c - b >= 0):
            raise Exception('invalid result: set bliss points further away')

        return c, p

    def equilibrium_with_monopoly(self):
        """
        Compute the equilibrium price and allocation when there is a monopolist supplier
        """
        Pi, b, h, mu, J = self.Pi, self.b, self.h, self.mu, self.J
        H = .5*(J+J.T)

        # allocation
        q = inv(mu*H + 2*Pi.T@Pi)@(Pi.T@b - mu*h)

        # price
        p = 1/mu * (Pi.T@b - Pi.T@Pi@q)

        if any(Pi @ q - b >= 0):
            raise Exception('invalid result: set bliss points further away')

        return q, p

    def compute_surplus(self):
        """
        Compute consumer and producer surplus for single good case
        """
        if self.n!=1:
            raise Exception('not single good')
        h, J, Pi, b, mu = self.h.item(), self.J.item(), self.Pi.item(), self.b.item(), self.mu
        H = J

        # supply/demand curve coefficients
        s0, s1 = h, H
        d0, d1 = 1/mu * Pi * b, 1/mu * Pi**2

        # competitive equilibrium
        c, p = self.competitive_equilibrium()

        # calculate surplus
        c_surplus = d0*c - .5*d1*c**2 - p*c
        p_surplus = p*c - s0*c - .5*s1*c**2

        return c_surplus, p_surplus


def plot_competitive_equilibrium(PE):
    """
    Plot demand and supply curves, producer/consumer surpluses, and equilibrium for
    a single good production economy

    Args:
        PE (class): A initialized production economy class
    """
    # get singleton value
    J, h, Pi, b, mu = PE.J.item(), PE.h.item(), PE.Pi.item(), PE.b.item(), PE.mu
    H = J

    # compute competitive equilibrium
    c, p = PE.competitive_equilibrium()
    c, p = c.item(), p.item()

    # inverse supply/demand curve
    supply_inv = lambda x: h + H*x
    demand_inv = lambda x: 1/mu*(Pi*b - Pi*Pi*x)

    xs = np.linspace(0, 2*c, 100)
    ps = np.ones(100) * p
    supply_curve = supply_inv(xs)
    demand_curve =  demand_inv(xs)

    # plot
    plt.figure()
    plt.plot(xs, supply_curve, label='Supply', color='#020060')
    plt.plot(xs, demand_curve, label='Demand', color='#600001')

    plt.fill_between(xs[xs<=c], demand_curve[xs<=c], ps[xs<=c], label='Consumer surplus', color='#EED1CF')
    plt.fill_between(xs[xs<=c], supply_curve[xs<=c], ps[xs<=c], label='Producer surplus', color='#E6E6F5')

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

#### Example: single agent with one good and  with production

Now let's construct an example of a production economy with one good.

To do this we

  * specify a single **person** and a **cost curve** in a way that let's us replicate the simple
    single-good supply demand example with which we started
  * compute equilibrium $p$ and $c$ and consumer and producer surpluses

  * draw graphs of both surpluses

  * do experiments in which we shift $b$ and watch what happens to $p, c$.

```{code-cell} ipython3
Pi  = np.array([[1]])        # the matrix now is a singleton
b   = np.array([10])
h   = np.array([0.5])
J   = np.array([[1]])
mu = 1

PE = Production_economy(Pi, b, h, J, mu)
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

Let's give consumers a lower welfare weight by raising $\mu$.

```{code-cell} ipython3
PE.mu = 2
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

Now we change  the bliss point  so that the consumer derives more utility from consumption.

```{code-cell} ipython3
PE.mu = 1
PE.b = PE.b * 1.5
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p.item())
print('Competitive equilibrium allocation:', c.item())

# plot
plot_competitive_equilibrium(PE)
```

This raises both the equilibrium price and quantity.


#### Example: single agent two-good economy **with** production

  * we'll do some experiments like those above
  * we can do experiments with a  **diagonal** $\Pi$ and also with a **non-diagonal** $\Pi$ matrices to study  how cross-slopes affect responses of $p$ and $c$ to various shifts in $b$


```{code-cell} ipython3
Pi  = np.array([[1, 0],
                [0, 1]])
b   = np.array([10, 10])

h   = np.array([0.5, 0.5])
J   = np.array([[1, 0.5],
                [0.5, 1]])
mu = 1

PE = Production_economy(Pi, b, h, J, mu)
c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)
```

```{code-cell} ipython3
PE.b   = np.array([12, 10])

c, p = PE.competitive_equilibrium()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)
```

```{code-cell} ipython3
Pi  = np.array([[1, 0.5],
                [0.5, 1]])
b   = np.array([10, 10])

h   = np.array([0.5, 0.5])
J   = np.array([[1, 0.5],
                [0.5, 1]])
mu = 1

PE = Production_economy(Pi, b, h, J, mu)
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

### A Monopolist

Let's  consider a monopolist  supplier.

We have included a method in  our `production_economy` class to compute an equilibrium price and allocation when the supplier is  a monopolist.

Since the supplier now has the price-setting power

- we first compute the optimal quantity that solves the monopolist's profit maximization problem.
- Then we back out  an equilibrium  price from the consumer's inverse demand curve.

Next, we use a graph for the single good case to illustrate the difference between a competitive equilibrium and an equilibrium with a monopolist supplier.

Recall that in a competitive equilibrium, a price-taking supplier equates  marginal revenue $p$ to marginal cost $h + Hq$.

This yields a competitive producer's  inverse supply curve.

A monopolist's marginal revenue is not constant but instead  is a non-trivial function of the quantity it sets.

The monopolist's marginal revenue is

$$
MR(q) = -2\mu^{-1}\Pi^{\top}\Pi q+\mu^{-1}\Pi^{\top}b,
$$

which the monopolist equates to its  marginal cost.

The plot indicates that the monopolist's sets output  lower than either the competitive equilibrium quantity.

In a single good case, this equilibrium is associated with a higher price of the good.

```{code-cell} ipython3
def plot_monopoly(PE):
    """
    Plot demand curve, marginal production cost and revenue, surpluses and the
    equilibrium in a monopolist supplier economy with a single good

    Args:
        PE (class): A initialized production economy class
    """
    # get singleton value
    J, h, Pi, b, mu = PE.J.item(), PE.h.item(), PE.Pi.item(), PE.b.item(), PE.mu
    H = J

    # compute competitive equilibrium
    c, p = PE.competitive_equilibrium()
    q, pm = PE.equilibrium_with_monopoly()
    c, p, q, pm = c.item(), p.item(), q.item(), pm.item()

    # compute

    # inverse supply/demand curve
    marg_cost = lambda x: h + H*x
    marg_rev  = lambda x: -2*1/mu*Pi*Pi*x + 1/mu*Pi*b
    demand_inv = lambda x: 1/mu*(Pi*b - Pi*Pi*x)

    xs = np.linspace(0, 2*c, 100)
    pms = np.ones(100) * pm
    marg_cost_curve = marg_cost(xs)
    marg_rev_curve  = marg_rev(xs)
    demand_curve    = demand_inv(xs)

    # plot
    plt.figure()
    plt.plot(xs, marg_cost_curve, label='Marginal cost', color='#020060')
    plt.plot(xs, marg_rev_curve, label='Marginal revenue', color='#E55B13')
    plt.plot(xs, demand_curve, label='Demand', color='#600001')

    plt.fill_between(xs[xs<=q], demand_curve[xs<=q], pms[xs<=q], label='Consumer surplus', color='#EED1CF')
    plt.fill_between(xs[xs<=q], marg_cost_curve[xs<=q], pms[xs<=q], label='Producer surplus', color='#E6E6F5')

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

#### A multiple good example

Let's study compare competitive equilibrium and monopoly outcomes in a multiple goods economy.

```{code-cell} ipython3
Pi  = np.array([[1, 0],
                [0, 1.2]])
b   = np.array([10, 10])

h   = np.array([0.5, 0.5])
J   = np.array([[1, 0.5],
                [0.5, 1]])
mu = 1

PE = Production_economy(Pi, b, h, J, mu)
c, p = PE.competitive_equilibrium()
q, pm = PE.equilibrium_with_monopoly()

print('Competitive equilibrium price:', p)
print('Competitive equilibrium allocation:', c)

print('Equilibrium with monopolist supplier price:', pm)
print('Equilibrium with monopolist supplier allocation:', q)
```

#### A single-good example

```{code-cell} ipython3
Pi  = np.array([[1]])        # the matrix now is a singleton
b   = np.array([10])
h   = np.array([0.5])
J   = np.array([[1]])
mu = 1

PE = Production_economy(Pi, b, h, J, mu)
c, p = PE.competitive_equilibrium()
q, pm = PE.equilibrium_with_monopoly()

print('Competitive equilibrium price:', p.item())
print('Competitive equilibrium allocation:', c.item())

print('Equilibrium with monopolist supplier price:', pm.item())
print('Equilibrium with monopolist supplier allocation:', q.item())

# plot
plot_monopoly(PE)
```
