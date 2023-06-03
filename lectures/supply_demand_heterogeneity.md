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

# Market Equilibrium with Heterogeneity


## An Endowment Economy

Let's study a **pure exchange** economy without production.

There are two consumers who differ in their endowment vectors $e_i$ and their bliss-point vectors $b_i$ for $i=1,2$.

The total endowment is $e_1 + e_2$.

A competitive equilibrium requires that

$$
c_1 + c_2 = e_1 + e_2
$$

Assume the demand curves

$$
    c_i = (\Pi^\top \Pi )^{-1}(\Pi^\top b_i -  \mu_i p )
$$

Competitive equilibrium then requires that

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

```{exercise-start}
:label: sdh_ex1
```

Show that, up to normalization by a positive scalar, the same competitive equilibrium price vector that you computed in the preceding two-consumer economy would prevail in a single-consumer economy in which a single **representative consumer** has utility function

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

```{exercise-end}
```

## Designing some Python Code


Below we shall construct a Python class with the following attributes:

 * **Preferences** in the form of

     * an $n \times n$  positive definite matrix $\Pi$
     * an $n \times 1$ vector of bliss points $b$

 * **Endowments** in the form of

     * an $n \times 1$ vector $e$
     * a scalar "wealth" $W$ with default value $0$

 * **Production Costs** pinned down by

     * an $n \times 1$ nonnegative vector $h$
     * an $n \times n$ positive definite matrix $J$

The class will include a test to make sure that $b  > > \Pi e $ and raise an exception if it is violated
(at some threshold level we'd have to specify).

 * **A Person** in the form of a pair that consists of

    * **Preferences** and **Endowments**

 * **A Pure Exchange Economy** will consist of

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

```{code-cell} ipython3
# import some packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
```

### Pure Exchange Economy

Let's first explore a pure exchange economy with $n$ goods and $m$ people.

We'll compute a competitive equilibrium.

To compute a competitive equilibrium of a pure exchange economy, we use the fact that

- Relative prices in a competitive equilibrium are the same as those in a special single person or  representative consumer economy with preference $\Pi$ and $b=\sum_i b_i$, and endowment $e = \sum_i e_{i}$.

We can use the following steps to compute a competitive equilibrium:

- First we solve the single representative consumer economy by normalizing $\mu = 1$. Then, we renormalize the price vector by using the first consumption good as numeraire.

- Next we use the competitive equilibrium prices to compute each consumer's marginal utility of wealth:

$$
\mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}$$

- Finally we compute a competitive equilibrium allocation by using the demand curves:
  
$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p 
$$



## Deducing a Representative Consumer

In the class of multiple consumer economies that we are studying here, it turns out that there
exists a single **representative consumer** whose preferences and endowments can be deduced from lists of preferences and endowments for the separate individual consumers.

Consider a multiple consumer economy with initial distribution of wealth $W_i$ satisfying $\sum_i W_{i}=0$

We allow an initial redistribution of wealth.

We have the following objects


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
\mu = \sum_i\mu_{i}=\frac{0 + p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}.
$$

Now consider the representative consumer economy specified above.

Denote the marginal utility of wealth of the representative consumer by $\tilde{\mu}$.

The demand function is

$$
c=\Pi^{-1}b-(\Pi^{\top}\Pi)^{-1}\tilde{\mu} p
$$

Substituting this into the budget constraint gives

$$
\tilde{\mu}=\frac{p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$

In an equilibrium $c=e$, so

$$
p=\tilde{\mu}^{-1}(\Pi^{\top}b-\Pi^{\top}\Pi e)
$$

Thus, we have  verified that, up to choice of a numeraire in which to express absolute prices,  the price vector in our representative consumer economy is the same as that in an underlying  economy with multiple consumers.

```{code-cell} ipython3
class ExchangeEconomy:
    def __init__(self, 
                 Pi, 
                 bs, 
                 es, 
                 Ws=None, 
                 thres=4):
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

        if Ws == None:
            Ws = np.zeros(m)
        else:
            if sum(Ws) != 0:
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
        p = p / p[0]

        # compute marg util of wealth
        mu_s = []
        c_s = []
        A = p.T @ slope_dc @ p

        for i in range(m):
            mu_i = (-Ws[i] + p.T @ (Pi_inv @ bs[i] - es[i])) / A
            c_i = Pi_inv @ bs[i] - mu_i * slope_dc @ p
            mu_s.append(mu_i)
            c_s.append(c_i)

        for c_i in c_s:
            if any(c_i < 0):
                print('allocation: ', c_s)
                raise Exception('negative allocation: equilibrium does not exist')

        return p, c_s, mu_s
```

### Example: Two-Person Economy **without** Production
  * Study how competitive equilibrium $p, c^1, c^2$ respond to  different

     * $b^i$'s
     * $e^i$'s

```{code-cell} ipython3
Pi = np.array([[1, 0],
               [0, 1]])

bs = [np.array([5, 5]),  # first consumer's bliss points
      np.array([5, 5])]  # second consumer's bliss points

es = [np.array([0, 2]),  # first consumer's endowment
      np.array([2, 0])]  # second consumer's endowment

EE = ExchangeEconomy(Pi, bs, es)
p, c_s, mu_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

What happens if the first consumer likes the first good more and the second consumer likes the second good more?

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # first consumer's bliss points
         np.array([5, 6])]  # second consumer's bliss points

p, c_s, mu_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

Let the first consumer be poorer.

```{code-cell} ipython3
EE.es = [np.array([0.5, 0.5]),  # first consumer's endowment
         np.array([1, 1])]  # second consumer's endowment

p, c_s, mu_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

Now let's construct an autarky (i.e., no-trade) equilibrium.

```{code-cell} ipython3
EE.bs = [np.array([4, 6]),  # first consumer's bliss points
      np.array([6, 4])]  # second consumer's bliss points

EE.es = [np.array([0, 2]),  # first consumer's endowment
      np.array([2, 0])]  # second consumer's endowment

p, c_s, mu_s = EE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

Now let's redistribute endowments before trade.

```{code-cell} ipython3
bs = [np.array([5, 5]),  # first consumer's bliss points
      np.array([5, 5])]  # second consumer's bliss points

es = [np.array([1, 1]),  # first consumer's endowment
      np.array([1, 1])]  # second consumer's endowment

Ws = [0.5, -0.5]
EE_new = ExchangeEconomy(Pi, bs, es, Ws)
p, c_s, mu_s = EE_new.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

### A **Dynamic Economy**

Now let's use the tricks described above to study a dynamic economy, one with two periods.

```{code-cell} ipython3
beta = 0.95

Pi = np.array([[1, 0],
               [0, np.sqrt(beta)]])

bs = [np.array([5, np.sqrt(beta) * 5])]

es = [np.array([1, 1])]

EE_DE = ExchangeEconomy(Pi, bs, es)
p, c_s, mu_s = EE_DE.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```

### Example: **Arrow Securities**

We use the tricks described above to interpret  $c_1, c_2$ as "Arrow securities" that are state-contingent claims to consumption goods.

```{code-cell} ipython3
prob = 0.7

Pi = np.array([[np.sqrt(prob), 0],
               [0, np.sqrt(1 - prob)]])

bs = [np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5]),
      np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])]

es = [np.array([1, 0]),
      np.array([0, 1])]

EE_AS = ExchangeEconomy(Pi, bs, es)
p, c_s, mu_s = EE_AS.competitive_equilibrium()

print('Competitive equilibrium price vector:', p)
print('Competitive equilibrium allocation:', c_s)
```
