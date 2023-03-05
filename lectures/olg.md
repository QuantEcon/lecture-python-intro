---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# The Overlapping Generations Model

In this lecture we study the overlapping generations (OLG) model.

The dynamics of this model are quite similar to Solow-Swan growth model.

At the same time, the OLG model adds an important new feature: the choice of
how much to save is endogenous.

To see why this is important, suppose, for example, that we are interested in
predicting the effect of a new tax on long-run growth.

We could add a tax to the Solow-Swan model and look at the change in the
steady state.

But this ignores something important: households will change their behavior
when they face the new tax rate.

Some might decide to save less, and some might decide to save more.

Such changes can substantially alter the predictions of the model.

Hence, if we care about accurate predictions, we should model the decision
problems of the agents.

In particular, households in the model should decide how much to save and how
much to consume, given the environment that they face (technology, taxes,
prices, etc.)

The OLG model takes up this challenge.

We will present a simple version of the OLG model that clarifies the decision
problem of households and studies the implications for long run growth.

Let's start with some imports.

```{code-cell} ipython3
import numpy as np
from scipy import optimize
from collections import namedtuple
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
```


## The Model

We assume that

- time is discrete, so that $t=0, 1, \ldots$ and
- individuals born at time $t$ live for two periods: $t$ and $t + 1$.


### Preferences

Suppose that the utility functions take the familiar constant relative risk
aversion (CRRA) form,

```{math}
:label: eq_crra

    U_t = u(c^1_t) + \beta u(c^2_{t+1})
```


Here

- $u(c) = \frac{c^{1-\gamma}-1}{1-\gamma}$
- $\gamma$ is a parameter and $\beta \in (0, 1)$ is the discount factor
- $c^1_t$ is time $t$ consumption of the individual born at time $t$
- $c^2_{t+1}$ is time $t+1$ consumption of the same individual (born at time $t$)

### Production

For each integer $t \geq 0$, output $Y_t$ in period $t$ is given by

$$
    Y_t = F(K_t, L_t) = K_t^{\alpha} L_t^{1-\alpha}
$$

Here $K_t$ is capital, $L_t$ is labor, $F$ is **Cobb-Douglas production function**, and  $\alpha$ is the output elasticity of capital in $F$.

Without population growth, $L_t$ equals some constant $L$.

### Prices

Setting $k_t := K_t / L_t$, $f(k)=F(K, 1)$ and using homogeneity of degree one now yields:

```{math}
:label: R_func
    1 + r_t = R_t = f'(k_t) = \alpha k^{\alpha-1}_t
```

Here, the gross rate of return to saving $1 + r_t$ is equal to the rental rate of capital $R_t$.

The wage rate is given by

```{math}
:label: w_func
    w_t = f(k_t) - k_t f'(k_t) = (1-\alpha) k^{\alpha}_t
```



### Equilibrium

Savings by an individual of generation $t$, $s_t$, is determined as a
solution to:

$$
    \begin{aligned}
    \max_{c^1_t, c^2_{t+1}, s_t} \ & u(c^1_t) + \beta u(c^2_{t+1}) \\
    \mbox{subject to } \ & c^1_t + s_t \le w_t \\
                         & c^2_{t+1} \le R_{t+1}s_t\\
    \end{aligned}
$$

The second constraint incorporates notion that individuals only spend
money on their own end of life consumption. Also, Since $u(.)$ is strictly increasing, both constraints will hold as equalities.


Substituting $s_t$ we get from the first constraint into the second constraint we get $c^2_{t+1}$ in terms of $c^1_t$, i.e.,

$$
    c^2_{t+1} = R_{t+1}(w_t - c^1_t)
$$
Thus first-order condition for a maximum can be written in the
familiar form of the consumption Euler equation.
Plugging $c^2_{t+1}$ into the objective function and taking derivative with respect to $c^1_t$ yield the Euler equation,

$$
    (c^1_t)^{-\gamma} = \beta R^{1-\gamma}_{t+1}  (w_t - c^1_t)^{-\gamma}
$$

Solving for consumption and thus for savings,

$$
    s_t = s(w_t, R_{t+1}) = w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
$$

Total savings in the economy will be equal to:

$$
    S_t = s_t L_t
$$



### Dynamics

We assume a closed economy, so domestic investment equals aggregate domestic
saving. Therefore, we have

$$
    K_{t+1} = L_t s(w_t, R_{t+1})
$$

Setting $k_t := K_t / L_t$, where $L_{t+1} = (1 + n) L_t,$ and using homogeneity of degree one now yields:

```{math}
:label: k_dyms_crra
    k_{t+1} = \frac{s(w_t, R_{t+1})}{1 + n} = \frac{w_t}{(1+n) \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]}
```



A steady state is given by a solution to this equation such that
$k_{t+1} = k_t = k^*$, i.e,

```{math}
:label: k_star
    k^* = \frac{s(f(k^*)-k^*f'(k^*), f'(k^*))}{1+n} = \frac{(1-\alpha)(k^*)^{\alpha}}{(1+n) \left [ 1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} \right ]}
```

+++

Let us define a function that takes some parameters and returns the OLG model.

```{code-cell} ipython3
Model = namedtuple('Model', ['α', 'β', 'γ', 'n'])
```

```{code-cell} ipython3
def create_olg_model(α=0.3, β=0.9, γ=0.1, n=0.02):
    return Model(α=α, β=β, γ=γ, n=n)
```

## A Graphical Perspective

To understand the dynamics of the sequence $(k_t)_{t \ge 0}$ we use a 45 degree diagram.

Using {eq}`k_dyms_crra`, and substituting from {eq}`R_func` and {eq}`w_func` we have

$$
    k_{t+1} = \frac{w_t}{(1+n) \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]} = \frac{(1-\alpha) k^{\alpha}_t}{(1+n) \left [ 1 + \beta^{-1/\gamma} (\alpha k^{\alpha-1}_{t+1})^{(\gamma-1)/\gamma} \right ]}
$$

From the above equation, we see that in order to find $k_{t+1}$ we need some root-finding algorithm that solves for $k_{t+1}$ given that we have $k_{t}$.

And suppose, $k_{t+1} = g(k_t)$


So for that we will use [scipy.optimize.newton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton).

```{code-cell} ipython3
def solve_for_k(x, model, k_t):
    z = (1 - model.α) * k_t**model.α
    R1 = model.α ** (1-1/model.γ)
    R2 = x**((model.α * model.γ - model.α + 1) / model.γ)
    p = (1 + model.n) * (x + model.β**(-1/model.γ) * R1 * R2)
    return p - z
```

Let's define a function `k_next` that finds the value of $k_{t+1}$.

```{code-cell} ipython3
def k_next(model, k):
    return optimize.newton(solve_for_k, k, args=(model, k))
```

Let's plot the 45 degree diagram of $k$

```{code-cell} ipython3
def plot45(olg, kstar=None):
    kmin, kmax = 0, 0.3
    m = 1000
    k_grid = np.linspace(kmin, kmax, m)
    k_grid_next = np.empty_like(k_grid)

    for i in range(m):
        k_grid_next[i] = k_next(olg, k_grid[i])

    fig, ax = plt.subplots(figsize=(6, 6))

    ymin, ymax = np.min(k_grid_next), np.max(k_grid_next)

    ax.plot(k_grid, k_grid_next,  lw=2, alpha=0.6, label='$g$')
    ax.plot(k_grid, k_grid, 'k-', lw=1, alpha=0.7, label='45')

    if kstar:
        fps = (kstar,)

        ax.plot(fps, fps, 'go', ms=10, alpha=0.6)

        ax.annotate(r'$k^*$',
                 xy=(kstar, kstar),
                 xycoords='data',
                 xytext=(0, -60),
                 textcoords='offset points',
                 fontsize=14,
                 arrowprops=dict(arrowstyle="->"))

    ax.legend(loc='upper left', frameon=False, fontsize=12)
    ax.set_xlabel('$k_t$', fontsize=12)
    ax.set_ylabel('$k_{t+1}$', fontsize=12)

    plt.show()
```

```{code-cell} ipython3
olg = create_olg_model()
plot45(olg)
```

Suppose, at some $k_t$, the value $g(k_t)$ lies strictly above the 45 degree line.

Then we have $k_{t+1} = g(k_t) > k_t$ and capital per worker rises.

If $g(k_t) < k_t$ then capital per worker falls.

If $g(k_t) = k_t$, then we are at a **steady state** and $k_t$ remains constant.

(A steady state of the model is a [fixed point](https://en.wikipedia.org/wiki/Fixed_point_(mathematics)) of the mapping $g$.)

From the shape of the function $g$ in the figure, we see that
there is a unique steady state in $(0, \infty)$.

+++

Let's find the value of $k^*$.

By observing the above graph, we can see that the value of $k^*$ roughly falls between $(0.15, 0.2)$.
Using this information, we will again use [scipy.optimize.newton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton).

```{code-cell} ipython3
def solve_for_k_star(x, model):
    z = (1 - model.α) * x**model.α
    R1 = model.α ** (1-1/model.γ)
    R2 = x**((model.α * model.γ - model.α + 1) / model.γ)
    p = (1 + model.n) * (x + model.β**(-1/model.γ) * R1 * R2)
    return p - z
```

```{code-cell} ipython3
k_star = optimize.newton(solve_for_k_star, 0.2, args=(olg,))
print(f"k_star = {k_star}")
```

```{code-cell} ipython3
plot45(olg, k_star)
```

From our graphical analysis, it appears that $(k_t)$ converges to $k^*$, regardless of initial capital
$k_0$.

This is a form of global stability.


The next figure shows three time paths for capital, from
three distinct initial conditions, under the parameterization listed above.

At this parameterization, $k^* \approx 0.161$.

Let's define the constants and three distinct intital conditions

```{code-cell} ipython3
ts_length = 10
x0 = np.array([0.001, 0.5, 1.8, 3.5])
```

```{code-cell} ipython3
def simulate_ts(olg, x0_values, ts_length):

    k_star = optimize.newton(solve_for_k_star, 0.2, args=(olg,))
    fig, ax = plt.subplots()

    ts = np.zeros(ts_length)

    # simulate and plot time series
    for x_init in x0_values:
        ts[0] = x_init
        for t in range(1, ts_length):
            ts[t] = k_next(olg, ts[t-1])
        ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
                label=r'$k_0=%g$' %x_init)
    ax.plot(np.arange(ts_length), np.full(ts_length,k_star),
            alpha=0.6, color='red', label=r'$k_*$')
    ax.legend(fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$k_t$', fontsize=14)

    plt.show()
```

```{code-cell} ipython3
simulate_ts(olg, x0, ts_length)
```


## Exercises

```{exercise}
:label: olg_ex1

Replace the utility function $u(c)$ in equation {eq}`eq_crra`  with a quasilinear form $u(c)=c + c^{\alpha}$.

Now we don't have an analytical solution.

Try to compute the time path capital $\{k_t\}$ in this case.
```

```{code-cell} ipython3

```
