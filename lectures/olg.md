---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
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

## model setup

TODO Write a sentence to clarify what the OLG model's structure is.

We assume that

- time is discrete, so that $t=0, 1, \ldots$,
- individuals born at time $t$ live for two periods: $t$ and $t + 1$,


First let's consider the household side.

## household

TODO Write a sentence to illustrate what the household does and what its goal is.

### preference

Suppose that the utility functions for individuals born at time $t$ take the following form

```{math}
:label: eq_crra

    U_t = u(c^1_t) + \beta u(c^2_{t+1}) 
```

Here
- $u: \mathbb R_+ \to \mathbb R$ is the flow utility function satisfying some properties
- $\beta \in (0, 1)$ is the discount factor
- $c^1_t$ is time $t$ consumption of the individual born at time $t$
- $c^2_{t+1}$ is time $t+1$ consumption of the same individual (born at time $t$)

### utility maximization

Savings by an individual of generation $t$, $s_t$, is determined as a
solution to:

```{math}
:label: max_sav_olg
    \begin{aligned}
    \max_{c^1_t, c^2_{t+1}, s_t} \ & \left \{ u(c^1_t) + \beta u(c^2_{t+1}) \right \} \\
    \mbox{subject to } \ & c^1_t + s_t \le w_t \\
                         & c^2_{t+1} \le R_{t+1}s_t\\
    \end{aligned}
```

where 
- $w_t$ is the wage rate at time
- $R_{t+1}$ is time $t+1$ rental rate of capital.

The second constraint incorporates notion that individuals only spend
money on their own end of life consumption. Also, Since $u(\cdot)$ is strictly increasing, both constraints will hold as equalities.

Substituting $s_t$ we get from the first constraint into the second constraint we get $c^2_{t+1}$ in terms of $c^1_t$, i.e., 

```{math}
:label: c_2_olg
    c^2_{t+1} = R_{t+1}(w_t - c^1_t)
```

+++

Thus first-order condition for a maximum can be written in the
familiar form of the consumption Euler equation by plugging $c^2_{t+1}$ into the objective function and taking derivative with respect to $c^1_t$

$$
    u'(c^1_t) = \beta R_{t+1}  u'( R_{t+1} (w_t - c^1_t))
$$

+++

### Further simplification and optimalities

Unless we specify let's assume $u(c) = \log c$. 

Now the Euler equation is simplified to
$$
\frac{w_t - c^1_t}{c^1_t} = \beta
$$

Solving for consumption
$$
c^1_t = \frac{w_t}{1+\beta}, \qquad c^2_{t+1} = \frac{\beta}{1+\beta} R_{t+1} w_t
$$
and thus for savings,
$$
s_t = s(w_t, R_{t+1}) = \frac{\beta}{1+\beta} w_t
$$

+++

Let $L_t$ be the time $t$ labor.

Total savings in the economy will be equal to
$$
    S_t = s_t L_t
$$

+++

## firm

### production function

For each integer $t \geq 0$, output $Y_t$ in period $t$ is given by 

$$
    Y_t = K_t^{\alpha} L_t^{1-\alpha}
$$

Here $K_t$ is capital, $L_t$ is labor, and  $\alpha$ is the output elasticity of capital in the **Cobb-Douglas production function**.

Without population growth, $L_t$ equals some constant $L$.

### profit maximization

Demand for labor $L_t$ and capital $K_t$ is determined by the profit maximization problem
```{math}
\max_{K_t, L_t} \{ K^{\alpha}_t L^{1-\alpha}_t - R_t K_t - L_t w_t   \}
```

+++

### equilibria of capital and labor markets

The first-order conditions for a maximum can be obtained by taking derivative of the objective function with respect to capital and labor respectively and setting to $0$
$$
(1-\alpha)(K_t / L_t)^{\alpha} = w_t
$$
and 
$$
\alpha (K_t / L_t)^{\alpha - 1} = R_t
$$

+++

Equation (12) is the aggregate demand function for capital. 

+++ {"tags": []}

Setting $k_t := K_t / L_t$ and now yields equilibrium condition for capital market:

```{math}
:label: R_func
    1 + r_{t+1} = R_{t+1} = \alpha k^{\alpha-1}_{t+1}
```

Here, the gross rate of return to saving $1 + r_{t+1}$ is equal to the rental rate of capital $R_{t+1}$.

The wage rate is given by

```{math}
:label: w_func
    w_t = (1-\alpha) k^{\alpha}_t
```

+++ {"tags": []}

## equilibrium

+++

### supply and demand

We assume a closed economy, so the goods market equilibrium requires that the demand for goods in each period be equal to supply.


Equivalently, domestic investment equals aggregate domestic
saving
$$
K_{t+1} - K_t = L_t s(w_t, R_{t+1}) - K_t
$$
where the LHS is net investment of the change in capital stock between $t$ and $t+1$, while the RHS is net saving of saving of the young and dissaving of the old.

Eliminating $K_t$ from both sides yields

```{math}
\begin{equation}
    K_{t+1} & = S_{t} \\
            & = L_t s(w_t, R_{t+1})
\end{equation}
```

+++

Together with optimal saving (7) and labor market equilibrium condition (11) gives
```{math}
\begin{equation}
K_{t+1} & = L_t s(w_t, R_{t+1})\\
        & = \frac{\beta}{1+\beta} w_t \\
        & = \frac{\beta}{1+\beta} (1-\alpha)(K_t / L_t)^{\alpha}
\end{equation}
```

+++

Equation (16) is the aggregate supply function for capital $K_{t+1}$ of interest rate of capital $R_{t+1}$.

Notice that in the log preference case the aggregate capital is supplied at a fixed term, regardless of its price.

+++

Equation (11) and letting $L_{t+1} = (1 + n) L_t$ gives the aggregate demand for capital 
$$
K_{t+1} = L_{t+1} ( \frac{R_{t+1}}{\alpha} )^{1/ (\alpha - 1)} = (1 + n) L_t ( \frac{R_{t+1}}{\alpha} )^{1/ (\alpha - 1)}
$$

+++

### Visualising the equilibrium

Once $n, \alpha$ and $\beta$ are fixed and $K_t$ and $L_t$ are given we can plot the aggregate demand and supply for capital.

First let's define the model with parameters using the name tuple.

```{code-cell} ipython3
Model = namedtuple('Model', ['α',  # output elasticity of capital in the Cobb-Douglas production function
                             'β',  # discount factor
                             'u',  # parameter which defines the flow utility 
                             'n',  # population growth rate
                             'L0'] # initial population size
                   )
```

Then define the utility function and create an instance of the model.

```{code-cell} ipython3
# def u(c, γ=2):
#     return c**(1 - γ) / (1 - γ)
```

```{code-cell} ipython3
def u(c):
    return np.log(c)
```

```{code-cell} ipython3
def create_olg_model(α=0.3, β=0.9, u=u, n=0.02, L0=10_000.0):
    return Model(α=α, β=β, u=u, n=n, L0=L0)
```

```{code-cell} ipython3
m = create_olg_model()
```

Define some functions

```{code-cell} ipython3
def aggregate_population(t, model):
    n, L0 = model.n, model.L0
    return L0 * (1+n)**t
```

```{code-cell} ipython3
def aggregate_capital_demand(R, t, model):
    α = model.α
    return aggregate_population(t, model) * (R/α)**(1/(α-1))
```

```{code-cell} ipython3
def aggregate_capital_supply(R, t, model, K_prev=30):
    α, β = model.α, model.β
    λ = np.ones_like(R)
    return λ * β / (1+β) * (1-α) * (K_prev / aggregate_population(t-1, model))**(1/(α-1))
```

Then we can get the market equilibrium.

```{code-cell} ipython3
t=100
R_vals = np.linspace(0.4, 1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(aggregate_capital_demand(R_vals, t, m), R_vals, label="aggregate demand")
ax.plot(aggregate_capital_supply(R_vals, t, m), R_vals, label="aggregate supply")
ax.set_xlabel("capital")
ax.set_ylabel("interest rate")
ax.legend()
plt.show()
```

### Dynamics and steady state

+++



+++

Recalling equation (16) and $L_{t+1} = (1 + n) L_t$. Setting $k_t := K_t / L_t$ now yields:

```{math}
:label: k_dyms_crra
    k_{t+1} = \frac{\beta (1-\alpha) (k_t)^{\alpha}}{(1+\beta)(1+n)}
```
With function defined below we can plot this law of motion for per capita capital against a 45 degree line.

```{code-cell} ipython3
def k_update(k, model):
    α, β, n = model.α, model.β, model.n
    return β * (1 - α) * k**α / ((1 + n) * (1 + β))
```

```{code-cell} ipython3
def plot_45(k_update, kstar=None):
    kmin, kmax = 0, 0.4
    x = 1000
    k_grid = np.linspace(kmin, kmax, x)
    k_grid_next = np.empty_like(k_grid)

    for i in range(x):
        # print(k_grid[i])
        k_grid_next[i] = k_update(k_grid[i], m)

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
plot_45(k_update, kstar=None)
```

Here a steady state can be solved analytically.

That is, $k_{t+1} = k_t = k^*$,
$$
k^* = \frac{\beta (1-\alpha) (k^*)^{\alpha}}{(1+\beta)(1+n)}
$$

+++

We can solve for
$$
k^* = \left (\frac{\beta (1-\alpha)}{ (1+\beta) (1+n)} \right )^{1/(1-\alpha)}
$$

```{code-cell} ipython3
def k_star(model):
    α, β, n = model.α, model.β, model.n
    return (β * (1-α) / ((1+β) * (1+n)) )**(1/(1-α))
```

```{code-cell} ipython3
plot_45(k_update, kstar=k_star(m))
```

## crra


+++

Now let's assume that the model is the same except that $u(c) = \frac{ c^{1- \gamma}-1}{1-\gamma}$, where $\gamma >0$.

+++

### equilibria

+++

Similarly we can solve the optimization problem of the household and the firm analytically.

For households now the euler equation becomes
$$
    (c^1_t)^{-\gamma} = \beta R^{1-\gamma}_{t+1}  (w_t - c^1_t)^{-\gamma}
$$

Solving for consumption and thus for savings,

$$
    s_t = s(w_t, R_{t+1}) = w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
$$

Updating Equation (16) with the new $s_t$ from (22) and labor market equilibrium condition from (11) gives 

```{math}
\begin{equation}
    K_{t+1} & = L_t s(w_t, R_{t+1}) \\
            & = L_t w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1} \\
            & = L_t (1-\alpha)(K_t / L_t)^{\alpha} \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
\end{equation}
```

```{code-cell} ipython3
def crra(c, γ=2):
    return c**(1 - γ) / (1 - γ)
```

```{code-cell} ipython3
m = create_olg_model(u=crra)
```

```{code-cell} ipython3
def aggregate_capital_supply_crra(R, t, model, K_prev=5000):
    α, β, γ= model.α, model.β, model.u.__defaults__[0]
    λ = np.ones_like(R)
    B = 1/(1+ β**(-1/γ) * R**((γ-1)/γ))
    return λ * aggregate_population(t-1, model)**(1-α) * (1-α) * K_prev**α * B
```

```{code-cell} ipython3
t=10
R_vals = np.linspace(0.4, 1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(aggregate_capital_demand(R_vals, t, m), R_vals, label="aggregate demand")
ax.plot(aggregate_capital_supply_crra(R_vals, t, m), R_vals, label="aggregate supply")
ax.set_xlabel("capital")
ax.set_ylabel("interest rate")
ax.legend()
plt.show()
```

### difficulty in obtaining dynamics and analytical solution

Recalling $L_{t+1} = (1 + n) L_t$ and setting $k_t := K_t / L_t$ now yields:

```{math}
    k_{t+1} = \frac{1-\alpha}{1+n} k^{\alpha}_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
```

+++

Together with capital market equilibrium condition (13) yields the law of motion for the per capita capital
```{math}
    k_{t+1} = \frac{1-\alpha}{1+n} k^{\alpha}_t \left [ 1 + \beta^{-1/\gamma} \left ( \alpha k^{\alpha-1}_{t+1} \right )^{(\gamma-1)/\gamma} \right ]^{-1}
```
Note that with this equation and $k_t$ as given we cannot solve for $k_{t+1}$ by pencil and paper.

To solve for $k_{t+1}$ we need to turn to newton's method.

+++

Suppose, 
$$
f(k_{t+1}; k_t)=(1+n)k_{t+1} \left [ 1 + \beta^{-1/\gamma} \left ( \alpha k^{\alpha-1}_{t+1} \right )^{(\gamma-1)/\gamma} \right ] - \frac{1-\alpha}{1+n} k^{\alpha}_t =0
$$

If $k_t$ is given then $f(\cdot)$ is a function of unknown $k_{t+1}$.

Then we can use [scipy.optimize.newton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton) to solve the Equation $f(k_{t+1})=0$ for $k_{t+1}$.

First let define $f(\cdot)$.

```{code-cell} ipython3
def f(k_prime, k, model):
    α, β, n, γ = model.α, model.β, model.n, model.u.__defaults__[0]
    z = (1 - α) * k**α
    R1 = α ** (1-1/γ)
    R2 = k_prime**((α * γ - α + 1) / γ)
    p = (1 + n) * (k_prime + β**(-1/γ) * R1 * R2)
    return p - z
```

Let's define a function `k_next` that finds the value of $k_{t+1}$.

```{code-cell} ipython3
def k_next(k_prime, model):
    return optimize.newton(f, k_prime, args=(k_prime, model))
```

```{code-cell} ipython3
plot_45(k_next, kstar=None)
```

Unlike the log preference case now a steady state cannot be solved analytically either.

To see this recall that, a steady state can be obtained by setting equation (25) to $k_{t+1} = k_t = k^*$, i.e.,
```{math}
    k^* = \frac{s(f(k^*)-k^*f'(k^*), f'(k^*))}{1+n} = \frac{(1-\alpha)(k^*)^{\alpha}}{(1+n) \left [ 1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} \right ]}
```

+++

Similarly to solve for $k^*$ we need newton's method.

Suppose that
$$
    g(k^*) = k^* (1+n) \left [ 1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} \right ] - (1-\alpha)(k^*)^{\alpha}=0
$$

```{code-cell} ipython3
def g(k_star, model):
    α, β, n, γ = model.α, model.β, model.n, model.u.__defaults__[0]
    z = (1 - α) * k_star**α
    R1 = α ** (1-1/γ)
    R2 = k_star**((α * γ - α + 1) / γ)
    p = (1 + n) * (k_star + β**(-1/γ) * R1 * R2)
    return p - z
```

```{code-cell} ipython3
k_star = optimize.newton(g, 0.2, args=(m,))
print(f"k_star = {k_star}")
```

```{code-cell} ipython3
plot_45(k_next, k_star)
```

The next figure shows three time paths for capital, from
three distinct initial conditions, under the parameterization listed above.

At this parameterization, $k^* \approx 0.161$.

Let's define the constants and three distinct intital conditions

```{code-cell} ipython3
ts_length = 10
x0 = np.array([0.001, 1.2, 2.6])
```

```{code-cell} ipython3
def simulate_ts(m, x0_values, ts_length):

    k_star = optimize.newton(g, 0.2, args=(m,))
    fig, ax = plt.subplots(figsize=(10, 5))

    ts = np.zeros(ts_length)

    # simulate and plot time series
    for x_init in x0_values:
        ts[0] = x_init
        for t in range(1, ts_length):
            ts[t] = k_next(ts[t-1], m)
        ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
                label=r'$k_0=%g$' %x_init)
    ax.plot(np.arange(ts_length), np.full(ts_length,k_star),
            alpha=0.6, color='red', label=r'$k^*$')
    ax.legend(fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$k_t$', fontsize=14)

    plt.show()
```

```{code-cell} ipython3
simulate_ts(m, x0, ts_length)
```

## exercises

+++

TODO Add exercise-solution environment

### Exercises


Let's keep the model the same except for replacing the utility function $u(c)$ in equation {eq}`eq_crra`  with a quasilinear form $u(c)=c + c^{\theta}$.

Like what we did in the crra case we don't have an analytical solution. 

Try to compute the time path capital $\{k_t\}$ in this case.

+++

### Solution

+++

To get the time path capital $\{k_t\}$ first we need to solve the household's utility maximization problem for the optimal consumption and optimal saving.

+++

#### step 1

With the quasilinear preference the euler equation becomes

$$
    1 + \theta (c^1_t)^{\theta-1} = \beta R_{t+1} + \beta R^{\theta}_{t+1} \theta (w_t - c^1_t)^{\theta - 1}
$$

Obviously it cannot be solved by pencil and paper. 

To solve for the optimal consumption and saving we need to turn to the newton's method.

```{code-cell} ipython3
# Hi Smit please fill in the trial answers here
```

#### step 2

With equilibrium conditions for capital and labor (11) and (12) and the result from step 1, we can calculate the law of motion for per capita capital 
$$
k_{t+1} = \frac{s(w_t, R_{t+1})}{1 + n}
$$

We will find again that we cannot simply solve for $k_{t+1}$ given $k_t$ from this equation. 

Therefore we need to turn to newton's method again.

```{code-cell} ipython3
# Hi Smit please fill in the trial answers here
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# This code cell stores Smit's previous code

def u_prime(c, θ):
    return 1 + θ * c ** (θ - 1)

def solve_for_c(x, θ, model, k_t, k_t_1):
    l = u_prime(x, θ)
    R = model.α * k_t_1**(model.α-1)
    r = model.β * R
    y = R * ((1 - model.α) * k_t**model.α - x)
    r = r * u_prime(y, θ)
    return l - r

def solve_for_k(x, θ, model, k_t):
    c = optimize.newton(solve_for_c, k_t, args=(θ, model, k_t, x))
    num = (1 - model.α) * k_t**model.α - c
    den = (1 + model.n)
    return x - num/den

def k_next(model, θ, k_t):
    return optimize.newton(solve_for_k, k_t, args=(θ, model, k_t), maxiter=200)

ts_length = 10
θ = 1.5
x0 = np.array([1.2, 2.6])

def simulate_ts(olg, θ, x0_values, ts_length):

    fig, ax = plt.subplots(figsize=(10, 5))

    ts = np.zeros(ts_length)

    # simulate and plot time series
    for x_init in x0_values:
        ts[0] = x_init
        for t in range(1, ts_length):
            ts[t] = k_next(olg, θ, ts[t-1])
        ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
                label=r'$k_0=%g$' %x_init)
    ax.plot(np.arange(ts_length), np.full(ts_length,k_star),
            alpha=0.6, color='red', label=r'$k^*$')
    ax.legend(fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$k_t$', fontsize=14)

    plt.show()
    
simulate_ts(olg, θ, x0, ts_length)
```
