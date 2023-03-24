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

## Environment

TODO add timing and basic ideas of OLG

We assume that

- time is discrete, so that $t=0, 1, \ldots$,
- individuals born at time $t$ live for two periods: $t$ and $t + 1$,
- capital depreciates fully after one period (TODO to be checked)


First let's consider the household side.

+++

TODO label and recall math equations using correct internal reference

## Supply of capital

### Consumer's problem

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
Thus first-order condition for a maximum can be written in the
familiar form of the consumption Euler equation by plugging $c^2_{t+1}$ into the objective function and taking derivative with respect to $c^1_t$

$$
    u'(c^1_t) = \beta R_{t+1}  u'( R_{t+1} (w_t - c^1_t))
$$

From the first constraint we get
```{math}
    c^1_{t} = w_t - s_t
```

With it the Euler equation (4) becomes
```{math}
    u'(w_t - s_t) = \beta R_{t+1}  u'( R_{t+1} s_t)
```

+++

From this we can solve for savings analytically or numerically
```{math}
s_t = s(w_t, R_{t+1})
```

+++

Let $L_t$ be the time $t$ labor.

Furthermore let's also assume a constant population size, i.e., $L_{t+1}=L_t=L$ for all $t$.

Total savings in the economy will be equal to
$$
    S_t = s_t L
$$

In our closed economy, the net saving this period will be equal to the supply next period, i.e., 

$$
K_{t+1} = K^S(w_t, R_{t+1}) =  S_t = L s_t = L s(w_t, R_{t+1}) 
$$

Here $K^S(w_t, R_{t+1})$ means the invariant function relationship between aggregate capital supply $K_{t+1}$ and wage $w_t$ and return rate $R_{t+1}$.

+++

### Special case: log preference

Unless we specify let's assume $u(c) = \log c$. 

Now the Euler equation is simplified to
$$
s_t= \beta (w_t - s_t) 
$$

Solving for saving,
$$
s_t = s(w_t, R_{t+1}) = \frac{\beta}{1+\beta} w_t
$$

+++

And hence aggregate supply of capital
$$
K_{t+1} = K^s(R_{t+1}) = Ls_t = L \frac{\beta}{1+\beta} w_t
$$

+++

## Demand for capital

### Firm's problem

For each integer $t \geq 0$, output $Y_t$ in period $t$ is given by 

$$
    Y_t = K_t^{\alpha} L_t^{1-\alpha}
$$

Here $K_t$ is capital, $L_t$ is labor, and  $\alpha$ is the output elasticity of capital in the **Cobb-Douglas production function**.

Without population growth, $L_t$ equals some constant $L$.

Demand for labor $L$ and capital $K_t$ is determined by the profit maximization problem
```{math}
\max_{K_t, L} \{ K^{\alpha}_t L^{1-\alpha} - R_t K_t - L w_t   \}
```

### Demand for capital

The first-order conditions for a maximum can be obtained by taking derivative of the objective function with respect to capital and labor respectively and setting to $0$
$$
(1-\alpha)(K_t / L)^{\alpha} = w_t
$$
and 
$$
\alpha (K_t / L)^{\alpha - 1} = R_t
$$

Rearranging Equation (16) gives the aggregate demand for capital

$$
K_{t+1} = K^d (R_{t+1}) = L \left (\frac{R_{t+1}}{\alpha} \right )^{1/(\alpha - 1)}
$$

+++

## Equilibrium

The equality of aggregate demand (12) and aggregate supply (16) for capital yields the equalibrium. 

Specifically we have
$$
K^s(R_{t+1}) = K^d(R_{t+1})
$$
or equivalently 
$$
L\frac{\beta}{1+\beta} (1-\alpha)(K_t / L)^{\alpha} = L\left (\frac{R_{t+1}}{\alpha} \right )^{1/(\alpha - 1)}
$$

+++

Then we can solve for the equilibrium price
$$
R^*_{t+1} = \alpha \left ( \frac{\beta (1-\alpha)(K_t / L)^{\alpha}}{1+\beta} \right )^{\alpha - 1}
$$

+++

Plugging it into either aggregate demand or supply function gives equilibrium quantity
$$
K^*_{t+1} = \frac{\beta }{1+\beta} (1-\alpha)(K_t / L)^{\alpha} L
$$

```{code-cell} ipython3
Model = namedtuple('Model', ['α',  # output elasticity of capital in the Cobb-Douglas production function
                             'β',  # discount factor
                             'u',  # parameter which defines the flow utility 
                             'L']  # population size
                   )
```

```{code-cell} ipython3
def u(c):
    return np.log(c)
```

```{code-cell} ipython3
def create_olg_model(α=0.3, β=0.9, u=u, L=10.0):
    return Model(α=α, β=β, u=u, L=L)
```

```{code-cell} ipython3
def equilibrium(model, K_prev):
    α, β, L = model.α, model.β, model.L
    R = α * ( β * (1-α) * (K_prev / L)**α / (1+β))**(-1)
    K = β / (1+β) * (1-α) * (K_prev / L)**α * L
    return R, K
```

```{code-cell} ipython3
def aggregate_capital_demand(R, model, K_prev):
    α, L = model.α, model.L
    return (R/α)**(1/(α-1)) * L
```

```{code-cell} ipython3
def aggregate_capital_supply(R, model, K_prev):
    α, β, L = model.α, model.β, model.L
    λ = np.ones_like(R)
    return λ * β / (1+β) * (1-α) * (K_prev / L)**α * L
```

```{code-cell} ipython3
m = create_olg_model()
R_vals = np.linspace(0.3, 1)
K_prev = 50
```

```{code-cell} ipython3
equilibrium(m, K_prev)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(R_vals, aggregate_capital_demand(R_vals, m, K_prev), label="aggregate demand")
ax.plot(R_vals, aggregate_capital_supply(R_vals, m, K_prev), label="aggregate supply")
ax.set_xlabel("$R_{t+1}$")
ax.set_ylabel("$K_{t+1}$")
ax.legend()
plt.show()
```

## Dynamics and steady state

Setting $k_t := K_t / L$.


Aggregate supply of capital (12) becomes
$$
k_{t+1} = k^s(R_{t+1}) = \frac{\beta}{1+\beta} w_t
$$

Equation (15) becomes
$$
(1-\alpha)(k_t)^{\alpha} = w_t
$$

+++

Combining (22) and (23) yields the law of motion for capital
$$
k_{t+1} = \frac{\beta}{1+\beta} (1-\alpha)(k_t)^{\alpha}
$$

+++

A steady state can be solved analytically.

That is, $k_{t+1} = k_t = k^*$,
$$
k^* = \frac{\beta (1-\alpha) (k^*)^{\alpha}}{(1+\beta)}
$$

+++

We can solve for
$$
k^* = \left (\frac{\beta (1-\alpha)}{1+\beta} \right )^{1/(1-\alpha)}
$$

```{code-cell} ipython3
def k_update(k, model):
    α, β = model.α, model.β
    return β * (1 - α) * k**α /  (1 + β)
```

```{code-cell} ipython3
def plot_45(m, k_update, kstar=None):
    kmin, kmax = 0, 0.6
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
plot_45(m, k_update, kstar=None)
```

```{code-cell} ipython3
def k_star(model):
    α, β = model.α, model.β
    return (β * (1-α) / (1+β)  )**(1/(1-α))
```

```{code-cell} ipython3
plot_45(m, k_update, kstar=k_star(m))
```

## Another special case: CRRA preference

+++

If we assume that the model is the same except that $u(c) = \frac{ c^{1- \gamma}-1}{1-\gamma}$, where $\gamma >0$.

```{code-cell} ipython3
def crra(c, γ=2):
    return c**(1 - γ) / (1 - γ)
```

```{code-cell} ipython3
m_crra = create_olg_model(u=crra)
```

For households now the euler equation becomes
$$
    (w_t - s_t)^{-\gamma} = \beta R^{1-\gamma}_{t+1}  (s_t)^{-\gamma}
$$

+++

Solving for savings, we have

$$
    s_t = s(w_t, R_{t+1}) = w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
$$

+++

Setting $k_t := K_t / L$ and using (28).


Aggregate supply of capital (12) becomes
$$
k_{t+1} = k^s(R_{t+1}) = \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1} w_t
$$

+++

Equation (16) becomes
$$
\alpha k^{\alpha - 1}_t = R_t
$$

+++

Combing with equations (23) and (30) gives
$$
k_{t+1} = \left [ 1 + \beta^{-1/\gamma} (\alpha k^{\alpha - 1}_{t+1})^{(\gamma-1)/\gamma} \right ]^{-1} (1-\alpha)(k_t)^{\alpha} 
$$

+++

Note that with this equation and $k_t$ as given we cannot solve for $k_{t+1}$ by pencil and paper.

That is we don't have an analytical solution for the sample path $\{k_{t+1}\}$ now.

To solve for $k_{t+1}$ we need to turn to newton's method.

Suppose, 
$$
f(k_{t+1}; k_t)=k_{t+1} \left [ 1 + \beta^{-1/\gamma} \left ( \alpha k^{\alpha-1}_{t+1} \right )^{(\gamma-1)/\gamma} \right ] - (1-\alpha) k^{\alpha}_t =0
$$

If $k_t$ is given then $f(\cdot)$ is a function of unknown $k_{t+1}$.

Then we can use [scipy.optimize.newton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton) to solve the Equation $f(k_{t+1})=0$ for $k_{t+1}$.

First let define $f(\cdot)$.

```{code-cell} ipython3
def f(k_prime, k, model):
    α, β, γ = model.α, model.β, model.u.__defaults__[0]
    z = (1 - α) * k**α
    R1 = α ** (1-1/γ)
    R2 = k_prime**((α * γ - α + 1) / γ)
    p = k_prime + k_prime * β**(-1/γ) * R1 * R2
    return p - z
```

Let's define a function `k_next` that finds the value of $k_{t+1}$.

```{code-cell} ipython3
def k_next(k_prime, model):
    return optimize.newton(f, k_prime, args=(k_prime, model))
```

```{code-cell} ipython3
plot_45(m_crra, k_next, kstar=None)
```

Unlike the log preference case now a steady state cannot be solved analytically either.

To see this recall that, a steady state can be obtained by setting equation (25) to $k_{t+1} = k_t = k^*$, i.e.,
```{math}
\begin{equation}
    k^* & = \frac{(1-\alpha)(k^*)^{\alpha}}{  1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma}}
\end{equation}
```

+++

Similarly we can solve for $k^*$ using newton's method.

Suppose that
$$
    g(k^*) = k^*  \left [ 1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} \right ] - (1-\alpha)(k^*)^{\alpha}
$$

```{code-cell} ipython3
def g(k_star, model):
    α, β, γ = model.α, model.β, model.u.__defaults__[0]
    z = (1 - α) * k_star**α
    R1 = α ** (1-1/γ)
    R2 = k_star**((α * γ - α + 1) / γ)
    p = k_star + k_star * β**(-1/γ) * R1 * R2
    return p - z
```

```{code-cell} ipython3
k_star = optimize.newton(g, 0.2, args=(m_crra,))
print(f"k_star = {k_star}")
```

```{code-cell} ipython3
plot_45(m_crra, k_next, k_star)
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
simulate_ts(m_crra, x0, ts_length)
```

## Exercises

+++

TODO Add exercise-solution environment

### Exercise 1


Let's keep the model the same except for replacing the utility function $u(c)$ in equation {eq}`eq_crra`  with a quasilinear form $u(c)=c + c^{\theta}$.

Like what we did in the crra case we don't have an analytical solution. 

Try to compute the time path capital $\{k_t\}$ in this case.

+++

### Solution to exercise 1

To get the time path capital $\{k_t\}$ first we need to solve the household's utility maximization problem for the optimal consumption and optimal saving.

#### step 1

With the quasilinear preference the euler equation becomes

$$
    1 + \theta (w_t - s_t)^{\theta-1} = \beta R_{t+1} + \beta R^{\theta}_{t+1} \theta s_t^{\theta - 1}
$$

Obviously $s_t$ cannot be solved by pencil and paper. 

To solve for $s_t$ we need to turn to the newton's method.

```{code-cell} ipython3
# Hi Smit please fill in the trial answers here
```

#### step 2

With equilibrium conditions for capital and labor (11) and (12) and the result from step 1, we can calculate the law of motion for per capita capital 
$$
k_{t+1} = s(w_t, R_{t+1})
$$

We will find again that we cannot analytically solve for $k_{t+1}$ given $k_t$ from this equation. 

Therefore we need to turn to newton's method again.

```{code-cell} ipython3
# Hi Smit please fill in the trial answers here
```

```{code-cell} ipython3
# Smit's previous code

# def u_prime(c, θ):
#     return 1 + θ * c ** (θ - 1)

# def solve_for_c(x, θ, model, k_t, k_t_1):
#     l = u_prime(x, θ)
#     R = model.α * k_t_1**(model.α-1)
#     r = model.β * R
#     y = R * ((1 - model.α) * k_t**model.α - x)
#     r = r * u_prime(y, θ)
#     return l - r

# def solve_for_k(x, θ, model, k_t):
#     c = optimize.newton(solve_for_c, k_t, args=(θ, model, k_t, x))
#     num = (1 - model.α) * k_t**model.α - c
#     den = (1 + model.n)
#     return x - num/den

# def k_next(model, θ, k_t):
#     return optimize.newton(solve_for_k, k_t, args=(θ, model, k_t), maxiter=200)

# ts_length = 10
# θ = 1.5
# x0 = np.array([1.2, 2.6])

# def simulate_ts(olg, θ, x0_values, ts_length):

#     fig, ax = plt.subplots(figsize=(10, 5))

#     ts = np.zeros(ts_length)

#     # simulate and plot time series
#     for x_init in x0_values:
#         ts[0] = x_init
#         for t in range(1, ts_length):
#             ts[t] = k_next(olg, θ, ts[t-1])
#         ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
#                 label=r'$k_0=%g$' %x_init)
#     ax.plot(np.arange(ts_length), np.full(ts_length,k_star),
#             alpha=0.6, color='red', label=r'$k^*$')
#     ax.legend(fontsize=10)

#     ax.set_xlabel(r'$t$', fontsize=14)
#     ax.set_ylabel(r'$k_t$', fontsize=14)

#     plt.show()
    
# simulate_ts(olg, θ, x0, ts_length)
```

```{code-cell} ipython3

```
