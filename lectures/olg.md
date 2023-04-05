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
from functools import partial
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



## Supply of capital

### Consumer's problem

Suppose that utility for individuals born at time $t$ take the form

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

The second constraint incorporates the notion that individuals only spend
money on their own end of life consumption. Also, Since $u$ is strictly increasing, both constraints will hold as equalities.

Substituting $s_t$ from the first constraint into the second we get $c^2_{t+1}$ in terms of $c^1_t$, i.e.,

```{math}
:label: c_2_olg
    c^2_{t+1} = R_{t+1}(w_t - c^1_t)
```
Thus first-order condition for a maximum can be written in the
familiar form of the consumption Euler equation by plugging $c^2_{t+1}$ into the objective function and taking derivative with respect to $c^1_t$

```{math}
:label: euler_1_olg
    u'(c^1_t) = \beta R_{t+1}  u'( R_{t+1} (w_t - c^1_t))
```

From the first constraint we get
```{math}
:label: c_1_olg
    c^1_{t} = w_t - s_t
```

With it the Euler equation (4) becomes
```{math}
:label: euler_2_olg
    u'(w_t - s_t) = \beta R_{t+1}  u'( R_{t+1} s_t)
```



From this we can solve for savings analytically or, if necessary, numerically
```{math}
:label: saving_1_olg
    s_t = s(w_t, R_{t+1})
```



Let $L_t$ be the time $t$ labor.

Furthermore let's assume a constant population size, i.e., $L_{t+1}=L_t=L$ for all $t$.

Total savings in the economy will be equal to
```{math}
:label: total_savings_1_olg
    S_t = s_t L
```

In our closed economy, net saving this period will be equal to supply next period, i.e.,

```{math}
:label: aggregate_supply_capital_1_olg
    K_{t+1} = K^S(w_t, R_{t+1}) =  S_t = L s_t = L s(w_t, R_{t+1})
```

Here $K^S$ is a time-invariant function mapping wage $w_t$ and capital return rate $R_{t+1}$ to aggregate capital supply $K_{t+1}$.



### Special case: log preference

Assume $u(c) = \log c$.

The Euler equation simplifies to
```{math}
:label: saving_log_1_olg
    s_t= \beta (w_t - s_t)
```

Solving for saving,
```{math}
:label: saving_log_2_olg
    s_t = s(w_t, R_{t+1}) = \frac{\beta}{1+\beta} w_t
```



And hence
```{math}
:label: aggregate_supply_capital_log_olg
    K_{t+1} = K^s(R_{t+1}) = Ls_t = L \frac{\beta}{1+\beta} w_t
```



## Demand for capital

### Firm's problem

For each integer $t \geq 0$, output $Y_t$ in period $t$ is given by

```{math}
:label: cobb_douglas
    Y_t = K_t^{\alpha} L_t^{1-\alpha}
```

Here $K_t$ is capital, $L_t$ is labor, and  $\alpha$ is the output elasticity of capital in the **Cobb-Douglas production function**.


Demand for labor $L$ and capital $K_t$ is determined by the profit maximization problem
```{math}
:label: opt_profit_olg
    \max_{K_t, L} \{ K^{\alpha}_t L^{1-\alpha} - R_t K_t - L w_t   \}
```

### Demand for capital

The first-order conditions for a maximum can be obtained by taking the derivative of the objective function with respect to capital and labor respectively and setting it to zero:

```{math}
:label: wage
    (1-\alpha)(K_t / L)^{\alpha} = w_t
```

and

```{math}
:label: interest_rate
    \alpha (K_t / L)^{\alpha - 1} = R_t
```

Rearranging [](interest_rate) gives the aggregate demand for capital

```{math}
:label: aggregate_demand_capital_olg
    K_{t+1} = K^d (R_{t+1}) = L \left (\frac{R_{t+1}}{\alpha} \right )^{1/(\alpha - 1)}
```



## Equilibrium

The equality of aggregate demand [](aggregate_demand_capital_olg) and aggregate supply [](aggregate_supply_capital_log_olg) for capital yields the equalibrium.

Specifically we have

```{math}
:label: equilibrium_1
    K^s(R_{t+1}) = K^d(R_{t+1})
```

or, equivalently,

```{math}
:label: equilibrium_2
    L\frac{\beta}{1+\beta} (1-\alpha)(K_t / L)^{\alpha} = L\left (\frac{R_{t+1}}{\alpha} \right )^{1/(\alpha - 1)}
```



Then we can solve for the equilibrium price

```{math}
:label: equilibrium_price
    R^*_{t+1} = \alpha \left ( \frac{\beta (1-\alpha)(K_t / L)^{\alpha}}{1+\beta} \right )^{\alpha - 1}
```



Plugging it into either the demand or the supply function gives the equilibrium quantity
```{math}
:label: equilibrium_quantity
    K^*_{t+1} = \frac{\beta }{1+\beta} (1-\alpha)(K_t / L)^{\alpha} L
```

```{code-cell} ipython3
Model = namedtuple('Model', ['α',        # output elasticity of capital in the Cobb-Douglas production function
                             'β',        # discount factor
                             'u',        # parameter which defines the flow utility
                             'L',        # population size
                             'u_params'] # other params used to define u
                   )
```

```{code-cell} ipython3
def u(c):
    return np.log(c)
```

```{code-cell} ipython3
def create_olg_model(α=0.3, β=0.9, u=u, L=10.0, u_params=dict()):
    return Model(α=α, β=β, u=u, L=L, u_params=u_params)
```

```{code-cell} ipython3
def equilibrium(model, K_prev):
    α, β, L = model.α, model.β, model.L
    R = α * ( β * (1-α) * (K_prev / L)**α / (1+β))**(α-1)
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
def plot_ad_as(demand, supply, m, K_prev, E_star=None):

    R_vals = np.linspace(0.3, 1)

    fig, ax = plt.subplots()

    ax.plot(R_vals, demand(R_vals, m, K_prev), label="aggregate demand")
    ax.plot(R_vals, supply(R_vals, m, K_prev), label="aggregate supply")

    if E_star:
        R_star, K_star = E_star

        ax.plot(R_star, K_star, 'go', ms=10, alpha=0.6)

        ax.annotate(r'Equilibrium',
                 xy=(R_star, K_star),
                 xycoords='data',
                 xytext=(0, -60),
                 textcoords='offset points',
                 fontsize=14,
                 arrowprops=dict(arrowstyle="->"))

    ax.set_xlabel("$R_{t+1}$")
    ax.set_ylabel("$K_{t+1}$")
    ax.legend()
    plt.show()
```

```{code-cell} ipython3
m = create_olg_model()
K_prev = 50
```

```{code-cell} ipython3
E_star = equilibrium(m, K_prev)
```

```{code-cell} ipython3
R_star, K_star = E_star
```

```{code-cell} ipython3
plot_ad_as(aggregate_capital_demand, aggregate_capital_supply, m, K_prev=50, E_star=E_star)
```

Let's observe the dynamics of the equilibrium price $R^*_{t+1}$.

```{code-cell} ipython3
K_t_vals = np.linspace(10, 500, 10_000)
R_t1_vals = m.α * (m.β * (1-m.α) * (K_t_vals / m.L)**m.α / (1+m.β))**(m.α-1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(K_t_vals, R_t1_vals)
ax.set_ylabel("$R^*_{t+1}$")
ax.set_xlabel("$K_{t}$")
plt.show()
```

## Dynamics and steady state

Let $k_t := K_t / L$.


Aggregate supply of capital [](aggregate_supply_capital_log_olg) becomes
```{math}
:label: supply_capital_log_olg
    k_{t+1} = k^s(R_{t+1}) = \frac{\beta}{1+\beta} w_t
```

[](wage) becomes
```{math}
:label: wage_2
    (1-\alpha)(k_t)^{\alpha} = w_t
```



Combining [](supply_capital_log_olg) and [](wage_2) yields the law of motion for capital
```{math}
:label: law_of_motion_capital
    k_{t+1} = \frac{\beta}{1+\beta} (1-\alpha)(k_t)^{\alpha}
```



A steady state can be solved analytically in this case.

That is, $k_{t+1} = k_t = k^*$, where
```{math}
:label: steady_state_1
    k^* = \frac{\beta (1-\alpha) (k^*)^{\alpha}}{(1+\beta)}
```



We can solve this equation to obtain
```{math}
:label: steady_state_2
    k^* = \left (\frac{\beta (1-\alpha)}{1+\beta} \right )^{1/(1-\alpha)}
```

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



Let's now assume that the model is the same except that $u(c) = \frac{ c^{1- \gamma}-1}{1-\gamma}$, where $\gamma >0, \gamma\neq 1$.

```{code-cell} ipython3
def crra(c, γ=0.5):
    return c**(1 - γ) / (1 - γ)
```

```{code-cell} ipython3
m_crra = create_olg_model(u=crra, u_params={'γ': 0.5})
```

### New aggregate supply



For households, the Euler equation becomes
```{math}
:label: euler_crra
    (w_t - s_t)^{-\gamma} = \beta R^{1-\gamma}_{t+1}  (s_t)^{-\gamma}
```



Solving for savings, we have

```{math}
:label: saving_crra
    s_t = s(w_t, R_{t+1}) = w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
```



With the CRRA utility, the aggregate supply of capital [](aggregate_supply_capital_1_olg) becomes
```{math}
:label: aggregate_supply_capital_crra_olg
    K_{t+1} = K^S(w_t, R_{t+1}) =  S_t = L s_t = L s(w_t, R_{t+1}) = L w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1}
```



### Equilibrium

The equality of aggregate demand [](aggregate_demand_capital_olg) and new aggregate supply [](aggregate_supply_capital_crra_olg) for capital yields the equalibrium.

Specifically we have

```{math}
:label: equilibrium_crra_1
    K^S(R_{t+1}) = K^d(R_{t+1})
```

or equivalently

```{math}
:label: equilibrium_crra_2
    L w_t \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1} = L\left (\frac{R_{t+1}}{\alpha} \right )^{1/(\alpha - 1)}
```

Combining with [](wage) yields
```{math}
:label: equilibrium_crra
    L (1-\alpha)(K_t / L)^{\alpha} \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1} = L\left (\frac{R_{t+1}}{\alpha} \right )^{1/(\alpha - 1)}
```



Here we cannot solve for the equilibrium price and quantity analytically.

Exercise 1 asks you to solve [](equilibrium_crra) to obtain the equilibrium quantity $R_{t+1}$ and the equilibrium price numerically, and verify it using the plot below.

Below we just show a plot of the equilibrium.

```{code-cell} ipython3
def aggregate_supply_capital_crra(R, model, K_prev):
    α, β, γ, L = model.α, model.β, model.u_params['γ'], model.L
    return L**(1-α) * (1-α) * K_prev**α / ( 1 + β**(-1/γ) * R**((γ-1)/γ) )
```

```{code-cell} ipython3
plot_ad_as(aggregate_capital_demand, aggregate_supply_capital_crra, m_crra, K_prev=50, E_star=None)  # John this is to be fixed.
```

Let's plot the aggregate supply with different values of utility parameter $\gamma$ and observe it's behaviour.

```{code-cell} ipython3
γ_vals = [0.1, 0.5, 1.5, 2.0]
K_prev = 50


fig, ax = plt.subplots()
R_vals = np.linspace(0.3, 1)

for γ in γ_vals:
    m = create_olg_model(u=partial(crra, γ=γ), u_params={'γ': γ})
    ax.plot(R_vals, aggregate_supply_capital_crra(R_vals, m, K_prev),
            label=r"$\gamma=$" + str(γ))

ax.set_xlabel("$R_{t+1}$")
ax.set_title("Aggregate Supply")
ax.legend()
plt.show()
```

When $\gamma <1$ the supply curve is downward sloping. When $\gamma > 1$ the supply curve is upward sloping.

TODO: Do we need to add some explanation?

+++

### Dynamics and steady state



Setting $k_t := K_t / L$ and using [](saving_crra).


Aggregate supply of capital [](aggregate_supply_capital_1_olg) becomes
```{math}
:label: supply_capital_crra_olg
    k_{t+1} = k^s(R_{t+1}) = \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]^{-1} w_t
```

[](interest_rate) becomes
```{math}
:label: interest_rate_2
    \alpha k^{\alpha - 1}_t = R_t
```



Combing with [](wage_2) and [](supply_capital_crra_olg) gives
```{math}
:label: law_of_motion_capital_crra
    k_{t+1} = \left [ 1 + \beta^{-1/\gamma} (\alpha k^{\alpha - 1}_{t+1})^{(\gamma-1)/\gamma} \right ]^{-1} (1-\alpha)(k_t)^{\alpha}
```



Note that with this equation and $k_t$ as given we cannot solve for $k_{t+1}$ by pencil and paper.

That is we don't have an analytical solution for the sample path $\{k_{t+1}\}$ now.

To solve for $k_{t+1}$ we need to turn to newton's method.

Suppose,
```{math}
:label: crra_newton_1
    f(k_{t+1}; k_t)=k_{t+1} \left [ 1 + \beta^{-1/\gamma} \left ( \alpha k^{\alpha-1}_{t+1} \right )^{(\gamma-1)/\gamma} \right ] - (1-\alpha) k^{\alpha}_t =0
```

If $k_t$ is given then $f(\cdot)$ is a function of unknown $k_{t+1}$.

Then we can use [scipy.optimize.newton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton) to solve the Equation $f(k_{t+1})=0$ for $k_{t+1}$.

First let define $f(\cdot)$.

```{code-cell} ipython3
def f(k_prime, k, model):
    α, β, γ = model.α, model.β, model.u_params['γ']
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

Unlike the log preference case now a steady state cannot be solved analytically.

To see this recall that, a steady state can be obtained by setting [](law_of_motion_capital_crra) to $k_{t+1} = k_t = k^*$, i.e.,
```{math}
:label: steady_state_crra
    k^* = \frac{(1-\alpha)(k^*)^{\alpha}}{  1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma}}
```



Similarly we can solve for $k^*$ using newton's method.

Suppose that
```{math}
:label: crra_newton_2
    g(k^*) = k^*  \left [ 1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} \right ] - (1-\alpha)(k^*)^{\alpha}
```

```{code-cell} ipython3
def g(k_star, model):
    α, β, γ = model.α, model.β, model.u_params['γ']
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

At this parameterization, $k^* \approx 0.314$.

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
    ax.plot(np.arange(ts_length), np.full(ts_length, k_star),
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


```{exercise}
:label: olg_ex1

Solve the equilibrium of the CRRA case numerically using [](equilibrium_crra).

Plot the equilibrium quantity and equilibrium price in the equilibrium plot with the CRRA utility (TODO label and refer the plot generated by the code).
```


```{solution-start} olg_ex1
:class: dropdown
```

To solve the equation we need to turn to Newton's method.

The function `find_Rstar` is used to find $R^*_{t+1}$ by finding
the zero of equation [](equilibrium_crra) using the helper
function `find_Rstar_newton` for a given value of $K_t$.

Similary, `find_Kstar` finds the equilibrium quantity $K^*_{t+1}$ using the value of $R^*_{t+1}$.

```{code-cell} ipython3
def find_Rstar_newton(x, K_prev, model):
    α, β, γ, L = model.α, model.β, model.u_params['γ'], model.L
    lhs = L * (1-α) * (K_prev / L)**α
    lhs /= (1 + β**(-1/γ) * x**((γ-1)/γ))
    rhs = L * (x / α)**(1/(α-1))
    return lhs - rhs
```

```{code-cell} ipython3
def find_Rstar(K_prev, model):
    return optimize.newton(find_Rstar_newton, 0.5, args=(K_prev, model))

def find_Kstar(R_star, model):
    return model.L * (R_star / model.α)**(1/(model.α-1))
```

The following function plots the equilibrium quantity and equilibrium price.

```{code-cell} ipython3
def plot_ks_rs(K_t_vals, model):
    n = len(K_t_vals)
    R_star = np.empty(n)
    K_star = np.empty(n)

    for i in range(n):
        R_star[i] = find_Rstar(K_t_vals[i], model)
        K_star[i] = find_Kstar(R_star[i], model)

    fig, ax = plt.subplots()

    ax.plot(K_t_vals, R_star, label="equilibrium price")
    ax.plot(K_t_vals, K_star, label="equilibrium quantity")

    ax.set_xlabel("$K_{t}$")
    ax.legend()
    plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 'Equilibrium price and quantity

      '
    name: equi_ps_q_crra
  image:
    alt: equi_ps_q_crra
    classes: shadow bg-primary
    width: 200px
---
K_t_vals = np.linspace(0.1, 50, 50)
m_crra = create_olg_model(u=crra, u_params={'γ': 0.5})
plot_ks_rs(K_t_vals, m_crra)
```

```{solution-end}
```

```{exercise}
:label: olg_ex2

Let's keep the model the same except for replacing the utility function $u(c)$ in {eq}`eq_crra`  with a nonlinear form $u(c)=c + c^{\theta}$.

Like what we did in the CRRA case we don't have an analytical solution.

Try to compute the time path capital $\{k_t\}$ in this case.
```

```{solution-start} olg_ex2
:class: dropdown
```

To get the time path capital $\{k_t\}$ first we need to solve the household's utility maximization problem for the optimal consumption and optimal saving.

With the quasilinear preference the Euler equation [](euler_2_olg) becomes

```{math}
:label: euler_quasilinear
    1 + \theta (w_t - s_t)^{\theta-1} = \beta R_{t+1} + \beta R^{\theta}_{t+1} \theta s_t^{\theta - 1}
```

Let $k_t := K_t / L$.

Since [](aggregate_supply_capital_log_olg), [](wage_2) and [](interest_rate_2) the Euler equation becomes

```{math}
:label: euler_quasilinear1
    1 + \theta ((1-\alpha)k^{\alpha}_t - k_{t+1})^{\theta-1} = \beta \alpha k^{\alpha - 1}_{t+1} + \beta \theta \alpha^{\theta} k^{\alpha \theta - 1}_{t+1}
```

Obviously $k_{t+1}$ cannot be solved by pencil and paper.

To solve for $k_{t+1}$ we need to turn to the newton's method.

Let's start by defining the utility function.

```{code-cell} ipython3
def u_quasilinear(c, θ=6):
    return c + c**θ
```

The function `find_k_next` is used to find $k_{t+1}$ by finding
the root of equation [](euler_quasilinear1) using the helper
function `solve_for_k_next` for a given value of $k_t$.

```{code-cell} ipython3
def solve_for_k_next(x, k_t, model):
    α, β, L, θ = model.α, model.β, model.L, model.u_params['θ']
    l = 1 + θ * ((1 - α) * k_t**α - x)**(θ - 1)
    r = β * α * x**(α - 1)
    r += β * θ * α**θ * x**(α * θ - 1)
    return l - r
```

```{code-cell} ipython3
def find_k_next(k_t, model):
    return optimize.newton(solve_for_k_next, k_t, args=(k_t, model))
```

```{code-cell} ipython3
def solve_for_k_star_q(x, model):
    α, β, L, θ = model.α, model.β, model.L, model.u_params['θ']
    l = 1 + θ * ((1 - α) * x**α - x)**(θ - 1)
    r = β * α * x**(α - 1)
    r += β * θ * α**θ * x**(α * θ - 1)
    return l - r

def find_k_star_q(model):
    return optimize.newton(solve_for_k_star_q, 0.3, args=(model,))
```

Let's simulate and plot the time path capital $\{k_t\}$.

```{code-cell} ipython3
def simulate_ts(k0_values, model, ts_length=10):
    k_star = find_k_star_q(model)

    print("k_star:", k_star)
    fig, ax = plt.subplots(figsize=(10, 5))

    ts = np.zeros(ts_length)

    # simulate and plot time series
    for x_init in k0_values:
        ts[0] = x_init
        for t in range(1, ts_length):
            ts[t] = find_k_next(ts[t-1], model)
        ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
                label=r'$k_0=%g$' %x_init)
    ax.plot(np.arange(ts_length), np.full(ts_length, k_star),
            alpha=0.6, linestyle='dashed', color='black', label=r'$k^*$')
    ax.legend(fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$k_t$', fontsize=14)

    plt.show()
```

```{code-cell} ipython3
k0_values = [0.2, 10, 50, 100]
m_quasilinear = create_olg_model(u=u_quasilinear, u_params={'θ': 6})
simulate_ts(k0_values, m_quasilinear)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
