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

+++ {"tags": []}

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

Here $K_t$ is capital, $L_t$ is labor, $F$ is   $\alpha$ is the output elasticity of capital in the Cobb-Douglas production function.

Without population growth, $L_t$ equals some constant $L$.

### Prices

Setting $k_t := K_t / L_t$, $f(k)=F(K, 1)$ and using homogeneity of degree one now yields:

$$
    1 + r_t = R_t = f'(k_t) = \alpha k^{\alpha-1}_t
$$

Here the gross rate of return to saving $1 + r_t$ is equal to the rental rate of capital $R_t$.

The wage rate is given by

$$
    w_t = f(k_t) - k_t f'(k_t) = (1-\alpha) k^{\alpha}_t
$$



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
money on their own end of life consumption.

Substituting $s_t$ we get from the first constraint into the second constraint we get $c^2_{t+1}$ in terms of $c^1_t$. 

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
:label: k_dyms
    k_t = \frac{s(w_t, R_{t+1})}{1 + n} = \frac{w_t}{(1+n) \left [ 1 + \beta^{-1/\gamma} R_{t+1}^{(\gamma-1)/\gamma} \right ]}
```



A steady state is given by a solution to this equation such that
$k_{t+1} = k_t = k^*$, i.e,

```{math}
:label: k_star
    k^* = \frac{s(f(k^*)-k^*f'(k^*), f'(k^*))}{1+n} = \frac{(1-\alpha)(k^*)^{\alpha}}{(1+n) \left [ 1 + \beta^{-1/\gamma} (\alpha (k^*)^{\alpha-1})^{(\gamma-1)/\gamma} \right ]}
```


## Exercises

```{exercise}
:label: olg_ex1

Replace the utility function $u(c)$ in equation {ref}`eq_crra`  with a quasilinear form $u(c)=c + c^{\alpha}$.

Now we don't have an analytical solution. 

Try to compute the time path capital $\{k_t\}$ in this case.
```

