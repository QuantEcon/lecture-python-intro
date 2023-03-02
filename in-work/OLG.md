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

In this lecture we study the overlapping generations model.

This model provide controllable alternative to infinite-horizon representative agent.
The dynamics of this model in some special cases are quite similar to Solow model.

## The Model

Let's assume that:

- Time is discrete and runs to infinity.
- Each individual lives for two time periods.
- Individuals born at time $t$ live for dates $t$ and $t + 1$.

Suppose that the utility functions take the familiar Constant Relative Risk Aversion (CRRA) form, given by:

```{math}
:label: eq_crra
    U(t) = \frac{c_1(t)^{1-\theta}-1}{1-\theta} +
           \beta \left( \frac{c_2(1+t)^{1-\theta}-1}{1-\theta} \right )
```
where,

- $\theta > 0$, and $\beta \in (0, 1)$ is the discount factor.
- $c_1(t)$: consumption of the individual born at $t$.
- $c_2(t)$: consumption of the individual at $t+1$.

For each integer $t \geq 0$, output $Y(t)$ in period $t$ is given by
$$
Y(t) = F(K(t), L(t))
$$
where $K(t)$ is capital, $L(t)$ is labor and $F$ is an aggregate
production function.

 
Without population growth, $L(t)$ equals some constant $L$.

Setting $k(t) := K(t) / L$, $f(k)=F(K, 1)$ and using homogeneity of degree one now yields:

$$
    1 + r(t) = R(t) = f'(k(t))
$$
The gross rate of return to saving is equal to the rental rate of capital.

And, the wage rate is given by,

$$
    w(t) = f(k(t)) - k (t)f'(k(t))
$$


Savings by an individual of generation $t$, $s(t)$, is determined as a
solution to:
 

$$
    \begin{aligned}
    \max_{c_1(t), c_2(t+1), s(t)} \ & u(c_1 (t)) + \beta u(c_2(t + 1)) \\
    \mbox{subject to } \ & c_1(t) + s(t) \le w(t) \\
                         & c_2(t + 1) \le R (t + 1)s(t)\\
    \end{aligned}
$$

Second constraint incorporates notion that individuals only spend
money on their own end of life consumption.

Solving for consumption and thus for savings,

$$
    s(t) = s(w(t), R(t + 1))
$$

Total savings in the economy will be equal to:

$$
    S(t) = s(t) L(t)
$$

We assume a closed economy, so domestic investment equals aggregate domestic
saving. Therefore, we have

$$
    K(t + 1) = L(t) s(w (t), R (t + 1))
$$

Setting $k(t) := K(t) / L(t)$, where $L(t + 1) = (1 + n) L(t),$ and using homogeneity of degree one now yields:

```{math}
:label: k_dyms
    k(t) = \frac{s(w (t), R (t + 1))}{1 + n}
```



A steady state is given by a solution to this equation such that
$k(t + 1) = k (t) = k^*$, i.e,

```{math}
:label: k_star
    k^* = \frac{s(f(k^*)-k^*f'(k^*), f'(k^*))}{1+n}
```
