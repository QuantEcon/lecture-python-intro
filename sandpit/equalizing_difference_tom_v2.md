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

+++ {"user_expressions": []}

### Equalizing Difference Model

This notebook presents a model of the college-high-school wage gap in which the
"time to build" a college graduate plays a key role.

The model is "incomplete" in the sense that it is just one "condition" in the form of one 
equation that would be part of set equations comprising all of the "equilibrium conditions" of   a more fully articulated model.

The condition featured in our model determies  a college, high-school wage ratio that equalizes the
present values of a high school worker and a college educated worker.

It is just one instance of an  "equalizing difference" theory of relative wage rates, a class dating back at least to Adam Smith's **Wealth of Nations**.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

+++ {"user_expressions": []}

Let

 * $R > 0$ be the gross rate of return on a one-period bond
 
 * $0$ denote the first  period after high school that a person can go to work
 
 * $T$ denote the last period a person can work
 
 * $w_t^h$ be the wage at time $t$ of a high school graduate
 
 * $w_t^c$ be the wage at time $t$ of a college graduate
 
 * $\gamma_h > 0$ be the (gross) rate of growth of wages of a  high school graduate, so that
 $ w_t^h = w_0^h \gamma_h^t$
 
 * $\gamma_c > 0$ be the (gross) rate of growth of wages of a  college  graduate, so that
 $ w_t^c = w_0^c \gamma_c^t$



If someone goes to work immediately after high school  and  work for $T+1$ years, she earns present value
$$
h_0 = \sum_{t=0}^T R^{-t} w_t^h = w_0^h \left[ \frac{1 - (R^{-1} \gamma_h)^{T+1} }{1 - R^{-1} \gamma_h } \right] \equiv w_0^h A_h 
$$




If someone goes to college for four years during which she earns $0$, but then goes to work  immediately after college   and  work for $T-3$ years, she earns present value

$$
c_0 = \sum_{t=4}^T R^{-t} w_t^c = w_0^c (R^{-1} \gamma_c)^3  \left[ \frac{1 - (R^{-1} \gamma_c)^{T-3} }{1 - R^{-1} \gamma_c } \right] \equiv w_0^c A_c
$$


Assume that college tuition plus four years of room and board paid for up front costs $D$.

So net of monetary cost of college, the present value of attending college as of the first period after high school is

$$ 
c_0 - D
$$

We now formulate a pure **equalizing difference** model of the initial college-high school wage gap $\phi$ defined by 

Let

$$
w_0^c = \phi w_0^h 
$$

We suppose that $R, \gamma_h, \gamma_c, T$ and also $w_0^h$  are fixed parameters. 

We start by noting that the pure equalizing difference model asserts that the college-high-school wage gap $\phi$ solves


$$
h_0 = c_0 - D
$$ 

or

$$ 
w_0^h A_h  = \phi w_0^h A_c - D
$$

or

$$
\phi  = \frac{A_h}{A_c} + \frac{D}{w_0^h A_c}
$$ 

In a **free college** special case $D =0$ so that the only cost of going to college is the forgone earnings from not working as a high school worker.  

In that case,

$$
\phi  = \frac{A_h}{A_c} . 
$$


#### Tweaked Model: Workers and Entrepreneurs


We can add a parameter and reinterpret variables to get a model of entrepreuneurs versus workers.

We now let $h$ be  the present value of a "worker".

We define the present value of an entrepreneur to be

$$
c_0 = \pi \sum_{t=4}^T R^{-t} w_t^c
$$

where $\pi \in (0,1) $ is  the probability that an entrepreneur's "project" succeeds.

We set $D =0$.

What we used to call the college, high school wage gap $\phi$ now becomes the ratio
of a successful entreneur's earnings to a worker's earnings.  

We'll find that as $\pi$ decreases, $\phi$ increases.  

We can have some fun providing some example calculations that tweak various parameters,
prominently including $\gamma_h, \gamma_c, R$.

```{code-cell} ipython3
class equalizing_diff:
    """
    A class of the equalizing difference model
    """
    
    def __init__(self, R, T, γ_h, γ_c, w_h0, D=0, π=None):
        # one switches to the weak model by setting π
        self.R, self.γ_h, self.γ_c, self.w_h0, self.D = R, γ_h, γ_c, w_h0, D
        self.T, self.π = T, π
    
    def compute_gap(self):
        R, γ_h, γ_c, w_h0, D = self.R, self.γ_h, self.γ_c, self.w_h0, self.D
        T, π = self.T, self.π
        
        A_h = (1 - (γ_h/R)**(T+1)) / (1 - γ_h/R)
        A_c = (1 - (γ_c/R)**(T-3)) / (1 - γ_c/R) * (γ_c/R)**3
        
        # tweaked model
        if π!=None:
            A_c = π*A_c 
        
        ϕ = A_h/A_c + D/(w_h0*A_c)
        return ϕ
```

+++ {"user_expressions": []}

We can build some functions to help do comparative statics using vectorization instead of loops.

For a given instance of the class, we want to compute $\phi$ when one parameter changes and others remain unchanged.

```{code-cell} ipython3
# ϕ_R
def ϕ_R(mc, R_new):
    mc_new = equalizing_diff(R_new, mc.T, mc.γ_h, mc.γ_c, mc.w_h0, mc.D, mc.π)
    return mc_new.compute_gap()

ϕ_R = np.vectorize(ϕ_R)

# ϕ_γh
def ϕ_γh(mc, γh_new):
    mc_new = equalizing_diff(mc.R, mc.T, γh_new, mc.γ_c, mc.w_h0, mc.D, mc.π)
    return mc_new.compute_gap()

ϕ_γh = np.vectorize(ϕ_γh)

# ϕ_γc
def ϕ_γc(mc, γc_new):
    mc_new = equalizing_diff(mc.R, mc.T, mc.γ_h, γc_new, mc.w_h0, mc.D, mc.π)
    return mc_new.compute_gap()

ϕ_γc = np.vectorize(ϕ_γc)

# ϕ_π
def ϕ_π(mc, π_new):
    mc_new = equalizing_diff(mc.R, mc.T, mc.γ_h, mc.γ_c, mc.w_h0, mc.D, π_new)
    return mc_new.compute_gap()

ϕ_π = np.vectorize(ϕ_π)
```

```{code-cell} ipython3
# set benchmark parameters
R = 1.05
T = 40
γ_h, γ_c = 1.01, 1.02
w_h0 = 1
D = 2

# create an instance
ex1 = equalizing_diff(R=R, T=T, γ_h=γ_h, γ_c=γ_c, w_h0=w_h0, D=D)
gap1 = ex1.compute_gap()

print(gap1)
```

+++ {"user_expressions": []}

Let us plot $\phi$ against some parameters.

```{code-cell} ipython3
R_arr = np.linspace(1, 1.2, 50)
plt.plot(R_arr, φ_R(ex1, R_arr))
plt.xlabel(r'$R$')
plt.ylabel(r'wage gap')
plt.show()
```

```{code-cell} ipython3
γc_arr = np.linspace(1, 1.2, 50)
plt.plot(γc_arr, φ_γc(ex1, γc_arr))
plt.xlabel(r'$\gamma_c$')
plt.ylabel(r'wage gap')
plt.show()
```

```{code-cell} ipython3
γh_arr = np.linspace(1, 1.1, 50)
plt.plot(γh_arr, φ_γh(ex1, γh_arr))
plt.xlabel(r'$\gamma_h$')
plt.ylabel(r'wage gap')
plt.show()
```

```{code-cell} ipython3
# free college
ex2 = equalizing_diff(R, T, γ_h, γ_c, w_h0, D=0)
gap2 = ex2.compute_gap()

print(gap2)
```

```{code-cell} ipython3
# a model of enterpreneur
ex3 = equalizing_diff(R, T, γ_h, γ_c, w_h0, π=0.8)
gap3 = ex3.compute_gap()

print(gap3)
```

```{code-cell} ipython3
π_arr = np.linspace(0.2, 1, 50)
plt.plot(π_arr, φ_π(ex3, π_arr))
plt.ylabel(r'wage gap')
plt.xlabel(r'$\pi$')
plt.show()
```

```{code-cell} ipython3

```
