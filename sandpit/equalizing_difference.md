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

### Equalizing Difference Model

This notebook presents a model of the college-high-school wage gap in which the
"time to build" a college graduate plays a key role.

The model is "incomplete" in the sense that it is just one "condition" in the form of one 
equation that would be part of set equations comprising all of the "equilibrium conditions" of   a more fully articulated model.

The condition featured in our model determies  a college, high-school wage ratio that equalizes the
present values of a high school worker and a college educated worker.

It is just one instance of a class of "equalizing difference" theories of relative wage rates, a class dating back at least to Adam Smith's **Wealth of Nations**.

+++

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


#### Tweaked Model 


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

```
