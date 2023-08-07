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

# Equalizing Difference Model

## Overview

This lecture presents a model of the college-high-school wage gap in which the
"time to build" a college graduate plays a key role.

The model is "incomplete" in the sense that it is just one "condition" in the form of a single equation that would be part of set equations comprising all  "equilibrium conditions" of   a more fully articulated model.

The condition featured in our model determines  a college, high-school wage ratio that equalizes the present values of a high school worker and a college educated worker.

The idea behind this condition is that lifetime earnings have to adjust to make someone indifferent between going to college and not going to college.

(The job of the "other equations" in a more complete model would be to fill in details about what adjusts to bring about this outcome.)

It is just one instance of an  "equalizing difference" theory of relative wage rates, a class of theories dating back at least to Adam Smith's **Wealth of Nations** {cite}`smith2010wealth`.  

For most of this lecture, the only mathematical tools that we'll use are from linear algebra, in particular, matrix multiplication and matrix inversion.

However, at the very end of the lecture, we'll use calculus just in case readers want to see how computing partial derivatives could let us present some findings more concisely.  

(And doing that will let us show off how good Python is at doing calculus!)

But if you don't know calculus, our tools from linear algebra are certainly enough.

As usual, we'll start by importing some Python modules.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## The indifference condition

The key idea is that the initial college wage premium has to adjust to make a representative worker indifferent between going to college and not going to college.

Let

 * $R > 1$ be the gross rate of return on a one-period bond

 * $t = 0, 1, 2, \ldots T$ denote the years that a person either works or attends college
 
 * $0$ denote the first period after high school that a person can go to work
 
 * $T$ denote the last period  that a person  works
 
 * $w_t^h$ be the wage at time $t$ of a high school graduate
 
 * $w_t^c$ be the wage at time $t$ of a college graduate
 
 * $\gamma_h > 1$ be the (gross) rate of growth of wages of a  high school graduate, so that
 $ w_t^h = w_0^h \gamma_h^t$
 
 * $\gamma_c > 1$ be the (gross) rate of growth of wages of a  college  graduate, so that
 $ w_t^c = w_0^c \gamma_c^t$

 * $D$ be the upfront monetary costs of going to college



If someone goes to work immediately after high school  and  works for the  $T+1$ years $t=0, 1, 2, \ldots, T$, she earns present value

$$
h_0 = \sum_{t=0}^T R^{-t} w_t^h = w_0^h \left[ \frac{1 - (R^{-1} \gamma_h)^{T+1} }{1 - R^{-1} \gamma_h } \right] \equiv w_0^h A_h 
$$

where 

$$
A_h = \left[ \frac{1 - (R^{-1} \gamma_h)^{T+1} }{1 - R^{-1} \gamma_h } \right].
$$

The present value $h_0$ is the "human wealth" at the beginning of time $0$ of someone who chooses not to attend college but instead to go to work immediately at the wage of a high school graduate.


If someone goes to college for the four years $t=0, 1, 2, 3$ during which she earns $0$, but then goes to work  immediately after college   and  works for the $T-3$ years $t=4, 5, \ldots ,T$, she earns present value

$$
c_0 = \sum_{t=4}^T R^{-t} w_t^c = w_0^c (R^{-1} \gamma_c)^4  \left[ \frac{1 - (R^{-1} \gamma_c)^{T-3} }{1 - R^{-1} \gamma_c } \right] \equiv w_0^c A_c
$$

where

$$
A_c = (R^{-1} \gamma_c)^4  \left[ \frac{1 - (R^{-1} \gamma_c)^{T-3} }{1 - R^{-1} \gamma_c } \right] 
$$ 

The present value $c_0$  is the "human wealth" at the beginning of time $0$ of someone who chooses to attend college for four years and then start to work at time $t=4$ at the wage of a college graduate.


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
"equalizing" equation that sets the present value not going to college equal to the present value of going go college:


$$
h_0 = c_0 - D
$$ 

or

$$ 
w_0^h A_h  = \phi w_0^h A_c - D .
$$ (eq:equalize)

This is the "indifference condition" that is at the heart of the model.

Solving equation {eq}`eq:equalize` for the college wage premium $\phi$ we obtain

$$
\phi  = \frac{A_h}{A_c} + \frac{D}{w_0^h A_c} .
$$ (eq:wagepremium)

In a **free college** special case $D =0$ so that the only cost of going to college is the forgone earnings from not working as a high school worker.  

In that case,

$$
\phi  = \frac{A_h}{A_c} . 
$$

Soon we'll write Python code to compute the gap and plot it as a function of its determinants.

But first we'll describe a possible alternative interpretation of our model.



## Reinterpreting the model: workers and entrepreneurs


We can add a parameter and reinterpret variables to get a model of entrepreneurs versus workers.

We now let $h$ be  the present value of a "worker".

We define the present value of an entrepreneur to be

$$
c_0 = \pi \sum_{t=4}^T R^{-t} w_t^c
$$

where $\pi \in (0,1) $ is  the probability that an entrepreneur's "project" succeeds.

For our model of workers and firms, we'll interpret $D$ as the cost of becoming an entrepreneur.  

This cost might include costs of hiring workers, office space, and lawyers. 



What we used to call the college, high school wage gap $\phi$ now becomes the ratio
of a successful entrepreneur's earnings to a worker's earnings.  

We'll find that as $\pi$ decreases, $\phi$ increases.  

Now let's write some Python code to compute $\phi$ and plot it as a function of some of its determinants.

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
        A_c = (1 - (γ_c/R)**(T-3)) / (1 - γ_c/R) * (γ_c/R)**4
        
        # tweaked model
        if π!=None:
            A_c = π*A_c 
        
        ϕ = A_h/A_c + D/(w_h0*A_c)
        return ϕ
```

We can build some functions to help do comparative statics using vectorization instead of loops.

For a given instance of the class, we want to compute $\phi$ when one parameter changes and others remain unchanged.

Let's do an example.

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
γ_h, γ_c = 1.01, 1.01
w_h0 = 1
D = 10

# create an instance
ex1 = equalizing_diff(R=R, T=T, γ_h=γ_h, γ_c=γ_c, w_h0=w_h0, D=D)
gap1 = ex1.compute_gap()

print(gap1)
```

Let's not charge for college and recompute $\phi$.

The initial college wage premium should go down.

```{code-cell} ipython3
# free college
ex2 = equalizing_diff(R, T, γ_h, γ_c, w_h0, D=0)
gap2 = ex2.compute_gap()
print(gap2)
```

Let us construct some graphs that show us how the initial college-high-school wage ratio $\phi$ would change if one of its determinants were to change. 

Let's start with the gross interest rate $R$.

```{code-cell} ipython3
R_arr = np.linspace(1, 1.2, 50)
plt.plot(R_arr, φ_R(ex1, R_arr))
plt.xlabel(r'$R$')
plt.ylabel(r'wage gap')
plt.show()
```

Evidently, the initial wage ratio $\phi$ must rise to compensate a prospective high school student for **waiting** to start receiving income -- remember that while she is earning nothing in years $t=0, 1, 2, 3$, the high school worker is earning a salary.

Not let's study what happens to the initial wage ratio $\phi$ if the rate of growth of college wages rises, holding constant other 
determinants of $\phi$.

```{code-cell} ipython3
γc_arr = np.linspace(1, 1.2, 50)
plt.plot(γc_arr, φ_γc(ex1, γc_arr))
plt.xlabel(r'$\gamma_c$')
plt.ylabel(r'wage gap')
plt.show()
```

Notice how  the intitial wage gap falls when the rate of growth $\gamma_c$ of college wages rises.  

It falls to "equalize" the present values of the two types of career, one as a high school worker, the other as a college worker.

Can you guess what happens to the initial wage ratio $\phi$ when next we vary the rate of growth of high school wages, holding all other determinants of $\phi$ constant?  

The following graph shows what happens.

```{code-cell} ipython3
γh_arr = np.linspace(1, 1.1, 50)
plt.plot(γh_arr, φ_γh(ex1, γh_arr))
plt.xlabel(r'$\gamma_h$')
plt.ylabel(r'wage gap')
plt.show()
```

## Entrepreneur-worker interpretation

Now let's adopt the entrepreneur-worker interpretation of our model.

If the probability that a new business succeeds is $.2$, let's compute the initial wage premium for successful entrepreneurs.

```{code-cell} ipython3
# a model of enterpreneur
ex3 = equalizing_diff(R, T, γ_h, γ_c, w_h0, π=0.2)
gap3 = ex3.compute_gap()

print(gap3)
```

Now let's study how the initial wage premium for successful entrepreneurs depend on the success probability.

```{code-cell} ipython3
π_arr = np.linspace(0.2, 1, 50)
plt.plot(π_arr, φ_π(ex3, π_arr))
plt.ylabel(r'wage gap')
plt.xlabel(r'$\pi$')
plt.show()
```

Does the graph make sense to you?

## An application of SymPy

So far, we have used only linear algebra and it has been a good enough tool for us to  figure out how our model works.

However, someone who knows calculus might ask "Instead of plotting those graphs, why didn't you just take partial derivatives?"

We'll briefly do just that,  yes, the questioner is correct and that partial derivatives are indeed a good tool for discovering the "comparative statics" properities of our model.

A reader who doesn't know calculus could read no further and feel confident that applying linear algebra has taught us the main properties of the model.

But for a reader interested in how we can get Python to do all the hard work involved in computing partial derivatives, we'll say a few things about that now.  

We'll use the Python module SymPy to compute partial derivatives of $\phi$ with respect to the parameters that determine it.

Let's import key functions from SymPy.

```{code-cell} ipython3
from sympy import Symbol, Lambda, symbols, refine, \
                  Sum, simplify, Eq, solve, Lambda, lambdify

# Define the symbols
R, w_h0, w_c0, γ_c, γ_h, ϕ, D, t, T = symbols(
    'R w^h_0 w^c_0 gamma_c gamma_h phi D t T', positive=True)

refine(γ_c, γ_c>1)
refine(γ_h, γ_h>1)
refine(R, R>1)

# Define the wage for college 
# and high school graduates at time t
w_ct = w_c0 * γ_c**t
w_ht = w_h0 * γ_h**t
```

```{code-cell} ipython3
w_ct
```

```{code-cell} ipython3
w_ht
```

### Defining the Present Value Equations

The present value of the earnings after going to college is

$$
PV_{\text{{college}}} = \sum_{t=4}^T R^{-t} w_t^c
$$

It is the sum of the discounted earnings from the first year of graduation to the last year of work assuming the degree is obtained in the fourth year and no salary is earned while in the college.

The present value of the earnings from going to work after high school is

$$
PV_{\text{{highschool}}} = \sum_{t=0}^T R^{-t} w_t^h
$$

It is the sum of the discounted earnings from the first year after high school to the last year of work.

```{code-cell} ipython3
PV_college = Sum(R**-t * w_ct, (t, 4, T))
PV_college
```

```{code-cell} ipython3
PV_highschool = Sum(R**-t * w_ht, (t, 0, T))
PV_highschool
```

We can evaluate the sum using the `doit` method and derive $A_h$ and $A_c$ from present values

```{code-cell} ipython3
A_h = simplify(PV_highschool.doit() / w_h0)
A_h
```

```{code-cell} ipython3
A_c = simplify(PV_college.doit() / w_c0)
A_c
```

```{code-cell} ipython3
A_h = simplify(A_h.args[1][0])
A_c = simplify(A_c.args[1][0])
```

```{code-cell} ipython3
A_h
```

```{code-cell} ipython3
A_c
```

Now, we define $\phi$

```{code-cell} ipython3
ϕ = A_h/A_c + D/(w_h0*A_c)
ϕ
```

We begin our calculation of derivatives by setting the following default parameter values

```{code-cell} ipython3
R_value = 1.05
T_value = 40
γ_h_value, γ_c_value = 1.01, 1.01
w_h0_value = 1
D_value = 10

symbol_subs = {D: D_value,
               γ_h: γ_h_value,
               γ_c: γ_c_value,
               R: R_value,
               T: T_value,
               w_h0: w_h0_value}
```

```{code-cell} ipython3
ϕ_D, ϕ_T, ϕ_γ_h, ϕ_γ_c, ϕ_R = (ϕ.diff(D), ϕ.diff(T), 
                               ϕ.diff(γ_h), ϕ.diff(γ_c), 
                               ϕ.diff(R))
```

This anchors the idea to show how $\phi$ changes with different tuition fees $D$ and time $T$?

```{code-cell} ipython3
grid = np.meshgrid(np.arange(10, 60, 1), 
                   np.arange(0, 60, 1))

ϕ_TD = ϕ.subs({γ_h: γ_h_value,
               γ_c: γ_c_value,
               R: R_value,
               w_h0: w_h0_value})
```

```{code-cell} ipython3
ϕ_TD_lambda = lambdify([T, D], ϕ_TD)
```

```{code-cell} ipython3
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_box_aspect(aspect=None, zoom=0.85)

ax.plot_surface(grid[0], 
                grid[1],
                ϕ_TD_lambda(grid[0], grid[1]))
                
ax.set_xlabel('T')
ax.set_ylabel('D')
ax.set_zlabel(r'$\phi$')
plt.show()
```

Now let's compute $\frac{\partial \phi}{\partial D}$ and then evaluate it at the default values

```{code-cell} ipython3
ϕ_D
```

```{code-cell} ipython3
ϕ_D.subs(symbol_subs)
```

Thus, as with our graph above, we find that raising $R$ increases the initial college wage premium $\phi$.

Compute $\frac{\partial \phi}{\partial T}$ and evaluate it a default parameters

```{code-cell} ipython3
ϕ_T
```

```{code-cell} ipython3
ϕ_T.subs(symbol_subs)
```

We find that raising $T$ decreases the initial college wage premium $\phi$. 

This is because college graduates now have longer career lengths to "pay off" the time and other costs they paid to go to college

Let's compute $\frac{\partial \phi}{\partial γ_h}$ and evaluate it at default parameters.

```{code-cell} ipython3
ϕ_γ_h
```

```{code-cell} ipython3
ϕ_γ_h.subs(symbol_subs)
```

We find that raising $\gamma_h$ increases the initial college wage premium $\phi$, as we did with our graphical analysis earlier.

Compute $\frac{\partial \phi}{\partial γ_c}$ and evaluate it numerically at default parameter values

```{code-cell} ipython3
ϕ_γ_c
```

```{code-cell} ipython3
ϕ_γ_c.subs(symbol_subs)
```

We find that raising $\gamma_c$ decreases the initial college wage premium $\phi$, as we did with our graphical analysis earlier

Let's compute $\frac{\partial \phi}{\partial R}$ and evaluate it numerically at default parameter values

```{code-cell} ipython3
ϕ_R
```

```{code-cell} ipython3
ϕ_R.subs(symbol_subs)
```

We find that raising the gross interest rate $R$ increases the initial college wage premium $\phi$, as we did with our graphical analysis earlier.
