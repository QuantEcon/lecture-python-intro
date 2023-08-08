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
from sympy import Symbol, Lambda, symbols, refine, \
                  Sum, simplify, Eq, solve, Lambda,\
                  lambdify, And
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

First, we define these symbols in SymPy

```{code-cell} ipython3
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

If someone goes to work immediately after high school  and  works for the  $T+1$ years $t=0, 1, 2, \ldots, T$, she earns present value

$$
h_0 = \sum_{t=0}^T R^{-t} w_t^h = w_0^h \left[ \frac{1 - (R^{-1} \gamma_h)^{T+1} }{1 - R^{-1} \gamma_h } \right] \equiv w_0^h A_h 
$$

```{code-cell} ipython3
h_0 = Sum(R**-t * w_ht, (t, 0, T))
h_0
```

where 

$$
A_h = \left[ \frac{1 - (R^{-1} \gamma_h)^{T+1} }{1 - R^{-1} \gamma_h } \right].
$$

```{code-cell} ipython3
A_h = simplify(h_0.doit() / w_h0)
A_h = simplify(A_h.args[1][0])
A_h
```

The present value $h_0$ is the "human wealth" at the beginning of time $0$ of someone who chooses not to attend college but instead to go to work immediately at the wage of a high school graduate.

If someone goes to college for the four years $t=0, 1, 2, 3$ during which she earns $0$, but then goes to work  immediately after college   and  works for the $T-3$ years $t=4, 5, \ldots ,T$, she earns present value

$$
c_0 = \sum_{t=4}^T R^{-t} w_t^c = w_0^c (R^{-1} \gamma_c)^4  \left[ \frac{1 - (R^{-1} \gamma_c)^{T-3} }{1 - R^{-1} \gamma_c } \right] \equiv w_0^c A_c
$$

```{code-cell} ipython3
c_0 = Sum(R**-t * w_ct, (t, 4, T))
c_0
```

where

$$
A_c = (R^{-1} \gamma_c)^4  \left[ \frac{1 - (R^{-1} \gamma_c)^{T-3} }{1 - R^{-1} \gamma_c } \right] 
$$

```{code-cell} ipython3
A_c = simplify(c_0.doit() / w_c0)
A_c = simplify(A_c.args[1][0])
A_c
```

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

We suppose that $R, \gamma_h, \gamma_c, T$ and also $w_0^h$ are fixed parameters. 

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

```{code-cell} ipython3
ϕ = A_h/A_c + D/(w_h0*A_c)
ϕ
```

Now we can compute $\phi$ using default values below

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

ϕ.subs(symbol_subs)
```

In a **free college** special case $D =0$ so that the only cost of going to college is the forgone earnings from not working as a high school worker.  

In that case,

$$
\phi  = \frac{A_h}{A_c} . 
$$


Let's not charge for college and recompute $\phi$.

The initial college wage premium should go down as $\phi$ goes up

```{code-cell} ipython3
symbol_subs[D] = 0

ϕ.subs(symbol_subs)
```

Let's construct some graphs that show us how the initial high-school-college wage ratio $\phi$ would change if one of its determinants were to change. 

Let's start with the gross interest rate $R$.

```{code-cell} ipython3
# Substitute default values into formula for ϕ
symbol_subs[R] = R
ϕ_R = ϕ.subs(symbol_subs)

# Lambdify ϕ to take arrays as inputs
ϕ_R_lambda = lambdify(R, ϕ_R)
R_arr = np.linspace(1, 1.2, 50)
plt.plot(R_arr, ϕ_R_lambda(R_arr))
plt.xlabel(r'$R$')
plt.ylabel(r'wage gap ($\phi$)')
plt.show()
```

Evidently, the initial wage ratio $\phi$ must rise to compensate a prospective college student for **waiting** to start receiving income -- remember that while she is earning nothing in years $t=0, 1, 2, 3$, the high school worker is earning a salary.


Let's introduce time horizon $T$ into the plot

```{code-cell} ipython3
symbol_subs[T] = T

ϕ_TR = ϕ.subs(symbol_subs)

grid = np.meshgrid(np.linspace(10, 60, 100),
                   np.linspace(1, 1.2, 50))
```

```{code-cell} ipython3
ϕ_TR_lambda = lambdify([T, R], ϕ_TR)
```

```{code-cell} ipython3
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_box_aspect(aspect=None, zoom=0.8)

ax.plot_surface(grid[0], 
                grid[1],
                ϕ_TR_lambda(grid[0], grid[1]))
                
ax.set_xlabel('T')
ax.set_ylabel('R')
ax.set_zlabel(r'wage gap ($\phi$)')
plt.show()
```

We find wage ratio $\phi$ decreases as the expected years of work $T$ increases across gross interest rate $R$.


Now let's study what happens to the initial wage ratio $\phi$ if the rates of growth of wages rises ($\gamma_c$ and $\gamma_h$) change, holding constant other 
determinants of $\phi$.

We substitute the default values into the formula for $\phi$ and then plot $\phi$ as a function of $\gamma_c$ and $\gamma_h$.

```{code-cell} ipython3
grid = np.meshgrid(np.linspace(1, 1.2, 50), 
                   np.linspace(1, 1.2, 50))

symbol_subs_γ = {D: D_value,
                 R: R_value,
                 T: T_value,
                 w_h0: w_h0_value}

# Substitute default values into formula for ϕ
ϕ_γ = ϕ.subs(symbol_subs_γ)

# Lambdify ϕ to take arrays as inputs
ϕ_γ_lambda = lambdify([γ_c, γ_h], ϕ_γ)
```

Notice how  the intitial wage gap falls when the rate of growth $\gamma_c$ of college wages rises

```{code-cell} ipython3
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_box_aspect(aspect=None, zoom=0.85)

ax.plot_surface(grid[0], 
                grid[1],
                ϕ_γ_lambda(grid[0], grid[1]))
                
ax.set_xlabel(r'$\gamma_c$')
ax.set_ylabel(r'$\gamma_h$')
ax.set_zlabel(r'wage gap ($\phi$)')
plt.show()
```

It falls to "equalize" the present values of the two types of career, one as a high school worker, the other as a college worker.

Note the pattern when we vary the rate of growth of high school wages $\gamma_h$, holding all other determinants of $\phi$ constant.

What difference did you observe?

(worker_entrepreneur)=
## Reinterpreting the model: workers and entrepreneurs

We can add a parameter and reinterpret variables to get a model of entrepreneurs versus workers.

We now let $h$ be the present value of a "worker".

We define the present value of an entrepreneur to be

$$
c_0 = \pi \sum_{t=4}^T R^{-t} w_t^c
$$

where $\pi \in (0,1) $ is  the probability that an entrepreneur's "project" succeeds.


As shown above, we get

```{code-cell} ipython3
π, c_0 = symbols('pi c_0')

refine(π, And(π > 0, π < 1))

A_c = π*A_c
A_c
```

```{code-cell} ipython3
ϕ = A_h/A_c + D/(w_h0*A_c)
ϕ
```

For our model of workers and firms, we'll interpret $D$ as the cost of becoming an entrepreneur.  

This cost might include costs of hiring workers, office space, and lawyers. 

What we used to call the college, high school wage gap $\phi$ now becomes the ratio
of a successful entrepreneur's earnings to a worker's earnings.  

Now let's study how the initial wage premium for successful entrepreneurs depend on the success probability.

We'll find that as $\pi$ increases, $\phi$ decreases.

```{code-cell} ipython3
π_arr = np.linspace(0.2, 1, 50)

symbol_subs = {D: D_value,
               γ_h: γ_h_value,
               γ_c: γ_c_value,
               R: R_value,
               T: T_value,
               w_h0: w_h0_value}

print('default values:', symbol_subs)

# Substitute default values into formula for ϕ
ϕ_π = ϕ.subs(symbol_subs)

ϕ_π_lambda = lambdify(π, ϕ_π.subs(symbol_subs))

plt.plot(π_arr, ϕ_π_lambda(π_arr))
plt.ylabel(r'wage gap ($\phi$)')
plt.xlabel(r'$\pi$')
plt.show()
```

When $\pi$ is small, the risk of becoming an entrepreneur is high, so only a few people become entrepreneurs.

```{code-cell} ipython3
ϕ_π_lambda(0.2)
```

This gives a higher wage gap $\phi$.


## An application of Calculus

So far, we have used only linear algebra and it has been a good enough tool for us to  figure out how our model works.

However, someone who knows calculus might ask "Instead of plotting those graphs, why didn't you just take partial derivatives?"

We'll briefly do just that,  yes, the questioner is correct and that partial derivatives are indeed a good tool for discovering the "comparative statics" properities of our model.

A reader who doesn't know calculus could read no further and feel confident that applying linear algebra has taught us the main properties of the model.

But for a reader interested in how we can get Python to do all the hard work involved in computing partial derivatives, we'll say a few things about that now.  

We'll compute partial derivatives of $\phi$ with respect to the parameters that determine it.

```{code-cell} ipython3
ϕ_D, ϕ_T, ϕ_γ_h, ϕ_γ_c, ϕ_R, ϕ_π = (ϕ.diff(D), ϕ.diff(T), 
                                    ϕ.diff(γ_h), ϕ.diff(γ_c), 
                                    ϕ.diff(R), ϕ.diff(π))

symbol_subs[π] = 0.2
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

```{code-cell} ipython3
ϕ_π
```

```{code-cell} ipython3
ϕ_π.subs(symbol_subs)
```

We find that raising the gross interest rate $\pi$ decreases the initial college wage premium $\phi$, as we did with our graphical analysis earlier.


## Exercises

```{exercise-start}
:label: edm_ex1
```
In this exercise, replicate the result in section {ref}`worker_entrepreneur` using NumPy and Matplotlib. 

Compare your solution to SymPy's solution.

```{exercise-end}
```

```{solution-start} edm_ex1
:class: dropdown
```

Here is one solution

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

We vectorize the function to take a vector of $\pi$'s as an input

```{code-cell} ipython3
# ϕ_π
def ϕ_π(mc, π_new):
    mc_new = equalizing_diff(mc.R, mc.T, mc.γ_h, mc.γ_c, mc.w_h0, mc.D, π_new)
    return mc_new.compute_gap()

ϕ_π = np.vectorize(ϕ_π)
```

Let's compute the gap for the default parameters

```{code-cell} ipython3
# set benchmark parameters
R = 1.05
T = 40
γ_h, γ_c = 1.01, 1.01
w_h0 = 1
D = 10
π = 0.2

# create an instance
edm = equalizing_diff(R=R, T=T, γ_h=γ_h, γ_c=γ_c, w_h0=w_h0, D=D, π=0.2)

# compute the gap
edm.compute_gap()
```

Now let's generate the plot of the wage gap as a function of $\pi$.

```{code-cell} ipython3
π_arr = np.linspace(0.2, 1, 50)
plt.plot(π_arr, φ_π(edm, π_arr))
plt.ylabel(r'wage gap')
plt.xlabel(r'$\pi$')
plt.show()
```

```{solution-end}
```
