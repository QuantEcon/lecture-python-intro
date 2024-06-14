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

# Computing Square Roots


## Introduction

This lectures provides an  example of **invariant subspace** methods for analyzing linear difference equations. 

These methods are applied throughout applied economic dynamics, for example, in this QuantEcon lecture {doc}`money financed government deficits and inflation <money_inflation>`

Our approach in this lecture is to illustrate the method with an ancient example, one that ancient Greek mathematicians used to compute square roots of positive integers.

An integer is called a **perfect square** if its square root is also an integer.

An ordered sequence of  perfect squares starts with 

$$
4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, \ldots 
$$

If an integer is not a perfect square, then its square root is an irrational number -- i.e., it cannot be expressed as a ratio of two integers, and its decimal expansion is indefinite.

The ancient Greeks invented an algorithm to compute square roots of integers, including integers that are not perfect squares.

Their method involved

 * computing a particular sequence of integers $\{y_t\}_{t=0}^\infty$
 
 * computing $\lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) = \bar r$
 
 * deducing the desired square root from $\bar r$
 
In this lecture, we'll describe this method.

We'll also use invariant subspaces to describe variations on this method that are faster.

## Primer on second order linear difference equation

Consider the following second-order linear difference equation

$$
y_t = a_1 y_{t-1} + a_2 y_{t-2}, \quad t \geq 0
$$ (eq:2diff1)

where $(y_{-1},  y_{-2})$ is a pair of given initial conditions.  

We want to find expressions for $y_t, t \geq 0$ as functions of the initial conditions  $(y_{-1},  y_{-2})$:

$$ 
y_t = g((y_{-1},  y_{-2});t), \quad t \geq 0
$$ (eq:2diff2)

We call such a function $g$ a **solution** of the difference equation {eq}`eq:2diff1`.

One way to discover a solution is to use a guess and verify method.

We shall begin by considering a special initial pair of initial  conditions
that satisfy

$$
y_{-1} = \delta y_{-2}
$$ (eq:2diff3)

where $\delta$ is a scalar to be determined.

For initial condition that satisfy {eq}`eq:2diff3`
equation {eq}`eq:2diff1` impllies that

$$
y_0 = \left(a_1 + \frac{a_2}{\delta}\right) y_{-1}
$$ (eq:2diff4)

We want 

$$
\left(a_1 + \frac{a_2}{\delta}\right) = \delta
$$ (eq:2diff5)

which we can rewrite as the **characteristic equation** 

$$
\delta^2 - a_1 \delta - a_2 = 0
$$ (eq:2diff6)

Applying the quadratic formula to solve for the roots of {eq}`eq:2diff6` we find that

$$
\delta = \frac{ a_1 \pm \sqrt{a_1^2 + 4 a_2}}{2}
$$ (eq:2diff7)

For either of the two $\delta$'s that satisfy equation {eq}`eq:2diff7`, 
a solution of difference equation {eq}`eq:2diff1` is 

$$
y_t = \delta^t y_0 , \forall t \geq 0
$$ (eq:2diff8)

and $y_0 = a_1 y_{-1} + a_2 y_{-2}$

The **general** solution of difference equation {eq}`eq:2diff1` takes the form

$$
y_t = \eta_1 \delta_1^t + \eta_2 \delta_2^t
$$ (eq:2diff9)

where $\delta_1, \delta_2$ are the two solutions {eq}`eq:2diff7` of the characteristic equation {eq}`eq:2diff6`, and  $\eta_1, \eta_2$ are two constants chosen to satisfy
    
$$ 
    \begin{bmatrix} y_{-1} \cr y_{-2} \end{bmatrix} = \begin{bmatrix} \delta_1^{-1}  & \delta_2^{-1} \cr \delta_1^{-2} & \delta_2^{-2} \end{bmatrix} \begin{bmatrix} \eta_1 \cr \eta_2 \end{bmatrix} 
$$ (eq:2diff10)

or

$$
\begin{bmatrix} \eta_1 \cr \eta_2 \end{bmatrix} = \begin{bmatrix} \delta_1^{-1}  & \delta_2^{-1} \cr \delta_1^{-2} & \delta_2^{-2} \end{bmatrix}^{-1} \begin{bmatrix} y_{-1} \cr y_{-2} \end{bmatrix}
$$ (eq:2diff11)

Sometimes we are free to choose the initial conditions $(y_{-1}, y_{-2})$, in which case we 
use system {eq}`eq:2diff10` to find the associated $(\eta_1, \eta_2)$.

If we choose $(y_{-1}, y_{-2})$ to set $(\eta_1, \eta_2) = (1, 0)$, then $y_t = \delta_1^t$ for all $t \geq 0$.


If we choose $(y_{-1}, y_{-2})$ to set $(\eta_1, \eta_2) = (0, 1)$, then $y_t = \delta_2^t$ for all $t \geq 0$.


## Setup

Let $\sigma$ be a positive  integer greater than $1$

So $\sigma \in {\mathcal I} \equiv  \{2, 3, \ldots \}$ 

We want an algorithm to compute the square root of $\sigma \in {\mathcal I}$.

If $\sqrt{\sigma} \in {\mathcal I}$, $\sigma $ is said to be a **perfect square**.

If $\sqrt{\sigma} \not\in {\mathcal I}$, it turns out that it is irrational.

Ancient Greeks used a recursive algorithm to compute square roots of integers that are not perfect squares. 

The algorithm iterates on a  second order  linear  difference equation in the sequence $\{y_t\}_{t=0}^\infty$:

$$
y_{t} = 2 y_{t-1} - (1 - \sigma) y_{t-2}, \quad t \geq 0
$$ (eq:second_order)

together with a pair of integers that are  initial conditions for   $y_{-1}, y_{-2}$.

First, we'll deploy some techniques for solving difference equations that are also deployed in this QuantEcon lecture about the multiplier-accelerator model:
<https://python.quantecon.org/samuelson.html>



The characteristic equation associated with difference equation {eq}`eq:second_order` is

$$
c(x) \equiv x^2 - 2 x + (1 - \sigma) = 0
$$ (eq:cha_eq0)

+++

(This is an instance of equation {eq}`eq:2diff6` above.)

If we factor the right side of the  equation {eq}`eq:cha_eq0`, we obtain 

$$
c(x)= (x - \lambda_1) (x-\lambda_2) = 0
$$(eq:cha_eq)


where 

$$ 
c(x) = 0 
$$

for $x = \lambda_1$ or $x = \lambda_2$.

These two special values of $x$ are sometimes called zeros or roots of $c(x)$.


By applying the quadratic formula to solve for the roots  the characteristic equation 
{eq}`eq:cha_eq0`, we find that

$$
\lambda_1 = 1 + \sqrt{\sigma}, \quad \lambda_2 = 1 - \sqrt{\sigma} 
$$ (eq:secretweapon)

Formulas {eq}`eq:secretweapon` indicate that  $\lambda_1$ and  $\lambda_2$ are both simple functions
of a single variable, namely,  $\sqrt{\sigma}$, the object that some Ancient Greeks   wanted to compute.

Ancient Greeks had an indirect way of exploiting this fact to compute square roots of a positive integer.

They did this by starting from particular initial conditions $y_{-1}, y_{-2}$ and iterating on the difference equation {eq}`eq:second_order`.


Solutions  of  difference equation {eq}`eq:second_order` take the form

$$
y_t = \lambda_1^t \eta_1 + \lambda_2^t \eta_2
$$

where $\eta_1$ and $\eta_2$ are chosen to satisfy the  prescribed initial conditions $y_{-1}, y_{-2}$:

$$
\begin{align}
\lambda_1^{-1} \eta_1 + \lambda_2^{-1} \eta_2 & =  y_{-1} \cr
\lambda_1^{-2} \eta_1 + \lambda_2^{-2} \eta_2 & =  y_{-2}
\end{align}
$$(eq:leq_sq)

System {eq}`eq:leq_sq` of simultaneous linear equations will play a big role in the remainder of this lecture.  

Since $\lambda_1 = 1 + \sqrt{\sigma} > 1 > \lambda_2 = 1 - \sqrt{\sigma} $
it follows that for **almost all** (but not all) initial conditions

$$
\lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) = 1 + \sqrt{\sigma}
$$

Thus,

$$
\sqrt{\sigma} = \lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) - 1
$$

However, notice that if $\eta_1 = 0$, then

$$
\lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right) = 1 - \sqrt{\sigma}
$$

so that 

$$
\sqrt{\sigma} = 1 - \lim_{t \rightarrow \infty} \left(\frac{y_{t+1}}{y_t}\right)
$$

Actually, if $\eta_1 =0$, it follows that

$$
\sqrt{\sigma} = 1 - \left(\frac{y_{t+1}}{y_t}\right) \quad \forall t \geq 0,
$$

so that convergence is immediate and there is no need to take limits.

Symmetrically, if $\eta_2 =0$, it follows that 


$$
\sqrt{\sigma} =  \left(\frac{y_{t+1}}{y_t}\right) - 1 \quad \forall t \geq 0
$$

so again, convergence is immediate, and we have no need to compute a limit.


System {eq}`eq:leq_sq` of simultaneous linear equations can be used in various ways.

 * we can take $y_{-1}, y_{-2}$ as given initial conditions and solve for $\eta_1, \eta_2$
 
 * we can instead take $\eta_1, \eta_2$ as given and solve for initial conditions  $y_{-1}, y_{-2}$ 
 
Notice how we used the  second approach above when we set  $\eta_1, \eta_2$  either to $(0, 1)$, for example, or $(1, 0)$, for example.

In taking this second approach, we were in effect finding  an **invariant subspace** of ${\bf R}^2$. 

Here is what is going on.  

For $ t \geq 0$ and for most pairs of  initial conditions $(y_{-1}, y_{-2}) \in {\bf R}^2$ for equation {eq}`eq:second_order', $y_t$ can be expressed as a linear combination  of $y_{t-1}$ and $y_{t-2}$.

But for some special initial conditions $(y_{-1}, y_{-2}) \in {\bf R}^2$, $y_t$ can be expressed as a linear function  of $y_{t-1}$ only. 

These special initial conditions require that $y_{-1}$ be a linear function of $y_{-2}$.

We'll study these special initial conditions soon.  But first let's write some Python code to iterate on equation {eq}`eq:second_order` starting from an arbitrary $(y_{-1}, y_{-2}) \in {\bf R}^2$.

## Implementation

We now implement the above algorithm to compute the square root of $\sigma$.


In this lecture, we use the following import:

```{code-cell} ipython3
:tags: []

import numpy as np
```

```{code-cell} ipython3
:tags: []

def solve_λs(coefs):    
    # Calculate the roots using numpy.roots
    λs = np.roots(coefs)
    
    # Sort the roots for consistency
    return sorted(λs, reverse=True)

def solve_η(λ_1, λ_2, y_neg1, y_neg2):
    # Solve the system of linear equation
    A = np.array([
        [1/λ_1, 1/λ_2],
        [1/(λ_1**2), 1/(λ_2**2)]
    ])
    b = np.array((y_neg1, y_neg2))
    ηs = np.linalg.solve(A, b)
    
    return ηs

def solve_sqrt(σ, coefs, y_neg1, y_neg2, t_max=100):
    # Ensure σ is greater than 1
    if σ <= 1:
        raise ValueError("σ must be greater than 1")
        
    # Characteristic roots
    λ_1, λ_2 = solve_λs(coefs)
    
    # Solve for η_1 and η_2
    η_1, η_2 = solve_η(λ_1, λ_2, y_neg1, y_neg2)

    # Compute the sequence up to t_max
    t = np.arange(t_max + 1)
    y = (λ_1 ** t) * η_1 + (λ_2 ** t) * η_2
    
    # Compute the ratio y_{t+1} / y_t for large t
    sqrt_σ_estimate = (y[-1] / y[-2]) - 1
    
    return sqrt_σ_estimate

# Use σ = 2 as an example
σ = 2

# Encode characteristic equation
coefs = (1, -2, (1 - σ))

# Solve for the square root of σ
sqrt_σ = solve_sqrt(σ, coefs, y_neg1=2, y_neg2=1)

# Calculate the deviation
dev = abs(sqrt_σ-np.sqrt(σ))
print(f"sqrt({σ}) is approximately {sqrt_σ:.5f} (error: {dev:.5f})")
```

Now we consider cases where $(\eta_1, \eta_2) = (0, 1)$ and $(\eta_1, \eta_2) = (1, 0)$

```{code-cell} ipython3
:tags: []

# Compute λ_1, λ_2
λ_1, λ_2 = solve_λs(coefs)
print(f'Roots for the characteristic equation are ({λ_1:.5f}, {λ_2:.5f}))')
```

```{code-cell} ipython3
:tags: []

# Case 1: η_1, η_2 = (0, 1)
ηs = (0, 1)

# Compute y_{t} and y_{t-1} with t >= 0
y = lambda t, ηs: (λ_1 ** t) * ηs[0] + (λ_2 ** t) * ηs[1]
sqrt_σ = 1 - y(1, ηs) / y(0, ηs)

print(f"For η_1, η_2 = (0, 1), sqrt_σ = {sqrt_σ:.5f}")
```

```{code-cell} ipython3
:tags: []

# Case 2: η_1, η_2 = (0, 1)
ηs = (1, 0)
sqrt_σ = y(1, ηs) / y(0, ηs) - 1

print(f"For η_1, η_2 = (1, 0), sqrt_σ = {sqrt_σ:.5f}")
```

We find that convergence is immediate.

+++

Let's represent the preceding analysis by vectorizing our second order difference equation {eq}`eq:second_order` and then using  eigendecompositions of a state transition matrix.

## Vectorizing the difference equation


Represent {eq}`eq:second_order` with the first-order matrix difference equation

$$
\begin{bmatrix} y_{t+1} \cr y_{t} \end{bmatrix}
= \begin{bmatrix} 2 & - ( 1 - \sigma) \cr 1 & 0 \end{bmatrix} \begin{bmatrix} y_{t} \cr y_{t-1} \end{bmatrix}
$$

or

$$
x_{t+1} = M x_t 
$$

where 

$$
M = \begin{bmatrix} 2 & - (1 - \sigma )  \cr 1 & 0 \end{bmatrix},  \quad x_t= \begin{bmatrix} y_{t} \cr y_{t-1} \end{bmatrix}
$$

Construct an eigendecomposition of $M$:

$$
M = V \begin{bmatrix} \lambda_1 & 0 \cr 0 & \lambda_2  \end{bmatrix} V^{-1} 
$$ (eq:eigen_sqrt)

where columns of $V$ are eigenvectors corresponding to  eigenvalues $\lambda_1$ and $\lambda_2$.

The eigenvalues can be ordered so that  $\lambda_1 > 1 > \lambda_2$.

Write equation {eq}`eq:second_order` as

$$
x_{t+1} = V \Lambda V^{-1} x_t
$$

Define

$$
x_t^* = V^{-1} x_t
$$

We can recover $x_t$ from $x_t^*$:

$$
x_t = V x_t^*
$$


The following notations and equations will help us.

Let 

$$

V = \begin{bmatrix} V_{1,1} & V_{1,2} \cr 
                         V_{2,2} & V_{2,2} \end{bmatrix}, \quad
V^{-1} = \begin{bmatrix} V^{1,1} & V^{1,2} \cr 
                         V^{2,2} & V^{2,2} \end{bmatrix}
$$

Notice that it follows from

$$
 \begin{bmatrix} V^{1,1} & V^{1,2} \cr 
                         V^{2,2} & V^{2,2} \end{bmatrix} \begin{bmatrix} V_{1,1} & V_{1,2} \cr 
                         V_{2,2} & V_{2,2} \end{bmatrix} = \begin{bmatrix} 1  & 0 \cr 0 & 1 \end{bmatrix}
$$

that

 

$$
V^{2,1} V_{1,1} + V^{2,2} V_{2,1} = 0
$$

and

$$
V^{1,1}V_{1,2} + V^{1,2} V_{2,2} = 0
$$

These equations will be very useful soon.


Notice that

$$
\begin{bmatrix} x_{1,t+1}^* \cr x_{2,t+1}^* \end{bmatrix} = \begin{bmatrix} \lambda_1  & 0 \cr 0 & \lambda_2 \end{bmatrix}
\begin{bmatrix} x_{1,t}^* \cr x_{2,t}^* \end{bmatrix}
$$

To deactivate $\lambda_1$ we want to set

$$
x_{1,0}^* = 0
$$


This can be achieved by setting 

$$
x_{2,0} =  -( V^{1,2})^{-1} V^{1,1} = V_{2,1} V_{1,1}^{-1} x_{1,0}.
$$ (eq:deactivate1)

To deactivate $\lambda_2$, we want to  set

$$
x_{2,0}^* = 0
$$

This can be achieved by setting 

$$
x_{2,0} = -(V^{2,2})^{-1} V^{2,1} = V_{2,1} V_{1,1}^{-1} x_{1,0}
$$ (eq:deactivate2)


We shall encounter equations very similar to {eq}`eq:deactivate1` and {eq}`eq:deactivate2`
in  this QuantEcon lecture {doc}`money financed government deficits and inflation <money_inflation>`
and in many other places in dynamic economic theory.

### Implementation

Now we implement the algorithm above.

First we write a function that iterates $M$

```{code-cell} ipython3
:tags: []

def iterate_M(x_0, M, num_steps):
    # Eigendecomposition of M
    Λ, V = np.linalg.eig(M)
    V_inv = np.linalg.inv(V)
    
    print(f"eigenvalue:\n{Λ}")
    print(f"eigenvector:\n{V}")
    
    # Initialize the array to store results
    x = np.zeros((x_0.shape[0], num_steps))
    
    # Perform the iterations
    for t in range(num_steps):
        x[:, t] = V @ np.diag(Λ**t) @ V_inv @ x_0
    
    return x

# Define the state transition matrix M
M = np.array([[2, -(1 - σ)],
              [1, 0]])

# Initial condition vector x_0
x_0 = np.array([1, 0])

# Perform the iteration
xs = iterate_M(x_0, M, num_steps=100)
```

Compare the eigenvector to the roots we obtained above

```{code-cell} ipython3
:tags: []

roots = solve_λs((1, -2, (1 - σ)))
print(f"roots: {np.round(roots, 8)}")
```

Hence we confirmed {eq}`eq:eigen_sqrt`.
