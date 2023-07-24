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

+++ {"user_expressions": []}

# SymPy

## Overview

Unlike numerical libraries that deal with concrete values, [SymPy](https://www.sympy.org/en/index.html)  focuses on manipulating mathematical symbols and expressions directly.

SymPy provides [a wide range of functionalities](https://www.sympy.org/en/features.html) including 

- Symbolic Expression
- Equation Solving
- Simplification
- Calculus
- Matrices
- Discrete Math, etc.

Unlike other popular tools for symbolic computations such as Mathematica it is an open-source library supported by an active community. 

## Getting Started

Let’s first import the library and initialize the printer to print the symbolic output

```{code-cell} ipython3
from sympy import *
from sympy.plotting import plot, plot3d_parametric_line 
import numpy as np

# Enable best printer available
init_printing()
```

+++ {"user_expressions": []}

## Symbolic algebra

### Symbols

+++ {"user_expressions": []}

Before we start manipulating the symbols, let's initialize some symbols to work with

```{code-cell} ipython3
x, y, z = symbols('x y z')
```

+++ {"user_expressions": []}

### Expressions

We can now use these symbols `x`, `y`, and `z` to build expressions and equations

Let's build a simple expression first

```{code-cell} ipython3
---
mystnb:
  image:
    width: 10%
---
expr = (x + y) ** 2
expr
```

+++ {"user_expressions": []}

We can expand this expression with the `expand` function

```{code-cell} ipython3
expand_expr = expand(expr)
expand_expr
```

+++ {"user_expressions": []}

and factorize it back to the factored form with the `factor` function

```{code-cell} ipython3
factor(expand_expr)
```

+++ {"user_expressions": []}

We can solve this expression

```{code-cell} ipython3
solve(expr)
```

+++ {"user_expressions": []}

Note this is equivalent to solving the equation

$$
(x + y)^2 = 0 
$$

+++ {"user_expressions": []}

### Equations

SymPy provides several functions to manipulate equations.

Let's develop an equation with the expression we defined before

```{code-cell} ipython3
eq = Eq(expr, 0)
eq
```

+++ {"user_expressions": []}

Solving this expression with respect to x gives the same output as solving the expression directly

```{code-cell} ipython3
solve(eq, x)
```

+++ {"user_expressions": []}

SymPy can easily handle equations with multiple solutions

```{code-cell} ipython3
eq = Eq(expr, 1)
solve(eq, x)
```

+++ {"user_expressions": []}

`solve` function can also combine multiple equations together and solve a system of equations

```{code-cell} ipython3
eq2 = Eq(x, y)
eq2
```

```{code-cell} ipython3
solve([eq, eq2], [x, y])
```

+++ {"user_expressions": []}

We can also solve for y by simply substituting the $x$ with $y$

```{code-cell} ipython3
expr_sub = expr.subs(x, y)
```

```{code-cell} ipython3
solve(Eq(expr_sub, 1))
```

+++ {"user_expressions": []}

Let's we create another equation with symbol `x` and functions `sin`, `cos`, and `tan` using the `Eq` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
# Create an equation
eq = Eq(cos(x) / (tan(x)/sin(x)), 0)
eq
```

+++ {"user_expressions": []}

Now we simplify this equation using the `simplify` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 10%
---
# Simplify an expression
simplified_expr = simplify(eq)
simplified_expr
```

+++ {"user_expressions": []}

We can solve equations using the **solve** function in **sympy**

```{code-cell} ipython3
---
mystnb:
  image:
    width: 5.5%
---
# Solve the equation
sol = solve(eq, x)
sol
```

+++ {"user_expressions": []}

SymPy can also handle more complex equations involving trignomotry and complex number.

We demonstrate this using [Euler's formula](https://en.wikipedia.org/wiki/Euler%27s_formula)

```{code-cell} ipython3
# 'I' represents imaginary number i 
euler = cos(x) + I*sin(x)
euler
```

```{code-cell} ipython3
simplify(euler)
```

+++ {"user_expressions": []}

#### Example: fixed point computation

One version of Solow-Swan growth dynamics is 

$$
k_{t+1}=s f\left(k_t\right)+(1-\delta) k_t, \quad t=0,1, \ldots
$$

where $k_t$ is the capital stock, $f$ is a production function, $\delta$ is a rate of depreciation.

With $f(k) = Ak^a$, one can show the unique fixed point of the dynamics is 

$$
k^*:=\left(\frac{s A}{\delta}\right)^{1 /(1-\alpha)}
$$ 

This can be easily computed in SymPy

```{code-cell} ipython3
A, s, k, α, δ = symbols('A s k α δ')
```

```{code-cell} ipython3
# Define Solow-Swan growth dynamics
solow = Eq(s*A*k**α + (1 - δ) * k, k)
```

```{code-cell} ipython3
solve(solow, k)
```

+++ {"user_expressions": []}

### Series

Series are widely used in economics and statistics from asset pricing to expectation of discrete random variables.

We can construct a simple series of summations using `Sum` function

```{code-cell} ipython3
x, y, i, j = symbols("x y i j")
sum_xy = Sum(Indexed('x',i) * Indexed('y', j), 
        (i, 0, 3),
        (j, 0, 3))
sum_xy
```

+++ {"user_expressions": []}

To evaluate the sum, we can `lamdify` the formula.

The lamdified expression can take inputs and compute the result

```{code-cell} ipython3
sum_xy = lambdify([x, y], sum_xy)
grid = np.arange(0, 4, 1)
sum_xy(grid, grid)
```

+++ {"user_expressions": []}

#### Example: asset pricing

Imagine a bank with $D_0$ as the deposit at time $t$, it loans $(1-r)$ of its deposits and keeps a fraction 
$r$ as cash reserves, one can calculate the deposite at time with

$$
\sum_{i=0}^\infty (1-r)^i D_0
$$

```{code-cell} ipython3
r, D = symbols('r D')

Dt = Sum('(1 - r)^i * D', (i, 0, oo))
Dt
```

+++ {"user_expressions": []}

We can call `doit` method to evaluate the series

```{code-cell} ipython3
Dt.doit()
```

+++ {"user_expressions": []}

Simplifying the expression above gives

```{code-cell} ipython3
simplify(Dt.doit())
```

+++ {"user_expressions": []}

This is consistent with the solution we provided in our lecture on [geometric series](https://python.quantecon.org/geom_series.html#example-the-money-multiplier-in-fractional-reserve-banking).

+++ {"user_expressions": []}

## Symbolic Calculus

**sympy** allows us to perform various calculus operations such as differentiation and integration.


### Limits

We can compute limits for a given function using the `limit` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 3%
---
# Define a function
f = x ** 2 / (x - 1)

# Compute the limit
lim = limit(f, x, 0)
lim
```

+++ {"user_expressions": []}

### Derivatives

We can differentiate any **sympy** expression using `diff(func, var)`

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
---
# Differentiate a function
df = diff(f, x)
df
```

```{code-cell} ipython3
limit(df, x, 0)
```

+++ {"user_expressions": []}

### Integrals

We can compute definite and indefinite integrals using `integrate` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 12%
---
# Calculate the indefinite integral
indef_int = integrate(df, x)
indef_int
```

+++ {"user_expressions": []}

## Plotting

sympy provides a powerful plotting feature. We'll plot a function using `sympy.plot()`

```{code-cell} ipython3
f = x ** 2
df = diff(f)
p = plot(f, df, (x, -10, 10), title="Graph", legend= True, xlabel='x', ylabel='f(x)', show=False)
p.title = 'A Simple Plot'
p.xlabel = 'x'
p.ylabel = 'f(x)'
p.show()
```

+++ {"user_expressions": []}

There are more plotting functions for more complicated equations

```{code-cell} ipython3
t = symbols('t')
alpha = [cos(t), sin(t), t]
plot3d_parametric_line(*alpha)
```

+++ {"user_expressions": []}

## Applications
In this section, we apply sympy to an economic model that explores the wage gap between college and high school graduates. 

We'll use symbolic computation to define, manipulate, and solve equations in the model, thereby deriving insights that would be challenging to obtain through numerical methods alone. 

Let's get started.


### Defining the Symbols

The first step in symbolic computation is to define the symbols that represent the variables in our equations. In our model, we have the following variables:

 * $R$ The gross rate of return on a one-period bond

 * $t = 0, 1, 2, \ldots T$ Denote the years that a person either works or attends college
 
 * $T$ denote the last period that a person  works
 
 * $w$ The wage of a high school graduate
 
 * $g$ The growth rate of wages

 * $D$ The upfront monetary costs of going to college
 
 * $\phi$ The wage gap, defined as the ratio of the wage of a college graduate to the wage of a high school graduate
 
 Let's define these symbols using sympy

```{code-cell} ipython3
# Define the symbols
w, R, g, D, phi = symbols('w R g D phi')
```

+++ {"user_expressions": []}

### Defining the Present Value Equations
We calculate the present value of the earnings of a high school graduate and a college graduate. For a high school graduate, we sum the discounted wages from year 1 to $T = 4$. For a college graduate, we sum the discounted wages from year 5 to $T = 4$, considering the cost of college $D$ paid upfront

+++ {"user_expressions": []}

$$
PV_{\text{{highschool}}} = \frac{w}{R} + \frac{w*(1 + g)}{R^2} + \frac{w*(1 + g)^2}{R^3} + \frac{w*(1 + g)^3}{R^4}
$$

$$
PV_{\text{{college}}} = D + \frac{\phi * w * (1 + g)^4}{R} + \frac{\phi * w * (1 + g)^5}{R^2} + \frac{\phi * w * (1 + g)^6}{R^3} + \frac{\phi * w * (1 + g)^7}{R^4}
$$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
# Define the present value equations
PV_highschool = w/R + w*(1 + g)/R**2 + w*(1 + g)**2/R**3 + w*(1 + g)**3/R**4
PV_college = D + phi*w*(1 + g)**4/R + phi*w*(1 + g)**5 / \
    R**2 + phi*w*(1 + g)**6/R**3 + phi*w*(1 + g)**7/R**4
```

+++ {"user_expressions": []}

### Defining the Indifference Equation
The indifference equation represents the condition that a worker is indifferent between going to college or not. This is given by the equation $PV_h$ = $PV_c$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
# Define the indifference equation
indifference_eq = Eq(PV_highschool, PV_college)
```

+++ {"user_expressions": []}

We can now solve the indifference equation for the wage gap $\phi$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 65%
---
# Solve for phi
solution = solve(indifference_eq, phi)

# Simplify the solution
solution = simplify(solution[0])
solution
```

+++ {"user_expressions": []}

To compute a numerical value for $\phi$, we replace symbols $w$, $R$, $g$, and $D$ with specific numbers. 

Let's take $w = 1$, $R = 1.05$, $g = 0.02$, and $D = 0.5$. With these values, we can substitute them into the equation and find a specific value for $\phi$.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
---
# Substitute specific values
solution_num = solution.subs({w: 1, R: 1.05, g: 0.02, D: 0.5})
solution_num
```

+++ {"user_expressions": []}

The wage of a college graduate is approximately 0.797 times the wage of a high school graduate.

+++ {"user_expressions": []}

#### Example: L'Hôpital's rule

```{code-cell} ipython3
f_upper = y**x - 1
f_lower = x
f = f_upper/f_lower
f
```

```{code-cell} ipython3
lim = limit(f, x, 0)
lim
```

```{code-cell} ipython3
lim = limit(diff(f_upper, x)/
            diff(f_lower, x), x, 0)
lim
```
