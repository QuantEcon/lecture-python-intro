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

# Introduction to Symbolic Algebra and Calculus with Sympy


## Overview

Symbolic computation, or computer algebra, involves using algorithms and software to perform exact computations and manipulate mathematical equations. While **Mathematica** is a popular tool for symbolic computations, it's proprietary and can be costly. As an open-source alternative, the **sympy** library in Python offers a comprehensive set of functionalities for symbolic mathematics. This allows users to perform a range of operations, from algebraic manipulations to calculus, providing exact solutions instead of numerical approximations.


## Getting Started

Let's get started with **sympy** by installing and setting up the package.

```{code-cell} ipython3
! pip install sympy
```

We'll start by importing the necessary **sympy** functions and initializing our environment.

```{code-cell} ipython3
from sympy import symbols, init_printing

# Enable best printer available
init_printing()

# Let's define some symbols to work with
x, y, z = symbols('x y z')
```

We can now use these symbols **x**, **y**, and **z** to build symbolic expressions.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 10%
---
# An expression
expr = (x + y) ** 2
expr
```

## Symbolic Algebra


### Algebraic Expressions

**sympy** provides several functions to create and manipulate algebraic expressions. Let's look at a few examples.

Creating Algebraic Expressions

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
from sympy import Eq, solve

# Create an equation
eq = Eq(x**2 - 3*x + 2, 0)
eq
```

Manipulating Algebraic Expressions

```{code-cell} ipython3
---
mystnb:
  image:
    width: 10%
---
from sympy import simplify

# Simplify an expression
simplified_expr = simplify(expr)
simplified_expr
```

### Solving Algebraic Equations

We can solve equations using the **solve** function in **sympy**.

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

## Symbolic Calculus

**sympy** allows us to perform various calculus operations such as differentiation and integration.


### Limits

We can compute limits for a given function.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 3%
---
from sympy import limit

# Define a function
f = x ** 2 / (x - 1)

# Compute the limit
lim = limit(f, x, 1)
lim
```

### Derivatives

We can differentiate any **sympy** expression using **diff(func, var)**.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
---
from sympy import diff

# Differentiate a function
df = diff(f, x)
df
```

### Integrals

We can compute definite and indefinite integrals.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 12%
---
from sympy import integrate

# Calculate the indefinite integral
indef_int = integrate(df, x)
indef_int
```

## Plotting

sympy provides a powerful plotting feature. We'll plot a function using **sympy.plot()**.

```{code-cell} ipython3
from sympy.plotting import plot

p = plot(f, (x, -10, 10), show=False)
p.title = 'A Simple Plot'
p.xlabel = 'x'
p.ylabel = 'f(x)'
p.show()
```

## Applications
In this section, we apply sympy to an economic model that explores the wage gap between college and high school graduates. We'll use symbolic computation to define, manipulate, and solve equations in the model, thereby deriving insights that would be challenging to obtain through numerical methods alone. Let's get started.


### Defining the Symbols

The first step in symbolic computation is to define the symbols that represent the variables in our equations. In our model, we have the following variables:

 * $R$ The gross rate of return on a one-period bond

 * $t = 0, 1, 2, \ldots T$ Denote the years that a person either works or attends college
 
 * $T$ denote the last period that a person  works
 
 * $w$ The wage of a high school graduate
 
 * $g$ The growth rate of wages

 * $D$ The upfront monetary costs of going to college
 
 * $\phi$ The wage gap, defined as the ratio of the wage of a college graduate to the wage of a high school graduate
 
 Let's define these symbols using sympy.

```{code-cell} ipython3
from sympy import symbols

# Define the symbols
w, R, g, D, phi = symbols('w R g D phi')
```

### Defining the Present Value Equations
We calculate the present value of the earnings of a high school graduate and a college graduate. For a high school graduate, we sum the discounted wages from year 1 to $T = 4$. For a college graduate, we sum the discounted wages from year 5 to $T = 4$, considering the cost of college $D$ paid upfront.

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

### Defining the Indifference Equation
The indifference equation represents the condition that a worker is indifferent between going to college or not. This is given by the equation $PV_h$ = $PV_c$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
from sympy import Eq

# Define the indifference equation
indifference_eq = Eq(PV_highschool, PV_college)
```

We can now solve the indifference equation for the wage gap $\phi$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 65%
---
from sympy import solve, simplify

# Solve for phi
solution = solve(indifference_eq, phi)

# Simplify the solution
solution = simplify(solution[0])
solution
```

If you want to compute a numerical value for $\phi$, you need to replace the symbols $w$, $R$, $g$ and $D$ with specific numbers. For instance, if you want to use $w = 1$, $R = 1.05$, $g = 0.02$ and $D = 0.5$, you can do this:

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

The wage of a college graduate is approximately 0.797 times the wage of a high school graduate
