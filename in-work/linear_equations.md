---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Linear Equations and Matrix Algebra

```{index} single: Linear Equations and Matrix Algebra
```

```{contents} Contents
:depth: 2
```

## Overview

Many problems in economics and finance require solving linear equations.

In this lecture we discuss linear equations and their solutions.

We also discuss how to compute the solutions with matrix algebra.

We assume that students are familiar with matrices and understand the basics
of matrix algebra.

We will use the following imports:

```python
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
```

<!-- #region -->
## A Two Good Example

We discuss a simple two good example and solve it by

1. pencil and paper
2. matrix algebra

The second method is more general, as we will see.


### Pencil and Paper Methods

Suppose that we have two goods, such as propane and ethanol.

To keep things simple, we label them as good 0 and good 1.

The demand for each good depends on the price of both goods:

```{math}
:label: two_eq_demand
\begin{aligned}
    q_0^d = 100 - 10 p_0 - 5 p_1 \\
    q_1^d = 50 - p_0 - 10 p_1 
\end{aligned}
```

(We are assuming demand decreases when the price of either good goes up.)

Let's suppose that supply is given by 

```{math}
:label: two_eq_supply
\begin{aligned}
    q_0^s = 10 p_0 + 5 p_1 \\
    q_1^s = 5 p_0 + 10 p_1 
\end{aligned}
```

Equilibrium holds when supply equals demand ($q_0^s = q_0^d$ and $q_1^s = q_1^d$).

This yields the linear system

```{math}
:label: two_equilibrium
\begin{aligned}
    100 - 10 p_0 - 5 p_1 = 10 p_0 + 5 p_1 \\
    50 - p_0 - 10 p_1 = 5 p_0 + 10 p_1 
\end{aligned}
```

We can solve this with pencil and paper to get

$$
    p_0 = 4.41 \quad \text{and} \quad p_1 = 1.18
$$    

Inserting these results into either {eq}`two_eq_demand` or {eq}`two_eq_supply` yields the
equilibrium quantities 

$$
    q_0 = 50 \quad \text{and} \quad q_1 = 33.82
$$

We can also obtain a solution via matrix algebra but first let's discuss the basics of vectors and
matrices, in both theory and computation.

## {index}`Vectors <single: Vectors>`

```{index} single: Linear Algebra; Vectors
```

A **vector** of length $n$ is just a sequence (or array, or tuple) of $n$ numbers, which we write as $x = (x_1, \ldots, x_n)$.

We can write these sequences either horizontally or vertically.

But when we use matrix operations, our default assumption is that vectors are column vectors.

The set of all $n$-vectors is denoted by $\mathbb R^n$.

For example, $\mathbb R ^2$ is the plane and a vector in $\mathbb R^2$ is just a point in the plane.

Traditionally, vectors are represented visually as arrows from the origin to the point.

The following figure represents three vectors in this manner.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(10, 8))
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-5, 5), ylim=(-5, 5))
ax.grid()
vecs = ((2, 4), (-3, 3), (-4, -3.5))
for v in vecs:
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='blue',
                shrink=0,
                alpha=0.7,
                width=0.5))
    ax.text(1.1 * v[0], 1.1 * v[1], str(v))
plt.show()
```

### Vector Operations

```{index} single: Vectors; Operations
```

The two most common operators for vectors are addition and scalar multiplication, which we now describe.

When we add two vectors, we add them element-by-element.

For example,

$$
\begin{bmatrix}
    4 \\
    -2 
\end{bmatrix}
+
\begin{bmatrix}
    3 \\
    3 
\end{bmatrix}
=
\begin{bmatrix}
    4 & + & 3 \\
    -2 & + & 3 
\end{bmatrix}
=
\begin{bmatrix}
    7 \\
    1
\end{bmatrix}
$$

In general,

$$
x + y =
\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
\end{bmatrix} +
\begin{bmatrix}
     y_1 \\
     y_2 \\
    \vdots \\
     y_n
\end{bmatrix} :=
\begin{bmatrix}
    x_1 + y_1 \\
    x_2 + y_2 \\
    \vdots \\
    x_n + y_n
\end{bmatrix}
$$

We can visualise vector addition in $\mathbb{R}^2$ as follows.

```python
fig, ax = plt.subplots(figsize=(10, 8))
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-2, 10), ylim=(-4, 4))
#ax.grid()
vecs = ((4, -2), (3, 3),(7,1))
tags = ('(x1, x2)','(y1, y2)','(x1+x2, y1+y2)')
colors = ('blue','green','red')
for i, v in enumerate(vecs):
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(color = colors[i],
                shrink=0,
                alpha=0.7,
                width=0.5,
                headwidth=8,
                headlength=15))
    ax.text(v[0] + 0.2, v[1] + 0.1, tags[i])

for i,v in enumerate(vecs):
    ax.annotate('', xy=(7,1), xytext=v,
                arrowprops=dict(color = 'gray',
                shrink=0,
                alpha=0.3,
                width=0.5,
                headwidth=5,
                headlength=20))
plt.show()
```

Scalar multiplication is an operation that multiplies a vector $x$ with a scalar elementwise.

For example,

$$
-2
\begin{bmatrix}
    3 \\
    -7 
\end{bmatrix}
=
\begin{bmatrix}
    -2 & \times & 3 \\
    -2 & \times & -7
\end{bmatrix}
=
\begin{bmatrix}
    -6 \\
    14
\end{bmatrix}
$$

More generally, it takes a number $\gamma$ and a vector $x$ and produces

$$
\gamma x :=
\begin{bmatrix}
    \gamma x_1 \\
    \gamma x_2 \\
    \vdots \\
    \gamma x_n
\end{bmatrix}
$$

Scalar multiplication is illustrated in the next figure.

```python
fig, ax = plt.subplots(figsize=(10, 8))
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-5, 5), ylim=(-5, 5))
x = (2, 2)
ax.annotate('', xy=x, xytext=(0, 0),
            arrowprops=dict(facecolor='blue',
            shrink=0,
            alpha=1,
            width=0.5))
ax.text(x[0] + 0.4, x[1] - 0.2, '$x$', fontsize='16')

scalars = (-2, 2)
x = np.array(x)

for s in scalars:
    v = s * x
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='red',
                shrink=0,
                alpha=0.5,
                width=0.5))
    ax.text(v[0] + 0.4, v[1] - 0.2, f'${s} x$', fontsize='16')
plt.show()
```

In Python, a vector can be represented as a list or tuple, such as `x = [2, 4, 6]` or `x = (2, 4, 6)`.

However, it is more common to represent vectors with [NumPy arrays](https://python-programming.quantecon.org/numpy.html#numpy-arrays).

One advantage of NumPy arrays is that scalar multiplication and addition have very natural syntax.

```python
x = np.ones(3)            # Vector of three ones
y = np.array((2, 4, 6))   # Converts tuple (2, 4, 6) into a NumPy array
x + y                     # Add (element-by-element)
```

```python
4 * x                     # Scalar multiply
```

### Inner Product and Norm

```{index} single: Vectors; Inner Product
```

```{index} single: Vectors; Norm
```

The **inner product** of vectors $x,y \in \mathbb R ^n$ is defined as

$$
x' y = 
\begin{bmatrix}
    \color{red}{x_1} & \color{blue}{x_2} & \cdots & x_n
\end{bmatrix}
\begin{bmatrix}
    \color{red}{y_1} \\
    \color{blue}{y_2} \\
    \vdots \\
    y_n
\end{bmatrix}
= {\color{red}{x_1 y_1}} + {\color{blue}{x_2 y_2}} + \cdots + x_n y_n
:= \sum_{i=1}^n x_i y_i
$$

The **norm** of a vector $x$ represents its "length" (i.e., its distance from the zero vector) and is defined as

$$
\| x \| := \sqrt{x' x} := \left( \sum_{i=1}^n x_i^2 \right)^{1/2}
$$

The expression $\| x - y\|$ can be thought of as the "distance" between $x$ and $y$.

The inner product and norm can be computed as follows

```python
np.sum(x*y)      # Inner product of x and y
```

```python
x @ y            # Inner product of x and y
```

```python
np.sqrt(np.sum(x**2))  # Norm of x, take one
```

```python
np.linalg.norm(x)      # Norm of x, take two
```

<!-- #region -->
## Matrix Operations

```{index} single: Matrix; Operations
```

When we discussed linear price systems, we mentioned a solution using matrix algebra.

Matrix algebra is similar to algebra for numbers.

Let's review some details.

Just as was the case for vectors, we can add, subtract and scalar multiply
matrices.

Scalar multiplication and addition are generalizations of the vector case:

Here is an example of scalar multiplication,

$$
3
\begin{bmatrix}
    2 & -13 \\
    0 & 5
\end{bmatrix}
=
\begin{bmatrix}
    6 & -39 \\
    0 & 15
\end{bmatrix}
$$

In general for a number $\gamma$ and any matrix $A$,

$$
\gamma A =
\gamma
\begin{bmatrix}
    a_{11} &  \cdots & a_{1k} \\
    \vdots & \vdots  & \vdots \\
    a_{n1} &  \cdots & a_{nk}
\end{bmatrix} :=
\begin{bmatrix}
    \gamma a_{11} & \cdots & \gamma a_{1k} \\
    \vdots & \vdots & \vdots \\
    \gamma a_{n1} & \cdots & \gamma a_{nk}
\end{bmatrix}
$$

Consider this example of matrix addition,

$$
\begin{bmatrix}
    1 & 5 \\
    7 & 3 \\
\end{bmatrix}
+
\begin{bmatrix}
    12 & -1 \\
    0 & 9
\end{bmatrix}
=
\begin{bmatrix}
    13 & 4 \\
    7 & 12
\end{bmatrix}
$$

In general,

$$
A + B =
\begin{bmatrix}
    a_{11} & \cdots & a_{1k} \\
    \vdots & \vdots & \vdots \\
    a_{n1} & \cdots & a_{nk}
\end{bmatrix} +
\begin{bmatrix}
    b_{11} & \cdots & b_{1k} \\
    \vdots & \vdots & \vdots \\
    b_{n1} & \cdots & b_{nk}
\end{bmatrix} :=
\begin{bmatrix}
    a_{11} + b_{11} &  \cdots & a_{1k} + b_{1k} \\
    \vdots & \vdots & \vdots \\
    a_{n1} + b_{n1} &  \cdots & a_{nk} + b_{nk}
\end{bmatrix}
$$

In the latter case, the matrices must have the same shape in order for the
definition to make sense.

We also have a convention for *multiplying* two matrices.

The rule for matrix multiplication generalizes the idea of inner products
discussed above.

If $A$ and $B$ are two matrices, then their product $A B$ is formed by taking
as its $i,j$-th element the inner product of the $i$-th row of $A$ and the
$j$-th column of $B$.

If $A$ is $n \times k$ and $B$ is $j \times m$, then to multiply $A$ and $B$
we require $k = j$, and the resulting matrix $A B$ is $n \times m$.

Let's start with an example of a $2 \times 2$ matrix multiplied by a $2 \times 1$ vector.

$$
Ax =
\begin{bmatrix}
    \color{red}{a_{11}} & \color{red}{a_{12}} \\
    a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
    \color{red}{x_1} \\
    \color{red}{x_2}
\end{bmatrix}
=
\begin{bmatrix}
    \color{red}{a_{11}x_1 + a_{12}x_2} \\
    a_{21}x_1 + a_{22}x_2
\end{bmatrix}
$$

As perhaps the most important special case, consider multiplying $n \times k$
matrix $A$ and $k \times 1$ column vector $x$.

According to the preceding rule, this gives us an $n \times 1$ column vector.

```{math}
:label: la_atx

A x =
{\begin{bmatrix}
    a_{11} & a_{12} &  \cdots & a_{1k} \\
    \vdots & \vdots & & \vdots \\
    \color{red}{a_{i1}} & \color{red}{a_{i2}} & \color{red}{\cdots} & \color{red}{a_{i}k} \\
    \vdots & \vdots & & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nk}
\end{bmatrix}}_{\textcircled{n} \times k}
{\begin{bmatrix}
    \color{red}{x_{1}}  \\
    \color{red}{x_{2}}  \\
    \color{red}{\vdots} \\
    \color{red}{\vdots}  \\
    \color{red}{x_{k}}
\end{bmatrix}}_{k \times \textcircled{1}} :=
{\begin{bmatrix}
    a_{11} x_1 + a_{22} x_2 + \cdots + a_{1k} x_k \\
    \vdots \\
    \color{red}{a_{i1} x_1 + a_{i2} x_2 + \cdots + a_{ik} x_k} \\
    \vdots \\
    a_{n1} x_1 + a_{n2} x_2 + \cdots + a_{nk} x_k
\end{bmatrix}}_{n \times 1}
```

Here is a simple illustration of multiplication of two matrices.

$$
AB =
\begin{bmatrix}
    a_{11} & a_{12} \\
    \color{red}{a_{21}} & \color{red}{a_{22}} \\
\end{bmatrix}
\begin{bmatrix}
    b_{11} & \color{red}{b_{12}} \\
    b_{21} & \color{red}{b_{22}} \\
\end{bmatrix} :=
\begin{bmatrix}
    a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\
    a_{21}b_{11} + a_{22}b_{21} & \color{red}{a_{21}b_{12} + a_{22}b_{22}}
\end{bmatrix}
$$

There are many tutorials to help you further visualize this operation, such as [this
one](http://www.mathsisfun.com/algebra/matrix-multiplying.html), or the
discussion on the [Wikipedia page](https://en.wikipedia.org/wiki/Matrix_multiplication).


```{note}
Unlike number products, $A B$ and $B A$ are not generally the same thing.
```

Another important special case is the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix)

$$
    I = 
    \begin{bmatrix}
        1 & \cdots & 0 \\
        \vdots & \ddots & \vdots \\
        0 &  \cdots & 1
    \end{bmatrix}
$$

that has ones on the principal diagonal and zero elsewhere.

It is a useful exercise to check the following:

* If $A$ is $n \times k$ and $I$ is the $k \times k$ identity matrix, then $AI = A$.
* If $I$ is the $n \times n$ identity matrix, then $IA = A$.



### Matrices in NumPy

```{index} single: Matrix; Numpy
```

NumPy arrays are also used as matrices, and have fast, efficient functions and methods for all the standard matrix operations.

You can create them manually from tuples of tuples (or lists of lists) as follows
<!-- #endregion -->

```python
A = ((1, 2),
     (3, 4))

type(A)
```

```python
A = np.array(A)

type(A)
```

```python
A.shape
```

The `shape` attribute is a tuple giving the number of rows and columns ---
see [here](https://python-programming.quantecon.org/numpy.html#shape-and-dimension)
for more discussion.

To get the transpose of `A`, use `A.transpose()` or, more simply, `A.T`.

There are many convenient functions for creating common matrices (matrices of zeros,
ones, etc.) --- see [here](https://python-programming.quantecon.org/numpy.html#creating-arrays).

Since operations are performed elementwise by default, scalar multiplication and addition have very natural syntax

```python
A = np.identity(3)    # 3 x 3 identity matrix
B = np.ones((3, 3))   # 3 x 3 matrix of ones
2 * A
```

```python
A + B
```

<!-- #region -->
To multiply matrices we use the `@` symbol.


```{note}
In particular, `A @ B` is matrix multiplication, whereas `A * B` is element-by-element multiplication.
```

### Two Good Model in Matrix Form

We can now revisit the two good model and solve {eq}`two_equilibrium` numerically via matrix algebra.

This involves some extra steps but the method is widely applicable --- as we
will see when we include more goods.

First we rewrite {eq}`two_eq_demand` as

```{math}
:label: two_eq_demand_mat
    q^d = D p + h
    \quad \text{where} \quad
    q^d = 
    \begin{bmatrix}
        q_0^d \\
        q_1^d
    \end{bmatrix}
    \quad
    D = 
    \begin{bmatrix}
         -10 & - 5  \\
         - 1  & - 10  
    \end{bmatrix}
    \quad \text{and} \quad
    h =
    \begin{bmatrix}
        100 \\
        50
    \end{bmatrix}
```

(Please check that $q^d = D p + h$ represents the same equations as {eq}`two_eq_demand`.)

We rewrite {eq}`two_eq_supply` as

```{math}
:label: two_eq_supply_mat
    q^s = C p 
    \quad \text{where} \quad
    q^s = 
    \begin{bmatrix}
        q_0^s \\
        q_1^s
    \end{bmatrix}
    \quad \text{and} \quad
    C = 
    \begin{bmatrix}
         10 & 5  \\
         5 & 10  
    \end{bmatrix}
```

Now equality of supply and demand can be expressed as $q^s = q^d$, or

$$
    C p = D p + h
$$

Matrix algebra is similar in many ways to ordinary algebra, with numbers

In this case, we can rearrange the terms to get 

$$
    (C - D) p = h
$$

If all of the terms were numbers, we could solve for prices as $p = h /
(C-D)$.

Matrix algebra allows us to do something similar: we can solve for equilibrium
prices using the inverse of $C - D$:

```{math}
:label: two_matrix
    p = (C - D)^{-1} h
```

Before we implement the solution let us consider a more general setting to completely
grasp the need for matrix algebra to solve a linear system of equation.

### More Goods

It is natural to think about demand systems with more goods.

For example, even within energy commodities there are many different goods,
including crude oil, gasoline, coal, natural gas, ethanol and uranium.

Obviously pencil and paper solutions become very time consuming with large
systems.

But fortunately the matrix methods described above are essentially unchanged.

In general, we can write the demand equation as $q^d = Dp + h$, where

* $q^d$ is an $n \times 1$ vector of demand quantities for $n$ different goods
* $D$ is an $n \times n$ "coefficient" matrix
* $h$ is an $n \times 1$ vector of constant values

Similarly, we can write the supply equation as $q^s = Cp + e$, where

* $q^d$ is an $n \times 1$ vector of supply quantities for the same goods
* $C$ is an $n \times n$ "coefficient" matrix
* $e$ is an $n \times 1$ vector of constant values

To find an equilibrium, we solve $Dp + h = Cp + e$, or

```{math}
:label: n_eq_sys_la
    (D- C)p = e - h 
```

The solution is

$$ 
    p = (D- C)^{-1}(e - h) 
$$


### General Linear Systems

A more general version of the problem described above looks as follows.

```{math}
:label: la_se

\begin{matrix}
    a_{11} x_1 & + & a_{12} x_2 & + & \cdots & + & a_{1n} x_n & = & b_1 \\
    \vdots & & \vdots & & & & \vdots & & \vdots \\
    a_{n1} x_1 & + & a_{n2} x_2 & + & \cdots & + & a_{nn} x_n & = & b_n
\end{matrix}
```

The objective here is to solve for the "unknowns" $x_1, \ldots, x_n$ 

We take as given the coefficients $a_{11}, \ldots, a_{nn}$ and constants $b_1, \ldots, b_n$.

Notice that we are treating a setting where the number of unknowns equals the
number of equations.

This is the case where we are most likely to find a well-defined solution.

(The other cases are referred to as [overdetermined](https://en.wikipedia.org/wiki/Overdetermined_system) and [underdetermined](https://en.wikipedia.org/wiki/Underdetermined_system) systems
of equations --- we defer discussion of these cases until later lectures.)

In matrix form, the system {eq}`la_se` becomes

```{math}
:label: la_gf
    A x = b
    \quad \text{where} \quad
    A = 
    \begin{bmatrix}
        a_{11} &  \cdots & a_{1n} \\
        \vdots & \vdots  & \vdots \\
        a_{n1} &  \cdots & a_{nn}
    \end{bmatrix}
    \quad \text{and} \quad
    b =
    \begin{bmatrix}
        b_1 \\
        \vdots \\
        b_n
    \end{bmatrix}
```

For example, {eq}`n_eq_sys_la` has this form with 

$$ 
    A = D - C, 
    \quad
    b = e - h
    \quad \text{and} \quad
    x = p
$$


When considering problems such as {eq}`la_gf`, we need to ask at least some of the following questions

* Does a solution actually exist?
* If a solution exists, how should we compute it?

## Solving Systems of Equations

```{index} single: Matrix; Solving Systems of Equations
```

Recall again the system of equations {eq}`la_se`, which we write here again as

```{math}
:label: la_se2
    A x = b
```

The problem we face is to find a vector $x \in \mathbb R^n$ that solves
{eq}`la_se2`, taking $b$ and $A$ as given.

We may not always find a unique vector $x$ that solves {eq}`la_se2`.

We illustrate two such cases below.

### No Solution

Consider the system of equations given by,

$$
\begin{aligned}
    x + 3y = 3 \\
    2x + 6y = -8
\end{aligned}
$$

It can be verified manually that this system has no possible solution.

To illustrate why this situation arises let's plot the two lines.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 4))
x = np.linspace(-10,10)
plt.plot(x, (3-x)/3, label=f'$x + 3y = 3$')
plt.plot(x, (-8-2*x)/6, label=f'$2x + 6y = -8$')
plt.legend()
plt.show()
```

<!-- #region -->
Clearly, these are parallel lines and hence we will never find a point $x \in \mathbb{R}^2$
such that these lines intersect.

Thus, this system has no possible solution.

We can rewrite this system in matrix form as

```{math}
:label: no_soln
    A x = b
    \quad \text{where} \quad
    A =
    \begin{bmatrix}
        1 & 3 \\
        2 & 6 
    \end{bmatrix}
    \quad \text{and} \quad
    b =
    \begin{bmatrix}
        3 \\
        -8
    \end{bmatrix}
```

It can be noted that the $2^{nd}$ row of matrix $A = (2, 6)$ is just a scalar multiple of the $1^{st}$ row of matrix $A = (1, 3)$.

Matrix $A$ in this case is called **linearly dependent.**

Linear dependence arises when one row of a matrix can be expressed as a [linear combination](https://en.wikipedia.org/wiki/Linear_combination)
of the other rows.

A matrix that is **not** linearly dependent is called **linearly independent**.

We will keep our discussion of linear dependence and independence limited but a more detailed and generalized
explanation can be found [here](https://python.quantecon.org/linear_algebra.html#linear-independence).

### Many Solutions

Now consider,

$$
\begin{aligned}
    x - 2y = -4 \\
    -2x + 4y = 8
\end{aligned}
$$

Any vector $v = (x,y)$ such that $x = 2y - 4$ will solve the above system.

Since we can find infinite such vectors this system has infinitely many solutions.

Check whether the matrix

```{math}
:label: many_solns
    A =
    \begin{bmatrix}
        1 & -2 \\
        -2 & 4
    \end{bmatrix}
```

is linearly dependent or independent.

We can now impose conditions on $A$ in {eq}`la_se2` that rule out these problems.

### Nonsingular Matrices

To every square matrix we can
assign a unique number called the *determinant* of the matrix --- you can find
the expression for it [here](https://en.wikipedia.org/wiki/Determinant).

For $2 \times 2$ matrices, the determinant is given by,

$$
\begin{bmatrix}
    \color{red}{a} & \color{blue}{b} \\
    \color{blue}{c} & \color{red}{d}
\end{bmatrix}
=
{\color{red}{ad}} - {\color{blue}{bc}}
$$

If the determinant of $A$ is not zero, then we say that $A$ is
*nonsingular*.

A square matrix $A$ is nonsingular if and only if $A$ is linearly independent.

You can check yourself that the linearly dependent matrices in {eq}`no_soln` and {eq}`many_solns` are singular matrices
as well.

This gives us a useful one-number summary of whether or not a square matrix can be
inverted.

In particular, a square matrix $A$ has a nonzero determinant, if and only if it possesses an 
*inverse matrix* $A^{-1}$, with the property that $A A^{-1} = A^{-1} A = I$.

As a consequence, if we pre-multiply both sides of $Ax = b$ by $A^{-1}$, we
get
```{math}
:label: la_se_inv
    x = A^{-1} b.
```

This is the solution that we're looking for.


### Linear Equations with NumPy

```{index} single: Linear Algebra; SciPy
```

In the two good example we obtained the matrix equation,

$$
p = (C-D)^{-1} h
$$

where $C$, $D$ and $h$ are given by {eq}`two_eq_demand_mat` and {eq}`two_eq_supply_mat`.

This equation is analogous to {eq}`la_se_inv` with $A = (C-D)^{-1}$, $b = h$, and $x = p$.

We can now solve for equilibrium prices with NumPy's `linalg` submodule.

All of these routines are Python front ends to time-tested and highly optimized FORTRAN code.
<!-- #endregion -->

```python
C = ((10, 5),      #matrix C
     (5, 10))
```

Now we change this to a NumPy array.

```python
C = np.array(C)
```

```python
D = ((-10, -5),     #matrix D
     (-1, -10))
D = np.array(D)
```

```python
h = np.array((100, 50))   #vector h
h.shape = 2,1             #transforming h to a column vector
```

```python
from numpy.linalg import det, inv
A = C - D
det(A) # check that A is nonsingular (non-zero determinant), and hence invertible
```

```python
A_inv = inv(A) # Compute the inverse
A_inv
```

```python
p = A_inv @ h # equilibrium prices
p
```

```python
q = C @ p # equilibrium quantities
q
```

Notice that we get the same solutions as the pencil and paper case.

We can also solve for $p$ using `solve(A, h)` as follows.

```python
from numpy.linalg import solve
p = solve(A, h) # equilibrium prices
p
```

```python
q = C @ p # equilibrium quantities
q
```

<!-- #region -->
Observe how we can solve for $x = A^{-1} y$ by either via `inv(A) @ y`, or using `solve(A, y)`.

The latter method uses a different algorithm that is numerically more stable and hence should be the default option.

NOTE Add more examples.  Perhaps Tom has suggestions.

NOTE Perhaps discuss LU decompositions in a very simple way?





### Further Reading

The documentation of the `numpy.linalg` submodule can be found [here](https://numpy.org/devdocs/reference/routines.linalg.html).

More advanced topics in linear algebra can be found [here](https://python.quantecon.org/linear_algebra.html#id5).

NOTE Add more references.

NOTE Add exercises.

## Exercises

```{exercise-start}
:label: lin_eqs_ex1
```

Let's consider a market with 3 commodities - good 0, good 1 and good 2.

The demand of each good depends on the price of the other two goods and is given by:

$$
\begin{aligned}
    q_0^d & = 90 - 15p_0 + 5p_1 + 5p_2 \\
    q_1^d & = 60 + 5p_0 - 10p_1 + 10p_2 \\
    q_2^d & = 50 + 5p_0 + 5p_1 - 5p_2
\end{aligned}
$$

(Here demand decreases when own price increases but increases when prices of other goods increase.)

The supply of each good is given by:

$$
\begin{aligned}
    q_0^s & = -10 + 20p_0 \\
    q_1^s & = -15 + 15p_1 \\
    q_2^s & =  -5 + 10p_2
\end{aligned}
$$

Equilibrium holds when supply equals demand, i.e, $q_0^d = q_0^s$, $q_1^d = q_1^s$ and $q_2^d = q_2^s$.

1. Set up the market as a system of linear equations.
2. Use matrix algebra to solve for equilibrium prices. Do this using both the `numpy.linalg.solve`
   and `inv(A)` methods. Compare the solutions.

```{exercise-end}
```


```{exercise-start}
:label: lin_eqs_ex2
```
Earlier in the lecture we discussed cases where the system of equations given by $Ax = b$ has no solution.

In this case $Ax = b$ is called an _inconsistent_ system of equations.

When faced with an inconsistent system we try to find the best "approximate" solution.

There are various methods to do this, one such method is the **method of least squares.**

Suppose we have an inconsistent system 

```{math}
:label: inconsistent
    Ax = b
```
where $A$ is an $m \times n$ matrix and $b$ is an $m \times 1$ column vector.

A **least squares solution** to {eq}`inconsistent` is an $n \times 1$ column vector $\hat{x}$ such that for all other
vectors $x \in \mathbb{R}^n$

$$
\begin{aligned}
distance(A\hat{x} - b) & \leq distance(Ax - b) \\
\|A\hat{x} - b\| & \leq \|Ax - b\| \\
\|A\hat{x} - b\|^2 & \leq \|Ax - b\|^2 \\
(A\hat{x}_1 - b_1)^2 + (A\hat{x}_2 - b_2)^2 + \cdots + (A\hat{x}_m - b_m)^2 & \leq
(Ax_1 - b_1)^2 + (Ax_2 - b_2)^2 + \cdots + (Ax_m - b_m)^2
\end{aligned}
$$

We will not detail how the following expression is obtained but for a system of equations $Ax = b$ the least squares solution
$\hat{x}$ is given by:

```{math}
:label: least_squares

    \begin{aligned}
        {A^T} A \hat{x} &  = {A^T} b \\
        \hat{x} & = (A^T A)^{-1} A^T b
    \end{aligned}
```

Consider the general equation of a linear demand curve of a good given by:
$$
p = a - bq
$$
where $p$ is the price of the good and $q$ is the quantity demanded.

We have observed prices and quantities for a certain good and are trying to find the values for $a$ and $b$.

We have the following observations:
| Price | Quantity Demanded |
|:-----:|:-----------------:|
|   1   |         9         |
|   3   |         7         |
|   8   |         3         |

Since the demand curve should pass through all these points we have the following three equations:

$$
\begin{aligned}
    1 = a - 9b \\
    3 = a - 7b \\
    8 = a - 3b
\end{aligned}
$$

Thus we obtain a system of equations $Ax = b$ where $A = \begin{bmatrix} 1 & -9 \\ 1 & -7 \\ 1 & -3 \end{bmatrix}$,
$x = \begin{bmatrix} a \\ b \end{bmatrix}$ and $b = \begin{bmatrix} 1 \\ 3 \\ 8 \end{bmatrix}$.

It can be easily verified this system has no solutions.

We will thus try to find the best approximate solution for $x$.

1. Use {eq}`least_squares` and matrix algebra to find the least squares solution $\hat{x}$.
2. Find the least squares solution using `numpy.linalg.lstsq` and compare the results.

```{exercise-end}
```



## Solutions

```{solution-start} lin_eqs_ex1
:class: dropdown
```

The generated system would be:

$$
\begin{aligned}
    35p_0 - 5p_1 - 5p_2 = 100 \\
    -5p_0 + 25p_1 - 10p_2 = 75 \\
    -5p_0 - 5p_1 + 15p_2 = 55
\end{aligned}
$$

In matrix form we will write this as:

$$
Ap = b
\quad \text{where} \quad
A =
\begin{bmatrix}
    35 & -5 & -5 \\
    -5 & 25 & -10 \\
    -5 & -5 & 15
\end{bmatrix}
, \quad p =
\begin{bmatrix}
    p_0 \\
    p_1 \\
    p_2
\end{bmatrix}
\quad \text{and} \quad
b = 
\begin{bmatrix}
    100 \\
    75 \\
    55
\end{bmatrix}
$$
<!-- #endregion -->

```python
import numpy as np
from numpy.linalg import det

A = np.array([[35, -5, -5],        #matrix A
              [-5, 25, -10],
              [-5, -5, 15]])

b = np.array((100, 75, 55))        #column vector b
b.shape = (3,1)

det(A)    #check if A is nonsingular
```

```python
#using inverse
from numpy.linalg import det

A_inv = inv(A)

p = A_inv @ b
p
```

```python
#using numpy.linalg.solve
from numpy.linalg import solve
p = solve(A,b)
p
```

Th solution is given by:
$$
p_0 = 4.6925, \; p_1 = 7.0625 \;\; \text{and} \;\; p_2 = 7.675
$$

```{solution-end}
```

```{solution-start} lin_eqs_ex2
:class: dropdown
```

```python
import numpy as np
from numpy.linalg import inv
```

```python
#using matrix algebra
A = np.array([[1, -9],    #matrix A
              [1, -7],
              [1, -3]])

A_T = np.transpose(A)    #transpose of matrix A

b = np.array((1, 3, 8))    #column vector b
b.shape = (3,1)

x = inv(A_T @ A) @ A_T @ b
x
```

```python
#using numpy.linalg.lstsq
x = np.linalg.lstsq(A, b, rcond = None)
x
```

Here is a visualization of how the least squares method approximates the equation of a line connecting a set of points.

We can also describe this as "fitting" a line between a set of points.

```python
fig, ax = plt.subplots(figsize=(6, 6))
p = np.array((1, 3, 8))
q = np.array((9, 7 ,3))

a, b = x[0]

plt.plot(q, p, 'o', label='observations', markersize=5)
plt.plot(q, a - b*q, 'r', label='Fitted line')
plt.xlabel('quantity demanded')
plt.ylabel('price')
plt.legend()
plt.show()
```

```{solution-end}
```

<!-- #region tags=[] -->
```{solution-end}
```

solution end
<!-- #endregion -->
