---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
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

Finally, we introduce eigenvalues and show how they can be useful for economic
problems.

We assume that students are familiar with matrices and understand the basics
of matrix algebra.

We will use the following imports:


```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
```



## A Two Good Example

We discuss a simple two good example and solve it by

1. pencil and paper
2. matrix algebra

The second method is more general, as we will see.


### Pencil and Paper Methods

Suppose that we have two goods, such as propane and ethanol.

To keep things simple, we label them as good 0 and good 1.

The demand for each good depends on the price of both goods:

$$
:label: two_eq_demand
\begin{aligned}
    q_0^d = 100 - 10 p_0 - 5 p_1 \\
    q_1^d = 50 - p_0 - 10 p_1 
\end{aligned}
$$

(We are assuming demand decreases when the price of either good goes up.)

Let's suppose that supply is given by 

$$
:label: two_eq_supply
\begin{aligned}
    q_0^s = 10 p_0 + 5 p_1 \\
    q_1^s = 5 p_0 + 10 p_1 
\end{aligned}
$$

Equilibrium holds when supply equals demand ($q_0^s = q_0^d$ and $q_1^s = q_1^d$).

This yields the linear system

$$
\begin{aligned}
    100 - 10 p_0 - 5 p_1 = 100 - 10 p_0 - 5 p_1 \\
    50 - p_0 - 10 p_1 = 50 - p_0 - 10 p_1 
\end{aligned}
$$

We can solve this with pencil and paper to get

$$
    p_0 = xxx and p_1 = yyy
$$

Inserting these results into either two_eq_demand or two_eq_supply yields the
equilibrium quantities

$$
    q_0 = xxx and q_1 = yyy
$$



### Using Matrix Algebra

We can also solve this system numerically via matrix algebra.

This involves some extra steps but the method is widely applicable --- as we
will see when we include more goods.

First we rewrite eq:two_eq_demand as

$$
:label: two_eq_demand_mat
    q^d = D p + h
    \quad \text{where} \quad
    q^d = 
    \begin{pmatrix}
        q_0^d \\
        q_1^d
    \end{pmatrix}
    \quad
    D = 
    \begin{pmatrix}
         - 10 & - 5  \\
         - 1  & - 10  
    \end{pmatrix}
    \quad \text{and} \quad
    h =
    \begin{pmatrix}
        100 \\
        50
    \end{pmatrix}
$$

(Please check that $q^d = D p + h$ represents the same equations as eq:two_eq_demand.)

We rewrite eq:two_eq_supply as

$$
:label: two_eq_supply_mat
    q^s = C p 
    \quad \text{where} \quad
    q^s = 
    \begin{pmatrix}
        q_0^s \\
        q_1^s
    \end{pmatrix}
    \quad \text{and} \quad
    C = 
    \begin{pmatrix}
         10 & 5  \\
         5 & 10  
    \end{pmatrix}
$$

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

$$
    p = (C - D)^{-1} h
$$

Let's do this using NumPy.

```{code-cell} ipython
add code here.
```

Notice that we get the same solutions as above.

NOTE Use inv first and then solve, check that they give the same result, mention
that we discuss this again below.




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

$$ 
:label: n_eq_sys_la
    (D- C)p = e - h 
$$

The solution is

$$ 
    p = (D- C)^{-1}(e - h) 
$$


### General Linear Systems

A more general version of the problem described above looks as follows.

```{math}
:label: la_se

\begin{aligned}
    a_{11} x_1 + a_{12} x_2 + \cdots + a_{1n} x_n = b_1 \\
    \vdots  \\
    a_{n1} x_1 + a_{n2} x_2 + \cdots + a_{nn} x_n = b_n
\end{aligned}
```

The objective here is to solve for the "unknowns" $x_1, \ldots, x_k$ 

We take as given the coefficients $a_{11}, \ldots, a_{nn}$ and constants $b_1, \ldots, b_n$.

Notice that we are treating a setting where the number of unknowns equals the
number of equations.

This is the case where we are most likely to find a well-defined solution.

(The other cases are referred to as overdetermined and underdetermined systems
of equations --- we defer discussion of these cases until later lectures.)

NOTE add wiki links for overdetermined and underdetermined

In matrix form, the system eq:la_se becomes

$$
:label: la_gf
    A x = b
    \quad \text{where} \quad
    A = 
    \begin{pmatrix}
        a_{11} &  \cdots & a_{1n} \\
        \vdots & \vdots  & \vdots \\
        a_{n1} &  \cdots & a_{nn}
    \end{pmatrix}
    \quad \text{and} \quad
    b =
    \begin{pmatrix}
        b_1 \\
        \ldots
        b_n
    \end{pmatrix}
$$

For example, eq:n_eq_sys_la has this form with 

$$ 
    A = D - C, 
    quad
    b = e - h
    \quad \text{and} \quad
    x = p
$$


When considering problems such as eq:la_gf, we need to ask at least some of the following questions

* Does a solution actually exist?
* If a solution exists, how should we compute it?


We work up to answering these questions, starting by reviewing some basics.

Readers familiar with these basic facts can skim them quickly.


## {index}`Vectors <single: Vectors>`

```{index} single: Linear Algebra; Vectors
```

A **vector** of length $n$ is just a sequence (or array, or tuple) of $n$ numbers, which we write as $x = (x_1, \ldots, x_n)$.

We can write these sequences either horizontally or vertically.

But when we use matrix operations, our default assumption is that vectors are column vectors.

The set of all $n$-vectors is denoted by $\mathbb R^n$.

For example, $\mathbb R ^2$ is the plane and a vector in $\mathbb R^2$ is just a point in the plane.

Traditionally, vectors are represented visually as arrows from the origin to the point.

The following figure represents three vectors in this manner

```{code-cell} ipython
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

When we add two vectors, we add them element-by-element

$$
x + y =
\begin{pmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
\end{pmatrix} +
\begin{pmatrix}
     y_1 \\
     y_2 \\
    \vdots \\
     y_n
\end{pmatrix} :=
\begin{pmatrix}
    x_1 + y_1 \\
    x_2 + y_2 \\
    \vdots \\
    x_n + y_n
\end{pmatrix}
$$

Scalar multiplication is an operation that takes a number $\gamma$ and a
vector $x$ and produces

$$
\gamma x :=
\begin{pmatrix}
    \gamma x_1 \\
    \gamma x_2 \\
    \vdots \\
    \gamma x_n
\end{pmatrix}
$$

Scalar multiplication is illustrated in the next figure


```{code-cell} python3
---
tags: [output_scroll]
---
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

However, it is more common to represented vectors with [NumPy arrays](https://python-programming.quantecon.org/numpy.html#numpy-arrays).

One advantage of NumPy arrays is that scalar multiplication and addition have very natural syntax

```{code-cell} python3
x = np.ones(3)            # Vector of three ones
y = np.array((2, 4, 6))   # Converts tuple (2, 4, 6) into a NumPy array
x + y                     # Add (element-by-element)
```

```{code-cell} python3
4 * x                     # Scalar multiply
```


### Inner Product and Norm

```{index} single: Vectors; Inner Product
```

```{index} single: Vectors; Norm
```

The **inner product** of vectors $x,y \in \mathbb R ^n$ is defined as

$$
x' y := \sum_{i=1}^n x_i y_i
$$

The **norm** of a vector $x$ represents its "length" (i.e., its distance from the zero vector) and is defined as

$$
\| x \| := \sqrt{x' x} := \left( \sum_{i=1}^n x_i^2 \right)^{1/2}
$$

The expression $\| x - y\|$ can be thought of as the "distance" between $x$ and $y$.

The inner product and norm can be computed as follows

```{code-cell} python3
np.sum(x * y)          # Inner product of x and y
```

```{code-cell} python3
np.sqrt(np.sum(x**2))  # Norm of x, take one
```

```{code-cell} python3
np.linalg.norm(x)      # Norm of x, take two
```



## Matrix Operations

```{index} single: Matrix; Operations
```

When we discussed linear price systems, we mentioned that matrix algebra is
similar to algebra for numbers.

Let's review some details.

Just as was the case for vectors, we can add, subtract and scalar multiply
matrices.

Scalar multiplication and addition are generalizations of the vector case:

$$
\gamma A =
\gamma
\begin{pmatrix}
    a_{11} &  \cdots & a_{1k} \\
    \vdots & \vdots  & \vdots \\
    a_{n1} &  \cdots & a_{nk}
\end{pmatrix} :=
\begin{pmatrix}
    \gamma a_{11} & \cdots & \gamma a_{1k} \\
    \vdots & \vdots & \vdots \\
    \gamma a_{n1} & \cdots & \gamma a_{nk}
\end{pmatrix}
$$

and

$$
A + B =
\begin{pmatrix}
    a_{11} & \cdots & a_{1k} \\
    \vdots & \vdots & \vdots \\
    a_{n1} & \cdots & a_{nk}
\end{pmatrix} +
\begin{pmatrix}
    b_{11} & \cdots & b_{1k} \\
    \vdots & \vdots & \vdots \\
    b_{n1} & \cdots & b_{nk}
\end{pmatrix} :=
\begin{pmatrix}
    a_{11} + b_{11} &  \cdots & a_{1k} + b_{1k} \\
    \vdots & \vdots & \vdots \\
    a_{n1} + b_{n1} &  \cdots & a_{nk} + b_{nk}
\end{pmatrix}
$$

In the latter case, the matrices must have the same shape in order for the
definition to make sense.

We also have a convention for *multiplying* two matrices.

The rule for matrix multiplication generalizes the idea of inner products
discussed above.

If $A$ and $B$ are two matrices, then their product $A B$ is formed by taking
as its $i,j$-th element the inner product of the $i$-th row of $A$ and the
$j$-th column of $B$.

There are many tutorials to help you visualize this operation, such as [this
one](http://www.mathsisfun.com/algebra/matrix-multiplying.html), or the
discussion on the [Wikipedia page](https://en.wikipedia.org/wiki/Matrix_multiplication).

If $A$ is $n \times k$ and $B$ is $j \times m$, then to multiply $A$ and $B$
we require $k = j$, and the resulting matrix $A B$ is $n \times m$.

As perhaps the most important special case, consider multiplying $n \times k$
matrix $A$ and $k \times 1$ column vector $x$.

According to the preceding rule, this gives us an $n \times 1$ column vector

```{math}
:label: la_atx

A x =
\begin{pmatrix}
    a_{11} &  \cdots & a_{1k} \\
    \vdots & \vdots  & \vdots \\
    a_{n1} &  \cdots & a_{nk}
\end{pmatrix}
\begin{pmatrix}
    x_{1}  \\
    \vdots  \\
    x_{k}
\end{pmatrix} :=
\begin{pmatrix}
    a_{11} x_1 + \cdots + a_{1k} x_k \\
    \vdots \\
    a_{n1} x_1 + \cdots + a_{nk} x_k
\end{pmatrix}
```

```{note}
Unlike number products, $A B$ and $B A$ are not generally the same thing.
```

Another important special case is the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix)

$$
    I = 
    \begin{pmatrix}
        1 &  \cdots & 0 \\
        \vdots & \vdots  & \vdots \\
        0 &  \cdots & 1
    \end{pmatrix}
$$

that has ones on the principle diagonal and zero elsewhere.

It is a useful exercise to check the following:

* If $A$ is $n \times k$ and $I$ is the $k \times k$ identity matrix, then $AI = A$.
* If $I$ is the $n \times n$ identity matrix, then $IA = A$.



### Matrices in NumPy

```{index} single: Matrix; Numpy
```

NumPy arrays are also used as matrices, and have fast, efficient functions and methods for all the standard matrix operations.

You can create them manually from tuples of tuples (or lists of lists) as follows

```{code-cell} python3
A = ((1, 2),
     (3, 4))

type(A)
```

```{code-cell} python3
A = np.array(A)

type(A)
```

```{code-cell} python3
A.shape
```

The `shape` attribute is a tuple giving the number of rows and columns ---
see [here](https://python-programming.quantecon.org/numpy.html#shape-and-dimension)
for more discussion.

To get the transpose of `A`, use `A.transpose()` or, more simply, `A.T`.

There are many convenient functions for creating common matrices (matrices of zeros,
ones, etc.) --- see [here](https://python-programming.quantecon.org/numpy.html#creating-arrays).

Since operations are performed elementwise by default, scalar multiplication and addition have very natural syntax

```{code-cell} python3
A = np.identity(3)
B = np.ones((3, 3))
2 * A
```

```{code-cell} python3
A + B
```

To multiply matrices we use the `@` symbol.


```{note}
In particular, `A @ B` is matrix multiplication, whereas `A * B` is element-by-element multiplication.
```


## Solving Systems of Equations

```{index} single: Matrix; Solving Systems of Equations
```

Recall again the system of equations {eq}`la_se`, which we write here again as

$$
:label: la_se2
    A x = b
$$

The problem we face is to find a vector $x \in \mathbb R^n$ that solves
{eq}`la_se2`, taking $b$ and $A$ as given.

GIVE EXAMPLES WHERE 

* no solutions exist
* many solutions exist

Can we impose conditions on $A$ in {eq}`la_se2` that rule out these problems?

DISCUSS NONZERO DETERMINANT

To every square matrix we
assign a unique number called the *determinant* of the matrix --- you can find
the expression for it [here](https://en.wikipedia.org/wiki/Determinant).

If the determinant of $A$ is not zero, then we say that $A$ is
*nonsingular*.

Perhaps the most important fact about determinants is that $A$ is nonsingular if and only if $A$ is of full column rank.

This gives us a useful one-number summary of whether or not a square matrix can be
inverted.

In particular, if square matrix $A$ has a nonzero determinant, then it possesses an 
*inverse matrix* $A^{-1}$, with the property that $A A^{-1} = A^{-1} A = I$.

As a consequence, if we pre-multiply both sides of $Ax = b$ by $A^{-1}$, we
get $x = A^{-1} b$.

This is the solution that we're looking for.





### Linear Equations with SciPy

NOTE: CHANGE ALL TO numpy.linalg
      Swap SciPy for NumPy


```{index} single: Linear Algebra; SciPy
```

Let's review how to solve linear equations with SciPy's `linalg` submodule.

All of these routines are Python front ends to time-tested and highly optimized FORTRAN code

```{code-cell} python3
A = ((1, 2), (3, 4))
A = np.array(A)
y = np.ones((2, 1))  # Column vector
det(A)  # Check that A is nonsingular, and hence invertible
```

```{code-cell} python3
A_inv = inv(A)  # Compute the inverse
A_inv
```

```{code-cell} python3
x = A_inv @ y  # Solution
A @ x          # Should equal y
```

```{code-cell} python3
solve(A, y)  # Produces the same solution
```

Observe how we can solve for $x = A^{-1} y$ by either via `inv(A) @ y`, or using `solve(A, y)`.

The latter method uses a different algorithm that is numerically more stable and hence should be the default option.

NOTE Add more examples.  Perhaps Tom has suggestions.

NOTE Perhaps discuss LU decompositions in a very simple way?





### Further Reading

The documentation of the `scipy.linalg` submodule can be found [here](http://docs.scipy.org/doc/scipy/reference/linalg.html).

NOTE Add more references.

NOTE Add exercises.

## Exercises

```{exercise-start}
:label: lin_eqs_ex1
```

Add an exercise

```{exercise-end}
```

## Solutions

```{solution-start} simple_la_ex1
:class: dropdown
```

Add solution

```{solution-end}
```

