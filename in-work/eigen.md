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


# Eigenvalues and Eigenvectors 

```{index} single: Eigenvalues and Eigenvectors
```

```{contents} Contents
:depth: 2
```

## Overview

Eigenvalues and eigenvectors are a somewhat advanced topic in for students who
are studying linear algebra for the first time.

At the same time, these concepts are extremely useful for economic modeling,
particularly when we start to talk about dynamics.

In this lecture we explain the basics of eigenvalues and eigenvectors, and
state two very important results from linear algebra.

The first is called the Neumann series theorem and the second is called the
Perron-Frobenius theorem.

We will explain what these theorems tell us and how we can use them to
understand the predictions of economic models.

We assume in this lecture that students are familiar with matrices and
understand the basics of matrix algebra.

We will use the following imports:


```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
```





(la_eigen)=
## Matrices as Transformations

NOTE

    Explain that matrices can be viewed as maps.
    Focus on the n times n case.
    Give some visualizations of transformations (rotating, stretching, etc.)
    Explain that matrix multiplication is just composition of the maps.
    Give some visualizations of iterating with a fixed map.



(la_eigen)=
## Eigenvectors 

```{index} single: Linear Algebra; Eigenvalues
```

Let $A$ be an $n \times n$ square matrix.

If $\lambda$ is scalar and $v$ is a non-zero vector in $\mathbb R^n$ such that

$$
A v = \lambda v
$$

then we say that $\lambda$ is an *eigenvalue* of $A$, and
$v$ is an *eigenvector*.

Thus, an eigenvector of $A$ is a vector such that when the map $f(x) = Ax$ is applied, $v$ is merely scaled.

The next figure shows two eigenvectors (blue arrows) and their images under $A$ (red arrows).

As expected, the image $Av$ of each $v$ is just a scaled version of the original

```{code-cell} python3
---
tags: [output_scroll]
---
A = ((1, 2),
     (2, 1))
A = np.array(A)
evals, evecs = eig(A)
evecs = evecs[:, 0], evecs[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')
ax.grid(alpha=0.4)

xmin, xmax = -3, 3
ymin, ymax = -3, 3
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

# Plot each eigenvector
for v in evecs:
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='blue',
                shrink=0,
                alpha=0.6,
                width=0.5))

# Plot the image of each eigenvector
for v in evecs:
    v = A @ v
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='red',
                shrink=0,
                alpha=0.6,
                width=0.5))

# Plot the lines they run through
x = np.linspace(xmin, xmax, 3)
for v in evecs:
    a = v[1] / v[0]
    ax.plot(x, a * x, 'b-', lw=0.4)

plt.show()
```

The eigenvalue equation is equivalent to $(A - \lambda I) v = 0$, and
this has a nonzero solution $v$ only when the columns of $A -
\lambda I$ are linearly dependent.

This in turn is equivalent to stating that the determinant is zero.

Hence to find all eigenvalues, we can look for $\lambda$ such that the
determinant of $A - \lambda I$ is zero.

This problem can be expressed as one of solving for the roots of a polynomial
in $\lambda$ of degree $n$.

This in turn implies the existence of $n$ solutions in the complex
plane, although some might be repeated.

Some nice facts about the eigenvalues of a square matrix $A$ are as follows

1. The determinant of $A$ equals  the product of the eigenvalues.
1. The trace of $A$ (the sum of the elements on the principal diagonal) equals the sum of the eigenvalues.
1. If $A$ is symmetric, then all of its eigenvalues are real.
1. If $A$ is invertible and $\lambda_1, \ldots, \lambda_n$ are its eigenvalues, then the eigenvalues of $A^{-1}$ are $1/\lambda_1, \ldots, 1/\lambda_n$.

A corollary of the first statement is that a matrix is invertible if and only if all its eigenvalues are nonzero.

Using SciPy, we can solve for the eigenvalues and eigenvectors of a matrix as
follows

```{code-cell} python3
A = ((1, 2),
     (2, 1))

A = np.array(A)
evals, evecs = eig(A)
evals
```

```{code-cell} python3
evecs
```

Note that the *columns* of `evecs` are the eigenvectors.

Since any scalar multiple of an eigenvector is an eigenvector with the same
eigenvalue (check it), the eig routine normalizes the length of each eigenvector
to one.




### {index}`Positive Definite Matrices <single: Positive Definite Matrices>`

```{index} single: Linear Algebra; Positive Definite Matrices
```

Let $A$ be a symmetric $n \times n$ matrix.

We say that $A$ is

1. *positive definite* if $x' A x > 0$ for every $x \in \mathbb R ^n \setminus \{0\}$
1. *positive semi-definite* or *nonnegative definite* if $x' A x \geq 0$ for every $x \in \mathbb R ^n$

Analogous definitions exist for negative definite and negative semi-definite matrices.

It is notable that if $A$ is positive definite, then all of its eigenvalues
are strictly positive, and hence $A$ is invertible (with positive
definite inverse).


NOTE Add examples, verify that results hold using NumPy



## {index}`The Perron-Frobenius Theorem`

In what follows, a matrix $A$ is called **nonnegative** if every element of $A$ is
nonnegative.

Often, in economics, the matrix that we are dealing with is nonnegative.

Remarkably, as soon as we know that a matrix is nonnegative, we can also state
that all of the eigenvalues and eigenvectors are real-valued rather than
complex (i.e., they have no imaginary part).

NOTE 

* continue describing a simple version of the Perron--Frobenius theorem
* link to it
* use an example of an economy that transitions between "recession" and "normal growth"
* check the claims using NumPy



(la_neumann)=
## {index}`The Neumann Series Theorem <single: Neumann's Theorem>`

```{index} single: Neumann's Theorem
```

Here's a fundamental result about series that you surely know:

If $a$ is a number and $|a| < 1$, then 

$$
    \sum_{k=0}^{\infty} a^k = (1 - a)^{-1} 
$$

A generalization of this idea exists in the matrix setting.

Let $A$ be a square matrix and let $A^k$ be the $k$-th power of $A$.

Let $r(A)$ be the *spectral radius*, defined as $\max_i |\lambda_i|$, where 

* $\{\lambda_i\}_i$ is the set of eigenvalues of $A$ and
* $|\lambda_i|$ is the modulus of the complex number $\lambda_i$

Neumann's theorem states the following: If $r(A) < 1$, then $I - A$ is invertible, and

```{math}
:label: la_neumann

(I - A)^{-1} = \sum_{k=0}^{\infty} A^k
```



NOTE give examples, solve with NumPy

NOTE Add exercises below



## Exercises

```{exercise-start}
:label: eig_ex1
```

Add an exercise

```{exercise-end}
```

## Solutions

```{solution-start} eig_ex1
:class: dropdown
```

Add solution

```{solution-end}
```

