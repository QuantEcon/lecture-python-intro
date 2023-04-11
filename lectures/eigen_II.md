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

# Theorems of Nonnegative Matrices and Eigenvalues

```{index} single: Eigenvalues and Eigenvectors
```

```{contents} Contents
:depth: 2
```

In this lecture we will begin with the basic properties of nonnegative matrices.

Then we will explore the Perron-Frobenius Theorem and the Neumann Series Lemma, and connect them to applications in Markov chains and networks. 

We will use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
import graphviz as gv
```

## Nonnegative Matrices

Often, in economics, the matrix that we are dealing with is nonnegative.

Nonnegative matrices have several special and useful properties.

In this section we discuss some of them --- in particular, the connection
between nonnegativity and eigenvalues.

Let $a^{k}_{ij}$ be element $(i,j)$ of $A^k$.

An $n \times m$ matrix $A$ is called **nonnegative** if every element of $A$
is nonnegative, i.e., $a_{ij} \geq 0$ for every $i,j$.

We denote this as $A \geq 0$.

(irreducible)=
### Irreducible Matrices

We have (informally) introduced irreducible matrices in the Markov chain lecture (TODO: link to Markov chain lecture).

Here we will introduce this concept formally.

$A$ is called **irreducible** if for *each* $(i,j)$ there is an integer $k \geq 0$ such that $a^{k}_{ij} > 0$.

A matrix $A$ that is not irreducible is called reducible.

Here are some examples to illustrate this further.

1. $A = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.2 \end{bmatrix}$ is irreducible since $a_{ij}>0$ for all $(i,j)$.

2. $A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ is irreducible since $a_{12},a_{21} >0$ and $a^{2}_{11},a^{2}_{22} >0$.

3. $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ is reducible since $A^k = A$ for all $k \geq 0$ and thus
   $a^{k}_{12},a^{k}_{21} = 0$ for all $k \geq 0$.

### Primitive Matrices

Let $A$ be a square nonnegative matrix and let $A^k$ be the $k^{th}$ power of $A$.

A matrix is consisdered **primitive** if there exists a $k \in \mathbb{N}$ such that $A^k$ is everywhere positive.

It means that $A$ is called primitive if there is an integer $k \geq 0$ such that $a^{k}_{ij} > 0$ for *all* $(i,j)$.

We can see that if a matrix is primitive, then it implies the matrix is irreducible.

This is becuase if there exists an $A^k$ such that $a^{k}_{ij} > 0$ for all $(i,j)$, then it guarantees the same property for ${k+1}^th, {k+2}^th ... {k+n}^th$ iterations.

In other words, a primitive matrix is both irreducible and aperiodical as aperiodicity requires the a state to be visited with a guarantee of returning to itself after certain amount of iterations.

### Left Eigenvectors

We have previously discussed right (ordinary) eigenvectors $Av = \lambda v$.

Here we introduce left eigenvectors.

Left eigenvectors will play important roles in what follows, including that of stochastic steady states for dynamic models under a Markov assumption.

We will talk more about this later, but for now, let's define left eigenvectors.

A vector $\varepsilon$ is called a left eigenvector of $A$ if $\varepsilon$ is an eigenvector of $A^T$.

In other words, if $\varepsilon$ is a left eigenvector of matrix A, then $A^T \varepsilon = \lambda \varepsilon$, where $\lambda$ is the eigenvalue associated with the left eigenvector $v$.

This hints on how to compute left eigenvectors

```{code-cell} ipython3
# Define a sample matrix
A = np.array([[3, 2], 
              [1, 4]])

# Compute right eigenvectors and eigenvalues
right_eigenvalues, right_eigenvectors = np.linalg.eig(A)

# Compute left eigenvectors and eigenvalues
left_eigenvalues, left_eigenvectors = np.linalg.eig(A.T)

# Transpose left eigenvectors for comparison (because they are returned as column vectors)
left_eigenvectors = left_eigenvectors.T

print("Matrix A:")
print(A)
print("\nRight Eigenvalues:")
print(right_eigenvalues)
print("\nRight Eigenvectors:")
print(right_eigenvectors)
print("\nLeft Eigenvalues:")
print(left_eigenvalues)
print("\nLeft Eigenvectors:")
print(left_eigenvectors)
```

Note that the eigenvalues for both left and right eigenvectors are the same, but the eigenvectors themselves are different.

We can then take transpose to obtain $A^T \varepsilon = \lambda \varepsilon$ and obtain $\varepsilon^T A= \lambda \varepsilon^T$.

This is a more common expression and where the name left eigenvectors originates.

(perron-frobe)=
### The Perron-Frobenius Theorem

For a nonnegative matrix $A$ the behaviour of $A^k$ as $k \to \infty$ is controlled by the eigenvalue with the largest
absolute value, often called the **dominant eigenvalue**.

For a matrix $A$, the Perron-Frobenius theorem characterises certain
properties of the dominant eigenvalue and its corresponding eigenvector when
$A$ is a nonnegative square matrix.

```{prf:theorem} Perron-Frobenius Theorem
:label: perron-frobenius

If a matrix $A \geq 0$ then,

1. the dominant eigenvalue of $A$, $r(A)$, is real-valued and nonnegative. 
2. for any other eigenvalue (possibly complex) $\lambda$ of $A$, $|\lambda| \leq r(A)$.
3. we can find a nonnegative and nonzero eigenvector $v$ such that $Av = r(A)v$.

Moreover if $A$ is also irreducible then,

4. the eigenvector $v$ associated with the eigenvalue $r(A)$ is strictly positive.
5. there exists no other positive eigenvector $v$ (except scalar multiples of $v$) associated with $r(A)$.

If $A$ is primitive then,

6. the inequality $|\lambda| \leq r(A)$ is strict for all eigenvalues $\lambda$ of $A$ distinct from $r(A)$, and
7. with $e$ and $\varepsilon$ normalized so that the inner product of $\varepsilon$ and  $e = 1$, we have
$ r(A)^{-m} A^m$ converges to $\varepsilon^{\top}$ when $m \rightarrow \infty$
```

(This is a relatively simple version of the theorem --- for more details see
[here](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)).

We will see applications of the theorem below.

(la_neumann)=
## The Neumann Series Lemma 

```{index} single: Neumann's Lemma
```

In this section we present a famous result about series of matrices that has
many applications in economics.

### Scalar Series

Here's a fundamental result about series that you surely know:

If $a$ is a number and $|a| < 1$, then 

```{math}
:label: gp_sum
    
    \sum_{k=0}^{\infty} a^k =\frac{1}{1-a} = (1 - a)^{-1} 

```

For a one-dimensional linear equation $x = ax + b$ where x is unknown we can thus conclude that the solution $x^{*}$ is given by:

$$
    x^{*} = \frac{b}{1-a} = \sum_{k=0}^{\infty} a^k b
$$

### Matrix Series

A generalization of this idea exists in the matrix setting.

Consider the system of equations $x = Ax + b$ where $A$ is an $n \times n$
square matrix and $x$ and $b$ are both column vectors in $\mathbb{R}^n$.

Using matrix algebra we can conclude that the solution to this system of equations will be given by:

```{math}
:label: neumann_eqn
    
    x^{*} = (I-A)^{-1}b

```

What guarantees the existence of a unique vector $x^{*}$ that satisfies
{eq}`neumann_eqn` ?

The following is a fundamental result in functional analysis that generalises
{eq}`gp_sum` to a multivariate case.

(neumann_series_lemma)=
```{prf:theorem} Neumann Series Lemma
:label: neumann_series_lemma

Let $A$ be a square matrix and let $A^k$ be the $k$-th power of $A$.

Let $r(A)$ be the dominant eigenvector or as it is commonly called the *spectral radius*, defined as $\max_i |\lambda_i|$, where 

* $\{\lambda_i\}_i$ is the set of eigenvalues of $A$ and
* $|\lambda_i|$ is the modulus of the complex number $\lambda_i$

Neumann's theorem states the following: If $r(A) < 1$, then $I - A$ is invertible, and

$$
(I - A)^{-1} = \sum_{k=0}^{\infty} A^k
$$
```

We can see the Neumann series lemma in action in the following example.

```{code-cell} ipython3
A = np.array([[0.4, 0.1],
              [0.7, 0.2]])

evals, evecs = eig(A)   # finding eigenvalues and eigenvectors

r = max(abs(λ) for λ in evals)    # compute spectral radius
print(r)
```

The spectral radius $r(A)$ obtained is less than 1. 

Thus, we can apply the Neumann Series lemma to find $(I-A)^{-1}$.

```{code-cell} ipython3
I = np.identity(2)      #2 x 2 identity matrix
B = I - A
```

```{code-cell} ipython3
B_inverse = np.linalg.inv(B)     #direct inverse method
```

```{code-cell} ipython3
A_sum = np.zeros((2,2))        #power series sum of A
A_power = I
for i in range(50):
    A_sum += A_power
    A_power = A_power @ A
```

Let's check equality between the sum and the inverse methods.

```{code-cell} ipython3
np.allclose(A_sum, B_inverse)     
```

Although we truncate the infinite sum at $k = 50$, both methods give us the same
result which illustrates the result of the Neumann Series lemma.

## Exercises

```{exercise-start} Leontief's Input-Output Model
:label: eig_ex1
```
[Wassily Leontief](https://en.wikipedia.org/wiki/Wassily_Leontief) developed a model of an economy with $n$ sectors producing $n$ different commodities representing the interdependencies of different sectors of an economy.

Under this model some of the output is consumed internally by the industries and the rest is consumed by external consumers.

We define a simple model with 3 sectors - agriculture, industry, and service.

The following table describes how output is distributed within the economy:

|             | Total output | Agriculture | Industry | Service | Consumer |
|:-----------:|:------------:|:-----------:|:--------:|:-------:|:--------:|
| Agriculture |     $x_1$    |   0.3$x_1$  | 0.2$x_2$ |0.3$x_3$ |     4    |
|   Industry  |     $x_2$    |   0.2$x_1$  | 0.4$x_2$ |0.3$x_3$ |     5    |
|   Service   |     $x_3$    |   0.2$x_1$  | 0.5$x_2$ |0.1$x_3$ |    12    |

The first row depicts how agriculture's total output $x_1$ is distributed 

* $0.3x_1$ is used as inputs within agriculture itself,
* $0.2x_2$ is used as inputs by the industry sector to produce $x_2$ units
* $0.3x_3$ is used as inputs by the service sector to produce $x_3$ units and 
* 4 units is the external demand by consumers.

We can transform this into a system of linear equations for the 3 sectors as
given below:

$$
    x_1 = 0.3x_1 + 0.2x_2 + 0.3x_3 + 4 \\
    x_2 = 0.2x_1 + 0.4x_2 + 0.3x_3 + 5 \\
    x_3 = 0.2x_1 + 0.5x_2 + 0.1x_3 + 12
$$

This can be transformed into the matrix equation $x = Ax + d$ where

$$
x =
\begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3
\end{bmatrix}
, \; A =
\begin{bmatrix}
    0.3 & 0.2 & 0.3 \\
    0.2 & 0.4 & 0.3 \\
    0.2 & 0.5 & 0.1
\end{bmatrix}
\; \text{and} \;
d =
\begin{bmatrix}
    4 \\
    5 \\
    12
\end{bmatrix}
$$

The solution $x^{*}$ is given by the equation $x^{*} = (I-A)^{-1} d$

1. Since $A$ is a nonnegative irreducible matrix, find the Perron-Frobenius eigenvalue of $A$.

2. Use the Neumann Series lemma to find the solution $x^{*}$ if it exists.

```{exercise-end}
```

```{solution-start} eig_ex1
:class: dropdown
```

```{code-cell} ipython3
A = np.array([[0.3, 0.2, 0.3],
              [0.2, 0.4, 0.3],
              [0.2, 0.5, 0.1]])

evals, evecs = eig(A)

r = max(abs(λ) for λ in evals)   #dominant eigenvalue/spectral radius
print(r)
```

Since we have $r(A) < 1$ we can thus find the solution using the Neumann Series lemma.

```{code-cell} ipython3
I = np.identity(3)
B = I - A

d = np.array([4, 5, 12])
d.shape = (3,1)

B_inv = np.linalg.inv(B)
x_star = B_inv @ d
print(x_star)
```

```{solution-end}
```
