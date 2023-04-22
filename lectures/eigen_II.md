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

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install graphviz
```

```{admonition} graphviz
:class: warning
If you are running this lecture locally it requires [graphviz](https://www.graphviz.org)
to be installed on your computer. Installation instructions for graphviz can be found
[here](https://www.graphviz.org/download/) 
```

In this lecture we will begin with the basic properties of nonnegative matrices.

Then we will explore the Perron-Frobenius Theorem and the Neumann Series Lemma, and connect them to applications in Markov chains and networks. 

We will use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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

A matrix is considered **primitive** if there exists a $k \in \mathbb{N}$ such that $A^k$ is everywhere positive.

It means that $A$ is called primitive if there is an integer $k \geq 0$ such that $a^{k}_{ij} > 0$ for *all* $(i,j)$.

We can see that if a matrix is primitive, then it implies the matrix is irreducible.

This is because if there exists an $A^k$ such that $a^{k}_{ij} > 0$ for all $(i,j)$, then it guarantees the same property for ${k+1}^th, {k+2}^th ... {k+n}^th$ iterations.

In other words, a primitive matrix is both irreducible and aperiodical as aperiodicity requires a state to be visited with a guarantee of returning to itself after a certain amount of iterations.

### Left Eigenvectors

We have previously discussed right (ordinary) eigenvectors $Av = \lambda v$.

Here we introduce left eigenvectors.

Left eigenvectors will play important roles in what follows, including that of stochastic steady states for dynamic models under a Markov assumption.

We will talk more about this later, but for now, let's define left eigenvectors.

A vector $\varepsilon$ is called a left eigenvector of $A$ if $\varepsilon$ is an eigenvector of $A^T$.

In other words, if $\varepsilon$ is a left eigenvector of matrix A, then $A^T \varepsilon = \lambda \varepsilon$, where $\lambda$ is the eigenvalue associated with the left eigenvector $v$.

This hints at how to compute left eigenvectors

```{code-cell} ipython3
A = np.array([[3, 2], 
              [1, 4]])

# Compute right eigenvectors and eigenvalues
eigvals_r, e = np.linalg.eig(A)

# Compute left eigenvectors and eigenvalues
eigvals_l, ε = np.linalg.eig(A.T)

print("Right Eigenvalues:")
print(eigvals_r)
print("\nRight Eigenvectors:")
print(e)
print("\nLeft Eigenvalues:")
print(eigvals_l)
print("\nLeft Eigenvectors:")
print(ε)
```

We can use `scipy.linalg.eig` with argument `left=True` to find left eigenvectors directly

```{code-cell} ipython3
eigenvals, ε, e = sp.linalg.eig(A, left=True)

print("Right Eigenvalues:")
print(eigvals_r)
print("\nRight Eigenvectors:")
print(e)
print("\nLeft Eigenvalues:")
print(eigvals_l)
print("\nLeft Eigenvectors:")
print(ε)
```

Note that the eigenvalues for both left and right eigenvectors are the same, but the eigenvectors themselves are different.

We can then take transpose to obtain $A^T \varepsilon = \lambda \varepsilon$ and obtain $\varepsilon^T A= \lambda \varepsilon^T$.

This is a more common expression and where the name left eigenvectors originates.

(perron-frobe)=
### The Perron-Frobenius Theorem

For a nonnegative matrix $A$ the behavior of $A^k$ as $k \to \infty$ is controlled by the eigenvalue with the largest
absolute value, often called the **dominant eigenvalue**.

For a matrix $A$, the Perron-Frobenius theorem characterizes certain
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
$ r(A)^{-m} A^m$ converges to $e \varepsilon^{\top}$ when $m \rightarrow \infty$
```

(This is a relatively simple version of the theorem --- for more details see
[here](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)).

We will see applications of the theorem below.

Let's build our intuition for the theorem using a simple example we have seen [before](mc_eg1).

Now let's consider examples for each case.

#### Example 1: Irreducible Matrix

Consider the following irreducible matrix A:`

```{code-cell} ipython3
A = np.array([[0, 1, 0], 
              [.5, 0, .5], 
              [0, 1, 0]])
```

We can compute the dominant eigenvalue and the corresponding eigenvector

```{code-cell} ipython3
np.linalg.eig(A)
```

Now we can go through our checklist to verify the claims of the Perron-Frobenius theorem for the irreducible matrix A:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values less than or equal to the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. As the matrix is irreducible, the eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.

#### Example 2: Primitive Matrix

Consider the following primitive matrix B:

```{code-cell} ipython3
B = np.array([[0, 1, 1], 
              [1, 0, 1], 
              [1, 1, 0]])

np.linalg.matrix_power(B, 2)
```

We can compute the dominant eigenvalue and the corresponding eigenvector using the power iteration method as discussed {ref} `earlier<eig1_ex1>`:

```{code-cell} ipython3
num_iters = 20
b = np.random.rand(B.shape[1])

for i in range(num_iters):
    b = B @ b
    b = b / np.linalg.norm(b)

dominant_eigenvalue = np.dot(B @ b, b) / np.dot(b, b)
np.round(dominant_eigenvalue, 2)
```

```{code-cell} ipython3
np.linalg.eig(B)
```

Now let's verify the claims of the Perron-Frobenius theorem for the primitive matrix B:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values strictly less than the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. The eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.
6. The inequality $|\lambda| < r(B)$ holds for all eigenvalues $\lambda$ of $B distinct from the dominant eigenvalue.

Furthermore, we can verify the convergence property (7) of the theorem:

```{code-cell} ipython3
import numpy as np

def compute_perron_projection(A):
    # Compute the eigenvalues and right eigenvectors of A
    eigval, v = np.linalg.eig(A)
    eigval, w = np.linalg.eig(A.T)

    r = np.max(eigval)

    # Find the index of the Perron eigenvalue (the largest one)
    i = np.argmax(eigval)

    # Get the Perron eigenvalue and its corresponding right eigenvector
    v_col = v[:, i].reshape(-1, 1)
    w_col = w[:, i].reshape(-1, 1)

    # Normalize the left and right eigenvectors so that w^T * v = 1
    norm_factor = w_col.T @ v_col
    v_norm = v_col / norm_factor
    w_norm = w_col

    # Compute the Perron projection matrix by multiplying the right eigenvector by the transpose of the left eigenvector
    P = v_norm @ w_norm.T
    return P, r

A1 = np.array([[0.971, 0.029, 0.1],
               [0.145, 0.778, 0.077],
               [0.1, 0.508, 0.492]])

A2 = np.array([[1, 2],
               [1, 4]])

for A in [A1, A2]:
    P, r = compute_perron_projection(A)
    print("Matrix A:")
    print(A)
    print("Perron projection matrix:")
    print(P)

    # Define a list of values for n
    n_list = [1, 10, 100, 1000, 10000]

    # Loop over n_list and compute the matrix power A^n / r^n
    for n in n_list:
        # Compute A^n / r^n using numpy.linalg.matrix_power function
        An_rn = np.linalg.matrix_power(A, n) / r**n

        # Compute the difference between A^n / r^n and the Perron projection matrix
        diff = np.abs(An_rn - P)

        # Calculate the Frobenius norm of the difference matrix
        frobenius_norm = np.linalg.norm(diff, 'fro')

        # Print the Frobenius norm for the current value of n
        print(f"n = {n}, Frobenius norm of the difference: {frobenius_norm:.10f}")
```

```{math}
P
= \left(
\begin{array}{cc}
    1 - \alpha & \alpha \\
    \beta & 1 - \beta
\end{array}
  \right) \quad \text{where} \quad \alpha, \beta \in \left[0,1 \right]
```

Calculating the eigenvalues and eigenvectors of $P$ by hand we find that the dominant eigenvalue is $1$ ($\lambda_1 = 1$), and ($\lambda_2 = 1 - \alpha - \beta$).

In this case, $r(A) = 1$.

As $A \geq 0$, we can apply the first part of the theorem to say that r(A) is an eigenvalue.

This verifies the first part of the theorem.

In fact, we have already seen Perron-Frobenius theorem in action before in {ref}`the exercise <mc1_ex_1>`.

In the exercise, we stated that the convegence rate is determined by the spectral gap, the difference between the largest and the second largest eigenvalue.

This can be proved using Perron-Frobenius theorem.

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

The following is a fundamental result in functional analysis that generalizes
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
