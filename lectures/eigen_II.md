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

# Spectral Theory

```{index} single: Spectral Theory
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

In this lecture we will begin with the foundational concepts in spectral theory.

Then we will explore the Perron-Frobenius Theorem and the Neumann Series Lemma, and connect them to applications in Markov chains and networks. 

We will use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
import scipy as sp
import graphviz as gv
import quantecon as qe
```

## Nonnegative matrices

Often, in economics, the matrix that we are dealing with is nonnegative.

Nonnegative matrices have several special and useful properties.

In this section we discuss some of them --- in particular, the connection
between nonnegativity and eigenvalues.

Let $a^{k}_{ij}$ be element $(i,j)$ of $A^k$.

An $n \times m$ matrix $A$ is called **nonnegative** if every element of $A$
is nonnegative, i.e., $a_{ij} \geq 0$ for every $i,j$.

We denote this as $A \geq 0$.

(irreducible)=
### Irreducible matrices

We have (informally) introduced irreducible matrices in the Markov chain lecture (TODO: link to Markov chain lecture).

Here we will introduce this concept formally.

$A$ is called **irreducible** if for *each* $(i,j)$ there is an integer $k \geq 0$ such that $a^{k}_{ij} > 0$.

A matrix $A$ that is not irreducible is called reducible.

Here are some examples to illustrate this further.

1. $A = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.2 \end{bmatrix}$ is irreducible since $a_{ij}>0$ for all $(i,j)$.

2. $A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ is irreducible since $a_{12},a_{21} >0$ and $a^{2}_{11},a^{2}_{22} >0$.

3. $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ is reducible since $A^k = A$ for all $k \geq 0$ and thus
   $a^{k}_{12},a^{k}_{21} = 0$ for all $k \geq 0$.

### Primitive matrices

Let $A$ be a square nonnegative matrix and let $A^k$ be the $k^{th}$ power of $A$.

A matrix is considered **primitive** if there exists a $k \in \mathbb{N}$ such that $A^k$ is everywhere positive.

It means that $A$ is called primitive if there is an integer $k \geq 0$ such that $a^{k}_{ij} > 0$ for *all* $(i,j)$.

We can see that if a matrix is primitive, then it implies the matrix is irreducible.

This is because if there exists an $A^k$ such that $a^{k}_{ij} > 0$ for all $(i,j)$, then it guarantees the same property for ${k+1}^th, {k+2}^th ... {k+n}^th$ iterations.

In other words, a primitive matrix is both irreducible and aperiodic as aperiodicity requires a state to be visited with a guarantee of returning to itself after a certain amount of iterations.

### Left eigenvectors

We have previously discussed right (ordinary) eigenvectors $Av = \lambda v$.

Here we introduce left eigenvectors.

Left eigenvectors will play important roles in what follows, including that of stochastic steady states for dynamic models under a Markov assumption.

We will talk more about this later, but for now, let's define left eigenvectors.

A vector $w$ is called a left eigenvector of $A$ if $w$ is an eigenvector of $A^T$.

In other words, if $w$ is a left eigenvector of matrix A, then $A^T w = \lambda w$, where $\lambda$ is the eigenvalue associated with the left eigenvector $v$.

This hints at how to compute left eigenvectors

```{code-cell} ipython3
A = np.array([[3, 2], 
              [1, 4]])

# Compute right eigenvectors and eigenvalues
λ_r, v = eig(A)

# Compute left eigenvectors and eigenvalues
λ_l, w = eig(A.T)

print("Right Eigenvalues:")
print(λ_r)
print("\nRight Eigenvectors:")
print(v)
print("\nLeft Eigenvalues:")
print(λ_l)
print("\nLeft Eigenvectors:")
print(w)
```

We can use `scipy.linalg.eig` with argument `left=True` to find left eigenvectors directly

```{code-cell} ipython3
eigenvals, ε, e = sp.linalg.eig(A, left=True)

print("Right Eigenvalues:")
print(λ_r)
print("\nRight Eigenvectors:")
print(v)
print("\nLeft Eigenvalues:")
print(λ_l)
print("\nLeft Eigenvectors:")
print(w)
```

Note that the eigenvalues for both left and right eigenvectors are the same, but the eigenvectors themselves are different.

We can then take transpose to obtain $A^T w = \lambda w$ and obtain $w^T A= \lambda w^T$.

This is a more common expression and where the name left eigenvectors originates.

(perron-frobe)=
### The Perron-Frobenius Theorem

For a nonnegative matrix $A$ the behavior of $A^k$ as $k \to \infty$ is controlled by the eigenvalue with the largest
absolute value, often called the **dominant eigenvalue**.

For a matrix $A$, the Perron-Frobenius Theorem characterizes certain
properties of the dominant eigenvalue and its corresponding eigenvector when
$A$ is a nonnegative square matrix.

```{prf:Theorem} Perron-Frobenius Theorem
:label: perron-frobenius

If a matrix $A \geq 0$ then,

1. the dominant eigenvalue of $A$, $r(A)$, is real-valued and nonnegative. 
2. for any other eigenvalue (possibly complex) $\lambda$ of $A$, $|\lambda| \leq r(A)$.
3. we can find a nonnegative and nonzero eigenvector $v$ such that $Av = r(A)v$.

Moreover if $A$ is also irreducible then,

4. the eigenvector $v$ associated with the eigenvalue $r(A)$ is strictly positive.
5. there exists no other positive eigenvector $v$ (except scalar multiples of $v$) associated with $r(A)$.

If $A$ is primitive then,

6. the inequality $|\lambda| \leq r(A)$ is **strict** for all eigenvalues $\lambda$ of $A$ distinct from $r(A)$, and
7. with $v$ and $w$ normalized so that the inner product of $w$ and  $v = 1$, we have
$ r(A)^{-m} A^m$ converges to $v w^{\top}$ when $m \rightarrow \infty$. $v w^{\top}$ is called the **Perron projection** of $A$.
```

(This is a relatively simple version of the theorem --- for more details see
[here](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)).

We will see applications of the theorem below.

Let's build our intuition for the theorem using a simple example we have seen [before](mc_eg1).

Now let's consider examples for each case.

#### Example 1: irreducible matrix

Consider the following irreducible matrix A:

```{code-cell} ipython3
A = np.array([[0, 1, 0], 
              [.5, 0, .5], 
              [0, 1, 0]])
```

We can compute the dominant eigenvalue and the corresponding eigenvector

```{code-cell} ipython3
eig(A)
```

Now we can go through our checklist to verify the claims of the Perron-Frobenius Theorem for the irreducible matrix A:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values less than or equal to the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. As the matrix is irreducible, the eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.

#### Example 2: primitive matrix

Consider the following primitive matrix B:

```{code-cell} ipython3
B = np.array([[0, 1, 1], 
              [1, 0, 1], 
              [1, 1, 0]])

np.linalg.matrix_power(B, 2)
```

We can compute the dominant eigenvalue and the corresponding eigenvector using the power iteration method as discussed {ref}`earlier<eig1_ex1>`:

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
eig(B)
```

+++ {"user_expressions": []}

Now let's verify the claims of the Perron-Frobenius Theorem for the primitive matrix B:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values strictly less than the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. The eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.
6. The inequality $|\lambda| < r(B)$ holds for all eigenvalues $\lambda$ of $B$ distinct from the dominant eigenvalue.

Furthermore, we can verify the convergence property (7) of the theorem on the following examples:

```{code-cell} ipython3
def compute_perron_projection(M):

    eigval, v = eig(M)
    eigval, w = eig(M.T)

    r = np.max(eigval)

    # Find the index of the dominant (Perron) eigenvalue
    i = np.argmax(eigval)

    # Get the Perron eigenvectors
    v_P = v[:, i].reshape(-1, 1)
    w_P = w[:, i].reshape(-1, 1)

    # Normalize the left and right eigenvectors
    norm_factor = w_P.T @ v_P
    v_norm = v_P / norm_factor

    # Compute the Perron projection matrix
    P = v_norm @ w_P.T
    return P, r

def check_convergence(M):
    P, r = compute_perron_projection(M)
    print("Perron projection:")
    print(P)

    # Define a list of values for n
    n_list = [1, 10, 100, 1000, 10000]

    for n in n_list:
        
        # Compute (A/r)^n
        M_n = np.linalg.matrix_power(M/r, n)

        # Compute the difference between A^n / r^n and the Perron projection
        diff = np.abs(M_n - P)

        # Calculate the norm of the difference matrix
        diff_norm = np.linalg.norm(diff, 'fro')
        print(f"n = {n}, norm of the difference: {diff_norm:.10f}")


A1 = np.array([[1, 2],
               [1, 4]])

A2 = np.array([[0, 1, 1], 
              [1, 0, 1], 
              [1, 1, 0]])

A3 = np.array([[0.971, 0.029, 0.1, 1],
               [0.145, 0.778, 0.077, 0.59],
               [0.1, 0.508, 0.492, 1.12],
               [0.2, 0.8, 0.71, 0.95]])

for M in A1, A2, A3:
    print("Matrix:")
    print(M)
    check_convergence(M)
    print()
    print("-"*36)
    print()
```

The convergence is not observed in cases of non-primitive matrices.

Let's go through an example

```{code-cell} ipython3
B = np.array([[0, 1, 1], 
              [1, 0, 0], 
              [1, 0, 0]])

# This shows that the matrix is not primitive
print("Matrix:")
print(B)
print("100th power of matrix B:")
print(np.linalg.matrix_power(B, 100))

check_convergence(B)
```

The result shows that the matrix is not primitive as it is not everywhere positive.

These examples show how the Perron-Frobenius Theorem relates to the eigenvalues and eigenvectors of positive matrices and the convergence of the power of matrices.

In fact we have already seen the theorem in action before in {ref}`the markov chain lecture <mc1_ex_1>`.

(spec_markov)=
#### Example 3: Connection to Markov chains

We are now prepared to bridge the languages spoken in the two lectures. 

A primitive matrix is both irreducible (or strongly connected in the language of graph) and aperiodic.

So Perron-Frobenius Theorem explains why both Imam and Temple matrix and Hamilton matrix converge to a stationary distribution, which is the Perron projection of the two matrices

```{code-cell} ipython3
P = np.array([[0.68, 0.12, 0.20],
              [0.50, 0.24, 0.26],
              [0.36, 0.18, 0.46]])

print(compute_perron_projection(P)[0])
```

```{code-cell} ipython3
mc = qe.MarkovChain(P)
ψ_star = mc.stationary_distributions[0]
ψ_star
```

```{code-cell} ipython3
P_hamilton = np.array([[0.971, 0.029, 0.000],
                       [0.145, 0.778, 0.077],
                       [0.000, 0.508, 0.492]])

print(compute_perron_projection(P_hamilton)[0])
```

```{code-cell} ipython3
mc = qe.MarkovChain(P_hamilton)
ψ_star = mc.stationary_distributions[0]
ψ_star
```

We can also verify other properties hinted by Perron-Frobenius in these stochastic matrices.

+++

Another example is the relationship between convergence gap and convergence rate.

In the {ref}`exercise<mc1_ex_1>`, we stated that the convergence rate is determined by the spectral gap, the difference between the largest and the second largest eigenvalue.

This can be proven using what we have learned here.

With Markov model $M$ with state space $S$ and transition matrix $P$, we can write $P^t$ as

$$
P^t=\sum_{i=1}^{n-1} \lambda_i^t v_i w_i^{\top}+\mathbb{1} \psi^*,
$$

This is proven in {cite}`sargent2023economic` and a nice discussion can be found [here](https://math.stackexchange.com/questions/2433997/can-all-matrices-be-decomposed-as-product-of-right-and-left-eigenvector).

In the formula $\lambda_i$ is an eigenvalue of $P$ and $v_i$ and $w_i$ are the right and left eigenvectors corresponding to $\lambda_i$. 

Premultiplying $P^t$ by arbitrary $\psi \in \mathscr{D}(S)$ and rearranging now gives

$$
\psi P^t-\psi^*=\sum_{i=1}^{n-1} \lambda_i^t \psi v_i w_i^{\top}
$$

Recall that eigenvalues are ordered from smallest to largest from $i = 1 ... n$. 

As we have seen, the largest eigenvalue for a primitive stochastic matrix is one.

This can be proven using [Gershgorin Circle Theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem), 
but it is out of the scope of this lecture.

So by the statement (6) of Perron-Frobenius Theorem, $\lambda_i<1$ for all $i<n$, and $\lambda_n=1$ when $P$ is primitive (strongly connected and aperiodic). 


Hence, after taking the Euclidean norm deviation, we obtain

$$
\left\|\psi P^t-\psi^*\right\|=O\left(\eta^t\right) \quad \text { where } \quad \eta:=\left|\lambda_{n-1}\right|<1
$$

Thus, the rate of convergence is governed by the modulus of the second largest eigenvalue.


(la_neumann)=
## The Neumann Series Lemma 

```{index} single: Neumann's Lemma
```

In this section we present a famous result about series of matrices that has
many applications in economics.

### Scalar series

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

### Matrix series

A generalization of this idea exists in the matrix setting.

Consider the system of equations $x = Ax + b$ where $A$ is an $n \times n$
square matrix and $x$ and $b$ are both column vectors in $\mathbb{R}^n$.

Using matrix algebra we can conclude that the solution to this system of equations will be given by:

```{math}
:label: neumann_eqn
    
    x^{*} = (I-A)^{-1}b

```

What guarantees the existence of a unique vector $x^{*}$ that satisfies
{eq}`neumann_eqn`?

The following is a fundamental result in functional analysis that generalizes
{eq}`gp_sum` to a multivariate case.

(neumann_series_lemma)=
```{prf:Theorem} Neumann Series Lemma
:label: neumann_series_lemma

Let $A$ be a square matrix and let $A^k$ be the $k$-th power of $A$.

Let $r(A)$ be the dominant eigenvector or as it is commonly called the *spectral radius*, defined as $\max_i |\lambda_i|$, where 

* $\{\lambda_i\}_i$ is the set of eigenvalues of $A$ and
* $|\lambda_i|$ is the modulus of the complex number $\lambda_i$

Neumann's Theorem states the following: If $r(A) < 1$, then $I - A$ is invertible, and

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
* $0.2x_2$ is used as inputs by the industry sector to produce $x_2$ units,
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
