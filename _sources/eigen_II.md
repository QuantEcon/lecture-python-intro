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

# The Perron-Frobenius Theorem

```{index} single: The Perron-Frobenius Theorem
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

In this lecture we will begin with the foundational concepts in spectral theory.

Then we will explore the Perron-Frobenius Theorem and connect it to applications in Markov chains and networks.

We will use the following imports:

```{code-cell} ipython3
import numpy as np
from numpy.linalg import eig
import scipy as sp
import quantecon as qe
```

## Nonnegative matrices

Often, in economics, the matrix that we are dealing with is nonnegative.

Nonnegative matrices have several special and useful properties.

In this section we will discuss some of them --- in particular, the connection
between nonnegativity and eigenvalues.

An $n \times m$ matrix $A$ is called **nonnegative** if every element of $A$
is nonnegative, i.e., $a_{ij} \geq 0$ for every $i,j$.

We denote this as $A \geq 0$.

(irreducible)=
### Irreducible matrices

We introduced irreducible matrices in the [Markov chain lecture](mc_irreducible).

Here we generalize this concept:

Let $a^{k}_{ij}$ be element $(i,j)$ of $A^k$.

An $n \times n$ nonnegative matrix $A$ is called irreducible if $A + A^2 + A^3 + \cdots \gg 0$, where $\gg 0$ indicates that every element in $A$ is strictly positive.

In other words, for each $i,j$ with $1 \leq i, j \leq n$, there exists a $k \geq 0$ such that $a^{k}_{ij} > 0$.

Here are some examples to illustrate this further:

$$
A = \begin{bmatrix} 0.5 & 0.1 \\ 
                    0.2 & 0.2 
\end{bmatrix}
$$

$A$ is irreducible since $a_{ij}>0$ for all $(i,j)$.

$$
B = \begin{bmatrix} 0 & 1 \\ 
                    1 & 0 
\end{bmatrix}
, \quad
B^2 = \begin{bmatrix} 1 & 0 \\ 
                      0 & 1
\end{bmatrix}
$$

$B$ is irreducible since $B + B^2$ is a matrix of ones.

$$
C = \begin{bmatrix} 1 & 0 \\ 
                    0 & 1 
\end{bmatrix}
$$

$C$ is not irreducible since $C^k = C$ for all $k \geq 0$ and thus
   $c^{k}_{12},c^{k}_{21} = 0$ for all $k \geq 0$.

### Left eigenvectors

Recall that we previously discussed eigenvectors in {ref}`Eigenvalues and Eigenvectors <la_eigenvalues>`.

In particular, $\lambda$ is an eigenvalue of $A$ and $v$ is an eigenvector of $A$ if $v$ is nonzero and satisfy

$$
Av = \lambda v.
$$

In this section we introduce left eigenvectors.

To avoid confusion, what we previously referred to as "eigenvectors" will be called "right eigenvectors".

Left eigenvectors will play important roles in what follows, including that of stochastic steady states for dynamic models under a Markov assumption.

A vector $w$ is called a left eigenvector of $A$ if $w$ is a right eigenvector of $A^\top$.

In other words, if $w$ is a left eigenvector of matrix $A$, then $A^\top w = \lambda w$, where $\lambda$ is the eigenvalue associated with the left eigenvector $v$.

This hints at how to compute left eigenvectors

```{code-cell} ipython3
A = np.array([[3, 2],
              [1, 4]])

# Compute eigenvalues and right eigenvectors
λ, v = eig(A)

# Compute eigenvalues and left eigenvectors
λ, w = eig(A.T)

# Keep 5 decimals
np.set_printoptions(precision=5)

print(f"The eigenvalues of A are:\n {λ}\n")
print(f"The corresponding right eigenvectors are: \n {v[:,0]} and {-v[:,1]}\n")
print(f"The corresponding left eigenvectors are: \n {w[:,0]} and {-w[:,1]}\n")
```

We can also use `scipy.linalg.eig` with argument `left=True` to find left eigenvectors directly

```{code-cell} ipython3
eigenvals, ε, e = sp.linalg.eig(A, left=True)

print(f"The eigenvalues of A are:\n {eigenvals.real}\n")
print(f"The corresponding right eigenvectors are: \n {e[:,0]} and {-e[:,1]}\n")
print(f"The corresponding left eigenvectors are: \n {ε[:,0]} and {-ε[:,1]}\n")
```

The eigenvalues are the same while the eigenvectors themselves are different.

(Also note that we are taking the nonnegative value of the eigenvector of {ref}`dominant eigenvalue <perron-frobe>`, this is because `eig` automatically normalizes the eigenvectors.)

We can then take transpose to obtain $A^\top w = \lambda w$ and obtain $w^\top A= \lambda w^\top$.

This is a more common expression and where the name left eigenvectors originates.

(perron-frobe)=
### The Perron-Frobenius theorem

For a square nonnegative matrix $A$, the behavior of $A^k$ as $k \to \infty$ is controlled by the eigenvalue with the largest
absolute value, often called the **dominant eigenvalue**.

For any such matrix $A$, the Perron-Frobenius Theorem characterizes certain
properties of the dominant eigenvalue and its corresponding eigenvector.

```{prf:Theorem} Perron-Frobenius Theorem
:label: perron-frobenius

If a matrix $A \geq 0$ then,

1. the dominant eigenvalue of $A$, $r(A)$, is real-valued and nonnegative.
2. for any other eigenvalue (possibly complex) $\lambda$ of $A$, $|\lambda| \leq r(A)$.
3. we can find a nonnegative and nonzero eigenvector $v$ such that $Av = r(A)v$.

Moreover if $A$ is also irreducible then,

4. the eigenvector $v$ associated with the eigenvalue $r(A)$ is strictly positive.
5. there exists no other positive eigenvector $v$ (except scalar multiples of $v$) associated with $r(A)$.

(More of the Perron-Frobenius theorem about primitive matrices will be introduced {ref}`below <prim_matrices>`.)
```

(This is a relatively simple version of the theorem --- for more details see
[here](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)).

We will see applications of the theorem below.

Let's build our intuition for the theorem using a simple example we have seen [before](mc_eg1).

Now let's consider examples for each case.

#### Example: Irreducible matrix

Consider the following irreducible matrix $A$:

```{code-cell} ipython3
A = np.array([[0, 1, 0],
              [.5, 0, .5],
              [0, 1, 0]])
```

We can compute the dominant eigenvalue and the corresponding eigenvector

```{code-cell} ipython3
eig(A)
```

Now we can see the claims of the Perron-Frobenius Theorem holds for the irreducible matrix $A$:

1. The dominant eigenvalue is real-valued and non-negative.
2. All other eigenvalues have absolute values less than or equal to the dominant eigenvalue.
3. A non-negative and nonzero eigenvector is associated with the dominant eigenvalue.
4. As the matrix is irreducible, the eigenvector associated with the dominant eigenvalue is strictly positive.
5. There exists no other positive eigenvector associated with the dominant eigenvalue.

(prim_matrices)=
### Primitive matrices

We know that in real world situations it's hard for a matrix to be everywhere positive (although they have nice properties).

The primitive matrices, however, can still give us helpful properties with looser definitions.

Let $A$ be a square nonnegative matrix and let $A^k$ be the $k^{th}$ power of $A$.

A matrix is called **primitive** if there exists a $k \in \mathbb{N}$ such that $A^k$ is everywhere positive.

Recall the examples given in irreducible matrices:

$$
A = \begin{bmatrix} 0.5 & 0.1 \\ 
                    0.2 & 0.2 
\end{bmatrix}
$$

$A$ here is also a primitive matrix since $A^k$ is everywhere nonnegative for $k \in \mathbb{N}$.

$$
B = \begin{bmatrix} 0 & 1 \\ 
                    1 & 0 
\end{bmatrix}
, \quad
B^2 = \begin{bmatrix} 1 & 0 \\ 
                      0 & 1
\end{bmatrix}
$$

$B$ is irreducible but not primitive since there are always zeros in either principal diagonal or secondary diagonal.

We can see that if a matrix is primitive, then it implies the matrix is irreducible but not vice versa.

Now let's step back to the primitive matrices part of the Perron-Frobenius Theorem

```{prf:Theorem} Continous of Perron-Frobenius Theorem
:label: con-perron-frobenius

If $A$ is primitive then,

6. the inequality $|\lambda| \leq r(A)$ is **strict** for all eigenvalues $\lambda$ of $A$ distinct from $r(A)$, and
7. with $v$ and $w$ normalized so that the inner product of $w$ and  $v = 1$, we have
$ r(A)^{-m} A^m$ converges to $v w^{\top}$ when $m \rightarrow \infty$. The matrix $v w^{\top}$ is called the **Perron projection** of $A$.
```

#### Example 1: Primitive matrix

Consider the following primitive matrix $B$:

```{code-cell} ipython3
B = np.array([[0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]])

np.linalg.matrix_power(B, 2)
```

We compute the dominant eigenvalue and the corresponding eigenvector

```{code-cell} ipython3
eig(B)
```

Now let's give some examples to see if the claims of the Perron-Frobenius Theorem hold for the primitive matrix $B$:

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
        print(f"n = {n}, error = {diff_norm:.10f}")


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

In fact we have already seen the theorem in action before in {ref}`the Markov chain lecture <mc1_ex_1>`.

(spec_markov)=
#### Example 2: Connection to Markov chains

We are now prepared to bridge the languages spoken in the two lectures.

A primitive matrix is both irreducible and aperiodic.

So Perron-Frobenius Theorem explains why both {ref}`Imam and Temple matrix <mc_eg3>` and [Hamilton matrix](https://en.wikipedia.org/wiki/Hamiltonian_matrix) converge to a stationary distribution, which is the Perron projection of the two matrices

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

Please note that we use $\mathbb{1}$ for a vector of ones in this lecture.

With Markov model $M$ with state space $S$ and transition matrix $P$, we can write $P^t$ as

$$
P^t=\sum_{i=1}^{n-1} \lambda_i^t v_i w_i^{\top}+\mathbb{1} \psi^*,
$$

This is proven in {cite}`sargent2023economic` and a nice discussion can be found [here](https://math.stackexchange.com/questions/2433997/can-all-matrices-be-decomposed-as-product-of-right-and-left-eigenvector).

In this formula $\lambda_i$ is an eigenvalue of $P$ with corresponding right and left eigenvectors $v_i$ and $w_i$ .

Premultiplying $P^t$ by arbitrary $\psi \in \mathscr{D}(S)$ and rearranging now gives

$$
\psi P^t-\psi^*=\sum_{i=1}^{n-1} \lambda_i^t \psi v_i w_i^{\top}
$$

Recall that eigenvalues are ordered from smallest to largest from $i = 1 ... n$.

As we have seen, the largest eigenvalue for a primitive stochastic matrix is one.

This can be proven using [Gershgorin Circle Theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem),
but it is out of the scope of this lecture.

So by the statement (6) of Perron-Frobenius Theorem, $\lambda_i<1$ for all $i<n$, and $\lambda_n=1$ when $P$ is primitive.

Hence, after taking the Euclidean norm deviation, we obtain

$$
\left\|\psi P^t-\psi^*\right\|=O\left(\eta^t\right) \quad \text { where } \quad \eta:=\left|\lambda_{n-1}\right|<1
$$

Thus, the rate of convergence is governed by the modulus of the second largest eigenvalue.


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

2. Use the {ref}`Neumann Series Lemma <la_neumann>` to find the solution $x^{*}$ if it exists.

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

Since we have $r(A) < 1$ we can thus find the solution using the Neumann Series Lemma.

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
