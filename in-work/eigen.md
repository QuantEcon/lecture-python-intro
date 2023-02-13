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

```{code-cell} ipython3
import matplotlib.pyplot as plt
from math import sqrt
#plt.rcParams["figure.figsize"] = (10, 5)  # set default figure size
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

+++

Let's start by discussing an important concept around matrices in linear algebra.

Consider the following equation where we multiply a $n \times m$ matrix $A$ with a $m \times 1$
column vector $x$ to obtain a $n \times 1$ column vector $y$.

$$
Ax = y
$$

One way to interpret this equation is to view the matrix $A$ as a function or a "transformation" that maps
$x$ to $y$.

This is the same as saying $A : \mathbb{R}^m \rightarrow \mathbb{R}^n$ such that,

$$
A(x) = y
$$

In this lecture we will restrict our discussion to only square matrices.

Thus, $A : \mathbb{R}^n \rightarrow \mathbb{R}^n$, i.e., $A$ is an $n \times n$ matrix that maps a vector
$x$ in $\mathbb{R}^n$ to a vector $y$ also in $\mathbb{R}^n$.

We prefer to use the term **transformations** instead of functions to characterize _movement_.

Consider the following example:

$$
\begin{bmatrix}
    2 & 1 \\
    -1 & 1
\end{bmatrix}
\begin{bmatrix}
    1 \\
    3
\end{bmatrix}
=
\begin{bmatrix}
    5 \\
    2
\end{bmatrix}
$$

Here, the matrix
$A = \begin{bmatrix} 2 & 1 \\ -1 & 1 \end{bmatrix}$
is mapping the vector $x = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$ to the vector
$y = \begin{bmatrix} 5 \\ 2 \end{bmatrix}$.

This is also illustrated below.

```{code-cell} ipython3
:tags: []

A = np.array([[2, 1], [-1, 1]])
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots()
# Set the axes through the origin

for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-2, 6), ylim=(-2, 4), aspect=1)

vecs = ((1, 3),(5, 2))
c = ['r','black']
for i, v in enumerate(vecs):
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(color=c[i],
                shrink=0,
                alpha=0.7,
                width=0.5))
    
ax.text(0.2 + 1 , 0.2 + 3, 'x=$(1,3)$')
ax.text(0.2 + 5 , 0.2 + 2, 'Ax=$(5,2)$')
    
ax.annotate('', xy=(sqrt(10/29)* 5, sqrt(10/29) *2), xytext=(0, 0),
                arrowprops=dict(color='purple',
                shrink=0,
                alpha=0.7,
                width=0.5))

ax.annotate('',xy=(1,2/5),xytext=(1/3, 1),
             arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3'}
             ,horizontalalignment='center')
ax.text(0.8,0.8, f'θ',fontsize =14)

plt.show()
```

One way to understand this is that matrix $A$ first rotates $x$ by some angle $\theta$
and then scales it by some scalar $\gamma$ to obtain the image $y$ of $x$.

Thus we can say that the matrix $A$ _transforms_ the vector $x$ into the vector $y$.

It is easier to visualise transformations if we think of vectors as points instead of arrows.

If we consider each vector $x \in \mathbb{R}^n$ as a point in the space then the matrix $A$
maps or _transforms_ each point to some other point in the space.

Let's examine some standard transformations we can perform on matrices.

We visualise these transformations on a grid of points and on a unit circle in $\mathbb{R}^2$

```{code-cell} ipython3
:tags: []

def colorizer(x, y):
    r = min(1, 1-y/3)
    g = min(1, 1+y/3)
    b = 1/4 + x/16
    return (r, g, b)
```

```{code-cell} ipython3
:tags: []

def grid_transform(A = np.array([[1, -1], [1, 1]])):
    xvals = np.linspace(-4, 4, 9)
    yvals = np.linspace(-3, 3, 7)
    xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])
    uvgrid = A @ xygrid
    
    colors = list(map(colorizer, xygrid[0], xygrid[1]))
    
    figure, ax = plt.subplots(1,2, figsize = (10,5))
    
    for axes in ax:
        axes.set(xlim=(-11, 11), ylim=(-11, 11))
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in ['left', 'bottom']:
            axes.spines[spine].set_position('zero')
        for spine in ['right', 'top']:
            axes.spines[spine].set_color('none')
    
    # Plot x-y grid points
    ax[0].scatter(xygrid[0], xygrid[1], s=36, c=colors, edgecolor="none")
    #ax[0].grid(True)
    #ax[0].axis("equal")
    ax[0].set_title("points $x_1, x_2, \cdots, x_k$")
    
    # Plot transformed grid points
    ax[1].scatter(uvgrid[0], uvgrid[1], s=36, c=colors, edgecolor="none")
    #ax[1].grid(True)
    #ax[1].axis("equal")
    ax[1].set_title("points $Ax_1, Ax_2, \cdots, Ax_k$")
    
    plt.show()
```

```{code-cell} ipython3
:tags: []

def circle_transform(A = np.array([[-1, 2], [0, 1]])):
    
    figure, ax = plt.subplots(1,2, figsize = (10,5))
    
    for axes in ax:
        axes.set(xlim=(-4, 4), ylim=(-4, 4))
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in ['left', 'bottom']:
            axes.spines[spine].set_position('zero')
        for spine in ['right', 'top']:
            axes.spines[spine].set_color('none')
    
    θ = np.linspace( 0 , 2 * np.pi , 150) 
    r = 1
    
    θ_1 = np.empty(12)
    for i in range(12):
        θ_1[i] = 2 * np.pi * (i/12)
    
    x = r * np.cos(θ) 
    y = r * np.sin(θ)
    a = r * np.cos(θ_1)
    b = r * np.sin(θ_1)
    a_1 = a.reshape(1,-1)
    b_1 = b.reshape(1,-1)
    colors = list(map(colorizer, a, b))
    ax[0].plot(x, y, color = 'black', zorder=1)
    ax[0].scatter(a_1,b_1, c = colors, alpha = 1, s = 60, edgecolors = 'black', zorder =2)
    ax[0].set_title("unit circle in $\mathbb{R}^2$")
    
    x1= x.reshape(1,-1)
    y1 = y.reshape(1, -1)
    ab = np.concatenate((a_1,b_1), axis=0)
    transformed_ab = A @ ab
    transformed_circle_input = np.concatenate((x1,y1), axis=0)
    transformed_circle = A @ transformed_circle_input
    ax[1].plot(transformed_circle[0,:], transformed_circle[1,:], color = 'black', zorder= 1) 
    ax[1].scatter(transformed_ab[0,:],transformed_ab[1:,], color = colors, alpha = 1, s = 60, edgecolors = 'black', zorder =2)
    ax[1].set_title("transformed circle")
   
    plt.show()
```

### Scaling
A matrix of the form $\begin{bmatrix} \alpha & 0 \\ 0 & \beta \end{bmatrix}$ scales vectors
across the x-axis by a factor $\alpha$ and along the y-axis by a factor $\beta$.

Here we illustrate a simple example where $\alpha = \beta = 3$.

```{code-cell} ipython3
:tags: []

A = np.array([[3 ,0],    #scaling by 3 in both directions
              [0, 3]])
grid_transform(A)
circle_transform(A)
```

### Shearing

A shear matrix of the form $\begin{bmatrix} 1 & \lambda \\ 0 & 1 \end{bmatrix}$ stretches vectors
along the x-axis by an amount proportional to the y-coordinate of a point.

```{code-cell} ipython3
A = np.array([[1, 2],     # shear along x-axis
              [0, 1]])
grid_transform(A)
circle_transform(A)
```

### Rotation
A matrix of the form $\begin{bmatrix} \cos \theta & \sin \theta \\ - \sin \theta & \cos \theta \end{bmatrix}$
is called a _rotation matrix_.

This matrix rotates vectors clockwise by an angle $\theta$.

```{code-cell} ipython3
θ = np.pi/4      #45 degree clockwise rotation
A = np.array([[np.cos(θ), np.sin(θ)],
              [-np.sin(θ), np.cos(θ)]])
grid_transform(A)
```

### Permutation
The permutation matrix $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ interchanges
the coordinates of a vector.

```{code-cell} ipython3
A = np.column_stack([[0, 1], [1, 0]])
grid_transform(A)
```

More examples of common transition matrices can be found [here](https://en.wikipedia.org/wiki/Transformation_matrix#Examples_in_2_dimensions).

+++ {"tags": []}

## Matrix Multiplication as Composition

Since matrices act as functions that transform one vector to another, we can apply the concept of
function composition to matrices as well. 

Consider the two matrices $A = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$ and 
$B = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$.

What will the output be when we try to obtain $ABx$ for some $2 \times 1$ vector $x$?

$$
\color{red}{\underbrace{
 \color{black}{\begin{bmatrix}
  0 & 1 \\
 -1 & 0
 \end{bmatrix}}
}_{\textstyle A} }
\color{red}{\underbrace{
 \color{black}{\begin{bmatrix}
  1 & 2 \\
  0 & 1
 \end{bmatrix}}
}_{\textstyle B}}
\color{red}{\overbrace{
 \color{black}{\begin{bmatrix}
  1 \\
  3
 \end{bmatrix}}
}^{\textstyle x}}
\rightarrow
\color{red}{\underbrace{
 \color{black}{\begin{bmatrix}
  0 & 1 \\
  -1 & -2
 \end{bmatrix}}
}_{\textstyle AB}}
\color{red}{\overbrace{
 \color{black}{\begin{bmatrix}
  1 \\
  3
 \end{bmatrix}}
}^{\textstyle x}}
\rightarrow
\color{red}{\overbrace{
 \color{black}{\begin{bmatrix}
  3 \\
  -7
 \end{bmatrix}}
}^{\textstyle y}}
$$

$$
\color{red}{\underbrace{
 \color{black}{\begin{bmatrix}
  0 & 1 \\
 -1 & 0
 \end{bmatrix}}
}_{\textstyle A} }
\color{red}{\underbrace{
 \color{black}{\begin{bmatrix}
  1 & 2 \\
  0 & 1
 \end{bmatrix}}
}_{\textstyle B}}
\color{red}{\overbrace{
 \color{black}{\begin{bmatrix}
  1 \\
  3
 \end{bmatrix}}
}^{\textstyle x}}
\rightarrow
\color{red}{\underbrace{
 \color{black}{\begin{bmatrix}
  0 & 1 \\
  -1 & 0
 \end{bmatrix}}
}_{\textstyle A}}
\color{red}{\overbrace{
 \color{black}{\begin{bmatrix}
  7 \\
  3
 \end{bmatrix}}
}^{\textstyle Bx}}
\rightarrow
\color{red}{\overbrace{
 \color{black}{\begin{bmatrix}
  3 \\
  -7
 \end{bmatrix}}
}^{\textstyle y}}
$$

We can observe that applying the transformation $AB$ on the vector $x$ is the same as first applying $B$ on $x$ and then applying
$A$ on the vector $Bx$.

Thus the matrix product $AB$ is the **composition** of the matrix transformations $A$ and $B$.

When we matrix multiply an $n \times m$ matrix $A$ with an $m \times k$ matrix $B$ the obtained matrix product is an $n \times k$
matrix $AB$.

This is the same as if $A$ and $B$ are transformations such that,
$A: \mathbb{R}^m \to \mathbb{R}^n$ and $B: \mathbb{R}^k \to \mathbb{R}^m$
then $AB$ is a transformation such that $AB: \mathbb{R}^k \to \mathbb{R}^n$.

Viewing matrix multiplication as composition of maps is also useful to easily understand why usually under matrix multiplication
$AB \neq BA$.

Let $A$ be the $90^{\circ}$ clockwise rotation matrix given by $\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$ and let
$B$ be a shear matrix along the x-axis given by $\begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$.

We will visualise how a grid of points changes when we apply the transformation $AB$ and then compare it with the
transformation $BA$.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
tags: []
---
def grid_composition_transform(A = np.array([[1, -1], [1, 1]]), B = np.array([[1, -1], [1, 1]])):
    xvals = np.linspace(-4, 4, 9)
    yvals = np.linspace(-3, 3, 7)
    xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])
    uvgrid = B @ xygrid
    abgrid = A @ uvgrid
    
    colors = list(map(colorizer, xygrid[0], xygrid[1]))
    
    figure, ax = plt.subplots(1,3, figsize = (15,5))
    
    for axes in ax:
        axes.set(xlim=(-12, 12), ylim=(-12, 12))
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in ['left', 'bottom']:
            axes.spines[spine].set_position('zero')
        for spine in ['right', 'top']:
            axes.spines[spine].set_color('none')
            
    # Plot grid points
    ax[0].scatter(xygrid[0], xygrid[1], s=36, c=colors, edgecolor="none")
    ax[0].set_title("points $x_1, x_2, \cdots, x_k$")
    
    # Plot intermediate grid points
    ax[1].scatter(uvgrid[0], uvgrid[1], s=36, c=colors, edgecolor="none")
    ax[1].set_title("points $Bx_1, Bx_2, \cdots, Bx_k$")
    
    #Plot transformed grid points
    ax[2].scatter(abgrid[0], abgrid[1], s=36, c=colors, edgecolor="none")
    ax[2].set_title("points $ABx_1, ABx_2, \cdots, ABx_k$")

    plt.show()
```

```{code-cell} ipython3
θ = np.pi/2 
#B = np.array([[np.cos(θ), np.sin(θ)],
#              [-np.sin(θ), np.cos(θ)]])
A = np.array([[0, 1],     # 90 degree clockwise rotation
              [-1, 0]])
B = np.array([[1, 2],     # shear along x-axis
              [0, 1]])
```

### Rotate then Shear

```{code-cell} ipython3
grid_composition_transform(A,B)        #transformation AB
```

### Shear then Rotate

```{code-cell} ipython3
grid_composition_transform(B,A)         #transformation BA
```

It is quite evident that the transformation $AB$ is not the same as the transformation $BA$.

+++

## Iterating on a Fixed Map

In economics we often are interested in analyzing the behavior of iterating on a fixed map.

Consider a vector $v$ under the mapping $A$. 

We are interested in what happens when we repeatedly multiply $v$ by $A$, i.e., what happens to $AAv = A^2v$, $A^3v$ and so on for $A^kv$

Let's first see examples of a sequence of iterates $(A^kx)_{k \geq 0}$ under different maps $A$.

```{code-cell} ipython3
def plot_series(B, v, n):
    
    A = np.array([[1, -1],
                  [1, 0]])
    
    figure, ax = plt.subplots()
    
    ax.set(xlim=(-4, 4), ylim=(-4, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
    for spine in ['right', 'top']:
        ax.spines[spine].set_color('none')
        
    θ = np.linspace( 0 , 2 * np.pi , 150) 
    r = 2.5
    x = r * np.cos(θ) 
    y = r * np.sin(θ)
    x1 = x.reshape(1,-1)
    y1 = y.reshape(1, -1)
    xy = np.concatenate((x1,y1), axis=0)
    
    ellipse = A @ xy
    ax.plot(ellipse[0,:], ellipse[1,:], color = 'black', linestyle = (0, (5,10)), linewidth = 0.5)
    
    colors = plt.cm.rainbow(np.linspace(0,1,20))# Initialize holder for trajectories
    
    for i in range(n):
        iteration = matrix_power(B, i) @ v
        v1 = iteration[0]
        v2 = iteration[1]
        ax.scatter(v1, v2, color=colors[i])
        if i == 0:
            ax.text(v1+0.25, v2, f'$v$')
        if i == 1:
            ax.text(v1+0.25, v2, f'$Av$')
        if 1< i < 4:
            ax.text(v1+0.25, v2, f'$A^{i}v$')
            
    plt.show()
```

```{code-cell} ipython3
B = np.array([[sqrt(3) + 1, -2],
              [1, sqrt(3) - 1]])
B = (1/(2*sqrt(2))) * B
v = (-3,-3)
n = 12

plot_series(B, v, n)
```

Here with each iteration the vectors get shorter, i.e., move closer to the origin.

In this case, repeatedly multiplying a vector by $A$ makes the vector “spiral in”.

```{code-cell} ipython3
B = np.array([[sqrt(3) + 1, -2],
              [1, sqrt(3) - 1]])
B = (1/2) * B
v = (2.5,0)
n = 12

plot_series(B, v, n)
```

Here with each iteration vectors do not tend to get longer or shorter. 

In this case, repeatedly multiplying a vector by $A$ simply “rotates it around an ellipse”.

```{code-cell} ipython3
B = np.array([[sqrt(3) + 1, -2],
              [1, sqrt(3) - 1]])
B = (1/sqrt(2)) * B
v = (-1,-0.25)
n = 6

plot_series(B, v, n)
```

Here with each iteration vectors tend to get longer, i.e., farther from the origin. 

In this case, repeatedly multiplying a vector by $A$ makes the vector “spiral out”.

We thus observe that the sequence $(A^kv)_{k \geq 0}$ behaves differently depending on the map $A$ itself.

We now discuss the property of A that determines this behaviour.

+++

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

```{code-cell} ipython3
:tags: [output_scroll]

from numpy.linalg import eig

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
#ax.grid(alpha=0.4)

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

Using NumPy, we can solve for the eigenvalues and eigenvectors of a matrix as
follows

```{code-cell} ipython3
from numpy.linalg import eig

A = ((1, 2),
     (2, 1))

A = np.array(A)
evals, evecs = eig(A)
evals   #eigenvalues
```

```{code-cell} ipython3
evecs   #eigenvectors
```

Note that the *columns* of `evecs` are the eigenvectors.

Since any scalar multiple of an eigenvector is an eigenvector with the same
eigenvalue (check it), the eig routine normalizes the length of each eigenvector
to one.

The eigenvectors and eigenvalues of a map $A$ determine how a vector $v$ is transformed when we repeatedly multiply by $A$.

This is discussed further below.


### {index}`Nonnegative Matrices <single: Nonnegative Matrices>`

An $n \times m$ matrix $A$ is called **nonnegative** if every element of $A$ is nonnegative, i.e.,
$a_{ij} > 0$ for every $i = \{1,2,\cdots,n\}$ and $j = \{1,2,\cdots,m\}$.

We can denote this as $A > 0$.

Let $A$ be a square nonnegative matrix and let $A^k$ be the $k^{th}$ power of A.

Let $a^{k}_{ij}$ be element $(i,j)$ of $A^k$.

$A$ is called **irreducible** if for each $(i,j)$ there is an integer $k \geq 0$ such that $a^{k}_{ij} > 0$.

A matrix $A$ that is not irreducible is called reducible.

Here are some examples to illustrate this further.

1. $A = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.2 \end{bmatrix}$ is irreducible since $a_{ij}>0$ for all $(i,j)$.

2. $A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ is irreducible since $a_{12},a_{21} >0$ and $a^{2}_{11},a^{2}_{22} >0$.

3. $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ is reducible since $A^k = A$ for all $k \geq 0$ and thus
   $a^{k}_{12},a^{k}_{21} = 0$ for all $k \geq 0$.

## {index}`The Perron-Frobenius Theorem`

Often, in economics, the matrix that we are dealing with is nonnegative.

For a nonnegative matrix $A$ the behaviour of $A^k$ as $k \to \infty$ is controlled by the eigenvalue with the largest
absolute value, often called the **dominant eigenvalue**.

For a matrix $A$, the Perron-Frobenius theorem characterises certain properties of the dominant eigenvalue and its corresponding
eigenvector when $A$ is a nonnegative square matrix.

````{prf:theorem} Perron-Frobenius Theorem

If a matrix $A \geq 0$ then,
1. the dominant eigenvalue of A, r(A), is real-valued and nonnegative. 
2. for any other eigenvalue (possibly complex) $\lambda$ of $A$, $|\lambda| \leq r(A)$.
3. we can find a nonnegative and nonzero eigenvector $v$ such that $Av = r(A)v$.

Moreover if A is also irreducible then,
4. the eigenvector $v$ associated with the eigenvalue $r(A)$ is strictly positive.
5. there exists no other positive eigenvector $v$ (except scalar multiples of v) associated with $r(A)$.

````

+++

This is a very simple version of the theorem.

The extended version of the theorem can be found [here](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem).

We can use the Perron-Frobenius theorem to obtain a useful bounds on the dominant eigenvalue $r(A)$.

For a $n \times n$ matrix $A = (a_{ij})$, we set:
- $rowsum_i(A)$ := $\sum_j a_{ij}$ = the $i^{th}$ row sum of A 
- $colsum_j(A)$ := $\sum_i a_{ij}$ = the $j^{th}$ column sum of A

Then for $A \geq 0$:
```{math}
:label: PF_bounds

    \begin{aligned}
        min_i \; rowsum_i(A) \leq& r(A) \leq max_i \; rowsum_i(A) \\
        min_j \; colsum_j(A) \leq& r(A) \leq max_j \; colsum_j(A)
    \end{aligned}

```

### Application: Lake Model of Employment

The importance of the Perron-Frobenius theorem stems from the fact that firstly in the real world most matrices we encounter are
nonnegative matrices.

Secondly, a lot of important models are simply linear iterative models that begin with an initial condition $x_0$ and then evolve
recursively by the rule $x_{t+1} = Ax_t$ or in short $x_t = A^tx_0$.

This theorem helps characterise the dominant eigenvalue $r(A)$ which determines the behavior of this iterative process. 

We now illustrate the power of the Perron-Frobenius theorem by showing how it helps us to analyze a model of
employment and unemployment flows in a large population.

This model is sometimes called the **lake model** because there are two pools of workers:
1. those who are currently employed.
2. those who are currently unemployed but are seeking employment.

The "flows" between the two lakes are as follows:
1. workers exit the labor market at rate $d$.
2. new workers enter the labor market at rate $b$.
3. employed workers separate from their jobs at rate $\alpha$.
4. unemployed workers find jobs at rate $\lambda$.

Let $e_t$ and $u_t$ be the number of employed and unemployed workers at time $t$ respectively.

The total population of workers is $n_t = e_t + u_t$.

The number of unemployed and employed workers thus evolve according to:

```{math}
:label: lake_model
\begin{aligned}
    u_{t+1} = (1-d)(1-\lambda)u_t + \alpha(1-d)e_t + bn_t = ((1-d)(1-\lambda) +& b)u_t + (\alpha(1-d) + b)e_t \\
    e_{t+1} = (1-d)\lambda u_t + (1 - \alpha)(1-d)e_t&
\end{aligned}
```

We can arrange {eq}`lake_model` as a linear system of equations in matrix form $x_{t+1} = Ax_t$ such that:

$$
x_{t+1} =
\begin{bmatrix}
    u_{t+1} \\
    e_{t+1}
\end{bmatrix}
, \; A =
\begin{bmatrix}
    (1-d)(1-\lambda) + b & \alpha(1-d) + b \\
    (1-d)\lambda & (1 - \alpha)(1-d)
\end{bmatrix}
\; \text{and} \;
x_t =
\begin{bmatrix}
    u_t \\
    e_t
\end{bmatrix}
$$

Suppose at $t=0$ we have $x_0 = \begin{bmatrix} u_0 \\ e_0 \end{bmatrix}$.

Then, $x_1=Ax_0$, $x_2=Ax_1=A^2x_0$ and thus $x_t = A^tx_0$.

Thus the long run outcomes of this system depend on the initial condition $x_0$ and the matrix $A$.

$A$ is a nonnegative and irreducible matrix and thus we can use the Perron-Frobenius theorem to obtain some useful results about A.

Note that $colsum_j(A) = 1 + b - d$ for $j=1,2$ and by {eq}`PF_bounds` we can thus conclude that the dominant eigenvalue
$r(A) = 1 + b - d$.

If we consider $g = b - d$ as the overall growth rate of the total labor force, then $r(A) = 1 + g$.

We can thus find a unique positive vector $\bar{x} = \begin{bmatrix} \bar{u} \\ \bar{e} \end{bmatrix}$
such that $A\bar{x} = r(A)\bar{x}$ and $\begin{bmatrix} 1 & 1 \end{bmatrix} \bar{x} = 1$.

Since $\bar{x}$ is the eigenvector corresponding to the dominant eigenvalue $r(A)$ we can also call this the dominant eigenvector.

This eigenvector plays an important role in determining long run outcomes as is illustrated below.

```{code-cell} ipython3
:tags: []

def lake_model(α, λ, d, b):
    g = b-d
    A = np.array([[(1-d)*(1-λ) + b,   α*(1-d) + b],
              [(1-d)*λ,          (1-α)*(1-d)]])

    ū = (1 + g - (1 - d) * (1 - α)) / (1 + g - (1 - d) * (1 - α) + (1 - d) * λ)
    ē = 1 - ū
    x̄ = np.array([ū, ē])
    x̄.shape = (2,1)
    
    ts_length = 1000
    x_ts_1 = np.zeros((2, ts_length))
    x_ts_2 = np.zeros((2, ts_length))
    x_0_1 = (5.0, 0.1)
    x_0_2 = (0.1, 4.0)
    x_ts_1[0, 0] = x_0_1[0]
    x_ts_1[1, 0] = x_0_1[1]
    x_ts_2[0, 0] = x_0_2[0]
    x_ts_2[1, 0] = x_0_2[1]
    
    for t in range(1, ts_length):
        x_ts_1[:, t] = A @ x_ts_1[:, t-1]
        x_ts_2[:, t] = A @ x_ts_2[:, t-1]
        
    fig, ax = plt.subplots()
    
    # Set the axes through the origin
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position("zero")
    for spine in ["right", "top"]:
        ax.spines[spine].set_color("none")
        
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_xlabel("unemployed workforce")
    ax.set_ylabel("employed workforce")
    ax.set_xticks((0, 6))
    ax.set_yticks((0, 6))
    s = 10
    ax.plot([0, s * ū], [0, s * ē], "k--", lw=1)
    ax.scatter(x_ts_1[0, :], x_ts_1[1, :], s=4, c="blue")
    ax.scatter(x_ts_2[0, :], x_ts_2[1, :], s=4, c="green")
    
    ax.plot([ū], [ē], "ko", ms=4, alpha=0.6)
    ax.annotate(r'$\bar{x}$', 
            xy=(ū, ē),
            xycoords="data",
            xytext=(20, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle = "->"))

    x, y = x_0_1[0], x_0_1[1]
    ax.plot([x], [y], "ko", ms=2, alpha=0.6)
    ax.annotate(f'$x_0 = ({x},{y})$', 
            xy=(x, y),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle = "->"))
    
    x, y = x_0_2[0], x_0_2[1]
    ax.plot([x], [y], "ko", ms=2, alpha=0.6)
    ax.annotate(f'$x_0 = ({x},{y})$', 
            xy=(x, y),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle = "->"))
    
    plt.show()
```

```{code-cell} ipython3
lake_model(α=0.01, λ=0.1, d=0.02, b=0.025)
```

If $\bar{x}$ is an eigenvector corresponding to the eigenvalue $r(A)$ then all the vectors in the set
$D := \{ x \in \mathbb{R}^2 : x = \alpha \bar{x} \; \text{for some} \; \alpha >0 \}$ are also eigenvectors corresponding
to $r(A)$.

This set is represented by a dashed line in the above figure.

We can observe that for two distinct initial conditions $x_0$ the sequence of iterates $(A^t x_0)_{t \geq 0}$ move towards D over time.

This suggests that all such sequences share strong similarities in the long run, determined by the dominant eigenvector $\bar{x}$.

In the example illustrated above we considered parameter such that overall growth rate of the labor force $g>0$.

Suppose now we are faced with a situation where the $g<0$, i.e, negative growth in the labor force.

This means that $b-d<0$, i.e., workers exit the market faster than they enter.

What would the behavior of the iterative sequence $x_{t+1} = Ax_t$ be now?

This is visualised below.

```{code-cell} ipython3
lake_model(α=0.01, λ=0.1, d=0.025, b=0.02)
```

Thus, while the sequence of iterates still move towards the dominant eigenvector $\bar{x}$ however in this case
they converge to the origin.

This is a result of the fact that $r(A)<1$, which ensures that the iterative sequence  $(A^t x_0)_{t \geq 0}$ will converge
to some point, in this case to $(0,0)$.

This leads us into the next result.


(la_neumann)=
## {index}`The Neumann Series Theorem <single: Neumann's Theorem>`

```{index} single: Neumann's Theorem
```

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

A generalization of this idea exists in the matrix setting.

Consider the system of equations $x = Ax + b$ where $A$ is an $n \times n$ square matrix and
$x$ and $b$ are both column vectors in $\mathbb{R}^n$.

Using matrix algebra we can conclude that the solution to this system of equations will be given by:

```{math}
:label: neumann_eqn
    
    x^{*} = (I-A)^{-1}b

```

What guarantees the existence of a unique vector $x^{*}$ that satisfies {eq}`neumann_eqn` ?

The following is a fundamental result in functional analysis that generalises {eq}`gp_sum` to a multivariate case.

````{prf:theorem} Neumann Series Lemma
Let $A$ be a square matrix and let $A^k$ be the $k$-th power of $A$.

Let $r(A)$ be the dominant eigenvector or as it is commonly called the *spectral radius*, defined as $\max_i |\lambda_i|$, where 

* $\{\lambda_i\}_i$ is the set of eigenvalues of $A$ and
* $|\lambda_i|$ is the modulus of the complex number $\lambda_i$

Neumann's theorem states the following: If $r(A) < 1$, then $I - A$ is invertible, and

$$
(I - A)^{-1} = \sum_{k=0}^{\infty} A^k
$$
````

+++

We verify this lemma with the following example.

```{code-cell} ipython3
A = np.array([[0.4, 0.1],
              [0.7, 0.2]])

evals, evecs = eig(A)   #finding eigenvalues and eigenvectors

r = max(abs(λ) for λ in evals)    #finding spectral radius
print(r)
```

The spectral radius $r(A)$ obtained is less than 1. 

Thus, we can apply the Neumann Series lemma to find $(I-A)^{1}$.

```{code-cell} ipython3
I = np.identity(2)      #2 x 2 identity matrix
B = I-A
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

```{code-cell} ipython3
np.allclose(A_sum, B_inverse)     #checking equality between the sum and the inverse method
```

Although we truncate the infite sum at $k = 50$,
both methods give us the same result which illustrates the result of the Neumann Series lemma.

+++


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

The first row depicts how agriculture's total output $x_1$ is distributed - $30\%$ is used as inputs within agriculture itself,
$0.2x_2$ is used as inputs by the industry sector to produce $x_2$ units, $0.2x_3$ is used as inputs by the service sector to produce $x_3$ units 
and 4 units is the external demand by consumers.

We can transform this into a system of linear equations for the 3 sectors as given below:

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

```{code-cell} ipython3

```
