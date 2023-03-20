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

(input_output)=

In this lecture, we will need the following library.

+++

# Input-Output Models

## Overview

We adopt notation in chapters 8 and 9 of the classic book {cite}`DoSSo`.

We let 

 * $X_0$ be the amount of a single exogenous input to production. We'll call this input labor
 * $X_j, j = 1,\ldots n$ be the gross output of final good $j$
 *  $C_j, j = 1,\ldots n$ be the net output of final good $j$ that is available for final consumption
 * $x_{ij} $ be the quantity of good $i$ allocated to be  an input to producing good $j$ for $i=1, \ldots n$, $j = 1, \ldots n$
 * $x_{0j}$ be the quantity of labor allocated  to produce one unit of good $j$.
 * $a_{ij}$ be the number of units of good $i$ required to produce one unit of good $j$, $i=0, \ldots, n, j= 1, \ldots n$. 
 * $w >0$ be the exogenous wage of labor, denominated in dollars per unit of labor
 * $p$ be an $n \times 1$ vector of prices of produced goods $i = 1, \ldots , n$. 
 


The production function for goods $j \in \{1, \ldots , n\}$ is the **Leontief** function

$$
X_j = \min_{i \in \{0, \ldots , n \}} \left( \frac{x_{ij}}{a_{ij}}\right) 
$$


To illustrate ideas, we'll begin by setting $n =2$.

Feasible allocations must satisfy

$$
\begin{aligned}
(1 - a_{11}) X_1 - a_{12} X_2 & \geq C_1 \cr 
-a_{21} X_1 + (1 - a_{22}) X_2 & \geq C_2 \cr 
a_{01} X_1 + a_{02} X_2 & \leq X_0 
\end{aligned}
$$

or more generally

$$
\begin{aligned}
(I - a) X &  \geq C \cr 
a_0^\top X & \leq X_0
\end{aligned}
$$ (eq:inout_1)

where $a$ is the $n \times n$ matrix with typical element $a_{ij}$ and $a_0^\top = \begin{bmatrix} a_{01} & \cdots & a_{0n} \end{bmatrix}$.



If we solve the first block of equations of {eq}`eq:inout_1` for gross output $X$ we get 

$$ 
X = (I -a )^{-1} C \equiv A C 
$$ (eq:inout_2)

where $A = (I-a)^{-1}$.  

The coefficient $A_{ij} $ is the amount of good $i$ that is required as an intermediate input to produce one unit of final output $j$.

We assume the **Hawkins-Simon condition** 

$$ 
\det (I - a) > 0 
$$

to assure that the solution $X$ of {eq}`eq:inout_2` is a positive vector. 


## Production Possibility Frontier

The second equation of {eq}`eq:inout_1` can be written

$$
a_0^\top X = X_0 
$$

or 

$$
A_0^\top C = X_0
$$ (eq:inout_frontier)

where

$$
A_0^\top = a_0^\top (I - a)^{-1}
$$

The $i$th Component $A_0$ is the amount of labor that is required to produce one unit of final output of good $i$ for $i \in \{1, \ldots , n\}$.

Equation {eq}`eq:inout_frontier` sweeps out a  **production possibility frontier** of final consumption bundles $C$ that can be produced with exogenous labor input $X_0$. 


## Prices

{cite}`DoSSo` argue that relative prices of the $n$ produced goods must satisfy  

$$ 
p = a^\top p + a_0 w
$$

which states that the price of each final good equals the total cost 
of production, which consists of costs of intermediate inputs $a^\top p$
plus costs of labor $a_0 w$.

This equation can be written as 

$$
(I - a^\top) p = a_0 w
$$ (eq:inout_price)

which implies

$$
p = (I - a^\top)^{-1} a_0 w
$$

Notice how  {eq}`eq:inout_price` with {eq}`eq:inout_1` form a
**conjugate pair**  through the appearance of operators 
that are transposes of one another.  

This connection surfaces again in a classic linear program and its dual.


## Linear Programs

A **primal** problem is 

$$
\min_{X} w a_0 ^\top X 
$$

subject to 

$$
(I -a ) X \geq C
$$


The associated **dual** problem is

$$
\max_{p} p^\top C 
$$

subject to

$$
(I -a)^\top p \leq a_0 w 
$$

The primal problem chooses a feasible production plan to minimize costs for delivering a pre-assigned vector of final goods consumption $C$.

The dual problem chooses prices to maxmize the value of a pre-assigned vector of final goods $C$ subject to prices covering costs of production. 

Under  sufficient conditions discussed XXXX, optimal value of the primal and dual problems coincide:

$$
w a_0^\top X^* = p^* C
$$

where $^*$'s denote optimal choices for the primal and dual problems.






+++


## Exercise

{cite}`DoSSo`, chapter 9, carries along an example with the following
parameter settings:



$$ 
a = \begin{bmatrix} .1 & 1.46 \cr
    .16 & .17 \end{bmatrix}
$$

$$ 
a_0 = \begin{bmatrix} .04 & .33 \end{bmatrix}
$$

$$
C = \begin{bmatrix} 50 \cr 60 \end{bmatrix}
$$

$$ 
X_0 = \begin{bmatrix} 250 \cr 120 \end{bmatrix}
$$

$$
X = 50
$$


```{code-cell} ipython3

```

```{code-cell} ipython3

```
