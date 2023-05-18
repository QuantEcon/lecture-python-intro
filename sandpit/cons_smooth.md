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

## Some dynamic models with matrices

In this notebook, we'll present  some useful models of economic dynamics using only linear algebra -- matrix multiplication and matrix inversion.

**Present value formulas** are at the core of the models. 

+++

## Consumption smoothing

Let 

 * $T \geq 2$  be a positive integer that constitutes a time-horizon

 * $\vec y = \{y_t\}_{t=0}^T$ be an exogenous  sequence of non-negative financial incomes $y_t$

 * $\vec a = \{a_t\}_{t=0}^{T+1}$ be a sequence of financial wealth
 
 * $\vec c = \{c_t\}_{t=0}^T$ be a sequence of non-negative consumption rates

 * $R \geq 1$ be a fixed gross one period rate of return on financial assets
 
 * $\beta \in (0,1)$ be a fixed discount factor

 * $a_0$ be a given initial level of financial assets

 * $a_{T+1} \geq 0$  be a terminal condition on final assets

A sequence of budget constraints constrains the triple of sequences $\vec y, \vec c, \vec a$

$$
a_{t+1} = R (a_t+ y_t - c_t), \quad t =0, 1, \ldots T
$$

Our model has the following logical flow

 * start with an exogenous income sequence $\vec y$, an initial financial wealth $a_0$, and 
 a candidate consumption path $\vec c$.
 
 * use equation (1) to compute a path $\vec a$ of financial wealth
 
 * verify that $a_{T+1}$ satisfies the terminal wealth constraint $a_{T+1} \geq 0$. 
    
     * If it does, declare that the candiate path is budget feasible. 
 
     * if the candidate consumption path is not budget feasible, propose a path with less consumption sometimes and start over
     
Below, we'll describe how to execute these steps using linear algebra -- matrix inversion and multiplication.


We shall eventually evaluate alternative budget feasible consumption paths $\vec c$ using the following **welfare criterion**

$$
W = \sum_{t=0}^T \beta^t (g_1 c_t - \frac{g_2}{2} c_t^2 )
$$

where $g_1 > 0, g_2 > 0$.  

We shall see that when $\beta R = 1$ (a condition assumed by Milton Friedman and Robert Hall), this criterion assigns higher welfare to **smoother** consumption paths.




+++

## Difference equations with linear algebra ##

As a warmup, we'll describe a useful way of representing and "solving" linear difference equations. 

To generate some $y$ vectors, we'll just write down a linear difference equation
with appropriate initial conditions and then   use linear algebra to solve it.

#### First-order difference equation

A first-order linear difference equation cast as a matrix equation

$$
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 \cr
              -\lambda & 1 & 0 & \cdots & 0 & 0 \cr
                0 & -\lambda & 1 & \cdots & 0 & 0 \cr
                 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
                0 & 0 & 0 & \cdots & -\lambda & 1 

\end{bmatrix} \begin{bmatrix} y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T \end{bmatrix}
= \begin{bmatrix} \lambda y_0 \cr 0 \cr 0 \cr \vdots \cr 0 \end{bmatrix}
$$

Here $y_0$ is an initial condition.

Multiplying both sides by  inverse of the matrix on the left provides the solution

$$
\begin{bmatrix} y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T \end{bmatrix} = 
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 \cr
              \lambda & 1 & 0 & \cdots & 0 & 0 \cr
                \lambda^2 & \lambda & 1 & \cdots & 0 & 0 \cr
                 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
                \lambda^{T-1} & \lambda^{T-2} & \lambda^{T-3} & \cdots & -\lambda & 1 
\end{bmatrix}
\begin{bmatrix} \lambda y_0 \cr 0 \cr 0 \cr \vdots \cr 0 \end{bmatrix}

$$


#### Second order difference equation


$$
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
              -\lambda_1 & 1 & 0 & \cdots & 0 & 0 & 0 \cr
                -\lambda_2 & -\lambda_1 & 1 & \cdots & 0 & 0 & 0 \cr
                 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
                0 & 0 & 0 & \cdots & \lambda_2 & -\lambda_1 & 1 

\end{bmatrix} \begin{bmatrix} y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T \end{bmatrix}
= \begin{bmatrix} \lambda_1 y_0 + \lambda_2 y_{-1} \cr \lambda_2 y_0 \cr 0 \cr \vdots \cr 0 \end{bmatrix}
$$

Multiplying both sides by  inverse of the matrix on the left again provides the solution.

#### Extensions

As an exercise, we ask you to represent and solve a **third order linear difference equation**.
How many initial conditions must you specify?

+++

## Friedman-Hall consumption-smoothing model


A key object is what Milton Friedman called "non-human" or "non-financial" wealth at time $0$:


$$
h_0 \equiv \sum_{t=0}^T R^t y_t = \begin{bmatrix} 1 & R & \cdots & R^T \end{bmatrix}
\begin{bmatrix} y_0 \cr y_1  \cr \vdots \cr y_T \end{bmatrix}
$$

By iterating on equation (1) and imposing the terminal condition 

$$
a_{T+1} = 0,
$$

it is possible to convert a sequence of budget constraints into the single intertemporal constraint

$$
\sum_{t=0}^T R^t c_t = a_0 + h_0,
$$

which says that the present value of the consumption stream equals the sum of finanical and non-financial wealth.

Robert Hall (1978) showed that when $\beta R = 1$, a condition Milton Friedman had assumed,
it is "optimal" for a consumer to **smooth consumption** by setting 

$$ 
c_t = c_0 \quad t =0, 1, \ldots, T
$$

In this case, we can use the intertemporal budget constraint to write 

$$
c_0 = \left(\sum_{t=0}^T R^t\right)^{-1} (a_0 + h_0)
$$

This is the consumption-smoothing model in a nutshell.

We'll put the model through some paces with Python code below.

+++

## Permanent income model of consumption 

As promised, we'll provide step by step instructions on how to use linear algebra, readily implemented
in Python, to solve the consumption smoothing model.

**Note to programmer teammate:**

In the calculations below, please we'll  set default values of  $R > 1$, e.g., $R = 1.05$, and $\beta = R^{-1}$.

#### Step 1 ####

For some $T+1 \times 1$ $y$ vector, use matrix algebra to compute 

$$
\sum_{t=0}^T R^t y_t = \begin{bmatrix} 1 & R & \cdots & R^T \end{bmatrix}
\begin{bmatrix} y_0 \cr y_1  \cr \vdots \cr y_T \end{bmatrix}
$$

#### Step 2 ####

Compute

$$
c_0 = \left( \frac{1 - R^{-1}}{1 - R^{-(T+1)}} \right) (a_0 + \sum_{t=0}^T R^t y_t )
$$


#### Step 3 ####

Formulate system

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-R & 1 & 0 & \cdots & 0 & 0 & 0 \cr
0 & -R & 1 & \cdots & 0 & 0 & 0 \cr
\vdots  &\vdots & \vdots & \cdots & \vdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -R & 1 & 0 \cr
0 & 0 & 0 & \cdots & 0 & -R & 1
\end{bmatrix} \begin{bmatrix} a_1 \cr a_2 \cr a_3 \cr \vdots \cr a_T \cr a_{T+1} \end{bmatrix}
= R \begin{bmatrix} y_0 + a_0 - c_0 \cr y_1 - c_0 \cr y_2 - c_0 \cr \vdots\cr y_T - y_0 \cr 0
\end{bmatrix}

$$

Multiply both sides by the inverse of the matrix on the left side to compute

$$
 \begin{bmatrix} a_1 \cr a_2 \cr a_3 \cr \vdots \cr a_T \cr a_{T+1} \end{bmatrix}
$$

It should turn out automatically  that 

$$
a_{T+1} = 0.
$$

Let's verify this with our Python code.



### Feasible consumption variations ###

To explore what types of consumption paths are welfare-improving, we shall create an **admissible consumption path variation** sequence $\{v_t\}_{t=0}^T$
that satisfies

$$
\sum_{t=0}^T v_t = 0
$$

We'll compute a two-parameter class of admissible variations
of the form

$$
v_t = \xi_1 \phi^t - \xi_0
$$

We say two and not three-parameter class because $\xi_0$ will be a function of $(\phi, \xi_1; R)$ that guarantees that the variation is feasibile. 

Let's compute that function.

We require

$$
\sum_{t=0}^T \left[ \xi_1 \phi^t - \xi_0 \right] = 0
$$

which implies that

$$
\xi_1 \sum_{t=0}^T \phi_t R^{-t} - \xi_0 \sum_{t=0}^T R^{-t} = 0
$$

which implies that

$$
\xi_1 \frac{1 - (\phi R^{-1})^{T+1}}{1 - \phi R^{-1}} - \xi_0 \frac{1 - R^{-(T+1)}}{1-R^{-1} } =0
$$

which implies that

$$
\xi_0 = \xi_0(\phi, \xi_1; R) = \xi_1 \left(\frac{1 - R^{-1}}{1 - R^{-(T+1)}}\right) \left(\frac{1 - (\phi R^{-1})^{T+1}}{1 - \phi R^{-1}}\right)
$$ 

This is our formula for $\xi_0$.  

Evidently, if $\vec c^o$ is a budget-feasible consumption path, then so is $\vec c^o + \vec v$,
where $\vec v$ is a budget-feasible variation.

Given $R$, we thus have a two parameter class of budget feasible variations $\vec v$ that we can use
to compute alternative consumption paths, then evaluate their welfare.

**Note to John:** We can do some fun simple experiments with these variations -- we can use
graphs to show that, when $\beta R =1$ and  starting from the smooth path, all nontrivial budget-feasible variations lower welfare according to the criterion above.  

We can even use the Python numpy grad command to compute derivatives of welfare with respect to our two parameters.  Notice that we are teaching the key idea beneath the calculus of variations.


```{code-cell} ipython3

```
