<!-- #region -->
## Present Values


This lecture describes the  **present value model** that is a starting point
of much asset pricing theory.

We'll use the calculations described here in several subsequent lectures.

Our only tool is some elementary linear algebra operations, namely, matrix multiplication and matrix inversion.

Let's dive in.

Let 

 * $\{d_t\}_{t=0}^T $ be a sequence of dividends or ``payouts''
 
 * $\{p_t\}_{t=0}^T $ be a sequence of prices of a claim on the continuation of
    the asset stream from date $t$ on, namely, $\{p_s\}_{s=t}^T $ 
    
 * $ \delta  \in (0,1) $ be a one-period ``discount rate'' 
 
 * $p_{T+1}^*$ a terminal price of the asset at time $T+1$
 
We  assume that the dividend stream $\{d_t\}_{t=0}^T $ and the terminal price 
$p_{T+1}$ are both exogenous.

Assume the sequence of asset pricing equations

$$
p_t = d_t + \delta p_{t+1}, \quad t = 0, 1, \ldots , T
$$ (eq:Euler1)


We want to solve for the asset price sequence  $\{p_t\}_{t=0}^T $ as a function
of $\{d_t\}_{t=0}^T $ and $p_{T+1}^*$.



Write the system {eq}`eq:Euler1` of $T+1$ asset pricing  equations as the single matrix equation

$$
\begin{bmatrix} 1 & -\delta & 0 & 0 & \cdots & 0 & 0 \cr
                0 & 1 & -\delta & 0 & \cdots & 0 & 0 \cr
                0 & 0 & 1 & -\delta & \cdots & 0 & 0 \cr
                \vdots & \vdots & \vdots & \vdots & \vdots & 0 & 0 \cr
                0 & 0 & 0 & 0 & \cdots & 1 & -\delta \cr
                0 & 0 & 0 & 0 & \cdots & 0 & 1 \end{bmatrix}
\begin{bmatrix} p_0 \cr p_1 \cr p_2 \cr \vdots \cr p_{T-1} \cr p_T 
\end{bmatrix} 
=  \begin{bmatrix}  
d_0 \cr d_1 \cr d_2 \cr \vdots \cr d_{T-1} \cr d_T
\end{bmatrix}
+ \begin{bmatrix} 
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr \delta p_{T+1}^*
\end{bmatrix}
$$ (eq:pieq)

Call the matrix on the left side of equation {eq}`eq:pieq` $A$.


It is easy to verify that the  inverse of the matrix on the left side of equation
{eq}`eq:pieq` is


$$ A^{-1} = 
\begin{bmatrix}
1 & \delta & \delta^2 & \cdots & \delta^{T-1} & \delta^T \cr
0 & 1 & \delta & \cdots & \delta^{T-2} & \delta^{T-1} \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & 1  & \delta \cr
0 & 0 & 0 & \cdots & 0 & 1 \cr
\end{bmatrix}
$$ (eq:Ainv)

By multiplying both sides of equation {eq}`eq:pieq` by the inverse of the matrix on the left side, we can calculate

$$
\vec p \equiv \begin{bmatrix} p_0 \cr p_1 \cr p_2 \cr \vdots \cr p_{T-1} \cr p_T 
\end{bmatrix} 
$$

If we perform the indicated matrix multiplication, we shall find  that

$$
p_t =  \sum_{s=t}^T \delta^{s-t} d_s +  \delta^{T+1-t} p_{T+1}^*
$$ (eq:fisctheory1)

Pricing formula {eq}`eq:fisctheory1` asserts that  two components sum to the asset price 
$p_t$:

  * a **fundamental component** $\sum_{s=t}^T \delta^{s-t} d_s$ that equals the discounted present value of prospective dividends
  
  * a **bubble component** $\delta^{T+1-t} p_{T+1}^*$
  
It is sometimes convenient to rewrite the bubble component as

$$ 
c \delta^{-t}
$$

where 

$$ 
c \equiv \delta^{T+1}p_{T+1}^*
$$


#### More about bubbles

For a few moments, let's focus on  the special case of an asset that  will never pay dividends, in which case

$$
\begin{bmatrix}  
d_0 \cr d_1 \cr d_2 \cr \vdots \cr d_{T-1} \cr d_T
\end{bmatrix} = 
\begin{bmatrix}  
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr 0
\end{bmatrix}
$$


In this case  system {eq}`eq:Euler1` of our $T+1$ asset pricing  equations takes the
form of the single matrix equation

$$
\begin{bmatrix} 1 & -\delta & 0 & 0 & \cdots & 0 & 0 \cr
                0 & 1 & -\delta & 0 & \cdots & 0 & 0 \cr
                0 & 0 & 1 & -\delta & \cdots & 0 & 0 \cr
                \vdots & \vdots & \vdots & \vdots & \vdots & 0 & 0 \cr
                0 & 0 & 0 & 0 & \cdots & 1 & -\delta \cr
                0 & 0 & 0 & 0 & \cdots & 0 & 1 \end{bmatrix}
\begin{bmatrix} p_0 \cr p_1 \cr p_2 \cr \vdots \cr p_{T-1} \cr p_T 
\end{bmatrix}  =
\begin{bmatrix} 
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr \delta p_{T+1}^*
\end{bmatrix}
$$ (eq:pieq2)

Evidently, if $p_{T+1}^* = 0$, a price vector $\vec p$ of with all entries zero
solves this equation and the only the **fundamental** component of our pricing 
formula {eq}`eq:fisctheory1` is present. 

But let's activate the **bubble**  component by setting 

$$
p_{T+1}^* = c \delta^{-(T+1)} 
$$ (eq:eqbubbleterm)

for some positive constant $c$.

In this case, it can be verified that when we multiply both sides of {eq}`eq:pieq2` by
the matrix $A^{-1}$ presented in equation {eq}`eq:Ainv`, we shall find that

$$
p_t = c \delta^{-t}
$$ (eq:bubble)


#### Gross rate of return

Define the gross rate of return on holding the asset from period $t$ to period $t+1$
as 

$$
R_t = \frac{p_{t+1}}{p_t}
$$ (eq:rateofreturn)

Equation {eq}`eq:bubble` confirms that an asset whose  sole source of value is a bubble 
earns a  gross rate of return of

$$
R_t = \delta^{-1} > 1 .
$$



<!-- #endregion -->

## Experiments

We'll try various settings for $\vec d, p_{T+1}^*$:

  * $p_{T+1}^* = 0, d_t = g^t d_0$ to get a modified version of the Gordon growth formula
  
  * $p_{T+1}^* = g^{T+1} d_0,  d_t = g^t d_0$ to get the plain vanilla  Gordon growth formula
  
  * $p_{T+1}^* = 0, d_t = 0$ to get a worthless stock
  
  * $p_{T+1}^* = c \delta^{-(T+1)}, d_t = 0$ to get a bubble stock 
