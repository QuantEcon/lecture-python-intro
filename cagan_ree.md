## A fiscal theory of the price level

<!-- #region -->
We'll use linear algebra to do some experiments with  a "fiscal theory of the price level".

According to this model, when the government persistently spends more than it collects in taxes and prints money to finance the shortfall (called the "government deficit"), it puts upward pressure on the price level and generates
persistent inflation.

Our model is a "rational expectations" (or "perfect foresight") version of a model that Philip Cagan  used to study the monetary dynamics of hyperinflations.  

While Cagan didn't use the perfect foresight, or "rational expectations" version of the model, Thomas Sargent did when
he studied the Ends of Four Big Inflations.

To facilitate using  linear matrix algebra as our only mathematical tool, we'll use a finite horizon version of
the model.

Let 

 * $ m_t $ be the log of the supply of  nominal money balances;
 * $\mu_t = m_{t+1} - m_t $ be the net rate of growth of  nominal balances;
 * $p_t $ be the log of the price level;
 * $\pi_t = p_{t+1} - p_t $ be the net rate of inflation  between $t$ and $ t+1$;
 * $\pi_t^*$  be the public's expected rate of inflation between  $t$ and $t+1$;
 * $T$ the horizon -- i.e., the last period for which the model will determine $p_t$
 * $\pi_{T+1}^*$ the terminal rate of inflation between times $T$ and $T+1$.
  
  
The demand for real balances $\exp\left(\frac{m_t^d}{p_t}\right)$ is governed by the following  version of the Cagan demand function
  
$$  
m_t^d - p_t = -\alpha \pi_t^* \: , \: \alpha > 0 ; \quad t = 0, 1, \ldots, T .
$$ (eq:caganmd)


This equation  asserts that the demand for real balances
is inversely related to the public's expected rate of inflation.


People somehow  acquire **perfect foresight** by their having solved a forecasting
problem.

This lets us set


$$ 
\pi_t^* = \pi_t , % \forall t 
$$ (eq:ree)

while equating demand for money to  supply lets us set $m_t^d = m_t$ for all $t \geq 0$. 

The preceding equations  then   imply

$$
m_t - p_t = -\alpha(p_{t+1} - p_t) \: , \: \alpha > 0 
$$ (eq:cagan)

To fill in details about what it means for private agents
to have perfect foresight,  we subtract equation {eq}`eq:cagan`  at time $ t $ from the same equation at $ t+1$ to get

$$
\mu_t - \pi_t = -\alpha \pi_{t+1} + \alpha \pi_t  ,
$$

which we rewrite as a forward-looking first-order linear difference
equation in $\pi_s$ with $\mu_s$  as a "forcing variable":

\begin{equation}  
\pi_t = \frac{\alpha}{1+\alpha} \pi_{t+1} + \frac{1}{1+\alpha} \mu_t , \quad t= 0, 1, \ldots , T 
\end{equation}

where $ 0< \frac{\alpha}{1+\alpha} <1 $.

Setting $\delta =\frac{1}{1+\alpha}$ let's us represent the preceding equation as

\begin{equation}
\pi_t = (1-\delta) \mu_t + \delta \pi_{t+1} , \quad t =0, 1, \ldots, T
\end{equation}

Write this system of $T+1$ equations as the single matrix equation

$$
\begin{bmatrix} 1 & -\delta & 0 & 0 & \cdots & 0 & 0 \cr
                0 & 1 & -\delta & 0 & \cdots & 0 & 0 \cr
                0 & 0 & 1 & -\delta & \cdots & 0 & 0 \cr
                \vdots & \vdots & \vdots & \vdots & \vdots & 0 & 0 \cr
                0 & 0 & 0 & 0 & \cdots & 1 & -\delta \cr
                0 & 0 & 0 & 0 & \cdots & 0 & 1 \end{bmatrix}
\begin{bmatrix} \pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_{T-1} \cr \pi_T 
\end{bmatrix} 
= (1 - \delta) \begin{bmatrix}  
\mu_0 \cr \mu_1 \cr \mu_2 \cr \vdots \cr \mu_{T-1} \cr \mu_T
\end{bmatrix}
+ \begin{bmatrix} 
0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr \delta \pi_{T+1}^*
\end{bmatrix}
$$ (eq:pieq)

By multiplying both sides of equation {eq}`eq:pieq` by the inverse of the matrix on the left side, we can calculate

$$
\vec \pi \equiv \begin{bmatrix} \pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_{T-1} \cr \pi_T 
\end{bmatrix} 
$$

It turns out that

$$
\pi_t = (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s +  \delta^{T+1-t} \pi_{T+1}^*
$$ (eq:fisctheory1)

We can represent the equations 

$$ 
m_{t+1} = m_t + \mu_t , \quad t = 0, 1, \ldots, T
$$

as the matrix equation

$$
\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 & 0 \cr
-1 & 1 & 0 & \cdots & 0 & 0 \cr
0  & -1 & 1 & \cdots & 0 & 0 \cr
\vdots  & \vdots & \vdots & \vdots & 0 & 0 \cr
0  & 0 & 0 & \cdots & 1 & 0 \cr
0  & 0 & 0 & \cdots & -1 & 1 
\end{bmatrix}
\begin{bmatrix}  
m_1 \cr m_2 \cr m_3 \cr \vdots \cr m_T \cr m_{T+1}
\end{bmatrix}
= \begin{bmatrix}  
\mu_0 \cr \mu_1 \cr \mu_2 \cr \vdots \cr \mu_{T-1} \cr \mu_T
\end{bmatrix}
+ \begin{bmatrix}  
m_0 \cr 0 \cr 0 \cr \vdots \cr 0 \cr 0
\end{bmatrix}
$$ (eq:eq101)

Multiplying both sides of equation {eq}`eq:eq101`  with the inverse of the matrix on the left will give 

$$
m_t = m_0 + \sum_{s=0}^{t-1} \mu_s, \quad t =1, \ldots, T+1
$$ (eq:mcum)

Equation {eq}`eq:mcum` shows that the log of the money supply at $t$ equals the log $m_0$ of the initial money supply 
plus accumulation of rates of money growth between times $0$ and $t$.

#### The Log Price Level


We can use equations {eq}`eq:caganmd` and {eq}`eq:ree`
to discover that the log of the price level satisfies

$$
p_t = m_t + \alpha \pi_t
$$ (eq:pfiscaltheory)

or, by using equation  {eq}`eq:fisctheory1`,

$$ 
p_t = m_t + \alpha \left[ (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s +  \delta^{T+1-t} \pi_{T+1}^*  \right] 
$$ (eq:pfiscaltheory2)
<!-- #endregion -->

<!-- #region -->
### Requests for Jiacheng


I'd like to do some experiments that simply involve inputting 
$ m_0, T,  \pi_{T+1}^*, \{\mu_s\}_{s=0}^T$ and graphing as outputs $\{p_t\}_{t=0}^T, \{m_t\}_{t=0}^T, \{\mu_t\}_{t=0}^T,
\{\pi_t\}_{t=0}^T$.


Some examples of interesting   $\{\mu_s\}_{s=0}^T$ processes -- with high value of $T$ are

  * geometric growth or decay $\mu_t  = \gamma_\mu^t$.
  
  * geometric growth or decay $\mu_t  = \gamma_\mu^t$ for $t = 0, \ldots, T_1$ followed by constant $\mu_t$ for $t > T_1$ (to represent a foreseen stabilization)
  
With this little machinery we can show some important principles that constitute about 75\% of John Cochrane's new book!
<!-- #endregion -->
