# A fiscal theory of the price level

```python
import numpy as np
import matplotlib.pyplot as plt
```

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


## Continuation values


To determine the continuation inflation rate $\pi_{T+1}^*$ we shall proceed by applying the following infinite-horizon
version of equation {eq}`eq:fisctheory1` at time $t = T+1$:


$$
\pi_t = (1-\delta) \sum_{s=t}^\infty \delta^{s-t} \mu_s , 
$$ (eq:fisctheory2)

and by also assuming the following continuation path for $\mu_t$ beyond $T$:

$$
\mu_{t+1} = \gamma^* \mu_t, \quad t \geq T .
$$

Plugging the preceding equation into equation {eq}`eq:fisctheory2` at $t = T+1$ and rearranging we can deduce that



$$ 
\pi_{T+1}^* = \frac{1 - \delta}{1 - \delta \gamma^*} \gamma^* \mu_T
$$ (eq:piterm)

where we require that $\vert \gamma^* \delta \vert < 1$.

In the experiments below, we'll use formula {eq}`eq:piterm` as our terminal condition for expected inflation.

In devising these experiments, we'll  make assumptions about $\{\mu_t\}$ that are consistent with formula
{eq}`eq:piterm`.

### Note to Jiacheng

Hi. You did a great job -- much better than I did.

I want to alter our three experiments -- it will simplify them and make the points that I want to make.
It will involve some changes in your code for the experiments, but not big ones I hope.

Below I'll describe the changes.

### End of Note to Jiacheng


We now describe three such experiments.

In all of these experiments, 

$$ 
\mu_t = \mu^* , \quad t \geq T_1
$$

so that, in terms of our notation and formula for $\theta_{T+1}^*$ above, $\tilde \gamma = 1$. 

### Experiment 1

**A foreseen stabilization**

Let $T_1 \in (0, T)$. Assume that 

$$
\mu_{t+1} = \begin{cases}
    \mu_0  , & t = 0, \ldots, T_1 -1 \\
     \mu^* , & t \geq T_1
     \end{cases}
$$

### Experiment 2

**Gradual stabilization**



Suppose that $\phi \in (0,1)$, that  $\mu_0 > \mu^*$,  and that for $t = 0, \ldots, T-1$

$$
\mu_t = \phi^t \mu_0 + (1 - \phi^t) \mu^* .
$$ 


### Experiment 3

**An MIT shock**

A completely unanticipated shock is popularly known as an "MIT shock".

This imagines  comparing two paths for $\{\mu_t\}_{t=0}^\infty$.

**Path 1:** $\mu_t = \mu_0$ for all $t \geq 0$. 

**Path 2:** Where $ \mu_0 > \mu^*$, $\mu_t = \mu_0$ for all $t \geq 0$.





The idea is first to construct **two** $\{\mu_t, \pi_t, m_t, p_t\}$ sequences,
one for each of our two assumptions.

The MIT shock procedure then glues together the first part of the first sequence with the second part of the second sequence.

We can do the MIT shock calculations mostly by hand. 

Thus, for path 1, $\pi_t = \mu_0 $ for all $t \geq 0$, while for path 2,
$\mu_t = \mu^*$ for all $t \geq 0$.  



### Experiment 4

Eventually, we might want to perform the following experiment right after we do experiment 3.


Experiment 4 is just a simple version of a foreseen stablilization like experiment 1.  

$$
\mu_{t} = \begin{cases}
    \mu_0,  & t = 0, \ldots, T_1 -1 \\
     \mu^*  & t \geq T_1
     \end{cases}
$$


### The Log Price Level


We can use equations {eq}`eq:caganmd` and {eq}`eq:ree`
to discover that the log of the price level satisfies

$$
p_t = m_t + \alpha \pi_t
$$ (eq:pformula2)

or, by using equation  {eq}`eq:fisctheory1`,

$$ 
p_t = m_t + \alpha \left[ (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s +  \delta^{T+1-t} \pi_{T+1}^*  \right] 
$$ (eq:pfiscaltheory2)

At time $T_1$ when the "surprise" regime change occurs, the log of real balances jumps along with $\pi_t$ to satisfy
equation {eq}`eq:pformula2`.


### Jump in $p_{T_1}$ or jump in $m_{T_1}$?


If we insist that the money supply $m_{T_1}$ is locked at its value $m_{T_1}^1$ inherited from the past, then formula {eq}`eq:pformula2` implies  that the price level jumps downward  at time $T_1$, to coincide with the downward jump in 
$\pi_{T_1}$ 

An alternative assumption about the money supply level is that as part of the "inflation stabilization",
the government resets $m_{T_1}$ according to

$$
 m_{T_1}^2 - m_{T_1}^1 = \alpha (\pi^1 - \pi^2)
$$ (eq:eqnmoneyjump)

By letting money jump according to equation {eq}`eq:eqnmoneyjump` the monetary authority  prevents  the price level
from **falling** at the moment that the unanticipated stabilization arrives.

In various research papers about stabilizations of high inflations, the jump in the money supply described by equation {eq}`eq:eqnmoneyjump` has been called
"the velocity dividend" that a government reaps from implementin a regime change that sustains a permanently lower inflation rate.

### Another note to Jiacheng

I apologize that you'll have to change the code -- actually simplify it -- to do the revised experiments.

I recommend setting  $\mu_0 = .2$ and $\mu^* =0 $ to start our three experiments.

### End of another note to Jiacheng

<!-- #endregion -->

```python
class Cagan_REE:
    " Solve the rational expectation version of Cagan model in finite time. "
    
    def __init__(self, m0, α, T, μ_seq):
        self.m0, self.T, self.μ_seq, self.α = m0, T, μ_seq, α
        
        δ = 1/(1 + α)
        π_end = μ_seq[-1]    # compute terminal expected inflation
        
        self.δ, self.π_end = δ, π_end
        
    def solve(self):
        m0, T, π_end, μ_seq, α, δ = self.m0, self.T, self.π_end, self.μ_seq, self.α, self.δ
        
        A1 = np.eye(T+1, T+1) - δ * np.eye(T+1, T+1, k=1)
        A2 = np.eye(T+1, T+1) - np.eye(T+1, T+1, k=-1)
        
        b1 = (1-δ) * μ_seq + np.concatenate([np.zeros(T), [δ * π_end]])
        b2 = μ_seq + np.concatenate([[m0], np.zeros(T)])
        
        π_seq = np.linalg.inv(A1) @ b1
        m_seq = np.linalg.inv(A2) @ b2
        
        π_seq = np.append(π_seq, π_end)
        m_seq = np.append(m0, m_seq)
        
        p_seq = m_seq + α * π_seq

        return π_seq, m_seq, p_seq


def solve_and_plot(m0, α, T, μ_seq):
    
    mc = Cagan_REE(m0=m0, α=α, T=T, μ_seq=μ_seq)
    π_seq, m_seq, p_seq = mc.solve()
    T_seq = range(T+2)
    
    
    fig, ax = plt.subplots(2, 3, figsize=[10,5], dpi=200)
    ax[0,0].plot(T_seq[:-1], μ_seq)
    ax[0,1].plot(T_seq, π_seq)
    ax[0,2].plot(T_seq, m_seq - p_seq)
    ax[1,0].plot(T_seq, m_seq)
    ax[1,1].plot(T_seq, p_seq)
    
    ax[0,0].set_ylabel(r'$\mu$')
    ax[0,0].set_xlabel(r'$t$')
    ax[0,1].set_ylabel(r'$\pi$')
    ax[0,1].set_xlabel(r'$t$')
    ax[0,2].set_xlabel(r'$t$')
    ax[0,2].set_ylabel(r'$m - p}$')
    ax[1,0].set_ylabel(r'$m$')
    ax[1,0].set_xlabel(r'$t$')
    ax[1,1].set_ylabel(r'$p$')
    ax[1,1].set_xlabel(r'$t$')

    ax[1,2].set_axis_off()
    plt.tight_layout()
    plt.show()
    
    return π_seq, m_seq, p_seq
```

```python
# parameters
T = 80
T1 = 60
α = 30
m0 = 1

μ0 = 0.5
μ_star = 0
```

### Experiment 1

```python
μ_seq_1 = np.append(μ0*np.ones(T1), μ_star*np.ones(T+1-T1))

# solve and plot
π_seq_1, m_seq_1, p_seq_1 = solve_and_plot(m0=m0, α=α, T=T, μ_seq=μ_seq_1)
```

### Experiment 2

```python
# parameters
ϕ = 0.9
μ_seq_2 = np.array([ϕ**t * μ0 + (1-ϕ**t)*μ_star for t in range(T)])
μ_seq_2 = np.append(μ_seq_2, μ_star)


# solve and plot
π_seq_2, m_seq_2, p_seq_2 = solve_and_plot(m0=m0, α=α, T=T, μ_seq=μ_seq_2)
```

### Experiment 3

With a constant expected forward sequence $\mu_s = \bar \mu$ for $s\geq t$, $\pi_{t} =\bar{\mu}$. 

#### Regime 1: Insist on $m_{T_{1}}$.

$$
\begin{align*}
m_{T_{1}}&=m_{T_{1}-1}+\mu_{0}\\\pi_{T_{1}}&=\mu^{*}\\p_{T_{1}}&=m_{T_{1}}+\alpha\pi_{T_{1}}
\end{align*}
$$
Simply glue the sequences $t\leq T_1$ and $t > T_1$.

#### Regime 2: Reset $m_{T_{1}}$.

We want to reset $m_{T_{1}}$  so that $p_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\mu_{0}$, with $\pi_{T_{1}}=\mu^{*}$.

Then, 
$$ m_{T_{1}}=p_{T_{1}}-\alpha\pi_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\left(\mu_{0}-\mu^{*}\right) $$
Next, solve for the remaining $T-T_{1}$ periods with $\mu_{s}=\mu^{*},\forall s\geq T_{1}$ and the initial condition $m_{T_{1}}$ from above.

```python
μ_seq_3_path1 = μ0 * np.ones(T+1)
μ_seq_3_path2 = np.concatenate([μ0 * np.ones(T1), μ_star * np.ones(T+1-T1)])

# two paths
T_seq = range(T+2)

mc1 = Cagan_REE(m0=m0, α=α, T=T, μ_seq=μ_seq_3_path1)
mc2 = Cagan_REE(m0=m0, α=α, T=T, μ_seq=μ_seq_3_path2)

π_seq_3_path1, m_seq_3_path1, p_seq_3_path1 = mc1.solve()
π_seq_3_path2, m_seq_3_path2, p_seq_3_path2 = mc2.solve()

# glue π_seq, μ_seq
π_seq_3 = np.concatenate([π_seq_3_path1[:T1], π_seq_3_path2[T1:]])
μ_seq_3 = np.concatenate([μ_seq_3_path1[:T1], μ_seq_3_path1[T1:]])

# plot both paths
fig, ax = plt.subplots(2, 3, figsize=[10,5], dpi=200)
ax[0,0].plot(T_seq[:-1], μ_seq_3_path1)
ax[0,0].plot(T_seq[:-1], μ_seq_3_path2)
ax[0,1].plot(T_seq, π_seq_3_path1)
ax[0,1].plot(T_seq, π_seq_3_path2)
ax[0,2].plot(T_seq, m_seq_3_path1 - p_seq_3_path1)
ax[0,2].plot(T_seq, m_seq_3_path2 - p_seq_3_path2)
ax[1,0].plot(T_seq, m_seq_3_path1)
ax[1,0].plot(T_seq, m_seq_3_path2)
ax[1,1].plot(T_seq, p_seq_3_path1)
ax[1,1].plot(T_seq, p_seq_3_path2)

ax[0,0].set_ylabel(r'$\mu$')
ax[0,0].set_xlabel(r'$t$')
ax[0,1].set_ylabel(r'$\pi$')
ax[0,1].set_xlabel(r'$t$')
ax[0,2].set_xlabel(r'$t$')
ax[0,2].set_ylabel(r'$m - p}$')
ax[1,0].set_ylabel(r'$m$')
ax[1,0].set_xlabel(r'$t$')
ax[1,1].set_ylabel(r'$p$')
ax[1,1].set_xlabel(r'$t$')

ax[1,2].set_axis_off()
plt.tight_layout()
plt.show()
```

```python
# Insist on m_T1 - simply glue the sequences
m_seq_3_regime1 = np.concatenate([m_seq_3_path1[:T1+1], m_seq_3_path2[T1+1:]])
p_seq_3_regime1 = np.concatenate([p_seq_3_path1[:T1+1], p_seq_3_path2[T1+1:]])

# Reset m_T1
m_T1 = (m_seq_3_path1[T1-1] + μ0) + α*(μ0 - μ_star)

mc = Cagan_REE(m0=m_T1, α=α, T=T-1-T1, μ_seq=μ_seq_3_path2[T1+1:])
π_seq_3_remain2, m_seq_3_remain2, p_seq_3_remain2 = mc.solve()

m_seq_3_regime2 = np.concatenate([m_seq_3_path1[:T1+1], m_seq_3_remain2])
p_seq_3_regime2 = np.concatenate([p_seq_3_path1[:T1+1], p_seq_3_remain2])
```

```python
# plot both regimes
fig, ax = plt.subplots(2, 3, figsize=[10,5], dpi=200)
ax[0,0].plot(T_seq[:-1], μ_seq_3)
ax[0,1].plot(T_seq, π_seq_3)
ax[0,2].plot(T_seq, m_seq_3_regime1 - p_seq_3_regime1)
ax[0,2].plot(T_seq, m_seq_3_regime2 - p_seq_3_regime2)
ax[1,0].plot(T_seq, m_seq_3_regime1, label='Insist on $m_{T_1}$')
ax[1,0].plot(T_seq, m_seq_3_regime2, label='Reset $m_{T_1}$')
ax[1,1].plot(T_seq, p_seq_3_regime1, label='Insist on $m_{T_1}$')
ax[1,1].plot(T_seq, p_seq_3_regime2, label='Reset $m_{T_1}$')

ax[0,0].set_ylabel(r'$\mu$')
ax[0,0].set_xlabel(r'$t$')
ax[0,1].set_ylabel(r'$\pi$')
ax[0,1].set_xlabel(r'$t$')
ax[0,2].set_xlabel(r'$t$')
ax[0,2].set_ylabel(r'$m - p}$')
ax[1,0].set_ylabel(r'$m$')
ax[1,0].set_xlabel(r'$t$')
ax[1,1].set_ylabel(r'$p$')
ax[1,1].set_xlabel(r'$t$')
ax[1,2].set_axis_off()

for i,j in zip([1,1], [0,1]):
    ax[i,j].legend()

plt.tight_layout()
plt.show()
```
