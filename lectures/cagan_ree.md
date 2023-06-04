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

# A Fiscal Theory of the Price Level

## Introduction

As usual, we'll start by importing some Python modules.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

+++ {"user_expressions": []}

<!-- #region -->
We'll use linear algebra first to explain and then do  some experiments with  a "fiscal theory of the price level".


According to this model, when the government persistently spends more than it collects in taxes and prints money to finance the shortfall (the "shortfall is called the "government deficit"), it puts upward pressure on the price level and generates
persistent inflation.

The "fiscal theory of the price level" asserts that 

  * to **start** a persistent inflation the government simply persistently runs a money-financed government deficit
  
  * to **stop** a persistent inflation the government simply  stops persistently running  a money-financed government deficit

Our model is a "rational expectations" (or "perfect foresight") version of a model that Philip Cagan  used to study the monetary dynamics of hyperinflations.  

While Cagan didn't use that  "rational expectations" version of the model, Thomas Sargent did when he studied the Ends of Four Big Inflations.

Some of our quantitative experiments with the model are designed to illustrate how the fiscal theory explains the abrupt end of those big inflations.

In those experiments, we'll encounter an instance of a ``velocity dividend'' that has sometimes accompanied successful inflation stabilization programs.
principles 

To facilitate using  linear matrix algebra as our main mathematical tool, we'll use a finite horizon version of the model.

As in several other lectures, the only linear algebra that we'll be  using
are matrix multplication and matrix inversion.


## Structure of the Model


The model consists of

 * a function that expresses the demand for real balances of government printed money as an inverse function of the public's expected rate of inflation
 
 * an exogenous sequence of rates of growth of the money supply.  The money supply grows because the government is printing it to finance some of its expenditures
 
 * an equilibrium condition that equates the demand for money to the supply
 
 * a "perfect foresight"  assumption that the public's expected rate of inflation equals the actual rate of inflation.
 
To represent the model formally, let 

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

Setting $\delta =\frac{\alpha}{1+\alpha}$ let's us represent the preceding equation as

\begin{equation}
\pi_t = \delta \pi_{t+1} + (1-\delta) \mu_t , \quad t =0, 1, \ldots, T
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

### Some quantitative experiments

In the experiments below, we'll use formula {eq}`eq:piterm` as our terminal condition for expected inflation.

In devising these experiments, we'll  make assumptions about $\{\mu_t\}$ that are consistent with formula
{eq}`eq:piterm`.




We  describe several such experiments.

In all of them, 

$$ 
\mu_t = \mu^* , \quad t \geq T_1
$$

so that, in terms of our notation and formula for $\theta_{T+1}^*$ above, $\tilde \gamma = 1$. 

#### Experiment 1: foreseen sudden stabilization

In this experiment, we'll study how, when $\alpha >0$, a foreseen inflation stabilization has effects on inflation that proceed it.

We'll study a situation in which the rate of growth of the money supply is $\mu_0$
from $t=0$ to $t= T_1$ and then permanently falls to $\mu^*$ at $t=T_1$.

Thus, let $T_1 \in (0, T)$. 

So where $\mu_0 > \mu^*$, we assume that

$$
\mu_{t+1} = \begin{cases}
    \mu_0  , & t = 0, \ldots, T_1 -1 \\
     \mu^* , & t \geq T_1
     \end{cases}
$$


#### Experiment 2: an unforeseen sudden stabilization

This experiment deviates a little bit from a pure version our "perfect foresight"
assumption by assuming that a sudden permanent reduction in $\mu_t$ like that
analyzed in experiment 1 is completely unanticipated.  

Such a  completely unanticipated shock is popularly known as an "MIT shock".

The mental experiment involves switching at at time $T_1$ from an initial "continuation path" for $\{\mu_t, \pi_t\} $ to another path that involves a permanently lower inflation frate.   

**Initial Path:** $\mu_t = \mu_0$ for all $t \geq 0$. So this path is for $\{\mu_t\}_{t=0}^\infty$; the associated 
path for $\pi_t$ has $\pi_t = \mu_0$. 

**Revised Continuation Path** Where $ \mu_0 > \mu^*$, we construct a continuation path  $\{\mu_s\}_{s=T_1}^\infty$
by setting $\mu_s = \mu^*$ for all $s \geq T_1$.  The perfect foresight continuation path for 
$\pi$ is  $\pi_s = \mu^*$ 

To capture a "completely unanticipated permanent  shock to the $\{\mu\}$ process at time $T_1$, we simply  glue the $\mu_t, \pi_t$
that emerges under path 2 for $t \geq T_1$ to the $\mu_t, \pi_t$ path that had emerged under path 1 for $ t=0, \ldots,
T_1 -1$.

We can do the MIT shock calculations entirely by hand. 

Thus, for path 1, $\pi_t = \mu_0 $ for all $t \in  [0, T_1-1]$, while for path 2,
$\mu_s = \mu^*$ for all $s \geq T_1$.  

### The log price level

We can use equations {eq}`eq:caganmd` and {eq}`eq:ree`
to discover that the log of the price level satisfies

$$
p_t = m_t + \alpha \pi_t
$$ (eq:pformula2)

or, by using equation  {eq}`eq:fisctheory1`,

$$ 
p_t = m_t + \alpha \left[ (1-\delta) \sum_{s=t}^T \delta^{s-t} \mu_s +  \delta^{T+1-t} \pi_{T+1}^*  \right] 
$$ (eq:pfiscaltheory2)

At time $T_1$ when the "surprise" regime change occurs,  to satisfy
equation {eq}`eq:pformula2`, the log of real balances jumps 
**upward* as  $\pi_t$ jumps **downward**.

But in order for $m_t - p_t$ to jump, which variable jumps, $m_{T_1}$ or $p_{T_1}$?


###  What jumps?

What jumps at $T_1$?

Is it $p_{T_1}$ or  $m_{T_1}$?


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

#### Technical Details about whether $p$ or $m$ jumps at $T_1$

We have noted that  with a constant expected forward sequence $\mu_s = \bar \mu$ for $s\geq t$, $\pi_{t} =\bar{\mu}$.

A consequence is that at $T_1$, either $m$ or $p$ must "jump" at $T_1$.

We'll study both cases. 

#### $m_{T_{1}}$ does not jump.

$$
\begin{align*}
m_{T_{1}}&=m_{T_{1}-1}+\mu_{0}\\\pi_{T_{1}}&=\mu^{*}\\p_{T_{1}}&=m_{T_{1}}+\alpha\pi_{T_{1}}
\end{align*}
$$
Simply glue the sequences $t\leq T_1$ and $t > T_1$.

####  $m_{T_{1}}$ jumps.

We reset $m_{T_{1}}$  so that $p_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\mu_{0}$, with $\pi_{T_{1}}=\mu^{*}$.

Then, 
$$ m_{T_{1}}=p_{T_{1}}-\alpha\pi_{T_{1}}=\left(m_{T_{1}-1}+\mu_{0}\right)+\alpha\left(\mu_{0}-\mu^{*}\right) $$
We then compute for the remaining $T-T_{1}$ periods with $\mu_{s}=\mu^{*},\forall s\geq T_{1}$ and the initial condition $m_{T_{1}}$ from above.


#### Experiment 3

**Foreseen gradual stabilization**

Instead of a foreseen sudden stabilization of the type studied with experiment 1,
it is also interesting to study the consequences of a foreseen gradual stabilization.

Thus, suppose that $\phi \in (0,1)$, that  $\mu_0 > \mu^*$,  and that for $t = 0, \ldots, T-1$

$$
\mu_t = \phi^t \mu_0 + (1 - \phi^t) \mu^* .
$$ 

#### Python Code

Let's prepare a Python class to perform our experiments by implementing our formulas using linear algebra
<!-- #endregion -->

```{code-cell} ipython3
class Cagan_REE:
    " Solve the rational expectation version of Cagan model in finite time. "
    
    def __init__(self, m0, α, T, μ_seq):
        self.m0, self.T, self.μ_seq, self.α = m0, T, μ_seq, α
        
        δ = α/(1 + α)
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
    
    
    fig, ax = plt.subplots(2, 3, figsize=[10, 5], dpi=200)
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
    ax[0,2].set_ylabel(r'$m - p$')
    ax[1,0].set_ylabel(r'$m$')
    ax[1,0].set_xlabel(r'$t$')
    ax[1,1].set_ylabel(r'$p$')
    ax[1,1].set_xlabel(r'$t$')

    ax[1,2].set_axis_off()
    plt.tight_layout()
    plt.show()
    
    return π_seq, m_seq, p_seq
```

```{code-cell} ipython3
# parameters
T = 80
T1 = 60
α = 5
m0 = 1

μ0 = 0.5
μ_star = 0
```

+++ {"user_expressions": []}

### Experiment 1

We'll start by executing a version of our "experiment 1" in which the government  implements a **foreseen** sudden permanent reduction in the rate of money creation at time $T_1$.  

The following code  performs the experiment and plots outcomes.

```{code-cell} ipython3
μ_seq_1 = np.append(μ0*np.ones(T1+1), μ_star*np.ones(T-T1))

# solve and plot
π_seq_1, m_seq_1, p_seq_1 = solve_and_plot(m0=m0, α=α, T=T, μ_seq=μ_seq_1)
```

+++ {"user_expressions": []}

The  plot of the money growth rate $\mu_t$ in the top level panel portrays
a sudden reduction from $.5$ to $0$ at time $T_1 = 60$.  

This brings about a gradual reduction of the inflation rate $\pi_t$ that precedes the
money supply growth rate reduction at time $T_1$.

Notice how the inflation rate declines smoothly (i.e., continuously) to $0$ at $T_1$ -- 
unlike the money growth rate, it does not suddenly "jump" downward at $T_1$.

This is because the reduction in $\mu$ at $T_1$ has been foreseen from the start.  

While the log money supply portrayed in the bottom panel has a kink at $T_1$, the log  price level does not -- it is "smooth" -- once again a consequence of the fact that the
reduction in $\mu$ has been foreseen.

<!-- #region -->
#### Experiment 2


We now move on to experiment 2, our "MIT shock", completely unforeseen 
sudden stabilization.

We set this up so that the $\{\mu_t\}$ sequences that describe the sudden stabilization
are identical to those for experiment 1, the foreseen suddent stabilization.

The following code does the calculations and plots outcomes.
<!-- #endregion -->

```{code-cell} ipython3
# path 1
μ_seq_3_path1 = μ0 * np.ones(T+1)

mc1 = Cagan_REE(m0=m0, α=α, T=T, μ_seq=μ_seq_3_path1)
π_seq_3_path1, m_seq_3_path1, p_seq_3_path1 = mc1.solve()

# continuation path
μ_seq_3_cont = μ_star * np.ones(T-T1)

mc2 = Cagan_REE(m0=m_seq_3_path1[T1+1], α=α, T=T-1-T1, μ_seq=μ_seq_3_cont)
π_seq_3_cont, m_seq_3_cont1, p_seq_3_cont1 = mc2.solve()


# regime 1 - simply glue π_seq, μ_seq
μ_seq_3 = np.concatenate([μ_seq_3_path1[:T1+1], μ_seq_3_cont])
π_seq_3 = np.concatenate([π_seq_3_path1[:T1+1], π_seq_3_cont])
m_seq_3_regime1 = np.concatenate([m_seq_3_path1[:T1+1], m_seq_3_cont1])
p_seq_3_regime1 = np.concatenate([p_seq_3_path1[:T1+1], p_seq_3_cont1])

# regime 2 - reset m_T1
m_T1 = (m_seq_3_path1[T1] + μ0) + α*(μ0 - μ_star)

mc = Cagan_REE(m0=m_T1, α=α, T=T-1-T1, μ_seq=μ_seq_3_cont)
π_seq_3_cont2, m_seq_3_cont2, p_seq_3_cont2 = mc.solve()

m_seq_3_regime2 = np.concatenate([m_seq_3_path1[:T1+1], m_seq_3_cont2])
p_seq_3_regime2 = np.concatenate([p_seq_3_path1[:T1+1], p_seq_3_cont2])

T_seq = range(T+2)

# plot both regimes
fig, ax = plt.subplots(2, 3, figsize=[10,5], dpi=200)
 
ax[0,0].plot(T_seq[:-1], μ_seq_3)
ax[0,1].plot(T_seq, π_seq_3)
ax[0,2].plot(T_seq, m_seq_3_regime1 - p_seq_3_regime1)
ax[1,0].plot(T_seq, m_seq_3_regime1, label='Smooth $m_{T_1}$')
ax[1,0].plot(T_seq, m_seq_3_regime2, label='Jumpy $m_{T_1}$')
ax[1,1].plot(T_seq, p_seq_3_regime1, label='Smooth $m_{T_1}$')
ax[1,1].plot(T_seq, p_seq_3_regime2, label='Jumpy $m_{T_1}$')

ax[0,0].set_ylabel(r'$\mu$')
ax[0,0].set_xlabel(r'$t$')
ax[0,1].set_ylabel(r'$\pi$')
ax[0,1].set_xlabel(r'$t$')
ax[0,2].set_xlabel(r'$t$')
ax[0,2].set_ylabel(r'$m - p$')
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

+++ {"user_expressions": []}

We invite you to compare these graphs with corresponding ones for the foreseen stabilization analyzed in experiment 1 above.  

Note how the inflation graph in the top middle panel is now identical to the 
money growth graph in the top left panel, and how now the log of real balances portrayed in the top right panel jumps upward at time $T_1$.

The bottom panels plot $m$ and $p$ under two possible ways that $m_{T_1}$ might adjust
as required by the upward jump in $m - p$ at $T_1$.  

  * the orange line lets $m_{T_1}$ jump upward in order to make sure that the log price level $p_{T_1}$ does not fall.
  
  * the blue line lets $p_{T_1}$ fall while stopping the money supply from jumping.
  
Here is a way to interpret what the government is doing when the orange line policy is in place.

The government  prints money to finance expenditure with  the "velocity dividend" that it reaps from the increased demand for real balances brought about by the permanent decrease in the rate of growth of the money supply.


The next code generates a multi-panel graph that includes outcomes of both experiments 1 and 2.

That allows us to assess how important it is to understand whether the sudden permanent drop in $\mu_t$ at $t=T_1$ is fully unanticipated, as in experiment 1, or completely
unanticipated, as in experiment 2.

```{code-cell} ipython3
# compare foreseen vs unforeseen shock
fig, ax = plt.subplots(2, 3, figsize=[12,6], dpi=200)
ax[0,0].plot(T_seq[:-1], μ_seq_3)

ax[0,1].plot(T_seq, π_seq_3, label='Unforeseen')
ax[0,1].plot(T_seq, π_seq_1, label='Foreseen', color='tab:green')

ax[0,2].plot(T_seq, m_seq_3_regime1 - p_seq_3_regime1, label='Unforeseen')
ax[0,2].plot(T_seq, m_seq_1 - p_seq_1, label='Foreseen', color='tab:green')

ax[1,0].plot(T_seq, m_seq_3_regime1, label=r'Unforseen (Insist on $m_{T_1}$)')
ax[1,0].plot(T_seq, m_seq_3_regime2, label=r'Unforseen (Reset $m_{T_1}$)')
ax[1,0].plot(T_seq, m_seq_1, label='Foreseen shock')

ax[1,1].plot(T_seq, p_seq_3_regime1, label=r'Unforseen (Insist on $m_{T_1}$)')
ax[1,1].plot(T_seq, p_seq_3_regime2, label=r'Unforseen (Reset $m_{T_1}$)')
ax[1,1].plot(T_seq, p_seq_1, label='Foreseen')

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

for i,j in zip([0,0,1,1], [1,2,0,1]):
    ax[i,j].legend()

plt.tight_layout()
plt.show()
```

+++ {"user_expressions": []}

### Experiment 3

Next we perform an experiment in which there is a perfectly foreseen **gradual** decrease in the rate of growth of the money supply.

The following  code does the calculations and plots the results.

```{code-cell} ipython3
# parameters
ϕ = 0.9
μ_seq_2 = np.array([ϕ**t * μ0 + (1-ϕ**t)*μ_star for t in range(T)])
μ_seq_2 = np.append(μ_seq_2, μ_star)


# solve and plot
π_seq_2, m_seq_2, p_seq_2 = solve_and_plot(m0=m0, α=α, T=T, μ_seq=μ_seq_2)
```

```{code-cell} ipython3
# compare foreseen vs unforeseen shock
fig, ax = plt.subplots(2, 3, figsize=[12,6], dpi=200)
ax[0,0].plot(T_seq[:-1], μ_seq_3)

ax[0,1].plot(T_seq, π_seq_3, label='Unforeseen')
ax[0,1].plot(T_seq, π_seq_1, label='Foreseen', color='tab:green')

ax[0,2].plot(T_seq, m_seq_3_regime1 - p_seq_3_regime1, label='Unforeseen')
ax[0,2].plot(T_seq, m_seq_1 - p_seq_1, label='Foreseen', color='tab:green')

ax[1,0].plot(T_seq, m_seq_3_regime1, label=r'Unforseen (Insist on $m_{T_1}$)')
ax[1,0].plot(T_seq, m_seq_3_regime2, label=r'Unforseen (Reset $m_{T_1}$)')
ax[1,0].plot(T_seq, m_seq_1, label='Foreseen shock')

ax[1,1].plot(T_seq, p_seq_3_regime1, label=r'Unforseen (Insist on $m_{T_1}$)')
ax[1,1].plot(T_seq, p_seq_3_regime2, label=r'Unforseen (Reset $m_{T_1}$)')
ax[1,1].plot(T_seq, p_seq_1, label='Foreseen')

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

for i,j in zip([0,0,1,1], [1,2,0,1]):
    ax[i,j].legend()

plt.tight_layout()
plt.show()
```
