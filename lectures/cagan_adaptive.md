# Adaptive expectations version of Cagan model

```python
import numpy as np
import matplotlib.pyplot as plt
```

We'll use linear algebra to do some experiments with  a "fiscal theory of the price level".

According to this model, when the government persistently spends more than it collects in taxes and prints money to finance the shortfall (called the "government deficit"), it puts upward pressure on the price level and generates
persistent inflation.

Our model is an "adaptive expectations"  version of a model that Philip Cagan  used to study the monetary dynamics of hyperinflations.  

It combines these components:

 * a demand function for real money balances that says asserts that the logarithm of the quantity of real balances demanded depends inversely on the public's expected rate of inflation
 
 * an **adaptive expectations** model that describes how the public's anticipated rate of inflation responds to past values of actual inflation
 
 * an equilibrium condition that equates the demand for money to the supply
 
 * an exogenous sequence of rates of growth of the money supply

Our model stays quite close to Cagan's original specification.  

To facilitate using  linear matrix algebra as our principal mathematical tool, we'll use a finite horizon version of
the model.

Let 

 * $ m_t $ be the log of the supply of  nominal money balances;
 * $\mu_t = m_{t+1} - m_t $ be the net rate of growth of  nominal balances;
 * $p_t $ be the log of the price level;
 * $\pi_t = p_{t+1} - p_t $ be the net rate of inflation  between $t$ and $ t+1$;
 * $\pi_t^*$  be the public's expected rate of inflation between  $t$ and $t+1$;
 * $T$ the horizon -- i.e., the last period for which the model will determine $p_t$
 * $\pi_0^*$ public's initial expected rate of inflation between time $0$ and time $1$.
  
  
The demand for real balances $\exp\left(\frac{m_t^d}{p_t}\right)$ is governed by the following  version of the Cagan demand function
  
$$  
m_t^d - p_t = -\alpha \pi_t^* \: , \: \alpha > 0 ; \quad t = 0, 1, \ldots, T .
$$ (eq:caganadmd)


This equation  asserts that the demand for real balances
is inversely related to the public's expected rate of inflation.

Equating the logarithm $m_t^d$ of the demand for money  to the logarithm  $m_t$ of the supply of money in equation {eq}`eq:caganadmd` and solving for the logarithm $p_t$
of the price level gives

$$
p_t = m_t + \alpha \pi_t^*
$$ (eq:eqfiscth1)

Taking the difference between equation {eq}`eq:eqfiscth1` at time $t+1$ and at time 
$t$ gives



$$
\pi_t = \mu_t + \alpha \pi_{t+1}^* - \alpha \pi_t^*
$$ (eq:eqpipi)

We assume that the expected rate of inflation $\pi_t^*$ is governed
by the Friedman-Cagan adaptive expectations scheme

$$
\pi_{t+1}^* = \lambda \pi_t^* + (1 -\lambda) \pi_t 
$$ (eq:adaptexpn)

As exogenous inputs into the model, we take initial conditions $m_0, \pi_0^*$
and a money growth sequence $\vec \mu = \{\mu_t\}_{t=0}^T$.  

As endogenous outputs of our model we want to find sequences $\vec \pi = \{\pi_t\}_{t=0}^T, \vec p = \{p_t\}_{t=0}^T$ as functions of the endogenous inputs.

We'll do some mental experiments by studying how the model outputs vary as we vary
the model inputs.

## Representing key equations with linear algebra

We begin by writing the equation {eq}`eq:adaptexpn`  adaptive expectations model for $\pi_t^*$ for $t=0, \ldots, T$ as



$$
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 & 0 \cr
-\lambda & 1 & 0 & \cdots & 0 & 0 \cr
0 & - \lambda  & 1  & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda & 1
\end{bmatrix}
\begin{bmatrix} \pi_0^* \cr
  \pi_1^* \cr
  \pi_2^* \cr
  \vdots \cr
  \pi_{T+1}^* 
  \end{bmatrix} =
  (1-\lambda) \begin{bmatrix} 
  0 & 0 & 0 & \cdots & 0  \cr
  1 & 0 & 0 & \cdots & 0   \cr
   0 & 1 & 0 & \cdots & 0  \cr
    \vdots &\vdots & \vdots & \cdots & \vdots  \cr
     0 & 0 & 0 & \cdots & 1  \end{bmatrix}
     \begin{bmatrix}\pi_0 \cr \pi_1 \cr \pi_2 \cr \vdots \cr \pi_T
  \end{bmatrix} +
  \begin{bmatrix} \pi_0^* \cr 0 \cr 0 \cr \vdots \cr 0 \end{bmatrix}
$$

Write this equation as

$$
A \vec \pi^* = (1-\lambda) B \vec \pi + \vec \pi_0^*
$$ (eq:eq1)

where the $(T+2) \times (T+2) $matrix $A$, the $(T+2)\times (T+1)$ matrix $B$, and the vectors $\vec \pi^* , \vec \pi_0, \pi_0^*$
are defined implicitly by aligning these two equations.

Next we write the key equation {eq}`eq:eqpipi` in matrix notation as

$$ \begin{bmatrix}
\pi_0 \cr \pi_1 \cr \pi_1 \cr \vdots \cr \pi_T \end{bmatrix}
= \begin{bmatrix}
\mu_0 \cr \mu_1 \cr \mu_2 \cr  \vdots \cr \mu_T \end{bmatrix}
+ \begin{bmatrix} - \alpha &  \alpha & 0 & \cdots & 0 & 0 \cr
0 & -\alpha & \alpha & \cdots & 0 & 0 \cr
0 & 0 & -\alpha & \cdots & 0 & 0 \cr
\vdots & \vdots & \vdots & \cdots & \alpha & 0 \cr
0 & 0 & 0 & \cdots & -\alpha  & \alpha 
\end{bmatrix}
\begin{bmatrix} \pi_0^* \cr
  \pi_1^* \cr
  \pi_2^* \cr
  \vdots \cr
  \pi_{T+1}^* 
  \end{bmatrix}
$$

Represent the previous equation system in terms of vectors and matrices as

$$
\vec \pi = \vec \mu + C \vec \pi^*
$$ (eq:eq2)

where the $(T+1) \times (T+2)$ matrix $C$ is defined implicitly to align this equation with the preceding
equation system.



## Harvesting payoffs from our matrix formulation


We now have all of the ingredients we need to solve for $\vec \pi$ as
a function of $\vec \mu, \pi_0, \pi_0^*$.  

Combine equations {eq}`eq:eq1`and {eq}`eq:eq2`  to get

$$
\begin{align*}
A \vec \pi^* & = (1-\lambda) B \vec \pi + \vec \pi_0^* \cr
 & = (1-\lambda) B \left[ \vec \mu + C \vec \pi^* \right] + \vec \pi_0^*
\end{align*}
$$

which implies that

$$
\left[ A - (1-\lambda) B C \right] \vec \pi^* = (1-\lambda) B \vec \mu+ \vec \pi_0^*
$$

Multiplying both sides of the above equation by the inverse of the matrix on the left side gives

$$
\vec \pi^* = \left[ A - (1-\lambda) B C \right]^{-1} \left[ (1-\lambda) B \vec \mu+ \vec \pi_0^* \right]
$$ (eq:eq4)

Having solved equation {eq}`eq:eq4` for $\vec \pi^*$, we can use  equation {eq}`eq:eq2`  to solve for $\vec \pi$:

$$
\vec \pi = \vec \mu + C \vec \pi^*
$$


We have thus solved for two of the key endogenous time series determined by our model, namely, the sequence $\vec \pi^*$
of expected inflation rates and the sequence $\vec \pi$ of actual inflation rates.  

Knowing these, we can then quickly calculate the associated sequence $\vec p$  of the logarithm of the  price level
from equation {eq}`eq:eqfiscth1`. 

Let's fill in the details for this step.
<!-- #endregion -->

Since we now know $\vec \mu$  it is easy to compute $\vec m$.

Thus, notice that we can represent the equations 

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
$$ (eq:adeq101)

Multiplying both sides of equation {eq}`eq:adeq101`  with the inverse of the matrix on the left will give 

$$
m_t = m_0 + \sum_{s=0}^{t-1} \mu_s, \quad t =1, \ldots, T+1
$$ (eq:mcum_ad)

Equation {eq}`eq:mcum_ad` shows that the log of the money supply at $t$ equals the log $m_0$ of the initial money supply 
plus accumulation of rates of money growth between times $0$ and $t$.

We can then compute $p_t$ for each $t$ from equation {eq}`eq:eqfiscth1`.

We can write a compact formula for $\vec p $ as

$$ 
\vec p = \vec m + \alpha \hat \pi^*
$$

where 

$$
\hat \pi^* = \begin{bmatrix} \pi_0^* \cr
  \pi_1^* \cr
  \pi_2^* \cr
  \vdots \cr
  \pi_{T}^* 
  \end{bmatrix},
 $$

which is just $\vec \pi^*$ with the last element dropped.
 


## Expectational Error Gap 

Our computations will verify that 

$$
\hat \pi^* \neq \vec  \pi,
$$

so that in general

$$ 
\pi_t^* \neq \pi_t, \quad t = 0, 1, \ldots , T
$$ (eq:notre)

This outcome is typical in models in which adaptive expectations hypothesis like equation {eq}`eq:adaptexpn` appear as a
component.  

In a companion lecture, we'll discuss a version of the model that replaces hypothesis {eq}`eq:adaptexpn` with
a "perfect foresight" or "rational expectations" hypothesis.

```python
class Cagan_Adaptive:
    " Solve the Cagan model in finite time. "
    
    def __init__(self, α, m0, Eπ0, μ_seq, T, λ):
        self.α, self.m0, self.Eπ0, self.μ_seq, self.T, self.λ = α, m0, Eπ0, μ_seq, T, λ
    
    def solve(self):
        α, m0, Eπ0, μ_seq, T, λ = self.α, self.m0, self.Eπ0, self.μ_seq, self.T, self.λ
        
        A = np.eye(T+2, T+2) - λ*np.eye(T+2, T+2, k=-1)
        B = np.eye(T+2, T+1, k=-1)
        C = -α*np.eye(T+1, T+2) + α*np.eye(T+1, T+2, k=1)
        Eπ0_seq = np.append(Eπ0, np.zeros(T+1))
        
        # Eπ_seq is of length T+2
        Eπ_seq = np.linalg.inv(A - (1-λ)*B @ C) @ ((1-λ) * B @ μ_seq + Eπ0_seq)
        
        # π_seq is of length T+1
        π_seq = μ_seq + C @ Eπ_seq
        
        D = np.eye(T+1, T+1) - np.eye(T+1, T+1, k=-1)
        m0_seq = np.append(m0, np.zeros(T))
        
        # m_seq is of length T+2
        m_seq = np.linalg.inv(D) @ (μ_seq + m0_seq)
        m_seq = np.append(m0, m_seq)
        
        # p_seq is of length T+2
        p_seq = m_seq + α * Eπ_seq
        
        return π_seq, Eπ_seq, m_seq, p_seq
    
        
def solve_and_plot(α, m0, Eπ0, μ_seq, T, λ):
    
    mc = Cagan_Adaptive(α=α, m0=m0, Eπ0=Eπ0, μ_seq=μ_seq, T=T, λ=λ)
    π_seq, Eπ_seq, m_seq, p_seq = mc.solve()
    T_seq = range(T+2)
    
    fig, ax = plt.subplots(2, 3, figsize=[10,5], dpi=200)
    ax[0,0].plot(T_seq[:-1], μ_seq)
    ax[0,1].plot(T_seq[:-1], π_seq, label=r'$\pi_t$')
    ax[0,1].plot(T_seq, Eπ_seq, label=r'$\pi^{*}_{t}$')
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

    ax[0,1].legend()
    ax[1,2].set_axis_off()
    plt.tight_layout()
    plt.show()
    
    return π_seq, Eπ_seq, m_seq, p_seq
```

## Experiment 1

Jiacheng: 
$$
\begin{align*}
\pi_{t}&=\mu_{t}+\alpha\pi_{t+1}^{*}-\alpha\pi_{t}^{*}\\\pi_{t+1}^{*}&=\lambda\pi_{t}^{*}+(1-\lambda)\pi_{t}\\\left[1-\alpha(1-\lambda)\right]\pi_{t}&=\mu_{t}+\left(\alpha\lambda-\alpha\right)\pi_{t}^{*}\\\pi_{t}&=\frac{\mu_{t}}{1-\alpha(1-\lambda)}-\frac{\alpha\left(1-\lambda\right)}{1-\alpha(1-\lambda)}\pi_{t}^{*}
\end{align*}
$$

If $1-\alpha(1-\lambda)<0$, expectation would explode.

```python
# parameters
T = 80
T1 = 60
α = 4
λ = 0.9
m0 = 1

μ0 = 0.2
μ_star = 0
```

```python
print(1 - α*(1-λ))
```

```python
μ_seq_1 = np.append(μ0*np.ones(T1), μ_star*np.ones(T+1-T1))

# solve and plot
π_seq_1, Eπ_seq_1, m_seq_1, p_seq_1 = solve_and_plot(α=α, m0=m0, Eπ0=μ0, μ_seq=μ_seq_1, T=T, λ=λ)
```

### Experiment 2

```python
# parameters
ϕ = 0.9
μ_seq_2 = np.array([ϕ**t * μ0 + (1-ϕ**t)*μ_star for t in range(T)])
μ_seq_2 = np.append(μ_seq_2, μ_star)


# solve and plot
π_seq_2, Eπ_seq_2, m_seq_2, p_seq_2 = solve_and_plot(α=α, m0=m0, Eπ0=μ0, μ_seq=μ_seq_2, T=T, λ=λ)
```
