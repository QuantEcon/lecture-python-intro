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

# Some dynamic models with matrices

In this notebook, we'll present  some useful models of economic dynamics using only linear algebra -- matrix multiplication and matrix inversion.

**Present value formulas** are at the core of the models.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

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



## Difference equations with linear algebra

As a warmup, we'll describe a useful way of representing and "solving" linear difference equations. 

To generate some $y$ vectors, we'll just write down a linear difference equation
with appropriate initial conditions and then   use linear algebra to solve it.

### First-order difference equation

We'll start with a first-order linear difference equation for $\{y_t\}_{t=0}^T$:

$$
y_{t} = \lambda y_{t-1}, \quad t = 1, 2, \ldots, T
$$

where  $y_0$ is a given  initial condition.


We can cast this set of $T$ equations as a single  matrix equation

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
-\lambda & 1 & 0 & \cdots & 0 & 0 \cr
0 & -\lambda & 1 & \cdots & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda & 1 
\end{bmatrix} 
\begin{bmatrix}
y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T 
\end{bmatrix}
= 
\begin{bmatrix} 
\lambda y_0 \cr 0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$




Multiplying both sides by  inverse of the matrix on the left provides the solution

$$
\begin{bmatrix} 
y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T 
\end{bmatrix} 
= 
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
\lambda & 1 & 0 & \cdots & 0 & 0 \cr
\lambda^2 & \lambda & 1 & \cdots & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
\lambda^{T-1} & \lambda^{T-2} & \lambda^{T-3} & \cdots & -\lambda & 1 
\end{bmatrix}
\begin{bmatrix} 
\lambda y_0 \cr 0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$


### Second order difference equation


$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-\lambda_1 & 1 & 0 & \cdots & 0 & 0 & 0 \cr
-\lambda_2 & -\lambda_2 & 1 & \cdots & 0 & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & \lambda_2 & -\lambda_1 & 1 
\end{bmatrix} 
\begin{bmatrix} 
y_1 \cr y_2 \cr y_3 \cr \vdots \cr y_T 
\end{bmatrix}
= 
\begin{bmatrix} 
\lambda_1 y_0 + \lambda_2 y_{-1} \cr \lambda_2 y_0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$

Multiplying both sides by  inverse of the matrix on the left again provides the solution.

### Extensions

As an exercise, we ask you to represent and solve a **third order linear difference equation**.
How many initial conditions must you specify?



## Friedman-Hall consumption-smoothing model


A key object is what Milton Friedman called "non-human" or "non-financial" wealth at time $0$:


$$
h_0 \equiv \sum_{t=0}^T R^{-t} y_t = \begin{bmatrix} 1 & R^{-1} & \cdots & R^{-T} \end{bmatrix}
\begin{bmatrix} y_0 \cr y_1  \cr \vdots \cr y_T \end{bmatrix}
$$

By iterating on equation (1) and imposing the terminal condition 

$$
a_{T+1} = 0,
$$

it is possible to convert a sequence of budget constraints into the single intertemporal constraint

$$
\sum_{t=0}^T R^{-t} c_t = a_0 + h_0,
$$

which says that the present value of the consumption stream equals the sum of finanical and non-financial wealth.

Robert Hall (1978) showed that when $\beta R = 1$, a condition Milton Friedman had assumed,
it is "optimal" for a consumer to **smooth consumption** by setting 

$$ 
c_t = c_0 \quad t =0, 1, \ldots, T
$$

In this case, we can use the intertemporal budget constraint to write 

$$
c_0 = \left(\sum_{t=0}^T R^{-t}\right)^{-1} (a_0 + h_0)
$$

This is the consumption-smoothing model in a nutshell.

We'll put the model through some paces with Python code below.


## Permanent income model of consumption 

As promised, we'll provide step by step instructions on how to use linear algebra, readily implemented
in Python, to solve the consumption smoothing model.

**Note to programmer teammate:**

In the calculations below, please we'll  set default values of  $R > 1$, e.g., $R = 1.05$, and $\beta = R^{-1}$.

### Step 1

For some $T+1 \times 1$ $y$ vector, use matrix algebra to compute 

$$
\sum_{t=0}^T R^{-t} y_t = \begin{bmatrix} 1 & R^{-1} & \cdots & R^{-T} \end{bmatrix}
\begin{bmatrix} y_0 \cr y_1  \cr \vdots \cr y_T \end{bmatrix}
$$

### Step 2

Compute

$$
c_0 = \left( \frac{1 - R^{-1}}{1 - R^{-(T+1)}} \right) (a_0 + \sum_{t=0}^T R^t y_t )
$$

**Jiacheng:** The same for $R^t$ here.

### Step 3

Formulate system

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-R & 1 & 0 & \cdots & 0 & 0 & 0 \cr
0 & -R & 1 & \cdots & 0 & 0 & 0 \cr
\vdots  &\vdots & \vdots & \cdots & \vdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -R & 1 & 0 \cr
0 & 0 & 0 & \cdots & 0 & -R & 1
\end{bmatrix} 
\begin{bmatrix} a_1 \cr a_2 \cr a_3 \cr \vdots \cr a_T \cr a_{T+1} 
\end{bmatrix}
= R 
\begin{bmatrix} y_0 + a_0 - c_0 \cr y_1 - c_0 \cr y_2 - c_0 \cr \vdots\cr y_{T-1} - c_0 \cr y_T - c_0
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


### Feasible consumption variations

To explore what types of consumption paths are welfare-improving, we shall create an **admissible consumption path variation sequence** $\{v_t\}_{t=0}^T$
that satisfies

$$
\sum_{t=0}^T R^{-t} v_t = 0
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

We can even use the Python numpy grad command to compute derivatives of welfare with respect to our two parameters.  

We are teaching the key idea beneath the **calculus of variations**.

```{code-cell} ipython3
class Consumption_smoothing:
    "A class of the Permanent Income model of consumption"
    
    def __init__(self, R, y_seq, a0, g1, g2, T):
        self.a0, self.y_seq, self.R, self.β = a0, y_seq, R, 1/R    # set β = 1/R
        self.g1, self.g2 = g1, g2       # welfare parameter
        self.T = T
        
        self.β_seq = np.array([self.β**i for i in range(T+1)])
        
    def compute_optimal(self, verbose=1):
        R, y_seq, a0, T = self.R, self.y_seq, self.a0, self.T
        
        # non-financial wealth
        h0 = self.β_seq @ y_seq     # since β = 1/R
        
        # c0
        c0 = (1 - 1/R) / (1 - (1/R)**(T+1)) * (a0 + h0)
        c_seq = c0*np.ones(T+1)
        
        # verify
        A = np.diag(-R*np.ones(T), k=-1) + np.eye(T+1)
        b = y_seq - c_seq
        b[0] = b[0] + a0
        
        a_seq = np.linalg.inv(A) @ b
        a_seq = np.concatenate([[a0], a_seq])
        
        # check that a_T+1 = 0
        if verbose==1:
            print('check a_T+1=0:', np.abs(a_seq[-1] - 0) <= 1e-8)
        
        return c_seq, a_seq
    
    def welfare(self, c_seq):
        β_seq, g1, g2 = self.β_seq, self.g1, self.g2
        
        u_seq = g1 * c_seq - g2/2 * c_seq**2
        return β_seq @ u_seq
        
    
    def compute_variation(self, ξ1, ϕ, verbose=1):
        R, T, β_seq = self.R, self.T, self.β_seq
        
        ξ0 = ξ1*((1 - 1/R) / (1 - (1/R)**(T+1))) * ((1 - (ϕ/R)**(T+1)) / (1 - ϕ/R))
        v_seq = np.array([(ξ1*ϕ**t - ξ0) for t in range(T+1)])
        
        # check if it is feasible
        if verbose==1:
            print('check feasible:', np.round(β_seq @ v_seq, 7)==0)     # since β = 1/R
        
        c_opt, _ = self.compute_optimal(verbose=verbose)
        cvar_seq = c_opt + v_seq
        
        return cvar_seq
```


Below is an example where the consumer inherits $a_0<0$ (which can be interpreted as a student debt).

The income process $\{y_t\}_{t=0}^{T}$ is constant and positive up to $t=45$ and then becomes zero afterward.

```{code-cell} ipython3
# parameters
T=65
R = 1.05
g1 = 1
g2 = 1/2

# financial wealth
a0 = -2     # such as "student debt"

# income process
y_seq = np.concatenate([np.ones(46), np.zeros(20)])

# create an instance
mc = Consumption_smoothing(R=R, y_seq=y_seq, a0=a0, g1=g1, g2=g2, T=T)
c_seq, a_seq = mc.compute_optimal()

# compute welfare 
print('Welfare:', mc.welfare(c_seq))
```

```{code-cell} ipython3
plt.plot(range(T+1), y_seq, label='income')
plt.plot(range(T+1), c_seq, label='consumption')
plt.plot(range(T+2), a_seq, label='asset')
plt.plot(range(T+2), np.zeros(T+2), '--')

plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$c_t,y_t,a_t$')
plt.show()
```



We can visualize how $\xi_1$ and $\phi$ controls **budget-feasible variations**.

```{code-cell} ipython3
# visualize variational paths
cvar_seq1 = mc.compute_variation(ξ1=.01, ϕ=.95)
cvar_seq2 = mc.compute_variation(ξ1=.05, ϕ=.95)
cvar_seq3 = mc.compute_variation(ξ1=.01, ϕ=1.02)
cvar_seq4 = mc.compute_variation(ξ1=.05, ϕ=1.02)
```

```{code-cell} ipython3
print('welfare of optimal c: ', mc.welfare(c_seq))
print('variation 1: ', mc.welfare(cvar_seq1))
print('variation 2:', mc.welfare(cvar_seq2))
print('variation 3: ', mc.welfare(cvar_seq3))
print('variation 4:', mc.welfare(cvar_seq4))
```

```{code-cell} ipython3
plt.plot(range(T+1), c_seq, color='orange', label=r'Optimal $\vec{c}$ ')
plt.plot(range(T+1), cvar_seq1, color='tab:blue', label=r'$\xi_1 = 0.01, \phi = 0.95$')
plt.plot(range(T+1), cvar_seq2, color='tab:blue', ls='-.', label=r'$\xi_1 = 0.05, \phi = 0.95$')
plt.plot(range(T+1), cvar_seq3, color='tab:green', label=r'$\xi_1 = 0.01, \phi = 1.02$')
plt.plot(range(T+1), cvar_seq4, color='tab:green', ls='-.', label=r'$\xi_1 = 0.05, \phi = 1.02$')


plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$c_t$')
plt.show()
```

```{code-cell} ipython3
def welfare_ϕ(mc, ξ1, ϕ):
    "Compute welfare of variation sequence for given ϕ, ξ1 with an instance of our model mc"
    cvar_seq = mc.compute_variation(ξ1=ξ1, ϕ=ϕ, verbose=0)
    return mc.welfare(cvar_seq)

welfare_φ = np.vectorize(welfare_φ)
ξ1_arr = np.linspace(-0.5, 0.5, 20)

plt.plot(ξ1_arr, welfare_φ(mc, ξ1=ξ1_arr , ϕ=1.02))
plt.ylabel('welfare')
plt.xlabel(r'$\xi_1$')
plt.show()
```
