---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Consumption Smoothing

## Overview


In this lecture, we'll study a famous model of the "consumption function" that Milton Friedman {cite}`Friedman1956` and Robert Hall {cite}`Hall1978`)  proposed to fit some empirical data patterns that the original  Keynesian consumption function  described in this QuantEcon lecture {doc}`geometric series <geom_series>`  missed.

In this lecture, we'll study what is often  called the "consumption-smoothing model"  using  matrix multiplication and matrix inversion, the same tools that we used in this QuantEcon lecture {doc}`present values <pv>`. 

Formulas presented in  {doc}`present value formulas<pv>` are at the core of the consumption-smoothing model because we shall use them to define a consumer's "human wealth".

The  key idea that inspired Milton Friedman was that a person's non-financial income, i.e., his or
her wages from working, could be viewed as a dividend stream from that person's ''human capital''
and that standard asset-pricing formulas could be applied to compute a person's
''non-financial wealth'' that capitalizes the  earnings stream.  

```{note}
As we'll see in this QuantEcon lecture  {doc}`equalizing difference model <equalizing_difference>`,
Milton Friedman had used this idea  in his PhD thesis at Columbia University, 
eventually published as {cite}`kuznets1939incomes` and {cite}`friedman1954incomes`.
```

It will take a while for a "present value" or asset price explicitly to appear in this lecture, but when it does it will be a key actor.


## Analysis

As usual, we'll start by importing some Python modules.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
```

The model describes  a consumer who lives from time $t=0, 1, \ldots, S$, receives a stream $\{G_t\}_{t=0}^S$ of non-financial income and chooses a consumption stream $\{T_t\}_{t=0}^S$.

We usually think of the non-financial income stream as coming from the person's salary from supplying labor.  

The model  takes a non-financial income stream as an input, regarding it as "exogenous" in the sense of not being determined by the model. 

The consumer faces a gross interest rate of $R >1$ that is constant over time, at which she is free to borrow or lend, up to  limits that we'll describe below.

To set up the model, let 

 * $S \geq 2$  be a positive integer that constitutes a time-horizon. 
 * $G = \{G_t\}_{t=0}^S$ be a sequence of government expenditures. 
 * $B = \{B_t\}_{t=0}^{S+1}$ be a sequence of government debt.  
 * $T = \{T_t\}_{t=0}^S$ be a sequence of tax collections. 
 * $R \geq 1$ be a fixed gross one period interest rate. 
 * $\beta \in (0,1)$ be a fixed discount factor.  
 * $B_0$ be a given initial level of government debt
 * $B_{S+1} \geq 0$  be a terminal condition on final government debt. 

The sequence of financial wealth $a$ is to be determined by the model.

We require it to satisfy  two  **boundary conditions**:

   * it must  equal an exogenous value  $B_0$ at time $0$ 
   * it must equal or exceed an exogenous value  $B_{S+1}$ at time $S+1$.

The **terminal condition** $B_{S+1} \geq 0$ requires that the consumer not leave the model in debt.

(We'll soon see that a utility maximizing consumer won't want to die leaving positive assets, so she'll arrange her affairs to make
$B_{S+1} = 0$.)

The consumer faces a sequence of budget constraints that  constrains   sequences $(G, T, B)$

$$
B_{t+1} = R (B_t+ G_t - T_t), \quad t =0, 1, \ldots S
$$ (eq:B_t)

Equations {eq}`eq:B_t` constitute  $S+1$ such budget constraints, one for each $t=0, 1, \ldots, S$. 

Given a sequence $G$ of non-financial incomes, a large  set of pairs $(a, c)$ of (financial wealth, consumption) sequences  satisfy the sequence of budget constraints {eq}`eq:B_t`. 

Our model has the following logical flow.

 * start with an exogenous non-financial income sequence $G$, an initial financial wealth $B_0$, and 
 a candidate consumption path $c$.
 
 * use the system of equations {eq}`eq:B_t` for $t=0, \ldots, S$ to compute a path $a$ of financial wealth
 
 * verify that $B_{S+1}$ satisfies the terminal wealth constraint $B_{S+1} \geq 0$. 
    
     * If it does, declare that the candidate path is **budget feasible**. 
 
     * if the candidate consumption path is not budget feasible, propose a less greedy consumption  path and start over
     
Below, we'll describe how to execute these steps using linear algebra -- matrix inversion and multiplication.

The above procedure seems like a sensible way to find "budget-feasible" consumption paths $c$, i.e., paths that are consistent
with the exogenous non-financial income stream $G$, the initial financial  asset level $B_0$, and the terminal asset level $B_{S+1}$.

In general, there are **many** budget feasible consumption paths $c$.

Among all budget-feasible consumption paths, which one should a consumer want?


To answer this question, we shall eventually evaluate alternative budget feasible consumption paths $c$ using the following utility functional or **welfare criterion**:

```{math}
:label: welfare

L = \sum_{t=0}^S \beta^t (g_1 T_t - \frac{g_2}{2} T_t^2 )
```

where $g_1 > 0, g_2 > 0$.  

When $\beta R \approx 1$, the fact that the utility function $g_1 T_t - \frac{g_2}{2} T_t^2$ has diminishing marginal utility imparts a preference for consumption that is very smooth.  

Indeed, we shall see that when $\beta R = 1$ (a condition assumed by Milton Friedman {cite}`Friedman1956` and Robert Hall {cite}`Hall1978`),  criterion {eq}`welfare` assigns higher welfare to smoother consumption paths.

By **smoother** we mean as close as possible to being constant over time.  

The preference for smooth consumption paths that is built into the model gives it the  name "consumption-smoothing model".

Let's dive in and do some calculations that will help us understand how the model works. 

Here we use default parameters $R = 1.05$, $g_1 = 1$, $g_2 = 1/2$, and $S = 65$. 

We create a Python **namedtuple** to store these parameters with default values.

```{code-cell} ipython3
TaxSmoothing = namedtuple("TaxSmoothing", 
                        ["R", "g1", "g2", "β_seq", "S"])

def create_tax_smoothing_model(R=1.05, g1=1, g2=1/2, S=65):
    β = 1/R
    β_seq = np.array([β**i for i in range(S+1)])
    return TaxSmoothing(R, g1, g2, 
                                β_seq, S)
```

## Friedman-Hall consumption-smoothing model

A key object is what Milton Friedman called "human" or "non-financial" wealth at time $0$:


$$
h_0 \equiv \sum_{t=0}^S R^{-t} G_t = \begin{bmatrix} 1 & R^{-1} & \cdots & R^{-S} \end{bmatrix}
\begin{bmatrix} G_0 \cr G_1  \cr \vdots \cr G_S \end{bmatrix}
$$

Human or non-financial wealth  at time $0$ is evidently just the present value of the consumer's non-financial income stream $G$. 

Formally it very much resembles the asset price that we computed in this QuantEcon lecture {doc}`present values <pv>`.

Indeed, this is why Milton Friedman called it "human capital". 

By iterating on equation {eq}`eq:B_t` and imposing the terminal condition 

$$
B_{S+1} = 0,
$$

it is possible to convert a sequence of budget constraints {eq}`eq:B_t` into a single intertemporal constraint

$$ 
\sum_{t=0}^S R^{-t} T_t = B_0 + h_0. 
$$ (eq:budget_intertemp)

Equation {eq}`eq:budget_intertemp`  says that the present value of the consumption stream equals the sum of financial and non-financial (or human) wealth.

Robert Hall {cite}`Hall1978` showed that when $\beta R = 1$, a condition Milton Friedman had also  assumed, it is "optimal" for a consumer to smooth consumption by setting 

$$ 
T_t = T_0 \quad t =0, 1, \ldots, S
$$

(Later we'll present a "variational argument" that shows that this constant path maximizes
criterion {eq}`welfare` when $\beta R =1$.)

In this case, we can use the intertemporal budget constraint to write 

$$
T_t = T_0  = \left(\sum_{t=0}^S R^{-t}\right)^{-1} (B_0 + h_0), \quad t= 0, 1, \ldots, S.
$$ (eq:conssmoothing)

Equation {eq}`eq:conssmoothing` is the consumption-smoothing model in a nutshell.


## Mechanics of consumption-smoothing model  

As promised, we'll provide step-by-step instructions on how to use linear algebra, readily implemented in Python, to compute all  objects in play in  the consumption-smoothing model.

In the calculations below,  we'll  set default values of  $R > 1$, e.g., $R = 1.05$, and $\beta = R^{-1}$.

### Step 1

For a $(S+1) \times 1$  vector $G$, use matrix algebra to compute $h_0$

$$
h_0 = \sum_{t=0}^S R^{-t} G_t = \begin{bmatrix} 1 & R^{-1} & \cdots & R^{-S} \end{bmatrix}
\begin{bmatrix} G_0 \cr G_1  \cr \vdots \cr G_S \end{bmatrix}
$$

### Step 2

Compute an  time $0$   consumption $T_0 $ :

$$
T_t = T_0 = \left( \frac{1 - R^{-1}}{1 - R^{-(S+1)}} \right) (B_0 + \sum_{t=0}^S R^{-t} G_t ) , \quad t = 0, 1, \ldots, S
$$

### Step 3

Use  the system of equations {eq}`eq:B_t` for $t=0, \ldots, S$ to compute a path $a$ of financial wealth.

To do this, we translate that system of difference equations into a single matrix equation as follows:


$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-R & 1 & 0 & \cdots & 0 & 0 & 0 \cr
0 & -R & 1 & \cdots & 0 & 0 & 0 \cr
\vdots  &\vdots & \vdots & \cdots & \vdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -R & 1 & 0 \cr
0 & 0 & 0 & \cdots & 0 & -R & 1
\end{bmatrix} 
\begin{bmatrix} B_1 \cr B_2 \cr B_3 \cr \vdots \cr B_S \cr B_{S+1} 
\end{bmatrix}
= R 
\begin{bmatrix} G_0 + B_0 - T_0 \cr G_1 - T_0 \cr G_2 - T_0 \cr \vdots\cr G_{S-1} - T_0 \cr G_S - T_0
\end{bmatrix}
$$

Multiply both sides by the inverse of the matrix on the left side to compute

$$
 \begin{bmatrix} B_1 \cr B_2 \cr B_3 \cr \vdots \cr B_S \cr B_{S+1} \end{bmatrix}
$$


Because we have built into  our calculations that the consumer leaves the model  with exactly zero assets, just barely satisfying the
terminal condition that $B_{S+1} \geq 0$, it should turn out   that 

$$
B_{S+1} = 0.
$$
 

Let's verify this with  Python code.

First we implement the model with `compute_optimal`

```{code-cell} ipython3
def compute_optimal(model, B0, G_seq):
    R, S = model.R, model.S

    # non-financial wealth
    h0 = model.β_seq @ G_seq     # since β = 1/R

    # c0
    c0 = (1 - 1/R) / (1 - (1/R)**(S+1)) * (B0 + h0)
    T_seq = c0*np.ones(S+1)

    # verify
    A = np.diag(-R*np.ones(S), k=-1) + np.eye(S+1)
    b = G_seq - T_seq
    b[0] = b[0] + B0

    B_seq = np.linalg.inv(A) @ b
    B_seq = np.concatenate([[B0], B_seq])

    return T_seq, B_seq, h0
```

We use an example where the consumer inherits $B_0<0$.

This  can be interpreted as a student debt.

The non-financial process $\{G_t\}_{t=0}^{S}$ is constant and positive up to $t=45$ and then becomes zero afterward.

The drop in non-financial income late in life reflects retirement from work.

```{code-cell} ipython3
# Financial wealth
B0 = -2     # such as "student debt"

# non-financial Income process
G_seq = np.concatenate([np.ones(46), np.zeros(20)])

cs_model = create_tax_smoothing_model()
T_seq, B_seq, h0 = compute_optimal(cs_model, B0, G_seq)

print('check B_S+1=0:', 
      np.abs(B_seq[-1] - 0) <= 1e-8)
```

The graphs below  show  paths of non-financial income, consumption, and financial assets.

```{code-cell} ipython3
# Sequence Length
S = cs_model.S

plt.plot(range(S+1), G_seq, label='non-financial income')
plt.plot(range(S+1), T_seq, label='consumption')
plt.plot(range(S+2), B_seq, label='financial wealth')
plt.plot(range(S+2), np.zeros(S+2), '--')

plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$T_t,G_t,B_t$')
plt.show()
```

Note that $B_{S+1} = 0$, as anticipated.

We can  evaluate  welfare criterion {eq}`welfare`

```{code-cell} ipython3
def welfare(model, T_seq):
    β_seq, g1, g2 = model.β_seq, model.g1, model.g2

    u_seq = g1 * T_seq - g2/2 * T_seq**2
    return β_seq @ u_seq

print('Welfare:', welfare(cs_model, T_seq))
```

### Experiments

In this section we describe  how a  consumption sequence would optimally respond to different  sequences sequences of non-financial income.

First we create  a function `plot_cs` that generates graphs for different instances of the  consumption-smoothing model `cs_model`.

This will  help us avoid rewriting code to plot outcomes for different non-financial income sequences.

```{code-cell} ipython3
def plot_cs(model,    # consumption-smoothing model      
            B0,       # initial financial wealth
            G_seq     # non-financial income process
           ):
    
    # Compute optimal consumption
    T_seq, B_seq, h0 = compute_optimal(model, B0, G_seq)
    
    # Sequence length
    S = cs_model.S
    
    # Generate plot
    plt.plot(range(S+1), G_seq, label='non-financial income')
    plt.plot(range(S+1), T_seq, label='consumption')
    plt.plot(range(S+2), B_seq, label='financial wealth')
    plt.plot(range(S+2), np.zeros(S+2), '--')
    
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$T_t,G_t,B_t$')
    plt.show()
```

In the experiments below, please study how consumption and financial asset sequences vary across different sequences for non-financial income.

#### Experiment 1: one-time gain/loss

We first assume a one-time windfall of $W_0$ in year 21 of the income sequence $G$.  

We'll make $W_0$ big - positive to indicate a one-time windfall, and negative to indicate a one-time "disaster".

```{code-cell} ipython3
# Windfall W_0 = 2.5
G_seq_pos = np.concatenate([np.ones(21), np.array([2.5]), np.ones(24), np.zeros(20)])

plot_cs(cs_model, B0, G_seq_pos)
```

```{code-cell} ipython3
# Disaster W_0 = -2.5
G_seq_neg = np.concatenate([np.ones(21), np.array([-2.5]), np.ones(24), np.zeros(20)])

plot_cs(cs_model, B0, G_seq_neg)
```

#### Experiment 2: permanent wage gain/loss

Now we assume a permanent  increase in income of $L$ in year 21 of the $G$-sequence.

Again we can study positive and negative cases

```{code-cell} ipython3
# Positive permanent income change L = 0.5 when t >= 21
G_seq_pos = np.concatenate(
    [np.ones(21), 1.5*np.ones(25), np.zeros(20)])

plot_cs(cs_model, B0, G_seq_pos)
```

```{code-cell} ipython3
# Negative permanent income change L = -0.5 when t >= 21
G_seq_neg = np.concatenate(
    [np.ones(21), .5*np.ones(25), np.zeros(20)])

plot_cs(cs_model, B0, G_seq_neg)
```

#### Experiment 3: a late starter

Now we simulate a $G$ sequence in which a person gets zero for 46 years, and then works and gets 1 for the last 20 years of life (a "late starter")

```{code-cell} ipython3
# Late starter
G_seq_late = np.concatenate(
    [np.zeros(46), np.ones(20)])

plot_cs(cs_model, B0, G_seq_late)
```

#### Experiment 4: geometric earner

Now we simulate a geometric $G$ sequence in which a person gets $G_t = \lambda^t G_0$ in first 46 years.

We first experiment with $\lambda = 1.05$

```{code-cell} ipython3
# Geometric earner parameters where λ = 1.05
λ = 1.05
G_0 = 1
t_max = 46

# Generate geometric G sequence
geo_seq = λ ** np.arange(t_max) * G_0 
G_seq_geo = np.concatenate(
            [geo_seq, np.zeros(20)])

plot_cs(cs_model, B0, G_seq_geo)
```

Now we show the behavior when $\lambda = 0.95$

```{code-cell} ipython3
λ = 0.95

geo_seq = λ ** np.arange(t_max) * G_0 
G_seq_geo = np.concatenate(
            [geo_seq, np.zeros(20)])

plot_cs(cs_model, B0, G_seq_geo)
```

What happens when $\lambda$ is negative

```{code-cell} ipython3
λ = -0.95

geo_seq = λ ** np.arange(t_max) * G_0 
G_seq_geo = np.concatenate(
            [geo_seq, np.zeros(20)])

plot_cs(cs_model, B0, G_seq_geo)
```

### Feasible consumption variations

We promised to justify  our claim that a constant consumption play $T_t = T_0$ for all
$t$ is optimal.  

Let's do that now.

The approach we'll take is  an elementary  example  of the "calculus of variations". 

Let's dive in and see what the key idea is.  

To explore what types of consumption paths are welfare-improving, we shall create an **admissible consumption path variation sequence** $\{v_t\}_{t=0}^S$
that satisfies

$$
\sum_{t=0}^S R^{-t} v_t = 0
$$

This equation says that the **present value** of admissible consumption path variations must be zero.

So once again, we encounter a formula for the present value of an "asset":

   * we require that the present value of consumption path variations be zero.

Here we'll restrict ourselves to  a two-parameter class of admissible consumption path variations
of the form

$$
v_t = \xi_1 \phi^t - \xi_0
$$

We say two and not three-parameter class because $\xi_0$ will be a function of $(\phi, \xi_1; R)$ that guarantees that the variation sequence is feasible. 

Let's compute that function.

We require

$$
\sum_{t=0}^S R^{-t}\left[ \xi_1 \phi^t - \xi_0 \right] = 0
$$

which implies that

$$
\xi_1 \sum_{t=0}^S \phi_t R^{-t} - \xi_0 \sum_{t=0}^S R^{-t} = 0
$$

which implies that

$$
\xi_1 \frac{1 - (\phi R^{-1})^{S+1}}{1 - \phi R^{-1}} - \xi_0 \frac{1 - R^{-(S+1)}}{1-R^{-1} } =0
$$

which implies that

$$
\xi_0 = \xi_0(\phi, \xi_1; R) = \xi_1 \left(\frac{1 - R^{-1}}{1 - R^{-(S+1)}}\right) \left(\frac{1 - (\phi R^{-1})^{S+1}}{1 - \phi R^{-1}}\right)
$$ 

This is our formula for $\xi_0$.  

**Key Idea:** if $c^o$ is a budget-feasible consumption path, then so is $c^o + v$,
where $v$ is a budget-feasible variation.

Given $R$, we thus have a two parameter class of budget feasible variations $v$ that we can use
to compute alternative consumption paths, then evaluate their welfare.

Now let's compute and plot consumption path variations

```{code-cell} ipython3
def compute_variation(model, ξ1, ϕ, B0, G_seq, verbose=1):
    R, S, β_seq = model.R, model.S, model.β_seq

    ξ0 = ξ1*((1 - 1/R) / (1 - (1/R)**(S+1))) * ((1 - (ϕ/R)**(S+1)) / (1 - ϕ/R))
    v_seq = np.array([(ξ1*ϕ**t - ξ0) for t in range(S+1)])
    
    if verbose == 1:
        print('check feasible:', np.isclose(β_seq @ v_seq, 0))     # since β = 1/R

    T_opt, _, _ = compute_optimal(model, B0, G_seq)
    cvar_seq = T_opt + v_seq

    return cvar_seq
```

We visualize variations for $\xi_1 \in \{.01, .05\}$ and $\phi \in \{.95, 1.02\}$

```{code-cell} ipython3
fig, ax = plt.subplots()

ξ1s = [.01, .05]
ϕs= [.95, 1.02]
colors = {.01: 'tab:blue', .05: 'tab:green'}

params = np.array(np.meshgrid(ξ1s, ϕs)).T.reshape(-1, 2)

for i, param in enumerate(params):
    ξ1, ϕ = param
    print(f'variation {i}: ξ1={ξ1}, ϕ={ϕ}')
    cvar_seq = compute_variation(model=cs_model, 
                                 ξ1=ξ1, ϕ=ϕ, B0=B0, 
                                 G_seq=G_seq)
    print(f'welfare={welfare(cs_model, cvar_seq)}')
    print('-'*64)
    if i % 2 == 0:
        ls = '-.'
    else: 
        ls = '-'  
    ax.plot(range(S+1), cvar_seq, ls=ls, 
            color=colors[ξ1], 
            label=fr'$\xi_1 = {ξ1}, \phi = {ϕ}$')

plt.plot(range(S+1), T_seq, 
         color='orange', label=r'Optimal $\vec{c}$ ')

plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$T_t$')
plt.show()
```

We can even use the Python `np.gradient` command to compute derivatives of welfare with respect to our two parameters.  

We are teaching the key idea beneath the **calculus of variations**.

First, we define the welfare with respect to $\xi_1$ and $\phi$

```{code-cell} ipython3
def welfare_rel(ξ1, ϕ):
    """
    Compute welfare of variation sequence 
    for given ϕ, ξ1 with a consumption-smoothing model
    """
    
    cvar_seq = compute_variation(cs_model, ξ1=ξ1, 
                                 ϕ=ϕ, B0=B0, 
                                 G_seq=G_seq, 
                                 verbose=0)
    return welfare(cs_model, cvar_seq)

# Vectorize the function to allow array input
welfare_vec = np.vectorize(welfare_rel)
```

Then we can visualize the relationship between welfare and $\xi_1$ and compute its derivatives

```{code-cell} ipython3
ξ1_arr = np.linspace(-0.5, 0.5, 20)

plt.plot(ξ1_arr, welfare_vec(ξ1_arr, 1.02))
plt.ylabel('welfare')
plt.xlabel(r'$\xi_1$')
plt.show()

welfare_grad = welfare_vec(ξ1_arr, 1.02)
welfare_grad = np.gradient(welfare_grad)
plt.plot(ξ1_arr, welfare_grad)
plt.ylabel('derivative of welfare')
plt.xlabel(r'$\xi_1$')
plt.show()
```

The same can be done on $\phi$

```{code-cell} ipython3
ϕ_arr = np.linspace(-0.5, 0.5, 20)

plt.plot(ξ1_arr, welfare_vec(0.05, ϕ_arr))
plt.ylabel('welfare')
plt.xlabel(r'$\phi$')
plt.show()

welfare_grad = welfare_vec(0.05, ϕ_arr)
welfare_grad = np.gradient(welfare_grad)
plt.plot(ξ1_arr, welfare_grad)
plt.ylabel('derivative of welfare')
plt.xlabel(r'$\phi$')
plt.show()
```

## Wrapping up the consumption-smoothing model

The consumption-smoothing model of Milton Friedman {cite}`Friedman1956` and Robert Hall {cite}`Hall1978`) is a cornerstone of modern macro that has important ramifications for the size of the Keynesian  "fiscal policy multiplier" described briefly in
QuantEcon lecture {doc}`geometric series <geom_series>`.  

In particular,  it  **lowers** the government expenditure  multiplier relative to  one implied by
the original Keynesian consumption function presented in {doc}`geometric series <geom_series>`.

Friedman's   work opened the door to an enlightening literature on the aggregate consumption function and associated government expenditure  multipliers that
remains  active today.  


## Appendix: solving difference equations with linear algebra

In the preceding sections we have used linear algebra to solve a consumption-smoothing model.  

The same tools from linear algebra -- matrix multiplication and matrix inversion -- can be used  to study many other dynamic models.

We'll conclude this lecture by giving a couple of examples.

We'll describe a useful way of representing and "solving" linear difference equations. 

To generate some $G$ vectors, we'll just write down a linear difference equation
with appropriate initial conditions and then   use linear algebra to solve it.

### First-order difference equation

We'll start with a first-order linear difference equation for $\{G_t\}_{t=0}^S$:

$$
G_{t} = \lambda G_{t-1}, \quad t = 1, 2, \ldots, S
$$

where  $G_0$ is a given  initial condition.


We can cast this set of $S$ equations as a single  matrix equation

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
-\lambda & 1 & 0 & \cdots & 0 & 0 \cr
0 & -\lambda & 1 & \cdots & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda & 1 
\end{bmatrix} 
\begin{bmatrix}
G_1 \cr G_2 \cr G_3 \cr \vdots \cr G_S 
\end{bmatrix}
= 
\begin{bmatrix} 
\lambda G_0 \cr 0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$ (eq:first_order_lin_diff)


Multiplying both sides of {eq}`eq:first_order_lin_diff`  by the  inverse of the matrix on the left provides the solution

```{math}
:label: fst_ord_inverse

\begin{bmatrix} 
G_1 \cr G_2 \cr G_3 \cr \vdots \cr G_S 
\end{bmatrix} 
= 
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
\lambda & 1 & 0 & \cdots & 0 & 0 \cr
\lambda^2 & \lambda & 1 & \cdots & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
\lambda^{S-1} & \lambda^{S-2} & \lambda^{S-3} & \cdots & \lambda & 1 
\end{bmatrix}
\begin{bmatrix} 
\lambda G_0 \cr 0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
```

```{exercise}
:label: consmooth_ex1

To get {eq}`fst_ord_inverse`, we multiplied both sides of  {eq}`eq:first_order_lin_diff` by  the inverse of the matrix $A$. Please confirm that 

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 \cr
\lambda & 1 & 0 & \cdots & 0 & 0 \cr
\lambda^2 & \lambda & 1 & \cdots & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
\lambda^{S-1} & \lambda^{S-2} & \lambda^{S-3} & \cdots & \lambda & 1 
\end{bmatrix}
$$

is the inverse of $A$ and check that $A A^{-1} = I$

```

### Second-order difference equation

A second-order linear difference equation for $\{G_t\}_{t=0}^S$ is

$$
G_{t} = \lambda_1 G_{t-1} + \lambda_2 G_{t-2}, \quad t = 1, 2, \ldots, S
$$

where now $G_0$ and $G_{-1}$ are two given initial equations determined outside the model. 

As we did with the first-order difference equation, we can cast this set of $S$ equations as a single matrix equation

$$
\begin{bmatrix} 
1 & 0 & 0 & \cdots & 0 & 0 & 0 \cr
-\lambda_1 & 1 & 0 & \cdots & 0 & 0 & 0 \cr
-\lambda_2 & -\lambda_1 & 1 & \cdots & 0 & 0 & 0 \cr
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \cr
0 & 0 & 0 & \cdots & -\lambda_2 & -\lambda_1 & 1 
\end{bmatrix} 
\begin{bmatrix} 
G_1 \cr G_2 \cr G_3 \cr \vdots \cr G_S 
\end{bmatrix}
= 
\begin{bmatrix} 
\lambda_1 G_0 + \lambda_2 G_{-1} \cr \lambda_2 G_0 \cr 0 \cr \vdots \cr 0 
\end{bmatrix}
$$

Multiplying both sides by  inverse of the matrix on the left again provides the solution.

```{exercise}
:label: consmooth_ex2

As an exercise, we ask you to represent and solve a **third-order linear difference equation**.
How many initial conditions must you specify?
```
