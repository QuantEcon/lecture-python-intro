# The Solow-Swan Growth Model

We consider a model due
to Robert Solow (1925--2014) and Trevor Swan (1918--1989).

The Solow--Swan economy contains a collection of identical agents, each of
whom saves the same fixed fraction of their current incomes.

Savings sustain or increase the stock of capital.  

Capital is combined with labor to produce
output, which in turn is paid out to workers and owners of capital.  

To keep things simple, we ignore population and productivity growth.


We will use the following imports

```{code-cell} ipython
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
```


## The Model

For each integer $t \geq 0$, output $Y_t$ in period $t$ is given by $Y_t =
F(K_t, L_t)$, where $K_t$ is capital, $L_t$ is labor and $F$ is an aggregate
production function.  

The function $F$ is assumed to be nonnegative and
**homogeneous of degree one**, meaning
that
%
$$
    F(\lambda K, \lambda L) = \lambda F(K, L)
    \quad \text{for all } \lambda \geq 0.
$$

Production functions with this property include 
%
* the \navy{Cobb-Douglas}\index{Cobb-Douglas} function $F(K, L) = A K^{\alpha}
  L^{1-\alpha}$ with $0 \leq \alpha \leq 1$ and
* the \navy{CES} function $F(K, L) = \left\{ a K^\rho + b L^\rho \right\}^{1/\rho}$
        with $a, b, \rho > 0$.

We assume a closed economy, so domestic investment equals aggregate domestic
saving.  

The saving rate is a positive constant $s$, so that aggregate
investment and saving both equal  $s Y_t$.  

Capital depreciates: without replenishing through investment, one unit of capital today
becomes $1-\delta$ units tomorrow. 

Thus, 
%
$$
    K_{t+1} = s F(K_t, L_t) + (1 - \delta) K_t.
$$
%

Without population growth, $L_t$ equals some constant $L$.  

Setting $k_t := K_t / L$ and using homogeneity of degree one now yields
%
\begin{equation*}
    k_{t+1} 
    = s \frac{F(K_t, L)}{L} + (1 - \delta) k_t
    = s F(k_t, 1) + (1 - \delta) k_t.
\end{equation*}
%

With  $f(k) := F(k, 1)$, the final expression for capital dynamics is
%
\begin{equation}
    \label{eq:solow}
    k_{t+1} = g(k_t) 
    \text{ where } g(k) := s f(k) + (1 - \delta) k.
\end{equation}
%

Our aim is to learn about the evolution of $k_t$ over time,
given an exogenous initial capital stock  $k_0$.


## A Graphical Perspective

To understand the dynamics of the sequence $(k_t)_{t \geq 0}$ we use a 45
degree diagram.  

To do so, we first
need to specify the functional form for $f$ and assign values to the parameters.

We choose the Cobb--Douglas specification $f(k) = A k^\alpha$. We set $A=2.0$,
$\alpha=0.3$, $s=0.3$ and $\delta=0.4$.  

The function $g$ from \eqref{eq:solow} is then plotted, along with the 45
degree line.  


```{code-cell} ipython
A, s, alpha, delta = 2, 0.3, 0.3, 0.4
x0 = 0.25
num_arrows = 8
ts_length = 12
xmin, xmax = 0, 3
g = lambda k: A * s * k**alpha + (1 - delta) * k

kstar = ((s * A) / delta)**(1/(1 - alpha))

xgrid = np.linspace(xmin, xmax, 12000)

fig, ax = plt.subplots()

#ax.set_xlim(xmin, xmax)
#ax.set_ylim(xmin, xmax)

lb = r'$g(k) = sAk^{\alpha} + (1 - \delta)k$'
ax.plot(xgrid, g(xgrid),  lw=2, alpha=0.6, label=lb)
ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='45')

fps = (kstar,)

ax.plot(fps, fps, 'go', ms=10, alpha=0.6)

ax.annotate(r'$k^* = (sA / \delta)^{(1/(1-\alpha))}$', 
         xy=(kstar, kstar),
         xycoords='data',
         xytext=(-40, -60),
         textcoords='offset points',
         fontsize=14,
         arrowprops=dict(arrowstyle="->"))

ax.legend(loc='upper left', frameon=False, fontsize=12)

ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))

ax.set_xlabel('$k_t$', fontsize=12)
ax.set_ylabel('$k_{t+1}$', fontsize=12)

plt.show()
```

Suppose, at some $k_t$, the value $g(k_t)$ lies strictly above the 45 degree line.

Then we have $k_{t+1} = g(k_t) > k_t$ and capital per worker rises.  

If $g(k_t) < k_t$ then capital per worker falls.  

If $g(k_t) = k_t$, then we are at a \navy{steady state} and $k_t$ remainds constant.  

A steady state of the model is a fixed point of the mapping $g$.

From the shape of the function $g$ in the figure, we see that
there is a unique steady state in $(0, \infty)$. 

It solves $k = s Ak^{\alpha} + (1-\delta)k$ and hence is given by
%
\begin{equation}
    \label{eq:kstarss}
    k^* := \left( \frac{s A}{\delta} \right)^{1/(1 - \alpha)}.
\end{equation}

If initial capital is below $k^*$, then capital increases over time.

If initial capital is above this level, then the reverse is true.  

Thus, we can
say that $(k_t)$ converges to $k^*$, regardless of initial capital
$k_0$.  

This is a form of global stability.


The next figure shows two time paths for capital, from
two distinct initial conditions, under the parameterization listed above.

At this parameterization, $k^* \approx 1.78$.

As expected, the time paths in the figure both converge to this value. 

```{code-cell} ipython
A, s, alpha, delta = 2, 0.3, 0.3, 0.4
x0 = np.array([.25, 1.25, 3.25])
ts_length = 20
ts = np.zeros(ts_length)

xmin, xmax = 0, ts_length
ymin, ymax = 0, 3.5

g = lambda k: A * s * k**alpha + (1 - delta) * k
# -

k_star = (s * A / delta)**(1/(1-alpha))
k_star

# +
fig, ax = plt.subplots()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


# simulate and plot time series
for x_init, col in zip(x0, colors):
    ts[0] = x_init
    for t in range(1, ts_length):
        ts[t] = g(ts[t-1])
    ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6, label=r'$k_0=%g$' %x_init)
    
ax.legend(fontsize=12)

ax.set_xlabel(r'$t$', fontsize=14)
ax.set_ylabel(r'$k_t$', fontsize=14)

plt.show()
```




## Growth in Continuous Time

In this section we investigate a continuous time version of the Solow--Swan
growth model.  

We will see how the smoothing provided by continuous time can
simplify analysis.


Recall  that the discrete time dynamics for capital are
given by $k_{t+1} = s f(k_t) + (1 - \delta) k_t$.  

A simple rearrangement gives the rate of change per unit of time:
%
\begin{equation}
    \Delta k_t = s f(k_t) - \delta k_t
    \quad \text{where} \quad 
    \Delta k_t := k_{t+1}  - k_t.
\end{equation}
%

Taking the time step to zero gives the continuous time limit
%
\begin{equation}
    \label{eq:solowc}
    k'_t = s f(k_t) - \delta k_t
    \qquad \text{with} \qquad
    k'_t := \frac{\diff}{\diff t} k_t.
\end{equation}
%

Our aim is to learn about the evolution of $k_t$ over time,
given initial stock  $k_0$.

A **steady state** for~\eqref{eq:solowc} is a value $k^*$
at which capital is unchanging, meaning $k'_t = 0$ or, equivalently,
$s f(k^*) = \delta k^*$.  

As with the discrete time case, when
$f(0)=0$ there is a trivial steady state at $k^*=0$.  

Hence we restrict the
state space for capital to $(0, \infty)$.  

Let's also assume
$f(k) = Ak^\alpha$, so $k^*$ solves 
$s A k^\alpha = \delta k$.  

The solution is the same as the discrete time case---see~\eqref{eq:kstarss}.

The dynamics of the Cobb--Douglas case are represented in
the next figure, maintaining the parameterization we used
above.  

Writing $k'_t = g(k_t)$ with $g(k) =
s Ak^\alpha - \delta k$, values of $k$ with $g(k) > 0$ imply that $k'_t > 0$, so
capital is increasing.  

When $g(k) < 0$, the opposite occurs.  Once again, high marginal returns to
savings at low levels of capital combined with low rates of return at high
levels of capital combine to yield global stability.


```{code-cell} ipython
A, s, alpha, delta = 2, 0.3, 0.3, 0.4


k_grid = np.linspace(0, 2.8, 10000)

g = lambda k: A * s * k**alpha + - delta * k

kstar = ((s * A) / delta)**(1/(1 - alpha))

# +
fig, ax = plt.subplots()

ax.plot(k_grid, g(k_grid), label='$g(k)$')
ax.plot(k_grid, 0 * k_grid, label="$k'=0$")

fps = (kstar,)

ax.plot(fps, 0, 'go', ms=10, alpha=0.6)


ax.annotate(r'$k^* = (sA / \delta)^{(1/(1-\alpha))}$', 
         xy=(kstar, 0),
         xycoords='data',
         xytext=(0, 60),
         textcoords='offset points',
         fontsize=12,
         arrowprops=dict(arrowstyle="->"))

ax.legend(loc='lower left', fontsize=12)

ax.set_xlabel("$k$",fontsize=12)
ax.set_ylabel("$k'$", fontsize=12)

ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((-0.3, 0, 0.3))

plt.show()
```

This shows global stability heuristically for a fixed parameterization, but
how would we show the same thing formally for a continuum of plausible parameters?

In the discrete time case,
a neat expression for $k_t$ is hard to obtain. 

In continuous time the process is easier: we can obtain a relatively simple
expression for $k_t$ that specifies the entire path.  

The first step is
to set $x_t := k_t^{1-\alpha}$, so that $x'_t = (1-\alpha) k_t^{-\alpha}
k'_t$.  

Substituting into $k'_t = sAk_t^\alpha - \delta k_t$ leads to the
linear differential equation
%
\begin{equation}\label{eq:xsolow}
    x'_t = (1-\alpha) (sA - \delta x_t).
\end{equation}
%

This equation has the exact solution
%
\begin{equation*}
    x_t 
    = \left(
        k_0^{1-\alpha} - \frac{sA}{\delta} 
      \right)
      \me^{-\delta (1-\alpha) t} + 
    \frac{sA}{\delta}.
\end{equation*}
%

(You can confirm that this function $x_t$ satisfies~\eqref{eq:xsolow} by
differentiating it.)  

Converting back to $k_t$ yields
%
\begin{equation}\label{eq:ssivs}
    k_t 
    = 
    \left[
        \left(
        k_0^{1-\alpha} - \frac{sA}{\delta} 
      \right)
      \me^{-\delta (1-\alpha) t} + 
    \frac{sA}{\delta}
    \right]^{1/(1-\alpha)}.
\end{equation}
%

Since $\delta > 0$ and $\alpha \in (0, 1)$, we see immediately that $k_t \to
k^*$ as $t \to \infty$ independent of $k_0$.

Thus, global stability holds.




## Stochastic Productivity

To bring the Solow--Swan model closer to data, we need to think about handling
random fluctuations in aggregate quantities. 

Among other things, this will
eliminate the unrealistic prediction that per-capita output $y_t = A
k^\alpha_t$ converges to a constant $y^* := A (k^*)^\alpha$.  

We shift to discrete time for the following discussion.

One approach is to replace constant productivity with some
stochastic sequence $(A_t)_{t \geq 1}$.  

Dynamics are now
%
\begin{equation}\label{eq:solowran}
    k_{t+1} = s A_{t+1} f(k_t) + (1 - \delta) k_t.
\end{equation}
%

The next figure shows some time series generated by this
model when $f$ is Cobb--Douglas, $(A_t)$ is {\sc iid} and lognormal, and other
parameters are as above.  

Now the long run convergence obtained in in the deterministic case breaks
down, since the system is hit with new shocks at each point in time.


```{code-cell} ipython
sig = 0.2
mu = np.log(2) - sig**2 / 2

def lgnorm():
    return np.exp(mu + sig * np.random.randn())

def G(k):
    return lgnorm() * s * k**alpha + (1 - delta) * k 

fig, ax = plt.subplots()


colors = ('g', 'b')
x0 = np.array([.25, 3.25])
ts_length = 50
ts = np.zeros(ts_length)

xmin, xmax = 0, ts_length
ymin, ymax = 0, 3.5

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# simulate and plot time series
for x_init, col in zip(x0, colors):
    ts[0] = x_init
    for t in range(1, ts_length):
        ts[t] = G(ts[t-1])
    ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6, label=r'$k_0=%g$' %x_init)
    
ax.legend(fontsize=12)

ax.set_xlabel(r'$t$', fontsize=14)
ax.set_ylabel(r'$k_t$', fontsize=14)


plt.show()
```

At the same time, if we look at the figure carefully,
we observe that the influence of initial conditions seems to die out, and the
two time series eventually fluctuate in  similar ranges.  

This hints at the fact that, for this model, stability is not lost after all.   

There is a higher-level notion of stability at work here, sometimes called
stochastic stability.

These notions are discussed in more advanced lectures on Markov processes.
