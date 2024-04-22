---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

(scalar_dynam)=
# Dynamics in One Dimension

```{admonition} Migrated lecture
:class: warning

This lecture has moved from our [Intermediate Quantitative Economics with Python](https://python.quantecon.org/intro.html) lecture series and is now a part of [A First Course in Quantitative Economics](https://intro.quantecon.org/intro.html).
```

## Overview

In this lecture we give a quick introduction to discrete time dynamics in one dimension.

* In one-dimensional models, the state of the system is described by a single variable.
* The variable is a number (that is, a point in $\mathbb R$).

While most quantitative models have two or more state variables, the
one-dimensional setting is a good place to learn the foundations of dynamics
and understand key concepts.

Let's start with some standard imports:

```{code-cell} ipython
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

## Some definitions

This section sets out the objects of interest and the kinds of properties we study.

### Composition of functions

For this lecture you should know the following.

If 

* $g$ is a function from $A$ to $B$ and
* $f$ is a function from $B$ to $C$, 

then the **composition** $f \circ g$ of $f$ and $g$ is defined by

$$ 
    (f \circ g)(x) = f(g(x))
$$

For example, if 

* $A=B=C=\mathbb R$, the set of real numbers, 
* $g(x)=x^2$ and $f(x)=\sqrt{x}$, then $(f \circ g)(x) = \sqrt{x^2} = |x|$.

If $f$ is a function from $A$ to itself, then $f^2$ is the composition of $f$
with itself.

For example, if $A = (0, \infty)$, the set of positive numbers, and $f(x) =
\sqrt{x}$, then 

$$
    f^2(x) = \sqrt{\sqrt{x}} = x^{1/4}
$$

Similarly, if $n$ is an integer, then $f^n$ is $n$ compositions of $f$ with
itself.

In the example above, $f^n(x) = x^{1/(2^n)}$.



### Dynamic systems

A **(discrete time) dynamic system** is a set $S$ and a function $g$ that sends
set $S$ back into to itself.


Examples of dynamic systems include

*  $S = (0, 1)$ and $g(x) = \sqrt{x}$
*  $S = (0, 1)$ and $g(x) = x^2$
*  $S = \mathbb Z$ (the integers) and $g(x) = 2 x$


On the other hand, if  $S = (-1, 1)$ and $g(x) = x+1$, then $S$ and $g$ do not
form a dynamic system, since $g(1) = 2$.

* $g$ does not always send points in $S$ back into $S$.



### Dynamic systems

We care about dynamic systems because we can use them to study dynamics!

Given a dynamic system consisting of set $S$ and function $g$, we can create
a sequence $\{x_t\}$ of points in $S$ by setting

```{math}
:label: sdsod
    x_{t+1} = g(x_t)
    \quad \text{ with } 
    x_0 \text{ given}.
```

This means that we choose some number $x_0$ in $S$ and then take

```{math}
:label: sdstraj
    x_0, \quad
    x_1 = g(x_0), \quad
    x_2 = g(x_1) = g(g(x_0)), \quad \text{etc.}
```

This sequence $\{x_t\}$ is called the **trajectory** of $x_0$ under $g$.

In this setting, $S$ is called the **state space** and $x_t$ is called the
**state variable**.

Recalling that $g^n$ is the $n$ compositions of $g$ with itself, 
we can write the trajectory more simply as 

$$
    x_t = g^t(x_0) \quad \text{ for } t \geq 0.
$$

In all of what follows, we are going to assume that $S$ is a subset of
$\mathbb R$, the real numbers.

Equation {eq}`sdsod` is sometimes called a **first order difference equation**

* first order means dependence on only one lag (i.e., earlier states such as $x_{t-1}$ do not enter into {eq}`sdsod`).



### Example: A Linear Model

One simple example of a dynamic system is when $S=\mathbb R$ and $g(x)=ax +
b$, where $a, b$ are fixed constants.

This leads to the **linear difference equation**

$$
    x_{t+1} = a x_t + b 
    \quad \text{ with } 
    x_0 \text{ given}.
$$


The trajectory of $x_0$ is 

```{math}
:label: sdslinmodpath

x_0, \quad
a x_0 + b, \quad
a^2 x_0 + a b + b, \quad \text{etc.}
```

Continuing in this way, and using our knowledge of {doc}`geometric series
<geom_series>`, we find that, for any $t \geq 0$,

```{math}
:label: sdslinmod
    x_t = a^t x_0 + b \frac{1 - a^t}{1 - a}
```

We have an exact expression for $x_t$ for all $t$ and hence a full
understanding of the dynamics.

Notice in particular that $|a| < 1$, then, by {eq}`sdslinmod`, we have

```{math}
:label: sdslinmodc

x_t \to  \frac{b}{1 - a} \text{ as } t \to \infty
```

regardless of $x_0$

This is an example of what is called global stability, a topic we return to
below.




### Example: A Nonlinear Model

In the linear example above, we obtained an exact analytical expression for
$x_t$ in terms of arbitrary $t$ and $x_0$.

This made analysis of dynamics very easy.

When models are nonlinear, however, the situation can be quite different.

For example, recall how we [previously studied](https://python-programming.quantecon.org/python_oop.html#example-the-solow-growth-model) the law of motion for the Solow growth model, a simplified version of which is

```{math}
:label: solow_lom2

k_{t+1} = s z k_t^{\alpha} + (1 - \delta) k_t
```

Here $k$ is capital stock and $s, z, \alpha, \delta$ are positive
parameters with $0 < \alpha, \delta < 1$.

If you try to iterate like we did in {eq}`sdslinmodpath`, you will find that
the algebra gets messy quickly.

Analyzing the dynamics of this model requires a different method (see below).






## Stability

Consider a fixed dynamic system consisting of set $S \subset \mathbb R$ and
$g$ mapping $S$ to $S$.

(scalar-dynam:steady-state)=
### Steady states

A **steady state** of this system is a
point $x^*$ in $S$ such that $x^* = g(x^*)$.

In other words, $x^*$ is a **fixed point** of the function $g$ in
$S$.

For example, for the linear model $x_{t+1} = a x_t + b$, you can use the
definition to check that

* $x^* := b/(1-a)$ is a steady state whenever $a \not= 1$.
* if $a = 1$ and $b=0$, then every $x \in \mathbb R$ is a
  steady state.
* if $a = 1$ and $b \not= 0$, then the linear model has no steady
  state in $\mathbb R$.



(scalar-dynam:global-stability)=
### Global stability

A steady state $x^*$ of the dynamic system is called
**globally stable** if, for all $x_0 \in S$,

$$
x_t = g^t(x_0) \to x^* \text{ as } t \to \infty
$$

For example, in the linear model $x_{t+1} = a x_t + b$ with $a
\not= 1$, the steady state $x^*$

* is globally stable if $|a| < 1$ and
* fails to be globally stable otherwise.

This follows directly from {eq}`sdslinmod`.


### Local stability

A steady state $x^*$ of the dynamic system is called
**locally stable** if there exists an $\epsilon > 0$ such that

$$
| x_0 - x^* | < \epsilon
\; \implies \;
x_t = g^t(x_0) \to x^* \text{ as } t \to \infty
$$

Obviously every globally stable steady state is also locally stable.

We will see examples below where the converse is not true.







## Graphical analysis

As we saw above, analyzing the dynamics for nonlinear models is nontrivial.

There is no single way to tackle all nonlinear models.

However, there is one technique for one-dimensional models that provides a
great deal of intuition.

This is a graphical approach based on **45 degree diagrams**.

Let's look at an example: the Solow model with dynamics given in {eq}`solow_lom2`.

We begin with some plotting code that you can ignore at first reading.

The function of the code is to produce 45 degree diagrams and time series
plots.



```{code-cell} ipython
---
tags: [hide-input,
       output_scroll]
---
def subplots():
    "Custom subplots with axes throught the origin"
    fig, ax = plt.subplots()

    # Set the axes through the origin
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
        ax.spines[spine].set_color('green')
    for spine in ['right', 'top']:
        ax.spines[spine].set_color('none')

    return fig, ax


def plot45(g, xmin, xmax, x0, num_arrows=6, var='x'):

    xgrid = np.linspace(xmin, xmax, 200)

    fig, ax = subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    hw = (xmax - xmin) * 0.01
    hl = 2 * hw
    arrow_args = dict(fc="k", ec="k", head_width=hw,
            length_includes_head=True, lw=1,
            alpha=0.6, head_length=hl)

    ax.plot(xgrid, g(xgrid), 'b-', lw=2, alpha=0.6, label='g')
    ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='45')

    x = x0
    xticks = [xmin]
    xtick_labels = [xmin]

    for i in range(num_arrows):
        if i == 0:
            ax.arrow(x, 0.0, 0.0, g(x), **arrow_args) # x, y, dx, dy
        else:
            ax.arrow(x, x, 0.0, g(x) - x, **arrow_args)
            ax.plot((x, x), (0, x), 'k', ls='dotted')

        ax.arrow(x, g(x), g(x) - x, 0, **arrow_args)
        xticks.append(x)
        xtick_labels.append(r'${}_{}$'.format(var, str(i)))

        x = g(x)
        xticks.append(x)
        xtick_labels.append(r'${}_{}$'.format(var, str(i+1)))
        ax.plot((x, x), (0, x), 'k', ls='dotted')

    xticks.append(xmax)
    xtick_labels.append(xmax)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(xtick_labels)

    bbox = (0., 1.04, 1., .104)
    legend_args = {'bbox_to_anchor': bbox, 'loc': 'upper right'}

    ax.legend(ncol=2, frameon=False, **legend_args, fontsize=14)
    plt.show()

def ts_plot(g, xmin, xmax, x0, ts_length=6, var='x'):
    fig, ax = subplots()
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'${}_t$'.format(var), fontsize=14)
    x = np.empty(ts_length)
    x[0] = x0
    for t in range(ts_length-1):
        x[t+1] = g(x[t])
    ax.plot(range(ts_length),
            x,
            'bo-',
            alpha=0.6,
            lw=2,
            label=r'${}_t$'.format(var))
    ax.legend(loc='best', fontsize=14)
    ax.set_xticks(range(ts_length))
    plt.show()
```

Let's create a 45 degree diagram for the Solow model with a fixed set of
parameters

```{code-cell} ipython
A, s, alpha, delta = 2, 0.3, 0.3, 0.4
```

Here's the update function corresponding to the model.

```{code-cell} ipython
def g(k):
    return A * s * k**alpha + (1 - delta) * k
```

Here is the 45 degree plot.

```{code-cell} ipython
xmin, xmax = 0, 4  # Suitable plotting region.

plot45(g, xmin, xmax, 0, num_arrows=0)
```

The plot shows the function $g$ and the 45 degree line.

Think of $k_t$ as a value on the horizontal axis.

To calculate $k_{t+1}$, we can use the graph of $g$ to see its
value on the vertical axis.

Clearly,

* If $g$ lies above the 45 degree line at this point, then we have $k_{t+1} > k_t$.
* If $g$ lies below the 45 degree line at this point, then we have $k_{t+1} < k_t$.
* If $g$ hits the 45 degree line at this point, then we have $k_{t+1} = k_t$, so $k_t$ is a steady state.

For the Solow model, there are two steady states when $S = \mathbb R_+ =
[0, \infty)$.

* the origin $k=0$
* the unique positive number such that $k = s z k^{\alpha} + (1 - \delta) k$.

By using some algebra, we can show that in the second case, the steady state is

$$
k^* = \left( \frac{sz}{\delta} \right)^{1/(1-\alpha)}
$$

### Trajectories

By the preceding discussion, in regions where $g$ lies above the 45 degree line, we know that the trajectory is increasing.

The next figure traces out a trajectory in such a region so we can see this more clearly.

The initial condition is $k_0 = 0.25$.

```{code-cell} ipython
k0 = 0.25

plot45(g, xmin, xmax, k0, num_arrows=5, var='k')
```

We can plot the time series of capital corresponding to the figure above as
follows:

```{code-cell} ipython
ts_plot(g, xmin, xmax, k0, var='k')
```

Here's a somewhat longer view:

```{code-cell} ipython
ts_plot(g, xmin, xmax, k0, ts_length=20, var='k')
```

When capital stock is higher than the unique positive steady state, we see that
it declines:

```{code-cell} ipython
k0 = 2.95

plot45(g, xmin, xmax, k0, num_arrows=5, var='k')
```

Here is the time series:

```{code-cell} ipython
ts_plot(g, xmin, xmax, k0, var='k')
```

### Complex dynamics

The Solow model is nonlinear but still generates very regular dynamics.

One model that generates irregular dynamics is the **quadratic map**

$$
g(x) = 4 x (1 - x),
\qquad x \in [0, 1]
$$

Let's have a look at the 45 degree diagram.

```{code-cell} ipython
xmin, xmax = 0, 1
g = lambda x: 4 * x * (1 - x)

x0 = 0.3
plot45(g, xmin, xmax, x0, num_arrows=0)
```

Now let's look at a typical trajectory.

```{code-cell} ipython
plot45(g, xmin, xmax, x0, num_arrows=6)
```

Notice how irregular it is.

Here is the corresponding time series plot.

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=6)
```

The irregularity is even clearer over a longer time horizon:

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=20)
```

## Exercises

```{exercise}
:label: sd_ex1

Consider again the linear model $x_{t+1} = a x_t + b$ with $a
\not=1$.

The unique steady state is $b / (1 - a)$.

The steady state is globally stable if $|a| < 1$.

Try to illustrate this graphically by looking at a range of initial conditions.

What differences do you notice in the cases $a \in (-1, 0)$ and $a
\in (0, 1)$?

Use $a=0.5$ and then $a=-0.5$ and study the trajectories

Set $b=1$ throughout.
```

```{solution-start} sd_ex1
:class: dropdown
```

We will start with the case $a=0.5$.

Let's set up the model and plotting region:

```{code-cell} ipython
a, b = 0.5, 1
xmin, xmax = -1, 3
g = lambda x: a * x + b
```

Now let's plot a trajectory:

```{code-cell} ipython
x0 = -0.5
plot45(g, xmin, xmax, x0, num_arrows=5)
```

Here is the corresponding time series, which converges towards the steady
state.

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=10)
```

Now let's try $a=-0.5$ and see what differences we observe.

Let's set up the model and plotting region:

```{code-cell} ipython
a, b = -0.5, 1
xmin, xmax = -1, 3
g = lambda x: a * x + b
```

Now let's plot a trajectory:

```{code-cell} ipython
x0 = -0.5
plot45(g, xmin, xmax, x0, num_arrows=5)
```

Here is the corresponding time series, which converges towards the steady
state.

```{code-cell} ipython
ts_plot(g, xmin, xmax, x0, ts_length=10)
```

Once again, we have convergence to the steady state but the nature of
convergence differs.

In particular, the time series jumps from above the steady state to below it
and back again.

In the current context, the series is said to exhibit **damped oscillations**.

```{solution-end}
```
