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

# The Cobweb Model

+++

The cobweb model {cite}:`10.2307/1236509` is a model of prices and quantities in a given market --- let's say, a market for soy beans.

The model dates back to the 1930s.

Although the model is quite old and rather simple, it helps build economic understanding because of its focus on expectations.

One aspect of the model is that soy beans cannot be produced instantaneously.

Due to this "production lag", sellers need to forecast the prices they expect to hold when their soy beans are ready to take to the market.

This informational friction can cause complicated dynamics.

Here we investigate and simulate the basic model under different assumptions regarding the way that produces form expectations.

Our discussion and simulations draw on [high quality lectures](https://comp-econ.org/CEF_2013/downloads/Complex%20Econ%20Systems%20Lecture%20II.pdf) by [Cars Hommes](https://www.uva.nl/en/profile/h/o/c.h.hommes/c.h.hommes.html).

We will use the following imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## The Model

Suppose demand for soy beans is given by

$$
    D(p_t) = a - b p_t
$$

where $a, b$ are nonnegative constants and $p_t$ is the spot (i.e, current market) price at time $t$.

($D(p_t)$ is the quantity demanded in some fixed unit, such as thousands of tons.)

Supply of soy beans depends on *expected* prices at time $t$, which we denote $p^e_t$.

We suppose that supply is nonlinear in expected prices, and takes the form

$$
    S(p^e_t) = \tanh(\lambda(p^e_t - c)) + d
$$

where $\lambda$ is a positive constant and $c, d \geq 0$.

Let's make a plot of supply and demand for particular choices of the parameter values.

We will store the parameters in a class and define the functions above as methods.

```{code-cell} ipython3
class Market:

    def __init__(self,
                 a=8,      # demand parameter
                 b=1,      # demand parameter
                 c=6,      # supply parameter
                 d=1,      # supply parameter
                 λ=2.0):   # supply parameter
        self.a, self.b, self.c, self.d = a, b, c, d
        self.λ = λ

    def demand(self, p):
        a, b = self.a, self.b
        return a - b * p

    def supply(self, p):
        c, d, λ = self.c, self.d, self.λ
        return np.tanh(λ * (p - c)) + d
```

Here's the plot.

```{code-cell} ipython3
p_grid = np.linspace(5, 8, 200)
m = Market()
fig, ax = plt.subplots()

ax.plot(p_grid, m.demand(p_grid), label="$D$")
ax.plot(p_grid, m.supply(p_grid), label="S")
ax.legend()

plt.show()
```

Market equilibrium requires that supply equals demand, or

$$
    a - b p_t = S(p^e_t)
$$

Rewriting in terms of $p_t$ gives

$$
    p_t = - \frac{1}{b} [S(p^e_t) - a]
$$

Finally, to complete the model, we need to describe how price expectations are formed.

For now, let's just assume that expected prices at time $t$ depend on past prices.

In particular, we suppose that

```{math}
:label: p_et
    p^e_t = f(p_{t-1}, p_{t-2})
```

where $f$ is some function.

Thus, we are assuming that producers expect the time-$t$ price to be some function of lagged prices, up to $2$ lags.

(We could of course add additional lags and readers are encouraged to experiment with such cases.)

Combining the last two equations gives the dynamics for prices:

```{math}
:label: price_t
    p_t = - \frac{1}{b} [ S(f(p_{t-1}, p_{t-2}))) - a]
```

The price dynamics depend on the parameter values and also on the function $f$ that tells us how producers form expectations.


+++

## Naive Expectations

Naive expectations refers to the case where producers expect the next period spot price to be whatever the price is in the current period.

In other words,

$$ p_t^e = p_{t-1} $$

Using {eq}`price_t`,

$$
    p_t = - \frac{1}{b} [ S(p_{t-1})) - a]
$$

+++

Let's try to simulate the above model.

```{code-cell} ipython3
def find_next_price(m, current_price):
    """
    Function to find the next price given the current price
    and Market model
    """
    next_price = - (m.supply(current_price) - m.a) / m.b
    return next_price
```

```{code-cell} ipython3
m = Market()
```

Let's try to simulate the values of price using some initial value and plot a 45 degree curve to observe the dynamics.

```{code-cell} ipython3
---
tags: [hide-input]
---
def plot45(model, pmin, pmax, p0, num_arrows=5):
    """
    Function to plot a 45 degree plot

    Parameters
    ==========

    model: Market model

    pmin: Lower price limit

    pmax: Upper price limit

    p0: Initial value of price (needed to simulate prices)

    num_arrows: Number of simulations to plot
    """
    pgrid = np.linspace(pmin, pmax, 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(pmin, pmax)
    ax.set_ylim(pmin, pmax)

    hw = (pmax - pmin) * 0.01
    hl = 2 * hw
    arrow_args = dict(fc="k", ec="k", head_width=hw,
            length_includes_head=True, lw=1,
            alpha=0.6, head_length=hl)

    ax.plot(pgrid, find_next_price(model, pgrid), 'b-',
            lw=2, alpha=0.6, label='p')
    ax.plot(pgrid, pgrid, lw=1, alpha=0.7, label='45')

    x = p0
    xticks = [pmin]
    xtick_labels = [pmin]

    for i in range(num_arrows):
        if i == 0:
            ax.arrow(x, 0.0, 0.0, find_next_price(model, x),
                     **arrow_args) # x, y, dx, dy
        else:
            ax.arrow(x, x, 0.0, find_next_price(model, x) - x,
                     **arrow_args)
            ax.plot((x, x), (0, x), ls='dotted')

        ax.arrow(x, find_next_price(model, x),
                 find_next_price(model, x) - x, 0, **arrow_args)
        xticks.append(x)
        xtick_labels.append(r'$p_{}$'.format(str(i)))

        x = find_next_price(model, x)
        xticks.append(x)
        xtick_labels.append(r'$p_{}$'.format(str(i+1)))
        ax.plot((x, x), (0, x), '->', alpha=0.5, color='orange')

    xticks.append(pmax)
    xtick_labels.append(pmax)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(xtick_labels)

    bbox = (0., 1.04, 1., .104)
    legend_args = {'bbox_to_anchor': bbox, 'loc': 'upper right'}

    ax.legend(ncol=2, frameon=False, **legend_args, fontsize=14)
    plt.show()
```

```{code-cell} ipython3
plot45(m, 0, 9, 2, num_arrows=3)
```

The plot shows the function $ p $ and the $45$ degree line.

Think of $ p_t $ as a value on the horizontal axis.

To calculate $ p_{t+1} $, we can use the graph of $ p $ to see its
value on the vertical axis.

Clearly,

- If $ p $ lies above the 45 degree line at this point, then we have $ p_{t+1} > p_t $.
- If $ p $ lies below the 45 degree line at this point, then we have $ p_{t+1} < p_t $.
- If $ p $ hits the 45 degree line at this point, then we have $ p_{t+1} = p_t $, so $ p_t $ is a steady state.

```{code-cell} ipython3
def ts_plot_price(model, p0, ts_length=10):
    """
    Function to simulate and plot the time series of price.

    Parameters
    ==========

    model: Market model

    p0: Initial value of price

    ts_length: Number of iterations
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$p_t$', fontsize=12)
    p = np.empty(ts_length)
    p[0] = p0
    for t in range(1, ts_length):
        p[t] = find_next_price(model, p[t-1])
    ax.plot(np.arange(ts_length),
            p,
            'bo-',
            alpha=0.6,
            lw=2,
            label=r'$p_t$')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(np.arange(ts_length))
    plt.show()
```

```{code-cell} ipython3
ts_plot_price(m, 2, 15)
```

## Adaptive Expectations

Adaptive expectations refers to the case where producers form expectations for the next period price as a weighted average of their last guess and the current spot price.

That is,

```{math}
:label: pe_adaptive
p_t^e = \alpha p_{t-1} + (1-\alpha) p^e_{t-1}
```

Using {eq}`pe_adaptive`,

$$
    p_t = - \frac{1}{b} [ S(\alpha p_{t-1} + (1-\alpha) p^e_{t-1})) - a]
$$

+++

Let's try to simulate the price and observe the dynamics using different values of $\alpha$.

```{code-cell} ipython3
def find_next_price_adaptive(model, curr_price_exp):
    """
    Function to find the next price given the current price expectation
    and Market model
    """
    return - (model.supply(curr_price_exp) - model.a) / model.b
```

```{code-cell} ipython3
def ts_price_plot_adaptive(model, p0, ts_length=10, α=[1.0, 0.9, 0.75]):
    fig, axs = plt.subplots(1, len(α), figsize=(12, 5))
    for i_plot, a in enumerate(α):
        pe_last = p0
        p_values = np.empty(ts_length)
        p_values[0] = p0
        for i in range(1, ts_length):
            p_values[i] = find_next_price_adaptive(model, pe_last)
            pe_last = a*p_values[i] + (1 - a)*pe_last

        axs[i_plot].plot(np.arange(ts_length), p_values)
        axs[i_plot].set_title(r'$\alpha={}$'.format(a))
    plt.show()
```

```{code-cell} ipython3
ts_price_plot_adaptive(m, 5, 30)
```

## Exercises

```{exercise-start}
:label: ex1
```

### Exercise 1

Use the default Market model and Naive expectation to plot the time series simulation of supply function.

```{exercise-end}
```

```{solution-start} ex1
:class: dropdown
```


```{code-cell} ipython3
def ts_plot_supply(model, p0, ts_length=10):
    """
    Function to simulate and plot the supply function
    given the initial price.
    """
    pe_last = p0
    s_values = np.empty(ts_length)
    for i in range(ts_length):
        s_values[i] = model.supply(pe_last)
        pe_last = - (s_values[i] - model.a) / model.b


    fig, ax = plt.subplots()
    ax.plot(np.arange(ts_length),
            s_values,
            'bo-',
            alpha=0.6,
            lw=2,
            label=r'$S_t$')

    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(np.arange(ts_length))
    plt.show()
```

```{code-cell} ipython3
m = Market()
ts_plot_supply(m, 5, 15)
```

```{solution-end}
```

```{exercise-start}
:label: ex2
```

### Exercise 2

#### Backward looking average expectations

Backward looking average expectations refers to the case where producers form expectations for the next period price as a linear combination of their last guess and the second last guess.

That is,

```{math}
:label: pe_adaptive
p_t^e = w_1 p_{t-1} + w_2 p_{t-2}
```


Simulate and plot the price dynamics for $w_1=0.2, w_2=0.01, p_0=1,$ and $p_1=2.5$,

```{exercise-end}
```

```{solution-start} ex2
:class: dropdown
```

```{code-cell} ipython3
def find_next_price_blae(model, curr_price_exp):
    """
    Function to find the next price given the current price expectation
    and Market model
    """
    return - (model.supply(curr_price_exp) - model.a) / model.b
```

```{code-cell} ipython3
def ts_plot_price_blae(model, p0, p1, w1, w2, ts_length=15):
    """
    Function to simulate and plot the time series of price
    using backward looking average expectations.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$p_t$', fontsize=12)
    p = np.empty(ts_length)
    p[0] = p[0]
    p[1] = p[1]
    for t in range(2, ts_length):
        pe = w1*p[i-1] + w2*p[i-2]
        p[t] = -(model.supply(pe) - model.a) / model.b
    ax.plot(np.arange(ts_length),
            p,
            'bo-',
            alpha=0.6,
            lw=2,
            label=r'$p_t$')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(np.arange(ts_length))
    plt.show()
```

```{code-cell} ipython3
m = Market()
ts_plot_price_blae(m, 1, 2.5, 0.2, 0.01, 15)
```

```{solution-end}
```
