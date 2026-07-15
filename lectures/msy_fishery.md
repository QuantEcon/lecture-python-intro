---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Fishery Management: Quotas and MSY

## Overview

In this lecture we study fishery management through quotas and **maximum sustainable yield** (MSY).

The MSY is defined as the largest catch a fishery can support year after year without running itself down.

After the Second World War, fisheries managers around the world adopted MSY and
imposed quotas and restrictions based on MSY.

In some cases these management efforts were successful.

In other cases outcomes were disappointing.

In fact, several major fisheries collapsed under MSY-based management,
including the Peruvian anchovy in the 1970s and the Atlantic cod off
Newfoundland in 1992.

In this lecture we provide an introduction to MSY-based management of fisheries.

Our main task is to explain how MSY is computed.

We begin with a relatively elementary treatment in discrete time.

Next we discuss a continuous time formulation that conveys the same ideas via calculus.

Finally, we turn to problems associated with MSY and quotas derived from MSY.

To illustrate these ideas, we add random shocks to the model and see how collapse can easily occur.

The MSY framework is due to {cite:t}`schaefer1954` and {cite:t}`gordon1954`; a
classic textbook treatment is {cite:t}`clark1990`.

This lecture was partly inspired by [a discussion of MSY](https://maths-from-the-past.org/2025/05/12/oversimplified-models-fishery-collapses-and-msy/) posted on [Maths from the past](https://maths-from-the-past.org).

Let's start with some standard imports.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

We will use the following parameterization throughout.

```{code-cell} ipython3
r = 0.5      # intrinsic growth rate (per year)
K = 1000.0   # carrying capacity (tonnes)
q = 0.01     # catchability coefficient
```

## The model

We build the model in two steps: first how the stock grows on its own, then what
fishing does to it.

### Growth without fishing

In the model, the population adds new fish each year according to the **logistic
growth** function

$$
    G(x) = r\,x\left(1 - \frac{x}{K}\right)
$$ (eq:logistic)

Here 

* $x$ is the **current biomass** --- the total weight of fish in tonnes,
* $G(x)$ is **annual growth** in biomass --- the difference between current and next year's biomass,
* $r > 0$ is called the **intrinsic growth rate**, and
* $K > 0$ is called the **carrying capacity**.

Let's encode the growth function in Python.

```{code-cell} ipython3
def G(x):
    "Logistic growth over one year."
    return r * x * (1 - x / K)
```

As the next figure shows, growth is small when the stock is small (few spawners)
and small again when the stock is near $K$ (crowding, limited food), and is
largest in between.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Logistic growth
    name: fig:logistic-growth
---
x_grid = np.linspace(0, K, 400)

fig, ax = plt.subplots()
ax.plot(x_grid, G(x_grid), lw=2)
ax.set_xlabel('stock biomass  $x$')
ax.set_ylabel('annual growth  $G(x)$')
ax.set_xlim(0, K)
ax.set_ylim(0, G(K / 2) * 1.15)
plt.tight_layout()
plt.show()
```

With no fishing, next year's stock is just this year's stock plus current growth:

$$
    x_{t+1} = x_t + G(x_t).
$$

One way to see where the dynamics lead is via a 45-degree diagram, which plots
next year's stock $x_{t+1}$ against this year's stock $x_t$.

Wherever the curve crosses the $45^\circ$ line we have $x_{t+1} = x_t$.

The corresponding value of $x$ obeys $x = x + G(x)$.

Such an $x$ is called a **steady state**: a stock that exactly reproduces itself.

We can trace the dynamics of the model by "staircasing": from a starting stock go *up* to
the curve (that gives next year's stock), *across* to the $45^\circ$ line (that
becomes this year's stock), and repeat.

We start with some plotting code.

```{code-cell} ipython3
:tags: [hide-input]

def plot_45(
        ax, f, x0, x_max, steady_state, ss_label, map_label, n_years=30
    ):
    "Draw a 45-degree (cobweb) diagram for a function f."
    grid = np.linspace(0, x_max, 400)
    ax.plot(grid, f(grid), lw=2, label=map_label)
    ax.plot(grid, grid, color='0.6', lw=1, ls='--',
            label=r'$45^\circ$ line')
    # cobweb staircase starting from x0
    x = x0
    cx, cy = [x], [0.0]
    for _ in range(n_years):
        y = f(x)
        cx += [x, y]
        cy += [y, y]
        x = y
    ax.plot(cx, cy, color='black', lw=1, alpha=0.9)
    ax.plot([steady_state], [steady_state], 'o', color='black', ms=6, zorder=5)
    ax.annotate(ss_label, xy=(steady_state, steady_state),
                xytext=(0.55 * x_max, 0.28 * x_max), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.set_xlabel('stock this year  $x_t$')
    ax.set_ylabel('stock next year  $x_{t+1}$')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, x_max)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
```

Here is the unfished case.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Stock dynamics without fishing
    name: fig:cobweb-unfished
---
fig, ax = plt.subplots(figsize=(4.95, 4.95))
plot_45(ax, lambda x: x + G(x), x0=80, x_max=1100,
        steady_state=K, ss_label=r'$x=K$ (carrying capacity)',
        map_label=r'$x_{t+1}=x_t+G(x_t)$')
plt.tight_layout()
plt.show()
```

Starting from any positive stock and following the staircase, we see that $x_t$
climbs to the steady state carrying capacity $x = K$.

The unfished population fills up the environment and stays there.

(The empty ocean $x = 0$ is also a steady state, but an unstable one --- any small
stock grows away from it.)


### Adding fishing

Now let a fishing fleet remove some catch quantity $h_t$ each year.

Following {cite:t}`schaefer1954`, the catch is proportional to fishing **effort**
$e$ (e.g. boat-days) and to the stock available to be caught:

$$
    h_t = q\,e\,x_t
$$ (eq:harvest)

Here $q > 0$ is the **catchability coefficient**.

Subtracting the catch, next year's stock becomes

$$
    x_{t+1} = x_t + G(x_t) - qe\,x_t.
$$ (eq:update)


Here's Python code to implement this update rule

```{code-cell} ipython3
def update(x, e):
    "Next year's stock given current stock x and fishing effort e."
    return x + G(x) - q * e * x
```

Here's the 45 degree diagram, now with the fishing term included.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Stock dynamics at a fixed effort
    name: fig:cobweb-fished
---
e_demo = 20.0    # an illustrative fixed effort level

def x_bar(e):
    "Sustainable (steady-state) stock at effort e."
    return K * (1 - q * e / r)

fig, ax = plt.subplots(figsize=(4.95, 4.95))
plot_45(ax, lambda x: update(x, e_demo), x0=80, x_max=1100,
        steady_state=x_bar(e_demo), ss_label=r'$\bar{x}(e)$',
        map_label=r'$x_{t+1}=x_t+G(x_t)-qex_t$')
plt.tight_layout()
plt.show()
```

On the 45-degree diagram, fishing pulls the update curve down by the harvest
term, so it now crosses the $45^\circ$ line at a lower steady state.

Since this steady state is a function of $e$ now, we denote it by $\bar{x}(e)$.

Here's the dynamics for two different levels of $e$, with the staircases omitted.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Stock dynamics at two effort levels
    name: fig:cobweb-two-efforts
---
grid = np.linspace(0, 1100, 400)

fig, ax = plt.subplots(figsize=(4.95, 4.95))
ax.plot(grid, grid, color='0.6', lw=1, ls='--', label=r'$45^\circ$ line')
for e in (10.0, 30.0):
    line, = ax.plot(grid, update(grid, e), lw=2, label=f'$e={e:.0f}$')
    xs = x_bar(e)
    ax.plot([xs], [xs], 'o', color=line.get_color(), ms=8, zorder=5)

ax.set_xlabel('stock this year  $x_t$')
ax.set_ylabel('stock next year  $x_{t+1}$')
ax.set_xlim(0, 1100)
ax.set_ylim(0, 1100)
ax.set_aspect('equal')
ax.legend(loc='upper left', frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
```

Not surprisingly, the steady state $\bar{x}(e)$ is decreasing in fishing effort
(more fishing leads to less fish).


## Sustainable yield

Suppose, as above, that the fleet applies a constant effort $e$ every year.

Let's look for the associated steady state and associated catch.

At any steady state, we must have $x_{t+1} = x_t$.

Looking at {eq}`eq:update`, this happens exactly when growth replaces the catch,
$G(x) = qex$.

Writing it out and ignoring the empty-ocean case $x = 0$, we get

$$
    r\left(1 - \frac{x}{K}\right) = qe
    \quad\Longrightarrow\quad
    \bar{x}(e) = K\left(1 - \frac{qe}{r}\right).
$$ (eq:xstar)


Provided $e < r/q$, we see that each effort level $e$ pins down one steady-state stock $\bar{x}(e)$.

The catch this steady state delivers is called the **sustainable yield**, and defined as

$$
    \bar{y}(e) := q\,e\,\bar{x}(e).
$$ (eq:yield)

```{code-cell} ipython3
def sustainable_yield(e):
    "Sustainable catch delivered each year at effort e."
    return q * e * x_bar(e)
```

To visualize the sustainable yield, we plot the growth curve $G(x)$ together
with the harvest line $q e x$ for a single effort level $e$.

The two cross where growth exactly replaces the catch: that crossing sits at the
steady-state stock $\bar{x}(e)$.

The height of the line there is the sustainable catch $\bar{y}(e)$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Sustainable stock and catch
    name: fig:sustainable-point
---
x = np.linspace(0, K, 400)

fig, ax = plt.subplots()
ax.plot(x, G(x), lw=2, label=r'growth  $G(x)$')
ax.plot(x, q * e_demo * x, lw=2, label=r'harvest  $q e x$')

xs = x_bar(e_demo)
ys = sustainable_yield(e_demo)
ax.plot([xs], [ys], 'o', color='black', ms=8, zorder=5)
ax.vlines(xs, 0, ys, ls='--', color='black', lw=1)
ax.hlines(ys, 0, xs, ls='--', color='black', lw=1)
ax.annotate(r'$\bar{x}(e)$', xy=(xs, 0), xytext=(xs + 12, 8), fontsize=12)
ax.annotate(r'$\bar{y}(e)$', xy=(0, ys), xytext=(10, ys + 5), fontsize=12)

ax.set_xlabel('stock biomass  $x$')
ax.set_ylabel('catch')
ax.set_xlim(0, K)
ax.set_ylim(0, G(K / 2) * 1.4)
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
```

Different effort levels give different sustainable catches $\bar{y}(e)$.

In the next figure we plot the growth curve $G(x)$ together with the harvest
lines $q e x$ for several values of $e$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Steady states at several effort levels
    name: fig:steady-states
---
fig, ax = plt.subplots()
ax.plot(x, G(x), lw=2, label='growth  $G(x)$')

efforts = [12.5, 25.0, 37.5]
labels  = [r'low $e$', r'moderate $e$', r'high $e$']

for e, lab in zip(efforts, labels):
    line, = ax.plot(x, q * e * x, lw=2, label=lab)
    ax.plot([x_bar(e)], [q * e * x_bar(e)], 'o', color=line.get_color(), ms=7, zorder=5)

ax.set_xlabel('stock $x$')
ax.set_ylabel('catch')
ax.set_xlim(0, K)
ax.set_ylim(0, G(K / 2) * 1.4)
ax.legend(loc='upper left', frameon=False, fontsize=10)
plt.tight_layout()
plt.show()
```

Each dot marks a steady state, and its height is the sustainable catch at that
effort.

As effort rises, the steady-state stock falls, while the catch rises and then
falls --- exactly the trade-off that the maximum sustainable yield captures.


### The yield-effort curve

To visualize the MSY, we plot the sustainable catch $\bar{y}(e)$ against effort $e$.

This gives the classic dome-shaped Schaefer curve that fisheries managers use.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Yield-effort curve
    name: fig:yield-effort
---
e_grid = np.linspace(0, r / q, 400)
y_grid = sustainable_yield(e_grid)

fig, ax = plt.subplots()
ax.plot(e_grid, y_grid, lw=2, label=r'$\bar{y}(e)=qeK\,(1-qe/r)$')
ax.set_xlabel('fishing effort  $e$')
ax.set_ylabel(r'sustainable yield  $\bar{y}(e)$')
ax.set_xlim(0, r / q)
ax.set_ylim(0, y_grid.max() * 1.25)
ax.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.show()
```

The sustainable yield rises with effort and then falls.

The maximum height of this curve is called the **maximum sustainable yield** (MSY).

It is the largest steady state catch attainable, assuming a constant effort rate $e$.


### Calculating the maximum sustainable yield

The maximum sustainable yield can be defined more mathematically as

$$
    \text{MSY} \;=\; \max_{0 \le e < r/q} \; \bar{y}(e).
$$

The effort level that solves this maximization problem is represented by $e^*$.

We can compute it either by calculus or numerically, by evaluating $\bar{y}(e)$ on a fine grid of effort
levels and picking the largest.

Below we look at the solution using calculus.

For now we will use the numerical approach:

```{code-cell} ipython3
e_search = np.linspace(0, r / q, 100_001)
i = np.argmax(sustainable_yield(e_search))

e_star = e_search[i]
MSY = sustainable_yield(e_star)
x_bar_star = x_bar(e_star)

print(f"effort at MSY     e*    = {e_star:.2f}")
print(f"stock at MSY      x̄     = {x_bar_star:.1f} tonnes      (= K/2)")
print(f"maximum yield     MSY   = {MSY:.1f} tonnes/year  (= rK/4)")
```

The search lands on the round numbers $\bar{x}(e^*) = K/2$ and $\text{MSY} = rK/4$; the
optional calculus section below shows why.


### Dynamics

What happens to biomass when we set $e = e^*$ from some arbitrary initial
$x_0$?

Below we run the yearly recursion {eq}`eq:update` forward from several
starting stocks, under the MSY effort.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Stock paths under MSY effort
    name: fig:msy-convergence
---
def simulate(x0, e, years=40):
    "Run the deterministic yearly stock recursion forward from x0."
    x = np.empty(years + 1)
    x[0] = x0
    for t in range(years):
        x[t + 1] = update(x[t], e)
    return np.arange(years + 1), x

fig, ax = plt.subplots()
for x0 in [100, 300, 700, 950]:
    t, xt = simulate(x0, e_star)
    ax.plot(t, xt, 'o-', ms=3, lw=2, label=f'$x_0={x0}$')

ax.axhline(x_bar_star, ls='--', color='black', lw=1)
ax.annotate(r'$\bar{x}(e^*)=K/2$', xy=(0, x_bar_star), xytext=(1, x_bar_star + 25), color='black')
ax.set_xlabel('year  $t$')
ax.set_ylabel('stock biomass  $x_t$')
ax.set_xlim(0, 40)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
```

We see that, under the MSY effort, every path climbs or falls toward $\bar{x}(e^*) = K/2$ and stays.

As a result, the catch settles down to the MSY.

```{note}
The smooth, steady convergence above relies on $r$ being modest.

For large $r$ the yearly model can overshoot and oscillate --- even behave
chaotically {cite}`may1976` --- but that is an artifact of the one-year time step,
not the biology.

The continuous-time version in the next section converges smoothly for *any*
$r > 0$, which is one reason it is the textbook standard.
```

## The same result with calculus


The following section is optional.

Our discussion above used only algebra and figures. 

For readers who are familiar with calculus, we now provide a continuous time version of the model.

The continuous time version is frequently used and has some advantages (and some disadvantages).

If we shrink the time step to zero, the recursion {eq}`eq:update` becomes the
differential equation

$$
    \dot{x} = G(x) - qex.
$$

The sustainability condition is unchanged --- a steady state still needs
$G(x) = qex$ --- so $\bar{x}(e) = K(1 - qe/r)$ and $\bar{y}(e) = qeK(1 - qe/r)$ exactly as
before.

Now $\bar{y}$ is a smooth, concave parabola in $e$, and its peak is where the slope
vanishes:

$$
    \frac{d \bar{y}}{d e} = qK\left(1 - \frac{2qe}{r}\right) = 0
    \quad\Longrightarrow\quad
    e^* = \frac{r}{2q},
$$

which gives $\bar{x}(e^*) = K/2$ and $\text{MSY} = rK/4$.


```{note}
Readers who have seen the Solow-Swan growth model may recognize the structure.

A single control --- here fishing effort, there the savings rate --- picks one
point on a one-parameter family of steady states.

The sustainable flow --- here the yield, there consumption --- is hump-shaped in
that control, and the optimum sits at an interior peak.

Over-fishing is then the exact analogue of over-saving (dynamic inefficiency):
more of the control, less of the flow.
```

## A success story: Pacific Coast lingcod

Before turning to the risks, it is worth seeing the MSY framework succeed.

A clean example is the lingcod (*Ophiodon elongatus*) fishery off the U.S.
Pacific Coast.

Lingcod is managed using MSY-based analysis.

This analysis leads to a target biomass and a target fishing pressure that
together define the maximum sustainable yield.

To follow the fishery we use two ratios.

The first is $B / B_{MSY}$, the stock biomass relative to the biomass $B_{MSY}$
that supports the MSY.

(In our model this reference stock is $\bar{x}(e^*) = K/2$.)

The second is $F / F_{MSY}$, fishing pressure relative to the pressure
$F_{MSY}$ that achieves the MSY.

(In our model this is the MSY effort $e^*$.)

The data come from the RAM Legacy Stock Assessment Database {cite:t}`ricard2012`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Pacific Coast lingcod recovery
    name: fig:lingcod
---
data_url = "https://github.com/QuantEcon/data-lectures/raw/main/lectures/lingcod_msy_recovery.csv"
lingcod = pd.read_csv(data_url)

fig, ax = plt.subplots()

ax.axhline(1.0, color='black', lw=1, ls='--')
ax.text(1930, 1.05, r'MSY reference ($=1$)', fontsize=9, va='bottom')

ymax = max(lingcod['B_over_Bmsy'].max(), lingcod['F_over_Fmsy'].max()) * 1.06

# shade the years when fishing pressure exceeded the MSY level
over = lingcod['F_over_Fmsy'] > 1
ax.fill_between(lingcod['year'], 0, ymax, where=over,
                color='0.5', alpha=0.12)

ax.plot(lingcod['year'], lingcod['B_over_Bmsy'], lw=2,
        label=r'$B / B_{MSY}$  (biomass)')
ax.plot(lingcod['year'], lingcod['F_over_Fmsy'], lw=2,
        label=r'$F / F_{MSY}$  (fishing pressure)')

ax.axvline(1999, color='grey', lw=0.8, ls=':')
ax.axvline(2005, color='grey', lw=0.8, ls=':')

ax.set_xlabel('year')
ax.set_ylabel('ratio to MSY reference point')
ax.set_xlim(lingcod['year'].min(), lingcod['year'].max())
ax.set_ylim(0, ymax)
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
```


Through the 1960s the stock was lightly exploited: $F$ sat well below $F_{MSY}$
and biomass drifted down slowly from a high level.

From the early 1970s fishing pressure climbed above $F_{MSY}$ and stayed there for
nearly three decades.

Biomass fell steadily, reaching a trough of about $0.27\,B_{MSY}$ in 1993 ---
close to collapse.

In 1999 the stock was formally declared *overfished*, and a rebuilding plan
based around MSY cut catches sharply.

Fishing pressure dropped well below $F_{MSY}$, and biomass climbed back through
$B_{MSY}$ by 2004.

The stock was declared fully rebuilt in 2005, ahead of schedule, and then
overshot its target.

As suggested by the model, fishing above the MSY level has led to falling stock, while
keeping it below has led to recovery.


Of course, the real world is not as clean as the model and random factors
also influenced outcomes.

For example, a strong 1999 year-class also helped, so the
cut in fishing was important but recruitment luck set the speed of recovery.

In addition, the recovery was aided by coast-wide measures aimed at the whole
groundfish community.

The next section expands our model somewhat, with the aim of introducing more
realistic dynamics.


## Randomness, risk and collapse

As mentioned in the introduction to this lecture, MSY-based policies have not
always been successful in sustaining fish populations.

A major reason is that the simple MSY model ignores randomness and risk.

This randomness and risk is associated with several features of fisheries.

One feature is that fish stocks are difficult to track, so policies may be based
on incorrect measurements.

Another is that ocean environments are complex and nonstationary: a policy that looks safe in a deterministic model
can be dangerously fragile once we admit randomness.

In this section we investigate how one form of randomness --- stochastic growth
--- can impact fishery dynamics under MSY-based management.

### Adding randomness

So far the ocean has been a perfectly predictable place.

Each year the stock grows by exactly $G(x_t)$.

Real fish populations are not like this.

Recruitment depends on water temperature, currents, food supply, predators and
disease --- all of which vary from year to year, sometimes wildly.

Let's add this randomness to the model.

A simple way is to multiply each year's growth by a random **environmental shock**
$\xi_t$ with mean one:

$$
    x_{t+1} = x_t + \xi_t \, G(x_t) - h_t,
$$ (eq:stochastic)

where $h_t$ is the catch taken in year $t$.

We take $\xi_t$ to be lognormal with mean one, controlled by a volatility
parameter $\sigma$.

```{code-cell} ipython3
def shocks(σ, n, rng):
    "Return n lognormal shocks with mean one and volatility σ."
    return np.exp(σ * rng.standard_normal(n) - σ**2 / 2)
```

Now we face a management choice that did not matter in the deterministic world but
matters enormously here.

How should we set the catch $h_t$?

We compare two policies, both designed to take the MSY *on average*:

* **constant effort**: set $h_t = q\, e^*\, x_t$, so the catch scales with the
  current stock, and
* **constant quota**: set $h_t = \text{MSY}$ every year, regardless of the stock.

The constant quota is the more literal reading of "take the maximum sustainable
yield each year".

It is also closer to how catch limits have often been set in practice.

Let's simulate both, starting every fishery from the MSY stock $\bar{x}(e^*) = K/2$.

```{code-cell} ipython3
def simulate_stochastic(policy, σ, years, rng, x0=x_bar_star):
    "Simulate the stochastic fishery under a given harvest policy."
    x = np.empty(years + 1)
    x[0] = x0
    ξ = shocks(σ, years, rng)
    for t in range(years):
        growth = ξ[t] * G(x[t])
        if policy == 'effort':
            harvest = q * e_star * x[t]
        else:                       # constant quota
            harvest = MSY
        x[t + 1] = max(0.0, x[t] + growth - harvest)
    return x
```

Here are a few sample paths under each policy, with a moderate amount of
environmental noise.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 'Two MSY policies under noise: constant effort (top) and constant quota
      (bottom)'
    name: fig:msy-policies
---
σ = 0.15
years = 100
rng = np.random.default_rng(seed=1)

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharey=True)

for ax, policy, label in zip(
        axes, ['effort', 'quota'],
        ['constant effort  $h_t = q e^* x_t$',
         'constant quota  $h_t = \\mathrm{MSY}$']):
    for k in range(8):
        path = simulate_stochastic(policy, σ, years, rng)
        ax.plot(path, lw=2, alpha=0.8)
    ax.axhline(x_bar_star, ls='--', color='black', lw=1)
    ax.text(0.5, 0.92, label, transform=ax.transAxes, ha='center', va='top')
    ax.set_xlabel('year  $t$')
    ax.set_ylabel('stock biomass  $x_t$')
    ax.set_xlim(0, years)
    ax.set_ylim(0, K)

plt.tight_layout()
plt.show()
```

The difference is stark.

Under **constant effort**, the paths jiggle around $\bar{x}(e^*) = K/2$ but always recover:
a bad year cuts the catch (because the catch is proportional to the stock), giving
the population room to bounce back.

The policy is **self-correcting**.

Under **constant quota**, several paths slide down and hit zero --- the fishery
**collapses**.

The quota keeps demanding the full MSY even after a run of bad years has thinned
the stock, and eventually the catch exceeds what the depleted population can
replace.


### Why the quota is a knife-edge

The collapse is not bad luck --- it is built into the constant-quota policy.

Set the catch to the constant $h = \text{MSY} = rK/4$ in the deterministic model
and look for steady states, $G(x) = rK/4$:

$$
    r\,x\left(1 - \frac{x}{K}\right) = \frac{rK}{4}
    \quad\Longrightarrow\quad
    \left(x - \frac{K}{2}\right)^2 = 0.
$$

The two steady states have merged into a single **tangent** point at $x = K/2$.

This point is *semi-stable*: stable from above but unstable from below.

If the stock ever drifts below $K/2$, then $G(x) < rK/4$, so the constant quota
takes more than the stock can grow, and the stock falls further --- a one-way
ratchet down to zero.

Fishing exactly at the MSY with a fixed quota leaves **no margin for error**.

The deterministic model hides this fragility because nothing ever pushes the stock
off its perch.

Add even a little randomness and the knife-edge reveals itself.

### How often does the fishery collapse?

Let's quantify the risk by simulating many fisheries under each policy and
counting how often the stock collapses within 100 years.

```{code-cell} ipython3
def collapse_fraction(policy, σ, years=100, n_paths=2000, seed=0):
    "Fraction of simulated fisheries that collapse within the horizon."
    rng = np.random.default_rng(seed)
    collapses = 0
    for _ in range(n_paths):
        path = simulate_stochastic(policy, σ, years, rng)
        if path[-1] < 1.0:
            collapses += 1
    return collapses / n_paths

for policy in ['effort', 'quota']:
    frac = collapse_fraction(policy, σ=0.15)
    print(f"{policy:>8s} policy:  collapse within 100 years = {frac:.1%}")
```

With this level of noise the constant-effort fishery essentially never collapses,
while the constant-quota fishery collapses much of the time.

The two policies deliver the same average catch in calm conditions, yet one is
robust and the other is fragile.

This is the heart of the matter: **the MSY is a deterministic, steady-state
concept, and steering straight at it ignores the risk that randomness creates.**

## Historical collapses

These are not just theoretical worries.

The **Peruvian anchovy** fishery was the largest in the world in the early 1970s.

Management was guided by sustainable-yield calculations, but those calculations
left out the **El Niño** warming events that periodically disrupt the cold,
nutrient-rich currents the anchovy depend on.

When a strong El Niño struck in 1972--73, the stock --- already fished hard ---
collapsed, taking tens of thousands of jobs with it.

The **Atlantic cod** off Newfoundland tells a similar story.

Catch limits set with too little margin, combined with overoptimistic stock
estimates, allowed fishing to continue as the population fell.

In 1992 the stock had dropped by more than 90% and Canada closed the fishery
entirely; it has still not fully recovered.

In both cases the deterministic logic of "take the maximum sustainable yield"
proved dangerously fragile once the real, noisy ocean was taken into account.

The lesson modern fisheries science has drawn is to manage **precautionarily** ---
to aim below the MSY, leaving a buffer against the bad years that the simple model
pretends do not exist.

## Exercises

```{exercise}
:label: msy_ex1

The constant-quota policy above demanded the *full* MSY every year, and we saw
that this is a knife-edge.

A natural fix is to be **precautionary**: take a fixed quota that is only a
fraction $\alpha \in (0, 1]$ of the MSY.

For each $\alpha$ in a grid from $0.5$ to $1.0$, estimate the probability that the
fishery collapses within 100 years under the constant quota $h = \alpha \cdot
\text{MSY}$, using $\sigma = 0.15$.

Plot the collapse probability against $\alpha$.

How much does pulling the quota back below the MSY reduce the risk?
```

```{solution-start} msy_ex1
:class: dropdown
```

We adapt `simulate_stochastic` to take an arbitrary fixed quota.

```{code-cell} ipython3
def collapse_fraction_quota(quota, σ=0.15, years=100, n_paths=2000, seed=0):
    "Collapse probability under a fixed quota."
    rng = np.random.default_rng(seed)
    collapses = 0
    for _ in range(n_paths):
        x = x_bar_star
        ξ = shocks(σ, years, rng)
        for t in range(years):
            x = max(0.0, x + ξ[t] * G(x) - quota)
            if x <= 0.0:
                break
        if x < 1.0:
            collapses += 1
    return collapses / n_paths

alphas = np.linspace(0.5, 1.0, 11)
probs = [collapse_fraction_quota(α * MSY) for α in alphas]

fig, ax = plt.subplots()
ax.plot(alphas, probs, 'o-', lw=2)
ax.set_xlabel(r'quota as a fraction $\alpha$ of MSY')
ax.set_ylabel('collapse probability within 100 years')
plt.tight_layout()
plt.show()
```

The collapse probability falls steeply as the quota is pulled back below the MSY.

A modest precautionary margin --- taking, say, 80% of the MSY --- buys a large
reduction in the risk of collapse, at the cost of only a small reduction in the
average catch.

This is exactly the trade-off that precautionary fisheries management is built
around.

```{solution-end}
```

```{exercise}
:label: msy_ex2

The note after the convergence plot warned that for large $r$ the *yearly*
recursion can misbehave, even though the underlying biology is benign.

Investigate this for the unfished model $x_{t+1} = x_t + r x_t(1 - x_t/K)$.

Rescale by writing $u_t = x_t / K$, so that

$$
    u_{t+1} = u_t + r\,u_t(1 - u_t).
$$

Simulate this map from $u_0 = 0.3$ for $r = 1.5$, $r = 2.2$ and $r = 2.7$, and plot
the three paths.

Describe what happens as $r$ increases.
```

```{solution-start} msy_ex2
:class: dropdown
```

```{code-cell} ipython3
def logistic_path(r_val, u0=0.3, years=40):
    u = np.empty(years + 1)
    u[0] = u0
    for t in range(years):
        u[t + 1] = u[t] + r_val * u[t] * (1 - u[t])
    return u

fig, ax = plt.subplots()
for r_val in [1.5, 2.2, 2.7]:
    ax.plot(logistic_path(r_val), 'o-', ms=3, lw=2, label=f'$r={r_val}$')
ax.axhline(1.0, ls='--', color='black', lw=1)
ax.set_xlabel('year  $t$')
ax.set_ylabel(r'scaled stock  $u_t = x_t / K$')
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
```

For $r = 1.5$ the path converges smoothly to the carrying capacity $u = 1$.

For $r = 2.2$ it overshoots and settles into a steady oscillation (a two-year
cycle).

For $r = 2.7$ the path never settles --- it bounces around erratically, the onset
of the chaotic behavior analyzed by {cite:t}`may1976`.

The biology is unchanged; it is the large discrete time step that manufactures
these wild dynamics, which is why the smooth continuous-time version is the
textbook standard.

```{solution-end}
```

## Summary

The key formulas of the deterministic Schaefer model are collected below.

| Quantity | Formula | Value |
|---|---|---|
| Stock at MSY | $\bar{x}(e^*) = K/2$ | 500 t |
| Maximum sustainable yield | $\text{MSY} = rK/4$ | 125 t/yr |
| Effort at MSY | $e^* = r/2q$ | 25 |

The MSY is a deterministic, single-species, steady-state concept.

It ignores economic costs and prices, age and size structure, environmental
randomness, and interactions with other species.

As we saw, the omission of randomness is not a minor technicality.

A constant quota at the MSY sits on a knife-edge, and in a noisy ocean it leads
the fishery to collapse.

Real fisheries management therefore treats the MSY as a reference point --- often
a cautionary upper bound --- rather than a precise target.
