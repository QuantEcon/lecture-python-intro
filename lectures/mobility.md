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

(mobility)=
# Measuring Mobility

```{index} single: Mobility
```

In addition to what's in Anaconda, this lecture will need the following library:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

## Overview

In {doc}`inequality` we measured how unequally income and wealth are distributed at a point in time.

Such measures are snapshots.

They tell us how far apart the rich and the poor are, but nothing about whether the same families stay rich and poor.

Two economies can have identical Lorenz curves and identical Gini coefficients while offering their citizens completely different life prospects.

In one, position is fixed at birth and never changes.

In the other, families rise and fall constantly, and today's poor household has a good chance of being tomorrow's rich one.

The difference between these two economies is **mobility**: the rate at which households change position within the distribution.

Mobility matters for policy.

Attitudes to redistribution, the case for taxing wealth rather than capital income, and our sense of how much opportunity an economy offers all depend on it.

In this lecture we study how to measure mobility when the data take the form of a transition matrix over wealth quantiles.

This is a natural application of the Markov chain theory developed in {doc}`markov_chains_I` and {doc}`markov_chains_II`, and it gives us a second use for the {doc}`Perron-Frobenius theorem <eigen_II>`.

```{note}
This lecture draws heavily on Sections 2 and 3 of the working paper "Mobility" by Daniel Carroll, Nicholas Hoffman and Eric R. Young {cite}`carroll2026mobility`.

That paper collects the standard mobility measures in one place, applies them to US wealth data, and then asks whether workhorse macroeconomic models can reproduce what it finds.

We use their measures and their estimated transition matrices, with thanks.

The broader literature on measuring mobility is surveyed in {cite}`fields1999measurement`.
```

Let's start with some imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantecon as qe

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

## Mobility matrices

### From quantiles to a stochastic matrix

Suppose we observe the wealth of a large number of households at two dates, $s$ and $s + t$.

We sort households by wealth at each date and divide them into $N$ equally sized groups, or **quantiles**.

With $N = 5$ these are quintiles, each containing 20% of households.

Now we ask, for each household, which quantile it started in and which quantile it ended in.

Averaging over households gives us a matrix $M$ with typical element

$$
    m_{ij}
    = \mathbb P \{ \text{household is in quantile } j \text{ at } s+t
                   \mid \text{it was in quantile } i \text{ at } s \}
$$

Each row of $M$ is a probability mass function, so $M$ is a stochastic matrix in the sense of {doc}`markov_chains_I`.

We call $M$ a **mobility matrix**.

Two points deserve emphasis.

First, $M$ describes *relative* mobility: it records movement of households relative to one another, not growth in wealth as such.

An economy in which everyone's wealth doubles has no mobility at all by this definition.

Second, the time unit of the chain is the horizon $t$, which might be five years or twenty.

Everything below depends on that choice, and we return to it when we look at data.

### An example

Here is a mobility matrix estimated from US data over the five years from 1984 to 1989, which we discuss properly in {ref}`a later section <mobility_data>`.

```{code-cell} ipython3
M_ex = [[0.70, 0.23, 0.05, 0.02, 0.00],
        [0.25, 0.45, 0.22, 0.06, 0.02],
        [0.06, 0.24, 0.44, 0.19, 0.06],
        [0.02, 0.06, 0.22, 0.47, 0.23],
        [0.01, 0.01, 0.06, 0.22, 0.70]]
```

These figures are published rounded to two decimal places, so the rows do not quite sum to one.

We will need them to, so let's write a helper that rescales each row.

```{code-cell} ipython3
def normalize_rows(M):
    M = np.asarray(M, dtype=float)
    return M / M.sum(axis=1, keepdims=True)

M_ex = normalize_rows(M_ex)
```

Row 1 says that a household in the poorest quintile in 1984 had a 70% chance of still being there in 1989, a 23% chance of moving up one quintile, and essentially no chance of reaching the top.

The mass concentrates near the diagonal, and the two extreme quintiles are the stickiest.

To make this concrete, let's simulate the quintile histories of a few households.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Simulated quintile paths for four households
    name: fig:mobility-paths
tags: [hide-input]
---
mc = qe.MarkovChain(M_ex)
periods = 10
styles = ('-', '--', '-.', ':')

fig, ax = plt.subplots()
for i, ls in enumerate(styles):
    X = mc.simulate(periods, init=i, random_state=10 + i)
    ax.step(range(periods), X + 1, where='post', lw=2, ls=ls,
            label=f'household {i + 1}')
ax.set_xlabel('period (five years)')
ax.set_ylabel('wealth quintile')
ax.set_yticks(range(1, 6))
ax.set_ylim(0.5, 6.6)
ax.legend(frameon=False, ncol=2, loc='upper center')
plt.show()
```

Some households wander a long way and others hardly move.

Our task is to summarize this behavior in a single number.

(mobility_benchmarks)=
### Two benchmarks

Before choosing a measure, it helps to fix the two extreme cases against which any measure should be calibrated.

**Complete immobility** is the identity matrix $M = I$.

Every household stays where it starts, forever.

**Perfect mobility** is the matrix with every entry equal to $1/N$,

$$
    M^* = \frac{1}{N} \mathbb 1 \mathbb 1^\top
$$

where $\mathbb 1$ is an $N \times 1$ vector of ones.

Here the ending quantile is independent of the starting quantile, so knowing where a household began tells us nothing about where it ends up.

This property is called **origin independence**, and it is the natural upper reference point for mobility: the distribution is reshuffled completely at every step.

```{code-cell} ipython3
N = 5
M_immobile = np.identity(N)
M_perfect = np.ones((N, N)) / N
```

A good measure of mobility should return 0 at $M_{\text{immobile}}$ and 1 at $M_{\text{perfect}}$.

We will see that all four measures below are constructed to do exactly this.

```{note}
Values above 1 are possible and meaningful.

They arise when a chain reverses ranks *systematically* --- for example a matrix that sends the poorest quintile to the richest with probability one moves households around more than pure chance does.

So 1 marks origin independence, not a maximum.
```

## Four measures of mobility

Reducing an $N \times N$ matrix to a single number throws away information.

Different measures throw away different information, which is why it is standard practice to report several.

We follow {cite}`carroll2026mobility` in considering four, each discussed at length in {cite}`dardanoni1993measuring`.

(mobility_shorrocks)=
### The Shorrocks index

The measure of {cite}`shorrocks1978measurement` looks only at the diagonal of $M$,

```{math}
:label: shorrocks_def

\mu_S(M) = \frac{N - \mathrm{trace}(M)}{N - 1}
```

The idea is that $m_{ii}$ is the probability a household in quantile $i$ stays put, so $1 - m_{ii}$ is its probability of escaping.

Rewriting {eq}`shorrocks_def` as

$$
    \mu_S(M)
    = \frac{N}{N-1} \cdot \frac{1}{N} \sum_{i=1}^N (1 - m_{ii})
$$

shows that $\mu_S$ is the average escape probability, divided by the escape probability $(N-1)/N$ under perfect mobility.

In this sense $\mu_S$ measures the stickiness of initial conditions relative to origin independence.

```{code-cell} ipython3
def shorrocks(M):
    N = len(M)
    return (N - np.trace(M)) / (N - 1)
```

```{code-cell} ipython3
shorrocks(M_immobile), shorrocks(M_perfect)
```

The measure is completely blind to how probability is arranged *off* the diagonal.

An economy in which households move from rags to riches in one step and an economy in which the poor become only slightly less poor receive the same score, provided they leave their starting quantile equally often.

(mobility_bartholomew)=
### Bartholomew's measure

The measure of {cite}`bartholomew1967stochastic` takes the opposite view and looks only at the off-diagonal elements,

```{math}
:label: bartholomew_def

\mu_B(M) = \frac{1}{N-1} \sum_{i=1}^N \sum_{j=1}^N m_{ij} \, |i - j|
```

The weight $|i-j|$ is the number of quantile boundaries crossed, so transitions that cross several quantiles count for more than transitions to a neighbor.

Up to the factor $1/(N-1)$, this is the expected number of quantiles a household crosses per period.

{cite}`fields1999measurement` describe $\mu_B$ as a measure of total *movement*.

```{code-cell} ipython3
def bartholomew(M):
    N = len(M)
    i, j = np.indices((N, N))
    return np.sum(M * np.abs(i - j)) / (N - 1)
```

```{code-cell} ipython3
bartholomew(M_immobile), bartholomew(M_perfect)
```

Notice that $\mu_B$ is not calibrated the same way as $\mu_S$: perfect mobility gives 2 rather than 1.

In fact, a short calculation using $\sum_{i,j} |i-j| = N(N-1)(N+1)/3$ gives

$$
    \mu_B(M^*)
    = \frac{1}{N-1} \cdot \frac{1}{N} \cdot \frac{N(N-1)(N+1)}{3}
    = \frac{N+1}{3}
$$

So to put Bartholomew's measure on the same footing as the others we rescale,

```{math}
:label: bartholomew_norm

\tilde \mu_B(M) = \frac{3}{N+1} \, \mu_B(M)
```

which is the expected number of quantiles crossed, relative to the number crossed under origin independence.

```{code-cell} ipython3
def bartholomew_normalized(M):
    N = len(M)
    return 3 * bartholomew(M) / (N + 1)

bartholomew_normalized(M_immobile), bartholomew_normalized(M_perfect)
```

We report $\mu_B$ when we want the interpretation "quantiles crossed per period" and $\tilde \mu_B$ when comparing across measures.

(mobility_eigenvalue)=
### The second eigenvalue

Our third measure comes from the theory of convergence rather than from counting transitions.

Recall from {doc}`markov_chains_I` that the distribution $\psi_t = \psi_0 M^t$ converges to the stationary distribution $\psi^*$ when $M$ is everywhere positive.

The {doc}`Perron-Frobenius theorem <eigen_II>` tells us that the largest eigenvalue of a stochastic matrix is $\lambda_1 = 1$, and that the *rate* at which $\psi_t \to \psi^*$ is governed by the modulus of the second largest eigenvalue $\lambda_2$.

A chain that mixes quickly forgets its initial condition quickly, which is exactly what we mean by mobility.

This suggests

```{math}
:label: eigen_def

\mu_{2E}(M) = 1 - |\lambda_2(M)|
```

{cite}`sommers1979eigenvalue` show that $\mu_{2E}$ measures the total deviation of $M$ from a matrix of perfect mobility.

```{code-cell} ipython3
def second_eigenvalue(M):
    λ = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
    return 1 - λ[1]
```

```{code-cell} ipython3
second_eigenvalue(M_immobile), second_eigenvalue(M_perfect)
```

The identity matrix has $\lambda_2 = 1$ and never mixes, while $M^*$ has $\lambda_2 = 0$ and mixes in a single step.

This measure has a further attraction: for a two-state chain, $|\lambda_2|$ is exactly the autocorrelation of the process, as you are asked to verify in {ref}`an exercise <mob_ex2>` below.

(mobility_mfp)=
### Mean first passage time

The final measure asks a question about waiting times: how long does it take a household to reach a given quantile?

Let $T$ be the matrix whose $(i,j)$-th element is the expected number of periods until a household starting in quantile $i$ first arrives in quantile $j$.

For $i \neq j$ we can compute $T_{ij}$ by conditioning on the first step, exactly as we computed expected unemployment durations in {doc}`markov_chains_I`.

Either the chain jumps straight to $j$, or it moves to some $k \neq j$ and we start again, so

```{math}
:label: mfp_recursion

T_{ij} = 1 + \sum_{k \neq j} m_{ik} T_{kj}
```

Holding $j$ fixed, this is a linear system in the $N-1$ unknowns $\{T_{ij}\}_{i \neq j}$ that we can solve directly.

On the diagonal we use the mean *return* time, which for an irreducible chain is $T_{jj} = 1/\psi^*(j)$.

```{code-cell} ipython3
def mean_first_passage(M):
    """
    Mean first passage matrix T, where T[i, j] is the expected number of
    periods to reach quantile j starting from quantile i.

    The diagonal holds mean return times.

    """
    N = len(M)
    ψ_star = qe.MarkovChain(M).stationary_distributions[0]
    T = np.zeros((N, N))
    for j in range(N):
        idx = [i for i in range(N) if i != j]
        A = np.identity(N - 1) - M[np.ix_(idx, idx)]
        T[idx, j] = np.linalg.solve(A, np.ones(N - 1))
        T[j, j] = 1 / ψ_star[j]
    return T
```

```{code-cell} ipython3
np.round(mean_first_passage(M_ex), 1)
```

Reading the top right entry, a household starting in the poorest quintile waits about 23 periods before first reaching the richest quintile.

Since a period here is five years, that is well over a century.

To get a single number, {cite}`conlisk1990monotone` proposes averaging $T$ over a randomly drawn pair of households.

Since quantiles contain equal numbers of households, the relevant weights are $\psi = (1/N, \ldots, 1/N)$, and

$$
    \text{MFP}(M) = \psi^\top T \psi
$$

is the expected number of periods before one household reaches the quantile of another, both drawn at random.

Under perfect mobility $\text{MFP}(M^*) = N$, so the normalized measure is

```{math}
:label: mfp_def

\mu_{MFP}(M) = \frac{N}{\text{MFP}(M)}
```

which carries units of quantiles per period.

```{code-cell} ipython3
def mfp_measure(M):
    N = len(M)
    ψ = np.ones(N) / N
    return N / (ψ @ mean_first_passage(M) @ ψ)
```

```{code-cell} ipython3
mfp_measure(M_perfect)
```

Unlike the other three, this measure requires irreducibility.

If some quantile cannot be reached from another then the expected waiting time is infinite and $\mu_{MFP} = 0$, regardless of how much movement occurs elsewhere in the matrix.

```{note}
{cite}`meyer1978alternative` gives a closed-form expression for $T$ in terms of a partitioned inverse, which is what {cite}`carroll2026mobility` use.

The first-step argument in {eq}`mfp_recursion` is equivalent and easier to remember.
```

### Collecting the measures

Let's gather the four measures into one function.

```{code-cell} ipython3
def mobility_measures(M):
    "Return the four mobility measures for stochastic matrix M."
    return pd.Series({'μ_S':   shorrocks(M),
                      'μ_B':   bartholomew(M),
                      '~μ_B':  bartholomew_normalized(M),
                      'μ_2E':  second_eigenvalue(M),
                      'μ_MFP': mfp_measure(M)})

def mobility_table(matrices):
    "Apply the measures to a dict of labelled stochastic matrices."
    return pd.DataFrame({k: mobility_measures(M)
                         for k, M in matrices.items()}).T
```

The identity matrix is reducible, so we check the perfect mobility benchmark alone.

```{code-cell} ipython3
mobility_table({'perfect mobility': M_perfect}).round(3)
```

All four measures equal one, apart from the unnormalized $\mu_B$.

## What the measures miss

Each measure discards information, and the cleanest way to see what is lost is to find matrices that a measure cannot tell apart.

The following examples are adapted from Appendix A.1 of {cite}`carroll2026mobility`, perturbed slightly so that all three chains are irreducible.

Consider three economies with three wealth terciles.

```{code-cell} ipython3
ladder = np.array([[0.50, 0.50, 0.00],
                   [0.25, 0.50, 0.25],
                   [0.00, 0.50, 0.50]])

jumper = np.array([[0.50, 0.10, 0.40],
                   [0.25, 0.50, 0.25],
                   [0.40, 0.10, 0.50]])

sticky = np.array([[0.70, 0.10, 0.20],
                   [0.25, 0.50, 0.25],
                   [0.20, 0.10, 0.70]])
```

In the *ladder* economy households can only move to an adjacent tercile, so the poor must pass through the middle class to become rich.

In the *jumper* economy households leave their tercile just as often, but when they move they usually move all the way.

In the *sticky* economy households move rarely, but when they do they tend to move a long way.

```{code-cell} ipython3
mobility_table({'ladder': ladder,
                'jumper': jumper,
                'sticky': sticky}).round(3)
```

Three lessons follow.

The ladder and jumper economies have identical Shorrocks index, because they have the same diagonal.

Shorrocks cannot see that one economy sends households from the bottom to the top in a single step while the other requires two.

The ladder and sticky economies have identical Bartholomew measure, because the extra distance travelled in the sticky economy exactly offsets its lower frequency of movement.

Bartholomew cannot separate frequent small moves from rare large ones.

The second eigenvalue is also equal across the ladder and sticky economies.

Most striking is that the measures *disagree about the ranking*.

Bartholomew and the second eigenvalue both rank the jumper economy as more mobile than the ladder economy, while mean first passage time ranks it as slightly less mobile.

The reason is that a household in the jumper economy that wants to reach the middle tercile has to wait a long time, since almost all of the movement is between the extremes.

There is, in short, no complete ordering of mobility matrices, and any single index imposes one by fiat.

This is the central message of {cite}`fields1999measurement` and {cite}`dardanoni1993measuring`, and it is why we report four numbers rather than one.

(mobility_data)=
## Wealth mobility in the US data

### The data

We now turn to the wealth mobility matrices estimated by {cite}`carroll2026mobility` from the Panel Study of Income Dynamics (PSID).

The PSID follows the same families over time and includes wealth supplements at irregular intervals between 1984 and 2015.

For each pair of survey years, families are sorted into wealth quintiles in the starting year and in the ending year, and the fraction moving from quintile $i$ to quintile $j$ is recorded.

The authors report matrices at three horizons: short (5--6 years), medium (9--10 years) and long (19--21 years).

We start with three horizons that share the same starting year, 1984.

```{code-cell} ipython3
psid = {}

psid['1984-1989'] = [[0.70, 0.23, 0.05, 0.02, 0.00],
                     [0.25, 0.45, 0.22, 0.06, 0.02],
                     [0.06, 0.24, 0.44, 0.19, 0.06],
                     [0.02, 0.06, 0.22, 0.47, 0.23],
                     [0.01, 0.01, 0.06, 0.22, 0.70]]

psid['1984-1994'] = [[0.63, 0.24, 0.09, 0.03, 0.02],
                     [0.23, 0.41, 0.21, 0.10, 0.05],
                     [0.10, 0.28, 0.33, 0.21, 0.09],
                     [0.05, 0.08, 0.26, 0.37, 0.23],
                     [0.02, 0.03, 0.09, 0.25, 0.61]]

psid['1984-2003'] = [[0.58, 0.25, 0.11, 0.05, 0.02],
                     [0.26, 0.35, 0.22, 0.12, 0.05],
                     [0.09, 0.29, 0.27, 0.22, 0.13],
                     [0.05, 0.11, 0.27, 0.32, 0.26],
                     [0.03, 0.06, 0.11, 0.26, 0.55]]
```

As before, the published rounding leaves the rows slightly off.

```{code-cell} ipython3
np.array(psid['1984-2003']).sum(axis=1)
```

```{code-cell} ipython3
psid = {k: normalize_rows(M) for k, M in psid.items()}
```

Here is what the three matrices look like.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: US wealth mobility matrices at three horizons
    name: fig:mobility-heatmaps
tags: [hide-input]
---
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (label, M) in zip(axes, psid.items()):
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=0.7)
    ax.set_title(label)
    ax.set_xticks(range(5), range(1, 6))
    ax.set_yticks(range(5), range(1, 6))
    ax.set_xlabel('ending quintile')
axes[0].set_ylabel('starting quintile')
fig.colorbar(im, ax=axes, shrink=0.8)
plt.show()
```

The mass spreads away from the diagonal as the horizon lengthens, which is what we should expect.

There is a good deal of movement even at five years.

Families in the middle three quintiles are more likely to leave their starting quintile than to remain in it, and a household starting in the top quintile has a 30% chance of ending elsewhere.

### Mobility rises with the horizon

Let's apply our measures.

```{code-cell} ipython3
horizon_table = mobility_table(psid)
horizon_table.round(3)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Mobility measures at three horizons
    name: fig:mobility-horizon
tags: [hide-input]
---
cols = ['μ_S', '~μ_B', 'μ_2E', 'μ_MFP']
fig, ax = plt.subplots()
horizon_table[cols].plot.bar(ax=ax, rot=0, width=0.75)
ax.set_ylabel('mobility')
ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=4)
plt.show()
```

All four measures agree that mobility rises with the horizon, and they agree by roughly the same proportion.

This is reassuring but not informative: given enough time, any irreducible chain forgets where it started, so every measure must approach one as the horizon lengthens.

The lesson is that mobility measures are only comparable *across matrices with the same horizon*.

This warning applies with particular force to $\mu_{MFP}$, whose units are quantiles per period --- and here a "period" is five years in one column and nineteen in another.

### Has mobility declined?

A more interesting comparison holds the horizon fixed and varies the sample period.

Here are three long-horizon matrices, each spanning about twenty years, starting in 1984, 1989 and 1994.

```{code-cell} ipython3
long_horizon = {}

long_horizon['1984-2003'] = psid['1984-2003']

long_horizon['1989-2009'] = normalize_rows(
                            [[0.56, 0.28, 0.10, 0.04, 0.03],
                             [0.27, 0.37, 0.20, 0.12, 0.05],
                             [0.12, 0.25, 0.29, 0.22, 0.12],
                             [0.08, 0.11, 0.29, 0.32, 0.20],
                             [0.02, 0.05, 0.09, 0.25, 0.60]])

long_horizon['1994-2015'] = normalize_rows(
                            [[0.58, 0.24, 0.11, 0.04, 0.03],
                             [0.28, 0.38, 0.20, 0.10, 0.04],
                             [0.13, 0.25, 0.32, 0.21, 0.08],
                             [0.07, 0.11, 0.24, 0.34, 0.25],
                             [0.03, 0.05, 0.09, 0.25, 0.58]])
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Long-horizon mobility over three sample periods
    name: fig:mobility-decline
tags: [hide-input]
---
decline_table = mobility_table(long_horizon)

fig, ax = plt.subplots()
for col in cols:
    ax.plot(decline_table.index, decline_table[col], 'o-', lw=2, label=col)
ax.set_ylabel('mobility')
ax.set_ylim(0, 0.8)
ax.legend(frameon=False)
plt.show()
```

```{code-cell} ipython3
decline_table.round(3)
```

All four measures fall as we move the twenty-year window forward, suggesting that US wealth mobility has declined since the mid-1980s.

The decline is modest --- $\mu_S$ falls from 0.74 to 0.70 --- and we should be cautious about it.

{cite}`carroll2026mobility` bootstrap the PSID sample to place confidence intervals around these numbers, and conclude that the decline is statistically significant at the medium horizon but *not* at the long horizon shown here.

A drop of this size is well within the sampling error of a panel of a few thousand families.

### Is the quintile chain Markov?

There is one more question the data let us ask.

So far we have treated each matrix as a separate object.

But if a household's quintile really were a Markov chain, the twenty-year matrix would be the five-year matrix raised to the fourth power.

Let's check.

```{code-cell} ipython3
M_short = psid['1984-1989']

comparison = mobility_table(
    {'1984-1994 (data)':  psid['1984-1994'],
     'M^2 (predicted)':   np.linalg.matrix_power(M_short, 2),
     '1984-2003 (data)':  psid['1984-2003'],
     'M^4 (predicted)':   np.linalg.matrix_power(M_short, 4)})

comparison.round(3)
```

Iterating the five-year matrix predicts substantially *more* mobility than we observe.

At twenty years the gap is large: $\mu_S$ is 0.87 under the Markov prediction against 0.74 in the data.

Households are therefore more persistent over long horizons than their five-year behavior implies, which means that current quintile alone is not a sufficient statistic for a household's future position.

Something else, unobserved and persistent, is at work.

{cite}`carroll2026mobility` find direct evidence for this: a family that makes one large jump through the wealth distribution is significantly more likely to make another, and families holding stocks or private businesses move much more than others.

In the language of the older sociological literature, the population contains both movers and stayers.

This matters for modelling.

It means that a calibration matching mobility at one horizon will generally miss it at another, and that the state of a realistic model must include something beyond position in the wealth distribution.

## Exercises

```{exercise}
:label: mob_ex1

This exercise asks you to check the calibration of the Shorrocks index and explore its range.

1. Show analytically that $\mu_S(I) = 0$ and $\mu_S(M^*) = 1$, where $M^* = \mathbb 1 \mathbb 1^\top / N$.

2. Show that $\mu_S(M) \leq N/(N-1)$ for any stochastic matrix $M$, with equality if and only if the diagonal of $M$ is zero.

3. Hence find a $3 \times 3$ stochastic matrix with $\mu_S(M) > 1$ and confirm your answer in code.

4. Explain in words what such a matrix does, and why exceeding one is not a defect of the measure.
```

```{solution-start} mob_ex1
:class: dropdown
```

For part 1, the identity matrix has $\mathrm{trace}(I) = N$, so $\mu_S(I) = (N - N)/(N-1) = 0$.

The perfect mobility matrix has every diagonal element equal to $1/N$, so $\mathrm{trace}(M^*) = 1$ and $\mu_S(M^*) = (N-1)/(N-1) = 1$.

For part 2, every element of a stochastic matrix is nonnegative, so $\mathrm{trace}(M) \geq 0$, giving $\mu_S(M) \leq N/(N-1)$.

Equality requires $\mathrm{trace}(M) = 0$, which for a nonnegative matrix means every diagonal element is zero.

For part 3, any stochastic matrix with a zero diagonal will do, and the simplest is a cyclic permutation.

```{code-cell} ipython3
M_cycle = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0]])

shorrocks(M_cycle)
```

This is $3/2$, the maximum for $N = 3$.

For part 4, this matrix moves every household out of its current tercile with probability one, which is more movement than origin independence delivers --- under $M^*$ a household stays put with probability $1/N$.

So values above one indicate systematic rank reversal rather than an error.

Note that this chain is periodic, so the distribution never converges, even though the chain is irreducible.

Compare the discussion of periodic chains in {doc}`markov_chains_II`.

```{solution-end}
```

```{exercise}
:label: mob_ex2

Consider the two-state chain from {doc}`markov_chains_I`,

$$
M =
\begin{bmatrix}
    1 - \alpha & \alpha \\
    \beta & 1 - \beta
\end{bmatrix}
$$

with $\alpha, \beta \in (0,1)$ and $\alpha + \beta \leq 1$.

1. Show that $\mu_S(M) = \tilde\mu_B(M) = \mu_{2E}(M) = \alpha + \beta$.

2. Let $\{X_t\}$ be a stationary chain with this transition matrix, viewed as taking values in $\{0, 1\}$.

   Show that its autocorrelation is $\mathrm{corr}(X_t, X_{t+1}) = 1 - \alpha - \beta = |\lambda_2(M)|$.

3. Verify both results numerically.

4. Comment on what this tells us about when the choice of mobility measure matters.
```

```{solution-start} mob_ex2
:class: dropdown
```

For part 1, the trace is $2 - \alpha - \beta$, so $\mu_S = (2 - (2 - \alpha - \beta))/1 = \alpha + \beta$.

For Bartholomew, only the two off-diagonal terms contribute and each has $|i - j| = 1$, so $\mu_B = \alpha + \beta$, and with $N = 2$ the normalization factor $3/(N+1)$ equals one.

For the eigenvalues, the rows of $M$ sum to one so $\lambda_1 = 1$, and since the eigenvalues sum to the trace we have $\lambda_2 = 1 - \alpha - \beta$, which is nonnegative under our assumption.

Hence $\mu_{2E} = 1 - (1 - \alpha - \beta) = \alpha + \beta$.

For part 2, the stationary distribution is $\psi^* = (\beta, \alpha)/(\alpha + \beta)$, so with $p := \psi^*(1) = \alpha/(\alpha+\beta)$ we have $\mathbb E X_t = p$ and $\mathrm{var}(X_t) = p(1-p)$.

Since $X_t$ is a zero-one variable,

$$
\mathbb E [X_t X_{t+1}]
= \mathbb P\{X_t = 1, X_{t+1} = 1\}
= p (1 - \beta)
$$

Therefore

$$
\mathrm{cov}(X_t, X_{t+1})
= p(1-\beta) - p^2
= p (1 - p - \beta)
= p(1-p)(1 - \alpha - \beta)
$$

where the last step uses $1 - p = \beta/(\alpha+\beta)$, so that $1 - p - \beta = (1-p)(1 - \alpha - \beta)$.

Dividing by $\mathrm{var}(X_t) = p(1-p)$ gives the result.

```{code-cell} ipython3
α, β = 0.2, 0.3
M2 = np.array([[1 - α, α],
               [β, 1 - β]])

print(f'μ_S  = {shorrocks(M2):.4f}')
print(f'~μ_B = {bartholomew_normalized(M2):.4f}')
print(f'μ_2E = {second_eigenvalue(M2):.4f}')
print(f'α+β  = {α + β:.4f}')
```

```{code-cell} ipython3
mc2 = qe.MarkovChain(M2)
X = mc2.simulate(1_000_000, random_state=0)
print(f'sample autocorrelation = {np.corrcoef(X[:-1], X[1:])[0, 1]:.4f}')
print(f'1 - α - β              = {1 - α - β:.4f}')
```

For part 4, with only two states there is a single way to leave a state and a single distance to travel, so the measures cannot disagree.

Mobility measures start to differ only when $N \geq 3$, because only then is there a distinction between moving far and moving often.

Note that $\mu_{MFP}$ is an exception and differs from the others even at $N = 2$.

```{code-cell} ipython3
mfp_measure(M2), 8 * α * β / ((1 + α + β) * (α + β))
```

```{solution-end}
```

```{exercise}
:label: mob_ex3

The Shorrocks index averages escape probabilities, but a closely related and more interpretable quantity is the expected time a household spends in its current quantile before leaving.

1. Explain why, for a household currently in quantile $i$, the number of periods until it leaves is geometric with success probability $1 - m_{ii}$, so that the mean sojourn time is $1/(1 - m_{ii})$.

2. Compute the mean sojourn time in each quintile, in years, for the 1984--1989 matrix.

3. Repeat for the perfect mobility matrix and comment.
```

```{solution-start} mob_ex3
:class: dropdown
```

For part 1, by the Markov property, at each period the household leaves quantile $i$ with probability $1 - m_{ii}$, independently of how long it has already been there.

The number of periods until the first departure is therefore geometric, with mean $1/(1 - m_{ii})$.

For part 2, each period of the 1984--1989 matrix is five years.

```{code-cell} ipython3
sojourn = 5 / (1 - np.diag(psid['1984-1989']))

pd.Series(sojourn.round(1), index=[f'quintile {i}' for i in range(1, 6)],
          name='years')
```

Households remain in the bottom and top quintiles for about 17 years on average, roughly twice as long as in the middle three quintiles.

This is the pattern that {cite}`carroll2026mobility` find standard incomplete-markets models struggle to reproduce.

For part 3, under perfect mobility every diagonal element is $1/N$.

```{code-cell} ipython3
5 / (1 - np.diag(M_perfect))
```

A household stays for $N/(N-1) = 1.25$ periods --- 6.25 years --- which is the minimum consistent with independence rather than zero.

Even with no persistence at all, a household has a one in five chance of drawing its own quintile again.

```{solution-end}
```

```{exercise}
:label: mob_ex4

Take the 1984--1989 matrix $M$ and compute all four mobility measures for $M^k$, $k = 1, \ldots, 20$.

Plot the results and explain what you see.

What does this tell us about comparing mobility measures across horizons?
```

```{solution-start} mob_ex4
:class: dropdown
```

```{code-cell} ipython3
:tags: [hide-input]

ks = np.arange(1, 21)
paths = pd.DataFrame([mobility_measures(np.linalg.matrix_power(M_short, k))
                      for k in ks], index=ks)

fig, ax = plt.subplots()
for col in cols:
    ax.plot(ks, paths[col], lw=2, label=col)
ax.axhline(1.0, ls='--', color='black', lw=1)
ax.set_xlabel('k')
ax.set_ylabel('mobility')
ax.legend(frameon=False)
plt.show()
```

Every measure increases monotonically towards one.

This must happen: the matrix is irreducible and aperiodic, so by the {doc}`Perron-Frobenius theorem <eigen_II>` the rows of $M^k$ converge to the stationary distribution, and $M^k$ therefore converges to a matrix of the perfect mobility form.

The practical implication is that a mobility measure means nothing without a stated horizon.

Reporting that one economy has $\mu_S = 0.75$ and another $\mu_S = 0.56$ is uninformative if the first figure covers twenty years and the second covers five.

```{solution-end}
```

```{exercise}
:label: mob_ex5

Mobility matrices cannot be completely ordered, and the toy examples above showed one disagreement.

Search for others.

Generate a large number of random $5 \times 5$ stochastic matrices, compute $\tilde\mu_B$ and $\mu_{MFP}$ for each, and find a pair that the two measures rank in opposite directions.

Report the pair and explain the disagreement.
```

```{solution-start} mob_ex5
:class: dropdown
```

We generate random stochastic matrices by drawing each row from a Dirichlet distribution.

```{code-cell} ipython3
rng = np.random.default_rng(seed=42)

n_draws = 300
draws = [rng.dirichlet(np.ones(5), size=5) for _ in range(n_draws)]
scores = np.array([[bartholomew_normalized(M), mfp_measure(M)] for M in draws])
```

Some draws are close to reducible, which sends $\mu_{MFP}$ towards zero for reasons that have nothing to do with the comparison we want.

We set those aside.

```{code-cell} ipython3
keep = np.flatnonzero(scores[:, 1] > 0.4)
len(keep)
```

Now we look for a pair that the two measures rank in opposite directions.

We score each pair by the *smaller* of the two gaps, so that the winner disagrees substantially on both measures rather than hugely on one.

```{code-cell} ipython3
B, T = scores[keep, 0], scores[keep, 1]
Δ_B = B[:, None] - B[None, :]
Δ_T = T[:, None] - T[None, :]

disagree = np.sign(Δ_B) != np.sign(Δ_T)
score = np.where(disagree, np.minimum(np.abs(Δ_B), np.abs(Δ_T)), 0.0)
a, b = np.unravel_index(np.argmax(score), score.shape)

A, C = draws[keep[a]], draws[keep[b]]
pd.DataFrame({'matrix A': mobility_measures(A),
              'matrix C': mobility_measures(C)}).round(3)
```

Bartholomew ranks matrix C as far more mobile than matrix A, and mean first passage time ranks it as far less mobile.

Here are the two matrices.

```{code-cell} ipython3
np.round(A, 2)
```

```{code-cell} ipython3
np.round(C, 2)
```

The explanation is visible in the second column of matrix C.

```{code-cell} ipython3
C[:, 1]
```

Almost no probability enters quantile 2, so a household waits a very long time to reach it, and the mean first passage measure is dragged down accordingly.

Bartholomew's measure never notices, because it counts only distance travelled, and matrix C moves plenty of mass between the remaining quantiles --- including between the extremes.

The two measures reward different things.

Bartholomew's measure counts distance travelled and so favours a matrix that shifts mass towards the corners.

Mean first passage time asks how long a household waits to reach an arbitrary quantile, and so penalises any matrix that makes some quantile hard to enter, however much movement occurs elsewhere.

```{solution-end}
```

```{exercise}
:label: mob_ex6

Our function `mean_first_passage` solves a linear system.

An alternative is to estimate first passage times by simulation.

1. Write a function that estimates $T_{ij}$ by simulating many paths from state $i$ and recording the first time each path hits state $j$.

   Use `qe.MarkovChain.simulate`, which is JIT compiled.

2. Compare your estimates to the exact values for the 1984--1989 matrix.

3. As a second check, compute the stationary distribution used on the diagonal of $T$ by hand --- by solving $\psi^* (I - M) = 0$ subject to $\psi^* \mathbb 1 = 1$ --- and compare against `qe.MarkovChain.stationary_distributions`.
```

```{solution-start} mob_ex6
:class: dropdown
```

For part 1, we simulate long paths from each starting state and record first hitting times.

```{code-cell} ipython3
def mfp_simulated(M, num_paths=2_000, path_length=400, seed=1234):
    N = len(M)
    mc = qe.MarkovChain(M)
    T = np.zeros((N, N))
    for i in range(N):
        hits = np.full((num_paths, N), np.nan)
        for m in range(num_paths):
            X = mc.simulate(path_length, init=i, random_state=seed + m)
            for j in range(N):
                first = np.flatnonzero(X[1:] == j)
                if first.size > 0:
                    hits[m, j] = first[0] + 1
        T[i] = np.nanmean(hits, axis=0)
    return T
```

Note that we search `X[1:]`, so that the diagonal records the mean *return* time rather than zero, matching the convention in `mean_first_passage`.

```{code-cell} ipython3
T_exact = mean_first_passage(M_short)
T_sim = mfp_simulated(M_short)

print('exact:')
print(np.round(T_exact, 2))
print('simulated:')
print(np.round(T_sim, 2))
```

The estimates are close, and the remaining differences are Monte Carlo error.

One bias is worth watching for, though.

Paths that never reach $j$ within `path_length` steps contribute nothing, so if the paths are too short relative to the rarest transition the estimates are biased downwards.

Try rerunning with `path_length=30` to see this.

For part 3, the stationary distribution is a left eigenvector of $M$.

We can find it by solving a linear system, replacing one redundant equation with the normalization $\psi^* \mathbb 1 = 1$.

```{code-cell} ipython3
def stationary_by_hand(M):
    N = len(M)
    A = np.identity(N) - M.T
    A[-1, :] = 1.0              # replace last equation with normalisation
    b = np.zeros(N)
    b[-1] = 1.0
    return np.linalg.solve(A, b)

ψ_hand = stationary_by_hand(M_short)
ψ_qe = qe.MarkovChain(M_short).stationary_distributions[0]

print(np.round(ψ_hand, 6))
print(np.round(ψ_qe, 6))
print(f'max abs difference = {np.max(np.abs(ψ_hand - ψ_qe)):.2e}')
```

The two agree to machine precision.

It is worth noticing that the stationary distribution is not exactly uniform, even though quintiles contain equal numbers of households by construction.

In a stationary environment it would be, so the deviation reflects sampling error together with the fact that the wealth distribution itself changed between 1984 and 1989.

```{solution-end}
```

## Further reading

### Measurement theory

The formal study of mobility indices begins with {cite}`prais1955measuring`, who applied transition matrices to occupational classes in England, and {cite}`bartholomew1967stochastic`.

{cite}`shorrocks1978measurement` put the subject on an axiomatic footing.

He asks what properties a mobility index should satisfy --- among them that it equal zero only under complete immobility, that it be invariant to relabelling, that it increase when probability mass is shifted off the diagonal, and that it take a common value under origin independence --- and shows that natural-looking lists of such axioms turn out to be mutually inconsistent.

The upshot is that indices necessarily trade one desirable property against another, which is exactly the tension we saw in the toy examples above.

{cite}`dardanoni1993measuring` develops the alternative response to that impossibility: rather than forcing a complete ranking, ask when one matrix is unambiguously more mobile than another, and accept a partial order.

{cite}`sommers1979eigenvalue` treat the eigenvalue measures, {cite}`conlisk1990monotone` the mean first passage approach and the role of monotonicity, and {cite}`kemeny1976finite` remains the standard reference for the underlying Markov chain theory.

{cite}`fields1999measurement` survey the whole literature and are the best place to start.

{cite}`cowell2018measuring` give a more recent treatment that works directly with the underlying distributions rather than with a discretized transition matrix.

```{note}
This section describes the shape of Shorrocks' impossibility result rather than stating it formally, since the precise axiom list matters.

Readers should consult {cite}`shorrocks1978measurement` directly.
```

### Intergenerational mobility

This lecture has studied *intragenerational* mobility: the movement of a given family through the distribution over its own lifetime.

A large parallel literature studies *intergenerational* mobility, meaning the relationship between the economic position of parents and that of their children.

The headline statistic there is the intergenerational elasticity, the coefficient from regressing a child's log earnings on the parent's.

Its most famous appearance is the **Great Gatsby curve**, named by Alan Krueger in 2012, which plots this elasticity against income inequality across countries {cite}`corak2013income`.

The curve slopes upward: more unequal countries tend to have less intergenerational mobility, with Denmark and Norway at one end and the United States and the United Kingdom at the other.

The interpretation is contested --- the relationship is a cross-country correlation across a few dozen data points, and causality could run either way --- but the pattern has been influential in policy debate.

{cite}`chetty2014land` use US administrative tax records covering tens of millions of families to show that intergenerational mobility varies enormously *within* the United States, across commuting zones, and relate that variation to segregation, school quality and family structure.

For wealth specifically, {cite}`hurst1998wealth` and {cite}`jianakoplos1997wealth` are early studies using the PSID, and {cite}`benhabib_wealth_2019` estimate a model of wealth mobility whose transition matrix we met in {doc}`markov_chains_II`.

### Mobility in economic models

Everything in this lecture describes data.

The natural next question is whether our standard models of household saving can reproduce it.

The answer, developed in the second half of {cite}`carroll2026mobility`, is that they largely cannot.

A standard incomplete-markets model in the tradition of Bewley, Huggett and Aiyagari, calibrated to match the observed level of wealth inequality, generates far too little short-run mobility.

Households in the model remain in the bottom and top quintiles for around 38 and 63 years respectively, against roughly 15 and 17 in the data, and when they move they move only one quintile at a time.

The reason is that saving is a *smoothing* device: agents accumulate assets precisely in order to blunt the effect of income shocks on consumption, and this dampening slows their passage through the wealth distribution.

The paper shows that adding idiosyncratic risk to the *return* on wealth, rather than to labour income, is what brings model mobility into line with the data --- consistent with its empirical finding that families making large jumps are the ones holding stocks and private businesses.

It then shows that this matters for policy: across model economies calibrated to identical wealth inequality, the capital income tax rate that households prefer varies with the level of mobility.

That is the case for measuring mobility rather than inequality alone.

Readers interested in the model side should turn to that paper.
