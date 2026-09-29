---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Estimating Wealth Tax Revenue

## Overview

How much revenue would a tax on the wealth of the very rich raise?

This question has been at the center of a lively policy debate.

In 2019, US Senator Elizabeth Warren proposed an annual tax on household net
worth above 50 million dollars, with a higher rate above one billion dollars
(see the [proposal](https://elizabethwarren.com/plans/ultra-millionaire-tax)).

Emmanuel Saez and Gabriel Zucman estimated that the tax would raise about 2.75
trillion dollars over ten years ([letter to
Warren](https://www.warren.senate.gov/wp-content/uploads/media/doc/saez-zucman-wealthtax.pdf)),
and discuss wealth taxation in detail in {cite}`saez2019progressive`.

Other economists argued that these estimates were too high, and much of the
disagreement concerned the number and wealth of the very richest households.

In this lecture we estimate the revenue from a wealth tax using US household
survey data.

We will see that the obvious estimate, based directly on the survey, misses
the part of the distribution that matters most, because the richest households
are not in the data.

To fill the gap we model the upper tail of the wealth distribution with a
Pareto distribution, which we fit by {doc}`maximum likelihood <mle>`.

We will use the following imports.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


## The tax

We study a modification of the Warren proposal, with the following marginal
tax rates on household net worth:

| Net worth | Marginal rate |
|---|---|
| up to \$10 million | 0% |
| \$10 million to \$50 million | 1% |
| \$50 million to \$1 billion | 2% |
| above \$1 billion | 6% |

The last two brackets follow Warren's revised proposal from November 2019,
while the first bracket is our addition.

Let $h(w)$ be the tax paid by a household with net worth $w$.

A convenient way to write $h$ is as a sum of terms, one for each bracket
threshold $t_k$:

$$
h(w) = \sum_{k=1}^{3} r_k \, (w - t_k)^+
$$ (eq:wt_tax)

Here $x^+ = \max\{x, 0\}$, the thresholds are $t_1 = 10$ million,
$t_2 = 50$ million and $t_3 = 1$ billion, and $r_k$ is the *increase* in the
marginal rate at $t_k$.

So $r_1 = 0.01$, $r_2 = 0.01$ and $r_3 = 0.04$, which gives marginal rates of
1%, 2% and 6%.

```{code-cell} ipython3
thresholds = np.array([10e6, 50e6, 1e9])
rate_increases = np.array([0.01, 0.01, 0.04])

def h(w, thresholds=thresholds, rate_increases=rate_increases):
    "Tax paid by a household with net worth w."
    w = np.asarray(w, dtype=float)[..., None]
    return np.sum(rate_increases * np.maximum(w - thresholds, 0), axis=-1)
```

Let's plot the tax and the average tax rate $h(w)/w$, using a log scale for
wealth.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The tax and the average tax rate
    name: fig:wt-schedule
---
w_grid = np.geomspace(1e6, 1e11, 400)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(w_grid, h(w_grid), lw=2)
axes[0].set_yscale('log')
axes[0].set_ylim(1e4, 1e10)
axes[0].set_ylabel('tax (US$)')
axes[0].set_title('tax')
axes[1].plot(w_grid, 100 * h(w_grid) / w_grid, lw=2)
axes[1].set_ylabel('average tax rate (%)')
axes[1].set_title('average tax rate')
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel('net worth (US$)')
plt.tight_layout()
plt.show()
```

The average tax rate rises steadily with wealth, approaching 6% for the very
largest fortunes.

Total revenue is the sum of $h(w)$ over all households in the country.


## The survey estimate

Our data come from the 2022 [Survey of Consumer
Finances](https://www.federalreserve.gov/econres/scfindex.htm) (SCF), run by
the Federal Reserve Board.

It is the most detailed source of information on the wealth of US households.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/scf_2022_networth.csv')
scf = pd.read_csv(url)
scf.head()
```

Each row records the net worth `networth` of a surveyed household and a
**survey weight** `wgt`.

The weight is the number of US households that the row represents.

Weights are needed because the SCF deliberately oversamples wealthy
households, which it selects using tax records {cite}`bricker2017updates`.

Oversampling gives the survey many more rich households than a purely random
sample would, which makes it much more informative about the top of the
distribution.

The weights undo the oversampling, by giving each rich household a small
weight.

(You may notice that each household appears five times, with the same
identifier `yy1`.

The five rows, called *implicates*, are alternative versions of the household's
answers that the Fed uses to fill in missing values.

The weights have been scaled so that we can simply work with all rows at once.)

```{code-cell} ipython3
w = scf['networth'].to_numpy()
ω = scf['wgt'].to_numpy()

ω.sum() / 1e6     # number of US households, in millions
```

Here is the effect of the oversampling.

```{code-cell} ipython3
rich = w > 10e6
print(f"share of rows above $10 million:       {rich.mean():.3f}")
print(f"share of households above $10 million: {ω[rich].sum() / ω.sum():.3f}")
```

About 14% of the rows in the survey are households worth more than 10 million
dollars, but they represent only about 1.6% of US households.

The obvious estimate of total revenue replaces the sum over all households by
the weighted sum over the survey:

$$
\hat T_S = \sum_i \omega_i \, h(w_i)
$$

where $\omega_i$ is the weight of row $i$.

This is a sample average of $h(w)$, scaled up to the population, and is
justified by the {doc}`law of large numbers <lln_clt>`.

```{code-cell} ipython3
T_survey = np.sum(ω * h(w))
print(f"survey estimate: ${T_survey / 1e9:.0f} billion per year")
```


## The missing rich

There is a problem with this estimate.

By design, the SCF excludes the people on the Forbes 400 list of the wealthiest
Americans.

In addition, the Fed removed from the public data the few survey respondents
whose wealth would have qualified them for the list, since such people would be
too easy to identify.

Let's count the households in the survey above various levels of wealth.

```{code-cell} ipython3
levels = [10e6, 50e6, 100e6, 1e9, 2.7e9]
pd.DataFrame({
    'survey households': [scf.loc[w > c, 'yy1'].nunique() for c in levels],
    'US households represented': [round(ω[w > c].sum()) for c in levels]},
    index=['> $10M', '> $50M', '> $100M', '> $1B', '> $2.7B'])
```

The survey contains only about twenty households above one billion dollars,
and none above 2.7 billion dollars.

2.7 billion dollars is exactly the wealth needed to make the 2022 Forbes 400
list, whose members were worth a combined 4.0 trillion dollars
([Forbes](https://www.forbes.com/sites/chasewithorn/2022/09/27/the-2022-forbes-400-list-of-richest-americans-facts-and-figures/)).

```{code-cell} ipython3
forbes_count = 400
forbes_cutoff = 2.7e9
forbes_wealth = 4.0e12

print(f"total wealth in the survey: ${np.sum(ω * w) / 1e12:.0f} trillion")
```

So the Forbes 400 hold about 3% of US household wealth, and all of it is
missing from the survey.

Under our tax, every Forbes 400 member pays the full amount on the first two
brackets, plus 6% of their wealth above one billion dollars.

```{code-cell} ipython3
forbes_tax = (forbes_count * (0.01 * 40e6 + 0.02 * 950e6)
              + 0.06 * (forbes_wealth - forbes_count * 1e9))
print(f"tax owed by the Forbes 400: ${forbes_tax / 1e9:.0f} billion per year")
```

This is a large amount relative to the survey estimate, and it is entirely
missed by $\hat T_S$.

There is a second, subtler problem.

Even where the survey does contain very rich households, it contains very few
of them, so the survey estimate of revenue from the top bracket rests on a
handful of observations and is highly uncertain.


## A Pareto model of the upper tail

To deal with these problems we need a model of the upper tail of the wealth
distribution, which we can estimate from the part of the tail the survey
observes and then extend to the part it misses.

As discussed in {doc}`heavy_tails`, the standard model is the Pareto
distribution, going back to {cite:t}`pareto1896cours`.

There is strong empirical support for this choice {cite}`saez2016wealth`,
{cite}`vermeulen2018fat`.

There is also theoretical support: many models in which fortunes grow at
random rates generate Pareto upper tails {cite}`gabaix2016power`,
{cite}`jones2015pareto`, {cite}`benhabib2018skewed`.

Recall from {doc}`heavy_tails` that if wealth $W$ has a Pareto tail with tail
index α above some threshold $u$, then

$$
\mathbb{P}\{W > x \mid W > u\} = \left( \frac{u}{x} \right)^\alpha
\qquad (x \geq u)
$$

Taking logs, $\ln \mathbb{P}\{W > x\}$ is a linear function of $\ln x$ with
slope $-\alpha$.

So a Pareto tail shows up as a straight line when we plot the counter CDF (the
probability of exceeding $x$) against $x$ on log-log axes.

Let's plot the counter CDF of the survey data, using the weights so that each
row counts in proportion to the households it represents.

```{code-cell} ipython3
def weighted_ccdf(x_grid, w, ω):
    "Fraction of households with wealth above each point of x_grid."
    return np.array([ω[w > x].sum() for x in x_grid]) / ω.sum()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Counter CDF of US household wealth
    name: fig:wt-ccdf
---
x_grid = np.geomspace(1e6, 2.5e9, 300)

fig, ax = plt.subplots()
ax.loglog(x_grid, weighted_ccdf(x_grid, w, ω), lw=2, label='survey')
ax.axvline(10e6, color='k', ls=':', lw=1)
ax.set_ylim(1e-8, 1)
ax.set_xlabel('net worth (US$)')
ax.set_ylabel('fraction of households above')
ax.legend()
plt.show()
```

Below about 10 million dollars (the dotted line), the curve bends, so the
distribution is not Pareto there.

Above 10 million dollars it is close to a straight line.

At the far right the curve plunges, because the survey data stop just below
2.4 billion dollars.

We therefore fit a Pareto tail above the threshold $u = 10$ million dollars,
which is also where our tax begins.


## Estimating the tail index

In {doc}`mle` we found that the maximum likelihood estimate of the tail index,
given observations $x_1, \ldots, x_n$ above a known threshold $u$, is

$$
\hat \alpha = \frac{n}{\sum_{i=1}^n \ln (x_i / u)}
$$

Here we have to account for the weights.

If row $i$ represents $\omega_i$ households, then it contributes $\omega_i$
copies of its log density to the log likelihood of the population:

$$
\ell(\alpha) = \sum_{i: w_i > u} \omega_i \ln f(w_i; \alpha)
$$

Repeating the calculation in {doc}`mle` with these weights gives

$$
\hat \alpha = \frac{\sum_{i: w_i > u} \omega_i}
                  {\sum_{i: w_i > u} \omega_i \ln (w_i / u)}
$$

```{code-cell} ipython3
def tail_index(u, w, ω):
    "Weighted maximum likelihood estimate of the Pareto tail index above u."
    above = w > u
    return ω[above].sum() / np.sum(ω[above] * np.log(w[above] / u))

u = 10e6
α_hat = tail_index(u, w, ω)
α_hat
```

This is close to published estimates for the United States, which are around
1.5 {cite}`vermeulen2018fat`.

How sensitive is the estimate to the choice of threshold?

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Estimated tail index against the threshold
    name: fig:wt-alpha-u
---
u_grid = np.geomspace(1e6, 100e6, 60)
α_grid = [tail_index(v, w, ω) for v in u_grid]

fig, ax = plt.subplots()
ax.plot(u_grid, α_grid, lw=2)
ax.axvline(u, color='k', ls=':', lw=1)
ax.set_xscale('log')
ax.set_xlabel('threshold $u$ (US$)')
ax.set_ylabel(r'estimate of $\alpha$')
plt.show()
```

With a low threshold, the estimate is close to one.

This is misleading, since it treats the curved part of the distribution below
10 million dollars as if it were Pareto.

From about 10 million dollars onward, the estimate stabilizes at around 1.5.

The remaining fluctuations are of the size we would expect from sampling
variation, as you can check in {ref}`an exercise <wt_ex_bootstrap>`.


## The Pareto estimate of revenue

With the tail estimated, we can compute revenue from the model instead of from
the survey.

Let $N_u$ be the number of households with wealth above $u$, which we estimate
from the survey weights.

```{code-cell} ipython3
N_u = ω[w > u].sum()
N_u
```

Revenue is $N_u$ times the average tax paid by a household above $u$.

Using {eq}`eq:wt_tax`, this average is

$$
\mathbb{E}[h(W) \mid W > u] = \sum_{k} r_k \, \mathbb{E}[(W - t_k)^+ \mid W > u]
$$

For a Pareto tail above $u$ and a threshold $t \geq u$, a short calculation
shows that

$$
\mathbb{E}[(W - t)^+ \mid W > u]
= \int_t^\infty (x - t) \, \frac{\alpha u^\alpha}{x^{\alpha + 1}} \, dx
= \frac{u^\alpha t^{1 - \alpha}}{\alpha - 1}
$$ (eq:wt_excess)

provided that α > 1.

(When α ≤ 1 the integral is infinite, so the expected tax is infinite too.)

Putting these together gives the Pareto estimate of revenue:

$$
\hat T_P = N_u \sum_k r_k \frac{u^{\hat \alpha} t_k^{1 - \hat \alpha}}{\hat \alpha - 1}
$$

```{code-cell} ipython3
def expected_excess(t, α, u):
    "Expected value of (W - t)^+ for a Pareto tail above u, with t >= u."
    return u**α * t**(1 - α) / (α - 1)

def pareto_revenue(α, u, N_u,
                   thresholds=thresholds, rate_increases=rate_increases):
    "Pareto estimate of revenue, for a tail above u containing N_u households."
    return N_u * np.sum(rate_increases * expected_excess(thresholds, α, u))

T_pareto = pareto_revenue(α_hat, u, N_u)
print(f"Pareto estimate: ${T_pareto / 1e9:.0f} billion per year")
```


## Comparing the two estimates

The Pareto estimate is higher than the survey estimate.

To see where the difference comes from, let's break revenue down by bracket.

The revenue from bracket $k$ is the tax on wealth between $t_k$ and
$t_{k+1}$, at the marginal rate for that bracket.

```{code-cell} ipython3
marginal_rates = np.cumsum(rate_increases)
upper = np.append(thresholds[1:], np.inf)

survey_by_bracket = [np.sum(ω * m * np.clip(np.minimum(w, t_up) - t, 0, None))
                     for t, t_up, m in zip(thresholds, upper, marginal_rates)]

def excess_between(t, t_up, α, u):
    "Expected wealth between t and t_up, for a Pareto tail above u."
    top = expected_excess(t_up, α, u) if np.isfinite(t_up) else 0.0
    return expected_excess(t, α, u) - top

pareto_by_bracket = [N_u * m * excess_between(t, t_up, α_hat, u)
                     for t, t_up, m in zip(thresholds, upper, marginal_rates)]

table = pd.DataFrame({'survey': survey_by_bracket,
                      'Pareto': pareto_by_bracket},
                     index=['$10M-$50M at 1%', '$50M-$1B at 2%', 'above $1B at 6%'])
table.loc['total'] = table.sum()
(table / 1e9).round(0)     # billions of dollars per year
```

In the first bracket, where the survey has hundreds of households, the two
estimates are almost identical.

This is reassuring, since it tells us that the Pareto model describes the part
of the tail that the survey observes well.

The estimates diverge as we move up, and in the top bracket the Pareto
estimate is several times larger.

This is exactly where the survey is thin and then stops, and where our
concern about missing households applies.


## A check against the Forbes 400

The Pareto model makes predictions about households that are not in the
survey, and we can check them against the Forbes 400.

According to the model, the number of households with wealth above the Forbes
cutoff $c = 2.7$ billion dollars is $N_u (u / c)^{\alpha}$.

The average wealth of these households is $\alpha c / (\alpha - 1)$, which
follows from {eq}`eq:wt_excess` by setting $u = t = c$ and adding $c$.

```{code-cell} ipython3
n_forbes_model = N_u * (u / forbes_cutoff)**α_hat
wealth_forbes_model = n_forbes_model * α_hat * forbes_cutoff / (α_hat - 1)

pd.DataFrame({'Pareto model': [n_forbes_model, wealth_forbes_model / 1e12],
              'Forbes': [forbes_count, forbes_wealth / 1e12]},
             index=['households above $2.7B', 'their wealth ($ trillion)']).round(1)
```

The model predicts somewhat fewer households at this level than Forbes finds,
and considerably less wealth, but it gets the orders of magnitude right.

This is a demanding test, since the model was estimated entirely from survey
households worth less than 2.4 billion dollars.

We can see this graphically, by extending the fitted Pareto line beyond the
survey data and adding the Forbes 400 as a single point.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fitted Pareto tail and the Forbes 400
    name: fig:wt-forbes
---
N = ω.sum()
x_fit = np.geomspace(u, 1e11, 100)

fig, ax = plt.subplots()
ax.loglog(x_grid, weighted_ccdf(x_grid, w, ω), lw=2, label='survey')
ax.loglog(x_fit, (N_u / N) * (u / x_fit)**α_hat, 'k--', lw=2,
          label='fitted Pareto tail')
ax.loglog(forbes_cutoff, forbes_count / N, 'o', ms=8, label='Forbes 400')
ax.set_ylim(1e-9, 1)
ax.set_xlabel('net worth (US$)')
ax.set_ylabel('fraction of households above')
ax.legend()
plt.show()
```

The Forbes point lies a little above the fitted line.

There are several possible reasons.

* The estimate of α is uncertain, and a slightly smaller value would pass
  through the Forbes point.
* The very top of the distribution may have a heavier tail than the range
  from which we estimated α.
* Rich households may be less likely to respond to the survey, which biases
  estimates of α upward {cite}`vermeulen2018fat`.
* Forbes counts individuals and families using its own valuations, which
  differ from the survey's definition of a household's net worth.

The comparison suggests that, if anything, our Pareto estimate of revenue is
conservative.


## How much does α matter?

The formula for $\hat T_P$ contains the term $\hat \alpha - 1$ in the
denominator.

So, as α falls toward one, estimated revenue rises without bound.

Let's plot revenue against α.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Pareto estimate of revenue against the tail index
    name: fig:wt-revenue-alpha
---
α_values = np.linspace(1.3, 2.0, 100)
revenue = [pareto_revenue(a, u, N_u) for a in α_values]

fig, ax = plt.subplots()
ax.plot(α_values, np.array(revenue) / 1e9, lw=2, label='Pareto estimate')
ax.axhline(T_survey / 1e9, color='C1', ls='--', lw=2, label='survey estimate')
ax.axvline(α_hat, color='k', ls=':', lw=1)
ax.set_xlabel(r'tail index $\alpha$')
ax.set_ylabel('revenue (US$ billion per year)')
ax.legend()
plt.show()
```

Moving α from 1.6 to 1.4, a change well within the range of our estimates in
{numref}`fig:wt-alpha-u`, roughly doubles estimated revenue.

This sensitivity is one reason why estimates of wealth tax revenue are so
contested.

Finally, all of our estimates are *mechanical*: they assume that households do
not change their behavior in response to the tax.

In practice, wealthy households may respond by moving assets, changing how
assets are valued, or avoiding the tax in other ways, and the size of these
responses is a central question in the policy debate {cite}`saez2019progressive`.


## Exercises

```{exercise-start}
:label: wt_ex_warren
```

Warren's original proposal taxed net worth above 50 million dollars at 2% and
net worth above one billion dollars at 3%.

Compute the survey estimate and the Pareto estimate of revenue under this
schedule.

Compare them with the Saez and Zucman estimate of about 275 billion dollars per
year (2.75 trillion dollars over ten years).

```{exercise-end}
```

```{solution-start} wt_ex_warren
:class: dropdown
```

We pass the new thresholds and rate increases to our functions.

```{code-cell} ipython3
warren = dict(thresholds=np.array([50e6, 1e9]),
              rate_increases=np.array([0.02, 0.01]))

T_s = np.sum(ω * h(w, **warren))
T_p = pareto_revenue(α_hat, u, N_u, **warren)
print(f"survey estimate: ${T_s / 1e9:.0f} billion per year")
print(f"Pareto estimate: ${T_p / 1e9:.0f} billion per year")
```

Both estimates are of the same magnitude as the Saez and Zucman figure.

Their estimate differs from ours in several ways: it covers the years 2019 to
2028, uses different data for the top of the distribution, and includes an
allowance for tax avoidance.

The comparison is therefore only a rough one.

```{solution-end}
```

```{exercise-start}
:label: wt_ex_bootstrap
```

The estimate $\hat \alpha$ is subject to sampling variation.

One way to measure this is the **bootstrap**: draw a new sample of households,
with replacement, from the survey households, recompute the estimate, and
repeat many times.

Since each household appears in five rows, resample households (identified by
`yy1`) rather than rows.

Use 200 bootstrap samples to compute a 90% interval for $\hat \alpha$ at
$u = 10$ million dollars, and the corresponding interval for $\hat T_P$.

```{exercise-end}
```

```{solution-start} wt_ex_bootstrap
:class: dropdown
```

```{code-cell} ipython3
rng = np.random.default_rng(1234)
rows_by_household = scf.groupby('yy1').indices
households = np.array(list(rows_by_household.keys()))

α_boot, T_boot = [], []
for _ in range(200):
    draw = rng.choice(households, size=len(households))
    idx = np.concatenate([rows_by_household[hh] for hh in draw])
    w_b, ω_b = w[idx], ω[idx]
    a = tail_index(u, w_b, ω_b)
    α_boot.append(a)
    T_boot.append(pareto_revenue(a, u, ω_b[w_b > u].sum()))

print("90% interval for α:", np.percentile(α_boot, [5, 95]).round(2))
print("90% interval for revenue ($ billion):",
      (np.percentile(T_boot, [5, 95]) / 1e9).round(0))
```

The interval for α is roughly ±0.15 around the estimate, which is consistent
with the fluctuations in {numref}`fig:wt-alpha-u`.

The interval for revenue is wide, reflecting the sensitivity to α discussed
above.

```{solution-end}
```

```{exercise-start}
:label: wt_ex_hybrid
```

The survey estimate and the Pareto estimate agree in the first bracket, where
the survey has many observations.

This suggests a **hybrid** estimate: use the survey for households below 50
million dollars and the Pareto model (still fitted above $u = 10$ million
dollars) for the rest.

Compute it.

Hint: a household with wealth $W > c = 50$ million dollars pays
$0.01 (c - 10\text{ million})$ on the first bracket, plus
$0.02 (W - c) + 0.04 (W - 1 \text{ billion})^+$.

```{exercise-end}
```

```{solution-start} wt_ex_hybrid
:class: dropdown
```

Averaging the hint over the Pareto tail and multiplying by the number of
households, the revenue from households above $c$ is

$$
N_c \cdot 0.01 (c - 10\text{ million})
+ N_u \left( 0.02 \, \mathbb{E}[(W - c)^+ \mid W > u]
            + 0.04 \, \mathbb{E}[(W - 1\text{ billion})^+ \mid W > u] \right)
$$

where $N_c = N_u (u / c)^\alpha$ is the number of households above $c$.

```{code-cell} ipython3
c = 50e6
below = w <= c
T_below = np.sum(ω[below] * h(w[below]))

N_c = N_u * (u / c)**α_hat
T_above = (N_c * 0.01 * (c - 10e6)
           + N_u * (0.02 * expected_excess(c, α_hat, u)
                    + 0.04 * expected_excess(1e9, α_hat, u)))

print(f"hybrid estimate: ${(T_below + T_above) / 1e9:.0f} billion per year")
```

The hybrid estimate lies between the survey and Pareto estimates, and is close
to the Pareto estimate, since most of the difference between the two comes from
the top brackets.

```{solution-end}
```
