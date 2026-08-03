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

(fitting_distributions)=
# Fitting Distributions to Data

```{index} single: Fitting Distributions
```

## Outline

In {doc}`prob_dist` we studied a collection of common probability distributions.

In {doc}`observed_distributions` we studied observed data.

In this lecture we connect the two, by asking a question that arises constantly
in applied work:

*given a data set, which probability distribution should we use to describe it?*

The question has two parts.

First, we must choose a **family** --- normal, lognormal, gamma, Poisson, and
so on.

Second, having chosen a family, we must choose the **parameters** that make it
match our data as closely as possible.

We take the second part first, since it turns out to be easier.

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade yfinance
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats
import statsmodels.api as sm

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

Let's use the Ames house price data that we met in {doc}`observed_distributions`.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/ames_house_prices.csv')
houses = pd.read_csv(url)
price = houses['price']
```


## The method of moments

Suppose we have settled on a family of distributions and now want to choose its
parameters.

One simple and general strategy is the **method of moments**.

If the family has $k$ parameters, we

1. compute the first $k$ sample moments of the data,
2. write down the corresponding population moments as functions of the
   parameters, and
3. choose the parameters that make the two sets of numbers equal.

In effect we are asking the distribution to reproduce the features of the data
that we consider most important.

We have already used this idea once, in {doc}`prob_dist`, when we fitted a
normal distribution to the heights of US adults.

The normal distribution has two parameters, so we used two moments: we set
$\mu$ equal to the sample mean and $\sigma$ equal to the sample standard
deviation.

Let's apply the same idea to the house price data, using two families that live
on $(0, \infty)$ and hence respect the fact that prices are positive.

The **lognormal** distribution has parameters $\mu$ and $\sigma$, with

$$
\mathbb{E}[X] = \exp \left( \mu + \frac{\sigma^2}{2} \right)
\qquad \text{and} \qquad
\mathbb{V}[X] = \left[ \exp(\sigma^2) - 1 \right] \exp(2\mu + \sigma^2)
$$

Setting these equal to the sample mean $\bar x$ and sample variance $s^2$ and
solving gives

$$
\hat \sigma^2 = \ln \left( 1 + \frac{s^2}{\bar x^2} \right)
\qquad \text{and} \qquad
\hat \mu = \ln \bar x - \frac{\hat \sigma^2}{2}
$$

The **gamma** distribution has parameters $\alpha$ and $\beta$, with mean
$\alpha / \beta$ and variance $\alpha / \beta^2$.

Solving in the same way is easier here:

$$
\hat \alpha = \frac{\bar x^2}{s^2}
\qquad \text{and} \qquad
\hat \beta = \frac{\bar x}{s^2}
$$

Let's implement all three fits.

```{code-cell} ipython3
def fit_normal(sample):
    return scipy.stats.norm(sample.mean(), sample.std())

def fit_lognormal(sample):
    m, v = sample.mean(), sample.var()
    σ_squared = np.log(1 + v / m**2)
    μ = np.log(m) - σ_squared / 2
    return scipy.stats.lognorm(s=np.sqrt(σ_squared), scale=np.exp(μ))

def fit_gamma(sample):
    m, v = sample.mean(), sample.var()
    return scipy.stats.gamma(a=m**2 / v, scale=v / m)
```

Each function returns a distribution object of the kind we worked with in
{doc}`prob_dist`.

Let's check that the fitted lognormal reproduces the mean and variance of the
data, as it was constructed to do.

```{code-cell} ipython3
u = fit_lognormal(price)
u.mean(), price.mean()
```

```{code-cell} ipython3
u.var(), price.var()
```

Now let's plot the three fitted densities against a histogram of the data.

```{code-cell} ipython3
fits = {'normal': fit_normal(price),
        'lognormal': fit_lognormal(price),
        'gamma': fit_gamma(price)}

x_grid = np.linspace(0, price.max(), 400)

fig, ax = plt.subplots()
ax.hist(price, bins=50, density=True, alpha=0.25, color='C0')
for label, u in fits.items():
    ax.plot(x_grid, u.pdf(x_grid), lw=2, alpha=0.8, label=label)
ax.set_xlabel('sale price (US$)')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

The normal density is visibly wrong: it is symmetric, while the data are not,
and it puts weight on negative prices.

The other two look plausible.

To choose between them we need something sharper than a glance at a figure.


## Q-Q plots

A **Q-Q plot** (short for quantile-quantile plot) compares two distributions by
plotting their quantiles against each other.

To compare a sample with a fitted distribution, we sort the data

$$
x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}
$$

and note that $x_{(i)}$ is a natural estimate of the quantile of order
$(i - 0.5)/n$.

If the fitted distribution is a good description of the data, then $x_{(i)}$
should be close to the corresponding quantile of that distribution, which is

$$
F^{-1} \left( \frac{i - 0.5}{n} \right)
$$

So we plot the fitted quantiles on the horizontal axis and the sample values on
the vertical axis.

A good fit puts the points on the 45 degree line.

```{code-cell} ipython3
def qq_plot(sample, u, ax, **kwargs):
    "Plot sample quantiles against the quantiles of the distribution u."
    x_sorted = np.sort(sample)
    n = len(x_sorted)
    p = (np.arange(1, n+1) - 0.5) / n
    ax.plot(u.ppf(p), x_sorted, '.', ms=3, alpha=0.6, **kwargs)
    lo, hi = u.ppf(p[0]), u.ppf(p[-1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('fitted quantiles')
    ax.set_ylabel('sample quantiles')
```

Let's start with a case where we expect a good fit.

In {doc}`observed_distributions` we found that the heights of US adult women
have sample skewness and excess kurtosis close to zero.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/us_adult_heights.csv')
heights = pd.read_csv(url)
female = heights[heights['sex'] == 'female']['height_cm']

fig, ax = plt.subplots()
qq_plot(female, fit_normal(female), ax)
ax.set_title('female heights against a fitted normal')
plt.show()
```

The points lie almost exactly on the line, apart from mild wandering at the two
ends, where the sample contains few observations and the quantile estimates are
noisy.

Now let's try the house prices against a fitted normal.

```{code-cell} ipython3
fig, ax = plt.subplots()
qq_plot(price, fit_normal(price), ax)
ax.set_title('house prices against a fitted normal')
plt.show()
```

This is a very different picture.

The points curve away from the line, and the departure has a clear meaning:
towards the right, the sample quantiles are far larger than the fitted ones, so
the data have a longer right tail than the normal distribution allows.

The shape of a departure tells us how the fit fails.

- Points curving upwards, as here, indicate skewness to the right.
- Points forming an S-shape --- below the line on the left, above it on the
  right --- indicate that *both* tails of the data are heavier than the fitted
  distribution.

Let's check the second case by taking logs, which we know makes the house price
data roughly symmetric.

```{code-cell} ipython3
log_price = np.log(price)

fig, ax = plt.subplots()
qq_plot(log_price, fit_normal(log_price), ax)
ax.set_title('log house prices against a fitted normal')
plt.show()
```

The curvature is gone, confirming what the sample skewness told us in
{doc}`observed_distributions`.

The `statsmodels` package provides `sm.qqplot`, which produces such figures in
one line.

By default it compares the data with a normal distribution, which is by far the
most common use.

```{code-cell} ipython3
sm.qqplot(log_price, line='45', fit=True)
plt.show()
```

The axes differ from ours because `statsmodels` standardizes the data, but the
message is the same.


## The Kolmogorov-Smirnov statistic

Q-Q plots are informative but they require us to judge a picture.

Sometimes we want a single number that measures how far the data are from a
fitted distribution.

One natural measure compares the ECDF of the data, which we met in
{doc}`observed_distributions`, with the CDF of the fitted distribution.

The **Kolmogorov-Smirnov statistic** is the largest vertical gap between them:

$$
D = \max_x \, | F_n(x) - F(x) |
$$

Since $F_n$ only jumps at the observations, we can compute $D$ by checking the
gap immediately before and after each jump.

```{code-cell} ipython3
def ks_statistic(sample, u):
    "Largest vertical distance between the ECDF of the sample and the CDF of u."
    x_sorted = np.sort(sample)
    n = len(x_sorted)
    F = u.cdf(x_sorted)
    above = np.arange(1, n+1) / n - F     # gap just after each jump
    below = F - np.arange(0, n) / n       # gap just before each jump
    return max(above.max(), below.max())
```

Let's see what it measures, by drawing the ECDF and the fitted CDF for the log
prices together with the gap that attains the maximum.

```{code-cell} ipython3
u = fit_normal(log_price)
x_sorted = np.sort(log_price)
n = len(x_sorted)
F = u.cdf(x_sorted)

# locate the largest gap
gaps = np.maximum(np.arange(1, n+1) / n - F, F - np.arange(0, n) / n)
i = gaps.argmax()

fig, ax = plt.subplots()
ax.step(x_sorted, np.arange(1, n+1) / n, where='post', label='ECDF')
x_grid = np.linspace(x_sorted[0], x_sorted[-1], 200)
ax.plot(x_grid, u.cdf(x_grid), 'k--', alpha=0.7, label='fitted normal CDF')
ax.vlines(x_sorted[i], F[i], (i+1) / n, color='C3', lw=3, label='largest gap')
ax.set_xlabel('log of sale price')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

```{code-cell} ipython3
ks_statistic(log_price, u)
```

The statistic is small, which tells us that the ECDF never strays far from the
fitted CDF.

```{note}
You might expect that we can now *test* whether the data came from the fitted
distribution, by asking whether $D$ is larger than chance alone would produce.

This is exactly what the Kolmogorov-Smirnov test does, and `scipy` implements
it as `scipy.stats.kstest`.

We do not pursue it here, because it requires knowing how $D$ behaves when the
distribution really is correct, which needs more theory than we have developed.

There is also a trap: the usual theory assumes that the distribution is
specified in advance, whereas we chose its parameters using the same data.
```


## Choosing between families

We now have a way to choose between candidate families.

For each family, we fit the parameters by the method of moments and then
compute $D$.

The family with the smallest $D$ is the one whose CDF stays closest to the data.

Let's apply this to the house prices.

```{code-cell} ipython3
results = pd.Series({label: ks_statistic(price, u) for label, u in fits.items()})
results.sort_values()
```

The lognormal distribution wins, with the gamma second and the normal a distant
third.

This agrees with what we found in {doc}`observed_distributions`, where taking
logs of the price data produced a sample skewness of almost exactly zero.

Here are the three fitted CDFs against the ECDF of the data, which shows the
same ranking.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.step(np.sort(price), np.arange(1, len(price)+1) / len(price),
        where='post', color='k', lw=2, label='ECDF')
x_grid = np.linspace(price.min(), price.max(), 400)
for label, u in fits.items():
    ax.plot(x_grid, u.cdf(x_grid), lw=2, alpha=0.7, label=label)
ax.set_xlabel('sale price (US$)')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

Three warnings are in order.

First, the comparison is only fair when the families have the same number of
parameters, as they do here.

A family with more parameters can bend itself closer to any data set, and $D$
does not charge it for the privilege.

In particular, if one family is a special case of another, the larger family
can never do worse.

Second, $D$ is most sensitive in the middle of the distribution, where the CDF
is changing quickly, and least sensitive in the tails.

If we care mainly about extreme outcomes, as we often do in economics and
finance, then a small $D$ can be misleading.

Third, the winner is only the best of the candidates we happened to try.

Nothing here tells us that the winning family is a good description of the data
--- only that it is better than the alternatives.

We take up that point below.


## Count data

So far our data have been continuous.

The method of moments applies just as well to discrete data.

Consider the **Poisson** distribution, which we met in {doc}`prob_dist` as a
model for the number of events in a fixed interval.

It has a single parameter $\lambda$, and its mean is $\lambda$, so the method
of moments gives

$$
\hat \lambda = \bar x
$$

Let's try it on the number of goals scored in football matches.

The data set contains the full-time score of every match in ten seasons of the
English Premier League.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/epl_match_goals.csv')
matches = pd.read_csv(url)
matches.head()
```

We are interested in the total number of goals scored in each match.

```{code-cell} ipython3
goals = matches['home_goals'] + matches['away_goals']
len(goals), goals.mean()
```

The Poisson distribution has the unusual property that its variance equals its
mean.

This gives us a diagnostic that we can apply before fitting anything.

```{code-cell} ipython3
goals.mean(), goals.var()
```

These are close, which is encouraging.

Let's fit the distribution and compare the fitted probabilities with the
observed frequencies.

```{code-cell} ipython3
u = scipy.stats.poisson(goals.mean())

counts = goals.value_counts().sort_index()
frequencies = counts / counts.sum()
S = np.arange(counts.index.max() + 1)

fig, ax = plt.subplots()
ax.bar(counts.index, frequencies, alpha=0.4, label='observed frequency')
ax.plot(S, u.pmf(S), linestyle='', marker='o', color='C1', label='fitted Poisson')
ax.vlines(S, 0, u.pmf(S), lw=0.5, color='C1')
ax.set_xlabel('goals per match')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

The fit is good.

This is a well-known empirical regularity, and the reason for it is worth
stating: goals are rare events, arising at a roughly constant rate over the
course of a match, and largely independently of each other.

Those are exactly the conditions under which the Poisson distribution arises.


## When nothing fits

The methods above always return an answer.

It is important to remember that the answer can be bad.

Let's return to the monthly returns on Amazon shares, which we studied in
{doc}`observed_distributions`.

```{code-cell} ipython3
:tags: [hide-output]

data = yf.download('AMZN', '2000-1-1', '2024-1-1', interval='1mo')
prices = data['Close']['AMZN']
returns = prices.pct_change().dropna() * 100
```

Returns take both signs, so of our continuous families only the normal is
available.

Let's look at the Q-Q plot.

```{code-cell} ipython3
fig, ax = plt.subplots()
qq_plot(returns, fit_normal(returns), ax)
ax.set_title('Amazon monthly returns against a fitted normal')
plt.show()
```

This is the S-shape described above: the smallest returns are more negative
than the fitted normal predicts, and the largest are more positive.

In other words, both tails of the data are heavier than the normal distribution
allows.

Now let's compute the KS statistic.

```{code-cell} ipython3
ks_statistic(returns, fit_normal(returns))
```

Taken on its own, the number looks unremarkable.

```{note}
Values of $D$ should not be compared across data sets of different size.

Even when the fitted distribution is exactly right, $D$ shrinks as $n$ grows,
so a small value from a small sample means less than the same value from a
large one.

The point here is not that $D$ is smaller or larger than some earlier number,
but that it gives no hint of the trouble that the Q-Q plot displays so
plainly.
```

This illustrates the second warning above.

The normal distribution describes the middle of the return data reasonably
well, and that is the region the KS statistic looks at.

The failure is in the tails, and the tails are precisely what a study of asset
returns cares about, since they contain the large losses.

The lesson is that a single summary number is never a substitute for looking at
the data.

Distributions with heavier tails than the normal are the subject of
{doc}`heavy_tails`.


## Exercises

The next data set records every earthquake of magnitude 5 and above in the
region around Japan between 2000 and 2024, as reported by the
[US Geological Survey](https://earthquake.usgs.gov/earthquakes/search/).

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/japan_earthquakes.csv')
quakes = pd.read_csv(url)
quakes.head()
```

```{exercise}
:label: fit_ex1

Earthquakes are often modeled as arriving randomly and independently at some
constant rate.

If that is true, then the time between successive earthquakes is exponentially
distributed.

Using the data above, compute the time in days between successive earthquakes,
fit an exponential distribution by the method of moments, and assess the fit.

(The exponential distribution has one parameter $\lambda$ and mean $1/\lambda$.)

Does the model hold up?
```

```{solution-start} fit_ex1
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
times = pd.to_datetime(quakes['time'], format='ISO8601')
gaps = times.diff().dropna().dt.total_seconds() / (60 * 60 * 24)

u = scipy.stats.expon(scale=gaps.mean())

fig, ax = plt.subplots()
qq_plot(gaps, u, ax)
ax.set_title('time between earthquakes against a fitted exponential')
plt.show()
```

The fit fails badly, with the largest gaps far longer than the exponential
distribution predicts.

A second diagnostic makes the problem clearer.

For the exponential distribution the standard deviation equals the mean.

```{code-cell} ipython3
gaps.mean(), gaps.std()
```

The standard deviation is around half again the mean, so the data are much more
variable than the model allows.

The reason is that earthquakes are *not* independent.

Large earthquakes are followed by aftershocks, so events arrive in clusters,
with long quiet periods in between.

We can see this directly by counting events per month.

```{code-cell} ipython3
monthly = times.dt.to_period('M').value_counts().sort_index()
monthly.mean(), monthly.var()
```

If the arrivals were Poisson, these two numbers would be roughly equal.

Instead the variance is many times the mean, which is the signature of
clustering.

This connects to the discussion of independence in
{doc}`observed_distributions`: a sample tells us about a distribution only when
its observations bring new information, and an aftershock tells us mostly what
the main shock already did.

```{solution-end}
```

```{exercise}
:label: fit_ex2

In {doc}`observed_distributions` we found that the ages at death in Japan have
sample skewness of about $-1.6$.

Fit a normal distribution to that data by the method of moments and use a Q-Q
plot to display the failure.

Which way do the points bend, and why?
```

```{solution-start} fit_ex2
:class: dropdown
```

Here is one solution.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/japan_deaths_by_age.csv')
deaths = pd.read_csv(url)
age_at_death = np.repeat(deaths['age'], deaths['deaths_total'])

fig, ax = plt.subplots()
qq_plot(age_at_death, fit_normal(age_at_death), ax)
ax.set_title('age at death against a fitted normal')
plt.show()
```

The points bend downwards, which is the mirror image of the house price figure.

At the left-hand end the sample quantiles fall well below the fitted ones,
because the data have a long left tail of deaths at young ages that the normal
distribution cannot reproduce.

The figure also shows a second, unrelated problem: the points run flat along
$y = 100$ at the right-hand end.

That is the "100 and over" category discussed in {doc}`observed_distributions`,
which caps every age at 100.

A Q-Q plot displays such recording conventions very clearly, since they show up
as flat segments where many observations share a single value.

```{solution-end}
```
