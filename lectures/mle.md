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

# Maximum Likelihood Estimation

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade yfinance
```

```{code-cell} ipython3
from scipy.stats import lognorm, pareto, expon
from scipy.stats import t as student_t
from scipy.stats import kurtosis
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
```

## Overview

Maximum likelihood estimation is a method for choosing the parameters of a
statistical model.

The idea is simple: among all candidate parameter values, choose the one under
which the data we actually observed are most probable.

Put differently, we treat the probability that the model assigns to the data as
a function of the parameters, and then maximize it.

This principle is one of the foundations of modern statistics.

It is used to fit distributions, as in this lecture, and also regression models,
macroeconomic models, and models in physics, biology and machine learning.

(For example, training a classifier by minimizing cross-entropy loss is maximum
likelihood estimation.)

One reason for its wide use is generality: once we have a model that assigns
probabilities to data, maximum likelihood tells us how to fit it, with no need
to invent a new method for each model.

Another is that, under fairly general conditions, maximum likelihood estimates
converge to the true parameter values as the sample grows, and in large samples
no other method gives more precise estimates.

The idea has a long history.

Daniel Bernoulli and Carl Friedrich Gauss used versions of it in the late 18th
and early 19th centuries, and Gauss justified the method of least squares by
showing that, when errors are normally distributed, it picks out the most
probable parameter values.

The method was developed systematically, and given its name, by Ronald Fisher
between 1912 and 1922 {cite}`fisher1922mathematical`.

{cite:t}`aldrich1997fisher` and {cite:t}`stigler2007epic` tell the story.

In this lecture we apply maximum likelihood to a policy problem: estimating the
revenue raised by a wealth tax.

We also compare it with the method of moments, which we studied in
{doc}`fitting_distributions`, and see why maximum likelihood is the standard
choice when data have heavy tails.


## A wealth tax

Consider a situation where a policymaker is trying to estimate how much revenue
a proposed wealth tax will raise.

The proposed tax is 

$$
    h(w) = 
    \begin{cases}
    a w                       & \text{if } w \leq \bar w  \\
    a \bar{w} + b (w-\bar{w}) & \text{if } w > \bar w  
    \end{cases}
$$ 

where $w$ is wealth.

```{prf:example}
:label: mle_ex_wt

For example, if $a = 0.05$, $b = 0.1$, and $\bar w = 2.5$, this means 

* a 5% tax on wealth up to 2.5 and 
* a 10% tax on wealth in excess of 2.5.

The unit is 100,000, so $w= 2.5$ means 250,000 dollars.
```
Let's go ahead and define $h$:

```{code-cell} ipython3
def h(w, a=0.05, b=0.1, w_bar=2.5):
    if w <= w_bar:
        return a * w
    else:
        return a * w_bar + b * (w - w_bar)
```

For a population of size $N$, where individual $i$ has wealth $w_i$, total revenue raised by 
the tax will be 

$$
    T = \sum_{i=1}^{N} h(w_i)
$$

We wish to calculate this quantity.

The problem we face is that, in most countries, wealth is not observed for all individuals.

Collecting and maintaining accurate wealth data for all individuals or households in a country
is just too hard.

So let's suppose instead that we obtain a sample $w_1, w_2, \cdots, w_n$ telling us the wealth of $n$ randomly selected individuals.

For our exercise we are going to use a sample of $n = 10,000$ observations from wealth data in the US in 2016.

```{code-cell} ipython3
n = 10_000
```

The data is derived from the
[Survey of Consumer Finances](https://en.wikipedia.org/wiki/Survey_of_Consumer_Finances) (SCF).


The following code imports this data  and reads it into an array called `sample`.

```{code-cell} ipython3
:tags: [hide-input]

url = 'https://raw.githubusercontent.com/QuantEcon/data-lectures/main/lectures/SCF_plus_mini_no_weights.csv'
df = pd.read_csv(url)
df = df.dropna()
df = df[df['year'] == 2016]
df = df.loc[df['n_wealth'] > 1 ]   # restricting data to net worth > 1
rv = df['n_wealth'].sample(n=n, random_state=1234)
rv = rv.to_numpy() / 100_000
sample = rv
```

Let's histogram this sample.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Histogram of US household wealth
    name: fig:mle-wealth-hist
---
fig, ax = plt.subplots()
ax.set_xlim(-1, 20)
density, edges = np.histogram(sample, bins=5000, density=True)
prob = density * np.diff(edges)
ax.stairs(prob, edges, fill=True, alpha=0.8, label=r"unit: $\$100,000$")
ax.set_ylabel("prob")
ax.set_xlabel("net wealth")
ax.legend()
plt.show()
```

The histogram shows that many people have very low wealth and a few people have
very high wealth.


We will take the full population size to be

```{code-cell} ipython3
N = 100_000_000
```

How can we estimate total revenue from the full population using only the sample data?

Our plan is to assume that wealth of each individual is a draw from a distribution with density $f$.

If we obtain an estimate of $f$ we can then approximate $T$ as follows:

$$
    T = \sum_{i=1}^{N} h(w_i) 
      = N \frac{1}{N} \sum_{i=1}^{N} h(w_i) 
      \approx N \int_{0}^{\infty} h(w)f(w) dw
$$ (eq:est_rev)

(The sample mean should be close to the mean by the law of large numbers.)

The problem now is: how do we estimate $f$?


## Maximum likelihood estimation

[Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) 
is a method of estimating an unknown distribution.

Maximum likelihood estimation has two steps:

1. Guess what the underlying distribution is (e.g., normal with mean $\mu$ and
   standard deviation $\sigma$).
2. Estimate the parameter values (e.g., estimate $\mu$ and $\sigma$ for the
   normal distribution)

These are the same two steps discussed in {doc}`fitting_distributions`, which
chooses parameters by the method of moments instead, and which also shows how
to judge the resulting fit.

One possible assumption for the wealth is that each
$w_i$ is [log-normally distributed](https://en.wikipedia.org/wiki/Log-normal_distribution),
with parameters $\mu \in (-\infty,\infty)$ and $\sigma \in (0,\infty)$.

(This means that $\ln w_i$ is normally distributed with mean $\mu$ and standard deviation $\sigma$.)

You can see that this assumption is not completely unreasonable because, if we
histogram log wealth instead of wealth, the picture starts to look something
like a bell-shaped curve.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Histogram of log wealth
    name: fig:mle-log-wealth-hist
---
ln_sample = np.log(sample)
fig, ax = plt.subplots()
ax.hist(ln_sample, density=True, bins=200, histtype='stepfilled', alpha=0.8)
ax.set_xlabel("log of net wealth")
ax.set_ylabel("density")
plt.show()
```

Now our job is to obtain the maximum likelihood estimates of $\mu$ and $\sigma$, which
we denote by $\hat{\mu}$ and $\hat{\sigma}$.

These estimates can be found by maximizing the likelihood function given the
data.

The pdf of a lognormally distributed random variable $X$ is given by:

$$
    f(x; \mu, \sigma) 
    = \frac{1}{x \sigma \sqrt{2\pi}} 
    \exp \left( - \frac{(\ln x - \mu)^2}{2 \sigma^2} \right)
$$

For our sample $w_1, w_2, \cdots, w_n$, the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) is given by

$$
    L(\mu, \sigma) = \prod_{i=1}^{n} f(w_i; \mu, \sigma)
$$

The likelihood function can be viewed as both

* the joint distribution of the sample (which is assumed to be IID) and
* the "likelihood" of parameters $(\mu, \sigma)$ given the data.

Taking logs on both sides gives us the log likelihood function, which is

$$
\begin{aligned}
    \ell(\mu, \sigma) 
    & = \ln \left[ \prod_{i=1}^{n} f(w_i; \mu, \sigma) \right] \\
    & = -\sum_{i=1}^{n} \ln w_i 
        - \frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln \sigma^2 - \frac{1}{2\sigma^2}
            \sum_{i=1}^n (\ln w_i - \mu)^2
\end{aligned}
$$

To find where this function is maximized we find its partial derivatives with respect to $\mu$ and $\sigma^2$ and set them to zero.

Let's first find the maximum likelihood estimate (MLE) of $\mu$

$$
\begin{aligned}
\frac{\partial \ell}{\partial \mu} 
    = \frac{1}{\sigma^2} \sum_{i=1}^n (\ln w_i - \mu) = 0 
    & \implies \sum_{i=1}^n \ln w_i - n \mu = 0 \\
    & \implies \hat{\mu} = \frac{1}{n} \sum_{i=1}^n \ln w_i
\end{aligned}
$$

Now let's find the MLE of $\sigma$

$$
\begin{aligned}
\frac{\partial \ell}{\partial \sigma^2} 
    = - \frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} 
    \sum_{i=1}^n (\ln w_i - \mu)^2 = 0 
    & \implies \frac{n}{2\sigma^2} = 
    \frac{1}{2\sigma^4} \sum_{i=1}^n (\ln w_i - \mu)^2 \\
    & \implies \hat{\sigma} = 
    \left( \frac{1}{n} \sum_{i=1}^{n}(\ln w_i - \hat{\mu})^2 \right)^{1/2}
\end{aligned}
$$

Now that we have derived the expressions for $\hat{\mu}$ and $\hat{\sigma}$,
let's compute them for our wealth sample.

```{code-cell} ipython3
μ_hat = np.mean(ln_sample)
μ_hat
```

```{code-cell} ipython3
num = (ln_sample - μ_hat)**2
σ_hat = (np.mean(num))**(1/2)
σ_hat
```

Let's plot the lognormal pdf using the estimated parameters against our sample data.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Lognormal fit to wealth
    name: fig:mle-lognormal-fit
---
dist_lognorm = lognorm(σ_hat, scale=np.exp(μ_hat))
x = np.linspace(0, 50, 10000)

fig, ax = plt.subplots()
ax.set_xlim(-1, 20)

ax.hist(sample, density=True, bins=5_000, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_lognorm.pdf(x), 'k-', lw=2, label='lognormal pdf')
ax.set_xlabel("net wealth")
ax.set_ylabel("density")
ax.legend()
plt.show()
```

Our estimated lognormal distribution appears to be a reasonable fit for the overall data.


### Comparison with the method of moments

Notice that $\hat{\mu}$ and $\hat{\sigma}$ are just the sample mean and sample
standard deviation of log wealth.

So, for the lognormal class, maximum likelihood gives the same answer as
fitting a normal distribution to log wealth by the method of moments.

This is *not* the same as the lognormal fit in {doc}`fitting_distributions`,
which matches the mean and variance of wealth itself, rather than of log
wealth.

Let's compute that fit too.

```{code-cell} ipython3
def fit_lognormal_mom(sample):
    "Lognormal fit matching the sample mean and variance of the data."
    m, v = sample.mean(), sample.var()
    σ_squared = np.log(1 + v / m**2)
    μ = np.log(m) - σ_squared / 2
    return lognorm(s=np.sqrt(σ_squared), scale=np.exp(μ))

dist_lognorm_mom = fit_lognormal_mom(sample)
```

To compare the two fits we use the Kolmogorov-Smirnov statistic from
{doc}`fitting_distributions`, which is the largest vertical gap between the
ECDF of the data and the fitted CDF.

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

```{code-cell} ipython3
pd.DataFrame({
    'σ': [σ_hat, dist_lognorm_mom.kwds['s']],
    'KS statistic': [ks_statistic(sample, dist_lognorm),
                     ks_statistic(sample, dist_lognorm_mom)]},
    index=['maximum likelihood', 'method of moments'])
```

The two methods give quite different parameters, and maximum likelihood gives
a much closer fit.

The reason is that the sample variance of wealth is dominated by a handful of
very wealthy households, so matching it pulls the whole fitted distribution
toward them.

Maximum likelihood works with log wealth, where these households are far less
extreme.

We now use {eq}`eq:est_rev` and the maximum likelihood fit to calculate total
revenue.

We will compute the integral using numerical integration via SciPy's
[quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
function

```{code-cell} ipython3
def total_revenue(dist):
    integral, _ = quad(lambda x: h(x) * dist.pdf(x), 0, 100_000)
    T = N * integral
    return T
```

```{code-cell} ipython3
tr_lognorm = total_revenue(dist_lognorm)
tr_lognorm
```

(Our unit was 100,000 dollars, so this means that actual revenue is 100,000
times as large.)


## Pareto distribution

We mentioned above that using maximum likelihood estimation requires us to make
a prior assumption of the underlying distribution.

Previously we assumed that the distribution is lognormal.

Suppose instead we assume that $w_i$ are drawn from the 
[Pareto Distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
with parameters $b$ and $x_m$.

In this case, the maximum likelihood estimates are known to be

$$
    \hat{b} = \frac{n}{\sum_{i=1}^{n} \ln (w_i/\hat{x}_m)}
    \quad \text{and} \quad
    \hat{x}_m = \min_{i} w_i
$$

Let's calculate them.

```{code-cell} ipython3
xm_hat = min(sample)
xm_hat
```

```{code-cell} ipython3
den = np.log(sample/xm_hat)
b_hat = 1/np.mean(den)
b_hat
```

Now let's recompute total revenue.

```{code-cell} ipython3
dist_pareto = pareto(b=b_hat, scale=xm_hat)
tr_pareto = total_revenue(dist_pareto) 
tr_pareto
```

The number is very different!

```{code-cell} ipython3
tr_pareto / tr_lognorm
```

We see that choosing the right distribution is extremely important.



Let's compare the fitted Pareto distribution to the histogram:

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Pareto fit to wealth
    name: fig:mle-pareto-fit
---
fig, ax = plt.subplots()
ax.set_xlim(-1, 20)
ax.set_ylim(0, 1.75)

ax.hist(sample, density=True, bins=5_000, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_pareto.pdf(x), 'k-', lw=2, label='Pareto pdf')
ax.set_xlabel("net wealth")
ax.set_ylabel("density")
ax.legend()

plt.show()
```

The fit is poor, and the KS statistic confirms it.

```{code-cell} ipython3
ks_statistic(sample, dist_pareto), ks_statistic(sample, dist_lognorm)
```

For the sample as a whole, the lognormal distribution is clearly preferred.


## Fitting the right tail

There is no "best" distribution --- every choice we make is an assumption.

The lognormal distribution fits the sample as a whole, but for questions about
the richest households, what matters is the fit in the upper tail.

To study the tail, let's set a minimum threshold of net worth in our dataset.

We set an arbitrary threshold of $500,000 and read the data into `sample_tail`.

```{code-cell} ipython3
:tags: [hide-input]

df_tail = df.loc[df['n_wealth'] > 500_000 ]
rv_tail = df_tail['n_wealth'].sample(n=10_000, random_state=4321)
rv_tail = rv_tail.to_numpy()
sample_tail = rv_tail/500_000
```

Let's fit a lognormal and a Pareto distribution to this data by maximum
likelihood, using the formulas derived above.

```{code-cell} ipython3
ln_sample_tail = np.log(sample_tail)
μ_hat_tail = np.mean(ln_sample_tail)
σ_hat_tail = np.std(ln_sample_tail)
dist_lognorm_tail = lognorm(σ_hat_tail, scale=np.exp(μ_hat_tail))

xm_hat_tail = min(sample_tail)
b_hat_tail = 1/np.mean(np.log(sample_tail/xm_hat_tail))
dist_pareto_tail = pareto(b=b_hat_tail, scale=xm_hat_tail)
```

Since we care about the tail, we compare the fits with Q-Q plots, which were
introduced in {doc}`fitting_distributions` and show clearly how a fit behaves
at the extremes.

Tail wealth spans several orders of magnitude, so we draw the plots on log
scales.

```{code-cell} ipython3
def qq_plot(sample, u, ax, **kwargs):
    "Plot sample quantiles against the quantiles of the distribution u."
    x_sorted = np.sort(sample)
    n = len(x_sorted)
    p = (np.arange(1, n+1) - 0.5) / n
    ax.plot(u.ppf(p), x_sorted, '.', ms=3, alpha=0.6, **kwargs)
    lo, hi = u.ppf(p[0]), u.ppf(p[-1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=2)
    ax.set_xlabel('fitted quantiles')
    ax.set_ylabel('sample quantiles')
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Q-Q plots for tail wealth
    name: fig:mle-tail-qq
---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, u, label in zip(axes,
                        (dist_lognorm_tail, dist_pareto_tail),
                        ('lognormal', 'Pareto')):
    qq_plot(sample_tail, u, ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(label)
plt.tight_layout()
plt.show()
```

The lognormal fit bends away from the 45 degree line: it puts probability on
wealth below the threshold, which cannot occur in this sample, and it
underpredicts large fortunes.

The Pareto fit tracks the data closely until far into the tail, up to around 20
times the threshold, or 10 million dollars.

Beyond that point the data flatten out, because no household in our data set
is worth more than about 50 million dollars.

This reflects the survey rather than the population, since the SCF does not
capture the very wealthiest households.

The KS statistic also favors the Pareto distribution.

```{code-cell} ipython3
ks_statistic(sample_tail, dist_lognorm_tail), ks_statistic(sample_tail, dist_pareto_tail)
```

So, while the lognormal distribution was a good fit for the entire dataset, the
Pareto distribution is a better fit for the right hand tail.


### Why not the method of moments?

Here is the estimated Pareto parameter for the tail.

```{code-cell} ipython3
b_hat_tail
```

This value is below 1, which has a striking implication.

A Pareto distribution with parameter $b$ has mean $b x_m / (b - 1)$ when
$b > 1$, and infinite mean when $b \leq 1$.

So the fitted distribution has no finite mean, let alone a finite variance.

(We discuss distributions of this kind in {doc}`heavy_tails`.)

This is a problem for the method of moments.

With $x_m$ fixed, matching the population mean to the sample mean $\bar w$ gives

$$
\hat{b} = \frac{\bar w}{\bar w - x_m}
$$

which is greater than 1 for *every* data set, since $\bar w > x_m$.

The method of moments therefore cannot recover a value of $b$ at or below 1,
however much data we have.

Let's see what it gives here.

```{code-cell} ipython3
w_bar_tail = np.mean(sample_tail)
w_bar_tail / (w_bar_tail - xm_hat_tail)
```

The trouble is that the method of moments assumes the moments it matches
exist.

Maximum likelihood makes no such assumption, since it works with the density
itself, which is well defined for every value of $b$.

This is one reason why maximum likelihood is the standard method for fitting
heavy-tailed distributions.


## Heavy-tailed returns

In {doc}`fitting_distributions` we fitted a Student's t distribution to monthly
returns on Amazon shares, using the method of moments.

We noted there that the fitted value of the degrees of freedom $\nu$ was
unreliable, because it came from the sample kurtosis, a fourth moment.

Let's fit the same class by maximum likelihood instead.

```{code-cell} ipython3
:tags: [hide-output]

data = yf.download('AMZN', '2000-1-1', '2024-1-1', interval='1mo')
prices = data['Close']['AMZN']
returns = prices.pct_change().dropna() * 100
```

Unlike the lognormal and Pareto cases, there is no closed-form expression for
the maximum likelihood estimates of the Student's t distribution.

Instead, the `fit` method in SciPy maximizes the log likelihood numerically.

```{code-cell} ipython3
ν_mle, loc_mle, scale_mle = student_t.fit(returns)
dist_t_mle = student_t(df=ν_mle, loc=loc_mle, scale=scale_mle)
ν_mle
```

For comparison, here is the method of moments fit from {doc}`fitting_distributions`.

```{code-cell} ipython3
def fit_t_mom(sample):
    m, s = sample.mean(), sample.std()
    ν = 4 + 6 / kurtosis(sample)
    return student_t(df=ν, loc=m, scale=s * np.sqrt((ν - 2) / ν))

dist_t_mom = fit_t_mom(returns)
dist_t_mom.kwds['df']
```

Maximum likelihood gives a noticeably smaller $\nu$, and hence heavier tails.

It also gives a closer fit.

```{code-cell} ipython3
ks_statistic(returns, dist_t_mom), ks_statistic(returns, dist_t_mle)
```

The Q-Q plot confirms that the maximum likelihood fit tracks the extreme
returns well.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Amazon returns against a t fitted by MLE
    name: fig:mle-qq-returns-t
---
fig, ax = plt.subplots()
qq_plot(returns, dist_t_mle, ax)
plt.show()
```

The lesson is the same as for the wealth tail: when the tails are heavy, the
sample moments are dominated by a few extreme observations, and maximum
likelihood makes better use of the data.


## Exercises

```{exercise-start}
:label: mle_ex1
```
Suppose we assume wealth is [exponentially](https://en.wikipedia.org/wiki/Exponential_distribution)
distributed with parameter $\lambda > 0$.

The maximum likelihood estimate of $\lambda$ is given by

$$
\hat{\lambda} = \frac{n}{\sum_{i=1}^n w_i}
$$

1. Compute $\hat{\lambda}$ for our initial sample.
2. Use $\hat{\lambda}$ to find the total revenue 

```{exercise-end}
```

```{solution-start} mle_ex1
:class: dropdown
```

```{code-cell} ipython3
λ_hat = 1/np.mean(sample)
λ_hat
```

```{code-cell} ipython3
dist_exp = expon(scale=1/λ_hat)
tr_expo = total_revenue(dist_exp) 
tr_expo
```

```{solution-end}
```

```{exercise-start}
:label: mle_ex2
```

Plot the exponential distribution against the sample and check if it is a good fit or not.

```{exercise-end}
```

```{solution-start} mle_ex2
:class: dropdown
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.set_xlim(-1, 20)

ax.hist(sample, density=True, bins=5000, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_exp.pdf(x), 'k-', lw=2, label='exponential pdf')
ax.legend()

plt.show()
```

Clearly, this distribution is not a good fit for our data.

```{solution-end}
```
