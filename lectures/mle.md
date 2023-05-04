---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Maximum Likelihood Estimation`

```{code-cell} ipython3
from scipy.stats import lognorm, pareto, expon, norm
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
from math import exp, log
```

+++ {"user_expressions": []}

## Introduction

Consider a situation where a policymaker is trying to estimate the revenue a proposed wealth tax
scheme will raise.

The proposed tax is $h(w)$ where $h$ is a function of wealth $w$. For example,

* $h(w) = 0.05w$ means a 5% tax on wealth.
* $h(w) = 0.05 \bar{w} + 0.10(w-\bar{w})$ means a 5% tax on wealth upto $\bar{w}$ and 10% on wealth in excess of $\bar{w}$

For a population of size $N$, the total revenue will be given by:

$$
T = \sum_{i=1}^{N} h(w_i)
$$

However, in most scenarios wealth is not observed for all individuals.

Recording observations for all individuals can be tedious especially when $N$ is very large.

Instead we obtain a sample $w_1, w_2, \cdots, w_n$ of $n$ individuals.

Suppose we draw a sample of $n = 10,000$ from real data on wealth in the US in 2016.


The following code block imports a subset of the dataset SCF_plus, which is derived from the
[Survey of Consumer Finances](https://en.wikipedia.org/wiki/Survey_of_Consumer_Finances) (SCF).

```{code-cell} ipython3
url = 'https://media.githubusercontent.com/media/QuantEcon/high_dim_data/update_scf_noweights/SCF_plus/SCF_plus_mini_no_weights.csv'
df = pd.read_csv(url)
```

```{code-cell} ipython3
df = df.dropna()
df = df[df['year'] == 2016]
df = df.loc[df['n_wealth'] > 0 ]   #restrcting data to net worth > 0
```

```{code-cell} ipython3
df.head()
```

+++ {"user_expressions": []}

We now generate our sample from the obtained dataframe.

```{code-cell} ipython3
rv = df['n_wealth'].sample(n=10_000, random_state=1234)
rv = rv.to_numpy()
sample = rv/100_000

fig, ax = plt.subplots()
ax.set_xlim(-1,20)
ax.hist(sample, density=True, bins=5_000, histtype='stepfilled', alpha=0.8)

plt.show()
```

```{code-cell} ipython3
N = 100_000_000
```

```{code-cell} ipython3
n = 10_000
```

+++ {"user_expressions": []}

How do we obtain total revenue from the sample data?

One possibility is that we assume that wealth of each individual is a draw from a distribution with density $f$.

If we obtain an estimate of $f$ we can then approximate $T$ as follows:

$$
T = \sum_{i=1}^{N} h(w_i) = N \frac{1}{N} \sum_{i=1}^{N} h(w_i) \approx N \int_{0}^{\infty} h(w)f(w) dw
$$ (eq:est_rev)

The problem now is how do we estimate f?

+++ {"user_expressions": []}

## Maximum Likelihood Estimation

Maximum Likelihood Estimation is a method of estimating the
parameters of a distribution from observed data.

Note that maximum likelihood estimation requires a prior assumption of what the distribution could be.

The theory behind MLE can be read [here](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) but our
concern in this lecture is to illustrate its applications.

Suppose we assume that $w_i$ are [log-normally distributed](https://en.wikipedia.org/wiki/Log-normal_distribution)
with parameters $\mu \in (-\infty,\infty)$ and $\sigma \in (0,\infty)$.

We wish to obtain the maximum likelihood estimates $\hat{\mu}$ and $\hat{\sigma}$ given by:
$$
\hat{\mu} = \frac{\sum_{i=1}^{n} \ln w_i}{n}
\text{ , }
\hat{\sigma^2} = \frac{\sum_{i=1}^{n}(\ln w_i - \hat{\mu})^2}{n}
$$

```{code-cell} ipython3
ln_sample = np.log(sample)
```

```{code-cell} ipython3
μ_hat = np.mean(ln_sample)
μ_hat
```

```{code-cell} ipython3
num = (ln_sample - μ_hat)**2
σ_hat = (np.mean(num))**(1/2)
σ_hat
```

Let's plot the log-normal pdf using the estimated parameters against our sample data.

```{code-cell} ipython3
dist_lognorm = lognorm(σ_hat, scale = exp(μ_hat))
x = np.linspace(0,50,10000)

fig, ax = plt.subplots()
ax.set_xlim(-1,20)

ax.hist(sample, density=True, bins=5_000, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_lognorm.pdf(x), 'k-', lw=0.5, label='lognormal pdf')
ax.legend()
plt.show()
```

+++ {"user_expressions": []}

Our estimated lognormal distribution appears to be a decent fit for the overall data.

We now use {eq}`eq:est_rev` to calculate total revenue.

Let $g(w) = h(w)f(w)$

```{code-cell} ipython3
def total_revenue(dist, N):
    def g(x):
        return 0.05 * x * dist.pdf(x)
    q = quad(g, 0, 100_000_000)
    T = N*q[0]
    return T
```

```{code-cell} ipython3
total_revenue(dist_lognorm, 100)
```

+++ {"user_expressions": []}

## Pareto Distribution

We discussed below that MLE requires a prior assumption of the underlying distribution.

Suppose instead we assume that $w_i$ are drawn from the [Pareto Distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
with parameters $b>0$ and $x_m >0$.

The maximum likelihood estimates are given by:

$$
\hat{b} = \frac{n}{\sum_{i=1}^{n} \ln (w_i/\hat{x_m})}
,\;
\hat{x_m} = \min_{i} w_i
$$

```{code-cell} ipython3
xm_hat = min(sample)
xm_hat
```

```{code-cell} ipython3
den = np.log(sample/xm_hat)
b_hat = 1/np.mean(den)
b_hat
```

```{code-cell} ipython3
dist_pareto = pareto(b = b_hat, scale = xm_hat)

fig, ax = plt.subplots()
ax.set_xlim(-1, 20)
ax.set_ylim(0,1.75)

ax.hist(sample, density=True, bins=5_000, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_pareto.pdf(x), 'k-', lw=0.5, label='pareto pdf')
ax.legend()

plt.show()
```

+++ {"user_expressions": []}

We observe that the lognormal distribution was a better fit for this data
as compared to the pareto distribution.

```{code-cell} ipython3
total_revenue(dist_pareto, 100)
```

+++ {"user_expressions": []}

## Distribution of Right Hand Tail

The existing literature on distribution of wealth seems to suggest that the lognormal
distribution is a better fit for the entire distrubtion.

However when the data is truncated at the upper tail the pareto distribution
may be a better fit.

Suppose we now set a minimum threshold of net worth in our dataset.

We set an arbitrary threshold of $500,000.

```{code-cell} ipython3
df_tail = df.loc[df['n_wealth'] > 500_000 ]
df_tail.head()
```

```{code-cell} ipython3
rv_tail = df_tail['n_wealth'].sample(n=10_000, random_state=4321)
rv_tail = rv_tail.to_numpy()
sample_tail = rv_tail/500_000

fig, ax = plt.subplots()
ax.set_xlim(0,50)
ax.hist(sample_tail, density=True, bins=500, histtype='stepfilled', alpha=0.8)

plt.show()
```

+++ {"user_expressions": []}

### Lognormal Distribution

Again, let's first assume our distribution is lognormally distributed.

We can thus obtain the following maximum likelihood estimates.

```{code-cell} ipython3
ln_sample_tail = np.log(sample_tail)
```

```{code-cell} ipython3
μ_hat_tail = np.mean(ln_sample_tail)
μ_hat_tail
```

```{code-cell} ipython3
num_tail = (ln_sample_tail - μ_hat_tail)**2
σ_hat_tail = (np.mean(num_tail))**(1/2)
σ_hat_tail
```

```{code-cell} ipython3
dist_lognorm_tail = lognorm(σ_hat_tail, scale = exp(μ_hat_tail))

fig, ax = plt.subplots()
ax.set_xlim(0,50)

ax.hist(sample_tail, density=True, bins=500, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_lognorm_tail.pdf(x), 'k-', lw=0.5, label='lognormal pdf')
ax.legend()

plt.show()
```

+++ {"user_expressions": []}

As expected, while the lognormal distribution was a good fit for the entire dataset
it is not a good fit for the right hand tail of the data.

```{code-cell} ipython3
total_revenue(dist_lognorm_tail, 100)
```

+++ {"user_expressions": []}

### Pareto Distribution

Let's now assume the truncated dataset has a pareto distribution.

The maximum likelihood estimates thus obtained would be

```{code-cell} ipython3
xm_hat_tail = min(sample_tail)
xm_hat_tail
```

```{code-cell} ipython3
den_tail = np.log(sample_tail/xm_hat_tail)
b_hat_tail = 1/np.mean(den_tail)
b_hat_tail
```

Let's plot the pdf against our data.

```{code-cell} ipython3
dist_pareto_tail = pareto(b = b_hat_tail, scale = xm_hat_tail)

fig, ax = plt.subplots()
ax.set_xlim(0, 50)
ax.set_ylim(0,0.65)

ax.hist(sample_tail, density=True, bins= 500, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_pareto_tail.pdf(x), 'k-', lw=0.5, label='pareto pdf')

plt.show()
```

+++ {"user_expressions": []}

Thus we clearly observe that the pareto distribution is a better fit for the
right hand tail of our dataset.

```{code-cell} ipython3
total_revenue(dist_pareto_tail, 100)
```

+++ {"user_expressions": []}

## Light-tailed Distributions

Both the lognormal and the pareto distributions are heavy-tailed.

What happens if our initial assumption of the underlying distribution is light-tailed?

Suppose we assume that the distribution is [exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
with parameter $\lambda > 0$.

The maximum likelihood estimate of $\lambda$ is given by:
$$
\hat{\lambda} = \frac{n}{\sum_{i=1}^n w_i}
$$

+++ {"user_expressions": []}

### Entire Dataset

Let's first consider the distribution of the entire dataset.

The maximum likelihood estimate is

```{code-cell} ipython3
λ_hat = 1/np.mean(sample)
λ_hat
```

```{code-cell} ipython3
dist_exp = expon(scale = 1/λ_hat)

fig, ax = plt.subplots()
ax.set_xlim(-1, 20)

ax.hist(sample, density=True, bins=5000, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_exp.pdf(x), 'k-', lw=0.5, label='exponential pdf')
ax.legend()

plt.show()
```

+++ {"user_expressions": []}

### Right-hand Tail

```{code-cell} ipython3
λ_hat_tail = 1/np.mean(sample_tail)
λ_hat_tail
```

```{code-cell} ipython3
dist_exp_tail = expon(scale = 1/λ_hat_tail)

fig, ax = plt.subplots()
ax.set_xlim(0, 50)

ax.hist(sample_tail, density=True, bins= 500, histtype='stepfilled', alpha=0.5)
ax.plot(x, dist_exp_tail.pdf(x), 'k-', lw=0.5, label='exponential pdf')
ax.legend()

plt.show()
```

+++ {"user_expressions": []}

## Log wealth

```{code-cell} ipython3
sample_lnwealth = np.log(rv)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(sample_lnwealth, density=True, bins=100, histtype='stepfilled', alpha=0.8)

plt.show()
```

+++ {"user_expressions": []}

### Normal Distribution

```{code-cell} ipython3
m_hat = np.mean(sample_lnwealth)
m_hat
```

```{code-cell} ipython3
num_lnwealth = (sample_lnwealth - m_hat)**2
s_hat = (np.mean(num_lnwealth))**(1/2)
s_hat
```

```{code-cell} ipython3
dist_norm = norm(loc = m_hat, scale = s_hat)
x2 = np.linspace(0,20,10_000)

fig, ax = plt.subplots()
#ax.set_xlim(-1, 20)
#ax.set_ylim(0,1.75)

ax.hist(sample_lnwealth, density=True, bins=100, histtype='stepfilled', alpha=0.5)
ax.plot(x2, dist_norm.pdf(x2), 'k-', lw=0.5, label='pareto pdf')

plt.show()
```

```{code-cell} ipython3

```
