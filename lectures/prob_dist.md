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

+++ {"user_expressions": []}

# Distributions and Probabilities

## Outline

In this lecture we give a quick introduction to data and probability distributions using Python

```{code-cell} ipython3
!pip install --upgrade yfinance  
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats
import seaborn as sns
```

+++ {"user_expressions": []}

## Common distributions

In this section we recall the definitions of some well-known distributions and show how to manipulate them with SciPy.

### Discrete distributions

Let's start with discrete distributions.

A discrete distribution is defined by a set of numbers $S = \{x_1, \ldots, x_n\}$ and a **probability mass function** (PMF) on $S$, which is a function $p$ from $S$ to $[0,1]$ with the property 

$$ \sum_{i=1}^n p(x_i) = 1 $$

We say that a random variable $X$ **has distribution** $p$ if $X$ takes value $x_i$ with probability $p(x_i)$.

That is,

$$ \mathbb P\{X = x_i\} = p(x_i) \quad \text{for } i= 1, \ldots, n $$

The **mean** or **expected value** of a random variable $X$ with distribution $p$ is 

$$ 
    \mathbb E X = \sum_{i=1}^n x_i p(x_i)
$$

Expectation is also called the *first moment* of the distribution.

We also refer to this number as the mean of the distribution (represented by) $p$.

The **variance** of $X$ is defined as 

$$ 
    \mathbb V X = \sum_{i=1}^n (x_i - \mathbb E X)^2 p(x_i)
$$

Variance is also called the *second central moment* of the distribution.

The **cumulative distribution function** (CDF) of $X$ is defined by

$$
    F(x) = \mathbb P\{X \leq x\}
         = \sum_{i=1}^n \mathbb 1\{x_i \leq x\} p(x_i)
$$

Here $\mathbb 1\{ \textrm{statement} \} = 1$ if "statement" is true and zero otherwise.

Hence the second term takes all $x_i \leq x$ and sums their probabilities.


#### Uniform distribution

One simple example is the **uniform distribution**, where $p(x_i) = 1/n$ for all $n$.

We can import the uniform distribution on $S = \{1, \ldots, n\}$  from SciPy like so:

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```

+++ {"user_expressions": []}

Here's the mean and variance

```{code-cell} ipython3
u.mean()
```

```{code-cell} ipython3
u.var()
```

+++ {"user_expressions": []}

Now let's evaluate the PMF

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
u.pmf(2)
```

+++ {"user_expressions": []}

Here's a plot of the probability mass function:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

+++ {"user_expressions": []}

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

+++ {"user_expressions": []}

The CDF jumps up by $p(x_i)$ and $x_i$.

+++ {"user_expressions": []}

```{exercise}
:label: prob_ex1

Calculate the mean and variance directly from the PMF, using the expressions given above.

Check that your answers agree with `u.mean()` and `u.var()`.
```

#### Binomial distribution

Another useful (and more interesting) distribution is the **binomial distribution** on $S=\{0, \ldots, n\}$, which has PMF

$$ 
    p(i) = \binom{i}{n} \theta^i (1-\theta)^{n-i}
$$

Here $\theta \in [0,1]$ is a parameter.

The interpretatin of $p(i)$ is: the number of successes in $n$ independent trials with success probability $\theta$.

(If $\theta=0.5$, this is "how many heads in $n$ flips of a fair coin")

Here's the PDF

```{code-cell} ipython3
n = 10
θ = 0.5
u = scipy.stats.binom(n, θ)
```

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

+++ {"user_expressions": []}

Here's the CDF

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

+++ {"user_expressions": []}

```{exercise}
:label: prob_ex2

Using `u.pmf`, check that our definition of the CDF given above calculates the same function as `u.cdf`.
```

```{solution-start} mc_ex2
:class: dropdown
```

Here is one solution

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
u_sum = np.cumsum(u.pmf(S))
ax.step(S, u_sum)
ax.vlines(S, 0, u_sum, lw=0.2)
ax.set_xticks(S)
plt.show()
```

We can see that the output graph is the same as the one above.

```{solution-end}
```

#### Poisson distribution

Poisson distribution on $S = \{0, 1, \ldots\}$ with parameter $\lambda > 0$ has PMF

$$
    p(i) = \frac{\lambda^i}{i!} e^{-\lambda}
$$

The interpretation of $p(i)$ is: the number of events in a fixed time interval, where the events occur at a constant rate $\lambda$ and independently of each other.

Here's the PMF

```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
```

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
plt.show()
```

+++ {"user_expressions": []}

### Continuous distributions

+++ {"user_expressions": []}

Continuous distributions are represented by a **density function**, which is a function $p$ over $\mathbb R$ (the set of all numbers) such that $p(x) \geq 0$ for all $x$ and

$$ \int_{-\infty}^\infty p(x) = 1 $$

We say that random variable $X$ has distribution $p$ if

$$
    \mathbb P\{a < X < b\} = \int_a^b p(x) dx
$$

for all $a \leq b$.

The definition of the mean and variance of a random variable $X$ with distribution $p$ are the same as the discrete case, after replacing the sum with an integral.

For example, the mean of $X$ is

$$
    \mathbb E X = \int_{-\infty}^\infty x p(x) dx
$$

The **cumulative distribution function** (CDF) of $X$ is defined by

$$
    F(x) = \mathbb P\{X \leq x\}
         = \int_{-\infty}^y p(y) dy
$$

+++ {"user_expressions": []}

#### Normal distribution

Perhaps the most famous distribution is the **normal distribution**, which as density

$$
    p(x) = \frac{1}{\sqrt{2\pi}\sigma}
        \exp \left( - \frac{x - \mu}{2 \sigma^2} \right)
$$

This distribution has two parameters, $\mu$ and $\sigma$.  

It can be shown that, for this distribution, the mean is $\mu$ and the variance is $\sigma^2$.

We can obtain the moments, PDF, CDF of the normal density via SciPy as follows:

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.norm(μ, σ)
```

```{code-cell} ipython3
u.mean(), u.var()
```

Here's a plot of the density --- the famous "bell-shaped curve":

```{code-cell} ipython3
μ_vals = [-1, 0, 1]
σ_vals = [0.4, 1, 1.6]
fig, ax = plt.subplots()
x_grid = np.linspace(-4, 4, 200)

for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')

plt.legend()
plt.show()
```

+++ {"user_expressions": []}

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
plt.legend()
plt.show()
```

+++ {"user_expressions": []}

#### Lognormal distribution

The **lognormal distribution** is a distribution on $\left(0, \infty\right)$ with density

$$
    p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
        \exp \left(- \frac{\left(\log x - \mu\right)^2}{2 \sigma^2} \right)
$$

This distribution has two parameters, $\mu$ and $\sigma$.

It can be shown that, for this distribution, the mean is $\exp\left(\mu + \sigma^2/2\right)$ and the variance is $\left[\exp\left(\sigma^2\right) - 1\right] \exp\left(2\mu + \sigma^2\right)$.

It has a nice interpretation: if $X$ is lognormally distributed, then $\log X$ is normally distributed.

It is often used to model variables that are "multiplicative" in nature, such as income or asset prices.

We can obtain the moments, PDF, CDF of the normal density via SciPy as follows:

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.lognorm(s=σ, scale=np.exp(μ))
```

```{code-cell} ipython3
u.mean(), u.var()
```

```{code-cell} ipython3
μ_vals = [-1, 0, 1]
σ_vals = [0.25, 0.5, 1]
fig, ax = plt.subplots()

x_grid = np.linspace(0, 3, 200)

for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.lognorm(σ, scale=np.exp(μ))
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')

plt.legend()
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
μ = 1
for σ in σ_vals:
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 3)
plt.legend()
plt.show()
```

#### Exponential distribution

The **exponential distribution** is a distribution on $\left(0, \infty\right)$ with density

$$
    p(x) = \lambda \exp \left( - \lambda x \right)
$$

This distribution has one parameter, $\lambda$.

It is related to the Poisson distribution as it describes the distribution of the length of the time interval between two consecutive events in a Poisson process.

It can be shown that, for this distribution, the mean is $1/\lambda$ and the variance is $1/\lambda^2$.

We can obtain the moments, PDF, CDF of the normal density via SciPy as follows:

```{code-cell} ipython3
λ = 1.0
u = scipy.stats.expon(scale=1/λ)
```

```{code-cell} ipython3
u.mean(), u.var()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
λ_vals = [0.5, 1, 2]
for λ in λ_vals:
    u = scipy.stats.expon(scale=1/λ)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\lambda={λ}$')
plt.legend()
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for λ in λ_vals:
    u = scipy.stats.expon(scale=1/λ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\lambda={λ}$')
    ax.set_ylim(0, 1)
plt.legend()
plt.show()
```

#### Beta distribution

The **beta distribution** is a distribution on $\left(0, 1\right)$ with density

$$
    p(x) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}
        x^{\alpha - 1} (1 - x)^{\beta - 1}
$$

where $\Gamma$ is the gamma function ($\Gamma(n) = (n - 1)!$ for $n \in \mathbb{N}$).

This distribution has two parameters, $\alpha$ and $\beta$.

It has a nice interpretation: if $X$ is beta distributed, then $X$ is the probability of success in a Bernoulli trial with a number of successes $\alpha$ and a number of failures $\beta$.

For example, if $\alpha = \beta = 1$, then the beta distribution is uniform on $\left(0, 1\right)$ as the number of successes and failures are both 1.

While, if $\alpha = 3$ and $\beta = 2$, then the beta distribution is located more towards 1 as there are more successes than failures.

It can be shown that, for this distribution, the mean is $\alpha / (\alpha + \beta)$ and the variance is $\alpha \beta / (\alpha + \beta)^2 (\alpha + \beta + 1)$.

We can obtain the moments, PDF, CDF of the normal density via SciPy as follows:

```{code-cell} ipython3
α, β = 1.0, 1.0
u = scipy.stats.beta(α, β)
```

```{code-cell} ipython3
u.mean(), u.var()
```

```{code-cell} ipython3
α_vals = [0.5, 1, 50, 250, 3]
β_vals = [3, 1, 100, 200, 1]
x_grid = np.linspace(0, 1, 200)

fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.beta(α, β)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
plt.legend()
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.beta(α, β)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
    ax.set_ylim(0, 1)
plt.legend()
plt.show()
```

#### Gamma distribution

The **gamma distribution** is a distribution on $\left(0, \infty\right)$ with density

$$
    p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}
        x^{\alpha - 1} \exp(-\beta x)
$$

This distribution has two parameters, $\alpha$ and $\beta$.

It can be shown that, for this distribution, the mean is $\alpha / \beta$ and the variance is $\alpha / \beta^2$.

One interpretation is that if $X$ is gamma distributed, then $X$ is the sum of $\alpha$ independent exponentially distributed random variables with mean $1/\beta$.

We can obtain the moments, PDF, CDF of the normal density via SciPy as follows:

```{code-cell} ipython3
α, β = 1.0, 1.0
u = scipy.stats.gamma(α, scale=1/β)
```

```{code-cell} ipython3
u.mean(), u.var()
```

```{code-cell} ipython3
α_vals = [1, 3, 5, 10]
β_vals = [3, 5, 3, 3]
x_grid = np.linspace(0, 7, 200)

fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.gamma(α, scale=1/β)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
plt.legend()
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.gamma(α, scale=1/β)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
    ax.set_ylim(0, 1)
plt.legend()
plt.show()
```

## Observed distributions

+++ {"user_expressions": []}

Sometimes we refer to observed data or measurements as "distributions".

For example, let's say we observe the income of 10 people over a year:

```{code-cell} ipython3
data = [['Hiroshi', 1200], 
        ['Ako', 1210], 
        ['Emi', 1400],
        ['Daiki', 990],
        ['Chiyo', 1530],
        ['Taka', 1210],
        ['Katsuhiko', 1240],
        ['Daisuke', 1124],
        ['Yoshi', 1330],
        ['Rie', 1340]]

df = pd.DataFrame(data, columns=['name', 'income'])
df
```

+++ {"user_expressions": []}

In this situation, we might refer to the set of their incomes as the "income distribution."

The terminology is confusing because this is not the same thing as a probability distribution --- it's just a collection of numbers.

Below we explore some observed distributions.

We will see that there are connections between observed distributions---like the income distribution above---and probability distributions, as we'll see below.

+++ {"user_expressions": []}

### Summary statistics

Suppose we have an observed distribution with values $\{x_1, \ldots, x_n\}$

The **sample mean** of this distribution is defined as

$$
    \bar x = \frac{1}{n} \sum_{i=1}^n x_i
$$

The **sample variance** is defined as 

$$
    \frac{1}{n} \sum_{i=1}^n (x_i - \bar x)^2
$$

+++ {"user_expressions": []}

For the income distribution given above, we can calculate these numbers via

```{code-cell} ipython3
x = np.asarray(df['income'])
```

```{code-cell} ipython3
x.mean(), x.var()
```

+++ {"user_expressions": []}

```{exercise}
:label: prob_ex3

Check that the formulas given above produce the same numbers.
```

### Visualization

Let's look at different ways that we can visualize one or more observed distributions.

We will cover

- histograms
- kernel density estimates and
- violin plots

+++ {"user_expressions": []}

#### Histograms

+++ {"user_expressions": []}

We can histogram the income distribution we just constructed as follows

```{code-cell} ipython3
x = df['income']
fig, ax = plt.subplots()
ax.hist(x, bins=5, density=True, histtype='bar')
plt.show()
```

+++ {"user_expressions": []}

Let's look at a distribution from real data.

In particular, we will look at the monthly return on Amazon shares between 2000/1/1 and 2023/1/1.

The monthly return is calculated as the percent change in the share price over each month.

So we will have one observation for each month.

```{code-cell} ipython3
df = yf.download('AMZN', '2000-1-1', '2023-1-1', interval='1mo' )
prices = df['Adj Close']
data = prices.pct_change()[1:] * 100
data.head()
```

+++ {"user_expressions": []}

The first observation is the monthly return (percent change) over January 2000, which was

```{code-cell} ipython3
data[0] 
```

+++ {"user_expressions": []}

Let's turn the return observations into an array and histogram it.

```{code-cell} ipython3
x_amazon = np.asarray(data)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(x_amazon, bins=20)
plt.show()
```

+++ {"user_expressions": []}

#### Kernel density estimates

Kernel density estimate (KDE) is a non-parametric way to estimate and visualize the PDF of a distribution.

KDE will generate a smooth curve that approximates the PDF.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax)
plt.show()
```

The smoothness of the KDE is dependent on how we choose the bandwidth.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.1, alpha=0.5, label="bw=0.1")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.5, alpha=0.5, label="bw=0.5")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=1, alpha=0.5, label="bw=1")
plt.legend()
plt.show()
```

When we use a larger bandwidth, the KDE is smoother.

A suitable bandwith is the one that is not too smooth (underfitting) or too wiggly (overfitting).


#### Violin plots

+++ {"user_expressions": []}

Yet another way to display an observed distribution is via a violin plot.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot(x_amazon)
plt.show()
```

+++ {"user_expressions": []}

Violin plots are particularly useful when we want to compare different distributions.

For example, let's compare the monthly returns on Amazon shares with the monthly return on Apple shares.

```{code-cell} ipython3
df = yf.download('AAPL', '2000-1-1', '2023-1-1', interval='1mo' )
prices = df['Adj Close']
data = prices.pct_change()[1:] * 100
x_apple = np.asarray(data)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot([x_amazon, x_apple])
plt.show()
```

+++ {"user_expressions": []}

### Connection to probability distributions

+++ {"user_expressions": []}

Let's discuss the connection between observed distributions and probability distributions.

Sometimes it's helpful to imagine that an observed distribution is generated by a particular probability distribution.

For example, we might look at the returns from Amazon above and imagine that they were generated by a normal distribution.

Even though this is not true, it might be a helpful way to think about the data.

Here we match a normal distribution to the Amazon monthly returns by setting the sample mean to the mean of the normal distribution and the sample variance equal to the variance.

Then we plot the density and the histogram.

```{code-cell} ipython3
μ = x_amazon.mean()
σ_squared = x_amazon.var()
σ = np.sqrt(σ_squared)
u = scipy.stats.norm(μ, σ)
```

```{code-cell} ipython3
x_grid = np.linspace(-50, 65, 200)
fig, ax = plt.subplots()
ax.plot(x_grid, u.pdf(x_grid))
ax.hist(x_amazon, density=True, bins=40)
plt.show()
```

+++ {"user_expressions": []}

The match between the histogram and the density is not very bad but also not very good.

One reason is that the normal distribution is not really a good fit for this observed data --- we will discuss this point again when we talk about heavy tailed distributions in TODO add link.

+++ {"user_expressions": []}

Of course, if the data really *is* generated by the normal distribution, then the fit will be better.

Let's see this in action

- first we generate random draws from the normal distribution
- then we histogram them and compare with the density.

```{code-cell} ipython3
μ, σ = 0, 1
u = scipy.stats.norm(μ, σ)
N = 2000  # Number of observations
x_draws = u.rvs(N)
x_grid = np.linspace(-4, 4, 200)
fig, ax = plt.subplots()
ax.plot(x_grid, u.pdf(x_grid))
ax.hist(x_draws, density=True, bins=40)
plt.show()
```

+++ {"user_expressions": []}

Note that if you keep increasing $N$, which is the number of observations, the fit will get better and better.

This convergence is a version of the "law of large numbers", which we will discuss {ref}`later<lln_mr>`.

