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

We also refer to this number as the mean of the distribution (represented by) $p$.

The **variance** of $X$ is defined as 

$$ 
    \mathbb V X = \sum_{i=1}^n (x_i - \mathbb E X)^2 p(x_i)
$$

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

#### Exercise

Calculate the mean and variance directly from the PMF, using the expressions given above.

Check that your answers agree with `u.mean()` and `u.var()`.

+++ {"user_expressions": []}

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

#### Exercise

Using `u.pmf`, check that our definition of the CDF given above calculates the same function as `u.cdf`.

+++ {"user_expressions": []}

#### Poisson distribution


+++ {"user_expressions": []}

## Continuous distributions

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

We can obtain the PDF, CDF and moments of the normal density via SciPy as follows.

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.norm(μ, σ)
```

```{code-cell} ipython3
u.mean(), u.var()
```

+++ {"user_expressions": []}

Here's a plot of the density --- the famous "bell-shaped curve":

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = np.linspace(-4, 4, 200)
ax.plot(x_grid, u.pdf(x_grid))
plt.show()
```

+++ {"user_expressions": []}

Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x_grid, u.cdf(x_grid))
ax.set_ylim(0, 1)
plt.show()
```

+++ {"user_expressions": []}

#### Lognormal distribution

+++ {"user_expressions": []}

#### Exponential distribution

+++ {"user_expressions": []}

#### Beta distribution

+++ {"user_expressions": []}

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

#### Exercise

Check that the formulas given above produce the same numbers.

+++ {"user_expressions": []}

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

This convergence is a version of the "law of large numbers", which we will discuss in {ref}`lln_mr`.
