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


# Distributions and Probabilities

```{index} single: Distributions and Probabilities
```

## Outline

In this lecture we give a quick introduction to data and probability distributions using Python.

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
import seaborn as sns
```


## Common distributions

In this section we recall the definitions of some well-known distributions and explore how to manipulate them with SciPy.

### Discrete distributions

Let's start with discrete distributions.

A discrete distribution is defined by a set of numbers $S = \{x_1, \ldots, x_n\}$ and a **probability mass function** (PMF) on $S$, which is a function $p$ from $S$ to $[0,1]$ with the property 

$$ \sum_{i=1}^n p(x_i) = 1 $$

We say that a random variable $X$ **has distribution** $p$ if $X$ takes value $x_i$ with probability $p(x_i)$.

That is,

$$ \mathbb P\{X = x_i\} = p(x_i) \quad \text{for } i= 1, \ldots, n $$

The **mean** or **expected value** of a random variable $X$ with distribution $p$ is 

$$ 
    \mathbb{E}[X] = \sum_{i=1}^n x_i p(x_i)
$$

Expectation is also called the *first moment* of the distribution.

We also refer to this number as the mean of the distribution (represented by) $p$.

The **variance** of $X$ is defined as 

$$ 
    \mathbb{V}[X] = \sum_{i=1}^n (x_i - \mathbb{E}[X])^2 p(x_i)
$$

Variance is also called the *second central moment* of the distribution.

The **cumulative distribution function** (CDF) of $X$ is defined by

$$
    F(x) = \mathbb{P}\{X \leq x\}
         = \sum_{i=1}^n \mathbb 1\{x_i \leq x\} p(x_i)
$$

Here $\mathbb 1\{ \textrm{statement} \} = 1$ if "statement" is true and zero otherwise.

Hence the second term takes all $x_i \leq x$ and sums their probabilities.


#### Uniform distribution

One simple example is the **uniform distribution**, where $p(x_i) = 1/n$ for all $i$.

We can import the uniform distribution on $S = \{1, \ldots, n\}$  from SciPy like so:

```{code-cell} ipython3
n = 10
u = scipy.stats.randint(1, n+1)
```


Here's the mean and variance:

```{code-cell} ipython3
u.mean(), u.var()
```

The formula for the mean is $(n+1)/2$, and the formula for the variance is $(n^2 - 1)/12$.


Now let's evaluate the PMF:

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
u.pmf(2)
```


Here's a plot of the probability mass function:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('PMF')
plt.show()
```


Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('CDF')
plt.show()
```


The CDF jumps up by $p(x_i)$ at $x_i$.


```{exercise}
:label: prob_ex1

Calculate the mean and variance for this parameterization (i.e., $n=10$)
directly from the PMF, using the expressions given above.

Check that your answers agree with `u.mean()` and `u.var()`. 
```


#### Bernoulli distribution

Another useful distribution is the Bernoulli distribution on $S = \{0,1\}$, which has PMF:

$$
p(x_i)=
\begin{cases}
p & \text{if $x_i = 1$}\\
1-p & \text{if $x_i = 0$}
\end{cases}
$$

Here $x_i \in S$ is the outcome of the random variable.

We can import the Bernoulli distribution on $S = \{0,1\}$ from SciPy like so:

```{code-cell} ipython3
p = 0.4 
u = scipy.stats.bernoulli(p)
```


Here's the mean and variance:

```{code-cell} ipython3
u.mean(), u.var()
```

The formula for the mean is $p$, and the formula for the variance is $p(1-p)$.


Now let's evaluate the PMF:

```{code-cell} ipython3
u.pmf(0)
u.pmf(1)
```


#### Binomial distribution

Another useful (and more interesting) distribution is the **binomial distribution** on $S=\{0, \ldots, n\}$, which has PMF:

$$ 
    p(i) = \binom{n}{i} \theta^i (1-\theta)^{n-i}
$$

Here $\theta \in [0,1]$ is a parameter.

The interpretation of $p(i)$ is: the probability of $i$ successes in $n$ independent trials with success probability $\theta$.

For example, if $\theta=0.5$, then $p(i)$ is the probability of $i$ heads in $n$ flips of a fair coin.

The mean and variance are:

```{code-cell} ipython3
n = 10
θ = 0.5
u = scipy.stats.binom(n, θ)
```

```{code-cell} ipython3
u.mean(), u.var()
```

The formula for the mean is $n \theta$ and the formula for the variance is $n \theta (1-\theta)$.

Here's the PMF:

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('PMF')
plt.show()
```


Here's the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.step(S, u.cdf(S))
ax.vlines(S, 0, u.cdf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('CDF')
plt.show()
```


```{exercise}
:label: prob_ex3

Using `u.pmf`, check that our definition of the CDF given above calculates the same function as `u.cdf`.
```

```{solution-start} prob_ex3
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
u_sum = np.cumsum(u.pmf(S))
ax.step(S, u_sum)
ax.vlines(S, 0, u_sum, lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('CDF')
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

The interpretation of $p(i)$ is: the probability of $i$ events in a fixed time interval, where the events occur at a constant rate $\lambda$ and independently of each other.

The mean and variance are:
```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
u.mean(), u.var()
```
    
The expectation of Poisson distribution is $\lambda$ and the variance is also $\lambda$.

Here's the PMF:

```{code-cell} ipython3
u.pmf(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
S = np.arange(1, n+1)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('PMF')
plt.show()
```


### Continuous distributions


Continuous distributions are represented by a **probability density function**, which is a function $p$ over $\mathbb R$ (the set of all real numbers) such that $p(x) \geq 0$ for all $x$ and

$$ \int_{-\infty}^\infty p(x) dx = 1 $$

We say that random variable $X$ has distribution $p$ if

$$
    \mathbb P\{a < X < b\} = \int_a^b p(x) dx
$$

for all $a \leq b$.

The definition of the mean and variance of a random variable $X$ with distribution $p$ are the same as the discrete case, after replacing the sum with an integral.

For example, the mean of $X$ is

$$
    \mathbb{E}[X] = \int_{-\infty}^\infty x p(x) dx
$$

The **cumulative distribution function** (CDF) of $X$ is defined by

$$
    F(x) = \mathbb P\{X \leq x\}
         = \int_{-\infty}^x p(x) dx
$$


#### Normal distribution

Perhaps the most famous distribution is the **normal distribution**, which has density

$$
    p(x) = \frac{1}{\sqrt{2\pi}\sigma}
              \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

This distribution has two parameters, $\mu$ and $\sigma$.  

It can be shown that, for this distribution, the mean is $\mu$ and the variance is $\sigma^2$.

We can obtain the moments, PDF and CDF of the normal density as follows:

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
ax.set_xlabel('x')
ax.set_ylabel('PDF')
plt.legend()
plt.show()
```


Here's a plot of the CDF:

```{code-cell} ipython3
fig, ax = plt.subplots()
for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.norm(μ, σ)
    ax.plot(x_grid, u.cdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```


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

We can obtain the moments, PDF, and CDF of the lognormal density as follows:

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
x_grid = np.linspace(0, 3, 200)

fig, ax = plt.subplots()
for μ, σ in zip(μ_vals, σ_vals):
    u = scipy.stats.lognorm(σ, scale=np.exp(μ))
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\mu={μ}, \sigma={σ}$')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
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
ax.set_xlabel('x')
ax.set_ylabel('CDF')
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

We can obtain the moments, PDF, and CDF of the exponential density as follows:

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
x_grid = np.linspace(0, 6, 200)

for λ in λ_vals:
    u = scipy.stats.expon(scale=1/λ)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=f'$\lambda={λ}$')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
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
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```

#### Beta distribution

The **beta distribution** is a distribution on $(0, 1)$ with density

$$
    p(x) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}
        x^{\alpha - 1} (1 - x)^{\beta - 1}
$$

where $\Gamma$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

(The role of the gamma function is just to normalize the density, so that it
integrates to one.)

This distribution has two parameters, $\alpha > 0$ and $\beta > 0$.

It can be shown that, for this distribution, the mean is $\alpha / (\alpha + \beta)$ and 
the variance is $\alpha \beta / (\alpha + \beta)^2 (\alpha + \beta + 1)$.

We can obtain the moments, PDF, and CDF of the Beta density as follows:

```{code-cell} ipython3
α, β = 3.0, 1.0
u = scipy.stats.beta(α, β)
```

```{code-cell} ipython3
u.mean(), u.var()
```

```{code-cell} ipython3
α_vals = [0.5, 1, 5, 25, 3]
β_vals = [3, 1, 10, 20, 0.5]
x_grid = np.linspace(0, 1, 200)

fig, ax = plt.subplots()
for α, β in zip(α_vals, β_vals):
    u = scipy.stats.beta(α, β)
    ax.plot(x_grid, u.pdf(x_grid),
    alpha=0.5, lw=2,
    label=fr'$\alpha={α}, \beta={β}$')
ax.set_xlabel('x')
ax.set_ylabel('PDF')
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
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```


#### Gamma distribution

The **gamma distribution** is a distribution on $\left(0, \infty\right)$ with density

$$
    p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}
        x^{\alpha - 1} \exp(-\beta x)
$$

This distribution has two parameters, $\alpha > 0$ and $\beta > 0$.

It can be shown that, for this distribution, the mean is $\alpha / \beta$ and
the variance is $\alpha / \beta^2$.

One interpretation is that if $X$ is gamma distributed and $\alpha$ is an
integer, then $X$ is the sum of $\alpha$ independent exponentially distributed
random variables with mean $1/\beta$.

We can obtain the moments, PDF, and CDF of the Gamma density as follows:

```{code-cell} ipython3
α, β = 3.0, 2.0
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
ax.set_xlabel('x')
ax.set_ylabel('PDF')
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
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```

## Observed distributions


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


In this situation, we might refer to the set of their incomes as the "income distribution."

The terminology is confusing because this set is not a probability distribution
--- it's just a collection of numbers.

However, as we will see, there are connections between observed distributions (i.e., sets of
numbers like the income distribution above) and probability distributions.

Below we explore some observed distributions.


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

For the income distribution given above, we can calculate these numbers via

```{code-cell} ipython3
x = np.asarray(df['income'])
```

```{code-cell} ipython3
x.mean(), x.var()
```


```{exercise}
:label: prob_ex4

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
ax.set_xlabel('income')
ax.set_ylabel('density')
plt.show()
```

+++ {"user_expressions": []}

Let's look at a distribution from real data.

In particular, we will look at the monthly return on Amazon shares between 2000/1/1 and 2023/1/1.

The monthly return is calculated as the percent change in the share price over each month.

So we will have one observation for each month.

```{code-cell} ipython3
:tags: [hide-output]
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
ax.set_xlabel('monthly return (percent change)')
ax.set_ylabel('density')
plt.show()
```

+++ {"user_expressions": []}

#### Kernel density estimates

Kernel density estimate (KDE) is a non-parametric way to estimate and visualize the PDF of a distribution.

KDE will generate a smooth curve that approximates the PDF.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax)
ax.set_xlabel('monthly return (percent change)')
ax.set_ylabel('KDE')
plt.show()
```

The smoothness of the KDE is dependent on how we choose the bandwidth.

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.1, alpha=0.5, label="bw=0.1")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=0.5, alpha=0.5, label="bw=0.5")
sns.kdeplot(x_amazon, ax=ax, bw_adjust=1, alpha=0.5, label="bw=1")
ax.set_xlabel('monthly return (percent change)')
ax.set_ylabel('KDE')
plt.legend()
plt.show()
```

When we use a larger bandwidth, the KDE is smoother.

A suitable bandwidth is not too smooth (underfitting) or too wiggly (overfitting).


#### Violin plots

+++ {"user_expressions": []}

Yet another way to display an observed distribution is via a violin plot.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot(x_amazon)
ax.set_ylabel('monthly return (percent change)')
ax.set_xlabel('KDE')
plt.show()
```

+++ {"user_expressions": []}

Violin plots are particularly useful when we want to compare different distributions.

For example, let's compare the monthly returns on Amazon shares with the monthly return on Apple shares.

```{code-cell} ipython3
:tags: [hide-output]
df = yf.download('AAPL', '2000-1-1', '2023-1-1', interval='1mo' )
prices = df['Adj Close']
data = prices.pct_change()[1:] * 100
x_apple = np.asarray(data)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.violinplot([x_amazon, x_apple])
ax.set_ylabel('monthly return (percent change)')
ax.set_xlabel('KDE')
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
ax.set_xlabel('monthly return (percent change)')
ax.set_ylabel('density')
plt.show()
```

+++ {"user_expressions": []}

The match between the histogram and the density is not very bad but also not very good.

One reason is that the normal distribution is not really a good fit for this observed data --- we will discuss this point again when we talk about {ref}`heavy tailed distributions<heavy_tail>`.

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
ax.set_xlabel('x')
ax.set_ylabel('density')
plt.show()
```

+++ {"user_expressions": []}

Note that if you keep increasing $N$, which is the number of observations, the fit will get better and better.

This convergence is a version of the "law of large numbers", which we will discuss {ref}`later<lln_mr>`.

