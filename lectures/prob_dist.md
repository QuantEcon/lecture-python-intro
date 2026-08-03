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

# Probability Distributions

```{index} single: Common Distributions
```

## Outline

In data science applications, we are often interested in data on a specific variable.

In this lecture we give a quick introduction to probability distributions using Python.

A companion lecture, {doc}`observed_distributions`, treats observed data --- sets
of numbers that we measure or collect --- and its connection to the probability
distributions studied here.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

To motivate what follows, let's start with a real example: the heights of adult men and women in the United States.

The data come from the US [National Health and Nutrition Examination Survey](https://www.cdc.gov/nchs/nhanes/index.htm) (NHANES).

The next figure shows histograms of the two datasets, with heights measured in centimeters.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Heights of US adults (NHANES)
    name: fig:us-heights
tags: [hide-input]
---
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/us_adult_heights.csv')
heights = pd.read_csv(url)
male = heights[heights['sex'] == 'male']['height_cm']
female = heights[heights['sex'] == 'female']['height_cm']

fig, ax = plt.subplots()
ax.hist(male, bins=40, density=True, alpha=0.6, label='male')
ax.hist(female, bins=40, density=True, alpha=0.6, label='female')
ax.set_xlabel('height (cm)')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

Each histogram has the familiar "bell" shape.

This suggests that we can approximate the data using a **normal distribution** --- a continuous distribution with a bell-shaped density that we study in detail below.

To do so, we fit a normal distribution to each dataset, choosing its mean and standard deviation to match the sample mean and standard deviation of the heights.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normal fit to US adult heights
    name: fig:us-heights-fit
tags: [hide-input]
---
fig, ax = plt.subplots()
x_grid = np.linspace(130, 205, 200)
for sample, color, label in ((male, 'C0', 'male'), (female, 'C1', 'female')):
    ax.hist(sample, bins=40, density=True, alpha=0.4, color=color)
    u = scipy.stats.norm(sample.mean(), sample.std())
    ax.plot(x_grid, u.pdf(x_grid), color=color, lw=2, label=label)
ax.set_xlabel('height (cm)')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

The fit is remarkably good.

Notice what this achieves: each dataset of around 5,000 individual measurements is now summarized by a smooth density with just **two parameters** --- the mean $\mu$, which sets the center, and the standard deviation $\sigma$, which sets the spread.

Such compact summaries are extremely useful.

They are one reason we study **common distributions**: named families of distributions, each governed by a small number of parameters, that have proven useful for describing data.

We turn to these now, recalling the definitions of some well-known distributions and exploring how to manipulate them with SciPy.

## Discrete distributions

Let's start with discrete distributions.

A discrete distribution is defined by a set of numbers $S = \{x_1, \ldots, x_n\}$ and a **probability mass function** (PMF) on $S$, which is a function $p$ from $S$ to $[0,1]$ with the property 

$$ 
\sum_{i=1}^n p(x_i) = 1 
$$

For example, the next figure shows the fraction of people at each age in Japan in 2024 (Japanese nationals), from 0 to 100 and over.

The data come from the [Statistics Bureau of Japan](https://www.stat.go.jp/english/data/jinsui/index.html).

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Population share by age, Japan 2024
    name: fig:japan-age
tags: [hide-input]
---
# Data file is stored in this repo for now; switch to the QuantEcon/datasets
# URL once that repo exists (see QuantEcon/meta#336).
url = '_static/lecture_specific/prob_dist/japan_population_by_age.xlsx'
# Column 14 holds the Japanese-national population (in thousands) by single year
# of age; rows run from age 0 to "100 and over".
data = pd.read_excel(url, sheet_name='第１表', header=None, skiprows=10,
                     usecols=[14], names=['population'], nrows=101)
population = data['population'].to_numpy()
age = np.arange(101)     # 0, 1, ..., 100, where 100 means "100 and over"

p = population / population.sum()

fig, ax = plt.subplots()
ax.bar(age, p)
ax.set_xlabel('age')
ax.set_ylabel('fraction of population')
plt.show()
```

Here each $x_i$ is an age and $p(x_i)$ is the fraction of the population at that age, and the fractions sum to one.

We say that a random variable $X$ **has distribution** $p$ if $X$ takes value $x_i$ with probability $p(x_i)$.

That is,

$$ 
\mathbb P\{X = x_i\} = p(x_i) \quad \text{for } i= 1, \ldots, n 
$$

The **mean** or **expected value** of a random variable $X$ with distribution $p$ is 

$$ 
\mathbb{E}[X] = \sum_{i=1}^n x_i p(x_i)
$$

Expectation is also called the *first moment* of the distribution.

We also refer to this number as the mean of the distribution (represented by) $p$.

More generally, if $f$ is a function on $S$, then $f(X)$ is a random variable that takes the value $f(x_i)$ whenever $X$ takes the value $x_i$.

Its expectation is obtained by weighting each of these values by its probability:

$$
\mathbb{E}[f(X)] = \sum_{i=1}^n f(x_i) p(x_i)
$$

Every quantity we define below is an expectation of this form, for a suitable choice of $f$.

The **variance** of $X$ is defined as 

$$ 
\mathbb{V}[X] 
    = \mathbb{E}[(X - \mathbb{E}[X])^2]
    = \sum_{i=1}^n (x_i - \mathbb{E}[X])^2 p(x_i)
$$

Variance is also called the *second central moment* of the distribution.

The **standard deviation** of $X$ is the square root of the variance:

$$
\sigma = \sqrt{\mathbb{V}[X]}
$$

We often prefer the standard deviation to the variance because it is measured in the same units as $X$ itself.

For example, if $X$ is a height in centimeters, then $\sigma$ is in centimeters, while the variance is in centimeters squared.

This means that $\sigma$ can be read directly off the horizontal axis of a histogram of the data, as a measure of spread.

Means and variances are special cases of moments.

Writing $\mu = \mathbb{E}[X]$, the $k$-th **moment** of $X$ is $\mathbb{E}[X^k]$, while the $k$-th **central moment** is $\mathbb{E}[(X - \mu)^k]$.

Thus the mean is the first moment and the variance is the second central moment.

It is often convenient to work with **standardized moments**

$$
\mathbb{E} \left[ \left( \frac{X - \mu}{\sigma} \right)^k \right]
$$

which are unchanged when we shift $X$ or rescale it.

(If $X$ is measured in centimeters, we obtain the same standardized moments after converting to inches.)

The third standardized moment is called the **skewness**:

$$
S = \mathbb{E} \left[ \left( \frac{X - \mu}{\sigma} \right)^3 \right]
$$

Skewness measures asymmetry.

Any distribution that is symmetric about its mean has zero skewness, while a distribution with a long right tail has positive skewness.

The fourth standardized moment is called the **kurtosis**:

$$
K = \mathbb{E} \left[ \left( \frac{X - \mu}{\sigma} \right)^4 \right]
$$

Kurtosis measures how much probability mass sits far out in the tails.

For *every* normal distribution, $K = 3$, regardless of $\mu$ and $\sigma$.

Since the normal distribution is such a useful benchmark, it is common to subtract 3 and work with the **excess kurtosis**

$$
K - 3
$$

which is zero for the normal distribution.

Positive excess kurtosis means more mass in the tails than the normal distribution --- extreme values are more likely.

```{note}
Take care when reading software documentation, since these two names are not always used correctly.

For example, `scipy.stats.kurtosis` returns the excess kurtosis by default, rather than the kurtosis.

(Set `fisher=False` to obtain the kurtosis.)
```

We will use skewness and excess kurtosis in {doc}`observed_distributions` to help judge whether a given data set looks normally distributed.

The **cumulative distribution function** (CDF) of $X$ is defined by

$$
F(x) = \mathbb{P}\{X \leq x\}
        = \sum_{i=1}^n \mathbb 1\{x_i \leq x\} p(x_i)
$$

Here $\mathbb 1\{ \textrm{statement} \} = 1$ if "statement" is true and zero otherwise.

Hence the second term takes all $x_i \leq x$ and sums their probabilities.


### Uniform distribution

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


### Bernoulli distribution

Another useful distribution is the Bernoulli distribution on $S = \{0,1\}$, which has PMF:

$$
p(i) = \theta^i (1 - \theta)^{1-i}
\qquad (i = 0, 1)
$$

Here $\theta \in [0,1]$ is a parameter.

We can think of this distribution as modeling probabilities for a random trial with success probability $\theta$.

* $p(1) = \theta$ means that the trial succeeds (takes value 1) with probability $\theta$
* $p(0) = 1 - \theta$ means that the trial fails (takes value 0) with
  probability $1-\theta$

The formula for the mean is $\theta$, and the formula for the variance is $\theta(1-\theta)$.

We can import the Bernoulli distribution on $S = \{0,1\}$ from SciPy like so:

```{code-cell} ipython3
θ = 0.4
u = scipy.stats.bernoulli(θ)
```

Here's the mean and variance at $\theta=0.4$

```{code-cell} ipython3
u.mean(), u.var()
```

We can evaluate the PMF as follows

```{code-cell} ipython3
u.pmf(0), u.pmf(1)
```

### Binomial distribution

Another useful (and more interesting) distribution is the **binomial distribution** on $S=\{0, \ldots, n\}$, which has PMF:

$$ 
p(i) = \binom{n}{i} \theta^i (1-\theta)^{n-i}
$$

Again, $\theta \in [0,1]$ is a parameter.

The interpretation of $p(i)$ is: the probability of $i$ successes in $n$ independent trials with success probability $\theta$.

For example, if $\theta=0.5$, then $p(i)$ is the probability of $i$ heads in $n$ flips of a fair coin.

The formula for the mean is $n \theta$ and the formula for the variance is $n \theta (1-\theta)$.

Let's investigate an example

```{code-cell} ipython3
n = 10
θ = 0.5
u = scipy.stats.binom(n, θ)
```

According to our formulas, the mean and variance are

```{code-cell} ipython3
n * θ,  n *  θ * (1 - θ)  
```

Let's see if SciPy gives us the same results:

```{code-cell} ipython3
u.mean(), u.var()
```

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

### Geometric distribution

The geometric distribution has infinite support $S = \{0, 1, 2, \ldots\}$ and its PMF is given by 

$$
p(i) = (1 - \theta)^i \theta
$$

where $\theta \in [0,1]$ is a parameter

(A discrete distribution has infinite support if the set of points to which it assigns positive probability is infinite.)

To understand the distribution, think of repeated independent random trials, each with success probability $\theta$.

The interpretation of $p(i)$ is: the probability there are $i$ failures before the first success occurs.

It can be shown that the mean of the distribution is $1/\theta$ and the variance is $(1-\theta)/\theta$.

Here's an example.

```{code-cell} ipython3
θ = 0.1
u = scipy.stats.geom(θ)
u.mean(), u.var()
```

Here's part of the PMF:

```{code-cell} ipython3
fig, ax = plt.subplots()
n = 20
S = np.arange(n)
ax.plot(S, u.pmf(S), linestyle='', marker='o', alpha=0.8, ms=4)
ax.vlines(S, 0, u.pmf(S), lw=0.2)
ax.set_xticks(S)
ax.set_xlabel('S')
ax.set_ylabel('PMF')
plt.show()
```

### Poisson distribution

The Poisson distribution on $S = \{0, 1, \ldots\}$ with parameter $\lambda > 0$ has PMF

$$
p(i) = \frac{\lambda^i}{i!} e^{-\lambda}
$$

The interpretation of $p(i)$ is: the probability of $i$ events in a fixed time interval, where the events occur independently at a constant rate $\lambda$.

It can be shown that the mean is $\lambda$ and the variance is also $\lambda$.

Here's an example.

```{code-cell} ipython3
λ = 2
u = scipy.stats.poisson(λ)
u.mean(), u.var()
```

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

## Continuous distributions


A continuous distribution is represented by a **probability density function**, which is a function $p$ over $\mathbb R$ (the set of all real numbers) such that $p(x) \geq 0$ for all $x$ and

$$ 
\int_{-\infty}^\infty p(x) dx = 1 
$$

We say that random variable $X$ has distribution $p$ if

$$
\mathbb P\{a < X < b\} = \int_a^b p(x) dx
$$

for all $a \leq b$.

Expectations are defined as in the discrete case, after replacing the sum with an integral.

For example, the mean of $X$ is

$$
\mathbb{E}[X] = \int_{-\infty}^\infty x p(x) dx
$$

while, for a function $f$,

$$
\mathbb{E}[f(X)] = \int_{-\infty}^\infty f(x) p(x) dx
$$

The variance, standard deviation, moments, skewness and kurtosis are then defined by exactly the same expressions as before.

The **cumulative distribution function** (CDF) of $X$ is defined by

$$
F(x) = \mathbb P\{X \leq x\}
        = \int_{-\infty}^x p(x) dx
$$

For the continuous distributions we study below, $F$ is strictly increasing, so it has an inverse $F^{-1}$, which is called the **quantile function**.

Given $\tau \in (0,1)$, the value $q_\tau = F^{-1}(\tau)$ is called the $\tau$-th **quantile** of the distribution.

It is the point such that $X$ falls below it with probability $\tau$.

The 0.5 quantile is called the **median**, which is an alternative measure of the center of a distribution.

The 0.25 and 0.75 quantiles are called the first and third **quartiles**, and the distance between them is the **interquartile range**, an alternative measure of spread.

These alternatives are useful because, unlike the mean and the standard deviation, they are barely affected by a small number of extreme values.

(We will see in {doc}`heavy_tails` that this robustness matters a great deal for some data sets.)


### Normal distribution

Perhaps the most famous distribution is the **normal distribution**, which has density

$$
p(x) = \frac{1}{\sqrt{2\pi}\sigma}
            \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

This distribution has two parameters, $\mu \in \mathbb R$ and $\sigma \in (0, \infty)$.  

Using calculus, it can be shown that, for this distribution, the mean is $\mu$ and the variance is $\sigma^2$.

We can obtain the moments, PDF and CDF of the normal density via SciPy as follows:

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.norm(μ, σ)
```

```{code-cell} ipython3
u.mean(), u.var()
```

The `stats` method returns the skewness and excess kurtosis when we ask for moments `'sk'`:

```{code-cell} ipython3
u.stats(moments='sk')
```

Both are zero, as promised.

(The skewness is zero because the density is symmetric about $\mu$.)

Here are the median and the two quartiles, obtained via the `ppf` method (SciPy's name for the quantile function):

```{code-cell} ipython3
u.ppf(0.5), u.ppf(0.25), u.ppf(0.75)
```

The median equals the mean because the density is symmetric.

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
    label=rf'$\mu={μ}, \sigma={σ}$')
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
    label=rf'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```

### Lognormal distribution

The **lognormal distribution** is a distribution on $\left(0, \infty\right)$ with density

$$
p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
    \exp \left(- \frac{\left(\log x - \mu\right)^2}{2 \sigma^2} \right)
$$

This distribution has two parameters, $\mu$ and $\sigma$.

It can be shown that, for this distribution, the mean is $\exp\left(\mu + \sigma^2/2\right)$ and the variance is $\left[\exp\left(\sigma^2\right) - 1\right] \exp\left(2\mu + \sigma^2\right)$.

It can be proved that 

* if $X$ is lognormally distributed, then $\log X$ is normally distributed, and
* if $X$ is normally distributed, then $\exp X$ is lognormally distributed.

We can obtain the moments, PDF, and CDF of the lognormal density as follows:

```{code-cell} ipython3
μ, σ = 0.0, 1.0
u = scipy.stats.lognorm(s=σ, scale=np.exp(μ))
```

```{code-cell} ipython3
u.mean(), u.var()
```

The lognormal distribution provides a sharp contrast with the normal distribution in terms of higher moments:

```{code-cell} ipython3
u.stats(moments='sk')
```

The skewness is large and positive, reflecting the long right tail, and the excess kurtosis is enormous.

The gap between the mean and the median is correspondingly large:

```{code-cell} ipython3
u.mean(), u.ppf(0.5)
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
    label=fr'$\mu={μ}, \sigma={σ}$')
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
    label=rf'$\mu={μ}, \sigma={σ}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 3)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```

### Exponential distribution

The **exponential distribution** is a distribution supported on $\left(0, \infty\right)$ with density

$$
p(x) = \lambda \exp \left( - \lambda x \right)
\qquad (x > 0)
$$

This distribution has one parameter $\lambda$.

The exponential distribution can be thought of as the continuous analog of the geometric distribution.

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
    label=rf'$\lambda={λ}$')
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
    label=rf'$\lambda={λ}$')
    ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```

### Beta distribution

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
    label=rf'$\alpha={α}, \beta={β}$')
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
    label=rf'$\alpha={α}, \beta={β}$')
    ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```

### Gamma distribution

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
    label=rf'$\alpha={α}, \beta={β}$')
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
    label=rf'$\alpha={α}, \beta={β}$')
    ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('CDF')
plt.legend()
plt.show()
```
