---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## LLN and CLT

## Overview

This lecture illustrates two of the most important theorems of probability and statistics: The
law of large numbers (LLN) and the central limit theorem (CLT).

These beautiful theorems lie behind many of the most fundamental results in econometrics and quantitative economic modeling.

The lecture is based around simulations that show the LLN and CLT in action.

We also demonstrate how the LLN and CLT break down when the assumptions they are based on do not hold.

In addition, we examine several useful extensions of the classical theorems, such as

* The delta method, for smooth functions of random variables, and
* the multivariate case.

Some of these extensions are presented as exercises.

We'll need the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.stats as st
```

## Relationships


The LLN gives conditions under which sample moments converge to population moments as sample size increases.

The CLT provides information about the rate at which sample moments converge to population moments as sample size increases.

(lln_mr)=
## LLN

```{index} single: Law of Large Numbers
```

We begin with the law of large numbers, which tells us when sample averages
will converge to their population means.

### The LLN in Action

Let's see an example of the LLN in action before we go further.

Consider a [Bernoulli random variable](https://en.wikipedia.org/wiki/Bernoulli_distribution) $X$ with parameter $p$.

This means that $X$ takes values in $\{0,1\}$ and $\mathbb P\{X=1\} = p$.

We can think of drawing $X$ as tossing a biased coin where

* the coin falls on "heads" with probability $p$ and
* we set $X=1$ if the coin is "heads" and zero otherwise.

The mean of $X$ is 

$$
\mathbb E X = 0 \cdot \mathbb P\{X=0\} + 1 \cdot \mathbb P\{X=1\} = \mathbb P\{X=1\} = p
$$

We can generate a draw of $X$ with `scipy.stats` (imported as `st`) as follows:

```{code-cell} ipython3
p = 0.8
X = st.bernoulli.rvs(p)
print(X)
```

In this setting, the LLN tells us if we flip the coin many times, the fraction of heads that we see will be close to $p$.

Let's check this:

```{code-cell} ipython3
n = 1_000_000
X_draws = st.bernoulli.rvs(p, size=n)
print(X_draws.mean())  # count the number of 1's and divide by n
```

If we change $p$ the claim still holds:

```{code-cell} ipython3
p = 0.3
X_draws = st.bernoulli.rvs(p, size=n)
print(X_draws.mean())
```

Let's connect this to the discussion above, where we said the sample average converges to the "population mean".

The population mean is the mean in an infinite sample, which equals the true mean, or $\mathbb E X$.

The sample mean of the draws $X_1, \ldots, X_n$ is

$$
\bar X_n := \frac{1}{n} \sum_{i=1}^n X_i
$$

which, in this case, is the fraction of draws that equal one (the number of heads divided by $n$).

Thus, the LLN tells us that

$$
\bar X_n \to \mathbb E X = p
\qquad (n \to \infty)
$$

This is exactly what we illustrated in the code above.

+++

(lln_ksl)=
### Statement of the LLN

Let's state the LLN more carefully.

The traditional version of the law of large numbers concerns independent and identically distributed (IID) random variables.

Let $X_1, \ldots, X_n$ be independent and identically distributed random variables.

This random variables can be continuous or discrete.

For simplicity we will assume they are continuous and we let $f$ denote their density function, so that, for any $i$ in $\{1, \ldots, n\}$

$$ 
    \mathbb P\{a \leq X_i \leq b\} = \int_a^b f(x) dx
$$

(For the discrete case, we need to replace densities with probability mass functions and integrals with sums.)

Let $\mu$ denote the common mean of this sample:

$$
    \mu := \mathbb E X = \int_{-\infty}^{\infty} x f(dx)
$$

In addition, let

$$
\bar X_n := \frac{1}{n} \sum_{i=1}^n X_i
$$

TODO -- use a theorem environment (```{prf:theorem}...```)

The law of large numbers (specifically, Kolmogorov's strong law) states that, if $\mathbb E |X|$ is finite, then

```{math}
:label: lln_as

\mathbb P \left\{ \bar X_n \to \mu \text{ as } n \to \infty \right\} = 1
```

### Comments on the Theorem

What does this last expression mean?

Let's think about it from a simulation perspective, imagining for a moment that
our computer can generate perfect random samples (which of course [it can't](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)).

Let's also imagine that we can generate infinite sequences so that the statement $\bar X_n \to \mu$ can be evaluated.

In this setting, {eq}`lln_as` should be interpreted as meaning that the probability of the computer producing a sequence where $\bar X_n \to \mu$ fails to occur
is zero.

+++

### Illustration

```{index} single: Law of Large Numbers; Illustration
```

Let's now illustrate the LLN using simulation.

When we illustrate it, we will use a key idea: the sample mean $\bar X_n$ is itself a random variable.

In a sense this is obvious but it can be easy to forget.

The reason $\bar X_n$ is a random variable is that it's a function of the random variables $X_1, \ldots, X_n$.

What we are going to do now is 

1. Pick some distribution to draw each $X_i$ from  
1. Set $n$ to some large number
1. Generate the draws $X_1, \ldots, X_n$
1. Calculate the sample mean $\bar X_n$ and record its value in an array `sample_means`
1. Go to step 3

We will continue the loop over steps 3-4 a total of $m$ times, where $m$ is some large integer.

The array `sample_means` will now contain $m$ draws of the random variable $\bar X_n$.

If we histogram these observations of $\bar X_n$, we should see that they are clustered around the population mean $\mathbb E X$.

Moreover, if we repeat the exercise with a larger value of $n$, we should see that the observations are even more tightly clustered around the population mean.

This is, in essence, what the LLN is telling us.


```{code-cell} ipython3
# TODO: write the code and put the plot here
```

## Breaking the LLN

We have to pay attention to the assumptions in the statement of the LLN when we apply it.

TODO

* Illustrate by simulation that the LLN can fail when the population mean is not finite
* Illustrate by simulation that the IID assumption is important

+++

## CLT

```{index} single: Central Limit Theorem
```

Next, we turn to the central limit theorem, which tells us about the distribution of the deviation between sample averages and population means.

### Statement of the Theorem

The central limit theorem is one of the most remarkable results in all of mathematics.

In the IID setting, it tells us the following:

TODO use a theorem environment (```{prf:theorem...```)

(statement_clt)=
If the sequence $X_1, \ldots, X_n$ is IID, with common mean
$\mu$ and common variance $\sigma^2 \in (0, \infty)$, then

```{math}
:label: lln_clt

\sqrt{n} ( \bar X_n - \mu ) \stackrel { d } {\to} N(0, \sigma^2)
\quad \text{as} \quad
n \to \infty
```

Here $\stackrel { d } {\to} N(0, \sigma^2)$ indicates [convergence in distribution](https://en.wikipedia.org/wiki/Convergence_of_random_variables#Convergence_in_distribution) to a centered (i.e, zero mean) normal with standard deviation $\sigma$.

### Intuition

```{index} single: Central Limit Theorem; Intuition
```

The striking implication of the CLT is that for **any** distribution with
finite second moment, the simple operation of adding independent
copies **always** leads to a Gaussian curve.

+++


### Simulation 1

Since the CLT seems almost magical, running simulations that verify its implications is one good way to build intuition.

To this end, we now perform the following simulation

1. Choose an arbitrary distribution $F$ for the underlying observations $X_i$.
1. Generate independent draws of $Y_n := \sqrt{n} ( \bar X_n - \mu )$.
1. Use these draws to compute some measure of their distribution --- such as a histogram.
1. Compare the latter to $N(0, \sigma^2)$.

Here's some code that does exactly this for the exponential distribution
$F(x) = 1 - e^{- \lambda x}$.

(Please experiment with other choices of $F$, but remember that, to conform with the conditions of the CLT, the distribution must have a finite second moment.)

(sim_one)=

```{code-cell} ipython3
# Set parameters
n = 250                  # Choice of n
k = 1_000_000               # Number of draws of Y_n
distribution = st.expon(2)  # Exponential distribution, λ = 1/2
μ, s = distribution.mean(), distribution.std()

# Draw underlying RVs. Each row contains a draw of X_1,..,X_n
data = distribution.rvs((k, n))
# Compute mean of each row, producing k draws of \bar X_n
sample_means = data.mean(axis=1)
# Generate observations of Y_n
Y = np.sqrt(n) * (sample_means - μ)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
xmin, xmax = -3 * s, 3 * s
ax.set_xlim(xmin, xmax)
ax.hist(Y, bins=60, alpha=0.4, density=True)
xgrid = np.linspace(xmin, xmax, 200)
ax.plot(xgrid, st.norm.pdf(xgrid, scale=s), 'k-', lw=2, label='$N(0, \sigma^2)$')
ax.legend()

plt.show()
```

(Notice the absence of for loops --- every operation is vectorized, meaning that the major calculations are all shifted to optimized C code.)

The fit to the normal density is already tight and can be further improved by increasing `n`.



+++

## Exercises

+++

## Ex 1

+++

As the reader to rerun the last simulation and experiment with other specifications of $F$ that have finite second moment, making sure that they 

+++

Although NumPy doesn't give us a `bernoulli` function, we can generate a draw of $X$ using NumPy via

```{code-cell} ipython3
U = np.random.rand()
X = 1 if U < p else 0
print(X)
```

Explain why this provides a random variable $X$ with the right distribution.

+++

Solution:

+++

We can write $X$ as $X = \mathbf 1\{U < p\}$ where $\mathbf 1$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function) (i.e., 1 if the statement is true and zero otherwise).

Here we generated a uniform draw $U$ on $[0,1]$ and then used the fact that

$$
\mathbb P\{0 \leq U < p\} = p - 0 = p
$$

This means that $X = \mathbf 1\{U < p\}$ has the right distribution.

+++

```{solution-end}
```
