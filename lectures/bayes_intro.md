---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction to Bayesian Methods

## Overview

In this lecture we study one of the most important ideas in statistics: how to update our beliefs about an unknown quantity as new data arrives.

The technique we will use is called **Bayesian updating**.

We start with a belief about some unknown number, expressed as a probability distribution.

As we observe data, we revise that belief in a way that is mathematically precise.

We will develop these ideas through a concrete example drawn from development
finance: estimating the default rate on a new type of loan.

Along the way we will meet conditional probability, Bayes' law, the Bernoulli and binomial distributions, and the beta distribution.

Let's begin by importing the libraries we need.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
```

## Conditional probability

Before we can talk about updating beliefs, we need to recall the idea of **conditional probability**.

### Definition

Suppose $A$ and $B$ are two events with $P(B) > 0$.

The probability of $A$ given that $B$ has occurred is written $P(A \mid B)$ and defined by

$$
    P(A \mid B) = \frac{P(A \cap B)}{P(B)}.
$$ (eq:cond_prob)

Here $A \cap B$ is the event that both $A$ and $B$ occur.

The intuition is that learning $B$ has occurred restricts us to the world in which $B$ is true.

Within that restricted world, we ask how likely $A$ is.

### Example: risky borrowing

A bank classifies its borrowers as either *low risk* or *high risk*.

Among all borrowers, 80% are low risk and 20% are high risk.

The conditional default probabilities are:

* low-risk borrowers: 0.05
* high-risk borrowers: 0.40

Suppose we pick a borrower at random and find that they defaulted.

What is the probability that the borrower was high risk?

Let $H = $ high risk, $L = $ low risk and $D = $ default.

We know that $P(H) = 0.2$, $P(L) = 0.8$, $P(D \mid H) = 0.40$ and $P(D \mid L) = 0.05$.

The probability that a borrower is both high risk and defaults is

$$
    P(H \cap D) = P(D \mid H)\, P(H) = 0.40 \times 0.2 = 0.08.
$$

The overall probability of default can be computed from the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability):

$$
    P(D) = P(D \mid H)\, P(H) + P(D \mid L)\, P(L) = 0.40 \times 0.2 + 0.05 \times 0.8 = 0.08 + 0.04 = 0.12.
$$

Applying {eq}`eq:cond_prob`, we get

$$
    P(H \mid D) = \frac{P(H \cap D)}{P(D)} = \frac{0.08}{0.12} \approx 0.667.
$$

So observing a default raises our assessment that the borrower is high risk from 20% to about 67%.


## Bayes' law

The last computation was, in fact, a typical Bayesian calculation.

To formalize ideas, let's consider two abstract events $A$ and $B$.

Notice that the event $A \cap B$ (both $A$ and $B$ occur) is the same as $B \cap A$.

Hence, applying definition {eq}`eq:cond_prob` two ways, we have

$$
    P(A \mid B)\, P(B) = P(A \cap B) = P(B \mid A)\, P(A).
$$

Dividing through by $P(B)$ gives **Bayes' law**:

$$
    P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}.
$$ (eq:bayes_law)

Each piece of {eq}`eq:bayes_law` has a name.

We call $P(A)$ the **prior** — our belief about $A$ before seeing data.

We call $P(B \mid A)$ the **likelihood** — how probable the data $B$ is when $A$ is true.

We call $P(A \mid B)$ the **posterior** — our updated belief about $A$ after seeing $B$.

The denominator $P(B)$ is a normalizing constant that makes the posterior probabilities sum to one.

You can check that the borrower calculation above is exactly Bayes' law with $A = H$ and $B = D$.

Bayes' law tells us how to "reverse" a conditional probability: it converts $P(D \mid H)$, which we know, into $P(H \mid D)$, which we want.

## A microloan default problem

Now we turn to our main example.

A development bank is entering a new lending market — say, smallholder farmers in a region where no historical lending data exists.

The bank makes a series of small loans (microloans).

Each loan either **defaults** or is **repaid**.

We encode the outcome of loan $i$ as a random variable $Y_i$, where

$$
    Y_i =
    \begin{cases}
    1 & \text{if loan } i \text{ defaults}, \\
    0 & \text{if loan } i \text{ is repaid}.
    \end{cases}
$$

Let $\theta$ be the probability that any given loan defaults.

We assume that, given $\theta$, the outcomes $Y_1, Y_2, \ldots$ are independent draws with

$$
    P(Y_i = 1 \mid \theta) = \theta,
    \qquad
    P(Y_i = 0 \mid \theta) = 1 - \theta.
$$

A random variable of this form is called a **Bernoulli** random variable with parameter $\theta$.

The catch is that the bank does **not** know $\theta$.

Since the market is new, $\theta$ is uncertain.

At the same time, the bank has experience in similar markets, so it does not start from total ignorance.

The Bayesian approach is to treat $\theta$ as a random quantity and describe our beliefs about it with a probability density $\pi$ on the interval $[0, 1]$.

This density is called the **prior**.

One possible option for $\pi$ is to use a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), whose density on $[0,1]$ is

$$
    \pi(\theta) = \frac{\theta^{a - 1} (1 - \theta)^{b - 1}}{B(a, b)},
    \qquad
    B(a, b) = \int_0^1 t^{a-1}(1-t)^{b-1}\, dt,
$$

for parameters $a > 0$ and $b > 0$.

The denominator $B(a, b)$ is called the **beta function**, and it is simply the
constant that makes the density integrate to one.

For our purposes the important part is the shape, $\theta^{a-1}(1-\theta)^{b-1}$, which is all that depends on $\theta$.

One nice property of the beta distribution is that, by varying $a$ and $b$, we can represent a wide range of beliefs.

The four examples below range from "no idea at all" to at least some opinion as to whether $\theta$ is 
likely to take a low or high value.

```{code-cell} ipython3
θ_grid = np.linspace(0, 1, 500)
plot_grid = np.linspace(0.001, 0.999, 500)   # avoids the spikes at the endpoints
params = [(0.5, 0.5), (1, 1), (2, 5), (8, 3)]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for (a, b), ax in zip(params, axes.flatten()):
    ax.plot(plot_grid, beta.pdf(plot_grid, a, b), lw=2)
    ax.set_title(f"Beta({a}, {b})")
    ax.set_xlabel(r"$\theta$")
fig.tight_layout()
plt.show()
```

The shapes are strikingly different.

$\text{Beta}(1, 1)$ is flat — it expresses complete ignorance, treating every default rate as equally likely.

$\text{Beta}(0.5, 0.5)$ piles weight near 0 and 1, a belief that the market is probably either very safe or very risky.

$\text{Beta}(2, 5)$ leans toward low default rates, while $\text{Beta}(8, 3)$ leans toward high ones.

For our development bank we will adopt the $\text{Beta}(2, 5)$ prior shown in the bottom left.

```{code-cell} ipython3
a_0, b_0 = 2, 5
```

This prior puts most of its weight on default rates below 0.5, with a peak around 0.2, reflecting cautious optimism together with genuine uncertainty.


## A one-step update

Now suppose the bank observes the outcome of a single loan.

Call this outcome $y \in \{0, 1\}$.

We want to update the prior $\pi(\theta)$ into a **posterior** $\pi(\theta \mid y)$.

The probability of $y$ given $\theta$ is 

$$
    p(y \mid \theta) = \theta^{y} (1 - \theta)^{1 - y}.
$$ (eq:bernoulli_lik)

This formula (which is called the Bernoulli distribution) gives us the right numbers:

* When $y = 1$ it gives $\theta^1 (1-\theta)^0 = \theta$, the probability of a default.
* When $y = 0$ it gives $\theta^0 (1-\theta)^1 = 1 - \theta$, the probability of repayment.

Bayes' law for a continuous parameter takes the form

$$
    \pi(\theta \mid y) = \frac{p(y \mid \theta)\, \pi(\theta)}{\int_0^1 p(y \mid t)\, \pi(t)\, dt}.
$$ (eq:bayes_density)

This is the exact analogue of {eq}`eq:bayes_law`.

The numerator is likelihood times prior.

The denominator is an integral that sums the numerator over all possible values of $\theta$, ensuring the posterior integrates to one.

Substituting the Bernoulli likelihood {eq}`eq:bernoulli_lik` gives our complete one-step update rule:

$$
    \pi(\theta \mid y) = \frac{\theta^{y}(1-\theta)^{1-y}\, \pi(\theta)}{\int_0^1 t^{y}(1-t)^{1-y}\, \pi(t)\, dt}.
$$

This rule takes any prior $\pi$ and any single observation $y$ and returns the posterior.


## Computing the update numerically

The integral in the denominator of {eq}`eq:bayes_density` is not trivial to compute.

A simple and general approach is to compute it numerically, using a technique
such as the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).

We lay down a fine grid of points across $[0, 1]$ and represent each density by its values at those grid points.

Every integral then becomes a sum that `numpy` can evaluate for us.

Let's build the update in two steps.

First, recall that the denominator in {eq}`eq:bayes_density` is the integral

$$
    \int_0^1 p(y \mid t)\, \pi(t)\, dt .
$$

This is just the likelihood times the prior, integrated over all values of $\theta$.

The function below computes it on the grid, approximating the integral with `np.trapezoid`.

```{code-cell} ipython3
def normalizing_constant(prior_vals, y, θ_grid):
    "Compute the denominator of Bayes' law: likelihood times prior, integrated."
    likelihood = θ_grid**y * (1 - θ_grid)**(1 - y)
    return np.trapezoid(likelihood * prior_vals, θ_grid)
```

The second step divides the (unnormalized) product of likelihood and prior by this constant.

The result is the posterior density, evaluated on the grid.

```{code-cell} ipython3
def update(prior_vals, y, θ_grid):
    "Update prior density values to posterior values after observing y."
    likelihood = θ_grid**y * (1 - θ_grid)**(1 - y)
    C = normalizing_constant(prior_vals, y, θ_grid)
    return likelihood * prior_vals / C
```

Let's start from our Beta(2, 5) prior and suppose that we observe a single default ($y = 1$).

```{code-cell} ipython3
prior_vals = beta.pdf(θ_grid, a_0, b_0)
posterior_vals = update(prior_vals, y=1, θ_grid=θ_grid)

fig, ax = plt.subplots()
ax.plot(θ_grid, prior_vals, lw=2, label="prior")
ax.plot(θ_grid, posterior_vals, lw=2, label="posterior after one default")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.legend()
plt.show()
```

Observing a default shifts the posterior to the right, toward higher default probabilities.

This makes sense: a default is evidence that $\theta$ might be larger than we thought.

If instead the loan had been repaid, the posterior would shift left.

```{code-cell} ipython3
posterior_repaid = update(prior_vals, y=0, θ_grid=θ_grid)

fig, ax = plt.subplots()
ax.plot(θ_grid, prior_vals, lw=2, label="prior")
ax.plot(θ_grid, posterior_repaid, lw=2, label="posterior after one repayment")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.legend()
plt.show()
```

## A closed form: the beta prior

The numerical approach always works, but in this case there is something special going on.

It has to do with the fact that we chose a Beta prior.

Recall that this prior is proportional to $\theta^{a-1}(1-\theta)^{b-1}$.

Multiply it by the Bernoulli likelihood $\theta^{y}(1-\theta)^{1-y}$:

$$
    \theta^{y}(1-\theta)^{1-y} \cdot \theta^{a-1}(1-\theta)^{b-1}
    = \theta^{(a + y) - 1}(1-\theta)^{(b + 1 - y) - 1}.
$$

The right-hand side has exactly the form of another beta density.

So if the prior is $\text{Beta}(a, b)$, the posterior is again a beta distribution — with updated parameters.

We say that the beta distribution is a **conjugate prior** for the Bernoulli likelihood.

The update rule for the parameters is beautifully simple:

- a default ($y = 1$) sends $(a, b) \mapsto (a + 1,\, b)$,
- a repayment ($y = 0$) sends $(a, b) \mapsto (a,\, b + 1)$.

Let's verify that this closed form agrees with our numerical computation.

We overlay the analytical $\text{Beta}(a_0 + 1, b_0)$ posterior on the grid posterior after one default.

```{code-cell} ipython3
closed_form = beta.pdf(θ_grid, a_0 + 1, b_0)

fig, ax = plt.subplots()
ax.plot(θ_grid, posterior_vals, lw=4, alpha=0.5, label="numerical posterior")
ax.plot(θ_grid, closed_form, ls='--', lw=2, label=f"Beta({a_0 + 1}, {b_0})")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.legend()
plt.show()
```

The two curves lie exactly on top of one another, confirming that our numerics and the closed form agree.

## Iterating the update

Here is a key observation: the posterior after one step is itself a perfectly good prior for the next step.

So we can repeat the update as each new loan resolves.

Starting from a prior $\pi_0$, observing $Y_1$ gives a posterior $\pi_1$.

Treating $\pi_1$ as the new prior and observing $Y_2$ gives $\pi_2$, and so on.

This produces a sequence of densities $\pi_0, \pi_1, \pi_2, \ldots$ that captures our evolving beliefs.

Let's simulate a stream of loan outcomes and watch the beliefs evolve.

We will generate data from a "true" default rate $\theta^* = 0.15$, which the bank does not know.

```{code-cell} ipython3
θ_true = 0.15
n = 100
rng = np.random.default_rng(seed=42)
outcomes = (rng.random(n) < θ_true).astype(int)
```

Now we iterate the grid update over these outcomes, recording the posterior at a few selected stages.

```{code-cell} ipython3
snapshots = [1, 5, 20, 100]
current = beta.pdf(θ_grid, a_0, b_0)

fig, ax = plt.subplots()
ax.plot(θ_grid, current, 'k-', lw=2, alpha=0.7, label="prior")

for i in range(1, n + 1):
    current = update(current, outcomes[i - 1], θ_grid)
    if i in snapshots:
        ax.plot(θ_grid, current, lw=2, label=f"posterior after {i} loans")

ax.axvline(θ_true, color='k', ls=':', label=r"true $\theta^*$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.legend()
plt.show()
```

Two things stand out.

First, the posterior **concentrates** around the true value $\theta^* = 0.15$ as more loans resolve.

Second, the posterior becomes **tighter** — our uncertainty about $\theta$ steadily shrinks.

Early on, the prior has a strong influence on our beliefs.

As data accumulates, that influence fades and the data takes over.

## The batch update via the binomial likelihood

There is a second, equally natural way to think about the same problem.

Instead of processing outcomes one at a time, suppose we wait and observe all $n$ outcomes $Y_1, \ldots, Y_n$ together.

Then we update directly from the prior $\pi_0$ to the posterior $\pi_n$ in a single step.

For independent Bernoulli draws, the only feature of the data that matters is the **total number of defaults**,

$$
    k = \sum_{i=1}^n Y_i.
$$

The probability of observing exactly $k$ defaults out of $n$ loans is given by the **binomial** distribution:

$$
    p(k \mid \theta) = \binom{n}{k} \theta^{k} (1 - \theta)^{n - k}.
$$ (eq:binom_lik)

Bayes' law then gives the one-shot update

$$
    \pi_n(\theta) = \frac{p(k \mid \theta)\, \pi_0(\theta)}{\int_0^1 p(k \mid t)\, \pi_0(t)\, dt}.
$$

With a $\text{Beta}(a, b)$ prior, conjugacy again gives a clean closed form.

The binomial likelihood is proportional to $\theta^{k}(1-\theta)^{n-k}$, so the posterior is

$$
    \text{Beta}(a + k,\ b + n - k).
$$

In words: add the number of defaults to $a$, and add the number of repayments to $b$.

## Sequential and batch updates agree

We now have two routes from prior to posterior.

One processes the $n$ outcomes one at a time; the other processes them all at once.

Reassuringly, they give exactly the same answer.

We can see why with a short algebraic argument.

The likelihood of the full sequence $Y_1, \ldots, Y_n$, for given $\theta$, is the product of the individual Bernoulli likelihoods:

$$
    \prod_{i=1}^n \theta^{Y_i}(1-\theta)^{1 - Y_i}
    = \theta^{k}(1-\theta)^{n-k}.
$$

This is identical to the binomial likelihood {eq}`eq:binom_lik`, except for the factor $\binom{n}{k}$.

But $\binom{n}{k}$ does not depend on $\theta$, so it cancels between the numerator and denominator when we normalize.

Hence the two posteriors are exactly equal.

Let's confirm this numerically.

We compare the sequential grid posterior from before against a single binomial update on the same data.

```{code-cell} ipython3
k = outcomes.sum()           # total number of defaults

# Route 1: sequential update, one outcome at a time
seq_post = beta.pdf(θ_grid, a_0, b_0)
for y in outcomes:
    seq_post = update(seq_post, y, θ_grid)

# Route 2: single batch update with the binomial likelihood
binom_lik = binom.pmf(k, n, θ_grid)
batch_post = binom_lik * beta.pdf(θ_grid, a_0, b_0)
batch_post = batch_post / np.trapezoid(batch_post, θ_grid)

fig, ax = plt.subplots()
ax.plot(θ_grid, seq_post, lw=4, alpha=0.5, label="sequential (one at a time)")
ax.plot(θ_grid, batch_post, ls='--', lw=2, label="batch (binomial)")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.legend()
plt.show()
```

The curves coincide.


## From posterior to loan pricing

Why does any of this matter for the bank?

The whole point of estimating $\theta$ is to make better lending decisions.

Suppose each loan has size 1, and the bank loses the full amount when a loan defaults.

Then the **expected loss** on a new loan is just the probability of default.

Given the data, our best estimate of that probability is the **posterior mean**.

Recall that the posterior is $\text{Beta}(a + k,\ b + n - k)$.

The [mean of a beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) $\text{Beta}(\alpha, \beta)$ is $\alpha / (\alpha + \beta)$, so here

$$
    \mathbb{E}[\theta \mid \text{data}] = \frac{a + k}{a + b + n}.
$$

This is the number the bank should plug into its pricing: to break even, the interest rate must at least cover the expected loss.

Let's track how the expected loss, and our uncertainty about it, evolve as loans resolve.

```{code-cell} ipython3
cum_defaults = np.cumsum(outcomes)
loans = np.arange(1, n + 1)

a_n = a_0 + cum_defaults
b_n = b_0 + loans - cum_defaults

post_mean = a_n / (a_n + b_n)
post_std = np.sqrt(a_n * b_n / ((a_n + b_n)**2 * (a_n + b_n + 1)))

fig, ax = plt.subplots()
ax.plot(loans, post_mean, lw=2, label="posterior mean (expected loss)")
ax.fill_between(loans, post_mean - post_std, post_mean + post_std,
                alpha=0.2, label="± one posterior std. dev.")
ax.axhline(θ_true, color='k', ls=':', label=r"true $\theta^*$")
ax.set_xlabel("number of loans observed")
ax.set_ylabel(r"estimate of $\theta$")
ax.legend()
plt.show()
```

As the bank observes more loans, its estimate of the expected loss settles down near the true default rate.

At the same time, the band of uncertainty narrows.

This is the practical payoff of Bayesian updating: the bank can price cautiously when data is scarce, and sharpen its pricing as experience accumulates.

## The break-even interest rate

Let's turn the expected loss into an actual interest rate.

Consider a loan of size 1, carrying interest rate $r$.

If the borrower repays, the bank receives $1 + r$.

If the borrower defaults, the bank receives nothing.

Suppose the bank is risk neutral, faces no funding costs, and operates in a competitive market that drives expected profits to zero.

The zero-profit condition sets the expected repayment equal to the amount lent:

$$
    (1 - \theta)(1 + r) = 1 .
$$

Solving for $r$ gives the break-even interest rate

$$
    r = \frac{\theta}{1 - \theta} .
$$

This rate has a clean interpretation: the expected interest income $(1 - \theta) r$ exactly equals the expected loss $\theta$.

Of course the bank does not know $\theta$.

Because the bank is risk neutral, only the mean of $\theta$ enters the calculation, so it replaces $\theta$ by its posterior mean $\hat\theta = \mathbb{E}[\theta \mid \text{data}]$:

$$
    r = \frac{\hat\theta}{1 - \hat\theta} .
$$

Let's see how this break-even rate evolves as the loan book grows.

```{code-cell} ipython3
implied_rate = post_mean / (1 - post_mean)

fig, ax = plt.subplots()
ax.plot(loans, implied_rate, lw=2, label="break-even rate")
ax.axhline(θ_true / (1 - θ_true), color='k', ls=':',
           label=r"rate at true $\theta^*$")
ax.set_xlabel("number of loans observed")
ax.set_ylabel(r"interest rate $r$")
ax.legend()
plt.show()
```

Early on, with little data, the rate reflects the prior and is somewhat unstable.

As loans resolve, it settles toward $\theta^* / (1 - \theta^*) \approx 0.176$, the rate the bank would charge if it knew the true default probability.

## Exercises

```{exercise}
:label: bayes_ex1

The prior matters a lot when data is scarce, but its influence should fade as data accumulates.

Illustrate this by repeating the iteration of the previous sections from two very different priors:

* an *optimistic* prior $\text{Beta}(2, 8)$, which expects a low default rate, and
* a *skeptical* prior $\text{Beta}(8, 2)$, which expects a high default rate.

Use the same simulated data (with $\theta^* = 0.15$) for both.

Plot the two posterior means as a function of the number of loans observed, and confirm that they converge as $n$ grows.
```

```{solution-start} bayes_ex1
:class: dropdown
```

We can use the conjugate update rule directly, since both priors are beta distributions.

```{code-cell} ipython3
priors = {"optimistic Beta(2, 8)": (2, 8),
          "skeptical Beta(8, 2)": (8, 2)}

cum_defaults = np.cumsum(outcomes)
loans = np.arange(1, n + 1)

fig, ax = plt.subplots()
for label, (a, b) in priors.items():
    a_seq = a + cum_defaults
    b_seq = b + loans - cum_defaults
    ax.plot(loans, a_seq / (a_seq + b_seq), lw=2, label=label)

ax.axhline(θ_true, color='k', ls=':', label=r"true $\theta^*$")
ax.set_xlabel("number of loans observed")
ax.set_ylabel("posterior mean")
ax.legend()
plt.show()
```

The two estimates start far apart, reflecting the disagreement between the priors.

As loans accumulate, both are pulled toward the true value $\theta^* = 0.15$ and the gap between them shrinks.

With enough data, the choice of prior barely matters — the data dominates.

```{solution-end}
```

```{exercise}
:label: bayes_ex2

Suppose the bank starts with a $\text{Beta}(2, 5)$ prior and then observes 200 loans, of which 26 default.

1. Compute the posterior mean default rate.
2. Compute a 90% **credible interval** for $\theta$ — an interval that contains $\theta$ with posterior probability 0.90.

For the credible interval, use the 5th and 95th percentiles of the posterior, available via `beta.ppf`.
```

```{solution-start} bayes_ex2
:class: dropdown
```

The posterior is $\text{Beta}(a_0 + k,\ b_0 + n - k)$ with $a_0 = 2$, $b_0 = 5$, $n = 200$ and $k = 26$.

```{code-cell} ipython3
a_post = 2 + 26
b_post = 5 + (200 - 26)

post_mean = a_post / (a_post + b_post)
lower, upper = beta.ppf([0.05, 0.95], a_post, b_post)

print(f"Posterior: Beta({a_post}, {b_post})")
print(f"Posterior mean default rate: {post_mean:.4f}")
print(f"90% credible interval: ({lower:.4f}, {upper:.4f})")
```

The bank's best estimate of the default rate is about 13.5%, and it is 90% sure that the true rate lies within the reported interval.

These two numbers — a point estimate and a measure of uncertainty — are exactly what is needed to price the loans.

```{solution-end}
```
