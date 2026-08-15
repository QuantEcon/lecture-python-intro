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

(bivariate_dist)=
# Bivariate Distributions

```{index} single: Bivariate Distributions
```

## Outline

The lectures {doc}`prob_dist`, {doc}`observed_distributions` and {doc}`fitting_distributions` all study a single variable at a time --- a set of heights, a set of house prices.

In practice we are usually interested in more than one variable, and, more importantly, in how they move together.

Do taller people also tend to weigh more?

Do larger houses also tend to sell for more?

In this lecture we give a quick introduction to **bivariate distributions**: probability distributions over pairs of random variables.

We cover joints and marginals, independence, covariance and correlation, some of the ways that joint distributions arise, and the **bivariate normal distribution**.

As before, we then turn to observed data, and fit a bivariate normal by the method of moments.

We end with a preview of {doc}`simple_linear_regression`, showing that the regression line is already hiding inside the bivariate normal distribution.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

To motivate what follows, let's look at the sale prices and floor areas of the Ames houses that we studied in {doc}`observed_distributions` and {doc}`fitting_distributions` {cite}`decock2011ames`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: House price against floor area
    name: fig:bivariate-ames-scatter
tags: [hide-input]
---
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/ames_house_prices.csv')
houses = pd.read_csv(url)
price = houses['price']
area = houses['living_area_sqft']

fig, ax = plt.subplots()
ax.scatter(area, price, alpha=0.3, s=10)
ax.set_xlabel('living area (square feet)')
ax.set_ylabel('sale price (US$)')
plt.show()
```

Each point is one house.

In {doc}`observed_distributions` we compared prices across four floor-area groups using box plots, and found that floor area is a good predictor of price.

The scatter plot above shows the same fact directly, without first cutting the data into groups: as area increases, price tends to increase with it, although not perfectly --- for any given area there is still a range of prices.

This is exactly the kind of pattern that a bivariate distribution is built to describe.

## Joint distributions

### Discrete case

Let's start with two discrete random variables $X$ and $Y$, taking values in finite sets $S_X$ and $S_Y$.

The **joint probability mass function** of $X$ and $Y$ is the function $p$ on $S_X \times S_Y$ with

$$
p(x,y) = \mathbb P\{X = x, Y = y\}
$$

As with a single variable, the values of $p$ are nonnegative and sum to one, now over both variables:

$$
\sum_{x \in S_X} \sum_{y \in S_Y} p(x,y) = 1
$$

Let's build an example from the house price data.

Let $X = 1$ if a house's floor area is above the sample median and $X=0$ otherwise, and define $Y$ the same way for price.

```{code-cell} ipython3
X = (area > area.median()).astype(int)
Y = (price > price.median()).astype(int)
joint = pd.crosstab(X, Y, normalize=True)
joint.index.name = 'x (area above median)'
joint.columns.name = 'y (price above median)'
joint
```

This table *is* the joint PMF: each cell is the fraction of houses with that particular combination of $X$ and $Y$.

For example, the top-left cell tells us that around 39\% of houses have both below-median area and below-median price.

### Continuous case

Most of the variables we care about, such as price and area themselves, are continuous rather than discrete.

The continuous analog of the joint PMF is the **joint probability density function** $p(x,y)$, a nonnegative function on $\mathbb R^2$ with

$$
\iint_{\mathbb R^2} p(x,y) \, dx \, dy = 1
$$

We say that $(X,Y)$ has joint density $p$ if, for every region $A \subset \mathbb R^2$,

$$
\mathbb P\{(X,Y) \in A\} = \iint_A p(x,y) \, dx \, dy
$$

We will meet a specific and very useful example, the bivariate normal density, below.

## Marginal distributions

Given a joint distribution, we can always recover the distribution of $X$ on its own, by summing (or integrating) out $Y$.

In the discrete case,

$$
p_X(x) = \sum_{y \in S_Y} p(x,y)
$$

and symmetrically for $p_Y$.

We call $p_X$ and $p_Y$ the **marginal distributions** of $X$ and $Y$.

In our table above, the marginals are the row and column sums:

```{code-cell} ipython3
joint.sum(axis=1), joint.sum(axis=0)
```

Both marginals are close to 0.5, which makes sense: $X$ and $Y$ were each constructed to be 1 for exactly half of the sample.

Notice, though, that the marginals alone do not tell us the whole story.

Knowing that $\mathbb P\{X=1\} \approx 0.5$ and $\mathbb P\{Y=1\} \approx 0.5$ says nothing about whether large houses also tend to be the expensive ones --- for that we need the joint distribution, not just the two marginals separately.

In the continuous case the same idea applies, with sums replaced by integrals:

$$
p_X(x) = \int_{-\infty}^\infty p(x,y) \, dy
$$

## Independence

$X$ and $Y$ are called **independent** if the joint distribution factors into the product of the marginals:

$$
p(x,y) = p_X(x) \, p_Y(y) \qquad \text{for all } x, y
$$

Independence means that learning the value of $X$ tells us nothing about $Y$, and vice versa.

Let's check whether our discrete example is close to independent, by comparing the joint table with the table we would get if $X$ and $Y$ *were* independent, i.e. the product of the marginals.

```{code-cell} ipython3
p_X, p_Y = joint.sum(axis=1), joint.sum(axis=0)
independent_table = pd.DataFrame(np.outer(p_X, p_Y),
                                  index=joint.index, columns=joint.columns)
independent_table
```

Under independence, every cell would be close to $0.5 \times 0.5 = 0.25$.

Let's put the two tables side by side as heatmaps, which makes the difference much easier to see than comparing raw numbers.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Actual joint versus independent joint
    name: fig:bivariate-independence-heatmap
---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.heatmap(joint, annot=True, fmt='.2f', cmap='viridis', cbar=False,
            vmin=0, vmax=0.45, ax=axes[0])
axes[0].set_title('actual joint')
sns.heatmap(independent_table, annot=True, fmt='.2f', cmap='viridis', cbar=False,
            vmin=0, vmax=0.45, ax=axes[1])
axes[1].set_title('if independent')
plt.show()
```

The independent heatmap is flat, with every cell close to 0.25.

The actual heatmap is not: it is heavy on the diagonal, with around 39% of houses in the (below, below) and (above, above) cells, and only around 11% off the diagonal.

So $X$ and $Y$ are far from independent --- exactly as we would expect, since larger houses tend to be more expensive.

## Covariance and correlation

Independence is an all-or-nothing property.

To measure the *strength* and *direction* of dependence, we use the **covariance**

$$
\mathrm{Cov}(X,Y) = \mathbb E \left[ (X - \mu_X)(Y - \mu_Y) \right]
$$

where $\mu_X = \mathbb E[X]$ and $\mu_Y = \mathbb E[Y]$.

The covariance is positive when $X$ and $Y$ tend to be above their means together (and below their means together), and negative when one tends to be above its mean while the other is below.

If $X$ and $Y$ are independent, then $\mathrm{Cov}(X,Y) = 0$.

```{note}
The converse is false: zero covariance does not imply independence.

It only rules out *linear* dependence.

We will see an example of this below, in the discussion of the bivariate normal distribution.
```

The covariance is measured in the units of $X$ times the units of $Y$, which makes it hard to interpret on its own.

We therefore usually standardize it into the **correlation coefficient**

$$
\rho = \mathrm{Corr}(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$

where $\sigma_X$ and $\sigma_Y$ are the standard deviations of $X$ and $Y$.

Correlation is unit-free and always lies in $[-1, 1]$, with the extreme values attained only when $Y$ is an exact linear function of $X$.

Let's compute it for our discrete house example.

```{code-cell} ipython3
np.corrcoef(X, Y)[0, 1]
```

A correlation of around 0.56 between the above/below-median indicators confirms what the table already showed us: floor area and price move together.

## How joint distributions arise

It is worth pausing to think about *why* two variables end up correlated.

Here are two simple and common mechanisms.

### Independent components

The simplest case is no relationship at all: draw $X$ and $Y$ independently.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Two independent normal variables
    name: fig:bivariate-independent
---
rng = np.random.default_rng(1234)
N = 500
x_indep = rng.standard_normal(N)
y_indep = rng.standard_normal(N)

fig, ax = plt.subplots()
ax.scatter(x_indep, y_indep, alpha=0.5, s=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.show()
```

There is no visible pattern: knowing $x$ tells us nothing about where $y$ will land.

### A common building block

A second and very common mechanism is that $Y$ is built partly *from* $X$.

For example, suppose

$$
Y = a X + b + U
$$

where $U$ is noise, independent of $X$, with mean zero and standard deviation $\sigma_U$.

(Think of $X$ as floor area and $Y$ as price: bigger houses mechanically cost more to build, plus some noise from location, finish quality, and so on.)

Since $U$ is independent of $X$, $\mathrm{Cov}(X, U) = 0$, and so

$$
\mathrm{Cov}(X,Y) = \mathrm{Cov}(X, aX + b + U) = a \, \mathrm{Cov}(X,X) = a \sigma_X^2
$$

Similarly, $\mathbb V[Y] = a^2 \sigma_X^2 + \sigma_U^2$, so that

$$
\mathrm{Corr}(X,Y) = \frac{a \sigma_X}{\sqrt{a^2 \sigma_X^2 + \sigma_U^2}}
$$

The correlation is driven entirely by the *relative* size of the signal ($a X$) and the noise ($U$).

Let's see this in a picture, fixing $a=1$ and $\sigma_X = 1$ and increasing $\sigma_U$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Correlation strength as noise grows
    name: fig:bivariate-signal-noise
---
a = 1.0
sigma_U_vals = [0.2, 1.0, 3.0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, sigma_U in zip(axes, sigma_U_vals):
    x = rng.standard_normal(N)
    u = rng.normal(scale=sigma_U, size=N)
    y = a * x + u
    rho = a / np.sqrt(a**2 + sigma_U**2)
    ax.scatter(x, y, alpha=0.5, s=10)
    ax.set_title(rf'$\sigma_U={sigma_U}$, $\rho={rho:.2f}$')
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
plt.show()
```

As the noise grows, the cloud of points fattens and the correlation falls, even though the underlying mechanism --- $Y$ built from $X$ plus independent noise --- never changes.

This is a useful mental model to have in mind whenever you see two correlated variables: often there is some shared component driving both, plus independent noise on top.

## The bivariate normal distribution

Just as the normal distribution is the workhorse univariate distribution, the **bivariate normal distribution** is the workhorse joint distribution.

It has five parameters: the two means $\mu_X, \mu_Y$, the two standard deviations $\sigma_X, \sigma_Y$, and the correlation $\rho \in (-1,1)$.

Its density is

$$
p(x,y) = \frac{1}{2\pi \sigma_X \sigma_Y \sqrt{1-\rho^2}}
\exp\left(
-\frac{1}{2(1-\rho^2)}
\left[
\frac{(x-\mu_X)^2}{\sigma_X^2}
- \frac{2\rho (x-\mu_X)(y-\mu_Y)}{\sigma_X \sigma_Y}
+ \frac{(y-\mu_Y)^2}{\sigma_Y^2}
\right]
\right)
$$

It can be shown that, for this distribution, the marginal of $X$ is $N(\mu_X, \sigma_X^2)$, the marginal of $Y$ is $N(\mu_Y, \sigma_Y^2)$, and $\rho$ is exactly the correlation between $X$ and $Y$.

SciPy provides this distribution as `scipy.stats.multivariate_normal`, which takes a mean and a covariance matrix (built here from $\sigma_X, \sigma_Y, \rho$).

```{code-cell} ipython3
def bivariate_normal(μ_x, μ_y, σ_x, σ_y, ρ):
    cov = [[σ_x**2, ρ * σ_x * σ_y],
           [ρ * σ_x * σ_y, σ_y**2]]
    return scipy.stats.multivariate_normal([μ_x, μ_y], cov)
```

Before looking at it from above, let's see the density as it really is: a surface over the $(x,y)$ plane.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The bivariate normal density surface
    name: fig:bivariate-normal-surface
---
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

u = bivariate_normal(0, 0, 1, 1, 0.6)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X_mesh, Y_mesh, u.pdf(pos), cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('density')
plt.show()
```

This is the same kind of bell-shaped hill as the univariate normal density, just built over a plane instead of a line, and tilted by the correlation $\rho$.

In practice it is much more convenient to look straight down at this hill and draw its contour lines, the way a topographic map shows the shape of a hill without drawing it in 3D.

Let's do that for a few values of $\rho$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bivariate normal contours by correlation
    name: fig:bivariate-normal-contours
---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, ρ in zip(axes, (-0.8, 0.0, 0.8)):
    u = bivariate_normal(0, 0, 1, 1, ρ)
    ax.contour(X_mesh, Y_mesh, u.pdf(pos), levels=6, cmap='viridis')
    ax.set_title(rf'$\rho={ρ}$')
    ax.set_xlabel('x')
    ax.set_aspect('equal')
axes[0].set_ylabel('y')
plt.show()
```

When $\rho = 0$ the contours are circles: the density factors into the product of two independent normal densities, so $X$ and $Y$ are independent.

When $\rho \neq 0$ the contours become tilted ellipses, oriented along the line $y=x$ when $\rho > 0$ and $y=-x$ when $\rho < 0$.

For the bivariate normal --- and *only* for the bivariate normal --- zero correlation is equivalent to independence.

```{note}
Two further properties that are special to the bivariate normal, and useful to know:

* any linear combination $a X + b Y$ is normally distributed, and
* the conditional distribution of $Y$ given $X=x$ is itself normal, with a mean that is *linear* in $x$.

We use the first property in the next subsection, and the second property later, when we preview linear regression.
```

It is worth seeing what a *sample* from this distribution looks like, since real data will never arrive as a clean density --- only as points.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Simulated draws from a bivariate normal
    name: fig:bivariate-normal-sample
---
u = bivariate_normal(0, 0, 1, 1, 0.7)
sample = u.rvs(500, random_state=1234)

fig, ax = plt.subplots()
ax.scatter(sample[:, 0], sample[:, 1], alpha=0.4, s=10)
ax.contour(X_mesh, Y_mesh, u.pdf(pos), levels=6, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

The points scatter around the contours in exactly the elliptical cloud shape we should now expect, denser near the center and thinner in the tails.

Keep this picture in mind: it is the shape we will be looking for when we turn to real data below.

### A word of caution

It is tempting to think that, if $X$ and $Y$ are each individually normal, then the pair $(X,Y)$ must be bivariate normal.

This is false.

Here is a simple counterexample.

Let $X$ be standard normal, and construct $Y$ by

$$
Y = \begin{cases} X & \text{if } |X| < c \\ -X & \text{if } |X| \geq c \end{cases}
$$

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normal marginals, but not jointly normal
    name: fig:bivariate-not-normal
---
N = 2000
c = 1.0
x = rng.standard_normal(N)
y = np.where(np.abs(x) < c, x, -x)

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.4, s=8)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

Since the standard normal density is symmetric about zero, flipping the sign of $X$ whenever $|X| \geq c$ does not change its distribution, so $Y$ is also standard normal.

```{code-cell} ipython3
scipy.stats.skew(y), scipy.stats.kurtosis(y)
```

Both are close to zero, as they should be for a normal marginal.

Yet the joint distribution of $(X,Y)$ looks nothing like the elliptical contours above --- it is concentrated on two crossing lines.

We noted above that, for a genuinely bivariate normal pair, *every* linear combination $aX+bY$ is normal.

That gives us a sharper test than checking the marginals one at a time: let's look at $X+Y$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Sum of the two variables
    name: fig:bivariate-sum-not-normal
---
s = x + y

fig, ax = plt.subplots()
ax.hist(s, bins=60, density=True)
ax.set_xlabel('x + y')
ax.set_ylabel('density')
plt.show()
```

Roughly a third of the mass sits in a single spike at zero --- exactly the houses with $|X| \geq c$, for which $Y=-X$ and so $X+Y=0$ --- with the rest spread out on either side.

This is about as far from a normal density as a distribution can look, even though the skewness and kurtosis of $X+Y$ are both close to zero and would tell a much less dramatic story.

The lesson: checking that each variable *individually* looks normal is not enough to justify a bivariate normal model, and even standard numerical diagnostics can miss a failure that a picture catches instantly.

We should check the joint distribution directly, using the tools below.

## Back to the data

Let's return to the house price and floor area data, and look at their joint distribution more closely.

We already saw the raw scatter plot at the start of the lecture.

A histogram-style alternative is the two-dimensional analog of the histograms we used in {doc}`observed_distributions`: a **hexbin plot**, which counts the number of points falling into each small hexagonal bin and colors the bin accordingly.

Let's draw one using `seaborn`, together with the marginal histogram of each variable along the edges --- a direct picture of the joint distribution and the two marginal distributions we defined above, all in one figure.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Joint and marginal distributions
    name: fig:bivariate-jointplot
tags: [hide-input]
---
g = sns.jointplot(x=area, y=price, kind='hex', height=5)
g.set_axis_labels('living area (square feet)', 'sale price (US$)')
plt.show()
```

The center panel is the joint distribution, in hexbin form, and matches the shape we already saw in the scatter plot.

The panels on the top and right are the two marginal distributions, exactly the row and column sums we used to build marginals in the discrete example above, now applied to the raw continuous data.

The sample correlation confirms the positive relationship numerically.

```{code-cell} ipython3
np.corrcoef(area, price)[0, 1]
```

Recall from {doc}`fitting_distributions` that taking logarithms made the price data look much closer to normal.

The same is true of floor area, and the correlation between the two logged variables is almost the same as before.

```{code-cell} ipython3
log_price = np.log(price)
log_area = np.log(area)
np.corrcoef(log_area, log_price)[0, 1]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Log price against log area
    name: fig:bivariate-log-scatter
tags: [hide-input]
---
fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.3, s=10)
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
plt.show()
```

This log-log scatter plot looks like a good candidate for a bivariate normal fit: an elliptical cloud, roughly symmetric along its main axis.

## Fitting a bivariate normal by the method of moments

Recall from {doc}`fitting_distributions` that the method of moments chooses parameters by matching sample moments to population moments.

The bivariate normal has five parameters, so we match five sample moments: the two means, the two standard deviations, and the correlation.

```{code-cell} ipython3
def fit_bivariate_normal(x, y):
    μ_x, μ_y = x.mean(), y.mean()
    σ_x, σ_y = x.std(), y.std()
    ρ = np.corrcoef(x, y)[0, 1]
    return bivariate_normal(μ_x, μ_y, σ_x, σ_y, ρ), (μ_x, μ_y, σ_x, σ_y, ρ)
```

```{code-cell} ipython3
fitted, (μ_x, μ_y, σ_x, σ_y, ρ) = fit_bivariate_normal(log_area, log_price)
μ_x, μ_y, σ_x, σ_y, ρ
```

Let's overlay the contours of the fitted density on the scatter plot of the data.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fitted bivariate normal density
    name: fig:bivariate-fit
---
x_grid = np.linspace(log_area.min(), log_area.max(), 100)
y_grid = np.linspace(log_price.min(), log_price.max(), 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.2, s=8)
ax.contour(X_mesh, Y_mesh, fitted.pdf(pos), levels=6, cmap='viridis')
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
plt.show()
```

The elliptical contours line up well with the shape of the cloud of points.

The fit is not perfect --- real data rarely is --- but it captures the center, spread, and tilt of the data reasonably well.

## Preview: regression as a conditional mean

We noted above that, for the bivariate normal, the conditional distribution of $Y$ given $X=x$ is normal with a mean that is linear in $x$.

The formula for that conditional mean is

$$
\mathbb E[Y \mid X=x] = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X} (x - \mu_X)
$$

This is a genuinely useful fact: it tells us the best guess of $Y$ (in a mean-squared-error sense) given that we observe $X=x$, and it is a straight line in $x$.

Let's plot it on top of our fitted contours.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The conditional mean line
    name: fig:bivariate-conditional-mean
---
slope = ρ * σ_y / σ_x
intercept = μ_y - slope * μ_x

fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.2, s=8)
ax.plot(x_grid, intercept + slope * x_grid, 'k--', lw=2,
        label=r'$\mathbb{E}[Y \mid X=x]$')
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
ax.legend()
plt.show()
```

This line runs straight through the middle of the cloud of points, in the direction that best summarizes how $y$ moves with $x$.

In fact, if we compute the ordinary least squares line for this data --- the method used in {doc}`simple_linear_regression` --- we get exactly the same slope and intercept.

```{code-cell} ipython3
β, α = np.polyfit(log_area, log_price, 1)
(α, β), (intercept, slope)
```

This is not a coincidence.

Fitting a line by OLS and computing the conditional mean of a fitted bivariate normal are, for this kind of data, the same calculation viewed two different ways.

{doc}`simple_linear_regression` develops the OLS approach in full, without relying on an assumption of joint normality.

## Exercises

```{exercise}
:label: bivariate_ex1

In {doc}`observed_distributions` we compared the monthly returns on Amazon and Costco shares using violin plots, treating each series separately.

Using `yfinance`, download monthly closing prices for `'AMZN'` and `'COST'` between 2000-1-1 and 2024-1-1, and compute monthly returns (percent change) for each, as in {doc}`observed_distributions`.

1. Produce a scatter plot of the two return series and compute their sample correlation.
2. Fit a bivariate normal by the method of moments and overlay its contours on the scatter plot.
3. Recall from {doc}`fitting_distributions` that monthly stock returns have heavier tails than the normal distribution.

   Given this, would you expect the bivariate normal fit here to be better or worse than the fit we found for house prices?

   Check your answer against the figure.
```

```{solution-start} bivariate_ex1
:class: dropdown
```

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade yfinance
```

```{code-cell} ipython3
import yfinance as yf

df_amzn = yf.download('AMZN', '2000-1-1', '2024-1-1', interval='1mo')
x_amazon = df_amzn['Close'].pct_change()[1:]['AMZN'] * 100

df_cost = yf.download('COST', '2000-1-1', '2024-1-1', interval='1mo')
x_costco = df_cost['Close'].pct_change()[1:]['COST'] * 100

np.corrcoef(x_amazon, x_costco)[0, 1]
```

The correlation is modest and positive: the two stocks tend to move together, but far from perfectly.

```{code-cell} ipython3
fitted, (μ_x, μ_y, σ_x, σ_y, ρ) = fit_bivariate_normal(x_amazon.values, x_costco.values)

x_grid = np.linspace(x_amazon.min(), x_amazon.max(), 100)
y_grid = np.linspace(x_costco.min(), x_costco.max(), 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

fig, ax = plt.subplots()
ax.scatter(x_amazon, x_costco, alpha=0.4, s=10)
ax.contour(X_mesh, Y_mesh, fitted.pdf(pos), levels=6, cmap='viridis')
ax.set_xlabel('Amazon monthly return (%)')
ax.set_ylabel('Costco monthly return (%)')
plt.show()
```

The fit here is visibly worse than it was for house prices.

The scatter has several points well outside the outermost contour, particularly the extreme returns during 2000--2001 and 2008--2009.

This matches what we already knew from {doc}`fitting_distributions`: monthly stock returns have heavier tails than the normal distribution allows for, and that failure carries over directly to the bivariate case.

```{solution-end}
```
