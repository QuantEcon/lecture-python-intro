# Measuring Inequality


## Overview

Readers will have some intuitive understanding of the term ``inequality''.

Many economic policies, from taxation to the welfare state, are clearly
aimed at addressing inequality.

However, debate on inequality is often tied to political beliefs.

This is dangerous for us because allowing political beliefs to
shape our findings reduces objectivity.

To bring a scientific perspective to the topic of inequality we must start
with careful definitions.

In this lecture we discuss measures of inequality used in economic research.

We will also look at inequality in 


## Inequality in Distributions

When we consider inequality, we are looking at distributions where some people
have more than others.

Below we look at some examples from both simulated and real data.

### An Illustration

For example, suppose we have a population

TODO 

* show two histograms from pre-generated data, interpret as income
distributions of some population
* which of these distributions is more ``unequal''?

* discuss top shares


### Wealth and Income Shares in the US

TODO 

* plot the top income and wealth shares in the US over time

Use the file ./temp/gini_lorenz_us.ipynb but cut unnecessary.

Replicate the relevant figures f:gini_lorenz_us_x in ch_income_wealth.tex of
macro dynamics text --- ask JS.

Explain carefully the pandas steps.


## The Lorenz Curve

Another popular measure of inequality is the Lorenz curve.

The Lorenz curve is more complex than measuring specific shares but also more
informative.

The Lorenz Curve takes a sample $w_1, \ldots, w_n$ and produces a curve $L$.

The curve $L$ is just a function $y = L(x)$ that we can plot and interpret.

The steps are as follow.

We suppose that the sample $w_1, \ldots, w_n$ has been sorted from smallest to largest

Then we generate data points $(x_i, y_i)$ for the Lorenz curve according to

\begin{equation*}
    x_i = \frac{i}{n},
    \quad
    y_i = \frac{\sum_{j \leq i} w_j}{\sum_{j \leq n} w_j},
    \quad i = 1, \ldots, n
\end{equation*}

Now the Lorenz curve $L$ is formed from these data points using linear interpolation.

(If we plot the points with Matplotlib, the interpolation will be done for us.)

The meaning of the curve is:  $y = L(x)$ indicates that the lowest $(100
\times x)$\% of people have $(100 \times y)$\% of all wealth.

Let's look at some examples and try to understand what this means.

In the next figure, we generate 
$n=2000$ draws from the standard lognormal distribution.  

The straight line corresponds to perfect equality.  

The lognormal draws produce a less equal distribution.  

For example, if we imagine these draws as being observations of wealth across
a sample of households, then the dashed lines show that the bottom 80\% of
households own just over 40\% of total wealth.


```{code-cell} ipython3
n = 2000
fig, ax = plt.subplots()
sample = np.exp(np.random.randn(n))

f_vals, l_vals = qe.lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
    
ax.plot(f_vals, f_vals, label='equality', lw=2)
ax.legend(fontsize=12)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')

ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)

ax.set_ylim((0, 1))
ax.set_xlim((0, 1))

plt.show()
```



## The Gini Coefficient

The Lorenz curve is a useful visual representation of inequality in a
distribution.

Another popular measure of income and wealth inequality is the Gini coefficient.

The Gini coefficient is just a number, rather than a curve.

As before, suppose that the sample $w_1, \ldots, w_n$ has been sorted from smallest to largest

The Gini coefficient is defined for the sample above as 

\begin{equation}
    \label{eq:gini}
    G :=
    \frac
        {\sum_{i=1}^n \sum_{j = 1}^n |w_j - w_i|}
        {2n\sum_{i=1}^n w_i}.
\end{equation}


The Gini coefficient is closely related to the Lorenz curve.

In fact, it can be shown that its value is twice the area between the line of
equality and the Lorenz curve (e.g., the shaded area in Figure~????).

The idea is that $G=0$ indicates complete equality, while $G=1$ indicates complete inequality.


## Dynamics of the Gini Coefficient


TODO 

Plot the top income and wealth shares in the US over time

Discuss with Shu

Use the file ./temp/gini_lorenz_us.ipynb but cut unnecessary, use quantecon
functions to compute gini coefficient

Replicate the relevant figures f:gini_lorenz_us_x in ch_income_wealth.tex of
macro dynamics text --- ask JS.

Carefully discuss the pandas code.


## Exercises

NOTE fix up ex 1 and add more exercises -- perhaps empirical

Ex 1

Using simulation, compute the Lorenz curves and Gini coefficients for the
collection of lognormal distributions associated with the random variables
$w_\sigma = \exp(\mu + \sigma Z)$, where $Z \sim N(0, 1)$ and $\sigma$ varies
over a finite grid between $0.2$ and $4$.  

As $\sigma$ increases, so does the variance of $w_\sigma$.  

To focus on volatility, adjust $\mu$ at each step
    to maintain the equality $\mu=-\sigma^2/2$.

(Confirm: this implies that the mean of $w_\sigma$ does not change with $\sigma$.) 

For each $\sigma$, generate 2,000 independent draws of $w_\sigma$ and
calculate the Lorenz curve and Gini coefficient.  

Confirm that higher variance
generates more dispersion in the sample, and hence greater inequality.


Solution to Ex 1

```{code-cell} ipython3
k = 5
σ_vals = np.linspace(0.5, 2, k)

ginis = []

fig, ax = plt.subplots()

ax.plot(f_vals, f_vals, label='equality')

for σ in σ_vals:
    μ = -σ**2/2
    y = np.exp(μ + σ * np.random.randn(n))
    f_vals, l_vals = qe.lorenz_curve(y)
    ginis.append(qe.gini_coefficient(y))
    ax.plot(f_vals, l_vals, label=f'$\sigma = {σ:.1f}$')
    
ax.legend(fontsize=12)

plt.show()
```
