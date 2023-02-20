---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Ordinary Least Squares and Regression

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## The Simple Regression Model

The simple regression model estimates the relationship between two variables $X$ and $Y$

$$
y_i = \alpha + \beta x_i + \epsilon_i, i = 1,2,...,N
$$

where $\epsilon_i$ represents the error in the estimates. 

We would like to choose values for $\alpha$ and $\beta$ to build a line of "best" fit for some data that is available for variables $x_i$ and $y_i$. 

Let us consider a simple dataset of 10 observations for variables $x_i$ and $y_i$:

| | $y_i$  | $x_i$ |
|-|---|---|
|1| 2000 | 32 |
|2| 1000 | 21 | 
|3| 1500 | 24 | 
|4| 2500 | 35 | 
|5| 500 | 10 |
|6| 900 | 11 |
|7| 1100 | 22 | 
|8| 1500 | 21 | 
|9| 1800 | 27 |
|10 | 250 | 2 |

Let us think about $y_i$ as sales for an ice-cream cart, while $x_i$ is a variable the records the temperature in Celcius. 

```{code-cell} ipython3
x = [32, 21, 24, 35, 10, 11, 22, 21, 27, 2]
y = [2000,1000,1500,2500,500,900,1100,1500,1800, 250]
df = pd.DataFrame([x,y]).T
df.columns = ['X', 'Y']
df
```

We can use a scatter plot of the data to see the relationship between $y_i$ (ice-cream sales in dollars (\$\'s)) and $x_i$ (degrees celcius).

```{code-cell} ipython3
ax = df.plot(
    x='X', 
    y='Y', 
    kind='scatter', 
    ylabel='Ice-Cream Sales ($\'s)', 
    xlabel='Degrees Celcius'
)
```

as you can see the data suggests that more ice-cream is typically sold on hotter days. 

---

To build a linear model of the data we need to choose values for $\alpha$ and $\beta$ that represents a line of "best" fit such that

$$
\hat{y}_i = \hat{\alpha} + \hat{\beta} x_i
$$

Let's start with $\alpha = 5$ and $\beta = 10$

```{code-cell} ipython3
α = 5
β = 10
df['Y_hat'] = α + β * df['X']
```

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X',y='Y', kind='scatter', ax=ax)
df.plot(x='X',y='Y_hat', kind='line', ax=ax)
```

We can continue to guess and iterate towards a line of "best" fit by adjusting the parameters

```{code-cell} ipython3
β = 100
df['Y_hat'] = α + β * df['X']
```

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X',y='Y', kind='scatter', ax=ax)
df.plot(x='X',y='Y_hat', kind='line', ax=ax)
```

```{code-cell} ipython3
β = 65
df['Y_hat'] = α + β * df['X']
```

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X',y='Y', kind='scatter', ax=ax)
df.plot(x='X',y='Y_hat', kind='line', ax=ax, color='g')
```

We need to think about formalising this process by thinking of this problem as an optimization problem. 

Let's consider the error $\epsilon_i$ and define the difference between the observed values $y_i$ and the estimated values $\hat{y}_i$ which we will call the residuals

$$
\begin{aligned}
\hat{e}_i &= y_i - \hat{y}_i \\
          &= y_i - \hat{\alpha} - \hat{\beta} x_i
\end{aligned}
$$

```{code-cell} ipython3
df['error'] = df['Y_hat'] - df['Y']
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
fig, ax = plt.subplots()
df.plot(x='X',y='Y', kind='scatter', ax=ax)
df.plot(x='X',y='Y_hat', kind='line', ax=ax, color='g')
plt.vlines(df['X'], df['Y_hat'], df['Y'], color='r')
```

The Ordinary Least Squares (OLS) method, as the name suggests, chooses $\alpha$ and $\beta$ in such a way that **minimises** the Sum of the Squared Residuals (SSR). 

$$
\min \sum_{i=1}^{N}{\hat{e}_i^2} = \min \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}
$$

Let's call this a cost function

$$
C = \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}
$$

that we would like to minimise.

We can use calculus to find a solution by taking the partial derivative of the cost function $C$ with respect to $\alpha$ and $\beta$

First taking the partial derivative with respect to $\alpha$

$$
\frac{\partial C}{\partial \alpha}[\sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}]
$$

and setting it equal to $0$

$$
0 = \sum_{i=1}^{N}{-2(y_i - \alpha - \beta * x_i)}
$$

we can remove the constant $-2$ from the summation and devide both sides by $-2$


$$
0 = \sum_{i=1}^{N}{(y_i - \alpha - \beta * x_i)}
$$

Now we can split this equation up into the components

$$
0 = \sum_{i=1}^{N}{y_i} - \sum_{i=1}^{N}{\alpha} - \beta \sum_{i=1}^{N}{x_i}
$$

The middle term is simple to sum from $i=1,...N$ by a constant $\alpha$

$$
0 = \sum_{i=1}^{N}{y_i} - N*\alpha - \beta \sum_{i=1}^{N}{x_i}
$$

and rearranging terms 

$$
\alpha = \frac{\sum_{i=1}^{N}{y_i} - \beta \sum_{i=1}^{N}{x_i}}{N}
$$

Both fractions resolve to the means $\bar{y_i}$ and $\bar{x_i}$ 

$$
\alpha = \bar{y_i} - \beta\bar{x_i}
$$

Now let's take the partial derivative of the cost function $C$ with respect to $\beta$

$$
\frac{\partial C}{\partial \beta}[\sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}]
$$

and setting it equal to $0$

$$
0 = \sum_{i=1}^{N}{-2 x_i (y_i - \alpha - \beta x_i)}
$$

we can again take the constant outside of the summation and divide both sides by $-2$

$$
0 = \sum_{i=1}^{N}{x_i (y_i - \alpha - \beta x_i)}
$$

which becomes

$$
0 = \sum_{i=1}^{N}{(x_i y_i - \alpha x_i - \beta x_i^2)}
$$

now substituting $\alpha$

$$
0 = \sum_{i=1}^{N}{(x_i y_i - (\bar{y_i} - \beta \bar{x_i}) x_i - \beta x_i^2)}
$$

and rearranging terms

$$
0 = \sum_{i=1}^{N}{(x_i y_i - \bar{y_i} x_i - \beta \bar{x_i} x_i - \beta x_i^2)}
$$

This can be split into two summations

$$
0 = \sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i) + \beta \sum_{i=1}^{N}(\bar{x_i} x_i - x_i^2)
$$

and solving for B

$$
\beta = \frac{\sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i)}{\sum_{i=1}^{N}(x_i^2 - \bar{x_i} x_i)}
$$

+++

We can now use these formulas to calculate the optimal values for $\alpha$ and $\beta$

+++

Calculating $\beta$

```{code-cell} ipython3
df = df[['X','Y']].copy() # Original Data

# Calcuate the sample means
x_bar = df['X'].mean()
y_bar = df['Y'].mean()
```

```{code-cell} ipython3
# Compute the Sums
df['num'] = df['X'] * df['Y'] - y_bar * df['X']
df['den'] = pow(df['X'],2) - x_bar * df['X']
β = df['num'].sum() / df['den'].sum()
print(β)
```

Calculating $\alpha$

```{code-cell} ipython3
α = y_bar - β * x_bar
print(α)
```

Now we can plot the OLS solution

```{code-cell} ipython3
df['Y_hat'] = α + β * df['X']
df['error'] = df['Y_hat'] - df['Y']

fig, ax = plt.subplots()
df.plot(x='X',y='Y', kind='scatter', ax=ax)
df.plot(x='X',y='Y_hat', kind='line', ax=ax, color='g')
plt.vlines(df['X'], df['Y_hat'], df['Y'], color='r')
```

How does error change with respect to $\alpha$ and $\beta$

Let us first look at how the total error changes with respect to $\beta$ (holding the intercept $\alpha$ constant)

```{code-cell} ipython3
# Save the optimal values
β_optimal = β 
α_optimal = α
```

```{code-cell} ipython3
errors = {}
for β in np.arange(20,100,0.5):
    errors[β] = abs((α_optimal + β * df['X']) - df['Y']).sum()
```

```{code-cell} ipython3
ax = pd.Series(errors).plot(xlabel='β', ylabel='error')
plt.axvline(β_optimal, color='r')
```

Now we can hold $\beta$ constant and vary $\alpha$

```{code-cell} ipython3
errors = {}
for α in np.arange(-500,500,5):
    errors[α] = abs((α + β_optimal * df['X']) - df['Y']).sum()
```

```{code-cell} ipython3
ax = pd.Series(errors).plot(xlabel='α', ylabel='error')
plt.axvline(α_optimal, color='r')
```

**Note:** I don't think this surface plot adds a whole lot

```{code-cell} ipython3
from matplotlib import cm
```

```{code-cell} ipython3
alphas = []
betas = []
errors = []
for α in range(-20,20,1):
    for β in range(20,100,1):
        alphas.append(α)
        betas.append(β)
        errors.append(abs((α + β * df['X']) - df['Y']).sum())
```

```{code-cell} ipython3
# Plot the Error Surface with respect to α and β
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_trisurf(alphas, betas, errors, cmap=cm.coolwarm, linewidth=0.1)
```

As you can see we were able to use calculus to compute the optimal values for $\alpha$ and $\beta$ to solve the ordinary least squares solution.

:::{admonition} Why use OLS?
TODO

1. Discuss mathematical properties for why we have chosen OLS
:::

+++

:::{admonition} Exercise

Minimising the sum of squares is not the **only** way to generate the line of best fit. 

For example, we could also consider minimising the sum of the **absolute values**, that would give less weight to outliers. 

Solve for $\alpha$ and $\beta$ using the least absolute values

:::