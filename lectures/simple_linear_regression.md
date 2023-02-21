---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Simple Linear Regression Model

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

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
\min_{\alpha,\beta} \sum_{i=1}^{N}{\hat{e}_i^2} = \min_{\alpha,\beta} \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)^2}
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
0 = \sum_{i=1}^{N}{-2(y_i - \alpha - \beta x_i)}
$$

we can remove the constant $-2$ from the summation and devide both sides by $-2$


$$
0 = \sum_{i=1}^{N}{(y_i - \alpha - \beta x_i)}
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
$$ (eq:optimal-alpha)

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

and solving for $\beta$

$$
\beta = \frac{\sum_{i=1}^{N}(x_i y_i - \bar{y_i} x_i)}{\sum_{i=1}^{N}(x_i^2 - \bar{x_i} x_i)}
$$ (eq:optimal-beta)

## How does error change with respect to $\alpha$ and $\beta$

Let us first look at how the total error changes with respect to $\beta$ (holding the intercept $\alpha$ constant)

We know from [the next section](slr:optimal-values) the optimal values for $\alpha$ and $\beta$  are:

```{code-cell} ipython3
β_optimal = 64.38
α_optimal = -14.72
```

We can then calculate the error for a range of $\beta$ values

```{code-cell} ipython3
errors = {}
for β in np.arange(20,100,0.5):
    errors[β] = abs((α_optimal + β * df['X']) - df['Y']).sum()
```

Ploting the error

```{code-cell} ipython3
ax = pd.Series(errors).plot(xlabel='β', ylabel='error')
plt.axvline(β_optimal, color='r')
```

Now let us vary $\alpha$ (holding $\beta$ constant)

```{code-cell} ipython3
errors = {}
for α in np.arange(-500,500,5):
    errors[α] = abs((α + β_optimal * df['X']) - df['Y']).sum()
```

Ploting the error

```{code-cell} ipython3
ax = pd.Series(errors).plot(xlabel='α', ylabel='error')
plt.axvline(α_optimal, color='r')
```

(slr:optimal-values)=
## Calculating Optimal Values

We can use calculus to compute the optimal values for $\alpha$ and $\beta$ to solve the ordinary least squares solution.

We can now use {eq}`eq:optimal-alpha` and {eq}`eq:optimal-beta` to calculate the optimal values for $\alpha$ and $\beta$

Calculating $\beta$

```{code-cell} ipython3
df = df[['X','Y']].copy()  # Original Data

# Calcuate the sample means
x_bar = df['X'].mean()
y_bar = df['Y'].mean()
```

Now computing across the 10 observations and then summing the numerator and denominator

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

:::{admonition} Why use OLS?
TODO

1. Discuss mathematical properties for why we have chosen OLS
:::


:::{exercise}
:label: slr-ex1

Now that you know the equations to solve the simple linear regression model using OLS
you can now run your own regressions to build a model between $y$ and $x$.

Consider two economic variables GDP per capita and Life Expectancy.

1. What do you think their relationship would be?
2. Gather some data [from our world in data](https://ourworldindata.org)
3. Use `pandas` to import the `csv` formatted data and plot a few different countries of interest
4. Use {eq}`eq:optimal-alpha` and {eq}`eq:optimal-beta` to compute optimal values for  $\alpha$ and $\beta$
5. Plot the line of best fit found using OLS
6. Interpret the coefficients and write a summary sentence of the relationship between GDP per capita and Life Expectancy

:::

:::{solution-start} slr-ex1
:::

**Q2:** Gather some data [from our world in data](https://ourworldindata.org)

:::{raw} html
<iframe src="https://ourworldindata.org/grapher/life-expectancy-vs-gdp-per-capita" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
:::

You can download {download}`a copy of the data here <_static/lecture_specific/simple_linear_regression/life-expectancy-vs-gdp-per-capita.csv>` if you get stuck

**Q3:** Use `pandas` to import the `csv` formatted data and plot a few different countries of interest

```{code-cell} ipython3
fl = "_static/lecture_specific/simple_linear_regression/life-expectancy-vs-gdp-per-capita.csv"  # TODO: Replace with GitHub link
df = pd.read_csv(fl, nrows=10)
```

```{code-cell} ipython3
df
```

You can see that the data downloaded from Our World in Data has provided a global set of countries with the GDP per capita and Life Expectancy Data.

It is often a good idea to at first import a few lines of data from a csv to understand its structure so that you can then choose the columns that you want to read into your program.

There are a bunch of columns we won't need to import such as `Continent`

So let's built a list of the columns we want to import

```{code-cell} ipython3
cols = ['Code', 'Year', 'Life expectancy at birth (historical)', 'GDP per capita']
df = pd.read_csv(fl, usecols=cols)
df
```

Sometimes it can be useful to rename your columns to make it easier to work with in the DataFrame

```{code-cell} ipython3
df.columns = ["cntry", "year", "life_expectency", "gdppc"]
df
```

We can see there are `NaN` values or missing data so let us go ahead and drop those

```{code-cell} ipython3
df.dropna(inplace=True)
```

```{code-cell} ipython3
df
```

We have now droped the number of rows in our DataFrame from 62156 to 12445 removing a lot of empty data relationships.

Now we have a dataset containing life expectency and GDP per capita for a range of years.

It is always a good idea to spend a bit of time understanding what data you actually have. 

For example, you may want to explore this data using `.reshape` to see if data is consistently reported for all countries across years

Let's first look at the Life Expectency Data

```{code-cell} ipython3
le_years = df[['cntry', 'year', 'life_expectency']].set_index(['cntry', 'year']).unstack()['life_expectency']
le_years
```

As you can see there are a lot of countries where data is not available for the Year 1543!

Which country does report this data

```{code-cell} ipython3
le_years[~le_years[1543].isna()]
```

You can see that Great Britain (GBR) is the only one available

You can also take a closer look at the time series to find that it is also non-continuous, even for GBR.

```{code-cell} ipython3
le_years.loc['GBR'].plot()
```

In fact we can use pandas to quickly check how many countries are captured in each year

So it is clear that if you are doing cross-sectional comparisons then more recent data will include a wider set of countries

```{code-cell} ipython3
le_years.stack().unstack(level=0).count(axis=1).plot(xlabel="Year", ylabel="Number of countries");
```

Now let us consider the most recent year in the dataset 2018

```{code-cell} ipython3
df = df[df.year == 2018].reset_index(drop=True).copy()
```

```{code-cell} ipython3
df.plot(x='gdppc', y='life_expectency', kind='scatter',  xlabel="GDP per capita", ylabel="Life Expectency (Years)",);
```

This data shows a couple of interesting relationships.

1. there are a number of countries with similar GDP per capita levels but a wide range in Life Expectency
2. appears to be a positive relationship between GDP per capita and life expectency. Countries with higher GDP per capita tend to have higher life expectency outcomes

Even though OLS is solving linear equations -- one option is to transform the variables, such as through a log transform, and then use OLS to estimate the relationships

:::{tip}
ln -> ln == elasticities
:::

By specifying `logx` you can plot the GDP per Capita data on a log scale

```{code-cell} ipython3
df.plot(x='gdppc', y='life_expectency', kind='scatter',  xlabel="GDP per capita", ylabel="Life Expectency (Years)", logx=True);
```

As you can see from this transformation -- a linear model fits the shape of the data more closely. 

```{code-cell} ipython3
df['log_gdppc'] = df['gdppc'].apply(np.log10)
```

```{code-cell} ipython3
df
```

**Q4:** Use {eq}`eq:optimal-alpha` and {eq}`eq:optimal-beta` to compute optimal values for  $\alpha$ and $\beta$

```{code-cell} ipython3
data = df[['log_gdppc', 'life_expectency']].copy()  # Get Data from DataFrame

# Calcuate the sample means
x_bar = data['log_gdppc'].mean()
y_bar = data['life_expectency'].mean()
```

```{code-cell} ipython3
data
```

```{code-cell} ipython3
# Compute the Sums
data['num'] = data['log_gdppc'] * data['life_expectency'] - y_bar * data['log_gdppc']
data['den'] = pow(data['log_gdppc'],2) - x_bar * data['log_gdppc']
β = data['num'].sum() / data['den'].sum()
print(β)
```

```{code-cell} ipython3
α = y_bar - β * x_bar
print(α)
```

**Q5:** Plot the line of best fit found using OLS

```{code-cell} ipython3
data['life_expectency_hat'] = α + β * df['log_gdppc']
data['error'] = data['life_expectency_hat'] - data['life_expectency']

fig, ax = plt.subplots()
data.plot(x='log_gdppc',y='life_expectency', kind='scatter', ax=ax)
data.plot(x='log_gdppc',y='life_expectency_hat', kind='line', ax=ax, color='g')
plt.vlines(data['log_gdppc'], data['life_expectency_hat'], data['life_expectency'], color='r')
```

:::{solution-end}
:::

:::{exercise}
:label: slr-ex2

Minimising the sum of squares is not the **only** way to generate the line of best fit. 

For example, we could also consider minimising the sum of the **absolute values**, that would give less weight to outliers. 

Solve for $\alpha$ and $\beta$ using the least absolute values
:::
