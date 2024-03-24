---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Income and Wealth Inequality

## Overview

In this section we 

* provide motivation for the techniques deployed in the lecture and
* import code libraries needed for our work.

### Some history

Many historians argue that inequality played a key role in the fall of the
Roman Republic.

After defeating Carthage and invading Spain, money flowed into Rome and
greatly enriched those in power.

Meanwhile, ordinary citizens were taken from their farms to fight for long
periods, diminishing their wealth.

The resulting growth in inequality caused political turmoil that shook the
foundations of the republic. 

Eventually, the Roman Republic gave way to a series of dictatorships, starting
with Octavian (Augustus) in 27 BCE.

This history is fascinating in its own right, and we can see some
parallels with certain countries in the modern world.

Many recent political debates revolve around inequality.

Many economic policies, from taxation to the welfare state, are 
aimed at addressing inequality.

### Measurement

One problem with these debates is that inequality is often poorly defined.

Moreover, debates on inequality are often tied to political beliefs.

This is dangerous for economists because allowing political beliefs to
shape our findings reduces objectivity.

To bring a truly scientific perspective to the topic of inequality we must
start with careful definitions.

In this lecture we discuss standard measures of inequality used in economic research.

For each of these measures, we will look at both simulated and real data.

We will need to install the following packages

```{code-cell} ipython3
:tags: [hide-output]

!pip install wbgapi plotly
```

We will also use the following imports.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import wbgapi as wb
import plotly.express as px
```

## The Lorenz curve

One popular measure of inequality is the Lorenz curve.

In this section we define the Lorenz curve and examine its properties.


### Definition

The Lorenz curve takes a sample $w_1, \ldots, w_n$ and produces a curve $L$.

We suppose that the sample $w_1, \ldots, w_n$ has been sorted from smallest to largest.

To aid our interpretation, suppose that we are measuring wealth 

*  $w_1$ is the wealth of the poorest member of the population, and
*  $w_n$ is the wealth of the richest member of the population.

The curve $L$ is just a function $y = L(x)$ that we can plot and interpret.

To create it we first generate data points $(x_i, y_i)$  according to

```{prf:definition}
:label: define-lorenz

$$
x_i = \frac{i}{n},
\qquad
y_i = \frac{\sum_{j \leq i} w_j}{\sum_{j \leq n} w_j},
\qquad i = 1, \ldots, n
$$
```

Now the Lorenz curve $L$ is formed from these data points using interpolation.

If we use a line plot in `matplotlib`, the interpolation will be done for us.

The meaning of the statement $y = L(x)$ is that the lowest $(100
\times x)$\% of people have $(100 \times y)$\% of all wealth.

* if $x=0.5$ and $y=0.1$, then the bottom 50% of the population
  owns 10% of the wealth.

In the discussion above we focused on wealth but the same ideas apply to
income, consumption, etc.


### Lorenz curves of simulated data

Let's look at some examples and try to build understanding.

First let us construct a `lorenz_curve` function that we can
use in our simulations below.

It is useful to construct a function that translates an array of
income or wealth data into the cumulative share
of individuals (or households) and the cumulative share of income (or wealth).

```{code-cell} ipython3
def lorenz_curve(y):
    """
    Calculates the Lorenz Curve, a graphical representation of
    the distribution of income or wealth.

    It returns the cumulative share of people (x-axis) and
    the cumulative share of income earned.

    Parameters
    ----------
    y : array_like(float or int, ndim=1)
        Array of income/wealth for each individual.
        Unordered or ordered is fine.

    Returns
    -------
    cum_people : array_like(float, ndim=1)
        Cumulative share of people for each person index (i/n)
    cum_income : array_like(float, ndim=1)
        Cumulative share of income for each person index


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lorenz_curve

    Examples
    --------
    >>> a_val, n = 3, 10_000
    >>> y = np.random.pareto(a_val, size=n)
    >>> f_vals, l_vals = lorenz(y)

    """

    n = len(y)
    y = np.sort(y)
    s = np.zeros(n + 1)
    s[1:] = np.cumsum(y)
    cum_people = np.zeros(n + 1)
    cum_income = np.zeros(n + 1)
    for i in range(1, n + 1):
        cum_people[i] = i / n
        cum_income[i] = s[i] / s[n]
    return cum_people, cum_income
```

In the next figure, we generate $n=2000$ draws from a lognormal
distribution and treat these draws as our population.  

The straight 45-degree line ($x=L(x)$ for all $x$) corresponds to perfect equality.  

The log-normal draws produce a less equal distribution.  

For example, if we imagine these draws as being observations of wealth across
a sample of households, then the dashed lines show that the bottom 80\% of
households own just over 40\% of total wealth.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Lorenz curve of simulated data
    name: lorenz_simulated
---
n = 2000
sample = np.exp(np.random.randn(n))

fig, ax = plt.subplots()

f_vals, l_vals = lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')
ax.set_xlim((0, 1))
ax.set_xlabel("share of households (%)")
ax.set_ylim((0, 1))
ax.set_ylabel("share of income (%)")
ax.legend()
plt.show()
```

### Lorenz curves for US data

Next let's look at data, focusing on income and wealth in the US in 2016.

(data:survey-consumer-finance)=
The following code block imports a subset of the dataset `SCF_plus`,
which is derived from the [Survey of Consumer Finances](https://en.wikipedia.org/wiki/Survey_of_Consumer_Finances) (SCF).

```{code-cell} ipython3
url = 'https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/SCF_plus/SCF_plus_mini.csv'
df = pd.read_csv(url)
df_income_wealth = df.dropna()
```

```{code-cell} ipython3
df_income_wealth.head(n=5)
```

The following code block uses data stored in dataframe `df_income_wealth` to generate the Lorenz curves.

(The code is somewhat complex because we need to adjust the data according to
population weights supplied by the SCF.)

```{code-cell} ipython3
:tags: [hide-input]

df = df_income_wealth 

varlist = ['n_wealth',    # net wealth 
           't_income',    # total income
           'l_income']    # labor income

years = df.year.unique()

# Create lists to store Lorenz data

F_vals, L_vals = [], []

for var in varlist:
    # create lists to store Lorenz curve data
    f_vals = []
    l_vals = []
    for year in years:

        # Repeat the observations according to their weights
        counts = list(round(df[df['year'] == year]['weights'] )) 
        y = df[df['year'] == year][var].repeat(counts)
        y = np.asarray(y)
        
        # Shuffle the sequence to improve the plot
        rd.shuffle(y)    
               
        # calculate and store Lorenz curve data
        f_val, l_val = lorenz_curve(y)
        f_vals.append(f_val)
        l_vals.append(l_val)
        
    F_vals.append(f_vals)
    L_vals.append(l_vals)

f_vals_nw, f_vals_ti, f_vals_li = F_vals
l_vals_nw, l_vals_ti, l_vals_li = L_vals
```

Now we plot Lorenz curves for net wealth, total income and labor income in the
US in 2016.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 2016 US Lorenz curves
    name: lorenz_us
  image:
    alt: lorenz_us
---
fig, ax = plt.subplots()
ax.plot(f_vals_nw[-1], l_vals_nw[-1], label=f'net wealth')
ax.plot(f_vals_ti[-1], l_vals_ti[-1], label=f'total income')
ax.plot(f_vals_li[-1], l_vals_li[-1], label=f'labor income')
ax.plot(f_vals_nw[-1], f_vals_nw[-1], label=f'equality')
ax.set_xlabel("share of households (%)")
ax.set_ylabel("share of income/wealth (%)")
ax.legend()
plt.show()
```

Here all the income and wealth measures are pre-tax.

Total income is the sum of households' all income sources, including labor income but excluding capital gains.

One key finding from this figure is that wealth inequality is significantly
more extreme than income inequality. 

We will take a look at this trend over time {ref}`in a later section<compare-income-wealth-usa-over-time>`. 

## The Gini coefficient

The Lorenz curve is a useful visual representation of inequality in a distribution.

Another popular measure of income and wealth inequality is the Gini coefficient.

The Gini coefficient is just a number, rather than a curve.

In this section we discuss the Gini coefficient and its relationship to the
Lorenz curve.


### Definition

As before, suppose that the sample $w_1, \ldots, w_n$ has been sorted from
smallest to largest.

The Gini coefficient is defined for the sample above as 

```{prf:definition}
:label: define-gini

$$
G :=
\frac{\sum_{i=1}^n \sum_{j = 1}^n |w_j - w_i|}
     {2n\sum_{i=1}^n w_i}.
$$
```

The Gini coefficient is closely related to the Lorenz curve.

In fact, it can be shown that its value is twice the area between the line of
equality and the Lorenz curve (e.g., the shaded area in {numref}`lorenz_gini`).

The idea is that $G=0$ indicates complete equality, while $G=1$ indicates complete inequality.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Shaded Lorenz curve of simulated data
    name: lorenz_gini
---
fig, ax = plt.subplots()
f_vals, l_vals = lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)
ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')
ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)
ax.set_ylim((0, 1))
ax.set_xlim((0, 1))
ax.text(0.04, 0.5, r'$G = 2 \times$ shaded area')
ax.set_xlabel("share of households (%)")
ax.set_ylabel("share of income/wealth (%)")
ax.legend()
plt.show()
```

Another way to think of the Gini coefficient is as a ratio of the area between the 45-degree line of 
perfect equality and the Lorenz curve (A) divided by the total area below the 45-degree line (A+B) as shown in {numref}`lorenz_gini2`. 

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Lorenz curve and Gini coefficient
    name: lorenz_gini2
---
fig, ax = plt.subplots()
f_vals, l_vals = lorenz_curve(sample)
ax.plot(f_vals, l_vals, label='lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)
ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)
ax.fill_between(f_vals, l_vals, np.zeros_like(f_vals), alpha=0.06)
ax.set_ylim((0, 1))
ax.set_xlim((0, 1))
ax.text(0.55, 0.4, 'A')
ax.text(0.75, 0.15, 'B')
ax.set_xlabel("share of households (%)")
ax.set_ylabel("share of income/wealth (%)")
ax.legend()
plt.show()
```

$$
G = \frac{A}{A+B}
$$

It is an average measure of deviation from the line of equality.

```{seealso}
The World in Data project has a [nice graphical exploration of the Lorenz curve and the Gini coefficient](https://ourworldindata.org/what-is-the-gini-coefficient)
```

### Gini coefficient of simulated data

Let's examine the Gini coefficient in some simulations.

First the code below enables us to compute the Gini coefficient.

```{code-cell} ipython3

def gini_coefficient(y):
    r"""
    Implements the Gini inequality index

    Parameters
    ----------
    y : array_like(float)
        Array of income/wealth for each individual.
        Ordered or unordered is fine

    Returns
    -------
    Gini index: float
        The gini index describing the inequality of the array of income/wealth

    References
    ----------

    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    n = len(y)
    i_sum = np.zeros(n)
    for i in range(n):
        for j in range(n):
            i_sum[i] += abs(y[i] - y[j])
    return np.sum(i_sum) / (2 * n * np.sum(y))
```

Now we can compute the Gini coefficients for five different populations.

Each of these populations is generated by drawing from a 
lognormal distribution with parameters $\mu$ (mean) and $\sigma$ (standard deviation).

To create the five populations, we vary $\sigma$ over a grid of length $5$
between $0.2$ and $4$.

In each case we set $\mu = - \sigma^2 / 2$.

This implies that the mean of the distribution does not change with $\sigma$. 

You can check this by looking up the expression for the mean of a lognormal
distribution.

```{code-cell} ipython3
k = 5
σ_vals = np.linspace(0.2, 4, k)
n = 2_000

ginis = []

for σ in σ_vals:
    μ = -σ**2 / 2
    y = np.exp(μ + σ * np.random.randn(n))
    ginis.append(gini_coefficient(y))
```

Let's build a function that returns a figure (so that we can use it later in the lecture).

```{code-cell} ipython3
def plot_inequality_measures(x, y, legend, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', label=legend)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients of simulated data
    name: gini_simulated
---
fix, ax = plot_inequality_measures(σ_vals, 
                                  ginis, 
                                  'simulated', 
                                  '$\sigma$', 
                                  'Gini coefficients')
plt.show()
```

The plots show that inequality rises with $\sigma$, according to the Gini
coefficient.

### Gini coefficient for income (US data)

Let's look at the Gini coefficient for the distribution of income in the US.

We will get pre-computed Gini coefficients (based on income) from the World Bank using the [wbgapi](https://blogs.worldbank.org/opendata/introducing-wbgapi-new-python-package-accessing-world-bank-data).

Let's use the `wbgapi` package we imported earlier to search the world bank data for Gini to find the Series ID.

```{code-cell} ipython3
wb.search("gini")
```

We now know the series ID is `SI.POV.GINI`.

Another, and often useful way to find series ID, is to use the [World Bank data portal](https://data.worldbank.org) and then use `wbgapi` to fetch the data.

Using `pandas` we can take a quick look across all countries and all years in the World Bank dataset. 

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Histogram of Gini coefficients
    name: gini_histogram
---
# Fetch gini data for all countries
gini_all = wb.data.DataFrame("SI.POV.GINI")
# remove 'YR' in index and convert to integer
gini_all.columns = gini_all.columns.map(lambda x: int(x.replace('YR',''))) 

# Create a long series with a multi-index of the data to get global min and max values
gini_all = gini_all.unstack(level='economy').dropna()

# Build a histogram
ax = gini_all.plot(kind="hist", bins=20)
ax.set_xlabel("Gini coefficient")
ax.set_ylabel("frequency")
plt.show()
```

We can see in {numref}`gini_histogram` that across 50 years of data and all countries
the measure only varies between 20 and 65.

Let us fetch the data `DataFrame` for the USA. 

```{code-cell} ipython3
data = wb.data.DataFrame("SI.POV.GINI", "USA")
data.head(n=5)
# remove 'YR' in index and convert to integer
data.columns = data.columns.map(lambda x: int(x.replace('YR','')))
```

**Note:** This package often returns data with year information contained in the columns. This is not always convenient for simple plotting with pandas so it can be useful to transpose the results before plotting

```{code-cell} ipython3
data = data.T           # Obtain years as rows
data_usa = data['USA']  # pd.Series of US data
```

Let us zoom on the US data so we can more clearly observe trends.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients for income distribution (USA)
    name: gini_usa1
---
fig, ax = plt.subplots()
ax = data_usa.plot(ax=ax)
ax.set_ylim(data_usa.min()-1, data_usa.max()+1)
ax.set_ylabel("Gini coefficient (income)")
ax.set_xlabel("year")
plt.show()
```

As can be seen in {numref}`gini_usa1` the Gini coefficient:

1. trended upward from 1980 to 2020 and then dropped slightly following at the start of the COVID pandemic
2. moves slowly over time

(compare-income-wealth-usa-over-time)=
### Gini coefficient for wealth (US data)

In the previous section we looked at the Gini coefficient for income using US data.

Now let's look at the Gini coefficient for the distribution of wealth.

We can use the data collected above {ref}`survey of consumer finances <data:survey-consumer-finance>` to look at the Gini coefficient
computed over the wealth distribution.

The Gini coefficient for net wealth and labour income is computed over many years.

```{code-cell} ipython3
df_income_wealth.year.describe()
```

**Note:** This code can be used to compute this information over the full dataset.

```{code-cell} ipython3
:tags: [skip-execution, hide-input, hide-output]

!pip install quantecon
import quantecon as qe

varlist = ['n_wealth',   # net wealth 
           't_income',   # total income
           'l_income']   # labor income

df = df_income_wealth

# create lists to store Gini for each inequality measure
results = {}

for var in varlist:
    # create lists to store Gini
    gini_yr = []
    for year in years:
        # repeat the observations according to their weights
        counts = list(round(df[df['year'] == year]['weights'] ))
        y = df[df['year'] == year][var].repeat(counts)
        y = np.asarray(y)
        
        rd.shuffle(y)    # shuffle the sequence
      
        # calculate and store Gini
        gini = qe.gini_coefficient(y)
        gini_yr.append(gini)
        
    results[var] = gini_yr

# Convert to DataFrame
results = pd.DataFrame(results, index=years)
results.to_csv("_static/lecture_specific/inequality/usa-gini-nwealth-tincome-lincome.csv", index_label='year')
```

However, to speed up execution we will import a pre-computed dataset from the lecture repository.

<!-- TODO: update from csv to github location -->

```{code-cell} ipython3
ginis = pd.read_csv("_static/lecture_specific/inequality/usa-gini-nwealth-tincome-lincome.csv", index_col='year')
ginis.head(n=5)
```

Let's plot the Gini coefficients for net wealth.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients of US net wealth
    name: gini_wealth_us
---
fig, ax = plt.subplots()
ax.plot(years, ginis["n_wealth"], marker='o')
ax.set_xlabel("year")
ax.set_ylabel("Gini coefficient")
plt.show()
```

Looking at each data series we see an outlier in Gini coefficient computed for 1965 for `labour income`. 

We will smooth our data and take an average of the data either side of it for the time being.

```{code-cell} ipython3
ginis["l_income"][1965] = (ginis["l_income"][1962] + ginis["l_income"][1968]) / 2
```

Now we can compare net wealth and labour income.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients of US net wealth and labour income
    name: gini_wealth_us
---
fig, ax = plt.subplots()
ax.plot(years, ginis["n_wealth"], marker='o', label="net wealth")
ax.plot(years, ginis["l_income"], marker='o', label="labour income")
ax.set_xlabel("year")
ax.set_ylabel("Gini coefficient")
ax.legend()
plt.show()
```

We see that, by this measure, inequality in both wealth and income has risen
substantially since 1980.

The wealth time series exhibits a strong U-shape.

### Cross-country comparisons of income inequality

As we saw earlier in this lecture we used `wbgapi` to get Gini data across many countries and saved it in a variable called `gini_all`

In this section we will compare a few western economies and look at the evolution in their respective Gini coefficients

```{code-cell} ipython3
data = gini_all.unstack()
data.columns
```

There are 167 countries represented in this dataset. 

Let us compare three western economies: USA, United Kingdom, and Norway

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients for income (USA, United Kingdom, and Norway)
    name: gini_usa_gbr_nor1
---
ax = data[['USA','GBR', 'NOR']].plot()
ax.set_xlabel('year')
ax.set_ylabel('Gini coefficient')
ax.legend(title="")
plt.show()
```

We see that Norway has a shorter time series so let us take a closer look at the underlying data

```{code-cell} ipython3
data[['NOR']].dropna().head(n=5)
```

The data for Norway in this dataset goes back to 1979 but there are gaps in the time series and matplotlib is not showing those data points. 

We can use the `.ffill()` method to copy and bring forward the last known value in a series to fill in these gaps

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients for income (USA, United Kingdom, and Norway)
    name: gini_usa_gbr_nor2
---
data['NOR'] = data['NOR'].ffill()
ax = data[['USA','GBR', 'NOR']].plot()
ax.set_xlabel('year')
ax.set_ylabel('Gini coefficient')
ax.legend(title="")
plt.show()
```

From this plot we can observe that the USA has a higher Gini coefficient (i.e. higher income inequality) when compared to the UK and Norway. 

Norway has the lowest Gini coefficient over the three economies and is substantially lower than the US.

### Gini Coefficient and GDP per capita (over time)

We can also look at how the Gini coefficient compares with GDP per capita (over time). 

Let's take another look at the USA, Norway, and the United Kingdom.

```{code-cell} ipython3
countries = ['USA', 'NOR', 'GBR']
gdppc = wb.data.DataFrame("NY.GDP.PCAP.KD", countries)
# remove 'YR' in index and convert to integer
gdppc.columns = gdppc.columns.map(lambda x: int(x.replace('YR',''))) 
gdppc = gdppc.T
```

We can rearrange the data so that we can plot gdp per capita and the Gini coefficient across years

```{code-cell} ipython3
plot_data = pd.DataFrame(data[countries].unstack())
plot_data.index.names = ['country', 'year']
plot_data.columns = ['gini']
```

Now we can get the gdp per capita data into a shape that can be merged with `plot_data`

```{code-cell} ipython3
pgdppc = pd.DataFrame(gdppc.unstack())
pgdppc.index.names = ['country', 'year']
pgdppc.columns = ['gdppc']
plot_data = plot_data.merge(pgdppc, left_index=True, right_index=True)
plot_data.reset_index(inplace=True)
```

Now using plotly to build a plot with gdp per capita on the y-axis and the Gini coefficient on the x-axis.

```{code-cell} ipython3
min_year = plot_data.year.min()
max_year = plot_data.year.max()
```


**Note:** The time series for all three countries start and stop in different years. We will add a year mask to the data to
improve clarity in the chart including the different end years associated with each countries time series.

```{code-cell} ipython3
labels = [1979, 1986, 1991, 1995, 2000, 2020, 2021, 2022] + \
         list(range(min_year,max_year,5))
plot_data.year = plot_data.year.map(lambda x: x if x in labels else None)
```

(fig:plotly-gini-gdppc-years)=

```{code-cell} ipython3
fig = px.line(plot_data, 
              x = "gini", 
              y = "gdppc", 
              color = "country", 
              text = "year", 
              height = 800,
              labels = {"gini" : "Gini coefficient", "gdppc" : "GDP per capita"}
             )
fig.update_traces(textposition="bottom right")
fig.show()
```

```{only} latex
This figure is built using `plotly` and is {ref}` available on the website <fig:plotly-gini-gdppc-years>`
```

This plot shows that all three western economies GDP per capita has grown over time with some fluctuations
in the Gini coefficient. 

From the early 80's the United Kingdom and the US economies both saw increases in income 
inequality. 

Interestingly, since the year 2000, the United Kingdom saw a decline in income inequality while
the US exhibits persistent but stable levels around a Gini coefficient of 40. 

## Top shares

Another popular measure of inequality is the top shares.

Measuring specific shares is less complex than the Lorenz curve or the Gini
coefficient.

In this section we show how to compute top shares.

### Definition

As before, suppose that the sample $w_1, \ldots, w_n$ has been sorted from smallest to largest.

Given the Lorenz curve $y = L(x)$ defined above, the top $100 \times p \%$
share is defined as

```{prf:definition}
:label: top-shares

$$
T(p) = 1 - L (1-p) 
    \approx \frac{\sum_{j\geq i} w_j}{ \sum_{j \leq n} w_j}, \quad i = \lfloor n (1-p)\rfloor
$$ (topshares)
```

Here $\lfloor \cdot \rfloor$ is the floor function, which rounds any
number down to the integer less than or equal to that number.

The following code uses the data from dataframe `df_income_wealth` to generate another dataframe `df_topshares`.

`df_topshares` stores the top 10 percent shares for the total income, the labor income and net wealth from 1950 to 2016 in US.

```{code-cell} ipython3
:tags: [hide-input]

# transfer the survey weights from absolute into relative values
df1 = df_income_wealth
df2 = df1.groupby('year').sum(numeric_only=True).reset_index()
df3 = df2[['year', 'weights']]
df3.columns = 'year', 'r_weights'
df4 = pd.merge(df3, df1, how="left", on=["year"])
df4['r_weights'] = df4['weights'] / df4['r_weights']

# create weighted nw, ti, li
df4['weighted_n_wealth'] = df4['n_wealth'] * df4['r_weights']
df4['weighted_t_income'] = df4['t_income'] * df4['r_weights']
df4['weighted_l_income'] = df4['l_income'] * df4['r_weights']

# extract two top 10% groups by net wealth and total income.
df6 = df4[df4['nw_groups'] == 'Top 10%']
df7 = df4[df4['ti_groups'] == 'Top 10%']

# calculate the sum of weighted top 10% by net wealth,
#   total income and labor income.
df5 = df4.groupby('year').sum(numeric_only=True).reset_index()
df8 = df6.groupby('year').sum(numeric_only=True).reset_index()
df9 = df7.groupby('year').sum(numeric_only=True).reset_index()

df5['weighted_n_wealth_top10'] = df8['weighted_n_wealth']
df5['weighted_t_income_top10'] = df9['weighted_t_income']
df5['weighted_l_income_top10'] = df9['weighted_l_income']

# calculate the top 10% shares of the three variables.
df5['topshare_n_wealth'] = df5['weighted_n_wealth_top10'] / \
    df5['weighted_n_wealth']
df5['topshare_t_income'] = df5['weighted_t_income_top10'] / \
    df5['weighted_t_income']
df5['topshare_l_income'] = df5['weighted_l_income_top10'] / \
    df5['weighted_l_income']

# we only need these vars for top 10 percent shares
df_topshares = df5[['year', 'topshare_n_wealth',
                    'topshare_t_income', 'topshare_l_income']]
```

Then let's plot the top shares.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: US top shares
    name: top_shares_us
---
fig, ax = plt.subplots()
ax.plot(years, df_topshares["topshare_l_income"],
        marker='o', label="labor income")
ax.plot(years, df_topshares["topshare_n_wealth"],
        marker='o', label="net wealth")
ax.plot(years, df_topshares["topshare_t_income"],
        marker='o', label="total income")
ax.set_xlabel("year")
ax.set_ylabel("top $10\%$ share")
ax.legend()
plt.show()
```

## Exercises

```{exercise}
:label: inequality_ex1

Using simulation, compute the top 10 percent shares for the collection of
lognormal distributions associated with the random variables $w_\sigma =
\exp(\mu + \sigma Z)$, where $Z \sim N(0, 1)$ and $\sigma$ varies over a
finite grid between $0.2$ and $4$.  

As $\sigma$ increases, so does the variance of $w_\sigma$.  

To focus on volatility, adjust $\mu$ at each step to maintain the equality
$\mu=-\sigma^2/2$.

For each $\sigma$, generate 2,000 independent draws of $w_\sigma$ and
calculate the Lorenz curve and Gini coefficient.  

Confirm that higher variance
generates more dispersion in the sample, and hence greater inequality.
```

```{solution-start} inequality_ex1
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
def calculate_top_share(s, p=0.1):
    
    s = np.sort(s)
    n = len(s)
    index = int(n * (1 - p))
    return s[index:].sum() / s.sum()
```

```{code-cell} ipython3
k = 5
σ_vals = np.linspace(0.2, 4, k)
n = 2_000

topshares = []
ginis = []
f_vals = []
l_vals = []

for σ in σ_vals:
    μ = -σ ** 2 / 2
    y = np.exp(μ + σ * np.random.randn(n))
    f_val, l_val = lorenz_curve(y)
    f_vals.append(f_val)
    l_vals.append(l_val)
    ginis.append(gini_coefficient(y))
    topshares.append(calculate_top_share(y))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Top shares of simulated data
    name: top_shares_simulated
  image:
    alt: top_shares_simulated
---
fig, ax = plot_inequality_measures(σ_vals, 
                                  topshares, 
                                  "simulated data", 
                                  "$\sigma$", 
                                  "top $10\%$ share") 
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gini coefficients of simulated data
    name: gini_coef_simulated
  image:
    alt: gini_coef_simulated
---
fig, ax = plot_inequality_measures(σ_vals, 
                                  ginis, 
                                  "simulated data", 
                                  "$\sigma$", 
                                  "gini coefficient")
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Lorenz curves for simulated data
    name: lorenz_curve_simulated
  image:
    alt: lorenz_curve_simulated
---
fig, ax = plt.subplots()
ax.plot([0,1],[0,1], label=f"equality")
for i in range(len(f_vals)):
    ax.plot(f_vals[i], l_vals[i], label=f"$\sigma$ = {σ_vals[i]}")
plt.legend()
plt.show()
```

```{solution-end}
```


```{exercise}
:label: inequality_ex2

According to the definition of the top shares {eq}`topshares` we can also calculate the top percentile shares using the Lorenz curve.

Compute the top shares of US net wealth using the corresponding Lorenz curves data: ``f_vals_nw, l_vals_nw`` and linear interpolation.

Plot the top shares generated from Lorenz curve and the top shares approximated from data together.

```

```{solution-start} inequality_ex2
:class: dropdown
```

+++

Here is one solution:

```{code-cell} ipython3
def lorenz2top(f_val, l_val, p=0.1):
    t = lambda x: np.interp(x, f_val, l_val)
    return 1- t(1 - p)
```

```{code-cell} ipython3
top_shares_nw = []
for f_val, l_val in zip(f_vals_nw, l_vals_nw):
    top_shares_nw.append(lorenz2top(f_val, l_val))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 'US top shares: approximation vs Lorenz'
    name: top_shares_us_al
  image:
    alt: top_shares_us_al
---
fig, ax = plt.subplots()

ax.plot(years, df_topshares["topshare_n_wealth"], marker='o',\
   label="net wealth-approx")
ax.plot(years, top_shares_nw, marker='o', label="net wealth-lorenz")

ax.set_xlabel("year")
ax.set_ylabel("top $10\%$ share")
ax.legend()
plt.show()
```

```{solution-end}
```
