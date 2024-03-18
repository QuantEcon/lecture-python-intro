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

We will install the following libraries.

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

And we use the following imports.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import random as rd
```

## The Lorenz curve

One popular measure of inequality is the Lorenz curve.

In this section we define the Lorenz curve and examine its properties.


### Definition

The Lorenz curve takes a sample $w_1, \ldots, w_n$ and produces a curve $L$.

We suppose that the sample $w_1, \ldots, w_n$ has been sorted from smallest to largest.

To aid our interpretation, suppose that we are measuring wealth 

*  $w_1$ is the wealth of the poorest member of the population and
*  $w_n$ is the wealth of the richest member of the population.

The curve $L$ is just a function $y = L(x)$ that we can plot and interpret.

To create it we first generate data points $(x_i, y_i)$  according to

\begin{equation*}
    x_i = \frac{i}{n},
    \qquad
    y_i = \frac{\sum_{j \leq i} w_j}{\sum_{j \leq n} w_j},
    \qquad i = 1, \ldots, n
\end{equation*}

Now the Lorenz curve $L$ is formed from these data points using interpolation.

(If we use a line plot in Matplotlib, the interpolation will be done for us.)

The meaning of the statement $y = L(x)$ is that the lowest $(100
\times x)$\% of people have $(100 \times y)$\% of all wealth.

* if $x=0.5$ and $y=0.1$, then the bottom 50% of the population
  owns 10% of the wealth.

In the discussion above we focused on wealth but the same ideas apply to
income, consumption, etc.

+++

### Lorenz curves of simulated data

Let's look at some examples and try to build understanding.

In the next figure, we generate $n=2000$ draws from a lognormal
distribution and treat these draws as our population.  

The straight line ($x=L(x)$ for all $x$) corresponds to perfect equality.  

The lognormal draws produce a less equal distribution.  

For example, if we imagine these draws as being observations of wealth across
a sample of households, then the dashed lines show that the bottom 80\% of
households own just over 40\% of total wealth.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Lorenz curve of simulated data"
    name: lorenz_simulated
---
n = 2000
sample = np.exp(np.random.randn(n))

fig, ax = plt.subplots()

f_vals, l_vals = qe.lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)

ax.legend(fontsize=12)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')

ax.set_ylim((0, 1))
ax.set_xlim((0, 1))

plt.show()
```

### Lorenz curves for US data

Next let's look at the real data, focusing on income and wealth in the US in
2016.

The following code block imports a subset of the dataset ``SCF_plus``,
which is derived from the [Survey of Consumer Finances](https://en.wikipedia.org/wiki/Survey_of_Consumer_Finances) (SCF).

```{code-cell} ipython3
url = 'https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/SCF_plus/SCF_plus_mini.csv'
df = pd.read_csv(url)
df = df.dropna()
df_income_wealth = df
```

```{code-cell} ipython3
df_income_wealth.head()
```

The following code block uses data stored in dataframe ``df_income_wealth`` to generate the Lorenz curves.

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
        f_val, l_val = qe.lorenz_curve(y)
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
    caption: "2016 US Lorenz curves"
    name: lorenz_us
  image:
    alt: lorenz_us
---
fig, ax = plt.subplots()

ax.plot(f_vals_nw[-1], l_vals_nw[-1], label=f'net wealth')
ax.plot(f_vals_ti[-1], l_vals_ti[-1], label=f'total income')
ax.plot(f_vals_li[-1], l_vals_li[-1], label=f'labor income')
ax.plot(f_vals_nw[-1], f_vals_nw[-1], label=f'equality')

ax.legend(fontsize=12)   
plt.show()
```

Here all the income and wealth measures are pre-tax.

Total income is the sum of households' all income sources, including labor income but excluding capital gains.

One key finding from this figure is that wealth inequality is significantly
more extreme than income inequality.

+++

## The Gini coefficient

The Lorenz curve is a useful visual representation of inequality in a
distribution.

Another popular measure of income and wealth inequality is the Gini coefficient.

The Gini coefficient is just a number, rather than a curve.

In this section we discuss the Gini coefficient and its relationship to the
Lorenz curve.


### Definition


As before, suppose that the sample $w_1, \ldots, w_n$ has been sorted from
smallest to largest.

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
equality and the Lorenz curve (e.g., the shaded area in the following Figure below).

The idea is that $G=0$ indicates complete equality, while $G=1$ indicates complete inequality.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Shaded Lorenz curve of simulated data"
    name: lorenz_gini
  image:
    alt: lorenz_gini
---
fig, ax = plt.subplots()

f_vals, l_vals = qe.lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'lognormal sample', lw=2)
ax.plot(f_vals, f_vals, label='equality', lw=2)

ax.legend(fontsize=12)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')

ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)

ax.set_ylim((0, 1))
ax.set_xlim((0, 1))

ax.text(0.04, 0.5, r'$G = 2 \times$ shaded area', fontsize=12)
  
plt.show()
```

### Gini coefficient dynamics of simulated data

Let's examine the Gini coefficient in some simulations.

The following code computes the Gini coefficients for five different
populations.

Each of these populations is generated by drawing from a 
lognormal distribution with parameters $\mu$ (mean) and $\sigma$ (standard deviation).

To create the five populations, we vary $\sigma$ over a grid of length $5$
between $0.2$ and $4$.

In each case we set $\mu = - \sigma^2 / 2$.

This implies that the mean of the distribution does not change with $\sigma$. 

(You can check this by looking up the expression for the mean of a lognormal
distribution.)

```{code-cell} ipython3
k = 5
σ_vals = np.linspace(0.2, 4, k)
n = 2_000

ginis = []

for σ in σ_vals:
    μ = -σ**2 / 2
    y = np.exp(μ + σ * np.random.randn(n))
    ginis.append(qe.gini_coefficient(y))
```

```{code-cell} ipython3
def plot_inequality_measures(x, y, legend, xlabel, ylabel):
    
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', label=legend)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.legend(fontsize=12)
    plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Gini coefficients of simulated data"
    name: gini_simulated
  image:
    alt: gini_simulated
---
plot_inequality_measures(σ_vals, 
                         ginis, 
                         'simulated', 
                         '$\sigma$', 
                         'gini coefficients')
```

The plots show that inequality rises with $\sigma$, according to the Gini
coefficient.

+++

### Gini coefficient dynamics for US data

Now let's look at Gini coefficients for US data derived from the SCF.

The following code creates a list called ``Ginis``.

 It stores data of Gini coefficients generated from the dataframe ``df_income_wealth`` and method [gini_coefficient](https://quanteconpy.readthedocs.io/en/latest/tools/inequality.html#quantecon.inequality.gini_coefficient), from [QuantEcon](https://quantecon.org/quantecon-py/) library.

```{code-cell} ipython3
:tags: [hide-input]

varlist = ['n_wealth',   # net wealth 
           't_income',   # total income
           'l_income']   # labor income

df = df_income_wealth

# create lists to store Gini for each inequality measure

Ginis = []

for var in varlist:
    # create lists to store Gini
    ginis = []
    
    for year in years:
        # repeat the observations according to their weights
        counts = list(round(df[df['year'] == year]['weights'] ))
        y = df[df['year'] == year][var].repeat(counts)
        y = np.asarray(y)
        
        rd.shuffle(y)    # shuffle the sequence
        
        # calculate and store Gini
        gini = qe.gini_coefficient(y)
        ginis.append(gini)
        
    Ginis.append(ginis)
```

```{code-cell} ipython3
ginis_nw, ginis_ti, ginis_li = Ginis
```

Let's plot the Gini coefficients for net wealth, labor income and total income.

```{code-cell} ipython3
# use an average to replace an outlier in labor income gini
ginis_li_new = ginis_li
ginis_li_new[5] = (ginis_li[4] + ginis_li[6]) / 2
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Gini coefficients of US net wealth"
    name: gini_wealth_us
  image:
    alt: gini_wealth_us
---
xlabel = "year"
ylabel = "gini coefficient"

fig, ax = plt.subplots()

ax.plot(years, ginis_nw, marker='o')

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)
    
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Gini coefficients of US income"
    name: gini_income_us
  image:
    alt: gini_income_us
---
xlabel = "year"
ylabel = "gini coefficient"

fig, ax = plt.subplots()

ax.plot(years, ginis_li_new, marker='o', label="labor income")
ax.plot(years, ginis_ti, marker='o', label="total income")

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

ax.legend(fontsize=12)
plt.show()
```

We see that, by this measure, inequality in wealth and income has risen
substantially since 1980.

The wealth time series exhibits a strong U-shape.


## Top shares

Another popular measure of inequality is the top shares.

Measuring specific shares is less complex than the Lorenz curve or the Gini
coefficient.

In this section we show how to compute top shares.

### Definition


As before, suppose that the sample $w_1, \ldots, w_n$ has been sorted from smallest to largest.

Given the Lorenz curve $y = L(x)$ defined above, the top $100 \times p \%$
share is defined as

$$
T(p) = 1 - L (1-p) 
    \approx \frac{\sum_{j\geq i} w_j}{ \sum_{j \leq n} w_j}, \quad i = \lfloor n (1-p)\rfloor
$$(topshares)

Here $\lfloor \cdot \rfloor$ is the floor function, which rounds any
number down to the integer less than or equal to that number.

+++

The following code uses the data from dataframe ``df_income_wealth`` to generate another dataframe ``df_topshares``.

``df_topshares`` stores the top 10 percent shares for the total income, the labor income and net wealth from 1950 to 2016 in US.

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
    caption: "US top shares"
    name: top_shares_us
  image:
    alt: top_shares_us
---
xlabel = "year"
ylabel = "top $10\%$ share"

fig, ax = plt.subplots()

ax.plot(years, df_topshares["topshare_l_income"],
        marker='o', label="labor income")
ax.plot(years, df_topshares["topshare_n_wealth"],
        marker='o', label="net wealth")
ax.plot(years, df_topshares["topshare_t_income"],
        marker='o', label="total income")

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

ax.legend(fontsize=12)
plt.show()
```

## Exercises

+++

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

+++

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
    f_val, l_val = qe._inequality.lorenz_curve(y)
    f_vals.append(f_val)
    l_vals.append(l_val)
    ginis.append(qe._inequality.gini_coefficient(y))
    topshares.append(calculate_top_share(y))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Top shares of simulated data"
    name: top_shares_simulated
  image:
    alt: top_shares_simulated
---
plot_inequality_measures(σ_vals, 
                         topshares, 
                         "simulated data", 
                         "$\sigma$", 
                         "top $10\%$ share") 
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Gini coefficients of simulated data"
    name: gini_coef_simulated
  image:
    alt: gini_coef_simulated
---
plot_inequality_measures(σ_vals, 
                         ginis, 
                         "simulated data", 
                         "$\sigma$", 
                         "gini coefficient") 
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Lorenz curves for simulated data"
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

+++

```{solution-start} inequality_ex2
:class: dropdown
```

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
    caption: "US top shares: approximation vs Lorenz"
    name: top_shares_us_al
  image:
    alt: top_shares_us_al
---
xlabel = "year"
ylabel = "top $10\%$ share"

fig, ax = plt.subplots()

ax.plot(years, df_topshares["topshare_n_wealth"], marker='o',\
   label="net wealth-approx")
ax.plot(years, top_shares_nw, marker='o', label="net wealth-lorenz")

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

ax.legend(fontsize=12)
plt.show()
```

```{solution-end}
```
