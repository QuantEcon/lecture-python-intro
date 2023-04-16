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

+++ {"user_expressions": []}

# Business Cycles

## Overview

In this lecture we study business cycles, which 
are fluctuations in economic activity over time.

These fluctuations can be observed in the form of expansions (booms), contractions (recessions), and recoveries.

We will look into a series of economic indicators to visualize the expansions and contractions of economies from the 1960s to the recent pandemic using [World Bank](https://documents.worldbank.org/en/publication/documents-reports/api) and [FRED](https://fred.stlouisfed.org/) data.

In addition to those installed by Anaconda, this lecture requires
libraries to obtain World Bank and FRED data:

```{code-cell} ipython3
:tags: [hide-output]

!pip install wbgapi
!pip install pandas-datareader
```

We use the following imports

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import datetime
import wbgapi as wb
import pandas_datareader.data as web
```

```{code-cell} ipython3
:tags: [hide-input]

# Set Graphical Parameters
cycler = plt.cycler(linestyle=['-', '-.', '--', ':'], color=['#377eb8', '#ff7f00', '#4daf4a', '#ff334f'])
plt.rc('axes', prop_cycle=cycler)
```

+++ {"user_expressions": []}

## Data Acquisition

We will use `wbgapi` and `pandas_datareader` to retrieve data throughout this
lecture.

Let's explore how to query data first.

We can use `wb.series.info` with the argument `q` to query available data from
the [World Bank](https://www.worldbank.org/en/home).

For example, let's retrieve the ID to query GDP growth data.

```{code-cell} ipython3
wb.series.info(q='GDP growth')
```

+++ {"user_expressions": []}

Now we use this series ID to obtain the data.

```{code-cell} ipython3
:tags: [hide-output]

gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['USA', 'ARG', 'GBR', 'GRC', 'JPN'], labels=True)
gdp_growth
```

+++ {"user_expressions": []}

We can learn more about the data by checking the series metadata.

```{code-cell} ipython3
:tags: [hide-output]

wb.series.metadata.get('NY.GDP.MKTP.KD.ZG')
```

+++ {"user_expressions": []}

Let's dive into the data with the tools we have.




## GDP Growth Rate

First we look at the GDP growth rate. 

Let's source our data from the World Bank and clean it.

```{code-cell} ipython3
# Use the series ID retrived before
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['USA', 'ARG', 'GBR', 'GRC', 'JPN'], labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

Here's a first look at the data, which measures the GDP growth rate in
percentages.

```{code-cell} ipython3
gdp_growth
```

+++ {"user_expressions": []}

The cell below contains a function to generate plots for individual countries.

```{code-cell} ipython3
:tags: [hide-input]

def plot_comparison(data, country, title, 
                    ylabel, title_pos, ax, g_params,
                     b_params, t_params, ylim=15, baseline=0):
    
    ax.plot(data.loc[country], label=country, **g_params)
    
    # Highlight Recessions
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    if ylim != None:
        ax.set_ylim([-ylim, ylim])
    else:
        ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim * title_pos,
            'Oil Crisis\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim * title_pos,
            '1990s recession\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim * title_pos,
            'GFC\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim * title_pos,
            'Covid-19\n(2020)', **t_params)
    if baseline != None:
        ax.axhline(y=baseline, color='black', linestyle='--')
    ax.set_title(title, pad=40)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# Define graphical parameters 
g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 
            'va':'center', 'ha':'center'}
```

+++ {"user_expressions": []}

Now we can plot the data as a time series.

Let's start with the United States.

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'United States'
title = 'United States (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
_ = plot_comparison(gdp_growth, country, 
                    title, ylabel, 0.1, ax, 
                    g_params, b_params, t_params)
```

+++ {"user_expressions": []}

GDP growth is positive on average and trending slightly downward over time.

We also see fluctuations over GDP growth over time, some of which are quite large.

Let's look at a few more countries to get a basis for comparison.

+++

The United Kingdom (UK) has a similar pattern to the US, with a slow decline
in the growth rate and significant fluctuations.

Notice the very large dip during the Covid-19 pandemic.

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'United Kingdom'
title = ' United Kingdom (GDP Growth Rate %)'
title_height = 0.1
_ = plot_comparison(gdp_growth, country, title,
                    ylabel, 0.1, ax, g_params, 
                    b_params, t_params)
```

+++ {"user_expressions": []}

Now let's consider Japan, which experienced rapid growth in the 1960s and
1970s, followed by slowed expansion in the past two decades.

Major dips in the growth rate coincided with the Oil Crisis of the 1970s, the
GFC and the Covid-19 pandemic.

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'Japan'
title = 'Japan (GDP Growth Rate %)'
_ = plot_comparison(gdp_growth, country, title, 
                    ylabel, 0.1, ax, g_params, 
                    b_params, t_params)
```

Now let's study Greece.

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'Greece'
title = ' Greece (GDP Growth Rate %)'
_ = plot_comparison(gdp_growth, country, title, 
                    ylabel, 0.1, ax, g_params, 
                    b_params, t_params)
```

Greece had a significant drop in GDP growth around 2010-2011, during the peak
of the Greek debt crisis.

Next let's consider Argentina.

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'Argentina'
title = 'Argentina (GDP Growth Rate %)'
_ = plot_comparison(gdp_growth, country, title, 
                    ylabel, 0.1, ax, 
                    g_params, b_params, t_params)
```

+++ {"user_expressions": []}

The figure shows that Argentina has experienced more volatile cycles than
the economies mentioned above.

At the same time, growth of Argentina did not fall during the two developed
economy recessions in the 1970s and 1990s.

+++ {"user_expressions": []}

## Unemployment

Another important measure of business cycles is the unemployment rate.

During a recession, it is more likely that a larger proportion of the working
population will be laid off.

We demonstrate this using a long-run unemployment rate from FRED spanning from [1929-1942](https://fred.stlouisfed.org/series/M0892AUSM156SNBR) to [1948-2022](https://fred.stlouisfed.org/series/UNRATE) with the unemployment rate between 1942 and 1948 estimated by [The Census Bureau](https://www.census.gov/library/publications/1975/compendia/hist_stats_colonial-1970.html).

```{code-cell} ipython3
start_date = datetime.datetime(1929, 1, 1)
end_date = datetime.datetime(1942, 6, 1)

unrate_history = web.DataReader('M0892AUSM156SNBR', 'fred', start_date,end_date)
unrate_history.rename(columns={'M0892AUSM156SNBR': 'UNRATE'}, inplace=True)
```

```{code-cell} ipython3
start_date = datetime.datetime(1948, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

unrate = web.DataReader('UNRATE', 'fred', start_date, end_date)
```

```{code-cell} ipython3
# We use the census bureau's estimate for the unemployment rate 
# between 1942 and 1948
years = [datetime.datetime(year, 6, 1) for year in range(1942,1948)]
unrate_census = [4.7, 1.9, 1.2, 1.9, 3.9, 3.9]

unrate_census = {'DATE': years, 'UNRATE': unrate_census}
unrate_census = pd.DataFrame(unrate_census)
unrate_census.set_index('DATE', inplace=True)
```

```{code-cell} ipython3
# Obtain the NBER-defined recession periods
start_date = datetime.datetime(1929, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

nber = web.DataReader('USREC', 'fred', start_date, end_date)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(unrate_history, **g_params, color='#377eb8', linestyle='-', linewidth=2)
ax.plot(unrate_census, **g_params, color='black', linestyle='--', label='Census Estimates', linewidth=2)
ax.plot(unrate, **g_params, color='#377eb8', linestyle='-', linewidth=2)

# Draw gray boxes according to NBER recession indicators
ax.fill_between(nber.index, 0, 1, where=nber['USREC']==1, 
                color='grey', edgecolor='none',
                alpha=0.3, transform=ax.get_xaxis_transform(), 
                label='NBER Recession Indicators')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)
ax.set_ylabel('Unemployment Rate (%)')

_ = ax.set_title('Long-run Unemployment Rate, 1929-2022\n with Recession Indicators (United States)', pad=30)
```

+++ {"user_expressions": []}

In the plot, we can see that the expansions and contractions of the labor
market have been highly correlated with recessions. 

However, there is often a delay in the recovery of the labor market after
recessions.

This trend is clearly visible in the 1930s, as well as in recessions in the
1980s. 

It also shows us how unique labor market conditions have been during the
post-pandemic recovery. 

The labor market has recovered at an unprecedented rate, leading to the
tightest point in the past decades after the shock in 2020-2021.

+++ {"user_expressions": []}

(synchronization)=
## Synchronization

In our previous discussion, we found that developed economies have had
relatively synchronized periods of recession. 

At the same time, this synchronization does not appear in Argentina until the 2000s. 

Let's examine this trend further. 

With slight modifications, we can use our previous function to draw a plot
that includes many countries

```{code-cell} ipython3
def plot_comparison_multi(data, countries, title, 
                        ylabel, title_pos, y_lim, ax, 
                        g_params, b_params, t_params, baseline=0):

    # Allow the function to go through more than one series
    for country in countries:
        ax.plot(data.loc[country], label=country, **g_params)
    
    # Highlight Recessions
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    if y_lim != None:
        ax.set_ylim([-y_lim, y_lim])
    ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim * title_pos, 
            'Oil Crisis\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim * title_pos, 
            '1990s recession\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim * title_pos, 
            'GFC\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim * title_pos, 
            'Covid-19\n(2020)', **t_params) 
    if baseline != None:
        ax.hlines(y=baseline, xmin=ax.get_xlim()[0], 
                  xmax=ax.get_xlim()[1], color='black', 
                  linestyle='--')
    ax.set_title(title, pad=40)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# Define graphical parameters 
g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 
            'va':'center', 'ha':'center'}
```

```{code-cell} ipython3
# Obtain GDP growth rate for a list of countries
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR', 'JPN', 'MEX'], labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
countries = ['United Kingdom', 'United States', 'Germany', 'Japan']
title = 'United Kingdom, United States, Germany, and Japan (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
_ = plot_comparison_multi(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, 20, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
countries = ['Brazil', 'China', 'Argentina', 'Mexico']
title = 'Brazil, China, Argentina, and Mexico (GDP Growth Rate %)'
_ = plot_comparison_multi(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, 20, ax, g_params, b_params, t_params)
```

+++ {"user_expressions": []}

On comparison of GDP growth rates between developed and developing
economies, we find that business cycles are becoming more synchronized in 21st-century recessions.

However, emerging and less developed economies often experience more volatile
changes throughout the economic cycles. 

Although we see synchronization in GDP growth as a general trend, the experience of individual countries during
the recession often differs. 

We use unemployment rate and the recovery of labor market conditions
as another example.

```{code-cell} ipython3
unempl_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS',
    ['USA', 'FRA', 'GBR', 'JPN'], labels=True)
unempl_rate = unempl_rate.set_index('Country')
unempl_rate.columns = unempl_rate.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Japan', 'France']
title = 'United Kingdom, United States, Japan, and France (Unemployment Rate %)'
ylabel = 'Unemployment Rate (National Estimate) (%)'
_ = plot_comparison_multi(unempl_rate, countries, title, 
                          ylabel, 0.05, None, ax, g_params, 
                          b_params, t_params, baseline=None)
```

France, with its strong labor unions, has a prolonged labor market recovery
compared to the US and UK. 

However, Japan has a history of very low and stable unemployment rates due to
a constellation of social, demographic, and cultural factors.

+++ {"user_expressions": []}

## Leading Indicators and Correlated Factors for Business Cycles

Examining leading indicators and correlated factors helps policymakers to
understand the causes and results of business cycles. 

We will discuss potential leading indicators and correlated factors from three
perspectives: consumption, production, and credit level.

### Consumption

+++ {"user_expressions": []}

Consumption depends on consumers' confidence towards their
income and the overall performance of the economy in the future. 

One widely cited indicator for consumer confidence is the [Consumer Sentiment Index](https://fred.stlouisfed.org/series/UMCSENT) published by the University
of Michigan.

We find that consumer sentiment remains high during periods of expansion, but there are significant drops before recession hits.

There is also a clear negative correlation between consumer sentiment and [core consumer price index](https://fred.stlouisfed.org/series/CPILFESL).

This trend is more significant in the period of [stagflation](https://en.wikipedia.org/wiki/Stagflation).

When the price of consumer commodities rises, consumer confidence diminishes.

```{code-cell} ipython3
start_date = datetime.datetime(1978, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

# Limit the plot to a specific range
start_date_graph = datetime.datetime(1977, 1, 1)
end_date_graph = datetime.datetime(2023, 12, 31)

nber = web.DataReader('USREC', 'fred', start_date, end_date)
consumer_confidence = web.DataReader('UMCSENT', 'fred', start_date, end_date)

fig, ax = plt.subplots()
ax.plot(consumer_confidence, **g_params, 
        color='#377eb8', linestyle='-', 
        linewidth=2)
ax.fill_between(nber.index, 0, 1, 
            where=nber['USREC']==1, color='grey', edgecolor='none',
            alpha=0.3, transform=ax.get_xaxis_transform(), 
            label='NBER Recession Indicators')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_ylabel('Consumer Sentiment Index')

# Plot CPI on another y-axis
ax_t = ax.twinx()
inflation = web.DataReader('CPILFESL', 'fred', 
                start_date, end_date).pct_change(12) * 100

# Add CPI on the legend without drawing the line again
ax_t.plot(2020, 0, **g_params, linestyle='-', 
          linewidth=2, label='Consumer Sentiment Index')
ax_t.plot(inflation, **g_params, color='#ff7f00', linestyle='--', 
          linewidth=2, label='CPI YoY Change (%)')
ax_t.fill_between(nber.index, 0, 1, where=nber['USREC']==1, 
                  color='grey', edgecolor='none',
                  alpha=0.3, transform=ax.get_xaxis_transform(), 
                  label='NBER Recession Indicators')
ax_t.set_ylim([0, ax_t.get_ylim()[1]])
ax_t.set_xlim([start_date_graph, end_date_graph])
ax_t.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fontsize=9)
ax_t.set_ylabel('Consumer Price Index (% Change)',)

# Suppress the text output
_ = ax.set_title('University of Michigan Consumer Sentiment Index,\n and \
Year-over-year Consumer Price Index Change, 1978-2022 (United States)', pad=30)
```

+++ {"user_expressions": []}

### Production

Consumers' confidence often influences their consumption pattern.

This often manifests on the production side.

We find that real industrial output is highly correlated with
recessions in the economy. 

However, it is not a leading indicator, as the peak of contraction in production delays compared to consumer confidence and inflation.

```{code-cell} ipython3
start_date = datetime.datetime(1980, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

nber = web.DataReader('USREC', 'fred', start_date, end_date)
consumer_confidence = web.DataReader('INDPRO', 'fred', start_date, end_date).pct_change(12) * 100

fig, ax = plt.subplots()
ax.plot(consumer_confidence, **g_params, color='#377eb8', linestyle='-', linewidth=2, label='Consumer Price Index')
ax.fill_between(nber.index, 0, 1, where=nber['USREC']==1, color='grey', edgecolor='none',
                alpha=0.3, transform=ax.get_xaxis_transform(), 
                label='NBER Recession Indicators')
ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
ax.set_ylabel('YoY Real Ouput Change (%)')
ax = ax.set_title('Year-over-year Industrial Production: \
Total Index, 1919-2022 (United States)', pad=20)
```

+++ {"user_expressions": []}

### Credit Level

Credit contractions often occur during recessions, as lenders become more
cautious and borrowers become more hesitant to take on additional debt.

This can be due to several factors such as a decrease in overall economic
activity, rising unemployment, and gloomy expectations for the future.

One example is domestic credit to the private sector by banks in the UK.

```{code-cell} ipython3
private_credit = wb.data.DataFrame('FS.AST.PRVT.GD.ZS',['GBR'], labels=True)
private_credit = private_credit.set_index('Country')
private_credit.columns = private_credit.columns.str.replace('YR', '').astype(int)
private_credit
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = 'United Kingdom'
title = 'Domestic Credit to Private Sector by Banks, United Kingdom (% of GDP)'
ylabel = 'Credit Level (% of GDP)'
ax = plot_comparison(private_credit, countries, title, 
                     ylabel, 0.05, ax, g_params, b_params, 
                     t_params, ylim=None, baseline=None)
```

+++ {"user_expressions": []}

Note that the credit rises in periods of economic expansion
and stagnates or even contracts after recessions.

```{code-cell} ipython3

```
