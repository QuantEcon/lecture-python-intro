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

# Business Cycle

## Overview

This lecture is about illustrateing business cycles in different countries and period.

The business cycle refers to the fluctuations in economic activity over time. These fluctuations can be observed in the form of expansions, contractions, recessions, and recoveries in the economy.

In this lecture, we will see expensions and contractions of economies from 1960s to the recent pandemic using [World Bank API](https://documents.worldbank.org/en/publication/documents-reports/api), [IMF API](http://www.bd-econ.com/imfapi1.html), and [FRED](https://fred.stlouisfed.org/) data.

In addition to what's in Anaconda, this lecture will need the following libraries to get World Bank and IMF data

```{code-cell} ipython3
:tags: [hide-output]

!pip install wbgapi
!pip install imfpy
```

We use the following imports

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import wbgapi as wb
from imfpy.retrievals import dots
```

```{code-cell} ipython3
# Set Graphical Parameters
cycler = plt.cycler(linestyle=['-', '-.', '--'], color=['#377eb8', '#ff7f00', '#4daf4a'])
plt.rc('axes', prop_cycle=cycler)
```

## Data Acquaisition

We will use `wbgapi`, `imfpy`, and Pandas `datareader` to retrieve data throughout this lecture.

This help us speed up the quary since we do not need to handle the raw JSON files.

So let's explore how to query data together.

We can use `wb.series.info` with parameter `q` to query available data from the World Bank (`imfpy. searches.database_codes()` in `imfpy`)

For example, GDP growth is a key indicator to show the expension and contraction of level of economic activities.

Let's retrive GDP growth data together

```{code-cell} ipython3
wb.series.info(q='GDP growth')
```

We can always learn more about the data by checking the metadata of the series

```{code-cell} ipython3
wb.series.metadata.get('NY.GDP.MKTP.KD.ZG')
```

We can now dive into the data we have.


## GDP Growth Rate

First we look at the GDP growth rate and unemployment rate.

Let's source our data from the World Bank and clean the data

```{code-cell} ipython3
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',['USA', 'ARG', 'GBR', 'GRC', 'JPN'], labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
gdp_growth
```

Now we write a function to generate plots

```{code-cell} ipython3
fig, ax = plt.subplots()

# Draw x-axis
ax.set_xticks([i for i in range(1960, 2021, 10)], minor=False)
plt.locator_params(axis='x', nbins=10)

def plot_comparison(data, countries, title, ylabel, title_pos, ax, g_params, b_params, t_params):
    # Allow the function to go through more than one series
    for country in countries:
        ax.plot(data.loc[country], label=country, **g_params)
    
    # Highlight Recessions
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim * title_pos, 'Oil Crisis\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim * title_pos, '1990s recession\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim * title_pos, 'GFC\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim * title_pos, 'Covid-19\n(2020)', **t_params) 
    ax.set_title(title, pad=40)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# Define graphical parameters 
g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 'va':'center', 'ha':'center'}
```

Let's start with individual coutries

```{code-cell} ipython3
country = 'United States'
title = ' United States (Real GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

## Unemployment

## Synchronization


```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Brazil', 'China', 'Argentina']
title = 'Brazil, China, Argentina (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Argentina']
title = 'Argentina (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Mexico']
title = 'Mexico (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Chile']
title = 'Chile (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Colombia']
title = 'Colombia (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['El Salvador']
title = 'El Salvador (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Haiti']
title = 'Haiti (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
unempl_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR', 'MEX', 'CHL', 'COL', 'SLV', 'HTI'], labels=True)
unempl_rate = unempl_rate.set_index('Country')
unempl_rate.columns = unempl_rate.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Germany']
title = 'United Kingdom, United States, and Germany (Unemployment Rate %)'
ylabel = 'Unemployment Rate (National Estimate) (%)'
ax = plot_comparison(unempl_rate, countries, title, ylabel, 0.03, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Brazil', 'China', 'Argentina']
title = 'Brazil, China, Argentina (Unemployment Rate %)'
ax = plot_comparison(unempl_rate, countries, title, ylabel, 0.04, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Brazil']
title = 'Brazil (Unemployment Rate %)'
ax = plot_comparison(unempl_rate, countries, title, ylabel, 0.04, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Chile']
title = 'Chile (Unemployment Rate %)'
ax = plot_comparison(unempl_rate, countries, title, ylabel, 0.04, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Colombia']
title = 'Colombia (Unemployment Rate %)'
ax = plot_comparison(unempl_rate, countries, title, ylabel, 0.04, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['El Salvador']
title = 'El Salvador (Unemployment Rate %)'
ax = plot_comparison(unempl_rate, countries, title, ylabel, 0.04, ax, g_params, b_params, t_params)
```

## Credit Level

```{code-cell} ipython3
private_credit = wb.data.DataFrame('FD.AST.PRVT.GD.ZS',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR'], labels=True)
private_credit = private_credit.set_index('Country')
private_credit.columns = private_credit.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Germany']
title = 'United Kingdom, United States, and Germany \n Domestic credit to private sector by banks (% of GDP)'
ylabel = '% of GDP'
ax = plot_comparison(private_credit, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Brazil', 'China', 'Argentina']
title = 'Brazil, China, Argentina \n Domestic credit to private sector by banks (% of GDP)'
ax = plot_comparison(private_credit, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'China']
title = 'United Kingdom and China \n Domestic credit to private sector by banks (% of GDP)'
ax = plot_comparison(private_credit, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

## Inflation

```{code-cell} ipython3
cpi = wb.data.DataFrame('FP.CPI.TOTL.ZG',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR'], labels=True)
cpi = cpi.set_index('Country')
cpi.columns = cpi.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Germany']
title = 'United Kingdom, United States, and Germany \n Inflation, consumer prices (annual %)'
ylabel = 'annual %'
ax = plot_comparison(cpi, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

## International Trade

```{code-cell} ipython3
trade_us = dots('US','W00', 1960, 2020, freq='A')
trade_us['Period'] = trade_us['Period'].astype('int')
```

```{code-cell} ipython3
def plot_trade(data, title, ylabel, title_pos, ax, g_params, b_params, t_params):
    ax.plot(data['Period'], data['Twoway Trade'], **g_params)
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim * title_pos, 'Oil Crisis\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim * title_pos, '1990s recession\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim * title_pos, 'GFC\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim * title_pos, 'Covid-19\n(2020)', **t_params) 
    ax.set_title(title, pad=40)
    ax.set_ylabel(ylabel)
    return ax


fig, ax = plt.subplots()
title = 'United States (International Trade Volumn)'
ylabel = 'US Dollars, Millions'
plot_UStrade = plot_trade(trade_us[['Period', 'Twoway Trade']], title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
trade_cn = dots('CN','W00', 1960, 2020, freq='A')

trade_cn['Period'] = trade_cn['Period'].astype('int')
title = 'China (International Trade Volumn)'
ylabel = 'US Dollars, Millions'
plot_trade_cn = plot_trade(trade_cn[['Period', 'Twoway Trade']], title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
trade_mx = dots('MX','W00', 1960, 2020, freq='A')

trade_mx['Period'] = trade_mx['Period'].astype('int')
title = 'Mexico (International Trade Volumn)'
ylabel = 'US Dollars, Millions'
plot_trade_mx = plot_trade(trade_mx[['Period', 'Twoway Trade']], title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
trade_ar = dots('AR','W00', 1960, 2020, freq='A')

trade_ar['Period'] = trade_ar['Period'].astype('int')
title = 'Argentina (International Trade Volumn)'
ylabel = 'US Dollars, Millions'
plot_trade_ar = plot_trade(trade_ar[['Period', 'Twoway Trade']], title, ylabel, 0.05, ax, g_params, b_params, t_params)
```
