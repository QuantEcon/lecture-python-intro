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

```{code-cell} ipython3
!pip install wbgapi
```

# Business Cycle

## Overview

This lecture is about illustrateing business cycles in different countries and period.

Business cycle is one of the widely studied field since the birth of economics as a subject from .

In this lecture, we will see expensions and contractions of economies throughout the history with an emphasise on contemprary business cycles.

We use the following imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import wbgapi as wb
```

```{code-cell} ipython3
wb.series.info(q='GDP growth')
```

```{code-cell} ipython3
wb.series.info(q='unemployment')
```

```{code-cell} ipython3
wb.series.info(q='credit')
```

```{code-cell} ipython3
wb.series.info(q='consumer')
```

## GDP Growth Rate and Unemployment

First we look at the GDP growth rate and unemployment rate.

Let's source our data from the World Bank and clean the data

```{code-cell} ipython3
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR'], labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
gdp_growth
```

```{code-cell} ipython3
fig, ax = plt.subplots()
plt.locator_params(axis='x', nbins=10)
cycler = plt.cycler(linestyle=['-', '-.', '--'], color=['#377eb8', '#ff7f00', '#4daf4a'])
plt.rc('axes', prop_cycle=cycler)
ax.set_xticks([i for i in range(1960, 2021, 10)], minor=False)

def plot_comparison(data, countries, title, ylabel, title_pos, ax, g_params, b_params, t_params):
    for country in countries:
        ax.plot(data.loc[country], label=country, **g_params)
        # ax.plot(data.loc[country, 1990:1992], alpha=0.8)
        # ax.plot(data.loc[country, 1973:1975], alpha=0.8)
        # ax.plot(data.loc[country, 2007:2009], alpha=0.8)
        # ax.plot(data.loc[country, 2019:2021], alpha=0.8)
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

g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 'va':'center', 'ha':'center'}
countries = ['United Kingdom', 'United States', 'Germany']
title = 'United Kingdom, United States, and Germany (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Brazil', 'China', 'Argentina']
title = 'Brazil, China, Argentina (GDP Growth Rate %)'
ax = plot_comparison(gdp_growth, countries, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
unempl_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR'], labels=True)
unempl_rate = unempl_rate.set_index('Country')
unempl_rate.columns = unempl_rate.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
unempl_rate
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
private_credit = wb.data.DataFrame('FD.AST.PRVT.GD.ZS',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR'], labels=True)
private_credit = private_credit.set_index('Country')
private_credit.columns = private_credit.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
private_credit
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
cpi = wb.data.DataFrame('FP.CPI.TOTL.ZG',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR'], labels=True)
cpi = cpi.set_index('Country')
cpi.columns = cpi.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
cpi
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Germany']
title = 'United Kingdom, United States, and Germany \n Domestic credit to private sector by banks (% of GDP)'
ylabel = '% of GDP'
ax = plot_comparison(cpi, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['Brazil', 'China', 'Argentina']
title = 'Brazil, China, Argentina \n Domestic credit to private sector by banks (% of GDP)'
ax = plot_comparison(cpi, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params)
```
