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

In this lecture, we will see expensions and contractions of economies from 1960s to the recent pandemic using [World Bank API](https://documents.worldbank.org/en/publication/documents-reports/api), and [FRED](https://fred.stlouisfed.org/) data.

In addition to what's in Anaconda, this lecture will need the following libraries to get World Bank and FRED

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
# Set Graphical Parameters
cycler = plt.cycler(linestyle=['-', '-.', '--', ':'], color=['#377eb8', '#ff7f00', '#4daf4a', '#ff334f'])
plt.rc('axes', prop_cycle=cycler)
```

## Data Acquaisition

We will use `wbgapi`, and Pandas `datareader` to retrieve data throughout this lecture.

This help us speed up the quary since we do not need to handle the raw JSON files.

So let's explore how to query data together.

We can use `wb.series.info` with parameter `q` to query available data from the World Bank.

For example, GDP growth is a key indicator to show the expension and contraction of level of economic activities.

Let's retrive GDP growth data together

```{code-cell} ipython3
wb.series.info(q='GDP growth')
```

```{code-cell} ipython3
wb.series.info(q='income share')
```

We can always learn more about the data by checking the metadata of the series

```{code-cell} ipython3
:tags: [hide-output]

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
def plot_comparison(data, country, title, ylabel, title_pos, ax, g_params, b_params, t_params, ylim=15, baseline=True):
    
    ax.plot(data.loc[country], label=country, **g_params)
    
    # Highlight Recessions
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    if ylim != None:
        ax.set_ylim([-ylim, ylim])
    ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim * title_pos, 'Oil Crisis\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim * title_pos, '1990s recession\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim * title_pos, 'GFC\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim * title_pos, 'Covid-19\n(2020)', **t_params) 
    if baseline:
        plt.axhline(y=0, color='black', linestyle='--')
    ax.set_title(title, pad=40)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# Define graphical parameters 
g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 'va':'center', 'ha':'center'}
```

Let's start with individual coutries.

We start with plotting the GDP growth rate for Unitied States

```{code-cell} ipython3
fig, ax = plt.subplots()

# Draw x-axis
plt.locator_params(axis='x', nbins=10)
ax.set_xticks([i for i in range(1960, 2021, 10)], minor=False)

country = 'United States'
title = 'United States (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, country, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

We find that there is a cyclical pattern across time, and volatile drops and rises during recessions

Let's look at a few more countries across the world

+++

Britain has a relative similar pattern compared to the US.

However it has a more signicant drop in the GDP growth during the global economic recessions.

```{code-cell} ipython3
fig, ax = plt.subplots()

# Draw x-axis
plt.locator_params(axis='x', nbins=10)
ax.set_xticks([i for i in range(1960, 2021, 10)], minor=False)

country = 'United Kingdom'
title = ' United Kingdom (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, country, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

Japan and Greece both had a history of rapid growth in the 1960s, but a slowed economic expension in the past decade.

We can see there is a general downward trend in additonal to fluctuations in the growth rate

```{code-cell} ipython3
fig, ax = plt.subplots()


country = 'Japan'
title = 'Japan (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, country, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'Greece'
title = ' Greece (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, country, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

Some countries have more volitile cycles.

```{code-cell} ipython3
fig, ax = plt.subplots()

country = 'Argentina'
title = 'Argentina (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison(gdp_growth, country, title, ylabel, 0.1, ax, g_params, b_params, t_params)
```

We find similar cyclical patterns across different countries.

Countries such as Argentina has a more volatile cycle compared to other economies. 

One interesting insight is that the GDP growth of Argentina did not fall in the two recessions in 1970s and 1990s when most of developed economy is affected.

We will come back to this point later.

+++

## Unemployment

Another important indicator of business cycles is the unemployment rate.

When there is a recession, it is more likely to have more working population.

We will show this using a long-run unemployment rate from FRED from [1929-1942](https://fred.stlouisfed.org/series/M0892AUSM156SNBR) and [1948-1011](https://fred.stlouisfed.org/series/UNRATE) with the unemployment rate between 1942-1948 estimated by [The Census Bureau](https://www.census.gov/library/publications/1975/compendia/hist_stats_colonial-1970.html).

```{code-cell} ipython3
start_date = datetime.datetime(1929, 1, 1)
end_date = datetime.datetime(1942, 6, 1)

unrate_history = web.DataReader("M0892AUSM156SNBR", "fred", start_date,end_date)
unrate_history.rename(columns={'M0892AUSM156SNBR': 'UNRATE'}, inplace=True)
```

```{code-cell} ipython3
import datetime

start_date = datetime.datetime(1948, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

unrate = web.DataReader("UNRATE", "fred", start_date, end_date)
```

For years between 1942-1948, we use data from

```{code-cell} ipython3
years = [datetime.datetime(year, 6, 1) for year in range(1942,1948)]
unrate_census = [4.7, 1.9, 1.2, 1.9, 3.9, 3.9]

unrate_census = {'DATE': years, 'UNRATE': unrate_census}
unrate_census = pd.DataFrame(unrate_census)
unrate_census.set_index('DATE', inplace=True)
```

```{code-cell} ipython3
:tags: []

start_date = datetime.datetime(1929, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

nber = web.DataReader("USREC", "fred", start_date, end_date)
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots()

ax.plot(unrate_history, **g_params, color='#377eb8', linestyle='-', linewidth=2)
ax.plot(unrate_census, **g_params, color='black', linestyle="--", label='Census Estimates', linewidth=2)
ax.plot(unrate, **g_params, color='#377eb8', linestyle="-", linewidth=2)

# Draw gray boxes according to NBER recession indicators
ax.fill_between(nber.index, 0, 1, where=nber['USREC']==1, color='grey', edgecolor="none",
                alpha=0.3, transform=ax.get_xaxis_transform(), 
                label='NBER Recession Indicators')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
ax.set_ylabel('Unemployment Rate (%)')

# Suppress Output
ax = ax.set_title('Long-run Unemployment Rate, 1929-2022\n with Recession Records (United States)', pad=20)
```

In the plot we see the expensions and contraction of the labour market has been highly correlated with recessions.

However, there is a delay in the improvement of labor market after recession, which is clearly visible for the Great Recession before the war started, and recessions from 1980s.

It also shows us how special the labor market condition during the pandemic is. 

The labour market recovers at an unprecedent rate.

+++

## Synchronization

In our previous dicussion, we find that developed economies have a more synchronized period of recessions.

Let's examine this trend further.

With slight modification, we can draw a plot that includes many countries

```{code-cell} ipython3
def plot_comparison_multi(data, countries, title, ylabel, title_pos, y_lim, ax, g_params, b_params, t_params):
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

```{code-cell} ipython3
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR', 'JPN', 'MEX'], labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
countries = ['United Kingdom', 'United States', 'Germany', 'Japan']
title = 'United Kingdom, United States, Germany, and Japan (GDP Growth Rate %)'
ylabel = 'GDP Growth Rate (%)'
title_height = 0.1
ax = plot_comparison_multi(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, 20, ax, g_params, b_params, t_params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
countries = ['Brazil', 'China', 'Argentina', 'Mexico']
title = 'Brazil, China, Argentina, and Mexico (GDP Growth Rate %)'
ax = plot_comparison_multi(gdp_growth.loc[countries, 1962:], countries, title, ylabel, 0.1, 20, ax, g_params, b_params, t_params)
```

By comparing the trend of GDP growth rates between developed and developing economies, we find the business cycles are more and more synchronized in 21st-century recessions.

Although we have seen synchronization in GDP growth as a general trend, we also need to acknowledge the experience of individual countries during the recession is very different.

Here we use unemployment rate as an example

```{code-cell} ipython3
unempl_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS',['CHN', 'USA', 'DEU', 'FRA', 'BRA', 'ARG', 'GBR', 'JPN'], labels=True)
unempl_rate = unempl_rate.set_index('Country')
unempl_rate.columns = unempl_rate.columns.str.replace('YR', '').astype(int)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Germany', 'France']
title = 'United Kingdom, United States, and Germany (Unemployment Rate %)'
ylabel = 'Unemployment Rate (National Estimate) (%)'
ax = plot_comparison_multi(unempl_rate, countries, title, ylabel, 0.05, None, ax, g_params, b_params, t_params)
```

Labor market in German was resilient to the GFC from 2007 to 2008, which is mostly linked to [its labor market policy and various socio-economic factors](http://ilo.org/wcmsp5/groups/public/---dgreports/---inst/documents/publication/wcms_449926.pdf).

The recovery from the crisis is another aspect.

France, as a country with strong labor union, has prolonged labour market recovery compared to the US and UK.

+++

## Leading Indicators and Correlated Factors for Business Cycles

Understanding leading indicators and correlated factors help policy maker to better understand the causes and results of business cycles.

We will discuss potential leading indicators and correlated factors from consumption, production, and credit level.

### Consumption

+++

One widely cited indicator for consumer confidence is [Consumer Sentiment Index](https://fred.stlouisfed.org/series/UMCSENT) published by University of Michigan.

We find the consumer sentiment maintains at a high level during the expension period, but there are significant drops before the recession hits.

There is also a clear negative correlation between consumer sentiment and [core consumer price index](https://fred.stlouisfed.org/series/CPILFESL).

This trend is more significant in period of [stagflation](https://en.wikipedia.org/wiki/Stagflation).

When the price of consumer commodities in the market is higher, the consumer confidence diminishes.

We will have a detailed look into inflation in the following lecture (TODO: Link to inflation lecture) 

```{code-cell} ipython3
start_date = datetime.datetime(1978, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

start_date_graph = datetime.datetime(1977, 1, 1)
end_date_graph = datetime.datetime(2023, 12, 31)


nber = web.DataReader("USREC", "fred", start_date, end_date)
consumer_confidence = web.DataReader("UMCSENT", "fred", start_date, end_date)

fig, ax = plt.subplots()
ax.plot(consumer_confidence, **g_params, color='#377eb8', linestyle='-', linewidth=2, label='Consumer Price Index')
ax.fill_between(nber.index, 0, 1, where=nber['USREC']==1, color='grey', edgecolor="none",
                alpha=0.3, transform=ax.get_xaxis_transform(), 
                label='NBER Recession Indicators')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_ylabel('Consumer Sentiment Index')

ax_t=ax.twinx()

inflation = web.DataReader("CPILFESL", "fred", start_date, end_date).pct_change(12) * 100

ax_t.plot(2020, 0, **g_params, linestyle='-', linewidth=2, label='Consumer Price Index')
ax_t.plot(inflation, **g_params, color='#ff7f00', linestyle='--', linewidth=2, label='CPI YoY Change (%)')
ax_t.fill_between(nber.index, 0, 1, where=nber['USREC']==1, color='grey', edgecolor="none",
                alpha=0.3, transform=ax.get_xaxis_transform(), label='NBER Recession Indicators')
ax_t.set_ylim([0, ax_t.get_ylim()[1]])
ax_t.set_xlim([start_date_graph, end_date_graph])
ax_t.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3)
ax_t.set_ylabel('Consumer Price Index (% Change)',)

# Suppress Output
ax = ax.set_title('University of Michigan Consumer Sentiment Index,\n and Year-over-year Consumer Price Index Change, 1978-2022 (United States)', pad=40)
```

### Production

Consumer confidence often affects the consumption pattern of consumers.

This is often manifest on the production side.

We find that the real output of the industry is also highly correlated with recessions in the economy. 

However, instead of being a leading factor, the peak of the contraction in the production delays compared to consumer confidence and inflation

```{code-cell} ipython3
start_date = datetime.datetime(1919, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

nber = web.DataReader("USREC", "fred", start_date, end_date)
consumer_confidence = web.DataReader("INDPRO", "fred", start_date, end_date).pct_change(12) * 100

fig, ax = plt.subplots()
ax.plot(consumer_confidence, **g_params, color='#377eb8', linestyle='-', linewidth=2, label='Consumer Price Index')
ax.fill_between(nber.index, 0, 1, where=nber['USREC']==1, color='grey', edgecolor="none",
                alpha=0.3, transform=ax.get_xaxis_transform(), 
                label='NBER Recession Indicators')
ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
ax.set_ylabel('YoY Real Ouput Change (%)')
ax = ax.set_title('Year-over-year Industrial Production: Total Index, 1919-2022 (United States)', pad=20)
```

### Credit Level

Credit contraction usually happends with recessions as lenders become more cautious and borrowers become more hesitant to take on additional debt. 

This can be due to several factors such as a decrease in overall economic activity, rising unemployment, and gloomy expections for the future.

One example is domestic credit to private sector by banks in the UK.

Note that the credit level expends rapidly in the period of economic expension, and stagnate or decreased after recessions

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
ylabel = '% of GDP'
ax = plot_comparison(private_credit, countries, title, ylabel, 0.05, ax, g_params, b_params, t_params, ylim=None, baseline=False)
```
