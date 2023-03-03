---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# Long Run Growth

```{contents} Contents
:depth: 2
```

## Overview

This lecture looks at different growth trajectories across countries over the long term. 

While some countries have experienced long term rapid growth across that last hundred years, others have not. 

```{admonition} TODO
write an introduction
```

```{code-cell} ipython3
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data.set_index(['countrycode', 'year'], inplace=True)
```

```{code-cell} ipython3
gdppc = data['gdppc'].unstack('countrycode')
```

```{code-cell} ipython3
gdppc
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc['GBR'].interpolate().plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc['GBR'].plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc[['USA', 'GBR', 'CHN']].plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc[['AUS', 'ARG']].plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc[['AUS', 'NZL']].plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc[['CHN']].plot(ax = fig.gca())
```

```{code-cell} ipython3
gdppc[['CHN']]
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc[['CHN', 'GBR']].interpolate().plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
gdppc[['CHN', 'GBR']][0:500].interpolate().plot(ax=fig.gca())
```

+++ {"user_expressions": []}

## Regional Analysis

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Regional data', header=(0,1,2), index_col=0)
data.columns = data.columns.droplevel(level=2)
```

```{code-cell} ipython3
regionalgdppc = data['gdppc_2011'].copy()
regionalgdppc.index = pd.to_datetime(regionalgdppc.index, format='%Y')
```

```{code-cell} ipython3
regionalgdppc.interpolate(method='time', inplace=True)
```

```{code-cell} ipython3
worldgdppc = regionalgdppc['World GDP pc']
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
ax = worldgdppc.plot(
    ax = fig.gca(),
    title='World GDP per capita',
    xlabel='Year',
    ylabel='2011 US$',
)
```

```{code-cell} ipython3
fig = plt.figure(dpi=110)
regionalgdppc[['Western Offshoots', 'Sub-Sahara Africa']].plot(ax = fig.gca())
```

```{code-cell} ipython3
fig = plt.figure(dpi=200)
line_styles = ['-', '--', ':', '-.', '.', 'o']  # TODO: Improve this
ax = regionalgdppc.plot(ax = fig.gca(), style=line_styles)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

The following code reads in the data into a pandas data frame.

```{code-cell} ipython3
wbi = pd.read_csv("datasets/GDP_per_capita_world_bank.csv")
```

+++ {"user_expressions": []}

## Comparison of GDP between different Income Groups

A few countries from different Income Groups where chosen to compare the GDP at a more targeted level. GDP of countries from all income groups were compared. 

```{list-table}
:header-rows: 1

* - High Income
  - Upper middle income
  - Lower middle income
  - Low income
* - USA
  - China
  - India
  - Congo
* - Canada
  - Brazil
  - Pakistan
  - Uganda
* - Australia
  - Fiji
  - Bangladesh
  - Yemen
* - Japan
  - Jamaica
  - Vietnam
  - Afghanistan
```

### Plot for all countries

We compare time series graphs of all the countries in the list. The clear separation between high income countries and other groups are clearly seen. It seems at first glance that other income groups probably have similar economies. Let's look at that further in the following sections.

```{code-cell} ipython3
# USA, Canada, Australia, Japan, China, Brazil, Fiji, Jamaica, India, Pakistan, Bangladesh, Vietnam, Congo, Uganda, Yemen, Afghanistan
country_list = ['USA', 'CAN', 'AUS', 'JPN', 'CHN', 'BRA', 'FJI', 'JAM', 'IND', 'PAK', 'BGD', 'VNM', 'COD', 'UGA', 'YEM', 'AFG']
```

```{code-cell} ipython3
def filter_country_list_data(dataframe, country_list):
    wbi_cl = dataframe.loc[dataframe['Country Code'].isin(country_list)]
    wbi_cl = wbi_cl.drop(['Country Name' , 'Indicator Name', 'Indicator Code'], axis=1)
    wbi_cl = wbi_cl.transpose()
    wbi_cl = wbi_cl.rename(columns=wbi_cl.iloc[0])
    wbi_cl = wbi_cl.drop(wbi_cl.index[0])
    return wbi_cl
```

```{code-cell} ipython3
wbi_country_filtered = filter_country_list_data(wbi, country_list)
ax = wbi_country_filtered.plot()
ax.set_xlabel("year")
ax.set_ylabel("GDP per capita (current US$) ")
```

+++ {"user_expressions": []}

### Plot for Upper middle and lower middle income groups

Now, we compare the time-series graphs of GDP per capita for  upper middle and lower middle income group countries, taking one country from each group. China and Pakistan was chosen as they are from the same region. 
On analysing the graph, the difference is quite striking from 90s onwards. But also expected, as during that time China opened up for trade and labour. 
It can be concluded that, further inspection reveals the economies are vastly different in the present time, unlike what the previous graph was suggesting. 

```{code-cell} ipython3
# China, Pakistan (Upper middle income and lower middle income)
country_list_umi_lmi = ['CHN', 'PAK']
wbi_filtered_umi_lmi = filter_country_list_data(wbi, country_list_umi_lmi)
ax = wbi_filtered_umi_lmi.plot()
ax.set_xlabel("year")
ax.set_ylabel("GDP per capita (current US$) ")
```

### Plot for lower middle income

Here, we compare the time-series graphs of GDP per capita for two lower middle income group countries. Keeping Pakistan fixed in our set, we chose Vietnam as the second country. Apart from its turbulent past, its comeback from it and a steady growing economy qualifies it to be in this set. 
From the graph, we can see that Vietnam has done quite well from around 1990 onwards, and has quite surpassed Pakistan. We can also conclude that countries in the same income groups can be also be quite different.

```{code-cell} ipython3
# Vietnam, Pakistan (Lower middle income)
country_list_lmi = ['VNM', 'PAK']
wbi_filtered_lmi = filter_country_list_data(wbi, country_list_lmi)
ax = wbi_filtered_lmi.plot()
ax.set_xlabel("year")
ax.set_ylabel("GDP per capita (current US$) ")
```

### Plot for lower middle income and low income

Finally, we compare time-series graphs of GDP per capita between a lower middle income country and a low income country. Again, keeping Pakistan fixed in our set as a lower middle income country, we choose Democratic Republic of Congo as our second country from a low income group. Congo is chosen for no particular reason apart from its unstable political atmoshpere and a dwindling economy. 
On comapring we see quite a bit of difference between these countries. With Pakistan's GDP per capita being almost four times as much. Further strengthning our assumption that countries from different income groups can be quite different.

```{code-cell} ipython3
# Pakistan, Congo (Lower middle income, low income)
country_list_lmi_li = ['PAK', 'COD']
wbi_filtered_lmi = filter_country_list_data(wbi, country_list_lmi_li)
ax = wbi_filtered_lmi.plot()
ax.set_xlabel("year")
ax.set_ylabel("GDP per capita (current US$) ")
```

+++ {"user_expressions": []}

## Histogram comparison between 1960, 1990, 2020

We compare histograms of the **log** of GDP per capita for the years 1960, 1990 and 2020 for around 170 countries. The years have been chosen to give sufficient time gap between the histograms. We see that the overall plot is shifting towards right, denoting the upward trend in GDP per capita worldwide. And also, the overall distribution is becoming more Gaussian. Which indicates that the economies have gotten more uniform over the years. Economic disparities are getting lesser possibly because of globalisation, technological advancements, better use of resources etc. 

```{code-cell} ipython3
def get_log_hist(data, years):
    filtered_data = data.filter(items=['Country Code', years[0], years[1], years[2]])
    log_gdp = filtered_data.iloc[:,1:].transform(lambda x: np.log(x))
    max_log_gdp = log_gdp.max(numeric_only=True).max()
    min_log_gdp = log_gdp.min(numeric_only=True).min()
    log_gdp.hist(bins=16, range=[min_log_gdp, max_log_gdp])
```

```{code-cell} ipython3
## All countries
wbiall = wbi.drop(['Country Name' , 'Indicator Name', 'Indicator Code'], axis=1)
get_log_hist(wbiall, ['1960', '1990', '2020'])
```

```{code-cell} ipython3

```
