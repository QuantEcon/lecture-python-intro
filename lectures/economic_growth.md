---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Economic growth in the short and long run

```{index} single: Introduction to Economics
```

```{contents} Contents
:depth: 2
```

## World Bank Data - GDP Per Capita (Current US$)

GDP per capita is gross domestic product divided by midyear population. GDP is the sum of gross value added by all resident producers in the economy plus any product taxes and minus any subsidies not included in the value of the products. It is calculated without making deductions for depreciation of fabricated assets or for depletion and degradation of natural resources. Data are in current U.S. dollars.

```{code-cell} ipython3
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell} ipython3
wbi = pd.read_csv("datasets/GDP_per_capita_world_bank.csv")
```

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
