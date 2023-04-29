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

# Long Run Growth

```{admonition} Lecture IN-WORK
:class: warning

This lecture is still **under construction**
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture looks at different growth trajectories across countries over the long term. 

While some countries have experienced long term rapid growth across that has last a hundred years, others have not. 

First let us import the packages needed to explore what the data says about long run growth.

```{code-cell} ipython3
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
```


A project initiated by [Angus Maddison](https://en.wikipedia.org/wiki/Angus_Maddison) has collected many historical time series that study economic growth. 

We can use the [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) to look at many different countries, including some countries dating back to the first century. 

```{tip}
The data can be downloaded from [this webpage](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) and clicking on the `Latest Maddison Project Release`. In this lecture we use the [Maddison Project Database 2020](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2020) using the `Excel` Format. The code we use here assumes you have downloaded that file and will teach you how to use [pandas](https://pandas.pydata.org) to import that data into a DataFrame.
```

If you don't want to fetch the data file from [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) you can download the file directly {download}`datasets/mpd2020.xlsx`.

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data
```

We can see that this dataset contains GDP per capita (gdppc) and population (pop) for many countries and years.

Let's look at how many and which countries are available in this dataset

```{code-cell} ipython3
data.country.unique()
```

```{code-cell} ipython3
len(data.country.unique())
```

We can now explore some of the 169 countries that are available. 

Let's loop over each country to understand which years are available for each country

```{code-cell} ipython3
cntry_years = []
for cntry in data.country.unique():
    cy_data = data[data.country == cntry]['year']
    ymin, ymax = cy_data.min(), cy_data.max()
    cntry_years.append((cntry, ymin, ymax))
cntry_years = pd.DataFrame(cntry_years, columns=['country', 'Min Year', 'Max Year']).set_index('country')
cntry_years
```

You can query this dataframe for each country of interest such as `Australia` by using `.loc`

```{code-cell} ipython3
cntry_years.loc['Australia']
```

Let us now reshape the original data into some convenient variables to enable quicker access to countries time series data.

We can build a useful mapping between country code's and country names in this dataset

```{code-cell} ipython3
code_to_name = data[['countrycode','country']].drop_duplicates().reset_index(drop=True).set_index(['countrycode'])
```

Then we can quickly focus on GDP per capita (gdp)

```{code-cell} ipython3
data
```

```{code-cell} ipython3
gdppc = data.set_index(['countrycode','year'])['gdppc']
gdppc = gdppc.unstack('countrycode')
```

```{code-cell} ipython3
gdppc
```

Looking at the United Kingdom we can first confirm we are using the correct country code

```{code-cell} ipython3
code_to_name.loc['GBR']
```

and then using that code to access and plot the data

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita (GBR)
    name: gdppc_gbr1
---
fig = plt.figure()
gdppc['GBR'].plot(ax = fig.gca())
plt.show()
```

We can see that the data is non-continuous for longer periods in early part of this milenium so we could choose to interpolate to get a continuous line plot.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita (GBR)
    name: gdppc_gbr2
---
fig = plt.figure(dpi=300)
cntry = 'GBR'
gdppc[cntry].interpolate().plot(
    ax = fig.gca(),
    ylabel = 'International $\'s',
    xlabel = 'Year'
)
plt.show()
```

:::{note}
[International Dollars](https://en.wikipedia.org/wiki/International_dollar) are a hypothetical unit of currency that has the same purchasing power parity that the U.S. Dollar has in the United States and any given time. They are also known as Geary–Khamis dollar (GK Dollars).
:::

As you can see from this chart economic growth started in earnest in the 18th Century and continued for the next two hundred years. 

How does this compare with other countries growth trajectories? Let's look at the United States (USA), United Kingdom (GBR), and China (CHN)

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita
    name: gdppc_usa_gbr_chn
---
fig = plt.figure(dpi=300)
ax = fig.gca()
cntry = ['USA', 'GBR', 'CHN']
line_color = ['blue', 'orange', 'green']
gdppc[cntry].plot(
    ax = ax,
    ylabel = 'International $\'s',
    xlabel = 'Year',
    color = line_color
)

# Build Custom Legend
legend_elements = []
for i,c in enumerate(cntry):
    line = Line2D([0], [0], color=line_color[i], lw=2, label=code_to_name.loc[c]['country'])
    legend_elements.append(line)
ax.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=[0.5, -0.25])
plt.show()
```

This dataset has been carefully curated to enable cross-country comparisons.

Let's compare the growth trajectories of Australia (AUS) and Argentina (ARG)

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per capita
    name: gdppc_aus_arg
---
fig = plt.figure(dpi=300)
ax = fig.gca()
cntry = ['AUS', 'ARG']
line_color = ['blue', 'orange']
gdppc[cntry].plot(
    ax = ax,
    ylabel = 'International $\'s',
    xlabel = 'Year',
    color = line_color
)

# Build Custom Legend
legend_elements = []
for i,c in enumerate(cntry):
    line = Line2D([0], [0], color=line_color[i], lw=2, label=code_to_name.loc[c]['country'])
    legend_elements.append(line)
ax.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=[0.5, -0.25])
plt.show()
```

As you can see the countries had similar GDP per capita levels with divergence starting around 1940. Australia's growth experience is both more continuous and less volatile post 1940.


## The Industrialized World

Now we can look at total Gross Domestic Product (GDP) rather than focusing on GDP per capita (as a proxy for living standards).

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data.set_index(['countrycode', 'year'], inplace=True)
data['gdp'] = data['gdppc'] * data['pop']
gdp = data['gdp'].unstack('countrycode')
```


### Early Industralization (1820 to 1940)


Gross Domestic Product

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP
    name: gdp1
---
fig = plt.figure(dpi=110)
ax = fig.gca()
cntry = ['DEU', 'SUN', 'USA', 'GBR', 'FRA', 'JPN', 'CHN']
start_year, end_year = (1820,1940)
line_color = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'slategrey']
gdp[cntry].loc[start_year:end_year].interpolate().plot(
    ax = ax,
    ylabel = 'International $\'s',
    xlabel = 'Year',
    color = line_color
)

# Build Custom Legend
legend_elements = []
for i,c in enumerate(cntry):
    line = Line2D([0], [0], color=line_color[i], lw=2, label=code_to_name.loc[c]['country'])
    legend_elements.append(line)
ax.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=[0.5, -0.26])
plt.show()
```

GDP per Capita

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita
    name: gdppc1
---
fig = plt.figure(dpi=110)
ax = fig.gca()
cntry = ['DEU', 'SUN', 'USA', 'GBR', 'FRA', 'JPN', 'CHN']
start_year, end_year = (1820,1940)
line_color = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'slategrey']
gdppc[cntry].loc[start_year:end_year].interpolate().plot(
    ax = ax,
    ylabel = 'International $\'s',
    xlabel = 'Year',
    color = line_color
)

# Build Custom Legend
legend_elements = []
for i,c in enumerate(cntry):
    line = Line2D([0], [0], color=line_color[i], lw=2, label=code_to_name.loc[c]['country'])
    legend_elements.append(line)
ax.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=[0.5, -0.25])
plt.show()
```

## The Modern Era (1970 to 2018)

Gross Domestic Product (GDP)

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP
    name: gdp2
---
fig = plt.figure(dpi=300)
ax = fig.gca()
cntry = ['DEU', 'SUN', 'USA', 'GBR', 'FRA', 'JPN', 'CHN']
start_year, end_year = (1970, 2018)
line_color = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'slategrey']
gdp[cntry].loc[start_year:end_year].interpolate().plot(
    ax = ax,
    ylabel = 'International $\'s',
    xlabel = 'Year',
    color = line_color
)

# Build Custom Legend
legend_elements = []
for i,c in enumerate(cntry):
    line = Line2D([0], [0], color=line_color[i], lw=2, label=code_to_name.loc[c]['country'])
    legend_elements.append(line)
ax.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=[0.5, -0.25])
plt.show()
```

GDP per Capita

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita
    name: gdppc2
---
fig = plt.figure(dpi=300)
ax = fig.gca()
cntry = ['DEU', 'SUN', 'USA', 'GBR', 'FRA', 'JPN', 'CHN']
start_year, end_year = (1970, 2018)
line_color = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'slategrey']
gdppc[cntry].loc[start_year:end_year].interpolate().plot(
    ax = ax,
    ylabel = 'International $\'s',
    xlabel = 'Year',
    color = line_color
)

# Build Custom Legend
legend_elements = []
for i,c in enumerate(cntry):
    line = Line2D([0], [0], color=line_color[i], lw=2, label=code_to_name.loc[c]['country'])
    legend_elements.append(line)
ax.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=[0.5, -0.3])
plt.show()
```


## Other Interesting Plots

Here are a collection of interesting plots that could be linked to interesting stories

Looking at China GDP per capita levels from 1500 through to the 1970's showed a long period of declining GDP per capital levels from 1700's to early 20th century. (Closed Border / Inward Looking Domestic Focused Policies?)

```{code-cell} ipython3
fig = plt.figure(dpi=300)
gdppc['CHN'].loc[1500:1980].interpolate().plot(ax=fig.gca())
plt.show()
```


China (CHN) then followed a very similar growth story from the 1980s through to current day China.

```{code-cell} ipython3
fig = plt.figure(dpi=300)
gdppc[['CHN', 'GBR']].interpolate().plot(ax = fig.gca())
plt.show()
```


## Regional Analysis

The [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) dataset also includes regional aggregations

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Regional data', header=(0,1,2), index_col=0)
data.columns = data.columns.droplevel(level=2)
```


We can save the raw data in a more convenient format to build a single table of regional GDP per capita

```{code-cell} ipython3
regionalgdppc = data['gdppc_2011'].copy()
regionalgdppc.index = pd.to_datetime(regionalgdppc.index, format='%Y')
```


Let us interpolate based on time to fill in any gaps in the dataset for the purpose of plotting

```{code-cell} ipython3
regionalgdppc.interpolate(method='time', inplace=True)
```


and record a dataset of world GDP per capita

```{code-cell} ipython3
worldgdppc = regionalgdppc['World GDP pc']
```

```{code-cell} ipython3
fig = plt.figure(dpi=300)
ax = fig.gca()
ax = worldgdppc.plot(
    ax = ax,
    title='World GDP per capita',
    xlabel='Year',
    ylabel='2011 US$',
)
```


Looking more closely, let us compare the time series for `Western Offshoots` and `Sub-Saharan Africa`

```{code-cell} ipython3
fig = plt.figure(dpi=300)
ax = fig.gca()
regionalgdppc[['Western Offshoots', 'Sub-Sahara Africa']].plot(ax = ax)
ax.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.26])
plt.show()
```


and more broadly at a number of different regions around the world

```{code-cell} ipython3
fig = plt.figure(dpi=300)
ax = fig.gca()
line_styles = ['-', '--', ':', '-.', '.', 'o', '-', '--', '-']
ax = regionalgdppc.plot(ax = ax, style=line_styles)
plt.legend(loc='lower center', ncol=3, bbox_to_anchor=[0.5, -0.4])
plt.show()
```
