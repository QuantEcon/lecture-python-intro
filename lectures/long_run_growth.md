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

# Long Run Growth

## Overview

This lecture looks at different growth trajectories across countries over the long term. 

While some countries have experienced long term rapid growth across that has lasted a hundred years, others have not. 

First let's import the packages needed to explore what the data says about long run growth.

```{code-cell} ipython3
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from matplotlib.lines import Line2D
```

+++ {"user_expressions": []}

## Setting up

+++ {"user_expressions": []}

A project initiated by [Angus Maddison](https://en.wikipedia.org/wiki/Angus_Maddison) has collected many historical time series that study economic growth. 

We can use the [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) to look at many different countries, including some countries dating back to the first century. 

```{tip}
The data can be downloaded from [this webpage](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) and clicking on the `Latest Maddison Project Release`. In this lecture we use the [Maddison Project Database 2020](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2020) using the `Excel` Format.
```

If you don't want to fetch the data file from [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) you can download the file directly {download}`datasets/mpd2020.xlsx`.

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data
```

+++ {"user_expressions": []}

We can see that this dataset contains GDP per capita (gdppc) and population (pop) for many countries and years.

Let's look at how many and which countries are available in this dataset

```{code-cell} ipython3
len(data.country.unique())
```

+++ {"user_expressions": []}

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

+++ {"user_expressions": []}

You can query this dataframe for each country of interest such as `Australia` by using `.loc`

```{code-cell} ipython3
cntry_years.loc['Australia']
```

+++ {"user_expressions": []}

Let us now reshape the original data into some convenient variables to enable quicker access to countries time series data.

We can build a useful mapping between country code's and country names in this dataset

```{code-cell} ipython3
code_to_name = data[['countrycode','country']].drop_duplicates().reset_index(drop=True).set_index(['countrycode'])
```

+++ {"user_expressions": []}

Then we can quickly focus on GDP per capita (gdp)

```{code-cell} ipython3
data
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

country_names = data['countrycode']

# Generate a colormap with the number of colors matching the number of countries
colors = cm.Dark2(np.linspace(0, 0.8, len(country_names)))

# Create a dictionary to map each country to its corresponding color
color_mapping = {country: color for country, color in zip(country_names, colors)}
```

```{code-cell} ipython3
gdppc = data.set_index(['countrycode','year'])['gdppc']
gdppc = gdppc.unstack('countrycode')
```

```{code-cell} ipython3
gdppc
```

+++ {"user_expressions": []}

Looking at the United Kingdom we can first confirm we are using the correct country code

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita (GBR)
    name: gdppc_gbr1
---
fig, ax = plt.subplots(dpi=300)
cntry = 'GBR'
_ = gdppc[cntry].plot(
    ax = fig.gca(),
    ylabel = 'International $\'s',
    xlabel = 'Year',
    linestyle='-',
    color=color_mapping['GBR'])
```

+++ {"user_expressions": []}

We can see that the data is non-continuous for longer periods in the early part of this millennium, so we could choose to interpolate to get a continuous line plot.

Here we use dashed lines to indicate interpolated trends

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita (GBR)
    name: gdppc_gbr2
---
fig, ax = plt.subplots(dpi=300)
cntry = 'GBR'
ax.plot(gdppc[cntry].interpolate(),
        linestyle='--',
        lw=2,
        color=color_mapping[cntry])

ax.plot(gdppc[cntry],
        linestyle='-',
        lw=2,
        color=color_mapping[cntry])
ax.set_ylabel('International $\'s')
ax.set_xlabel('Year')
plt.show()
```

+++ {"user_expressions": []}

We can now put this into a function to generate plots for a list of countries

```{code-cell} ipython3
def draw_interp_plots(series, xlabel, ylabel, color_mapping, code_to_name, lw, logscale, ax):

    for i, c in enumerate(cntry):
        
        df_interpolated = series[c].interpolate(limit_area='inside')
        interpolated_data = df_interpolated[series[c].isnull()]
        ax.plot(interpolated_data,
                linestyle='--',
                lw=lw,
                alpha=0.7,
                color=color_mapping[c])

        ax.plot(series[c],
                linestyle='-',
                lw=lw,
                color=color_mapping[c],
                alpha=0.8,
                label=code_to_name.loc[c]['country'])
        
        if logscale == True:
            ax.set_yscale('log')
            
    ax.legend(loc='lower center', ncol=5, bbox_to_anchor=[0.5, -0.25])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return ax
```

+++ {"user_expressions": []}

:::{note}
[International Dollars](https://en.wikipedia.org/wiki/International_dollar) are a hypothetical unit of currency that has the same purchasing power parity that the U.S. Dollar has in the United States at any given time. They are also known as Geary–Khamis dollars (GK Dollars).
:::

As you can see from this chart economic growth started in earnest in the 18th century and continued for the next two hundred years. 

How does this compare with other countries' growth trajectories? 

Let's look at the United States (USA), United Kingdom (GBR), and China (CHN)

```{code-cell} ipython3
Event = namedtuple('Event', ['year_range', 'y_text', 'text', 'color', 'ymax'])

fig, ax = plt.subplots(dpi=300)

cntry = ['CHN', 'GBR', 'USA']
ax = draw_interp_plots(gdppc[cntry].loc[1200:],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, True, ax)

ylim = ax.get_ylim()[1]
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'fontsize': 5, 
            'va':'center', 'ha':'center'}

events = [
Event((1315, 1321), ylim + ylim*0.1, 'the Great Famine\n(1315-1321)', color_mapping['GBR'], 1),
Event((1348, 1375), ylim + ylim*0.4, 'the Black Death\n(1348-1375)', color_mapping['GBR'], 1.05),
Event((1650, 1652), ylim + ylim*0.1, 'the Navigation Act\n(1651)', color_mapping['GBR'], 1),
Event((1848, 1850), ylim + ylim*0.8, 'the Repeal of Navigation Act\n(1849)', color_mapping['GBR'], 1.1),
Event((1655, 1684), ylim + ylim*0.4, 'Closed-door Policy\n(1655-1684)', color_mapping['CHN'], 1.05),
Event((1760, 1840), ylim + ylim*0.4, 'Industrial Revolution\n(1760-1840)', 'grey', 1.05),
Event((1788, 1790), ylim + ylim*0.1, 'US Federation\n(1789)', color_mapping['USA'], 1),
Event((1929, 1939), ylim + ylim*0.1, 'the Great Depression\n(1929–1939)', 'grey', 1),
Event((1978, 1979), ylim + ylim*0.4, 'Reform and Opening-up\n(1978-1979)', color_mapping['CHN'], 1.05)
]

def draw_events(events, ax):
    # Iterate over events and add annotations and vertical lines
    for event in events:
        event_mid = sum(event.year_range)/2
        ax.text(event_mid, 
                event.y_text, event.text, 
                color=event.color, **t_params)
        ax.axvspan(*event.year_range, color=event.color, alpha=0.2)
        ax.axvline(event_mid, ymin=1, ymax=event.ymax, color=event.color, linestyle='-', clip_on=False, alpha=0.15)
        
# Draw events
draw_events(events, ax)
plt.show()
```

(TODO: Finalize trend)
We can see some interesting trends:

- Most of the growth happened in the past 150 years after the industrial revolution.
- There was a divergence between the West and the East during the process of industrialization (from 1820 to 1940).
- The gap is rapidly closing in the modern era.
- The shift in the paradigm in policy is usually intertwined with the technological and political.

+++ {"user_expressions": []}

Looking at China's GDP per capita levels from 1500 through to the 1970s showed a long period of declining GDP per capital levels from the 1700s to the early 20th century.

(TODO: Finalize trend)
Trends to note:
- Period of economic downturn after the Closed-door Policy by the Qing government
- Missing out on the industrial revolution
- Self-Strengthening Movement may help the growth but in a very mild way
- Modern Chinese economic policies and the growth after the founding of the PRC (political stability) and after the Reform and Opening-up 

```{code-cell} ipython3
fig, ax = plt.subplots(dpi=300)
cntry = ['CHN']
ax = draw_interp_plots(gdppc[cntry].loc[1600:2000],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, True, ax)

# Define the namedtuple for the data points
ylim = ax.get_ylim()[1]

events = [
Event((1655, 1684), ylim + ylim*0.05, 'Closed-door Policy\n(1655-1684)', 'tab:orange', 1),
Event((1760, 1840), ylim + ylim*0.05, 'Industrial Revolution\n(1760-1840)', 'grey', 1),
Event((1839, 1842), ylim + ylim*0.15, 'First Opium War\n(1839–1842)', 'tab:red', 1.05),
Event((1861, 1895), ylim + ylim*0.25, 'Self-Strengthening Movement\n(1861–1895)', 'tab:blue', 1.09),
Event((1939, 1945), ylim + ylim*0.05, 'WW 2\n(1939-1945)', 'tab:red', 1),
Event((1948, 1950), ylim + ylim*0.2, 'Founding of PRC\n(1949)', color_mapping['CHN'], 1.07),
Event((1958, 1962), ylim + ylim*0.35, 'Great Leap Forward\n(1958-1962)', 'tab:orange', 1.13),
Event((1978, 1979), ylim + ylim*0.5, 'Reform and Opening-up\n(1978-1979)', 'tab:blue', 1.18)
]

# Draw events
draw_events(events, ax)
plt.show()
```

+++ {"user_expressions": []}

(TODO: Finalize trend)
Trends to note:
- The impact of trade policy (Navigation Act)
- The productivity change created by the industrial revolution
- US surpasses UK -- any specific event?
- Wars and business cycles (link to business cycles lecture)

```{code-cell} ipython3
# Create the plot
fig, ax = plt.subplots(dpi=300)

cntry = ['GBR', 'USA']
ax = draw_interp_plots(gdppc[cntry].loc[1500:2000],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, True, ax)

ylim = ax.get_ylim()[1]

# Create a list of data points=
events = [
    Event((1651, 1651), ylim + ylim*0.1, 'Navigation Act (UK)\n(1651)', 'tab:orange', 1),
    Event((1788, 1790), ylim + ylim*0.4, 'Federation (US)\n(1789)', color_mapping['USA'], 1.055),
    Event((1760, 1840), ylim + ylim*0.1, 'Industrial Revolution\n(1760-1840)', 'grey', 1),
    Event((1848, 1850), ylim + ylim*0.6, 'Repeal of Navigation Act (UK)\n(1849)', 'tab:blue', 1.085),
    Event((1861, 1865), ylim + ylim*1, 'American Civil War (US)\n(1861-1865)', color_mapping['USA'], 1.14),
    Event((1914, 1918), ylim + ylim*0.1, 'WW 1\n(1914-1918)', 'tab:red', 1),
    Event((1929, 1939), ylim + ylim*0.4, 'the Great Depression\n(1929–1939)', 'grey', 1.06),
    Event((1939, 1945), ylim + ylim*0.8, 'WW 2\n(1939-1945)', 'tab:red', 1.11)
]

# Draw events
draw_events(events, ax)
plt.show()
```

## The Industrialized World

(TODO: Write description for this section)

Now we can look at total Gross Domestic Product (GDP) rather than focusing on GDP per capita (as a proxy for living standards).

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data.set_index(['countrycode', 'year'], inplace=True)
data['gdp'] = data['gdppc'] * data['pop']
gdp = data['gdp'].unstack('countrycode')
```

+++ {"user_expressions": []}

### Early Industralization (1820 to 1940)


Gross Domestic Product

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP
    name: gdp1
---
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['CHN', 'SUN', 'JPN', 'GBR', 'USA']
start_year, end_year = (1820, 1940)
ax = draw_interp_plots(gdp[cntry].loc[start_year:end_year],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
```

+++ {"user_expressions": []}

GDP per Capita

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita
    name: gdppc1
---
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['CHN', 'SUN', 'JPN', 'GBR', 'USA']
start_year, end_year = (1820, 1940)
ax = draw_interp_plots(gdppc[cntry].loc[start_year:end_year],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
```

+++ {"user_expressions": []}

## The Modern Era (1970 to 2018)

Gross Domestic Product (GDP)

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP
    name: gdp2
---
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['CHN', 'SUN', 'JPN', 'USA']
start_year, end_year = (1970, 2020)
ax = draw_interp_plots(gdp[cntry].loc[start_year:end_year],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
```

+++ {"user_expressions": []}

GDP per Capita

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita
    name: gdppc2
---
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['CHN', 'SUN', 'JPN', 'USA']
start_year, end_year = (1970, 2020)
ax = draw_interp_plots(gdppc[cntry].loc[start_year:end_year],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
```

+++ {"user_expressions": []}

## Regional Analysis
(TODO: Write descriptions for this section)

The [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) dataset also includes regional aggregations

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Regional data', header=(0,1,2), index_col=0)
data.columns = data.columns.droplevel(level=2)
```

+++ {"user_expressions": []}

We can save the raw data in a more convenient format to build a single table of regional GDP per capita

```{code-cell} ipython3
regionalgdppc = data['gdppc_2011'].copy()
regionalgdppc.index = pd.to_datetime(regionalgdppc.index, format='%Y')
```

+++ {"user_expressions": []}

Let us interpolate based on time to fill in any gaps in the dataset for the purpose of plotting

```{code-cell} ipython3
regionalgdppc.interpolate(method='time', inplace=True)
```

+++ {"user_expressions": []}

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

+++ {"user_expressions": []}

Looking more closely, let us compare the time series for `Western Offshoots` and `Sub-Saharan Africa`

+++ {"user_expressions": []}

and more broadly at a number of different regions around the world

```{code-cell} ipython3
fig = plt.figure(dpi=300)
ax = fig.gca()
line_styles = ['-', '--', ':', '-.', '.', 'o', '-', '--', '-']
ax = regionalgdppc.plot(ax = ax, style=line_styles)
ax.set_yscale('log')
plt.legend(loc='lower center', ncol=3, bbox_to_anchor=[0.5, -0.4])
plt.show()
```
