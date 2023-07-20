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

# Economic Growth Evidence

## Overview

Adam Tooze's account of the geopolitical precedents and antecedents of World War I includes a comparison of how Gross National Products of European Great Powers had evolved during the 70 years preceding 1914 (see chapter 1 of {cite}`Tooze_2014`).

```{figure} _static/lecture_specific/long_run_growth/rgdp-2011-y1820to1945-leadingeconomies.png
:width: 75%
```

We construct a version of Tooze's graph later in this lecture.

(An impatient reader can jump ahead and look at figure {numref}`gdp1`.)

Looking at his graph and how it set the geopolitical stage for "the American (20th) century" naturally 
tempts one to want a counterpart to his graph for 2014 or later.

(An impatient reader might now want to jump ahead and look at figure {numref}`gdp2`.)

As we'll see, reasoning  by analogy, this graph perhaps set the stage for an "XXX (21st) century", where you are free to fill in your guess for country XXX.

As we gather data to construct those two graphs, we'll also study growth experiences for a number of countries for time horizons extending as far back as possible.

These graphs will portray how the "Industrial Revolution" began in Britain in the late 18th century, then migrated to one country after another.  

In a nutshell, this lecture records growth trajectories of various countries over long time periods. 

While some countries have experienced long term rapid growth across that has lasted a hundred years, others have not. 

Since populations differ across countries and vary within a country over time, it will
be interesting to describe both total GDP and GDP per capita as it evolves within a country.

First let's import the packages needed to explore what the data says about long run growth

```{code-cell} ipython3
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import namedtuple
from matplotlib.lines import Line2D
```

## Setting up

A project initiated by [Angus Maddison](https://en.wikipedia.org/wiki/Angus_Maddison) has collected many historical time series related to economic growth,
some dating back to the first century.

The data can be downloaded from the [Maddison Historical Statistics webpage](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) by clicking on the "Latest Maddison Project Release". 

For convenience, here is a copy of the {download}`2020 data <datasets/mpd2020.xlsx>` in `Excel` format.

Let's read it into a pandas dataframe:

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data
```

We can see that this dataset contains GDP per capita (gdppc) and population (pop) for many countries and years.

Let's look at how many and which countries are available in this dataset

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

Let's now reshape the original data into some convenient variables to enable quicker access to countries time series data.

We can build a useful mapping between country codes and country names in this dataset

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

We create a color mapping between country codes and colors for consistency

```{code-cell} ipython3
:tags: [hide-input]

country_names = data['countrycode']

# Generate a colormap with the number of colors matching the number of countries
colors = cm.Dark2(np.linspace(0, 0.8, len(country_names)))

# Create a dictionary to map each country to its corresponding color
color_mapping = {country: color for country, color in zip(country_names, colors)}
```

## GPD plots

Looking at the United Kingdom we can first confirm we are using the correct country code

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita (GBR)
    name: gdppc_gbr1
    width: 500px
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

:::{note}
[International Dollars](https://en.wikipedia.org/wiki/International_dollar) are a hypothetical unit of currency that has the same purchasing power parity that the U.S. Dollar has in the United States at any given time. They are also known as Geary–Khamis dollars (GK Dollars).
:::

We can see that the data is non-continuous for longer periods in the early 250 years of this millennium, so we could choose to interpolate to get a continuous line plot.

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

We can now put this into a function to generate plots for a list of countries

```{code-cell} ipython3
def draw_interp_plots(series, ylabel, xlabel, color_mapping, code_to_name, lw, logscale, ax):

    for i, c in enumerate(cntry):
        # Get the interpolated data
        df_interpolated = series[c].interpolate(limit_area='inside')
        interpolated_data = df_interpolated[series[c].isnull()]

        # Plot the interpolated data with dashed lines
        ax.plot(interpolated_data,
                linestyle='--',
                lw=lw,
                alpha=0.7,
                color=color_mapping[c])

        # Plot the non-interpolated data with solid lines
        ax.plot(series[c],
                linestyle='-',
                lw=lw,
                color=color_mapping[c],
                alpha=0.8,
                label=code_to_name.loc[c]['country'])
        
        if logscale == True:
            ax.set_yscale('log')
    
    # Draw the legend outside the plot
    ax.legend(loc='lower center', ncol=5, bbox_to_anchor=[0.5, -0.25])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    return ax
```

As you can see from this chart, economic growth started in earnest in the 18th century and continued for the next two hundred years. 

How does this compare with other countries' growth trajectories? 

Let's look at the United States (USA), United Kingdom (GBR), and China (CHN)

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita, 1500- (China, UK, USA)
    name: gdppc_comparison
tags: [hide-input]
---
# Define the namedtuple for the events
Event = namedtuple('Event', ['year_range', 'y_text', 'text', 'color', 'ymax'])

fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

cntry = ['CHN', 'GBR', 'USA']
ax = draw_interp_plots(gdppc[cntry].loc[1500:],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)

# Define the parameters for the events and the text
ylim = ax.get_ylim()[1]
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'fontsize': 9, 
            'va':'center', 'ha':'center'}

# Create a list of events to annotate
events = [
    Event((1650, 1652), ylim + ylim*0.04, 
          'the Navigation Act\n(1651)',
          color_mapping['GBR'], 1),
    Event((1655, 1684), ylim + ylim*0.13, 
          'Closed-door Policy\n(1655-1684)', 
          color_mapping['CHN'], 1.1),
    Event((1848, 1850), ylim + ylim*0.22,
          'the Repeal of Navigation Act\n(1849)', 
          color_mapping['GBR'], 1.18),
    Event((1765, 1791), ylim + ylim*0.04, 
          'American Revolution\n(1765-1791)', 
          color_mapping['USA'], 1),
    Event((1760, 1840), ylim + ylim*0.13, 
          'Industrial Revolution\n(1760-1840)', 
          'grey', 1.1),
    Event((1929, 1939), ylim + ylim*0.04, 
          'the Great Depression\n(1929–1939)', 
          'grey', 1),
    Event((1978, 1979), ylim + ylim*0.13, 
          'Reform and Opening-up\n(1978-1979)', 
          color_mapping['CHN'], 1.1)
]

def draw_events(events, ax):
    # Iterate over events and add annotations and vertical lines
    for event in events:
        event_mid = sum(event.year_range)/2
        ax.text(event_mid, 
                event.y_text, event.text, 
                color=event.color, **t_params)
        ax.axvspan(*event.year_range, color=event.color, alpha=0.2)
        ax.axvline(event_mid, ymin=1, 
        ymax=event.ymax, color=event.color, 
        linestyle='-', clip_on=False, alpha=0.15)
        
# Draw events
draw_events(events, ax)
plt.show()
```

The preceding graph of per capita GDP strikingly reveals how the spread of the industrial revolution has over time gradually lifted the living standards of substantial
groups of people  

- most of the growth happened in the past 150 years after the industrial revolution.
- per capita GDP in the US and UK rose and diverged from that of China from 1820 to 1940.
- the gap has closed rapidly after 1950 and especially after the late 1970s.
- these outcomes reflect complicated combinations of technological and economic-policy factors that students of economic growth try to understand and quantify.


It is fascinating to see China's GDP per capita levels from 1500 through to the 1970s.

Notice the long period of declining GDP per capital levels from the 1700s until the early 20th century.

Thus, the graph indicates 

- a long economic downturn and stagnation after the Closed-door Policy by the Qing government.
- China's very different experience than the UK's after the onset of the industrial revolution in the UK.
- how the Self-Strengthening Movement seemed mostly to help China to grow.
- how stunning have been the growth achievements of modern Chinese economic policies by the PRC that culminated with its late 1970s reform and liberalization.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita, 1500-2000 (China)
    name: gdppc_china
tags: [hide-input]
---
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

cntry = ['CHN']
ax = draw_interp_plots(gdppc[cntry].loc[1600:2000],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, True, ax)

ylim = ax.get_ylim()[1]

events = [
Event((1655, 1684), ylim + ylim*0.06, 
      'Closed-door Policy\n(1655-1684)', 
      'tab:orange', 1),
Event((1760, 1840), ylim + ylim*0.06, 
      'Industrial Revolution\n(1760-1840)', 
      'grey', 1),
Event((1839, 1842), ylim + ylim*0.2, 
      'First Opium War\n(1839–1842)', 
      'tab:red', 1.07),
Event((1861, 1895), ylim + ylim*0.4, 
      'Self-Strengthening Movement\n(1861–1895)', 
      'tab:blue', 1.14),
Event((1939, 1945), ylim + ylim*0.06, 
      'WW 2\n(1939-1945)', 
      'tab:red', 1),
Event((1948, 1950), ylim + ylim*0.23, 
      'Founding of PRC\n(1949)', 
      color_mapping['CHN'], 1.08),
Event((1958, 1962), ylim + ylim*0.5, 
      'Great Leap Forward\n(1958-1962)', 
      'tab:orange', 1.18),
Event((1978, 1979), ylim + ylim*0.7, 
      'Reform and Opening-up\n(1978-1979)', 
      'tab:blue', 1.24)
]

# Draw events
draw_events(events, ax)
plt.show()
```

We can also look at the United States (USA) and United Kingdom (GBR) in more detail

In the following graph, please watch for 
- impact of trade policy (Navigation Act).
- productivity changes brought by the industrial revolution.
- how the US gradually approaches and then surpasses the UK, setting the stage for the ''American Century''.
- the often unanticipated consequences of wars.
- interruptions and scars left by [business cycle](business_cycle) recessions and depressions.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita, 1500-2000 (UK and US)
    name: gdppc_ukus
tags: [hide-input]
---
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

cntry = ['GBR', 'USA']
ax = draw_interp_plots(gdppc[cntry].loc[1500:2000],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, True, ax)

ylim = ax.get_ylim()[1]

# Create a list of data points=
events = [
    Event((1651, 1651), ylim + ylim*0.15, 
          'Navigation Act (UK)\n(1651)', 
          'tab:orange', 1),
    Event((1765, 1791), ylim + ylim*0.15, 
          'American Revolution\n(1765-1791)',
          color_mapping['USA'], 1),
    Event((1760, 1840), ylim + ylim*0.6, 
          'Industrial Revolution\n(1760-1840)', 
          'grey', 1.08),
    Event((1848, 1850), ylim + ylim*1.1, 
          'Repeal of Navigation Act (UK)\n(1849)', 
          'tab:blue', 1.14),
    Event((1861, 1865), ylim + ylim*1.8, 
          'American Civil War\n(1861-1865)', 
          color_mapping['USA'], 1.21),
    Event((1914, 1918), ylim + ylim*0.15, 
          'WW 1\n(1914-1918)', 
          'tab:red', 1),
    Event((1929, 1939), ylim + ylim*0.6, 
          'the Great Depression\n(1929–1939)', 
          'grey', 1.08),
    Event((1939, 1945), ylim + ylim*1.1, 
          'WW 2\n(1939-1945)', 
          'tab:red', 1.14)
]

# Draw events
draw_events(events, ax)
plt.show()
```

## The industrialized world

Now we'll construct some graphs of interest to geopolitical historians like Adam Tooze.

We'll focus on total Gross Domestic Product (GDP) (as a proxy for ''national geopolitical-military power'') rather than focusing on GDP per capita (as a proxy for living standards).

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Full data')
data.set_index(['countrycode', 'year'], inplace=True)
data['gdp'] = data['gdppc'] * data['pop']
gdp = data['gdp'].unstack('countrycode')
```

### Early industrialization (1820 to 1940)

We first visualize the trend of China, the Former Soviet Union, Japan, the UK and the US.

The most notable trend is the rise of the US, surpassing the UK in the 1860s and China in the 1880s.

The growth continued until the large dip in the 1930s when the Great Depression hit.

Meanwhile, Russia experienced significant setbacks during World War I and recovered significantly after the February Revolution.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP in the early industrialization era
    name: gdp1
---
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['CHN', 'SUN', 'JPN', 'GBR', 'USA']
start_year, end_year = (1820, 1945)
ax = draw_interp_plots(gdp[cntry].loc[start_year:end_year],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
```

## Constructing a plot similar to Tooze's

Let's first define a collection of countries that consist of the British Empire (BEM) so we can replicate that element of Tooze's chart. 

```{code-cell} ipython3
BEM = ['GBR', 'IND', 'AUS', 'NZL', 'CAN', 'ZAF']
gdp['BEM'] = gdp[BEM].loc[start_year-1:end_year].interpolate(method='index').sum(axis=1) # Interpolate incomplete time-series
```

Let's take a look at the aggregation that represents the British Empire

```{code-cell} ipython3
gdp['BEM'].plot() # The first year is np.nan due to interpolation
```

```{code-cell} ipython3
code_to_name
```

```{code-cell} ipython3
# Define colour mapping and name for BEM
color_mapping['BEM'] = color_mapping['GBR']  # Set the color to be the same as Great Britain
# Add British Empire to code_to_name
bem = pd.DataFrame(["British Empire"], index=["BEM"], columns=['country'])
bem.index.name = 'countrycode'
code_to_name = pd.concat([code_to_name, bem])
```

```{code-cell} ipython3
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['DEU', 'USA', 'SUN', 'BEM', 'FRA', 'JPN']
start_year, end_year = (1821, 1945)
ax = draw_interp_plots(gdp[cntry].loc[start_year:end_year],
    'Real GDP in 2011 $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
plt.savefig("./_static/lecture_specific/long_run_growth/rgdp-2011-y1820to1945-leadingeconomies.png")
plt.show()
```

### The modern era (1950 to 2020)

The following graph displays how quickly China has grown, especially since the late 1970s.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP in the modern era
    name: gdp2
---
fig, ax = plt.subplots(dpi=300)
ax = fig.gca()
cntry = ['CHN', 'SUN', 'JPN', 'GBR', 'USA']
start_year, end_year = (1950, 2020)
ax = draw_interp_plots(gdp[cntry].loc[start_year:end_year],
    'International $\'s','Year',
    color_mapping, code_to_name, 2, False, ax)
```

It is tempting to compare this graph with  figure  {numref}`gdp1` that showed the US overtaking the UK near the start of the "American Century", a version of the graph featured in chapter 1 of  {cite}`Tooze_2014`.

## Regional analysis

We often want to study historical experiences of countries outside the club of "World Powers".

Fortunately, the [Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) dataset also includes regional aggregations

```{code-cell} ipython3
data = pd.read_excel("datasets/mpd2020.xlsx", sheet_name='Regional data', header=(0,1,2), index_col=0)
data.columns = data.columns.droplevel(level=2)
```

We can save the raw data in a more convenient format to build a single table of regional GDP per capita

```{code-cell} ipython3
regionalgdppc = data['gdppc_2011'].copy()
regionalgdppc.index = pd.to_datetime(regionalgdppc.index, format='%Y')
```

Let's interpolate based on time to fill in any gaps in the dataset for the purpose of plotting

```{code-cell} ipython3
regionalgdppc.interpolate(method='time', inplace=True)
```

and record a dataset of world GDP per capita

```{code-cell} ipython3
worldgdppc = regionalgdppc['World GDP pc']
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: World GDP per capita
    name: world_gdppc
---
fig = plt.figure(dpi=300)
ax = fig.gca()
ax = worldgdppc.plot(
    ax = ax,
    xlabel='Year',
    ylabel='2011 US$',
)
```

Looking more closely, let's compare the time series for `Western Offshoots` and `Sub-Saharan Africa` and more broadly at a number of different regions around the world.

Again we see the divergence of the West from the rest of the world after the industrial revolution and the convergence of the world after the 1950s

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Regional GDP per capita
    name: region_gdppc
---
fig = plt.figure(dpi=300)
ax = fig.gca()
line_styles = ['-', '--', ':', '-.', '.', 'o', '-', '--', '-']
ax = regionalgdppc.plot(ax = ax, style=line_styles)
ax.set_yscale('log')
plt.legend(loc='lower center', 
ncol=3, bbox_to_anchor=[0.5, -0.4])
plt.show()
```
