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

# Price Level Histories 

This lecture offers some scraps of historical evidence about fluctuations in  levels of aggregate price indexes.  

The rate of growth of the price level is called **inflation** in the popular press and in discussions among central bankers and treasury officials.

The price level is measured in units of domestic currency per units of a representative bundle of consumption goods.  

Thus, in the US, the price level at  $t$  is measured in dollars in month $t$ or year $t$  per unit of the consumption bundle.

Until the early 20th century, throughout much of the west, although price levels fluctuated from year to  year,
they didn't have much of a trend.  

Thus, they tended to end a century at close to a level at which they started it.

Things were different in the 20th century, as we shall see in this lecture.

This lecture will set the stage for some subsequent lectures about a particular theory that  economists use to
think about determinants of the price level.



## Four Centuries of Price Levels

We begin by displaying  some  data  that originally appeared on page 35 of {cite}`sargent2002big`.

The data  price levels for four "hard currency" countries from 1600 to 1914.

The four countries are

* France 
* Spain (Castile)
* United Kingdom
* United States

In the present context, the  phrase hard currency means that the countries were on a commodity-money standard:  money consisted of gold and silver coins that circulated at values largely determined by the weights of their gold and silver contents.

(Under a gold or silver standard, some money also consisted of "warehouse certificates" that represented paper claims on gold or silver coins. Bank notes issued by the government or private banks can be viewed as examples of such "warehouse certificate".)

The data we want to study data  originally appeared in a graph on page 35 of {cite}`sargent2002big`.


As usual, we'll start by importing some Python modules.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
```

```{code-cell} ipython3
:tags: [hide-output]

!pip install xlrd 
```

We'll start by bringing these data into Pandas from a spreadsheet.

```{code-cell} ipython3
# import data
df_fig5 = pd.read_excel('datasets/longprices.xls', sheet_name='all', header=2, index_col=0).iloc[1:]
df_fig5.index = df_fig5.index.astype(int) 

df_fig5.head(5)
```

We first plot price levels over the period 1600-1914.

During most years in this time interval, the countries were on a gold or silver standard.

```{code-cell} ipython3
df_fig5_bef1914 = df_fig5[df_fig5.index <= 1915]

# create plot
cols = ['UK', 'US', 'France', 'Castile']

fig, ax = plt.subplots(1, 1, figsize=[8, 5], dpi=200)

for col in cols:
    ax.plot(df_fig5_bef1914.index, df_fig5_bef1914[col], label=col)

ax.spines[['right', 'top']].set_visible(False)
ax.legend()
ax.set_ylabel('Index  1913 = 100')
ax.set_xlim(xmin=1600)
plt.tight_layout()
fig.text(.5, .0001, "Price Levels", ha='center')
plt.show()
```

We say "most years" because there were temporary lapses from the gold or silver standard.

By staring at the graph carefully, you might be able to guess when these temporary lapses occurred, because they were also times during which price levels rose markedly from what had been  average values during more typical years.

 * 1791-1797 in France (the French Revolution)
 * 1776-1793 in the US (the US War for Independence from Great Britain)
 * 1861-1865 in the US (the US Civil War)

During each of these episodes, the gold/silver standard was temporarily abandoned as a government printed paper money to help it finance war expenditures.

Despite these temporary lapses, a striking thing about the figure is that price levels hovered around roughly constant long-term levels for over three centuries.  

Two other features of the figure attracted the attention of leading economists such as Irving Fisher of Yale University and John Maynard Keynes of Cambridge University in the early century.

  * There was considerable year-to-year instability of the price levels despite their long begin anchored to the same average level in the long term
  
  * While using valuable gold and silver as coins was a time-tested way to anchor  the price level by limiting the supply of money, it cost real resources.
     
      * that is, society paid a high "opportunity cost" for using gold and silver as coins; gold and silver could instead be used as valuable jewelry and also as an industrial input.

Keynes and Fisher proposed what they suggested would be  a socially more  efficient way to achieve a price level that  would be at least as firmly  anchored, and would also exhibit less  year-to-year short-term fluctuations.  

In particular, they argued that a well-managed central bank could achieve price level stability by

  * issuing a **limited supply** of paper currency
  * guaranteeing that it would  not  print money to finance government expenditures

Thus, the waste from using gold and silver as coins  prompted John Maynard Keynes to call a commodity standard a “barbarous relic.”

A paper fiat money system disposes of all reserves behind a currency. 

But notice that in doing so, it also eliminates an automatic supply mechanism constraining the price level.

A low-inflation paper fiat money system replaces that automatic mechanism with an enlightened government that commits itself to limiting the quantity of a pure token, no-cost currency.

Now let's see what happened to the price level in our four countries when after 1914 one after another of them 
left the gold/silver standard.

We'll show a version of the complete  graph that originally appeared on page 35 of {cite}`sargent2002big`.

The graph shows logarithms of price levels our  four "hard currency" countries from 1600 to 2000.

Allthough we didn't have  to use   logarithms in our earlier graphs that  had stopped in 1914 -- we use logarithms now because we want also  to fit observations after 1914 in the same graph as the earlier observations.

All four of the countries eventually permanently left the gold standard by modifying their monetary and fiscal policies in several ways, starting the outbreak of the Great War in 1914.

```{code-cell} ipython3
# create plot
cols = ['UK', 'US', 'France', 'Castile']

fig, ax = plt.subplots(1, 1, figsize=[8, 5], dpi=200)

for col in cols:
    ax.plot(df_fig5.index, df_fig5[col])
    ax.text(x=df_fig5.index[-1]+2, y=df_fig5[col].iloc[-1], s=col)

ax.spines[['right', 'top']].set_visible(False)
ax.set_yscale('log')
ax.set_ylabel('Index  1913 = 100')
ax.set_xlim(xmin=1600)
ax.set_ylim([10, 1e6])
plt.tight_layout()
fig.text(.5, .0001, "Logs of Price Levels", ha='center')
plt.show()
```

The graph shows that achieving a price level system with a well-managed paper money system proved to be more challenging  than Irving Fisher and Keynes perhaps imagined.

Actually, earlier economists and statesmen knew about the possibility of fiat money systems long before
Keynes and Fisher advocated them in the early 20th century.

It was because earlier  proponents of a commodity money system did not trust governments properly to manage a fiat money system that they were willing to pay the resource costs associated with setting up and maintaining a commodity money system.

In light of the high inflation episodes that many countries experienced in the twentieth century after they abandoned commodity monies,  it is difficult to criticize them for their preference to stay on the pre-1914 gold/silver standard. 

The breadth and length of the inflationary experiences of the twentieth century, the century of paper money, are  historically unprecedented.

## Ends of Four Big Inflations

In the wake of World War I, which ended in November 1918, monetary and fiscal authorities struggled to achieve   price level stability without being on a gold or silver standard.

We present  four  graphs from "The Ends of Four Big Inflations" from chapter 3 of {cite}`sargent2013rational`.

The graphs depict logarithms of price levels during the early post World War I years for four countries:

 * Figure 3.1, Retail prices Austria, 1921-1924 (page 42)
 * Figure 3.2, Wholesale prices Hungary, 1921-1924 (page 43)
 * Figure 3.3, Wholesale prices, Poland, 1921-1924 (page 44)
 * Figure 3.4, Wholesale prices, Germany, 1919-1924 (page 45)

We have added logarithms of the exchange rates vis a vis the US dollar to each of the four graphs
from chapter 3 of {cite}`sargent2013rational`.

Data underlying our graphs appear in tables in an appendix to chapter 3 of {cite}`sargent2013rational`.
We have transcribed all of these data  into a spreadsheet `chapter_3.xls` that we  read into Pandas.

```{code-cell} ipython3
:tags: [hide-input]

def process_entry(entry):
    "Clean each entry of a dataframe."
    
    if type(entry) == str:
        # remove leading and trailing whitespace
        entry = entry.strip()
        # remove comma
        entry = entry.replace(',', '')
    
        # remove HTML markers
        item_to_remove = ['<s>a</s>', '<s>c</s>', '<s>d</s>', '<s>e</s>']

        # <s>b</s> represents a billion
        if '<s>b</s>' in entry:
            entry = entry.replace('<s>b</s>', '')
            entry = float(entry) * 1e9
        else:
            for item in item_to_remove:
                if item in entry:
                    entry = entry.replace(item, '')
    return entry

def process_df(df):
    "Clean and reorganize the entire dataframe."
    
    # remove HTML markers from column names
    for item in ['<s>a</s>', '<s>c</s>', '<s>d</s>', '<s>e</s>']:
        df.columns = df.columns.str.replace(item, '')
    
    df['Year'] = df['Year'].apply(lambda x: int(x))
    
    # set index to date time
    df = df.set_index(
            pd.to_datetime((df['Year'].astype(str) + df['Month'].astype(str)), format='%Y%B'))
    df = df.drop(['Year', 'Month'], axis=1)
    
    # handle duplicates by keeping the first
    df = df[~df.index.duplicated(keep='first')]
    
    # convert to numeric
    df = df.applymap(lambda x: float(x) if x != '—' else np.nan)
    
    # finally, we only focus on data between 1919 and 1925
    mask = (df.index >= '1919-01-01') & (df.index < '1925-01-01')
    df = df.loc[mask]

    return df

def create_pe_plot(p_seq, e_seq, index, labs, ax):
    
    p_lab, e_lab = labs
    
    # price and exchange rates
    ax.plot(index, p_seq, label=p_lab, color='tab:blue')
    ax1 = ax.twinx()
    ax1.plot([None], [None], label=p_lab, color='tab:blue')
    ax1.plot(index, e_seq, label=e_lab, color='tab:orange')
    ax.set_yscale('log')
    ax1.set_yscale('log')
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    ax.text(-0.08, 1.03, 'Price Level', transform=ax.transAxes)
    ax.text(0.92, 1.03, 'Exchange Rate', transform=ax.transAxes)
    
    ax1.legend(loc='upper left')

    return ax1

def create_pr_plot(p_seq, index, ax):

    # Calculate the difference of log p_seq
    log_diff_p_seq = np.diff(np.log(p_seq))
    
    # Graph for the difference of log p_seq
    ax.scatter(index[1:], log_diff_p_seq, label='Monthly Inflation Rate', color='tab:grey')
    diff_smooth = pd.DataFrame(log_diff_p_seq).rolling(3).mean()
    ax.plot(index[1:], diff_smooth, alpha=0.5, color='tab:grey')
    ax.text(-0.08, 1.03, 'Monthly Inflation Rate', transform=ax.transAxes)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    ax.legend(loc='upper left')
    
    return ax
    
```

```{code-cell} ipython3
# import data
xls = pd.ExcelFile('datasets/chapter_3.xlsx')

# unpack and combine all series
sheet_index = [(2, 3, 4), (9, 10), (14, 15, 16), (21, 18, 19)]
remove_row = [(-2, -2, -2), (-7, -10), (-6, -4, -3), (-19, -3, -6)]

df_list = []

for i in range(4):
    
    indices, rows = sheet_index[i], remove_row[i]
    sheet_list = [pd.read_excel(xls, 'Table3.' + str(ind), header=1).iloc[:row].applymap(process_entry)
                  for ind, row in zip(indices, rows)]
    
    sheet_list = [process_df(df) for df in sheet_list]
    df_list.append(pd.concat(sheet_list, axis=1))

df_Aus, df_Hung, df_Pol, df_Germ = df_list
```

Let's dive in and construct graphs for our four countries.

For each country, we'll plot two graphs.

The first graph plots logarithms of 

  * price levels
  * exchange rates vis a vis US dollars

For each country, the scale on the right side of a graph will pertain to the price level while the scale on the left side of a graph will pertain
to the exchange rate. 

For each country, the second graph plots a three-month moving average of the inflation rate defined as $p_t - p_{t-1}$.

### Austria

The sources of our data are:


* Table 3.3, $\exp p$
* Table 3.4, exchange rate with US

```{code-cell} ipython3
df_Aus.head(5)
```

```{code-cell} ipython3
p_seq = df_Aus['Retail price index, 52 commodities']
e_seq = df_Aus['Exchange Rate']

lab = ['Retail Price Index', 'Exchange Rate']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_pe_plot(p_seq, e_seq, df_Aus.index, lab, ax)

# connect disjunct parts
plt.figtext(0.5, -0.02, 'Austria', horizontalalignment='center', fontsize=12)
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_pr_plot(p_seq, df_Aus.index, ax)

plt.figtext(0.5, -0.02, 'Austria', horizontalalignment='center', fontsize=12)
plt.show()
```

Staring at the above  graphs conveys the following impressions to the authors of this lecture at quantecon.

 * an episode of  "hyperinflation" with  rapidly rising log price level and very high monthly inflation rates
 * a sudden stop of the hyperinflation as indicated by the abrupt flattening of the log price level and a marked permanent drop in the three-month average of inflation
 * a US dollar exchange rate that shadows the price level.  
  
We'll see similar patterns in the next three episodes  that we'll study now.

### Hungary

The source of our data for Hungary is:

* Table 3.10, price level $\exp p$ and exchange rate

```{code-cell} ipython3
df_Hung.head(5)
```

```{code-cell} ipython3
m_seq = df_Hung['Notes in circulation']
p_seq = df_Hung['Hungarian index of prices']
e_seq = 1/df_Hung['Cents per crown in New York']
rb_seq = np.log(m_seq) - np.log(p_seq)

lab = ['Hungarian Index of Prices', '1/Cents per Crown in New York']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_pe_plot(p_seq, e_seq, df_Hung.index, lab, ax)

plt.figtext(0.5, -0.02, 'Hungary', horizontalalignment='center', fontsize=12)
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_pr_plot(p_seq, df_Hung.index, ax)

plt.figtext(0.5, -0.02, 'Hungary', horizontalalignment='center', fontsize=12)
plt.show()
```

### Poland

The sources of our data for Poland are:

* Table 3.15, price level $\exp p$ 
* Table 3.15, exchange rate


````{note}
To construct the price level series from the data in the spreadsheet, we instructed  Pandas to follow the same procedures implemented in chapter 3 of {cite}`sargent2013rational`. We spliced together  three series - Wholesale price index, Wholesale Price Index: On paper currency basis, and Wholesale Price Index: On zloty basis. We  adjusted the sequence based on the price level ratio at the last period of the available previous series and glued them  to construct a single series.
We dropped the exchange rate after June 1924, when the  zloty was adopted. We did this  because we don't have the price measured in zloty. We used the  old currency in June to compute the exchange rate adjustment.
````

```{code-cell} ipython3
df_Pol.head(5)
```

```{code-cell} ipython3
# splice three price series in different units
p_seq1 = df_Pol['Wholesale price index'].copy()
p_seq2 = df_Pol['Wholesale Price Index: On paper currency basis'].copy()
p_seq3 = df_Pol['Wholesale Price Index: On zloty basis'].copy()

# non-nan part
ch_index_1 = p_seq1[~p_seq1.isna()].index[-1]
ch_index_2 = p_seq2[~p_seq2.isna()].index[-2]

adj_ratio12 = p_seq1[ch_index_1]/p_seq2[ch_index_1]
adj_ratio23 = p_seq2[ch_index_2]/p_seq3[ch_index_2]

# glue three series
p_seq = pd.concat([p_seq1[:ch_index_1], 
                   adj_ratio12 * p_seq2[ch_index_1:ch_index_2], 
                   adj_ratio23 * p_seq3[ch_index_2:]])
p_seq = p_seq[~p_seq.index.duplicated(keep='first')]

# exchange rate
e_seq = 1/df_Pol['Cents per Polish mark (zloty after May 1924)']
e_seq[e_seq.index > '05-01-1924'] = np.nan
```

```{code-cell} ipython3
lab = ['Wholesale Price Index', '1/Cents per Polish Mark']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
ax1 = create_pe_plot(p_seq, e_seq, df_Pol.index, lab, ax)

plt.figtext(0.5, -0.02, 'Poland', horizontalalignment='center', fontsize=12)
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_pr_plot(p_seq, df_Pol.index, ax)

plt.figtext(0.5, -0.02, 'Poland', horizontalalignment='center', fontsize=12)
plt.show()
```

### Germany

The sources of our data for Germany are the following tables from chapter 3 of {cite}`sargent2013rational`:

* Table 3.18, wholesale price level $\exp p$ 
* Table 3.19, exchange rate

```{code-cell} ipython3
df_Germ.head(5)
```

```{code-cell} ipython3
p_seq = df_Germ['Price index (on basis of marks before July 1924,  reichsmarks after)'].copy()
e_seq = 1/df_Germ['Cents per mark']

lab = ['Price Index', '1/Cents per Mark']

# create plot
fig, ax = plt.subplots(figsize=[9,5], dpi=200)
ax1 = create_pe_plot(p_seq, e_seq, df_Germ.index, lab, ax)

plt.figtext(0.5, -0.06, 'Germany', horizontalalignment='center', fontsize=12)
plt.show()
```

```{code-cell} ipython3
p_seq = df_Germ['Price index (on basis of marks before July 1924,  reichsmarks after)'].copy()
e_seq = 1/df_Germ['Cents per mark'].copy()

# adjust the price level/exchange rate after the currency reform
p_seq[p_seq.index > '06-01-1924'] = p_seq[p_seq.index > '06-01-1924'] * 1e12
e_seq[e_seq.index > '12-01-1923'] = e_seq[e_seq.index > '12-01-1923'] * 1e12

lab = ['Price Index (Marks or converted to Marks)', '1/Cents per Mark (or Reichsmark converted to Mark)']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
ax1 = create_pe_plot(p_seq, e_seq, df_Germ.index, lab, ax)

plt.figtext(0.5, -0.02, 'Germany', horizontalalignment='center', fontsize=12)
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_pr_plot(p_seq, df_Germ.index, ax)

plt.figtext(0.5, -0.02, 'Germany', horizontalalignment='center', fontsize=12)
plt.show()
```

## Starting and Stopping Big Inflations

A striking thing about our four graphs is how **quickly** the (log) price levels in Austria, Hungary, Poland,
and Germany leveled off after having been rising so quickly.

These "sudden stops" are also revealed by the permanent drops in three-month moving averages of inflation for the four countries.

In addition, the US dollar exchange rates for each of the four countries shadowed their price levels. 

  * This pattern is an instance of a force modeled in the **purchasing power parity** theory of exchange rates.

Each of these big inflations seemed to have "stopped on a dime".

Chapter 3 of {cite}`sargent2002big` attempts to offer an explanation for this remarkable pattern.

In a nutshell, here is his story.

After World War I, the United States was on the gold standard. The US government stood ready to convert a dollar into a specified amount of gold on demand. To understate things, immediately after the war, Hungary, Austria, Poland, and Germany were not on the gold standard. 

In practice, their currencies were largely “fiat” or "unbacked",  meaning that they were not backed by credible government promises to convert them into gold or silver coins on demand. The governments of these countries resorted to the printing of new unbacked money to finance government deficits. (The notes were "backed" mainly by treasury bills that, in those times, could not be expected to be paid off by levying taxes, but only by printing more notes or treasury bills.) This was done on such a scale that it led to a depreciation of the currencies of spectacular proportions. In the end, the German mark stabilized at 1 trillion ($10^{12}$) paper marks to the prewar gold mark, the Polish mark at 1.8 million paper marks to the gold zloty, the Austrian crown at 14,400 paper crowns to the prewar Austro-Hungarian crown, and the Hungarian krone at 14,500 paper crowns to the prewar Austro-Hungarian crown.

Chapter 3 of {cite}`sargent2002big`  focuses on the deliberate changes in policy that Hungary, Austria, Poland, and Germany made to end their hyperinflations.
The hyperinflations were each ended by restoring or virtually restoring convertibility to the dollar or equivalently to gold.

The story told in {cite}`sargent2002big` is grounded in a "monetarist theory of the price level" described in {doc}`this lecture <cagan_ree>` and further discussed in 
{doc}`this lecture <cagan_adaptive>`.

Those lectures discuss theories about what holders of those rapidly depreciating currencies were thinking about them and how that shaped responses of inflation to government policies.
