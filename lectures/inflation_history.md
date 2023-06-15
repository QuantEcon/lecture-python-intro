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

# Price Level Histories 

## Ends of Four Big Inflations



We want to construct four graphs from "The Ends of Four Big Inflations" from chapter 3 of {cite}`

This is chapter 3 of the book **Rational Expectations and Inflation** that I shared with you a couple of days ago.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
```

```python
!pip install xlrd
```

The graphs depict logarithms of price levels during the early post World War I years for four countries:

 * Figure 3.1, Retail prices Austria 1921-1924 (page 42)
 * Figure 3.2, Wholesale prices Hungary, 1921-1924 (page 43)
 * Figure 3.3, Wholesale prices, Poland, 1921-1924 (page 44)
 * Figure pd.dataframe(3.4, Wholesale prices, Germany, 1919-1924 (page 45)

Data underlying these graphs appear in the tables in the appendix; all of the data have been organized into a spreadsheet *chapter_3.xls* that I include in our dropbox folder.

The spreadsheet refers to the tables that I now describe for each of our four countries.


## Tweak request for Jiacheng

* for each of the four countries, please delete the graphs of the "money supplies" and "real balances" in the right panels. So now we'll just have your excellent graphs of the price level and exchange rate

* for each graph please add a caption below the graph just saying the country name. Maybe Zejin can quickly tell you how to do this.

* for Poland I think that there are some additional price level data in the table, but their units have changed.  Please take a look at figure 3.3. in the chapter.  I am pretty sure that to draw that graph I just made a guess about the units change -- sort of using "continuity" -- and adjusted the units of the second series and spliced the series to get the one in figure 3.3.  We can talk about this if you wish

* for Germany, I'd like to do another "splicing operation" to avoid the big jump down in the price series and the exchange rate series.  What those jumps reflect is the "units change" associated with the "currency reform". It was a pure units change.
  
    * Here is one idea -- make two versions of the graph in two separate graphs. The first version is what you have
    
     * The second version does the "splicing" by converting the new units to the old so that there is no drop.  Then the graphs will be "continuous.
     
     * By comparing the graphs we can teach about the "currency reform"
     
     

* Comment: I really like the way you put the exchange rate and the price level on the same graph for each country. Overall the graphs are great -- really exciting to me! And I love the "long series" graph at the end.

* Thanks so much.



```python
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

def create_plot(p_seq, e_seq, index, labs, ax):
    
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
```

```python
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

### Austria

* Table 3.2, money supply, $\exp M$
* Table 3.3, rdf_Aus.indexetail prices, $\exp p$
* Table 3.4, exchange rate with US

```python
df_Aus.head(5)
```

```python
p_seq = df_Aus['Retail price index, 52 commodities']
e_seq = df_Aus['Exchange Rate']

lab = ['Retail Price Index', 'Exchange Rate']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_plot(p_seq, e_seq, df_Aus.index, lab, ax)

# connect disjunct parts
plt.figtext(0.5, 0.0, 'Austria', horizontalalignment='center', fontsize=12)
plt.show()
```

### Hungary

* Table 3.9, money supply, $\exp M$
* Table 3.10, price level $\exp p$ and exchange rate


```python
df_Hung.head(5)
```

```python
m_seq = df_Hung['Notes in circulation']
p_seq = df_Hung['Hungarian index of prices']
e_seq = 1/df_Hung['Cents per crown in New York']
rb_seq = np.log(m_seq) - np.log(p_seq)

lab = ['Hungarian Index of Prices', '1/Cents per Crown in New York']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = create_plot(p_seq, e_seq, df_Hung.index, lab, ax)

plt.figtext(0.5, 0.0, 'Hungary', horizontalalignment='center', fontsize=12)
plt.show()
```

### Poland

* Table 3.14, money supply, $\exp M$
* Table 3.15, price level $\exp p$ 
* Table 3.15, exchange rate


Jiacheng: 

I spliced the three series - Wholesale price index, Wholesale Price Index: On paper currency basis, and Wholesale Price Index: On zloty basis. I made the adjustment by adjusting the sequence based on the price level ratio at the last period of the available previous series and glue them to a single series.

I dropped the exchange rate after June 1924, when zloty was adopted, because we don't have the price measured in zloty and old currency in June to compute the exchange rate adjustment.

```python
df_Pol.head(5)
```

```python
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

```python
lab = ['Wholesale Price Index', '1/Cents per Polish Mark']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
ax1 = create_plot(p_seq, e_seq, df_Pol.index, lab, ax)

plt.figtext(0.5, 0.0, 'Poland', horizontalalignment='center', fontsize=12)
plt.show()
```

### Germany

* Table 3.21, money supply, $\exp M$ (last column)
* Table 3.18, wholesale price level $\exp p$ 
* Table 3.19, exchange rate

```python
df_Germ.head(5)
```


```python
p_seq = df_Germ['Price index (on basis of marks before July 1924,  reichsmarks after)'].copy()
e_seq = 1/df_Germ['Cents per mark']

lab = ['Price Index', '1/Cents per Mark']

# create plot
fig, ax = plt.subplots(figsize=[9,5], dpi=200)
ax1 = create_plot(p_seq, e_seq, df_Germ.index, lab, ax)

plt.figtext(0.5, 0.0, 'Germany', horizontalalignment='center', fontsize=12)
plt.show()
```

Jiacheng: I add the new graph here.

```python
p_seq = df_Germ['Price index (on basis of marks before July 1924,  reichsmarks after)'].copy()
e_seq = 1/df_Germ['Cents per mark'].copy()

# adjust the price level/exchange rate after the currency reform
p_seq[p_seq.index > '06-01-1924'] = p_seq[p_seq.index > '06-01-1924'] * 1e12
e_seq[e_seq.index > '12-01-1923'] = e_seq[e_seq.index > '12-01-1923'] * 1e12

lab = ['Price Index (Marks or converted to Marks)', '1/Cents per Mark (or Reichsmark converted to Mark)']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
ax1 = create_plot(p_seq, e_seq, df_Germ.index, lab, ax)

plt.figtext(0.5, 0.0, 'Germany', horizontalalignment='center', fontsize=12)
plt.show()
```

**Note to Jiacheng:** 

There might be some ambiguity about exactly which column in the "balance sheets" of the central bank that we want to interpret as "money".  Typically it will be something like "notes" or "total notes" on the liability sides of the balance sheets in the spreadsheet table.  We can resolve uncertainties in your mind quickly with a meeting.

**First Steps:** What I'd like you to do as  first is to use matplotlib in a Jupyter notebook to take logs of the price level and reproduce pretty versions of our four tables.

**Seecond Steps:** There are some fun additonal things we can plot to set the stage for our  cagan_ree and cagan_adaptive notebooks.  For example, we have the data to plot logs of real balances around the times of the stabilizations. We can hunt for instances of "velocity dividends".


```python
# import data
df_fig5 = pd.read_excel('datasets/longprices.xls', sheet_name='all', header=2, index_col=0).iloc[1:]
df_fig5.index = df_fig5.index.astype(int) 

df_fig5.head(5)
```

```python
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
plt.show()
```
