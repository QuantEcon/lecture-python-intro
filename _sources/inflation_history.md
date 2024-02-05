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

This lecture offers some  historical evidence about fluctuations in  levels of aggregate price indexes.  

The rate of growth of the price level is called **inflation** in the popular press and in discussions among central bankers and treasury officials.

The price level is measured in units of domestic currency per units of a representative bundle of consumption goods.  

Thus, in the US, the price level at  $t$  is measured in dollars in month $t$ or year $t$  per unit of the consumption bundle.

Until the early 20th century, throughout much of the west, although price levels fluctuated from year to  year, they didn't have much of a trend.  

They tended to end a century near where  they started it.

Things were different in the 20th century, as we shall see in this lecture.

We'll indicater a widely believed explanation of this big difference -- countries' abandoning gold and silver standards in early twentieth century. 

This lecture  sets the stage for some subsequent lectures about a theory that macroeconomists   economists use to think about determinants of the price level, namely, {doc}`this lecture <cagan_ree>` and 
{doc}`this lecture <cagan_adaptive>`



## Four Centuries of Price Levels

We begin by displaying    data  that originally appeared on page 35 of {cite}`sargent2002big` and that show  price levels for four "hard currency" countries from 1600 to 1914.


* France 
* Spain (Castile)
* United Kingdom
* United States

In the present context, the  phrase ''hard currency'' means that the countries were on a commodity-money standard:  money consisted of gold and silver coins that circulated at values largely determined by the weights of their gold and silver contents.

```{note}
Under a gold or silver standard, some money also consisted of "warehouse certificates" that represented paper claims on gold or silver coins. Bank notes issued by the government or private banks can be viewed as examples of such "warehouse certificates''.
```


As usual, we'll start by importing some Python modules.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
```

```{code-cell} ipython3
:tags: [hide-output]

!pip install xlrd 
```

We'll start by bringing these data into Pandas from a spreadsheet.

```{code-cell} ipython3
# import data and clean up the index
df_fig5 = pd.read_excel('datasets/longprices.xls', 
                        sheet_name='all', 
                        header=2, 
                        index_col=0).iloc[1:]
df_fig5.index = df_fig5.index.astype(int)
```

We first plot price levels over the period 1600-1914.

During most years in this time interval, the countries were on a gold or silver standard.

```{code-cell} ipython3
df_fig5_bef1914 = df_fig5[df_fig5.index <= 1915]

# create plot
cols = ['UK', 'US', 'France', 'Castile']

fig, ax = plt.subplots(figsize=[8, 5], dpi=200)

for col in cols:
    ax.plot(df_fig5_bef1914.index, 
            df_fig5_bef1914[col], label=col)

ax.spines[['right', 'top']].set_visible(False)
ax.legend()
ax.set_ylabel('Index  1913 = 100')
ax.set_xlim(xmin=1600)
plt.tight_layout()
fig.text(.5, .0001, 
         "Price Levels", ha='center')
plt.show()
```

We say "most years" because there were temporary lapses from the gold or silver standard.

By staring at the graph carefully, you might be able to guess when these temporary lapses occurred, because they were also times during which price levels temporarily rose markedly:

 * 1791-1797 in France (French Revolution)
 * 1776-1790 in the US (War for Independence from Great Britain)
 * 1861-1865 in the US (Civil War)

During these episodes, the gold/silver standard was temporarily abandoned when a government printed paper money to pay for  war expenditures.

Despite these temporary lapses, a striking thing about the figure is that price levels hovered around roughly constant long-term levels for over three centuries.  

In the early centuryTwo other features of these data  attracted the attention of  Irving Fisher of Yale University and John Maynard Keynes of Cambridge University.

  *  Despite beig  anchored to the same average level over long time spans, there were considerable year-to-year variations in   price levels
  
  * While using valuable gold and silver as coins succeeded in anchoring   the price level by limiting the supply of money, it cost real resources.
     
      * a country paid  a high "opportunity cost" for using gold and silver as coins as money: that  gold and silver could instead  have been made into  valuable jewelry and other durable goods. 

Keynes and Fisher proposed what they claimed  would be  a more  efficient way to achieve a price level that 
  *  would be at least as firmly  anchored as achieved under a gold or silver standard, and
  *  would also exhibit less  year-to-year short-term fluctuations.  

They said  that central bank could achieve price level stability by

  * issuing  **limited supplies** of paper currency
  * refusing to   print money to finance government expenditures

This logic   prompted John Maynard Keynes to call a commodity standard a “barbarous relic.”

A paper currency or ''fiat money''  system disposes of all reserves behind a currency. 

But adhereing to a gold or silver standard had provided  an automatic  mechanism for limiting the supply of money, thereby anchoring  the price level.

To anchor the price level, a  pure  paper or fiat money system replaces that automatic mechanism with a central bank with the authority and determination to limit the supply of money
(and to deter counterfeiters!) 

Now let's see what happened to the price level in our four countries when after 1914 one after another of them left the gold/silver standard by showing the complete  graph that originally appeared on page 35 of {cite}`sargent2002big`.

The graph shows logarithms of price levels our  four "hard currency" countries from 1600 to 2000.

Although we didn't have  to use   logarithms in our earlier graphs that  had stopped in 1914, now  we use logarithms  because we want also  to fit observations after 1914 in the same graph as the earlier observations.

After the outbreak of the Great War in 1914, the four countries  left the gold standard and in so doing acquired the ability to print money to finance government expenditures. 


```{code-cell} ipython3
fig, ax = plt.subplots(figsize=[8, 5], dpi=200)

for col in cols:
    ax.plot(df_fig5.index, df_fig5[col])
    ax.text(x=df_fig5.index[-1]+2, 
            y=df_fig5[col].iloc[-1], s=col)

ax.spines[['right', 'top']].set_visible(False)
ax.set_yscale('log')
ax.set_ylabel('Index  1913 = 100')
ax.set_xlim(xmin=1600)
ax.set_ylim([10, 1e6])
plt.tight_layout()
fig.text(.5, .0001, 
         "Logs of Price Levels", ha='center')
plt.show()
```

The graph shows that paper-money-printing central banks didn't do as well as the gold and standard silver standard in   anchoring   price levels.

That would probably have surprised or disappointed  Irving Fisher and John Maynard Keynes.

Actually, earlier economists and statesmen knew about the possibility of fiat money systems long before Keynes and Fisher advocated them in the early 20th century.

Proponents of a commodity money system did not trust governments and central banks  properly to manage a fiat money system.

They were willing to pay the resource costs associated with setting up and maintaining a commodity money system.

In light of the high and persistent  inflation  that many countries experienced  after they abandoned commodity monies in the twentieth century,  we hesitate to criticize advocates of a gold or silver standard  for their preference to stay on the pre-1914 gold/silver standard. 

The breadth and lengths of the inflationary experiences of the twentieth century under paper money fiat standards are  historically unprecedented.

## Four Big Inflations

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
        item_to_remove = ['<s>a</s>', '<s>c</s>', 
                          '<s>d</s>', '<s>e</s>']

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
        
    # convert years to int
    df['Year'] = df['Year'].apply(lambda x: int(x))
    
    # set index to datetime with year and month
    df = df.set_index(
            pd.to_datetime(
                (df['Year'].astype(str) + \
                 df['Month'].astype(str)), 
                format='%Y%B'))
    df = df.drop(['Year', 'Month'], axis=1)
    
    # handle duplicates by keeping the first
    df = df[~df.index.duplicated(keep='first')]
    
    # convert attribute values to numeric
    df = df.applymap(lambda x: float(x) \
                if x != '—' else np.nan)
    
    # finally, we only focus on data between 1919 and 1925
    mask = (df.index >= '1919-01-01') & \
           (df.index < '1925-01-01')
    df = df.loc[mask]

    return df

def pe_plot(p_seq, e_seq, index, labs, ax):
    "Generate plots for price and exchange rates."

    p_lab, e_lab = labs
    
    # plot price and exchange rates
    ax.plot(index, p_seq, label=p_lab, color='tab:blue')
    
    # add a new axis
    ax1 = ax.twinx()
    ax1.plot([None], [None], label=p_lab, color='tab:blue')
    ax1.plot(index, e_seq, label=e_lab, color='tab:orange')
    
    # set log axes
    ax.set_yscale('log')
    ax1.set_yscale('log')
    
    # define the axis label format
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=5))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    # set labels
    ax.text(-0.08, 1.03, 'Price Level', transform=ax.transAxes)
    ax.text(0.92, 1.03, 'Exchange Rate', transform=ax.transAxes)
    
    ax1.legend(loc='upper left')

    return ax1

def pr_plot(p_seq, index, ax):
    "Generate plots for inflation rates."

    #  Calculate the difference of log p_seq
    log_diff_p = np.diff(np.log(p_seq))
    
    # graph for the difference of log p_seq
    ax.scatter(index[1:], log_diff_p, 
               label='Monthly Inflation Rate', 
               color='tab:grey')
    
    # calculate and plot moving average
    diff_smooth = pd.DataFrame(log_diff_p).rolling(3).mean()
    ax.plot(index[1:], diff_smooth, alpha=0.5, color='tab:grey')
    ax.text(-0.08, 1.03, 
            'Monthly Inflation Rate', 
            transform=ax.transAxes)
    
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=5))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y'))
    
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    ax.legend(loc='upper left')
    
    return ax
    
```

```{code-cell} ipython3
# import data
xls = pd.ExcelFile('datasets/chapter_3.xlsx')

# select relevant sheets
sheet_index = [(2, 3, 4), 
               (9, 10), 
               (14, 15, 16), 
               (21, 18, 19)]

# remove redundant rows
remove_row = [(-2, -2, -2), 
              (-7, -10), 
              (-6, -4, -3), 
              (-19, -3, -6)]

# unpack and combine series for each country
df_list = []

for i in range(4):
    
    indices, rows = sheet_index[i], remove_row[i]
    
    # apply process_entry on the selected sheet
    sheet_list = [
        pd.read_excel(xls, 'Table3.' + str(ind), 
            header=1).iloc[:row].applymap(process_entry)
        for ind, row in zip(indices, rows)]
    
    sheet_list = [process_df(df) for df in sheet_list]
    df_list.append(pd.concat(sheet_list, axis=1))

df_Aus, df_Hung, df_Pol, df_Germ = df_list
```

Let's  construct graphs for our four countries.

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
p_seq = df_Aus['Retail price index, 52 commodities']
e_seq = df_Aus['Exchange Rate']

lab = ['Retail Price Index', 'Exchange Rate']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = pe_plot(p_seq, e_seq, df_Aus.index, lab, ax)

# connect disjunct parts
plt.figtext(0.5, -0.02, 'Austria', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

```{code-cell} ipython3
# plot moving average
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = pr_plot(p_seq, df_Aus.index, ax)

plt.figtext(0.5, -0.02, 'Austria', 
            horizontalalignment='center', 
            fontsize=12)
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
m_seq = df_Hung['Notes in circulation']
p_seq = df_Hung['Hungarian index of prices']
e_seq = 1 / df_Hung['Cents per crown in New York']

lab = ['Hungarian Index of Prices', 
       '1/Cents per Crown in New York']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = pe_plot(p_seq, e_seq, df_Hung.index, lab, ax)

plt.figtext(0.5, -0.02, 'Hungary', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

```{code-cell} ipython3
# plot moving average
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = pr_plot(p_seq, df_Hung.index, ax)

plt.figtext(0.5, -0.02, 'Hungary', 
            horizontalalignment='center', 
            fontsize=12)
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
# splice three price series in different units
p_seq1 = df_Pol['Wholesale price index'].copy()
p_seq2 = df_Pol['Wholesale Price Index: '
                'On paper currency basis'].copy()
p_seq3 = df_Pol['Wholesale Price Index: ' 
                'On zloty basis'].copy()

# non-nan part
mask_1 = p_seq1[~p_seq1.isna()].index[-1]
mask_2 = p_seq2[~p_seq2.isna()].index[-2]

adj_ratio12 = (p_seq1[mask_1] / p_seq2[mask_1])
adj_ratio23 = (p_seq2[mask_2] / p_seq3[mask_2])

# glue three series
p_seq = pd.concat([p_seq1[:mask_1], 
                   adj_ratio12 * p_seq2[mask_1:mask_2], 
                   adj_ratio23 * p_seq3[mask_2:]])
p_seq = p_seq[~p_seq.index.duplicated(keep='first')]

# exchange rate
e_seq = 1/df_Pol['Cents per Polish mark (zloty after May 1924)']
e_seq[e_seq.index > '05-01-1924'] = np.nan
```

```{code-cell} ipython3
lab = ['Wholesale Price Index', 
       '1/Cents per Polish Mark']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
ax1 = pe_plot(p_seq, e_seq, df_Pol.index, lab, ax)

plt.figtext(0.5, -0.02, 'Poland', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

```{code-cell} ipython3
# plot moving average
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = pr_plot(p_seq, df_Pol.index, ax)

plt.figtext(0.5, -0.02, 'Poland', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

### Germany

The sources of our data for Germany are the following tables from chapter 3 of {cite}`sargent2013rational`:

* Table 3.18, wholesale price level $\exp p$ 
* Table 3.19, exchange rate

```{code-cell} ipython3
p_seq = df_Germ['Price index (on basis of marks before July 1924,'
                '  reichsmarks after)'].copy()
e_seq = 1/df_Germ['Cents per mark']

lab = ['Price Index', 
       '1/Cents per Mark']

# create plot
fig, ax = plt.subplots(figsize=[9,5], dpi=200)
ax1 = pe_plot(p_seq, e_seq, df_Germ.index, lab, ax)

plt.figtext(0.5, -0.06, 'Germany', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

```{code-cell} ipython3
p_seq = df_Germ['Price index (on basis of marks before July 1924,'
                '  reichsmarks after)'].copy()
e_seq = 1/df_Germ['Cents per mark'].copy()

# adjust the price level/exchange rate after the currency reform
p_seq[p_seq.index > '06-01-1924'] = p_seq[p_seq.index 
                                          > '06-01-1924'] * 1e12
e_seq[e_seq.index > '12-01-1923'] = e_seq[e_seq.index 
                                          > '12-01-1923'] * 1e12

lab = ['Price Index (Marks or converted to Marks)', 
       '1/Cents per Mark (or Reichsmark converted to Mark)']

# create plot
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
ax1 = pe_plot(p_seq, e_seq, df_Germ.index, lab, ax)

plt.figtext(0.5, -0.02, 'Germany', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

```{code-cell} ipython3
# plot moving average
fig, ax = plt.subplots(figsize=[10,7], dpi=200)
_ = pr_plot(p_seq, df_Germ.index, ax)

plt.figtext(0.5, -0.02, 'Germany', 
            horizontalalignment='center', 
            fontsize=12)
plt.show()
```

## Starting and Stopping Big Inflations

It is  striking  how **quickly**  (log) price levels in Austria, Hungary, Poland,
and Germany leveled off after  rising so quickly.

These "sudden stops" are also revealed by the permanent drops in three-month moving averages of inflation for the four countries plotted above.

In addition, the US dollar exchange rates for each of the four countries shadowed their price levels. 

  * This pattern is an instance of a force featured in the **purchasing power parity** theory of exchange rates (see <https://en.wikipedia.org/wiki/Purchasing_power_parity>).

Each of these big inflations seemed to have "stopped on a dime".

Chapter 3 of {cite}`sargent2002big` offers an explanation for this remarkable pattern.

In a nutshell, here is the explanation offered there.

After World War I, the United States was on a gold standard. 

The US government stood ready to convert a dollar into a specified amount of gold on demand.

Immediately after World War I, Hungary, Austria, Poland, and Germany were not on the gold standard. 

Their currencies were  “fiat” or "unbacked",  meaning that they were not backed by credible government promises to convert them into gold or silver coins on demand.

The governments  printed new paper notes to pay for goods and services. 

```{note}
Technically the notes were "backed" mainly by treasury bills. But people could not expect that those treasury bills would  be paid off by levying taxes, but instead by printing more notes or treasury bills.
```
 This was done on such a scale that it led to a depreciation of the currencies of spectacular proportions. 
 
 In the end, the German mark stabilized at 1 trillion ($10^{12}$) paper marks to the prewar gold mark, the Polish mark at 1.8 million paper marks to the gold zloty, the Austrian crown at 14,400 paper crowns to the prewar Austro-Hungarian crown, and the Hungarian krone at 14,500 paper crowns to the prewar Austro-Hungarian crown.

Chapter 3 of {cite}`sargent2002big`  described  deliberate changes in policy that Hungary, Austria, Poland, and Germany made to end their hyperinflations.

Each governent  stoppped printing money to pay for goods and services once again made its currency  convertible  to the US dollar or the UK pound, thereby vitually  to gold.

The story told in {cite}`sargent2002big` is grounded in a "monetarist theory of the price level" described in {doc}`this lecture <cagan_ree>` and 
{doc}`this lecture <cagan_adaptive>`.

Those lectures discuss theories about what owners  of those rapidly depreciating currencies were thinking  and how their beliefs shaped responses of inflation to government monetary and fiscal  policies.
