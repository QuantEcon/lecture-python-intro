---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<!-- #region user_expressions=[] -->
# Inflation During French Revolution 


## Overview 

This lecture describes some monetary and fiscal  features of the French Revolution
described by {cite}`sargent_velde1995`.

In order to finance public expenditures and service debts issued by earlier French governments, 
successive French governments performed several policy experiments.

Authors of these experiments were guided by their having decided to put in place monetary-fiscal policies recommended by particular theories.  

As a consequence, data on money growth and inflation from the period 1789 to 1787 at least temorarily illustrated outcomes  predicted by these   arrangements:

* some *unpleasant monetarist arithmetic* like that described in this quanteon lecture XXX
that governed French government debt dynamics in the decades preceding 1789 

* a *real bills* theory of the effects of government open market operations in which the government *backs* its issues of paper money with valuable real property or financial assets

* a classical ``gold or silver'' standard

* a classical inflation-tax theory of inflation in which Philip Cagan's  demand for money studied 
in this lecture is a key component

* a *legal restrictions*  or *financial repression* theory of the demand for real balances 

We use matplotlib to replicate several of the graphs that they used to present salient patterns.



## Data Sources

This notebook uses data from three spreadsheets:

  * datasets/fig_3.ods
  * datasets/dette.xlsx
  * datasets/assignat.xlsx

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
```


## Figure 1
<!-- #endregion -->

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Ratio of debt service to taxes, Britain and France"
    name: fig1
---

# Read the data from the Excel file
data1 = pd.read_excel('datasets/dette.xlsx', sheet_name='Debt', usecols='R:S', skiprows=5, nrows=99, header=None)
data1a = pd.read_excel('datasets/dette.xlsx', sheet_name='Debt', usecols='P', skiprows=89, nrows=15, header=None)

# Plot the data
plt.figure()
plt.plot(range(1690, 1789), 100 * data1.iloc[:, 1], linewidth=0.8)

date = np.arange(1690, 1789)
index = (date < 1774) & (data1.iloc[:, 0] > 0)
plt.plot(date[index], 100 * data1[index].iloc[:, 0], '*:', color='r', linewidth=0.8)

# Plot the additional data
plt.plot(range(1774, 1789), 100 * data1a, '*:', color='orange')

# Note about the data
# The French data before 1720 don't match up with the published version
# Set the plot properties
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_xlim([1688, 1788])
plt.ylabel('% of Taxes')

plt.tight_layout()
plt.show()

#plt.savefig('frfinfig1.pdf', dpi=600)
#plt.savefig('frfinfig1.jpg', dpi=600)
```


 {numref}`fig1` plots ratios of debt service to total taxes collected for Great Britain and France.
 The figure shows 

  * ratios of debt service to taxes rise  for both countries  at the  beginning of the century and at the end of the century 
  * ratios that are similar for both countries in most years 



<!-- #region user_expressions=[] -->

## Figure 2
<!-- #endregion -->

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Government Expenditures and Tax Revenues in Britain"
    name: fig2
---

# Read the data from Excel file
data2 = pd.read_excel('datasets/dette.xlsx', sheet_name='Militspe', usecols='M:X', skiprows=7, nrows=102, header=None)

# Plot the data
plt.figure()
plt.plot(range(1689, 1791), data2.iloc[:, 5], linewidth=0.8)
plt.plot(range(1689, 1791), data2.iloc[:, 11], linewidth=0.8, color='red')
plt.plot(range(1689, 1791), data2.iloc[:, 9], linewidth=0.8, color='orange')
plt.plot(range(1689, 1791), data2.iloc[:, 8], 'o-', markerfacecolor='none', linewidth=0.8, color='purple')

# Customize the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.ylabel('millions of pounds', fontsize=12)

# Add text annotations
plt.text(1765, 1.5, 'civil', fontsize=10)
plt.text(1760, 4.2, 'civil plus debt service', fontsize=10)
plt.text(1708, 15.5, 'total govt spending', fontsize=10)
plt.text(1759, 7.3, 'revenues', fontsize=10)


plt.tight_layout()
plt.show()

# Save the figure as a PDF
#plt.savefig('frfinfig2.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->

{numref}`fig2` plots total taxes, total government expenditures, and the composition of government expenditures in Great Britain during much of the 18th century.

## Figure 3 


<!-- #endregion -->

```{code-cell} ipython3
# Read the data from the Excel file
data1 = pd.read_excel('datasets/fig_3.xlsx', sheet_name='Sheet1', usecols='C:F', skiprows=5, nrows=30, header=None)

data1.replace(0, np.nan, inplace=True)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Government Spending and Tax Revenues in France"
    name: fr_fig3
---
# Plot the data
plt.figure()

plt.plot(range(1759, 1789, 1), data1.iloc[:, 0], '-x', linewidth=0.8)
plt.plot(range(1759, 1789, 1), data1.iloc[:, 1], '--*', linewidth=0.8)
plt.plot(range(1759, 1789, 1), data1.iloc[:, 2], '-o', linewidth=0.8, markerfacecolor='none')
plt.plot(range(1759, 1789, 1), data1.iloc[:, 3], '-*', linewidth=0.8)

plt.text(1775, 610, 'total spending', fontsize=10)
plt.text(1773, 325, 'military', fontsize=10)
plt.text(1773, 220, 'civil plus debt service', fontsize=10)
plt.text(1773, 80, 'debt service', fontsize=10)
plt.text(1785, 500, 'revenues', fontsize=10)



plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([0, 700])
plt.ylabel('millions of livres')

plt.tight_layout()
plt.show()

#plt.savefig('frfinfig3.jpg', dpi=600)
```


TO TEACH TOM:  By staring at {numref}`fr_fig3` carefully

{numref}`fr_fig3` plots total taxes, total government expenditures, and the composition of government expenditures in France  during much of the 18th century.

```{code-cell} ipython3

---
mystnb:
  figure:
    caption: "Government Spending and Tax Revenues in France"
    name: fr_fig3b
---
# Plot the data
plt.figure()

plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 0])], data1.iloc[:, 0][~np.isnan(data1.iloc[:, 0])], '-x', linewidth=0.8)
plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 1])], data1.iloc[:, 1][~np.isnan(data1.iloc[:, 1])], '--*', linewidth=0.8)
plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 2])], data1.iloc[:, 2][~np.isnan(data1.iloc[:, 2])], '-o', linewidth=0.8, markerfacecolor='none')
plt.plot(np.arange(1759, 1789, 1)[~np.isnan(data1.iloc[:, 3])], data1.iloc[:, 3][~np.isnan(data1.iloc[:, 3])], '-*', linewidth=0.8)

plt.text(1775, 610, 'total spending', fontsize=10)
plt.text(1773, 325, 'military', fontsize=10)
plt.text(1773, 220, 'civil plus debt service', fontsize=10)
plt.text(1773, 80, 'debt service', fontsize=10)
plt.text(1785, 500, 'revenues', fontsize=10)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([0, 700])
plt.ylabel('millions of livres')

plt.tight_layout()
plt.show()

#plt.savefig('frfinfig3_ignore_nan.jpg', dpi=600)
```

{numref}`fr_fig3b` plots total taxes, total government expenditures, and the composition of government expenditures in France  during much of the 18th century.

<!-- #region user_expressions=[] -->

<!-- #region user_expressions=[] -->
## Figure 4
<!-- #endregion -->

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Military Spending in Britain and France"
    name: fig4
---
# French military spending, 1685-1789, in 1726 livres
data4 = pd.read_excel('datasets/dette.xlsx', sheet_name='Militspe', usecols='D', skiprows=3, nrows=105, header=None).squeeze()
years = range(1685, 1790)

plt.figure()
plt.plot(years, data4, '*-', linewidth=0.8)

plt.plot(range(1689, 1791), data2.iloc[:, 4], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.xlabel('*: France')
plt.ylabel('Millions of livres')
plt.ylim([0, 475])

plt.tight_layout()
plt.show()

#plt.savefig('frfinfig4.pdf', dpi=600)
```


{numref}`fig4` plots total taxes, total government expenditures, and the composition of government expenditures in France  during much of the 18th century.

TO TEACH TOM:  By staring at {numref}`fig4` carefully


## Figure 5
<!-- #endregion -->

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Index of real per capital revenues, France"
    name: fig5
---
# Read data from Excel file
data5 = pd.read_excel('datasets/dette.xlsx', sheet_name='Debt', usecols='K', skiprows=41, nrows=120, header=None)

# Plot the data
plt.figure()
plt.plot(range(1726, 1846), data5.iloc[:, 0], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim([1726, 1845])
plt.ylabel('1726 = 1', fontsize=12)

plt.tight_layout()
plt.show()

# Save the figure as a PDF
#plt.savefig('frfinfig5.pdf', dpi=600)
```

TO TEACH TOM:  By staring at {numref}`fig5` carefully

## Rise and Fall of the *Assignat*



 We have partitioned Figures~\ref{fig:fig7}, \ref{fig:fig8}, and \ref{fig:fig9}
 into three periods, corresponding
to different monetary regimes or episodes. The three clouds of points in
Figure~\ref{fig:fig7}
 depict different real balance-inflation relationships. Only the cloud for the
third period has the inverse relationship familiar to us now from twentieth-century
hyperinflations. The first period ends in the late summer of 1793, and is characterized
by growing real balances and moderate inflation. The second period begins and ends
with the Terror. It is marked by high real balances, around 2,500 millions, and
roughly stable prices. The fall of Robespierre in late July 1794 begins the third
of our episodes, in which real balances decline and prices rise rapidly. We interpret
these three episodes in terms of three separate theories about money: a ``backing''
or ''real bills'' theory (the text is Adam Smith (1776)),
a legal restrictions theory (TOM: HERE PLEASE CITE 
Keynes,1940, AS WELL AS  Bryant/Wallace:1984 and Villamil:1988) 
and a classical hyperinflation theory.%
```{note}
According to the empirical  definition of hyperinflation adopted by {cite}`Cagan`,
beginning in the month that inflation exceeds 50 percent
per month and ending in the month before inflation drops below 50 percent per month
for at least a year, the *assignat*  experienced a hyperinflation from May to December
1795.
```
We view these
theories not as competitors but as alternative collections of ``if-then''
statements about government note issues, each of which finds its conditions more
nearly met in one of these episodes than in the other two.

<!-- #region user_expressions=[] -->



## Figure 7


## To Do for Zejin

I want to tweak and consolidate the extra lines that Zejin drew on   the beautiful **Figure 7**.  

I'd like to experiment in plotting the **six** extra lines all on one graph -- a pair of lines for each of our subsamples

  * one for the $y$ on $x$ regression line
  * another for the $x$ on $y$ regression line

I'd like the  $y$ on $x$ and $x$ on $y$ lines to be in separate colors.

Once we are satisfied with this new graph with its six additional lines, we can dispense with the other graphs that add one line at a time. 

Zejin, I can explain on zoom the lessons I want to convey with this.  



Just to recall, to compute the regression lines, Zejin wrote  a  function that use standard formulas
for a and b in a least squares regression y = a + b x + residual -- i.e., b is ratio of sample covariance of y,x to sample variance of x; while a is then computed from a =  sample mean of y - \hat b *sample mean of x

We could presumably tell students how to do this with a couple of numpy lines
I'd like to create three additional versions of the following figure. 

To remind you, we focused on  three  subperiods:


* subperiod 1: ("real bills period): January 1791 to July 1793

* subperiod 2: ("terror:):  August 1793 - July 1794

* subperiod 3: ("classic Cagan hyperinflation): August 1794 - March 1796


I can explain what this is designed to show.

<!-- #endregion -->

```{code-cell} ipython3
def fit(x, y):

    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()

    return a, b
```

```{code-cell} ipython3
# load data
caron = np.load('datasets/caron.npy')
nom_balances = np.load('datasets/nom_balances.npy')

infl = np.concatenate(([np.nan], -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000
```

```{code-cell} ipython3
# fit data

# reg y on x for three periods
a1, b1 = fit(bal[1:31], infl[1:31])
a2, b2 = fit(bal[31:44], infl[31:44])
a3, b3 = fit(bal[44:63], infl[44:63])

# reg x on y for three periods
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7.pdf', dpi=600)
```



```{code-cell} ipython3
# fit data

# reg y on x for three periods
a1, b1 = fit(bal[1:31], infl[1:31])
a2, b2 = fit(bal[31:44], infl[31:44])
a3, b3 = fit(bal[44:63], infl[44:63])

# reg x on y for three periods
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[34:44], infl[34:44], '+', color='red', label='terror')

# third subsample  # Tom tinkered with subsample period
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7.pdf', dpi=600)
```


<p style="color:blue;">The above graph is Tom's experimental lab. We'll delete it eventually.</p>

<p style="color:red;">Zejin: below is the grapth with six lines in one graph. The lines generated by regressing y on x have the same color as the corresponding data points, while the lines generated by regressing x on y are all in green.</p>

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue', linewidth=0.8)
plt.plot(a1_rev + b1_rev * infl[1:31], infl[1:31], color='green', linewidth=0.8)

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red', linewidth=0.8)
plt.plot(a2_rev + b2_rev * infl[31:44], infl[31:44], color='green', linewidth=0.8)

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange', linewidth=0.8)
plt.plot(a3_rev + b3_rev * infl[44:63], infl[44:63], color='green', linewidth=0.8)

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()
#plt.savefig('frfinfig7.pdf', dpi=600)
```



<p style="color:blue;">The graph below is Tom's version of the six lines in one graph. The lines generated by regressing y on x have the same color as the corresponding data points, while the lines generated by regressing x on y are all in green.</p>

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue', linewidth=0.8)
plt.plot(a1_rev + b1_rev * infl[1:31], infl[1:31], color='green', linewidth=0.8)

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
plt.plot(bal[34:44], a2 + bal[34:44] * b2, color='red', linewidth=0.8)
plt.plot(a2_rev + b2_rev * infl[34:44], infl[34:44], color='green', linewidth=0.8)

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange', linewidth=0.8)
plt.plot(a3_rev + b3_rev * infl[44:63], infl[44:63], color='green', linewidth=0.8)

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7.pdf', dpi=600)
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7_line1.pdf', dpi=600)
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')
plt.plot(a1_rev + b1_rev * infl[1:31], infl[1:31], color='blue')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7_line1_rev.pdf', dpi=600)
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
plt.plot(bal[31:44], a2 + bal[31:44] * b2, color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7_line2.pdf', dpi=600)
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')
plt.plot(a2_rev + b2_rev * infl[31:44], infl[31:44], color='red')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7_line2_rev.pdf', dpi=600)
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7_line3.pdf', dpi=600)
```

```{code-cell} ipython3
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# first subsample
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='real bills period')

# second subsample
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='terror')

# third subsample
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='classic Cagan hyperinflation')
plt.plot(a3_rev + b3_rev * infl[44:63], infl[44:63], color='orange')

plt.xlabel('real balances')
plt.ylabel('inflation')
plt.legend()

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig7_line3_rev.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 8
<!-- #endregion -->

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Real balances of assignats (in gold and goods)"
    name: fig8
---
# Read the data from Excel file
data7 = pd.read_excel('datasets/assignat.xlsx', sheet_name='Data', usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel('datasets/assignat.xlsx', sheet_name='Data', usecols='L', skiprows=4, nrows=80, header=None)

# Create the figure and plot
plt.figure()
h = plt.plot(pd.date_range(start='1789-11-01', periods=len(data7), freq='M'), (data7a.values * [1, 1]) * data7.values, linewidth=1.)
plt.setp(h[1], linestyle='--', color='red')

plt.vlines([pd.Timestamp('1793-07-15'), pd.Timestamp('1793-07-15')], 0, 3000, linewidth=0.8, color='orange')
plt.vlines([pd.Timestamp('1794-07-15'), pd.Timestamp('1794-07-15')], 0, 3000, linewidth=0.8, color='purple')

plt.ylim([0, 3000])

# Set properties of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim(pd.Timestamp('1789-11-01'), pd.Timestamp('1796-06-01'))
plt.ylabel('millions of livres', fontsize=12)

# Add text annotations
plt.text(pd.Timestamp('1793-09-01'), 200, 'Terror', fontsize=12)
plt.text(pd.Timestamp('1791-05-01'), 750, 'gold value', fontsize=12)
plt.text(pd.Timestamp('1794-10-01'), 2500, 'real value', fontsize=12)


plt.tight_layout()
plt.show()

# Save the figure as a PDF
#plt.savefig('frfinfig8.pdf', dpi=600)
```

TO TEACH TOM:  By staring at {numref}`fig8` carefully

<!-- #region user_expressions=[] -->
## Figure 9
<!-- #endregion -->

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Price Level and Price of Gold (log scale)"
    name: fig9
---
# Create the figure and plot
plt.figure()
x = np.arange(1789 + 10/12, 1796 + 5/12, 1/12)
h, = plt.plot(x, 1. / data7.iloc[:, 0], linestyle='--')
h, = plt.plot(x, 1. / data7.iloc[:, 1], color='r')

# Set properties of the plot
plt.gca().tick_params(labelsize=12)
plt.yscale('log')
plt.xlim([1789 + 10/12, 1796 + 5/12])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add vertical lines
plt.axvline(x=1793 + 6.5/12, linestyle='-', linewidth=0.8, color='orange')
plt.axvline(x=1794 + 6.5/12, linestyle='-', linewidth=0.8, color='purple')

# Add text
plt.text(1793.75, 120, 'Terror', fontsize=12)
plt.text(1795, 2.8, 'price level', fontsize=12)
plt.text(1794.9, 40, 'gold', fontsize=12)


plt.tight_layout()
plt.show()
#plt.savefig('frfinfig9.pdf', dpi=600)
```

TO TEACH TOM:  By staring at {numref}`fig9` carefully

<!-- #region user_expressions=[] -->
## Figure 11
<!-- #endregion -->



```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Spending (blue) and Revenues (orange), (real values)"
    name: fig11
---
# Read data from Excel file
data11 = pd.read_excel('datasets/assignat.xlsx', sheet_name='Budgets', usecols='J:K', skiprows=22, nrows=52, header=None)

# Prepare the x-axis data
x_data = np.concatenate([
    np.arange(1791, 1794 + 8/12, 1/12),
    np.arange(1794 + 9/12, 1795 + 3/12, 1/12)
])

# Remove NaN values from the data
data11_clean = data11.dropna()

# Plot the data
plt.figure()
h = plt.plot(x_data, data11_clean.values[:, 0], linewidth=0.8)
h = plt.plot(x_data, data11_clean.values[:, 1], '--', linewidth=0.8)



# Set plot properties
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlim([1791, 1795 + 3/12])
plt.xticks(np.arange(1791, 1796))
plt.yticks(np.arange(0, 201, 20))

# Set the y-axis label
plt.ylabel('millions of livres', fontsize=12)



plt.tight_layout()
plt.show()

#plt.savefig('frfinfig11.pdf', dpi=600)
```
TO TEACH TOM:  By staring at {numref}`fig11` carefully

<!-- #region user_expressions=[] -->
## Figure 12
<!-- #endregion -->

```{code-cell} ipython3
# Read data from Excel file
data12 = pd.read_excel('datasets/assignat.xlsx', sheet_name='seignor', usecols='F', skiprows=6, nrows=75, header=None).squeeze()


# Create a figure and plot the data
plt.figure()
plt.plot(pd.date_range(start='1790', periods=len(data12), freq='M'), data12, linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=472.42/12, color='r', linestyle=':')
plt.xticks(ticks=pd.date_range(start='1790', end='1796', freq='AS'), labels=range(1790, 1797))
plt.xlim(pd.Timestamp('1791'), pd.Timestamp('1796-02') + pd.DateOffset(months=2))
plt.ylabel('millions of livres', fontsize=12)
plt.text(pd.Timestamp('1793-11'), 39.5, 'revenues in 1788', verticalalignment='top', fontsize=12)


plt.tight_layout()
plt.show()

#plt.savefig('frfinfig12.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 13
<!-- #endregion -->

```{code-cell} ipython3
# Read data from Excel file
data13 = pd.read_excel('datasets/assignat.xlsx', sheet_name='Exchge', usecols='P:T', skiprows=3, nrows=502, header=None)

# Plot the last column of the data
plt.figure()
plt.plot(data13.iloc[:, -1], linewidth=0.8)

# Set properties of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_xlim([1, len(data13)])

# Set x-ticks and x-tick labels
ttt = np.arange(1, len(data13) + 1)
plt.xticks(ttt[~np.isnan(data13.iloc[:, 0])], 
           ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb',
           'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])

# Add text to the plot
plt.text(1, 120, '1795', fontsize=12, ha='center')
plt.text(262, 120, '1796', fontsize=12, ha='center')

# Draw a horizontal line and add text
plt.axhline(y=186.7, color='red', linestyle='-', linewidth=0.8)
plt.text(150, 190, 'silver parity', fontsize=12)

# Add an annotation with an arrow
plt.annotate('end of the assignat', xy=(340, 172), xytext=(380, 160),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)


plt.tight_layout()
plt.show()
#plt.savefig('frfinfig13.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 14
<!-- #endregion -->

```{code-cell} ipython3
# figure 14
data14 = pd.read_excel('datasets/assignat.xlsx', sheet_name='Post-95', usecols='I', skiprows=9, nrows=91, header=None).squeeze()
data14a = pd.read_excel('datasets/assignat.xlsx', sheet_name='Post-95', usecols='F', skiprows=100, nrows=151, header=None).squeeze()

plt.figure()
h = plt.plot(data14, '*-', markersize=2, linewidth=0.8)
plt.plot(np.concatenate([np.full(data14.shape, np.nan), data14a]), linewidth=0.8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_xticks(range(20, 237, 36))
plt.gca().set_xticklabels(range(1796, 1803))
plt.xlabel('*: Before the 2/3 bankruptcy')
plt.ylabel('Francs')

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig14.pdf', dpi=600)
```

<!-- #region user_expressions=[] -->
## Figure 15
<!-- #endregion -->

```{code-cell} ipython3
# figure 15
data15 = pd.read_excel('datasets/assignat.xlsx', sheet_name='Post-95', usecols='N', skiprows=4, nrows=88, header=None).squeeze()

plt.figure()
h = plt.plot(range(2, 90), data15, '*-', linewidth=0.8)
plt.setp(h, markersize=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.text(47.5, 11.4, '17 brumaire', horizontalalignment='left', fontsize=12)
plt.text(49.5, 14.75, '19 brumaire', horizontalalignment='left', fontsize=12)
plt.text(15, -1, 'Vendémiaire 8', fontsize=12, horizontalalignment='center')
plt.text(45, -1, 'Brumaire', fontsize=12, horizontalalignment='center')
plt.text(75, -1, 'Frimaire', fontsize=12, horizontalalignment='center')
plt.ylim([0, 25])
plt.xticks([], [])
plt.ylabel('Francs')

plt.tight_layout()
plt.show()
#plt.savefig('frfinfig15.pdf', dpi=600)
```

```{code-cell} ipython3

```


## Fiscal Situation and Response of National Assembly


In response to a motion by Catholic Bishop Talleyrand,
the National Assembly confiscated and nationalized  Church lands. 

But the National Assembly was dominated by free market advocates, not socialists.

The National Assembly intended to use earnings from  Church lands to service its national debt.

To do this, it  began to implement a ''privatization plan'' that would let it service its debt while
not raising taxes.

Their plan involved issuing paper notes called ''assignats'' that entitled bearers to use them to purchase state lands.  

These paper notes would be ''as good as silver coins'' in the sense that both were acceptable means of payment in exchange for those (formerly) church lands.  

Finance Minister Necker and the Constituants planned
to solve the privatization problem **and** the debt problem simultaneously
by creating a new currency. 

They devised a scheme to raise revenues by auctioning
the confiscated lands, thereby withdrawing paper notes issued on the security of
the lands sold by the government.

 This ''tax-backed money'' scheme propelled the National Assembly  into the domain of monetary experimentation.
 
Records of their debates show
how members of the Assembly marshaled theory and evidence to assess the likely
effects of their innovation. 

They quoted David Hume and Adam Smith and cited John
Law's System of 1720 and the American experiences with paper money fifteen years
earlier as examples of how paper money schemes can go awry.


### Necker's plan and how it was tweaked

Necker's original plan embodied two components: a national bank and a new
financial instrument, the ''assignat''. 


Necker's national
bank was patterned after the Bank of England. He proposed to transform the *Caisse d'Escompte* into a national bank by granting it a monopoly on issuing
notes and marketing government debt. The *Caisse*  was a
discount bank founded in 1776 whose main function was to discount commercial bills
and issue convertible notes. Although independent of the government in principle,
it had occasionally been used as a source of loans. Its notes had been declared
inconvertible in August 1788, and by the time of Necker's proposal, its reserves
were exhausted. Necker's plan placed the National Estates (as the Church lands
became known after the addition of the royal demesne) at the center of the financial
picture: a ''Bank of France'' would issue a $5\%$ security mortgaged on the prospective
receipts from the modest sale of some 400 millions' worth of National Estates in
the years 1791 to 1793.
```{note}
 Only 170 million was to be used initially
to cover the deficits of 1789 and 1790.
```


By mid-1790, members of the National Assembly had agreed to sell the National
Estates and to use the proceeds to service the debt in a ``tax-backed money'' scheme 
```{note}
Debt service costs absorbed 
 over 60\% of French government expenditures. 
```

The government would issue securities with which it would reimburse debt.

The securities
were acceptable as payment for National Estates purchased at auctions; once received
in payment, they were to be burned. 

```{note} 
The appendix to {cite}`sargent_velde1995` describes  the
auction rules in detail.
```
The Estates available for sale were thought to be worth about 2,400
million, while the exactable debt (essentially fixed-term loans, unpaid arrears,
and liquidated offices) stood at about 2,000 million. The value of the land was
sufficient to let the Assembly retire all of the exactable debt and thereby eliminate
the interest payments on it. After lengthy debates, in August 1790, the Assembly set the denomination
and interest rate structure of the debt. 


```{note} Two distinct
aspects of monetary theory help in thinking about the assignat plan. First, a system
beginning with a commodity standard typically has room for a once-and-for-all emission
of (an unbacked) paper currency that can replace the commodity money without generating
inflation. \citet{Sargent/Wallace:1983} describe models with this property. That
commodity money systems are wasteful underlies Milton Friedman's (1960) TOM:ADD REFERENCE preference
for a fiat money regime over a commodity money. Second, in a small country on a
commodity money system that starts with restrictions on intermediation, those restrictions
can be relaxed by letting the government issue bank notes on the security of safe
private indebtedness, while leaving bank notes convertible into gold at par. See
Adam Smith  and Sargent and Wallace (1982) for expressions of this idea. TOM: ADD REFERENCES HEREAND IN BIBTEX FILE.
```


```{note} 
The
National Assembly debated many now classic questions in monetary economics. Under
what conditions would money creation generate inflation, with what consequences
for business conditions? Distinctions were made between issue of money to pay off
debt, on one hand, and monetization of deficits, on the other. Would *assignats* be akin
to notes emitted under a real bills regime, and cause loss of specie, or would
they circulate alongside specie, thus increasing the money stock? Would inflation
affect real wages? How would it impact foreign trade, competitiveness of French
industry and agriculture, balance of trade, foreign exchange?
```
