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

This lecture describes some monetary and fiscal  features of the French Revolution (1789-1799)
described by {cite}`sargent_velde1995`.

To finance public expenditures and service its debts, 
the French Revolutionaries  performed several policy experiments.

The Revolutionary legislators who authored these experiments were guided by their having decided to put in place monetary-fiscal policies recommended to them  by theories that they believed.

Some of those theories make contact with modern theories about monetary and fiscal policies that interest us today.

* a *tax-smoothing* model like Robert Barro's {cite}`Barro1979`

   * this normative (i.e., prescriptive mode) advises a government to finance temporary war-time surges in government expenditures mostly by issuing government debt; after the war to roll over whatever debt accumulated during the war, and  to increase taxes permanently by enough to finance interest payments on that post-war debt

*  *unpleasant monetarist arithmetic* like that described in this quanteon lecture  {doc}`unpleasant`
   
    * this arithmetic governed French government debt dynamics in the decades preceding 1789 and according to leading historians set the stage for the French Revolution 

* a *real bills* theory of the effects of government open market operations in which the government *backs* its issues of paper money with valuable real property or financial assets

    * the Revolutionaries learned about this theory from Adam Smith's 1776 book The Wealth of Nations
     and other contemporary sources

    * It shaped how the Revolutionaries issued paper money called assignats from 1789 to 1791 

* a classical **gold**  or **silver** standard
  
    * Napoleon, who became head of government in 1799 used this theory to guide his monetary and fiscal policies

* a classical inflation-tax theory of inflation in which Philip Cagan's  demand for money studied 
in this lecture  {doc}`cagan_ree` is a key component

   * This theory helps us explain French price level and money supply data from 1794 to 1797  s

* a *legal restrictions*  or *financial repression* theory of the demand for real balances 
 
    * the Twelve Members comprising the Committee of Public Safety who adminstered the Terror from June 1793 to July 1794 used this theory to guide their monetary policy 

We use matplotlib to replicate several of the graphs that {cite}`sargent_velde1995` used to portray outcomes of these experiments 

---



## Data Sources

This lecture uses data from three spreadsheets:

  * datasets/fig_3.ods
  * datasets/dette.xlsx
  * datasets/assignat.xlsx

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
```

## Government Expenditures and Taxes Collected



We'll start by using matplotlib to construct two graphs that will provide important historical context.



```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Military Spending in Britain and France"
    name: fig4
---
# Read the data from Excel file
data2 = pd.read_excel('datasets/dette.xlsx', sheet_name='Militspe', usecols='M:X', skiprows=7, nrows=102, header=None)
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

During the 18th century, Britain and France fought four large wars.

Britain won the first three wars and lost the fourth.

Each  of those wars  produced surges in both countries government expenditures that each country somehow had to finance.

Figure {numref}`fig4` shows surges in military expenditures in France (in blue) and Great Britain.
during those four wars.  

A remarkable aspect of figure {numref}`fig4` is that despite having a population less than half of France's, Britain was able to finance military expenses of about the same amount as France's.

This testifies to Britain's success in having created state institutions that could tax, spend, and borrow.  






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


Figures  {numref}`fig2` and  {numref}`fr_fig3` summarize British and French government   fiscal policies  during the century before the start the French Revolution in 1789.


Progressive forces in France before 1789 thought admired how  Britain had financed its government expenditures and advocated reforms in French institutions designed to make them more like Britain's.

Figure  {numref}`fig2` shows government expenditures and how it was distributed among expenditures for 

   * civil (non military) activities
   * debt service, i.e., interest payments 
   * military expenditures (the yellow line minus the red line) 

Figure  {numref}`fig2` also plots total government revenues from tax collections (the purple circled line)

Notice the surges in total government expenditures associated with surges in military expenditures
in these four wars

   * Wars against France's King Louis XIV early in the 18th century
   * The War of the Spanish Succession in the 1740s
   * The French and Indian War in the 1750's and 1760s
   * The American War for Independence from 1775 to 1783

Figure {numref}`fig2` indicates that

   * during times of peace, the expenditures approximately equal taxes and debt service payments neither grow nor decline over time
   * during times of wars, government expenditures exceed tax revenues
      * the government finances the deficit of revenues relative to expenditures by issuing debt
   * after a war is over, the government's tax revenues exceed its non-interest expenditures by just enough to service the debt that the government issued to finance earlier deficits
      * thus, after a war, the government does **not** raise taxes by enough to pay off its debt
      * instead, it just rolls over whatever debt it inherits, raising taxes by just enough to service the interest payments on that debt

Eighteenth century British fiscal policy portrayed Figure {numref}`fig2` thus looks very much like a text-book example of a **tax-smoothing** model like Robert Barro's {cite}`Barro1979`.  

A striking feature of the graph is what we'll nick name a  **law of gravity** for taxes and expenditures. 

   * levels of government expenditures at taxes  attract each other
   * while they can temporarily differ -- as they do during wars -- they come back together when peace returns



Next we'll plot data on debt service costs as fractions of government revenues in Great Britain and France during the 18th century.

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


Figure  {numref}`fig1` shows that interest payments on government debt (i.e., so-called ``debt service'') were high fractions of government tax revenues in both Great Britain and France.  

Figure {numref}`fig2` showed us that Britain managed to balance its budget despite those large 
interest costs. 
<!-- #region user_expressions=[] -->





But as  we'll see in our next graph, on the eve of the French Revolution in 1788, that  fiscal policy   **law of gravity** that worked so well in Britain, did not seem to be working in France.



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

Figure {numref}`fr_fig3` shows that in 1788  on the eve of the French Revolution government expenditures exceeded   tax revenues.   

Especially during and after France's expenditures to help the Americans in their War of Independence from Great Britain,   growing government debt service (i.e., interest payments) 
contributed to this situation. 

This was partly a consequence of the unfolding of the debt dynamics that underlies the Unpleasant Arithmetic discussed in this quantecon lecture  {doc}`unpleasant`.  


{cite}`sargent_velde1995` describe how the Ancient Regime that until 1788 had  governed France  had stable institutional features that made it difficult for the government to balance its budget.

Powerful contending interests had prevented from the government from closing the gap between its
total expenditures and its tax revenues by 

 * raising taxes 
 * lowering government's non debt service (i.e., non-interest)   expenditures 
 * lowering its debt service (i.e., interest) costs by rescheduling its debt, i.e., defaulting on on part of its debt

The French constitution and prevailing arrangements had empowered three constituencies to block adjustments to components of the government budget constraint that they cared especially about

* tax payers
* beneficiaries of government expenditures
* government creditors (i.e., owners of government bonds)

When the French government had confronted a similar situation around 1720 after King  Louis XIV's
Wars had left it with a debt crisis, it had ``solved'' the problem at the expense of 
government creditors, i.e., by defaulting enough of its debt to bring  reduce interest payments down enough to balance the budget.

Somehow, in 1789, creditors of the French government were more powerful than they had been in 1720.

Therefore, King Louis XVI convened the Estates General together to ask them to redesign the French constitution in a way that would lower government expenditures or increase taxes, thereby
allowing him to balance the budget while also honoring his promises to creditors of the French government.  



The King called the Estates General together in an effort to promote the reforms that would
would bring sustained budget balance.  

{cite}`sargent_velde1995` describe how the French Revolutionaries set out to accomplish that.



 
## Remaking the tax code and tax administration

In 1789 the French Revolutionaries formed a National Assembly and set out to remake French
fiscal policy.

They wanted to honor government debts -- interests of French government creditors were well represented in the National Assembly.

But they set out to remake  the French tax code and the administrative machinery for collecting taxes.

  * they abolished all sorts of taxes
  * they abolished the Ancient Regimes scheme for ``tax farming''
      * tax farming meant that the government had privatized tax collection by hiring private citizes -- so called  tax farmers to collect taxes, while retaining a fraction of them as payment for their services
      * the great chemist Lavoisier was also a tax farmer, one of the reasons that the Committee for Public Safety sent him to the guillotine in 1794

As a consequence of these tax reforms, government tax revenues declined

The next figure shows this

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

According to Figure {numref}`fig5`, tax revenues per capita did not rise to their pre 1789 levels
until after 1815, when Napoleon Bonaparte was exiled to St Helena and King Louis XVIII was restored to the French Crown.



  * from 1799 to 1814, Napoleon Bonaparte had other sources of revenues -- booty and reparations from provinces and nations that he defeated in war

  * from 1789 to 1799, the French Revolutionaries turned to another source to raise resources to pay for government purchases of goods and services and to service French government debt. 

And as the next figure shows, government expenditures exceeded tax revenues by substantial
amounts during the period form 1789 to 1799.



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
To cover the disrepancies between government expenditures and tax revenues revealed in Figure {numref}`fig11`, the French revolutionaries  printed paper money and spent it.  

The next figure shows that by printing money, they were able to finance substantial purchases 
of goods and services, including military goods and soldiers' pay.



```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Revenues raised by printing paper money notes"
    name: fig24
---

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

Figure {numref}`fig24` compares the revenues raised by printing money from 1789 to 1796 with tax revenues that the Ancient Regime had raised in 1788.

Measured in goods, revenues raised at time $t$ by printing new money equal

$$
\frac{M_{t+1} - M_t}{p_t}
$$

where 

* $M_t$ is the stock of paper money at time $t$ measured in livres
* $p_t$ is the price level at time $t$ measured in units of goods per livre at time $t$
* $M_{t+1} - M_t$ is the amount of new money printed at time $t$



Notice the 1793-1794  surge in revenues raised by printing money. 

* this reflects extraordinary measures that the Committee for Public Safety adopted to force citizens to accept paper money, or else.

Also note the abrupt fall off in revenues raised by 1797 and the absence of further observations after 1797. 

* this reflects the end using the printing press to raise revenues.




What French paper money  entitled its holders to changed over time in interesting ways.

These  led to outcomes  that vary over time and that illustrate the playing out in practice of  theories that guided the Revolutionaries' monetary policy decisions.


The next figure shows the price level in France  during the time that the Revolutionaries used paper money to finance parts of their expenditures.

Note that we use a log scale because the price level rose so much.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Price Level and Price of Gold (log scale)"
    name: fig9
---

# Read the data from Excel file
data7 = pd.read_excel('datasets/assignat.xlsx', sheet_name='Data', usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel('datasets/assignat.xlsx', sheet_name='Data', usecols='L', skiprows=4, nrows=80, header=None)
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


We have partioned  {numref}`fig9` that shows the log of the price level and   Figure {numref}`fig8`
below  that plots real balances $\frac{M_t}{p_t}$ into three periods that correspond to 
to different monetary  experiments. 

The first period ends in the late summer of 1793, and is characterized
by growing real balances and moderate inflation. 

The second period begins and ends
with the Terror. It is marked by high real balances, around 2,500 millions, and
roughly stable prices. The fall of Robespierre in late July 1794 begins the third
of our episodes, in which real balances decline and prices rise rapidly.

We interpret
these three episodes in terms of three separate theories about money: a ``backing''
or ''real bills'' theory (the text is Adam Smith  {cite}`smith2010wealth`),
a legal restrictions theory ( {cite}`keynes1940pay`, {cite}`bryant1984price` )
and a classical hyperinflation theory ({cite}`Cagan`).%
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


The three clouds of points in Figure
{numref}`fig104`
 depict different real balance-inflation relationships. 
 
Only the cloud for the
third period has the inverse relationship familiar to us now from twentieth-century
hyperinflations.




* subperiod 1: ("real bills period): January 1791 to July 1793

* subperiod 2: ("terror:):  August 1793 - July 1794

* subperiod 3: ("classic Cagan hyperinflation): August 1794 - March 1796




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
---
mystnb:
  figure:
    caption: "Inflation and Real Balances"
    name: fig104
---
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


The three clouds of points in Figure
{numref}`fig104` evidently 
 depict different real balance-inflation relationships. 

Only the cloud for the
third period has the inverse relationship familiar to us now from twentieth-century
hyperinflations.


 To bring this out, we'll use linear regressions to draw straight lines that compress the 
 inflation-real balance relationship for our three sub periods. 

 Before we do that, we'll drop some of the early observations during the terror period 
 to obtain the following graph.
 

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
---
mystnb:
  figure:
    caption: "Inflation and Real Balances"
    name: fig104b
---
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



Now let's regress inflation on real balances during the real bills period and plot the regression
line.



```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Inflation and Real Balances"
    name: fig104c
---
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


Now let's regress real balances on inflation  during the terror  and plot the regression
line.


```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Inflation and Real Balances"
    name: fig104d
---
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

The following two graphs are for the classical hyperinflation period.

One regresses inflation on real balances, the other regresses real balances on inflation.

Both show a prounced inverse relationship that is the hallmark of the hyperinflations studied by 
Cagan {cite}`Cagan`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Inflation and Real Balances"
    name: fig104e
---
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
---
mystnb:
  figure:
    caption: "Inflation and Real Balances"
    name: fig104f
---
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


STUFF FROM SV 1995


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