---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Plotly Example

```{code-cell} ipython3
!pip install plotly
```

An example plot

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "A Plotly Scatter Plot"
    name: plotlyfig
---
# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()
```


```{only} latex
This figure is interactive you may [click here to see this figure on the website](https://intro.quantecon.org/plotly.html#plotlyfig)
```