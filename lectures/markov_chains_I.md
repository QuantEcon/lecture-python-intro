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

+++ {"user_expressions": []}

# Markov Chains I

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

+++ {"user_expressions": []}

## Overview

Markov chains are a standard way to model time series with some dependence
between observations.

For example,

* inflation next year depends on inflation this year
* unemployment next month depends on unemployment this month

Markov chains are one of the workhorse models of economics and finance.

The theory of Markov chains is beautiful and provides many insights into
probability and dynamics.

In this introductory lecture, we will

* review some of the key ideas from the theory of Markov chains and
* show how Markov chains appear in some economic applications.

Let's start with some standard imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
import quantecon as qe
import numpy as np
from graphviz import Digraph
```

+++ {"user_expressions": []}

## Definitions and Examples

In this section we provide the basic definitions and some elementary examples.

(finite_dp_stoch_mat)=
### Stochastic Matrices

Recall that a **probability mass function** over $n$ possible outcomes is a
nonnegative $n$-vector $p$ that sums to one.

For example, $p = (0.2, 0.2, 0.6)$ is a probability mass function over $3$ outcomes.

A **stochastic matrix** (or **Markov matrix**)  is an $n \times n$ square matrix $P$
such that each row of $P$ is a probability mass function over $n$ outcomes.

In other words,

1. each element of $P$ is nonnegative, and
1. each row of $P$ sums to one

If $P$ is a stochastic matrix, then so is the $k$-th power $P^k$ for all $k \in \mathbb N$.

Checking this is {ref}`one of the exercises <mc_ex_pk>` below.


### Markov Chains

Now we can introduce Markov chains.

First we will give some examples and then we will define them more carefully.

At that time, the connection between stochastic matrices and Markov chains
will become clear.


(mc_eg2)=
#### Example 1

From  US unemployment data, Hamilton {cite}`Hamilton2005` estimated the following dynamics.

```{code-cell} ipython3
:tags: [hide-input]

dot = Digraph(comment='Graph')
dot.attr(rankdir='LR')
dot.node("ng")
dot.node("mr")
dot.node("sr")

dot.edge("ng", "ng", label="0.971")
dot.edge("ng", "mr", label="0.029")
dot.edge("mr", "ng", label="0.145")

dot.edge("mr", "mr", label="0.778")
dot.edge("mr", "sr", label="0.077")
dot.edge("sr", "mr", label="0.508")

dot.edge("sr", "sr", label="0.492")
dot
```

+++ {"user_expressions": []}

Here there are three **states**

* "ng" represents normal growth
* "mr" represents mild recession
* "sr" represents severe recession

The arrows represent **transition probabilities** over one month.

For example, the arrow from mild recession to normal growth has 0.145 next to it.

This tells us that, according to past data, there is a 14.5% probability of transitioning from mild recession to normal growth in one month.

The arrow from normal growth back to normal growth tells us that there is a
97% probability of transitioning from normal growth to normal growth (staying
in the same state).

Note that these are *conditional* probabilities --- the probability of
transitioning from one state to another (or staying at the same one) conditional on the
current state.

To make the problem easier to work with numerically, let's convert states to
numbers.

In particular, we agree that

* state 0 represents normal growth
* state 1 represents mild recession
* state 2 represents severe recession

Let $X_t$ record the value of the state at time $t$.

Now we can write the statement "there is a 14.5% probability of transitioning from mild recession to normal growth in one month" as

$$
    \mathbb P\{X_{t+1} = 0 \,|\, X_t = 1\} = 0.145
$$

We can collect all of these conditional probabilities into a matrix, as follows

$$
    P =
    \left(
      \begin{array}{ccc}
         0.971 & 0.029 & 0 \\
         0.145 & 0.778 & 0.077 \\
         0 & 0.508 & 0.492
      \end{array}
    \right)
$$

Notice that $P$ is a stochastic matrix.

Now we have the following relationship

$$
    P(i,j)
    = \mathbb P\{X_{t+1} = j \,|\, X_t = i\}
$$

This holds for any $i,j$ between 0 and 2.

In particular, $P(i,j)$ is the
     probability of transitioning from state $i$ to state $j$ in one month.




(mc_eg1)=
#### Example 2

Consider a worker who, at any given time $t$, is either unemployed (state 0)
or employed (state 1).

Suppose that, over a one month period,

1. the unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. the employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.

Given the above information, we can write out the transition probabilities in matrix form as

```{math}
:label: p_unempemp

P
= \left(
\begin{array}{cc}
    1 - \alpha & \alpha \\
    \beta & 1 - \beta
\end{array}
  \right)
```

For example,

$$
\begin{aligned}
    P(0,1)
        & =
        \text{ probability of transitioning from state $0$ to state $1$ in one month}
        \\
        & =
        \text{ probability finding a job next month}
        \\
        & = \alpha
\end{aligned}
$$

Suppose we can estimate the values $\alpha$ and $\beta$.

Then we can address a range of questions, such as

* What is the average duration of unemployment?
* Over the long-run, what fraction of the time does a worker find herself unemployed?
* Conditional on employment, what is the probability of becoming unemployed at least once over the next 12 months?

We'll cover some of these applications below.



### Defining Markov Chains

So far we've given examples of Markov chains but now let's define them more
carefully.

To begin, let $S$ be a finite set $\{x_1, \ldots, x_n\}$ with $n$ elements.

The set $S$ is called the **state space** and $x_1, \ldots, x_n$ are the **state values**.

A **distribution** $\psi$ on $S$ is a probability mass function of length $n$, where $\psi(i)$ is the amount of probability allocated to state $x_i$.

A **Markov chain** $\{X_t\}$ on $S$ is a sequence of random variables taking values in $S$ 
that have the **Markov property**.

This means that, for any date $t$ and any state $y \in S$,

```{math}
:label: fin_markov_mp

\mathbb P \{ X_{t+1} = y  \,|\, X_t \}
= \mathbb P \{ X_{t+1}  = y \,|\, X_t, X_{t-1}, \ldots \}
```

In other words, knowing the current state is enough to know probabilities for the future states.

In particular, the dynamics of a Markov chain are fully determined by the set of values

```{math}
:label: mpp

P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}
\qquad (x, y \in S)
```

By construction,

* $P(x, y)$ is the probability of going from $x$ to $y$ in one unit of time (one step)
* $P(x, \cdot)$ is the conditional distribution of $X_{t+1}$ given $X_t = x$

We can view $P$ as a stochastic matrix where

$$
    P_{ij} = P(x_i, x_j)
    \qquad 1 \leq i, j \leq n
$$

Going the other way, if we take a stochastic matrix $P$, we can generate a Markov
chain $\{X_t\}$ as follows:

* draw $X_0$ from a distribution $\psi_0$ on $S$
* for each $t = 0, 1, \ldots$, draw $X_{t+1}$ from $P(X_t,\cdot)$

By construction, the resulting process satisfies {eq}`mpp`.




## Simulation

```{index} single: Markov Chains; Simulation
```

One natural way to answer questions about Markov chains is to simulate them.

Let's start by doing this ourselves and then look at libraries that can help
us.

In these exercises, we'll take the state space to be $S = 0,\ldots, n-1$.

(We start at $0$ because Python arrays are indexed from $0$.)


### Writing Our Own Simulation Code

To simulate a Markov chain, we need 

1. a stochastic matrix $P$ and 
1. a probability mass function $\psi_0$ of length $n$ from which to draw a initial realization of $X_0$.

The Markov chain is then constructed as follows:

1. At time $t=0$, draw a realization of $X_0$ from the distribution $\psi_0$.
1. At each subsequent time $t$, draw a realization of the new state $X_{t+1}$ from $P(X_t, \cdot)$.

(That is, draw from row $X_t$ of $P$.)

To implement this simulation procedure, we need a method for generating draws
from a discrete distribution.

For this task, we'll use `random.draw` from [QuantEcon](http://quantecon.org/quantecon-py).

To use `random.draw`, we first need to convert the probability mass function
to a cumulative distribution

```{code-cell} ipython3
ψ_0 = (0.3, 0.7)           # probabilities over {0, 1}
cdf = np.cumsum(ψ_0)       # convert into cumulative distribution
qe.random.draw(cdf, 5)   # generate 5 independent draws from ψ
```

+++ {"user_expressions": []}

We'll write our code as a function that accepts the following three arguments

* A stochastic matrix `P`
* An initial distribution `ψ_0`
* A positive integer `ts_length` representing the length of the time series the function should return

```{code-cell} ipython3
def mc_sample_path(P, ψ_0=None, ts_length=1_000):

    # set up
    P = np.asarray(P)
    X = np.empty(ts_length, dtype=int)

    # Convert each row of P into a cdf
    n = len(P)
    P_dist = np.cumsum(P, axis=1)  # Convert rows into cdfs

    # draw initial state, defaulting to 0
    if ψ_0 is not None:
        X_0 = qe.random.draw(np.cumsum(ψ_0))
    else:
        X_0 = 0

    # simulate
    X[0] = X_0
    for t in range(ts_length - 1):
        X[t+1] = qe.random.draw(P_dist[X[t], :])

    return X
```

+++ {"user_expressions": []}

Let's see how it works using the small matrix

```{code-cell} ipython3
P = [[0.4, 0.6],
     [0.2, 0.8]]
```

+++ {"user_expressions": []}

Here's a short time series.

```{code-cell} ipython3
mc_sample_path(P, ψ_0=[1.0, 0.0], ts_length=10)
```

+++ {"user_expressions": []}

It can be shown that for a long series drawn from `P`, the fraction of the
sample that takes value 0 will be about 0.25.

(We will explain why {ref}`below <ergodicity>`.)

Moreover, this is true regardless of the initial distribution from which
$X_0$ is drawn.

The following code illustrates this

```{code-cell} ipython3
X = mc_sample_path(P, ψ_0=[0.1, 0.9], ts_length=1_000_000)
np.mean(X == 0)
```

+++ {"user_expressions": []}

You can try changing the initial distribution to confirm that the output is
always close to 0.25 (for the `P` matrix above).


### Using QuantEcon's Routines

[QuantEcon.py](http://quantecon.org/quantecon-py) has routines for handling Markov chains, including simulation.

Here's an illustration using the same $P$ as the preceding example

```{code-cell} ipython3
from quantecon import MarkovChain

mc = qe.MarkovChain(P)
X = mc.simulate(ts_length=1_000_000)
np.mean(X == 0)
```

+++ {"user_expressions": []}

The `simulate` routine is faster (because it is [JIT compiled](https://python-programming.quantecon.org/numba.html#numba-link)).

```{code-cell} ipython3
%time mc_sample_path(P, ts_length=1_000_000) # Our homemade code version
```

```{code-cell} ipython3
%time mc.simulate(ts_length=1_000_000) # qe code version
```

+++ {"user_expressions": []}

#### Adding State Values and Initial Conditions

If we wish to, we can provide a specification of state values to `MarkovChain`.

These state values can be integers, floats, or even strings.

The following code illustrates

```{code-cell} ipython3
mc = qe.MarkovChain(P, state_values=('unemployed', 'employed'))
mc.simulate(ts_length=4, init='employed')
```

```{code-cell} ipython3
mc.simulate(ts_length=4, init='unemployed')
```

```{code-cell} ipython3
mc.simulate(ts_length=4)  # Start at randomly chosen initial state
```

+++ {"user_expressions": []}

If we want to see indices rather than state values as outputs as  we can use

```{code-cell} ipython3
mc.simulate_indices(ts_length=4)
```

+++ {"user_expressions": []}

(mc_md)=
## Distributions over Time

We learnt that

1. $\{X_t\}$ is a Markov chain with stochastic matrix $P$
1. the distribution of $X_t$ is known to be $\psi_t$

What then is the distribution of $X_{t+1}$, or, more generally, of $X_{t+m}$?

To answer this, we let $\psi_t$ be the distribution of $X_t$ for $t = 0, 1, 2, \ldots$.

Our first aim is to find $\psi_{t + 1}$ given $\psi_t$ and $P$.

To begin, pick any $y  \in S$.

To get the probability of being at $y$ tomorrow (at $t+1$), we account for
all ways this can happen and sum their probabilities.

This leads to

$$
\mathbb P \{X_{t+1} = y \}
   = \sum_{x \in S} \mathbb P \{ X_{t+1} = y \, | \, X_t = x \}
               \cdot \mathbb P \{ X_t = x \}
$$



(We are using the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability).)

Rewriting this statement in terms of  marginal and conditional probabilities gives

$$
    \psi_{t+1}(y) = \sum_{x \in S} P(x,y) \psi_t(x)
$$

There are $n$ such equations, one for each $y \in S$.

If we think of $\psi_{t+1}$ and $\psi_t$ as *row vectors*, these $n$ equations are summarized by the matrix expression

```{math}
:label: fin_mc_fr

\psi_{t+1} = \psi_t P
```

Thus, to move a distribution forward one unit of time, we postmultiply by $P$.

By postmultiplying $m$ times, we move a distribution forward $m$ steps into the future.

Hence, iterating on {eq}`fin_mc_fr`, the expression $\psi_{t+m} = \psi_t P^m$ is also valid --- here $P^m$ is the $m$-th power of $P$.

As a special case, we see that if $\psi_0$ is the initial distribution from
which $X_0$ is drawn, then $\psi_0 P^m$ is the distribution of
$X_m$.

This is very important, so let's repeat it

```{math}
:label: mdfmc

X_0 \sim \psi_0 \quad \implies \quad X_m \sim \psi_0 P^m
```

The general rule is that post-multiplying a distribution by $P^m$ shifts it forward $m$ units of time.

Hence the following is also valid.

```{math}
:label: mdfmc2

X_t \sim \psi_t \quad \implies \quad X_{t+m} \sim \psi_t P^m
```

+++ {"user_expressions": []}

(finite_mc_mstp)=
### Multiple Step Transition Probabilities

We know that the probability of transitioning from $x$ to $y$ in
one step is $P(x,y)$.

It turns out that the probability of transitioning from $x$ to $y$ in
$m$ steps is $P^m(x,y)$, the $(x,y)$-th element of the
$m$-th power of $P$.

To see why, consider again {eq}`mdfmc2`, but now with a $\psi_t$ that puts all probability on state $x$.

Then $\psi_t$ is a vector with $1$ in position $x$ and zero elsewhere.

Inserting this into {eq}`mdfmc2`, we see that, conditional on $X_t = x$, the distribution of $X_{t+m}$ is the $x$-th row of $P^m$.

In particular

$$
\mathbb P \{X_{t+m} = y \,|\, X_t = x \} = P^m(x, y) = (x, y) \text{-th element of } P^m
$$


### Example: Probability of Recession

```{index} single: Markov Chains; Future Probabilities
```

Recall the stochastic matrix $P$ for recession and growth {ref}`considered above <mc_eg2>`.

Suppose that the current state is unknown --- perhaps statistics are available only  at the *end* of the current month.

We guess that the probability that the economy is in state $x$ is $\psi_t(x)$ at time t.

The probability of being in recession (either mild or severe) in 6 months time is given by 

$$
(\psi_t P^6)(1) + (\psi_t P^6)(2)
$$

+++ {"user_expressions": []}

(mc_eg1-1)=
### Example 2: Cross-Sectional Distributions

The distributions we have been studying can be viewed either 

1. as probabilities or 
1. as cross-sectional frequencies that a Law of Large Numbers leads us to anticipate for  large samples.

To illustrate, recall our model of employment/unemployment dynamics for a given worker {ref}`discussed above <mc_eg1>`.

Consider a large population of workers, each of whose lifetime experience is
described by the specified dynamics, with each worker's outcomes being
realizations of processes that are statistically independent of all other
workers' processes.

Let $\psi_t$ be the current *cross-sectional* distribution over $\{ 0, 1 \}$.

The cross-sectional distribution records fractions of workers employed and unemployed at a given moment t.

* For example, $\psi_t(0)$ is the unemployment rate.

What will the cross-sectional distribution be in 10 periods hence?

The answer is $\psi_t P^{10}$, where $P$ is the stochastic matrix in
{eq}`p_unempemp`.

This is because each worker's state evolves according to $P$, so
$\psi_t P^{10}$ is a marginal distribution  for a single randomly selected
worker.

But when the sample is large, outcomes and probabilities are roughly equal (by an application of the Law
of Large Numbers).

So for a very large (tending to infinite) population,
$\psi_t P^{10}$ also represents  fractions of workers in
each state.

This is exactly the cross-sectional distribution.

```{code-cell} ipython3
dot = Digraph(comment='Graph')
dot.attr(rankdir='LR')
dot.node("Growth")
dot.node("Stagnation")
dot.node("Collapse")

dot.edge("Growth", "Growth", label="0.68")
dot.edge("Growth", "Stagnation", label="0.12")
dot.edge("Growth", "Collapse", label="0.20")

dot.edge("Stagnation", "Stagnation", label="0.24")
dot.edge("Stagnation", "Growth", label="0.50")
dot.edge("Stagnation", "Collapse", label="0.26")

dot.edge("Collapse", "Collapse", label="0.46")
dot.edge("Collapse", "Stagnation", label="0.18")
dot.edge("Collapse", "Growth", label="0.36")

dot
```

```{code-cell} ipython3
nodes = ['DG', 'DS', 'DC', 'AG', 'AS', 'AC']
trans_matrix = [[0.72, 0.11, 0.11, 0.05, 0.00, 0.01],
                [0.53, 0.26, 0.08, 0.06, 0.00, 0.02],
                [0.42, 0.21, 0.25, 0.06, 0.00, 0.06],
                [0.05, 0.00, 0.00, 0.63, 0.10, 0.22],
                [0.03, 0.03, 0.00, 0.42, 0.21, 0.31],
                [0.05, 0.01, 0.01, 0.26, 0.14, 0.53]]
```

```{code-cell} ipython3
import networkx as nx
from matplotlib import cm
import matplotlib as mpl

G = nx.MultiDiGraph()
edge_ls = []
label_dict = {}

for start_idx, node_start in enumerate(nodes):
    for end_idx, node_end in enumerate(nodes):
        value = trans_matrix[start_idx][end_idx]
        if value != 0:
            G.add_edge(node_start,node_end, weight=value, len=100)
            
pos = nx.spring_layout(G, seed=10)
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size=600, edgecolors='black', node_color='white')
nx.draw_networkx_labels(G, pos)

arc_rad = 0.2
curved_edges = [edge for edge in G.edges() if (edge[1], edge[0]) in G.edges()]
edges = nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f'arc3, rad = {arc_rad}', edge_cmap=cm.Greys, width=2,
    edge_color=[G[nodes[0]][nodes[1]][0]['weight'] for nodes in G.edges])

pc = mpl.collections.PatchCollection(edges, cmap=cm.Greys)

ax = plt.gca()
ax.set_axis_off()
plt.colorbar(pc, ax=ax)
plt.show()
```

```{code-cell} ipython3
import networkx as nx

G = nx.MultiDiGraph()
edge_ls = []
label_dict = {}

for start_idx, node_start in enumerate(nodes):
    for end_idx, node_end in enumerate(nodes):
        value = trans_matrix[start_idx][end_idx]
        if value != 0:
            G.add_edge(node_start,node_end, weight=value, len=100)
            
pos = nx.spring_layout(G, seed=10)
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size=600, edgecolors='black', node_color='white')
nx.draw_networkx_labels(G, pos)

arc_rad = 0.2
curved_edges = [edge for edge in G.edges() if (edge[1], edge[0]) in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f'arc3, rad = {arc_rad}', edge_cmap=cm.Blues, width=2,
    edge_color=[G[nodes[0]][nodes[1]][0]['weight'] for nodes in G.edges])

pc = mpl.collections.PatchCollection(edges, cmap=cm.Blues)

ax = plt.gca()
ax.set_axis_off()
plt.colorbar(pc, ax=ax)
plt.show()
```
