---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(short_path)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Shortest Paths

```{index} single: Dynamic Programming; Shortest Paths
```

## Overview

The shortest path problem is a [classic problem](https://en.wikipedia.org/wiki/Shortest_path) in mathematics and computer science with applications in

* Economics (sequential decision making, analysis of social networks, etc.)
* Operations research and transportation
* Robotics and artificial intelligence
* Telecommunication network design and routing
* etc., etc.

Variations of the methods we discuss in this lecture are used millions of times every day, in applications such as

* Google Maps
* routing packets on the internet

For us, the shortest path problem also provides a nice introduction to the logic of **dynamic programming**.

Dynamic programming is an extremely powerful optimization technique that we apply in many lectures on this site.

The only scientific library we'll need in what follows is NumPy:

```{code-cell} ipython3
import numpy as np
```

## Outline of the problem

The shortest path problem is one of finding how to traverse a [graph](https://en.wikipedia.org/wiki/Graph_%28mathematics%29) from one specified node to another at minimum cost.

Consider the following graph

```{figure} /_static/lecture_specific/short_path/graph.png

```

We wish to travel from node (vertex) A to node G at minimum cost

* Arrows (edges) indicate the movements we can take.
* Numbers on edges indicate the cost of traveling that edge.

(Graphs such as the one above are called weighted [directed graphs](https://en.wikipedia.org/wiki/Directed_graph).)

Possible interpretations of the graph include

* Minimum cost for supplier to reach a destination.
* Routing of packets on the internet (minimize time).
* etc., etc.

For this simple graph, a quick scan of the edges shows that the optimal paths are

* A, C, F, G at cost 8

```{figure} /_static/lecture_specific/short_path/graph4.png

```

* A, D, F, G at cost 8

```{figure} /_static/lecture_specific/short_path/graph3.png

```

## Finding least-cost paths

For large graphs, we need a systematic solution.

Let $J(v)$ denote the minimum cost-to-go from node $v$, understood as the total cost from $v$ if we take the best route.

Suppose that we know $J(v)$ for each node $v$, as shown below for the graph from the preceding example.

```{figure} /_static/lecture_specific/short_path/graph2.png

```

Note that $J(G) = 0$.

The best path can now be found as follows

1. Start at node $v = A$
1. From current node $v$, move to any node that solves

```{math}
:label: spprebell

\min_{w \in F_v} \{ c(v, w) + J(w) \}
```

where

* $F_v$ is the set of nodes that can be reached from $v$ in one step.
* $c(v, w)$ is the cost of traveling from $v$ to $w$.

Hence, if we know the function $J$, then finding the best path is almost trivial.

But how can we find the cost-to-go function $J$?

Some thought will convince you that, for every node $v$,
the function $J$ satisfies

```{math}
:label: spbell

J(v) = \min_{w \in F_v} \{ c(v, w) + J(w) \}
```

This is known as the **Bellman equation**, after the mathematician [Richard Bellman](https://en.wikipedia.org/wiki/Richard_E._Bellman).

The Bellman equation can be thought of as a restriction that $J$ must
satisfy.

What we want to do now is use this restriction to compute $J$.

## Solving for minimum cost-to-go

Let's look at an algorithm for computing $J$ and then think about how to
implement it.

### The algorithm

The standard algorithm for finding $J$ is to start an initial guess and then iterate.

This is a standard approach to solving nonlinear equations, often called
the method of **successive approximations**.

Our initial guess will be

```{math}
:label: spguess

J_0(v) = 0 \text{ for all } v
```

Now

1. Set $n = 0$
1. Set $J_{n+1} (v) = \min_{w \in F_v} \{ c(v, w) + J_n(w) \}$ for all $v$
1. If $J_{n+1}$ and $J_n$ are not equal then increment $n$, go to 2

This sequence converges to $J$.

Although we omit the proof, we'll prove similar claims in our other lectures
on dynamic programming.

### Implementation

Having an algorithm is a good start, but we also need to think about how to
implement it on a computer.

First, for the cost function $c$, we'll implement it as a matrix
$Q$, where a typical element is

$$
Q(v, w)
=
\begin{cases}
   & c(v, w) \text{ if } w \in F_v \\
   & +\infty \text{ otherwise }
\end{cases}
$$

In this context $Q$ is usually called the **distance matrix**.

We're also numbering the nodes now, with $A = 0$, so, for example

$$
Q(1, 2)
=
\text{ the cost of traveling from B to C }
$$

For example, for the simple graph above, we set

```{code-cell} ipython3
from numpy import inf

Q = np.array([[inf, 1,   5,   3,   inf, inf, inf],
              [inf, inf, inf, 9,   6,   inf, inf],
              [inf, inf, inf, inf, inf, 2,   inf],
              [inf, inf, inf, inf, inf, 4,   8],
              [inf, inf, inf, inf, inf, inf, 4],
              [inf, inf, inf, inf, inf, inf, 1],
              [inf, inf, inf, inf, inf, inf, 0]])
```

Notice that the cost of staying still (on the principle diagonal) is set to

* `np.inf` for non-destination nodes --- moving on is required.
* 0 for the destination node --- here is where we stop.

For the sequence of approximations $\{J_n\}$ of the cost-to-go functions, we can use NumPy arrays.

Let's try with this example and see how we go:

```{code-cell} ipython3
nodes = range(7)                              # Nodes = 0, 1, ..., 6
J = np.zeros_like(nodes, dtype=int)        # Initial guess
next_J = np.empty_like(nodes, dtype=int)   # Stores updated guess

max_iter = 500
i = 0

while i < max_iter:
    for v in nodes:
        # Minimize Q[v, w] + J[w] over all choices of w
        next_J[v] = np.min(Q[v, :] + J)
    
    if np.array_equal(next_J, J):                
        break
    
    J[:] = next_J                                # Copy contents of next_J to J
    i += 1

print("The cost-to-go function is", J)
```

This matches with the numbers we obtained by inspection above.

But, importantly, we now have a methodology for tackling large graphs.

## Exercises


```{exercise-start}
:label: short_path_ex1
```

The file data below describes a weighted directed graph.

The line `node0, node1 0.04, node8 11.11, node14 72.21` means that from node0 we can go to

* node1 at cost 0.04
* node8 at cost 11.11
* node14 at cost 72.21

No other nodes can be reached directly from node0.

Other lines have a similar interpretation.

Your task is to use the algorithm given above to find the optimal path and its cost.

```{note}
You will be dealing with floating point numbers now, rather than
integers, so consider replacing `np.equal()` with `np.allclose()`.
```

```{code-cell} ipython3
import requests

file_url = "https://raw.githubusercontent.com/QuantEcon/lecture-python-intro/main/lectures/graph.txt"
graph_file_response = requests.get(file_url)
```

```{code-cell} ipython3
graph_file_data = str(graph_file_response.content, 'utf-8')
print(graph_file_data)
```

```{exercise-end}
```

```{solution-start} short_path_ex1
:class: dropdown
```

First let's write a function that reads in the graph data above and builds a distance matrix.

```{code-cell} ipython3
num_nodes = 100
destination_node = 99

def map_graph_to_distance_matrix(in_file_data):

    # First let's set of the distance matrix Q with inf everywhere
    Q = np.full((num_nodes, num_nodes), np.inf)

    # Now we read in the data and modify Q
    lines = in_file_data.split('\n')
    for line_ in lines:
        line = line_.strip()
        if line == '':
            continue
        elements = line.split(',')
        node = elements.pop(0)
        node = int(node[4:])    # convert node description to integer
        if node != destination_node:
            for element in elements:
                destination, cost = element.split()
                destination = int(destination[4:])
                Q[node, destination] = float(cost)
        Q[destination_node, destination_node] = 0
    return Q
```

In addition, let's write

1. a "Bellman operator" function that takes a distance matrix and current guess of J and returns an updated guess of J, and
1. a function that takes a distance matrix and returns a cost-to-go function.

We'll use the algorithm described above.

The minimization step is vectorized to make it faster.

```{code-cell} ipython3
def bellman(J, Q):
    return np.min(Q + J, axis=1)


def compute_cost_to_go(Q):
    num_nodes = Q.shape[0]
    J = np.zeros(num_nodes)      # Initial guess
    max_iter = 500
    i = 0

    while i < max_iter:
        next_J = bellman(J, Q)
        if np.allclose(next_J, J):
            break
        else:
            J[:] = next_J   # Copy contents of next_J to J
            i += 1

    return(J)
```

We used np.allclose() rather than testing exact equality because we are
dealing with floating point numbers now.

Finally, here's a function that uses the cost-to-go function to obtain the
optimal path (and its cost).

```{code-cell} ipython3
def print_best_path(J, Q):
    sum_costs = 0
    current_node = 0
    while current_node != destination_node:
        print(current_node)
        # Move to the next node and increment costs
        next_node = np.argmin(Q[current_node, :] + J)
        sum_costs += Q[current_node, next_node]
        current_node = next_node

    print(destination_node)
    print('Cost: ', sum_costs)
```

Okay, now we have the necessary functions, let's call them to do the job we were assigned.

```{code-cell} ipython3
Q = map_graph_to_distance_matrix(graph_file_data)
J = compute_cost_to_go(Q)
print_best_path(J, Q)
```

The total cost of the path should agree with $J[0]$ so let's check this.

```{code-cell} ipython3
J[0]
```

```{solution-end}
```
