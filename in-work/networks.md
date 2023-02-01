# Networks

## Outline

In recent years there has been rapid growth in a field called [network science](https://en.wikipedia.org/wiki/Network_science).

Network science studies relationships between groups of objects.

One important example is the world wide web, where web pages are connected by hyperlinks.

Another is the human brain: studies of brain function emphasize the network of
connections between nerve cells (neurons).

Artificial neural networks are based on this idea, using data to build
intricate connections between simple processing units.

Biologists studying transmission of diseases like COVID-19 have to consider
the interactions between human hosts.

In operations research, network analysis is related to such fundamental
problems as on minimum cost flow, traveling salesman, shortest path, and
assignment.

TODO add wiki links for above defs

This lecture gives an introduction to economic and financial networks.

(Some parts are drawn from the text https://networks.quantecon.org/ but the
level of this lecture is less advanced.)


## Economic and Financial Networks

Within economics, important examples of networks include

* financial networks
* production networks 
* trade networks
* transport networks and 
* social networks   

For example, social networks affect trends in market sentiment and consumer decisions.  

The structure of financial networks helps to determine relative fragility of the financial system.

The structure of production networks affects trade, innovation and the propagation of local shocks.

Let's look at some examples in more depth.


### Example: Aircraft Exports

TODO -- add commercial aircraft network figure from https://networks.quantecon.org/ch_intro.html but hide the code

Figure [TODO add numref] shows international trade in large commercial aircraft in 2019 based on International Trade Data SITC Revision 2.  

The circles in the figure are called **nodes** or **vertices** -- in this case they represent countries.

The arrows in the figure are called **edges** or **links**.

Node size is proportional to total exports and edge width is proportional to exports to the target country.

The data is for trade in commercial aircraft weighing at least 15,000kg and was sourced from CID Dataverse.

The figure shows that the US, France and Germany are major export hubs.  

In the discussion below, we learn to quantify such ideas.


### Example: A Markov Chain

Recall that, in our lecture on Markov chains (TODO add link to new Markov
chain lecture markov_chains.md) we studied a dynamic model of business cycles
where the states are

* "ng" = "normal growth"
* "mr" = "mild recession"
* "sr" = "severe recession"

TODO add the figure here

This is an example of a network, where the set of nodes $V$ equals the states:

$$
    V = \{ \text{"ng", "mr", "sr"} \}
$$

The edges between the nodes show the one month transition probabilities.





## An Introduction to Graph Theory

Network science is built on top of a major branch of mathematics called [graph
theory](https://en.wikipedia.org/wiki/Graph_theory).

To understand and analyze networks, we need some understanding of graph theory.

Graph theory can be complicated and we will cover only the basics.

However, these concepts will already be enough for us to discuss interesting and
important ideas on economic and financial networks.

We focus on "directed" graphs, where connects are one way rather that symmetric

E.g.,

* bank $A$ lends money to bank $B$
* firm $A$ supplies goods to firm $B$



### Key Definitions

A **directed graph** consists of 
%
* a finite set $V$ and
* a collection of pairs $(u, v)$ where $u$ and $v$ are elements of $V$

The elements of $V$ are called the **vertices** or **nodes** of the graph.

In the aircraft export example above, the set $V$ is all countries included in the data set.

The pairs $(u,v)$ are called the **edges** of the graph and the set of all edges will usually be denoted by $E$

Intuitively and visually, an edge $(u,v)$ is understood as an arrow from vertex $u$ to vertex $v$.  

(A neat way to represent an arrow is to record the location of the tail and
head of the arrow, and that's exactly what an edge does.)

In the aircraft export example above, $E$ is all the arrows in the figure,
each indicating some positive amount of aircraft exports from one country to
another.

Let's look at more examples.

Two graphs are shown below, each with three vertices.  

TODO -- convert these to use https://h1ros.github.io/posts/introduction-to-graphviz-in-jupyter-notebook/

:label: rich_poor_no_label
digraph { 
    rankdir=LR;
    "poor" -> "poor" ;
    "poor" -> "middle class" ;
    "middle class" -> "poor" ;
    "middle class" -> "middle class" ;
    "middle class" -> "rich" ;
    "rich" -> "poor" ;
    "rich" -> "middle class" ;
    "rich" -> "rich" ;
} 


:label: poverty_trap
digraph { 
    rankdir=LR;
    "poor" -> "poor" ;
    "middle class" -> "poor" ;
    "middle class" -> "middle class" ;
    "middle class" -> "rich" ;
    "rich" -> "poor" ;
    "rich" -> "middle class" ;
    "rich" -> "rich" ;
} 


For these graphs, the arrows (edges) could be thought of as representing positive possibility of transition over a given unit of time.  

In general, for a given edge $(u, v)$, the vertex $u$ is called a **direct predecessor** of $v$
and $v$ is called a **direct successor** of $u$.  

Also,  for $v \in V$,

* the **in-degree** is $i_d(v) = $ the number of direct predecessors of $v$ and
* the **out-degree** is $o_d(v) = $ the number of direct successors of $v$.



### Digraphs in Networkx

The Python package [Networkx](https://networkx.org/) provides a convenient
data structure for representing directed graphs and implements many common routines
for analyzing them.

To import it into Python we run

\begin{minted}{python}
import networkx as nx
\end{minted}

In all of the code snippets shown below, we assume readers have executed this
import statement, as well as 

\begin{minted}{python}
import numpy as np
import matplotlib.pyplot as plt
\end{minted}

As an example, let us create the directed graph in TODO poverty_trap.

To do so, we first create an empty `DiGraph` object:

\begin{minted}{python}
G_p = nx.DiGraph()    
\end{minted}

Next we populate it with nodes and edges.  

To do this we write down a list of
all edges, with \texttt{poor} represented by \texttt{p} and so on:

\begin{minted}{python}
edge_list = [
    ('p', 'p'),
    ('m', 'p'), ('m', 'm'), ('m', 'r'),
    ('r', 'p'), ('r', 'm'), ('r', 'r')
]
\end{minted}

Finally, we add the edges to our \texttt{DiGraph} object:

\begin{minted}{python}
for e in edge_list:
    u, v = e
    G_p.add_edge(u, v)
\end{minted}

Adding the edges automatically adds the nodes, so \texttt{G\_p} is now a
correct representation of our graph.

We can verify this by plotting the graph via Networkx with the following code: 

\begin{minted}{python}
fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True, 
                 font_weight='bold', arrows=True, alpha=0.8,
                 connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()
\end{minted}

This code produces Figure TODO [fix ref], which matches the
original directed graph in Figure [fix ref]


\texttt{DiGraph} objects have methods that calculate in-degree and out-degree
of vertices.   

For example,
%
\begin{minted}{python}
G_p.in_degree('p')
\end{minted}
%





### Communication

Next we study communication and connectedness, which have important implications for economic networks.

Vertex $v$ is called **accessible** from vertex $u$ if either $u=v$ or
there exists a sequence of edges that lead from $u$ to $v$.  

* in this case, we write $u \to v$

(Visually, there is a sequence of arrows leading from $u$ to $v$.)

For example, suppose we have a directed graph representing a production network, where 

* elements of $V$ are industrial sectors and
* existence of an edge $(i, j)$ means that $i$ supplies products or services to $j$.  

Then sector $m$ is an upstream supplier of sector $\ell$ whenever $m \to \ell$.

Two vertices $u$ and $v$ are said to **communicate** if $u \to v$ and $v \to u$.

A graph is called **strongly connected** if any two nodes in $V$ are
accessible from each other.

TODO -- fix this

\begin{example}
    In Figure~\ref{f:rich_poor_no_label}, the directed graph is strongly connected.  In
    contrast, in Figure~\ref{f:poverty_trap}, rich is not accessible from
    poor, so the graph is not strongly connected.  
\end{example}

Networkx can be used to test for strong connectedness.

TODO give an example





## Weighted Graphs

Figure~\ref{f:financial_network_analysis_visualization} shows flows of funds
(i.e., loans) between private banks, grouped by country of origin.

TODO --- Add this from https://networks.quantecon.org/ch_intro.html but hide the code

An arrow from Japan to the US indicates aggregate claims held by Japanese
banks on all US-registered banks, as collected by the Bank of International
Settlements (BIS). 

The size of each node in the figure is increasing in the
total foreign claims of all other nodes on this node. 

The widths of the arrows are proportional to the foreign claims they represent.

TODO convert table to markdown

\begin{table}
    \small
    \centering
    %\fontsize{9.5pt}{10.25pt}\selectfont
    \addtolength{\tabcolsep}{-2pt}
    \centering
    \begin{tabular}{ll|ll|ll|ll}
        \hline 
        AU  & Australia  & DE  & Germany & CL  & Chile & ES & Spain \\
        PT  &  Portugal & FR & France & TR  & Turkey & GB & United Kingdom \\
        US  & United States & IE & Ireland & AT  & Austria & IT & Italy \\
        BE  & Belgium & JP & Japan & SW & Switzerland & SE & Sweden \\
        \hline
    \end{tabular}
    \caption{\label{table:cfn} Codes for the 16-country financial network}
\end{table}


In this network, an edge $(u, v)$ exists for almost every choice of $u$ and
$v$ (i.e., almost every country in the network).

Hence existence of an edge is not particularly informative.  

To understand the network, we need to record not just the existence or absence
of a credit flow, but also the size of the flow.

The correct data structure for recording this information is a ``weighted
directed graph''


### Definitions



A **weighted directed graph** is a directed graph with vertices $V$ and edges
$E$ to which we have added a **weight function** $w$ that assigns a positive
number to each edge.

For example, Figure~\ref{f:rich_poor} shows a weighted directed graph, with arrows
representing edges of the induced directed graph.

:label: rich_poor
digraph { 
    rankdir=LR;
    "poor" -> "poor" [label = "0.9"];
    "poor" -> "middle class" [label = "0.1"];
    "middle class" -> "poor" [label = "0.4"];
    "middle class" -> "middle class" [label = "0.4"];
    "middle class" -> "rich" [label = "0.2"];
    "rich" -> "poor" [label = "0.1"];
    "rich" -> "middle class" [label = "0.1"];
    "rich" -> "rich" [label = "0.8"];
} 

The numbers next to the edges are the weights.  

In this case, you can think of the numbers on the arrows as transition
probabilities for a household over, say, one year.  

We see that a rich household has a 10\% chance of becoming poor in one year.  


## Adjacency Matrices 

The **adjacency matrix** of a weighted directed graph $(V, E, w)$ with vertices $\{v_1, \ldots, v_n\}$ is the matrix
%
\begin{equation*}
    A = (a_{ij})_{1 \leq i,j \leq n}
    \quad \text{with} \quad
    a_{ij} =
    %
    \begin{cases}
        w(v_i, v_j) & \text{ if } (v_i, v_j) \in E
        \\
        0           & \text{ otherwise}.
    \end{cases}
    %
\end{equation*}
%

Once the vertices in $V$ are enumerated, the weight function and
adjacency matrix provide essentially the same information.  

We often work with the latter, since it facilitates computations.

For example, with $\{$poor, middle, rich$\}$ mapped to $(0, 1, 2)$, 
    the adjacency matrix corresponding to the weighted directed graph in
    Figure~\ref{f:rich_poor} is
    %
    \begin{equation}\label{eq:fnegwa0}
        A = 
        \begin{pmatrix}
            0.9 & 0.1 & 0 \\
            0.4 & 0.4 & 0.2 \\
            0.1 & 0.1 & 0.8
        \end{pmatrix}.
    \end{equation}

In QuantEcon's \texttt{DiGraph} implementation, weights are recorded via the
keyword \texttt{weighted}:

\begin{minted}{python}
A = ((0.9, 0.1, 0.0),
     (0.4, 0.4, 0.2),
     (0.1, 0.1, 0.8))
A = np.array(A)
G = qe.DiGraph(A, weighted=True)    
\end{minted}


One of the key points to remember about adjacency matrices is that taking the
transpose ``reverses all the arrows'' in the associated directed graph.  

For example,  the directed graph in Figure~\ref{f:network_liabfin} can be interpreted as a
    stylized version of a financial network, with vertices as banks and edges
    showing flow of funds, similar to
    Figure~\ref{f:financial_network_analysis_visualization}.

We see that bank 2 extends a loan of size 200 to bank 3.

The corresponding adjacency matrix is
%
\begin{equation}\label{eq:fnegwa}
    A = 
    \begin{pmatrix}
        0 & 100 & 0 & 0 & 0 \\
        50 & 0 & 200 & 0 & 0 \\
        0 & 0 & 0 & 100 & 0 \\
        0 & 500 & 0 & 0 & 50 \\
        150 & 0 & 250 & 300 & 0 
    \end{pmatrix}.
\end{equation}
%

The transposition is
%
\begin{equation}\label{eq:fnegwat}
    A^\top = 
    \begin{pmatrix}
        0   & 50  & 0   & 0   & 150 \\
        100 & 0   & 0   & 500 & 0 \\
        0   & 200 & 0   & 0   & 250 \\
        0   & 0   & 100 & 0   & 300 \\
        0   & 0   & 0   & 50  & 0 
    \end{pmatrix}.
\vspace{0.3em}
\end{equation}
%

The corresponding network is visualized in Figure~\ref{f:network_liabfin_trans}.  

This figure shows the network of liabilities after the loans have been granted.

Both of these networks (original and transpose) are useful for analysis of
    financial markets.

TODO convert tikz code below to graphviz? Or use some other solution? original figs are in networks text section 1.4.2

:label: network_liabfin

\begin{tikzpicture}
  \node[circle, draw] (1) at (2.5, 3) {1};
  \node[circle, draw] (2) at (-1, 2) {2};
  \node[circle, draw] (3) at (-2, -0.5) {3};
  \node[circle, draw] (4) at (1.5, -1) {4};
  \node[circle, draw] (5) at (3.5, 0) {5};
  \draw[->, thick, black]
  (1) edge [bend left=20, below] node {$100$} (2)
  (2) edge [bend left=20, above] node {$50$} (1)
  (2) edge [bend right=20, left] node {$200$} (3)
  (3) edge [bend right=20, below] node {$100$} (4)
  (4) edge [bend right=20, right] node {$500$} (2)
  (5) edge [bend right=20, below left] node {$250$} (3)
  (5) edge [bend left=30, below] node {$300$} (4)
  (4) edge [bend left=30, below] node {$50$} (5)
  (5) edge [bend right=30, right] node {$150$} (1);
\end{tikzpicture}

:label: network_liabfin_trans


\begin{tikzpicture}
  \node[circle, draw] (1) at (2.5, 3) {1};
  \node[circle, draw] (2) at (-1, 2) {2};
  \node[circle, draw] (3) at (-2, -0.5) {3};
  \node[circle, draw] (4) at (1.5, -1) {4};
  \node[circle, draw] (5) at (3.5, 0) {5};
  \draw[<-, thick, black]
  (1) edge [bend left=20, below] node {$100$} (2)
  (2) edge [bend left=20, above] node {$50$} (1)
  (2) edge [bend right=20, left] node {$200$} (3)
  (3) edge [bend right=20, below] node {$100$} (4)
  (4) edge [bend right=20, right] node {$500$} (2)
  (5) edge [bend right=20, below left] node {$250$} (3)
  (5) edge [bend left=30, below] node {$300$} (4)
  (4) edge [bend left=30, below] node {$50$} (5)
  (5) edge [bend right=30, right] node {$150$} (1);
\end{tikzpicture}


In general, every nonnegative $n \times n$ matrix $A = (a_{ij})$ can be
viewed as the adjacency matrix of a weighted directed graph.

To build the graph we set $V = 0, \ldots, n-1$ and take the edge set $E$ to be
all $(i,j)$ such that $a_{ij} > 0$.

For the weight function we set $w(i, j) = a_{ij}$  for all edges $(i,j)$.

We call this graph the weighted directed graph induced by $A$.




## Properties

Consider a weighted directed graph with adjacency matrix $A$.

Let $a^k_{ij}$ be element $i,j$ of $A^k$, the $k$-th power of $A$.

The following result is useful in many applications:

TODO -- theorem environment

For distinct vertices $i, j$ in $V$ and any integer $k$, we have
%
\begin{equation*}
    a^k_{i j} > 0
    \; \iff \;
    \text{ $j$ is accessible from $i$}.
\end{equation*}
%

The above result is obvious when $k=1$ and a proof of the general case can be
found in \cite{sargent2022economic}.

Now recall from [TODO add link to Maanasee's eigenvalues lecture] that a
nonnegative matrix is called irreducible if [give def].

From the proceeding theorem it is not too difficult (see
\cite{sargent2022economic} for details) to get the next result.

TODO thm environment

For a weighted directed graph.  The following statements are equivalent:
   
1. The directed graph is strongly connected. 
1. The adjacency matrix of the graph is irreducible.
 
        

TODO add a simple example where we can see the theorem is working




## Network Centrality

When studying networks of all varieties, a recurring topic is the relative
"centrality" or "importance" of different nodes. 

Examples include

* ranking of web pages by search engines  
* determining the most important bank in a financial network (which one a
  central bank should rescue if there is a financial crisis)
* determining the most important industrial sector in an economy.



In what follows, a **centrality measure** associates to each weighted directed
graph a vector $m$ where the $m_i$ is interpreted as the centrality (or rank)
of vertex $v_i$.  

### Degree Centrality

Two elementary measures of ``importance'' of a vertex in a given directed
graph are its in-degree and out-degree.

Both of these provide a centrality measure.  

In-degree centrality is a vector containing the in-degree of each vertice in
the graph.

In-degree centrality is a vector containing the in-degree of each vertice in
the graph.

If the graph is expressed as a Networkx \texttt{DiGraph} called \texttt{G}
(see, e.g., \S\ref{sss:nx}), then the in-degree centrality vector can be
calculated via
%
\begin{minted}{python}
iG = [G.in_degree(v) for v in G.nodes()]    
\end{minted}

Unfortunately, while these measures of centrality are simple to calculate, they are
not always informative.  

For example, consider the task of a web search engine, which ranks pages
by relevance whenever a user enters a search.

Suppose web page A has twice as many inbound links as page B.  

This might suggest that page A deserves a higher rank, due to its larger in-degree centrality

But in fact page A might be less "important" than page B.

To see why, suppose that the links to A are from pages that almost no traffic,
while the links to B are from pages that receive very heavy traffic.

In this case, page B probably receives more visitors, which in turn suggests
that page B contains more valuable (or entertaining) content. 

Thinking about this point suggests that importance might be *recursive*.

What this means is that the importance of a given node depends on the
importance of other nodes that link to it.  

The next centrality measure we turn to has this recursive property.


### Eigenvector Centrality

Suppose we have a weighted directed graph with adjacency matrix $A$.

For simplicity we will suppose that the vertices $V$ of the graph are just the integers $1, \ldots, n$.

Let $r(A)$ denote the spectral radius of $A$.  [TODO link to Maanasee's lecture on eigenvalues]

The **eigenvector centrality** of the graph is defined as the $n$-vector $e$ that solves

$$
:label: ev_central
    e = \frac{1}{r(A)} A e.
$$

To better understand this, we write out the full expression for some element $e_i$

%
\begin{equation}\label{eq:eicen}
    e_i = \frac{1}{r(A)} \sum_{1 \leq j \leq n} a_{ij} e_j
\end{equation}


Note the recursive nature of the definition: the centrality obtained by vertex
$i$ is proportional to a sum of the centrality of all vertices, weighted by
the ``rates of flow'' from $i$ into these vertices.   

A vertex $i$ is highly ranked if (a) there are many edges leaving $i$, (b)
these edges have large weights, and (c) the edges point to other highly ranked
vertices.

Later, when we study demand shocks production networks, we will provide a more
concrete interpretation of eigenvector centrality.  

We will see that, in production networks, sectors with high eigenvector
centrality are important suppliers.  

In particular, they are activated by a wide array of demand shocks once orders flow backwards through the network.


To compute eigenvector centrality we can use the following function.


import numpy as np
from scipy.sparse import linalg

def eigenvector_centrality(A, m=40):
    """
    Computes and normalizes the dominant eigenvector of A.  
    """
    r, vec_r = linalg.eigs(A, k=1, which='LR')
    e = vec_r.flatten().real
    return e / np.sum(e)  # normalize e


TODO 

- add simple example and compute eigenvector centrality.
- revisit financial networks fig financial_network_analysis_visualization and discuss eigenvector centrality.

Countries that are rated highly according to this rank tend to be important players in terms of supply
of credit. 

Japan takes the highest rank according to this measure, although
countries with large financial sectors such as Great Britain and France are
not far behind.  



### Katz Centrality

One problem with eigenvector centrality is that $r(A)$ might be zero, in which
case $1/r(A)$ is not defined.

For this and other reasons, some researchers prefer another measure of
centrality for networks called Katz centrality. 

Fixing $\beta$ in $(0, 1/r(A))$, the
**Katz centrality** of a weighted directed graph 
with adjacency matrix $A$ is defined as the
vector $\kappa$ that solves
%

$$
:label: katz_central
    \kappa_i =  \beta \sum_{1 \leq j 1} a_{ij} \kappa_j + 1
    \qquad  \text{for all } i \in \{0, \ldots, n-1}.
\end{equation}

Here $\beta$ is a parameter that we can choose.

In vector form we can write 

$$
:label: katz_central_vec
    \kappa = \mathbf 1 + \beta A \kappa
$$


where $\mathbf 1$ is a column vector of ones.

The intuition behind this centrality measure is similar to that provided for
eigenvector centrality: high centrality is conferred on $i$ when it is linked
to by vertices that themselves have high centrality.  

Provided that $0 < \beta < 1/r(A)$, Katz centrality is always finite and well-defined
because then $r(\beta A) < 1$.

This means that eq:katz_central_vec has the unique solution

$$
    \kappa = (I - \beta A)^{-1} \mathbf 1
$$


This follows from the Neumann series theorem [TODO link to Maanasee's lecture].

The parameter $\beta$ is used to ensure that $\kappa$ is finite

When $r(A)<1$, we use $\beta=1$ as the default for Katz centrality computations.  





### Authorities vs Hubs

Search engine designers recognize that web pages can be important in two
different ways.  

Some pages have high **hub centrality**, meaning that they link to valuable
sources of information (e.g., news aggregation sites).  

Other pages have high **authority centrality**, meaning that they contain
valuable information, as indicated by the number and significance of incoming
links (e.g., websites of respected news organizations).  

Similar ideas can and have been applied to economic networks (often using
different terminology).  

The eigenvector centrality and Katz centrality measures we discussed above
measure hub centrality.

If we care more about authority centrality, we can use the same definitions
except that we take the transpose of the adjacency matrix.

For example, the **authority-based eigenvector centrality** of a weighted directed graph with adjacency matrix $A$
is the vector $e$ solving

\begin{equation}\label{eq:eicena0}
    e = \frac{1}{r(A)} A^\top e.
\end{equation}

The only difference from the original definition is that $A$ is replaced by
its transpose.

(Transposes do not affect the spectral radius of a matrix so we wrote $r(A)$ instead of $r(A^\top)$.)

Element-by-element, this is
%
\begin{equation}\label{eq:eicena}
    e_j = \frac{1}{r(A)} \sum_{1 \leq i \leq n} a_{ij} e_i
\end{equation}

We see $e_j$ will be high if many nodes with high authority rankings link to $j$.

The middle right panel of Figure~\ref{f:financial_network_analysis_centrality}
shows the authority-based eigenvector centrality ranking for the international
credit network shown in
Figure~\ref{f:financial_network_analysis_visualization}.  


Highly ranked countries are those that attract large inflows of credit, or
credit inflows from other major players.  

The US clearly dominates the rankings as a target of interbank credit.  





## Further Reading

Textbooks on economic and social networks include \cite{jackson2010social},
\cite{easley2010networks}, \cite{borgatti2018analyzing},
\cite{sargent2022economic}, and \cite{goyal2023networks}.

@article{sargent2022economic,
  title={Economic Networks: Theory and Computation},
  author={Sargent, Thomas J and Stachurski, John},
  journal={arXiv preprint arXiv:2203.11972},
  year={2022}
}


@book{borgatti2018analyzing,
  title={Analyzing social networks},
  author={Borgatti, Stephen P and Everett, Martin G and Johnson, Jeffrey C},
  year={2018},
  publisher={Sage}
}

@book{jackson2010social,
  title={Social and economic networks},
  author={Jackson, Matthew O},
  year={2010},
  publisher={Princeton university press}
}

@book{easley2010networks,
  title={Networks, crowds, and markets},
  author={Easley, David and Kleinberg, Jon and others},
  volume={8},
  year={2010},
  publisher={Cambridge university press Cambridge}
}

@book{goyal2023networks,
  title={Networks: An economics approach},
  author={Goyal, Sanjeev},
  year={2023},
  publisher={MIT Press}
}



Within the realm of network science, the texts
by \cite{newman2018networks}, \cite{menczer2020first} and
\cite{coscia2021atlas} are excellent.


@book{newman2018networks,
  title={Networks},
  author={Newman, Mark},
  year={2018},
  publisher={Oxford university press}
}

@book{menczer2020first,
  title={A first course in network science},
  author={Menczer, Filippo and Fortunato, Santo and Davis, Clayton A},
  year={2020},
  publisher={Cambridge University Press}
}

@article{coscia2021atlas,
  title={The atlas for the aspiring network scientist},
  author={Coscia, Michele},
  journal={arXiv preprint arXiv:2101.00863},
  year={2021}
}


## Exercises


### Ex

Here is a mathematical exercise for those who like proofs.

Let $(V, E)$ be a directed graph and write $u \sim v$ if $u$ and $v$ communicate.  

Show that $\sim$ is an [equivalence relation](https://en.wikipedia.org/wiki/Equivalence_relation) on $V$.


