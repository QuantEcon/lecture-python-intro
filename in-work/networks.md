# Networks

## Outline

In recent years there has been rapid growth in a field called *network science*.  

Network science studies links (or relationships) between groups of objects.

One of the most important examples is the world wide web, where web pages are
connected by hyperlinks.

Analysis of the human brain emphasizes the network of connections between
nerve cells (called neurons).

Artificial neural networks are based on this idea, using data to build
intricate connections between simple processing units.

Biologists studying transmission of diseases like COVID-19 have to consider
the connections and interactions between human hosts.

In operations research, network analysis is related to such fundamental
problems as on minimum cost flow, traveling salesman, shortest path, and
assignment.

TODO add wiki links for above defs

Within economics, important examples of networks include

* financial networks
* production networks 
* trade networks
* transport networks and 
* social networks   

For example, social networks affect trends in
market sentiment and consumer decisions.  

The structure of financial networks helps to determine relative fragility of the
system

The structure of production networks affects trade, innovation and the
propagation of local shocks.

TODO -- add figure f:commercial_aircraft_2019_1 from networks -- ask JS 

Figure [TODO add numref] shows international trade in large
commercial aircraft in 2019 based on International Trade Data SITC Revision 2.  

The circles in the figure are called **nodes** or **vertices** -- in this case
they represent countries.

The arrows in the figure are called **edges** or **links**.

Node size is proportional to total exports and edge width is proportional to
exports to the target country.

The data is for trade in commercial aircraft weighing at least 15,000kg
and was sourced from CID Dataverse.

The figure shows that the US, France and Germany are major export hubs.  


## Graph Theory

Network analysis is built on top of a major branch of mathematics called
graph theory.

Graph theory is often regarded as originating from work by the brilliant Swiss
mathematician Leonhard Euler (1707--1783), including his famous paper on the
``Seven Bridges of K\"onigsberg.''

TODO -- add links for above sentence

We now give a short introduction to graph theory.

We focus on "directed" graphs, where connects are one way rather that
symmetric

* bank $A$ lends money to bank $B$
* firm $A$ supplies goods to firm $B$, etc.


(This costs no generality, since undirected graphs, where relationships are
symmetric two-way connections, can be studied by imposing symmetry --
existence of a connection from $A$ to $B$ implies existence of a connection
from $B$ to $A$).  


### Unweighted Graphs


A \navy{directed graph}\index{Directed graph} or \navy{digraph}\index{Digraph}
is a pair $\gG = (V, E)$, where
%
\begin{itemize}
    \item $V$ is a finite nonempty set and 
    \item $E$ is a collection of ordered pairs $(u, v) \in V \times V$ called
        \navy{edges}\index{Edges}.
\end{itemize}
%
Elements of $V$ are called  the \navy{vertices}\index{Vertices} or
\navy{nodes}\index{Nodes} of $\gG$. Intuitively and visually, an edge $(u,v)$
is understood as an arrow from vertex $u$ to vertex $v$.  

Two graphs are given in
Figures~\ref{f:rich_poor_no_label}--\ref{f:poverty_trap}.  Each graph has
three vertices.  In these cases, the arrows (edges) could be thought of as
representing positive possibility of transition over a given unit of time.  


\begin{figure}
   \begin{center}
       \scalebox{1.0}{\input{tikz/rich_poor_no_label.tex}}
   \end{center}
   \caption{\label{f:rich_poor_no_label} A digraph of classes}
\end{figure}

\begin{figure}
   \centering
   \scalebox{1.0}{\input{tikz/poverty_trap.tex}}
   \caption{\label{f:poverty_trap} An alternative edge list}
\end{figure}

For a given edge $(u, v)$, the vertex $u$ is called the \navy{tail}\index{Tail
of an edge} of the
edge, while $v$ is called the \navy{head}\index{Head of an edge}.  Also,
$u$ is called a \navy{direct predecessor}\index{Direct predecessor} of $v$ and $v$ is called a 
\navy{direct successor}\index{Direct successor} of $u$.  For $v \in V$, we use the following notation:
%
\begin{itemize}
    \item $\iI(v) :=$  the set of all direct predecessors of $v$
    \item $\oO(v) :=$  the set of all direct successors of $v$
\end{itemize}
%
Also, the \navy{in-degree}\index{In-degree} and \navy{out-degree}\index{Out-degree} of $v \in V$ are defined by
%
\begin{itemize}
    \item the $i_d(v) := |\iI(v)|$ and
    \item the $o_d(v) := |\oO(v)|$ respectively.
\end{itemize}

If $i_d(v)=0$ and $o_d(v) > 0$, then $v$ is called a \navy{source}\index{Source}.  If either
$\oO(v)=\emptyset$ or $\oO(v)=\{v\}$, then 
$v$ is called a \navy{sink}\index{Sink}.  For example, in Figure~\ref{f:poverty_trap},
``poor'' is a sink with an in-degree of 3.


\subsubsection{Digraphs in Networkx}\label{sss:nx}

Both Python and Julia provide valuable interfaces to numerical computing with
graphs.  Of these libraries, the Python package Networkx is probably the most
mature and fully developed.  It provides a convenient data structure for
representing digraphs and implements many common routines for analyzing them.
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

As an example, let us create the digraph in Figure~\ref{f:poverty_trap}, which
we denote henceforth by $\gG_p$.  To do so, we first create an empty
\texttt{DiGraph} object:

\begin{minted}{python}
G_p = nx.DiGraph()    
\end{minted}

Next we populate it with nodes and edges.  To do this we write down a list of
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
correct representation of $\gG_p$.  For our small digraph we can verify this
by plotting the graph via Networkx with the following code: 

\begin{minted}{python}
fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True, 
                 font_weight='bold', arrows=True, alpha=0.8,
                 connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()
\end{minted}

This code produces Figure~\ref{f:networkx_basics_1}, which matches the
original digraph in Figure~\ref{f:poverty_trap}.

\begin{figure}
   \centering
   \scalebox{0.6}{\includegraphics[trim = 10mm 10mm 0mm 10mm, clip]{figures/networkx_basics_1.pdf}}
   \caption{\label{f:networkx_basics_1} Networkx digraph plot}
\end{figure}

\texttt{DiGraph} objects have methods that calculate in-degree and out-degree
of vertices.   For example,
%
\begin{minted}{python}
G_p.in_degree('p')
\end{minted}
%
prints 3.


\subsubsection{Communication}\label{sss:coms}

Next we study communication and connectedness, which have important
implications for production, financial, transportation and other networks, as
well as for dynamic properties of Markov chains.  

A \navy{directed walk}\index{Directed walk} from vertex $u$ to vertex $v$  of
a digraph $\gG$ is a finite sequence of vertices, starting with $u$ and ending
with $v$, such that any consecutive pair in the sequence is an edge of $\gG$.
A \navy{directed path}\index{Directed path} from $u$ to $v$ is a directed walk
from $u$ to $v$ such that all vertices in the path are distinct.  
For example, in Figure~\ref{f:strong_connected_components}, $(3, 2, 3, 2, 1)$
is a directed walk from $3$ to $1$ but not a directed path,
while $(3, 2, 1)$ is both a directed path and a directed walk from $3$ to $1$.

As is standard, the \navy{length}\index{Length (directed walk)} of a directed walk (or path) counts the
number of edges rather than vertices.   For example, the directed path $(3, 2,
1)$ from $3$ to $1$ in Figure~\ref{f:strong_connected_components} is said to have length 2.

Vertex $v$ is called \navy{accessible}\index{Accessible} (or \navy{reachable}) from vertex $u$, and
we write \navy{$u \to v$}, if either $u=v$ or there exists a directed path
from $u$ to $v$.  A set $U \subset V$ is called
\navy{absorbing}\index{Absorbing} for the directed graph $(V, E)$ if no
element of $V \setminus U$ is accessible from $U$.

\begin{example}
    Let $\gG = (V, E)$ be a digraph representing a production network, where
    elements of $V$ are sectors and $(i, j) \in E$ means that $i$ supplies
    products or services to
    $j$.  Then sector $m$ is an upstream supplier of sector $\ell$ whenever $m
    \to \ell$.
\end{example}

\begin{figure}
   \begin{center}
       \input{tikz/strong_connected_components.tex}
   \end{center}
   \caption{\label{f:strong_connected_components} Strongly connected
   components of a digraph (rectangles)}
\end{figure}

\begin{example}
    The vertex $\{ \text{poor} \}$ in the Markov digraph displayed in
    Figure~\ref{f:poverty_trap} is absorbing, since  $\{ \text{middle, rich}
    \}$ is not accessible from  $\{ \text{poor} \}$.
\end{example}

Two vertices $u$ and $v$ are said to \navy{communicate}\index{Communication
(graphs)} if $u \to v$ and $v \to u$.

\begin{Exercise}\label{ex:sceqrel}
    Let $(V, E)$ be a directed graph and write $u \sim v$ if $u$ and $v$
    communicate.  Show that $\sim$ is an equivalence relation (see  \S\ref{sss:eqclass}).
\end{Exercise}

Since communication is an equivalence relation, it induces a partition of $V$
into a finite collection of equivalence classes.  Within each of these
classes, all elements communicate.  These classes are called \navy{strongly
connected components}.  The graph itself is called \navy{strongly
connected}\index{Strongly connected} if there is only one such component; that
is, $v$ is accessible from $u$ for any pair $(u, v) \in V \times V$.  This
corresponds to the idea that any node can be reached from any other.  

\begin{example}
    Figure~\ref{f:strong_connected_components} shows a digraph with 
    strongly connected components $\{1\}$ and $\{2, 3\}$.
    The digraph is not strongly connected.
\end{example}

\begin{example}
    In Figure~\ref{f:rich_poor_no_label}, the digraph is strongly connected.  In
    contrast, in Figure~\ref{f:poverty_trap}, rich is not accessible from
    poor, so the graph is not strongly connected.  
\end{example}

Networkx can be used to test for communication and strong connectedness, as
well as to compute strongly connected components.  For example, applied to the
digraph in Figure~\ref{f:strong_connected_components}, the code
%
\begin{minted}{python}
G = nx.DiGraph()
G.add_edge(1, 1)
G.add_edge(2, 1)
G.add_edge(2, 3)
G.add_edge(3, 2)

list(nx.strongly_connected_components(G)) 
\end{minted}
%
prints \texttt{[\{1\}, \{2, 3\}]}.



Early quantitative work on networks tended to focus on unweighted digraphs,
where the existence or absence of an edge is treated as sufficient information
(e.g., following or not following on social media, existence or absence of a
road connecting two towns). However, for some networks, this binary measure is
less significant than the size or strength of the connection.  

As one illustration, consider
Figure~\ref{f:financial_network_analysis_visualization}, which shows flows of
funds (i.e., loans) between private banks, grouped by country of origin.  An
arrow from Japan to the US, say, indicates aggregate claims held by Japanese
banks on all US-registered banks, as collected by the Bank of International
Settlements (BIS). The size of each node in the figure is increasing in the
total foreign claims of all other nodes on this node. The widths of the arrows
are proportional to the foreign claims they represent.\footnote{Data for the
    figure was obtained from the BIS consolidated
    banking statistics, for Q4 of 2019. Our calculations used the immediate
    counterparty basis for financial claims of domestic and foreign banks,
    which calculates the sum of cross-border claims and local claims of
    foreign affiliates in both foreign and local currency. The foreign claim
    of a node to itself is set to zero.}
The country codes are given in Table~\ref{table:cfn}.


\begin{figure}
   \centering
   \scalebox{0.9}{\includegraphics[trim = 20mm 20mm 0mm 20mm, clip]{
   figures/financial_network_analysis_visualization.pdf}}
   \caption{\label{f:financial_network_analysis_visualization} International
   private credit flows by country}
\end{figure}


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
$v$ (i.e., almost every country in the network).\footnote{In fact arrows
    representing foreign claims less than US\$10 million are cut from
    Figure~\ref{f:financial_network_analysis_visualization}, so the network is
even denser than it appears.}  Hence existence of an edge is not
particularly informative.  To understand the network, we need to record not
just the existence or absence of a credit flow, but also the size of the flow.
The correct data structure for recording this information is a ``weighted
directed graph,'' or ``weighted digraph.'' In this section we define this
object and investigate its properties.


### Weighted Graphs


A \navy{weighted digraph}\index{Weighted digraph} $\gG$ is a triple $(V, E, w)$
such that $(V, E)$ is a digraph and $w$ is a function from $E$ to $(0,
\infty)$, called the \navy{weight function}\index{weight function}.

\begin{remark}
    Weights are traditionally regarded as nonnegative. In this text we insist
    that weights are also positive, in the sense that $w(u, v) > 0$ for all
    $(u, v) \in E$.  The reason is that the intuitive notion of zero weight is
    understood, here and below, as absence of a connection.  In other words,
    if $(u, v)$ has ``zero weight,'' then $(u, v)$ is not in $E$, so $w$ is
    not defined on $(u, v)$.
\end{remark}

\begin{example}
    As suggested by the discussion above, 
    the graph shown in Figure~\ref{f:financial_network_analysis_visualization}
    can be viewed as a weighted digraph.  Vertices are countries of origin
    and an edge exists between country $u$ and country $v$ when private banks
    in $u$ lend nonzero quantities to banks in $v$.  The weight assigned to
    edge $(u, v)$ gives total loans from $u$ to $v$ as measured according to
    the discussion of Figure~\ref{f:financial_network_analysis_visualization}.
\end{example}

\begin{example}
    Figure~\ref{f:rich_poor} shows a weighted digraph, with arrows 
    representing edges of the induced digraph (compare with the unweighted
    digraph in Figure~\ref{f:rich_poor_no_label}). The numbers next to
    the edges are the weights.  In this case, you can think of the numbers
    on the arrows as transition probabilities for a household over, say, one
    year.  For example, a rich household has a 10\% chance of becoming poor.  
\end{example}

\begin{figure}
   \begin{center}
       \scalebox{1.0}{\input{tikz/rich_poor.tex}}
   \end{center}
   \caption{\label{f:rich_poor} A weighted digraph}
\end{figure}


The definitions of \navy{accessibility}, \navy{communication},
and \navy{connectedness} extend to any weighted digraph $\gG
= (V, E, w)$ by applying them to $(V, E)$.  

For example,
$(V, E, w)$ is called strongly connected if $(V, E)$ is strongly connected. The
weighted digraph in Figure~\ref{f:rich_poor} is strongly connected.


\subsubsection{Adjacency Matrices of Weighted Digraphs}

The \navy{adjacency matrix}\index{Adjacency matrix} of a \emph{weighted}
digraph $(V, E, w)$ with vertices $\{v_1, \ldots, v_n\}$ is the matrix
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

\begin{example}
    With $\{$poor, middle, rich$\}$ mapped to $(1, 2, 3)$, 
    the adjacency matrix corresponding to the weighted digraph in
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
\end{example}

In QuantEcon's \texttt{DiGraph} implementation, weights are recorded via the
keyword \texttt{weighted}:

\begin{minted}{python}
A = ((0.9, 0.1, 0.0),
     (0.4, 0.4, 0.2),
     (0.1, 0.1, 0.8))
A = np.array(A)
G = qe.DiGraph(A, weighted=True)    # Store weights
\end{minted}



One of the key points to remember about adjacency matrices is that taking the
transpose ``reverses all the arrows'' in the associated digraph.  

\begin{example}
    The digraph in Figure~\ref{f:network_liabfin} can be interpreted as a
    stylized version of a financial network, with vertices as banks and edges
    showing flow of funds, similar to
    Figure~\ref{f:financial_network_analysis_visualization} on
    page~\pageref{f:financial_network_analysis_visualization}.  For example,
    we see that bank 2 extends a loan of size 200 to bank 3.
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
    The corresponding network is visualized in
    Figure~\ref{f:network_liabfin_trans}.  This figure shows the network of
    liabilities after the loans have been granted.
    Both of these networks (original and transpose) are useful for analysis of
    financial markets (see, e.g., Chapter~\ref{c:fpms}).
\end{example}

\begin{figure}
   \begin{center}
    \input{tikz/network_liabfin.tex}
    \caption{\label{f:network_liabfin} A network of credit flows across institutions}
   \end{center}
\end{figure}

\begin{figure}
   \begin{center}
    \input{tikz/network_liabfin_trans.tex}
    \caption{\label{f:network_liabfin_trans} The transpose: a network of liabilities}
   \end{center}
\end{figure}

It is not difficult to see that each nonnegative $n \times n$ matrix $A = (a_{ij})$ can be
viewed as the adjacency matrix of a weighted digraph with vertices equal
to $\natset{n}$.  The weighted digraph $\gG = (V, E, w)$ in question is 
formed by setting 
%
\begin{equation*}
    V = \natset{n},
    \quad
    E = \setntn{(i,j) \in V \times V}{ a_{ij} > 0}
    \quad \text{and} \quad
    w(i, j) = a_{ij} \text{ for all } (i,j) \in E.
\end{equation*}
%
We call $\gG$ the \navy{weighted digraph induced by $A$}.

The next exercise helps to reinforce the point that transposes reverse the
edges.

\begin{Exercise}\label{ex:rar}
    Let $A= (a_{ij})$ be a nonnegative $n \times n$ matrix and let $\gG =
    (\natset{n}, E, w)$ and $\gG' = (\natset{n}, E', w')$ be the weighted
    digraphs induced by $A$ and $A^\top$, respectively.  Show that 
    %
    \begin{enumerate}
        \item $(j, k) \in E'$ if and only if $(k, j) \in E$.
        \item $j \to k$ in $\gG'$ if and only if $k \to j$ in $\gG$.
    \end{enumerate}

\end{Exercise}

\begin{Answer}
    Let $A^{\top} = (a'_{ij})$, so that $a'_{ij} = a_{ji}$ for each $i, j$.  
    By definition, we have 
    %
    \begin{equation*}
        (j, k) \in E' 
        \; \iff \;
        a'_{jk} > 0
        \; \iff \;
        a_{kj} > 0
        \; \iff \;
        (k, j) \in E,
    \end{equation*}
    %
    which proves (i).  Regarding (ii), to say that 
    $k$ is accessible from $j$ in $\gG'$ means that we can find vertices
    $i_1, \ldots, i_m$ that form a directed path from $j$ to $k$ under $\gG'$, in the sense that 
    such that $i_1 = j$, $i_m=k$, and each successive
    pair $(i_{\ell}, i_{\ell+1})$ is in $E'$.  But then, by (i), 
    $i_m, \ldots, i_1$ provides a directed path from $k$ to $j$ under $\gG$,
    since and each successive pair $(i_{\ell+1}, i_{\ell})$ is in
    $E$.
\end{Answer}




## Properties

In this section, we examine some of the fundamental properties of and
relationships among digraphs, weight functions and adjacency matrices.
Throughout this section, the vertex set $V$ of any graph we examine will be
set to $\natset{n}$.  This costs no loss of generality, since, in this
text, the vertex set of a digraph is always finite and nonempty.

Also, while we refer to weighted digraphs for their additional generality, the
results below connecting adjacency matrices and digraphs are valid for
unweighted digraphs.  Indeed, an unweighted digraph $\gG = (V, E)$ can be mapped
to a weighted digraph by introducing a weight function that maps each element
of $E$ to unity.  The resulting adjacency matrix agrees with our original
definition for unweighted digraphs in \eqref{eq:digrapham}.

As an additional convention, if $A$ is an adjacency matrix, and $A^k$ is the
$k$-th power of $A$, then we write $a^k_{ij}$ for a typical element of $A^k$. 
With this notation, we observe that, since
$A^{(s+t)} = A^s A^t$, the rules of matrix multiplication imply
%
\begin{equation}\label{eq:accip}
    a^{s+t}_{ij}
    = \sum_{\ell=1}^n a^s_{i \ell} \, a^t_{\ell j}
    \qquad (i, j \in \natset{n}, \;\; s,t \in \NN).
\end{equation}
%
($A^0$ is the identity.) The next proposition explains the significance of the
powers.


\begin{proposition}\label{p:accesspos}
    Let $\gG$ be a weighted digraph with adjacency matrix $A$. For distinct
    vertices $i, j \in \natset{n}$ and $k \in \NN$, we have
    %
    \begin{equation*}
        a^k_{i j} > 0
        \; \iff \;
        \text{ there exists a directed walk of length $k$ from $i$ to $j$}.
    \end{equation*}
    %
\end{proposition}

\begin{proof}
    ($\Leftarrow$ ).  The statement is true by definition when $k=1$.  Suppose
    in addition that $\Leftarrow$ holds at $k-1$, and suppose
    there exists a directed walk $(i, \ell, m, \ldots, n, j)$ of length $k$
    from $i$ to $j$.  By the induction hypothesis we have $a^{k-1}_{i n} >
    0$.  Moreover, $(n, j)$ is part of a directed walk, so $a_{n j} > 0$.  Applying
    \eqref{eq:accip} now gives $a^k_{i j} > 0$.

    ($\Rightarrow$).  Left as an exercise (just use the same logic).
\end{proof}


\begin{example}
    In \S\ref{s:amcs} we show that if
    elements of $A$ represent
    one-step transition probabilities across states, then elements of $A^t$,
    the $t$-th power of $A$, provide $t$-step transition probabilities.  
    In Markov process theory, \eqref{eq:accip} is called the
    \emph{Chapman--Kolmogorov equation}.
\end{example}


In this context, the next result is fundamental.

\begin{theorem}\label{t:sconir}
    Let $\gG$ be a weighted digraph.  The following statements are equivalent:
    %
    \begin{enumerate}
        \item $\gG$ is strongly connected. 
        \item The adjacency matrix generated by $\gG$ is irreducible.
    \end{enumerate}
    %
\end{theorem}

\begin{proof}
    Let $\gG$ be a weighted digraph with adjacency matrix $A$.   By
    Proposition~\ref{p:accesspos}, strong connectedness of $\gG$ is equivalent
    to the statement that, for each $i, j \in V$, we can find a $k \geq 0$
    such that $a^k_{ij} > 0$. (If $i=j$  then set $k=0$.) This, in turn, is
    equivalent to $\sum_{m=0}^\infty A^m \gg 0$, which is irreducibility of
    $A$.
\end{proof}

\begin{example}
    Strong connectivity fails in the digraph in Figure~\ref{f:io_reducible}, since
    vertex 4 is a source.  By
    Theorem~\ref{t:sconir}, the adjacency matrix must be reducible.
\end{example}


\begin{figure}
   \begin{center}
    \input{tikz/io_reducible.tex}
    \caption{\label{f:io_reducible} Failure of strong connectivity}
   \end{center}
\end{figure}

We will find that the property of being primitive is valuable for analysis.
(The Perron--Frobenius Theorem hints at this.)  What do we need to add to strong
connectedness to obtain primitiveness?


\begin{theorem}\label{t:scaperpr}
    For a weighted digraph $\gG=(V, E, w)$, the following statements are equivalent:
    %
    \begin{enumerate}
        \item $\gG$ is strongly connected and aperiodic. 
        \item The adjacency matrix generated by $\gG$ is primitive.
    \end{enumerate}
    %
\end{theorem}


\begin{proof}[Proof of Theorem~\ref{t:scaperpr}]
    Throughout the proof we set $V=\natset{n}$.
    First we show that, if $\gG$ is aperiodic and strongly connected, then, for all $i,
    j \in V$, there exists a $q \in \NN$ such that $a^k_{ij} > 0$
    whenever $k \geq q$.  To this end, pick any $i,j$ in $V$.   Since $\gG$ is
    strongly connected,
    there exists an $s \in \NN$ such that $a^s_{ij} > 0$.  Since $\gG$ is
    aperiodic, we can find an $m \in \NN$ such that $\ell \geq m$ implies
    $a^\ell_{jj} > 0$.  Picking $\ell \geq m$ and applying~\eqref{eq:accip},
    we have
    %
    \begin{equation*}
        a^{s+\ell}_{ij}
        = \sum_{r \in V} a^s_{i r} a^\ell_{r j}
        \geq  a^s_{ij} a^\ell_{jj}
        > 0.
    \end{equation*}
    %
    Thus, with $t = s + m$, we have $a^k_{ij} > 0$ whenever $k \geq t$.

    ((i) $\Rightarrow$ (ii)).  By the preceding argument, given any $i, j \in V$,
    there exists an $s(i,j) \in \NN$ such that $a^m_{ij} > 0$ whenever $m
    \geq s(i,j)$.  Setting $k := \max s(i, j)$ over all $(i,j)$ yields
    $A^k \gg 0$.

    ((ii) $\Rightarrow$ (i)). Suppose that $A$ is primitive.  Then, for some
    $k \in \NN$, we have $A^k \gg 0$.  Strong connectedness of the digraph
    follows directly from Proposition~\ref{p:accesspos}.  It remains to check
    aperiodicity.

    Aperiodicity will hold if we can establish that $a^{k+t}_{ii} > 0$ for all $t \geq 0$.
    To show this, it suffices to show that $A^{k + t}  \gg 0$ for all $t \geq 0$.
    Moreover, to prove the latter, we need only show that $A^{k + 1}  \gg 0$,
    since the claim then follows from induction.

    To see that $A^{k+1} \gg 0$, observe that, for any given
    $i, j$, the relation \eqref{eq:accip} implies
    %
    \begin{equation*}
        a^{k+1}_{ij}
        = \sum_{\ell \in V} a_{i\ell} a^k_{\ell j}
        \geq \bar a  \sum_{\ell \in V} a_{i \ell}.
    \end{equation*}
    %
    where $\bar a := \min_{\ell \in V} a^k_{\ell j} > 0$.
    The proof will be done if $\sum_{\ell \in V} a_{i \ell} > 0$.  But this
    must be true, since otherwise vertex $i$ is a sink, which contradicts
    strong connectedness.
\end{proof}


\begin{example}
    In Exercise~\ref{ex:pwprop} we worked hard to show that $P_w$ is
    irreducible if and only if $0 < \alpha, \beta \leq 1$, using the approach
    of calculating and then examining the powers of $P_w$ (as shown in
    \eqref{eq:pwpk}).  However, the result is trivial when we examine
    the corresponding digraph in Figure~\ref{f:worker_switching} and use the
    fact that irreducibility is equivalent to strong connectivity.
    Similarly, the result in Exercise~\ref{ex:pwprop} that $P_w$ is
    primitive if and only if $0 < \alpha, \beta \leq 1$ and $\min\{\alpha,
    \beta \} < 1$ becomes much easier to establish if we examine the digraph and use
    Theorem~\ref{t:scaperpr}.
\end{example}






## Network Centrality

When studying networks of all varieties, a recurring topic is the relative
``centrality'' or ``importance'' of different nodes. One classic application
is the ranking of web pages by search engines.  Here are some examples related
to economics:

\begin{itemize}
    \item In which industry will one dollar of additional demand have the most
        impact on aggregate production, once we take into account all the
        backward linkages?  In which sector will a rise in productivity have
        the largest effect on national output?
    \item A negative shock endangers the solvency of the entire banking sector.
        Which institutions should the government rescue, if any?  
    \item In the network games considered in \S\ref{sss:quadgame},
        the Nash equilibrium is
        $x^* = (I - \alpha A)^{-1} \epsilon$. Players' actions are
        dependent on the topology of the network, as encoded in $A$.  A common
        finding is that the level of activity or effort exerted by an agent
        (e.g., severity of criminal activity by a participant in a criminal
        network) can be predicted from their ``centrality'' within the
        network.  
\end{itemize}

In this section we review essential concepts related to network
centrality.\footnote{Centrality measures are sometimes called ``influence
measures,'' particularly in connection with social networks.}




\subsubsection{Centrality Measures}

Let $G$ be the set of weighted digraphs. A \navy{centrality measure}
associates to each $\gG = (V, E, w)$ in $G$ a vector $m(\gG) \in
\RR^{|V|}$, where the $i$-th element of $m(\gG)$ is interpreted as the
centrality (or rank) of vertex $v_i$.  In most cases $m(\gG)$ is nonnegative.
In what follows, to simplify notation, we take $V = \natset{n}$.

(Unfortunately, the definitions and terminology associated with even the most
common centrality measures vary widely across the applied literature.  Our
convention is to follow the mathematicians, rather than the physicists.  For
example, our terminology is consistent with \cite{benzi2015limiting}.)


\subsubsection{Authorities vs Hubs}


Search engine designers recognize that web pages can be important in two
different ways.  Some pages have high \navy{hub centrality}\index{Hub
centrality}, meaning that they \emph{link to} valuable sources of information
(e.g., news aggregation sites) .  Other pages have high \navy{authority
centrality}, meaning that they contain valuable information, as indicated by
the number and significance of \emph{incoming} links (e.g., websites of
respected news organizations).  Figure~\ref{f:hub_and_authority} helps to
visualize the difference.

\begin{figure*}
   \begin{center}
    \input{tikz/hub_and_authority.tex}
    \caption{\label{f:hub_and_authority} Hub vs authority}
   \end{center}
\end{figure*}


Similar ideas can and have been applied to economic networks (often using
different terminology).  For example, in production networks we study below,
high hub centrality is related to upstreamness: such sectors tend to supply
intermediate goods to many important industries. Conversely, a high authority
ranking will coincide with downstreamness.

In what follows we discuss both hub-based and authority-based centrality
measures, providing definitions and illustrating the relationship between
them.




\subsubsection{Degree Centrality}\label{sss:degcen}

Two of the most elementary measures of ``importance''
of a vertex in a given digraph $\gG = (V, E)$ are its in-degree and
out-degree. Both of these provide a centrality measure.  
\navy{In-degree centrality}\index{In-degree centrality} $i(\gG)$ is defined as  
the vector $(i_d(v))_{v \in V}$.
\navy{Out-degree centrality}\index{out-degree centrality} $o(\gG)$ is defined
as $(o_d(v))_{v \in V}$.  If $\gG$ is expressed as a Networkx \texttt{DiGraph} called
\texttt{G} (see, e.g., \S\ref{sss:nx}), then $i(\gG)$ can be calculated via
%
\begin{minted}{python}
iG = [G.in_degree(v) for v in G.nodes()]    
\end{minted}

This method is relatively slow when $\gG$ is a large digraph. Since vectorized
operations are generally faster, let's look at an alternative method using
operations on arrays.  

To illustrate the method, recall the network of financial institutions in
Figure~\ref{f:network_liabfin}.  We can compute the in-degree and out-degree
centrality measures by first converting the adjacency matrix, which is shown in
\eqref{eq:fnegwa}, to a binary matrix that corresponds to the adjacency matrix
of the same network viewed as an unweighted graph:
%
\begin{equation}\label{eq:fnegwau}
    U = 
    \begin{pmatrix}
        0 & 1 & 0 & 0 & 0 \\
        1 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 \\
        0 & 1 & 0 & 0 & 1 \\
        1 & 0 & 1 & 1 & 0 
    \end{pmatrix}
\end{equation}
%
Now $U(i, j) = 1$ if and only if $i$ points to $j$.  The out-degree and
in-degree centrality measures can be computed as 
%
\begin{equation}\label{eq:ogig}
    o(\gG) = U \1
    \quad \text{and} \quad
    i(\gG) = U^\top \1,
\end{equation}
%
respectively.  That is, summing the rows of $U$ gives the out-degree
centrality measure, while summing the columns gives the in-degree measure.

The out-degree centrality measure is a hub-based ranking, while the vector of
in-degrees is an authority-based ranking.  For the financial network in
Figure~\ref{f:network_liabfin}, a high out-degree for a given institution
means that it lends to many other institutions.  A high in-degree indicates
that many institutions lend to it.

Notice that, to switch from a hub-based ranking to an authority-based ranking,
we need only transpose the (binary) adjacency matrix $U$.  We will see that
the same is true for other centrality measures.   This is intuitive, since
transposing the adjacency matrices reverses the direction of the edges
(Exercise~\ref{ex:rar}).

For a weighted digraph $\gG = (V, E, w)$ with adjacency matrix $A$, the
\navy{weighted out-degree centrality} and \navy{weighted in-degree centrality}
measures are defined as
%
\begin{equation}\label{eq:wogig}
    o(\gG) = A \1
    \quad \text{and} \quad
    i(\gG) = A^\top \1,
\end{equation}
%
respectively, by analogy with \eqref{eq:ogig}.  We present some intuition for
these measures in applications below.

Unfortunately, while in- and out-degree measures of centrality are simple to
calculate, they are not always informative.  As an example, consider again the
international credit network shown in
Figure~\ref{f:financial_network_analysis_visualization}.  There, an edge
exists between almost every node, so the in- or out-degree based centrality
ranking fails to effectively separate the countries. This can be seen in the
out-degree ranking of countries corresponding to that network in the top left panel of
Figure~\ref{f:financial_network_analysis_centrality}, and in the in-degree
ranking in the top right.  


\begin{figure}
   \centering
   \scalebox{0.64}{\includegraphics[trim = 0mm 10mm 0mm 10mm, clip]{ % lbrt
   figures/financial_network_analysis_centrality.pdf}}
   \caption{\label{f:financial_network_analysis_centrality} Centrality measures for the credit network}
\end{figure}

There are other limitations of degree-based centrality rankings. For example,
suppose web page A has many inbound links, while page B has fewer.  Even
though page A dominates in terms of in-degree, it might be less important than
web page B to, say, a potential advertiser, when the links into B are from
more heavily trafficked pages.  Thinking about this point suggests
that importance can be recursive: the importance of a given node depends on
the importance of other nodes that link to it.  The next set of centrality
measures we turn to has this recursive property.




\subsubsection{Eigenvector Centrality}\label{sss:eigcen}

Let $\gG = (V, E, w)$ be a weighted digraph with adjacency matrix $A$.
Recalling that $r(A)$ is the spectral radius of $A$, the \navy{hub-based eigenvector
centrality}\index{Eigenvector centrality} of $\gG$ is
defined as the $e \in \RR^n_+$ that solves
%
\begin{equation}\label{eq:eicen0}
    e = \frac{1}{r(A)} A e.
\end{equation}
%
Element-by-element, this is
%
\begin{equation}\label{eq:eicen}
    e_i = \frac{1}{r(A)} \sum_{j \in \natset{n}} a_{ij} e_j
    \qquad  \text{for all } i \in \natset{n}.
\end{equation}
%
Note the recursive nature of the definition:
 the centrality obtained by vertex $i$ is proportional to a
sum of the centrality of all vertices, weighted by the ``rates of flow'' from
$i$ into these vertices.   A vertex $i$ is highly ranked if (a) there are many
edges leaving $i$, (b) these edges have large weights, and (c) the edges
point to other highly ranked vertices.

When we study demand shocks in \S\ref{ss:dshocks}, we will provide a more
concrete interpretation of eigenvector centrality.  We will see that, in
production networks, sectors with high hub-based eigenvector centrality are
important \emph{suppliers}.  In particular, they are activated by a wide array
of demand shocks once orders flow backwards through the network.

\begin{Exercise}
    Show that~\eqref{eq:eicen} has a unique solution, up to a positive scalar
    multiple, whenever $A$ is strongly connected.
\end{Exercise}

\begin{Answer}
    When $A$ is strongly connected, the Perron--Frobenius theorem tells us that
    $r(A)>0$ and $A$ has a unique (up to a scalar multiple) dominant right
    eigenvector satisfying $r(A) e = A e$.  Rearranging
    gives~\eqref{eq:eicen}.\footnote{While the dominant eigenvector is only
        defined up to a positive scaling constant, this is no reason for
        concern, since positive scaling has no impact on the ranking.  In most
        cases, users of this centrality ranking choose the dominant
    eigenvector $e$ satisfying $\| e \| = 1$.}
\end{Answer}

As the name suggests, hub-based eigenvector centrality is a measure of hub
centrality: vertices are awarded high rankings when they \emph{point to}
important vertices.  The next two exercises help to reinforce this point.

\begin{Exercise}\label{ex:sinkev}
    Show that nodes with zero out-degree always have zero hub-based
    eigenvector centrality. 
\end{Exercise}

To compute eigenvector centrality when the adjacency matrix $A$ is primitive,
we can employ the Perron--Frobenius Theorem, which tells us that 
$r(A)^{-m} A^m \to e \, \epsilon^\top$ as $m \to \infty$,
where $\epsilon$ and $e$ are the dominant left and right eigenvectors
of $A$.  This implies 
%
\begin{equation}\label{eq:ramm}
    r(A)^{-m} A^m \1 \to c e 
    \quad \text{where } c := \epsilon^\top \1.
\end{equation}
%
Thus, evaluating $r(A)^{-m} A^m \1$ at large $m$ returns a scalar multiple of
$e$.  The package Networkx provides a function for computing eigenvector
centrality via \eqref{eq:ramm}.

One issue with this method is the assumption of primitivity,
since the convergence in \eqref{eq:ramm} can fail without it.
The following function uses an alternative technique, based on Arnoldi
iteration, which typically works even when primitivity fails.
(The \texttt{authority} option is explained below.) 

\begin{minted}{python}
import numpy as np
from scipy.sparse import linalg

def eigenvector_centrality(A, m=40, authority=False):
    """
    Computes and normalizes the dominant eigenvector of A.  
    """
    A_temp = A.T if authority else A
    r, vec_r = linalg.eigs(A_temp, k=1, which='LR')
    e = vec_r.flatten().real
    return e / np.sum(e)
\end{minted}

\begin{Exercise}\label{ex:ha0}
    Show that the digraph in Figure~\ref{f:hub_vs_authorith} is not primitive.
    Using the code above or another suitable routine, compute the hub-based
    eigenvector centrality rankings. You should obtain values close to $e =
    (0.3694,0.2612,0.3694,0)$.  Note that the sink vertex (vertex 4) obtains
    the lowest rank.
\end{Exercise}

\begin{figure}
   \begin{center}
    \input{tikz/hub_vs_authorith.tex}
    \caption{\label{f:hub_vs_authorith} A network with a source and a sink}
   \end{center}
\end{figure}

The middle left panel of Figure~\ref{f:financial_network_analysis_centrality}
shows the hub-based eigenvector centrality ranking for the international
credit network shown in
Figure~\ref{f:financial_network_analysis_visualization}.  Countries that are rated
highly according to this rank tend to be important players in terms of supply
of credit.  Japan takes the highest rank according to this measure, although
countries with large financial sectors such as Great Britain and France are
not far behind.  (The color scheme in
Figure~\ref{f:financial_network_analysis_visualization} is also matched to 
hub-based eigenvector centrality.)

The \navy{authority-based eigenvector centrality}\index{Eigenvector
centrality} of $\gG$ is defined as the $e \in \RR^n_+$ 
solving
%
\begin{equation}\label{eq:eicena0}
    e = \frac{1}{r(A)} A^\top e.
\end{equation}
%
The difference between~\eqref{eq:eicena0} and~\eqref{eq:eicen} is just transposition of $A$.
(Transposes do not affect the spectral radius of a matrix.)
Element-by-element, this is
%
\begin{equation}\label{eq:eicena}
    e_j = \frac{1}{r(A)} \sum_{i \in \natset{n}} a_{ij} e_i
    \qquad  \text{for all } j \in \natset{n}.
\end{equation}
%
We see $e_j$ will be high if many nodes with high authority
rankings link to $j$.



The middle right panel of Figure~\ref{f:financial_network_analysis_centrality}
shows the authority-based eigenvector centrality ranking for the international
credit network shown in
Figure~\ref{f:financial_network_analysis_visualization}.  Highly ranked
countries are those that attract large inflows of credit, or credit inflows
from other major players.  The US clearly dominates the rankings as a target
of interbank credit.  

\begin{Exercise}
    Assume that $A$ is strongly connected.  Show that authority-based eigenvector
    centrality is uniquely defined up to a positive scaling constant and
    equal to the dominant \emph{left} eigenvector of $A$.
\end{Exercise}




\subsubsection{Katz Centrality}\label{sss:hbkc}

Eigenvector centrality can be problematic.  Although the definition in
\eqref{eq:eicen} makes sense when $A$ is strongly connected (so that, by the
Perron--Frobenius theorem, $r(A) > 0$), strong connectedness fails in many real
world networks. We will see examples of this in \S\ref{ss:mutmod}, for
production networks defined by input-output matrices.

In addition, while strong connectedness yields strict positivity of the
dominant eigenvector, many vertices can be assigned a zero ranking when it
fails (see, e.g., Exercise~\ref{ex:sinkev}).   This zero ranking often runs
counter to our intuition when we examine specific networks.

Considerations such as these encourage use of an alternative notion of
centrality for networks called Katz centrality, originally due to
\cite{katz1953new}, which is positive under weaker conditions and uniquely
defined up to a tuning parameter.  Fixing $\beta$ in $(0, 1/r(A))$, the
\navy{hub-based Katz centrality}\index{Katz centrality} of weighted digraph
$\gG$ with adjacency matrix $A$, at parameter $\beta$, is defined as the
vector $\kappa := \kappa(\beta, A) \in \RR^n_+$ that solves
%
\begin{equation}\label{eq:katz}
    \kappa_i =  \beta \sum_{j \in \natset{n}} a_{ij} \kappa_j + 1
    \qquad  \text{for all } i \in \natset{n}.
\end{equation}
%
The intuition is very similar to that provided for eigenvector centrality: 
high centrality is conferred on $i$ when it is linked to by
vertices that themselves have high centrality.  The difference between
\eqref{eq:katz} and \eqref{eq:eicen} is just in the additive constant $1$.

\begin{Exercise}
    Show that, under the stated condition $0 < \beta < 1/r(A)$, hub-based Katz
    centrality is always finite and
    uniquely defined by
    %
    \begin{equation}\label{eq:katzhub}
        \kappa 
        = (I - \beta A)^{-1} \1
        = \sum_{\ell \geq 0} (\beta A)^\ell \1,
    \end{equation}
    %
    where $\1$ is a column vector of ones.
\end{Exercise}

\begin{Answer}
    When $\beta < 1/r(A)$ we have $r(\beta A) < 1$.  Hence,
     we can express~\eqref{eq:katz} as
    $\kappa = \1 + \beta A \kappa$ and employ the Theorem~\ref{t:nsl} to
    obtain the stated result.
\end{Answer}

\begin{Exercise}
    We know from the Perron--Frobenius theorem that the eigenvector centrality
    measure will be everywhere positive when the digraph is strongly
    connected.  A condition weaker than strong connectivity is that every
    vertex has positive out-degree.  Show that the Katz measure of centrality
    is strictly positive on each vertex under this condition.
\end{Exercise}

The attenuation parameter $\beta$ is used to ensure that $\kappa$ is finite
and uniquely defined under the condition $0 < \beta < 1/r(A)$.  It can be
proved that, when the graph is strongly connected, hub-based (resp.,
authority-based) Katz centrality converges to the hub-based (resp.,
authority-based) eigenvector centrality as $\beta \uparrow
1/r(A)$.\footnote{See, for example, \cite{benzi2015limiting}.} This is why, in
the bottom two panels of Figure~\ref{f:financial_network_analysis_centrality},
the hub-based (resp., authority-based) Katz centrality ranking is seen to be
close to its eigenvector-based counterpart.

When $r(A)<1$, we use $\beta=1$ as the default for Katz centrality computations.  


\begin{Exercise}\label{ex:ha}
    Compute the hub-based Katz centrality rankings for the simple digraph in
    Figure~\ref{f:hub_vs_authorith} when $\beta=1$.   You should obtain
    $\kappa = (5, 4, 5, 1)$.  Hence, the source vertex (vertex 1) obtains
    equal highest rank and the sink vertex (vertex 4) obtains the lowest rank.
\end{Exercise}


Analogously, the \navy{authority-based Katz centrality}\index{Katz centrality}
 of $\gG$ is defined as
the $\kappa \in \RR^n_+$ that solves
%
\begin{equation}\label{eq:katza}
    \kappa_j =  \beta \sum_{i \in \natset{n}} a_{ij} \kappa_i + 1
    \qquad  \text{for all } j \in \natset{n}.
\end{equation}
%

\begin{Exercise}
    Show that, under the restriction $0 < \beta < 1/r(A)$, 
    the unique solution to \eqref{eq:katza} is given by 
    %
    \begin{equation}\label{eq:katzav}
        \kappa = (I - \beta A^\top)^{-1} \1
        \quad \iff \quad
        \kappa^\top = \1^\top (I - \beta A)^{-1}.
    \end{equation}
    %
    (Verify the stated equivalence.)
\end{Exercise}


\begin{Exercise}\label{ex:ha2}
    Compute the authority-based Katz centrality rankings for the digraph
    in Figure~\ref{f:hub_vs_authorith} when $\beta=1$.   You should obtain
    $\kappa = (1, 6, 4, 4)$.  Notice that the source vertex now has the lowest
    rank. This is due to the fact that hubs are devalued relative to authorities.
\end{Exercise}







## Further Reading

textbooks on economic and social networks by
\cite{jackson2010social}, \cite{easley2010networks} and
\cite{borgatti2018analyzing}, as well as the handbook by
\cite{bramoulle2016oxford}.   

\cite{jackson2014networks} gives a survey of the
literature.  

Within the realm of network science, the high
level texts by \cite{newman2018networks}, \cite{menczer2020first} and
\cite{coscia2021atlas} are excellent.

One good text on graphs and graph-theoretic algorithms is
\cite{kepner2011graph}.  \cite{ballester2006s} provide an interpretation of
Katz centrality (which they call Bonacich centrality) in terms of Nash
equilibria of quadratic games. \cite{du2015competitive} show how PageRank can
be obtained as a competitive equilibrium of an economic problem.
\cite{calvo2009peer} develop a model in which the outcomes for agents embedded
in a network are proportional to the Katz centrality.
\cite{elliott2019network} show that, in a setting where agents can create
nonrival, heterogeneous public goods, an important set of efficient solutions
are characterized by contributions being proportional to agents' eigenvector
centralities in the network.  

\cite{kumamoto2018power} provide a detailed survey of power laws in
economics and social sciences, including a discussion of the preferential
attachment model of \cite{barabasi1999emergence}.  \cite{newman2005power} is
also highly readable.  The textbook of \cite{durrett2007random} is rigorous,
carefully written and contains interesting motivational background, as well as
an extensive citation list for studies of scale-free networks. 


It should be clear from the symbol $\approx$ in \eqref{eq:apla} that the
definition of scale-free networks is not entirely rigorous.  Moreover, when
connecting the definition to observed networks, we cannot obtain complete
clarity by taking a limit, as we did when we defined power laws in
\S\ref{ss:pow}, since the number of vertices is always finite. This
imprecision in the definition has led to heated debate (see, e.g.,
\cite{holme2019rare}).  Given the preponderance of positive empirical studies,
we take the view that, up to a reasonable degree of approximation, the
scale-free property is remarkably widespread.

In \S\ref{sss:quadgame} we briefly mentioned network games, social networks
and key players.  These topics deserve more attention than we have been able
to provide.  An excellent overview is given in \cite{zenou2016key}.
\cite{amarasinghe2020key} apply these ideas to problems in economic
development.  Valuable related papers include 
\cite{allouch2015private}, \cite{belhaj2016efficient},
\cite{demange2017optimal}, \cite{belhaj2019group},
\cite{galeotti2020targeting}.

Another topic we reluctantly omitted in order to keep the textbook short is
endogenous network formation in economic environments.  Influential papers in
this field include \cite{bala2000noncooperative}, \cite{watts2001dynamic},
\cite{graham2017econometric}, \cite{galeotti2010law}, \cite{hojman2008core},
and \cite{jackson1996strategic}.

Finally, \cite{candogan2012optimal} study the profit maximization problem
for a monopolist who sells items to participants in a social network.  The
main idea is that, in certain settings, the monopolist will find it profitable
to offer discounts to key players in the network. \cite{atalay2011network}
argue that in-degrees observed in US buyer-supplier networks have lighter
tails than a power law, and supply a model that better fits their data.


## Exercises

\subsubsection{Application: Quadratic Network Games}\label{sss:quadgame}

\cite{acemoglu2016networks} and \cite{zenou2016key} consider quadratic
games with $n$ agents where agent $k$ seeks to maximize
%
\begin{equation}\label{eq:uing}
    u_k(x) 
    := -\frac{1}{2} x_k^2 + \alpha x^\top A x + x_k \epsilon_k.
\end{equation}
%
Here $x = (x_i)_{i=1}^n$, $A$ is a symmetric matrix with $a_{ii}=0$ for all
$i$, $\alpha \in (0,1)$ is a parameter and $\epsilon = (\epsilon_i)_{i=1}^n$
is a random vector.  (This is the set up for the quadratic game in \S21.2.1 of
\cite{acemoglu2016networks}.)  The $k$-th agent takes the decisions $x_j$ as
given for all $j \not= k$ when maximizing \eqref{eq:uing}.  

In this context, $A$ is understood as the adjacency matrix of a graph with
vertices $V = \natset{n}$, where each vertex is one agent.  We can reconstruct
the weighted digraph $(V, E, w)$ by setting $w(i, j) = a_{ij}$ and letting $E$
be all $(i,j)$ pairs in $\natset{n} \times \natset{n}$ with $a_{ij} > 0$.
The weights identify some form of relationship between the agents, such as
influence or friendship.

\begin{Exercise}
    A \navy{Nash equilibrium}\index{Nash equilibrium} for the quadratic network
    game is a vector $x^* \in \RR^n$ such that, for all $i \in \natset{n}$,
    the choice $x_i^*$ of agent $i$ maximizes~\eqref{eq:uing} taking $x_j^*$
    as given for all $j \not= i$.  Show that, whenever $r(A) < 1/\alpha$,
    a unique Nash equilibrium $x^*$ exists in $\RR^n$ and, moreover,
    $x^* := (I - \alpha A)^{-1} \epsilon$.
\end{Exercise}

\begin{Answer}
    Recalling that $\partial / (\partial x_k) x^\top A x = (x^\top A)_k$, the first
    order condition corresponding to \eqref{eq:uing}, taking the actions of other
    players as given, is
    %
    \begin{equation*}
        x_k = \alpha (x^\top A)_k + \epsilon_k
        \qquad (k \in \natset{n}).
    \end{equation*}
    %
    Concatenating into a row vector and then taking the transpose yields $x =
    \alpha A x + \epsilon$, where we used the fact that $A$ is symmetric.
    Since $r(\alpha A) = \alpha r(A)$, the condition $r(A) < 1/\alpha$ implies
    that $r(\alpha A) < 1$, so, by the Neumann series lemma, the unique solution 
    is $x^* = (I - \alpha A)^{-1} \epsilon$.
\end{Answer}

The network game described in this section has many interesting applications,
including social networks, crime networks and peer networks.  References are
provided in \S\ref{s:cnni}.


