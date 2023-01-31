# Networks

## Outline

In recent years there has been rapid growth in a field called [network science](https://en.wikipedia.org/wiki/Network_science).

Network science studies links (or connections, or relationships) between groups of objects.

One of the most important examples is the world wide web, where web pages are
connected by hyperlinks.

Analysis of the human brain emphasizes the network of connections between nerve cells (called neurons).

Artificial neural networks are based on this idea, using data to build intricate connections between simple processing units.

Biologists studying transmission of diseases like COVID-19 have to consider the connections and interactions between human hosts.

In operations research, network analysis is related to such fundamental problems as on minimum cost flow, traveling salesman, shortest path, and assignment.

TODO add wiki links for above defs

Within economics, important examples of networks include

* financial networks
* production networks 
* trade networks
* transport networks and 
* social networks   

For example, social networks affect trends in market sentiment and consumer decisions.  

The structure of financial networks helps to determine relative fragility of the system

The structure of production networks affects trade, innovation and the propagation of local shocks.

This lecture gives an introduction to economic networks.

(Some parts are drawn from the text https://networks.quantecon.org/ but the
level of this lecture is less advanced.)


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

TODO repeat the Hamilton example from the Markov chain lecture, plot the
graph, explain how we can view it as a network. 




## An Introduction to Graph Theory

Network science is built on top of a major branch of mathematics called [graph theory](https://en.wikipedia.org/wiki/Graph_theory).

To understand and analyze networks, we need some understanding of graph theory.

While graph theory can be complicated, we will cover only the basic ideas.

However, these ideas will already be enough for us to discuss interesting and
important ideas on economic and financial networks.


We focus on "directed" graphs, where connects are one way rather that symmetric

E.g.,

* bank $A$ lends money to bank $B$
* firm $A$ supplies goods to firm $B$



### Key Definitions

A **directed graph** consists of 
%
\begin{itemize}
    \item a finite set $V$ and
    \item a collection of pairs $(u, v)$ where $u$ and $v$ are elements of $V$
\end{itemize}
%

The elements of $V$ are called the \navy{vertices} or \navy{nodes} of the graph.

In the aircraft export example above, the set $V$ is all countries included in the data set.

The pairs $(u,v)$ are called the **edges** of the graph and the set of all edges will usually be denoted by $E$

Intuitively and visually, an edge $(u,v)$ is understood as an arrow from vertex $u$ to vertex $v$.  

(A neat way to represent an arrow is to record the location of the tail and
head of the arrow, and that's exactly what an edge does.)

In the aircraft export example above, $E$ is all the arrows in the figure,
each indicating some positive amount of aircraft exports from one country to
another.

Let's look at more examples.

Two graphs are given in Figures~\ref{f:rich_poor_no_label}--\ref{f:poverty_trap}.  Each graph has three vertices.  

TODO -- convert these to use https://h1ros.github.io/posts/introduction-to-graphviz-in-jupyter-notebook/, discuss with JS

In these cases, the arrows (edges) could be thought of as representing positive possibility of transition over a given unit of time.  


\begin{figure}
   \begin{center}
       \scalebox{1.0}{\input{tikz/rich_poor_no_label.tex}}
   \end{center}
   \caption{\label{f:rich_poor_no_label} A directed graph of classes}
\end{figure}

\begin{figure}
   \centering
   \scalebox{1.0}{\input{tikz/poverty_trap.tex}}
   \caption{\label{f:poverty_trap} An alternative edge list}
\end{figure}


For a given edge $(u, v)$, $u$ is called a \navy{direct predecessor} of $v$
and $v$ is called a \navy{direct successor} of $u$.  

Also, the \navy{in-degree} and \navy{out-degree} of $v \in V$ are defined by
%
\begin{itemize}
    \item the $i_d(v) = $ the number of direct predecessors of $v$ and
    \item the $o_d(v) = $ the number of direct successors of $v$.
\end{itemize}



### Digraphs in Networkx

The Python package Networkx is a mature library that provides a convenient
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

As an example, let us create the directed graph in Figure~\ref{f:poverty_trap}, which
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
correct representation of $\gG_p$.  For our small directed graph we can verify this
by plotting the graph via Networkx with the following code: 

\begin{minted}{python}
fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True, 
                 font_weight='bold', arrows=True, alpha=0.8,
                 connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()
\end{minted}

This code produces Figure~\ref{f:networkx_basics_1}, which matches the
original directed graph in Figure~\ref{f:poverty_trap}.

\begin{figure}
   \centering
   \scalebox{0.6}{\includegraphics[trim = 10mm 10mm 0mm 10mm, clip]{figures/networkx_basics_1.pdf}}
   \caption{\label{f:networkx_basics_1} Networkx directed graph plot}
\end{figure}

\texttt{DiGraph} objects have methods that calculate in-degree and out-degree
of vertices.   For example,
%
\begin{minted}{python}
G_p.in_degree('p')
\end{minted}
%
prints 3.





### Communication

Next we study communication and connectedness, which have important implications for economic networks.

Vertex $v$ is called \navy{accessible} from vertex $u$ if either $u=v$ or
there exists a sequence of edges that lead from $u$ to $v$.  

* in this case, we write $u \to v$

(Visually, there is a sequence of arrows leading from $u$ to $v$.)

For example, suppose we have a directed graph representing a production
network, where 

* elements of $V$ are industrial sectors and
* existence of an edge $(i, j)$ means that $i$ supplies products or services to $j$.  

Then sector $m$ is an upstream supplier of sector $\ell$ whenever $m \to \ell$.

Two vertices $u$ and $v$ are said to \navy{communicate} if $u \to v$ and $v \to u$.

A graph is called \navy{strongly connected} if any two nodes in $V$ are
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
directed graph,'' or ``weighted digraph.'' 


### Definitions

A \navy{weighted digraph} $\gG$ is a triple $(V, E, w)$
such that $(V, E)$ is a digraph and $w$ is a function from $E$ to $(0,
\infty)$, called the \navy{weight function}.

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


## Adjacency Matrices 

The \navy{adjacency matrix} of a \emph{weighted}
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


\begin{proposition}\label{p:accesspos}
    Let $\gG$ be a weighted digraph with adjacency matrix $A$. For distinct
    vertices $i, j \in \natset{n}$ and $k \in \NN$, we have
    %
    \begin{equation*}
        a^k_{i j} > 0
        \; \iff \;
        \text{ $j$ is accessible from $i$}.
    \end{equation*}
    %
\end{proposition}



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
different ways.  Some pages have high \navy{hub centrality}, meaning that they \emph{link to} valuable sources of information
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
\navy{In-degree centrality} $i(\gG)$ is defined as  
the vector $(i_d(v))_{v \in V}$.
\navy{Out-degree centrality} $o(\gG)$ is defined
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
centrality} of $\gG$ is
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

The \navy{authority-based eigenvector centrality} of $\gG$ is defined as the $e \in \RR^n_+$ 
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
\navy{hub-based Katz centrality} of weighted digraph
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


Analogously, the \navy{authority-based Katz centrality}
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
    A \navy{Nash equilibrium} for the quadratic network
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




### Ex

Here is a mathematical exercise for those who like proofs.

Let $(V, E)$ be a directed graph and write $u \sim v$ if $u$ and $v$ communicate.  

Show that $\sim$ is an [equivalence relation](https://en.wikipedia.org/wiki/Equivalence_relation) on $V$.


