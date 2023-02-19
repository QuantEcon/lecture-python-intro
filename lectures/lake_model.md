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

# A Lake Model of Employment

\
The importance of the Perron-Frobenius theorem stems from the fact that
firstly in the real world most matrices we encounter are nonnegative matrices.

Secondly, a lot of important models are simply linear iterative models that
begin with an initial condition $x_0$ and then evolve recursively by the rule
$x_{t+1} = Ax_t$ or in short $x_t = A^tx_0$.

This theorem helps characterise the dominant eigenvalue $r(A)$ which
determines the behavior of this iterative process. 

We now illustrate the power of the Perron-Frobenius theorem by showing how it
helps us to analyze a model of employment and unemployment flows in a large
population.

This model is sometimes called the **lake model** because there are two pools of workers:
1. those who are currently employed.
2. those who are currently unemployed but are seeking employment.

The "flows" between the two lakes are as follows:
1. workers exit the labor market at rate $d$.
2. new workers enter the labor market at rate $b$.
3. employed workers separate from their jobs at rate $\alpha$.
4. unemployed workers find jobs at rate $\lambda$.

Let $e_t$ and $u_t$ be the number of employed and unemployed workers at time $t$ respectively.

The total population of workers is $n_t = e_t + u_t$.

The number of unemployed and employed workers thus evolve according to:

```{math}
:label: lake_model
\begin{aligned}
    u_{t+1} = (1-d)(1-\lambda)u_t + \alpha(1-d)e_t + bn_t = ((1-d)(1-\lambda) +& b)u_t + (\alpha(1-d) + b)e_t \\
    e_{t+1} = (1-d)\lambda u_t + (1 - \alpha)(1-d)e_t&
\end{aligned}
```

We can arrange {eq}`lake_model` as a linear system of equations in matrix form $x_{t+1} = Ax_t$ such that:

$$
x_{t+1} =
\begin{bmatrix}
    u_{t+1} \\
    e_{t+1}
\end{bmatrix}
, \; A =
\begin{bmatrix}
    (1-d)(1-\lambda) + b & \alpha(1-d) + b \\
    (1-d)\lambda & (1 - \alpha)(1-d)
\end{bmatrix}
\; \text{and} \;
x_t =
\begin{bmatrix}
    u_t \\
    e_t
\end{bmatrix}
$$

Suppose at $t=0$ we have $x_0 = \begin{bmatrix} u_0 \\ e_0 \end{bmatrix}$.

Then, $x_1=Ax_0$, $x_2=Ax_1=A^2x_0$ and thus $x_t = A^tx_0$.

Thus the long run outcomes of this system depend on the initial condition $x_0$ and the matrix $A$.

$A$ is a nonnegative and irreducible matrix and thus we can use the Perron-Frobenius theorem to obtain some useful results about A.

Note that $colsum_j(A) = 1 + b - d$ for $j=1,2$ and by {eq}`PF_bounds` we can thus conclude that the dominant eigenvalue
$r(A) = 1 + b - d$.

If we consider $g = b - d$ as the overall growth rate of the total labor force, then $r(A) = 1 + g$.

We can thus find a unique positive vector $\bar{x} = \begin{bmatrix} \bar{u} \\ \bar{e} \end{bmatrix}$
such that $A\bar{x} = r(A)\bar{x}$ and $\begin{bmatrix} 1 & 1 \end{bmatrix} \bar{x} = 1$.

Since $\bar{x}$ is the eigenvector corresponding to the dominant eigenvalue $r(A)$ we can also call this the dominant eigenvector.

This eigenvector plays an important role in determining long run outcomes as is illustrated below.

```{code-cell} ipython3
:tags: []

def lake_model(α, λ, d, b):
    g = b-d
    A = np.array([[(1-d)*(1-λ) + b,   α*(1-d) + b],
              [(1-d)*λ,          (1-α)*(1-d)]])

    ū = (1 + g - (1 - d) * (1 - α)) / (1 + g - (1 - d) * (1 - α) + (1 - d) * λ)
    ē = 1 - ū
    x̄ = np.array([ū, ē])
    x̄.shape = (2,1)
    
    ts_length = 1000
    x_ts_1 = np.zeros((2, ts_length))
    x_ts_2 = np.zeros((2, ts_length))
    x_0_1 = (5.0, 0.1)
    x_0_2 = (0.1, 4.0)
    x_ts_1[0, 0] = x_0_1[0]
    x_ts_1[1, 0] = x_0_1[1]
    x_ts_2[0, 0] = x_0_2[0]
    x_ts_2[1, 0] = x_0_2[1]
    
    for t in range(1, ts_length):
        x_ts_1[:, t] = A @ x_ts_1[:, t-1]
        x_ts_2[:, t] = A @ x_ts_2[:, t-1]
        
    fig, ax = plt.subplots()
    
    # Set the axes through the origin
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position("zero")
    for spine in ["right", "top"]:
        ax.spines[spine].set_color("none")
        
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_xlabel("unemployed workforce")
    ax.set_ylabel("employed workforce")
    ax.set_xticks((0, 6))
    ax.set_yticks((0, 6))
    s = 10
    ax.plot([0, s * ū], [0, s * ē], "k--", lw=1)
    ax.scatter(x_ts_1[0, :], x_ts_1[1, :], s=4, c="blue")
    ax.scatter(x_ts_2[0, :], x_ts_2[1, :], s=4, c="green")
    
    ax.plot([ū], [ē], "ko", ms=4, alpha=0.6)
    ax.annotate(r'$\bar{x}$', 
            xy=(ū, ē),
            xycoords="data",
            xytext=(20, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle = "->"))

    x, y = x_0_1[0], x_0_1[1]
    ax.plot([x], [y], "ko", ms=2, alpha=0.6)
    ax.annotate(f'$x_0 = ({x},{y})$', 
            xy=(x, y),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle = "->"))
    
    x, y = x_0_2[0], x_0_2[1]
    ax.plot([x], [y], "ko", ms=2, alpha=0.6)
    ax.annotate(f'$x_0 = ({x},{y})$', 
            xy=(x, y),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle = "->"))
    
    plt.show()
```

```{code-cell} ipython3
lake_model(α=0.01, λ=0.1, d=0.02, b=0.025)
```

If $\bar{x}$ is an eigenvector corresponding to the eigenvalue $r(A)$ then all the vectors in the set
$D := \{ x \in \mathbb{R}^2 : x = \alpha \bar{x} \; \text{for some} \; \alpha >0 \}$ are also eigenvectors corresponding
to $r(A)$.

This set is represented by a dashed line in the above figure.

We can observe that for two distinct initial conditions $x_0$ the sequence of iterates $(A^t x_0)_{t \geq 0}$ move towards D over time.

This suggests that all such sequences share strong similarities in the long run, determined by the dominant eigenvector $\bar{x}$.

In the example illustrated above we considered parameter such that overall growth rate of the labor force $g>0$.

Suppose now we are faced with a situation where the $g<0$, i.e, negative growth in the labor force.

This means that $b-d<0$, i.e., workers exit the market faster than they enter.

What would the behavior of the iterative sequence $x_{t+1} = Ax_t$ be now?

This is visualised below.

```{code-cell} ipython3
lake_model(α=0.01, λ=0.1, d=0.025, b=0.02)
```

Thus, while the sequence of iterates still move towards the dominant eigenvector $\bar{x}$ however in this case
they converge to the origin.

This is a result of the fact that $r(A)<1$, which ensures that the iterative sequence  $(A^t x_0)_{t \geq 0}$ will converge
to some point, in this case to $(0,0)$.

This leads us into the next result.
