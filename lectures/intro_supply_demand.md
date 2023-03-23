# Introduction to Supply and Demand

This lecture is about some  linear models of equilibrium prices and
quantities, one of the main topics of elementary microeconomics.

Our approach is first to offer a  scalar version with one good and one price.

## Outline

We shall describe two classic welfare theorems:

* **first welfare theorem:** for a  given a distribution of wealth among consumers,  a competitive  equilibrium  allocation of goods solves a  social planning problem.

* **second welfare theorem:** An allocation of goods to consumers that solves a social planning problem can be supported by a competitive equilibrium with an appropriate initial distribution of  wealth.

Key infrastructure concepts that we'll encounter in this lecture are

* inverse demand curves
* marginal utilities of wealth
* inverse supply curves
* consumer surplus
* producer surplus
* social welfare as a sum of consumer and producer surpluses
* competitive equilibrium

## Supply and Demand

We study a market for a single good in which buyers and sellers exchange a  quantity $q$ for a price $p$.

Quantity $q$ and price  $p$ are  both scalars.

We assume that inverse demand and supply curves for the good are:

$$
p = d_0 - d_1 q, \quad d_0, d_1 > 0
$$

$$
p = s_0 + s_1 q , \quad s_0, s_1 > 0
$$

We call them inverse demand and supply curves because price is on the left side of the equation rather than on the right side as it would be in a direct demand or supply function.



We define **consumer surplus** as the  area under an inverse demand curve minus $p q$:

$$
\int_0^q (d_0 - d_1 x) dx - pq = d_0 q -.5 d_1 q^2 - pq
$$

We define **producer surplus** as $p q$ minus   the area under an inverse supply curve:

$$
p q - \int_0^q (s_0 + s_1 x) dx = pq - s_0 q - .5 s_1 q^2
$$

Sometimes economists measure social welfare by a **welfare criterion** that equals  consumer surplus plus producer surplus

$$
\int_0^q (d_0 - d_1 x) dx - \int_0^q (s_0 + s_1 x) dx  \equiv \textrm{Welf}
$$

or

$$
\textrm{Welf} = (d_0 - s_0) q - .5 (d_1 + s_1) q^2
$$

To compute a quantity that  maximizes  welfare criterion $\textrm{Welf}$, we differentiate $\textrm{Welf}$ with respect to   $q$ and then set the derivative to zero.

We get

$$
\frac{d \textrm{Welf}}{d q} = d_0 - s_0 - (d_1 + s_1) q  = 0
$$

which implies

$$
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:old1)

Let's remember the quantity $q$ given by equation {eq}`eq:old1` that a social planner would choose to maximize consumer plus producer surplus.

We'll compare it to the quantity that emerges in a competitive  equilibrium equilibrium that equates
supply to demand.

Instead of equating quantities supplied and demanded, we'll can accomplish the same thing by equating demand price to supply price:

$$
p =  d_0 - d_1 q = s_0 + s_1 q ,
$$


It we solve the equation defined by the second equality in the above line for $q$, we obtain the
competitive equilibrium quantity; it equals the same $q$ given by equation  {eq}`eq:old1`.

The outcome that the quantity determined by equation {eq}`eq:old1` equates
supply to demand brings us a **key finding:**

*  a competitive equilibrium quantity maximizes our  welfare criterion

It also brings  a useful  **competitive equilibrium computation strategy:**

* after solving the welfare problem for an optimal  quantity, we can read a competitive equilibrium price from  either supply price or demand price at the  competitive equilibrium quantity

Soon we'll derive generalizations of the above demand and supply
curves from other objects.

Our generalizations will extend the preceding analysis of a market for a single good to the analysis
of $n$ simultaneous markets in $n$ goods.

In addition

 * we'll derive  **demand curves** from a consumer problem that maximizes a **utility function** subject to a **budget constraint**.

 * we'll derive  **supply curves** from the problem of a producer who is price taker and maximizes his profits minus total costs that are described by a  **cost function**.
<!-- #endregion -->
