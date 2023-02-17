<!-- #region -->

## Elements of  Supply and Demand

This document describe a class of linear models that determine competitive equilibrium prices and quantities.  

Linear algebra and some multivariable calculus are the  tools deployed.

Versions of the two classic welfare theorems prevail.

* **first welfare theorem:** for a  given a distribution of wealth among consumers,  a competitive  equilibrium  allocation of goods solves a particular social planning problem.

* **second welfare theorem:** An allocation of goods among consumers that solves a social planning problem can be supported by a compeitive equilibrium provided that wealth is appropriately  distributed among consumers.   

Key infrastructure concepts  are

* inverse demand curves
* marginal utility of wealth or   money
* inverse supply curves
* consumer surplus
* producer surplus
* welfare maximization
* competitive equilibrium
* homogeneity of degree zero of 
    * demand functions
    * supply function
* dynamics as a special case
* risk as a special case

Our approach is first to offer a 

* scalar version with one good and one price

Then we'll offer a version with  

* $n$ goods
* $n$ prices or relative prices

We'll offer versions of

* pure exchange economies with fixed endowments of goods
* economies in which goods can be produced a cost

### Scalar setting 

We study a market for a single good.

The quantity is $q$ and price is $p$, both scalars.

Inverse demand and supply curves are:

$$ 
p = d_0 - d_1 q, \quad d_0, d_1 > 0 
$$

$$
p = s_0 + s_1 q , \quad s_0, s_1 > 0 
$$

**Consumer surplus** equals  area under an inverse demand curve minus $p q$:

$$ 
\int_0^q (d_0 - d_1 x) dx - pq = d_0 q -.5 d_1 q^2 - pq 
$$

**Producer surplus** equals $p q$ minus   the area under an inverse supply curve:

$$
p q - \int_0^q (s_0 + s_1 x) dx = pq - s_0 q - .5 s_1 q^2
$$

Intimately associated with a competitive equilibrium is the following:

**Welfare criterion** is consumer surplus plus producer surplus

$$ 
\int_0^q (d_0 - d_1 x) dx - \int_0^q (s_0 + s_1 x) dx  
$$

or 

$$ 
\textrm{Welf} = (d_0 - s_0) q - .5 (d_1 + s_1) q^2 
$$

To compute the quantity that  maximizes  welfare criterion $\textrm{Welf}$, we differentiate $\textrm{Welf}$ with respect to   $q$ and set the derivative to zero.  

Solving that equation for $q$ gives

$$   
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:old1)

A  competitive equilibrium quantity equates demand price to supply price:

$$ 
p =  d_0 - d_1 q = s_0 + s_1 q ,   
$$

which implies {eq}`eq:old1`.  

The outcome that the quantity determined by equation {eq}`eq:old1` equates
supply to demand brings us the following important **key finding:** 

*  a competitive equilibrium quantity maximizes our  welfare criterion

It also brings us a convenient **competitive equilibrium computation strategy:** 

* after solving the welfare problem for an optimal  quantity, we can read a competive equilibrium price from  either supply price or demand price at the  competitive equilibrium quantity

Soon we'll derive generalizations of the above demand and supply 
curves from other objects.

We'll derive the **demand curve** from a problem that maximizes a **utility function** subject to a **budget constraint**.

We'll derive the **supply curve** from a **cost function**.

# Multiple goods 

We study a setting with $n$ goods and $n$ corresponding prices.

## Formulas from linear algebra

We  apply  formulas from linear algebra for

* differentiating an inner product
* differentiating a product of a matrix and a vector
* differentiating a quadratic form

Where $a$ is an $n \times 1$ vector, $A$ is an $n \times n$ matrix, and $x$ is an $n \times 1$ vector:

$$ 
\frac{\partial a^\top x }{\partial x} = a 
$$

$$
\frac{\partial A x} {\partial x} = A
$$

$$ 
\frac{\partial x^\top A x}{\partial x} = (A + A^\top)x 
$$

## From utility function to demand curve 

Let

* $\Pi$ be an $n\times n$ matrix,
* $c$ be an $n \times 1$ vector of consumptions of various goods, 
* $b$ be an $n \times 1$ vector of bliss points, 
* $e$ be an $n \times 1$ vector of endowments, and 
* $p$ be an $n\times 1$ vector of prices

We assume that $\Pi$ has an inverse $\Pi^{-1}$ and that $\Pi^\top \Pi$ is a positive definite matrix.

It follows that $\Pi^\top \Pi$ has an inverse.


The matrix $\Pi$ describes a consumer's willingness to substitute one good for another, for each pair of
good in the $n \times 1$ vector $c$.

In particular, we shall see below that $(\Pi^T \Pi)^{-1}$ is a matrix of slopes of (compensated) demand curves for $c$ with respect to a vector of prices:

$$
{\partial c } {\partial p} = (\Pi^T \Pi)^{-1}
$$

A consumer faces $p$ as a price taker and chooses $c$ to maximize the utility function 

$$
-.5 (\Pi c -b) ^\top (\Pi c -b ) 
$$ (eq:old0)

subject to the budget constraint

$$ 
p ^\top (c -e ) = 0 
$$ (eq:old2)

We shall specify examples in which  $\Pi$ and $b$ are such that it typically happens that

$$
\Pi c < < b  
$$ (eq:bversusc)

so that utility function {eq}`eq:old2` tells us that the consumer has much less  of each good than he wants. 

Condition {eq}`eq:bversusc` will ultimately  assure us that competitive equilibrium prices  are all positive. 

## Digression: Marshallian and Hicksian Demand Curves

**Remark:** We'll use budget constraint {eq}`eq:old2` in situations in which a consumers's endowment vector $e$ is his **only** source of income. But sometimes we'll instead assume that the consumer has other sources of income (positive or negative) and write his budget constraint as

$$ 
p ^\top (c -e ) = W 
$$ (eq:old2p)

where $W$ is measured in "dollars" (or some other **numeraire**) and component $p_i$ of the price vector is measured in dollars per good $i$. 

Whether the consumer's budget constraint is  {eq}`eq:old2` or {eq}`eq:old2p` and whether we take $W$ as a free parameter or instead as an endogenous variable  to be solved for will  affect the consumer's marginal utility of wealth.

How we set $\mu$  determines whether we are constucting  

* a **Marshallian** demand curve, when we use {eq}`eq:old2` and solve for $\mu$ using equation {eq}`eq:old4` below, or
* a **Hicksian** demand curve, when we  treat $\mu$ as a fixed parameter and solve for $W$ from {eq}`eq:old2p`.

Marshallian and Hicksian demand curves describe different mental experiments:

* For a Marshallian demand curve, hypothetical price vector  changes produce changes in   quantities determined that have  both **substitution** and **income** effects
  
  * income effects are consequences of  changes in $p^\top e$ associated with the change in the price vector

* For a Hicksian demand curve, hypothetical price vector  changes produce changes in   quantities determined that have only **substitution**  effects
    
  * changes in the price vector leave the $p^\top e + W$ unaltered because we freeze $\mu$ and solve for $W$
  
Sometimes a Hicksian demand curve is called a **compensated** demand curve in order to emphasize that, to disarm the income (or wealth) effect associated with a price change, the consumer's wealth $W$ is adjusted.

We'll discuss these distinct demand curves more  below.

## Demand Curve as Constrained Utility Maximization

For now, we assume that the budget constraint is {eq}`eq:old2`. 

So we'll be deriving  a **Marshallian** demand curve.

Form a Lagrangian

$$ L = -.5 (\Pi c -b) ^\top (\Pi c -b ) + \mu [p^\top (e-c)] $$

where $\mu$ is a Lagrange multiplier that is often called a **marginal utility of wealth**.  

The consumer chooses $c$ to maximize $L$ and $\mu$ to minimize it.

First-order conditions for $c$ are

$$ 
\frac{\partial L} {\partial c} = - \Pi^\top \Pi c + \Pi^\top b - \mu p = 0 
$$

so that, given $\mu$, the consumer chooses

$$
c = \Pi^{-1} b - \Pi^{-1} (\Pi^\top)^{-1} \mu p
$$ (eq:old3)

Substituting {eq}`eq:old3` into budget constraint {eq}`eq:old2` and solving for $\mu$ gives 

$$
\mu(p,e) = \frac{p^\top (\Pi^{-1} b - e)}{p^\top (\Pi^\top \Pi )^{-1} p}.
$$ (eq:old4)

Equation {eq}`eq:old4` tells how marginal utility of wealth depends on  the endowment vector  $e$ and the price vector  $p$. 

**Remark:** Equation {eq}`eq:old4` is a consequence of imposing that $p^\top (c - e) = 0$.  We could instead take $\mu$ as a parameter and use {eq}`eq:old3` and the budget constraint {eq}`eq:old2p` to solve for $W.$ Which way we proceed determines whether we are constructing a **Marshallian** or **Hicksian** demand curve.  

## Endowment economy, I

We now study a pure-exchange economy, or what is sometimes called an  endowment economy.

Consider a single-consumer, multiple-goods economy without production.  

The only source of goods is the single consumer's endowment vector   $e$. 

A competitive equilibium price vector  induces the consumer to choose  $c=e$.

This implies that the equilibrium price vector satisfies

$$ 
p = \mu^{-1} (\Pi^\top b - \Pi^\top \Pi e)
$$

In the present case where we have imposed budget constraint in the form {eq}`eq:old2`, we are free to normalize the price vector by setting the marginal utility of wealth $\mu =1$ (or any other value for that matter).

This amounts to choosing a common  unit (or numeraire) in which prices of all goods are expressed.  

(Doubling all prices will affect neither quantities nor relative prices.)

We'll set $\mu=1$.  

**Exercise:** Verify that setting $\mu=1$ in {eq}`eq:old3` implies that   formula {eq}`eq:old4` is satisfied. 

**Exercise:** Verify that setting  $\mu=2$ in {eq}`eq:old3` also implies that formula {eq}`eq:old4` is satisfied. 

**Endowment Economy, II**

Let's study a **pure exchange** economy without production. 

There are two consumers who differ in their endowment vectors $e_i$ and their bliss-point vectors $b_i$ for $i=1,2$.  

The total endowment is $e_1 + e_2$.

A competitive equilibrium  requires that

$$ 
c_1 + c_2 = e_1 + e_2 
$$

Assume the demand curves   

$$
c_i = \Pi^{-1}b_i - (\Pi^\top \Pi)^{-1} \mu_i p 
$$

Competitive equilibrium  then requires that

$$ 
e_1 + e_2 =  \Pi^{-1} (b_1 + b_2) - (\Pi^\top \Pi)^{-1} (\mu_1 + \mu_2) p 
$$

which after a line or two of linear algebra implies that

$$
(\mu_1 + \mu_2) p = \Pi^\top(b_1+ b_2) - \Pi^\top \Pi (e_1 + e_2) 
$$ (eq:old6)

We can normalize prices by setting $\mu_1 + \mu_2 =1$ and then solving

$$ 
\mu_i(p,e) = \frac{p^\top (\Pi^{-1} b_i - e_i)}{p^\top (\Pi^\top \Pi )^{-1} p}
$$ (eq:old7)

for $\mu_i, i = 1,2$. 

**Exercise:** Show that, up to normalization by a positive scalar,  the same competitive equilibrium price vector that you computed in the preceding two-consumer economy would prevail in a single-consumer economy in which a single **representative consumer** has utility function  

$$
-.5 (\Pi c -b) ^\top (\Pi c -b )
$$

and endowment vector $e$,  where

$$
b = b_1 + b_2 
$$

and 

$$
e = e_1 + e_2 . 
$$

## Dynamics and Risk as Special Cases of Pure Exchange Economy

Special cases of our model can be created to represent

* dynamics 
  - by putting different dates on different commodities
* risk
  - by making commodities contingent on states whose realizations are described by a **known probability distribution**

Let's illustrate how.

### Dynamics

We want to represent the utility function

$$
  -.5 [(c_1 - b_1)^2 + \beta (c_2 - b_2)^2]
$$

where $\beta \in (0,1)$ is a discount factor, $c_1$ is consumption at time $1$ and $c_2$ is consumption at time 2.

To capture this with our quadratic utility function {eq}`eq:old0`,  set

$$ 
\Pi = \begin{bmatrix} 1 & 0 \cr 
         0 & \sqrt{\beta} \end{bmatrix}
$$

$$ 
c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}
$$

and

$$ 
b = \begin{bmatrix} b_1 \cr \sqrt{\beta} b_2
\end{bmatrix}
$$

The  budget constraint {eq}`eq:old2` becomes

$$
p_1 c_1 + p_2 c_2 = p_1 e_1 + p_2 e_2
$$

The left side is the **discounted present value** of consumption.

The right side is the **discounted present value** of the consumer's endowment.

The relative price  $\frac{p_1}{p_2}$ has units of time $2$ goods per unit of time $1$ goods.  

Consequently, $(1+r) = R \equiv \frac{p_1}{p_2}$ is the  **gross interest rate** and $r$ is the **net interest rate**.

### Risk and state-contingent claims 

We study a **static** environment,  meaning that there is only one period.

There is **risk**.

There are two states of nature, $1$ and $2$.

The probability that state $1$ occurs is $\lambda$.

The probability that state $2$ occurs is $(1-\lambda)$.

The consumer's **expected utility** is

$$
-.5 [\lambda (c_1 - b_1)^2 + (1-\lambda)(c_2 - b_2)^2]
$$

where 

* $c_1$ is consumption in state $1$ 
* $c_2$ is consumption in state $2$

To capture these preferences we set

$$ 
\Pi = \begin{bmatrix} \sqrt{\lambda} & 0 \cr
                     0  & \sqrt{1-\lambda} \end{bmatrix} 
$$

$$ 
c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}
$$

$$ 
b = \begin{bmatrix} b_1 \cr b_2 \end{bmatrix}
$$

The endowment vector is

$$ 
e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}
$$

The price vector is

$$
p = \begin{bmatrix} p_1 \cr p_2 \end{bmatrix}
$$

where $p_i$ is the price of one unit of consumption in state $i$. 

The state commodities being traded are often called **Arrow securities**.

Before the random state of the world $i$ is realized, the consumer  sells his/her state-contingent endowment bundle and purchases a state-contingent consumption bundle. 

Trading such securities is a way economists often model **insurance**.

## Possible Exercises

To illustrate consequences of demand and supply shifts, we have lots of parameters to shift in the above models

* distribution of endowments $e_1, e_2$
* bliss point vectors $b_1, b_2$
* probability $\lambda$

We can study how these things affect equilibrium prices and allocations. 

Plenty of fun exercises that could be executed with a single Python class.

It would be easy to build another  example with two consumers who have different beliefs ($\lambda$'s)

# Economies with Endogenous Supplies of Goods

## Supply

Start from a cost function 

$$ 
C(q) = h ^\top q + .5 q^\top J q 
$$

where $J$ is a positive definite matrix.

The $n\times 1$ vector of marginal costs is

$$ 
\frac{\partial C(q)}{\partial q} = h + H q 
$$

where

$$ 
H = .5 (J + J') 
$$

The inverse supply curve implied by marginal cost pricing is 

$$ 
p = h + H q 
$$

## Competitive equilibrium

### $\mu=1$ warmup

As a special case, let's pin down a demand curve by setting the marginal utility of wealth  $\mu =1$.

Equate supply price to demand price

$$
p = h + H c = \Pi^\top b - \Pi^\top \Pi c
$$

which implies the equilibrium quantity vector

$$ 
c = (\Pi^\top \Pi + H )^{-1} ( \Pi^\top b - h)
$$ (eq:old5)

This equation is the counterpart of equilbrium quantity {eq}`eq:old1` for the scalar $n=1$ model with which we began.

### General $\mu\neq 1$ case

Now let's extend the preceding analysis to a more 
general case by allowing $\mu \neq 1$.

Then the inverse depend curve is

$$
p = \mu^{-1} [\Pi^\top b - \Pi^\top \Pi c]
$$

Equating this to the inverse supply curve and solving
for $c$ gives

$$ 
c = [\Pi^\top \Pi + \mu H]^{-1} [ \Pi^\top b - \mu h]
$$ (eq:old5p)

## Multi-good social welfare maximization problem

Our welfare or social planning  problem is to choose $c$ to maximize

$$
-.5 \mu^{-1}(\Pi c -b) ^\top (\Pi c -b )
$$

minus the area under the inverse supply curve, namely, 

$$
h c + .5 c^\top J c  .
$$

So the welfare criterion is

$$
-.5 \mu^{-1}(\Pi c -b) ^\top (\Pi c -b ) -h c - .5 c^\top J c 
$$

In this formulation, $\mu$ is a parameter that describes how the planner weights interests of outside suppliers and our representative consumer.  

The first-order condition with respect to $c$ is

$$
- \mu^{-1} \Pi^\top \Pi c + \mu^{-1}\Pi^\top b - h -  H c = 0 
$$

which implies {eq}`eq:old5p`. 

Thus,  as for the single-good case, with  multiple goods   a competitive equilibrium quantity vector solves a planning problem.

(This is another version of the first welfare theorem.)

We can read the competitive equilbrium price vector off the inverse demand curve or the inverse supply curve.

<!-- #endregion -->

<!-- #region -->
## Notes to Jiacheng for coding

Hi Jiacheng.  

These are just some suggestions about how to begin writing code.  

I sketch some things that are "not even pseudo code".  Here goes.

I recommend that we start "general" and write a Python class that will do "everything".  

Once we get that working with a bunch of fun examples, we can then "work backwards" and make some very simple "baby code" that starts with simple cases (e.g., a scalar case) and graduallly builds up to the class.

The fun thing will be to make some revealing examples.

I recommend making a Python class with the following attributes:

 * **Preferences** in the form of 
  
     * an $n \times n$  positive definite matrix $\Pi$ 
     * an $n \times 1$ vector of bliss points $b$

 * **Endowments** in the form of 
     
     * an $n \times 1$ vector $e$
     * a scalar "wealth" $W$ with default value $0$
     
 * **Production Costs** $C(q) = h^\top q + .5 q^\top J q $ indexed by
 
     * an $n \times 1$ nonnegative vector $h$
     * an $n \times n$ positive definite matrix $J$

Along the lines of your great suggestion (now incorporated in the main text in the previous cell)
the class would do a test to make sure that $b  > > \Pi e $ and raise an exception if it is violated
(at some threshold level we'd have to specify).

 * **A Person** in the form of a pair that consists of 
   
    * **Preferences** and **Endowments**
    
 * **A Pure Exchange Economy** consists of 
 
    * a collection of $m$ **persons**
    
       * $m=1$ for our single-agent economy
       * $m=2$ for our illustrations of a pure exchange economy
    
    * an equilibrium price vector $p$ (normalized somehow) 
    * an equilibrium allocation $c^1, c^2, \ldots, c^m$ -- a collection of $m$ vectors of dimension $n \times 1$
    
 * **A Production Economy** consists of 
 
    * a single **person** that we'll interpret as a representative consumer
    * a single set of **production costs**
    * a multiplier $\mu$ that weights "consumers" versus "producers" in a planner's welfare function, as described above in the main text
    * an $n \times 1$ vector $p$ of competitive equilibrium prices
    * an $n \times 1$ vector $c$ of competitive equilibrium quantities
    * **consumer surplus**
    * **producer surplus**
       
 
    
**Remark:** I don't know whether we want to be simple or fancy in terms of using class inheritance in creating a person or an economy.


## Some experiments


* Single agent with one good and  with production 

  * specify a single **person** and a **cost curve** in a way that let's us replicate the simple
    single-good supply demand example with which we started
  * compute equilibrium $p$ and $c$ and consumer and producer surpluses
  
     *if possible draw graphs of both surpluses
    
  * do some experiments in which we shift $b$ and watch what happens to $p, c$.
  
* Single agent two-good economy **with** production

  * do the counterparts of the above
  * do some experiments with **diagonal** $\Pi$ and also with **non-diagonal** $\Pi$ matrices to study 
  how cross-slopes affect responses of $p$ and $c$ to various shifts in $b$
  
* Two-person economy **without** production

  * study how competive equilibrium $p, c^1, c^2$ responds to various combinations of different
  
     * $b^i$'s
     * $e^i$'s 

* The simple example in the text of a **dynamic economy**

* The simple example in the text of an exchange economy with two **Arrow securities**

 
     
  

<!-- #endregion -->

```python

```
