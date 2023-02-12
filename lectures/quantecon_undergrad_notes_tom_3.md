## Elements of  Supply and Demand


This document describe a class of linear models that determine competitive equilibrium prices and quantities.  

Linear algebra and some multivariable calculus are the  tools deployed.

Versions of the classic welfare theorems prevail.

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

$$ p = d_0 - d_1 q, \quad d_0, d_1 > 0 $$
$$ p = s_0 + s_1 q , \quad s_0, s_1 > 0 $$

**Consumer surplus** equals  area under an inverse demand curve minus $p q$:

$$ \int_0^q (d_0 - d_1 x) dx = d_0 q -.5 d_1 q^2 - pq $$

**Producer surplus** equals $p q$ minus   the area under an inverse supply curve:


$$ p q - \int_0^q (s_0 + s_1 x) dx  $$

or




**Welfare criterion** is consumer surplus plus producer surplus

$$ \int_0^q (d_0 - d_1 x) dx - \int_0^q (s_0 + s_1 x) dx  $$

or 

$$ (d_0 - s_0) q - .5 (d_1 + s_1) q^2 $$

The quantity that  maximizes the  welfare criterion is 

$$ q = \frac{ d_0 - s_0}{s_1 + d_1} \tag{1}$$


Competitive equilibrium quantity equates demand price to supply price:


$$ p =  d_0 - d_1 q = s_0 + s_1 q ,   $$

which implies (1) and leads to both 

**Key finding:** a competitive equilibrium quantity maximizes our  welfare criterion.

and

**A competitive equilibrium computation strategy:** after using the welfare problem to find  a competitive equilibrium quantity, we can read a competive equilibrium price from  either supply price or demand price at the  competitive equilibrium quantity.


# Multiple goods 

We study a setting with $n$ goods and $n$ corresponding prices.

## Formulas from linear algebra

We  apply  formulas from linear algebra for

  * differentiating an inner product
  * differentiating a quadratic form

Where $a$ is an $n \times 1$ vector, $A$ is an $n \times n$ matrix, and $x$ is an $n \times 1$ vector:

$$ \frac{\partial a^T x }{\partial x} = a $$

$$ \frac{\partial x^T A x}{\partial x} = (A + A^T)x $$

## From utility function to demand curve 

Let $\Pi$ be an $n\times n$ matrix, $c$ be $n \times 1$ vector of consumptions of various goods, $b$ be $n \times 1$ vector of bliss points, $e$ an $n \times 1$ vector of endowments, and $p$ be an $n\times 1$ vector of prices

A consumer faces $p$ as a price taker and chooses $c$ to maximize

$$ -.5 (\Pi c -b) \cdot (\Pi c -b ) \tag{0} $$

subject to the budget constraint

$$ p \cdot (c -e ) = 0 \tag{2}$$


## Digression about Marshallian and Hicksian Demand Curves

**Remark:** We'll use budget constraint (2) in situations in which a consumers's endowment vector $e$ is his **only** source of income. But sometimes we'll instead assume that the consumer has other sources of income (positive or negative) and write his budget constraint as


$$ p \cdot (c -e ) = W \tag{2'}$$

where $W$ is measured in "dollars" (or some other **numeraire**) and component $p_i$ of the price vector is measured in dollars per good $i$. 

Whether the consumer's budget constraint is  (2) or (2') and whether we take $W$ as a free parameter or instead as an endogenous variable  to be solved for will  affect the consumer's marginal utility of wealth.

How we treat these things will determine whether we are constucting  

* a **Marshallian** demand curve, when we use (2) and solve for $\mu$ using equation (4) below, or
* a **Hicksian** demand curve, when we  treat $\mu$ as a fixed parameter and solve for $W$ from (2').

Marshallian and Hicksian demand curves correspond to different mental experiments:

* For a Marshallian demand curve, hypothetical price vector  changes produce changes in   quantities determined that have  both **substitution** and **income** effects
  
    * income effects are consequences of  changes in 
$p^T e$ associated with the change in the price vector

* For a Hicksian demand curve, hypothetical price vector  changes produce changes in   quantities determined that have only **substitution**  effects
    
    * changes in the price vector leave the $p^e + W$ unaltered because we freeze $\mu$ and solve for $W$

We'll discuss these alternative concepts of demand curves more  below.


## Demand Curve as Constrained Utility Maximization

For now, we assume that the budget constraint is (2). 

So we'll be deriving  a **Marshallian** demand curve.

Form a Lagrangian

$$ L = -.5 (\Pi c -b) \cdot (\Pi c -b ) + \mu [p\cdot (e-c)] $$

where $\mu$ is a Lagrange multiplier that is often called a **marginal utility of wealth**.  

The consumer chooses $c$ to maximize $L$ and $\mu$ to minimize it.

First-order conditions for $c$ are

$$ \frac{\partial L} {\partial c} = - \Pi^T \Pi c + \Pi^T b - \mu p = 0 $$

so that, given $\mu$, the consumer chooses

$$ c = \Pi^{-1} b - \Pi^{-1} (\Pi^T)^{-1} \mu p \tag{3}  $$

Substituting (3) into budget constraint (2) and solving for $\mu$ gives 

$$ \mu(p,e) = \frac{p^T (\Pi^{-1} b - e)}{p^T (\Pi^T \Pi )^{-1} p}. \tag{4} $$

Equation (4) tells how marginal utility of wealth depend on  the endowment vector  $e$ and the price vector  $p$. 

**Remark:** Equation (4) is a consequence of imposing that $p (c - e) = 0$.  We could instead take $\mu$ as a parameter and use (3) and the budget constraint (2') to solve for $W.$ Which way we proceed determines whether we are constructing a **Marshallian** or **Hicksian** demand curve.  



## Endowment economy, I

We now study a pure-exchange or endowment economy.

Consider a single-consumer, multiple-goods economy without production.  

The only source of goods is the single consumer's endowment vector   $e$. 

Competitive equilibium prices must be set to  induce the consumer to choose  $c=e$.

This implies that the equilibrium price vector must satisfy

$$ p = \mu^{-1} (\Pi^T b - \Pi^T \Pi e)$$

In the present case where we have imposed budget constraint in the form (2), we are free to normalize the price vector by setting the marginal utility of wealth $\mu =1$ (or any other value for that matter).

This amounts to choosing a common  unit (or numeraire) in which prices of all goods are expressed.  

(Doubling all prices will affect neither quantities nor relative prices.)

We'll set $\mu=1$.  

 

**Exercise:** Verify that $\mu=1$ satisfies formula (4). 

**Exercise:** Verify that setting  $\mu=2$ also implies that formula (4) is satisfied. 


**Endowment Economy, II**

Let's study a **pure exchange** economy without production. 

There are two consumers who differ in their endowment vectors $e_i$ and their bliss-point vectors $b_i$ for $i=1,2$.  

The total endowment is $e_1 + e_2$.

A competitive equilibrium  requires that

$$ c_1 + c_2 = e_1 + e_2 $$

Assume the demand curves   

$$ c_i = \Pi^{-1}b_i - (\Pi^T \Pi)^{-1} \mu_i p $$

Competitive equilibrium  then requires that

$$ e_1 + e_2 =  \Pi^{-1} (b_1 + b_2) - (\Pi^T \Pi)^{-1} (\mu_1 + \mu_2) p $$

which after a line or two of linear algebra implies that

$$ (\mu_1 + \mu_2) p = \Pi^T(b_1+ b_2) (e_1 + e_2) \tag{6} $$

We can normalize prices by setting $\mu_1 + \mu_2 =1$ and then deducing


$$ \mu_i(p,e) = \frac{p^T (\Pi^{-1} bi - e_i)}{p^T (\Pi^T \Pi )^{-1} p} \tag{7} $$

for $\mu_i, i = 1,2$. 


**Exercise:** Show that, up to normalization by a positive scalar,  the same competitive equilibrium price vector that you computed in the preceding two-consumer economy would prevail in a single-consumer economy in which a single **representative consumer** has utility function  
$$ -.5 (\Pi c -b) \cdot (\Pi c -b ) $$
and endowment vector $e$,  where
$$
b = b_1 + b_2 
$$

and 

$$e = e_1 + e_2 . $$

## Dynamics and Risk as Special Cases of Pure Exchange Economy



Special cases of out model can   handle

  * dynamics 
     - by putting different dates on different commodities
  * risk
     - by making commodities contingent on states whose realizations are described by a **known probability distribution**

Let's illustrate how.

### Dynamics

We want to represent the utility function

$$ - .5 [(c_1 - b_1)^2 + \beta (c_2 - b_2)^2] $$

where $\beta \in (0,1)$ is a discount factor, $c_1$ is consumption at time $1$ and $c_2$ is consumption at time 2.

To capture this with our quadratic utility function (0),  set

$$ \Pi = \begin{bmatrix} 1 & 0 \cr 
         1 & \sqrt{\beta} \end{bmatrix}$$
and
$$ b = \begin{bmatrix} b_1 \cr \sqrt{\beta} b_2
\end{bmatrix}$$

The  budget constraint becomes

$$ p_1 c_1 + p_2 c_2 = p_1 e_1 + p_2 e_2 $$

The left side is the **discounted present value** of consumption.

The right side is the **discounted present value** of the consumer's endowment.

### Risk and state-contingent claims 

The enviroment is  static meaning that there is only one period.

There is risk.

There are two states of nature, $1$ and $2$.

The probability that state $1$ occurs is $\lambda$.

The probability that state $2$ occurs is $(1-\lambda)$.

The consumer's **expected utility** is
$$ -.5 [\lambda (c_1 - b_1)^2 + (1-\lambda)(c_2 - b_2)^2] $$

where 

* $c_1$ is consumption in state $1$ 
* $c_2$ is consumption in state $2$

To capture these preferences we set

$$ \Pi = \begin{bmatrix} \lambda & 0 \cr
                     0  & (1-\lambda) \end{bmatrix} $$

$$ c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}$$


$$ b = \begin{bmatrix} b_1 \cr b_2 \end{bmatrix}$$

The endowment vector is


$$ e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}$$

The price vector is

$$ p = \begin{bmatrix} p_1 \cr p_2 \end{bmatrix} $$

where $p_i$ is the price of one unit of consumption in state $i$. 

Before the random state of the world $i$ is realized, the consumer  sells his/her state-contingent endowment bundle and purchases a state-contingent consumption bundle. 


## Possible Exercises

To illustrate consequences of demand and supply shifts, we have lots of parameters to shift in the above models

  * distribution of endowments $e_1, e_2$
  * bliss point vectors $b_1, b_2$
  * probability $\lambda$

We can study how these things affect equilibrium prices and allocations. 

Plenty of fun exercises that could be executed with a single Python class.

It would be easy to build a example with two consumers who have different beliefs ($\lambda$'s)


# Economies with Endogenous Supplies of Goods


## Supply

Start from a cost function 

$$ C(q) = h \cdot q + .5 q^T J q $$

where $J$ is a positive definite matrix.

The $n\times 1$ vector of marginal costs is

$$ \frac{\partial C(q)}{\partial q} = h + H q $$

where

$$ H = .5 (J + J') $$

The inverse supply curve implied by marginal cost pricing is 

$$ p = h + H q $$

## Competitive equilibrium

### $\mu=1$ warmup case

As a special case, let's pin down a demand curve by setting the marginal utility of wealth  $\mu =1$.


Equate supply price to demand price

$$ p = h + H c = \Pi^T b - \Pi^T \Pi c $$

which implies the equilibrium quantity vector

$$ c = (\Pi^T \Pi + H )^{-1} ( \Pi^T b - h) \tag{5} $$

This equation is the counterpart of equilbrium quantity (1) for the scalar $n=1$ model with which we began.

### General $\mu\neq 1$ case

Now let's extend the preceding analysis to a more 
general case by allowing $\mu \neq 1$.

Then the inverse depend curve is

$$ p = \mu^{-1} [\Pi^T b - \Pi^T \Pi c] $$

Equating this to the inverse supply curve and solving
for $c$ gives

$$ c = [\Pi^T \Pi + \mu H]^{-1} [ \Pi^T b - \mu h] \tag{5'} $$

## Multi-good social welfare maximization problem


Our welfare or social planning  problem is to choose $c$ to maximize 
$$-.5 \mu^{-1}(\Pi c -b) \cdot (\Pi c -b )$$  minus the area under the inverse supply curve:

$$ -.5 \mu^{-1}(\Pi c -b) \cdot (\Pi c -b ) -h c - .5 c^T J c $$

The first-order condition with respect to $c$ is

$$ - \mu^{-1} \Pi^T \Pi c + \mu^{-1}\Pi^T b - h - .5 H c = 0 $$

which implies (5'). 

Thus, in the multiple case as for the single-good case,  a competitive equilibrium quantity  solves a planning problem.  

We can read the competitive equilbrium price vector off the inverse demand curve or the inverse supply curve. 






