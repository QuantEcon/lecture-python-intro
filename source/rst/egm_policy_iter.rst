.. include:: /_static/includes/header.raw

.. highlight:: python3

********************************************************
:index:`Optimal Growth IV: The Endogenous Grid Method`
********************************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install quantecon
  !pip install interpolation

Overview
============

Previously, we solved the stochastic optimal growth model using

#. :doc:`value function iteration <optgrowth_fast>`
#. :doc:`Euler equation based time iteration <coleman_policy_iter>`

We found time iteration to be significantly more accurate and efficient.

In this lecture, we'll look at a clever twist on time iteration called the **endogenous grid method** (EGM).

EGM is a numerical method for implementing policy iteration invented by `Chris Carroll <http://www.econ2.jhu.edu/people/ccarroll/>`__.

The original reference is :cite:`Carroll2006`.

Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    from interpolation import interp
    from numba import njit, float64
    from numba.experimental import jitclass
    from quantecon.optimize import brentq
    import matplotlib.pyplot as plt
    %matplotlib inline


Key Idea
========

Let's start by reminding ourselves of the theory and then see how the numerics fit in.



Theory
------

Take the model set out in :doc:`the time iteration lecture <coleman_policy_iter>`, following the same terminology and notation.

The Euler equation is

.. math::
    :label: egm_euler

    (u'\circ \sigma^*)(y)
    = \beta \int (u'\circ \sigma^*)(f(y - \sigma^*(y)) z) f'(y - \sigma^*(y)) z \phi(dz)


As we saw, the Coleman-Reffett operator is a nonlinear operator :math:`K` engineered so that :math:`\sigma^*` is a fixed point of :math:`K`.

It takes as its argument a continuous strictly increasing consumption policy :math:`\sigma \in \Sigma`.

It returns a new function :math:`K \sigma`,  where :math:`(K \sigma)(y)` is the :math:`c \in (0, \infty)` that solves

.. math::
    :label: egm_coledef

    u'(c)
    = \beta \int (u' \circ \sigma) (f(y - c) z ) f'(y - c) z \phi(dz)



Exogenous Grid
-------------------

As discussed in :doc:`the lecture on time iteration <coleman_policy_iter>`, to implement the method on a computer, we need a numerical approximation.

In particular, we represent a policy function by a set of values on a finite grid.

The function itself is reconstructed from this representation when necessary, using interpolation or some other method.

:doc:`Previously <coleman_policy_iter>`, to obtain a finite representation of an updated consumption policy, we

* fixed a grid of income points :math:`\{y_i\}`

* calculated the consumption value :math:`c_i` corresponding to each
  :math:`y_i` using :eq:`egm_coledef` and a root-finding routine

Each :math:`c_i` is then interpreted as the value of the function :math:`K \sigma` at :math:`y_i`.

Thus, with the points :math:`\{y_i, c_i\}` in hand, we can reconstruct :math:`K \sigma` via approximation.

Iteration then continues...




Endogenous Grid
--------------------

The method discussed above requires a root-finding routine to find the
:math:`c_i` corresponding to a given income value :math:`y_i`.

Root-finding is costly because it typically involves a significant number of
function evaluations.

As pointed out by Carroll :cite:`Carroll2006`, we can avoid this if
:math:`y_i` is chosen endogenously.

The only assumption required is that :math:`u'` is invertible on :math:`(0, \infty)`.

Let :math:`(u')^{-1}` be the inverse function of :math:`u'`.

The idea is this:

* First, we fix an *exogenous* grid :math:`\{k_i\}` for capital (:math:`k = y - c`).

* Then we obtain  :math:`c_i` via

.. math::
    :label: egm_getc

    c_i =
    (u')^{-1}
    \left\{
        \beta \int (u' \circ \sigma) (f(k_i) z ) \, f'(k_i) \, z \, \phi(dz)
    \right\}


* Finally, for each :math:`c_i` we set :math:`y_i = c_i + k_i`.

It is clear that each :math:`(y_i, c_i)` pair constructed in this manner satisfies :eq:`egm_coledef`.

With the points :math:`\{y_i, c_i\}` in hand, we can reconstruct :math:`K \sigma` via approximation as before.

The name EGM comes from the fact that the grid :math:`\{y_i\}` is  determined **endogenously**.


Implementation
================

As :doc:`before <coleman_policy_iter>`, we will start with a simple setting
where 

* :math:`u(c) = \ln c`, 

* production is Cobb-Douglas, and 

* the shocks are lognormal.

This will allow us to make comparisons with the analytical solutions

.. literalinclude:: /_static/lecture_specific/optgrowth/cd_analytical.py

We reuse the ``OptimalGrowthModel`` class 

.. literalinclude:: /_static/lecture_specific/optgrowth_fast/ogm.py



The Operator
----------------


Here's an implementation of :math:`K` using EGM as described above.

.. code-block:: python3

    @njit
    def K(??_array, og):
        """
        The Coleman-Reffett operator using EGM

        """

        # Simplify names
        f, ?? = og.f, og.??
        f_prime, u_prime = og.f_prime, og.u_prime
        u_prime_inv = og.u_prime_inv
        grid, shocks = og.grid, og.shocks

        # Determine endogenous grid
        y = grid + ??_array  # y_i = k_i + c_i

        # Linear interpolation of policy using endogenous grid
        ?? = lambda x: interp(y, ??_array, x)

        # Allocate memory for new consumption array
        c = np.empty_like(grid)

        # Solve for updated consumption value
        for i, k in enumerate(grid):
            vals = u_prime(??(f(k) * shocks)) * f_prime(k) * shocks
            c[i] = u_prime_inv(?? * np.mean(vals))

        return c



Note the lack of any root-finding algorithm.

Testing
-------

First we create an instance.

.. code-block:: python3

    og = OptimalGrowthModel()
    grid = og.grid

Here's our solver routine:

.. literalinclude:: /_static/lecture_specific/coleman_policy_iter/solve_time_iter.py 

Let's call it:

.. code-block:: python3

    ??_init = np.copy(grid)
    ?? = solve_model_time_iter(og, ??_init)

Here is a plot of the resulting policy, compared with the true policy:

.. code-block:: python3

    y = grid + ??  # y_i = k_i + c_i

    fig, ax = plt.subplots()

    ax.plot(y, ??, lw=2,
            alpha=0.8, label='approximate policy function')

    ax.plot(y, ??_star(y, og.??, og.??), 'k--',
            lw=2, alpha=0.8, label='true policy function')

    ax.legend()
    plt.show()

The maximal absolute deviation between the two policies is

.. code-block:: python3

    np.max(np.abs(?? - ??_star(y, og.??, og.??)))


How long does it take to converge?

.. code-block:: python3

    %%timeit -n 3 -r 1
    ?? = solve_model_time_iter(og, ??_init, verbose=False)


Relative to time iteration, which as already found to be highly efficient, EGM
has managed to shave off still more run time without compromising accuracy.

This is due to the lack of a numerical root-finding step.

We can now solve the optimal growth model at given parameters extremely fast.
