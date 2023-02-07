---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(schelling)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Racial Segregation 

```{index} single: Schelling Segregation Model
```

```{index} single: Models; Schelling's Segregation Model
```

```{contents} Contents
:depth: 2
```

## Outline

In 1969, Thomas C. Schelling developed a simple but striking model of racial
segregation {cite}`Schelling1969`.

His model studies the dynamics of racially mixed neighborhoods.

Like much of Schelling's work, the model shows how local interactions can lead
to surprising aggregate structure.

In particular, it shows that relatively mild preference for neighbors of
similar race can lead in aggregate to the collapse of mixed neighborhoods, and
high levels of segregation.

In other words, extreme segregation outcomes arise even though people's
preferences are not particularly extreme.

These extreme outcomes happen because of interactions between people that
drive self-reinforcing dynamics in the model.

These ideas will become clearer as the lecture unfolds.

In recognition of his work on segregation and other research, Schelling was
awarded the 2005 Nobel Prize in Economic Sciences (joint with Robert Aumann).


Let's start with some imports:

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
from random import uniform, seed
from math import sqrt
```

## The Model

In this lecture, we (in fact you) will build and run a version of Schelling's model.

### Set-Up

We will cover a variation of Schelling's model that is different from the
original but is easy to program and, at the same time, captures his main idea.

Suppose we have two types of people: orange people and green people.

Assume there are $n$ of each type.

These agents all live on a single unit square.

Thus, the location of an agent is just a point $(x, y)$,  where $0 < x, y < 1$.

* The set of all points $(x,y)$ satisfying $0 < x, y < 1$ is called the **unit square**
* Below we denote the unit square by $S$

+++

### Preferences

We will say that an agent is *happy* if 5 or more of her 10 nearest neighbors are of the same type.

An agent who is not happy is called *unhappy*.

For example,

*  if an agent is orange and 5 of her 10 nearest neighbors are orange, then she is happy.
*  if an agent is green and 8 of her 10 nearest neighbors are orange, then she is unhappy.

'Nearest' is in terms of [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance).

An important point to note is that agents are **not** averse to living in mixed areas.

They are perfectly happy if half of their neighbors are of the other color.

+++

### Behavior

Initially, agents are mixed together (integrated).

In particular, we assume that the initial location of each agent is an
independent draw from a bivariate uniform distribution on the unit square $S$.

(First their $x$ coordinate is drawn from a uniform distribution on $(0,1)$
and then, independently, their $y$ coordinate is drawn from the same
distribution.)

Now, cycling through the set of all agents, each agent is now given the chance to stay or move.

We assume that each agent will stay put if they are happy and move if unhappy.

The algorithm for moving is as follows

```{prf:algorithm} Jump Chain Algorithm
:label: move_algo

1. Draw a random location in $S$
1. If happy at new location, move there
1. Otherwise, go to step 1

```

We cycle continuously through the agents, each time allowing an unhappy agent to move.

We continue to cycle until no one wishes to move.

+++

## Results

Let's now implement and run this simulation.

In what follows, agents are modeled as [objects](https://python-programming.quantecon.org/python_oop.html).

Here's an indication of they look

```{code-block} none
* Data:

    * type (green or orange)
    * location

* Methods:

    * determine whether happy or not given locations of other agents
    * If not happy, move
        * find a new location where happy
```

```{code-cell} ipython3
class Agent:

    def __init__(self, type):
        self.type = type
        self.draw_location()

    def draw_location(self):
        self.location = uniform(0, 1), uniform(0, 1)

    def get_distance(self, other):
        "Computes the euclidean distance between self and other agent."
        a = (self.location[0] - other.location[0])**2
        b = (self.location[1] - other.location[1])**2
        return sqrt(a + b)

    def happy(self, agents):
        "True if sufficient number of nearest neighbors are of the same type."
        distances = []
        
        # Distances is a list of pairs (d, agent), where d is distance from
        # agent to self
        for agent in agents:
            if self != agent:
                distance = self.get_distance(agent)
                distances.append((distance, agent))
                
        # Sort from smallest to largest, according to distance
        distances.sort()
        
        # Extract the neighboring agents
        neighbors = [agent for d, agent in distances[:num_neighbors]]
        
        # Count how many neighbors have the same type as self
        num_same_type = sum(self.type == agent.type for agent in neighbors)
        return num_same_type >= require_same_type

    def update(self, agents):
        "If not happy, then randomly choose new locations until happy."
        while not self.happy(agents):
            self.draw_location()
```

```{code-cell} ipython3

def plot_distribution(agents, cycle_num):
    "Plot the distribution of agents after cycle_num rounds of the loop."
    x_values_0, y_values_0 = [], []
    x_values_1, y_values_1 = [], []
    # == Obtain locations of each type == #
    for agent in agents:
        x, y = agent.location
        if agent.type == 0:
            x_values_0.append(x)
            y_values_0.append(y)
        else:
            x_values_1.append(x)
            y_values_1.append(y)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_args = {'markersize': 8, 'alpha': 0.7}
    ax.set_facecolor('azure')
    ax.plot(x_values_0, y_values_0, 'o', markerfacecolor='orange', **plot_args)
    ax.plot(x_values_1, y_values_1, 'o', markerfacecolor='green', **plot_args)
    ax.set_title(f'Cycle {cycle_num-1}')
    plt.show()
```

And here's some pseudocode for the main loop

```{code-block} none
while agents are still moving
    for agent in agents
        give agent the opportunity to move
```

```{code-cell} ipython3
def run_simulation(num_of_type_0=600,
                   num_of_type_1=600,
                   max_iter=100_000,       # Maximum number of iterations 
                   num_neighbors=10,       # No. of agents viewed as neighbors
                   require_same_type=5,    # Required to be happy (same type)
                   set_seed=1234):         

    # Set the seed for reproducibility
    seed(set_seed)   
    
    # == Create a list of agents == #
    agents = [Agent(0) for i in range(num_of_type_0)]
    agents.extend(Agent(1) for i in range(num_of_type_1))

    count = 1
    
    # Plot the initial distribution
    plot_distribution(agents, count)
    
    # ==  Loop until none wishes to move == #
    while True and count < max_iter:
        print('Entering loop ', count)
        count += 1
        no_one_moved = True
        for agent in agents:
            old_location = agent.location
            agent.update(agents)
            if agent.location != old_location:
                no_one_moved = False
        if no_one_moved:
            break
            
    # Plot final distribution
    plot_distribution(agents, count)

    if count < max_iter:
        print(f'Converged after {count} iterations.')
    else:
        print('Hit iteration bound and terminated.')
    

```

Let's have a look at the results we got when we coded and ran this model.

```{code-cell} ipython3
run_simulation()
```

As discussed above, agents are initially mixed randomly together.

But after several cycles, they become segregated into distinct regions.

In this instance, the program terminated after a small number of cycles through the set of
agents, indicating that all agents had reached a state of happiness.

What is striking about the pictures is how rapidly racial integration breaks down.

This is despite the fact that people in the model don't actually mind living mixed with the other type.

Even with these preferences, the outcome is a high degree of segregation.



## Exercises

```{exercise-start}
:label: schelling_ex1
```

If you feel like a further exercise, you can probably speed up some of the computations and
then increase the number of agents.

```{exercise-end}
```

```{solution-start} schelling_ex1
:class: dropdown
```
solution here

```{solution-end}
```

```{code-cell} ipython3

```
