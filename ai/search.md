# [State Space Search](https://en.wikipedia.org/wiki/State_space_search)

## 1. Agents

- State-space representation

## 2. Blind / Uninformed Search

### 2.1. DFS

### 2.2. BFS

### 2.3. UCS

## 3. Informed Search

### 3.1. A\*

## 4. Constraint Satisfaction Problems, SAT Solving & Logic

- Backtracking, backtrack variants, intelligent and dependency directed backtracking
- Arc consistency techniques
- hybrid constraint propagation methods

## 5. Game Trees

### 5.1. Minimax

### 5.2. Alpha-Beta Pruning

## 6. Uncertainty in Game Trees

### 6.1. Expectimax

## 7. Markov Decision Processes

- States $S$
- Actions $A$
- Rewards $R(s, a, s')$
- Transition probabilities $T(s, a, s') = P(s' \mid s, a)$
- Policy $\pi(s) = \arg \max_a{Q(s, a)}$, where
  - Q-value $Q(s, a) = \sum_{s'}{T(s, a, s') \big( R(s, a, s') + \gamma V(s') \big)}$ (weighted average of the rewards at $s'$), where
    - State value $V(s) = \max_a{Q(s, a)} = \max_a{\sum_{s'}{T(s, a, s') \big( R(s, a, s') + \gamma V(s') \big)}}$

Note: this is not a linear set of equations. Therefore, value iteration: $V_0(s) = 0, \, V_{k+1}(s) = \max_a{Q(s, a)}$

States: [(1, 1), (1, 2), (1, 3),
(2, 1), (2, 2), (2, 3),
(3, 1), (3, 2), (3, 3)]
Current state $s$ = (2, 2)
Actions: [N, E, S, W]
Rewards:

- $R(s, a, s'=(1, 1)) = 8$
- $R(s, a, s'=(1, 2)) = 15$
- $R(s, a, s'=(1, 3)) = 12$
- $R(s, a, s'=(2, 1)) = 2$
- $R(s, a, s'=(2, 3)) = 10$
- $R(s, a, s'=(3, 1)) = 7$
- $R(s, a, s'=(3, 2)) = 16$
- $R(s, a, s'=(3, 3)) = 11$

Transition probabilities:

- $T(s, a, s' = \text{intended}) = P(s' \mid s, a) = 0.7$
- $T(s, a, s' = \text{left}) = P(s' \mid s, a) = 0.1$
- $T(s, a, s' = \text{right}) = P(s' \mid s, a) = 0.2$

Q-values (with $R = 0, \gamma = 1$):

- $Q(s, a = N) = \sum_{(1, 1), (1, 2), (1, 3)}{T(s, a = N, s') \big( R(s, a = N, s') + \gamma V(s') \big)}$
- ...

Reinforcement learning can be viewed as a Markov decision process (MDP), having the following properties:

- Set of states $S$
- Set of actions $\text{Actions}(S)$
- Transition model $P(s' \mid s, a)$
- Reward function $R(s, a, s')$

The transition model gives us the new state after performing an action, and the reward function is what kind of feedback the agent gets.

## 8. Planning

## 9. Genetic Algorithms and Evolutionary Computation

High-level contents and materials

    - Lecture 1: Introduction
    - Lecture 2: Problems, representation, and variation
    - Lecture 3: Population management
    - Online module 1: Local search operators
    - Online module 2: Multiobjective optimization and diversity promotion
    - Online case study modules: Hands-on programming exercises

Detailed contents

Basics of evolutionary algorithms

    - Exploration versus exploitation
    - Computational and optimization problems
    - Objective function
    - Representation
    - Constraints
    - Variation operators
    - Selection and elimination operators
    - Hyperparameter self-adaptivity

Local search operators

    - Steepest descent
    - Monte Carlo sampling
    - k-opt

Multi-objective optimization and diversity promotion

    - Crowding
    - Island model
    - Fitness sharing
    - Scalarization (fixed tradeoff)
    - Pareto front
