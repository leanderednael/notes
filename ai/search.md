# Search

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
- Transition model $P(s’ \mid s, a)$
- Reward function $R(s, a, s’)$

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

## 10. Reinforcement Learning

Introduction to planning and reinforcement learning

Multi-armed bandits and their algorithms

    - exploration vs exploitation
    - rewards and regret
        - The problem of the sparse reward.
        - Introduction to advanced exploration techniques: curiosity and empowerment in RL.
        - Introduction to curriculum learning to easy the learning of the goal.
        - Hierarchical RL to learn complex tasks.
        - The learning of Universal Value Functions and Hindsight Experience Replay (HER).
    - greedy algorithms
    - upper confidence bounds

Markov Decision Processes and their variants

    - Bellman Equations
    - Policies and value functions
    - Optimality
    - Partial and full observability

Dynamic Programming

    - Policy evaluation, improvement and iteration
    - Value iteration

Monte Carlo Methods

Temporal-difference learning

    - TD Prediction
    - Q-learning
    - Sarsa
    - On-policy vs off-policy
    - n-Step bootstrapping

Planning and learning with tabular methods

    - Dyna: integrated planning, acting and learning
    - Real time dynamic programming
    - Monte-Carlo tree search

Approximate methods

    - Value function approximation
    - Gradient methods
    - on-policy and off-policy variants

Policy gradient methods

    - Policy approximation
    - Policy gradients
        - What to do in continuous action spaces.
        - How probabilistic policies allow to apply the gradient method directly in the policy network.
        - The REINFORCE algorithm.
        - The Actor-Critic algorithms.
        - State-of-the-art algorithms in continuous action spaces: DDPG, TD3 and SAC.
    - Actor Critic

Contemporary topics

    - Deep Reinforcement learning
        -  Dealing with the deadly triad with the DQN algorithm.
        -  Application to the Atari games case.
        -  Evolutions of the DQN algorithm: Double DQN, Prioritized Experience Replay, multi-step learning and Distributional value functions.
        -  Rainbow: the state-of-the-art algorithm in discrete action space.
    - multi-agent reinforcement learning
        - Learning of behaviours in environment where several agents act.
        - Learning of cooperative behaviors, Learning of competitive behaviors, and mixed cases.
        - State-of-the art algorithms.
        - The special case of games: The Alpha-Go case and the extension to Alpha-Zero.
    - shielding and safe reinforcement learning
    - relational reinforcement learning and traditional planning
    - Towards life-long learning in agents
        - Is RL a way to obtain a General Artificial Intelligence?
        - Multi-task learning in RL, Transfer learning in RL and Meta-learning in RL.

Applications in game playing and beyond

## References

Abbeel, P. (2014). CS188 Intro to AI [Course materials]. UC Berkeley. Retrieved from <https://ai.berkeley.edu/home.html>

Finn, C. (2025). CS 224R Deep Reinforcement Learning [Course materials]. Stanford University. Retrieved from <http://cs224r.stanford.edu/>

Malan, D., & Yu, B. (2024). CS50’s Introduction to Artificial Intelligence with Python [Course materials]. Harvard OpenCourseWare. Retrieved from <https://cs50.harvard.edu/ai/2024/notes/3/>

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
