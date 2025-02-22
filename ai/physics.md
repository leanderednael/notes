# Physics

## [Boltzmann Law: Physics to Computing](https://www.edx.org/learn/engineering/purdue-university-boltzmann-law-physics-to-computing)

After completing this course, you will be able to:

- Explain the _law of equilibrium_ and _entropy_.
- Analyze and design simple _Boltzmann machines_.
- Analyze and design simple _quantum circuits_.

### 1. The Boltzmann Law

Upon completion of this week, you will be able to:

- differentiate between _Boltzmann law_ and Boltzmann _approximation_
- describe _entropy_, _free energy_
- explain the _self-consistent field method_

#### 1.1. The State Space

In a material like a molecule or a solid, electrons can occupy different **energy levels** $\varepsilon_i, i \in \{1, \dots, n\}$ according to the **exclusion principle** that each level can accomodate at most one electron so that the lower levels are filled first.

The dividing line between energy levels is called the **electrochemical potential** $\mu$.

At higher temperatures $\mu$ becomes more diffuse. The **Fermi function** describes the probability that a given energy level $\varepsilon$ is occupied by an electron at a given temperature $T$, with the Boltzmann constant $k$ and electrochemical potential $\mu$:

$$
f(\varepsilon) = \frac{1}{1 + e^{\frac{\varepsilon - \mu}{kT}}},
\qquad
f(x)  = \frac{1}{1 + e^x}, \, x = \frac{\varepsilon - \mu}{kT}
$$

Each energy level represents a one-electron space that can either be occupied or unoccupied. The **state space** of the system is the set of all possible configurations of the system, of which there are $2^n$ for $n$ one-electron energy levels.

With $8$ one-electron energy levels, the number of levels in state space is $2^8 = 256$.

The **Boltzmann law** defines _equilibrium in statistical mechanics_: it states that the probability of a system being in a particular state is proportional to the exponential of the negative of the difference of that state's energy and number of electrons (with partition function $Z = \sum_i e^{- \frac{E_i - \mu N_i}{kT}}$ which serves as a normalisation constant so the probabilities of all possible states sum to $1$, thus "partitioning" the probability space to be able to calculate individual probabilities):

$$p_i = \frac{1}{Z} e^{-\frac{E_i - \mu N_i}{kT}}$$

_Derivation of the Fermi function_ (for one level): there are two states, $0$ and $1$, for each position. The $0$ state has $0$ electrons and energy $0$, the $1$ state has $1$ electron and energy $\varepsilon_1$. By the law of total probability, $p_0 + p_1 = \frac{1}{Z} e^{-\frac{0}{kT}} + \frac{1}{Z} e^{-\frac{\epsilon_1 - \mu}{kT}} = \frac{1}{Z} + \frac{1}{Z} e^{-x_1} = \frac{1 + e^{-x_1}}{Z} = 1$. It follows that $Z = 1 + e^{-x_1}$. Thus, $p_1 = \frac{e^{-x_1}}{1 + e^{-x_1}} = \frac{1}{1 + e^{x_1}} = f(\varepsilon), \, p_0 = \frac{1}{1 + e^{-x_1}} = \frac{1 + e^{-x_1} - e^{-x_1}}{1 + e^{-x_1}} = 1 - \frac{e^{-x_1}}{1 + e^{-x_1}} = 1 - f(\varepsilon)$.

_Two one-electron energy levels_ without interactions: $2^2$ states, $Z = 1 + e^{-x_1} + e^{-x_2} + e^{-x_1}e^{-x_2} = (1 + e^{-x_1}) (1 + e^{-x_2})$

| $i$ | $N$ | $E$                             | $\frac{E - \mu N}{kT}$                                         | $p_i$                                    |
| --- | --- | ------------------------------- | -------------------------------------------------------------- | ---------------------------------------- |
| 11  | 2   | $\varepsilon_1 + \varepsilon_2$ | $\frac{\varepsilon_1 + \varepsilon_2 - 2 \mu}{kT} = x_1 + x_2$ | $\frac{1}{Z} e^{-x_1}e^{-x_2} = f_1 f_2$ |
| 10  | 1   | $\varepsilon_2$                 | $x_2$                                                          | $\frac{1}{Z} e^{-x_2} = (1 - f_1) f_2$   |
| 01  | 1   | $\varepsilon_1$                 | $x_1$                                                          | $\frac{1}{Z} e^{-x_1} = (1 - f_2) f_1$   |
| 00  | 0   | $0$                             | $0$                                                            | $\frac{1}{Z} = (1 - f_1)(1 - f_2)$       |

Note: The Fermi function can be generalised to multiple energy levels in this way iff the levels are **non-interacting**; systems with interactions between electrons can only be solved with the general Boltzmann law as the interaction term prevents factorisation into separate Fermi functions: $x_1 + x_2 + \frac{U_0}{kT}$.

The **Boltzmann approximation** gives a valid approximation _to the Fermi function_ for $x_1 \gg 1$ (i.e. $\varepsilon_1 \gg \mu$): $f(\varepsilon_1) \approx e^{-x_1}$.

#### 1.2. The Boltzmann Law

Every state $i$ is a function of the number of electrons $N_i$ and amount of energy $E_i$. In order to change states, a non-isolated system continuously exchanges particles $\Delta N$ and energy $\Delta E$ with its surroundings, called the **reservoir**, which is a function of $\mu, T$.

- A **grand canonical ensemble** is an ensemble of systems which exchanges both energy and particles: $p_i = \frac{1}{Z} e^{-\frac{E_i - \mu N_i}{kT}}$.
- A **canonical ensemble** is an ensemble of systems which exchange only energy but no particles: $p_i = \frac{1}{Z} e^\frac{\mu N_i}{kT} e^{-\frac{E_i}{kT}} = \frac{1}{Z'} e^{-\frac{E_i}{kT}}$ with $Z' = \frac{Z}{e^\frac{\mu N_i}{kT}}$ with $N_i$ also a constant.

The Boltzmann law states that in equilibrium, _states with lower energy have higher probability of being occupied_. The system and reservoir have the same energy $E_0$ and the system can exchange energy $\varepsilon$ with the reservoir. The probability of the system having energy $E_0 + \varepsilon$ is $p_1 = \frac{1}{Z} e^{-\frac{E_0 + \varepsilon - \mu N}{kT}}$ and the probability of the system having energy $E_0$ is $p_2 = \frac{1}{Z} e^{-\frac{E_0 - \mu N}{kT}}$. The ratio of the probabilities shows that the probability of the system being in a higher energy state compared to a lower energy state decreases exponentially with the increase in energy $\varepsilon$:

$$\frac{p_1}{p_2} = \frac{W_1}{W_2} = \frac{W(E_0 + \varepsilon)}{W(E_0)} = \frac{e^{-\frac{E_0 + \varepsilon - \mu N}{kT}}}{e^{-\frac{E_0 - \mu N}{kT}}} = e^{-\frac{\varepsilon}{kT}}$$

- _When the system has lower energy, it is more likely to be in that state_; which is a reflection of the second law of thermodynamics where systems evolve towards states of higher entropy (or disorder) while maintaining energy conservation.
- _If the system has lower energy, the reservoir has higher energy_; the system and reservoir are in equilibrium when the system has the same energy as the reservoir.
- The weight of the states $W$ represents the _number of accessible states of the reservoir_ and _is a rapidly increasing function of energy_: $W_1 = W(E_0 + \varepsilon) > W_2 = W(E_0)$.

Entropy $S$ is proportional to the logarithm of the number of states available with energy $E$:

$$S(E) = k \ln{W(E)}$$

The temperature $T$ is the inverse of the rate of change of entropy with respect to energy:

$$\frac{1}{T} = \frac{dS}{dE}, \qquad \frac{dS}{dE}\bigg\vert_{E=E_0} \approx \frac{S(E_0 + \varepsilon) - S(E_0)}{\varepsilon}$$

The ratio of the probabilities of the system being in a higher energy state compared to a lower energy state can thus be shown to be equal to the ratios of the probabilities $p_i = \frac{1}{Z} e^{- \frac{E_i}{kT}}$ as predicted by the Boltzmann law for a canonical ensemble:

$$\frac{p_1}{p_2} = \frac{W_1}{W_2} = \frac{W(E_0 + \varepsilon)}{W(E_0)} = \frac{e^{\frac{S(E_0 + \varepsilon)}{k}}}{e^{\frac{S(E_0)}{k}}} = e^{\frac{S(E_0 + \varepsilon) - S(E_0)}{k}} = e^{\frac{\varepsilon}{kT}} = e^{\frac{E_2 - E_1}{kT}} = \frac{e^{\frac{-E_1}{kT}}}{e^{\frac{-E_2}{kT}}}$$

For a grand canonical ensemble, $p_i = \frac{1}{Z} e^{-\frac{E_i - \mu N_i}{kT}}$, where also particles are exchanged, $W$ is a function of both energy and the number of particles $N$:

$$\frac{p_1}{p_2} = \frac{W(E_0 + \varepsilon, N_0 + n)}{W(E_0, N_0)} = e^\frac{\varepsilon - \mu n}{kT}$$

The Boltzmann law applies to all systems irrespective of its details (such as interactions, superconduction) because it _reflects a property of the reservoir_, $W$, (not of the system, $p$):

$$S(E, N) = k \ln{W(E, N)}, \quad \frac{\partial S}{\partial E} = \frac{1}{T}, \frac{\partial S}{\partial N} = \frac{-\mu}{T}$$

#### 1.3. Shannon Entropy

Consider a reservoir composed of two-electron levels with $n$ identical non-interacting units (where the tilde is used to distinguish from the system's equilibrium values $p_0, p_1$):

$$n \times \tilde{p}_0, \, n \times \tilde{p}_1, \quad \tilde{p}_0 + \tilde{p}_1 = 1, \quad N = n \tilde{p}_1, \, E = \varepsilon n \tilde{p}_1$$

Suppose $\tilde{p}_1 = 1$, then there is only one possible state $1111111\dots$; suppose $\tilde{p}_1 = 0$, then there is only one possible state $00000000\dots$; suppose $\tilde{p}_1 = 0.5$, then there are many possible states $0101010\dots, 1010101\dots, \dots$:

$$W = {}^n \mathrm{C}_{n\tilde{p}_1} = {}^n \mathrm{C}_{n\tilde{p}_0} = \frac{n!}{(n\tilde{p}_0)! (n\tilde{p}_1)!}$$

$$\frac{S}{n} \leq -k \ln{\max{W}} = -k (\tilde{p}_0 \ln{\tilde{p}_0} + \tilde{p}_1 \ln{\tilde{p}_1})$$

A collection of 8 one-electron levels are all full. The entropy is zero: $S = k \ln{1} = 0k = 0$.

Consider a collection of $n$ units each having $3$ levels that are all equally probable. The entropy is $S = nk \ln{3}$.

#### 1.4. Free Energy

#### 1.5. Self-Consistent Field

### 2. Boltzmann Machines

#### 2.1. Sampling

#### 2.2. Orchestrating Interactions

#### 2.3. Optimisation

#### 2.4. Inference

#### 2.5. Learning

### 3. The Transition Matrix

#### 3.1. Markov Chain Monte Carlo (MCMC)

#### 3.2. Gibbs Sampling

#### 3.3. Sequential vs. Simultaneous

#### 3.4. Bayesian Networks

#### 3.5. Feynman Paths

### 4. The Quantum Boltzmann Law

#### 4.1. Quantum Spins

#### 4.2. One Q-Bit systems

#### 4.3. Spin-Spin Interactions

#### 4.4. Two Q-Bit systems

#### 4.5. Quantum Annealing

### 5. The Quantum Transition Matrix

#### 5.1. Adiabatic Gated Computing

#### 5.2. Hadamard Gates

#### 5.3. Grover Search

#### 5.4. Shor's Algorithm

#### 5.5. Feynman Paths
