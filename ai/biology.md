# Bioinformatics

- [Bioinformatics](#bioinformatics)
  - [Introduction to molecular biology](#introduction-to-molecular-biology)
    - [DNA](#dna)
    - [RNA](#rna)
      - [Functional RNA](#functional-rna)
    - [Proteins](#proteins)
      - [Basics](#basics)
      - [Protein Structure](#protein-structure)
      - [Protein Function](#protein-function)
      - [Protein Function \& Purification](#protein-function--purification)
      - [Protein Characterisation](#protein-characterisation)
  - [Sequence alignment](#sequence-alignment)
    - [Similarity vs. Homology](#similarity-vs-homology)
    - [Dynamic programming](#dynamic-programming)
    - [Global alignment: Needleman-Wunsch algorithm](#global-alignment-needleman-wunsch-algorithm)
    - [Local alignment: Smith-Waterman algorithm](#local-alignment-smith-waterman-algorithm)
    - [Significance Evaluation](#significance-evaluation)
    - [Substitution matrices (PAM, BLOSUM50) and gap penalties](#substitution-matrices-pam-blosum50-and-gap-penalties)
      - [PAM (Point Accepted Mutations) matrix](#pam-point-accepted-mutations-matrix)
      - [BLOSUM (BLOCKS SUbstitution Matrix)](#blosum-blocks-substitution-matrix)
    - [Basic Local Alignment Search Tool (BLAST)](#basic-local-alignment-search-tool-blast)
  - [Introduction to Bayesian statistics](#introduction-to-bayesian-statistics)
    - [The Cox-Jaynes axioms](#the-cox-jaynes-axioms)
    - [Maximum likelihood, maximum a posteriori, and Bayesian inference](#maximum-likelihood-maximum-a-posteriori-and-bayesian-inference)
    - [Multinomial and Dirichlet distributions](#multinomial-and-dirichlet-distributions)
    - [Estimation of frequency matrices](#estimation-of-frequency-matrices)
      - [Pseudocounts](#pseudocounts)
      - [Dirichlet mixtures](#dirichlet-mixtures)
  - [Hidden Markov Models (HMMs)](#hidden-markov-models-hmms)
    - [Viterbi decoding of the best state path](#viterbi-decoding-of-the-best-state-path)
    - [Forward-backward algorithm](#forward-backward-algorithm)
    - [Parameter estimation with known paths](#parameter-estimation-with-known-paths)
    - [Parameter estimation with known paths: Viterbi learning](#parameter-estimation-with-known-paths-viterbi-learning)
    - [Parameter estimation with known paths: Baum-Welch algorithm](#parameter-estimation-with-known-paths-baum-welch-algorithm)
    - [Numerical stability](#numerical-stability)
  - [Applications of HMMs](#applications-of-hmms)
    - [Profile HMMs](#profile-hmms)
      - [Estimation](#estimation)
      - [Database search](#database-search)
      - [Multiple alignments](#multiple-alignments)
    - [Gene finding](#gene-finding)
      - [Elements of gene prediction](#elements-of-gene-prediction)
      - [Prokaryotes vs. eukaryotes](#prokaryotes-vs-eukaryotes)
      - [Gene prediction by homology](#gene-prediction-by-homology)
      - [GENSCAN](#genscan)
  - [Expectation-Maximization for learning HMMs and motif finding](#expectation-maximization-for-learning-hmms-and-motif-finding)
    - [The EM algorithm](#the-em-algorithm)
    - [EM interpretation of the Baum-Welch algorithm for the learning of HMMs](#em-interpretation-of-the-baum-welch-algorithm-for-the-learning-of-hmms)
    - [EM for motif finding](#em-for-motif-finding)
      - [Multiple EM for Motif Elicitation (MEME)](#multiple-em-for-motif-elicitation-meme)
  - [Gibbs sampling for Motif Finding and Biclustering](#gibbs-sampling-for-motif-finding-and-biclustering)
    - [Markov Chain Monte Carlo Methods](#markov-chain-monte-carlo-methods)
    - [Gibbs Sampling](#gibbs-sampling)
    - [Gibbs Sampling for Motif Finding](#gibbs-sampling-for-motif-finding)
  - [Analysis of one and two-dimensional linear systems](#analysis-of-one-and-two-dimensional-linear-systems)
    - [Autonomous systems](#autonomous-systems)
    - [Continuous vs. discrete systems](#continuous-vs-discrete-systems)
    - [Equilibrium points + characterisation](#equilibrium-points--characterisation)
    - [Stability](#stability)
  - [Nonlinear system analysis](#nonlinear-system-analysis)
    - [Equilibrium points](#equilibrium-points)
    - [Stability analysis](#stability-analysis)
    - [Phase plane and phase portraits](#phase-plane-and-phase-portraits)
    - [Linearisation](#linearisation)
    - [Bifurcations](#bifurcations)
    - [Chaos](#chaos)
  - [Feedback](#feedback)
  - [Synchronisation](#synchronisation)
  - [References](#references)

## Introduction to molecular biology

**Central Dogma of Molecular Biology**: DNA makes RNA makes proteins.
(Different mechanisms for prokaryotes and eukaryotes.)

- Genes carry the information for the production of proteins.
- Transcription (from DNA to mRNA) by RNA polymerase.
- Translation (from mRNA to protein) by ribosomes.

There are 64 codons. There is always a start and a stop codon, with 6 reading frames.

Phylogeny of protein families: Genes or proteins can be related on the basis of their sequence.

### DNA

The DNA helix consists of Adenine, Thymine, Cytosine, Guanine.
They take the shape of complementary / double strands, `(A-T, G-C)`, that turn "clockwise", 10 nucleotides per revolution.

### RNA

Probably the ancestor of DNA. Serves as information messenger from DNA to protein.
Single strand (A-U bond weaker than A-T): Adenine - Uracil (vs. Thymine), Guanine - Cytosine.

#### Functional RNA

Importance of RNA as a molecule in its own right increasingly recognised (noncoding RNA).

### [Proteins](https://www.edx.org/learn/biology/rice-university-proteins-biology-s-workforce)

Proteins are _large polymers of 20 aminoacids_.

#### Basics

1. **Avogadro's number** allows conversion of weight into number of molecules: $6.02 \times 10^{23}$.
2. **First Law of Thermodynamics**: Energy can neither be created nor destroyed, but it can be transformed; e.g. photosynthesis converts light energy into chemical energy in the form of e.g. sugar.
3. **Second Law of Thermodynamics**: In an isolated system, the degree of disorder (entropy) can only increase; heat can never pass from a colder object to a warmer body without some other connected change; e.g. melted ice cream does not spontaneously refreeze.
4. **Free energy** is the useful energy in a system, i.e. energy that can do something, e.g. a ball can do work as it moves from the top of a hill to the bottom. It is related to two thermodynamic values - which tells us what work can be done and what directions reactions will proceed:
   1. **Enthalpy** is the measure of heat.
   2. **Entropy** is the measure of randomness.
      $$\Delta G = \Delta H - T \Delta S$$
5. **Covalent bonding** occurs if pairs of electrons are shared by atoms - these bonds are generally high energy; that is, it takes energy to break them.
6. **Non-covalent bonding** involves interactions without electron sharing, e.g. ionic interactions; low energy but contribute significantly to protein structure and function.
7. **Carbon chemistry** is key to living organisms; forms four valence bonds, that are stable but also can be reactive; allows constructing large complex molecules and polymers.
8. **Properties of water** determined by non-covalent hydrogen bonding are important to living organisms: high melting point, high boiling point, solvent properties.
9. An **acid** can act as a proton donor; a **base** can act as a proton acceptor; **pH** is a measure of the acidity (acid < 7) or basicity (base > 7) of an aqueous solution and is usually neutral (= 7) in living cells.
   $$pH \approx - \log[ H^+ ]$$
10. **Prokaryotes** are single-celled organisms without a nucleus to protect their DNA (small, e.g. bacteria, archaea); **eukaryotes** have a nucleus and can be single or multi-celled (large).
11. Phases of the **cell cycle**.

#### Protein Structure

1. Prokaryotic cells:
   - Cellular **precursors** (18-44 MW): $CO_2, H_2O, N_2, NH_3$, salts
   - **Intermediates** (50-250 MW): $\alpha$-keto acids, ribose, pyruvate, malate, acetate, malonate
   - **Building blocks** (100-350 MW): amino acids, nucleotides, simple sugars, fatty acids / glycerol
   - **Macromolecules** ($10^3 - 10^9$ MW): proteins, nucleic acids, polysaccharides, lipids
   - Supramolecular **assemblies** ($10^6 - 10^9$ MW): enzyme complexes, ribosomes, contractile systems, membranes
2. Eukaryotic cells:
   - **Organelles**: nucleus, mitochondria, chloroplasts
3. Types of **non-covalent** (_low energy, easily broken_) **bonding** in biological systems:
   - Hydrogen bonds: shared hydrogen between two molecules or parts of a molecule
   - Ionic / electrostatic interactions
   - van der Waals interactions: attraction between partially positive and partially negative atoms
   - Hydrophobic interactions / forces
   - _Large number of small forces creates flexibility of structures_.
4. Proteins are **polymers** assembled from amino acid units:
   - Amino acids: acidic, basic, apolar, polar
   - Structure, properties, abbreviations of side chains for 6 assigned amino acids: Glu, Lys, Trp, Ser, Gly, Pro
   - Each amino acid provides a unique side chain
5. **Levels of protein structure**:
   - Primary structure: Amino acid sequence; peptide bonds link amino acids to form a polymer (backbone for protein and side chain arrangement).
   - Secondary structure: $\alpha$-helix, $\beta$-sheet structures (identified by Pauling)
   - Tertiary structure: folding into 3 dimensions
   - Quaternary (oligomeric) structure: assembly into higher oligomers
6. **Protein folding**:
   - Amino acid sequence of a protein not functional without folding
   - 3-D structure is the active form of a protein
   - Non-covalent bonds between side chains (and peptide backbone interactions) create the “folded” form of the protein (IMPORTANT)
   - Folded form exhibits a biological activity
   - Protein folding funnels - energetic pathways to function: Lowest energy state is the folded structure. Large energy penalty for loss of entropy - can think of it as loss of options for different states so that the overall difference in energy between folded/unfolded is small.
   - Stabilizing energy for protein folding: Primarily non-covalent interactions

#### Protein Function

1. **Shape = Function**: Proteins must fold to carry out function.
   - Types of function: enzymes (cellular catalysts), structural proteins, regulatory proteins, et al.
   - Similar folds can have distinct functions. Conversely, dissimilar folds can have similar functions.
   - Individual subunits can associate to higher-order structures.
   - Some proteins are "intrinsically disordered": At least some parts of many proteins are not well folded, which allows formation of _multiple partners_ to carry out different processes. Especially found in proteins that are regulatory, more common in eukaryotes.
2. Proteins are essential to life:
   - Provide **structure**: external (collagens, keratins such as skin, fur, wool, claws, etc.) and internal (some structural proteins are fibers).
   - Most are **dynamic** (constantly in motion, not fixed in shape or position): some proteins create movement, e.g. transport (membrane proteins, muscles).
   - Bind a variety of **ligands**.
   - **Transport** materials: muscles, transport within cells, cell movement; transport across membranes.
   - Are **catalysts**: for metabolic pathways (enzymes).
   - Are **regulators** / signal carriers for cellular processes: DNA synthesis, RNA, synthesis, protein synthesis / breakdown, metabolic reactions.
3. Proteins bind a variety of "**ligands**": water, ions, sugars, amino acids, nucleotides, DNA, lipids, membranes, each other.
   - Strength of binding is measured by equilibrium binding constant (free energy!).
   - Individual proteins are selective for specific ligand(s).
4. Types of proteins: structural, binding, movement, enzymes.
   - Enzymes are catalysts that lower the energy barrier to a specific reaction.
5. **Amphiphilic** nature of **membrane proteins**: Membrane separates outside from inside of cell; protein exterior inside the membrane is hydrophobic, protein interior inside the membrane is hydrophilic.
   - Hemoglobin (Hb) carries oxygen in blood to tissues.
   - Myoglobin takes oxygen from Hb and stores for use.

#### Protein Function & Purification

1. **Enzymes** are the catalysts that carry out specific reactions in complex mixtures that otherwise would not occur by _lowering energy barrier to reaction_.
2. **Gibbs free energy** ($\Delta G$) characterizes the reaction and coupling of reactions.
3. Feedback inhibition and allosteric regulation process for enzymes
4. Understanding structure / function requires purification and detailed biochemical characterisation
5. Conformational changes required for function (dynamic nature)
6. Complex proteins assembly to oligomers, higher order structures
7. Regulation of protein function: Allostery; covalent modification (e.g., phosphorylation, acetylation)
8. Protein dynamics/intrinsic disorder (repeated for emphasis!), conformational flexibility (adaptation)
9. Protein interaction networks
10. Methods for purifying proteins: Sources, breaking cells, separating cellular contents, crude methods (precipitation by salts, temperature, pI), chromatographic methods

#### Protein Characterisation

1. Protein characterization: Spectroscopy, fluorescence, antibodies, assays
2. Protein characterization/quantitation: Monoclonal antibodies - ELISA, immunoprecipitation
3. Monomer and oligomer size determination: SDS gel electrophoresis (Weber & Osborn), 2D gels, Western blotting, gel chromatography
4. Protein structure determination: X-ray crystallography, NMR
5. Biochemical methods: Leverage the function of the protein (reaction catalyzed, inhibitors, activators) to characterize interaction networks

## Sequence alignment

### Similarity vs. Homology

Sequences are **similar** if they are sufficiently resembling at the sequence level (DNA, proteins). Similarity can arise from: homology (common ancestor), convergence (functional constraints), chance.

**Molecular evolution**: natural selection, imperfect replication (point mutations), gene duplications (create gene families). The degree of similarity depends on how many mutations occurred since the "fork".

**Phylogeny** is the reconstruction of molecular evolution. Thus relationships between genes and proteins can be inferred on the basis of their sequences.

Sequences are **homologous** if they arise from a common ancestor. Homologous sequences are **paralogous** if their differences involve a gene duplication event, and **orthologous** if their differences do _not_ involve a gene duplication event.

Homologous proteins have comparable structures and potentially similar functions:

- ortholog: similar cellular role.
- paralog: similar biochemical function.

Homology in **comparative genomics**: Conserved regions arise from evolutionary pressure and are therefore functionally important. Genes can be predicted by comparing genomes at an appropriate evolutionary distance (e.g. mouse and human; human and chimp is too close and would not work).

The alignment of two residues can be more or less likely. To compute the quality of an alignment, we assign a gain or a penalty to the alignment of two residues. Gaps also have a penalty.

Fill in scores by looking up pairwise scores in the subsitution matrix (e.g. BLOSUM50) and defining a gap penalty for gaps (e.g. -8) for two sequences, e.g. for the following _query_ and _target_:

`HEAGAWGHE-E`

`--P-AW-HEAE`

### Dynamic programming

**"What-if" matrix (deletions, insertions, substitutions)**: Every path is an alignment. Find the minimum penalty / maximum score _path_ through the penalty table. (It is possible to have more than one best alignment, i.e. with the same maximal score.)

TODO: Bellman optimality principle w.r.t. DP? Example: finding the shortest train route between two cities.

**No-shortcut principle**: If you know that a path is the shortest path from the start to the end node, then it is also the shortest path between any two nodes on the path.

### Global alignment: Needleman-Wunsch algorithm

1. Fill in the top left with value zero, and other cells in first row and first column with the value of the gap penalty.

   $$F(0, 0) = 0, \forall j > 0, \, \quad F(0, j) = d, \quad \forall i > 0, \,F(i,  0) = d$$

2. Complete the maximum alignment score table progressively starting from the top left, and use traceback pointers for the paths from the highest score to the starting point in the top left.

   $$
   F(i, j) = \max{\begin{cases}
     F(i-1, j-1) + s(x_i, y_j) & \text{substitution} \\
     F(i-1, j) - d & \text{deletion} \\
     F(i, j-1) - d & \text{insertion} \\
   \end{cases}}
   $$

Note the algorithm is similar to the Levenshtein distance, which is used in NLP to calculate the minimum number of edits to transform one string into another, except the Levenshtein distance does not support weighted operations (there all edits are treated equally).

### Local alignment: Smith-Waterman algorithm

Best alignment between _subsequences_.

If the current alignment has a negative score, it is better to start a new alignment.

$$
F(i, j) = \max{\begin{cases}
  0 & \text{restart} \\
  F(i-1, j-1) + s(x_i, y_j) & \text{substitution} \\
  F(i-1, j) - d & \text{deletion} \\
  F(i, j-1) - d & \text{insertion} \\
\end{cases}}
$$

Traceback pointers then go from the highest score to zero (not all the way back to the starting point; local alignment!).

### Significance Evaluation

The use of such algorithms is to compare a query with a database of (randomised, i.e. shuffled) sequences, and finding the best matches, i.e. that have the maximal score.

For an ungapped alignment, the _score_ of a match is the sum of the i.i.d. random contributions and follows a normal distribution (by the CLT). For a normal distribution, the distribution of the _maximum_ $M_N$ of a series of $N$ random samples follows the **extreme value distribution (EVD)**:

$$P(M_N \geq x) = e^{ -KNe^{- \lfloor x \rfloor} }$$

The parameters are derived from $P_i$ and $s(i, j)$.

For gapped alignments the EVD has the following form (even though the random contributions are not normally distributed; where $S$ = the score, $m$ = size of the database, $n$ = length of the query):

$$P(S \geq x) = e^{ -Kmne^{- \lambda S} }$$

The parameters can be estimated by regression.

An alignment is significant if its probability is sufficiently small (e.g. $P < 0.01$).

### Substitution matrices (PAM, BLOSUM50) and gap penalties

A substitution matrix can be computed by looking at the confirmed alignments (with gaps) and computing the amino acid frequencies $q_a$, the substitution frequencies $p_{ab}$, and the gap function $f(g)$.

The following likelihood model (drop the gapped positions) can be constructed as an odds ratio of alignments over random sequences:

$$\frac{P(x, y \mid M)}{P(x, y \mid R)} = \frac{\prod_i{p_{x_i y_i}}}{ \prod_i{q_{x_i}} \prod_j{q_{y_j}}}$$

The substitution matrix is:

$$s(a, b) = \log{\frac{p_{ab}}{q_a q_b}}$$

A positive score occurs when $p_{ab} > q_a q_b$, i.e. when they occur together more often than would be epxected from individual chance.

#### PAM (Point Accepted Mutations) matrix

Problems: Garbage data in, garbage out:

- Alignments are not independent for related proteins.
- Different alignments correspond to different evolution times.

PAM1 matrix: 1% Point Accepted Mutations (PAM1); that is, only sequences that are 99% identical.

PAM250 is 250% Point Accepted Mutations (~20%
similarity) = 250ste power of PAM1

#### BLOSUM (BLOCKS SUbstitution Matrix)

PAM does not work so well at large evolutionary
distances.

- Ungapped alignments of protein families from the
  BLOCKS database.
- Group sequences with more than $L\%$ identical amino acids (e.g., BLOSUM62).
- Use the substitution frequency of amino acids between the different groups (with correction for the group size) to derive the substitution matrix.

### Basic Local Alignment Search Tool (BLAST)

For large databases, Smith-Waterman local alignment is too slow. Basic Local Alignment Search Tool (BLAST) is a fast heuristic algorithm for local alignment.

- BLASTN – nucleotide query on nucleotide database
- BLASTX – translated nucleotide query on protein database (translation into the six reading frames)
- TBLASTN – protein query on translated nucleotide db
- TBLASTX – translated nucleotide query on translated nucleotide db
- BLASTP – protein query on protein database
  1. Find all words of length $w$ (e.g. $w = 3$) for which there is a match in the query sequence with score at least $T$ (e.g. $T = 11$) for the chosen substitution matrix (e.g. BLOSUM62 with gap penalty $10 + g$).
  2. Use a finite state automaton to find all matches (_hits_) with the word list in the database.
  3. Check which hits have another hit without overlap within a distance of $A$ (e.g. $A = 40$; two-hits); the distance must be identical on the query and on the target.
  4. Extend the left hit of the two-hits in both directions by ungapped alignment; stop the extension when the score drops by $X_g$ (e.g. $X_g = 40$) under the best score so far (high scoring segment pair HSP).
  5. Extend the HSPs with normalized score above $S_g$ ($S_g = 22$ bits) by gapped alignment; stop the extension when the score drops by $X_g$ (e.g., $X_g = 40$) under the best score so far; select the best gapped local alignment.
  6. Compute the significance of the alignments ; for the significant alignments, repeat the gapped alignment with a higher dropoff parameter $X_g$ for more accuracy.

## Introduction to Bayesian statistics

### The Cox-Jaynes axioms

The **frequentist** view of probability (the limit of relative frequencies in repeated trials) is somewhat circular because of the dependence on the CLT (which itself is proven using probability theory that already assumes the frequentist interpretation is valid).

The **Bayesian** approach views probabilities as models of the uncertainty regarding propositions within a given domain:

- Deduction: $\text{if} \, ( A \Rarr B \cap A = \text{true} ) \, \text{then} \, B = \text{true}$.
- Induction / abduction: $\text{if} \, ( A \Rarr B \cap B = \text{true} ) \, \text{then} \, A \, \text{becomes more plausible}$.
- Probabilities satisfy Bayes' rule.

A **proposition** $A$ may be true or false. A **domain** $\mathcal{D}$ contains the available information about a situation. Then define $\angle(A = \text{true} \mid \mathcal{D})$ as a **belief** regarding proposition $A$ given the domain knowledge $\mathcal{D}$.

The Cox-Jaynes axioms allow the buildup of a probabilistic framework, consistent with both Bayesian and frequentist views, with the following minimal assumptions:

1. Suppose we can compare beliefs, and suppose the comparison is transitive (i.e. $\big( \angle(A \mid \mathcal{D}) > \angle(B \mid \mathcal{D}) \big) \land \big( \angle(B \mid \mathcal{D}) > \angle(C \mid \mathcal{D}) \big) \Rarr \big( \angle(A \mid \mathcal{D}) > \angle(C \mid \mathcal{D}) \big)$). Then there exists an ordering relation, and so $\angle$ is a number.
2. Suppose there exists a fixed relation between the belief in a proposition and the belief in its negation, $\angle(\bar{A} \mid \mathcal{D}) = f\big( \angle(A \mid \mathcal{D}) \big)$. Then $\angle(A \mid \mathcal{D}) = \angle(B \mid \mathcal{D}) \Rarr \angle(\bar{A} \mid \mathcal{D}) = \angle(\bar{B} \mid \mathcal{D})$.
3. Suppose there exists a fixed relation between the belief in the union of two propositions and the belief in the first proposition and the belief in the second proposition given the first one, $\angle(A, B \mid \mathcal{D}) = g\big( \angle(A \mid \mathcal{D}), \angle(B \mid A, \mathcal{D}) \big)$. Then, after rescaling the beliefs,

   $$P(A \mid \mathcal{D}) + P(\bar{A} \mid \mathcal{D}) = 1$$

   $$P(A, B \mid \mathcal{D}) = P(B \mid A, \mathcal{D}) P(A \mid \mathcal{D})$$

Thus, under the Cox-Jaynes axioms, Bayes' rule can always be applied - independently of the specific definition of the probabilities.

- Bayes' rule holds for any distribution: $P(Y \mid X, \mathcal{D}) = \frac{P(X \mid Y, \mathcal{D}) P(Y \mid \mathcal{D})}{P(X \mid \mathcal{D})}$.
- Bayes' rule holds for specific realisations: $P(Y = y \mid X = x, \mathcal{D}) = \frac{P(X = x \mid Y = y, \mathcal{D}) P(Y = y \mid \mathcal{D})}{P(X = x \mid \mathcal{D})}$.

It is important to set up the problem within the right domain $\mathcal{D}$!

### Maximum likelihood, maximum a posteriori, and Bayesian inference

Consider a domain $\mathcal{D}$, observational data $D$, and a model $M$ with parameters $\bm{\theta}$.

Bayes' rule: posterior is proportional to likelihood of the data times prior, $P(\theta \mid D, M) = \frac{P(D \mid \theta, M) P(\theta \mid M)}{P(D \mid M)}$.

Generative models (of the likelihood of the data): $P(D \mid M, \theta) = \prod_{i=1}^L{\theta_{D_i}}$. We want to find the model that describes our observations:

$$\theta^\text{MLE} = \argmax_\theta{P(D \mid \theta, M)}$$

$$\theta^\text{MAP} = \argmax_\theta{P(\theta \mid D, M)} \propto \argmax_\theta{P(D \mid \theta, M) P(\theta \mid M)}$$

$$\theta^\text{PME} = \int{\theta P(\theta \mid D, M) \, d\theta}$$

Updating the probability of the parameters with new observations $D$:

$$P(\theta \mid D, M) = \frac{P(D \mid \theta, M) P(\theta \mid M)}{P(D \mid M)} = \frac{P(D \mid \theta, M) P(\theta \mid M)}{\int_\omega{ P(D \mid \omega, M) P(\omega \mid M) \, d\omega }}$$

1. Choose a reasonable prior $P(\theta \mid M)$.
2. Add the information from the data $\frac{P(D \mid \theta, M)}{\int_\omega{ P(D \mid \omega, M) P(\omega \mid M) \, d\omega }}$.
3. Get the updated distributions of the parameters $P(\theta \mid D, M)$ (often log-based).

Discrete and continuous **marginalisation** to introduce or remove a variable wherever appropriate:

$$\sum_{y=1}^Y{P(Y = y)} = 1, \qquad \sum_{y=1}^Y{P(X, Y = y)} = P(X)$$

$$\int_{y \in Y}^K{P(Y = y)} = 1, \qquad \int_{y \in Y}^K{P(X, Y = y)} = P(X)$$

**Inference**: if $K$ is not too large, can compute all the likelihoods and prior probabilities

$$P(\theta = i \mid D, M) = \frac{P(D \mid \theta = i, M) P(\theta = i \mid M)}{\sum_{j=1}^K{P(D \mid \theta = j, M) P(\theta = j \mid M)}}$$

### Multinomial and Dirichlet distributions

Consider $K$ independent outcomes with probabilities $P(X = i) = \theta_i, \, i = 1, \dots, K$ (for $K = 2$, Bernoulli variable with binomial distribution).

The multinomial distribution is used to model biological sequences. It gives the number of times different outcomes are observed:

$$\mathcal{M}(n; \theta) = P(N_1 = n_1, N_2 = n_2, \dots, N_K = n_K) = \frac{1}{M((n_1, n_2, \dots, n_K))} \prod_{i=1}^K{\theta_i^{n_i}}$$

$$M((n_1, n_2, \dots, n_K)) = \frac{\prod_{k=1}^K{n_k!}}{\Big( \sum_{k=1}^K{n_k} \Big)!}$$

The Dirichlet distribution gives the probability of the $P(X = i)$ s for $a_i > 0, \, i = 1, \dots, K$:

$$\mathcal{D}(\theta; a) = \frac{1}{Z(a)} \prod_{i=1}^K{\theta_i^{(a_i - 1)}}, \qquad Z(a) = \int_{\theta \in \Theta}{\prod_{i=1}^K{\theta_i^{(a - 1)}} \, d\theta} = \frac{\prod_{i=1}^K{\Gamma(a_i)}}{\Gamma\Big( \sum_{k=1}^K{a_i} \Big)} \text{ s.t. } \int_\theta{P(\theta \mid a) \, d\theta} = 1$$

The Gamma function is the generalisation of the factorial function to real numbers: $\Gamma(n) = (n-1)!, \, \Gamma(x + 1) = x \Gamma(x)$.

The Dirichlet distribution is the natural prior for sequence analysis because it is conjugate to the multinomial distribution; that is, given a Dirichlet prior and multinomial observations, the posterior also follows a Dirichlet distribution.

### Estimation of frequency matrices

Estimation on the basis of counts (e.g., Position-Specific Scoring Matrix in PSI-BLAST): count the number of instances in each column, if $N \gg$: $\theta_A = \frac{n_A}{N}, \, \theta_C = \frac{n_C}{N}, \, \theta_T = \frac{n_T}{N}, \, \theta_G = \frac{n_G}{N}$.

This is the maximum likelihood estimate for $P(n \mid \theta) = P(N_A = n_A, N_C = n_C, N_G = n_G, N_T = n_T \mid \theta_A, \theta_C, \theta_G, \theta_T)$:

$$\theta^\text{MLE} = \argmax_\theta{P(n \mid \theta)} = \frac{n}{N}$$

#### Pseudocounts

If we have a limited number of counts, the maximum likelihood estimate will not be reliable (e.g., for symbols not observed in the data). In such a situation, we can combine the observations with prior knowledge, e.g. a Dirichlet prior $\mathcal{D}(\theta; a)$ s.t. the Bayesian update is

$$
\begin{align*}
  P(\theta \mid n) & = \frac{P(n \mid \theta) \mathcal{D}(\theta; a)}{P(n)}
  = \frac{\frac{1}{M(n)} \prod_{i=1}^K{\theta_i^{n_i}} \quad \frac{1}{Z(a)} \prod_{i=1}^K{\theta_i^{(a_i - 1)}} }{P(n)} \\
  & = \frac{\prod_{i=1}^K{\theta_i^{(n_i + a_i - 1)}}}{M(n) P(n) Z(a)}
  = \frac{Z(n + a)}{M(n) P(n) Z(a)} \mathcal{D}(\theta; n + a) \\
  & = \mathcal{D}(\theta; n + a) \qquad \text{(because both distributions are normalised)}
\end{align*}
$$

$$
\begin{align*}
  \theta_i^\text{PME} & = \int{\theta_i \mathcal{D}(\theta; n + a) \, d\theta}
  = \frac{1}{Z(n + a)} \int{\theta_i \prod_k{\theta_k^{n_k + a_k - 1}} \, d\theta}
  = \frac{Z(n + a + \delta_i)}{Z(n + a)} \\
  & = \frac{n_i + a_i}{N + A}, \quad A = \sum_i{a_i}
\end{align*}
$$

- The prior contributes to the estimation through pseudo- observations
- If few observations are available, then the prior plays an important role
- If many observations are available, then the pseudocounts play a negligible role

With uniform prior, ML = MAP.

$$\theta_i = \frac{n_i + a - 1}{N + A - K}$$

#### Dirichlet mixtures

Sometimes the observations are generated by a heterogeneous process (e.g., hydrophobic vs. hydrophilic domains in proteins). In such situations, we should use different priors in function of the context. But we do not necessarily know the context beforehand. A possibility is the use of a Dirichlet mixture.

The frequency parameter $\theta$ can be generated from $m$ different sources $S$ with different Dirichlet parameter vectors $\alpha^k$.

$$P(\theta) = \sum_k{P(S = k) \mathcal{D}(\theta; \alpha^k)}$$

$$P(S = k \mid n) = \frac{P(n \mid S = k) P(S = k)}{\sum_l{P(n \mid S = l) P(S = l)}} = \frac{P(n \mid S = k) P(S = k) / Z(\alpha^k)}{\sum_l{P(n \mid S = l) P(S = l) / Z(\alpha^l)}}$$

$$\theta_i^\text{PME} = \sum_k{P(S = k \mid n) \frac{n_i + \alpha_i^k}{N + A^k}}$$

- The different components of the Dirichlet mixture are first considered as separate pseudocounts $\frac{n_i + \alpha_i^k}{N + A^k}$.
- These components are then combined with a weight depending on the likelihood of the Dirichlet component $P(S = k \mid n)$.

## Hidden Markov Models (HMMs)

A sequence $x = x_1, x_2, \dots, x_L, \, x_i \in \mathcal{A} = \{A, C, T, G\}$ can be modelled probabilistically as a **Markov chain**, with transition probabilities $a_{st} = P(x_i = t \mid x_{i-1} = s)$. By the **Markov property**:

$$
\begin{align*}
  P(x) & = P(x_L \mid x_{L-1}, \dots, x_1) P(x_{L-1} \mid x_{L-2}, \dots, x_1) \dots P(x_1) \\
  & = P(x_L \mid x_{L-1}) P(x_{L-1} \mid x_{L-2}) \dots P(x_1) \\
  & = P(x_1) \prod_{i=2}^L{a_{x_{i-1} x_i}}
\end{align*}
$$

The length distribution is not modelled, i.e. $P(\text{length} = L)$ is undefined. Solution: Define sequence: $a, x_1, \dots, x_L, \omega$. Then $a_{as} = P(x_1 = s), a_{t\omega} = P(\omega \mid x_L = t)$. _The probability to observe a sequence of a given length decreases with the length of the sequence_.

A casino uses mostly a fair die but switches sometimes to a loaded die. We observe the outcome $x$ of the successive throws but want to know when the die was fair or loaded.

In a HMM, the symbol sequence $x$ is observed, and the objective is to reconstruct the hidden state sequence path.

- Transition probabilities $a_{kl} = P(\pi_i = l \mid \pi_{i-1} = k, \theta)$
- Emission probabilities $e_k(b) = P(x_i = b \mid \pi_i = k, \theta)$
- Joint probability of the sequence and the path $P(x, \pi \mid \theta) = \prod_{i=0}^L{e_{\pi_i}(x_i) a_{\pi_i \pi_{i+1}}}$

### Viterbi decoding of the best state path

The most probable path $\pi^* = \argmax_\pi{P(x, \pi)} = \argmax_\pi{P(\pi \mid x)}$ can be found with dynamic programming: Define $v_k(i) = \max_{\pi_1, \dots \pi_{i-1}}{P(x_1, \dots, x_i, \pi_1, \dots, \pi_{i-1}, \pi_i = k)}$ as the probability of the most probable path that ends in state $k$ for the emission of symbol $x_i$. Then this probability can be computed recursively as

$$v_l(i+1) = e_l(x_{i+1}) \max_k\big(v_k(i) a_{kl}\big)$$

- Initial condition: sequence in beginning state $v_0(0) = 1, v_k(0) = 0, \forall k > 0$
- Recursion ($i = 1, \dots, L$): $v_l(i+1) = e_l(x_{i+1}) \max_k{ \big(v_k(i) a_{kl}\big) }$, $\text{ptr}_i(l) = \argmax_k{ \big(v_k(i-1) a_{kl}\big) }$
- Termination: $P(x, \pi^*) = \max_k{\big(v_k(L) a_{k0}\big)}$, $\pi_L^* = \argmax_k{\big(v_k(L) a_{k0}\big)}$
- Traceback ($i = L, \dots, 1$): $\pi_{i-1}^* = \text{ptr}_i{(\pi_i^*)}$

### Forward-backward algorithm

- The forward algorithm for the computation of the probability of a sequence
- The backward algorithm for the computation of state probabilitie

The forward algorithm computes the probability of a sequence $x$ w.r.t. a HMM:

$$P(x) = \sum_\pi{P(x, \pi)}$$

Define $f_k(i) = P(x_1, \dots, x_i, \pi_i = k)$ as the probability of the sequence for the paths that end in state $k$ with the emission of symbol $x_i$. Then it can be computed as

$$f_l(i+1) = e_(x_{i+1}) \sum_k{f_k(i) a_{kl}}$$

The forward algorithm grows the total probability dynamically from the beginning to the end of the sequence.

- Initial condition: sequence in beginning state ($i=0$): $f_0(0) = 1, f_k(0) = 0, \forall k > 0$
- Recursion ($i = 1, \dots, L$): $f_l(i) = e_l(x_i) \sum_k{f_k(i-1) a_{kl}}$
- Termination: all states converge to the end state $P(x) = \sum_k{f_k(L) a_{k0}}$

The backward algorithm computes the probability of the complete sequence together with the condition that symbol $x_i$ is emitted from state $k$. This is important to compute the probability of a given state at symbol $x_i$:

$$
\begin{align*}
  P(x, \pi_i = k) & = P(x_1, \dots, x_i, \pi_i = k) P(x_{i+1}, \dots, x_L \mid x_1, \dots, x_i, \pi_i = k) \\
   & = P(x_1, \dots, x_i, \pi_i = k) P(x_{i+1}, \dots, x_L \mid \pi_i = k)
\end{align*}
$$

- $P(x_1, \dots, x_i, \pi_i = k)$ can be computed by the forward algorithm.
- The backward algorithm grows the probability $b_k(i) = P(x_{i+1}, \dots, x_L \mid \pi_i = k)$ dynamically backwards (from end to beginning) - boundary condition: start in end state:
  - Initialisation ($i = L$): $b_k(L) = a_{k0}, \forall k$
  - Recursion ($i = L-1, \dots, l$): $b_k(i) = \sum_l{a_{kl} e_l(x_{i+1}) b_l(i+1)}$
  - Termination: $P(x) = \sum_l{a_{0l} e_l(x_1) b_l(1)}$

The posterior probability of the state can be computed using the forward and backward probabilities:

$$P(\pi_i = k \mid x) = \frac{f_k(i) b_k(i)}{P(x)}$$

Instead of using the most probable path for decoding (Viterbi) the path of the most probable states can be used:

$$\hat{\pi}_i = \argmax_k{P(\pi_i = k \mid x)}$$

### Parameter estimation with known paths

Assumption: HMM architecture known. (Choice of architecture is an essential design choice.) Architecture may include "silent states" for gaps.

The score of the model is the likelihood of the parameters given the training data $D$ of $N$ sequences $x^1, \dots, x^N$:

$$\text{Score}(D, \theta) = \log{P(x^1, \dots, x^N \mid \theta) = \sum_{j=1}^N{\log{P(x^j \mid \theta)}}}$$

If the state paths are known, the parameters are estimated through counts (how often is a transition used, how often is a symbol produced by a given state):

$$a_{kl} = \frac{A_{kl}}{\sum_{l'}{A_{kl}}}, \, e_k(b) = \frac{E_k(b)}{\sum_{b'}{E_k(b')}}$$

using pseudocounts:

- $A_{kl}$ is the number of transitions from $k$ to $l$ in training set + pseudocount $r_{kl}$
- $E_k(b)$ is the number of emissions of $b$ from $k$ in training set + pseudocount $r_k(b)$

### Parameter estimation with known paths: Viterbi learning

Iterative method:

- Suppose that the parameters are known and find the best path.
- Use Viterbi decoding to estimate the parameters.
- Iterate till convergence. Viterbi training converges exactly in a finite number of steps.

Viterbi training does not maximise the likelihood of the parameters.

$$\theta^\text{Vit} = \argmax_\theta{P(x^1, \dots, x^N \mid \theta, \pi_\theta^*(x^1), \dots, \pi_\theta^*(x^N))}$$

### Parameter estimation with known paths: Baum-Welch algorithm

Strategy: use the expected value for the transition and emission counts (instead of using only the best path)

- Initialization: Choose arbitrary model parameters
- Recursion:
  - Set all transitions and emission variables to their pseudocount
  - For all sequences $j = 1, \dots, n$
    - Compute $f_k(i)$ for sequence $j$ with the forward algorithm
    - Compute $b_k(i)$ for sequence $j$ with the backward algorithm
    - Add the contributions to $A$ and $E$
  - Compute the new model parameters $a_{kl}, e_k(b)$
  - Compute the log-likelihood of the model
- End: stop when the log-likelihood does not change more than by some threshold or when the maximum number of iterations is exceeded

### Numerical stability

Many expressions contain products of many probabilities. This causes underflow when we compute these expressions.

- For Viterbi, this can be solved by working with the logarithms
- For the forward and backward algorithms, we can work with an approximation to the logarithm or by working with rescaled variables

## Applications of HMMs

### Profile HMMs

#### Estimation

Deletions could be modeled by shortcut jumps between states. Problem: the number of transitions grows quadratically. Other solution: use parallel states that do not produce any symbol (**silent states**).

Zero probabilities in HMM causes the rejection of sequences containing previously unseen residues. To avoid this problem, add **pseudocounts** (add extra counts as if prior data was available).

#### Database search

The estimated model can be used to detect new members of the protein family in a sequence database (more sensitive than PSI-BLAST).

For each sequence in the datbase, $P(x, \pi^* \mid M)$ (Viterbi) or $P(x \mid M)$ (forward-backward) is computed; in practice, with log-odds (w.r.t. the random model $P(x \mid R$).

#### Multiple alignments

Through Viterbi (search for the best alignment path), we can align sequences w.r.t a profile HMM.

If the sequences are not aligned, it is possible to train a profile HMM to align them:

- Initialization: choose the length of the profile HMM
  - Length of profile HMM is number of match states $\neq$ sequence length
- Training: estimate the model via Viterbi training or Baum-Welch training
  - Heuristics to avoid local minimas
- Multiple alignment: use Viterbi decoding to align sequences

### Gene finding

#### Elements of gene prediction

- Sources of evidence (positive and negative)
  - Sequence similarity to known genes (e.g., found by BLASTX)
  - Statistical measure of codon bias
  - Template matches to functional sites (e.g., splice site)
  - Similarity to features not likely to overlap coding sequence (e.g., Alu repeats)
- The structure must respect the biological grammar (promoter, exon, intron, ...)

Search by signal vs. search by content:

- Search by signal
  - Detect short signals in the genome
  - E.g., splice site, signal peptide, glycosylation site
  - Neural networks can be useful here
- Search by content
  - Detect extended regions in the genome
  - e.g., coding regions, CpG islands
  - Hidden Markov Models are useful here
- Gene finding algorithms combine both
  - Hidden Markov Models can be used to predict genes
  - Homology to a known gene is also a strong method for detecting genes
  - More and more gene prediction packages combine both approaches

#### Prokaryotes vs. eukaryotes

Problems for prokaryotes:

- Short genes are hard to detect
- Operons
- Overlapping genes

Amino-acids and the genetic code: 64 codons, start & stop codon, 6 reading frames.

- Sequence can be translated into the six possible reading frames to check for start and stop codons.
- **Codon bias**: In coding sequences, genomes have specific biases for the use of codons encoding the same amino acid.
- **Coding potential**:
  - Most coding potentials are based on analysis of codon usage
  - The HMMs keeps track of some kind of average coding potential around each position
  - The increase and decreae of the coding potential will “push” the HMM in and out of the exons
- **Promoter region** contains the elements that control the expression of the gene. Its prediction is difficult.
- **Intron-exon splicing**

#### Gene prediction by homology

- Coding regions evolve more slowly than noncoding ones (conserved by natural selection because of their functional role)
- Not only the protein sequence but also the gene structure can be conserved
- Use standard homology methods
- Gene syntax must be respected

Procrustes:

- Find all possible blocks (exons) on the basis of acceptor/donor location
- Look which blocks can be aligned with model sequences
- Look for best alignment of blocks with the query sequence

Advantages of gene prediction by homology:

- Recognition of short exons and atypical exons
- Correct assembly of complex genes (> 10 exons)

Disadvantages of gene prediction by homology:

- Genes without known homologs are missed
- Good homologs necessary for the prediction of the gene
  structure
- Very sensitive to sequencing errors

#### GENSCAN

Gene prediction with Hidden Semi-Markov Models, used for the annotation of the human genome in the HGP.

## Expectation-Maximization for learning HMMs and motif finding

EM serves to find a maximum likelihood solution. This can also be achieved by gradient descent. But the computation of the gradients of the likelihood $P(D \mid l)$ is often difficult.

By introducing the unobserved data in the EM algorithm, we can compute the Expectation step more easily

### The EM algorithm

Consider the maximum likelihood estimate $\theta^* = \argmax_\theta{P(D \mid \theta)} = \argmax_\theta{\ln{P(D \mid \theta)}}$.

Assume an algorithm that tries to optimise the likelihood. The change in likelihood between two iterations of the algorithms is

$$\Delta(\theta_{i+1}, \theta_i) = \ln{P(D \mid \theta_{i+1})} - \ln{P(D \mid \theta_i)} = \ln\frac{P(D \mid \theta_{i+1})}{P(D \mid \theta_i)}$$

The likelihood is difficult to compute. We use a simpler generative model based on unobserved data (data augmentation):

$$
\begin{align*}
  \Delta(\theta_{i+1}, \theta_i) & = \ln\frac{\mathbb{E}_{m \mid \theta}[P(D, m \mid \theta_{i+1})]}{P(D \mid \theta_i)} \\
  & = \ln\frac{\int_m{P(D, m \mid \theta_{i+1}) \, dm}}{P(D \mid \theta_i)}
  = \ln\frac{\int_m{P(D \mid m, \theta_{i+1}) P(m \mid \theta_{i+1}) \, dm}}{P(D \mid \theta_i)} \\
  & = \ln\int_m{\frac{P(D, m \mid \theta_{i+1}) \, dm}{P(D \mid \theta_i)}}
  = \ln\int_m{\frac{P(D \mid m, \theta_{i+1}) P(m \mid \theta_{i+1}) \, dm}{P(D \mid \theta_i)}}
\end{align*}
$$

Problem: the expression contains the logarithm of a sum (or integral).

**Jensen's inequality**: $\sum_j{\lambda_j} = 1 \Rarr \ln{\sum_j{\lambda_j y_j}} \geq \sum_j{\lambda_j \ln{y_j}}$

$$
\begin{align*}
  \Delta(\theta_{i+1}, \theta_i) & = \ln\sum_m{\frac{P(D, m \mid \theta_{i+1})}{P(D \mid \theta_i)}}
  = \ln\sum_m{\frac{P(D, m \mid \theta_{i+1}) P(m \mid D, \theta_i)}{P(D \mid \theta_i) P(m \mid D, \theta_i)}} \\
  & \geq \sum_m{P(m \mid D, \theta_i) \ln\frac{P(D, m \mid \theta_{i+1})}{P(D \mid \theta_i)}}
  = \delta(\theta_{i+1}, \theta_i)
\end{align*}
$$

Lower bound for the variation of the likelihood:

$$\ln{P(D \mid \theta_{i+1})} \geq \ln{P(D \mid \theta_i)} + \delta(\theta_{i+1}, \theta_i)$$

TODO: finish derivation

**Generalised EM**: It is not absolutely necessary to maximise $Q$ at the expectation step. If $Q > 0$, convergence can also be achieved. This algorithm applies when the results of the expectation step are too complex to maximise directly.

### EM interpretation of the Baum-Welch algorithm for the learning of HMMs

We want to estimate the parameters of a HMM (transition and emission probabilities that maximize the likelihood of the sequence) $\theta^* = \argmax_\theta{P(x \mid \theta)}$.

The unobserved data are the paths $\pi$: $P(x \mid \theta) = \sum_\pi{P(x, \pi \mid \theta)}$

The EM algorithm is then

$$\theta_{i+1}^\text{EM} = \argmax_\theta{Q_{\theta_i^\text{EM}}(\theta)} = \argmax_\theta{\sum_\pi{ P(\pi \mid x, \theta_i^\text{EM}) \ln{P(x, \pi \mid \theta)} }}$$

The generative model gives the joint probability of the sequence and the path.

- Define the number of times that a given probability gets used for a given path: $A_{kl}(\pi)$
- Define the number of times that a given emission is observed for a given sequence and a given path: $E_k(b, \pi)$

The joint probability of the sequence and the path can be written as

$$P(x, \pi \mid \theta) = \prod_{k=1}^M{ \prod_b{ \big( e_k(b) \big)^{E_k(b, \pi)} } } \prod_{k=0}^M{ \prod_{l=1}^{M+1}{ \big( a_{kl} \big)^{A_{kl}(\pi)} } }$$

Taking the logarithm,

$$Q_{\theta_i^\text{EM}}(\theta) = \sum_\pi{ P(\pi \mid x, \theta_i^\text{EM}) } \Bigg[ \sum_{k=1}^M{ \sum_b{E_k(b, \pi) \ln{e_k(b)}} } + \sum_{k=0}^M{\sum_{l=1}^{M+1}{ A_{kl}(\pi) \ln{a_{kl}} }} \Bigg]$$

Now,

- Define the expected number of times that a transition gets used (independently of the path): $A_{kl} = \sum_\pi{P(\pi \mid x, \theta_i) A_{kl}(\pi)}$
- Define the expected number of times that an emission is observed (independently of the path): $E_k(b) = \sum_\pi{P(\pi \mid x, \theta_i) E_k(b, \pi)}$

Given that $P(x, \pi \mid \theta)$ is independent of $k$ and $b$, reorder the sums and use the definitions of $A_{kl}$ and $E_k(b)$:

$$Q_{\theta_i}^\text{EM}(\theta) = \sum_{k=1}^M{ \sum_b{E_k(b) \ln{e_k(b)}} } + \sum_{k=0}^M{ \sum_{l=1}^{M+1}{A_{kl} \ln{a_{kl}}} }$$

Find candidates for $a_{kl}, e_k(b)$ to maximise $Q$:

- For the $A$ term, define the following candidate for the optimum: $a_{kl}^0 = \frac{A_{kl}}{\sum_m{A_{km}}}$. Compare with other parameter choices: $\sum_{k=0}^M{ \sum_{l=1}^{M+1}{ A_{kl} \ln{a_{kl}^0} } } - \sum_{k=0}^M{ \sum_{l=1}^{M+1}{ A_{kl} \ln{a_{kl}} } } = \sum_{k=0}^M{ \sum_{l=1}^{M+1}{ A_{kl} \ln{\frac{a_{kl}^0}{a_{kl}}} } } = \sum_{k=0}^M{\Big(\sum_{m=1}^M{A_{km}}\Big)} \sum_{l=1}^{M+1}{a_{kl}^0 \ln{\frac{a_{kl}^0}{a_{kl}}}}$. Since the previous sum has the form of a relative entropy and is always positive, $\sum_{l=1}^{M+1}{a_{kl}^0 \ln{\frac{a_{kl}^0}{a_{kl}}}} \geq 0$, the candidate maximises the $A$ term.
- Same for $e_k^0(b) = \frac{E_k(b)}{\sum_{b'}{E_k(b')}}$.

The Baum-Welch algorithm:

- Expectation step (forward and backward algorithm):
  - Compute the expected number of times that a transition gets used:
    $$
    \begin{align*}
      A_{kl} & = \sum_\pi{P(\pi \mid x, \theta_i) A_{kl}(\pi) } \\
      & = \sum_j{ \sum_i{ P(\pi_i = k, \pi_{i+1} = l \mid x^j, \theta) } } \\
      & = \sum_j{ \frac{1}{P(x^j \mid \theta)} } \sum_i{ f_k^j(i) a_{kl} e_l(x_{i+1}^j) b_l^j(i+1) }
    \end{align*}
    $$
  - Compute the expected number of times that an emission is observed:
    $$
    \begin{align*}
      E_k(b) & = \sum_\pi{P(\pi \mid x, \theta_i) E_k(b, \pi) } \\
      & = \sum_j{ \sum_{\{ i \mid x_i^j = b \}}{ P(\pi_i = k \mid x^j, \theta) } } \\
      & = \sum_j{ \frac{1}{P(x^j \mid \theta)} } \sum_{\{ i \mid x_i^j = b \}}{ f_k^j(i) b_k^j(i) }
    \end{align*}
    $$
- Maximisation step:
  - Update the parameters with the normalised counts:
    $$a_{kl}^{i+1} = \frac{A_{kl}}{\sum_m{A_{kl}}}, \qquad e_k^{i+1}(b) = \frac{E_k(b)}{\sum_{b'}{E_k(b')}}$$

### EM for motif finding

Complex integration of multiple cis-regulatory signals controls gene activity.

Iterative motif discovery:

- Initialisation
  - Sequences
  - Random motif matrix
- Iteration
  - Sequence scoring
  - Alignment update
  - Motif instances
  - Motif matrix
- Termination
  - Convergence of the alignment and of the motif matrix

#### Multiple EM for Motif Elicitation (MEME)

If the data set consists of $N$ independent records, we can introduce independent unobserved data

## Gibbs sampling for Motif Finding and Biclustering

### Markov Chain Monte Carlo Methods

Consider a Markov Chain with transition matrix T, after two steps and after $n$ steps:

$$
\begin{align*}
  T_{ij}^{(2)} & = P(X_{t+2} = j \mid X_t = i) \\
  & = \sum_{k=1}^S{P(X_{t+2} = j \mid X_{t+1} = k, X_t = i) P(X_{t+1} = k \mid X_t = i)} \\
  & = \sum_{k=1}^S{P(X_{t+2} = j \mid X_{t+1} = k) P(X_{t+1} = k \mid X_t = i)} \\
  & = \sum_{k=1}^S{T_{ik} T_{kj}}
\end{align*}
$$

$$T^{(2)} = T \cdot T, \dots, T^{(n)} = P(X_{t+n} \mid X_t) = T^n$$

A distribution $\pi$ is stationary, i.e. $\pi T = \pi$ if:

- if the samples are generated to the distribution $\pi$, the samples at the next step will also be generated according to $\pi$,
- $\pi$ is a left eigenvector of $T$ (/ a right eigenvector of $T'$).

Equilibrium distribution: rows of $T^\infty = \lim_{n \rarr \infty}{T^n} = \lim_{n \rarr \infty}{T^{n-1}} = T^\infty T$ are stationary.

From an arbitrary initial condition and after a sufficient number of steps (burn-in), the successive states of the Markov chains are (correlated) samples from a stationary distribution.

A sufficient condition for the Markov chain to converge to the stationary distribution p is that they satisfy the condition of **detailed balance**:

$$p_i T_{ij} = p_j T_{ji}, \forall i, j$$

Proof: $(pT)_i = \sum_j{p_j T_{ji}} = p_i \sum_j{T_{ij}} = p_i, \forall i$

Problem: disjoint regions in probability space

### Gibbs Sampling

$$P(A, B, C) = P(A \mid B, C) P(B \mid A, C) P(C \mid A, B)$$

- Initialisation: $(a_0, b_0, c_0)$
- Burn-in sampling until convergence:
  - $a_{i+1} \larr P(A \mid B = b_i, C = c_i) \larr (a_i, b_i, c_i)$
  - $b_{i+1} \larr P(B \mid A = a_{i+1}, C = c_i) \larr (a_{i+1}, b_i, c_i)$
  - $c_{i+1} \larr P(C \mid A = a_{i+1}, B = b_{i+1}) \larr (a_{i+1}, b_{i+1}, c_i)$

**Data augmentation**: Introducing unobserved variables often simplifies the expression of the likelihood. A Gibbs sampler can then be set up like so (where $M$ is missing data):

$$P(\theta, M \mid D) = P(\theta \mid M, D) \otimes P(M \mid \theta, D) = \otimes_i{P(\theta_i \mid \theta_i, M, D)} \otimes_j{P(M_j \mid M_j, \theta, D)}$$

Samples from the Gibbs sampler can be used to estimate parameters: $\theta^\text{PME} = \mathbb{E}[\theta \mid D] = \int_\theta{ \int_M{ P(\theta, M \mid D) \theta \, dM } \, d\theta } \approx \frac{1}{N} \sum_{k=1}^N{\theta^k}$

- Pros:
  - Clear probabilistic interpretation
  - Bayesian framework
  - “Global optimization”
- Cons:
  - Mathematical details not easy to work out
  - Relatively slow

### Gibbs Sampling for Motif Finding

1. Set up a Gibbs sampler for the joint probability of the motif matrix and the alignment given the sequences: $P(\theta, A \mid S) = P(\theta \mid A, S) \otimes P(A \mid \theta, S)$, where $\theta$ is the motif matrix, $A$ the alignment, $S$ the sequences.
2. Sequence by sequence: $P(A \mid \theta, S) = \otimes_{i=1}^K{P(a_i \mid a_i, \theta_i, S_i)}$

- Initialization
  - Sequences
  - Random motif matrix
- Iteration
  - Sequence scoring
  - Alignment update
  - Motif instances
  - Motif matrix
- Termination
  - Convergence of the alignment and of the motif matrix / Stabilization of the motif matrix (not of the alignment) (TODO: ???)

Motif Sampler (extended Gibbs sampling):

- One motif of fixed length per round
- Several occurrences per sequence
  - Sequence have a discrete probability distribution over the number of copies of the motif (under a maximum bound)
- Multiple motifs found in successive rounds by masking occurrences of previous motifs
- Improved background model based on oligonucleotides
- Gapped motifs

## Analysis of one and two-dimensional linear systems

### Autonomous systems

### Continuous vs. discrete systems

### Equilibrium points + characterisation

### Stability

## Nonlinear system analysis

### Equilibrium points

### Stability analysis

### Phase plane and phase portraits

### Linearisation

### Bifurcations

### Chaos

## Feedback

## Synchronisation

## References

Durbinm, R., & Eddy, S. (1998). Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids (1st ed.).
