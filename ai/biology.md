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
    - [Viterbi decoding](#viterbi-decoding)
    - [Forward-backward algorithm](#forward-backward-algorithm)
    - [Parameter estimation with known paths](#parameter-estimation-with-known-paths)
    - [Viterbi learning](#viterbi-learning)
    - [Baum-Welch algorithm](#baum-welch-algorithm)
  - [Applications of HMMs](#applications-of-hmms)
    - [Modeling protein families](#modeling-protein-families)
    - [Gene prediction](#gene-prediction)
  - [Expectation-Maximization for clustering and motif finding](#expectation-maximization-for-clustering-and-motif-finding)
    - [The EM algorithm](#the-em-algorithm)
    - [EM for clustering](#em-for-clustering)
    - [EM for motif finding](#em-for-motif-finding)
  - [Gibbs sampling for Motif Finding and Biclustering](#gibbs-sampling-for-motif-finding-and-biclustering)
    - [Markov Chain Monte Carlo Methods](#markov-chain-monte-carlo-methods)
    - [Gibbs Sampling](#gibbs-sampling)
    - [Motif Finding](#motif-finding)
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

The frequentist view of probability is somewhat circular because of the dependence on the CLT.

The Bayesian approach views probabilities as models of the uncertainty regarding propositions within a given domain:

- Deduction: $\text{if} \, ( A \Rarr B \cap A = \text{true} ) \, \text{then} \, B = \text{true}$.
- Induction / abduction: $\text{if} \, ( A \Rarr B \cap B = \text{true} ) \, \text{then} \, A \, \text{becomes more plausible}$.

Probabilities satisfy Bayes rule.

A proposition $A$ may be true or false. A domain $\mathcal{D}$ contains the available information about a situation. Then $\angle(A = \text{true} \mid \mathcal{D})$ is a belief regarding proposition $A$ given the domain knowledge $\mathcal{D}$.

The Cox-Jaynes axioms allow the buildup of a large probabilistic framework with the following minimal assumptions (consistent with both Bayesian and frequentist views):

### Maximum likelihood, maximum a posteriori, and Bayesian inference

### Multinomial and Dirichlet distributions

### Estimation of frequency matrices

#### Pseudocounts

#### Dirichlet mixtures

## Hidden Markov Models (HMMs)

### Viterbi decoding

### Forward-backward algorithm

### Parameter estimation with known paths

### Viterbi learning

### Baum-Welch algorithm

## Applications of HMMs

### Modeling protein families

### Gene prediction

## Expectation-Maximization for clustering and motif finding

### The EM algorithm

### EM for clustering

### EM for motif finding

## Gibbs sampling for Motif Finding and Biclustering

### Markov Chain Monte Carlo Methods

### Gibbs Sampling

### Motif Finding

- Initialization
  - Sequences
  - Random motif matrix
- Iteration
  - Sequence scoring
  - Alignment update
  - Motif instances
  - Motif matrix
- Termination
  - Convergence of the alignment and of the motif matrix

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
