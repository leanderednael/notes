# Biology

1. [**Introduction to molecular biology**](#introduction-to-molecular-biology)
   - [DNA](#dna)
   - [RNA](#rna)
   - [Proteins](#rna)
2. **Sequence alignment**
   - Dynamic programming
   - Global and local alignment
   - BLAST
3. **Introduction to Bayesian statistics**
   - The Cox-Jaynes axioms
   - Maximum likelihood, maximum a posteriori, and Bayesian inference
   - Dirichlet distributions and pseudocounts
4. **Hidden Markov Models (HMMs)**
   - Viterbi decoding
   - Forward-backward algorithm
   - Parameter estimation with known paths
   - Viterbi learning
   - Baum-Welch algorithm
5. **Applications of HMMs**
   - Modeling protein families
   - Gene prediction
6. **Expectation-Maximization for clustering and motif finding**
   - The EM algorithm
   - EM for clustering
   - EM for motif finding
7. [**Gibbs sampling for motif finding and biclustering**](#gibbs-sampling-for-motif-finding-and-biclustering)
   - [Markov Chain Monte Carlo methods](#markov-chain-monte-carlo-methods)
   - Gibbs sampling
   - Motif finding
8. **Analysis of one and two-dimensional linear systems**
   - autonomous systems
   - continuous vs. discrete systems
   - equilibrium points + characterisation
   - stability
9. **Nonlinear system analysis**
   - equilibrium points
   - stability analysis
   - phase plane and phase portraits
   - linearisation
   - bifurcations
   - chaos
10. **Feedback**
11. **Synchronisation**

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

Importance of RNA as a molecule in its own right increasingly recognized (noncoding RNA).

### [Proteins](https://www.edx.org/learn/biology/rice-university-proteins-biology-s-workforce)

Proteins are large polymers of 20 aminoacids.

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

1. Prokaryotic cells
   - Cellular **precursors** (18-44 MW): $CO_2, H_2O, N_2, NH_3$, salts
   - **Intermediates** (50-250 MW): $\alpha$-keto acids, ribose, pyruvate, malate, acetate, malonate
   - **Building blocks** (100-350 MW): amino acids, nucleotides, simple sugars, fatty acids / glycerol
   - **Macromolecules** ($10^3 - 10^9$ MW): proteins, nucleic acids, polysaccharides, lipids
   - Supramolecular **assemblies** ($10^6 - 10^9$ MW): enzyme complexes, ribosomes, contractile systems, membranes
2. Eukaryotic cells:
   - **Organelles**: nucleus, mitochondria, chloroplasts
3. Types of non-covalent (low energy, easily broken) bonding in biological systems:
   - Hydrogen bonds: shared hydrogen between two molecules or parts of a molecule
   - Ionic / electrostatic interactions
   - van der Waals interactions: attraction between partially positive and partially negative atoms
   - Hydrophobic interactions / forces
   - _Large number of small forces creates flexibility of structures_.
4. Proteins are polymers assembled from amino acid units:
   - Amino acids: acidic, basic, apolar, polar
   - Structure, properties, abbreviations of side chains for 6 assigned amino acids: Glu, Lys, Trp, Ser, Gly, Pro
   - Each amino acid provides a unique side chain
5. Levels of protein structure:
   - Primary structure: Amino acid sequence; peptide bonds link amino acids to form a polymer (backbone for protein and side chain arrangement).
   - Secondary structure: $\alpha$-helix, $\beta$-sheet structures (identified by Pauling)
   - Tertiary structure: folding into 3 dimensions
   - Quaternary (oligomeric) structure: assembly into higher oligomers
6. Protein folding:
   - Amino acid sequence of a protein not functional without folding
   - 3-D structure is the active form of a protein
   - Non-covalent bonds between side chains (and peptide backbone interactions) create the “folded” form of the protein (IMPORTANT)
   - Folded form exhibits a biological activity
   - Protein folding funnels - energetic pathways to function: Lowest energy state is the folded structure. Large energy penalty for loss of entropy - can think of it as loss of options for different states so that the overall difference in energy between folded/unfolded is small.
   - Stabilizing energy for protein folding: Primarily non-covalent interactions

#### Protein Function

1. Relationship of shape to function (e.g., enzymes, structural proteins, movement)
2. Role of binding in protein function — binding to small molecules, to each other, to other cellular components
3. Types of proteins: Structural, binding, movement, enzymes
4. Amphiphilic nature of membrane proteins

#### Protein Function & Purification

1. Enzymes that carry out specific reactions in complex mixtures by lowering energy barrier to reaction
2. Gibbs free energy (ΔG) characterizes the reaction and coupling of reactions
3. Feedback inhibition and allosteric regulation process for enzymes
4. Conformational changes required for function (dynamic nature)
5. Complex proteins assembly to oligomers, higher order structures
6. Regulation of protein function: Allostery; covalent modification (e.g., phosphorylation, acetylation)
7. Protein dynamics/intrinsic disorder (repeated for emphasis!), conformational flexibility (adaptation)
8. Protein interaction networks
9. Methods for purifying proteins: Sources, breaking cells, separating cellular contents, crude methods (precipitation by salts, temperature, pI), chromatographic methods

#### Protein Characterisation

1. Protein characterization: Spectroscopy, fluorescence, antibodies, assays
2. Protein characterization/quantitation: Monoclonal antibodies - ELISA, immunoprecipitation
3. Monomer and oligomer size determination: SDS gel electrophoresis (Weber & Osborn), 2D gels, Western blotting, gel chromatography
4. Protein structure determination: Explain what X-ray crystallography, NMR can do
5. Biochemical methods: Leverage the function of the protein (reaction catalyzed, inhibitors, activators) to characterize interaction networks

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

## References

Durbinm, R., & Eddy, S. (1998). Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids (1st ed.).
