# Learning

## Inductive logic programming

Deduction = general to specific (truth-preserving), induction = specific to general (falsity-preserving)

Predicate vs. function: output of predicate is always true or false, output of function can be anything

Clause and logical expression are the same but often more convenient to express clause in set notation.

Horn clause: at most one positive literal (expressed to the left of the implication operator $\leftarrow$ in the logical notation)

Inductive logic programming: need to allow for a certain level of incorrectness, else would not find anything

## 1. Neural Networks

Questions to consider for different architectures: How they work, why, and how can they be improved.

- Basic concepts: different architectures, learning rules, supervised and unsupervised learning. Shallow versus deep architectures. Applications in character recognition, image processing, diagnostics, associative memories, time-series prediction, modelling and control.
- Single- and multilayer feedforward networks and backpropagation, on-line learning, perceptron learning
- Training, validation and test set, generalization, overfitting, early stopping, regularization, double descent phenomenon
- Fast learning algorithms and optimization: Newton method, Gauss-Newton, Levenberg-Marquardt, conjugate gradient, adam
- Bayesian learning
  - Introduction to Bayesian thinking for machine learning. Learning by solving a regularized problem. Illustrative example.
- Associative memories, Hopfield networks, recurrent neural networks
- Unsupervised learning: principal component analysis, Oja's rule, nonlinear pca analysis, vector quantization, self-organizing maps
- Neural networks for time-series prediction, system identification and control; basics of LSTM; basics of deep reinforcement learning
- Basic principles of support vector machines and kernel methods, and its connection to neural networks
- Deep learning: stacked autoencoders, convolutional neural networks, residual networks
- Deep generative models: restricted Boltzmann machines, deep Boltzmann machines, generative adversarial networks, variational autoencoders, normalizing flow, diffusion models
- Normalization, attention, transformers

### 1.1. Optimisation

#### 1.1.1. Hill Climbing

#### 1.1.2. Simulated Annealing

**Gradient descent** is an algorithm for minimizing loss when training neural networks. As was mentioned earlier, a neural network is capable of inferring knowledge about the structure of the network itself from the data. Whereas, so far, we defined the different weights, neural networks allow us to compute these weights based on the training data. To do this, we use the gradient descent algorithm, which works the following way:

- Start with a random choice of weights. This is our naive starting place, where we don’t know how much we should weight each input.
- Repeat:

  - Calculate the gradient based on all data points that will lead to decreasing loss. Ultimately, the gradient is a vector (a sequence of numbers).
  - Update weights according to the gradient.

The problem with this kind of algorithm is that it requires to calculate the gradient based on all data points, which is computationally costly. There are a multiple ways to minimize this cost. For example, in Stochastic Gradient Descent, the gradient is calculated based on one point chosen at random. This kind of gradient can be quite inaccurate, leading to the Mini-Batch Gradient Descent algorithm, which computes the gradient based on on a few points selected at random, thus finding a compromise between computation cost and accuracy. As often is the case, none of these solutions is perfect, and different solutions might be employed in different situations.

**Backpropagation** is the main algorithm used for training neural networks with hidden layers. It does so by starting with the errors in the output units, calculating the gradient descent for the weights of the previous layer, and repeating the process until the input layer is reached. In pseudocode, we can describe the algorithm as follows:

- Calculate error for output layer
- For each layer, starting with output layer and moving inwards towards earliest hidden layer:
  - Propagate error back one layer. In other words, the current layer that’s being considered sends the errors to the preceding layer.
  - Update weights.

This can be extended to any number of hidden layers, creating deep neural networks, which are neural networks that have more than one hidden layer.

## 2. Support Vector Machines

- Introduction and motivation; Maximal Margin Separator
- Basics of statistical decision theory and pattern recognition
- Basics of convex optimisation theory, Karush-Kuhn-Tucker conditions, primal and dual problems
- Maximal margin classifier, linear SVM classifiers, separable and non-separable case
- Kernel trick and Mercer theorem, nonlinear SVM classifiers, choice of the kernel function, special kernels suitable for textmining
  - Definition of a kernel, how it relates to a feature space, The reproducing kernel Hilbert space.
  - Learning in functional spaces: Reproducing kernel Hilbert spaces. The representer theorem. Example 1: Kernel ridge regression. Example 2: The Perceptron and the kernel Perceptron.
  - Kernel functions in R^d: Polynomial and Gaussian kernels. General properties of kernel functions.
- Applications: classification of microarray data in bioinformatics, classification problems in biomedicine; kernel PCA, kernel ridge regression
- VC theory and structural risk minimisation, generalisation error versus empirical risk, estimating the VC dimension of SVM classifiers, optimal tuning of SVMs
- SVMs for nonlinear function estimation
- Least squares support vector machines, issues of sparseness and robustness, Bayesian framework, probabilistic interpretations, automatic relevance determination and input selection, links with Gaussian processes and regularisation networks, function estimation in RKHS.
  - Introduction to Bayesian thinking for machine learning. Learning by solving a regularized problem. Illustrative example.
  - Distance between means in RKHS, integral probability metrics, the maximum mean discrepancy (MMD), two-sample tests.
  - Choice of kernels for distinguishing distributions, characteristic kernels.
  - Covariance operator in RKHS: proof of existence, definition of norms (including HSIC, the Hilbert-Schmidt independence criterion).
  - Application of HSIC to independence testing.
- Applications: time-series prediction, finance
- Kernel versions of classical pattern recognition algorithms, kernel Fisher discriminant analysis
- Kernel trick in unsupervised learning: kernel based clustering, SVM and kernel based density estimation, kernel principal component analysis, kernel canonical correlation analysis
- Applications: datamining, bioinformatics
- Methods for large scale data sets, approximation to the feature map (Nystrom method, Random Fourier features), estimation in the primal
- SVM extensions to recurrent models and control; Kernel spectral clustering; Deep learning and kernel machines; attention and transformers from a kernel machines perspective.

## References

Blockeel, H. (2024). Machine Learning and Inductive Inference. Acco.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press. Retrieved from <https://www.deeplearningbook.org/>

Suykens, J. A. K., Van Gestel, T., De Brabanter, J., De Moor, B., & Vandewalle, J. (2002). Least-squares support vector machines. World Scientific Publishing.

Malan, D., & Yu, B. (2024). CS50’s Introduction to Artificial Intelligence with Python [Course materials]. Harvard OpenCourseWare. Retrieved from <https://cs50.harvard.edu/ai/2024/notes/1/>, <https://cs50.harvard.edu/ai/2024/notes/3/>, <https://cs50.harvard.edu/ai/2024/notes/4/>, <https://cs50.harvard.edu/ai/2024/notes/5/>

Onken, A. (2024). Machine Learning and Pattern Recognition [Course materials]. The University of Edinburgh. Retrieved from <https://mlpr.inf.ed.ac.uk/2024/>
