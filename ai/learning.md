# Learning

1. Version spaces
2. Induction of decision trees
3. Learning sets of rules
4. Instance-based learning
5. Clustering
6. Evaluating hypotheses
7. Computational Learning Theory
8. Probabilistic approaches
9. Ensembles
10. Reinforcement learning
11. Inductive logic programming
12. Neural Networks
13. Support Vector Machines

## 1. Neural Networks

- Basic concepts: different architectures, learning rules, supervised and unsupervised learning. Shallow versus deep architectures. Applications in character recognition, image processing, diagnostics, associative memories, time-series prediction, modelling and control.
- Single- and multilayer feedforward networks and backpropagation, on-line learning, perceptron learning
- Training, validation and test set, generalization, overfitting, early stopping, regularization, double descent phenomenon
- Fast learning algorithms and optimization: Newton method, Gauss-Newton, Levenberg-Marquardt, conjugate gradient, adam
- Bayesian learning
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

## 2. Support Vector Machines

- Introduction and motivation; Maximal Margin Separator
- Basics of statistical decision theory and pattern recognition
- Basics of convex optimisation theory, Karush-Kuhn-Tucker conditions, primal and dual problems
- Maximal margin classifier, linear SVM classifiers, separable and non-separable case
- Kernel trick and Mercer theorem, nonlinear SVM classifiers, choice of the kernel function, special kernels suitable for textmining
  - Definition of a kernel, how it relates to a feature space, The reproducing kernel Hilbert space.
- Applications: classification of microarray data in bioinformatics, classification problems in biomedicine; kernel PCA, kernel ridge regression
- VC theory and structural risk minimisation, generalisation error versus empirical risk, estimating the VC dimension of SVM classifiers, optimal tuning of SVMs
- SVMs for nonlinear function estimation
- Least squares support vector machines, issues of sparseness and robustness, Bayesian framework, probabilistic interpretations, automatic relevance determination and input selection, links with Gaussian processes and regularisation networks, function estimation in RKHS.
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
