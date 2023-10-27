# Kernel Methods
____
## Feature maps
Recall that in our discussion about linear regression, we considered the problem of predicting the price of a house (denoted by $y$) from the living area of the house (denoted by $x$), and we fit a linear function of x to the training data. What if the price $y$ can be more accurately represented as a non-linear function of $x$? In this case, we need a more expressive family of models than linear models.
We start by considering fitting cubic functions $y = \theta_3x^3+\theta_2x^2+\theta_1x+\theta_0$. It turns out that we can view the cubic function as a linear function over the different set of feature variables (defined below). Correctly, let the function $\phi : \reals \to \reals^4$ be defined as
```math
\phi(x) = \begin{bmatrix} 1 \\ x \\ x^2 \\ x^3 \end{bmatrix} \in \reals^4
```
Let $\theta \in \reals^4$ be the vector containing $\theta_0, \theta_1, \theta_2, \theta_3$ as entries. Then we can rewrite the cubic functin in $x$ as:
```math
\theta_3x^3+\theta_2x^2+\theta_1x+\theta_0 = \theta^T\phi(x)
```
Thus, a cubic function of the variable $x$ can be viewed as a linear function over the variables $\phi(x)$. To distinguish between these two sets of variables, in the context of kernel methods, we will call the "original" input value the input __attributes__ of a problem (in this case, $x$, the living area). When the original input is mapped to some new set of quantities $\phi(x)$, we will call those new quantities the __features__ variables. We will call $\phi$ a __feature map__, which maps the attributes to the features.
## LMS (least mean squares) with the kernel trick
The gradient descent update, or stochastic gradient update
```math
\theta := \theta + \alpha(y^{(i)}-\theta^T\phi(x^{(i)}))\phi(x^{(i)})
```
computationally expensive when the features $\phi(x)$ is high-dimensional. For example, consider
```math
\phi(x) = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ ... \\ x_1^2 \\ x_1x_2 \\ x_1x_3 \\ ... \\ x_2x_1 \\ ... \\ x_1^3 \\ x_1^2x_2 \\ ... \end{bmatrix}
```
The dimension of the features $\phi(x)$ is on the order of $d^3$. This is a prohibitively long vector for computational purpose - when $d = 1000$, each update requires at least computing and storing a $1000^3 = 10^9$ dimensional vector, which is $10^6$ times slower than the update rule for ordinary least squares update.
We will introduce the kernel trick with which we will not need to store $\theta$ explicitly, and the runtime can be significantly improved.
```math
\theta^T\phi(x) = \sum_{i=1}^n \beta_i\phi(x^{(i)})^T\phi(x) = \sum_{i=1}^n \beta_i K(x^{(i)}, x) \\
\beta := \beta + \alpha(\vec{y} - K\beta)
```
Where K is $n\times n$ matrix with $K_{ij} = K(x^{(i)}, x{(j)})$.
Fundamentally all we need to know about the feature map $\phi(⋅)$ is encapsulated in the corresponding kernel function $K(⋅,⋅)$
__Application of kernel methods:__ Kernels can be directly applied to support vector machines. In fact, the idea of kernels has significantly broader applicability than linear regression and SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products $\langle x, z\rangle$ between input attribute vectors, then by replacing this with $K(x,z)$ where $K$ is a kernel, you can “magically” allow your algorithm to work efficiently in the high dimensional feature space corresponding to K.