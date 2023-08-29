# Regularization and model selection
____
## Regularization
Overfitting is typically a result of using too complex models, and we need to choose a proper model complexity to achieve the optimal bias-variance tradeoff. We will use regularization to control the model complexity and prevent overfitting.

Regularization typically involves adding an additional term, called a regularizer and denoted by $R(\theta)$ here, to the training loss/cost function:
```math
J_\lambda(\theta) = J(\theta) + \lambda R(\theta)
```
Here $J_\lambda$ is often called the regularized loss, and $\lambda \ge 0$ is called the regularization parameter. The regularizer $R(\theta)$ is typically chosen to be some measure of the complexity of the model $\theta$.

When $\lambda = 0$, the regularized loss is equivalent to the original loss. When $\lambda$ is a sufficiently small positive number, minimizing the regularized loss is effectively minimizing the original loss with the regularizer as the tie-breaker.

## Model selection via cross validation
In __hold-out cross validation__ (also called __simple cross validation__), we do the following:
1. Randomly split $S$ into $S_{train}$ (say 70% of the data) and $S_{CV}$ (the remaining 30%). Here, $S_{CV}$ is called the hold-out cross validation set.
2. Train each model $M_i$ on $S_{train}$ only, to get some hypothesis $h_i$.
3. Select and output the hypothesis $h_i$ that had the smallest error on the hold out cross validation set. The error on the hold out validation set is also referred to as the validation error.

This approach is essentially picking the model with the smallest estimated generalization/test error. When the total dataset is huge, validation set can be a smaller fraction of the total examples as long as the absolute number of validation examples is decent. For example, for the ImageNet dataset that has about 1M training images, the validation set is sometimes set to be 50K images, which is only about 5% of the total examples.

The disadvantage of using hold out cross validation is that it “wastes” about 30% of the data. Even if we were to take the optional step of retraining the model on the entire training set, it’s still as if we’re trying to find a good model for a learning problem in which we had $0.7n$ training examples, rather than n training examples, since we’re testing models that were trained on only $0.7n$ examples each time. While this is fine if data is abundant and/or cheap, in learning problems in which data is scarce (consider a problem with $n = 20$, say), we’d like to do something better.

Here is a method, called __k-fold cross validation__, that holds out less data each time:

Let's say we have dataset of 20 values. We randomly pick 19 samples, train our model and use that 1 value as a test sample. Then we pick another 19 training samples and so on, eventually comparing the error.