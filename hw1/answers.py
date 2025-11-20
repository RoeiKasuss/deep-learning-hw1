r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. False. Not every split of the data produces a useful train–test split. A random split can create a situation where the training set does not include certain classes or scenarios that appear in the test set, or the sets may be too small to train or evaluate the model reliably.  
   A good split must be representative, of adequate size, and sometimes stratified according to important classes or features.

2. False. The test set should not be used during cross-validation. Using it during CV can cause overfitting to the test set, leading to an unreliable estimate of the model’s generalization performance. The test set should only be used at the very end to evaluate the final model.

3. True. In cross-validation, the training set is split into $k$ folds. In each round, one fold serves as a validation fold that is not used for training. Its performance reflects the model’s ability on unseen data, and averaging the results across all folds provides a reliable estimate of the generalization error.

4. True. Injecting noise into the labels creates situations where some of the data is inaccurate. A robust model should be able to learn the general patterns without relying on every single example. If the model overfits to the clean data, its performance will drop when noise is added.  
Therefore, injecting noise is an effective way to test the model’s robustness — its ability to handle inaccurate or unexpected data.
"""

part1_q2 = r"""
**Your answer:**

Not justified.

**Problem:**  
The test set is meant for the final evaluation of the model on unseen data. If it is used to select hyperparameters $(\lambda)$, this effectively leaks information from the test set into the training process.  
This data leakage causes the model to overfit to the test set, making the estimate of generalization performance unreliable.

**Correct approach:** Split the data into three sets:

- **Training set** — for training the model  
- **Validation set** — for selecting hyperparameters ($\lambda$)  
- **Test set** — for final evaluation only, after the model and hyperparameters are chosen

This ensures that the test set remains unseen during training and hyperparameter selection, providing a reliable estimate of performance on new data.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

The SVM loss function uses the hinge loss to require that the score of the correct class be higher than the scores of the other classes by at least $\Delta$.  
For each example, the loss includes terms of the form:

$
        \max(0,\; s_j - s_{y_i} + \Delta),
$

where if an incorrect class score is too close to the correct score, or exceeds it, the expression becomes positive and the model is penalized.  
If the correct score exceeds the others by at least $\Delta$, the loss is zero.

When we allow $\Delta$, this requirement is reversed.

The expression can become negative even when the model is wrong - that is, when $\(s_j > s_{y_i}\)$.  
In such cases, $\(\max(0,\cdot)\)$ returns 0, meaning that the mistake produces no penalty.  
For a wide range of score combinations, the hinge loss becomes zero, and even correct predictions may incur a penalty that does not reflect meaningful classification quality.

When the data-dependent part of the loss is mostly zero, the only remaining term that influences optimization is the regularization term $\lambda \|W\|^2$

which decreases as the weights become smaller and is minimized at $\(W = 0\)$.  
Therefore, if the hinge loss does not penalize mistakes, the model “prefers” to converge to the solution $\(W = 0\)$, meaning all weights collapse to zero.

In other words, once $\Delta < 0$, the SVM learning mechanism collapses: the hinge loss no longer functions as an error penalty, and the model converges to a meaningless solution.
"""

part2_q2 = r"""
**Your answer:**

The linear model learns average patterns for each class.  
For each digit $k$, it produces a weight vector $w_k$ that represents its characteristic shape.  
For an input image $x$, it computes scores $s_k = w_k^\top x$ and predicts the class with the highest score.

Each weight image acts like a template or prototype.  
For example, the weights for **0** emphasize a closed ring, while the weights for **1** emphasize a vertical line in the center.

**Misclassifications occur when handwriting deviates from the linear template. Examples:**  

- **Digit 9 → classified as 4:**  
  If the 9 is open at the bottom or the vertical line is short, it resembles a 4.

- **Digit 7 → classified as 9:**  
  If the 7 has a large hook at the end of it, it resembles a 9.
"""

part2_q3 = r"""
**Your answer:**

The chosen learning rate appears to be good.

The training set Loss function graph shows a smooth, consistent, and fast decrease early in the training process, continuing to converge steadily until the end of the epochs.

This trajectory lacks the extreme oscillations or immediate plateaus that would indicate a learning rate that is too high or too low, respectively.

Had the learning rate been too high, we would have seen large jumps and fluctuations in the Loss (due to overshooting the minimum), while a too low rate would have resulted in an extremely slow loss decrease, likely remaining significantly high even after 30 epochs.

The model shows **slight overfitting**. In the accuracy graph, the training accuracy remains consistently a few percent higher than the validation accuracy.  
The gap is stable and small, indicating mild overfitting. In strong overfitting, validation accuracy would drop while training accuracy continued to rise — which does not happen here.
"""

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The best residual plot should look like random points scattered around the horizontal line at 0, with no clear pattern and with roughly the same spread everywhere. This means the linear model is doing a good job.

In the top-5-features model, the residuals look much worse. They spread out a lot, their variance changes across the range, and the model often underestimates the true values.

In contrast, the final model (after cross-validation) is much better. Its residuals are tighter, more evenly spread, and show no clear pattern. 
It also generalizes well because the train and test residuals look similar. This shows that using a more complete feature selection process reduced bias and produced a more accurate model.

"""

part3_q2 = r"""
**Your answer:**

1. Yes — it is still a linear regression model.  
A regression model is linear if it is linear in the weights.  
Even if we define a new feature $\phi(x)$, the prediction remains

    $
    y = w_0 + w_1 \phi_1(x) + \dots + w_d \phi_d(x),
    $
    
    which is linear in the parameters.


2. No — we cannot fit any nonlinear function.  
The model can express only functions that are linear combinations of the chosen nonlinear features.  
By adding many polynomial terms (high-degree), one can approximate almost any continuous function — but not necessarily fit it exactly.


3. Adding nonlinear features changes the classifier’s decision boundary:

- In the extended feature space, the decision boundary is still a **hyperplane**.  
- When projected back to the original space, the boundary becomes **nonlinear**.  
  For example, adding $x^2$ yields a parabolic boundary.

    This allows a linear classifier to solve problems that are not linearly separable in the original space.
"""

part3_q3 = r"""
**Your answer:**

**Compute $\mathbb{E}_{x,y}[|y-x|]$ for $x,y \sim U(0,1)$**

We have:

$$
\mathbb{E}[|y-x|] = \int_0^1 \int_0^1 |y-x| \, dy \, dx
$$

Split into two cases $y \geq x$ and $y < x$:

$$
\mathbb{E}[|y-x|] = \int_0^1 \int_0^x (x-y) \, dy \, dx + \int_0^1 \int_x^1 (y-x) \, dy \, dx
$$

**First integral ($y < x$):**

$$
\int_0^1 \int_0^x (x-y) dy dx = \int_0^1 \left[xy - \frac{y^2}{2}\right]_0^x dx = \int_0^1 \frac{x^2}{2} dx = \frac{1}{6}
$$

**Second integral ($y \geq x$):**

$$
\int_0^1 \int_x^1 (y-x) dy dx = \int_0^1 \left[\frac{y^2}{2} - xy\right]_x^1 dx = \int_0^1 \left(\frac{1}{2} - x + \frac{x^2}{2}\right) dx = \frac{1}{6}
$$

**Sum:**

$$
\mathbb{E}[| y - x |] = \frac{1}{6} + \frac{1}{6} = \frac{1}{3}
$$

**Compute $\mathbb{E}_x[|\hat{x}-x|]$ for fixed $\hat{x}$**

Assume $\hat{x}$ is fixed. Then:

$$
\mathbb{E}_x[|\hat{x}-x|] = \int_0^1 |\hat{x}-x| dx
$$

Split at $x = \hat{x}$:

$$
\int_0^1 |\hat{x}-x| dx = \int_0^{\hat{x}} (\hat{x}-x) dx + \int_{\hat{x}}^1 (x-\hat{x}) dx
$$

**First integral:**

$$
\int_0^{\hat{x}} (\hat{x}-x) dx = \left[\hat{x}x - \frac{x^2}{2}\right]_0^{\hat{x}} = \frac{\hat{x}^2}{2}
$$

**Second integral:**

$$
\int_{\hat{x}}^1 (x-\hat{x}) dx = \left[\frac{x^2}{2} - \hat{x}x\right]_{\hat{x}}^1 = \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}
$$

**Sum:**

$$
\mathbb{E}_x[|\hat{x}-x|] = \hat{x}^2 - \hat{x} + \frac{1}{2}
$$

**Optimization Analysis**

The constant term $\frac{1}{2}$ does not depend on $\hat{x}$.

When optimizing $L_{ED}$ to find the best $\hat{x}$, the constant does not affect the location of the minimum.
Therefore, for the purpose of optimization, we can ignore it and focus only on the polynomial part.
"""

# ==============
