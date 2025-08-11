```
/*
	cross-entropy-loss.md
	Written by, Sohail Qayum Malik.
 */
```

### Cross-Entropy in Word2Vec Models (CBOW and Skip-gram)

```C++
/*
    This article explais the role of cross entropy in Word2Vec models (CBOW and Skip-gram).

    1. Key Concepts Covered:
       - Dot product between hidden layer (h) and output weights (W2) produces logits (u).
       - Softmax converts logits into predicted probabilities (y_pred).
    2. Cross-Entropy Loss:
       - Defined as L = -∑(y_true * log(y_pred)).
       - Measures divergence between true (one-hot) and predicted distributions.
    3. Backpropagation:
       - Gradient of loss w.r.t. logits (grad_u) simplifies to y_pred - y_true due to the softmax-cross-entropy combination.
       - Perfect predictions yield zero gradient/loss.
    4. Implementation Insight:
       - Shows how the theoretical gradient (∂L/∂u = y_pred - y_true) translates directly into code (Numcy::subtract()).
 */    
```

From the perspective of the propagation.md article, cross-entropy plays a crucial role in both the CBOW and Skip-gram models during training or during forward and backward propagation processes.

```C++
/*
    Forward Propagation:    
    Both algorithms then perform a dot product between the hidden layer representation (h) and the output weight matrix (W2). 
    This step is essential to transform the hidden layer activations into the vocabulary space for prediction (logits).
 */
Collective<E> u = Numcy::dot(h, W2); 

/*
    Forward Propagation:
    The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
    The softmax function converts the raw scores into probabilities.
 */
Collective<E> y_pred = softmax<E>(u);
```

#### __Cross-Entropy as the Loss Function__


During backpropagation, we compute:

```C++
/*
    Back Propagation:
    For softmax activation (y_pred) with cross-entropy loss (L =  -∑(y_true * log(y_pred)), the gradient (grad_u, ∂L/∂u) with respect to the logits (u) is indeed exactly y_pred - y_true (where y_true is your oneHot vector).
 */    
Collective<T> grad_u = Numcy::subtract<double>(fp.positive_predicted_probabilities, oneHot);

```
- Each element in grad_u shows how much the predicted probability differs from the true label
    - For correct classes (where oneHot=1): grad_u = y_pred - 1
    - For incorrect classes: grad_u = y_pred - 0 = y_pred
- Smaller absolute values in grad_u indicate better predictions

**When predictions are perfect (y_pred matches oneHot exactly), both grad_u and cross-entropy loss become zero**.

This gradient calculation actually comes directly from the cross-entropy loss function when combined with softmax activation. Here's why:


1. **Cross-Entropy** Definition: For a single training example, cross-entropy loss is:    
    - Cross-entropy (of loss) measures the difference between the true probability distribution (y_true, typically one-hot encoded labels) and the predicted distribution (y_pred, from softmax).

```BASH
L = -∑(y_true * log(y_pred)) # Measures divergence between true (one-hot) and predicted distributions
```

Where **y_true** is the one-hot encoded true label and **y_pred** is the predicted probability distribution.

2. **Derivative of Cross-Entropy** with **Softmax**: When using **softmax activation**, the gradient of the loss with respect to the logits (**u**) simplifies to:

```BASH
∂L/∂u = y_pred - y_true
```

This is exactly what we see in the code with **Numcy::subtract()**...
```BASH
# Hypothetical outputs for a 3-word vocabulary:
y_true = [0, 1, 0]  # True class = word 2
y_pred = [0.1, 0.7, 0.2]  # Model predictions

grad_u = y_pred - y_true = [0.1, -0.3, 0.2]  # Largest error for word 2 (true class)
```