# Understanding a Simple Neural Network: Step by Step
#### Written by, Sohail Qayum Malik

Neural networks might sound complicated, but they're really just a series of mathematical operations that transform input data into useful outputs. In this article, we’ll walk through a basic **neural network** with one hidden layer and we'll break down how information flows through a simple neural network in what's called a "forward pass". This kind of network is used in **machine learning (ML)** to make predictions from input data.

### What Is a Neural Network?

A neural network is like a series of filters that transform data step by step. Data goes in one end, passes through one or more "hidden layers," and comes out the other end as predictions or classifications.

### Our Example Neural Network Structure

- **Batch size**: 4 (we process 4 samples at once)
- **Input dimension**: 16 (each sample has 16 features)
- **Hidden dimension**: 32 (our middle layer has 32 neurons)
- **Output dimension**: 8 (our final output has 8 values per sample, or our output layer has 8 neurons)

---

### The Forward Pass Made Simple

The forward pass is how data flows through the network from input to output. It involves two main steps:


## Step 1: Input to Hidden Layer

We start with our input data:

- 4 samples (batch size)
- Each sample has 16 features (input dimension)
- So our input is shaped like a table with 4 rows and 16 columns: [4][16]

To transform this input, we need weights:
- We have 32 neurons in our hidden layer
- Each neuron connects to all 16 input features
- So we have weights shaped as [32][16]

The transformation works like this:
1. We transpose (flip) the weights to get shape [16][32]. To multiply the input with the weights, we need to transpose `W1` (flip its rows and columns).
2. We multiply: Input [4][16] × Weights [16][32] = Hidden [4][32]
3. We add biases (like adjustment factors) to each result

After this step, we have 4 samples, each with 32 values representing the hidden layer activations.

### Step 2: Hidden to Output Layer

Now we take our hidden layer values and transform them again:
- Our hidden layer output is shaped [4][32]
- Our output layer has 8 neurons, each connected to all 32 hidden neurons
- So we have weights shaped as [8][32]

The transformation is similar:
1. Transpose weights to get shape [32][8]
2. Multiply: Hidden [4][32] × Weights [32][8] = Output [4][8]
3. Add biases to each result

After this step, we have our final output: 4 samples, each with 8 values.

## Summary of Shapes

Think of each shape as [rows][columns]:

| Name | Shape | Meaning |
|------|-------|---------|
| Input | [4][16] | 4 samples, 16 features each |
| Weights 1 | [32][16] | Connections from inputs to hidden layer |
| Hidden | [4][32] | Output from hidden layer |
| Weights 2 | [8][32] | Connections from hidden to output layer |
| Output | [4][8] | Final predictions (4 samples, 8 values each) |

## The Math in Plain Language

The key formula is: Y = X · W^T + b

This means:
- Take your input (X)
- Multiply it by the transposed weights (W^T)
- Add the bias values (b)
- The result is your output (Y)

We do this twice - once to get from input to hidden layer, and again to get from hidden to output layer.

### Why Does This Matter?

This process allows neural networks to learn complex patterns. During training, these weights and biases are adjusted to make the network's predictions more accurate. The more you train, the better your network becomes at its task!

Neural networks can seem intimidating at first, but they're really just applying these simple math operations over and over again to transform data into meaningful outputs.

This is the basic idea behind many deep learning models — just with more layers and data.

![NN](https://github.com/KHAAdotPK/MachineLearning/blob/main/neural_network_diagram.svg)
![NN](https://github.com/KHAAdotPK/MachineLearning/blob/main/nn.png)